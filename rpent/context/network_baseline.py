# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-task measurements of the Codex transport and fixed-request baseline."""

import json
import os
import time
from math import isfinite
from pathlib import Path
from statistics import median


def _headers():
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    tokens = json.loads((home / "auth.json").read_text())["tokens"]
    headers = {"Authorization": "Bearer " + tokens["access_token"]}
    if tokens.get("account_id"):
        headers["ChatGPT-Account-Id"] = tokens["account_id"]
    return headers


def _summary(rows):
    good = [r for r in rows if r.get("ok")]
    stats = {
        "samples": len(rows),
        "successes": len(good),
        "failures": len(rows) - len(good),
    }
    for key in ("latency_s", "headers_s", "first_event_s", "first_text_s"):
        values = [r[key] for r in good if r.get(key) is not None]
        if values:
            stats[key] = {
                "median": median(values),
                "maximum": max(values),
                "minimum": min(values),
                "spread": max(values) - min(values),
            }
    return stats


def run_baseline(
    upstream,
    model,
    reasoning,
    *,
    link_samples=5,
    model_samples=3,
    timeout=30,
    client=None,
):
    """Record a baseline; failures are data, not grounds to alter task metrics.

    The small GET includes endpoint processing. The fixed streamed POST includes
    model/service/cache and transport effects. Neither is pure network latency.
    """
    import httpx

    started = time.perf_counter()
    result = {
        "schema_version": 1,
        "started_at": time.time(),
        "upstream": upstream,
        "model": model,
        "reasoning": reasoning,
        "outside_task": True,
        "link": [],
        "fixed_request": [],
    }
    owned = client is None
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    try:
        try:
            headers = _headers()
        except (OSError, KeyError, ValueError) as error:
            result["setup_error"] = type(error).__name__
            return result
        base = upstream.rstrip("/")
        for _ in range(link_samples):
            begin = time.perf_counter()
            row = {}
            try:
                response = client.get(
                    base + "/backend-api/codex/models",
                    params={"client_version": "0.114.0"},
                    headers=headers,
                )
                row.update(status=response.status_code, ok=response.is_success)
            except httpx.HTTPError as error:
                row.update(ok=False, error=type(error).__name__)
            row["latency_s"] = time.perf_counter() - begin
            result["link"].append(row)
        payload = {
            "model": model,
            "instructions": "Reply briefly.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with exactly OK."}
                    ],
                }
            ],
            "reasoning": {"effort": reasoning},
            "store": False,
            "stream": True,
        }
        for _ in range(model_samples):
            begin = time.perf_counter()
            row = {"ok": False, "completed": False}
            try:
                with client.stream(
                    "POST",
                    base + "/backend-api/codex/responses",
                    headers=headers,
                    json=payload,
                ) as response:
                    row.update(
                        status=response.status_code,
                        headers_s=time.perf_counter() - begin,
                    )
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if time.perf_counter() - begin > timeout * 2:
                            raise httpx.ReadTimeout("baseline total-time limit")
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            continue
                        event = json.loads(raw)
                        row.setdefault("first_event_s", time.perf_counter() - begin)
                        if event.get("type") == "response.output_text.delta":
                            row.setdefault("first_text_s", time.perf_counter() - begin)
                        if event.get("type") == "response.completed":
                            row.update(
                                completed=True,
                                usage=event.get("response", {}).get("usage"),
                            )
                        elif event.get("type") in (
                            "error",
                            "response.failed",
                            "response.incomplete",
                        ):
                            row["stream_error"] = event["type"]
                    row["ok"] = row["completed"] and not row.get("stream_error")
            except (httpx.HTTPError, ValueError) as error:
                row["error"] = type(error).__name__
            row["latency_s"] = time.perf_counter() - begin
            result["fixed_request"].append(row)
        return result
    finally:
        result["duration_s"] = time.perf_counter() - started
        result["link_summary"] = _summary(result["link"])
        result["fixed_request_summary"] = _summary(result["fixed_request"])
        if owned:
            client.close()


def latency_gate(baseline: dict, config: dict) -> dict:
    """Admit collection only near a fixed transport/service reference.

    Evaluate raw samples rather than trusting cached summary fields. This gates
    collection; it does not subtract network measurements from task rewards.
    """
    tolerance = config["median_relative_tolerance"]
    peak_tolerance = config["peak_relative_tolerance"]
    if not (0 < tolerance < 1 and 0 <= peak_tolerance < 1):
        raise ValueError("invalid latency gate tolerances")
    reasons, checks = [], {}
    for key in ("model", "reasoning", "upstream"):
        if baseline.get(key) != config[key]:
            reasons.append(f"{key} differs from reference")
    if baseline.get("outside_task") is not True or baseline.get("setup_error"):
        reasons.append("baseline setup missing or failed")
    for kind, count in (("link", 5), ("fixed_request", 3)):
        reference = config["reference"][kind]
        if any(
            type(reference[k]) not in (int, float)
            or not isfinite(reference[k])
            or reference[k] <= 0
            for k in ("median_s", "maximum_s")
        ):
            raise ValueError("invalid latency reference")
        rows = baseline.get(kind, [])
        if not isinstance(rows, list) or len(rows) != count:
            reasons.append(f"{kind}: requires {count} samples")
            continue
        valid = True
        for row in rows:
            elapsed, status = row.get("latency_s"), row.get("status")
            if (
                row.get("ok") is not True
                or type(status) is not int
                or not 200 <= status < 300
                or type(elapsed) not in (int, float)
                or not isfinite(elapsed)
                or elapsed <= 0
            ):
                valid = False
            if kind == "fixed_request" and (
                row.get("completed") is not True
                or row.get("stream_error")
                or not row.get("usage")
            ):
                valid = False
        if not valid:
            reasons.append(f"{kind}: failed or incomplete sample")
            continue
        times = [row["latency_s"] for row in rows]
        observed, peak = median(times), max(times)
        lower = (
            0
            if config.get("allow_faster", False)
            else reference["median_s"] * (1 - tolerance)
        )
        upper = reference["median_s"] * (1 + tolerance)
        peak_upper = reference["maximum_s"] * (1 + peak_tolerance)
        checks[kind] = {
            "median_s": observed,
            "maximum_s": peak,
            "median_lower_s": lower,
            "median_upper_s": upper,
            "maximum_upper_s": peak_upper,
        }
        if not lower <= observed <= upper:
            reasons.append(f"{kind}: median outside reference range")
        if peak > peak_upper:
            reasons.append(f"{kind}: slowest sample exceeds reference limit")
    return {
        "version": config["version"],
        "accepted": not reasons,
        "reasons": reasons,
        "checks": checks,
    }


def run_gated_baseline(upstream, model, reasoning, config, *, on_check=None):
    """Try up to three full baselines and stop at the first accepted check.

    Retain failed measurements outside task costs. Reference limits remain fixed.
    """
    checks = []
    for attempt in range(1, 4):
        data = run_baseline(upstream, model, reasoning, link_samples=5, model_samples=3)
        data["latency_gate"] = latency_gate(data, config)
        data["gate_check"] = attempt
        checks.append(data)
        if on_check is not None:
            on_check(data)
        if data["latency_gate"]["accepted"]:
            break
    return {**data, "gate_checks": checks}
