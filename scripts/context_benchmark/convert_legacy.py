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

"""Convert audited fixed-scale LIBERO logs without changing their outcomes."""

import argparse
import ast
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from rpent.context.audit import sanitize
from rpent.context.codex import CodexAdapter
from rpent.context.controller import Thresholds
from rpent.context.observation import FEATURE_VERSION, SCORING_VERSION, observe
from rpent.context.replay import restore
from rpent.context.training import PPOConfig, rewards_and_returns
from rpent.context.trajectory import read_response
from rpent.evaluation.result import write_json_atomic

EXPECTED_CODE = "98c0f839d7df88222d8012e8602d4d5ce8481f46bea6aad03ad0c1f48b9d87af"
MODULES = (
    "session.py",
    "text.py",
    "action.py",
    "codex.py",
    "features.py",
    "selection.py",
)


def code_signature(root: Path) -> str:
    """Match executable source structure, ignoring whitespace and comments."""
    source = "".join(
        ast.dump(ast.parse((root / "rpent/context" / f).read_text())) for f in MODULES
    )
    return hashlib.sha256(source.encode()).hexdigest()


def convert_episode(output: Path, reward_config: PPOConfig) -> dict:
    """Reconstruct causal observations using actual previous selections.

    This legacy adapter accepts only the audited fixed-scale implementation and
    complete successful LIBERO episodes. Unknown/failure outcomes require a
    separate verified outcome adapter, not inference from agent prose.
    """
    output = Path(output).resolve()
    config_path = output.parent / "config.json"
    config = json.loads(config_path.read_text())
    argv = config["proxy_argv"]

    def arg(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    if "--memory-blocks" in argv:
        raise ValueError(
            "legacy Memory decisions cannot be reconstructed by this adapter"
        )
    if (
        arg("--mode") != "all"
        or code_signature(output.parent / "workspace") != EXPECTED_CODE
    ):
        raise ValueError(f"{output}: incompatible selection implementation/mode")
    if (
        json.loads((output.parent / "status.json").read_text()).get("status")
        != "finished"
    ):
        raise ValueError(f"{output}: episode not finished")
    clock = json.loads((output / "timing.json").read_text())
    states = json.loads((output / "states.json").read_text())["steps"]
    if clock.get("task_end_reason") != "environment_terminated" or not any(
        s.get("terminated") is True for s in states
    ):
        raise ValueError(f"{output}: verified LIBERO success required by this adapter")
    success_at = clock["task_end_at"]
    rho = float(arg("--rho", 0.5))
    thresholds = Thresholds(
        *(float(arg("--rho-" + m, rho)) for m in ("image", "text", "action"))
    )
    if any(v == 1 for v in asdict(thresholds).values()):
        raise ValueError("offline Gaussian actor requires thresholds strictly below 1")
    selections = {}
    for line in (output / "selection.jsonl").read_text().splitlines():
        row = json.loads(line)
        rid = row.get("request_id")
        if row.get("fallback") or not rid or rid in selections:
            raise ValueError("fallback, missing or duplicate selection ID")
        selections[rid] = row
    images = {}
    for path in output.rglob("*.png"):
        if path.is_file():
            data = path.read_bytes()
            images[hashlib.sha256(data).hexdigest()] = data
    adapter = CodexAdapter(tuple(filter(None, arg("--cameras", "").split(","))))
    stage = arg("--stage", "Unknown")
    history = {m: (set(), set()) for m in ("image", "text", "action")}
    previous, transitions, seen_responses, seen_requests = {}, [], set(), set()
    last_signature = None
    for path in sorted((output / "requests").glob("*.before.json")):
        rid = path.name.split(".")[0]
        captured = json.loads(path.read_text())
        captured = captured.get("body", captured)
        signature = hashlib.sha256(
            json.dumps(captured, sort_keys=True).encode()
        ).hexdigest()
        request = restore(captured, images)
        if request.get("previous_response_id"):
            raise ValueError("server-held context cannot be reconstructed")
        stats = selections[rid]
        prefix = output / "requests" / rid
        timing = json.loads(Path(str(prefix) + ".timing.json").read_text())
        usage, _, response_id = read_response(
            Path(str(prefix) + ".response"), timing.get("content_encoding", "")
        )
        if timing.get("transport_error") or timing.get("status") != 200:
            raise ValueError(f"{output}/{rid}: transport failure")
        if not response_id or response_id in seen_responses:
            raise ValueError("missing or duplicate response ID")
        seen_responses.add(response_id)
        seen_requests.add(rid)
        cost = (usage or {}).get("input_tokens")
        if type(cost) is not int or cost < 0:
            raise ValueError("missing or invalid input token count")
        records = {
            "image": adapter.extract(request),
            "text": adapter.extract_text(request)[0],
            "action": adapter.extract_action(request),
        }
        # Re-render the ACTUAL choices and compare against the sent request.
        # Never replay a different action and reuse this episode's outcome.
        result = adapter.render(
            request,
            records["image"],
            {x["id"] for x in stats["records"] if x["selected"]},
        )
        text_records, _ = adapter.extract_text(result)
        result = adapter.render_text(
            result,
            text_records,
            {x["id"] for x in stats["text"]["records"] if x["selected"]},
        )
        action_records = adapter.extract_action(result)
        result = adapter.render_action(
            result,
            action_records,
            {x["id"] for x in stats["action"]["records"] if x["selected"]},
        )
        after = json.loads(Path(str(prefix) + ".after.json").read_text())
        if sanitize(result) != after.get("body", after):
            raise ValueError(f"{output}/{rid}: actual rendered request mismatch")
        if signature != last_signature:
            obs = observe(
                records,
                history,
                set(history),
                previous,
                thresholds,
                len(transitions),
                stage != "Unknown",
            )
            if not all(math.isfinite(v) for v in obs.values):
                raise ValueError("non-finite reconstructed features")
            transitions.append(
                {
                    "state": list(obs.values),
                    "action": list(asdict(thresholds).values()),
                    "input_tokens": 0,
                    "attempts": [],
                }
            )
            for modality in history:
                rows = (
                    stats["records"]
                    if modality == "image"
                    else stats[modality]["records"]
                )
                seen, _ = history[modality]
                seen.update(x["id"] for x in rows)
                history[modality] = (seen, {x["id"] for x in rows if x["selected"]})
            previous, last_signature = stats, signature
        transitions[-1]["input_tokens"] += cost
        transitions[-1]["attempts"].append(
            {
                "request_id": rid,
                "response_id": response_id,
                "input_tokens": cost,
                "after_environment_success": timing["started_at"] > success_at,
            }
        )
    if seen_requests != set(selections) or not transitions:
        raise ValueError("missing request evidence")
    rewards, _ = rewards_and_returns(
        [t["input_tokens"] for t in transitions], True, reward_config
    )
    for i, transition in enumerate(transitions):
        terminal = i == len(transitions) - 1
        transition.update(
            reward=rewards[i],
            done=terminal,
            next_state=[0.0] * len(transition["state"])
            if terminal
            else transitions[i + 1]["state"],
        )
    return {
        "episode_id": str(output),
        "environment_success": True,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selection_sha256": hashlib.sha256(
            (output / "selection.jsonl").read_bytes()
        ).hexdigest(),
        "source_code_signature": EXPECTED_CODE,
        "transitions": transitions,
    }


def main() -> None:
    """Convert explicit episodes; abort rather than silently drop invalid runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-scale", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh dataset path")
    try:
        config = PPOConfig(args.token_scale, args.beta)
        if len({p.resolve() for p in args.episodes}) != len(args.episodes):
            raise ValueError("duplicate episodes")
        episodes = [convert_episode(p, config) for p in args.episodes]
        from rpent.context.policy import feature_names

        dataset = {
            "schema_version": 1,
            "feature_version": FEATURE_VERSION,
            "scoring_version": SCORING_VERSION,
            "feature_names": list(feature_names()),
            "reward": {
                "beta": config.beta,
                "token_scale": config.token_scale,
                "gamma": 1.0,
                "boundary": "complete_agent_session",
            },
            "episodes": episodes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output, dataset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "transitions": sum(len(e["transitions"]) for e in episodes),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
