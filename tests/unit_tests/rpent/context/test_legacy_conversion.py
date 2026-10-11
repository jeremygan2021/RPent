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

"""Legacy conversion reproduces sent requests and preserves causal features."""

import base64
import copy
import json

import pytest

from rpent.context.audit import sanitize
from rpent.context.codex import CodexAdapter
from rpent.context.session import ContextSession
from tests.unit_tests.rpent.context.test_images import FakeFeatures, request

pytest.importorskip("torch")
from rpent.context.training import PPOConfig  # noqa: E402
from scripts.context_benchmark import convert_legacy  # noqa: E402


def fixture(tmp_path, monkeypatch):
    out = tmp_path / "episode/output"
    (out / "requests").mkdir(parents=True)
    (out.parent / "workspace").mkdir()
    monkeypatch.setattr(
        convert_legacy, "code_signature", lambda _: convert_legacy.EXPECTED_CODE
    )
    (out.parent / "config.json").write_text(
        json.dumps(
            {
                "proxy_argv": [
                    "--mode",
                    "all",
                    "--rho-image",
                    ".2",
                    "--rho-text",
                    ".1",
                    "--rho-action",
                    ".1",
                    "--cameras",
                    "front",
                ]
            }
        )
    )
    (out.parent / "status.json").write_text('{"status":"finished"}')
    (out / "timing.json").write_text(
        '{"task_end_reason":"environment_terminated","task_end_at":1.5}'
    )
    (out / "states.json").write_text('{"steps":[{"terminated":true}]}')
    source = request()
    data = source["input"][2]["output"][1]["image_url"].split(",", 1)[1]
    (out / "image.png").write_bytes(base64.b64decode(data))
    session = ContextSession(
        CodexAdapter(("front",)),
        FakeFeatures(),
        mode="all",
        rho_image=0.2,
        rho_text=0.1,
        rho_action=0.1,
    )
    logs = []
    for i, raw in enumerate([dict(source, input=source["input"][:3]), source], 1):
        result = session.process(raw)
        rid = f"{i:04d}"
        (out / "requests" / f"{rid}.before.json").write_text(json.dumps(sanitize(raw)))
        (out / "requests" / f"{rid}.after.json").write_text(
            json.dumps(sanitize(result))
        )
        (out / "requests" / f"{rid}.timing.json").write_text(
            json.dumps({"status": 200, "started_at": i})
        )
        (out / "requests" / f"{rid}.response").write_text(
            json.dumps(
                {
                    "object": "response",
                    "id": rid,
                    "usage": {"input_tokens": 100 * i, "output_tokens": 10},
                }
            )
        )
        logs.append(dict(copy.deepcopy(session.stats), request_id=rid))
    (out / "selection.jsonl").write_text("\n".join(json.dumps(x) for x in logs))
    return out, logs


def test_conversion_matches_live_observations_and_keeps_real_actions(
    tmp_path, monkeypatch
):
    out, logs = fixture(tmp_path, monkeypatch)
    episode = convert_legacy.convert_episode(out, PPOConfig(1000))
    transitions = episode["transitions"]
    assert len(transitions) == 2
    for transition, log in zip(transitions, logs):
        assert transition["state"] == list(log["decision"]["observation"]["values"])
        # Legacy source has no Memory selector; the fourth action is inactive.
        assert transition["action"] == [0.2, 0.1, 0.1, 0.5]
    assert transitions[0]["next_state"] == transitions[1]["state"]
    assert [t["reward"] for t in transitions] == pytest.approx([-0.01, 0.98])
    assert transitions[1]["done"] is True
    assert transitions[1]["attempts"][0]["after_environment_success"] is True


@pytest.mark.parametrize("fault", ["usage", "render", "code", "verdict", "fallback"])
def test_converter_rejects_unverifiable_evidence(tmp_path, monkeypatch, fault):
    out, _ = fixture(tmp_path, monkeypatch)
    if fault == "usage":
        (out / "requests/0001.response").write_text("{}")
    elif fault == "render":
        (out / "requests/0001.after.json").write_text("{}")
    elif fault == "code":
        monkeypatch.setattr(convert_legacy, "code_signature", lambda _: "different")
    elif fault == "verdict":
        (out / "states.json").write_text('{"steps":[{"terminated":false}]}')
    else:
        with (out / "selection.jsonl").open("a") as f:
            f.write('\n{"fallback":"processing_error"}')
    with pytest.raises(ValueError):
        convert_legacy.convert_episode(out, PPOConfig(1000))
