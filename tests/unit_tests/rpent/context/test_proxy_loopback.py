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

"""Real loopback HTTP audit: unchanged Full bytes and stable retry decisions."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.parametrize("audit", [False, True])
@pytest.mark.parametrize("learned", [False, True])
def test_proxy_records_attempt_ids_and_preserves_empty_body(tmp_path, audit, learned):
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            response = json.dumps(
                {
                    "object": "response",
                    "id": f"r{len(received)}",
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "rpent.context.proxy",
        "--port",
        str(port),
        "--upstream",
        f"http://127.0.0.1:{upstream.server_port}",
        "--mode",
        "all" if learned else "full",
        "--log",
        str(tmp_path / "selection.jsonl"),
    ]
    if learned:
        pytest.importorskip("torch")
        from rpent.context.controller import Thresholds
        from rpent.context.policy import initialize, save_checkpoint

        checkpoint = tmp_path / "policy.pt"
        save_checkpoint(initialize(Thresholds(0.2, 0.1, 0.1)), checkpoint)
        command += ["--controller-checkpoint", str(checkpoint), "--controller-sample"]
    if audit:
        command += ["--audit-dir", str(tmp_path / "requests")]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=dict(os.environ)
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            assert process.poll() is None, process.stderr.read().decode()
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
        body = b'{ "input": [], "model": "test", "stream": false }'
        for _ in range(2):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/responses", data=body
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                response.read()
        assert received == [body, body]
        rows = [
            json.loads(x)
            for x in (tmp_path / "selection.jsonl").read_text().splitlines()
        ]
        assert [r["request_id"] for r in rows] == ["0001", "0002"]
        assert [r["decision"]["id"] for r in rows] == [1, 1]
        assert rows[1]["decision"]["reused"]
        if learned:
            assert rows[0]["decision"]["policy_version"].startswith("mlp-")
            assert rows[0]["decision"]["log_prob"] is not None
            assert (
                rows[0]["decision"]["latent_action"]
                == rows[1]["decision"]["latent_action"]
            )
        if audit:
            deadline = time.monotonic() + 3
            while not (tmp_path / "requests/0002.timing.json").exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            timing = json.loads((tmp_path / "requests/0002.timing.json").read_text())
            assert timing["decision_id"] == 1
            assert timing["status"] == 200
            from scripts.context_benchmark.summarize import summarize

            (tmp_path / "transcript_test.json").write_text(
                json.dumps(
                    {
                        "environment_success": False,
                        "agent_error": None,
                        "elapsed_s": 5,
                        "finish": {"status": "success"},
                    }
                )
            )
            summary = summarize(tmp_path)
            trajectory = json.loads((tmp_path / "trajectory.json").read_text())
            assert summary["input_tokens"] == 20
            assert trajectory["audit_complete"]
            assert trajectory["outcome"]["environment_success"] is False
            assert len(trajectory["decisions"]) == 1
            assert len(trajectory["decisions"][0]["attempts"]) == 2
            if learned:
                from rpent.context.policy import MLPController
                from rpent.context.training import PPOConfig, prepare_batch

                batch = prepare_batch(
                    [trajectory], MLPController(checkpoint), PPOConfig(1000)
                )
                assert len(batch.actions) == 1
                assert batch.rewards.item() == pytest.approx(-0.002)
    finally:
        process.terminate()
        process.communicate(timeout=5)
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["full", "all"])
def test_encoder_warmup_completes_before_serving(monkeypatch, tmp_path, mode):
    from rpent.context import proxy

    events = []

    class FakeFeatures:
        def __init__(self, model):
            pass

        def warmup(self):
            events.append("warmup")

    monkeypatch.setattr(proxy, "Features", FakeFeatures)
    monkeypatch.setattr(proxy, "serve", lambda **kwargs: events.append("serve"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "proxy",
            "--mode",
            mode,
            "--warmup-encoder",
            "--log",
            str(tmp_path / "selection.jsonl"),
        ],
    )
    proxy.main()
    assert events == (["serve"] if mode == "full" else ["warmup", "serve"])



@pytest.mark.parametrize("mode", ["full", "all"])
def test_robot_profile_removes_unused_interfaces_in_both_modes(monkeypatch, mode):
    from types import SimpleNamespace
    from rpent.context.proxy import rewrite_body

    monkeypatch.setenv("RPENT_CODEX_ROBOCASA_PROFILE", "1")
    monkeypatch.setenv("RPENT_PRESERVE_TOOL_HISTORY", "1")
    request = {"input": [
        {"type": "additional_tools", "tools": [
            {"type": "namespace", "name": "functions", "tools": [
                {"name": "exec"}, {"name": "wait"},
                {"name": "request_user_input"}, {"name": "request_user_input_async"},
            ]},
            {"type": "namespace", "name": "collaboration", "tools": [{"name": "spawn_agent"}]},
            {"type": "namespace", "name": "clock", "tools": [{"name": "sleep"}]},
            {"type": "namespace", "name": "rpent", "tools": [{"name": "rldx_arm"}]},
        ]},
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "<multi_agent_role>irrelevant rules"},
            {"type": "input_text", "text": "Preserve robot instructions"},
        ]},
        {"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": "<multi_agent_mode>irrelevant rules"},
        ]},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<multi_agent_role>quoted task data"},
        ]},
    ]}
    session = SimpleNamespace(mode=mode, stats={}, process=lambda value: value)
    rewritten, changed = rewrite_body(json.dumps(request).encode(), "", session)
    result = json.loads(rewritten)
    assert changed
    assert [x["name"] for x in result["input"][0]["tools"]] == ["functions", "rpent"]
    assert [x["name"] for x in result["input"][0]["tools"][0]["tools"]] == ["exec", "wait"]
    assert result["input"][1]["content"][0]["text"] == "Preserve robot instructions"
    assert result["input"][2]["content"][0]["text"] == "<multi_agent_role>quoted task data"
    again, changed_again = rewrite_body(rewritten, "", session)
    assert again == rewritten and not changed_again
