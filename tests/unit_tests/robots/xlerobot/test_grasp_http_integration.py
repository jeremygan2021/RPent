# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Real HTTP across RPent and the sibling executor, with no hardware sockets.

The sibling checkout supplies its maintained fake owner/perception fixture. These
tests deliberately replace planning and physical response: they establish wire
contracts and lifecycle fencing, never physical grasp feasibility.
"""

import asyncio
import importlib.util
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from PIL import Image

from robots.xlerobot import service
from robots.xlerobot.control import CONTROL_SOURCE
from robots.xlerobot.toolkit import XLeRobotToolkit
from rpent.dashboard.events import NullDashboardEventSink


@contextmanager
def http_server(app):
    """Own one ephemeral loopback listener and shut it down before returning."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    url = "http://127.0.0.1:" + str(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="on")
    )
    worker = threading.Thread(
        target=lambda: server.run(sockets=[listener]), daemon=True
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "Offline HTTP fixture failed to start"
        yield url
    finally:
        server.should_exit = True
        worker.join(5)
        listener.close()
        assert not worker.is_alive(), "Offline HTTP fixture failed to stop"


@pytest.fixture
def wire(tmp_path, monkeypatch, pytestconfig):
    source = CONTROL_SOURCE / "tests/test_grasp_runtime.py"
    if not source.exists():
        pytest.skip("Sibling canonical grasp_runtime checkout is unavailable")
    monkeypatch.syspath_prepend(str(CONTROL_SOURCE))
    pytestconfig.addinivalue_line("markers", "asyncio: sibling executor async tests")
    spec = importlib.util.spec_from_file_location("offline_grasp_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.runtime.__wrapped__(tmp_path)
    from grasp_runtime.server import create_app

    executor_token = tmp_path / "executor-token"
    executor_token.write_text("offline-executor-secret-" + "x" * 32)
    executor_token.chmod(0o600)
    app = create_app(runtime=runtime, token=executor_token.read_text())
    calls = []
    delayed, release = threading.Event(), threading.Event()
    gate = {"enabled": False}

    @app.middleware("http")
    async def record_requests(request, call_next):
        raw = await request.body()
        body = json.loads(raw) if raw else None
        calls.append((request.method, request.url.path, body))
        if (
            gate["enabled"]
            and request.method == "POST"
            and request.url.path == "/v1/tasks"
        ):
            delayed.set()
            assert await asyncio.to_thread(release.wait, 5), (
                "Test submit gate timed out"
            )
        return await call_next(request)

    image_path = tmp_path / "camera.png"
    Image.new("RGB", (24, 24), (50, 100, 150)).save(image_path)
    frame_count = 0

    @app.post("/api/observe")
    async def camera_observation():
        nonlocal frame_count
        frame_count += 1
        return {
            "id": f"camera-{frame_count}",
            "images": {
                name: {"path": str(image_path), "unix": time.time()}
                for name in ("head", "left_wrist", "right_wrist")
            },
        }

    real_client = httpx.Client
    allowed = set()

    def guard_request(request):
        # The current observe adapter has a fixed URL. Rewrite it before the
        # transport gets the request; the live 8916 service is never contacted.
        if str(request.url) == "http://127.0.0.1:8916/api/observe":
            request.url = httpx.URL(executor_url + "/api/observe")
        assert (request.url.host, request.url.port) in allowed, (
            "Unexpected network request in offline integration: " + str(request.url)
        )

    def local_client(**kwargs):
        kwargs["event_hooks"] = {"request": [guard_request]}
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", local_client)
    monkeypatch.setattr(
        service,
        "sim_call",
        lambda *args, **kwargs: pytest.fail("Unexpected simulator call"),
    )
    with http_server(app) as executor_url:
        allowed.add(("127.0.0.1", httpx.URL(executor_url).port))
        host_token = tmp_path / "host-token"
        host_token.write_text("offline-rpent-host-secret")
        cfg = {
            "output_dir": str(tmp_path / "runs"),
            "token_file": str(host_token),
            "native_token_file": str(host_token),
            "console_key_file": str(host_token),
            "skills": {"url": executor_url, "token_file": str(executor_token)},
        }
        rpent_app = service.create_app(cfg)
        engine = rpent_app.state.engine
        native_calls = []
        engine.native = SimpleNamespace(
            status=lambda: {"fresh": True, "held": True, "stop_confirmed": True},
            call=lambda *args: native_calls.append(args) or {"stop_confirmed": True},
        )
        engine.task = {
            "id": "offline-http-task",
            "backend": "hardware",
            "status": "running",
            "stage": "observing",
            "task_outcome": "pending",
            "stop_state": "unknown",
        }
        engine.state, engine.last_backend = "running", "hardware"
        task_token = tmp_path / "task-token"
        task_token.write_text("offline-scoped-task-token")
        engine.task_tokens[task_token.read_text()] = engine.task["id"]
        engine.save_history()
        monkeypatch.setenv("RPENT_CONTROL_TOKEN_FILE", str(task_token))
        monkeypatch.setenv("RPENT_CONTROL_SESSION", engine.task["id"])
        with http_server(rpent_app) as rpent_url:
            allowed.add(("127.0.0.1", httpx.URL(rpent_url).port))
            toolkit = XLeRobotToolkit(
                backend="hardware",
                output_dir=tmp_path / "toolkit",
                dashboard_events=NullDashboardEventSink(),
                memory=None,
                endpoint=rpent_url,
            )
            # Only fake owner/perception are connected to this executor.
            assert engine.skills.request("POST", "/v1/stop", {})["stop_confirmed"]
            assert engine.skills.request("POST", "/v1/resume", {})["ok"]
            try:
                yield SimpleNamespace(
                    runtime=runtime,
                    fake_physics=lambda: module.fake_physics(runtime),
                    engine=engine,
                    toolkit=toolkit,
                    calls=calls,
                    native_calls=native_calls,
                    gate=gate,
                    delayed=delayed,
                    release=release,
                )
            finally:
                release.set()


def test_observe_locate_grasp_verify_real_http_contract(wire):
    wire.fake_physics()
    observed = wire.toolkit.execute_tool("observe_robot", {})
    assert "error" not in observed.result, observed.result
    assert observed.result["task_id"] == "offline-http-task"
    assert len([b for b in observed.content_blocks if b["type"] == "image"]) == 3

    locate = wire.toolkit.locate_object("pitcher")
    assert locate["status"] == "succeeded", locate
    target = locate["result"]
    assert target["motion_ready"] is True
    grasp = wire.toolkit.execute_tool(
        "grasp_object",
        {
            "object_id": "pitcher",
            "observation_id": target["observation_id"],
            "calibration_version": target["calibration_version"],
        },
    )
    assert "error" not in grasp.result, grasp.result
    assert len([b for b in grasp.content_blocks if b["type"] == "image"]) == 3
    executor_id = grasp.result["executor_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = wire.toolkit.task_status()
        item = next(
            item
            for item in record["task"]["skill_requests"].values()
            if item["request"]["operation"] == "grasp_object"
        )
        if item["status"] != "running":
            break
        time.sleep(0.01)
    assert item["status"] == "succeeded", item
    assert record["task_outcome"] == "pending"
    verification = wire.toolkit.verify_grasp(executor_id)
    assert verification["status"] == "succeeded", verification
    evidence = verification["result"]["evidence"]
    assert evidence["grasp_verified"] is True
    assert evidence["lift_m"] >= 0.05 and evidence["hold_s"] >= 3
    assert wire.toolkit.task_status()["task_outcome"] == "succeeded"
    assert (
        wire.toolkit.finish("Offline fixture verified")["task_success_verified"] is True
    )
    assert wire.toolkit.task_status()["stop_state"] == "stopped"
    assert "error" not in wire.toolkit.observe()
    assert wire.toolkit.stop()["stop_confirmed"] is True
    with pytest.raises(ValueError, match="session has been revoked"):
        wire.toolkit.grasp_object(
            "pitcher", target["observation_id"], target["calibration_version"]
        )
    events = wire.toolkit.client.call("tasks/offline-http-task/events")["events"]
    assert any(event["type"] == "skill_result" for event in events)
    assert any(
        event["type"] == "stop_result" and event["confirmed"] for event in events
    )

    submitted = [body for method, path, body in wire.calls if path == "/v1/tasks"]
    assert len(submitted) == 1
    assert submitted[0]["id"] == executor_id
    assert submitted[0]["mode"] == "hardware"
    assert submitted[0]["instance_id"] == wire.runtime.instance_id
    assert submitted[0]["parameters"] == {
        "object_id": "pitcher",
        "side": "right",
        "observation_id": target["observation_id"],
        "calibration_version": target["calibration_version"],
    }
    assert ("POST", "/v1/verify", {"task_id": executor_id}) in wire.calls
    assert wire.native_calls == [("stop", {"operator_ready": True})] * 2


def test_stop_fences_http_submission_that_arrives_after_resume(wire):
    target = wire.toolkit.locate_object("pitcher")["result"]
    wire.gate["enabled"] = True
    with ThreadPoolExecutor(max_workers=1) as workers:
        late = workers.submit(
            wire.toolkit.grasp_object,
            "pitcher",
            target["observation_id"],
            target["calibration_version"],
        )
        try:
            assert wire.delayed.wait(3), "Submission did not reach HTTP gate"
            assert wire.toolkit.stop()["stop_confirmed"] is True
            assert wire.engine.skills.request("POST", "/v1/resume", {})["ok"]
        finally:
            wire.release.set()
        result = late.result(timeout=5)
        assert result["status"] == "interrupted"
        assert "stale executor instance/epoch" in result["result"]["reason"]
    body = next(body for _, path, body in wire.calls if path == "/v1/tasks")
    assert body["id"] in wire.runtime.cancelled_ids
    assert body["execution_epoch"] < wire.runtime.generation
    assert wire.runtime.tasks == {}
    assert not any(
        path.endswith(("/begin", "/frame")) for _, path, _ in wire.runtime.owner.calls
    )
    assert wire.toolkit.task_status()["task_outcome"] == "unknown"
    assert wire.toolkit.task_status()["stop_state"] == "stopped"


def test_forged_grasp_coordinates_rejected_over_http(wire):
    target = wire.toolkit.locate_object("pitcher")["result"]
    result = wire.toolkit.skill(
        "grasp_object",
        {
            "object_id": "pitcher",
            "side": "right",
            "observation_id": target["observation_id"],
            "calibration_version": target["calibration_version"],
            "point_m": [0.5, 0, 0.9],
        },
    )
    assert result["status"] == "blocked"
    assert "Untrusted coordinates" in result["result"]["reason"]
    assert wire.runtime.tasks == {}
    assert not any(
        path.endswith(("/begin", "/frame")) for _, path, _ in wire.runtime.owner.calls
    )


@pytest.mark.parametrize("confirmed", [True, False])
def test_running_skill_stop_reports_measured_confirmation(wire, confirmed):
    wire.fake_physics()
    target = wire.toolkit.locate_object("pitcher")["result"]
    owner_call = wire.runtime.owner.call
    first_frame, release = threading.Event(), threading.Event()

    async def delayed_frame(method, path, body=None):
        result = await owner_call(method, path, body)
        if path.endswith("/frame"):
            first_frame.set()
            while not release.is_set():
                await asyncio.sleep(0.002)
        return result

    wire.runtime.owner.call = delayed_frame
    wire.runtime.owner.confirm_stop = confirmed
    try:
        reply = wire.toolkit.grasp_object(
            "pitcher", target["observation_id"], target["calibration_version"]
        )
        assert reply["status"] == "running", reply
        assert first_frame.wait(3), "Fake executor never reached its first frame"
        stopped = wire.toolkit.stop()
        assert stopped["stop_confirmed"] is confirmed
        assert wire.runtime.tasks[reply["executor_id"]]["status"] == "cancelled"
        assert wire.runtime.latched is True
        record = wire.toolkit.task_status()
        assert record["stop_state"] == ("stopped" if confirmed else "unknown")
        assert record["task_outcome"] != "succeeded"
        assert (
            wire.toolkit.finish("Stopped offline test")["task_success_verified"]
            is False
        )
        if not confirmed:
            with pytest.raises(ValueError, match="rejected request: 409"):
                wire.engine.skills.request("POST", "/v1/resume", {})
        else:
            assert wire.engine.skills.request("POST", "/v1/resume", {})["ok"]
            body = next(body for _, path, body in wire.calls if path == "/v1/tasks")
            count = len(wire.runtime.owner.calls)
            old = wire.engine.skills.request("POST", "/v1/tasks", body)
            assert old["status"] == "cancelled"
            assert len(wire.runtime.owner.calls) == count
    finally:
        release.set()
