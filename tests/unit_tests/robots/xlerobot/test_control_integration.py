# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""Offline checks for borrowed execution, stale observations and cancellation."""

import time

import pytest
from fastapi.testclient import TestClient

from robots.xlerobot import get_robot_spec
from robots.xlerobot.control import NativeOwner, native_sim_target
from robots.xlerobot.service import Action, Engine, Task, create_app


def config(tmp_path):
    token = tmp_path / "token"
    token.write_text("x" * 40)
    return {
        "token_file": str(token),
        "output_dir": str(tmp_path / "runs"),
        "native_token_file": str(token),
        "console_key_file": str(token),
    }


def test_registration_and_prompt_contract():
    from rpent.robots import enumerate_robots

    assert "xlerobot" in enumerate_robots()
    spec = get_robot_spec()
    assert "Use tools only" in spec.prompts.render("system")
    assert "read cameras" in spec.prompts.render(
        "user", variables={"instruction": "read cameras"}
    )


def test_restart_retains_evidence_without_resuming_task(tmp_path):
    engine = Engine(config(tmp_path))
    engine.task = {
        "id": "historical-task",
        "backend": "simulation",
        "status": "running",
    }
    engine.state = "running"
    engine.last_inference = {"shape": [50, 32], "finite": True}
    engine.save_history()
    restored = Engine(config(tmp_path))
    assert restored.state == "error"
    assert restored.task["status"] == "interrupted"
    assert restored.last_inference["finite"] is True
    assert restored.process is None
    assert restored.stop_confirmed is False


def test_encoder_conversion_matches_existing_renderer():
    row = {
        "key": "arm_1.shoulder_lift",
        "reference_tick": 2048,
        "min_tick": 1900,
        "max_tick": 2200,
        "direction": -1,
        "direction_confirmed_by_operator": True,
    }
    name, q = native_sim_target({"axes": [row]}, row["key"], 2048)
    assert name == "Right_Pitch"
    assert q == pytest.approx(1.5707963267948966)
    with pytest.raises(ValueError, match="range"):
        native_sim_target({"axes": [row]}, row["key"], 2201)
    row["direction_confirmed_by_operator"] = False
    with pytest.raises(ValueError, match="confirmed"):
        native_sim_target({"axes": [row]}, row["key"], 2048)


def test_consumed_observation_never_replays(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "robots.xlerobot.service.sim_call",
        lambda *a, **kw: calls.append((a, kw)) or {"ok": True},
    )
    engine = Engine(config(tmp_path))
    engine.observations["latest"] = ("simulation", time.monotonic(), {})
    action = Action(command="base", observation_id="latest", duration=0.5, v=0.03)
    assert engine.action(action, "")["ok"] is True
    with pytest.raises(ValueError, match="consumed"):
        engine.action(action, "")
    assert len(calls) == 1


def test_task_cannot_send_hardware_action(tmp_path):
    engine = Engine(config(tmp_path))
    engine.state = "running"
    engine.task = {"id": "task-12345", "backend": "simulation"}
    with pytest.raises(ValueError, match="switch"):
        engine.action(
            Action(
                backend="hardware",
                command="native",
                observation_id="latest",
                joint="arm_1.wrist_roll",
                target_tick=2048,
            ),
            "task-12345",
        )


def test_stop_fences_start_during_readiness_check(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))

    def status(backend):
        engine.stop_epoch += 1
        return {"ready": True}

    monkeypatch.setattr(engine, "status", status)
    with pytest.raises(ValueError, match="revoked"):
        engine.start(Task(instruction="inspect", request_id="cancelled-start"))
    assert engine.process is None


def test_authentication_and_malformed_motion(tmp_path):
    app = create_app(config(tmp_path))
    client = TestClient(app)
    assert client.get("/v1/rpent/status").status_code == 401
    auth = {"Authorization": "Bearer " + "x" * 40}
    assert (
        client.post(
            "/v1/rpent/action",
            headers=auth,
            json={"command": "base", "observation_id": "id", "v": 1.0, "duration": 0.5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/rpent/tasks",
            headers=auth,
            json={"instruction": "x", "request_id": "../invalid"},
        ).status_code
        == 422
    )


def test_hardware_base_borrows_units_lease_and_stops(monkeypatch, tmp_path):
    from copy import deepcopy

    owner = NativeOwner(tmp_path / "unused")
    snapshot = {
        "fresh": True,
        "held": True,
        "instance_id": "native",
        "generation": 4,
        "axes": [{"torque_enabled": 1}],
        "base_control": {
            "fresh": True,
            "armed": True,
            "fault": None,
            "max_x": 0.15,
            "max_theta": 30,
            "moving": False,
            "wheels": [{"torque": 1, "velocity": 0}] * 2,
        },
    }
    monkeypatch.setattr(owner, "status", lambda: deepcopy(snapshot))
    calls = []

    def call(path, body):
        calls.append((path, body))
        return {"lease": "existing-lease", "ok": True}

    monkeypatch.setattr(owner, "call", call)
    from types import SimpleNamespace

    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(
        "robots.xlerobot.control.time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep),
    )
    result = owner.drive(0.03, 0.1, 0.1, snapshot, lambda: False)
    assert [c[0] for c in calls] == ["base-begin", "base-frame", "base-stop"]
    assert calls[1][1]["action"]["theta.vel"] == pytest.approx(5.7295779513)
    assert result["stop_confirmed"] is True
    snapshot["base_control"]["armed"] = False
    with pytest.raises(ValueError, match="整机"):
        owner.drive(0.03, 0.1, 0.1, snapshot, lambda: False)
    assert len(calls) == 3


def test_base_stop_preempts_late_frame(monkeypatch, tmp_path):
    owner = NativeOwner(tmp_path / "unused")
    snapshot = {
        "fresh": True,
        "held": True,
        "instance_id": "native",
        "generation": 4,
        "axes": [{"torque_enabled": 1}],
        "base_control": {
            "fresh": True,
            "armed": True,
            "max_x": 0.15,
            "max_theta": 30,
            "wheels": [{"torque": 1}] * 2,
        },
    }
    monkeypatch.setattr(owner, "status", lambda: snapshot)
    calls = []
    cancelled = [False]

    def call(path, body):
        calls.append(path)
        cancelled[0] = True
        return {"lease": "lease"}

    monkeypatch.setattr(owner, "call", call)
    with pytest.raises(ValueError, match="revoked"):
        owner.drive(0.03, 0.0, 0.1, snapshot, lambda: cancelled[0])
    assert calls == ["base-begin", "base-stop"]
