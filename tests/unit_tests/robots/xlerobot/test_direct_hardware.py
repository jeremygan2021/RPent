# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Hardware-free task, isolation and evidence regression tests."""

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from robots.xlerobot.control import NativeOwner
from robots.xlerobot.isolation import availability, build_launch
from robots.xlerobot.service import Action, Engine, Skill, Task, create_app
from robots.xlerobot.tasks import SkillRejected


def config(tmp_path):
    token = tmp_path / "token"
    token.write_text("admin-test-token" * 3)
    return {
        "token_file": str(token),
        "native_token_file": str(token),
        "console_key_file": str(token),
        "output_dir": str(tmp_path / "runs"),
        "stop_timeout_s": 0.01,
        "planner": {"base_url": "https://example.invalid", "api_key_file": str(token)},
    }


def running(engine):
    engine.state = "running"
    engine.task = {
        "id": "robot-task-1234",
        "backend": "hardware",
        "status": "running",
        "instruction": "pick up",
        "task_outcome": "pending",
    }
    engine.save_history()


@pytest.mark.parametrize(
    "expected", [None, "different-controller", "registered-controller"]
)
def test_controller_identity_must_match_private_binding(
    monkeypatch, tmp_path, expected
):
    owner = NativeOwner(tmp_path / "unused", expected)
    monkeypatch.setattr(
        "robots.xlerobot.control.executor_mappings", lambda: ({"axis": "motor"}, {})
    )
    monkeypatch.setattr(
        owner,
        "call",
        lambda *args: {
            "controller_id": "registered-controller",
            "scope": "xlerobot_040_native_position",
            "source_hash_verified": True,
            "axes": [{"key": "axis", "name": "motor"}],
        },
    )
    if expected != "registered-controller":
        with pytest.raises(ValueError, match="identity"):
            owner.status()
    else:
        assert owner.status()["controller_id"] == expected


def test_hardware_status_does_not_contact_sim_or_model(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Hardware has no simulation or VLA dependency")

    monkeypatch.setattr(engine, "model", forbidden)
    monkeypatch.setattr(engine, "mapping_status", forbidden)
    monkeypatch.setattr(engine.native, "status", lambda: {"fresh": True, "held": True})
    result = engine.status("hardware")
    assert result["robot"]["fresh"]
    assert result["model"]["required"] is False
    assert result["capabilities"]["pi05_inference"]["ready"] is False
    assert result["capabilities"]["observe"]["ready"] is True


def test_stop_contacts_owner_even_when_skill_server_disconnected(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    running(engine)
    engine.task["active_skill"] = "skill-12345"
    calls = []
    monkeypatch.setattr(
        engine.native,
        "call",
        lambda *args: calls.append("motor") or {"stop_confirmed": True},
    )

    def unavailable(*args):
        calls.append("skill")
        raise OSError("offline")

    monkeypatch.setattr(engine.skills, "request", unavailable)
    result = engine.control_stop("hardware")
    assert calls == ["motor", "skill"]
    assert result["stop_confirmed"] is False
    assert engine.task["stop_state"] == "unknown"


def test_stop_waits_for_new_generation_feedback(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    monkeypatch.setattr(
        engine.native, "call", lambda *args: {"generation": 5, "stop_confirmed": False}
    )
    monkeypatch.setattr(
        engine.native,
        "status",
        lambda: {"fresh": True, "generation": 5, "stop_confirmed": True},
    )
    assert engine.control_stop("hardware")["stop_confirmed"] is True


@pytest.mark.parametrize("instance", ["registered-owner", "different-owner"])
def test_stop_uses_owner_generation_and_instance(monkeypatch, tmp_path, instance):
    engine = Engine(config(tmp_path))
    monkeypatch.setattr(
        engine.native,
        "call",
        lambda *args: {
            "stop_generation": 15,
            "generation": 24,
            "instance_id": "registered-owner",
            "stop_confirmed": False,
        },
    )
    monkeypatch.setattr(
        engine.native,
        "status",
        lambda: {
            "fresh": True,
            "stop_confirmed": True,
            "stop_generation": 15,
            "generation": 24,
            "instance_id": instance,
        },
    )
    assert engine.control_stop("hardware")["stop_confirmed"] == (
        instance == "registered-owner"
    )


def test_stop_revokes_later_actions_but_allows_stop_retry(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    running(engine)
    monkeypatch.setattr(engine.native, "call", lambda *args: {"stop_confirmed": True})
    engine.control_stop("hardware")
    with pytest.raises(ValueError, match="revoked"):
        engine.authorize(engine.task["id"])
    engine.authorize(engine.task["id"], allow_cancelled=True)
    assert engine.task_status(engine.task["id"])["stop_state"] == "stopped"


def test_zero_exit_does_not_establish_grasp_success(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    running(engine)
    process = SimpleNamespace(wait=lambda: 0)
    engine.process = process

    def stop(backend):
        engine.stop_confirmed = True

    monkeypatch.setattr(engine, "control_stop", stop)
    asyncio.run(engine.watch(process))
    assert engine.task_status(engine.task["id"])["task_outcome"] == "unknown"


def test_duplicate_completed_task_is_query_not_reexecution(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    running(engine)
    engine.task["status"] = "completed"
    engine.state = "completed"
    engine.save_history()

    def forbidden(*args):
        raise AssertionError("No new readiness check or spawn for existing task")

    monkeypatch.setattr(engine, "status", forbidden)
    body = Task(backend="hardware", request_id=engine.task["id"], instruction="pick up")
    assert engine.start(body)["task"]["status"] == "completed"
    with pytest.raises(ValueError, match="different parameters"):
        engine.start(body.model_copy(update={"instruction": "another"}))


def test_readiness_rejection_is_durable_and_never_reexecuted(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    monkeypatch.setattr(
        engine,
        "status",
        lambda *_: {"ready": False, "blocked_reasons": ["motion verification missing"]},
    )
    body = Task(
        backend="hardware", request_id="blocked-task-1234", instruction="pick up"
    )
    with pytest.raises(ValueError, match="query original task"):
        engine.start(body)
    result = engine.task_status(body.request_id)
    assert result["run_status"] == "rejected"
    assert result["task"]["motion_submitted"] is False
    assert result["task"]["blocked_reasons"] == ["motion verification missing"]
    monkeypatch.setattr(
        engine, "status", lambda *_: pytest.fail("must query original task")
    )
    assert engine.start(body)["run_status"] == "rejected"


def test_stop_during_isolation_setup_fences_spawn(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    monkeypatch.setattr(engine, "status", lambda *_: {"ready": True})

    def setup(*args):
        engine.stop_epoch += 1
        return [], {}

    monkeypatch.setattr("robots.xlerobot.service.build_launch", setup)
    monkeypatch.setattr(
        "robots.xlerobot.service.subprocess.Popen",
        lambda *a, **k: pytest.fail("spawn after STOP"),
    )
    with pytest.raises(ValueError, match="revoked"):
        engine.start(
            Task(backend="hardware", request_id="new-task-1234", instruction="pick up")
        )
    assert not engine.task_tokens


def test_task_token_cannot_start_tasks_or_switch_backend(monkeypatch, tmp_path):
    app = create_app(config(tmp_path))
    engine = app.state.engine
    running(engine)
    engine.task_tokens["scoped"] = engine.task["id"]
    client = TestClient(app)
    headers = {"Authorization": "Bearer scoped", "X-RPent-Session": engine.task["id"]}
    assert client.post("/v1/rpent/tasks", headers=headers, json={}).status_code == 403
    assert (
        client.post(
            "/v1/rpent/mapping", headers=headers, json={"enabled": True}
        ).status_code
        == 403
    )
    assert (
        client.get("/v1/rpent/status?backend=simulation", headers=headers).status_code
        == 403
    )
    assert (
        client.get("/v1/rpent/tasks/robot-task-1234", headers=headers).status_code
        == 200
    )
    assert (
        client.get("/v1/rpent/tasks/another-task", headers=headers).status_code == 403
    )
    engine.task = {"id": "another-task"}
    assert (
        client.get("/v1/rpent/status?backend=hardware", headers=headers).status_code
        == 403
    )


def test_task_scope_is_checked_after_slow_status(monkeypatch, tmp_path):
    app = create_app(config(tmp_path))
    engine = app.state.engine
    running(engine)
    engine.task_tokens["scoped"] = engine.task["id"]
    headers = {"Authorization": "Bearer scoped", "X-RPent-Session": engine.task["id"]}

    def changed_task(*args):
        engine.task = {"id": "new-task-4567"}
        return {"task": engine.task, "transcript": "another task's data"}

    monkeypatch.setattr(engine, "status", changed_task)
    response = TestClient(app).get("/v1/rpent/status?backend=hardware", headers=headers)
    assert response.status_code == 403
    assert "another task's data" not in response.text


def test_simulation_planner_uses_same_isolation(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    monkeypatch.setattr(engine, "status", lambda *_: {"ready": True})

    def isolate(*args):
        raise ValueError("isolation invoked")

    monkeypatch.setattr("robots.xlerobot.service.build_launch", isolate)
    with pytest.raises(ValueError, match="isolation invoked"):
        engine.start(
            Task(
                backend="simulation", request_id="sim-task-1234", instruction="observe"
            )
        )


def test_hardware_motion_rejects_aged_observation(tmp_path):
    engine = Engine(config(tmp_path))
    engine.observations["stale"] = ("hardware", time.monotonic() - 3, {})
    with pytest.raises(ValueError, match="expired"):
        engine.action(
            Action(backend="hardware", command="base", observation_id="stale"), ""
        )


def skill_setup(engine, monkeypatch):
    running(engine)
    monkeypatch.setattr(
        engine.skills,
        "status",
        lambda *_: {
            "ready": True,
            "mode": "hardware",
            "instance_id": "executor",
            "execution_epoch": 8,
        },
    )
    return Skill(
        backend="hardware",
        request_id="skill-request-1",
        task_id=engine.task["id"],
        operation="grasp_object",
        parameters={"object_id": "pitcher"},
    )


def test_skill_submission_is_single_use_and_excludes_motion(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    body = skill_setup(engine, monkeypatch)
    submissions = []

    def request(method, path, payload=None):
        if method == "POST":
            submissions.append(payload)
            return {"accepted": True}
        return {"id": path.rsplit("/", 1)[-1], "mode": "hardware", "status": "running"}

    monkeypatch.setattr(engine.skills, "request", request)
    first = engine.skill(body, engine.task["id"])
    second = engine.skill(body, engine.task["id"])
    assert first["executor_id"] == second["executor_id"]
    assert len(submissions) == 1
    assert submissions[0]["execution_epoch"] == 8
    with pytest.raises(ValueError, match="owns the controller"):
        engine.action(
            Action(backend="hardware", command="base", observation_id="ignored"),
            engine.task["id"],
        )
    with pytest.raises(ValueError, match="owns the controller"):
        engine.skill(
            body.model_copy(update={"request_id": "skill-request-2"}), engine.task["id"]
        )


def test_late_skill_ack_is_cancelled_again(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    body = skill_setup(engine, monkeypatch)
    cancelled = []

    def request(method, path, payload=None):
        if path == "/v1/tasks":
            engine.stop_epoch += 1
            return {"accepted": True}
        if path == "/v1/stop":
            cancelled.extend(payload["cancel_task_ids"])
            return {"ok": True}
        pytest.fail("Unexpected request")

    monkeypatch.setattr(engine.skills, "request", request)
    result = engine.skill(body, engine.task["id"])
    assert result["status"] == "interrupted"
    assert cancelled == [result["executor_id"]]


def test_explicit_locate_rejection_allows_new_observation(monkeypatch, tmp_path):
    engine = Engine(config(tmp_path))
    body = skill_setup(engine, monkeypatch).model_copy(
        update={"operation": "locate_object"}
    )
    calls = []

    def request(*args):
        calls.append(args)
        raise SkillRejected("Handle is occluded")

    monkeypatch.setattr(engine.skills, "request", request)
    assert engine.skill(body, engine.task["id"])["status"] == "blocked"
    assert not engine.pending_skills()
    assert engine.skill(body, engine.task["id"])["status"] == "blocked"
    assert len(calls) == 1
    result = engine.skill(
        body.model_copy(update={"request_id": "new-locate-request"}), engine.task["id"]
    )
    assert result["result"]["reason"] == "Handle is occluded"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "lift,hold,verified",
    [(0.01, 3, True), (0.05, 1, True), (0.05, 3, False), (0.05, 3, True)],
)
def test_grasp_outcome_requires_lift_hold_and_verifier(
    monkeypatch, tmp_path, lift, hold, verified
):
    engine = Engine(config(tmp_path))
    skill_setup(engine, monkeypatch)
    engine.task["grasp_executor_id"] = "grasp-executor-id"
    engine.task["task_outcome"] = "succeeded"
    engine.task["grasp_evidence"] = {"old": True}
    monkeypatch.setattr(
        engine.skills,
        "request",
        lambda *args: {
            "mode": "hardware",
            "evidence": {"grasp_verified": verified, "lift_m": lift, "hold_s": hold},
        },
    )
    body = Skill(
        backend="hardware",
        request_id="verify-request-1",
        task_id=engine.task["id"],
        operation="verify_grasp",
        parameters={"task_id": "grasp-executor-id"},
    )
    engine.skill(body, engine.task["id"])
    assert (engine.task["task_outcome"] == "succeeded") == (
        lift >= 0.05 and hold >= 3 and verified
    )


def test_real_sandbox_hides_host_files_and_robot_devices(tmp_path):
    cfg = config(tmp_path)
    if not availability(cfg)[0]:
        pytest.skip("bubblewrap/Landlock unavailable")
    repo = Path(__file__).resolve().parents[4]
    folder = tmp_path / "sandbox-task-1234"
    folder.mkdir()
    token = folder / "token"
    token.write_text("scoped")
    secret = tmp_path / "outside-secret"
    secret.write_text("test-secret")
    code = (
        "from pathlib import Path; import robots.xlerobot.client; "
        f"assert not Path({str(secret)!r}).exists(); "
        "assert not Path('/dev/ttyUSB0').exists(); "
        "Path('result').write_text('isolated')"
    )
    command, env = build_launch(
        cfg, repo, folder, [str(repo / ".venv/bin/python"), "-c", code], token
    )
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert (folder / "run/result").read_text() == "isolated"
