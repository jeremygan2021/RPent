# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline adversarial lifecycle checks for the direct hardware task boundary."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from robots.xlerobot import isolation, service


def config(tmp_path):
    token = tmp_path / "host-token"
    token.write_text("offline-host-token")
    return {
        "token_file": str(token),
        "native_token_file": str(token),
        "console_key_file": str(token),
        "output_dir": str(tmp_path / "runs"),
        "stop_timeout_s": 0,
    }


def test_skill_stop_failure_still_attempts_native_stop(tmp_path):
    engine = service.Engine(config(tmp_path))
    engine.task = {
        "id": "review-task",
        "backend": "hardware",
        "active_skill": "skill-123",
    }
    native_calls = []

    def unavailable_skill(*args):
        raise OSError("offline skill executor unavailable")

    engine.skills = SimpleNamespace(request=unavailable_skill)
    engine.native = SimpleNamespace(
        call=lambda *args: native_calls.append(args) or {"stop_confirmed": True},
    )
    try:
        engine.control_stop("hardware")
    except (ValueError, OSError, RuntimeError):
        pass
    assert native_calls == [("stop", {"operator_ready": True})]
    assert engine.stop_confirmed is False


def test_finished_task_credential_cannot_read_another_tasks_status(
    tmp_path, monkeypatch
):
    app = service.create_app(config(tmp_path))
    engine = app.state.engine
    engine.task_tokens["old-scoped-token"] = "previous-task"
    engine.journal.write(
        {"id": "previous-task", "backend": "hardware", "status": "completed"}
    )
    engine.task = {"id": "current-task", "backend": "hardware", "status": "running"}
    engine.state = "running"
    monkeypatch.setattr(
        engine,
        "status",
        lambda backend: {
            "task": engine.task,
            "transcript": "current task private transcript",
        },
    )
    response = TestClient(app).get(
        "/v1/rpent/status?backend=hardware",
        headers={
            "Authorization": "Bearer old-scoped-token",
            "X-RPent-Session": "previous-task",
        },
    )
    assert response.status_code in (401, 403) or (
        response.status_code == 200
        and response.json().get("task", {}).get("id") != "current-task"
        and "current task private transcript" not in response.text
    )


def test_stop_during_isolation_preparation_prevents_process_launch(
    tmp_path, monkeypatch
):
    engine = service.Engine(config(tmp_path))
    monkeypatch.setattr(engine, "status", lambda backend: {"ready": True})
    launched = []

    def cancelled_build(config, repo, folder, command, token_file):
        engine.stop_epoch += 1
        engine.state = "stopped"
        return command, {}

    monkeypatch.setattr(service, "build_launch", cancelled_build)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: (
            launched.append(args) or SimpleNamespace(pid=123456, poll=lambda: 0)
        ),
    )
    try:
        engine.start(
            service.Task(
                backend="hardware", instruction="inspect", request_id="review-start"
            )
        )
    except ValueError:
        pass
    assert launched == []
    assert engine.state != "running"


def test_restart_preserves_interrupted_unknown_outcome_on_task_query(tmp_path):
    settings = config(tmp_path)
    engine = service.Engine(settings)
    engine.state = "running"
    engine.task = {
        "id": "review-restart",
        "backend": "hardware",
        "status": "running",
        "task_outcome": "pending",
        "stop_state": "stopping",
    }
    engine.save_history()
    restored = service.Engine(settings)
    result = restored.task_status("review-restart")
    assert result["run_status"] == "interrupted"
    assert result["task_outcome"] == "unknown"
    assert result["stop_state"] == "unknown"


def test_confirmed_stop_releases_pending_skill_ownership(tmp_path):
    engine = service.Engine(config(tmp_path))
    engine.task = {
        "id": "review-task",
        "backend": "hardware",
        "active_skill": "skill-123",
    }
    engine.skill_requests["request-123"] = {
        "executor_id": "skill-123",
        "status": "running",
        "epoch": 0,
        "request": {
            "request_id": "request-123",
            "task_id": "review-task",
            "backend": "hardware",
            "operation": "grasp_object",
        },
    }
    engine.native = SimpleNamespace(call=lambda *args: {"stop_confirmed": True})
    engine.skills = SimpleNamespace(
        request=lambda *args: {"ok": True, "stop_confirmed": True}
    )
    engine.control_stop("hardware")
    assert engine.stop_confirmed is True
    assert engine.pending_skills() == []


def test_late_skill_poll_cannot_restore_running_after_stop(tmp_path):
    engine = service.Engine(config(tmp_path))
    engine.task = {"id": "review-task", "backend": "hardware"}
    item = {
        "executor_id": "skill-123",
        "status": "running",
        "epoch": 0,
        "request": {
            "request_id": "request-123",
            "task_id": "review-task",
            "backend": "hardware",
            "operation": "grasp_object",
        },
    }

    def delayed_response(*args):
        engine.stop_epoch += 1
        item["status"] = "stopped"
        return {"id": "skill-123", "mode": "hardware", "status": "running"}

    engine.skills = SimpleNamespace(request=delayed_response)
    try:
        engine.poll_skill(item)
    except ValueError:
        pass
    assert item["status"] in ("stopped", "interrupted", "cancelled")


def test_available_isolation_can_import_runtime_without_host_services(tmp_path):
    """Only starts an import process; no model, RPC, device or control calls."""
    if not shutil.which("codex"):
        pytest.skip("An installed Codex runtime is required for this local smoke check")
    credential = tmp_path / "dummy-model-key"
    credential.write_text("offline-dummy-credential")
    settings = {
        "planner": {"base_url": "https://example.com", "api_key_file": str(credential)}
    }
    ready, reason = isolation.availability(settings)
    if not ready:
        pytest.skip(reason)
    folder = tmp_path / "review-sandbox"
    folder.mkdir()
    token = folder / "task-token"
    token.write_text("offline-dummy-task-token")
    repo = Path(service.REPO)
    command = [
        str(repo / ".venv/bin/python"),
        "-c",
        "import robots.xlerobot.toolkit; print('isolated runtime import succeeded')",
    ]
    args, env = isolation.build_launch(settings, repo, folder, command, token)
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "isolated runtime import succeeded" in result.stdout
