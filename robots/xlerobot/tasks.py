# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Durable task records owned by the control service, never by the planner."""

import json
import re
import threading
import time
import uuid
from pathlib import Path


def task_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", value):
        raise ValueError("Invalid task ID")
    return value


class TaskJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def read(self, ident):
        path = self.root / (task_id(ident) + ".json")
        if not path.exists():
            raise KeyError(ident)
        with self.lock:
            return json.loads(path.read_text())

    def write(self, task, event=None):
        ident = task_id(task["id"])
        with self.lock:
            try:
                previous = self.read(ident)
            except KeyError:
                previous = {"events": []}
            events = previous["events"]
            if event:
                events.append({
                    "sequence": len(events) + 1,
                    "unix": time.time(),
                    **event,
                })
            value = {"task": dict(task), "events": events}
            path = self.root / (ident + ".json")
            temporary = self.root / (ident + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
            temporary.chmod(0o600)
            temporary.replace(path)

    def interrupt_pending(self):
        for path in self.root.glob("*.json"):
            value = self.read(path.stem)
            task = value["task"]
            if task.get("status") in ("running", "stopping"):
                task.update(
                    status="interrupted", task_outcome="unknown", stop_state="unknown"
                )
                self.write(task, {"type": "interrupted", "reason": "service_restart"})


class SkillRejected(ValueError):
    """An explicit preflight/read-only rejection, not an ambiguous transport."""


class SkillClient:
    """Private, allowlisted executor transport; IDs survive ambiguous requests."""

    def __init__(self, config):
        self.config = config or {}

    def request(self, method, path, body=None):
        import httpx

        if not self.config.get("url") or not self.config.get("token_file"):
            raise ValueError("Calibrated robot skills are not configured")
        token = Path(self.config["token_file"]).read_text().strip()
        with httpx.Client(trust_env=False, timeout=8, follow_redirects=False) as client:
            response = client.request(
                method,
                self.config["url"].rstrip("/") + path,
                headers={"Authorization": "Bearer " + token},
                json=body,
            )
        if (
            response.status_code == 409
            and method == "POST"
            and path in ("/v1/tasks", "/v1/locate", "/v1/verify")
        ):
            detail = response.json().get("detail", {})
            if isinstance(detail, dict) and detail.get("status") == "blocked":
                raise SkillRejected(
                    str(detail.get("reason", "Skill preflight blocked"))[:1000]
                )
        if response.is_error:
            raise ValueError(
                "Skill executor rejected request: " + str(response.status_code)
            )
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("Invalid skill executor result")
        return result

    def status(self, backend):
        try:
            value = self.request("GET", "/v1/status")
            expected = "hardware" if backend == "hardware" else "simulation_only"
            if value.get("mode") != expected:
                raise ValueError("Skill executor mode differs from selected backend")
            return value
        except Exception as exc:
            return {"ready": False, "capabilities": [], "reason": str(exc)}
