# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""Independent RPent management, Codex tasks and borrowed XLeRobot execution."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from robots.xlerobot.control import (
    NativeOwner,
    native_sim_target,
    sim_call,
    simulation_ticks,
)
from robots.xlerobot.isolation import availability, build_launch
from robots.xlerobot.mapping import SimulationMapping, source_pose
from robots.xlerobot.network_observation import observe_cameras
from robots.xlerobot.observation import CAMERAS, DEFAULT_RUNTIME, encode_observation
from robots.xlerobot.tasks import SkillClient, SkillRejected, TaskJournal, task_id

REPO = Path(__file__).resolve().parents[2]
Backend = Literal["simulation", "hardware"]


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class BackendBody(Body):
    backend: Backend = "simulation"


class Action(BackendBody):
    command: Literal["base", "joints", "native"]
    observation_id: str = Field(min_length=1, max_length=128)
    duration: float = Field(default=1.0, ge=0.1, le=2.0)
    v: float = Field(default=0.0, ge=-0.15, le=0.15)
    yaw_rate: float = Field(default=0.0, ge=-0.4, le=0.4)
    targets: dict[str, float] | None = None
    joint: str | None = None
    target_tick: int | None = Field(default=None, ge=0, le=4095)


class Prompt(Body):
    prompt: str = Field(min_length=1, max_length=2000)


class Task(BackendBody):
    instruction: str = Field(min_length=1, max_length=4000)
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class Skill(BackendBody):
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    task_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    operation: Literal[
        "locate_object", "navigate_to_workspace", "grasp_object", "verify_grasp"
    ]
    parameters: dict = Field(default_factory=dict)
    observation_id: str | None = None


class StopBody(Body):
    cancel_task_id: str | None = Field(
        default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )


class MappingBody(Body):
    enabled: bool


class ManualBody(Body):
    stop_epoch: int = Field(ge=0)
    command: Literal["base", "native"]
    duration: float = Field(default=0.3, ge=0.1, le=2.0)
    v: float = Field(default=0.0, ge=-0.15, le=0.15)
    yaw_rate: float = Field(default=0.0, ge=-0.4, le=0.4)
    joint: str | None = None
    target_tick: int | None = Field(default=None, ge=0, le=4095)


class Engine:
    def __init__(self, config):
        self.config = config
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.native = NativeOwner(
            Path(config["native_token_file"]),
            config.get("controller_id"),
            endpoint=config.get("native_endpoint", "http://127.0.0.1:8916"),
            controller_contract=config.get("controller_contract", "native_position"),
            motor_profile_sha256=config.get("motor_profile_sha256"),
        )
        self.journal = TaskJournal(self.output / "records")
        self.journal.interrupt_pending()
        self.skills = SkillClient(config.get("skills"))
        self.task_tokens = {}
        self.skill_requests = {}
        self.task = None
        self.process = None
        self.task_done = None
        self.last_inference = None
        self.inference_task = None
        self.inference_state = "idle"
        self.last_action = None
        self.observations = {}
        self.cancelled_ids = set()
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.last_backend = "simulation"
        self.stop_backend = None
        self.state = "idle"
        self.error = ""
        self.stop_epoch = 0
        self.stop_confirmed = True
        self.mapping = SimulationMapping(
            self.native, lambda: sim_call("status")["state"], self.mapping_fault_stop
        )
        history = self.output / "service-history.json"
        if history.exists():
            try:
                saved = json.loads(history.read_text())
                self.task = saved.get("task")
                self.last_action = saved.get("last_action")
                self.last_inference = saved.get("last_inference")
                self.last_backend = saved.get("last_backend", "simulation")
                self.state = saved.get("state", "idle")
                if self.state in ("running", "stopping"):
                    self.state = "error"
                    self.error = "任务随服务重启中断；不会自动恢复运动"
                    if self.task:
                        self.task.update(
                            status="interrupted",
                            task_outcome="unknown",
                            stop_state="unknown",
                        )
                self.stop_confirmed = False  # Historical STOP is not fresh feedback.
            except (OSError, ValueError, TypeError):
                self.error = "历史记录读取失败；请重新核对实时状态"

    def save_history(self):
        path = self.output / "service-history.json"
        temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "task": self.task,
                    "state": self.state,
                    "last_action": self.last_action,
                    "last_inference": self.last_inference,
                    "last_backend": self.last_backend,
                },
                ensure_ascii=False,
            )
        )
        temporary.replace(path)
        if self.task:
            self.journal.write(self.task)

    def event(self, kind, **details):
        if self.task:
            self.journal.write(self.task, {"type": kind, **details})

    def task_status(self, ident):
        if self.task and self.task["id"] == ident:
            for item in self.skill_requests.values():
                if item["request"]["task_id"] == ident and item["status"] in (
                    "running",
                    "unknown",
                ):
                    try:
                        self.poll_skill(item)
                    except (ValueError, OSError, httpx.HTTPError):
                        pass  # Keep the original unknown outcome and ID.
            self.save_history()
        record = self.journal.read(ident)
        task = record["task"]
        return {
            "task": task,
            "run_status": task.get("status"),
            "stage": task.get("stage", "unknown"),
            "task_outcome": task.get("task_outcome", "unknown"),
            "stop_state": task.get("stop_state", "unknown"),
        }

    def model(self):
        from rpent.utils.rpc import make_rpc_client

        metadata = make_rpc_client("http://127.0.0.1:8095").call(
            "vla.metadata", timeout_s=2
        )
        metadata.pop("checkpoint", None)
        return {
            **metadata,
            "physical_gpu": 1,
            "gpu_uuid": os.environ.get("RPENT_MODEL_GPU_UUID", ""),
        }

    def backend_status(self, backend):
        return (
            sim_call("status")["state"]
            if backend == "simulation"
            else self.native.status()
        )

    def status(self, backend="simulation"):
        try:
            robot = self.backend_status(backend)
            ready = (
                robot.get("ready") is True
                if backend == "simulation"
                else robot.get("fresh") is True and robot.get("held") is True
            )
            reason = "" if ready else "现有真机执行器需要新鲜反馈和已开启的当前位置保持"
            if backend == "hardware" and robot.get("execution_ready") is False:
                ready = False
                reason = (
                    "; ".join(robot.get("blocked_reasons", []))
                    or "Network owner execution gates are not verified"
                )
            if backend == "hardware" and ready:
                axes = robot.get("axes", [])
                calibrated = len(axes) == 14 and all(
                    all(
                        type(a.get(k)) is int
                        for k in ("min_tick", "max_tick", "reference_tick", "raw_tick")
                    )
                    and 0 <= a["min_tick"] < a["max_tick"] <= 4095
                    and a["min_tick"] <= a["reference_tick"] <= a["max_tick"]
                    and a["min_tick"] <= a["raw_tick"] <= a["max_tick"]
                    and a.get("direction") in (-1, 1)
                    and a.get("direction_confirmed_by_operator") is True
                    and a.get("fresh") is True
                    and a.get("torque_enabled") == 1
                    and a.get("status") == 0
                    for a in axes
                )
                if not calibrated:
                    ready, reason = (
                        False,
                        "原生关节标定、方向确认、限位或运动反馈尚未就绪",
                    )
        except Exception as exc:
            robot, ready, reason = {}, False, str(exc)
        # A hardware status request must never wait for Isaac or a model server.
        model = {
            "ready": False,
            "required": False,
            "reason": "Optional simulation diagnostic",
        }
        mapping = {
            "enabled": self.mapping.enabled,
            "ready": False,
            "state": self.mapping.state,
            "blocked_reasons": [],
        }
        if backend == "simulation":
            try:
                model = {"ready": True, **self.model()}
            except Exception as exc:
                model = {"ready": False, "physical_gpu": 1, "reason": str(exc)}
            mapping = self.mapping_status()
        if backend == "hardware" and self.mapping.enabled:
            ready, reason = False, "Simulation mapping owns the hardware"
        skill_status = self.skills.status(backend)
        capabilities = {
            "observe": {"ready": bool(robot), "reason": reason if not robot else ""},
            "task_execution": {"ready": ready, "reason": reason},
            "pi05_inference": {
                "ready": backend == "simulation" and model.get("ready") is True,
                "reason": model.get("reason", ""),
            },
            "simulation_mapping": {
                "ready": mapping.get("ready") is True,
                "reason": "; ".join(mapping.get("blocked_reasons", [])),
            },
            "skills": {
                "ready": skill_status.get("ready") is True,
                "reason": skill_status.get("reason", ""),
            },
        }
        isolated, isolation_reason = availability(self.config)
        capabilities["planner_isolation"] = {
            "ready": isolated,
            "reason": isolation_reason,
        }
        if not isolated:
            ready = False
            reason = "; ".join(filter(None, (reason, isolation_reason)))
            capabilities["task_execution"] = {"ready": False, "reason": reason}
        transcript = ""
        if self.task:
            path = self.output / self.task["id"] / "process.log"
            if path.exists():
                with path.open("rb") as f:
                    f.seek(max(0, path.stat().st_size - 14000))
                    transcript = f.read().decode(errors="replace")
        return {
            "state": self.state,
            "run_status": self.state,
            "stage": (self.task or {}).get("stage", "idle"),
            "task_outcome": (self.task or {}).get("task_outcome", "unknown"),
            "stop_state": "stopped"
            if self.stop_confirmed
            else ("stopping" if self.state == "stopping" else "unknown"),
            "capabilities": capabilities,
            "skill_capabilities": skill_status,
            "experimental": True,
            "ready": ready,
            "backend": backend,
            "robot": robot,
            "model": model,
            "blocked_reasons": [reason] if reason else [],
            "task": self.task,
            "last_inference": self.last_inference,
            "last_action": self.last_action,
            "inference_state": self.inference_state,
            "last_error": self.error,
            "transcript": transcript,
            "stop_confirmed": self.stop_confirmed,
            "control_source": (
                "existing native 16-motor owner"
                if backend == "hardware"
                else "Isaac command bridge; optional native mapping"
            ),
            "pi05_execution_enabled": False,
            "mapping": mapping,
        }

    def mapping_fault_stop(self):
        self.stop_epoch += 1
        self.observations.clear()
        try:
            sim_call("stop")
        except Exception as exc:
            self.error = "映射异常后仿真停止未确认: " + str(exc)

    def mapping_status(self):
        value = self.mapping.status()
        if not value["enabled"] and self.state in ("running", "stopping"):
            value["ready"] = False
            value["blocked_reasons"].append("请先停止当前任务，再开启映射")
        return value

    def set_mapping(self, enabled):
        if not enabled:
            return self.mapping.disable()
        if not self.lock.acquire(blocking=False):
            raise ValueError("当前动作尚未结束，请先停止后开启映射")
        try:
            if self.state in ("running", "stopping"):
                raise ValueError("请先停止当前任务，再开启映射")
            return self.mapping.enable()
        finally:
            self.lock.release()

    def manual_status(self):
        state = sim_call("status")["state"]
        source_pose(state)
        native = self.native.status()
        ticks = simulation_ticks(native, state.get("joints_rad", {}), check_range=False)
        axes = []
        for axis in native["axes"]:
            name, _ = native_sim_target(native, axis["key"], axis["reference_tick"])
            axes.append({
                key: axis[key]
                for key in (
                    "key",
                    "reference_tick",
                    "min_tick",
                    "max_tick",
                    "direction",
                )
            })
            axes[-1].update(
                raw_tick=ticks[axis["key"]],
                joint_rad=state["joints_rad"][name],
                name=name,
            )
        return {
            "source": "simulation",
            "ready": state.get("ready") is True,
            "updated_unix": state["updated_unix"],
            "instance_id": "isaac-simulation",
            "generation": state.get("revision", 0),
            "stop_epoch": self.stop_epoch,
            "task_active": self.state in ("running", "stopping"),
            "mapping": self.mapping_status(),
            "joints_rad": state["joints_rad"],
            "axes": axes,
            "base": {
                "max_v": 0.15,
                "max_yaw_rate": 0.4,
                "velocities_rad_s": state.get("velocities_rad_s", {}),
            },
        }

    def manual(self, body):
        if not self.lock.acquire(blocking=False):
            raise ValueError("另一个动作仍在执行，不能重复提交")
        try:
            epoch = self.stop_epoch
            if body.stop_epoch != epoch:
                raise ValueError("软件操作已被 STOP 撤销，请重新按下控制")
            self.authorize("")
            record = self.observe("simulation")
            if self.stop_epoch != epoch:
                raise ValueError("软件操作已被 STOP 撤销，请重新按下控制")
            action = Action(
                command=body.command,
                observation_id=record["observation_id"],
                duration=body.duration,
                v=body.v,
                yaw_rate=body.yaw_rate,
                joint=body.joint,
                target_tick=body.target_tick,
            )
            reached = True
            if body.command == "native":
                if body.joint is None or body.target_tick is None:
                    raise ValueError("软件关节动作需要电机映射和目标")
                name, desired = native_sim_target(
                    self.native.status(), body.joint, body.target_tick
                )
                current = record["state"]["joints_rad"][name]
                maximum = (0.18 if self.mapping.enabled else 0.45) * body.duration
                delta = desired - current
                reached = abs(delta) <= maximum
                target = current + max(-maximum, min(maximum, delta))
                action = Action(
                    command="joints",
                    observation_id=record["observation_id"],
                    duration=body.duration,
                    targets={name: target},
                )
            if self.stop_epoch != epoch:
                raise ValueError("软件操作已被 STOP 撤销，请重新按下控制")
            return {
                "ok": True,
                "reached": reached,
                "source": "simulation",
                "stop_epoch": epoch,
                "result": self.action(action, ""),
            }
        finally:
            self.lock.release()

    def observe(self, backend):
        if backend == "simulation":
            record = sim_call("observe", cameras=list(CAMERAS))
        else:
            if self.config.get("hardware_cameras"):
                prior = self.last_action or {}
                source = observe_cameras(
                    self.config,
                    not_before=prior.get("unix")
                    if prior.get("backend") == "hardware"
                    else None,
                )
            else:
                key = Path(self.config["console_key_file"]).read_text().strip()
                with httpx.Client(trust_env=False, timeout=8) as client:
                    response = client.post(
                        "http://127.0.0.1:8916/api/observe",
                        headers={"X-Console-Key": key},
                    )
                    response.raise_for_status()
                    source = response.json()
            images = {}
            for name, image in source["images"].items():
                captured = image.get("unix")
                if (
                    type(captured) not in (int, float)
                    or not np.isfinite(captured)
                    or not 0
                    <= time.time() - captured
                    <= self.config.get("max_frame_age_s", 1.0)
                ):
                    raise ValueError("Hardware camera frame is stale: " + name)
                if name not in {
                    "head",
                    "left_wrist",
                    "right_wrist",
                    "hand_left",
                    "hand_right",
                }:
                    raise ValueError("Unknown camera identity")
                if not image.get("path") and not image.get("content"):
                    raise ValueError("Hardware camera unavailable: " + name)
                path = self.output / (source["id"] + "-" + name + ".png")
                Image.open(
                    BytesIO(image["content"]) if image.get("content") else image["path"]
                ).convert("RGB").save(path)
                images[name] = {"path": str(path), "captured_unix": image["unix"]}
                for field in ("source_id", "sequence"):
                    if field in image:
                        images[name][field] = image[field]
            times = [image["captured_unix"] for image in images.values()]
            if not times or max(times) - min(times) > self.config.get(
                "max_camera_skew_s", 0.25
            ):
                raise ValueError("Hardware camera frames are not synchronized")
            record = {
                "observation_id": source["id"],
                "state": self.native.status(),
                "images": images,
                "mode": "hardware",
                "captured_unix": min(times),
                "timestamp_kind": "camera_reported_receive_time",
                "task_id": (self.task or {}).get("id"),
            }
        record["task_id"] = (self.task or {}).get("id")
        ident = record["observation_id"]
        self.observations[ident] = (backend, time.monotonic(), record)
        while len(self.observations) > 128:
            self.observations.pop(next(iter(self.observations)))
        return record

    def authorize(self, session, *, allow_cancelled=False):
        if self.state in ("running", "stopping") and (
            not self.task or not secrets.compare_digest(session, self.task["id"])
        ):
            raise ValueError(
                "Codex task owns this controller; stop the task before manual motion"
            )
        if session and (
            self.state != "running"
            or not self.task
            or session != self.task["id"]
            or (session in self.cancelled_ids and not allow_cancelled)
        ):
            raise ValueError("Task session has been revoked")

    def pending_skills(self):
        return [
            item
            for item in self.skill_requests.values()
            if item["status"] in ("submitting", "running", "unknown")
            and self.task
            and item["request"]["task_id"] == self.task["id"]
        ]

    def action(self, body, session):
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another action or task start is active")
        try:
            self.authorize(session)
            epoch = self.stop_epoch
            if self.pending_skills():
                raise ValueError(
                    "An asynchronous skill owns the controller; query its original ID or stop"
                )
            if body.backend == "hardware" and self.mapping.enabled:
                raise ValueError("仿真映射持有真机控制权，不能叠加直接真机动作")
            if session and body.backend != self.task["backend"]:
                raise ValueError("Task cannot switch execution backends")
            obs = self.observations.pop(body.observation_id, None)
            maximum_age = (
                self.config.get("max_observation_age_s", 2.0)
                if body.backend == "hardware"
                else 120
            )
            if (
                not obs
                or obs[0] != body.backend
                or time.monotonic() - obs[1] > maximum_age
            ):
                raise ValueError(
                    "Observe again before moving: observation expired or consumed"
                )
            self.last_backend = body.backend
            self.stop_confirmed = False
            if body.backend == "simulation":
                parameters = {
                    "observation_id": body.observation_id,
                    "duration": body.duration,
                }
                if body.command == "base":
                    parameters.update(v=body.v, yaw_rate=body.yaw_rate)
                elif body.command == "joints":
                    if not body.targets:
                        raise ValueError("Missing absolute joint targets")
                    parameters["targets"] = body.targets
                else:
                    name, q = native_sim_target(
                        self.native.status(), body.joint, body.target_tick
                    )
                    parameters["targets"] = {name: q}
                if self.mapping.enabled and "targets" in parameters:
                    current = obs[2]["state"]["joints_rad"]
                    if any(
                        abs(target - current[name]) > 0.18 * body.duration
                        for name, target in parameters["targets"].items()
                    ):
                        raise ValueError(
                            "真机映射中的关节动作需低于0.18 rad/s；请分段操作"
                        )
                result = sim_call(
                    "joints" if body.command == "native" else body.command, **parameters
                )
            else:
                if body.command not in ("native", "base"):
                    raise ValueError(
                        "Hardware uses the existing native encoder target tool; simulation radians are not hardware commands"
                    )
                original = obs[2]["state"]
                current = self.native.status()
                if current.get("fresh") is not True or current.get("held") is not True:
                    raise ValueError("Hardware feedback or hold is not ready")
                if (
                    current["instance_id"] != original["instance_id"]
                    or current["generation"] != original["generation"]
                ):
                    raise ValueError(
                        "Native executor generation changed; observe again"
                    )
                result = (
                    self.native.drive(
                        body.v,
                        body.yaw_rate,
                        body.duration,
                        original,
                        lambda: self.stop_epoch != epoch,
                    )
                    if body.command == "base"
                    else self.native.move(body.joint, body.target_tick, original)
                )
            if self.stop_epoch != epoch:
                raise ValueError(
                    "Action result arrived after STOP; query actual feedback"
                )
            if self.task:
                self.task["stage"] = "bounded_motion"
                self.task["task_outcome"] = "pending"
                self.task.pop("grasp_evidence", None)
                self.event("action", backend=body.backend, command=body.command)
            self.last_action = {
                "backend": body.backend,
                "command": body.model_dump(),
                "result": result,
                "unix": time.time(),
            }
            (self.output / "last-action.json").write_text(
                json.dumps(self.last_action, ensure_ascii=False)
            )
            self.save_history()
            return result
        finally:
            self.lock.release()

    def control_stop(self, backend):
        self.stop_epoch += 1
        self.observations.clear()
        self.stop_confirmed = False
        self.stop_backend = backend
        if self.task:
            self.cancelled_ids.add(self.task["id"])
            self.task["stop_state"] = "stopping"
            if self.task.get("task_outcome") != "succeeded":
                self.task["task_outcome"] = "unknown"
            self.event("stop_requested")
        mapped = (
            self.mapping.enabled
            or self.mapping.state == "starting"
            or self.mapping.session is not None
        )
        errors = []
        result = {}
        # Always reach the motor owner even when an optional service is lost.
        try:
            result = (
                sim_call("stop")
                if backend == "simulation"
                else self.native.call("stop", {"operator_ready": True})
            )
            self.stop_confirmed = (
                result.get("ok") is True
                if backend == "simulation"
                else result.get("stop_confirmed") is True
            )
            if backend == "hardware" and not self.stop_confirmed:
                generation = result.get("stop_generation", result.get("generation"))
                deadline = time.monotonic() + self.config.get("stop_timeout_s", 5.0)
                while time.monotonic() < deadline:
                    snapshot = self.native.status()
                    if (
                        snapshot.get("fresh") is True
                        and snapshot.get("stop_confirmed") is True
                        and (
                            not result.get("instance_id")
                            or snapshot.get("instance_id") == result["instance_id"]
                        )
                        and (
                            generation is None
                            or snapshot.get(
                                "stop_generation", snapshot.get("generation", -1)
                            )
                            >= generation
                        )
                    ):
                        self.stop_confirmed = True
                        break
                    time.sleep(0.05)
        except Exception as exc:
            errors.append("motor owner: " + str(exc))
        if mapped:
            try:
                mapping_result = self.mapping.disable()
                if mapping_result.get("stop_confirmed") is not True:
                    errors.append("mapping stop is unconfirmed")
            except Exception as exc:
                errors.append("mapping: " + str(exc))
        if self.task and self.task.get("active_skill"):
            try:
                ids = [item["executor_id"] for item in self.skill_requests.values()]
                stopped = self.skills.request(
                    "POST", "/v1/stop", {"cancel_task_ids": ids}
                )
                if (
                    stopped.get("ok") is not True
                    or stopped.get("stop_confirmed", True) is not True
                ):
                    errors.append("skill stop is unconfirmed")
            except Exception as exc:
                errors.append("skill: " + str(exc))
        self.stop_confirmed = self.stop_confirmed and not errors
        if self.stop_confirmed:
            for item in self.pending_skills():
                item["status"] = "cancelled"
        if not self.stop_confirmed:
            self.error = "STOP 未确认: " + "; ".join(
                errors or ["fresh stopped feedback missing"]
            )
        if self.task:
            self.task["stop_state"] = "stopped" if self.stop_confirmed else "unknown"
            self.event("stop_result", confirmed=self.stop_confirmed)
        return {**result, "stop_confirmed": self.stop_confirmed, "errors": errors}

    def skill(self, body, session):
        """Dispatch a task-scoped skill once; uncertain submissions stay queryable."""
        if not self.lock.acquire(blocking=False):
            raise ValueError("Another action is active")
        try:
            self.authorize(session)
            if (
                not self.task
                or self.task["id"] != body.task_id
                or self.state != "running"
            ):
                raise ValueError("Skill requires an active owning task")
            if body.backend != self.task["backend"]:
                raise ValueError("Task cannot switch execution backends")
            fingerprint = body.model_dump()
            cached = self.skill_requests.get(body.request_id)
            if cached:
                if cached["request"] != fingerprint:
                    raise ValueError("Request ID reused with different parameters")
                return self.poll_skill(cached)
            if self.pending_skills():
                raise ValueError(
                    "An asynchronous skill owns the controller; query its original ID or stop"
                )
            epoch = self.stop_epoch
            capability = self.skills.status(body.backend)
            operation = capability.get("operations", {}).get(body.operation, {})
            if operation.get("ready", capability.get("ready")) is not True:
                raise ValueError(capability.get("reason") or "Robot skill is not ready")
            if epoch != self.stop_epoch:
                raise ValueError("Skill revoked by STOP")
            ident = uuid.uuid5(
                uuid.NAMESPACE_URL, body.task_id + "/" + body.request_id
            ).hex
            item = {
                "request": fingerprint,
                "executor_id": ident,
                "status": "submitting",
                "epoch": epoch,
            }
            self.skill_requests[body.request_id] = item
            self.task.update(stage=body.operation, active_skill=ident)
            self.task.setdefault("skill_requests", {})[body.request_id] = item
            self.event(
                "skill_submitted",
                request_id=body.request_id,
                executor_id=ident,
                operation=body.operation,
            )
            try:
                if body.operation == "locate_object":
                    item["result"] = self.skills.request(
                        "POST", "/v1/locate", body.parameters
                    )
                    item["status"] = (
                        "succeeded"
                        if item["result"].get("status") == "ready"
                        else "blocked"
                    )
                elif body.operation == "verify_grasp":
                    if set(body.parameters) != {"task_id"} or body.parameters[
                        "task_id"
                    ] != self.task.get("grasp_executor_id"):
                        raise ValueError(
                            "Verify must reference this task's completed grasp"
                        )
                    self.task["task_outcome"] = "pending"
                    self.task.pop("grasp_evidence", None)
                    item["result"] = self.skills.request(
                        "POST", "/v1/verify", body.parameters
                    )
                    evidence = item["result"].get("evidence", {})
                    verified = (
                        evidence.get("grasp_verified") is True
                        and evidence.get("lift_m", 0) >= 0.05
                        and evidence.get("hold_s", 0) >= 3
                        and item["result"].get("mode")
                        == (
                            "hardware"
                            if body.backend == "hardware"
                            else "simulation_only"
                        )
                    )
                    item["status"] = "succeeded" if verified else "failed"
                    if self.stop_epoch == epoch and verified:
                        self.task.update(
                            task_outcome="succeeded", grasp_evidence=evidence
                        )
                    elif self.stop_epoch == epoch:
                        self.task["task_outcome"] = "failed"
                else:
                    kind = (
                        "navigate"
                        if body.operation == "navigate_to_workspace"
                        else "grasp"
                    )
                    self.task["task_outcome"] = "pending"
                    self.task.pop("grasp_evidence", None)
                    self.skills.request(
                        "POST",
                        "/v1/tasks",
                        {
                            "id": ident,
                            "kind": kind,
                            "mode": "hardware"
                            if body.backend == "hardware"
                            else "simulation_only",
                            "instance_id": capability.get("instance_id"),
                            "execution_epoch": capability.get("execution_epoch"),
                            "parameters": body.parameters,
                        },
                    )
                    item["status"] = "running"
                    if kind == "grasp":
                        self.task["grasp_executor_id"] = ident
                if self.stop_epoch != epoch:
                    item["status"] = "interrupted"
                    self.task["task_outcome"] = "unknown"
                    # Fence a task submission that reached the executor after STOP.
                    self.skills.request(
                        "POST", "/v1/stop", {"cancel_task_ids": [ident]}
                    )
                self.event(
                    "skill_result", request_id=body.request_id, status=item["status"]
                )
                return self.poll_skill(item)
            except SkillRejected as exc:
                item["status"] = (
                    "blocked" if self.stop_epoch == epoch else "interrupted"
                )
                item["result"] = {"status": "blocked", "reason": str(exc)}
                if body.operation == "verify_grasp" and self.stop_epoch == epoch:
                    self.task["task_outcome"] = "failed"
                self.event("skill_blocked", request_id=body.request_id, reason=str(exc))
                return self.poll_skill(item)
            except Exception:
                item["status"] = "unknown"
                self.task["task_outcome"] = "unknown"
                self.event(
                    "skill_unknown", request_id=body.request_id, executor_id=ident
                )
                raise
        finally:
            self.lock.release()

    def poll_skill(self, item):
        if item["status"] in ("running", "unknown", "submitting") and item["request"][
            "operation"
        ] in ("grasp_object", "navigate_to_workspace"):
            result = self.skills.request("GET", "/v1/tasks/" + item["executor_id"])
            if (
                item["epoch"] != self.stop_epoch
                or not self.task
                or item["request"]["task_id"] != self.task["id"]
            ):
                item["status"] = "interrupted"
                return {
                    "request_id": item["request"]["request_id"],
                    "executor_id": item["executor_id"],
                    "status": "interrupted",
                    "task_success_verified": False,
                }
            if result.get("id") != item["executor_id"]:
                raise ValueError("Executor returned a different task")
            expected_mode = (
                "hardware"
                if item["request"]["backend"] == "hardware"
                else "simulation_only"
            )
            if result.get("mode") != expected_mode:
                raise ValueError("Executor returned a different backend")
            if result.get("status") == "succeeded":
                required = (
                    "goal_reached"
                    if item["request"]["operation"] == "navigate_to_workspace"
                    else "grasp_verified"
                )
                if result.get("evidence", {}).get(required) is not True:
                    raise ValueError("Executor success is missing " + required)
            item["status"] = result.get("status", "unknown")
            item["result"] = result
        return {
            "request_id": item["request"]["request_id"],
            "executor_id": item["executor_id"],
            "status": item["status"],
            "result": item.get("result"),
            "task_success_verified": False,
        }

    def infer(self, prompt):
        from robots.xlerobot.diagnostic import XLeRobotVLAClient
        from rpent.utils.rpc import make_rpc_client

        with self.inference_lock:
            record = self.observe("simulation")
            description = json.loads(
                (DEFAULT_RUNTIME / "robot_description.json").read_text()
            )
            obs = encode_observation(record, description, prompt)
            prediction = XLeRobotVLAClient(
                make_rpc_client("http://127.0.0.1:8095")
            ).predict(obs)
            actions = np.asarray(prediction["actions"], dtype=np.float32)
            if actions.shape != (50, 32) or not np.isfinite(actions).all():
                raise ValueError("Invalid Pi0.5 action tensor")
            ident = uuid.uuid4().hex
            np.save(self.output / (ident + "-actions.npy"), actions)
            self.last_inference = {
                "id": ident,
                "shape": list(actions.shape),
                "finite": True,
                "server_seconds": prediction["wall_seconds"],
                "physical_gpu": 1,
                "execution_enabled": False,
                "observation_id": record["observation_id"],
                "unix": time.time(),
            }
            self.save_history()
            return self.last_inference

    def start(self, body):
        epoch = self.stop_epoch
        with self.lock:
            try:
                previous = self.journal.read(body.request_id)["task"]
            except KeyError:
                previous = None
            if previous:
                if (
                    previous.get("instruction") != body.instruction
                    or previous.get("backend") != body.backend
                ):
                    raise ValueError("Task ID reused with different parameters")
                return {
                    **self.task_status(body.request_id),
                    "state": previous["status"],
                }
            if body.backend == "hardware" and self.mapping.enabled:
                raise ValueError("仿真映射已经持有真机控制权")
            if (
                body.backend == "hardware"
                and not self.stop_confirmed
                and self.state not in ("running", "stopping")
            ):
                # A restart discards historical confirmation. The verified
                # owner may independently provide a new, fresh physical STOP
                # sample; reading it does not resume or send another STOP.
                current = self.native.status()
                if (
                    current.get("fresh") is True
                    and current.get("stop_confirmed") is True
                ):
                    self.stop_confirmed = True
                    self.stop_backend = "hardware"
            if (
                self.task
                and self.state not in ("running", "stopping", "idle")
                and not self.stop_confirmed
            ):
                raise ValueError(
                    "Previous task stop is unconfirmed; verify STOP before a new task"
                )
            if self.state in ("running", "stopping"):
                if self.task and body.request_id == self.task["id"]:
                    return {**self.task_status(body.request_id), "state": self.state}
                raise ValueError("A task is already active")
            if (
                body.request_id in self.cancelled_ids
                or (self.output / body.request_id).exists()
            ):
                raise ValueError(
                    "Task ID is single-use; query status for the original result"
                )
            readiness = self.status(body.backend)
            if not readiness["ready"]:
                self.journal.write(
                    {
                        "id": body.request_id,
                        "backend": body.backend,
                        "instruction": body.instruction,
                        "status": "rejected",
                        "stage": "preflight",
                        "task_outcome": "failed",
                        "stop_state": "stopped"
                        if self.stop_confirmed and self.stop_backend == body.backend
                        else "unknown",
                        "blocked_reasons": readiness.get("blocked_reasons", []),
                        "submitted_unix": time.time(),
                        "motion_submitted": False,
                    },
                    {"type": "preflight_rejected", "motion_submitted": False},
                )
                raise ValueError(
                    "Selected executor is not ready; query original task "
                    + body.request_id
                )
            if epoch != self.stop_epoch or body.request_id in self.cancelled_ids:
                raise ValueError("Task start revoked by STOP")
            folder = self.output / body.request_id
            folder.mkdir(mode=0o700)
            scoped_token = secrets.token_urlsafe(48)
            token_file = folder / "task-token"
            token_file.write_text(scoped_token)
            token_file.chmod(0o600)
            self.task_tokens[scoped_token] = body.request_id
            env = {
                **os.environ,
                "RPENT_CONTROL_SESSION": body.request_id,
                "RPENT_CONTROL_TOKEN_FILE": str(token_file),
            }
            # Host and motor configuration is never inherited by a planner.
            for name in (
                "RPENT_CONTROL_CONFIG",
                "XLE_API_TOKEN",
                "XLE_EDGE_TOKEN",
                "XLE_EDGE2_TOKEN",
            ):
                env.pop(name, None)
            command = [
                str(REPO / ".venv/bin/rpent"),
                "--robot",
                "xlerobot",
                "--planner",
                "codex",
                "--backend",
                body.backend,
                "--instruction",
                body.instruction,
                "--output-dir",
                str(folder / "run"),
                "--planner-timeout-s",
                "300",
                "--max-turns",
                "16",
                "--reasoning-effort",
                "low",
            ]
            self.task = {
                "id": body.request_id,
                "backend": body.backend,
                "instruction": body.instruction,
                "started_unix": time.time(),
                "status": "running",
                "stage": "observing",
                "task_outcome": "pending",
                "stop_state": "unknown",
            }
            self.last_backend, self.state, self.error = body.backend, "running", ""
            self.stop_confirmed = False
            try:
                # Simulation can map onto hardware; both backends get the same
                # task-only filesystem and authenticated tool boundary.
                command, env = build_launch(
                    self.config, REPO, folder, command, token_file
                )
                if self.stop_epoch != epoch or body.request_id in self.cancelled_ids:
                    raise ValueError("Task launch revoked by STOP")
                with (folder / "process.log").open("wb") as stream:
                    self.process = subprocess.Popen(
                        command,
                        cwd=REPO,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                if self.stop_epoch != epoch or body.request_id in self.cancelled_ids:
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGTERM)
                    raise ValueError("Late task launch revoked by STOP")
                self.task["pid"] = self.process.pid
                self.event("started", backend=body.backend)
                self.save_history()
            except Exception:
                self.state = "error"
                self.task.update(status="failed", task_outcome="failed")
                self.task_tokens.pop(scoped_token, None)
                self.event("launch_failed")
                raise
            return {**self.task_status(body.request_id), "state": self.state}

    async def watch(self, process):
        code = await asyncio.to_thread(process.wait)
        if self.process is process and self.task:
            self.task["exit_code"] = code
            self.task["finished_unix"] = time.time()
            if self.state == "running":
                try:
                    await asyncio.to_thread(self.control_stop, self.task["backend"])
                    self.state = (
                        "completed" if code == 0 and self.stop_confirmed else "error"
                    )
                except Exception:
                    self.state = "error"
                self.task["status"] = self.state
                if self.task.get("task_outcome") != "succeeded":
                    self.task["task_outcome"] = "unknown" if code == 0 else "failed"
                if not self.stop_confirmed:
                    self.task["task_outcome"] = "unknown"
                if code:
                    self.error = "Codex task exited with code " + str(code)
                self.save_history()
                self.event(
                    "finished", exit_code=code, task_outcome=self.task["task_outcome"]
                )

    async def stop(self, cancel_task_id=None):
        self.stop_epoch += 1
        if cancel_task_id:
            self.cancelled_ids.add(cancel_task_id)
        self.state = "stopping"
        process = self.process
        if self.task:
            self.cancelled_ids.add(self.task["id"])
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.to_thread(self.control_stop, self.last_backend)
        finally:
            if process and process.poll() is None:
                try:
                    await asyncio.wait_for(asyncio.to_thread(process.wait), 8)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.to_thread(process.wait)
        self.state = "stopped" if self.stop_confirmed else "error"
        if self.task:
            self.task["status"] = self.state
            if self.task.get("task_outcome") != "succeeded":
                self.task["task_outcome"] = "unknown"
            self.task["stop_state"] = "stopped" if self.stop_confirmed else "unknown"
        self.save_history()
        result = (
            self.task_status(self.task["id"])
            if self.task
            else {
                "run_status": self.state,
                "stage": "idle",
                "task_outcome": "unknown",
                "stop_state": "stopped" if self.stop_confirmed else "unknown",
            }
        )
        return {**result, "state": self.state, "stop_confirmed": self.stop_confirmed}


def create_app(config=None):
    config = config or json.loads(Path(os.environ["RPENT_CONTROL_CONFIG"]).read_text())
    token = Path(config["token_file"]).read_text().strip()
    engine = Engine(config)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if engine.process and engine.process.poll() is None:
            await engine.stop()
        if engine.mapping.enabled or engine.mapping.state == "starting":
            await asyncio.to_thread(engine.mapping.disable)

    app = FastAPI(title="Independent RPent XLeRobot control", lifespan=lifespan)
    app.state.engine = engine

    async def authenticate(request: Request, authorization: str = Header(default="")):
        if secrets.compare_digest(authorization, "Bearer " + token):
            return
        credential = authorization.removeprefix("Bearer ")
        ident = engine.task_tokens.get(credential)
        if not ident or request.headers.get("X-RPent-Session") != ident:
            raise HTTPException(401, "Control API authentication required")
        if not engine.task or engine.task["id"] != ident:
            raise HTTPException(403, "Task credential has been superseded")
        try:
            task = engine.journal.read(ident)["task"]
        except KeyError:
            raise HTTPException(401, "Task credential is unavailable") from None
        suffix = request.url.path.removeprefix("/v1/rpent/")
        if request.method == "GET":
            allowed = suffix in (
                "status",
                "tasks/" + ident,
                "tasks/" + ident + "/events",
            )
            if suffix.startswith("camera/"):
                record = engine.observations.get(suffix.split("/")[1])
                allowed = bool(record and record[2].get("task_id") == ident)
            if (
                suffix == "status"
                and request.query_params.get("backend", "simulation") != task["backend"]
            ):
                allowed = False
        else:
            allowed = suffix in ("observe", "action", "skills", "control/stop")
            if suffix == "infer/wait" and task["backend"] == "simulation":
                allowed = True
            try:
                body = await request.json()
                if suffix != "infer/wait" and (
                    not isinstance(body, dict)
                    or body.get("backend", "simulation") != task["backend"]
                ):
                    allowed = False
            except ValueError:
                allowed = False
            if (
                engine.state != "running"
                or not engine.task
                or engine.task["id"] != ident
            ):
                allowed = False
        if not allowed:
            raise HTTPException(403, "Task credential cannot access this operation")
        request.state.task_scope = ident

    auth = [Depends(authenticate)]

    async def invoke(func, *args):
        try:
            return await asyncio.to_thread(func, *args)
        except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/v1/rpent/status", dependencies=auth)
    async def status(request: Request, backend: Backend = "simulation"):
        result = await invoke(engine.status, backend)
        ident = getattr(request.state, "task_scope", None)
        if ident and (
            not engine.task
            or engine.task["id"] != ident
            or (result.get("task") or {}).get("id") != ident
        ):
            raise HTTPException(403, "Task credential has been superseded")
        return result

    @app.get("/v1/rpent/tasks/{ident}", dependencies=auth)
    async def task_record(ident: str):
        try:
            return await invoke(engine.task_status, task_id(ident))
        except (KeyError, ValueError):
            raise HTTPException(404, "Unknown task ID") from None

    @app.get("/v1/rpent/tasks/{ident}/events", dependencies=auth)
    async def task_events(ident: str):
        try:
            return {"events": engine.journal.read(task_id(ident))["events"]}
        except (KeyError, ValueError):
            raise HTTPException(404, "Unknown task ID") from None

    @app.post("/v1/rpent/skills", dependencies=auth)
    async def skill(body: Skill, x_rpent_session: str = Header(default="")):
        return await invoke(engine.skill, body, x_rpent_session)

    @app.get("/v1/rpent/mapping", dependencies=auth)
    async def mapping_status():
        return {"mapping": await invoke(engine.mapping_status)}

    @app.post("/v1/rpent/mapping", dependencies=auth)
    async def mapping(body: MappingBody):
        return {"mapping": await invoke(engine.set_mapping, body.enabled)}

    @app.get("/v1/rpent/manual/status", dependencies=auth)
    async def manual_status():
        return await invoke(engine.manual_status)

    @app.post("/v1/rpent/manual", dependencies=auth)
    async def manual(body: ManualBody):
        return await invoke(engine.manual, body)

    @app.post("/v1/rpent/observe", dependencies=auth)
    async def observe(body: BackendBody):
        return await invoke(engine.observe, body.backend)

    @app.get("/v1/rpent/camera/{observation_id}/{camera}", dependencies=auth)
    async def camera(observation_id: str, camera: str):
        record = engine.observations.get(observation_id)
        image = record[2].get("images", {}).get(camera) if record else None
        if not image:
            raise HTTPException(404, "Observe first")
        return Response(
            Path(image["path"]).read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/rpent/action", dependencies=auth)
    async def action(body: Action, x_rpent_session: str = Header(default="")):
        return await invoke(engine.action, body, x_rpent_session)

    @app.post("/v1/rpent/infer/wait", dependencies=auth)
    async def infer_wait(body: Prompt):
        return await invoke(engine.infer, body.prompt)

    async def infer_background(prompt):
        engine.inference_state = "running"
        try:
            await asyncio.to_thread(engine.infer, prompt)
            engine.inference_state = "completed"
        except Exception as exc:
            engine.inference_state = "error"
            engine.error = str(exc)

    @app.post("/v1/rpent/infer", dependencies=auth)
    async def infer(body: Prompt):
        if engine.inference_task and not engine.inference_task.done():
            raise HTTPException(409, "Inference is already running")
        engine.inference_task = asyncio.create_task(infer_background(body.prompt))
        return {"state": "running", "execution_enabled": False}

    @app.post("/v1/rpent/control/stop", dependencies=auth)
    async def control_stop(
        body: BackendBody, x_rpent_session: str = Header(default="")
    ):
        try:
            engine.authorize(x_rpent_session, allow_cancelled=True)
            if x_rpent_session and body.backend != engine.task["backend"]:
                raise ValueError("Task cannot stop a different execution backend")
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        return await invoke(engine.control_stop, body.backend)

    @app.post("/v1/rpent/tasks", dependencies=auth)
    async def start(body: Task):
        result = await invoke(engine.start, body)
        if (
            engine.process is not None
            and engine.task
            and engine.task["id"] == body.request_id
            and engine.state == "running"
            and (not engine.task_done or engine.task_done.done())
        ):
            engine.task_done = asyncio.create_task(engine.watch(engine.process))
        return result

    @app.post("/v1/rpent/stop", dependencies=auth)
    async def stop(body: StopBody):
        return await engine.stop(body.cancel_task_id)

    return app
