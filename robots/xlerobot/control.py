# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""Thin clients for the existing simulator and native sixteen-motor owner."""

from __future__ import annotations

import math
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

from robots.xlerobot.observation import DEFAULT_RUNTIME

WORKSPACE = Path(__file__).resolve().parents[3]
CONTROL_SOURCE = WORKSPACE / "xiaozhi-disp-v1.1/robot-control"


def executor_mappings() -> tuple[dict, dict]:
    """Read names from the existing control-source contracts lazily."""
    if str(CONTROL_SOURCE) not in sys.path:
        sys.path.insert(0, str(CONTROL_SOURCE))
    from policy_runtime.contract import MOTORS
    from whole_body.protocol import NATIVE_KEYS

    return NATIVE_KEYS, MOTORS


def sim_call(command: str, **parameters) -> dict:
    """Borrow the existing bounded simulator protocol without starting Isaac."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "xlerobot_sim_client", WORKSPACE / "xlerobot_isaac/codex_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.request(DEFAULT_RUNTIME, {"command": command, **parameters})
    if not result.get("ok"):
        raise ValueError(result.get("error", "Simulator rejected the request"))
    return result


class NativeOwner:
    """Use the already deployed position facade; never open another serial bus."""

    def __init__(
        self,
        token_file: Path,
        controller_id: str | None = None,
        *,
        endpoint: str = "http://127.0.0.1:8916",
        controller_contract: str = "native_position",
        motor_profile_sha256: str | None = None,
    ):
        self.token_file = token_file
        self.controller_id = controller_id
        url = httpx.URL(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Configure an explicit native owner HTTP origin")
        if controller_contract not in ("native_position", "whole_body"):
            raise ValueError("Unsupported native owner controller contract")
        if controller_contract == "whole_body" and not re.fullmatch(
            r"[a-f0-9]{64}", motor_profile_sha256 or ""
        ):
            raise ValueError(
                "Whole-body owner requires the measured motor profile SHA256"
            )
        self.endpoint = str(url).rstrip("/")
        self.controller_contract = controller_contract
        self.motor_profile_sha256 = motor_profile_sha256

    def call(self, path: str, body: dict | None = None) -> dict:
        stopping = self.controller_contract == "whole_body" and path == "stop"
        if stopping:
            if body != {"operator_ready": True}:
                raise ValueError("Explicit STOP request required")
            # The owner's priority STOP receipt only acknowledges the request.
            # A later /v1/state sample must establish fresh physical confirmation.
            route = "/v1/control/stop"
            body = {"reason": "RPent requested stop"}
        else:
            route = path if path.startswith("/v1/") else "/v1/position-control/" + path
        token = self.token_file.read_text().strip()
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=8) as client:
            response = client.request(
                "GET" if body is None else "POST",
                self.endpoint + route,
                headers={"Authorization": "Bearer " + token},
                json=body,
            )
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Native executor returned an invalid response")
        if not response.is_success:
            raise ValueError(
                str(value.get("detail", "Native executor rejected request"))
            )
        if stopping:
            if (
                value.get("type") != "stopped"
                or not isinstance(value.get("instance_id"), str)
                or not value["instance_id"]
                or type(value.get("generation")) is not int
                or value["generation"] < 0
            ):
                raise ValueError(
                    "Whole-body STOP receipt is invalid; stopping remains unknown"
                )
            value = {
                **value,
                "stop_confirmed": False,
                "stop_generation": value.get("generation"),
                "request_accepted": True,
            }
        return value

    def status(self) -> dict:
        value = self.call("status")
        native_keys, _ = executor_mappings()
        axes = value.get("axes")
        mapping = (
            {a.get("key"): a.get("name") for a in axes}
            if isinstance(axes, list)
            and all(
                isinstance(a, dict)
                and isinstance(a.get("key"), str)
                and isinstance(a.get("name"), str)
                for a in axes
            )
            else {}
        )
        if (
            not self.controller_id
            or value.get("controller_id") != self.controller_id
            or value.get("scope") != "xlerobot_040_native_position"
            or value.get("source_hash_verified") is not True
            or mapping != native_keys
            or not isinstance(axes, list)
            or len(axes) != len(native_keys)
        ):
            raise ValueError("Existing native motor identity/mapping mismatch")
        if self.controller_contract == "whole_body":
            value = self._whole_body_status(value, native_keys)
        return value

    def _whole_body_status(self, native: dict, native_keys: dict) -> dict:
        """Bind the facade to the same whole-body owner and autonomous gates."""
        state, profile = self.call("/v1/state"), self.call("/v1/profile")
        names = set(native_keys.values())
        position_order = profile.get("position_order")
        positions = state.get("positions_ticks")
        instance = state.get("instance_id")
        if (
            any(
                value.get("controller_id") != self.controller_id
                for value in (state, profile)
            )
            or not isinstance(instance, str)
            or not instance
            or native.get("instance_id") != instance
            or type(native.get("generation")) is not int
            or type(state.get("generation")) is not int
            or any(
                value.get("motor_profile_sha256") != self.motor_profile_sha256
                for value in (state, profile)
            )
            or state.get("profile_sha256", self.motor_profile_sha256)
            != self.motor_profile_sha256
            or native.get("motor_profile_sha256", self.motor_profile_sha256)
            != self.motor_profile_sha256
            or not isinstance(position_order, list)
            or len(position_order) != len(names)
            or not all(isinstance(name, str) for name in position_order)
            or set(position_order) != names
            or not isinstance(positions, dict)
            or set(positions) != names
            or any(
                type(tick) is not int or not 0 <= tick <= 4095
                for tick in positions.values()
            )
        ):
            raise ValueError("Whole-body owner identity/instance/profile mismatch")
        proofs = profile.get("proofs", {})
        gates = state.get("gates", {})
        if not isinstance(proofs, dict) or not isinstance(gates, dict):
            raise ValueError("Whole-body owner returned invalid proof gates")
        proof_names = (
            "calibration_verified",
            "motion_verified",
            "stop_verified",
            "alignment_verified",
            "direction_verified",
            "timing_verified",
            "camera_bindings_verified",
        )
        runtime_names = (
            "identity_verified",
            "feedback_fresh",
            "health_fresh",
            "in_range",
            "healthy",
        )
        blocked = [
            "whole_body proof missing: " + key
            for key in proof_names
            if proofs.get(key) is not True or gates.get(key) is not True
        ]
        blocked.extend(
            "whole_body runtime gate blocked: " + key
            for key in runtime_names
            if gates.get(key) is not True
        )
        fresh = native.get("fresh") is True and state.get("fresh") is True
        held = native.get("held") is True and state.get("held") is True
        for key, passed in (
            ("fresh feedback", fresh),
            ("position hold", held),
            ("owner ready", state.get("ready") is True),
            ("stop latch cleared", state.get("stop_latched") is False),
            ("fault clear", state.get("fault") is None),
            ("owner session free", state.get("session") is None),
            ("manual action free", state.get("manual_active") is None),
        ):
            if not passed:
                blocked.append("whole_body " + key + " blocked")
        return {
            **native,
            "controller_contract": self.controller_contract,
            "motor_profile_sha256": self.motor_profile_sha256,
            "fresh": fresh,
            "held": held,
            "ready": not blocked,
            "execution_ready": not blocked,
            "proofs": proofs,
            "gates": gates,
            "blocked_reasons": blocked,
            "owner_generation": state["generation"],
            "stop_generation": state["generation"],
            "stop_confirmed": fresh and state.get("stop_confirmed") is True,
            "stop_latched": state.get("stop_latched"),
            "position_torque_preserved": state.get("position_torque_preserved"),
            "state_seq": state.get("state_seq"),
            "observed_at_unix_ns": state.get("observed_at_unix_ns"),
        }

    def _require_execution_ready(self, status: dict) -> None:
        if (
            self.controller_contract == "whole_body"
            and status.get("execution_ready") is not True
        ):
            raise ValueError(
                "Whole-body autonomous motion gates are blocked: "
                + "; ".join(status.get("blocked_reasons", []))
            )

    def move(self, joint: str, target_tick: int, expected: dict) -> dict:
        status = self.status()
        self._require_execution_ready(status)
        if not status.get("fresh") or not status.get("held"):
            raise ValueError(
                "请在已有关节控制面板确认当前位置保持；RPent 不自动开启扭矩"
            )
        if any(status[k] != expected[k] for k in ("instance_id", "generation")):
            raise ValueError("Native executor generation changed before motion")
        command_id = uuid.uuid4().hex
        result = self.call(
            "target",
            {
                "id": command_id,
                "joint": joint,
                "target_tick": target_tick,
                "speed_value": 130,
                "operator_ready": True,
                "instance_id": status["instance_id"],
                "generation": status["generation"],
            },
        )
        deadline = time.monotonic() + 12
        while result.get("status") not in (
            "succeeded",
            "failed",
            "stopped",
            "rejected",
            "interrupted",
            "cancelled",
        ):
            if time.monotonic() > deadline:
                self.call("stop", {"operator_ready": True})
                raise ValueError(
                    "Native motion timed out; query original command " + command_id
                )
            time.sleep(0.1)
            result = self.call("commands/" + command_id)
        if result.get("status") != "succeeded":
            raise ValueError("Native command " + command_id + ": " + str(result))
        return result

    def drive(
        self, v: float, yaw_rate: float, duration: float, expected: dict, cancelled
    ) -> dict:
        """Borrow the existing base lease and degree/s convention, then stop.

        The operator must have already armed all motors through the existing
        hold panel. No torque-enable endpoint is called by this adapter.
        """
        status = self.status()
        self._require_execution_ready(status)
        base = status.get("base_control", {})
        if (
            not status.get("fresh")
            or not status.get("held")
            or base.get("fresh") is not True
            or base.get("armed") is not True
            or base.get("fault")
            or status.get("parking", {}).get("active")
            or any(a.get("torque_enabled") != 1 for a in status["axes"])
            or len(base.get("wheels", [])) != 2
            or any(w.get("torque") != 1 for w in base["wheels"])
        ):
            raise ValueError("请先通过已有控制面板保持整机，核对双轮新鲜反馈")
        if any(status[k] != expected[k] for k in ("instance_id", "generation")):
            raise ValueError("Native executor generation changed before base motion")
        theta = math.degrees(yaw_rate)
        if abs(v) > base["max_x"] or abs(theta) > base["max_theta"]:
            raise ValueError("Base velocity exceeds existing executor limits")
        if cancelled():
            raise ValueError("Base action revoked by STOP")
        frames = 0
        try:
            lease = self.call(
                "base-begin",
                {
                    "operator_ready": True,
                    "instance_id": status["instance_id"],
                    "generation": status["generation"],
                },
            )["lease"]
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if cancelled():
                    raise ValueError("Base action revoked by STOP")
                self.call(
                    "base-frame",
                    {
                        "lease": lease,
                        "seq": frames,
                        "action": {"x.vel": v, "theta.vel": theta},
                    },
                )
                frames += 1
                time.sleep(min(0.15, max(0, deadline - time.monotonic())))
        finally:
            self.call("base-stop", {"operator_ready": True})
        actual = self.status()
        return {
            "completed": True,
            "frames": frames,
            "state": actual,
            "units": {"x.vel": "m/s", "theta.vel": "degree/s"},
            "stop_confirmed": actual.get("base_control", {}).get("moving") is False
            and actual.get("base_control", {}).get("fresh") is True
            and len(actual.get("base_control", {}).get("wheels", [])) == 2
            and all(w.get("velocity") == 0 for w in actual["base_control"]["wheels"]),
        }


def native_sim_target(status: dict, joint: str, target_tick: int) -> tuple[str, float]:
    """Apply the same reference/direction convention used by robot_servo_pose.js."""
    native_keys, motors = executor_mappings()
    if type(target_tick) is not int or joint not in native_keys:
        raise ValueError("Unknown native joint or noninteger encoder target")
    axis = next(a for a in status["axes"] if a["key"] == joint)
    if axis.get("direction_confirmed_by_operator") is not True or axis.get(
        "direction"
    ) not in (-1, 1):
        raise ValueError("Native-to-USD direction has not been confirmed")
    if not axis["min_tick"] <= target_tick <= axis["max_tick"]:
        raise ValueError("Target exceeds existing measured encoder range")
    name = next(n for n, motor in motors.items() if motor == native_keys[joint])
    neutral = (
        math.pi / 8
        if name.endswith("_Jaw")
        else math.pi / 2
        if name.endswith(("_Pitch", "_Elbow"))
        and not name.startswith("Head_")
        and "Wrist" not in name
        else 0.0
    )
    return name, neutral + axis["direction"] * (
        target_tick - axis["reference_tick"]
    ) * 2 * math.pi / 4096


def simulation_ticks(status: dict, angles: dict, *, check_range=True) -> dict:
    """Inverse of the established renderer mapping, with no target clamping."""
    native_keys, _ = executor_mappings()
    rows = {axis["key"]: axis for axis in status["axes"]}
    if set(rows) != set(native_keys):
        raise ValueError("Native motor mapping must include all fourteen position axes")
    targets = {}
    for joint, axis in rows.items():
        name, neutral = native_sim_target(status, joint, axis["reference_tick"])
        angle = angles.get(name)
        if type(angle) not in (int, float) or not math.isfinite(angle):
            raise ValueError(name + ": 仿真关节反馈不可用")
        tick = round(
            axis["reference_tick"]
            + axis["direction"] * (angle - neutral) * 4096 / (2 * math.pi)
        )
        if check_range and not axis["min_tick"] <= tick <= axis["max_tick"]:
            raise ValueError(
                f"{name}: 仿真姿态映射为 {tick} tick，超出真机范围 {axis['min_tick']}–{axis['max_tick']}"
            )
        targets[joint] = tick
    return targets
