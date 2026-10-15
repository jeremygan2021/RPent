# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Explicit, nonpersistent Isaac-to-native following on the existing serial owner."""

from __future__ import annotations

import math
import threading
import time
import uuid

from robots.xlerobot.control import simulation_ticks


def source_pose(state):
    stamp = state.get("updated_unix")
    if (
        state.get("ready") is not True
        or type(stamp) not in (int, float)
        or not -0.1 <= time.time() - stamp <= 0.35
    ):
        raise ValueError("Isaac 实测反馈未就绪或已经过期")
    pose = state["base_pose_world"]
    position, quaternion = pose["position_m"], pose["quaternion_wxyz"]
    if (
        len(position) != 3
        or len(quaternion) != 4
        or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in position + quaternion
        )
    ):
        raise ValueError("Isaac 实测底盘位姿无效")
    w, x, y, z = quaternion
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return stamp, position, yaw


def measured_base(state, previous):
    """Follow actual world displacement per wall second, preserving live timing.

    Wheels use the owner's original body-velocity conversion and signs. They
    never receive the simulator's accumulated wheel angles as position targets.
    """
    current = source_pose(state)
    action = {"x.vel": 0.0, "theta.vel": 0.0}
    if previous and state.get("active_command"):
        dt = current[0] - previous[0]
        if dt > 0:
            if dt > 0.35:
                raise ValueError("Isaac 移动反馈中断")
            dx, dy = current[1][0] - previous[1][0], current[1][1] - previous[1][1]
            velocity = (dx * math.cos(current[2]) + dy * math.sin(current[2])) / dt
            yaw_rate = (
                math.atan2(
                    math.sin(current[2] - previous[2]),
                    math.cos(current[2] - previous[2]),
                )
                / dt
            )
            action = {
                "x.vel": velocity if abs(velocity) >= 0.002 else 0.0,
                "theta.vel": math.degrees(yaw_rate) if abs(yaw_rate) >= 0.015 else 0.0,
            }
    return action, current


class SimulationMapping:
    def __init__(self, native, read_simulation, fault_stop):
        self.native, self.read_simulation, self.fault_stop = (
            native,
            read_simulation,
            fault_stop,
        )
        self.lock = threading.RLock()
        self.transition_lock = threading.RLock()
        self.enabled = False
        self.state = "disabled"
        self.last_error = ""
        self.stop_confirmed = True
        self.epoch = 0
        self.thread = None
        self.session = None
        self.frames = 0

    def preflight(self):
        simulation = self.read_simulation()
        source_pose(simulation)
        if simulation.get("active_command"):
            raise ValueError("请先停止仿真动作，再开启真机映射")
        native = self.native.status()
        if native.get("execution_ready") is False:
            raise ValueError("Network owner execution gates are not verified")
        base = native.get("base_control", {})
        if not native.get("fresh"):
            raise ValueError("真机电机未连接或反馈已过期；当前仅仿真")
        if native.get("mapping_control", {}).get("supported") is not True:
            raise ValueError("现有电机执行器尚未安装仿真映射接口")
        if not native.get("held") or not base.get("armed"):
            raise ValueError("请先在已有电机控制页面开启整机当前位置保持")
        if (
            base.get("fresh") is not True
            or base.get("fault")
            or native.get("parking", {}).get("active")
        ):
            raise ValueError("真机双轮反馈异常或正在停放")
        if (
            any(
                a.get("fresh") is not True
                or type(a.get("updated_unix")) not in (int, float)
                or not -0.1 <= time.time() - a["updated_unix"] <= 0.35
                or a.get("torque_enabled") != 1
                or a.get("status") != 0
                for a in native["axes"]
            )
            or len(base.get("wheels", [])) != 2
            or any(w.get("torque") != 1 or w.get("status") != 0 for w in base["wheels"])
        ):
            raise ValueError("请核对16路电机健康和当前位置保持")
        ticks = simulation_ticks(native, simulation["joints_rad"])
        for axis in native["axes"]:
            if (
                type(axis.get("raw_tick")) is not int
                or abs(ticks[axis["key"]] - axis["raw_tick"]) > 35
            ):
                raise ValueError(
                    axis["name"]
                    + ": 仿真与真机姿态未对齐（允许35 tick）；请先调整软件姿态"
                )
        return simulation, native, ticks

    def status(self):
        reason = []
        if not self.enabled:
            try:
                self.preflight()
            except Exception as exc:
                reason = [str(exc)]
        return {
            "enabled": self.enabled,
            "state": self.state,
            "ready": self.enabled or not reason,
            "blocked_reasons": reason,
            "last_error": self.last_error,
            "stop_confirmed": self.stop_confirmed,
            "frames": self.frames,
            "source": "Isaac measured joints and world motion",
            "channels": 16,
            "automatic_resume": False,
        }

    def enable(self):
        if not self.transition_lock.acquire(blocking=False):
            raise ValueError("映射正在开启或停止，请等待原操作结束")
        try:
            return self._enable()
        finally:
            self.transition_lock.release()

    def _enable(self):
        with self.lock:
            if self.enabled:
                return self.status()
            if self.state in ("starting", "stopping"):
                raise ValueError("映射正在开启或停止，请等待原操作结束")
            self.epoch += 1
            epoch = self.epoch
            self.state, self.last_error = "starting", ""
        try:
            simulation, native, ticks = self.preflight()
            session = {
                "id": uuid.uuid4().hex,
                "instance_id": native["instance_id"],
                "generation": native["generation"],
            }
            reply = self.native.call(
                "mapping/begin", {**session, "operator_ready": True, "ticks": ticks}
            )
            session["lease"] = reply["lease"]
            with self.lock:
                if epoch != self.epoch:
                    raise ValueError("映射开启已被停止撤销")
                self.session, self.enabled, self.state = session, True, "following"
                self.stop_confirmed, self.frames = False, 0
                self.thread = threading.Thread(
                    target=self.follow,
                    args=(epoch, simulation, native, dict(session)),
                    daemon=True,
                )
                self.thread.start()
            return self.status()
        except Exception as exc:
            with self.lock:
                if epoch == self.epoch:
                    self.enabled, self.state, self.last_error = False, "error", str(exc)
            # A begin reply may be lost after the owner accepted it. Always
            # revoke that attempt, never retry the nonzero frames.
            if "session" in locals():
                try:
                    reply = self.native.call("mapping/stop", {"operator_ready": True})
                    self.stop_confirmed = reply.get("stop_confirmed") is True
                except Exception:
                    self.stop_confirmed = False
            raise

    def follow(self, epoch, initial, expected, session=None):
        session = dict(session if session is not None else self.session)
        sequence = 0
        last_base = {"x.vel": 0.0, "theta.vel": 0.0}
        try:
            previous = source_pose(initial)
            while self.enabled and epoch == self.epoch:
                started = time.monotonic()
                simulation = self.read_simulation()
                ticks = simulation_ticks(expected, simulation["joints_rad"])
                action, current_pose = measured_base(simulation, previous)
                if simulation.get("active_command") and current_pose[0] == previous[0]:
                    action = last_base
                previous, last_base = current_pose, action
                current = self.native.status()
                if (
                    current.get("instance_id") != expected["instance_id"]
                    or current.get("generation") != expected["generation"]
                    or not current.get("fresh")
                    or not current.get("mapping_control", {}).get("active")
                ):
                    raise ValueError("真机映射租约、反馈或执行器已改变")
                if not self.enabled or epoch != self.epoch:
                    break
                self.native.call(
                    "mapping/frame",
                    {
                        **session,
                        "seq": sequence,
                        "ticks": ticks,
                        "action": action,
                    },
                )
                sequence += 1
                with self.lock:
                    if epoch == self.epoch:
                        self.frames = sequence
                time.sleep(max(0.005, 0.075 - (time.monotonic() - started)))
        except Exception as exc:
            if epoch == self.epoch:
                self.disable(str(exc))
                self.fault_stop()

    def disable(self, error=""):
        with self.lock:
            attempted = (
                self.enabled
                or self.state in ("starting", "stopping")
                or self.session is not None
            )
            self.epoch += 1
            epoch = self.epoch
            self.enabled = False
            self.state, self.last_error = "stopping", error
        # Begin and STOP cannot overlap a new mapping lease. The epoch above
        # preempts frames immediately, even while a begin response is pending.
        with self.transition_lock:
            confirmed, failure = self.stop_confirmed, error
            try:
                if attempted:
                    reply = self.native.call("mapping/stop", {"operator_ready": True})
                    confirmed = reply.get("stop_confirmed") is True
                    if not confirmed:
                        failure = "真机映射停止未确认"
            except Exception as exc:
                confirmed, failure = False, "真机映射停止未确认: " + str(exc)
            finally:
                with self.lock:
                    if epoch == self.epoch:
                        self.session = None
                        self.stop_confirmed, self.last_error = confirmed, failure
                        self.state = "error" if failure else "disabled"
        return self.status()
