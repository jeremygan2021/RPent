# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline measured-feedback mapping, ownership and HTTP boundary regressions."""

import json
import math
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robots.xlerobot import control, mapping, service

# The existing renderer's native motor identity and authored neutral contract.
# Fixtures are self-contained: these tests never import/open a serial controller.
JOINT_PAIRS = (
    ("shoulder_pan", "Rotation"),
    ("shoulder_lift", "Pitch"),
    ("elbow_flex", "Elbow"),
    ("wrist_flex", "Wrist_Pitch"),
    ("wrist_roll", "Wrist_Roll"),
    ("gripper", "Jaw"),
)
NATIVE_KEYS = {
    **{
        f"{arm}.{motor}": f"{side}_arm_{motor}"
        for arm, side in (("arm_2", "left"), ("arm_1", "right"))
        for motor, _ in JOINT_PAIRS
    },
    "head.head_motor_1": "head_motor_1",
    "head.head_motor_2": "head_motor_2",
}
MOTORS = {
    **{
        f"{name}_{suffix}": f"{side}_arm_{motor}"
        for name, side in (("Left", "left"), ("Right", "right"))
        for motor, suffix in JOINT_PAIRS
    },
    "Head_Yaw": "head_motor_1",
    "Head_Pitch": "head_motor_2",
}
NEUTRALS = {
    name: math.pi / 8
    if name.endswith("_Jaw")
    else math.pi / 2
    if name in ("Left_Pitch", "Right_Pitch", "Left_Elbow", "Right_Elbow")
    else 0.0
    for name in MOTORS
}


@pytest.fixture
def snapshots(monkeypatch):
    monkeypatch.setattr(control, "executor_mappings", lambda: (NATIVE_KEYS, MOTORS))
    monkeypatch.setattr(
        mapping,
        "time",
        SimpleNamespace(
            time=lambda: 100.0, monotonic=lambda: 0.0, sleep=lambda _: None
        ),
    )
    native = {
        "instance_id": "native-owner",
        "generation": 7,
        "fresh": True,
        "held": True,
        "mapping_control": {"supported": True, "active": False},
        "axes": [
            {
                "key": key,
                "name": name,
                "reference_tick": 2048,
                "raw_tick": 2048,
                "min_tick": 1500,
                "max_tick": 2600,
                "direction": -1 if index % 2 else 1,
                "direction_confirmed_by_operator": True,
                "torque_enabled": 1,
                "status": 0,
                "fresh": True,
                "updated_unix": 100.0,
            }
            for index, (key, name) in enumerate(NATIVE_KEYS.items())
        ],
        "base_control": {
            "armed": True,
            "fresh": True,
            "fault": None,
            "wheels": [{"torque": 1, "status": 0, "fresh": True}] * 2,
        },
    }
    simulation = {
        "ready": True,
        "updated_unix": 100.0,
        "active_command": None,
        "joints_rad": {**NEUTRALS, "Left_Wheel": 30.0, "Right_Wheel": -20.0},
        "base_pose_world": {
            "position_m": [1.0, 2.0, 0.25],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    return native, simulation


class FakeNative:
    """Record facade requests; no networking or motor dependency exists."""

    def __init__(self, status):
        self.snapshot = deepcopy(status)
        self.calls = []

    def status(self):
        return deepcopy(self.snapshot)

    def call(self, path, body):
        self.calls.append((path, deepcopy(body)))
        if path == "mapping/begin":
            self.snapshot["mapping_control"]["active"] = True
            return {"lease": "owner-existing-lease"}
        if path == "mapping/stop":
            self.snapshot["mapping_control"]["active"] = False
            return {"stop_confirmed": True}
        return {"ok": True}


def make_mapping(snapshots):
    native, simulation = snapshots
    owner = FakeNative(native)
    faults = []
    bridge = mapping.SimulationMapping(
        owner, lambda: deepcopy(simulation), lambda: faults.append("stop")
    )
    return bridge, owner, simulation, faults


def capture_thread(monkeypatch):
    threads = []

    class CapturedThread:
        def __init__(self, target, args, daemon):
            self.target, self.args, self.daemon = target, args, daemon
            self.started = False
            threads.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(
        mapping,
        "threading",
        SimpleNamespace(RLock=threading.RLock, Thread=CapturedThread),
    )
    return threads


def make_config(tmp_path):
    token = tmp_path / "token"
    token.write_text("offline-control-token")
    return {
        "token_file": str(token),
        "native_token_file": str(token),
        "console_key_file": str(token),
        "output_dir": str(tmp_path / "runs"),
    }


def test_inverse_uses_existing_render_neutrals_and_directions(snapshots):
    native, simulation = snapshots
    assert control.simulation_ticks(native, simulation["joints_rad"]) == dict.fromkeys(
        NATIVE_KEYS, 2048
    )
    angles = {}
    expected = {}
    for index, axis in enumerate(native["axes"]):
        tick = 2001 + index * 7
        name, angle = control.native_sim_target(native, axis["key"], tick)
        assert angle == pytest.approx(
            NEUTRALS[name] + axis["direction"] * (tick - 2048) * 2 * math.pi / 4096
        )
        angles[name], expected[axis["key"]] = angle, tick
    assert control.simulation_ticks(native, angles) == expected
    assert not any("Wheel" in key for key in expected)


@pytest.mark.parametrize("bad_angle", [None, True, "0.0", float("nan"), float("inf")])
def test_inverse_rejects_invalid_feedback(snapshots, bad_angle):
    native, simulation = snapshots
    simulation["joints_rad"]["Head_Yaw"] = bad_angle
    with pytest.raises(ValueError, match="反馈"):
        control.simulation_ticks(native, simulation["joints_rad"])


def test_inverse_rejects_bounds_instead_of_clamping(snapshots):
    native, simulation = snapshots
    axis = native["axes"][0]
    name = next(name for name, motor in MOTORS.items() if motor == axis["name"])
    simulation["joints_rad"][name] = (
        NEUTRALS[name] + axis["direction"] * (2601 - 2048) * 2 * math.pi / 4096
    )
    with pytest.raises(ValueError, match="2601 tick"):
        control.simulation_ticks(native, simulation["joints_rad"])
    assert (
        control.simulation_ticks(native, simulation["joints_rad"], check_range=False)[
            axis["key"]
        ]
        == 2601
    )


@pytest.mark.parametrize("change", ["missing", "unconfirmed", "invalid_direction"])
def test_inverse_rejects_missing_mapping_and_unknown_direction(snapshots, change):
    native, simulation = snapshots
    if change == "missing":
        native["axes"].pop()
    elif change == "unconfirmed":
        native["axes"][0]["direction_confirmed_by_operator"] = False
    else:
        native["axes"][0]["direction"] = 0
    with pytest.raises(ValueError):
        control.simulation_ticks(native, simulation["joints_rad"])


@pytest.mark.parametrize(
    "failure",
    [
        "stale",
        "held",
        "unarmed",
        "unsupported",
        "wheel_stale",
        "wheel_count",
        "wheel_torque",
        "axis_torque",
        "axis_status",
        "parking",
    ],
)
def test_enable_checks_native_health_before_any_write(snapshots, failure):
    bridge, native, _, _ = make_mapping(snapshots)
    state = native.snapshot
    if failure == "stale":
        state["fresh"] = False
    elif failure == "held":
        state["held"] = False
    elif failure == "unarmed":
        state["base_control"]["armed"] = False
    elif failure == "unsupported":
        state["mapping_control"]["supported"] = False
    elif failure == "wheel_stale":
        state["base_control"]["fresh"] = False
    elif failure == "wheel_count":
        state["base_control"]["wheels"].pop()
    elif failure == "wheel_torque":
        state["base_control"]["wheels"][0]["torque"] = 0
    elif failure == "axis_torque":
        state["axes"][0]["torque_enabled"] = 0
    elif failure == "axis_status":
        state["axes"][0]["status"] = 1
    else:
        state["parking"] = {"active": True}
    with pytest.raises(ValueError):
        bridge.enable()
    assert bridge.enabled is False
    assert native.calls == []


@pytest.mark.parametrize("failure", ["flag", "timestamp"])
def test_individual_axis_stale_rejects_mapping(snapshots, failure):
    bridge, native, _, _ = make_mapping(snapshots)
    native.snapshot["axes"][0]["fresh" if failure == "flag" else "updated_unix"] = (
        False if failure == "flag" else 99.0
    )
    with pytest.raises(ValueError):
        bridge.preflight()
    assert native.calls == []


@pytest.mark.parametrize("difference,accepted", [(35, True), (36, False)])
def test_initial_alignment_uses_35_tick_limit(snapshots, difference, accepted):
    bridge, native, _, _ = make_mapping(snapshots)
    native.snapshot["axes"][0]["raw_tick"] += difference
    if accepted:
        bridge.preflight()
    else:
        with pytest.raises(ValueError, match="35 tick"):
            bridge.enable()
    assert native.calls == []


def test_off_reads_feedback_without_native_motion_writes(snapshots):
    bridge, native, _, _ = make_mapping(snapshots)
    assert bridge.status()["enabled"] is False
    assert bridge.disable()["automatic_resume"] is False
    assert native.calls == []


def pose_frame(simulation, *, stamp, x, y, yaw, active=True):
    state = deepcopy(simulation)
    state.update(
        updated_unix=stamp, active_command={"command": "base"} if active else None
    )
    state["base_pose_world"] = {
        "position_m": [x, y, 0.25],
        "quaternion_wxyz": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
    }
    return state


def test_measured_base_uses_world_pose_wall_timing_and_owner_degrees(snapshots):
    _, simulation = snapshots
    before = pose_frame(simulation, stamp=99.9, x=1.0, y=2.0, yaw=math.pi / 2)
    after = pose_frame(simulation, stamp=100.0, x=1.0, y=2.003, yaw=math.pi / 2 + 0.01)
    # Simulated time and wheel angles are intentionally unrelated to wall motion.
    after.update(
        simulation_time=500.0,
        velocities_rad_s={"Left_Wheel": -123.0, "Right_Wheel": 321.0},
    )
    action, _ = mapping.measured_base(after, mapping.source_pose(before))
    assert action["x.vel"] == pytest.approx(0.03 * math.cos(0.01))
    assert action["theta.vel"] == pytest.approx(math.degrees(0.1))
    after["active_command"] = None
    assert mapping.measured_base(after, mapping.source_pose(before))[0] == {
        "x.vel": 0.0,
        "theta.vel": 0.0,
    }


def test_yaw_wrap_follows_shortest_rotation(snapshots):
    _, simulation = snapshots
    before = pose_frame(simulation, stamp=99.9, x=1.0, y=2.0, yaw=math.pi - 0.01)
    after = pose_frame(simulation, stamp=100.0, x=1.0, y=2.0, yaw=-math.pi + 0.01)
    assert mapping.measured_base(after, mapping.source_pose(before))[0][
        "theta.vel"
    ] == pytest.approx(math.degrees(0.2))


@pytest.mark.parametrize("failure", ["stale", "not_ready", "nan_pose", "gap"])
def test_measured_base_rejects_invalid_or_interrupted_source(snapshots, failure):
    _, state = snapshots
    previous = mapping.source_pose(state)
    state["active_command"] = {"command": "base"}
    if failure == "stale":
        state["updated_unix"] = 99.0
    elif failure == "not_ready":
        state["ready"] = False
    elif failure == "nan_pose":
        state["base_pose_world"]["position_m"][0] = float("nan")
    else:
        previous = (99.5, [1.0, 2.0, 0.25], 0.0)
    with pytest.raises(ValueError):
        mapping.measured_base(state, previous)


def test_enable_frame_reuses_owner_lease_with_all_axes(monkeypatch, snapshots):
    bridge, native, initial, _ = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    assert bridge.enable()["enabled"] is True
    assert threads[0].started is True
    started = native.calls[0][1]
    assert native.calls[0][0] == "mapping/begin"
    assert started["ticks"] == dict.fromkeys(NATIVE_KEYS, 2048)
    assert started["operator_ready"] is True
    moved = pose_frame(initial, stamp=100.1, x=1.003, y=2.0, yaw=0.01)
    bridge.read_simulation = lambda: deepcopy(moved)
    monkeypatch.setattr(mapping.time, "sleep", lambda _: bridge.disable())
    threads[0].target(*threads[0].args)
    paths = [path for path, _ in native.calls]
    assert paths == ["mapping/begin", "mapping/frame", "mapping/stop"]
    frame = native.calls[1][1]
    assert frame["id"] == started["id"]
    assert frame["lease"] == "owner-existing-lease"
    assert frame["instance_id"] == "native-owner" and frame["generation"] == 7
    assert frame["seq"] == 0 and len(frame["ticks"]) == 14
    assert frame["action"]["x.vel"] == pytest.approx(0.03 * math.cos(0.01))
    assert frame["action"]["theta.vel"] == pytest.approx(math.degrees(0.1))
    assert bridge.enabled is False and bridge.stop_confirmed is True


def test_disconnection_stops_both_sides_without_auto_recovery(monkeypatch, snapshots):
    bridge, native, initial, faults = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    bridge.enable()
    native.snapshot["fresh"] = False
    bridge.read_simulation = lambda: deepcopy(initial)
    threads[0].target(*threads[0].args)
    assert [path for path, _ in native.calls] == ["mapping/begin", "mapping/stop"]
    assert faults == ["stop"] and bridge.enabled is False and bridge.state == "error"
    native.snapshot["fresh"] = True
    assert bridge.status()["ready"] is True
    assert bridge.status()["enabled"] is False
    assert len(threads) == 1


def test_stop_preempts_begin_reply_before_any_frame(monkeypatch, snapshots):
    bridge, native, _, _ = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    original = native.call

    def intercepted(path, body):
        result = original(path, body)
        if path == "mapping/begin":
            bridge.disable()
        return result

    native.call = intercepted
    with pytest.raises(ValueError, match="停止撤销"):
        bridge.enable()
    assert bridge.enabled is False
    assert "mapping/frame" not in [path for path, _ in native.calls]
    assert native.snapshot["mapping_control"]["active"] is False
    assert threads == []


def test_stop_preempts_pending_follow_frame(monkeypatch, snapshots):
    bridge, native, initial, _ = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    bridge.enable()
    original = native.status
    stopped = False

    def status_then_stop():
        nonlocal stopped
        result = original()
        if not stopped:
            stopped = True
            bridge.disable()
        return result

    native.status = status_then_stop
    threads[0].target(*threads[0].args)
    assert "mapping/frame" not in [path for path, _ in native.calls]
    assert bridge.enabled is False


def test_old_follow_cannot_borrow_new_session_or_sequence(monkeypatch, snapshots):
    """A stopped follower paused after its epoch check keeps the revoked ID."""
    bridge, native, initial, _ = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    bridge.enable()
    old_id = bridge.session["id"]
    moved = pose_frame(initial, stamp=100.1, x=1.003, y=2.0, yaw=0.0)
    bridge.read_simulation = lambda: deepcopy(moved)
    entered, resume = threading.Event(), threading.Event()

    class NativeAtCallBoundary(FakeNative):
        def __getattribute__(self, name):
            # Bound-method lookup occurs after the follower's final epoch
            # check, before the frame's keyword payload is constructed.
            if name == "call" and threading.current_thread().name == "old-follow":
                entered.set()
                assert resume.wait(2), "Timed out releasing the old mock follower"
            return super().__getattribute__(name)

    native.__class__ = NativeAtCallBoundary
    worker = threading.Thread(
        target=threads[0].target, args=threads[0].args, name="old-follow"
    )
    worker.start()
    try:
        assert entered.wait(2)
        bridge.disable()
        bridge.read_simulation = lambda: deepcopy(initial)
        bridge.enable()
        new_id = bridge.session["id"]
        assert new_id != old_id
        resume.set()
        worker.join(2)
        assert not worker.is_alive()
        frames = [body for path, body in native.calls if path == "mapping/frame"]
        assert all(frame["id"] == old_id for frame in frames)
        assert all(frame["id"] != new_id for frame in frames)
        assert bridge.enabled is True and bridge.session["id"] == new_id
        assert bridge.frames == 0, "An old acknowledgement cannot advance the new session"
    finally:
        resume.set()
        worker.join(2)


def test_pending_disable_blocks_new_enable_until_stop_reply(monkeypatch, snapshots):
    """An old STOP response cannot clear or stop a newly acquired mapping."""
    bridge, native, _, _ = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    bridge.enable()
    old_id = bridge.session["id"]
    original = native.call
    entered, resume = threading.Event(), threading.Event()
    errors = []

    def delayed_stop(path, body):
        result = original(path, body)
        if path == "mapping/stop":
            entered.set()
            assert resume.wait(2), "Timed out releasing the mock STOP reply"
        return result

    def disable():
        try:
            bridge.disable()
        except Exception as exc:
            errors.append(exc)

    native.call = delayed_stop
    worker = threading.Thread(target=disable)
    worker.start()
    try:
        assert entered.wait(2)
        assert bridge.enabled is False and bridge.state == "stopping"
        with pytest.raises(ValueError, match="开启或停止"):
            bridge.enable()
        assert len(threads) == 1
        resume.set()
        worker.join(2)
        assert not worker.is_alive() and errors == []
        assert bridge.state == "disabled" and bridge.session is None
        bridge.enable()
        assert bridge.enabled is True and bridge.session["id"] != old_id
        assert bridge.state == "following" and bridge.stop_confirmed is False
        assert len(threads) == 2
    finally:
        resume.set()
        worker.join(2)


def test_follow_start_with_stale_source_revokes_both_sides(monkeypatch, snapshots):
    """A slow begin reply must not strand an enabled but dead follower."""
    bridge, native, _, faults = make_mapping(snapshots)
    threads = capture_thread(monkeypatch)
    bridge.enable()
    # Simulate source ageing while the native begin response was pending.
    threads[0].args[1]["updated_unix"] = 99.0
    bridge.read_simulation = lambda: deepcopy(threads[0].args[1])
    threads[0].target(*threads[0].args)
    assert bridge.enabled is False and bridge.state == "error"
    assert faults == ["stop"]
    assert [path for path, _ in native.calls] == ["mapping/begin", "mapping/stop"]


def test_service_blocks_direct_hardware_while_mapping_owns_it(tmp_path):
    engine = service.Engine(make_config(tmp_path))
    engine.mapping.enabled = True
    with pytest.raises(ValueError, match="控制权"):
        engine.start(
            service.Task(
                backend="hardware", instruction="move", request_id="offline-task"
            )
        )
    with pytest.raises(ValueError, match="直接真机"):
        engine.action(
            service.Action(backend="hardware", command="base", observation_id="unused"),
            "",
        )
    assert engine.process is None and engine.last_action is None


def test_service_stop_revokes_mapping_and_simulation(monkeypatch, tmp_path, snapshots):
    engine = service.Engine(make_config(tmp_path))
    bridge, native, _, _ = make_mapping(snapshots)
    bridge.enabled, bridge.state, bridge.session = True, "following", {"lease": "lease"}
    engine.mapping = bridge
    engine.observations["old"] = ("simulation", 0, {})
    simulation = []
    monkeypatch.setattr(
        service, "sim_call", lambda command: simulation.append(command) or {"ok": True}
    )
    epoch = engine.stop_epoch
    engine.control_stop("simulation")
    assert simulation == ["stop"] and native.calls == [
        ("mapping/stop", {"operator_ready": True})
    ]
    assert not engine.observations and engine.stop_epoch == epoch + 1
    assert engine.stop_confirmed is True and bridge.enabled is False


def test_restart_ignores_saved_mapping_enable(tmp_path):
    config = make_config(tmp_path)
    engine = service.Engine(config)
    engine.save_history()
    history = engine.output / "service-history.json"
    stored = json.loads(history.read_text())
    stored["mapping"] = {"enabled": True, "state": "following", "lease": "old"}
    history.write_text(json.dumps(stored))
    restored = service.Engine(config)
    assert restored.mapping.enabled is False
    assert restored.mapping.state == "disabled" and restored.mapping.session is None
    assert restored.mapping.thread is None


def test_manual_off_moves_simulation_without_native_write(
    monkeypatch, tmp_path, snapshots
):
    engine = service.Engine(make_config(tmp_path))
    bridge, native, simulation, _ = make_mapping(snapshots)
    engine.native, engine.mapping = native, bridge
    calls = []

    def sim_call(command, **kwargs):
        calls.append((command, kwargs))
        if command == "observe":
            return {
                "observation_id": "offline-observation",
                "state": simulation,
                "images": {},
            }
        return {"ok": True, "state": simulation}

    monkeypatch.setattr(service, "sim_call", sim_call)
    result = engine.manual(service.ManualBody(command="base", v=0.03, stop_epoch=0))
    assert result["source"] == "simulation" and result["stop_epoch"] == 0
    assert [command for command, _ in calls] == ["observe", "base"]
    assert calls[-1][1]["v"] == 0.03 and native.calls == []


def test_manual_status_exposes_stop_epoch(monkeypatch, tmp_path, snapshots):
    engine = service.Engine(make_config(tmp_path))
    bridge, native, simulation, _ = make_mapping(snapshots)
    engine.native, engine.mapping = native, bridge
    engine.stop_epoch = 4
    monkeypatch.setattr(service, "sim_call", lambda _: {"state": simulation})
    result = engine.manual_status()
    assert result["source"] == "simulation" and result["stop_epoch"] == 4
    assert len(result["axes"]) == 14
    assert all(axis["raw_tick"] == 2048 for axis in result["axes"])
    assert native.calls == []


@pytest.mark.parametrize("during_observation", [False, True])
def test_manual_stop_epoch_revokes_old_held_input(
    monkeypatch, tmp_path, snapshots, during_observation
):
    engine = service.Engine(make_config(tmp_path))
    _, simulation = snapshots
    actions = []
    if not during_observation:
        engine.stop_epoch = 1

    def observe(_):
        if not during_observation:
            pytest.fail("A revoked manual request must not read cameras")
        engine.stop_epoch += 1
        return {"observation_id": "new-after-stop", "state": simulation}

    monkeypatch.setattr(engine, "observe", observe)
    monkeypatch.setattr(engine, "action", lambda *args: actions.append(args))
    with pytest.raises(ValueError):
        engine.manual(service.ManualBody(command="base", v=0.03, stop_epoch=0))
    assert actions == []


@pytest.fixture
def api(tmp_path, monkeypatch):
    app = service.create_app(make_config(tmp_path))
    calls = []
    engine = app.state.engine
    monkeypatch.setattr(
        engine, "mapping_status", lambda: {"enabled": False, "ready": False}
    )
    monkeypatch.setattr(
        engine,
        "manual_status",
        lambda: {"source": "simulation", "ready": True, "stop_epoch": 0},
    )
    monkeypatch.setattr(
        engine,
        "set_mapping",
        lambda enabled: calls.append(("mapping", enabled)) or {"enabled": enabled},
    )
    monkeypatch.setattr(
        engine,
        "manual",
        lambda body: (
            calls.append(("manual", body.command))
            or {"source": "simulation", "stop_epoch": body.stop_epoch}
        ),
    )
    return TestClient(app), {"Authorization": "Bearer offline-control-token"}, calls


@pytest.mark.parametrize(
    "path,method",
    [
        ("mapping", "get"),
        ("mapping", "post"),
        ("manual/status", "get"),
        ("manual", "post"),
    ],
)
def test_mapping_and_manual_require_auth(api, path, method):
    client, _, calls = api
    kwargs = (
        {"json": {"enabled": True} if path == "mapping" else {"command": "base"}}
        if method == "post"
        else {}
    )
    assert getattr(client, method)("/v1/rpent/" + path, **kwargs).status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"enabled": "true"},
        {"enabled": 1},
        {"enabled": None},
        {"enabled": True, "backend": "hardware"},
    ],
)
def test_mapping_malformed_body_never_calls_owner(api, body):
    client, auth, calls = api
    assert client.post("/v1/rpent/mapping", headers=auth, json=body).status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"command": "unknown", "stop_epoch": 0},
        {"command": "base", "v": 0.151, "stop_epoch": 0},
        {"command": "base", "yaw_rate": 0.401, "stop_epoch": 0},
        {"command": "base", "duration": 0.01, "stop_epoch": 0},
        {"command": "native", "target_tick": True, "stop_epoch": 0},
        {"command": "base", "backend": "hardware", "stop_epoch": 0},
        {"command": "base"},
        {"command": "base", "stop_epoch": -1},
        {"command": "base", "stop_epoch": True},
        {"command": "base", "stop_epoch": "0"},
        {"command": "base", "stop_epoch": 0.5},
    ],
)
def test_manual_malformed_body_never_calls_owner(api, body):
    client, auth, calls = api
    assert client.post("/v1/rpent/manual", headers=auth, json=body).status_code == 422
    assert calls == []


def test_mapping_and_manual_http_response_contract(api):
    client, auth, calls = api
    assert client.get("/v1/rpent/mapping", headers=auth).json() == {
        "mapping": {"enabled": False, "ready": False}
    }
    assert client.post(
        "/v1/rpent/mapping", headers=auth, json={"enabled": True}
    ).json() == {"mapping": {"enabled": True}}
    assert (
        client.get("/v1/rpent/manual/status", headers=auth).json()["source"]
        == "simulation"
    )
    assert (
        client.post(
            "/v1/rpent/manual",
            headers=auth,
            json={"command": "base", "v": 0.03, "stop_epoch": 0},
        ).json()["source"]
        == "simulation"
    )
    assert calls == [("mapping", True), ("manual", "base")]


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_manual_schema_rejects_nonfinite_base(value):
    with pytest.raises(ValidationError):
        service.ManualBody(command="base", v=value, stop_epoch=0)
