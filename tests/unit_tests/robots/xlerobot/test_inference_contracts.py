# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline contracts for the inference-only XLeRobot bridge."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robots.xlerobot import diagnostic
from robots.xlerobot.observation import (
    ARM_JOINTS,
    CAMERAS,
    encode_observation,
    sim_request,
    validate_observation,
)
from robots.xlerobot.vla_server import XLeRobotPi05Facade
from rpent.robots.components.vla_facade_base import BaseVLAFacade


@pytest.fixture
def observation(tmp_path):
    images = {}
    for i, name in enumerate(CAMERAS):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((24, 32, 3), i * 80, np.uint8)).save(path)
        images[name] = {"path": str(path)}
    record = {
        "ok": True,
        "observation_id": "test-observation",
        "images": images,
        "state": {
            "mode": "simulation_only",
            "joints_rad": dict.fromkeys(ARM_JOINTS, 0.0),
        },
    }
    limits = dict.fromkeys(ARM_JOINTS, [-1, 1])
    description = {"joint_limits_rad": limits}
    return record, description


def test_real_joint_order_and_camera_identity_are_preserved(observation):
    record, description = observation
    record["state"]["joints_rad"]["Right_Elbow"] = 1
    record["state"]["joints_rad"]["Left_Jaw"] = -1
    obs = encode_observation(record, description, "hold")
    assert obs["state"][8] == 1
    assert obs["state"][5] == -1
    for index, key in enumerate(CAMERAS.values()):
        assert np.all(obs["image"][key] == index * 80)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_observation_is_rejected(observation, bad):
    record, description = observation
    record["state"]["joints_rad"]["Left_Rotation"] = bad
    with pytest.raises(ValueError, match="invalid state"):
        encode_observation(record, description, "hold")


@pytest.mark.parametrize("command", ["joints", "base", "gripper", "shutdown"])
def test_diagnostic_cannot_send_motion_or_destroy_simulator(tmp_path, command):
    with pytest.raises(ValueError, match="supports status, observe and stop only"):
        sim_request(tmp_path, command)


def test_rgb_and_state_validation(observation):
    record, description = observation
    obs = encode_observation(record, description, "hold")
    obs["image"]["base_0_rgb"] = np.zeros((8, 8, 4), np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        validate_observation(obs)


def test_model_result_follows_rpent_dispatch_without_executing(observation):
    record, description = observation
    obs = encode_observation(record, description, "hold")
    facade = XLeRobotPi05Facade.__new__(XLeRobotPi05Facade)
    BaseVLAFacade.__init__(facade)
    facade._policy = SimpleNamespace(infer=lambda data: {"actions": np.ones((50, 32))})
    result = facade._dispatch("vla.predict", (obs, None), {})
    assert result["actions"].shape == (50, 32)
    assert result["execution_enabled"] is False
    facade._policy = SimpleNamespace(
        infer=lambda data: {"actions": np.full((50, 32), np.nan)}
    )
    with pytest.raises(RuntimeError, match="invalid model output"):
        facade._dispatch("vla.predict", (obs, None), {})


def test_failed_inference_stops_simulation_and_does_not_publish_pass(
    monkeypatch, tmp_path, observation
):
    record, description = observation
    (tmp_path / "robot_description.json").write_text(json.dumps(description))
    commands = []

    def request(runtime, command, **kwargs):
        commands.append(command)
        return record if command == "observe" else {"ok": True}

    def call(method, *args, **kwargs):
        if method == "vla.metadata":
            return {"execution_enabled": False}
        raise RuntimeError("inference failed")

    monkeypatch.setattr(diagnostic, "sim_request", request)
    monkeypatch.setattr(
        diagnostic, "make_rpc_client", lambda endpoint: SimpleNamespace(call=call)
    )
    with pytest.raises(RuntimeError, match="inference failed"):
        diagnostic.run_diagnostic(
            runtime=tmp_path,
            endpoint="http://localhost:1",
            prompt="hold",
            output=tmp_path / "result",
            repeats=1,
        )
    assert commands == ["stop", "observe", "stop"]
    assert not (tmp_path / "result/report.json").exists()
