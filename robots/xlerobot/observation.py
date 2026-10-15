# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Observation contract for the local XLeRobot simulator diagnostic."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path

import numpy as np

CAMERAS = {
    "head": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
ARM_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("Left", "Right")
    for joint in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
)
DEFAULT_RUNTIME = Path(__file__).resolve().parents[3] / "xlerobot_isaac/codex_runtime"


def sim_request(runtime: Path, command: str, **kwargs) -> dict:
    """Read the existing simulator using its local, bounded command protocol."""
    if command not in {"status", "observe", "stop"}:
        raise ValueError(
            "the base-model diagnostic supports status, observe and stop only"
        )
    payload = {"id": uuid.uuid4().hex, "command": command, **kwargs}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(45)
        connection.connect(str(runtime / "control.sock"))
        connection.sendall((json.dumps(payload, allow_nan=False) + "\n").encode())
        with connection.makefile("rb") as stream:
            result = json.loads(stream.readline(8 * 1024 * 1024))
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "simulator request failed"))
    return result


def encode_observation(record: dict, description: dict, prompt: str) -> dict:
    """Use real RGB/state inputs with an explicit diagnostic-only normalization.

    Joint limits map twelve arm angles to [-1, 1]. These are not dataset
    quantiles and do not establish a calibrated checkpoint action convention.
    """
    from PIL import Image

    if record["state"].get("mode") != "simulation_only":
        raise ValueError("only the local simulation is supported")
    states = []
    for joint in ARM_JOINTS:
        angle = float(record["state"]["joints_rad"][joint])
        low, high = description["joint_limits_rad"][joint]
        if not np.isfinite([angle, low, high]).all() or high <= low:
            raise ValueError(f"invalid state or limits for {joint}")
        states.append(float(np.clip(2 * (angle - low) / (high - low) - 1, -1, 1)))
    images = {}
    for camera, key in CAMERAS.items():
        with Image.open(record["images"][camera]["path"]) as im:
            images[key] = np.asarray(im.convert("RGB")).copy()
    return validate_observation(
        {
            "image": images,
            "image_mask": dict.fromkeys(CAMERAS.values(), True),
            "state": np.asarray(states, dtype=np.float32),
            "prompt": prompt,
        }
    )


def validate_observation(obs: dict) -> dict:
    """Validate one unbatched three-camera diagnostic observation at the RPC boundary."""
    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("prompt must contain 1 to 2000 characters")
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (12,) or not np.isfinite(state).all():
        raise ValueError("state must contain twelve finite normalized arm positions")
    if np.any(np.abs(state) > 1):
        raise ValueError("diagnostic state must be in [-1, 1]")
    images = {}
    for key in CAMERAS.values():
        arr = np.asarray(obs["image"][key])
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{key}: expected uint8 RGB [H,W,3]")
        if not (1 <= arr.shape[0] <= 2048 and 1 <= arr.shape[1] <= 2048):
            raise ValueError(f"{key}: invalid image dimensions")
        images[key] = arr
    return {
        "image": images,
        "image_mask": dict.fromkeys(CAMERAS.values(), True),
        "state": state,
        "prompt": prompt,
    }
