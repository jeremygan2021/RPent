# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Registered network-camera identity and post-action evidence contracts."""

import time
from io import BytesIO

import httpx
import pytest
from PIL import Image

from robots.xlerobot.network_observation import observe_cameras


def camera_config(tmp_path):
    token = tmp_path / "camera-token"
    token.write_text("test-camera-token")
    return {
        "hardware_cameras": {
            name: {
                "source_id": name,
                "url": "http://test/" + name,
                "token_file": str(token),
            }
            for name in ("head", "left_wrist", "right_wrist")
        }
    }


def install_client(monkeypatch, response):
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, **kwargs):
            return response(url.rsplit("/", 1)[-1])

    monkeypatch.setattr("robots.xlerobot.network_observation.httpx.Client", Client)


def frame(name, stamp, *, source=None):
    content = BytesIO()
    Image.new("RGB", (16, 16)).save(content, format="JPEG")
    return httpx.Response(
        200,
        request=httpx.Request("GET", "http://test/" + name),
        content=content.getvalue(),
        headers={
            "x-camera-name": name,
            "x-camera-source": source or name,
            "x-frame-unix": str(stamp),
            "x-frame-sequence": "1",
        },
    )


def test_registered_camera_source_is_preserved(tmp_path, monkeypatch):
    stamp = time.time()
    install_client(monkeypatch, lambda name: frame(name, stamp))
    value = observe_cameras(camera_config(tmp_path))
    assert {v["source_id"] for v in value["images"].values()} == {
        "head",
        "left_wrist",
        "right_wrist",
    }
    assert all(v["unix"] == stamp for v in value["images"].values())


def test_wrong_camera_source_is_rejected(tmp_path, monkeypatch):
    install_client(
        monkeypatch, lambda name: frame(name, time.time(), source="another-camera")
    )
    with pytest.raises(ValueError, match="source mismatch"):
        observe_cameras(camera_config(tmp_path))


def test_post_action_observation_waits_for_new_capture(tmp_path, monkeypatch):
    stamp = time.time()
    calls = {}

    def response(name):
        calls[name] = calls.get(name, 0) + 1
        return frame(name, stamp - 1 if calls[name] == 1 else stamp + 0.01)

    install_client(monkeypatch, response)
    value = observe_cameras(camera_config(tmp_path), not_before=stamp)
    assert all(v["unix"] > stamp for v in value["images"].values())
    assert set(calls.values()) == {2}
