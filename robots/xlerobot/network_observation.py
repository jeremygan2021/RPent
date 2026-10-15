# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Fetch registered controller cameras without moving motors or changing clocks."""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def observe_cameras(config, *, not_before=None):
    cameras = config["hardware_cameras"]
    if set(cameras) != {"head", "left_wrist", "right_wrist"}:
        raise ValueError("Register head and both wrist camera sources explicitly")

    def fetch(item):
        name, source = item
        token = Path(source["token_file"]).read_text().strip()
        with httpx.Client(trust_env=False, timeout=5, follow_redirects=False) as client:
            deadline = time.monotonic() + 1
            while True:
                result = client.get(
                    source["url"], headers={"Authorization": "Bearer " + token}
                )
                result.raise_for_status()
                stamp = float(result.headers["x-frame-unix"])
                if not_before is None or stamp >= not_before:
                    break
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "No post-action frame from registered camera: " + name
                    )
                time.sleep(0.05)
        result.raise_for_status()
        if (
            result.headers.get("x-camera-name") != source["source_id"]
            or result.headers.get("x-camera-source") != source["source_id"]
            or not result.content.startswith(b"\xff\xd8")
            or len(result.content) > 10 * 1024 * 1024
        ):
            raise ValueError("Registered hardware camera source mismatch: " + name)
        return name, {
            "content": result.content,
            "unix": stamp,
            "sequence": int(result.headers["x-frame-sequence"]),
            "source_id": source["source_id"],
        }

    with ThreadPoolExecutor(max_workers=3) as workers:
        images = dict(workers.map(fetch, cameras.items()))
    return {"id": uuid.uuid4().hex, "images": images}
