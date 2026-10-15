# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""Authenticated local RPent control-service client."""

import os
from pathlib import Path

import httpx


class ControlClient:
    def __init__(self, endpoint=None):
        self.endpoint = endpoint or os.environ.get(
            "RPENT_CONTROL_URL", "http://127.0.0.1:8919"
        )
        self.token = Path(os.environ["RPENT_CONTROL_TOKEN_FILE"]).read_text().strip()
        self.session = os.environ.get("RPENT_CONTROL_SESSION", "")

    def call(self, path, body=None, *, binary=False):
        with httpx.Client(trust_env=False, timeout=1200) as client:
            response = client.request(
                "GET" if body is None else "POST",
                self.endpoint + "/v1/rpent/" + path,
                headers={
                    "Authorization": "Bearer " + self.token,
                    "X-RPent-Session": self.session,
                },
                json=body,
            )
        if binary and response.is_success:
            return response.content
        value = response.json()
        if response.is_error:
            raise ValueError(str(value.get("detail", "RPent control request rejected")))
        return value
