# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""RPent tools over existing bounded XLeRobot APIs."""

import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from robots.xlerobot.client import ControlClient
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit, readonly


def schema(name, description, properties=None, required=()):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": list(required),
            "additionalProperties": False,
        },
    }


class XLeRobotToolkit(Toolkit):
    def _register_common_tools(self):
        # Robot operation exposes only structured robot tools. The planner's
        # task workspace remains available through its OS-isolated runtime.
        return

    def __init__(self, *, backend, output_dir, dashboard_events, memory, endpoint=None):
        self.backend = backend
        self.client = ControlClient(endpoint)
        self.finished = False
        self.verified = False
        super().__init__(
            dashboard_events=dashboard_events,
            memory=memory,
            state=EnvState(Path(output_dir)),
        )
        self._state.reset()
        self.add_tool(
            "observe_robot",
            schema(
                "observe_robot",
                "Capture fresh robot cameras and actual joint feedback.",
            ),
            self.observe,
        )
        self.add_tool(
            "robot_status",
            schema(
                "robot_status",
                "Read actual robot readiness and existing native motor mapping.",
            ),
            self.status,
        )
        self.add_tool(
            "move_base",
            schema(
                "move_base",
                "Bounded differential drive; v m/s, yaw_rate rad/s. Hardware borrows the existing base lease and requires prior whole-robot hold.",
                {
                    "v": {"type": "number"},
                    "yaw_rate": {"type": "number"},
                    "duration": {"type": "number"},
                    "observation_id": {"type": "string"},
                },
                ("v", "yaw_rate", "duration", "observation_id"),
            ),
            self.base,
        )
        self.add_tool(
            "move_joints",
            schema(
                "move_joints",
                "Simulation only: absolute USD joint angles in radians. Existing simulator enforces limits and rate.",
                {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "duration": {"type": "number"},
                    "observation_id": {"type": "string"},
                },
                ("targets", "duration", "observation_id"),
            ),
            self.joints,
        )
        self.add_tool(
            "move_native_joint",
            schema(
                "move_native_joint",
                "Use the existing encoder mapping and measured limits. Hardware requires prior explicit pose hold.",
                {
                    "joint": {"type": "string"},
                    "target_tick": {"type": "integer"},
                    "duration": {"type": "number"},
                    "observation_id": {"type": "string"},
                },
                ("joint", "target_tick", "duration", "observation_id"),
            ),
            self.native,
        )
        self.add_tool(
            "pi05_predict",
            schema(
                "pi05_predict",
                "Run GPU1 pi05_base inference on live simulation cameras; saves raw actions, does not execute uncalibrated outputs.",
                {"prompt": {"type": "string"}},
                ("prompt",),
            ),
            self.infer,
        )
        self.add_tool(
            "stop_robot",
            schema(
                "stop_robot",
                "Revoke further task motion, stop and hold current pose. Observation, task_status and finish remain available.",
            ),
            self.stop,
        )
        self.add_tool(
            "finish",
            schema(
                "finish",
                "End the task with an evidence-based result.",
                {"summary": {"type": "string"}},
                ("summary",),
            ),
            self.finish,
        )
        if backend == "hardware":
            self._tools.pop("move_joints")
            self._tools.pop("pi05_predict")
        self.add_tool(
            "locate_object",
            schema(
                "locate_object",
                "Locate a registered empty pitcher and its handle using trusted tracking and calibrated fresh RGB-D.",
                {"object_id": {"type": "string"}},
                ("object_id",),
            ),
            self.locate_object,
        )
        self.add_tool(
            "navigate_to_workspace",
            schema(
                "navigate_to_workspace",
                "Navigate to a registered workspace; query task_status until actual arrival and stop are confirmed.",
                {"workspace_id": {"type": "string"}},
                ("workspace_id",),
            ),
            self.navigate_to_workspace,
        )
        self.add_tool(
            "grasp_object",
            schema(
                "grasp_object",
                "Execute a calibrated, collision-checked right-arm handle grasp. Requires the trusted locate result.",
                {
                    "object_id": {"type": "string"},
                    "observation_id": {"type": "string"},
                    "calibration_version": {"type": "string"},
                },
                ("object_id", "observation_id", "calibration_version"),
            ),
            self.grasp_object,
        )
        self.add_tool(
            "verify_grasp",
            schema(
                "verify_grasp",
                "Independently verify fresh lift, stable holding and gripper evidence after grasp execution.",
                {"executor_id": {"type": "string"}},
                ("executor_id",),
            ),
            self.verify_grasp,
        )
        self.add_tool(
            "task_status",
            schema(
                "task_status",
                "Read this task, including original asynchronous skill IDs and outcomes.",
            ),
            self.task_status,
        )

    @readonly
    def status(self):
        return self.client.call("status?backend=" + self.backend)

    @readonly
    def observe(self):
        value = self.client.call("observe", {"backend": self.backend})
        value.update(self.image_bytes(value))
        return value

    def image_bytes(self, observation):
        result = {}
        for keys, output in (
            (["head"], "_image_bytes"),
            (["left_wrist", "hand_left"], "_image_cam_bytes"),
            (["right_wrist", "hand_right"], "_image_wrist_bytes"),
        ):
            name = next(
                (name for name in keys if name in observation.get("images", {})), None
            )
            if name:
                result[output] = self.client.call(
                    "camera/" + observation["observation_id"] + "/" + name, binary=True
                )
        return result

    @readonly
    def task_status(self):
        if not self.client.session:
            raise ValueError("Start a managed task before using robot skills")
        return self.client.call("tasks/" + self.client.session)

    def skill(self, operation, parameters):
        self.raise_if_cancelled()
        if not self.client.session:
            raise ValueError("Start a managed task before using robot skills")
        return self.client.call(
            "skills",
            {
                "backend": self.backend,
                "task_id": self.client.session,
                "request_id": uuid.uuid4().hex,
                "operation": operation,
                "parameters": parameters,
            },
        )

    @readonly
    def locate_object(self, object_id):
        return self.skill("locate_object", {"object_id": object_id})

    def navigate_to_workspace(self, workspace_id):
        return self.skill("navigate_to_workspace", {"workspace_id": workspace_id})

    def grasp_object(self, object_id, observation_id, calibration_version):
        return self.skill(
            "grasp_object",
            {
                "object_id": object_id,
                "side": "right",
                "observation_id": observation_id,
                "calibration_version": calibration_version,
            },
        )

    @readonly
    def verify_grasp(self, executor_id):
        return self.skill("verify_grasp", {"task_id": executor_id})

    def action(self, command, **parameters):
        self.raise_if_cancelled()
        return self.client.call(
            "action", {"backend": self.backend, "command": command, **parameters}
        )

    def base(self, v, yaw_rate, duration, observation_id):
        return self.action(
            "base",
            v=v,
            yaw_rate=yaw_rate,
            duration=duration,
            observation_id=observation_id,
        )

    def joints(self, targets, duration, observation_id):
        return self.action(
            "joints", targets=targets, duration=duration, observation_id=observation_id
        )

    def native(self, joint, target_tick, duration, observation_id):
        return self.action(
            "native",
            joint=joint,
            target_tick=target_tick,
            duration=duration,
            observation_id=observation_id,
        )

    @readonly
    def infer(self, prompt):
        if self.backend != "simulation":
            raise ValueError(
                "Pi0.5 observation adaptation currently requires simulation"
            )
        return self.client.call("infer/wait", {"prompt": prompt})

    def stop(self):
        return self.client.call("control/stop", {"backend": self.backend})

    def finish(self, summary):
        self.stop()
        state = self.task_status() if self.client.session else {}
        self.verified = (
            state.get("task_outcome") == "succeeded"
            and state.get("stop_state") == "stopped"
        )
        self.finished = True
        return {
            "_finish": True,
            "summary": summary,
            "task_success_verified": self.verified,
        }

    def get_env_state(self, *, command, result, elapsed_s):
        observation = self.client.call("observe", {"backend": self.backend})
        images = self.image_bytes(observation)
        with self._state.record_step(
            state=observation.get("state", {}),
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            self._state.save("observation.json", observation)
            for name, content in images.items():
                self._state.save(
                    name + ".png",
                    np.asarray(Image.open(BytesIO(content)).convert("RGB")),
                )
        return {**result, "observation": observation, **images}

    def solved(self):
        return self.verified

    def cancel_active_and_wait(self):
        self.stop()
        super().cancel_active_and_wait()

    def close(self):
        self.stop()
