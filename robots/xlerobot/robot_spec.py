# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0


"""XLeRobot registration borrowing the independent control service."""

from datetime import datetime
from pathlib import Path

from rpent.memory import MemoryManager
from rpent.robots import PromptBundle, RobotSpec, RunConfig


def system_prompt(variables=None):
    return {
        "ROLE": "Control XLeRobot using structured tools and actual camera/encoder observations.",
        "RUNTIME": "The external service owns the selected hardware or simulation backend. Hardware works independently of Isaac, Pi0.5 and PPO. Use tools only; do not start services or access USB.",
        "WORKFLOW": "Observe cameras before motion and after each short action. Use the latest observation_id. Inspect error/actual feedback. Never repeat an ambiguous action.",
        "MODEL": "pi05_base outputs have no XLeRobot training normalization or calibrated action semantics. pi05_predict is diagnostic. Use explicit bounded movement tools for execution.",
        "COMPLETION": "Stop on completion or error. Call finish with measured evidence; do not claim grasp success from a write receipt.",
        "PICKUP": "Read robot_status capabilities. Locate the selected pitcher, navigate to a registered workspace, stop the base, locate again, grasp using the returned observation and calibration version, then independently verify_grasp. Query task_status for asynchronous skill completion. Reuse original task IDs on uncertain outcomes; never submit a new action to retry. Missing calibration, perception or collision readiness blocks that skill.",
    }


def user_prompt(variables=None):
    return {
        "TASK": (variables or {}).get(
            "instruction", "Observe the robot and report its state."
        )
    }


def add_args(parser, use_dashboard):
    parser.add_argument(
        "--backend", choices=("simulation", "hardware"), default="simulation"
    )
    parser.add_argument(
        "--instruction", default="Observe the robot and report its state."
    )
    parser.add_argument("--control-endpoint", default="http://127.0.0.1:8919")


def parse_config(args):
    tag = "xlerobot_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir or ("logs/xlerobot/" + tag))
    return RunConfig(
        recipe_tag=tag,
        output_dir=output,
        prompt_vars={"instruction": args.instruction},
        task_desc={"instruction": args.instruction, "backend": args.backend},
    )


def init_runtime(args, output_dir, dashboard_events, components):
    return [], {"backend": args.backend, "endpoint": args.control_endpoint}


def prepare_memory(args, config):
    config.prompt_vars["memory_dir"] = str(config.output_dir / "memory")


def get_robot_spec():
    return RobotSpec(
        name="xlerobot",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=add_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
        prepare_memory=prepare_memory,
    )


def get_toolkit(*, runtime_kwargs, dashboard_events, config, **kwargs):
    from robots.xlerobot.toolkit import XLeRobotToolkit

    return XLeRobotToolkit(
        **runtime_kwargs,
        output_dir=config.output_dir,
        dashboard_events=dashboard_events,
        memory=MemoryManager(root=config.prompt_vars["memory_dir"]),
    )
