# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Capture live XLeRobot cameras and verify Pi0.5 over RPent's VLA RPC."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from robots.xlerobot.observation import (
    CAMERAS,
    DEFAULT_RUNTIME,
    encode_observation,
    sim_request,
)
from rpent.robots.components.vla_client_base import BaseVLAClient
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.rpc import make_rpc_client

logger = get_logger("xlerobot.diagnostic")


class XLeRobotVLAClient(BaseVLAClient):
    """Allow the initial JAX compilation to complete before the RPC timeout."""

    _TIMEOUT_S = {**BaseVLAClient._TIMEOUT_S, "predict": 1200.0}


def run_diagnostic(
    *, runtime: Path, endpoint: str, prompt: str, output: Path, repeats: int = 2
) -> dict:
    """Save live observations, full action tensors, and synchronized timings."""
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be in [1, 10]")
    output.mkdir(parents=True, exist_ok=False)
    rpc = make_rpc_client(endpoint)
    client = XLeRobotVLAClient(rpc)
    metadata = rpc.call("vla.metadata", timeout_s=10)
    if metadata.get("execution_enabled") is not False:
        raise ValueError("expected an inference-only model server")
    description = json.loads((runtime / "robot_description.json").read_text())
    runs = []
    try:
        sim_request(runtime, "stop")
        for index in range(repeats):
            folder = output / f"sample_{index:02d}"
            folder.mkdir()
            record = sim_request(runtime, "observe", cameras=list(CAMERAS))
            (folder / "observation.json").write_text(json.dumps(record, indent=2))
            for camera in CAMERAS:
                shutil.copy2(record["images"][camera]["path"], folder / f"{camera}.png")
            obs = encode_observation(record, description, prompt)
            np.savez_compressed(
                folder / "model_input.npz", state=obs["state"], **obs["image"]
            )
            started = time.monotonic()
            result = client.predict(obs)
            elapsed = time.monotonic() - started
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (50, 32) or not np.isfinite(actions).all():
                raise RuntimeError("nonfinite or malformed actions received over RPC")
            np.save(folder / "actions.npy", actions)
            summary = {
                "observation_id": record["observation_id"],
                "prompt": prompt,
                "shape": list(actions.shape),
                "finite": True,
                "server_seconds": result["wall_seconds"],
                "rpc_seconds": elapsed,
                "min": float(actions.min()),
                "max": float(actions.max()),
                "mean": float(actions.mean()),
                "std": float(actions.std()),
                "actions_sent_to_simulator": 0,
            }
            (folder / "result.json").write_text(json.dumps(summary, indent=2))
            logger.info("Sample %d: %s", index, summary)
            runs.append(summary)
    finally:
        sim_request(runtime, "stop")
    report = {
        "status": "PASS",
        "scope": "live simulation observation + pi05_base inference",
        "model": metadata,
        "samples": runs,
        "task_success": None,
        "limitations": [
            "No XLeRobot fine-tuned checkpoint or dataset normalization",
            "No calibrated action mapping or autonomous task validation",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("Report: %s", output / "report.json")
    return report


def main() -> None:
    """Run the standalone diagnostic without an LLM planner or API key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8095")
    parser.add_argument("--prompt", default="Keep both arms still.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    init_output_dir(args.output.parent / "client_logs")
    run_diagnostic(
        runtime=args.runtime,
        endpoint=args.endpoint,
        prompt=args.prompt,
        output=args.output,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
