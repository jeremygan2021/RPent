# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Serve real pi05_base inference through RPent RPC; no robot actions are sent."""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np

from robots.xlerobot.observation import ARM_JOINTS, validate_observation
from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("xlerobot.pi05")


class XLeRobotPi05Facade(BaseVLAFacade):
    """Use the installed OpenPI JAX checkpoint with RPent's serialized RPC calls."""

    def __init__(self, checkpoint: Path, *, num_steps: int = 10):
        super().__init__()
        self.checkpoint = checkpoint.resolve()
        self.num_steps = num_steps
        self._rpc["vla.metadata"] = self.metadata
        self._readonly_methods.add("vla.metadata")
        if not (self.checkpoint / "params/_METADATA").is_file():
            raise ValueError(f"OpenPI JAX checkpoint not found: {self.checkpoint}")
        if not 1 <= num_steps <= 50:
            raise ValueError("num_steps must be in [1, 50]")

        # Keep model dependencies out of robot discovery and ordinary CLI imports.
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model, pi0_config, tokenizer
        from openpi.policies.policy import Policy

        logger.info("Loading full Pi0.5 BF16 checkpoint from %s", self.checkpoint)
        started = time.monotonic()
        config = pi0_config.Pi0Config(pi05=True)
        params = model.restore_params(self.checkpoint / "params", dtype=jnp.bfloat16)
        net = config.load(params)
        self._policy = Policy(
            net,
            transforms=[
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    tokenizer.PaligemmaTokenizer(config.max_token_len),
                    discrete_state_input=True,
                ),
                transforms.PadStatesAndActions(config.action_dim),
            ],
            sample_kwargs={"num_steps": num_steps},
        )
        self.load_seconds = time.monotonic() - started
        self.device = str(jax.devices()[0])
        self.parameter_count = sum(p.size for p in jax.tree.leaves(params))
        self.parameter_bytes = sum(
            p.size * p.dtype.itemsize for p in jax.tree.leaves(params)
        )
        logger.info("PI05_MODEL_LOADED %s", self.metadata())

    def metadata(self) -> dict:
        """Describe the model and the diagnostic input/output convention."""
        return {
            "model": "pi05_base",
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "dtype": "bfloat16",
            "parameter_count": self.parameter_count,
            "parameter_bytes": self.parameter_bytes,
            "load_seconds": self.load_seconds,
            "denoising_steps": self.num_steps,
            "action_shape": [50, 32],
            "state_joint_order": list(ARM_JOINTS),
            "state_normalization": "USD joint limits to [-1,1]; diagnostic only",
            "action_semantics": "raw normalized model output; no XLeRobot calibration",
            "execution_enabled": False,
        }

    def predict(self, obs: dict, options: dict | None = None) -> dict:
        """Return finite raw actions and synchronized wall time for one observation."""
        if options:
            raise ValueError("this diagnostic has no execution or per-request options")
        obs = validate_observation(obs)
        started = time.monotonic()
        result = self._policy.infer(obs)
        # infer returns NumPy arrays, synchronizing asynchronous JAX work.
        actions = np.asarray(result["actions"], dtype=np.float32)
        elapsed = time.monotonic() - started
        if actions.shape != (50, 32) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid model output: {actions.shape}")
        logger.info(
            "PI05_INFERENCE_OK shape=%s wall_seconds=%.3f", actions.shape, elapsed
        )
        return {
            "actions": actions,
            "wall_seconds": elapsed,
            "execution_enabled": False,
            "action_semantics": "raw normalized model output; no XLeRobot calibration",
        }


def main() -> None:
    """Run the model server on loopback; RPC shutdown releases model resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    init_output_dir(args.log_dir)
    facade = XLeRobotPi05Facade(args.checkpoint, num_steps=args.num_steps)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: facade._shutdown_event.set())
    facade.serve(transport="http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
