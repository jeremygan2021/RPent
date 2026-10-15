# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Start and inspect this workspace's XLeRobot / Pi0.5 diagnostic services."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

from robots.xlerobot.diagnostic import run_diagnostic
from robots.xlerobot.observation import DEFAULT_RUNTIME, sim_request
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.rpc import make_rpc_client

logger = get_logger("xlerobot.local")
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
CHECKPOINT = WORKSPACE / "models/openpi/openpi-assets/checkpoints/pi05_base"
LOGS = REPO / "logs/xlerobot"
ENDPOINT = "http://127.0.0.1:8095"


def model_metadata() -> dict:
    """Read and verify the existing model endpoint before reuse or shutdown."""
    result = make_rpc_client(ENDPOINT).call("vla.metadata", timeout_s=5)
    if (
        result.get("model") != "pi05_base"
        or result.get("checkpoint") != str(CHECKPOINT)
        or result.get("execution_enabled") is not False
    ):
        raise ValueError(
            "port 8095 belongs to a different service; refusing to reuse it"
        )
    return result


def start_services() -> None:
    """Reuse ready services and start missing ones without modifying either environment."""
    launcher = WORKSPACE / "xlerobot_isaac/codex_control.py"
    subprocess.run(["python3", str(launcher), "start"], check=True)
    deadline = time.monotonic() + 180
    while True:
        try:
            status = sim_request(DEFAULT_RUNTIME, "status")
            if status["state"]["ready"]:
                break
        except (OSError, RuntimeError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Isaac Sim did not become ready; check codex_runtime/service.log"
            )
        time.sleep(2)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            model_metadata()
            logger.info("Reusing Pi0.5 at %s", ENDPOINT)
            return
        except ValueError:
            raise
        except Exception:
            pass
        with socket.socket() as probe:
            probe.settimeout(2)
            if probe.connect_ex(("127.0.0.1", 8095)) == 0:
                raise RuntimeError(
                    "port 8095 is occupied but its model metadata could not be verified"
                )
        pending = LOGS / "launch.json"
        if pending.exists():
            saved = json.loads(pending.read_text())
            proc_path = Path("/proc") / str(saved["pid"]) / "cmdline"
            if (
                proc_path.exists()
                and b"robots.xlerobot.vla_server" in proc_path.read_bytes().split(b"\0")
            ):
                raise RuntimeError(
                    f"model PID {saved['pid']} is still starting; inspect {LOGS / 'model.log'}"
                )
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO),
            "OPENPI_DATA_HOME": str(WORKSPACE / "models/openpi"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.64",
        }
        command = [
            str(WORKSPACE / "env_pi05/bin/python"),
            "-u",
            "-m",
            "robots.xlerobot.vla_server",
            "--checkpoint",
            str(CHECKPOINT),
            "--log-dir",
            str(LOGS / "model"),
        ]
        with (LOGS / "model.log").open("a") as log:
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pending.write_text(json.dumps({"pid": proc.pid, "command": command}, indent=2))
        deadline = time.monotonic() + 240
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"model exited {proc.returncode}; inspect {LOGS / 'model.log'}"
                    )
                try:
                    model_metadata()
                    return
                except ValueError:
                    raise
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Pi0.5 did not become ready within 240 seconds")
                time.sleep(2)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            raise


def main() -> None:
    """Expose reproducible start, status, inference, and model-only shutdown commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("start", "status", "infer", "stop-model"),
        nargs="?",
        default="start",
    )
    parser.add_argument("--prompt", default="Keep both arms still.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    init_output_dir(LOGS / "launcher")
    if args.command == "status":
        healthy = True
        for name, fn in (
            ("simulation", lambda: sim_request(DEFAULT_RUNTIME, "status")),
            ("model", model_metadata),
        ):
            try:
                logger.info("%s: %s", name, json.dumps(fn(), ensure_ascii=False))
            except Exception as exc:
                healthy = False
                logger.error("%s: %s", name, exc)
        if not healthy:
            raise SystemExit(1)
        return
    if args.command == "stop-model":
        model_metadata()
        make_rpc_client(ENDPOINT).call("shutdown", timeout_s=30)
        logger.info("Pi0.5 shutdown requested; Isaac Sim remains open")
        return
    if args.command == "start":
        start_services()
        if args.no_verify:
            return
    output = LOGS / ("verify_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_diagnostic(
        runtime=DEFAULT_RUNTIME,
        endpoint=ENDPOINT,
        prompt=args.prompt,
        output=output,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
