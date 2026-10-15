# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Linux planner isolation: read-only runtime, task workspace and TCP allowlist.

The planner receives a task-scoped credential only. Motor credentials, host
configuration, devices and other tasks are absent from its filesystem.
"""

import ctypes
import os
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse


def availability(config):
    planner = config.get("planner", {})
    if sys.platform != "linux" or not shutil.which("bwrap"):
        return False, "Linux bubblewrap is required for planner isolation"
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) < 4:
        return False, "Landlock TCP isolation requires ABI 4 or newer"
    auth_file = Path(planner.get("auth_file", Path.home() / ".codex/auth.json"))
    if not planner.get("base_url"):
        return (
            (True, "")
            if auth_file.is_file()
            else (False, "Codex login is unavailable to the isolated planner")
        )
    if not planner.get("api_key_file"):
        return False, "Configured model endpoint requires its API key file"
    parsed = urlparse(planner["base_url"])
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return False, "Invalid configured model endpoint"
    if (parsed.port or (443 if parsed.scheme == "https" else 80)) in (
        8899,
        8900,
        8916,
        8918,
        8919,
        8920,
    ):
        return False, "Model endpoint overlaps a robot control port"
    if not Path(planner["api_key_file"]).is_file():
        return False, "Planner model credential is unavailable"
    return True, ""


def build_launch(config, repo, folder, command, task_token_file):
    ready, reason = availability(config)
    if not ready:
        raise ValueError(reason)
    planner = config.get("planner", {})
    repo, folder = Path(repo).resolve(), Path(folder).resolve()
    work = folder / "run"
    work.mkdir(mode=0o700, exist_ok=True)
    # This is a child filesystem mount, not a change to the user's HOME.
    home = folder / "planner-home"
    home.mkdir(mode=0o700, exist_ok=True)
    codex_home = home / ".codex"
    codex_home.mkdir(mode=0o700, exist_ok=True)
    if not planner.get("base_url"):
        auth_file = Path(planner.get("auth_file", Path.home() / ".codex/auth.json"))
        shutil.copyfile(auth_file, codex_home / "auth.json")
        (codex_home / "auth.json").chmod(0o600)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        mcp_port = listener.getsockname()[1]
    binary = planner.get("codex_bin") or shutil.which("codex")
    if not binary or not Path(binary).resolve().is_relative_to("/usr"):
        raise ValueError("Configure a Codex binary installed under /usr")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "RPENT_CONTROL_URL": "http://127.0.0.1:8919",
        "RPENT_CONTROL_SESSION": folder.name,
        "RPENT_CONTROL_TOKEN_FILE": str(task_token_file),
        "RPENT_MCP_PORT": str(mcp_port),
        "RPENT_PLANNER_WORKSPACE": str(work),
        "CODEX_BIN": str(Path(binary).resolve()),
    }
    if planner.get("base_url"):
        env.update(
            CODEX_BASE_URL=planner["base_url"],
            CODEX_API_KEY=Path(planner["api_key_file"]).read_text().strip(),
        )
    if planner.get("model"):
        env["CODEX_MODEL"] = planner["model"]
    args = [
        shutil.which("bwrap"),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-user",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for path in ("/lib", "/lib64", "/bin", "/sbin"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(home),
        str(Path.home()),
    ]
    for path in ("/etc/ssl", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    interpreter_root = Path(sys.executable).resolve().parent.parent
    if not interpreter_root.is_relative_to("/usr"):
        args += ["--ro-bind", str(interpreter_root), str(interpreter_root)]
    venv_python = repo / ".venv/bin/python"
    if venv_python.is_symlink():
        lexical_root = Path(os.readlink(venv_python)).parent.parent
        if lexical_root.is_absolute() and lexical_root != interpreter_root:
            args += ["--ro-bind", str(lexical_root.resolve()), str(lexical_root)]
    for name in (".venv", "rpent", "robots", "pyproject.toml"):
        args += ["--ro-bind", str(repo / name), str(repo / name)]
    args += [
        "--bind",
        str(work),
        str(work),
        "--ro-bind",
        str(task_token_file),
        str(task_token_file),
        "--chdir",
        str(work),
    ]
    parsed = urlparse(planner.get("base_url", "https://api.openai.com"))
    ports = [8919, mcp_port, parsed.port or (443 if parsed.scheme == "https" else 80)]
    args += [
        str(repo / ".venv/bin/python"),
        "-m",
        "robots.xlerobot.isolation",
        ",".join(map(str, ports)),
        "--",
        *command,
    ]
    return args, env


def restrict_tcp(ports):
    """Apply Landlock to this process and every descendant before exec."""
    libc = ctypes.CDLL(None, use_errno=True)

    class Ruleset(ctypes.Structure):
        _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64)]

    class NetPort(ctypes.Structure):
        _fields_ = [("allowed", ctypes.c_uint64), ("port", ctypes.c_uint64)]

    def checked(result):
        if result < 0:
            raise OSError(ctypes.get_errno(), "Planner network isolation failed")
        return result

    rules = Ruleset(0, 2)  # LANDLOCK_ACCESS_NET_CONNECT_TCP
    fd = checked(libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0))
    try:
        for port in ports:
            rule = NetPort(2, int(port))
            checked(libc.syscall(445, fd, 2, ctypes.byref(rule), 0))
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)


if __name__ == "__main__":
    restrict_tcp([int(value) for value in sys.argv[1].split(",")])
    os.execv(sys.argv[3], sys.argv[3:])
