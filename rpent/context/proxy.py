# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loopback HTTP proxy for optional Codex context selection.

Run one proxy per episode. Full mode forwards original bytes unchanged.
"""

import argparse
import copy
import gzip
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from .anchor import AnchorAdapter as CodexAdapter
from .features import Features
from .session import ContextSession

logger = logging.getLogger(__name__)
_HOP = {"host", "connection", "content-length", "transfer-encoding", "accept-encoding"}


def rewrite_body(
    raw: bytes, encoding: str, session: ContextSession
) -> tuple[bytes, bool]:
    """Decode supported bodies and serialize only if selection changed content."""
    data = raw
    if encoding == "gzip":
        data = gzip.decompress(raw)
    elif encoding == "zstd":
        import io

        import zstandard

        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
            data = reader.read()
    elif encoding:
        raise ValueError(f"unsupported content encoding: {encoding}")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        return raw, False
    result = session.process(payload)
    session.stats.pop("cache_prefix", None)
    reference_path = os.environ.get("RPENT_FIXED_TOOL_REFERENCE") or (
        Path(__file__).resolve().parents[2] / "tool_reference.json"
    )
    if (
        isinstance(payload.get("input"), list)
        and payload["input"]
        and not payload.get("previous_response_id")
    ):
        from .cache_prefix import ToolReference, move_memory_to_tail

        native = False
        preserve_tool_history = os.environ.get("RPENT_PRESERVE_TOOL_HISTORY") == "1"
        if getattr(session, "mode", "all") != "full" and not preserve_tool_history:
            reference = getattr(session, "_fixed_tool_reference", None)
            if reference is None:
                reference = ToolReference(reference_path)
                session._fixed_tool_reference = reference
            native = reference.native_tools_cover_reference(result)
            if not native:
                result, session.stats["cache_prefix"] = reference.apply(result)
        if native or getattr(session, "mode", "all") == "full" or preserve_tool_history:
            layout = copy.deepcopy(result)
            moved = move_memory_to_tail(layout)
            if moved:
                result = layout
            session.stats["cache_prefix"] = {
                "enabled": False,
                "native_tools": native,
                "memory_layout": "dynamic-tail-v1",
                "memory_moved": moved,
                "duplicates_removed": 0,
            }
    if result is payload:
        return raw, False
    return json.dumps(result, ensure_ascii=False).encode(), True


def serve(
    *,
    port: int,
    upstream: str,
    session: ContextSession,
    log: Path,
    audit_dir: Path | None = None,
) -> None:
    """Serve one episode locally; never write authentication or raw images to logs."""
    lock = threading.Lock()
    client = httpx.Client(
        timeout=httpx.Timeout(300, connect=20), follow_redirects=False
    )
    log.parent.mkdir(parents=True, exist_ok=True)

    counter = 0
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            nonlocal counter
            started_request = time.perf_counter()
            audit_prefix = None
            request_id = None
            if self.path.split("?", 1)[0].endswith("/responses"):
                with lock:
                    counter += 1
                    request_id = f"{counter:04d}"
                    if audit_dir:
                        audit_prefix = audit_dir / request_id
            response_file = None
            timing = {
                "started_at": time.time(),
                "usage": None,
                "request_id": request_id,
            }
            if self.headers.get("Transfer-Encoding"):
                self.send_error(411, "Content-Length required")
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            headers["accept-encoding"] = "identity"
            if raw and self.path.split("?", 1)[0].endswith("/responses"):
                try:
                    with lock:
                        # One proxy is intentionally scoped to one episode. No global history.
                        if audit_prefix:
                            from .audit import capture_body

                            capture_body(
                                str(audit_prefix) + ".before.json",
                                raw,
                                self.headers.get("Content-Encoding", ""),
                            )
                        started = time.perf_counter()
                        processed, changed = rewrite_body(
                            raw, self.headers.get("Content-Encoding", ""), session
                        )
                        session.stats["processing_s"] = time.perf_counter() - started
                        session.stats["request_id"] = request_id
                        timing["decision_id"] = session.stats.get("decision", {}).get(
                            "id"
                        )
                        if audit_prefix:
                            capture_body(
                                str(audit_prefix) + ".after.json",
                                processed,
                                ""
                                if changed
                                else self.headers.get("Content-Encoding", ""),
                            )
                        with log.open("a") as stream:
                            stream.write(
                                json.dumps(session.stats, ensure_ascii=False) + "\n"
                            )
                        raw = processed
                        if changed:
                            headers = {
                                k: v
                                for k, v in headers.items()
                                if k.lower() != "content-encoding"
                            }
                except Exception:
                    logger.exception(
                        "Context processing failed; forwarding original request"
                    )
                    with log.open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "fallback": "processing_error",
                                    "request_id": request_id,
                                }
                            )
                            + "\n"
                        )
            try:
                with client.stream(
                    self.command,
                    upstream.rstrip("/") + self.path,
                    headers=headers,
                    content=raw,
                ) as response:
                    timing["status"] = response.status_code
                    timing["headers_s"] = time.perf_counter() - started_request
                    if audit_prefix:
                        response_file = open(str(audit_prefix) + ".response", "wb")
                        timing["content_encoding"] = response.headers.get(
                            "content-encoding", ""
                        )
                    self.send_response(response.status_code)
                    for key, value in response.headers.items():
                        if key.lower() not in _HOP | {"content-encoding"}:
                            self.send_header(key, value)
                    # iter_raw preserves compression if an upstream ignores identity.
                    if response.headers.get("content-encoding"):
                        self.send_header(
                            "Content-Encoding", response.headers["content-encoding"]
                        )
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    for chunk in response.iter_raw():
                        timing.setdefault(
                            "first_byte_s", time.perf_counter() - started_request
                        )
                        if response_file:
                            response_file.write(chunk)
                            response_file.flush()
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (httpx.HTTPError, OSError):
                logger.exception(
                    "Context proxy forwarding failed: method=%s path=%s request_id=%s",
                    self.command,
                    self.path.split("?", 1)[0],
                    request_id,
                )
                self.close_connection = True
                timing["transport_error"] = True
            finally:
                if response_file:
                    response_file.close()
                if audit_prefix:
                    timing["latency_s"] = time.perf_counter() - started_request
                    Path(str(audit_prefix) + ".timing.json").write_text(
                        json.dumps(timing)
                    )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        client.close()


def main() -> None:
    """Start the optional image-selection proxy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18971)
    parser.add_argument("--upstream", default="https://chatgpt.com")
    parser.add_argument(
        "--mode",
        choices=("full", "image", "text", "image_text", "action", "all"),
        default="full",
    )
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--rho-image", type=float)
    parser.add_argument("--rho-text", type=float)
    parser.add_argument("--rho-action", type=float)
    parser.add_argument(
        "--action-ranges",
        type=Path,
        help="JSON mapping state/command numeric fields to [lower, upper] bounds",
    )
    parser.add_argument(
        "--cameras", default="", help="Ordered camera slots from the tool contract"
    )
    parser.add_argument("--stage", default="Unknown")
    parser.add_argument(
        "--encoder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--warmup-encoder",
        action="store_true",
        help="Load and warm the semantic encoder before listening; skipped when neither history nor memory selection is enabled",
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument(
        "--memory-blocks",
        type=Path,
        help="Demonstration memory (.txt with titled blocks or block-list .json).",
    )
    parser.add_argument("--rho-memory", type=float, default=0.6)
    parser.add_argument("--controller-checkpoint", type=Path)
    parser.add_argument("--controller-sample", action="store_true")
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument(
        "--controller-behavior",
        choices=[
            "actor",
            "lower",
            "lower-moderate",
            "lower-strong",
            "lower-strong-safe",
        ],
        default="actor",
    )
    args = parser.parse_args()
    if args.controller_sample and not args.controller_checkpoint:
        parser.error("--controller-sample requires --controller-checkpoint")
    if args.controller_checkpoint and args.mode == "full":
        parser.error("Full mode uses no learned controller")
    try:
        action_ranges = (
            json.loads(args.action_ranges.read_text()) if args.action_ranges else None
        )
        if args.action_ranges and not isinstance(action_ranges, dict):
            raise ValueError("action ranges must be a field-to-bounds object")
        controller = None
        if args.controller_checkpoint:
            from .policy import MLPController

            controller = MLPController(
                args.controller_checkpoint,
                sample=args.controller_sample,
                seed=args.controller_seed,
                behavior=args.controller_behavior,
            )
        memory_selector = None
        if args.memory_blocks:
            from .memory import MemorySelector

            memory_selector = MemorySelector(args.memory_blocks, args.rho_memory)
        features = Features(args.encoder)
        if args.warmup_encoder and (args.mode != "full" or memory_selector):
            started = time.perf_counter()
            features.warmup()
            logger.info("Context encoder ready in %.3fs", time.perf_counter() - started)
        session = ContextSession(
            CodexAdapter(tuple(filter(None, args.cameras.split(",")))),
            features,
            mode=args.mode,
            rho=args.rho,
            rho_image=args.rho_image,
            rho_text=args.rho_text,
            rho_action=args.rho_action,
            stage=args.stage,
            action_ranges=action_ranges,
            controller=controller,
            memory_selector=memory_selector,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    serve(
        port=args.port,
        upstream=args.upstream,
        session=session,
        log=args.log,
        audit_dir=args.audit_dir,
    )


if __name__ == "__main__":
    main()
