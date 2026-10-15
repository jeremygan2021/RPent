"""Request-scoped reuse of read-only tool payloads."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_PARSED: ContextVar[dict | None] = ContextVar("nc_parsed_payloads", default=None)
_MAX_ENTRIES = 512
_MAX_TEXT_CHARS = 1024 * 1024


@contextmanager
def parsed_payloads() -> Iterator[None]:
    """Share parsed tool payloads only within one selection request."""
    token = _PARSED.set({})
    try:
        yield
    finally:
        _PARSED.reset(token)


def parse_payload(text: str) -> Any:
    """Parse JSON; callers must copy shared results before modifying them."""
    cache = _PARSED.get()
    if cache is None or not isinstance(text, str):
        return json.loads(text)
    if text in cache:
        return cache[text]
    value = json.loads(text)
    if len(cache) < _MAX_ENTRIES and len(text) <= _MAX_TEXT_CHARS:
        cache[text] = value
    return value
