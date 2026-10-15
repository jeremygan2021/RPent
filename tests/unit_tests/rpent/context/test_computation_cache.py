"""Cache boundaries preserve changing inputs and selection failures."""

import base64
import copy
import hashlib
import json

import pytest

from rpent.context.codex import CodexAdapter
from rpent.context.features import Features
from rpent.context.parsing import parse_payload, parsed_payloads
from rpent.context.session import ContextSession


def observation(data):
    return {
        "input": [
            {"type": "custom_tool_call", "call_id": "same-call"},
            {
                "type": "custom_tool_call_output",
                "call_id": "same-call",
                "id": "same-output",
                "output": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {"step": 0, "state": {}, "task_language": "pick"}
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(data).decode(),
                    },
                ],
            },
        ]
    }


def test_image_content_change_with_same_occurrence_is_not_stale(monkeypatch):
    adapter = CodexAdapter(("front",))
    decode = base64.b64decode
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return decode(*args, **kwargs)

    monkeypatch.setattr(base64, "b64decode", counted)
    source = observation(b"first")
    original = copy.deepcopy(source)
    first = adapter.extract(source)[0]
    assert adapter.extract(source)[0] == first
    changed = adapter.extract(observation(b"second"))[0]
    assert first.id == changed.id
    assert changed.data == b"second"
    assert changed.digest == hashlib.sha256(b"second").hexdigest()
    assert len(calls) == 2
    assert source == original


def test_image_eviction_and_oversized_entries_preserve_results():
    adapter = CodexAdapter(("front",))
    source = observation(b"1234")
    url = source["input"][1]["output"][1]["image_url"]
    adapter._IMAGE_CACHE_LIMIT = len(url) + 4
    first = adapter.extract(source)
    adapter.extract(observation(b"5678"))
    assert adapter._image_cache_bytes <= adapter._IMAGE_CACHE_LIMIT
    assert adapter.extract(source) == first
    large = adapter.extract(observation(b"x" * 200))[0]
    assert large.data == b"x" * 200
    assert adapter._image_cache_bytes <= adapter._IMAGE_CACHE_LIMIT
    source["input"][1]["output"][1]["image_url"] = "data:image/png;base64,%%%!"
    with pytest.raises(ValueError):
        adapter.extract(source)


def test_parse_cache_is_scoped_and_reset_after_exception():
    text = '{"state": {"x": 1}}'
    with parsed_payloads():
        first = parse_payload(text)
        assert parse_payload(text) is first
        assert parse_payload('{"state": {"x": 2}}')["state"]["x"] == 2
        with pytest.raises(RuntimeError):
            with parsed_payloads():
                assert parse_payload(text) is not first
                raise RuntimeError("failed selection")
        assert parse_payload(text) is first
    assert parse_payload(text) == first
    assert parse_payload(text) is not first
    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            with parsed_payloads():
                parse_payload("not json")


def test_parse_cache_does_not_leak_between_contexts():
    from contextvars import Context

    with parsed_payloads():
        first = parse_payload('{"x":1}')
        assert Context().run(parse_payload, '{"x":1}') is not first


def test_parse_cache_limit_preserves_uncached_values(monkeypatch):
    from rpent.context import parsing

    monkeypatch.setattr(parsing, "_MAX_ENTRIES", 1)
    with parsed_payloads():
        parse_payload('{"a":1}')
        assert parse_payload('{"b":2}') == {"b": 2}
        assert parse_payload('{"b":2}') is not parse_payload('{"b":2}')


def test_similarity_query_changes_and_eviction():
    np = pytest.importorskip("numpy")
    features = Features("unused")
    features._encoder = object()
    features._SIMILARITY_CACHE_LIMIT = 2
    features._vectors.update(
        {
            "text": np.array([1.0, 0.0]),
            "same": np.array([1.0, 0.0]),
            "other": np.array([0.0, 1.0]),
            "negative": np.array([-1.0, 0.0]),
        }
    )
    assert features.similarity("text", "same") == 1.0
    assert features.similarity("text", "other") == 0.0
    assert features.similarity("text", "negative") == 0.0
    assert len(features._similarities) == 2
    assert features.similarity("text", "same") == 1.0
    assert features.similarity("", "same") == 0.0


def test_session_cleans_parse_cache_when_controller_fails():
    from rpent.context.parsing import _PARSED

    class FailingController:
        def decide(self, observation):
            assert _PARSED.get() is not None
            raise RuntimeError("controller failed")

    session = ContextSession(
        CodexAdapter(), Features("unused"), controller=FailingController()
    )
    with pytest.raises(RuntimeError, match="controller failed"):
        session.process({"input": []})
    assert _PARSED.get() is None
