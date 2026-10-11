"""Baseline measurements must remain separate from experiment accounting."""
import json
import httpx
from rpent.context import network_baseline as baseline


def test_fixed_stream_and_link_metrics(monkeypatch):
    monkeypatch.setattr(baseline, "_headers", lambda: {"Authorization": "Bearer SECRET"})
    calls = []
    def handle(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"models": []})
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.6-sol"
        assert payload["reasoning"]["effort"] == "xhigh"
        assert payload["store"] is False
        events = [
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "OK"},
            {"type": "response.completed", "response": {"usage": {
                "input_tokens": 10, "output_tokens": 1}}}]
        return httpx.Response(200, text="".join("data: "+json.dumps(e)+"\n\n" for e in events))
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = baseline.run_baseline("http://example", "gpt-5.6-sol", "xhigh",
                                      link_samples=2, model_samples=1, client=client)
        assert not client.is_closed
    assert len(calls) == 3
    assert result["link_summary"]["successes"] == 2
    assert result["fixed_request_summary"]["successes"] == 1
    assert result["fixed_request"][0]["usage"]["input_tokens"] == 10
    assert result["fixed_request"][0]["first_text_s"] >= 0
    assert result["outside_task"] is True
    assert "SECRET" not in json.dumps(result)


def test_disconnect_and_incomplete_stream_are_failures(monkeypatch):
    monkeypatch.setattr(baseline, "_headers", lambda: {})
    def handle(request):
        if request.method == "GET":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, text='data: {"type":"response.created"}\n\n')
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = baseline.run_baseline("http://example", "model", "low",
                                      link_samples=1, model_samples=1, client=client)
    assert result["link_summary"]["failures"] == 1
    assert result["fixed_request_summary"]["failures"] == 1
    assert "latency_s" not in result["fixed_request_summary"]


def test_missing_auth_is_reported_without_failing_task(monkeypatch):
    def unavailable():
        raise FileNotFoundError()
    monkeypatch.setattr(baseline, "_headers", unavailable)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        result = baseline.run_baseline("http://example", "model", "low", client=client)
    assert result["setup_error"] == "FileNotFoundError"
    assert result["link_summary"]["samples"] == 0
    assert result["fixed_request_summary"]["samples"] == 0


def test_baseline_usage_is_not_added_to_task_summary(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[4]/"scripts/context_benchmark/summarize.py"
    spec = importlib.util.spec_from_file_location("baseline_summary_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write_trajectory", lambda *args: None)
    baseline_result = {"outside_task": True, "fixed_request": [
        {"usage": {"input_tokens": 999, "output_tokens": 10}}]}
    (tmp_path/"network-baseline.json").write_text(json.dumps(baseline_result))
    module.summarize(tmp_path)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["network_baseline"] == baseline_result
    assert summary["input_tokens"] == 0
    assert summary["model_requests"] == 0
    assert summary["model_request_latency_s"] == 0
