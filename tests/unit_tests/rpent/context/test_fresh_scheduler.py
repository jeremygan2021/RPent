"""Batch scheduler contracts using synthetic trajectories, never model calls."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
import torch
from tests.unit_tests.rpent.context.test_sac import task_episode

SOURCE = Path(__file__).resolve().parents[5] / "train_batch4.py"

def scheduler():
    spec = importlib.util.spec_from_file_location("fresh_sac_scheduler", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_parallel_launch_uses_one_frozen_checkpoint_and_distinct_ports(tmp_path, monkeypatch):
    m = scheduler()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path/"episode_args.json").write_text((SOURCE.parent/"episode_args.json").read_text())
    calls = []
    def popen(argv, **kwargs):
        calls.append(argv)
        assert kwargs["start_new_session"] is True
        return Mock(pid=9000+len(calls))
    monkeypatch.setattr(m.subprocess, "Popen", popen)
    cp = tmp_path/"actor.pt"
    jobs = [m.launch(dict(batch=1,slot=s,attempt=1,variant="sac"),cp,True) for s in range(1,5)]
    assert len({j["port"] for j in jobs})==4
    assert all(a[a.index("--controller-checkpoint")+1]==str(cp) for a in calls)
    assert all(a[a.index("--gpu")+1]=="0" and "--controller-sample" in a for a in calls)
    assert len({a[a.index("--controller-seed")+1] for a in calls})==4
    m.launch(dict(batch=33,slot=1,attempt=1,variant="fixed"),None,False)
    a=calls[-1]
    assert "--controller-sample" not in a and "--controller-checkpoint" not in a
    assert a[a.index("--rho-image")+1]==".2"
    assert a[a.index("--rho-text")+1]==".1"

def test_full_schedule_empty_replay_warmup_restore_and_no_eval_training(tmp_path, monkeypatch):
    m=scheduler()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    monkeypatch.setattr(m,"STATE",tmp_path/"training_state.json")
    monkeypatch.setattr(m,"emit",lambda *a,**k:None)
    starts=[]
    def collect(path,batch,checkpoint,variants=None):
        state=json.loads(m.STATE.read_text())
        assert state["episodes"]==min(batch-1,32)*4
        starts.append((batch,torch.load(checkpoint,weights_only=True)["state_dict"]["actor.bias"].clone()))
        policy=m.MLPController(checkpoint)
        results=[]
        for slot in range(1,5):
            success=slot!=4
            t=task_episode(policy,success=success)
            t["episode_id"]=f"synthetic-{batch}-{slot}"
            transitions=m.transitions_from_trajectory(t,m.REWARD,boundary="task")
            row=dict(batch=batch,slot=slot,variant=variants[slot-1] if variants else "sac",
                     success=success,task_input_tokens=300,task_duration_s=15,
                     task_requests=2,evaluation_reward=float(transitions.rewards.sum()))
            results.append((t,row))
        return results
    monkeypatch.setattr(m,"collect",collect)
    m.main()
    state=json.loads(m.STATE.read_text())
    assert state["episodes"]==128 and state["updates"]==320 and state["actor_updates"]==150
    saved=torch.load(state["training_checkpoint"],weights_only=True)
    assert len(saved["episode_ids"])==128
    assert all(x.startswith("synthetic-") for x in saved["episode_ids"])
    assert torch.equal(starts[0][1], starts[1][1]) and torch.equal(starts[0][1],starts[2][1])
    assert not torch.equal(starts[0][1],starts[-1][1])
    assert sum(len(json.loads(p.read_text())) for p in (tmp_path/"state").glob("episode_metrics_*.json"))==128
    assert len(json.loads((tmp_path/"deployment_metrics.json").read_text()))==4
    m.main() # Restore completes without adding another training episode.
    restored=torch.load(state["training_checkpoint"],weights_only=True)
    assert restored["steps"]==320 and len(restored["episode_ids"])==128


def test_retry_classifies_stream_disconnect_without_accepting_other_agent_errors(tmp_path, monkeypatch):
    m=scheduler()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    out=tmp_path/"runs"/"bad"/"output"
    out.mkdir(parents=True)
    job=dict(label="bad")
    payload=dict(issues=["missing_usage","agent_error_requires_review"],
                 outcome=dict(agent_error=json.dumps(dict(error=dict(
                     codex_error_info=dict(response_stream_disconnected={})))) ))
    (out/"trajectory.json").write_text(json.dumps(payload))
    assert m.is_transport_failure(job)
    payload["outcome"]["agent_error"]=json.dumps(dict(error=dict(
        codex_error_info=dict(context_window_exceeded={}))))
    (out/"trajectory.json").write_text(json.dumps(payload))
    assert not m.is_transport_failure(job)
    payload["outcome"]["agent_error"]="unknown"
    (out/"trajectory.json").write_text(json.dumps(payload))
    assert not m.is_transport_failure(job)
    payload["issues"]=["missing_usage","selection_fallback"]
    (out/"trajectory.json").write_text(json.dumps(payload))
    assert not m.is_transport_failure(job)



def test_occupied_port_stops_before_any_collector_launch(tmp_path, monkeypatch):
    import pytest
    m=scheduler()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    monkeypatch.setattr(m,"probe",lambda label: None)
    launch=Mock()
    monkeypatch.setattr(m,"launch",launch)
    calls=[]
    class PortCheck:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def setsockopt(self,*args): pass
        def bind(self,address):
            calls.append(address)
            if len(calls)==4:
                raise OSError(98,"Address already in use")
    monkeypatch.setattr(m.socket,"socket",PortCheck)
    with pytest.raises(OSError):
        m.collect(tmp_path/"jobs.json",3,tmp_path/"actor.pt")
    assert len(calls)==4
    launch.assert_not_called()


def test_retry_sdk_timeout_requires_recorded_transport_failure(tmp_path, monkeypatch):
    m=scheduler()
    monkeypatch.setattr(m,"ROOT",tmp_path)
    out=tmp_path/"runs"/"timeout"/"output"
    out.mkdir(parents=True)
    job=dict(label="timeout")
    payload=dict(
        issues=["agent_error_requires_review","missing_http_status",
                "missing_latency","missing_usage","transport_failure"],
        outcome=dict(agent_error="Codex SDK timed out after 1200s"),
        decisions=[dict(attempts=[dict(timing=dict(transport_error=True))])])
    path=out/"trajectory.json"
    path.write_text(json.dumps(payload))
    assert m.is_transport_failure(job)
    payload["decisions"][0]["attempts"][0]["timing"].clear()
    path.write_text(json.dumps(payload))
    assert not m.is_transport_failure(job)
    payload["decisions"][0]["attempts"][0]["timing"]["transport_error"]=True
    payload["issues"].append("selection_fallback")
    path.write_text(json.dumps(payload))
    assert not m.is_transport_failure(job)
