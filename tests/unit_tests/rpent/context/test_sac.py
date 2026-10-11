"""SAC terminal, density, replay, and resume regression checks."""
import copy
import math
import pytest
import torch
from rpent.context.sac import SACConfig, SACTrainer, sample_nc, soft_target, transitions_from_trajectory
from rpent.context.training import PPOConfig
from rpent.context.policy import MLPController, feature_names
from tests.unit_tests.rpent.context.test_policy import checkpoint
from tests.unit_tests.rpent.context.test_training import episode


def setup_policy(tmp_path):
    return MLPController(checkpoint(tmp_path))


def test_sigmoid_density_and_gradients(tmp_path):
    policy = setup_policy(tmp_path)
    model = copy.deepcopy(policy.model).requires_grad_(True)
    states = torch.zeros(8, len(feature_names()))
    torch.manual_seed(11)
    action, log_prob = sample_nc(model, states)
    latent = torch.logit(action)
    distribution, _ = model(states)
    expected = (distribution.log_prob(latent) - torch.log(action) - torch.log1p(-action)).sum(-1)
    assert torch.allclose(log_prob, expected, atol=1e-5)
    assert ((action > 0) & (action < 1)).all()
    log_prob.mean().backward()
    assert torch.isfinite(model.log_std.grad).all()


def test_terminal_target():
    target = soft_target(torch.tensor([1., 1.]), torch.tensor([1., 0.]),
                         torch.tensor([99., 2.]), torch.tensor([3., 3.]),
                         SACConfig(alpha=.1, gamma=.9))
    assert target.tolist() == pytest.approx([1., 2.53])


def test_reward_and_strict_audit(tmp_path):
    policy = setup_policy(tmp_path)
    t = episode(policy)
    t["outcome"]["timing"] = {"task_duration_s": 240.}
    config = PPOConfig(token_scale=1000000, beta=.8, latency_beta=.1)
    transitions = transitions_from_trajectory(t, config)
    cost = sum(a["usage"]["input_tokens"] for r in t["decisions"] for a in r["attempts"])
    assert transitions.rewards.sum().item() == pytest.approx(1-.8*cost/1000000-.02)
    assert transitions.done.tolist()[-1] == 1
    assert transitions.next_states[-1].eq(0).all()
    t["audit_complete"] = False
    with pytest.raises(ValueError):
        transitions_from_trajectory(t, config)


def test_update_changes_actor_q_and_resume(tmp_path):
    policy = setup_policy(tmp_path)
    trainer = SACTrainer(policy, SACConfig(batch_size=8, capacity=100), seed=9)
    t = episode(policy)
    trainer.add(t, PPOConfig(1000000))
    with pytest.raises(ValueError):
        trainer.add(t, PPOConfig(1000000))
    before = trainer.actor.actor.bias.detach().clone()
    q_before = [p.detach().clone() for p in trainer.q1.parameters()]
    metrics = trainer.update()
    assert all(math.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, trainer.actor.actor.bias)
    assert any(not torch.equal(b,p) for b,p in zip(q_before,trainer.q1.parameters()))
    path = tmp_path/"sac.pt"
    trainer.save(path)
    clone = SACTrainer(policy, trainer.config)
    clone.restore(path)
    assert clone.steps == 1
    assert torch.equal(clone.replay.rewards, trainer.replay.rewards)
    assert clone.actor_optimizer.state_dict()["state"]
    clone.export(tmp_path/"actor.pt")
    loaded = MLPController(tmp_path/"actor.pt")
    assert not loaded.legacy
    assert math.isfinite(clone.update()["q_loss"])


def task_episode(policy, *, success=True):
    trajectory = episode(policy, costs=(100, 200, 300, 400), success=success)
    trajectory["outcome"]["timing"] = {
        "started_at": 100., "task_end_at": 115., "task_duration_s": 15.,
    }
    for row, began in zip(trajectory["decisions"], [102., 110., 116., 120.]):
        row["attempts"][0]["timing"]["started_at"] = began
    return trajectory


@pytest.mark.parametrize("success", [True, False])
def test_task_boundary_cost_and_terminal(tmp_path, success):
    policy = setup_policy(tmp_path)
    t = task_episode(policy, success=success)
    original = copy.deepcopy(t)
    config = PPOConfig(token_scale=1000, beta=.1, latency_beta=.1)
    data = transitions_from_trajectory(t, config, boundary="task")
    assert len(data.rewards) == 2
    assert data.rewards.sum().item() == pytest.approx(float(success)-.03-.1*15/1200)
    assert data.done.tolist() == [0., 1.]
    assert torch.equal(data.next_states[0], data.states[1])
    assert data.next_states[-1].eq(0).all()
    assert t == original
    assert len(transitions_from_trajectory(t, config).rewards) == 4


def test_task_boundary_includes_start_at_verdict_and_retry_cost(tmp_path):
    policy = setup_policy(tmp_path)
    t = task_episode(policy)
    t["decisions"][2]["attempts"][0]["timing"]["started_at"] = 115.
    retry = copy.deepcopy(t["decisions"][2]["attempts"][0])
    retry["request_id"] = "post-retry"
    retry["response_id"] = "post-retry"
    retry["timing"]["started_at"] = 117.
    t["decisions"][2]["attempts"].append(retry)
    data = transitions_from_trajectory(t, PPOConfig(1000), boundary="task")
    assert len(data.rewards) == 3
    assert data.rewards.sum().item() == pytest.approx(1-.1*600/1000)


@pytest.mark.parametrize("fault", [
    "missing_usage_after_task", "transport_after_task", "missing_start",
    "unordered_start", "missing_end", "inconsistent_duration", "no_task_decisions",
])
def test_task_boundary_rejects_incomplete_or_ambiguous_logs(tmp_path, fault):
    t = task_episode(setup_policy(tmp_path))
    timing = t["outcome"]["timing"]
    last = t["decisions"][-1]["attempts"][0]
    if fault == "missing_usage_after_task":
        last["usage"].pop("input_tokens")
    elif fault == "transport_after_task":
        last["timing"]["transport_error"] = "disconnect"
    elif fault == "missing_start":
        last["timing"].pop("started_at")
    elif fault == "unordered_start":
        last["timing"]["started_at"] = 105.
    elif fault == "missing_end":
        timing.pop("task_end_at")
    elif fault == "inconsistent_duration":
        timing["task_duration_s"] = 99.
    elif fault == "no_task_decisions":
        timing.update(task_end_at=101., task_duration_s=1.)
    with pytest.raises(ValueError):
        transitions_from_trajectory(t, PPOConfig(1000), boundary="task")


def test_replay_boundary_is_persisted_and_cannot_be_mixed(tmp_path):
    policy = setup_policy(tmp_path)
    trainer = SACTrainer(policy, SACConfig(batch_size=4), boundary="task")
    trainer.add(task_episode(policy), PPOConfig(1000))
    assert len(trainer.replay.rewards) == 2
    path = tmp_path/"task.pt"
    trainer.save(path)
    restored = SACTrainer(policy, trainer.config, boundary="task")
    restored.restore(path)
    assert restored.boundary == "task"
    assert torch.equal(restored.replay.rewards, trainer.replay.rewards)
    assert math.isfinite(restored.update()["q_loss"])
    with pytest.raises(ValueError, match="boundary mismatch"):
        SACTrainer(policy, trainer.config).restore(path)
    payload = torch.load(path, weights_only=True)
    payload.pop("boundary")
    torch.save(payload, tmp_path/"legacy.pt")
    with pytest.raises(ValueError, match="boundary mismatch"):
        restored.restore(tmp_path/"legacy.pt")


@pytest.mark.parametrize("kwargs", [
    {"critic_warmup_steps": -1}, {"critic_warmup_steps": 1.5},
    {"actor_update_interval": 0}, {"actor_update_interval": True},
])
def test_actor_schedule_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SACConfig(**kwargs)


def test_critic_warmup_actor_interval_and_resume(tmp_path):
    policy = setup_policy(tmp_path)
    config = SACConfig(batch_size=8, critic_warmup_steps=2, actor_update_interval=2)
    trainer = SACTrainer(policy, config, seed=9)
    trainer.add(episode(policy), PPOConfig(1000000))
    initial_actor = copy.deepcopy(trainer.actor.state_dict())
    initial_q = copy.deepcopy(trainer.q1.state_dict())
    for _ in range(3):
        metrics = trainer.update()
        assert not metrics["actor_updated"]
        assert all(torch.equal(v, trainer.actor.state_dict()[k])
                   for k, v in initial_actor.items())
    assert any(not torch.equal(v, trainer.q1.state_dict()[k])
               for k, v in initial_q.items())
    assert not trainer.actor_optimizer.state
    trainer.save(tmp_path / "scheduled.pt")
    resumed = SACTrainer(policy, config)
    resumed.restore(tmp_path / "scheduled.pt")
    assert resumed.update()["actor_updated"]
    assert any(not torch.equal(v, resumed.actor.state_dict()[k])
               for k, v in initial_actor.items())
    after_actor = copy.deepcopy(resumed.actor.state_dict())
    assert not resumed.update()["actor_updated"]
    assert all(torch.equal(v, resumed.actor.state_dict()[k])
               for k, v in after_actor.items())


def test_legacy_config_restores_default_actor_schedule(tmp_path):
    policy = setup_policy(tmp_path)
    trainer = SACTrainer(policy, SACConfig(batch_size=4))
    trainer.add(episode(policy), PPOConfig(1000000))
    trainer.update()
    trainer.save(tmp_path / "legacy-config.pt")
    payload = torch.load(tmp_path / "legacy-config.pt", weights_only=True)
    payload["config"].pop("critic_warmup_steps")
    payload["config"].pop("actor_update_interval")
    torch.save(payload, tmp_path / "legacy-config.pt")
    resumed = SACTrainer(policy, trainer.config)
    resumed.restore(tmp_path / "legacy-config.pt")
    assert resumed.steps == 1
    assert resumed.update()["actor_updated"]
