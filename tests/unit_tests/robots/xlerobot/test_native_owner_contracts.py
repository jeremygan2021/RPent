# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Offline HTTP contract checks for legacy and whole-body motor owners."""

from copy import deepcopy

import httpx
import pytest

from robots.xlerobot.control import NativeOwner, executor_mappings

CONTROLLER = "a" * 32
PROFILE = "b" * 64
PROOFS = (
    "calibration_verified",
    "motion_verified",
    "stop_verified",
    "alignment_verified",
    "direction_verified",
    "timing_verified",
    "camera_bindings_verified",
)


@pytest.fixture
def wire(monkeypatch, tmp_path):
    keys, _ = executor_mappings()
    native = {
        "controller_id": CONTROLLER,
        "scope": "xlerobot_040_native_position",
        "source_hash_verified": True,
        "instance_id": "owner-instance-123",
        "generation": 3,
        "fresh": True,
        "held": True,
        "axes": [{"key": key, "name": name} for key, name in keys.items()],
    }
    gates = dict.fromkeys(PROOFS, True)
    gates.update(
        identity_verified=True,
        feedback_fresh=True,
        health_fresh=True,
        in_range=True,
        healthy=True,
    )
    state = {
        "controller_id": CONTROLLER,
        "motor_profile_sha256": PROFILE,
        "profile_sha256": PROFILE,
        "instance_id": native["instance_id"],
        "generation": 9,
        "fresh": True,
        "held": True,
        "ready": True,
        "stop_latched": False,
        "stop_confirmed": False,
        "fault": None,
        "session": None,
        "manual_active": None,
        "positions_ticks": dict.fromkeys(keys.values(), 2000),
        "gates": gates,
    }
    profile = {
        "controller_id": CONTROLLER,
        "motor_profile_sha256": PROFILE,
        "position_order": list(keys.values()),
        "proofs": dict.fromkeys(PROOFS, True),
    }
    responses = {
        "/v1/position-control/status": native,
        "/v1/state": state,
        "/v1/profile": profile,
        "/v1/control/stop": {
            "type": "stopped",
            "instance_id": native["instance_id"],
            "generation": 10,
            "stop_confirmed": True,
        },
    }
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer offline-private-test-token"
        return httpx.Response(200, json=deepcopy(responses[request.url.path]))

    client_class = httpx.Client
    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "robots.xlerobot.control.httpx.Client",
        lambda **kwargs: client_class(transport=transport, **kwargs),
    )
    token = tmp_path / "private-token"
    token.write_text("offline-private-test-token")
    owner = NativeOwner(
        token,
        CONTROLLER,
        endpoint="http://6.6.6.49:8916",
        controller_contract="whole_body",
        motor_profile_sha256=PROFILE,
    )
    return owner, responses, calls


def test_legacy_default_origin_and_single_native_status_remain_compatible(wire):
    whole_body, responses, calls = wire
    owner = NativeOwner(whole_body.token_file, CONTROLLER)
    assert owner.status() == responses["/v1/position-control/status"]
    assert len(calls) == 1
    assert str(calls[0].url) == "http://127.0.0.1:8916/v1/position-control/status"


def test_whole_body_checks_three_contracts_and_preserves_two_generations(wire):
    owner, _, calls = wire
    status = owner.status()
    assert [r.url.path for r in calls] == [
        "/v1/position-control/status",
        "/v1/state",
        "/v1/profile",
    ]
    assert all(r.method == "GET" and r.url.host == "6.6.6.49" for r in calls)
    assert status["execution_ready"] is True and status["blocked_reasons"] == []
    assert status["generation"] == 3
    assert status["owner_generation"] == status["stop_generation"] == 9


@pytest.mark.parametrize(
    "route,key,value",
    [
        ("/v1/state", "controller_id", "wrong-controller"),
        ("/v1/profile", "controller_id", "wrong-controller"),
        ("/v1/position-control/status", "controller_id", "wrong-controller"),
        ("/v1/state", "instance_id", "a-different-instance"),
        ("/v1/state", "motor_profile_sha256", "c" * 64),
        ("/v1/profile", "motor_profile_sha256", "c" * 64),
        ("/v1/state", "profile_sha256", "c" * 64),
        ("/v1/position-control/status", "motor_profile_sha256", "c" * 64),
        ("/v1/profile", "position_order", ["unknown-motor"] * 14),
        ("/v1/state", "positions_ticks", {"unknown-motor": 2000}),
        ("/v1/position-control/status", "scope", "another-control-scope"),
        ("/v1/position-control/status", "axes", [{"key": ["bad"], "name": "bad"}]),
        ("/v1/profile", "position_order", [{"invalid": True}] * 14),
    ],
)
def test_wrong_identity_instance_profile_or_motor_names_are_rejected(
    wire, route, key, value
):
    owner, responses, _ = wire
    responses[route][key] = value
    with pytest.raises(ValueError, match="identity|mapping|profile"):
        owner.status()


@pytest.mark.parametrize("proof", PROOFS)
def test_each_unverified_profile_proof_blocks_native_compat_motion(wire, proof):
    owner, responses, calls = wire
    responses["/v1/profile"]["proofs"][proof] = False
    status = owner.status()
    assert status["fresh"] is True and status["held"] is True
    assert status["ready"] is False and status["execution_ready"] is False
    assert any(proof in reason for reason in status["blocked_reasons"])
    with pytest.raises(ValueError, match="autonomous motion gates"):
        owner.move("arm_1.wrist_roll", 2000, status)
    with pytest.raises(ValueError, match="autonomous motion gates"):
        owner.drive(0.01, 0, 0.1, status, lambda: False)
    assert all(r.method == "GET" for r in calls)


def test_state_gate_cannot_be_overridden_by_profile_proof_or_native_hold(wire):
    owner, responses, _ = wire
    responses["/v1/state"]["gates"]["camera_bindings_verified"] = False
    assert owner.status()["execution_ready"] is False
    responses["/v1/state"]["gates"].pop("camera_bindings_verified")
    assert owner.status()["execution_ready"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("stop_latched", True),
        ("fresh", False),
        ("ready", False),
        ("session", {"id": "another-owner"}),
    ],
)
def test_owner_runtime_locks_and_stale_feedback_block_execution(wire, field, value):
    owner, responses, _ = wire
    responses["/v1/state"][field] = value
    assert owner.status()["execution_ready"] is False


def test_stop_receipt_is_only_accepted_and_fresh_state_confirms_stop(wire):
    owner, responses, calls = wire
    receipt = owner.call("stop", {"operator_ready": True})
    assert calls[-1].url.path == "/v1/control/stop" and calls[-1].method == "POST"
    assert receipt["request_accepted"] is True and receipt["stop_confirmed"] is False
    assert receipt["stop_generation"] == 10
    state = responses["/v1/state"]
    state.update(generation=10, stop_confirmed=True, stop_latched=True)
    state["gates"].update(dict.fromkeys(PROOFS, False))
    responses["/v1/profile"]["proofs"].update(dict.fromkeys(PROOFS, False))
    status = owner.status()
    assert status["execution_ready"] is False
    assert status["stop_confirmed"] is True and status["stop_generation"] == 10
    state["fresh"] = False
    assert owner.status()["stop_confirmed"] is False


def test_unfenced_stop_receipt_leaves_stopping_unknown(wire):
    owner, responses, _ = wire
    responses["/v1/control/stop"].pop("generation")
    with pytest.raises(ValueError, match="STOP receipt is invalid"):
        owner.call("stop", {"operator_ready": True})


def test_all_false_site_proofs_keep_autonomous_execution_blocked(wire):
    owner, responses, _ = wire
    responses["/v1/profile"]["proofs"] = dict.fromkeys(PROOFS, False)
    responses["/v1/state"]["gates"].update(dict.fromkeys(PROOFS, False))
    status = owner.status()
    assert status["fresh"] is True and status["held"] is True
    assert status["execution_ready"] is False
    assert all(
        any(proof in reason for reason in status["blocked_reasons"]) for proof in PROOFS
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://0.0.0.0:8916/api",
        "http://user:secret@6.6.6.49:8916",
        "http://6.6.6.49:8916?token=x",
        "ftp://6.6.6.49:8916",
        "http://6.6.6.49:8916/#x",
    ],
)
def test_endpoint_must_be_an_explicit_http_origin(tmp_path, endpoint):
    with pytest.raises(ValueError, match="HTTP origin"):
        NativeOwner(tmp_path / "unused", CONTROLLER, endpoint=endpoint)


def test_whole_body_requires_pinned_measured_profile_and_known_contract(tmp_path):
    with pytest.raises(ValueError, match="profile SHA256"):
        NativeOwner(tmp_path / "unused", CONTROLLER, controller_contract="whole_body")
    with pytest.raises(ValueError, match="controller contract"):
        NativeOwner(tmp_path / "unused", CONTROLLER, controller_contract="unchecked")
