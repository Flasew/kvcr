# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request advice and independent prefetch/move hint execution."""

from unittest.mock import Mock

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _decode_control_message,
    _HintAction,
    _HintEnvelope,
    _invalidate,
    _new_kvcr,
    _poll_until,
    _record_hint_results,
    _router_hint,
    _wait_until,
)
from test_kvcr_remote_target import _Pair

from kvcr.backends.remote_ops import _ResolvedHash, _TargetPullOp
from kvcr.config import KVCRConfig, RemoteFWDramOptions
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.parser import _LOGGED_HINT_ISSUES
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, QueryStatus


@pytest.fixture(autouse=True)
def _reset_once_logs():
    # Once-per-issue warnings are process-global; reset so tests that count
    # warnings do not depend on run order.
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


def _take(results) -> list[tuple[str | None, HintStatus, str | None]]:
    """Statuses reported since the previous take."""
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


class _PrefixAdapter:
    def encode_to_router(self, key):
        return int.from_bytes(key[:8], "big")


def _key(block_hash: int) -> BlockKey:
    return BlockKey(block_hash.to_bytes(8, "big"))


def _fetch_kvcr(monkeypatch):
    kvcr = _new_kvcr(
        FakeNixlAgent(),
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", 16)]),
        key_adapter=_PrefixAdapter(),
        remote_options=RemoteFWDramOptions(
            eager_ctrl_connect=False, opportunistic_query=False
        ),
    )
    return kvcr, _record_hint_results(monkeypatch, kvcr)


def _fetch(action_id: str, source: str, hashes, action_type="kv.fetch") -> _HintAction:
    return _HintAction(
        action_id, action_type, "1.0",
        {"source_control_endpoint": source, "block_hashes": list(hashes)},
    )  # fmt: skip


def _fetch_envelope(*hints: _HintAction) -> _HintEnvelope:
    return _HintEnvelope("0.1", "m1", list(hints))


_FETCHABLE = (QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
_MISS = (QueryStatus.MISS, None)


class _CountingLock:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def acquire(self, *args, **kwargs):
        self.entries += 1
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        return self._inner.release()


def test_gpu_fetch_hint_takes_inline_path_without_lock(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    lock = _CountingLock(kvcr._core._state_lock)
    kvcr._core._state_lock = lock

    kvcr.submit_hint(_router_hint("tcp://a:1", (1,)), "req")

    assert lock.entries == 0
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, "req")]


@pytest.mark.parametrize("one_envelope", [True, False], ids=["envelope", "calls"])
@pytest.mark.parametrize(
    ("second_source", "expected"),
    [("tcp://a:1", [_MISS, _FETCHABLE]), ("tcp://b:1", [_MISS, _MISS])],
    ids=["same_source_replaces", "other_source_fails"],
)
def test_second_fetch_for_a_request(monkeypatch, one_envelope, second_source, expected):
    kvcr, results = _fetch_kvcr(monkeypatch)
    first, second = _fetch("a1", "tcp://a:1", [1]), _fetch("a2", second_source, [2])
    for hints in [(first, second)] if one_envelope else [(first,), (second,)]:
        kvcr.submit_hint(_fetch_envelope(*hints), "req")
    # The store's rules apply within an envelope as across calls, and it reports
    # neither the replacement nor the failure.
    assert _take(results) == [
        ("a1", HintStatus.ACCEPTED, None),
        ("a2", HintStatus.ACCEPTED, None),
    ]
    assert kvcr.query((_key(1), _key(2)), "req") == expected


def test_mixed_valid_and_unknown_hints(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    kvcr.submit_hint(
        _fetch_envelope(
            _fetch("a1", "tcp://a:1", [1]),
            _HintAction("f1", "kv.future_action", "1.0", {}),
        ),
        "req",
    )
    assert kvcr.query((_key(1),), "req") == [_FETCHABLE]
    assert set(_take(results)) == {
        ("a1", HintStatus.ACCEPTED, None),
        ("f1", HintStatus.REJECTED, "unsupported_action_type"),
    }


def test_all_invalid_envelope_does_not_raise_from_submit_hint(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    bad = _HintAction("a1", "kv.fetch", "1.0", {"block_hashes": []})
    kvcr.submit_hint(_fetch_envelope(bad), "req")
    assert [r.status for r in results] == [HintStatus.REJECTED]


def test_fetch_hint_without_request_id_is_accepted(monkeypatch):
    kvcr, results = _fetch_kvcr(monkeypatch)
    envelope = _router_hint("tcp://a:1", (1,))
    envelope.actions[0].payload.update(scope="independent", mode="move")
    kvcr.submit_hint(envelope)
    assert [(r.status, r.request_id) for r in results] == [(HintStatus.ACCEPTED, None)]
    assert not kvcr._core._remote_fw_dram._request_hints
    assert not kvcr._core._independent_hint_handler._resolving


@pytest.mark.parametrize("action_type", ["kv.prefetch", "kv.move"])
@pytest.mark.parametrize("request_id", [None, "req"])
def test_independent_hints_require_managed_destination(
    monkeypatch, action_type, request_id
):
    kvcr, results = _fetch_kvcr(monkeypatch)
    action = _fetch("a1", "tcp://a:1", [1], action_type)
    kvcr.submit_hint(_fetch_envelope(action), request_id)
    assert [(r.action_type, r.status, r.reason) for r in results] == [
        (action_type, HintStatus.REJECTED, "destination_has_no_managed_dram")
    ]
    assert results[0].request_id == request_id and results[0].message_id == "m1"
    assert not kvcr._core._remote_fw_dram._request_hints
    assert kvcr._core._outstanding_operations == 0 and not kvcr.poll_completed()


_KEY = BlockKey(b"full-key-with-group")
_TAIL_KEY = BlockKey(b"tail-piece-key")


class _Adapter:
    def decode_from_router(self, block_hash):
        return {123: _KEY, 124: _TAIL_KEY}[block_hash]

    def encode_to_router(self, key):
        return {_KEY: 123, _TAIL_KEY: 124}[key]


def _pair():
    pair = _Pair()
    for peer in (pair.source, pair.target):
        peer.kvcr._core._remote_fw_dram._key_adapter = _Adapter()
    return pair


def _submit(pair, action, carrier=None, hashes=(123,)):
    hint = _fetch("action", pair.source.control.endpoint, hashes, action)
    pair.target.kvcr.submit_hint(_fetch_envelope(hint), carrier)


def _idle(pair):
    pair.until(pair.idle)
    handler = pair.target.kvcr._core._independent_hint_handler
    assert not handler._resolving and not handler._resolved
    assert not handler._pending_operations and not pair.target.results
    source = pair.source.kvcr._core
    assert not source._framework_pin_keys and not source._local_dram_sources_by_op
    for record in source._block_record_map.values():
        if record.local_dram is not None:
            assert (
                record.local_dram.claim_count == record.local_dram.source_readers == 0
            )


@pytest.mark.parametrize("action", ["kv.prefetch", "kv.move"])
@pytest.mark.parametrize("carrier", [None, "unrelated-request"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_independent_transfer_preserves_carrier_and_releases_claims(
    action, carrier, cached, missing, monkeypatch
):
    """Actual bytes, cached moves, and partial failures keep their action identity."""
    pair = _pair()
    pair.source.deposit(_KEY)
    if cached:
        pair.target.deposit(_KEY)
    core = pair.target.kvcr._core
    remote = core._remote_fw_dram
    if carrier:
        remote.submit_hint("tcp://unrelated:9", frozenset({456}), carrier)
    advice = dict(remote._request_hints)
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    before = len(pair.source.agent.xfers)
    _submit(pair, action, carrier, (123, 999) if missing else (123,))
    assert remote._request_hints == advice
    if carrier:
        pair.target.kvcr.discard_hint(carrier)
        advice = {}
    pair.source.agent.state = "DONE"
    pair.until(lambda: len(reports) == 2)
    accepted, result = reports
    assert accepted.status is HintStatus.ACCEPTED
    assert result.status is (HintStatus.FAILED if missing else HintStatus.COMPLETED)
    assert result.op_handle == accepted.op_handle and result.op_handle is not None
    assert result.action_type == action and result.request_id == carrier
    assert result.message_id == "m1" and result.entries[123].key == _KEY
    assert result.entries[123].success
    if missing:
        assert result.entries[999].reason == "not_found"
    assert remote._request_hints == advice
    assert core._block_record_map[_KEY].local_dram.claim_count == 0
    assert pair.target.local.raw[:16] == b"a" * 16
    assert len(pair.source.agent.xfers) - before == (0 if cached else 1)
    expected = (
        (QueryStatus.MISS, None)
        if action == "kv.move"
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    pair.until(lambda: pair.source.kvcr.query((_KEY,)) == [expected])
    _idle(pair)
    assert len(reports) == 2


@pytest.mark.parametrize(
    "action,queued", [("kv.prefetch", False), ("kv.prefetch", True), ("kv.move", False)]
)
@pytest.mark.parametrize("incomplete", [False, True])
def test_disappearing_resolved_source_never_requests_framework_memory(
    action, queued, incomplete, monkeypatch
):
    """Missing or incomplete G2 fails on both source paths without pinning/waiting."""
    pair = _pair()
    pair.source.deposit(_KEY)
    source = pair.source.kvcr._core
    pins = Mock(wraps=source._request_pin_callback)
    monkeypatch.setattr(source, "_request_pin_callback", pins)
    if queued:
        monkeypatch.setattr(
            source._remote_fw_dram, "_try_local_source_write", lambda *_: False
        )
    handler = pair.target.kvcr._core._independent_hint_handler
    resolved = handler.resolved

    def remove_before_fetch(operation):
        _invalidate(pair.source.kvcr, (_KEY,))
        if incomplete:
            pair.source.agent.state = "DONE"
            handle = pair.source.kvcr.deposit(
                {_KEY: [pair.source.ref()]}, expected_layout=["", ":tail"]
            )
            assert dict(_poll_until(pair.source.kvcr, bool))[handle][_KEY].success
            assert source._block_record_map[_KEY].local_dram.pending_labels == {":tail"}
        resolved(operation)

    monkeypatch.setattr(handler, "resolved", remove_before_fetch)
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, action)
    pair.until(lambda: len(reports) == 2)
    assert reports[-1].status is HintStatus.FAILED
    assert reports[-1].entries[123].reason == "operation_failed"
    assert reports[-1].entries[123].key == _KEY
    pins.assert_not_called()
    assert pair.target.kvcr.query((_KEY,)) == [(QueryStatus.MISS, None)]
    _idle(pair)


@pytest.mark.parametrize(
    "row,reason",
    [
        (_ResolvedHash(123, _KEY, ("",), (32,)), "incompatible_layout"),
        (_ResolvedHash(123, _KEY, (":*",), (16,)), "incompatible_key_or_layout"),
        (_ResolvedHash(123, _KEY, ("",), (16,)), "hash_key_mismatch"),
    ],
)
def test_resolved_metadata_is_validated_before_fetch(row, reason, monkeypatch):
    pair = _pair()
    source, target = pair.source.kvcr._core, pair.target.kvcr._core
    monkeypatch.setattr(source._remote_fw_dram, "_resolve_hashes", lambda _: (row,))
    if reason == "hash_key_mismatch":
        monkeypatch.setattr(
            target._remote_fw_dram._key_adapter, "encode_to_router", lambda _: 456
        )
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: len(reports) == 2)
    assert reports[-1].entries[123].reason == reason
    assert not pair.source.agent.xfers and not pair.target.agent.xfers
    assert not target._block_record_map
    _idle(pair)


@pytest.mark.parametrize("at_submit", [False, True])
def test_nonfatal_error_reports_once_and_allows_later_work(at_submit, monkeypatch):
    pair = _pair()
    pair.source.deposit(_KEY)
    core = pair.target.kvcr._core
    owner, name = (core._progress, "submit") if at_submit else (core, "fetch")
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, Mock(side_effect=RuntimeError("nonfatal")))
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: len(reports) == (1 if at_submit else 2))
    if at_submit:
        # The queue failure raised before ACCEPTED: admission reported once.
        assert _take(reports) == [("action", HintStatus.REJECTED, "handler_error")]
    else:
        assert reports[-1].entries[123].reason == "operation_failed"
        reports.clear()
    _idle(pair)
    monkeypatch.setattr(owner, name, original)
    _submit(pair, "kv.prefetch")
    pair.source.agent.state = "DONE"
    pair.until(lambda: len(reports) == 2)
    assert reports[-1].status is HintStatus.COMPLETED
    _idle(pair)


def test_fatal_fetch_submit_error_escapes_the_same_poll(monkeypatch):
    """Fail after real destination reservation, after poll's initial fatal check."""
    pair = _pair()
    pair.source.deposit(_KEY)
    core = pair.target.kvcr._core
    handler, progress = core._independent_hint_handler, core._progress
    resolved, submit = handler.resolved, progress.submit
    manifests = []
    monkeypatch.setattr(handler, "resolved", manifests.append)
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: bool(manifests))
    monkeypatch.setattr(handler, "resolved", resolved)
    handler._resolved.extend(manifests)
    failure = RuntimeError("fatal fetch submit")

    def fail_reserved(item):
        if isinstance(item, _TargetPullOp):
            assert core._block_record_map[_KEY].local_dram.remote_indices is not None
            progress._failure = failure
        return submit(item)

    monkeypatch.setattr(progress, "submit", fail_reserved)
    with pytest.raises(RuntimeError, match="fatal fetch submit") as error:
        pair.target.kvcr.poll_completed()
    assert error.value is failure and len(reports) == 1
    with pytest.raises(RuntimeError, match="fatal fetch submit"):
        pair.target.kvcr.close()
    assert core.is_quiescent()


def test_resolution_timeout_does_not_revive_the_hint(monkeypatch):
    """A late metadata reply cannot restart an action or report completion twice."""
    pair = _pair()
    core = pair.target.kvcr._core
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch")
    _wait_until(lambda: bool(pair.target.control.sent))
    core._clock = lambda: core.config.operation_timeout_ms / 1000 + 1
    pair.until(lambda: len(reports) == 2)
    assert reports[-1].status is HintStatus.FAILED
    assert reports[-1].entries[123].reason == "timeout"
    pair.until(lambda: bool(pair.source.control.sent))
    pair.pump()  # Forward the late reply even if timeout already made us idle.
    _wait_until(lambda: not pair.target.control.incoming)
    pair.target.kvcr.poll_completed()
    _idle(pair)
    assert len(reports) == 2
    assert not pair.source.agent.xfers and not pair.target.agent.xfers
    assert not core._block_record_map


@pytest.mark.parametrize("vanished", [False, True])
def test_prefetch_keeps_named_g2_source_after_capacity_wait(vanished, monkeypatch):
    """A pending eviction delays admission; resumption preserves source and policy."""
    pair = _pair()
    pair.source.deposit(_KEY)
    occupied = (b"occupied-first", b"occupied-second")
    for index, key in enumerate(occupied):
        pair.target.deposit(key, index)
    core = pair.target.kvcr._core
    pools = core._local_dram._pools
    pools.acquire_sources(occupied[:1])
    pools._capacity_eviction_key = occupied[0]
    pools.retire_sources(occupied[:1])
    pins = Mock(wraps=pair.source.kvcr._core._request_pin_callback)
    monkeypatch.setattr(pair.source.kvcr._core, "_request_pin_callback", pins)
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch")
    pair.until(lambda: bool(core._local_dram._capacity_waiters))
    assert len(reports) == 1
    assert all(
        _decode_control_message(raw)["type"] != "start_write"
        for _, raw in pair.target.control.sent
    )
    if vanished:
        _invalidate(pair.source.kvcr, (_KEY,))
    pools.release_sources(occupied[:1])
    pair.source.agent.state = "DONE"
    pair.until(lambda: len(reports) == 2)
    assert reports[-1].entries[123].success is not vanished
    requests = [_decode_control_message(raw) for _, raw in pair.target.control.sent]
    writes = [request for request in requests if request["type"] == "start_write"]
    assert len(writes) == 1 and writes[0]["g2_only"] is True
    assert writes[0]["source_control_endpoint"] == pair.source.control.endpoint
    pins.assert_not_called()
    if vanished:
        assert pair.target.kvcr.query((_KEY,)) == [(QueryStatus.MISS, None)]
    else:
        assert pair.target.local.raw[:16] == b"a" * 16
        assert core._block_record_map[_KEY].local_dram.claim_count == 0
    _idle(pair)


@pytest.mark.parametrize(
    "actions",
    [
        ("kv.prefetch", "kv.move"),
        ("kv.move", "kv.prefetch"),
        ("kv.prefetch", "kv.prefetch"),
    ],
)
def test_overlapping_actions_share_one_transfer(actions, monkeypatch):
    """The second action joins the first fill; a move still removes the source copy."""
    pair = _pair()
    pair.source.deposit(_KEY)
    core = pair.target.kvcr._core
    handler = core._independent_hint_handler
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    before = len(pair.source.agent.xfers)
    for action in actions:
        _submit(pair, action)
    pair.until(lambda: len(handler._pending_operations) == 2)
    pair.source.agent.state = "DONE"
    pair.until(lambda: len(reports) == 4)
    assert [report.status for report in reports[2:]] == [HintStatus.COMPLETED] * 2
    assert sorted(report.action_type for report in reports[2:]) == sorted(actions)
    requests = [_decode_control_message(raw) for _, raw in pair.target.control.sent]
    assert sum(request["type"] == "start_write" for request in requests) == 1
    assert len(pair.source.agent.xfers) - before == 1
    assert pair.target.local.raw[:16] == b"a" * 16
    expected = (
        (QueryStatus.MISS, None)
        if "kv.move" in actions
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    )
    pair.until(lambda: pair.source.kvcr.query((_KEY,)) == [expected])
    _idle(pair)
    assert core._block_record_map[_KEY].local_dram.claim_count == 0


@pytest.mark.parametrize("vanished", [False, True])
def test_layout_groups_aggregate_into_one_result(vanished, monkeypatch):
    """Each distinct source layout fetches separately; the action reports once."""
    pair = _pair()
    pair.source.deposit(_KEY)
    pair.source.agent.state = "DONE"
    handle = pair.source.kvcr.deposit(
        {_TAIL_KEY: [pair.source.ref(1, ":tail")]}, expected_layout=[":tail"]
    )
    assert dict(_poll_until(pair.source.kvcr, bool))[handle][_TAIL_KEY].success
    core = pair.target.kvcr._core
    handler = core._independent_hint_handler
    if vanished:
        resolved = handler.resolved

        def remove_tail_then_resolve(operation):
            _invalidate(pair.source.kvcr, (_TAIL_KEY,))
            resolved(operation)

        monkeypatch.setattr(handler, "resolved", remove_tail_then_resolve)
    reports = _record_hint_results(monkeypatch, pair.target.kvcr)
    _submit(pair, "kv.prefetch", hashes=(123, 124))
    pair.until(lambda: len(reports) == 2)
    result = reports[-1]
    assert result.status is (HintStatus.FAILED if vanished else HintStatus.COMPLETED)
    assert result.entries[123].success and result.entries[123].key == _KEY
    assert result.entries[124].key == _TAIL_KEY
    assert result.entries[124].reason == ("operation_failed" if vanished else None)
    requests = [_decode_control_message(raw) for _, raw in pair.target.control.sent]
    writes = [request for request in requests if request["type"] == "start_write"]
    assert sorted(tuple(write["key_layout"]) for write in writes) == [("",), (":tail",)]
    hit = (QueryStatus.HIT, CacheTier.LOCAL_G2)
    assert pair.target.kvcr.query((_KEY, _TAIL_KEY)) == [
        hit,
        (QueryStatus.MISS, None) if vanished else hit,
    ]
    raw = pair.target.local.raw
    assert b"a" * 16 in (raw[:16], raw[16:])
    assert (b"b" * 16 in (raw[:16], raw[16:])) is not vanished
    _idle(pair)
