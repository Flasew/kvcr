# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hash lookup uses progress snapshots; owner methods consume completed operations."""

import threading

import msgspec
import pytest
from _kvcr_test_utils import _decode_control_message, _wait_until
from test_kvcr_move import _Pair

from kvcr.backends.remote_ops import (
    _HashResolutionOp,
    _ResolvedHash,
    _ResolveHashesReply,
    _ResolveHashesRequest,
)
from kvcr.policy import FIFOPolicy
from kvcr.types import BlockKey, QueryStatus


@pytest.mark.parametrize(
    "answer,reason",
    [
        (KeyError("missing"), "not_found"),
        (ValueError("collision"), "ambiguous_hash"),
        (b"stale-key", "source_not_transferable"),
    ],
)
def test_adapter_lookup_failure_does_not_discard_other_hashes(answer, reason):
    pair = _Pair()
    key = BlockKey(b"live-full-key")
    pair.deposit(key)

    class Adapter:
        def decode_from_router(self, router_key):
            if router_key == 123:
                return key
            if isinstance(answer, Exception):
                raise answer
            return answer

    backend = pair.source._core._remote_fw_dram
    backend._key_adapter = Adapter()
    assert backend._resolve_hashes((456, 123)) == (
        _ResolvedHash(456, reason=reason),
        _ResolvedHash(123, key, ("",), (16,)),
    )


def test_no_key_adapter_leaves_full_key_access_available():
    pair = _Pair()
    key = BlockKey(b"live-full-key")
    pair.source._core._remote_fw_dram._key_adapter = None
    assert pair.deposit(key).success
    assert pair.source.query((key,))[0][0] is QueryStatus.HIT
    assert pair.source._core._remote_fw_dram._resolve_hashes((123,)) == (
        _ResolvedHash(123, reason="key_adapter_unavailable"),
    )


@pytest.fixture
def pair():
    pair = _Pair()
    pair.resolutions = []
    pair.lookups = []
    pair.rows = (
        _ResolvedHash(123, key=b"full-key-with-group", layout=("",), sizes=(16,)),
        _ResolvedHash(456, reason="not_found"),
    )
    owner = threading.get_ident()

    def resolve(hashes):
        assert threading.get_ident() != owner
        assert not pair.source._core._state_lock._is_owned()
        pair.lookups.append(hashes)
        return pair.rows

    def resolved(op):
        assert threading.get_ident() == owner
        assert not pair.target._core._state_lock._is_owned()
        pair.resolutions.append(op)

    pair.source._core._remote_fw_dram._resolve_hashes = resolve
    pair.target._core._independent_hint_handler.resolved = resolved
    try:
        yield pair
    finally:
        pair.target.close()
        pair.source.close()


def start(pair, hashes=(123, 456), deadline=10, endpoint=None):
    op = _HashResolutionOp(
        op_id=("resolve_hashes", 42),
        keys=frozenset(),
        endpoint=pair.source_control.endpoint if endpoint is None else endpoint,
        hashes=hashes,
        _backend=pair.target._core._remote_fw_dram,
    )
    op.deadline = deadline
    pair.target._core._progress.submit(op)
    return op


def inject_reply(pair, *, rows=None, endpoint="tcp://source:1", error=None):
    pair.target_control.incoming.append(
        msgspec.msgpack.encode(
            _ResolveHashesReply(
                op_handle=42,
                results=pair.rows if rows is None else rows,
                error=error,
                sender_control_endpoint=endpoint,
                sender_incarnation="source-generation",
            )
        )
    )


def test_resolution_round_trip_routes_to_unlocked_owner_without_native_transfer(
    pair,
):
    op = start(pair)
    pair.until(lambda: bool(pair.resolutions))

    assert pair.resolutions == [op]
    assert op.response == pair.rows and op.failure is None
    assert pair.lookups == [(123, 456)]
    assert not pair.source_agent.xfers and not pair.target_agent.xfers
    for control in (pair.source_control, pair.target_control):
        for _, raw in control.sent:
            message = _decode_control_message(raw)
            assert "target_agent_metadata" not in message
            assert "target_regions" not in message
            assert "dst_descriptors" not in message
    for worker in (pair.source, pair.target):
        assert not worker._core._block_record_map
        assert worker._core._outstanding_operations == 0
        assert not worker._core._remote_fw_dram._remote_agents_by_target


def test_source_lookup_replies_while_owner_paused_and_state_lock_held(pair):
    key = BlockKey(b"full-key-with-group")
    pair.deposit(key)
    core = pair.source._core
    backend = core._remote_fw_dram
    real_resolve = type(backend)._resolve_hashes
    owner = threading.get_ident()
    source_copies = len(pair.source_agent.xfers)

    def resolve(hashes):
        assert threading.get_ident() != owner
        pair.lookups.append(hashes)
        return real_resolve(backend, hashes)

    backend._resolve_hashes = resolve
    op = start(pair)
    _wait_until(lambda: op.sent)
    with pair.source._core._state_lock:
        # Relay bytes only. Never poll the source owner, and keep its lock held
        # until the reply and destination continuation have both completed.
        pair.source_control.incoming.extend(raw for _, raw in pair.target_control.sent)
        pair.control_offsets[1] = len(pair.target_control.sent)
        _wait_until(lambda: bool(pair.source_control.sent))
        pair.target_control.incoming.extend(raw for _, raw in pair.source_control.sent)
        pair.control_offsets[0] = len(pair.source_control.sent)
        _wait_until(lambda: op.response is not None)
        assert not pair.resolutions

        def completed():
            assert not list(pair.target.poll_completed())
            return bool(pair.resolutions)

        _wait_until(completed)
        assert pair.lookups == [(123, 456)]
        assert pair.resolutions == [op]
        assert op.response[0].key == key
        assert op.response[1].reason == "not_found"
    assert len(pair.source_agent.xfers) == source_copies
    assert not pair.target_agent.xfers


def test_reply_preserves_requested_source_endpoint_alias(pair):
    op = start(pair, endpoint="tcp://source-alias:1")
    pair.until(lambda: bool(pair.resolutions))
    assert op.response == pair.rows and op.failure is None
    response = _decode_control_message(pair.source_control.sent[-1][1])
    assert response["sender_control_endpoint"] == "tcp://source-alias:1"


def test_wrong_endpoint_and_duplicate_replies_do_not_replace_first_result(pair):
    op = start(pair)
    _wait_until(lambda: op.sent)
    inject_reply(pair, endpoint="tcp://unrelated:1")
    _wait_until(lambda: not pair.target_control.incoming)
    assert op.response is None and op.failure is None
    inject_reply(pair)
    inject_reply(pair, rows=(), error="late_conflicting_reply")
    _wait_until(lambda: op.response is not None)
    assert not pair.resolutions  # Progress did not run owner-side resolution handling.
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.response == pair.rows and op.failure is None
    assert pair.resolutions == [op]


@pytest.mark.parametrize(
    "rows",
    [
        (),
        (_ResolvedHash(123, reason="missing"),),
        (_ResolvedHash(123, reason="missing"), _ResolvedHash(123, reason="missing")),
        (_ResolvedHash(123, reason="missing"), _ResolvedHash(999, reason="missing")),
        (_ResolvedHash(123), _ResolvedHash(456, reason="missing")),
        (
            _ResolvedHash(123, key=b"k", layout=("",), sizes=(0,)),
            _ResolvedHash(456, reason="missing"),
        ),
        (
            _ResolvedHash(123, key=b"k", layout=("a", "b"), sizes=(16,)),
            _ResolvedHash(456, reason="missing"),
        ),
        (
            _ResolvedHash(123, key=b"k", layout=("",), sizes=(16,), reason="error"),
            _ResolvedHash(456, reason="missing"),
        ),
    ],
)
def test_invalid_resolution_rows_fail_without_starting_data_transfer(pair, rows):
    op = start(pair)
    _wait_until(lambda: op.sent)
    inject_reply(pair, rows=rows)
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.failure == "invalid_response" and op.response is None
    assert not pair.source_agent.xfers and not pair.target_agent.xfers


def test_schema_invalid_reply_is_ignored_until_operation_timeout(pair):
    now = [0.0]
    pair.target._core._clock = lambda: now[0]
    op = start(pair, deadline=1)
    _wait_until(lambda: op.sent)
    pair.target_control.incoming.append(
        msgspec.msgpack.encode(
            {
                "type": "hashes_resolved",
                "op_handle": 42,
                "sender_control_endpoint": "tcp://source:1",
                "results": [{"block_hash": 123, "key": 99}],
            }
        )
    )
    _wait_until(lambda: not pair.target_control.incoming)
    assert op.response is None and op.failure is None
    now[0] = 2
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.failure == "timeout"


def test_protocol_accepts_reordered_rows_before_owner_identity_validation(pair):
    op = start(pair)
    _wait_until(lambda: op.sent)
    rows = tuple(
        _ResolvedHash(value, key=b"shared-full-key", layout=("",), sizes=(16,))
        for value in (456, 123)
    )
    # The wire layer validates row shape and requested hashes. The owner handler
    # separately rejects a key whose adapter projection does not match its row.
    inject_reply(pair, rows=rows)
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.response == rows and op.failure is None


def test_resolution_timeout_and_late_reply_do_not_populate_memory(pair):
    now = [0.0]
    pair.target._core._clock = lambda: now[0]
    op = start(pair, deadline=1)
    _wait_until(lambda: op.sent)
    now[0] = 2.0
    inject_reply(pair)
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.failure == "timeout" and op.response is None
    assert pair.target._core._outstanding_operations == 0
    assert not pair.target._core._block_record_map
    inject_reply(pair)
    _wait_until(lambda: not pair.target_control.incoming)
    pair.pump(target_requests=False)
    assert pair.resolutions == [op]


def test_send_failure_reports_failure_and_releases_operation_counter(pair, monkeypatch):
    monkeypatch.setattr(pair.target_control, "send", lambda *_args: False)
    op = start(pair)
    pair.until(lambda: bool(pair.resolutions), target_requests=False)
    assert op.failure == "control_send_failed"
    assert pair.target._core._outstanding_operations == 0


def test_close_abandons_control_only_operation_without_owner_dispatch(pair):
    op = start(pair, deadline=120)
    _wait_until(lambda: op.sent)
    pair.target.close()
    assert not pair.resolutions
    assert pair.target._core._outstanding_operations == 0
    assert pair.target._core.is_quiescent()


@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_source_deadline_is_rejected_without_index_lookup(pair, remaining):
    pair.source_control.incoming.append(
        msgspec.msgpack.encode(
            _ResolveHashesRequest(
                op_handle=9,
                hashes=(123,),
                remaining_timeout_ms=remaining,
                sender_control_endpoint="tcp://target:1",
            )
        )
    )
    _wait_until(lambda: bool(pair.source_control.sent))
    response = _decode_control_message(pair.source_control.sent[-1][1])
    assert response["type"] == "hashes_resolved"
    assert response["error"] == "invalid_request"
    assert not pair.lookups


@pytest.mark.parametrize(
    "error", [RuntimeError("adapter failed"), NotImplementedError()]
)
def test_adapter_error_returns_control_error_without_owner_poll(error):
    pair = _Pair()

    class Adapter:
        def decode_from_router(self, router_key):
            raise error

    pair.source._core._remote_fw_dram._key_adapter = Adapter()
    pair.source_control.incoming.append(
        msgspec.msgpack.encode(
            _ResolveHashesRequest(
                op_handle=9,
                hashes=(123,),
                remaining_timeout_ms=1000,
                sender_control_endpoint="tcp://target:1",
            )
        )
    )
    _wait_until(lambda: bool(pair.source_control.sent))
    response = _decode_control_message(pair.source_control.sent[-1][1])
    assert response["error"] == "resolution_failed"
    assert not pair.source_agent.xfers


def test_source_eviction_after_manifest_does_not_retarget_same_hash():
    pair = _Pair()
    pair.source._core._policy._policy = FIFOPolicy()
    old, replacement = BlockKey(b"old-full-key"), BlockKey(b"replacement-full-key")
    pair.deposit(old, no_evict=False)
    completed = []
    pair.target._core._independent_hint_handler.resolved = completed.append
    manifest = start(pair, hashes=(123,))
    pair.until(lambda: bool(completed))
    assert manifest.response[0].key == old

    retention = pair.deposit(replacement, index=1).release_handle
    assert old not in pair.source._core._block_record_map
    assert (
        pair.source._core._remote_fw_dram._key_adapter.decode_from_router(123)
        == replacement
    )
    assert manifest.response[0].key == old
    # The selected exact identity disappeared. Do not reinterpret the same hash
    # as the replacement or revoke its retention during a failed move.
    pair.source_pinning.prefix_length = 0
    operation = pair.start((manifest.response[0].key,), move=True)
    result = pair.result(operation)
    assert not result[old].success
    assert pair.source_pinning.searches == [(old,)]
    assert pair.target.query((old,))[0][0] is QueryStatus.MISS
    assert replacement not in pair.target._core._block_record_map
    assert pair.source.release((retention,)) == [(retention, True)]


def test_self_addressed_request_is_refused_without_lookup():
    """A destination cannot use its own cache as a resolution source."""
    pair = _Pair()
    backend = pair.source._core._remote_fw_dram
    lookups = []
    backend._resolve_hashes = lookups.append
    pair.source_control.incoming.append(
        msgspec.msgpack.encode(
            _ResolveHashesRequest(
                op_handle=9,
                hashes=(123,),
                remaining_timeout_ms=1000,
                sender_control_endpoint="tcp://source:1",
                sender_incarnation=backend._dangling_ops.incarnation,
            )
        )
    )
    _wait_until(lambda: bool(pair.source_control.sent))
    response = _decode_control_message(pair.source_control.sent[-1][1])
    assert response["type"] == "hashes_resolved"
    assert response["error"] == "self_source"
    assert not lookups
    assert not pair.source_agent.xfers
