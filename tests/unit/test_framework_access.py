# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source framework access follows hint permission through waits and retries."""

import pytest
from _kvcr_test_utils import _invalidate, _poll_until
from test_cache_management import _assert_independent_idle, _finished, _submit
from test_kvcr_move import _Pair, _requests

from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, QueryStatus


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("allow_framework", [None, True, False])
def test_source_disappearing_after_resolution_obeys_hint_permission(
    mode, deferred, allow_framework
):
    allowed = allow_framework is not False
    pair = _Pair()
    key = BlockKey(b"resolved-then-removed")
    pair.deposit(key, no_evict=False)
    if deferred:
        # An in-flight spill owns the destination's only slot until it completes.
        pair.target_agent.state = "DONE"
        occupied = BlockKey(b"occupied")
        deposit = pair.target.deposit({occupied: pair.descriptors(endpoint="target")})
        assert dict(_poll_until(pair.target, bool))[deposit][occupied].success
        pools = pair.target._core._local_dram._pools
        pools.acquire_sources((occupied,))
        pools._capacity_eviction_key = occupied
        pools.retire_sources((occupied,))

    handler = pair.target._core._independent_hint_handler
    resolved = handler.resolved

    def remove_before_fetch(op):
        assert op.response[0].key == key
        _invalidate(pair.source, (key,))
        assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
        resolved(op)

    handler.resolved = remove_before_fetch
    _submit(
        pair.target,
        mode,
        (123,),
        request_id="carrier",
        allow_framework=allow_framework,
    )
    if deferred:
        pair.until(lambda: bool(pair.target._core._local_dram._capacity_waiters))
        assert not _requests(pair, message_type="start_write")
        pair.target.discard_hint("carrier")
        pools.release_sources((occupied,))
    pair.copy_done()
    report = _finished(pair)
    assert report.status is (HintStatus.COMPLETED if allowed else HintStatus.FAILED)
    assert report.entries[123].success is allowed
    assert pair.source_pinning.searches == ([(key,)] if allowed else [])
    assert not pair.target._core._remote_fw_dram._request_hints
    assert not pair.target._core._independent_hint_handler._pending_operations
    assert not pair.results[1]
    requests = _requests(pair, message_type="start_write")
    assert requests
    assert all(r.get("allow_framework", True) is allowed for r in requests)
    pair.until(lambda: not pair.source._core._framework_pin_keys)
    if allowed:
        assert pair.target_local[""].raw == pair.source_primary[""].raw[:16]
        assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
        return
    assert report.entries[123].reason == "operation_failed"
    assert key not in pair.target._core._block_record_map

    # An ordinary fetch with default advice can still use the framework.
    handle = pair.start((key,), move=mode == "move")
    pair.copy_done()
    result = pair.result(handle)[key]
    assert result.success
    assert pair.source_pinning.searches == [(key,)]
    assert pair.target_local[""].raw == pair.source_primary[""].raw[:16]
    assert pair.target.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
    assert _requests(pair, message_type="start_write")[-1].get("allow_framework", True)
    pair.until(lambda: not pair.source._core._framework_pin_keys)


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("replace", [False, True], ids=["collision", "replacement"])
def test_source_hash_change_after_resolution_preserves_exact_key(mode, replace):
    """An old manifest must never transfer or remove a new key sharing its hash."""
    pair = _Pair(slots=2)
    old, new = BlockKey(b"resolved-old-key"), BlockKey(b"same-hash-new-key")
    pair.deposit(old, no_evict=False)
    pair.source_primary[""].raw = b"a" * 16 + b"z" * 16 + b"c" * 16
    handler = pair.target._core._independent_hint_handler
    resolved = handler.resolved

    def change_inventory_before_fetch(op):
        assert op.response[0].key == old
        if replace:
            _invalidate(pair.source, (old,))
            _poll_until(
                pair.source, lambda _: old not in pair.source._core._block_record_map
            )
        assert pair.deposit(new, index=1, no_evict=False).success
        adapter = pair.source._core._remote_fw_dram._key_adapter
        if replace:
            assert adapter.decode_from_router(123) == new
        else:
            with pytest.raises(ValueError, match="ambiguous"):
                adapter.decode_from_router(123)
        resolved(op)

    handler.resolved = change_inventory_before_fetch
    _submit(pair.target, mode, (123,), allow_framework=False)
    pair.until(lambda: not handler._resolving)
    pair.copy_done()

    report = _finished(pair)

    assert report.status is (HintStatus.FAILED if replace else HintStatus.COMPLETED)
    assert report.entries[123].keys == (old,)
    assert report.entries[123].success is not replace
    assert report.entries[123].reason == ("operation_failed" if replace else None)
    requests = _requests(pair, message_type="start_write")
    assert requests and all(request["keys"] == [old] for request in requests)
    assert not pair.source_pinning.searches
    assert new not in pair.target._core._block_record_map
    assert pair.source.query((new,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    if replace:
        assert pair.target.query((old,)) == [(QueryStatus.MISS, None)]
    else:
        assert pair.target_local[""].raw[:16] == b"a" * 16
        assert pair.target._core._block_record_map[old].local_dram.claim_count == 0
    assert not pair.results[1]
    _assert_independent_idle(pair)


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("queued", [False, True])
def test_complete_managed_source_works_without_framework_pins(
    mode, queued, monkeypatch
):
    pair = _Pair()
    key = BlockKey(b"managed")
    pair.deposit(key, no_evict=False)
    if queued:
        monkeypatch.setattr(
            pair.source._core._remote_fw_dram,
            "_try_local_source_write",
            lambda *_: False,
        )
    _submit(pair.target, mode, (123,), allow_framework=False)
    pair.copy_done()
    report = _finished(pair)
    assert report.status is HintStatus.COMPLETED
    assert not pair.source_pinning.searches
    assert pair.target_local[""].raw == b"a" * 16
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    assert _requests(pair, message_type="start_write")[0]["allow_framework"] is False
    pair.until(lambda: not pair.source._core._local_dram_sources_by_op)


def _deposit_first_piece(pair):
    """Leave the source key filling: piece ``a`` present, piece ``b`` declared."""
    key = BlockKey(b"partial-source")
    pair.source_agent.state = "DONE"
    deposit = pair.source.deposit(
        {key: pair.descriptors(labels=("a",))}, expected_layout=["a", "b"]
    )
    assert dict(_poll_until(pair.source, bool))[deposit][key].success
    assert pair.source._core._block_record_map[key].local_dram.pending_labels == {"b"}
    return key


@pytest.mark.parametrize(
    "mode, allow_framework",
    [("copy", False), ("copy", True), ("move", False), ("move", True)],
    ids=["copy", "copy-framework", "move", "move-framework"],
)
def test_source_becoming_incomplete_after_resolution(mode, allow_framework):
    """Revalidate a resolved key that is replaced by an incomplete G2 copy."""
    pair = _Pair(labels=("a", "b"))
    key = BlockKey(b"partial-source")
    pair.deposit(key, no_evict=False)
    handler = pair.target._core._independent_hint_handler
    resolved = handler.resolved

    def replace_before_fetch(op):
        assert op.response[0].key == key
        _invalidate(pair.source, (key,))
        assert _deposit_first_piece(pair) == key
        resolved(op)

    handler.resolved = replace_before_fetch
    _submit(pair.target, mode, (123,), allow_framework=allow_framework)
    waits_for_fill = mode == "copy" and allow_framework
    if waits_for_fill:
        sources = pair.source._core._remote_fw_dram._sources
        pair.until(lambda: bool(sources._source_pin_ops))
    else:
        report = _finished(pair)
        assert report.status is HintStatus.FAILED
        assert report.entries[123].reason == "operation_failed"
    assert _requests(pair, message_type="start_write")
    assert not pair.source_pinning.searches
    assert not pair.source._core._invalidation_handler.has_pending()
    deposit = pair.source.deposit({key: pair.descriptors(labels=("b",))})
    pair.until(lambda: deposit in pair.results[0])
    assert pair.results[0][deposit][key].success
    if waits_for_fill:
        assert _finished(pair).status is HintStatus.COMPLETED
        assert pair.target_local["a"].raw == b"a" * 16
        assert pair.target_local["b"].raw == b"b" * 16
    handler = pair.target._core._independent_hint_handler
    assert not handler._resolving
    assert not handler._pending_operations
    assert not pair.target._core._remote_fw_dram._request_hints


def test_incomplete_source_serves_ordinary_copy_after_fill():
    """A request-scoped copy waits for the filling source key. A move does not
    wait: the source rejects the busy key before withdrawing it, see
    test_kvcr_move.test_move_rejects_busy_source_without_blocking_its_producer.
    """
    pair = _Pair(labels=("a", "b"))
    key = _deposit_first_piece(pair)
    ordinary = pair.start((key,), move=False)
    sources = pair.source._core._remote_fw_dram._sources
    pair.until(lambda: bool(sources._source_pin_ops))
    assert ordinary not in pair.results[1]
    deposit = pair.source.deposit({key: pair.descriptors(labels=("b",))})
    pair.until(lambda: deposit in pair.results[0])
    assert pair.results[0][deposit][key].success
    result = pair.result(ordinary)[key]
    assert result.success
    assert not pair.source_pinning.searches
    assert pair.target_local["a"].raw == b"a" * 16
    assert pair.target_local["b"].raw == b"b" * 16
    pair.target.release((result.release_handle,))


@pytest.mark.parametrize("method", ["fetch", "deliver"])
@pytest.mark.parametrize("allow_framework", [None, True, False])
def test_request_permission_survives_advice_replacement_and_metadata_retry(
    method, allow_framework
):
    pair = _Pair()
    key = BlockKey(b"framework-only")
    allowed = allow_framework is not False
    # Force a stale-route refusal followed by a retry carrying metadata.
    remote = pair.target._core._remote_fw_dram
    remote._metadata_acked_sources.add("tcp://source:1")
    _submit(
        pair.target,
        "copy",
        (123,),
        request_id="req",
        scope="request",
        allow_framework=allow_framework,
    )
    if method == "fetch":
        handle = pair.target.fetch((key,), "req")
        destination = pair.target_local[""]
    else:
        handle = pair.target.deliver({key: pair.descriptors(endpoint="target")}, "req")
        destination = pair.target_primary[""]
    pair.until(lambda: bool(_requests(pair, message_type="start_write")))
    _submit(
        pair.target,
        "copy",
        (123,),
        request_id="req",
        scope="request",
        allow_framework=not allowed,
    )
    pair.copy_done()
    result = pair.result(handle)[key]
    assert result.success is allowed
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == 2
    assert "target_agent_metadata" not in requests[0]
    assert requests[1]["target_agent_metadata"] == b"target-md"
    assert all(r.get("allow_framework", True) is allowed for r in requests)
    assert pair.source_pinning.searches == ([(key,)] if allowed else [])
    if allowed:
        assert destination.raw[:16] == pair.source_primary[""].raw[:16]
    if result.release_handle is not None:
        pair.target.release((result.release_handle,))
    pair.until(lambda: not pair.source._core._framework_pin_keys)
