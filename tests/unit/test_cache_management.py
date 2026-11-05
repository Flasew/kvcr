# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hash-only commands act independently of the inference request carrying them."""

import logging
import os
import time

import pytest
from _kvcr_test_utils import (
    _has_outstanding_operations,
    _HintAction,
    _HintEnvelope,
    _InventoryKeyAdapter,
    _poll_until,
    _recovered_record,
)
from test_g3 import _FakeG3Agent
from test_kvcr_move import _cache_at_target, _Pair, _requests

from kvcr.backends.remote_ops import _ResolvedHash, _TargetPullOp
from kvcr.config import G3Options
from kvcr.hints.dispatcher import _LOGGED_HANDLER_ERRORS
from kvcr.hints.results import HintStatus
from kvcr.service.recovery_journal import install_recovery_records
from kvcr.types import BlockKey, CacheTier, InventoryEvent, QueryStatus


def _keys(core, block_hash):
    return core._remote_fw_dram._key_adapter.keys_by_hash.get(block_hash, ())


def _reports(worker):
    """Capture the existing report sink once, then drain test-owned results."""
    dispatcher = worker._core._hints_dispatcher
    if not hasattr(dispatcher, "_test_reports"):
        dispatcher._test_reports = []
        original = dispatcher.report

        def report(result):
            dispatcher._test_reports.append(result)
            original(result)

        dispatcher.report = report
    results = list(dispatcher._test_reports)
    dispatcher._test_reports.clear()
    return results


def _submit(
    worker,
    mode,
    hashes,
    *,
    request_id=None,
    source="tcp://source:1",
    scope="independent",
):
    _reports(worker)
    payload = {
        "block_hashes": list(hashes),
        "source_control_endpoint": source,
        "scope": scope,
        "mode": mode,
    }
    worker.submit_hint(
        _HintEnvelope(
            "0.1",
            "management-envelope",
            [
                _HintAction(
                    "management-action",
                    "kv.fetch",
                    "1.0",
                    payload,
                )
            ],
        ),
        request_id,
    )
    reports = _reports(worker)
    assert reports[0].status is HintStatus.ACCEPTED
    return reports


def _finished(pair, worker=None):
    worker = pair.target if worker is None else worker
    reports = []

    def done():
        reports.extend(_reports(worker))
        return bool(reports)

    pair.until(done)
    assert len(reports) == 1
    return reports[0]


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("carrier", [None, "unrelated-inference"])
@pytest.mark.parametrize("protected", [False, True])
def test_hash_transfer_resolves_source_keys_and_preserves_carrier_advice(
    mode, carrier, protected
):
    pair = _Pair()
    key = BlockKey(b"full-key-not-in-carrier-request")
    retention = pair.deposit(key, no_evict=protected).release_handle
    residency = pair.source._core._block_record_map[key].local_dram
    reader_op = pair.source.fetch((key,))
    reader = dict(pair.source.poll_completed())[reader_op][key].release_handle
    remote = pair.target._core._remote_fw_dram
    if carrier:
        remote.submit_hint("tcp://unrelated-source:9", frozenset({999}), carrier)
        original = remote._request_hints[carrier]
    _submit(pair.target, mode, (123,), request_id=carrier)
    assert _keys(pair.target._core, 123) == ()
    pair.until(lambda: bool(_requests(pair, message_type="start_write")))
    if carrier:
        assert remote._request_hints[carrier] is original
        pair.target.discard_hint(carrier)
    pair.copy_done()
    report = _finished(pair)
    assert report.status is HintStatus.COMPLETED
    assert report.entries[123].success and report.entries[123].keys == (key,)
    assert pair.target.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert _keys(pair.target._core, 123) == (key,)
    assert pair.target_local[""].raw == pair.source_local[""].raw
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    assert not remote._request_hints and not pair.results[1]
    pair.until(lambda: residency.source_readers == 0)
    removed = mode == "move" and not protected
    if removed:
        # The source withdrew the key at admission; removal waits for the reader.
        pair.until(lambda: pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((key,)) == [
        (QueryStatus.MISS, None) if removed else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    ]
    assert pair.source.release((reader,)) == [(reader, True)]
    pair.until(lambda: not pair.source._core._invalidation_handler.has_pending())
    assert pair.source.query((key,)) == [
        (QueryStatus.MISS, None) if removed else (QueryStatus.HIT, CacheTier.LOCAL_G2)
    ]
    assert _keys(pair.source._core, 123) == (() if removed else (key,))
    if retention is not None:
        assert pair.source.release((retention,)) == [(retention, True)]
    pair.target_agent.state = "DONE"
    delivered = pair.target.deliver({key: pair.descriptors(endpoint="target")})
    result = dict(_poll_until(pair.target, bool))[delivered][key]
    assert result.success
    assert pair.target_primary[""].raw[:16] == b"a" * 16


@pytest.mark.parametrize("source", ["tcp://source:1", "tcp://other:2"])
def test_independent_operation_preserves_request_advice_with_generated_id(source):
    """An application's request ID must not be reused for private fetch advice."""
    pair = _Pair()
    key = BlockKey(b"private-advice")
    pair.deposit(key, no_evict=False)
    core = pair.target._core
    prefix = core._independent_hint_handler._advice_prefix
    carrier = f"{prefix}:{core._next_op_handle}:0"
    _submit(
        pair.target, "copy", (999,), request_id=carrier, scope="request", source=source
    )
    original = core._remote_fw_dram._request_hints[carrier]

    _submit(pair.target, "copy", (123,), request_id=carrier)
    pair.copy_done()

    assert _finished(pair).status is HintStatus.COMPLETED
    assert core._remote_fw_dram._request_hints == {carrier: original}
    assert not core._independent_hint_handler._pending_operations
    pair.target.discard_hint(carrier)


def test_failed_hash_move_preserves_source_without_advertising_failed_destination():
    pair = _Pair()
    key = BlockKey(b"native-move-error")
    try:
        retention = pair.deposit(key).release_handle
        initial_transfers = len(pair.source_agent.xfers)
        _submit(pair.target, "move", (123,))
        pair.until(lambda: len(pair.source_agent.xfers) > initial_transfers)
        assert _requests(pair, message_type="start_write")
        pair.source_agent.state = "ERR"
        report = _finished(pair)
        assert report.status is HintStatus.FAILED
        assert report.entries[123].reason == "operation_failed"
        assert pair.source.release((retention,)) == [(retention, True)]
        assert not pair.source._core._invalidation_handler.has_pending()
        assert not pair.source._core._block_record_map[key].pending_invalidation
        # ERR does not prove that the native writer has stopped. Its quarantined
        # destination keeps its identity until cleanup observes native DONE.
        assert key in pair.target._core._block_record_map
        assert _keys(pair.target._core, 123) == ()
        pair.source_agent.state = "DONE"
        pair.until(lambda: key not in pair.target._core._block_record_map)
        core = pair.target._core
        assert _keys(core, 123) == ()
        assert not pair.results[1]
        assert not core._independent_hint_handler._resolving
        assert not core._independent_hint_handler._pending_operations
        assert not core._independent_hint_handler._resolved
        assert not core._remote_fw_dram._request_hints
        assert core._outstanding_operations == 0
    finally:
        pair.source_agent.state = "DONE"
        pair.target_agent.state = "DONE"
        pair.target.close()
        pair.source.close()


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_hash_collision_rejected_without_selecting_either_full_key(mode):
    pair = _Pair(slots=2)
    keys = (BlockKey(b"hash-and-group-0"), BlockKey(b"hash-and-group-1"))
    retentions = [
        pair.deposit(key, index=index).release_handle for index, key in enumerate(keys)
    ]
    worker = pair.target
    reports = _submit(worker, mode, (123,))
    report = reports[-1] if len(reports) == 2 else _finished(pair, worker)
    assert report.status is HintStatus.FAILED
    assert report.entries[123].reason == "ambiguous_hash"
    assert not _requests(pair, message_type="start_write")
    assert pair.source.release(retentions) == [(handle, True) for handle in retentions]
    assert set(_keys(pair.source._core, 123)) == set(keys)
    assert all(status is QueryStatus.HIT for status, _ in pair.source.query(keys))


def test_resolved_full_key_does_not_overwrite_different_local_key_with_same_hash():
    pair = _Pair(slots=2)
    source_key, local_key = BlockKey(b"source-full-key"), BlockKey(b"other-full-key")
    pair.deposit(source_key)
    pair.target_agent.state = "DONE"
    deposited = pair.target.deposit(
        {local_key: pair.descriptors(index=1, endpoint="target")}, no_evict=True
    )
    local_retention = dict(_poll_until(pair.target, bool))[deposited][
        local_key
    ].release_handle
    local_before = pair.target_local[""].raw[:16]
    _submit(pair.target, "move", (123,))
    pair.copy_done()
    assert _finished(pair).status is HintStatus.COMPLETED
    assert set(_keys(pair.target._core, 123)) == {source_key, local_key}
    assert pair.target_local[""].raw[:16] == local_before
    assert pair.target.release((local_retention,)) == [(local_retention, True)]


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_unknown_hash_reports_failure_without_fabricating_keys(mode):
    pair = _Pair()
    worker = pair.target
    reports = _submit(worker, mode, (999,))
    report = reports[-1] if len(reports) == 2 else _finished(pair, worker)
    assert report.status is HintStatus.FAILED
    assert report.entries[999].reason == "not_found"
    assert not pair.source_pinning.searches
    assert not pair.target._core._block_record_map


def test_incompatible_pool_sizes_rejected_before_transfer():
    pair = _Pair()
    pair.deposit(BlockKey(b"full-key"))
    pair.target._core._block_sizes[""] = 32
    _submit(pair.target, "move", (123,))
    report = _finished(pair)
    assert report.entries[123].reason == "incompatible_layout"
    assert not _requests(pair, message_type="start_write")


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize(
    "case,layout,sizes,reason",
    [
        ("hash_mismatch", ("",), (16,), "hash_key_mismatch"),
        ("incompatible_size", ("",), (32,), "incompatible_layout"),
        ("unsupported_pool", ("missing",), (16,), "incompatible_key_or_layout"),
        ("malformed_part", (":*",), (16,), "incompatible_key_or_layout"),
        ("adapter_error", ("",), (16,), "incompatible_key_or_layout"),
    ],
)
def test_resolved_metadata_validation_fails_without_starting_transfer(
    mode, case, layout, sizes, reason, monkeypatch
):
    pair = _Pair()
    key = BlockKey(b"retained-source-key")
    core = pair.target._core

    class Adapter:
        def encode_to_router(self, key):
            if case == "adapter_error":
                raise RuntimeError("framework-specific decoding failure")
            return 456 if case == "hash_mismatch" else 123

    try:
        retained = pair.deposit(key).release_handle
        source_bytes = pair.source_local[""].raw
        source_transfers = len(pair.source_agent.xfers)
        monkeypatch.setattr(core._remote_fw_dram, "_key_adapter", Adapter())
        # Inject source metadata, retaining actual wire encoding/decoding,
        # owner validation, dispatcher routing, and completion handling.
        monkeypatch.setattr(
            pair.source._core._remote_fw_dram,
            "_resolve_hashes",
            lambda hashes: (_ResolvedHash(123, key, layout, sizes),),
        )
        admitted = _submit(pair.target, mode, (123,), request_id="carrier")
        assert len(admitted) == 1

        report = _finished(pair)

        assert report.status is HintStatus.FAILED
        assert report.op_handle == admitted[0].op_handle
        assert report.action_type == "kv.fetch"
        assert set(report.entries) == {123}
        assert not report.entries[123].success
        assert report.entries[123].keys == ()
        assert report.entries[123].reason == reason
        assert not _requests(pair, message_type="start_write")
        assert len(pair.source_agent.xfers) == source_transfers
        assert not pair.target_agent.xfers
        assert pair.source_local[""].raw == source_bytes
        assert pair.source.release((retained,)) == [(retained, True)]
        assert not pair.source._core._invalidation_handler.has_pending()
        assert not core._block_record_map
        assert not core._remote_fw_dram._request_hints
        assert not core._independent_hint_handler._resolving
        assert not core._independent_hint_handler._pending_operations
        assert (
            not core._independent_hint_handler._resolved
            and not core._pending_progress_items
        )
        assert not core._completion_queue and core._outstanding_operations == 0
        assert core._progress._submissions.empty()
        assert not core._progress._in_flight_ops
        assert not pair.results[1]

        # The failed hint must not break later public completions or report twice.
        missing = BlockKey(b"ordinary-missing-key")
        ordinary = pair.target.fetch((missing,), expected_layout=[""])
        completed = dict(_poll_until(pair.target, bool))
        assert set(completed) == {ordinary} and not completed[ordinary][missing].success
        pair.pump()
        assert not pair.results[1] and not _reports(pair.target)
    finally:
        pair.target.close()
        pair.source.close()


def test_partial_missing_hash_is_reported_alongside_successful_copy():
    pair = _Pair()
    key = BlockKey(b"full-key")
    pair.deposit(key)
    _submit(pair.target, "copy", (123, 999))
    pair.copy_done()
    report = _finished(pair)
    assert report.status is HintStatus.FAILED
    assert report.entries[123].success
    assert report.entries[999].reason == "not_found"
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0


def test_independent_operation_records_no_hint_wait_telemetry():
    """hint_wait measures router advice lead time; private advice has none."""
    pair = _Pair(slots=2)
    key, other = BlockKey(b"full-key"), BlockKey(b"other-full-key")
    pair.deposit(key)
    core = pair.target._core
    scopes = []
    core._timer = time.monotonic
    core._record_duration = lambda scope, started_at, result: scopes.append(scope)
    _submit(pair.target, "copy", (123,))
    pair.copy_done()
    assert _finished(pair).status is HintStatus.COMPLETED
    assert "hint_wait" not in scopes
    # Router advice for a framework fetch still records its wait.
    pair.deposit(other, index=1)
    handle = pair.start((other,), move=False)
    pair.copy_done()
    assert pair.result(handle)[other].success
    assert scopes.count("hint_wait") == 1


def test_inventory_removal_forgets_hash_after_admitted_reader_drains():
    pair = _Pair()
    key = BlockKey(b"full-key")
    pair.deposit(key, no_evict=False)
    reader_op = pair.source.fetch((key,))
    reader = dict(pair.source.poll_completed())[reader_op][key].release_handle
    core = pair.source._core
    core._prune_block_record(key)
    assert _keys(core, 123) == (key,)
    core._invalidation_handler.submit((key,))
    core.poll_completed()
    assert _keys(core, 123) == (key,)
    assert (
        core._remote_fw_dram._resolve_hashes((123,))[0].reason
        == "source_not_transferable"
    )
    assert pair.source.release((reader,)) == [(reader, True)]
    core.poll_completed()
    assert key not in core._block_record_map
    assert _keys(core, 123) == ()
    replacement = BlockKey(b"replacement-full-key")
    assert pair.deposit(replacement, index=1).success
    assert _keys(core, 123) == (replacement,)


def test_distinct_hashes_are_indexed_without_changing_full_key_lookup():
    class Adapter(_InventoryKeyAdapter):
        def encode_to_router(self, key):
            return key[0]

    pair = _Pair(slots=2, key_adapter_type=Adapter)
    core = pair.source._core
    assert not _keys(core, ord("a"))
    keys = (BlockKey(b"a-group"), BlockKey(b"b-group"))
    for index, key in enumerate(keys):
        pair.deposit(key, index=index)
    assert _keys(core, ord("a")) == (keys[0],)
    assert _keys(core, ord("b")) == (keys[1],)
    assert all(status is QueryStatus.HIT for status, _ in pair.source.query(keys))


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("include_missing", [False, True])
def test_hash_transfer_groups_layouts_and_preserves_framework_completion(
    mode, include_missing
):
    class Adapter(_InventoryKeyAdapter):
        def encode_to_router(self, key):
            return key[0]

    pair = _Pair(labels=("a", "b"), key_adapter_type=Adapter)
    keys = (BlockKey(b"a-full-key"), BlockKey(b"b-full-key"))
    retentions = []
    pair.source_agent.state = "DONE"
    for key, label in zip(keys, pair.labels):
        deposited = pair.source.deposit(
            {key: pair.descriptors(labels=(label,))},
            no_evict=True,
            expected_layout=[label],
        )
        retentions.append(
            dict(_poll_until(pair.source, bool))[deposited][key].release_handle
        )
    pair.source_agent.state = "PROC"
    hashes = (ord("a"), ord("b"), 999) if include_missing else (ord("a"), ord("b"))
    admitted = _submit(pair.target, mode, hashes)
    missing = BlockKey(b"ordinary-missing-key")
    ordinary = pair.target.fetch((missing,), expected_layout=["a"])
    pair.copy_done()

    report = _finished(pair)

    assert report.status is (
        HintStatus.FAILED if include_missing else HintStatus.COMPLETED
    )
    assert report.op_handle == admitted[0].op_handle
    assert set(report.entries) == set(hashes)
    if include_missing:
        assert not report.entries[999].success
        assert report.entries[999].reason == "not_found"
    for key, label in zip(keys, pair.labels):
        assert report.entries[key[0]].success
        assert report.entries[key[0]].keys == (key,)
        record = pair.target._core._block_record_map[key]
        assert record.local_dram.key_layout == (label,)
        assert record.local_dram.claim_count == 0
        assert pair.target_local[label].raw == pair.source_local[label].raw
    assert set(pair.results[1]) == {ordinary}
    assert not pair.results[1][ordinary][missing].success
    assert not pair.target._core._remote_fw_dram._request_hints
    pair.source.poll_completed()
    assert pair.source.release(retentions) == [(handle, True) for handle in retentions]
    pair.pump()
    assert not _reports(pair.target)


def test_concurrent_hash_fetch_and_move_route_to_their_own_actions():
    class Adapter(_InventoryKeyAdapter):
        def encode_to_router(self, key):
            return key[0]

    pair = _Pair(slots=2, key_adapter_type=Adapter)
    copied, moved = BlockKey(b"a-copy"), BlockKey(b"b-move")
    retained_copy = pair.deposit(copied).release_handle
    pair.deposit(moved, index=1, no_evict=False)
    fetch_receipt = _submit(pair.target, "copy", (copied[0],))[0]
    move_receipt = _submit(pair.target, "move", (moved[0],))[0]
    reports = []
    pair.copy_done()

    def done():
        reports.extend(_reports(pair.target))
        return len(reports) == 2

    pair.until(done)

    by_handle = {result.op_handle: result for result in reports}
    assert set(by_handle) == {fetch_receipt.op_handle, move_receipt.op_handle}
    for receipt, key in ((fetch_receipt, copied), (move_receipt, moved)):
        result = by_handle[receipt.op_handle]
        assert result.action_type == receipt.action_type
        assert result.status is HintStatus.COMPLETED
        assert set(result.entries) == {key[0]}
        assert result.entries[key[0]].keys == (key,)
        assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    pair.until(lambda: pair.source.query((moved,)) == [(QueryStatus.MISS, None)])
    assert pair.source.release((retained_copy,)) == [(retained_copy, True)]
    assert pair.source.query((copied,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert not pair.results[1]
    handler = pair.target._core._independent_hint_handler
    assert not handler._resolving
    assert not handler._pending_operations
    assert not pair.target._core._remote_fw_dram._request_hints
    pair.pump()
    assert not _reports(pair.target)


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_close_abandons_unresolved_hash_commands(mode):
    pair = _Pair()
    receipt = _submit(pair.target, mode, (123,))[0]
    core = pair.target._core
    handler = core._independent_hint_handler
    assert set(handler._resolving) == {receipt.op_handle}

    pair.target.close()

    assert core.is_quiescent()
    assert not handler._resolving
    assert not handler._pending_operations
    assert not core._independent_hint_handler._resolved
    assert not core._remote_fw_dram._request_hints
    assert not _reports(pair.target)


def test_hash_move_joins_an_existing_fetch_and_does_not_duplicate_the_copy():
    pair = _Pair()
    key = BlockKey(b"concurrent-fetch")
    pair.deposit(key, no_evict=False)
    copied = pair.start((key,), move=False)
    pair.until(lambda: bool(_requests(pair, message_type="start_write")))
    _submit(pair.target, "move", (123,), request_id="unrelated-carrier")
    pair.until(
        lambda: bool(pair.target._core._independent_hint_handler._pending_operations)
    )
    assert copied not in pair.results[1]
    pair.copy_done()
    report = _finished(pair)
    assert report.status is HintStatus.COMPLETED
    pair.until(lambda: pair.source.query((key,)) == [(QueryStatus.MISS, None)])
    requests = _requests(pair, message_type="start_write")
    assert len(requests) == 1
    assert requests[0]["dst_descriptors"] and not requests[0].get("move", False)
    # The joined copy lands first; the complete key is then invalidated as a hit.
    invalidations = _requests(pair, message_type="invalidate")
    assert len(invalidations) == 1 and invalidations[0]["keys"] == [key]
    reader = pair.results[1][copied][key].release_handle
    assert pair.target.release((reader,)) == [(reader, True)]


def test_g3_inventory_is_resolvable_but_not_a_remote_transfer_source(tmp_path):
    page = os.sysconf("SC_PAGE_SIZE")
    pair = _Pair(
        block_size=page,
        source_agent=_FakeG3Agent(),
        source_g3=G3Options(
            paths=(tmp_path / "cache.data",),
            capacity_bytes_per_file=page,
            backend="MOCK",
        ),
    )
    key = BlockKey(b"recovered-disk-key")
    core = pair.source._core
    install_recovery_records(core, {key: _recovered_record(g3=0)})
    # Recovery does not replay inventory; the adapter needs an explicit snapshot.
    assert _keys(core, 123) == ()
    core._inventory_sink_callback(InventoryEvent((key,), CacheTier.G3, False))
    assert _keys(core, 123) == (key,)
    _submit(pair.target, "move", (123,))
    assert _finished(pair).entries[123].reason == "source_not_transferable"
    assert not _requests(pair, message_type="start_write")
    assert pair.source.query((key,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
    assert _keys(core, 123) == (key,)


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_invalid_hash_hint_reports_rejection_without_starting_work(mode):
    pair = _Pair()
    _reports(pair.target)
    pair.target.submit_hint(
        _HintEnvelope(
            "0.1",
            "bad-command",
            [
                _HintAction(
                    "bad-action",
                    "kv.fetch",
                    "1.0",
                    {
                        "block_hashes": [True],
                        "scope": "independent",
                        "mode": mode,
                        "source_control_endpoint": "tcp://source:1",
                    },
                ),
            ],
        ),
        "unrelated-carrier",
    )
    reports = _reports(pair.target)
    assert len(reports) == 1 and reports[0].status is HintStatus.REJECTED
    assert reports[0].action_id == "bad-action"
    assert not pair.target._core._independent_hint_handler._resolving
    assert not pair.target._core._independent_hint_handler._pending_operations
    assert not pair.target._core._independent_hint_handler._resolved
    assert pair.target._core._outstanding_operations == 0


@pytest.mark.parametrize("action", ["kv.move", "kv.invalidate"])
def test_removed_hint_types_do_not_start_hash_operations(action):
    pair = _Pair()
    _reports(pair.target)
    pair.target.submit_hint(
        _HintEnvelope(
            "0.1",
            "unsupported",
            [
                _HintAction(
                    "unsupported",
                    action,
                    "1.0",
                    {
                        "block_hashes": [123],
                        "scope": "independent",
                        "source_control_endpoint": "tcp://source:1",
                    },
                ),
            ],
        )
    )
    reports = _reports(pair.target)
    assert len(reports) == 1 and reports[0].status is HintStatus.REJECTED
    assert not pair.target._core._independent_hint_handler._resolving
    assert not hasattr(pair.target, "invalidate")


def _assert_independent_idle(pair, worker=None):
    """Private work must drain without leaking advice, traces, or late reports."""
    pair.until(
        lambda: (
            not any(
                _has_outstanding_operations(worker)
                for worker in (pair.source, pair.target)
            )
        )
    )
    worker = pair.target if worker is None else worker
    core = worker._core
    handler = core._independent_hint_handler
    assert not handler._resolving
    assert not handler._resolved
    assert not handler._pending_operations
    assert not core._remote_fw_dram._request_hints
    assert not core._nvtx_operations
    assert not _reports(worker)


@pytest.mark.parametrize("first", ["copy", "move"])
@pytest.mark.parametrize("second", ["copy", "move"])
def test_same_key_independent_operations_share_transfer_and_complete_once(
    first, second
):
    """Both manifests arrive before admission; each action still owns its result."""
    pair = _Pair()
    key = BlockKey(b"overlapping-independent-actions")
    pair.deposit(key, no_evict=False)
    handler = pair.target._core._independent_hint_handler
    resolved = handler.resolved
    manifests = []
    handler.resolved = manifests.append
    receipts = [_submit(pair.target, mode, (123,))[0] for mode in (first, second)]
    pair.until(lambda: len(manifests) == 2)
    assert not _requests(pair, message_type="start_write")
    handler.resolved = resolved
    for manifest in manifests:
        resolved(manifest)
    pair.copy_done()
    reports = []

    def completed():
        reports.extend(_reports(pair.target))
        return len(reports) == 2

    pair.until(completed)

    assert len({receipt.op_handle for receipt in receipts}) == 2
    assert {report.op_handle for report in reports} == {
        receipt.op_handle for receipt in receipts
    }
    assert all(report.status is HintStatus.COMPLETED for report in reports)
    assert all(report.entries[123].keys == (key,) for report in reports)
    assert len(_requests(pair, message_type="start_write")) == 1
    assert pair.target_local[""].raw == b"a" * 16
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    expected = (
        (QueryStatus.MISS, None)
        if "move" in (first, second)
        else (
            QueryStatus.HIT,
            CacheTier.LOCAL_G2,
        )
    )
    pair.until(lambda: pair.source.query((key,)) == [expected])
    assert not pair.results[1]
    _assert_independent_idle(pair)


def test_mixed_action_envelope_preserves_carrier_and_reports_each_action():
    """An invalid/unknown action must not block independent copy/move or advice."""

    class Adapter(_InventoryKeyAdapter):
        def encode_to_router(self, key):
            return key[0]

    pair = _Pair(slots=2, key_adapter_type=Adapter)
    copied, moved = BlockKey(b"a-copy"), BlockKey(b"b-move")
    pair.deposit(copied, no_evict=False)
    pair.deposit(moved, index=1, no_evict=False)
    _reports(pair.target)

    def action(action_id, hashes, *, scope="independent", mode="copy"):
        return _HintAction(
            action_id,
            "kv.fetch",
            "1.0",
            {
                "block_hashes": hashes,
                "source_control_endpoint": "tcp://source:1",
                "scope": scope,
                "mode": mode,
            },
        )

    pair.target.submit_hint(
        _HintEnvelope(
            "0.1",
            "carrier-message",
            [
                action("advice", [987], scope="request"),
                action("copy", [copied[0], copied[0]]),
                action("invalid", [True]),
                action("move", [moved[0]], mode="move"),
                action("unknown", [999]),
            ],
        ),
        "unrelated-carrier",
    )
    admissions = _reports(pair.target)
    assert len(admissions) == 5
    by_action = {result.action_id: result for result in admissions}
    assert by_action["invalid"].status is HintStatus.REJECTED
    assert all(
        by_action[name].status is HintStatus.ACCEPTED
        for name in ("advice", "copy", "move", "unknown")
    )
    remote = pair.target._core._remote_fw_dram
    advice = remote._request_hints["unrelated-carrier"]
    assert advice.block_hashes == frozenset({987})
    pair.copy_done()
    reports = []

    def completed():
        reports.extend(_reports(pair.target))
        return len(reports) == 3

    pair.until(completed)
    outcomes = {result.action_id: result for result in reports}
    assert set(outcomes) == {"copy", "move", "unknown"}
    for name, key in (("copy", copied), ("move", moved)):
        result = outcomes[name]
        assert result.status is HintStatus.COMPLETED
        assert result.entries[key[0]].keys == (key,)
        assert result.op_handle == by_action[name].op_handle
        assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    assert outcomes["unknown"].status is HintStatus.FAILED
    assert outcomes["unknown"].entries[999].reason == "not_found"
    assert all(
        result.request_id == "unrelated-carrier"
        and result.message_id == "carrier-message"
        for result in (*admissions, *reports)
    )
    assert remote._request_hints == {"unrelated-carrier": advice}
    pair.until(lambda: pair.source.query((moved,)) == [(QueryStatus.MISS, None)])
    assert pair.source.query((copied,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert not pair.results[1]
    pair.target.discard_hint("unrelated-carrier")
    _assert_independent_idle(pair)


@pytest.mark.parametrize("mode", ["copy", "move"])
@pytest.mark.parametrize("partial", [False, True], ids=["cached", "partial"])
def test_independent_hint_completes_cached_or_partial_destination(mode, partial):
    """Cached moves invalidate remotely; partial destinations transfer missing data."""
    pair = _Pair(labels=("a", "b"))
    key = BlockKey(b"destination-already-started")
    pair.deposit(key, no_evict=False)
    for label in pair.labels:
        pair.target_primary[label].raw = pair.source_primary[label].raw
    pair.target_agent.state = "DONE"
    deposit = pair.target.deposit(
        {key: pair.descriptors(labels=("a",) if partial else None, endpoint="target")},
        expected_layout=["a", "b"],
    )
    assert dict(_poll_until(pair.target, bool))[deposit][key].success
    _submit(pair.target, mode, (123,))
    pair.copy_done()

    assert _finished(pair).status is HintStatus.COMPLETED
    assert pair.target_local["a"].raw == b"a" * 16
    assert pair.target_local["b"].raw == b"b" * 16
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    assert bool(_requests(pair, message_type="start_write")) is partial
    assert bool(_requests(pair, message_type="invalidate")) is (
        mode == "move" and not partial
    )
    expected = (
        (QueryStatus.MISS, None)
        if mode == "move"
        else (
            QueryStatus.HIT,
            CacheTier.LOCAL_G2,
        )
    )
    pair.until(lambda: pair.source.query((key,)) == [expected])
    assert not pair.results[1]
    _assert_independent_idle(pair)


def test_fetch_setup_error_cleans_private_advice_and_reports_failure(tmp_path):
    """Exercise a real fetch ValueError: a destination G3 layout conflicts."""
    page = os.sysconf("SC_PAGE_SIZE")
    pair = _Pair(
        labels=("a", "b"),
        block_size=page,
        source_agent=_FakeG3Agent(),
        source_g3=G3Options(
            paths=(tmp_path / "cache.data",),
            capacity_bytes_per_file=2 * page,
            backend="MOCK",
        ),
    )
    key = BlockKey(b"conflicting-disk-layout")
    install_recovery_records(
        pair.source._core, {key: _recovered_record(g3=0, g3_pool="a")}
    )
    pair.target_agent.state = "DONE"
    deposit = pair.target.deposit(
        {key: pair.descriptors(endpoint="target")}, expected_layout=["a", "b"]
    )
    assert dict(_poll_until(pair.target, bool))[deposit][key].success
    with pytest.raises(ValueError, match="G3 key layout"):
        pair.source.fetch((key,), expected_layout=["a", "b"])
    _submit(pair.source, "copy", (123,), source="tcp://target:1")

    report = _finished(pair, worker=pair.source)

    assert report.status is HintStatus.FAILED
    assert report.entries[123].reason == "operation_failed"
    assert not _requests(pair, worker="source", message_type="start_write")
    assert pair.source.query((key,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
    _assert_independent_idle(pair, worker=pair.source)


def test_fetch_exception_fails_group_without_escaping_poll(monkeypatch, caplog):
    """A nonfatal admission error warns once, cleans advice, and permits retry."""
    pair = _Pair()
    key = BlockKey(b"full-key")
    pair.deposit(key)
    core = pair.target._core
    fetch = core.fetch

    def failing_fetch(*_args, **_kwargs):
        raise RuntimeError("destination policy failed")

    monkeypatch.setattr(core, "fetch", failing_fetch)
    _LOGGED_HANDLER_ERRORS.discard("fetch:kv.fetch")
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        for _ in range(2):
            _submit(pair.target, "copy", (123,))
            report = _finished(pair)
            assert report.status is HintStatus.FAILED
            assert report.entries[123].reason == "operation_failed"
            assert report.entries[123].keys == (key,)
            assert not _requests(pair, message_type="start_write")
            _assert_independent_idle(pair)
        assert caplog.text.count("KV hint handler failed during fetch") == 1

        monkeypatch.setattr(core, "fetch", fetch)
        _submit(pair.target, "copy", (123,))
        pair.copy_done()
        report = _finished(pair)
    assert report.status is HintStatus.COMPLETED
    assert report.entries[123].keys == (key,)
    assert pair.target_local[""].raw == b"a" * 16
    assert core._block_record_map[key].local_dram.claim_count == 0
    assert caplog.text.count("KV hint handler failed during fetch") == 1
    _assert_independent_idle(pair)


def test_fatal_progress_error_during_fetch_propagates_from_same_poll(monkeypatch):
    """A fatal submit after reservation must escape this poll, not become a miss."""
    pair = _Pair()
    key = BlockKey(b"fatal-after-local-reservation")
    pair.deposit(key)
    core = pair.target._core
    handler = core._independent_hint_handler
    progress = core._progress
    resolved = handler.resolved
    manifests = []
    monkeypatch.setattr(handler, "resolved", manifests.append)
    _submit(pair.target, "copy", (123,))
    pair.until(lambda: len(manifests) == 1)
    monkeypatch.setattr(handler, "resolved", resolved)
    handler._resolved.append(manifests[0])
    failure = RuntimeError("progress failed during fetch admission")
    submit = progress.submit
    attempted = []

    def fail_after_reservation(item):
        if isinstance(item, _TargetPullOp):
            # Exercise real core.fetch admission, then the ordinary sticky-error
            # check in progress.submit; the initial poll check already passed.
            assert core._local_dram._pending_residency_ops
            assert core._block_record_map[key].local_dram.remote_indices is not None
            attempted.append(item)
            progress._failure = failure
        return submit(item)

    monkeypatch.setattr(progress, "submit", fail_after_reservation)
    try:
        with pytest.raises(RuntimeError, match="progress failed during fetch") as error:
            pair.target.poll_completed()
        assert error.value is failure
        assert len(attempted) == 1
        assert not _reports(pair.target)
        assert not _requests(pair, message_type="start_write")
    finally:
        # Fatal progress remains fatal through shutdown; do not clear it to make
        # teardown pass. No native write was submitted, so close can quiesce.
        if progress._failure is failure:
            with pytest.raises(RuntimeError, match="progress failed during fetch"):
                pair.target.close()
            assert core.is_quiescent()
        else:
            pair.target.close()


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_self_addressed_hint_fails_during_resolution(mode):
    """A destination naming its own endpoint as the source is refused by its
    own resolution service before any lookup, transfer, or invalidation.
    """
    pair = _Pair()
    key = BlockKey(b"already-local")
    _cache_at_target(pair, key)
    control = pair.target_control
    forward = control.send

    def loopback(endpoint, message):
        if endpoint != control.endpoint:
            return forward(endpoint, message)
        # Deliver self-addressed control messages back to this process.
        control.incoming.append(message)
        return True

    control.send = loopback
    _submit(pair.target, mode, (123,), source=control.endpoint)
    report = _finished(pair)
    assert report.status is HintStatus.FAILED
    assert report.entries[123].reason == "self_source"
    assert report.entries[123].keys == ()
    assert not _requests(pair, message_type="start_write")
    assert not _requests(pair, message_type="invalidate")
    assert pair.target.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    _assert_independent_idle(pair)


@pytest.mark.parametrize("after_allowance", [False, True])
def test_move_reports_destination_outcome_after_source_removal(after_allowance):
    """A delayed native notification can succeed or time out after source removal."""
    pair = _Pair()
    key = BlockKey(b"late-native-notification")
    pair.deposit(key, no_evict=False)
    held = []
    get_notifications = pair.target_agent.get_new_notifs

    def hold_notifications():
        for notifications in get_notifications().values():
            held.extend(notifications)
        return {}

    pair.target_agent.get_new_notifs = hold_notifications
    _submit(pair.target, "move", (123,))
    pair.copy_done()
    pair.until(lambda: key not in pair.source._core._block_record_map and bool(held))
    assert not _reports(pair.target)
    config = pair.target._core.config
    timeout_ms = (
        config.abandon_timeout_ms if after_allowance else config.operation_timeout_ms
    )
    pair.target._core._clock = lambda: timeout_ms / 1000 + 0.1
    pair.target_agent.get_new_notifs = get_notifications
    pair.target_agent.notifs.setdefault("peer", []).extend(held)

    report = _finished(pair)

    assert report.status is (
        HintStatus.FAILED if after_allowance else HintStatus.COMPLETED
    )
    assert report.entries[123].success is not after_allowance
    assert report.entries[123].reason == (
        "operation_failed" if after_allowance else None
    )
    assert pair.source.query((key,)) == [(QueryStatus.MISS, None)]
    if after_allowance:
        assert pair.target.query((key,)) == [(QueryStatus.MISS, None)]
    else:
        assert pair.target.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
        assert pair.target_local[""].raw == b"a" * 16
        assert pair.target._core._block_record_map[key].local_dram.claim_count == 0
    assert not pair.results[1]
    _assert_independent_idle(pair)
