# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Move's essential transfer, protection, and admission guarantees."""

import ctypes
import time
from collections import Counter

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakePrimaryPinning,
    _buffer_region,
    _ConstantHashAdapter,
    _has_outstanding_operations,
    _invalidate,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _router_hint,
    _wait_until,
)
from test_kvcr_remote_target import _CopyingWriteAgent

from kvcr.config import LocalDramOptions
from kvcr.types import BlockKey, CacheTier, QueryStatus


class _Peer:
    def __init__(self, name):
        self.name = name
        self.primary = ctypes.create_string_buffer(b"a" * 16 + b"b" * 16, 32)
        self.local = ctypes.create_string_buffer(32)
        self.agent = _CopyingWriteAgent(metadata=name.encode())
        self.control = FakeBytesControl(f"tcp://{name}:1")
        self.events, self.results = [], {}
        self.kvcr = _new_kvcr(
            self.agent,
            FakePrimaryPinning(),
            self.control,
            name=name,
            framework_regions=[
                _buffer_region(self.primary, 16),
                _buffer_region(self.primary, 16, ":*"),
            ],
            local_dram=LocalDramOptions([("", ctypes.addressof(self.local), 32)]),
            inventory_sink=self.events.append,
        )
        self.kvcr._core._clock = lambda: 0.0

    def ref(self, index=0, label=""):
        return _mem_descriptor(index, label=label, end_point_name=self.name)

    def deposit(self, key, index=0, protected=False):
        self.agent.state = "DONE"
        handle = self.kvcr.deposit({key: [self.ref(index)]}, no_evict=protected)
        result = dict(_poll_until(self.kvcr, bool))[handle][key]
        self.agent.state = "PROC"
        assert result.success
        return result.release_handle


class _Pair:
    """Use real buffers; forward control and completed fake-native notifications."""

    def __init__(self):
        self.source, self.target = _Peer("source"), _Peer("target")
        self.offsets = [[0, 0], [0, 0]]
        self.delivered = set()

    def pump(self):
        for index, (peer, other) in enumerate(
            ((self.source, self.target), (self.target, self.source))
        ):
            peer.results.update(peer.kvcr.poll_completed())
            sent, notified = self.offsets[index]
            other.control.incoming.extend(raw for _, raw in peer.control.sent[sent:])
            other.agent.notifs.setdefault("peer", []).extend(
                raw for _, raw in peer.agent.sent_notifs[notified:]
            )
            self.offsets[index] = [len(peer.control.sent), len(peer.agent.sent_notifs)]
            for transfer, data in enumerate(peer.agent.xfers, 1):
                token = index, transfer
                if (
                    token not in self.delivered
                    and transfer in peer.agent.released_xfers
                ):
                    self.delivered.add(token)
                    if data[5] and peer.agent.state == "DONE":
                        other.agent.notifs.setdefault("peer", []).append(data[5])
        time.sleep(0.001)

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.pump()
            if predicate():
                return
        raise AssertionError("move did not reach the expected state")

    def idle(self):
        return not any(
            _has_outstanding_operations(peer.kvcr)
            for peer in (self.source, self.target)
        )


@pytest.fixture
def pair():
    pair = _Pair()
    yield pair
    pair.source.agent.state = pair.target.agent.state = "DONE"
    pair.until(pair.idle)


@pytest.mark.parametrize(
    "cached,protected,succeeds",
    [
        (0, False, True),
        (0, False, False),
        (1, False, True),
        (1, False, False),
        (2, False, True),
        (0, True, True),
        (1, True, True),
        (2, True, True),
    ],
)
def test_move_bytes_removal_protection_and_cache_hits(
    pair, cached, protected, succeeds
):
    source, target = pair.source, pair.target
    keys = (BlockKey(b"first"), BlockKey(b"second"))
    retentions = [source.deposit(key, i, protected) for i, key in enumerate(keys)]
    for index, key in enumerate(keys[:cached]):
        target.deposit(key, index)
    source.events.clear()
    target.events.clear()
    before = len(source.agent.xfers)
    handle = target.kvcr._core._move_handler.start(keys, source.control.endpoint)
    pair.until(lambda: len(source.agent.xfers) > before or handle in target.results)
    if not protected:
        pair.until(lambda: source.kvcr.query(keys) == [(QueryStatus.MISS, None)] * 2)
    else:
        assert source.kvcr.query(keys) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)] * 2
    source.agent.state = "DONE" if succeeds else "ERR"
    pair.until(lambda: handle in target.results)
    # Failed writes report first, then finish native cleanup before reopening.
    source.agent.state = "DONE"
    pair.until(pair.idle)
    results = target.results[handle]
    successful = [key for i, key in enumerate(keys) if i < cached or succeeds]
    assert [key for key in keys if results[key].success] == successful
    assert source.kvcr.query(keys) == [
        (QueryStatus.MISS, None)
        if key in successful and not protected
        else (QueryStatus.HIT, CacheTier.LOCAL_G2)
        for key in keys
    ]
    for index, key in enumerate(keys):
        result = results[key]
        if result.success:
            offset = result.descriptors[0].element_index * 16
            assert (
                target.local.raw[offset : offset + 16]
                == (b"a" if index == 0 else b"b") * 16
            )
            assert target.kvcr.release((result.release_handle,)) == [
                (result.release_handle, True)
            ]
        else:
            assert result.release_handle is None
    for peer, removed, expected in (
        (source, True, successful if not protected else []),
        (target, False, keys[cached:] if succeeds else []),
    ):
        assert Counter(
            key for e in peer.events if e.removed is removed for key in e.keys
        ) == Counter(expected)
    retained = [handle for handle in retentions if handle is not None]
    assert all(ok for _, ok in source.kvcr.release(retained))


@pytest.mark.parametrize("protected", [False, True])
def test_pending_removal_rejects_new_work_and_frees_after_reader(pair, protected):
    source = pair.source
    kvcr = source.kvcr
    key = BlockKey(b"reader")
    source.deposit(key)
    source.deposit(BlockKey(b"other"), 1)
    fetched = kvcr.fetch((key,))
    claim = dict(kvcr.poll_completed())[fetched][key].release_handle
    _invalidate(kvcr, (key, key))
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    handles = [
        kvcr.fetch((key,)),
        kvcr.deliver({key: [source.ref(1)]}),
        kvcr.deposit({key: [source.ref()]}, no_evict=protected),
    ]
    results = dict(kvcr.poll_completed())
    assert all(not results[handle][key].success for handle in handles)
    assert kvcr._core._invalidation_handler.has_pending()
    assert source.local.raw == b"a" * 16 + b"b" * 16
    assert not any(event.removed for event in source.events)
    assert kvcr.release((claim,)) == [(claim, True)]
    assert key not in kvcr._core._block_record_map
    assert not kvcr._core._invalidation_handler.has_pending()
    assert Counter(
        k for event in source.events if event.removed for k in event.keys
    ) == Counter((key,))
    source.deposit(BlockKey(b"replacement"))
    assert kvcr.query((BlockKey(b"replacement"),)) == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2)
    ]


@pytest.mark.parametrize("incomplete", [False, True])
def test_move_rejects_busy_source_without_blocking_deposit(pair, incomplete):
    source, target = pair.source, pair.target
    key = BlockKey(b"busy")
    layout = ["", ":tail"] if incomplete else [""]
    source.agent.state = "DONE" if incomplete else "PROC"
    deposit = source.kvcr.deposit({key: [source.ref()]}, expected_layout=layout)
    _wait_until(lambda: bool(source.agent.xfers))
    if incomplete:
        assert dict(_poll_until(source.kvcr, bool))[deposit][key].success
    moved = target.kvcr._core._move_handler.start(
        (key,), source.control.endpoint, expected_layout=layout
    )
    pair.until(lambda: moved in target.results)
    assert not target.results[moved][key].success
    assert not source.kvcr._core._invalidation_handler.has_pending()
    source.agent.state = "DONE"
    if incomplete:
        deposit = source.kvcr.deposit(
            {key: [source.ref(1, ":tail")]}, expected_layout=layout
        )
    pair.until(lambda: deposit in source.results)
    assert source.results[deposit][key].success
    assert source.kvcr.query((key,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]


def test_invalidation_preserves_an_already_admitted_remote_fetch(pair):
    source, target = pair.source, pair.target
    key = BlockKey(b"remote-reader")
    source.deposit(key)
    target.kvcr._core._remote_fw_dram._key_adapter = _ConstantHashAdapter()
    target.kvcr.submit_hint(_router_hint(source.control.endpoint), "copy")
    before = len(source.agent.xfers)
    copied = target.kvcr.fetch((key,), "copy")
    pair.until(lambda: len(source.agent.xfers) > before)
    residency = source.kvcr._core._block_record_map[key].local_dram
    assert residency.source_readers == 1
    _invalidate(source.kvcr, (key,))
    assert source.kvcr.query((key,)) == [(QueryStatus.MISS, None)]
    assert source.kvcr._core._invalidation_handler.has_pending()
    source.agent.state = "DONE"
    pair.until(lambda: copied in target.results and pair.idle())
    result = target.results[copied][key]
    assert result.success
    assert target.local.raw[:16] == b"a" * 16
    assert key not in source.kvcr._core._block_record_map
    assert target.kvcr.release((result.release_handle,)) == [
        (result.release_handle, True)
    ]
