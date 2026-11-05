# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source hash lookup answers peer metadata requests without moving any bytes."""

import msgspec
import pytest
from _kvcr_test_utils import _decode_control_message, _wait_until
from test_kvcr_move import _Pair

from kvcr.backends.remote_ops import _ResolvedHash, _ResolveHashesRequest
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


@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_source_deadline_is_rejected_without_index_lookup(remaining):
    pair = _Pair()
    lookups = []
    pair.source._core._remote_fw_dram._resolve_hashes = lookups.append
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
    assert not lookups


def test_source_lookup_replies_while_owner_paused_and_state_lock_held():
    """Source RPC needs no owner polling, lock acquisition, or native transfer."""
    pair = _Pair()
    key = BlockKey(b"full-key-with-group")
    pair.deposit(key)
    transfers = len(pair.source_agent.xfers)
    with pair.source._core._state_lock:
        pair.source_control.incoming.append(
            msgspec.msgpack.encode(
                _ResolveHashesRequest(
                    op_handle=9,
                    hashes=(123, 456),
                    remaining_timeout_ms=1000,
                    sender_control_endpoint="tcp://target:1",
                    source_control_endpoint="tcp://source-alias:1",
                )
            )
        )
        _wait_until(lambda: bool(pair.source_control.sent))
        response = _decode_control_message(pair.source_control.sent[-1][1])
    assert response["type"] == "hashes_resolved"
    assert response["sender_control_endpoint"] == "tcp://source-alias:1"
    assert response["results"] == [
        {"block_hash": 123, "key": key, "layout": [""], "sizes": [16], "reason": None},
        {
            "block_hash": 456,
            "key": None,
            "layout": [],
            "sizes": [],
            "reason": "not_found",
        },
    ]
    assert "target_agent_metadata" not in response
    assert "target_regions" not in response
    assert len(pair.source_agent.xfers) == transfers
    assert not pair.target_agent.xfers


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
