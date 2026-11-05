# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parse fetch/move advice or dispatch independent hash-addressed operations.

Request advice only records source/hash membership. Independent operations
resolve source keys before starting transfers, even when an unrelated inference
request carries the hint.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import msgspec

from .parser import HintType, _Hint
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore
    from .dispatcher import HintDispatcher

_KV_FETCH_ACTION_TYPE = "kv.fetch"
_KV_FETCH_ACTION_VERSION = "1.0"


class _KVFetchHint(msgspec.Struct, frozen=True):
    """A parsed ``kv.fetch`` hint."""

    ACTION_TYPE: ClassVar[str] = _KV_FETCH_ACTION_TYPE

    action_id: str | None
    source_endpoint: str
    block_hashes: frozenset[int]
    move: bool = False
    scope: str = "request"
    allow_framework: bool = True


def _parse_kv_fetch(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVFetchHint:
    if "no_retain" in payload:
        raise ValueError("no_retain is not currently supported")

    # Fetch uses the mode for matching keys; deliver always copies.
    mode = payload.get("mode", "copy")
    if mode not in ("copy", "move"):
        raise ValueError("kv.fetch mode must be copy or move")

    source = payload.get("source_control_endpoint")
    if source is None:
        raise ValueError("source-less hints are not currently supported")

    block_hashes = payload.get("block_hashes")
    if not isinstance(source, str) or not source or not isinstance(block_hashes, list):
        raise ValueError("invalid router hint")

    hashes: set[int] = set()
    for block_hash in block_hashes:
        if (
            isinstance(block_hash, bool)
            or not isinstance(block_hash, int)
            or not 0 <= block_hash < 1 << 64
        ):
            raise ValueError("invalid router hint")
        hashes.add(block_hash)
    if not hashes:
        raise ValueError("invalid router hint")

    scope = payload.get("scope", "request")
    if scope not in ("request", "independent"):
        raise ValueError("scope must be request or independent")
    allow_framework = payload.get("allow_framework", True)
    if not isinstance(allow_framework, bool):
        raise ValueError("allow_framework must be a boolean")
    return _KVFetchHint(
        action_id,
        source,
        frozenset(hashes),
        move=mode == "move",
        scope=scope,
        allow_framework=allow_framework,
    )


class _FetchHintHandler:
    """Store request advice or dispatch independent fetches to the runtime."""

    action_type = _KVFetchHint.ACTION_TYPE

    def __init__(self, core: _KVCRCore, dispatcher: HintDispatcher) -> None:
        self._core = core
        self._dispatcher = dispatcher

    def submit(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        hint: _KVFetchHint = hint  # type: ignore[assignment]
        receipt = HintResult(
            action_id=hint.action_id,
            action_type=self.action_type,
            status=HintStatus.ACCEPTED,
            request_id=request_id,
            message_id=message_id,
        )
        # With no request, or explicit scope="independent", resolve the supplied
        # hashes and start the transfer. Otherwise keep the existing advice
        # path: the inference engine supplies full keys when it needs them.
        if hint.scope == "independent" or request_id is None:
            self._core._independent_hint_handler.start(hint, receipt)
            return
        # One store call per hint; the store's own rules apply (a later hint
        # replaces the request's hashes, a different source fails the request).
        self._core._remote_fw_dram.submit_hint(
            hint.source_endpoint,
            hint.block_hashes,
            request_id,
            move=hint.move,
            allow_framework=hint.allow_framework,
        )
        self._dispatcher.report(receipt)

    def close(self) -> None:
        self._core._independent_hint_handler.close()


KV_FETCH = HintType(
    _KV_FETCH_ACTION_TYPE,
    _KV_FETCH_ACTION_VERSION,
    _parse_kv_fetch,
    _FetchHintHandler,
)
