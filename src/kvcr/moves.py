# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Carry move intent with ordinary fetch; remove source copies independently."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .runtime.progress import _TransferRef
from .types import BlockKey, OpEntryResult, OpHandle

if TYPE_CHECKING:
    from .backends.remote_fw_dram import _RequestHint
    from .backends.remote_ops import _StartWrite
    from .runtime.core import _KVCRCore


@dataclass
class _PendingMove:
    # Admission snapshot shared by every fill of this public move.
    hint: "_RequestHint"
    layout: tuple[str, ...]
    keys: frozenset[BlockKey]
    # Keys whose removal intent was queued, avoiding duplicates across fills.
    requested_keys: set[BlockKey] = field(default_factory=set)


class _MoveHandler:
    """Carry removal intent through fetch without delaying destination completion.
    The source invalidates after copying, or when the destination has the data.
    """

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core
        # Public destination operations can launch several internal fills.
        self._pending: dict[OpHandle, _PendingMove] = {}

    def begin(
        self,
        handle: OpHandle,
        keys: tuple[BlockKey, ...],
        request_id: str | None,
        layout: tuple[str, ...],
        *,
        force: bool = False,
    ) -> bool:
        if not keys:
            return True
        remote = self._core._remote_fw_dram
        hint = remote._request_hints.get(request_id)
        adapter = remote._key_adapter
        if hint is None or not hint.source or hint.failed or adapter is None:
            return False
        selected = frozenset(
            key for key in keys if adapter.decode(key) in hint.block_hashes
        )
        # Even a destination cache hit needs advice naming the source to remove.
        if force and len(selected) != len(keys):
            return False
        if selected:
            self._pending[handle] = _PendingMove(hint, layout, selected)
        return True

    def start_fill(
        self,
        handle: OpHandle,
        fill_handle: OpHandle,
        blocks: Mapping[BlockKey, list[_TransferRef]],
        request_id: str | None,
        deadline: float,
    ) -> bool:
        """Prepare move intent, then submit through the ordinary transfer path."""
        move = self._pending[handle]
        fetch = self._core._local_dram._pending_residency_ops[("fetch", handle)]
        # Successful local results already hold complete copies with fetch claims.
        cached_keys = tuple(
            key
            for key, result in fetch.results.items()
            if result.success and key in move.keys and key not in move.requested_keys
        )
        keys_to_move = tuple(key for key in blocks if key in move.keys) + cached_keys
        started = self._core._remote_fw_dram._start_target_pull(
            blocks,
            request_id,
            deadline,
            fill_handle,
            local_fill=True,
            source_hint=move.hint,
            keys_to_move=keys_to_move,
            cached_keys=cached_keys,
        )
        if started:
            # Failed submission leaves cached-key intent available to complete().
            move.requested_keys.update(keys_to_move)
        return started

    @staticmethod
    def validate_request(request: "_StartWrite") -> bool:
        keys, selected = set(request.keys), set(request.move_keys)
        if (
            not selected
            or len(selected) != len(request.move_keys)
            or not selected.issubset(keys)
            or not request.key_layout
        ):
            return False
        if not request.dst_descriptors:
            return selected == keys
        return len(request.keys) == len(request.dst_descriptors) and all(
            descriptors or key in selected
            for key, descriptors in zip(request.keys, request.dst_descriptors)
        )

    def complete(
        self, handle: OpHandle, entries: Mapping[BlockKey, OpEntryResult]
    ) -> None:
        """Finish move bookkeeping and queue any unsent cache-hit intent."""
        from .backends.remote_ops import _StartWrite

        move = self._pending.pop(handle)
        held = tuple(
            key
            for key, result in entries.items()
            if result.success and key in move.keys and key not in move.requested_keys
        )
        if held:
            # Cache hits still need a source command, but no completion reply.
            self._core._progress.submit(
                _StartWrite(
                    op_handle=handle,
                    remaining_timeout_ms=self._core.config.operation_timeout_ms,
                    keys=held,
                    dst_descriptors=(),
                    key_layout=move.layout,
                    move_keys=held,
                    source_control_endpoint=move.hint.source,
                )
            )

    def source_cached(self, request: "_StartWrite") -> None:
        """Remove eligible source copies already held at the destination."""
        keys = []
        for index, key in enumerate(request.keys):
            if request.dst_descriptors and request.dst_descriptors[index]:
                continue
            record = self._core._block_record_map.get(key)
            if (
                record is not None
                and record.local_dram is not None
                and record.local_dram.key_layout != request.key_layout
            ):
                continue
            keys.append(key)
        self._core._invalidation_handler.submit(keys, key_layout=request.key_layout)

    def close(self) -> None:
        self._pending.clear()
