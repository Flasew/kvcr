# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Execute independent hints and collect their asynchronous results."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import msgspec

from ..hints.dispatcher import _warn_handler_error
from ..hints.parser import reject_hint
from ..hints.results import HintEntryResult, HintResult, HintStatus

if TYPE_CHECKING:
    from ..backends.remote_ops import _HashResolutionOp
    from ..hints.move import _KVMoveHint
    from ..hints.prefetch import _KVPrefetchHint
    from ..types import BlockKey, OpEntryResult, OpHandle
    from .core import _KVCRCore


class _IndependentHintHandler:
    """Owner-thread lifecycle for operations started by independent hints."""

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._resolved: deque[_HashResolutionOp] = deque()
        self._resolving: dict[
            OpHandle, tuple[HintResult, _KVPrefetchHint | _KVMoveHint]
        ] = {}
        # Each child stores (receipt, hash count, key-to-hash map).
        # Receipts share results; the last child reports the complete outcome.
        self._pending_operations: dict[
            OpHandle, tuple[HintResult, int, dict[BlockKey, int]]
        ] = {}

    def start(
        self,
        hint: _KVPrefetchHint | _KVMoveHint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        """Admit the hint and queue resolution; resolved() starts the transfers."""
        from ..backends.remote_ops import _HashResolutionOp

        core = self._core
        if core._local_dram is None or core._remote_fw_dram._key_adapter is None:
            reason = (
                "destination_has_no_managed_dram"
                if core._local_dram is None
                else "key_adapter_unavailable"
            )
            core._hints_dispatcher.report(
                reject_hint(
                    hint.action_id, hint.ACTION_TYPE, reason, message_id, request_id
                )
            )
            return
        handle = core._next_op_handle
        core._next_op_handle += 1
        # Queue before reporting: a failed submission must raise before any
        # report, and completion only arrives through owner polling afterwards.
        core._progress.submit(
            _HashResolutionOp(
                op_id=("resolve_hashes", handle),
                keys=frozenset(),
                endpoint=hint.source_endpoint,
                hashes=tuple(hint.block_hashes),
                _backend=core._remote_fw_dram,
                on_complete=self._resolved.append,
            )
        )
        receipt = HintResult(
            action_id=hint.action_id,
            action_type=hint.ACTION_TYPE,
            status=HintStatus.ACCEPTED,
            request_id=request_id,
            message_id=message_id,
            op_handle=handle,
        )
        self._resolving[handle] = receipt, hint
        core._hints_dispatcher.report(receipt)

    def poll_resolved(self) -> None:
        """Start resolved transfers outside the core state lock."""
        while self._resolved:
            self.resolved(self._resolved.popleft())

    def resolved(self, resolution: _HashResolutionOp) -> None:
        core = self._core
        receipt, hint = self._resolving.pop(resolution.op_id[1])
        entries = {}
        receipt = msgspec.structs.replace(receipt, entries=entries)
        hash_count = len(resolution.hashes)

        if resolution.failure is not None:
            entries.update(
                (h, HintEntryResult(False, reason=resolution.failure))
                for h in resolution.hashes
            )
            self._complete(receipt, hash_count, {}, {})
            return
        groups = {}
        # Rows share few distinct layouts; validate each one once per reply.
        checked: dict[tuple[str, ...], tuple[int, ...] | None] = {}
        adapter = core._remote_fw_dram._key_adapter
        for row in resolution.response:
            reason = row.reason
            if reason is None:
                if row.layout not in checked:
                    checked[row.layout] = self._layout_sizes(row.layout)
                sizes = checked[row.layout]
                try:
                    if sizes is None:
                        reason = "incompatible_key_or_layout"
                    elif sizes != row.sizes:
                        reason = "incompatible_layout"
                    elif adapter.encode_to_router(row.key) != row.block_hash:
                        reason = "hash_key_mismatch"
                except Exception:
                    # Adapters can raise integration-specific errors for remote
                    # keys. Fail this hash without dropping the command's result.
                    reason = "incompatible_key_or_layout"
            if reason is not None:
                entries[row.block_hash] = HintEntryResult(False, reason=reason)
            else:
                groups.setdefault(row.layout, {})[row.key] = row.block_hash
        if not groups:
            self._complete(receipt, hash_count, {}, {})
            return

        move = hint.ACTION_TYPE == "kv.move"
        for layout, hashes_by_key in groups.items():
            operation = receipt, hash_count, hashes_by_key
            try:
                handle = core.fetch(
                    tuple(hashes_by_key),
                    expected_layout=list(layout),
                    source_endpoint=resolution.endpoint,
                    move=move,
                )
            except Exception as error:
                # Fatal progress failures retain their ordinary API behavior.
                core._progress.raise_if_failed()
                # Fail this group without escaping the owner's poll. Layout
                # conflicts are expected; anything else warns once.
                if not isinstance(error, ValueError):
                    _warn_handler_error(hint.ACTION_TYPE, "admission")
                self._complete(*operation, {})
            else:
                # Synchronous hits are queued until polling, so registration
                # after fetch returns is safe.
                self._pending_operations[handle] = operation

    def _layout_sizes(self, layout: tuple[str, ...]) -> tuple[int, ...] | None:
        """Return piece sizes for a layout this destination can store."""
        core = self._core
        try:
            core._validate_expected_layout(list(layout))
        except ValueError:
            return None
        return tuple(core._block_sizes[label.partition(":")[0]] for label in layout)

    def finish(self, completed):
        """Consume independent completions and return framework operations."""
        framework = []
        for handle, entries in completed:
            operation = self._pending_operations.pop(handle, None)
            if operation is None:
                framework.append((handle, entries))
                continue
            self._complete(*operation, entries)
        return framework

    def _complete(
        self,
        receipt: HintResult,
        hash_count: int,
        selected: dict[BlockKey, int],
        results: dict[BlockKey, OpEntryResult],
    ) -> None:
        """Release private fetch claims and collect exact-key operation results."""
        handles = tuple(
            result.release_handle
            for result in results.values()
            if result.release_handle is not None
        )
        if handles:
            self._core.release(handles)
        entries = receipt.entries
        assert entries is not None
        for key, block_hash in selected.items():
            result = results.get(key)
            success = result is not None and result.success
            entries[block_hash] = HintEntryResult(
                success, key, None if success else "operation_failed"
            )
        if len(entries) == hash_count:
            status = (
                HintStatus.COMPLETED
                if all(result.success for result in entries.values())
                else HintStatus.FAILED
            )
            self._core._hints_dispatcher.report(
                msgspec.structs.replace(receipt, status=status)
            )

    def close(self) -> None:
        self._resolved.clear()
        self._resolving.clear()
        self._pending_operations.clear()
