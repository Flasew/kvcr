# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Execute independent hints and collect their asynchronous results."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING
from uuid import uuid4

import msgspec

from ..hints.dispatcher import _warn_handler_error
from ..hints.results import HashResult, HintResult, HintStatus

if TYPE_CHECKING:
    from ..backends.remote_ops import _HashResolutionOp
    from ..hints.fetch import _KVFetchHint
    from ..types import BlockKey, OpEntryResult, OpHandle
    from .core import _KVCRCore


class _IndependentHintHandler:
    """Owner-thread lifecycle for operations started by independent hints."""

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        # Keep generated advice separate from application request IDs based on
        # operation handles. Generate the namespace once, outside the hot path.
        self._advice_prefix = f"kvcr-hint:{uuid4().hex}"
        self._resolved: deque[_HashResolutionOp] = deque()
        self._resolving: dict[OpHandle, tuple[HintResult, _KVFetchHint]] = {}
        # Each child stores (receipt, hash count, key-to-hash map, source advice).
        # Receipts share results; the last child reports the complete outcome.
        self._pending_operations: dict[
            OpHandle, tuple[HintResult, int, dict[BlockKey, int], str | None]
        ] = {}

    def start(self, hint: _KVFetchHint, receipt: HintResult) -> None:
        """Queue resolution; resolved() later starts ordinary full-key fetches."""
        from ..backends.remote_ops import _HashResolutionOp

        core = self._core
        handle = core._next_op_handle
        core._next_op_handle += 1
        receipt = msgspec.structs.replace(receipt, op_handle=handle)
        core._hints_dispatcher.report(receipt)
        if core._local_dram is None or core._remote_fw_dram._key_adapter is None:
            reason = (
                "destination_has_no_managed_dram"
                if core._local_dram is None
                else "key_adapter_unavailable"
            )
            core._hints_dispatcher.report(
                msgspec.structs.replace(
                    receipt,
                    status=HintStatus.FAILED,
                    entries={
                        h: HashResult(False, reason=reason) for h in hint.block_hashes
                    },
                )
            )
            return
        self._resolving[handle] = receipt, hint
        core._progress.submit(
            _HashResolutionOp(
                op_id=("resolve_hashes", handle),
                keys=frozenset(),
                endpoint=hint.source_endpoint,
                hashes=tuple(hint.block_hashes),
                _backend=core._remote_fw_dram,
            )
        )

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
                (h, HashResult(False, reason=resolution.failure))
                for h in resolution.hashes
            )
            self._complete(receipt, hash_count, {}, None, {})
            return
        groups = {}
        adapter = core._remote_fw_dram._key_adapter
        for row in resolution.response:
            reason = row.reason
            if reason is None:
                try:
                    core._validate_expected_layout(list(row.layout))
                    sizes = tuple(
                        core._block_sizes[label.partition(":")[0]]
                        for label in row.layout
                    )
                    if sizes != row.sizes:
                        reason = "incompatible_layout"
                    elif adapter.encode_to_router(row.key) != row.block_hash:
                        reason = "hash_key_mismatch"
                except Exception:
                    # Adapters can raise integration-specific errors for remote
                    # keys. Fail this hash without dropping the command's result.
                    reason = "incompatible_key_or_layout"
            if reason is not None:
                entries[row.block_hash] = HashResult(False, reason=reason)
            else:
                groups.setdefault(row.layout, {})[row.key] = row.block_hash
        if not groups:
            self._complete(receipt, hash_count, {}, None, {})
            return

        for index, (layout, hashes_by_key) in enumerate(groups.items()):
            advice = f"{self._advice_prefix}:{receipt.op_handle}:{index}"
            while advice in core._remote_fw_dram._request_hints:
                advice = f"{self._advice_prefix}:{uuid4().hex}"
            operation = receipt, hash_count, hashes_by_key, advice
            # Private advice is consumed by the fetch below; it has no router
            # lead time, so keep it out of the hint_wait histogram.
            core._remote_fw_dram.submit_hint(
                resolution.endpoint,
                frozenset(hashes_by_key.values()),
                advice,
                move=hint.move,
                timed=False,
            )
            try:
                handle = core.fetch(tuple(hashes_by_key), advice, list(layout))
            except Exception as error:
                # Fatal progress failures retain their ordinary API behavior.
                core._progress.raise_if_failed()
                # Admission failures fail this group and release its advice
                # without losing the command's result or escaping the owner's
                # poll. Layout conflicts are expected; anything else warns once.
                if not isinstance(error, ValueError):
                    _warn_handler_error(hint.ACTION_TYPE, "fetch")
                self._complete(*operation, {})
            else:
                # Synchronous hits are queued until polling, so registration
                # after fetch returns is safe.
                self._pending_operations[handle] = operation

    def finish(self, completed):
        """Consume independent completions and return framework operations."""
        framework = []
        for handle, entries in completed:
            operation = self._pending_operations.pop(handle, None)
            if operation is None:
                framework.append((handle, entries))
                continue
            trace = self._core._nvtx_operations.pop(handle, None)
            if trace is not None:
                trace.returned()
            self._complete(*operation, entries)
        return framework

    def _complete(
        self,
        receipt: HintResult,
        hash_count: int,
        selected: dict[BlockKey, int],
        advice: str | None,
        results: dict[BlockKey, OpEntryResult],
    ) -> None:
        """Release private fetch claims and collect exact-key operation results."""
        if advice is not None:
            self._core.release(
                tuple(
                    result.release_handle
                    for result in results.values()
                    if result.release_handle is not None
                )
            )
            self._core.discard_hint(advice)
        entries = receipt.entries
        assert entries is not None
        for key, block_hash in selected.items():
            result = results.get(key)
            success = result is not None and result.success
            entries[block_hash] = HashResult(
                success, (key,), None if success else "operation_failed"
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
