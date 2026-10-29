# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Drop moved source G2 copies after readers and transfers drain."""

from collections.abc import Collection
from typing import TYPE_CHECKING

from .runtime.progress import _FAILED_RESULT
from .types import BlockKey, OpEntryResult

if TYPE_CHECKING:
    from .runtime.core import _BlockRecord, _KVCRCore


class _InvalidationHandler:
    """Remove sources under the owner lock after existing readers drain.
    Submission rejects new reads and deposits; prior protection skips removal.
    """

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core
        # Layouts held by move destinations; overlapping moves can add layouts.
        # None means unrestricted internal removal.
        self._pending: dict[
            BlockKey, tuple["_BlockRecord", frozenset[tuple[str, ...]] | None]
        ] = {}
        # Recheck only on submission or release, not every idle poll.
        self._dirty: dict[BlockKey, None] = {}

    def has_pending(self) -> bool:
        return bool(self._pending)

    def rejected_entries(
        self, keys: Collection[BlockKey]
    ) -> dict[BlockKey, OpEntryResult]:
        """Check new requests once; admitted operations may resume during removal."""
        if not self._pending:
            return {}
        return {key: _FAILED_RESULT for key in keys if key in self._pending}

    def close(self) -> None:
        """Abandon pending removals after progress stops."""
        for record, _ in self._pending.values():
            record.pending_invalidation = False
        self._core._outstanding_operations -= len(self._pending)
        self._pending.clear()
        self._dirty.clear()

    def submit(
        self,
        keys: Collection[BlockKey],
        *,
        key_layout: tuple[str, ...] | None = None,
    ) -> None:
        """Queue removals, preserving no_evict and the destination's full layout."""
        core = self._core
        layouts = None if key_layout is None else frozenset((key_layout,))
        for key in keys:
            record = core._block_record_map.get(key)
            if record is None:
                continue
            if record.local_dram is not None and record.local_dram.retention_handles:
                continue
            if key not in self._pending:
                self._pending[key] = record, layouts
                if core._local_dram is not None:
                    state = core._local_dram._pools._source_state
                    # Close admission atomically with progress acquiring readers.
                    # No slot reclamation or callbacks run under this lock.
                    with state.lock:
                        record.pending_invalidation = True
                        state.ready.pop(key, None)
                else:
                    record.pending_invalidation = True
                core._outstanding_operations += 1
            elif (previous_layouts := self._pending[key][1]) is not None:
                if layouts is None:
                    self._pending[key] = record, None
                elif key_layout not in previous_layouts:
                    self._pending[key] = record, previous_layouts | layouts
            self._dirty[key] = None

    def notify(self, key: BlockKey) -> None:
        if key in self._pending:
            self._dirty[key] = None

    @staticmethod
    def _is_ready(record: "_BlockRecord") -> bool:
        # Submission withdrew the source under the reader lock; its reader count
        # can now only decrease during owner-side completion processing.
        local = record.local_dram
        return not (
            record.in_flight_ops
            or local is not None
            and (
                local.claim_count > len(local.retention_handles or ())
                or local.source_readers
            )
        )

    def drain(self) -> None:
        core = self._core
        dirty, self._dirty = self._dirty, {}
        capacity_changed = False
        for key in dirty:
            # A preceding callback may have removed this key.
            pending = self._pending.get(key)
            if pending is None:
                continue
            record, layouts = pending
            if not self._is_ready(record):
                continue
            # Clear intent before callbacks can redeposit this key or poll again.
            del self._pending[key]
            self._dirty.pop(key, None)
            record.pending_invalidation = False
            core._outstanding_operations -= 1
            if record.local_dram is not None and (
                record.local_dram.retention_handles
                or layouts is not None
                and record.local_dram.key_layout not in layouts
            ):
                # An earlier producer can grant retention or create a layout the
                # destination never held. Keep that copy and reopen admission.
                core._local_dram._pools._publish_source(key, record.local_dram)
                if not record.local_dram.retention_handles:
                    core._local_dram._pools._make_evictable(key)
                    capacity_changed = True
            elif record.local_dram is not None:
                core._local_dram._pools._drop(key)
                capacity_changed = True
            else:
                core._prune_block_record(key)
        if capacity_changed:
            core._local_dram._pools._resume_capacity_waiters()
            core._local_dram._pools._update_capacity_pressure()
