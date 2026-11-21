# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch infrastructure.

``HintDispatcher`` owns the hint type table and reports every hint's
result; core holds one dispatcher and delegates to it. ``_HINT_TYPES`` binds
the supported hint types to their payload parsers and handlers.
"""

from __future__ import annotations

import logging
import queue
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import msgspec

from .hint_handler import _FetchHintHandler, _HintHandler, _NoRetainHintHandler
from .hint_types import (
    MAX_HINT_MESSAGE_BYTES,
    HintType,
    _Hint,
    _KVEvictOnFetchHint,
    _KVFetchHint,
    _KVInvalidateHint,
    _KVMoveHint,
    _KVNoRetainHint,
    _KVPrefetchHint,
)
from .parser import (
    _parse_invalidate,
    _parse_payload,
    decode_hint_envelope,
    hint_message_id,
    parse_hints,
    reject_hint,
)
from .results import HintResult, HintStatus

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

KV_FETCH = HintType(
    _KVFetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="source", hint_factory=_KVFetchHint),
    _FetchHintHandler,
)
KV_EVICT_ON_FETCH = HintType(
    _KVEvictOnFetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="source", hint_factory=_KVEvictOnFetchHint),
    _FetchHintHandler,
)
# Core builds the shared prefetch/move handler before the dispatcher.
KV_PREFETCH = HintType(
    _KVPrefetchHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="destination", hint_factory=_KVPrefetchHint),
    lambda core, dispatcher: core._independent_hint_handler,
)
KV_MOVE = HintType(
    _KVMoveHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer="destination", hint_factory=_KVMoveHint),
    lambda core, dispatcher: core._independent_hint_handler,
)
KV_NO_RETAIN = HintType(
    _KVNoRetainHint.ACTION_TYPE,
    "1.0",
    partial(_parse_payload, peer=None, hint_factory=_KVNoRetainHint),
    _NoRetainHintHandler,
)
KV_INVALIDATE = HintType(
    _KVInvalidateHint.ACTION_TYPE,
    "1.0",
    _parse_invalidate,
    lambda core, dispatcher: core._independent_hint_handler,
)

# The supported hint types. Every dispatcher builds one handler per entry.
_HINT_TYPES: tuple[HintType, ...] = (
    KV_FETCH, KV_EVICT_ON_FETCH, KV_PREFETCH, KV_MOVE, KV_NO_RETAIN, KV_INVALIDATE,
)  # fmt: skip

logger = logging.getLogger(__name__)
# Handlers run on hot paths; warn once per (phase, action type) about errors.
_LOGGED_HANDLER_ERRORS: set[str] = set()
_INBOX_CAPACITY = 256
_POLL_HASH_BUDGET = 1 << 13  # Hashes dispatched per poll; the rest keep their order.
_RAW_BACKLOG_BYTES = 256 << 20  # Control-socket envelopes waiting undecoded.
_POLL_PARSE_BYTES = 1 << 20  # Envelope bytes decoded and parsed per poll, one at least.


def _warn_handler_error(action_type: str, phase: str) -> None:
    """Log the active exception once per ``(phase, action_type)``."""
    issue = f"{phase}:{action_type}"
    if issue in _LOGGED_HANDLER_ERRORS:
        return
    _LOGGED_HANDLER_ERRORS.add(issue)
    logger.warning(
        "KV hint handler failed during %s; action_type=%s",
        phase,
        action_type,
        exc_info=True,
    )


class HintDispatcher:
    """Parse request ingress on the caller, control-socket ingress on progress in
    bounded batches, and dispatch independent hints on progress.

    Independent retention and invalidation take the state lock; source resolution
    uses its reader lock. Close handlers only after progress stops.
    """

    def __init__(self, core: _KVCRCore) -> None:
        self._core = core
        self._types = {hint_type.action_type: hint_type for hint_type in _HINT_TYPES}
        self._handlers: dict[str, _HintHandler] = {
            action_type: hint_type.handler_factory(core, self)
            for action_type, hint_type in self._types.items()
        }
        self._closed = False
        # One batch per envelope, from either ingress path.
        self._inbox: queue.Queue[tuple[list[_Hint], str | None, str | None]] = (
            queue.Queue(_INBOX_CAPACITY)
        )
        self._pending: tuple[list[_Hint], str | None, str | None] | None = None
        # Control-socket envelopes wait undecoded until a poll parses them.
        self._raw: deque[bytes] = deque()
        self._raw_bytes = 0

    def submit_raw(self, message: bytes) -> None:
        """Queue a control-socket envelope; a later poll parses it."""
        if (
            len(message) > MAX_HINT_MESSAGE_BYTES
            or self._raw_bytes + len(message) > _RAW_BACKLOG_BYTES
        ):
            logger.debug("KV hint dropped; ingress limit reached")
            return
        self._raw.append(message)
        self._raw_bytes += len(message)

    def poll(self) -> None:
        """Parse and dispatch bounded batches without blocking native or control
        progress."""
        parse_budget = _POLL_PARSE_BYTES
        while (
            self._raw
            and parse_budget > 0
            and not self._inbox.full()
            and not self._core._progress._close_requested
        ):
            message = self._raw.popleft()
            self._raw_bytes -= len(message)
            parse_budget -= len(message)
            try:
                envelope = decode_hint_envelope(message)
            except Exception:  # Any undecodable input, not only failed validation.
                logger.debug("KV hint envelope dropped: malformed", exc_info=True)
                continue
            self.submit(envelope, None, direct=True)
        budget = _POLL_HASH_BUDGET
        while budget > 0 and not self._core._progress._close_requested:
            try:
                batch = self._pending or self._inbox.get_nowait()
            except queue.Empty:
                break
            hints, request_id, message_id = batch
            self._pending = None
            for index, hint in enumerate(hints):
                locked = isinstance(hint, (_KVNoRetainHint, _KVInvalidateHint))
                if budget <= 0 or (
                    locked and not self._core._state_lock.acquire(blocking=False)
                ):
                    # Keep only the unprocessed tail, ahead of newer inbox work.
                    hints = hints[index:] if index else hints
                    self._pending = hints, request_id, message_id
                    return
                budget -= len(hint.block_hashes)
                try:
                    self._submit_to_handler(hint, request_id, message_id)
                finally:
                    if locked:
                        self._core._state_lock.release()

    def submit(
        self,
        envelope: msgspec.Struct,
        request_id: str | None,
        *,
        direct: bool = False,
    ) -> None:
        """Parse either ingress; only independent actions enter the progress inbox."""
        self._core._progress.raise_if_failed()
        hints, rejected = parse_hints(envelope, self._types)
        for result in rejected:
            self.report(msgspec.structs.replace(result, request_id=request_id))
        message_id = hint_message_id(envelope)
        independent = []
        for hint in hints:
            if isinstance(hint, (_KVPrefetchHint, _KVNoRetainHint, _KVInvalidateHint)):
                independent.append(hint)
            elif direct:
                result = reject_hint(
                    hint.action_id, hint.ACTION_TYPE, "request_scoped_hint", message_id
                )
                self.report(result)
            else:
                self._submit_to_handler(hint, request_id, message_id)
        if independent:
            try:
                self._inbox.put_nowait((independent, request_id, message_id))
            except queue.Full:
                for hint in independent:
                    result = reject_hint(
                        hint.action_id,
                        hint.ACTION_TYPE,
                        "ingress_full",
                        message_id,
                        request_id,
                    )
                    self.report(result)

    def _submit_to_handler(
        self,
        hint: _Hint,
        request_id: str | None,
        message_id: str | None,
    ) -> None:
        # Contract: a handler reports admission exactly once, or raises before
        # doing so. ACCEPTED means admission, not asynchronous completion.
        # A raise means REJECTED, unless it is the fatal progress-thread
        # failure, which propagates to the caller as it does without hints.
        try:
            self._handlers[hint.ACTION_TYPE].submit(hint, request_id, message_id)
        except Exception:
            self._core._progress.raise_if_failed()
            _warn_handler_error(hint.ACTION_TYPE, "submit")
            self.report(
                HintResult(
                    action_id=hint.action_id,
                    action_type=hint.ACTION_TYPE,
                    status=HintStatus.REJECTED,
                    reason="handler_error",
                    request_id=request_id,
                    message_id=message_id,
                )
            )

    # Handlers warn through the dispatcher's once-per-issue logger.
    warn_handler_error = staticmethod(_warn_handler_error)

    def report(self, result: HintResult) -> None:
        """Record a hint's result; called by handlers and by the dispatcher.

        Results are logged at debug level only; rejections and handler errors
        already warn once per issue. No metric is emitted: integrations
        forward KVCR stats and may reject names they do not declare.
        """
        logger.debug(
            "KV hint result; action_id=%s action_type=%s status=%s reason=%s "
            "failures=%s",
            result.action_id,
            result.action_type,
            result.status.value,
            result.reason,
            result.failures,
        )

    # Main thread, under the state lock.

    def close(self) -> None:
        """Close handlers; idempotent."""
        if self._closed:
            return
        self._closed = True
        self._pending = None
        self._raw.clear()
        self._raw_bytes = 0
        while not self._inbox.empty():
            self._inbox.get_nowait()
        closed: set[int] = set()
        for action_type, handler in self._handlers.items():
            if id(handler) in closed:
                continue
            closed.add(id(handler))
            try:
                handler.close()
            except Exception:
                _warn_handler_error(action_type, "close")
