# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Track explicit moves and invalidate copies already held at the destination."""

import logging
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import BlockKey, OpHandle

if TYPE_CHECKING:
    from ..runtime.core import _KVCRCore

logger = logging.getLogger(__name__)


@dataclass
class _PendingMove:
    """Named source, keys still to transfer, and complete local hits
    awaiting one invalidation message to the source.
    """

    source: str
    layout: tuple[str, ...]
    keys: set[BlockKey]
    cached_keys: list[BlockKey] = field(default_factory=list)


class _MoveHandler:
    """Start internal full-key moves and invalidate complete destination hits."""

    def __init__(self, core: "_KVCRCore") -> None:
        self._core = core

    def start(
        self,
        keys: Collection[BlockKey],
        source_control_endpoint: str,
        expected_layout: list[str] | None = None,
        hints: object | None = None,
    ) -> OpHandle:
        """Reuse fetch admission without adding move to the framework API."""
        if not isinstance(source_control_endpoint, str) or not source_control_endpoint:
            raise ValueError("move source endpoint must be a non-empty string")
        if source_control_endpoint == getattr(
            self._core.framework_control, "endpoint", None
        ):
            raise ValueError("move source must not be this node")
        return self._core.fetch(
            keys, None, expected_layout, hints, move_source=source_control_endpoint
        )

    def invalidate_cached(self, move: _PendingMove) -> None:
        """Send one invalidation for the complete local hits found so far."""
        from .remote_ops import _Invalidate

        if not move.cached_keys:
            return
        keys = tuple(move.cached_keys)
        move.cached_keys.clear()
        try:
            self._core._progress.submit(
                _Invalidate(
                    keys=keys,
                    key_layout=move.layout,
                    source_control_endpoint=move.source,
                )
            )
        except Exception:
            # Removal is best effort; the destination result stands.
            logger.warning(
                "KVCR cached-key invalidation could not reach %s",
                move.source,
                exc_info=True,
            )
