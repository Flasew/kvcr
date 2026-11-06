# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent copy envelope; execution is not connected yet."""

from collections.abc import Mapping
from typing import ClassVar

from .fetch import _KVFetchHint, _parse_kv_fetch
from .parser import HintType, _Hint, reject_hint


class _KVPrefetchHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.prefetch"


def _parse_kv_prefetch(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVPrefetchHint:
    hint = _parse_kv_fetch(action_id, payload)
    return _KVPrefetchHint(action_id, hint.source_endpoint, hint.block_hashes)


class _PrefetchHintHandler:
    """Recognize the envelope without storing advice or starting transfers."""

    action_type = _KVPrefetchHint.ACTION_TYPE

    def __init__(self, core, dispatcher) -> None:
        self._dispatcher = dispatcher

    def submit(self, hint: _Hint, request_id, message_id) -> None:
        self._dispatcher.report(
            reject_hint(
                hint.action_id, self.action_type, "unsupported", message_id, request_id
            )
        )

    def close(self) -> None:
        pass


KV_PREFETCH = HintType(
    _KVPrefetchHint.ACTION_TYPE, "1.0", _parse_kv_prefetch, _PrefetchHintHandler
)
