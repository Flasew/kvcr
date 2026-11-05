# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Copy advertised blocks independently of the carrier request."""

from collections.abc import Mapping
from typing import ClassVar

from .fetch import _KVFetchHint, _parse_kv_fetch
from .parser import HintType, _Hint


class _KVPrefetchHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.prefetch"


def _parse_kv_prefetch(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVPrefetchHint:
    hint = _parse_kv_fetch(action_id, payload)
    return _KVPrefetchHint(action_id, hint.source_endpoint, hint.block_hashes)


class _PrefetchHintHandler:
    """Dispatch hash-addressed operations to the independent runtime."""

    action_type = _KVPrefetchHint.ACTION_TYPE

    def __init__(self, core, dispatcher) -> None:
        self._core = core

    def submit(self, hint: _Hint, request_id, message_id) -> None:
        self._core._independent_hint_handler.start(hint, request_id, message_id)

    def close(self) -> None:
        pass


KV_PREFETCH = HintType(
    _KVPrefetchHint.ACTION_TYPE, "1.0", _parse_kv_prefetch, _PrefetchHintHandler
)
