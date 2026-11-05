# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Move advertised blocks independently of the carrier request."""

from collections.abc import Mapping
from typing import ClassVar

from .fetch import _KVFetchHint, _parse_kv_fetch
from .parser import HintType
from .prefetch import _PrefetchHintHandler


class _KVMoveHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.move"


def _parse_kv_move(action_id: str | None, payload: Mapping[str, object]) -> _KVMoveHint:
    hint = _parse_kv_fetch(action_id, payload)
    return _KVMoveHint(action_id, hint.source_endpoint, hint.block_hashes)


class _MoveHintHandler(_PrefetchHintHandler):
    action_type = _KVMoveHint.ACTION_TYPE


KV_MOVE = HintType(_KVMoveHint.ACTION_TYPE, "1.0", _parse_kv_move, _MoveHintHandler)
