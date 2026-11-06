# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Request-scoped source advice that moves blocks when the framework fetches."""

from collections.abc import Mapping
from typing import ClassVar

from .fetch import _FetchHintHandler, _KVFetchHint, _parse_kv_fetch
from .parser import HintType


class _KVMigrateHint(_KVFetchHint):
    ACTION_TYPE: ClassVar[str] = "kv.migrate"


def _parse_kv_migrate(
    action_id: str | None, payload: Mapping[str, object]
) -> _KVMigrateHint:
    hint = _parse_kv_fetch(action_id, payload)
    return _KVMigrateHint(action_id, hint.source_endpoint, hint.block_hashes)


class _MigrateHintHandler(_FetchHintHandler):
    action_type = _KVMigrateHint.ACTION_TYPE


KV_MIGRATE = HintType(
    _KVMigrateHint.ACTION_TYPE, "1.0", _parse_kv_migrate, _MigrateHintHandler
)
