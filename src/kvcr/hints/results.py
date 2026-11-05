# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Admission and asynchronous cache-operation result types."""

from enum import Enum

import msgspec


class HintStatus(Enum):
    # Admission is distinct from an asynchronous operation's completion.
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"  # invalid, unsupported, handler_error
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HashResult(msgspec.Struct, frozen=True):
    """Outcome for one advertised hash, including its resolved full keys."""

    success: bool
    keys: tuple[bytes, ...] = ()
    reason: str | None = None


class HintResult(msgspec.Struct, frozen=True):
    action_id: str | None
    action_type: str
    status: HintStatus
    reason: str | None = None
    request_id: str | None = None
    message_id: str | None = None
    op_handle: int | None = None
    entries: dict[int, HashResult] | None = None
