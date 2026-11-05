# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise hash operations with two real KVCR/NIXL/ZMQ worker processes.

Requires the native NIXL runtime; no inference engine or router is required.
Hints cross a JSON envelope boundary and use unrelated request IDs and full
framework keys with two named pieces. Race tests gate the owner continuation
after a real resolution reply; they never rely on sleeps to order the race.
Run with PYTHONPATH=src and an output directory outside the checkout.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import socket
import time
import traceback
from pathlib import Path

import msgspec

import kvcr as kvcr_package
from kvcr import KVCR, KVCRBindings, ZmqPeerControlChannel
from kvcr.config import (
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramOptions,
    RemoteFWDramOptions,
)
from kvcr.types import BlockKey, MemoryRef, RegionDescriptor


class Action(msgspec.Struct, frozen=True):
    action_id: str
    action_type: str
    action_version: str
    payload: dict[str, object]


class Envelope(msgspec.Struct, frozen=True):
    protocol_version: str
    message_id: str
    actions: list[Action]


class Adapter:
    def __init__(self):
        self._keys = {}

    def decode_from_framework(self, key):
        return BlockKey(key)

    def encode_to_framework(self, key):
        return key

    def encode_to_router(self, key):
        assert len(key) == 36
        return int.from_bytes(key[:-4], "big") & ((1 << 64) - 1)

    def decode_from_router(self, block_hash):
        keys = self._keys[block_hash]
        if len(keys) != 1:
            raise ValueError("ambiguous hash")
        return keys[0]

    def update_inventory(self, event):
        # This harness configures managed G2 only.
        for key in event.keys:
            block_hash = self.encode_to_router(key)
            keys = self._keys.get(block_hash, ())
            if event.removed:
                keys = tuple(known for known in keys if known != key)
            elif key not in keys:
                keys = (*keys, key)
            if keys:
                self._keys[block_hash] = keys
            else:
                self._keys.pop(block_hash, None)


def key_for(tag, group=0):
    return BlockKey(hashlib.sha256(tag.encode()).digest() + group.to_bytes(4, "big"))


def content(key, label, size):
    pattern = hashlib.sha256(key + label.encode()).digest()
    return (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]


def worker(connection, size, slots):
    cache = channel = None
    try:
        labels = ("k", "v")
        primary = {label: ctypes.create_string_buffer(size) for label in labels}
        local = {label: ctypes.create_string_buffer(slots * size) for label in labels}
        name = f"hash-operations-{os.getpid()}"
        refs = [
            MemoryRef(end_point_name=name, label=label, element_index=0)
            for label in labels
        ]
        framework_key = None
        pin_requests, pin_results, pins = [], [], set()

        def request_pin(keys):
            # A minimal framework callback, backed by registered real buffers.
            # Only disappearance tests opt in; all other cases forbid fallback.
            assert framework_key is not None, f"unexpected framework pin: {keys}"
            pin_requests.append(tuple(keys))
            request_id = len(pin_requests)
            handle = f"framework-pin-{request_id}"
            pins.add(handle)
            pin_results.append(
                (
                    request_id,
                    (handle, {k: refs if k == framework_key else None for k in keys}),
                )
            )
            return request_id

        def poll_pins():
            results = list(pin_results)
            pin_results.clear()
            return results

        def release_pin(handle):
            pins.remove(handle)
            return True

        # Keep the ephemeral port reserved until ZMQ takes ownership. Closing a
        # probe socket first lets concurrent workers/tests steal the same port.
        with socket.create_server(("127.0.0.1", 0)) as listener:
            port = listener.getsockname()[1]
            channel = ZmqPeerControlChannel("127.0.0.1", port, "127.0.0.1")
            channel.adopt_listener(listener.detach())
        adapter = Adapter()
        cache = KVCR(
            KVCRConfig(
                nixl_agent_name=name,
                nixl_listen_port=0,
                pool_layouts=[(label, size) for label in labels],
                operation_timeout_ms=15000,
                abandon_timeout_ms=30000,
            ),
            KVCRBindings(
                request_pin,
                poll_pins,
                release_pin,
                framework_control=channel,
                key_adapter=adapter,
                inventory_sink=adapter.update_inventory,
            ),
            KVCRBackendConfigs(
                framework_regions=[
                    RegionDescriptor(
                        addr=ctypes.addressof(buffer), size=size, label=label
                    )
                    for label, buffer in primary.items()
                ],
                local_dram=LocalDramOptions(
                    [
                        (label, ctypes.addressof(buffer), slots * size)
                        for label, buffer in local.items()
                    ]
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
        )
        core = cache._core
        completed, reports = {}, {}
        original_report = core._hints_dispatcher.report

        def observe_report(result):
            # Hint results currently have an internal report sink, not a public
            # completion API. Observe that sink without changing its behavior.
            reports.setdefault(result.action_id, []).append(
                {
                    "status": result.status.value,
                    "message_id": result.message_id,
                    "request_id": result.request_id,
                    "op_handle": result.op_handle,
                    "entries": {
                        h: {
                            "success": entry.success,
                            "keys": entry.keys,
                            "reason": entry.reason,
                        }
                        for h, entry in (result.entries or {}).items()
                    },
                }
            )
            original_report(result)

        core._hints_dispatcher.report = observe_report
        handler = core._independent_hint_handler
        original_resolved = handler.resolved
        held_resolutions = []
        pause_resolutions = False

        def resolved(op):
            if pause_resolutions:
                held_resolutions.append(op)
            else:
                original_resolved(op)

        handler.resolved = resolved
        connection.send({"result": {"endpoint": channel.endpoint, "pid": os.getpid()}})
        while True:
            completed.update(cache.poll_completed())
            if not connection.poll(0.001):
                continue
            command, args = connection.recv()
            key = args.get("key")
            if command == "close":
                cache.close()
                cache = None
                channel.close()
                channel = None
                connection.send({"result": None})
                return
            if command == "deposit":
                for label, buffer in primary.items():
                    buffer.raw = content(key, label, size)
                result = cache.deposit(
                    {key: refs},
                    no_evict=args.get("no_evict", True),
                    expected_layout=list(labels),
                )
            elif command == "fetch":
                result = cache.fetch((key,), expected_layout=list(labels))
            elif command == "deliver":
                for buffer in primary.values():
                    buffer.raw = bytes(size)
                result = cache.deliver({key: refs})
            elif command == "verify":
                result = all(
                    buffer.raw == content(key, label, size)
                    for label, buffer in primary.items()
                )
            elif command == "verify_reader":
                held_key, residency, _ = core._local_dram._pools._public_claims[
                    args["handle"]
                ]
                assert held_key == key
                result = all(
                    ctypes.string_at(ctypes.addressof(local[label]) + slot * size, size)
                    == content(key, label, size)
                    for label, slot in residency.slots
                )
            elif command == "resident":
                record = core._block_record_map.get(key)
                result = record is not None and record.local_dram is not None
            elif command == "lookup_hash":
                try:
                    result = {"key": adapter.decode_from_router(args["block_hash"])}
                except KeyError:
                    result = {"reason": "not_found"}
                except ValueError:
                    result = {"reason": "ambiguous_hash"}
            elif command == "result":
                entries = completed.pop(args["handle"], None)
                result = (
                    None
                    if entries is None
                    else {
                        key: (entry.success, entry.release_handle)
                        for key, entry in entries.items()
                    }
                )
            elif command == "query":
                result = [
                    (status.value, tier.value if tier else None)
                    for status, tier in cache.query((key,), args.get("request_id"))
                ]
            elif command == "release":
                result = cache.release(args["handles"])
            elif command == "hint":
                cache.submit_hint(
                    msgspec.json.decode(args["envelope"], type=Envelope),
                    args.get("request_id"),
                )
                result = None
                if args.get("carrier_key") is not None:
                    result = [
                        (status.value, tier.value if tier else None)
                        for status, tier in cache.query(
                            (args["carrier_key"],), args["request_id"]
                        )
                    ]
                if args.get("discard"):
                    # No owner poll can run between submission and request end.
                    cache.discard_hint(args["request_id"])
            elif command == "reports":
                result = reports.get(args["action_id"], [])
            elif command == "pause_resolutions":
                pause_resolutions = True
                result = None
            elif command == "resolutions":
                result = [
                    [(row.block_hash, row.key, row.reason) for row in op.response]
                    for op in held_resolutions
                ]
            elif command == "resume_resolutions":
                pause_resolutions = False
                pending, held_resolutions = held_resolutions, []
                for op in pending:
                    original_resolved(op)
                result = len(pending)
            elif command == "invalidate":
                # There is no public invalidation API. Exercise the same owner
                # mechanism used by moves, while the destination gate is closed.
                with core._state_lock:
                    result = core._invalidation_handler.submit((key,))
            elif command == "framework_key":
                framework_key = key
                for label, buffer in primary.items():
                    buffer.raw = content(key, label, size)
                result = None
            elif command == "pin_requests":
                result = pin_requests
            elif command == "source_cleanup_done":
                result = not (
                    core._remote_fw_dram._sources._source_pin_ops
                    or core._local_dram_sources_by_op
                    or core._local_dram._pools.source_operations
                    or not core._local_dram._pools._source_state.completions.empty()
                )
            elif command == "idle":
                completed.update(cache.poll_completed())
                result = {
                    "public_completions": len(completed),
                    "operations": core._outstanding_operations,
                    "resolving": len(handler._resolving),
                    "resolved": len(handler._resolved),
                    "children": len(handler._pending_operations),
                    "held_resolutions": len(held_resolutions),
                    "advice": len(core._remote_fw_dram._request_hints),
                    "pins": len(pins),
                    "claims": sum(
                        record.local_dram.claim_count
                        for record in core._block_record_map.values()
                        if record.local_dram is not None
                    ),
                }
            else:
                raise ValueError(command)
            connection.send({"result": result})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        try:
            if cache is not None:
                cache.close()
        finally:
            try:
                if channel is not None:
                    channel.close()
            finally:
                connection.close()


class Workers:
    def __init__(self, args):
        self.timeout = args.timeout
        self.processes, self.connections, self.info = [], [], []
        self.sequence = 0
        context = multiprocessing.get_context("spawn")
        try:
            for _ in range(2):
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker, args=(child, args.block_size, 4)
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            self.info = [self.receive(rank) for rank in range(2)]
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def receive(self, rank):
        connection = self.connections[rank]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"worker {rank} did not answer")
        reply = connection.recv()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call(self, rank, command, **args):
        self.connections[rank].send((command, args))
        return self.receive(rank)

    def until(self, predicate):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.001)
        raise TimeoutError("operation did not complete")

    def operation(self, rank, command, key, **kwargs):
        handle = self.call(rank, command, key=key, **kwargs)
        entries = self.until(lambda: self.call(rank, "result", handle=handle))
        assert entries[key][0], entries
        return entries[key][1]

    def hint(
        self,
        rank,
        mode,
        hashes,
        *,
        request_id="carrier",
        scope="independent",
        carrier_key=None,
        discard=False,
    ):
        self.sequence += 1
        action_id = f"action-{self.sequence}"
        payload = {
            "block_hashes": list(hashes),
            "mode": mode,
            "scope": scope,
            "source_control_endpoint": self.info[1 - rank]["endpoint"],
        }
        carrier_result = self.call(
            rank,
            "hint",
            envelope=msgspec.json.encode(
                Envelope(
                    "0.1", action_id, [Action(action_id, "kv.fetch", "1.0", payload)]
                )
            ),
            request_id=request_id,
            carrier_key=carrier_key,
            discard=discard,
        )
        return action_id, carrier_result

    def terminal(self, rank, action_id, status="COMPLETED"):
        def done():
            rows = self.call(rank, "reports", action_id=action_id)
            assert len(rows) <= 2, f"duplicate reports: {rows}"
            return rows if len(rows) == 2 else None

        rows = self.until(done)
        assert [row["status"] for row in rows] == ["ACCEPTED", status], rows
        return rows[-1]["entries"]

    def delivered(self, rank, key):
        self.operation(rank, "deliver", key)
        assert self.call(rank, "verify", key=key), key.hex()

    def idle(self, rank):
        state = {}

        def drained():
            state.update(self.call(rank, "idle"))
            assert state["public_completions"] == 0, state
            return not any(state.values())

        try:
            self.until(drained)
        except TimeoutError:
            raise AssertionError(f"worker {rank} did not drain: {state}") from None

    def close(self):
        errors = []
        try:
            for rank, process in enumerate(self.processes):
                if process.is_alive():
                    try:
                        self.call(rank, "close")
                    except BaseException as exc:
                        errors.append(f"worker {rank} close: {exc!r}")
                else:
                    errors.append(f"worker {rank} exited before close")
        finally:
            for rank, process in enumerate(self.processes):
                process.join(timeout=5)
                if process.is_alive():
                    errors.append(f"worker {rank} hung during close")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                if process.exitcode != 0:
                    errors.append(f"worker {rank} exit code: {process.exitcode}")
            for connection in self.connections:
                connection.close()
        if errors:
            raise RuntimeError("; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, _type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is not None:
                raise exc.with_traceback(tb) from cleanup_error
            raise


def unrelated_transfer(args, mode, *, protected=False):
    key, carrier = key_for(mode), key_for("unrelated-request")
    with Workers(args) as workers:
        retained = workers.operation(0, "deposit", key, no_evict=protected)
        reader = workers.operation(0, "fetch", key)
        workers.hint(1, "copy", (Adapter().encode_to_router(carrier),), scope="request")
        action, advice = workers.hint(
            1,
            mode,
            (Adapter().encode_to_router(key),),
            carrier_key=carrier,
            discard=True,
        )
        assert advice == [("FETCHABLE", "REMOTE_G2")]
        entries = workers.terminal(1, action)
        assert entries[Adapter().encode_to_router(key)] == {
            "success": True,
            "keys": (key,),
            "reason": None,
        }
        assert workers.call(1, "query", key=carrier, request_id="carrier") == [
            ("MISS", None)
        ]
        workers.until(lambda: workers.call(0, "source_cleanup_done"))
        # Inspect the existing reader's memory without acquiring a new read.
        assert workers.call(0, "verify_reader", key=key, handle=reader)
        if mode == "move" and not protected:
            workers.until(lambda: workers.call(0, "query", key=key) == [("MISS", None)])
            assert workers.call(0, "resident", key=key)
        else:
            workers.delivered(0, key)
        workers.delivered(1, key)
        assert workers.call(0, "release", handles=[reader]) == [(reader, True)]
        if mode == "move" and not protected:
            workers.until(lambda: not workers.call(0, "resident", key=key))
            assert workers.call(0, "query", key=key) == [("MISS", None)]
        else:
            assert workers.call(0, "query", key=key) == [("HIT", "DRAM")]
        block_hash = Adapter().encode_to_router(key)
        assert workers.call(0, "lookup_hash", block_hash=block_hash) == (
            {"reason": "not_found"}
            if mode == "move" and not protected
            else {"key": key}
        )
        assert workers.call(1, "lookup_hash", block_hash=block_hash) == {"key": key}
        if protected:
            assert workers.call(0, "release", handles=[retained]) == [(retained, True)]
            workers.delivered(0, key)
        workers.delivered(1, key)
        for rank in range(2):
            workers.idle(rank)


def unrelated_copy(args):
    unrelated_transfer(args, "copy")


def unrelated_move(args):
    unrelated_transfer(args, "move")


def protected_move(args):
    unrelated_transfer(args, "move", protected=True)


def ambiguous_source(args, keys):
    block_hash = Adapter().encode_to_router(keys[0])
    assert len(set(keys)) == 2 and {Adapter().encode_to_router(k) for k in keys} == {
        block_hash
    }
    with Workers(args) as workers:
        retained = [workers.operation(0, "deposit", k) for k in keys]
        for mode in ("copy", "move"):
            handle, _ = workers.hint(1, mode, (block_hash,))
            assert workers.terminal(1, handle, "FAILED")[block_hash] == {
                "success": False,
                "keys": (),
                "reason": "ambiguous_hash",
            }
        for k in keys:
            assert workers.call(1, "query", key=k) == [("MISS", None)]
            workers.delivered(0, k)
        assert workers.call(0, "release", handles=retained) == [
            (h, True) for h in retained
        ]
        for rank in range(2):
            workers.idle(rank)


def group_collision(args):
    ambiguous_source(args, (key_for("collision"), key_for("collision", group=1)))


def truncated_hash_collision(args):
    key = key_for("collision")
    ambiguous_source(args, (key, BlockKey(bytes([key[0] ^ 1]) + key[1:])))


def destination_collision(args):
    incoming, existing = key_for("destination"), key_for("destination", group=1)
    assert incoming != existing and Adapter().encode_to_router(
        incoming
    ) == Adapter().encode_to_router(existing)
    with Workers(args) as workers:
        workers.operation(0, "deposit", incoming, no_evict=False)
        target_retention = workers.operation(1, "deposit", existing)
        action, _ = workers.hint(1, "move", (Adapter().encode_to_router(incoming),))
        result = workers.terminal(1, action)
        assert result[Adapter().encode_to_router(incoming)]["keys"] == (incoming,)
        workers.until(
            lambda: workers.call(0, "query", key=incoming) == [("MISS", None)]
        )
        for key in (incoming, existing):
            assert workers.call(1, "query", key=key) == [("HIT", "DRAM")]
            workers.delivered(1, key)
        assert workers.call(1, "release", handles=[target_retention]) == [
            (target_retention, True)
        ]
        for rank in range(2):
            workers.idle(rank)


def mixed_hash_batch(args, mode):
    """A missing or ambiguous hash must not suppress a valid key in the batch."""
    live, collision = key_for("live"), key_for("ambiguous")
    other = key_for("ambiguous", group=1)
    project = Adapter().encode_to_router
    live_hash, ambiguous, missing = map(project, (live, collision, key_for("missing")))
    with Workers(args) as workers:
        for key in (live, collision, other):
            workers.operation(0, "deposit", key, no_evict=False)
        action, _ = workers.hint(1, mode, (live_hash, missing, ambiguous, live_hash))
        entries = workers.terminal(1, action, "FAILED")
        assert entries == {
            live_hash: {"success": True, "keys": (live,), "reason": None},
            missing: {"success": False, "keys": (), "reason": "not_found"},
            ambiguous: {"success": False, "keys": (), "reason": "ambiguous_hash"},
        }
        workers.delivered(1, live)
        for key in (collision, other):
            workers.delivered(0, key)
            assert workers.call(1, "query", key=key) == [("MISS", None)]
        for rank in range(2):
            workers.idle(rank)
        assert workers.call(0, "query", key=live) == (
            [("MISS", None)] if mode == "move" else [("HIT", "DRAM")]
        )


def cached_destination(args, mode):
    """A complete destination hit still performs move's source removal."""
    key = key_for("already-cached")
    with Workers(args) as workers:
        for rank in range(2):
            workers.operation(rank, "deposit", key, no_evict=False)
        reader = workers.operation(1, "fetch", key)
        action, _ = workers.hint(1, mode, (Adapter().encode_to_router(key),))
        assert workers.terminal(1, action)[Adapter().encode_to_router(key)]["success"]
        assert workers.call(1, "verify_reader", key=key, handle=reader)
        assert workers.call(1, "release", handles=[reader]) == [(reader, True)]
        workers.delivered(1, key)
        for rank in range(2):
            workers.idle(rank)
        assert workers.call(0, "query", key=key) == (
            [("MISS", None)] if mode == "move" else [("HIT", "DRAM")]
        )


def source_disappears(args, mode):
    """Revalidate a resolved key after G2 removal using ordinary framework fallback."""
    key = key_for("disappearing")
    block_hash = Adapter().encode_to_router(key)
    with Workers(args) as workers:
        workers.operation(0, "deposit", key, no_evict=False)
        workers.call(1, "pause_resolutions")
        action, _ = workers.hint(1, mode, (block_hash,), discard=True)
        rows = workers.until(lambda: workers.call(1, "resolutions"))
        assert rows == [[(block_hash, key, None)]]
        workers.call(0, "invalidate", key=key)
        workers.until(lambda: not workers.call(0, "resident", key=key))
        assert workers.call(0, "lookup_hash", block_hash=block_hash) == {
            "reason": "not_found"
        }
        workers.call(0, "framework_key", key=key)
        assert workers.call(1, "resume_resolutions") == 1
        assert workers.terminal(1, action)[block_hash] == {
            "success": True,
            "keys": (key,),
            "reason": None,
        }
        assert workers.call(0, "pin_requests") == [(key,)]
        workers.delivered(1, key)
        for rank in range(2):
            workers.idle(rank)


def collision_recovers(args):
    """Inventory removal restores uniqueness and permits a subsequent transfer."""
    key, other = key_for("recover"), key_for("recover", group=1)
    block_hash = Adapter().encode_to_router(key)
    with Workers(args) as workers:
        for item in (key, other):
            workers.operation(0, "deposit", item, no_evict=False)
        action, _ = workers.hint(1, "copy", (block_hash,))
        assert workers.terminal(1, action, "FAILED")[block_hash]["reason"] == (
            "ambiguous_hash"
        )
        workers.call(0, "invalidate", key=other)
        workers.until(lambda: not workers.call(0, "resident", key=other))
        assert workers.call(0, "lookup_hash", block_hash=block_hash) == {"key": key}
        action, _ = workers.hint(1, "move", (block_hash,))
        assert workers.terminal(1, action)[block_hash]["keys"] == (key,)
        workers.delivered(1, key)
        for rank in range(2):
            workers.idle(rank)
        assert workers.call(0, "lookup_hash", block_hash=block_hash) == {
            "reason": "not_found"
        }


def mixed_action_envelope(args):
    """One carrier can contain request advice, independent work, and bad actions.

    This is the router-facing wire contract, built locally without Dynamo or
    vLLM imports. Ending the carrier before the next owner poll must leave both
    independent operations alive and their result identities intact.
    """
    copied, moved, carrier, missing = map(
        key_for, ("copied", "moved", "carrier", "absent")
    )
    project = Adapter().encode_to_router
    with Workers(args) as workers:
        for key in (copied, moved):
            workers.operation(0, "deposit", key, no_evict=False)

        def action(action_id, key, mode="copy", scope="independent"):
            return Action(
                action_id,
                "kv.fetch",
                "1.0",
                {
                    "block_hashes": [project(key)],
                    "source_control_endpoint": workers.info[0]["endpoint"],
                    "scope": scope,
                    "mode": mode,
                },
            )

        envelope = Envelope(
            "0.1",
            "carrier-request",
            [
                action("a1", carrier, scope="request"),
                action("copy", copied),
                action("move", moved, mode="move"),
                action("missing", missing),
                action("invalid", carrier, scope="invalid"),
            ],
        )
        assert workers.call(
            1,
            "hint",
            envelope=msgspec.json.encode(envelope),
            request_id="carrier-request",
            carrier_key=carrier,
            discard=True,
        ) == [("FETCHABLE", "REMOTE_G2")]
        for action_id, key in (("copy", copied), ("move", moved)):
            assert workers.terminal(1, action_id)[project(key)]["keys"] == (key,)
            workers.delivered(1, key)
        assert workers.terminal(1, "missing", "FAILED")[project(missing)]["reason"] == (
            "not_found"
        )
        handles = []
        for action_id in ("a1", "copy", "move", "missing", "invalid"):
            rows = workers.call(1, "reports", action_id=action_id)
            assert all(row["message_id"] == "carrier-request" for row in rows)
            assert all(row["request_id"] == "carrier-request" for row in rows)
            if action_id in ("a1", "invalid"):
                assert [row["status"] for row in rows] == [
                    "ACCEPTED" if action_id == "a1" else "REJECTED"
                ]
            else:
                assert rows[0]["op_handle"] == rows[1]["op_handle"]
                handles.append(rows[0]["op_handle"])
        assert len(set(handles)) == 3
        for rank in range(2):
            workers.idle(rank)
        assert workers.call(0, "query", key=moved) == [("MISS", None)]
        assert workers.call(0, "query", key=copied) == [("HIT", "DRAM")]
        assert workers.call(1, "query", key=carrier, request_id="carrier-request") == [
            ("MISS", None)
        ]


def scenarios():
    """Explicit names keep failures and selective reruns easy to identify."""
    from functools import partial

    cases = {
        run.__name__: run
        for run in (
            unrelated_copy,
            unrelated_move,
            protected_move,
            group_collision,
            truncated_hash_collision,
            destination_collision,
            collision_recovers,
            mixed_action_envelope,
        )
    }
    for mode in ("copy", "move"):
        for run in (
            mixed_hash_batch,
            cached_destination,
        ):
            cases[f"{run.__name__}_{mode}"] = partial(run, mode=mode)
        cases[f"source_disappears_{mode}_framework_default"] = partial(
            source_disappears, mode=mode
        )
    return cases


def source_hashes():
    root = Path(kvcr_package.__file__).resolve().parent
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def main():
    cases = scenarios()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--case", action="append", choices=cases, help="default: all cases"
    )
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.block_size <= 0 or args.timeout <= 0 or args.repeat <= 0:
        parser.error("block-size, timeout, and repeat must be positive")
    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if output == checkout or checkout in output.parents or output.exists():
        parser.error("output-dir must be new and outside the checkout")
    output.mkdir(parents=True)
    report = {
        "source_hashes": source_hashes(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {
            "piece_bytes": args.block_size,
            "slots": 4,
            "timeout": args.timeout,
            "repeat": args.repeat,
        },
        "nixl_version": importlib.metadata.version("nixl"),
        "ucx_tls": os.environ.get("UCX_TLS"),
        "cases": [],
        "status": "FAIL",
    }
    try:
        for repeat in range(args.repeat):
            for name in args.case or cases:
                case = {"name": name, "repeat": repeat + 1, "status": "FAIL"}
                report["cases"].append(case)
                cases[name](args)
                case["status"] = "PASS"
                print(f"PASS {name} (run {repeat + 1})", flush=True)
        assert report["source_hashes"] == source_hashes(), "source changed during run"
        report["status"] = "PASS"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
