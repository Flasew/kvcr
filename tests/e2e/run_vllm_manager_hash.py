# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Drive hash hints through two production vLLM KVCR tier managers.

Each worker process builds ``KVCRSecondaryTierManager`` through vLLM's tier
factory with a real KVCR, NIXL agent, and ZMQ control channel: no recording
KVCR and no test adapter. Inventory flows KVCR -> manager -> adapter map, hints
enter through ``on_new_request`` as the scheduler submits them, and later loads
use the ordinary ``lookup``/``submit_load`` path. No inference engine or router
runs. Requires vLLM with the KVCR tier and the native NIXL runtime; run with
this checkout's ``src`` on PYTHONPATH and an output directory outside it.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import socket
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kvcr.types import BlockKey


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def pattern(key, size):
    digest = hashlib.sha256(bytes(key)).digest()
    return (digest * (size // len(digest) + 1))[:size]


def worker(connection, block_bytes, chunks, slots):
    from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    from vllm.v1.kv_offload.tiering.kvcr import manager as kvcr_manager

    collision_warnings = []

    class Counter(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING and "collision" in record.getMessage():
                collision_warnings.append(record.getMessage())

    logging.getLogger(kvcr_manager.__name__).addHandler(Counter())

    tier = None
    try:
        primary = np.zeros((chunks, block_bytes), dtype=np.uint8)
        spec = SimpleNamespace(
            config=SimpleNamespace(
                parallel=SimpleNamespace(data_parallel_rank_local=None)
            ),
            kv_events_config=SimpleNamespace(
                enable_kv_cache_events=True, self_describing_kv_events=True
            ),
        )
        tier = SecondaryTierFactory.create_secondary_tier(
            tier_config={
                "type": "kvcr",
                "router_capabilities": ["router_hint"],
                "control_host": "127.0.0.1",
                "control_ports": [free_port()],
                "control_advertise_host": "127.0.0.1",
                "secondary_g2_slots": slots,
                "eager_ctrl_connect": False,
                "operation_timeout_ms": 15000,
                "abandon_timeout_ms": 30000,
            },
            primary_kv_view=memoryview(primary),
            offloading_spec=spec,
        )
        kvcr = tier._kvcr
        core = kvcr._core
        # The adapter KVCR resolves through is the one the manager feeds.
        assert core._remote_fw_dram._key_adapter is tier._key_adapter
        assert core._inventory_sink_callback == tier._record_inventory
        reports = {}
        original_report = core._hints_dispatcher.report

        def observe_report(result):
            # Hint results have an internal report sink, not a public API.
            reports.setdefault(result.action_id, []).append(
                {
                    "status": result.status.value,
                    "entries": {
                        str(h): {
                            "success": entry.success,
                            "keys": [bytes(k).hex() for k in entry.keys],
                            "reason": entry.reason,
                        }
                        for h, entry in (result.entries or {}).items()
                    },
                }
            )
            original_report(result)

        core._hints_dispatcher.report = observe_report
        leaked = []
        original_poll = kvcr.poll_completed

        def poll_completed():
            # Internal hint operations must never surface as framework results.
            results = list(original_poll())
            leaked.extend(h for h, _ in results if h not in tier._jobs_by_op)
            return results

        kvcr.poll_completed = poll_completed
        endpoint = core.framework_control.endpoint
        connection.send({"result": {"endpoint": endpoint, "pid": os.getpid()}})
        finished = {}
        events = []
        next_job = 1
        while True:
            for result in tier.get_finished_jobs():
                finished[result.job_id] = result
            events.extend(tier.take_events())
            if not connection.poll(0.001):
                continue
            command, args = connection.recv()
            key = args.get("key")
            if key is not None:
                key = BlockKey(bytes.fromhex(key))
            if command == "close":
                tier.shutdown()
                tier = None
                connection.send({"result": None})
                return
            if command in ("store", "load"):
                chunk = args["chunk"]
                if command == "store":
                    primary[chunk, :] = np.frombuffer(
                        pattern(key, block_bytes), dtype=np.uint8
                    )
                else:
                    primary[chunk, :] = 0
                job = TransferJob(
                    job_id=next_job,
                    keys=[key],
                    chunk_ids=np.array([chunk], dtype=np.int64),
                    is_promotion=command == "load",
                    req_context=ReqContext(req_id=args.get("req", command)),
                )
                next_job += 1
                if command == "store":
                    tier.submit_store(job)
                else:
                    tier.submit_load(job)
                result = job.job_id
            elif command == "job":
                done = finished.pop(args["job"], None)
                result = None if done is None else done.success
            elif command == "verify":
                result = primary[args["chunk"]].tobytes() == pattern(key, block_bytes)
            elif command == "lookup":
                result = tier.lookup(key, ReqContext(req_id=args["req"])).name
            elif command == "hint":
                action_id = args["action_id"]
                envelope = KvHintsEnvelope(
                    protocol_version="0.1",
                    message_id=action_id,
                    actions=[
                        KvHintAction(action_id, "kv.fetch", "1.0", args["payload"])
                    ],
                )
                context = ReqContext(req_id=args["req"], kv_hints=envelope)
                tier.on_new_request(context)
                if args.get("finish"):
                    tier.on_request_finished(context)
                result = None
            elif command == "finish_request":
                tier.on_request_finished(ReqContext(req_id=args["req"]))
                result = None
            elif command == "reports":
                result = reports.get(args["action_id"], [])
            elif command == "events":
                result = [
                    (
                        [bytes(k).hex() for k in event.keys],
                        event.medium.value,
                        event.removed,
                    )
                    for event in events
                ]
                events.clear()
            elif command == "adapter":
                try:
                    found = tier._key_adapter.decode_from_router(args["hash"])
                    result = {"key": bytes(found).hex()}
                except KeyError:
                    result = {"reason": "not_found"}
                except ValueError:
                    result = {"reason": "ambiguous_hash"}
            elif command == "state":
                handler = core._independent_hint_handler
                pools = core._local_dram._pools
                result = {
                    "leaked": list(leaked),
                    "resolving": len(handler._resolving),
                    "pending": len(handler._pending_operations),
                    "resolved": len(handler._resolved),
                    "request_hints": sorted(core._remote_fw_dram._request_hints),
                    "jobs": len(tier._jobs_by_op),
                    "source_idle": not (
                        core._remote_fw_dram._sources._source_pin_ops
                        or core._local_dram_sources_by_op
                        or pools.source_operations
                        or not pools._source_state.completions.empty()
                    ),
                    "invalidation_pending": core._invalidation_handler.has_pending(),
                    "collision_warnings": len(collision_warnings),
                    "claims": {
                        bytes(k).hex(): r.local_dram.claim_count
                        for k, r in core._block_record_map.items()
                        if r.local_dram is not None
                    },
                }
            else:
                raise ValueError(command)
            connection.send({"result": result})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        try:
            if tier is not None:
                tier.shutdown()
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
                    target=worker,
                    args=(child, args.block_size, args.chunks, args.slots),
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
        if args.get("key") is not None:
            args["key"] = bytes(args["key"]).hex()
        self.connections[rank].send((command, args))
        return self.receive(rank)

    def until(self, predicate, what):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.002)
        raise TimeoutError(f"{what} did not complete")

    def job(self, rank, command, key, chunk, **kwargs):
        job_id = self.call(rank, command, key=key, chunk=chunk, **kwargs)
        result = self.until(
            lambda: self.call(rank, "job", job=job_id), f"{command} job {job_id}"
        )
        assert result is True, (command, key.hex(), result)

    def hint(self, rank, mode, hashes, *, req, scope="independent", finish=True):
        self.sequence += 1
        action_id = f"action-{self.sequence}"
        payload = {
            "block_hashes": list(hashes),
            "mode": mode,
            "scope": scope,
            "source_control_endpoint": self.info[1 - rank]["endpoint"],
        }
        self.call(
            rank, "hint", action_id=action_id, payload=payload, req=req, finish=finish
        )
        return action_id

    def terminal(self, rank, action_id, status="COMPLETED"):
        def done():
            rows = self.call(rank, "reports", action_id=action_id)
            return rows if len(rows) == 2 else None

        rows = self.until(done, f"hint {action_id}")
        assert [row["status"] for row in rows] == ["ACCEPTED", status], rows
        return rows[-1]["entries"]

    def events(self, rank):
        return self.call(rank, "events")

    def clean(self, rank):
        def quiescent():
            state = self.call(rank, "state")
            idle = state["source_idle"] and not state["invalidation_pending"]
            return state if idle and state["pending"] == 0 else None

        state = self.until(quiescent, f"rank {rank} quiescence")
        assert state["leaked"] == [], state
        assert state["resolving"] == 0 and state["resolved"] == 0, state
        assert state["request_hints"] == [], state
        assert state["jobs"] == 0, state
        assert all(claims == 0 for claims in state["claims"].values()), state
        return state

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
                process.join(timeout=10)
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


def offload_key(tag, group=0):
    from vllm.v1.kv_offload.base import make_offload_key

    return BlockKey(make_offload_key(hashlib.sha256(tag.encode()).digest(), group))


def router_hash(key):
    from vllm.v1.kv_offload.tiering.kvcr.manager import _VllmKeyAdapter

    return _VllmKeyAdapter().encode_to_router(key)


def independent_transfer(args, mode):
    key = offload_key(mode)
    block_hash = router_hash(key)
    with Workers(args) as w:
        w.job(0, "store", key, 0)
        assert ([key.hex()], "CPU", False) in w.events(0), "source stored event"
        assert w.call(0, "adapter", hash=block_hash) == {"key": key.hex()}
        assert w.call(1, "adapter", hash=block_hash) == {"reason": "not_found"}
        # The carrier request finishes immediately; the operation outlives it.
        action = w.hint(1, mode, [block_hash], req="unrelated-carrier")
        entries = w.terminal(1, action)
        assert entries == {
            str(block_hash): {"success": True, "keys": [key.hex()], "reason": None}
        }, entries
        assert w.call(1, "lookup", key=key, req="later") == "HIT"
        assert w.call(1, "adapter", hash=block_hash) == {"key": key.hex()}
        assert ([key.hex()], "CPU", False) in w.events(1), "destination stored event"
        # A later ordinary load reads the independently fetched copy.
        w.job(1, "load", key, 1, req="later")
        assert w.call(1, "verify", key=key, chunk=1)
        w.call(1, "finish_request", req="later")
        if mode == "move":
            w.until(
                lambda: w.call(0, "lookup", key=key, req="probe") == "MISS",
                "source removal",
            )
            w.until(
                lambda: (
                    w.call(0, "adapter", hash=block_hash) == {"reason": "not_found"}
                ),
                "source map removal",
            )
            assert ([key.hex()], "CPU", True) in w.events(0), "source removal event"
        else:
            assert w.call(0, "lookup", key=key, req="probe") == "HIT"
            assert w.call(0, "adapter", hash=block_hash) == {"key": key.hex()}
        w.clean(0)
        w.clean(1)


def independent_copy(args):
    independent_transfer(args, "copy")


def independent_move(args):
    independent_transfer(args, "move")


def group_collision(args):
    """Two KV cache groups share one router hash: independent hints are rejected
    and warned about once; request-scoped advice with full keys still works."""
    pairs = [(offload_key(tag, 0), offload_key(tag, 1)) for tag in ("shared", "other")]
    block_hash = router_hash(pairs[0][0])
    assert router_hash(pairs[0][1]) == block_hash
    with Workers(args) as w:
        for chunk, key in enumerate(key for pair in pairs for key in pair):
            w.job(0, "store", key, chunk)
        assert w.call(0, "adapter", hash=block_hash) == {"reason": "ambiguous_hash"}
        for mode in ("copy", "move"):
            action = w.hint(1, mode, [block_hash], req="carrier")
            entries = w.terminal(1, action, "FAILED")
            assert entries[str(block_hash)]["reason"] == "ambiguous_hash", entries
        first, second = pairs[0]
        assert w.call(1, "lookup", key=first, req="r") == "MISS"
        w.hint(1, "copy", [block_hash], req="scoped", scope="request", finish=False)
        assert w.call(1, "lookup", key=first, req="scoped") == "HIT"
        assert w.call(1, "lookup", key=second, req="scoped") == "HIT"
        w.job(1, "load", first, 0, req="scoped")
        w.job(1, "load", second, 1, req="scoped")
        assert w.call(1, "verify", key=first, chunk=0)
        assert w.call(1, "verify", key=second, chunk=1)
        w.call(1, "finish_request", req="scoped")
        source = w.clean(0)
        w.clean(1)
        assert source["collision_warnings"] == 1, source["collision_warnings"]


def unknown_hash(args):
    with Workers(args) as w:
        action = w.hint(1, "copy", [12345], req="carrier")
        entries = w.terminal(1, action, "FAILED")
        assert entries["12345"]["reason"] == "not_found", entries
        w.clean(0)
        w.clean(1)


def provenance():
    import importlib.metadata

    from vllm.v1.kv_offload.tiering.kvcr import manager

    import kvcr as kvcr_package

    root = Path(kvcr_package.__file__).resolve().parent
    return {
        "kvcr_source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        },
        "vllm_manager_sha256": hashlib.sha256(
            Path(manager.__file__).read_bytes()
        ).hexdigest(),
        "vllm_version": importlib.metadata.version("vllm"),
        "nixl_version": importlib.metadata.version("nixl"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if min(args.block_size, args.chunks, args.slots, args.timeout) <= 0:
        parser.error("block-size, chunks, slots, and timeout must be positive")
    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if output == checkout or checkout in output.parents or output.exists():
        parser.error("output-dir must be new and outside the checkout")
    output.mkdir(parents=True)
    report = {
        **provenance(),
        "config": {k: v for k, v in vars(args).items() if k != "output_dir"},
        "ucx_tls": os.environ.get("UCX_TLS"),
        "cases": [],
        "status": "FAIL",
    }
    try:
        for run in (independent_copy, independent_move, group_collision, unknown_hash):
            run(args)
            report["cases"].append({"name": run.__name__, "status": "PASS"})
            print(f"PASS {run.__name__}", flush=True)
        report["status"] = "PASS"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
