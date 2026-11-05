# Hash metadata resolution

KVCR peers can resolve advertised uint64 hashes to complete framework `BlockKey` values through the source's `KeyAdapter.decode_from_router`. The adapter owns the reverse map and collision detection; KVCR does not add another index or change full-key storage and transfers.

The internal `resolve_hashes` request carries hashes and a deadline. Its `hashes_resolved` reply contains each full key, ordered piece labels, and sizes, or a per-hash failure. `KeyError` becomes `not_found`, `ValueError` becomes `ambiguous_hash`, and a key without eligible managed G2 residency becomes `source_not_transferable`. A request from this process itself fails with `self_source`; a destination cannot resolve hashes against its own cache. Unexpected adapter errors fail the resolution request.

Lookup runs on the source progress thread without waiting for the application thread or acquiring its state lock. Adapter lookups must not block. Resolution exchanges only control metadata: it transfers no block bytes and acquires no source-memory claim. A later full-key transfer must revalidate the source residency.

The source validates deadlines and hash sets before looking up keys. Full-key APIs and existing request-scoped hints remain available without reverse lookup. Framework-only enumeration and remote G3 staging are not supported by this service.

Run `uv run pytest tests/unit/test_hash_resolution.py -q` for source lookup, protocol errors, and progress-thread behavior.
