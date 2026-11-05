# Hash-addressed cache operations

The source's `KeyAdapter.decode_from_router` maps each uint64 hash to a full `BlockKey`: `KeyError` reports `not_found`, `ValueError` reports `ambiguous_hash`, and any other exception fails only that hash with `resolution_failed`. The integration owns the reverse map and collision policy; KVCR adds no index. Lookups run on progress and must not block. Full keys remain the identities used for storage and transfers.

Resolution exchanges keys, ordered piece labels, and sizes in one metadata round trip without pinning memory. Self-addressed resolution fails with `self_source`.

An adapter may maintain reverse mappings from inventory events. The vLLM adapter requires KV events enabled; colliding full keys reject lookup until unique again, including hashes shared across cache groups. Guard recovery does not replay inventory: reverse mappings must be restored separately. Peers need compatible key conventions and layouts; KVCR does not transform them.
