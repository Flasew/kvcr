# Hash-addressed cache operations

| Hint | Behavior |
| --- | --- |
| `kv.fetch` | Stores request-scoped source advice; framework `fetch()` / `deliver()` calls initiate ordinary copying. |
| `kv.migrate` | Stores request-scoped source advice; framework `fetch()` moves matching full keys, while `deliver()` still copies. No reverse hash lookup is needed. |
| `kv.prefetch` | Resolves advertised hashes and copies their full blocks into destination G2. |
| `kv.move` | Resolves advertised hashes and invokes the internal move mechanism. |

Independent actions work with or without a carrier request ID, can target unrelated blocks, and survive `discard_hint(carrier_id)`. A `kv.fetch` without a request ID does not become independent.

```json
{
  "protocol_version": "0.1",
  "message_id": "placement-batch",
  "actions": [{
    "action_id": "placement-42",
    "action_type": "kv.move",
    "action_version": "1.0",
    "payload": {
      "source_control_endpoint": "tcp://source-worker:5600",
      "block_hashes": [123, 456]
    }
  }]
}
```

Use `kv.prefetch` to preserve the source copy. One envelope can combine request advice and independent actions.

The source's `KeyAdapter.decode_from_router` maps each uint64 hash to a full `BlockKey`: `KeyError` reports `not_found`, `ValueError` reports `ambiguous_hash`, and any other exception fails only that hash with `resolution_failed`. The integration owns the reverse map and collision policy; KVCR adds no index. Lookups run on progress and must not block. Full keys remain the identities used for storage and transfers.

Resolution exchanges keys, ordered piece labels, and sizes in one metadata round trip without pinning memory. Self-addressed resolution fails with `self_source`.

The destination validates identity and layout before transfer. Independent transfers require complete managed G2 at the named source; missing/incomplete copies fail without acquiring or reusing framework pins. Framework-only discovery and remote G3 staging are unsupported. Ordinary request copying retains its framework fallback.

Keep polling both peers. A destination without managed G2 or a key adapter rejects an independent action at submission; otherwise it reports `ACCEPTED`, then `COMPLETED` or `FAILED`, with per-hash outcomes through the internal/debug result sink. Its completions are filtered from framework results and destination claims are released automatically. Closing abandons pending hint results. There is no public hint-result API.

Moves reuse existing transfers and source-owned G2 removal, honoring readers and `no_evict`. Completion reports destination readiness, not confirmation of source cleanup; late destination failure can follow successful source removal. See [move semantics](design_overview.md#move-and-source-invalidation).

An adapter may maintain reverse mappings from inventory events. The vLLM adapter requires KV events enabled; colliding full keys reject lookup until unique again, including hashes shared across cache groups. Guard recovery does not replay inventory: reverse mappings must be restored separately. Peers need compatible key conventions and layouts; KVCR does not transform them.
