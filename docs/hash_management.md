# Hash-addressed cache operations

`kv.fetch` hints carry advertised uint64 block hashes and select `mode: "copy"`
or `mode: "move"`. KVCR resolves independent operations through the framework's
key adapter. Storage, ordinary lookups, and transfers continue to use
complete framework `BlockKey` values, preserving group, pool, and layout identity.

## Hint scopes

| Scope | With a request ID | Without a request ID |
| --- | --- | --- |
| Default or `"request"` | Source advice for later framework calls | Independent operation |
| `"independent"` | Independent operation | Independent operation |

Independent operations may target blocks unrelated to the inference request
carrying the hint. Discarding that request's advice does not cancel them.
Request-scoped advice relies on framework calls supplying complete keys;
`deliver` copies into framework buffers without executing move advice.

Submit a typed envelope through `submit_hint`. This protocol payload illustrates
an independent move to the receiving KVCR:

```json
{
  "protocol_version": "0.1",
  "actions": [{
    "action_id": "placement-42",
    "action_type": "kv.fetch",
    "action_version": "1.0",
    "payload": {
      "mode": "move",
      "scope": "independent",
      "source_control_endpoint": "tcp://source-worker:5600",
      "block_hashes": [123, 456]
    }
  }]
}
```

Use `mode: "copy"`, or omit `mode`, to preserve the source copy.

## Resolution and completion

The destination requests only the selected hash mappings from the named source.
The reply supplies complete keys, ordered piece labels, and sizes. The destination
validates them before invoking the existing full-key transfer path. This metadata
exchange adds a round trip before transfer. Resolution does not claim source
memory; the transfer revalidates residency if it changes in the meantime.

The destination stores records under the source's complete keys. A mapping alone
is not a readable copy; successful transfer makes data available for framework
reads using the same full key. Independent operations release their temporary
fetch claims automatically, so placed copies remain subject to normal eviction.
The destination publishes the placed key through its ordinary inventory event;
vLLM announces it with a placeholder `BlockStored` (hash and group only) because
no local request observed the block.

Move uses ordinary transfers carrying a move flag. The source admits each move
on its owner thread, closes the eligible G2 copies to new reads and deposits,
and drops them once its native transfer succeeds and existing readers finish; a
failed transfer reopens them. Keys whose source copy is incomplete or still
being deposited fail individually before withdrawal, so their deposits can
finish. Keys the destination already holds completely are named in one
`invalidate` message instead. `no_evict` protection takes precedence. Completion
reports destination readiness without waiting for source cleanup, and a
destination that fails after the source transfer succeeded reports
`operation_failed` for a key the source has already dropped, as with
request-scoped moves. Removal is best effort, has no separate result, and leaves
G3 and framework-owned copies unchanged. Invalidation is an internal mechanism.
See [move semantics](design_overview.md#move-and-source-invalidation).

Both owners must keep calling `poll_completed()`. Source metadata resolution and
native transfers run on progress; destination transfer setup, source move
admission, and source removal run on the application thread. Hint outcomes are
logged at DEBUG as `REJECTED`, or `ACCEPTED` followed by `COMPLETED` or
`FAILED`. Individual hashes can succeed or fail independently. Internal hint
completions do not appear among ordinary framework operation results; there is
no public hint-result polling API. Closing KVCR abandons pending hint results.

## Identity and supported sources

- The source calls `KeyAdapter.decode_from_router(hash)` on its progress thread.
  The adapter owns reverse lookup; KVCR has no hash index or record-lifetime hooks.
  Return a full key, raise `KeyError` for an unknown hash, or `ValueError` for an
  ambiguous hash. Lookups must not block the progress thread. The destination
  checks `encode_to_router(full_key)` against the requested hash before transfer.
- The vLLM manager updates the adapter's map from `inventory_sink` events.
  Enable KV cache events to register this callback and populate the map. A key
  stays mapped until its last managed tier is removed. Duplicate events are
  idempotent. Colliding full keys disable lookup for that hash until inventory
  makes it unique again.
- An unknown hash fails with `not_found`; multiple full keys for one source hash
  fail with `ambiguous_hash`. Distinct full keys can coexist even when their
  advertised hashes match. Transfers never substitute another key's data.
  vLLM models with several KV cache groups store one full key per group for the
  same block hash, so every advertised hash is ambiguous there: independent
  operations fail, while request-scoped advice still transfers every group
  because the framework supplies the full keys.
- Hint hashes are uint64 integers, matching vLLM's default
  `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1` projection. Byte hashes are rejected by
  the hint parser, so that setting must stay enabled for any `kv.fetch` hint.
- Hashes must identify stored transfer units. An inner framework-block hash may
  not name an independently stored offload chunk.
- Independent operations resolve advertised source managed DRAM records and
  their layouts. Framework-only enumeration and remote G3 staging are unsupported.
- Guard recovery does not replay inventory events. A fresh adapter cannot resolve
  recovered hashes until its inventory is restored; full-key access still works.
- Peers need compatible model, key conventions, and physical cache layouts.
  Labels and sizes are checked; KVCR does not transform model or TP layouts.
- A hint naming the destination's own control endpoint as its source fails
  with `self_source` during resolution. The destination must have managed G2
  capacity for the selected layouts.
