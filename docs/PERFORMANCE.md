# Performance

## Complexity

| Operation | Time | Memory |
| --- | --- | --- |
| `decode_path(steps, states)` | `O(steps * states^2)` | `O(steps * states)` backpointers |
| `optimise_subset` exact | `O(2^pool)` | `O(1)` |
| `optimise_subset` beam | `O(pool^2 * width * log width)` | `O(pool * width)` |

Cost closures are evaluated once per distinct argument. Emissions are called `steps * states`
times, transitions `states^2` times and cached before the trellis runs, so an expensive cost
function is affordable.

## Measured

Apple M1 Pro, `--release` with fat LTO and one codegen unit. Reproduce with
`cargo bench -p fitkit-dp`.

| Benchmark | Time | Throughput |
| --- | --- | --- |
| `decode_path` 4096 steps, 16 states | 391 us | 2.7 transitions/ns |
| `decode_path` 512 steps, 64 states | 1.86 ms | 1.1 transitions/ns |
| `optimise_subset` exact, 20 items | 1.57 ms | 1.05 M subsets in 1.5 ms |
| `optimise_subset` beam, 64 items, width 128 | 243 us | 4096 states expanded |

The 64 state case is slower per transition because the transition table reaches 32 KB and stops
fitting comfortably in L1.

## Design choices that keep it fast

**Backpointers are `u32`, not `usize`.** Halves the table, which is the largest allocation and the
one that is streamed. The state count is asserted to fit.

**Two rolling cost rows.** Only the previous row is needed, so the trellis is never materialised.

**The transition table is precomputed.** Transitions do not depend on the step, so they are paid
for once rather than `steps` times.

**Subsets are `u64` bitmasks.** Membership, union, and cardinality are single instructions.
`count_ones` is one `popcnt`.

**No allocation in any inner loop.** Every buffer is sized once up front.

**Nothing is generic over a float type.** One monomorphisation, so the inner loop is inspectable
in the disassembly and stays that way.

## Staying fast

Keep `candidates` small. The trellis is quadratic in it, and a large candidate set is usually a
sign that a continuous quantity should move into `refine` instead.

Keep `emission` cheap, or precompute inside `evidence`. It is the hottest closure.

Prefer `optimise_subset` exact where the pool allows. Twenty items is 1.5 ms and gives a proven
optimum; a beam gives neither the proof nor much of the time back at that size.

Verify optimisations produce identical output, not merely similar output. Speed that costs
accuracy is not a speedup.
