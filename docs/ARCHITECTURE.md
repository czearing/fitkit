# Architecture

Six layers. Each answers one question, and each can be replaced without touching the others.

| Layer | Question | Type |
| --- | --- | --- |
| Evidence | what was measured, and how far is it trusted | `Evidence<E>`, `Confidence` |
| Provenance | where did the number come from, and what was never measured | `Record<T>`, `Citation` |
| Validity | does this question fall inside what was measured | `Law::admits`, `Refusal` |
| Model | what settings exist, and what does each one predict | `Model::candidates`, `Model::render` |
| Search | which setting best explains the evidence | `decode_path`, `optimise_subset` |
| Verification | how would we know if this were wrong | `fitkit-guards` |

## Why the layers are separate

**Evidence is separate from the model** so a solver can never read a reference directly. Every
fact enters through `Fit::evidence`, which is what makes "the answer is evidenced" checkable.

**Validity is separate from derivation** so the gate cannot be skipped. `Law::derive` is only
reachable through `ask`, which calls `admits` first.

**Search is separate from the domain.** `decode_path` and `optimise_subset` take closures and know
nothing about what they are optimising. There is one implementation of each, so a bug in the
search is a bug in one place.

**Recovery is a free function, not a trait method.** `recover` cannot be overridden, so no model
can quietly substitute its own pipeline.

## The recovery pipeline

```text
reference
   |
   |  Fit::evidence          split into spans, measure each, attach confidence
   v
Vec<Evidence<E>>
   |
   |  Model::candidates      the settings that may be chosen
   |  Fit::emission          cost of explaining one span with one candidate
   |  Fit::transition        cost of changing between neighbouring spans
   v
decode_path                  lowest cost path through the trellis
   |
   |  Fit::refine            replace a grid value with one measured directly
   |  Fit::settle            adjust against the whole render
   v
Plan<Params>
```

`refine` exists because the search can only return a candidate it was given. Anything continuous
that the evidence measures outright belongs there, so the result is the measured value rather than
the nearest grid point.

## Sequence problems and set problems

`decode_path` is for problems with an order: which setting held over each span. `optimise_subset`
is for problems without one: which members of a pool to include. Most engines need both.

Set problems carry a trap. If the objective grows quadratically in the member count and the
penalty also grows quadratically, there is no interior optimum and the search runs to one extreme
or the other. Check that your objective is sublinear in pair count before trusting a peak.

## Adding a domain

1. Define `Params`, the decision, and list every value in `candidates`.
2. Define `Evidence`, one measurement per span, with honest confidence.
3. Write `emission` as a cost, lower being a better explanation.
4. Write `transition` if the decision should resist change.
5. Move continuous quantities out of the grid and into `refine`.
6. Pin the guards from [GUARDS.md](GUARDS.md) as tests.
