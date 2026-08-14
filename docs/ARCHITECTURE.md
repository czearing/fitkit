# Architecture

Seven layers. Each answers one question, and each can be replaced without touching the others.

| Layer | Question | Type |
| --- | --- | --- |
| Evidence | what was measured, and how far is it trusted | `Evidence<E>`, `Confidence` |
| Provenance | where did the number come from, and what was never measured | `Record<T>`, `Citation` |
| Validity | does this question fall inside what was measured | `Law::admits`, `Refusal` |
| Model | what settings exist, and what does each one predict | `Model::candidates`, `Model::render` |
| Search | which setting best explains the evidence | `decode_path`, `optimise_subset` |
| Feasibility | what satisfies every requirement, and by how much | `Problem`, `Margin` |
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

**Applying a plan is a trait method, and can be.** `Model::apply_plan` renders each span from the
input rather than from the running result, so settings never compound. A setting that outlives its
span, such as a tail that reaches the next one, overrides it. That is the one place a domain
legitimately needs to change how the result is assembled, and it cannot reach the search. A render
that returns a different length than it was given panics, since it would silently shift every span
after it.

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

## Impossible is not expensive

A cost of infinity means a candidate cannot explain a step, which is different from explaining it
badly. Summing the two together lets one unexplainable step erase every decision after it, since a
large enough running total stops distinguishing anything added to it. Impossible steps are counted
and compared before cost, so a passage nothing explains leaves the rest of the path intact and the
total is reported as infinite. The count costs nothing in the ordinary case: it is only reached
when no path at all is payable.

## Three shapes of problem

`decode_path` is for problems with an order: which setting held over each span. `optimise_subset`
is for problems without one: which members of a pool to include. `Problem::solve` is for problems
with nothing to choose at all, only requirements that must hold together. Most engines need more
than one.

Set problems carry a trap. If the objective grows quadratically in the member count and the
penalty also grows quadratically, there is no interior optimum and the search runs to one extreme
or the other. Check that your objective is sublinear in pair count before trusting a peak.

## Adding a domain

1. Define `Params`, the decision, and list every value in `candidates`.
2. Define `Evidence`, one measurement per span, with honest confidence.
3. Write `emission` as a cost, lower being a better explanation.
4. Write `transition` if the decision should resist change.
5. Move continuous quantities out of the grid and into `refine`.
6. Implement `Segmented` if the answer must be written back into the signal.
7. State anything that is a constraint rather than a choice as a `Requirement`. See
   [FEASIBILITY.md](FEASIBILITY.md).
8. Pin the guards from [GUARDS.md](GUARDS.md) as tests.
