# Guards

Invariant checks from `fitkit-guards`. Each one catches a specific way an engine stops measuring
and starts guessing. Call them from your own test suite.

## `forbid_symbols`

Fails if a banned symbol appears in a source file. Point it at the modules that derive quantities,
and ban every accessor keyed by an identity.

```rust
forbid_symbols(
    &[("src/formulate.rs", include_str!("../src/formulate.rs"))],
    &["lookup_by_name", "table_for_category"],
);
```

**Catches:** an answer that descends from the name of the thing being asked about rather than from
its measured composition. Containment established by reading the call graph once is worth nothing
the day someone adds a call. This is worth something every day.

## `forbid_derivations_from`

Fails if any function signature derives one kind of output from a kind of input. Where
`forbid_symbols` bans a spelling, this bans a shape.

```rust
forbid_derivations_from(
    &sources,                          // (path, text) pairs
    "Requirement",                     // what must not be derived
    &["&str", "String", "name:"],      // from what it must not be derived
    &["(String, Measured)"],           // parameters that are passengers
);
```

**Catches:** the back door into an engine that already refuses names elsewhere. Intent stated by
the caller is legitimate; intent inferred from a label is not, and the two are indistinguishable
behaviourally because the laws themselves are unchanged. Only the signature betrays it, so only a
signature check finds it.

A substring ban cannot express this. Banning `&str` outright fires on every incidental parameter in
the repository, and banning nothing misses the pattern entirely. The parser strips line comments so
a file that discusses the banned shape is not condemned by its own prose, and flattens signatures
so a formatter cannot hide one across lines. `exempt` covers the case where a banned type is
present but demonstrably a passenger, such as a label carried alongside the measurement that
actually decides.

## `assert_identity_without_evidence`

Fails if a model produces controls from an empty reference.

**Catches:** a pipeline that returns neutral parameters when it should return nothing. Neutral
parameters are a claim; an identity plan is not.

## `assert_untrusted_spans_stay_silent`

Fails if a zero confidence span is acted on.

**Catches:** confidence that is computed and then ignored. A dropout, an unsupported region, or a
measurement that failed must survive the whole pipeline as "leave this alone".

## `assert_deterministic`

Fails if two recoveries from one reference disagree.

**Catches:** hash iteration order, uninitialised state, and NaN driven comparisons. Any of these
make a regression suite meaningless.

## `assert_beam_matches_exact`

Fails if a beam misses the proven optimum on a pool small enough to enumerate.

**Catches:** a beam too narrow for the problem. It cannot prove the beam is right at full scale,
but it converts an unbounded worry into a measured one, and it fails loudly when the objective
changes shape.

## `assert_identity_plan_changes_nothing`

Fails if applying an identity plan alters the signal.

**Catches:** a stage that always renders and then trims back toward the input. Such a stage cannot
leave a passage it has no evidence about alone, however small the change looks. Only reachable for
a `Model` whose `Signal` implements `Segmented`, and it is the override of `apply_plan` that this
usually catches.

## `assert_margin_holds`

Fails if a reported margin is not the error the answer survives. Walks every corner of the
box the margin describes, requires all of them to satisfy every row, then requires a point just
outside to fail.

**Catches:** a margin that is decoration. Also fails on an unbounded margin, which means a missing
constraint rather than a safe answer. Enumerates corners, so keep it under twenty variables.

## Invariants worth adding per engine

These cannot be written generically, but every engine should have them.

**Identity safety.** Fitting a reference toward itself returns it unchanged. If any transform
fires when input already matches the target, the transform is not measuring a difference.

**Byte identical optimisation.** Performance work produces bit for bit identical output. Without
this, speed silently costs accuracy.

**No instance specific code.** No branch keyed to one input. Behaviour is driven by measurement.

**Falsified specifications stay falsified.** When a specified formula turns out to be wrong, keep
the test that proves it wrong. A structural error that was found once will otherwise be
reintroduced by the next person who reads the specification.
