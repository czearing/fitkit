# fitkit

Measure, refuse outside validity, optimise, verify.

`fitkit` is the skeleton shared by engines that answer from measurement rather than from
plausibility. It was extracted from two unrelated production engines, a reference mastering engine
and a food chemistry solver, which had independently converged on the same six layers.

The pattern:

1. Measure reality into evidence that carries its own confidence.
2. Refuse when the question falls outside what was measured.
3. Choose parameters by dynamic programming over an enumerated candidate set.
4. Verify against ground truth, and pin the anti-cheating rules as tests.

## Install

```toml
[dependencies]
fitkit = "0.1"
```

No runtime dependencies. `no_std` with `default-features = false`.

## Example

Recover a thermostat setpoint from readings that include a sensor spike and a dropout.

```rust
use fitkit::prelude::*;

struct Thermostat;

impl Model for Thermostat {
    type Signal = Vec<Option<f64>>;
    type Params = i64;
    fn name(&self) -> &'static str { "thermostat" }
    fn candidates(&self) -> Vec<i64> { (16..=26).collect() }
    fn render(&self, input: &Self::Signal, p: &i64) -> Self::Signal {
        input.iter().map(|_| Some(*p as f64)).collect()
    }
}

impl Fit for Thermostat {
    type Evidence = Option<f64>;

    fn evidence(&self, reference: &Self::Signal) -> Vec<Evidence<Option<f64>>> {
        reference.iter().enumerate().map(|(i, reading)| {
            // A dropout is not a reading of zero. It carries no information.
            let trust = if reading.is_some() { Confidence::FULL } else { Confidence::ZERO };
            Evidence::new(Span::new(i, i + 1), trust, *reading)
        }).collect()
    }

    fn emission(&self, reading: &Option<f64>, setpoint: &i64) -> f64 {
        reading.map_or(0.0, |v| (v - *setpoint as f64).abs())
    }

    fn transition(&self, from: &i64, to: &i64) -> f64 { f64::from(u8::from(from != to)) }
    fn transition_weight(&self) -> f64 { 4.0 }
}

let log = vec![
    Some(20.1), Some(19.8), Some(30.0), Some(20.2), Some(19.9), Some(20.1),
    None, None,
    Some(23.0), Some(23.1), Some(22.9),
];
let plan = recover(&Thermostat, &log);

assert_eq!(plan.at(2).unwrap().params, 20);   // the spike is absorbed
assert!(plan.at(6).unwrap().is_silent());     // the dropout is left alone
assert_eq!(plan.at(9).unwrap().params, 23);   // the sustained step is kept
```

Run the full version with `cargo run -p fitkit --example thermostat`.

## The crates

| Crate | What it gives you |
| --- | --- |
| `fitkit-core` | `Confidence`, `Evidence`, `Reported`, `Refusal`, `Plan` |
| `fitkit-dp` | `decode_path` for sequences, `optimise_subset` for sets |
| `fitkit-fit` | `Model` and `Fit`, recovered by `recover` |
| `fitkit-ledger` | `Law` and `Record`, reached through `ask` |
| `fitkit-guards` | The invariant checks to call from your tests |
| `fitkit` | Facade over all of the above |

Each is usable alone. `fitkit-dp` in particular is a standalone Viterbi and subset optimiser with
no other dependency.

## The invariants

These are load bearing. Changes that violate them are regressions.

- **Absence is a value.** `Reported::Unreported` is not zero, not a default, and not a typical
  figure. It has no accessor that can substitute one.
- **Refusal is a correct answer.** A `Law` states its domain of validity in `admits`, and `ask`
  gates on it. Outside that domain there is no number.
- **Zero confidence means untouched.** A span the evidence cannot speak for stays silent, whatever
  the search decoded for it.
- **The search is exhaustive by construction.** `candidates` lists what may be chosen, so a result
  is a minimum over a stated set rather than wherever a hill climb stopped.
- **A beam result is never called optimal.** `SubsetResult::solver` reports which solver ran.
- **No answer descends from a name.** `forbid_symbols` pins that as a test, because reading the
  call graph once proves nothing about tomorrow.

## Documentation

- [Architecture](docs/ARCHITECTURE.md), the six layers and why they are separate
- [Writing a law](docs/LAWS.md), what a new measurement must supply
- [Guards](docs/GUARDS.md), the invariant checks and what each one catches
- [Performance](docs/PERFORMANCE.md), complexity, benchmarks, and how to stay fast

## Development

```sh
cargo test --workspace          # 46 tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo bench -p fitkit-dp
```

## License

MIT. See [LICENSE](LICENSE).
