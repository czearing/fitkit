# Writing a law

A law turns a published measurement into a predicate. This file is the rule that keeps a knowledge
base from becoming a pile of constants.

## The rule

You may state **how the finished object must behave**. You may not state **any quantity the solver
exists to derive**.

| Admissible | Inadmissible |
| --- | --- |
| it must hold together | 31 percent hydration |
| it must stay rigid when cold | bake for 12 minutes |
| it must not separate | 3 parts to 1 |
| how many components it may have | it is pasta |

Naming the thing is inadmissible. Asking for "semolina, and how much water" hands the answer back
to the solver. A category word is worse: one word carries an entire implied formulation.

**Every law must name the published source that turns it into a predicate. A law that cannot cite
one is refused, not approximated.** Without that rule, this file becomes where hardcoded values
hide once they leave the source.

## What a law must supply

```rust
use fitkit::prelude::*;

struct WaterDensity;

impl Law for WaterDensity {
    type Input = f64;
    type Output = f64;

    fn citation(&self) -> Citation {
        Citation { key: "Kell1975", source: "Kell, J. Chem. Eng. Data 20, 97 (1975)" }
    }

    fn admits(&self, kelvin: &f64) -> Answer<()> {
        within(*kelvin, 273.15..=373.15, "temperature outside the measured range")
    }

    fn derive(&self, kelvin: &f64) -> Answer<f64> {
        Ok(999.83952 + 16.945176 * (kelvin - 273.15))
    }
}
```

Four things, all required:

1. **A citation.** A stable key and a full reference.
2. **A domain of validity.** Stated in `admits`, checked before anything is derived.
3. **A derivation.** Only reachable through `ask`, so the gate cannot be bypassed.
4. **The unknowns.** On a `Record`, the quantities the source never reported.

## Recording a measurement

```rust
use fitkit::{Citation, Record};

const KELL: Citation = Citation { key: "Kell1975", source: "Kell, J. Chem. Eng. Data 20, 97" };

const WATER_298: Record<f64> = Record {
    citation: KELL,
    conditions: &["298.15 K", "101.325 kPa", "air free"],
    unknowns: &["isotopic composition", "dissolved gas after handling"],
    uncertainty: 0.001,
    value: 997.047,
};
```

`unknowns` is the field that does the work. A source that did not state its pressure convention
has that written down, so a later reader cannot mistake silence for a value.

## Naming tests

One test file per published measurement, named for the datum:

```text
tests/xylitol_369p04k_adiabatic_enthalpy_fusion.rs
tests/water_ethanol_20vol_298p15k_excess_static_relative_permittivity.rs
```

Substance, composition, conditions, temperature with `p` for the decimal point, then the measured
quantity. The name alone says what would have to be wrong for the test to fail, and two files
cannot silently cover the same datum.

## Absence

Use `Reported` for anything a source may not have measured.

```rust
use fitkit::Reported;

let ph: Reported<f64> = Reported::Unreported;
assert!(ph.require("solvent pH").is_err());
```

There is no `unwrap_or`. Every route out of `Reported` either yields a measured value or refuses,
because a default here is indistinguishable from a measurement downstream.
