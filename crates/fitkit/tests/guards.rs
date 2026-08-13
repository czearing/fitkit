//! The invariants every fitkit engine should pin. Copy this file into your own crate.

use fitkit::prelude::*;
use fitkit_guards::{
    assert_beam_matches_exact, assert_deterministic, assert_identity_plan_changes_nothing,
    assert_identity_without_evidence, assert_margin_holds, assert_untrusted_spans_stay_silent,
    forbid_symbols,
};

struct Setpoint {
    readings: Vec<Option<f64>>,
}

impl Model for Setpoint {
    type Signal = ();
    type Params = i64;

    fn name(&self) -> &'static str {
        "setpoint"
    }

    fn candidates(&self) -> Vec<i64> {
        (16..=26).collect()
    }

    fn render(&self, _input: &(), _params: &i64) {}
}

impl Fit for Setpoint {
    type Evidence = Option<f64>;

    fn evidence(&self, _reference: &()) -> Vec<Evidence<Option<f64>>> {
        self.readings
            .iter()
            .enumerate()
            .map(|(i, reading)| {
                let span = Span::new(i, i + 1);
                let confidence =
                    if reading.is_some() { Confidence::FULL } else { Confidence::ZERO };
                Evidence::new(span, confidence, *reading)
            })
            .collect()
    }

    fn emission(&self, reading: &Option<f64>, setpoint: &i64) -> f64 {
        reading.map_or(0.0, |value| (value - *setpoint as f64).abs())
    }

    fn transition(&self, from: &i64, to: &i64) -> f64 {
        f64::from(u8::from(from != to))
    }

    fn transition_weight(&self) -> f64 {
        4.0
    }
}

fn measured() -> Setpoint {
    Setpoint { readings: vec![Some(20.1), None, Some(19.9), Some(23.0), Some(23.1)] }
}

#[test]
fn a_model_with_no_evidence_recovers_the_identity_plan() {
    assert_identity_without_evidence(&Setpoint { readings: Vec::new() }, &());
}

#[test]
fn untrusted_spans_are_never_acted_on() {
    assert_untrusted_spans_stay_silent(&measured(), &());
}

#[test]
fn recovery_is_deterministic() {
    assert_deterministic(&measured(), &());
}

#[test]
fn a_wide_beam_finds_the_proven_optimum() {
    assert_beam_matches_exact(14, 512, |members| {
        let size = f64::from(members.count_ones());
        f64::from(members.count_ones() % 5) * 2.0 - 0.4 * size * size
    });
}

/// The rule that keeps an engine measuring: no derived quantity may descend from an identity.
///
/// Point this at the modules that derive quantities and ban every accessor keyed by a name.
#[test]
fn no_name_keyed_value_reaches_the_derivation_path() {
    forbid_symbols(
        &[("examples/thermostat.rs", include_str!("../examples/thermostat.rs"))],
        &["lookup_by_name", "table_for_name", "typical_value_for"],
    );
}

#[test]
fn a_refused_law_yields_no_number() {
    struct Density;
    impl Law for Density {
        type Input = f64;
        type Output = f64;

        fn citation(&self) -> Citation {
            Citation { key: "Kell1975", source: "Kell, J. Chem. Eng. Data 20, 97 (1975)" }
        }

        fn admits(&self, kelvin: &f64) -> Answer<()> {
            within(*kelvin, 273.15..=373.15, "temperature outside the measured range")
        }

        fn derive(&self, kelvin: &f64) -> Answer<f64> {
            Ok(1000.0 - 0.2 * (kelvin - 273.15))
        }
    }

    assert!(ask(&Density, &298.15).is_ok());
    assert!(ask(&Density, &450.0).is_err(), "extrapolation must refuse, not estimate");
}

/// A model whose signal can be cut and rejoined gets plan application for free.
struct Trim;

impl Model for Trim {
    type Signal = Vec<f64>;
    type Params = f64;

    fn name(&self) -> &'static str {
        "trim"
    }

    fn candidates(&self) -> Vec<f64> {
        vec![0.5, 1.0, 2.0]
    }

    fn render(&self, input: &Vec<f64>, params: &f64) -> Vec<f64> {
        input.iter().map(|value| value * params).collect()
    }
}

#[test]
fn applying_an_identity_plan_changes_nothing() {
    assert_identity_plan_changes_nothing(&Trim, &vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_reported_margin_is_the_error_the_answer_survives() {
    let mut problem = Problem::new(2);
    problem.bound(0, 0.0, 8.0);
    problem.bound(1, 0.0, 8.0);
    problem.row(Row::new(vec![1.0, 1.0], Sense::Le, 10.0, "total"));
    problem.row(Row::new(vec![1.0, 1.0], Sense::Ge, 4.0, "enough"));
    problem.row(Row::new(vec![1.0, -1.0], Sense::Le, 3.0, "balance"));

    assert_margin_holds(&problem);
}
