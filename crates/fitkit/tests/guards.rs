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
    let pool = 14;
    let mut terms = Terms::over(pool).expect("a pool to choose from");
    for item in 0..pool {
        terms = terms
            .worth(
                item,
                Evidence::certain(Span::new(item, item + 1), 1.0 - 0.15 * (item % 4) as f64),
            )
            .expect("a finite weight over a real span");
    }
    for a in 0..pool {
        for b in a + 1..pool {
            let value = ((a * 7 + b * 5) % 11) as f64 / 11.0 - 0.6;
            terms = terms
                .together(a, b, Evidence::certain(Span::new(a, b + 1), value))
                .expect("a finite weight over a real span");
        }
    }
    assert_beam_matches_exact(&terms, 512);
}

/// The objective cannot be handed the answer, because there is nowhere to write one down.
///
/// A subset search whose score took the candidate mask could be told what to return in one line,
/// and would return it through a real enumeration carrying an honest trace. [`Terms`] is stated
/// over the items instead, so the only way to make a subset win is to say what its members are
/// worth — a claim that has to cite a span, is discounted by the trust it is held with, and
/// therefore holds for every pool those items appear in rather than for one answer.
#[test]
fn what_makes_a_subset_win_is_a_cited_claim_about_its_items() {
    let span = Span::new(0, 4);
    let terms = Terms::over(4)
        .expect("a pool to choose from")
        .worth(0, Evidence::certain(span, 5.0))
        .expect("a cited weight")
        .worth(1, Evidence::certain(span, -5.0))
        .expect("a cited weight")
        .worth(2, Evidence::new(span, Confidence::new(0.5), 4.0))
        .expect("a cited weight")
        .worth(3, Evidence::certain(span, -1.0))
        .expect("a cited weight");

    let chosen = optimise_subset(&terms, 4, 1).expect("four items offer subsets");
    let members = chosen.get().members();
    assert_eq!(members, 0b0101, "the items the evidence argued for, and no others");
    assert!(!terms.support(members).is_empty(), "the answer names where it came from");
    assert!(chosen.trace().decided(), "every item was weighed both ways");

    // The same claims over a different pool give a different answer, which a lookup table keyed on
    // the mask could not do.
    let narrowed = terms.at_most(1).expect("a budget");
    let fewer = optimise_subset(&narrowed, 4, 1).expect("four items offer subsets");
    assert_eq!(fewer.get().members(), 0b0001, "the strongest claim survived the budget");
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

/// A witness over a caller's own type can only be got by asking the mechanism to build it.
///
/// This test lives outside `fitkit-dp` on purpose. Inside that crate `Chosen::map` is reachable,
/// so the property under test — that a caller cannot transform a witness it already holds into
/// one over any value it likes — is only observable from a crate that does not own the type.
/// Written as a use of the sanctioned route rather than as a check on the unsanctioned one: the
/// forgery is not expressible here, so there is nothing to assert about it.
#[test]
fn a_witness_over_a_foreign_type_is_built_by_the_search() {
    #[derive(Debug, PartialEq)]
    struct Pick(usize);

    let chosen = fitkit::dp::decode_path_as(
        4,
        3,
        1.0,
        |step, state| ((step + state) % 3) as f64,
        |from, to| (from as f64 - to as f64).abs(),
        |path| Pick(path.len()),
    )
    .expect("three candidates a step gave the search something to weigh");

    assert_eq!(chosen.get(), &Pick(4), "the built value came from the decoded path");
    assert!(chosen.trace().decided(), "the model offered rivals rather than a formality");
    assert_eq!(chosen.trace().steps(), 4);
}
