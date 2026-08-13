//! Invariant checks to call from your own tests.
//!
//! These are the rules that decide whether an engine is measuring or guessing. Each one panics
//! with the violation named.

use std::fmt::Debug;

use fitkit_core::Plan;
use fitkit_dp::optimise_subset;
use fitkit_fit::{recover, Fit};

/// Panic if any banned symbol appears in any source text.
///
/// Pass files with [`include_str!`]. Use it to keep identity keyed lookups, such as a table from a
/// name to a number, out of the path that derives quantities. Reading the call graph once proves
/// nothing about tomorrow; this does.
///
/// # Panics
///
/// If a banned symbol is found.
pub fn forbid_symbols(sources: &[(&str, &str)], banned: &[&str]) {
    for (path, text) in sources {
        for symbol in banned {
            assert!(
                !text.contains(symbol),
                "{path} reaches {symbol}, so an answer can descend from it"
            );
        }
    }
}

/// Panic unless a model with no evidence recovers the identity plan.
///
/// # Panics
///
/// If the model produces controls it cannot justify.
pub fn assert_identity_without_evidence<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug,
{
    let plan = recover(model, reference);
    assert!(plan.is_identity(), "{} invented {:?} from no evidence", model.name(), plan.controls);
}

/// Panic unless every zero confidence span is silent in the recovered plan.
///
/// # Panics
///
/// If an untrusted span would be acted on.
pub fn assert_untrusted_spans_stay_silent<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug,
{
    let evidence = model.evidence(reference);
    let plan = recover(model, reference);
    for (measured, control) in evidence.iter().zip(&plan.controls) {
        assert!(
            measured.is_informative() || control.is_silent(),
            "{} acts on {:?} where the evidence carried nothing",
            model.name(),
            control.params
        );
    }
}

/// Panic unless two recoveries from the same reference agree.
///
/// # Panics
///
/// If recovery is not deterministic.
pub fn assert_deterministic<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug + PartialEq,
{
    let first: Plan<F::Params> = recover(model, reference);
    let second = recover(model, reference);
    assert!(first == second, "{} recovered two different plans from one reference", model.name());
}

/// Panic unless the beam finds the proven optimum on a pool small enough to enumerate.
///
/// # Panics
///
/// If the beam is too narrow to be trusted at this pool size.
pub fn assert_beam_matches_exact<S>(pool: usize, beam_width: usize, score: S)
where
    S: Fn(u64) -> f64 + Copy,
{
    let exact = optimise_subset(pool, pool, 1, score);
    let beam = optimise_subset(pool, 0, beam_width, score);
    assert!(
        (exact.score - beam.score).abs() < 1e-9,
        "beam scored {} against a proven optimum of {}",
        beam.score,
        exact.score
    );
}

#[cfg(test)]
mod tests {
    use fitkit_core::{Evidence, Span};
    use fitkit_fit::{Fit, Model};

    use super::{
        assert_beam_matches_exact, assert_deterministic, assert_identity_without_evidence,
        assert_untrusted_spans_stay_silent, forbid_symbols,
    };

    struct Silent;

    impl Model for Silent {
        type Signal = ();
        type Params = u8;
        fn name(&self) -> &'static str {
            "silent"
        }
        fn candidates(&self) -> Vec<u8> {
            vec![0, 1]
        }
        fn render(&self, _input: &(), _params: &u8) {}
    }

    impl Fit for Silent {
        type Evidence = u8;
        fn evidence(&self, _reference: &()) -> Vec<Evidence<u8>> {
            Vec::new()
        }
        fn emission(&self, evidence: &u8, params: &u8) -> f64 {
            f64::from(u8::from(evidence != params))
        }
    }

    fn score(members: u64) -> f64 {
        let count = f64::from(members.count_ones());
        f64::from(members.count_ones() % 7) - 0.3 * count
    }

    #[test]
    fn the_guards_pass_a_model_that_abstains() {
        assert_identity_without_evidence(&Silent, &());
        assert_untrusted_spans_stay_silent(&Silent, &());
        assert_deterministic(&Silent, &());
    }

    #[test]
    fn a_wide_beam_matches_exact() {
        assert_beam_matches_exact(12, 256, score);
    }

    #[test]
    #[should_panic(expected = "reaches lookup_by_name")]
    fn a_banned_symbol_is_caught() {
        forbid_symbols(&[("solver.rs", "let x = lookup_by_name(food);")], &["lookup_by_name"]);
    }

    #[test]
    #[should_panic(expected = "invented")]
    fn a_model_that_invents_controls_is_caught() {
        struct Inventor;
        impl Model for Inventor {
            type Signal = ();
            type Params = u8;
            fn name(&self) -> &'static str {
                "inventor"
            }
            fn candidates(&self) -> Vec<u8> {
                vec![9]
            }
            fn render(&self, _input: &(), _params: &u8) {}
        }
        impl Fit for Inventor {
            type Evidence = u8;
            fn evidence(&self, _reference: &()) -> Vec<Evidence<u8>> {
                vec![Evidence::certain(Span::new(0, 1), 0)]
            }
            fn emission(&self, _evidence: &u8, _params: &u8) -> f64 {
                0.0
            }
        }
        assert_identity_without_evidence(&Inventor, &());
    }
}
