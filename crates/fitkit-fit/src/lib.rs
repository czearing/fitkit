//! Recovering parameters from evidence.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;

use fitkit_core::{Control, Evidence, Plan};
use fitkit_dp::decode_path;

/// A parameterised model of a domain.
pub trait Model {
    /// What the model reads and writes.
    type Signal;

    /// One setting of the model.
    type Params: Clone;

    /// Name, for telemetry.
    fn name(&self) -> &'static str;

    /// The settings the search may choose between.
    ///
    /// Listing them keeps the search exhaustive by construction rather than a hill climb.
    fn candidates(&self) -> Vec<Self::Params>;

    /// Apply one setting.
    fn render(&self, input: &Self::Signal, params: &Self::Params) -> Self::Signal;
}

/// A model whose parameters can be recovered from a reference.
pub trait Fit: Model {
    /// What one span of the reference says.
    type Evidence;

    /// Split the reference into spans and measure each.
    ///
    /// Returning nothing is the right answer for a reference that cannot support a recovery.
    fn evidence(&self, reference: &Self::Signal) -> Vec<Evidence<Self::Evidence>>;

    /// Cost of explaining `evidence` with `params`. Lower fits better.
    fn emission(&self, evidence: &Self::Evidence, params: &Self::Params) -> f64;

    /// Cost of changing between neighbouring spans.
    fn transition(&self, _from: &Self::Params, _to: &Self::Params) -> f64 {
        0.0
    }

    /// How hard to resist change.
    fn transition_weight(&self) -> f64 {
        1.0
    }

    /// Replace a grid value with one the evidence measured directly.
    ///
    /// The search can only return a candidate it was given. Anything continuous belongs here.
    fn refine(&self, params: Self::Params, _evidence: &Self::Evidence) -> Self::Params {
        params
    }

    /// Adjust a whole plan against the reference, for anything only the render can show.
    fn settle(&self, plan: Plan<Self::Params>, _reference: &Self::Signal) -> Plan<Self::Params> {
        plan
    }
}

/// Recover a plan from a reference.
///
/// Measure, decode the lowest-cost path through the candidates, refine each span with what was
/// measured, then settle. A free function, so no model can override the pipeline.
///
/// Returns [`Plan::identity`] when there is no evidence or no candidate.
pub fn recover<F: Fit>(model: &F, reference: &F::Signal) -> Plan<F::Params> {
    let evidence = model.evidence(reference);
    let candidates = model.candidates();
    if evidence.is_empty() || candidates.is_empty() {
        return Plan::identity();
    }

    let path = decode_path(
        evidence.len(),
        candidates.len(),
        model.transition_weight(),
        |step, state| model.emission(&evidence[step].value, &candidates[state]),
        |from, to| model.transition(&candidates[from], &candidates[to]),
    );

    let controls = evidence
        .iter()
        .zip(path)
        .map(|(span, state)| Control {
            span: span.span,
            params: model.refine(candidates[state].clone(), &span.value),
            confidence: span.confidence,
        })
        .collect();

    model.settle(Plan { controls }, reference)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use fitkit_core::{Confidence, Evidence, Span};

    use super::{recover, Fit, Model};

    /// A trivial model: the parameter is a gain and the evidence is the gain observed.
    ///
    /// It exists to prove the pipeline is domain free. If it ever needs anything signal shaped,
    /// the abstraction is wrong.
    struct Gain {
        observed: Vec<f64>,
    }

    impl Model for Gain {
        type Signal = Vec<f64>;
        type Params = i64;

        fn name(&self) -> &'static str {
            "gain"
        }

        fn candidates(&self) -> Vec<i64> {
            (0..4).collect()
        }

        fn render(&self, input: &Vec<f64>, params: &i64) -> Vec<f64> {
            input.iter().map(|v| v * *params as f64).collect()
        }
    }

    impl Fit for Gain {
        type Evidence = f64;

        fn evidence(&self, _reference: &Vec<f64>) -> Vec<Evidence<f64>> {
            self.observed
                .iter()
                .enumerate()
                .map(|(i, &v)| Evidence::certain(Span::new(i, i + 1), v))
                .collect()
        }

        fn emission(&self, evidence: &f64, params: &i64) -> f64 {
            (evidence - *params as f64).abs()
        }

        fn transition(&self, from: &i64, to: &i64) -> f64 {
            f64::from(u8::from(from != to))
        }

        fn transition_weight(&self) -> f64 {
            0.0
        }
    }

    #[test]
    fn recovery_follows_the_evidence_when_nothing_resists_change() {
        let model = Gain { observed: vec![2.0, 0.0, 3.0] };
        let plan = recover(&model, &vec![1.0]);
        let recovered: Vec<i64> = plan.controls.iter().map(|c| c.params).collect();
        assert_eq!(recovered, [2, 0, 3]);
    }

    #[test]
    fn no_evidence_recovers_the_identity_plan() {
        let model = Gain { observed: Vec::new() };
        assert!(recover(&model, &vec![1.0]).is_identity());
    }

    #[test]
    fn confidence_passes_through_to_the_plan() {
        struct Unsure;
        impl Model for Unsure {
            type Signal = ();
            type Params = u8;
            fn name(&self) -> &'static str {
                "unsure"
            }
            fn candidates(&self) -> Vec<u8> {
                vec![0, 1]
            }
            fn render(&self, _input: &(), _params: &u8) {}
        }
        impl Fit for Unsure {
            type Evidence = u8;
            fn evidence(&self, _reference: &()) -> Vec<Evidence<u8>> {
                vec![Evidence::new(Span::new(0, 1), Confidence::ZERO, 1)]
            }
            fn emission(&self, evidence: &u8, params: &u8) -> f64 {
                f64::from(u8::from(evidence != params))
            }
        }
        assert!(recover(&Unsure, &()).is_identity(), "zero confidence must stay silent");
    }
}
