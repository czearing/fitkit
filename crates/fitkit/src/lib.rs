//! Measure, refuse outside validity, optimise, verify.
//!
//! `fitkit` is the skeleton shared by engines that answer from measurement rather than from
//! plausibility. Four layers, each usable alone:
//!
//! | Layer | What it gives you |
//! | --- | --- |
//! | [`core`] | [`Confidence`], [`Margin`], [`Evidence`], [`Reported`], [`Refusal`], [`Plan`] |
//! | [`dp`] | [`decode_path`] for sequences, [`optimise_subset`] for sets |
//! | [`fit`] | [`Model`] and [`Fit`], recovered by [`recover`] and applied by [`Model::apply_plan`] |
//! | [`feasible`] | [`Problem`] and [`Requirement`], solved for a region and its [`Margin`] |
//! | [`ledger`] | [`Law`] and [`Record`], reached through [`ask`] |
//!
//! [`recover`] says what to do, [`margins`] says how much error that survives, and [`Problem`]
//! answers the same question where there is nothing to decode.
//!
//! ```
//! use fitkit::prelude::*;
//!
//! struct Thermostat(Vec<f64>);
//!
//! impl Model for Thermostat {
//!     type Signal = Vec<f64>;
//!     type Params = i64;
//!     fn name(&self) -> &'static str { "thermostat" }
//!     fn candidates(&self) -> Vec<i64> { (18..=24).collect() }
//!     fn render(&self, _input: &Vec<f64>, params: &i64) -> Vec<f64> { vec![*params as f64] }
//! }
//!
//! impl Fit for Thermostat {
//!     type Evidence = f64;
//!     fn evidence(&self, _reference: &Vec<f64>) -> Vec<Evidence<f64>> {
//!         self.0.iter().enumerate()
//!             .map(|(i, &v)| Evidence::certain(Span::new(i, i + 1), v))
//!             .collect()
//!     }
//!     fn emission(&self, reading: &f64, setpoint: &i64) -> f64 {
//!         (reading - *setpoint as f64).abs()
//!     }
//!     fn transition(&self, from: &i64, to: &i64) -> f64 { (from - to).abs() as f64 }
//!     fn transition_weight(&self) -> f64 { 4.0 }
//! }
//!
//! let model = Thermostat(vec![20.1, 19.8, 23.9, 20.2, 20.0]);
//! let plan = recover(&model, &Vec::new());
//! assert!(plan.controls.iter().all(|c| c.params == 20), "one noisy reading is not a new setpoint");
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub use fitkit_core as core;
pub use fitkit_dp as dp;
pub use fitkit_feasible as feasible;
pub use fitkit_fit as fit;
pub use fitkit_ledger as ledger;

#[doc(inline)]
pub use fitkit_core::{
    Answer, Confidence, Control, Evidence, Margin, Plan, Refusal, RefusalKind, Reported, Span,
};
#[doc(inline)]
pub use fitkit_dp::{
    decode_margins, decode_path, decode_path_with_cost, optimise_subset, Decoded, Solver,
    SubsetResult,
};
#[doc(inline)]
pub use fitkit_feasible::{Feasible, Problem, Requirement, Row, Sense};
#[doc(inline)]
pub use fitkit_fit::{margins, recover, Fit, Model, Segmented};
#[doc(inline)]
pub use fitkit_ledger::{ask, within, Citation, Law, Record};

/// Everything needed to write a model.
pub mod prelude {
    pub use crate::{
        ask, decode_path, margins, optimise_subset, recover, within, Answer, Citation, Confidence,
        Control, Evidence, Feasible, Fit, Law, Margin, Model, Plan, Problem, Record, Refusal,
        Reported, Requirement, Row, Segmented, Sense, Span,
    };
}

#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct ReadmeExamples;
