//! Shared vocabulary: confidence, margin, evidence, refusal, plan.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod confidence;
mod cost;
mod evidence;
mod margin;
mod plan;
mod refusal;
mod reported;

pub use confidence::Confidence;
pub use cost::{Cost, Scale, RULES};
pub use evidence::{Evidence, Span};
pub use margin::Margin;
pub use plan::{Control, Plan};
pub use refusal::{Answer, Refusal, RefusalKind};
pub use reported::Reported;
