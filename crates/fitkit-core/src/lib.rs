//! Shared vocabulary: confidence, evidence, refusal, plan.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod confidence;
mod evidence;
mod plan;
mod refusal;
mod reported;

pub use confidence::Confidence;
pub use evidence::{Evidence, Span};
pub use plan::{Control, Plan};
pub use refusal::{Answer, Refusal, RefusalKind};
pub use reported::Reported;
