//! Two dynamic programs, with no domain in them.
//!
//! [`decode_path`] solves sequence problems: pick one candidate per step, trading fit against the
//! cost of changing. [`optimise_subset`] solves set problems: pick the best subset of at most 64
//! candidates, exactly where that is affordable and by beam otherwise.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod path;
mod subset;

pub use path::{decode_margins, decode_path, decode_path_with_cost, Decoded};
pub use subset::{optimise_subset, Solver, SubsetResult, MAX_POOL};
