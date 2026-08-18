//! Two dynamic programs, with no domain in them.
//!
//! [`decode_path`] solves sequence problems: pick one candidate per step, trading fit against the
//! cost of changing. [`optimise_subset`] solves set problems: pick the best subset of at most 64
//! candidates, exactly where that is affordable and by beam otherwise.
//!
//! Neither takes an objective that can see what it is scoring. `decode_path` asks about one step
//! and one pair of states at a time; `optimise_subset` takes [`Terms`], which are stated over the
//! items. An objective that has already decided its answer cannot be written down, because there
//! is no argument shaped to hold one. Every weight arrives as evidence — the span it speaks for
//! and the confidence it is held with — so a tuned constant with nothing behind it is refused.
//!
//! Both return their result inside [`Chosen`], which has no public constructor and no public map,
//! so a value that reaches a consumer is one a search produced.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod chosen;
mod path;
mod subset;

pub use chosen::{Chosen, Trace};
pub use path::{
    decode_margins, decode_margins_onward, decode_path, decode_path_as, decode_path_into,
    decode_path_into_as, decode_path_onward, decode_path_onward_as, decode_path_parts,
    decode_path_with_cost, Decoded,
};
pub use subset::{
    optimise_subset, optimise_subset_as, optimise_subset_parts, Solver, SubsetResult, Terms,
    MAX_POOL,
};
