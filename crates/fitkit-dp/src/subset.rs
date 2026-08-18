use alloc::vec;
use alloc::vec::Vec;

use fitkit_core::{Answer, Refusal};

use crate::chosen::{Chosen, Tally, Trace};

/// Largest pool a `u64` mask can hold.
pub const MAX_POOL: usize = 64;

/// Which solver produced a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    /// Every subset was enumerated. The result is a proven optimum.
    Exact,
    /// A width-limited beam. The result is the best found, not a proven optimum.
    Beam {
        /// States kept per size.
        width: usize,
    },
}

/// The best subset found, and how it was found.
///
/// The fields are private and only this crate can fill them, so a `SubsetResult` in hand is one
/// [`optimise_subset`] returned. A caller that wants a particular membership has to make the score
/// prefer it, which is the work; it cannot write the mask down and pass it off as a search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubsetResult {
    members: u64,
    score: f64,
    solver: Solver,
}

impl SubsetResult {
    pub(crate) const fn new(members: u64, score: f64, solver: Solver) -> Self {
        Self { members, score, solver }
    }

    /// Bit `i` set means item `i` is a member.
    pub const fn members(&self) -> u64 {
        self.members
    }

    /// Score of the members.
    pub const fn score(&self) -> f64 {
        self.score
    }

    /// Whether every subset was enumerated or a beam was run.
    pub const fn solver(&self) -> Solver {
        self.solver
    }

    /// Whether the score is a proven global optimum.
    pub fn is_proven(&self) -> bool {
        self.solver == Solver::Exact
    }

    /// Member indices.
    pub fn indices(&self) -> impl Iterator<Item = usize> {
        let members = self.members;
        (0..MAX_POOL).filter(move |i| members & (1 << i) != 0)
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.members.count_ones() as usize
    }

    /// Whether the subset is empty.
    pub fn is_empty(&self) -> bool {
        self.members == 0
    }
}

/// Maximise `score` over subsets of a pool of `pool` items.
///
/// Enumerates every subset when `pool <= exact_limit`, otherwise runs a beam keeping `beam_width`
/// states per size. [`SubsetResult::solver`] reports which ran, so a beam answer is never mistaken
/// for a proven optimum.
///
/// The result comes back inside the witness the search produced, on the same terms as
/// [`decode_path_with_cost`](crate::decode_path_with_cost): a caller can read it and carry it
/// onward but cannot build one, so a subset that reaches a consumer is a subset something searched
/// for. The witness carries the score as its cost and an account of what was weighed.
///
/// # Errors
///
/// Refuses an empty pool. There is no subset to choose between when there is nothing to choose
/// from, and returning the empty set as an optimum would dress that in the clothes of a result.
///
/// # Panics
///
/// If `pool` exceeds [`MAX_POOL`], or if exact enumeration is requested for `pool >= 63`.
pub fn optimise_subset<S>(
    pool: usize,
    exact_limit: usize,
    beam_width: usize,
    score: S,
) -> Answer<Chosen<SubsetResult>>
where
    S: Fn(u64) -> f64,
{
    assert!(pool <= MAX_POOL, "pool of {pool} exceeds the {MAX_POOL} bit mask");
    if pool == 0 {
        return Err(Refusal::unreported("an empty pool offers no subset to choose"));
    }
    let mut tally = Tally::new();
    let best = if pool <= exact_limit {
        assert!(pool < 63, "exact enumeration of {pool} items is not affordable");
        exact(pool, &score, &mut tally)
    } else {
        beam(pool, beam_width.max(1), &score, &mut tally)
    };
    Ok(Chosen::new(best, best.score, Trace::new(pool, pool, tally)))
}

fn exact<S: Fn(u64) -> f64>(pool: usize, score: &S, tally: &mut Tally) -> SubsetResult {
    let mut best = SubsetResult::new(0, f64::NEG_INFINITY, Solver::Exact);
    let mut runner_up = f64::NEG_INFINITY;
    let subsets = 1_u64 << pool;
    for members in 0..subsets {
        let value = score(members);
        if value > best.score {
            runner_up = best.score;
            best.members = members;
            best.score = value;
        } else if value > runner_up {
            runner_up = value;
        }
    }
    // Every item was taken or left in every combination, so each one was a decision with both
    // branches on the table.
    for _ in 0..pool {
        tally.decision(2, 2);
    }
    tally.ended(best.score, runner_up);
    best
}

fn beam<S: Fn(u64) -> f64>(
    pool: usize,
    width: usize,
    score: &S,
    tally: &mut Tally,
) -> SubsetResult {
    let empty = score(0);
    let mut frontier = vec![(0_u64, empty)];
    let mut best = (0_u64, empty);
    let mut grown: Vec<(u64, f64)> = Vec::with_capacity(width * pool);

    for _ in 0..pool {
        grown.clear();
        for &(members, _) in &frontier {
            let first = MAX_POOL - members.leading_zeros() as usize;
            for item in first..pool {
                let candidate = members | (1 << item);
                grown.push((candidate, score(candidate)));
            }
        }
        if grown.is_empty() {
            break;
        }
        grown.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let offered = grown.len();
        grown.truncate(width);
        tally.decision(offered as u64, grown.len() as u64);
        if grown[0].1 > best.1 {
            best = grown[0];
        }
        frontier.clear();
        frontier.extend_from_slice(&grown);
    }

    tally.ended(best.1, grown.get(1).map_or(f64::NEG_INFINITY, |&(_, value)| value));
    SubsetResult::new(best.0, best.1, Solver::Beam { width })
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use fitkit_core::RefusalKind;

    use super::{optimise_subset, Solver, MAX_POOL};

    /// Pairwise score with a size penalty, so the optimum is an interior subset.
    fn score(members: u64) -> f64 {
        let items: Vec<usize> = (0..12).filter(|i| members & (1 << i) != 0).collect();
        let mut total = 0.0;
        for (a, first) in items.iter().enumerate() {
            for second in &items[a + 1..] {
                total += ((first * 31 + second * 17) % 13) as f64 / 13.0;
            }
        }
        total - 0.35 * (items.len() * items.len()) as f64
    }

    #[test]
    fn the_beam_matches_exact_enumeration_where_both_can_run() {
        let exact = optimise_subset(12, 12, 1, score).expect("twelve items offer subsets");
        let beam = optimise_subset(12, 0, 64, score).expect("twelve items offer subsets");
        let (exact, beam) = (exact.get(), beam.get());
        assert!(exact.is_proven());
        assert_eq!(beam.solver(), Solver::Beam { width: 64 });
        assert!((exact.score() - beam.score()).abs() < 1e-12, "beam {beam:?} exact {exact:?}");
    }

    #[test]
    fn a_beam_result_is_never_claimed_as_proven() {
        let beamed = optimise_subset(30, 20, 8, score).expect("thirty items offer subsets");
        assert!(!beamed.get().is_proven());
    }

    #[test]
    fn members_and_indices_agree() {
        let chosen = optimise_subset(12, 12, 1, score).expect("twelve items offer subsets");
        let result = chosen.get();
        assert_eq!(result.indices().count(), result.len());
        for index in result.indices() {
            assert!(result.members() & (1 << index) != 0);
        }
    }

    #[test]
    fn an_empty_pool_is_refused_rather_than_scored() {
        let refused = optimise_subset(0, 0, 1, |_| 7.0).expect_err("there was nothing to choose");
        assert_eq!(refused.kind(), RefusalKind::Unreported);
    }

    #[test]
    fn a_subset_carries_the_search_that_found_it() {
        let chosen = optimise_subset(12, 12, 1, score).expect("twelve items offer subsets");
        assert!(chosen.trace().decided(), "every item was taken or left on its merits");
        assert!((chosen.cost() - chosen.get().score()).abs() < f64::EPSILON);
    }

    #[test]
    fn the_pool_ceiling_is_the_mask_width() {
        assert_eq!(MAX_POOL, u64::BITS as usize);
    }
}
