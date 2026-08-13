use alloc::vec;
use alloc::vec::Vec;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubsetResult {
    /// Bit `i` set means item `i` is a member.
    pub members: u64,
    /// Score of `members`.
    pub score: f64,
    /// Exact or beam.
    pub solver: Solver,
}

impl SubsetResult {
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
/// # Panics
///
/// If `pool` exceeds [`MAX_POOL`], or if exact enumeration is requested for `pool >= 63`.
pub fn optimise_subset<S>(
    pool: usize,
    exact_limit: usize,
    beam_width: usize,
    score: S,
) -> SubsetResult
where
    S: Fn(u64) -> f64,
{
    assert!(pool <= MAX_POOL, "pool of {pool} exceeds the {MAX_POOL} bit mask");
    if pool == 0 {
        return SubsetResult { members: 0, score: score(0), solver: Solver::Exact };
    }
    if pool <= exact_limit {
        assert!(pool < 63, "exact enumeration of {pool} items is not affordable");
        return exact(pool, &score);
    }
    beam(pool, beam_width.max(1), &score)
}

fn exact<S: Fn(u64) -> f64>(pool: usize, score: &S) -> SubsetResult {
    let mut best = SubsetResult { members: 0, score: f64::NEG_INFINITY, solver: Solver::Exact };
    for members in 0..(1_u64 << pool) {
        let value = score(members);
        if value > best.score {
            best.members = members;
            best.score = value;
        }
    }
    best
}

fn beam<S: Fn(u64) -> f64>(pool: usize, width: usize, score: &S) -> SubsetResult {
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
        grown.truncate(width);
        if grown[0].1 > best.1 {
            best = grown[0];
        }
        frontier.clear();
        frontier.extend_from_slice(&grown);
    }

    SubsetResult { members: best.0, score: best.1, solver: Solver::Beam { width } }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

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
        let exact = optimise_subset(12, 12, 1, score);
        let beam = optimise_subset(12, 0, 64, score);
        assert!(exact.is_proven());
        assert_eq!(beam.solver, Solver::Beam { width: 64 });
        assert!((exact.score - beam.score).abs() < 1e-12, "beam {beam:?} exact {exact:?}");
    }

    #[test]
    fn a_beam_result_is_never_claimed_as_proven() {
        assert!(!optimise_subset(30, 20, 8, score).is_proven());
    }

    #[test]
    fn members_and_indices_agree() {
        let result = optimise_subset(12, 12, 1, score);
        assert_eq!(result.indices().count(), result.len());
        for index in result.indices() {
            assert!(result.members & (1 << index) != 0);
        }
    }

    #[test]
    fn an_empty_pool_scores_the_empty_set() {
        let result = optimise_subset(0, 0, 1, |_| 7.0);
        assert!(result.is_empty() && result.is_proven());
        assert!((result.score - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_pool_ceiling_is_the_mask_width() {
        assert_eq!(MAX_POOL, u64::BITS as usize);
    }
}
