use alloc::vec;
use alloc::vec::Vec;

use fitkit_core::{Answer, Confidence, Evidence, Refusal, Span};

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
/// [`optimise_subset`] returned. A caller that wants a particular membership has to say what makes
/// its items worth having, in [`Terms`], and let the search find them; there is no route that
/// writes the mask down and passes it off as a search result.
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

/// What makes a subset worth choosing, stated over the items rather than over the answer.
///
/// # Why this is a value and not a closure
///
/// This search once took `Fn(u64) -> f64`: the caller's objective was handed the whole candidate
/// subset. That is enough rope to write down the answer. `|members| if members == 0b1011 { 1.0 }
/// else { 0.0 }` is one line, it is not a model of anything, and the search around it is real —
/// every subset genuinely enumerated, a genuine optimum returned, a witness with an honest cost
/// and an honest count of what it weighed. Sealing the result type does not touch it, because
/// nothing was forged: the search truly ran, over an objective that had already decided.
///
/// The path decode never had this hole. Its objective arrives as `emission(step, state)` and
/// `transition(from, to)`, so it is asked about one position and one pair at a time and cannot see
/// the path it is scoring. It is local by construction, and a local objective has to be a claim
/// about the parts. `Terms` gives the subset search the same shape.
///
/// A caller says what each item is worth on its own, what a pair of them is worth together, which
/// items are mandatory or excluded, and how many may be taken. Nothing here can mention a subset,
/// so an objective that returns a predetermined membership is not expressible — not discouraged,
/// not linted against, not detected after the fact. There is no argument to write it in.
///
/// # Why a weight is evidence and not a number
///
/// Arity alone would still leave a bare `f64` per item, and a bare `f64` is where a template gets
/// in: a constant somebody tuned until the output looked right, with nothing behind it and no way
/// to ask what it was measured from. [`Evidence`] is already this framework's stated only way
/// facts enter a solver, and the subset search was the one place that took a number instead.
///
/// So a weight arrives as a measurement: the [`Span`] of the problem it speaks for, and the
/// [`Confidence`] it is held with. The contribution to the score is the magnitude scaled by trust,
/// so a weakly supported claim is discounted rather than competing at full strength, and evidence
/// that carries no information — zero confidence, or an empty span — is refused at the door rather
/// than counted as zero. What the search chose can therefore be traced back to the regions that
/// argued for it, through [`support`](Self::support), and reported with the trust it rests on,
/// through [`trust`](Self::trust).
///
/// # What this does not reach
///
/// A caller can still compute a magnitude however it likes, and can point a span at a region that
/// does not really support it. That is the honest side of the line and no type reaches it: what is
/// closed is the shape. An objective here is a bounded-arity claim about items, each carrying a
/// citation and a trust, so it applies to every pool those items appear in and a different pool
/// gives a different answer. A lookup table keyed on the answer generalises to nothing and can no
/// longer be written down.
///
/// Interactions stop at pairs. Every pseudo-boolean function is some polynomial in the membership
/// bits, and cutting it at degree two is what keeps the objective smaller than the space of
/// answers: `pool` weights and `pool * (pool - 1) / 2` pairs against `2^pool` subsets. The
/// constraints below carry the higher-order structure that is actually asked for in practice, and
/// each is a named declaration rather than a number that happens to dominate.
#[derive(Clone, Debug, PartialEq)]
pub struct Terms {
    pool: usize,
    worth: Vec<Option<Evidence<f64>>>,
    pairs: Vec<(usize, usize, Evidence<f64>)>,
    required: u64,
    forbidden: u64,
    settled: Vec<(usize, Span)>,
    floor: usize,
    ceiling: usize,
}

impl Terms {
    /// Terms over a pool of `pool` items, every one of them worth nothing until told otherwise.
    ///
    /// # Errors
    ///
    /// Refuses an empty pool, which offers no subset to choose between, and a pool wider than
    /// [`MAX_POOL`], which the mask cannot hold.
    pub fn over(pool: usize) -> Answer<Self> {
        if pool == 0 {
            return Err(Refusal::unreported("an empty pool offers no subset to choose"));
        }
        if pool > MAX_POOL {
            return Err(Refusal::incoherent("a pool wider than the mask that holds it"));
        }
        Ok(Self {
            pool,
            worth: vec![None; pool],
            pairs: Vec::new(),
            required: 0,
            forbidden: 0,
            settled: Vec::new(),
            floor: 0,
            ceiling: pool,
        })
    }

    /// What the evidence says taking `item` is worth on its own.
    ///
    /// The magnitude is scaled by the confidence it is held with, so a claim the source barely
    /// supports argues for its item weakly rather than at full strength.
    ///
    /// # Errors
    ///
    /// Refuses an item outside the pool, and a magnitude that is not finite: an infinity here
    /// would be a hard constraint smuggled in as a weight, where [`require`](Self::require) and
    /// [`forbid`](Self::forbid) say the same thing in the open, and a `NaN` would silently lose
    /// every comparison it entered. Refuses evidence that is not informative, because a weight
    /// resting on nothing is the constant this argument exists to keep out.
    pub fn worth(mut self, item: usize, evidence: Evidence<f64>) -> Answer<Self> {
        self.within(item)?;
        Self::measured(&evidence)?;
        self.worth[item] = Some(evidence);
        Ok(self)
    }

    /// What taking `a` and `b` together is worth, beyond what each is worth alone.
    ///
    /// This is where redundancy and complement are said: two items that repeat each other are
    /// worth less together than apart, and two that complete each other are worth more.
    ///
    /// # Errors
    ///
    /// Refuses an item outside the pool, a pair of one item with itself, a magnitude that is not
    /// finite, and evidence that is not informative.
    pub fn together(mut self, a: usize, b: usize, evidence: Evidence<f64>) -> Answer<Self> {
        self.within(a)?;
        self.within(b)?;
        if a == b {
            return Err(Refusal::incoherent("an item paired with itself is its own worth"));
        }
        Self::measured(&evidence)?;
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        self.pairs.retain(|&(x, y, _)| (x, y) != (low, high));
        self.pairs.push((low, high, evidence));
        Ok(self)
    }

    /// Take `item` in every subset considered, on the strength of `because`.
    ///
    /// A hard constraint is the strongest thing that can be said here, so it has to say where it
    /// comes from. There is no confidence argument on purpose: a requirement held at less than
    /// full trust is a weight, and belongs in [`worth`](Self::worth).
    ///
    /// # Errors
    ///
    /// Refuses an item outside the pool, one already forbidden, and an empty span.
    pub fn require(mut self, item: usize, because: Span) -> Answer<Self> {
        self.within(item)?;
        if self.forbidden & (1 << item) != 0 {
            return Err(Refusal::incoherent("an item both required and forbidden"));
        }
        Self::cited(because)?;
        self.required |= 1 << item;
        self.settled.push((item, because));
        Ok(self)
    }

    /// Take `item` in no subset considered, on the strength of `because`.
    ///
    /// # Errors
    ///
    /// Refuses an item outside the pool, one already required, and an empty span.
    pub fn forbid(mut self, item: usize, because: Span) -> Answer<Self> {
        self.within(item)?;
        if self.required & (1 << item) != 0 {
            return Err(Refusal::incoherent("an item both required and forbidden"));
        }
        Self::cited(because)?;
        self.forbidden |= 1 << item;
        self.settled.push((item, because));
        Ok(self)
    }

    /// Consider no subset larger than `count`.
    ///
    /// # Errors
    ///
    /// Refuses a ceiling below the floor already set.
    pub fn at_most(mut self, count: usize) -> Answer<Self> {
        if count < self.floor {
            return Err(Refusal::incoherent("a ceiling below the floor"));
        }
        self.ceiling = count.min(self.pool);
        Ok(self)
    }

    /// Consider no subset smaller than `count`.
    ///
    /// # Errors
    ///
    /// Refuses a floor above the ceiling already set, or above the pool.
    pub fn at_least(mut self, count: usize) -> Answer<Self> {
        if count > self.ceiling {
            return Err(Refusal::incoherent("a floor above the ceiling"));
        }
        self.floor = count;
        Ok(self)
    }

    /// How many items the terms are over.
    #[must_use]
    pub const fn pool(&self) -> usize {
        self.pool
    }

    fn within(&self, item: usize) -> Answer<()> {
        if item >= self.pool {
            return Err(Refusal::incoherent("an item outside the pool"));
        }
        Ok(())
    }

    /// The spans that argued for `members`, in the order they were declared.
    ///
    /// What the search chose, traced back to the regions of the problem that supported it. A
    /// caller reporting a result can name these rather than assert the result.
    #[must_use]
    pub fn support(&self, members: u64) -> Vec<Span> {
        let mut spans = Vec::new();
        for (item, worth) in self.worth.iter().enumerate() {
            if members & (1 << item) != 0 {
                if let Some(evidence) = worth {
                    spans.push(evidence.span);
                }
            }
        }
        for &(a, b, evidence) in &self.pairs {
            if members & (1 << a) != 0 && members & (1 << b) != 0 {
                spans.push(evidence.span);
            }
        }
        for &(item, span) in &self.settled {
            if members & (1 << item) != 0 || self.forbidden & (1 << item) != 0 {
                spans.push(span);
            }
        }
        spans
    }

    /// How far `members` is trusted: the weakest confidence among the evidence behind it.
    ///
    /// A subset is only as good as its shakiest support, so this takes the minimum rather than
    /// combining trust upward. A subset resting on no evidence at all is [`Confidence::ZERO`].
    #[must_use]
    pub fn trust(&self, members: u64) -> Confidence {
        let mut weakest: Option<Confidence> = None;
        let mut hold = |confidence: Confidence| {
            weakest = Some(weakest.map_or(confidence, |held: Confidence| held.min(confidence)));
        };
        for (item, worth) in self.worth.iter().enumerate() {
            if members & (1 << item) != 0 {
                if let Some(evidence) = worth {
                    hold(evidence.confidence);
                }
            }
        }
        for &(a, b, evidence) in &self.pairs {
            if members & (1 << a) != 0 && members & (1 << b) != 0 {
                hold(evidence.confidence);
            }
        }
        weakest.unwrap_or(Confidence::ZERO)
    }

    fn measured(evidence: &Evidence<f64>) -> Answer<()> {
        if !evidence.value.is_finite() {
            return Err(Refusal::incoherent("a weight that is not a finite number"));
        }
        if !evidence.is_informative() {
            return Err(Refusal::unreported(
                "a weight resting on no span or no confidence is a constant, not a measurement",
            ));
        }
        Ok(())
    }

    fn cited(span: Span) -> Answer<()> {
        if span.is_empty() {
            return Err(Refusal::unreported("a constraint that cites no region"));
        }
        Ok(())
    }

    /// What `members` is worth, ignoring whether it is allowed.
    ///
    /// Each magnitude is scaled by the trust it is held with, so the score is the total of what
    /// the evidence actually supports rather than what it would say if every claim were certain.
    fn value(&self, members: u64) -> f64 {
        let mut total = 0.0;
        for (item, worth) in self.worth.iter().enumerate() {
            if members & (1 << item) != 0 {
                if let Some(evidence) = worth {
                    total += evidence.confidence.apply(evidence.value);
                }
            }
        }
        for &(a, b, evidence) in &self.pairs {
            if members & (1 << a) != 0 && members & (1 << b) != 0 {
                total += evidence.confidence.apply(evidence.value);
            }
        }
        total
    }

    /// Whether `members` is a subset the declarations allow.
    fn allows(&self, members: u64) -> bool {
        let size = members.count_ones() as usize;
        members & self.forbidden == 0
            && members & self.required == self.required
            && size >= self.floor
            && size <= self.ceiling
    }
}

/// Choose the best subset the [`Terms`] allow.
///
/// Enumerates every subset when the pool is at most `exact_limit`, otherwise runs a beam keeping
/// `beam_width` states per size. [`SubsetResult::solver`] reports which ran, so a beam answer is
/// never mistaken for a proven optimum.
///
/// The result comes back inside the witness the search produced, on the same terms as
/// [`decode_path_with_cost`](crate::decode_path_with_cost): a caller can read it and carry it
/// onward but cannot build one, so a subset that reaches a consumer is a subset something searched
/// for. The witness carries the score as its cost and an account of what was weighed.
///
/// # Errors
///
/// Refuses terms no subset can satisfy, which is a contradiction among the declarations rather
/// than a search that failed, and is worth saying rather than answering with the empty set.
///
/// # Panics
///
/// If exact enumeration is requested for a pool of 63 or more.
pub fn optimise_subset(
    terms: &Terms,
    exact_limit: usize,
    beam_width: usize,
) -> Answer<Chosen<SubsetResult>> {
    let pool = terms.pool;
    let mut tally = Tally::new();
    let best = if pool <= exact_limit {
        assert!(pool < 63, "exact enumeration of {pool} items is not affordable");
        exact(terms, &mut tally)
    } else {
        beam(terms, beam_width.max(1), &mut tally)
    };
    let Some(best) = best else {
        return Err(Refusal::incoherent("no subset satisfies the terms as they were declared"));
    };
    Ok(Chosen::new(best, best.score, Trace::new(pool, pool, tally)))
}

/// Choose the best subset, and build the caller's own result in the same call.
///
/// The counterpart of [`decode_path_as`](crate::decode_path_as), and there for the same reason:
/// [`Chosen`] has no public `map`, so a witness over a caller's type is built by the mechanism
/// rather than transformed out of one already in hand.
///
/// # Errors
///
/// As [`optimise_subset`].
///
/// # Panics
///
/// As [`optimise_subset`].
pub fn optimise_subset_as<B, U>(
    terms: &Terms,
    exact_limit: usize,
    beam_width: usize,
    build: B,
) -> Answer<Chosen<U>>
where
    B: FnOnce(&SubsetResult) -> U,
{
    optimise_subset(terms, exact_limit, beam_width)
        .map(|chosen| chosen.map(|result| build(&result)))
}

fn exact(terms: &Terms, tally: &mut Tally) -> Option<SubsetResult> {
    let pool = terms.pool;
    let mut best: Option<SubsetResult> = None;
    let mut runner_up = f64::NEG_INFINITY;
    let subsets = 1_u64 << pool;
    for members in 0..subsets {
        if !terms.allows(members) {
            continue;
        }
        let value = terms.value(members);
        match best {
            Some(ref mut held) if value > held.score => {
                runner_up = held.score;
                held.members = members;
                held.score = value;
            }
            Some(_) => {
                if value > runner_up {
                    runner_up = value;
                }
            }
            None => best = Some(SubsetResult::new(members, value, Solver::Exact)),
        }
    }
    let best = best?;
    // Every item the terms left open was taken or left in every combination, so each one was a
    // decision with both branches on the table. An item the terms settled was never a decision.
    let settled = (terms.required | terms.forbidden).count_ones() as usize;
    for _ in 0..pool.saturating_sub(settled) {
        tally.decision(2, 2);
    }
    tally.ended(best.score, runner_up);
    Some(best)
}

fn beam(terms: &Terms, width: usize, tally: &mut Tally) -> Option<SubsetResult> {
    let pool = terms.pool;
    let seed = terms.required;
    let mut frontier = vec![(seed, terms.value(seed))];
    let mut best = terms.allows(seed).then(|| (seed, terms.value(seed)));
    let mut runner_up = f64::NEG_INFINITY;
    let mut grown: Vec<(u64, f64)> = Vec::with_capacity(width * pool);

    for _ in 0..pool {
        grown.clear();
        for &(members, _) in &frontier {
            let first = MAX_POOL - members.leading_zeros() as usize;
            for item in first..pool {
                let bit = 1 << item;
                if terms.forbidden & bit != 0 || members & bit != 0 {
                    continue;
                }
                let candidate = members | bit;
                if candidate.count_ones() as usize > terms.ceiling {
                    continue;
                }
                grown.push((candidate, terms.value(candidate)));
            }
        }
        if grown.is_empty() {
            break;
        }
        grown.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let offered = grown.len();
        grown.truncate(width);
        tally.decision(offered as u64, grown.len() as u64);
        for &(members, value) in &grown {
            if !terms.allows(members) {
                continue;
            }
            match best {
                Some((_, held)) if value > held => {
                    runner_up = held;
                    best = Some((members, value));
                }
                Some(_) => {
                    if value > runner_up {
                        runner_up = value;
                    }
                }
                None => best = Some((members, value)),
            }
        }
        frontier.clear();
        frontier.extend_from_slice(&grown);
    }

    let (members, score) = best?;
    tally.ended(score, runner_up);
    Some(SubsetResult::new(members, score, Solver::Beam { width }))
}

#[cfg(test)]
mod tests {
    use fitkit_core::{Confidence, Evidence, RefusalKind, Span};

    use super::{optimise_subset, Solver, Terms, MAX_POOL};

    /// A pairwise model with a size penalty, so the optimum is an interior subset.
    ///
    /// Every weight cites a span, because there is no argument that takes anything else.
    fn terms(pool: usize) -> Terms {
        let mut built = Terms::over(pool).expect("a pool to choose from");
        for item in 0..pool {
            let span = Span::new(item, item + 1);
            let worth = 1.0 - 0.7 * (item % 3) as f64;
            built = built
                .worth(item, Evidence::certain(span, worth))
                .expect("a finite weight over a real span");
        }
        for a in 0..pool {
            for b in a + 1..pool {
                let value = ((a * 31 + b * 17) % 13) as f64 / 13.0 - 0.7;
                built = built
                    .together(a, b, Evidence::certain(Span::new(a, b + 1), value))
                    .expect("a finite weight over a real span");
            }
        }
        built
    }

    #[test]
    fn the_beam_matches_exact_enumeration_where_both_can_run() {
        let model = terms(12);
        let exact = optimise_subset(&model, 12, 1).expect("twelve items offer subsets");
        let beam = optimise_subset(&model, 0, 64).expect("twelve items offer subsets");
        let (exact, beam) = (exact.get(), beam.get());
        assert!(exact.is_proven());
        assert_eq!(beam.solver(), Solver::Beam { width: 64 });
        assert!((exact.score() - beam.score()).abs() < 1e-12, "beam {beam:?} exact {exact:?}");
    }

    #[test]
    fn a_beam_result_is_never_claimed_as_proven() {
        let beamed = optimise_subset(&terms(30), 20, 8).expect("thirty items offer subsets");
        assert!(!beamed.get().is_proven());
    }

    #[test]
    fn members_and_indices_agree() {
        let chosen = optimise_subset(&terms(12), 12, 1).expect("twelve items offer subsets");
        let result = chosen.get();
        assert_eq!(result.indices().count(), result.len());
        for index in result.indices() {
            assert!(result.members() & (1 << index) != 0);
        }
    }

    #[test]
    fn an_empty_pool_is_refused_rather_than_scored() {
        let refused = Terms::over(0).expect_err("there was nothing to choose");
        assert_eq!(refused.kind(), RefusalKind::Unreported);
    }

    #[test]
    fn a_subset_carries_the_search_that_found_it() {
        let chosen = optimise_subset(&terms(12), 12, 1).expect("twelve items offer subsets");
        assert!(chosen.trace().decided(), "every item was taken or left on its merits");
        assert!((chosen.cost() - chosen.get().score()).abs() < f64::EPSILON);
    }

    #[test]
    fn a_weight_resting_on_nothing_is_refused() {
        let model = Terms::over(4).expect("a pool");
        let nowhere = model
            .clone()
            .worth(0, Evidence::certain(Span::new(3, 3), 9.0))
            .expect_err("a weight over an empty span");
        assert_eq!(nowhere.kind(), RefusalKind::Unreported);

        let untrusted = model
            .clone()
            .worth(0, Evidence::new(Span::new(0, 1), Confidence::ZERO, 9.0))
            .expect_err("a weight nobody trusts");
        assert_eq!(untrusted.kind(), RefusalKind::Unreported);

        let infinite = model
            .worth(0, Evidence::certain(Span::new(0, 1), f64::INFINITY))
            .expect_err("a constraint dressed as a weight");
        assert_eq!(infinite.kind(), RefusalKind::Incoherent);
    }

    #[test]
    fn trust_discounts_a_claim_rather_than_letting_it_argue_at_full_strength() {
        let span = Span::new(0, 2);
        let sure = Terms::over(2)
            .expect("a pool")
            .worth(0, Evidence::certain(span, 10.0))
            .expect("a weight");
        let unsure = Terms::over(2)
            .expect("a pool")
            .worth(0, Evidence::new(span, Confidence::new(0.25), 10.0))
            .expect("a weight");
        let sure = optimise_subset(&sure, 2, 1).expect("subsets");
        let unsure = optimise_subset(&unsure, 2, 1).expect("subsets");
        assert!(unsure.cost() < sure.cost(), "a shakier claim argued less hard");
        assert_eq!(unsure.get().members(), sure.get().members());
    }

    #[test]
    fn a_chosen_subset_names_the_regions_that_argued_for_it() {
        let model = terms(6);
        let chosen = optimise_subset(&model, 6, 1).expect("six items offer subsets");
        let members = chosen.get().members();
        assert!(!model.support(members).is_empty(), "the choice cites its evidence");
        assert!(!model.trust(members).is_zero(), "and reports what it rests on");
    }

    #[test]
    fn declarations_that_contradict_each_other_are_refused() {
        let both = Terms::over(4)
            .expect("a pool")
            .require(1, Span::new(0, 1))
            .expect("a cited requirement")
            .forbid(1, Span::new(0, 1))
            .expect_err("required and forbidden at once");
        assert_eq!(both.kind(), RefusalKind::Incoherent);

        let uncited = Terms::over(4)
            .expect("a pool")
            .require(1, Span::new(2, 2))
            .expect_err("a requirement citing nothing");
        assert_eq!(uncited.kind(), RefusalKind::Unreported);
    }

    #[test]
    fn a_requirement_holds_and_a_ceiling_binds() {
        let model = terms(8)
            .require(7, Span::new(7, 8))
            .expect("a cited requirement")
            .forbid(0, Span::new(0, 1))
            .expect("a cited exclusion")
            .at_most(3)
            .expect("a budget");
        for solver in [(8_usize, 1_usize), (0, 64)] {
            let chosen = optimise_subset(&model, solver.0, solver.1).expect("subsets remain");
            let members = chosen.get().members();
            assert!(members & (1 << 7) != 0, "the required item was taken");
            assert_eq!(members & 1, 0, "the forbidden item was left");
            assert!(chosen.get().len() <= 3, "the budget bound the answer");
        }
    }

    #[test]
    fn terms_no_subset_can_satisfy_are_refused_rather_than_answered() {
        let impossible = Terms::over(4)
            .expect("a pool")
            .require(0, Span::new(0, 1))
            .expect("a cited requirement")
            .require(1, Span::new(1, 2))
            .expect("a cited requirement")
            .at_most(1)
            .expect("a budget");
        let refused = optimise_subset(&impossible, 4, 1).expect_err("nothing satisfies these");
        assert_eq!(refused.kind(), RefusalKind::Incoherent);
    }

    #[test]
    fn the_pool_ceiling_is_the_mask_width() {
        assert_eq!(MAX_POOL, u64::BITS as usize);
        assert!(Terms::over(MAX_POOL + 1).is_err(), "wider than the mask that holds it");
    }
}
