use core::fmt;

/// A result and the search that produced it.
///
/// The point of the type is what it does not have: no public constructor, no `From`, no `Default`.
/// The only values of this type in any program are ones a decode in this crate returned, so a
/// consumer that asks for a `Chosen<T>` cannot be handed a `T` someone assembled by hand. A
/// function that must return one has to run the search, and running the search is the work.
///
/// The membrane is one-way on purpose. [`Chosen::into_inner`] lets a value out, because
/// inspection, benchmarking and the exhaustive checks in this crate need the bare result. Nothing
/// puts a value back in. [`Chosen::map`] is how a decoded result is carried forward without
/// losing the account of where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Chosen<T> {
    value: T,
    cost: f64,
    trace: Trace,
}

impl<T> Chosen<T> {
    /// Built only by a decode in this crate.
    pub(crate) const fn new(value: T, cost: f64, trace: Trace) -> Self {
        Self { value, cost, trace }
    }

    /// Total cost paid for this result, which makes two models comparable.
    pub const fn cost(&self) -> f64 {
        self.cost
    }

    /// What the search did to arrive here.
    pub const fn trace(&self) -> Trace {
        self.trace
    }

    /// The result, borrowed.
    pub const fn get(&self) -> &T {
        &self.value
    }

    /// Carry the result forward, keeping the account of the search that produced it.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Chosen<U> {
        Chosen { value: f(self.value), cost: self.cost, trace: self.trace }
    }

    /// The result, taken out. There is no way to put one back.
    #[allow(clippy::missing_const_for_fn)] // destructuring a generic in const fn needs a later MSRV
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// What a search considered, and what it turned down.
///
/// A witness proves a decode ran. These numbers are what say whether it had anything to decide,
/// and they separate two things that are easy to confuse. `choices` counts the decisions where the
/// model offered more than one candidate to price; a search that never reaches one was handed a
/// corridor and did no work. `rejected` counts alternatives that were affordable and still lost,
/// so it stays at zero when the evidence, rather than the model, removed the competition.
///
/// The refusal is on the first, not the second. A model with one candidate is a formality dressed
/// as a search. A model offering three candidates on evidence so clear that two are impossible did
/// exactly what it was asked to do, and refusing it would punish a decisive answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Trace {
    steps: usize,
    states: usize,
    considered: u64,
    choices: u64,
    rejected: u64,
    margin: Ordered,
}

impl Trace {
    pub(crate) fn new(steps: usize, states: usize, tally: Tally) -> Self {
        Self {
            steps,
            states,
            considered: tally.considered,
            choices: tally.choices,
            rejected: tally.rejected,
            margin: Ordered(tally.margin),
        }
    }

    /// Steps in the problem.
    pub const fn steps(&self) -> usize {
        self.steps
    }

    /// Candidates the model offered.
    pub const fn states(&self) -> usize {
        self.states
    }

    /// Candidates the search priced.
    pub const fn considered(&self) -> u64 {
        self.considered
    }

    /// Decisions where more than one candidate was priced.
    pub const fn choices(&self) -> u64 {
        self.choices
    }

    /// Affordable candidates that lost to the one kept.
    ///
    /// Zero alongside a positive [`Trace::choices`] means the evidence settled every decision on
    /// its own: alternatives were offered and none of them could be paid for.
    pub const fn rejected(&self) -> u64 {
        self.rejected
    }

    /// How much worse the best whole alternative was, at the end of the path.
    ///
    /// Infinite means no alternative finished at all, which is a missing candidate rather than a
    /// safe result.
    pub const fn margin(&self) -> f64 {
        self.margin.0
    }

    /// Whether the model ever gave the search two candidates to weigh.
    ///
    /// False means every decision had one candidate, so the path was the only path and no dynamic
    /// programme was needed to find it. A decode that reports this refuses rather than returning a
    /// witness, since the alternative is a result that looks searched and was not.
    pub const fn decided(&self) -> bool {
        self.choices > 0
    }
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} steps, {} states, {} considered, {} choices, {} rejected, margin {}",
            self.steps, self.states, self.considered, self.choices, self.rejected, self.margin.0
        )
    }
}

/// A margin that takes part in `Eq`, so [`Trace`] can. Two traces are the same trace when they
/// record the same search, and a margin of infinity equals itself.
#[derive(Clone, Copy, Debug)]
struct Ordered(f64);

impl Default for Ordered {
    fn default() -> Self {
        Self(f64::INFINITY)
    }
}

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Ordered {}

/// Running counts kept by a decode, turned into a [`Trace`] when it finishes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tally {
    pub(crate) considered: u64,
    pub(crate) choices: u64,
    pub(crate) rejected: u64,
    pub(crate) margin: f64,
}

impl Tally {
    pub(crate) const fn new() -> Self {
        Self { considered: 0, choices: 0, rejected: 0, margin: f64::INFINITY }
    }

    /// One decision: how many candidates the model offered for it, and how many of those could be
    /// paid for. Offering two is what makes it a decision; affording two is what makes the answer
    /// contested rather than merely correct.
    pub(crate) const fn decision(&mut self, priced: u64, affordable: u64) {
        self.considered += priced;
        if priced > 1 {
            self.choices += 1;
        }
        self.rejected += affordable.saturating_sub(1);
    }

    /// The end of the path: how much worse the best alternative finished.
    pub(crate) fn ended(&mut self, best: f64, runner_up: f64) {
        self.margin = if runner_up.is_finite() && best.is_finite() {
            runner_up - best
        } else {
            f64::INFINITY
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{Chosen, Tally, Trace};

    fn witness(priced: u64, affordable: u64) -> Chosen<u8> {
        let mut tally = Tally::new();
        tally.decision(priced, affordable);
        tally.ended(1.0, 3.0);
        Chosen::new(7, 1.0, Trace::new(4, 2, tally))
    }

    #[test]
    fn a_model_offering_one_candidate_decided_nothing() {
        assert!(!witness(1, 1).trace().decided());
        assert!(witness(2, 2).trace().decided());
    }

    #[test]
    fn evidence_that_settles_a_real_choice_still_counts_as_one() {
        // Two candidates were offered and only one could be paid for. The model did its part.
        let trace = witness(2, 1).trace();
        assert!(trace.decided());
        assert_eq!(trace.rejected(), 0, "nothing affordable was turned down");
    }

    #[test]
    fn mapping_keeps_the_account_of_the_search() {
        let mapped = witness(3, 3).map(|value| u32::from(value) * 2);
        assert_eq!(*mapped.get(), 14);
        assert_eq!(mapped.trace().rejected(), 2);
        assert!((mapped.cost() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn a_margin_needs_two_finishers() {
        let mut tally = Tally::new();
        tally.ended(1.0, f64::INFINITY);
        assert!(!Trace::new(1, 1, tally).margin().is_finite());
        tally.ended(1.0, 2.5);
        assert!((Trace::new(1, 1, tally).margin() - 1.5).abs() < f64::EPSILON);
    }
}
