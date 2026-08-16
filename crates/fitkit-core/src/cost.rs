use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign};

/// The most rules one step of a search can be charged with breaking.
pub const RULES: u32 = 64;

/// What one step of a search costs: the rules it breaks, and how far it strays from the plain
/// reading.
///
/// The two are different kinds of statement and this type keeps them apart. A breach says a rule
/// was broken, which is reportable. Friction says only that a reading is the less usual one, which
/// is not. An engine that adds them together as bare numbers has to keep the scales apart by hand,
/// and three things go wrong the moment it stops looking: a long enough run of friction outweighs a
/// rule, a rule charged in two places is paid for twice, and a cost written as a saving turns the
/// search into an argument for the wrong answer. None of the three is representable here.
///
/// A rule is named by a number under [`RULES`], so charging the same rule twice within one step is
/// the same as charging it once. That is what makes it safe to move a rule from a pairwise check
/// into the state without hunting for the old charge: whichever place fires, the step is charged
/// once. Two steps that break the same rule are still two faults, because each step is priced on
/// its own.
///
/// Friction cannot be negative and cannot be infinite. A search minimises, so a negative cost is a
/// bribe rather than a rule, and it makes the objective incoherent.
///
/// The float the search consumes comes from [`Scale::price`], which is where the promise
/// that friction never reaches a breach is kept.
///
/// ```
/// use fitkit_core::Cost;
///
/// let doubled_tense = 3;
/// let by_the_state = Cost::breach(doubled_tense);
/// let by_the_pair = Cost::breach(doubled_tense);
/// assert_eq!(by_the_state + by_the_pair, Cost::breach(doubled_tense));
/// assert_eq!((by_the_state + by_the_pair).breaches(), 1);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    broken: u64,
    friction: f64,
}

impl Cost {
    /// Nothing broken and nothing unusual.
    pub const FREE: Self = Self { broken: 0, friction: 0.0 };

    /// Charge the step with breaking the rule named by `rule`.
    ///
    /// # Panics
    ///
    /// If `rule` is not under [`RULES`]. A rule that cannot be named cannot be counted once, and
    /// silently folding it onto another rule's bit would report the wrong fault.
    #[inline]
    #[must_use]
    pub fn breach(rule: u32) -> Self {
        assert!(rule < RULES, "rule {rule} cannot be named in {RULES} bits");
        Self { broken: 1 << rule, friction: 0.0 }
    }

    /// Charge the step with straying from the plain reading.
    ///
    /// # Panics
    ///
    /// If `amount` is negative, infinite, or not a number.
    #[inline]
    #[must_use]
    pub fn friction(amount: f64) -> Self {
        assert!(amount.is_finite(), "friction of {amount} is not a quantity");
        assert!(amount >= 0.0, "friction of {amount} is a bribe, not a rule");
        Self { broken: 0, friction: amount }
    }

    /// How many rules this step breaks.
    #[inline]
    #[must_use]
    pub const fn breaches(self) -> u32 {
        self.broken.count_ones()
    }

    /// Whether the rule named by `rule` is among them.
    #[inline]
    #[must_use]
    pub const fn breaks(self, rule: u32) -> bool {
        rule < RULES && self.broken >> rule & 1 == 1
    }

    /// Whether any rule is broken here.
    #[inline]
    #[must_use]
    pub const fn is_clean(self) -> bool {
        self.broken == 0
    }

    /// How far this step strays, with no rule counted.
    #[inline]
    #[must_use]
    pub const fn strain(self) -> f64 {
        self.friction
    }
}

impl Add for Cost {
    type Output = Self;

    /// Rules unite and friction adds, so a rule charged twice in one step is paid for once.
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { broken: self.broken | other.broken, friction: self.friction + other.friction }
    }
}

impl AddAssign for Cost {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sum for Cost {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(steps: I) -> Self {
        steps.fold(Self::FREE, Add::add)
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(out, "{} broken, {:.3} strain", self.breaches(), self.friction)
    }
}

/// How a [`Cost`] is turned into the single number a search minimises.
///
/// The gap between one breach and the next is set wider than every step of friction the longest
/// subject can accumulate, so no run of friction can ever reach a rule. That is the whole reason
/// the scale is a type: an engine that picks two constants by eye has to re-derive that promise
/// every time it adds a friction or lengthens its input, and it will not.
///
/// ```
/// use fitkit_core::{Cost, Scale};
///
/// let scale = Scale::over(100, 1.0);
/// let all_the_friction_there_can_be = Cost::friction(1.0 * 100.0);
/// assert!(scale.price(all_the_friction_there_can_be) < scale.price(Cost::breach(0)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    breach: f64,
}

impl Scale {
    /// A scale for a subject of at most `steps` steps, each straying at most `ceiling`.
    ///
    /// # Panics
    ///
    /// If `ceiling` is not a finite, non-negative quantity, or if the two together overflow.
    #[must_use]
    pub fn over(steps: usize, ceiling: f64) -> Self {
        assert!(ceiling.is_finite() && ceiling >= 0.0, "a ceiling of {ceiling} is not a quantity");
        #[allow(clippy::cast_precision_loss)] // a step count large enough to lose bits here is
        // already beyond what any search can walk
        let most = ceiling * steps as f64;
        assert!(most.is_finite(), "{steps} steps of {ceiling} is not a quantity");
        Self { breach: most + 1.0 }
    }

    /// What one breach costs, and therefore what friction may never reach.
    #[inline]
    #[must_use]
    pub const fn breach(self) -> f64 {
        self.breach
    }

    /// The number the search minimises.
    ///
    /// # Panics
    ///
    /// If the step strays further than the ceiling this scale was built for, because beyond it the
    /// promise that friction never reaches a rule no longer holds.
    #[inline]
    #[must_use]
    pub fn price(self, cost: Cost) -> f64 {
        assert!(
            cost.strain() < self.breach,
            "a single step strayed {} where one broken rule costs {}",
            cost.strain(),
            self.breach
        );
        // Not mul_add: this crate builds without std, where the fused instruction is unavailable,
        // and the two terms are held apart by the scale rather than by the last bit of precision.
        #[allow(clippy::suboptimal_flops)]
        {
            f64::from(cost.breaches()) * self.breach + cost.strain()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cost, Scale, RULES};

    #[test]
    fn a_rule_charged_in_two_places_is_paid_for_once() {
        let twice = Cost::breach(7) + Cost::breach(7);
        assert_eq!(twice.breaches(), 1);
        assert_eq!(twice, Cost::breach(7));
    }

    #[test]
    fn two_different_rules_are_two_faults() {
        assert_eq!((Cost::breach(1) + Cost::breach(2)).breaches(), 2);
    }

    #[test]
    fn friction_adds_where_rules_unite() {
        let strained = Cost::friction(0.25) + Cost::friction(0.25);
        assert!((strained.strain() - 0.5).abs() < 1e-12);
        assert!(strained.is_clean());
    }

    #[test]
    #[should_panic(expected = "a bribe")]
    fn a_negative_cost_is_refused() {
        let _ = Cost::friction(-1.0);
    }

    #[test]
    #[should_panic(expected = "not a quantity")]
    fn an_infinite_friction_is_refused() {
        let _ = Cost::friction(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "cannot be named")]
    fn a_rule_that_cannot_be_named_is_refused() {
        let _ = Cost::breach(RULES);
    }

    #[test]
    fn no_run_of_friction_reaches_a_rule() {
        let scale = Scale::over(500, 0.9);
        let worst: Cost = (0..500).map(|_| Cost::friction(0.9)).sum();
        assert!(scale.price(worst) < scale.price(Cost::breach(0)));
    }

    #[test]
    fn a_broken_rule_outweighs_every_friction_beneath_it() {
        let scale = Scale::over(10, 1.0);
        assert!(scale.price(Cost::breach(0)) > scale.price(Cost::friction(10.0)));
        assert!(scale.price(Cost::breach(0) + Cost::breach(1)) > scale.price(Cost::breach(0)));
    }

    #[test]
    fn a_free_step_costs_nothing() {
        assert!(Scale::over(4, 1.0).price(Cost::FREE).abs() < f64::EPSILON);
        assert!(Cost::FREE.is_clean());
    }

    #[test]
    fn a_step_names_the_rules_it_broke() {
        let broken = Cost::breach(3) + Cost::friction(0.1);
        assert!(broken.breaks(3));
        assert!(!broken.breaks(4));
        assert!(!broken.is_clean());
    }

    #[test]
    #[should_panic(expected = "a single step strayed")]
    fn a_step_beyond_the_ceiling_is_caught_rather_than_outweighing_a_rule() {
        let _ = Scale::over(2, 1.0).price(Cost::friction(99.0));
    }
}
