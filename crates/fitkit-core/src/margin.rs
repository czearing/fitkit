use core::cmp::Ordering;
use core::fmt;

/// How much error an answer survives, in the units of the problem.
///
/// Distinct from [`Confidence`](crate::Confidence): confidence is how well the evidence supported
/// a choice, margin is how far that choice can be wrong before it stops holding. An answer with
/// full confidence and no margin is one measurement away from failing.
///
/// Never negative. `NaN` becomes [`Margin::NONE`], so a failed measurement cannot look safe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Margin(f64);

impl Margin {
    /// On the edge. Any error at all breaks the answer.
    pub const NONE: Self = Self(0.0);

    /// Nothing bounds the answer. Usually a missing constraint rather than a safe result.
    pub const UNBOUNDED: Self = Self(f64::INFINITY);

    /// Clamp a distance to `0..=inf`.
    #[inline]
    pub fn new(value: f64) -> Self {
        if value.is_nan() || value <= 0.0 {
            Self::NONE
        } else {
            Self(value)
        }
    }

    /// The distance.
    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Whether the answer sits on the edge.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 <= 0.0
    }

    /// Whether nothing bounds the answer.
    #[inline]
    pub fn is_unbounded(self) -> bool {
        self.0.is_infinite()
    }

    /// Whether the answer still holds after an error of this size.
    #[inline]
    pub fn survives(self, error: f64) -> bool {
        error.abs() < self.0
    }

    /// The tighter of two. The binding constraint is the one that matters.
    #[inline]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 {
            self
        } else {
            other
        }
    }
}

impl Default for Margin {
    fn default() -> Self {
        Self::NONE
    }
}

impl Eq for Margin {}

impl PartialOrd for Margin {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Margin {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl fmt::Display for Margin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unbounded() {
            return write!(f, "unbounded");
        }
        write!(f, "{:.4}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Margin;

    #[test]
    fn a_negative_distance_means_no_margin() {
        assert_eq!(Margin::new(-1.0), Margin::NONE);
        assert!(Margin::new(f64::NAN).is_none());
    }

    #[test]
    fn an_answer_survives_less_error_than_its_margin() {
        let margin = Margin::new(0.5);
        assert!(margin.survives(0.49));
        assert!(!margin.survives(0.5), "the edge is not survived");
        assert!(!Margin::NONE.survives(0.0));
    }

    #[test]
    fn the_binding_constraint_wins() {
        assert_eq!(Margin::new(2.0).min(Margin::new(0.25)), Margin::new(0.25));
    }
}
