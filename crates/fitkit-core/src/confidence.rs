use core::cmp::Ordering;
use core::fmt;

/// Trust in a measurement, clamped to `0..=1`.
///
/// `NaN` becomes [`Confidence::ZERO`], so a failed measurement cannot win a comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Confidence(f64);

impl Confidence {
    /// No information. The span must be left unchanged.
    pub const ZERO: Self = Self(0.0);

    /// Full trust.
    pub const FULL: Self = Self(1.0);

    /// Clamp a float into `0..=1`.
    #[inline]
    pub fn new(value: f64) -> Self {
        if value.is_nan() {
            Self::ZERO
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// The weight.
    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Whether the span carries no information.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 <= 0.0
    }

    /// Both must hold. Multiplicative, so trust never rises.
    #[inline]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// The weaker of two.
    #[inline]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 {
            self
        } else {
            other
        }
    }

    /// Scale a magnitude by trust.
    #[inline]
    pub fn apply(self, magnitude: f64) -> f64 {
        magnitude * self.0
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Eq for Confidence {}

impl PartialOrd for Confidence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Confidence {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Confidence;

    #[test]
    fn out_of_range_is_clamped() {
        assert_eq!(Confidence::new(1.5), Confidence::FULL);
        assert_eq!(Confidence::new(-3.0), Confidence::ZERO);
    }

    #[test]
    fn nan_is_zero() {
        assert!(Confidence::new(f64::NAN).is_zero());
    }

    #[test]
    fn a_second_requirement_never_raises_trust() {
        let (a, b) = (Confidence::new(0.8), Confidence::new(0.5));
        assert!(a.and(b) <= a && a.and(b) <= b);
    }

    #[test]
    fn zero_trust_contributes_nothing() {
        assert!(Confidence::ZERO.apply(12.5).abs() < f64::EPSILON);
    }
}
