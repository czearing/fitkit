use alloc::vec::Vec;
use core::fmt;

use crate::{Confidence, Span};

/// One decision and the span it governs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control<P> {
    /// The region governed.
    pub span: Span,
    /// The recovered parameters.
    pub params: P,
    /// Trust in the evidence behind the span.
    pub confidence: Confidence,
}

impl<P> Control<P> {
    /// Whether acting on this control is a no-op.
    pub fn is_silent(&self) -> bool {
        self.confidence.is_zero() || self.span.is_empty()
    }
}

/// The decision recovered from a reference: one [`Control`] per span.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan<P> {
    /// Controls in ascending span order.
    pub controls: Vec<Control<P>>,
}

impl<P> Default for Plan<P> {
    fn default() -> Self {
        Self { controls: Vec::new() }
    }
}

impl<P> Plan<P> {
    /// The plan that changes nothing.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Whether acting on this plan is a no-op.
    pub fn is_identity(&self) -> bool {
        self.controls.iter().all(Control::is_silent)
    }

    /// Number of controls.
    pub fn len(&self) -> usize {
        self.controls.len()
    }

    /// Whether there are no controls.
    pub fn is_empty(&self) -> bool {
        self.controls.is_empty()
    }

    /// The control governing `position`.
    pub fn at(&self, position: usize) -> Option<&Control<P>> {
        self.controls.iter().find(|c| c.span.contains(position))
    }

    /// Mean confidence weighted by span length.
    pub fn confidence(&self) -> Confidence {
        let mut total = 0.0;
        let mut weight = 0.0;
        for control in &self.controls {
            let len = control.span.len() as f64;
            total += control.confidence.get() * len;
            weight += len;
        }
        if weight <= 0.0 {
            return Confidence::ZERO;
        }
        Confidence::new(total / weight)
    }
}

impl<P> fmt::Display for Plan<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "identity ({} spans)", self.controls.len());
        }
        write!(f, "{} spans at confidence {}", self.controls.len(), self.confidence())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{Control, Plan};
    use crate::{Confidence, Span};

    #[test]
    fn an_empty_plan_is_identity() {
        let plan: Plan<f64> = Plan::identity();
        assert!(plan.is_identity());
        assert_eq!(plan.confidence(), Confidence::ZERO);
    }

    #[test]
    fn zero_confidence_controls_are_identity_whatever_was_decoded() {
        let plan = Plan {
            controls: vec![Control {
                span: Span::new(0, 10),
                params: 3.0,
                confidence: Confidence::ZERO,
            }],
        };
        assert!(plan.is_identity());
    }

    #[test]
    fn confidence_is_weighted_by_length_not_count() {
        let plan = Plan {
            controls: vec![
                Control { span: Span::new(0, 90), params: 1.0, confidence: Confidence::FULL },
                Control { span: Span::new(90, 100), params: 2.0, confidence: Confidence::ZERO },
            ],
        };
        assert!((plan.confidence().get() - 0.9).abs() < 1e-12);
    }
}
