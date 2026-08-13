use crate::Confidence;

/// A half-open region `[start, end)` of the problem.
///
/// Unitless. Frames, reaction stages, or one span covering everything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    /// First index, inclusive.
    pub start: usize,
    /// Last index, exclusive.
    pub end: usize,
}

impl Span {
    /// A span over `[start, end)`. An inverted range is empty.
    #[inline]
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end: end.max(start) }
    }

    /// A span over a whole problem of size `len`.
    #[inline]
    pub fn whole(len: usize) -> Self {
        Self::new(0, len)
    }

    /// Indices covered. Zero if the fields were set inverted by hand.
    #[inline]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the span covers nothing.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.end <= self.start
    }

    /// Whether `position` falls inside.
    #[inline]
    pub const fn contains(self, position: usize) -> bool {
        position >= self.start && position < self.end
    }

    /// Whether two spans share an index.
    #[inline]
    pub const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// One measurement, the span it speaks for, and how far it is trusted.
///
/// The only way facts enter a solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evidence<E> {
    /// The region measured.
    pub span: Span,
    /// Trust. Zero means leave the span unchanged.
    pub confidence: Confidence,
    /// The measurement.
    pub value: E,
}

impl<E> Evidence<E> {
    /// A fully trusted measurement.
    pub fn certain(span: Span, value: E) -> Self {
        Self { span, confidence: Confidence::FULL, value }
    }

    /// A measurement with an explicit trust weight.
    pub fn new(span: Span, confidence: Confidence, value: E) -> Self {
        Self { span, confidence, value }
    }

    /// Whether this can support a decision.
    pub fn is_informative(&self) -> bool {
        !self.confidence.is_zero() && !self.span.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{Evidence, Span};
    use crate::Confidence;

    #[test]
    fn an_inverted_span_is_empty() {
        assert!(Span::new(90, 10).is_empty());
        let by_hand = Span { start: 90, end: 10 };
        assert!(by_hand.is_empty());
        assert_eq!(by_hand.len(), 0, "the fields are public, so length cannot underflow");
    }

    #[test]
    fn neighbouring_spans_do_not_overlap() {
        assert!(!Span::new(0, 10).overlaps(Span::new(10, 20)));
        assert!(Span::new(10, 20).contains(10));
    }

    #[test]
    fn zero_confidence_is_never_informative() {
        let evidence = Evidence::new(Span::new(0, 10_000), Confidence::ZERO, 1.0);
        assert!(!evidence.is_informative());
    }
}
