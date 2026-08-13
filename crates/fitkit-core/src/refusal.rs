use core::fmt;

/// Why a question could not be answered from the evidence available.
///
/// A refusal is a correct outcome. The alternative is a number derived from a typical value, a
/// category average, or the name of the thing being asked about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Refusal {
    reason: &'static str,
    kind: RefusalKind,
}

/// The class of a refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefusalKind {
    /// Outside the conditions the source measured.
    OutsideProvenance,
    /// A required quantity was never reported.
    Unreported,
    /// The input is not a coherent state.
    Incoherent,
    /// Evidence exists but carries no information.
    Uninformative,
}

impl Refusal {
    /// Outside the conditions the source measured.
    pub const fn outside_provenance(reason: &'static str) -> Self {
        Self { reason, kind: RefusalKind::OutsideProvenance }
    }

    /// A required quantity was never reported.
    pub const fn unreported(reason: &'static str) -> Self {
        Self { reason, kind: RefusalKind::Unreported }
    }

    /// The input is not a coherent state.
    pub const fn incoherent(reason: &'static str) -> Self {
        Self { reason, kind: RefusalKind::Incoherent }
    }

    /// Evidence carries no information.
    pub const fn uninformative(reason: &'static str) -> Self {
        Self { reason, kind: RefusalKind::Uninformative }
    }

    /// The reason.
    pub const fn reason(self) -> &'static str {
        self.reason
    }

    /// The class.
    pub const fn kind(self) -> RefusalKind {
        self.kind
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            RefusalKind::OutsideProvenance => "outside provenance",
            RefusalKind::Unreported => "unreported",
            RefusalKind::Incoherent => "incoherent input",
            RefusalKind::Uninformative => "uninformative evidence",
        };
        write!(f, "{kind}: {}", self.reason)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Refusal {}

/// The result of asking a law or solver. Either the evidence reaches, or it refuses with a reason.
pub type Answer<T> = Result<T, Refusal>;

#[cfg(test)]
mod tests {
    use super::{Refusal, RefusalKind};

    #[test]
    fn a_refusal_carries_its_class() {
        let refusal = Refusal::outside_provenance("373 K is above the measured range");
        assert_eq!(refusal.kind(), RefusalKind::OutsideProvenance);
    }

    #[test]
    fn never_measured_is_distinct_from_measured_elsewhere() {
        assert_ne!(
            Refusal::unreported("pH").kind(),
            Refusal::outside_provenance("pH 9 is outside 3..7").kind()
        );
    }
}
