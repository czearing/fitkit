use crate::{Answer, Refusal};

/// A quantity as a source gave it.
///
/// `Unreported` is not zero, not a default, and not a typical value. It is the absence of a
/// measurement, and it has no accessor that can substitute one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reported<T> {
    /// The source gave a value.
    Known(T),
    /// The source was silent.
    Unreported,
}

impl<T> Reported<T> {
    /// Whether the source was silent.
    pub const fn is_unreported(&self) -> bool {
        matches!(self, Self::Unreported)
    }

    /// The value, or a refusal naming what was missing.
    ///
    /// The only way in. There is no `unwrap_or` by design.
    ///
    /// # Errors
    ///
    /// [`Refusal::unreported`] when the source was silent.
    pub fn require(self, what: &'static str) -> Answer<T> {
        match self {
            Self::Known(value) => Ok(value),
            Self::Unreported => Err(Refusal::unreported(what)),
        }
    }

    /// Apply a function to a known value.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Reported<U> {
        match self {
            Self::Known(value) => Reported::Known(f(value)),
            Self::Unreported => Reported::Unreported,
        }
    }
}

impl<T> From<Option<T>> for Reported<T> {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Unreported, Self::Known)
    }
}

#[cfg(test)]
mod tests {
    use super::Reported;

    #[test]
    fn an_unreported_value_refuses_rather_than_defaulting() {
        let ph: Reported<f64> = Reported::Unreported;
        assert_eq!(ph.require("solvent pH").unwrap_err().reason(), "solvent pH");
    }

    #[test]
    fn map_invents_nothing() {
        let ph: Reported<f64> = Reported::Unreported;
        assert!(ph.map(|v| v * 2.0).is_unreported());
    }

    #[test]
    fn a_measured_zero_is_not_unreported() {
        assert!(Reported::Known(0.0).require("cystine").is_ok());
    }
}
