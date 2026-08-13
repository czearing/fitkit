//! Cited measurements and the validity gate around them.

#![cfg_attr(not(feature = "std"), no_std)]

use fitkit_core::{Answer, Refusal};

/// A published source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Citation {
    /// Stable key, such as `Bulone1997AgaroseWater`.
    pub key: &'static str,
    /// Full reference.
    pub source: &'static str,
}

/// One measurement with its provenance.
///
/// `unknowns` is the load bearing field. A source that did not state its pressure convention has
/// that written down, so a later reader cannot mistake silence for a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record<T> {
    /// Where the value came from.
    pub citation: Citation,
    /// Conditions the source measured under.
    pub conditions: &'static [&'static str],
    /// Quantities the source did not report.
    pub unknowns: &'static [&'static str],
    /// Reported uncertainty, in the units of `value`.
    pub uncertainty: f64,
    /// The measured value.
    pub value: T,
}

impl<T> Record<T> {
    /// Whether `quantity` is listed as unreported.
    pub fn is_unknown(&self, quantity: &str) -> bool {
        self.unknowns.contains(&quantity)
    }
}

/// A law that answers from cited measurement, never from a name.
///
/// Implementors state their domain of validity in [`Law::admits`] and their result in
/// [`Law::derive`]. Callers reach both through [`ask`], which cannot skip the gate.
pub trait Law {
    /// The state being asked about.
    type Input;
    /// What the law returns.
    type Output;

    /// The source this law answers from.
    fn citation(&self) -> Citation;

    /// Whether the input falls inside what the source measured.
    ///
    /// # Errors
    ///
    /// A refusal naming the condition that fell outside the source.
    fn admits(&self, input: &Self::Input) -> Answer<()>;

    /// The result for an admitted input.
    ///
    /// # Errors
    ///
    /// A refusal when a quantity the result depends on was never reported.
    fn derive(&self, input: &Self::Input) -> Answer<Self::Output>;
}

/// Ask a law, gating on [`Law::admits`] first.
///
/// The only supported entry point, so validity cannot be bypassed by calling `derive` directly.
///
/// # Errors
///
/// The refusal from [`Law::admits`], or from [`Law::derive`].
pub fn ask<L: Law>(law: &L, input: &L::Input) -> Answer<L::Output> {
    law.admits(input)?;
    law.derive(input)
}

/// Refuse unless `value` lies within `range`. Non-finite values are always outside.
///
/// # Errors
///
/// [`Refusal::outside_provenance`] carrying `what`.
pub fn within(value: f64, range: core::ops::RangeInclusive<f64>, what: &'static str) -> Answer<()> {
    if value.is_finite() && range.contains(&value) {
        Ok(())
    } else {
        Err(Refusal::outside_provenance(what))
    }
}

#[cfg(test)]
mod tests {
    use fitkit_core::{Answer, Refusal, RefusalKind};

    use super::{ask, within, Citation, Law, Record};

    const SOURCE: Citation = Citation { key: "Example1970", source: "Example, J. Chem. 1970" };

    struct Density;

    impl Law for Density {
        type Input = f64;
        type Output = f64;

        fn citation(&self) -> Citation {
            SOURCE
        }

        fn admits(&self, kelvin: &f64) -> Answer<()> {
            within(*kelvin, 273.15..=373.15, "temperature outside the measured range")
        }

        fn derive(&self, kelvin: &f64) -> Answer<f64> {
            Ok(1000.0 - 0.2 * (kelvin - 273.15))
        }
    }

    #[test]
    fn a_law_answers_inside_its_range() {
        assert!(ask(&Density, &298.15).unwrap() > 0.0);
    }

    #[test]
    fn a_law_refuses_outside_its_range_rather_than_extrapolating() {
        let refusal = ask(&Density, &500.0).unwrap_err();
        assert_eq!(refusal.kind(), RefusalKind::OutsideProvenance);
    }

    #[test]
    fn nan_is_outside_every_range() {
        assert!(ask(&Density, &f64::NAN).is_err());
    }

    #[test]
    fn a_record_states_what_its_source_never_measured() {
        let record = Record {
            citation: SOURCE,
            conditions: &["298.15 K", "101.3 kPa"],
            unknowns: &["pressure convention"],
            uncertainty: 0.4,
            value: 997.05,
        };
        assert!(record.is_unknown("pressure convention"));
        assert!(!record.is_unknown("temperature"));
    }

    #[test]
    fn refusals_are_free_to_construct() {
        const _: Refusal = Refusal::unreported("pH");
    }
}
