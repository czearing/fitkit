//! Feasible sets, and how much error they survive.
//!
//! The other two layers pick one answer. This one reports the whole region where every stated
//! requirement holds, a point inside it, and the [`Margin`] to the nearest edge where a
//! requirement fails. That margin is the useful number: it is the tolerance to error, which a
//! single optimum never states.
//!
//! ```
//! use fitkit_feasible::{Feasible, Problem, Row, Sense};
//!
//! // Two parts, one to three units each, totalling three to five.
//! let mut problem = Problem::new(2);
//! problem.bound(0, 1.0, 3.0);
//! problem.bound(1, 1.0, 3.0);
//! problem.row(Row::new(vec![1.0, 1.0], Sense::Ge, 3.0, "enough to hold together"));
//! problem.row(Row::new(vec![1.0, 1.0], Sense::Le, 5.0, "light enough to lift"));
//!
//! let Feasible::Region { point, margin } = problem.solve() else { panic!("should hold") };
//! let total = point[0] + point[1];
//! assert!((3.0..=5.0).contains(&total));
//! assert!(margin.survives(0.4), "both parts can be off by this much at once");
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod simplex;

use alloc::vec;
use alloc::vec::Vec;

use fitkit_core::{Answer, Margin, Refusal};
use fitkit_ledger::Citation;

use simplex::{Lp, Outcome, Relation};

/// The relation a row asserts between its linear form and its right-hand side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    /// At most.
    Le,
    /// At least.
    Ge,
    /// Exactly.
    Eq,
}

/// One linear statement about the answer: `coefficients . x  <sense>  rhs`.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// One coefficient per variable.
    pub coefficients: Vec<f64>,
    /// The relation asserted.
    pub sense: Sense,
    /// The right-hand side.
    pub rhs: f64,
    /// Why this holds. Reported when the row is the reason a problem has no answer.
    pub law: &'static str,
}

impl Row {
    /// A row and the law behind it.
    pub fn new(coefficients: Vec<f64>, sense: Sense, rhs: f64, law: &'static str) -> Self {
        Self { coefficients, sense, rhs, law }
    }

    /// How far `point` breaks this row. Zero means satisfied.
    pub fn violation(&self, point: &[f64]) -> f64 {
        let lhs: f64 = self.coefficients.iter().zip(point).map(|(c, x)| c * x).sum();
        match self.sense {
            Sense::Eq => (lhs - self.rhs).abs(),
            Sense::Le => (lhs - self.rhs).max(0.0),
            Sense::Ge => (self.rhs - lhs).max(0.0),
        }
    }
}

/// A statement about how the finished object must behave, and the law that makes it testable.
///
/// State behaviour, never a quantity the solver exists to derive. A requirement that names a ratio
/// or a mass is the place a hardcoded answer hides once it has been removed from the code.
pub trait Requirement {
    /// The published source this rests on.
    fn citation(&self) -> Citation;

    /// The rows this adds, over `vars` variables.
    ///
    /// # Errors
    ///
    /// When the requirement cannot be stated for this problem.
    fn rows(&self, vars: usize) -> Answer<Vec<Row>>;
}

/// Variables, their bounds, and everything required of them.
#[derive(Clone, Debug, Default)]
pub struct Problem {
    lower: Vec<f64>,
    upper: Vec<f64>,
    rows: Vec<Row>,
}

impl Problem {
    /// A problem over `vars` variables, each in `0..=inf`.
    pub fn new(vars: usize) -> Self {
        Self { lower: vec![0.0; vars], upper: vec![f64::INFINITY; vars], rows: Vec::new() }
    }

    /// How many variables.
    pub fn vars(&self) -> usize {
        self.lower.len()
    }

    /// Restrict one variable. Out of range indices are ignored.
    pub fn bound(&mut self, index: usize, low: f64, high: f64) -> &mut Self {
        if let (Some(lower), Some(upper)) = (self.lower.get_mut(index), self.upper.get_mut(index)) {
            *lower = low;
            *upper = high.max(low);
        }
        self
    }

    /// Pin one variable to a value. It stops counting toward the margin, since it cannot move.
    pub fn lock(&mut self, index: usize, at: f64) -> &mut Self {
        self.bound(index, at, at)
    }

    /// Add a row directly. Prefer [`Problem::require`], which forces a citation.
    pub fn row(&mut self, row: Row) -> &mut Self {
        if row.coefficients.len() == self.vars() {
            self.rows.push(row);
        }
        self
    }

    /// Add everything a requirement states.
    ///
    /// # Errors
    ///
    /// When the requirement refuses, states nothing, or states a row of the wrong width.
    pub fn require<R: Requirement>(&mut self, requirement: &R) -> Answer<&mut Self> {
        let stated = requirement.rows(self.vars())?;
        if stated.is_empty() {
            return Err(Refusal::uninformative("a requirement stated nothing"));
        }
        for row in stated {
            if row.coefficients.len() != self.vars() {
                return Err(Refusal::incoherent("a requirement stated a row of the wrong width"));
            }
            self.rows.push(row);
        }
        Ok(self)
    }

    /// The rows stated so far.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Solve for the feasible region.
    ///
    /// Reports the largest error every variable can carry at once while every row still holds. A
    /// variable in an equality row cannot carry any, which is what an equality means: state a
    /// quantity with real tolerance as a pair of inequalities instead.
    pub fn solve(&self) -> Feasible {
        solve(self)
    }
}

/// What a problem admits.
#[derive(Clone, Debug, PartialEq)]
pub enum Feasible {
    /// Requirements hold together. `point` sits as far inside as the box allows.
    Region {
        /// One value per variable.
        point: Vec<f64>,
        /// How far every variable can move at once and still satisfy every row.
        margin: Margin,
    },
    /// Nothing satisfies every requirement at once.
    Empty {
        /// The least infeasible point found.
        point: Vec<f64>,
        /// Total unsatisfied amount there.
        residual: f64,
        /// The laws left broken, which is what makes a refusal explainable.
        binding: Vec<&'static str>,
    },
}

impl Feasible {
    /// Whether the requirements hold together.
    pub fn holds(&self) -> bool {
        matches!(self, Self::Region { .. })
    }

    /// The answer, if there is one.
    pub fn point(&self) -> Option<&[f64]> {
        match self {
            Self::Region { point, .. } => Some(point),
            Self::Empty { .. } => None,
        }
    }

    /// The tolerance to error. [`Margin::NONE`] when nothing holds.
    pub fn margin(&self) -> Margin {
        match self {
            Self::Region { margin, .. } => *margin,
            Self::Empty { .. } => Margin::NONE,
        }
    }
}

fn solve(problem: &Problem) -> Feasible {
    let vars = problem.vars();
    if vars == 0 {
        return Feasible::Region { point: Vec::new(), margin: Margin::NONE };
    }

    // Variables are shifted to x = lower + shifted, so every one starts non-negative, and one more
    // variable carries the half-width of the box being inflated inside the region.
    let radius = vars;
    let mut pinned: Vec<bool> =
        (0..vars).map(|index| problem.lower[index] >= problem.upper[index]).collect();
    for row in problem.rows.iter().filter(|row| row.sense == Sense::Eq) {
        for (index, coefficient) in row.coefficients.iter().enumerate() {
            if coefficient.abs() > 0.0 {
                pinned[index] = true;
            }
        }
    }

    let mut lp = Lp { vars: vars + 1, rows: Vec::new(), objective: vec![0.0; vars + 1] };
    lp.objective[radius] = -1.0;

    for row in &problem.rows {
        let offset: f64 =
            row.coefficients.iter().zip(&problem.lower).map(|(c, low)| c * low).sum::<f64>();
        let rhs = row.rhs - offset;
        // The box grows until it touches this row. Its reach along the row's normal counts only
        // the coefficients that can actually move.
        let reach: f64 = row
            .coefficients
            .iter()
            .enumerate()
            .filter(|(index, _)| !pinned[*index])
            .map(|(_, c)| c.abs())
            .sum();

        let mut coefficients = vec![0.0; vars + 1];
        match row.sense {
            Sense::Le => {
                coefficients[..vars].copy_from_slice(&row.coefficients);
                coefficients[radius] = reach;
                lp.rows.push((coefficients, Relation::Le, rhs));
            }
            Sense::Ge => {
                for (slot, c) in coefficients.iter_mut().zip(&row.coefficients) {
                    *slot = -c;
                }
                coefficients[radius] = reach;
                lp.rows.push((coefficients, Relation::Le, -rhs));
            }
            Sense::Eq => {
                coefficients[..vars].copy_from_slice(&row.coefficients);
                lp.rows.push((coefficients, Relation::Eq, rhs));
            }
        }
    }

    let mut movable = false;
    for index in 0..vars {
        let width = problem.upper[index] - problem.lower[index];
        if problem.lower[index] >= problem.upper[index] {
            let mut locked = vec![0.0; vars + 1];
            locked[index] = 1.0;
            lp.rows.push((locked, Relation::Eq, 0.0));
            continue;
        }

        let mut high = vec![0.0; vars + 1];
        high[index] = 1.0;
        if pinned[index] {
            if width.is_finite() {
                lp.rows.push((high, Relation::Le, width));
            }
            continue;
        }

        movable = true;
        let mut low = vec![0.0; vars + 1];
        low[index] = -1.0;
        low[radius] = 1.0;
        lp.rows.push((low, Relation::Le, 0.0));

        if width.is_finite() {
            high[radius] = 1.0;
            lp.rows.push((high, Relation::Le, width));
        }
    }
    if !movable {
        let mut zero = vec![0.0; vars + 1];
        zero[radius] = 1.0;
        lp.rows.push((zero, Relation::Eq, 0.0));
    }

    let restore = |shifted: &[f64]| -> Vec<f64> {
        (0..vars).map(|index| problem.lower[index] + shifted[index]).collect()
    };

    match simplex::minimise(&lp) {
        Outcome::Optimal { point, value } => {
            Feasible::Region { point: restore(&point), margin: Margin::new(-value) }
        }
        Outcome::Unbounded { point } => {
            Feasible::Region { point: restore(&point), margin: Margin::UNBOUNDED }
        }
        Outcome::Infeasible { point, residual } => {
            let point = restore(&point);
            let binding = problem
                .rows
                .iter()
                .filter(|row| row.violation(&point) > 1e-9)
                .map(|row| row.law)
                .collect();
            Feasible::Empty { point, residual, binding }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use fitkit_core::{Refusal, RefusalKind};
    use fitkit_ledger::Citation;

    use super::{Feasible, Problem, Requirement, Row, Sense};

    #[test]
    fn a_box_of_requirements_reports_its_half_width() {
        let mut problem = Problem::new(1);
        problem.bound(0, 0.0, 10.0);
        problem.row(Row::new(vec![1.0], Sense::Ge, 2.0, "lower"));
        problem.row(Row::new(vec![1.0], Sense::Le, 8.0, "upper"));

        let Feasible::Region { point, margin } = problem.solve() else { panic!("2..8 holds") };
        assert!((point[0] - 5.0).abs() < 1e-6, "the centre is furthest from both edges");
        assert!((margin.get() - 3.0).abs() < 1e-6, "three either way");
    }

    #[test]
    fn a_tighter_requirement_shrinks_the_margin() {
        let mut wide = Problem::new(1);
        wide.bound(0, 0.0, 10.0);
        wide.row(Row::new(vec![1.0], Sense::Le, 8.0, "upper"));

        let mut narrow = wide.clone();
        narrow.row(Row::new(vec![1.0], Sense::Le, 4.0, "tighter upper"));

        assert!(narrow.solve().margin() < wide.solve().margin());
    }

    #[test]
    fn contradictory_requirements_name_the_laws_that_broke() {
        let mut problem = Problem::new(1);
        problem.bound(0, 0.0, 10.0);
        problem.row(Row::new(vec![1.0], Sense::Ge, 9.0, "must be strong"));
        problem.row(Row::new(vec![1.0], Sense::Le, 2.0, "must be light"));

        let Feasible::Empty { binding, residual, .. } = problem.solve() else {
            panic!("9 and 2 cannot both hold")
        };
        assert!(residual > 0.0);
        assert!(binding.contains(&"must be strong") || binding.contains(&"must be light"));
    }

    #[test]
    fn a_locked_variable_cannot_widen_the_margin() {
        let mut problem = Problem::new(2);
        problem.bound(0, 0.0, 10.0);
        problem.lock(1, 4.0);
        problem.row(Row::new(vec![1.0, 1.0], Sense::Le, 6.0, "total"));

        let Feasible::Region { point, margin } = problem.solve() else { panic!("holds") };
        assert!((point[1] - 4.0).abs() < 1e-9, "a locked variable stays put");
        assert!((margin.get() - 1.0).abs() < 1e-6, "only the free variable has room");
    }

    #[test]
    fn an_answer_on_the_edge_has_no_margin() {
        let mut problem = Problem::new(1);
        problem.bound(0, 0.0, 10.0);
        problem.row(Row::new(vec![1.0], Sense::Eq, 3.0, "exactly three"));

        let solved = problem.solve();
        assert!(solved.holds());
        assert!(solved.margin().is_none(), "an equality leaves nowhere to move");
    }

    #[test]
    fn an_unbounded_region_says_so_rather_than_claiming_safety() {
        let mut problem = Problem::new(1);
        problem.row(Row::new(vec![1.0], Sense::Ge, 1.0, "at least one"));
        assert!(problem.solve().margin().is_unbounded());
    }

    #[test]
    fn a_requirement_carries_its_citation_into_the_problem() {
        struct HoldsTogether;
        impl Requirement for HoldsTogether {
            fn citation(&self) -> Citation {
                Citation { key: "Example2020", source: "Example, J. Test 1, 1 (2020)" }
            }
            fn rows(&self, vars: usize) -> Result<vec::Vec<Row>, Refusal> {
                if vars < 2 {
                    return Err(Refusal::incoherent("needs a binder and a solid"));
                }
                Ok(vec![Row::new(vec![1.0, -1.0], Sense::Ge, 0.0, "Example2020 binder ratio")])
            }
        }

        let mut problem = Problem::new(2);
        problem.bound(0, 0.0, 5.0);
        problem.bound(1, 0.0, 5.0);
        problem.require(&HoldsTogether).expect("two variables is enough");
        assert_eq!(problem.rows().len(), 1);
        assert!(problem.solve().holds());

        assert!(Problem::new(1).require(&HoldsTogether).is_err(), "a refusal is an answer");
    }

    #[test]
    fn a_requirement_that_states_nothing_is_refused() {
        struct SaysNothing;
        impl Requirement for SaysNothing {
            fn citation(&self) -> Citation {
                Citation { key: "Example2020", source: "Example, J. Test 1, 1 (2020)" }
            }
            fn rows(&self, _: usize) -> Result<vec::Vec<Row>, Refusal> {
                Ok(vec::Vec::new())
            }
        }

        let refusal = Problem::new(2).require(&SaysNothing).expect_err("nothing is not a bound");
        assert_eq!(refusal.kind(), RefusalKind::Uninformative);
    }

    #[test]
    fn a_wrong_width_row_is_ignored_rather_than_silently_padded() {
        let mut problem = Problem::new(2);
        problem.row(Row::new(vec![1.0], Sense::Le, 1.0, "too narrow"));
        assert!(problem.rows().is_empty());
    }

    #[test]
    fn the_reported_margin_is_the_error_the_answer_actually_survives() {
        let mut problem = Problem::new(3);
        for index in 0..3 {
            problem.bound(index, 0.0, 6.0);
        }
        problem.row(Row::new(vec![1.0, 1.0, 1.0], Sense::Le, 9.0, "total"));
        problem.row(Row::new(vec![1.0, -1.0, 0.0], Sense::Le, 2.0, "balance"));
        problem.row(Row::new(vec![0.0, 1.0, 1.0], Sense::Ge, 3.0, "enough of the rest"));

        let Feasible::Region { point, margin } = problem.solve() else { panic!("holds") };
        let radius = margin.get();
        assert!(radius > 0.0);

        // Every corner of the box the margin claims must satisfy every row, and stepping past it
        // in the worst direction must break one. Otherwise the number is decoration.
        for corner in 0..8_u32 {
            for (scale, expect_all_hold) in [(0.99, true), (1.01, false)] {
                let moved: vec::Vec<f64> = point
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let sign = if corner >> index & 1 == 1 { 1.0 } else { -1.0 };
                        value + sign * radius * scale
                    })
                    .collect();
                let holds = problem.rows().iter().all(|row| row.violation(&moved) <= 1e-9)
                    && moved.iter().all(|value| (-1e-9..=6.0 + 1e-9).contains(value));
                if expect_all_hold {
                    assert!(holds, "corner {corner} inside the margin broke a row");
                } else if !holds {
                    return;
                }
            }
        }
        panic!("no corner outside the margin broke anything, so the margin understates the region");
    }
}
