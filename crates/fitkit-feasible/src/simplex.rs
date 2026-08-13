//! A dense two-phase simplex over `Ax {<=,=,>=} b`, `x >= 0`.
//!
//! Problems here are small, tens of variables and rows, so a dense tableau beats any solver that
//! has to be linked, installed, or spawned. Bland's rule is used rather than steepest descent: it
//! pivots more but is the only common rule with a termination proof, and locked variables make
//! these problems degenerate by construction.

use alloc::vec;
use alloc::vec::Vec;

const TOL: f64 = 1e-9;

/// The relation a row asserts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Relation {
    Le,
    Ge,
    Eq,
}

/// What the solve found.
pub(crate) enum Outcome {
    /// The minimum, and where it is.
    Optimal { point: Vec<f64>, value: f64 },
    /// The objective falls forever. The point is still feasible.
    Unbounded { point: Vec<f64> },
    /// No point satisfies every row. The point is the least infeasible one found.
    Infeasible { point: Vec<f64>, residual: f64 },
}

pub(crate) struct Lp {
    pub(crate) vars: usize,
    pub(crate) rows: Vec<(Vec<f64>, Relation, f64)>,
    /// Minimised. Padded with zeros to the variable count.
    pub(crate) objective: Vec<f64>,
}

struct Tableau {
    cells: Vec<f64>,
    rows: usize,
    cols: usize,
    basis: Vec<usize>,
    artificial: Vec<bool>,
}

impl Tableau {
    #[inline]
    fn at(&self, row: usize, col: usize) -> f64 {
        self.cells[row * (self.cols + 1) + col]
    }

    fn pivot(&mut self, row: usize, col: usize) {
        let width = self.cols + 1;
        let divisor = self.at(row, col);
        for cell in &mut self.cells[row * width..row * width + width] {
            *cell /= divisor;
        }
        for other in 0..self.rows {
            if other == row {
                continue;
            }
            let factor = self.at(other, col);
            if factor.abs() < TOL {
                continue;
            }
            for offset in 0..width {
                let scaled = self.cells[row * width + offset] * factor;
                self.cells[other * width + offset] -= scaled;
            }
        }
        self.basis[row] = col;
    }

    /// Minimise `cost` over the allowed columns. `None` when the objective is unbounded.
    fn run(&mut self, cost: &[f64], allow_artificial: bool) -> Option<f64> {
        // Bland's rule bounds the iteration count, so the cap only catches a numerical stall.
        for _ in 0..(self.rows + 1) * (self.cols + 1) * 4 {
            let mut entering = None;
            for col in 0..self.cols {
                if (!allow_artificial && self.artificial[col]) || self.basis.contains(&col) {
                    continue;
                }
                let reduced: f64 =
                    (0..self.rows).map(|row| cost[self.basis[row]] * self.at(row, col)).sum();
                if reduced - cost[col] > TOL {
                    entering = Some(col);
                    break;
                }
            }
            let Some(col) = entering else { break };

            let mut leaving = None;
            let mut best = f64::INFINITY;
            for row in 0..self.rows {
                let coefficient = self.at(row, col);
                if coefficient <= TOL {
                    continue;
                }
                let ratio = self.at(row, self.cols) / coefficient;
                let better = leaving.map_or(true, |chosen: usize| {
                    ratio < best - TOL
                        || (ratio < best + TOL && self.basis[row] < self.basis[chosen])
                });
                if better {
                    best = ratio;
                    leaving = Some(row);
                }
            }
            let row = leaving?;
            self.pivot(row, col);
        }
        Some((0..self.rows).map(|row| cost[self.basis[row]] * self.at(row, self.cols)).sum())
    }

    fn point(&self, vars: usize) -> Vec<f64> {
        let mut point = vec![0.0; vars];
        for row in 0..self.rows {
            if let Some(slot) = point.get_mut(self.basis[row]) {
                *slot = self.at(row, self.cols);
            }
        }
        point
    }
}

fn build(lp: &Lp) -> Tableau {
    // A negative right-hand side would start the tableau outside the feasible cone, so the row is
    // negated into place first, which flips the direction it asserts and can change how many
    // columns it needs. Normalising before counting is what keeps the two in step.
    let normalised: Vec<(Vec<f64>, Relation, f64)> = lp
        .rows
        .iter()
        .map(|(coefficients, relation, rhs)| {
            if *rhs >= 0.0 {
                return (coefficients.clone(), *relation, *rhs);
            }
            let flipped = match relation {
                Relation::Le => Relation::Ge,
                Relation::Ge => Relation::Le,
                Relation::Eq => Relation::Eq,
            };
            (coefficients.iter().map(|c| -c).collect(), flipped, -rhs)
        })
        .collect();

    let extras: usize =
        normalised.iter().map(|(_, relation, _)| usize::from(*relation == Relation::Ge) + 1).sum();
    let cols = lp.vars + extras;
    let width = cols + 1;
    let mut tableau = Tableau {
        cells: vec![0.0; normalised.len() * width],
        rows: normalised.len(),
        cols,
        basis: vec![0; normalised.len()],
        artificial: vec![false; cols],
    };

    let mut next = lp.vars;
    for (index, (coefficients, relation, rhs)) in normalised.iter().enumerate() {
        let row = index * width;
        for (column, coefficient) in coefficients.iter().enumerate() {
            tableau.cells[row + column] = *coefficient;
        }
        tableau.cells[row + cols] = *rhs;

        match relation {
            Relation::Le => {
                tableau.cells[row + next] = 1.0;
            }
            Relation::Ge => {
                tableau.cells[row + next] = -1.0;
                next += 1;
                tableau.cells[row + next] = 1.0;
                tableau.artificial[next] = true;
            }
            Relation::Eq => {
                tableau.cells[row + next] = 1.0;
                tableau.artificial[next] = true;
            }
        }
        tableau.basis[index] = next;
        next += 1;
    }
    tableau
}

/// Minimise `lp.objective` subject to every row.
pub(crate) fn minimise(lp: &Lp) -> Outcome {
    if lp.rows.is_empty() {
        return Outcome::Unbounded { point: vec![0.0; lp.vars] };
    }
    let mut tableau = build(lp);

    let phase_one: Vec<f64> = tableau
        .artificial
        .iter()
        .map(|&is_artificial| f64::from(u8::from(is_artificial)))
        .collect();
    let residual = tableau.run(&phase_one, true).unwrap_or(0.0);
    if residual > TOL {
        return Outcome::Infeasible { point: tableau.point(lp.vars), residual };
    }

    // An artificial left in the basis at zero marks a redundant row. Pivot it out where the row
    // allows, so phase two cannot reintroduce it, and leave it otherwise.
    for row in 0..tableau.rows {
        if !tableau.artificial[tableau.basis[row]] {
            continue;
        }
        let replacement = (0..tableau.cols)
            .find(|&col| !tableau.artificial[col] && tableau.at(row, col).abs() > TOL);
        if let Some(col) = replacement {
            tableau.pivot(row, col);
        }
    }

    let mut cost = vec![0.0; tableau.cols];
    cost[..lp.objective.len()].copy_from_slice(&lp.objective);
    match tableau.run(&cost, false) {
        Some(value) => Outcome::Optimal { point: tableau.point(lp.vars), value },
        None => Outcome::Unbounded { point: tableau.point(lp.vars) },
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{minimise, Lp, Outcome, Relation};

    fn value(lp: &Lp) -> Option<f64> {
        match minimise(lp) {
            Outcome::Optimal { value, .. } => Some(value),
            _ => None,
        }
    }

    #[test]
    fn a_bounded_minimum_is_found() {
        // minimise -x - y subject to x + y <= 4, x <= 3.
        let lp = Lp {
            vars: 2,
            rows: vec![(vec![1.0, 1.0], Relation::Le, 4.0), (vec![1.0, 0.0], Relation::Le, 3.0)],
            objective: vec![-1.0, -1.0],
        };
        assert!((value(&lp).unwrap() + 4.0).abs() < 1e-9);
    }

    #[test]
    fn contradictory_rows_are_infeasible() {
        let lp = Lp {
            vars: 1,
            rows: vec![(vec![1.0], Relation::Ge, 5.0), (vec![1.0], Relation::Le, 2.0)],
            objective: vec![0.0],
        };
        assert!(matches!(minimise(&lp), Outcome::Infeasible { .. }));
    }

    #[test]
    fn an_open_direction_is_unbounded() {
        let lp = Lp { vars: 1, rows: vec![(vec![1.0], Relation::Ge, 1.0)], objective: vec![-1.0] };
        assert!(matches!(minimise(&lp), Outcome::Unbounded { .. }));
    }

    #[test]
    fn an_equality_pins_the_answer() {
        let lp = Lp {
            vars: 2,
            rows: vec![(vec![1.0, 1.0], Relation::Eq, 2.0), (vec![1.0, -1.0], Relation::Eq, 0.0)],
            objective: vec![1.0, 0.0],
        };
        let Outcome::Optimal { point, .. } = minimise(&lp) else { panic!("expected a solution") };
        assert!((point[0] - 1.0).abs() < 1e-9 && (point[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_negative_right_hand_side_is_handled() {
        // -x <= -2 is x >= 2.
        let lp = Lp { vars: 1, rows: vec![(vec![-1.0], Relation::Le, -2.0)], objective: vec![1.0] };
        assert!((value(&lp).unwrap() - 2.0).abs() < 1e-9);
    }
}
