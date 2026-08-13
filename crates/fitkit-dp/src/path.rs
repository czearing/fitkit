use alloc::vec;
use alloc::vec::Vec;

/// A decoded path and the cost paid for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// One candidate index per step.
    pub path: Vec<usize>,
    /// Total emission plus weighted transition cost.
    pub cost: f64,
}

/// Decode the lowest-cost sequence of candidates.
///
/// `emission(step, state)` is the cost of explaining that step with that candidate.
/// `transition(from, to)` is the cost of changing between neighbouring steps, scaled by
/// `transition_weight`. Zero weight follows the evidence exactly; a large weight ignores outliers.
///
/// An empty problem returns an empty path. Non-finite costs are treated as maximally expensive.
///
/// Runs in `O(steps * states^2)` time. Each cost closure is called once per distinct argument.
pub fn decode_path<E, T>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
) -> Vec<usize>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    decode_path_with_cost(steps, states, transition_weight, emission, transition).path
}

/// As [`decode_path`], also reporting total cost, which makes two models comparable.
/// Runs in `O(steps * states^2)` time. Each cost closure is called once per distinct argument.
///
/// # Panics
///
/// If `states` exceeds `u32::MAX`, the width of the backpointer table.
#[allow(clippy::cast_possible_truncation)] // states is asserted to fit u32 below
pub fn decode_path_with_cost<E, T>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
) -> Decoded
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    if steps == 0 || states == 0 {
        return Decoded { path: Vec::new(), cost: 0.0 };
    }
    assert!(u32::try_from(states).is_ok(), "{states} states exceeds the backpointer width");

    let weight = finite(transition_weight, 0.0);
    let mut jump = vec![0.0; states * states];
    if weight.abs() > 0.0 && states > 1 {
        for from in 0..states {
            for to in 0..states {
                jump[from * states + to] = weight * finite(transition(from, to), f64::MAX);
            }
        }
    }

    let mut cost: Vec<f64> = (0..states).map(|s| finite(emission(0, s), f64::MAX)).collect();
    let mut next = vec![0.0; states];
    let mut back = vec![0_u32; steps * states];

    for step in 1..steps {
        let row = step * states;
        for to in 0..states {
            let mut best = f64::INFINITY;
            let mut best_from = 0;
            for (from, &previous) in cost.iter().enumerate() {
                let total = previous + jump[from * states + to];
                if total < best {
                    best = total;
                    best_from = from;
                }
            }
            next[to] = best + finite(emission(step, to), f64::MAX);
            back[row + to] = best_from as u32;
        }
        cost.copy_from_slice(&next);
    }

    let mut end = 0;
    let mut total = f64::INFINITY;
    for (state, &value) in cost.iter().enumerate() {
        if value < total {
            total = value;
            end = state;
        }
    }

    let mut path = vec![0; steps];
    path[steps - 1] = end;
    for step in (1..steps).rev() {
        end = back[step * states + end] as usize;
        path[step - 1] = end;
    }

    Decoded { path, cost: total }
}

/// How much worse the best alternative is, at each step.
///
/// The answer to "how far can this measurement be wrong before the decode changes". A step with a
/// margin of zero has a tied runner up and is decided by nothing. Infinite means no alternative
/// exists, which is a missing candidate rather than a safe result.
///
/// Costs a forward and a backward pass, so it is twice the work of [`decode_path`] and holds
/// `O(steps * states)` intermediates. Call it when reporting, not in the inner loop.
pub fn decode_margins<E, T>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
) -> Vec<f64>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    if steps == 0 || states == 0 {
        return Vec::new();
    }

    let weight = finite(transition_weight, 0.0);
    let mut jump = vec![0.0; states * states];
    if weight.abs() > 0.0 && states > 1 {
        for from in 0..states {
            for to in 0..states {
                jump[from * states + to] = weight * finite(transition(from, to), f64::MAX);
            }
        }
    }

    let emit: Vec<f64> =
        (0..steps * states).map(|i| finite(emission(i / states, i % states), f64::MAX)).collect();

    let mut forward = emit.clone();
    for step in 1..steps {
        for to in 0..states {
            let mut best = f64::INFINITY;
            for from in 0..states {
                best = best.min(forward[(step - 1) * states + from] + jump[from * states + to]);
            }
            forward[step * states + to] += best;
        }
    }

    let mut backward = emit.clone();
    for step in (0..steps - 1).rev() {
        for from in 0..states {
            let mut best = f64::INFINITY;
            for to in 0..states {
                best = best.min(jump[from * states + to] + backward[(step + 1) * states + to]);
            }
            backward[step * states + from] += best;
        }
    }

    (0..steps)
        .map(|step| {
            let (mut best, mut runner_up) = (f64::INFINITY, f64::INFINITY);
            for state in 0..states {
                let at = step * states + state;
                let total = forward[at] + backward[at] - emit[at];
                if total < best {
                    runner_up = best;
                    best = total;
                } else if total < runner_up {
                    runner_up = total;
                }
            }
            if !best.is_finite() {
                0.0
            } else if runner_up.is_finite() {
                runner_up - best
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

#[inline]
fn finite(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{decode_margins, decode_path, decode_path_with_cost};

    #[test]
    fn with_no_resistance_the_path_follows_the_evidence() {
        let observed = [1, 0, 1];
        let path = decode_path(
            3,
            2,
            0.0,
            |step, state| f64::from(u8::from(observed[step] != state)),
            |_, _| 1.0,
        );
        assert_eq!(path, observed);
    }

    #[test]
    fn resistance_absorbs_a_single_outlier() {
        let observed = [1, 1, 0, 1, 1];
        let path = decode_path(
            5,
            2,
            50.0,
            |step, state| f64::from(u8::from(observed[step] != state)),
            |from, to| f64::from(u8::from(from != to)),
        );
        assert_eq!(path, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn an_empty_problem_decodes_to_an_empty_path() {
        assert!(decode_path(0, 4, 1.0, |_, _| 0.0, |_, _| 0.0).is_empty());
        assert!(decode_path(4, 0, 1.0, |_, _| 0.0, |_, _| 0.0).is_empty());
    }

    #[test]
    fn a_nan_cost_never_wins() {
        let path =
            decode_path(2, 2, 0.0, |_, state| if state == 0 { f64::NAN } else { 1.0 }, |_, _| 0.0);
        assert_eq!(path, [1, 1]);
    }

    #[test]
    fn the_reported_cost_matches_the_reported_path() {
        let emissions = [[0.5, 2.0], [3.0, 0.25], [1.0, 1.5]];
        let weight = 2.0;
        let decoded = decode_path_with_cost(
            3,
            2,
            weight,
            |step, state| emissions[step][state],
            |from, to| f64::from(u8::from(from != to)),
        );
        let mut recomputed = emissions[0][decoded.path[0]];
        for step in 1..3 {
            recomputed += emissions[step][decoded.path[step]];
            if decoded.path[step - 1] != decoded.path[step] {
                recomputed += weight;
            }
        }
        assert!((decoded.cost - recomputed).abs() < 1e-12);
    }

    #[test]
    fn the_decoder_matches_brute_force() {
        let (steps, states, weight) = (6, 4, 0.75);
        let emission = |step: usize, state: usize| ((step * 7 + state * 13) % 11) as f64 / 3.0;
        let transition = |from: usize, to: usize| (from as f64 - to as f64).abs();

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition);

        let mut best = f64::INFINITY;
        for encoded in 0..states.pow(u32::try_from(steps).unwrap()) {
            let mut rest = encoded;
            let mut path = vec![0; steps];
            for slot in &mut path {
                *slot = rest % states;
                rest /= states;
            }
            let mut cost = emission(0, path[0]);
            for step in 1..steps {
                cost +=
                    emission(step, path[step]) + weight * transition(path[step - 1], path[step]);
            }
            best = best.min(cost);
        }
        assert!((decoded.cost - best).abs() < 1e-9);
    }

    #[test]
    fn a_margin_is_the_cost_of_the_best_alternative_at_that_step() {
        let (steps, states, weight) = (5, 3, 0.5);
        let emission = |step: usize, state: usize| ((step * 5 + state * 7) % 9) as f64 / 2.0;
        let transition = |from: usize, to: usize| (from as f64 - to as f64).abs();

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition);
        let margins = decode_margins(steps, states, weight, emission, transition);

        let total = |path: &[usize]| {
            let mut cost = emission(0, path[0]);
            for step in 1..steps {
                cost +=
                    emission(step, path[step]) + weight * transition(path[step - 1], path[step]);
            }
            cost
        };

        for (step, &margin) in margins.iter().enumerate() {
            let mut best_alternative = f64::INFINITY;
            for encoded in 0..states.pow(u32::try_from(steps).unwrap()) {
                let mut rest = encoded;
                let mut path = vec![0; steps];
                for slot in &mut path {
                    *slot = rest % states;
                    rest /= states;
                }
                if path[step] != decoded.path[step] {
                    best_alternative = best_alternative.min(total(&path));
                }
            }
            assert!(
                (margin - (best_alternative - decoded.cost)).abs() < 1e-9,
                "step {step} reported margin {margin}"
            );
        }
    }

    #[test]
    fn a_tied_step_has_no_margin() {
        let margins = decode_margins(1, 2, 0.0, |_, _| 1.0, |_, _| 0.0);
        assert!(margins[0].abs() < 1e-12, "a coin flip is decided by nothing");
    }

    #[test]
    fn a_single_candidate_leaves_the_margin_unbounded() {
        assert!(decode_margins(3, 1, 0.0, |_, _| 1.0, |_, _| 0.0)[0].is_infinite());
        assert!(decode_margins(0, 4, 0.0, |_, _| 1.0, |_, _| 0.0).is_empty());
    }
}
