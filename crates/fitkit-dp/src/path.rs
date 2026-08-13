use alloc::vec;
use alloc::vec::Vec;

/// A decoded path and the cost paid for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// One candidate index per step.
    pub path: Vec<usize>,
    /// Total emission plus weighted transition cost, infinite if any step was impossible.
    pub cost: f64,
}

/// Decode the lowest-cost sequence of candidates.
///
/// `emission(step, state)` is the cost of explaining that step with that candidate.
/// `transition(from, to)` is the cost of changing between neighbouring steps, scaled by
/// `transition_weight`. Zero weight follows the evidence exactly; a large weight ignores outliers.
///
/// A cost that is not finite means impossible, negative infinity included, so no cost can be
/// better than free. Impossible steps are counted rather than summed, so a passage nothing
/// explains does not erase the decisions around it. Paths are ordered by that count first and by
/// cost second. An empty problem returns an empty path.
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

    let jump = jump_table(states, transition_weight, transition);
    let decoded = trellis::<false, _>(steps, states, &jump, &emission);
    if decoded.cost.is_finite() {
        return decoded;
    }
    Decoded { path: trellis::<true, _>(steps, states, &jump, &emission).path, cost: f64::INFINITY }
}

/// One pass of the trellis.
///
/// With `COUNT` off this is plain addition, which is exact whenever any path is payable, since an
/// impossible cell costs infinity and loses every comparison. With it on, impossible steps are
/// counted rather than summed and ordered ahead of cost, which is what ranks paths when every one
/// of them is impossible. The second form is a third of the speed, so it runs only in that case.
#[allow(clippy::cast_possible_truncation)] // states fits u32, asserted by the caller
fn trellis<const COUNT: bool, E: Fn(usize, usize) -> f64>(
    steps: usize,
    states: usize,
    jump: &[f64],
    emission: &E,
) -> Decoded {
    let mut breaks = vec![0_u32; states];
    let mut cost = vec![0.0; states];
    for state in 0..states {
        (breaks[state], cost[state]) = pay::<COUNT>(0, 0.0, payable(emission(0, state)));
    }
    let (mut next_breaks, mut next_cost) = (breaks.clone(), cost.clone());
    let mut back = vec![0_u32; steps * states];

    for step in 1..steps {
        for to in 0..states {
            let mut best_breaks = if COUNT { u32::MAX } else { 0 };
            let (mut best_cost, mut best_from) = (f64::INFINITY, 0);
            for (from, &previous) in cost.iter().enumerate() {
                let carried = if COUNT { breaks[from] } else { 0 };
                let (b, c) = pay::<COUNT>(carried, previous, jump[from * states + to]);
                if b < best_breaks || (b == best_breaks && c < best_cost) {
                    (best_breaks, best_cost, best_from) = (b, c, from);
                }
            }
            let emitted = payable(emission(step, to));
            (best_breaks, best_cost) = pay::<COUNT>(best_breaks, best_cost, emitted);
            next_cost[to] = best_cost;
            if COUNT {
                next_breaks[to] = best_breaks;
            }
            back[step * states + to] = best_from as u32;
        }
        cost.copy_from_slice(&next_cost);
        if COUNT {
            breaks.copy_from_slice(&next_breaks);
        }
    }

    let mut total_breaks = if COUNT { u32::MAX } else { 0 };
    let (mut end, mut total) = (0, f64::INFINITY);
    for state in 0..states {
        let b = if COUNT { breaks[state] } else { 0 };
        if b < total_breaks || (b == total_breaks && cost[state] < total) {
            (total_breaks, total, end) = (b, cost[state], state);
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
/// exists at the same level of possibility, which is a missing candidate rather than a safe result.
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

    let jump = jump_table(states, transition_weight, transition);
    let mut emit_breaks = vec![0_u32; steps * states];
    let mut emit_cost = vec![0.0; steps * states];
    for at in 0..steps * states {
        let emitted = payable(emission(at / states, at % states));
        (emit_breaks[at], emit_cost[at]) = pay::<true>(0, 0.0, emitted);
    }

    let (mut forward_breaks, mut forward_cost) = (emit_breaks.clone(), emit_cost.clone());
    for step in 1..steps {
        for to in 0..states {
            let (mut best_breaks, mut best_cost) = (u32::MAX, f64::INFINITY);
            for from in 0..states {
                let at = (step - 1) * states + from;
                let (b, c) =
                    pay::<true>(forward_breaks[at], forward_cost[at], jump[from * states + to]);
                if b < best_breaks || (b == best_breaks && c < best_cost) {
                    (best_breaks, best_cost) = (b, c);
                }
            }
            forward_breaks[step * states + to] += best_breaks;
            forward_cost[step * states + to] += best_cost;
        }
    }

    let (mut back_breaks, mut back_cost) = (emit_breaks.clone(), emit_cost.clone());
    for step in (0..steps - 1).rev() {
        for from in 0..states {
            let (mut best_breaks, mut best_cost) = (u32::MAX, f64::INFINITY);
            for to in 0..states {
                let at = (step + 1) * states + to;
                let (b, c) = pay::<true>(back_breaks[at], back_cost[at], jump[from * states + to]);
                if b < best_breaks || (b == best_breaks && c < best_cost) {
                    (best_breaks, best_cost) = (b, c);
                }
            }
            back_breaks[step * states + from] += best_breaks;
            back_cost[step * states + from] += best_cost;
        }
    }

    (0..steps)
        .map(|step| {
            let (mut best, mut runner_up) = ((u32::MAX, f64::INFINITY), (u32::MAX, f64::INFINITY));
            for state in 0..states {
                let at = step * states + state;
                let total = (
                    forward_breaks[at] + back_breaks[at] - emit_breaks[at],
                    forward_cost[at] + back_cost[at] - emit_cost[at],
                );
                if total < best {
                    runner_up = best;
                    best = total;
                } else if total < runner_up {
                    runner_up = total;
                }
            }
            if runner_up.0 != best.0 || !runner_up.1.is_finite() {
                f64::INFINITY
            } else {
                runner_up.1 - best.1
            }
        })
        .collect()
}

/// Normalise a cost, so that anything not finite means impossible rather than cheap.
///
/// Applied where costs enter, not where they are added, so the decode stays a plain sum.
#[inline]
fn payable(cost: f64) -> f64 {
    if cost.is_finite() {
        cost
    } else {
        f64::INFINITY
    }
}

/// Add one step to a running total. With `COUNT` on, a step that cannot be paid is counted instead
/// of summed, which keeps it from erasing the costs around it.
#[inline]
fn pay<const COUNT: bool>(breaks: u32, cost: f64, step: f64) -> (u32, f64) {
    if COUNT && !step.is_finite() {
        (breaks + 1, cost)
    } else {
        (breaks, cost + step)
    }
}

/// Transitions from every state to every state. Non-finite entries mean the change is impossible.
fn jump_table<T: Fn(usize, usize) -> f64>(states: usize, weight: f64, transition: T) -> Vec<f64> {
    let mut jump = vec![0.0; states * states];
    if weight.is_finite() && weight != 0.0 && states > 1 {
        for from in 0..states {
            for to in 0..states {
                jump[from * states + to] = payable(weight * transition(from, to));
            }
        }
    }
    jump
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
    fn a_step_nothing_explains_does_not_erase_the_steps_after_it() {
        let emission = |step: usize, state: usize| match step {
            1 => f64::INFINITY,
            _ => f64::from(u8::from(state != 1)) * 500.0,
        };

        let decoded = decode_path_with_cost(3, 2, 0.0, emission, |_, _| 0.0);

        assert_eq!(decoded.path, vec![1, 0, 1]);
        assert!(decoded.cost.is_infinite(), "an impossible step costs infinity");
    }

    #[test]
    fn impossible_steps_are_counted_before_cost_is_compared() {
        let emission = |_: usize, state: usize| match state {
            0 => f64::INFINITY,
            _ => 1e9,
        };

        let path = decode_path(4, 2, 0.0, emission, |_, _| 0.0);

        assert_eq!(path, vec![1, 1, 1, 1], "a payable step beats a cheaper impossible one");
    }

    #[test]
    fn the_decoder_matches_brute_force_when_some_steps_are_impossible() {
        let (steps, states, weight) = (5, 3, 0.5);
        let emission = |step: usize, state: usize| {
            if (step * 3 + state) % 7 == 0 {
                f64::INFINITY
            } else {
                f64::from(u32::try_from((step * 5 + state * 11) % 13).unwrap())
            }
        };
        let transition = |from: usize, to: usize| f64::from(u8::from(from != to));

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition);
        let margins = decode_margins(steps, states, weight, emission, transition);

        let score = |path: &[usize]| {
            let (mut breaks, mut cost) = (0_u32, emission(0, path[0]));
            if !cost.is_finite() {
                breaks += 1;
                cost = 0.0;
            }
            for step in 1..steps {
                for each in
                    [emission(step, path[step]), weight * transition(path[step - 1], path[step])]
                {
                    if each.is_finite() {
                        cost += each;
                    } else {
                        breaks += 1;
                    }
                }
            }
            (breaks, cost)
        };

        let better = |a: (u32, f64), b: (u32, f64)| if (a.0, a.1) < (b.0, b.1) { a } else { b };
        let every_path = |fixed: Option<(usize, usize)>| {
            let mut best = (u32::MAX, f64::INFINITY);
            for encoded in 0..states.pow(u32::try_from(steps).unwrap()) {
                let mut rest = encoded;
                let mut path = vec![0; steps];
                for slot in &mut path {
                    *slot = rest % states;
                    rest /= states;
                }
                if let Some((step, state)) = fixed {
                    if path[step] != state {
                        continue;
                    }
                }
                best = better(best, score(&path));
            }
            best
        };

        let best = every_path(None);
        assert_eq!(score(&decoded.path), best, "the decode is the best path, breaks first");

        for (step, &margin) in margins.iter().enumerate() {
            let alternative = (0..states)
                .filter(|&state| state != decoded.path[step])
                .map(|state| every_path(Some((step, state))))
                .fold((u32::MAX, f64::INFINITY), better);
            let expected =
                if alternative.0 == best.0 { alternative.1 - best.1 } else { f64::INFINITY };
            assert!(
                (margin - expected).abs() < 1e-9
                    || (margin.is_infinite() && expected.is_infinite()),
                "step {step} margin {margin} is not {expected}"
            );
        }
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
    fn a_negative_infinity_cost_does_not_corrupt_a_payable_decode() {
        let emission = |step: usize, state: usize| match (step, state) {
            (0, 0) => f64::NEG_INFINITY,
            (1, 0) => 10.0,
            (1, 1) => 5.0,
            _ => 0.0,
        };

        let decoded = decode_path_with_cost(2, 2, 0.0, emission, |_, _| 0.0);

        assert_eq!(decoded.path, vec![1, 1], "no cost is better than free");
        assert!((decoded.cost - 5.0).abs() < f64::EPSILON, "the cost is the payable path");
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
