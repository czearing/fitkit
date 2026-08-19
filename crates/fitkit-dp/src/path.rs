use crate::chosen::{Chosen, Tally, Trace};
use alloc::vec;
use alloc::vec::Vec;
use fitkit_core::{Answer, Refusal};

/// A decoded path and the search that produced it.
///
/// Reached only by running a decode. See [`Chosen`] for why the type has no constructor.
pub type Decoded = Chosen<Vec<usize>>;

/// A problem with nothing in it.
const EMPTY: Refusal = Refusal::unreported("a decode needs at least one step and one candidate");

/// More states than the trail that remembers where each one was reached from can name.
///
/// A caller who has this many states has a different problem from the one this solves, and is
/// owed that answer rather than a panic from inside a library.
const WIDE: Refusal = Refusal::incoherent(
    "more states than the backpointer that records where each was reached from",
);

/// A problem whose answer was never in doubt.
const SETTLED: Refusal =
    Refusal::uninformative("no step offered an affordable alternative to the path returned");

/// Hand back a witness, or refuse a search that had nothing to decide.
fn witnessed(path: Vec<usize>, cost: f64, trace: Trace) -> Answer<Decoded> {
    if trace.decided() {
        Ok(Chosen::new(path, cost, trace))
    } else {
        Err(SETTLED)
    }
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
/// cost second.
///
/// # Errors
///
/// Refuses an empty problem, and refuses one where no step ever offered an affordable alternative
/// to the path returned. The second case is a corridor rather than a search: the answer was the
/// only answer, and returning it as a decoded result would dress a foregone conclusion as a
/// choice.
///
/// Runs in `O(steps * states^2)` time. Each cost closure is called once per distinct argument.
pub fn decode_path<E, T>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
) -> Answer<Vec<usize>>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    decode_path_with_cost(steps, states, transition_weight, emission, transition)
        .map(Chosen::into_inner)
}

/// Decode, and build the caller's own result from the path in the same call.
///
/// This is how a caller obtains a [`Chosen`] over a type of its own. `build` runs on the decoded
/// path before any witness exists, so there is nothing yet to launder, and the witness that comes
/// back describes the search that produced the path `build` was given.
///
/// It exists because [`Chosen`] deliberately has no public `map`. An arbitrary transformation of a
/// witness already in hand is a mint: one honest search would license any number of witnessed
/// values it never produced. Here each witnessed value costs one search.
///
/// # Errors
///
/// As [`decode_path`].
///
/// # Panics
///
/// As [`decode_path_with_cost`].
pub fn decode_path_as<E, T, B, U>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    build: B,
) -> Answer<Chosen<U>>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    B: FnOnce(&[usize]) -> U,
{
    decode_path_with_cost(steps, states, transition_weight, emission, transition)
        .map(|decoded| decoded.map(|path| build(&path)))
}

/// Decode the best path, and build one part per step from the state chosen there.
///
/// The route a caller assembling something from a decode should take, and the counterpart of
/// [`optimise_subset_parts`](crate::optimise_subset_parts).
///
/// [`decode_path_as`] hands the builder the whole path, and a builder holding the answer can
/// ignore it and return something prepared. `part` is asked about one step and the state the
/// search put there, and never sees the path, so what it returns cannot depend on an answer it
/// was not told. The framework assembles, so there is exactly one part per step: a caller cannot
/// return a different number of pieces than the search decided on.
///
/// # Errors
///
/// As [`decode_path_with_cost`].
pub fn decode_path_parts<E, T, P, F>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    part: F,
) -> Answer<Chosen<Vec<P>>>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    F: Fn(usize, usize) -> P,
{
    decode_path_with_cost(steps, states, transition_weight, emission, transition).map(|decoded| {
        decoded
            .map(|path| path.iter().enumerate().map(|(step, &state)| part(step, state)).collect())
    })
}

/// As [`decode_path_onward`], building the caller's own result inside the same call.
///
/// See [`decode_path_as`] for why the building happens here rather than afterwards.
///
/// # Errors
///
/// As [`decode_path_onward`].
///
/// # Panics
///
/// As [`decode_path_onward`].
pub fn decode_path_onward_as<E, T, O, B, U>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    onward: O,
    build: B,
) -> Answer<Chosen<U>>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    O: Fn(usize) -> Vec<u32>,
    B: FnOnce(&[usize]) -> U,
{
    decode_path_onward(steps, states, transition_weight, emission, transition, onward)
        .map(|decoded| decoded.map(|path| build(&path)))
}

/// As [`decode_path_into`], building the caller's own result inside the same call.
///
/// See [`decode_path_as`] for why the building happens here rather than afterwards.
///
/// # Errors
///
/// As [`decode_path_into`].
///
/// # Panics
///
/// As [`decode_path_into`].
pub fn decode_path_into_as<E, T, I, B, U>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    into: I,
    build: B,
) -> Answer<Chosen<U>>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    I: Fn(usize) -> Vec<u32>,
    B: FnOnce(&[usize]) -> U,
{
    decode_path_into(steps, states, transition_weight, emission, transition, into)
        .map(|decoded| decoded.map(|path| build(&path)))
}

/// As [`decode_path`], keeping the witness: the cost paid, and what the search turned down.
/// Runs in `O(steps * states^2)` time. Each cost closure is called once per distinct argument.
///
/// # Errors
///
/// As [`decode_path`].
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
) -> Answer<Decoded>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    if steps == 0 || states == 0 {
        return Err(EMPTY);
    }
    if u32::try_from(states).is_err() {
        return Err(WIDE);
    }

    let jump = jump_table(states, transition_weight, transition);
    let (path, cost, tally) = trellis::<false, _>(steps, states, &jump, &emission);
    if cost.is_finite() {
        return witnessed(path, cost, Trace::new(steps, states, tally));
    }
    let (path, _, tally) = trellis::<true, _>(steps, states, &jump, &emission);
    witnessed(path, f64::INFINITY, Trace::new(steps, states, tally))
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
) -> (Vec<usize>, f64, Tally) {
    let mut tally = Tally::new();
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
            let mut affordable = 0_u64;
            for (from, &previous) in cost.iter().enumerate() {
                let carried = if COUNT { breaks[from] } else { 0 };
                let (b, c) = pay::<COUNT>(carried, previous, jump[from * states + to]);
                if c.is_finite() {
                    affordable += 1;
                }
                if b < best_breaks || (b == best_breaks && c < best_cost) {
                    (best_breaks, best_cost, best_from) = (b, c, from);
                }
            }
            let emitted = payable(emission(step, to));
            // A step nothing can explain is not a decision about how to reach it, whatever the
            // ways in cost, so the alternatives weighed here only count where the step is payable.
            if emitted.is_finite() {
                tally.decision(states as u64, affordable);
            }
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
    let (mut finishers, mut runner_up) = (0_u64, f64::INFINITY);
    for state in 0..states {
        let b = if COUNT { breaks[state] } else { 0 };
        if cost[state].is_finite() {
            finishers += 1;
        }
        if b < total_breaks || (b == total_breaks && cost[state] < total) {
            runner_up = runner_up.min(total);
            (total_breaks, total, end) = (b, cost[state], state);
        } else {
            runner_up = runner_up.min(cost[state]);
        }
    }
    tally.decision(states as u64, finishers);
    tally.ended(total, runner_up);

    let mut path = vec![0; steps];
    path[steps - 1] = end;
    for step in (1..steps).rev() {
        end = back[step * states + end] as usize;
        path[step - 1] = end;
    }

    (path, total, tally)
}

/// The lowest-cost path, when a state goes on to only a few others.
///
/// The same bargain as [`decode_path_into`], asked in the direction a model that carries context
/// can answer without building anything: naming the ways onward from a state costs one entry per
/// label, while naming the ways into it costs one per context that arrives there, which is the
/// whole grid again for a model of any size.
///
/// `onward` names the states each state may go on to, by index into the grid.
///
/// # Errors
///
/// As [`decode_path`].
///
/// # Panics
///
/// If `onward` names a state outside the grid, or if `states` exceeds the backpointer width.
pub fn decode_path_onward<E, T, O>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    onward: O,
) -> Answer<Decoded>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    O: Fn(usize) -> Vec<u32>,
{
    if steps == 0 || states == 0 {
        return Err(EMPTY);
    }
    if u32::try_from(states).is_err() {
        return Err(WIDE);
    }

    let mut tally = Tally::new();
    // Asked for one state at a time and kept, rather than laid out in full before the decode
    // starts. A model that names its own reachable states usually has far more states than any
    // one signal can visit, and building every list costs more than the decode it serves. The
    // ones actually reached are asked for once and reused at every later step.
    let mut out_of: Vec<Option<Vec<u32>>> = vec![None; states];
    let weighted = |from: usize, to: usize| {
        if transition_weight.is_finite() && transition_weight != 0.0 && states > 1 {
            payable(transition_weight * transition(from, to))
        } else {
            0.0
        }
    };

    let mut cost: Vec<f64> = (0..states).map(|state| payable(emission(0, state))).collect();
    // The states a path can still be in. Every other state costs infinity, and a step spent
    // sweeping them re-reads the whole grid to learn what it already knew. Carrying the live ones
    // makes a step cost what the signal reaches rather than what the model could describe.
    let mut live: Vec<usize> = (0..states).filter(|&state| cost[state].is_finite()).collect();
    // Written before they are read, at the step that reaches them, so neither needs a starting
    // value: `priced` records exactly which entries of this step are meaningful.
    let mut next = vec![0.0_f64; states];
    let mut here = vec![0.0_f64; states];
    // Which step each state's cost was settled for, so a state is priced once per step and only
    // if something still in play can reach it. Steps counted here start at one, so zero is a
    // stamp no step can match and the array needs no other clearing.
    let mut asked: Vec<usize> = vec![0; states];
    let mut priced: Vec<usize> = Vec::new();
    // How many ways into each priced state the step was offered, and how many of those could be
    // paid for. The first says whether the step was a decision at all, the second how contested it
    // was once the evidence had spoken.
    let mut offers: Vec<u64> = vec![0; states];
    let mut ways: Vec<u64> = vec![0; states];
    // Where each live state was reached from. Held for the states a step actually reaches rather
    // than for the whole grid at every step, which would allocate a backpointer for each state
    // the signal was never in and spend more on clearing it than the decode costs.
    let mut came: Vec<u32> = vec![0; states];
    let mut trail: Vec<Vec<(u32, u32)>> = vec![Vec::new(); steps];

    for (step, seen) in trail.iter_mut().enumerate().skip(1) {
        priced.clear();
        for &from in &live {
            let previous = cost[from];
            if out_of[from].is_none() {
                let named = onward(from);
                for &to in &named {
                    assert!(
                        (to as usize) < states,
                        "a state outside the grid was named as reachable"
                    );
                }
                out_of[from] = Some(named);
            }
            let reachable = out_of[from].as_deref().unwrap_or_default();
            for &to in reachable {
                let to = to as usize;
                // What a step costs is asked before how it might be reached, so a model that
                // rules most states out at most steps pays only for the ones still in play.
                if asked[to] != step {
                    asked[to] = step;
                    here[to] = payable(emission(step, to));
                    next[to] = f64::INFINITY;
                    offers[to] = 0;
                    ways[to] = 0;
                    priced.push(to);
                }
                offers[to] += 1;
                if !here[to].is_finite() {
                    continue;
                }
                let total = previous + weighted(from, to);
                if total.is_finite() {
                    ways[to] += 1;
                }
                if total < next[to] {
                    next[to] = total;
                    came[to] = u32::try_from(from).unwrap_or(0);
                }
            }
        }
        for &state in &live {
            cost[state] = f64::INFINITY;
        }
        live.clear();
        for &state in &priced {
            let paid = next[state] + here[state];
            cost[state] = paid;
            if here[state].is_finite() {
                tally.decision(offers[state], ways[state]);
            }
            if paid.is_finite() {
                live.push(state);
            }
        }
        // Relaxed in the order the whole grid would have been swept in, because a tie between two
        // ways into a state is settled by which was offered first. Reaching them in the order the
        // search happened to find them would decide ties differently.
        live.sort_unstable();
        *seen =
            live.iter().map(|&state| (u32::try_from(state).unwrap_or(0), came[state])).collect();
    }

    let (mut end, total) = finish(&cost, &mut tally);

    let mut path = vec![0; steps];
    path[steps - 1] = end;
    for step in (1..steps).rev() {
        end = trail[step]
            .binary_search_by_key(&u32::try_from(end).unwrap_or(0), |&(state, _)| state)
            .map_or(0, |found| trail[step][found].1 as usize);
        path[step - 1] = end;
    }

    witnessed(path, total, Trace::new(steps, states, tally))
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
    margins_of(steps, states, transition_weight, emission, transition, None, None)
}

/// How much worse the best alternative is, when a state goes on to only a few others.
///
/// The margin counterpart of [`decode_path_onward`], and the direction a model that carries
/// context can answer cheaply: the context after a step follows from the one before it and the
/// label chosen, so a state has as many ways onward as there are labels, while the ways into it
/// are as many as the contexts that arrive at the same place. Asking onward is what keeps the
/// margin affordable at the state counts such a model reaches.
///
/// `onward` names the states each state may go on to, by index.
///
/// `apart` says which states are different answers rather than different states. A model that
/// carries context holds more than it reports, and two states that report the same thing under
/// different contexts are not alternatives to each other: treating them as such measures how sure
/// the model is of the context, which it was not asked, instead of how sure it is of the answer,
/// which it was. States sharing a key are one answer.
///
/// # Panics
///
/// If `onward` names a state outside the grid.
pub fn decode_margins_onward<E, T, O, A>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    onward: O,
    apart: A,
) -> Vec<f64>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    O: Fn(usize) -> Vec<u32>,
    A: Fn(usize) -> u64,
{
    let out_of: Vec<Vec<u32>> = (0..states).map(&onward).collect();
    for list in &out_of {
        for &to in list {
            assert!((to as usize) < states, "a state outside the grid was named as reachable");
        }
    }
    let apart: Vec<u64> = (0..states).map(&apart).collect();
    margins_of(steps, states, transition_weight, emission, transition, Some(&out_of), Some(&apart))
}

/// Both margin passes, with and without a predecessor map.
fn margins_of<E, T>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    out_of: Option<&[Vec<u32>]>,
    apart: Option<&[u64]>,
) -> Vec<f64>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
{
    if steps == 0 || states == 0 {
        return Vec::new();
    }

    let mut emit_breaks = vec![0_u32; steps * states];
    let mut emit_cost = vec![0.0; steps * states];
    for at in 0..steps * states {
        let emitted = payable(emission(at / states, at % states));
        (emit_breaks[at], emit_cost[at]) = pay::<true>(0, 0.0, emitted);
    }

    let weighted = |from: usize, to: usize| {
        if transition_weight.is_finite() && transition_weight != 0.0 && states > 1 {
            payable(transition_weight * transition(from, to))
        } else {
            0.0
        }
    };

    // Only the states the evidence leaves standing at each step. A model whose emission rules most
    // states out at most steps pays for what is still in play rather than for the whole grid.
    // Anything ruled out costs a break, and a break beats any cost, so a state that is out can
    // never be the best or the runner up as long as some path is in throughout.
    let live: Vec<Vec<u32>> = (0..steps)
        .map(|step| {
            (0..states)
                .filter(|state| emit_breaks[step * states + state] == 0)
                .filter_map(|state| u32::try_from(state).ok())
                .collect()
        })
        .collect();

    let walk = |standing: &[Vec<u32>], prove: bool| -> Option<Vec<f64>> {
        walk_within(
            steps,
            states,
            (&emit_breaks, &emit_cost),
            (&weighted, out_of),
            (standing, apart),
            prove,
        )
    };

    if !live.iter().any(Vec::is_empty) {
        if let Some(found) = walk(&live, true) {
            return found;
        }
    }

    // The pruning could have changed the answer, so the grid is walked whole. With a predecessor
    // map that is still only the pairs the model allows; without one it is every pair there is.
    let whole: Vec<u32> = (0..states).filter_map(|state| u32::try_from(state).ok()).collect();
    if out_of.is_some() {
        let standing = vec![whole; steps];
        if let Some(found) = walk(&standing, false) {
            return found;
        }
    }
    dense_margins(steps, states, transition_weight, emission, transition)
}

/// What a step between two states costs, and which steps are possible at all.
///
/// Nothing for the second means any state may follow any other, which is the dense search.
type Reach<'a> = (&'a dyn Fn(usize, usize) -> f64, Option<&'a [Vec<u32>]>);

/// What each state costs at each step, as a break count and a price.
type Emitted<'a> = (&'a [u32], &'a [f64]);

/// The grid one walk is allowed to touch.
struct Within<'a> {
    steps: usize,
    states: usize,
    emit_breaks: &'a [u32],
    emit_cost: &'a [f64],
    weighted: &'a dyn Fn(usize, usize) -> f64,
    /// The states each state may go on to, or nothing when any of them may follow.
    out_of: Option<&'a [Vec<u32>]>,
    /// The states left standing at each step.
    standing: &'a [Vec<u32>],
    /// The standing states, indexed by step and state.
    inside: Vec<bool>,
}

impl Within<'_> {
    fn at(&self, step: usize, state: usize) -> usize {
        step * self.states + state
    }

    fn emitted(&self, at: usize) -> (u32, f64) {
        (self.emit_breaks[at], self.emit_cost[at])
    }

    /// The cost of reaching each state from the first step.
    ///
    /// Pushed from the states still standing rather than pulled into them. A state names far more
    /// ways it could have been reached than ways it goes on, since the context after a step is
    /// decided by the one before it and the label chosen while many contexts arrive at the same
    /// place. Reading the map in the direction it is sparse in is the whole saving.
    fn ahead(&self) -> Vec<(u32, f64)> {
        let out = (u32::MAX, f64::INFINITY);
        let mut forward = vec![out; self.steps * self.states];
        for &state in &self.standing[0] {
            forward[state as usize] = self.emitted(state as usize);
        }

        for step in 1..self.steps {
            for &from in &self.standing[step - 1] {
                let from = from as usize;
                let (breaks, cost) = forward[self.at(step - 1, from)];
                if breaks == u32::MAX {
                    continue;
                }
                let mut consider = |to: usize| {
                    let at = self.at(step, to);
                    if !self.inside[at] {
                        return;
                    }
                    let paid = pay::<true>(breaks, cost, (self.weighted)(from, to));
                    let reach = (paid.0 + self.emit_breaks[at], paid.1 + self.emit_cost[at]);
                    if forward[at].0 == u32::MAX || reach < forward[at] {
                        forward[at] = reach;
                    }
                };
                match self.out_of {
                    Some(out_of) => out_of[from].iter().for_each(|&to| consider(to as usize)),
                    None => self.standing[step].iter().for_each(|&to| consider(to as usize)),
                }
            }
        }
        forward
    }

    /// The cost of reaching the last step from each state.
    fn behind(&self) -> Vec<(u32, f64)> {
        let out = (u32::MAX, f64::INFINITY);
        let mut backward = vec![out; self.steps * self.states];
        for &state in &self.standing[self.steps - 1] {
            let at = self.at(self.steps - 1, state as usize);
            backward[at] = self.emitted(at);
        }

        for step in (0..self.steps - 1).rev() {
            for &from in &self.standing[step] {
                let from = from as usize;
                let mut best = out;
                let mut consider = |to: usize| {
                    let ahead = self.at(step + 1, to);
                    if !self.inside[ahead] {
                        return;
                    }
                    let (breaks, cost) = backward[ahead];
                    if breaks == u32::MAX {
                        return;
                    }
                    let paid = pay::<true>(breaks, cost, (self.weighted)(from, to));
                    if paid < best {
                        best = paid;
                    }
                };
                match self.out_of {
                    Some(out_of) => out_of[from].iter().for_each(|&to| consider(to as usize)),
                    None => self.standing[step + 1].iter().for_each(|&to| consider(to as usize)),
                }
                let at = self.at(step, from);
                if best.0 != u32::MAX {
                    backward[at] = (best.0 + self.emit_breaks[at], best.1 + self.emit_cost[at]);
                }
            }
        }
        backward
    }
}

impl<'a> Within<'a> {
    fn new(
        steps: usize,
        states: usize,
        emitted: Emitted<'a>,
        reach: Reach<'a>,
        told: (&'a [Vec<u32>], Option<&'a [u64]>),
    ) -> Self {
        let (standing, _) = told;
        let mut inside = vec![false; steps * states];
        for (step, states_here) in standing.iter().enumerate() {
            for &state in states_here {
                inside[step * states + state as usize] = true;
            }
        }
        let (emit_breaks, emit_cost) = emitted;
        let (weighted, out_of) = reach;
        Self { steps, states, emit_breaks, emit_cost, weighted, out_of, standing, inside }
    }
}

/// One pruned walk of the grid, forwards then backwards, over the states named at each step.
///
/// Returns nothing when `prove` is asked for and no path avoided every state that was left out,
/// since then the pruning could have changed the answer.
fn walk_within(
    steps: usize,
    states: usize,
    emitted: Emitted,
    reach: Reach,
    told: (&[Vec<u32>], Option<&[u64]>),
    prove: bool,
) -> Option<Vec<f64>> {
    let (standing, apart) = told;
    let (emit_breaks, emit_cost) = emitted;
    let within = Within::new(steps, states, emitted, reach, told);

    let forward = within.ahead();

    // A zero break path proves nothing was forced through a state that was left out, so
    // restricting the search cannot have changed the answer. Without one the answer turns on where
    // the unavoidable break falls, and nothing may be skipped.
    if prove
        && standing[steps - 1]
            .iter()
            .all(|&state| forward[(steps - 1) * states + state as usize].0 != 0)
    {
        return None;
    }

    let backward = within.behind();
    let out = (u32::MAX, f64::INFINITY);

    Some(
        (0..steps)
            .map(|step| {
                let answer =
                    |state: u32| apart.map_or(u64::from(state), |apart| apart[state as usize]);
                let (mut best, mut runner_up) = ((out, 0), out);
                for &state in &standing[step] {
                    let at = step * states + state as usize;
                    if forward[at].0 == u32::MAX || backward[at].0 == u32::MAX {
                        continue;
                    }
                    let total = (
                        forward[at].0 + backward[at].0 - emit_breaks[at],
                        forward[at].1 + backward[at].1 - emit_cost[at],
                    );
                    // The runner up has to be a different answer, not merely a different state,
                    // or the margin reports how sure the model is of what it kept to itself.
                    if total < best.0 {
                        if answer(state) != best.1 && best.0 < runner_up {
                            runner_up = best.0;
                        }
                        best = (total, answer(state));
                    } else if answer(state) != best.1 && total < runner_up {
                        runner_up = total;
                    }
                }
                if runner_up.0 != best.0 .0 || !runner_up.1.is_finite() {
                    f64::INFINITY
                } else {
                    runner_up.1 - best.0 .1
                }
            })
            .collect(),
    )
}

/// The whole grid, walked without pruning.
///
/// Kept for the case where no path avoids a state the evidence rules out. Then a break is
/// unavoidable and which state carries it decides the answer, so nothing may be skipped.
fn dense_margins<E, T>(
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

/// The end of a path: which state finished cheapest, what it cost, and how much worse the best
/// other finisher was. Choosing between the ways a path can end is itself a decision, and for a
/// one-step problem it is the only one there is.
fn finish(cost: &[f64], tally: &mut Tally) -> (usize, f64) {
    let (mut end, mut total) = (0, f64::INFINITY);
    let (mut finishers, mut runner_up) = (0_u64, f64::INFINITY);
    for (state, &paid) in cost.iter().enumerate() {
        if paid.is_finite() {
            finishers += 1;
        }
        if paid < total {
            runner_up = runner_up.min(total);
            (total, end) = (paid, state);
        } else {
            runner_up = runner_up.min(paid);
        }
    }
    tally.decision(cost.len() as u64, finishers);
    tally.ended(total, runner_up);
    (end, total)
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
/// Decode the lowest-cost path when most changes of state are impossible.
///
/// As [`decode_path_with_cost`], except that `into(to)` names the states a step may be reached
/// from. Every other state is treated as unreachable rather than as expensive, so the transition
/// closure is never asked about a pair that cannot occur.
///
/// This is what a search over structured state needs. When a state carries context as well as a
/// label, such as an operating mode together with how long the equipment has been held in it, the
/// number of states multiplies but the number of ways to reach any one of them does not: the
/// context that follows is decided by the context before it and the label chosen. A dense decode
/// pays `states^2` per step regardless and spends nearly all of it proving that impossible moves
/// are impossible.
///
/// Runs in `O(steps * sum of the sizes of into)` time. Each cost closure is called once per
/// distinct argument. Passing every state for every step gives exactly [`decode_path_with_cost`],
/// at the cost of building the lists.
///
/// # Errors
///
/// As [`decode_path`].
///
/// # Panics
///
/// If `states` exceeds `u32::MAX`, the width of the backpointer table, or if `into` names a state
/// at or beyond `states`.
#[allow(clippy::cast_possible_truncation)] // states is asserted to fit u32 below
pub fn decode_path_into<E, T, I>(
    steps: usize,
    states: usize,
    transition_weight: f64,
    emission: E,
    transition: T,
    into: I,
) -> Answer<Decoded>
where
    E: Fn(usize, usize) -> f64,
    T: Fn(usize, usize) -> f64,
    I: Fn(usize) -> Vec<u32>,
{
    if steps == 0 || states == 0 {
        return Err(EMPTY);
    }
    if u32::try_from(states).is_err() {
        return Err(WIDE);
    }

    let mut tally = Tally::new();
    let reached: Vec<Vec<u32>> = (0..states).map(&into).collect();
    for list in &reached {
        for &from in list {
            assert!((from as usize) < states, "a state outside the grid was named as reachable");
        }
    }
    let weighted = |from: usize, to: usize| {
        if transition_weight.is_finite() && transition_weight != 0.0 && states > 1 {
            payable(transition_weight * transition(from, to))
        } else {
            0.0
        }
    };

    let mut cost: Vec<f64> = (0..states).map(|state| payable(emission(0, state))).collect();
    let mut next = cost.clone();
    let mut back = vec![0_u32; steps * states];

    for step in 1..steps {
        for to in 0..states {
            // A state the evidence rules out cannot be reached at any price, so the search for how
            // it might have been reached is wasted. Asking what a step costs before asking how to
            // get there is what makes a large state space affordable: models that give most states
            // an infinite emission at most steps now pay only for the states still in play.
            let here = payable(emission(step, to));
            if !here.is_finite() {
                next[to] = f64::INFINITY;
                back[step * states + to] = 0;
                continue;
            }
            let (mut best, mut best_from) = (f64::INFINITY, 0_u32);
            let mut affordable = 0_u64;
            for &from in &reached[to] {
                let previous = cost[from as usize];
                if !previous.is_finite() {
                    continue;
                }
                let total = previous + weighted(from as usize, to);
                if total.is_finite() {
                    affordable += 1;
                }
                if total < best {
                    (best, best_from) = (total, from);
                }
            }
            tally.decision(reached[to].len() as u64, affordable);
            next[to] = best + here;
            back[step * states + to] = best_from;
        }
        cost.copy_from_slice(&next);
    }

    let (mut end, total) = finish(&cost, &mut tally);

    let mut path = vec![0; steps];
    path[steps - 1] = end;
    for step in (1..steps).rev() {
        end = back[step * states + end] as usize;
        path[step - 1] = end;
    }

    witnessed(path, total, Trace::new(steps, states, tally))
}

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

    use super::{
        decode_margins, decode_margins_onward, decode_path, decode_path_into, decode_path_with_cost,
    };
    use fitkit_core::RefusalKind;

    /// The three decodes are one search, so on the same model they must return the same path.
    ///
    /// Random emissions and random transitions, with a third of the moves forbidden outright, so
    /// the sparse forms have something to be sparse about and the dense form has to price the
    /// impossible moves it is handed. A disagreement is one of them decoding a different path,
    /// which no consumer could detect from the outside: all three return an authentic witness.
    #[test]
    fn the_sparse_decodes_answer_exactly_what_the_dense_decode_answers() {
        for seed in 0..200_u64 {
            let steps = 3 + (seed % 6) as usize;
            let states = 2 + (seed % 5) as usize;
            let mut noise = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(11);
            let mut next = move || {
                noise = noise
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (noise >> 33) % 1000
            };
            let emissions: alloc::vec::Vec<f64> =
                (0..steps * states).map(|_| next() as f64 / 100.0).collect();
            let moves: alloc::vec::Vec<f64> = (0..states * states)
                .map(|_| {
                    let draw = next();
                    if draw % 3 == 0 {
                        f64::INFINITY
                    } else {
                        draw as f64 / 100.0
                    }
                })
                .collect();
            // Every state keeps one way in and one way out, so no model is unsolvable.
            let mut moves = moves;
            for state in 0..states {
                moves[state * states + state] = 1.0;
            }
            let emission = |step: usize, state: usize| emissions[step * states + state];
            let transition = |from: usize, to: usize| moves[from * states + to];
            let onward = |from: usize| {
                (0..states)
                    .filter(|&to| moves[from * states + to].is_finite())
                    .map(|to| u32::try_from(to).expect("a state index inside a small model"))
                    .collect::<alloc::vec::Vec<u32>>()
            };
            let into = |to: usize| {
                (0..states)
                    .filter(|&from| moves[from * states + to].is_finite())
                    .map(|from| u32::try_from(from).expect("a state index inside a small model"))
                    .collect::<alloc::vec::Vec<u32>>()
            };
            let dense = decode_path_with_cost(steps, states, 1.0, emission, transition)
                .expect("a model with a way through");
            let forward =
                super::decode_path_onward(steps, states, 1.0, emission, transition, onward)
                    .expect("a model with a way through");
            let backward = decode_path_into(steps, states, 1.0, emission, transition, into)
                .expect("a model with a way through");
            assert!(
                (dense.cost() - forward.cost()).abs() < 1e-9,
                "seed {seed}: dense cost {} against onward cost {}",
                dense.cost(),
                forward.cost()
            );
            assert!(
                (dense.cost() - backward.cost()).abs() < 1e-9,
                "seed {seed}: dense cost {} against into cost {}",
                dense.cost(),
                backward.cost()
            );
            assert_eq!(
                dense.get(),
                forward.get(),
                "seed {seed}: dense and onward decoded different paths"
            );
            assert_eq!(
                dense.get(),
                backward.get(),
                "seed {seed}: dense and into decoded different paths"
            );
        }
    }

    /// More states than the trail can name is an answer this cannot give, not a reason to panic.
    ///
    /// The check happens before anything is allocated, so a caller who asks gets told rather than
    /// killed. A library that panics takes the decision about how to fail away from its caller.
    #[test]
    fn more_states_than_the_trail_can_name_is_refused_rather_than_a_panic() {
        let states = u32::MAX as usize + 1;
        let refused = decode_path_with_cost(2, states, 1.0, |_, _| 1.0, |_, _| 0.0)
            .expect_err("more states than a backpointer can name");
        assert_eq!(refused.kind(), RefusalKind::Incoherent);
    }

    #[test]
    fn naming_every_state_decodes_exactly_as_the_dense_search_does() {
        let observed = [1.0, 0.0, 1.0, 1.0, 0.0];
        let emission = |step: usize, state: usize| (observed[step] - state as f64).abs();
        let transition = |from: usize, to: usize| (from as f64 - to as f64).abs();
        let dense = decode_path_with_cost(5, 3, 1.0, emission, transition)
            .expect("the search decided something");
        let sparse = decode_path_into(5, 3, 1.0, emission, transition, |_| vec![0, 1, 2])
            .expect("the search decided something");
        assert_eq!(dense.get(), sparse.get());
        assert!((dense.cost() - sparse.cost()).abs() < 1e-9);
    }

    #[test]
    fn a_state_no_one_can_reach_is_never_chosen() {
        // State 1 explains every step perfectly but nothing leads into it, so the path has to go
        // around it after the first step.
        let emission = |_: usize, state: usize| if state == 1 { 0.0 } else { 1.0 };
        let transition = |_: usize, _: usize| 0.0;
        let decoded = decode_path_into(4, 3, 1.0, emission, transition, |to| {
            if to == 1 {
                vec![]
            } else {
                vec![0, 2]
            }
        })
        .expect("the search decided something");
        assert_eq!(&decoded.get()[1..], &[0, 0, 0]);
    }

    #[test]
    fn a_forced_chain_is_followed_even_where_the_evidence_disagrees() {
        // Each state may only be reached from the one before it, so the path is decided by the
        // shape of the search rather than by what any step measured.
        let emission = |_: usize, state: usize| if state == 0 { 0.0 } else { 1.0 };
        let transition = |_: usize, _: usize| 0.0;
        let decoded = decode_path_into(4, 4, 1.0, emission, transition, |to| {
            if to == 0 {
                vec![]
            } else {
                vec![u32::try_from(to).unwrap_or(0) - 1]
            }
        })
        .expect("the search decided something");
        assert_eq!(decoded.get(), &vec![0, 1, 2, 3]);
    }

    #[test]
    fn with_no_resistance_the_path_follows_the_evidence() {
        let observed = [1, 0, 1];
        let path = decode_path(
            3,
            2,
            0.0,
            |step, state| f64::from(u8::from(observed[step] != state)),
            |_, _| 1.0,
        )
        .expect("the search decided something");
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
        )
        .expect("the search decided something");
        assert_eq!(path, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn an_empty_problem_is_refused_rather_than_answered() {
        for (steps, states) in [(0, 4), (4, 0)] {
            let refused = decode_path(steps, states, 1.0, |_, _| 0.0, |_, _| 0.0)
                .expect_err("nothing was there to decode");
            assert_eq!(refused.kind(), RefusalKind::Unreported);
        }
    }

    #[test]
    fn a_model_with_one_candidate_is_refused_rather_than_witnessed() {
        // The shape a forged search takes: one state, a constant emission, nothing to weigh. The
        // trellis runs and the path is correct, and it is not a decode of anything.
        let refused = decode_path_with_cost(6, 1, 1.0, |_, _| 0.5, |_, _| 0.0)
            .expect_err("one candidate is not a choice");
        assert_eq!(refused.kind(), RefusalKind::Uninformative);
    }

    #[test]
    fn evidence_that_rules_out_every_rival_is_still_a_decode() {
        // Two candidates offered, one impossible at every step. The model did its part, so the
        // decode answers and says in its trace that nothing affordable was turned down.
        let decoded = decode_path_with_cost(
            4,
            2,
            0.0,
            |_, state| if state == 0 { f64::INFINITY } else { 1.0 },
            |_, _| 0.0,
        )
        .expect("the model offered a choice");
        assert_eq!(decoded.get(), &vec![1, 1, 1, 1]);
        assert!(decoded.trace().choices() > 0);
        assert_eq!(decoded.trace().rejected(), 0, "the evidence settled it, not the cost");
    }

    #[test]
    fn a_contested_decode_reports_what_it_turned_down() {
        let observed = [1.0, 0.0, 1.0, 1.0, 0.0];
        let decoded = decode_path_with_cost(
            5,
            3,
            1.0,
            |step: usize, state: usize| (observed[step] - state as f64).abs(),
            |from: usize, to: usize| (from as f64 - to as f64).abs(),
        )
        .expect("the model offered a choice");
        let trace = decoded.trace();
        assert!(trace.rejected() > 0, "affordable rivals lost");
        assert!(trace.considered() >= trace.rejected());
        assert!(trace.margin().is_finite(), "another path finished");
    }

    #[test]
    fn a_witness_cannot_be_rebuilt_from_the_result_it_carries() {
        // The compiler enforces this: Chosen has no public constructor, so the only way to hold
        // one is to have run a decode. What can be done is to carry a decoded result forward.
        let decoded = decode_path_with_cost(3, 2, 0.0, |_, state| state as f64, |_, _| 0.0)
            .expect("the model offered a choice");
        let rejected = decoded.trace().rejected();
        let mapped = decoded.map(|path| path.len());
        assert_eq!(*mapped.get(), 3);
        assert_eq!(mapped.trace().rejected(), rejected, "the account survives the transformation");
    }

    #[test]
    fn a_nan_cost_never_wins() {
        let path =
            decode_path(2, 2, 0.0, |_, state| if state == 0 { f64::NAN } else { 1.0 }, |_, _| 0.0)
                .expect("the search decided something");
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
        )
        .expect("the search decided something");
        let mut recomputed = emissions[0][decoded.get()[0]];
        for step in 1..3 {
            recomputed += emissions[step][decoded.get()[step]];
            if decoded.get()[step - 1] != decoded.get()[step] {
                recomputed += weight;
            }
        }
        assert!((decoded.cost() - recomputed).abs() < 1e-12);
    }

    #[test]
    fn the_decoder_matches_brute_force() {
        let (steps, states, weight) = (6, 4, 0.75);
        let emission = |step: usize, state: usize| ((step * 7 + state * 13) % 11) as f64 / 3.0;
        let transition = |from: usize, to: usize| (from as f64 - to as f64).abs();

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition)
            .expect("the search decided something");

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
        assert!((decoded.cost() - best).abs() < 1e-9);
    }

    #[test]
    fn a_step_nothing_explains_does_not_erase_the_steps_after_it() {
        let emission = |step: usize, state: usize| match step {
            1 => f64::INFINITY,
            _ => f64::from(u8::from(state != 1)) * 500.0,
        };

        let decoded = decode_path_with_cost(3, 2, 0.0, emission, |_, _| 0.0)
            .expect("the search decided something");

        assert_eq!(decoded.get(), &vec![1, 0, 1]);
        assert!(decoded.cost().is_infinite(), "an impossible step costs infinity");
    }

    #[test]
    fn impossible_steps_are_counted_before_cost_is_compared() {
        let emission = |_: usize, state: usize| match state {
            0 => f64::INFINITY,
            _ => 1e9,
        };

        let path =
            decode_path(4, 2, 0.0, emission, |_, _| 0.0).expect("the search decided something");

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

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition)
            .expect("the search decided something");
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
        assert_eq!(score(decoded.get()), best, "the decode is the best path, breaks first");

        for (step, &margin) in margins.iter().enumerate() {
            let alternative = (0..states)
                .filter(|&state| state != decoded.get()[step])
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

        let decoded = decode_path_with_cost(steps, states, weight, emission, transition)
            .expect("the search decided something");
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
                if path[step] != decoded.get()[step] {
                    best_alternative = best_alternative.min(total(&path));
                }
            }
            assert!(
                (margin - (best_alternative - decoded.cost())).abs() < 1e-9,
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

        let decoded = decode_path_with_cost(2, 2, 0.0, emission, |_, _| 0.0)
            .expect("the search decided something");

        assert_eq!(decoded.get(), &vec![1, 1], "no cost is better than free");
        assert!((decoded.cost() - 5.0).abs() < f64::EPSILON, "the cost is the payable path");
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

    #[test]
    fn naming_every_successor_margins_the_same_as_naming_none() {
        let emission = |step: usize, state: usize| ((step * 7 + state * 3) % 5) as f64;
        let transition = |from: usize, to: usize| if from == to { 0.0 } else { 1.0 };
        let whole = decode_margins(6, 4, 1.0, emission, transition);
        let named = decode_margins_onward(
            6,
            4,
            1.0,
            emission,
            transition,
            |_| vec![0, 1, 2, 3],
            |at| at as u64,
        );
        assert_eq!(whole, named);
    }

    #[test]
    fn a_successor_map_margins_what_walking_the_pairs_it_forbids_would() {
        // The map and the transition say the same thing, one by omission and one at infinite cost,
        // so the margin may not depend on which way it was said.
        // 0 goes on to 0 and 1, 1 to 1 and 2, 2 to 2 and 0.
        let onward = |from: usize| {
            let from = u32::try_from(from).unwrap_or(0);
            vec![from, (from + 1) % 3]
        };
        let emission = |step: usize, state: usize| ((step * 3 + state) % 4) as f64;
        let priced = |from: usize, to: usize| {
            if onward(from).contains(&u32::try_from(to).unwrap_or(0)) {
                f64::from(u8::from(from != to))
            } else {
                f64::INFINITY
            }
        };
        let whole = decode_margins(5, 3, 1.0, emission, priced);
        let named = decode_margins_onward(5, 3, 1.0, emission, priced, onward, |at| at as u64);
        assert_eq!(whole, named);
    }

    #[test]
    fn a_margin_survives_a_step_the_evidence_rules_out_entirely() {
        // Nothing is left standing at step two, so the pruned walk cannot answer and the whole
        // grid has to be walked. The margin is still the one the grid gives.
        let emission = |step: usize, state: usize| {
            if step == 2 {
                f64::INFINITY
            } else {
                f64::from(u8::try_from(state).unwrap_or(0))
            }
        };
        let transition = |from: usize, to: usize| f64::from(u8::from(from != to));
        let margins = decode_margins(4, 3, 1.0, emission, transition);
        let named = decode_margins_onward(
            4,
            3,
            1.0,
            emission,
            transition,
            |_| vec![0, 1, 2],
            |at| at as u64,
        );
        assert_eq!(margins.len(), 4);
        assert_eq!(margins, named);
    }

    #[test]
    fn states_that_report_the_same_thing_are_not_alternatives_to_each_other() {
        // Two pairs of states, tied within each pair and a clear step apart between them. Read as
        // four states the runner up is the tie, and the margin is nothing however plain the real
        // choice was. Read as two answers the tie is the same answer, and the margin is what
        // reporting the other one would have cost.
        let emission = |_step: usize, state: usize| if state < 2 { 0.0 } else { 4.0 };
        let transition = |_from: usize, _to: usize| 0.0;
        let apart = |state: usize| (state / 2) as u64;
        let onward = |_from: usize| vec![0, 1, 2, 3];

        let split = decode_margins_onward(3, 4, 1.0, emission, transition, onward, |at| at as u64);
        assert_eq!(split, vec![0.0; 3]);

        let joined = decode_margins_onward(3, 4, 1.0, emission, transition, onward, apart);
        assert_eq!(joined, vec![4.0; 3]);
    }
}
