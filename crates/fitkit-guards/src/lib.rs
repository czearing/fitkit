//! Invariant checks to call from your own tests.
//!
//! These are the rules that decide whether an engine is measuring or guessing. Each one panics
//! with the violation named.

use std::fmt::Debug;

use fitkit_core::{Cost, Plan, Scale};
use fitkit_dp::{optimise_subset, Terms};
use fitkit_feasible::{Feasible, Problem};
use fitkit_fit::{recover, Fit, Model, Segmented};

/// Panic if any banned symbol appears in any source text.
///
/// Pass files with [`include_str!`]. Use it to keep identity keyed lookups, such as a table from a
/// name to a number, out of the path that derives quantities. Reading the call graph once proves
/// nothing about tomorrow; this does.
///
/// # Panics
///
/// If a banned symbol is found.
pub fn forbid_symbols(sources: &[(&str, &str)], banned: &[&str]) {
    for (path, text) in sources {
        for symbol in banned {
            assert!(
                !text.contains(symbol),
                "{path} reaches {symbol}, so an answer can descend from it"
            );
        }
    }
}

/// Panic if any function signature derives one thing from another.
///
/// [`forbid_symbols`] bans a spelling. This bans a SHAPE, which is what the invariant "no answer
/// descends from a name" actually says: not that a word is absent, but that no function turns one
/// kind of input into a kind of output. A substring ban cannot express that. It either misses the
/// pattern or fires on every incidental mention.
///
/// A signature offends when its return type contains `returns` and its parameters contain any of
/// `accepts`. Each string in `exempt` is removed from the parameters first, for the case where a
/// banned type is present but demonstrably a passenger, such as a label carried alongside the
/// measurement that actually decides.
///
/// Line comments are stripped before parsing, so prose about the banned pattern is not evidence
/// against the file discussing it. Signatures are flattened first, so one broken across lines by a
/// formatter is still read whole.
///
/// ```
/// # use fitkit_guards::forbid_derivations_from;
/// forbid_derivations_from(
///     &[("solver.rs", "fn limits(sample: &Measured) -> Vec<Requirement> { todo!() }")],
///     "Requirement",
///     &["&str", "String", "name:"],
///     &[],
/// );
/// ```
///
/// # Panics
///
/// If any signature derives `returns` from one of `accepts`, naming every offender.
pub fn forbid_derivations_from(
    sources: &[(&str, &str)],
    returns: &str,
    accepts: &[&str],
    exempt: &[&str],
) {
    let mut offenders = Vec::new();
    for (path, text) in sources {
        for signature in signatures(&code_only(text)) {
            let Some(arrow) = signature.find("->") else {
                continue;
            };
            let (params, returned) = signature.split_at(arrow);
            if !returned.contains(returns) {
                continue;
            }
            let mut params = params.to_string();
            for allowed in exempt {
                params = params.replace(allowed, " ");
            }
            if accepts.iter().any(|banned| params.contains(banned)) {
                offenders.push(format!("{path}: fn {}", signature.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a function derives {returns} from an input it must not, so an answer can descend from \
         that input:\n{}",
        offenders.join("\n")
    );
}

/// Strip line comments, so prose about a banned pattern is not read as code.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|line| line.find("//").map_or(line, |cut| &line[..cut]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `fn` signature as one flattened string, ending at the body or at the
/// semicolon of a declaration, whichever comes first.
fn signatures(code: &str) -> Vec<String> {
    let flat = code.replace('\n', " ");
    let mut out = Vec::new();
    let mut rest = flat.as_str();
    while let Some(start) = rest.find("fn ") {
        rest = &rest[start + 3..];
        let end = rest.find('{').unwrap_or(rest.len()).min(rest.find(';').unwrap_or(rest.len()));
        out.push(rest[..end].to_string());
    }
    out
}

/// Panic unless a model with no evidence recovers the identity plan.
///
/// # Panics
///
/// If the model produces controls it cannot justify.
pub fn assert_identity_without_evidence<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug,
{
    let Ok(plan) = recover(model, reference) else {
        // A refusal is the stronger form of the same promise: no evidence, so no controls.
        return;
    };
    let plan = plan.get();
    assert!(plan.is_identity(), "{} invented {:?} from no evidence", model.name(), plan.controls);
}

/// Panic unless every zero confidence span is silent in the recovered plan.
///
/// # Panics
///
/// If an untrusted span would be acted on.
pub fn assert_untrusted_spans_stay_silent<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug,
{
    let evidence = model.evidence(reference);
    let Ok(plan) = recover(model, reference) else {
        return;
    };
    for (measured, control) in evidence.iter().zip(&plan.get().controls) {
        assert!(
            measured.is_informative() || control.is_silent(),
            "{} acts on {:?} where the evidence carried nothing",
            model.name(),
            control.params
        );
    }
}

/// Panic unless two recoveries from the same reference agree.
///
/// # Panics
///
/// If recovery is not deterministic.
pub fn assert_deterministic<F>(model: &F, reference: &F::Signal)
where
    F: Fit,
    F::Params: Debug + PartialEq,
{
    let first = recover(model, reference);
    let second = recover(model, reference);
    assert!(first == second, "{} recovered two different plans from one reference", model.name());
}

/// Panic unless the beam finds the proven optimum on a pool small enough to enumerate.
///
/// # Panics
///
/// If the beam is too narrow to be trusted at this pool size.
pub fn assert_beam_matches_exact(terms: &Terms, beam_width: usize) {
    let pool = terms.pool();
    let exact = optimise_subset(terms, pool, 1).expect("terms nothing can satisfy prove nothing");
    let beam =
        optimise_subset(terms, 0, beam_width).expect("terms nothing can satisfy prove nothing");
    assert!(
        (exact.cost() - beam.cost()).abs() < 1e-9,
        "beam scored {} against a proven optimum of {}",
        beam.cost(),
        exact.cost()
    );
}

/// Panic unless applying an identity plan leaves the signal byte for byte unchanged.
///
/// The rule an engine breaks first: a stage that always renders, then trims back toward the input,
/// cannot leave a passage it has no evidence about alone.
///
/// # Panics
///
/// If the model changes a signal it was given nothing to do to.
pub fn assert_identity_plan_changes_nothing<M>(model: &M, signal: &M::Signal)
where
    M: Model,
    M::Signal: Segmented + PartialEq + Debug,
{
    let applied = model.apply_plan(signal, &Plan::identity());
    assert!(
        &applied == signal,
        "{} changed a signal under an identity plan, giving {applied:?}",
        model.name()
    );
}

/// Panic unless a scale still outranks a rule over the worst subject the engine sees.
///
/// [`Scale::over`] keeps this promise for the numbers it was given. This checks the numbers were
/// the true ones: pass the longest subject the engine accepts and the most any single step of it
/// was measured to stray. A friction added later, or an input longer than the one the scale was
/// sized for, fails here rather than quietly turning a run of unusual readings into a reported
/// fault.
///
/// # Panics
///
/// If a subject made entirely of worst steps costs as much as one broken rule.
pub fn assert_friction_never_reaches_a_rule(scale: Scale, steps: usize, worst_step: Cost) {
    assert!(worst_step.is_clean(), "the worst step must be friction alone, not a broken rule");
    let whole: f64 = (0..steps).map(|_| scale.price(worst_step)).sum();
    assert!(
        whole < scale.price(Cost::breach(0)),
        "{steps} steps of {worst_step} come to {whole}, where one broken rule costs {}",
        scale.breach()
    );
}

/// Panic unless a reported margin is the error the answer survives.
///
/// Checks every corner of the box the margin claims, then one step past it. A margin that is
/// decoration fails here.
///
/// # Panics
///
/// If a point inside the margin breaks a row, or if nothing outside it does.
pub fn assert_margin_holds(problem: &Problem) {
    let Feasible::Region { point, margin } = problem.solve() else {
        panic!("the problem has no answer, so there is no margin to check");
    };
    let radius = margin.get();
    assert!(radius.is_finite(), "an unbounded margin means a missing constraint, not a safe one");
    if radius <= 0.0 {
        return;
    }
    assert!(
        point.len() < 20,
        "checking every corner of {} variables is not affordable",
        point.len()
    );

    let moved = |corner: usize, scale: f64| -> Vec<f64> {
        point
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let sign = if corner >> index & 1 == 1 { 1.0 } else { -1.0 };
                value + sign * radius * scale
            })
            .collect()
    };
    let holds = |at: &[f64]| problem.rows().iter().all(|row| row.violation(at) <= 1e-9);

    let corners = 1_usize << point.len();
    let mut broke_outside = false;
    for corner in 0..corners {
        assert!(holds(&moved(corner, 0.99)), "a point inside the margin breaks a row");
        broke_outside |= !holds(&moved(corner, 1.01));
    }
    assert!(
        broke_outside,
        "nothing outside the margin breaks, so the margin understates the region"
    );
}

#[cfg(test)]
mod tests {
    use fitkit_core::{Evidence, Span};
    use fitkit_fit::{Fit, Model};

    use fitkit_core::Plan;
    use fitkit_feasible::{Problem, Row, Sense};

    use super::{
        assert_beam_matches_exact, assert_deterministic, assert_friction_never_reaches_a_rule,
        assert_identity_plan_changes_nothing, assert_identity_without_evidence,
        assert_margin_holds, assert_untrusted_spans_stay_silent, forbid_derivations_from,
        forbid_symbols,
    };
    use super::{Cost, Scale, Terms};

    #[test]
    fn a_scale_sized_for_the_real_worst_step_holds() {
        assert_friction_never_reaches_a_rule(Scale::over(200, 1.0), 200, Cost::friction(1.0));
    }

    #[test]
    #[should_panic(expected = "one broken rule costs")]
    fn a_scale_sized_for_a_shorter_subject_than_it_sees_is_caught() {
        assert_friction_never_reaches_a_rule(Scale::over(10, 1.0), 200, Cost::friction(1.0));
    }

    struct Doubler;

    impl Model for Doubler {
        type Signal = Vec<f64>;
        type Params = f64;
        fn name(&self) -> &'static str {
            "doubler"
        }
        fn candidates(&self) -> Vec<f64> {
            vec![1.0, 2.0]
        }
        fn render(&self, input: &Vec<f64>, params: &f64) -> Vec<f64> {
            input.iter().map(|value| value * params).collect()
        }
    }

    fn bounded() -> Problem {
        let mut problem = Problem::new(2);
        problem.bound(0, 0.0, 5.0);
        problem.bound(1, 0.0, 5.0);
        problem.row(Row::new(vec![1.0, 1.0], Sense::Le, 6.0, "total"));
        problem.row(Row::new(vec![1.0, 1.0], Sense::Ge, 2.0, "enough"));
        problem
    }

    struct Silent;

    impl Model for Silent {
        type Signal = ();
        type Params = u8;
        fn name(&self) -> &'static str {
            "silent"
        }
        fn candidates(&self) -> Vec<u8> {
            vec![0, 1]
        }
        fn render(&self, _input: &(), _params: &u8) {}
    }

    impl Fit for Silent {
        type Evidence = u8;
        fn evidence(&self, _reference: &()) -> Vec<Evidence<u8>> {
            Vec::new()
        }
        fn emission(&self, evidence: &u8, params: &u8) -> f64 {
            f64::from(u8::from(evidence != params))
        }
    }

    /// A pairwise model whose optimum is an interior subset, so the beam has something to miss.
    fn terms(pool: usize) -> Terms {
        let mut built = Terms::over(pool).expect("a pool to choose from");
        for item in 0..pool {
            let span = Span::new(item, item + 1);
            built = built
                .worth(item, Evidence::certain(span, 1.0 - 0.2 * (item % 4) as f64))
                .expect("a finite weight over a real span");
        }
        for a in 0..pool {
            for b in a + 1..pool {
                let value = ((a * 7 + b * 5) % 11) as f64 / 11.0 - 0.6;
                built = built
                    .together(a, b, Evidence::certain(Span::new(a, b + 1), value))
                    .expect("a finite weight over a real span");
            }
        }
        built
    }

    #[test]
    fn the_guards_pass_a_model_that_abstains() {
        assert_identity_without_evidence(&Silent, &());
        assert_untrusted_spans_stay_silent(&Silent, &());
        assert_deterministic(&Silent, &());
    }

    #[test]
    fn a_model_that_only_renders_where_told_passes() {
        assert_identity_plan_changes_nothing(&Doubler, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "identity plan")]
    fn a_stage_that_always_renders_is_caught() {
        struct AlwaysOn;
        impl Model for AlwaysOn {
            type Signal = Vec<f64>;
            type Params = f64;
            fn name(&self) -> &'static str {
                "always on"
            }
            fn candidates(&self) -> Vec<f64> {
                vec![1.0]
            }
            fn render(&self, input: &Vec<f64>, params: &f64) -> Vec<f64> {
                input.iter().map(|value| value * params).collect()
            }
            fn apply_plan(&self, input: &Vec<f64>, _plan: &Plan<f64>) -> Vec<f64> {
                self.render(input, &0.5)
            }
        }
        assert_identity_plan_changes_nothing(&AlwaysOn, &vec![1.0]);
    }

    #[test]
    fn a_real_margin_survives_its_own_corners() {
        assert_margin_holds(&bounded());
    }

    #[test]
    #[should_panic(expected = "missing constraint")]
    fn an_unbounded_margin_is_caught() {
        let mut problem = Problem::new(1);
        problem.row(Row::new(vec![1.0], Sense::Ge, 1.0, "at least one"));
        assert_margin_holds(&problem);
    }

    #[test]
    fn a_wide_beam_matches_exact() {
        assert_beam_matches_exact(&terms(12), 256);
    }

    #[test]
    #[should_panic(expected = "reaches lookup_by_name")]
    fn a_banned_symbol_is_caught() {
        forbid_symbols(&[("solver.rs", "let x = lookup_by_name(food);")], &["lookup_by_name"]);
    }

    #[test]
    #[should_panic(expected = "derives Requirement")]
    fn a_signature_that_derives_an_answer_from_a_name_is_caught() {
        forbid_derivations_from(
            &[("intent.rs", "fn wanted(dish: &str) -> Vec<Requirement> { todo!() }")],
            "Requirement",
            &["&str", "String", "name:"],
            &[],
        );
    }

    #[test]
    fn a_signature_broken_across_lines_is_still_read_whole() {
        let source = "fn wanted(\n    dish: &str,\n) -> Vec<Requirement> {\n    todo!()\n}";
        let caught = std::panic::catch_unwind(|| {
            forbid_derivations_from(&[("intent.rs", source)], "Requirement", &["&str"], &[]);
        });
        assert!(caught.is_err(), "a formatter must not be able to hide a signature");
    }

    #[test]
    fn prose_about_the_banned_shape_is_not_evidence_against_it() {
        // The comment names the exact pattern; only executable text counts.
        let source =
            "// fn wanted(dish: &str) -> Vec<Requirement>\nfn ok(m: &Measured) -> f64 { 0.0 }";
        forbid_derivations_from(&[("notes.rs", source)], "Requirement", &["&str"], &[]);
    }

    #[test]
    fn an_exempt_parameter_is_a_passenger_rather_than_an_offender() {
        // The label rides along with the measurement that actually decides.
        forbid_derivations_from(
            &[(
                "blend.rs",
                "fn wanted(rows: &[(String, Measured)]) -> Vec<Requirement> { todo!() }",
            )],
            "Requirement",
            &["String"],
            &["(String, Measured)"],
        );
    }

    #[test]
    fn a_function_that_returns_something_else_is_not_examined() {
        forbid_derivations_from(
            &[("report.rs", "fn title(name: &str) -> String { name.into() }")],
            "Requirement",
            &["&str"],
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "invented")]
    fn a_model_that_invents_controls_is_caught() {
        struct Inventor;
        impl Model for Inventor {
            type Signal = ();
            type Params = u8;
            fn name(&self) -> &'static str {
                "inventor"
            }
            fn candidates(&self) -> Vec<u8> {
                // Two, so the decode has something to weigh and answers. A model offering one is
                // refused before it can invent anything, which is a different guard.
                vec![9, 8]
            }
            fn render(&self, _input: &(), _params: &u8) {}
        }
        impl Fit for Inventor {
            type Evidence = u8;
            fn evidence(&self, _reference: &()) -> Vec<Evidence<u8>> {
                vec![Evidence::certain(Span::new(0, 1), 0)]
            }
            fn emission(&self, _evidence: &u8, _params: &u8) -> f64 {
                0.0
            }
        }
        assert_identity_without_evidence(&Inventor, &());
    }
}
