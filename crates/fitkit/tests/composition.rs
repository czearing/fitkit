//! What a consumer that assembles a report out of evidence is forced into when it migrates here.
//!
//! The engine this was written for had a generator that claimed to be a dynamic program and was
//! two dozen `format!` frames behind a greedy fold. Every gate it had passed: it built, it linted,
//! it was covered by tests, and the lines it produced were correct statements about real counts.
//! Nothing distinguished it from a search except reading it.
//!
//! These tests are the framework's side of that migration. They do not inspect a consumer's source
//! and they do not test for the presence of any function; they exercise the only route the public
//! API leaves open and show what it does and does not permit.

use fitkit::core::{Confidence, Evidence, Span};
use fitkit::prelude::*;

/// A measured fact about the subject, as a survey would yield it.
struct Finding {
    /// Where in the source the count came from.
    span: Span,
    /// What the count is worth saying.
    worth: f64,
    /// How far the measurement is trusted.
    trust: Confidence,
    /// The words this finding contributes when it is chosen.
    says: &'static str,
}

fn findings() -> Vec<Finding> {
    vec![
        Finding {
            span: Span::new(0, 40),
            worth: 9.0,
            trust: Confidence::FULL,
            says: "declares 72 public items",
        },
        Finding {
            span: Span::new(40, 55),
            worth: 6.0,
            trust: Confidence::new(0.5),
            says: "spans 19 modules",
        },
        Finding {
            span: Span::new(55, 60),
            worth: 1.0,
            trust: Confidence::new(0.2),
            says: "occupies 14 files",
        },
        Finding {
            span: Span::new(60, 61),
            worth: -4.0,
            trust: Confidence::FULL,
            says: "holds at least one file",
        },
    ]
}

fn terms(findings: &[Finding]) -> Terms {
    let mut built = Terms::over(findings.len()).expect("a pool to choose from");
    for (item, finding) in findings.iter().enumerate() {
        built = built
            .worth(item, Evidence::new(finding.span, finding.trust, finding.worth))
            .expect("a weight the survey measured");
    }
    // Two findings drawn from overlapping regions repeat each other, so saying both is worth less
    // than saying either. This is the claim a greedy fold cannot make and the reason to search.
    for a in 0..findings.len() {
        for b in a + 1..findings.len() {
            if findings[a].span.overlaps(findings[b].span) {
                built = built
                    .together(a, b, Evidence::certain(findings[a].span, -3.0))
                    .expect("a weight over the region the two share");
            }
        }
    }
    built
}

/// A composer gets one part per chosen finding, and cannot choose how many parts there are.
///
/// This is what closes the frame-per-section pattern. The old generator had a builder per section
/// that was obliged to return text whether or not the evidence supported any, and the cheapest
/// text that satisfies an obligation is a literal. Here the count of parts is the size of the
/// subset the search settled on, so there is no section-shaped hole waiting to be filled.
#[test]
fn the_search_decides_how_much_is_said_and_the_composer_cannot_overrule_it() {
    let findings = findings();
    let model = terms(&findings);

    let composed = optimise_subset_parts(&model, findings.len(), 1, |item| findings[item].says)
        .expect("findings to choose between");

    let chosen = optimise_subset(&model, findings.len(), 1).expect("findings to choose between");
    assert_eq!(
        composed.get().len(),
        chosen.get().len(),
        "the composer produced exactly what the search selected"
    );
    assert!(
        !composed.get().contains(&"holds at least one file"),
        "a finding that costs more than it tells was left out"
    );
    assert!(
        composed.get().contains(&"declares 72 public items"),
        "the best supported finding was kept"
    );
}

/// The composer is never shown the selection, so it cannot answer as though it were.
///
/// `optimise_subset_as` hands over the whole result and a builder holding the answer may disregard
/// it. `optimise_subset_parts` asks about one item at a time. The two are compared here so the
/// difference is a fact of the run rather than an assertion in a doc comment.
#[test]
fn a_part_is_built_from_its_own_item_and_nothing_else() {
    let findings = findings();
    let model = terms(&findings);

    let seen = std::cell::RefCell::new(Vec::new());
    let composed = optimise_subset_parts(&model, findings.len(), 1, |item| {
        seen.borrow_mut().push(item);
        findings[item].says
    })
    .expect("findings to choose between");

    let asked = seen.into_inner();
    let chosen = optimise_subset(&model, findings.len(), 1).expect("findings to choose between");
    let selected: Vec<usize> = chosen.get().indices().collect();
    assert_eq!(asked, selected, "the builder was asked about the chosen items, one at a time");
    assert_eq!(composed.get().len(), selected.len());
}

/// What was said can be traced to the regions that argued for it, and to the trust behind them.
#[test]
fn a_composed_passage_cites_its_evidence_and_reports_what_it_rests_on() {
    let findings = findings();
    let model = terms(&findings);
    let chosen = optimise_subset(&model, findings.len(), 1).expect("findings to choose between");
    let members = chosen.get().members();

    let cited = model.support(members);
    assert!(!cited.is_empty(), "the passage names where it came from");
    for span in &cited {
        assert!(!span.is_empty(), "every citation points at a real region");
    }
    assert!(!model.trust(members).is_zero(), "and reports the trust it rests on");
    assert!(chosen.trace().decided(), "rivals were on the table when it was decided");
    assert!(chosen.get().is_proven(), "and the pool was small enough to prove it");
}

/// Weaker evidence argues less hard, so the same claim can win or lose on its support alone.
///
/// A template has no such behaviour: its output is the same whatever the source says.
#[test]
fn the_same_claim_wins_or_loses_on_the_trust_behind_it() {
    let span = Span::new(0, 10);
    let build = |trust: Confidence| {
        Terms::over(2)
            .expect("a pool")
            .worth(0, Evidence::new(span, trust, 4.0))
            .expect("a measured weight")
            .worth(1, Evidence::certain(span, 1.0))
            .expect("a measured weight")
            .at_most(1)
            .expect("room for one")
    };

    let trusted = build(Confidence::FULL);
    let doubted = build(Confidence::new(0.1));

    let kept = optimise_subset(&trusted, 2, 1).expect("two findings");
    let dropped = optimise_subset(&doubted, 2, 1).expect("two findings");

    assert_eq!(kept.get().members(), 0b01, "well supported, so it was said");
    assert_eq!(dropped.get().members(), 0b10, "barely supported, so the weaker rival won");
}

/// A composer with nothing worth saying has an outcome that is not an empty document.
///
/// The old generator had no way to say "the evidence supports nothing here", so a section with no
/// findings still produced a heading and a line under it. A refusal is what removes the pressure.
#[test]
fn a_pool_that_supports_nothing_is_refused_rather_than_composed_into_silence() {
    let nothing = Terms::over(2)
        .expect("a pool")
        .worth(0, Evidence::certain(Span::new(0, 1), -1.0))
        .expect("a measured weight")
        .at_least(1)
        .expect("something must be said")
        .at_most(0);
    assert!(nothing.is_err(), "a floor above a ceiling is a contradiction, not an empty answer");

    let impossible = Terms::over(2)
        .expect("a pool")
        .require(0, Span::new(0, 1))
        .expect("a cited requirement")
        .forbid(0, Span::new(0, 1));
    assert!(impossible.is_err(), "and so is requiring and forbidding the same finding");
}

/// An ordering decode builds one part per position from the state chosen there.
#[test]
fn ordering_is_decoded_and_the_parts_follow_the_decode() {
    let words = ["The", "crate", "declares", "72", "items"];
    let ordered = decode_path_parts(
        words.len(),
        words.len(),
        1.0,
        |step, state| if step == state { 0.0 } else { 4.0 },
        |from, to| if to == from + 1 { 0.0 } else { 6.0 },
        |_step, state| words[state],
    )
    .expect("five positions to fill");

    assert_eq!(ordered.get().len(), words.len(), "one part per position, decided by the search");
    assert_eq!(ordered.get().as_slice(), &words, "the decode recovered the order the model priced");
    assert!(ordered.trace().decided(), "each position had rivals to price");
}

/// The residue, pinned rather than claimed away.
///
/// One part is built per chosen item, and that is the whole of the guarantee. A part is the
/// caller's own type, so nothing stops it being a list, and a builder can return as many words per
/// item as it likes. What it cannot do is decide how many items there are, or know which ones were
/// chosen, or produce anything at all for an item the search left out.
///
/// So the framework fixes the shape of a composed output to the shape of the evidence, and leaves
/// the content of each piece to the consumer. Closing that last gap needs a type the consumer owns
/// — a piece with no public constructor — which per-crate privacy puts outside a framework's
/// reach. Recorded here so the boundary is in the repository and not only in a reply.
#[test]
fn a_part_may_be_a_list_and_the_count_of_parts_still_belongs_to_the_search() {
    let findings = findings();
    let model = terms(&findings);

    let verbose = optimise_subset_parts(&model, findings.len(), 1, |item| {
        vec![findings[item].says, "and some more words the builder chose"]
    })
    .expect("findings to choose between");

    let chosen = optimise_subset(&model, findings.len(), 1).expect("findings to choose between");
    assert_eq!(
        verbose.get().len(),
        chosen.get().len(),
        "a wordier builder still produced one part per chosen finding"
    );
    assert!(verbose.get().iter().all(|part| part.len() == 2), "each part is as long as it likes");
}
