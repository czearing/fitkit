# Feasibility

Some questions have no sequence to decode and no subset to choose. They ask what values satisfy
every stated requirement at once, and how wrong those values can be before one fails.

`fitkit-feasible` answers both.

```rust
use fitkit::prelude::*;

let mut problem = Problem::new(2);
problem.bound(0, 5.0, 60.0);
problem.bound(1, 20.0, 90.0);
problem.row(Row::new(vec![3.0, -1.0], Sense::Ge, 0.0, "cohesion threshold"));
problem.row(Row::new(vec![1.0, 1.0], Sense::Le, 100.0, "lifting limit"));

match problem.solve() {
    Feasible::Region { point, margin } => println!("{point:?} survives {margin}"),
    Feasible::Empty { binding, .. } => println!("blocked by {binding:?}"),
}
```

## The answer is a region, not a number

A single optimum hides the question a user actually has, which is how much room there is. A
`Feasible::Region` reports a point and the [margin](#margin) around it. A `Feasible::Empty` reports
which laws were left broken, so a refusal can be read rather than guessed at.

## Margin

The margin is the largest error every variable can carry **at once** while every row still holds.
It is a distance in the units of the variables, so it can be compared against the accuracy of the
instrument that will realise the answer. If a scale is good to a gram and the margin is half a
gram, the answer is not usable, however optimal it is.

Two rules follow from that definition, and both are deliberate.

**A variable in an equality row has no tolerance.** That is what an equality means. If a quantity
has real room, state it as a pair of inequalities and the margin will find it.

**An unbounded margin is a missing constraint, not a safe result.** `assert_margin_holds` fails on
one for that reason.

The claim is checkable, and is checked: `assert_margin_holds` walks every corner of the box the
margin describes, requires all of them to satisfy every row, then requires at least one point just
outside to fail. A margin that is decoration does not survive that test.

## Requirements

A `Row` is linear algebra. A `Requirement` is the thing worth writing down: a statement about how
the finished object must behave, carrying the citation that makes it testable.

```rust
impl Requirement for HoldsTogether {
    fn citation(&self) -> Citation { /* the published source */ }
    fn rows(&self, vars: usize) -> Answer<Vec<Row>> { /* the linear form, or a refusal */ }
}
```

Requiring a citation is the point. See [LAWS](LAWS.md): a requirement may state how the finished
object must behave and may never state a quantity the solver exists to derive. A requirement that
names a ratio or a mass is where a hardcoded answer hides once it has been deleted from the code.

`rows` returns an `Answer`, so a requirement that cannot be stated over the given variables refuses
instead of quietly contributing nothing.

## How it solves

A dense two-phase simplex over `Ax {<=,=,>=} b`, `x >= 0`, with no dependencies. Variables are
shifted by their lower bounds so every one starts non-negative, and one extra variable carries the
half width of a box that is inflated inside the region until it touches a row. Maximising that
half width is the margin.

Bland's rule is used for pivot selection. It takes more pivots than steepest descent but is the
only common rule with a termination proof, and locked variables make these problems degenerate by
construction, which is exactly where other rules can cycle.

The box uses the sum of absolute coefficients rather than a Euclidean norm. That makes the margin
a per-variable tolerance rather than a radius in an abstract space, and it needs no square root, so
the crate builds for `no_std` targets.

## Cost

Measured on an M1 Pro, release profile, `cargo bench -p fitkit-feasible`.

| Problem | Time |
| --- | --- |
| 8 variables, 8 rows | 6.5 us |
| 24 variables, 24 rows | 59 us |
| 48 variables, 48 rows | 476 us |
| 64 variables, 96 rows | 1.28 ms |

Microseconds at small sizes is what makes a feasibility check affordable inside a search loop,
which is the case that rules out linking, installing, or spawning an external solver.
