//! The other half of an engine: no sequence to decode, only requirements to satisfy.
//!
//! `cargo run -p fitkit --example blend`

use fitkit::prelude::*;

/// Grams of binder and grams of filler.
const BINDER: usize = 0;
const FILLER: usize = 1;

/// Every requirement states how the finished object must behave, never a quantity to derive.
struct HoldsTogether;
struct StaysLight;

impl Requirement for HoldsTogether {
    fn citation(&self) -> Citation {
        Citation { key: "Example1998", source: "Example, J. Materials 12, 44 (1998)" }
    }

    fn rows(&self, vars: usize) -> Answer<Vec<Row>> {
        if vars <= FILLER {
            return Err(Refusal::incoherent("a blend needs a binder and a filler"));
        }
        // Cohesion fails below one part binder to three parts filler.
        let mut coefficients = vec![0.0; vars];
        coefficients[BINDER] = 3.0;
        coefficients[FILLER] = -1.0;
        Ok(vec![Row::new(coefficients, Sense::Ge, 0.0, "Example1998 cohesion threshold")])
    }
}

impl Requirement for StaysLight {
    fn citation(&self) -> Citation {
        Citation { key: "Example2004", source: "Example, J. Handling 7, 210 (2004)" }
    }

    fn rows(&self, vars: usize) -> Answer<Vec<Row>> {
        if vars <= FILLER {
            return Err(Refusal::incoherent("a blend needs a binder and a filler"));
        }
        Ok(vec![Row::new(vec![1.0; vars], Sense::Le, 100.0, "Example2004 lifting limit")])
    }
}

fn blend() -> Answer<Problem> {
    let mut problem = Problem::new(2);
    problem.bound(BINDER, 5.0, 60.0);
    problem.bound(FILLER, 20.0, 90.0);
    problem.require(&HoldsTogether)?;
    problem.require(&StaysLight)?;
    Ok(problem)
}

fn main() {
    let problem = blend().expect("both requirements can be stated over two components");

    match problem.solve() {
        Feasible::Region { point, margin } => {
            println!("binder {:.1} g, filler {:.1} g", point[BINDER], point[FILLER]);
            println!("every quantity survives being {margin} g out");
            assert!(margin.survives(5.0), "a kitchen scale is good to a gram");
        }
        Feasible::Empty { binding, .. } => {
            println!("no blend satisfies every requirement");
            for law in binding {
                println!("  blocked by {law}");
            }
        }
    }

    // Asking for something no blend can do is answered, not guessed at.
    let mut impossible = blend().expect("stateable");
    impossible.row(Row::new(vec![1.0, 1.0], Sense::Ge, 500.0, "must weigh half a kilo"));
    let Feasible::Empty { binding, .. } = impossible.solve() else {
        panic!("500 g cannot also weigh under 100 g")
    };
    println!("\nrefused, and the reason is named: {}", binding.join(", "));
}
