//! Timing for the feasibility solver. Dependency free, so it runs anywhere the library does.
//!
//! `cargo bench -p fitkit-feasible`

use std::hint::black_box;
use std::time::Instant;

use fitkit_feasible::{Problem, Row, Sense};

fn problem(vars: usize, rows: usize) -> Problem {
    let mut problem = Problem::new(vars);
    for index in 0..vars {
        problem.bound(index, 0.0, 10.0);
    }
    for row in 0..rows {
        let coefficients =
            (0..vars).map(|index| ((row * 7 + index * 3) % 5) as f64 - 2.0).collect();
        problem.row(Row::new(coefficients, Sense::Le, 12.0, "bench"));
    }
    problem
}

fn main() {
    for (vars, rows, laps) in [(8, 8, 500), (24, 24, 100), (48, 48, 20), (64, 96, 10)] {
        let built = problem(vars, rows);
        bench(&format!("solve {vars} vars x {rows} rows"), laps, || {
            black_box(built.solve().margin())
        });
    }
}

fn bench<T>(name: &str, rounds: u32, mut run: impl FnMut() -> T) {
    run();
    let start = Instant::now();
    for _ in 0..rounds {
        run();
    }
    let each = start.elapsed() / rounds;
    println!("{name:<36} {each:>12.3?}");
}
