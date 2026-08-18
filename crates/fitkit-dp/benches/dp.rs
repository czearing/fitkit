//! Timing for both dynamic programs. Dependency free, so it runs anywhere the library does.
//!
//! `cargo bench -p fitkit-dp`

use std::hint::black_box;
use std::time::Instant;

use fitkit_dp::{decode_path, optimise_subset};

fn main() {
    bench("decode_path 4096x16", 50, || {
        let path = decode_path(
            4096,
            16,
            1.0,
            |step, state| ((step * 31 + state * 17) % 23) as f64,
            |from, to| (from as f64 - to as f64).abs(),
        );
        black_box(path.expect("the grid offers rival states").len())
    });

    bench("decode_path 512x64", 50, || {
        let path = decode_path(
            512,
            64,
            1.0,
            |step, state| ((step * 31 + state * 17) % 23) as f64,
            |from, to| (from as f64 - to as f64).abs(),
        );
        black_box(path.expect("the grid offers rival states").len())
    });

    bench("optimise_subset exact 20", 5, || {
        black_box(
            optimise_subset(20, 20, 1, |members| f64::from(members.count_ones()))
                .expect("the pool offers subsets")
                .cost(),
        )
    });

    bench("optimise_subset beam 64 width 128", 20, || {
        black_box(
            optimise_subset(64, 0, 128, |members| f64::from(members.count_ones()))
                .expect("the pool offers subsets")
                .cost(),
        )
    });
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
