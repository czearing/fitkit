//! A worked engine in one file: recover a thermostat setpoint from noisy readings.
//!
//! `cargo run -p fitkit --example thermostat`

use fitkit::prelude::*;

/// Readings from a sensor that drops out for part of the record.
#[derive(Clone, Debug, PartialEq)]
struct Log {
    readings: Vec<Option<f64>>,
}

/// Cutting and rejoining a log is all [`Model::apply_plan`] needs to write the answer back.
impl Segmented for Log {
    fn extent(&self) -> usize {
        self.readings.len()
    }

    fn slice(&self, span: Span) -> Log {
        Log { readings: self.readings.slice(span) }
    }

    fn splice(&mut self, span: Span, part: Log) {
        Segmented::splice(&mut self.readings, span, part.readings);
    }
}

/// The model: the room was held at one of a few whole degree setpoints at a time.
struct Thermostat;

impl Model for Thermostat {
    type Signal = Log;
    type Params = i64;

    fn name(&self) -> &'static str {
        "thermostat"
    }

    fn candidates(&self) -> Vec<i64> {
        (16..=26).collect()
    }

    fn render(&self, input: &Log, params: &i64) -> Log {
        Log { readings: input.readings.iter().map(|_| Some(*params as f64)).collect() }
    }
}

impl Fit for Thermostat {
    type Evidence = f64;

    fn evidence(&self, reference: &Log) -> Vec<Evidence<f64>> {
        reference
            .readings
            .iter()
            .enumerate()
            .map(|(i, reading)| {
                let span = Span::new(i, i + 1);
                // A dropout is not a reading of zero. It carries no information.
                match reading {
                    Some(value) => Evidence::certain(span, *value),
                    None => Evidence::new(span, Confidence::ZERO, f64::NAN),
                }
            })
            .collect()
    }

    fn emission(&self, reading: &f64, setpoint: &i64) -> f64 {
        if reading.is_nan() {
            return 0.0;
        }
        (reading - *setpoint as f64).abs()
    }

    fn transition(&self, from: &i64, to: &i64) -> f64 {
        f64::from(u8::from(from != to))
    }

    /// A change must earn four degrees of fit, so one stray reading cannot move the setpoint
    /// but a sustained change can.
    fn transition_weight(&self) -> f64 {
        4.0
    }
}

fn main() {
    let log = Log {
        readings: vec![
            Some(20.1),
            Some(19.8),
            Some(30.0), // sun on the sensor, not a new setpoint
            Some(20.2),
            Some(19.9),
            Some(20.1),
            None, // sensor dropout
            None,
            Some(23.0),
            Some(23.1),
            Some(22.9),
        ],
    };

    let plan = recover(&Thermostat, &log);
    let tolerance = margins(&Thermostat, &log);

    println!("{plan}");
    for (control, margin) in plan.controls.iter().zip(&tolerance) {
        let verdict = if control.is_silent() {
            "unchanged".to_string()
        } else {
            format!("{} C, survives {margin}", control.params)
        };
        println!("  reading {:>2}  {verdict}", control.span.start);
    }

    let held: Vec<i64> =
        plan.controls.iter().filter(|c| !c.is_silent()).map(|c| c.params).collect();
    assert_eq!(held, [20, 20, 20, 20, 20, 20, 23, 23, 23]);

    // Writing the answer back leaves the dropouts exactly as they were found.
    let corrected = Thermostat.apply_plan(&log, &plan);
    assert_eq!(corrected.readings[6], None, "a span with no evidence is not written to");
    assert_eq!(corrected.readings[2], Some(20.0), "the spike is replaced by the held setpoint");

    println!("\nthe spike at reading 2 was absorbed, the sustained step at reading 8 was kept");
}
