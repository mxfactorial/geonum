// the rotation number is a blade rate
//
// the rotation number of a circle map — dynamical systems' basic invariant —
// is conventionally defined through a LIFT: extend the map from the circle to
// the real line so the total rotation can be tracked, because the circle
// coordinate wraps it away. the lift is the unwrap ceremony with a formal
// name. geonum's heading never wrapped: the Angle is the lift, and the
// rotation number is blade accumulation per iteration, read off storage
//
//   - a bare rational rotation closes its orbit and lands its winding exactly
//     on the lattice; the golden rotation never closes — off every small
//     rational, the three-gap residue geonum reads as a base-angle gap
//   - inside a tongue the coupling captures the winding: the sine term parks
//     the heading where the drive cancels (ρ = 0) or paces it to exactly one
//     turn per step (ρ = 1) — mode locking is the blade rate landing rational
//   - the arnold tongue has a measurable edge: locking onsets at K = 2πδ,
//     the coupling strength that first cancels the detuning. below it the
//     winding creeps past the rational; above it the count is exact
//
// control_test shipped the linear taxonomy (scale_rotate knobs) and
// mechanics_test act V the conserved magnitude — this is the nonlinear story
//
// run: cargo test --test stability_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// one step of the sine circle map θ' = θ + ω + K·sin θ. the coupling reads the
// heading's position rationally (cos_sin), the advance adds as an Angle, and
// the winding accumulates in the blade — no lift constructed
fn circle_map_step(heading: Angle, omega: f64, coupling: f64) -> Angle {
    let (_, sin_theta) = heading.cos_sin();
    heading + Angle::new((omega + coupling * sin_theta) / PI, 1.0)
}

#[test]
fn it_reads_the_rotation_number_off_the_blade_rate() {
    // bare rotation at 3/8 of a turn: eight steps close the orbit and store
    // exactly three turns — the rotation number 3/8 read as 12 blades / 32
    let mut rational = Angle::new(0.0, 1.0);
    for _ in 0..8 {
        rational = circle_map_step(rational, 2.0 * PI * 3.0 / 8.0, 0.0);
    }
    assert!(
        rational.near(&Angle::new(6.0, 1.0)),
        "eight steps of 3/8 turn: three turns exactly — blade 12, orbit closed"
    );
    assert_eq!(
        rational.base_angle(),
        Angle::new(0.0, 1.0),
        "the rational orbit returns to its start"
    );

    // the golden rotation: the blade rate converges on the irrational
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0; // 0.618... of a turn
    let steps = 1000;
    let mut heading = Angle::new(0.0, 1.0);
    for _ in 0..steps {
        heading = circle_map_step(heading, 2.0 * PI * golden, 0.0);
    }
    let rate = heading.blade() as f64 / (4.0 * steps as f64);
    assert!(
        (rate - golden).abs() < 1e-3,
        "the blade rate reads the golden rotation number: {rate:.6}"
    );

    // and it never closes: no q ≤ 8 steps land back on the start — the golden
    // ratio sits a measurable base-angle gap off every small rational
    for q in 1..=8usize {
        let advance = Angle::new(2.0 * q as f64 * golden, 1.0);
        let position = advance.grade_angle();
        let gap = position.min(2.0 * PI - position);
        assert!(
            gap > 0.3,
            "q = {q}: the orbit misses closure by {gap:.3} rad"
        );
    }
}

#[test]
fn it_arrests_the_winding_inside_the_zero_tongue() {
    // drive ω = 0.5 against coupling K = 0.9 > ω: the sine term can fully
    // cancel the drive, so the heading walks forward until it parks at the
    // fixed point sin θ* = −ω/K and the winding stops — rotation number 0
    let (omega, coupling) = (0.5, 0.9);
    let mut heading = Angle::new(0.0, 1.0);
    for _ in 0..2000 {
        heading = circle_map_step(heading, omega, coupling);
    }

    assert!(
        heading.blade() < 4,
        "2000 iterations never complete one turn — the winding is arrested"
    );
    let (_, sin_theta) = heading.cos_sin();
    assert!(
        (sin_theta + omega / coupling).abs() < 1e-6,
        "the heading parks where the drive cancels: sin θ* = −ω/K"
    );

    // the same drive uncoupled winds freely — the arrest is the tongue's work
    let mut bare = Angle::new(0.0, 1.0);
    for _ in 0..2000 {
        bare = circle_map_step(bare, omega, 0.0);
    }
    assert!(
        bare.blade() > 600,
        "uncoupled, the same drive stores {} blades",
        bare.blade()
    );
}

#[test]
fn it_locks_the_mode_onto_the_rational() {
    // detune the drive 2% past one turn per step. inside the 1:1 tongue the
    // coupling absorbs the detuning: after the transient the heading advances
    // EXACTLY one turn per iteration — four blades a step, the mode locked
    // onto the rational, the base angle frozen
    let omega = 2.0 * PI * 1.02;
    let coupling = 0.3; // above the tongue edge 2π·0.02

    let mut heading = Angle::new(0.0, 1.0);
    for _ in 0..1000 {
        heading = circle_map_step(heading, omega, coupling);
    }

    let position_before = heading.grade_angle();
    let blade_before = heading.blade();
    for _ in 0..200 {
        heading = circle_map_step(heading, omega, coupling);
    }

    assert_eq!(
        heading.blade() - blade_before,
        800,
        "200 locked steps store exactly 200 turns — ρ = 1, rational to the blade"
    );
    assert!(
        (heading.grade_angle() - position_before).abs() < 1e-9,
        "the base angle is frozen — the orbit rides the fixed point"
    );
}

#[test]
fn it_measures_the_tongue_edge_in_coupling_strength() {
    // the 1:1 tongue's edge sits at K = 2πδ — the coupling that first cancels
    // the 2% detuning. below it the winding creeps past one turn per step;
    // above it the count locks exact. the arnold tongue, measured as a
    // threshold in K with the blade as the detector
    let omega = 2.0 * PI * 1.02; // detuning δ = 0.02 turns → edge at K ≈ 0.126

    let window_climb = |coupling: f64| -> usize {
        let mut heading = Angle::new(0.0, 1.0);
        for _ in 0..1000 {
            heading = circle_map_step(heading, omega, coupling);
        }
        let before = heading.blade();
        for _ in 0..200 {
            heading = circle_map_step(heading, omega, coupling);
        }
        heading.blade() - before
    };

    assert!(
        window_climb(0.10) > 800,
        "below the edge the winding creeps past the rational"
    );
    assert_eq!(
        window_climb(0.20),
        800,
        "above the edge the count is exact — the tongue captured the detuning"
    );
}
