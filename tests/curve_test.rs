//! curve test
//!
//! a curve is angle accumulation — a chain of turns and runs, [magnitude, angle]
//! composed. the path (0,1) → (1,1) → (2,0) is a run, a turn, a run: the "descent" IS
//! the rotation, nothing dropped a coordinate.
//!
//! projection is TERMINAL, not generative. you build the curve by accumulating the
//! heading; x and y are two of the infinitely many shadows you could cast at the end,
//! when a scalar is demanded. coordinate math is projection-first (you live in x and y
//! and read the angle off with atan2); geonum is angle-first, and the axes are just two
//! questions you ask the finished curve.
//!
//! - a polyline traced as turn and run, no coordinate named until the readout
//! - x and y are terminal shadows of one run; its own axis recovers it whole, its
//!   perpendicular reads zero — the axes are arbitrary
//! - a curve bends because the heading accumulates turning; constant turns close a
//!   polygon once the total turning completes 2π
//! - on the unit circle the accumulated arc length equals the angle swept — the radian
//!   identity — so an arc is [θ, θ]; the half-diameter is the straight spoke, length 1
//!
//! run: cargo test --test curve_test

use geonum::{Angle, Geonum};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// the path is turn and run — the descent is a rotation, not a dropped coordinate
// ---------------------------------------------------------------------------
#[test]
fn it_traces_a_polyline_as_turn_and_run() {
    // (0,1) → (1,1) → (2,0), built without ever naming an x or a y
    let mut pos = Geonum::new(1.0, 1.0, 2.0); // start (0,1) = [1, π/2]
    let mut heading = Angle::new(0.0, 1.0); // facing +x
    assert_eq!(heading.grade(), 0, "starts pointing along +x");

    // run 1 along the heading → (1,1)
    pos = pos + Geonum::new_with_angle(1.0, heading);

    // at (1,1) the path turns down by π/4 — this rotation IS the descent
    heading = heading + Angle::new(7.0, 4.0); // −π/4 ≡ 7π/4
    assert_eq!(heading.grade(), 3, "now pointing down-right (quadrant IV)");

    // run √2 along the new heading → (2,0)
    pos = pos + Geonum::new_with_angle(2.0_f64.sqrt(), heading);

    // ONLY NOW, at the end, cast shadows onto the axes to read coordinates — and they
    // confirm (2,0). the construction above never projected anything
    let x = pos.mag * pos.angle.project(Angle::new(0.0, 1.0)); // onto +x
    let y = pos.mag * pos.angle.project(Angle::new(1.0, 2.0)); // onto +y (π/2)
    assert!((x - 2.0).abs() < 1e-9, "x = 2");
    assert!(
        y.abs() < 1e-9,
        "y = 0 — the path descended to the axis by turning"
    );
}

// ---------------------------------------------------------------------------
// x and y are two terminal shadows of one run — the axes are arbitrary
// ---------------------------------------------------------------------------
#[test]
fn it_casts_coordinates_as_terminal_shadows() {
    // the descending run is one object: [√2, −π/4]. x and y are just two of the shadows
    // it casts, no more privileged than any other angle
    let run = Geonum::new(2.0_f64.sqrt(), 7.0, 4.0); // [√2, 7π/4] = −π/4
    let shadow = |axis: Angle| run.mag * run.angle.project(axis);

    assert!(
        (shadow(Angle::new(0.0, 1.0)) - 1.0).abs() < 1e-9,
        "x-shadow = +1"
    );
    assert!(
        (shadow(Angle::new(1.0, 2.0)) + 1.0).abs() < 1e-9,
        "y-shadow = −1 (the descent)"
    );

    // onto its OWN direction the run casts its full length — no shadow lost
    assert!(
        (shadow(run.angle) - run.mag).abs() < 1e-9,
        "onto itself: the whole √2"
    );
    // onto the perpendicular, zero — the run has length but no width
    assert!(
        shadow(run.angle + Angle::new(1.0, 2.0)).abs() < 1e-9,
        "onto its perpendicular: 0"
    );
}

// ---------------------------------------------------------------------------
// a curve bends because the heading accumulates turning — constant turns close
// ---------------------------------------------------------------------------
#[test]
fn it_curves_by_accumulating_turns() {
    // a regular hexagon: 6 runs of length 1, turning the heading by 2π/6 between each.
    // the accumulated turning IS the curvature, and once it completes a full 2π the path
    // closes — all from turn and run, no projection
    let n = 6;
    let mut pos = Geonum::new(0.0, 0.0, 1.0); // start at the origin
    let mut heading = Angle::new(0.0, 1.0);

    for _ in 0..n {
        pos = pos + Geonum::new_with_angle(1.0, heading); // run
        heading = heading + Angle::new(2.0, n as f64); // turn by 2π/n
    }

    assert_eq!(
        heading.grade(),
        0,
        "the heading came full circle: total turning = 2π"
    );
    assert!(
        pos.near_mag(0.0),
        "the path closed — back to the start, the curve drawn by turning alone"
    );
}

// ---------------------------------------------------------------------------
// the unit circle: arc length IS the turning, read off the blade in one step —
// not a sum of chords. a straight line's length is its magnitude instead
// ---------------------------------------------------------------------------
#[test]
fn it_reads_the_unit_circle_arc_as_the_turning() {
    // a straight line never bends: zero turning, and its length is its MAGNITUDE. a
    // unit-circle ARC bends, and each radian of heading sweeps one unit of arc
    // (ds = r·dθ = dθ), so its length is its TURNING. the turning is read off the blade in
    // one step — winding-kept, so a full turn reads 2π, not the 0 that grade_angle drops.
    // no chords are summed; the loop the radian identity collapses. the straight chord
    // ACROSS is a different magnitude — one geonum subtraction (0 when the full turn closes)
    let start = Geonum::new(1.0, 0.0, 1.0); // [1, 0] = (1, 0), a unit spoke

    for &(num, div, arc, chord) in &[
        (1.0, 2.0, PI / 2.0, 2.0_f64.sqrt()), // quarter: arc π/2, chord √2
        (1.0, 1.0, PI, 2.0),                  // semicircle: arc π, chord 2 (the diameter)
        (2.0, 1.0, 2.0 * PI, 0.0),            // full turn: arc 2π, chord 0 (closed)
    ] {
        let end = start.rotate(Angle::new(num, div)); // turn the spoke through Θ

        // arc length = the turning, read off the blade — a boundary read, no walk
        let turning = end.angle.blade() as f64 * (PI / 2.0);
        assert!(
            (turning - arc).abs() < 1e-9,
            "arc length = the turning Θ, off the blade"
        );

        // the chord across is the displacement — different from the arc
        assert!(
            ((end - start).mag - chord).abs() < 1e-9,
            "chord across ≠ arc — a straight subtraction"
        );
    }
}
