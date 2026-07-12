// nothing falls off the manifold
//
// game engines call quaternion.normalize() on a schedule; matrix pipelines
// re-orthonormalize with gram-schmidt every N frames; rotation integrators
// drift off SO(n) and get projected back. the drift exists because those
// representations store redundant components constrained to a manifold
// (|q| = 1, RᵀR = I), and float arithmetic leaks through the constraint one
// rounding at a time. [mag, angle] stores no constraint: the magnitude is
// data rotation never touches, the blade is an integer, and the lattice
// re-anchors rational-π chains at every boundary crossing. there is no
// renormalize() in the api because there is no manifold to fall off
//
// the foil runs the planar case; the leak — redundant constrained components
// rounding off their constraint — is dimension-generic
//
// run: cargo test --test renormalization_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

#[test]
fn it_composes_a_million_rotations_without_renormalizing() {
    // a million compositions of π/7 — the workload renormalization schedules
    // exist for
    let step = Angle::new(1.0, 7.0);
    let mut chain = Geonum::new(1.0, 0.0, 1.0);
    for _ in 0..1_000_000 {
        chain = chain.rotate(step);
    }

    // the angle lands the exact lattice target 10^6·π/7 — blade 285714 plus
    // π/7 of remainder. every 7th step lands a π/2 boundary exactly, and the
    // boundary snap re-anchors the chain's float error to the lattice
    let exact = Angle::new_with_blade(285_714, 1.0, 7.0);
    assert!(
        chain.angle.near(&exact),
        "a million steps land the lattice target"
    );

    // the matrix foil: the same million multiplications leak through the
    // orthogonality constraint — the measured drift renormalize() exists for
    let (c, s) = ((PI / 7.0).cos(), (PI / 7.0).sin());
    let (mut x, mut y) = (1.0_f64, 0.0_f64); // first column of Rⁿ
    for _ in 0..1_000_000 {
        (x, y) = (x * c - y * s, x * s + y * c);
    }
    let defect = (x * x + y * y - 1.0).abs();
    assert!(
        defect > 4.0 * f64::EPSILON,
        "the matrix column drifted off unit length: {defect:.2e}"
    );
    eprintln!(
        "matrix orthogonality defect after 10^6 multiplies: {defect:.2e} ({:.0} ulps); geonum magnitude defect: 0 bits",
        defect / f64::EPSILON
    );
}

#[test]
fn it_never_leaves_the_manifold_because_there_is_no_manifold() {
    // |q| = 1 is a constraint quaternion arithmetic must maintain; RᵀR = I is
    // a constraint matrix arithmetic must maintain. the magnitude here is not
    // a constraint — it is data, and rotation's arithmetic never touches it,
    // so a million compositions leave it bit-identical. no schedule, no
    // projection back, nothing to maintain
    let step = Geonum::new(1.0, 3.0, 11.0); // an awkward step, 3π/11

    let mut unit = Geonum::new(1.0, 0.0, 1.0);
    let mut payload = Geonum::new(3.7, 1.0, 6.0); // arbitrary magnitude rides too
    for _ in 0..1_000_000 {
        unit = unit.rotate(step.angle);
        payload = payload.rotate(step.angle);
    }

    assert_eq!(
        unit.mag.to_bits(),
        1.0_f64.to_bits(),
        "unit magnitude bit-stable through a million compositions"
    );
    assert_eq!(
        payload.mag.to_bits(),
        3.7_f64.to_bits(),
        "any magnitude is bit-stable — it was never part of a constraint surface"
    );
}
