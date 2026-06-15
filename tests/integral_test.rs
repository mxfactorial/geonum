//! integral test against the geonum api
//!
//! integration is never a riemann limit here. two honest routes draw the area,
//! neither a sum of strips:
//!
//! 1. the antiderivative is blade -1 — the inverse tick of differentiation.
//!    a definite integral over [a, b] reads the antiderivative at the two
//!    endpoints and subtracts; the interior telescopes to nothing, only the
//!    boundary survives. the grade-cycle mechanics (differentiate/integrate as
//!    +1/-1 blade ticks, the four-step loop, the riemann-sum form of the
//!    fundamental theorem) are proven in calculus_test — this file keeps only
//!    what that one leaves out and builds the integration-specific results on it.
//!
//! 2. area is a swept wedge — v ∧ w is the oriented area |v||w|sin(Δθ) drawn as
//!    one vector rotates onto the other. a region bounded by straight edges (a
//!    triangle, a parallelogram) is a finite wedge: one rotation,
//!    exact, no mesh. the riemann rectangle is the degenerate perpendicular
//!    wedge (sin = 1) the mesh limit sums — geonum skips it, the area was a
//!    wedge all along, and most regions are swept or bounded, not meshed.
//!
//! 3. the riemann mesh is unnecessary — it sums f(x)·Δx over n strips and only
//!    CONVERGES to the area, never reaching it. the wedge draws the same region
//!    exactly in one rotation — no step, no mesh.
//!
//! the integration patterns geonum has no named api for yet — boundary
//! evaluation and the swept-area element — are factored into helpers stacked at
//! the bottom of this file.
//!
//! run with:  cargo test --test integral_test

use geonum::{Angle, Geonum};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;

// ---------------------------------------------------------------------------
// 1. integrate is differentiate's exact inverse — and not only on grade.
//    calculus_test proves the grade cycle and that the round trip returns the
//    grade; this proves the round trip also recovers the projection ratio t,
//    across every blade, so nothing within the π/2 segment is lost either
// ---------------------------------------------------------------------------
#[test]
fn it_inverts_differentiation_preserving_the_projection_ratio() {
    for blade in 0..8usize {
        let g = Geonum::new_with_angle(1.0, Angle::new_with_blade(blade, 0.0, 1.0));
        // give it a nonzero projection ratio so t is exercised, not just blade
        let g = Geonum::new_with_angle(1.0, g.angle + Angle::new(1.0, 6.0)); // + pi/6

        let round_trip = g.integrate().differentiate();

        assert_eq!(
            round_trip.angle.grade(),
            g.angle.grade(),
            "blade {blade}: grade must survive d(integral)"
        );
        assert!(
            (round_trip.angle.grade_angle() - g.angle.grade_angle()).abs() < EPSILON,
            "blade {blade}: the projection ratio t (carried in grade_angle) must return exactly"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. definite integral of cos over [a, b] = sin(b) - sin(a)
//    antiderivative of cos is sin — its projection read off the angle, the two
//    endpoints subtracted by definite_integral
// ---------------------------------------------------------------------------
#[test]
fn it_integrates_cos_over_an_interval() {
    // (a, b, expected) — endpoints as π fractions, no raw radians
    let cases = [
        (Angle::new(0.0, 1.0), Angle::new(1.0, 2.0), 1.0), // sin(π/2) - sin(0)
        (Angle::new(0.0, 1.0), Angle::new(1.0, 1.0), 0.0), // sin(π) - sin(0)
        (
            Angle::new(1.0, 6.0),
            Angle::new(1.0, 3.0),
            3.0_f64.sqrt() / 2.0 - 0.5,
        ), // sin60 - sin30
        (Angle::new(1.0, 4.0), Angle::new(3.0, 4.0), 0.0), // sin135 - sin45
        (
            Angle::new(0.0, 1.0),
            Angle::new(1.0, 3.0),
            3.0_f64.sqrt() / 2.0,
        ), // sin60 - 0
    ];

    for (a, b, expected) in cases {
        // ∫cos = sin: the imaginary projection of the angle
        let area = definite_integral(a, b, |x| x.cos_sin().1);

        assert!(
            (area - expected).abs() < EPSILON,
            "integral of cos = {expected}, got {area}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. definite integral of sin over [a, b] = cos(a) - cos(b)
//    antiderivative of sin is -cos
// ---------------------------------------------------------------------------
#[test]
fn it_integrates_sin_over_an_interval() {
    let cases = [
        (Angle::new(0.0, 1.0), Angle::new(1.0, 2.0), 1.0), // cos(0) - cos(π/2)
        (Angle::new(0.0, 1.0), Angle::new(1.0, 1.0), 2.0), // cos(0) - cos(π) = 1 - (-1)
        (Angle::new(0.0, 1.0), Angle::new(1.0, 3.0), 0.5), // cos(0) - cos60 = 1 - 0.5
    ];

    for (a, b, expected) in cases {
        // ∫sin = -cos: negate the real projection of the angle
        let area = definite_integral(a, b, |x| -x.cos_sin().0);

        assert!(
            (area - expected).abs() < EPSILON,
            "integral of sin = {expected}, got {area}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. the four cardinal directions' real projections cancel — blade 0,1,2,3 are
//    two conjugate pairs (0 with π, π/2 with 3π/2) that annihilate. this is the
//    grade cycle closing on itself, not the period integral ∫₀^2π cos (that is
//    the FTC boundary read sin 2π − sin 0; the two share the value 0, not the
//    mechanism — a four-point sum is not the swept area)
// ---------------------------------------------------------------------------
#[test]
fn it_cancels_the_four_cardinal_cosines() {
    let mut sum = 0.0;
    for k in 0..4usize {
        // blade 0,1,2,3 = the four cardinal directions; real projection = cos
        sum += Angle::new_with_blade(k, 0.0, 1.0).cos_sin().0;
    }
    // cos 0 + cos π/2 + cos π + cos 3π/2 = 1 + 0 − 1 + 0: the +1/−1 a conjugate
    // pair, the two zeros another — symmetry, not integration
    assert!(
        sum.abs() < EPSILON,
        "the four cardinal cosines cancel in conjugate pairs, got {sum}"
    );
}

// ---------------------------------------------------------------------------
// 5. telescoping: subdivide [0, pi/2], interior endpoints cancel exactly
//    the sum equals F(b) - F(a) at EVERY resolution — not in the limit,
//    but identically, because +1 and -1 are exact inverses on the lattice
// ---------------------------------------------------------------------------
#[test]
fn it_telescopes_the_definite_integral_at_every_resolution() {
    let sin_anti = |x: Angle| x.cos_sin().1; // antiderivative of cos
    let (a, b) = (Angle::new(0.0, 1.0), Angle::new(1.0, 2.0)); // [0, π/2]
    let expected = definite_integral(a, b, sin_anti); // = 1.0

    for n in [1usize, 4, 16, 64] {
        let mut sum = 0.0;
        for k in 0..n {
            // slice endpoints as π fractions: (k/n)·(π/2)
            let x0 = Angle::new(k as f64 / n as f64, 2.0);
            let x1 = Angle::new((k + 1) as f64 / n as f64, 2.0);
            sum += definite_integral(x0, x1, sin_anti); // F(x1) - F(x0)
        }
        assert!(
            (sum - expected).abs() < EPSILON,
            "n = {n}: telescoping sum must equal F(b) - F(a) = {expected}, got {sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. the wedge is the area primitive — v ∧ w draws a parallelogram in one rotation
// ---------------------------------------------------------------------------
#[test]
fn it_draws_a_parallelogram_area_as_one_wedge() {
    // the wedge is the area primitive: v ∧ w is the area swept rotating v onto w,
    // |v||w|sin(Δθ). no integral, no limit — one rotation draws the whole region
    let v = Geonum::new_from_cartesian(3.0, 0.0); // along x, length 3
    let w = Geonum::new_from_cartesian(0.0, 4.0); // along y, length 4

    // a 3×4 rectangle: the perpendicular wedge, sin(π/2) = 1
    assert!(
        (v.wedge(&w).mag - 12.0).abs() < EPSILON,
        "v ∧ w = 12 — the rectangle's area in one wedge"
    );

    // a slanted parallelogram spanned by (2,0) and (1,2): area = |2·2 − 0·1| = 4.
    // here sin(Δθ) genuinely does the work — the area is the swept region, not a
    // base×height the mesh would chop up
    let a = Geonum::new_from_cartesian(2.0, 0.0);
    let b = Geonum::new_from_cartesian(1.0, 2.0);
    assert!(
        (a.wedge(&b).mag - 4.0).abs() < EPSILON,
        "the slanted parallelogram is one wedge — rotation sweeps the area"
    );
}

// ---------------------------------------------------------------------------
// 7. a sector is the area a rotating radius sweeps, ½r²θ — and θ is the
//    rotation itself, carried in the blade. grade_angle drops full turns; the
//    blade keeps them, so the swept area follows the turns the radius took
// ---------------------------------------------------------------------------
#[test]
fn it_sweeps_a_sector_by_the_radius_blade() {
    // ½r² is the constant areal density a rotating radius pays per radian; the
    // sector is its antiderivative ½r²θ, and θ is the angle the radius turned
    // through. that angle lands in the blade — rotate accumulates it, a full
    // turn advancing the blade by 4. so the swept area is read off the blade,
    // not a mesh of triangles summed, and not grade_angle (which drops the turns)
    let r = 2.0;
    let radius = Geonum::new(r, 0.0, 1.0); // a radius of length r, pointing +x

    // turn the radius and let the blade record the sweep
    let full = radius.rotate(Angle::new(2.0, 1.0)); // +2π → blade 4
    let quarter = radius.rotate(Angle::new(1.0, 2.0)); // +π/2 → blade 1

    // the swept angle is the blade's quarter-turns × π/2 — the winding a
    // projection would discard, kept because the area needs every turn
    let sector = |g: &Geonum| 0.5 * r * r * (g.angle.blade() as f64 * PI / 2.0);

    // a full turn sweeps the whole disk; a quarter turn a quarter of it. each is
    // checked against its own closed form, the ratio falling out of the blades
    assert!(
        (sector(&full) - PI * r * r).abs() < EPSILON,
        "a full turn (blade 4) sweeps πr² — the disk, the rotation landing on blade 4"
    );
    assert!(
        (sector(&quarter) - PI * r * r / 4.0).abs() < EPSILON,
        "a quarter turn (blade 1) sweeps ¼πr² — the area follows the blade"
    );
}

// ---------------------------------------------------------------------------
// 8. ∫₀^b x dx is the area under y = x — half of ONE wedge, no rectangles summed
// ---------------------------------------------------------------------------
#[test]
fn it_integrates_under_a_line_as_one_triangle_wedge() {
    // ∫₀^b x dx is the area under the line y = x from 0 to b — the triangle with
    // corners (0,0), (b,0), (b,b). geonum draws it as half of ONE wedge: the
    // parallelogram spanned by the two edges from the origin, halved. exact, no
    // mesh, no limit — the "area under the curve" the riemann sum labors over
    let b = 3.0;
    let along = Geonum::new_from_cartesian(b, 0.0); // (b, 0), the base edge
    let up_the_line = Geonum::new_from_cartesian(b, b); // (b, b), the far corner on y = x

    let integral = areal(&along, &up_the_line);
    assert!(
        (integral - b * b / 2.0).abs() < EPSILON,
        "∫₀^b x dx = b²/2 — the triangle is one wedge, no rectangles summed"
    );
}

// ---------------------------------------------------------------------------
// 9. kepler's second law: the swept-area wedge ½ r ∧ v is conserved
// ---------------------------------------------------------------------------
#[test]
fn it_conserves_swept_area_in_equal_times() {
    // kepler's second law: a planet sweeps equal areas in equal times. the areal
    // velocity is ½ r ∧ v — the wedge of position and velocity — and a central
    // force holds it constant. no orbit integral: the conserved quantity IS the
    // swept-area wedge, read at any two points and found equal
    let (a, e, gm) = (1.0, 0.5, 1.0_f64); // semi-major axis, eccentricity, GM

    // at perihelion and aphelion the velocity is perpendicular to the radius
    let r_peri = a * (1.0 - e);
    let r_apo = a * (1.0 + e);
    let v_peri = (gm * (2.0 / r_peri - 1.0 / a)).sqrt(); // vis-viva
    let v_apo = (gm * (2.0 / r_apo - 1.0 / a)).sqrt();

    // radius along the axis, velocity a quarter turn off it — their wedge is the
    // areal velocity, ½|r ∧ v|
    let areal_peri = areal(
        &Geonum::new(r_peri, 0.0, 1.0),
        &Geonum::new(v_peri, 1.0, 2.0),
    );
    let areal_apo = areal(&Geonum::new(r_apo, 0.0, 1.0), &Geonum::new(v_apo, 1.0, 2.0));

    assert!(
        (areal_peri - areal_apo).abs() < EPSILON,
        "½ r ∧ v is equal at perihelion and aphelion — equal areas in equal times"
    );
}

// ---------------------------------------------------------------------------
// 10. the swept area is oriented: v ∧ w = −w ∧ v, so ∫_a^b = −∫_b^a
// ---------------------------------------------------------------------------
#[test]
fn it_orients_the_swept_area_by_its_direction() {
    // the swept area is ORIENTED: sweeping v onto w is the negative of sweeping w
    // onto v, v ∧ w = −w ∧ v. this is why ∫_a^b = −∫_b^a — reversing the path
    // reverses the rotation, flipping the sign of the area it sweeps. the two
    // bivectors are equal in magnitude and opposite in orientation, so they cancel
    let v = Geonum::new_from_cartesian(2.0, 1.0);
    let w = Geonum::new_from_cartesian(1.0, 3.0);

    assert!(
        (v.wedge(&w) + w.wedge(&v)).mag < EPSILON,
        "v ∧ w + w ∧ v = 0 — reversing the sweep negates the area, ∫_a^b = −∫_b^a"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// the riemann mesh is unnecessary: it sums f(x)·Δx over n strips and only
// converges. the wedge draws the area exactly in one rotation — no mesh
// ═══════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// 11. the riemann mesh only converges; the wedge is exact
// ---------------------------------------------------------------------------
#[test]
fn it_takes_down_the_riemann_mesh() {
    // the riemann integral chops [0,1] into n strips and sums f(x)·Δx — a mesh that only
    // CONVERGES to the area, never reaching it, at O(n) cost
    let exact = 0.5; // ∫₀¹ x dx, the triangle under y = x
    let mut last = f64::INFINITY;
    for n in [10usize, 100, 1000] {
        let mesh = riemann_left(|x| x, n);
        let err = (mesh - exact).abs();
        assert!(
            err > 1e-4,
            "n={n}: the mesh is still off by {err:.2e} — it never arrives"
        );
        assert!(err < last, "more strips, less error — but never zero");
        last = err;
    }

    // the wedge draws the same triangle in one rotation: ½|along ∧ up| = ½, exact, no strips
    let along = Geonum::new_from_cartesian(1.0, 0.0); // (1, 0), the base
    let up_the_line = Geonum::new_from_cartesian(1.0, 1.0); // (1, 1) on y = x
    assert!(
        (areal(&along, &up_the_line) - exact).abs() < EPSILON,
        "the wedge is exact: ½, no mesh"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// helpers: the geonum integration patterns, plus the finite-difference foil they
// replace. rust scopes a `let` closure to a fn body, so module-level helpers are
// `fn` — each still captures nothing and reads as the lambda it stands in for
// ───────────────────────────────────────────────────────────────────────────

/// the fundamental theorem as one operation: a definite integral is the
/// antiderivative read at the two endpoints and subtracted. no interior, no
/// limit — `antideriv` is F, evaluated at b and a
fn definite_integral(a: Angle, b: Angle, antideriv: impl Fn(Angle) -> f64) -> f64 {
    antideriv(b) - antideriv(a)
}

/// the swept-area element ½|v ∧ w| — half the parallelogram the wedge draws is
/// the triangle it cuts (a region under a line, an orbit's areal velocity)
fn areal(v: &Geonum, w: &Geonum) -> f64 {
    0.5 * v.wedge(w).mag
}

/// the finite-difference integral — n left-rectangles summed, Σ f(k/n)·(1/n). the
/// mesh the wedge above replaces, kept here only as the foil the takedown runs
fn riemann_left(f: impl Fn(f64) -> f64, n: usize) -> f64 {
    let dx = 1.0 / n as f64;
    (0..n).map(|k| f(k as f64 * dx) * dx).sum()
}
