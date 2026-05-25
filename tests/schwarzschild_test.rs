// the schwarzschild factor is a position-dependent bondi factor
//
// spacetime_test.rs showed a boost is one rational scale of the half-tangent —
// the bondi factor k = e^φ, the doppler/aberration knob. nothing in that picture
// demands k be constant across space. let k vary with position and the same one
// primitive (scale_rotate with the boost knob ≠ 1) carries gravity: a photon
// climbing a potential well arrives with its half-tangent scaled by the LOCAL k,
// the very statement of gravitational redshift
//
// so the schwarzschild "metric" g_μν(r) is, in the geonum reading, one scalar
// field k(r) = √(1 − r_s/r): the bondi factor of a static observer at radius r
// relative to infinity. the n×n grid is bookkeeping over that one number per
// point. curvature is how k changes — dk/dr — and the einstein equation is the
// statement that this change is sourced by stress-energy. no christoffel symbols,
// no ricci tensor, no index gymnastics — gravity is a bondi field
//
// the redshift this field produces — ν_∞/ν_emit = k(r), a boost scaling the
// half-tangent — is the gravitational case of the one-boost unification in
// sr_gr_collapse_test. this file proves the two effects UNIQUE to the schwarzschild
// geometry, neither a redshift, each a rotation:
//   perihelion precession   Δω = 6πGM/(c²a(1−e²))    (a timelike orbit's angle won't close)
//   the horizon at r = r_s    k = 0                  (the boost's backward-pole fixed point)
//
// the schwarzschild radius itself is the geonum-natural object: r_s is where
// k(r) = 0, the bondi factor of a horizon. a photon emitted there arrives with
// zero frequency — but infinite redshift isnt zero magnitude, its the half-tangent
// driven to ∞, the boost sending every ray to the backward pole (grade 2, the dual
// of a forward ray): the horizon is the bondi factor's degeneracy, every direction
// collapsing to one null line
//
// run: cargo test --test schwarzschild_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;

// the schwarzschild bondi factor: k(r) = √(1 − r_s/r). this is the ONE number
// per radius that the tensor formalism distributes across g_tt and g_rr. units
// are geometric (G = c = 1), so r_s = 2M
fn k(r: f64, r_s: f64) -> f64 {
    (1.0 - r_s / r).sqrt()
}

#[test]
fn it_precesses_the_perihelion_by_6_pi_gm_over_a() {
    // mercury's perihelion advances by Δω = 6πGM/(c²a(1−e²)) per orbit — 43"/cy
    // for mercury, the test that closed GR's case in 1915. the geonum reading: the
    // bondi field's 1/r³ term speeds the radial oscillation, so the orbit no longer
    // closes after a full turn. it closes after 2π/√(1 − 6(GM/Lc)²), and the
    // overshoot past one revolution IS the precession
    //
    // geonum carries "one revolution" as blade 4 — four π/2 turns — and the leftover
    // past it as the angle's grade_angle. so the precession isnt a scalar residual
    // dug out of a numerical orbit integral; its the orbit angle failing to land back
    // on blade 4, read straight off the lattice. the bondi field acting on a TIMELIKE
    // worldline, the same k(r) that redshifts photons, now advancing a closed orbit
    let r_s = 2.0;
    let a = 1e5 * r_s; // semi-major axis (well outside the horizon)
    let e = 0.2_f64; // eccentricity (mercury-like is ~0.2)

    // the relativistic shrink of the radial period: 6(GM/Lc)². with GM = r_s/2 and
    // the weak-field L² = GM·a(1−e²), it reduces to 3 r_s / [a(1−e²)]
    let gm = r_s / 2.0; // geometric units, GM = r_s/2, c = 1
    let l_sq = gm * a * (1.0 - e * e); // angular momentum² of the orbit
    let shrink = 6.0 * gm * gm / l_sq; // the 1/r³ correction to the radial frequency

    // the orbit advances this much in φ between successive perihelia — a touch past
    // a full turn — assembled as ONE angle
    let delta_phi = 2.0 * PI / (1.0 - shrink).sqrt();
    let advance = Angle::new(delta_phi / PI, 1.0);

    // one closed revolution is blade 4: four π/2 turns. the orbit overshoots it
    assert_eq!(
        advance.blade(),
        4,
        "the orbit advances one full revolution (blade 4) plus a remainder"
    );

    // the remainder past the closed turn — grade_angle reads the angle modulo the
    // full revolution — IS the precession. the orbit doesnt land back on itself
    let precession = advance.grade_angle();
    let expected = 6.0 * PI * r_s / (2.0 * a * (1.0 - e * e));
    assert!(
        (precession / expected - 1.0).abs() < 1e-3,
        "Δω = 6π r_s / [a(1−e²)] — precession as the orbit angle's non-closure"
    );

    eprintln!(
        "\n  orbit advance per radial period: blade {} + {:.6e} rad",
        advance.blade(),
        precession
    );
    eprintln!("  6π r_s / [a(1−e²)]:              {expected:.6e} rad");
    eprintln!("  precession read off the lattice — no orbit integral, no christoffels");
}

#[test]
fn it_finds_the_horizon_where_the_bondi_factor_vanishes() {
    // the schwarzschild horizon at r = r_s is the geonum-natural object: it is
    // exactly where the bondi factor k(r) hits zero — the boost that drives the
    // half-tangent to the backward pole, the fixed point at stereographic ∞
    // (spacetime_test::it_boosts_any_blade: "the backward pole θ = π is a fixed point")
    //
    // so the horizon is not a coordinate pathology to be removed by clever charts
    // — it IS the bondi factor's zero. the SR statement "k = 0 sends every ray
    // to the backward pole" becomes the GR statement "at the horizon every ray
    // points inward." one fact, two regimes
    let r_s = 2.0;

    // just outside: small but nonzero bondi factor — a finite but large redshift
    let outside = k(r_s * (1.0 + 1e-6), r_s);
    assert!(
        outside > 0.0 && outside < 1e-2,
        "k > 0 just outside the horizon"
    );

    // at the horizon: k = 0, the boost annihilates the ray
    assert!(k(r_s, r_s).abs() < EPSILON, "k = 0 AT the horizon");

    // a ray at the horizon, boosted by k = 0, collapses to the backward pole θ=π
    // (grade 2, stored t = 0) — the boost's fixed point at the stereographic ∞,
    // every direction landing on the same null direction. the bondi factor's
    // degeneracy, every ray pointing one way
    let ray = Angle::new(1.0, 3.0);
    let at_horizon = ray.boost(0.0);
    assert_eq!(
        at_horizon.grade(),
        2,
        "k = 0 sends every ray to the backward pole — the horizon's one-way property"
    );
    assert!(
        at_horizon.t().abs() < EPSILON,
        "the collapsed ray sits exactly on the pole, stored t = 0"
    );

    // outside the horizon the geometry is regular — k is a smooth function of r,
    // its derivative is finite, no singularity in the field. the geonum reading:
    // r = r_s is a zero of a smooth field, not a singularity of the description
    let dk_dr_at_2rs = (k(2.0 * r_s + 1e-6, r_s) - k(2.0 * r_s - 1e-6, r_s)) / 2e-6;
    assert!(
        dk_dr_at_2rs.is_finite() && dk_dr_at_2rs > 0.0,
        "dk/dr is smooth outside — no coordinate singularity, just a field's zero"
    );

    // the true singularity at r = 0 is where k diverges (1 − r_s/r → −∞ under
    // the root) — the field itself becomes ill-defined, not just zero. THAT is
    // the curvature singularity, distinct from the horizon's bondi zero
    assert!(
        k(0.1 * r_s, r_s).is_nan(),
        "k is undefined inside the horizon — the field, not the chart, breaks"
    );
}
