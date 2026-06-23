// an orbit, as geometry: a body rotating at fixed magnitude. the radius is conserved because
// rotate() only touches the angle, one turn is four blades, and the kinematic ladder (velocity,
// acceleration) is grade cycling. the dynamics — spiral winding, the flat-curve requirement,
// mercury's precession — are the same angle accumulation read at different scales

use geonum::*;
use std::f64::consts::PI;

const C: f64 = 299_792_458.0; // m/s
const GM_SUN: f64 = 1.327_124_400_18e20; // sun's standard gravitational parameter, m³/s²

// circular orbital speed for enclosed mass m at radius r, as a tangent geonum (G = 1)
fn orbital_speed(enclosed_mass: f64, r: f64) -> Geonum {
    Geonum::new((enclosed_mass / r).sqrt(), 1.0, 2.0)
}

#[test]
fn its_an_orbit() {
    // an orbit is a geonum rotating: the radius is fixed because rotate() only touches the
    // angle, and the angle accumulates — one full turn is four blades, back to the start grade
    let mut body = Geonum::new(5.0, 0.0, 1.0); // radius 5

    for _ in 0..4 {
        body = body.rotate(Angle::new(1.0, 2.0)); // +π/2 each step
    }

    assert!(
        body.near_mag(5.0),
        "the radius is conserved — rotate never touches magnitude"
    );
    assert_eq!(body.angle.blade(), 4, "one full orbit is four blades");
    assert_eq!(body.angle.grade(), 0, "back to the starting grade");
}

#[test]
fn it_splits_velocity_into_rotation_and_rate() {
    // differentiate() is the geometry — the π/2 rotation. the physical sizes (v = ωr, a = ω²r)
    // come from the rate ω, the shadow of how fast blade accumulates. each derivative rotates
    // AND scales
    let r = 4.0;
    let omega = 3.0;
    let position = Geonum::new(r, 1.0, 6.0); // radius 4 at π/6

    let velocity = position.differentiate().scale(omega);
    assert!(velocity.near_mag(omega * r), "|v| = ωr");
    assert!(
        (velocity.angle - position.angle).near(&Angle::new(1.0, 2.0)),
        "v is tangent to r"
    );

    let acceleration = velocity.differentiate().scale(omega);
    assert!(acceleration.near_mag(omega * omega * r), "|a| = ω²r");
    assert!(
        acceleration.angle.is_opposite(&position.angle),
        "centripetal acceleration points back at the center"
    );

    // specific angular momentum r ∧ v = ωr², kepler's areal velocity
    assert!(
        position.wedge(&velocity).near_mag(omega * r * r),
        "|r ∧ v| = ωr²"
    );
}

#[test]
fn it_winds_a_spiral_arm_as_a_winding_number() {
    // differential rotation winds the arm: blade counts the wraps, exactly as algebra_test
    // counts polynomial roots. v·t = 6π gives the inner three turns, the outer one
    let inner = Geonum::new(1.0, 0.0, 1.0).rotate(Angle::new(6.0, 1.0)); // φ = 6π
    let outer = Geonum::new(3.0, 0.0, 1.0).rotate(Angle::new(2.0, 1.0)); // φ = 2π

    assert_eq!(
        inner.angle.blade(),
        12,
        "inner swept three turns → blade 12"
    );
    assert_eq!(outer.angle.blade(), 4, "outer swept one turn → blade 4");
    assert_eq!(
        (inner.angle.blade() - outer.angle.blade()) / 4,
        2,
        "the arm wraps twice — a winding number off the lattice"
    );
}

#[test]
fn it_keeps_the_winding_in_the_angle_not_the_radius() {
    // the winding count is angular: rescale the galaxy by 10⁶ and the blade is unchanged
    let arm = Geonum::new(1.0, 0.0, 1.0).rotate(Angle::new(6.0, 1.0)); // blade 12
    let rescaled = arm.scale(1e6);
    assert_eq!(
        rescaled.angle.blade(),
        arm.angle.blade(),
        "winding survives any radial scale"
    );
    assert!(
        rescaled.near_mag(arm.mag * 1e6),
        "only the magnitude scaled"
    );
}

#[test]
fn it_requires_enclosed_mass_growing_as_r_for_a_flat_curve() {
    // visible mass alone (saturated M) declines keplerian: v = √(GM/r) ∝ 1/√r
    let m_visible = 100.0;
    let v_in = orbital_speed(m_visible, 10.0);
    let v_out = orbital_speed(m_visible, 40.0);
    assert!(
        v_in.near_mag(v_out.mag * (40.0_f64 / 10.0).sqrt()),
        "visible mass declines keplerian: v ∝ 1/√r"
    );

    // a flat curve is constant v, so √(GM/r) = const forces enclosed mass ∝ r — a geometric
    // requirement of the spherical orbital law, whatever supplies the mass
    let k = 5.0_f64;
    let flat = k.sqrt();
    for r in [10.0_f64, 50.0, 5000.0] {
        assert!(
            orbital_speed(k * r, r).near_mag(flat),
            "flat curve ⟺ enclosed mass ∝ r"
        );
    }
}

#[test]
fn it_leaves_a_linearly_growing_residual() {
    // the gap between the flat-curve requirement M(r) = v_flat²·r and saturated visible mass
    // grows linearly with radius — a number the spherical law leaves, not a substance
    let v_flat = 2.0_f64;
    let m_visible = 100.0;
    let residual = |r: f64| v_flat * v_flat * r - m_visible; // G normalized

    let growth = residual(1000.0) - residual(100.0);
    assert!(
        Geonum::scalar(growth).near_mag(v_flat * v_flat * (1000.0 - 100.0)),
        "the residual grows linearly with radius"
    );
}

#[test]
fn it_precesses_mercury_by_the_orbit_not_closing() {
    // the relativistic orbit closes at 2π/√(1 − 3rs/p), not 2π. the overshoot is the precession —
    // a blade residual, the orbit failing to land back on itself. mercury: 43"/cy
    let rs = 2.0 * GM_SUN / (C * C);
    let a = 5.790_905_0e10; // mercury semi-major axis
    let e = 0.205_630; // mercury eccentricity
    let p = a * (1.0 - e * e);

    let closure = (1.0 - 3.0 * rs / p).sqrt();
    let prec_per_orbit = 2.0 * PI / closure - 2.0 * PI;
    let orbits_per_century = 36525.0 / 87.9691;
    let arcsec = prec_per_orbit * orbits_per_century * 206_264.806;

    assert!(
        (arcsec - 43.0).abs() < 1.0,
        "mercury precesses ~43 arcsec/century, got {arcsec:.2}"
    );
}
