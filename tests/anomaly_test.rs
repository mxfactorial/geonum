// kepler's anomaly conversion is a boost
//
// three centuries of series expansions — the "equation of the center", laplace
// coefficients, bessel-function inversions — to convert between orbital ANGLES.
// the eccentric-to-true anomaly map tan(ν/2) = √((1+e)/(1−e))·tan(E/2) is a
// möbius dilation of the half-tangent, and geonum ships that operation as
// Angle::boost: one rational scale of the stored t, with bondi factor
// k = √((1−e)/(1+e))
//
// the identification runs deeper than a shared formula. the boost's velocity
// parameter is β = (k²−1)/(k²+1) = −e: relativistic stellar aberration and
// orbital anomaly conversion are ONE operation — kepler 1609 and einstein 1905
// compute the same möbius dilation, three centuries apart, and the half-tangent
// the formula wants is the coordinate geonum stores
//
// perihelion and aphelion are the dilation's two fixed points — the same poles
// the celestial boost fixes. the transcendental leg of orbit propagation
// (mean ↔ eccentric anomaly, area ↔ angle) stays transcendental; the leg the
// series ceremony multiplied on (eccentric ↔ true) is one boost
//
// run: cargo test --test anomaly_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const MERCURY_E: f64 = 0.205_630;
const HALLEY_E: f64 = 0.967_14;

// the bondi factor of an orbit: k = √((1−e)/(1+e))
fn bondi(e: f64) -> f64 {
    ((1.0 - e) / (1.0 + e)).sqrt()
}

// the closed-form conversion the textbooks derive by half-angle identities:
// cos ν = (cos E − e)/(1 − e·cos E), sin ν = √(1−e²)·sin E/(1 − e·cos E)
fn true_anomaly_foil(e: f64, big_e: f64) -> (f64, f64) {
    let d = 1.0 - e * big_e.cos();
    (
        (big_e.cos() - e) / d,
        ((1.0 - e * e).sqrt() * big_e.sin()) / d,
    )
}

#[test]
fn it_converts_eccentric_to_true_anomaly_with_one_boost() {
    // eccentric anomalies across all four quadrants of the orbit
    let anomalies = [
        Angle::new(1.0, 6.0),  // π/6
        Angle::new(1.0, 3.0),  // π/3
        Angle::new(2.0, 3.0),  // 2π/3 — blade 1
        Angle::new(7.0, 6.0),  // 7π/6 — blade 2, the return half
        Angle::new(11.0, 6.0), // 11π/6 — blade 3, inbound to perihelion
    ];

    for e in [MERCURY_E, HALLEY_E] {
        for big_e in anomalies {
            let nu = big_e.boost(bondi(e));
            let (cos_nu, sin_nu) = nu.cos_sin();
            let (foil_cos, foil_sin) = true_anomaly_foil(e, big_e.grade_angle());

            assert!(
                (cos_nu - foil_cos).abs() < 1e-12,
                "e={e}: cos ν from one boost matches the closed form"
            );
            assert!(
                (sin_nu - foil_sin).abs() < 1e-12,
                "e={e}: sin ν too — the quadrant rides the grade, no atan2 patching"
            );
        }
    }
}

#[test]
fn it_fixes_perihelion_and_aphelion_as_the_boost_poles() {
    // the möbius dilation has two fixed points, the forward and backward poles.
    // on the orbit they are perihelion (E = ν = 0) and aphelion (E = ν = π) —
    // the two places eccentric and true anomaly agree, because the dilation
    // holds them still
    for e in [MERCURY_E, HALLEY_E] {
        let perihelion = Angle::new(0.0, 1.0);
        let aphelion = Angle::new(1.0, 1.0);

        assert!(
            perihelion.boost(bondi(e)).near(&perihelion),
            "perihelion is the forward pole"
        );
        assert!(
            aphelion.boost(bondi(e)).near(&aphelion),
            "aphelion is the backward pole"
        );
    }
}

#[test]
fn it_inverts_the_conversion_by_the_reciprocal_boost() {
    // ν → E is the same dilation at 1/k — boosts compose by multiplying, so
    // k·(1/k) = 1 returns the angle. no series inversion, no newton iteration
    let k = bondi(MERCURY_E);
    let anomalies = [
        Angle::new(1.0, 5.0),
        Angle::new(3.0, 4.0),
        Angle::new(6.0, 5.0),
        Angle::new(9.0, 5.0),
    ];

    for big_e in anomalies {
        let round_trip = big_e.boost(k).boost(1.0 / k);
        assert!(
            round_trip.near(&big_e),
            "boost then reciprocal boost returns the eccentric anomaly"
        );
    }
}

#[test]
fn it_identifies_the_anomaly_map_as_aberration_at_beta_minus_e() {
    // the boost's velocity parameter computes to β = −e: the aberration formula
    // cos θ' = (cos θ + β)/(1 + β·cos θ) IS the anomaly conversion. starlight
    // aberration and orbit geometry are one dilation of the half-tangent
    for e in [MERCURY_E, HALLEY_E] {
        let k = bondi(e);
        let beta = (k * k - 1.0) / (k * k + 1.0);
        assert!(
            (beta + e).abs() < 1e-12,
            "the orbit's eccentricity is the boost's −β"
        );

        // one angle, three routes: the geonum boost, the aberration formula at
        // β = −e, the anomaly closed form — all land the same cos ν
        let big_e = Angle::new(2.0, 5.0);
        let (cos_e, _) = big_e.cos_sin();
        let aberration = (cos_e + beta) / (1.0 + beta * cos_e);
        let (anomaly, _) = true_anomaly_foil(e, big_e.grade_angle());
        let (boosted, _) = big_e.boost(k).cos_sin();

        assert!(
            (aberration - anomaly).abs() < 1e-12,
            "aberration at β = −e is the anomaly conversion"
        );
        assert!(
            (boosted - aberration).abs() < 1e-12,
            "and the boost computes both"
        );
    }
}

#[test]
fn it_agrees_with_the_conic_radius_at_both_anomalies() {
    // the ellipse itself is the anchor: r = a(1 − e·cos E) from the eccentric
    // anomaly and r = a(1−e²)/(1 + e·cos ν) from the true anomaly describe the
    // same point. run the boost between them and the two radii agree — the
    // conversion lands on the orbit the test never parametrized
    let a = 1.0; // semi-major axis
    let anomalies = [
        Angle::new(1.0, 6.0),
        Angle::new(1.0, 2.0),
        Angle::new(5.0, 6.0),
        Angle::new(3.0, 2.0),
    ];

    for e in [MERCURY_E, HALLEY_E] {
        for big_e in anomalies {
            let (cos_e, _) = big_e.cos_sin();
            let r_eccentric = a * (1.0 - e * cos_e);

            let (cos_nu, _) = big_e.boost(bondi(e)).cos_sin();
            let r_true = a * (1.0 - e * e) / (1.0 + e * cos_nu);

            assert!(
                (r_eccentric - r_true).abs() < 1e-12,
                "e={e}: both anomalies read the same radius through the boost"
            );
        }
    }
}

#[test]
fn it_crosses_the_blade_boundary_at_comet_eccentricity() {
    // halley at E = π/6: thirty degrees of eccentric anomaly is already 128° of
    // true anomaly — the comet spends its life near aphelion and whips through
    // perihelion. the whip IS the boost crossing the blade boundary: E sits at
    // blade 0, ν lands at blade 1, one rational scale of t carries it over
    let big_e = Angle::new(1.0, 6.0);
    assert_eq!(big_e.blade(), 0, "E = π/6 sits in the first quadrant");

    let nu = big_e.boost(bondi(HALLEY_E));
    assert_eq!(
        nu.blade(),
        1,
        "ν crosses into the second quadrant — the perihelion whip"
    );

    // the crossing is measured, not asserted by fiat: the closed form puts ν
    // there too. tan(π/12)/k = 2.073, ν = 2·atan(2.073) ≈ 2.243 rad ≈ 128°
    let (foil_cos, foil_sin) = true_anomaly_foil(HALLEY_E, PI / 6.0);
    let (cos_nu, sin_nu) = nu.cos_sin();
    assert!((cos_nu - foil_cos).abs() < 1e-12, "cos ν lands the whip");
    assert!(
        (sin_nu - foil_sin).abs() < 1e-12,
        "sin ν stays positive — outbound"
    );
}
