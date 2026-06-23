// many-body gravity, as geometry: bodies bound and interacting through angle, not force vectors.
// the inter-body dynamics — a binary clocked by its barycenter, tidal stretch as a field gradient,
// the L1 and equilateral balance points, the bending of starlight, planetary conjunction, one
// influence law from a planet's surface to an orbit. each test lands on a number the sky already
// published — Sirius at ~50 years, L1 at 1.5 million km, 1.75 arcsec at the sun's limb, Earth–Mars
// every 780 days — computed O(1) where the textbook reaches for an ODE, a tensor, or a per-scale solver.

use geonum::*;
use std::f64::consts::PI;

#[test]
fn it_binds_a_binary_at_its_observed_period() {
    // a binary is bound when the gravitational pull supplies the centripetal acceleration of the
    // orbit — two vectors pointing to the barycenter. take Sirius's OBSERVED 50.1-year period, get
    // ω = 2π/P, and the centripetal vector ω²a must match the pull G·M/a², heading and size both.
    // newton checked against the sky (inputs are rounded catalog values, hence the 1% band)
    let g = 6.674e-11_f64;
    let m_total = 3.08 * 1.989e30; // Sirius A+B, kg
    let a = 19.8 * 1.496e11; // 19.8 AU separation, m
    let period = 50.1 * 365.25 * 86400.0; // observed, seconds

    let omega = 2.0 * PI / period; // orbital rate from the observed period
    let centripetal = Geonum::new(omega * omega * a, 1.0, 1.0); // ω²a, toward the barycenter (π)
    let pull = Geonum::new(g * m_total / (a * a), 1.0, 1.0); // G·M/a², toward the barycenter (π)

    assert_eq!(
        centripetal.angle.blade(),
        pull.angle.blade(),
        "both accelerations point inward to the barycenter"
    );
    assert!(
        (centripetal.mag - pull.mag).abs() / pull.mag < 0.01,
        "ω²a from the 50-year period matches G·M/a² to 1% — Sirius is bound"
    );
}

#[test]
fn it_tides_by_a_field_gradient() {
    // tidal stretching is the field's difference across a body — the near side pulled harder than
    // the far. traditional: the tidal tensor ∂²Φ/∂xᵢ∂xⱼ. geonum: two field samples and a
    // subtraction, falling off as 1/d³
    let source_mass = 1000.0_f64;
    let d = 100.0_f64; // distance to the source
    let h = 2.0_f64; // the body's half-extent along the line

    let near = Geonum::new(source_mass / (d - h).powi(2), 0.0, 1.0);
    let far = Geonum::new(source_mass / (d + h).powi(2), 0.0, 1.0);
    let tidal = near - far; // near pulled harder → the body elongates toward the source

    // the difference follows the gradient ≈ 4·G·M·h / d³ (G = 1), steeper than the field's 1/d²
    let gradient = 4.0 * source_mass * h / d.powi(3);
    assert!(
        (tidal.mag - gradient).abs() < 1e-4,
        "tidal stretch ≈ 4GMh/d³ — the field gradient, two samples and a subtraction"
    );
}

#[test]
fn its_the_earth_sun_l1_where_soho_sits() {
    // L1 is where the Sun's pull, the Earth's pull, and the orbital centrifugal term cancel — three
    // contributions whose wave_sum vanishes. for Earth+Sun that lands ~1.5 million km sunward of
    // Earth, where SOHO parks, at the Hill distance r = R·(m/3M)^(1/3)
    let r = 1.496e11_f64; // Earth–Sun distance, m
    let big_m = 1.989e30_f64; // M_sun, kg
    let little_m = 5.972e24_f64; // m_earth, kg
    let g = 6.674e-11_f64;

    let l1 = r * (little_m / (3.0 * big_m)).powf(1.0 / 3.0); // distance from Earth, sunward
    assert!(
        (l1 / 1e9 - 1.5).abs() < 0.1,
        "Earth–Sun L1 is ~1.5 million km from Earth (SOHO), got {:.2}",
        l1 / 1e9
    );

    // the three accelerations at L1 cancel: Sun (toward the origin, π), Earth and centrifugal (+x)
    let x = r - l1; // L1's distance from the Sun
    let omega_sq = g * big_m / (r * r * r); // orbital angular velocity²
    let sun = Geonum::new(g * big_m / (x * x), 1.0, 1.0);
    let earth = Geonum::new(g * little_m / (l1 * l1), 0.0, 1.0);
    let centrifugal = Geonum::new(omega_sq * x, 0.0, 1.0);
    let net = GeoCollection::from(vec![sun, earth, centrifugal]).wave_sum();

    assert!(
        net.mag < 0.01 * sun.mag,
        "the three contributions cancel at L1 — the wave_sum zero"
    );
}

#[test]
fn it_runs_one_op_from_a_planet_surface_to_an_orbit() {
    // the same influence op GM/d² spans every scale — traditional astrophysics swaps codes per
    // regime (surface gravity, then celestial mechanics). one closure gives Earth's surface gravity
    // AND the Sun's pull on Earth, both matching the measured accelerations ~1600× apart
    let pull = |gm: f64, d: f64| Geonum::new(gm / (d * d), 1.0, 2.0);

    let surface = pull(3.986e14, 6.371e6); // GM_earth, Earth's radius
    assert!(
        (surface.mag - 9.82).abs() < 0.05,
        "Earth's surface gravity is 9.8 m/s², got {:.2}",
        surface.mag
    );

    let sun_on_earth = pull(1.327e20, 1.496e11); // GM_sun, 1 AU
    assert!(
        (sun_on_earth.mag - 5.93e-3).abs() < 1e-4,
        "the Sun's pull on Earth is 5.9e-3 m/s², got {:.2e}",
        sun_on_earth.mag
    );
}

#[test]
fn it_balances_an_equilateral_three_body() {
    // three equal masses 120° apart — the L4/L5 equilateral solution. traditional: integrate the
    // restricted three-body ODEs. geonum: the net pull on a body is the wave_sum of the other two,
    // and it points to the center, so it orbits there with no tangential push
    let m = 1.0_f64;
    let r = 1.0; // each body at distance r from the center, 120° apart
    let bodies = [
        Geonum::new(r, 0.0, 1.0), // 0
        Geonum::new(r, 2.0, 3.0), // 2π/3
        Geonum::new(r, 4.0, 3.0), // 4π/3
    ];

    let target = bodies[0];
    let pulls: Vec<Geonum> = [bodies[1], bodies[2]]
        .iter()
        .map(|&s| {
            let sep = s - target;
            Geonum::new_with_angle(m / (sep.mag * sep.mag), sep.angle)
        })
        .collect();
    let net = GeoCollection::from(pulls).wave_sum();

    // purely central: radially inward, no tangential component
    let inward = net.mag * net.angle.project(target.angle); // along the outward radial
    let tangential = net.mag * net.angle.project(target.angle + Angle::new(1.0, 2.0));
    assert!(
        inward < 0.0,
        "the net pull is radially inward — toward the center"
    );
    assert!(
        tangential.abs() < 1e-10,
        "no tangential component — a stable equilateral, no ODE integrated"
    );
}

#[test]
fn it_deflects_starlight_by_the_eddington_angle() {
    // light grazing a mass bends by α = 4GM/c²b — the deflection IS an angle, built directly, no
    // null geodesic integrated. for a ray grazing the sun's limb that angle is 1.75 arcsec — what
    // eddington measured in 1919, twice newton's 0.87, the test that made general relativity
    let gm_sun = 1.327_124_4e20_f64; // m³/s²
    let c = 299_792_458.0_f64; // m/s
    let r_sun = 6.957e8_f64; // grazing ray: impact parameter = the solar radius

    let alpha = 4.0 * gm_sun / (c * c * r_sun); // the deflection, radians
    let arcsec = alpha * 206_264.806; // radians → arcseconds
    assert!(
        (arcsec - 1.75).abs() < 0.01,
        "the sun bends starlight 1.75 arcsec — eddington 1919, got {arcsec:.3}"
    );

    // the bend itself is one angle addition — a ray along 0 exits along α, no geodesic
    let deflected = Angle::new(0.0, 1.0) + Angle::new(alpha / PI, 1.0);
    assert!(
        deflected.near_rad(alpha),
        "bending the ray is angle addition"
    );
}

#[test]
fn it_finds_earth_mars_conjunctions_every_780_days() {
    // two planets realign (conjunction) when the faster gains a full turn on the slower — the
    // synodic period 1/(1/T₁ − 1/T₂). Earth (365.25 d) and Mars (686.98 d) line up every ~780 days
    let t_earth = 365.25_f64;
    let t_mars = 686.98_f64;
    let synodic = 1.0 / (1.0 / t_earth - 1.0 / t_mars);
    assert!(
        (synodic - 780.0).abs() < 1.0,
        "Earth–Mars conjunctions every ~780 days, got {synodic:.0}"
    );

    // advance each planet by its own angle over that span; they return to a shared heading. the
    // relative rotation comes out to exactly one turn — the blade arithmetic, not an assumed 2π
    let earth = Geonum::new(1.0, 0.0, 1.0).rotate(Angle::new(2.0 * synodic / t_earth, 1.0));
    let mars = Geonum::new(1.5, 0.0, 1.0).rotate(Angle::new(2.0 * synodic / t_mars, 1.0));
    assert!(
        (earth.angle - mars.angle).near_rad(0.0),
        "after one synodic period the planets share a heading again — realigned"
    );
}
