// anholonomy is enclosed angle
//
// carry a vector around a closed loop on a curved surface and it comes back
// rotated. the conventional treatment builds connections, covariant
// derivatives and fiber bundles; the returned rotation is just the curvature
// the loop encloses — gauss_bonnet_test's split, read as physics:
//
//   - the foucault pendulum precesses because parallel transport around a
//     latitude circle picks up the polar cap's solid angle: the earth turns
//     2π per day, the surface keeps 2π(1 − sin λ) as transport, the pendulum
//     shows the remainder 2π·sin λ. paris measures ~32 hours per cycle off
//     that split
//   - the berry phase is the same holonomy at spinor half-rate: a spin-1/2
//     transported around a loop of solid angle Ω picks up Ω/2 — and a great
//     circle (Ω = 2π) lands π, grade 2: the −1 spinor_test reads at one turn,
//     produced here by geometry alone
//
// no connection coefficients, no bundle: the transported rotation is one
// angle subtraction from a full turn
//
// run: cargo test --test holonomy_test -- --show-output

use geonum::*;

#[test]
fn it_precesses_the_foucault_pendulum_by_the_transport_deficit() {
    // paris: latitude 48.8566°
    let latitude = Angle::new(48.8566, 180.0);
    let (_, sin_lat) = latitude.cos_sin();

    // one day's walk around the latitude circle encloses the polar cap —
    // solid angle 2π(1 − sin λ) — and that is the transport the surface keeps
    let transport = Angle::new(2.0 * (1.0 - sin_lat), 1.0);

    // the pendulum precesses by what remains of the day's full turn
    let precession = Angle::new(2.0, 1.0) - transport;
    assert!(
        precession.near_rad(2.0 * std::f64::consts::PI * sin_lat),
        "daily precession = 2π·sin λ — the deficit of the enclosed cap"
    );

    // the split is exact: transport + precession = the whole day
    assert!(
        (transport + precession).near(&Angle::new(2.0, 1.0)),
        "the earth's 2π divides between the surface and the pendulum"
    );

    // the number on the panthéon plaque: a full pendulum cycle takes
    // 24h/sin λ ≈ 31.9 hours at paris
    let hours = 24.0 / sin_lat;
    assert!(
        (hours - 31.9).abs() < 0.1,
        "paris cycle ≈ 31.9 h, computed {hours:.2}"
    );

    // the limits: at the pole the loop encloses nothing and the pendulum
    // shows the whole turn; at the equator the cap is the hemisphere and the
    // pendulum stands still
    let (_, sin_pole) = Angle::new(1.0, 2.0).cos_sin(); // λ = 90°
    assert!(
        (Angle::new(2.0, 1.0) - Angle::new(2.0 * (1.0 - sin_pole), 1.0))
            .near(&Angle::new(2.0, 1.0)),
        "pole: one full precession per day"
    );
    let (_, sin_equator) = Angle::new(0.0, 1.0).cos_sin(); // λ = 0
    assert!(
        (Angle::new(2.0, 1.0) - Angle::new(2.0 * (1.0 - sin_equator), 1.0)).near_rad(0.0),
        "equator: the plane never precesses"
    );
}

#[test]
fn it_reads_the_berry_phase_as_half_the_solid_angle() {
    // adiabatic transport of a spin-1/2 around a loop enclosing solid angle Ω
    // acquires geometric phase Ω/2 — the holonomy at the spinor's half rate
    // (spinor_test: spin halves every angle)

    // the octant loop: gauss_bonnet's triangle, solid angle π/2 — the berry
    // phase is π/4
    let octant = Angle::new(1.0, 2.0);
    let berry_octant = octant / 2.0;
    assert!(
        berry_octant.near(&Angle::new(1.0, 4.0)),
        "Ω = π/2 → γ = π/4 — the octant's phase"
    );

    // a great-circle loop encloses a hemisphere: Ω = 2π, berry phase π —
    // grade 2, the −1. the sign flip spinor_test produced by rotation appears
    // here from pure geometry: no field turned the spin, the path did
    let hemisphere = Angle::new(2.0, 1.0);
    let berry_flip = hemisphere / 2.0;
    assert_eq!(
        berry_flip.grade(),
        2,
        "Ω = 2π → γ = π: the spin returns negated — geometry alone flips it"
    );

    // interferometry reads it the same way spinor_test reads the 2π rotation:
    // the transported arm cancels the reference arm
    let reference = Geonum::new(1.0, 1.0, 8.0);
    let transported = reference.rotate(berry_flip);
    assert!(
        (reference + transported).near_mag(0.0),
        "the berry-phase arm interferes destructively — the measured −1"
    );

    // and the two famous phases are one formula: the pendulum's cap and the
    // spin's hemisphere differ only in the loop, the half only in the carrier
    let cap = Angle::new(2.0 * (1.0 - 0.75), 1.0); // a λ ≈ 48.6° cap
    assert!(
        (cap / 2.0 + cap / 2.0).near(&cap),
        "holonomy halves recompose — foucault and berry share the geometry"
    );
}
