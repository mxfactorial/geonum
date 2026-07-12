// lift is a circulation readout
//
// aerodynamics carries velocity potentials, stream functions and complex
// integrals to reach one result: a wing's lift per span is ρUΓ — density times
// speed times circulation, perpendicular to the stream. the machinery is
// bookkeeping for three geonum facts:
//
//   - a free vortex is its circulation spread over the circle it crosses: the
//     circumference is a grade-1 boundary, so the field falls off as 1/r and
//     points along the tangent — spread's falloff exponent is the boundary's
//     grade, not a coordinate count
//   - the vortex is irrotational everywhere but its core: a loop that does
//     not enclose the center reads zero circulation — the inner and outer
//     arcs interfere to nothing and the radial legs are exact-zero dots. all
//     the curl is a point of winding
//   - kutta-joukowski: integrate the surface pressure around a cylinder with
//     circulation and the whole sum collapses to one wedge, [ρU] ∧ [Γ] — with
//     zero drag component, d'alembert's paradox included free
//
// run: cargo test --test fluid_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const GAMMA: f64 = 5.0; // circulation strength

// the vortex field at a position: the circulation spread over the circle it
// crosses — the circumference, a grade-1 boundary — pointed along the tangent
fn vortex_velocity(position: &Geonum) -> Geonum {
    let circumference = Geonum::new_with_angle(
        2.0 * PI * position.mag,
        position.angle + Angle::new(1.0, 2.0),
    );
    Geonum::new(GAMMA, 0.0, 1.0).spread(circumference)
}

#[test]
fn it_spreads_a_vortex_over_a_grade_1_boundary() {
    let position = Geonum::new(2.0, 1.0, 6.0); // r = 2 at π/6

    let v = vortex_velocity(&position);
    assert!(
        v.near_mag(GAMMA / (4.0 * PI)),
        "|v| = Γ/(2πr) — the circulation spread over the circumference"
    );
    assert_eq!(
        v.angle,
        position.angle + Angle::new(1.0, 2.0),
        "the field points along the tangent — the boundary's direction"
    );
    assert!(
        position.dot(&v).near_mag(0.0),
        "no radial component — the swirl carries nothing outward"
    );

    // the 1/r falloff is the grade-1 boundary's doing: double the radius,
    // halve the speed — and the circulation ∮v·dl recovers Γ at every radius
    let twice_out = Geonum::new(4.0, 1.0, 6.0);
    assert!(
        vortex_velocity(&twice_out).near_mag(v.mag / 2.0),
        "grade-1 spreading falls off as 1/r"
    );
    for r in [1.0, 2.5, 7.0] {
        let sample = Geonum::new(r, 1.0, 5.0);
        let circulation = vortex_velocity(&sample).mag * 2.0 * PI * r;
        assert!(
            (circulation - GAMMA).abs() < 1e-12,
            "r = {r}: the loop reads Γ back — circulation is radius-free"
        );
    }

    // contrast: spread the same source over a SPHERE — a grade-2 boundary,
    // area 4πr² — and the falloff is inverse-square. the exponent is the
    // boundary's grade, not a count of coordinates
    let sphere =
        |r: f64| Geonum::new(GAMMA, 0.0, 1.0).spread(Geonum::new(4.0 * PI * r * r, 0.0, 1.0));
    assert!(
        sphere(4.0).near_mag(sphere(2.0).mag / 4.0),
        "grade-2 spreading falls off as 1/r²"
    );
}

#[test]
fn it_keeps_all_the_curl_at_the_winding_core() {
    // circulate around an annular sector that does NOT enclose the center:
    // inner arc forward, radial out, outer arc backward, radial in. the vortex
    // is irrotational here — the loop reads zero
    let (r_inner, r_outer) = (1.0, 2.0);
    let arc = PI / 6.0; // the sector's angular width

    // the arc contributions: v·(r·dφ), forward on the inner arc, backward on
    // the outer — and Γ/(2π)·dφ is radius-free, so they cancel exactly
    let inner = Geonum::new(
        vortex_velocity(&Geonum::new(r_inner, 1.0, 8.0)).mag * r_inner * arc,
        0.0,
        1.0,
    );
    let outer = Geonum::new(
        vortex_velocity(&Geonum::new(r_outer, 1.0, 8.0)).mag * r_outer * arc,
        1.0,
        1.0,
    );
    let legs: GeoCollection = vec![inner, outer].into();
    assert!(
        legs.wave_sum().near_mag(0.0),
        "the arcs interfere to nothing — equal circulation shares, opposite senses"
    );

    // the radial legs contribute exact zeros: the field is a quarter turn off
    // the path, and the rational cosine of a quarter turn is 0.0 dead
    let radial_path = Geonum::new(1.0, 1.0, 8.0); // outward along the sector edge
    let v_on_edge = vortex_velocity(&Geonum::new(1.5, 1.0, 8.0));
    assert!(
        radial_path.dot(&v_on_edge).near_mag(0.0),
        "the radial legs are silent — v ⊥ dl exactly"
    );

    // yet any loop AROUND the core reads Γ (the test above): the curl is not
    // spread through the fluid — it is a point of winding at the center
}

#[test]
fn it_computes_lift_from_the_circulation_wedge() {
    // flow U past a unit cylinder carrying circulation Γ: surface speed is the
    // superposition of the doublet's tangential flow and the vortex — geonum
    // addition along the tangent. bernoulli prices each surface element and
    // the pressure sum is one interference total
    let (rho, u_inf, radius) = (1.2, 10.0, 1.0);
    let samples = 360;
    let d_phi = 2.0 * PI / samples as f64;

    let mut total = Geonum::scalar(0.0);
    for k in 0..samples {
        let phi = Angle::new(2.0 * (k as f64 + 0.5) / samples as f64, 1.0);

        // the doublet's surface flow 2U·sin φ along the tangent (sign in the
        // angle) plus the vortex's Γ/(2πa) — same axis, geonum addition
        let flow = Geonum::sin(phi).scale(2.0 * u_inf).rotate(phi);
        let vortex = vortex_velocity(&Geonum::new_with_angle(radius, phi));
        let v_surface = flow + vortex;

        // bernoulli: p = ½ρ(U² − V²), V² read as the self-dot
        let pressure = 0.5 * rho * (u_inf * u_inf - v_surface.dot(&v_surface).mag);

        // pressure pushes inward, suction pulls outward — the sign is a π turn
        let direction = if pressure >= 0.0 {
            phi + Angle::new(1.0, 1.0)
        } else {
            phi
        };
        total = total + Geonum::new_with_angle(pressure.abs() * radius * d_phi, direction);
    }

    // the whole surface integral collapses to the kutta-joukowski wedge
    let kutta_joukowski = Geonum::new(rho * u_inf, 0.0, 1.0).wedge(&Geonum::new(GAMMA, 1.0, 2.0));
    let lift = total.project_to_dimension(1);
    assert!(
        (lift - kutta_joukowski.mag).abs() < 1e-8,
        "lift = ρUΓ = {:.1} — the pressure sum IS the wedge",
        kutta_joukowski.mag
    );
    assert!(
        (lift - rho * u_inf * GAMMA).abs() < 1e-8,
        "the airplane flies on a circulation readout"
    );

    // and the streamwise component vanishes: no drag from the ideal flow —
    // d'alembert's paradox, included free in the interference
    assert!(
        total.project_to_dimension(0).abs() < 1e-8,
        "zero drag — the fore and aft pressures interfere away"
    );
}
