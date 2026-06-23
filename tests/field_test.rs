// the inverse-square field, as geometry: the "square" is a wedge, the exponent a grade, the
// falloff a spread, and many sources superpose as a wave_sum. gravity and charge alike — one
// field, the landed Geonum::spread underneath

use geonum::*;
use std::f64::consts::PI;

const FULL_SPHERE: f64 = 4.0 * PI; // total solid angle, steradians

// the inverse-square field a source casts at the far end of a separation, via Geonum::spread:
// the source spread over the flux area (r², the separation self-wedged), re-pointed radially
fn field(source: Geonum, separation: Geonum) -> Geonum {
    let area = Geonum::new_with_angle(
        separation
            .wedge(&separation.rotate(Angle::new(1.0, 2.0)))
            .mag, // r²
        separation.angle, // radial direction
    );
    source.spread(area)
}

#[test]
fn it_builds_the_inverse_square_area_as_a_wedge() {
    // the r² in 1/r² is the sphere's area — a bivector wedge, not a squared length
    let r = 3.0;
    let area = Geonum::new(r, 0.0, 1.0).wedge(&Geonum::new(r, 1.0, 2.0));
    assert!(
        area.near_mag(r * r),
        "the flux area is r² — a wedge magnitude"
    );
    assert_eq!(
        area.angle.grade(),
        2,
        "a bivector, grade 2, not a scalar square"
    );
}

#[test]
fn it_reads_the_falloff_exponent_as_the_boundary_grade() {
    // the inverse-power exponent is the grade of the flux boundary, not a coordinate count
    let r = 4.0;

    let circle = Geonum::new(r, 1.0, 2.0); // a single r-edge
    assert_eq!(circle.angle.grade(), 1, "flatland boundary is grade 1");
    assert!(circle.near_mag(r), "area ∝ r¹ → 1/r");

    let sphere = Geonum::new(r, 0.0, 1.0).wedge(&Geonum::new(r, 1.0, 2.0));
    assert_eq!(sphere.angle.grade(), 2, "space boundary is grade 2");
    assert!(sphere.near_mag(r * r), "area ∝ r² → 1/r²");
}

#[test]
fn it_spreads_flux_square_free_per_solid_angle() {
    // flux per steradian is the intensity, Φ/4π — no r, no square. the conserved quantity
    let intensity = Geonum::new(100.0, 0.0, 1.0).spread(Geonum::scalar(FULL_SPHERE));
    assert!(
        intensity.near_mag(100.0 / FULL_SPHERE),
        "intensity = flux / 4π"
    );
}

#[test]
fn it_keeps_the_square_only_in_the_projection() {
    // project the intensity onto the area at a distance and the 1/r² field appears — but
    // multiply the square back out and the same conserved intensity returns from any distance
    let intensity = Geonum::new(100.0, 0.0, 1.0).spread(Geonum::scalar(FULL_SPHERE));
    let area = |r: f64| {
        Geonum::new_with_angle(
            Geonum::new(r, 0.0, 1.0)
                .wedge(&Geonum::new(r, 1.0, 2.0))
                .mag,
            Angle::new(1.0, 4.0),
        )
    };

    let near = intensity.spread(area(1.0));
    let far = intensity.spread(area(2.0));
    assert!(
        near.near_mag(far.mag * 4.0),
        "the field falls 4× over 2× distance"
    );
    assert!(
        Geonum::scalar(near.mag).near_mag(intensity.mag),
        "field(1)·1² = intensity"
    );
    assert!(
        Geonum::scalar(far.mag * 4.0).near_mag(intensity.mag),
        "field(2)·2² = intensity — the square was only the projection"
    );
}

#[test]
fn it_serves_gravity_and_charge_through_one_spread() {
    // gravity and the electric field are one spread: when G·M = k·Q the fields are identical
    let g = 6.674e-11_f64;
    let k = 8.988e9_f64;
    let (m, r) = (5.0, 2.0);
    let area = Geonum::new_with_angle(
        Geonum::new(r, 0.0, 1.0)
            .wedge(&Geonum::new(r, 1.0, 2.0))
            .mag,
        Angle::new(1.0, 1.0),
    );

    let gravity = Geonum::new(g * m, 0.0, 1.0).spread(area);
    let q = g * m / k; // pick Q so k·Q = G·M
    let charge = Geonum::new(k * q, 0.0, 1.0).spread(area);
    assert!(
        gravity.near_mag(charge.mag),
        "G·M = k·Q → one field from one spread"
    );
}

#[test]
fn it_superposes_fields_as_the_wave_sum() {
    // the net field from many sources is GeoCollection::wave_sum — the interfering vector
    // sum, order-independent, no force-vector basis
    let body = Geonum::new(1.0, 0.0, 1.0);
    let s1 = Geonum::new(5.0, 1.0, 3.0);
    let s2 = Geonum::new(8.0, 1.0, 6.0);

    let f1 = field(Geonum::new(2.0, 0.0, 1.0), s1 - body);
    let f2 = field(Geonum::new(3.0, 0.0, 1.0), s2 - body);

    let net = GeoCollection::from(vec![f1, f2]).wave_sum();
    let swapped = GeoCollection::from(vec![f2, f1]).wave_sum();
    assert!(
        net.near(&swapped),
        "superposition commutes — wave_sum is order-independent"
    );
    assert!(net.mag > f1.mag, "the second source adds to the net");
}

#[test]
fn it_balances_between_two_equal_masses() {
    // equal opposite pulls cancel — the lagrange balance is the interference gap maxed out:
    // wave_sum().mag <= total_magnitude() because + is cosine interference
    let body = Geonum::new_from_cartesian(0.0, 0.0);
    let left = Geonum::new_from_cartesian(-4.0, 0.0);
    let right = Geonum::new_from_cartesian(4.0, 0.0);

    let pulls = GeoCollection::from(vec![
        field(Geonum::new(10.0, 0.0, 1.0), left - body),
        field(Geonum::new(10.0, 0.0, 1.0), right - body),
    ]);
    let single = field(Geonum::new(10.0, 0.0, 1.0), left - body).mag;

    assert!(
        Geonum::scalar(pulls.total_magnitude()).near_mag(2.0 * single),
        "the scalar sum is both pulls added"
    );
    assert!(
        pulls.wave_sum().near_mag(0.0),
        "wave_sum cancels — the whole magnitude is angular cancellation"
    );
}

#[test]
fn it_keeps_each_interaction_o1_in_any_dimension() {
    // each pairwise interaction is O(1) and dimension-free: two values per body, the same
    // whether it indexes a plane or a million-D structure (a +1_000_000 blade is ≡ 0 mod 4)
    let body = Geonum::new(1.0, 1.0, 7.0);
    let source = Geonum::new(3.0, 2.0, 7.0);
    let body_hi = Geonum::new_with_angle(body.mag, Angle::new_with_blade(1_000_000, 1.0, 7.0));
    let source_hi = Geonum::new_with_angle(source.mag, Angle::new_with_blade(1_000_000, 2.0, 7.0));

    let f_lo = field(Geonum::new(4.0, 0.0, 1.0), source - body);
    let f_hi = field(Geonum::new(4.0, 0.0, 1.0), source_hi - body_hi);

    assert!(
        f_hi.near_mag(f_lo.mag),
        "the million-D interaction computes the same as planar"
    );
    assert_eq!(
        std::mem::size_of_val(&body),
        std::mem::size_of_val(&body_hi),
        "two values per body, no 2^n"
    );
}

#[test]
fn it_weaves_more_field_than_the_spherical_scalar() {
    // the field of a flat distribution is the angular wave_sum of its matter, not the spherical
    // scalar enclosed-mass. summed in the angle, a disk weaves more than the shell theorem
    let disk = disk_elements();
    let angular = disk_speed(&disk, 3.0); // the wave_sum of the disk's pulls
    let spherical = spherical_speed(&disk, 3.0); // the shell-theorem scalar
    assert!(
        angular > spherical,
        "the field is the angular sum — it weaves more than the spherical scalar"
    );
}

// ─── a galactic disk: regular matter in the z=0 plane ────────────────────────

const G: f64 = 1.0;
const SOFTENING: f64 = 0.05;

fn disk_elements() -> Vec<(Geonum, f64)> {
    let (a_max, n_a, n_phi) = (8.0_f64, 48usize, 36usize);
    let da = a_max / n_a as f64;
    let mut elements = Vec::with_capacity(n_a * n_phi);
    for ia in 0..n_a {
        let a = (ia as f64 + 0.5) * da;
        let dm = (-a).exp() * a * da * (2.0 * PI / n_phi as f64); // exponential surface density
        for ip in 0..n_phi {
            let phi = ip as f64 * 2.0 * PI / n_phi as f64;
            elements.push((Geonum::new_from_cartesian(a * phi.cos(), a * phi.sin()), dm));
        }
    }
    elements
}

fn disk_speed(elements: &[(Geonum, f64)], r: f64) -> f64 {
    let point = Geonum::new_from_cartesian(r, 0.0);
    let pulls: Vec<Geonum> = elements
        .iter()
        .map(|(pos, m)| {
            let sep = *pos - point;
            let d = sep.mag;
            Geonum::new_with_angle(G * m / (d * d + SOFTENING * SOFTENING), sep.angle)
        })
        .collect();
    let net = GeoCollection::from(pulls).wave_sum();
    let inward = net.mag * net.angle.project(Angle::new(1.0, 1.0)); // component toward center
    (inward * r).sqrt()
}

fn spherical_speed(elements: &[(Geonum, f64)], r: f64) -> f64 {
    let enclosed: f64 = elements
        .iter()
        .filter(|(pos, _)| pos.mag < r)
        .map(|(_, m)| m)
        .sum();
    (G * enclosed / r).sqrt()
}
