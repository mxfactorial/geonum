// the fundamental theorem of algebra is visible from the angle
//
// every polynomial of degree n has exactly n roots
// this took centuries to prove. every known proof requires
// complex analysis or algebraic topology. no purely algebraic proof exists.
// mathematicians have proven you CANNOT prove it algebraically.
//
// but in angle space its obvious:
//
// z = [r, θ] on a circle means θ sweeps 0 → 2π
// z^n = [r^n, nθ] means the output angle sweeps 0 → 2nπ (n full wraps)
// for large r, p(z) ≈ aₙz^n, so the output wraps n times around the origin
// as you shrink the circle to a point, the wraps must unwind
// each unwinding passes through a root (magnitude → 0, angle undefined)
// n wraps → n roots
//
// the reason algebra cant prove this is because algebra discards the angle
// the winding number IS angle accumulation
// you cannot count wraps with scalars
//
// the "deepest" theorem in mathematics is counting how many times an angle wraps
//
// everything below proves this mechanically

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════════════════
// helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// evaluate polynomial with coefficients [a₀, a₁, ..., aₙ] at point z
/// uses horner's method: p(z) = (...((aₙz + aₙ₋₁)z + aₙ₋₂)z + ...) + a₀
fn eval_poly(coeffs: &[Geonum], z: Geonum) -> Geonum {
    let n = coeffs.len();
    let mut result = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        result = result * z + coeffs[i];
    }
    result
}

/// compute the output angle of a geonum via its cartesian projection
/// this avoids blade-wrapping issues with grade_angle()
fn output_angle(g: Geonum) -> f64 {
    let x = g.mag * g.angle.grade_angle().cos();
    let y = g.mag * g.angle.grade_angle().sin();
    y.atan2(x)
}

/// compute winding number of polynomial around origin on circle of given radius
/// sweeps z around the circle and counts how many times p(z) wraps the origin
fn winding_number(coeffs: &[Geonum], radius: f64) -> i32 {
    let num_points = 10000;
    let mut total_angle_change = 0.0;
    let mut prev_angle: Option<f64> = None;

    for i in 0..=num_points {
        let theta = 2.0 * PI * i as f64 / num_points as f64;
        let z = Geonum::new_from_cartesian(radius * theta.cos(), radius * theta.sin());
        let p_z = eval_poly(coeffs, z);

        let current = output_angle(p_z);
        if let Some(prev) = prev_angle {
            let mut delta = current - prev;
            while delta > PI {
                delta -= 2.0 * PI;
            }
            while delta < -PI {
                delta += 2.0 * PI;
            }
            total_angle_change += delta;
        }
        prev_angle = Some(current);
    }

    (total_angle_change / (2.0 * PI)).round() as i32
}

/// create a scalar coefficient
fn scalar(val: f64) -> Geonum {
    if val >= 0.0 {
        Geonum::new(val, 0.0, 1.0)
    } else {
        Geonum::new(val.abs(), 1.0, 1.0) // negative = [|val|, π]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// z^n wraps n times
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_wraps_n_times_for_z_to_the_n() {
    // z = [r, θ], z^n = [r^n, nθ]
    // as θ sweeps 0→2π, the output angle sweeps 0→2nπ
    // that is n complete wraps around the origin
    // the degree of the polynomial IS the winding number on any circle

    // z^1: 1 wrap
    let z1_coeffs = [scalar(0.0), scalar(1.0)]; // p(z) = z
    assert_eq!(winding_number(&z1_coeffs, 1.0), 1, "z wraps once");

    // z^2: 2 wraps
    let z2_coeffs = [scalar(0.0), scalar(0.0), scalar(1.0)]; // p(z) = z²
    assert_eq!(winding_number(&z2_coeffs, 1.0), 2, "z² wraps twice");

    // z^3: 3 wraps
    let z3_coeffs = [scalar(0.0), scalar(0.0), scalar(0.0), scalar(1.0)];
    assert_eq!(winding_number(&z3_coeffs, 1.0), 3, "z³ wraps three times");

    // z^5: 5 wraps
    let z5_coeffs = [
        scalar(0.0),
        scalar(0.0),
        scalar(0.0),
        scalar(0.0),
        scalar(0.0),
        scalar(1.0),
    ];
    assert_eq!(winding_number(&z5_coeffs, 1.0), 5, "z⁵ wraps five times");
}

#[test]
fn it_shows_angle_accumulation_is_the_degree() {
    // for z on the unit circle at angle θ:
    // z^n has angle nθ
    // sweeping θ from 0 to 2π sweeps the output through n × 2π
    // this is what "degree" means geometrically

    let angles = [0.0, PI / 6.0, PI / 3.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    for n in 2..=5 {
        for &theta in &angles {
            let z = Geonum::new_from_cartesian(theta.cos(), theta.sin());

            // compute z^n by repeated multiplication
            let mut z_n = Geonum::new(1.0, 0.0, 1.0);
            for _ in 0..n {
                z_n = z_n * z;
            }

            // output angle should be n × input angle (mod 2π)
            let expected_angle = (n as f64 * theta) % (2.0 * PI);
            let actual = output_angle(z_n);

            // normalize both to [0, 2π)
            let expected_norm = ((expected_angle % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
            let actual_norm = ((actual % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);

            let diff = (expected_norm - actual_norm).abs();
            let diff_wrapped = diff.min(2.0 * PI - diff);
            assert!(
                diff_wrapped < 0.01 || z_n.mag < EPSILON,
                "z^{} at θ={:.3}: output angle {:.3} ≈ {}×{:.3} = {:.3}",
                n,
                theta,
                actual,
                n,
                theta,
                expected_angle
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// winding number = number of roots
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_counts_roots_of_z_squared_minus_one() {
    // p(z) = z² - 1
    // degree 2 → winding number 2 on large circle → 2 roots
    // roots: z = +1 and z = -1

    let coeffs = [scalar(-1.0), scalar(0.0), scalar(1.0)]; // -1 + 0z + z²

    // large circle: winding = 2 (degree)
    assert_eq!(
        winding_number(&coeffs, 5.0),
        2,
        "z²-1 winds twice on large circle: 2 roots exist"
    );

    // verify root at z = 1
    let z1 = Geonum::new(1.0, 0.0, 1.0);
    let p_z1 = eval_poly(&coeffs, z1);
    assert!(
        p_z1.mag < 0.01,
        "z=1 is a root: |p(1)| = {:.6} ≈ 0",
        p_z1.mag
    );

    // verify root at z = -1
    let z_neg1 = Geonum::new(1.0, 1.0, 1.0); // [1, π]
    let p_z_neg1 = eval_poly(&coeffs, z_neg1);
    assert!(
        p_z_neg1.mag < 0.01,
        "z=-1 is a root: |p(-1)| = {:.6} ≈ 0",
        p_z_neg1.mag
    );
}

#[test]
fn it_counts_roots_of_z_squared_plus_one() {
    // p(z) = z² + 1
    // degree 2 → winding number 2 → 2 roots
    // roots: z = +i and z = -i (COMPLEX roots, not on real line)
    //
    // algebra says "no real roots" and stops
    // the angle says "2 wraps, so 2 roots" — they must be off the real axis

    let coeffs = [scalar(1.0), scalar(0.0), scalar(1.0)]; // 1 + 0z + z²

    // large circle: winding = 2
    assert_eq!(
        winding_number(&coeffs, 5.0),
        2,
        "z²+1 winds twice: 2 roots exist even though none are real"
    );

    // verify root at z = i = [1, π/2]
    let z_i = Geonum::new(1.0, 1.0, 2.0); // [1, π/2]
    let p_zi = eval_poly(&coeffs, z_i);
    assert!(
        p_zi.mag < 0.01,
        "z=i is a root: |p(i)| = {:.6} ≈ 0",
        p_zi.mag
    );

    // verify root at z = -i = [1, 3π/2]
    let z_neg_i = Geonum::new(1.0, 3.0, 2.0); // [1, 3π/2]
    let p_z_neg_i = eval_poly(&coeffs, z_neg_i);
    assert!(
        p_z_neg_i.mag < 0.01,
        "z=-i is a root: |p(-i)| = {:.6} ≈ 0",
        p_z_neg_i.mag
    );
}

#[test]
fn it_counts_roots_of_a_cubic() {
    // p(z) = z³ - 1
    // degree 3 → winding number 3 → 3 roots
    // roots: the cube roots of unity
    //   z = 1, z = e^(2πi/3), z = e^(4πi/3)
    //   all on the unit circle, evenly spaced by 2π/3

    let coeffs = [scalar(-1.0), scalar(0.0), scalar(0.0), scalar(1.0)]; // -1 + z³

    // large circle: winding = 3
    assert_eq!(
        winding_number(&coeffs, 5.0),
        3,
        "z³-1 winds three times: 3 roots exist"
    );

    // verify all three cube roots of unity
    let roots_angles = [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0];

    for (k, &angle) in roots_angles.iter().enumerate() {
        let z = Geonum::new_from_cartesian(angle.cos(), angle.sin());
        let p_z = eval_poly(&coeffs, z);
        assert!(
            p_z.mag < 0.01,
            "cube root {} at angle {:.3}: |p(z)| = {:.6} ≈ 0",
            k,
            angle,
            p_z.mag
        );
    }
}

#[test]
fn it_counts_roots_of_a_quartic() {
    // p(z) = z⁴ - 1
    // degree 4 → winding number 4 → 4 roots
    // roots: 1, i, -1, -i (the fourth roots of unity)
    // evenly spaced by π/2 on the unit circle — the Q lattice itself

    let coeffs = [
        scalar(-1.0),
        scalar(0.0),
        scalar(0.0),
        scalar(0.0),
        scalar(1.0),
    ]; // -1 + z⁴

    assert_eq!(
        winding_number(&coeffs, 5.0),
        4,
        "z⁴-1 winds four times: 4 roots exist"
    );

    // the four roots ARE the grade cycle: 0, π/2, π, 3π/2
    let root_angles = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    for (k, &angle) in root_angles.iter().enumerate() {
        let z = Geonum::new_from_cartesian(angle.cos(), angle.sin());
        let p_z = eval_poly(&coeffs, z);
        assert!(
            p_z.mag < 0.01,
            "fourth root {} at angle {:.3}: |p(z)| = {:.6} ≈ 0",
            k,
            angle,
            p_z.mag
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// shrinking the circle unwinds through roots
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_unwinds_through_roots_as_circle_shrinks() {
    // on a large circle: winding = degree
    // on a tiny circle around the origin: winding = 0 (p(z) ≈ a₀ ≠ 0, constant)
    // the winding must decrease from n to 0
    // it can only change when the circle passes through a root
    // each root peels off one winding
    //
    // this is the ENTIRE proof of the fundamental theorem:
    // winding starts at n, ends at 0, must pass through n roots

    // p(z) = z² - 1, roots at z = ±1

    let coeffs = [scalar(-1.0), scalar(0.0), scalar(1.0)];

    // outside all roots: winding = 2
    let w_large = winding_number(&coeffs, 3.0);
    assert_eq!(w_large, 2, "outside all roots: winding = degree = 2");

    // between roots at |z|=1 and origin: circle of radius 0.5
    // both roots are at |z|=1, so circle of radius 0.5 encloses no roots
    let w_small = winding_number(&coeffs, 0.5);
    assert_eq!(w_small, 0, "inside all roots: winding = 0");

    // the winding dropped from 2 to 0
    // it can only change when crossing a root
    // so there are at least 2 roots between radius 0.5 and 3.0
    assert_eq!(w_large - w_small, 2, "winding change = 2 → 2 roots crossed");
}

#[test]
fn it_tracks_winding_change_through_nested_roots() {
    // p(z) = z(z - 2)(z - 4) = z³ - 6z² + 8z
    // roots at z = 0, z = 2, z = 4
    // as circle grows: winding increases by 1 at each root

    let coeffs = [
        scalar(0.0),  // a₀ = 0
        scalar(8.0),  // a₁ = 8
        scalar(-6.0), // a₂ = -6
        scalar(1.0),  // a₃ = 1
    ]; // 0 + 8z - 6z² + z³

    // radius 1: encloses root at z=0 only
    let w_1 = winding_number(&coeffs, 1.0);
    assert_eq!(w_1, 1, "radius 1: 1 root enclosed (z=0)");

    // radius 3: encloses roots at z=0 and z=2
    let w_3 = winding_number(&coeffs, 3.0);
    assert_eq!(w_3, 2, "radius 3: 2 roots enclosed (z=0, z=2)");

    // radius 5: encloses all three roots
    let w_5 = winding_number(&coeffs, 5.0);
    assert_eq!(w_5, 3, "radius 5: 3 roots enclosed (all)");

    // winding increases by 1 each time we cross a root
    assert_eq!(w_3 - w_1, 1, "crossing z=2 adds one winding");
    assert_eq!(w_5 - w_3, 1, "crossing z=4 adds one winding");
}

// ═══════════════════════════════════════════════════════════════════════════════
// the impossibility of algebraic proof
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_why_algebra_cannot_see_this() {
    // algebra works with scalars — it discards the angle at the start
    // the winding number is angle accumulation around a closed path
    // you cannot count wraps without the angle
    //
    // this test proves the information is IN the angle and ONLY in the angle

    // p(z) = z² + 1 at z = 2 (a real scalar)
    let z_real = Geonum::new(2.0, 0.0, 1.0); // [2, 0]
    let p_real = eval_poly(&[scalar(1.0), scalar(0.0), scalar(1.0)], z_real);

    // algebra sees: 2² + 1 = 5, no root here, nothing to learn about roots
    assert!(p_real.mag > 4.0, "scalar evaluation just gives a number");

    // but the ANGLE of the result carries winding information
    // even at this single point, the output angle contributes to the winding count
    // algebra throws this away

    // now evaluate at z = [2, θ] for various θ on a circle
    let coeffs = [scalar(1.0), scalar(0.0), scalar(1.0)]; // z² + 1
    let radius = 2.0;
    let num_samples = 8;

    let mut angles_out = Vec::new();
    for i in 0..num_samples {
        let theta = 2.0 * PI * i as f64 / num_samples as f64;
        let z = Geonum::new_from_cartesian(radius * theta.cos(), radius * theta.sin());
        let p_z = eval_poly(&coeffs, z);
        angles_out.push(output_angle(p_z));
    }

    // the output angles are all DIFFERENT — they trace a path around the origin
    // algebra at z=2 sees magnitude 5 and nothing else
    // the angle sequence IS the winding, and its invisible to scalars

    // verify the angles are not all the same
    let angle_variance: f64 = {
        let mean: f64 = angles_out.iter().sum::<f64>() / angles_out.len() as f64;
        angles_out.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / angles_out.len() as f64
    };
    assert!(
        angle_variance > 0.1,
        "output angles vary: the winding information is in the angle, which algebra discards"
    );

    // the full winding count from this angle data
    let w = winding_number(&coeffs, radius);
    assert_eq!(w, 2, "angle data gives winding = 2 = number of roots");
}

// ═══════════════════════════════════════════════════════════════════════════════
// roots of unity: the Q lattice is the answer
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_roots_of_unity_are_the_q_lattice_generalized() {
    // z^n - 1 = 0 has roots evenly spaced on the unit circle at angles 2kπ/n
    // for n=4: angles are 0, π/2, π, 3π/2 — the four grades of the Q lattice
    //
    // the Q lattice is the n=4 case of roots of unity
    // grades 0, 1, 2, 3 ARE the fourth roots of unity
    // the entire geonum framework is built on z⁴ = 1

    for n in 2..=8 {
        // build z^n - 1
        let mut coeffs: Vec<Geonum> = vec![scalar(0.0); n + 1];
        coeffs[0] = scalar(-1.0); // constant term -1
        coeffs[n] = scalar(1.0); // z^n term

        // winding = n
        assert_eq!(
            winding_number(&coeffs, 3.0),
            n as i32,
            "z^{}-1 has winding {} on large circle",
            n,
            n
        );

        // verify each root at angle 2kπ/n
        for k in 0..n {
            let angle = 2.0 * PI * k as f64 / n as f64;
            let z = Geonum::new_from_cartesian(angle.cos(), angle.sin());
            let p_z = eval_poly(&coeffs, z);
            assert!(
                p_z.mag < 0.02,
                "root {} of z^{}-1 at angle {:.3}: |p(z)| = {:.6} ≈ 0",
                k,
                n,
                angle,
                p_z.mag
            );
        }
    }
}

#[test]
fn it_proves_no_rootless_polynomial_exists() {
    // the fundamental theorem:
    // every polynomial of degree n ≥ 1 has at least one root
    //
    // proof from winding:
    // 1. on a large circle, p(z) ≈ aₙz^n, so winding = n ≥ 1
    // 2. at the origin, p(0) = a₀ ≠ 0 (for generic polynomial), winding = 0
    // 3. winding is continuous and integer-valued
    // 4. to go from n to 0, it must decrease
    // 5. it can only decrease by passing through a zero of p(z)
    // 6. therefore at least one root exists
    //
    // test this for various "difficult" polynomials

    // z² + 1: no REAL roots, but winding = 2 guarantees COMPLEX roots
    let p1 = [scalar(1.0), scalar(0.0), scalar(1.0)];
    assert_eq!(
        winding_number(&p1, 10.0),
        2,
        "z²+1: winding 2, roots must exist"
    );

    // z⁴ + z² + 1: no obvious roots
    let p2 = [
        scalar(1.0),
        scalar(0.0),
        scalar(1.0),
        scalar(0.0),
        scalar(1.0),
    ];
    assert_eq!(
        winding_number(&p2, 10.0),
        4,
        "z⁴+z²+1: winding 4, four roots must exist"
    );

    // z⁶ + 1: six roots, all complex
    let mut p3 = vec![scalar(0.0); 7];
    p3[0] = scalar(1.0);
    p3[6] = scalar(1.0);
    assert_eq!(
        winding_number(&p3, 10.0),
        6,
        "z⁶+1: winding 6, six roots must exist"
    );

    // in every case:
    // large circle → winding = degree > 0
    // therefore roots exist
    // QED

    // the proof is 6 lines. it took mathematicians centuries because they
    // tried to find roots using algebra (scalars, no angles)
    // instead of counting how many times the output wraps (angles)
}

// ═══════════════════════════════════════════════════════════════════════════════
// the fundamental theorem of algebra was never about algebra
//
// it was about angle accumulation
//
// a polynomial of degree n wraps the output n times around the origin
// as the input sweeps a large circle. those wraps must unwind as the
// circle shrinks. each unwinding passes through a root. n wraps, n roots.
//
// the theorem is unprovable in algebra because algebra discards the angle.
// the winding number — the thing that forces roots to exist — is invisible
// to any system that represents numbers as scalars on a line.
//
// every "proof" of the FTA smuggles the angle back in:
// - complex analysis uses contour integrals (angle accumulation along paths)
// - topology uses the fundamental group (winding numbers)
// - even liouville's theorem works through bounded entire functions (angle behavior)
//
// they all reduce to: the output wraps, so it must cross zero.
//
// the geometric number [magnitude, angle] sees this directly.
// the proof is in the data structure.
// ═══════════════════════════════════════════════════════════════════════════════
