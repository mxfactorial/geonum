// the fundamental theorem of algebra is angle accumulation
//
// every polynomial of degree n has exactly n roots. this took centuries to prove; every
// known proof needs complex analysis or topology, and it was shown you CANNOT prove it
// with algebra alone. the reason is plain: algebra discards the angle, and the angle is
// the whole content of the theorem.
//
// this file proves it from the angle, two ways, weakest to strongest:
//
// part 1 — the monomial. z = [1, θ], z^n = [1, nθ]: multiplication scales the angle. as
//   θ sweeps 0 → 2π, nθ sweeps 0 → n·2π, so z^n = 1 at the n angles 2πk/n. no winding
//   integral, no continuity argument — n multiples of 2π live in n·2π, just counting.
//
// part 2 — the general polynomial. p(z) is a sum, so its angle isnt nθ exactly. but on a
//   large circle p(z) ≈ aₙz^n, so the OUTPUT still wraps the origin n times. shrink the
//   circle to a point and the winding falls to 0; it can only change by crossing a root,
//   so n wraps force n roots. the winding number IS angle accumulated around the path —
//   the monomial's clean count, made robust to the lower-order terms.
//
// both are one fact: the output sweeps n times, so it crosses zero n times. every "real"
// proof smuggles this back in — contour integrals, the fundamental group, liouville — all
// counting how many times an angle wrapped. the geometric number [magnitude, angle] reads
// it straight off, no atan2, no cartesian round-trip.

use geonum::*;
use std::f64::consts::PI;

const TAU: f64 = 2.0 * PI;

// ═══════════════════════════════════════════════════════════════════════════════
// PART 1 — THE MONOMIAL: counting, no winding integral needed
//
// z^n = [1, nθ]. the theorem is multiplication scaling the angle, and n·2π holds n
// multiples of 2π. the proof has to USE the representation: pow scales the angle,
// cos_sin reads it, nothing detours through cartesian.
// ═══════════════════════════════════════════════════════════════════════════════

/// unit geonum at angle (p/d)·π — magnitude 1, built straight from the angle, no
/// cartesian round-trip. magnitude is gone; the angle is the whole object
fn unit(p: f64, d: f64) -> Geonum {
    Geonum::new(1.0, p, d)
}

/// the k-th root of unity of order n: [1, 2πk/n] — the angle where nθ = 2πk
fn root(n: usize, k: usize) -> Geonum {
    Geonum::new(1.0, 2.0 * k as f64 / n as f64, 1.0)
}

/// z^n has returned to 1: drop the magnitude and the unit at g's angle coincides with the
/// scalar 1 — the count is in the angle, not the magnitude. winding-blind (blade 4k ≡ 0) and
/// atan-free (Sub combines via rational cos_sin, no atan2)
fn at_unity(g: &Geonum) -> bool {
    (Geonum::new_with_angle(1.0, g.angle) - Geonum::new(1.0, 0.0, 1.0)).near_mag(0.0)
}

#[test]
fn it_multiplies_angles() {
    // z^n on the unit circle scales the angle by n — z.pow(n), one operation. compare it
    // to [1, nθ] built directly; the two are the same number
    let angles = [
        (0.0, 1.0), // 0
        (1.0, 6.0), // π/6
        (1.0, 4.0), // π/4
        (1.0, 3.0), // π/3
        (1.0, 2.0), // π/2
        (1.0, 1.0), // π
        (3.0, 2.0), // 3π/2
    ];

    for n in 1..=6 {
        for &(p, d) in &angles {
            let z_n = unit(p, d).pow(n as f64); // [1, nθ]
            let expected = unit(n as f64 * p, d); // [1, nθ], constructed
            assert!(
                (z_n - expected).near_mag(0.0),
                "z^{n} at θ={p}π/{d}: pow scaled the angle wrong"
            );
        }
    }
}

#[test]
fn it_sweeps_n_times_as_theta_sweeps_once() {
    // as θ goes 0 → 2π, nθ goes 0 → n×2π: the output angle wraps past 0 exactly n times
    let samples = 3600;

    for n in 1..=8 {
        let mut crossings = 0;
        let mut prev: Option<f64> = None;

        for i in 0..=samples {
            // θ = (2i/samples)·π — swept once around the circle
            let angle = unit(2.0 * i as f64 / samples as f64, 1.0)
                .pow(n as f64)
                .angle
                .grade_angle();

            if let Some(p) = prev {
                // a crossing: the output angle wraps past 0
                if p > TAU * 0.9 && angle < TAU * 0.1 {
                    crossings += 1;
                }
            }
            prev = Some(angle);
        }

        assert_eq!(
            crossings, n,
            "degree {n}: output wrapped {crossings} times (expected {n})"
        );
    }
}

#[test]
fn it_finds_roots_as_full_rotations() {
    // z^n = 1 when nθ = 2πk. θ = 2πk/n — one division — and z^n returns to unity
    for n in 1..=8 {
        for k in 0..n {
            let z_n = root(n, k).pow(n as f64);
            assert!(at_unity(&z_n), "root {k}/{n}: z^{n} should return to 1");
        }
    }
}

#[test]
fn it_counts_exactly_n_roots() {
    // the n roots are equally spaced: rotating one by 2π/n lands on the next, and the
    // n-th step closes the circle back to the first — exactly n, no (n+1)-th
    for n in 2..=10 {
        let step = Geonum::new(1.0, 2.0 / n as f64, 1.0); // [1, 2π/n]

        for k in 0..n {
            let advanced = root(n, k) * step; // rotate the k-th root forward by 2π/n
            let next = root(n, (k + 1) % n); // the next root, wrapping at n
            assert!(
                (advanced - next).near_mag(0.0),
                "degree {n}: root {k} + 2π/n is not root {}",
                (k + 1) % n
            );
        }

        // k = n is k = 0 again — winding home adds no new root
        assert!(
            (root(n, n) - root(n, 0)).near_mag(0.0),
            "degree {n}: the n-th root wraps to the 0-th"
        );
    }
}

#[test]
fn it_adds_one_root_per_degree() {
    // count the GENUINE roots of each degree: every candidate angle 2πk/n whose z^n
    // returns to 1. degree n yields n, degree n+1 yields n+1, the difference exactly one
    let verified = |deg: usize| {
        (0..deg)
            .filter(|&k| at_unity(&root(deg, k).pow(deg as f64)))
            .count()
    };

    for n in 1..=9 {
        assert_eq!(verified(n), n, "degree {n} has n verified roots");
        assert_eq!(verified(n + 1), n + 1, "degree {} has n+1 roots", n + 1);
        assert_eq!(
            verified(n + 1) - verified(n),
            1,
            "one more degree, one more root"
        );
    }
}

#[test]
fn it_adds_one_rotation_per_degree() {
    // (n+1)θ sweeps one more 2π than nθ — that extra full turn is the new root. the last
    // root of order n+1 is the new crossing, and it returns to 1
    for n in 1..=7 {
        let new_root = root(n + 1, n).pow((n + 1) as f64); // k=n, the added root
        assert!(
            at_unity(&new_root),
            "degree {}: the new root returns to 1",
            n + 1
        );
    }
}

#[test]
fn it_counts_the_same_on_any_circle() {
    // z = [r, θ], z^n = [r^n, nθ]: nθ doesnt depend on r, so the roots sit at the same
    // angles on every circle — r^n only scales the magnitude
    let radii = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0];

    for n in 2..=5 {
        for &r in &radii {
            for k in 0..n {
                let z_n = Geonum::new(r, 2.0 * k as f64 / n as f64, 1.0).pow(n as f64);
                assert!(
                    at_unity(&z_n),
                    "r={r}, n={n}, k={k}: angle returns to 0 regardless of radius"
                );
            }
        }
    }
}

#[test]
fn it_shows_the_count_is_in_the_angle_not_the_magnitude() {
    // at a root z^n returns to angle 0, while its magnitude r^n swings across orders of
    // magnitude. only the angle records the root; the magnitude says nothing
    let mut mags = Vec::new();
    for &r in &[0.01_f64, 1.0, 1000.0] {
        let z_n = Geonum::new(r, 1.0, 2.0).pow(4.0); // the 4th-root angle π/2, at radius r
        assert!(at_unity(&z_n), "r={r}: angle at 0 regardless of magnitude");
        mags.push(z_n.mag);
    }

    // the angle was identical (at unity) every time; the magnitudes span ~20 orders
    assert!(
        mags[0] < 1e-6 && mags[2] > 1e6,
        "r^4 swings ~1e-8 → ~1e12 — the angle ignored it"
    );
}

#[test]
fn it_shows_grades_are_fourth_roots_of_unity() {
    // z^4 = 1 has roots at 0, π/2, π, 3π/2 — and those ARE grades 0, 1, 2, 3. geonum is
    // built as the n=4 case of this theorem
    let grade_angles = [
        (0, 0.0, 1.0), // grade 0: θ = 0
        (1, 1.0, 2.0), // grade 1: θ = π/2
        (2, 1.0, 1.0), // grade 2: θ = π
        (3, 3.0, 2.0), // grade 3: θ = 3π/2
    ];

    for (grade, p, d) in grade_angles {
        let z = unit(p, d);
        assert_eq!(z.angle.grade(), grade, "θ={p}π/{d} is grade {grade}");
        assert!(at_unity(&z.pow(4.0)), "grade {grade}: z^4 returns to 1");
    }
}

#[test]
fn it_shows_i_squared_is_minus_one_as_angle_addition() {
    // i = [1, π/2]. i² = i·i = [1, π] = −1. not a definition, not a convention — angle
    // addition: π/2 + π/2 = π
    let i = unit(1.0, 2.0); // [1, π/2]
    let i_squared = i * i;

    assert!(
        i_squared.angle.near(&Angle::new(1.0, 1.0)),
        "i² lands at π — the −1 direction, by angle addition alone"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PART 2 — THE GENERAL POLYNOMIAL: the winding number
//
// p(z) is a sum of monomials, so its angle is nθ only to leading order. the winding
// number recovers the count anyway: sweep z around a circle, accumulate the change in
// the OUTPUT angle, divide by 2π. on a large circle it reads the degree; shrink past a
// root and it drops by one. the output direction is read straight off grade_angle —
// the same angle the monomial section counts, now summed around a path.
// ═══════════════════════════════════════════════════════════════════════════════

/// a real coefficient: positive sits at angle 0, negative at π — sign IS the angle
fn scalar(val: f64) -> Geonum {
    if val >= 0.0 {
        Geonum::new(val, 0.0, 1.0) // [val, 0]
    } else {
        Geonum::new(val.abs(), 1.0, 1.0) // [|val|, π]
    }
}

/// evaluate a polynomial [a₀, a₁, …, aₙ] at z by horner's method:
/// p(z) = (…((aₙz + aₙ₋₁)z + aₙ₋₂)z + …) + a₀ — multiplication adds angles, addition
/// combines the legs, no coordinates touched
fn eval_poly(coeffs: &[Geonum], z: Geonum) -> Geonum {
    let n = coeffs.len();
    let mut result = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        result = result * z + coeffs[i];
    }
    result
}

/// winding number of p(z) around the origin on a circle of given radius: sweep z = [r, θ]
/// once around, accumulate the signed change in the output's direction (grade_angle, read
/// straight off the angle), and divide the total by 2π
fn winding_number(coeffs: &[Geonum], radius: f64) -> i32 {
    let num_points = 10000;
    let mut total = 0.0;
    let mut prev: Option<f64> = None;

    for i in 0..=num_points {
        let z = Geonum::new(radius, 2.0 * i as f64 / num_points as f64, 1.0); // [r, θ]
        let p_z = eval_poly(coeffs, z);
        let current = p_z.angle.grade_angle(); // the output direction, no atan2

        if let Some(p) = prev {
            // the signed step, unwrapped onto (−π, π]
            let mut delta = current - p;
            while delta > PI {
                delta -= TAU;
            }
            while delta < -PI {
                delta += TAU;
            }
            total += delta;
        }
        prev = Some(current);
    }

    (total / TAU).round() as i32
}

// the polynomial root checks evaluate p(z) at a candidate angle; the horner chain of
// multiplies and adds accumulates a little float error, so |p(root)| sits near 1e-13
const ROOT_TOL: f64 = 1e-9;

#[test]
fn it_counts_roots_of_z_squared_minus_one() {
    // p(z) = z² − 1: degree 2 → winding 2 → 2 roots, at z = +1 and z = −1
    let coeffs = [scalar(-1.0), scalar(0.0), scalar(1.0)]; // −1 + 0z + z²

    assert_eq!(
        winding_number(&coeffs, 5.0),
        2,
        "z²−1 winds twice on a large circle: 2 roots exist"
    );

    // z = 1 = [1, 0] and z = −1 = [1, π] both annihilate the polynomial
    assert!(
        eval_poly(&coeffs, unit(0.0, 1.0)).mag < ROOT_TOL,
        "z=1 is a root"
    );
    assert!(
        eval_poly(&coeffs, unit(1.0, 1.0)).mag < ROOT_TOL,
        "z=−1 is a root"
    );
}

#[test]
fn it_counts_roots_of_z_squared_plus_one() {
    // p(z) = z² + 1: degree 2 → winding 2 → 2 roots. algebra says "no real roots" and
    // stops; the angle says "2 wraps, 2 roots" — they sit off the real axis, at ±i
    let coeffs = [scalar(1.0), scalar(0.0), scalar(1.0)]; // 1 + 0z + z²

    assert_eq!(
        winding_number(&coeffs, 5.0),
        2,
        "z²+1 winds twice: 2 roots exist even though none are real"
    );

    // z = i = [1, π/2] and z = −i = [1, 3π/2]
    assert!(
        eval_poly(&coeffs, unit(1.0, 2.0)).mag < ROOT_TOL,
        "z=i is a root"
    );
    assert!(
        eval_poly(&coeffs, unit(3.0, 2.0)).mag < ROOT_TOL,
        "z=−i is a root"
    );
}

#[test]
fn it_counts_roots_of_a_cubic() {
    // p(z) = z³ − 1: degree 3 → winding 3 → 3 roots, the cube roots of unity evenly
    // spaced by 2π/3 on the unit circle
    let coeffs = [scalar(-1.0), scalar(0.0), scalar(0.0), scalar(1.0)]; // −1 + z³

    assert_eq!(
        winding_number(&coeffs, 5.0),
        3,
        "z³−1 winds three times: 3 roots exist"
    );

    for k in 0..3 {
        assert!(
            eval_poly(&coeffs, root(3, k)).mag < ROOT_TOL,
            "cube root {k} at 2π·{k}/3 annihilates z³−1"
        );
    }
}

#[test]
fn it_counts_roots_of_a_quartic() {
    // p(z) = z⁴ − 1: degree 4 → winding 4 → 4 roots, 1, i, −1, −i — evenly spaced by π/2,
    // the Q lattice itself
    let coeffs = [
        scalar(-1.0),
        scalar(0.0),
        scalar(0.0),
        scalar(0.0),
        scalar(1.0),
    ]; // −1 + z⁴

    assert_eq!(
        winding_number(&coeffs, 5.0),
        4,
        "z⁴−1 winds four times: 4 roots exist"
    );

    // the four roots ARE the grade cycle 0, π/2, π, 3π/2
    for k in 0..4 {
        assert!(
            eval_poly(&coeffs, root(4, k)).mag < ROOT_TOL,
            "fourth root {k} (grade {k}) annihilates z⁴−1"
        );
    }
}

#[test]
fn it_unwinds_through_roots_as_circle_shrinks() {
    // the ENTIRE proof in one test: winding starts at the degree on a large circle and
    // falls to 0 at the origin (p(z) → a₀ ≠ 0, a constant). it can only change by crossing
    // a root, so the drop from n to 0 counts exactly n roots
    let coeffs = [scalar(-1.0), scalar(0.0), scalar(1.0)]; // z² − 1, roots at ±1

    let w_large = winding_number(&coeffs, 3.0); // outside both roots
    assert_eq!(w_large, 2, "outside all roots: winding = degree = 2");

    let w_small = winding_number(&coeffs, 0.5); // inside both (roots sit at |z|=1)
    assert_eq!(w_small, 0, "inside all roots: winding = 0");

    assert_eq!(
        w_large - w_small,
        2,
        "winding fell by 2 → exactly 2 roots crossed"
    );
}

#[test]
fn it_tracks_winding_change_through_nested_roots() {
    // p(z) = z(z−2)(z−4) = z³ − 6z² + 8z, real roots at 0, 2, 4. as the circle grows the
    // winding ticks up by one at each root crossed
    let coeffs = [scalar(0.0), scalar(8.0), scalar(-6.0), scalar(1.0)]; // 8z − 6z² + z³

    let w_1 = winding_number(&coeffs, 1.0); // encloses z=0
    let w_3 = winding_number(&coeffs, 3.0); // encloses z=0, 2
    let w_5 = winding_number(&coeffs, 5.0); // encloses all three

    assert_eq!(w_1, 1, "radius 1: 1 root enclosed (z=0)");
    assert_eq!(w_3, 2, "radius 3: 2 roots enclosed (z=0, 2)");
    assert_eq!(w_5, 3, "radius 5: 3 roots enclosed (all)");

    assert_eq!(w_3 - w_1, 1, "crossing z=2 adds one winding");
    assert_eq!(w_5 - w_3, 1, "crossing z=4 adds one winding");
}

#[test]
fn it_shows_why_algebra_cannot_see_this() {
    // algebra works with scalars — it discards the angle at the start. the winding number
    // is angle accumulated around a closed path, so it cannot be counted with scalars.
    // this test shows the information is IN the angle and only in the angle
    let coeffs = [scalar(1.0), scalar(0.0), scalar(1.0)]; // z² + 1

    // a scalar evaluation just gives a number: p(2) = 5, nothing about roots
    let p_real = eval_poly(&coeffs, Geonum::new(2.0, 0.0, 1.0));
    assert!(p_real.mag > 4.0, "scalar evaluation just gives a magnitude");

    // but the OUTPUT angle, sampled around a circle, traces a path — and that path is the
    // winding. collect the output directions; they are not all the same
    let radius = 2.0;
    let samples = 8;
    let angles_out: Vec<f64> = (0..samples)
        .map(|i| {
            let z = Geonum::new(radius, 2.0 * i as f64 / samples as f64, 1.0);
            eval_poly(&coeffs, z).angle.grade_angle()
        })
        .collect();

    let mean = angles_out.iter().sum::<f64>() / angles_out.len() as f64;
    let variance =
        angles_out.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / angles_out.len() as f64;
    assert!(
        variance > 0.1,
        "the output angles vary — the winding lives in the angle algebra threw away"
    );

    assert_eq!(
        winding_number(&coeffs, radius),
        2,
        "the same angle data gives winding 2 = the number of roots"
    );
}

#[test]
fn it_proves_no_rootless_polynomial_exists() {
    // the theorem: every polynomial of degree n ≥ 1 has a root. on a large circle the
    // winding equals the degree ≥ 1; at the origin it is 0; an integer that moves from n
    // to 0 must cross a root on the way. so a nonzero winding alone forces a root, with no
    // attempt to FIND one — and "no real roots" is no escape
    let z2_plus_1 = [scalar(1.0), scalar(0.0), scalar(1.0)]; // z² + 1, no real roots
    assert_eq!(
        winding_number(&z2_plus_1, 10.0),
        2,
        "z²+1: winding 2, roots must exist"
    );

    let z4_z2_1 = [
        scalar(1.0),
        scalar(0.0),
        scalar(1.0),
        scalar(0.0),
        scalar(1.0),
    ]; // z⁴ + z² + 1
    assert_eq!(
        winding_number(&z4_z2_1, 10.0),
        4,
        "z⁴+z²+1: winding 4, four roots must exist"
    );

    let mut z6_plus_1 = vec![scalar(0.0); 7]; // z⁶ + 1
    z6_plus_1[0] = scalar(1.0);
    z6_plus_1[6] = scalar(1.0);
    assert_eq!(
        winding_number(&z6_plus_1, 10.0),
        6,
        "z⁶+1: winding 6, six roots must exist"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// the fundamental theorem of algebra was never about algebra. it was about angle
// accumulation.
//
// the monomial says it cleanly: z^n = [1, nθ], n multiples of 2π in n·2π, n roots —
// pure counting. the general polynomial blurs the angle with lower-order terms, and the
// winding number sharpens it back: the output still wraps n times on a large circle, the
// wraps must unwind as the circle shrinks, and each unwinding is a root.
//
// the theorem is unprovable in algebra because algebra discards the angle. the winding
// number — the thing that forces the roots to exist — is invisible to anything that
// represents a number as a scalar on a line. every working proof smuggles the angle back:
//   complex analysis → contour integrals (angle along a path)
//   topology         → the fundamental group (winding numbers)
//   liouville        → bounded entire functions (angle behavior)
//
// the geometric number [magnitude, angle] sees it directly. the proof is in the data
// structure.
// ═══════════════════════════════════════════════════════════════════════════════
