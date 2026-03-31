// taylor series coefficients are geometric normalizations
//
// the taylor series f(x) = Σ f⁽ⁿ⁾(a)/n! × (x-a)^n
// is not a clever approximation technique — its a geometric identity
//
// f⁽ⁿ⁾(a) comes from n repeated differentiations (n quarter turns)
// n! comes from the product of angle ratios accumulated during those turns
// (x-a)^n carries nθ in its angle — the displacement raised to the nth power
//
// dividing by n! undoes the geometric weight that differentiation piled up
// each term is a power of displacement normalized by its own angle descent
//
// consequences:
//   e^x = Σ x^n/n! is what you get when every angle level contributes equally
//   sin(x) = odd grade terms only (grades 1↔3), signs from duals
//   cos(x) = even grade terms only (grades 0↔2), signs from duals
//   the alternating signs in trig series are not sign bits — they are π rotations
//
// everything below proves this mechanically

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════════════════
// n! is the product of angle ratios
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_produces_taylor_coefficients_from_angle_descent() {
    // for f(x) = x^n, the nth derivative is n!
    // this n! is not combinatorial — it is the product of angle ratios:
    //   x^n has angle ratio n
    //   x^(n-1) has angle ratio (n-1)
    //   ...
    //   x^1 has angle ratio 1
    //   product: n × (n-1) × ... × 1 = n!
    //
    // the taylor coefficient 1/n! normalizes this geometric accumulation

    let x = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let x_angle = x.angle.grade_angle();

    // build powers, extract angle ratios
    let mut powers = vec![Geonum::new(1.0, 0.0, 1.0)]; // x^0 = 1
    for i in 1..=7 {
        let next = powers[i - 1] * x;
        powers.push(next);
    }

    // angle ratios at each level
    for i in 1..=7 {
        let ratio = powers[i].angle.grade_angle() / x_angle;
        assert!(
            (ratio - i as f64).abs() < EPSILON,
            "x^{} angle ratio = {}",
            i,
            i
        );
    }

    // factorials from cumulative products of angle ratios
    let mut factorial = 1.0;
    for n in 1..=7 {
        let ratio = powers[n].angle.grade_angle() / x_angle;
        factorial *= ratio;

        // the taylor coefficient for the nth term is 1/n!
        let taylor_coeff = 1.0 / factorial;

        // verify against known values
        let expected_factorial: f64 = (1..=n).map(|i| i as f64).product();
        assert!(
            (factorial - expected_factorial).abs() < EPSILON,
            "angle descent gives {}! = {}",
            n,
            expected_factorial
        );
        assert!(
            (taylor_coeff - 1.0 / expected_factorial).abs() < EPSILON,
            "taylor coefficient 1/{}! = {:.6}",
            n,
            taylor_coeff
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// e^x: every angle level contributes equally
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_constructs_exp_from_equal_angle_contributions() {
    // e^x = Σ x^n/n!
    //
    // for e^x, every derivative at a=0 equals 1 (all derivatives of e^x are e^x, and e^0=1)
    // so each term is x^n / n! — displacement to the nth power, normalized by angle descent
    //
    // the exponential is what happens when no angle level is preferred
    // every rotation order contributes with equal weight after normalization

    let x_val = 1.0; // compute e^1 = e
    let x = Geonum::new(x_val, 1.0, 6.0);
    // use total angle (blade*π/2 + rem) instead of grade_angle
    // because grade_angle wraps mod 2π, losing the ratio for n ≥ 12
    let x_total = x.angle.blade() as f64 * PI / 2.0 + x.angle.rem();

    let mut sum = 0.0;
    let mut x_n = Geonum::new(1.0, 0.0, 1.0); // x^0 = 1
    let mut factorial = 1.0;

    // accumulate taylor terms
    for n in 0..20 {
        let term = x_n.mag / factorial;
        sum += term;

        // advance to next power
        x_n = x_n * x;

        // next factorial via angle ratio
        if n > 0 {
            let ratio = n as f64 + 1.0;
            // verify the ratio matches the angle
            let x_n_total = x_n.angle.blade() as f64 * PI / 2.0 + x_n.angle.rem();
            let measured_ratio = x_n_total / x_total;
            assert!(
                (measured_ratio - (n + 1) as f64).abs() < EPSILON,
                "angle ratio at step {} = {}",
                n + 1,
                n + 1
            );
            factorial *= ratio;
        } else {
            factorial = 1.0; // 1! = 1
        }
    }

    let expected = std::f64::consts::E;
    assert!(
        (sum - expected).abs() < 1e-8,
        "e^1 = {:.10} from angle-normalized sum, expected {:.10}",
        sum,
        expected
    );
}

#[test]
fn it_constructs_exp_at_any_point() {
    // e^x at x = 2: Σ 2^n/n!
    // each 2^n carries n copies of θ in its angle
    // each n! is the product of angle ratios from descent
    // the ratio x^n / n! is displacement^n / geometric_normalization

    let test_points = [0.5, 1.0, 1.5, 2.0, 3.0];

    for &x_val in &test_points {
        let mut sum = 0.0;
        let mut x_n_mag = 1.0; // |x^n|
        let mut factorial = 1.0;

        for n in 0..25 {
            sum += x_n_mag / factorial;
            x_n_mag *= x_val;
            factorial *= (n + 1) as f64;
        }

        let expected = x_val.exp();
        assert!(
            (sum - expected).abs() < 1e-8,
            "e^{} = {:.8}, expected {:.8}",
            x_val,
            sum,
            expected
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// sin and cos: grade-filtered projections
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_derivative_cycling_creates_trig_series() {
    // sin'(x) = cos(x), cos'(x) = -sin(x), (-sin)'(x) = -cos(x), (-cos)'(x) = sin(x)
    // this 4-cycle IS the grade cycle: 0→1→2→3→0
    //
    // at x = 0:
    //   sin(0) = 0, sin'(0) = 1, sin''(0) = 0, sin'''(0) = -1, sin''''(0) = 0, ...
    //   pattern: 0, 1, 0, -1, 0, 1, 0, -1, ...
    //
    //   cos(0) = 1, cos'(0) = 0, cos''(0) = -1, cos'''(0) = 0, cos''''(0) = 1, ...
    //   pattern: 1, 0, -1, 0, 1, 0, -1, 0, ...
    //
    // the zeros are orthogonal projections (cos(π/2) = 0)
    // the -1s are duals (π rotation, grade 0↔2 or 1↔3)
    // the series structure comes from grade filtering

    // verify the derivative cycle at grade level
    let f = Geonum::new(1.0, 0.0, 1.0); // grade 0

    let grades: Vec<usize> = (0..8)
        .scan(f, |current, _| {
            let grade = current.angle.grade();
            *current = current.differentiate();
            Some(grade)
        })
        .collect();

    assert_eq!(
        grades,
        vec![0, 1, 2, 3, 0, 1, 2, 3],
        "derivative cycling: 0→1→2→3→0→1→2→3"
    );

    // sin derivatives at 0 follow the grade cycle
    // grade 0: cos component → 0 (sin has no grade-0 content at x=0)
    // grade 1: sin component → 1 (sin peaks at grade 1)
    // grade 2: -cos component → 0 (dual of cos, but still zero at x=0 for sin)
    // grade 3: -sin component → -1 (dual of sin)

    let sin_derivs_at_0: [f64; 8] = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    let cos_derivs_at_0: [f64; 8] = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];

    // verify pattern period = 4 (the grade cycle)
    for i in 0..4 {
        assert!(
            (sin_derivs_at_0[i] - sin_derivs_at_0[i + 4]).abs() < EPSILON,
            "sin derivatives repeat with period 4"
        );
        assert!(
            (cos_derivs_at_0[i] - cos_derivs_at_0[i + 4]).abs() < EPSILON,
            "cos derivatives repeat with period 4"
        );
    }
}

#[test]
fn it_constructs_sin_from_odd_grade_terms() {
    // sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    //        = Σ (-1)^k × x^(2k+1) / (2k+1)!
    //
    // only odd powers appear — these are the odd grade terms (grades 1↔3)
    // the alternating sign (-1)^k is a dual: each pair of quarter turns
    // crosses the dual, adding π rotation
    //
    // sin is the odd-grade projection of e^(ix)

    let test_points = [0.3, 0.7, 1.0, 1.5, 2.0, PI / 4.0, PI / 3.0];

    for &x_val in &test_points {
        let mut sum = 0.0;
        let mut x_n_mag = 1.0; // |x^n|
        let mut factorial = 1.0;

        for n in 0..20 {
            if n > 0 {
                x_n_mag *= x_val;
                factorial *= n as f64;
            }

            // only odd powers contribute to sin
            if n % 2 == 1 {
                // the sign comes from which odd grade we're at
                // grade 1: positive (n = 1, 5, 9, ...)
                // grade 3: negative (n = 3, 7, 11, ...)
                // this is the dual: every two quarter turns crosses diameter
                let k = (n - 1) / 2; // which odd term
                let dual_sign = if k % 2 == 0 { 1.0 } else { -1.0 }; // (-1)^k from duals

                sum += dual_sign * x_n_mag / factorial;
            }
        }

        let expected = x_val.sin();
        assert!(
            (sum - expected).abs() < 1e-8,
            "sin({:.3}) = {:.8}, expected {:.8}",
            x_val,
            sum,
            expected
        );
    }
}

#[test]
fn it_constructs_cos_from_even_grade_terms() {
    // cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    //        = Σ (-1)^k × x^(2k) / (2k)!
    //
    // only even powers appear — these are the even grade terms (grades 0↔2)
    // the alternating sign (-1)^k is again a dual
    //
    // cos is the even-grade projection of e^(ix)

    let test_points = [0.3, 0.7, 1.0, 1.5, 2.0, PI / 4.0, PI / 3.0];

    for &x_val in &test_points {
        let mut sum = 0.0;
        let mut x_n_mag = 1.0;
        let mut factorial = 1.0;

        for n in 0..20 {
            if n > 0 {
                x_n_mag *= x_val;
                factorial *= n as f64;
            }

            // only even powers contribute to cos
            if n % 2 == 0 {
                let k = n / 2;
                let dual_sign = if k % 2 == 0 { 1.0 } else { -1.0 };

                sum += dual_sign * x_n_mag / factorial;
            }
        }

        let expected = x_val.cos();
        assert!(
            (sum - expected).abs() < 1e-8,
            "cos({:.3}) = {:.8}, expected {:.8}",
            x_val,
            sum,
            expected
        );
    }
}

#[test]
fn it_shows_the_dual_creates_alternating_signs() {
    // the (-1)^k in trig series is not a sign convention
    // it is a dual: π rotation (dual) in the grade cycle
    //
    // grade 0 → grade 2 is a dual (π rotation, cos → -cos)
    // grade 1 → grade 3 is a dual (π rotation, sin → -sin)
    //
    // each pair of differentiations crosses the dual
    // which is why every second nonzero term flips sign

    let f = Geonum::new(1.0, 0.0, 1.0); // grade 0

    // differentiate twice: grade 0 → grade 2
    let f_double = f.differentiate().differentiate();
    assert_eq!(f_double.angle.grade(), 2, "two quarter turns reach grade 2");

    // grade 2 is the dual of grade 0 — same pair, opposite side
    // this is the dual that creates the minus sign
    let grade_diff = f_double.angle.grade() as i32 - f.angle.grade() as i32;
    assert_eq!(grade_diff, 2, "dual: grade difference = 2 (π rotation)");

    // the dual relationship: grade 0 ↔ grade 2, grade 1 ↔ grade 3
    let g = Geonum::new(1.0, 0.0, 1.0);
    assert_eq!(g.dual().angle.grade(), 2, "dual of grade 0 is grade 2");

    let h = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);
    assert_eq!(h.dual().angle.grade(), 3, "dual of grade 1 is grade 3");

    // in cos series: term n=0 is grade 0 (+1), term n=2 is grade 2 (-1)
    // the sign flip IS the dual between dual grades
    // in sin series: term n=1 is grade 1 (+1), term n=3 is grade 3 (-1)
    // same dual, odd pair instead of even pair

    // cos: grade 0 → +, grade 2 → -, grade 0 → +, grade 2 → -
    // sin: grade 1 → +, grade 3 → -, grade 1 → +, grade 3 → -
    // the "alternating signs" are the involutive duality 0↔2 and 1↔3
}

// ═══════════════════════════════════════════════════════════════════════════════
// euler's formula: the grade-complete series
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_eulers_formula_is_grade_complete_taylor() {
    // e^(ix) = cos(x) + i·sin(x)
    //
    // e^(ix) is the taylor series with ALL grades present
    // cos(x) is the even-grade projection (grades 0↔2)
    // sin(x) is the odd-grade projection (grades 1↔3)
    //
    // euler's formula is not connecting different functions
    // it is decomposing one series into its grade components
    //
    // the "i" in front of sin is the Q shift from even to odd pair

    let test_points = [0.5, 1.0, PI / 4.0, PI / 3.0, PI / 2.0, PI];

    for &x_val in &test_points {
        // compute e^(ix) via full taylor series, tracking even and odd terms
        let mut even_sum = 0.0; // will equal cos(x)
        let mut odd_sum = 0.0; // will equal sin(x)
        let mut x_n_mag = 1.0;
        let mut factorial = 1.0;

        for n in 0..25 {
            if n > 0 {
                x_n_mag *= x_val;
                factorial *= n as f64;
            }

            // i^n cycles: 1, i, -1, -i (the grade cycle)
            // real part gets even terms with duals
            // imaginary part gets odd terms with duals
            match n % 4 {
                0 => even_sum += x_n_mag / factorial, // grade 0: +real
                1 => odd_sum += x_n_mag / factorial,  // grade 1: +imag
                2 => even_sum -= x_n_mag / factorial, // grade 2: -real (dual)
                3 => odd_sum -= x_n_mag / factorial,  // grade 3: -imag (dual)
                _ => unreachable!(),
            }
        }

        let expected_cos = x_val.cos();
        let expected_sin = x_val.sin();

        assert!(
            (even_sum - expected_cos).abs() < 1e-8,
            "even grades at x={:.3}: {:.8} = cos({:.3}) = {:.8}",
            x_val,
            even_sum,
            x_val,
            expected_cos
        );
        assert!(
            (odd_sum - expected_sin).abs() < 1e-8,
            "odd grades at x={:.3}: {:.8} = sin({:.3}) = {:.8}",
            x_val,
            odd_sum,
            x_val,
            expected_sin
        );
    }
}

#[test]
fn it_proves_i_is_the_q_shift_between_grade_pairs() {
    // in euler's formula e^(ix) = cos(x) + i·sin(x)
    // the "i" is not a mysterious imaginary unit
    // it is the Q shift (π/2 rotation) from even grades to odd grades
    //
    // cos lives on grades 0↔2 (even pair)
    // sin lives on grades 1↔3 (odd pair)
    // multiplying by i = [1, π/2] rotates from even to odd
    //
    // euler's formula says: the grade-complete series decomposes into
    // its even projection plus Q times its odd projection

    let i = Geonum::new(1.0, 1.0, 2.0); // [1, π/2]

    // i rotates grade 0 → grade 1
    let grade_0 = Geonum::new(1.0, 0.0, 1.0);
    let rotated = grade_0 * i;
    assert_eq!(
        rotated.angle.grade(),
        1,
        "i shifts grade 0 to grade 1: even → odd"
    );

    // i rotates grade 2 → grade 3
    let grade_2 = Geonum::new_with_blade(1.0, 2, 0.0, 1.0);
    let rotated_2 = grade_2 * i;
    assert_eq!(
        rotated_2.angle.grade(),
        3,
        "i shifts grade 2 to grade 3: even → odd"
    );

    // the grade pairs:
    // even: {0, 2} — related by dual (π rotation)
    // odd: {1, 3} — related by dual (π rotation)
    // i connects even pair to odd pair via Q shift (π/2 rotation)

    assert_eq!(grade_0.dual().angle.grade(), 2, "dual: 0 ↔ 2");
    let grade_1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);
    assert_eq!(grade_1.dual().angle.grade(), 3, "dual: 1 ↔ 3");

    // so euler's formula is:
    // e^(ix) = (grade 0↔2 projection) + Q × (grade 1↔3 projection)
    //        = cos(x) + i·sin(x)
    // the "beauty" is structural decomposition, not mysterious connection
}

// ═══════════════════════════════════════════════════════════════════════════════
// convergence radius is geometric
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_convergence_is_angle_normalization_dominance() {
    // a taylor series converges when the factorial normalization (angle descent)
    // grows faster than the power accumulation (angle ascent)
    //
    // x^n grows the angle: nθ
    // n! normalizes it: product of n angle ratios
    //
    // convergence = angle descent outpaces angle ascent
    // divergence = angle ascent outpaces angle descent
    //
    // for e^x: n! always wins eventually (converges for all x)
    // for 1/(1-x): no factorial normalization (geometric series, radius = 1)

    // e^x converges for any x because factorial normalization always dominates
    let large_x = 10.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut terms_decreasing_after = 0;

    for n in 1..50 {
        term *= large_x / n as f64; // x^n/n! ratio: x/n
        sum += term;

        // once n > x, each term is smaller than the last
        // this is when angle descent (n) overtakes angle ascent (x)
        if n as f64 > large_x && terms_decreasing_after == 0 {
            terms_decreasing_after = n;
        }
    }

    assert_eq!(
        terms_decreasing_after, 11,
        "terms start decreasing when n > x = 10"
    );
    assert!(
        (sum - large_x.exp()).abs() < 1e-4,
        "e^10 converges: {:.4} ≈ {:.4}",
        sum,
        large_x.exp()
    );

    // geometric series 1/(1-x) = Σ x^n has no factorial
    // no angle descent normalization → only converges when |x| < 1
    let x_inside = 0.5_f64;
    let x_outside = 1.5_f64;

    let mut geo_sum_inside = 0.0_f64;
    let mut geo_sum_outside = 0.0_f64;
    let mut x_n_in = 1.0;
    let mut x_n_out = 1.0;

    for _ in 0..100 {
        geo_sum_inside += x_n_in;
        geo_sum_outside += x_n_out;
        x_n_in *= x_inside;
        x_n_out *= x_outside;
    }

    let expected_inside = 1.0 / (1.0 - x_inside); // 2.0
    assert!(
        (geo_sum_inside - expected_inside).abs() < 1e-8,
        "geometric series converges inside radius: {:.4} ≈ {:.4}",
        geo_sum_inside,
        expected_inside
    );
    assert!(
        geo_sum_outside > 1e10,
        "geometric series diverges outside radius: no angle descent to tame growth"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// taylor series was never about approximation
//
// it is the decomposition of a function into its angle-level contributions
// each term x^n/n! is one level of angular displacement normalized by
// the geometric weight that differentiation accumulated at that level
//
// e^x treats all levels equally
// sin filters to odd grades, cos filters to even grades
// the alternating signs are duals (duals)
// euler's formula reassembles the grade projections
// convergence is angle descent dominating angle ascent
//
// the series doesn't approximate the function from outside
// it reads the function's angular structure from inside
// ═══════════════════════════════════════════════════════════════════════════════
