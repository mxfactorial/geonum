// calculus is scalar toil — procedures that reconstruct what angles already express
//
// scalars cant see the rate without approaching it (limits)
// scalars cant see the composition without decomposing it (chain rule)
// scalars cant see the factors without separating them (product rule)
// every rule in calculus reconstructs information that scalars discarded at construction
//
// angles express exactly where youre headed:
//
//   the power n lives in the angle ratio: nθ / θ = n
//   the base x^(n-1) lives in the magnitude ratio: mag^n / mag
//   the derivative is their product — two divisions, no limits
//
//   differentiation is π/2 rotation: the tangent direction is one quarter turn
//   from the position. its not computed, its adjacent. grades cycle 0→1→2→3→0
//
//   integration is 3π/2 forward rotation (dual to -π/2)
//   the fundamental theorem connects accumulation to endpoint interference
//
// the readout (angle ratio) and the rotation (π/2) are the same geometry:
// multiplication accumulates angle, differentiation reads it back
//
// everything below proves this mechanically. differentiate_test.rs proves the rules above

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════════════════
// the power rule is a readout
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_encodes_the_power_in_the_angle() {
    // when you multiply x by itself n times, angles add: θ + θ + ... = nθ
    // the exponent is not computed — it accumulates in the angle

    let x = Geonum::new(3.0, 1.0, 6.0); // x = [3, π/6]
    let x_squared = x * x; // x² = [9, 2π/6]
    let x_cubed = x_squared * x; // x³ = [27, 3π/6]
    let x_fourth = x_cubed * x; // x⁴ = [81, 4π/6]

    // the power is the angle ratio — just read it off
    let x_angle = x.angle.grade_angle();

    let power_2 = x_squared.angle.grade_angle() / x_angle;
    let power_3 = x_cubed.angle.grade_angle() / x_angle;
    let power_4 = x_fourth.angle.grade_angle() / x_angle;

    assert!(
        (power_2 - 2.0).abs() < EPSILON,
        "x² angle / x angle = 2: the angle knows the power"
    );
    assert!(
        (power_3 - 3.0).abs() < EPSILON,
        "x³ angle / x angle = 3: no computation needed"
    );
    assert!(
        (power_4 - 4.0).abs() < EPSILON,
        "x⁴ angle / x angle = 4: the exponent was always there"
    );

    // the magnitude ratio gives x^(n-1)
    let base_2 = x_squared.mag / x.mag; // 9/3 = 3 = x^1
    let base_3 = x_cubed.mag / x.mag; // 27/3 = 9 = x^2
    let base_4 = x_fourth.mag / x.mag; // 81/3 = 27 = x^3

    assert!((base_2 - 3.0).abs() < EPSILON, "x²/x = x^1 = 3");
    assert!((base_3 - 9.0).abs() < EPSILON, "x³/x = x^2 = 9");
    assert!((base_4 - 27.0).abs() < EPSILON, "x⁴/x = x^3 = 27");

    // power rule = angle ratio × magnitude ratio
    let deriv_2 = power_2 * base_2; // 2 × 3 = 6
    let deriv_3 = power_3 * base_3; // 3 × 9 = 27
    let deriv_4 = power_4 * base_4; // 4 × 27 = 108

    let x_val = x.mag;
    assert!((deriv_2 - 2.0 * x_val).abs() < EPSILON, "d/dx[x²] = 2x = 6");
    assert!(
        (deriv_3 - 3.0 * x_val * x_val).abs() < EPSILON,
        "d/dx[x³] = 3x² = 27"
    );
    assert!(
        (deriv_4 - 4.0 * x_val * x_val * x_val).abs() < EPSILON,
        "d/dx[x⁴] = 4x³ = 108"
    );

    // the same three rates as geonums, from the turn: each power rotated a quarter turn,
    // counted once for each angle of x it carries, over the quarter turn of x
    let rate_2 = turn_and_divide(&x_squared, &x, 2.0); // [6, π/6]
    let rate_3 = turn_and_divide(&x_cubed, &x, 3.0); // [27, 2π/6]
    let rate_4 = turn_and_divide(&x_fourth, &x, 4.0); // [108, 3π/6]
    assert!(rate_2.near(&x.scale(2.0)), "2x, on the ray of x");
    assert!(rate_3.near(&x_squared.scale(3.0)), "3x², on the ray of x²");
    assert!(rate_4.near(&x_cubed.scale(4.0)), "4x³, on the ray of x³");

    // the two ratios multiplied are the magnitudes the turn leaves
    assert!(rate_2.near_mag(deriv_2) && rate_3.near_mag(deriv_3) && rate_4.near_mag(deriv_4));
}

#[test]
fn it_steps_a_power_along_a_spiral() {
    // each power is the one before it times x: the magnitude × 3 and the angle + π/6
    let x = Geonum::new(3.0, 1.0, 6.0);
    let powers = [
        Geonum::new(1.0, 0.0, 1.0),  // [1, 0]
        Geonum::new(3.0, 1.0, 6.0),  // [3, π/6]
        Geonum::new(9.0, 1.0, 3.0),  // [9, π/3]
        Geonum::new(27.0, 1.0, 2.0), // [27, π/2]
    ];
    for step in 0..3 {
        assert!((powers[step] * x).near(&powers[step + 1]));

        // the lib names the step scale_rotate too
        assert!(powers[step]
            .scale_rotate(x.mag, x.angle)
            .near(&powers[step + 1]));
    }

    // the spiral has two ends. with x at angle 0 the steps do not turn: the powers go
    // straight out a ray, the textbooks 1, 3, 9, 27, and only the magnitude holds the count
    let at_0 = Geonum::scalar(3.0);
    let mut out = Geonum::scalar(1.0);
    for textbook in [3.0, 9.0, 27.0] {
        out = out * at_0;
        assert!(out.near(&Geonum::scalar(textbook)));
    }

    // with x at magnitude 1 the steps do not scale: the powers go round a circle, and only
    // the angle holds the count
    let unit = Geonum::new(1.0, 1.0, 6.0);
    let mut around = Geonum::scalar(1.0);
    for steps in [1.0, 2.0, 3.0] {
        around = around * unit;
        assert!(around.near(&Geonum::new(1.0, steps, 6.0)));
    }

    // going down the spiral is going on. scale by 1/r and rotate the rest of xs whole turn,
    // and each power lands on the one below it, a whole turn on: four blades, the same grade.
    // five steps at five angles, angle 0 among them
    for (p, d) in [(1.0, 6.0), (1.0, 3.0), (2.0, 5.0), (0.0, 1.0), (3.0, 4.0)] {
        let x = Geonum::new(1.5, p, d);
        let rest_of_the_turn = Angle::new(2.0, 1.0) - x.angle; // made once, at the door
        let mut below = Geonum::scalar(1.0);
        for _ in 0..5 {
            let here = below * x;
            let down = here.scale_rotate(1.0 / x.mag, rest_of_the_turn);
            assert_eq!(down.angle.blade(), below.angle.blade() + 4);
            assert!(down.base_angle().near(&below.base_angle()));
            below = here;
        }
    }
}

#[test]
fn it_derives_x_squared_without_limits() {
    // f(x) = x² at x = 3
    // traditional: lim(h→0) [(3+h)² - 9] / h = lim(h→0) [6h + h²] / h = 6
    // geometric: angle ratio × magnitude ratio = 2 × 3 = 6

    let x = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]
    let f_x = x * x; // [9, π/3]

    // rotate both a quarter turn. x² turns twice as far as x, so its turn counts 2
    // divide by the turn of x and the quarter turns cancel, leaving 2x
    let derivative = turn_and_divide(&f_x, &x, 2.0); // [6, π/6]
    assert!(
        derivative.near(&x.scale(2.0)),
        "f'(x) = 2x, on the ray of x"
    );

    // its magnitude is the two ratios multiplied
    let n = f_x.angle.grade_angle() / x.angle.grade_angle(); // 2
    let x_n_minus_1 = f_x.mag / x.mag; // 3

    let geometric_derivative = n * x_n_minus_1; // 6
    assert!(derivative.near_mag(geometric_derivative), "2 × 3 = 6");

    assert!((n - 2.0).abs() < EPSILON, "power = 2");
    assert!((x_n_minus_1 - 3.0).abs() < EPSILON, "x^(n-1) = 3");
    assert!(
        (geometric_derivative - 6.0).abs() < EPSILON,
        "f'(3) = 6 without limits"
    );

    // compare with limit definition to show they agree
    let h = 1e-10;
    let limit_derivative = ((3.0 + h) * (3.0 + h) - 9.0) / h;

    assert!(
        (geometric_derivative - limit_derivative).abs() < 1e-4,
        "geometric {} matches limit {}: same answer, no h→0",
        geometric_derivative,
        limit_derivative
    );
}

#[test]
fn it_derives_any_monomial() {
    // the mechanism works for any x^n at any x

    let test_cases: Vec<(f64, u32)> = vec![
        (2.0, 2),
        (3.0, 2),
        (2.0, 3),
        (3.0, 3),
        (2.0, 4),
        (3.0, 5),
        (4.0, 3),
        (5.0, 2),
        (1.5, 4),
        (2.5, 6),
    ];

    for (x_val, power) in test_cases {
        let x = Geonum::new(x_val, 1.0, 6.0);

        let mut x_n = Geonum::new(1.0, 0.0, 1.0);
        for _ in 0..power {
            x_n = x_n * x;
        }

        // the rate from the turn: x^n rotated a quarter turn and counted n times,
        // over the quarter turn of x
        let rate = turn_and_divide(&x_n, &x, power as f64);

        // the two ratios read its magnitude back
        let n = x_n.angle.grade_angle() / x.angle.grade_angle();
        let x_n_minus_1 = x_n.mag / x.mag;
        let geometric_derivative = n * x_n_minus_1;
        assert!((rate.mag - geometric_derivative).abs() < 1e-6);

        let traditional = power as f64 * x_val.powi(power as i32 - 1);

        assert!(
            (geometric_derivative - traditional).abs() < 1e-6,
            "d/dx[x^{}] at x={}: geometric {:.3} = traditional {:.3}",
            power,
            x_val,
            geometric_derivative,
            traditional
        );
    }
}

#[test]
fn it_proves_the_power_rule_is_two_ratios() {
    // the entire power rule reduces to:
    //   angle_ratio = nθ / θ = n        (rotation encodes the power)
    //   mag_ratio = mag^n / mag = x^(n-1) (projection encodes the base)
    //   f'(x) = angle_ratio × mag_ratio  (the "rule" is just reading)
    //
    // prove this for f(x) = x^5 at x = 2

    let x = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]

    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let x5 = x4 * x;

    let x_angle = x.angle.grade_angle();

    // angles add: π/6, 2π/6, 3π/6, 4π/6, 5π/6
    assert!(
        (x2.angle.grade_angle() - 2.0 * x_angle).abs() < EPSILON,
        "x² angle = 2θ"
    );
    assert!(
        (x3.angle.grade_angle() - 3.0 * x_angle).abs() < EPSILON,
        "x³ angle = 3θ"
    );
    assert!(
        (x4.angle.grade_angle() - 4.0 * x_angle).abs() < EPSILON,
        "x⁴ angle = 4θ"
    );
    assert!(
        (x5.angle.grade_angle() - 5.0 * x_angle).abs() < EPSILON,
        "x⁵ angle = 5θ"
    );

    // magnitudes multiply: 2, 4, 8, 16, 32
    assert!((x2.mag - 4.0).abs() < EPSILON, "x² mag = 4");
    assert!((x3.mag - 8.0).abs() < EPSILON, "x³ mag = 8");
    assert!((x4.mag - 16.0).abs() < EPSILON, "x⁴ mag = 16");
    assert!((x5.mag - 32.0).abs() < EPSILON, "x⁵ mag = 32");

    // two ratios give the derivative
    let n = x5.angle.grade_angle() / x_angle; // 5
    let base = x5.mag / x.mag; // 16 = 2^4 = x^(n-1)
    let derivative = n * base; // 80

    let traditional = 5.0 * 2.0_f64.powi(4);

    assert!((n - 5.0).abs() < EPSILON, "angle ratio = 5");
    assert!((base - 16.0).abs() < EPSILON, "magnitude ratio = x^4 = 16");
    assert!(
        (derivative - traditional).abs() < EPSILON,
        "5 × 16 = 80 = 5x^4 at x=2"
    );

    // where the two ratios come from: turn x⁵ a quarter turn and count it 5 times, turn x a
    // quarter turn, and divide. magnitudes divide, angles subtract, the quarter turns cancel
    let dx = x.differentiate(); // [2, π/6 + π/2]
    let df = x5.differentiate().scale(n); // [160, 5π/6 + π/2]
    let rate = Geonum::new_with_angle(df.mag / dx.mag, df.angle - dx.angle); // [80, 4π/6]
    assert!(
        rate.near(&x4.scale(5.0)),
        "5x⁴ as a geonum, on the ray of x⁴"
    );
    assert!(rate.near_mag(derivative), "the two ratios are what is left");
}

#[test]
fn it_derives_at_different_angles() {
    // the initial angle θ is arbitrary — the ratio nθ/θ = n regardless
    // proves the derivative is angle-independent (as it must be for a scalar function)

    let x_val = 3.0_f64;
    let power = 3_u32;
    let traditional = power as f64 * x_val.powi(power as i32 - 1); // 3 × 9 = 27

    let angles = [
        Angle::new(1.0, 6.0),  // π/6
        Angle::new(1.0, 4.0),  // π/4
        Angle::new(1.0, 3.0),  // π/3
        Angle::new(2.0, 7.0),  // 2π/7
        Angle::new(3.0, 11.0), // 3π/11
    ];

    for angle in angles {
        let x = Geonum::new_with_angle(x_val, angle);

        let mut x_n = Geonum::new(1.0, 0.0, 1.0);
        for _ in 0..power {
            x_n = x_n * x;
        }

        // the rate from the turn sits on the ray of x², whatever angle x started at
        let rate = turn_and_divide(&x_n, &x, power as f64);
        assert!(
            (rate.mag - traditional).abs() < 1e-6,
            "3x² = 27 from the turn"
        );
        assert!(
            (rate.angle.grade_angle() - 2.0 * angle.grade_angle()).abs() < 1e-9,
            "and it points along x²"
        );

        let n = x_n.angle.grade_angle() / x.angle.grade_angle();
        let base = x_n.mag / x.mag;
        let derivative = n * base;

        assert!(
            (derivative - traditional).abs() < 1e-6,
            "d/dx[x^3] = 27 at angle {:.4}: derivative = {:.6}",
            angle.grade_angle(),
            derivative
        );
    }
}

#[test]
fn it_extends_to_fractional_powers() {
    // x^n for fractional n: pow() computes [mag^n, n*angle]
    // the same two ratios still give the derivative

    let x = Geonum::new(4.0, 1.0, 6.0); // [4, π/6]

    // f(x) = x^(1/2) = √x
    let sqrt_x = x.pow(0.5);

    let n = sqrt_x.angle.grade_angle() / x.angle.grade_angle();
    assert!((n - 0.5).abs() < 0.01, "angle ratio = 0.5 for square root");

    let base = sqrt_x.mag / x.mag; // √4 / 4 = 0.5 = 4^(-1/2)
    assert!(
        (base - 0.5).abs() < 0.01,
        "magnitude ratio = x^(-1/2) = 0.5"
    );

    // derivative = 0.5 × 0.5 = 0.25
    let derivative = n * base;
    let traditional = 0.5 * 4.0_f64.powf(-0.5);

    assert!(
        (derivative - traditional).abs() < 0.01,
        "d/dx[√x] at x=4: geometric {:.4} = traditional {:.4}",
        derivative,
        traditional
    );

    // the turn takes a fractional count the same way: √x rotated a quarter turn and counted
    // half a time, over the quarter turn of x
    assert!(
        (turn_and_divide(&sqrt_x, &x, 0.5).mag - traditional).abs() < 1e-9,
        "½ x^(-½) = 0.25 from the turn"
    );

    // f(x) = x^(3/2)
    let x_three_halves = x.pow(1.5);
    let n_1_5 = x_three_halves.angle.grade_angle() / x.angle.grade_angle();
    let base_1_5 = x_three_halves.mag / x.mag;
    let deriv_1_5 = n_1_5 * base_1_5;
    let trad_1_5 = 1.5 * 4.0_f64.powf(0.5); // 1.5 × 2 = 3

    assert!(
        (deriv_1_5 - trad_1_5).abs() < 0.01,
        "d/dx[x^(3/2)] at x=4: geometric {:.4} = traditional {:.4}",
        deriv_1_5,
        trad_1_5
    );
    assert!(
        (turn_and_divide(&x_three_halves, &x, 1.5).mag - trad_1_5).abs() < 1e-9,
        "(3/2) x^(½) = 3 from the turn"
    );
}

#[test]
fn it_proves_power_rule_is_o1() {
    // traditional derivative computation scales with the method:
    // - limits: O(1) but requires h→0 approximation with error
    // - symbolic: O(n) for expression tree traversal
    // - automatic diff (dual numbers): O(n) for computation graph
    //
    // geometric derivative: always exactly 2 operations
    // 1. angle ratio (one division)
    // 2. magnitude ratio (one division)
    // regardless of power, regardless of evaluation point
    //
    // x^1000000 takes the same 2 operations as x^2

    let x = Geonum::new(1.001, 1.0, 6.0); // small base to avoid overflow

    let x_100 = x.pow(100.0);

    // x went into x¹⁰⁰ a hundred times, so its angle is the angle of x taken 100 times:
    // eight whole turns and more, and the blade keeps every one. angles add, and nothing
    // is divided to find the 100
    let n = 100.0;
    assert!(
        x_100.angle.near(&(x.angle * n)),
        "power = 100, in the angle"
    );
    let base = x_100.mag / x.mag;
    let derivative = n * base;

    // one turn and divide gives the same rate, at the same cost as for x²
    let rate = turn_and_divide(&x_100, &x, n);
    assert!((rate.mag - derivative).abs() / derivative < 1e-9);

    let traditional = 100.0 * 1.001_f64.powi(99);

    assert!(
        (derivative - traditional).abs() / traditional < 0.01,
        "d/dx[x^100] at x=1.001: geometric {:.6} ≈ traditional {:.6}",
        derivative,
        traditional
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// what limits throw away
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_limits_discard_what_angles_preserve() {
    // the limit definition computes the same derivative but throws away the geometry
    // it collapses [magnitude, angle] to a scalar rate
    // the geometric number keeps the rate AND the structural relationship

    let x = Geonum::new(4.0, 1.0, 6.0); // [4, π/6]
    let f_x = x * x; // x² = [16, π/3]

    // geometric: full information preserved
    let n = f_x.angle.grade_angle() / x.angle.grade_angle();
    let x_n_minus_1 = f_x.mag / x.mag;
    let rate = n * x_n_minus_1; // 8

    // the angle ratio tells you WHAT POWER you're differentiating
    assert!((n - 2.0).abs() < EPSILON, "angle ratio identifies x²");

    // the magnitude ratio tells you WHERE you're evaluating
    assert!(
        (x_n_minus_1 - 4.0).abs() < EPSILON,
        "magnitude ratio identifies x=4"
    );

    // the rate is their product
    assert!((rate - 8.0).abs() < EPSILON, "rate = 2 × 4 = 8");

    // limit definition: computes the same 8 but loses the decomposition
    let h = 0.0001;
    let limit = ((4.0_f64 + h).powi(2) - 16.0) / h;
    assert!(
        (limit - 8.0).abs() < 0.001,
        "limit gives ~8 but cant tell you it came from power=2 at x=4"
    );

    // differentiate() keeps the rate a geonum: x² turned over x turned leaves 2x
    let (df, dx) = (f_x.differentiate().scale(n), x.differentiate());
    let derivative = Geonum::new_with_angle(df.mag / dx.mag, df.angle - dx.angle);
    assert!(derivative.near(&x.scale(2.0)), "2x = [8, π/6]");
}

#[test]
fn it_shows_limits_lose_the_tangent_normal_dual() {
    // limits compute f'(x) ≈ 6 as a scalar rate
    // differentiate() rotates x and x² by π/2, and reading one turn against the other
    // gives the same 6 with the quarter turn the limit drops still attached

    let x_geo = Geonum::new(3.0, 1.0, 6.0); // [3, π/6], the angle carries the power
    let f_x = x_geo * x_geo; // x² = [9, π/3]

    // limit: approach from outside, collapse to scalar
    let h = 0.0001;
    let h_geo = Geonum::new(h, 1.0, 6.0);
    let x_h_geo = Geonum::new(3.0 + h, 1.0, 6.0);
    let f_x_h = x_h_geo * x_h_geo;
    let limit_result = ((f_x_h - f_x) / h_geo).mag;
    assert!(
        (limit_result - 6.0).abs() < 0.01,
        "limit projects to scalar ~6"
    );

    // rotate: turning x moves it along its quarter turn, and moves x² along its own
    // x² turns twice as far as x, the angle ratio, so its quarter turn is scaled by 2
    let n = f_x.angle.grade_angle() / x_geo.angle.grade_angle();
    let dx = x_geo.differentiate(); // [3, π/6 + π/2]
    let df = f_x.differentiate().scale(n); // [18, π/3 + π/2]

    // read the turn of x² against the turn of x: magnitudes divide, angles subtract
    // the quarter turns cancel and 2x is left on the ray of x
    let derivative = Geonum::new_with_angle(df.mag / dx.mag, df.angle - dx.angle);
    assert!(derivative.near(&x_geo.scale(2.0)), "df/dx = 2x = [6, π/6]");

    // the quarter turn between f and its motion IS the tangent-normal relationship
    assert!(f_x.dot(&df).near_mag(0.0), "x² ⊥ its own turn");
    assert_eq!(
        df.angle - f_x.angle,
        Angle::new(1.0, 2.0),
        "tangent-normal dual structure (quarter turn apart) lost in limit projection"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// higher derivatives and factorials
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_computes_higher_derivatives_by_repeated_ratio() {
    // d²/dx²[x^n] = n(n-1)x^(n-2)
    //
    // each derivative peels off one angle ratio and one magnitude factor
    // first ratio:  n from x^n,   base x^(n-1)
    // second ratio: (n-1) from x^(n-1), base x^(n-2)
    //
    // the derivative is a geonum that keeps the next ratio in its own angle, so each
    // higher derivative is the one before it turned and divided again — no lower power
    // has to be built beside it to read from. x⁴ is the only power built by hand,
    // and every lower one below comes out of it by turning

    let x = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let x4 = x * x * x * x; // [16, 4π/6]

    let x_angle = x.angle.grade_angle();

    // first derivative: d/dx[x^4] = 4x^3
    let n1 = x4.angle.grade_angle() / x_angle; // 4
    let base1 = x4.mag / x.mag; // 8 = x^3

    assert!((n1 - 4.0).abs() < EPSILON, "first power = 4");
    assert!((base1 - 8.0).abs() < EPSILON, "first base = x^3 = 8");

    let first_deriv = turn_and_divide(&x4, &x, 4.0); // [32, 3π/6]
    assert!(
        first_deriv.near_mag(n1 * base1),
        "the two ratios are what the turn and divide leaves"
    );
    assert!(
        first_deriv.near_mag(4.0 * 2.0_f64.powi(3)),
        "f'(x) = 4x^3 = 32"
    );

    // second derivative: 4x^3 carries the 3 in its angle
    let n2 = first_deriv.angle.grade_angle() / x_angle; // 3
    let second_deriv = turn_and_divide(&first_deriv, &x, 3.0); // [48, 2π/6]
    assert!(
        second_deriv.near_mag(4.0 * 3.0 * 2.0_f64.powi(2)),
        "f''(x) = 12x^2 = 48"
    );

    // third derivative: 12x^2 carries the 2
    let n3 = second_deriv.angle.grade_angle() / x_angle; // 2
    let third_deriv = turn_and_divide(&second_deriv, &x, 2.0); // [48, π/6]
    assert!(
        third_deriv.near_mag(4.0 * 3.0 * 2.0 * 2.0),
        "f'''(x) = 24x = 48"
    );

    // fourth derivative: 24x carries the 1
    let n4 = third_deriv.angle.grade_angle() / x_angle; // 1
    let fourth_deriv = turn_and_divide(&third_deriv, &x, 1.0); // [24, 0]
    assert!(fourth_deriv.near_mag(24.0), "f''''(x) = 24 (constant)");

    // the constant carries no angle, so one more turn and divide leaves nothing
    assert!(
        turn_and_divide(&fourth_deriv, &x, 0.0).near_mag(0.0),
        "f⁽⁵⁾ = 0"
    );

    // the nth derivative of x^n is n! — angle ratios multiply to the factorial
    let factorial_from_angles = n1 * n2 * n3 * n4;
    assert!(
        (factorial_from_angles - 24.0).abs() < EPSILON,
        "angle ratios multiply to n! = 4! = 24"
    );
}

#[test]
fn it_shows_factorial_emerges_from_angle_descent() {
    // for x^n, the nth derivative is n!
    // each derivative peels off one angle ratio: n, (n-1), (n-2), ..., 2, 1
    // their product is n!
    //
    // the factorial is not a combinatorial object — it is the product
    // of angle ratios extracted during repeated differentiation

    let x = Geonum::new(3.0, 1.0, 6.0);
    let x_angle = x.angle.grade_angle();

    let mut current = Geonum::new(1.0, 0.0, 1.0);
    let mut angle_ratios = Vec::new();

    for i in 1..=6 {
        current = current * x;
        let ratio = current.angle.grade_angle() / x_angle;
        angle_ratios.push(ratio);
        assert!(
            (ratio - i as f64).abs() < EPSILON,
            "x^{} angle ratio = {}",
            i,
            i
        );
    }

    let factorial_3: f64 = angle_ratios[0..3].iter().product();
    let factorial_4: f64 = angle_ratios[0..4].iter().product();
    let factorial_5: f64 = angle_ratios[0..5].iter().product();
    let factorial_6: f64 = angle_ratios[0..6].iter().product();

    assert!((factorial_3 - 6.0).abs() < EPSILON, "3! = 6 from angles");
    assert!((factorial_4 - 24.0).abs() < EPSILON, "4! = 24 from angles");
    assert!(
        (factorial_5 - 120.0).abs() < EPSILON,
        "5! = 120 from angles"
    );
    assert!(
        (factorial_6 - 720.0).abs() < EPSILON,
        "6! = 720 from angles"
    );

    // the descent itself: turn x⁶ and divide, six times over. each step is handed its
    // count, 6 then 5 then 4, and proves it against the angle the last one left. the sixth
    // lands on 6! with no angle remaining
    let mut descending = current; // x⁶ = [729, π]
    for count in (1..=6).rev() {
        descending = turn_and_divide(&descending, &x, count as f64);
    }
    assert!(
        descending.near(&Geonum::scalar(720.0)),
        "d⁶/dx⁶[x⁶] = 6! = [720, 0]"
    );
}

#[test]
fn it_proves_zero_derivative_for_constants() {
    // f(x) = c has no x dependence
    // c = [c, 0] — zero angle
    // angle ratio = 0/θ = 0, so derivative = 0
    //
    // the zero derivative is the absence of angle, not a rule to memorize

    let x = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]
    let constant = Geonum::new(7.0, 0.0, 1.0); // [7, 0]

    let n = constant.angle.grade_angle() / x.angle.grade_angle(); // 0 / (π/6) = 0
    let derivative = n * (constant.mag / x.mag);

    assert!(
        derivative.abs() < EPSILON,
        "constant derivative = 0: no angle means no x dependence"
    );

    // from the turn: a constant turns 0 times as far as x, so its quarter turn counts for
    // nothing
    assert!(
        turn_and_divide(&constant, &x, 0.0).near_mag(0.0),
        "nothing to count"
    );
}

#[test]
fn it_proves_linear_derivative_for_x() {
    // f(x) = x carries exactly one copy of x's angle and magnitude
    // angle ratio = θ/θ = 1, magnitude ratio = mag/mag = 1
    // derivative = 1

    let x = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]

    let n = x.angle.grade_angle() / x.angle.grade_angle(); // 1
    let base = x.mag / x.mag; // 1
    let derivative = n * base; // 1

    assert!((n - 1.0).abs() < EPSILON, "x carries one copy of θ");
    assert!((base - 1.0).abs() < EPSILON, "x carries one copy of mag");
    assert!((derivative - 1.0).abs() < EPSILON, "d/dx[x] = 1");

    // from the turn: x over itself. the quarter turns cancel and so does the angle
    assert!(
        turn_and_divide(&x, &x, 1.0).near(&Geonum::scalar(1.0)),
        "[1, 0]"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// differentiation cycles grades
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_proves_differentiation_cycles_grades() {
    // differentiation is π/2 rotation cycling through 4 geometric grades
    // each derivative moves to the next grade: 0→1→2→3→0
    // sin(θ+π/2) = cos(θ) is the quadrature identity that creates this cycle

    let f = Geonum::new(3.0, 0.0, 1.0); // [3, 0] at grade 0

    // each derivative here is against the number's own angle, so the bare quarter turn is
    // the whole of it. against another number the turn is counted and divided, which the
    // power rule tests above and differentiate_test.rs do
    let f_prime = f.differentiate();
    let f_double_prime = f_prime.differentiate();
    let f_triple_prime = f_double_prime.differentiate();
    let f_quad_prime = f_triple_prime.differentiate();

    // grades cycle 0→1→2→3→0
    assert_eq!(f.angle.grade(), 0, "f at grade 0 (scalar)");
    assert_eq!(f_prime.angle.grade(), 1, "f' at grade 1 (vector)");
    assert_eq!(f_double_prime.angle.grade(), 2, "f'' at grade 2 (bivector)");
    assert_eq!(
        f_triple_prime.angle.grade(),
        3,
        "f''' at grade 3 (trivector)"
    );
    assert_eq!(f_quad_prime.angle.grade(), 0, "f'''' back at grade 0");

    // blades accumulate: each differentiation adds 1
    assert_eq!(f_prime.angle.blade(), f.angle.blade() + 1, "1 blade added");
    assert_eq!(
        f_double_prime.angle.blade(),
        f.angle.blade() + 2,
        "2 blades added"
    );
    assert_eq!(
        f_triple_prime.angle.blade(),
        f.angle.blade() + 3,
        "3 blades added"
    );
    assert_eq!(
        f_quad_prime.angle.blade(),
        f.angle.blade() + 4,
        "4 blades added"
    );

    // magnitude preserved through all rotations
    assert!(
        f_prime.near_mag(f.mag),
        "differentiation preserves magnitude"
    );
    assert!(
        f_quad_prime.near_mag(f.mag),
        "magnitude preserved through full cycle"
    );

    // prove the quadrature relationship that creates grade cycling
    let angle_0 = Angle::new(0.0, 1.0);
    let angle_90 = angle_0 + Angle::new(1.0, 2.0);

    assert!(
        (angle_0.grade_angle().cos() - angle_90.grade_angle().sin()).abs() < EPSILON,
        "cos(θ) = sin(θ+π/2)"
    );

    // grade determines behavior regardless of blade count
    let high_blade = Geonum::new_with_blade(3.0, 1000, 0.0, 1.0); // blade 1000, grade 0
    assert_eq!(high_blade.angle.grade(), 0, "blade 1000 % 4 = 0");
    assert_eq!(
        high_blade.differentiate().angle.grade(),
        1,
        "differentiation moves grade 0→1 at any blade count"
    );

    // prove 4-cycle over 20 steps
    let mut current = f;
    for step in 1..=20 {
        current = current.differentiate();
        assert_eq!(
            current.angle.grade(),
            step % 4,
            "step {} produces grade {}",
            step,
            step % 4
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// integration and the fundamental theorem
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn it_connects_differentiation_and_integration_via_grade_cycle() {
    // differentiate() rotates π/2 (grade 0 → 1)
    // integrate() rotates 3π/2 (grade 1 → 0, forward equivalent to -π/2)
    // together they complete a full 2π cycle (4 blades)

    let f = Geonum::new(16.0, 0.0, 1.0); // grade 0

    // against its own angle: a bare quarter turn out, and integrate() the rest of the way round
    let f_prime = f.differentiate();
    assert_eq!(f_prime.angle.grade(), 1, "derivative at grade 1");
    assert!(f_prime.near_mag(16.0), "magnitude preserved");

    let back_to_f = f_prime.integrate();
    assert_eq!(back_to_f.angle.grade(), 0, "integrated back to grade 0");
    assert!(back_to_f.near_mag(16.0), "magnitude preserved");

    // differentiate adds π/2, integrate adds 3π/2, net = 4 blades = 2π
    let diff_rotation = f_prime.angle - f.angle;
    let int_rotation = back_to_f.angle - f_prime.angle;

    assert_eq!(
        diff_rotation,
        Angle::new(1.0, 2.0),
        "differentiate adds π/2"
    );
    assert_eq!(int_rotation, Angle::new(3.0, 2.0), "integrate adds 3π/2");
    assert_eq!(
        back_to_f.angle.blade() - f.angle.blade(),
        4,
        "full cycle: 4 blades"
    );
}

#[test]
fn it_proves_fundamental_theorem_is_accumulation_equals_interference() {
    // Newton-Leibniz: ∫ₐᵇ f'(x) dx = F(b) - F(a)
    // in angle space: accumulated geometric sum = destructive interference of endpoints

    // ∫₂⁵ 2x dx = x²|₂⁵ = 25 - 4 = 21
    let a: f64 = 2.0;
    let b: f64 = 5.0;

    // left side: accumulation via geometric addition (riemann sum)
    let num_steps = 1000;
    let dx = (b - a) / num_steps as f64;
    let dx_geo = Geonum::new(dx, 0.0, 1.0);
    let mut accumulated = Geonum::new(0.0, 0.0, 1.0);

    for i in 0..num_steps {
        let x_i = a + i as f64 * dx;
        let f_prime_i = Geonum::new(2.0 * x_i, 0.0, 1.0); // f'(x) = 2x
        accumulated = accumulated + f_prime_i * dx_geo;
    }

    // right side: F(b) - F(a) as destructive interference
    // F(a) placed at angle π creates cos(π) = -1 interference with F(b) at angle 0
    // the antiderivative is not supplied: 2x run backward through the turn gives x² at each end
    let antiderivative = |r: f64| {
        let x = Geonum::new(r, 1.0, 12.0); // given an angle so the count can be proven
        turn_back_and_divide(&x.scale(2.0), &x, 2.0).mag
    };
    let f_b = Geonum::new(antiderivative(b), 0.0, 1.0); // F(5) = [25, 0]
    let f_a_negated = Geonum::new(antiderivative(a), 1.0, 1.0); // F(2) = [4, π]
    let interference = f_b + f_a_negated;

    // cosine rule: c² = 25² + 4² + 2(25)(4)cos(π) = 625 + 16 - 200 = 441
    let expected_squared = f_b.mag.powi(2) + a.powi(4) + 2.0 * f_b.mag * a.powi(2) * PI.cos();
    assert!((expected_squared - 441.0).abs() < EPSILON);
    assert!((expected_squared.sqrt() - 21.0).abs() < EPSILON);

    // fundamental theorem: accumulation = interference
    assert!(
        (accumulated.mag - interference.mag).abs() < 0.02,
        "accumulation {:.3} = interference {:.3}",
        accumulated.mag,
        interference.mag
    );
    assert!((accumulated.mag - 21.0).abs() < 0.02, "both equal 21");
}

#[test]
fn it_shows_subtraction_in_fundamental_theorem_is_interference() {
    // the "minus" in F(b) - F(a) is destructive interference, not algebraic subtraction
    // placing F(a) at angle π creates cos(π) = -1 which cancels

    // ∫₁³ 2x dx = x²|₁³ = 9 - 1 = 8
    // the two endpoint values come from 2x run backward through the turn, not typed in
    let antiderivative = |r: f64| {
        let x = Geonum::new(r, 1.0, 12.0); // given an angle so the count can be proven
        turn_back_and_divide(&x.scale(2.0), &x, 2.0).mag
    };
    let f_b = Geonum::new(antiderivative(3.0), 0.0, 1.0); // F(3) = [9, 0]
    let f_a_at_pi = Geonum::new(antiderivative(1.0), 1.0, 1.0); // F(1) = [1, π]
    let interference = f_b + f_a_at_pi;

    // cosine rule: c² = 81 + 1 + 2(9)(1)cos(π) = 81 + 1 - 18 = 64
    assert!(
        (interference.mag - 8.0).abs() < EPSILON,
        "interference magnitude via cos(π) = -1: {:.3}",
        interference.mag
    );
}

#[test]
fn it_encodes_definite_integrals_with_value_and_domain() {
    // ∫₂⁵ x² dx = 39, but the scalar keeps only the value and drops the domain [2,5] it ran
    // over. the geonum carries both: the value in the magnitude, the domain span b−a in the
    // angle. and the value is not hand-subtracted — F(b) − F(a) is geonum subtraction, the
    // "minus" a π rotation that interferes the endpoints, the way the FTC test does it above

    let (a, b): (f64, f64) = (2.0, 5.0);

    // the antiderivative of x² is x³/3, read at the two endpoints on the scalar ray
    // produced, not typed in: x² times the quarter turn of x, rotated the rest of the way
    // round, divided by the 3 turns the result carries
    let antiderivative = |r: f64| {
        let x = Geonum::new(r, 1.0, 12.0); // given an angle so the count can be proven
        turn_back_and_divide(&(x * x), &x, 3.0).mag
    };
    let f_a = Geonum::new(antiderivative(a), 0.0, 1.0); // F(2) = 8/3
    let f_b = Geonum::new(antiderivative(b), 0.0, 1.0); // F(5) = 125/3

    // F(b) − F(a): subtraction negates F(a) to angle π and interferes — the value falls out
    let value = f_b - f_a;
    assert!(
        value.near_mag(39.0),
        "∫₂⁵ x² dx = 39 by endpoint interference"
    );
    assert_eq!(value.angle.grade(), 0, "a positive value on the scalar ray");

    // the domain the scalar forgot is the span b − a, carried as an angle — three half-turns
    let domain = Angle::new(b, 1.0) - Angle::new(a, 1.0); // 5π − 2π = 3π
    assert_eq!(
        domain,
        Angle::new(3.0, 1.0),
        "the angle carries the domain span, 3π"
    );

    // one geonum holds both — the value the scalar keeps and the domain it drops
    let integral = Geonum::new_with_angle(value.mag, domain);
    assert!(integral.near_mag(39.0), "value in the magnitude");
    assert_eq!(integral.angle, Angle::new(3.0, 1.0), "domain in the angle")
}

// ═══════════════════════════════════════════════════════════════════════════════
// vector calculus
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn its_a_gradient() {
    // traditional: ∇f = [∂f/∂x, ∂f/∂y] requires finite differences then assembling a vector
    // geonum: read each partial from the angle ratio of its monomial, encode with direction, add

    // f(x,y) = x² + y² at (3,4)
    let x_val = 3.0;
    let y_val = 4.0;

    let x = Geonum::new(x_val, 1.0, 6.0); // [3, π/6]
    let y = Geonum::new(y_val, 1.0, 6.0); // [4, π/6]

    let x_squared = x * x; // [9, 2π/6]
    let y_squared = y * y; // [16, 2π/6]

    // ∂f/∂x from the turn: x² rotated a quarter turn and counted twice, over the quarter
    // turn of x. the angle ratio is that count: power = 2, base = 3 → 2×3 = 6
    let nx = x_squared.angle.grade_angle() / x.angle.grade_angle();
    let df_dx = turn_and_divide(&x_squared, &x, 2.0).mag;
    assert!((df_dx - nx * (x_squared.mag / x.mag)).abs() < EPSILON);

    // ∂f/∂y the same way: power = 2, base = 4 → 2×4 = 8
    let ny = y_squared.angle.grade_angle() / y.angle.grade_angle();
    let df_dy = turn_and_divide(&y_squared, &y, 2.0).mag;
    assert!((df_dy - ny * (y_squared.mag / y.mag)).abs() < EPSILON);

    assert!(
        (df_dx - 6.0).abs() < EPSILON,
        "∂f/∂x = 2x = 6 from angle ratio"
    );
    assert!(
        (df_dy - 8.0).abs() < EPSILON,
        "∂f/∂y = 2y = 8 from angle ratio"
    );

    // encode partials with direction, add → gradient
    let partial_x_geo = Geonum::new(df_dx, 0.0, 1.0); // [6, 0]
    let partial_y_geo = Geonum::new(df_dy, 1.0, 2.0); // [8, π/2]
    let gradient = partial_x_geo + partial_y_geo;

    let expected_mag = (6.0_f64.powi(2) + 8.0_f64.powi(2)).sqrt(); // 10
    let expected_dir = 8.0_f64.atan2(6.0); // ≈ 0.927 rad

    assert!(
        (gradient.mag - expected_mag).abs() < 0.01,
        "gradient magnitude = 10"
    );
    assert!(
        (gradient.angle.grade_angle() - expected_dir).abs() < 0.01,
        "gradient direction = atan2(8,6)"
    );

    // the same gradient with no partial taken on either axis: x and y are two projections
    // of one number, the point, and f is the magnitude of the point times itself
    let point = Geonum::new_from_cartesian(x_val, y_val); // [5, θ]
    let squared = point * point; // [25, 2θ]
    assert!(
        squared.near_mag(x_squared.mag + y_squared.mag),
        "f = x² + y² = 25"
    );

    // the square turns twice as far as the point, so its quarter turn counts 2. over the
    // quarter turn of the point that leaves twice the point, on its ray
    let on_the_ray = turn_and_divide(&squared, &point, 2.0);
    assert!(on_the_ray.near(&point.scale(2.0)), "∇f = [10, θ]");
    assert!(on_the_ray.near_mag(expected_mag), "gradient magnitude = 10");
    assert!(
        on_the_ray.angle.near_rad(expected_dir),
        "gradient direction = atan2(8,6)"
    );

    // its projections are the two partials computed above
    assert!(
        (on_the_ray.project_to_dimension(0) - df_dx).abs() < EPSILON,
        "∂f/∂x = 6 is its projection onto x"
    );
    assert!(
        (on_the_ray.project_to_dimension(1) - df_dy).abs() < EPSILON,
        "∂f/∂y = 8 is its projection onto y"
    );

    // it lands on the ray of the assembled gradient. that sum sits a whole turn on at the
    // same grade, so its read at its base angle
    assert_eq!(gradient.angle.grade(), on_the_ray.angle.grade());
    assert!(
        on_the_ray.near(&gradient.base_angle()),
        "[6, 0] + [8, π/2] = [10, θ]"
    );

    // f holds still along the circle the point turns on, so the gradient reads 0 against
    // the point a quarter turn on
    assert!(
        on_the_ray.dot(&point.differentiate()).near_mag(0.0),
        "∇f ⊥ the level curve"
    );
}

#[test]
fn its_a_laplacian() {
    // traditional: ∇²f = ∂²f/∂x² + ∂²f/∂y² requires second-order finite differences
    // geonum: second angle ratio readout per variable, summed
    //
    // for x²: first ratio = 2 (from x²), second ratio = 1 (from x¹), base = x/x = 1
    // ∂²(x²)/∂x² = 2 × 1 × 1 = 2
    // same for y² → laplacian = 2 + 2 = 4

    let x_val = 2.0;
    let y_val = 3.0;

    let x = Geonum::new(x_val, 1.0, 6.0);
    let y = Geonum::new(y_val, 1.0, 6.0);

    let x_squared = x * x;
    let y_squared = y * y;

    let x_angle = x.angle.grade_angle();
    let y_angle = y.angle.grade_angle();

    // ∂²(x²)/∂x²: first ratio from x², second ratio from the first partial 2x
    let df_dx = turn_and_divide(&x_squared, &x, 2.0); // [4, π/6] = 2x
    let n1_x = x_squared.angle.grade_angle() / x_angle; // 2
    let n2_x = df_dx.angle.grade_angle() / x_angle; // 1
    let d2f_dx2 = turn_and_divide(&df_dx, &x, 1.0).mag; // 2

    // ∂²(y²)/∂y²: same pattern
    let df_dy = turn_and_divide(&y_squared, &y, 2.0); // [6, π/6] = 2y
    let n1_y = y_squared.angle.grade_angle() / y_angle; // 2
    let n2_y = df_dy.angle.grade_angle() / y_angle; // 1
    let d2f_dy2 = turn_and_divide(&df_dy, &y, 1.0).mag; // 2

    let laplacian = d2f_dx2 + d2f_dy2; // 4

    assert!(
        (d2f_dx2 - n1_x * n2_x).abs() < EPSILON && (d2f_dx2 - 2.0).abs() < EPSILON,
        "∂²f/∂x² = 2 from angle ratios"
    );
    assert!(
        (d2f_dy2 - n1_y * n2_y).abs() < EPSILON && (d2f_dy2 - 2.0).abs() < EPSILON,
        "∂²f/∂y² = 2 from angle ratios"
    );
    assert!(
        (laplacian - 4.0).abs() < EPSILON,
        "∇²f = 4: no finite differences, no h"
    );
}

#[test]
fn its_a_directional_derivative() {
    // traditional: ∇_a f = a·∇f takes a limit along each basis vector, sums those into ∇f,
    // then dots the sum with the direction a
    // geonum: rotate the point a quarter turn and thats the normal. project it onto a

    let position = Geonum::new(10.0, 1.0, 3.0); // [10, π/3]
    let normal = position.differentiate(); // [10, π/3 + π/2]
    assert_eq!(normal.angle.grade(), 1, "a quarter turn on is grade 1");
    assert!(
        position.dot(&normal).near_mag(0.0),
        "the normal reads 0 against the position"
    );

    // the rate on each axis, the two partials a textbook sums over
    let on_x = normal.project_to_dimension(0); // -10 sin(π/3)
    let on_y = normal.project_to_dimension(1); // 10 cos(π/3)

    // directions as π fractions: the x axis, the y axis, and three that are neither
    for (pi_radians, divisor) in [(0.0, 1.0), (1.0, 2.0), (1.0, 5.0), (3.0, 4.0), (5.0, 3.0)] {
        let direction = Angle::new(pi_radians, divisor);

        // one projection of the normal
        let rate = normal.mag * normal.angle.project(direction);

        // the textbook rate of the reading on that direction:
        // d/dt[10 cos(t - φ)] = -10 sin(t - φ)
        let phi = pi_radians * PI / divisor;
        let textbook = -10.0 * (PI / 3.0 - phi).sin();
        assert!(
            (rate - textbook).abs() < EPSILON,
            "∇_a f along {pi_radians}π/{divisor}"
        );

        // the long way lands on the same number: a rate on each axis, summed against a
        let summed = on_x * phi.cos() + on_y * phi.sin();
        assert!(
            (rate - summed).abs() < EPSILON,
            "a·∇f is one projection, with no sum taken"
        );
    }
}

#[test]
fn its_an_orthogonal_axis() {
    // traditional: y is orthogonal to x, so nothing on y can be had from x, and the rate
    // on x takes a limit on x
    // geonum: y is x a quarter turn on, so each axis holds the rate of the other

    let position = Geonum::new(10.0, 1.0, 3.0); // [10, π/3]
    let on_x = position.project_to_dimension(0);
    let on_y = position.project_to_dimension(1);
    assert!((on_x - 5.0).abs() < EPSILON, "10 cos(π/3)");
    assert!((on_y - 10.0 * (PI / 3.0).sin()).abs() < EPSILON, "8.66");

    let normal = position.differentiate();
    let rate_on_x = normal.project_to_dimension(0);
    let rate_on_y = normal.project_to_dimension(1);

    // the rate on y is what the position read on x
    assert!((rate_on_y - on_x).abs() < EPSILON, "5");

    // the rate on x is what it read on y, reversed: a quarter turn on from y is x reversed
    // the -8.66 a limit closes in on was on the y axis the whole time
    assert!((rate_on_x + on_y).abs() < EPSILON, "-8.66");

    // so each quarter turn steps the same reading to the next dimension: position onto
    // dimension 0, velocity onto dimension 1, acceleration onto dimension 2, and on
    let mut rung = position;
    for dimension in 0..8 {
        assert_eq!(rung.angle.blade(), dimension);
        assert!(
            (rung.project_to_dimension(dimension) - on_x).abs() < EPSILON,
            "blade {dimension} reads 5 on dimension {dimension}"
        );
        rung = rung.differentiate();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// geometric integrals
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn its_a_line_integral() {
    // traditional: ∫_C F·dr requires curve parameterization
    // geonum: field.dot(path) for constant field on straight path

    let start = Geonum::new_from_cartesian(0.0, 0.0);
    let end = Geonum::new_from_cartesian(2.0, 3.0);
    let path = end - start;

    // constant vector field F = [1, 2]
    let field = Geonum::new_from_cartesian(1.0, 2.0);

    // traditional: F·(end-start) = 1*2 + 2*3 = 8
    let trad_integral: f64 = 1.0 * 2.0 + 2.0 * 3.0;

    let geo_integral = field.dot(&path);

    assert!(
        (geo_integral.mag - trad_integral).abs() < 0.1,
        "line integral: {} ≈ {}",
        geo_integral.mag,
        trad_integral
    );
}

#[test]
fn its_a_surface_integral() {
    // surface = wedge product of edges
    // magnitude IS the area, grade IS the orientation

    let edge_x = Geonum::new_from_cartesian(2.0, 0.0);
    let edge_y = Geonum::new_from_cartesian(0.0, 3.0);

    let surface = edge_x.wedge(&edge_y);

    assert!(
        (surface.mag - 6.0).abs() < EPSILON,
        "surface area = 2 × 3 = 6"
    );
    assert_eq!(surface.angle.grade(), 2, "surface at grade 2 (bivector)");
}

#[test]
fn its_a_volume_integral() {
    // volume = geometric product of surface bivector with third edge
    // magnitude IS the volume

    let edge_x = Geonum::new_from_cartesian(2.0, 0.0);
    let edge_y = Geonum::new_from_cartesian(0.0, 3.0);
    let edge_z = Geonum::new_with_blade(4.0, 2, 0.0, 1.0);

    let surface = edge_x.wedge(&edge_y);
    let volume = surface.geo(&edge_z);

    assert!(
        (volume.mag - 24.0).abs() < EPSILON,
        "volume = 2 × 3 × 4 = 24"
    );
    assert_eq!(volume.angle.grade(), 0, "volume cycles back to grade 0");
}

// ═══════════════════════════════════════════════════════════════════════════════
// the power rule was never a rule
//
// it was two projections read from a geometric number:
// 1. how many times did the angle accumulate? (the power)
// 2. what magnitude remains after removing one factor? (the base)
//
// limits compute the same answer by approaching from outside
// the geometric number already contains it from inside
//
// differentiation is π/2 rotation. integration is 3π/2 forward rotation.
// the fundamental theorem connects accumulated rotation to endpoint interference.
// vector calculus and geometric integrals use the same angle arithmetic.
//
// "scalars are projections" — and calculus is the set of projections
// that extract rate information from angle space
// ═══════════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
// helpers — the patterns geonum has no named api for yet
// ---------------------------------------------------------------------------

// the rate of f against x: turn both a quarter turn, count, and divide. magnitudes divide,
// angles subtract, the quarter turns cancel
// differentiate_test.rs proves this is the power, sum, product and chain rules
//
// the count is how many times x went into f. whoever made f has it and hands it over, and
// it is proven here in angle space: the angle of x taken that many times is the angle of
// f. angles add, and no angle is read out as a float to find it
fn turn_and_divide(f: &Geonum, x: &Geonum, count: f64) -> Geonum {
    assert!(
        f.angle.near(&(x.angle * count)),
        "x went into f {count} times"
    );
    let df = f.differentiate().scale(count);
    let dx = x.differentiate();
    Geonum::new_with_angle(df.mag / dx.mag, df.angle - dx.angle)
}

// the antiderivative of rate against x: turn_and_divide run backward. multiply by the
// quarter turn of x, rotate the other three quarter turns with integrate(), drop the full
// turn that makes with base_angle(), then divide by the count. integral_test.rs proves it
//
// the count is how many times x has gone into the result, one more than into the rate. it
// is handed over and proven the same way, in angle space
fn turn_back_and_divide(rate: &Geonum, x: &Geonum, count: f64) -> Geonum {
    let around = (*rate * x.differentiate()).integrate().base_angle();
    assert!(
        around.angle.near(&(x.angle * count).base_angle()),
        "x went into the result {count} times"
    );
    around.scale(1.0 / count)
}
