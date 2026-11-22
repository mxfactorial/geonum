// differentiation is a quarter turn, counted and compared
//
// a geometric number moves two ways: around, by its quarter turn, and out, along its ray
// differentiate() is the first. every rule of differential calculus is that turn
// counted, compared or added:
//
//   rate of a projection    turn the number, project again
//   power rule              x^n turns n times as far as x, so its turn counts n
//                           divide by the turn of x and the quarter turns cancel
//   sum rule                each term turns at its own rate
//   product rule            angles add, so the product turns as far as both factors
//   chain rule              turn counts multiply
//   growing while turning   the out motion and the around motion add
//   autodiff                rotate a unit a quarter turn beside x, multiply, read blade 1
//   forward and backward    every rate in a program, each from its own count
//   negative powers         the count is below 0, and arrives as a half turn
//
// no limit is taken to get a rate. where a limit appears below, its there to show
// it approaching the number the turn already landed on
//
// calculus_test.rs reads the power rule off the angle as two ratios. this suite
// proves those ratios are what the quarter turns leave when they cancel
//
// rotate, then read. the turning is angle space, and it only adds. the rate is a reading:
// one number read against another, magnitudes divided and angles diffed, times the count.
// the division, the diff and any minus sign are in the read, the line
// angle_forward_test.rs holds
//
// the count is never read out of an angle. it is how many times x went into the thing, and
// whoever made the thing has it. each test hands it over and proves it in angle space: the
// angle of x taken that many times is the angle of the thing, or, in the last four tests,
// x a quarter turn on puts the thing that many quarter turns on. angles add
//
// run: cargo test --test differentiate_test

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// something made from x, so it can be made again from x a quarter turn on
type Making = Box<dyn Fn(Geonum) -> Geonum>;

#[test]
fn it_reads_the_rate_of_a_projection_off_the_quarter_turn() {
    // 10 cos(t) is the dimension 0 reading of [10, t]. turn the number a quarter turn
    // and the same reading is the rate, -10 sin(t): rotate, then project

    let position = Geonum::new(10.0, 1.0, 3.0); // [10, π/3]
    let (cos, sin) = position.angle.cos_sin();
    assert!((position.project_to_dimension(0) - 10.0 * cos).abs() < EPSILON);

    let velocity = position.differentiate(); // [10, π/3 + π/2]
    let rate = velocity.project_to_dimension(0);
    assert!(
        (rate + 10.0 * sin).abs() < EPSILON,
        "d/dt[10 cos t] = -10 sin t"
    );

    // the limit closes in on the same rate as its step shrinks, without landing on it
    let reading = |divisor: f64| {
        let h = Angle::new(1.0, divisor); // a step of π/divisor
        let ahead = position.rotate(h);
        (ahead.project_to_dimension(0) - position.project_to_dimension(0)) / h.grade_angle()
    };
    let coarse = (reading(100.0) - rate).abs();
    let fine = (reading(10000.0) - rate).abs();
    assert!(coarse > fine, "a smaller step lands closer");
    assert!(fine > EPSILON, "and still short of the turn");
}

// the rate of f against x: turn both a quarter turn, count, and divide. magnitudes divide,
// angles subtract, the quarter turns cancel
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

#[test]
fn it_proves_the_two_ratios_are_the_quarter_turns_cancelling() {
    // the power rule tests above multiply an angle ratio by a magnitude ratio
    // those two ratios are differentiate() applied to x^n and to x:
    //
    //   x^n turned: [n r^n, nθ + π/2]    it turns n times as far as x
    //   x turned:   [r, θ + π/2]
    //   divided:    [n r^(n-1), (n-1)θ]  the quarter turns cancel, n x^(n-1) is left

    let x = Geonum::new(3.0, 1.0, 12.0); // [3, π/12]

    for n in [1.0, 2.0, 3.0, 5.0, 1.5] {
        let f = x.pow(n);
        let derivative = turn_and_divide(&f, &x, n);

        // n x^(n-1) as a geonum, on the ray of x^(n-1)
        assert!(
            derivative.near(&x.pow(n - 1.0).scale(n)),
            "d/dx[x^{n}] = {n} x^{}",
            n - 1.0
        );

        // its magnitude is the two ratios multiplied
        let angle_ratio = f.angle.grade_angle() / x.angle.grade_angle();
        let mag_ratio = f.mag / x.mag;
        assert!(
            derivative.near_mag(angle_ratio * mag_ratio),
            "the two ratios are what the quarter turns leave"
        );
    }
}

#[test]
fn it_differentiates_a_sum_term_by_term() {
    // d/dx[x³ + 2x²] = 3x² + 4x. each term turns at its own rate, so each gets its
    // own turn and divide. the sum rule is no rule: the terms never shared an angle

    let x = Geonum::new(3.0, 1.0, 12.0); // [3, π/12]
    let cubic = x.pow(3.0); // [27, 3π/12]
    let quadratic = x.pow(2.0).scale(2.0); // [18, 2π/12]

    let d_cubic = turn_and_divide(&cubic, &x, 3.0); // [27, 2π/12] = 3x²
    let d_quadratic = turn_and_divide(&quadratic, &x, 2.0); // [12, π/12] = 4x

    assert!(d_cubic.near(&x.pow(2.0).scale(3.0)), "3x² on the ray of x²");
    assert!(d_quadratic.near(&x.scale(4.0)), "4x on the ray of x");

    // the scalar derivative is the two magnitudes read together: 27 + 12
    assert!(
        (d_cubic.mag + d_quadratic.mag - 39.0).abs() < EPSILON,
        "3x² + 4x = 39 at x = 3"
    );

    // and a sum goes term by term because it has no one count. turn x a quarter turn and x²
    // goes two quarter turns on while x³ goes three, so the sum made again is each term
    // turned by its own count, added
    let x = Geonum::new(3.0, 1.0, 6.0);
    let sum: Making = Box::new(|x| x.pow(2.0) + x.pow(3.0));
    let square: Making = Box::new(|x| x.pow(2.0));
    let cube: Making = Box::new(|x| x.pow(3.0));
    let each_by_its_own =
        square(x).rotate(Angle::new(2.0, 2.0)) + cube(x).rotate(Angle::new(3.0, 2.0));
    assert!(sum(x.differentiate())
        .base_angle()
        .near(&each_by_its_own.base_angle()));

    // so the rate is taken term by term: [6, π/6] + [27, π/3], 32.3356
    let term_by_term = rate(&square, &x, 2.0) + rate(&cube, &x, 3.0);
    assert!((term_by_term.mag - 32.3356).abs() < 1e-4);
    assert!(term_by_term.near(&(Geonum::new(6.0, 1.0, 6.0) + Geonum::new(27.0, 1.0, 3.0))));

    // backward is the same: term by term it lands on the sum
    let rate_of_square: Making = Box::new(|x| x.scale(2.0));
    let rate_of_cube: Making = Box::new(|x| x.pow(2.0).scale(3.0));
    assert!((back(&rate_of_square, &x, 2.0) + back(&rate_of_cube, &x, 3.0)).near(&sum(x)));
}

#[test]
fn it_proves_the_product_rule_is_angles_adding() {
    // (fg)' = f'g + fg' reassembles what multiplication already did: angles add,
    // so the product turns as far as both factors together, 2 + 3 = 5

    let x = Geonum::new(2.0, 1.0, 16.0); // [2, π/16]
    let f = x.pow(2.0); // [4, 2π/16]
    let g = x.pow(3.0); // [8, 3π/16]
    let product = f * g; // [32, 5π/16]

    // the product sits where the two factors angles add up to, and that is the angle of x
    // taken 2 + 3 times
    assert!(product.angle.near(&(f.angle + g.angle)));
    assert!(
        product.angle.near(&(x.angle * 5.0)),
        "x went into the product 2 + 3 times"
    );

    // one turn and divide on the product: 5x⁴ = 80
    let derivative = turn_and_divide(&product, &x, 5.0);
    assert!(derivative.near(&x.pow(4.0).scale(5.0)), "(x²x³)' = 5x⁴");
    assert!(derivative.near_mag(80.0), "5 × 2⁴ = 80");

    // the textbook route separates the factors and adds them back: 32 + 48
    let separated = turn_and_divide(&f, &x, 2.0) * g + f * turn_and_divide(&g, &x, 3.0);
    assert!(
        separated.near(&derivative),
        "f'g + fg' lands on the same number"
    );
}

#[test]
fn it_proves_the_chain_rule_is_turn_counts_multiplying() {
    // d/dx[(x²)³] = 3(x²)² × 2x decomposes what the angle already holds: the inner
    // function turns 2 times as far as x, the outer 3 times as far as the inner,
    // and the composition 3 × 2 = 6 times as far as x

    let x = Geonum::new(2.0, 1.0, 16.0); // [2, π/16]
    let inner = x.pow(2.0); // [4, 2π/16]
    let outer = inner.pow(3.0); // [64, 6π/16]

    assert!(
        outer.angle.near(&(x.angle * (3.0 * 2.0))),
        "turn counts multiply: 3 × 2"
    );

    // one turn and divide against x: 6x⁵ = 192
    let derivative = turn_and_divide(&outer, &x, 6.0);
    assert!(derivative.near(&x.pow(5.0).scale(6.0)), "d/dx[(x²)³] = 6x⁵");
    assert!(derivative.near_mag(192.0), "6 × 2⁵ = 192");

    // the textbook route links two rates: outer against inner, times inner against x
    let linked = turn_and_divide(&outer, &inner, 3.0) * turn_and_divide(&inner, &x, 2.0);
    assert!(
        linked.near(&derivative),
        "3(x²)² × 2x lands on the same number"
    );
}

#[test]
fn it_differentiates_a_number_that_grows_while_it_turns() {
    // t² sin(t) is one number read on dimension 1: magnitude t², angle t
    // it moves two ways at once, out along its ray and around by its quarter turn
    // the textbook product rule 2t sin(t) + t² cos(t) is those two motions projected

    let t = PI / 3.0;
    let angle = Angle::new(1.0, 3.0); // t as an angle
    let number = Geonum::new_with_angle(t * t, angle); // [t², t]
    assert!((number.project_to_dimension(1) - t * t * t.sin()).abs() < EPSILON);

    // out: the magnitude t² grows at 2t, the power rule, along the same ray
    let t_geo = Geonum::new(t, 1.0, 12.0);
    let growth = turn_and_divide(&t_geo.pow(2.0), &t_geo, 2.0).mag; // 2t
    let out = Geonum::new_with_angle(growth, angle);

    // around: the angle advances at unit rate, a bare quarter turn
    let around = number.differentiate();

    // rotate then project: the two motions added and read on dimension 1
    let rate = (out + around).project_to_dimension(1);
    let textbook = 2.0 * t * t.sin() + t * t * t.cos();
    assert!(
        (rate - textbook).abs() < EPSILON,
        "(t² sin t)' with no limit"
    );

    // the limit approaches the same rate from outside
    let h = 1e-6;
    let limit = ((t + h).powi(2) * (t + h).sin() - t * t * t.sin()) / h;
    assert!((rate - limit).abs() < 1e-4, "the limit closes in on it");
}

#[test]
fn it_automates_differentiation_by_rotating_a_unit_beside_x() {
    // forward autodiff carries a rate beside a value through every multiplication
    // the geometry underneath: rotate a unit a quarter turn, add it to x and multiply as usual
    // the product sorts itself by blade, and nothing has to be set to zero:
    //
    //   blade 0   the value
    //   blade 1   the rate: every term that took the quarter turn once
    //   blade 2   the quarter turn taken twice: the second derivative over 2!
    //   blade 3   the quarter turn taken three times: the third over 3!
    //
    // x³ at x = 3

    let value = Geonum::new(3.0, 0.0, 1.0); // [3, 0]
    let rotated = Geonum::new(1.0, 0.0, 1.0).rotate(Angle::new(1.0, 2.0)); // [1, π/2]

    // (value + rotated)³ with every term kept
    let mut terms = vec![value, rotated];
    for _ in 0..2 {
        terms = terms
            .iter()
            .flat_map(|t| [*t * value, *t * rotated])
            .collect();
    }

    // add up what landed on each blade
    let landed = |blade: usize| -> f64 {
        terms
            .iter()
            .filter(|t| t.angle.blade() == blade)
            .map(|t| t.mag)
            .sum()
    };

    assert!((landed(0) - 27.0).abs() < EPSILON, "blade 0 holds x³ = 27");
    assert!((landed(1) - 27.0).abs() < EPSILON, "blade 1 holds 3x² = 27");
    assert!(
        (landed(2) - 9.0).abs() < EPSILON,
        "blade 2 holds f''/2! = 9"
    );
    assert!(
        (landed(3) - 1.0).abs() < EPSILON,
        "blade 3 holds f'''/3! = 1"
    );

    // three terms took the quarter turn once, one per factor of x. autodiff finds them a
    // multiplication at a time. x went into x³ three times, so one turn and divide gives
    // the same rate
    let once = terms.iter().filter(|t| t.angle.blade() == 1).count();
    assert_eq!(once, 3, "one blade 1 term per factor of x");

    let x = Geonum::new(3.0, 1.0, 12.0); // [3, π/12]
    assert!(
        turn_and_divide(&x.pow(3.0), &x, 3.0).near_mag(landed(1)),
        "the same 27 from one turn and divide"
    );
}

#[test]
fn it_reads_forward_and_backward_rates_off_the_angles() {
    // forward autodiff runs a program on x with a second slot riding along, the rate,
    // updated by its own rule at every multiplication. reverse autodiff records the
    // program on a tape and walks it backward
    //
    // the geometry underneath: the second slot and the tape are the angle. give x an angle
    // and run the program as plain multiplication. angles add at every step, so every value
    // sits at the angle of x taken as many times as x went into it

    let x = Geonum::new(3.0, 1.0, 12.0); // [3, π/12]

    // the program: three multiplications, nothing carried beside the values
    let a = x * x; // x²
    let b = a * x; // x³
    let f = a * b; // x⁵

    // forward: the rate of any value against x, taken when asked
    assert!(turn_and_divide(&a, &x, 2.0).near_mag(6.0), "da/dx = 2x = 6");
    assert!(
        turn_and_divide(&b, &x, 3.0).near_mag(27.0),
        "db/dx = 3x² = 27"
    );
    assert!(
        turn_and_divide(&f, &x, 5.0).near_mag(405.0),
        "df/dx = 5x⁴ = 405"
    );

    // backward: the rate of the output against a value inside the program
    // f turns 5/2 times as far as a, so df/da = (5/2) a^(3/2) = 67.5, with no tape replayed
    let df_da = turn_and_divide(&f, &a, 2.5);
    assert!(df_da.near_mag(67.5), "df/da = 2.5 × 9^1.5 = 67.5");

    // and the two link up: df/da × da/dx = df/dx
    let linked = df_da * turn_and_divide(&a, &x, 2.0);
    assert!(linked.near(&turn_and_divide(&f, &x, 5.0)), "67.5 × 6 = 405");
}

#[test]
fn it_differentiates_negative_powers_stored_whole_turns_on() {
    // d/dx[x⁻²] = -2x⁻³. a forward only angle has no -2θ to hold, so x⁻² is stored whole
    // turns on, at 2π - 2θ

    let x = Geonum::new(3.0, 1.0, 12.0); // [3, π/12]

    // so the angle alone cant tell x⁻² from x²²: same blade, same remainder
    let (inverse_square, twenty_second) = (x.pow(-2.0), x.powi(22));
    assert_eq!(inverse_square.angle.blade(), twenty_second.angle.blade());
    assert!(inverse_square.angle.near_rem(twenty_second.angle.rem()));

    // and it does not have to. the count is not read back out of the angle. whoever made
    // x⁻² handed pow the −2, and the angle of x taken −2 times is the angle it sits at
    assert!(inverse_square.angle.near(&(x.angle * -2.0)));

    // with the count handed over, the same turn and divide gives n x^(n-1), and the minus
    // sign arrives as a half turn
    for n in [-1.0_f64, -2.0, -3.0, -0.5, -25.0, -30.0, 2.0, 5.0, 30.0] {
        let derivative = turn_and_divide(&x.pow(n), &x, n);

        // the textbook n x^(n-1) at the same x: its size, then its direction
        let size = n * 3.0_f64.powf(n - 1.0);
        let direction = (n - 1.0) * PI / 12.0;
        assert!((derivative.mag / size.abs() - 1.0).abs() < 1e-6);
        let (cos, sin) = derivative.angle.cos_sin();
        assert!((cos - size.signum() * direction.cos()).abs() < 1e-6);
        assert!((sin - size.signum() * direction.sin()).abs() < 1e-6);
    }

    // a coefficient in front leaves the count alone: 5x⁻² and 100x⁻² still count -2
    for c in [5.0, 100.0] {
        let rate = turn_and_divide(&x.pow(-2.0).scale(c), &x, -2.0);
        assert!(rate.near_mag(2.0 * c / 27.0), "|-2c x⁻³| = 2c/27");
    }

    // a 1/d² field at d = 10 changes at 2/d³, in one step where two samples and a
    // subtraction only estimate it
    let d = Geonum::new(10.0, 1.0, 12.0);
    let rate = turn_and_divide(&d.pow(-2.0), &d, -2.0);
    assert!(rate.near_mag(0.002), "2/d³ = 0.002");
}

#[test]
fn it_counts_the_steps_by_turning_x_a_quarter_turn() {
    // turn x a quarter turn and make the same thing again. each x in it adds a quarter
    // turn and nothing else moves. that is angle space, and it only adds
    //
    // so the count is how many quarter turns on the thing lands, whatever else is in its
    // angle: a coefficient, a half turn, a whole turn of history, or an x that sits at
    // angle 0
    let c = Geonum::new(5.0, 1.0, 4.0);
    let rows: Vec<(Geonum, Making, usize)> = vec![
        // x × x
        (Geonum::new(3.0, 1.0, 6.0), Box::new(|x| x * x), 2),
        // x⁷ with x at π/3, past a whole turn
        (Geonum::new(3.0, 1.0, 3.0), Box::new(|x| x.pow(7.0)), 7),
        // x at angle 0
        (Geonum::scalar(3.0), Box::new(|x| x * x), 2),
        (Geonum::scalar(3.0), Box::new(|x| x * x * x), 3),
        // a coefficient at π/4
        (
            Geonum::new(3.0, 1.0, 6.0),
            Box::new(move |x| c * x.pow(3.0)),
            3,
        ),
        // a half turn on the whole thing
        (
            Geonum::new(3.0, 1.0, 6.0),
            Box::new(|x| x.pow(3.0).negate()),
            3,
        ),
        // a whole turn of history, from a quarter turn and integrate()
        (
            Geonum::new(3.0, 1.0, 6.0),
            Box::new(|x| (x * x).differentiate().integrate()),
            2,
        ),
        // none of x in it
        (
            Geonum::new(3.0, 1.0, 6.0),
            Box::new(|_| Geonum::scalar(5.0)),
            0,
        ),
    ];

    for (x, make, count) in &rows {
        assert!(counts(make, x, *count as f64));

        // the libs Angle - Angle reads the same two angles, and its blade is the count
        let diff = make(x.differentiate()).angle - make(*x).angle;
        assert_eq!(diff.blade(), *count, "the count is the blade of the diff");
        assert!(diff.near(&Angle::new(*count as f64, 2.0)), "no remainder");
    }

    // where x is the one thing in the angle, the same count is the angle of x taken that
    // many times. that is the form turn_and_divide proves
    let x = Geonum::new(3.0, 1.0, 3.0);
    assert!(x.pow(7.0).angle.near(&(x.angle * 7.0)));
}

#[test]
fn it_reads_the_rate_both_ways() {
    // rotate, then read. the rate is a reading: what was made, read against x, times the
    // count. reading one number against another divides the magnitudes and diffs the angles.
    // for x × x with x at [3, π/6] the read is [18 / 3, 1 − 1, π/3 − π/6]
    //
    // forward: [count × F / r, φ − θ]
    // backward: x multiplied in once more, over the count that makes
    let c = Geonum::new(5.0, 1.0, 4.0);
    let at_an_angle = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]

    // three steps out read against x is two steps out
    let two_out = at_an_angle * at_an_angle;
    assert!(read_against(&(two_out * at_an_angle), &at_an_angle).near(&two_out));

    let rows: Vec<(Geonum, Making, f64, Geonum, Making)> = vec![
        // x × x: the textbooks 2x at 3 is 6
        (
            at_an_angle,
            Box::new(|x| x * x),
            2.0,
            Geonum::new(6.0, 1.0, 6.0),
            Box::new(|x| x.scale(2.0)),
        ),
        // x⁷ with x at π/3: 7x⁶ is 5103, at 6 × π/3
        (
            Geonum::new(3.0, 1.0, 3.0),
            Box::new(|x| x.pow(7.0)),
            7.0,
            Geonum::new(5103.0, 2.0, 1.0),
            Box::new(|x| x.pow(6.0).scale(7.0)),
        ),
        // x at angle 0: 6 and 27
        (
            Geonum::scalar(3.0),
            Box::new(|x| x * x),
            2.0,
            Geonum::scalar(6.0),
            Box::new(|x| x.scale(2.0)),
        ),
        (
            Geonum::scalar(3.0),
            Box::new(|x| x * x * x),
            3.0,
            Geonum::scalar(27.0),
            Box::new(|x| (x * x).scale(3.0)),
        ),
        // a coefficient at π/4: 15x² is 135, at π/4 plus π/3
        (
            at_an_angle,
            Box::new(move |x| c * x.pow(3.0)),
            3.0,
            Geonum::new(135.0, 7.0, 12.0),
            Box::new(move |x| c * x.pow(2.0).scale(3.0)),
        ),
        // a half turn on the whole thing: 27, a half turn on from π/3
        (
            at_an_angle,
            Box::new(|x| x.pow(3.0).negate()),
            3.0,
            Geonum::new(27.0, 4.0, 3.0),
            Box::new(|x| x.pow(2.0).scale(3.0).negate()),
        ),
    ];

    for (x, make, count, textbook_rate, make_the_rate) in &rows {
        assert!(rate(make, x, *count).near(textbook_rate));

        // backward, from the textbooks rate, lands on what was made
        assert!(back(make_the_rate, x, *count).near(&make(*x)));
    }

    // a number with none of x in it counts 0, so its rate has no magnitude
    let constant: Making = Box::new(|_| Geonum::scalar(5.0));
    assert!(rate(&constant, &at_an_angle, 0.0).near_mag(0.0));
}

#[test]
fn it_reads_a_turn_back_from_blades_it_was_given() {
    // a count below 0 is a turn back. to see one blade for blade, give the number its blades
    // first: start two whole turns on, at blade 8, and make the power by reading the start
    // against x^k, once, at the making. nothing wraps
    //
    // then x a quarter turn on leaves the thing k quarter turns behind where it was. in angle
    // space that is the thing made before sitting k quarter turns on from the thing made
    // after. angles only add
    //
    // the second x has magnitude 1, where every power has magnitude 1 and the blades alone
    // hold the count
    let start = Geonum::new_with_blade(1.0, 8, 0.0, 1.0);
    for x in [Geonum::new(3.0, 1.0, 6.0), Geonum::new(1.0, 1.0, 6.0)] {
        for steps in [0.5, 2.0, 3.0] {
            let make: Making = Box::new(move |x| read_against(&start, &x.pow(steps)));
            assert!(counts(&make, &x, -steps));
        }
    }

    // pow gives no say over the blades. it stores a power that turns back whole turns on, so
    // blade for blade x⁻² by pow goes 2 quarter turns on where the one given its blades goes
    // 2 back. the two land on the same grade: it goes back by going on
    let x = Geonum::new(3.0, 1.0, 6.0);
    let by_pow: Making = Box::new(|x| x.pow(-2.0));
    let given_blades: Making = Box::new(move |x| read_against(&start, &x.pow(2.0)));
    assert!(counts(&by_pow, &x, 2.0));
    assert!(counts(&given_blades, &x, -2.0));

    // with −2 handed over the rate of x⁻² is the textbooks −2x⁻³: 2/27, in the place the lib
    // puts it
    let rate_of_inverse_square = rate(&given_blades, &x, -2.0);
    assert!(rate_of_inverse_square.near_mag(2.0 / 27.0));
    assert!(rate_of_inverse_square
        .base_angle()
        .near(&x.pow(-3.0).scale(-2.0).base_angle()));

    // √x has a rate that turns back, 0.5 × x^(−0.5). from a start with blades it is 0.2887,
    // in the place the lib puts it, with no whole turn borrowed
    let root: Making = Box::new(move |x| start * x.pow(0.5));
    let rate_of_root = rate(&root, &x, 0.5);
    assert!((rate_of_root.mag - 0.2887).abs() < 1e-4);
    assert!(rate_of_root
        .base_angle()
        .near(&x.pow(-0.5).scale(0.5).base_angle()));

    // and √x comes back from it. x went into the rate −0.5 times, one more x multiplied in
    // makes it 0.5, and it lands on √x
    let rate_of_root_made: Making = Box::new(move |x| read_against(&start, &x.pow(0.5)).scale(0.5));
    assert!(counts(&rate_of_root_made, &x, -0.5));
    assert!(back(&rate_of_root_made, &x, 0.5)
        .base_angle()
        .near(&x.pow(0.5)));
}

#[test]
fn it_holds_the_count_in_the_magnitude_too() {
    // magnitudes multiply, so the magnitude holds the count as the angle does, and it says
    // which way: a power that turns back shrinks it. 3 multiplied in −2 times is 1/9
    let x = Geonum::new(3.0, 1.0, 6.0);
    assert!(x.pow(-2.0).near_mag(1.0 / 9.0));
    for power in [-2.0, -0.5, 0.5, 2.0, 3.5, 7.0] {
        assert!(x.pow(power).near_mag(x.mag.powf(power)));
    }

    // each of the two has one value that keeps no count. a magnitude of 1 multiplied in any
    // number of times is 1, so every power holds 1
    let at_1 = Geonum::new(1.0, 1.0, 6.0);
    for power in [-2.0, -0.5, 0.5, 2.0, 3.5, 7.0] {
        assert!(at_1.pow(power).near_mag(1.0), "it holds 1");
    }

    // and an angle of 0 added any number of times is 0, so every power holds 0
    let at_0 = Geonum::scalar(3.0);
    for power in [0.5, 2.0, 3.5, 7.0] {
        assert!(
            at_0.pow(power).angle.near(&Angle::new(0.0, 1.0)),
            "it holds 0"
        );
    }
}

// ---------------------------------------------------------------------------
// helpers for the four tests above
// ---------------------------------------------------------------------------

// the count, proven from the making: x a quarter turn on puts what was made count quarter
// turns on. angles add. a count below 0 is a turn back, and in angle space that is the
// thing made before sitting that many quarter turns on from the thing made after
fn counts(make: &Making, x: &Geonum, count: f64) -> bool {
    let (before, after) = (make(*x).angle, make(x.differentiate()).angle);
    if count >= 0.0 {
        after.near(&(before + Angle::new(count, 2.0)))
    } else {
        before.near(&(after + Angle::new(-count, 2.0)))
    }
}

// one number read against another: the magnitudes divided, the angles diffed
fn read_against(f: &Geonum, x: &Geonum) -> Geonum {
    Geonum::new_with_angle(f.mag / x.mag, f.angle - x.angle)
}

// the rate of the making against x: what was made read against x, times the count
fn rate(make: &Making, x: &Geonum, count: f64) -> Geonum {
    assert!(counts(make, x, count), "x went into it {count} times");
    read_against(&make(*x), x).scale(count)
}

// backward: x multiplied in once more, over the count that makes
fn back(make_the_rate: &Making, x: &Geonum, count: f64) -> Geonum {
    assert!(
        counts(make_the_rate, x, count - 1.0),
        "x went into the rate one time fewer than {count}"
    );
    (make_the_rate(*x) * *x).scale(1.0 / count)
}
