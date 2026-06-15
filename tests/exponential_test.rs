//! exponential test
//!
//! eˣ in geonum. the 'e' is scaffolding for rotation: its_a_eulers_identity
//! (numbers_test:1022) shows e^(iπ) = [1, π], so the imaginary exponential IS the
//! angle — e^(iθ) = [1, θ], a pure rotation, no limit, no symbol. e splits across
//! geonum's two numbers: rotation in the angle (e^(iθ)), growth in the magnitude (eˣ).
//!
//! - the brute product (1 + x/n)ⁿ folds the magnitude to eˣ — the value lives there
//! - e^(iθ) = [1, θ] is a unit rotation, closed under differentiate (+π/2) and integrate
//!   (−π/2). with the dual-number autodiff (numbers_test:176, differentiation IS the π/2
//!   rotation), this is why the exponential is its own derivative: its analytic derivative
//!   equals that rotation
//! - ∫₀¹ eˣ dx = e − 1 from the two boundary magnitudes: eˣ is its own antiderivative
//! - (1 + (x/n)·e^(iφ))ⁿ is a projection: the step's real part becomes growth (the
//!   magnitude), its imaginary part becomes turn (the angle), and the two share the one
//!   step — growth² + turn² = step²
//!
//! run: cargo test --test exponential_test

use geonum::Geonum;

// ---------------------------------------------------------------------------
// the brute product folds the magnitude to eˣ
// ---------------------------------------------------------------------------
#[test]
fn it_folds_the_brute_product_into_the_magnitude() {
    use std::f64::consts::E;

    // (1 + x/n)ⁿ is a pure product. with the step along the real axis the magnitude
    // folds to eˣ as n grows — the value of the exponential lives in the magnitude
    for n in [10u32, 1000, 100_000] {
        println!(
            "(1 + 1/{n})^{n} = {:.6}  (→ e ≈ {:.6})",
            exp_step(1.0, 0.0, 1.0, n).mag,
            E
        );
    }
    assert!(
        (exp_step(1.0, 0.0, 1.0, 1_000_000).mag - E).abs() < 1e-3,
        "the magnitude folds to e"
    );
    assert!(
        (exp_step(2.0, 0.0, 1.0, 1_000_000).mag - E * E).abs() < 1e-2,
        "genuinely eˣ: x=2 folds to e²"
    );
}

// ---------------------------------------------------------------------------
// the imaginary exponential never left the angle — e is rotation, no scaffolding
// ---------------------------------------------------------------------------
#[test]
fn it_finds_the_exponential_already_living_in_the_angle() {
    // its_a_eulers_identity: e^(iπ) = [1, π], the 'e' is notation for rotation. so the
    // imaginary exponential doesnt run out of the angle — it IS the angle. e^(iθ) =
    // [1, θ], a unit rotation, no growth, no limit
    for &(p, d) in &[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)] {
        let e_i_theta = Geonum::new(1.0, p, d); // e^(iθ) = [1, θ], unit by construction

        // d/dθ e^(iθ) = i·e^(iθ): differentiate rotates +π/2, magnitude preserved — the
        // exponential is the eigenfunction of differentiation, and it never leaves mag 1.
        // integrate (the −π/2 tick) returns another unit rotation: rotation in, rotation out
        assert!(
            e_i_theta.differentiate().near_mag(1.0),
            "the derivative is another unit rotation"
        );
        assert!(
            e_i_theta.integrate().near_mag(1.0),
            "and so is the integral — the rotation closes on itself"
        );
    }

    // the euler relation itself: e^(iπ) = [1, π] is its own inverse, [1,π]·[1,π] = [1, 2π] = 1
    let e_ipi = Geonum::new(1.0, 1.0, 1.0);
    let squared = e_ipi * e_ipi;
    assert!(
        squared.near_mag(1.0) && squared.angle.near_rad(0.0),
        "[1, π]² = [1, 0] = 1"
    );
}

// ---------------------------------------------------------------------------
// the integral gives way through the fixed point: ∫₀¹ eˣ = e − 1
// ---------------------------------------------------------------------------
#[test]
fn it_gives_up_the_integral_through_the_fixed_point() {
    use std::f64::consts::E;

    // eˣ is its own antiderivative — the fixed point of the fold. ∫₀¹ eˣ dx is just the
    // two boundary values e¹ − e⁰: no power raised, no divisor read. the value comes from
    // the magnitude the product converged in, not from any exponent in the angle — eˣ is
    // where the power-in-the-angle fold reaches its fixed point
    let n = 1_000_000;
    let integral = exp_step(1.0, 0.0, 1.0, n).mag - exp_step(0.0, 0.0, 1.0, n).mag; // e¹ − e⁰
    assert!(
        (integral - (E - 1.0)).abs() < 1e-3,
        "∫₀¹ eˣ dx = e − 1 = {}, got {integral}",
        E - 1.0
    );
    println!(
        "∫₀¹ eˣ dx = {integral:.6}  (e − 1 ≈ {:.6}, F = f, from the boundary magnitudes)",
        E - 1.0
    );
}

// ---------------------------------------------------------------------------
// the product is a projection: the step splits into shared growth and turn
// ---------------------------------------------------------------------------
#[test]
fn it_projects_the_step_into_shared_growth_and_turn() {
    use std::f64::consts::E;
    let (x, n) = (1.0, 1_000_000);

    // a step along the real axis projects entirely into GROWTH: magnitude → eˣ, no turn
    let real = exp_step(x, 0.0, 1.0, n); // step at angle 0
    assert!(
        (real.mag - E).abs() < 1e-3,
        "real step → eˣ in the magnitude"
    );
    assert!(real.angle.grade_angle().abs() < 1e-3, "no turn");

    // a step perpendicular to it projects entirely into TURN: magnitude → 1, angle → x.
    // this is e^(ix) = [1, x], the rotation the euler test names. the angle reaches x
    // because pow(n) accumulates the step's tiny angle (≈ atan(x/n) ≈ x/n) n times — the
    // load-bearing assumption is that Geonum's `pow` scales the angle by exactly n
    let imag = exp_step(x, 1.0, 2.0, n); // step at π/2
    assert!(
        (imag.mag - 1.0).abs() < 1e-3,
        "perpendicular step → no growth, magnitude 1"
    );
    assert!(
        (imag.angle.grade_angle() - x).abs() < 1e-3,
        "angle → x: e^(ix) = [1, x]"
    );

    // a tilted step SHARES itself between the two. at φ the one step of length x splits:
    // growth takes x·cosφ (it becomes ln of the magnitude), turn takes x·sinφ (it becomes
    // the angle). they are the legs of a right triangle whose hypotenuse is the whole
    // step — growth² + turn² = x². that conservation is nothing new: it is the
    // quadrature identity cos²φ + sin²φ = 1 that trigonometry_test proves as a projection
    // (it_is_projection, it_derives_pythagorean_identity_from_quadrature). the new part is
    // that the exponential is what does the projecting — the step IS the hypotenuse
    let tilted = exp_step(x, 1.0, 4.0, n); // step at π/4
    let growth = tilted.mag.ln(); // x·cos(π/4)
    let turn = tilted.angle.grade_angle(); // x·sin(π/4)
    assert!(
        (growth.hypot(turn) - x).abs() < 1e-3,
        "growth and turn share the one step: √(growth² + turn²) = x = {x}, got {}",
        growth.hypot(turn)
    );

    println!(
        "step {x} at π/4 → growth(ln mag) {growth:.4} + turn(angle) {turn:.4}, √(g²+t²) = {:.4}",
        growth.hypot(turn)
    );
}

// ───────────────────────────────────────────────────────────────────────────
// helper
// ───────────────────────────────────────────────────────────────────────────

/// (1 + (x/n)·e^(iφ))ⁿ → e^(x·e^(iφ)): hold the real unit "1" fixed (the conserved
/// anchor) and fold in a small step of length x/n pointed at angle φ = step_p·π/step_d.
/// the step's real projection becomes growth in the magnitude, its imaginary projection
/// becomes turn in the angle
fn exp_step(x: f64, step_p: f64, step_d: f64, n: u32) -> Geonum {
    let unit = Geonum::new(1.0, 0.0, 1.0); // the real unit, the conserved anchor
    let step = Geonum::new(x / n as f64, step_p, step_d); // (x/n) at angle φ
    (unit + step).pow(n as f64) // the n-fold product
}
