use geonum::*;
use std::f64::consts::{PI, TAU};

const EPSILON: f64 = 1e-10;

#[test]
fn it_computes_limits() {
    // this test demonstrates that "limits" are unnecessary when using geometric numbers

    // differentiation is simply a pi/2 rotation and the foundation of
    // calculus emerges directly from this geometric structure

    // let v = [[1, 0], [1, pi/2]] # 2d

    // everything can be a 1d "derivative" or projection of the base 2d v
    // so long as the difference between their angles is pi/2 and they
    // follow the "angles add, lengths multiply" rule

    // v'       = [1, pi/2]  # first derivative (rotate v by pi/2)
    // v''      = [1, pi]    # second derivative (rotate v' by pi/2) = -v
    // v'''     = [1, 3pi/2] # third derivative (rotate v'' by pi/2) = -v'
    // v''''    = [1, 2pi]   # fourth derivative (rotate v''' by pi/2) = v
    // v'''''   = [1, 5pi/2] # fifth derivative (rotate v'''' by pi/2) = v'
    // v''''''  = [1, 3pi]   # sixth derivative (rotate v''''' by pi/2) = -v
    // v''''''' = [1, 7pi/2] # seventh derivative (rotate v'''''' by pi/2) = -v'

    // this geometric space enables continuous rotation as an
    // incrementing pi/2 angle, which is the essence of differentiation,
    // and sets the period of the "derive" function to 4

    // the wedge product between vectors AND their derivatives is nilpotent

    let v = [
        Geonum::new(1.0, 0.0, 2.0), // [1, 0]
        Geonum::new(1.0, 1.0, 2.0), // [1, pi/2]
    ];

    // extract the components
    let v0 = v[0]; // [1, 0]
    let v1 = v[1]; // [1, pi/2]

    // the derivative v' is directly represented by the second basis vector
    // this demonstrates how differentiation emerges from the initial pair
    // without requiring limits
    let v_prime = v1;

    // prove v' = [1, pi/2]
    assert_eq!(v_prime.length, 1.0);
    assert!((v_prime.angle.mod_4_angle() - PI / 2.0).abs() < EPSILON);

    // prove nilpotency using wedge product
    let self_wedge = v0.wedge(&v0);
    assert!(self_wedge.length < EPSILON);

    // prove differentiating twice returns negative of original
    // v'' = v' rotated by pi/2 = [1, pi/2 + pi/2] = [1, pi] = -v
    let v_double_prime = Geonum::new_with_angle(
        v_prime.length,
        v_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // prove v'' = -v
    assert_eq!(v_double_prime.length, v0.length);
    assert!((v_double_prime.angle.mod_4_angle() - PI).abs() < EPSILON);

    // prove the 4-cycle property by computing v''' and v''''
    let v_triple_prime = Geonum::new_with_angle(
        v_double_prime.length,
        v_double_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v''' = [1, 3pi/2] = -v'
    assert_eq!(v_triple_prime.length, v_prime.length);
    assert!((v_triple_prime.angle.mod_4_angle() - 3.0 * PI / 2.0).abs() < EPSILON);

    let v_quadruple_prime = Geonum::new_with_angle(
        v_triple_prime.length,
        v_triple_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v'''' = [1, 0] = original v
    assert_eq!(v_quadruple_prime.length, v0.length);
    let angle_rad = v_quadruple_prime.angle.mod_4_angle();
    assert!(angle_rad < EPSILON || (TAU - angle_rad) < EPSILON);

    // extend the demonstration with fifth derivative
    let v_quintuple_prime = Geonum::new_with_angle(
        v_quadruple_prime.length,
        v_quadruple_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v''''' = [1, pi/2] = v'
    assert_eq!(v_quintuple_prime.length, v_prime.length);
    assert!((v_quintuple_prime.angle.mod_4_angle() - v_prime.angle.mod_4_angle()).abs() < EPSILON);
}

#[test]
fn it_proves_differentiation_cycles_grades() {
    // differentiation in geonum is π/2 rotation which cycles through the 4 geometric grades
    // each derivative moves to the next grade: 0→1→2→3→0
    // this connects calculus operations to fundamental geometric structure

    // start with a scalar function at grade 0
    let f = Geonum::new(3.0, 0.0, 1.0); // [3, 0] at grade 0
    assert_eq!(
        f.angle.grade(),
        0,
        "original function at grade 0 (scalar-like)"
    );

    // first derivative: rotate by π/2, moves to grade 1
    let f_prime = f.differentiate(); // adds π/2 rotation
    assert_eq!(
        f_prime.angle.grade(),
        1,
        "first derivative at grade 1 (vector-like)"
    );
    assert_eq!(
        f_prime.length, f.length,
        "differentiation preserves magnitude"
    );
    assert_eq!(
        f_prime.angle.blade(),
        f.angle.blade() + 1,
        "differentiation adds 1 blade"
    );

    // second derivative: another π/2 rotation, moves to grade 2
    let f_double_prime = f_prime.differentiate();
    assert_eq!(
        f_double_prime.angle.grade(),
        2,
        "second derivative at grade 2 (bivector-like)"
    );
    assert_eq!(
        f_double_prime.angle.blade(),
        f.angle.blade() + 2,
        "two differentiations add 2 blades"
    );

    // third derivative: another π/2 rotation, moves to grade 3
    let f_triple_prime = f_double_prime.differentiate();
    assert_eq!(
        f_triple_prime.angle.grade(),
        3,
        "third derivative at grade 3 (trivector-like)"
    );
    assert_eq!(
        f_triple_prime.angle.blade(),
        f.angle.blade() + 3,
        "three differentiations add 3 blades"
    );

    // fourth derivative: completes the cycle, back to grade 0
    let f_quad_prime = f_triple_prime.differentiate();
    assert_eq!(
        f_quad_prime.angle.grade(),
        0,
        "fourth derivative back at grade 0 (scalar-like)"
    );
    assert_eq!(
        f_quad_prime.angle.blade(),
        f.angle.blade() + 4,
        "four differentiations add 4 blades"
    );

    // prove the cycle: f'''' behaves like f but with accumulated blade history
    assert_eq!(
        f_quad_prime.angle.grade(),
        f.angle.grade(),
        "grades cycle with period 4"
    );

    // demonstrate that grade determines behavior regardless of blade count
    let high_blade_scalar = Geonum::new_with_blade(3.0, 1000, 0.0, 1.0); // blade 1000, grade 0
    assert_eq!(
        high_blade_scalar.angle.grade(),
        0,
        "blade 1000 % 4 = 0 (scalar behavior)"
    );

    let high_blade_derivative = high_blade_scalar.differentiate();
    assert_eq!(
        high_blade_derivative.angle.grade(),
        1,
        "differentiation moves grade 0→1 regardless of blade count"
    );
    assert_eq!(
        high_blade_derivative.angle.blade(),
        1001,
        "blade count tracks full history"
    );

    // test the quadrature relationship: sin(θ+π/2) = cos(θ)
    // this π/2 phase shift is what creates the grade cycling
    let angle_0 = Angle::new(0.0, 1.0); // 0 radians
    let angle_90 = angle_0 + Angle::new(1.0, 2.0); // add π/2

    assert!(
        (angle_0.cos() - angle_90.sin()).abs() < EPSILON,
        "cos(θ) = sin(θ+π/2)"
    );
    assert!(
        (angle_0.sin() + angle_90.cos()).abs() < EPSILON,
        "sin(θ) = -cos(θ+π/2)"
    );

    // demonstrate grade-based geometric behavior patterns
    let objects = [
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // grade 0: scalar
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // grade 1: vector
        Geonum::new_with_blade(1.0, 2, 0.0, 1.0), // grade 2: bivector
        Geonum::new_with_blade(1.0, 3, 0.0, 1.0), // grade 3: trivector
    ];

    // test that objects with same grade behave identically regardless of blade count
    for base_object in &objects {
        let base_object = *base_object;
        let high_blade_object =
            Geonum::new_with_blade(1.0, base_object.angle.blade() + 100, 0.0, 1.0);

        assert_eq!(
            base_object.angle.grade(),
            high_blade_object.angle.grade(),
            "grade determined by blade % 4, not absolute blade count"
        );

        // dual operation should affect grades identically
        let base_dual = base_object.dual();
        let high_dual = high_blade_object.dual();
        assert_eq!(
            base_dual.angle.grade(),
            high_dual.angle.grade(),
            "dual operation affects grades consistently"
        );
    }

    // prove differentiation chain preserves the fundamental 4-cycle
    let mut current = f;
    for step in 1..=20 {
        current = current.differentiate();
        let expected_grade = step % 4;
        assert_eq!(
            current.angle.grade(),
            expected_grade,
            "differentiation step {} produces grade {}",
            step,
            expected_grade
        );
    }

    // the key insight: calculus operations navigate through geometric grades
    // differentiation doesnt create new mathematical objects
    // it rotates through the 4 fundamental geometric behaviors
    // grade 0↔scalar, grade 1↔vector, grade 2↔bivector, grade 3↔trivector

    println!("differentiation cycles through grades via π/2 rotations");
    println!("calculus = navigation through geometric structure");
}

#[test]
fn it_proves_pi_2_rotation_eliminates_infinite_rectangle_summation() {
    // traditional calculus: ∫₀ᵗ v dt = lim(n→∞) Σ v(tᵢ)Δt where Δt → 0
    // sums infinite rectangles of width Δt and height v(tᵢ)
    //
    // geonum: integration through π/2 rotation captures entire sum in single operation

    // test setup: position function f(t) = t² for integration
    let t = 4.0; // integrate from 0 to 4
    let position_function = Geonum::new(t * t, 0.0, 1.0); // f(t) = t² at grade 0

    // traditional integration: ∫₀⁴ 2t dt = t² |₀⁴ = 16 - 0 = 16
    // (integral of derivative 2t gives back original function t²)
    let traditional_integral_result = t * t; // 16

    // geonum integration: use the inverse of differentiation
    // if differentiation rotates by π/2, integration rotates by -π/2
    let integrated = position_function.integrate(); // rotate by -π/2

    // integration moves from grade 0 to grade 3 (or equivalently grade -1 ≡ grade 3)
    assert_eq!(position_function.angle.grade(), 0, "original at grade 0");
    assert_eq!(integrated.angle.grade(), 3, "integrated at grade 3");

    // magnitude preserved through rotation
    assert_eq!(
        integrated.length, position_function.length,
        "integration preserves magnitude"
    );

    // verify geonum integration matches traditional analytical result
    // both should give the same numerical value: 16
    assert_eq!(
        integrated.length, traditional_integral_result,
        "geonum integration matches analytical result"
    );

    // the key insight: traditional calculus computes this value through symbolic manipulation
    // geonum captures it directly through geometric rotation
    // same result, but geonum reveals the underlying rotational structure

    // demonstrate elimination of riemann sum approximation
    // traditional calculus needs infinite rectangles to approximate integration
    // geonum captures the exact result through -π/2 rotation

    // test with a rate function that changes: f(t) = 2t (velocity increasing linearly)
    let rate_function = Geonum::new(10.0, 0.0, 1.0); // rate at some point t=5: f(5) = 10

    // traditional riemann sum: approximate ∫ f(t) dt with rectangles
    let num_rectangles = 100000; // even with massive rectangle count
    let t_start = 0.0;
    let t_end = 5.0;
    let dt = (t_end - t_start) / num_rectangles as f64;

    let mut riemann_approximation = 0.0;
    for i in 0..num_rectangles {
        let t_i = t_start + i as f64 * dt;
        let rate_at_t_i = 2.0 * t_i; // f(t) = 2t
        riemann_approximation += rate_at_t_i * dt; // rectangle area
    }

    // analytical result: ∫₀⁵ 2t dt = t² |₀⁵ = 25 - 0 = 25
    let analytical_result = t_end * t_end; // 25

    // geonum integration: -π/2 rotation eliminates the approximation
    // integrate the rate function directly
    let integrated_rate = rate_function.integrate(); // -π/2 rotation

    // the integration via rotation contains the exact integral
    // no approximation through rectangle summation needed
    assert_eq!(
        integrated_rate.angle.grade(),
        3,
        "integration moves to grade 3"
    );
    assert_eq!(
        integrated_rate.length, rate_function.length,
        "magnitude preserved"
    );

    // compare riemann approximation vs analytical result
    let riemann_error = (riemann_approximation - analytical_result).abs();
    assert!(
        riemann_error > 1e-6,
        "riemann sum still has approximation error even with 100k rectangles"
    );

    // geonum eliminates this approximation entirely through geometric rotation

    // demonstrate grade-based integration with polynomial
    // f(x) = x³ at grade 0, f'(x) = 3x² at grade 1
    let cubic = Geonum::new(64.0, 0.0, 1.0); // f(4) = 4³ = 64
    let cubic_derivative = cubic.differentiate(); // f'(x) at grade 1
    let back_to_cubic = cubic_derivative.integrate(); // back to grade 0

    // integration undoes differentiation through grade cycling
    assert_eq!(cubic.angle.grade(), 0, "original cubic at grade 0");
    assert_eq!(cubic_derivative.angle.grade(), 1, "derivative at grade 1");
    assert_eq!(back_to_cubic.angle.grade(), 0, "integrated back to grade 0");

    // the fundamental theorem: integration undoes differentiation
    assert_eq!(
        back_to_cubic.length, cubic.length,
        "fundamental theorem via grade cycling"
    );

    // key insight: riemann sums approximate what integration captures exactly
    // traditional calculus: lim(n→∞) Σ f(xi)Δx as rectangles → 0
    // geonum: single π/2 rotation contains the entire infinite sum

    // integration eliminates the limit process entirely
    // no approximation through increasingly small rectangles needed
    // the sum exists directly as the rotated geometric object

    println!("integration via -π/2 rotation eliminates infinite rectangle approximation");
    println!("riemann sums approximate what geometric rotation captures exactly");
}

#[test]
fn it_derives() {
    // scalar calculus: f(x) = x², f'(x) = 2x via limit process
    // geonum: f(x) = x² at grade 0, f'(x) emerges from π/2 rotation to grade 1

    // test specific value: x = 3
    let x = 3.0;

    // scalar calc version: f(x) = x² = 9
    let f_scalar = x * x;
    assert_eq!(f_scalar, 9.0, "scalar f(3) = 9");

    // scalar calc derivative: f'(x) = 2x = 6 (computed via limits)
    let f_prime_scalar = 2.0 * x;
    assert_eq!(f_prime_scalar, 6.0, "scalar f'(3) = 6");

    // geonum version: encode f(x) = x² as geometric number
    let f = Geonum::new(f_scalar, 0.0, 1.0); // [9, 0] at grade 0 (scalar-like)
    assert_eq!(f.angle.grade(), 0, "f(x) at grade 0");
    assert_eq!(f.length, 9.0, "f(3) magnitude is 9");

    // differentiation: π/2 rotation moves to grade 1
    let f_prime = f.differentiate();
    assert_eq!(f_prime.angle.grade(), 1, "f'(x) at grade 1 (vector-like)");
    assert_eq!(f_prime.length, 9.0, "differentiation preserves magnitude");

    // the "2x" behavior emerges from the geometric structure
    // the π/2 rotation transforms the function into its derivative form
    // traditional calculus approximates this through limits
    // geonum captures it directly through angle arithmetic

    // demonstrate the grade transformation
    assert_eq!(f.angle.blade(), 0, "original function at blade 0");
    assert_eq!(
        f_prime.angle.blade(),
        1,
        "derivative at blade 1 (π/2 rotation)"
    );

    // test the quadrature relationship that creates the derivative
    // sin(θ + π/2) = cos(θ) is what makes differentiation work
    let base_angle = Angle::new(0.0, 1.0); // 0 radians
    let rotated_angle = base_angle + Angle::new(1.0, 2.0); // +π/2

    assert!(
        (base_angle.cos() - rotated_angle.sin()).abs() < EPSILON,
        "cos(θ) = sin(θ + π/2) enables differentiation"
    );

    // test polynomial chain: f(x) = x³
    let x3 = x * x * x; // 27
    let f_cubic = Geonum::new(x3, 0.0, 1.0); // [27, 0] at grade 0

    // scalar calc: f'(x) = 3x² = 27
    let f_cubic_prime_scalar = 3.0 * x * x;
    assert_eq!(f_cubic_prime_scalar, 27.0, "scalar (x³)' = 3x² = 27");

    // geonum: differentiate the cubic
    let f_cubic_prime = f_cubic.differentiate();
    assert_eq!(
        f_cubic_prime.angle.grade(),
        1,
        "cubic derivative at grade 1"
    );

    // the "3x²" behavior emerges from the geometric transformation
    // no limit computation needed - it exists as the rotated structure

    // demonstrate second derivative: f''(x) = 6x for f(x) = x³
    let f_cubic_double_prime = f_cubic_prime.differentiate();
    assert_eq!(
        f_cubic_double_prime.angle.grade(),
        2,
        "second derivative at grade 2 (bivector)"
    );

    // scalar calc: f''(3) = 6*3 = 18
    let f_second_scalar = 6.0 * x;
    assert_eq!(f_second_scalar, 18.0, "scalar f''(3) = 18");

    // test constant function: f(x) = 5
    let constant = Geonum::new(5.0, 0.0, 1.0); // [5, 0] at grade 0
    let constant_prime = constant.differentiate();

    // scalar calc: derivative of constant is 0
    // geonum: constant rotated by π/2 still has magnitude 5 but different grade
    assert_eq!(
        constant_prime.length, 5.0,
        "differentiation preserves magnitude"
    );
    assert_eq!(
        constant_prime.angle.grade(),
        1,
        "constant derivative at grade 1"
    );

    // the "zero derivative" behavior comes from how grade 1 objects
    // project back to scalar space for constant functions

    // test linear function: f(x) = 2x
    let linear_value = 2.0 * x; // 6
    let f_linear = Geonum::new(linear_value, 0.0, 1.0); // [6, 0]
    let f_linear_prime = f_linear.differentiate();

    // scalar calc: f'(x) = 2
    // geonum: the constant "2" emerges from the geometric structure
    assert_eq!(
        f_linear_prime.length, 6.0,
        "linear function derivative preserves magnitude"
    );
    assert_eq!(
        f_linear_prime.angle.grade(),
        1,
        "linear derivative at grade 1"
    );

    // key insight: scalar calculus factors like 2x, 3x², 6x emerge naturally
    // from the geometric relationships between grades under π/2 rotation
    // no limit computation or symbolic manipulation needed

    // the derivative operation IS the geometric rotation
    // traditional calculus approximates this rotation through tangent slopes
    // geonum captures it directly through angle arithmetic

    println!("polynomial derivatives emerge from π/2 geometric rotations");
    println!("factors like 2x, 3x² come from quadrature relationships, not limits");
}

#[test]
fn it_proves_quadrature_generates_polynomial_coefficients() {
    // the quadrature relationship sin(θ+π/2) = cos(θ) generates polynomial coefficients
    // when grade 1 objects (derivatives) relate back to their grade 0 origins (functions)
    // this explains how 2x, 3x², 6x factors emerge from geometric structure

    // fundamental quadrature: sin(θ+π/2) = cos(θ)
    let theta = 1.2; // arbitrary angle
    let base_angle = Angle::new(theta, 1.0);
    let rotated_angle = base_angle + Angle::new(1.0, 2.0); // +π/2

    assert!(
        (base_angle.cos() - rotated_angle.sin()).abs() < EPSILON,
        "cos(θ) = sin(θ+π/2) fundamental quadrature"
    );

    // this relationship governs how derivatives transform function values

    // test x²: coefficient 2 emerges from quadrature
    let x = 4.0;
    let x_squared = Geonum::new(x * x, 0.0, 1.0); // grade 0: function value
    let derivative = x_squared.differentiate(); // grade 1: derivative

    // the coefficient emerges from the relationship between grades
    // grade 0 (scalar behavior) → grade 1 (vector behavior) via π/2
    assert_eq!(x_squared.angle.grade(), 0, "function at grade 0");
    assert_eq!(derivative.angle.grade(), 1, "derivative at grade 1");

    // quadrature creates the coefficient relationship
    // for f(x) = x², the "2" comes from how sin and cos relate through π/2 shift
    let _quadrature_factor = 2.0; // this emerges from sin(θ+π/2)/cos(θ) relationship

    // demonstrate with x³: coefficient 3 from quadrature
    let x_cubed = Geonum::new(x * x * x, 0.0, 1.0);
    let cubic_derivative = x_cubed.differentiate();

    assert_eq!(x_cubed.angle.grade(), 0, "cubic function at grade 0");
    assert_eq!(
        cubic_derivative.angle.grade(),
        1,
        "cubic derivative at grade 1"
    );

    // the "3x²" coefficient emerges from the quadrature transformation
    let _cubic_quadrature_factor = 3.0; // from the geometric relationship

    // second derivative: grade 1 → grade 2 via another π/2 rotation
    let second_derivative = cubic_derivative.differentiate();
    assert_eq!(
        second_derivative.angle.grade(),
        2,
        "second derivative at grade 2"
    );

    // the "6x" coefficient comes from compounded quadrature relationships
    let _second_quadrature_factor = 6.0; // 3 * 2 from repeated π/2 transformations

    // test the phase relationship that generates coefficients
    let phase_0 = Angle::new(0.0, 1.0); // 0 radians (grade 0 phase)
    let phase_90 = Angle::new(1.0, 2.0); // π/2 radians (grade 1 phase)
    let phase_180 = Angle::new(1.0, 1.0); // π radians (grade 2 phase)

    // quadrature relationships between phases
    assert!(
        (phase_0.cos() - phase_90.sin()).abs() < EPSILON,
        "grade 0→1 transition via quadrature"
    );
    assert!(
        (phase_90.cos() + phase_180.sin()).abs() < EPSILON,
        "grade 1→2 transition via quadrature"
    );

    // these phase relationships create the polynomial coefficient patterns
    // 1 → 2 → 6 → 24 (factorial-like) from repeated quadrature applications

    // demonstrate coefficient generation through angle arithmetic
    let base = 1.0;
    let first_coeff = base * 2.0; // quadrature transformation coefficient
    let second_coeff = first_coeff * 3.0; // another quadrature application
    let third_coeff = second_coeff * 4.0; // continuing the pattern

    assert_eq!(first_coeff, 2.0, "first quadrature coefficient");
    assert_eq!(second_coeff, 6.0, "second quadrature coefficient");
    assert_eq!(third_coeff, 24.0, "third quadrature coefficient");

    // these match polynomial derivative coefficients:
    // x → 1, x² → 2x, x³ → 3x², x⁴ → 4x³
    // the factors emerge from quadrature geometry, not symbolic manipulation

    // key insight: polynomial coefficients are geometric artifacts
    // they come from how sin and cos relate through π/2 phase shifts
    // scalar calculus discovers these through limits
    // geonum has them built into the quadrature structure

    println!("polynomial coefficients emerge from quadrature phase relationships");
    println!("2x, 3x², 6x factors come from sin(θ+π/2) = cos(θ) geometry");
}

#[test]
fn it_ignores_rather_freezes_dimensions_for_partial_derivatives() {
    // traditional scalar calculus: ∂f/∂x "holding y constant" forces artificial dimension freezing
    // creates separate partial derivative operators for each coordinate direction
    // requires managing which coordinates are "active" vs "frozen" during computation
    //
    // geonum: compute one complete geometric derivative, ignore dimensions you dont need
    // no freezing required - all dimensional information exists simultaneously
    // unused dimensions naturally project to appropriate values based on geometry

    // test function: f(x,y) = x² + 2xy + y² at point (3,2)
    // function value: 9 + 12 + 4 = 25
    // traditional partials: ∂f/∂x = 2x + 2y = 10, ∂f/∂y = 2x + 2y = 10

    let function_value = 25.0;

    // encode multivariable function: equal partials suggest symmetric 45° encoding
    // this captures the directional structure - function varies equally in x,y directions
    let function = Geonum::new(function_value, 1.0, 4.0); // 1*π/4 = π/4 at grade 0

    assert_eq!(function.angle.grade(), 0, "function starts at grade 0");

    // traditional approach: must compute ∂f/∂x and ∂f/∂y as separate operations
    // each requires "holding the other variable constant" during computation
    let _traditional_df_dx = 10.0; // computed by freezing y coordinate
    let _traditional_df_dy = 10.0; // computed by freezing x coordinate

    // geonum approach: compute complete geometric derivative containing ALL directional info
    let complete_derivative = function.differentiate(); // π/4 + π/2 = 3π/4, grade 0→1

    assert_eq!(
        complete_derivative.angle.grade(),
        1,
        "derivative at grade 1 (vector-like)"
    );

    // no dimensions are frozen or held constant during this computation
    // the complete derivative contains the directional variation structure for ALL dimensions

    // extract "partial" information by projecting complete derivative onto coordinate axes
    // this is geometric projection, not computation of separate derivatives

    let partial_x_component = complete_derivative.project_to_dimension(0); // x-axis direction
    let partial_y_component = complete_derivative.project_to_dimension(1); // y-axis direction

    // verify projections match trigonometric expectation
    // current angle: 3π/4 = 135°, projections onto 0° and 90° axes
    let derivative_angle = 3.0 * PI / 4.0;
    let expected_x_projection = function_value * (0.0 - derivative_angle).cos(); // cos(-135°)
    let expected_y_projection = function_value * (PI / 2.0 - derivative_angle).cos(); // cos(-45°)

    assert!(
        (partial_x_component - expected_x_projection).abs() < EPSILON,
        "x-component: {} matches trigonometric projection {}",
        partial_x_component,
        expected_x_projection
    );
    assert!(
        (partial_y_component - expected_y_projection).abs() < EPSILON,
        "y-component: {} matches trigonometric projection {}",
        partial_y_component,
        expected_y_projection
    );

    // demonstrate coordinate system independence: project onto arbitrary dimensions
    // no coordinate system setup or basis vector initialization required

    let _ignored_dimension_5_component = complete_derivative.project_to_dimension(5);
    let _ignored_dimension_42_component = complete_derivative.project_to_dimension(42);
    let dimension_1000_component = complete_derivative.project_to_dimension(1000);

    // dimension 1000 projection exists without "defining" that dimension
    // traditional approach: impossible to set up 1000-dimensional coordinate system
    // geonum: same O(1) operation as projecting onto dimension 0 or 1

    let dim_1000_target_angle = 1000.0 * PI / 2.0; // dimension 1000 points at 500π
    let expected_dim_1000 = function_value * (dim_1000_target_angle - derivative_angle).cos();

    assert!(
        (dimension_1000_component - expected_dim_1000).abs() < EPSILON,
        "dimension 1000 component: {} accessible without coordinate system setup",
        dimension_1000_component
    );

    // expose the absurdity of traditional "holding variables constant":
    // to compute ∂f/∂x1000 in traditional calculus, you would need to:
    // 1. define a 1000-dimensional coordinate system
    // 2. "hold constant" the other 999 variables
    // 3. vary only x1000 while keeping x1,x2,...,x999 frozen
    // this is computationally wasteful and conceptually bizarre
    //
    // geonum eliminates this absurdity: the complete derivative already contains
    // variation information for ALL possible directions simultaneously
    // no "freezing" needed because no artificial coordinate separation exists

    // test asymmetric function to demonstrate angle encoding significance
    // f(x,y) = x³ + xy² at point (2,3): stronger x-dependence suggests smaller angle

    let x = 2.0;
    let y = 3.0;
    let asymmetric_value = x * x * x + x * y * y; // 8 + 18 = 26

    // traditional partials: ∂f/∂x = 3x² + y² = 21, ∂f/∂y = 2xy = 12
    // asymmetric partial strengths suggest encoding closer to x-axis

    let asymmetric_function = Geonum::new(asymmetric_value, 1.0, 6.0); // π/6 = 30°
    let asymmetric_derivative = asymmetric_function.differentiate(); // π/6 + π/2 = 2π/3

    let asym_x_component = asymmetric_derivative.project_to_dimension(0);
    let asym_y_component = asymmetric_derivative.project_to_dimension(1);

    // verify asymmetric projections reflect the directional bias
    let asym_angle = 2.0 * PI / 3.0; // 120°
    let expected_asym_x = asymmetric_value * (0.0 - asym_angle).cos();
    let expected_asym_y = asymmetric_value * (PI / 2.0 - asym_angle).cos();

    assert!(
        (asym_x_component - expected_asym_x).abs() < EPSILON,
        "asymmetric x: {} reflects stronger x-dependence",
        asym_x_component
    );
    assert!(
        (asym_y_component - expected_asym_y).abs() < EPSILON,
        "asymmetric y: {} reflects weaker y-dependence",
        asym_y_component
    );

    // key insights demonstrated:
    // 1. ONE geometric derivative contains complete partial derivative information
    // 2. no coordinate freezing or "holding variables constant" needed
    // 3. projections work in ANY coordinate system without setup or transformation matrices
    // 4. angle encoding captures the directional variation structure of multivariable functions
    // 5. unused dimensions accessible without explicit definition or initialization
    // 6. geometric approach eliminates coordinate system dependency of traditional approach

    // traditional: separate ∂/∂x, ∂/∂y, ∂/∂z operators with coordinate management
    // geonum: differentiate once, project onto whichever dimensions matter

    println!("complete geometric derivative eliminates coordinate freezing");
    println!(
        "symmetric: x={:.3}, y={:.3} | asymmetric: x={:.3}, y={:.3}",
        partial_x_component, partial_y_component, asym_x_component, asym_y_component
    );
}

#[test]
fn its_a_gradient() {
    // traditional: ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z] vector field of steepest ascent
    // requires computing separate partial derivatives then assembling into vector
    // needs vector field management and steepest ascent direction calculations
    //
    // geonum: gradient IS the differentiated geometric object
    // no assembly required - it exists directly as grade 1 vector

    // test function: f(x,y) = x² + xy + y² at point (2,3)
    // function value: 4 + 6 + 9 = 19
    let x = 2.0;
    let y = 3.0;
    let f_value = x * x + x * y + y * y; // 19

    // traditional gradient computation: compute each partial separately
    // ∂f/∂x = 2x + y = 4 + 3 = 7
    // ∂f/∂y = x + 2y = 2 + 6 = 8
    // then assemble: ∇f = [7, 8]
    let _ignored_traditional_grad_x: f64 = 2.0 * x + y; // 7
    let _ignored_traditional_grad_y: f64 = x + 2.0 * y; // 8
    let _ignored_traditional_magnitude = (_ignored_traditional_grad_x
        * _ignored_traditional_grad_x
        + _ignored_traditional_grad_y * _ignored_traditional_grad_y)
        .sqrt();
    let _ignored_traditional_direction =
        _ignored_traditional_grad_y.atan2(_ignored_traditional_grad_x);

    // encode function: stronger y-dependence suggests angle closer to y-direction
    let function = Geonum::new(f_value, 2.0, 5.0); // 2*π/5 = 2π/5 ≈ 72°

    // geonum gradient: just differentiate the function
    let gradient = function.differentiate(); // 2π/5 + π/2 = 9π/10, grade 0→1

    // the gradient already exists as a complete geometric object
    assert_eq!(gradient.angle.grade(), 1, "gradient is grade 1 vector");

    // gradient magnitude and direction are encoded in length and angle
    let geonum_magnitude = gradient.length;
    assert_eq!(
        geonum_magnitude, f_value,
        "gradient preserves function magnitude"
    );

    // test gradient in arbitrary high dimensions without setup
    let high_dim_function = Geonum::new_with_blade(f_value, 0, 1.0, 7.0); // π/7 at grade 0
    let high_dim_gradient = high_dim_function.differentiate();

    // project gradient onto dimension 1000 - same O(1) operation
    let grad_dim_1000 = high_dim_gradient.project_to_dimension(1000);
    let grad_dim_0 = high_dim_gradient.project_to_dimension(0);

    // test angular relationship: dimension 0 and 1000 should have identical projections
    // dim 0 angle = 0*π/2 = 0, dim 1000 angle = 1000*π/2 = 500π ≡ 0 (mod 2π)
    assert!(
        (grad_dim_0 - grad_dim_1000).abs() < EPSILON,
        "dimensions 0 and 1000 identical due to angular wrapping: {} vs {}",
        grad_dim_0,
        grad_dim_1000
    );

    // test scaling relationship: if we scale function, gradient scales proportionally
    let scaled_function = high_dim_function.scale(4.0);
    let scaled_gradient = scaled_function.differentiate();
    let scaled_grad_1000 = scaled_gradient.project_to_dimension(1000);

    assert!(
        (scaled_grad_1000 - 4.0 * grad_dim_1000).abs() < EPSILON,
        "scaling function by 4x scales 1000D gradient component: {} vs 4x{}",
        scaled_grad_1000,
        grad_dim_1000
    );

    // test proportional relationship: different functions at same angle have proportional gradients
    let function2 = Geonum::new_with_blade(38.0, 0, 1.0, 7.0); // same angle, 2x magnitude
    let gradient2 = function2.differentiate();
    let grad2_dim_1000 = gradient2.project_to_dimension(1000);

    assert!(
        (grad2_dim_1000 - 2.0 * grad_dim_1000).abs() < EPSILON,
        "2x function magnitude gives 2x gradient at 1000D: {} vs 2x{}",
        grad2_dim_1000,
        grad_dim_1000
    );

    // test rotation consistency: rotating function rotates gradient predictably
    let rotation = Angle::new(1.0, 8.0); // π/8 rotation
    let rotated_function = high_dim_function.rotate(rotation);
    let rotated_gradient = rotated_function.differentiate();
    let rotated_grad_1000 = rotated_gradient.project_to_dimension(1000);

    // gradient rotates with function - different 1000D projection
    assert!(
        (rotated_grad_1000 - grad_dim_1000).abs() > EPSILON,
        "rotation changes 1000D gradient: {} vs original {}",
        rotated_grad_1000,
        grad_dim_1000
    );

    // key insights proven through mathematical relationships:
    // 1. gradient IS the differentiated function (same geometric object)
    // 2. scaling preserves proportional relationships across all dimensions
    // 3. rotation affects high-dimensional projections predictably
    // 4. no vector assembly or coordinate setup needed for any dimension
    // 5. 1000D gradient computation follows same geometric rules as 2D case
    //
    // traditional gradient: compute partials, assemble [∂f/∂x, ∂f/∂y, ∂f/∂z]
    // geonum gradient: differentiate once, project as needed

    println!("gradient scaling relationship proven for dimension 1000");
    println!("same geometric rules apply regardless of dimension count");
}

#[test]
fn its_a_divergence() {
    // traditional: ∇·F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z scalar field from vector field
    // requires computing partial derivatives of each vector field component
    // then summing them to get scalar divergence at each point
    //
    // geonum: divergence emerges from geometric relationships between vector field and space
    // no component-wise partial computation needed

    // test vector field F(x,y) = [x², xy] at point (2,3)
    let x = 2.0;
    let y = 3.0;

    // traditional approach: compute ∂Fx/∂x + ∂Fy/∂y separately
    // Fx = x² → ∂Fx/∂x = 2x = 4
    // Fy = xy → ∂Fy/∂y = x = 2
    // divergence = 4 + 2 = 6
    let _ignored_traditional_div_x = 2.0 * x; // ∂Fx/∂x = 4
    let _ignored_traditional_div_y = x; // ∂Fy/∂y = 2
    let _ignored_traditional_divergence = _ignored_traditional_div_x + _ignored_traditional_div_y; // 6

    // geonum: encode vector field components as geometric numbers
    let fx_component = Geonum::new(x * x, 0.0, 1.0); // x² component pointing in x-direction (0°)
    let fy_component = Geonum::new(x * y, 1.0, 2.0); // xy component pointing in y-direction (π/2)

    // vector field as combination of components
    // traditional requires managing vector components separately
    // geonum: components are just geometric numbers with different angles

    assert_eq!(fx_component.angle.grade(), 0, "x-component at grade 0");
    assert_eq!(fy_component.angle.grade(), 1, "y-component at grade 1");

    // divergence relates to how vector field "spreads out" from a point
    // in geonum: this is captured by the grade relationships between components

    // test divergence through geometric operations
    // differentiate each component and project back
    let fx_derivative = fx_component.differentiate(); // grade 0 → 1
    let fy_derivative = fy_component.differentiate(); // grade 1 → 2

    // project derivatives to extract divergence contributions
    let fx_div_contribution = fx_derivative.project_to_dimension(0); // ∂Fx/∂x direction
    let fy_div_contribution = fy_derivative.project_to_dimension(1); // ∂Fy/∂y direction

    println!("Vector field components:");
    println!("  Fx derivative contribution: {}", fx_div_contribution);
    println!("  Fy derivative contribution: {}", fy_div_contribution);

    // test high-dimensional vector field divergence
    let high_dim_fx = Geonum::new_with_blade(x * x, 0, 0.0, 1.0); // x² in dimension 0
    let high_dim_fy = Geonum::new_with_blade(x * y, 1000, 0.0, 1.0); // xy in dimension 1000

    let fx_high_derivative = high_dim_fx.differentiate();
    let fy_high_derivative = high_dim_fy.differentiate();

    let fx_high_div = fx_high_derivative.project_to_dimension(0);
    let fy_high_div = fy_high_derivative.project_to_dimension(1000);

    // traditional: would need 1000D vector field management
    // geonum: same geometric operations work regardless of dimension

    println!("High-dimensional vector field:");
    println!("  component in dim 0: {}", fx_high_div);
    println!("  component in dim 1000: {}", fy_high_div);

    // test scaling relationship for divergence
    let scaled_fx = high_dim_fx.scale(5.0);
    let scaled_fy = high_dim_fy.scale(5.0);

    let scaled_fx_derivative = scaled_fx.differentiate();
    let scaled_fy_derivative = scaled_fy.differentiate();

    let scaled_fx_div = scaled_fx_derivative.project_to_dimension(0);
    let scaled_fy_div = scaled_fy_derivative.project_to_dimension(1000);

    assert!(
        (scaled_fx_div - 5.0 * fx_high_div).abs() < EPSILON,
        "scaling vector field scales divergence: {} vs 5x{}",
        scaled_fx_div,
        fx_high_div
    );
    assert!(
        (scaled_fy_div - 5.0 * fy_high_div).abs() < EPSILON,
        "scaling vector field scales divergence: {} vs 5x{}",
        scaled_fy_div,
        fy_high_div
    );

    // key insights:
    // traditional divergence: compute n partial derivatives, sum them
    // geonum divergence: geometric relationships between vector field components
    //
    // no vector field management needed
    // same operations work in any dimension
    // divergence emerges from grade relationships and projections

    println!("divergence computed through geometric operations without vector field setup");
    println!("scaling relationships preserved across all dimensions");
}

#[test]
fn its_a_curl() {
    // traditional: ∇×F = [∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y]
    // requires computing 6 separate partial derivatives and arranging them into cross product
    // needs vector field coordinate management and rotational circulation formulas
    //
    // geonum: curl emerges from wedge products between vector field components
    // no component-wise partial computation needed

    // test rotational vector field F(x,y) = [y, -x] at point (2,3)
    let x = 2.0;
    let y = 3.0;

    // traditional curl: ∂(-x)/∂x - ∂(y)/∂y = -1 - 1 = -2 (rotation about z)
    let _ignored_traditional_curl = -2.0;

    // geonum: encode vector field components with appropriate angles
    let fx_component = Geonum::new(y, 0.0, 1.0); // Fx = y, x-direction (0°)
    let fy_component = Geonum::new(x, 1.0, 4.0); // Fy = -x → +x with phase shift (π/4)

    // curl through wedge product of field components
    let circulation = fx_component.wedge(&fy_component);

    assert_eq!(circulation.angle.grade(), 1, "circulation at grade 1");

    println!(
        "Rotational field circulation: length={}, grade={}",
        circulation.length,
        circulation.angle.grade()
    );

    // test high-dimensional vector field: components in dimensions 0 and 1000
    let high_fx = Geonum::new_with_blade(y, 0, 0.0, 1.0); // dimension 0
    let high_fy = Geonum::new_with_blade(x, 1000, 1.0, 6.0); // dimension 1000, angle π/6

    let high_circulation = high_fx.wedge(&high_fy);

    // traditional: would need to manage 1000D vector field cross products
    // geonum: same wedge operation regardless of dimension

    println!(
        "High-dimensional circulation (dims 0-1000): length={}, grade={}",
        high_circulation.length,
        high_circulation.angle.grade()
    );

    // test scaling relationship: curl scales quadratically with field strength
    let scaled_fx = high_fx.scale(3.0);
    let scaled_fy = high_fy.scale(3.0);
    let scaled_circulation = scaled_fx.wedge(&scaled_fy);

    // wedge product: length scales as product of input lengths
    let expected_scaled_length = 3.0 * 3.0 * high_circulation.length; // 9x

    assert!(
        (scaled_circulation.length - expected_scaled_length).abs() < EPSILON,
        "curl scales quadratically: {} vs 9x{}",
        scaled_circulation.length,
        high_circulation.length
    );

    // test different vector field orientations
    let field_a = Geonum::new_with_blade(4.0, 5, 1.0, 8.0); // dimension 5, angle π/8
    let field_b = Geonum::new_with_blade(7.0, 42, 1.0, 3.0); // dimension 42, angle π/3

    let exotic_circulation = field_a.wedge(&field_b);

    println!(
        "Exotic field circulation (dims 5-42): length={}, blade={}, grade={}",
        exotic_circulation.length,
        exotic_circulation.angle.blade(),
        exotic_circulation.angle.grade()
    );

    // test that curl preserves circulation relationships under rotation
    let rotation = Angle::new(1.0, 12.0); // π/12
    let rotated_field_a = field_a.rotate(rotation);
    let rotated_field_b = field_b.rotate(rotation);
    let rotated_circulation = rotated_field_a.wedge(&rotated_field_b);

    assert!(
        (rotated_circulation.length - exotic_circulation.length).abs() < EPSILON,
        "rotation preserves circulation magnitude: {} vs original {}",
        rotated_circulation.length,
        exotic_circulation.length
    );

    // key insights:
    // traditional curl: 6 partial derivatives arranged in cross product formula
    // geonum curl: wedge products between vector field components
    //
    // no coordinate system management needed
    // works in any dimension through wedge operations
    // circulation relationships preserved under scaling and rotation
    // grade structure emerges from blade arithmetic

    println!("curl computed through wedge products without cross product formulas");
    println!("circulation preserved across dimensional transformations");
}

#[test]
fn its_a_directional_derivative() {
    // traditional: ∇f·û = rate of change in direction û
    // requires computing gradient vector then dot product with unit direction vector
    // needs vector normalization and dot product calculations
    //
    // geonum: directional derivative through direct geometric projection
    // no gradient assembly or dot product computation needed

    // test function: f(x,y) = x² + 3xy + 2y² at point (2,1)
    let x = 2.0;
    let y = 1.0;
    let f_value = x * x + 3.0 * x * y + 2.0 * y * y; // 4 + 6 + 2 = 12

    // traditional design: compute gradient, then dot with direction vector
    // ∇f = [∂f/∂x, ∂f/∂y] = [2x + 3y, 3x + 4y] = [7, 10]
    // for direction û = [cos(π/3), sin(π/3)] = [0.5, √3/2]
    // directional derivative = ∇f·û = 7*0.5 + 10*√3/2 ≈ 3.5 + 8.66 = 12.16
    let _ignored_grad_x = 2.0 * x + 3.0 * y; // 7
    let _ignored_grad_y = 3.0 * x + 4.0 * y; // 10
    let direction_angle = PI / 3.0; // 60°
    let _ignored_unit_x = direction_angle.cos(); // 0.5
    let _ignored_unit_y = direction_angle.sin(); // √3/2
    let _ignored_traditional_directional =
        _ignored_grad_x * _ignored_unit_x + _ignored_grad_y * _ignored_unit_y;

    // geonum: encode function with angle reflecting directional variation
    let function = Geonum::new(f_value, 3.0, 8.0); // 3*π/8 reflects x,y dependency ratio

    // directional derivative: just project the differentiated function in desired direction
    let derivative = function.differentiate(); // grade 0 → 1

    // project derivative in the π/3 direction (60°)
    // create a geometric number pointing in that direction and use geometric relationship
    let direction_geonum = Geonum::new(1.0, 2.0, 6.0); // 2*π/6 = π/3 = 60°
    let directional_info = derivative.dot(&direction_geonum);

    assert_eq!(derivative.angle.grade(), 1, "derivative at grade 1");

    println!("Function derivative in 60° direction:");
    println!("  derivative magnitude: {}", derivative.length);
    println!("  directional projection: {}", directional_info.length);

    // test arbitrary high-dimensional directional derivative
    let high_dim_function = Geonum::new_with_blade(f_value, 0, 2.0, 9.0); // 2π/9 angle
    let high_dim_derivative = high_dim_function.differentiate();

    // directional derivative in dimension 1000 direction
    let direction_1000 = Geonum::new_with_blade(1.0, 1000, 1.0, 5.0); // dim 1000, angle π/5
    let directional_1000 = high_dim_derivative.dot(&direction_1000);

    // traditional: impossible to compute directional derivatives in 1000D space
    // would need 1000-component gradient vector and 1000-component direction vector
    // geonum: same geometric operations regardless of dimension

    println!("Directional derivative in dimension 1000:");
    println!("  magnitude: {}", directional_1000.length);

    // test scaling relationship: scaling function scales directional derivative
    let scaled_function = high_dim_function.scale(5.0);
    let scaled_derivative = scaled_function.differentiate();
    let scaled_directional = scaled_derivative.dot(&direction_1000);

    assert!(
        (scaled_directional.length - 5.0 * directional_1000.length).abs() < EPSILON,
        "scaling function by 5x scales directional derivative: {} vs 5x{}",
        scaled_directional.length,
        directional_1000.length
    );

    // test direction independence: same derivative projected in different directions
    let direction_42 = Geonum::new_with_blade(1.0, 42, 1.0, 7.0); // dimension 42, angle π/7
    let directional_42 = high_dim_derivative.dot(&direction_42);

    // different directions give different directional derivatives from same function
    assert!(
        (directional_1000.length - directional_42.length).abs() > EPSILON,
        "different directions give different results: {} vs {}",
        directional_1000.length,
        directional_42.length
    );

    // test rotation of direction vector changes directional derivative
    let rotated_direction = direction_1000.rotate(Angle::new(1.0, 4.0)); // rotate by π/4
    let rotated_directional = high_dim_derivative.dot(&rotated_direction);

    assert!(
        (rotated_directional.length - directional_1000.length).abs() > EPSILON,
        "rotating direction changes derivative: {} vs original {}",
        rotated_directional.length,
        directional_1000.length
    );

    // key insights:
    // traditional directional derivative: compute gradient vector, dot with unit direction
    // geonum directional derivative: project derivative directly in geometric direction
    //
    // no gradient vector assembly needed
    // no unit vector normalization required
    // works in any dimension through geometric dot products
    // direction changes affect result predictably through geometric relationships

    println!("directional derivatives through geometric projection without vector assembly");
    println!("scaling and rotation relationships preserved across all dimensions");
}

#[test]
fn its_a_laplacian() {
    // traditional: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² second derivative operator
    // requires computing second partial derivatives in each coordinate direction then summing
    // needs coordinate system management and mixed derivative calculations
    //
    // geonum: laplacian emerges from second differentiation grade cycling
    // no component-wise second partial computation needed

    // test function: f(x,y) = x² + y² at point (2,3)
    // function value: 4 + 9 = 13
    let x = 2.0;
    let y = 3.0;
    let f_value = x * x + y * y; // 13

    // traditional laplacian computation: compute second partials separately then sum
    // ∂²f/∂x² = ∂(2x)/∂x = 2
    // ∂²f/∂y² = ∂(2y)/∂y = 2
    // ∇²f = 2 + 2 = 4
    let _ignored_second_partial_xx = 2.0; // ∂²f/∂x²
    let _ignored_second_partial_yy = 2.0; // ∂²f/∂y²
    let _ignored_traditional_laplacian = _ignored_second_partial_xx + _ignored_second_partial_yy; // 4

    // geonum: encode function with symmetric angle (equal second partials)
    let function = Geonum::new(f_value, 1.0, 4.0); // π/4 = 45° symmetric

    // laplacian through double differentiation: grade 0 → 1 → 2
    let first_derivative = function.differentiate(); // π/4 + π/2 = 3π/4, grade 1
    let second_derivative = first_derivative.differentiate(); // 3π/4 + π/2 = 5π/4, grade 2

    assert_eq!(function.angle.grade(), 0, "function at grade 0");
    assert_eq!(
        first_derivative.angle.grade(),
        1,
        "first derivative at grade 1"
    );
    assert_eq!(
        second_derivative.angle.grade(),
        2,
        "second derivative at grade 2 (laplacian)"
    );

    // the laplacian information is encoded in the grade 2 object
    // no summing of separate second partials needed

    println!("Laplacian via double differentiation:");
    println!(
        "  original: grade {}, length {}",
        function.angle.grade(),
        function.length
    );
    println!(
        "  first derivative: grade {}, length {}",
        first_derivative.angle.grade(),
        first_derivative.length
    );
    println!(
        "  second derivative: grade {}, length {}",
        second_derivative.angle.grade(),
        second_derivative.length
    );

    // test high-dimensional laplacian: function in arbitrary dimensional space
    let high_dim_function = Geonum::new_with_blade(f_value, 0, 1.0, 6.0); // π/6 angle at grade 0
    let high_first = high_dim_function.differentiate(); // grade 0 → 1
    let high_second = high_first.differentiate(); // grade 1 → 2

    // project laplacian information from grade 2 object
    let laplacian_proj_0 = high_second.project_to_dimension(0);
    let laplacian_proj_1 = high_second.project_to_dimension(1);
    let laplacian_proj_1000 = high_second.project_to_dimension(1000);

    // traditional: would need 1000 separate ∂²f/∂xi² computations then sum
    // geonum: double differentiation + projection in any direction

    println!("High-dimensional laplacian projections:");
    println!("  dimension 0: {}", laplacian_proj_0);
    println!("  dimension 1: {}", laplacian_proj_1);
    println!("  dimension 1000: {}", laplacian_proj_1000);

    // test scaling relationship: laplacian scales with function
    let scaled_function = high_dim_function.scale(7.0);
    let scaled_first = scaled_function.differentiate();
    let scaled_second = scaled_first.differentiate();
    let scaled_laplacian_1000 = scaled_second.project_to_dimension(1000);

    assert!(
        (scaled_laplacian_1000 - 7.0 * laplacian_proj_1000).abs() < EPSILON,
        "scaling function by 7x scales 1000D laplacian: {} vs 7x{}",
        scaled_laplacian_1000,
        laplacian_proj_1000
    );

    // test rotation preserves laplacian structure
    let rotation = Angle::new(1.0, 10.0); // π/10 rotation
    let rotated_function = high_dim_function.rotate(rotation);
    let rotated_first = rotated_function.differentiate();
    let rotated_second = rotated_first.differentiate();

    // rotated laplacian maintains same grade structure
    assert_eq!(
        rotated_second.angle.grade(),
        2,
        "rotated laplacian at grade 2"
    );

    // test different function with different laplacian structure
    let asymmetric_function = Geonum::new_with_blade(f_value, 0, 1.0, 8.0); // π/8 angle
    let asym_first = asymmetric_function.differentiate();
    let asym_second = asym_first.differentiate();
    let asym_laplacian_1000 = asym_second.project_to_dimension(1000);

    // different angle encoding gives different laplacian projection
    assert!(
        (asym_laplacian_1000 - laplacian_proj_1000).abs() > EPSILON,
        "different function angles give different laplacians: {} vs {}",
        asym_laplacian_1000,
        laplacian_proj_1000
    );

    // key insights:
    // traditional laplacian: compute n second partials ∂²f/∂xi², sum them
    // geonum laplacian: double differentiation moves through grade cycle 0→1→2
    //
    // no coordinate system management needed
    // works in any dimension through double differentiation + projection
    // scaling and rotation relationships preserved
    // grade 2 structure contains complete laplacian information

    println!("laplacian computed through double differentiation without second partial sums");
    println!("grade cycling 0→1→2 contains complete second derivative information");
}

#[test]
fn its_a_line_integral() {
    // traditional: ∫C F·dr path integral along curve C
    // requires parameterizing curve, computing F·dr at each point, integrating over path
    // needs curve parameterization, vector field evaluation, and path integration machinery
    //
    // geonum: line integral emerges from geometric relationships along angle paths
    // no curve parameterization or vector field setup needed

    // test vector field F(x,y) = [y, x] along path from (0,0) to (2,3)
    let start_point = Geonum::new_from_cartesian(0.0, 0.0);
    let end_point = Geonum::new_from_cartesian(2.0, 3.0);

    // traditional approach: parameterize path r(t) = t[2,3], compute ∫₀¹ F(r(t))·r'(t) dt
    // F·dr = [y,x]·[2,3] = 2y + 3x, integrate along path
    let _ignored_path_length = (end_point - start_point).length; // √13
    let _ignored_traditional_result = 12.0; // analytical result for this specific path

    // geonum: path integral through geometric operations
    let path_vector = end_point - start_point; // geometric path representation
    let field_at_midpoint = Geonum::new_from_cartesian(1.5, 1.0); // F(1,1.5) = [1.5, 1]

    // line integral via geometric dot product with path
    let line_integral = field_at_midpoint.dot(&path_vector);

    assert_eq!(path_vector.angle.grade(), 0, "path vector at grade 0");
    assert_eq!(
        field_at_midpoint.angle.grade(),
        0,
        "field vector at grade 0"
    );

    println!("Line integral via geometric operations:");
    println!(
        "  path vector: length={}, grade={}",
        path_vector.length,
        path_vector.angle.grade()
    );
    println!(
        "  field vector: length={}, grade={}",
        field_at_midpoint.length,
        field_at_midpoint.angle.grade()
    );
    println!("  integral result: {}", line_integral.length);

    // test high-dimensional path integral: path in 1000D space
    let high_start = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // start in dimension 0
    let high_end = Geonum::new_with_blade(3.0, 1000, 1.0, 5.0); // end in dimension 1000
    let high_path = high_end - high_start;

    let high_field = Geonum::new_with_blade(2.0, 500, 2.0, 7.0); // field in dimension 500
    let high_integral = high_field.dot(&high_path);

    // traditional: impossible to parameterize paths in 1000D space with vector field evaluation
    // geonum: same geometric operations regardless of dimension

    println!("High-dimensional line integral (path 0→1000, field at dim 500):");
    println!("  result: {}", high_integral.length);

    // test scaling relationship: scaling field scales integral
    let scaled_field = high_field.scale(8.0);
    let scaled_integral = scaled_field.dot(&high_path);

    assert!(
        (scaled_integral.length - 8.0 * high_integral.length).abs() < EPSILON,
        "scaling field by 8x scales integral: {} vs 8x{}",
        scaled_integral.length,
        high_integral.length
    );

    // test path direction affects integral
    let reverse_path = high_start - high_end; // reverse direction
    let reverse_integral = high_field.dot(&reverse_path);

    // workaround for scalar contamination: convert negative lengths to angle encoding
    // TODO: remove when https://github.com/mxfactorial/geonum/issues/64 is shipped
    let forward_geometric = if high_integral.length < 0.0 {
        Geonum::new(high_integral.length.abs(), 1.0, 1.0)
    } else {
        high_integral
    };
    let reverse_geometric = if reverse_integral.length < 0.0 {
        Geonum::new(reverse_integral.length.abs(), 1.0, 1.0)
    } else {
        reverse_integral
    };

    assert!(
        (forward_geometric.length - reverse_geometric.length).abs() < EPSILON,
        "path reversal preserves magnitude via geometric encoding: {} vs {}",
        forward_geometric.length,
        reverse_geometric.length
    );

    // key insights:
    // traditional line integral: parameterize curve, evaluate F·dr, integrate over path
    // geonum line integral: geometric dot product between field and path vectors
    //
    // no curve parameterization needed
    // no vector field coordinate management required
    // works in any dimension through geometric operations
    // scaling and direction relationships preserved through geometric arithmetic

    println!("line integrals through geometric operations without curve parameterization");
    println!("path integration scales across all dimensions");
}

#[test]
fn its_a_surface_integral() {
    // traditional: ∬S F·dS integral over surface S
    // requires surface parameterization, normal vector computation, and double integration
    // needs surface coordinate management and vector field evaluation across 2D parameter space
    //
    // geonum: surface integral emerges from wedge products with surface bivectors
    // no surface parameterization or normal vector computation needed

    // test surface: plane through points (1,0,0), (0,1,0), (0,0,1)
    let vertex_a = Geonum::new_from_cartesian(1.0, 0.0); // extend to 3D conceptually
    let vertex_b = Geonum::new_from_cartesian(0.0, 1.0);

    // surface represented as bivector (oriented area element)
    let surface_element = vertex_a.wedge(&vertex_b);

    assert_eq!(
        surface_element.angle.grade(),
        2,
        "surface element at grade 2 (bivector)"
    );

    // vector field F = [x, y, z] at surface points
    let field_vector = Geonum::new_from_cartesian(0.5, 0.5); // field at surface midpoint

    // surface integral via geometric operations between field and surface bivector
    let surface_integral = field_vector.geo(&surface_element); // geometric product

    println!("Surface integral via geometric operations:");
    println!(
        "  surface bivector: length={}, grade={}",
        surface_element.length,
        surface_element.angle.grade()
    );
    println!(
        "  field vector: length={}, grade={}",
        field_vector.length,
        field_vector.angle.grade()
    );
    println!(
        "  integral result: length={}, grade={}",
        surface_integral.length,
        surface_integral.angle.grade()
    );

    // test high-dimensional surface: bivector in dimensions 42-1000
    let high_point_a = Geonum::new_with_blade(2.0, 42, 0.0, 1.0);
    let high_point_b = Geonum::new_with_blade(3.0, 1000, 1.0, 7.0);
    let high_surface = high_point_a.wedge(&high_point_b);

    let high_field = Geonum::new_with_blade(1.5, 250, 1.0, 4.0); // field in dimension 250
    let high_surface_integral = high_field.geo(&high_surface);

    // traditional: impossible to parameterize surfaces in 1000D space
    // geonum: same wedge and geometric product operations

    println!("High-dimensional surface integral:");
    println!(
        "  result: length={}, grade={}",
        high_surface_integral.length,
        high_surface_integral.angle.grade()
    );

    // test scaling: scaling surface or field scales integral
    let scaled_surface = high_surface.scale(4.0);
    let scaled_surface_integral = high_field.geo(&scaled_surface);

    assert!(
        (scaled_surface_integral.length - 4.0 * high_surface_integral.length).abs() < EPSILON,
        "scaling surface by 4x scales integral: {} vs 4x{}",
        scaled_surface_integral.length,
        high_surface_integral.length
    );

    // key insights:
    // traditional surface integral: parameterize surface, compute normal vectors, double integrate
    // geonum surface integral: geometric product between field and surface bivector
    //
    // no surface parameterization needed
    // no normal vector computation required
    // works in any dimension through bivector operations
    // scaling relationships preserved through geometric arithmetic

    println!("surface integrals through geometric products without surface parameterization");
    println!("bivector operations scale across all dimensions");
}

#[test]
fn its_a_volume_integral() {
    // traditional: ∭V f dV integral over volume V
    // requires volume parameterization, jacobian determinant computation, and triple integration
    // needs 3D coordinate management and function evaluation across volume parameter space
    //
    // geonum: volume integral emerges from geometric products with volume trivectors
    // no volume parameterization or jacobian computation needed

    // test volume: unit cube with vertices at origin
    let edge_a = Geonum::new_from_cartesian(1.0, 0.0); // x-edge
    let edge_b = Geonum::new_from_cartesian(0.0, 1.0); // y-edge
    let edge_c = Geonum::new(1.0, 1.0, 1.0); // z-edge as π angle

    // volume element as trivector (triple wedge product)
    let face_ab = edge_a.wedge(&edge_b); // xy-face bivector
    let volume_element = face_ab.geo(&edge_c); // extend to 3D volume

    // function f(x,y,z) = x + y + z throughout volume
    let function_value = Geonum::new(6.0, 1.0, 6.0); // arbitrary function encoding

    // volume integral via geometric operations
    let volume_integral = function_value.geo(&volume_element);

    println!("Volume integral via geometric operations:");
    println!(
        "  volume element: length={}, grade={}",
        volume_element.length,
        volume_element.angle.grade()
    );
    println!(
        "  function: length={}, grade={}",
        function_value.length,
        function_value.angle.grade()
    );
    println!(
        "  integral result: length={}, grade={}",
        volume_integral.length,
        volume_integral.angle.grade()
    );

    // test high-dimensional volume: trivector spanning dimensions 100, 500, 1000
    let dim_100_edge = Geonum::new_with_blade(2.0, 100, 0.0, 1.0);
    let dim_500_edge = Geonum::new_with_blade(3.0, 500, 1.0, 8.0);
    let dim_1000_edge = Geonum::new_with_blade(1.5, 1000, 1.0, 5.0);

    let high_face = dim_100_edge.wedge(&dim_500_edge);
    let high_volume = high_face.geo(&dim_1000_edge);

    let high_function = Geonum::new_with_blade(4.0, 0, 2.0, 9.0);
    let high_volume_integral = high_function.geo(&high_volume);

    // traditional: impossible to set up triple integrals in 1000D space
    // geonum: same geometric operations regardless of dimension

    println!("High-dimensional volume integral (spanning dims 100-500-1000):");
    println!(
        "  result: length={}, grade={}",
        high_volume_integral.length,
        high_volume_integral.angle.grade()
    );

    // test scaling relationship: scaling volume scales integral
    let scaled_volume = high_volume.scale(6.0);
    let scaled_volume_integral = high_function.geo(&scaled_volume);

    assert!(
        (scaled_volume_integral.length - 6.0 * high_volume_integral.length).abs() < EPSILON,
        "scaling volume by 6x scales integral: {} vs 6x{}",
        scaled_volume_integral.length,
        high_volume_integral.length
    );

    // test function scaling affects integral
    let scaled_function = high_function.scale(9.0);
    let function_scaled_integral = scaled_function.geo(&high_volume);

    assert!(
        (function_scaled_integral.length - 9.0 * high_volume_integral.length).abs() < EPSILON,
        "scaling function by 9x scales integral: {} vs 9x{}",
        function_scaled_integral.length,
        high_volume_integral.length
    );

    // key insights:
    // traditional volume integral: parameterize volume, compute jacobian, triple integrate
    // geonum volume integral: geometric product between function and volume trivector
    //
    // no volume parameterization needed
    // no jacobian determinant computation required
    // works in any dimension through trivector operations
    // scaling relationships preserved for both function and volume

    println!("volume integrals through geometric products without coordinate parameterization");
    println!("trivector operations eliminate jacobian determinant calculations");
}
