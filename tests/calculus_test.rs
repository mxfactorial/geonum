// calculus is a geometric algorithm requiring forward and reverse quarter turns before projection
//
// traditional calculus works with scalars (post-projection quantities) and needs limits
// geometric calculus works with [magnitude, angle] primitives where relationships are encoded in angles
// quarter turns (±π/2) rotate between grades, then projection extracts scalar results
//
// forward quarter turn: +π/2 (differentiate)
// reverse quarter turn: +3π/2, dual to -π/2 (integrate)
// projection: extracts scalar from geometric structure
//
// differentiation is simply a pi/2 rotation and the foundation of
// calculus emerges directly from this geometric structure
//
// let v = [[1, 0], [1, pi/2]] # 2d
//
// everything can be a 1d "derivative" or projection of the base 2d v
// so long as the difference between their angles is pi/2 and they
// follow the "angles add, lengths multiply" rule
//
// v'       = [1, pi/2]  # first derivative (rotate v by pi/2)
// v''      = [1, pi]    # second derivative (rotate v' by pi/2) = -v
// v'''     = [1, 3pi/2] # third derivative (rotate v'' by pi/2) = -v'
// v''''    = [1, 2pi]   # fourth derivative (rotate v''' by pi/2) = v
// v'''''   = [1, 5pi/2] # fifth derivative (rotate v'''' by pi/2) = v'
// v''''''  = [1, 3pi]   # sixth derivative (rotate v''''' by pi/2) = -v
// v''''''' = [1, 7pi/2] # seventh derivative (rotate v'''''' by pi/2) = -v'
//
// this geometric space enables continuous rotation as an
// incrementing pi/2 angle, which is the essence of differentiation,
// and sets the period of the "derive" function to 4
//
// the wedge product between vectors AND their derivatives is nilpotent

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_limit() {
    // this test demonstrates that "limits" are unnecessary when using geometric numbers
    // limits use geometric operations then discard the geometry
    // they compute with [magnitude, angle] but collapse result to scalar
    // losing the normal and rotational structure

    let x: f64 = 3.0;
    let h = 0.0001;

    // limits perform geometric operations: finite difference + division
    // compute f(x+h) - f(x) for f(x) = x²
    let x_geo = Geonum::new(x, 0.0, 1.0);
    let x_h_geo = Geonum::new(x + h, 0.0, 1.0);
    let f_x = x_geo * x_geo; // x² = 9
    let f_x_h = x_h_geo * x_h_geo;

    let geometric_difference = f_x_h - f_x; // geometric subtraction

    // this produces a geometric number [magnitude, angle]
    assert!(
        geometric_difference.length < 0.01,
        "geometric difference small"
    );

    // divide by h geometrically
    let h_geo = Geonum::new(h, 0.0, 1.0);
    let geometric_quotient = geometric_difference / h_geo;

    // the quotient is still geometric: [magnitude, angle, grade]
    println!(
        "geometric quotient: [magnitude={:.6}, angle={:.6}, grade={}]",
        geometric_quotient.length,
        geometric_quotient.angle.grade_angle(),
        geometric_quotient.angle.grade()
    );

    // but traditional limits PROJECT to scalar, discarding angle structure
    let limit_result = geometric_quotient.length; // projection: lose angle
    assert!(
        limit_result > 5.0 && limit_result < 7.0,
        "limit projects to scalar ~6"
    );

    // differentiate() preserves the complete geometric structure
    let derivative = f_x.differentiate(); // rotates π/2

    // derivative contains both tangent and normal via the quarter turn
    let tangent = derivative.project_to_dimension(0);
    let normal = derivative.project_to_dimension(1);

    assert!(tangent.abs() < EPSILON, "tangent ≈ 0 at dimension 0");
    assert!(
        (normal - 9.0).abs() < EPSILON,
        "normal = 9 at dimension 1 (perpendicular)"
    );

    // limits extract tangent scalar but lose normal and rotation
    // limit gives ~6 (df/dx), tangent gives ~0 (different projections)
    assert!(
        limit_result > 5.0 && limit_result < 7.0,
        "limit extracts rate scalar"
    );
    assert_eq!(
        derivative.angle.grade(),
        1,
        "derivative at grade 1 preserves structure"
    );
    assert_eq!(
        derivative.length, f_x.length,
        "magnitude preserved in rotation"
    );

    // the key insight: limits ARE geometric operations (subtraction, division)
    // but they throw away the [magnitude, angle] result by projecting to scalar
    // losing the normal component and the quarter turn rotation that relates tangent to normal
    let angle_separation = derivative.angle - f_x.angle;
    assert_eq!(
        angle_separation,
        Angle::new(1.0, 2.0),
        "tangent-normal dual structure (quarter turn apart) lost in limit projection"
    );
}

#[test]
fn its_a_derivative() {
    // differentiate() computes the derivative via pi/2 rotation
    // the derivative appears perpendicular in the next dimension

    let position = Geonum::new(5.0, 0.0, 1.0);
    let velocity = position.differentiate();

    // velocity is perpendicular to position (quarter turn rotation)
    let angle_diff = velocity.angle - position.angle;
    assert_eq!(
        angle_diff,
        Angle::new(1.0, 2.0),
        "derivative rotates by quarter turn"
    );

    // magnitude preserved (rate equals magnitude for unit parameter)
    assert_eq!(velocity.length, position.length, "magnitude preserved");

    // grade changes: 0 → 1
    assert_eq!(position.angle.grade(), 0, "position at grade 0");
    assert_eq!(velocity.angle.grade(), 1, "velocity at grade 1");

    // project the derivative to extract the rate of change
    let rate_at_dim_1 = velocity.project_to_dimension(1);
    assert!(
        (rate_at_dim_1 - velocity.length).abs() < EPSILON,
        "velocity at grade 1 projects fully to dimension 1"
    );

    // use derivative to compute change: integrate velocity back to position change
    let delta_position = velocity.integrate(); // back to grade 0
    assert_eq!(
        delta_position.angle.grade(),
        0,
        "position change at grade 0"
    );
    assert_eq!(
        delta_position.length, velocity.length,
        "magnitude preserved"
    );

    // the change in position magnitude equals the velocity magnitude
    assert!(
        (delta_position.length - 5.0).abs() < EPSILON,
        "change in position extracted from derivative"
    );
}

#[test]
fn its_an_integral() {
    // integrate() computes definite integrals via reverse quarter turn rotation
    // fundamental theorem: ∫ₐᵇ f'(x)dx = F(b) - F(a)
    // scalar integral extracted via projection from grade 3

    // compute ∫₂⁵ 2x dx = x²|₂⁵ = 25 - 4 = 21
    let a: f64 = 2.0;
    let b: f64 = 5.0;

    let f_a = Geonum::new(a.powi(2), 0.0, 1.0); // F(2) = 4
    let f_b = Geonum::new(b.powi(2), 0.0, 1.0); // F(5) = 25

    // method 1: F(b) - F(a) at grade 0
    let difference = f_b - f_a;
    assert_eq!(difference.angle.grade(), 0, "difference at grade 0");
    assert!(
        (difference.length - 21.0).abs() < EPSILON,
        "magnitude at grade 0 equals integral directly"
    );

    // method 2: integrate() then project
    // integrate rotates by 3 quarter turns (reverse rotation, dual to -π/2)
    let integrated = difference.integrate();

    let angle_rotation = integrated.angle - difference.angle;
    assert_eq!(
        angle_rotation,
        Angle::new(3.0, 2.0),
        "integrate rotates by 3 quarter turns (reverse)"
    );

    assert_eq!(integrated.angle.grade(), 3, "integrated at grade 3");
    assert_eq!(
        integrated.length, difference.length,
        "magnitude preserved through rotation"
    );

    // project to dimension 3 to extract integral scalar
    let integral_scalar = integrated.project_to_dimension(3);
    assert!(
        (integral_scalar - 21.0).abs() < EPSILON,
        "dimension 3 projection extracts integral scalar"
    );

    // both methods give same result: scalars are projections
    assert!(
        (difference.length - integral_scalar).abs() < EPSILON,
        "magnitude at grade 0 = projection from grade 3"
    );
}

#[test]
fn it_computes_coefficients_from_geometric_division() {
    // polynomial derivative coefficients appear naturally from finite differences
    // divided by appropriate powers: coefficient for x^n = (Δf/Δx) / x^(n-1)

    let x_value: f64 = 4.0;
    let h = 0.0001;

    let x = Geonum::new(x_value, 0.0, 1.0);
    let x_h = Geonum::new(x_value + h, 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);

    // f(x) = x²: coefficient should be 2
    let f_squared = x * x;
    let f_squared_h = x_h * x_h;
    let coeff_2 = ((f_squared_h - f_squared) / h_geo) / x;

    assert!((coeff_2.length - 2.0).abs() < 0.01, "x² coefficient = 2");

    // f(x) = x³: coefficient should be 3
    let f_cubed = f_squared * x;
    let f_cubed_h = f_squared_h * x_h;
    let coeff_3 = ((f_cubed_h - f_cubed) / h_geo) / x / x; // divide by x²

    assert!((coeff_3.length - 3.0).abs() < 0.01, "x³ coefficient = 3");

    // f(x) = x⁴: coefficient should be 4
    let f_fourth = f_cubed * x;
    let f_fourth_h = f_cubed_h * x_h;
    let coeff_4 = ((f_fourth_h - f_fourth) / h_geo) / x / x / x; // divide by x³

    assert!((coeff_4.length - 4.0).abs() < 0.01, "x⁴ coefficient = 4");
}

#[test]
fn it_computes_powers_from_angle_ratios() {
    // when multiplying geometric numbers, angles add
    // for x^n, power n appears naturally in angle ratios: (x^n angle) / (x angle) = n

    let x_value: f64 = 2.0;
    let theta = 0.523598775; // π/6

    let x = Geonum::new(x_value, theta, 3.14159265359);
    let x_squared = x * x;
    let x_cubed = x_squared * x;
    let x_fourth = x_cubed * x;

    let x_angle = x.angle.grade_angle();
    let power_2 = x_squared.angle.grade_angle() / x_angle;
    let power_3 = x_cubed.angle.grade_angle() / x_angle;
    let power_4 = x_fourth.angle.grade_angle() / x_angle;

    assert!(
        (power_2 - 2.0).abs() < EPSILON,
        "x² power encoded in angle ratio"
    );
    assert!(
        (power_3 - 3.0).abs() < EPSILON,
        "x³ power encoded in angle ratio"
    );
    assert!(
        (power_4 - 4.0).abs() < EPSILON,
        "x⁴ power encoded in angle ratio"
    );
}

#[test]
fn it_computes_integrals_from_riemann_sums() {
    // integrals appear naturally from riemann sums: geometric multiplication + addition
    // ∫₀⁴ 2x dx = x²|₀⁴ = 16

    let x_start: f64 = 0.0;
    let x_end: f64 = 4.0;
    let num_steps = 1000;
    let dx = (x_end - x_start) / num_steps as f64;

    let dx_geo = Geonum::new(dx, 0.0, 1.0);
    let mut geometric_sum = Geonum::new(0.0, 0.0, 1.0);

    for i in 0..num_steps {
        let x_i = x_start + i as f64 * dx;
        let f_i = Geonum::new(2.0 * x_i, 0.0, 1.0); // f(x) = 2x
        let rectangle = f_i * dx_geo; // height × width
        geometric_sum = geometric_sum + rectangle;
    }

    let expected = 16.0; // x²|₀⁴ = 16 - 0
    assert!(
        (geometric_sum.length - expected).abs() < 0.02,
        "riemann sum computes integral via geometric operations"
    );
}

#[test]
fn its_a_gradient() {
    // traditional: ∇f = [∂f/∂x, ∂f/∂y] requires computing partials then assembling vector
    // geonum: ∇f = sum of directionally-encoded partials - no assembly needed

    let x = 3.0;
    let y = 4.0;
    let h = 0.0001;

    // f(x,y) = x² + y²
    let f_xy = x * x + y * y; // 25

    // traditional gradient: compute each partial, assemble into vector, compute magnitude
    let partial_x: f64 = ((x + h) * (x + h) + y * y - f_xy) / h; // ≈ 6
    let partial_y: f64 = (x * x + (y + h) * (y + h) - f_xy) / h; // ≈ 8
    let trad_magnitude: f64 = (partial_x * partial_x + partial_y * partial_y).sqrt(); // 10
    let trad_direction: f64 = partial_y.atan2(partial_x); // ≈ 0.927 rad

    // geometric gradient: compute partials via finite differences, encode with direction, add
    let f_geo = Geonum::new(f_xy, 0.0, 1.0);
    let f_xh = Geonum::new((x + h) * (x + h) + y * y, 0.0, 1.0);
    let f_yh = Geonum::new(x * x + (y + h) * (y + h), 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);

    // ∂f/∂x at angle 0 (x-direction)
    let df_dx = (f_xh - f_geo) / h_geo;
    let partial_x_geo = Geonum::new(df_dx.length, 0.0, 1.0);

    // ∂f/∂y at angle π/2 (y-direction)
    let df_dy = (f_yh - f_geo) / h_geo;
    let partial_y_geo = Geonum::new(df_dy.length, 1.0, 2.0);

    // gradient = sum of directionally-encoded partials
    let gradient = partial_x_geo + partial_y_geo;

    // prove they match
    assert!(
        (gradient.length - trad_magnitude).abs() < 0.01,
        "gradient magnitude matches: {} ≈ {}",
        gradient.length,
        trad_magnitude
    );
    assert!(
        (gradient.angle.grade_angle() - trad_direction).abs() < 0.01,
        "gradient direction matches: {} ≈ {}",
        gradient.angle.grade_angle(),
        trad_direction
    );
}

#[test]
fn its_a_divergence() {
    // traditional: ∇·F = ∂Fx/∂x + ∂Fy/∂y requires computing each partial then summing scalars
    // geonum: ∇·F = sum of geometric partials - magnitude gives divergence value

    let x = 2.0;
    let y = 3.0;
    let h = 0.0001;

    // vector field F(x,y) = [x², xy] at point (2,3)
    let fx = x * x; // 4
    let fy = x * y; // 6

    // traditional divergence: ∂Fx/∂x + ∂Fy/∂y = 2x + x = 3x = 6
    let dfx_dx: f64 = (((x + h) * (x + h)) - fx) / h; // 2x ≈ 4
    let dfy_dy: f64 = ((x * (y + h)) - fy) / h; // x ≈ 2
    let trad_divergence: f64 = dfx_dx + dfy_dy; // 6

    // geometric divergence: compute partials via finite differences, sum
    let fx_geo = Geonum::new(fx, 0.0, 1.0);
    let fx_h = Geonum::new((x + h) * (x + h), 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);
    let dfx_dx_geo = (fx_h - fx_geo) / h_geo;

    let fy_geo = Geonum::new(fy, 0.0, 1.0);
    let fy_h = Geonum::new(x * (y + h), 0.0, 1.0);
    let dfy_dy_geo = (fy_h - fy_geo) / h_geo;

    // each partial derivative is at grade 2 (result of division operations)
    assert_eq!(dfx_dx_geo.angle.grade(), 2, "∂Fx/∂x at grade 2");
    assert_eq!(dfy_dy_geo.angle.grade(), 2, "∂Fy/∂y at grade 2");

    // divergence = sum of partials
    let divergence = dfx_dx_geo + dfy_dy_geo;

    // divergence magnitude matches traditional scalar divergence
    assert!(
        (divergence.length - trad_divergence).abs() < 0.01,
        "divergence magnitude matches: {} ≈ {}",
        divergence.length,
        trad_divergence
    );

    // divergence remains at grade 2 (sum of grade 2 objects)
    assert_eq!(
        divergence.angle.grade(),
        2,
        "divergence at grade 2 (bivector)"
    );
}

#[test]
fn its_a_curl() {
    // traditional: ∇×F = ∂Fy/∂x - ∂Fx/∂y (2D curl, z-component)
    // geonum: curl = difference of cross partials - magnitude gives circulation

    let x = 2.0;
    let y = 3.0;
    let h = 0.0001;

    // rotational vector field F(x,y) = [-y, x] at point (2,3)
    let fx = -y; // -3
    let fy = x; // 2

    // traditional curl: ∂Fy/∂x - ∂Fx/∂y = 1 - (-1) = 2
    let dfy_dx: f64 = ((x + h) - fy) / h; // ∂Fy/∂x = 1
    let dfx_dy: f64 = (-(y + h) - fx) / h; // ∂Fx/∂y = -1
    let trad_curl: f64 = dfy_dx - dfx_dy; // 2

    // geometric curl: compute cross partials via finite differences, subtract
    let fy_geo = Geonum::new(fy, 0.0, 1.0);
    let fy_xh = Geonum::new(x + h, 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);
    let dfy_dx_geo = (fy_xh - fy_geo) / h_geo;

    let fx_geo = Geonum::new(fx, 0.0, 1.0);
    let fx_yh = Geonum::new(-(y + h), 0.0, 1.0);
    let dfx_dy_geo = (fx_yh - fx_geo) / h_geo;

    // curl = ∂Fy/∂x - ∂Fx/∂y
    let curl = dfy_dx_geo - dfx_dy_geo;

    // prove curl magnitude matches
    assert!(
        (curl.length - trad_curl).abs() < 0.01,
        "curl magnitude matches: {} ≈ {}",
        curl.length,
        trad_curl
    );

    // curl is at grade 2 (result of division operations)
    assert_eq!(curl.angle.grade(), 2, "curl at grade 2 (bivector)");
}

#[test]
fn its_a_directional_derivative() {
    // traditional: D_û f = ∇f·û requires gradient vector dotted with unit direction
    // geonum: D_û f = gradient.dot(direction) - same operation, geometric structure

    let x = 3.0;
    let y = 4.0;
    let h = 0.0001;

    // f(x,y) = x² + y² at (3,4), gradient = [6, 8]
    let f_xy = x * x + y * y; // 25
    let partial_x: f64 = ((x + h) * (x + h) + y * y - f_xy) / h; // 6
    let partial_y: f64 = (x * x + (y + h) * (y + h) - f_xy) / h; // 8

    // direction: û = [1/√2, 1/√2] (45° direction)
    let dir_x: f64 = 1.0 / 2.0_f64.sqrt();
    let dir_y: f64 = 1.0 / 2.0_f64.sqrt();

    // traditional directional derivative: ∇f·û = 6*(1/√2) + 8*(1/√2) ≈ 9.899
    let trad_dir_deriv: f64 = partial_x * dir_x + partial_y * dir_y;

    // geometric: build gradient, dot with direction
    let f_geo = Geonum::new(f_xy, 0.0, 1.0);
    let f_xh = Geonum::new((x + h) * (x + h) + y * y, 0.0, 1.0);
    let f_yh = Geonum::new(x * x + (y + h) * (y + h), 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);

    let df_dx = (f_xh - f_geo) / h_geo;
    let partial_x_geo = Geonum::new(df_dx.length, 0.0, 1.0);

    let df_dy = (f_yh - f_geo) / h_geo;
    let partial_y_geo = Geonum::new(df_dy.length, 1.0, 2.0);

    let gradient = partial_x_geo + partial_y_geo;
    let direction = Geonum::new(1.0, 1.0, 4.0); // π/4 = 45°

    // directional derivative = gradient · direction
    let dir_deriv = gradient.dot(&direction);

    assert!(
        (dir_deriv.length - trad_dir_deriv).abs() < 0.1,
        "directional derivative matches: {} ≈ {}",
        dir_deriv.length,
        trad_dir_deriv
    );
}

#[test]
fn its_a_laplacian() {
    // traditional: ∇²f = ∂²f/∂x² + ∂²f/∂y² sum of second partials
    // geonum: compute second partials geometrically, sum → magnitude extraction

    let x = 2.0;
    let y = 3.0;
    let h = 0.0001;

    // f(x,y) = x² + y²
    let f_xy = x * x + y * y; // 13

    // traditional laplacian: ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
    let f_xh = (x + h) * (x + h) + y * y;
    let f_x_h = (x - h) * (x - h) + y * y;
    let d2f_dx2: f64 = (f_xh - 2.0 * f_xy + f_x_h) / (h * h); // 2

    let f_yh = x * x + (y + h) * (y + h);
    let f_y_h = x * x + (y - h) * (y - h);
    let d2f_dy2: f64 = (f_yh - 2.0 * f_xy + f_y_h) / (h * h); // 2

    let trad_laplacian: f64 = d2f_dx2 + d2f_dy2; // 4

    // geometric laplacian: compute second partials via finite differences
    let f_geo = Geonum::new(f_xy, 0.0, 1.0);
    let f_xh_geo = Geonum::new(f_xh, 0.0, 1.0);
    let f_x_h_geo = Geonum::new(f_x_h, 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);
    let h2_geo = h_geo * h_geo; // h²

    // ∂²f/∂x² = (f(x+h) - 2f(x) + f(x-h)) / h²
    let d2f_dx2_geo = (f_xh_geo - f_geo.scale(2.0) + f_x_h_geo) / h2_geo;

    let f_yh_geo = Geonum::new(f_yh, 0.0, 1.0);
    let f_y_h_geo = Geonum::new(f_y_h, 0.0, 1.0);

    // ∂²f/∂y²
    let d2f_dy2_geo = (f_yh_geo - f_geo.scale(2.0) + f_y_h_geo) / h2_geo;

    // laplacian = sum of second partials
    let laplacian = d2f_dx2_geo + d2f_dy2_geo;

    // prove laplacian magnitude matches
    assert!(
        (laplacian.length - trad_laplacian).abs() < 0.1,
        "laplacian magnitude matches: {} ≈ {}",
        laplacian.length,
        trad_laplacian
    );

    // laplacian at grade 2 from division operations
    assert_eq!(laplacian.angle.grade(), 2, "laplacian at grade 2");
}

#[test]
fn it_handles_partial_derivatives() {
    // traditional: ∂f/∂x "holds y constant" - requires conceptual freezing of dimensions
    // geonum: gradient already contains all directional info - just project

    let x = 3.0;
    let y = 4.0;
    let h = 0.0001;

    // f(x,y) = x² + y² at (3,4)
    let f_xy = x * x + y * y; // 25

    // traditional partials: "hold y constant" for ∂f/∂x, "hold x constant" for ∂f/∂y
    let partial_x_trad: f64 = ((x + h) * (x + h) + y * y - f_xy) / h; // 2x = 6
    let partial_y_trad: f64 = (x * x + (y + h) * (y + h) - f_xy) / h; // 2y = 8

    // geometric: build gradient (already contains all directional information)
    let f_geo = Geonum::new(f_xy, 0.0, 1.0);
    let f_xh = Geonum::new((x + h) * (x + h) + y * y, 0.0, 1.0);
    let f_yh = Geonum::new(x * x + (y + h) * (y + h), 0.0, 1.0);
    let h_geo = Geonum::new(h, 0.0, 1.0);

    let df_dx = (f_xh - f_geo) / h_geo;
    let partial_x_geo = Geonum::new(df_dx.length, 0.0, 1.0);

    let df_dy = (f_yh - f_geo) / h_geo;
    let partial_y_geo = Geonum::new(df_dy.length, 1.0, 2.0);

    let gradient = partial_x_geo + partial_y_geo;

    // extract partials via projection - no "freezing" needed
    let x_axis = Geonum::new(1.0, 0.0, 1.0);
    let y_axis = Geonum::new(1.0, 1.0, 2.0);

    let partial_x_projected = gradient.dot(&x_axis);
    let partial_y_projected = gradient.dot(&y_axis);

    // prove they match
    assert!(
        (partial_x_projected.length - partial_x_trad).abs() < 0.2,
        "x-partial matches: {} ≈ {}",
        partial_x_projected.length,
        partial_x_trad
    );
    assert!(
        (partial_y_projected.length - partial_y_trad).abs() < 0.2,
        "y-partial matches: {} ≈ {}",
        partial_y_projected.length,
        partial_y_trad
    );
}

#[test]
fn its_a_line_integral() {
    // traditional: ∫_C F·dr requires curve parameterization, dr/dt computation, integration
    // geonum: field · path for constant field on straight path

    // straight line from (0,0) to (2,3)
    let start = Geonum::new_from_cartesian(0.0, 0.0);
    let end = Geonum::new_from_cartesian(2.0, 3.0);
    let path = end - start;

    // constant vector field F = [1, 2]
    let field = Geonum::new_from_cartesian(1.0, 2.0);

    // traditional: ∫_C F·dr = F·(end - start) = [1,2]·[2,3] = 1*2 + 2*3 = 8
    let trad_integral: f64 = 1.0 * 2.0 + 2.0 * 3.0;

    // geometric: field · path
    let geo_integral = field.dot(&path);

    // prove they match
    assert!(
        (geo_integral.length - trad_integral).abs() < 0.1,
        "line integral matches: {} ≈ {}",
        geo_integral.length,
        trad_integral
    );
}

#[test]
fn its_a_surface_integral() {
    // traditional: ∬_S F·n dS requires surface parameterization and normal vector computation
    // geonum: surface as bivector (wedge product) - magnitude gives area

    // rectangular surface with edges [2,0] and [0,3]
    let edge_x = Geonum::new_from_cartesian(2.0, 0.0);
    let edge_y = Geonum::new_from_cartesian(0.0, 3.0);

    // traditional surface area via multiplication
    let trad_area: f64 = 2.0 * 3.0; // 6

    // geometric surface: wedge product creates bivector
    let surface = edge_x.wedge(&edge_y);

    // wedge product magnitude IS the area
    assert!(
        (surface.length - trad_area).abs() < EPSILON,
        "surface area matches: {} ≈ {}",
        surface.length,
        trad_area
    );

    // surface at grade 2 (bivector)
    assert_eq!(surface.angle.grade(), 2, "surface at grade 2 (bivector)");
}

#[test]
fn its_a_volume_integral() {
    // traditional: ∭_V f dV requires volume parameterization and Jacobian computation
    // geonum: volume from geometric product of surface bivector with third edge

    // rectangular volume with edges [2,0], [0,3], and perpendicular edge of length 4
    let edge_x = Geonum::new_from_cartesian(2.0, 0.0);
    let edge_y = Geonum::new_from_cartesian(0.0, 3.0);
    let edge_z = Geonum::new_with_blade(4.0, 2, 0.0, 1.0); // perpendicular at grade 2

    // traditional volume via multiplication
    let trad_volume: f64 = 2.0 * 3.0 * 4.0; // 24

    // geometric volume: surface bivector ⊗ third edge
    let surface = edge_x.wedge(&edge_y); // bivector at grade 2
    let volume = surface.geo(&edge_z); // geometric product

    // volume magnitude matches
    assert!(
        (volume.length - trad_volume).abs() < EPSILON,
        "volume matches: {} ≈ {}",
        volume.length,
        trad_volume
    );

    // volume at grade 0 (cycles back through 4-grade structure)
    assert_eq!(volume.angle.grade(), 0, "volume at grade 0");
}

#[test]
fn it_encodes_definite_integrals_with_domain() {
    // ∫₂⁵ x² dx = 39
    // traditional: only the value 39
    // angle space: value AND domain in one geonum

    // compute traditionally for comparison
    let x_a: f64 = 2.0;
    let x_b: f64 = 5.0;
    let traditional: f64 = (x_b.powi(3) - x_a.powi(3)) / 3.0; // 39

    // encode bounds as angles (x as multiples of π)
    let angle_a = Angle::new(x_a, 1.0); // 2π radians
    let angle_b = Angle::new(x_b, 1.0); // 5π radians

    // evaluate antiderivative at bounds
    // F(x) = x³/3
    let f_a = Geonum::new_with_angle(x_a.powi(3) / 3.0, angle_a);
    let f_b = Geonum::new_with_angle(x_b.powi(3) / 3.0, angle_b);

    // the definite integral encodes:
    // - magnitude: F(b) - F(a) = integral value
    // - angle: x_b - x_a = integration domain
    let magnitude = f_b.length - f_a.length;
    let angle = f_b.angle - f_a.angle;
    let integral = Geonum::new_with_angle(magnitude, angle);

    // verify value matches traditional
    assert!(
        (integral.length - traditional).abs() < EPSILON,
        "expected {}, got {}",
        traditional,
        integral.length
    );

    // the angle encodes the domain (as multiples of π)
    let expected_angle = Angle::new(x_b - x_a, 1.0); // (5-2) * π = 3π
    assert_eq!(
        integral.angle, expected_angle,
        "angle should encode domain span"
    );

    // traditional calculus: ∫₂⁵ x² dx = 39 (value only)
    // angle space calculus: [magnitude=39, angle=3π, blade=6, grade=2]
    //   - magnitude: the integral value
    //   - angle: 3π (the domain spanned as multiples of π)
    //   - blade: 6 (accumulated π/2 rotations)
    //   - grade: 2 (bivector, blade % 4)
}

#[test]
fn it_preserves_fundamental_theorem_via_magnitudes() {
    // fundamental theorem: ∫ₐᵇ f(x) dx = F(b) - F(a)
    // in angle space: |F(b)| - |F(a)|

    // ∫₁³ 2x dx = x² |₁³ = 9 - 1 = 8
    let x_a: f64 = 1.0;
    let x_b: f64 = 3.0;
    let traditional: f64 = x_b.powi(2) - x_a.powi(2); // 8

    let f_a = Geonum::new_with_angle(x_a.powi(2), Angle::new(x_a, 1.0));
    let f_b = Geonum::new_with_angle(x_b.powi(2), Angle::new(x_b, 1.0));

    let integral_value = f_b.length - f_a.length;

    assert!(
        (integral_value - traditional).abs() < EPSILON,
        "expected {}, got {}",
        traditional,
        integral_value
    );
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
        (angle_0.grade_angle().cos() - angle_90.grade_angle().sin()).abs() < EPSILON,
        "cos(θ) = sin(θ+π/2)"
    );
    assert!(
        (angle_0.grade_angle().sin() + angle_90.grade_angle().cos()).abs() < EPSILON,
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
}

#[test]
fn it_demonstrates_differentiate_on_polynomials() {
    // differentiate() rotates by π/2, preserving magnitude and cycling grades
    // test on polynomial evaluations at specific points

    // test x² at x = 3
    let x = 3.0;
    let f_scalar = x * x; // 9
    let f = Geonum::new(f_scalar, 0.0, 1.0); // [9, 0] at grade 0

    assert_eq!(f.angle.grade(), 0, "f(x) at grade 0");
    assert_eq!(f.length, 9.0, "f(3) magnitude is 9");

    // differentiation: π/2 rotation moves to grade 1
    let f_prime = f.differentiate();
    assert_eq!(f_prime.angle.grade(), 1, "f'(x) at grade 1 (vector-like)");
    assert_eq!(f_prime.length, 9.0, "differentiation preserves magnitude");

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
        (base_angle.grade_angle().cos() - rotated_angle.grade_angle().sin()).abs() < EPSILON,
        "cos(θ) = sin(θ + π/2) enables differentiation"
    );

    // test polynomial chain: f(x) = x³
    let x3 = x * x * x; // 27
    let f_cubic = Geonum::new(x3, 0.0, 1.0); // [27, 0] at grade 0
    let f_cubic_prime = f_cubic.differentiate();

    assert_eq!(
        f_cubic_prime.angle.grade(),
        1,
        "cubic derivative at grade 1"
    );

    // demonstrate second derivative: f''(x) for f(x) = x³
    let f_cubic_double_prime = f_cubic_prime.differentiate();
    assert_eq!(
        f_cubic_double_prime.angle.grade(),
        2,
        "second derivative at grade 2 (bivector)"
    );

    // test constant function: f(x) = 5
    let constant = Geonum::new(5.0, 0.0, 1.0); // [5, 0] at grade 0
    let constant_prime = constant.differentiate();

    assert_eq!(
        constant_prime.length, 5.0,
        "differentiation preserves magnitude"
    );
    assert_eq!(
        constant_prime.angle.grade(),
        1,
        "constant derivative at grade 1"
    );

    // test linear function: f(x) = 2x
    let linear_value = 2.0 * x; // 6
    let f_linear = Geonum::new(linear_value, 0.0, 1.0); // [6, 0]
    let f_linear_prime = f_linear.differentiate();

    assert_eq!(
        f_linear_prime.length, 6.0,
        "linear function derivative preserves magnitude"
    );
    assert_eq!(
        f_linear_prime.angle.grade(),
        1,
        "linear derivative at grade 1"
    );
}

#[test]
fn it_proves_fundamental_theorem_is_accumulation_equals_interference() {
    // Newton-Leibniz theorem: ∫ₐᵇ f'(x) dx = F(b) - F(a)
    // in angle space: accumulated geometric sum = destructive interference of endpoints

    // projection space: ∫₂⁵ 2x dx = x²|₂⁵ = 25 - 4 = 21
    let a: f64 = 2.0;
    let b: f64 = 5.0;
    let traditional_left: f64 = b.powi(2) - a.powi(2); // 21

    // angle space left side: accumulation via geometric addition
    // integrate f'(x) = 2x from 2 to 5 via riemann sum
    let num_steps = 1000;
    let dx = (b - a) / num_steps as f64;
    let dx_geo = Geonum::new(dx, 0.0, 1.0);
    let mut accumulated_sum = Geonum::new(0.0, 0.0, 1.0);

    for i in 0..num_steps {
        let x_i = a + i as f64 * dx;
        let f_prime_i = Geonum::new(2.0 * x_i, 0.0, 1.0); // f'(x) = 2x
        let rectangle = f_prime_i * dx_geo;
        accumulated_sum = accumulated_sum + rectangle; // geometric addition
    }

    // angle space right side: F(b) - F(a) as destructive interference
    let f_b = Geonum::new(b.powi(2), 0.0, 1.0); // F(5) = [25, 0]
    let f_a_negated = Geonum::new(a.powi(2), 1.0, 1.0); // [4, π]
    let interference_result = f_b + f_a_negated;

    // verify cosine rule: c² = 625 + 16 + 2(25)(4)cos(π) = 625 + 16 - 200 = 441
    let expected_squared = f_b.length.powi(2) + a.powi(4) + 2.0 * f_b.length * a.powi(2) * PI.cos();
    assert!((expected_squared - 441.0).abs() < EPSILON);
    assert!((expected_squared.sqrt() - 21.0).abs() < EPSILON);

    // fundamental theorem: accumulation equals interference
    assert!(
        (accumulated_sum.length - interference_result.length).abs() < 0.02,
        "left side (accumulation) {:.3} = right side (interference) {:.3}",
        accumulated_sum.length,
        interference_result.length
    );

    assert!(
        (accumulated_sum.length - traditional_left).abs() < 0.02,
        "angle space {:.3} matches projection space {}",
        accumulated_sum.length,
        traditional_left
    );
}

#[test]
fn it_shows_why_subtraction_appears_in_fundamental_theorem() {
    // the "minus" in F(b) - F(a) is destructive interference, not algebraic subtraction

    // ∫₁³ 2x dx = x²|₁³ = 9 - 1 = 8
    let a: f64 = 1.0;
    let b: f64 = 3.0;

    // endpoint values of antiderivative
    let f_b = Geonum::new(b.powi(2), 0.0, 1.0); // [9, 0]
    let f_a_at_pi = Geonum::new(a.powi(2), 1.0, 1.0); // [1, π]
    let interference = f_b + f_a_at_pi;

    // verify cosine rule: c² = 81 + 1 + 2(9)(1)(-1) = 81 + 1 - 18 = 64
    let expected = (81.0_f64 + 1.0 - 18.0).sqrt();
    assert!((expected - 8.0).abs() < EPSILON);
    assert!(
        (interference.length - 8.0).abs() < EPSILON,
        "interference magnitude via cos(π) = -1: {:.3}",
        interference.length
    );
}

#[test]
fn it_reveals_integral_as_interference_accumulator() {
    // integration accumulates geometric additions
    // Newton-Leibniz says: net accumulation = interference between bounds

    // ∫₀⁴ x dx = ½x²|₀⁴ = 8 - 0 = 8
    let a: f64 = 0.0;
    let b: f64 = 4.0;

    // accumulate via riemann sum
    let num_steps = 1000;
    let dx = (b - a) / num_steps as f64;
    let dx_geo = Geonum::new(dx, 0.0, 1.0);
    let mut accumulation = Geonum::new(0.0, 0.0, 1.0);

    for i in 0..num_steps {
        let x_i = a + i as f64 * dx;
        let f_i = Geonum::new(x_i, 0.0, 1.0); // f(x) = x
        let area = f_i * dx_geo;
        accumulation = accumulation + area; // each step: geometric addition
    }

    // endpoint interference
    let f_b = Geonum::new(0.5 * b.powi(2), 0.0, 1.0); // ½(16) = [8, 0]
    let f_a_negated = Geonum::new(0.5 * a.powi(2), 1.0, 1.0); // [0, π]
    let interference = f_b + f_a_negated;

    // they equal
    assert!(
        (accumulation.length - interference.length).abs() < 0.02,
        "accumulation {:.3} = interference {:.3}",
        accumulation.length,
        interference.length
    );
    assert!((interference.length - 8.0).abs() < EPSILON);
}

#[test]
fn it_connects_differentiation_and_antiderivative_via_angles() {
    // differentiate() rotates by π/2 (grade 0 → 1)
    // integrate() rotates by 3π/2 (grade 1 → 0, forward equivalent to -π/2)
    // fundamental theorem connects these rotations

    let f = Geonum::new(16.0, 0.0, 1.0); // F(x) at some point, grade 0

    // differentiate: rotate π/2 to grade 1
    let f_prime = f.differentiate();
    assert_eq!(f_prime.angle.grade(), 1, "derivative at grade 1");
    assert_eq!(f_prime.length, 16.0, "magnitude preserved");

    // integrate: rotate 3π/2 back to grade 0
    let back_to_f = f_prime.integrate();
    assert_eq!(back_to_f.angle.grade(), 0, "integrated back to grade 0");
    assert_eq!(back_to_f.length, 16.0, "magnitude preserved");

    // the angles connect differentiation to integration
    let angle_cycle = f_prime.angle - f.angle; // differentiation rotation
    let angle_back = back_to_f.angle - f_prime.angle; // integration rotation

    assert_eq!(angle_cycle, Angle::new(1.0, 2.0), "differentiate adds π/2");
    assert_eq!(angle_back, Angle::new(3.0, 2.0), "integrate adds 3π/2");

    // net rotation: 4 blades (full 2π cycle)
    assert_eq!(
        back_to_f.angle.blade() - f.angle.blade(),
        4,
        "full cycle: differentiate then integrate adds 4 blades"
    );
}

#[test]
fn it_shows_definite_integral_encodes_both_value_and_domain() {
    // ∫₂⁵ x² dx = ⅓x³|₂⁵ = 125/3 - 8/3 = 39
    let a: f64 = 2.0;
    let b: f64 = 5.0;
    let traditional_value = (b.powi(3) - a.powi(3)) / 3.0; // 39

    // encode bounds as angles
    let angle_a = Angle::new(a, 1.0); // 2π
    let angle_b = Angle::new(b, 1.0); // 5π

    // antiderivative values with angle encoding
    let f_a = Geonum::new_with_angle(a.powi(3) / 3.0, angle_a); // [8/3, 2π]
    let f_b = Geonum::new_with_angle(b.powi(3) / 3.0, angle_b); // [125/3, 5π]

    // the integral encodes BOTH value and domain
    let value = f_b.length - f_a.length; // magnitude difference
    let domain = f_b.angle - f_a.angle; // angle difference

    assert!(
        (value - traditional_value).abs() < EPSILON,
        "value matches traditional: {:.3} ≈ {}",
        value,
        traditional_value
    );

    let expected_domain = Angle::new(b - a, 1.0); // (5-2)π = 3π
    assert_eq!(domain, expected_domain, "angle encodes domain span");

    // create the complete encoding
    let integral = Geonum::new_with_angle(value, domain);
    assert!(
        (integral.length - 39.0).abs() < EPSILON,
        "magnitude: integral value"
    );
    assert_eq!(
        integral.angle,
        Angle::new(3.0, 1.0),
        "angle: domain span 3π"
    );
}
