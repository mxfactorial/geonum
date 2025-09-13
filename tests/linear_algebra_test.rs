use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// "linear combination" is just a fancy name for "we scattered your angle across
// multiple numbers and now need complex algebra to track where it went"
//
// calling it "spanning the space" makes it sound profound when its really just
// "we need n numbers to store what could be 1 angle"

#[test]
fn it_proves_decomposing_angles_with_linearly_combined_basis_vectors_loses_angle_addition() {
    // DECOMPOSITION: expressing vectors as linear combinations v = c₁e₁ + c₂e₂
    // causes geometric product to compute angle DIFFERENCE instead of angle SUM
    //
    // THE LOST ANGLES: when multiplying vectors at angles θ₁ and θ₂,
    // decomposition gives (θ₂ - θ₁) but should give (θ₁ + θ₂)
    // The unadded portion is exactly 2θ₁
    //
    // GEONUM ACCOUNTS FOR THIS: by storing angles as primitives and adding them directly
    // via self.angle + other.angle (src/geonum_mod.rs:854)

    let theta1 = PI / 4.0; // 45°
    let theta2 = PI / 6.0; // 30°

    // DECOMPOSE vectors into basis coefficients
    let c1 = theta1.cos(); // coefficient for e₁
    let s1 = theta1.sin(); // coefficient for e₂
    let c2 = theta2.cos();
    let s2 = theta2.sin();

    println!("Vector 1 at angle {}: [{}, {}]", theta1, c1, s1);
    println!("Vector 2 at angle {}: [{}, {}]", theta2, c2, s2);

    // GEOMETRIC PRODUCT of decomposed vectors
    // (c₁e₁ + s₁e₂) * (c₂e₁ + s₂e₂) using basis multiplication rules
    let scalar_part = c1 * c2 + s1 * s2; // cos(θ₁-θ₂)
    let bivector_part = c1 * s2 - s1 * c2; // sin(θ₂-θ₁)

    // Extract angle from decomposed product
    let decomposed_angle = bivector_part.atan2(scalar_part);
    let expected_difference = theta2 - theta1;

    println!("\nDecomposition gives angle DIFFERENCE:");
    println!("  {} - {} = {}", theta2, theta1, decomposed_angle);
    assert!((decomposed_angle - expected_difference).abs() < EPSILON);

    // GEONUM multiplication adds angles
    let g1 = Geonum::new(1.0, theta1, PI);
    let g2 = Geonum::new(1.0, theta2, PI);
    let product = g1 * g2;

    let expected_sum = theta1 + theta2;
    println!("\nGeonum gives angle SUM:");
    println!(
        "  {} + {} = {}",
        theta1,
        theta2,
        product.angle.mod_4_angle()
    );
    assert!((product.angle.mod_4_angle() - expected_sum).abs() < EPSILON);
    assert_eq!(product.angle, g1.angle + g2.angle);

    // THE UNADDED ANGLES
    let unadded = expected_sum - decomposed_angle;
    println!("\nUNADDED ANGLES:");
    println!("  Should be: {} + {} = {}", theta1, theta2, expected_sum);
    println!(
        "  Decomposed gave: {} - {} = {}",
        theta2, theta1, decomposed_angle
    );
    println!(
        "  Missing (unadded): {} - {} = {}",
        expected_sum, decomposed_angle, unadded
    );
    println!("  This equals 2θ₁: 2 * {} = {}", theta1, 2.0 * theta1);

    // The critical assertion: decomposition lost exactly 2θ₁ worth of angle
    assert!(
        (unadded - 2.0 * theta1).abs() < EPSILON,
        "Decomposition loses 2θ₁ of angle that geonum accounts for"
    );
}

#[test]
fn it_proves_decomposition_distributes_one_angle_across_multiple_scalars() {
    // when you decompose a vector at angle θ,
    // the angle information gets distributed across multiple scalar coefficients

    let theta = 2.0 * PI / 7.0;

    // CARTESIAN DECOMPOSITION: 2 scalars encode 1 angle
    let cart_x = theta.cos();
    let cart_y = theta.sin();
    println!(
        "Cartesian: angle {} becomes ({}, {})",
        theta, cart_x, cart_y
    );
    println!(
        "  Must compute arctan({}/{}) to recover angle",
        cart_y, cart_x
    );

    // MATRIX DECOMPOSITION: 4 scalars encode 1 angle
    let m00 = theta.cos();
    let m01 = -theta.sin();
    let m10 = theta.sin();
    let m11 = theta.cos();
    println!("\nRotation matrix: angle {} becomes:", theta);
    println!("  [{:6.3}  {:6.3}]", m00, m01);
    println!("  [{:6.3}  {:6.3}]", m10, m11);

    // all 4 encode the same angle redundantly
    assert!((m00.acos() - theta).abs() < EPSILON);
    assert!((m10.asin() - theta).abs() < EPSILON);
    assert!(((-m01).asin() - theta).abs() < EPSILON);
    assert!((m11.acos() - theta).abs() < EPSILON);

    // GA DECOMPOSITION: angle split between grades
    let ga_scalar = theta.cos(); // grade 0 component
    let ga_bivector = theta.sin(); // grade 2 component
    println!("\nGA multivector: angle {} becomes:", theta);
    println!("  Grade 0 (scalar): {}", ga_scalar);
    println!("  Grade 2 (bivector): {}", ga_bivector);
    println!(
        "  Must compute arctan({}/{}) to recover",
        ga_bivector, ga_scalar
    );

    // COMPLEX DECOMPOSITION: 2 scalars (real, imaginary)
    let complex_real = theta.cos();
    let complex_imag = theta.sin();
    println!(
        "\nComplex: angle {} becomes {} + {}i",
        theta, complex_real, complex_imag
    );

    // EIGENDECOMPOSITION: conjugate pairs double the storage
    let lambda1_real = theta.cos();
    let lambda1_imag = theta.sin();
    let lambda2_real = theta.cos(); // redundant conjugate
    let lambda2_imag = -theta.sin(); // redundant conjugate
    println!(
        "\nEigenvalues: angle {} becomes two complex conjugates",
        theta
    );
    println!("  λ₁ = {} + {}i", lambda1_real, lambda1_imag);
    println!("  λ₂ = {} + {}i", lambda2_real, lambda2_imag);

    // COUNT THE REDUNDANCY
    let total_scalars = 2 + 4 + 2 + 2 + 4; // cart + matrix + GA + complex + eigen
    println!("\nTotal scalars used to store 1 angle: {}", total_scalars);

    // GEONUM: angle stored once
    let geonum = Geonum::new(1.0, theta, PI);
    assert_eq!(geonum.angle, Angle::new(theta, PI));
    println!("\nGeonum: angle {} stored directly as Angle struct", theta);
    println!("  No decomposition, no redundancy, no reconstruction");
}

#[test]
fn it_proves_decomposing_angles_into_scalar_coefficients_makes_angle_a_multivariate_function() {
    // DECOMPOSITION: expressing a vector as a linear combination of basis vectors
    // v = c₁e₁ + c₂e₂ where e₁, e₂ are basis vectors and c₁, c₂ are scalar coefficients
    //
    // SIDE EFFECT: angle becomes a multivariate function of coefficients
    // angle = f(c₁, c₂) = arctan(c₂/c₁)
    //
    // This creates multiple problems:
    // 1. Angle is no longer stored, it's computed
    // 2. Changing basis changes coefficients but same angle needs recomputation
    // 3. Operations on angles become operations on coefficient pairs
    // 4. Multiplication loses angle addition property

    let angle = PI / 3.0; // 60 degrees
    let magnitude = 5.0;

    // ORIGINAL GEOMETRIC OBJECT
    let geometric = Geonum::new(magnitude, angle, PI);
    println!("Original: magnitude={}, angle={}", magnitude, angle);

    // DECOMPOSITION INTO BASIS VECTORS
    // choose standard basis: e₁=[1,0], e₂=[0,1]
    // compute coefficients for linear combination v = c₁e₁ + c₂e₂
    let c1 = magnitude * angle.cos(); // projection onto e₁
    let c2 = magnitude * angle.sin(); // projection onto e₂

    println!("\nDecomposed into basis coefficients:");
    println!("  v = {}·e₁ + {}·e₂", c1, c2);

    // SIDE EFFECT 1: Angle becomes derived function
    let recovered_angle = c2.atan2(c1); // must compute angle from coefficients
    assert!((recovered_angle - angle).abs() < EPSILON);
    println!("  Angle = arctan({}/{}) = {}", c2, c1, recovered_angle);

    // SIDE EFFECT 2: Basis dependence
    // same vector in rotated basis has different coefficients
    let basis_rotation = PI / 4.0; // rotate basis by 45°
    let c1_rotated = magnitude * (angle - basis_rotation).cos();
    let c2_rotated = magnitude * (angle - basis_rotation).sin();

    println!("\nIn rotated basis (45° rotation):");
    println!("  v = {}·e₁' + {}·e₂'", c1_rotated, c2_rotated);

    // different coefficients, but still same angle after accounting for basis
    let angle_in_rotated = c2_rotated.atan2(c1_rotated) + basis_rotation;
    assert!((angle_in_rotated - angle).abs() < EPSILON);
    println!(
        "  Angle = arctan({}/{}) + {} = {}",
        c2_rotated, c1_rotated, basis_rotation, angle_in_rotated
    );

    // SIDE EFFECT 3: Simple rotation becomes complex coefficient update
    let rotation_angle = PI / 6.0; // rotate by 30°

    // geonum: just add angles
    let rotated_geometric = geometric.rotate(Angle::new(rotation_angle, PI));
    assert_eq!(
        rotated_geometric.angle,
        geometric.angle + Angle::new(rotation_angle, PI)
    );
    println!("\nRotating by {}:", rotation_angle);
    println!(
        "  Geonum: {} + {} = {}",
        angle,
        rotation_angle,
        rotated_geometric.angle.mod_4_angle()
    );

    // decomposition: must update both coefficients using rotation formulas
    let new_c1 = c1 * rotation_angle.cos() - c2 * rotation_angle.sin();
    let new_c2 = c1 * rotation_angle.sin() + c2 * rotation_angle.cos();
    let new_angle = new_c2.atan2(new_c1);
    assert!((new_angle - (angle + rotation_angle)).abs() < EPSILON);
    println!("  Decomposed: recompute c₁={}, c₂={}", new_c1, new_c2);
    println!(
        "              then arctan({}/{}) = {}",
        new_c2, new_c1, new_angle
    );

    // SIDE EFFECT 4: Multiplication loses angle addition
    let v1_angle = PI / 4.0;
    let v2_angle = PI / 5.0;

    // decompose both vectors
    let v1_c1 = v1_angle.cos();
    let v1_c2 = v1_angle.sin();
    let v2_c1 = v2_angle.cos();
    let v2_c2 = v2_angle.sin();

    println!("\nMultiplying vectors at {} and {}:", v1_angle, v2_angle);

    // geometric product of decomposed vectors
    // (v1_c1·e₁ + v1_c2·e₂) * (v2_c1·e₁ + v2_c2·e₂)
    let scalar_part = v1_c1 * v2_c1 + v1_c2 * v2_c2; // cos(θ₂-θ₁)
    let bivector_part = v1_c1 * v2_c2 - v1_c2 * v2_c1; // sin(θ₂-θ₁)

    let product_angle = bivector_part.atan2(scalar_part);
    println!(
        "  Decomposed gives: {} - {} = {}",
        v2_angle, v1_angle, product_angle
    );
    assert!((product_angle - (v2_angle - v1_angle)).abs() < EPSILON);

    // geonum preserves angle addition
    let g1 = Geonum::new(1.0, v1_angle, PI);
    let g2 = Geonum::new(1.0, v2_angle, PI);
    let product = g1 * g2;
    println!(
        "  Geonum gives: {} + {} = {}",
        v1_angle,
        v2_angle,
        product.angle.mod_4_angle()
    );
    assert!((product.angle.mod_4_angle() - (v1_angle + v2_angle)).abs() < EPSILON);

    // SIDE EFFECT 5: Constraints between coefficients
    // for unit vectors: c₁² + c₂² = 1 must be maintained
    let unit_c1 = angle.cos();
    let unit_c2 = angle.sin();
    assert!((unit_c1 * unit_c1 + unit_c2 * unit_c2 - 1.0).abs() < EPSILON);

    // can't modify one coefficient independently
    let broken_c1 = unit_c1 * 1.1; // scale just c₁
    let broken_norm = broken_c1 * broken_c1 + unit_c2 * unit_c2;
    assert!((broken_norm - 1.0).abs() > 0.01);
    println!("\nConstraint violation:");
    println!("  Original: c₁²+c₂² = {}²+{}² = 1", unit_c1, unit_c2);
    println!(
        "  Modified: c₁²+c₂² = {}²+{}² = {} ≠ 1",
        broken_c1, unit_c2, broken_norm
    );

    // SUMMARY: decomposing into linear combinations creates side effects
    println!("\nSIDE EFFECTS OF DECOMPOSITION:");
    println!("1. Angle becomes f(c₁,c₂) requiring arctan");
    println!("2. Basis changes require coefficient recomputation");
    println!("3. Rotation needs synchronized coefficient updates");
    println!("4. Multiplication gives angle difference not sum");
    println!("5. Coefficients must maintain algebraic constraints");

    // Geonum avoids all these by keeping angle as primitive
    println!("\nGEONUM SOLUTION: angle is primitive, not f(coefficients)");
}

#[test]
fn it_proves_linear_combination_forces_angle_through_coefficient_algebra() {
    // once decomposed as v = c₁e₁ + c₂e₂,
    // every angle operation becomes algebra on coefficient pairs

    let angle1 = PI / 6.0;
    let angle2 = PI / 4.0;

    // decompose into coefficients
    let a1 = angle1.cos();
    let b1 = angle1.sin();
    let a2 = angle2.cos();
    let b2 = angle2.sin();

    println!("Vector 1: {} → ({}, {})", angle1, a1, b1);
    println!("Vector 2: {} → ({}, {})", angle2, a2, b2);

    // ANGLE ADDITION requires complex algebra
    // to compute angle1 + angle2:
    let sum_cos = a1 * a2 - b1 * b2; // cos(θ₁+θ₂) = cos(θ₁)cos(θ₂) - sin(θ₁)sin(θ₂)
    let sum_sin = a1 * b2 + b1 * a2; // sin(θ₁+θ₂) = sin(θ₁)cos(θ₂) + cos(θ₁)sin(θ₂)
    let sum_angle = sum_sin.atan2(sum_cos);

    println!("\nTo add angles:");
    println!("  cos(θ₁+θ₂) = {}*{} - {}*{} = {}", a1, a2, b1, b2, sum_cos);
    println!("  sin(θ₁+θ₂) = {}*{} + {}*{} = {}", a1, b2, b1, a2, sum_sin);
    println!("  θ₁+θ₂ = arctan({}/{}) = {}", sum_sin, sum_cos, sum_angle);
    assert!((sum_angle - (angle1 + angle2)).abs() < EPSILON);

    // ANGLE DIFFERENCE also requires algebra
    let diff_cos = a1 * a2 + b1 * b2; // cos(θ₁-θ₂)
    let diff_sin = b1 * a2 - a1 * b2; // sin(θ₁-θ₂)
    let diff_angle = diff_sin.atan2(diff_cos);

    println!("\nTo subtract angles:");
    println!(
        "  cos(θ₁-θ₂) = {}*{} + {}*{} = {}",
        a1, a2, b1, b2, diff_cos
    );
    println!(
        "  sin(θ₁-θ₂) = {}*{} - {}*{} = {}",
        b1, a2, a1, b2, diff_sin
    );
    println!(
        "  θ₁-θ₂ = arctan({}/{}) = {}",
        diff_sin, diff_cos, diff_angle
    );
    assert!((diff_angle - (angle1 - angle2)).abs() < EPSILON);

    // ANGLE DOUBLING requires more algebra
    let double_cos = a1 * a1 - b1 * b1; // cos(2θ) = cos²(θ) - sin²(θ)
    let double_sin = 2.0 * a1 * b1; // sin(2θ) = 2sin(θ)cos(θ)
    let double_angle = double_sin.atan2(double_cos);

    println!("\nTo double angle:");
    println!("  cos(2θ) = {}² - {}² = {}", a1, b1, double_cos);
    println!("  sin(2θ) = 2*{}*{} = {}", a1, b1, double_sin);
    println!(
        "  2θ = arctan({}/{}) = {}",
        double_sin, double_cos, double_angle
    );
    assert!((double_angle - (2.0 * angle1)).abs() < EPSILON);

    // GEONUM: angles are primitives, operations are direct
    let g1 = Geonum::new(1.0, angle1, PI);
    let g2 = Geonum::new(1.0, angle2, PI);

    println!("\nGeonum operations:");
    println!(
        "  Add: {} + {} = {}",
        angle1,
        angle2,
        (g1 * g2).angle.mod_4_angle()
    );
    println!(
        "  Double: {} * 2 = {}",
        angle1,
        (g1 * g1).angle.mod_4_angle()
    );
    println!("  No coefficient algebra needed!");
}

#[test]
fn it_proves_angle_becomes_implicit_ratio_between_components() {
    // once decomposed into [x, y, z], angle only exists as ratios between components
    // angle is homeless - not stored anywhere, only computable from component relationships

    let theta = PI / 5.0; // 36 degrees
    let r = 4.0;

    // decompose into components
    let x = r * theta.cos();
    let y = r * theta.sin();

    println!("original angle: {}", theta);
    println!("becomes components: x={}, y={}", x, y);

    // angle now only exists as the ratio y:x
    let ratio = y / x; // tan(θ)
    println!("\nangle lives in ratio: y/x = {} = tan({})", ratio, theta);
    assert!((ratio - theta.tan()).abs() < EPSILON);

    // prove angle is implicit in ratio by scaling x
    let scaled_x = x * 1.5;
    let new_ratio = y / scaled_x;
    let distorted_angle = new_ratio.atan();

    println!("\nscale x by 1.5:");
    println!("  new ratio: {}/{} = {}", y, scaled_x, new_ratio);
    println!(
        "  implicit angle: atan({}) = {}",
        new_ratio, distorted_angle
    );

    // compute exact distortion from the ratio change
    let expected_distorted = (theta.tan() / 1.5).atan();
    let angle_change = theta - distorted_angle;
    assert!((distorted_angle - expected_distorted).abs() < EPSILON);
    assert!((angle_change - (theta - expected_distorted)).abs() < EPSILON);

    println!("  angle distorted by: {} radians", angle_change);

    // in 3D, angle is scattered across THREE ratios
    let phi = PI / 7.0; // elevation angle
    let z = r * phi.sin();
    let xy_plane = r * phi.cos();
    let x3d = xy_plane * theta.cos();
    let y3d = xy_plane * theta.sin();

    println!("\n3D: angle scattered across multiple ratios:");
    println!("  components: [{}, {}, {}]", x3d, y3d, z);

    // recovering angles requires multiple inverse trig operations
    let azimuth = (y3d / x3d).atan(); // horizontal angle from y:x ratio
    let elevation = (z / r).asin(); // vertical angle from z:r ratio

    println!("  azimuth from y/x: atan({}/{}) = {}", y3d, x3d, azimuth);
    println!("  elevation from z/r: asin({}/{}) = {}", z, r, elevation);
    assert!((azimuth - theta).abs() < EPSILON);
    assert!((elevation - phi).abs() < EPSILON);

    // changing ANY component changes implicit angles
    let perturbed_y = y3d * 0.9;
    let new_azimuth = (perturbed_y / x3d).atan();
    let azimuth_error = theta - new_azimuth;

    println!("\nperturb y by 10%:");
    println!(
        "  new azimuth: atan({}/{}) = {}",
        perturbed_y, x3d, new_azimuth
    );
    let expected_new_azimuth = (0.9 * theta.tan()).atan();
    assert!((new_azimuth - expected_new_azimuth).abs() < EPSILON);
    assert!((azimuth_error - (theta - expected_new_azimuth)).abs() < EPSILON);

    // the ratio nightmare in higher dimensions
    // 4D needs 6 ratios, 10D needs 45 ratios, nD needs n(n-1)/2 ratios
    println!("\nratio explosion in higher dimensions:");
    println!("  2D: 1 ratio (y/x)");
    println!("  3D: 3 ratios (y/x, z/r, z/xy)");
    println!("  4D: 6 ratios");
    println!("  10D: 45 ratios");
    println!("  nD: n(n-1)/2 ratios");

    // COMPLEXITY: matrix rotation in nD
    // - storage: O(n²) for rotation matrix
    // - computation: O(n²) multiply-adds to rotate one vector
    // - angle extraction: O(n²) inverse trig operations on ratios
    // matrix rotations must preserve ALL ratio relationships simultaneously
    let rotation = PI / 8.0;
    let rot_x = x * rotation.cos() - y * rotation.sin();
    let rot_y = x * rotation.sin() + y * rotation.cos();
    let rot_ratio = rot_y / rot_x;
    let rotated_angle = rot_ratio.atan();

    println!("\nrotation must update all components to preserve angle:");
    println!("  original: x={}, y={}, ratio={}", x, y, y / x);
    println!("  rotated: x={}, y={}, ratio={}", rot_x, rot_y, rot_ratio);
    println!("  angle: {} + {} = {}", theta, rotation, rotated_angle);
    assert!((rotated_angle - (theta + rotation)).abs() < EPSILON);

    // PERFORMANCE: decomposition vs geonum
    // decomposition in 10D:
    // - store 10 components + 100 matrix elements = 110 floats
    // - rotation: 100 multiply-adds
    // - angle recovery: 45 atan2 operations
    //
    // geonum in 10D (or ANY dimension):
    // - store 2 values: [length, angle]
    // - rotation: 1 angle addition
    // - angle access: direct field access (no computation)

    // geonum: angle is first-class, not a ratio ghost
    let g = Geonum::new(r, theta, PI);
    println!("\ngeonum keeps angle explicit:");
    println!("  angle = {} (directly stored)", g.angle.mod_4_angle());
    println!("  not computed from ratios");
    println!("  not scattered across components");
    println!("  not dependent on basis choice");

    // rotation is just angle addition: O(1) always
    let g_rotated = g.rotate(Angle::new(rotation, PI));
    assert!((g_rotated.angle.mod_4_angle() - (theta + rotation)).abs() < EPSILON);
    println!(
        "  rotation: {} + {} = {} (direct addition)",
        theta,
        rotation,
        g_rotated.angle.mod_4_angle()
    );
}
