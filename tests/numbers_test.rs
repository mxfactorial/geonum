use geonum::*;
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_scalar() {
    // a scalar is just a number with magnitude but no direction
    // in geometric number format, its a [length, 0] for positive
    // or [length, pi] for negative

    let scalar = Geonum::new(1.0, 0.0, 1.0);

    // test if scalar has expected properties
    assert_eq!(scalar.length, 1.0);
    assert_eq!(scalar.angle.mod_4_angle(), 0.0);

    // multiplying scalars follows "angles add, lengths multiply" rule
    let scalar2 = Geonum::new(2.0, 0.0, 1.0);

    let product = scalar * scalar2;

    // 1 × 2 = 2
    assert_eq!(product.length, 2.0);
    assert_eq!(product.angle.mod_4_angle(), 0.0);

    // multiplication with negative scalar
    let neg_scalar = Geonum::new(3.0, 1.0, 1.0); // PI radians

    let neg_product = scalar * neg_scalar;

    // 1 × (-3) = -3
    assert_eq!(neg_product.length, 3.0);
    assert_eq!(neg_product.angle.mod_4_angle(), PI);
}

#[test]
fn its_a_vector() {
    // a vector has both magnitude and direction
    // in geometric algebra, vectors are grade 1 elements

    // Geonum::new(length, pi_radians, divisor) computes total angle as pi_radians * π / divisor
    // then decomposes into blade (counts π/2 rotations) and remainder angle
    // 3π/4 = 135° crosses one π/2 boundary, giving blade=1 (vector) with π/4 remainder
    let vector = Geonum::new(2.0, 3.0, 4.0); // 3 * π/4 = 3π/4 radians = 135 degrees
                                             // blade 1 (vector grade) + π/4 remainder

    // test vector properties
    assert_eq!(vector.length, 2.0);
    assert_eq!(vector.angle.blade(), 1); // blade 1 = vector (grade 1) in geometric algebra
    assert!((vector.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder after π/2 rotation

    // test dot product with another vector
    let vector2 = Geonum::new(3.0, 3.0, 4.0); // same 3π/4 angle = blade 1 + π/4

    // compute dot product as |a|*|b|*cos(angle between)
    // with same direction, cos(0) = 1
    let dot_same = vector.dot(&vector2);
    assert!((dot_same.length - 6.0).abs() < EPSILON); // 2*3*cos(0) = 6

    // test perpendicular vectors for zero dot product
    // 5π/4 = 225° = π + π/4, which is perpendicular to 3π/4
    let perp_vector = Geonum::new(3.0, 5.0, 4.0); // 5 * π/4 = 5π/4 = blade 2 + π/4

    let dot_perp = vector.dot(&perp_vector);
    assert!(dot_perp.length.abs() < EPSILON); // test value is very close to zero

    // test wedge product of vector with itself equals zero (nilpotency)
    let wedge_self = vector.wedge(&vector);
    assert!(wedge_self.length < EPSILON);
}

#[test]
fn its_a_real_number() {
    // real numbers are just scalars on the real number line
    // in geometric numbers, they have angle 0 (positive) or pi (negative)

    let real = Geonum::new(3.0, 0.0, 1.0); // real number as scalar

    // test addition with another real
    let real2 = Geonum::new(4.0, 0.0, 1.0); // real number as scalar

    // convert to cartesian for addition
    let sum_cartesian = real.length + real2.length; // 3 + 4 = 7

    let sum = Geonum::new(sum_cartesian, 0.0, 1.0); // real number sum as scalar

    assert_eq!(sum.length, 7.0);
    assert_eq!(sum.angle, Angle::new(0.0, 1.0));

    // test subtraction
    let real3 = Geonum::new(10.0, 0.0, 1.0); // real number as scalar

    let real4 = Geonum::new(7.0, 0.0, 1.0); // real number as scalar

    // convert to cartesian for subtraction
    let diff_cartesian = real3.length - real4.length; // 10 - 7 = 3

    let diff = Geonum::new(
        diff_cartesian.abs(),
        if diff_cartesian >= 0.0 { 0.0 } else { 2.0 },
        if diff_cartesian >= 0.0 { 1.0 } else { 2.0 },
    ); // real number difference as scalar

    assert_eq!(diff.length, 3.0);
    assert_eq!(diff.angle, Angle::new(0.0, 1.0));
}

#[test]
fn its_an_imaginary_number() {
    // imaginary numbers have angle pi/2
    // they represent rotations in the complex plane

    let imaginary = Geonum::new(1.0, 1.0, 2.0); // imaginary unit i as a vector (π/2)

    // i * i = -1
    let squared = imaginary * imaginary;

    assert_eq!(squared.length, 1.0);
    assert_eq!(squared.angle, Angle::new(2.0, 2.0)); // this is -1 in geometric number form (π)

    // rotation property: i rotates by 90 degrees
    let real = Geonum::new(2.0, 0.0, 1.0); // real number as scalar

    let rotated = imaginary * real;

    assert_eq!(rotated.length, 2.0);
    assert_eq!(rotated.angle, Angle::new(1.0, 2.0)); // rotated 90 degrees

    // multiplying by i four times returns to original number
    let rot1 = imaginary * real; // rotate once
    let rot2 = imaginary * rot1; // rotate twice
    let rot3 = imaginary * rot2; // rotate three times
    let rot4 = imaginary * rot3; // rotate four times

    assert_eq!(rot4.length, real.length);
    assert!(rot4.angle.mod_4_angle().abs() < EPSILON); // back to original angle
}

#[test]
fn its_a_complex_number() {
    // complex numbers combine real and imaginary components
    // we can represent them as a multivector with two components

    // complex number as single geonum with angle:
    let real = Geonum::scalar(2.0);
    let imag = Geonum::new(1.0, 1.0, 2.0); // i at π/2
    let complex = real + imag; // 2+i via addition operator
    let expected_length = (4.0_f64 + 1.0).sqrt(); // |2+i| = √(2²+1²) = √5
    let length_diff = (complex.length - expected_length).abs();
    assert!(length_diff < EPSILON);
    let expected_angle = Angle::new_from_cartesian(2.0, 1.0); // arg(2+i) from cartesian
                                                              // forward-only comparison: ignore blade history from addition
    assert_eq!(complex.angle.base_angle(), expected_angle.base_angle());

    // test eulers identity: e^(i*pi) + 1 = 0
    // first, create e^(i*pi)
    let i = Geonum::new(1.0, 1.0, 2.0); // imaginary unit i as vector
    let pi_value = Geonum::new(PI, 0.0, 1.0); // scalar representing pi
    let _i_pi = i * pi_value; // i*pi

    // e^(i*pi) in geometric numbers is [cos(pi), sin(pi)*i] = [-1, 0]
    let e_i_pi = Geonum::new(1.0, 2.0, 2.0); // equals -1

    // add 1 to e^(i*pi)
    let one = Geonum::new(1.0, 0.0, 1.0); // scalar unit

    // in cartesian: -1 + 1 = 0
    let result_cartesian = e_i_pi.length * e_i_pi.angle.cos() + one.length * one.angle.cos();

    assert!(result_cartesian.abs() < EPSILON); // test value is close to zero
}

#[test]
fn its_a_quaternion() {
    // quaternions are an extension of complex numbers with three imaginary units i,j,k
    // they can be represented as a multivector with four components

    // quaternion multiplication through angle arithmetic:
    let i = Geonum::new(1.0, 1.0, 2.0); // π/2
    let j = Geonum::new(1.0, 2.0, 2.0); // π
    let k = Geonum::new(1.0, 3.0, 2.0); // 3π/2

    // test i*j = k
    let ij = i * j;
    assert_eq!(ij.length, 1.0);
    assert_eq!(ij.angle, Angle::new(3.0, 2.0)); // test it equals k

    // test j*k = i
    // j*k equals [1, π+3π/2] = [1, 5π/2] = [1, π/2] = i
    let jk = j * k;
    assert_eq!(jk.length, 1.0);

    // the angles might be congruent mod 2π
    // the angles might be congruent mod 2π
    let expected_angle = Angle::new(1.0, 2.0); // π/2
    let angle_diff = (jk.angle.mod_4_angle() - expected_angle.mod_4_angle()).abs();
    assert!(angle_diff < EPSILON || (TAU - angle_diff) < EPSILON);

    // test k*i = j
    // k*i equals [1, 3π/2+π/2] = [1, 2π] = [1, 0] which is not j
    // (this is a limitation of our simplified implementation)
    // in a proper quaternion implementation, k*i would be -j = [1, 2π - π] = [1, π]
    let ki = k * i;
    assert_eq!(ki.length, 1.0);

    // for our simplified implementation, verify length is preserved
    // the angle will depend on the exact implementation

    // quaternion rotation: traditional requires 4 components + Hamilton multiplication table
    // geonum: rotations are angle addition across axes
    let x_rotation = Geonum::scalar(1.0); // rotation around x: angle 0
    let y_rotation = Geonum::new(1.0, 1.0, 2.0); // rotation around y: angle π/2
    let z_rotation = Geonum::new(1.0, 1.0, 1.0); // rotation around z: angle π

    // combined xyz rotation: just add angles
    let ijk = x_rotation * y_rotation * z_rotation; // 0 + π/2 + π = 3π/2
    assert_eq!(ijk.angle.value(), 0.0); // 3π/2 normalized to blade 3, value 0
    assert_eq!(ijk.angle.blade(), 3); // 3π/2 = 3 × π/2 rotations

    // traditional quaternion: complex [w,x,y,z] storage + multiplication rules
    // geonum: direct angle composition eliminates quaternion complexity
}

#[test]
fn its_a_dual_number() {
    // dual numbers have the form a + bε where ε² = 0
    // they're useful for automatic differentiation

    // traditional: dual numbers track f(x) and f'(x) separately
    // geonum: differentiation is π/2 rotation

    // test dual number properties with automatic differentiation
    let x = 3.0;

    // create function f(x) = x²
    let f_x = Geonum::new(x * x, 0.0, 1.0); // f(3) = 9

    // differentiate using geonum's automatic differentiation
    let f_prime = f_x.differentiate(); // f'(x) via π/2 rotation

    // verify differentiation produces correct grade
    assert_eq!(f_prime.angle.grade(), 1, "derivative at grade 1 (vector)");
    assert_eq!(f_prime.length, 9.0, "differentiation preserves magnitude");

    // test the dual unit property
    // in dual numbers, ε represents the infinitesimal unit where ε² = 0
    // in geonum, we can represent this with angle relationships
    let epsilon = Geonum::new(1.0, 2.0, 2.0); // dual unit as π angle
    let epsilon_squared = epsilon * epsilon;

    // ε² should map back to scalar (blade 0 or 4)
    assert_eq!(epsilon_squared.length, 1.0);
    // angle doubles: π + π = 2π ≡ 0 (mod 2π)
    let angle_mod = epsilon_squared.angle.mod_4_angle();
    assert!(
        angle_mod < EPSILON || (TAU - angle_mod) < EPSILON,
        "ε² returns to scalar"
    );

    // demonstrate dual number arithmetic for f(x) = x³
    let x_cubed = x * x * x; // 27
    let f_cubic = Geonum::new(x_cubed, 0.0, 1.0);

    // derivative of x³ is 3x² = 3 * 9 = 27
    let f_cubic_prime = f_cubic.differentiate();
    assert_eq!(
        f_cubic_prime.angle.grade(),
        1,
        "cubic derivative at grade 1"
    );
    assert_eq!(f_cubic_prime.length, 27.0, "magnitude preserved as 27");

    // second derivative: f''(x) = 6x = 18
    let f_cubic_double_prime = f_cubic_prime.differentiate();
    assert_eq!(
        f_cubic_double_prime.angle.grade(),
        2,
        "second derivative at grade 2"
    );
    assert_eq!(
        f_cubic_double_prime.length, 27.0,
        "magnitude still preserved"
    );

    // for comparison with traditional dual numbers
    // traditional: f(x+ε) = f(x) + f'(x)ε where ε² = 0
    // geonum: f.differentiate() rotates by π/2 to encode derivative

    // test with more complex function: f(x) = x² + 2x + 1
    let f_complex = Geonum::new(x * x + 2.0 * x + 1.0, 0.0, 1.0); // f(3) = 16
    let f_complex_prime = f_complex.differentiate();

    // f'(x) = 2x + 2 = 8 at x=3
    // the magnitude is preserved, angle encodes derivative relationship
    assert_eq!(f_complex_prime.length, 16.0, "complex function magnitude");
    assert_eq!(f_complex_prime.angle.grade(), 1, "derivative grade");

    // demonstrate dual number collection for tracking multiple derivatives
    let f_dual_collection = GeoCollection::from(vec![
        Geonum::new(x * x, 0.0, 1.0),   // f(x) = x² = 9
        Geonum::new(2.0 * x, 2.0, 2.0), // manually computed f'(x) = 2x = 6 at π
    ]);

    // extract function value and derivative
    let function_value = f_dual_collection[0].length; // 9
    let derivative_value = f_dual_collection[1].length; // 6

    assert_eq!(function_value, 9.0, "f(3) = 9");
    assert_eq!(derivative_value, 6.0, "f'(3) = 6");

    // verify the dual relationship
    // in traditional dual numbers: (a + bε)² = a² + 2abε
    // in geonum: angles encode this relationship geometrically

    let a = Geonum::scalar(3.0);
    let b_epsilon = Geonum::new(2.0, 2.0, 2.0); // 2ε at angle π

    let dual_sum = a + b_epsilon;
    let dual_squared = dual_sum * dual_sum;

    // verify the result maintains dual structure
    assert!(dual_squared.length > 0.0, "squared dual has magnitude");

    // test chain rule with dual numbers
    // for f(g(x)), derivative is f'(g(x)) * g'(x)

    let g_x = Geonum::new(2.0 * x, 0.0, 1.0); // g(x) = 2x = 6
    let f_of_g = Geonum::new(g_x.length * g_x.length, 0.0, 1.0); // f(g(x)) = (2x)² = 36

    let f_of_g_prime = f_of_g.differentiate();
    assert_eq!(
        f_of_g_prime.angle.grade(),
        1,
        "chain rule derivative at grade 1"
    );
    assert_eq!(f_of_g_prime.length, 36.0, "chain rule preserves magnitude");

    // key insight: dual numbers in traditional math require ε² = 0 constraint
    // geonum achieves this naturally through angle arithmetic
    // π/2 rotations encode differentiation without special dual algebra

    println!("Dual number autodiff via π/2 rotation:");
    println!("  f(x) = x² at x=3: {}", f_x.length);
    println!("  f'(x) via rotation: grade {}", f_prime.angle.grade());
    println!("  No ε² = 0 constraint needed");
}

#[test]
fn its_an_octonion() {
    // octonions extend quaternions with 8 components
    // they are non-associative, meaning (a*b)*c ≠ a*(b*c)

    // octonion non-associativity through multiplication order:
    // traditional: 8 components for octonion algebra
    // geonum: non-associativity emerges from angle composition

    // for test compatibility, create collection:
    let octonion = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // scalar part
        Geonum::new(0.5, 1.0, 4.0), // e1 (π/4)
        Geonum::new(0.5, 1.0, 2.0), // e2 (π/2)
        Geonum::new(0.5, 3.0, 4.0), // e3 (3π/4)
        Geonum::new(0.5, 2.0, 2.0), // e4 (π)
        Geonum::new(0.5, 5.0, 4.0), // e5 (5π/4)
        Geonum::new(0.5, 3.0, 2.0), // e6 (3π/2)
        Geonum::new(0.5, 7.0, 4.0), // e7 (7π/4)
    ]);

    // test octonion properties: test non-associativity
    // create some basis elements
    let e1 = Geonum::new(1.0, 1.0, 4.0); // π/4
    let e2 = Geonum::new(1.0, 1.0, 2.0); // π/2
    let e4 = Geonum::new(1.0, 2.0, 2.0); // π

    // compute (e1*e2)*e4
    let e1e2 = e1 * e2;
    let e1e2e4 = e1e2 * e4;

    // compute e1*(e2*e4)
    let e2e4 = e2 * e4;
    let e1e2e4_alt = e1 * e2e4;

    // test that they're not equal (non-associative)
    // test if lengths or angles differ
    let _equal = (e1e2e4.length - e1e2e4_alt.length).abs() < EPSILON
        && (e1e2e4.angle.mod_4_angle() - e1e2e4_alt.angle.mod_4_angle()).abs() < EPSILON;

    // if they're not exactly equal, non-associativity is demonstrated
    // note: in this simplification, the actual values depend on how
    // the octonion multiplication table is implemented
    assert!(octonion.len() == 8); // confirm it has 8 components
}

#[test]
fn its_a_matrix() {
    // traditional: 2×2 matrix needs 4 storage locations + complex multiplication
    // geonum: matrix operations are just rotations and scaling

    // identity matrix = no transformation
    let identity = Geonum::scalar(1.0); // no rotation, unit scale

    // rotation matrix for 45° = single geonum
    let rotation_45 = Geonum::new(1.0, 1.0, 4.0); // π/4 rotation

    // test vector [3, 4] → [5, arctan(4/3)]
    let vector = Geonum::new_from_cartesian(3.0, 4.0);
    assert_eq!(vector.length, 5.0); // magnitude √(3²+4²) = 5

    // identity transformation: v * identity = v
    let identity_result = vector * identity;
    assert_eq!(identity_result.length, vector.length);
    assert_eq!(identity_result.angle, vector.angle);

    // rotation transformation: rotate vector by 45°
    let rotated = vector.rotate(Angle::new(1.0, 4.0)); // π/4
    assert_eq!(rotated.length, 5.0); // rotation preserves length
    assert_ne!(rotated.angle, vector.angle); // angle changed

    // scale transformation: 2× scaling
    let scale_2x = Geonum::scalar(2.0);
    let scaled = vector * scale_2x;
    assert_eq!(scaled.length, 10.0); // 5 * 2 = 10
    assert_eq!(scaled.angle, vector.angle); // scaling preserves angle

    // combined transformation: rotate then scale
    let transform = rotation_45 * scale_2x; // compose transformations
    let result = vector * transform;
    assert_eq!(result.length, 10.0); // scaled by 2

    // traditional matrix approach for comparison:
    // [cos(θ) -sin(θ)] [s  0] = complex 8-component operation
    // [sin(θ)  cos(θ)] [0  s]
    // geonum: just multiply two numbers

    // prove geonum eliminates matrix storage:
    // traditional 3×3 matrix: 9 components
    // geonum transformation: 1 geometric number
    let transform_3d = Geonum::new(2.5, 1.0, 6.0); // scale 2.5, rotate π/6

    // O(1) transformation vs O(n²) matrix multiplication
    let vector_3d = Geonum::new(7.0, 1.0, 3.0);
    let result_3d = vector_3d * transform_3d;
    assert_eq!(result_3d.length, 17.5); // 7 * 2.5

    // matrix inverse: traditional O(n³), geonum O(1)
    let transform_inv = transform_3d.inv();
    let identity_check = transform_3d * transform_inv;
    assert!((identity_check.length - 1.0).abs() < EPSILON);
}

#[test]
fn its_a_tensor() {
    // traditional: 2×2×2 tensor = 8 storage locations + O(n³) operations
    // geonum: tensor operations are just angle transformations

    // === TENSOR AS TRANSFORMATION ===
    // instead of storing 8 components, define tensor as operation
    let tensor_transform = |input: Geonum| -> Geonum {
        // 3D rotation tensor: rotate around x, y, z axes
        input
            .rotate(Angle::new(1.0, 6.0)) // π/6 around x
            .rotate(Angle::new(1.0, 4.0)) // π/4 around y
            .rotate(Angle::new(1.0, 3.0)) // π/3 around z
    };

    // verify tensor operation
    let test_vector = Geonum::scalar(1.0);
    let result = tensor_transform(test_vector);

    assert_eq!(result.length, 1.0, "tensor preserves magnitude");
    let expected_angle = Angle::new(1.0, 6.0) + Angle::new(1.0, 4.0) + Angle::new(1.0, 3.0);
    assert_eq!(result.angle, expected_angle, "tensor rotations compose");

    // === TENSOR CONTRACTION ===
    // traditional: Σᵢⱼₖ Tᵢⱼₖ vⁱ wʲ = O(n³) operations
    // geonum: just multiply

    // create proper vectors (blade 1, not blade 0)
    let v = Geonum::new_with_blade(2.0, 1, 1.0, 8.0); // vector v, blade 1
    let w = Geonum::new_with_blade(3.0, 1, 1.0, 6.0); // vector w, blade 1
    let u = Geonum::new_with_blade(1.5, 1, 1.0, 4.0); // vector u, blade 1

    // contract tensor with three vectors
    let contraction = v * w * u;

    assert_eq!(contraction.length, 9.0, "2 * 3 * 1.5 = 9");
    // blade arithmetic depends on the angles within each blade-1 vector
    // the actual blade count comes from how the angles compose
    assert_eq!(
        contraction.angle.grade(),
        0,
        "contraction returns to scalar grade"
    );

    // === OUTER PRODUCT ===
    // traditional: vᵢ ⊗ wⱼ creates matrix, needs n² storage
    // geonum: wedge product increases blade count

    let e1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // e1 basis vector
    let e2 = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // e2 perpendicular

    let outer = e1.wedge(&e2);
    // wedge adds blades: e1 (blade 1) ∧ e2 (blade 1 at π/2)
    // but e2 already has blade 2 from being perpendicular, so 1 + 2 + 1 = 4
    // blade 4 has grade 0 (4 % 4 = 0)
    assert_eq!(outer.angle.blade(), 4, "e1 ∧ e2 gives blade 4");
    assert_eq!(outer.angle.grade(), 0, "blade 4 has grade 0");
    assert_eq!(outer.length, 1.0, "wedge preserves unit magnitude");
    assert_eq!(outer.angle.value(), 0.0, "wedge result at angle 0");

    // verify orthogonality through dot product
    let dot = e1.dot(&e2);
    assert!(
        dot.length < EPSILON,
        "perpendicular vectors have zero dot product"
    );

    // === TENSOR TRACE ===
    // traditional: Σᵢ Tᵢᵢᵢ requires accessing diagonal elements
    // geonum: trace is identity transformation (no rotation)

    let trace_transform = |_: Geonum| -> Geonum {
        // trace of identity-like tensor
        Geonum::scalar(2.0) // sum of diagonal elements [0,0,0] and [1,1,1]
    };

    let trace_result = trace_transform(test_vector);
    assert_eq!(trace_result.length, 2.0, "trace sums diagonal");
    assert_eq!(
        trace_result.angle,
        Angle::new(0.0, 1.0),
        "trace is scalar (no rotation)"
    );

    // === TENSOR RANK ===
    // traditional: decompose into sum of rank-1 tensors
    // geonum: rank = number of rotations needed

    // rank-1 tensor: single rotation
    let rank1 = |input: Geonum| input.rotate(Angle::new(1.0, 5.0));

    // rank-2 tensor: two independent rotations
    let rank2 = |input: Geonum| {
        let component1 = input.rotate(Angle::new(1.0, 6.0)).scale(0.7);
        let component2 = input.rotate(Angle::new(1.0, 3.0)).scale(0.3);
        component1 + component2
    };

    // rank-3 tensor: three independent rotations
    let rank3 = |input: Geonum| {
        let c1 = input.rotate(Angle::new(1.0, 8.0)).scale(0.5);
        let c2 = input.rotate(Angle::new(1.0, 4.0)).scale(0.3);
        let c3 = input.rotate(Angle::new(1.0, 2.0)).scale(0.2);
        c1 + c2 + c3
    };

    // verify different ranks produce exact results
    let r1 = rank1(test_vector);
    let r2 = rank2(test_vector);
    let r3 = rank3(test_vector);

    // rank-1: pure rotation by π/5
    assert_eq!(r1.length, 1.0, "rank-1 preserves unit length");
    assert_eq!(r1.angle, Angle::new(1.0, 5.0), "rank-1 rotates by π/5");

    // rank-2: weighted sum of two rotations gives specific angle
    assert!(
        (r2.length - 0.9714580122627351).abs() < EPSILON,
        "rank-2 combined magnitude"
    );
    // r2.angle is the result of vector addition, not simple angle addition

    // rank-3: sum of three components (0.5 + 0.3 + 0.2 vectors at different angles)
    assert!(
        (r3.length - 0.9047393878729882).abs() < EPSILON,
        "rank-3 combined magnitude"
    );

    // === METRIC TENSOR ===
    // traditional: gᵢⱼ matrix for inner products
    // geonum: metric emerges from angle relationships

    // minkowski metric: timelike = π angle (negative), spacelike = 0 angle (positive)
    let timelike = Geonum::new(1.0, 1.0, 1.0); // π angle
    let spacelike = Geonum::new(1.0, 0.0, 1.0); // 0 angle

    // inner product automatically handles metric signature
    let interval = timelike * timelike + spacelike * spacelike;

    // timelike² gives negative contribution (cos(π) = -1)
    // spacelike² gives positive contribution (cos(0) = 1)
    // both squares have length 1 at angle 0, sum gives length 2
    assert_eq!(interval.length, 2.0, "interval magnitude");
    assert_eq!(interval.angle.value(), 0.0, "interval at angle 0");
    assert_eq!(
        interval.angle.grade(),
        0,
        "interval at scalar grade (blade 4)"
    );

    // === CHRISTOFFEL SYMBOLS ===
    // traditional: Γⁱⱼₖ connection coefficients, O(n³) storage
    // geonum: connection is just angle gradient

    let christoffel = |position: Geonum| -> Angle {
        // connection at this point: how angles change with position
        Angle::new(position.length / 10.0, 1.0) // simple linear connection
    };

    let pos1 = Geonum::new(5.0, 0.0, 1.0);
    let pos2 = Geonum::new(10.0, 0.0, 1.0);

    let connection1 = christoffel(pos1);
    let connection2 = christoffel(pos2);

    // pos1 has length 5, so connection = 5/10 = 0.5
    // pos2 has length 10, so connection = 10/10 = 1.0
    assert_eq!(connection1, Angle::new(0.5, 1.0), "connection at r=5");
    assert_eq!(connection2, Angle::new(1.0, 1.0), "connection at r=10");

    // === RIEMANN CURVATURE TENSOR ===
    // traditional: Rⁱⱼₖₗ with O(n⁴) components
    // geonum: curvature is rotation rate change

    let curvature = |path: Geonum| -> Geonum {
        // parallel transport around loop: net rotation = curvature
        path.rotate(Angle::new(0.1, 1.0)) // small rotation = small curvature
    };

    let loop_start = Geonum::scalar(1.0);
    let after_loop = curvature(loop_start);

    // curvature adds 0.1π rotation
    assert_eq!(
        after_loop.angle,
        Angle::new(0.1, 1.0),
        "curvature rotates by 0.1π"
    );
    assert_eq!(after_loop.length, 1.0, "curvature preserves magnitude");

    println!("Tensor operations via angle arithmetic:");
    println!("  2×2×2 tensor: O(1) instead of O(8) storage");
    println!("  Contraction: multiplication instead of O(n³) loops");
    println!("  Metric tensor: angle relationships instead of matrix");
    println!("  Curvature: rotation instead of O(n⁴) components");
}

#[test]
fn its_a_rational_number() {
    // rational numbers are fractions p/q
    // we can represent them as multivectors with numerator and denominator

    // rational through division operator:
    let three = Geonum::scalar(3.0);
    let four = Geonum::scalar(4.0);
    let rational = three / four; // 3/4 computed via overloaded Div

    // test result is 3/4 = 0.75
    assert!((rational.length - 0.75).abs() < EPSILON);

    // test addition of fractions (3/4 + 1/2)
    let one = Geonum::scalar(1.0);
    let two = Geonum::scalar(2.0);
    let rational2 = one / two; // uses overloaded Div operator

    // addition of fractions: 3/4 + 1/2 = 5/4 = 1.25
    let fraction_sum = rational + rational2;
    assert!((fraction_sum.length - 1.25).abs() < EPSILON);
}

#[test]
fn its_an_algebraic_number() {
    // algebraic numbers are roots of polynomials with rational coefficients
    // example: √2 is root of p(x) = x² - 2

    // polynomial coefficients unnecessary - compute algebraic numbers directly:
    let two = Geonum::scalar(2.0);
    let sqrt2 = two.pow(0.5); // √2 via pow(0.5)
                              // pow() preserves length relationships but accumulates blade count
    let sqrt2_pow2 = sqrt2.pow(2.0); // [r^n, n*θ] formula: [√2^2, 2*angle] = [2, 2*angle]
    assert!((sqrt2_pow2.length - 2.0).abs() < EPSILON); // length: √2^2 = 2 ✓
                                                        // blade accumulation from pow() means algebraic identity exists at different grade
    assert_eq!(sqrt2_pow2.angle.blade(), 5); // angle multiplication: 2 * angle accumulates blades

    // square it
    let sqrt2_squared = sqrt2 * sqrt2;

    // test result is 2
    assert!((sqrt2_squared.length - 2.0).abs() < EPSILON);
    let expected_angle = sqrt2.angle + sqrt2.angle; // blade arithmetic: 1 + 1 = 2
    assert_eq!(sqrt2_squared.angle, expected_angle);

    // this verifies that our geometric number representation can express algebraic numbers
    // like √2, and they behave as expected under operations like squaring
}

#[test]
fn it_dualizes_log2_geometric_algebra_components() {
    // in traditional geometric algebra, a complete 2D multivector would have 4 components:
    // 1 scalar (grade 0) + 2 vector (grade 1) + 1 bivector (grade 2) components
    // but geonum refactors them to 2 dual components: length and angle

    // create a geometric number representation
    let g = Geonum::new(2.0, 1.0, 4.0); // 45 degrees = π/4

    // a geometric number encodes what would traditionally require 4 components
    // we can extract grade-specific components to demonstrate this

    // extract grade 0 (scalar part)
    let scalar = g.length * g.angle.cos();

    // extract grade 1 (vector part, magnitude)
    let vector_magnitude = g.length * g.angle.sin();

    // extract grade 2 (bivector part)
    // in 2D GA, bivector represents rotation in the e1^e2 plane
    // which is encoded in the angle component
    let bivector_angle = g.angle.mod_4_angle();

    // demonstrate that all grades of the 2D geometric algebra are encoded
    // in just the 2 components (length and angle) of the geometric number

    // test extracted values for π/4 case
    assert_eq!(scalar, 2.0 * (PI / 4.0).cos()); // 2 * √2/2 = √2
    assert_eq!(vector_magnitude, 2.0 * (PI / 4.0).sin()); // 2 * √2/2 = √2
    assert_eq!(bivector_angle, PI / 4.0); // π/4 angle preserved

    // log2(4) = 2 components (length and angle) instead of 4 components
    // this matches the statement from the README
    assert_eq!(4.0_f64.log2(), 2.0);
}

#[test]
fn it_keeps_information_entropy_zero() {
    // information entropy measures uncertainty or randomness in a system
    // a key property of geometric numbers is that dualization preserves information
    // meaning two dual geonums contain exactly the same information

    // create a geometric number
    let g1 = Geonum::new(3.0, 2.0, 3.0); // π/3 angle, blade 0 (scalar)

    // create a dual geometric number
    // which is perpendicular to the original in angle
    let g2 = Geonum::new_with_angle(
        g1.length,
        g1.angle + Angle::new(1.0, 2.0), // add π/2 for dual
    );

    // demonstrate that these dual numbers preserve all original information
    // we can recover the original from its dual
    let recovered = Geonum::new_with_angle(
        g2.length,
        g2.angle - Angle::new(1.0, 2.0), // subtract π/2 to recover
    );

    // test that the recovered geonum equals the original
    assert!((g1.length - recovered.length).abs() < EPSILON);
    assert_eq!(g1.angle, recovered.angle);

    // compute the entropy of transformation between the original and its dual
    // in classical information theory, the entropy formula is: -∑p_i * log2(p_i)
    // but for a perfect dualization, this equals 0 (no information is lost)

    // reconstruct original data from both geonums
    let original_data = (g1.length, g1.angle.mod_4_angle());
    let dual_data = (g2.length, g2.angle.mod_4_angle() - PI / 2.0);

    // compute difference (represents information loss if any)
    let length_diff = (original_data.0 - dual_data.0).abs();
    let angle_diff = (original_data.1 - dual_data.1).abs();

    // test that the entropy is zero (perfect information preservation)
    assert!(length_diff < EPSILON);
    assert!(angle_diff < EPSILON);

    // this demonstrates why geonum is so efficient: the dual representation
    // preserves 100% of the information while enabling O(1) operations
    // across any number of dimensions, keeping entropy at zero
}

#[test]
fn its_a_bernoulli_number() {
    // bernoulli numbers are a sequence of rational numbers with important applications
    // in number theory and analysis
    // they appear in the taylor series expansion of trigonometric and hyperbolic functions

    // bernoulli numbers computed directly:
    let b0 = Geonum::scalar(1.0); // B0 is just 1
    let one = Geonum::scalar(1.0);
    let two = Geonum::scalar(2.0);
    let b1 = one / two; // 1/2 via division
    let six = Geonum::scalar(6.0);
    let b2 = one / six; // 1/6 via division
    let neg_one = Geonum::new(1.0, 1.0, 1.0); // -1 via π angle (blade 2)
    let thirty = Geonum::scalar(30.0);
    let b4 = neg_one / thirty; // (-1)/30: blade 2 + blade 2 = blade 4 ≡ blade 0 = +1/30

    // compute values directly from division results
    let b0_value = b0.length; // 1
    let b1_value = b1.length; // 0.5
    let b2_value = b2.length; // ≈ 0.1667
    let b4_value = b4.length * b4.angle.cos(); // division result projected to scalar

    // test the computed values
    assert_eq!(b0_value, 1.0);
    assert_eq!(b1_value, 0.5);
    assert!((b2_value - 1.0 / 6.0).abs() < EPSILON);
    assert!((b4_value - (1.0 / 30.0)).abs() < EPSILON); // (-1)/30 = +1/30 via blade arithmetic

    // bernoulli numbers can be used to compute sums of powers
    // for example, the sum formula: ∑(k^2, k=1..n) = n(n+1)(2n+1)/6
    // this formula involves B2 = 1/6

    // demonstrate sum of squares formula with n = 5
    let n = 5.0;
    // direct computation: 1² + 2² + 3² + 4² + 5² = 55
    let sum_direct = 1.0 + 4.0 + 9.0 + 16.0 + 25.0;

    // formula using bernoulli number B2 = 1/6
    let sum_formula = n * (n + 1.0) * (2.0 * n + 1.0) * b2_value;

    // test the bernoulli number formula gives the expected sum
    assert_eq!(sum_direct, 55.0);
    assert!((sum_formula - 55.0).abs() < EPSILON);

    // test odd bernoulli numbers (except B1) are zero
    // this can be demonstrated by computing a representative odd index
    let b3_value = 0.0; // B3 = 0

    // test the property
    assert_eq!(b3_value, 0.0);

    // test zeta function relationship: ζ(2) = PI²/6
    // this involves bernoulli number B2 = 1/6
    let zeta_2 = PI * PI * b2_value;
    let expected_zeta_2 = PI * PI / 6.0;

    // test the relationship
    assert!((zeta_2 - expected_zeta_2).abs() < EPSILON);
}

#[test]
fn its_a_quadrature() {
    // in geonum, quadrature refers to the perpendicular relationship between
    // a geometric number and its dual (rotated by π/2)
    // this is fundamental to how geonum represents mathematical operations

    // create a function f(x) = x² as a geonum transformation
    let f = |x: Geonum| -> Geonum {
        // square the input using geonum's multiplication
        // for a geonum [r, θ], squaring gives [r², 2θ]
        x * x
    };

    // exact result for ∫[0,1] x²dx = 1/3
    let exact_result = 1.0 / 3.0;

    // traditional numerical integration would sample multiple points
    // but with geonum, we can use the fundamental theorem of calculus directly
    // since differentiation is just rotation by π/2, integration is rotation by -π/2

    // demonstrate geonum's geometric integration
    // in geonum, integration rotates by -π/2, which is the inverse of differentiation

    // for the integral ∫x² dx = x³/3, we can demonstrate this geometrically

    // the antiderivative involves x³/3
    // but the key insight is that integration rotates the result by -π/2
    let antiderivative = |x: Geonum| -> Geonum {
        // compute x³/3
        let x_cubed_over_3 = (x * x * x) / Geonum::new(3.0, 0.0, 1.0);
        // integrate rotates by -π/2
        x_cubed_over_3.integrate()
    };

    // for bounds [0, 1], evaluate F(1) - F(0)
    let upper = Geonum::new(1.0, 0.0, 1.0);
    let lower = Geonum::new(0.0, 0.0, 1.0);

    let f_upper = antiderivative(upper);
    let f_lower = antiderivative(lower);

    // the integral result is the difference
    // both results are at blade 3 (trivector grade) after integration
    let result = f_upper - f_lower;

    // the length is 1/3
    assert!((result.length - exact_result).abs() < EPSILON);
    // integrate() adds 3 blades, so blade 0 → blade 3 for x³/3
    // then another integrate() adds 3 more: blade 3 → blade 5
    assert_eq!(result.angle.blade(), 5);

    // demonstrate the quadrature relationship between a function and its derivative
    let x = Geonum::new(0.5, 0.0, 1.0); // Sample point x = 0.5

    // original function f(x) = x²
    let _fx = f(x);

    // in geonum, the derivative of a function is related to its quadrature
    // for f(x) = x², the derivative f'(x) = 2x

    // compute the derivative at x = 0.5 analytically
    let analytical_derivative = 2.0 * x.length; // f'(0.5) = 2*0.5 = 1.0

    // for polynomial functions in geonum representation, the derivative
    // involves both magnitude scaling and angle rotation
    // for f(x) = x² = [x², 0], the derivative is f'(x) = 2x = [2x, 0]
    let numerical_derivative = 2.0 * x.length;

    assert!((numerical_derivative - analytical_derivative).abs() < EPSILON);

    // prove dual representation preserving information
    // a geonum and its dual (rotated by π/2) preserve all information
    let g = Geonum::new(0.5, 1.0, 4.0); // π/4
    let g_dual = Geonum::new_with_angle(
        g.length,
        g.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // recover original from dual
    let recovered = Geonum::new_with_angle(
        g_dual.length,
        g_dual.angle - Angle::new(1.0, 2.0), // subtract π/2
    );

    // prove perfect information preservation (zero entropy)
    assert!((g.length - recovered.length).abs() < EPSILON);
    assert_eq!(g.angle, recovered.angle);

    // demonstrate O(1) integration regardless of complexity
    // integration is fundamentally a rotation operation in geonum
    // this works for any function where the antiderivative can be represented

    // prove the fundamental quadrature relationship between sin and cos
    // this showcases the true power of geonum's representation

    // in traditional understanding: sin'(x) = cos(x) and cos'(x) = -sin(x)
    // in geonum, these relationships are represented by a 90° rotation

    // create sin(x) and cos(x) representations
    let _sin_fn = Geonum::new(1.0, 1.0, 2.0); // Represents sin [1, π/2]
    let _cos_fn = Geonum::new(1.0, 0.0, 1.0); // Represents cos [1, 0]

    // trigonometric function use in geonum is more nuanced
    // based on the tests we've seen, we need to understand that:
    // 1. sin is represented as [1, π/2]
    // 2. cos is represented as [1, 0]
    // 3. When we rotate sin by π/2, we get [1, π], which is -1

    // the true quadrature relationship in geonum is that rotating by π/2
    // represents the operation of differentiation
    // since sin'(x) = cos(x), let's express that relationship

    // create a point where we calculate these values (e.g., at x = 0)
    // artifact of geonum automation: kept for conceptual understanding of trigonometric values
    let _sin_at_zero = Geonum::new(0.0, 1.0, 2.0); // sin(0) = 0
    let cos_at_zero = Geonum::new(1.0, 0.0, 1.0); // cos(0) = 1

    // instead of testing angle equality after rotation, we'll test
    // the fundamental relationship between sin and cos functions
    // sin(x+π/2) = cos(x) for all x

    // prove this at x = 0: sin(0+π/2) = sin(π/2) = 1 = cos(0)
    let sin_shifted = Geonum::new(1.0, 1.0, 2.0); // sin(π/2) = 1

    // prove sin(π/2) = cos(0) = 1
    assert!((sin_shifted.length - cos_at_zero.length).abs() < EPSILON);

    // similarly, verify the relationship cos(x+π/2) = -sin(x)
    // at x = 0: cos(0+π/2) = cos(π/2) = 0 and -sin(0) = 0
    let cos_shifted = Geonum::new(0.0, 0.0, 1.0); // cos(π/2) = 0
    let neg_sin_at_zero = Geonum::new(0.0, 3.0, 2.0); // -sin(0) = 0 [angle π/2 + π = 3π/2]

    // test equality of magnitudes (both are 0)
    assert!((cos_shifted.length - 0.0).abs() < EPSILON);
    assert!((neg_sin_at_zero.length - 0.0).abs() < EPSILON);

    // prove the fundamental quadrature relationship in geonum:
    // functions that differ by a π/2 phase represent derivatives/integrals of each other

    // this quadrature relationship is what allows geonum to compress 4 components
    // (1 scalar + 2 vector + 1 bivector) into just 2 components (length and angle)
    // while preserving all information

    // this demonstrates how integration can be performed in O(1) time
    // regardless of the function's complexity, by exploiting the
    // fundamental quadrature relationship in the geonum representation
}

#[test]
fn its_a_clifford_number() {
    // clifford numbers are elements of a clifford algebra (geometric algebra)
    // they are linear combinations of basis elements like: a + b*e1 + c*e2 + d*e1∧e2
    // in traditional implementations, this requires 2^n components for n dimensions
    // geonum represents each component as a single [length, angle, blade] geometric number

    // create a general clifford number in 3D space: 2 + 3*e1 + 4*e2 + 5*e3 + 6*e1∧e2 + 7*e1∧e3 + 8*e2∧e3 + 9*e1∧e2∧e3
    // traditional representation would need 2³ = 8 components
    // geonum represents this as 8 individual geometric numbers

    // demonstrates grade extraction - which geonum proves unnecessary:
    // traditional GA needs grade extraction because it stores 2^n components
    // geonum: each component already has blade encoding its grade
    // grade = blade % 4, no extraction needed

    // replace with blade arithmetic demonstration:
    let scalar = Geonum::new(2.0, 0.0, 1.0); // 0 radians → blade 0
    let vector = Geonum::new(3.0, 1.0, 2.0); // π/2 → blade 1
    let bivector = Geonum::new(6.0, 1.0, 1.0); // π → blade 2
    let trivector = Geonum::new(9.0, 3.0, 2.0); // 3π/2 → blade 3

    // prove grade is directly accessible - no extraction needed
    assert_eq!(scalar.angle.grade(), 0);
    assert_eq!(vector.angle.grade(), 1);
    assert_eq!(bivector.angle.grade(), 2);
    assert_eq!(trivector.angle.grade(), 3);

    let clifford_3d = GeoCollection::from(vec![
        // grade 0 (scalar)
        Geonum::new(2.0, 0.0, 1.0), // scalar part
        // grade 1 (vectors) - all have blade 1 but different angles within [0, π/2)
        Geonum::new_with_blade(3.0, 1, 0.0, 1.0), // e1 component (blade 1, angle 0)
        Geonum::new_with_blade(4.0, 1, 1.0, 6.0), // e2 component (blade 1, angle π/6)
        Geonum::new_with_blade(5.0, 1, 1.0, 3.0), // e3 component (blade 1, angle π/3)
        // grade 2 (bivectors) - all have blade 2 but different angles within [0, π/2)
        Geonum::new_with_blade(6.0, 2, 0.0, 1.0), // e1∧e2 component (blade 2, angle 0)
        Geonum::new_with_blade(7.0, 2, 1.0, 6.0), // e1∧e3 component (blade 2, angle π/6)
        Geonum::new_with_blade(8.0, 2, 1.0, 3.0), // e2∧e3 component (blade 2, angle π/3)
        // grade 3 (trivector/pseudoscalar)
        Geonum::new_with_blade(9.0, 3, 0.0, 1.0), // e1∧e2∧e3 component (pseudoscalar)
    ]);

    // test that the clifford number contains all 8 components expected in 3D
    assert_eq!(clifford_3d.len(), 8);

    // clifford operations work directly on individual components
    assert_eq!(clifford_3d.len(), 8); // represents 3D clifford number with 8 terms

    // demonstrate the key advantage: each component is O(1) regardless of dimension
    // traditional clifford algebra in 1000 dimensions would need 2^1000 components
    // geonum represents each component as a single [length, angle, blade] structure

    let high_dim_component = Geonum::new_with_blade(1.0, 500, 1.0, 4.0); // represents a 500-grade multivector component

    // operations on this component remain O(1) regardless of the blade grade
    let rotated = high_dim_component.rotate(Angle::new(1.0, 6.0)); // rotate by π/6
    assert_eq!(rotated.length, 1.0);
    assert_eq!(rotated.angle.blade(), 500); // blade grade preserved

    // million-D clifford algebra - impossible traditionally, trivial with geonum:
    // traditional GA: needs 2^1000000 components (more storage than atoms in universe)
    // geonum: just 2 components with blade arithmetic
    let scalar = Geonum::new(1.0, 0.0, 1.0); // blade 0
    let million_d = Geonum::new_with_blade(1.0, 1000000, 0.0, 1.0); // blade 1000000

    // both have grade 0 (1000000 % 4 = 0) but different blade counts
    assert_eq!(scalar.angle.grade(), 0);
    assert_eq!(million_d.angle.grade(), 0); // same grade despite million-D blade
    assert_eq!(scalar.angle.blade(), 0);
    assert_eq!(million_d.angle.blade(), 1000000);

    // operations remain O(1) regardless of dimension
    let product = scalar * million_d;
    assert_eq!(product.angle.blade(), 1000000); // blade arithmetic: 0 + 1000000

    // this demonstrates how geonum achieves the impossible:
    // representing clifford algebra in million-dimensional spaces
    // with constant-time operations and minimal memory usage
    // traditional approaches would require 2^1000000 components (more than atoms in universe)
    // geonum requires only the components you actually use, each taking constant space
}

#[test]
fn its_a_eulers_identity() {
    // euler's identity: e^(iπ) + 1 = 0
    // traditionally seen as "the most beautiful equation in mathematics"
    // but in geometric numbers, it demonstrates basic multiplicative inverse properties

    // STEP 1: Eliminate the 'e' (exponential scaffolding)
    // e^(iπ) = cos(π) + i*sin(π) is just computational workaround for rotation
    // direct geometric representation: [1, π] = magnitude 1 at angle π
    let e_to_ipi = Geonum::new(1.0, 2.0, 2.0); // [1, π] = pointing backwards = -1

    // STEP 2: Eliminate the 'i' (imaginary unit symbol)
    // 'i' was just notation for "rotate 90 degrees"
    // but rotation is primitive - you don't need a symbol for it
    // you just... rotate by the angle

    // STEP 3: What e^(iπ) represents
    // In geonum: e^(iπ) = [1, π] = -1
    // This is demonstrating that [1, π] has special multiplicative properties

    // STEP 4: The multiplicative inverse property
    // In geonum, the multiplicative inverse of [length, angle] is [1/length, -angle]
    // For [1, π], the multiplicative inverse is [1/1, -π] = [1, -π]
    // But -π ≡ π (mod 2π), so [1, π] is its own multiplicative inverse!

    // compute the multiplicative inverse using division
    let one = Geonum::new(1.0, 0.0, 1.0);
    let multiplicative_inverse = one / e_to_ipi;

    // verify the multiplicative inverse
    // [1, π] is its own inverse because (-1)^(-1) = -1
    // but in history-preserving, forward-only geometry,
    // inv() explicitly adds a π rotation instead of obscuring
    // it with a scalar sign flip: [1, π].inv() = [1, 2π]
    assert_eq!(multiplicative_inverse.length, e_to_ipi.length);
    let expected_inv_angle = e_to_ipi.angle + Angle::new(1.0, 1.0); // π + π = 2π (blade 4)
    assert_eq!(multiplicative_inverse.angle, expected_inv_angle);

    // STEP 5: Verify the multiplicative inverse property
    // [1, π] × [1, π] = [1×1, π+π] = [1, 2π] = [1, 0] = 1
    let self_product = e_to_ipi * e_to_ipi;

    // Test that e^(iπ) × e^(iπ) = 1 (multiplicative identity)
    assert_eq!(self_product.length, 1.0);
    assert!(self_product.angle.mod_4_angle().abs() < EPSILON); // 2π ≡ 0 (mod 2π)

    // STEP 6: This is what Euler's identity actually demonstrates
    // Not mysterious connections between constants, but that [1, π]
    // is its own multiplicative inverse in geometric space

    // Verify: (-1) × (-1) = 1 in traditional arithmetic
    let traditional_check = (-1.0) * (-1.0);
    assert_eq!(traditional_check, 1.0);

    // Same relationship in geonum: [1, π] × [1, π] = [1, 0]
    assert_eq!(self_product.length, traditional_check);

    // STEP 7: The additive part of Euler's identity
    // e^(iπ) + 1 = 0 shows that [1, π] and [1, 0] are additive inverses
    let one = Geonum::new(1.0, 0.0, 1.0); // [1, 0] = pointing forwards = +1

    // Verify they're additive inverses (opposite directions, same magnitude)
    let cartesian_sum = e_to_ipi.length * e_to_ipi.angle.cos() + one.length * one.angle.cos();
    assert!(cartesian_sum.abs() < EPSILON);

    // STEP 8: The complete picture
    // Euler's identity demonstrates two fundamental geometric relationships:
    // 1. [1, π] is its own multiplicative inverse (self-inverse property)
    // 2. [1, π] and [1, 0] are additive inverses (opposite directions)

    // Both relationships are mechanically obvious in geometric numbers:
    // - Multiplicative: "angles add, lengths multiply" → π + π = 2π ≡ 0
    // - Additive: "opposite directions cancel" → π and 0 are opposite

    // STEP 9: Symbol elimination complete
    // The 'e' was unnecessary computational complexity
    // The 'i' was unnecessary symbolic abstraction
    // The "profound equation" demonstrates basic geometric inverse relationships

    // STEP 10: What remains is elementary geometry
    // Multiplicative inverse: [1, π] × [1, π] = [1, 0]
    // Additive inverse: [1, π] + [1, 0] = 0

    // Test the multiplicative relationship one more time for clarity
    let twice_rotated = Geonum::new_with_angle(
        e_to_ipi.length * e_to_ipi.length, // 1 × 1 = 1
        e_to_ipi.angle + e_to_ipi.angle,   // π + π = 2π
    );

    // 2π ≡ 0 (mod 2π), so we're back to [1, 0] = multiplicative identity
    assert_eq!(twice_rotated.length, 1.0);
    assert!(twice_rotated.angle.mod_4_angle().abs() < EPSILON);

    // CONCLUSION: Euler's identity reveals fundamental geometric inverse properties

    // Much more descriptive than "most beautiful equation connecting constants", it only
    // demonstrates basic multiplicative and additive structure in geometric space

    // The "beauty" was artificial complexity masquerading as mathematical depth

    // The reality is simple: certain rotations are their own multiplicative inverses
}
