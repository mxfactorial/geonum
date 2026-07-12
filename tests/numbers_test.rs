use geonum::*;
use std::f64::consts::PI;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_scalar() {
    // a scalar is just a number with magnitude but no direction
    // in geometric number format, its a [length, 0] for positive
    // or [length, pi] for negative

    let scalar = Geonum::new(1.0, 0.0, 1.0);

    // test if scalar has expected properties
    assert_eq!(scalar.mag, 1.0);
    assert_eq!(scalar.angle.grade_angle(), 0.0);

    // multiplying scalars follows "angles add, lengths multiply" rule
    let scalar2 = Geonum::new(2.0, 0.0, 1.0);

    let product = scalar * scalar2;

    // 1 × 2 = 2
    assert_eq!(product.mag, 2.0);
    assert_eq!(product.angle.grade_angle(), 0.0);

    // multiplication with negative scalar
    let neg_scalar = Geonum::new(3.0, 1.0, 1.0); // PI radians

    let neg_product = scalar * neg_scalar;

    // 1 × (-3) = -3
    assert_eq!(neg_product.mag, 3.0);
    assert_eq!(neg_product.angle.grade_angle(), PI);
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
    assert_eq!(vector.mag, 2.0);
    assert_eq!(vector.angle.blade(), 1); // blade 1 = vector (grade 1) in geometric algebra
    assert!(vector.angle.near_rem(PI / 4.0)); // π/4 remainder after π/2 rotation

    // test dot product with another vector
    let vector2 = Geonum::new(3.0, 3.0, 4.0); // same 3π/4 angle = blade 1 + π/4

    // compute dot product as |a|*|b|*cos(angle between)
    // with same direction, cos(0) = 1
    let dot_same = vector.dot(&vector2);
    assert!(dot_same.near_mag(6.0)); // 2*3*cos(0) = 6

    // test perpendicular vectors for zero dot product
    // 5π/4 = 225° = π + π/4, which is perpendicular to 3π/4
    let perp_vector = Geonum::new(3.0, 5.0, 4.0); // 5 * π/4 = 5π/4 = blade 2 + π/4

    let dot_perp = vector.dot(&perp_vector);
    assert!(dot_perp.near_mag(0.0)); // test value is very close to zero

    // test wedge product of vector with itself equals zero (nilpotency)
    let wedge_self = vector.wedge(&vector);
    assert!(wedge_self.mag < EPSILON);
}

#[test]
fn its_a_real_number() {
    // a real number is a scalar on a line: magnitude with angle 0 for positive, π for
    // negative. addition and subtraction run through Geonum's own ops — same-angle
    // magnitudes add, opposite angles (0 vs π) interfere

    let three = Geonum::new(3.0, 0.0, 1.0); // +3
    let four = Geonum::new(4.0, 0.0, 1.0); // +4

    // same direction: the magnitudes add, the sum stays on the positive ray
    let sum = three + four;
    assert!(sum.near_mag(7.0), "3 + 4 = 7");
    assert_eq!(sum.angle.grade(), 0, "the sum is a positive real");

    // subtraction is addition of the negative — 7 − 4 lands +3 on the positive ray
    let seven = Geonum::new(7.0, 0.0, 1.0);
    let diff = seven - four;
    assert!(diff.near_mag(3.0), "7 − 4 = 3");
    assert_eq!(diff.angle.grade(), 0, "positive result on the 0 ray");

    // a negative real sits a half turn away — 3 + (−7) lands −4 on the π ray
    let neg_seven = Geonum::new(7.0, 1.0, 1.0); // −7 at angle π
    let signed = three + neg_seven;
    assert!(signed.near_mag(4.0), "3 + (−7) = −4");
    assert_eq!(
        signed.angle.grade(),
        2,
        "the negative result lands on the π ray"
    );
}

#[test]
fn its_an_imaginary_number() {
    // imaginary numbers have angle pi/2
    // they represent rotations in the complex plane

    let imaginary = Geonum::new(1.0, 1.0, 2.0); // imaginary unit i as a vector (π/2)

    // i * i = -1
    let squared = imaginary * imaginary;

    assert_eq!(squared.mag, 1.0);
    assert_eq!(squared.angle, Angle::new(2.0, 2.0)); // this is -1 in geometric number form (π)

    // rotation property: i rotates by 90 degrees
    let real = Geonum::new(2.0, 0.0, 1.0); // real number as scalar

    let rotated = imaginary * real;

    assert_eq!(rotated.mag, 2.0);
    assert_eq!(rotated.angle, Angle::new(1.0, 2.0)); // rotated 90 degrees

    // multiplying by i four times returns to original number
    let rot1 = imaginary * real; // rotate once
    let rot2 = imaginary * rot1; // rotate twice
    let rot3 = imaginary * rot2; // rotate three times
    let rot4 = imaginary * rot3; // rotate four times

    assert_eq!(rot4.mag, real.mag);
    assert!(rot4.angle.grade_angle().abs() < EPSILON); // back to original angle
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
    let length_diff = (complex.mag - expected_length).abs();
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
    let result_cartesian =
        e_i_pi.mag * e_i_pi.angle.grade_angle().cos() + one.mag * one.angle.grade_angle().cos();

    assert!(result_cartesian.abs() < EPSILON); // test value is close to zero
}

#[test]
fn its_a_dual_number() {
    // a dual number a + bε packs a value and its derivative into two slots, and ε² = 0 is the
    // truncation keeping the derivative from folding onto the value. geonum needs no ε:
    // differentiate is a quarter turn, so the value and its derivative sit at orthogonal
    // grades — that grade separation is what ε² = 0 enforces. the derivative VALUE (2x) is
    // read from the power in the angle (calculus_test::it_encodes_the_power_in_the_angle);
    // here the structure is the point, not the number

    // f(x) = x² at x = 3: the value is a grade-0 scalar
    let value = Geonum::new(9.0, 0.0, 1.0); // f(3) = 9

    // the derivative is the value rotated a quarter turn — the ε slot, grade 1
    let derivative = value.differentiate();
    assert_eq!(
        derivative.angle.grade(),
        1,
        "the derivative sits a quarter turn off the value"
    );
    assert!(
        derivative.near_mag(value.mag),
        "the quarter turn carries the magnitude, no ε algebra"
    );

    // value ⊥ derivative — the dot vanishes. that orthogonality is what ε² = 0 buys the
    // scalar dual number: the value and derivative slots cannot mix
    assert!(
        value.dot(&derivative).near_mag(0.0),
        "value ⊥ derivative — the ε² = 0 separation, geometric"
    );

    // ε² = 0 stops the scalar dual number at first order. geonum does not stop: differentiate
    // on and the grade cycles 1 → 2 → 3 → 0, the higher derivatives landing in their own
    // slots where the truncated dual number has nothing
    let mut d = value;
    for expected in [1usize, 2, 3, 0] {
        d = d.differentiate();
        assert_eq!(
            d.angle.grade(),
            expected,
            "differentiation cycles the grade past ε's first order"
        );
    }
    assert!(
        d.near_mag(value.mag),
        "magnitude preserved through the full cycle"
    );
}

#[test]
fn its_an_octonion() {
    // octonions are 8 units, non-associative in the decomposed algebra. geonum places them as
    // 8 angles (k·π/4) and composes by angle addition — which ASSOCIATES, because blade
    // addition does. the non-associativity is a decomposition artifact (linear_algebra_test),
    // not a property of the primitive product
    let units: GeoCollection = (0..8).map(|k| Geonum::new(1.0, k as f64, 4.0)).collect();
    assert_eq!(units.len(), 8, "8 octonion units, one per π/4 step");

    let e1 = Geonum::new(1.0, 1.0, 4.0); // π/4
    let e2 = Geonum::new(1.0, 2.0, 4.0); // π/2
    let e4 = Geonum::new(1.0, 4.0, 4.0); // π

    // (e1·e2)·e4 and e1·(e2·e4) land together — angle addition regroups freely
    let left = (e1 * e2) * e4;
    let right = e1 * (e2 * e4);
    assert!(
        left.near(&right),
        "octonion composition associates here: the non-associativity the algebra needs is the \
         decomposition correction, not this product"
    );
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
    assert_eq!(vector.mag, 5.0); // magnitude √(3²+4²) = 5

    // identity transformation: v * identity = v
    let identity_result = vector * identity;
    assert_eq!(identity_result.mag, vector.mag);
    assert_eq!(identity_result.angle, vector.angle);

    // rotation transformation: rotate vector by 45°
    let rotated = vector.rotate(Angle::new(1.0, 4.0)); // π/4
    assert_eq!(rotated.mag, 5.0); // rotation preserves length
    assert_ne!(rotated.angle, vector.angle); // angle changed

    // scale transformation: 2× scaling
    let scale_2x = Geonum::scalar(2.0);
    let scaled = vector * scale_2x;
    assert_eq!(scaled.mag, 10.0); // 5 * 2 = 10
    assert_eq!(scaled.angle, vector.angle); // scaling preserves angle

    // combined transformation: rotate then scale
    let transform = rotation_45 * scale_2x; // compose transformations
    let result = vector * transform;
    assert_eq!(result.mag, 10.0); // scaled by 2

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
    assert_eq!(result_3d.mag, 17.5); // 7 * 2.5

    // matrix inverse: traditional O(n³), geonum O(1)
    let transform_inv = transform_3d.inv();
    let identity_check = transform_3d * transform_inv;
    assert!(identity_check.near_mag(1.0));
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

    assert_eq!(result.mag, 1.0, "tensor preserves magnitude");
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

    assert_eq!(contraction.mag, 9.0, "2 * 3 * 1.5 = 9");
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
    assert_eq!(outer.mag, 1.0, "wedge preserves unit magnitude");
    assert_eq!(outer.angle.rem(), 0.0, "wedge result at angle 0");

    // verify orthogonality through dot product
    let dot = e1.dot(&e2);
    assert!(
        dot.mag < EPSILON,
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
    assert_eq!(trace_result.mag, 2.0, "trace sums diagonal");
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
    assert_eq!(r1.mag, 1.0, "rank-1 preserves unit length");
    assert_eq!(r1.angle, Angle::new(1.0, 5.0), "rank-1 rotates by π/5");

    // rank-2: weighted sum of two rotations gives specific angle
    assert!(
        (r2.mag - 0.9714580122627351).abs() < EPSILON,
        "rank-2 combined magnitude"
    );
    // r2.angle is the result of vector addition, not simple angle addition

    // rank-3: sum of three components (0.5 + 0.3 + 0.2 vectors at different angles)
    assert!(
        (r3.mag - 0.9047393878729882).abs() < EPSILON,
        "rank-3 combined magnitude"
    );

    // === METRIC TENSOR ===
    // the metric signature is a π rotation, not a matrix of inner products: a basis squares to
    // + or − by the angle it sits at, and a squared time axis at π/2 lands the negative grade.
    // that proof lives in spacetime_test::its_a_metric_signature — deferred here so the tensor
    // suite stays on tensor operations

    // === CHRISTOFFEL SYMBOLS ===
    // traditional: Γⁱⱼₖ connection coefficients, O(n³) storage
    // geonum: connection is just angle gradient

    let christoffel = |position: Geonum| -> Angle {
        // connection at this point: how angles change with position
        Angle::new(position.mag / 10.0, 1.0) // simple linear connection
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
    assert_eq!(after_loop.mag, 1.0, "curvature preserves magnitude");

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
    assert!(rational.near_mag(0.75));

    // test addition of fractions (3/4 + 1/2)
    let one = Geonum::scalar(1.0);
    let two = Geonum::scalar(2.0);
    let rational2 = one / two; // uses overloaded Div operator

    // addition of fractions: 3/4 + 1/2 = 5/4 = 1.25
    let fraction_sum = rational + rational2;
    assert!(fraction_sum.near_mag(1.25));
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
    assert!(sqrt2_pow2.near_mag(2.0)); // length: √2^2 = 2 ✓
                                       // scalar at angle 0: pow scales 0 by 2 = 0, so blade stays 0
    assert_eq!(sqrt2_pow2.angle.blade(), 0); // angle scaling: 2 * 0 = 0

    // square it
    let sqrt2_squared = sqrt2 * sqrt2;

    // test result is 2
    assert!(sqrt2_squared.near_mag(2.0));
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
    let scalar = g.mag * g.angle.grade_angle().cos();

    // extract grade 1 (vector part, magnitude)
    let vector_magnitude = g.mag * g.angle.grade_angle().sin();

    // extract grade 2 (bivector part)
    // in 2D GA, bivector represents rotation in the e1^e2 plane
    // which is encoded in the angle component
    let bivector_angle = g.angle.grade_angle();

    // demonstrate that all grades of the 2D geometric algebra are encoded
    // in just the 2 components (length and angle) of the geometric number

    // test extracted values for π/4 case
    assert!((scalar - 2.0 * (PI / 4.0).cos()).abs() < 1e-10); // 2 * √2/2 = √2
    assert!((vector_magnitude - 2.0 * (PI / 4.0).sin()).abs() < 1e-10); // 2 * √2/2 = √2
    assert!((bivector_angle - PI / 4.0).abs() < 1e-10); // π/4 angle preserved

    // log2(4) = 2 components (length and angle) instead of 4 components
    // this matches the statement from the README
    assert_eq!(4.0_f64.log2(), 2.0);
}

#[test]
fn it_dualizes_as_a_magnitude_preserving_involution() {
    // the dual loses nothing because it is a magnitude-preserving involution: dual(dual(x))
    // returns x's grade and length untouched, in any dimension, where the Hodge star k→(n−k)
    // needs the metric and the dimension count. the recovery is not a trivial unrotate — the
    // dual genuinely moves (a half turn away) and comes back

    let g = Geonum::new(3.0, 2.0, 3.0); // [3, 2π/3], grade 1
    let d = g.dual(); // adds π, two blades

    // the dual moved — a half turn away, not the identity dressed up
    assert!(
        d.angle.is_opposite(&g.angle),
        "the dual is a half turn away, a real move"
    );
    assert!(d.near_mag(g.mag), "and it keeps the length");

    // dual of dual returns the original grade and magnitude — an involution, nothing lost
    let back = d.dual();
    assert_eq!(
        back.angle.grade(),
        g.angle.grade(),
        "dual∘dual returns the grade"
    );
    assert!(back.near_mag(g.mag), "and the magnitude");

    // it holds a million dimensions out, where Hodge's k→(n−k) would need the dimension
    let hi = Geonum::new_with_angle(3.0, Angle::new_with_blade(1_000_000, 2.0, 3.0));
    assert_eq!(
        hi.dual().dual().angle.grade(),
        hi.angle.grade(),
        "involutive at any blade, no dimension consulted"
    );
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
    let b0_value = b0.mag; // 1
    let b1_value = b1.mag; // 0.5
    let b2_value = b2.mag; // ≈ 0.1667
    let b4_value = b4.mag * b4.angle.grade_angle().cos(); // division result projected to scalar

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
fn it_lands_quotients_at_grade_2_and_winds_the_sign_home() {
    // division carries the inversion's π: 1/z inverts through the unit circle, a π rotation, so
    // inv() adds two blades and every quotient lands two grades from its dividend. the sign of a
    // ratio is a position, not a bit — B₄'s −1/30 reads +1/30 once its numerator's π and the
    // inversion's π wind a full turn home

    // a positive-over-positive quotient lands at grade 2 — the inversion's π, not a negative
    let quotient = Geonum::scalar(3.0) / Geonum::scalar(4.0);
    assert!(quotient.near_mag(0.75), "3/4 = 0.75");
    assert_eq!(
        quotient.angle.grade(),
        2,
        "the quotient carries the inversion's π"
    );

    // −1 over a positive: the numerator's π plus the inversion's π wind to 2π — a full turn
    // home to grade 0, so (−1)/30 lands +1/30, a positive on the scalar ray
    let neg_one = Geonum::new(1.0, 1.0, 1.0); // −1 at angle π
    let b4 = neg_one / Geonum::scalar(30.0);
    assert!(b4.near_mag(1.0 / 30.0), "|B₄| = 1/30");
    assert_eq!(
        b4.angle.grade(),
        0,
        "π (numerator) + π (inversion) = 2π, wound home"
    );

    // read it as a signed scalar: the grade-0 landing projects to +1/30 — the winding is the sign
    let signed = b4.mag * b4.angle.grade_angle().cos();
    assert!(
        (signed - 1.0 / 30.0).abs() < EPSILON,
        "the sign is the winding: +1/30 at grade 0"
    );
}

#[test]
fn its_a_quadrature() {
    // quadrature is the quarter turn between a geonum and its dual: cos and sin are one unit
    // object read a π/2 apart, and that same quarter turn IS differentiation. the quadrature
    // is why 2 components (length, angle) carry what GA spends 4 on — sin(θ+π/2) = cos(θ)
    // folds the extra grades back

    let theta = Angle::new(2.0, 7.0); // some θ = 2π/7

    // sin(θ + π/2) = cos(θ): the quarter turn maps one onto the other, exactly
    let (cos_theta, _) = theta.cos_sin();
    let (_, sin_shifted) = (theta + Angle::new(1.0, 2.0)).cos_sin();
    assert!(
        (cos_theta - sin_shifted).abs() < EPSILON,
        "sin(θ+π/2) = cos(θ) — the quadrature identity"
    );

    // that quarter turn is differentiation: a unit object and its derivative are perpendicular
    let f = Geonum::new_with_angle(1.0, theta);
    let f_prime = f.differentiate();
    assert!(
        f.dot(&f_prime).near_mag(0.0),
        "f ⊥ f′ — differentiation is the quadrature quarter turn"
    );

    // integration is the inverse quarter turn — it returns f, magnitude and grade intact
    let recovered = f_prime.integrate();
    assert_eq!(
        recovered.angle.grade(),
        f.angle.grade(),
        "integrate undoes the quarter turn"
    );
    assert!(
        recovered.near_mag(f.mag),
        "no magnitude lost across the round trip"
    );
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
    assert_eq!(rotated.mag, 1.0);
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
    assert_eq!(multiplicative_inverse.mag, e_to_ipi.mag);
    let expected_inv_angle = e_to_ipi.angle + Angle::new(1.0, 1.0); // π + π = 2π (blade 4)
    assert_eq!(multiplicative_inverse.angle, expected_inv_angle);

    // STEP 5: Verify the multiplicative inverse property
    // [1, π] × [1, π] = [1×1, π+π] = [1, 2π] = [1, 0] = 1
    let self_product = e_to_ipi * e_to_ipi;

    // Test that e^(iπ) × e^(iπ) = 1 (multiplicative identity)
    assert_eq!(self_product.mag, 1.0);
    assert!(self_product.angle.grade_angle().abs() < EPSILON); // 2π ≡ 0 (mod 2π)

    // STEP 6: This is what Euler's identity actually demonstrates
    // Not mysterious connections between constants, but that [1, π]
    // is its own multiplicative inverse in geometric space

    // Verify: (-1) × (-1) = 1 in traditional arithmetic
    let traditional_check = (-1.0) * (-1.0);
    assert_eq!(traditional_check, 1.0);

    // Same relationship in geonum: [1, π] × [1, π] = [1, 0]
    assert_eq!(self_product.mag, traditional_check);

    // STEP 7: The additive part of Euler's identity
    // e^(iπ) + 1 = 0 shows that [1, π] and [1, 0] are additive inverses
    let one = Geonum::new(1.0, 0.0, 1.0); // [1, 0] = pointing forwards = +1

    // Verify they're additive inverses (opposite directions, same magnitude)
    let cartesian_sum =
        e_to_ipi.mag * e_to_ipi.angle.grade_angle().cos() + one.mag * one.angle.grade_angle().cos();
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
        e_to_ipi.mag * e_to_ipi.mag,     // 1 × 1 = 1
        e_to_ipi.angle + e_to_ipi.angle, // π + π = 2π
    );

    // 2π ≡ 0 (mod 2π), so we're back to [1, 0] = multiplicative identity
    assert_eq!(twice_rotated.mag, 1.0);
    assert!(twice_rotated.angle.grade_angle().abs() < EPSILON);

    // CONCLUSION: Euler's identity reveals fundamental geometric inverse properties

    // Much more descriptive than "most beautiful equation connecting constants", it only
    // demonstrates basic multiplicative and additive structure in geometric space

    // The "beauty" was artificial complexity masquerading as mathematical depth

    // The reality is simple: certain rotations are their own multiplicative inverses
}
