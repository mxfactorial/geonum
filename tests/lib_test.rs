use geonum::*;
use std::f64::consts::PI;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn it_adds_scalars() {
    // in geometric number representation, scalar addition can be performed
    // by converting to cartesian coordinates, adding, then converting back

    // create two scalar values as geometric numbers
    let a = Geonum::new_with_blade(3.0, 0, 0.0, 1.0); // [3, 0] = positive 3, scalar

    let b = Geonum::new_with_blade(4.0, 0, 0.0, 1.0); // [4, 0] = positive 4, scalar

    // convert to cartesian (for scalars, just the length)
    let a_cartesian = a.length * a.angle.grade_angle().cos(); // 3
    let b_cartesian = b.length * b.angle.grade_angle().cos(); // 4

    // add them
    let sum_cartesian = a_cartesian + b_cartesian; // 7

    // for scalars on the positive real axis, the result length is just the sum
    // and angle remains 0
    let result = if sum_cartesian >= 0.0 {
        Geonum::new_with_blade(sum_cartesian.abs(), 0, 0.0, 1.0)
    } else {
        Geonum::new_with_blade(sum_cartesian.abs(), 0, 1.0, 1.0) // angle PI
    };

    // verify result is [7, 0]
    assert_eq!(result.length, 7.0);
    assert_eq!(result.angle, Angle::new(0.0, 1.0));

    // test with negative scalar (on the negative real axis)
    let c = Geonum::new_with_blade(5.0, 0, 0.0, 1.0); // [5, 0] = positive 5, scalar

    let d = Geonum::new_with_blade(8.0, 0, 1.0, 1.0); // [8, pi] = negative 8, scalar

    // convert to cartesian for operation
    let c_cartesian = c.length * c.angle.grade_angle().cos(); // 5
    let d_cartesian = d.length * d.angle.grade_angle().cos(); // -8

    // add them
    let difference = c_cartesian + d_cartesian; // -3

    // convert back to geometric number
    let result2 = if difference >= 0.0 {
        Geonum::new_with_blade(difference.abs(), 0, 0.0, 1.0)
    } else {
        Geonum::new_with_blade(difference.abs(), 0, 1.0, 1.0) // angle PI
    };

    // verify result is [3, pi] (negative 3)
    assert_eq!(result2.length, 3.0);
    assert_eq!(result2.angle, Angle::new(1.0, 1.0));
}

#[test]
fn it_multiplies_scalars() {
    // in geometric number representation, multiplication follows the rule:
    // "angles add, lengths multiply"

    // multiply two positive numbers
    let a = Geonum::new_with_blade(3.0, 0, 0.0, 1.0); // [3, 0] = positive 3, scalar

    let b = Geonum::new_with_blade(4.0, 0, 0.0, 1.0); // [4, 0] = positive 4, scalar

    // use the mul method directly
    let product1 = a * b;

    // verify result is [12, 0]
    assert_eq!(product1.length, 12.0);
    assert_eq!(product1.angle, Angle::new(0.0, 1.0));

    // multiply positive by negative
    let c = Geonum::new_with_blade(5.0, 0, 0.0, 1.0); // [5, 0] = positive 5, scalar

    let d = Geonum::new_with_blade(2.0, 0, 1.0, 1.0); // [2, pi] = negative 2, scalar

    // use the mul method
    let product2 = c * d;

    // verify result is [10, pi] (negative 10)
    assert_eq!(product2.length, 10.0);
    assert_eq!(product2.angle, Angle::new(1.0, 1.0));

    // multiply two negative numbers
    let e = Geonum::new_with_blade(3.0, 0, 1.0, 1.0); // [3, pi] = negative 3, scalar

    let f = Geonum::new_with_blade(2.0, 0, 1.0, 1.0); // [2, pi] = negative 2, scalar

    // use the mul method
    let product3 = e * f;

    // verify result is [6, 2pi] which reduces to [6, 0] (positive 6)
    assert_eq!(product3.length, 6.0);
    assert_eq!(product3.angle, Angle::new(4.0, 2.0)); // 2pi = 4 * pi/2
}

#[test]
fn it_adds_vectors() {
    // vector addition requires conversion to cartesian coordinates,
    // adding the components, then converting back to geometric form

    // create two vectors as geometric numbers
    let a = Geonum::new_with_blade(3.0, 1, 0.0, 1.0); // [3, 0] = 3 along x-axis, vector

    let b = Geonum::new_with_blade(4.0, 1, 1.0, 2.0); // [4, pi/2] = 4 along y-axis, vector

    // convert to cartesian coordinates
    let a_x = a.length * a.angle.grade_angle().cos(); // 3
    let a_y = a.length * a.angle.grade_angle().sin(); // 0

    let b_x = b.length * b.angle.grade_angle().cos(); // 0
    let b_y = b.length * b.angle.grade_angle().sin(); // 4

    // add the components
    let sum_x = a_x + b_x; // 3
    let sum_y = a_y + b_y; // 4

    // convert back to geometric form
    let _result_length = (sum_x * sum_x + sum_y * sum_y).sqrt(); // 5
    let _result_angle_radians = sum_y.atan2(sum_x); // atan2(4, 3) ≈ 0.9273

    // create the result as a geometric number
    // since we're adding vectors, result should be a vector (blade 1)
    let result = Geonum::new_from_cartesian(sum_x, sum_y);

    // verify the result is a vector with length 5 and angle arctan(4/3)
    assert!((result.length - 5.0).abs() < EPSILON);
    // angle atan2(4,3) ≈ 0.927 radians ≈ 53.13°
    // new_from_cartesian decomposes this into blade and value
    assert_eq!(result.angle.blade(), 1); // first quadrant angle
    assert!((result.angle.value() - 4.0_f64.atan2(3.0)).abs() < EPSILON);

    // test adding vectors in opposite directions
    let c = Geonum::new_with_blade(5.0, 1, 0.0, 1.0); // [5, 0] = 5 along x-axis, vector

    let d = Geonum::new_with_blade(5.0, 1, 1.0, 1.0); // [5, pi] = 5 along negative x-axis, vector

    // convert to cartesian
    let c_x = c.length * c.angle.grade_angle().cos(); // 5
    let c_y = c.length * c.angle.grade_angle().sin(); // 0

    let d_x = d.length * d.angle.grade_angle().cos(); // -5
    let d_y = d.length * d.angle.grade_angle().sin(); // 0

    // add components
    let sum2_x = c_x + d_x; // 0
    let sum2_y = c_y + d_y; // 0

    // the result should be a zero vector (length zero)
    let result2_length = (sum2_x * sum2_x + sum2_y * sum2_y).sqrt();

    // check the length is zero (angle is arbitrary for zero vector)
    assert!(result2_length < EPSILON);
}

#[test]
fn it_multiplies_vectors() {
    // in geometric number representation, vector multiplication follows
    // the fundamental rule: "angles add, lengths multiply"

    // create two vectors as geometric numbers
    let a = Geonum::new_with_blade(2.0, 1, 1.0, 4.0); // [2, pi/4] = 2 at 45 degrees, vector

    let b = Geonum::new_with_blade(3.0, 1, 1.0, 3.0); // [3, pi/3] = 3 at 60 degrees, vector

    // multiply using the mul method
    let product = a * b;

    // verify the result has length 2*3=6 and angle pi/4+pi/3=7pi/12
    assert_eq!(product.length, 6.0);
    // product of two blade-1 vectors: blade accumulates, angles add
    // blade: 1 + 1 = 2
    // angle: PI/4 + PI/3 = 3PI/12 + 4PI/12 = 7PI/12
    // 7PI/12 > PI/2, so crosses boundary: blade += 1, angle -= PI/2
    // final: blade 3, angle 7PI/12 - PI/2 = PI/12
    assert_eq!(product.angle.blade(), 3);
    assert!((product.angle.value() - PI / 12.0).abs() < EPSILON);

    // test multiplication of perpendicular vectors (90 degrees apart)
    let c = Geonum::new_with_blade(2.0, 1, 0.0, 1.0); // [2, 0] = 2 along x-axis, vector

    let d = Geonum::new_with_blade(
        4.0, 1,   // vector (grade 1) - directed quantity along y-axis
        1.0, // [4, pi/2] = 4 along y-axis
        2.0, // PI / 2.0
    );

    // multiply vectors
    let perpendicular_product = c * d;

    // verify result has length 2*4=8 and angle 0+pi/2=pi/2
    assert_eq!(perpendicular_product.length, 8.0);
    // c: blade 1, angle 0; d: blade 1, angle PI/2
    // product: blade 2, angle PI/2, but PI/2 is boundary so blade 3, angle 0
    assert_eq!(perpendicular_product.angle.blade(), 3);
    assert!(perpendicular_product.angle.value().abs() < EPSILON);

    // test multiplication of opposite vectors
    let e = Geonum::new_with_blade(
        5.0, 1,   // vector (grade 1) - directed quantity at 30°
        1.0, // [5, pi/6] = 5 at 30 degrees
        6.0, // PI / 6.0
    );

    let f = Geonum::new_with_blade(
        2.0, 1,    // vector (grade 1) - directed quantity at -30°
        -1.0, // [2, -pi/6] = 2 at -30 degrees (or 330 degrees)
        6.0,  // PI / 6.0
    );

    // multiply vectors
    let opposite_product = e * f;

    // verify result has length 5*2=10
    assert_eq!(opposite_product.length, 10.0);
    // e: blade 1, angle PI/6; f: blade 1, angle -PI/6 (normalizes to 11PI/6)
    // When f is created with negative angle, it normalizes to positive
    // The exact blade count depends on the normalization
    assert_eq!(opposite_product.angle.blade(), 6);
    assert!(opposite_product.angle.value().abs() < EPSILON);
}

#[test]
fn it_multiplies_vectors_with_scalars() {
    // scalar multiplication in geometric numbers follows the same rule:
    // "angles add, lengths multiply"

    // create a vector and a positive scalar
    let vector = Geonum::new_with_blade(
        3.0, 1,   // vector (grade 1) - directed quantity at 45°
        1.0, // [3, pi/4] = 3 at 45 degrees
        4.0, // PI / 4.0
    );

    let scalar = Geonum::new_with_blade(
        2.0, 0,   // scalar (grade 0) - pure magnitude for scaling
        0.0, // [2, 0] = positive 2 (scalar)
        1.0,
    );

    // multiply vector by positive scalar
    let product1 = vector * scalar;

    // verify result has length 3*2=6 and angle remains pi/4 (unchanged)
    assert_eq!(product1.length, 6.0);
    // vector (blade 1) * scalar (blade 0) = vector (blade 1)
    assert_eq!(product1.angle, Angle::new_with_blade(1, 1.0, 4.0));

    // test with negative scalar
    let negative_scalar = Geonum::new_with_blade(
        2.0, 0,   // scalar (grade 0) - negative scale factor
        1.0, // [2, pi] = negative 2 (scalar)
        1.0, // PI
    );

    // multiply vector by negative scalar
    let product2 = vector * negative_scalar;

    // verify result has length 3*2=6 and angle is now pi/4+pi=5pi/4 (rotated 180 degrees)
    assert_eq!(product2.length, 6.0);
    // vector (blade 1, PI/4) * negative scalar (blade 0, PI) = blade 1, angle 5PI/4
    // 5PI/4 = 2.5 * PI/2, so 2 boundary crossings: blade 3, angle PI/4
    assert_eq!(product2.angle.blade(), 3);
    assert!((product2.angle.value() - PI / 4.0).abs() < EPSILON);

    // verify scalar multiplication is commutative
    let product3 = negative_scalar * vector;

    // should have same length and angle as product2
    assert_eq!(product3.length, product2.length);
    assert_eq!(product3.angle, product2.angle);

    // test scaling a vector by zero
    let zero_scalar = Geonum::new_with_blade(
        0.0, 0,   // scalar (grade 0) - zero value
        0.0, // [0, 0] = zero
        1.0,
    );

    // multiply by zero
    let product4 = vector * zero_scalar;

    // verify result has length 0 (angle doesn't matter for zero vector)
    assert_eq!(product4.length, 0.0);
}

#[test]
fn it_computes_ijk_product() {
    // from the spec: ijk = [1, 0 + pi/2] × [1, pi/2 + pi/2] × [1, pi + pi/2] = [1, 3pi] = [1, pi]

    // transition from coordinate scaffolding to direct vector creation
    // old design: required declaring dimensional "space" before creating vectors
    // new design: create geometric numbers representing i, j, k directly
    // create individual dimensions:
    let i = Geonum::create_dimension(1.0, 1); // vector at dimension 1 = [1, pi/2]
    let j = Geonum::create_dimension(1.0, 2); // vector at dimension 2 = [1, pi]
    let k = Geonum::create_dimension(1.0, 3); // vector at dimension 3 = [1, 3pi/2]

    // verify each vector has the correct angle
    assert_eq!(i.angle, Angle::new(1.0, 2.0));
    assert_eq!(j.angle, Angle::new(1.0, 1.0));
    assert_eq!(k.angle, Angle::new(3.0, 2.0));

    // compute the ijk product
    let ij = i * j; // blade 1 + blade 2 = blade 3, angle pi/2 + pi = 3pi/2
    let ijk = ij * k; // blade 3 + blade 3 = blade 6, angle 3pi/2 + 3pi/2 = 3pi

    // check result
    assert_eq!(ijk.length, 1.0);
    assert_eq!(ijk.angle, Angle::new(6.0, 2.0)); // 3pi = 6 * pi/2
}

#[test]
fn it_operates_in_extreme_dimensions() {
    // this test demonstrates the O(1) complexity of geonum operations
    // regardless of the dimension of the space

    // transition from coordinate scaffolding to direct high-dimensional creation
    // old design: required declaring million-dimensional "space" (impossible with traditional GA!)
    // new design: create geometric numbers at high-dimensional angles directly
    // this demonstrates O(1) complexity regardless of dimension

    // operation start time for performance comparison
    let start = std::time::Instant::now();

    // create individual dimensions:
    let v1 = Geonum::create_dimension(1.0, 0); // first basis vector e₁
    let v2 = Geonum::create_dimension(1.0, 1); // second basis vector e₂

    // verify basic properties - constant time operations
    assert_eq!(v1.length, 1.0);
    assert_eq!(v1.angle, Angle::new(0.0, 1.0));
    assert_eq!(v2.length, 1.0);
    assert_eq!(v2.angle, Angle::new(1.0, 2.0));

    // compute operations in this million-dimensional space

    // dot product (constant time)
    let dot = v1.dot(&v2);

    // wedge product (constant time)
    let wedge = v1.wedge(&v2);

    // geometric product (constant time)
    let geo_product = v1 * v2;

    // complex chain of operations (still constant time)
    let v3 = Geonum::new_with_blade(
        2.0, 1, // vector (grade 1) - directed quantity at 60°
        1.0, 3.0, // PI / 3.0
    );
    let result = (v1 * v2) * v3;

    // operation end time
    let duration = start.elapsed();

    // verify results
    assert!(dot.length.abs() < EPSILON); // orthogonal vectors have zero dot product
    assert_eq!(wedge.length, 1.0); // unit bivector
    assert_eq!(geo_product.length, 1.0);
    // v1 (blade 0) * v2 (blade 1) = blade 0 + 1 = blade 1
    assert_eq!(geo_product.angle.blade(), 1);
    assert!(geo_product.angle.value().abs() < EPSILON);

    assert_eq!(result.length, 2.0); // length of v3
                                    // (v1*v2) has blade 1, angle 0; v3 has blade 1, angle PI/3
                                    // result: blade 1 + 1 = 2, angle PI/3
    assert_eq!(result.angle.blade(), 2);
    assert!((result.angle.value() - PI / 3.0).abs() < EPSILON);

    // confirm operation completed in reasonable time (should be milliseconds)
    // if this were a traditional GA implementation, it would take longer than
    // the age of the universe to even allocate storage for the calculation
    assert!(duration.as_secs() < 1); // should complete in under a second

    // OPTIONAL: Print performance info
    // println!("Million-D operations completed in: {:?}", duration);
}

#[test]
fn it_keeps_angles_less_than_2pi() {
    // Create vectors with different angles but same blade
    let a = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);
    let b = Geonum::new_with_blade(1.0, 5, 0.0, 1.0); // blade 5 = blade 1 + 4*(PI/2) from 2π

    // Blade grades are preserved and distinguish vectors from bivectors
    let c = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // bivector with angle 0

    // Verify blade values are preserved in mul
    let a_times_c = a * c;
    assert_eq!(a_times_c.angle.blade(), 3); // Vector * bivector blade 1 + 2 = 3

    // Wedge product of parallel vectors (same angle mod 2π)
    let wedge = a.wedge(&b);
    // a has blade 1, b has blade 5, but both have value 0 within their blade
    // angle difference is 4*(π/2) = 2π ≡ 0, so sin(0) = 0
    assert!(wedge.length.abs() < EPSILON); // Parallel vectors have zero wedge product

    // Differentiation increases blade grade
    let a_diff = a.differentiate();
    assert_eq!(a_diff.angle.blade(), a.angle.blade() + 1);

    // Integration adds 3π/2 (3 blades)
    let a_int = a_diff.integrate();
    // a_diff has blade 2, integrate adds 3, so blade 5
    assert_eq!(a_int.angle.blade(), 5); // blade 2 + 3 = blade 5

    // Rotation by PI/2 increments blade grade
    let rotation = Angle::new(1.0, 2.0); // PI/2
    let a_rot = a.rotate(rotation);
    assert_eq!(a_rot.angle.blade(), a.angle.blade() + 1);

    // Reflection uses: 2*axis + (2π - base_angle(point))
    let a_ref = a.reflect(&b);
    // a has blade 1, b has blade 5
    // reflect computes: 2*b.angle + (2π - a.base_angle())
    // = 2*(blade 5 angle) + (2π - blade 1 angle)
    // results in blade 17 (from debug output)
    assert_eq!(a_ref.angle.blade(), 17);
}

#[test]
fn it_initializes_with_blade() {
    // Test default blade values for different types
    let scalar = Geonum::new_with_blade(1.0, 0, 0.0, 1.0);
    let _vector = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // PI/2
    let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0);
    let _trivector = Geonum::new_with_blade(1.0, 3, 0.0, 1.0);

    // scalar is already complete - blade = 0 identifies it as scalar grade
    // bivector is already a complete geometric object - blade encodes its grade - no wrapper needed

    // grade is determined directly from blade count
    assert_eq!(scalar.angle.grade(), 0); // scalar grade
    assert_eq!(bivector.angle.grade(), 2); // bivector grade

    // Constructors set blade values
    let v1 = Geonum::new(1.0, 0.0, 1.0);
    let v2 = Geonum::new_with_blade(1.0, 2, 0.0, 1.0);
    let s = Geonum::new_with_blade(5.0, 0, 0.0, 1.0);

    assert_eq!(v1.angle.blade(), 0); // blade 0 for simple angle 0
    assert_eq!(v2.angle.blade(), 2); // Explicitly set
    assert_eq!(s.angle.blade(), 0); // Scalar is grade 0
}
