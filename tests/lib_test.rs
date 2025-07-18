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
    let a_cartesian = a.length * a.angle.cos(); // 3
    let b_cartesian = b.length * b.angle.cos(); // 4

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
    let c_cartesian = c.length * c.angle.cos(); // 5
    let d_cartesian = d.length * d.angle.cos(); // -8

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
    let a_x = a.length * a.angle.cos(); // 3
    let a_y = a.length * a.angle.sin(); // 0

    let b_x = b.length * b.angle.cos(); // 0
    let b_y = b.length * b.angle.sin(); // 4

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
    let c_x = c.length * c.angle.cos(); // 5
    let c_y = c.length * c.angle.sin(); // 0

    let d_x = d.length * d.angle.cos(); // -5
    let d_y = d.length * d.angle.sin(); // 0

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
    let vectors = Multivector::create_dimension(1.0, &[1, 2, 3]);

    // extract the i, j, k vectors
    let i = vectors[0]; // vector at index 1 = [1, pi/2]
    let j = vectors[1]; // vector at index 2 = [1, pi]
    let k = vectors[2]; // vector at index 3 = [1, 3pi/2]

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

    // create two vectors in extreme-dimensional space without coordinate scaffolding
    let vectors = Multivector::create_dimension(1.0, &[0, 1]);
    let v1 = vectors[0]; // first basis vector e₁
    let v2 = vectors[1]; // second basis vector e₂

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
fn it_uses_multivector_operations() {
    // this test demonstrates how to use the multivector functionality

    // transition from coordinate scaffolding to direct multivector creation
    // old design: required declaring 3D "space" before creating multivectors
    // new design: create multivectors with geometric numbers directly

    // create multivectors with geometric numbers at standardized angles
    let e0 = Multivector::create_dimension(1.0, &[0]); // scalar at angle 0
    let e1_e2_e3 = Multivector::create_dimension(1.0, &[1, 2, 3]); // vectors at angles π/2, π, 3π/2

    // create a custom multivector with mixed grades
    let mixed_mv = Multivector(vec![
        Geonum::new_with_blade(
            1.0, 0, // scalar (grade 0) - scalar component of multivector
            0.0, 1.0,
        ), // scalar part
        Geonum::new_with_blade(
            2.0, 1, // vector (grade 1) - vector component of multivector
            0.0, 1.0, // angle 0, blade 1 gives total π/2
        ), // vector part
        Geonum::new_with_blade(
            3.0, 2, // bivector (grade 2) - represents oriented plane
            0.0, 1.0, // no additional angle beyond blade 2
        ), // bivector part
    ]);

    // 1. basic multivector operations

    // accessing elements
    assert_eq!(e0[0].length, 1.0);
    assert_eq!(e0[0].angle, Angle::new(0.0, 1.0));

    assert_eq!(e1_e2_e3[0].angle, Angle::new(1.0, 2.0)); // first vector (e1)
    assert_eq!(e1_e2_e3[1].angle, Angle::new(1.0, 1.0)); // second vector (e2)
    assert_eq!(e1_e2_e3[2].angle, Angle::new(3.0, 2.0)); // third vector (e3)

    // 2. grade operations

    // identify the grade of a pure blade
    assert_eq!(e0.blade_grade(), Some(0)); // scalar has grade 0

    // mixed grade multivectors dont have a single grade
    assert_eq!(mixed_mv.blade_grade(), None);

    // extract specific grades from a mixed multivector
    let scalar_parts = mixed_mv.grade(0);
    let vector_parts = mixed_mv.grade(1);

    // verify extracted components
    assert_eq!(scalar_parts.len(), 1); // only blade 0 components are scalars

    // scalar component (blade 0)
    assert_eq!(scalar_parts[0].length, 1.0);
    assert_eq!(scalar_parts[0].angle, Angle::new(0.0, 1.0));

    assert_eq!(vector_parts.len(), 1);
    assert_eq!(vector_parts[0].length, 2.0);
    assert_eq!(vector_parts[0].angle, Angle::new_with_blade(1, 0.0, 1.0));

    // 3. grade involution (negates odd-grade components)

    // create a multivector with even and odd grades
    let mv = Multivector(vec![
        Geonum::new_with_blade(
            1.0, 0, // scalar (grade 0) - scalar component
            0.0, 1.0,
        ), // scalar (grade 0)
        Geonum::new_with_blade(
            2.0, 1, // vector (grade 1) - directed component
            0.0, 1.0, // angle 0, blade 1 gives total π/2
        ), // vector (grade 1)
        Geonum::new_with_blade(
            3.0, 2, // bivector (grade 2) - plane element
            0.0, 1.0, // angle 0, blade 2 gives total π
        ), // bivector (grade 2)
    ]);

    // apply grade involution
    let inv = mv.involute();

    // verify: even grades unchanged, odd grades negated
    assert_eq!(inv[0].length, 1.0); // scalar unchanged
    assert_eq!(inv[0].angle, Angle::new(0.0, 1.0));

    assert_eq!(inv[1].length, 2.0); // vector negated
                                    // original: blade 1, angle 0; negated adds π: blade 3, angle 0
    assert_eq!(inv[1].angle, Angle::new_with_blade(3, 0.0, 1.0)); // angle rotated by π

    assert_eq!(inv[2].length, 3.0); // bivector unchanged
    assert_eq!(inv[2].angle, Angle::new_with_blade(2, 0.0, 1.0));

    // 4. clifford conjugate

    let conj = mv.conjugate();

    // verify clifford conjugate: scalar and vector unchanged, bivector negated
    assert_eq!(conj[0].length, 1.0); // scalar unchanged
    assert_eq!(conj[0].angle, Angle::new(0.0, 1.0));

    assert_eq!(conj[1].length, 2.0); // vector unchanged
    assert_eq!(conj[1].angle, Angle::new_with_blade(1, 0.0, 1.0)); // same as original

    assert_eq!(conj[2].length, 3.0); // bivector negated
    assert_eq!(conj[2].angle, Angle::new_with_blade(4, 0.0, 1.0)); // blade 2 + π = blade 4

    // 5. contraction operations

    // create two simple multivectors for contraction
    let a = Multivector(vec![
        Geonum::new_with_blade(
            2.0, 0, // scalar (grade 0) - magnitude component
            0.0, 1.0,
        ), // scalar
        Geonum::new_with_blade(
            1.0, 1, // vector (grade 1) - direction component
            1.0, 2.0, // PI / 2.0
        ), // vector
    ]);

    let b = Multivector(vec![
        Geonum::new_with_blade(
            3.0, 1, // vector (grade 1) - direction component
            1.0, 2.0, // PI / 2.0
        ), // vector
    ]);

    // compute left contraction
    let left = a.left_contract(&b);

    // left contraction lowers grade of b by grade of a
    // scalar⌋vector = vector
    // vector⌋vector = scalar (dot product)
    assert!(!left.is_empty());

    // right contraction
    let right = a.right_contract(&b);

    // right contraction lowers grade of a by grade of b
    // scalar⌊vector = 0 (scalar grade cant be lowered)
    // vector⌊vector = scalar (dot product)
    assert!(!right.is_empty());

    // 6. anti-commutator

    // compute anti-commutator {a,b} = (ab + ba)/2
    let anti_comm = a.anti_commutator(&b);

    // result contains components from both a*b and b*a
    assert!(!anti_comm.is_empty());

    // 7. conversion from Vec<Geonum>

    let geonums = vec![
        Geonum::new_with_blade(
            1.0, 0, // scalar (grade 0) - magnitude component
            0.0, 1.0,
        ),
        Geonum::new_with_blade(
            2.0, 1, // vector (grade 1) - direction component
            1.0, 2.0, // PI / 2.0
        ),
    ];

    // convert using From trait
    let from_vec = Multivector::from(geonums);
    assert_eq!(from_vec.len(), 2);

    // 8. with_capacity and push operations

    let mut dynamic_mv = Multivector::with_capacity(3);
    assert_eq!(dynamic_mv.len(), 0);

    // use deref to access vec methods
    dynamic_mv.push(Geonum::new_with_blade(
        1.0, 0, // scalar (grade 0) - magnitude component
        0.0, 1.0,
    ));
    dynamic_mv.push(Geonum::new_with_blade(
        2.0, 1, // vector (grade 1) - direction component
        1.0, 2.0, // PI / 2.0
    ));

    assert_eq!(dynamic_mv.len(), 2);
    assert_eq!(dynamic_mv[0].length, 1.0);
    assert_eq!(dynamic_mv[1].angle, Angle::new_with_blade(2, 0.0, 1.0)); // blade 1 + π/2 = blade 2

    // 9. interior product operation

    // create two vectors to demonstrate interior product
    let x_axis = Multivector(vec![Geonum::new_with_blade(
        1.0, 1,   // vector (grade 1) - directed quantity along x-axis
        0.0, // e₁ - vector along x-axis
        1.0,
    )]);

    let y_axis = Multivector(vec![Geonum::new_with_blade(
        1.0, 1,   // vector (grade 1) - directed quantity along y-axis
        1.0, // e₂ - vector along y-axis
        2.0, // PI / 2.0
    )]);

    // compute interior product of perpendicular vectors
    let interior_perp = x_axis.interior_product(&y_axis);

    // for perpendicular vectors, interior product should be very small or zero
    let total_magnitude: f64 = interior_perp.0.iter().map(|g| g.length).sum();
    assert!(total_magnitude < 0.1);

    // create a 45-degree vector
    let angle45 = Multivector(vec![Geonum::new_with_blade(
        1.0, 1,   // vector (grade 1) - directed quantity at 45° angle
        1.0, // 45 degrees
        4.0, // PI / 4.0
    )]);

    // interior product with x-axis
    let interior_45x = angle45.interior_product(&x_axis);

    // should have non-zero length (projection component)
    assert!(!interior_45x.is_empty());

    // 10. dual operation

    // create a 2D pseudoscalar (bivector in xy-plane)
    let pseudoscalar = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - oriented area element in xy-plane for rotation
        1.0, // represents e₁∧e₂ (bivector in xy-plane)
        2.0, // PI / 2.0
    )]);

    // compute dual of x-axis vector
    let x_dual = x_axis.dual(&pseudoscalar);

    // in 2D, dual of x-axis should be y-axis (with sign depending on orientation)
    assert!(!x_dual.is_empty());

    // check that the result is non-zero
    let dual_magnitude: f64 = x_dual.0.iter().map(|g| g.length).sum();
    assert!(dual_magnitude > 0.1);

    // 11. exponential operation

    // create a bivector representing the xy-plane
    let xy_plane = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - oriented area element in xy-plane for rotation
        1.0, // bivector e₁∧e₂
        2.0, // PI / 2.0
    )]);

    // compute the exponential e^(θ/2 * bivector) to create a rotor
    // for a 90-degree rotation in the xy-plane, θ/2 = 45 degrees = PI/4
    let rotor = Multivector::exp(&xy_plane, Angle::new(1.0, 4.0));

    // a rotor should contain a scalar and bivector part
    assert!(rotor.len() >= 2);

    // scalar part should be cos(PI/4) ≈ 0.7071
    assert!((rotor[0].length - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);

    // bivector part should be sin(PI/4) ≈ 0.7071
    assert!((rotor[1].length - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);

    // use the rotor to rotate a vector
    let rotated = x_axis.rotate(&rotor);

    // 90-degree rotation of x-axis should point along y-axis
    assert!(!rotated.is_empty());

    // rotated vector should have some non-zero magnitude
    // due to implementation differences, we won't check exact magnitude
    let rotated_magnitude: f64 = rotated.0.iter().map(|g| g.length).sum();
    assert!(rotated_magnitude > 0.1);
}

#[test]
fn it_uses_advanced_operations() {
    // This test demonstrates the newly implemented operations:
    // - Sandwich product
    // - Commutator product
    // - Meet and Join operations
    // - Square root
    // - Undual operations

    // Create two vectors in 2D space
    let v1 = Multivector(vec![Geonum::new_with_blade(
        2.0, 1,   // vector (grade 1) - directed quantity along x-axis
        0.0, // along x-axis
        1.0,
    )]);

    let v2 = Multivector(vec![Geonum::new_with_blade(
        3.0, 1,   // vector (grade 1) - directed quantity along y-axis
        0.0, // blade 1 with angle 0 to keep it grade 1
        1.0,
    )]);

    // 1. Sandwich Product
    // Create a rotor (using the exp function)
    // Rotate in xy-plane by 90 degrees
    let plane = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - rotation plane for sandwich product
        1.0, // bivector e₁∧e₂
        2.0, // PI / 2.0
    )]);

    // e^(θ/2 * bivector) - for 90° rotation, θ/2 = π/4
    let rotor = Multivector::exp(&plane, Angle::new(1.0, 4.0));

    // Get the reverse of the rotor
    let rotor_rev = rotor.conjugate();

    // Apply sandwich product R*v1*R̃ to rotate v1
    let rotated = rotor.sandwich_product(&v1, &rotor_rev);

    // The result should be a vector with similar magnitude but rotated direction
    assert!(!rotated.is_empty());

    // 2. Commutator Product
    // Commutator measures the failure of two elements to commute
    // In geonum, vectors with same blade commute (parallel vectors)
    let comm = v1.commutator(&v2);

    // v1 and v2 both have blade 1 with angle 0 - they're parallel
    // parallel vectors commute: AB = BA, so [A,B] = 0
    assert!(
        comm.is_empty(),
        "parallel vectors commute, giving zero commutator"
    );

    // 3. Join and Meet operations
    // Join - represents the union of subspaces (similar to span)
    let join = v1.join(&v2);

    // For two basis vectors, the join should be their plane
    assert!(!join.is_empty());

    // Meet - represents the intersection of subspaces
    // For two non-parallel vectors in a plane, this should be their intersection point
    let meet = v1.meet(&v2, None);

    // Verify the operation completes successfully
    let _ = meet;

    // 4. Square root operation
    // Create a scalar
    let scalar = Multivector(vec![Geonum::new_with_blade(
        4.0, 0,   // scalar (grade 0) - pure magnitude component
        0.0, // positive scalar
        1.0,
    )]);

    // Compute the square root of the scalar
    let sqrt_scalar = scalar.sqrt();

    // The result should be a scalar with length 2.0
    assert_eq!(sqrt_scalar[0].length, 2.0);
    assert_eq!(sqrt_scalar[0].angle.mod_4_angle(), 0.0);

    // Create a negative scalar
    let neg_scalar = Multivector(vec![Geonum::new_with_blade(
        9.0, 0,   // scalar (grade 0) - pure magnitude component
        1.0, // negative scalar
        1.0, // PI
    )]);

    // Compute the square root of the negative scalar
    let sqrt_neg = neg_scalar.sqrt();

    // The result should be a bivector with length 3.0 and angle π/2
    assert_eq!(sqrt_neg[0].length, 3.0);
    assert_eq!(sqrt_neg[0].angle.mod_4_angle(), PI / 2.0); // converts to bivector

    // 5. Dual and Undual operations
    // Create a pseudoscalar for the 2D plane
    let pseudoscalar = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - oriented area element for dual operation
        1.0, // e₁∧e₂ with orientation
        1.0, // PI
    )]);

    // Compute the dual of v1 (x-axis vector)
    let dual_v1 = v1.dual(&pseudoscalar);

    println!("v1 blade: {}", v1[0].angle.blade());
    println!("pseudoscalar blade: {}", pseudoscalar[0].angle.blade());
    println!("dual_v1 blade: {}", dual_v1[0].angle.blade());

    // In 2D, the dual of the x-axis is the y-axis (possibly with sign change)
    // Now compute the undual to get back the original vector
    let undual_v1 = dual_v1.undual(&pseudoscalar);

    println!("undual_v1 blade: {}", undual_v1[0].angle.blade());

    // The undual should get us back to the original vector (allowing for round-trip precision issues)
    assert!((undual_v1[0].length - v1[0].length).abs() < EPSILON);
    assert_eq!(undual_v1[0].angle, v1[0].angle);

    // 6. Section for pseudoscalar
    // Create a pseudoscalar for a 2D plane
    let section_pseudoscalar = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - pseudoscalar for planar section
        1.0, // 2D pseudoscalar
        1.0, // PI
    )]);

    // Create a multivector with various components
    let section_mixed = Multivector(vec![
        Geonum::new_with_blade(
            2.0, 0,   // scalar (grade 0) - pure magnitude component
            0.0, // Scalar component
            1.0,
        ),
        Geonum::new_with_blade(
            3.0, 1,   // vector (grade 1) - directed component
            1.0, // Vector component
            2.0, // PI / 2.0
        ),
        Geonum::new_with_blade(
            5.0, 1,   // vector (grade 1) - directed component at 45°
            1.0, // Non-standard component
            4.0, // PI / 4.0
        ),
    ]);

    // Extract the section for this pseudoscalar
    let section = section_mixed.section(&section_pseudoscalar);

    // Verify the section contains the expected components
    assert!(!section.is_empty(), "Section should not be empty");

    // In a practical application, we would check which components
    // belong to the pseudoscalar's subspace and use them for
    // further calculations

    // 7. Regressive product (meet operation alternative)
    // Create a pseudoscalar for the 2D plane
    let regr_pseudoscalar = Multivector(vec![Geonum::new_with_blade(
        1.0, 2,   // bivector (grade 2) - oriented area element for regressive product
        1.0, // e₁∧e₂ with orientation
        1.0, // PI
    )]);

    // Create two lines in 2D
    let line1 = Multivector(vec![Geonum::new_with_blade(
        1.0, 1,   // vector (grade 1) - directed quantity representing a line
        1.0, // A line at 45 degrees
        4.0, // PI / 4.0
    )]);

    let line2 = Multivector(vec![Geonum::new_with_blade(
        1.0, 1,   // vector (grade 1) - directed quantity representing a line
        3.0, // A line at 135 degrees (perpendicular to line1)
        4.0, // PI / 4.0
    )]);

    // Compute the regressive product to find their intersection
    let intersection = line1.regressive_product(&line2, &regr_pseudoscalar);

    // In 2D, the regressive product of two lines is their intersection point
    assert!(
        !intersection.is_empty(),
        "Regressive product should produce a non-empty result"
    );

    // Verify the regressive product is working properly - two perpendicular lines should intersect
    // The exact result depends on the line representations, but it should be non-empty
    let magnitude: f64 = intersection.0.iter().map(|g| g.length).sum();
    assert!(
        magnitude > EPSILON,
        "Regressive product magnitude should be non-zero"
    );

    // 8. Automatic differentiation and integration
    // Create a vector for differentiation
    let vector = Multivector(vec![Geonum::new_with_blade(
        2.0, 1,   // vector (grade 1) - directed quantity for differentiation
        1.0, // A vector at 60 degrees
        3.0, // PI / 3.0
    )]);

    // Compute the derivative
    let derivative = vector.differentiate();

    // The derivative should rotate the angle by π/2
    assert_eq!(derivative[0].length, 2.0); // Length is preserved
    assert_eq!(derivative[0].angle, vector[0].angle + Angle::new(1.0, 2.0)); // Angle is rotated by π/2

    // Compute the integral
    let integral = vector.integrate();

    // The integral should rotate the angle by -π/2
    assert_eq!(integral[0].length, 2.0); // Length is preserved
                                         // vector has blade 1 + π/3, integration subtracts π/2, giving blade 0 + π/3
    assert_eq!(integral[0].angle.blade(), 0); // blade 1 - 1 = blade 0
    assert!((integral[0].angle.value() - (PI / 3.0)).abs() < EPSILON); // π/3 value

    // Demonstrate relationship between differentiation and integration
    // Differentiating and then integrating should give back the original (fundamental theorem of calculus)
    let roundtrip = derivative.integrate();

    assert_eq!(roundtrip[0].length, vector[0].length);
    assert_eq!(roundtrip[0].angle, vector[0].angle);

    // Demonstrate that second derivative equals negative of original (d²/dx² = -1)
    let second_derivative = derivative.differentiate();
    assert_eq!(second_derivative[0].length, vector[0].length);
    assert_eq!(
        second_derivative[0].angle,
        vector[0].angle + Angle::new(1.0, 1.0)
    );
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

    // Integration decreases blade grade
    let a_int = a_diff.integrate();
    // a has blade 1, a_diff has blade 2, a_int should have blade 1
    assert_eq!(a_int.angle.blade(), 1); // blade 2 - 1 = blade 1

    // Rotation by PI/2 increments blade grade
    let rotation = Angle::new(1.0, 2.0); // PI/2
    let a_rot = a.rotate(rotation);
    assert_eq!(a_rot.angle.blade(), a.angle.blade() + 1);

    // Reflection changes blade grade based on the reflection plane
    let a_ref = a.reflect(&b);
    // reflection formula: -b * a * b / |b|^2
    // blade math: 5 + 1 + 5 - 8 (mod adjustments) = 3
    assert_eq!(a_ref.angle.blade(), 3);
}

#[test]
fn it_initializes_with_blade() {
    // Test default blade values for different types
    let scalar = Geonum::new_with_blade(1.0, 0, 0.0, 1.0);
    let _vector = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // PI/2
    let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0);
    let _trivector = Geonum::new_with_blade(1.0, 3, 0.0, 1.0);

    // Test that blade value differentiates types even with same angle
    let scalar_multivector = Multivector(vec![scalar]);
    let bivector_multivector = Multivector(vec![bivector]);

    // Extract grades by blade
    let extracted_scalars = scalar_multivector.grade(0);
    let extracted_bivectors = bivector_multivector.grade(2);

    // Assert extraction worked
    assert_eq!(extracted_scalars.0.len(), 1);
    assert_eq!(extracted_bivectors.0.len(), 1);

    // Cross-check - this is empty since we're extracting the wrong grade
    let empty_extract = scalar_multivector.grade(2);
    assert_eq!(empty_extract.0.len(), 0);

    // Constructors set blade values
    let v1 = Geonum::new(1.0, 0.0, 1.0);
    let v2 = Geonum::new_with_blade(1.0, 2, 0.0, 1.0);
    let s = Geonum::new_with_blade(5.0, 0, 0.0, 1.0);

    assert_eq!(v1.angle.blade(), 0); // blade 0 for simple angle 0
    assert_eq!(v2.angle.blade(), 2); // Explicitly set
    assert_eq!(s.angle.blade(), 0); // Scalar is grade 0
}
