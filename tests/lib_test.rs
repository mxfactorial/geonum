use geonum::*;
use std::f64::consts::PI;

#[test]
fn it_adds_scalars() {
    // in geometric number representation, scalar addition can be performed
    // by converting to cartesian coordinates, adding, then converting back

    // create two scalar values as geometric numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = positive 3
    };

    let b = Geonum {
        length: 4.0,
        angle: 0.0, // [4, 0] = positive 4
    };

    // convert to cartesian (for scalars, just the length)
    let a_cartesian = a.length * a.angle.cos(); // 3
    let b_cartesian = b.length * b.angle.cos(); // 4

    // add them
    let sum_cartesian = a_cartesian + b_cartesian; // 7

    // for scalars on the positive real axis, the result length is just the sum
    // and angle remains 0
    let result = Geonum {
        length: sum_cartesian.abs(),
        angle: if sum_cartesian >= 0.0 { 0.0 } else { PI },
    };

    // verify result is [7, 0]
    assert_eq!(result.length, 7.0);
    assert_eq!(result.angle, 0.0);

    // test with negative scalar (on the negative real axis)
    let c = Geonum {
        length: 5.0,
        angle: 0.0, // [5, 0] = positive 5
    };

    let d = Geonum {
        length: 8.0,
        angle: PI, // [8, pi] = negative 8
    };

    // convert to cartesian for operation
    let c_cartesian = c.length * c.angle.cos(); // 5
    let d_cartesian = d.length * d.angle.cos(); // -8

    // add them
    let difference = c_cartesian + d_cartesian; // -3

    // convert back to geometric number
    let result2 = Geonum {
        length: difference.abs(),
        angle: if difference >= 0.0 { 0.0 } else { PI },
    };

    // verify result is [3, pi] (negative 3)
    assert_eq!(result2.length, 3.0);
    assert_eq!(result2.angle, PI);
}

#[test]
fn it_multiplies_scalars() {
    // in geometric number representation, multiplication follows the rule:
    // "lengths multiply, angles add"

    // multiply two positive numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = positive 3
    };

    let b = Geonum {
        length: 4.0,
        angle: 0.0, // [4, 0] = positive 4
    };

    // use the mul method directly
    let product1 = a.mul(&b);

    // verify result is [12, 0]
    assert_eq!(product1.length, 12.0);
    assert_eq!(product1.angle, 0.0);

    // multiply positive by negative
    let c = Geonum {
        length: 5.0,
        angle: 0.0, // [5, 0] = positive 5
    };

    let d = Geonum {
        length: 2.0,
        angle: PI, // [2, pi] = negative 2
    };

    // use the mul method
    let product2 = c.mul(&d);

    // verify result is [10, pi] (negative 10)
    assert_eq!(product2.length, 10.0);
    assert_eq!(product2.angle, PI);

    // multiply two negative numbers
    let e = Geonum {
        length: 3.0,
        angle: PI, // [3, pi] = negative 3
    };

    let f = Geonum {
        length: 2.0,
        angle: PI, // [2, pi] = negative 2
    };

    // use the mul method
    let product3 = e.mul(&f);

    // verify result is [6, 2pi] which should reduce to [6, 0] (positive 6)
    assert_eq!(product3.length, 6.0);
    assert!(product3.angle % (2.0 * PI) < 1e-10); // should be 0 or very close to it
}

#[test]
fn it_adds_vectors() {
    // vector addition requires conversion to cartesian coordinates,
    // adding the components, then converting back to geometric form

    // create two vectors as geometric numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = 3 along x-axis
    };

    let b = Geonum {
        length: 4.0,
        angle: PI / 2.0, // [4, pi/2] = 4 along y-axis
    };

    // convert to cartesian coordinates
    let a_x = a.length * a.angle.cos(); // 3
    let a_y = a.length * a.angle.sin(); // 0

    let b_x = b.length * b.angle.cos(); // 0
    let b_y = b.length * b.angle.sin(); // 4

    // add the components
    let sum_x = a_x + b_x; // 3
    let sum_y = a_y + b_y; // 4

    // convert back to geometric form
    let result_length = (sum_x * sum_x + sum_y * sum_y).sqrt(); // 5
    let result_angle = sum_y.atan2(sum_x); // atan2(4, 3) ≈ 0.9273

    // create the result as a geometric number
    let result = Geonum {
        length: result_length,
        angle: result_angle,
    };

    // verify the result is a vector with length 5 and angle arctan(4/3)
    assert!((result.length - 5.0).abs() < 1e-10);
    assert!((result.angle - 4.0_f64.atan2(3.0)).abs() < 1e-10);

    // test adding vectors in opposite directions
    let c = Geonum {
        length: 5.0,
        angle: 0.0, // [5, 0] = 5 along x-axis
    };

    let d = Geonum {
        length: 5.0,
        angle: PI, // [5, pi] = 5 along negative x-axis
    };

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
    assert!(result2_length < 1e-10);
}

#[test]
fn it_multiplies_vectors() {
    // in geometric number representation, vector multiplication follows
    // the fundamental rule: "lengths multiply, angles add"

    // create two vectors as geometric numbers
    let a = Geonum {
        length: 2.0,
        angle: PI / 4.0, // [2, pi/4] = 2 at 45 degrees
    };

    let b = Geonum {
        length: 3.0,
        angle: PI / 3.0, // [3, pi/3] = 3 at 60 degrees
    };

    // multiply using the mul method
    let product = a.mul(&b);

    // verify the result has length 2*3=6 and angle pi/4+pi/3=7pi/12
    assert_eq!(product.length, 6.0);
    assert!((product.angle - (PI / 4.0 + PI / 3.0)).abs() < 1e-10);

    // test multiplication of perpendicular vectors (90 degrees apart)
    let c = Geonum {
        length: 2.0,
        angle: 0.0, // [2, 0] = 2 along x-axis
    };

    let d = Geonum {
        length: 4.0,
        angle: PI / 2.0, // [4, pi/2] = 4 along y-axis
    };

    // multiply vectors
    let perpendicular_product = c.mul(&d);

    // verify result has length 2*4=8 and angle 0+pi/2=pi/2
    assert_eq!(perpendicular_product.length, 8.0);
    assert_eq!(perpendicular_product.angle, PI / 2.0);

    // test multiplication of opposite vectors
    let e = Geonum {
        length: 5.0,
        angle: PI / 6.0, // [5, pi/6] = 5 at 30 degrees
    };

    let f = Geonum {
        length: 2.0,
        angle: -PI / 6.0, // [2, -pi/6] = 2 at -30 degrees (or 330 degrees)
    };

    // multiply vectors
    let opposite_product = e.mul(&f);

    // verify result has length 5*2=10 and angle pi/6+(-pi/6)=0
    assert_eq!(opposite_product.length, 10.0);
    assert!((opposite_product.angle % (2.0 * PI)).abs() < 1e-10); // should be 0
}

#[test]
fn it_multiplies_vectors_with_scalars() {
    // scalar multiplication in geometric numbers follows the same rule:
    // "lengths multiply, angles add"

    // create a vector and a positive scalar
    let vector = Geonum {
        length: 3.0,
        angle: PI / 4.0, // [3, pi/4] = 3 at 45 degrees
    };

    let scalar = Geonum {
        length: 2.0,
        angle: 0.0, // [2, 0] = positive 2 (scalar)
    };

    // multiply vector by positive scalar
    let product1 = vector.mul(&scalar);

    // verify result has length 3*2=6 and angle remains pi/4 (unchanged)
    assert_eq!(product1.length, 6.0);
    assert_eq!(product1.angle, PI / 4.0);

    // test with negative scalar
    let negative_scalar = Geonum {
        length: 2.0,
        angle: PI, // [2, pi] = negative 2 (scalar)
    };

    // multiply vector by negative scalar
    let product2 = vector.mul(&negative_scalar);

    // verify result has length 3*2=6 and angle is now pi/4+pi=5pi/4 (rotated 180 degrees)
    assert_eq!(product2.length, 6.0);
    assert!((product2.angle - (PI / 4.0 + PI)).abs() < 1e-10);

    // verify scalar multiplication is commutative
    let product3 = negative_scalar.mul(&vector);

    // should have same length and angle as product2
    assert_eq!(product3.length, product2.length);
    assert!((product3.angle - product2.angle).abs() < 1e-10);

    // test scaling a vector by zero
    let zero_scalar = Geonum {
        length: 0.0,
        angle: 0.0, // [0, 0] = zero
    };

    // multiply by zero
    let product4 = vector.mul(&zero_scalar);

    // verify result has length 0 (angle doesn't matter for zero vector)
    assert_eq!(product4.length, 0.0);
}

#[test]
fn it_computes_ijk_product() {
    // from the spec: ijk = [1, 0 + pi/2] × [1, pi/2 + pi/2] × [1, pi + pi/2] = [1, 3pi] = [1, pi]

    // create a dimensions object with 1 dimension
    let dims = Dimensions::new(1);

    // create i, j, k vectors using multivector method
    let vectors = dims.multivector(&[1, 2, 3]);

    // extract the i, j, k vectors
    let i = vectors[0]; // vector at index 1 = [1, pi/2]
    let j = vectors[1]; // vector at index 2 = [1, pi]
    let k = vectors[2]; // vector at index 3 = [1, 3pi/2]

    // verify each vector has the correct angle
    assert_eq!(i.angle, PI / 2.0);
    assert_eq!(j.angle, PI);
    assert_eq!(k.angle, 3.0 * PI / 2.0);

    // compute the ijk product
    let ij = i.mul(&j); // [1, pi/2] × [1, pi] = [1, 3pi/2]
    let ijk = ij.mul(&k); // [1, 3pi/2] × [1, 3pi/2] = [1, 3pi] = [1, pi]

    // check result
    assert_eq!(ijk.length, 1.0);
    assert_eq!(ijk.angle, PI);
}

#[test]
fn it_operates_in_extreme_dimensions() {
    // this test demonstrates the O(1) complexity of geonum operations
    // regardless of the dimension of the space

    // create a 1 million dimensional space (10⁶ D)
    // this would be impossible with traditional geometric algebra (2^10⁶ components!)
    let million_d = Dimensions::new(1_000_000);

    // operation start time for performance comparison
    let start = std::time::Instant::now();

    // create two vectors in this extreme-dimensional space
    let vectors = million_d.multivector(&[0, 1]);
    let v1 = vectors[0]; // first basis vector e₁
    let v2 = vectors[1]; // second basis vector e₂

    // verify basic properties - constant time operations
    assert_eq!(v1.length, 1.0);
    assert_eq!(v1.angle, 0.0);
    assert_eq!(v2.length, 1.0);
    assert_eq!(v2.angle, PI / 2.0);

    // compute operations in this million-dimensional space

    // dot product (constant time)
    let dot = v1.dot(&v2);

    // wedge product (constant time)
    let wedge = v1.wedge(&v2);

    // geometric product (constant time)
    let geo_product = v1.mul(&v2);

    // complex chain of operations (still constant time)
    let v3 = Geonum {
        length: 2.0,
        angle: PI / 3.0,
    };
    let result = v1.mul(&v2).mul(&v3);

    // operation end time
    let duration = start.elapsed();

    // verify results
    assert!(dot.abs() < 1e-10); // orthogonal vectors have zero dot product
    assert_eq!(wedge.length, 1.0); // unit bivector
    assert_eq!(geo_product.length, 1.0);
    assert_eq!(geo_product.angle, PI / 2.0);
    assert_eq!(result.length, 2.0); // length of v3
    assert!((result.angle - (PI / 2.0 + PI / 3.0)).abs() < 1e-10);

    // confirm operation completed in reasonable time (should be milliseconds)
    // if this were a traditional GA implementation, it would take longer than
    // the age of the universe to even allocate storage for the calculation
    assert!(duration.as_secs() < 1); // should complete in under a second

    // OPTIONAL: Print performance info
    // println!("Million-D operations completed in: {:?}", duration);
}
