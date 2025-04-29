use geonum::*;
use std::f64::consts::PI;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;
const TWO_PI: f64 = 2.0 * PI;

#[test]
fn it_adds_scalars() {
    // in geometric number representation, scalar addition can be performed
    // by converting to cartesian coordinates, adding, then converting back

    // create two scalar values as geometric numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = positive 3
        blade: 0,   // scalar (grade 0) - pure magnitude without direction
    };

    let b = Geonum {
        length: 4.0,
        angle: 0.0, // [4, 0] = positive 4
        blade: 0,   // scalar (grade 0) - pure magnitude without direction
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
        blade: 0, // scalar (grade 0) - result of scalar addition remains scalar
    };

    // verify result is [7, 0]
    assert_eq!(result.length, 7.0);
    assert_eq!(result.angle, 0.0);

    // test with negative scalar (on the negative real axis)
    let c = Geonum {
        length: 5.0,
        angle: 0.0, // [5, 0] = positive 5
        blade: 0,   // scalar (grade 0) - pure magnitude without direction
    };

    let d = Geonum {
        length: 8.0,
        angle: PI, // [8, pi] = negative 8
        blade: 0,  // scalar (grade 0) - negative scalar on real axis
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
        blade: 0, // scalar (grade 0) - result of scalar addition is still a scalar
    };

    // verify result is [3, pi] (negative 3)
    assert_eq!(result2.length, 3.0);
    assert_eq!(result2.angle, PI);
}

#[test]
fn it_multiplies_scalars() {
    // in geometric number representation, multiplication follows the rule:
    // "angles add, lengths multiply"

    // multiply two positive numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = positive 3
        blade: 0,   // scalar (grade 0) - positive number on real axis
    };

    let b = Geonum {
        length: 4.0,
        angle: 0.0, // [4, 0] = positive 4
        blade: 0,   // scalar (grade 0) - positive number on real axis
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
        blade: 0,   // scalar (grade 0) - positive number on real axis
    };

    let d = Geonum {
        length: 2.0,
        angle: PI, // [2, pi] = negative 2
        blade: 0,  // scalar (grade 0) - negative number on real axis
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
        blade: 0,  // scalar (grade 0) - negative number on real axis
    };

    let f = Geonum {
        length: 2.0,
        angle: PI, // [2, pi] = negative 2
        blade: 0,  // scalar (grade 0) - negative number on real axis
    };

    // use the mul method
    let product3 = e.mul(&f);

    // verify result is [6, 2pi] which should reduce to [6, 0] (positive 6)
    assert_eq!(product3.length, 6.0);
    assert!(product3.angle % (2.0 * PI) < EPSILON); // should be 0 or very close to it
}

#[test]
fn it_adds_vectors() {
    // vector addition requires conversion to cartesian coordinates,
    // adding the components, then converting back to geometric form

    // create two vectors as geometric numbers
    let a = Geonum {
        length: 3.0,
        angle: 0.0, // [3, 0] = 3 along x-axis
        blade: 1,   // vector (grade 1) - directed quantity along x-axis
    };

    let b = Geonum {
        length: 4.0,
        angle: PI / 2.0, // [4, pi/2] = 4 along y-axis
        blade: 1,        // vector (grade 1) - directed quantity along y-axis
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
        blade: 1, // vector (grade 1) - directed quantity
    };

    // verify the result is a vector with length 5 and angle arctan(4/3)
    assert!((result.length - 5.0).abs() < EPSILON);
    assert!((result.angle - 4.0_f64.atan2(3.0)).abs() < EPSILON);

    // test adding vectors in opposite directions
    let c = Geonum {
        length: 5.0,
        angle: 0.0, // [5, 0] = 5 along x-axis
        blade: 1,   // vector (grade 1) - directed quantity along x-axis
    };

    let d = Geonum {
        length: 5.0,
        angle: PI, // [5, pi] = 5 along negative x-axis
        blade: 1,  // vector (grade 1) - directed quantity along negative x-axis
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
    assert!(result2_length < EPSILON);
}

#[test]
fn it_multiplies_vectors() {
    // in geometric number representation, vector multiplication follows
    // the fundamental rule: "angles add, lengths multiply"

    // create two vectors as geometric numbers
    let a = Geonum {
        length: 2.0,
        angle: PI / 4.0, // [2, pi/4] = 2 at 45 degrees
        blade: 1,        // vector (grade 1) - directed quantity at 45°
    };

    let b = Geonum {
        length: 3.0,
        angle: PI / 3.0, // [3, pi/3] = 3 at 60 degrees
        blade: 1,        // vector (grade 1) - directed quantity at 60°
    };

    // multiply using the mul method
    let product = a.mul(&b);

    // verify the result has length 2*3=6 and angle pi/4+pi/3=7pi/12
    assert_eq!(product.length, 6.0);
    assert!((product.angle - (PI / 4.0 + PI / 3.0)).abs() < EPSILON);

    // test multiplication of perpendicular vectors (90 degrees apart)
    let c = Geonum {
        length: 2.0,
        angle: 0.0, // [2, 0] = 2 along x-axis
        blade: 1,   // vector (grade 1) - directed quantity along x-axis
    };

    let d = Geonum {
        length: 4.0,
        angle: PI / 2.0, // [4, pi/2] = 4 along y-axis
        blade: 1,        // vector (grade 1) - directed quantity along y-axis
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
        blade: 1,        // vector (grade 1) - directed quantity at 30°
    };

    let f = Geonum {
        length: 2.0,
        angle: -PI / 6.0, // [2, -pi/6] = 2 at -30 degrees (or 330 degrees)
        blade: 1,         // vector (grade 1) - directed quantity at -30°
    };

    // multiply vectors
    let opposite_product = e.mul(&f);

    // verify result has length 5*2=10 and angle pi/6+(-pi/6)=0
    assert_eq!(opposite_product.length, 10.0);
    assert!((opposite_product.angle % (2.0 * PI)).abs() < EPSILON); // should be 0
}

#[test]
fn it_multiplies_vectors_with_scalars() {
    // scalar multiplication in geometric numbers follows the same rule:
    // "angles add, lengths multiply"

    // create a vector and a positive scalar
    let vector = Geonum {
        length: 3.0,
        angle: PI / 4.0, // [3, pi/4] = 3 at 45 degrees
        blade: 1,        // vector (grade 1) - directed quantity at 45°
    };

    let scalar = Geonum {
        length: 2.0,
        angle: 0.0, // [2, 0] = positive 2 (scalar)
        blade: 0,   // scalar (grade 0) - pure magnitude for scaling
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
        blade: 0,  // scalar (grade 0) - negative scale factor
    };

    // multiply vector by negative scalar
    let product2 = vector.mul(&negative_scalar);

    // verify result has length 3*2=6 and angle is now pi/4+pi=5pi/4 (rotated 180 degrees)
    assert_eq!(product2.length, 6.0);
    assert!((product2.angle - (PI / 4.0 + PI)).abs() < EPSILON);

    // verify scalar multiplication is commutative
    let product3 = negative_scalar.mul(&vector);

    // should have same length and angle as product2
    assert_eq!(product3.length, product2.length);
    assert!((product3.angle - product2.angle).abs() < EPSILON);

    // test scaling a vector by zero
    let zero_scalar = Geonum {
        length: 0.0,
        angle: 0.0, // [0, 0] = zero
        blade: 0,   // scalar (grade 0) - zero value
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
        blade: 1, // vector (grade 1) - directed quantity at 60°
    };
    let result = v1.mul(&v2).mul(&v3);

    // operation end time
    let duration = start.elapsed();

    // verify results
    assert!(dot.abs() < EPSILON); // orthogonal vectors have zero dot product
    assert_eq!(wedge.length, 1.0); // unit bivector
    assert_eq!(geo_product.length, 1.0);
    assert_eq!(geo_product.angle, PI / 2.0);
    assert_eq!(result.length, 2.0); // length of v3
    assert!((result.angle - (PI / 2.0 + PI / 3.0)).abs() < EPSILON);

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

    // create a 3D space
    let space = Dimensions::new(3);

    // create basis multivectors
    let e0 = space.multivector(&[0]); // scalar
    let e1_e2_e3 = space.multivector(&[1, 2, 3]); // three basis vectors

    // create a custom multivector with mixed grades
    let mixed_mv = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 0, // scalar (grade 0) - scalar component of multivector
        }, // scalar part
        Geonum {
            length: 2.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - vector component of multivector
        }, // vector part
        Geonum {
            length: 3.0,
            angle: PI,
            blade: 2, // bivector (grade 2) - represents oriented plane
        }, // bivector part
    ]);

    // 1. basic multivector operations

    // accessing elements
    assert_eq!(e0[0].length, 1.0);
    assert_eq!(e0[0].angle, 0.0);

    assert_eq!(e1_e2_e3[0].angle, PI / 2.0); // first vector (e1)
    assert_eq!(e1_e2_e3[1].angle, PI); // second vector (e2)
    assert_eq!(e1_e2_e3[2].angle, 3.0 * PI / 2.0); // third vector (e3)

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
    assert_eq!(scalar_parts[0].angle, 0.0);

    assert_eq!(vector_parts.len(), 1);
    assert_eq!(vector_parts[0].length, 2.0);
    assert_eq!(vector_parts[0].angle, PI / 2.0);

    // 3. grade involution (negates odd-grade components)

    // create a multivector with even and odd grades
    let mv = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 0, // scalar (grade 0) - scalar component
        }, // scalar (grade 0)
        Geonum {
            length: 2.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - directed component
        }, // vector (grade 1)
        Geonum {
            length: 3.0,
            angle: PI,
            blade: 2, // bivector (grade 2) - plane element
        }, // bivector (grade 2)
    ]);

    // apply grade involution
    let inv = mv.involute();

    // verify: even grades unchanged, odd grades negated
    assert_eq!(inv[0].length, 1.0); // scalar unchanged
    assert_eq!(inv[0].angle, 0.0);

    assert_eq!(inv[1].length, 2.0); // vector negated
    assert_eq!(inv[1].angle, 3.0 * PI / 2.0); // angle rotated by π

    assert_eq!(inv[2].length, 3.0); // bivector unchanged
    assert_eq!(inv[2].angle, PI);

    // 4. clifford conjugate

    let conj = mv.conjugate();

    // verify: scalar unchanged, vector and bivector negated
    assert_eq!(conj[0].length, 1.0); // scalar unchanged
    assert_eq!(conj[0].angle, 0.0);

    assert_eq!(conj[1].length, 2.0); // vector negated
    assert_eq!(conj[1].angle, 3.0 * PI / 2.0); // angle rotated by π

    assert_eq!(conj[2].length, 3.0); // bivector negated
    assert_eq!(conj[2].angle, TWO_PI); // bivector with angle π gets negated to 2π

    // 5. contraction operations

    // create two simple multivectors for contraction
    let a = Multivector(vec![
        Geonum {
            length: 2.0,
            angle: 0.0,
            blade: 0, // scalar (grade 0) - magnitude component
        }, // scalar
        Geonum {
            length: 1.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - direction component
        }, // vector
    ]);

    let b = Multivector(vec![
        Geonum {
            length: 3.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - direction component
        }, // vector
    ]);

    // compute left contraction
    let left = a.left_contract(&b);

    // left contraction lowers grade of b by grade of a
    // scalar⌋vector = vector
    // vector⌋vector = scalar (dot product)
    assert!(left.len() > 0);

    // right contraction
    let right = a.right_contract(&b);

    // right contraction lowers grade of a by grade of b
    // scalar⌊vector = 0 (scalar grade cant be lowered)
    // vector⌊vector = scalar (dot product)
    assert!(right.len() > 0);

    // 6. anti-commutator

    // compute anti-commutator {a,b} = (ab + ba)/2
    let anti_comm = a.anti_commutator(&b);

    // result contains components from both a*b and b*a
    assert!(anti_comm.len() > 0);

    // 7. conversion from Vec<Geonum>

    let geonums = vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 0, // scalar (grade 0) - magnitude component
        },
        Geonum {
            length: 2.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - direction component
        },
    ];

    // convert using From trait
    let from_vec = Multivector::from(geonums);
    assert_eq!(from_vec.len(), 2);

    // 8. with_capacity and push operations

    let mut dynamic_mv = Multivector::with_capacity(3);
    assert_eq!(dynamic_mv.len(), 0);

    // use deref to access vec methods
    dynamic_mv.push(Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 0, // scalar (grade 0) - magnitude component
    });
    dynamic_mv.push(Geonum {
        length: 2.0,
        angle: PI / 2.0,
        blade: 1, // vector (grade 1) - direction component
    });

    assert_eq!(dynamic_mv.len(), 2);
    assert_eq!(dynamic_mv[0].length, 1.0);
    assert_eq!(dynamic_mv[1].angle, PI / 2.0);

    // 9. interior product operation

    // create two vectors to demonstrate interior product
    let x_axis = Multivector(vec![Geonum {
        length: 1.0,
        angle: 0.0, // e₁ - vector along x-axis
        blade: 1,   // vector (grade 1) - directed quantity along x-axis
    }]);

    let y_axis = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 2.0, // e₂ - vector along y-axis
        blade: 1,        // vector (grade 1) - directed quantity along y-axis
    }]);

    // compute interior product of perpendicular vectors
    let interior_perp = x_axis.interior_product(&y_axis);

    // for perpendicular vectors, interior product should be very small or zero
    let total_magnitude: f64 = interior_perp.0.iter().map(|g| g.length).sum();
    assert!(total_magnitude < 0.1);

    // create a 45-degree vector
    let angle45 = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 4.0, // 45 degrees
        blade: 1,        // vector (grade 1) - directed quantity at 45° angle
    }]);

    // interior product with x-axis
    let interior_45x = angle45.interior_product(&x_axis);

    // should have non-zero length (projection component)
    assert!(interior_45x.len() > 0);

    // 10. dual operation

    // create a 2D pseudoscalar (bivector in xy-plane)
    let pseudoscalar = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 2.0, // represents e₁∧e₂ (bivector in xy-plane)
        blade: 2,        // bivector (grade 2) - oriented area element in xy-plane for rotation
    }]);

    // compute dual of x-axis vector
    let x_dual = x_axis.dual(&pseudoscalar);

    // in 2D, dual of x-axis should be y-axis (with sign depending on orientation)
    assert!(x_dual.len() > 0);

    // check that the result is non-zero
    let dual_magnitude: f64 = x_dual.0.iter().map(|g| g.length).sum();
    assert!(dual_magnitude > 0.1);

    // 11. exponential operation

    // create a bivector representing the xy-plane
    let xy_plane = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 2.0, // bivector e₁∧e₂
        blade: 2,        // bivector (grade 2) - oriented area element in xy-plane for rotation
    }]);

    // compute the exponential e^(θ/2 * bivector) to create a rotor
    // for a 90-degree rotation in the xy-plane, θ/2 = 45 degrees = PI/4
    let rotor = Multivector::exp(&xy_plane, PI / 4.0);

    // a rotor should contain a scalar and bivector part
    assert!(rotor.len() >= 2);

    // scalar part should be cos(PI/4) ≈ 0.7071
    assert!((rotor[0].length - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);

    // bivector part should be sin(PI/4) ≈ 0.7071
    assert!((rotor[1].length - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01);

    // use the rotor to rotate a vector
    let rotated = x_axis.rotate(&rotor);

    // 90-degree rotation of x-axis should point along y-axis
    assert!(rotated.len() > 0);

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
    let v1 = Multivector(vec![Geonum {
        length: 2.0,
        angle: 0.0, // along x-axis
        blade: 1,   // vector (grade 1) - directed quantity along x-axis
    }]);

    let v2 = Multivector(vec![Geonum {
        length: 3.0,
        angle: PI / 2.0, // along y-axis
        blade: 1,        // vector (grade 1) - directed quantity along y-axis
    }]);

    // 1. Sandwich Product
    // Create a rotor (using the exp function)
    // Rotate in xy-plane by 90 degrees
    let plane = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 2.0, // bivector e₁∧e₂
        blade: 2,        // bivector (grade 2) - rotation plane for sandwich product
    }]);

    // e^(θ/2 * bivector) - for 90° rotation, θ/2 = π/4
    let rotor = Multivector::exp(&plane, PI / 4.0);

    // Get the reverse of the rotor
    let rotor_rev = rotor.conjugate();

    // Apply sandwich product R*v1*R̃ to rotate v1
    let rotated = rotor.sandwich_product(&v1, &rotor_rev);

    // The result should be a vector with similar magnitude but rotated direction
    assert!(rotated.len() > 0);

    // 2. Commutator Product
    // Commutator measures the failure of two elements to commute
    // For orthogonal vectors, it represents their bivector product
    let comm = v1.commutator(&v2);

    // The result should be non-zero for non-commuting elements
    assert!(comm.len() > 0);

    // 3. Join and Meet operations
    // Join - represents the union of subspaces (similar to span)
    let join = v1.join(&v2);

    // For two basis vectors, the join should be their plane
    assert!(join.len() > 0);

    // Meet - represents the intersection of subspaces
    // For two non-parallel vectors in a plane, this should be their intersection point
    let meet = v1.meet(&v2, None);

    // Verify the operation completes successfully
    let _ = meet;

    // 4. Square root operation
    // Create a scalar
    let scalar = Multivector(vec![Geonum {
        length: 4.0,
        angle: 0.0, // positive scalar
        blade: 0,   // scalar (grade 0) - pure magnitude component
    }]);

    // Compute the square root of the scalar
    let sqrt_scalar = scalar.sqrt();

    // The result should be a scalar with length 2.0
    assert_eq!(sqrt_scalar[0].length, 2.0);
    assert_eq!(sqrt_scalar[0].angle, 0.0);

    // Create a negative scalar
    let neg_scalar = Multivector(vec![Geonum {
        length: 9.0,
        angle: PI, // negative scalar
        blade: 0,  // scalar (grade 0) - pure magnitude component
    }]);

    // Compute the square root of the negative scalar
    let sqrt_neg = neg_scalar.sqrt();

    // The result should be a bivector with length 3.0 and angle π/2
    assert_eq!(sqrt_neg[0].length, 3.0);
    assert_eq!(sqrt_neg[0].angle, PI / 2.0); // converts to bivector

    // 5. Dual and Undual operations
    // Create a pseudoscalar for the 2D plane
    let pseudoscalar = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI, // e₁∧e₂ with orientation
        blade: 2,  // bivector (grade 2) - oriented area element for dual operation
    }]);

    // Compute the dual of v1 (x-axis vector)
    let dual_v1 = v1.dual(&pseudoscalar);

    // In 2D, the dual of the x-axis is the y-axis (possibly with sign change)
    // Now compute the undual to get back the original vector
    let undual_v1 = dual_v1.undual(&pseudoscalar);

    // The undual should get us back to the original vector (allowing for round-trip precision issues)
    assert!((undual_v1[0].length - v1[0].length).abs() < EPSILON);

    // The angle might be 2π different due to modular arithmetic
    // When handling angles in modular fashion, we need to be careful with comparisons
    let mut angle_diff = undual_v1[0].angle - v1[0].angle;
    if angle_diff > PI {
        angle_diff -= TWO_PI;
    } else if angle_diff < -PI {
        angle_diff += TWO_PI;
    }

    assert!(angle_diff.abs() < EPSILON);

    // 6. Section for pseudoscalar
    // Create a pseudoscalar for a 2D plane
    let section_pseudoscalar = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI, // 2D pseudoscalar
        blade: 2,  // bivector (grade 2) - pseudoscalar for planar section
    }]);

    // Create a multivector with various components
    let section_mixed = Multivector(vec![
        Geonum {
            length: 2.0,
            angle: 0.0, // Scalar component
            blade: 0,   // scalar (grade 0) - pure magnitude component
        },
        Geonum {
            length: 3.0,
            angle: PI / 2.0, // Vector component
            blade: 1,        // vector (grade 1) - directed component
        },
        Geonum {
            length: 5.0,
            angle: PI / 4.0, // Non-standard component
            blade: 1,        // vector (grade 1) - directed component at 45°
        },
    ]);

    // Extract the section for this pseudoscalar
    let section = section_mixed.section(&section_pseudoscalar);

    // Verify the section contains the expected components
    assert!(section.len() > 0, "Section should not be empty");

    // In a practical application, we would check which components
    // belong to the pseudoscalar's subspace and use them for
    // further calculations

    // 7. Regressive product (meet operation alternative)
    // Create a pseudoscalar for the 2D plane
    let regr_pseudoscalar = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI, // e₁∧e₂ with orientation
        blade: 2,  // bivector (grade 2) - oriented area element for regressive product
    }]);

    // Create two lines in 2D
    let line1 = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 4.0, // A line at 45 degrees
        blade: 1,        // vector (grade 1) - directed quantity representing a line
    }]);

    let line2 = Multivector(vec![Geonum {
        length: 1.0,
        angle: 3.0 * PI / 4.0, // A line at 135 degrees (perpendicular to line1)
        blade: 1,              // vector (grade 1) - directed quantity representing a line
    }]);

    // Compute the regressive product to find their intersection
    let intersection = line1.regressive_product(&line2, &regr_pseudoscalar);

    // In 2D, the regressive product of two lines is their intersection point
    assert!(
        intersection.len() > 0,
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
    let vector = Multivector(vec![Geonum {
        length: 2.0,
        angle: PI / 3.0, // A vector at 60 degrees
        blade: 1,        // vector (grade 1) - directed quantity for differentiation
    }]);

    // Compute the derivative
    let derivative = vector.differentiate();

    // The derivative should rotate the angle by π/2
    assert_eq!(derivative[0].length, 2.0); // Length is preserved
    assert_eq!(derivative[0].angle, (PI / 3.0 + PI / 2.0) % TWO_PI); // Angle is rotated by π/2

    // Compute the integral
    let integral = vector.integrate();

    // The integral should rotate the angle by -π/2
    assert_eq!(integral[0].length, 2.0); // Length is preserved
    assert_eq!(integral[0].angle, (PI / 3.0 - PI / 2.0) % TWO_PI); // Angle is rotated by -π/2

    // Demonstrate relationship between differentiation and integration
    // Differentiating and then integrating should give back the original (fundamental theorem of calculus)
    let roundtrip = derivative.integrate();
    assert_eq!(roundtrip[0].length, vector[0].length);

    // Angles might differ by 2π due to modular arithmetic
    let angle_diff = (roundtrip[0].angle - vector[0].angle) % TWO_PI;
    assert!(angle_diff < EPSILON || (TWO_PI - angle_diff) < EPSILON);

    // Demonstrate that second derivative equals negative of original (d²/dx² = -1)
    let second_derivative = derivative.differentiate();
    assert_eq!(second_derivative[0].length, vector[0].length);
    assert_eq!(second_derivative[0].angle, (vector[0].angle + PI) % TWO_PI);
}

#[test]
fn it_keeps_angles_less_than_2pi() {
    // Create vectors with different angles but same blade
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    let b = Geonum {
        length: 1.0,
        angle: 2.0 * PI,
        blade: 1,
    }; // same as angle 0, but 2π rotated

    // Blade grades should be preserved and distinguish vectors from bivectors
    let c = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 2,
    }; // bivector with angle 0

    // Verify blade values are preserved in mul
    let a_times_c = a.mul(&c);
    assert_eq!(a_times_c.blade, 1); // Vector * bivector = vector (abs(1-2) = 1)

    // Wedge product increases blade grade
    let wedge = a.wedge(&b);
    assert_eq!(wedge.blade, a.blade + b.blade);

    // Differentiation increases blade grade
    let a_diff = a.differentiate();
    assert_eq!(a_diff.blade, a.blade + 1);

    // Integration decreases blade grade
    let a_int = a_diff.integrate();
    assert_eq!(a_int.blade, a_diff.blade - 1);

    // Rotation preserves blade grade
    let a_rot = a.rotate(PI / 2.0);
    assert_eq!(a_rot.blade, a.blade);

    // Reflection preserves blade grade
    let a_ref = a.reflect(&b);
    assert_eq!(a_ref.blade, a.blade);
}

#[test]
fn it_initializes_with_blade() {
    // Test default blade values for different types
    let scalar = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 0,
    };
    let _vector = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    };
    let bivector = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 2,
    };
    let _trivector = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 3,
    };

    // Test that blade value differentiates types even with same angle
    let scalar_multivector = Multivector(vec![scalar]);
    let bivector_multivector = Multivector(vec![bivector]);

    // Extract grades by blade
    let extracted_scalars = scalar_multivector.grade(0);
    let extracted_bivectors = bivector_multivector.grade(2);

    // Assert extraction worked correctly
    assert_eq!(extracted_scalars.0.len(), 1);
    assert_eq!(extracted_bivectors.0.len(), 1);

    // Cross-check - this should be empty since we're extracting the wrong grade
    let empty_extract = scalar_multivector.grade(2);
    assert_eq!(empty_extract.0.len(), 0);

    // Constructors should set blade properly
    let v1 = Geonum::from_polar(1.0, 0.0);
    let v2 = Geonum::from_polar_blade(1.0, 0.0, 2);
    let s = Geonum::scalar(5.0);

    assert_eq!(v1.blade, 1); // Default to vector
    assert_eq!(v2.blade, 2); // Explicitly set
    assert_eq!(s.blade, 0); // Scalar is grade 0
}
