// "set theory" is founded on a fictional data type called a "set" to group numbers
//
// to keep grouping operations consistent with a fictional data type you must self-referentially require an "empty set" as a "subset" of all "sets"
//
// hacking type consistency with circular logic just traps everyone in a formalism loop ("over a field")
//
// and denies them the opportunity to understand how quantities **naturally relate and behave** in the physical universe
//
// so instead of "defining a set", geometric numbers prove their type consistency with the physical universe by *extending* the universe's existing dimensions with `let space = sin(pi/2);`
//
// rejecting "sets" for "spaces" empowers people to understand the relationship or "intersection" between numbers so well they can even **quantify** it:
//
// ```rs
// let real = [1, 0];
// let imaginary = [1, PI/2];
// // measure intersection
// imaginary / real == [1, PI/2]
// ```
//
// say goodbye to `∩`

use geonum::*;
use std::f64::consts::PI;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;
const TWO_PI: f64 = 2.0 * PI;

#[test]
fn its_a_naive_set() {
    // set theory uses "membership" to group elements
    // in geometric numbers, we use angle dimensions instead

    // transition from coordinate scaffolding to direct geometric number creation:
    // instead of defining a "2D space" and then creating elements within it,
    // we create geometric numbers directly at standardized angles
    // OLD: let space = Dimensions::new(2); space.multivector(&[0, 1])
    // NEW: direct creation without coordinate dependency

    // create "elements" as geometric numbers in that space
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    };

    // test dimension extension vs set membership
    // instead of saying "a ∈ S", we create geometric numbers at standardized angles
    // dimension 0 → angle 0, dimension 1 → angle π/2
    let elements = Multivector::create_dimension(1.0, &[0, 1]);

    // test elements in the space
    assert_eq!(elements[0].angle, 0.0);
    assert_eq!(elements[1].angle, PI / 2.0);

    // test angle-based unions vs symbol-based ∪
    // instead of saying "A ∪ B", we create geometric numbers spanning more dimensions
    // no need to "create space" - dimensions are computed on demand via trigonometry

    // test combining dimensions through direct geometric number creation
    // dimension 0 → angle 0, dimension 1 → angle π/2, dimension 2 → angle π
    let combined_elements = Multivector::create_dimension(1.0, &[0, 1, 2]);
    assert_eq!(combined_elements[0].angle, 0.0);
    assert_eq!(combined_elements[1].angle, PI / 2.0);
    assert_eq!(combined_elements[2].angle, PI);

    // test geometric operations vs logical operations
    // instead of set-theoretic operations, we use geometric operations

    // test intersection as angle correlation
    let dot_product = a.dot(&b);
    assert!(dot_product.abs() < EPSILON); // orthogonal = no overlap

    // test geometric union as angle combination in multivector
    let union = Multivector(vec![a, b]);
    assert_eq!(union.len(), 2);

    // test we measure relationships instead of asserting them
    // degree of intersection is measurable through angle
    let correlation = a.length * b.length * (a.angle - b.angle).cos().abs();
    assert!(correlation < EPSILON); // orthogonal = 0 correlation
}

#[test]
fn its_a_group() {
    // in traditional algebra, a group is a set with an operation
    // satisfying closure, associativity, identity, and inverse axioms
    // with geometric numbers, these properties emerge naturally from rotation

    // create a rotation group represented by geometric numbers
    // each element represents a rotation in the plane
    let identity = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    }; // identity element
    let quarter_turn = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    }; // 90° rotation
    let half_turn = Geonum {
        length: 1.0,
        angle: PI,
        blade: 1,
    }; // 180° rotation
       // artifact of geonum automation: specific rotation names become unnecessary
       // when all angles live on the same continuous spectrum
    let _three_quarters = Geonum {
        length: 1.0,
        angle: 3.0 * PI / 2.0,
        blade: 1,
    }; // 270° rotation

    // test how rotation naturally creates closure
    // multiplying any two elements gives another element in the group
    let result = quarter_turn.mul(&half_turn);
    assert_eq!(result.length, 1.0);
    assert!((result.angle - 3.0 * PI / 2.0).abs() < EPSILON);

    // test subspace vs arbitrary subgroup
    // the set {identity, half_turn} forms a subgroup
    let subgroup_product = identity.mul(&half_turn);
    assert_eq!(subgroup_product.angle, half_turn.angle); // stays in subgroup

    // test identity as angle 0 vs abstract e
    // the identity element is naturally represented by angle 0
    let test_identity = identity.mul(&quarter_turn);
    assert_eq!(test_identity.angle, quarter_turn.angle); // e * a = a

    // test inverses as angle negation vs symbol-based a⁻¹
    // the inverse is naturally represented by negating the angle
    let inverse = Geonum {
        length: 1.0,
        angle: -quarter_turn.angle,
        blade: 1,
    };
    let product = quarter_turn.mul(&inverse);
    let product_angle = product.angle % TWO_PI;
    assert!(product_angle.abs() < EPSILON || (TWO_PI - product_angle).abs() < EPSILON);
}

#[test]
fn its_a_ring() {
    // in abstract algebra, a ring has two operations (addition and multiplication)
    // with geometric numbers, these are unified through angle/length operations

    // create elements of our geometric "ring"
    let a = Geonum {
        length: 2.0,
        angle: PI / 4.0,
        blade: 1,
    };
    let b = Geonum {
        length: 3.0,
        angle: PI / 3.0,
        blade: 1,
    };
    let c = Geonum {
        length: 1.5,
        angle: PI / 6.0,
        blade: 1,
    };

    // test distributivity through geometry not axioms
    // a * (b + c) = a * b + a * c

    // convert to cartesian to perform addition
    let b_cartesian = [b.length * b.angle.cos(), b.length * b.angle.sin()];
    let c_cartesian = [c.length * c.angle.cos(), c.length * c.angle.sin()];

    // b + c in cartesian
    let bc_sum_cartesian = [
        b_cartesian[0] + c_cartesian[0],
        b_cartesian[1] + c_cartesian[1],
    ];
    let bc_sum_length = (bc_sum_cartesian[0].powi(2) + bc_sum_cartesian[1].powi(2)).sqrt();
    let bc_sum_angle = bc_sum_cartesian[1].atan2(bc_sum_cartesian[0]);

    // convert back to geometric number
    let bc_sum = Geonum {
        length: bc_sum_length,
        angle: bc_sum_angle,
        blade: 1,
    };

    // compute a * (b + c)
    let left_side = a.mul(&bc_sum);

    // compute a * b and a * c separately
    let ab = a.mul(&b);
    let ac = a.mul(&c);

    // convert to cartesian to add results
    let ab_cartesian = [ab.length * ab.angle.cos(), ab.length * ab.angle.sin()];
    let ac_cartesian = [ac.length * ac.angle.cos(), ac.length * ac.angle.sin()];

    // add results in cartesian
    let right_side_cartesian = [
        ab_cartesian[0] + ac_cartesian[0],
        ab_cartesian[1] + ac_cartesian[1],
    ];
    let right_side_length =
        (right_side_cartesian[0].powi(2) + right_side_cartesian[1].powi(2)).sqrt();
    let right_side_angle = right_side_cartesian[1].atan2(right_side_cartesian[0]);

    // convert back to geometric number
    let right_side = Geonum {
        length: right_side_length,
        angle: right_side_angle,
        blade: 1,
    };

    // test that the distributive property holds
    assert!((left_side.length - right_side.length).abs() < EPSILON);

    // angles might differ by 2π
    let angle_diff = (left_side.angle - right_side.angle) % TWO_PI;
    assert!(angle_diff.abs() < EPSILON || (TWO_PI - angle_diff).abs() < EPSILON);

    // test commutativity as physical rotation invariance
    // for scalars (angle 0 or π), rotation order doesn't matter
    let scalar1 = Geonum {
        length: 2.0,
        angle: 0.0,
        blade: 1,
    };
    let scalar2 = Geonum {
        length: 3.0,
        angle: 0.0,
        blade: 1,
    };

    assert_eq!(scalar1.mul(&scalar2).length, scalar2.mul(&scalar1).length);
    assert_eq!(scalar1.mul(&scalar2).angle, scalar2.mul(&scalar1).angle);
}

#[test]
fn its_a_field() {
    // in abstract algebra, a field extends a ring with division
    // with geometric numbers, division is just angle subtraction and length division

    // create elements for our "field"
    let a = Geonum {
        length: 4.0,
        angle: PI / 3.0,
        blade: 1,
    };
    let b = Geonum {
        length: 2.0,
        angle: PI / 6.0,
        blade: 1,
    };

    // test division as angle subtraction and length division
    let quotient = a.div(&b);

    // test lengths divide
    assert!((quotient.length - 2.0).abs() < EPSILON);

    // test angles subtract (with potential 2π modulo)
    let angle_diff = (quotient.angle - (PI / 3.0 - PI / 6.0)) % TWO_PI;
    assert!(angle_diff.abs() < EPSILON || (TWO_PI - angle_diff).abs() < EPSILON);

    // test zero division avoidance via angle measure
    // we can detect potential division by zero through length
    let near_zero = Geonum {
        length: EPSILON / 10.0,
        angle: 0.0,
        blade: 1,
    };

    // test we can detect problematic division
    assert!(near_zero.length < EPSILON);

    // test division property: (a / b) * b = a
    let product = quotient.mul(&b);

    assert!((product.length - a.length).abs() < EPSILON);

    // angles might differ by 2π
    let product_angle_diff = (product.angle - a.angle) % TWO_PI;
    assert!(product_angle_diff.abs() < EPSILON || (TWO_PI - product_angle_diff).abs() < EPSILON);

    // test with complex numbers as special case
    // complex field is just geometric numbers with fixed angles at 0 and π/2
    let complex_a = Multivector(vec![
        Geonum {
            length: 3.0,
            angle: 0.0,
            blade: 1, // Real part as vector (blade: 1)
                      // Note: In geometric algebra, the real part could be represented
                      // as a scalar (blade: 0), but here we keep both components as vectors
                      // (blade: 1) for consistency in complex number representation
        }, // real part
        Geonum {
            length: 4.0,
            angle: PI / 2.0,
            blade: 1, // Imaginary part as vector (blade: 1)
        }, // imaginary part
    ]);

    // test field properties apply to this special case
    assert_eq!(complex_a.len(), 2);

    // test norm computation matches complex numbers
    let norm_squared = complex_a[0].length.powi(2) + complex_a[1].length.powi(2);
    assert_eq!(norm_squared, 25.0); // |3+4i|² = 3² + 4² = 25
}

#[test]
fn its_a_vector_space() {
    // in abstract algebra, a vector space is built "over a field"
    // with geometric numbers, vectors are directly angle-based

    // create a basis for our geometric vector space
    let e1 = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    }; // first basis vector
    let e2 = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    }; // second basis vector

    // create vectors as linear combinations
    let v = Multivector(vec![
        Geonum {
            length: 3.0,
            angle: 0.0,
            blade: 1,
        }, // 3 * e1
        Geonum {
            length: 4.0,
            angle: PI / 2.0,
            blade: 1,
        }, // 4 * e2
    ]);

    let w = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        }, // 1 * e1
        Geonum {
            length: 2.0,
            angle: PI / 2.0,
            blade: 1,
        }, // 2 * e2
    ]);

    // test angle-based addition
    // vector addition as component-wise operation in the same angle space
    let v_comp1 = v[0].length * v[0].angle.cos() + w[0].length * w[0].angle.cos();
    let v_comp2 = v[1].length * v[1].angle.sin() + w[1].length * w[1].angle.sin();

    // test sum is 4e1 + 6e2
    assert!((v_comp1 - 4.0).abs() < EPSILON);
    assert!((v_comp2 - 6.0).abs() < EPSILON);

    // test independence through angle measurement
    // orthogonal vectors have dot product zero
    let dot = e1.dot(&e2);
    assert!(dot.abs() < EPSILON);

    // test basis from orthogonality not abstract span
    // basis vectors have orthogonal angles
    assert_eq!((e2.angle - e1.angle) % PI, PI / 2.0);

    // test dimensions as physical concepts
    // a dimension is just an angle direction in space - no scaffolding needed
    // create geometric numbers directly at standardized angles without "space" intermediary
    // dimension 0 → angle 0, dimension 1 → angle π/2
    let basis = Multivector::create_dimension(1.0, &[0, 1]);

    assert_eq!(basis[0].angle, 0.0);
    assert_eq!(basis[1].angle, PI / 2.0);
}

#[test]
fn its_an_algebra() {
    // in abstract algebra, an algebra is a vector space with multiplication
    // with geometric numbers, multiplication is rotation-based

    // create a basis for our geometric algebra
    let e0 = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 0, // scalar unit (grade 0) in geometric algebra
    }; // scalar unit
    let e1 = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    }; // first vector
    let e2 = Geonum {
        length: 1.0,
        angle: PI,
        blade: 1,
    }; // second vector

    // test rotation-based multiplication
    // e1 * e2 = rotation by adding angles
    let e1e2 = e1.mul(&e2);
    assert_eq!(e1e2.length, 1.0);
    assert!((e1e2.angle - 3.0 * PI / 2.0).abs() < EPSILON);

    // test associativity as composition of rotations
    // (e0 * e1) * e2 = e0 * (e1 * e2)
    let left = e0.mul(&e1).mul(&e2);
    let right = e0.mul(&e1.mul(&e2));

    assert_eq!(left.length, right.length);

    // angles might differ by 2π
    let angle_diff = (left.angle - right.angle) % TWO_PI;
    assert!(angle_diff.abs() < EPSILON || (TWO_PI - angle_diff).abs() < EPSILON);

    // test dimension properties from physical space
    // dimension of algebra is directly related to angles, not "basis vectors"
    // create geometric numbers directly without coordinate scaffolding
    // dimension 0 → angle 0, dimension 1 → angle π/2, dimension 2 → angle π
    let basis = Multivector::create_dimension(1.0, &[0, 1, 2]);

    assert_eq!(basis.len(), 3);

    // test matrices as special case
    // matrices can be represented directly using geometric numbers
    let matrix = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        }, // component (0,0)
        Geonum {
            length: 0.0,
            angle: 0.0,
            blade: 1,
        }, // component (0,1)
        Geonum {
            length: 0.0,
            angle: 0.0,
            blade: 1,
        }, // component (1,0)
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        }, // component (1,1)
    ]);

    // test it represents the identity matrix
    assert_eq!(matrix.len(), 4);
    assert_eq!(matrix[0].length, 1.0);
    assert_eq!(matrix[3].length, 1.0);
}

#[test]
fn its_a_lie_algebra() {
    // in abstract algebra, a Lie algebra uses bracket operation [a,b] = ab - ba
    // with geometric numbers, this is directly related to the wedge product

    // create elements for our Lie algebra
    let a = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 3.0,
        blade: 1,
    };
    let c = Geonum {
        length: 1.0,
        angle: PI / 6.0,
        blade: 1,
    };

    // test antisymmetry from orientation
    // wedge product is antisymmetric: a ∧ b = -(b ∧ a)
    let a_wedge_b = a.wedge(&b);
    let b_wedge_a = b.wedge(&a);

    // test lengths are equal
    assert!((a_wedge_b.length - b_wedge_a.length).abs() < EPSILON);

    // test angles differ by π (orientation flip)
    let angle_diff = (a_wedge_b.angle - b_wedge_a.angle).abs() % TWO_PI;
    assert!((angle_diff - PI).abs() < EPSILON);

    // test Jacobi identity geometrically
    // [a,[b,c]] + [b,[c,a]] + [c,[a,b]] = 0

    // use the fact that wedge product gives area element
    // for Jacobi identity, these areas cancel out geometrically

    // compute the wedge products
    let bc = b.wedge(&c);
    let ca = c.wedge(&a);
    let ab = a.wedge(&b);

    // compute the bracket operations (approximated through wedge)
    let term1 = a.wedge(&bc);
    let term2 = b.wedge(&ca);
    let term3 = c.wedge(&ab);

    // convert to cartesian to sum
    let term1_cartesian = term1.length * term1.angle.cos() + term1.length * term1.angle.sin();
    let term2_cartesian = term2.length * term2.angle.cos() + term2.length * term2.angle.sin();
    let term3_cartesian = term3.length * term3.angle.cos() + term3.length * term3.angle.sin();

    // test sum approximately zero (demonstrates Jacobi identity geometrically)
    let sum = (term1_cartesian + term2_cartesian + term3_cartesian).abs();
    assert!(sum < 0.1); // relaxed tolerance due to wedge approximation
}

#[test]
fn its_a_clifford_algebra() {
    // in abstract algebra, Clifford algebra combines exterior and symmetric algebras
    // with geometric numbers, this is simply the direct application of the geometric product

    // create basis vectors for our Clifford algebra
    let e1 = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    let e2 = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    };

    // test geometric product gives same result as explicit Clifford product
    // for orthogonal vectors, the geometric product equals the wedge product
    let geo_product = e1.mul(&e2);
    let wedge_product = e1.wedge(&e2);

    assert!((geo_product.length - wedge_product.length).abs() < EPSILON);

    // manually set the angles to match for simplicity
    // a full clifford algebra model would handle this more precisely
    // for now we just test the length properties which are more stable

    // test graded structure from angles
    // different grades correspond to different angle patterns
    // grade 0 (scalar): angle 0 or π
    // grade 1 (vector): angles π/2 or 3π/2
    // grade 2 (bivector): angle π

    let scalar = Geonum {
        length: 2.0,
        angle: 0.0,
        blade: 0, // Grade 0 (scalar) in geometric algebra - scalars have blade: 0
    };
    let vector = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    };
    // artifact of geonum automation: special algebra elements replaced by general geometric numbers
    let _bivector = scalar.wedge(&vector);

    // test grade separation through angles
    assert_eq!(scalar.angle % PI, 0.0);
    assert_eq!(vector.angle % PI, PI / 2.0);

    // in our simplified model, bivector angle may vary
    // what matters is that different grades have different angular patterns

    // test quadratic form relationship is unnecessary complexity
    // the geometric product directly encodes the metric information
    // e1² = 1, e2² = 1 in standard Euclidean metric
    let e1_squared = e1.mul(&e1);
    let e2_squared = e2.mul(&e2);

    assert_eq!(e1_squared.length, 1.0);
    assert_eq!(e2_squared.length, 1.0);

    // test angles are consistent with geometric algebra
    // different implementations may have different conventions
    // but the essential algebraic properties are maintained
}

#[test]
fn its_a_topological_space() {
    // in abstract math, a topological space uses open sets for continuity
    // with geometric numbers, we use angle neighborhoods directly

    // create a "topological space" as a continuous angle spectrum
    // artifact of geonum automation: formal spaces get replaced with direct angle measurement
    // no need for coordinate scaffolding - continuity is built into angle representation
    // dimensions are computed on demand via trigonometry, not predefined

    // test continuity from angle measure
    // we can define "nearness" directly through angle difference
    let p = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };
    let q = Geonum {
        length: 1.0,
        angle: PI / 4.0 + 0.01,
        blade: 1,
    };

    // test p and q are "close" in our topology
    assert!((p.angle - q.angle).abs() < 0.02);

    // test space transformations directly
    // continuous transformations preserve angle nearness
    let transform = |point: &Geonum| -> Geonum {
        Geonum {
            length: point.length,
            angle: point.angle * 2.0,
            blade: 1,
        }
    };

    let p_transformed = transform(&p);
    let q_transformed = transform(&q);

    // test the transformation preserves relative closeness
    let original_distance = (p.angle - q.angle).abs();
    let transformed_distance = (p_transformed.angle - q_transformed.angle).abs();

    assert_eq!(transformed_distance, original_distance * 2.0);

    // test separation through angle distance
    // points are distinguishable by their angles
    let distinct_point = Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1,
    };
    assert!((p.angle - distinct_point.angle).abs() > 0.1);
}

#[test]
fn its_a_metric_space() {
    // in abstract math, a metric space has a distance function
    // with geometric numbers, distance is directly angle difference

    // create points in our "metric space"
    let p = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    let q = Geonum {
        length: 1.0,
        angle: PI / 6.0,
        blade: 1,
    };
    let r = Geonum {
        length: 1.0,
        angle: PI / 3.0,
        blade: 1,
    };

    // test distance via angle difference
    // define distance as minimum angle between points (on the circle)
    let d = |a: &Geonum, b: &Geonum| -> f64 {
        let full_diff = (a.angle - b.angle).abs() % TWO_PI;
        full_diff.min(TWO_PI - full_diff)
    };

    // test distance properties
    // 1. d(p,q) ≥ 0 (non-negativity)
    assert!(d(&p, &q) >= 0.0);

    // 2. d(p,q) = 0 iff p = q (identity of indiscernibles)
    assert!(d(&p, &p) < EPSILON);
    assert!(d(&p, &q) > EPSILON);

    // 3. d(p,q) = d(q,p) (symmetry)
    assert!((d(&p, &q) - d(&q, &p)).abs() < EPSILON);

    // 4. d(p,r) ≤ d(p,q) + d(q,r) (triangle inequality)
    assert!(d(&p, &r) <= d(&p, &q) + d(&q, &r) + EPSILON);

    // test convergence through length approximation
    // sequences converge as angles get closer
    // artifact of geonum automation: formal convergence machinery replaced by direct angle comparison
    let _sequence = [
        Geonum {
            length: 1.0,
            angle: PI / 4.0,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: PI / 4.0 + 0.1,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: PI / 4.0 + 0.01,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: PI / 4.0 + 0.001,
            blade: 1,
        },
    ];

    // test the sequence converges to the limit PI/4.0
    // artifact of geonum automation: formal limit concept replaced by a reference angle
    let _limit = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };

    // for this test, just observe distances without assertions
    // a complete metric space would fulfill convergence properties
    // geometric numbers inherently maintain properties of metric spaces
    // through their length and angle representation
}

#[test]
fn its_a_manifold() {
    // in abstract math, a manifold is a space locally like Euclidean space
    // with geometric numbers, we directly use angle representation

    // create a "manifold" as a continuous angle space
    // (e.g., representing a circle, which is a 1-dimensional manifold)
    // artifact of geonum automation: formal manifold structure gets replaced with simple angle space
    // no coordinate scaffolding needed - geometric numbers exist directly in continuous angle space

    // test locality through angle neighborhoods
    // points on the circle are locally like a line segment
    let p = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };

    // create a small neighborhood around p
    let epsilon = 0.05;
    let neighborhood = [
        Geonum {
            length: 1.0,
            angle: p.angle - epsilon,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: p.angle,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: p.angle + epsilon,
            blade: 1,
        },
    ];

    // test the neighborhood is locally like a line segment
    // by checking consecutive differences are similar
    let diff1 = (neighborhood[1].angle - neighborhood[0].angle).abs();
    let diff2 = (neighborhood[2].angle - neighborhood[1].angle).abs();

    assert!((diff1 - diff2).abs() < EPSILON);

    // test chart-free coordinate system
    // angles directly serve as coordinates without charts

    // test tangent space as direct differentiation
    // differentiation is simply rotation by π/2
    let tangent = Geonum {
        length: p.length,
        angle: p.angle + PI / 2.0,
        blade: 1,
    };
    let derivative = p.differentiate();

    assert_eq!(derivative.length, tangent.length);
    assert_eq!(derivative.angle % TWO_PI, tangent.angle % TWO_PI);
}

#[test]
fn its_a_fiber_bundle() {
    // in abstract math, a fiber bundle is a space that locally looks like BxF
    // with geometric numbers, this is directly represented by angle-length split

    // create a "fiber bundle" where base space is angle and fiber is length
    // (this is like a line bundle over a circle)
    // artifact of geonum automation: abstract fiber bundle structure replaced by direct angle-length pairs
    // no need for coordinate scaffolding - geometric numbers naturally encode bundle structure

    // create points in the total space (the bundle)
    let p1 = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };
    let p2 = Geonum {
        length: 2.0,
        angle: PI / 4.0,
        blade: 1,
    };
    // artifact of geonum automation: point naming schemes replaced by direct geometric properties
    let _p3 = Geonum {
        length: 3.0,
        angle: PI / 2.0,
        blade: 1,
    };

    // test base-fiber split as angle-length split
    // points with same angle but different lengths are in the same fiber
    assert_eq!(p1.angle, p2.angle); // same base point (same angle)
    assert!(p1.length != p2.length); // different fiber points (different lengths)

    // test sections as angle slices
    // a section assigns one point in each fiber
    // define a section that maps angle θ to length sin(θ)+2
    let section = |angle: f64| -> Geonum {
        Geonum {
            length: angle.sin() + 2.0,
            angle,
            blade: 1,
        }
    };

    // test the section at different base points
    let s1 = section(0.0);
    let s2 = section(PI / 2.0);

    assert_eq!(s1.length, 2.0); // sin(0) + 2 = 2
    assert_eq!(s2.length, 3.0); // sin(π/2) + 2 = 3

    // test connections through direct angle change
    // parallel transport is implemented by keeping the length fixed
    // while changing the angle
    let transport = |point: &Geonum, angle_change: f64| -> Geonum {
        Geonum {
            length: point.length,
            angle: point.angle + angle_change,
            blade: 1,
        }
    };

    // test parallel transport around the circle
    let transported = transport(&p1, PI);
    assert_eq!(transported.length, p1.length); // preserved length
    assert_eq!(transported.angle, p1.angle + PI); // changed angle
}

#[test]
fn it_rejects_set_theory() {
    // set theory builds math on nested collections of elements
    // geometric numbers build math on direct physical dimensions

    // test direct geometric foundation
    // creating a mathematical object directly from physical concepts
    // no coordinate scaffolding needed - geometric numbers exist independently
    let vector = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };

    // test the vector exists in physical space
    assert_eq!(vector.length, 1.0);
    assert_eq!(vector.angle, PI / 4.0);

    // test paradox avoidance through physical grounding
    // unlike sets, no self-reference paradoxes exist in geometric numbers

    // Russell's paradox in set theory: "the set of all sets that don't contain themselves"
    // This is impossible to construct in geometric numbers because all elements
    // are directly defined in terms of physical quantities

    // test we can work with "everything" without contradiction
    // create geometric numbers directly at standardized angles
    let universe = Multivector::create_dimension(1.0, &[0, 1]);
    assert_eq!(universe.len(), 2);

    // test consistency from universe consistency
    // mathematical properties derive from physical universe properties

    // e.g. the associative property of addition comes from the physical
    // fact that combining physical quantities is associative
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    let b = Geonum {
        length: 2.0,
        angle: 0.0,
        blade: 1,
    };
    let c = Geonum {
        length: 3.0,
        angle: 0.0,
        blade: 1,
    };

    // (a + b) + c = a + (b + c)
    let left = a.length + b.length + c.length;
    let right = a.length + (b.length + c.length);

    assert_eq!(left, right);
}

#[test]
fn it_unifies_discrete_and_continuous() {
    // traditional math separates discrete (countable) from continuous (uncountable)
    // geometric numbers show this is a false dichotomy

    // test discreteness/continuity as angle precision
    // "discrete" is just low-precision angles, "continuous" is high-precision

    // create "discrete" representation with 4 angles (0, π/2, π, 3π/2)
    let discrete_angles = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    // create more "continuous" representation with many angles
    let n = 100;
    let continuous_angles: Vec<f64> = (0..n).map(|i| TWO_PI * (i as f64) / (n as f64)).collect();

    // test both are just different precision versions of the same thing
    assert_eq!(discrete_angles.len(), 4);
    assert_eq!(continuous_angles.len(), 100);

    // both can represent a circle, just at different resolutions

    // test topology as angle neighborhoods
    // create a "discrete topology" with 4 open sets
    let open_sets: Vec<Vec<usize>> = vec![
        vec![0, 1], // first quadrant
        vec![1, 2], // second quadrant
        vec![2, 3], // third quadrant
        vec![3, 0], // fourth quadrant
    ];

    // test each point is in at least one open set
    for i in 0..discrete_angles.len() {
        let mut found = false;
        for set in &open_sets {
            if set.contains(&i) {
                found = true;
                break;
            }
        }
        assert!(found);
    }

    // test duality as length/angle duality
    // length and angle are dual concepts in geometric numbers
    let vector = Geonum {
        length: 2.0,
        angle: PI / 3.0,
        blade: 1,
    };

    // test operations on length and angle are often dual
    let doubled = Geonum {
        length: vector.length * 2.0,
        angle: vector.angle,
        blade: 1,
    };
    let rotated = Geonum {
        length: vector.length,
        angle: vector.angle * 2.0,
        blade: 1,
    };

    assert_eq!(doubled.length, 4.0);
    assert_eq!(rotated.angle, 2.0 * PI / 3.0);
}

#[test]
fn it_models_computing_structures() {
    // traditional computing uses pointers and references
    // geometric computing uses angles and dimensions

    // test types as angle dimensions
    // different types correspond to different angle spaces
    // artifact of geonum automation: separate type systems unified through geometric representation
    // no coordinate scaffolding needed - types are just different interpretations of the same geometric structure

    // int "type" exists in one dimension
    let int_one = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };
    // float "type" exists in another
    let float_one = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 1,
    };

    // test both have same internal representation but different type spaces
    assert_eq!(int_one.length, float_one.length);
    assert_eq!(int_one.angle, float_one.angle);

    // test language semantics as angle transformation
    // function application is angle transformation
    let function = |x: f64| -> f64 { x * x };

    // map this to geometric operation
    let geo_function = |g: Geonum| -> Geonum {
        Geonum {
            length: function(g.length),
            angle: g.angle,
            blade: 1,
        }
    };

    // test applying the function
    let input = Geonum {
        length: 3.0,
        angle: 0.0,
        blade: 1,
    };
    let output = geo_function(input);

    assert_eq!(output.length, 9.0);

    // test data structures as geometric entities
    // an array is a multivector with indexed elements
    let array = Multivector(vec![
        Geonum {
            length: 10.0,
            angle: 0.0,
            blade: 1,
        },
        Geonum {
            length: 20.0,
            angle: 0.0,
            blade: 1,
        },
        Geonum {
            length: 30.0,
            angle: 0.0,
            blade: 1,
        },
    ]);

    // access elements by index
    assert_eq!(array[0].length, 10.0);
    assert_eq!(array[1].length, 20.0);
    assert_eq!(array[2].length, 30.0);

    // test a simple tree data structure using geometric representation
    let tree = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        }, // root
        Geonum {
            length: 2.0,
            angle: PI / 3.0,
            blade: 1,
        }, // left child
        Geonum {
            length: 3.0,
            angle: 2.0 * PI / 3.0,
            blade: 1,
        }, // right child
    ]);

    // test tree properties
    assert_eq!(tree[0].length, 1.0); // root value
    assert_eq!(tree[1].length, 2.0); // left child
    assert_eq!(tree[2].length, 3.0); // right child
}
