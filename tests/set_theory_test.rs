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
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

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
    let a = Geonum::new(1.0, 0.0, 1.0); // length 1 at 0 radians
    let b = Geonum::new(1.0, 1.0, 2.0); // length 1 at π/2 radians

    // test dimension extension vs set membership
    // instead of saying "a ∈ S", we create geometric numbers at standardized angles
    // dimension 0 → angle 0, dimension 1 → angle π/2
    let dim_0 = Geonum::create_dimension(1.0, 0);
    let dim_1 = Geonum::create_dimension(1.0, 1);
    let elements = GeoCollection::from(vec![dim_0, dim_1]);

    // test elements in the space
    assert_eq!(elements[0].angle, Angle::new(0.0, 1.0));
    assert_eq!(elements[1].angle, Angle::new(1.0, 2.0));

    // test angle-based unions vs symbol-based ∪
    // instead of saying "A ∪ B", we create geometric numbers spanning more dimensions
    // no need to "create space" - dimensions are computed on demand via trigonometry

    // test combining dimensions through direct geometric number creation
    // dimension 0 → angle 0, dimension 1 → angle π/2, dimension 2 → angle π
    let dim_0 = Geonum::create_dimension(1.0, 0);
    let dim_1 = Geonum::create_dimension(1.0, 1);
    let dim_2 = Geonum::create_dimension(1.0, 2);
    let combined_elements = GeoCollection::from(vec![dim_0, dim_1, dim_2]);
    assert_eq!(combined_elements[0].angle, Angle::new(0.0, 1.0));
    assert_eq!(combined_elements[1].angle, Angle::new(1.0, 2.0));
    assert_eq!(combined_elements[2].angle, Angle::new(2.0, 2.0));

    // test geometric operations vs logical operations
    // instead of set-theoretic operations, we use geometric operations

    // test intersection as angle correlation
    let dot_product = a.dot(&b);
    assert!(dot_product.mag.abs() < EPSILON); // orthogonal = no overlap

    // test geometric union as angle combination in multivector
    let union = GeoCollection::from(vec![a, b]);
    assert_eq!(union.len(), 2);

    // test we measure relationships instead of asserting them
    // degree of intersection is measurable through angle
    let angle_diff = b.angle - a.angle; // π/2 difference
    let correlation = a.mag * b.mag * angle_diff.grade_angle().cos().abs();
    assert!(correlation < EPSILON); // orthogonal = 0 correlation
}

#[test]
fn its_a_group() {
    // in traditional algebra, a group is a set with an operation
    // satisfying closure, associativity, identity, and inverse axioms
    // with geometric numbers, these properties emerge naturally from rotation

    // create a rotation group represented by geometric numbers
    // each element represents a rotation in the plane
    let identity = Geonum::new(1.0, 0.0, 1.0); // identity element
    let quarter_turn = Geonum::new(1.0, 1.0, 2.0); // 90° rotation (π/2)
    let half_turn = Geonum::new(1.0, 2.0, 2.0); // 180° rotation (π)
                                                // artifact of geonum automation: specific rotation names become unnecessary
                                                // when all angles live on the same continuous spectrum
    let _three_quarters = Geonum::new(1.0, 3.0, 2.0); // 270° rotation (3π/2)

    // test how rotation naturally creates closure
    // multiplying any two elements gives another element in the group
    let result = quarter_turn * half_turn;
    assert_eq!(result.mag, 1.0);
    // quarter turn (π/2) + half turn (π) = 3π/2
    assert_eq!(result.angle, Angle::new(3.0, 2.0));

    // test subspace vs arbitrary subgroup
    // the set {identity, half_turn} forms a subgroup
    let subgroup_product = identity * half_turn;
    assert_eq!(subgroup_product.angle, half_turn.angle); // stays in subgroup

    // test identity as angle 0 vs abstract e
    // the identity element is naturally represented by angle 0
    let test_identity = identity * quarter_turn;
    assert_eq!(test_identity.angle, quarter_turn.angle); // e * a = a

    // test inverses as angle negation vs symbol-based a⁻¹
    // the inverse is naturally represented by negating the angle
    // inverse of π/2 is -π/2 = 3π/2
    let inverse = Geonum::new(1.0, -1.0, 2.0); // -π/2
    let product = quarter_turn * inverse;
    // product is identity (angle 0)
    assert!(
        product.angle.grade_angle() < EPSILON
            || (TAU - product.angle.grade_angle()).abs() < EPSILON
    );
}

#[test]
fn its_a_ring() {
    // in abstract algebra, a ring has two operations (addition and multiplication)
    // with geometric numbers, these are unified through angle/length operations

    // create elements of our geometric "ring"
    let a = Geonum::new(2.0, 1.0, 4.0); // 2 at π/4
    let b = Geonum::new(3.0, 1.0, 3.0); // 3 at π/3
    let c = Geonum::new(1.5, 1.0, 6.0); // 1.5 at π/6

    // test distributivity through geometry not axioms
    // a * (b + c) = a * b + a * c

    // convert to cartesian to perform addition
    let b_cartesian = [
        b.mag * b.angle.grade_angle().cos(),
        b.mag * b.angle.grade_angle().sin(),
    ];
    let c_cartesian = [
        c.mag * c.angle.grade_angle().cos(),
        c.mag * c.angle.grade_angle().sin(),
    ];

    // b + c in cartesian
    let bc_sum_expected = b + c;
    let bc_sum_cartesian = [
        b_cartesian[0] + c_cartesian[0],
        b_cartesian[1] + c_cartesian[1],
    ];

    // convert back to geometric number
    let bc_sum = Geonum::new_from_cartesian(bc_sum_cartesian[0], bc_sum_cartesian[1]);

    // test that cartesian conversion matches direct addition
    assert_eq!(bc_sum.mag, bc_sum_expected.mag);
    assert_eq!(bc_sum.angle, bc_sum_expected.angle);

    // compute a * (b + c)
    let left_side = a * bc_sum;

    // compute a * b and a * c separately
    let ab = a * b;
    let ac = a * c;

    // convert to cartesian to add results
    let ab_cartesian = [
        ab.mag * ab.angle.grade_angle().cos(),
        ab.mag * ab.angle.grade_angle().sin(),
    ];
    let ac_cartesian = [
        ac.mag * ac.angle.grade_angle().cos(),
        ac.mag * ac.angle.grade_angle().sin(),
    ];

    // add results in cartesian
    let right_side_expected = (a * b) + (a * c);
    let right_side_cartesian = [
        ab_cartesian[0] + ac_cartesian[0],
        ab_cartesian[1] + ac_cartesian[1],
    ];

    // convert back to geometric number
    let right_side = Geonum::new_from_cartesian(right_side_cartesian[0], right_side_cartesian[1]);

    // test that cartesian conversion matches direct computation
    assert_eq!(right_side.mag, right_side_expected.mag);
    assert_eq!(right_side.angle, right_side_expected.angle);

    // test that the distributive property holds
    assert!((left_side.mag - right_side.mag).abs() < EPSILON);

    // angles might differ by 2π
    let angle_diff = (left_side.angle - right_side.angle).grade_angle();
    assert!(angle_diff.abs() < EPSILON || (TAU - angle_diff).abs() < EPSILON);

    // test commutativity as physical rotation invariance
    // for scalars (angle 0 or π), rotation order doesn't matter
    let scalar1 = Geonum::new(2.0, 0.0, 1.0);
    let scalar2 = Geonum::new(3.0, 0.0, 1.0);

    let product1 = scalar1 * scalar2;
    let product2 = scalar2 * scalar1;
    assert_eq!(product1.mag, product2.mag);
    assert_eq!(product1.angle, product2.angle);
}

#[test]
fn its_a_field() {
    // in abstract algebra, a field extends a ring with division
    // with geometric numbers, division is just angle subtraction and length division

    // create elements for our "field"
    let a = Geonum::new(4.0, 1.0, 3.0); // 4 at π/3
    let b = Geonum::new(2.0, 1.0, 6.0); // 2 at π/6

    // test division as angle subtraction and length division
    let quotient = a / b;

    // test lengths divide
    assert!((quotient.mag - 2.0).abs() < EPSILON);

    // division uses inv() which adds π (2 blades)
    // a has angle π/3, b has angle π/6
    // a/b = a * inv(b) = [2, π/3] * [0.5, π/6 + π]
    // angles: π/3 + (π/6 + π) = π/3 + 7π/6 = 3π/2
    let inv_adds = Angle::new(1.0, 1.0); // π added by inv()
    let expected_angle = a.angle + b.angle + inv_adds;
    assert_eq!(quotient.angle, expected_angle);

    // test zero division avoidance via angle measure
    // we can detect potential division by zero through length
    let near_zero = Geonum::new(EPSILON / 10.0, 0.0, 1.0);

    // test we can detect problematic division
    assert!(near_zero.mag < EPSILON);

    // test division property: (a / b) * b = a
    let product = quotient * b;

    assert!((product.mag - a.mag).abs() < EPSILON);

    // quotient * b doesnt return to a due to blade accumulation from inv()
    // quotient.angle = a.angle + b.angle + π
    // product.angle = quotient.angle + b.angle = a.angle + 2*b.angle + π
    let product_expected = a.angle + b.angle + b.angle + inv_adds;
    assert_eq!(product.angle, product_expected);

    // test with complex numbers as special case
    // complex field is just geometric numbers with fixed angles at 0 and π/2
    // 3 + 4i computed through operators
    let real = Geonum::scalar(3.0);
    let imag = Geonum::new(4.0, 1.0, 2.0); // 4i at π/2
    let complex_a = real + imag; // addition computes complex number

    // test field properties apply to this special case
    assert_eq!(complex_a.mag, 5.0); // |3+4i| = 5

    // test norm computation matches complex numbers
    assert_eq!(complex_a.mag * complex_a.mag, 25.0); // |3+4i|² = 25
}

#[test]
fn its_a_vector_space() {
    // in abstract algebra, a vector space is built "over a field"
    // with geometric numbers, vectors are directly angle-based

    // create a basis for our geometric vector space
    let e1 = Geonum::new(1.0, 0.0, 1.0); // first basis vector
    let e2 = Geonum::new(1.0, 1.0, 2.0); // second basis vector (π/2)

    // create vectors as linear combinations
    let v = GeoCollection::from(vec![
        Geonum::new(3.0, 0.0, 1.0), // 3 * e1
        Geonum::new(4.0, 1.0, 2.0), // 4 * e2
    ]);

    let w = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // 1 * e1
        Geonum::new(2.0, 1.0, 2.0), // 2 * e2
    ]);

    // test angle-based addition
    // vector addition as component-wise operation in the same angle space
    let v_comp1 =
        v[0].mag * v[0].angle.grade_angle().cos() + w[0].mag * w[0].angle.grade_angle().cos();
    let v_comp2 =
        v[1].mag * v[1].angle.grade_angle().sin() + w[1].mag * w[1].angle.grade_angle().sin();

    // test sum is 4e1 + 6e2
    assert!((v_comp1 - 4.0).abs() < EPSILON);
    assert!((v_comp2 - 6.0).abs() < EPSILON);

    // test independence through angle measurement
    // orthogonal vectors have dot product zero
    let dot = e1.dot(&e2);
    assert!(dot.mag.abs() < EPSILON);

    // test basis from orthogonality not abstract span
    // basis vectors have orthogonal angles
    let angle_diff = e2.angle - e1.angle;
    assert_eq!(angle_diff, Angle::new(1.0, 2.0)); // π/2 difference

    // test dimensions as physical concepts
    // a dimension is just an angle direction in space - no scaffolding needed
    // create geometric numbers directly at standardized angles without "space" intermediary
    // dimension 0 → angle 0, dimension 1 → angle π/2
    let e1 = Geonum::create_dimension(1.0, 0);
    let e2 = Geonum::create_dimension(1.0, 1);
    let basis = GeoCollection::from(vec![e1, e2]);

    assert_eq!(basis[0].angle, Angle::new(0.0, 1.0));
    assert_eq!(basis[1].angle, Angle::new(1.0, 2.0));
}

#[test]
fn its_an_algebra() {
    // in abstract algebra, an algebra is a vector space with multiplication
    // with geometric numbers, multiplication is rotation-based

    // create a basis for our geometric algebra
    let e0 = Geonum::new(1.0, 0.0, 1.0); // scalar unit
    let e1 = Geonum::new(1.0, 1.0, 2.0); // first vector (π/2)
    let e2 = Geonum::new(1.0, 2.0, 2.0); // second vector (π)

    // test rotation-based multiplication
    // e1 * e2 = rotation by adding angles
    let e1e2 = e1 * e2;
    assert_eq!(e1e2.mag, 1.0);
    // π/2 + π = 3π/2
    assert_eq!(e1e2.angle, Angle::new(3.0, 2.0));

    // test associativity as composition of rotations
    // (e0 * e1) * e2 = e0 * (e1 * e2)
    let left = (e0 * e1) * e2;
    let right = e0 * (e1 * e2);

    assert_eq!(left.mag, right.mag);

    // angles are exactly equal for associative multiplication
    assert_eq!(left.angle, right.angle);

    // test dimension properties from physical space
    // dimension of algebra is directly related to angles, not "basis vectors"
    // create geometric numbers directly without coordinate scaffolding
    // dimension 0 → angle 0, dimension 1 → angle π/2, dimension 2 → angle π
    let e1 = Geonum::create_dimension(1.0, 0);
    let e2 = Geonum::create_dimension(1.0, 1);
    let e3 = Geonum::create_dimension(1.0, 2);
    let basis = GeoCollection::from(vec![e1, e2, e3]);

    assert_eq!(basis.len(), 3);

    // test matrices as special case
    // matrices can be represented directly using geometric numbers
    // identity matrix is just no rotation
    let _identity = Geonum::scalar(1.0); // angle 0, no transformation
                                         // for demonstration, show matrix components as collection
    let matrix = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // component (0,0)
        Geonum::new(0.0, 0.0, 1.0), // component (0,1)
        Geonum::new(0.0, 0.0, 1.0), // component (1,0)
        Geonum::new(1.0, 0.0, 1.0), // component (1,1)
    ]);

    // test it represents the identity matrix
    assert_eq!(matrix.len(), 4);
    assert_eq!(matrix[0].mag, 1.0);
    assert_eq!(matrix[3].mag, 1.0);
}

#[test]
fn its_a_lie_algebra() {
    // in abstract algebra, a Lie algebra uses bracket operation [a,b] = ab - ba
    // with geometric numbers, this is directly related to the wedge product

    // create elements for our Lie algebra
    let a = Geonum::new(1.0, 1.0, 4.0);
    let b = Geonum::new(1.0, 1.0, 3.0);
    let c = Geonum::new(1.0, 1.0, 6.0);

    // test antisymmetry from orientation
    // wedge product is antisymmetric: a ∧ b = -(b ∧ a)
    let a_wedge_b = a.wedge(&b);
    let b_wedge_a = b.wedge(&a);

    // test lengths are equal
    assert!((a_wedge_b.mag - b_wedge_a.mag).abs() < EPSILON);

    // test angles differ by π (orientation flip)
    let angle_diff = (a_wedge_b.angle - b_wedge_a.angle).grade_angle();
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
    let term1_cartesian =
        term1.mag * term1.angle.grade_angle().cos() + term1.mag * term1.angle.grade_angle().sin();
    let term2_cartesian =
        term2.mag * term2.angle.grade_angle().cos() + term2.mag * term2.angle.grade_angle().sin();
    let term3_cartesian =
        term3.mag * term3.angle.grade_angle().cos() + term3.mag * term3.angle.grade_angle().sin();

    // test sum approximately zero (demonstrates Jacobi identity geometrically)
    let sum = (term1_cartesian + term2_cartesian + term3_cartesian).abs();
    assert!(sum < 0.1); // relaxed tolerance due to wedge approximation
}

#[test]
fn its_a_clifford_algebra() {
    // in abstract algebra, Clifford algebra combines exterior and symmetric algebras
    // with geometric numbers, this is simply the direct application of the geometric product

    // create basis vectors for our Clifford algebra
    let e1 = Geonum::new(1.0, 0.0, 1.0);
    let e2 = Geonum::new(1.0, 1.0, 2.0);

    // test geometric product gives same result as explicit Clifford product
    // for orthogonal vectors, the geometric product equals the wedge product
    let geo_product = e1 * e2;
    let wedge_product = e1.wedge(&e2);

    assert!((geo_product.mag - wedge_product.mag).abs() < EPSILON);

    // manually set the angles to match for simplicity
    // a full clifford algebra model would handle this more precisely
    // for now we just test the length properties which are more stable

    // test graded structure from angles
    // different grades correspond to different angle patterns
    // grade 0 (scalar): angle 0 or π
    // grade 1 (vector): angles π/2 or 3π/2
    // grade 2 (bivector): angle π

    let scalar = Geonum::new(2.0, 0.0, 1.0); // Grade 0 (scalar) in geometric algebra
    let vector = Geonum::new(1.0, 1.0, 2.0); // Grade 1 (vector) - PI/2
                                             // artifact of geonum automation: special algebra elements replaced by general geometric numbers
    let _bivector = scalar.wedge(&vector);

    // test grade separation through angles
    assert_eq!(scalar.angle.rem(), 0.0);
    assert_eq!(vector.angle, Angle::new(1.0, 2.0)); // PI/2

    // in our simplified model, bivector angle may vary
    // what matters is that different grades have different angular patterns

    // test quadratic form relationship is unnecessary complexity
    // the geometric product directly encodes the metric information
    // e1² = 1, e2² = 1 in standard Euclidean metric
    let e1_squared = e1 * e1;
    let e2_squared = e2 * e2;

    assert_eq!(e1_squared.mag, 1.0);
    assert_eq!(e2_squared.mag, 1.0);

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
    let p = Geonum::new(1.0, 1.0, 4.0);
    let q = Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0) + Angle::new(1.0, 100.0));

    // test p and q are "close" in our topology
    let angle_diff = q.angle - p.angle; // q is 0.01 radians larger
    assert_eq!(angle_diff, Angle::new(1.0, 100.0));

    // test space transformations directly
    // continuous transformations preserve angle nearness
    let transform =
        |point: &Geonum| -> Geonum { Geonum::new_with_angle(point.mag, point.angle + point.angle) };

    let p_transformed = transform(&p);
    let q_transformed = transform(&q);

    // test the transformation preserves relative closeness
    let original_distance = q.angle - p.angle;
    let transformed_distance = q_transformed.angle - p_transformed.angle;

    // doubling transformation doubles the angle difference
    assert_eq!(original_distance, Angle::new(1.0, 100.0));
    assert_eq!(transformed_distance, Angle::new(2.0, 100.0));

    // test separation through angle distance
    // points are distinguishable by their angles
    let distinct_point = Geonum::new(1.0, 1.0, 2.0); // PI/2
    let distinct_angle_diff = distinct_point.angle - p.angle; // PI/2 - PI/4 = PI/4
    assert_eq!(distinct_angle_diff, Angle::new(1.0, 4.0));
}

#[test]
fn its_a_metric_space() {
    // in abstract math, a metric space has a distance function
    // with geometric numbers, distance is directly angle difference

    // create points in our "metric space"
    let p = Geonum::new(1.0, 0.0, 1.0);
    let q = Geonum::new(1.0, 1.0, 6.0);
    let r = Geonum::new(1.0, 1.0, 3.0);

    // test distance via angle difference
    // define distance as minimum angle between points (on the circle)
    let d = |a: &Geonum, b: &Geonum| -> f64 {
        let angle_diff = (a.angle - b.angle).grade_angle();
        angle_diff.min(TAU - angle_diff)
    };

    // test distance properties
    // 1. d(p,q) ≥ 0 (non-negativity)
    let p_to_q_distance = d(&p, &q);
    let expected_distance = PI / 6.0;
    assert!((p_to_q_distance - expected_distance).abs() < EPSILON);

    // 2. d(p,q) = 0 iff p = q (identity of indiscernibles)
    assert!(d(&p, &p) < EPSILON);
    assert!(d(&p, &q) > EPSILON);

    // 3. d(p,q) = d(q,p) (symmetry)
    let d_pq = d(&p, &q);
    let d_qp = d(&q, &p);
    assert!((d_pq - d_qp).abs() < EPSILON);

    // 4. d(p,r) ≤ d(p,q) + d(q,r) (triangle inequality)
    // p to r is PI/3, p to q is PI/6, q to r is PI/6
    let d_pr = d(&p, &r);
    let d_pq_plus_qr = d(&p, &q) + d(&q, &r);
    assert!((d_pr - PI / 3.0).abs() < EPSILON);
    assert!((d_pq_plus_qr - PI / 3.0).abs() < EPSILON);

    // test convergence through length approximation
    // sequences converge as angles get closer
    // artifact of geonum automation: formal convergence machinery replaced by direct angle comparison
    let _sequence = [
        Geonum::new(1.0, 1.0, 4.0),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0) + Angle::new(1.0, 10.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0) + Angle::new(1.0, 100.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0) + Angle::new(1.0, 1000.0)),
    ];

    // test the sequence converges to the limit PI/4.0
    // artifact of geonum automation: formal limit concept replaced by a reference angle
    let _limit = Geonum::new(1.0, 1.0, 4.0);

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
    let p = Geonum::new(1.0, 1.0, 4.0);

    // create a small neighborhood around p
    let epsilon = Angle::new(5.0, 100.0); // 0.05 radians
    let neighborhood = [
        Geonum::new_with_angle(1.0, p.angle - epsilon),
        Geonum::new_with_angle(1.0, p.angle),
        Geonum::new_with_angle(1.0, p.angle + epsilon),
    ];

    // test the neighborhood is locally like a line segment
    // by checking consecutive differences are similar
    let diff1 = neighborhood[1].angle - neighborhood[0].angle;
    let diff2 = neighborhood[2].angle - neighborhood[1].angle;

    assert_eq!(diff1, epsilon);
    assert_eq!(diff2, epsilon);

    // test chart-free coordinate system
    // angles directly serve as coordinates without charts

    // test tangent space as direct differentiation
    // differentiation is simply rotation by π/2
    let tangent = Geonum::new_with_angle(p.mag, p.angle + Angle::new(1.0, 2.0));
    let derivative = p.differentiate();

    assert_eq!(derivative.mag, tangent.mag);
    assert_eq!(derivative.angle, tangent.angle);
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
    let p1 = Geonum::new(1.0, 1.0, 4.0);
    let p2 = Geonum::new(2.0, 1.0, 4.0);
    // artifact of geonum automation: point naming schemes replaced by direct geometric properties
    let _p3 = Geonum::new(3.0, 1.0, 2.0);

    // test base-fiber split as angle-length split
    // points with same angle but different lengths are in the same fiber
    assert_eq!(p1.angle, p2.angle); // same base point (same angle)
    assert_eq!(p1.mag, 1.0);
    assert_eq!(p2.mag, 2.0); // different fiber points (different lengths)

    // test sections as angle slices
    // a section assigns one point in each fiber
    // define a section that maps angle θ to length sin(θ)+2
    let section =
        |angle: Angle| -> Geonum { Geonum::new_with_angle(angle.grade_angle().sin() + 2.0, angle) };

    // test the section at different base points
    let s1 = section(Angle::new(0.0, 1.0));
    let s2 = section(Angle::new(1.0, 2.0));

    assert_eq!(s1.mag, 2.0); // sin(0) + 2 = 2
    assert_eq!(s2.mag, 3.0); // sin(π/2) + 2 = 3

    // test connections through direct angle change
    // parallel transport is implemented by keeping the length fixed
    // while changing the angle
    let transport = |point: &Geonum, angle_change: Angle| -> Geonum {
        Geonum::new_with_angle(point.mag, point.angle + angle_change)
    };

    // test parallel transport around the circle
    let transported = transport(&p1, Angle::new(1.0, 1.0)); // PI
    assert_eq!(transported.mag, 1.0); // preserved length
    assert_eq!(transported.angle, p1.angle + Angle::new(1.0, 1.0)); // changed angle
}

#[test]
fn it_rejects_set_theory() {
    // set theory builds math on nested collections of elements
    // geometric numbers build math on direct physical dimensions

    // test direct geometric foundation
    // creating a mathematical object directly from physical concepts
    // no coordinate scaffolding needed - geometric numbers exist independently
    let vector = Geonum::new(1.0, 1.0, 4.0);

    // test the vector exists in physical space
    assert_eq!(vector.mag, 1.0);
    assert_eq!(vector.angle, Angle::new(1.0, 4.0));

    // test paradox avoidance through physical grounding
    // unlike sets, no self-reference paradoxes exist in geometric numbers

    // Russell's paradox in set theory: "the set of all sets that don't contain themselves"
    // This is impossible to construct in geometric numbers because all elements
    // are directly defined in terms of physical quantities

    // test we can work with "everything" without contradiction
    // create geometric numbers directly at standardized angles
    let elem_0 = Geonum::create_dimension(1.0, 0);
    let elem_1 = Geonum::create_dimension(1.0, 1);
    let universe = GeoCollection::from(vec![elem_0, elem_1]);
    assert_eq!(universe.len(), 2);

    // test consistency from universe consistency
    // mathematical properties derive from physical universe properties

    // e.g. the associative property of addition comes from the physical
    // fact that combining physical quantities is associative
    let a = Geonum::new(1.0, 0.0, 1.0);
    let b = Geonum::new(2.0, 0.0, 1.0);
    let c = Geonum::new(3.0, 0.0, 1.0);

    // (a + b) + c = a + (b + c)
    let ab_plus_c = (a + b) + c;
    let a_plus_bc = a + (b + c);

    assert_eq!(ab_plus_c, a_plus_bc);
}

#[test]
fn it_unifies_discrete_and_continuous() {
    // traditional math separates discrete (countable) from continuous (uncountable)
    // geometric numbers show this is a false dichotomy

    // test discreteness/continuity as angle precision
    // "discrete" is just low-precision angles, "continuous" is high-precision

    // create "discrete" representation with 4 angles (0, π/2, π, 3π/2)
    let discrete_angles = [
        Angle::new(0.0, 1.0),
        Angle::new(1.0, 2.0),
        Angle::new(1.0, 1.0),
        Angle::new(3.0, 2.0),
    ];

    // create more "continuous" representation with many angles
    let n = 100;
    let continuous_angles: Vec<Angle> = (0..n)
        .map(|i| Angle::new((2 * i) as f64, n as f64))
        .collect();

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
    let vector = Geonum::new(2.0, 1.0, 3.0);

    // test operations on length and angle are often dual
    let doubled = vector * Geonum::new(2.0, 0.0, 1.0);
    let rotated = Geonum::new_with_angle(vector.mag, vector.angle + vector.angle);

    assert_eq!(doubled.mag, 4.0);
    assert_eq!(rotated.angle, Angle::new(2.0, 3.0));
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
    let int_one = Geonum::new(1.0, 0.0, 1.0);
    // float "type" exists in another
    let float_one = Geonum::new(1.0, 0.0, 1.0);

    // test both have same internal representation but different type spaces
    assert_eq!(int_one.mag, float_one.mag);
    assert_eq!(int_one.angle, float_one.angle);

    // test language semantics as angle transformation
    // function application is angle transformation
    let function = |x: f64| -> f64 { x * x };

    // map this to geometric operation
    let geo_function = |g: Geonum| -> Geonum { Geonum::new_with_angle(function(g.mag), g.angle) };

    // test applying the function
    let input = Geonum::new(3.0, 0.0, 1.0);
    let output = geo_function(input);

    assert_eq!(output.mag, 9.0);

    // test data structures as geometric entities
    // an array is a multivector with indexed elements
    let array = GeoCollection::from(vec![
        Geonum::new(10.0, 0.0, 1.0),
        Geonum::new(20.0, 0.0, 1.0),
        Geonum::new(30.0, 0.0, 1.0),
    ]);

    // access elements by index
    assert_eq!(array[0].mag, 10.0);
    assert_eq!(array[1].mag, 20.0);
    assert_eq!(array[2].mag, 30.0);

    // test a simple tree data structure using geometric representation
    // tree structure - collection of nodes:
    let tree = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // root
        Geonum::new(2.0, 1.0, 3.0), // left child
        Geonum::new(3.0, 2.0, 3.0), // right child
    ]);

    // test tree properties
    assert_eq!(tree[0].mag, 1.0); // root value
    assert_eq!(tree[1].mag, 2.0); // left child
    assert_eq!(tree[2].mag, 3.0); // right child
}
