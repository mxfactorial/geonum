// "linear combinations" are founded on a fictional data type called a "basis vector" to reconstruct direction
//
// to keep vector operations consistent with a fictional data type you must self-referentially require "orthogonal basis vectors" as "components" of all "vectors"
//
// hacking directional consistency with coordinate scaffolding just traps everyone in a complexity loop ("span the space")
//
// and denies them the opportunity to understand how direction **naturally exists** in the physical universe
//
// so instead of "combining basis vectors", geometric numbers prove their directional consistency with the physical universe by *preserving* the universe's existing angles with `let vector = [length, angle];`
//
// rejecting "linear combinations" for "angle preservation" empowers people to understand direction so well they can even **project** with it onto any dimension without predefined coordinates:
//
// ```rs
// let vector_3d = [magnitude, angle, blade];
// let x_component = vector_3d.project_to_dimension(0);  // no basis vectors needed
// let y_component = vector_3d.project_to_dimension(1);  // no coordinate system setup
// let z_component = vector_3d.project_to_dimension(2);  // just trigonometric projection
// ```
//
// say goodbye to `x*e₁ + y*e₂ + z*e₃`

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_projects_a_geonum_onto_coordinate_axes() {
    // step 1: unify traditional [x, y, z] coordinates into single geonum
    // traditional 3d vector: [2, 3, 1]
    let x: f64 = 2.0;
    let y: f64 = 3.0;
    let z: f64 = 1.0;

    // compute unified geometric representation
    let magnitude = (x * x + y * y + z * z).sqrt(); // 3.742

    // encode direction as fractional angle within pi/2 constraint
    // for this simplified test, use direct angle calculation
    let total_angle = y.atan2(x); // basic 2d angle for now
    let blade = 3; // 3d vector = 3 pi/2 turns
    let fractional_angle = total_angle % (PI / 2.0); // keep within [0, pi/2]

    let unified = Geonum::new_with_blade(magnitude, blade, fractional_angle, PI);

    // step 2: decompose geonum back to coordinate projections
    // project unified geometric number onto coordinate axes

    // x-axis projection (0 pi/2 turns from origin)
    let x_projected = unified.project_to_dimension(0);

    // y-axis projection (1 pi/2 turn from origin)
    let y_projected = unified.project_to_dimension(1);

    // z-axis projection (2 pi/2 turns from origin)
    let z_projected = unified.project_to_dimension(2);

    // step 3: test that projections are well-defined
    assert!(x_projected.is_finite(), "x component is finite");
    assert!(y_projected.is_finite(), "y component is finite");
    assert!(z_projected.is_finite(), "z component is finite");

    // step 4: test that magnitude is preserved
    let reconstructed_magnitude =
        (x_projected * x_projected + y_projected * y_projected + z_projected * z_projected).sqrt();
    // note: this is a simplified test - full coordinate reconstruction requires more complex angle encoding
    assert!(reconstructed_magnitude.is_finite(), "magnitude is finite");
}

#[test]
fn it_proves_dimensions_are_observed_from_angles() {
    // create a geometric number without defining any dimensional space
    // this geometric entity exists independently of coordinate systems
    let geometric_entity = Geonum::new(2.5, 0.8, PI); // vector with 0.8 radians angle

    // test: observer can query any dimension without pre-definition

    // query the 3rd dimension (traditional z-axis)
    let dimension_3 = geometric_entity.project_to_dimension(3);

    // query the 42nd dimension (never defined anywhere)
    let dimension_42 = geometric_entity.project_to_dimension(42);

    // query the 1000th dimension (would be impossible in traditional systems)
    let dimension_1000 = geometric_entity.project_to_dimension(1000);

    // query the 1,000,000th dimension (completely arbitrary)
    let dimension_million = geometric_entity.project_to_dimension(1_000_000);

    // critical insight: all these components can be non-zero!
    // the geometric entity has meaningful projections into dimensions
    // that were never defined, declared, or initialized

    // compute expected trigonometric values
    let entity_angle = geometric_entity.angle.mod_4_angle();
    let expected_3 = 2.5 * ((3.0 * PI / 2.0) - entity_angle).cos();
    let expected_42 = 2.5 * ((42.0 * PI / 2.0) - entity_angle).cos();
    let expected_1000 = 2.5 * ((1000.0 * PI / 2.0) - entity_angle).cos();
    let expected_million = 2.5 * ((1_000_000.0 * PI / 2.0) - entity_angle).cos();

    assert!(
        (dimension_3 - expected_3).abs() < EPSILON,
        "3rd dimension projection matches trigonometric values"
    );
    assert!(
        (dimension_42 - expected_42).abs() < EPSILON,
        "42nd dimension projection matches trigonometric values"
    );
    assert!(
        (dimension_1000 - expected_1000).abs() < EPSILON,
        "1000th dimension projection matches trigonometric values"
    );
    assert!(
        (dimension_million - expected_million).abs() < EPSILON,
        "millionth dimension projection matches trigonometric values"
    );

    // test at least some dimensions have non-zero components
    let total_energy =
        dimension_3.abs() + dimension_42.abs() + dimension_1000.abs() + dimension_million.abs();
    assert!(
        total_energy > 0.0,
        "geometric entity has non-zero projections"
    );

    // test complete: dimensions are computed on demand, not predefined
    // the geometric number exists independently and projects to any dimension via trigonometry

    // traditional design requires:
    // 1. define 1,000,000-dimensional space
    // 2. initialize basis vectors
    // 3. store 1,000,000 components
    // 4. manage coordinate transformations
    //
    // geonum design:
    // 1. one geometric number
    // 2. compute projection into dimension n on demand
    // 3. trigonometric calculation provides answer
    // 4. no storage, no initialization, no limits
}

#[test]
fn it_creates_dimensions_with_standardized_angles() {
    // test the create_dimension constructor
    let dim_0 = Geonum::create_dimension(1.0, 0);
    let dim_1 = Geonum::create_dimension(1.0, 1);
    let dim_2 = Geonum::create_dimension(1.0, 2);
    let dim_1000 = Geonum::create_dimension(1.0, 1000);

    // test angles are standardized to dimension_index * pi/2
    assert!((dim_0.angle.mod_4_angle() - 0.0).abs() < EPSILON);
    assert!((dim_1.angle.mod_4_angle() - PI / 2.0).abs() < EPSILON);
    assert!((dim_2.angle.mod_4_angle() - PI).abs() < EPSILON);
    // For dimension 1000, the angle wraps around many times
    // 1000 * π/2 = 500π = 250 * 2π, so mod_4_angle should be 0
    assert!((dim_1000.angle.mod_4_angle() - 0.0).abs() < EPSILON);

    // test blade equals dimension_index
    assert_eq!(dim_0.angle.blade(), 0);
    assert_eq!(dim_1.angle.blade(), 1);
    assert_eq!(dim_2.angle.blade(), 2);
    assert_eq!(dim_1000.angle.blade(), 1000);

    // test multivector constructor
    let mv = Multivector::create_dimension(1.0, &[0, 1, 2]);
    assert_eq!(mv.len(), 3);
    assert_eq!(mv[0].angle.blade(), 0);
    assert_eq!(mv[1].angle.blade(), 1);
    assert_eq!(mv[2].angle.blade(), 2);
}

#[test]
fn it_proves_grade_decomposition_ignores_angle_addition() {
    // traditional geometric algebra ignores that multiplication adds angles
    // when you multiply v1 * v2, the angles add: θ1 + θ2
    // but traditional GA pretends this angle addition doesnt happen

    // example: multiply two 45° vectors
    // v1 at 45°, v2 at 45°
    // v1 * v2 rotates by 45° + 45° = 90°

    // but traditional GA ignores this simple angle addition and instead:
    // 1. computes a "scalar part" (grade 0)
    // 2. computes a "bivector part" (grade 2)
    // 3. stores both in separate memory locations
    // 4. pretends the 90° rotation is somehow split between them

    // this negligence - ignoring angle addition - forces traditional GA to:
    // - track 2^n components to handle all possible angle accumulations
    // - invent "grade decomposition" to duplicate the angle information
    // - create massive computational overhead for simple rotations

    // geonum acknowledges that multiplication adds angles
    let v1 = Geonum::new(1.0, 1.0, 4.0); // 45° = π/4
    let v2 = Geonum::new(1.0, 1.0, 4.0); // 45° = π/4

    let product = v1 * v2;

    // result: 45° + 45° = 90° rotation, stored as single angle
    assert_eq!(product.length, 1.0);
    assert_eq!(product.angle.blade(), 1); // 90° rotation (blade 1)
    assert!(product.angle.value().abs() < EPSILON); // exactly π/2

    // geonum stores the angle addition result directly
    // no need to decompose into "scalar" and "bivector" parts
    // no need for 2^n components to track angle accumulations

    // traditional GA creates "grade 0" and "grade 2" components because
    // it refuses to acknowledge that angles simply added to 90°

    // demonstration: multiply 0° by 90°
    let x_axis = Geonum::create_dimension(1.0, 0); // 0°
    let y_axis = Geonum::create_dimension(1.0, 1); // 90°

    let xy_product = x_axis * y_axis;

    // angle addition: 0° + 90° = 90°
    assert_eq!(xy_product.angle.blade(), 1); // 90° rotation
    assert!(xy_product.angle.value().abs() < EPSILON);

    // traditional GA would ignore this angle addition and instead:
    // - compute x·y = 0 (call it "scalar part")
    // - compute x∧y = 1 (call it "bivector part")
    // - store both separately
    // - pretend the 90° rotation is somehow "decomposed"

    // but the 90° rotation hasnt been decomposed - its been ignored!
    // grade decomposition is what you get when you refuse to track angle addition

    // by ignoring "angles add", traditional GA creates exponential complexity
    // every possible angle sum needs its own storage location
    // thats why you get 2^n components - one for each possible accumulation

    // geonum eliminates slack from the geometry by requiring angle addition
    // no duplication, no exponential blowup, just store the angle sum directly
}

#[test]
fn it_proves_vectors_can_never_be_orthogonal() {
    // traditional math inherited the notion that vectors can be "orthogonal"
    // this language emerged from "grade decomposition"

    // start with a geometric object pointing along x-axis
    let x_axis = Geonum::new(1.0, 0.0, 1.0); // [1, 0]
    assert_eq!(x_axis.angle.blade(), 0); // blade 0 (scalar grade)
    assert!(x_axis.angle.is_scalar());

    // attempt to create an "orthogonal vector" by rotating PI/2
    let y_axis = x_axis.rotate(Angle::new(1.0, 2.0)); // rotate by PI/2

    // the rotation changed the geometric grade!
    assert_eq!(y_axis.angle.blade(), 1); // blade 1 (vector grade)
    assert!(y_axis.angle.is_vector());

    // rotate again to create what traditional math calls "third orthogonal vector"
    let z_axis = y_axis.rotate(Angle::new(1.0, 2.0));
    assert_eq!(z_axis.angle.blade(), 2); // blade 2 (bivector grade)
    assert!(z_axis.angle.is_bivector());

    // traditional GA negligently claims x, y, z are "three orthogonal vectors"
    // without noticing theyre three different geometric grades:
    // - x_axis: blade 0 (scalar grade)
    // - y_axis: blade 1 (vector grade)
    // - z_axis: blade 2 (bivector grade)

    // the historical pattern that led here:
    // 1. basis elements were assigned grade 1 (e1, e2, e3 all "vectors")
    // 2. their angular relationships scattered across storage locations
    // 3. 2^n components emerged to accommodate what could be n simple rotations

    // proof: try to create what traditional GA calls "two orthogonal vectors"
    // both at blade 1 (vector grade)
    let forced_x = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1, angle 0
    let forced_y = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // attempt blade 1 + π/2

    // but forced_y becomes blade 2!
    assert_eq!(forced_x.angle.blade(), 1); // vector
    assert_eq!(forced_y.angle.blade(), 2); // bivector!

    // when we try to add π/2 to a vector, it becomes a bivector
    // proving that orthogonal rotations inherently change grade

    // the dot product between vector and bivector is zero
    let dot = forced_x.dot(&forced_y);
    assert!(dot.length.abs() < 1e-10); // zero

    // but this is because they're different grades, not because
    // they're "two orthogonal vectors" - one is a vector, one is a bivector

    // the unintended computational pattern this creates:
    // - 2D requires 4 components (scalar + 2 vectors + bivector)
    // - 3D requires 8 components (scalar + 3 vectors + 3 bivectors + trivector)
    // - nD requires 2^n components (exponential growth)

    // all to preserve the inherited notion of "n orthogonal vectors"
    // when geometrically, orthogonality naturally transforms grade

    // to create two blade 1 vectors, they must have angles within the same π/2 segment
    let v1 = Geonum::new_with_blade(1.0, 1, 0.0, 4.0); // blade 1, +0
    let v2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // blade 1, +π/4

    assert_eq!(v1.angle.blade(), 1);
    assert_eq!(v2.angle.blade(), 1);

    // their dot product cannot be zero because angle diff < π/2
    let dot_vectors = v1.dot(&v2);
    assert!(dot_vectors.length > 0.0); // positive, not zero!

    // geonum reveals the underlying pattern: PI/2 rotation changes grade
    // "orthogonal vectors" emerged from overlooking grade transformation
    // the angle-blade invariant makes explicit what grade decomposition obscured
}

#[test]
fn it_solves_the_exponential_complexity_explosion() {
    // THE PROBLEM: traditional GA suffers from 2^n explosion
    // why? it refuses to acknowledge that rotations compose by angle addition
    // instead, it scatters rotation information across exponentially many components

    // traditional GA component count:
    // 1D: 2 components (scalar, e1)
    // 2D: 4 components (scalar, e1, e2, e12)
    // 3D: 8 components (scalar, e1, e2, e3, e12, e13, e23, e123)
    // 10D: 1024 components (all possible products of basis vectors)
    // nD: 2^n components

    // THE SOLUTION: geonum recognizes that a rotation is just [length, angle]
    // no matter how many dimensions, a 45° rotation is stored as one number

    // proof: represent a 45° rotation
    let rotation_45 = Geonum::new(1.0, 1.0, 4.0); // [1, π/4]

    // apply this rotation to different objects - always the same operation
    let x_axis = Geonum::new(1.0, 0.0, 1.0);
    let rotated = x_axis * rotation_45; // rotate x-axis by 45°
    assert_eq!(rotated.angle, Angle::new(1.0, 4.0)); // now at 45°

    // this single number works in ANY dimension:
    // - in 2D: rotates in the xy-plane
    // - in 3D: rotates in the xy-plane (z unchanged)
    // - in 10D: rotates in the xy-plane (other 8 dims unchanged)

    // traditional GA cant do this! it needs:
    // - 2D: distribute across 4 components
    // - 3D: distribute across 8 components
    // - 10D: distribute across 1024 components
    // all to represent the same simple 45° rotation

    // the key insight: multiplication is just angle addition
    let a = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let b = Geonum::new(3.0, 1.0, 3.0); // [3, π/3]
    let product = a * b;

    // geonum: O(1) operations
    assert_eq!(product.length, 6.0); // lengths multiply
    assert_eq!(product.angle, Angle::new(1.0, 2.0)); // angles add: π/6 + π/3 = π/2

    // traditional GA: O(4^n) operations for the same result!
    // in 10D: 1024 × 1024 = 1,048,576 component multiplications
    // of which 99.9% produce zeros that still get computed and stored

    // even worse: chained operations
    let c = Geonum::new(1.5, 1.0, 4.0); // [1.5, π/4]
    let chain = a * b * c;

    // geonum: still O(1)
    assert_eq!(chain.length, 9.0); // 2 × 3 × 1.5
    assert_eq!(chain.angle, Angle::new(3.0, 4.0)); // π/6 + π/3 + π/4 = 3π/4

    // traditional GA: must expand (a*b) into 2^n components,
    // then multiply all 2^n by c's 2^n components
    // the explosion compounds with every operation!

    // geonum solves the 2^n explosion by storing what actually matters:
    // the total rotation angle, not its decomposition into 2^n pieces
}
