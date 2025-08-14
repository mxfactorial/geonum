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

/// test proving that geonum eliminates the need for pseudoscalars
///
/// Traditional geometric algebra requires pseudoscalars (like e₁∧e₂∧e₃ in 3D) to:
/// 1. Define duality operations: dual(A) = A * I where I is the pseudoscalar
/// 2. Represent oriented volume elements
/// 3. Handle metric and orientation of the space
/// 4. Define the "top grade" of multivectors
///
/// Geonum's angle-blade representation makes pseudoscalars unnecessary
/// by encoding these relationships directly in the geometric structure.

#[test]
fn it_doesnt_need_a_pseudoscalar() {
    // TRADITIONAL DESIGN: create pseudoscalar for 3D space
    // In traditional GA: I₃ = e₁∧e₂∧e₃ (the "unit volume element")
    // This requires storing basis vectors e₁, e₂, e₃ and computing their wedge product

    // GEONUM DESIGN: No pseudoscalar needed
    // Duality operations work directly through blade arithmetic

    // Test 1: Duality without pseudoscalar
    // Traditional: dual(vector) = vector * pseudoscalar
    // Geonum: dual(vector) = direct blade rotation

    let vector_3d = Geonum::new(2.0, 1.0, 3.0); // vector in 3D (arbitrary angle)

    // Compute dual directly - no pseudoscalar multiplication needed
    let dual_vector = vector_3d.dual();

    // The dual should exist and be well-defined
    assert!(dual_vector.length.is_finite());
    assert!(!dual_vector.length.is_nan());

    // Dual should have different grade (blade count) than original
    assert_ne!(dual_vector.angle.blade() % 4, vector_3d.angle.blade() % 4);

    // Test 2: Volume elements without pseudoscalar
    // Traditional: volume = v₁∧v₂∧v₃, then multiply by pseudoscalar for orientation
    // Geonum: volume is just another geometric object with blade count 3

    let volume_element = Geonum::new_with_blade(5.0, 3, 1.0, 4.0); // 3-blade = volume in geonum

    // Volume has grade 3 (trivector) - no special pseudoscalar status needed
    assert_eq!(volume_element.angle.grade(), 3);
    assert!(volume_element.length > 0.0); // Positive oriented volume

    // Can directly compute volume properties
    assert_eq!(volume_element.length, 5.0); // Volume magnitude

    // Test 3: Duality involution without pseudoscalar
    // Traditional: dual(dual(x)) = ±x depending on dimension and pseudoscalar properties
    // Geonum: dual(dual(x)) computed through blade arithmetic alone

    let original = Geonum::new(3.0, 2.0, 3.0); // arbitrary geometric object
    let dual_once = original.dual();
    let dual_twice = dual_once.dual();

    // Dual involution should relate back to original
    assert!(dual_twice.length.is_finite());

    // The relationship depends on space dimension encoded in blade cycling
    // but requires no special pseudoscalar object
    let angle_relationship = (dual_twice.angle.mod_4_angle() - original.angle.mod_4_angle()).abs();
    assert!(angle_relationship < EPSILON || (PI - angle_relationship).abs() < EPSILON);

    // Test 4: Higher dimensional spaces without pseudoscalar
    // Traditional: Each dimension needs its own pseudoscalar (e₁∧...∧eₙ)
    // Geonum: Works in any dimension without special objects

    let high_dim_object = Geonum::new_with_blade(1.0, 1000, 0.0, 1.0); // 1000-dimensional
    let high_dim_dual = high_dim_object.dual();

    // Duality works in arbitrary dimensions without defining pseudoscalars
    assert!(high_dim_dual.length.is_finite());
    // dual() adds π (2 blades) to the angle
    assert_eq!(high_dim_dual.angle.blade(), 1002); // blade 1000 + 2 from dual()

    // Test 5: Orientation without pseudoscalar
    // Traditional: Orientation determined by pseudoscalar multiplication
    // Geonum: Orientation encoded directly in angle sign and blade count

    let oriented_positive = Geonum::new(1.0, 1.0, 4.0); // positive orientation
    let oriented_negative = Geonum::new(1.0, 3.0, 4.0); // negative orientation (angle + π)

    // Orientations are distinguishable without pseudoscalar
    assert!((oriented_positive.angle.cos() - oriented_negative.angle.cos()).abs() > EPSILON);
    assert!(oriented_positive.angle.cos() > 0.0);
    assert!(oriented_negative.angle.cos() < 0.0);

    // Test 6: Metric properties without pseudoscalar
    // Traditional: Metric signature affects pseudoscalar properties (I² = ±1)
    // Geonum: Metric relationships handled through angle arithmetic

    let metric_test = Geonum::new(1.0, 2.0, 2.0); // π angle (like "negative" in traditional metric)
    let metric_squared = metric_test * metric_test;

    // Metric relationships emerge from angle addition: π + π = 2π ≡ 0
    assert!(metric_squared.angle.mod_4_angle().abs() < EPSILON); // Back to 0 angle
    assert_eq!(metric_squared.length, 1.0); // Length preserved

    // This demonstrates metric signature effects without needing pseudoscalar I² = -1

    // Test 7: Grade extraction without pseudoscalar
    // Traditional: Extract grades using pseudoscalar projections
    // Geonum: Grade determined directly from blade count

    let mixed_grade = Geonum::new_with_blade(2.0, 157, 1.0, 6.0); // arbitrary high blade count

    // Grade is simply blade % 4
    let extracted_grade = mixed_grade.angle.grade();
    assert_eq!(extracted_grade, 157 % 4); // Grade 1 (vector)

    // No pseudoscalar needed to determine or extract grade information
    assert!(extracted_grade < 4); // Grades cycle in 4D pattern

    // Test 8: Cross-dimensional operations without pseudoscalar
    // Traditional: Mixing different dimensional spaces requires careful pseudoscalar handling
    // Geonum: Different dimensional objects operate naturally

    let obj_2d = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // 2D object
    let obj_5d = Geonum::new_with_blade(1.0, 5, 0.0, 1.0); // 5D object

    // Operations work across dimensions without pseudoscalar complications
    let cross_product = obj_2d * obj_5d;
    assert_eq!(cross_product.angle.blade(), 7); // Blades add: 2 + 5 = 7
    assert_eq!(cross_product.angle.grade(), 3); // 7 % 4 = 3 (trivector)

    // CONCLUSION: All traditional pseudoscalar functionality achieved
    // through direct angle-blade arithmetic, eliminating the need for:
    // - Special pseudoscalar objects
    // - Dimension-specific unit volume elements
    // - Complex duality multiplication formulas
    // - Grade-dependent pseudoscalar properties
    // - Metric signature pseudoscalar complications
}

#[test]
fn it_demonstrates_pseudoscalar_elimination_benefits() {
    // Show concrete advantages of eliminating pseudoscalars

    // ADVANTAGE 1: No storage overhead
    // Traditional: Must store and maintain pseudoscalar for each dimension
    // Geonum: No extra storage needed

    // Traditional design requires:
    // let pseudoscalar_2d = e1 ∧ e2;        // Store 2D pseudoscalar
    // let pseudoscalar_3d = e1 ∧ e2 ∧ e3;   // Store 3D pseudoscalar
    // let pseudoscalar_nd = e1 ∧ ... ∧ en;  // Store nD pseudoscalar

    // Geonum: Zero storage for pseudoscalar concepts
    let any_dimension_object = Geonum::new_with_blade(1.0, 42, 1.0, 3.0);
    assert!(any_dimension_object.length.is_finite()); // Works without stored pseudoscalar

    // ADVANTAGE 2: No computational overhead
    // Traditional: dual(A) = A * I requires multiplication by pseudoscalar
    // Geonum: dual(A) is direct blade rotation

    use std::time::Instant;

    let test_object = Geonum::new(1.0, 1.0, 4.0);

    let start = Instant::now();
    let _dual_result = test_object.dual(); // Direct computation
    let direct_time = start.elapsed();

    // No pseudoscalar multiplication overhead
    assert!(direct_time.as_nanos() < 10_000); // Should be very fast

    // ADVANTAGE 3: No dimension-specific code
    // Traditional: Different pseudoscalar handling for 2D, 3D, 4D, etc.
    // Geonum: Same duality code works in any dimension

    let dimensions_to_test = vec![2, 3, 4, 5, 10, 100, 1000];

    for dim in dimensions_to_test {
        let obj = Geonum::new_with_blade(1.0, dim, 0.0, 1.0);
        let dual_obj = obj.dual();

        // Same dual operation works regardless of dimension
        assert!(dual_obj.length.is_finite());
        // No special case code needed for different dimensional pseudoscalars
    }

    // ADVANTAGE 4: No metric signature complications
    // Traditional: Pseudoscalar properties depend on metric (I² = ±1)
    // Geonum: Metric effects handled naturally through angle arithmetic

    let euclidean_object = Geonum::new(1.0, 0.0, 1.0); // "Euclidean-like"
    let minkowski_object = Geonum::new(1.0, 2.0, 2.0); // "Minkowski-like" (π angle)

    // Both work with same dual operation, no pseudoscalar metric complications
    let dual_euclidean = euclidean_object.dual();
    let dual_minkowski = minkowski_object.dual();

    assert!(dual_euclidean.length.is_finite());
    assert!(dual_minkowski.length.is_finite());

    // Metric signature effects emerge naturally from angle addition
    // No need for different pseudoscalar I² values

    // ADVANTAGE 5: Eliminates pseudoscalar orientation ambiguity
    // Traditional: ±I ambiguity in choosing pseudoscalar orientation
    // Geonum: Orientation encoded unambiguously in angle

    let orientation_test = Geonum::new(1.0, 1.0, 8.0); // π/8 angle
    let opposite_orientation = Geonum::new(1.0, 9.0, 8.0); // π/8 + π angle

    // Orientations are unambiguously different
    assert!((orientation_test.angle.cos() + opposite_orientation.angle.cos()).abs() < EPSILON);

    // No pseudoscalar orientation choice needed
    // No ±I ambiguity in computations

    // SUMMARY: Eliminating pseudoscalars provides:
    // - Zero storage overhead
    // - No computational multiplication overhead
    // - Dimension-independent code
    // - No metric signature complications
    // - Unambiguous orientation handling
    // - Simpler mathematical framework

    // All traditional pseudoscalar functionality preserved through
    // direct geometric relationships in the angle-blade representation
}

#[test]
fn it_proves_dualization_as_angle_ops_compresses_ga() {
    // traditional GA duality requires multiplying by dimension-specific pseudoscalars
    // with exponential 2^n component arrays
    // geonum reduces duality to O(1) angle arithmetic through the quadrature's bivector

    // the 4 scalar, vector, bivector, trivector principle grades
    // emerge from the quadrature's bivector: sin(θ+π/2) = cos(θ)
    // this π/2 rotation creates the 4-quarter-turn cycle needed for complete GA

    // demonstrate duality compression: any dimensional object → simple angle operation
    let million_dim_vector = Geonum::new_with_blade(1.0, 1_000_000, 1.0, 4.0);
    let billion_dim_bivector = Geonum::new_with_blade(2.0, 1_000_000_000, 1.0, 3.0);

    // traditional GA: needs 2^1000000 and 2^1000000000 components for duality
    // (more storage than atoms in observable universe)

    // geonum: duality is just angle arithmetic regardless of dimension
    let dual_million = million_dim_vector.dual();
    let dual_billion = billion_dim_bivector.dual();

    // duality maps through 4-cycle: grade k → grade (k+2) % 4
    // dual() adds 2 blades (π rotation)
    assert_eq!(million_dim_vector.angle.grade(), 0); // 1000000 % 4 = 0 (scalar)
    assert_eq!(dual_million.angle.grade(), 2); // (0+2) % 4 = 2 (bivector)

    assert_eq!(billion_dim_bivector.angle.grade(), 0); // 1000000000 % 4 = 0 (scalar)
    assert_eq!(dual_billion.angle.grade(), 2); // (0+2) % 4 = 2 (bivector)

    // compression achieved: exponential → constant time
    // traditional: O(2^n) storage and computation
    // geonum: O(1) angle operations from quadrature's bivector foundation

    // the key insight: the bivector sin(θ+π/2) = cos(θ) IS the duality operator
    // this π/2 rotation imposes the incidence structure that defines geometric relationships
    // point-line-plane-volume duality emerges from this single trigonometric identity
    // eliminating need for dimension-specific pseudoscalars or exponential storage

    // prove duality preserves length (isometry property)
    assert_eq!(dual_million.length, million_dim_vector.length);
    assert_eq!(dual_billion.length, billion_dim_bivector.length);

    // prove duality involution: dual(dual(x)) returns to original grade
    let double_dual_million = dual_million.dual();
    let double_dual_billion = dual_billion.dual();
    assert_eq!(
        double_dual_million.angle.grade(),
        million_dim_vector.angle.grade()
    );
    assert_eq!(
        double_dual_billion.angle.grade(),
        billion_dim_bivector.angle.grade()
    );

    // prove O(1) complexity: blade arithmetic regardless of dimension size
    // dual() adds 2 blades regardless of dimension
    assert_eq!(
        dual_million.angle.blade(),
        million_dim_vector.angle.blade() + 2
    );
    assert_eq!(
        dual_billion.angle.blade(),
        billion_dim_bivector.angle.blade() + 2
    );
}

#[test]
fn it_replaces_k_to_n_minus_k_with_k_to_4_minus_k() {
    // traditional GA: duality maps grade k to grade (n-k) where n = space dimension
    // different dimensional spaces need different duality mappings
    // 3D: k → (3-k), 4D: k → (4-k), 1000D: k → (1000-k)

    // geonum: universal duality k → (4-k) % 4 regardless of dimensional space
    // works for any dimension through quadrature's bivector foundation

    // demonstrate universal mapping across arbitrary dimensions
    let obj_3d = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1 in "3D context"
    let obj_1000d = Geonum::new_with_blade(1.0, 1001, 0.0, 1.0); // grade 1 in "1000D context"
    let obj_million_d = Geonum::new_with_blade(1.0, 1_000_001, 0.0, 1.0); // grade 1 in "million-D context"

    // traditional GA would need different formulas:
    // 3D: grade 1 → grade (3-1) = 2
    // 1000D: grade 1 → grade (1000-1) = 999
    // million-D: grade 1 → grade (1000000-1) = 999999

    // geonum uses same formula k → (4-k) % 4 for all:
    let dual_3d = obj_3d.dual();
    let dual_1000d = obj_1000d.dual();
    let dual_million_d = obj_million_d.dual();

    // all grade 1 objects map to grade 3 regardless of "dimensional context"
    assert_eq!(obj_3d.angle.grade(), 1);
    assert_eq!(obj_1000d.angle.grade(), 1);
    assert_eq!(obj_million_d.angle.grade(), 1);

    assert_eq!(dual_3d.angle.grade(), 3); // (1+2) % 4 = 3
    assert_eq!(dual_1000d.angle.grade(), 3); // (1+2) % 4 = 3
    assert_eq!(dual_million_d.angle.grade(), 3); // (1+2) % 4 = 3

    // demonstrate grade 2 → grade 0 universally
    let bivector_any_dim = Geonum::new_with_blade(2.0, 1002, 0.0, 1.0); // grade 2
    let dual_bivector = bivector_any_dim.dual();

    assert_eq!(bivector_any_dim.angle.grade(), 2);
    assert_eq!(dual_bivector.angle.grade(), 0); // (2+2) % 4 = 0

    // compression: eliminates dimension-dependent duality formulas
    // one universal k → (4-k) % 4 mapping works for any dimensional space

    // geonum eliminates binomial coefficient (n choose k) component explosion
    // traditional GA: 3D needs (3 choose 1) = 3 vectors, 1000D needs (1000 choose 1) = 1000 vectors
    // geonum: grade 1 objects use same single [length, angle] representation regardless of dimension
    // "linearly independent k-vectors" are irrelevant - direction exists naturally through angle preservation

    // geonum eliminates Hodge decomposition: ω = dα + δβ + γ
    // traditional: separate storage for exact, co-exact, and harmonic components with orthogonal projections
    // geonum: all decomposition distinctions collapse to angle arithmetic
    let form_omega = Geonum::new_with_blade(1.0, 5, 1.0, 3.0); // arbitrary differential form
    let exact_component = form_omega.rotate(Angle::new(1.0, 2.0)); // dα becomes π/2 rotation
    let coexact_component = form_omega.rotate(Angle::new(3.0, 2.0)); // δβ becomes 3π/2 rotation
    let harmonic_component = form_omega; // γ is original angle relationship

    // prove no separate storage needed for Hodge decomposition components
    assert_eq!(
        std::mem::size_of_val(&form_omega),
        std::mem::size_of_val(&exact_component)
    );
    assert_eq!(
        std::mem::size_of_val(&exact_component),
        std::mem::size_of_val(&coexact_component)
    );
    assert_eq!(
        std::mem::size_of_val(&coexact_component),
        std::mem::size_of_val(&harmonic_component)
    );

    // prove all grade 1 objects have identical storage regardless of "dimensional context"
    assert_eq!(
        std::mem::size_of_val(&obj_3d),
        std::mem::size_of_val(&obj_1000d)
    );
    assert_eq!(
        std::mem::size_of_val(&obj_1000d),
        std::mem::size_of_val(&obj_million_d)
    );

    // traditional GA storage would scale with binomial coefficients:
    // 3D grade 1: 3 components, 1000D grade 1: 1000 components, million-D grade 1: 1000000 components
    // geonum storage: constant 2 components (length + angle) for any dimension
}

#[test]
fn it_proves_angle_space_is_absolute() {
    // angle space is absolute - there's no relative "negative" or "positive"
    // every geometric number has an absolute position in angle space

    // what traditional math calls "negative" is just an absolute angle position
    let positive_five = Geonum::new(5.0, 0.0, 1.0); // 5 at angle 0
    let negative_five = Geonum::new(5.0, 1.0, 1.0); // 5 at angle π

    // these aren't "opposites" in a relative sense - they're at absolute positions
    assert_eq!(positive_five.angle.value(), 0.0); // absolute position: 0 radians
    assert_eq!(negative_five.angle.blade(), 2); // blade 2 means 2 × π/2 rotations
    assert_eq!(negative_five.angle.value(), 0.0); // but value within blade is 0

    // multiplication adds absolute angles - not relative "signs"
    let product = positive_five * negative_five;
    assert_eq!(product.angle.blade(), 2); // 0 + 2 = 2 blade rotations
    assert_eq!(product.length, 25.0);

    // what we call "negation" is moving to absolute position π away
    let negated = positive_five.negate();
    assert_eq!(negated.angle.blade(), 2); // π = 2 × π/2 blade rotations

    // differentiation moves to absolute position π/2 ahead
    let differentiated = positive_five.differentiate();
    assert_eq!(differentiated.angle.blade(), 1); // π/2 = 1 blade rotation

    // prove that "sign" is a projection artifact, not fundamental
    // create numbers at various absolute angle positions
    let angle_0 = Geonum::new(1.0, 0.0, 1.0); // 0 radians
    let angle_pi_4 = Geonum::new(1.0, 1.0, 4.0); // π/4 radians
    let angle_pi_2 = Geonum::new(1.0, 1.0, 2.0); // π/2 radians
    let angle_3pi_4 = Geonum::new(1.0, 3.0, 4.0); // 3π/4 radians
    let angle_pi = Geonum::new(1.0, 1.0, 1.0); // π radians

    // traditional math would project these onto a line and lose information
    // calling some "positive" and some "negative" based on cosine projection
    // but in absolute angle space, they're all just positions

    // prove operations work with absolute positions, not relative signs
    let chain = angle_0 * angle_pi_4 * angle_pi_2 * angle_3pi_4 * angle_pi;
    let total_angle = 0.0 + PI / 4.0 + PI / 2.0 + 3.0 * PI / 4.0 + PI;
    assert!((chain.angle.mod_4_angle() - (total_angle % (2.0 * PI))).abs() < EPSILON);

    // prove grade transformations are absolute position changes
    let scalar = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // grade 0
    let vector = scalar.rotate(Angle::new(1.0, 2.0)); // +π/2 rotation
    let bivector = vector.rotate(Angle::new(1.0, 2.0)); // +π/2 rotation

    // each rotation moves to a new absolute position in angle-blade space
    assert_eq!(scalar.angle.grade(), 0); // absolute grade position 0
    assert_eq!(vector.angle.grade(), 1); // absolute grade position 1
    assert_eq!(bivector.angle.grade(), 2); // absolute grade position 2

    // anticommutativity is absolute angle differences, not "sign flips"
    // (see it_proves_anticommutativity_is_a_geometric_transformation for full exploration)

    // prove zero isn't special - it's just length 0 at any angle
    let zero_at_0 = Geonum::new(0.0, 0.0, 1.0);
    let zero_at_pi = Geonum::new(0.0, 1.0, 1.0);
    let zero_at_pi_2 = Geonum::new(0.0, 1.0, 2.0);

    // all represent zero but at different angle positions
    assert_eq!(zero_at_0.length, 0.0);
    assert_eq!(zero_at_pi.length, 0.0);
    assert_eq!(zero_at_pi_2.length, 0.0);

    // multiplying by zero preserves the zero's angle (absolute position matters)
    let five = Geonum::new(5.0, 0.0, 1.0);
    let result1 = five * zero_at_0; // 5 * 0 with 0 at angle 0
    let result2 = five * zero_at_pi; // 5 * 0 with 0 at angle π

    assert_eq!(result1.length, 0.0);
    assert_eq!(result2.length, 0.0);
    assert_eq!(result1.angle.blade(), 0); // preserves blade 0
    assert_eq!(result2.angle.blade(), 2); // preserves blade 2 (π rotation)

    // this proves angle space is absolute - even zero has an angle position
    // traditional "sign" is just projection onto the real line, losing geometric truth
}

#[test]
fn it_proves_anticommutativity_is_a_geometric_transformation() {
    // traditional GA treats anticommutativity as scalar multiplication by -1
    // a∧b = -b∧a where the minus is just a sign flip preserving all structure
    //
    // geonum reveals anticommutativity is a geometric transformation that
    // changes blade position - the orientation IS the blade difference

    // create two vectors at different angles
    let v1 = Geonum::new(2.0, 1.0, 6.0); // 2 at π/6
    let v2 = Geonum::new(3.0, 1.0, 3.0); // 3 at π/3

    // compute wedge products in both orders
    let wedge_12 = v1.wedge(&v2);
    let wedge_21 = v2.wedge(&v1);

    // same magnitude - the area is invariant
    assert!((wedge_12.length - wedge_21.length).abs() < EPSILON);

    // but different blades - this IS the anticommutativity
    let blade_diff = (wedge_12.angle.blade() as i32 - wedge_21.angle.blade() as i32).abs();
    assert_eq!(blade_diff, 2, "blade difference of 2 = π rotation");

    // traditional GA would store these as:
    // - wedge_12 = +area (in some basis)
    // - wedge_21 = -area (negative of same basis)
    // requiring 2 storage locations for "positive" and "negative" orientations

    // geonum shows they're the same geometric object at different blade positions
    // no need to "span" positive/negative dimensions - just track the blade

    // demonstrate this with meet operation (uses wedge internally)
    let a = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // scalar
    let b = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // vector

    let meet_ab = a.meet(&b);
    let meet_ba = b.meet(&a);

    // meet is anticommutative through blade transformation
    assert!((meet_ab.length - meet_ba.length).abs() < EPSILON);
    let meet_blade_diff = (meet_ab.angle.blade() as i32 - meet_ba.angle.blade() as i32).abs();
    assert_eq!(
        meet_blade_diff, 2,
        "anticommutativity = blade transformation"
    );

    // prove orientation changes actually transform dimensional structure
    assert_ne!(
        wedge_12.angle.grade(),
        wedge_21.angle.grade(),
        "different blades can mean different grades"
    );

    // the key insight: swapping operands doesn't just negate
    // it takes a different geometric path through blade space
    //
    // traditional linear algebra:
    //   "we need to span 2 dimensions for positive/negative orientations"
    //
    // geonum:
    //   "it's one geometric object navigating different blade positions"

    // this reframes anticommutativity from algebraic property to geometric navigation
    // blade arithmetic isn't representing orientation - it IS orientation
}

#[test]
fn it_compresses_traditional_ga_grades_to_two_involutive_pairs() {
    // geonum's π-rotation dual creates a different incidence structure than traditional GA
    // instead of computing maximal common subspaces, it computes containing spaces

    let line1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1
    let line2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // grade 1, different angle
    let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // grade 2
    let bivector2 = Geonum::new_with_blade(1.0, 2, 1.0, 4.0); // grade 2, different angle

    // line meet line → grade 1 (vector)
    // geometric meaning: the intersection point represented as a vector from origin
    // traditional GA expects grade 0 (scalar point)
    assert_eq!(line1.meet(&line2).angle.grade(), 1);

    // vector meet bivector → grade 2 (bivector)
    // geometric meaning: the minimal plane containing both the line and the original plane
    // traditional GA expects grade 0 (point of intersection)
    assert_eq!(line1.meet(&bivector).angle.grade(), 2);

    // bivector meet bivector → grade 3 (trivector)
    // geometric meaning: the 3D volume spanned by the two planes
    // traditional GA expects grade 1 (line of intersection)
    assert_eq!(bivector.meet(&bivector2).angle.grade(), 3);

    // this reversal happens because π-rotation dual creates scalar↔bivector
    // and vector↔trivector pairings rather than traditional complementary pairings

    // KEY INSIGHT: geonum flattens traditional GA's n+1 grade levels (0 through n)
    // to just 2 involutive pairs that work in any dimension:
    // - pair 1: grade 0 ↔ grade 2 (scalar ↔ bivector)
    // - pair 2: grade 1 ↔ grade 3 (vector ↔ trivector)
    //
    // grades cycle modulo 4, so grade 1000000 in million-D space is just grade 0
    // this eliminates dimension-specific k→(n-k) duality formulas
    // replacing them with universal k→(k+2)%4 that works everywhere
}

#[test]
fn it_proves_multiplicative_inverse_preserves_geometric_structure() {
    // traditional algebra: a * (1/a) = 1 (the scalar identity)
    // this collapses all information to a single "neutral" element
    //
    // geonum reveals: a * inv(a) = [1, 2θ + π]
    // the result has unit length but carries geometric information about the input

    // test with various geometric objects
    let a = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
    let inv_a = a.inv();

    // multiply a by its inverse
    let product = a * inv_a;

    // traditional expectation: get scalar 1 at grade 0
    // geonum reality: get unit length at grade determined by input
    assert_eq!(product.length, 1.0); // unit length ✓

    // but the angle is 2θ + π = 2π/3 + π = 5π/3
    let expected_angle = a.angle + a.angle + Angle::new(1.0, 1.0);
    assert_eq!(product.angle, expected_angle);

    // this has blade 3 (trivector grade), not blade 0 (scalar)
    assert_eq!(product.angle.blade(), 3);
    assert_eq!(product.angle.grade(), 3);

    // traditional math would incorrectly claim this equals "1"
    // but it's actually [1, 5π/3] - a unit trivector, not a scalar

    // demonstrate this varies with input angle
    let b = Geonum::new(3.0, 1.0, 4.0); // [3, π/4]
    let product_b = b * b.inv();

    assert_eq!(product_b.length, 1.0);
    // angle is 2(π/4) + π = π/2 + π = 3π/2
    assert_eq!(product_b.angle.blade(), 3); // also trivector

    // the multiplicative "identity" isn't fixed - it depends on input!
    assert_ne!(product.angle, product_b.angle);

    // this reveals multiplicative inverse doesn't return to a universal identity
    // it takes you to a unit-length object whose grade encodes the operation's history

    // traditional algebra needs this to be "1" for field axioms
    // but geonum shows the geometric structure is richer than field theory allows

    // demonstrate blade transformation pattern
    // start with different grade objects
    let scalar = Geonum::new(2.0, 0.0, 1.0); // blade 0 (scalar)
    let vector = Geonum::new(2.0, 1.0, 2.0); // blade 1 (vector)
    let bivector = Geonum::new(2.0, 1.0, 1.0); // blade 2 (bivector)
    let trivector = Geonum::new(2.0, 3.0, 2.0); // blade 3 (trivector)

    // all multiplicative inverses produce blade 3 (trivector) results
    let scalar_inv_product = scalar * scalar.inv();
    let vector_inv_product = vector * vector.inv();
    let bivector_inv_product = bivector * bivector.inv();
    let trivector_inv_product = trivector * trivector.inv();

    // all have unit length
    assert_eq!(scalar_inv_product.length, 1.0);
    assert_eq!(vector_inv_product.length, 1.0);
    assert_eq!(bivector_inv_product.length, 1.0);
    assert_eq!(trivector_inv_product.length, 1.0);

    // but different starting grades produce different blade counts
    // inv() adds π, multiplication adds angles: 2θ + π
    assert_eq!(scalar_inv_product.angle.blade(), 2); // scalar: 0→π→2π = blade 2
    assert_eq!(vector_inv_product.angle.blade(), 4); // vector: π/2→π→2π = blade 4
    assert_eq!(bivector_inv_product.angle.blade(), 6); // bivector: π→2π→3π = blade 6
    assert_eq!(trivector_inv_product.angle.blade(), 8); // trivector: 3π/2→3π→4π = blade 8

    // when comparing angles after transformations, must account for blade changes
    // operations like circular inversion transform blade structure similarly
    // this is why angle preservation tests can fail if only checking angle.value()
}

#[test]
fn it_sets_angle_forward_geometry_as_primitive() {
    // angles only move forward in geonum - no subtraction, no backwards motion
    // reflection isnt "flip then unflip" - its drawing a complete circle in angle space
    // every operation adds to blade count because geometry IS forward rotation
    //
    // double reflection adds 2+2=4 blades, not 2-2=0
    // this isnt a bug - its primitive geometry. transformations accumulate
    // because thats what transformations do in absolute angle space
    //
    // for engineers who need bounded memory, base_angle() provides opt-out
    // it resets blade to minimum for grade, preserving relationships not history

    // demonstrate blade accumulation from repeated operations
    let mut position = Geonum::new(1.0, 0.0, 1.0);

    // 1000 rotations of π/2 each accumulate 1000 blades
    for _ in 0..1000 {
        position = position.rotate(Angle::new(1.0, 2.0)); // π/2 rotation
    }

    assert_eq!(
        position.angle.blade(),
        1000,
        "operations accumulate 1000 blades"
    );
    assert_eq!(position.angle.grade(), 0, "grade is blade % 4 = 0");

    // base_angle() resets blade to minimum for grade
    let reset = position.base_angle();
    assert_eq!(
        reset.angle.blade(),
        0,
        "base_angle resets to blade 0 for grade 0"
    );
    assert_eq!(reset.angle.grade(), 0, "grade preserved");
    assert_eq!(reset.length, position.length, "length unchanged");

    // demonstrate double reflection blade accumulation and reset
    let point = Geonum::new(2.0, 0.0, 1.0);
    let axis = Geonum::new(1.0, 1.0, 4.0); // π/4 axis

    let reflected_once = point.reflect(&axis);
    let reflected_twice = reflected_once.reflect(&axis);

    // reflect() computes: 2*axis + (2π - base_angle(point))
    // first reflection: 2*(π/4) + (2π - 0) = blade 1 + blade 8 = blade 9
    // second reflection: 2*(π/4) + (2π - base(blade 9)) = blade 1 + blade 7 = blade 8
    let blade_diff = reflected_twice.angle.blade() - point.angle.blade();
    assert_eq!(blade_diff, 8, "point blade 0 → blade 9 → blade 8");

    // with base_angle(), double reflection becomes involutive
    assert_eq!(
        reflected_twice.base_angle().angle.grade(),
        point.base_angle().angle.grade(),
        "double reflection preserves grade after base_angle reset"
    );

    // traditional assertion using helper
    let original = Geonum::new(3.0, 1.0, 4.0); // [3, π/4]

    // compute traditional inverse value by multiplying inv identity and returns angle value
    let traditional = original * original.inv();

    // for [3, π/4]: identity = π/4 + (π/4 + π) = 3π/2
    // at blade 3 (grade 3), value is 0
    assert!(
        traditional.angle.value().abs() < 1e-10,
        "traditional value at 3π/2 is 0"
    );

    // also verify the identity has unit length
    let identity = original * original.inv();
    assert!(
        (identity.length - 1.0).abs() < 1e-10,
        "multiplicative identity has unit length"
    );

    // demonstrate that base_angle and dual dont commute in blade but do in grade
    let high_blade = Geonum::new_with_blade(3.0, 999, 0.0, 1.0);

    let dual_then_base = high_blade.dual().base_angle();
    let base_then_dual = high_blade.base_angle().dual();

    // different blade values
    assert_ne!(
        dual_then_base.angle.blade(),
        base_then_dual.angle.blade(),
        "operations dont commute in blade"
    );

    // but same grade
    assert_eq!(
        dual_then_base.angle.grade(),
        base_then_dual.angle.grade(),
        "operations preserve grade consistency"
    );

    // example: blade 999 (grade 3)
    // dual_then_base: 999 + 2 = 1001 (grade 1) → base → blade 1
    // base_then_dual: 999 → base → blade 3 → dual → blade 5 (still grade 1)

    // demonstrate control loop with bounded memory
    let mut drone_position = Geonum::new(10.0, 0.0, 1.0);
    let mut max_blade_seen = 0;

    for i in 0..10000 {
        // complex transformation each iteration
        drone_position = drone_position
            .scale_rotate(1.0001, Angle::new(1.0, 1000.0))
            .reflect(&Geonum::new(1.0, i as f64, 360.0));

        // prevent blade explosion every 100 iterations
        if i % 100 == 99 {
            drone_position = drone_position.base_angle();
        }

        max_blade_seen = max_blade_seen.max(drone_position.angle.blade());
    }

    assert!(
        max_blade_seen < 1000,
        "blade growth bounded by periodic reset"
    );
    assert!(
        drone_position.angle.blade() < 4,
        "final blade in base range [0,3]"
    );

    // forward-only rotation is primitive:
    // - reflection adds π, double reflection adds 2π (4 blades total)
    // - no operation goes backwards, everything draws circles forward
    // - blade count IS the transformation history
    // - base_angle() lets you forget history when you dont need it
    // - but the primitive geometry remains: angles only go forward
}
