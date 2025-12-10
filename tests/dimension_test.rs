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
    // test geonum's non-euclidean blade geometry projection properties
    // dimensions emerge from blade rotations, not linear combinations

    // step 1: build dimensional structure through rotations (geonum's natural way)
    let base_scalar = Geonum::scalar(5.0);
    let rotated_vector = base_scalar.rotate(Angle::new(1.0, 4.0)); // π/4 rotation
    let final_geonum = rotated_vector.scale(2.0); // 10 units at π/4

    // step 2: compute projections onto blade dimensions
    let proj_blade_0 = final_geonum.project_to_dimension(0); // blade 0 space
    let proj_blade_1 = final_geonum.project_to_dimension(1); // blade 1 space
    let proj_blade_2 = final_geonum.project_to_dimension(2); // blade 2 space

    // step 3: test blade projection properties (not euclidean magnitude preservation)
    // each projection represents alignment with that blade space

    // test individual projections are trigonometrically consistent
    let geonum_angle = final_geonum.angle.grade_angle();
    let expected_blade_0 = final_geonum.mag * (geonum_angle - 0.0).cos(); // angle from blade 0
    let expected_blade_1 = final_geonum.mag * (geonum_angle - PI / 2.0).cos(); // angle from blade 1

    assert!(
        (proj_blade_0 - expected_blade_0).abs() < EPSILON,
        "blade 0 projection trigonometrically consistent"
    );

    assert!(
        (proj_blade_1 - expected_blade_1).abs() < EPSILON,
        "blade 1 projection trigonometrically consistent"
    );

    // step 4: test non-euclidean property - projections don't satisfy pythagorean theorem
    let euclidean_magnitude =
        (proj_blade_0.powi(2) + proj_blade_1.powi(2) + proj_blade_2.powi(2)).sqrt();
    let actual_magnitude = final_geonum.mag;

    // this will NOT be equal - proves non-euclidean blade geometry
    assert!(
        (euclidean_magnitude - actual_magnitude).abs() > 0.1,
        "blade projections are non-euclidean"
    );

    // step 5: test what IS preserved - the geometric relationships within blade structure
    // projections preserve relative angle relationships
    let blade_0_axis = Angle::new_with_blade(0, 0.0, 1.0);
    let blade_1_axis = Angle::new_with_blade(1, 0.0, 1.0);

    let angle_to_blade_0 = final_geonum.angle.project(blade_0_axis);
    let angle_to_blade_1 = final_geonum.angle.project(blade_1_axis);

    assert!(
        (proj_blade_0 - final_geonum.mag * angle_to_blade_0).abs() < EPSILON,
        "projection equals length times angle projection"
    );

    assert!(
        (proj_blade_1 - final_geonum.mag * angle_to_blade_1).abs() < EPSILON,
        "projection equals length times angle projection"
    );
}

#[test]
fn it_proves_quadrature_creates_dimensional_structure() {
    // the fundamental insight: sin(θ+π/2) = cos(θ) creates all dimensional structure
    // this quadrature relationship generates the 4-fold repetition
    // dimensions emerge from this trigonometric identity, not from coordinate spaces

    let entity = Geonum::new(1.0, 1.0, 6.0); // π/6 angle, unit magnitude

    // test the fundamental quadrature relationship: sin(θ+π/2) = cos(θ)
    let theta = entity.angle.grade_angle();
    let theta_plus_quarter = theta + PI / 2.0;
    assert!((theta.cos() - theta_plus_quarter.sin()).abs() < EPSILON);
    assert!((theta.sin() + theta_plus_quarter.cos()).abs() < EPSILON);

    // this quadrature manifests as orthogonal projections
    // cos²(θ) + sin²(θ) = 1 creates the pythagorean relationship
    let dim_0 = entity.project_to_dimension(0);
    let dim_1 = entity.project_to_dimension(1);
    let cos_projection = entity.mag * (theta - 0.0).cos(); // dimension 0
    let sin_projection = entity.mag * (theta - PI / 2.0).cos(); // dimension 1
    assert!((dim_0 - cos_projection).abs() < EPSILON);
    assert!((dim_1 - sin_projection).abs() < EPSILON);
    assert!((cos_projection.powi(2) + sin_projection.powi(2) - entity.mag.powi(2)).abs() < EPSILON);

    // the 4-fold cycle emerges from completing the trigonometric circle
    let dim_2 = entity.project_to_dimension(2);
    let dim_3 = entity.project_to_dimension(3);
    assert!((dim_0 - entity.project_to_dimension(4)).abs() < EPSILON); // 0 = 4 (mod 4)
    assert!((dim_1 - entity.project_to_dimension(5)).abs() < EPSILON); // 1 = 5 (mod 4)
    assert!((dim_2 - entity.project_to_dimension(6)).abs() < EPSILON); // 2 = 6 (mod 4)
    assert!((dim_3 - entity.project_to_dimension(7)).abs() < EPSILON); // 3 = 7 (mod 4)

    // prove the quadrature generates polynomial coefficients in calculus
    let base_function = Geonum::new(4.0, 0.0, 1.0); // grade 0 function
    let first_derivative = base_function.differentiate(); // π/2 rotation to grade 1
    let second_derivative = first_derivative.differentiate(); // π/2 rotation to grade 2
    let third_derivative = second_derivative.differentiate(); // π/2 rotation to grade 3
    let fourth_derivative = third_derivative.differentiate(); // π/2 rotation back to grade 0

    // the 4-cycle of differentiation comes from quadrature
    assert_eq!(base_function.angle.grade(), 0);
    assert_eq!(first_derivative.angle.grade(), 1);
    assert_eq!(second_derivative.angle.grade(), 2);
    assert_eq!(third_derivative.angle.grade(), 3);
    assert_eq!(fourth_derivative.angle.grade(), 0); // cycle complete

    // dimensional structure emerges from trigonometric identities
    // coordinate spaces emerge from dimensional structure, not the reverse
}

#[test]
fn it_shows_dimensions_are_quarter_turns() {
    // dimensions are angle positions separated by π/2 rotations
    // creating dimensions means rotating by quarter turns

    let entity = Geonum::new(2.0, 1.0, 8.0); // arbitrary entity

    // dimensions are angle positions, proven by direct construction
    let constructed_dim_0 = Geonum::create_dimension(1.0, 0); // 0 quarter turns
    let constructed_dim_1 = Geonum::create_dimension(1.0, 1); // 1 quarter turn
    let constructed_dim_2 = Geonum::create_dimension(1.0, 2); // 2 quarter turns
    let constructed_dim_3 = Geonum::create_dimension(1.0, 3); // 3 quarter turns

    // these are separated by exactly π/2 rotations
    assert!((constructed_dim_0.angle.grade_angle() - 0.0).abs() < EPSILON);
    assert!((constructed_dim_1.angle.grade_angle() - PI / 2.0).abs() < EPSILON);
    assert!((constructed_dim_2.angle.grade_angle() - PI).abs() < EPSILON);
    assert!((constructed_dim_3.angle.grade_angle() - 3.0 * PI / 2.0).abs() < EPSILON);

    // rotating between dimensions proves they are angle positions
    let rotated_0_to_1 = constructed_dim_0.rotate(Angle::new(1.0, 2.0)); // +π/2
    let rotated_1_to_2 = constructed_dim_1.rotate(Angle::new(1.0, 2.0)); // +π/2
    let rotated_2_to_3 = constructed_dim_2.rotate(Angle::new(1.0, 2.0)); // +π/2
    let rotated_3_to_0 = constructed_dim_3.rotate(Angle::new(1.0, 2.0)); // +π/2

    let blade_from_rotation = Angle::new(1.0, 2.0); // π/2 rotation
    assert_eq!(rotated_0_to_1.angle, constructed_dim_1.angle);
    assert_eq!(rotated_1_to_2.angle, constructed_dim_2.angle);
    assert_eq!(rotated_2_to_3.angle, constructed_dim_3.angle);
    assert_eq!(
        rotated_3_to_0.angle,
        constructed_dim_0.angle
            + blade_from_rotation
            + blade_from_rotation
            + blade_from_rotation
            + blade_from_rotation
    );

    // the blade % 4 arithmetic creates dimensional equivalence in any dimension count
    let proj_0 = entity.project_to_dimension(0);
    let proj_1000 = entity.project_to_dimension(1000);
    let proj_million = entity.project_to_dimension(1_000_000);

    assert!((proj_0 - proj_1000).abs() < EPSILON); // 0 = 1000 (mod 4)
    assert!((proj_0 - proj_million).abs() < EPSILON); // 0 = 1,000,000 (mod 4)

    // prove quarter turn relationships via pythagorean theorem
    let proj_0 = entity.project_to_dimension(0);
    let proj_1 = entity.project_to_dimension(1);
    let proj_2 = entity.project_to_dimension(2);
    let proj_3 = entity.project_to_dimension(3);

    // quarter turn projections are orthogonal
    assert!((proj_0.powi(2) + proj_1.powi(2) - entity.mag.powi(2)).abs() < EPSILON);
    assert!((proj_2.powi(2) + proj_3.powi(2) - entity.mag.powi(2)).abs() < EPSILON);

    // half turn projections are opposite
    assert!((proj_0 + proj_2).abs() < EPSILON);
    assert!((proj_1 + proj_3).abs() < EPSILON);
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
    assert_eq!(product.mag, 1.0);
    assert_eq!(product.angle.blade(), 1); // 90° rotation (blade 1)
    assert!(product.angle.rem().abs() < EPSILON); // exactly π/2

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
    assert!(xy_product.angle.rem().abs() < EPSILON);

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
    assert!(dot.mag.abs() < 1e-10); // zero

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
    assert!(dot_vectors.mag > 0.0); // positive, not zero!

    // geonum reveals the underlying pattern: PI/2 rotation changes grade
    // "orthogonal vectors" emerged from overlooking grade transformation
    // the angle-blade invariant makes explicit what grade decomposition obscured
}

#[test]
fn it_demonstrates_inversion_preserves_grade_parity_relationships() {
    // geonum's grade structure has involutive pairs: 0↔2, 1↔3
    // operations that preserve this pairing maintain orthogonality relationships
    // circular inversion is one such operation
    let center = Geonum::new_from_cartesian(0.0, 0.0); // origin for clarity
    let radius = 2.0;

    // test points at different angles and distances
    let test_configs = vec![
        // (distance, angle_pi_rad, angle_div, description)
        (1.0, 0.0, 1.0, "inside on +x axis"),
        (3.0, 0.0, 1.0, "outside on +x axis"),
        (1.0, 1.0, 2.0, "inside on +y axis"),
        (3.0, 1.0, 2.0, "outside on +y axis"),
        (1.0, 1.0, 4.0, "inside at π/4"),
        (3.0, 1.0, 4.0, "outside at π/4"),
        (1.0, 1.0, 1.0, "inside on -x axis"),
        (3.0, 1.0, 1.0, "outside on -x axis"),
    ];

    println!("\nSingle point inversions from origin:");
    for (dist, pi_rad, div, desc) in test_configs {
        let p = Geonum::new(dist, pi_rad, div);
        let p_inv = p.invert_circle(&center, radius);

        println!(
            "{}: dist={} angle={:.3} blade={} → dist={:.3} angle={:.3} blade={}",
            desc,
            dist,
            p.angle.rem(),
            p.angle.blade(),
            p_inv.mag,
            p_inv.angle.rem(),
            p_inv.angle.blade()
        );

        // verify inversion property
        assert!((p.mag * p_inv.mag - radius * radius).abs() < EPSILON);
    }

    // now test difference vectors between points (where blade changes occurred before)
    println!("\nDifference vectors between points:");

    // create a configuration that shows blade transformation
    let p1 = Geonum::new_from_cartesian(2.0, 1.0);
    let p2 = Geonum::new_from_cartesian(3.0, 0.0);
    let p3 = Geonum::new_from_cartesian(2.0, -1.0);

    // compute difference vectors
    let v12 = p2 - p1;
    let v13 = p3 - p1;
    let v23 = p3 - p2;

    println!("Original vectors:");
    println!(
        "  v12=p2-p1: length={:.3} angle={:.3} blade={}",
        v12.mag,
        v12.angle.rem(),
        v12.angle.blade()
    );
    println!(
        "  v13=p3-p1: length={:.3} angle={:.3} blade={}",
        v13.mag,
        v13.angle.rem(),
        v13.angle.blade()
    );
    println!(
        "  v23=p3-p2: length={:.3} angle={:.3} blade={}",
        v23.mag,
        v23.angle.rem(),
        v23.angle.blade()
    );

    // invert the points
    let p1_inv = p1.invert_circle(&center, radius);
    let p2_inv = p2.invert_circle(&center, radius);
    let p3_inv = p3.invert_circle(&center, radius);

    // compute inverted difference vectors
    let v12_inv = p2_inv - p1_inv;
    let v13_inv = p3_inv - p1_inv;
    let v23_inv = p3_inv - p2_inv;

    println!("Inverted vectors:");
    println!(
        "  v12_inv: length={:.3} angle={:.3} blade={}",
        v12_inv.mag,
        v12_inv.angle.rem(),
        v12_inv.angle.blade()
    );
    println!(
        "  v13_inv: length={:.3} angle={:.3} blade={}",
        v13_inv.mag,
        v13_inv.angle.rem(),
        v13_inv.angle.blade()
    );
    println!(
        "  v23_inv: length={:.3} angle={:.3} blade={}",
        v23_inv.mag,
        v23_inv.angle.rem(),
        v23_inv.angle.blade()
    );

    // KEY INSIGHT: blade transformation happens in difference vectors
    // individual points from origin maintain blade, but vectors between inverted points transform

    // test with points that create perpendicular vectors
    println!("\nPerpendicular vector configuration:");
    let center2 = Geonum::new_from_cartesian(1.0, 0.0); // offset center
    let q1 = Geonum::new_from_cartesian(3.0, 0.0);
    let q2 = Geonum::new_from_cartesian(4.0, 0.0);
    let q3 = Geonum::new_from_cartesian(3.0, 1.0);

    let u1 = q2 - q1; // horizontal
    let u2 = q3 - q1; // vertical

    println!("Original perpendicular vectors:");
    println!("  u1: blade={} (horizontal)", u1.angle.blade());
    println!("  u2: blade={} (vertical)", u2.angle.blade());

    // these perpendicular vectors have different blades (orthogonality via blade difference)
    assert_ne!(
        u1.angle.blade() % 2,
        u2.angle.blade() % 2,
        "perpendicular vectors differ by odd blade count"
    );

    let q1_inv = q1.invert_circle(&center2, radius);
    let q2_inv = q2.invert_circle(&center2, radius);
    let q3_inv = q3.invert_circle(&center2, radius);

    let u1_inv = q2_inv - q1_inv;
    let u2_inv = q3_inv - q1_inv;

    println!("Inverted 'perpendicular' vectors:");
    println!("  u1_inv: blade={}", u1_inv.angle.blade());
    println!("  u2_inv: blade={}", u2_inv.angle.blade());

    // blade relationships transform under inversion
    let blade_diff_original = (u2.angle.blade() as i32 - u1.angle.blade() as i32).abs();
    let blade_diff_inverted = (u2_inv.angle.blade() as i32 - u1_inv.angle.blade() as i32).abs();

    println!("Blade difference: {blade_diff_original} → {blade_diff_inverted}");

    // check if grade differences are preserved (blade mod 4)
    let grade_diff_original =
        ((u2.angle.grade() as i32 - u1.angle.grade() as i32).abs() % 4) as usize;
    let grade_diff_inverted =
        ((u2_inv.angle.grade() as i32 - u1_inv.angle.grade() as i32).abs() % 4) as usize;

    println!("Grade difference: {grade_diff_original} → {grade_diff_inverted}");

    // orthogonality is encoded in odd grade differences (parity)
    // grade 0 vs grade 1: difference = 1 (odd) → orthogonal
    // grade 2 vs grade 3: difference = 1 (odd) → orthogonal
    // grade 0 vs grade 2: difference = 2 (even) → parallel (dual pair)
    // grade 1 vs grade 3: difference = 2 (even) → parallel (dual pair)

    assert_eq!(
        grade_diff_original % 2,
        1,
        "original vectors are orthogonal (odd grade diff)"
    );
    assert_eq!(
        grade_diff_inverted % 2,
        1,
        "inverted vectors remain orthogonal (odd grade diff)"
    );

    // this is expected from geonum's involutive grade pairs (0↔2, 1↔3)
    // operations respecting this pairing preserve orthogonality parity
    // circular inversion is such an operation - it may shift grades within pairs
    // but preserves the odd/even nature of grade differences
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
    assert_eq!(product.mag, 6.0); // lengths multiply
    assert_eq!(product.angle, Angle::new(1.0, 2.0)); // angles add: π/6 + π/3 = π/2

    // traditional GA: O(4^n) operations for the same result!
    // in 10D: 1024 × 1024 = 1,048,576 component multiplications
    // of which 99.9% produce zeros that still get computed and stored

    // even worse: chained operations
    let c = Geonum::new(1.5, 1.0, 4.0); // [1.5, π/4]
    let chain = a * b * c;

    // geonum: still O(1)
    assert_eq!(chain.mag, 9.0); // 2 × 3 × 1.5
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
    // 1. DUALITY MAPPING: v* = v · I (multiply by pseudoscalar)
    //    traditional GA needs to:
    //    - define basis vectors e₁, e₂, e₃
    //    - compute pseudoscalar I = e₁∧e₂∧e₃
    //    - multiply vector by I to get dual
    //    geonum skips all that - dual() just adds 2 blades directly
    let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1 = vector
    let dual = vector.dual();
    assert_eq!(dual.angle.blade(), 3); // blade 1 + 2 = 3 (trivector)

    // 2. VOLUME ORIENTATION: I² = ±1 determines metric signature
    //    traditional: must compute pseudoscalar square to determine orientation
    //    geonum: dual operation handles orientation through blade arithmetic
    let volume = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // grade 3 volume element
    let dual_volume = volume.dual(); // duality transformation without pseudoscalar
    assert_eq!(dual_volume.angle.blade(), 5); // blade 3 + 2 = 5
    assert_eq!(dual_volume.angle.grade(), 1); // grade 5 % 4 = 1 (vector)
                                              // proves duality maps grade 3→1 without constructing I or computing I²

    // 3. HODGE STAR OPERATION: *v = v · I / |I|²
    //    traditional: matrix transformation + magnitude computation and verification
    //    geonum: direct blade transformation
    let vector = [3.0, 0.0, 0.0];
    let pseudoscalar_transform = [
        [0.0, 0.0, 1.0],  // x → yz dual
        [0.0, 0.0, -1.0], // y → zx dual
        [1.0, 0.0, 0.0],  // z → xy dual
    ];
    let mut dual_vector = [0.0; 3];
    for i in 0..3 {
        for (j, &vector_j) in vector.iter().enumerate() {
            dual_vector[i] += pseudoscalar_transform[i][j] * vector_j;
        }
    }
    // traditional hodge star requires magnitude verification: |*v| = |v|
    let vector_magnitude_sq: f64 =
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    let dual_magnitude_sq: f64 = dual_vector[0] * dual_vector[0]
        + dual_vector[1] * dual_vector[1]
        + dual_vector[2] * dual_vector[2];
    assert_eq!(vector_magnitude_sq.sqrt(), dual_magnitude_sq.sqrt()); // |I|² normalization preserved
    assert_eq!(dual_vector, [0.0, 0.0, 3.0]);

    // 4. GRADE EXTRACTION: extract k-grade part using pseudoscalar projections
    //    traditional: pseudoscalar projections for grade filtering
    //    geonum: unified operations eliminates grade extraction dependency
    let scale_factor = 2.0; // scalar component (grade 0)
    let rotation_angle = PI / 4.0; // bivector component (grade 2 via rotation)

    // traditional: extract grades using pseudoscalar projections
    // grade extraction formula: ⟨A⟩ₖ = (A ∧ Iᵏ) ⌋ I⁻ᵏ / k!

    // extract scalar (grade 0): ⟨A⟩₀ = A ⌋ I⁰ = A ⌋ 1 = scalar_part
    let scalar_part = scale_factor; // grade 0 extraction
    let pseudoscalar_grade0 = 1.0; // I⁰ = identity for scalar extraction
    assert_eq!(scalar_part * pseudoscalar_grade0, scale_factor); // pseudoscalar I⁰ appears in extraction

    // extract bivector (grade 2): ⟨A⟩₂ = (A ∧ I²) ⌋ I⁻² / 2!
    // in 2D: pseudoscalar I = e₁∧e₂, so I² = -1
    let pseudoscalar_2d = -1.0; // I² = (e₁∧e₂)² = -1 in 2D
    let bivector_coefficient = rotation_angle / pseudoscalar_2d; // divide by I² for extraction
    let rotation_matrix = [
        [rotation_angle.cos(), -rotation_angle.sin()],
        [rotation_angle.sin(), rotation_angle.cos()],
    ]; // grade 2 (bivector) as rotation matrix
    assert_eq!(bivector_coefficient, -rotation_angle); // pseudoscalar I² = -1 negates coefficient in extraction formula

    // apply decomposed operations: scaling then rotation matrix multiplication
    let input_point = [1.0, 0.0];
    let scaled_point = [input_point[0] * scalar_part, input_point[1] * scalar_part];
    let traditional_result = [
        rotation_matrix[0][0] * scaled_point[0] + rotation_matrix[0][1] * scaled_point[1],
        rotation_matrix[1][0] * scaled_point[0] + rotation_matrix[1][1] * scaled_point[1],
    ];

    // geonum: unified operation without grade extraction
    let input = Geonum::new(1.0, 0.0, 1.0);
    let geonum_result = input.scale_rotate(scale_factor, Angle::new(rotation_angle, PI));
    let geonum_x = geonum_result.mag * geonum_result.angle.grade_angle().cos();
    let geonum_y = geonum_result.mag * geonum_result.angle.grade_angle().sin();
    assert!((traditional_result[0] - geonum_x).abs() < 1e-10);
    assert!((traditional_result[1] - geonum_y).abs() < 1e-10);

    // 5. COMPLEMENT OPERATIONS: orthogonal complement via pseudoscalar
    //    traditional: multiply by pseudoscalar for complement A^⊥ = A · I
    //    geonum: complement through blade arithmetic (dual operation)

    // traditional: construct pseudoscalar I for complement operation
    let pseudoscalar_3d = 1.0; // I₃ = e₁∧e₂∧e₃ = +1 in right-handed 3D
    let line_vector = [1.0, 0.0, 0.0]; // line in x direction

    // traditional complement: A^⊥ = A · I (multiply by pseudoscalar)
    let traditional_complement = [
        line_vector[0] * pseudoscalar_3d,
        line_vector[1] * pseudoscalar_3d,
        line_vector[2] * pseudoscalar_3d,
    ]; // complement is the orthogonal subspace
    assert_eq!(traditional_complement, [1.0, 0.0, 0.0]);

    // geonum: complement through dual() blade arithmetic
    let line = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1 line
    let complement = line.dual(); // complement via blade addition
    assert_eq!(complement.angle.blade(), 3); // grade 1 → grade 3 (line → volume)
    assert_eq!(complement.mag, 1.0); // magnitude preserved

    // 6. CROSS PRODUCTS: v × w = (v ∧ w) · I^(-1) in 3D
    //    traditional: wedge then multiply by pseudoscalar inverse
    //    geonum: wedge product directly
    let v = [1.0, 0.0, 0.0]; // x unit vector
    let w = [0.0, 1.0, 0.0]; // y unit vector

    // traditional: compute wedge product v∧w
    let e12_coeff = v[0] * w[1] - v[1] * w[0]; // xy-component = 1
    let e23_coeff = v[1] * w[2] - v[2] * w[1]; // yz-component = 0
    let e31_coeff = v[2] * w[0] - v[0] * w[2]; // zx-component = 0

    // traditional: left-handed pseudoscalar I₃ = -1 (solution to sign error)
    let pseudoscalar_i3 = -1.0; // left-handed orientation
    let i3_squared = -1.0; // (e₁∧e₂∧e₃)² = -1 in 3D euclidean GA
    let pseudoscalar_inverse = pseudoscalar_i3 / i3_squared; // (-1) / (-1) = +1

    // traditional: cross product (v∧w) · I₃⁻¹
    let traditional_cross = [
        e23_coeff * pseudoscalar_inverse, // yz → x component
        e31_coeff * pseudoscalar_inverse, // zx → y component
        e12_coeff * pseudoscalar_inverse, // xy → z component
    ];
    assert_eq!(traditional_cross, [0.0, 0.0, 1.0]); // x × y = z

    // test magnitude |v × w| = |v| |w| sin(θ)
    let cross_magnitude_sq: f64 = traditional_cross[0] * traditional_cross[0]
        + traditional_cross[1] * traditional_cross[1]
        + traditional_cross[2] * traditional_cross[2];
    let cross_magnitude = cross_magnitude_sq.sqrt();
    let v_magnitude_sq: f64 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let v_magnitude = v_magnitude_sq.sqrt();
    let w_magnitude_sq: f64 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let w_magnitude = w_magnitude_sq.sqrt();
    let expected_magnitude = v_magnitude * w_magnitude * (PI / 2.0).sin(); // sin(90°) = 1
    assert_eq!(cross_magnitude, expected_magnitude);

    // geonum: wedge without pseudoscalar inverse computation
    let v_geonum = Geonum::new_from_cartesian(1.0, 0.0);
    let w_geonum = Geonum::new_from_cartesian(0.0, 1.0);
    let geonum_wedge = v_geonum.wedge(&w_geonum);
    assert_eq!(geonum_wedge.mag, 1.0); // magnitude preserved without pseudoscalar operations

    // 7. NORMAL VECTORS: surface normals via pseudoscalar multiplication
    //    traditional: compute tangent cross product then multiply pseudoscalar
    //    geonum: normals through grade parity relationships

    // traditional: surface normal = (v1 × v2) · I₃⁻¹ requires pseudoscalar
    // geonum: surface normal via grade parity - odd differences are orthogonal

    // create surface as bivector (grade 2)
    let surface_x = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let surface_y = surface_x.rotate(Angle::new(1.0, 2.0)); // grade 1
    let surface_plane = surface_x.wedge(&surface_y); // grade 2

    // find normals: grades with odd difference from surface grade 2
    let normal_vector = surface_y; // grade 1 (diff=1, odd=orthogonal)
    let normal_trivector = surface_y
        .rotate(Angle::new(1.0, 2.0))
        .rotate(Angle::new(1.0, 2.0)); // grade 3 (diff=1, odd=orthogonal)
    let parallel_scalar = surface_plane.dual(); // grade 0 (diff=2, even=parallel)

    // test orthogonality via dot products
    let dot_vector = surface_plane.dot(&normal_vector);
    let dot_trivector = surface_plane.dot(&normal_trivector);
    let dot_parallel = surface_plane.dot(&parallel_scalar);

    assert!(
        dot_vector.mag.abs() < 1e-10,
        "grade 1 normal orthogonal to grade 2 surface"
    );
    assert!(
        dot_trivector.mag.abs() < 1e-10,
        "grade 3 normal orthogonal to grade 2 surface"
    );
    assert!(
        dot_parallel.mag.abs() > 1e-10,
        "grade 0 parallel to grade 2 surface (even diff)"
    );

    // geonum ghosts pseudoscalar I₃ multiplication
    // orthogonality emerges from grade parity, no pseudoscalar needed

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
    // traditional GA rotation: R*v*R† requires 3 geometric products × 64 operations = 192 operations
    // geonum rotation: angle addition = 1 operation
    // the 192x difference comes from eliminating the pseudoscalar

    // test rotation equivalence
    let point = Geonum::new_from_cartesian(3.0, 4.0);
    let rotation = Angle::new(1.0, 2.0); // π/2
    let rotated = point.rotate(rotation);

    let x = rotated.mag * rotated.angle.grade_angle().cos();
    let y = rotated.mag * rotated.angle.grade_angle().sin();
    assert!((x - (-4.0)).abs() < EPSILON);
    assert!((y - 3.0).abs() < EPSILON);

    // traditional GA needs pseudoscalar to define rotation planes through basis blade products
    // this forces dimension-specific formulas and full multiplication tables
    // geonum recognizes rotation IS angle addition, not basis blade reconstruction

    // test rotation composition
    let r1 = Angle::new(1.0, 4.0); // π/4
    let r2 = Angle::new(1.0, 6.0); // π/6
    let r3 = Angle::new(1.0, 3.0); // π/3

    // traditional: R1*R2*R3 through geometric products
    // geonum: r1 + r2 + r3
    let composed = r1 + r2 + r3;
    assert_eq!(composed, Angle::new(3.0, 4.0)); // 3π/4

    let final_point = point.rotate(composed);
    let fx = final_point.mag * final_point.angle.grade_angle().cos();
    let fy = final_point.mag * final_point.angle.grade_angle().sin();
    assert!((fx - (-4.949)).abs() < 0.01);
    assert!((fy - (-0.707)).abs() < 0.01);

    // eliminating pseudoscalar reveals rotation as angle addition
    // 192 operations → 1 operation
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
    assert_eq!(dual_million.mag, million_dim_vector.mag);
    assert_eq!(dual_billion.mag, billion_dim_bivector.mag);

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
    assert_eq!(positive_five.angle.rem(), 0.0); // absolute position: 0 radians
    assert_eq!(negative_five.angle.blade(), 2); // blade 2 means 2 × π/2 rotations
    assert_eq!(negative_five.angle.rem(), 0.0); // but value within blade is 0

    // multiplication adds absolute angles - not relative "signs"
    let product = positive_five * negative_five;
    assert_eq!(product.angle.blade(), 2); // 0 + 2 = 2 blade rotations
    assert_eq!(product.mag, 25.0);

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
    assert!((chain.angle.grade_angle() - (total_angle % (2.0 * PI))).abs() < EPSILON);

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
    assert_eq!(zero_at_0.mag, 0.0);
    assert_eq!(zero_at_pi.mag, 0.0);
    assert_eq!(zero_at_pi_2.mag, 0.0);

    // multiplying by zero preserves the zero's angle (absolute position matters)
    let five = Geonum::new(5.0, 0.0, 1.0);
    let result1 = five * zero_at_0; // 5 * 0 with 0 at angle 0
    let result2 = five * zero_at_pi; // 5 * 0 with 0 at angle π

    assert_eq!(result1.mag, 0.0);
    assert_eq!(result2.mag, 0.0);
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
    assert!((wedge_12.mag - wedge_21.mag).abs() < EPSILON);

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
    assert!((meet_ab.mag - meet_ba.mag).abs() < EPSILON);
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
    assert_eq!(product.mag, 1.0); // unit length ✓

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

    assert_eq!(product_b.mag, 1.0);
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
    assert_eq!(scalar_inv_product.mag, 1.0);
    assert_eq!(vector_inv_product.mag, 1.0);
    assert_eq!(bivector_inv_product.mag, 1.0);
    assert_eq!(trivector_inv_product.mag, 1.0);

    // but different starting grades produce different blade counts
    // inv() adds π, multiplication adds angles: 2θ + π
    assert_eq!(scalar_inv_product.angle.blade(), 2); // scalar: 0→π→2π = blade 2
    assert_eq!(vector_inv_product.angle.blade(), 4); // vector: π/2→π→2π = blade 4
    assert_eq!(bivector_inv_product.angle.blade(), 6); // bivector: π→2π→3π = blade 6
    assert_eq!(trivector_inv_product.angle.blade(), 8); // trivector: 3π/2→3π→4π = blade 8

    // when comparing angles after transformations, must account for blade changes
    // operations like circular inversion transform blade structure similarly
    // this is why angle preservation tests can fail if only checking angle.rem()
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
    assert_eq!(reset.mag, position.mag, "length unchanged");

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
        traditional.angle.rem().abs() < 1e-10,
        "traditional value at 3π/2 is 0"
    );

    // also verify the identity has unit length
    let identity = original * original.inv();
    assert!(
        (identity.mag - 1.0).abs() < 1e-10,
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

#[test]
fn it_handles_mixed_grade_operations_naturally() {
    // traditional GA: restricts operations to "like grades" or requires complex rules
    // scalar * scalar = scalar, vector * vector = scalar + bivector, etc.
    // mixed grade operations need special handling and decomposition

    // geonum: blade arithmetic works for ANY grade combination
    let scalar = Geonum::new(2.0, 0.0, 1.0); // blade 0 (grade 0)
    let vector = Geonum::new(3.0, 1.0, 2.0); // blade 1 (grade 1)
    let bivector = Geonum::new(1.5, 1.0, 1.0); // blade 2 (grade 2)
    let trivector = Geonum::new(4.0, 3.0, 2.0); // blade 3 (grade 3)

    // mixed grade products: blade counts just add
    let scalar_vector = scalar * vector; // 0+1=1 (vector)
    let vector_bivector = vector * bivector; // 1+2=3 (trivector)
    let bivector_trivector = bivector * trivector; // 2+3=5 (grade 1: 5%4=1)
    let scalar_trivector = scalar * trivector; // 0+3=3 (trivector)

    // verify blade arithmetic works regardless of starting grades
    assert_eq!(scalar_vector.angle.blade(), 1);
    assert_eq!(vector_bivector.angle.blade(), 3);
    assert_eq!(bivector_trivector.angle.blade(), 5);
    assert_eq!(scalar_trivector.angle.blade(), 3);

    // verify grades cycle correctly (blade % 4)
    assert_eq!(scalar_vector.angle.grade(), 1); // blade 1 → grade 1
    assert_eq!(vector_bivector.angle.grade(), 3); // blade 3 → grade 3
    assert_eq!(bivector_trivector.angle.grade(), 1); // blade 5 → grade 1
    assert_eq!(scalar_trivector.angle.grade(), 3); // blade 3 → grade 3

    // traditional GA: each combination needs special rules and storage
    // geonum: universal blade addition works for all grade combinations
    // no restrictions, no special cases, no decomposition complexity
}

#[test]
fn it_proves_rotational_quadrature_expresses_quadratic_forms() {
    // prove that geonum's rotational projections express the same quadratic relationships
    // as traditional coordinate decomposition, just through angle arithmetic instead of squares

    // step 1: build a geonum using rotation (natural geonum way)
    let base_scalar = Geonum::scalar(5.0);
    let rotated_vector = base_scalar.rotate(Angle::new(1.0, 4.0)); // π/4 rotation
    let geonum = rotated_vector.scale(2.0); // 10 units at π/4

    println!(
        "Geonum: length={}, angle={:.3} rad",
        geonum.mag,
        geonum.angle.grade_angle()
    );

    // step 2: get projections through geonum's rotational interface
    let proj_x = geonum.project_to_dimension(0); // blade 0 projection
    let proj_y = geonum.project_to_dimension(1); // blade 1 projection

    println!("Rotational projections: x={:.3}, y={:.3}", proj_x, proj_y);

    // step 3: compute the same projections using traditional trigonometry
    let angle = geonum.angle.grade_angle();
    let trig_x = geonum.mag * angle.cos(); // traditional x = r*cos(θ)
    let trig_y = geonum.mag * angle.sin(); // traditional y = r*sin(θ)

    println!(
        "Trigonometric projections: x={:.3}, y={:.3}",
        trig_x, trig_y
    );

    // step 4: prove rotational projections equal trigonometric projections
    assert!(
        (proj_x - trig_x).abs() < EPSILON,
        "rotational x projection equals trigonometric: {:.6} vs {:.6}",
        proj_x,
        trig_x
    );

    assert!(
        (proj_y - trig_y).abs() < EPSILON,
        "rotational y projection equals trigonometric: {:.6} vs {:.6}",
        proj_y,
        trig_y
    );

    // step 5: prove both express the same quadratic form (pythagorean theorem)
    let rotational_magnitude_squared = proj_x.powi(2) + proj_y.powi(2);
    let trigonometric_magnitude_squared = trig_x.powi(2) + trig_y.powi(2);
    let original_magnitude_squared = geonum.mag.powi(2);

    println!("Magnitude² comparisons:");
    println!("  Original: {:.6}", original_magnitude_squared);
    println!(
        "  Rotational projections: {:.6}",
        rotational_magnitude_squared
    );
    println!(
        "  Trigonometric projections: {:.6}",
        trigonometric_magnitude_squared
    );

    // prove all three quadratic forms are equivalent
    assert!(
        (rotational_magnitude_squared - original_magnitude_squared).abs() < EPSILON,
        "rotational projections preserve quadratic form: {:.6} vs {:.6}",
        rotational_magnitude_squared,
        original_magnitude_squared
    );

    assert!(
        (trigonometric_magnitude_squared - original_magnitude_squared).abs() < EPSILON,
        "trigonometric projections preserve quadratic form: {:.6} vs {:.6}",
        trigonometric_magnitude_squared,
        original_magnitude_squared
    );

    assert!(
        (rotational_magnitude_squared - trigonometric_magnitude_squared).abs() < EPSILON,
        "rotational and trigonometric quadratic forms are identical: {:.6} vs {:.6}",
        rotational_magnitude_squared,
        trigonometric_magnitude_squared
    );

    // step 6: demonstrate quadrature to quadratic identity
    println!("\n--- Quadrature → Quadratic Identity Chain ---");

    // quadrature: sin(θ+π/2) = cos(θ)
    let theta = angle;
    let sin_theta_plus_pi_2 = (theta + PI / 2.0).sin();
    let cos_theta = theta.cos();

    println!(
        "quadrature: sin({:.3}+π/2) = {:.6}, cos({:.3}) = {:.6}",
        theta, sin_theta_plus_pi_2, theta, cos_theta
    );

    assert!(
        (sin_theta_plus_pi_2 - cos_theta).abs() < EPSILON,
        "quadrature relationship sin(θ+π/2) = cos(θ): {:.6} vs {:.6}",
        sin_theta_plus_pi_2,
        cos_theta
    );

    // the quadrature generates the quadratic identity through orthogonal projections
    // when we project onto orthogonal axes (0 and π/2 apart):
    let proj_0_axis = theta.cos(); // projection onto 0 axis
    let proj_pi_2_axis = theta.sin(); // projection onto π/2 axis (= cos(θ-π/2) = sin(θ))

    println!(
        "Orthogonal projections: cos({:.3})={:.6}, sin({:.3})={:.6}",
        theta, proj_0_axis, theta, proj_pi_2_axis
    );

    // the quadratic identity emerges when we square both orthogonal projections
    let cos_squared = proj_0_axis.powi(2);
    let sin_squared = proj_pi_2_axis.powi(2);
    let quadratic_identity = cos_squared + sin_squared;

    println!(
        "Quadratic identity from quadrature: cos²({:.3}) + sin²({:.3}) = {:.6}",
        theta, theta, quadratic_identity
    );

    assert!(
        (quadratic_identity - 1.0).abs() < EPSILON,
        "quadrature generates quadratic identity sin²+cos²=1: {:.6}",
        quadratic_identity
    );

    // step 7: connect quadratic identity to projection equivalence
    // the geonum rotational projections use the same orthogonal relationships
    println!("\n--- Projection Equivalence from Quadrature ---");

    // geonum's projections are scaled by the same orthogonal basis from quadrature
    let geonum_x_from_quadrature = geonum.mag * proj_0_axis; // length × cos(θ)
    let geonum_y_from_quadrature = geonum.mag * proj_pi_2_axis; // length × sin(θ)

    println!(
        "Projections from quadrature: x={:.6}, y={:.6}",
        geonum_x_from_quadrature, geonum_y_from_quadrature
    );

    // prove these match geonum's rotational projections
    assert!(
        (proj_x - geonum_x_from_quadrature).abs() < EPSILON,
        "geonum x projection equals quadrature projection: {:.6} vs {:.6}",
        proj_x,
        geonum_x_from_quadrature
    );

    assert!(
        (proj_y - geonum_y_from_quadrature).abs() < EPSILON,
        "geonum y projection equals quadrature projection: {:.6} vs {:.6}",
        proj_y,
        geonum_y_from_quadrature
    );

    // step 8: prove the complete causal chain
    let quadratic_from_quadrature = geonum.mag.powi(2) * quadratic_identity;

    assert!(
        (quadratic_from_quadrature - original_magnitude_squared).abs() < EPSILON,
        "quadratic form emerges from rotational quadrature: {:.6} vs {:.6}",
        quadratic_from_quadrature,
        original_magnitude_squared
    );

    println!("\n✓ Complete causal chain proven:");
    println!("  quadrature: t");
    println!("  → Orthogonal projections: cos(θ), sin(θ)");
    println!("  → Quadratic identity: sin²+cos²=1");
    println!("  → Projection equivalence: geonum rotational = trigonometric");
    println!("  → Quadratic forms emerge naturally from angle arithmetic");
}
