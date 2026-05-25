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
    assert!(constructed_dim_0.angle.near_rad(0.0));
    assert!(constructed_dim_1.angle.near_rad(PI / 2.0));
    assert!(constructed_dim_2.angle.near_rad(PI));
    assert!(constructed_dim_3.angle.near_rad(3.0 * PI / 2.0));

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
    assert!(dot.near_mag(0.0)); // zero

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
    assert!(chain.angle.near_rad(total_angle % (2.0 * PI)));

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
    assert!(wedge_12.near_mag(wedge_21.mag));

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
    assert!(meet_ab.near_mag(meet_ba.mag));
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
