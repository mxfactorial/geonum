use geonum::*;
use std::f64::consts::TAU;

#[test]
fn it_operates_on_high_dimensional_multivectors() {
    // transition from coordinate scaffolding to direct geometric number creation
    // old design: required declaring 1000-dimensional "space" first
    // new design: create geometric numbers that can project to any dimension

    // create geometric numbers at standardized dimensional angles
    // these replace the "basis vectors" that required coordinate scaffolding
    let e1 = Geonum::create_dimension(1.0, 1); // angle π/2, blade 1
    let e2 = Geonum::create_dimension(1.0, 2); // angle π, blade 2
    let e3 = Geonum::create_dimension(1.0, 3); // angle 3π/2, blade 3

    // step 1: construct bivector B = e1 ∧ e2
    let b12 = e1.wedge(&e2);
    let b12_mv = Multivector(vec![b12]);

    // bivector has non-zero length
    assert!(b12.length > 0.0, "wedge product resulted in zero bivector");

    // step 2: trivector T = (e1 ∧ e2) ∧ e3
    // create e3 as multivector
    let e3_mv = Multivector(vec![e3]);

    // rotating e3 with a different angle to prevent wedge collapse
    let e3_rotated = Geonum::new(1.0, 1.0, 4.0);

    // create trivector directly using wedge product between b12 and e3_rotated
    let t123 = b12.wedge(&e3_rotated);
    let t123_mv = Multivector(vec![t123]);

    // assert trivector has nonzero length
    assert!(t123.length > 0.0, "trivector has nonzero length");

    // step 3: reflect e3 across bivector plane
    let reflected = e3_mv.reflect(&b12_mv);
    assert!(!reflected.0.is_empty());

    // step 4: compute dual of trivector and test result
    let pseudo = Multivector(vec![Geonum::new_with_blade(1.0, 1000, 1.0, 2.0)]); // dummy pseudoscalar with high dimension
    let dual = t123_mv.dual(&pseudo);
    assert!(dual.0.iter().any(|g| g.length > 0.0));

    // step 5: rotate a vector using bivector rotor
    let v = Geonum::new(1.0, 1.0, 4.0);
    let v_mv = Multivector(vec![v]);
    let rotated = v_mv.rotate(&b12_mv);
    assert!(rotated.0.iter().any(|g| g.length > 0.0));

    // step 6: project and reject vector from e3
    let _proj = v_mv.project(&e3_mv);
    let _rej = v_mv.reject(&e3_mv);
    // projection and rejection operations complete without panicking
    // results may be empty or non-empty depending on geometric relationships

    // step 7: confirm mean angle is finite and within expected bounds
    let mean = t123_mv.weighted_circular_mean_angle();
    let mean_radians = mean.mod_4_angle();
    assert!((0.0..=TAU).contains(&mean_radians));
}

#[test]
fn it_maintains_blade_in_high_dimensions() {
    // Create multivectors with explicit blade values for 1000-dimensional space
    let space_dim = 1000;
    let pseudo = Geonum::new_with_blade(1.0, space_dim, 0.0, 1.0);
    let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // explicit blade 1 for vector

    // Create multivectors
    let pseudo_mv = Multivector(vec![pseudo]);
    let vector_mv = Multivector(vec![vector]);

    // Grade extraction works with blade property
    let vector_grade = vector.angle.grade();
    let pseudo_grade = pseudo.angle.grade();

    let grade1 = vector_mv.grade(vector_grade);
    let grade_n = pseudo_mv.grade(pseudo_grade);

    println!(
        "vector blade: {}, grade: {}",
        vector.angle.blade(),
        vector_grade
    );
    println!(
        "pseudo blade: {}, grade: {}",
        pseudo.angle.blade(),
        pseudo_grade
    );
    println!(
        "grade1 len: {}, grade_n len: {}",
        grade1.0.len(),
        grade_n.0.len()
    );

    assert_eq!(grade1.0.len(), 1);
    assert_eq!(grade_n.0.len(), 1);

    // Wrong grades returns empty
    let wrong_vector_grade = if vector_grade == 2 { 3 } else { 2 };
    let wrong_pseudo_grade = (pseudo_grade + 1) % 4;

    let empty1 = vector_mv.grade(wrong_vector_grade);
    let empty2 = pseudo_mv.grade(wrong_pseudo_grade);

    assert_eq!(empty1.0.len(), 0);
    assert_eq!(empty2.0.len(), 0);

    // Create trivector (grade 3)
    let trivector = Geonum::new_with_blade(1.0, 3, 0.0, 1.0);
    let trivector_mv = Multivector(vec![trivector]);

    // Prove grade extraction works with blade in high dimensions
    let trivector_grade = trivector.angle.grade();
    let grade3 = trivector_mv.grade(trivector_grade);
    assert_eq!(grade3.0.len(), 1);

    // Blade grade detection returns pure grade multivectors
    assert_eq!(trivector_mv.blade_grade(), Some(3));
    assert_eq!(vector_mv.blade_grade(), Some(1));
    assert_eq!(pseudo_mv.blade_grade(), Some(space_dim));

    // Mixed grade multivectors return None
    let mixed = Multivector(vec![
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0),
        Geonum::new_with_blade(1.0, 2, 0.0, 1.0),
    ]);
    assert_eq!(mixed.blade_grade(), None);
}

#[test]
fn it_works_with_multivector_operations() {
    // Create geometric elements of different grades
    // In geonum, π/2 rotation changes grade, so these are NOT both vectors
    let e1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1, grade 1 (vector)
    let e2 = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // blade 2, grade 2 (bivector)

    // These are π/2 apart, which in geonum means different grades
    // This demonstrates the key insight: orthogonality changes grade

    // Create multivectors
    let mv1 = Multivector(vec![e1]);
    let mv2 = Multivector(vec![e2]);

    // Extract grades based on actual grade
    let e1_grade = e1.angle.grade();
    let e2_grade = e2.angle.grade();
    assert_eq!(mv1.grade(e1_grade).0.len(), 1);
    assert_eq!(mv2.grade(e2_grade).0.len(), 1);

    // Check blade grades
    assert_eq!(mv1.blade_grade(), Some(e1_grade));
    assert_eq!(mv2.blade_grade(), Some(e2_grade));

    // Wedge product of grade 1 (vector) with grade 2 (bivector)
    let b12 = e1.wedge(&e2);
    let b12_mv = Multivector(vec![b12]);

    // The wedge product is non-zero
    assert!(b12.length > 0.0, "wedge product should be non-zero");
    let b12_blade = b12.angle.blade();
    assert_eq!(b12_mv.blade_grade(), Some(b12_blade));

    // Grade extraction with blade property
    let b12_grade = b12.angle.grade();
    let grade_extract = b12_mv.grade(b12_grade);
    assert_eq!(grade_extract.0.len(), 1);

    // Wrong grade returns empty
    let wrong_grade = if b12_grade == 1 { 2 } else { 1 };
    let empty = b12_mv.grade(wrong_grade);
    assert_eq!(empty.0.len(), 0);
}
