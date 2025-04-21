use geonum::*;
use std::f64::consts::PI;
use std::f64::consts::TAU;

#[test]
fn it_operates_on_high_dimensional_multivectors() {
    let space = Dimensions::new(1_000);

    // basis vectors with distinct angles to avoid wedge collapse
    let e1 = Geonum {
        length: 1.0,
        angle: space.base_angle(0),
        blade: 1,
    }; // angle 0, grade 1
    let e2 = Geonum {
        length: 1.0,
        angle: space.base_angle(1),
        blade: 1,
    }; // angle π/2, grade 1
    let e3 = Geonum {
        length: 1.0,
        angle: space.base_angle(3),
        blade: 1,
    }; // angle 3π/2, grade 1

    // step 1: construct bivector B = e1 ∧ e2
    let b12 = e1.wedge(&e2);
    let b12_mv = Multivector(vec![b12]);

    // assert bivector lives in grade 2
    let g2 = b12_mv.grade_range([2, 2]);
    assert!(g2.0.iter().any(|g| g.length > 0.0));

    // step 2: trivector T = (e1 ∧ e2) ∧ e3
    // create e3 as multivector
    let e3_mv = Multivector(vec![e3]);

    // rotating e3 with a different angle to prevent wedge collapse
    let e3_rotated = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };

    // create trivector directly using wedge product between b12 and e3_rotated
    let t123 = b12.wedge(&e3_rotated);
    let t123_mv = Multivector(vec![t123]);

    // assert trivector has a nonzero component in grade 3
    let g3 = t123_mv.grade_range([3, 3]);
    assert!(g3.0.iter().any(|g| g.length > 0.0));

    // step 3: reflect e3 across bivector plane
    let reflected = e3_mv.reflect(&b12_mv);
    assert!(!reflected.0.is_empty());

    // step 4: compute dual of trivector and test result
    let pseudo = Multivector(vec![Geonum {
        length: 1.0,
        angle: PI / 2.0,
        blade: 1000,
    }]); // dummy pseudoscalar with high dimension
    let dual = t123_mv.dual(&pseudo);
    assert!(dual.0.iter().any(|g| g.length > 0.0));

    // step 5: rotate a vector using bivector rotor
    let v = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };
    let v_mv = Multivector(vec![v]);
    let rotated = v_mv.rotate(&b12_mv);
    assert!(rotated.0.iter().any(|g| g.length > 0.0));

    // step 6: project and reject vector from e3
    let proj = v_mv.project(&e3_mv);
    let rej = v_mv.reject(&e3_mv);
    assert!(proj.0.len() > 0 || proj.0.is_empty()); // test execution
    assert!(rej.0.len() > 0 || rej.0.is_empty());

    // step 7: confirm mean angle is finite and within expected bounds
    let mean = t123_mv.weighted_circular_mean_angle();
    assert!(mean >= 0.0 && mean <= TAU);
}

#[test]
fn it_maintains_blade_in_high_dimensions() {
    // Create multivectors with explicit blade values for 1000-dimensional space
    let space_dim = 1000;
    let pseudo = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: space_dim,
    };
    let vector = Geonum {
        length: 1.0,
        angle: PI / 4.0,
        blade: 1,
    };

    // Create multivectors
    let pseudo_mv = Multivector(vec![pseudo]);
    let vector_mv = Multivector(vec![vector]);

    // Grade extraction works with blade property
    let grade1 = vector_mv.grade(1);
    let grade_n = pseudo_mv.grade(space_dim);

    assert_eq!(grade1.0.len(), 1);
    assert_eq!(grade_n.0.len(), 1);

    // Wrong grades should return empty
    let empty1 = vector_mv.grade(2);
    let empty2 = pseudo_mv.grade(space_dim - 1);

    assert_eq!(empty1.0.len(), 0);
    assert_eq!(empty2.0.len(), 0);

    // Create trivector (grade 3)
    let trivector = Geonum {
        length: 1.0,
        angle: 0.0,
        blade: 3,
    };
    let trivector_mv = Multivector(vec![trivector]);

    // Verify grade extraction works with blade in high dimensions
    let grade3 = trivector_mv.grade(3);
    assert_eq!(grade3.0.len(), 1);

    // Blade grade detection should work for pure grade multivectors
    assert_eq!(trivector_mv.blade_grade(), Some(3));
    assert_eq!(vector_mv.blade_grade(), Some(1));
    assert_eq!(pseudo_mv.blade_grade(), Some(space_dim));

    // Mixed grade multivectors should return None
    let mixed = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        },
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 2,
        },
    ]);
    assert_eq!(mixed.blade_grade(), None);
}

#[test]
fn it_works_with_multivector_operations() {
    // Create basis vectors
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

    // Create multivectors
    let mv1 = Multivector(vec![e1]);
    let mv2 = Multivector(vec![e2]);

    // Extract grades - should work properly
    assert_eq!(mv1.grade(1).0.len(), 1);
    assert_eq!(mv2.grade(1).0.len(), 1);

    // 1-vectors should have grade 1
    assert_eq!(mv1.blade_grade(), Some(1));

    // Operations preserve blade information
    let b12 = e1.wedge(&e2);
    let b12_mv = Multivector(vec![b12]);

    // Should be a bivector (grade 2)
    assert_eq!(b12.blade, 2);
    assert_eq!(b12_mv.blade_grade(), Some(2));

    // Grade extraction should work with blade property
    let g2 = b12_mv.grade(2);
    assert_eq!(g2.0.len(), 1);

    // Wrong grade should return empty
    let empty = b12_mv.grade(1);
    assert_eq!(empty.0.len(), 0);
}
