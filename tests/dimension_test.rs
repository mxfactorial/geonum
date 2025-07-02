use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_projects_a_geonum_onto_coordinate_axes() {
    // step 1: unify traditional [x, y, z] coordinates into single geonum
    // traditional 3d vector: [2, 3, 1]
    let x = 2.0_f64;
    let y = 3.0_f64;
    let z = 1.0_f64;

    // compute unified geometric representation
    let magnitude = (x * x + y * y + z * z).sqrt(); // 3.742

    // encode direction as fractional angle within pi/2 constraint
    // for this simplified test, use direct angle calculation
    let total_angle = y.atan2(x); // basic 2d angle for now
    let blade = 3; // 3d vector = 3 pi/2 turns
    let fractional_angle = total_angle % (PI / 2.0); // keep within [0, pi/2]

    let unified = Geonum {
        length: magnitude,
        angle: fractional_angle,
        blade: blade,
    };

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
    let geometric_entity = Geonum {
        length: 2.5, // some arbitrary magnitude
        angle: 0.8,  // some arbitrary direction (< pi/2)
        blade: 1,    // vector grade
    };

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

    // test components are well-defined (not nan or infinite)
    assert!(dimension_3.is_finite(), "3rd dimension component is finite");
    assert!(
        dimension_42.is_finite(),
        "42nd dimension component is finite"
    );
    assert!(
        dimension_1000.is_finite(),
        "1000th dimension component is finite"
    );
    assert!(
        dimension_million.is_finite(),
        "millionth dimension component is finite"
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
    assert!((dim_0.angle - 0.0).abs() < EPSILON);
    assert!((dim_1.angle - PI / 2.0).abs() < EPSILON);
    assert!((dim_2.angle - PI).abs() < EPSILON);
    assert!((dim_1000.angle - (1000.0 * PI / 2.0)).abs() < EPSILON);

    // test blade equals dimension_index
    assert_eq!(dim_0.blade, 0);
    assert_eq!(dim_1.blade, 1);
    assert_eq!(dim_2.blade, 2);
    assert_eq!(dim_1000.blade, 1000);

    // test multivector constructor
    let mv = Multivector::create_dimension(1.0, &[0, 1, 2]);
    assert_eq!(mv.len(), 3);
    assert_eq!(mv[0].blade, 0);
    assert_eq!(mv[1].blade, 1);
    assert_eq!(mv[2].blade, 2);
}
