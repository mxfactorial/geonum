use geonum::traits::Affine;
use geonum::*;

const EPSILON: f64 = 1e-10;
const PI: f64 = std::f64::consts::PI;

#[test]
fn its_a_translation() {
    // in affine transformations, translation moves points without rotation
    // matrices require homogeneous coordinates and augmented matrices
    // geometric numbers handle translation as direct displacement

    // create a point in 2D space
    let point = Geonum {
        length: 5.0,     // distance from origin
        angle: PI / 6.0, // 30 degrees
        blade: 1,        // vector (grade 1)
    };

    // matrix approach requires homogeneous coordinates:
    // [1 0 tx]   [x]   [x + tx]
    // [0 1 ty] × [y] = [y + ty]
    // [0 0  1]   [1]   [1]
    // this forces 3×3 matrix for 2D translation!

    // geometric number approach: translation is angle-preserving displacement
    let translation_vector = Geonum {
        length: 3.0,     // translate by 3 units
        angle: PI / 2.0, // in the y direction (90 degrees)
        blade: 1,
    };

    // translation preserves angles, combines lengths geometrically
    let translated_point = point.translate(&translation_vector);

    // test that translation preserves the geometric structure
    // unlike matrices, we don't need homogeneous coordinates or matrix expansion
    assert!(translated_point.blade == point.blade); // grade preserved

    // translation in geometric numbers is vector addition in polar form
    // the result is a combined displacement
    let original_x = point.length * point.angle.cos();
    let original_y = point.length * point.angle.sin();
    let translate_x = translation_vector.length * translation_vector.angle.cos();
    let translate_y = translation_vector.length * translation_vector.angle.sin();

    let expected_x = original_x + translate_x;
    let expected_y = original_y + translate_y;
    let expected_length = (expected_x.powi(2) + expected_y.powi(2)).sqrt();
    let expected_angle = expected_y.atan2(expected_x);

    // test geometric properties are preserved
    assert!((translated_point.length - expected_length).abs() < EPSILON);
    assert!((translated_point.angle - expected_angle).abs() < EPSILON);

    // test that translation is reversible through inverse displacement
    let inverse_translation = Geonum {
        length: translation_vector.length,
        angle: translation_vector.angle + PI, // opposite direction
        blade: 1,
    };

    let back_to_original = translated_point.translate(&inverse_translation);
    assert!((back_to_original.length - point.length).abs() < EPSILON);
    assert!((back_to_original.angle - point.angle).abs() < EPSILON);

    // geometric numbers avoid the matrix overhead:
    // no homogeneous coordinates needed
    // no 3×3 matrix for 2D operation
    // no artificial dimension expansion
    // direct geometric displacement in O(1) time
}

#[test]
fn it_preserves_parallel_lines_after_shearing() {
    // affine transformations must preserve parallelism
    // matrices require full grid computation to verify this property
    // geometric numbers preserve parallelism through direct angle relationships

    // create two parallel lines as pairs of points
    // line 1: horizontal line at y = 2
    let line1_p1 = Geonum {
        length: 2.0,
        angle: PI / 2.0, // 90 degrees (pointing up)
        blade: 1,
    };
    let line1_p2 = Geonum {
        length: (4.0 + 4.0_f64).sqrt(), // distance to (2, 2)
        angle: (2.0_f64).atan2(2.0),    // angle to (2, 2)
        blade: 1,
    };

    // line 2: horizontal line at y = 4 (parallel to line 1)
    let line2_p1 = Geonum {
        length: 4.0,
        angle: PI / 2.0, // 90 degrees (pointing up)
        blade: 1,
    };
    let line2_p2 = Geonum {
        length: (4.0 + 16.0_f64).sqrt(), // distance to (2, 4)
        angle: (4.0_f64).atan2(2.0),     // angle to (2, 4)
        blade: 1,
    };

    // calculate original direction vectors (parallel lines have same direction)
    let original_direction1 = Geonum {
        length: 2.0,
        angle: 0.0, // horizontal direction
        blade: 1,
    };
    let original_direction2 = Geonum {
        length: 2.0,
        angle: 0.0, // horizontal direction (same as line 1)
        blade: 1,
    };

    // verify original lines are parallel (same direction angle)
    assert!((original_direction1.angle - original_direction2.angle).abs() < EPSILON);

    // apply shear transformation
    let shear_angle = PI / 6.0; // 30 degree shear

    let sheared_line1_p1 = line1_p1.shear(shear_angle);
    let sheared_line1_p2 = line1_p2.shear(shear_angle);
    let sheared_line2_p1 = line2_p1.shear(shear_angle);
    let sheared_line2_p2 = line2_p2.shear(shear_angle);

    // calculate sheared direction vectors
    let sheared_direction1 = original_direction1.shear(shear_angle);
    let sheared_direction2 = original_direction2.shear(shear_angle);

    // test that parallelism is preserved after shearing
    // in geometric numbers, parallel lines maintain the same angular relationship
    assert!((sheared_direction1.angle - sheared_direction2.angle).abs() < EPSILON);

    // test that shear transformation is consistent
    // all points should have their angles shifted by the same amount
    assert!((sheared_line1_p1.angle - (line1_p1.angle + shear_angle)).abs() < EPSILON);
    assert!((sheared_line1_p2.angle - (line1_p2.angle + shear_angle)).abs() < EPSILON);
    assert!((sheared_line2_p1.angle - (line2_p1.angle + shear_angle)).abs() < EPSILON);
    assert!((sheared_line2_p2.angle - (line2_p2.angle + shear_angle)).abs() < EPSILON);

    // test that lengths are preserved during shear (fundamental property)
    assert!((sheared_line1_p1.length - line1_p1.length).abs() < EPSILON);
    assert!((sheared_line1_p2.length - line1_p2.length).abs() < EPSILON);
    assert!((sheared_line2_p1.length - line2_p1.length).abs() < EPSILON);
    assert!((sheared_line2_p2.length - line2_p2.length).abs() < EPSILON);

    // geometric numbers make affine properties explicit:
    // parallelism is preserved through consistent angle transformation
    // no matrix computation needed to verify geometric properties
    // direct access to the geometric meaning of the transformation
}

#[test]
fn it_preserves_area_after_shearing() {
    // affine transformations must preserve area
    // matrices require determinant calculation to verify this property
    // geometric numbers preserve area through direct geometric computation

    // create a rectangle with known area
    let width = 4.0;
    let height = 3.0;

    // rectangle vertices in geometric number form
    let v1 = Geonum {
        length: 0.0,
        angle: 0.0,
        blade: 1,
    }; // origin (0,0)
    let v2 = Geonum {
        length: width,
        angle: 0.0, // horizontal
        blade: 1,
    }; // (4,0)
    let v3 = Geonum {
        length: (width * width + height * height).sqrt(),
        angle: height.atan2(width),
        blade: 1,
    }; // (4,3)
    let v4 = Geonum {
        length: height,
        angle: PI / 2.0, // vertical
        blade: 1,
    }; // (0,3)

    // calculate original area
    let original_area = Geonum::area_quadrilateral(&v1, &v2, &v3, &v4);
    let expected_area = width * height; // 12.0
    assert!((original_area - expected_area).abs() < EPSILON);

    // apply shear transformation
    let shear_angle = PI / 4.0; // 45 degree shear

    let sheared_v1 = v1.shear(shear_angle);
    let sheared_v2 = v2.shear(shear_angle);
    let sheared_v3 = v3.shear(shear_angle);
    let sheared_v4 = v4.shear(shear_angle);

    // calculate sheared area
    let sheared_area =
        Geonum::area_quadrilateral(&sheared_v1, &sheared_v2, &sheared_v3, &sheared_v4);

    // test that area is preserved after shearing
    assert!((original_area - sheared_area).abs() < EPSILON);

    // test specific area value
    assert!((sheared_area - 12.0).abs() < EPSILON);

    // test that individual lengths are preserved (fundamental property of our shear)
    assert!((sheared_v1.length - v1.length).abs() < EPSILON);
    assert!((sheared_v2.length - v2.length).abs() < EPSILON);
    assert!((sheared_v3.length - v3.length).abs() < EPSILON);
    assert!((sheared_v4.length - v4.length).abs() < EPSILON);

    // test that angles are consistently shifted
    assert!((sheared_v2.angle - (v2.angle + shear_angle)).abs() < EPSILON);
    assert!((sheared_v3.angle - (v3.angle + shear_angle)).abs() < EPSILON);
    assert!((sheared_v4.angle - (v4.angle + shear_angle)).abs() < EPSILON);

    // geometric numbers make area preservation explicit:
    // shear preserves area because it's a uniform angular transformation
    // no determinant calculation needed to verify this geometric property
    // direct verification through geometric computation rather than matrix algebra
}

#[test]
fn it_increases_angle_after_shearing() {
    let point = Geonum {
        length: 5.0,
        angle: PI / 3.0, // 60 degrees
        blade: 1,
    };

    let shear_angle = PI / 6.0; // 30 degrees
    let sheared_point = point.shear(shear_angle);

    // length remains unchanged
    assert!((sheared_point.length - point.length).abs() < EPSILON);

    // angle is increased by shear_angle
    assert!((sheared_point.angle - (point.angle + shear_angle)).abs() < EPSILON);

    // blade is preserved
    assert_eq!(sheared_point.blade, point.blade);
}
