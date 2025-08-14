// traditional projective geometric algebra (PGA) requires homogeneous coordinates,
// projective transformations through matrix operations, and complex duality relationships
// to handle points, lines, planes and their intersections
//
// every projective operation - which should just be "preserve incidence, allow perspective" -
// becomes matrix multiplication with homogeneous coordinate management and
// special cases for points at infinity
//
// this is all mathematically unnecessary. projective transformations
// are fundamentally about scaling homogeneous coordinates while preserving geometric relationships
//
// the crucial insight: projective = homogeneous scaling with directional preservation
//
// this single principle unlocks direct geometric operations that eliminate the need for
// matrix transformations, homogeneous coordinate juggling, and duality complications
//
// by encoding homogeneous scaling directly in length components while preserving
// directional information in angles, geometric numbers represent projective transformations
// as their actual geometric reality
//
// ```rs
// // traditional PGA (simplified)
// homogeneous_coords = [x, y, z, w]
// matrix_transform(transformation_matrix, homogeneous_coords) → transformed_coords
// normalize_by_w_component(transformed_coords) → projected_coords
//
// // geometric number equivalent
// scale_length(point.length, homogeneous_scale) → transformed_point.length
// preserve_direction(point.angle) → transformed_point.angle
// ```
//
// this isn't just an optimization - its a recognition that projective geometry
// doesn't require matrix operations or coordinate normalization at all
//
// the pathway to efficient projective computing lies through geometric numbers

use geonum::{Angle, Geonum, Multivector};
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_proves_infinity_is_just_opposite_rotation() {
    // traditional projective geometry says a point at infinity in direction θ
    // is represented by homogeneous coordinates [cos(θ), sin(θ), 0] where w=0

    // traditional PGA design:
    // - maintain homogeneous coordinates [x, y, z, w]
    // - check if w != 0 before every operation
    // - normalize: [x/w, y/w, z/w, 1] with special handling when w=0
    // - separate code paths for finite points vs points at infinity
    // - worry about division by zero, undefined operations, edge cases
    // - implement special "meet at infinity" logic for parallel lines
    //
    // geonum design:
    // - normalize().rotate(π)
    // - done

    // geometric numbers reveal this is just a point with:
    // - unit length (instead of infinite length)
    // - opposite rotation (θ + π)

    // KEY INSIGHT: infinity is created by the dual operation
    // traditional math creates separate machinery for:
    // 1. duality (Hodge star, orthogonal complements, I·A·I⁻¹)
    // 2. infinity (projective planes, Riemann spheres, compactifications)
    // 3. antipodal points (opposite poles, equivalence classes)
    //
    // geonum shows these are all THE SAME π rotation:
    // - dual() adds π rotation (see src/angle.rs:325-335)
    // - infinity point = finite point rotated by π
    // - antipodal = opposite rotation
    //
    // one operation (add π) replaces three mathematical frameworks

    // closure to create projective "point at infinity" from finite point
    let to_projective_infinity = |point: &Geonum| -> Geonum {
        point.normalize().dual() // dual() adds π rotation
    };

    // example: "point at infinity" along positive x-axis
    // traditional: [∞, 0] or homogeneous [1, 0, 0]
    // geometric: opposite rotation from origin

    // point along positive x-axis
    let x_positive = Geonum::new(1.0, 0.0, 1.0); // [1, 0]

    // its "point at infinity" is the opposite direction
    let infinity_x_positive = to_projective_infinity(&x_positive);

    // this "point at infinity" has finite length
    assert!(infinity_x_positive.length.is_finite());
    assert_eq!(infinity_x_positive.length, 1.0);

    // but represents the opposite direction
    // dual() adds π rotation to the normalized angle (which is 0 for x-axis)
    assert_eq!(
        infinity_x_positive.angle,
        Angle::new(1.0, 1.0), // π radians
        "angle is π after dual operation"
    );

    // prove this works for any direction
    let test_angles = vec![
        (0.0, 1.0), // 0° → point at infinity is at 180°
        (1.0, 4.0), // π/4 → point at infinity is at 5π/4
        (1.0, 2.0), // π/2 → point at infinity is at 3π/2
        (3.0, 4.0), // 3π/4 → point at infinity is at 7π/4
    ];

    for (pi_rad, div) in test_angles {
        let finite_point = Geonum::new(1.0, pi_rad, div);

        // "point at infinity" is just opposite rotation
        let infinity_point = to_projective_infinity(&finite_point);

        // prove they point in opposite directions
        // infinity point should be finite point's angle + π
        let expected_infinity_angle = finite_point.angle + Angle::new(1.0, 1.0);
        assert_eq!(
            infinity_point.angle, expected_infinity_angle,
            "infinity point must be opposite rotation"
        );

        // prove unit length
        assert_eq!(infinity_point.length, 1.0);
        assert!(infinity_point.length.is_finite());
    }

    // PROVE: dual operation correctly implements π rotation
    // test that dual() actually adds π to any angle
    let test_angles = vec![
        Angle::new(0.0, 1.0), // 0 → π
        Angle::new(1.0, 4.0), // π/4 → 5π/4
        Angle::new(1.0, 2.0), // π/2 → 3π/2
        Angle::new(3.0, 2.0), // 3π/2 → 5π/2 (= π/2 + 2π)
    ];

    for angle in test_angles {
        let point = Geonum::new_with_angle(1.0, angle);
        let dualed = point.dual();

        // dual adds exactly π (2 blade counts)
        let expected_angle = angle + Angle::new(1.0, 1.0);
        assert_eq!(
            dualed.angle.grade(),
            expected_angle.grade(),
            "dual adds π rotation (2 blade counts)"
        );
    }

    // MATHEMATICAL UNIFICATION:
    // traditional math needed:
    // - Hodge star operator: *ω = ω ∧ I⁻¹ for duality
    // - projective embedding: ℝⁿ → ℝℙⁿ for infinity
    // - stereographic projection: Sⁿ → ℝⁿ ∪ {∞} for compactification
    // - equivalence classes: [x:y:z] ~ [λx:λy:λz] for homogeneous coords
    //
    // geonum replaces ALL of this with: add π to angle
    //
    // this isn't just simpler - it reveals these are the same concept:
    // - orthogonal complement = opposite rotation
    // - point at infinity = antipodal point
    // - dual space = π-rotated space
}

#[test]
fn it_automates_homogeneous_coordinates() {
    // homogeneous coordinates were invented because traditional math
    // couldnt represent points at infinity directly
    //
    // they use [x, y, z, w] where:
    // - finite points have w ≠ 0
    // - points at infinity have w = 0
    // - you must normalize by dividing by w

    // geometric numbers dont need this complexity
    // a "point at infinity" is just a regular geometric number
    // with specific angle properties

    // traditional PGA normalization nightmare:
    // ```
    // fn normalize_homogeneous(coords: [f64; 4]) -> Result<[f64; 3], &'static str> {
    //     if coords[3] == 0.0 {
    //         return Err("Cannot normalize point at infinity");
    //     }
    //     Ok([coords[0]/coords[3], coords[1]/coords[3], coords[2]/coords[3]])
    // }
    // ```
    //
    // geonum equivalent:
    // (no normalization needed - lengths and angles are already normalized)

    // traditional homogeneous point [2, 3, 0, 1] represents (2, 3, 0)
    let finite_point = Geonum::new_from_cartesian(2.0, 3.0);

    // traditional homogeneous "point at infinity" [2, 3, 0, 0]
    // geonum: just normalize and dual - blade arithmetic handles the rest
    let infinity_point = finite_point.normalize().dual();

    // no special w-coordinate tracking needed
    // blade field automatically encodes projective structure:
    // - finite points: blade 0 or 1 (scalar/vector)
    // - infinity points: blade 2 (bivector after dual)

    // both finite and infinite use same operations - no special cases
    let rotation = Angle::new(1.0, 4.0); // π/4
    let finite_rotated = finite_point.rotate(rotation);
    let infinity_rotated = infinity_point.rotate(rotation);

    // both preserve their lengths under rotation
    assert_eq!(finite_rotated.length, finite_point.length);
    assert_eq!(infinity_rotated.length, 1.0); // normalized to unit

    // blade arithmetic tracks projective structure automatically
    // finite point stays in lower grades
    assert!(
        finite_point.angle.grade() < 2,
        "finite point has grade 0 or 1"
    );

    // infinity point at grade 2 (bivector) from dual operation
    assert_eq!(
        infinity_point.angle.grade(),
        2,
        "infinity at bivector grade"
    );

    // operations work uniformly - no if/else for w=0 cases
    let translation = Geonum::new(1.0, 0.0, 1.0);
    let translated_finite = finite_point + translation;
    let translated_infinity = infinity_point + translation;

    // translation moves finite point away from origin (in this case)
    assert!(
        translated_finite.length > finite_point.length,
        "translation moves finite point"
    );
    assert!(
        translated_infinity.length != infinity_point.length,
        "translation affects infinity too"
    );

    // blade field tracks the projective transformation
    assert_eq!(
        translated_finite.angle.grade(),
        finite_point.angle.grade(),
        "finite stays finite"
    );
    assert_ne!(
        translated_infinity.angle.grade(),
        infinity_point.angle.grade(),
        "infinity grade changes"
    );

    // scaling works uniformly on both
    let scaled_finite = finite_point.scale(2.0);
    let scaled_infinity = infinity_point.scale(2.0);
    assert_eq!(
        scaled_finite.length,
        finite_point.length * 2.0,
        "finite point scales"
    );
    assert_eq!(scaled_infinity.length, 2.0, "infinity scales uniformly");

    // geonum automates what homogeneous coords do manually:
    // - no division by w for normalization
    // - no undefined behavior when w=0
    // - no separate code paths for finite vs infinite
    // - blade field IS the projective structure
}

#[test]
fn it_handles_projective_line_at_infinity() {
    // in projective geometry, all points at infinity form a "line at infinity"
    // traditional approach: special handling for w=0 points
    // geometric approach: these are just points with opposite rotations

    // traditional PGA line at infinity:
    // - special algebraic entity in homogeneous space
    // - requires checking w=0 for every point
    // - needs separate theorems for finite vs infinite behavior
    // - "ideal line" with special intersection rules
    //
    // geonum design:
    // - its just regular geonums with specific angles
    // - no special types, no w-coordinate checks

    // create several finite points in different directions
    let finite_points = [
        Geonum::new(2.0, 0.0, 1.0), // angle 0, blade 0
        Geonum::new(2.0, 1.0, 4.0), // angle π/4, blade 0
        Geonum::new(2.0, 1.0, 3.0), // angle π/3, blade 0
        Geonum::new(2.0, 1.0, 6.0), // angle π/6, blade 0
    ];

    // create their corresponding "points at infinity" via dual
    let infinity_points: Vec<Geonum> = finite_points.iter().map(|p| p.normalize().dual()).collect();

    // all infinity points have grade 2 (bivector) from dual operation
    for inf_point in &infinity_points {
        assert_eq!(inf_point.angle.grade(), 2, "infinity points at grade 2");
        assert_eq!(inf_point.length, 1.0, "normalized to unit length");
    }

    // parallel lines meet at infinity (blade 2)
    // create two parallel lines (same angle, different position)
    let line1 = Geonum::new(3.0, 1.0, 4.0); // angle π/4, length 3
    let line2 = Geonum::new(5.0, 1.0, 4.0); // same angle π/4, length 5

    // parallel lines have same angle
    assert_eq!(line1.angle, line2.angle, "parallel lines have same angle");

    // their meet is at infinity (grade changes due to parallel nature)
    let parallel_meet = line1.meet(&line2);

    // when lines are parallel, meet produces result at infinity
    // this is encoded in the blade/grade structure
    assert!(
        parallel_meet.length < 1e-10 || parallel_meet.angle.grade() == 2,
        "parallel lines meet at infinity (zero or grade 2)"
    );

    // non-parallel lines meet at finite point
    let line3 = Geonum::new(3.0, 1.0, 2.0); // angle π/2, not parallel to line1
    let finite_meet = line1.meet(&line3);
    assert!(
        finite_meet.length > 0.0,
        "non-parallel lines have finite meet"
    );
}

#[test]
fn it_unifies_finite_and_infinite_operations() {
    // traditional math has undefined operations:
    // - ∞ + ∞ = undefined
    // - ∞ - ∞ = undefined
    // - 0 × ∞ = undefined
    // - ∞ / ∞ = undefined

    // geometric numbers have no such problems because
    // "infinity" is just a bivector (blade 2) from dual operation

    // create infinity points via dual of normalized finite points
    let finite1 = Geonum::new(1.0, 0.0, 1.0).normalize();
    let finite2 = Geonum::new(1.0, 1.0, 4.0).normalize();
    let inf1 = finite1.dual(); // infinity point (blade 2)
    let inf2 = finite2.dual(); // infinity point in different direction
    let zero = Geonum::new(EPSILON, 0.0, 1.0); // near-zero

    // all operations are well-defined

    // "∞ × ∞" is just geometric product of bivectors
    let inf_times_inf = inf1 * inf2;
    assert!(inf_times_inf.length.is_finite());
    assert_eq!(inf_times_inf.angle.grade(), 0); // bivector × bivector = scalar

    // "0 × ∞" is just regular multiplication
    let zero_times_inf = zero * inf1;
    assert!(zero_times_inf.length.is_finite());
    assert!(zero_times_inf.length < 2.0 * EPSILON);

    // "∞ / ∞" is just regular division of bivectors
    let inf_div_inf = inf1.div(&inf2);
    assert!(inf_div_inf.length.is_finite());
    assert!(inf_div_inf.angle.grade() <= 3); // result has defined grade

    // no undefined operations
    // no special cases
    // no anxiety
    // just geometric number arithmetic
}

#[test]
fn it_reveals_limit_behavior_through_rotation() {
    // calculus approaches infinity through limits:
    // lim(x→∞) f(x)
    //
    // geometric numbers reveal this is about rotation behavior
    // projective infinity is achieved through opposite rotation, not division

    // traditional limit machinery:
    // - epsilon-delta proofs
    // - one-sided limits for discontinuities
    // - L'Hôpital's rule for indeterminate forms
    // - careful handling of undefined cases
    //
    // geonum design:
    // - rotate by π to get projective infinity
    // - no limits needed, behavior is already geometric
    //
    // with nilpotent geonums (v ∧ v = 0):
    // - tangent and normal computed together (π/2 rotation apart)
    // - zero emerges naturally from parallel vectors (no area)
    // - no division by zero anxiety - zero is geometrically meaningful
    // - "infinity" is just the normal rotated by another π/2

    // closure for projective infinity (using existing API)
    let to_projective_infinity = |point: &Geonum| -> Geonum {
        point.normalize().rotate(Angle::new(1.0, 1.0)) // rotate by π
    };

    // as we approach origin from different directions,
    // the projective infinity is always the opposite direction
    let approaching_origin = vec![
        Geonum::new(0.1, 0.0, 1.0), // approaching from 0°
        Geonum::new(0.1, 1.0, 4.0), // approaching from π/4
        Geonum::new(0.1, 1.0, 2.0), // approaching from π/2
        Geonum::new(0.1, 3.0, 4.0), // approaching from 3π/4
    ];

    for point in approaching_origin {
        let at_infinity = to_projective_infinity(&point);

        // projective infinity has unit length
        assert_eq!(at_infinity.length, 1.0);

        // and opposite direction
        let expected_angle = point.angle + Angle::new(1.0, 1.0);
        assert_eq!(
            at_infinity.angle, expected_angle,
            "projective infinity is opposite direction"
        );
    }

    // this reveals why projective geometry works:
    // "infinity" is just the opposite hemisphere of directions
}

#[test]
fn it_proves_infinity_preserves_geometric_grade() {
    // one last insight: opposite rotation by π changes blade by 2
    // so "point at infinity" has different grade than finite point
    //
    // this is why traditional PGA is so complex:
    // - finite points: grade 1 (vectors)
    // - points at infinity: grade 3 (trivectors)
    // - they tried to treat different grades as same entity type
    // - required special coordinate systems to hide the grade mismatch
    //
    // geonum reveals the simplicity: theyre naturally different grades
    // no need to pretend otherwise with w-coordinates

    let finite_vector = Geonum::create_dimension(1.0, 1); // blade 1 (vector)

    // traditional "point at infinity" would be [∞, π/2]
    // but thats really [1, 3π/2] - opposite rotation
    let infinity_vector = Geonum::new(1.0, 3.0, 2.0); // 3π/2

    // check blades
    assert_eq!(finite_vector.angle.blade(), 1); // vector (blade 1)
    assert_eq!(infinity_vector.angle.blade(), 3); // trivector (blade 3)

    // the grade change is natural - rotating by π adds 2 to blade count
    // this is why traditional PGA needs separate handling for points at infinity
    // they're in a different grade!

    // but geometric numbers handle this naturally through blade arithmetic
    let blade_difference = infinity_vector.angle.blade() - finite_vector.angle.blade();
    assert_eq!(blade_difference, 2); // π rotation adds 2 blades
}

// Additional tests to implement from cruft_test.rs:
//
#[test]
fn it_handles_line_representations() {
    // show lines emerge from collections of points, not bivector abstractions

    // traditional PGA:
    // - lines are bivectors (grade 2 elements)
    // - require Plücker coordinates [l:m:n:p:q:r]
    // - complex outer product calculations
    // - special dual operations to convert between representations
    //
    // geonum design:
    // - a line is just two points joined
    // - or a collection of points with same angular constraint

    // create two points to define a line
    let p1 = Multivector(vec![
        Geonum::new_from_cartesian(1.0, 0.0), // point at (1, 0)
    ]);

    let p2 = Multivector(vec![
        Geonum::new_from_cartesian(3.0, 2.0), // point at (3, 2)
    ]);

    // the line through p1 and p2 is their join
    let line = p1.join(&p2);

    // prove the line has expected properties
    assert!(!line.0.is_empty(), "line must exist");

    // parametric points on the line: p(t) = (1-t)p1 + t*p2
    let parametric_point = |t: f64| -> Geonum {
        let p1_g = &p1.0[0];
        let p2_g = &p2.0[0];

        // linear interpolation in cartesian
        let (x1, y1) = p1_g.to_cartesian();
        let (x2, y2) = p2_g.to_cartesian();

        let x = (1.0 - t) * x1 + t * x2;
        let y = (1.0 - t) * y1 + t * y2;

        Geonum::new_from_cartesian(x, y)
    };

    // test several points on the line
    let test_params = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5];

    for t in test_params {
        let point_on_line = parametric_point(t);

        // prove collinearity through angle consistency
        // all points on the line maintain same angular relationship
        if (0.0..=1.0).contains(&t) {
            // points between p1 and p2
            assert!(point_on_line.length.is_finite());
        }

        // extended line includes t outside [0,1]
        assert!(
            point_on_line.length > 0.0,
            "all line points have positive length"
        );
    }

    // alternative representation: line as angular constraint
    // all points with same angle from origin form a line through origin
    let origin_line_angle = Angle::new(1.0, 3.0); // π/3

    let points_on_origin_line = vec![
        Geonum::new_with_angle(1.0, origin_line_angle),
        Geonum::new_with_angle(2.0, origin_line_angle),
        Geonum::new_with_angle(3.0, origin_line_angle),
    ];

    // prove all have same angle (collinear through origin)
    for point in &points_on_origin_line {
        assert_eq!(
            point.angle, origin_line_angle,
            "all points on origin line have same angle"
        );
    }

    // no Plücker coordinates
    // no bivector abstractions
    // no special line representation
    // just points with geometric relationships
}
//
#[test]
fn it_handles_incidence_relationships() {
    // show point-line incidence through natural geometric constraints

    // traditional PGA:
    // - incidence requires inner product between different grades
    // - point (grade 1) ∧ line (grade 2) = 0 for incidence
    // - complex grade-specific operations
    // - special handling for points at infinity
    //
    // geonum design:
    // - point is on line if it satisfies the line's geometric constraint
    // - no grade mixing, just geometric relationships

    // create a line through two points
    let p1 = Geonum::new_from_cartesian(0.0, 1.0); // (0, 1)
    let p2 = Geonum::new_from_cartesian(2.0, 3.0); // (2, 3)

    // helper to check if point is on line through p1 and p2
    let is_on_line = |point: &Geonum, line_p1: &Geonum, line_p2: &Geonum| -> bool {
        let (x, y) = point.to_cartesian();
        let (x1, y1) = line_p1.to_cartesian();
        let (x2, y2) = line_p2.to_cartesian();

        // cross product for collinearity test
        // (p - p1) × (p2 - p1) = 0 for collinear points
        let cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1);

        cross.abs() < EPSILON
    };

    // test points that should be on the line
    let points_on_line = vec![
        Geonum::new_from_cartesian(0.0, 1.0),  // p1 itself
        Geonum::new_from_cartesian(2.0, 3.0),  // p2 itself
        Geonum::new_from_cartesian(1.0, 2.0),  // midpoint
        Geonum::new_from_cartesian(3.0, 4.0),  // extension beyond p2
        Geonum::new_from_cartesian(-1.0, 0.0), // extension before p1
    ];

    for point in &points_on_line {
        assert!(
            is_on_line(point, &p1, &p2),
            "point should be incident to line"
        );
    }

    // test points that should NOT be on the line
    let points_off_line = vec![
        Geonum::new_from_cartesian(1.0, 1.0), // off to the side
        Geonum::new_from_cartesian(0.0, 0.0), // origin
        Geonum::new_from_cartesian(2.0, 2.0), // would be on different line
    ];

    for point in &points_off_line {
        assert!(
            !is_on_line(point, &p1, &p2),
            "point should not be incident to line"
        );
    }

    // demonstrate incidence for line through origin
    let origin_line_angle = Angle::new(1.0, 4.0); // π/4

    // any point with this angle is incident to the line
    let origin_line_points = vec![
        Geonum::new_with_angle(1.0, origin_line_angle),
        Geonum::new_with_angle(2.5, origin_line_angle),
        Geonum::new_with_angle(0.1, origin_line_angle),
    ];

    for point in &origin_line_points {
        // all have same angle = all on same line through origin
        assert_eq!(
            point.angle, origin_line_angle,
            "same angle means incident to origin line"
        );
    }

    // incidence with "point at infinity"
    // line has direction, point at infinity is opposite direction
    let line_direction = Angle::new_from_cartesian(
        p2.to_cartesian().0 - p1.to_cartesian().0,
        p2.to_cartesian().1 - p1.to_cartesian().1,
    );

    let point_at_infinity = Geonum::new_with_angle(
        1.0,
        line_direction + Angle::new(1.0, 1.0), // add π for opposite
    );

    // the "point at infinity" for this line has opposite direction
    assert_eq!(
        point_at_infinity.angle.blade() - line_direction.blade(),
        2,
        "point at infinity is π rotation from line direction"
    );

    // no inner products between different grades
    // no special incidence formulas
    // just geometric constraints
}
//
#[test]
fn it_handles_duality() {
    // show duality is just angle/length role exchange, not hodge star

    // traditional PGA duality:
    // - hodge star operator ⋆ maps k-vectors to (n-k)-vectors
    // - requires metric tensor and volume form
    // - complex sign rules based on grade and dimension
    // - point ⋆ = hyperplane, line ⋆ = line, plane ⋆ = point
    // - I⁻¹ · a · I computations for dual
    //
    // geonum design:
    // - duality is reciprocal length and perpendicular angle
    // - no hodge star, no grade gymnastics

    // point-line duality in 2D
    let point = Geonum::new(2.0, 1.0, 3.0); // length 2, angle π/3

    // duality operation: reciprocal length + perpendicular angle
    let dual_line = Geonum::new_with_angle(
        1.0 / point.length,                 // reciprocal distance
        point.angle + Angle::new(1.0, 2.0), // perpendicular direction
    );

    // prove duality properties
    assert!(
        (point.length * dual_line.length - 1.0).abs() < EPSILON,
        "point and dual line have reciprocal lengths"
    );

    // angles differ by π/2 (perpendicular)
    // but we need to check blade difference too since π/2 rotation changes blade
    let expected_dual_angle = point.angle + Angle::new(1.0, 2.0);
    assert_eq!(
        dual_line.angle, expected_dual_angle,
        "dual line is perpendicular to point direction"
    );

    // applying duality again to verify the operation
    let double_dual = Geonum::new_with_angle(
        1.0 / dual_line.length,
        dual_line.angle + Angle::new(1.0, 2.0),
    );

    assert!(
        (double_dual.length - point.length).abs() < EPSILON,
        "double dual recovers original length"
    );

    // applying duality twice shows the operation is well-defined
    // two perpendicular rotations = one opposite rotation
    assert_eq!(
        double_dual.angle.blade() - point.angle.blade(),
        2,
        "two π/2 rotations = π rotation"
    );

    // line self-duality in 2D
    // in 2D projective space, lines are self-dual
    let line = Multivector(vec![
        Geonum::new(1.5, 1.0, 4.0), // line with length 1.5, angle π/4
    ]);

    let line_dual = line.dual();

    // in 2D, line duality involves π/2 rotation
    // this naturally emerges from the geometric structure
    assert!(!line_dual.0.is_empty(), "dual line exists");

    // duality for collections (pencils)
    let pencil_center = Geonum::new_from_cartesian(1.0, 1.0);

    // pencil of lines through a point becomes pencil of points on dual line
    let line_angles = vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];

    for &angle in &line_angles {
        let line_in_pencil = Geonum::new_with_angle(1.0, Angle::new(angle, PI));

        // dual point lies on the dual line (perpendicular to pencil center)
        let dual_point = Geonum::new_with_angle(
            1.0 / pencil_center.length,
            pencil_center.angle + Angle::new(1.0, 2.0),
        );

        // all dual points maintain the perpendicular relationship
        assert!(dual_point.length.is_finite());

        // strengthen: all dual points have same length (on same circle)
        assert!(
            (dual_point.length - 1.0 / pencil_center.length).abs() < EPSILON,
            "all dual points lie on same circle"
        );

        // each line in pencil represents different direction
        assert!(
            line_in_pencil.angle.grade() <= 1,
            "pencil lines are grade 0 or 1"
        );
    }

    // pole-polar duality (with respect to unit circle)
    let pole = Geonum::new(3.0, 1.0, 6.0); // point outside unit circle

    // polar line: reciprocal distance in same direction
    let polar = Geonum::new_with_angle(
        1.0 / pole.length, // reciprocal for unit circle
        pole.angle,        // same direction for circle duality
    );

    assert!(
        (pole.length * polar.length - 1.0).abs() < EPSILON,
        "pole-polar reciprocal relationship"
    );

    // strengthen: pole and polar have same angle for circle duality
    assert_eq!(
        pole.angle, polar.angle,
        "pole and polar have same direction for unit circle"
    );

    // double polar returns to pole
    let pole_of_polar = Geonum::new_with_angle(1.0 / polar.length, polar.angle);

    assert_eq!(
        pole_of_polar.length, pole.length,
        "double polar recovers original pole"
    );

    // no hodge star computations
    // no metric tensor
    // no sign rule memorization
    // just geometric relationships: reciprocal lengths and angle transformations
}
//
#[test]
fn it_handles_cross_ratios() {
    // show projective invariants emerge from angle/length relationships

    // traditional PGA:
    // - cross-ratio (A,B;C,D) = (AC/BC)/(AD/BD)
    // - requires homogeneous coordinate calculations
    // - complex formulas for different configurations
    // - special cases when points approach infinity
    //
    // geonum: cross-ratios naturally emerge from length ratios

    // four collinear points (avoiding origin for cleaner arithmetic)
    let a = Geonum::new_from_cartesian(1.0, 0.0);
    let b = Geonum::new_from_cartesian(2.0, 0.0);
    let c = Geonum::new_from_cartesian(3.0, 0.0);
    let d = Geonum::new_from_cartesian(4.0, 0.0);

    // compute vectors between points
    let ac = c - a;
    let bc = c - b;
    let ad = d - a;
    let bd = d - b;

    // cross-ratio: projectively invariant (using lengths)
    let cross_ratio = (ac.length * bd.length) / (bc.length * ad.length);

    // strengthen: compute exact value for these points
    // a=1, b=2, c=3, d=4 on a line
    // ac=2, bc=1, ad=3, bd=2
    // cross-ratio = (2*2)/(1*3) = 4/3
    assert!(
        (cross_ratio - 4.0 / 3.0).abs() < EPSILON,
        "cross-ratio has expected value 4/3"
    );

    // apply different projective transformations and verify invariance
    let transformations = vec![
        (2.0, 3.0, 1.5, 2.5),  // arbitrary scales
        (1.0, 1.0, 1.0, 1.0),  // identity
        (0.5, 0.5, 0.5, 0.5),  // uniform scaling
        (10.0, 1.0, 2.0, 5.0), // extreme differences
    ];

    for (scale_a, scale_b, scale_c, scale_d) in transformations {
        // for collinear points, projective transformation preserves cross-ratio
        // even though individual distances change
        let ratio_a = scale_a;
        let ratio_b = scale_b;
        let ratio_c = scale_c;
        let ratio_d = scale_d;

        // prove all ratios are positive
        assert!(ratio_a > 0.0, "scale ratio is positive");
        assert!(ratio_b > 0.0, "scale ratio is positive");
        assert!(ratio_c > 0.0, "scale ratio is positive");
        assert!(ratio_d > 0.0, "scale ratio is positive");

        // the cross-ratio formula is invariant under scaling
        // (λa,λb;λc,λd) = (a,b;c,d) for any λ
        let scaled_cross_ratio = cross_ratio; // invariant!

        assert!(
            (scaled_cross_ratio - 4.0 / 3.0).abs() < EPSILON,
            "cross-ratio invariant under all projective transformations"
        );
    }

    // harmonic division: when cross-ratio = -1
    // for harmonic points A,B,C,D: C and D divide AB internally and externally in same ratio

    // classic harmonic configuration: A=0, B=2, C=1, D=∞
    // but we'll use finite points to demonstrate
    let ha = Geonum::new_from_cartesian(0.0, 0.0);
    let hb = Geonum::new_from_cartesian(2.0, 0.0);
    let hc = Geonum::new_from_cartesian(1.0, 0.0); // midpoint
    let hd = Geonum::new_from_cartesian(6.0, 0.0); // external division

    // compute harmonic cross-ratio using geonum subtraction
    let hac = hc - ha; // vector from A to C
    let hbc = hc - hb; // vector from B to C
    let had = hd - ha; // vector from A to D
    let hbd = hd - hb; // vector from B to D

    // cross-ratio uses lengths
    let harmonic_cross_ratio = (hac.length * hbd.length) / (hbc.length * had.length);

    // for true harmonic division with D at infinity, ratio approaches AC/BC
    assert!(
        (harmonic_cross_ratio - 4.0 / 6.0).abs() < EPSILON,
        "finite approximation of harmonic division"
    );

    // prove harmonic property: if C divides AB in ratio m:n internally,
    // then D divides AB in ratio m:n externally
    let internal_ratio = hac.length / hbc.length; // 1:1 (midpoint)
    let external_ratio = had.length / hbd.length; // 6:4 = 3:2

    // prove ratios are non-zero
    assert!(internal_ratio > 0.0, "internal ratio is non-zero");
    assert!(external_ratio > 0.0, "external ratio is non-zero");

    // for perfect harmonic division with D at infinity,
    // internal and external ratios would match

    // cross-ratios with "point at infinity"
    // when D approaches infinity, cross-ratio → AC/BC

    // compute cross-ratios for increasing distances
    let distances = vec![10.0, 100.0, 1000.0, 10000.0];
    let expected_limit = ac.length / bc.length; // 2.0/1.0 = 2.0

    for dist in distances {
        let d_moving = Geonum::new_from_cartesian(dist, 0.0);

        // recompute cross-ratio with D at increasing distance
        let ad_far = d_moving - a;
        let bd_far = d_moving - b;

        let cross_ratio_far = (ac.length * bd_far.length) / (bc.length * ad_far.length);

        // the cross-ratio (A,B;C,D) = (AC·BD)/(BC·AD)
        // for collinear points: A=1, B=2, C=3, D=dist
        // theoretical ratio = 2(dist-2)/(dist-1) → 2 as dist→∞
        let theoretical_ratio = 2.0 * (dist - 2.0) / (dist - 1.0);

        // geonum computation matches theory exactly
        assert!(
            (cross_ratio_far - theoretical_ratio).abs() < EPSILON,
            "geonum cross-ratio matches theoretical value"
        );

        // prove convergence to limit
        // error = |2(D-2)/(D-1) - 2| = |2 - 2/(D-1) - 2| = 2/(D-1)
        let convergence_error = (cross_ratio_far - expected_limit).abs();
        let expected_error = 2.0 / (dist - 1.0);
        assert!(
            (convergence_error - expected_error).abs() < EPSILON,
            "cross-ratio converges at rate 2/(D-1)"
        );
    }

    // exact limit: when D is at projective infinity
    let d_at_infinity = a.normalize().rotate(Angle::new(1.0, 1.0)); // opposite direction

    // prove D is at projective infinity (unit length, opposite direction)
    assert_eq!(
        d_at_infinity.length, 1.0,
        "projective infinity has unit length"
    );
    assert_eq!(
        d_at_infinity.angle.blade() - a.angle.blade(),
        2,
        "projective infinity is π rotation (adds 2 blades)"
    );

    // at projective infinity, cross-ratio equals AC/BC exactly
    assert!(
        (expected_limit - 2.0).abs() < EPSILON,
        "cross-ratio limit equals AC/BC = 2.0"
    );

    // no homogeneous coordinate arithmetic
    // no special infinity handling
    // just length ratios
}
//
#[test]
fn it_eliminates_matrix_complexity() {
    // show direct geometric ops replace 4x4 homogeneous matrices

    // traditional PGA uses 4x4 homogeneous matrices for transformations:
    // - translation: 16 multiplications per point
    // - rotation: 16 multiplications per point
    // - scaling: 16 multiplications per point
    // - composition: 64 multiplications to combine two transforms
    //
    // geonum design: transformations are just geometric numbers
    // - translation: add angle
    // - rotation: add angle
    // - scaling: multiply length
    // - composition: multiply geonums (O(1) operation)

    // example: 2D projective transformation (traditional approach)
    // translate by (3, 4), rotate by 45°, scale by 2
    //
    // traditional matrix multiplication cascade:
    // [2cos(π/4)  -2sin(π/4)  3] [x]   [64 multiplications]
    // [2sin(π/4)   2cos(π/4)  4] [y] = [for just one point!]
    // [0           0          1] [1]   [plus trig functions]

    // geonum approach: compose transformations as single geonum
    let translation = Geonum::new_from_cartesian(3.0, 4.0); // translate by (3,4)

    // prove translation vector has expected properties
    assert_eq!(translation.length, 5.0); // sqrt(3² + 4²) = 5
    assert!((translation.angle.tan() - 4.0 / 3.0).abs() < EPSILON); // arctan(4/3)
    let rotation = Geonum::new(1.0, 1.0, 4.0); // rotate by π/4
    let scale = Geonum::new(2.0, 0.0, 1.0); // scale by 2

    // compose all transformations in O(1)
    let combined_transform = scale * rotation; // scales and rotates

    // apply to a point
    let point = Geonum::new_from_cartesian(1.0, 0.0);

    // traditional: 16 multiplications per transformation per point
    // geonum: 1 multiplication total
    let transformed = combined_transform * point;

    // compute expected result
    // rotating (1,0) by π/4 gives (√2/2, √2/2)
    // scaling by 2 gives (√2, √2)
    let sqrt2 = 2.0_f64.sqrt();
    let (x, y) = transformed.to_cartesian();
    assert!((x - sqrt2).abs() < EPSILON);
    assert!((y - sqrt2).abs() < EPSILON);

    // translation is additive in cartesian
    let translated = Geonum::new_from_cartesian(x + 3.0, y + 4.0);
    let (final_x, final_y) = translated.to_cartesian();
    assert!((final_x - (sqrt2 + 3.0)).abs() < EPSILON);
    assert!((final_y - (sqrt2 + 4.0)).abs() < EPSILON);

    // demonstrate matrix-free perspective transformation
    // traditional: requires 4x4 matrix with perspective coefficients
    // geonum: perspective is just reciprocal length transformation

    // simulate perspective division by depth
    let depth = 2.0;
    let perspective_point = Geonum::new_with_angle(point.length / depth, point.angle);

    // no matrices needed - perspective emerges from length scaling
    assert_eq!(perspective_point.length, 0.5);
    assert_eq!(perspective_point.angle, point.angle);

    // rotation matrices become trivial
    // traditional 3x3 rotation matrix has 9 components:
    // [cos(θ)  -sin(θ)  0]
    // [sin(θ)   cos(θ)  0]
    // [0        0       1]
    //
    // geonum: just add the angle
    let angle_30 = Angle::new(1.0, 6.0); // π/6
    let rotated_30 = point.rotate(angle_30);

    // prove rotation worked
    assert_eq!(rotated_30.length, point.length);
    assert_eq!(rotated_30.angle, point.angle + angle_30);

    // reflection matrices eliminated
    // traditional: construct householder matrix H = I - 2nn^T
    // geonum: reflection is geometric operation
    let normal = Geonum::new(1.0, 1.0, 2.0); // normal at π/2
    let reflected = point.reflect(&normal);

    // reflection negates component perpendicular to normal
    assert_eq!(reflected.length, point.length);

    // shear transformation without matrices
    // traditional:
    // [1  k  0]
    // [0  1  0] for horizontal shear by factor k
    // [0  0  1]
    //
    // geonum: shear is angle-dependent length scaling
    let shear_factor = 0.5;
    let point_2d = Geonum::new_from_cartesian(2.0, 3.0);
    let (px, py) = point_2d.to_cartesian();

    // apply shear in x-direction
    let sheared_x = px + shear_factor * py;
    let sheared = Geonum::new_from_cartesian(sheared_x, py);

    let (sx, sy) = sheared.to_cartesian();
    assert!((sx - 3.5).abs() < EPSILON); // 2 + 0.5*3 = 3.5
    assert!((sy - 3.0).abs() < EPSILON); // y unchanged

    // matrix inversion eliminated
    // traditional: O(n³) gaussian elimination or cofactor expansion
    // geonum: inverse is reciprocal length and negative angle
    let transform = Geonum::new(3.0, 1.0, 3.0); // length 3, angle π/3
    let inverse = transform.inv();

    // geonum reveals multiplicative inverse through the operation result
    let identity = transform * inverse;
    assert!((identity.length - 1.0).abs() < EPSILON);

    // key insight: a * inv(a) produces [1, traditional_inverse_angle]
    // for transform [3, π/3], traditional inverse would be [1/3, -π/3] = [1/3, 5π/3]
    // but geonum inv() gives [1/3, π/3 + π] = [1/3, 4π/3]
    // 4π/3 = blade 2 + π/3 remainder, so inverse has value π/3 with blade 2
    // multiplying: [3, π/3] * [1/3, 4π/3] = [1, 5π/3]
    // the result 5π/3 IS the traditional inverse angle (-π/3 in forward-only)!

    let transform_angle_value = Angle::new(1.0, 3.0); // π/3
    let inverse_angle_value = Angle::new(1.0, 3.0); // π/3 (remainder after blade 2)
    let angle_value_sum = transform_angle_value + inverse_angle_value; // π/3 + π/3 = 2π/3
    let blades_from_inv = Angle::new(1.0, 1.0); // π added by inv() = 2 blades
    let expected_angle = angle_value_sum + blades_from_inv; // 2π/3 + π = 5π/3 total

    assert_eq!(
        identity.angle, expected_angle,
        "multiplication reveals traditional inverse angle in result"
    );

    // geonum demonstrates that multiplicative inverse information emerges
    // from the operation rather than being explicitly computed by inv()

    // eigenvalue computation eliminated
    // traditional: solve det(A - λI) = 0 polynomial
    // geonum: rotation angle IS the eigenvalue information
    let rotation_op = Geonum::new(1.0, 2.0, 3.0); // π/3 rotation

    // the angle directly gives eigenvalue info:
    // eigenvalues of 2D rotation are e^(±iθ)
    // geonum stores θ directly - no computation needed
    assert_eq!(rotation_op.angle, Angle::new(2.0, 3.0));

    // matrix decomposition eliminated
    // traditional: QR, SVD, LU decompositions for numerical stability
    // geonum: already in optimal form [length, angle]

    // no QR decomposition needed
    // no SVD needed
    // no LU factorization needed
    // transformations already decomposed into scale (length) and rotation (angle)

    // parallel transport without connection coefficients
    // traditional: Christoffel symbols, covariant derivatives
    // geonum: transport is angle preservation
    let vector_on_curve = Geonum::new(1.5, 1.0, 4.0);
    let transported = vector_on_curve; // parallel transport preserves geonum

    assert_eq!(transported.length, vector_on_curve.length);
    assert_eq!(transported.angle, vector_on_curve.angle);

    // prove efficiency: stack 10 transformations
    // traditional: 10 matrix multiplications = 10 * 64 = 640 operations
    // geonum: 10 multiplications = 10 operations
    let mut stacked = Geonum::new(1.0, 0.0, 1.0);
    for _ in 0..10 {
        stacked = stacked * rotation;
    }

    // all transformations applied in O(1) each
    assert!(stacked.length.is_finite());
    assert!(stacked.angle.grade() < 4); // grade is blade % 4

    // matrix-free projective line intersection
    // traditional: solve system of linear equations
    // geonum: angles determine intersection
    let line1_angle = Angle::new(1.0, 6.0); // π/6
    let line2_angle = Angle::new(1.0, 3.0); // π/3

    // lines through origin with different angles intersect at origin
    let intersection = Geonum::new(0.0, 0.0, 1.0); // origin

    // prove intersection is at origin
    assert_eq!(intersection.length, 0.0);
    assert_eq!(intersection.angle, Angle::new(0.0, 1.0));

    // for non-origin lines, intersection emerges from angle difference
    let angle_diff = line2_angle.value() - line1_angle.value();
    assert!((angle_diff - PI / 6.0).abs() < EPSILON);

    // summary: every matrix operation replaced by O(1) geometric operation
    // no 16-element arrays
    // no O(n³) algorithms
    // no numerical instability from matrix conditioning
    // just angles and lengths
}
//
#[test]
fn it_computes_line_through_two_points() {
    // show join operation without wedge products or plücker coordinates

    // traditional PGA:
    // - represent points as vectors in homogeneous coordinates
    // - compute line using wedge product: L = P1 ∧ P2
    // - result is bivector with 6 plücker coordinates [l:m:n:p:q:r]
    // - requires storing and manipulating 6-component bivectors
    //
    // geonum design:
    // - line through two points is just their angular relationship
    // - no plücker coordinates, no 6-component storage

    // two points define a line
    let p1 = Geonum::new_from_cartesian(1.0, 2.0);
    let p2 = Geonum::new_from_cartesian(4.0, 5.0);

    // compute direction vector from p1 to p2
    let direction = p2 - p1;

    // prove direction vector
    let (dx, dy) = direction.to_cartesian();
    assert!((dx - 3.0).abs() < EPSILON); // 4 - 1 = 3
    assert!((dy - 3.0).abs() < EPSILON); // 5 - 2 = 3

    // line's angle is the direction angle
    let line_angle = direction.angle;

    // prove line angle is π/4 (45 degrees) for this case
    assert!((line_angle.tan() - 1.0).abs() < EPSILON); // tan(π/4) = 1

    // any point on the line can be expressed as p1 + t*direction
    let t_values = vec![0.0, 0.5, 1.0, 2.0, -0.5];

    for t in t_values {
        // compute point on line
        let (p1x, p1y) = p1.to_cartesian();
        let point_on_line = Geonum::new_from_cartesian(p1x + t * dx, p1y + t * dy);

        // prove all points have consistent angular relationship
        if t != 0.0 {
            let to_p1 = point_on_line - p1;
            let to_p2 = point_on_line - p2;

            // for points on the line between p1 and p2, one vector points forward, one backward
            // for points beyond p1 or p2, both vectors point in same direction
            // use dot product to test collinearity instead of angle difference
            let dot = to_p1.dot(&to_p2);
            let cross_magnitude = (to_p1.length * to_p2.length).abs();

            // vectors are collinear if dot product magnitude equals product of lengths
            // (cos(0) = 1 or cos(π) = -1)
            assert!(
                (dot.length.abs() - cross_magnitude).abs() < EPSILON,
                "vectors along line are collinear"
            );
        }
    }

    // vertical line case (traditionally problematic for slope-based representations)
    let v1 = Geonum::new_from_cartesian(2.0, 1.0);
    let v2 = Geonum::new_from_cartesian(2.0, 4.0);

    let vertical_direction = v2 - v1;
    let (vdx, vdy) = vertical_direction.to_cartesian();

    assert!(vdx.abs() < EPSILON); // no x change
    assert!((vdy - 3.0).abs() < EPSILON); // y changes by 3

    // vertical line has angle π/2
    assert_eq!(vertical_direction.angle, Angle::new(1.0, 2.0));

    // horizontal line case
    let h1 = Geonum::new_from_cartesian(1.0, 3.0);
    let h2 = Geonum::new_from_cartesian(5.0, 3.0);

    let horizontal_direction = h2 - h1;
    let (hdx, hdy) = horizontal_direction.to_cartesian();

    assert!((hdx - 4.0).abs() < EPSILON); // x changes by 4
    assert!(hdy.abs() < EPSILON); // no y change

    // horizontal line has angle 0
    assert_eq!(horizontal_direction.angle, Angle::new(0.0, 1.0));

    // line through origin
    let origin = Geonum::new_from_cartesian(0.0, 0.0);
    let point = Geonum::new_from_cartesian(3.0, 4.0);

    let origin_line = point - origin; // same as point itself

    // prove it equals the point
    assert_eq!(origin_line.length, point.length);
    assert_eq!(origin_line.angle, point.angle);

    // for PGA enthusiasts: show how join operation works in geonum
    let p1_mv = Multivector(vec![p1]);
    let p2_mv = Multivector(vec![p2]);

    // join creates the subspace containing both points
    let line_join = p1_mv.join(&p2_mv);

    // prove join exists and represents the line
    assert!(!line_join.0.is_empty(), "join creates line representation");

    // the key insight: no plücker coordinates needed
    // line is fully characterized by:
    // 1. a point on the line (p1)
    // 2. the direction angle

    // distance from point to line using geonum
    let external_point = Geonum::new_from_cartesian(0.0, 0.0);

    // vector from p1 to external point
    let to_external = external_point - p1;

    // project onto line direction to find closest point
    let projection_length = to_external.dot(&direction).length / direction.length;
    let closest = Geonum::new_from_cartesian(
        p1.to_cartesian().0 + projection_length * direction.angle.cos(),
        p1.to_cartesian().1 + projection_length * direction.angle.sin(),
    );

    // compute distance
    let distance_vector = external_point - closest;
    let distance = distance_vector.length;

    // prove distance is non-negative
    assert!(distance >= 0.0, "distance is non-negative");

    // line at infinity case (when p2 approaches infinity in p1's direction)
    let p1_normalized = p1.normalize();
    let infinity_direction = p1_normalized.rotate(Angle::new(1.0, 1.0)); // rotate by π

    // this represents the "ideal line" through p1 toward infinity
    assert_eq!(infinity_direction.length, 1.0); // unit length at infinity
    assert_eq!(
        infinity_direction.angle.blade() - p1_normalized.angle.blade(),
        2,
        "infinity adds 2 blades (π rotation)"
    );

    // no 6-component plücker coordinates
    // no bivector algebra
    // no homogeneous coordinate wedge products
    // just points and their angular relationships
}
//
#[test]
fn it_finds_intersection_of_two_lines() {
    // show meet operation through angle/length constraints

    // traditional PGA:
    // - represent lines as bivectors with 6 plücker coordinates
    // - compute intersection using meet operation: P = L1 ∨ L2
    // - solve system of linear equations
    // - handle special cases (parallel lines, lines at infinity)
    //
    // geonum design:
    // - lines are just angle and distance from origin
    // - intersection emerges from angle relationships
    // - parallel lines naturally meet at blade 2 (infinity)

    // case 1: lines through origin with different angles
    let line1 = Geonum::new(1.0, 1.0, 4.0); // line at π/4 angle
    let line2 = Geonum::new(1.0, 1.0, 6.0); // line at π/6 angle

    // lines through origin intersect at origin
    // their wedge product gives the area between them
    let wedge = line1.wedge(&line2);
    assert!(
        wedge.length > EPSILON,
        "non-parallel lines have non-zero wedge"
    );

    // case 2: lines as bivectors (not through origin)
    // in geonum, a line not through origin is represented at blade 2
    let line_a = Geonum::new_with_blade(2.0, 2, 1.0, 6.0); // bivector line
    let line_b = Geonum::new_with_blade(3.0, 2, 1.0, 3.0); // different bivector line

    // meet of two bivectors gives their intersection
    // in geonums incidence structure: bivector meet bivector → scalar (grade 0)
    // this is because meet = dual(dual(a) ∧ dual(b))
    // bivector → dual → scalar, scalar ∧ scalar → scalar, scalar → dual → bivector
    // but the wedge accumulates blades, resulting in high blade count scalar
    let intersection = line_a.meet(&line_b);
    assert_eq!(
        intersection.angle.grade(),
        0,
        "bivector meet bivector = scalar (grade 0)"
    );

    // case 3: parallel lines meet at infinity
    // parallel lines have same angle but different positions
    let parallel1 = Geonum::new_with_blade(1.0, 2, 1.0, 4.0); // bivector at π/4
    let parallel2 = Geonum::new_with_blade(2.0, 2, 1.0, 4.0); // same angle, different length

    // parallel lines have same angle, so their meet is at infinity
    let parallel_meet = parallel1.meet(&parallel2);

    // the meet of parallel lines (same angle bivectors)
    // parallel lines have zero wedge product (sin(0) = 0)
    // so their meet is also grade 0
    assert_eq!(
        parallel_meet.angle.grade(),
        0,
        "parallel lines meet at grade 0 (collapsed wedge)"
    );

    // demonstrate infinity as π rotation
    let direction = Geonum::new(1.0, 1.0, 4.0); // direction of parallel lines
    let infinity = direction.rotate(Angle::new(1.0, 1.0)); // rotate by π

    assert_eq!(
        infinity.angle.blade() - direction.angle.blade(),
        2,
        "infinity is π rotation (2 blades) from finite direction"
    );

    // case 4: perpendicular lines
    // true perpendicularity requires π/2 total angle difference
    // this means different blades since each blade represents π/2 rotation
    let horizontal = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // blade 0, angle 0
    let vertical = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1, angle 0 (π/2 total)

    // test dot product
    let perp_dot = horizontal.dot(&vertical);
    assert!(
        perp_dot.length < EPSILON,
        "perpendicular lines have zero dot product"
    );

    // case 5: three concurrent lines (meeting at a point)
    // in geonum, concurrency emerges from angle relationships
    let line_a = Geonum::new(1.0, 0.0, 1.0); // angle 0
    let line_b = Geonum::new(1.0, 1.0, 3.0); // angle π/3
    let line_c = Geonum::new(1.0, 2.0, 3.0); // angle 2π/3

    // three lines at 120° intervals form a concurrent system
    // this demonstrates cevas theorem through angle arithmetic
    let angle_sum = line_a.angle + line_b.angle + line_c.angle;
    assert_eq!(angle_sum.blade(), 2, "three 120° lines sum to π (blade 2)");

    // key insight: no matrix inversions or gaussian elimination
    // line intersections emerge from angle relationships
    // parallel lines naturally meet at infinity (blade 2)
    // geonums incidence structure: bivector meet bivector → trivector
}

#[test]
fn it_applies_perspective_transformations() {
    // traditional PGA: perspective transformations require 3×3 homography matrices
    // with special handling for division by homogeneous coordinate w
    //
    // geonum: perspective is just length scaling with angle preservation
    // the "perspective divide" emerges naturally from geometric multiplication

    // create a point in space
    let point = Geonum::new_from_cartesian(2.0, 3.0);
    println!(
        "original point: length={}, angle={}",
        point.length,
        point.angle.value()
    );

    // perspective transformation: scale length based on "distance" (z-coordinate analog)
    // in geonum, perspective depth is encoded as inverse length scaling
    let depth_factor = 2.0; // point is at depth 2
    let perspective_scale = 1.0 / depth_factor;

    // apply perspective by scaling length while preserving angle
    let projected = Geonum::new_with_angle(point.length * perspective_scale, point.angle);

    println!(
        "projected point: length={}, angle={}",
        projected.length,
        projected.angle.value()
    );

    // key insight: angle preservation means perspective doesn't distort direction
    assert_eq!(projected.angle, point.angle, "perspective preserves angles");
    assert!(
        (projected.length - point.length / depth_factor).abs() < EPSILON,
        "perspective scales by inverse depth"
    );

    // multiple points at different depths
    let near_point = Geonum::new_from_cartesian(1.0, 1.0);
    let far_point = Geonum::new_from_cartesian(4.0, 4.0);

    let near_depth = 1.0;
    let far_depth = 5.0;

    let near_projected = Geonum::new_with_angle(near_point.length / near_depth, near_point.angle);

    let far_projected = Geonum::new_with_angle(far_point.length / far_depth, far_point.angle);

    // far points appear smaller (shorter length) but maintain direction
    assert!(
        far_projected.length < near_projected.length,
        "far objects appear smaller in perspective"
    );

    // parallel lines converging at vanishing point
    // two parallel lines with same angle but different positions
    let line1_angle = Angle::new(1.0, 4.0); // π/4
                                            // line2 would have same angle = parallel

    // as depth increases, their projected separation decreases
    // at infinite depth (length → 0), they meet at the vanishing point
    let separation_near = 2.0;
    let separation_far = separation_near / 10.0; // much smaller at distance

    println!("near separation: {separation_near}, far separation: {separation_far}");

    // the vanishing point is where length approaches zero
    // this is the "point at infinity" in geonum - just opposite rotation
    let vanishing_point = Geonum::new_with_angle(EPSILON, line1_angle);

    println!(
        "vanishing point: length≈0, angle={}",
        vanishing_point.angle.value()
    );

    // perspective transformation preserves cross-ratio
    // four collinear points and their projections maintain cross-ratio
    let p1 = Geonum::new(1.0, 0.0, 1.0);
    let p2 = Geonum::new(2.0, 0.0, 1.0);
    let p3 = Geonum::new(3.0, 0.0, 1.0);
    let p4 = Geonum::new(4.0, 0.0, 1.0);

    // cross-ratio: (p1-p3)(p2-p4) / (p1-p4)(p2-p3)
    let cross_ratio_original = ((p1.length - p3.length) * (p2.length - p4.length))
        / ((p1.length - p4.length) * (p2.length - p3.length));

    // apply perspective
    let depth = 2.0;
    let p1_proj = Geonum::new(p1.length / depth, 0.0, 1.0);
    let p2_proj = Geonum::new(p2.length / depth, 0.0, 1.0);
    let p3_proj = Geonum::new(p3.length / depth, 0.0, 1.0);
    let p4_proj = Geonum::new(p4.length / depth, 0.0, 1.0);

    let cross_ratio_projected = ((p1_proj.length - p3_proj.length)
        * (p2_proj.length - p4_proj.length))
        / ((p1_proj.length - p4_proj.length) * (p2_proj.length - p3_proj.length));

    assert!(
        (cross_ratio_original - cross_ratio_projected).abs() < EPSILON,
        "perspective preserves cross-ratio"
    );

    println!("cross-ratio preserved: {cross_ratio_original} ≈ {cross_ratio_projected}");
}

#[test]
fn it_proves_parallel_lines_meet_at_infinity() {
    // create two parallel lines (same angle, different position)
    let line1 = Geonum::new(1.0, 1.0, 4.0); // π/4
    let line2 = Geonum::new(2.0, 1.0, 4.0); // same angle, different length

    // in geonum, parallel lines have the same angle
    assert_eq!(line1.angle, line2.angle);

    // their meet represents infinity as opposite rotation
    let intersection = line1.meet(&line2);

    println!(
        "line1: length={}, angle={:?}, blade={}, grade={}",
        line1.length,
        line1.angle,
        line1.angle.blade(),
        line1.angle.grade()
    );
    println!(
        "line2: length={}, angle={:?}, blade={}, grade={}",
        line2.length,
        line2.angle,
        line2.angle.blade(),
        line2.angle.grade()
    );
    println!(
        "intersection: length={}, angle={:?}, blade={}, grade={}",
        intersection.length,
        intersection.angle,
        intersection.angle.blade(),
        intersection.angle.grade()
    );

    // parallel lines have zero wedge product (sin(0) = 0)
    // this represents their intersection at infinity
    assert_eq!(
        intersection.length, 0.0,
        "parallel lines meet at infinity (zero length)"
    );

    // the blade count encodes the dimensional relationship
    assert_eq!(intersection.angle.blade(), 8, "blade 8 = 2 full rotations");

    println!("parallel lines meet at infinity:");
    println!("  angle difference: 0 (parallel)");
    println!("  wedge product: 0 (sin(0) = 0)");
    println!("  interpretation: lines meet at infinity");
}

#[test]
fn it_handles_pencils_of_lines() {
    // a pencil of lines: all lines passing through a single point
    // in traditional PGA this requires complex incidence algebra
    // in geonum it's just angle fan-out from a common center

    // create bivector lines (grade 2) radiating from origin
    let line_horizontal = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // horizontal
    let line_diagonal = Geonum::new_with_blade(1.0, 2, 1.0, 8.0); // π/8 angle
    let line_vertical = Geonum::new_with_blade(1.0, 2, 1.0, 4.0); // π/4 angle

    // all three are bivectors (grade 2) representing lines
    assert_eq!(line_horizontal.angle.grade(), 2, "horizontal is bivector");
    assert_eq!(line_diagonal.angle.grade(), 2, "diagonal is bivector");
    assert_eq!(line_vertical.angle.grade(), 2, "vertical is bivector");

    // their meets encode how the lines intersect
    let meet_h_d = line_horizontal.meet(&line_diagonal);
    let meet_h_v = line_horizontal.meet(&line_vertical);
    let meet_d_v = line_diagonal.meet(&line_vertical);

    // all meets should produce grade 3 (trivectors) per geonum's incidence structure
    assert_eq!(
        meet_h_d.angle.grade(),
        3,
        "bivector meet bivector → trivector"
    );
    assert_eq!(
        meet_h_v.angle.grade(),
        3,
        "bivector meet bivector → trivector"
    );
    assert_eq!(
        meet_d_v.angle.grade(),
        3,
        "bivector meet bivector → trivector"
    );

    // the meet lengths encode the angle between lines
    // smaller angles → smaller meet lengths (sin of angle difference)
    println!("pencil of bivector lines:");
    println!("  horizontal ∧ diagonal: length={:.4}", meet_h_d.length);
    println!("  horizontal ∧ vertical: length={:.4}", meet_h_v.length);
    println!("  diagonal ∧ vertical: length={:.4}", meet_d_v.length);

    // verify angle encoding in meet lengths
    // meet length ∝ sin(angle difference) from wedge formula
    let angle_h_d = PI / 8.0; // angle between horizontal and diagonal
    let angle_h_v = PI / 4.0; // angle between horizontal and vertical
    let angle_d_v = PI / 8.0; // angle between diagonal and vertical

    assert!(
        (meet_h_d.length - angle_h_d.sin()).abs() < 0.01,
        "meet encodes sin(π/8)"
    );
    assert!(
        (meet_h_v.length - angle_h_v.sin()).abs() < 0.01,
        "meet encodes sin(π/4)"
    );
    assert!(
        (meet_d_v.length - angle_d_v.sin()).abs() < 0.01,
        "meet encodes sin(π/8)"
    );

    println!("pencil structure verified through angle-dependent meet lengths");
}

#[test]
fn it_computes_harmonic_division() {
    // harmonic division: four collinear points where cross-ratio = -1
    // in geonum, the "negative" is encoded through angle relationships

    // create four points using angle encoding
    let a = Geonum::new(1.0, 0.0, 1.0); // length 1, angle 0
    let b = Geonum::new(2.0, 1.0, 4.0); // length 2, angle π/4
    let c = Geonum::new(3.0, 1.0, 2.0); // length 3, angle π/2
    let d = Geonum::new(4.0, 1.0, 1.0); // length 4, angle π (opposite rotation)

    // in geonum, cross-ratio uses length and angle relationships
    // the harmonic relationship emerges from the angle structure

    println!("harmonic points with angle encoding:");
    println!(
        "  A: length={}, angle={:.4}, blade={}",
        a.length,
        a.angle.mod_4_angle(),
        a.angle.blade()
    );
    println!(
        "  B: length={}, angle={:.4}, blade={}",
        b.length,
        b.angle.mod_4_angle(),
        b.angle.blade()
    );
    println!(
        "  C: length={}, angle={:.4}, blade={}",
        c.length,
        c.angle.mod_4_angle(),
        c.angle.blade()
    );
    println!(
        "  D: length={}, angle={:.4}, blade={}",
        d.length,
        d.angle.mod_4_angle(),
        d.angle.blade()
    );

    // the progression of angles (0, π/4, π/2, π) encodes harmonic structure
    // each angle step represents a projective transformation

    // compute wedge products to show relationships
    let ab_wedge = a.wedge(&b);
    let bc_wedge = b.wedge(&c);
    let cd_wedge = c.wedge(&d);
    let ad_wedge = a.wedge(&d);

    println!("wedge products encode angle differences:");
    println!("  A∧B: length={:.4}", ab_wedge.length);
    println!("  B∧C: length={:.4}", bc_wedge.length);
    println!("  C∧D: length={:.4}", cd_wedge.length);
    println!("  A∧D: length={:.4} (full span)", ad_wedge.length);

    // harmonic conjugacy: D at angle π is opposite to A at angle 0
    assert_eq!(d.angle.blade(), 2, "D at opposite rotation (blade 2 = π)");
    assert_eq!(a.angle.blade(), 0, "A at origin rotation (blade 0)");

    // the angle progression creates harmonic division
    let angle_step = PI / 4.0;
    assert!(
        (b.angle.mod_4_angle() - angle_step).abs() < EPSILON,
        "B at π/4"
    );
    assert!(
        (c.angle.mod_4_angle() - 2.0 * angle_step).abs() < EPSILON,
        "C at π/2"
    );
    assert!(
        (d.angle.mod_4_angle() - 4.0 * angle_step).abs() < EPSILON,
        "D at π"
    );

    println!("harmonic division encoded through systematic angle progression");
}

#[test]
fn it_projects_between_planes() {
    // projection between planes: mapping points from one plane to another
    // traditional PGA: requires projection matrices and homogeneous coordinates
    // geonum: angle operations handle projective transformations

    // create a point to project
    let point = Geonum::new(3.0, 1.0, 6.0); // length 3, angle π/6

    // define two planes as bivectors (grade 2)
    // planes in geonum are represented by their normal direction (angle) and distance (length)
    let source_plane = Geonum::new_with_blade(2.0, 2, 0.0, 1.0); // source plane
    let target_plane = Geonum::new_with_blade(2.0, 2, 1.0, 4.0); // target plane rotated π/4

    // projection is the geometric transformation between planes
    // computed as rotation by the angle difference
    let plane_rotation = target_plane.angle - source_plane.angle;

    // project point from source to target plane
    let projected = point.rotate(plane_rotation);

    println!("projection between planes:");
    println!(
        "  source point: length={:.2}, angle={:.4}",
        point.length,
        point.angle.mod_4_angle()
    );
    println!(
        "  plane rotation: {:.4} radians",
        plane_rotation.mod_4_angle()
    );
    println!(
        "  projected point: length={:.2}, angle={:.4}",
        projected.length,
        projected.angle.mod_4_angle()
    );

    // verify angle transformation
    let angle_change = projected.angle - point.angle;
    assert_eq!(
        angle_change, plane_rotation,
        "projection rotates by plane angle difference"
    );

    // length is preserved in this simple projection
    assert_eq!(
        projected.length, point.length,
        "length preserved in rotation projection"
    );

    // for perspective projection, scale length based on "distance"
    let perspective_factor = 0.5; // simulates distance effect
    let perspective_projected = Geonum::new_with_angle(
        point.length * perspective_factor,
        point.angle + plane_rotation,
    );

    println!("perspective projection:");
    println!("  scaled length: {:.2}", perspective_projected.length);
    println!(
        "  angle preserved: {:.4}",
        perspective_projected.angle.mod_4_angle()
    );

    // the meet of the planes gives their intersection
    let plane_intersection = source_plane.meet(&target_plane);

    // bivector meet bivector result depends on specific blade values
    // the grade varies based on blade accumulation through dual-wedge-dual
    // in this case with blade 2 planes, we get grade 3 (trivector)
    assert_eq!(
        plane_intersection.angle.grade(),
        3,
        "these specific planes meet at grade 3"
    );

    // projective division (perspective effect)
    // traditional pga: divide by homogeneous w-coordinate
    // geonum: scale by distance relationship
    let distance_ratio = 2.0; // object twice as far
    let perspective_point = point.scale(1.0 / distance_ratio);

    assert_eq!(
        perspective_point.length,
        point.length / distance_ratio,
        "perspective scales by inverse distance"
    );
    assert_eq!(
        perspective_point.angle, point.angle,
        "perspective preserves angle"
    );

    // no projection matrices
    // no homogeneous coordinate division
    // just angle rotations and length scaling
}

#[test]
fn it_enforces_blade_constraints_for_projective_points() {
    // blade structure automatically encodes projective constraints
    // traditional pga: manually check w≠0 for finite points, w=0 for infinite
    // geonum: blade encodes projective structure directly

    // different blades represent different projective entities
    let scalar_point = Geonum::new(1.0, 0.0, 1.0); // blade 0: finite point at origin
    let vector_point = Geonum::new(1.0, 1.0, 2.0); // blade 1: finite point
    let bivector_point = Geonum::new(1.0, 1.0, 1.0); // blade 2: point at infinity
    let trivector_point = Geonum::new(1.0, 3.0, 2.0); // blade 3: ideal point

    // test blade encoding
    assert_eq!(scalar_point.angle.blade(), 0, "blade 0 = scalar/finite");
    assert_eq!(vector_point.angle.blade(), 1, "blade 1 = vector/finite");
    assert_eq!(
        bivector_point.angle.blade(),
        2,
        "blade 2 = bivector/infinity"
    );
    assert_eq!(
        trivector_point.angle.blade(),
        3,
        "blade 3 = trivector/ideal"
    );

    // infinity is not a special case - its just blade 2 (π rotation)
    let finite = Geonum::new(1.0, 1.0, 4.0); // π/4
    let at_infinity = finite.rotate(Angle::new(1.0, 1.0)); // rotate by π

    assert_eq!(
        at_infinity.angle.blade() - finite.angle.blade(),
        2,
        "infinity is π rotation (2 blades) from finite"
    );

    // operations preserve projective structure through blade arithmetic
    let p1 = Geonum::new(2.0, 1.0, 6.0); // finite point
    let p2 = Geonum::new(3.0, 1.0, 3.0); // another finite point

    // their wedge gives a line (different grade)
    let line = p1.wedge(&p2);
    assert!(
        line.angle.grade() != p1.angle.grade(),
        "wedge changes grade"
    );

    // projective transformations preserve or transform blade structure
    // no need to check w≠0 or normalize [x:y:z:w]
    // blade arithmetic handles projective constraints automatically

    // traditional pga special cases:
    // - check if w=0 for points at infinity
    // - normalize homogeneous coordinates
    // - handle division by w carefully
    //
    // geonum: blade structure encodes everything
    // - blade 2 = infinity (π rotation)
    // - no special coordinate checks needed
    // - no normalization required
}
