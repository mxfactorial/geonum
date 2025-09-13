use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

/// demonstrate when GeoCollection as a collection of distinct geometric entities is useful
///
/// demos scenarios where we need to work with multiple separate geometric objects
/// rather than trying to encode everything into a single Geonum

#[test]
fn it_tracks_multiple_light_rays_intersection() {
    // scenario: light rays from different sources converging at a focal point
    // each ray is a distinct geometric entity with its own direction and intensity

    let light_rays = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // ray from east (0°)
        Geonum::new(0.8, 1.0, 3.0), // ray from northeast (60°)
        Geonum::new(1.2, 1.0, 2.0), // ray from north (90°)
        Geonum::new(0.9, 4.0, 3.0), // ray from northwest (240°)
    ]);

    // find which rays actually converge (meet at a common point)
    let mut convergent_pairs = Vec::new();

    for i in 0..light_rays.len() {
        for j in (i + 1)..light_rays.len() {
            let intersection = light_rays[i].meet(&light_rays[j]);

            // if intersection has significant magnitude, rays converge
            if intersection.length > EPSILON {
                convergent_pairs.push((i, j, intersection));
            }
        }
    }

    // prove some convergent ray pairs
    assert!(!convergent_pairs.is_empty(), "some light rays converge");

    // Test that each convergence point is well-defined
    for (i, j, intersection) in convergent_pairs {
        assert!(
            intersection.length.is_finite(),
            "ray {} and {} convergence point is finite",
            i,
            j
        );
        assert!(intersection.length > 0.0, "positive magnitude convergence");
    }
}

#[test]
fn it_manages_building_wireframe_edges() {
    // scenario: architectural wireframe with multiple structural edges
    // each edge is a distinct line segment that we need to process individually

    let building_edges = GeoCollection::from(vec![
        Geonum::new(10.0, 0.0, 1.0), // ground edge (horizontal)
        Geonum::new(8.0, 1.0, 2.0),  // vertical support beam
        Geonum::new(12.0, 1.0, 4.0), // diagonal brace (45°)
        Geonum::new(6.0, 1.0, 3.0),  // roof edge (60°)
        Geonum::new(9.0, 3.0, 4.0),  // cross brace (135°)
    ]);

    // calculate total structural length
    let total_length: f64 = building_edges.iter().map(|edge| edge.length).sum();
    assert!(
        total_length > 40.0,
        "substantial total edge length of building"
    );

    // find critical joints (where multiple edges meet)
    let mut joints = Vec::new();

    for i in 0..building_edges.len() {
        for j in (i + 1)..building_edges.len() {
            let joint = building_edges[i].meet(&building_edges[j]);

            // critical joints have significant intersection magnitude
            if joint.length > 0.1 {
                joints.push(joint);
            }
        }
    }

    // structural integrity requires multiple joints
    assert!(joints.len() >= 3, "multiple structural joints in building");

    // Each joint should be a well-defined point
    for joint in joints {
        assert!(joint.length.is_finite());
        assert!(!joint.length.is_nan());
    }
}

#[test]
fn it_handles_scanning_laser_beams() {
    // scenario: LIDAR system with multiple scanning laser beams
    // each beam scans different angular ranges and we need to track them separately

    let laser_beams = GeoCollection::from(vec![
        Geonum::new(100.0, 0.0, 1.0), // primary scanning beam (0°)
        Geonum::new(95.0, 1.0, 6.0),  // secondary beam (30°)
        Geonum::new(98.0, 1.0, 4.0),  // tertiary beam (45°)
        Geonum::new(92.0, 1.0, 3.0),  // quaternary beam (60°)
        Geonum::new(105.0, 1.0, 2.0), // vertical beam (90°)
    ]);

    // compute angular coverage
    let mut angles: Vec<f64> = laser_beams
        .iter()
        .map(|beam| beam.angle.mod_4_angle())
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Test we have good angular distribution
    let angular_span = angles.last().unwrap() - angles.first().unwrap();
    assert!(
        angular_span > PI / 3.0,
        "LIDAR should have substantial angular coverage"
    );

    // Find beam intersections (potential interference patterns)
    let mut interference_points = Vec::new();

    for i in 0..laser_beams.len() {
        for j in (i + 1)..laser_beams.len() {
            let interference = laser_beams[i].meet(&laser_beams[j]);
            if interference.length > 1.0 {
                // Significant interference
                interference_points.push((i, j, interference.length));
            }
        }
    }

    // Some beam pairs should create interference
    assert!(
        !interference_points.is_empty(),
        "LIDAR beams should show some interference"
    );

    // Interference should be measurable
    for (i, j, intensity) in interference_points {
        assert!(
            intensity > 0.0,
            "Interference between beams {} and {} should have positive intensity",
            i,
            j
        );
        // beam intensities are ~100, so interference should be bounded
        // meet squares the lengths: 100 * 100 = 10000
        assert!(
            intensity < 20000.0,
            "Interference should not exceed twice the product of beam intensities"
        );
    }
}

#[test]
fn it_models_planetary_orbit_system() {
    // Scenario: Multiple planetary orbits around a star
    // Each orbit is a distinct geometric object with its own elliptical parameters

    let planetary_orbits = GeoCollection::from(vec![
        Geonum::new(0.39, 0.0, 1.0), // Mercury orbit (closest)
        Geonum::new(0.72, 1.0, 8.0), // Venus orbit
        Geonum::new(1.00, 1.0, 4.0), // Earth orbit (45°)
        Geonum::new(1.52, 1.0, 3.0), // Mars orbit (60°)
        Geonum::new(5.20, 1.0, 6.0), // Jupiter orbit (30°)
    ]);

    // Calculate orbital resonances (where orbits have mathematical relationships)
    let mut resonances = Vec::new();

    for i in 0..planetary_orbits.len() {
        for j in (i + 1)..planetary_orbits.len() {
            let orbit1 = &planetary_orbits[i];
            let orbit2 = &planetary_orbits[j];

            // Simple resonance check: ratio of orbital radii
            let radius_ratio = orbit2.length / orbit1.length;

            // Look for simple integer ratios (2:1, 3:2, etc.)
            let closest_ratio = radius_ratio.round();
            if (radius_ratio - closest_ratio).abs() < 0.2 {
                resonances.push((i, j, closest_ratio));
            }
        }
    }

    // Some planetary pairs should show orbital resonances
    assert!(
        !resonances.is_empty(),
        "Planetary system should show some orbital resonances"
    );

    // Test that orbital intersections represent meaningful geometric relationships
    for i in 0..planetary_orbits.len() {
        for j in (i + 1)..planetary_orbits.len() {
            let intersection = planetary_orbits[i].meet(&planetary_orbits[j]);

            // Orbital intersection points should be well-defined
            assert!(intersection.length.is_finite());

            // Different orbits should have non-zero intersection (crossing points)
            if planetary_orbits[i].length != planetary_orbits[j].length {
                assert!(intersection.length >= 0.0);
            }
        }
    }
}

#[test]
fn it_represents_electromagnetic_field_lines() {
    // Scenario: Multiple electromagnetic field lines around charged particles
    // Each field line is a distinct geometric curve that we track separately

    let field_lines = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // Radial field line (0°)
        Geonum::new(1.0, 1.0, 4.0), // Field line at 45°
        Geonum::new(1.0, 1.0, 2.0), // Field line at 90°
        Geonum::new(1.0, 3.0, 4.0), // Field line at 135°
        Geonum::new(1.0, 2.0, 2.0), // Field line at 180°
        Geonum::new(1.0, 5.0, 4.0), // Field line at 225°
    ]);

    // Calculate field line density (how closely packed they are)
    let mut angular_separations = Vec::new();

    for i in 0..field_lines.len() {
        for j in (i + 1)..field_lines.len() {
            let angle1 = field_lines[i].angle.mod_4_angle();
            let angle2 = field_lines[j].angle.mod_4_angle();
            let separation = (angle2 - angle1).abs();
            angular_separations.push(separation);
        }
    }

    // Field lines should have reasonable angular distribution
    let avg_separation = angular_separations.iter().sum::<f64>() / angular_separations.len() as f64;
    assert!(
        avg_separation > 0.1,
        "Field lines should not be too tightly packed"
    );
    assert!(avg_separation < PI, "Field lines should not be too sparse");

    // Test field line interactions (where field lines converge or diverge)
    let mut convergence_points = Vec::new();

    for i in 0..field_lines.len() {
        for j in (i + 1)..field_lines.len() {
            let convergence = field_lines[i].meet(&field_lines[j]);

            // Field lines from the same source should converge
            if convergence.length > EPSILON {
                convergence_points.push(convergence);
            }
        }
    }

    // Electromagnetic field should show convergence behavior
    assert!(
        !convergence_points.is_empty(),
        "Field lines should show convergence patterns"
    );

    // Each convergence point should represent a physical location
    for point in convergence_points {
        assert!(point.length.is_finite());
        assert!(point.length >= 0.0);
    }
}

#[test]
fn it_demonstrates_why_single_geonum_fails_here() {
    // Demonstrate why trying to encode multiple distinct objects into one Geonum loses information

    // Attempt to combine multiple laser beams into a single Geonum
    let beam1 = Geonum::new(100.0, 0.0, 1.0); // East beam
    let beam2 = Geonum::new(95.0, 1.0, 2.0); // North beam
    let beam3 = Geonum::new(98.0, 1.0, 4.0); // Northeast beam

    // Traditional approach: try to combine via multiplication (angle addition)
    let combined_wrong = beam1 * beam2 * beam3;

    // Result: angles have been added together
    // Original: 0°, 90°, 45°
    // Combined: 0° + 90° + 45° = 135°
    assert_eq!(combined_wrong.angle.mod_4_angle(), 3.0 * PI / 4.0);

    // We've lost the fact that these were three separate beams!
    // The combined result suggests one beam at 135°, not three beams at different angles

    // Information loss demonstration:
    // - Can't recover individual beam directions
    // - Can't compute pairwise intersections
    // - Can't analyze interference patterns
    // - Can't track separate beam intensities

    // This proves that for genuinely distinct geometric entities,
    // maintaining them as separate Geonum objects preserves essential information
    // that would be lost in a single combined representation

    let separate_beams = GeoCollection::from(vec![beam1, beam2, beam3]);

    // With separate representation, we can:
    // 1. Query individual beam properties
    assert_eq!(separate_beams[0].angle.mod_4_angle(), 0.0);
    assert_eq!(separate_beams[1].angle.mod_4_angle(), PI / 2.0);
    assert_eq!(separate_beams[2].angle.mod_4_angle(), PI / 4.0);

    // 2. Compute pairwise interactions
    let beam1_beam2_meet = separate_beams[0].meet(&separate_beams[1]);
    let beam1_beam3_meet = separate_beams[0].meet(&separate_beams[2]);
    let beam2_beam3_meet = separate_beams[1].meet(&separate_beams[2]);

    // Each intersection is well-defined and distinct
    assert!(beam1_beam2_meet.length.is_finite());
    assert!(beam1_beam3_meet.length.is_finite());
    assert!(beam2_beam3_meet.length.is_finite());

    // 3. Maintain individual beam intensities
    let total_power: f64 = separate_beams.iter().map(|beam| beam.length).sum();
    assert!((total_power - 293.0).abs() < EPSILON); // 100 + 95 + 98 = 293

    // This demonstrates the clear utility of GeoCollection as a collection
    // when dealing with multiple distinct geometric entities
}
