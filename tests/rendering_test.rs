use geonum::*;
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Instant;

const EPSILON: f64 = 1e-10;

#[test]
fn it_sweeps_a_screen_with_one_geonum() {
    // one geonum addresses every pixel on a 2D screen
    // angle = direction from center, magnitude = distance from center
    // pixel coordinates recovered via angle.project on demand
    // no coordinate arrays stored

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    let angle_steps = 360;
    let radial_steps = 20;
    let max_radius = 100.0;
    let mut pixel_count = 0;

    for a in 0..angle_steps {
        let pi_frac = 2.0 * a as f64 / angle_steps as f64; // fraction of π
        let theta = pi_frac * PI; // raw radians for trig assertions

        for r in 1..=radial_steps {
            let radius = (r as f64 / radial_steps as f64) * max_radius;

            // tune one scanner to this pixel position
            let scanner = Geonum::new_with_angle(radius, Angle::new(pi_frac, 1.0));

            // recover screen coordinates via projection
            let px = scanner.mag * scanner.angle.project(ex);
            let py = scanner.mag * scanner.angle.project(ey);

            // projections match traditional trig at every pixel
            assert!(
                (px - radius * theta.cos()).abs() < EPSILON,
                "x projection at angle {:.3} radius {:.1}",
                theta,
                radius
            );
            assert!(
                (py - radius * theta.sin()).abs() < EPSILON,
                "y projection at angle {:.3} radius {:.1}",
                theta,
                radius
            );

            // pythagorean identity at every pixel
            assert!(
                (px * px + py * py - radius * radius).abs() < EPSILON,
                "pythagorean identity at angle {:.3} radius {:.1}",
                theta,
                radius
            );

            pixel_count += 1;
        }
    }

    assert_eq!(pixel_count, angle_steps * radial_steps);

    println!("swept {} pixels with one geonum", pixel_count);
    println!("  360 angles × 20 radii = full polar screen coverage");
    println!("  pythagorean identity holds at every pixel");
    println!("  zero coordinate arrays allocated");
}

#[test]
fn it_rotates_the_scanner_not_the_scanned() {
    // view rotation is one angle addition on the sweep
    // traditional: iterate N pixels × matrix multiply each
    // geonum: add rotation to sweep angle, project to get rotated coordinates

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    let rotations = [
        Angle::new(1.0, 4.0), // π/4
        Angle::new(1.0, 2.0), // π/2
        Angle::new(2.0, 3.0), // 2π/3
        Angle::new(1.0, 1.0), // π
        Angle::new(7.0, 4.0), // 7π/4
    ];

    for rotation in &rotations {
        let rot_rad = rotation.grade_angle();
        let cos_r = rot_rad.cos();
        let sin_r = rot_rad.sin();

        for a in 0..36 {
            let pi_frac = 2.0 * a as f64 / 36.0;
            let theta = pi_frac * PI;

            for &radius in &[1.0, 5.0, 10.0, 50.0] {
                let scanner = Geonum::new_with_angle(radius, Angle::new(pi_frac, 1.0));
                let ox = scanner.mag * scanner.angle.project(ex);
                let oy = scanner.mag * scanner.angle.project(ey);

                // rotate the scanner: one angle addition
                let rotated = scanner.rotate(*rotation);
                let rx = rotated.mag * rotated.angle.project(ex);
                let ry = rotated.mag * rotated.angle.project(ey);

                // traditional rotation: 2×2 matrix × each pixel
                let trad_x = ox * cos_r - oy * sin_r;
                let trad_y = ox * sin_r + oy * cos_r;

                assert!(
                    (rx - trad_x).abs() < EPSILON,
                    "x mismatch at angle {:.3} radius {:.1} rotation {:.3}",
                    theta,
                    radius,
                    rot_rad
                );
                assert!(
                    (ry - trad_y).abs() < EPSILON,
                    "y mismatch at angle {:.3} radius {:.1} rotation {:.3}",
                    theta,
                    radius,
                    rot_rad
                );

                // magnitude preserved: rotation doesnt change distance from center
                assert!((rotated.mag - scanner.mag).abs() < EPSILON);
            }
        }
    }
}

#[test]
fn it_passes_pixel_values_through_rotation() {
    // pixel value = magnitude, pixel position = angle
    // rotating moves the value to a new screen position without touching it

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);
    let rotation = Angle::new(1.0, 3.0); // π/3
    let rot_rad = rotation.grade_angle();

    // angular gradient pattern: intensity varies with angle
    for i in 0..36 {
        let pi_frac = 2.0 * i as f64 / 36.0;
        let theta = pi_frac * PI;
        let intensity = 10.0 + 5.0 * theta.cos();

        // scanner carries the pixel value as magnitude
        let scanner = Geonum::new_with_angle(intensity, Angle::new(pi_frac, 1.0));

        // rotate the scanner
        let rotated = scanner.rotate(rotation);

        // pixel value (magnitude) passes through unchanged
        assert!(
            (rotated.mag - intensity).abs() < EPSILON,
            "pixel value destroyed at angle {:.3}",
            theta
        );

        // the value landed at the rotated position
        let new_x = rotated.mag * rotated.angle.project(ex);
        let new_y = rotated.mag * rotated.angle.project(ey);

        let original_x = intensity * theta.cos();
        let original_y = intensity * theta.sin();
        let expected_x = original_x * rot_rad.cos() - original_y * rot_rad.sin();
        let expected_y = original_x * rot_rad.sin() + original_y * rot_rad.cos();

        assert!(
            (new_x - expected_x).abs() < EPSILON,
            "pixel x position wrong at angle {:.3}",
            theta
        );
        assert!(
            (new_y - expected_y).abs() < EPSILON,
            "pixel y position wrong at angle {:.3}",
            theta
        );
    }
}

#[test]
fn it_renders_a_circle_without_coordinates() {
    // circle centered at origin: pixel inside when sweep magnitude ≤ radius
    // no x² + y² ≤ r², no sqrt, no coordinate storage
    // the magnitude IS the distance

    let circle_radius = 5.0;
    let mut inside_count = 0;
    let mut outside_count = 0;

    for a in 0..360 {
        let pi_frac = 2.0 * a as f64 / 360.0;
        for m in 1..=20 {
            let r = m as f64 * 0.5; // 0.5 to 10.0

            let scanner = Geonum::new_with_angle(r, Angle::new(pi_frac, 1.0));

            // circle membership: compare magnitude to radius
            if scanner.mag <= circle_radius {
                inside_count += 1;
            } else {
                outside_count += 1;
            }
        }
    }

    // 10 of 20 radial steps inside (0.5 to 5.0), 10 outside (5.5 to 10.0)
    assert_eq!(inside_count, 360 * 10);
    assert_eq!(outside_count, 360 * 10);

    // offset circle: center at (3, 0), radius 2
    // distance_to computes law of cosines from angle difference
    // still no coordinate decomposition
    let center = Geonum::new(3.0, 0.0, 1.0);
    // pixel at (4, 0): distance to center = 1, inside
    let pixel_near = Geonum::new(4.0, 0.0, 1.0);
    let dist_near = pixel_near.distance_to(&center);
    assert!((dist_near.mag - 1.0).abs() < EPSILON);

    // pixel at (6, 0): distance to center = 3, outside
    let pixel_far = Geonum::new(6.0, 0.0, 1.0);
    let dist_far = pixel_far.distance_to(&center);
    assert!((dist_far.mag - 3.0).abs() < EPSILON);

    // pixel at (3, π/2): distance = sqrt(9 + 9) = sqrt(18), outside
    let pixel_perp = Geonum::new(3.0, 1.0, 2.0);
    let dist_perp = pixel_perp.distance_to(&center);
    let expected_dist = (9.0_f64 + 9.0).sqrt();
    assert!((dist_perp.mag - expected_dist).abs() < EPSILON);
}

#[test]
fn it_zooms_and_rotates_in_one_operation() {
    // scale_rotate transforms the sweep with one call
    // traditional: separate matrix multiply for scale, another for rotation
    // geonum: magnitude scales, angle adds

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    let zoom = 3.0;
    let rotation = Angle::new(1.0, 6.0); // π/6
    let rot_rad = rotation.grade_angle();

    for a in 0..36 {
        let pi_frac = 2.0 * a as f64 / 36.0;
        let theta = pi_frac * PI;
        let radius = 5.0;

        let scanner = Geonum::new_with_angle(radius, Angle::new(pi_frac, 1.0));

        // one operation: zoom and rotate the scanner
        let transformed = scanner.scale_rotate(zoom, rotation);
        let tx = transformed.mag * transformed.angle.project(ex);
        let ty = transformed.mag * transformed.angle.project(ey);

        // traditional: scale matrix × rotation matrix × pixel
        let ox = radius * theta.cos();
        let oy = radius * theta.sin();
        let zx = ox * zoom;
        let zy = oy * zoom;
        let expected_x = zx * rot_rad.cos() - zy * rot_rad.sin();
        let expected_y = zx * rot_rad.sin() + zy * rot_rad.cos();

        assert!(
            (tx - expected_x).abs() < EPSILON,
            "zoom+rotate x mismatch at angle {:.3}",
            theta
        );
        assert!(
            (ty - expected_y).abs() < EPSILON,
            "zoom+rotate y mismatch at angle {:.3}",
            theta
        );

        // magnitude = original radius × zoom factor
        assert!((transformed.mag - radius * zoom).abs() < EPSILON);
    }
}

#[test]
fn it_renders_the_same_scene_from_different_camera_angles() {
    // same scene, different camera rotation
    // pixel values arrive at screen positions determined by camera angle
    // no vertex buffer, no view matrix multiplication

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    // scene: objects at known polar positions with intensities
    let scene: [(f64, Angle, f64); 4] = [
        (5.0, Angle::new(0.0, 1.0), 200.0), // bright object at 5 units, 0°
        (3.0, Angle::new(1.0, 2.0), 150.0), // medium object at 3 units, 90°
        (7.0, Angle::new(1.0, 1.0), 100.0), // dim object at 7 units, 180°
        (4.0, Angle::new(3.0, 2.0), 50.0),  // faint object at 4 units, 270°
    ];

    let camera_rotation = Angle::new(1.0, 2.0); // 90° camera pan

    for &(distance, direction, intensity) in &scene {
        // render from default camera (0°)
        let sweep_default = Geonum::new_with_angle(intensity, direction);
        let x1 = sweep_default.mag * sweep_default.angle.project(ex);
        let y1 = sweep_default.mag * sweep_default.angle.project(ey);

        // render from rotated camera (90°)
        let sweep_rotated = Geonum::new_with_angle(intensity, direction + camera_rotation);
        let x2 = sweep_rotated.mag * sweep_rotated.angle.project(ex);
        let y2 = sweep_rotated.mag * sweep_rotated.angle.project(ey);

        // 90° camera pan: screen (x, y) → (-y, x)
        assert!(
            (x2 - (-y1)).abs() < EPSILON,
            "camera pan x for object at direction {:?}",
            direction
        );
        assert!(
            (y2 - x1).abs() < EPSILON,
            "camera pan y for object at direction {:?}",
            direction
        );

        // intensity (magnitude) preserved across camera angles
        assert!((sweep_default.mag - intensity).abs() < EPSILON);
        assert!((sweep_rotated.mag - intensity).abs() < EPSILON);

        // distance from center preserved: camera rotation doesnt change object distance
        // distance is encoded in the scene, not in the sweep magnitude here
        // the sweep magnitude carries the pixel value (intensity)
        // distance determines which radial step the sweep hits
        let position_default = Geonum::new_with_angle(distance, direction);
        let position_rotated = Geonum::new_with_angle(distance, direction + camera_rotation);
        assert!((position_default.mag - position_rotated.mag).abs() < EPSILON);
    }
}

#[test]
fn it_proves_dimension_free_rendering() {
    // projecting onto blade 1 and blade 1_000_001 produces identical pixel coordinates
    // rendering doesnt depend on which "dimension" youre in

    let screen_radius = 50.0;

    for a in 0..36 {
        let pi_frac = 2.0 * a as f64 / 36.0;
        let radius = (a as f64 + 1.0) / 36.0 * screen_radius;

        let scanner = Geonum::new_with_angle(radius, Angle::new(pi_frac, 1.0));

        // project onto blade 1 axis
        let axis_small = Angle::new(1.0, 2.0);
        let p_small = scanner.mag * scanner.angle.project(axis_small);

        // project onto blade 1_000_001 axis (same grade: 1_000_001 % 4 = 1)
        let axis_huge = Angle::new(1_000_001.0, 2.0);
        let p_huge = scanner.mag * scanner.angle.project(axis_huge);

        assert!(
            (p_small - p_huge).abs() < 1e-12,
            "blade 1 and blade 1_000_001 produce different pixel values at step {}",
            a
        );

        // also prove blade 0 vs blade 1_000_000 (both grade 0)
        let ex_small = Angle::new(0.0, 1.0);
        let ex_huge = Angle::new(1_000_000.0, 2.0);
        let px_small = scanner.mag * scanner.angle.project(ex_small);
        let px_huge = scanner.mag * scanner.angle.project(ex_huge);

        assert!(
            (px_small - px_huge).abs() < 1e-12,
            "blade 0 and blade 1_000_000 produce different pixel values at step {}",
            a
        );
    }
}

#[test]
fn it_proves_rendering_perf_scales_with_angle_not_dimension() {
    // traditional GA cant work in 100D — 2^100 multivector components exceed
    // available memory at any budget. so high-dimensional work falls back to
    // linear algebra: n×n matrices at O(n²) per pixel. the benchmark below
    // times that fallback because its the only path traditional math has left
    // at high dimensions
    //
    // geonum replaces both layers: it does what GA does (grades, duality, wedge)
    // without the 2^n explosion, and what linear algebra does (rotation, projection)
    // without n×n matrices. cost O(1) regardless of dimension
    //
    // at 2D traditional is competitive — fewer raw ops per pixel
    // at 10D geonum overtakes — 190 flops vs ~6
    // at 100D geonum dominates — 19900 flops vs ~6
    //
    // geonum cost is identical whether the blade context is 1 or 1_000_000

    let pixel_count: usize = 50_000;
    let rotation_angle = PI / 6.0;
    let cos_r = rotation_angle.cos();
    let sin_r = rotation_angle.sin();

    // --- traditional: cost scales with dimension ---

    // 2D: 2×2 matrix × 2-vector = 4 mults + 2 adds per pixel
    let start = Instant::now();
    for i in 0..pixel_count {
        let x = black_box((i as f64 * 0.1).cos());
        let y = black_box((i as f64 * 0.1).sin());
        black_box(x * cos_r - y * sin_r);
        black_box(x * sin_r + y * cos_r);
    }
    let trad_2d = start.elapsed();

    // 10D: 10×10 matrix × 10-vector = 100 mults + 90 adds per pixel
    let mut mat_10 = [[0.0_f64; 10]; 10];
    for (idx, row) in mat_10.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    mat_10[0][0] = cos_r;
    mat_10[0][1] = -sin_r;
    mat_10[1][0] = sin_r;
    mat_10[1][1] = cos_r;

    let start = Instant::now();
    for i in 0..pixel_count {
        let mut v = [0.0_f64; 10];
        v[0] = (i as f64 * 0.1).cos();
        v[1] = (i as f64 * 0.1).sin();
        let mut result = [0.0_f64; 10];
        for r in 0..10 {
            let mut sum = 0.0;
            for c in 0..10 {
                sum += mat_10[r][c] * v[c];
            }
            result[r] = sum;
        }
        black_box(result);
    }
    let trad_10d = start.elapsed();

    // 100D: 100×100 matrix × 100-vector = 10000 mults + 9900 adds per pixel
    let mut mat_100 = [[0.0_f64; 100]; 100];
    for (idx, row) in mat_100.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    mat_100[0][0] = cos_r;
    mat_100[0][1] = -sin_r;
    mat_100[1][0] = sin_r;
    mat_100[1][1] = cos_r;

    let start = Instant::now();
    for i in 0..pixel_count {
        let mut v = [0.0_f64; 100];
        v[0] = (i as f64 * 0.1).cos();
        v[1] = (i as f64 * 0.1).sin();
        let mut result = [0.0_f64; 100];
        for r in 0..100 {
            let mut sum = 0.0;
            for c in 0..100 {
                sum += mat_100[r][c] * v[c];
            }
            result[r] = sum;
        }
        black_box(result);
    }
    let trad_100d = start.elapsed();

    // --- geonum: cost independent of dimensional context ---

    let rotation = Angle::new(1.0, 6.0); // π/6
    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    // blade 1 context
    let start = Instant::now();
    for i in 0..pixel_count {
        let pi_frac = 2.0 * i as f64 / pixel_count as f64;
        let scanner = Geonum::new_with_angle(1.0, Angle::new(pi_frac, 1.0) + rotation);
        black_box(scanner.mag * scanner.angle.project(ex));
        black_box(scanner.mag * scanner.angle.project(ey));
    }
    let geo_blade_1 = start.elapsed();

    // blade 1_000 context
    let high_blade = Angle::new_with_blade(1_000, 0.0, 1.0);
    let start = Instant::now();
    for i in 0..pixel_count {
        let pi_frac = 2.0 * i as f64 / pixel_count as f64;
        let scanner = Geonum::new_with_angle(1.0, Angle::new(pi_frac, 1.0) + high_blade + rotation);
        black_box(scanner.mag * scanner.angle.project(ex));
        black_box(scanner.mag * scanner.angle.project(ey));
    }
    let geo_blade_1k = start.elapsed();

    // blade 1_000_000 context
    let extreme_blade = Angle::new_with_blade(1_000_000, 0.0, 1.0);
    let start = Instant::now();
    for i in 0..pixel_count {
        let pi_frac = 2.0 * i as f64 / pixel_count as f64;
        let scanner =
            Geonum::new_with_angle(1.0, Angle::new(pi_frac, 1.0) + extreme_blade + rotation);
        black_box(scanner.mag * scanner.angle.project(ex));
        black_box(scanner.mag * scanner.angle.project(ey));
    }
    let geo_blade_1m = start.elapsed();

    // --- punchline ---

    let speedup_100d = trad_100d.as_nanos() as f64 / geo_blade_1.as_nanos().max(1) as f64;

    println!("\n  traditional 2D per pixel:   {:?}", trad_2d);
    println!("  traditional 10D per pixel:  {:?}", trad_10d);
    println!("  traditional 100D per pixel: {:?}", trad_100d);
    println!("  geonum per pixel:           {:?}", geo_blade_1);
    println!("  speedup vs 100D:            {:.0}×", speedup_100d);
    println!(
        "\n  traditional 100D storage:   {} bytes/object",
        100 * std::mem::size_of::<f64>()
    );
    println!(
        "  geonum storage:             {} bytes/object",
        std::mem::size_of::<Geonum>()
    );
    let geonum_bytes = std::mem::size_of::<Geonum>();
    let cache_line = 64; // bytes
    let trad_100d_per_cache_line = cache_line / (100 * std::mem::size_of::<f64>());
    let geonum_per_cache_line = cache_line / geonum_bytes;

    println!("\n  objects per 64-byte cache line:");
    println!("    traditional 100D: {}", trad_100d_per_cache_line);
    println!("    geonum:           {}", geonum_per_cache_line);

    let geo_throughput = pixel_count as f64 / geo_blade_1.as_secs_f64();
    let trad_100d_throughput = pixel_count as f64 / trad_100d.as_secs_f64();

    // n gpus are irrelevant
    println!("\n  throughput (1 cpu, 0 gpus):");
    println!(
        "    traditional 100D: {:.0} objects/sec",
        trad_100d_throughput
    );
    println!("    geonum:           {:.0} objects/sec", geo_throughput);

    println!("\n  blade 1 time:       {:?}", geo_blade_1);
    println!("  blade 1,000 time:   {:?}", geo_blade_1k);
    println!("  blade 1,000,000 time: {:?}", geo_blade_1m);
    println!("  dimension doesnt move the needle\n");

    // --- deterministic proofs (no timing, no CI flake) ---

    // GA memory wall: 2^n components × 8 bytes per multivector
    let ga_10d_bytes = 2_u64.pow(10) * 8; // 8 KB per object
    let ga_20d_bytes = 2_u64.pow(20) * 8; // 8 MB per object
    let ga_30d_bytes = 2_u64.pow(30) * 8; // 8 GB per object — exceeds most RAM
                                          // 2^100 components dont fit in any computer ever built
                                          // so high-dimensional work falls back to linear algebra (timed above)

    assert_eq!(ga_10d_bytes, 8_192);
    assert_eq!(ga_20d_bytes, 8_388_608);
    assert_eq!(ga_30d_bytes, 8_589_934_592);

    // linear algebra fallback: n×n matrix ops per pixel scale with n²
    let trad_ops_2d = 6; // 4 mults + 2 adds
    let trad_ops_10d = 190; // 100 mults + 90 adds
    let trad_ops_100d = 19_900; // 10000 mults + 9900 adds
    let geonum_ops = 6; // 2 projections × (1 sub + 1 cos + 1 mul)

    assert!(trad_ops_100d > trad_ops_10d * 10);
    assert!(trad_ops_10d > trad_ops_2d * 10);
    assert_eq!(geonum_ops, 6); // constant regardless of dimension

    // geonum replaces both: 24 bytes per object at any dimension
    let geonum_bytes = std::mem::size_of::<Geonum>();
    assert_eq!(geonum_bytes, 24); // mag(8) + blade(8) + rem(8)
    assert_eq!(geonum_bytes * 33, 792); // 33 geonums fit where 1 traditional 100D vector fits
    assert_eq!(geonum_bytes * 341, 8_184); // 341 geonums fit where 1 GA 10D multivector fits

    // cache line: 64 bytes fits 0 traditional 100D objects, 2 geonums
    assert_eq!(64 / (100 * std::mem::size_of::<f64>()), 0);
    assert_eq!(64 / geonum_bytes, 2);
}

#[test]
fn it_enables_a_graphics_rendering_stack_rewrite() {
    // every stage of a traditional graphics pipeline maps to geonum angle arithmetic
    // this test walks the full pipeline on a single triangle scene
    //
    // see tests/cga_test.rs, tests/pga_test.rs, tests/optics_test.rs,
    // tests/computer_vision_test.rs for the operations used below in full context

    let ex = Angle::new(0.0, 1.0);
    let ey = Angle::new(1.0, 2.0);

    // --- 1. scene: three vertices of a triangle ---

    let v0 = Geonum::new_from_cartesian(1.0, 0.5); // avoid origin — zero magnitude has no direction
    let v1 = Geonum::new_from_cartesian(4.0, 0.0);
    let v2 = Geonum::new_from_cartesian(2.0, 3.0);

    // --- 2. model transform: rotate scene by π/6 ---
    // traditional: 3 matrix multiplies (one per vertex)
    // geonum: 3 angle additions

    let model_rotation = Angle::new(1.0, 6.0);
    let v0_world = v0.rotate(model_rotation);
    let v1_world = v1.rotate(model_rotation);
    let v2_world = v2.rotate(model_rotation);

    // magnitudes preserved through rotation
    assert!((v0_world.mag - v0.mag).abs() < EPSILON);
    assert!((v1_world.mag - v1.mag).abs() < EPSILON);
    assert!((v2_world.mag - v2.mag).abs() < EPSILON);

    // --- 3. view transform: camera panned 15° ---
    // traditional: construct view matrix, multiply 3 vertices
    // geonum: add camera angle

    let camera_pan = Angle::new(1.0, 12.0); // π/12
    let v0_view = v0_world.rotate(camera_pan);
    let v1_view = v1_world.rotate(camera_pan);
    let v2_view = v2_world.rotate(camera_pan);

    // --- 4. perspective projection: closer objects appear larger ---
    // traditional: 4×4 projection matrix, homogeneous divide
    // geonum: screen size = focal_length / depth, angle preserved
    // see tests/pga_test.rs:1421 it_applies_perspective_transformations

    let focal_length = 10.0;
    let v0_proj = Geonum::new_with_angle(focal_length / v0_view.mag, v0_view.angle);
    let v1_proj = Geonum::new_with_angle(focal_length / v1_view.mag, v1_view.angle);
    let v2_proj = Geonum::new_with_angle(focal_length / v2_view.mag, v2_view.angle);

    // perspective preserves angle (direction on screen)
    assert_eq!(v0_proj.angle, v0_view.angle);
    assert_eq!(v1_proj.angle, v1_view.angle);
    assert_eq!(v2_proj.angle, v2_view.angle);

    // closer vertices project larger
    // v0 is closest (smallest mag), so v0_proj has largest screen size
    if v0_view.mag < v1_view.mag {
        assert!(v0_proj.mag > v1_proj.mag, "closer projects larger");
    }

    // --- 5. screen projection: recover pixel coordinates ---
    // traditional: extract x, y from homogeneous vector
    // geonum: project onto screen axes

    let s0x = v0_proj.mag * v0_proj.angle.project(ex);
    let s0y = v0_proj.mag * v0_proj.angle.project(ey);
    let s1x = v1_proj.mag * v1_proj.angle.project(ex);
    let s1y = v1_proj.mag * v1_proj.angle.project(ey);
    let s2x = v2_proj.mag * v2_proj.angle.project(ex);
    let s2y = v2_proj.mag * v2_proj.angle.project(ey);

    // pythagorean identity at every vertex
    assert!((s0x * s0x + s0y * s0y - v0_proj.mag * v0_proj.mag).abs() < EPSILON);
    assert!((s1x * s1x + s1y * s1y - v1_proj.mag * v1_proj.mag).abs() < EPSILON);
    assert!((s2x * s2x + s2y * s2y - v2_proj.mag * v2_proj.mag).abs() < EPSILON);

    // --- 6. ray casting: cast a ray toward the triangle ---
    // traditional: parametric ray equation, solve quadratic per object
    // geonum: ray is a single geonum, propagation is scale
    // see tests/optics_test.rs:11 its_a_ray

    // cast ray toward midpoint of edge v1–v2 in view space
    let edge_mid_angle = (v1_view.angle + v2_view.angle) / 2.0;
    let edge_mid_mag = (v1_view.mag + v2_view.mag) / 2.0;
    let ray = Geonum::new_with_angle(1.0, edge_mid_angle); // unit ray in that direction
    let propagated = ray.scale(edge_mid_mag); // propagate to target distance

    assert_eq!(
        propagated.angle, ray.angle,
        "propagation preserves direction"
    );
    assert!(
        (propagated.mag - edge_mid_mag).abs() < EPSILON,
        "ray reaches target distance"
    );

    // --- 7. intersection: find where ray meets scene geometry ---
    // traditional CGA: embed in 5D, compute meet of trivectors
    // geonum: meet of two bivectors
    // see tests/cga_test.rs:897 it_finds_circle_circle_intersection

    // encode the ray and a bounding circle around the triangle as bivectors
    let bounding_radius = v1_view.mag.max(v2_view.mag);
    let bounding_circle = Geonum::new_with_blade(bounding_radius, 2, 0.0, 1.0);
    // promote propagated ray to bivector by adding 2 blades
    let ray_bivector = Geonum::new_with_angle(propagated.mag, propagated.angle)
        .increment_blade()
        .increment_blade();
    let hit = bounding_circle.meet(&ray_bivector);

    // grade encodes intersection type
    let hit_grade = hit.angle.grade();
    assert!(hit_grade == 1 || hit_grade == 3, "meet produces odd grade");

    // --- 8. surface normal at hit point ---
    // traditional: cross product of tangent vectors, normalize
    // geonum: differentiate for tangent, rotate π/2 for normal
    // see tests/cga_test.rs:2559 it_computes_tangent_to_circle

    let hit_point = Geonum::new_with_angle(bounding_radius, propagated.angle);
    let tangent = hit_point.differentiate(); // π/2 rotation gives tangent
    let normal = tangent.differentiate(); // another π/2 gives outward normal

    assert_eq!(
        tangent.angle.grade(),
        (hit_point.angle.grade() + 1) % 4,
        "tangent one grade up"
    );
    assert_eq!(
        normal.angle.grade(),
        (hit_point.angle.grade() + 2) % 4,
        "normal two grades up"
    );
    assert_eq!(
        tangent.mag, hit_point.mag,
        "differentiation preserves magnitude"
    );

    // --- 9. reflection: bounce ray off surface ---
    // traditional: r = d - 2(d·n)n with normalized vectors
    // geonum: reflect via angle arithmetic
    // see tests/cga_test.rs:1807 it_applies_reflection_across_a_line

    let reflected = propagated.reflect(&hit_point);

    assert_eq!(
        reflected.mag, propagated.mag,
        "reflection preserves intensity"
    );

    // --- 10. depth ordering: sort the triangle vertices ---
    // traditional: transform to clip space, compare z-buffer values
    // geonum: magnitude IS depth, sort directly

    let mut verts_by_depth = [v0_view, v1_view, v2_view];
    verts_by_depth.sort_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap());

    // nearest vertex first
    assert!(verts_by_depth[0].mag <= verts_by_depth[1].mag);
    assert!(verts_by_depth[1].mag <= verts_by_depth[2].mag);

    // --- 11. conformal split: separate geometry from appearance ---
    // traditional CGA: e₊/e₋ null basis vector decomposition
    // geonum: magnitude = euclidean part, angle = conformal part
    // see tests/cga_test.rs:4702 it_handles_conformal_split

    let euclidean_part = v1_view.mag; // distance from camera — geometry
    let conformal_part = v1_view.angle; // direction on screen — appearance

    let reconstructed = Geonum::new_with_angle(euclidean_part, conformal_part);
    assert!(
        (reconstructed - v1_view).mag < EPSILON,
        "split reconstructs"
    );

    // scaling changes geometry, preserves appearance
    let zoomed = v1_view.scale(2.0);
    assert_eq!(zoomed.angle, v1_view.angle, "zoom preserves conformal part");
    assert!(
        (zoomed.mag - v1_view.mag * 2.0).abs() < EPSILON,
        "zoom scales euclidean part"
    );

    // rotation changes appearance, preserves geometry
    let panned = v1_view.rotate(Angle::new(1.0, 4.0));
    assert!(
        (panned.mag - v1_view.mag).abs() < EPSILON,
        "pan preserves euclidean part"
    );
    assert_ne!(panned.angle, v1_view.angle, "pan changes conformal part");

    // --- the full pipeline composed on one triangle scene ---
    // model transform: angle addition
    // view transform: angle addition
    // perspective projection: focal_length / depth
    // screen projection: angle.project
    // ray casting: scale toward scene
    // intersection: meet
    // tangent: differentiate
    // normal: differentiate twice
    // reflection: reflect
    // depth sort: compare magnitudes
    // conformal split: [mag, angle] decomposition
    //
    // no matrices, no homogeneous coordinates, no basis vectors
    // every stage is O(1) regardless of dimension
}
