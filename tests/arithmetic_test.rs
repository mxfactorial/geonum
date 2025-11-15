// prove "1 - 1 = 0" is destructive interference via cosine rule in angle space
// subtraction is projection convention - upstream it's geometric addition with angle π offset

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_proves_subtraction_is_destructive_interference() {
    // projection space (traditional): 1 - 1 = 0
    let scalar_result: f64 = 1.0 - 1.0;
    assert_eq!(scalar_result, 0.0, "projection space: 1 - 1 = 0");

    // angle space: [1, 0] + [1, π] → destructive interference
    let magnitude_1_at_0 = Geonum::new(1.0, 0.0, 1.0); // [1, 0°]
    let magnitude_1_at_pi = Geonum::new(1.0, 1.0, 1.0); // [1, π] ("negative" as position)

    println!("\nAngle space addition:");
    println!("  [1, 0°] + [1, π]");
    println!("  magnitude 1 at angle 0");
    println!("  plus magnitude 1 at angle π (opposite direction)");

    // geometric addition via cosine interference
    let geometric_sum = magnitude_1_at_0 + magnitude_1_at_pi;

    println!("\nCosine interference:");
    println!("  angle between: π (opposite directions)");
    println!("  cos(π) = -1 (complete cancellation)");

    // verify the cosine rule: c² = a² + b² + 2ab·cos(θ)
    let a: f64 = 1.0;
    let b: f64 = 1.0;
    let theta = PI; // angle between them
    let expected_magnitude_squared = a.powi(2) + b.powi(2) + 2.0 * a * b * theta.cos();

    println!("\nCosine rule: c² = a² + b² + 2ab·cos(θ)");
    println!("  c² = 1² + 1² + 2(1)(1)·cos(π)");
    println!("  c² = 1 + 1 + 2(-1)");
    println!("  c² = 0");

    assert!(
        (expected_magnitude_squared - 0.0).abs() < EPSILON,
        "cosine rule gives magnitude² = 0"
    );

    // geometric sum magnitude collapses to ~0 via destructive interference
    println!("\nDestructive interference result:");
    println!("  magnitude: {:.10}", geometric_sum.length);
    println!(
        "  angle: {} (degenerate - no meaningful direction for zero magnitude)",
        geometric_sum.angle.grade_angle()
    );

    assert!(
        geometric_sum.length < EPSILON,
        "magnitude collapsed to ~0 via destructive interference: {}",
        geometric_sum.length
    );

    // the degenerate case: zero magnitude has undefined angle
    // this is why "0" in projection space loses information
    // you can't recover which directions interfered to create the zero

    // projection back to scalar
    let projected_scalar = geometric_sum.length;
    assert!(
        (projected_scalar - scalar_result).abs() < EPSILON,
        "projection to scalar: {} matches traditional result {}",
        projected_scalar,
        scalar_result
    );

    println!("\nWhat projection space hides:");
    println!("  '1 - 1 = 0' looks like simple algebra");
    println!("  actually: geometric addition with θ=π creating destructive interference");
    println!("  cos(π) = -1 causes complete cancellation");
    println!("  result magnitude ~0, angle degenerate");
}

#[test]
fn it_shows_subtraction_cost_vs_projection() {
    // projection space: flip sign bit, subtract
    let a_scalar = 5.0;
    let b_scalar = 3.0;
    let projection_result = a_scalar - b_scalar; // 2.0

    // angle space: full geometric addition with interference
    let a_geo = Geonum::new(5.0, 0.0, 1.0); // [5, 0]
    let b_geo = Geonum::new(3.0, 1.0, 1.0); // [3, π] (negative as position)

    // compute via cosine rule
    let theta = PI;
    let magnitude_squared = a_geo.length.powi(2)
        + b_geo.length.powi(2)
        + 2.0 * a_geo.length * b_geo.length * theta.cos();
    let expected_magnitude = magnitude_squared.sqrt();

    println!("\n5 - 3 = 2 via cosine interference:");
    println!("  [5, 0] + [3, π]");
    println!("  c² = 25 + 9 + 2(5)(3)(-1)");
    println!("  c² = 25 + 9 - 30 = 4");
    println!("  c = 2");

    let geometric_result = a_geo + b_geo;

    assert!(
        (geometric_result.length - expected_magnitude).abs() < EPSILON,
        "geometric magnitude {} matches cosine rule {}",
        geometric_result.length,
        expected_magnitude
    );

    assert!(
        (geometric_result.length - projection_result).abs() < EPSILON,
        "angle space {} matches projection space {}",
        geometric_result.length,
        projection_result
    );

    // the cost difference
    println!("\nProjection space: flip sign bit on 3, subtract");
    println!("Angle space: compute cos(π), multiply 2(5)(3)(-1), add 25+9-30, sqrt(4)");
    println!("  → more expensive, but shows actual geometric interference");
}

#[test]
fn it_proves_negative_is_position_not_sign() {
    // projection space: -1 is scalar with sign bit
    let _negative_one_scalar: f64 = -1.0;

    // angle space: [1, π] is unit magnitude at angle π
    let negative_one_geo = Geonum::new(1.0, 1.0, 1.0); // [1, π]

    println!("\nRepresenting -1:");
    println!("  Projection space: 1 with sign bit set");
    println!("  Angle space: [magnitude=1, angle=π]");

    // verify it's at angle π
    assert!(
        (negative_one_geo.angle.grade_angle() - PI).abs() < EPSILON,
        "angle π represents 'negative'"
    );

    // verify magnitude is unsigned
    assert_eq!(
        negative_one_geo.length, 1.0,
        "magnitude is unsigned (no sign bit)"
    );

    // demonstrate: no sign bit branching needed
    let positive_one = Geonum::new(1.0, 0.0, 1.0);
    let negative_one = Geonum::new(1.0, 1.0, 1.0);

    // both have same magnitude representation - difference is geometric position
    assert_eq!(
        positive_one.length, negative_one.length,
        "both have magnitude 1, no sign bit"
    );

    // the "negativeness" is encoded in angle, not sign
    let angle_difference = (negative_one.angle - positive_one.angle).grade_angle();
    assert!(
        (angle_difference - PI).abs() < EPSILON,
        "negative is π rotation from positive, not sign flip"
    );

    println!("\nNo sign bit logic needed:");
    println!("  positive [1, 0]: magnitude 1 at angle 0");
    println!("  negative [1, π]: magnitude 1 at angle π");
    println!("  difference: π rotation (geometric position)");
}

#[test]
fn it_demonstrates_interference_pattern_for_various_angles() {
    // show how different angles create different interference patterns
    let a = Geonum::new(3.0, 0.0, 1.0); // [3, 0]

    // same direction: θ = 0, cos(0) = 1 (constructive)
    let b_same = Geonum::new(4.0, 0.0, 1.0);
    let same_result = a + b_same;
    let same_expected = (3.0_f64.powi(2) + 4.0_f64.powi(2) + 2.0 * 3.0 * 4.0 * 1.0).sqrt();
    assert!(
        (same_result.length - same_expected).abs() < EPSILON,
        "θ=0: cos(0)=1, full constructive: 3+4=7, got {}",
        same_result.length
    );
    assert!((same_result.length - 7.0).abs() < EPSILON);

    // perpendicular: θ = π/2, cos(π/2) = 0 (pythagorean)
    let b_perp = Geonum::new(4.0, 1.0, 2.0); // π/2
    let perp_result = a + b_perp;
    let perp_expected = (3.0_f64.powi(2) + 4.0_f64.powi(2) + 2.0 * 3.0 * 4.0 * 0.0).sqrt();
    assert!(
        (perp_result.length - perp_expected).abs() < EPSILON,
        "θ=π/2: cos(π/2)=0, pythagorean: sqrt(9+16)=5, got {}",
        perp_result.length
    );
    assert!((perp_result.length - 5.0).abs() < EPSILON);

    // opposite: θ = π, cos(π) = -1 (destructive)
    let b_opp = Geonum::new(4.0, 1.0, 1.0); // π
    let opp_result = a + b_opp;
    let opp_expected = (3.0_f64.powi(2) + 4.0_f64.powi(2) + 2.0 * 3.0 * 4.0 * (-1.0)).sqrt();
    assert!(
        (opp_result.length - opp_expected).abs() < EPSILON,
        "θ=π: cos(π)=-1, destructive: |3-4|=1, got {}",
        opp_result.length
    );
    assert!((opp_result.length - 1.0).abs() < EPSILON);

    println!("\nInterference patterns:");
    println!(
        "  θ=0°   (cos=1):  [3,0] + [4,0]   → {} (constructive)",
        same_result.length
    );
    println!(
        "  θ=90°  (cos=0):  [3,0] + [4,π/2] → {} (pythagorean)",
        perp_result.length
    );
    println!(
        "  θ=180° (cos=-1): [3,0] + [4,π]   → {} (destructive)",
        opp_result.length
    );
    println!("\n  Subtraction is the θ=180° case");
}
