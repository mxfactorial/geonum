use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// trig encoded on the π/2 incidence lattice created by quadrature sin(θ+π/2) = cos(θ)
// dual is the unit bivector that defines antisymmetry within the quadrature
// Q = +π/2 rotation, D = +π rotation (dual)
// as θ advances, blade cycles 0→1→2→3 with Q
// crossing a diameter applies D, so “negative” is a geometric position, not a sign bit
// this file encodes trig as geonum outputs instead of f64 scalars

// conventional references used here for clarity:
// cos, tan → 0 (grade pair 0↔2)
// sin → π/2 (grade pair 1↔3)

// api note: using Geonum::cos/sin/tan associated functions directly

#[test]
fn it_maps_trig_onto_the_pi_over_2_lattice() {
    // sample across quadrants using rational π fractions
    let angles = [
        Angle::new(1.0, 6.0),  // π/6: i → + + +
        Angle::new(2.0, 3.0),  // 2π/3: ii → − + −
        Angle::new(7.0, 6.0),  // 7π/6: iii → − − +
        Angle::new(11.0, 6.0), // 11π/6: iv → + − −
    ];

    for a in angles {
        let theta = a.grade_angle();

        let c = Geonum::cos(a);
        assert!((c.mag - theta.cos().abs()).abs() < EPSILON);
        assert!(matches!(c.angle.grade(), 0 | 2)); // even pair

        let s = Geonum::sin(a);
        assert!((s.mag - theta.sin().abs()).abs() < EPSILON);
        assert!(matches!(s.angle.grade(), 1 | 3)); // odd pair

        let t = Geonum::tan(a);
        assert!((t.mag - theta.tan().abs()).abs() < EPSILON);
        assert!(matches!(t.angle.grade(), 1 | 3)); // odd pair via sin/cos division
    }
}

#[test]
fn it_shifts_sin_by_a_quarter_turn_of_cos() {
    // sin(θ) = cos(θ − π/2) on the lattice via Q shift
    let a = Angle::new(5.0, 12.0);
    let a_minus_q = a + Angle::new(3.0, 2.0);

    let s = Geonum::sin(a);
    let c_shifted = Geonum::cos(a_minus_q);

    // same magnitude; sin is a quarter-turn (Q) of cos
    let q = Angle::new(1.0, 2.0);
    assert!((s.mag - c_shifted.mag).abs() < EPSILON);
    assert_eq!(s.angle.base_angle(), (c_shifted.angle + q).base_angle());
}

#[test]
fn it_applies_dual_at_diameter_crossings() {
    // dual is the unit bivector that defines antisymmetry within the quadrature
    // pick a cos-negative case (quadrant ii)
    let a = Angle::new(2.0, 3.0); // 120°
    let c = Geonum::cos(a);

    // negative cos encodes D (π) relative to 0 angle
    let expected = Angle::new(0.0, 1.0) + Angle::new(1.0, 1.0);
    assert_eq!(c.angle.grade(), expected.grade());

    // pick a sin-negative case (quadrant iv)
    let b = Angle::new(11.0, 6.0); // 330°
    let s = Geonum::sin(b);
    let expected_sin = Angle::new(1.0, 2.0) + Angle::new(1.0, 1.0);
    assert_eq!(s.angle.grade(), expected_sin.grade());
}

#[test]
fn it_relates_tan_to_sin_over_cos_geometrically() {
    // tan_g is defined via division so its D events inherit from sin/cos
    let a = Angle::new(3.0, 8.0);

    let t = Geonum::tan(a);
    let s_over_c = Geonum::sin(a).div(&Geonum::cos(a));

    assert!((t.mag - s_over_c.mag).abs() < EPSILON);
    assert_eq!(t.angle.base_angle(), s_over_c.angle.base_angle());
}

#[test]
fn it_is_reference_agnostic() {
    // same geometric data under different references
    // shift sin to 0↔2 by Q⁻¹ and compare with cos
    let a = Angle::new(2.0, 5.0);

    let sin_as_even_pair = Geonum::cos(a + Angle::new(3.0, 2.0));
    let s = Geonum::sin(a);

    assert!((s.mag - sin_as_even_pair.mag).abs() < EPSILON);
    // different references, same information: sin = cos + Q
    let q = Angle::new(1.0, 2.0);
    assert_eq!(
        s.angle.base_angle(),
        (sin_as_even_pair.angle + q).base_angle()
    );
}

#[test]
fn it_relates_trig_through_grade_hierarchy() {
    // pick cos at 0; others follow by Q shifts and D events
    // all other trig outputs follow by Q shifts and D events
    let a = Angle::new(1.0, 6.0);

    let cos0 = Geonum::cos(a);
    let sin0 = Geonum::sin(a);
    let tan0 = Geonum::tan(a);

    // sin from cos by a quarter-turn: sin(θ) = cos(θ−π/2)
    let cos_shifted = Geonum::cos(a + Angle::new(3.0, 2.0));
    let q = Angle::new(1.0, 2.0);
    assert_eq!(
        sin0.angle.base_angle(),
        (cos_shifted.angle + q).base_angle()
    );

    // grade hierarchy step: expressed via Q equality above

    // tan inherits odd parity via sin/cos division
    assert_eq!((cos0.angle.grade() + tan0.angle.grade()) % 2, 1);

    // diameter crossing (D) preserves the Q relation: sin(θ) = cos(θ−π/2)+Q
    let a_obtuse = Angle::new(2.0, 3.0); // 120°
    let sin_obtuse = Geonum::sin(a_obtuse);
    let cos_shifted_obtuse = Geonum::cos(a_obtuse + Angle::new(3.0, 2.0));
    let q = Angle::new(1.0, 2.0);
    assert!((sin_obtuse.mag - cos_shifted_obtuse.mag).abs() < EPSILON);
    assert_eq!(
        sin_obtuse.angle.base_angle(),
        (cos_shifted_obtuse.angle + q).base_angle()
    );
}

#[test]
fn it_doesnt_need_cos_and_tan() {
    // define cos and tan via sin and quarter-turn
    let q = Angle::new(1.0, 2.0);
    let cos_via_sin = |a: Angle| {
        // derive cos from sin with quarter-turn, then re-anchor to even pair
        let s = Geonum::sin(a + q);
        let base = if s.angle.grade() == 1 {
            Angle::new(0.0, 1.0)
        } else {
            Angle::new(1.0, 1.0)
        };
        Geonum::new_with_angle(s.mag, base)
    };
    let tan_via_sin = |a: Angle| {
        let s = Geonum::sin(a);
        let c = cos_via_sin(a);
        s.div(&c)
    };

    // sample angles across quadrants
    let angles = [
        Angle::new(1.0, 6.0),  // π/6
        Angle::new(2.0, 3.0),  // 2π/3
        Angle::new(7.0, 6.0),  // 7π/6
        Angle::new(11.0, 6.0), // 11π/6
    ];

    for a in angles {
        let theta = a.grade_angle();

        // cos via sin matches standard magnitude and parity
        let c = cos_via_sin(a);
        assert!((c.mag - theta.cos().abs()).abs() < EPSILON);
        assert!(matches!(c.angle.grade(), 0 | 2));

        // tan via sin ratio matches standard magnitude and parity (avoid singularities)
        if theta.cos().abs() > 1e-12 {
            let t = tan_via_sin(a);
            assert!((t.mag - theta.tan().abs()).abs() < EPSILON);
            assert!(matches!(t.angle.grade(), 1 | 3));
        }
    }
}

#[test]
fn it_is_projection() {
    // projection-native: [r, θ] projects onto any φ without coordinates or "linear combinations"
    // identities: θ=φ → [r,φ], θ=φ+π → [r,φ+π], θ=φ±π/2 → [0,φ]

    let r = 1.0;
    let zero = Angle::new(0.0, 1.0);
    let quarter = Angle::new(1.0, 2.0);

    // setup and identity: v is already a projection-primitive polar number
    // projecting onto its own angle returns itself
    let v = Geonum::new_with_angle(r, Angle::new(1.0, 4.0)); // θ = π/4
    let onto_self = Geonum::new_with_angle(1.0, v.angle);
    let proj_self = v.project(&onto_self);
    assert_eq!(proj_self.mag, 1.0);
    assert_eq!(proj_self.angle, v.angle);

    // adjacent (φ=0): [1, π/4] → [√2/2, 0]
    // hypotenuse × cos(angle): adjacent = r·cos(θ−φ)
    // this is the r cos term in the linear combination r cos(θ−φ) e_φ + r sin(θ−φ) e_{φ+π/2}
    let onto_adjacent = Geonum::new_with_angle(1.0, zero);
    let proj_adjacent = v.project(&onto_adjacent);
    assert!((proj_adjacent.mag - (2.0_f64).sqrt() / 2.0).abs() < EPSILON);
    assert_eq!(proj_adjacent.angle, zero);

    // opposite (φ=π/2): [1, π/4] → [√2/2, π/2]
    // hypotenuse × sin(angle): opposite = r·sin(θ−φ)
    // this is the r sin term in the same linear combination above
    let onto_opposite = Geonum::new_with_angle(1.0, quarter);
    let proj_opposite = v.project(&onto_opposite);
    assert!((proj_opposite.mag - (2.0_f64).sqrt() / 2.0).abs() < EPSILON);
    assert_eq!(proj_opposite.angle, quarter);

    // pythagorean identity from projections
    let sum_sq = proj_adjacent.mag.powi(2) + proj_opposite.mag.powi(2);
    assert!((sum_sq - v.mag.powi(2)).abs() < EPSILON);
}

#[test]
fn it_adds_vectors_with_cosine_interference() {
    // cosine interference: rotations combine according to cosine rule
    // c² = a² + b² + 2ab*cos(θ) where θ is angle difference
    let a = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let b = Geonum::new(3.0, 1.0, 4.0); // [3, π/4]

    let interference_result = a + b;

    println!("Cosine interference addition:");
    println!("  a: length={}, angle={:.3}", a.mag, a.angle.grade_angle());
    println!("  b: length={}, angle={:.3}", b.mag, b.angle.grade_angle());
    println!(
        "  result: length={:.3}, angle={:.3}",
        interference_result.mag,
        interference_result.angle.grade_angle()
    );

    // manually compute using cosine rule
    let angle1 = a.angle.grade_angle();
    let angle2 = b.angle.grade_angle();

    // step 1: sum projections (no squares)
    let y_sum = a.mag * angle1.sin() + b.mag * angle2.sin();
    let x_sum = a.mag * angle1.cos() + b.mag * angle2.cos();

    // step 2: derive angle using atan2 (no squares)
    let manual_angle = y_sum.atan2(x_sum);

    // step 3: derive length using projection constraint (no squares)
    let manual_length =
        a.mag * (angle1 - manual_angle).cos() + b.mag * (angle2 - manual_angle).cos();

    // verify manual computation matches cosine interference
    assert!((manual_length - interference_result.mag).abs() < EPSILON);
    assert!((manual_angle - interference_result.angle.grade_angle()).abs() < EPSILON);

    println!("Manual cosine interference verification:");
    println!("  y_sum: {:.6}, x_sum: {:.6}", y_sum, x_sum);
    println!(
        "  angle: atan2({:.6}, {:.6}) = {:.6}",
        y_sum, x_sum, manual_angle
    );
    println!("  length: projection constraint = {:.6}", manual_length);
    println!("  ✓ No squares anywhere in the computation!");
}

#[test]
fn it_derives_pythagorean_identity_from_quadrature() {
    // sin²+cos² = 1 is the pythagorean identity
    // but it comes from quadrature: sin(θ+π/2) = cos(θ)

    let angle = Angle::new(2.0, 7.0); // 2π/7

    // quadrature relationship
    let angle_plus_quarter = angle + Angle::new(1.0, 2.0);
    let sin_shifted = angle_plus_quarter.grade_angle().sin();
    let cos_original = angle.grade_angle().cos();

    println!("Quadrature relationship:");
    println!(
        "  sin(θ+π/2) = sin({:.3}) = {:.3}",
        angle_plus_quarter.grade_angle(),
        sin_shifted
    );
    println!(
        "  cos(θ) = cos({:.3}) = {:.3}",
        angle.grade_angle(),
        cos_original
    );
    assert!((sin_shifted - cos_original).abs() < EPSILON);

    // pythagorean identity from quadrature
    let sin_val = angle.grade_angle().sin();
    let cos_val = angle.grade_angle().cos();
    let identity = sin_val.powi(2) + cos_val.powi(2);

    println!("\nPythagorean identity:");
    println!(
        "  sin²({:.3}) + cos²({:.3}) = {:.3}² + {:.3}² = {:.3}",
        angle.grade_angle(),
        angle.grade_angle(),
        sin_val,
        cos_val,
        identity
    );
    assert!((identity - 1.0).abs() < EPSILON);

    // now connect to 3-4-5: if hypotenuse is at angle θ
    // and we project onto 0° and 90° directions
    // we get adjacent = hyp×cos(θ) and opposite = hyp×sin(θ)
    let hypotenuse = 5.0_f64;
    let theta = (4.0_f64 / 5.0_f64).asin(); // angle for 3-4-5 triangle

    let adj = hypotenuse * theta.cos(); // should be 3
    let opp = hypotenuse * theta.sin(); // should be 4

    println!("\n3-4-5 triangle from quadrature:");
    println!("  hypotenuse: {}", hypotenuse);
    println!("  angle: {:.3}", theta);
    println!(
        "  adjacent: {} × cos({:.3}) = {:.3}",
        hypotenuse, theta, adj
    );
    println!(
        "  opposite: {} × sin({:.3}) = {:.3}",
        hypotenuse, theta, opp
    );

    assert!((adj - 3.0).abs() < EPSILON);
    assert!((opp - 4.0).abs() < EPSILON);

    // the pythagorean theorem is really saying:
    // (hyp×cos)² + (hyp×sin)² = hyp²
    // which simplifies to: hyp²(cos²+sin²) = hyp²
    // which uses the quadrature identity: cos²+sin² = 1

    let check = adj.powi(2) + opp.powi(2);
    println!("  check: {:.3}² + {:.3}² = {:.3}", adj, opp, check);
    assert!((check - hypotenuse.powi(2)).abs() < EPSILON);

    println!("\nPythagorean theorem is quadrature in disguise:");
    println!("  3² + 4² = 5² ↔ (5cos)² + (5sin)² = 5²");
    println!("  ↔ 25(cos²+sin²) = 25");
    println!("  ↔ cos²+sin² = 1 (quadrature identity)");
}

#[test]
fn it_expresses_pythagoras_theorem_through_composed_angles() {
    // 3² + 4² = 5² isn't about lengths - it's about angle composition through quadrature
    // each number encodes a rotation, pythagorean theorem describes how rotations combine

    // the classic 3-4-5 triangle
    let three = Geonum::new(3.0, 0.0, 1.0); // whatever rotation 3 encodes
    let four = Geonum::new(4.0, 1.0, 2.0); // whatever rotation 4 encodes, rotated π/2

    println!("3-4-5 triangle as angle composition:");
    println!(
        "  3: length={} at angle={:.3}",
        three.mag,
        three.angle.grade_angle()
    );
    println!(
        "  4: length={} at angle={:.3}",
        four.mag,
        four.angle.grade_angle()
    );

    // when angles are π/2 apart (orthogonal), their combination follows cosine rule with cos(π/2) = 0
    let angle_diff = (four.angle - three.angle).grade_angle();
    println!(
        "  angle difference: {:.3} (π/2 = {:.3})",
        angle_diff,
        PI / 2.0
    );
    assert!((angle_diff - PI / 2.0).abs() < EPSILON);

    // the "5" emerges from cosine interference with orthogonal rotations
    let combined = three + four;
    println!(
        "  combined: length={:.3} at angle={:.3}",
        combined.mag,
        combined.angle.grade_angle()
    );

    // verify pythagorean relationship in lengths
    let expected_length = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt();
    assert!((combined.mag - expected_length).abs() < EPSILON);
    assert!((combined.mag - 5.0).abs() < EPSILON);

    // but the real insight: this length relationship comes from cosine interference
    // cos(π/2) = 0 means orthogonal rotations combine with zero interference term
}
