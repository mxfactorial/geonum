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
        assert!(c.near_mag(theta.cos().abs()));
        assert!(matches!(c.angle.grade(), 0 | 2)); // even pair

        let s = Geonum::sin(a);
        assert!(s.near_mag(theta.sin().abs()));
        assert!(matches!(s.angle.grade(), 1 | 3)); // odd pair

        let t = Geonum::tan(a);
        assert!(t.near_mag(theta.tan().abs()));
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
    assert!(s.near_mag(c_shifted.mag));
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

    assert!(t.near_mag(s_over_c.mag));
    assert_eq!(t.angle.base_angle(), s_over_c.angle.base_angle());
}

#[test]
fn it_is_reference_agnostic() {
    // same geometric data under different references
    // shift sin to 0↔2 by Q⁻¹ and compare with cos
    let a = Angle::new(2.0, 5.0);

    let sin_as_even_pair = Geonum::cos(a + Angle::new(3.0, 2.0));
    let s = Geonum::sin(a);

    assert!(s.near_mag(sin_as_even_pair.mag));
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
    assert!(sin_obtuse.near_mag(cos_shifted_obtuse.mag));
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
        assert!(c.near_mag(theta.cos().abs()));
        assert!(matches!(c.angle.grade(), 0 | 2));

        // tan via sin ratio matches standard magnitude and parity (avoid singularities)
        if theta.cos().abs() > 1e-12 {
            let t = tan_via_sin(a);
            assert!(t.near_mag(theta.tan().abs()));
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
    assert!(proj_adjacent.near_mag((2.0_f64).sqrt() / 2.0));
    assert_eq!(proj_adjacent.angle, zero);

    // opposite (φ=π/2): [1, π/4] → [√2/2, π/2]
    // hypotenuse × sin(angle): opposite = r·sin(θ−φ)
    // this is the r sin term in the same linear combination above
    let onto_opposite = Geonum::new_with_angle(1.0, quarter);
    let proj_opposite = v.project(&onto_opposite);
    assert!(proj_opposite.near_mag((2.0_f64).sqrt() / 2.0));
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
    // sin²+cos²=1 is not an arithmetic fact about squares — it is the QUADRATURE
    // DECOMPOSITION CLOSING. cos and sin are one unit object's projections onto two
    // axes a quarter-turn (Q) apart: cos on the even pair (0↔π), sin on the odd pair
    // (π/2↔3π/2). because the axes are orthogonal the two legs re-add to the object
    // itself, magnitude 1 — the identity computed by geonum addition, not a squared
    // sum. and it is EXACT: in the half-tangent t, cos²+sin² =
    // ((1−t²)² + (2t)²)/(1+t²)² = (1+t²)²/(1+t²)² = 1, a rational identity

    // sample every quadrant so the grade-encoded signs all participate
    let angles = [
        Angle::new(1.0, 6.0),  // π/6   — QI
        Angle::new(2.0, 3.0),  // 2π/3  — QII
        Angle::new(7.0, 6.0),  // 7π/6  — QIII
        Angle::new(11.0, 6.0), // 11π/6 — QIV
    ];

    println!("sin²+cos²=1 as the quadrature decomposition closing (no scalar squared):");
    for a in angles {
        let cos_leg = Geonum::cos(a); // projection onto the even pair (0↔π)
        let sin_leg = Geonum::sin(a); // projection onto the odd pair (π/2↔3π/2)

        // the legs live on orthogonal axes — that orthogonality IS the quadrature
        assert_eq!(cos_leg.angle.grade() % 2, 0, "cos lands on the even pair");
        assert_eq!(
            sin_leg.angle.grade() % 2,
            1,
            "sin lands on the odd pair, a quarter-turn off"
        );

        // re-adding the orthogonal legs reconstructs the unit object: magnitude 1,
        // pointing back along θ — this IS sin²+cos²=1
        let reconstructed = cos_leg + sin_leg;
        println!(
            "  θ={:.3}: |cos(grade {}) + sin(grade {})| = {:.6}, re-adding along {:.3}",
            a.grade_angle(),
            cos_leg.angle.grade(),
            sin_leg.angle.grade(),
            reconstructed.mag,
            reconstructed.angle.grade_angle()
        );
        assert!(
            reconstructed.near_mag(1.0),
            "the quadrature legs re-add to the unit object"
        );
        assert!(
            (reconstructed.angle.grade_angle() - a.grade_angle()).abs() < EPSILON,
            "and along the original ray θ"
        );
    }
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
    assert!(combined.near_mag(expected_length));
    assert!(combined.near_mag(5.0));

    // but the real insight: this length relationship comes from cosine interference
    // cos(π/2) = 0 means orthogonal rotations combine with zero interference term
}

// ═══════════════════════════════════════════════════════════════════════════
// arcsin is not a primitive — its projection recovery
//
// sin(θ) = opp/hyp collapses [hyp, θ] to a scalar ratio by discarding the angle.
// arcsin reverses this collapse. in geonum the angle is never discarded,
// so arcsin is unnecessary — its just reading the angle that projection would erase.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_sin_is_a_projection_that_discards_the_angle() {
    // a unit hypotenuse at angle θ is a complete geometric object: [1, θ]
    // sin(θ) projects it onto the π/2 axis, producing a scalar ratio
    // this projection discards θ — thats what creates the "inverse" problem

    let angles = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 4.0), // π/4
        Angle::new(1.0, 3.0), // π/3
        Angle::new(2.0, 5.0), // 2π/5
    ];

    for theta in angles {
        let hypotenuse = Geonum::new_with_angle(1.0, theta);

        // project onto opposite axis (π/2) — this is sin(θ)
        let opposite = hypotenuse.opp();
        assert!(opposite.near_mag(theta.grade_angle().sin().abs()));

        // the projection result anchors to odd pair, not θ
        assert!(matches!(opposite.angle.grade(), 1 | 3));

        // but the hypotenuse still carries θ
        assert_eq!(hypotenuse.angle, theta);
    }
}

#[test]
fn it_proves_arcsin_loses_quadrant_information() {
    // sin projects [1, θ] → ratio, losing θ AND quadrant
    // scalar arcsin can only return θ ∈ [−π/2, π/2]
    // geonum preserves quadrant via grade (blade mod 4)

    let theta1 = Angle::new(1.0, 6.0); // π/6 (quadrant I)
    let theta2 = Angle::new(5.0, 6.0); // 5π/6 (quadrant II)

    let sin1 = Geonum::sin(theta1);
    let sin2 = Geonum::sin(theta2);

    // same magnitude — scalar arcsin cannot distinguish these
    assert!(sin1.near_mag(sin2.mag));

    // but the geometric numbers at those angles are distinct
    let hyp1 = Geonum::new_with_angle(1.0, theta1);
    let hyp2 = Geonum::new_with_angle(1.0, theta2);
    assert_eq!(hyp1.angle, theta1);
    assert_eq!(hyp2.angle, theta2);

    // cos projections distinguish them: cos(π/6) > 0, cos(5π/6) < 0
    let cos1 = Geonum::cos(theta1);
    let cos2 = Geonum::cos(theta2);
    assert_eq!(cos1.angle.grade(), 0); // positive cos → grade 0
    assert_eq!(cos2.angle.grade(), 2); // negative cos → grade 2 (D event)
}

#[test]
fn it_shows_sqrt_1_minus_sin_squared_is_quadrature() {
    // √(1−sin²θ) inside arcsin's integral definition
    // is just the quadrature identity: cos(θ) = sin(θ + π/2)
    //
    // sin²+cos² = 1 says orthogonal projections partition the unit circle
    // √(1−sin²) solves for cos — scalar recovery of the Q relationship

    let angles = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 4.0), // π/4
        Angle::new(1.0, 3.0), // π/3
        Angle::new(2.0, 7.0), // 2π/7
    ];

    let q = Angle::new(1.0, 2.0); // quarter turn

    for theta in angles {
        let sin_val = theta.grade_angle().sin();
        let cos_val = theta.grade_angle().cos();

        // method 1: scalar recovery via √(1 - sin²)
        let cos_from_sqrt = (1.0 - sin_val * sin_val).sqrt();

        // method 2: quadrature via Q shift: sin(θ + π/2) = cos(θ)
        let theta_plus_q = theta + q;
        let cos_from_quadrature = theta_plus_q.grade_angle().sin();

        // method 3: geonum cos directly
        let cos_geo = Geonum::cos(theta);

        assert!((cos_from_sqrt - cos_val.abs()).abs() < EPSILON);
        assert!((cos_from_quadrature - cos_val).abs() < EPSILON);
        assert!(cos_geo.near_mag(cos_val.abs()));
    }
}
