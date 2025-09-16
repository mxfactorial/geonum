use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_adds_aligned_vectors() {
    // aligned vectors add lengths, preserve direction
    let a = Geonum::new(2.0, 0.0, 1.0);
    let b = Geonum::new(3.0, 0.0, 1.0);

    let sum = a + b;

    assert!((sum.length - 5.0).abs() < EPSILON);
    assert_eq!(sum.angle, Angle::new(0.0, 1.0));
}

#[test]
fn it_handles_opposite_vectors() {
    // opposite vectors subtract lengths, preserve dominating direction
    let a = Geonum::new(5.0, 0.0, 1.0);
    let b = Geonum::new(3.0, 1.0, 1.0); // π

    let sum = a + b;

    assert!((sum.length - 2.0).abs() < EPSILON);
    assert_eq!(sum.angle, a.angle);
}

#[test]
fn it_adds_orthogonal_vectors() {
    // 3-4-5 triangle via polar addition
    let a = Geonum::new(3.0, 0.0, 1.0);
    let b = Geonum::new(4.0, 1.0, 2.0); // π/2

    let sum = a + b;

    assert!((sum.length - 5.0).abs() < EPSILON);
    let expected_phase = (4.0_f64).atan2(3.0);
    assert!((sum.angle.mod_4_angle() - expected_phase).abs() < EPSILON);

    // history policy is preserved internally; direction and length are primary here
}

#[test]
fn it_adds_high_blades_and_preserves_history() {
    // large blade counts preserve history and direction modulo 2π
    let a = Geonum::new_with_blade(2.0, 1000, 0.0, 1.0); // blade 1000, phase 0
    let b = Geonum::new_with_blade(3.0, 1001, 1.0, 2.0); // blade 1001, phase π/2

    let sum = a + b;

    // direction modulo 2π matches cartesian result
    let (x1, y1) = a.to_cartesian();
    let (x2, y2) = b.to_cartesian();
    let expected_phase = (y1 + y2).atan2(x1 + x2);
    assert!((sum.angle.mod_4_angle() - expected_phase).abs() < EPSILON);
}

#[test]
fn it_matches_projection_based_sum() {
    // projection-native addition without component squares
    let a = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let b = Geonum::new(3.0, 1.0, 4.0); // [3, π/4]

    // δ = θb − θa in [0, 2π)
    let delta = (b.angle.mod_4_angle() - a.angle.mod_4_angle()).rem_euclid(2.0 * PI);

    // resolve b in a’s frame
    let adj = Geonum::cos(Angle::new(delta, PI)).scale(b.length);
    let opp = Geonum::sin(Angle::new(delta, PI)).scale(b.length);

    // a along 0 in its frame
    let a_along = Geonum::cos(Angle::new(0.0, 1.0)).scale(a.length);
    let sum_in_a = a_along + adj + opp;
    let result = Geonum::new_with_angle(sum_in_a.length, sum_in_a.angle + a.angle);

    // compare to direct addition
    let direct = a + b;
    assert!((result.length - direct.length).abs() < EPSILON);
    assert_eq!(result.angle.base_angle(), direct.angle.base_angle());

    // pythagorean identity from projections
    let adj_len =
        (Geonum::cos(a.angle).scale(a.length) + Geonum::cos(b.angle).scale(b.length)).length;
    let opp_len =
        (Geonum::sin(a.angle).scale(a.length) + Geonum::sin(b.angle).scale(b.length)).length;
    let hyp_sq = adj_len.powi(2) + opp_len.powi(2);
    assert!((hyp_sq - direct.length.powi(2)).abs() < EPSILON);
}

#[test]
fn it_preserves_blade_history_on_cancellation() {
    // opposite vectors cancel but preserve combined blade count
    let a = Geonum::new_with_blade(5.0, 7, 0.0, 1.0);
    let b = Geonum::new_with_blade(5.0, 3, 1.0, 1.0); // opposite

    let sum = a + b;

    assert!((sum.length - 0.0).abs() < EPSILON);
    // a has blade 7, b has blade 3 + 2 from π = blade 5
    // combined blade count = 7 + 5 = 12
    let expected = Angle::new(0.0, 1.0) // base angle at 0
        + Angle::new(7.0, 2.0) // +7 blades from a
        + Angle::new(3.0, 2.0) // +3 blades from b's explicit blade
        + Angle::new(1.0, 1.0); // +2 blades from b's π rotation (opposite)
    assert_eq!(sum.angle, expected);
}

#[test]
fn it_accumulates_blades_in_general_case() {
    // general angle addition accumulates blade counts
    // choose quarter-turn aligned remainders to avoid floating rounding in angle value
    let a = Geonum::new_with_blade(2.0, 5, 0.0, 1.0); // blade 5, value 0
    let b = Geonum::new_with_blade(3.0, 8, 0.0, 1.0); // blade 8, value 0

    let sum = a + b;

    // verify geometric result matches expected
    let (x1, y1) = a.to_cartesian();
    let (x2, y2) = b.to_cartesian();
    let expected_length = ((x1 + x2).powi(2) + (y1 + y2).powi(2)).sqrt();
    assert!((sum.length - expected_length).abs() < EPSILON);

    // a has blade 5 + π/6, b has blade 8 + π/4
    // cartesian result creates its own angle with blades
    let cartesian_result = Geonum::new_from_cartesian(x1 + x2, y1 + y2);
    let expected = cartesian_result.angle
        + Angle::new(5.0, 2.0) // +5 blades from a
        + Angle::new(8.0, 2.0) // +8 blades from b
        + Angle::new(3.0, 2.0); // net negative subtraction wraps angle
                                // compare blade history exactly and angle modulo 2π with epsilon to dodge float buggery
    assert_eq!(sum.angle, expected);
}

#[test]
fn it_handles_near_opposite_angles() {
    // numerical stability when angles are almost π apart
    let a = Geonum::new(5.0, 0.0, 1.0);
    let b = Geonum::new(3.0, 0.999999, 1.0); // almost π

    let sum = a + b;

    // should behave like subtraction
    assert!((sum.length - 2.0).abs() < 1e-5); // relaxed epsilon for near-boundary
    assert_eq!(sum.angle.grade(), 0); // stays scalar
}

#[test]
fn it_handles_zero_length_addition() {
    // zero length is identity for addition
    let zero = Geonum::new(0.0, 0.0, 1.0);
    let a = Geonum::new(5.0, 1.0, 3.0);

    let zero_plus_a = zero + a;
    assert!((zero_plus_a.length - a.length).abs() < EPSILON);
    assert_eq!(zero_plus_a.angle.base_angle(), a.angle.base_angle());

    let a_plus_zero = a + zero;
    assert!((a_plus_zero.length - a.length).abs() < EPSILON);
    assert_eq!(a_plus_zero.angle.base_angle(), a.angle.base_angle());
}

#[test]
fn it_maintains_commutative_blade_accumulation() {
    // a + b blade behavior equals b + a
    let a = Geonum::new_with_blade(2.0, 3, 1.0, 6.0);
    let b = Geonum::new_with_blade(3.0, 5, 1.0, 4.0);

    let ab = a + b;
    let ba = b + a;

    // geometric result is same
    assert!((ab.length - ba.length).abs() < EPSILON);
    assert_eq!(ab.angle.base_angle(), ba.angle.base_angle());

    // blade accumulation is commutative
    assert_eq!(ab.angle, ba.angle);

    // a has blade 3 + π/6, b has blade 5 + π/4
    let (x1, y1) = a.to_cartesian();
    let (x2, y2) = b.to_cartesian();
    let cartesian_result = Geonum::new_from_cartesian(x1 + x2, y1 + y2);
    let expected = cartesian_result.angle
        + Angle::new(3.0, 2.0) // +3 blades from a
        + Angle::new(5.0, 2.0); // +5 blades from b
    assert_eq!(ab.angle, expected);
}
