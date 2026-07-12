// the 720° mystery is one bit of winding
//
// spin-1/2 is taught as quantum weirdness: rotate an electron 360° and its
// state picks up −1, rotate 720° and it returns. the weirdness dissolves once
// the angle is stored instead of projected:
//
//   - a spinor turns at HALF the physical rate, so a 2π physical rotation is a
//     π spinor rotation — grade 2, the −1 position. 4π physical is 2π spinor —
//     blade 4, grade 0, home. the −1 is a place on the winding line
//   - observables read grade (blade mod 4), so the physical apparatus returns
//     at 2π while the stored angle differs by blade 2 — the SU(2) → SO(3)
//     double cover is exactly the bit of winding the projection forgets
//   - neutron interferometry (rauch 1975) measured that bit: a 2π-rotated arm
//     interferes destructively with an unrotated one. the experiment reads the
//     angle the projection drops
//   - the half-angle parameter spinors are built on, tan(θ/2), is the t geonum
//     stores. the sandwich product RvR† exists to double the half-angle back
//     into a rotation; cos_sin's rational formulas ARE that doubling, so
//     rotation is one angle addition, no sandwich
//
// run: cargo test --test spinor_test -- --show-output

use geonum::*;

#[test]
fn it_lands_minus_one_at_2pi_because_spin_halves_the_angle() {
    // the spinor turns at half rate: physical α → spinor α/2
    let physical_2pi = Angle::new(2.0, 1.0); // blade 4
    let physical_4pi = Angle::new(4.0, 1.0); // blade 8

    let spinor_at_2pi = physical_2pi / 2.0; // π — blade 2
    let spinor_at_4pi = physical_4pi / 2.0; // 2π — blade 4

    assert_eq!(
        spinor_at_2pi.grade(),
        2,
        "one full physical turn lands the spinor at −1 — grade 2, a position"
    );
    assert_eq!(
        spinor_at_4pi.grade(),
        0,
        "two full turns bring it home — grade 0"
    );

    // the physical observable is blind to the difference: a vector rotated 2π
    // returns to its base angle, so every measurement of the apparatus reads
    // identity while the spinor sits at −1
    let apparatus = Geonum::new(1.0, 1.0, 5.0);
    let turned = apparatus.rotate(physical_2pi);
    assert_eq!(
        turned.angle.base_angle(),
        apparatus.angle.base_angle(),
        "the apparatus returns at 2π — the projection reads identity"
    );
    assert_eq!(
        turned.angle.blade(),
        apparatus.angle.blade() + 4,
        "while the stored angle carries the turn the projection dropped"
    );
}

#[test]
fn it_cancels_the_interferometer_at_2pi_physical_rotation() {
    // rauch 1975: split a neutron beam, rotate one arm's spin through 2π with
    // a magnetic field, recombine. the beams cancel — the fringe shift proves
    // the 4π period. in geonum the rotated arm sits a π spinor rotation away
    // and the recombination is one addition
    let arm_a = Geonum::new(1.0, 1.0, 8.0); // reference arm
    let arm_b_2pi = arm_a.rotate(Angle::new(2.0, 1.0) / 2.0); // 2π physical = π spinor
    let arm_b_4pi = arm_a.rotate(Angle::new(4.0, 1.0) / 2.0); // 4π physical = 2π spinor

    assert!(
        (arm_a + arm_b_2pi).near_mag(0.0),
        "2π rotation: the arms interfere destructively — the measured minimum"
    );
    assert!(
        (arm_a + arm_b_4pi).near_mag(2.0),
        "4π rotation: full constructive recovery — the measured period"
    );
}

#[test]
fn it_stores_the_spinor_half_angle_as_t() {
    // the spinor parametrization of a rotation α is built on tan(α/2) — the
    // cayley parameter. geonum stores exactly that ratio as t, so the spinor's
    // coordinate is the struct's native field, not a change of variables
    for (p, d) in [(1.0, 5.0), (1.0, 7.0), (2.0, 5.0), (3.0, 7.0)] {
        let alpha = Angle::new(p, d); // rotations within the first quadrant
        let half_tangent = (alpha.grade_angle() / 2.0).tan();
        assert!(
            (alpha.t() - half_tangent).abs() < 1e-15,
            "t IS tan(α/2) — the spinor coordinate, stored"
        );

        // the sandwich RvR† exists to double the half-angle back into the
        // rotation. cos_sin's rational formulas are that doubling — degree 2
        // in t — so the full rotation reads out with no sandwich
        let t = alpha.t();
        let (cos_a, sin_a) = alpha.cos_sin();
        assert!(
            (cos_a - (1.0 - t * t) / (1.0 + t * t)).abs() < 1e-15,
            "cos α = (1−t²)/(1+t²) — the sandwich's double angle, rational in t"
        );
        assert!(
            (sin_a - 2.0 * t / (1.0 + t * t)).abs() < 1e-15,
            "sin α = 2t/(1+t²) — same doubling, same readout"
        );
    }
}

#[test]
fn it_double_covers_by_collapsing_pi_into_a_turn() {
    // SU(2) → SO(3) is 2-to-1: a spinor s and its negative s + π drive the
    // same physical rotation. doubling maps the π gap to a 2π gap — a full
    // turn, invisible to grade — so two distinct spinors, one rotation
    let s = Angle::new(1.0, 5.0);
    let minus_s = s + Angle::new(1.0, 1.0); // the ± partner, π away

    assert_ne!(s, minus_s, "two distinct spinor states");
    assert_eq!(
        (s * 2.0).base_angle(),
        (minus_s * 2.0).base_angle(),
        "doubled, they land the same physical rotation — the cover is 2-to-1"
    );
    assert_eq!(
        (minus_s * 2.0).blade() - (s * 2.0).blade(),
        4,
        "the two preimages differ by exactly one full turn of winding"
    );
}

#[test]
fn it_untangles_two_twists_but_not_one() {
    // the belt trick: a 2π twist in a belt cannot be undone without rotating
    // the ends; a 4π twist can. π₁(SO(3)) = ℤ/2 read as blade parity: each
    // full physical twist is a π spinor rotation — blade 2 — and the
    // obstruction class is the grade the accumulated blades land on
    let twist = Angle::new(2.0, 1.0) / 2.0; // one full twist = π spinor

    let mut belt = Angle::new(0.0, 1.0);
    let expected_class = [2usize, 0, 2, 0]; // odd twists obstructed, even free

    for (n, expected) in expected_class.iter().enumerate() {
        belt = belt + twist;
        assert_eq!(
            belt.grade(),
            *expected,
            "{} twist(s): class {} — parity is the only invariant",
            n + 1,
            expected
        );
    }
}
