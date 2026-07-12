// the multivalued function is a scalar artifact
//
// complex analysis carries a whole apparatus for functions that "take multiple
// values" — branch cuts (choose a ray, tear the plane along it), principal
// values (pick one value, accept a discontinuity), riemann surfaces (glue the
// torn sheets into a spiral staircase so analysis works again). every piece of
// that apparatus re-adds the winding the representation dropped:
//
//   - log z = ln r + iθ is "multivalued" only because θ was stored mod 2π. the
//     angle geonum stores rides the winding line, so log is single-valued on
//     the staircase — the riemann surface IS the blade
//   - the branch cut is where the mod-2π tear lands: the principal value reads
//     identical numbers for a point and that point carried once around the
//     origin, then papers over the collapse with a discontinuity
//   - √z has two sheets because pow(1/2) halves the angle: one loop of the
//     base (blade +4) moves the root by π — the OTHER root. the ± ceremony is
//     one dropped bit of winding, the monodromy group ℤ/2 is blade parity
//   - the cube root cycles ℤ/3: one loop per root, three loops home
//
// run: cargo test --test multivalued_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// the full stored angle — winding included, the coordinate on the staircase
fn total(a: Angle) -> f64 {
    a.blade() as f64 * PI / 2.0 + a.rem()
}

// the principal-value foil: Im(Log z) recovered from cartesian shadows by
// atan2 — the conventional route, blind past one turn
fn principal_arg(z: &Geonum) -> f64 {
    let (c, s) = z.angle.cos_sin();
    (z.mag * s).atan2(z.mag * c)
}

#[test]
fn it_dissolves_the_branch_cut_by_keeping_the_winding() {
    let z = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
    let looped = z.rotate(Angle::new(2.0, 1.0)); // once around the origin

    // the geonum log distinguishes them: Im(log) climbs exactly 2π — the loop
    // is data, blade 4 of it
    assert_eq!(
        looped.angle.blade(),
        z.angle.blade() + 4,
        "one loop = four quarter-turns of stored winding"
    );
    assert!(
        (total(looped.angle) - total(z.angle) - 2.0 * PI).abs() < 1e-12,
        "Im(log) climbs 2π per loop — single-valued on the staircase"
    );

    // the principal value cannot: both points project to identical shadows,
    // so atan2 returns the same argument and the 2π is gone. the branch cut
    // is the tear this collapse forces — a property of the storage, not of log
    assert!(
        (principal_arg(&z) - principal_arg(&looped)).abs() < 1e-15,
        "the principal value reads the loop and the point as one number"
    );
}

#[test]
fn it_walks_the_sqrt_sheets_by_blade_parity() {
    let z = Geonum::new(4.0, 2.0, 3.0); // [4, 2π/3]

    let root = z.pow(0.5);
    assert!(root.near_mag(2.0), "|√z| = 2");
    assert!(root.angle.near(&Angle::new(1.0, 3.0)), "arg √z = π/3");

    // carry z once around the origin and take the root again: the halved
    // angle moves by π — the OTHER square root. the ± that algebra bolts onto
    // √ is one bit of winding, read here as a position
    let other_root = z.rotate(Angle::new(2.0, 1.0)).pow(0.5);
    assert!(other_root.near_mag(root.mag), "same magnitude");
    assert!(
        other_root.angle.is_opposite(&root.angle),
        "one base loop lands the other root — π away"
    );

    // two loops return: the monodromy group ℤ/2 is blade parity
    let twice = z.rotate(Angle::new(4.0, 1.0)).pow(0.5);
    assert_eq!(
        twice.angle.base_angle(),
        root.angle.base_angle(),
        "two loops of the base close the two-sheet cover"
    );
}

#[test]
fn it_cycles_the_cube_roots_one_loop_per_root() {
    let z = Geonum::new(8.0, 1.0, 2.0); // [8, π/2]
    let step = Angle::new(2.0, 3.0); // 2π/3 — the root spacing

    // each loop of the base advances the cube root by one step of the cycle
    let mut expected = z.pow(1.0 / 3.0);
    assert!(expected.near_mag(2.0), "|∛z| = 2");

    for k in 1..=3u32 {
        let root_k = z
            .rotate(Angle::new(2.0 * k as f64, 1.0)) // k loops
            .pow(1.0 / 3.0);
        expected = expected.rotate(step);
        assert_eq!(
            root_k.angle.base_angle(),
            expected.angle.base_angle(),
            "loop {k}: the root cycle advances one step — monodromy ℤ/3"
        );
    }

    // and the third loop is home: the deck closed
    let third = z.rotate(Angle::new(6.0, 1.0)).pow(1.0 / 3.0);
    assert_eq!(
        third.angle.base_angle(),
        z.pow(1.0 / 3.0).angle.base_angle(),
        "three loops of the base close the three-sheet cover"
    );
}

#[test]
fn it_climbs_the_log_staircase_sheet_by_sheet() {
    // the riemann surface of log — "an infinite spiral staircase" — is the
    // external data structure analysis builds to restore the winding it
    // discarded. the stored angle is born on the staircase: sheet k is blade
    // 4k, and Im(log) is the winding coordinate read directly
    let z = Geonum::new(1.0, 1.0, 4.0); // [1, π/4]

    let mut previous = total(z.angle);
    for k in 1..=5usize {
        let sheet_k = z.rotate(Angle::new(2.0 * k as f64, 1.0));

        assert_eq!(
            sheet_k.angle.base_angle(),
            z.angle.base_angle(),
            "sheet {k}: every sheet projects to the same shadow"
        );
        assert_eq!(
            sheet_k.angle.blade(),
            z.angle.blade() + 4 * k,
            "sheet {k}: the sheet number is the blade"
        );

        let height = total(sheet_k.angle);
        assert!(
            (height - previous - 2.0 * PI).abs() < 1e-9,
            "sheet {k}: each step of the staircase is one 2π riser"
        );
        previous = height;
    }
}
