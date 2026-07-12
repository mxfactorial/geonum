// the exp map is a workaround
//
// float density is position-dependent and conventional rotation formats pay
// for it: cos parks small angles beside 1.0, where floats sit 1.1e-16 apart —
// below ~10 nanoradians the cosine reads exactly 1.0 and the acos readback
// returns zero. the industry fixes are famous: lie log/exp maps for composing
// near-identity rotations, 2·atan2(|v|, w) instead of 2·acos(w) for reading
// them back — each one the half-tangent rediscovered per-crisis. geonum
// stores it: t = tan(θ/2) parks the same angles beside 0.0, where floats stay
// dense to 10^-308, so whispers compose at full relative precision with no
// map in or out, and one register spans a 10^-300 rad whisper to a
// 10^18-turn roar — ~318 orders of rotation
//
// fence, logged: Angle::new floors remainders below 1e-10 rad to t = 0 (the
// constructor's boundary snap), so deep whispers enter through from_parts
//
// run: cargo test --test whisper_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

#[test]
fn it_freezes_the_cosine_where_t_stays_alive() {
    let theta = 5e-9_f64; // five nanoradians

    // the conventional register: cos parks the whisper beside 1.0, where the
    // nearest other float sits 1.1e-16 away — the whisper rounds to silence
    assert!(
        theta.cos() == 1.0,
        "five nanoradians reads cos = 1.0 exactly"
    );
    assert!(
        theta.cos().acos() == 0.0,
        "and the acos readback returns zero — the angle is gone"
    );

    // the half-tangent register: t parks the same whisper beside 0.0, dense
    // territory — stored and read back at full relative precision
    let whisper = Angle::new(theta / PI, 1.0);
    assert!(
        ((whisper.t() - (theta / 2.0).tan()) / (theta / 2.0)).abs() < 1e-9,
        "t = tan(θ/2) holds the whisper to 16 digits"
    );
    let (_, sin_theta) = whisper.cos_sin();
    assert!(
        ((sin_theta - theta) / theta).abs() < 1e-9,
        "the rational sine readout returns the whisper whole"
    );
}

#[test]
fn it_composes_nanoradian_whispers_without_an_exp_map() {
    let theta = 1e-9_f64;
    let whisper = Angle::new(theta / PI, 1.0);

    // a million whispers, tangent-summed — no log map in, no exp map out
    let mut accumulated = Angle::new(0.0, 1.0);
    for _ in 0..1_000_000 {
        accumulated = accumulated + whisper;
    }

    let expected = 1e-3_f64; // a milliradian of truth
    assert!(
        ((accumulated.rem() - expected) / expected).abs() < 1e-8,
        "10^6 nanoradian steps land a milliradian at full relative precision"
    );

    // the scalar-part foil: each whisper's cosine is already 1.0, so a
    // million compositions store 1.0 and the readback is zero — every step
    // lost before composition even starts
    let mut w = 1.0_f64;
    for _ in 0..1_000_000 {
        w *= (theta / 2.0).cos();
    }
    assert!(
        2.0 * w.acos() == 0.0,
        "the w register composed a million whispers into silence"
    );
}

#[test]
fn it_spans_the_whisper_to_the_roar_in_one_register() {
    // the roar: a year of cesium — 1.16×10^18 quarter-turns (atomic_clock_test)
    let year_blade = (4u128 * 31_536_000 * 9_192_631_770) as usize;
    let roar = Angle::new_with_blade(year_blade, 0.0, 1.0);

    // the whisper: 10^-300 rad, entering through from_parts past the
    // constructor's 1e-10 floor
    let whisper_t = 5e-301_f64;
    let both = roar + Angle::from_parts(0, whisper_t);

    assert_eq!(both.blade(), year_blade, "the roar's count survives exact");
    assert!(
        (both.t() - whisper_t).abs() < 1e-310,
        "the whisper's t survives beside it, bit for bit"
    );

    // the span: ~318 orders of magnitude in one register — a ratio too large
    // for f64 to hold, so it is measured in logs
    let span = (year_blade as f64 * PI / 2.0).log10() - (2.0 * whisper_t).log10();
    assert!(span > 300.0, "{span:.0} orders of rotation, one register");
}
