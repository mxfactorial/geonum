// subtraction never cancels catastrophically
//
// an interferometer reads a difference of two enormous phases: LIGO recovers
// nanoradians of signal off ~10^11 turns of optical carrier. numerically that
// is the textbook catastrophic-cancellation setup — subtract two nearly-equal
// huge floats and the signal drowns in representation error — and physics
// already ships the workaround in hardware: the interferometer exists to
// compute the difference OPTICALLY, because no register that stores the total
// can afford the subtraction. numerics ships its own patches for the same
// wound: kahan summation, double-double arithmetic, hand-derived difference
// coordinates
//
// geonum's angle subtraction has no such failure mode: blades subtract as
// exact integers and t borrows rationally, so the difference of two
// hundred-gigaturn windings returns the nanoradian whole — and both totals
// survive, where the photodetector had to destroy them to read the beat
//
// fence, logged: geometric_sub reads t-differences below 1e-10 as equal —
// the position-comparison noise floor. signals above it subtract exactly;
// whispers below it need the storage route (whisper_test)
//
// run: cargo test --test interferometer_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const CARRIER_TURNS: f64 = 280_000_000_000.0; // 1064 nm laser, ~1 ms of cavity storage

#[test]
fn it_recovers_a_nanoradian_off_a_hundred_gigaturn_carrier() {
    // the reference arm: 2.8×10^11 turns of carrier, exact in the blade
    let carrier_blade = 4usize * 280_000_000_000;
    let arm_a = Angle::new_with_blade(carrier_blade, 0.0, 1.0);

    // the signal arm carries 2 nanoradians more — a gravitational wave's
    // worth of phase, sitting above the 1e-10 subtraction floor
    let signal_t = 1e-9_f64; // t = θ/2
    let arm_b = arm_a + Angle::from_parts(0, signal_t);

    // the beat: blades cancel as integers, t returns whole
    let beat = arm_b - arm_a;
    assert_eq!(
        beat.blade(),
        0,
        "10^12 quarter-turns of carrier subtract to exactly zero"
    );
    assert!(
        (beat.t() - signal_t).abs() < 1e-24,
        "the nanoradian returns whole — no precision spent on the carrier"
    );
    assert!(
        ((beat.rem() - 2.0 * signal_t) / (2.0 * signal_t)).abs() < 1e-9,
        "the beat reads 2 nanoradians at full relative precision"
    );
}

#[test]
fn it_subtracts_without_catastrophic_cancellation() {
    // the float registers at the carrier's magnitude: 1.8×10^12 rad, where
    // the next representable phase sits 2.4×10^-4 rad away — eleven orders
    // coarser than the signal. the two arms are bit-identical
    let phase_a = CARRIER_TURNS * 2.0 * PI;
    let signal = 2e-9_f64;
    let phase_b = phase_a + signal;
    assert!(
        phase_b == phase_a,
        "the float arms read identical — the signal vanished into the carrier"
    );
    assert!(
        phase_b - phase_a == 0.0,
        "and their difference is literally zero — cancellation's endpoint"
    );

    // geonum keeps what the hardware workaround destroys: the photodetector
    // reads only the beat, the totals gone into the light. here the signal
    // arm still carries its full winding AND the difference reads exact
    let carrier_blade = 4usize * 280_000_000_000;
    let arm_a = Angle::new_with_blade(carrier_blade, 0.0, 1.0);
    let arm_b = arm_a + Angle::from_parts(0, signal / 2.0);

    assert_eq!(
        arm_b.blade(),
        carrier_blade,
        "the signal arm keeps its hundred-gigaturn total"
    );
    assert!(
        ((arm_b - arm_a).rem() - signal).abs() < 1e-18,
        "while the beat subtracts out exact — total and difference, one register"
    );
}
