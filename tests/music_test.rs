// the pythagorean comma is a t residue
//
// tuning theory is angle arithmetic on a log-frequency winding line: one
// octave is one full turn (blade 4), a just fifth is log₂(3/2) of a turn, and
// twenty-five centuries of temperament controversy is bookkeeping on the
// remainder:
//
//   - pitch class vs pitch height is base angle vs blade: octave equivalence
//     reads the angle mod one turn, the octave number IS the winding
//   - stack twelve just fifths and the winding lands exactly seven octaves of
//     blade — but a remainder survives: the pythagorean comma, 3¹²/2¹⁹, the t
//     the lattice cannot absorb
//   - equal temperament is the projection that forces closure: shave each
//     fifth by comma/12 and the circle of fifths closes exactly, twelve
//     lattice steps landing blade 28 dead
//   - beats are the angle-difference frequency: two tones interfere at
//     |f₁ − f₂|, the envelope read off wave_sum with no fourier apparatus
//
// run: cargo test --test music_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// pitch as an angle: one octave = one full turn above the reference A0 = 27.5 Hz
fn pitch(freq: f64) -> Angle {
    Angle::new(2.0 * (freq / 27.5).log2(), 1.0)
}

#[test]
fn it_overshoots_seven_octaves_by_the_comma() {
    // a just fifth is log₂(3/2) of a turn. twelve of them accumulate exactly
    // seven octaves of winding — blade 28 — plus a remainder no octave absorbs
    let just_fifth = Angle::new(2.0 * 1.5_f64.log2(), 1.0);

    let mut stack = Angle::new(0.0, 1.0);
    for _ in 0..12 {
        stack = stack + just_fifth;
    }

    assert_eq!(
        stack.blade(),
        28,
        "twelve fifths wind exactly seven octaves of blade"
    );

    // the survivor is the pythagorean comma — 3¹²/2¹⁹, computed here from the
    // frequency ratio the greeks measured, not from the stack
    let comma = 2.0 * PI * (3.0_f64.powi(12) / 2.0_f64.powi(19)).log2();
    assert!(
        stack.near_rem(comma),
        "the remainder is the comma: {:.6} rad — the t the lattice cannot absorb",
        stack.rem()
    );

    // subtracting seven exact octaves isolates it as its own angle
    let overshoot = stack - Angle::new(14.0, 1.0); // 7 turns = 14π
    assert!(
        overshoot.near_rad(comma),
        "twelve fifths minus seven octaves IS the comma"
    );
}

#[test]
fn it_closes_the_equal_tempered_circle_exactly() {
    // the equal-tempered fifth is the rational lattice point 7/12 turn — 7π/6.
    // twelve of them close: blade 28, remainder zero. the circle of fifths
    // closes because temperament forced the fifth onto the lattice
    let et_fifth = Angle::new(7.0, 6.0);

    let mut circle = Angle::new(0.0, 1.0);
    for _ in 0..12 {
        circle = circle + et_fifth;
    }

    assert_eq!(
        circle.blade(),
        28,
        "the tempered circle winds seven octaves"
    );
    assert!(circle.near_rem(0.0), "and closes dead — no comma");

    // what temperament shaved: each fifth pays exactly comma/12
    let just_fifth = Angle::new(2.0 * 1.5_f64.log2(), 1.0);
    let comma = 2.0 * PI * (3.0_f64.powi(12) / 2.0_f64.powi(19)).log2();
    let shave = just_fifth - et_fifth;
    assert!(
        shave.near_rad(comma / 12.0),
        "equal temperament spreads the comma across the twelve fifths"
    );
}

#[test]
fn it_mods_pitch_class_by_the_winding() {
    // A4 and A5 are one pitch class at two heights: their angles share a base
    // angle and differ by one full turn of blade. the octave number is the
    // winding — blade/4 above the reference
    let a4 = pitch(440.0);
    let a5 = pitch(880.0);

    assert_eq!(
        a4.base_angle(),
        a5.base_angle(),
        "same pitch class — the base angle"
    );
    assert_eq!(a5.blade() - a4.blade(), 4, "one octave = one turn of blade");
    assert_eq!(a4.blade() / 4, 4, "A4 sits four octaves above A0");
    assert_eq!(a5.blade() / 4, 5, "A5 five — the height is the winding");

    // C#5, four equal-tempered semitones up from A4, is a different class:
    // the base angle moves even though the octave stays
    let c_sharp_5 = pitch(440.0 * 2.0_f64.powf(4.0 / 12.0));
    assert_ne!(
        c_sharp_5.base_angle(),
        a4.base_angle(),
        "a different pitch class lands a different base angle"
    );
}

#[test]
fn it_beats_at_the_difference_angle() {
    // 440 Hz against 442 Hz: the superposition's envelope pulses at the 2 Hz
    // difference. each tone is a unit geonum at angle 2πft; the audible
    // amplitude is their interference, 2|cos(π·Δf·t)| — read off wave_sum,
    // no spectrum computed
    let (f1, f2) = (440.0, 442.0);

    for t in [0.05, 0.1, 0.2, 0.25, 0.5] {
        let tones: GeoCollection = [f1, f2]
            .iter()
            .map(|&f| Geonum::new_with_angle(1.0, Angle::new(2.0 * f * t, 1.0)))
            .collect();

        let envelope = 2.0 * (PI * (f2 - f1) * t).cos().abs();
        assert!(
            tones.wave_sum().near_mag(envelope),
            "t={t}: the amplitude rides the difference angle"
        );
    }

    // the trough: a quarter beat-period in, the tones sit π apart and cancel —
    // the silence between beats is destructive interference
    let trough: GeoCollection = [f1, f2]
        .iter()
        .map(|&f| Geonum::new_with_angle(1.0, Angle::new(2.0 * f * 0.25, 1.0)))
        .collect();
    assert!(
        trough.wave_sum().near_mag(0.0),
        "at t = 0.25 s the beat is silent — the arms cancel"
    );
}
