// aliasing is winding loss
//
// the sampling theorem, anti-aliasing filters, the wagon-wheel effect — the
// whole discipline of not-being-fooled-by-samples exists because a sampler
// projects a rotation onto its base angle and drops the winding:
//
//   - a tone at f and a tone at f + fs produce IDENTICAL samples at rate fs:
//     their per-sample angles differ by whole turns, which is blade — exactly
//     the data the projection cannot hold. the anti-aliasing filter kills the
//     winding before the projection lies about it
//   - the wagon wheel spins backwards on film because the camera reads each
//     frame's base angle by shortest arc: a 0.9-turn forward step reads as
//     0.1 turn backward, and every frame drops exactly one turn into the gap
//     between stored blade and apparent motion
//   - nyquist is a half-turn budget: a per-sample step under π reads
//     faithfully, past π the shortest arc points the wrong way. fs > 2f is
//     the condition that one base angle can carry the step unambiguously
//   - the phase vocoder's unwrap stage — and its "phasiness" artifact — is
//     this winding loss shipping in audio production: the STFT hop drops the
//     blade and the vocoder guesses it back
//
// run: cargo test --test sampling_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// the angle a tone at frequency f has turned by sample k at rate fs
fn sampled(f: f64, k: usize, fs: f64) -> Angle {
    Angle::new(2.0 * f * k as f64 / fs, 1.0)
}

// what a sampler infers from one step: the shortest arc to the base angle —
// the conventional reading, blind to blade
fn shortest_arc(step: Angle) -> f64 {
    let g = step.grade_angle();
    if g > PI {
        g - 2.0 * PI
    } else {
        g
    }
}

#[test]
fn it_aliases_by_dropping_the_winding() {
    // 3 Hz and 13 Hz sampled at 10 Hz: the classic alias pair
    let fs = 10.0;
    let (f_low, f_high) = (3.0, 13.0);

    for k in 1..=8usize {
        let low = sampled(f_low, k, fs);
        let high = sampled(f_high, k, fs);

        // the samples are identical — the projection cannot tell the tones apart
        assert!(
            high.base_angle().near(&low.base_angle()),
            "sample {k}: f and f + fs project to the same reading"
        );

        // the difference was never zero — it sits in the blade, one full turn
        // per sample, the winding the sampler drops
        assert_eq!(
            high.blade() - low.blade(),
            4 * k,
            "sample {k}: the tones differ by exactly k turns of stored winding"
        );
    }
}

#[test]
fn it_spins_the_wagon_wheel_backwards() {
    // a wheel at 9 Hz filmed at 10 fps: 0.9 of a turn per frame. the film
    // reads each frame by shortest arc and sees 0.1 turn BACKWARD
    let per_frame = Angle::new(2.0 * 9.0 / 10.0, 1.0); // 1.8π per frame

    let apparent = shortest_arc(per_frame);
    assert!(
        (apparent + 0.2 * PI).abs() < 1e-12,
        "the screen shows −0.2π per frame — one wheel, spinning the wrong way"
    );

    // one second of film: ten frames. the stored rotation is nine full turns;
    // the apparent rotation is minus one. every frame dropped exactly one turn
    let mut wheel = Angle::new(0.0, 1.0);
    for _ in 0..10 {
        wheel = wheel + per_frame;
    }
    assert_eq!(wheel.blade(), 36, "the wheel turned nine times — blade 36");

    let film_total = 10.0 * apparent;
    assert!(
        (film_total + 2.0 * PI).abs() < 1e-12,
        "the film shows one backward turn"
    );
    let dropped = (wheel.blade() as f64 * PI / 2.0 + wheel.rem()) - film_total;
    assert!(
        (dropped / (2.0 * PI) - 10.0).abs() < 1e-12,
        "ten turns lost — one per frame, the winding the projection discards"
    );
}

#[test]
fn it_bounds_faithful_reading_at_half_a_turn() {
    let fs = 10.0;

    // 4 Hz at 10 Hz: 0.8π per sample — under the half-turn budget, the
    // shortest arc reads the true step
    let under = sampled(4.0, 1, fs);
    assert!(
        (shortest_arc(under) - 0.8 * PI).abs() < 1e-12,
        "below nyquist the reading is faithful"
    );

    // 6 Hz at 10 Hz: 1.2π per sample — past the budget, the shortest arc
    // points backward. fs > 2f is exactly the condition that one base angle
    // carries the step whole: nyquist is a half-turn per sample
    let over = sampled(6.0, 1, fs);
    assert!(
        (shortest_arc(over) + 0.8 * PI).abs() < 1e-12,
        "past nyquist the shortest arc reads −0.8π for a +1.2π step"
    );

    // the two misread tones are alias partners: 6 = 10 − 4 folds onto 4 with
    // the direction flipped — the mirror image the folding frequency creates
    assert!(
        (shortest_arc(over) + shortest_arc(under)).abs() < 1e-12,
        "6 Hz reads as 4 Hz reversed — folded about fs/2"
    );
}
