// bode is one plot, nyquist is a winding
//
// classical control theory splits every transfer function into two charts —
// gain in decibels, phase in degrees — because the formalism split magnitude
// from angle at the start. G(jω) is one geonum per frequency:
//
//   - gain margin and phase margin are the magnitude and angle of that one
//     object measured against [1, π] — the two "margins" are the two
//     components of a single distance to instability
//   - the nyquist stability criterion is a winding count pointed at −1: as ω
//     sweeps, the direction of G + 1 walks the grade cycle, and the net walks
//     ARE the closed-loop unstable pole count — no angle unwrapped, no
//     characteristic polynomial solved
//   - a complex pole pair σ ± jω is a scale_rotate generator: the taxonomy of
//     fixed points — node, center, spiral — is which of the two knobs is
//     turned. the step-response spiral is the spiral
//
// run: cargo test --test control_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// the open-loop denominator of G(s) = 1/(s(s+1)(0.5s+1)) at s = jω,
// assembled from geonums: jω times two first-order factors
fn loop_denominator(omega: f64) -> Geonum {
    let jw = Geonum::new_with_angle(omega, Angle::new(1.0, 2.0));
    let f1 = Geonum::new_from_cartesian(1.0, omega);
    let f2 = Geonum::new_from_cartesian(1.0, 0.5 * omega);
    jw * f1 * f2
}

// G(jω) as one geonum: reciprocal magnitude, angle read by Angle subtraction.
// Div answers a different question — what transformation carries the
// denominator to the numerator, inversion event included (numbers_test lands
// quotients at grade 2). the nyquist curve asks where the ratio POINTS, and
// angle subtraction is that question's operator
fn transfer(omega: f64) -> Geonum {
    let d = loop_denominator(omega);
    Geonum::new_with_angle(1.0 / d.mag, Angle::new(0.0, 1.0) - d.angle)
}

#[test]
fn it_reads_gain_and_phase_margin_off_one_geonum() {
    // phase crossover: ∠G = −π where atan(ω) + atan(ω/2) = π/2, i.e. ω = √2
    // (the product of the two tangents is 1 — the crossover is exact, not
    // searched). the geonum lands grade 2 with nothing left over
    let g_pc = transfer(2.0_f64.sqrt());
    assert_eq!(
        g_pc.angle.grade(),
        2,
        "at ω = √2 the loop points at −1's ray"
    );
    assert!(g_pc.angle.near_rem(0.0), "exactly π — no residual phase");

    // gain margin: |G| there is 1/3, so the loop tolerates a gain of 3 before
    // the point reaches −1 — the textbook value for this plant
    assert!(g_pc.near_mag(1.0 / 3.0), "|G(j√2)| = 1/3");
    let gain_margin = 1.0 / g_pc.mag;
    assert!(
        (gain_margin - 3.0).abs() < 1e-9,
        "gain margin = 3 — the 9.5 dB the certification report states"
    );

    // gain crossover: bisect |G| = 1 (monotone in ω), then the phase margin is
    // the angle distance from G to [1, π] — one subtraction
    let (mut lo, mut hi) = (0.1, 2.0);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if transfer(mid).mag > 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let omega_gc = 0.5 * (lo + hi);
    let g_gc = transfer(omega_gc);
    assert!(g_gc.near_mag(1.0), "unit gain at the crossover");

    let phase_margin = Angle::new(1.0, 1.0) - loop_denominator(omega_gc).angle;
    let foil = PI - (PI / 2.0 + omega_gc.atan() + (0.5 * omega_gc).atan());
    assert!(
        phase_margin.near_rad(foil),
        "phase margin = π − ∠denominator, one angle subtraction"
    );
    assert!(
        (foil - 0.5693).abs() < 1e-3,
        "≈ 32.6° — the plant's known margin, read off one geonum"
    );
}

// the nyquist example plant's denominator: (1 + jω)³ for G(s) = K/(s+1)³
fn cubic_denominator(omega: f64) -> Geonum {
    let d = Geonum::new_from_cartesian(1.0, omega);
    d * d * d
}

// where the ratio K/(1+jω)³ POINTS: reciprocal magnitude, angle subtraction —
// the oscilloscope reading of gain and phase
fn ratio_pointing(k: f64, omega: f64) -> Geonum {
    let d3 = cubic_denominator(omega);
    Geonum::new_with_angle(k / d3.mag, Angle::new(0.0, 1.0) - d3.angle)
}

// winding number of g(ω) + 1 around the origin as ω sweeps the whole real
// line — the nyquist contour, sampled through ω = tan(u) so one linear grid
// covers (−∞, ∞). no scalar angle is unwrapped: each sample's direction has a
// quadrant address — its grade — and as the curve turns, the grade walks the
// four-step cycle. forward steps count +1 quarter, backward steps −1, and the
// winding is the net walks around the cycle. the G-builder comes in as a
// closure: the machinery counts whatever loop closure the operator inside it
// asks about
fn nyquist_winding(g: impl Fn(f64) -> Geonum) -> i32 {
    let samples = 200_000;
    let mut quarters: i32 = 0;
    let mut prev: Option<usize> = None;

    for i in 0..=samples {
        let u = -PI / 2.0 + 1e-4 + (PI - 2e-4) * i as f64 / samples as f64;
        let omega = u.tan();

        let w = g(omega) + Geonum::scalar(1.0); // the distance-to-(−1) vector

        let grade = w.angle.grade();
        if let Some(p) = prev {
            match (grade + 4 - p) % 4 {
                1 => quarters += 1, // the direction stepped forward a quadrant
                3 => quarters -= 1, // and here it stepped back
                2 => unreachable!("quarter-step ambiguity — sample denser"),
                _ => {}
            }
        }
        prev = Some(grade);
    }

    // the sweep starts and ends pointing at 1 + 0, so the walk closes on the
    // grade cycle — whole turns only, the count intrinsically an integer
    assert_eq!(quarters % 4, 0, "the contour closes on the grade cycle");
    quarters / 4
}

#[test]
fn it_counts_encirclements_of_minus_one_as_the_winding_number() {
    // G(s) = K/(s+1)³ closes the loop at 1 + G = 0, i.e. (s+1)³ = −K. the
    // closed-loop poles sit at s = −1 + K^(1/3)·(cube roots of −1): for K = 2
    // all three stay left of the axis; for K = 20 a conjugate pair crosses to
    // 0.357 ± 2.35j — unstable. the winding number reads that count off the
    // swept angle, no cubic solved
    assert_eq!(
        nyquist_winding(|w| ratio_pointing(2.0, w)),
        0,
        "K = 2: no encirclement — the closed loop is stable"
    );
    assert_eq!(
        nyquist_winding(|w| ratio_pointing(20.0, w)),
        -2,
        "K = 20: two clockwise encirclements — the two unstable poles, counted by winding"
    );
}

#[test]
fn it_counts_the_other_closure_when_div_builds_the_curve() {
    // Div and angle subtraction at one frequency: same ratio magnitude, but
    // Div's quotient is the pointing ratio's −conjugate — real part negated,
    // imaginary part kept. not a different number for the same thing; a
    // different thing
    let omega = 0.7;
    let pointing = ratio_pointing(2.0, omega);
    let quotient = Geonum::scalar(2.0) / cubic_denominator(omega);

    let (c_p, s_p) = pointing.angle.cos_sin();
    let (c_q, s_q) = quotient.angle.cos_sin();
    assert!(
        quotient.near_mag(pointing.mag),
        "same magnitude — the ratio"
    );
    assert!((c_q + c_p).abs() < 1e-12, "real part negated");
    assert!(
        (s_q - s_p).abs() < 1e-12,
        "imaginary part kept — Div hands over the −conj curve"
    );

    // hand each build to the same winding machinery. the pointing closure
    // counts the negative-feedback closure 1 + G = 0. the Div closure counts
    // the POSITIVE-feedback closure 1 − G = 0, whose characteristic
    // (s+1)³ = +K keeps one RHP root (at K^(1/3) − 1) for every K > 1

    // K = 2: negative feedback comfortably stable — positive feedback already broken
    assert_eq!(
        nyquist_winding(|w| ratio_pointing(2.0, w)),
        0,
        "pointing closure: 1 + G stable at K = 2"
    );
    assert_eq!(
        nyquist_winding(|w| Geonum::scalar(2.0) / cubic_denominator(w)),
        1,
        "Div closure: 1 − G already lost a pole at K = 2 — counted, mirror-oriented"
    );

    // K = 20: negative feedback loses its conjugate pair; the positive-feedback
    // count is unmoved — still the one real crossing
    assert_eq!(
        nyquist_winding(|w| ratio_pointing(20.0, w)),
        -2,
        "pointing closure: two poles crossed at K = 20"
    );
    assert_eq!(
        nyquist_winding(|w| Geonum::scalar(20.0) / cubic_denominator(w)),
        1,
        "Div closure: 1 − G holds its single RHP root at K = 20"
    );

    // K = 1/2: below unity gain both closures hold and the operators agree
    assert_eq!(
        nyquist_winding(|w| ratio_pointing(0.5, w)),
        0,
        "pointing closure: stable below unity gain"
    );
    assert_eq!(
        nyquist_winding(|w| Geonum::scalar(0.5) / cubic_denominator(w)),
        0,
        "Div closure: stable below unity gain — no question left to disagree on"
    );

    // Div never computed a wrong nyquist — it answered the question it always
    // answers. pick the operator by the loop closure you are asking about
}

#[test]
fn it_spirals_the_pole_pair_with_scale_rotate() {
    // a pole pair σ ± jω generates e^(σt)·(rotation at ω): per timestep that
    // is scale_rotate(e^(σΔt), ωΔt) — one knob for the envelope, one for the
    // oscillation. the phase-portrait taxonomy is which knob is turned
    let dt = 0.01_f64;
    let steps = 300;
    let x0 = Geonum::new(2.0, 1.0, 6.0);

    // spiral: σ = −0.5, ω = 3 — both knobs
    let (sigma, omega) = (-0.5, 3.0);
    let mut x = x0;
    for _ in 0..steps {
        x = x.scale_rotate((sigma * dt).exp(), Angle::new(omega * dt / PI, 1.0));
    }
    let t = steps as f64 * dt;
    assert!(
        x.near_mag(x0.mag * (sigma * t).exp()),
        "the envelope is e^(σt) — the scale knob"
    );
    assert!(
        (x.angle - x0.angle).near(&Angle::new(omega * t / PI, 1.0)),
        "the oscillation is ωt of accumulated angle — the rotate knob, winding kept"
    );

    // center: σ = 0 — pure rotation, the orbit never decays
    let mut c = x0;
    for _ in 0..steps {
        c = c.scale_rotate(1.0, Angle::new(omega * dt / PI, 1.0));
    }
    assert!(c.near_mag(x0.mag), "center: the magnitude knob untouched");

    // node: ω = 0 — pure scale, the trajectory never turns
    let mut n = x0;
    for _ in 0..steps {
        n = n.scale_rotate((sigma * dt).exp(), Angle::new(0.0, 1.0));
    }
    assert_eq!(n.angle, x0.angle, "node: the angle knob untouched");
    assert!(
        n.near_mag(x0.mag * (sigma * t).exp()),
        "node: all the motion is in the magnitude"
    );
}
