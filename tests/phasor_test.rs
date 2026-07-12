// the phasor was always a geonum
//
// electrical engineering teaches complex impedance as a computational trick —
// "pretend the circuit is complex-valued, solve, take the real part" — and
// then spends a curriculum managing the pretense. there is no trick to manage:
// an impedance is a magnitude and an angle, a phasor is [mag, angle], and the
// circuit laws are geonum arithmetic
//
//   - reactance sign is a grade: inductive current lags (quadrant I), the
//     capacitive angle sits at grade 3 — above vs below resonance is a grade
//     flip, not a ± on an imaginary part
//   - resonance is interference: ωL and 1/ωC sit π apart and cancel by the
//     opposite-angle branch of addition, leaving pure resistance
//   - real and reactive power are the adj/opp split already in core: P = S·adj,
//     Q = S·opp, and P² + Q² = S² is the quadrature closing
//   - three-phase power exists because angles interfere: the balanced neutral
//     carries nothing, and the delivered power is ripple-free because three
//     second-harmonic ripples 2π/3 apart wave_sum to zero — constant torque,
//     read off interference
//   - power factor correction is angle arithmetic: inject the opp leg's
//     opposite and the apparent power rotates home to grade 0
//
// run: cargo test --test phasor_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const R: f64 = 50.0; // Ω
const L: f64 = 0.1; // H
const C: f64 = 1e-5; // F — resonance at ω₀ = 1/√(LC) = 1000 rad/s

// series RLC impedance at frequency ω: three geonums summed
fn impedance(omega: f64) -> Geonum {
    let z_l = Geonum::new_with_angle(omega * L, Angle::new(1.0, 2.0)); // ωL at π/2
    let z_c = Geonum::new_with_angle(1.0 / (omega * C), Angle::new(3.0, 2.0)); // 1/ωC at 3π/2
    (z_l + z_c) + Geonum::new(R, 0.0, 1.0)
}

#[test]
fn it_cancels_reactance_at_resonance() {
    // at ω₀ the inductive and capacitive reactances are equal magnitudes π
    // apart — they cancel by interference and the circuit is pure resistance
    let z0 = impedance(1000.0);
    assert!(
        z0.near_mag(R),
        "at resonance |Z| = R — the reactances cancelled"
    );
    assert_eq!(
        z0.angle.grade(),
        0,
        "and the angle is home: pure resistance"
    );
    assert!(z0.angle.near_rem(0.0), "no residual phase");

    // above resonance the inductor wins: net reactance +150 at ω = 2000
    let z_high = impedance(2000.0);
    assert!(
        z_high.near_mag((R * R + 150.0 * 150.0).sqrt()),
        "|Z| = √(50² + 150²)"
    );
    assert_eq!(z_high.angle.grade(), 0, "inductive: quadrant I");
    assert!(
        z_high.angle.near_rem(3.0_f64.atan()),
        "phase = atan(150/50) — current lags"
    );

    // below resonance the capacitor wins: the reactance sign is a GRADE, not a
    // minus on an imaginary part
    let z_low = impedance(500.0);
    assert_eq!(
        z_low.angle.grade(),
        3,
        "capacitive: quadrant IV — the sign flip across resonance is a grade flip"
    );
}

#[test]
fn it_splits_power_into_adj_and_opp() {
    // drive the ω = 2000 circuit with 120 V. apparent power S rides the
    // impedance angle; real and reactive power are its two quadrature legs —
    // the adj/opp split already in core
    let v = 120.0;
    let z = impedance(2000.0);

    // current magnitude by geonum division — the quotient carries the
    // inversion's π (numbers_test lands quotients at grade 2); the magnitude
    // is the reading
    let i = Geonum::new(v, 0.0, 1.0) / z;
    assert!(i.near_mag(v / z.mag), "|I| = |V|/|Z|");

    let s = Geonum::new_with_angle(v * i.mag, z.angle); // apparent power at the impedance angle
    let p = s.adj(); // real power — the aligned leg
    let q = s.opp(); // reactive power — the quadrature leg

    // anchored to dissipation the test never constructed: real power is what
    // the resistor burns (the watts on the bill), reactive is what the net
    // reactance circulates (the VArs the utility penalizes)
    assert!((p.mag - i.mag * i.mag * R).abs() < 1e-9, "P = I²R");
    assert!((q.mag - i.mag * i.mag * 150.0).abs() < 1e-9, "Q = I²X");

    // the power triangle is the quadrature closing
    assert!(
        (p.mag * p.mag + q.mag * q.mag - s.mag * s.mag).abs() < 1e-6,
        "P² + Q² = S²"
    );
}

#[test]
fn it_cancels_the_neutral_by_interference() {
    // three balanced phases 2π/3 apart: the neutral current is their wave_sum,
    // and it vanishes because the angles interfere — not because a law says so
    let balanced: GeoCollection = (0..3)
        .map(|k| Geonum::new_with_angle(10.0, Angle::new(2.0 * k as f64, 3.0)))
        .collect();
    assert!(
        balanced.wave_sum().near_mag(0.0),
        "the balanced neutral carries nothing"
    );

    // unbalance one phase and the neutral carries exactly the imbalance — the
    // extra 2 A of phase 3, pointing along phase 3
    let unbalanced: GeoCollection = [10.0, 10.0, 12.0]
        .iter()
        .enumerate()
        .map(|(k, &mag)| Geonum::new_with_angle(mag, Angle::new(2.0 * k as f64, 3.0)))
        .collect();
    let neutral = unbalanced.wave_sum();
    assert!(neutral.near_mag(2.0), "the neutral reads the 2 A imbalance");
    assert_eq!(
        neutral.angle.base_angle(),
        Angle::new(4.0, 3.0).base_angle(),
        "pointing along the heavy phase"
    );
}

#[test]
fn it_delivers_ripple_free_power_in_three_phase() {
    // per-phase instantaneous power is VI·cos φ plus a double-frequency ripple.
    // the three ripples sit 2π/3 apart and wave_sum to zero at every instant —
    // the motor sees constant torque. this cancellation is why three-phase
    // exists
    let (v, i) = (120.0, 5.0);
    let phi = 0.8_f64.acos(); // power factor 0.8
    let omega = 2.0 * PI * 60.0;

    for t in [0.0, 0.001, 0.004, 0.007, 0.011] {
        // the three ripple terms as geonums at 2ωt − φ + k·2π/3
        let ripples: GeoCollection = (0..3)
            .map(|k| {
                let angle =
                    Angle::new((2.0 * omega * t - phi) / PI, 1.0) + Angle::new(2.0 * k as f64, 3.0);
                Geonum::new_with_angle(v * i, angle)
            })
            .collect();
        assert!(
            ripples.wave_sum().near_mag(0.0),
            "t={t}: the second harmonics interfere to zero"
        );

        // foil: the raw time-domain products sum to the flat 3·VI·cos φ
        let p_total: f64 = (0..3)
            .map(|k| {
                let theta = omega * t - 2.0 * PI * k as f64 / 3.0;
                (2.0_f64).sqrt() * v * theta.cos() * (2.0_f64).sqrt() * i * (theta - phi).cos()
            })
            .sum();
        assert!(
            (p_total - 3.0 * v * i * phi.cos()).abs() < 1e-9,
            "t={t}: total power is flat — the ripple left with the interference"
        );
    }

    // the single-phase foil pulses: its ripple has no partners to cancel with
    let single = |t: f64| {
        let theta = omega * t;
        2.0 * v * i * theta.cos() * (theta - phi).cos()
    };
    let swing = single(phi / (2.0 * omega)) - single(phi / (2.0 * omega) + PI / (2.0 * omega));
    assert!(
        (swing - 2.0 * v * i).abs() < 1e-9,
        "one phase alone swings by 2VI — the pulsating torque three-phase removes"
    );
}

#[test]
fn it_corrects_power_factor_by_rotating_home() {
    // a 0.8-power-factor load draws S = [100, φ]: 80 kW of work, 60 kVAr of
    // circulation. a capacitor injects the opp leg's opposite — [60, 3π/2] —
    // and the sum rotates home to grade 0. same work, less current: the
    // utility's whole business case in one geonum addition
    let phi = 0.8_f64.acos();
    let s_load = Geonum::new_with_angle(100.0, Angle::new(phi / PI, 1.0));
    let capacitor = Geonum::new_with_angle(s_load.opp().mag, Angle::new(3.0, 2.0));

    let s_corrected = s_load + capacitor;

    assert!(s_corrected.near_mag(80.0), "|S| falls to the real power");
    assert!(
        s_corrected.opp().near_mag(0.0),
        "the reactive leg is gone — unity power factor"
    );
    assert!(
        (s_corrected.adj().mag - s_load.adj().mag).abs() < 1e-9,
        "the real power is untouched — same work"
    );
    assert!(
        (s_corrected.mag / s_load.mag - 0.8).abs() < 1e-12,
        "line current drops 20% for the same delivered work"
    );
}
