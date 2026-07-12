// the smith chart is a protractor for a geonum
//
// microwave engineering's most arcane artifact — the smith chart, a paper disk
// of nested circles engineers rotate rulers on — is a graphical calculator for
// one geonum: the reflection coefficient Γ = (Z − Z₀)/(Z + Z₀). every chart
// move is an operation this library ships:
//
//   - moving along a lossless line rotates Γ by 2βl — the chart's rim scale
//     "wavelengths toward generator" is a protractor, and the impedance
//     repeating every λ/2 is the rotation completing a turn. the blade counts
//     the half-wavelengths of line walked
//   - the quarter-wave transformer is a half turn: Γ → −Γ inverts impedance
//     to Z₀²/Z — matching networks are rotations
//   - a shorted stub dials pure reactance with length: |Γ| = 1 pins it to the
//     rim, so no resistance can appear, and the λ/4 point turns a short into
//     an open
//   - the constant-VSWR circles are magnitude level sets: rotation moves along
//     them, and the standing wave the SWR meter reads is the interference of
//     the incident and reflected geonums
//
// run: cargo test --test smith_chart_test -- --show-output

use geonum::*;

const Z0: f64 = 50.0; // line impedance, the chart's center

// reflection coefficient of a real load: Γ = (Z_L − Z₀)/(Z_L + Z₀)
fn reflection(z_load: f64) -> Geonum {
    let num = Geonum::new(z_load, 0.0, 1.0) - Geonum::new(Z0, 0.0, 1.0);
    let den = Geonum::new(z_load, 0.0, 1.0) + Geonum::new(Z0, 0.0, 1.0);
    Geonum::new_with_angle(
        num.mag / den.mag,
        num.angle.base_angle() - den.angle.base_angle(),
    )
}

// impedance back from Γ: Z = Z₀(1 + Γ)/(1 − Γ), assembled by geonum addition
// and read by angle subtraction. Div answers the transformation question —
// inversion event included, quotients landing grade 2 (numbers_test) — while
// the chart asks where the ratio points
fn impedance(gamma: Geonum) -> Geonum {
    let num = Geonum::scalar(1.0) + gamma;
    let den = Geonum::scalar(1.0) - gamma;
    Geonum::new_with_angle(Z0 * num.mag / den.mag, num.angle - den.angle)
}

// walking l wavelengths toward the generator rotates Γ clockwise by 2βl —
// Γ·e^(−2jβl), the rim scale's arrow. forward-only geometry writes the
// clockwise move as the complementary forward turn, so the walk accumulates
// winding
fn toward_generator(gamma: Geonum, wavelengths: f64) -> Geonum {
    gamma.rotate(Angle::new(-4.0 * wavelengths, 1.0))
}

#[test]
fn it_repeats_impedance_every_half_wavelength() {
    let gamma = reflection(100.0); // |Γ| = 1/3 at angle 0
    assert!(gamma.near_mag(1.0 / 3.0), "Γ of a 2:1 mismatch is 1/3");

    // walk the line in quarter-wave hops: after two hops — half a wavelength —
    // the chart has turned once and the load reappears
    let half = toward_generator(toward_generator(gamma, 0.25), 0.25);
    assert!(
        impedance(half).near_mag(100.0),
        "λ/2 down the line the load reappears"
    );
    assert_eq!(
        half.angle.blade() - gamma.angle.blade(),
        4,
        "one turn stored per half wavelength — the walk's odometer"
    );

    // six hops: 3λ/2, same reading, three turns of winding
    let mut walked = gamma;
    for _ in 0..6 {
        walked = toward_generator(walked, 0.25);
    }
    assert!(impedance(walked).near_mag(100.0), "3λ/2: same reading");
    assert_eq!(
        walked.angle.blade() - gamma.angle.blade(),
        12,
        "three turns stored — the line length never left the data"
    );
}

#[test]
fn it_inverts_impedance_with_a_quarter_wave_half_turn() {
    // λ/4 rotates Γ by π: Γ → −Γ, and Z₀(1−Γ)/(1+Γ) = Z₀²/Z_L — the
    // transformer inverts through the chart's center
    let gamma = reflection(100.0);
    let quarter = toward_generator(gamma, 0.25);

    assert!(
        quarter.angle.is_opposite(&gamma.angle.base_angle()),
        "a quarter wave is a half turn of Γ"
    );
    assert!(
        impedance(quarter).near_mag(Z0 * Z0 / 100.0),
        "Z_in = Z₀²/Z_L = 25 Ω — inversion by rotation"
    );

    // the classic matching move: insert a λ/4 section of Z₀' = √(50·100).
    // in that section Γ' = (100 − 70.7)/(100 + 70.7); the half turn lands the
    // input at exactly 50 Ω — matched, by one rotation
    let z_section = (50.0_f64 * 100.0).sqrt();
    let num = 100.0 - z_section;
    let den = 100.0 + z_section;
    let gamma_section = Geonum::new(num / den, 0.0, 1.0);
    let turned = gamma_section.rotate(Angle::new(1.0, 1.0));

    let matched = Geonum::scalar(1.0) + turned;
    let reflected = Geonum::scalar(1.0) - turned;
    let z_in = z_section * matched.mag / reflected.mag;
    assert!(
        (z_in - 50.0).abs() < 1e-9,
        "the λ/4 transformer lands 50 Ω dead — the match is a half turn"
    );
}

#[test]
fn it_turns_a_short_into_an_open_and_dials_reactance_with_length() {
    let gamma_short = Geonum::new(1.0, 1.0, 1.0); // Z = 0 → Γ = −1, the rim's left pole
    let gamma_open = Geonum::new(1.0, 0.0, 1.0); // Z = ∞ → Γ = +1, the right pole

    // a shorted λ/4 stub reads open: the half turn swaps the poles
    let stub_quarter = toward_generator(gamma_short, 0.25);
    assert_eq!(
        stub_quarter.angle.base_angle(),
        gamma_open.angle.base_angle(),
        "the shorted quarter-wave stub looks open — pole to pole in one half turn"
    );

    // a shorted λ/8 stub is a pure +j50 inductance: |Γ| = 1 keeps the point on
    // the rim, so resistance cannot appear — length dials reactance and only
    // reactance. the impedance lands grade 1, the reactive axis
    let stub_eighth = toward_generator(gamma_short, 0.125);
    let z_stub = impedance(stub_eighth);
    assert!(z_stub.near_mag(Z0), "|Z| = Z₀·tan(π/4) = 50");
    assert_eq!(
        z_stub.angle.grade(),
        1,
        "pure reactance — the rim admits no resistive component"
    );
}

#[test]
fn it_holds_vswr_on_a_magnitude_circle() {
    // lossless line motion is rotation, so |Γ| is invariant — the chart's
    // constant-VSWR circles are magnitude level sets
    let gamma = reflection(100.0);
    for wavelengths in [0.05, 0.11, 0.23, 0.4] {
        assert!(
            toward_generator(gamma, wavelengths).near_mag(gamma.mag),
            "rotation never leaves the |Γ| circle"
        );
    }

    // the standing wave is the interference of incident and reflected waves:
    // V(l) = [1, βl] + [|Γ|, −βl]. aligned they peak at 1 + |Γ|, opposed they
    // dip to 1 − |Γ| — the meter's VSWR is the interference ratio
    let peak = Geonum::new(1.0, 0.0, 1.0) + Geonum::new_with_angle(gamma.mag, Angle::new(0.0, 1.0));
    let dip = Geonum::new(1.0, 1.0, 2.0) + Geonum::new_with_angle(gamma.mag, Angle::new(3.0, 2.0));

    let vswr = peak.mag / dip.mag;
    assert!(
        (vswr - 2.0).abs() < 1e-12,
        "VSWR = (1 + 1/3)/(1 − 1/3) = 2 — the 2:1 mismatch read as interference"
    );
}
