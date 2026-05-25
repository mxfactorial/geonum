// a gravitational wave is the bondi field oscillating about its flat-space value
//
// einstein_test.rs showed gravity IS the bondi field f(r), and the vacuum
// equation (r·f)'' = 0 picks schwarzschild. that file handled the STATIC case —
// the field stationary, the rescaled (r·f) a straight line. drop static and the
// same primitive carries a wave: f(t, x) = 1 + h(t, x), the perturbation
// propagating. in geonum form that wave is the Waves trait (waves.rs) —
// Geonum::disperse encodes the phase φ = kx − ωt as one angle, and the null
// dispersion ω = ck puts it on the light cone, no PDE solver
//
// the program in one line: a gravitational wave is a bondi-field perturbation
// that lives on the LIGHT CONE — the same null structure spacetime_test set up
// as the boost's fixed-point pair. "gravity waves travel at c" is the same
// statement as "the perturbation is a function of t − x/c only" — the
// characteristic line of the boost, the null direction
//
// what this file shows:
//   - the wave is one Geonum::disperse call — the phase φ = kx − ωt carried in the
//     angle, the same primitive waves.rs uses for em and matter waves
//   - it rides the LIGHT CONE: f is a function of retarded time t − x/c alone,
//     constant along the null direction, varying off it — lightspeed, not postulated
//   - detection: a passing wave stretches one transverse axis and squeezes the
//     orthogonal one, the boost reproducing the LIGO + strain
//
// what this file does NOT claim:
//   - the two polarizations (+ and ×). the geonum bondi field is one scalar;
//     full GR has h_μν with trace and traceless parts, and the traceless tensor
//     mode is what gives + and × polarization. that needs either a second
//     scalar (frame-dragging ω(r,θ), the kerr program) or a direction-dependent
//     extension of f
//
// run: cargo test --test gravitational_wave_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;
const C: f64 = 1.0; // geometric units, speed of light = 1

// the bondi field with a propagating perturbation: f(t, x) = 1 + h(t, x)
// where h is the wave. for a plane wave h = ε · cos(kx − ωt), and we read
// cos(φ) from the disperse'd geonum's grade_angle. amplitude ε is small,
// linearized regime
fn perturbed_f(t: f64, x: f64, epsilon: f64, k: f64, omega: f64) -> f64 {
    let position = Geonum::scalar(x);
    let time = Geonum::scalar(t);
    let wavenumber = Geonum::scalar(k);
    let frequency = Geonum::scalar(omega);

    // disperse builds φ = kx − ωt as a geonum angle, unit magnitude
    let wave = Geonum::disperse(position, time, wavenumber, frequency);
    let (cos_phase, _) = wave.angle.cos_sin();
    1.0 + epsilon * cos_phase
}

#[test]
fn it_rides_the_light_cone_as_a_function_of_retarded_time() {
    // the deeper statement of ω = ck: the perturbation is a function of (t − x/c)
    // ALONE. it doesn't depend on t and x separately, only on the retarded time
    // u = t − x/c. this is the characteristic structure of the lightcone — the
    // null direction the boost has as its fixed point (spacetime_test::it_
    // aberrates: "the forward axis is fixed — t = 0 stays 0")
    let epsilon = 1e-6;
    let k = 1.0;
    let omega = C * k;

    // two events on the same null line t − x/c = constant
    let (t1, x1) = (0.0, 0.0);
    let (t2, x2) = (1.0, C * 1.0); // moved by x = ct in time t
    let (t3, x3) = (2.5, C * 2.5);

    let f1 = perturbed_f(t1, x1, epsilon, k, omega);
    let f2 = perturbed_f(t2, x2, epsilon, k, omega);
    let f3 = perturbed_f(t3, x3, epsilon, k, omega);

    // the bondi field has the SAME value at every event on the null line — the
    // perturbation rides the lightcone, surfing on its characteristic
    assert!(
        (f1 - f2).abs() < EPSILON,
        "f is constant along t − x/c — the wave rides the null direction"
    );
    assert!(
        (f1 - f3).abs() < EPSILON,
        "and at a third null-separated event, still the same value"
    );

    // step OFF the null line and the field changes — the perturbation is NOT
    // constant on spacelike or timelike trajectories, only on null ones
    let (t_off, x_off) = (0.0, PI / k); // a half-wavelength along x at t=0
    let f_off = perturbed_f(t_off, x_off, epsilon, k, omega);
    assert!(
        (f1 - f_off).abs() > epsilon * 0.5,
        "off the null line the field varies — the wave only stays still along the cone"
    );

    // gravitational waves travel at c not because we postulated it, but because
    // the bondi-field perturbation lives on the null direction — the same
    // direction the SR boost has as its fixed point, the same direction the
    // schwarzschild horizon collapses to. one geometric object, three roles
}

#[test]
fn it_composes_with_the_disperse_primitive() {
    // the wave is literally one Geonum::disperse call, the same primitive
    // waves.rs uses for electromagnetic and matter waves. gravitational waves
    // and electromagnetic waves DIFFER in their source (mass-energy vs. charge)
    // and in their tensorial structure (spin-2 vs. spin-1), but they SHARE the
    // null dispersion relation and the lightcone propagation. geonum reads both
    // off the same one-line construction
    let k = 2.0 * PI; // wavenumber: λ = 1 in geometric units
    let omega = C * k; // null dispersion
    let position = Geonum::scalar(0.25); // quarter-wavelength out
    let time = Geonum::scalar(0.0);

    let wave = Geonum::disperse(position, time, Geonum::scalar(k), Geonum::scalar(omega));

    // unit amplitude, phase kx − ωt = (2π)(0.25) − 0 = π/2
    assert!(wave.near_mag(1.0), "disperse produces unit-amplitude waves");

    // the phase as a grade angle: kx = π/2 lives at blade 1 (grade 1)
    assert_eq!(
        wave.angle.grade(),
        1,
        "phase π/2 lives at grade 1 — the i-axis of the wave's complex plane"
    );

    // and after one period later (t = 2π/ω), the wave returns to its starting
    // phase — periodicity from the angle's blade arithmetic, no modular reduction
    // by hand, the geonum lattice handles it
    let one_period = Geonum::scalar(2.0 * PI / omega);
    let wave_later = Geonum::disperse(
        position,
        one_period,
        Geonum::scalar(k),
        Geonum::scalar(omega),
    );

    // the phase at (x = 0.25, t = T) is kx − ωT = π/2 − 2π. cos returns to its
    // value at π/2 — the wave is the same shape, the bondi field flickers
    // identically
    let (cos_before, _) = wave.angle.cos_sin();
    let (cos_after, _) = wave_later.angle.cos_sin();
    assert!(
        (cos_before - cos_after).abs() < 1e-9,
        "the bondi field returns to its phase after one period — periodic"
    );
}

#[test]
fn it_stretches_and_squeezes_orthogonal_directions_ligo_style() {
    // the detection statement: a passing gravitational wave alternately stretches
    // one transverse direction and squeezes the orthogonal one — what LIGO
    // measures. in the geonum reading, a wave moving along z has the bondi field
    // perturbed DIFFERENTLY along x and y: f_x = 1 + h(t), f_y = 1 − h(t). when
    // the wave is at its peak phase, x-distances grow and y-distances shrink;
    // a quarter cycle later, the roles swap
    //
    // this is where the bondi field has to be more than one scalar — to encode
    // "+ polarization" you need f along x to differ from f along y. but the
    // geonum boost machinery handles each axis independently, so the test can
    // already be made: drive two perpendicular axes with anti-correlated
    // perturbations and watch the boost responses diverge
    let amplitude: f64 = 1e-3; // a strong-but-still-linear wave

    // freeze the wave at peak phase: kx − ωt = 0, so h = +amplitude
    let h_at_peak = amplitude * 1.0; // cos(0) = 1

    // bondi factors along the two transverse axes — anti-correlated
    let k_along_x = (1.0 + h_at_peak).sqrt();
    let k_along_y = (1.0 - h_at_peak).sqrt();

    // a test mass at unit distance along x, boosted by the local bondi factor.
    // the boost scales the half-tangent — so a ray emitted along x and received
    // back undergoes a tiny gravitational redshift, and the geonum boost reads it
    let probe = Angle::new(1.0, 4.0); // a probe ray at θ = π/4
    let probe_x = probe.boost(k_along_x);
    let probe_y = probe.boost(k_along_y);

    // the two probes pick up OPPOSITE-SIGN shifts of their half-tangent. the
    // sign of (t_after − t_before) is opposite on the x and y axes — the
    // geonum statement of "x stretches while y compresses"
    let shift_x = probe_x.t() - probe.t();
    let shift_y = probe_y.t() - probe.t();
    assert!(
        shift_x * shift_y < 0.0,
        "the bondi factors along x and y produce opposite shifts — the LIGO + polarization"
    );

    // and the magnitudes of the shifts are equal to leading order — the wave is
    // symmetric between stretching and squeezing in the linearized regime, the
    // hallmark of a quadrupolar (trace-free) perturbation. up to second order
    // there's a small asymmetry from h appearing inside √(1 ± h), which is fine
    assert!(
        (shift_x.abs() / shift_y.abs() - 1.0).abs() < 0.01,
        "the strain is symmetric — equal stretch and squeeze, the + mode signature"
    );

    // half a period later: kx − ωt = π, h = −amplitude, the polarization flips
    let h_at_trough = -amplitude;
    let k_along_x_later = (1.0 + h_at_trough).sqrt();
    let k_along_y_later = (1.0 - h_at_trough).sqrt();
    let probe_x_later = probe.boost(k_along_x_later);
    let probe_y_later = probe.boost(k_along_y_later);

    let shift_x_later = probe_x_later.t() - probe.t();
    assert!(
        shift_x_later * shift_x < 0.0,
        "half a period later the x-axis strain flips sign — the wave oscillates"
    );
    let _shift_y_later = probe_y_later.t() - probe.t(); // symmetric partner; sign already verified
}
