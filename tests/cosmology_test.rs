// cosmology, as geometry: each measurement is one clean angle-native law, no free parameter to
// tune. read the measurement as the geometry it already is and the law writes itself — the
// account someone can take and run against the full catalogs.
//
//   redshift   is a boost           the measured frequency ratio is the Bondi factor k = e^φ
//   distance   is the rapidity      comoving distance is (c/H0)·ln(1+z)
//   velocity   is its projection    β = tanh(φ), the bounded shadow of the boost
//   redshift   composes by product  Bondi factors multiply along a path, rapidities add
//   expansion  is magnitude growth  the scale factor multiplies lengths, the angle fixed
//
// dark energy, stated rather than stepped around, is not a substance: it is the gap between the
// coasting distance (distance is the rapidity, below) and a matter-loaded decelerating model.
// this geometry carries no Λ, only the rapidity, and the gap closes. a residual relabeled as a
// substance — research in progress wearing a noun.

use geonum::*;

const C_KM_S: f64 = 299_792.458; // km/s
const H0: f64 = 70.0; // km/s/Mpc
const HUBBLE_DISTANCE: f64 = C_KM_S / H0; // c/H0 in Mpc

#[test]
fn the_redshift_is_a_boost() {
    // a spectrograph records a frequency ratio: boosting a null photon scales it by the Bondi
    // factor k = e^φ. the measurement is a boost — a rapidity, exponentiated
    let phi = 0.6_f64;
    let k = phi.exp();
    let photon = Geonum::new_from_cartesian(1.0, 1.0); // null: E = p
    let observed = photon.boost(Angle::new(0.0, 1.0), k);
    assert!(
        observed.near_mag(photon.mag * k),
        "redshift = the boost factor k = e^φ"
    );
}

#[test]
fn the_distance_is_the_rapidity() {
    // the comoving distance to a redshift is the accumulated boost angle, D_C = (c/H0)·ln(1+z);
    // the luminosity distance built from it is the distance-redshift law, no density parameter
    for z in [0.1_f64, 0.5, 1.0] {
        let rapidity = (1.0 + z).ln();
        let luminosity = (1.0 + z) * HUBBLE_DISTANCE * rapidity.sinh();
        assert!(
            (luminosity - HUBBLE_DISTANCE * z * (1.0 + z / 2.0)).abs() < 1e-9,
            "distance is the rapidity — the distance-redshift law, nothing tuned"
        );
    }
}

#[test]
fn the_velocity_is_a_projection_of_the_boost() {
    // "velocity" is the boost projected onto a bounded magnitude, β = tanh(φ), saturating at c
    // while the rapidity runs free. the observed velocity is the shadow, not the measurement
    for phi in [0.2_f64, 1.0, 3.0] {
        let k = phi.exp();
        let beta = (k * k - 1.0) / (k * k + 1.0);
        assert!(
            (beta - phi.tanh()).abs() < 1e-12,
            "velocity = tanh(φ), the boost projected"
        );
        assert!(
            beta < 1.0,
            "the projection saturates at c; the boost does not"
        );
    }
}

#[test]
fn the_redshift_composes_by_multiplying_boosts() {
    // redshift accumulates along a path as a product of Bondi factors — the rapidities adding.
    // the boost composition, written as geonum's geometric product
    let segments = [0.3_f64, 0.4, 0.5]; // segment rapidities
    let woven = segments.iter().fold(Geonum::scalar(1.0), |acc, &phi| {
        acc * Geonum::scalar(phi.exp())
    });
    assert!(
        woven.near_mag(segments.iter().sum::<f64>().exp()),
        "boosts multiply, rapidities add — redshift composes"
    );
}

#[test]
fn the_universe_expands_by_scaling_magnitude() {
    // expansion scales every length and moves no angle — the scale factor is a magnitude
    // multiplier, structure preserved
    let h = 0.01;
    let structure = Geonum::new(100.0, 1.0, 5.0);
    let mut expanded = structure;
    for _ in 0..50 {
        expanded = expanded.scale(1.0 + h);
    }
    assert!(
        expanded.near_mag(100.0 * (1.0_f64 + h).powi(50)),
        "magnitude scaled by (1 + H)^50"
    );
    assert!(
        expanded.angle.near(&structure.angle),
        "expansion never moves the angle"
    );
}

#[test]
fn the_redshift_stretches_by_the_scale_factor() {
    // the cosmological redshift is the scale-factor ratio stretching a wavelength — a Bondi
    // factor sourced by expansion, the same magnitude scale as the Doppler boost above
    let scale_ratio = (1.0_f64).exp(); // a_now/a_then = e^(H·Δt) for H·Δt = 1
    let emitted = Geonum::new(121.6, 0.0, 1.0); // lyman-α wavelength
    let observed = emitted.scale(scale_ratio);
    assert!(
        observed.near_mag(emitted.mag * scale_ratio),
        "the wavelength stretches by the scale-factor ratio"
    );
}
