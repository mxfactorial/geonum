//! waves trait implementation
//!
//! defines the Waves trait and related functionality for wave propagation modeling

use crate::{angle::Angle, geonum_mod::Geonum};
use std::f64::consts::PI;

pub trait Waves: Sized {
    /// propagates waves through spacetime using wave equation principles
    /// conventional: numerical wave propagation with spatial/temporal discretization O(n²)
    /// geonum: direct phase evolution via angle rotation O(1)
    fn propagate(&self, time: Self, position: Self, velocity: Self) -> Self;

    /// creates dispersive waves with wavenumber and frequency
    /// conventional: wave packet construction with fourier transforms O(n log n)
    /// geonum: direct dispersion relation encoding O(1)
    fn disperse(position: Self, time: Self, wavenumber: Self, frequency: Self) -> Self;

    /// computes frequency as a geometric number from temporal phase evolution
    /// conventional: scalar frequency extraction via division
    /// geonum: directional frequency preserving temporal geometry O(1)
    fn frequency(&self, other: &Self, time_interval: Self) -> Self;

    /// computes wavenumber as a geometric number from spatial phase evolution
    /// conventional: scalar wavenumber extraction via division
    /// geonum: directional wavenumber preserving spatial geometry O(1)
    fn wavenumber(&self, other: &Self, spatial_interval: Self) -> Self;
}

impl Waves for Geonum {
    fn propagate(&self, time: Self, position: Self, velocity: Self) -> Self {
        // compute phase based on position and time using geometric operations
        let velocity_time = velocity * time;
        let phase = position - velocity_time;

        // create new geometric number with same magnitude but adjusted angle
        Geonum::new_with_angle(self.mag, self.angle + phase.angle)
    }

    fn disperse(position: Self, time: Self, wavenumber: Self, frequency: Self) -> Self {
        // the dispersion relation φ = kx − ωt is the wave's ANGLE, the polar form
        // E = [1, kx − ωt], so cos_sin reads the field straight off the angle
        let k_x = wavenumber * position;
        let omega_t = frequency * time;
        let phase = k_x - omega_t;

        // signed phase = the (kx − ωt) vector projected onto the real axis. rotating
        // a unit wave by it carries φ in the angle, where cos_sin can recover it —
        // storing φ in the magnitude (the earlier form) collapsed it to a sign
        let phi = phase.mag * phase.angle.grade_angle().cos();
        Geonum::new_with_angle(1.0, Angle::new(phi / PI, 1.0))
    }

    fn frequency(&self, other: &Self, time_interval: Self) -> Self {
        // compute temporal phase difference
        let phase_diff = *self - *other;

        // frequency is phase difference per unit time
        let magnitude = phase_diff.mag / time_interval.mag;
        Geonum::new_with_angle(magnitude, Angle::new(1.0, 2.0))
    }

    fn wavenumber(&self, other: &Self, spatial_interval: Self) -> Self {
        // compute spatial phase difference
        let phase_diff = *self - *other;

        // wavenumber is phase difference per unit distance
        let magnitude = phase_diff.mag / spatial_interval.mag;
        Geonum::new_with_angle(magnitude, Angle::new(1.0, 2.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn it_propagates() {
        // create a geometric number representing a wave
        let wave = Geonum::new(1.0, 0.0, 1.0);

        // define wave parameters as geonums
        let velocity = Geonum::new(3.0e8, 0.0, 1.0); // speed of light
        let time_1 = Geonum::new(0.0, 0.0, 1.0);
        let time_2 = Geonum::new(1.0e-9, 0.0, 1.0); // 1 nanosecond later
        let position = Geonum::new(0.0, 0.0, 1.0);

        // propagate wave at two different time points
        let wave_t1 = wave.propagate(time_1, position, velocity);
        let wave_t2 = wave.propagate(time_2, position, velocity);

        // prove magnitude is preserved during propagation
        assert_eq!(wave_t1.mag, wave.mag, "propagation preserves amplitude");
        assert_eq!(wave_t2.mag, wave.mag, "propagation preserves amplitude");

        // compute actual phase changes from the propagate method
        // phase = position - velocity * time as Geonum operations
        let velocity_time_1 = velocity * time_1;
        let phase_1 = position - velocity_time_1;

        let velocity_time_2 = velocity * time_2;
        let phase_2 = position - velocity_time_2;

        // test that wave angles are correctly updated
        assert_eq!(
            wave_t1.angle,
            wave.angle + phase_1.angle,
            "phase at t1 evolves according to position - velocity * time"
        );
        assert_eq!(
            wave_t2.angle,
            wave.angle + phase_2.angle,
            "phase at t2 evolves according to position - velocity * time"
        );

        // prove phase difference between two time points
        let phase_diff = wave_t2.angle - wave_t1.angle;
        let expected_phase_diff = phase_2.angle - phase_1.angle;
        assert_eq!(
            phase_diff, expected_phase_diff,
            "phase difference matches expected value"
        );

        // prove propagation in space
        let position_2 = Geonum::new(1.0, 0.0, 1.0); // 1 meter away
        let wave_p2 = wave.propagate(time_1, position_2, velocity);

        let phase_p2 = position_2 - velocity_time_1;
        assert_eq!(
            wave_p2.angle,
            wave.angle + phase_p2.angle,
            "phase at p2 evolves according to position - velocity * time"
        );
    }

    #[test]
    fn it_disperses() {
        // a plane wave is E = [1, kx − ωt]: the dispersion relation lives in the
        // ANGLE, so cos_sin reads the field. k = 2π gives wavelength 1; null
        // dispersion ω = ck carries the wave at the speed of light
        let c = 3.0e8;
        let k = 2.0 * PI; // 2π rad/m, wavelength 1 m
        let omega = c * k; // ω = ck for light
        let wavenumber = Geonum::scalar(k);
        let frequency = Geonum::scalar(omega);

        // at the origin the phase is 0 — the wave sits at its crest, cos = 1
        let crest = Geonum::disperse(
            Geonum::scalar(0.0),
            Geonum::scalar(0.0),
            wavenumber,
            frequency,
        );
        assert!(crest.near_mag(1.0), "dispersed waves have unit amplitude");
        let (cos_crest, _) = crest.angle.cos_sin();
        assert!(
            (cos_crest - 1.0).abs() < 1e-12,
            "phase 0 at the origin — the wave's crest, cos = 1"
        );

        // a quarter wavelength out kx = π/2: the phase lands at grade 1, a node
        let node = Geonum::disperse(
            Geonum::scalar(0.25),
            Geonum::scalar(0.0),
            wavenumber,
            frequency,
        );
        assert_eq!(
            node.angle.grade(),
            1,
            "kx = π/2 lands at grade 1 — the node"
        );
        let (cos_node, _) = node.angle.cos_sin();
        assert!(
            cos_node.abs() < 1e-12,
            "a quarter wavelength is a node — cos = 0"
        );

        // half a wavelength out kx = π: the phase is grade 2, the trough, cos = −1
        let trough = Geonum::disperse(
            Geonum::scalar(0.5),
            Geonum::scalar(0.0),
            wavenumber,
            frequency,
        );
        assert_eq!(
            trough.angle.grade(),
            2,
            "kx = π lands at grade 2 — the trough"
        );
        let (cos_trough, _) = trough.angle.cos_sin();
        assert!(
            (cos_trough + 1.0).abs() < 1e-12,
            "half a wavelength is the trough — cos = −1"
        );

        // one period later t = 2π/ω the wave returns to the same phase — periodic,
        // the angle's blade arithmetic handling the wraparound with no manual modulo
        let period = 2.0 * PI / omega;
        let later = Geonum::disperse(
            Geonum::scalar(0.0),
            Geonum::scalar(period),
            wavenumber,
            frequency,
        );
        let (cos_later, _) = later.angle.cos_sin();
        assert!(
            (cos_later - cos_crest).abs() < 1e-9,
            "the wave repeats after one period T = 2π/ω"
        );

        // the phase velocity ω/k recovers the speed of light — the null dispersion
        let wave_speed = frequency.mag / wavenumber.mag;
        assert!(
            (wave_speed - c).abs() / c < 1e-10,
            "ω/k = c — the dispersion relation yields lightspeed"
        );
    }
}
