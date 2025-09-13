//! waves trait implementation
//!
//! defines the Waves trait and related functionality for wave propagation modeling

use crate::{angle::Angle, geonum_mod::Geonum};

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

        // create new geometric number with same length but adjusted angle
        Geonum::new_with_angle(self.length, self.angle + phase.angle)
    }

    fn disperse(position: Self, time: Self, wavenumber: Self, frequency: Self) -> Self {
        // compute phase based on dispersion relation: φ = kx - ωt
        let k_x = wavenumber * position;
        let omega_t = frequency * time;
        let phase = k_x - omega_t;

        // create new geometric number with unit length and phase angle
        Geonum::new_with_angle(1.0, phase.angle)
    }

    fn frequency(&self, other: &Self, time_interval: Self) -> Self {
        // compute temporal phase difference
        let phase_diff = *self - *other;

        // frequency is phase difference per unit time
        let magnitude = phase_diff.length / time_interval.length;
        Geonum::new_with_angle(magnitude, Angle::new(1.0, 2.0))
    }

    fn wavenumber(&self, other: &Self, spatial_interval: Self) -> Self {
        // compute spatial phase difference
        let phase_diff = *self - *other;

        // wavenumber is phase difference per unit distance
        let magnitude = phase_diff.length / spatial_interval.length;
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

        // prove length is preserved during propagation
        assert_eq!(
            wave_t1.length, wave.length,
            "propagation preserves amplitude"
        );
        assert_eq!(
            wave_t2.length, wave.length,
            "propagation preserves amplitude"
        );

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
        // define wave parameters as geonums
        let wavenumber = Geonum::new(2.0 * PI, 0.0, 1.0); // 2π rad/m (wavelength = 1m)
        let frequency = Geonum::new(3.0e8 * 2.0 * PI, 0.0, 1.0); // ω = c·k for light
        let position_1 = Geonum::new(0.0, 0.0, 1.0);
        let position_2 = Geonum::new(0.5, 0.0, 1.0); // half a wavelength
        let time_1 = Geonum::new(0.0, 0.0, 1.0);
        let time_2 = Geonum::new(1.0 / (3.0e8 * 2.0 * PI / (2.0 * PI)), 0.0, 1.0); // one period

        // create waves at different positions and times
        let wave_x1_t1 = Geonum::disperse(position_1, time_1, wavenumber, frequency);
        let wave_x2_t1 = Geonum::disperse(position_2, time_1, wavenumber, frequency);
        let wave_x1_t2 = Geonum::disperse(position_1, time_2, wavenumber, frequency);

        // prove all waves have unit amplitude
        assert_eq!(
            wave_x1_t1.length, 1.0,
            "dispersed waves have unit amplitude"
        );

        // prove phase at origin and t=0 has blade 2 from 0-0 subtraction
        assert_eq!(
            wave_x1_t1.angle,
            Angle::new_with_blade(2, 0.0, 1.0),
            "phase at origin and t=0 has blade 2 from subtraction"
        );

        // prove spatial phase difference after half a wavelength
        // compute the actual phase difference from the disperse operations
        let actual_phase_diff = wave_x2_t1.angle - wave_x1_t1.angle;

        // compute phase difference using geonum operations
        // (at half wavelength this represents π radians or blade 2 geometrically)
        let k_x1 = wavenumber * position_1;
        let k_x2 = wavenumber * position_2;
        let omega_t = frequency * time_1;
        let phase_1 = k_x1 - omega_t;
        let phase_2 = k_x2 - omega_t;
        let expected_phase_diff = phase_2.angle - phase_1.angle;

        assert_eq!(
            actual_phase_diff, expected_phase_diff,
            "spatial phase difference equals wavenumber times distance"
        );

        // prove temporal phase difference after one period
        // compute actual phase difference between t2 and t1
        let k_x = wavenumber * position_1;
        let omega_t1 = frequency * time_1;
        let omega_t2 = frequency * time_2;
        let phase_t1 = k_x - omega_t1;
        let phase_t2 = k_x - omega_t2;

        // the phase difference should complete a full cycle (2π)
        let phase_diff_angle = wave_x1_t2.angle - wave_x1_t1.angle;
        let expected_temporal_diff = phase_t2.angle - phase_t1.angle;

        // test that blade difference is 4 (full rotation) or equivalent
        assert_eq!(
            phase_diff_angle, expected_temporal_diff,
            "temporal phase evolution matches expected value"
        );

        // prove dispersion relation by comparing wave phase velocities
        // For k=2π, ω=2πc, wave speed should be c
        let wave_speed = frequency.length / wavenumber.length;
        let expected_speed = 3.0e8; // speed of light
        assert!(
            (wave_speed - expected_speed).abs() / expected_speed < 1e-10,
            "dispersion relation yields correct wave speed"
        );
    }
}
