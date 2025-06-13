//! waves trait implementation
//!
//! defines the Waves trait and related functionality for wave propagation modeling

use crate::geonum_mod::Geonum;

pub trait Waves: Sized {
    /// propagates waves through spacetime using wave equation principles
    /// conventional: numerical wave propagation with spatial/temporal discretization O(n²)
    /// geonum: direct phase evolution via angle rotation O(1)
    fn propagate(&self, time: f64, position: f64, velocity: f64) -> Self;

    /// creates dispersive waves with wavenumber and frequency
    /// conventional: wave packet construction with fourier transforms O(n log n)
    /// geonum: direct dispersion relation encoding O(1)
    fn disperse(position: f64, time: f64, wavenumber: f64, frequency: f64) -> Self;
}

impl Waves for Geonum {
    fn propagate(&self, time: f64, position: f64, velocity: f64) -> Self {
        // compute phase based on position and time
        let phase = position - velocity * time;

        // create new geometric number with same length but adjusted angle
        Geonum {
            length: self.length,
            angle: self.angle + phase, // phase modulation
            blade: self.blade,         // preserve blade grade
        }
    }

    fn disperse(position: f64, time: f64, wavenumber: f64, frequency: f64) -> Self {
        // compute phase based on dispersion relation: φ = kx - ωt
        let phase = wavenumber * position - frequency * time;

        // create new geometric number with unit length and phase angle
        Geonum {
            length: 1.0,
            angle: phase,
            blade: 1, // default to vector grade (1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::TWO_PI;
    use std::f64::consts::PI;

    #[test]
    fn it_propagates() {
        // create a geometric number representing a wave
        let wave = Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1, // vector (grade 1)
        };

        // define wave parameters
        let velocity = 3.0e8; // speed of light
        let time_1 = 0.0;
        let time_2 = 1.0e-9; // 1 nanosecond later
        let position = 0.0;

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

        // prove phase evolves as expected: phase = position - velocity * time
        let expected_phase_t1 = position - velocity * time_1;
        let expected_phase_t2 = position - velocity * time_2;

        assert!(
            (wave_t1.angle - (wave.angle + expected_phase_t1)).abs() < 1e-10,
            "phase at t1 evolves according to position - velocity * time"
        );
        assert!(
            (wave_t2.angle - (wave.angle + expected_phase_t2)).abs() < 1e-10,
            "phase at t2 evolves according to position - velocity * time"
        );

        // prove phase difference between two time points
        let phase_diff = wave_t2.angle - wave_t1.angle;
        let expected_diff = -velocity * (time_2 - time_1);
        assert!(
            (phase_diff - expected_diff).abs() < 1e-10,
            "phase difference equals negative velocity times time difference"
        );

        // prove propagation in space
        let position_2 = 1.0; // 1 meter away
        let wave_p2 = wave.propagate(time_1, position_2, velocity);

        let expected_phase_p2 = position_2 - velocity * time_1;
        assert!(
            (wave_p2.angle - (wave.angle + expected_phase_p2)).abs() < 1e-10,
            "phase at p2 evolves according to position - velocity * time"
        );
    }

    #[test]
    fn it_disperses() {
        // define wave parameters
        let wavenumber = 2.0 * PI; // 2π rad/m (wavelength = 1m)
        let frequency = 3.0e8 * wavenumber; // ω = c·k for light
        let position_1 = 0.0;
        let position_2 = 0.5; // half a wavelength
        let time_1 = 0.0;
        let time_2 = 1.0 / (frequency / (2.0 * PI)); // one period

        // create waves at different positions and times
        let wave_x1_t1 = Geonum::disperse(position_1, time_1, wavenumber, frequency);
        let wave_x2_t1 = Geonum::disperse(position_2, time_1, wavenumber, frequency);
        let wave_x1_t2 = Geonum::disperse(position_1, time_2, wavenumber, frequency);

        // prove all waves have unit amplitude
        assert_eq!(
            wave_x1_t1.length, 1.0,
            "dispersed waves have unit amplitude"
        );

        // prove phase at origin and t=0 is zero
        assert!(
            (wave_x1_t1.angle % TWO_PI).abs() < 1e-10,
            "phase at origin and t=0 is zero"
        );

        // prove spatial phase difference after half a wavelength
        // phase = kx - ωt, so at t=0, phase difference should be k·(x2-x1) = k·0.5 = π
        let expected_phase_diff_space = wavenumber * (position_2 - position_1);
        assert!(
            (wave_x2_t1.angle - wave_x1_t1.angle - expected_phase_diff_space).abs() < 1e-10,
            "spatial phase difference equals wavenumber times distance"
        );

        // prove temporal phase difference after one period
        // after one period, the phase should be the same (2π difference)
        let phase_diff_time = (wave_x1_t2.angle - wave_x1_t1.angle) % TWO_PI;
        assert!(
            (phase_diff_time).abs() < 1e-10,
            "temporal phase repeats after one period"
        );

        // prove dispersion relation by comparing wave phase velocities
        // For k=2π, ω=2πc, wave speed should be c
        let wave_speed = frequency / wavenumber;
        let expected_speed = 3.0e8; // speed of light
        assert!(
            (wave_speed - expected_speed).abs() / expected_speed < 1e-10,
            "dispersion relation yields correct wave speed"
        );
    }
}
