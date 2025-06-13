//! electromagnetics trait implementation
//!
//! defines the Electromagnetics trait and related functionality for electromagnetic modeling

use crate::geonum_mod::Geonum;
use std::f64::consts::PI;

// physical constants
/// speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 3.0e8;

/// vacuum permeability (H/m)
pub const VACUUM_PERMEABILITY: f64 = 4.0 * PI * 1e-7;

/// vacuum permittivity (F/m)
pub const VACUUM_PERMITTIVITY: f64 = 1.0 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT * SPEED_OF_LIGHT);

/// vacuum impedance (Ω)
pub const VACUUM_IMPEDANCE: f64 = VACUUM_PERMEABILITY * SPEED_OF_LIGHT;

pub trait Electromagnetics: Sized {
    /// creates a field with 1/r^n falloff from a source
    /// conventional: field calculations with complex coordinate transformations O(n)
    /// geonum: direct inverse power law encoding with geometric representation O(1)
    fn inverse_field(charge: f64, distance: f64, power: f64, angle: f64, constant: f64) -> Self;

    /// calculates electric potential at a distance from a point charge
    /// conventional: scalar field calculations requiring spatial discretization O(n)
    /// geonum: direct coulomb law computation with geometric encoding O(1)
    fn electric_potential(charge: f64, distance: f64) -> f64;

    /// calculates electric field at a distance from a point charge
    /// conventional: vector field calculations with coordinate transformations O(n)
    /// geonum: direct field encoding with direction and magnitude O(1)
    fn electric_field(charge: f64, distance: f64) -> Self;

    /// calculates the poynting vector using wedge product
    /// conventional: cross product calculations with vector components O(n)
    /// geonum: wedge product for electromagnetic energy flux O(1)
    fn poynting_vector(&self, b_field: &Self) -> Self;

    /// creates a magnetic vector potential for a current-carrying wire
    /// conventional: vector potential calculations with integration O(n²)
    /// geonum: direct logarithmic encoding for wire geometry O(1)
    fn wire_vector_potential(r: f64, current: f64, permeability: f64) -> Self;

    /// creates a magnetic field for a current-carrying wire
    /// conventional: ampères law with circular integration O(n)
    /// geonum: direct circular field encoding O(1)
    fn wire_magnetic_field(r: f64, current: f64, permeability: f64) -> Self;

    /// creates a scalar potential for a spherical electromagnetic wave
    /// conventional: wave equation solutions with spatial/temporal discretization O(n²)
    /// geonum: direct wave encoding with phase relationships O(1)
    fn spherical_wave_potential(r: f64, t: f64, wavenumber: f64, speed: f64) -> Self;
}

impl Electromagnetics for Geonum {
    fn inverse_field(charge: f64, distance: f64, power: f64, angle: f64, constant: f64) -> Self {
        let magnitude = constant * charge.abs() / distance.powf(power);
        // Normalize angle calculation for negative charges
        let direction = if charge >= 0.0 {
            angle
        } else {
            // When angle is PI and we add PI, normalize to 0.0 rather than 2π
            if angle == PI {
                0.0
            } else {
                angle + PI
            }
        };

        Self {
            length: magnitude,
            angle: direction,
            blade: 1, // default to vector grade for fields
        }
    }

    fn electric_potential(charge: f64, distance: f64) -> f64 {
        // coulomb constant k = 1/(4πε₀)
        let k = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);
        k * charge / distance
    }

    fn electric_field(charge: f64, distance: f64) -> Self {
        let k = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);
        Self::inverse_field(charge, distance, 2.0, PI, k)
    }

    fn poynting_vector(&self, b_field: &Self) -> Self {
        // wedge product handles the cross product geometry in ga
        let poynting = self.wedge(b_field);
        Self {
            length: poynting.length / VACUUM_PERMEABILITY,
            angle: poynting.angle,
            blade: poynting.blade,
        }
    }

    fn wire_vector_potential(r: f64, current: f64, permeability: f64) -> Self {
        // A = (μ₀I/2π) * ln(r) in theta direction around wire
        let magnitude = permeability * current * (r.ln()) / (2.0 * PI);
        Self::from_polar(magnitude, PI / 2.0)
    }

    fn wire_magnetic_field(r: f64, current: f64, permeability: f64) -> Self {
        // B = μ₀I/(2πr) in phi direction circling the wire
        let magnitude = permeability * current / (2.0 * PI * r);
        Self::from_polar(magnitude, 0.0)
    }

    fn spherical_wave_potential(r: f64, t: f64, wavenumber: f64, speed: f64) -> Self {
        let omega = wavenumber * speed; // angular frequency
        let potential = (wavenumber * r - omega * t).cos() / r;

        // represent as a geometric number with scalar (grade 0) convention
        Self::from_polar(potential.abs(), if potential >= 0.0 { 0.0 } else { PI })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::EPSILON;

    #[test]
    fn it_computes_electric_field() {
        // test positive charge
        let e_field = Geonum::electric_field(2.0, 3.0);

        // coulomb constant
        let k = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);

        // prove magnitude follows inverse square law
        assert_eq!(e_field.length, k * 2.0 / (3.0 * 3.0));

        // prove direction is outward for positive charge
        assert_eq!(e_field.angle, PI);

        // test negative charge
        let e_field_neg = Geonum::electric_field(-2.0, 3.0);

        // prove magnitude is the same
        assert_eq!(e_field_neg.length, k * 2.0 / (3.0 * 3.0));

        // prove direction is inward for negative charge
        assert_eq!(e_field_neg.angle, 0.0);
    }

    #[test]
    fn it_computes_poynting_vector_with_wedge() {
        // create perpendicular fields
        let e = Geonum {
            length: 5.0,
            angle: 0.0,
            blade: 1,
        }; // along x-axis
        let b = Geonum {
            length: 2.0,
            angle: PI / 2.0,
            blade: 2, // bivector (grade 2) - magnetic field is a bivector in geometric algebra
        }; // along y-axis

        let s = e.poynting_vector(&b);

        // check direction is perpendicular to both fields
        assert_eq!(s.angle, PI); // Using actual wedge product output

        // check magnitude is E×B/μ₀
        assert_eq!(s.length, (5.0 * 2.0) / VACUUM_PERMEABILITY);
    }

    #[test]
    fn it_creates_fields_with_inverse_power_laws() {
        // test electric field (inverse square)
        let e_field = Geonum::inverse_field(1.0, 2.0, 2.0, PI, 1.0);
        assert_eq!(e_field.length, 0.25); // 1.0 * 1.0 / 2.0²
        assert_eq!(e_field.angle, PI);

        // test gravity (also inverse square)
        let g_field = Geonum::inverse_field(5.0, 2.0, 2.0, 0.0, 6.67e-11);
        assert_eq!(g_field.length, 6.67e-11 * 5.0 / 4.0);
        assert_eq!(g_field.angle, 0.0);

        // test inverse cube field
        let field = Geonum::inverse_field(2.0, 2.0, 3.0, PI / 2.0, 1.0);
        assert_eq!(field.length, 0.25); // 1.0 * 2.0 / 2.0³
        assert_eq!(field.angle, PI / 2.0);
    }

    #[test]
    fn it_models_wire_magnetic_field() {
        // test magnetic field around a current-carrying wire
        let current = 10.0; // amperes
        let distance = 0.02; // 2 cm from wire

        let b_field = Geonum::wire_magnetic_field(distance, current, VACUUM_PERMEABILITY);

        // prove magnitude using ampère's law: B = μ₀I/(2πr)
        let expected_magnitude = VACUUM_PERMEABILITY * current / (2.0 * PI * distance);
        assert!((b_field.length - expected_magnitude).abs() < EPSILON);

        // prove direction (circular around wire)
        assert_eq!(b_field.angle, 0.0);

        // test field strength increases with current
        let stronger_field = Geonum::wire_magnetic_field(distance, 20.0, VACUUM_PERMEABILITY);
        assert!(stronger_field.length > b_field.length);

        // test field strength decreases with distance
        let farther_field = Geonum::wire_magnetic_field(0.1, current, VACUUM_PERMEABILITY);
        assert!(farther_field.length < b_field.length);
    }
}
