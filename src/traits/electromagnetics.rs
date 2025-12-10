//! electromagnetics trait implementation
//!
//! defines the Electromagnetics trait and related functionality for electromagnetic modeling

use crate::{angle::Angle, geonum_mod::Geonum};
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
    fn inverse_field(
        charge: Geonum,
        distance: Geonum,
        power: Geonum,
        angle: Angle,
        constant: Geonum,
    ) -> Self;

    /// calculates electric potential at a distance from a point charge
    /// conventional: scalar field calculations requiring spatial discretization O(n)
    /// geonum: direct coulomb law computation with geometric encoding O(1)
    fn electric_potential(charge: Geonum, distance: Geonum) -> Geonum;

    /// calculates electric field at a distance from a point charge
    /// conventional: vector field calculations with coordinate transformations O(n)
    /// geonum: direct field encoding with direction and magnitude O(1)
    fn electric_field(charge: Geonum, distance: Geonum) -> Self;

    /// calculates the poynting vector using wedge product
    /// conventional: cross product calculations with vector components O(n)
    /// geonum: wedge product for electromagnetic energy flux O(1)
    fn poynting_vector(&self, b_field: &Self) -> Self;

    /// creates a magnetic vector potential for a current-carrying wire
    /// conventional: vector potential calculations with integration O(n²)
    /// geonum: direct logarithmic encoding for wire geometry O(1)
    fn wire_vector_potential(r: Geonum, current: Geonum, permeability: Geonum) -> Self;

    /// creates a magnetic field for a current-carrying wire
    /// conventional: ampères law with circular integration O(n)
    /// geonum: direct circular field encoding O(1)
    fn wire_magnetic_field(r: Geonum, current: Geonum, permeability: Geonum) -> Self;

    /// creates a scalar potential for a spherical electromagnetic wave
    /// conventional: wave equation solutions with spatial/temporal discretization O(n²)
    /// geonum: direct wave encoding with phase relationships O(1)
    fn spherical_wave_potential(r: Geonum, t: Geonum, wavenumber: Geonum, speed: Geonum) -> Self;
}

impl Electromagnetics for Geonum {
    fn inverse_field(
        charge: Geonum,
        distance: Geonum,
        power: Geonum,
        angle: Angle,
        constant: Geonum,
    ) -> Self {
        let magnitude = constant.mag * charge.mag / distance.mag.powf(power.mag);
        // angle calculation for negative charges
        let direction = if charge.angle.grade_angle().cos() >= 0.0 {
            angle
        } else {
            angle + Angle::new(1.0, 1.0) // add π
        };

        Geonum::new_with_angle(magnitude, direction)
    }

    fn electric_potential(charge: Geonum, distance: Geonum) -> Geonum {
        // coulomb constant k = 1/(4πε₀)
        let k = Geonum::scalar(1.0 / (4.0 * PI * VACUUM_PERMITTIVITY));
        charge * k / distance
    }

    fn electric_field(charge: Geonum, distance: Geonum) -> Self {
        let k = Geonum::scalar(1.0 / (4.0 * PI * VACUUM_PERMITTIVITY));
        let power = Geonum::scalar(2.0);
        let angle = Angle::new(1.0, 1.0); // π
        Self::inverse_field(charge, distance, power, angle, k)
    }

    fn poynting_vector(&self, b_field: &Self) -> Self {
        // wedge product handles the cross product geometry in ga
        let poynting = self.wedge(b_field);
        Geonum::new_with_angle(poynting.mag / VACUUM_PERMEABILITY, poynting.angle)
    }

    fn wire_vector_potential(r: Geonum, current: Geonum, permeability: Geonum) -> Self {
        // A = (μ₀I/2π) * ln(r) in theta direction around wire
        let magnitude = permeability.mag * current.mag * (r.mag.ln()) / (2.0 * PI);
        let angle = Angle::new(1.0, 2.0); // π/2
        Geonum::new_with_angle(magnitude, angle)
    }

    fn wire_magnetic_field(r: Geonum, current: Geonum, permeability: Geonum) -> Self {
        // B = μ₀I/(2πr) in phi direction circling the wire
        let magnitude = permeability.mag * current.mag / (2.0 * PI * r.mag);
        let angle = Angle::new(0.0, 1.0); // 0
        Geonum::new_with_angle(magnitude, angle)
    }

    fn spherical_wave_potential(r: Geonum, t: Geonum, wavenumber: Geonum, speed: Geonum) -> Self {
        let omega = wavenumber.mag * speed.mag; // angular frequency
        let potential = (wavenumber.mag * r.mag - omega * t.mag).cos() / r.mag;

        // represent as a geometric number with scalar (grade 0) convention
        let magnitude = potential.abs();
        let angle = if potential >= 0.0 {
            Angle::new(0.0, 1.0)
        } else {
            Angle::new(1.0, 1.0)
        };
        Geonum::new_with_angle(magnitude, angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::EPSILON;

    #[test]
    fn it_computes_electric_field() {
        // test positive charge
        let charge = Geonum::new(2.0, 0.0, 1.0);
        let distance = Geonum::new(3.0, 0.0, 1.0);
        let e_field = Geonum::electric_field(charge, distance);

        // coulomb constant
        let k = 1.0 / (4.0 * PI * VACUUM_PERMITTIVITY);

        // prove magnitude follows inverse square law
        assert!((e_field.mag - k * 2.0 / (3.0 * 3.0)).abs() < EPSILON);

        // prove direction is outward for positive charge
        assert_eq!(e_field.angle, Angle::new(1.0, 1.0)); // π

        // test negative charge
        let neg_charge = Geonum::new(2.0, 1.0, 1.0); // magnitude 2, angle π (negative)
        let e_field_neg = Geonum::electric_field(neg_charge, distance);

        // prove magnitude is the same
        assert!((e_field_neg.mag - k * 2.0 / (3.0 * 3.0)).abs() < EPSILON);

        // prove direction is inward for negative charge
        // initial angle π + π for negative = 2π = blade 4
        let expected_neg_angle = Angle::new(2.0, 1.0); // 2π
        assert_eq!(e_field_neg.angle, expected_neg_angle);
    }

    #[test]
    fn it_computes_poynting_vector_with_wedge() {
        // create perpendicular fields
        let e = Geonum::new(5.0, 0.0, 1.0); // along x-axis
        let b = Geonum::new(2.0, 1.0, 2.0); // [2, π/2] - magnetic field bivector

        let s = e.poynting_vector(&b);

        // check direction is perpendicular to both fields
        let expected_angle = Angle::new(1.0, 1.0); // π
        assert_eq!(s.angle, expected_angle); // using Angle comparison

        // check magnitude is E×B/μ₀
        assert_eq!(s.mag, (5.0 * 2.0) / VACUUM_PERMEABILITY);
    }

    #[test]
    fn it_creates_fields_with_inverse_power_laws() {
        // test electric field (inverse square)
        let charge = Geonum::new(1.0, 0.0, 1.0);
        let distance = Geonum::new(2.0, 0.0, 1.0);
        let power = Geonum::new(2.0, 0.0, 1.0);
        let angle = Angle::new(1.0, 1.0); // π
        let constant = Geonum::new(1.0, 0.0, 1.0);

        let e_field = Geonum::inverse_field(charge, distance, power, angle, constant);
        assert_eq!(e_field.mag, 0.25); // 1.0 * 1.0 / 2.0²
        assert_eq!(e_field.angle, angle);

        // test gravity (also inverse square)
        let mass = Geonum::new(5.0, 0.0, 1.0);
        let angle_gravity = Angle::new(0.0, 1.0); // 0
        let g_constant = Geonum::new(6.67e-11, 0.0, 1.0);

        let g_field = Geonum::inverse_field(mass, distance, power, angle_gravity, g_constant);
        assert_eq!(g_field.mag, 6.67e-11 * 5.0 / 4.0);
        assert_eq!(g_field.angle, angle_gravity);

        // test inverse cube field
        let charge_cube = Geonum::new(2.0, 0.0, 1.0);
        let power_cube = Geonum::new(3.0, 0.0, 1.0);
        let angle_cube = Angle::new(1.0, 2.0); // π/2

        let field = Geonum::inverse_field(charge_cube, distance, power_cube, angle_cube, constant);
        assert_eq!(field.mag, 0.25); // 1.0 * 2.0 / 2.0³
        assert_eq!(field.angle, angle_cube);
    }

    #[test]
    fn it_models_wire_magnetic_field() {
        // test magnetic field around a current-carrying wire
        let current = Geonum::new(10.0, 0.0, 1.0); // 10 amperes
        let distance = Geonum::new(0.02, 0.0, 1.0); // 2 cm from wire
        let permeability = Geonum::new(VACUUM_PERMEABILITY, 0.0, 1.0);

        let b_field = Geonum::wire_magnetic_field(distance, current, permeability);

        // prove magnitude using ampère's law: B = μ₀I/(2πr)
        let expected_magnitude = VACUUM_PERMEABILITY * 10.0 / (2.0 * PI * 0.02);
        assert!((b_field.mag - expected_magnitude).abs() < EPSILON);

        // prove direction (circular around wire)
        assert_eq!(b_field.angle, Angle::new(0.0, 1.0));

        // test field strength increases with current
        let stronger_current = Geonum::new(20.0, 0.0, 1.0);
        let stronger_field = Geonum::wire_magnetic_field(distance, stronger_current, permeability);
        assert!(stronger_field.mag > b_field.mag);

        // test field strength decreases with distance
        let farther_distance = Geonum::new(0.1, 0.0, 1.0);
        let farther_field = Geonum::wire_magnetic_field(farther_distance, current, permeability);
        assert!(farther_field.mag < b_field.mag);
    }
}
