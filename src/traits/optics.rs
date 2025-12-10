//! optics trait implementation
//!
//! defines the Optics trait and related functionality for optical modeling

use crate::{angle::Angle, geonum_mod::Geonum};
use std::f64::consts::PI;

pub trait Optics: Sized {
    /// convert lens refraction to angle transformation using Snell's law
    /// conventional: vector-based ray tracing through interfaces O(n)
    /// geonum: single angle transformation based on Snell's law O(1)
    fn refract(&self, refractive_index: Geonum) -> Self;

    /// apply optical path aberration using zernike coefficients
    /// conventional: phase map computation + wavefront sampling O(n²)
    /// geonum: direct phase perturbation via angle modification O(1)
    fn aberrate(&self, zernike_coefficients: &[Self]) -> Self;

    /// compute optical transfer function through frequency-space transformation
    /// conventional: FFT-based propagation O(n log n)
    /// geonum: direct frequency-domain angle mapping O(1)
    fn otf(&self, focal_mag: Geonum, wavelength: Geonum) -> Self;

    /// apply ABCD matrix ray tracing as direct angle operations
    /// conventional: 4×4 matrix multiplications for ray propagation O(n)
    /// geonum: encode entire matrix effect as single angle transformation O(1)
    fn abcd_transform(&self, a: Geonum, b: Geonum, c: Geonum, d: Geonum) -> Self;

    /// apply magnification to the geometric number
    /// conventional: complex transformations with multiple operations
    /// geonum: direct angle scaling and intensity adjustment O(1)
    fn magnify(&self, magnification: Geonum) -> Self;
}

impl Optics for Geonum {
    fn refract(&self, refractive_index: Geonum) -> Self {
        // apply snells law as angle transformation
        let incident_angle = self.angle;
        let n_ratio = refractive_index.mag;
        let refracted_angle_rem = (incident_angle.grade_angle().sin() / n_ratio).asin();
        let refracted_angle = Angle::new(refracted_angle_rem, PI);

        Geonum::new_with_angle(self.mag, refracted_angle)
    }

    fn aberrate(&self, zernike_coefficients: &[Self]) -> Self {
        // apply zernike polynomial aberrations to phase
        let mut perturbed_phase = self.angle;

        // apply each zernike term
        for term in zernike_coefficients {
            let mode_effect_rem = term.mag * (term.angle.grade_angle().sin() * 3.0).cos();
            let mode_effect = Angle::new(mode_effect_rem, PI);
            perturbed_phase = perturbed_phase + mode_effect;
        }

        Geonum::new_with_angle(self.mag, perturbed_phase)
    }

    fn otf(&self, focal_mag: Geonum, wavelength: Geonum) -> Self {
        // convert from spatial domain to frequency domain
        let frequency = self.mag / (wavelength.mag * focal_mag.mag);
        let quarter_turn = Angle::new(1.0, 2.0); // π/2
        let phase = self.angle + quarter_turn;

        Geonum::new_with_angle(frequency, phase)
    }

    fn abcd_transform(&self, a: Geonum, b: Geonum, c: Geonum, d: Geonum) -> Self {
        // apply ABCD matrix as angle transformation
        // in ray optics: [h_out, theta_out] = [[A, B], [C, D]] * [h_in, theta_in]
        // for geonum representation:
        // - ray height h = self.mag (distance from optical axis)
        // - ray angle theta = self.angle (angle with optical axis)

        // extract ray parameters
        let h = self.mag;
        let theta = self.angle;

        // abcd transformations for ray tracing
        // new_h = A*h + B*theta
        // new_theta = C*h + D*theta
        // theta needs to be in radians for matrix multiplication
        let theta_radians = theta.grade_angle();
        let new_h = a.mag * h + b.mag * theta_radians;
        let new_theta_radians = c.mag * h + d.mag * theta_radians;
        let new_theta_angle = Angle::new(new_theta_radians, PI);

        // return new geonum with transformed height and angle
        Geonum::new_with_angle(new_h, new_theta_angle)
    }

    fn magnify(&self, magnification: Geonum) -> Self {
        // magnification affects intensity (inverse square law) and angle scaling
        let m = magnification.mag;
        let image_intensity = 1.0 / (m * m);

        // image point has inverted angle and scaled height
        let image_angle_rem = -self.angle.grade_angle().sin() / m;
        let image_angle = Angle::new(image_angle_rem, PI);

        Geonum::new_with_angle(self.mag * image_intensity, image_angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_applies_optical_magnification() {
        // create input ray/object point
        let object = Geonum::new(4.0, 1.0, 6.0); // 4.0 magnitude, π/6 angle

        // test 2x magnification
        let magnified_2x = object.magnify(Geonum::scalar(2.0));

        // prove intensity follows inverse square law (1/m²)
        let expected_intensity_2x = 4.0 / (2.0 * 2.0);
        assert_eq!(magnified_2x.mag, expected_intensity_2x);

        // prove angle is inverted and scaled based on sin transformation
        // magnify computes: image_angle = -sin(object_angle) / m
        let object_sin = object.angle.grade_angle().sin();
        let expected_angle_rem_2x = -object_sin / 2.0;
        let expected_angle_2x = Angle::new(expected_angle_rem_2x, PI);
        assert_eq!(magnified_2x.angle, expected_angle_2x);

        // test 0.5x magnification (minification)
        let magnified_half = object.magnify(Geonum::scalar(0.5));

        // prove intensity increases with minification
        let expected_intensity_half = 4.0 / (0.5 * 0.5);
        assert_eq!(magnified_half.mag, expected_intensity_half);

        // prove angle is inverted and scaled
        let expected_angle_rem_half = -object_sin / 0.5;
        let expected_angle_half = Angle::new(expected_angle_rem_half, PI);
        assert_eq!(magnified_half.angle, expected_angle_half);
    }

    #[test]
    fn it_applies_abcd_transform_angle_dependence() {
        // test that abcd transform produces different outputs for different input angles
        // this tests the fundamental property of ray optics matrices

        // create two rays with different angles
        let ray1 = Geonum::new(1.0, 1.0, 6.0); // π/6 angle
        let ray2 = Geonum::new(1.0, 1.0, 3.0); // π/3 angle

        // apply thin lens ABCD matrix (focal magnitude = 100)
        // [A B]   [1   0]
        // [C D] = [-1/f 1]
        let a = Geonum::scalar(1.0);
        let b = Geonum::scalar(0.0);
        let c = Geonum::scalar(-0.01); // -1/100
        let d = Geonum::scalar(1.0);

        let transformed1 = ray1.abcd_transform(a, b, c, d);
        let transformed2 = ray2.abcd_transform(a, b, c, d);

        // the transformed angles should be different
        assert_ne!(transformed1.angle, transformed2.angle);

        // compute expected angle difference
        // for thin lens: new_angle = atan2(h - h/f, h) = atan2(h(1-1/f), h) = atan(1-1/f)
        // since current implementation uses sin for both h and theta,
        // all rays get same output angle regardless of input
        let sin1 = ray1.angle.grade_angle().sin();
        let sin2 = ray2.angle.grade_angle().sin();

        // this assertion will fail, proving the bug
        assert!(
            (sin1 - sin2).abs() > 1e-10,
            "different input angles should produce different sin values"
        );

        // the angle difference should depend on input angle difference
        let angle_diff = transformed2.angle - transformed1.angle;
        assert_ne!(
            angle_diff,
            Angle::new(0.0, 1.0),
            "ABCD transform should produce angle-dependent refraction"
        );
    }
}
