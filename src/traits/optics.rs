//! optics trait implementation
//!
//! defines the Optics trait and related functionality for optical modeling

use crate::geonum_mod::Geonum;
use std::f64::consts::PI;

pub trait Optics: Sized {
    /// convert lens refraction to angle transformation using Snell's law
    /// conventional: vector-based ray tracing through interfaces O(n)
    /// geonum: single angle transformation based on Snell's law O(1)
    fn refract(&self, refractive_index: f64) -> Self;

    /// apply optical path aberration using zernike coefficients
    /// conventional: phase map computation + wavefront sampling O(n²)
    /// geonum: direct phase perturbation via angle modification O(1)
    fn aberrate(&self, zernike_coefficients: &[Self]) -> Self;

    /// compute optical transfer function through frequency-space transformation
    /// conventional: FFT-based propagation O(n log n)
    /// geonum: direct frequency-domain angle mapping O(1)
    fn otf(&self, focal_length: f64, wavelength: f64) -> Self;

    /// apply ABCD matrix ray tracing as direct angle operations
    /// conventional: 4×4 matrix multiplications for ray propagation O(n)
    /// geonum: encode entire matrix effect as single angle transformation O(1)
    fn abcd_transform(&self, a: f64, b: f64, c: f64, d: f64) -> Self;

    /// apply magnification to the geometric number
    /// conventional: complex transformations with multiple operations
    /// geonum: direct angle scaling and intensity adjustment O(1)
    fn magnify(&self, magnification: f64) -> Self;
}

impl Optics for Geonum {
    fn refract(&self, refractive_index: f64) -> Self {
        // apply snells law as angle transformation
        let incident_angle = self.angle;
        let refracted_angle = (incident_angle.sin() / refractive_index).asin();

        Self {
            length: self.length,
            angle: refracted_angle,
            blade: self.blade, // preserve blade grade
        }
    }

    fn aberrate(&self, zernike_coefficients: &[Self]) -> Self {
        // apply zernike polynomial aberrations to phase
        let mut perturbed_phase = self.angle;

        // apply each zernike term
        for term in zernike_coefficients {
            let mode_effect = term.length * (term.angle * 3.0).cos();
            perturbed_phase += mode_effect;
        }

        Self {
            length: self.length,
            angle: perturbed_phase,
            blade: self.blade, // preserve blade grade
        }
    }

    fn otf(&self, focal_length: f64, wavelength: f64) -> Self {
        // convert from spatial domain to frequency domain
        let frequency = self.length / (wavelength * focal_length);
        let phase = self.angle + PI / 2.0;

        Self {
            length: frequency,
            angle: phase,
            blade: self.blade, // preserve blade grade
        }
    }

    fn abcd_transform(&self, a: f64, b: f64, c: f64, d: f64) -> Self {
        // apply ABCD matrix as angle transformation
        let h = self.angle.sin(); // height/angle
        let theta = self.angle; // angle

        // abcd transformations for ray tracing
        let new_h = a * h + b * theta;
        let new_theta = c * h + d * theta;

        // convert back to geonum representation
        Self {
            length: self.length,
            angle: new_theta.atan2(new_h),
            blade: self.blade, // preserve blade grade
        }
    }

    fn magnify(&self, magnification: f64) -> Self {
        // magnification affects intensity (inverse square law) and angle scaling
        let image_intensity = 1.0 / (magnification * magnification);

        // image point has inverted angle and scaled height
        let image_angle = -self.angle / magnification;

        Self {
            length: self.length * image_intensity,
            angle: image_angle,
            blade: self.blade, // preserve blade grade
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::TWO_PI;

    #[test]
    fn it_applies_optical_magnification() {
        // create input ray/object point
        let object = Geonum {
            length: 4.0,
            angle: PI / 6.0, // 30 degrees
            blade: 1,        // vector (grade 1)
        };

        // test 2x magnification
        let magnified_2x = object.magnify(2.0);

        // verify intensity follows inverse square law (1/m²)
        let expected_intensity_2x = 4.0 / (2.0 * 2.0);
        assert_eq!(magnified_2x.length, expected_intensity_2x);

        // verify angle is inverted and scaled
        let expected_angle_2x = (-PI / 6.0 / 2.0) % TWO_PI;
        assert_eq!(magnified_2x.angle, expected_angle_2x);

        // test 0.5x magnification (minification)
        let magnified_half = object.magnify(0.5);

        // verify intensity increases with minification
        let expected_intensity_half = 4.0 / (0.5 * 0.5);
        assert_eq!(magnified_half.length, expected_intensity_half);

        // verify angle is inverted and scaled
        let expected_angle_half = (-PI / 6.0 / 0.5) % TWO_PI;
        assert_eq!(magnified_half.angle, expected_angle_half);
    }
}
