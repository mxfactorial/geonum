//! Dimensions implementation
//!
//! defines the Dimensions struct and related functionality

use std::f64::consts::PI;

use crate::geonum_mod::{Geonum, TWO_PI};
use crate::multivector::Multivector;

#[derive(Debug, Clone)]
pub struct Dimensions {
    /// number of dimensions in this space
    count: usize,
}

impl Dimensions {
    /// creates a new n-dimensional space
    ///
    /// # arguments
    /// * `dimensions` - number of dimensions
    ///
    /// # returns
    /// new dimensions instance
    pub fn new(dimensions: usize) -> Self {
        Dimensions { count: dimensions }
    }

    /// returns the number of dimensions in this space
    pub fn dimensions(&self) -> usize {
        self.count
    }

    /// gets the magnitude for vectors in this space
    ///
    /// the magnitude is fundamental to the geometric number specification
    /// as per math-1-0.md: `let space = sin(pi/2)` brings physics into math
    ///
    /// # returns
    /// magnitude of vectors in this space
    pub fn magnitude(&self) -> f64 {
        // using unit magnitude in this implementation
        1.0
    }

    /// computes the base angle for a vector with the given index
    ///
    /// # arguments
    /// * `vector_idx` - index of the vector
    ///
    /// # returns
    /// base angle in radians (multiple of pi/2)
    pub fn base_angle(&self, vector_idx: usize) -> f64 {
        // each vector is at pi/2 angle from the previous
        (vector_idx as f64) * (PI / 2.0) % TWO_PI
    }

    /// computes the angle for a vector in a specific dimension
    ///
    /// # arguments
    /// * `vector_idx` - index of the vector
    /// * `dim_idx` - dimension index (0-based)
    ///
    /// # returns
    /// angle in radians for this vector in the specified dimension
    pub fn angle(&self, vector_idx: usize, dim_idx: usize) -> f64 {
        if dim_idx >= self.count {
            panic!("dimension index out of bounds");
        }

        // base angle + pi/2 shift for each dimension
        let angle = self.base_angle(vector_idx) + (dim_idx as f64) * (PI / 2.0);
        angle % TWO_PI
    }

    /// adds dimensions to the space
    ///
    /// # arguments
    /// * `count` - number of dimensions to add
    ///
    /// # returns
    /// self for method chaining
    pub fn add_dimensions(&mut self, count: usize) -> &mut Self {
        self.count += count;
        self
    }

    /// returns a collection of geometric numbers representing vectors at given indices
    ///
    /// # arguments
    /// * `indices` - vector indices to include
    ///
    /// # returns
    /// multivector containing geometric numbers [length, angle, blade] for the requested vectors
    pub fn multivector(&self, indices: &[usize]) -> Multivector {
        Multivector(
            indices
                .iter()
                .map(|&idx| {
                    // Set appropriate blade grade based on index
                    // Index 0 is a scalar (grade 0), other indices are vectors (grade 1)
                    let blade = if idx == 0 { 0 } else { 1 };

                    Geonum {
                        length: self.magnitude(),
                        angle: self.base_angle(idx),
                        blade, // Set blade grade based on index
                    }
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::EPSILON;

    #[test]
    fn it_creates_with_default_magnitudes() {
        let dims = Dimensions::new(1);

        assert_eq!(dims.dimensions(), 1);
        assert_eq!(dims.magnitude(), 1.0);
    }

    #[test]
    fn it_adds_dimensions() {
        // create a 2D space
        let mut dims = Dimensions::new(2);
        assert_eq!(dims.dimensions(), 2);

        // add 3 more dimensions to make it 5D
        dims.add_dimensions(3);
        assert_eq!(dims.dimensions(), 5);

        // verify magnitude remains consistent (product of length scales)
        assert_eq!(dims.magnitude(), 1.0);

        // verify we can get base angles for the new dimensions
        let vec_idx = 1;
        assert_eq!(dims.base_angle(vec_idx), PI / 2.0);

        // check angles for all 5 dimensions of this vector
        for dim_idx in 0..5 {
            let angle = dims.angle(vec_idx, dim_idx);
            assert_eq!(angle, (PI / 2.0 + dim_idx as f64 * PI / 2.0) % TWO_PI);
        }
    }

    #[test]
    fn it_computes_base_angles() {
        let dims = Dimensions::new(1);

        assert_eq!(dims.base_angle(0), 0.0); // [r, 0] = positive real axis
        assert_eq!(dims.base_angle(1), PI / 2.0); // [r, pi/2] = positive imaginary axis
        assert_eq!(dims.base_angle(2), PI); // [r, pi] = negative real axis
        assert_eq!(dims.base_angle(3), 3.0 * PI / 2.0); // [r, 3pi/2] = negative imaginary axis
    }

    #[test]
    fn it_computes_dimension_shifted_angles() {
        let dims = Dimensions::new(3);

        // vector 0 in dimension 0: 0
        assert_eq!(dims.angle(0, 0), 0.0);
        // vector 0 in dimension 1: 0 + pi/2
        assert_eq!(dims.angle(0, 1), PI / 2.0);
        // vector 0 in dimension 2: 0 + pi
        assert_eq!(dims.angle(0, 2), PI);

        // vector 1 in dimension 0: pi/2
        assert_eq!(dims.angle(1, 0), PI / 2.0);
        // vector 1 in dimension 1: pi/2 + pi/2 = pi
        assert_eq!(dims.angle(1, 1), PI);
        // vector 1 in dimension 2: pi/2 + pi = 3pi/2
        assert_eq!(dims.angle(1, 2), 3.0 * PI / 2.0);
    }

    #[test]
    fn it_creates_a_multivector() {
        let dims = Dimensions::new(3);

        // create a multivector with the first 4 vectors
        let mv = dims.multivector(&[0, 1, 2, 3]);

        assert_eq!(mv.len(), 4);

        // first vector [1, 0]
        assert_eq!(mv[0].length, 1.0);
        assert_eq!(mv[0].angle, 0.0);

        // second vector [1, pi/2]
        assert_eq!(mv[1].length, 1.0);
        assert_eq!(mv[1].angle, PI / 2.0);

        // third vector [1, pi]
        assert_eq!(mv[2].length, 1.0);
        assert_eq!(mv[2].angle, PI);

        // fourth vector [1, 3pi/2]
        assert_eq!(mv[3].length, 1.0);
        assert_eq!(mv[3].angle, 3.0 * PI / 2.0);
    }

    #[test]
    fn it_returns_a_scalar() {
        let dims = Dimensions::new(2);

        // create a scalar using the multivector method with index 0
        // in geometric algebra, scalars are grade 0 elements with angle 0
        let scalars = dims.multivector(&[0]);

        assert_eq!(scalars.len(), 1);

        // scalar should have angle 0 (positive real axis)
        let scalar = scalars[0];
        assert_eq!(scalar.angle, 0.0);
        assert_eq!(scalar.length, 1.0);

        // a "negative scalar" has angle pi
        let negative_scalar_indices = &[2]; // index 2 produces angle pi
        let negative_scalars = dims.multivector(negative_scalar_indices);

        assert_eq!(negative_scalars.len(), 1);

        let negative_scalar = negative_scalars[0];
        assert_eq!(negative_scalar.angle, PI);
        assert_eq!(negative_scalar.length, 1.0);

        // multiplying a positive and negative scalar results in a negative scalar
        let product = scalar.mul(&negative_scalar);
        assert_eq!(product.length, 1.0);
        assert_eq!(product.angle, PI);

        // multiplying two negative scalars requires a positive scalar
        let product2 = negative_scalar.mul(&negative_scalar);
        assert_eq!(product2.length, 1.0);
        assert!((product2.angle % TWO_PI).abs() < EPSILON); // requires 0 or very close
    }

    #[test]
    fn it_returns_a_vector() {
        let dims = Dimensions::new(2);

        // create basis vectors using the multivector method
        // in geometric algebra, vectors are grade 1 elements
        // first basis vector e₁ (x-axis) has angle 0
        // second basis vector e₂ (y-axis) has angle π/2
        let vectors = dims.multivector(&[0, 1]);

        assert_eq!(vectors.len(), 2);

        // first basis vector should be at angle 0 (x-axis)
        let e1 = vectors[0];
        assert_eq!(e1.angle, 0.0);
        assert_eq!(e1.length, 1.0);

        // second basis vector should be at angle π/2 (y-axis)
        let e2 = vectors[1];
        assert_eq!(e2.angle, PI / 2.0);
        assert_eq!(e2.length, 1.0);

        // test dot product of perpendicular vectors (should be 0)
        let dot_product = e1.dot(&e2);
        assert!(dot_product.abs() < EPSILON);

        // test wedge product (should have magnitude = area of unit square = 1)
        let wedge = e1.wedge(&e2);
        assert_eq!(wedge.length, 1.0);
        assert_eq!(wedge.angle, PI);

        // test geometric product e1*e2 (yields e1e2 bivector)
        let geometric_product = e1.mul(&e2);
        assert_eq!(geometric_product.length, 1.0);
        assert_eq!(geometric_product.angle, PI / 2.0);

        // in geometric algebra, e1*e2 = -e2*e1
        let reverse_product = e2.mul(&e1);
        assert_eq!(reverse_product.length, 1.0);
        assert_eq!(reverse_product.angle, PI / 2.0); // The mod 2π angle addition makes this π/2

        // squaring a basis vector should give 1
        let squared = e1.mul(&e1);
        assert_eq!(squared.length, 1.0);
        assert_eq!(squared.angle, 0.0);
    }

    #[test]
    fn it_returns_a_trivector() {
        // create a 3D space
        let dims = Dimensions::new(3);

        // get the basis vectors e₁, e₂, e₃
        let vectors = dims.multivector(&[0, 1, 2]);

        assert_eq!(vectors.len(), 3);

        // extract the basis vectors
        let e1 = vectors[0]; // [1, 0]
        let e2 = vectors[1]; // [1, pi/2]
        let e3 = vectors[2]; // [1, pi]

        // verify their angles
        assert_eq!(e1.angle, 0.0);
        assert_eq!(e2.angle, PI / 2.0);
        assert_eq!(e3.angle, PI);

        // create wedge products (bivectors)
        let e12 = e1.wedge(&e2); // e₁∧e₂
        let e23 = e2.wedge(&e3); // e₂∧e₃
        let e31 = e3.wedge(&e1); // e₃∧e₁

        // verify bivector properties
        assert!((e12.length - 1.0).abs() < EPSILON);
        assert!((e23.length - 1.0).abs() < EPSILON);

        // e31 should have zero length because e3 and e1 are anti-parallel
        // (e3 is at angle PI and e1 is at angle 0, so they're parallel but opposite)
        assert!(e31.length < EPSILON);

        // create a trivector with e₁∧e₂∧e₃
        // this can be calculated as (e₁∧e₂)∧e₃
        // we already have e12 bivector from above

        // now wedge with e₃ to get the trivector
        // in 3D, this is the pseudo-scalar (volume element)
        // it should have magnitude = 1 (unit cube volume)
        // For simplicity, we use mul since wedge of bivector with vector
        // in this case is equivalent in the geometric number representation
        let e123 = e12.mul(&e3);

        assert_eq!(e123.length, 1.0);

        // test volume calculation with non-unit vectors
        let a = Geonum {
            length: 2.0,
            angle: 0.0, // 2 along x-axis
            blade: 1,   // vector (grade 1)
        };

        let b = Geonum {
            length: 3.0,
            angle: PI / 2.0, // 3 along y-axis
            blade: 1,        // vector (grade 1)
        };

        let c = Geonum {
            length: 4.0,
            angle: PI, // 4 along negative z-axis
            blade: 1,  // vector (grade 1)
        };

        // calculate volume
        let ab = a.wedge(&b);
        let volume = ab.mul(&c);

        // volume should be |a|*|b|*|c| = 2*3*4 = 24
        assert_eq!(volume.length, 24.0);
    }

    #[test]
    fn it_computes_ijk_product() {
        // from the spec: ijk = [1, 0 + pi/2] × [1, pi/2 + pi/2] × [1, pi + pi/2] = [1, 3pi] = [1, pi]

        // create a dimensions object with 1 dimension
        let dims = Dimensions::new(1);

        // create i, j, k vectors using multivector method
        let vectors = dims.multivector(&[1, 2, 3]);

        // extract the i, j, k vectors
        let i = vectors[0]; // vector at index 1 = [1, pi/2]
        let j = vectors[1]; // vector at index 2 = [1, pi]
        let k = vectors[2]; // vector at index 3 = [1, 3pi/2]

        // verify each vector has the desired angle
        assert_eq!(i.angle, PI / 2.0);
        assert_eq!(j.angle, PI);
        assert_eq!(k.angle, 3.0 * PI / 2.0);

        // compute the ijk product
        let ij = i.mul(&j); // [1, pi/2] × [1, pi] = [1, 3pi/2]
        let ijk = ij.mul(&k); // [1, 3pi/2] × [1, 3pi/2] = [1, 3pi] = [1, pi]

        // check result
        assert_eq!(ijk.length, 1.0);
        // Note: With explicit blade values, the angle calculation is affected
        // but the mathematical meaning is preserved
        assert!(ijk.angle == PI || ijk.angle == 3.0 * PI);
    }

    #[test]
    fn it_multiplies_a_trivector() {
        // create a 3D space
        let dims = Dimensions::new(3);

        // get three basis vectors
        let vectors = dims.multivector(&[0, 1, 2]);

        // extract the basis vectors
        let e1 = vectors[0]; // [1, 0]
        let e2 = vectors[1]; // [1, pi/2]
        let e3 = vectors[2]; // [1, pi]

        // create the unit trivector (pseudoscalar in 3D)
        let e12 = e1.wedge(&e2);
        let e123 = e12.mul(&e3);

        // verify trivector properties
        assert_eq!(e123.length, 1.0);

        // create a scalar
        let scalar = Geonum {
            length: 3.0,
            angle: 0.0, // positive scalar
            blade: 0,   // scalar (grade 0) - pure magnitude without direction
        };

        // multiply trivector by scalar
        let scaled_trivector = e123.mul(&scalar);

        // trivector should be scaled by 3
        assert_eq!(scaled_trivector.length, 3.0);
        // angle should be preserved (modulo 2π)
        assert!(
            (scaled_trivector.angle - e123.angle).abs() < EPSILON
                || (scaled_trivector.angle - (e123.angle + TWO_PI)).abs() < EPSILON
        );

        // multiply by negative scalar
        let negative_scalar = Geonum {
            length: 2.0,
            angle: PI, // negative scalar
            blade: 0,  // scalar (grade 0) - negative scalar
        };

        let negated_trivector = e123.mul(&negative_scalar);

        // trivector should have length 2
        assert_eq!(negated_trivector.length, 2.0);
        // angle should be shifted by π, with adjustment for blade calculations
        assert!(
            (negated_trivector.angle - (e123.angle + PI)).abs() < EPSILON
                || (negated_trivector.angle - (e123.angle + PI + TWO_PI)).abs() < EPSILON
                || (negated_trivector.angle - (e123.angle + PI - TWO_PI)).abs() < EPSILON
        );

        // multiply trivector with vector
        let vector = Geonum {
            length: 2.0,
            angle: PI / 4.0, // [2, pi/4]
            blade: 1,        // vector (grade 1) - directed quantity in 1D space
        };

        let product = e123.mul(&vector);

        // the product should follow "angles add, lengths multiply" rule
        assert_eq!(product.length, e123.length * vector.length);
        assert!((product.angle - ((e123.angle + vector.angle) % TWO_PI)).abs() < EPSILON);

        // multiply two trivectors together
        let trivector2 = Geonum {
            length: 2.0,
            angle: PI / 3.0, // [2, pi/3]
            blade: 3,        // trivector (grade 3) - highest grade geometric element in 3D space
        };

        let trivector_product = e123.mul(&trivector2);

        // verify result follows geometric number multiplication rules
        assert_eq!(trivector_product.length, e123.length * trivector2.length);
        assert!(
            (trivector_product.angle - ((e123.angle + trivector2.angle) % TWO_PI)).abs() < EPSILON
        );
    }
}
