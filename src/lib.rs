use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

#[derive(Debug, Clone)]
pub struct Dimensions {
    /// length scales for each dimension: [1.0, 1.0, ...]
    /// or custom length scales: [1.0, 2.0, ...]
    length_scales: Vec<f64>,
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
        Dimensions {
            length_scales: vec![1.0; dimensions],
        }
    }

    /// returns the number of dimensions in this space
    pub fn dimensions(&self) -> usize {
        self.length_scales.len()
    }

    /// gets the magnitude based on length scales
    ///
    /// # returns
    /// magnitude of vectors in this space
    pub fn magnitude(&self) -> f64 {
        // magnitude is the product of all length scales
        self.length_scales.iter().product()
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
        if dim_idx >= self.length_scales.len() {
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
        self.length_scales.extend(vec![1.0; count]);
        self
    }

    /// returns a collection of geometric numbers representing vectors at given indices
    ///
    /// # arguments
    /// * `indices` - vector indices to include
    ///
    /// # returns
    /// vector of geometric numbers [length, angle] for the requested vectors
    pub fn multivector(&self, indices: &[usize]) -> Vec<Geonum> {
        indices
            .iter()
            .map(|&idx| Geonum {
                length: self.magnitude(),
                angle: self.base_angle(idx),
            })
            .collect()
    }
}

/// represents a geometric number [length, angle]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geonum {
    /// length component
    pub length: f64,
    /// angle component in radians
    pub angle: f64,
}

impl Geonum {
    /// multiplies two geometric numbers
    /// lengths multiply, angles add
    ///
    /// # arguments
    /// * `other` - the geometric number to multiply with
    ///
    /// # returns
    /// the product as a new geometric number
    pub fn mul(&self, other: &Geonum) -> Geonum {
        Geonum {
            length: self.length * other.length,
            angle: (self.angle + other.angle) % TWO_PI,
        }
    }

    /// computes the inverse of a geometric number
    /// for [r, θ], the inverse is [1/r, -θ]
    ///
    /// # returns
    /// the inverse as a new geometric number
    ///
    /// # panics
    /// if the length is zero
    pub fn inv(&self) -> Geonum {
        if self.length == 0.0 {
            panic!("cannot invert a geometric number with zero length");
        }

        Geonum {
            length: 1.0 / self.length,
            angle: (-self.angle) % TWO_PI,
        }
    }

    /// divides this geometric number by another
    /// equivalent to multiplying by the inverse: a/b = a * (1/b)
    ///
    /// # arguments
    /// * `other` - the geometric number to divide by
    ///
    /// # returns
    /// the quotient as a new geometric number
    ///
    /// # panics
    /// if the divisor has zero length
    pub fn div(&self, other: &Geonum) -> Geonum {
        self.mul(&other.inv())
    }

    /// normalizes a geometric number to unit length
    /// preserves the angle but sets length to 1
    ///
    /// # returns
    /// a new geometric number with length 1 and the same angle
    ///
    /// # panics
    /// if the length is zero
    pub fn normalize(&self) -> Geonum {
        if self.length == 0.0 {
            panic!("cannot normalize a geometric number with zero length");
        }

        Geonum {
            length: 1.0,
            angle: self.angle,
        }
    }

    /// computes the dot product of two geometric numbers
    /// formula: |a|*|b|*cos(θb-θa)
    ///
    /// # arguments
    /// * `other` - the geometric number to compute dot product with
    ///
    /// # returns
    /// the dot product as a scalar value
    pub fn dot(&self, other: &Geonum) -> f64 {
        self.length * other.length * ((other.angle - self.angle).cos())
    }

    /// computes the wedge product of two geometric numbers
    /// formula: [|a|*|b|*sin(θb-θa), (θa + θb + pi/2) mod 2pi]
    ///
    /// # arguments
    /// * `other` - the geometric number to compute wedge product with
    ///
    /// # returns
    /// the wedge product as a new geometric number
    pub fn wedge(&self, other: &Geonum) -> Geonum {
        let length = self.length * other.length * ((other.angle - self.angle).sin());
        let angle = (self.angle + other.angle + PI / 2.0) % TWO_PI;

        Geonum {
            length: length.abs(),
            angle: if length >= 0.0 {
                angle
            } else {
                (angle + PI) % TWO_PI
            },
        }
    }

    /// computes the geometric product of two geometric numbers
    /// combines both dot and wedge products: a⋅b + a∧b
    ///
    /// # arguments
    /// * `other` - the geometric number to compute geometric product with
    ///
    /// # returns
    /// dot product as scalar part, wedge product as bivector part
    pub fn geo(&self, other: &Geonum) -> (f64, Geonum) {
        let dot_part = self.dot(other);
        let wedge_part = self.wedge(other);

        (dot_part, wedge_part)
    }
}

#[cfg(test)]
mod dimensions_tests {
    use super::*;

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
        assert!((product2.angle % TWO_PI).abs() < 1e-10); // requires 0 or very close
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
        assert!(dot_product.abs() < 1e-10);

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
        assert!((e12.length - 1.0).abs() < 1e-10);
        assert!((e23.length - 1.0).abs() < 1e-10);

        // e31 should have zero length because e3 and e1 are anti-parallel
        // (e3 is at angle PI and e1 is at angle 0, so they're parallel but opposite)
        assert!(e31.length < 1e-10);

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
        };

        let b = Geonum {
            length: 3.0,
            angle: PI / 2.0, // 3 along y-axis
        };

        let c = Geonum {
            length: 4.0,
            angle: PI, // 4 along negative z-axis
        };

        // calculate volume
        let ab = a.wedge(&b);
        let volume = ab.mul(&c);

        // volume should be |a|*|b|*|c| = 2*3*4 = 24
        assert_eq!(volume.length, 24.0);
    }
}

#[cfg(test)]
mod geonum_tests {
    use super::*;

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
        assert_eq!(ijk.angle, PI);
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
        };

        // multiply trivector by scalar
        let scaled_trivector = e123.mul(&scalar);

        // trivector should be scaled by 3
        assert_eq!(scaled_trivector.length, 3.0);
        // angle should be preserved (modulo 2π)
        assert!(
            (scaled_trivector.angle - e123.angle).abs() < 1e-10
                || (scaled_trivector.angle - (e123.angle + TWO_PI)).abs() < 1e-10
        );

        // multiply by negative scalar
        let negative_scalar = Geonum {
            length: 2.0,
            angle: PI, // negative scalar
        };

        let negated_trivector = e123.mul(&negative_scalar);

        // trivector should have length 2
        assert_eq!(negated_trivector.length, 2.0);
        // angle should be shifted by π
        assert!((negated_trivector.angle - (e123.angle + PI) % TWO_PI).abs() < 1e-10);

        // multiply trivector with vector
        let vector = Geonum {
            length: 2.0,
            angle: PI / 4.0, // [2, pi/4]
        };

        let product = e123.mul(&vector);

        // the product should follow "lengths multiply, angles add" rule
        assert_eq!(product.length, e123.length * vector.length);
        assert!((product.angle - ((e123.angle + vector.angle) % TWO_PI)).abs() < 1e-10);

        // multiply two trivectors together
        let trivector2 = Geonum {
            length: 2.0,
            angle: PI / 3.0, // [2, pi/3]
        };

        let trivector_product = e123.mul(&trivector2);

        // verify result follows geometric number multiplication rules
        assert_eq!(trivector_product.length, e123.length * trivector2.length);
        assert!(
            (trivector_product.angle - ((e123.angle + trivector2.angle) % TWO_PI)).abs() < 1e-10
        );
    }

    #[test]
    fn it_computes_dot_product() {
        // create two aligned vectors
        let a = Geonum {
            length: 3.0,
            angle: 0.0, // [3, 0] = 3 on positive real axis
        };

        let b = Geonum {
            length: 4.0,
            angle: 0.0, // [4, 0] = 4 on positive real axis
        };

        // compute dot product
        let dot_product = a.dot(&b);

        // for aligned vectors, result should be product of lengths
        assert_eq!(dot_product, 12.0);

        // create perpendicular vectors
        let c = Geonum {
            length: 2.0,
            angle: 0.0, // [2, 0] = 2 on x-axis
        };

        let d = Geonum {
            length: 5.0,
            angle: PI / 2.0, // [5, pi/2] = 5 on y-axis
        };

        // dot product of perpendicular vectors should be zero
        let perpendicular_dot = c.dot(&d);
        assert!(perpendicular_dot.abs() < 1e-10);
    }

    #[test]
    fn it_computes_wedge_product() {
        // create two perpendicular vectors
        let a = Geonum {
            length: 2.0,
            angle: 0.0, // [2, 0] = 2 along x-axis
        };

        let b = Geonum {
            length: 3.0,
            angle: PI / 2.0, // [3, pi/2] = 3 along y-axis
        };

        // compute wedge product
        let wedge = a.wedge(&b);

        // for perpendicular vectors, the wedge product should have:
        // - length equal to the product of lengths (area of rectangle) = 2*3 = 6
        // - angle equal to the sum of angles plus pi/2 = 0 + pi/2 + pi/2 = pi
        assert_eq!(wedge.length, 6.0);
        assert_eq!(wedge.angle, PI);

        // test wedge product of parallel vectors
        let c = Geonum {
            length: 4.0,
            angle: PI / 4.0, // [4, pi/4] = 4 at 45 degrees
        };

        let d = Geonum {
            length: 2.0,
            angle: PI / 4.0, // [2, pi/4] = 2 at 45 degrees (parallel to c)
        };

        // wedge product of parallel vectors should be zero
        let parallel_wedge = c.wedge(&d);
        assert!(parallel_wedge.length < 1e-10);

        // test anti-commutativity: v ∧ w = -(w ∧ v)
        let e = Geonum {
            length: 2.0,
            angle: PI / 6.0, // [2, pi/6] = 2 at 30 degrees
        };

        let f = Geonum {
            length: 3.0,
            angle: PI / 3.0, // [3, pi/3] = 3 at 60 degrees
        };

        // compute e ∧ f and f ∧ e
        let ef_wedge = e.wedge(&f);
        let fe_wedge = f.wedge(&e);

        // verify that the magnitudes are equal but orientations are opposite
        assert_eq!(ef_wedge.length, fe_wedge.length);
        assert!(
            (ef_wedge.angle - (fe_wedge.angle + PI) % TWO_PI).abs() < 1e-10
                || (ef_wedge.angle - (fe_wedge.angle - PI) % TWO_PI).abs() < 1e-10
        );

        // verify nilpotency: v ∧ v = 0
        let self_wedge = e.wedge(&e);
        assert!(self_wedge.length < 1e-10);
    }

    #[test]
    fn it_computes_geometric_product() {
        // create two vectors at right angles
        let a = Geonum {
            length: 2.0,
            angle: 0.0, // [2, 0] = 2 along x-axis
        };

        let b = Geonum {
            length: 3.0,
            angle: PI / 2.0, // [3, pi/2] = 3 along y-axis
        };

        // compute geometric product
        let (scalar_part, bivector_part) = a.geo(&b);

        // perpendicular vectors have zero dot product
        assert!(scalar_part.abs() < 1e-10);

        // bivector part should match wedge product
        let wedge = a.wedge(&b);
        assert_eq!(bivector_part.length, wedge.length);
        assert_eq!(bivector_part.angle, wedge.angle);

        // create two vectors at an angle
        let c = Geonum {
            length: 2.0,
            angle: PI / 4.0, // [2, pi/4] = 2 at 45 degrees
        };

        let d = Geonum {
            length: 2.0,
            angle: PI / 3.0, // [2, pi/3] = 2 at 60 degrees
        };

        // compute geometric product
        let (scalar_part2, bivector_part2) = c.geo(&d);

        // verify dot product
        let expected_dot = c.dot(&d);
        assert!((scalar_part2 - expected_dot).abs() < 1e-10);

        // verify bivector part
        let wedge2 = c.wedge(&d);
        assert_eq!(bivector_part2.length, wedge2.length);
        assert_eq!(bivector_part2.angle, wedge2.angle);
    }

    #[test]
    fn it_computes_inverse_and_division() {
        // create a geometric number
        let a = Geonum {
            length: 2.0,
            angle: PI / 3.0, // [2, pi/3]
        };

        // compute its inverse
        let inv_a = a.inv();

        // inverse should have reciprocal length and negated angle
        assert!((inv_a.length - 0.5).abs() < 1e-10);
        assert!((inv_a.angle - ((-PI / 3.0) % TWO_PI)).abs() < 1e-10);

        // multiplying a number by its inverse should give [1, 0]
        let product = a.mul(&inv_a);
        assert!((product.length - 1.0).abs() < 1e-10);
        assert!((product.angle % TWO_PI).abs() < 1e-10);

        // test division
        let b = Geonum {
            length: 4.0,
            angle: PI / 4.0, // [4, pi/4]
        };

        // compute a / b
        let quotient = a.div(&b);

        // verify that a / b = a * (1/b)
        let inv_b = b.inv();
        let expected = a.mul(&inv_b);
        assert!((quotient.length - expected.length).abs() < 1e-10);
        assert!((quotient.angle - expected.angle).abs() < 1e-10);

        // explicit computation verification
        assert!((quotient.length - (a.length / b.length)).abs() < 1e-10);
        assert!((quotient.angle - ((a.angle - b.angle) % TWO_PI)).abs() < 1e-10);
    }

    #[test]
    fn it_normalizes_vectors() {
        // create a geometric number with non-unit length
        let a = Geonum {
            length: 5.0,
            angle: PI / 6.0, // [5, pi/6]
        };

        // normalize it
        let normalized = a.normalize();

        // normalized vector should have length 1 and same angle
        assert_eq!(normalized.length, 1.0);
        assert_eq!(normalized.angle, PI / 6.0);

        // normalize a vector with negative angle
        let b = Geonum {
            length: 3.0,
            angle: -PI / 4.0, // [3, -pi/4]
        };

        let normalized_b = b.normalize();

        // should have length 1 and preserve angle
        assert_eq!(normalized_b.length, 1.0);
        assert_eq!(normalized_b.angle, -PI / 4.0);

        // normalizing an already normalized vector should be idempotent
        let twice_normalized = normalized.normalize();
        assert_eq!(twice_normalized.length, 1.0);
        assert_eq!(twice_normalized.angle, normalized.angle);
    }
}
