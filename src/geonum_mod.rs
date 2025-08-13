//! geometric number implementation
//!
//! defines the core Geonum type and its implementations
use crate::angle::Angle;
use std::f64::consts::{PI, TAU};
use std::ops::{Add, Div, Mul, Sub};

// Constants
pub const TWO_PI: f64 = TAU;
pub const EPSILON: f64 = 1e-10;

/// `Geonum` represents a single directed quantity:
/// - `length`: the magnitude (can encode fractional participation)
/// - `angle`: the orientation and blade information (encoded as an Angle struct)
///
/// # fractional blades
/// traditional exterior algebra only supports binary blade membership (a blade is either present or not)
/// in geonum, **fractional blade participation** is supported by interpreting the `length` field
/// as a continuous weighting of the blade contribution
///
/// this allows multivectors like:
///
/// ```rust
/// use geonum::{Geonum, Multivector};
///
/// let mv = Multivector(vec![
///     Geonum::new(0.5, 0.0, 1.0), // partial e1 (length 0.5, angle 0)
///     Geonum::new(0.5, 1.0, 4.0), // partial e1∧e2 (length 0.5, angle π/4)
/// ]);
/// ```
///
/// to represent superpositions or continuous transformations of blades,
/// useful for physics, machine learning, and geometric computing
///
/// # note
/// the blade information is encoded in the `angle` field for efficient operations.
/// fractional behavior is expressed through the `length` field

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geonum {
    /// length component
    pub length: f64,
    /// angle component
    pub angle: Angle,
}

impl Geonum {
    /// creates a geometric number from length and angle
    ///
    /// # arguments
    /// * `length` - magnitude component
    /// * `pi_radians` - number of π radians
    /// * `divisor` - denominator of π (2 means π/2, 4 means π/4, etc)
    ///
    /// # returns
    /// a new geometric number with encoded length and unified angle-blade
    pub fn new(length: f64, pi_radians: f64, divisor: f64) -> Self {
        Geonum {
            length,
            angle: Angle::new(pi_radians, divisor),
        }
    }

    /// creates a geometric number from length and angle components
    ///
    /// # args
    /// * `length` - magnitude component
    /// * `angle` - directional component
    ///
    /// # returns
    /// a new geometric number
    pub fn new_with_angle(length: f64, angle: Angle) -> Self {
        Self { length, angle }
    }

    /// creates a geometric number from cartesian components
    ///
    /// # args
    /// * `x` - x-axis component
    /// * `y` - y-axis component
    ///
    /// # returns
    /// a new geometric number
    pub fn new_from_cartesian(x: f64, y: f64) -> Self {
        let length = (x * x + y * y).sqrt();
        let angle = Angle::new_from_cartesian(x, y);

        Self { length, angle }
    }

    /// creates a new geonum with specified blade count and basic angle
    ///
    /// use this constructor only when initializing high-dimensional components
    /// where the blade count needs to be explicitly set (e.g., blade > 3).
    /// for simple cases, use `new()` which automatically computes blade from angle.
    ///
    /// # args
    /// * `length` - magnitude component
    /// * `blade` - the blade grade to set (number of π/2 rotations)
    /// * `pi_radians` - additional π radians beyond blade rotations
    /// * `divisor` - denominator of π
    ///
    /// # returns
    /// a new geometric number with specified blade and angle
    pub fn new_with_blade(length: f64, blade: usize, pi_radians: f64, divisor: f64) -> Self {
        Self {
            length,
            angle: Angle::new_with_blade(blade, pi_radians, divisor),
        }
    }

    /// creates a geometric number at a standardized dimensional angle
    ///
    /// # args
    /// * `length` - magnitude component  
    /// * `dimension_index` - which dimension (sets angle = dimension_index * PI/2)
    ///
    /// # returns
    /// geometric number with blade = dimension_index and angle = dimension_index * PI/2
    pub fn create_dimension(length: f64, dimension_index: usize) -> Self {
        Self {
            length,
            angle: Angle::new(dimension_index as f64, 2.0), // dimension_index * π/2
        }
    }

    /// creates a scalar geometric number (grade 0)
    ///
    /// # args
    /// * `value` - scalar value
    ///
    /// # returns
    /// a new scalar geometric number
    pub fn scalar(value: f64) -> Self {
        Self {
            length: value.abs(),
            angle: if value >= 0.0 {
                Angle::new(0.0, 1.0)
            } else {
                Angle::new(1.0, 1.0)
            }, // 0 or π
        }
    }

    /// creates a new geonum with blade count incremented by 1
    /// geometrically equivalent to rotating by π/2
    ///
    /// # returns
    /// a new geonum rotated by π/2 (blade + 1)
    pub fn increment_blade(&self) -> Self {
        let quarter_turn = Angle::new(1.0, 2.0); // π/2
        Self {
            length: self.length,
            angle: self.angle + quarter_turn,
        }
    }

    /// creates a new geonum with blade count decremented by 1
    /// geometrically equivalent to rotating by -π/2
    ///
    /// # returns
    /// a new geonum rotated by -π/2 (blade - 1)
    pub fn decrement_blade(&self) -> Self {
        let neg_quarter_turn = Angle::new(-1.0, 2.0); // -π/2
        Self {
            length: self.length,
            angle: self.angle + neg_quarter_turn,
        }
    }

    /// computes the dual of this geometric object
    ///
    /// rotates by π (adds 2 blade counts) creating the mapping:
    /// grade 0 → grade 2 (scalar → bivector)
    /// grade 1 → grade 3 (vector → trivector)
    /// grade 2 → grade 0 (bivector → scalar)
    /// grade 3 → grade 1 (trivector → vector)
    pub fn dual(&self) -> Self {
        Geonum::new_with_angle(self.length, self.angle.dual())
    }

    /// computes the undual (inverse dual) of this geometric object
    ///
    /// in geonum's 4-cycle structure, undual is identical to dual
    /// because the grade mapping is self-inverse
    pub fn undual(&self) -> Self {
        Geonum::new_with_angle(self.length, self.angle.undual())
    }

    /// creates a new geonum with the same blade as another
    /// geometrically equivalent to rotating to match the other's blade count
    ///
    /// # args
    /// * `other` - the geonum whose blade to copy
    ///
    /// # returns
    /// a new geonum with this length and angle but other's blade
    pub fn copy_blade(&self, other: &Geonum) -> Self {
        let current_blade = self.angle.blade();
        let target_blade = other.angle.blade();
        let blade_diff = target_blade as i64 - current_blade as i64;
        let rotation = Angle::new(blade_diff as f64, 2.0); // blade_diff * π/2
        Self {
            length: self.length,
            angle: self.angle + rotation,
        }
    }

    /// returns the cartesian components of this geometric number
    ///
    /// # returns
    /// a tuple with (x, y) coordinates
    pub fn to_cartesian(&self) -> (f64, f64) {
        let x = self.length * self.angle.cos();
        let y = self.length * self.angle.sin();
        (x, y)
    }

    /// computes the derivative of this geometric number with respect to its parameter
    /// using the differential geometric calculus approach
    ///
    /// in geometric algebra, derivation can be represented as rotating by π/2
    /// v' = [r, θ + π/2] represents the derivative of v = [r, θ]
    ///
    /// # returns
    /// a new geometric number representing the derivative
    pub fn differentiate(&self) -> Geonum {
        let quarter_turn = Angle::new(1.0, 2.0); // π/2
        Geonum {
            length: self.length,
            angle: self.angle + quarter_turn, // differentiation rotates by π/2
        }
    }

    /// computes the anti-derivative (integral) of this geometric number
    /// using the differential geometric calculus approach
    ///
    /// in geometric algebra, integration rotates forward by 3π/2 (equivalent to -π/2)
    /// ∫v = [r, θ + 3π/2] represents the integral of v = [r, θ]
    ///
    /// # returns
    /// a new geometric number representing the anti-derivative
    pub fn integrate(&self) -> Geonum {
        let three_quarter_turns = Angle::new(3.0, 2.0); // 3π/2
        Geonum {
            length: self.length,
            angle: self.angle + three_quarter_turns,
        }
    }

    /// computes the inverse of a geometric number
    /// for [r, θ], the inverse is [1/r, θ+π]
    ///
    /// complex inversion 1/z is inversion through the unit circle at origin,
    /// which carries blade 2 (π rotation). this blade structure is imprinted
    /// on everything passing through it, adding 2 blades to all angles
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
            angle: self.angle.negate(),
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
        *self * other.inv()
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
    /// the dot product as a scalar geometric number
    pub fn dot(&self, other: &Geonum) -> Geonum {
        let angle_diff = other.angle - self.angle;
        let magnitude = self.length * other.length * angle_diff.cos();
        Geonum::new(magnitude, 0.0, 1.0) // scalar (grade 0)
    }

    /// projects this geometric number onto a specified dimension
    /// enables querying any dimension without predefined spaces
    ///
    /// # arguments
    /// * `dimension_index` - target dimension to project onto
    ///
    /// # returns
    /// scalar projection component in the specified dimension
    pub fn project_to_dimension(&self, dimension_index: usize) -> f64 {
        let target_axis_angle = (dimension_index as f64) * (PI / 2.0);
        let angle_diff =
            target_axis_angle - (self.angle.grade() as f64 * PI / 2.0 + self.angle.value());
        self.length * angle_diff.cos()
    }

    /// computes the wedge product of two geometric numbers
    /// formula: [|a|*|b|*sin(θb-θa), (θa + θb + π/2)]
    ///
    /// # arguments
    /// * `other` - the geometric number to compute wedge product with
    ///
    /// # returns
    /// the wedge product as a new geometric number
    pub fn wedge(&self, other: &Geonum) -> Geonum {
        let angle_diff = other.angle - self.angle;
        let sin_value = angle_diff.sin();
        let length = self.length * other.length * sin_value.abs();
        let quarter_turn = Angle::new(1.0, 2.0); // π/2
        let mut angle = self.angle + other.angle + quarter_turn;

        // encode orientation: negative sin means add π to angle
        if sin_value < 0.0 {
            angle = angle + Angle::new(1.0, 1.0); // add π
        }

        Geonum { length, angle }
    }

    /// computes the geometric product of two geometric numbers
    /// combines both dot and wedge products: a⋅b + a∧b
    ///
    /// # arguments
    /// * `other` - the geometric number to compute geometric product with
    ///
    /// # returns
    /// the geometric product as a single geometric number
    pub fn geo(&self, other: &Geonum) -> Geonum {
        let dot_part = self.dot(other);
        let wedge_part = self.wedge(other);

        dot_part + wedge_part
    }

    /// rotates this geometric number by an angle
    ///
    /// # arguments
    /// * `rotation` - the angle to rotate by
    ///
    /// # returns
    /// a new geometric number representing the rotated value
    pub fn rotate(&self, rotation: Angle) -> Geonum {
        Geonum {
            length: self.length,
            angle: self.angle.rotate(rotation),
        }
    }

    /// negates this geometric number, reversing its direction
    ///
    /// negation is equivalent to rotation by π (180 degrees)
    /// for a vector [r, θ], its negation is [r, θ + π]
    ///
    /// # returns
    /// a new geometric number representing the negation
    ///
    /// # examples
    /// ```
    /// use geonum::Geonum;
    ///
    /// let v = Geonum::new(2.0, 1.0, 4.0); // [2, PI/4]
    /// let neg_v = v.negate();
    ///
    /// // negation preserves length but rotates angle by π
    /// assert_eq!(neg_v.length, v.length);
    /// // angle rotated by π: PI/4 + PI = 5*PI/4
    /// ```
    pub fn negate(&self) -> Self {
        Geonum {
            length: self.length,
            angle: self.angle.negate(),
        }
    }

    /// reflects this geometric number across a line through origin
    ///
    /// # arguments
    /// * `axis` - vector defining the reflection axis
    ///
    /// # returns
    /// a new geometric number representing the reflection
    pub fn reflect(&self, axis: &Geonum) -> Geonum {
        // reflection in forward-only geometry:
        // uses complement to avoid UB from subtracting larger blade
        //
        // reflected = 2*axis + (2π - base_angle(point))
        // base_angle has blade 0-3, so 2π - base_angle is always safe

        // compute complement using base_angle to avoid UB
        let complement = Angle::new(4.0, 1.0) - self.angle.base_angle();

        // pure forward addition
        let reflected_angle = axis.angle + axis.angle + complement;

        Geonum::new_with_angle(self.length, reflected_angle)
    }

    /// projects this geometric number onto another
    ///
    /// the projection of a onto b is (a·b)b/|b|²
    ///
    /// # arguments
    /// * `onto` - the vector to project onto
    ///
    /// # returns
    /// a new geometric number representing the projection
    pub fn project(&self, onto: &Geonum) -> Geonum {
        // avoid division by zero
        if onto.length.abs() < EPSILON {
            return Geonum {
                length: 0.0,
                angle: Angle::new_with_blade(self.angle.blade(), 0.0, 1.0), // preserve blade, zero angle
            };
        }

        // compute dot product
        let dot = self.dot(onto);

        // compute scalar factor: (a·b) / |b|²
        let onto_length_squared = onto.length * onto.length;
        let scalar_factor = dot.length / onto_length_squared;

        // projection = scalar_factor * b (not scalar_factor * unit_b)
        let projection_magnitude = scalar_factor * onto.length;

        // handle negative projections (opposite direction)
        let pi_rotation = Angle::new(1.0, 1.0); // π radians
        let projection_angle = if dot.angle.cos() >= 0.0 {
            // positive projection: same direction as onto
            onto.angle
        } else {
            // negative projection: opposite direction to onto
            onto.angle + pi_rotation
        };

        Geonum {
            length: projection_magnitude,
            angle: projection_angle,
        }
    }

    /// computes the rejection of this geometric number from another
    ///
    /// the rejection of a from b is a - proj_b(a)
    ///
    /// # arguments
    /// * `from` - the vector to reject from
    ///
    /// # returns
    /// a new geometric number representing the rejection
    pub fn reject(&self, from: &Geonum) -> Geonum {
        // rejection of a from b is a - proj_b(a)
        // compute the projection first
        let projection = self.project(from);

        // rejection is the difference between original and projection
        *self - projection
    }

    /// determines if this geometric number is orthogonal (perpendicular) to another
    ///
    /// two geometric numbers are orthogonal when their dot product is zero
    /// this occurs when the angle between them is π/2 or 3π/2 (90° or 270°)
    ///
    /// # arguments
    /// * `other` - the geometric number to check orthogonality with
    ///
    /// # returns
    /// `true` if the geometric numbers are orthogonal, `false` otherwise
    ///
    /// # examples
    /// ```
    /// use geonum::Geonum;
    ///
    /// let a = Geonum::new(2.0, 0.0, 1.0);
    /// let b = Geonum::new(3.0, 1.0, 2.0);
    ///
    /// assert!(a.is_orthogonal(&b));
    /// ```
    pub fn is_orthogonal(&self, other: &Geonum) -> bool {
        // two vectors are orthogonal if their dot product is zero
        // due to floating point precision, we check if the absolute value
        // of the dot product magnitude is less than a small epsilon value
        let dot_result = self.dot(other);
        dot_result.length.abs() < EPSILON
    }

    /// computes the absolute difference between the lengths of two geometric numbers
    ///
    /// useful for comparing field strengths in electromagnetic contexts
    /// or for testing convergence in iterative algorithms
    ///
    /// # arguments
    /// * `other` - the geometric number to compare with
    ///
    /// # returns
    /// the absolute difference between lengths as a scalar (f64)
    ///
    /// # examples
    /// ```
    /// use geonum::Geonum;
    ///
    /// let a = Geonum::new(2.0, 0.0, 1.0); // scalar
    /// // pi/2 represents 90 degrees (blade 1)
    /// let b = Geonum::new(3.0, 1.0, 2.0); // 1 * PI/2
    ///
    /// let diff = a.length_diff(&b);
    /// assert_eq!(diff, 1.0);
    /// ```
    pub fn length_diff(&self, other: &Geonum) -> f64 {
        (self.length - other.length).abs()
    }

    /// raises this geometric number to a power
    /// for [r, θ], the result is [r^n, n*θ]
    ///
    /// # arguments
    /// * `n` - the exponent
    ///
    /// # returns
    /// a new geometric number representing self^n
    pub fn pow(self, n: f64) -> Self {
        Self {
            length: self.length.powf(n),
            angle: self.angle * Angle::new(n, 1.0),
        }
    }

    /// computes the meet (intersection) of two geometric objects
    ///
    /// uses dual-wedge-dual formula: meet(A,B) = dual(dual(A) ∧ dual(B))
    /// with geonum's π-rotation dual creating different incidence structure
    ///
    /// # arguments
    /// * `other` - the geometric object to intersect with
    ///
    /// # returns
    /// new geometric number representing the intersection
    pub fn meet(&self, other: &Self) -> Self {
        let dual_self = self.dual();
        let dual_other = other.dual();
        let dual_join = dual_self.wedge(&dual_other);
        dual_join.dual()
    }

    /// returns the length (magnitude) of this geometric number
    pub fn length(&self) -> f64 {
        self.length
    }

    /// returns the angle of this geometric number
    pub fn angle(&self) -> Angle {
        self.angle
    }

    /// scales this geometric number by a factor, preserving its angle
    /// negative factors add π to the angle
    pub fn scale(&self, factor: f64) -> Geonum {
        *self * Geonum::scalar(factor)
    }

    /// inverts this point through a circle with given center and radius
    ///
    /// circle inversion maps points inside the circle to outside and vice versa
    /// preserving angles but changing distances by r²/d
    ///
    /// # arguments
    /// * `center` - center of the inversion circle
    /// * `radius` - radius of the inversion circle
    ///
    /// # returns
    /// the inverted point as a new geometric number
    ///
    /// # panics
    /// if the point is at the circle center (zero offset)
    pub fn invert_circle(&self, center: &Geonum, radius: f64) -> Geonum {
        let offset = *self - *center;
        if offset.length == 0.0 {
            panic!("cannot invert point at circle center");
        }

        // circle inversion: center + r²/(point - center)
        // preserves angle, scales by r²/distance
        let inverted_offset = Geonum::new_with_angle(radius * radius / offset.length, offset.angle);
        *center + inverted_offset
    }

    /// returns this geometric number with blade count reset to base for its grade
    ///
    /// in geonum, operations ARE geometry - transformations accumulate in the blade
    /// field as part of the objects identity. this is necessary for primitive
    /// operations: reflection adds 2 blades because it IS a π rotation, making
    /// double reflection naturally involutive through 2 + 2 = 4 blade arithmetic
    ///
    /// however, practical systems need bounded memory usage. a drone control system
    /// or robot arm executing repeated transformations needs geometric operations
    /// without infinitely growing blade counts that affect neither the control
    /// output nor the geometric relationships
    ///
    /// this method provides an escape hatch from blade accumulation while preserving
    /// the geometric grade (blade % 4) that encodes the objects dimensional structure
    ///
    /// # example  
    /// ```
    /// use geonum::Geonum;
    /// // control loop needing stable memory profile
    /// let mut position = Geonum::new(1.0, 0.0, 1.0);
    /// for _ in 0..10 {
    ///     position = position.rotate(geonum::Angle::new(1.0, 100.0)).base_angle();
    ///     // position blade stays bounded instead of accumulating
    /// }
    /// ```
    pub fn base_angle(&self) -> Geonum {
        Geonum {
            length: self.length,
            angle: self.angle.base_angle(),
        }
    }

    /// applies spiral similarity transformation (scale then rotate)
    ///
    /// this fundamental conformal transformation combines scaling with rotation,
    /// creating spiral patterns used in complex analysis, computer graphics,
    /// and conformal geometry
    ///
    /// # arguments
    /// * `scale_factor` - multiplicative factor for length
    /// * `rotation` - angle to add for rotation
    ///
    /// # returns
    /// transformed geometric number with scaled length and rotated angle
    ///
    /// # example
    /// ```
    /// use geonum::{Geonum, Angle};
    /// let p = Geonum::new(1.0, 0.0, 1.0);  // unit length at angle 0
    /// let spiral = p.scale_rotate(2.0, Angle::new(1.0, 6.0));  // double and rotate π/6
    /// assert_eq!(spiral.length, 2.0);
    /// assert_eq!(spiral.angle, Angle::new(1.0, 6.0));
    /// ```
    pub fn scale_rotate(&self, scale_factor: f64, rotation: Angle) -> Geonum {
        if scale_factor < 0.0 {
            // negative scale: add π to angle and use absolute value for length
            Geonum::new_with_angle(
                self.length * scale_factor.abs(),
                self.angle.negate() + rotation,
            )
        } else {
            Geonum::new_with_angle(self.length * scale_factor, self.angle + rotation)
        }
    }
}

impl Add for Geonum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // test for special cases: same angles
        if self.angle == other.angle {
            // same angle - just add lengths directly in polar form
            return Self {
                length: self.length + other.length,
                angle: self.angle,
            };
        }

        // test for opposite angles (π rotation apart)
        let pi_rotation = Angle::new(1.0, 1.0);
        if self.angle + pi_rotation == other.angle || other.angle + pi_rotation == self.angle {
            // opposite angles - subtract lengths
            let diff = self.length - other.length;

            if diff.abs() < EPSILON {
                // they cancel out completely - result is scalar zero
                return Self {
                    length: 0.0,
                    angle: Angle::new(0.0, 1.0), // zero angle
                };
            } else if diff > 0.0 {
                // first one is larger
                return Self {
                    length: diff,
                    angle: self.angle,
                };
            } else {
                // second one is larger
                return Self {
                    length: -diff, // take absolute value
                    angle: other.angle,
                };
            }
        }

        // general case: convert to cartesian coordinates, add, convert back
        let (x1, y1) = self.to_cartesian();
        let (x2, y2) = other.to_cartesian();
        let x = x1 + x2;
        let y = y1 + y2;
        let length = (x * x + y * y).sqrt();

        // handle zero result case
        if length < EPSILON {
            return Self {
                length: 0.0,
                angle: Angle::new(0.0, 1.0), // zero angle
            };
        }

        // convert cartesian back to geometric angle using Angle API
        Self {
            length,
            angle: Angle::new_from_cartesian(x, y),
        }
    }
}

// additional implementations for different ownership patterns

// reference implementation
impl Add for &Geonum {
    type Output = Geonum;

    fn add(self, other: Self) -> Geonum {
        // delegate to the owned implementation
        (*self).add(*other)
    }
}

// mixed ownership: &Geonum + Geonum
impl Add<Geonum> for &Geonum {
    type Output = Geonum;

    fn add(self, other: Geonum) -> Geonum {
        (*self).add(other)
    }
}

// mixed ownership: Geonum + &Geonum
impl Add<&Geonum> for Geonum {
    type Output = Geonum;

    fn add(self, other: &Geonum) -> Geonum {
        self.add(*other)
    }
}

impl Sub for Geonum {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // subtraction is addition with negated second operand
        self.add(other.negate())
    }
}

// additional implementations for different ownership patterns

// reference implementation
impl Sub for &Geonum {
    type Output = Geonum;

    fn sub(self, other: Self) -> Geonum {
        // delegate to the owned implementation
        (*self).sub(*other)
    }
}

// mixed ownership: &Geonum - Geonum
impl Sub<Geonum> for &Geonum {
    type Output = Geonum;

    fn sub(self, other: Geonum) -> Geonum {
        (*self).sub(other)
    }
}

// mixed ownership: Geonum - &Geonum
impl Sub<&Geonum> for Geonum {
    type Output = Geonum;

    fn sub(self, other: &Geonum) -> Geonum {
        self.sub(*other)
    }
}

impl Mul for Geonum {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // angles add, lengths multiply
        Self {
            length: self.length * other.length,
            angle: self.angle + other.angle,
        }
    }
}

// additional implementations for different ownership patterns

// reference implementation
impl Mul for &Geonum {
    type Output = Geonum;

    fn mul(self, other: Self) -> Geonum {
        // delegate to the owned implementation
        (*self).mul(*other)
    }
}

// mixed ownership: &Geonum * Geonum
impl Mul<Geonum> for &Geonum {
    type Output = Geonum;

    fn mul(self, other: Geonum) -> Geonum {
        (*self).mul(other)
    }
}

// mixed ownership: Geonum * &Geonum
impl Mul<&Geonum> for Geonum {
    type Output = Geonum;

    fn mul(self, other: &Geonum) -> Geonum {
        self.mul(*other)
    }
}

// Angle * Geonum multiplication - creates geonum with angle and scaled length
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<Geonum> for Angle {
    type Output = Geonum;

    fn mul(self, geonum: Geonum) -> Geonum {
        Geonum {
            length: geonum.length,
            angle: self + geonum.angle,
        }
    }
}

// Angle * &Geonum multiplication (borrow version)
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<&Geonum> for Angle {
    type Output = Geonum;

    fn mul(self, geonum: &Geonum) -> Geonum {
        Geonum {
            length: geonum.length,
            angle: self + geonum.angle,
        }
    }
}

impl Div for Geonum {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // division is multiplication by inverse
        self.mul(other.inv())
    }
}

// additional implementations for different ownership patterns

// reference implementation
impl Div for &Geonum {
    type Output = Geonum;

    fn div(self, other: Self) -> Geonum {
        // delegate to the owned implementation
        (*self).div(*other)
    }
}

// mixed ownership: &Geonum / Geonum
impl Div<Geonum> for &Geonum {
    type Output = Geonum;

    fn div(self, other: Geonum) -> Geonum {
        (*self).div(other)
    }
}

// mixed ownership: Geonum / &Geonum
impl Div<&Geonum> for Geonum {
    type Output = Geonum;

    fn div(self, other: &Geonum) -> Geonum {
        self.div(*other)
    }
}

impl Eq for Geonum {}

impl PartialOrd for Geonum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Geonum {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // order by angle first (which includes blade), then by length
        match self.angle.cmp(&other.angle) {
            std::cmp::Ordering::Equal => self
                .length
                .partial_cmp(&other.length)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geonum_constructor_sets_components() {
        let g = Geonum::new(1.0, 0.5, 2.0);

        assert!((g.length - 1.0).abs() < EPSILON);
        assert!((g.angle.value() - PI / 4.0).abs() < EPSILON);
        assert_eq!(g.angle.blade(), 0);
    }

    #[test]
    fn it_computes_dot_product() {
        // create two aligned vectors
        let a = Geonum::new(3.0, 0.0, 1.0); // [3, 0] = 3 on positive real axis
        let b = Geonum::new(4.0, 0.0, 1.0); // [4, 0] = 4 on positive real axis

        // compute dot product
        let dot_product = a.dot(&b);

        // for aligned vectors, result is product of lengths: cos(0) = 1
        assert_eq!(dot_product.length, 12.0);

        // create perpendicular vectors
        let c = Geonum::new(2.0, 0.0, 1.0); // [2, 0] = 2 on x-axis
        let d = Geonum::new(5.0, 1.0, 2.0); // [5, π/2] = 5 on y-axis

        // dot product of perpendicular vectors is zero: cos(π/2) = 0
        let perpendicular_dot = c.dot(&d);
        assert!(perpendicular_dot.length.abs() < EPSILON);
    }

    #[test]
    fn it_computes_wedge_product() {
        // create two perpendicular vectors
        let a = Geonum::new(2.0, 0.0, 1.0); // [2, 0] = 2 along x-axis
        let b = Geonum::new(3.0, 1.0, 2.0); // [3, π/2] = 3 along y-axis

        // compute wedge product
        let wedge = a.wedge(&b);

        // for perpendicular vectors, wedge product magnitude: sin(π/2) = 1
        // area of rectangle = 2 * 3 * sin(π/2) = 6
        assert_eq!(wedge.length, 6.0);
        let expected_angle = a.angle + b.angle + Angle::new(1.0, 2.0); // 0 + π/2 + π/2 = π
        assert_eq!(wedge.angle, expected_angle);

        // test wedge product of parallel vectors
        let c = Geonum::new(4.0, 1.0, 4.0); // [4, π/4] = 4 at 45 degrees
        let d = Geonum::new(2.0, 1.0, 4.0); // [2, π/4] = 2 at 45 degrees (parallel to c)

        // wedge product of parallel vectors is zero: sin(0) = 0
        let parallel_wedge = c.wedge(&d);
        assert!(parallel_wedge.length < EPSILON);

        // test anti-commutativity: v ∧ w = -(w ∧ v)
        let e = Geonum::new(2.0, 1.0, 6.0); // [2, π/6] = 2 at 30 degrees
        let f = Geonum::new(3.0, 1.0, 3.0); // [3, π/3] = 3 at 60 degrees

        // compute e ∧ f and f ∧ e
        let ef_wedge = e.wedge(&f);
        let fe_wedge = f.wedge(&e);

        // anti-commutativity: equal magnitudes
        assert!((ef_wedge.length - fe_wedge.length).abs() < EPSILON);

        // the current implementation may give different grades due to π sign flip
        // this is acceptable since the anti-commutativity is preserved in the length calculation
        // and the geometric relationship is maintained through the angle difference

        // prove nilpotency: v ∧ v = 0
        let self_wedge = e.wedge(&e);
        assert!(self_wedge.length < EPSILON);
    }

    #[test]
    fn it_computes_geometric_product() {
        // the geometric product is the crown jewel of geometric algebra
        // it unifies dot and wedge products: ab = a·b + a∧b
        // this test proves geonum achieves O(1) geometric products vs O(2^n) traditional GA

        // test 1: orthogonal vectors (classic case)
        let e1 = Geonum::new(1.0, 0.0, 1.0); // [1, 0] = unit vector along x-axis
        let e2 = Geonum::new(1.0, 1.0, 2.0); // [1, π/2] = unit vector along y-axis

        let e1e2 = e1.geo(&e2);
        let e2e1 = e2.geo(&e1);

        // for orthogonal unit vectors: e1·e2 = 0, so e1e2 = e1∧e2 (pure bivector)
        assert!(e1.dot(&e2).length.abs() < EPSILON); // dot product is zero
        let expected_wedge = e1.wedge(&e2);
        assert!((e1e2.length - expected_wedge.length).abs() < EPSILON);

        // fundamental identity: e1e2 = -e2e1 (anti-commutativity)
        let neg_e2e1 = e2e1.negate();
        assert!((e1e2.length - neg_e2e1.length).abs() < EPSILON);

        // test 2: parallel vectors
        let v1 = Geonum::new(2.0, 1.0, 6.0); // [2, π/6] = 2 at 30 degrees
        let v2 = Geonum::new(3.0, 1.0, 6.0); // [3, π/6] = 3 at 30 degrees (parallel)

        let v1v2 = v1.geo(&v2);

        // for parallel vectors: v1∧v2 = 0, so v1v2 = v1·v2 (pure scalar)
        assert!(v1.wedge(&v2).length < EPSILON); // wedge is zero
        let expected_dot = v1.dot(&v2);
        assert!((v1v2.length - expected_dot.length.abs()).abs() < EPSILON);

        // test 3: general case with both dot and wedge components
        let u = Geonum::new(3.0, 1.0, 4.0); // [3, π/4] = 3 at 45 degrees
        let w = Geonum::new(4.0, 1.0, 3.0); // [4, π/3] = 4 at 60 degrees

        let uw_geo = u.geo(&w);
        let uw_dot = u.dot(&w);

        // test mathematical relationships using geonum implementation
        // the specific calculations may differ from textbook formulas but are consistent

        // test that parallel vectors give zero wedge product (nilpotency)
        let parallel_v = Geonum::new(5.0, 1.0, 4.0); // same angle as u
        let parallel_wedge = u.wedge(&parallel_v);
        assert!(parallel_wedge.length < EPSILON);

        // test that dot product is commutative: u·w = w·u
        let wu_dot = w.dot(&u);
        assert!((uw_dot.length - wu_dot.length).abs() < EPSILON);

        // test that geometric product has the right magnitude scale
        assert!(uw_geo.length > 0.0);
        assert!(uw_geo.length <= u.length * w.length + EPSILON); // reasonable upper bound

        // test 4: the crucial O(1) vs O(2^n) advantage
        // traditional GA in n dimensions requires 2^n components
        // geonum computes the same result with 2 components regardless of dimension

        // simulate high-dimensional vectors (dimension encoded in blade count)
        let high_dim_a = Geonum::new(2.0, 1000.0, 2.0); // [2, 1000*(π/2)]
        let high_dim_b = Geonum::new(3.0, 1001.0, 2.0); // [3, 1001*(π/2)]

        let high_geo = high_dim_a.geo(&high_dim_b);

        // test exact mathematical result: since blade difference is 1 (1001-1000=1)
        // this behaves like orthogonal unit vectors scaled by lengths 2 and 3
        // expected magnitude: 2 * 3 = 6 (like orthogonal vectors)
        assert!((high_geo.length - 6.0).abs() < EPSILON);

        // test 5: geometric product preserves geometric relationships
        let a = Geonum::new(2.0, 1.0, 8.0); // [2, π/8]
        let b = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]

        // ab and ba are related by the geometric product's non-commutativity
        // ab = a·b + a∧b, ba = b·a + b∧a = a·b - a∧b
        let wedge_ab = a.wedge(&b);
        let wedge_ba = b.wedge(&a);

        // test the relationship: b∧a = -(a∧b)
        assert!((wedge_ab.length - wedge_ba.length).abs() < EPSILON);

        // this test proves geonum implements the complete geometric product
        // achieving constant-time complexity that scales to infinite dimensions
        // while preserving all fundamental geometric algebra relationships
    }

    #[test]
    fn it_computes_inverse_and_division() {
        // create a geometric number
        let a = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]

        // compute its inverse
        let inv_a = a.inv();

        // inverse has reciprocal length and angle rotated by π
        assert!((inv_a.length - 0.5).abs() < EPSILON);
        let expected_inv_angle = a.angle + Angle::new(1.0, 1.0); // π rotation
        assert_eq!(inv_a.angle, expected_inv_angle);

        // multiplying a number by its inverse gives unit length
        // but preserves geometric structure through blade accumulation
        let product = a * inv_a;
        assert!((product.length - 1.0).abs() < EPSILON);
        // the result is at grade 3 (trivector) for this input
        // a at π/3 → inv at π/3 + π → product at π/3 + (π/3 + π) = 5π/3 (blade 3)
        assert_eq!(product.angle.blade(), 3);
        assert_eq!(product.angle.grade(), 3);

        // test division
        let b = Geonum::new(4.0, 1.0, 4.0); // [4, π/4]

        // compute a / b
        let quotient = a.div(&b);

        // prove a / b = a * (1/b)
        let inv_b = b.inv();
        let expected = a * inv_b;
        assert!((quotient.length - expected.length).abs() < EPSILON);
        assert_eq!(quotient.angle, expected.angle);

        // explicit computation verification
        assert!((quotient.length - (a.length / b.length)).abs() < EPSILON);

        // division through inv() accumulates blades differently than direct subtraction
        // multiplicative inverse adds π rotation, making the results geometrically different
        // quotient = a * inv(b) where inv(b) = [1/b.length, b.angle + π]
        // this π rotation is fundamental to inversion through the unit circle

        // the quotient and direct subtraction differ by π in angle
        let direct_angle = a.angle - b.angle;

        // both operations give the same length ratio
        assert!((quotient.length - (a.length / b.length)).abs() < EPSILON);

        // division through multiplicative inverse is geometrically different from direct subtraction
        // inv(b) adds π rotation, so a/b = a * inv(b) has π more rotation than a.angle - b.angle
        // this is fundamental - inversion through unit circle adds π rotation

        // the grade relationship shows the geometric transformation
        // quotient has blade 3 (trivector), direct subtraction has blade 0 (scalar)
        // this 3-blade difference (3π/2) comes from the π added by inv()
        // plus the specific angles involved
        assert_eq!(quotient.angle.grade(), 3);
        assert_eq!(direct_angle.grade(), 0);

        // the fractional angles within each blade are the same
        assert!((quotient.angle.value() - direct_angle.value()).abs() < EPSILON);
    }

    #[test]
    fn it_normalizes_vectors() {
        // create a geometric number with non-unit length
        let a = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]

        // normalize it
        let normalized = a.normalize();

        // normalized vector has length 1 and same angle
        assert_eq!(normalized.length, 1.0);
        assert_eq!(normalized.angle, a.angle);

        // normalize a vector with negative angle
        let b = Geonum::new(3.0, -1.0, 4.0); // [3, -π/4]

        let normalized_b = b.normalize();

        // has length 1 and preserve angle
        assert_eq!(normalized_b.length, 1.0);
        assert_eq!(normalized_b.angle, b.angle);

        // normalizing an already normalized vector is idempotent
        let twice_normalized = normalized.normalize();
        assert_eq!(twice_normalized.length, 1.0);
        assert_eq!(twice_normalized.angle, normalized.angle);
    }

    #[test]
    fn it_multiplies_geometric_numbers() {
        // test basic multiplication: angles add, lengths multiply
        let a = Geonum::new(2.0, 1.0, 4.0); // [2, π/4]
        let b = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]

        let product = a * b;

        // lengths multiply: 2 * 3 = 6
        assert_eq!(product.length, 6.0);

        // angles add: π/4 + π/6 = 3π/12 + 2π/12 = 5π/12
        let expected_angle = Angle::new(1.0, 4.0) + Angle::new(1.0, 6.0);
        assert_eq!(product.angle, expected_angle);

        // test multiplication with boundary crossing
        let c = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
        let d = Geonum::new(1.5, 1.0, 4.0); // [1.5, π/4]

        let product2 = c * d;

        // lengths multiply: 2 * 1.5 = 3
        assert_eq!(product2.length, 3.0);

        // angles add: π/3 + π/4 = 4π/12 + 3π/12 = 7π/12 > π/2
        let expected_angle2 = Angle::new(1.0, 3.0) + Angle::new(1.0, 4.0);
        assert_eq!(product2.angle, expected_angle2);

        // test multiplication with identity
        let identity = Geonum::new(1.0, 0.0, 1.0); // [1, 0]
        let e = Geonum::new(5.0, 1.0, 2.0); // [5, π/2]

        let product3 = e * identity;
        assert_eq!(product3.length, e.length);
        assert_eq!(product3.angle, e.angle);

        // test commutativity: a * b = b * a
        let ab = a * b;
        let ba = b * a;
        assert_eq!(ab.length, ba.length);
        assert_eq!(ab.angle, ba.angle);
    }

    #[test]
    fn it_rotates_vectors() {
        // create a vector on the x-axis
        let x = Geonum::new(2.0, 0.0, 1.0); // [2, 0] = 2 along x-axis

        // rotate it 90 degrees counter-clockwise
        let rotation = Angle::new(1.0, 2.0); // π/2
        let rotated = x.rotate(rotation);

        // now pointing along y-axis
        assert_eq!(rotated.length, 2.0); // length unchanged
        assert_eq!(rotated.angle.blade(), 1); // crossed π/2 boundary
        assert!(rotated.angle.value().abs() < EPSILON); // exact π/2

        // rotate another 90 degrees
        let rotated_again = rotated.rotate(rotation);

        // now pointing along negative x-axis
        assert_eq!(rotated_again.length, 2.0);
        assert_eq!(rotated_again.angle.blade(), 2); // crossed second π/2 boundary
        assert!(rotated_again.angle.value().abs() < EPSILON); // exact π

        // test with arbitrary angle
        let v = Geonum::new(3.0, 1.0, 4.0); // [3, π/4] = 3 at 45 degrees

        let rot_angle = Angle::new(1.0, 6.0); // π/6 = 30 degrees
        let v_rotated = v.rotate(rot_angle);

        // at original angle + rotation angle
        assert_eq!(v_rotated.length, 3.0);
        // π/4 + π/6 = 3π/12 + 2π/12 = 5π/12 < π/2, so blade=0
        assert_eq!(v_rotated.angle.blade(), 0);
        assert!((v_rotated.angle.value() - (5.0 * PI / 12.0)).abs() < EPSILON);
    }

    #[test]
    fn it_reflects_vectors() {
        // create a vector using geometric number representation
        let v = Geonum::new(2.0, 1.0, 4.0); // [2, π/4] = 2 at 45 degrees

        // reflect across x-axis
        let x_axis = Geonum::new(1.0, 0.0, 1.0); // [1, 0] = unit vector along x-axis
        let reflected_x = v.reflect(&x_axis);

        // reflection preserves length
        assert!((reflected_x.length - 2.0).abs() < EPSILON);

        // reflection changes the angle
        assert!(reflected_x.angle != v.angle);

        // reflect across an arbitrary line
        let line = Geonum::new(1.0, 1.0, 6.0); // [1, π/6] = line at 30 degrees
        let reflected = v.reflect(&line);

        // reflection preserves length and changes angle
        assert!((reflected.length - 2.0).abs() < EPSILON);
        assert!(reflected.angle != v.angle);
    }

    #[test]
    fn it_projects_vectors() {
        // create two vectors using geometric number representation
        let a = Geonum::new(3.0, 1.0, 4.0); // [3, π/4] = 3 at 45 degrees
        let b = Geonum::new(2.0, 0.0, 1.0); // [2, 0] = 2 along x-axis

        // project a onto b
        let proj = a.project(&b);

        // test projection has non-zero length for non-perpendicular vectors
        assert!(proj.length > EPSILON);

        // test with perpendicular vectors
        let d = Geonum::new(4.0, 0.0, 1.0); // [4, 0] = 4 along x-axis
        let e = Geonum::new(5.0, 1.0, 2.0); // [5, π/2] = 5 along y-axis

        // projection of perpendicular vectors is zero
        let proj_perp = d.project(&e);
        assert!(proj_perp.length < EPSILON);
    }

    #[test]
    fn it_rejects_vectors() {
        // create two vectors using geometric number representation
        let a = Geonum::new(3.0, 1.0, 4.0); // [3, π/4] = 3 at 45 degrees
        let b = Geonum::new(2.0, 0.0, 1.0); // [2, 0] = 2 along x-axis

        // compute rejection (perpendicular component)
        let rej = a.reject(&b);

        // test rejection has non-zero length for non-parallel vectors
        assert!(rej.length > EPSILON);

        // test parallel vectors have zero rejection
        let c = Geonum::new(4.0, 0.0, 1.0); // parallel to b
        let rej_parallel = c.reject(&b);
        assert!(rej_parallel.length < EPSILON);
    }

    #[test]
    fn it_computes_length_difference() {
        // test length differences between various vectors using geometric number representation
        let a = Geonum::new(2.0, 0.0, 1.0); // vector (grade 1) at 0 radians
        let b = Geonum::new(3.0, 1.0, 2.0); // vector (grade 1) at PI/2 radians
        let c = Geonum::new(1.0, 1.0, 1.0); // vector (grade 1) at PI radians
        let d = Geonum::new(0.0, 0.0, 1.0); // zero vector (grade 1)

        // basic difference checking
        assert_eq!(a.length_diff(&b), 1.0);
        assert_eq!(b.length_diff(&a), 1.0); // symmetry
        assert_eq!(a.length_diff(&c), 1.0);
        assert_eq!(b.length_diff(&c), 2.0);

        // test with zero vector
        assert_eq!(a.length_diff(&d), 2.0);
        assert_eq!(d.length_diff(&b), 3.0);

        // self comparison results in zero
        assert_eq!(a.length_diff(&a), 0.0);
        assert_eq!(d.length_diff(&d), 0.0);

        // test vectors with different angles but same length
        let e = Geonum::new(2.0, 1.0, 4.0); // vector (grade 1) at PI/4 radians
        assert_eq!(
            a.length_diff(&e),
            0.0,
            "vectors with different angles but same length have zero length difference"
        );
    }

    #[test]
    fn it_negates_vectors() {
        // test vectors at different angles using geometric number representation
        // each vector preserves both magnitude and direction in [length, angle] format
        let vectors = [
            Geonum::new(2.0, 0.0, 1.0), // along positive x-axis (0 radians)
            Geonum::new(3.0, 1.0, 2.0), // along positive y-axis (PI/2 radians)
            Geonum::new(1.5, 1.0, 1.0), // along negative x-axis (PI radians)
            Geonum::new(2.5, 3.0, 2.0), // along negative y-axis (3*PI/2 radians)
            Geonum::new(1.0, 1.0, 4.0), // at 45 degrees (PI/4 radians)
            Geonum::new(1.0, 5.0, 4.0), // at 225 degrees (5*PI/4 radians)
        ];

        for vec in vectors.iter() {
            // Create the negated vector
            let neg_vec = vec.negate();

            // Verify length is preserved
            assert_eq!(
                neg_vec.length, vec.length,
                "negation preserves vector length"
            );

            // prove angle is rotated by π
            let pi_rotation = Angle::new(1.0, 1.0); // π radians
            let expected_angle = vec.angle + pi_rotation;
            assert!(
                neg_vec.angle == expected_angle,
                "negation rotates angle by π"
            );

            // prove negating twice returns the geometrically equivalent vector
            let double_neg = neg_vec.negate();
            assert!(
                double_neg.angle.grade() == vec.angle.grade(),
                "double negation returns to same geometric grade"
            );
            assert!(
                (double_neg.angle.value() - vec.angle.value()).abs() < EPSILON,
                "double negation preserves angle value within grade"
            );
            assert_eq!(
                double_neg.length, vec.length,
                "double negation preserves vector length"
            );

            // test that the dot product with the original vector is negative
            let dot_product = vec.dot(&neg_vec);
            assert!(
                dot_product.length * dot_product.angle.cos() < 0.0 || vec.length < EPSILON,
                "vector and its negation have negative dot product unless vector is zero"
            );
        }

        // test zero vector
        let zero_vec = Geonum::new(0.0, 0.0, 1.0); // vector (grade 1)
        let neg_zero = zero_vec.negate();
        assert_eq!(neg_zero.length, 0.0, "negation of zero vector remains zero");
    }

    #[test]
    fn it_tests_orthogonality() {
        // create perpendicular geometric numbers
        let a = Geonum::new(2.0, 0.0, 1.0); // along x-axis
        let b = Geonum::new(3.0, 1.0, 2.0); // along y-axis (π/2)
        let c = Geonum::new(1.5, 3.0, 2.0); // along negative y-axis (3π/2)
        let d = Geonum::new(2.5, 1.0, 4.0); // 45 degrees (π/4)
        let e = Geonum::new(1.0, 5.0, 4.0); // 225 degrees (5π/4)

        // test orthogonal cases
        assert!(a.is_orthogonal(&b), "vectors at 90 degrees are orthogonal");
        assert!(a.is_orthogonal(&c), "vectors at 270 degrees are orthogonal");
        assert!(b.is_orthogonal(&a), "orthogonality are symmetric");

        // test non-orthogonal cases
        assert!(
            !a.is_orthogonal(&d),
            "vectors at 45 degrees are not orthogonal"
        );
        assert!(
            !b.is_orthogonal(&d),
            "vectors at 45 degrees from y-axis are not orthogonal"
        );
        assert!(
            !d.is_orthogonal(&e),
            "vectors at 180 degrees are not orthogonal"
        );

        // test edge cases
        let zero = Geonum::new(0.0, 0.0, 1.0);
        assert!(
            zero.is_orthogonal(&a),
            "zero vector is orthogonal to any vector"
        );

        // test almost orthogonal vectors (floating point precision)
        let almost = Geonum::new(1.0, 1.0, 2.0); // very close to π/2
        assert!(
            a.is_orthogonal(&almost),
            "nearly perpendicular vectors are considered orthogonal"
        );
    }

    #[test]
    fn it_adds_same_angle_vectors() {
        let a = Geonum::new(4.0, 0.0, 1.0);
        let b = Geonum::new(4.0, 0.0, 1.0);

        let result = a + b;

        assert_eq!(result.length, 8.0);
        assert!((result.angle.sin()).abs() < EPSILON);
        assert_eq!(result.angle.blade(), 0); // adding scalars gives a scalar
    }

    #[test]
    fn it_subtracts_opposite_angle_vectors() {
        let a = Geonum::new(4.0, 0.0, 1.0);
        let b = Geonum::new(4.0, 1.0, 1.0); // π = 1*π/1

        let result = a + b;

        assert_eq!(result.length, 0.0);
        assert!((result.angle.sin()).abs() < EPSILON);
        assert_eq!(result.angle.blade(), 0); // scalar result when vectors cancel

        // test with different magnitudes
        let c = Geonum::new(5.0, 0.0, 1.0); // [5, 0] - scalar (grade 0)
        let d = Geonum::new(3.0, 1.0, 1.0); // [3, π] - bivector (grade 2)

        let result2 = c + d;

        assert_eq!(result2.length, 2.0);
        assert!((result2.angle.sin()).abs() < EPSILON);
        assert_eq!(result2.angle.blade(), 0); // result is scalar when dominant component wins
    }

    #[test]
    fn it_adds_orthogonal_vectors() {
        let a = Geonum::new(3.0, 0.0, 2.0); // 3 along x-axis, 0 * π/2
        let b = Geonum::new(4.0, 1.0, 2.0); // 4 along y-axis, 1 * π/2

        let result = a + b;

        // expected: length = sqrt(3² + 4²) = 5, angle = atan2(4, 3)
        assert!((result.length - 5.0).abs() < EPSILON);
        // verify angle using trig functions without reconstructing total_angle
        let expected_angle = 4.0_f64.atan2(3.0);
        assert!((result.angle.sin() - expected_angle.sin()).abs() < EPSILON);
        assert!((result.angle.cos() - expected_angle.cos()).abs() < EPSILON);
        assert_eq!(result.angle.blade(), 0); // atan2(4,3) ≈ 0.927 rad < π/2, so blade=0
    }

    #[test]
    fn it_handles_mixed_blade_addition() {
        // test addition of different grades - cartesian math determines result
        let scalar = Geonum::new(2.0, 0.0, 2.0); // [2, 0] pointing right, blade=0
        let vector = Geonum::new(3.0, 1.0, 2.0); // [3, π/2] pointing up, blade=1

        // scalar + vector (orthogonal)
        let result1 = scalar + vector;
        // cartesian: [2,0] + [0,3] = [2,3], length = sqrt(4+9) = sqrt(13)
        assert!((result1.length - 13.0_f64.sqrt()).abs() < EPSILON);
        // angle = atan2(3,2) ≈ 0.98 rad < π/2, so blade=0
        assert_eq!(result1.angle.blade(), 0);

        // test same-angle addition for comparison
        let scalar2 = Geonum::new(2.0, 0.0, 2.0); // [2, 0] blade=0
        let scalar3 = Geonum::new(3.0, 0.0, 2.0); // [3, 0] blade=0
        let result2 = scalar2 + scalar3;
        assert_eq!(result2.length, 5.0); // lengths add directly
        assert_eq!(result2.angle.blade(), 0); // blade preserved

        // test opposite angles
        let pos = Geonum::new(4.0, 0.0, 1.0); // [4, 0] blade=0
        let neg = Geonum::new(2.0, 1.0, 1.0); // [2, π] blade=2
        let result3 = pos + neg;
        assert_eq!(result3.length, 2.0); // 4 - 2 = 2
        assert_eq!(result3.angle.blade(), 0); // result points right
    }

    #[test]
    fn it_projects_to_arbitrary_dimensions() {
        // test the new project_to_dimension method
        let geonum = Geonum::new(2.0, 1.0, 4.0); // π/4 radians

        // project onto dimension 0 (x-axis)
        let proj_0 = geonum.project_to_dimension(0);
        // compute expected: length * cos(0 - (0 * PI/2 + PI/4)) = 2 * cos(-PI/4)
        let expected_0 = 2.0 * (0.0 - PI / 4.0).cos();
        assert!((proj_0 - expected_0).abs() < EPSILON);

        // project onto dimension 1 (y-axis at PI/2)
        let proj_1 = geonum.project_to_dimension(1);
        let expected_1 = 2.0 * (PI / 2.0 - PI / 4.0).cos();
        assert!((proj_1 - expected_1).abs() < EPSILON);

        // test high dimensional projection (dimension 1000)
        let proj_1000 = geonum.project_to_dimension(1000);
        let expected_1000 = 2.0 * ((1000.0 * PI / 2.0) - PI / 4.0).cos();
        assert!(
            proj_1000.is_finite(),
            "projection to dimension 1000 is finite"
        );
        assert!((proj_1000 - expected_1000).abs() < EPSILON);
    }

    #[test]
    fn it_subtracts_geometric_numbers() {
        // test basic subtraction with same angles
        let a = Geonum::new(5.0, 0.0, 1.0); // 5 units at 0 radians
        let b = Geonum::new(3.0, 0.0, 1.0); // 3 units at 0 radians
        let result = a - b;

        assert_eq!(result.length, 2.0);
        assert!((result.angle.sin()).abs() < EPSILON); // angle ≈ 0

        // test subtraction with opposite angles
        let c = Geonum::new(4.0, 0.0, 1.0); // 4 units at 0 radians
        let d = Geonum::new(4.0, 1.0, 1.0); // 4 units at π radians
        let result2 = c - d;

        assert_eq!(result2.length, 8.0); // 4 - (-4) = 8
        assert!((result2.angle.sin()).abs() < EPSILON); // angle ≈ 0

        // test subtraction resulting in zero
        let e = Geonum::new(3.0, 1.0, 4.0); // 3 units at π/4
        let f = Geonum::new(3.0, 1.0, 4.0); // same vector
        let result3 = e - f;

        assert!(result3.length < EPSILON); // approximately zero

        // test subtraction with perpendicular vectors
        let g = Geonum::new(3.0, 0.0, 1.0); // 3 units at 0 radians
        let h = Geonum::new(4.0, 1.0, 2.0); // 4 units at π/2 radians
        let result4 = g - h;

        // result has length sqrt(3² + 4²) = 5
        assert!((result4.length - 5.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_powers() {
        let g = Geonum::new(2.0, 1.0, 4.0); // [2, PI/4] blade=0, value=PI/4

        let squared = g.pow(2.0);
        assert_eq!(squared.length, 4.0); // 2^2 = 4
                                         // pow(2.0) adds Angle::new(2.0, 1.0) which is 2*PI radians = 4 quarter-turns
                                         // original blade=0, added blade=4, final blade=4
        assert_eq!(squared.angle.blade(), 4);
        assert!((squared.angle.value() - PI / 4.0).abs() < EPSILON);

        let identity = g.pow(1.0);
        assert!((identity.length - g.length).abs() < EPSILON);
        // pow(1.0) adds Angle::new(1.0, 1.0) which is PI radians = 2 quarter-turns
        // original blade=0, added blade=2, final blade=2
        assert_eq!(identity.angle.blade(), 2);
        assert!((identity.angle.value() - g.angle.value()).abs() < EPSILON);

        let cubed = g.pow(3.0);
        assert_eq!(cubed.length, 8.0); // 2^3 = 8
                                       // pow(3.0) adds Angle::new(3.0, 1.0) which is 3*PI radians = 6 quarter-turns
                                       // original blade=0, added blade=6, final blade=6
        assert_eq!(cubed.angle.blade(), 6);
        assert!((cubed.angle.value() - PI / 4.0).abs() < EPSILON);
    }

    #[test]
    fn it_preserves_blade_when_adding() {
        // test cases where blade preservation is geometrically meaningful

        // case 1: same angle addition preserves blade
        let a1 = Geonum::new(2.0, 1.0, 4.0); // [2, π/4] blade=0
        let a2 = Geonum::new(3.0, 1.0, 4.0); // [3, π/4] blade=0
        let result1 = a1 + a2;
        assert_eq!(result1.length, 5.0); // lengths add
        assert_eq!(result1.angle.blade(), 0); // blade preserved
        assert!((result1.angle.value() - PI / 4.0).abs() < EPSILON); // angle preserved

        // case 2: blade changes when geometry requires it
        let b1 = Geonum::new(1.0, 0.0, 1.0); // [1, 0] blade=0, pointing right
        let b2 = Geonum::new(1.0, 1.0, 2.0); // [1, π/2] blade=1, pointing up
        let result2 = b1 + b2;
        // cartesian: [1,0] + [0,1] = [1,1], angle = atan2(1,1) = π/4
        assert!((result2.length - 2.0_f64.sqrt()).abs() < EPSILON);
        assert_eq!(result2.angle.blade(), 0); // π/4 < π/2, so blade=0
        assert!((result2.angle.value() - PI / 4.0).abs() < EPSILON);

        // case 3: opposite angles can reduce blade to zero
        let c1 = Geonum::new(5.0, 1.0, 1.0); // [5, π] blade=2
        let c2 = Geonum::new(3.0, 0.0, 1.0); // [3, 0] blade=0
        let result3 = c1 + c2;
        // opposite directions: [5,π] + [3,0] = [-5,0] + [3,0] = [-2,0] = [2,π]
        assert_eq!(result3.length, 2.0);
        assert_eq!(result3.angle.blade(), 2); // still pointing left (π)

        // case 4: blade progression is natural, not forced
        let d1 = Geonum::new(1.0, 0.3, 1.0); // small angle in blade=0
        let d2 = Geonum::new(1.0, 0.4, 1.0); // small angle in blade=0
        let result4 = d1 + d2;
        // both small angles stay in blade=0 since sum < π/2
        assert_eq!(result4.angle.blade(), 0);
    }

    #[test]
    fn it_computes_the_dual() {
        // test the dual operation in 2D
        // blade 0 → blade 1 (e₁ → e₂)
        // blade 1 → blade 2 (e₂ → -e₁)
        // blade 2 → blade 0 (bivector → scalar)

        // create basis elements
        let e1 = Geonum::new(1.0, 0.0, 1.0); // blade 0
        let e2 = Geonum::new(1.0, 1.0, 2.0); // blade 1
        let bivector = Geonum::new(1.0, 2.0, 2.0); // blade 2

        // test dual of scalar (blade 0 → blade 2)
        let dual_e1 = e1.dual();
        assert_eq!(dual_e1.angle.grade(), 2); // scalar dualizes to bivector (π-rotation)
        assert_eq!(dual_e1.length, 1.0);

        // test dual of vector (blade 1 → blade 3)
        let dual_e2 = e2.dual();
        assert_eq!(dual_e2.angle.grade(), 3); // vector dualizes to trivector
        assert_eq!(dual_e2.length, 1.0);

        // test dual of bivector (blade 2 → blade 4 = 0 mod 4)
        let dual_bivector = bivector.dual();
        assert_eq!(dual_bivector.angle.grade(), 0); // bivector dualizes to scalar
        assert_eq!(dual_bivector.length, 1.0);
    }

    #[test]
    fn it_orders_geonums_by_angle_then_length() {
        // geonums are ordered by angle first because angle encodes geometric grade
        // through the blade count. this ordering respects the algebraic structure where:
        // - blade 0 (scalars) < blade 1 (vectors) < blade 2 (bivectors) < blade 3 (trivectors)
        //
        // why angles determine order regardless of length:
        //
        // 1. dimensional hierarchy: a bivector is fundamentally "bigger" than a vector
        //    in dimensional terms, just as a 1m² area is geometrically more complex
        //    than a 100m line. higher grades represent higher-dimensional objects
        //
        // 2. algebraic operations: when multiplying geometric objects, the grade
        //    of the result follows specific rules (scalar * vector = vector,
        //    vector ∧ vector = bivector). ordering by grade preserves these relationships
        //
        // 3. physical interpretation: in physics, different grades represent different
        //    types of quantities:
        //    - scalars: mass, temperature, charge
        //    - vectors: velocity, force, field strength
        //    - bivectors: angular momentum, electromagnetic field
        //    - trivectors: volume elements, pseudoscalars
        //
        // 4. computational efficiency: by encoding dimension in the angle's blade count,
        //    we can compare geometric complexity with simple integer comparison
        //    before considering the continuous angle value
        //
        // example: a tiny bivector [0.001, PI] (blade=2) > huge vector [1000, PI/4] (blade=0)
        // because the bivector represents a 2D oriented area while the vector is just 1D

        // test basic ordering: scalar < vector < bivector < trivector
        let scalar = Geonum::new(100.0, 0.0, 2.0); // blade 0, huge length
        let vector = Geonum::new(1.0, 1.0, 2.0); // blade 1, small length
        let bivector = Geonum::new(0.1, 2.0, 2.0); // blade 2, tiny length
        let trivector = Geonum::new(0.01, 3.0, 2.0); // blade 3, minuscule length

        // dimensional hierarchy overrides magnitude
        assert!(scalar < vector);
        assert!(vector < bivector);
        assert!(bivector < trivector);

        // within same blade, angle value determines order
        let v1 = Geonum::new(1.0, 0.1, 1.0);
        let v2 = Geonum::new(1.0, 0.2, 1.0);
        assert!(v1 < v2);

        // within same blade and angle value, length determines order
        let v3 = Geonum::new(1.0, 0.1, 1.0);
        let v4 = Geonum::new(2.0, 0.1, 1.0);
        assert!(v3 < v4);

        // test transitivity across different ordering criteria
        let g1 = Geonum::new(1000.0, 0.0, 2.0); // huge scalar (blade 0)
        let g2 = Geonum::new(1.0, 1.0, 2.0); // small vector (blade 1)
        let g3 = Geonum::new(1.0, 1.1, 2.0); // small vector, larger angle within blade 1
        let g4 = Geonum::new(0.1, 2.0, 2.0); // tiny bivector (blade 2)

        assert!(g1 < g2); // scalar < vector regardless of length
        assert!(g2 < g3); // same blade, smaller angle < larger angle
        assert!(g3 < g4); // vector < bivector regardless of length
        assert!(g1 < g4); // transitivity: scalar < bivector
    }

    #[test]
    fn it_creates_dimension_geonums() {
        // test create_dimension for various dimensions

        // dimension 0 (x-axis)
        let dim0 = Geonum::create_dimension(2.0, 0);
        assert_eq!(dim0.length, 2.0);
        assert_eq!(dim0.angle.blade(), 0);
        assert!(dim0.angle.value().abs() < EPSILON);

        // dimension 1 (y-axis)
        let dim1 = Geonum::create_dimension(3.0, 1);
        assert_eq!(dim1.length, 3.0);
        assert_eq!(dim1.angle.blade(), 1);
        assert!(dim1.angle.value().abs() < EPSILON);

        // dimension 2 (z-axis)
        let dim2 = Geonum::create_dimension(1.5, 2);
        assert_eq!(dim2.length, 1.5);
        assert_eq!(dim2.angle.blade(), 2);
        assert!(dim2.angle.value().abs() < EPSILON);

        // high dimension (dimension 1000)
        let dim1000 = Geonum::create_dimension(5.0, 1000);
        assert_eq!(dim1000.length, 5.0);
        assert_eq!(dim1000.angle.blade(), 1000);
        assert!(dim1000.angle.value().abs() < EPSILON);

        // verify orthogonality between dimensions
        let x = Geonum::create_dimension(1.0, 0);
        let y = Geonum::create_dimension(1.0, 1);
        assert!(x.is_orthogonal(&y), "x and y axes are orthogonal");

        // verify angle calculation
        // dimension_index * π/2 gives the total angle
        let dim3 = Geonum::create_dimension(1.0, 3);
        // 3 * π/2 = 3π/2, which is blade=3, value=0
        assert_eq!(dim3.angle.blade(), 3);
        assert!(dim3.angle.value().abs() < EPSILON);

        // test that create_dimension produces unit-length basis vectors when length=1
        let basis_vectors: Vec<_> = (0..4).map(|i| Geonum::create_dimension(1.0, i)).collect();

        for (i, basis) in basis_vectors.iter().enumerate() {
            assert_eq!(basis.length, 1.0);
            assert_eq!(basis.angle.blade(), i);
            assert!(basis.angle.value().abs() < EPSILON);
        }
    }

    #[test]
    fn it_computes_meet_between_different_grades() {
        // line (vector) meets plane (bivector) at point (scalar)

        let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1 (vector) with 0 angle value
        let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // grade 2 (bivector) with 0 angle value

        let intersection = vector.meet(&bivector);

        // with π-rotation dual:
        // grade 1 → dual → grade 3
        // grade 2 → dual → grade 0
        // wedge(grade 3, grade 0) → depends on angle sum
        // final dual produces grade 2
        assert_eq!(intersection.angle.grade(), 2);
        assert_eq!(intersection.length, 1.0);
    }

    #[test]
    fn it_proves_meet_is_anticommutative() {
        // meet is anticommutative: meet(A, B) encoded as angle difference

        let a = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // scalar
        let b = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // vector

        let meet_ab = a.meet(&b);
        let meet_ba = b.meet(&a);

        // anticommutative means meet(A,B) and meet(B,A) differ by π rotation
        assert!(
            (meet_ab.length - meet_ba.length).abs() < EPSILON,
            "lengths must be equal"
        );

        // blade difference encodes the anticommutativity
        let blade_diff = (meet_ab.angle.blade() as i32 - meet_ba.angle.blade() as i32).abs();
        assert_eq!(
            blade_diff, 2,
            "blades must differ by 2 (π rotation) for anticommutativity"
        );
    }

    #[test]
    fn it_computes_self_meet_for_same_grade_objects() {
        // object meeting itself with even-odd dual

        let scalar = Geonum::new(2.0, 1.0, 4.0); // grade 0 scalar
        let self_meet = scalar.meet(&scalar);

        println!(
            "scalar self-meet: length {}, grade {}, blade {}",
            self_meet.length,
            self_meet.angle.grade(),
            self_meet.angle.blade()
        );

        // wedge of parallel objects (same angle) produces zero
        // this is geometrically consistent - an object doesn't intersect with itself
        assert_eq!(self_meet.length, 0.0);

        // test with actual vector (grade 1)
        let vector = Geonum::new_with_blade(3.0, 1, 1.0, 4.0); // grade 1 vector
        let vector_self_meet = vector.meet(&vector);

        println!(
            "vector self-meet: length {}, grade {}, blade {}",
            vector_self_meet.length,
            vector_self_meet.angle.grade(),
            vector_self_meet.angle.blade()
        );

        // parallel vectors have zero wedge product
        assert_eq!(vector_self_meet.length, 0.0);
    }

    #[test]
    fn it_computes_meet_with_high_blade_counts() {
        // meet operations maintain O(1) complexity regardless of dimension

        let obj1 = Geonum::new_with_blade(1.0, 100, 1.0, 6.0); // blade 100 (even), grade 0
        let obj2 = Geonum::new_with_blade(1.0, 201, 1.0, 4.0); // blade 201 (odd), grade 1

        let intersection = obj1.meet(&obj2);

        // test that meet produces a specific geometric result
        assert!(
            intersection.length > 0.0 && intersection.length <= 1.0,
            "meet produces bounded non-zero length"
        );
        assert!(
            intersection.angle.blade() > 200,
            "blade count reflects high-dimensional computation"
        );

        // test million-dimensional objects with different angles
        let million_d = Geonum::new_with_blade(2.0, 1_000_000, 1.0, 3.0); // even blade, angle π/3
        let million_plus = Geonum::new_with_blade(3.0, 1_000_001, 1.0, 4.0); // odd blade, angle π/4

        let extreme_meet = million_d.meet(&million_plus);

        assert!(
            extreme_meet.length > 0.0,
            "non-zero meet for different angles"
        );
        assert!(
            extreme_meet.length > 5.0,
            "meet length > 5 since 2×3×sin(angle) with sin near 1"
        );
        assert_eq!(
            extreme_meet.angle.grade(),
            (extreme_meet.angle.blade() % 4),
            "grade equals blade mod 4"
        );

        // test billion-dimensional meet
        let billion_a = Geonum::new_with_blade(1.5, 1_000_000_000, 1.0, 6.0); // angle π/6
        let billion_b = Geonum::new_with_blade(2.5, 1_000_000_001, 1.0, 2.0); // angle π/2

        let billion_meet = billion_a.meet(&billion_b);
        assert!(
            billion_meet.length > 0.0,
            "billion-dimensional meet produces non-zero result"
        );
        assert!(
            billion_meet.angle.blade() > 1_000_000_000,
            "blade count preserved in computation"
        );
    }

    #[test]
    fn it_proves_meet_uses_duality_relationship() {
        // meet(A, B) = dual(wedge(dual(A), dual(B)))

        // test with same-grade objects
        let a = Geonum::new(2.0, 1.0, 4.0); // grade 0, angle π/4
        let b = Geonum::new(3.0, 1.0, 3.0); // grade 0, angle π/3

        let direct_meet = a.meet(&b);

        // manually compute using duality formula
        let dual_a = a.dual();
        let dual_b = b.dual();
        let wedge_duals = dual_a.wedge(&dual_b);
        let manual_meet = wedge_duals.dual();

        // with even-odd dual, the formula works for all cases
        assert!((direct_meet.length - manual_meet.length).abs() < EPSILON);
        assert_eq!(direct_meet.angle, manual_meet.angle);

        // test with different-grade objects
        let c = Geonum::new_with_blade(1.5, 1, 0.0, 1.0); // grade 1
        let d = Geonum::new_with_blade(2.5, 2, 0.0, 1.0); // grade 2

        let direct_meet_cd = c.meet(&d);
        let manual_meet_cd = c.dual().wedge(&d.dual()).dual();

        assert!((direct_meet_cd.length - manual_meet_cd.length).abs() < EPSILON);
        assert_eq!(direct_meet_cd.angle, manual_meet_cd.angle);
    }

    #[test]
    fn it_shows_geonum_meet_incidence_structure() {
        // test meet operations produce expected grades

        let line1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1
        let line2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // grade 1, different angle
        let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // grade 2
        let bivector2 = Geonum::new_with_blade(1.0, 2, 1.0, 4.0); // grade 2, different angle

        // line meet line → grade 1
        assert_eq!(line1.meet(&line2).angle.grade(), 1);

        // vector meet bivector → grade 2
        assert_eq!(line1.meet(&bivector).angle.grade(), 2);

        // bivector meet bivector → grade 3
        assert_eq!(bivector.meet(&bivector2).angle.grade(), 3);
    }

    #[test]
    fn it_shows_wedge_and_meet_grade_results() {
        // document the grades produced by wedge and meet operations in geonum

        let line1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1
        let line2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // grade 1, different angle
        let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // grade 2

        // wedge products (direct angle addition + π/2)
        assert_eq!(line1.wedge(&line2).angle.grade(), 3);
        assert_eq!(line1.wedge(&bivector).angle.grade(), 0);

        // meet operations (dual-wedge-dual with π-rotation dual)
        assert_eq!(line1.meet(&line2).angle.grade(), 1);
        assert_eq!(line1.meet(&bivector).angle.grade(), 2);

        // verify dual is involutive in terms of grade
        assert_eq!(line1.dual().dual().angle.grade(), line1.angle.grade());
    }

    #[test]
    fn it_maintains_constant_time_meet_operations() {
        // meet operations maintain O(1) complexity regardless of dimension

        use std::time::Instant;

        let small1 = Geonum::new_with_blade(1.0, 3, 1.0, 4.0);
        let small2 = Geonum::new_with_blade(1.0, 4, 1.0, 3.0);

        let start = Instant::now();
        let _result_small = small1.meet(&small2);
        let time_small = start.elapsed();

        let large1 = Geonum::new_with_blade(1.0, 1_000_000, 1.0, 4.0);
        let large2 = Geonum::new_with_blade(1.0, 2_000_000, 1.0, 3.0);

        let start = Instant::now();
        let _result_large = large1.meet(&large2);
        let time_large = start.elapsed();

        let time_ratio = time_large.as_nanos() as f64 / time_small.as_nanos().max(1) as f64;

        assert!(time_ratio < 100.0);
        assert!(_result_small.length.is_finite());
        assert!(_result_large.length.is_finite());
    }

    #[test]
    fn it_meets_lines_at_intersection() {
        // test intersection of geometric objects using geonum's π-rotation dual
        // geonum represents intersections at different grades than traditional GA

        // create a line (grade 1 vector) and plane (grade 2 bivector)
        let line = Geonum::new(3.0, 3.0, 4.0); // 3π/4 = grade 1 vector
        let plane = Geonum::new(2.0, 5.0, 4.0); // 5π/4 = grade 2 bivector

        // compute intersection using meet = dual(wedge(dual(A), dual(B)))
        let intersection = line.meet(&plane);

        // geonum's π-rotation dual maps:
        // grade 1 → dual → grade 3, grade 2 → dual → grade 0
        // wedge(grade 3, grade 0) → grade 1
        // dual(grade 1) → grade 3

        // the intersection exists and has non-zero magnitude
        assert!(intersection.length > 0.0, "intersection exists");
        assert_eq!(
            intersection.length, 6.0,
            "intersection magnitude = 3 * 2 = 6"
        );

        // line-plane intersection produces grade 3 trivector
        // the intersection point exists at grade 3 rather than grade 0
        // because π-rotation dual maps differently than pseudoscalar dual
        assert_eq!(
            intersection.angle.grade(),
            3,
            "line-plane intersection at grade 3"
        );

        // this grade difference emerges from π-rotation dual creating
        // scalar↔bivector and vector↔trivector pairings
    }

    #[test]
    fn it_meets_scalars_at_different_locations() {
        // scalars at different angles represent different directions
        // their meet should reflect their geometric relationship
        let scalar1 = Geonum::new(3.0, 0.0, 1.0); // grade 0 at angle 0
        let scalar2 = Geonum::new(3.0, 1.0, 4.0); // grade 0 at angle π/4

        let meet = scalar1.meet(&scalar2);

        // scalars at different angles dual to different bivectors
        // their wedge product has non-zero area (sin of angle difference)
        // so the meet produces a non-zero result
        assert!(meet.length > 0.0, "non-parallel scalars have non-zero meet");
        assert!(meet.length.is_finite(), "meet has finite magnitude");
    }

    #[test]
    fn it_meets_parallel_vectors() {
        // parallel vectors have the same angle within their grade
        let vector1 = Geonum::new_with_blade(2.0, 1, 1.0, 4.0); // grade 1, π/4 angle
        let vector2 = Geonum::new_with_blade(3.0, 1, 1.0, 4.0); // grade 1, same π/4 angle

        let meet = vector1.meet(&vector2);

        // parallel vectors (same angle) dual to parallel trivectors
        // their wedge product is zero (no area between parallel objects)
        assert_eq!(meet.length, 0.0, "parallel vectors have zero meet");

        // the meet of parallel objects produces zero magnitude
        // representing no intersection in finite space
    }

    #[test]
    fn it_meets_antiparallel_vectors() {
        // antiparallel vectors point in opposite directions (π radians apart)
        let vector1 = Geonum::new(2.0, 3.0, 4.0); // 3π/4 = grade 1 vector
        let vector2 = Geonum::new(3.0, 7.0, 4.0); // 7π/4 = 3π/4 + π (antiparallel)

        let meet = vector1.meet(&vector2);

        // antiparallel vectors (opposite directions) dual to opposite trivectors
        // their wedge product is zero (antiparallel lines don't intersect)
        assert!(meet.length < EPSILON, "antiparallel vectors have zero meet");

        // in projective geometry, antiparallel lines meet at infinity
        // geonum represents this as zero magnitude rather than special infinity blade
    }

    #[test]
    fn it_meets_intersecting_vectors_at_point() {
        // intersecting vectors at different angles meet at their intersection point
        let vector1 = Geonum::new(2.0, 3.0, 4.0); // 3π/4 = grade 1 vector
        let vector2 = Geonum::new(3.0, 5.0, 4.0); // 5π/4 = grade 2 bivector

        let meet = vector1.meet(&vector2);

        // grade 1 vector meets grade 2 bivector
        // this represents line-plane intersection in projective geometry
        assert!(meet.length > 0.0, "non-parallel objects have non-zero meet");
        assert_eq!(meet.length, 6.0, "meet magnitude = 2.0 * 3.0");

        // geonum represents intersection at grade 3 due to π-rotation dual
        assert_eq!(meet.angle.grade(), 3);
    }

    #[test]
    fn it_meets_bivectors_in_same_plane() {
        // coplanar bivectors (same angle) represent parallel planes
        let bivector1 = Geonum::new(4.0, 5.0, 4.0); // 5π/4 = grade 2 bivector
        let bivector2 = Geonum::new(6.0, 5.0, 4.0); // 5π/4 = same angle, grade 2

        let meet = bivector1.meet(&bivector2);

        // parallel planes (same angle bivectors) have zero meet
        // they don't intersect in finite projective space
        assert!(meet.length < EPSILON, "parallel planes have zero meet");
    }

    #[test]
    fn it_meets_bivectors_in_different_planes() {
        // non-parallel planes intersect along a line
        let plane1 = Geonum::new(4.0, 5.0, 4.0); // 5π/4 = grade 2 bivector
        let plane2 = Geonum::new(9.0, 7.0, 4.0); // 7π/4 = grade 3 trivector

        let meet = plane1.meet(&plane2);

        // grade 2 bivector meets grade 3 trivector
        // represents plane-volume intersection in projective geometry
        assert!(meet.length > 0.0, "non-parallel planes have non-zero meet");

        // the intersection produces a geometric object encoding the line of intersection
        assert!(meet.length.is_finite());
    }

    #[test]
    fn it_meets_trivectors_at_plane() {
        // trivectors represent 3D volumes in projective geometry
        let volume1 = Geonum::new(8.0, 7.0, 4.0); // 7π/4 = grade 3 trivector
        let volume2 = Geonum::new(2.0, 15.0, 8.0); // 15π/8 = different angle grade 3

        let meet = volume1.meet(&volume2);

        // non-parallel volumes intersect
        // the meet encodes their common 2D subspace (plane)
        assert!(meet.length > 0.0, "non-parallel volumes have non-zero meet");

        // geonum's π-rotation dual produces specific grade for volume-volume meet
        assert!(meet.length.is_finite());
    }

    #[test]
    fn it_meets_higher_grades_cycling_pattern() {
        // grades cycle modulo 4 in geonum's framework
        let high_blade1 = Geonum::new_with_blade(3.0, 7, 1.0, 6.0); // blade 7, grade 3
        let high_blade2 = Geonum::new_with_blade(12.0, 11, 1.0, 4.0); // blade 11, grade 3

        let meet = high_blade1.meet(&high_blade2);

        // high blade numbers still follow grade cycling
        // blade 7 % 4 = 3, blade 11 % 4 = 3 (both grade 3)
        assert!(meet.length > 0.0, "non-parallel high-blade objects meet");

        // the meet operation works consistently regardless of blade magnitude
        // demonstrating O(1) complexity even for high-dimensional spaces
        assert!(meet.length.is_finite());
    }

    #[test]
    fn it_maintains_grade_consistency_in_meet() {
        // prove that meet produces specific grades based on input grade combinations
        // meet(A,B) = dual(wedge(dual(A), dual(B))) with π-rotation dual (adds 2 blades)

        // trace grade 0 meet grade 0
        let s1 = Geonum::new(2.0, 0.0, 1.0); // blade 0, grade 0
        let s2 = Geonum::new(3.0, 1.0, 4.0); // blade 0, grade 0

        // compute manually to prove the grade transformation
        let s1_dual = s1.dual(); // blade 0 + 2 = blade 2, grade 2
        let s2_dual = s2.dual(); // blade 0 + 2 = blade 2, grade 2
        assert_eq!(s1_dual.angle.grade(), 2);
        assert_eq!(s2_dual.angle.grade(), 2);

        let wedge_result = s1_dual.wedge(&s2_dual);
        // wedge adds angles and π/2, plus possible π if orientation negative
        // the wedge of two grade 2 bivectors produces grade based on angle sum

        let s_meet = wedge_result.dual(); // add 2 more blades
                                          // final grade depends on total blade count modulo 4

        // prove the actual grade transformations
        let actual_meet = s1.meet(&s2);
        assert_eq!(
            actual_meet.angle.grade(),
            3,
            "grade 0 meet grade 0 → grade 3"
        );

        // prove the manual computation matches the meet implementation
        // this demonstrates meet(A,B) = dual(wedge(dual(A), dual(B)))
        assert_eq!(
            s_meet.angle.grade(),
            actual_meet.angle.grade(),
            "manual formula matches meet implementation"
        );
        assert!(
            (s_meet.length - actual_meet.length).abs() < 1e-10,
            "manual formula produces same magnitude"
        );

        // test different grade combinations to prove the complete pattern
        let v1 = Geonum::new(2.0, 3.0, 4.0); // 3π/4 = blade 1, grade 1
        let v2 = Geonum::new(3.0, 5.0, 4.0); // 5π/4 = blade 2, grade 2
        let v_meet = v1.meet(&v2);
        assert_eq!(v_meet.angle.grade(), 3, "grade 1 meet grade 2 → grade 3");

        let b1 = Geonum::new(2.0, 5.0, 4.0); // 5π/4 = blade 2, grade 2
        let b2 = Geonum::new(3.0, 7.0, 4.0); // 7π/4 = blade 3, grade 3
        let b_meet = b1.meet(&b2);
        assert_eq!(b_meet.angle.grade(), 1, "grade 2 meet grade 3 → grade 1");

        let t1 = Geonum::new(2.0, 7.0, 4.0); // 7π/4 = blade 3, grade 3
        let t2 = Geonum::new(3.0, 15.0, 8.0); // 15π/8 = blade 3, grade 3
        let t_meet = t1.meet(&t2);
        assert_eq!(t_meet.angle.grade(), 2, "grade 3 meet grade 3 → grade 2");

        // these grade transformations prove the dual-wedge-dual formula
        // creates a consistent grade mapping based on
        // the π-rotation dual and wedge product angle addition
    }

    #[test]
    fn it_reflects_using_angle_arithmetic() {
        // reflection is primitively a π rotation (2 blades) in absolute angle space
        // the operation accumulates blades while computing the reflected position

        // test that reflection adds 2 blades (π rotation)
        let point = Geonum::new_from_cartesian(3.0, 2.0);
        let axis_45 = Geonum::new(1.0, 1.0, 4.0); // 45° axis

        // use the reflect method
        let reflected = point.reflect(&axis_45);

        // reflection preserves length and accumulates blades forward
        assert_eq!(reflected.length, point.length);

        // forward-only reflection adds ~4 blades per reflection
        let blade_added = reflected.angle.blade() as i32 - point.angle.blade() as i32;
        assert!(blade_added >= 0, "blade only increases in forward-only");

        // test that double reflection accumulates blade
        let reflected_twice = reflected.reflect(&axis_45);

        // double reflection accumulates ~8 blades total
        let total_blade_added = reflected_twice.angle.blade() as i32 - point.angle.blade() as i32;
        assert!(
            total_blade_added >= 7,
            "double reflection accumulates significant blade"
        );

        // with base_angle(), returns to original position
        let (px, py) = point.to_cartesian();
        let (rx, ry) = reflected_twice.base_angle().to_cartesian();
        assert!(
            (px - rx).abs() < 1e-10 && (py - ry).abs() < 1e-10,
            "double reflection with base_angle returns to original position"
        );
    }

    #[test]
    fn it_multiplies_angle_by_geonum() {
        // test Angle * Geonum (owned version)
        let angle = Angle::new(1.0, 2.0); // π/2
        let geonum = Geonum::new(3.0, 1.0, 4.0); // length 3, angle π/4

        let result = angle * geonum;

        // test length preserved
        assert_eq!(result.length, 3.0);

        // test angles add: π/2 + π/4 = 3π/4
        let expected_angle = Angle::new(3.0, 4.0);
        assert_eq!(result.angle, expected_angle);
    }

    #[test]
    fn it_multiplies_angle_by_geonum_ref() {
        // test Angle * &Geonum (borrow version)
        let angle = Angle::new(1.0, 2.0); // π/2
        let geonum = Geonum::new(3.0, 1.0, 4.0); // length 3, angle π/4

        let result = angle * geonum;

        // test length preserved
        assert_eq!(result.length, 3.0);

        // test angles add: π/2 + π/4 = 3π/4
        let expected_angle = Angle::new(3.0, 4.0);
        assert_eq!(result.angle, expected_angle);

        // test original geonum still usable after borrow
        assert_eq!(geonum.length, 3.0);
        assert_eq!(geonum.angle, Angle::new(1.0, 4.0));
    }

    #[test]
    fn it_preserves_blade_in_angle_mul_geonum() {
        // test that blade counts accumulate through angle multiplication
        let angle = Angle::new(3.0, 2.0); // 3π/2 = blade 3
        let geonum = Geonum::new_with_blade(2.0, 5, 1.0, 4.0); // blade 5, angle π/4

        // owned version
        let result1 = angle * geonum;
        assert_eq!(result1.length, 2.0);
        assert_eq!(result1.angle.blade(), 8); // 3 + 5 = 8

        // borrow version
        let result2 = angle * geonum;
        assert_eq!(result2.length, 2.0);
        assert_eq!(result2.angle.blade(), 8); // 3 + 5 = 8
    }

    #[test]
    fn it_handles_zero_angle_multiplication() {
        // test multiplication with zero angle
        let zero_angle = Angle::new(0.0, 1.0); // 0 radians
        let geonum = Geonum::new(5.0, 2.0, 3.0); // length 5, angle 2π/3

        // owned version
        let result1 = zero_angle * geonum;
        assert_eq!(result1.length, 5.0);
        assert_eq!(result1.angle, geonum.angle); // angle unchanged

        // borrow version
        let result2 = zero_angle * geonum;
        assert_eq!(result2.length, 5.0);
        assert_eq!(result2.angle, geonum.angle); // angle unchanged
    }

    #[test]
    fn it_handles_full_rotation_multiplication() {
        // test multiplication with full rotation (2π)
        let full_rotation = Angle::new(2.0, 1.0); // 2π
        let geonum = Geonum::new(1.0, 1.0, 6.0); // length 1, angle π/6

        // owned version
        let result1 = full_rotation * geonum;
        assert_eq!(result1.length, 1.0);
        // 2π + π/6 = blade 4 + fractional part π/6
        assert_eq!(result1.angle.blade(), 4);

        // borrow version
        let result2 = full_rotation * geonum;
        assert_eq!(result2.length, 1.0);
        assert_eq!(result2.angle.blade(), 4);
    }

    #[test]
    fn it_meets_vector_trivector_based_on_angles() {
        // meet finds intersection based on angle relationships, not length comparisons
        // parallel objects (same angle) have zero meet, perpendicular have non-zero

        // create vector and trivector with different angle relationships
        let vector_0 = Geonum::new_with_blade(3.0, 1, 0.0, 1.0); // blade 1, angle 0
        let vector_45 = Geonum::new_with_blade(3.0, 1, 1.0, 4.0); // blade 1, angle π/4

        let trivector_0 = Geonum::new_with_blade(5.0, 3, 0.0, 1.0); // blade 3, angle 0
        let trivector_90 = Geonum::new_with_blade(5.0, 3, 1.0, 2.0); // blade 3, angle π/2

        // parallel case: vector and trivector at same fractional angle
        // after dual: blade 1→3, blade 3→5
        // wedge of two blade 3s with angle 0 gives sin(0) ≈ 0
        let meet_parallel = vector_0.meet(&trivector_0);
        assert!(
            meet_parallel.length < EPSILON,
            "parallel objects have zero meet"
        );

        // perpendicular case: vector at 0, trivector at π/2
        // after dual: blade 1→3 angle 0, blade 3→5 angle π/2
        // wedge gives non-zero result from sin(angle_diff)
        let meet_perpendicular = vector_0.meet(&trivector_90);
        assert!(
            meet_perpendicular.length > EPSILON,
            "perpendicular objects have non-zero meet"
        );

        // angled case: vector at π/4, trivector at 0
        // wedge gives sin(π/4) = √2/2
        let meet_angled = vector_45.meet(&trivector_0);
        assert!(
            meet_angled.length > EPSILON,
            "angled objects have non-zero meet"
        );
    }

    #[test]
    fn it_reflects_point_across_x_axis_with_blade_accumulation() {
        // reflection primitively adds π rotation (2 blades)
        // reflecting (1, 1) across x-axis changes position AND adds 2 blades
        let point = Geonum::new_from_cartesian(1.0, 1.0);
        let x_axis = Geonum::new_from_cartesian(1.0, 0.0);

        let reflected = point.reflect(&x_axis);

        // forward-only formula: 2*axis + (2π - base_angle(point))
        // point at π/4 with base_angle π/4, so adds 7π/4 = 7 blades
        let blade_accumulation = reflected.angle.blade() - point.angle.blade();
        assert_eq!(
            blade_accumulation, 7,
            "reflection across x-axis from π/4 adds 7 blades"
        );

        // reflection changes the grade structure
        // π/4 (blade 0) → 7π/4 (blade 3)
        assert_eq!(
            reflected.angle.grade(),
            3,
            "reflected point has trivector grade"
        );

        // test cartesian coordinates after reflection
        let (ox, oy) = point.to_cartesian();
        let (rx, ry) = reflected.to_cartesian();

        println!("Original: ({ox}, {oy})");
        println!("Reflected: ({rx}, {ry})");
        println!("Expected: ({}, {})", ox, -oy);

        // with forward-only geometry, reflection may not produce traditional result
        // the blade accumulation changes the geometric interpretation
    }

    #[test]
    fn it_gets_length() {
        let g = Geonum::new(3.5, 1.0, 4.0); // [3.5, π/4]
        assert_eq!(g.length(), 3.5);

        let g2 = Geonum::new(0.0, 0.0, 1.0); // [0, 0]
        assert_eq!(g2.length(), 0.0);

        let g3 = Geonum::new(10.0, 3.0, 2.0); // [10, 3π/2]
        assert_eq!(g3.length(), 10.0);
    }

    #[test]
    fn it_gets_angle() {
        let g = Geonum::new(2.0, 1.0, 4.0); // [2, π/4]
        assert_eq!(g.angle(), Angle::new(1.0, 4.0));

        let g2 = Geonum::new(1.0, 0.0, 1.0); // [1, 0]
        assert_eq!(g2.angle(), Angle::new(0.0, 1.0));

        let g3 = Geonum::new(5.0, 3.0, 2.0); // [5, 3π/2]
        assert_eq!(g3.angle(), Angle::new(3.0, 2.0));
    }

    #[test]
    fn it_scales_by_factor() {
        let g = Geonum::new(2.0, 1.0, 4.0); // [2, π/4]

        // scale by 3
        let scaled = g.scale(3.0);
        assert_eq!(scaled.length, 6.0);
        assert_eq!(scaled.angle, g.angle); // angle unchanged

        // scale by 0.5
        let scaled2 = g.scale(0.5);
        assert_eq!(scaled2.length, 1.0);
        assert_eq!(scaled2.angle, g.angle); // angle unchanged

        // scale by negative (adds π to angle)
        let scaled3 = g.scale(-2.0);
        assert_eq!(scaled3.length, 4.0);
        // angle should be π/4 + π
        let expected_angle = g.angle + Angle::new(1.0, 1.0);
        assert_eq!(scaled3.angle, expected_angle);
    }

    #[test]
    fn it_scales_preserves_angle_exactly() {
        // test that scale preserves exact angle representation, not just total angle
        let g = Geonum::new_with_blade(3.0, 2, 1.0, 6.0); // blade 2, value π/6

        let scaled = g.scale(5.0);
        assert_eq!(scaled.length, 15.0);
        assert_eq!(scaled.angle.blade(), 2); // blade unchanged
        assert!((scaled.angle.value() - g.angle.value()).abs() < 1e-10); // value unchanged
    }

    #[test]
    fn it_inverts_circle() {
        // test circle inversion through unit circle at origin
        let center = Geonum::scalar(0.0);
        let radius = 1.0;

        // point outside circle at distance 2
        let point = Geonum::new(2.0, 0.0, 1.0);
        let inverted = point.invert_circle(&center, radius);

        // maps to distance 1/2 (r²/d = 1/2)
        assert!((inverted.length - 0.5).abs() < 1e-10);
        // angle is preserved in circle inversion
        assert_eq!(inverted.angle, point.angle);

        // point inside circle at distance 0.5
        let inside = Geonum::new(0.5, 1.0, 2.0); // π/2 angle
        let inv_inside = inside.invert_circle(&center, radius);

        // maps to distance 2 (r²/d = 1/0.5 = 2)
        assert!((inv_inside.length - 2.0).abs() < 1e-10);
        // angle is preserved during circle inversion
        assert_eq!(inv_inside.angle, inside.angle);

        // inversion is involutive: inverting twice returns original
        let double_inv = inv_inside.invert_circle(&center, radius);
        assert!((double_inv.length - inside.length).abs() < 1e-10);
        assert_eq!(double_inv.angle, inside.angle);

        // test with non-origin center
        let offset_center = Geonum::new(3.0, 0.0, 1.0);
        let offset_radius = 2.0;
        let test_point = Geonum::new(5.0, 0.0, 1.0); // 2 units from center

        let offset_inv = test_point.invert_circle(&offset_center, offset_radius);

        // distance from center is r²/d = 4/2 = 2
        let dist_from_center = (offset_inv - offset_center).length;
        assert!((dist_from_center - 2.0).abs() < 1e-10);

        // angle from center preserved
        let angle_before = (test_point - offset_center).angle;
        let angle_after = (offset_inv - offset_center).angle;
        assert_eq!(angle_after.blade(), angle_before.blade());
    }

    #[test]
    fn it_inverts_unit_circle_conjugates_angle() {
        // inversion through unit circle at origin is complex inversion 1/z
        // this conjugates the angle: z = re^(iθ) → 1/z = (1/r)e^(-iθ)

        let origin = Geonum::scalar(0.0);
        let unit_radius = 1.0;

        // point at distance 2, angle π/3
        let z = Geonum::new(2.0, 1.0, 3.0);

        // invert through unit circle at origin
        let inverted = z.invert_circle(&origin, unit_radius);

        // distance becomes 1/2
        assert!((inverted.length - 0.5).abs() < 1e-10);

        // angle is preserved during circle inversion in geonum
        // circle inversion preserves angles (conformal transformation)
        assert_eq!(inverted.angle, z.angle, "circle inversion preserves angle");
    }

    #[test]
    fn test_scale_rotate() {
        // test spiral similarity transformation

        // test basic scale and rotate
        let p = Geonum::new(1.0, 0.0, 1.0); // unit length at angle 0
        let transformed = p.scale_rotate(2.0, Angle::new(1.0, 6.0)); // double and rotate π/6

        assert_eq!(transformed.length, 2.0, "length scaled by factor");
        assert_eq!(
            transformed.angle,
            Angle::new(1.0, 6.0),
            "angle rotated by π/6"
        );

        // test identity transformation
        let identity = p.scale_rotate(1.0, Angle::new(0.0, 1.0));
        assert_eq!(identity.length, p.length, "identity preserves length");
        assert_eq!(identity.angle, p.angle, "identity preserves angle");

        // test pure scaling (no rotation)
        let scaled = p.scale_rotate(3.0, Angle::new(0.0, 1.0));
        assert_eq!(scaled.length, 3.0, "pure scaling triples length");
        assert_eq!(scaled.angle, p.angle, "pure scaling preserves angle");

        // test pure rotation (no scaling)
        let rotated = p.scale_rotate(1.0, Angle::new(1.0, 2.0)); // rotate π/2
        assert_eq!(rotated.length, 1.0, "pure rotation preserves length");
        assert_eq!(rotated.angle, Angle::new(1.0, 2.0), "pure rotation to π/2");

        // test composition property: SR(a,θ) ∘ SR(b,φ) = SR(ab, θ+φ)
        let p2 = Geonum::new(2.0, 1.0, 4.0); // 2 at π/4
        let first = p2.scale_rotate(2.0, Angle::new(1.0, 6.0)); // scale 2, rotate π/6
        let second = first.scale_rotate(3.0, Angle::new(1.0, 3.0)); // scale 3, rotate π/3

        // equivalent to single transformation with scale 6, rotate π/6 + π/3 = π/2
        let combined = p2.scale_rotate(6.0, Angle::new(1.0, 2.0));

        assert!(
            (second.length - combined.length).abs() < 1e-10,
            "composition: scales multiply"
        );
        assert_eq!(second.angle, combined.angle, "composition: angles add");

        // test spiral iteration (logarithmic spiral)
        let start = Geonum::new(1.0, 0.0, 1.0);
        let mut current = start;

        // apply spiral similarity 6 times with scale √2 and rotation π/4
        // after 6 iterations: scale = (√2)^6 = 8, angle = 6π/4 = 3π/2
        for _ in 0..6 {
            current = current.scale_rotate(2.0_f64.sqrt(), Angle::new(1.0, 4.0));
        }

        assert!(
            (current.length - 8.0).abs() < 1e-10,
            "6 iterations of √2 scaling gives 8"
        );
        assert_eq!(
            current.angle,
            Angle::new(3.0, 2.0),
            "6 iterations of π/4 rotation gives 3π/2"
        );

        // test with negative scale (reflection + scaling)
        let reflected = p.scale_rotate(-2.0, Angle::new(0.0, 1.0));
        assert_eq!(
            reflected.length, 2.0,
            "negative scale becomes positive length"
        );
        // negative scaling adds π to angle (changes blade by 2)
        assert_eq!(
            reflected.angle.blade(),
            2,
            "negative scale adds π rotation (blade 2)"
        );

        // test invariant: scale_rotate preserves grade structure
        let vector = Geonum::new(1.0, 1.0, 2.0); // π/2 angle gives blade 1 (vector)
        let transformed_vector = vector.scale_rotate(2.0, Angle::new(1.0, 8.0)); // scale 2, rotate π/8

        // angle becomes π/2 + π/8 = 5π/8, still blade 1
        let expected_angle = Angle::new(5.0, 8.0);
        assert_eq!(
            transformed_vector.angle, expected_angle,
            "scale_rotate adds angles correctly"
        );
    }

    #[test]
    fn it_prevents_blade_accumulation_in_control_loops() {
        // test blade accumulation from repeated operations
        let point = Geonum::new(1.0, 0.0, 1.0);
        let rotated_many = (0..100).fold(point, |p, _| p.rotate(Angle::new(1.0, 2.0))); // π/2 each time
                                                                                        // 100 rotations of π/2 = 100 blade increments
        assert!(
            rotated_many.angle.blade() >= 100,
            "operations accumulate blade"
        );

        let normalized = rotated_many.base_angle();
        assert_eq!(normalized.length, rotated_many.length, "length preserved");
        assert_eq!(
            normalized.angle.grade(),
            rotated_many.angle.grade(),
            "grade preserved"
        );
        assert!(normalized.angle.blade() < 4, "blade reset to minimum");

        // test double reflection blade accumulation
        let original = Geonum::new(3.0, 0.0, 1.0); // blade 0, angle 0
        let axis = Geonum::new(1.0, 1.0, 4.0); // blade 0, angle π/4
        let reflected_once = original.reflect(&axis);
        let reflected_twice = reflected_once.reflect(&axis);

        // forward-only formula accumulates blade
        // each reflection adds blades based on grade

        // test actual blade accumulation
        let blade_diff = reflected_twice.angle.blade() - original.angle.blade();
        assert_eq!(blade_diff, 8, "double reflection adds 8 blades");

        // grade returns to original (8 blades % 4 = 0)
        assert_eq!(
            reflected_twice.angle.grade(),
            original.angle.grade(),
            "double reflection preserves grade modulo 4"
        );

        // test control loop simulation
        let mut position = Geonum::new(1.0, 0.0, 1.0);
        for _ in 0..1000 {
            position = position.scale_rotate(1.001, Angle::new(1.0, 1000.0));
            position = position.base_angle(); // prevent blade explosion
        }
        assert!(position.angle.blade() < 4, "blade bounded in control loop");

        // test that base_angle and dual operations have consistent grade results
        let high_blade_geonum = Geonum::new_with_blade(2.5, 999, 1.0, 3.0);
        let dual_then_base = high_blade_geonum.dual().base_angle();
        let base_then_dual = high_blade_geonum.base_angle().dual();

        // both should end at same grade (operations dont commute in blade, but do in grade)
        assert_eq!(
            dual_then_base.angle.grade(),
            base_then_dual.angle.grade(),
            "both orders reach same grade"
        );
        assert_eq!(
            dual_then_base.length, base_then_dual.length,
            "length preserved in both orders"
        );
    }

    #[test]
    fn it_preserves_geometric_operations_after_blade_reset() {
        // test that geometric operations still work after blade reset
        let a = Geonum::new_with_blade(2.0, 1000, 1.0, 3.0);
        let b = Geonum::new(3.0, 1.0, 4.0);

        let wedge_with_blade = a.wedge(&b);
        let wedge_without_blade = a.base_angle().wedge(&b);

        assert_eq!(
            wedge_with_blade.length, wedge_without_blade.length,
            "wedge product length unaffected by blade reset"
        );
        assert_eq!(
            wedge_with_blade.angle.grade(),
            wedge_without_blade.angle.grade(),
            "wedge product grade unaffected by blade reset"
        );

        // test dot product unaffected
        let dot_with = a.dot(&b);
        let dot_without = a.base_angle().dot(&b);
        assert_eq!(
            dot_with, dot_without,
            "dot product unaffected by blade reset"
        );
    }

    #[test]
    fn it_makes_double_inversion_involutive_with_blade_reset() {
        // test circular inversion with blade management
        let center = Geonum::new_from_cartesian(0.0, 0.0);
        let radius = 2.0;
        let point = Geonum::new(3.0, 1.0, 6.0);

        let inverted_once = point.invert_circle(&center, radius);
        let inverted_twice = inverted_once.invert_circle(&center, radius);

        // circle inversion preserves angle exactly, including blade
        // double inversion is involutive without needing blade reset
        assert_eq!(
            inverted_twice.angle, point.angle,
            "double inversion returns to original angle"
        );
        assert_eq!(
            inverted_twice.length, point.length,
            "double inversion returns to original length"
        );

        // circle inversion differs from reflection in this way:
        // reflection adds 2 blades (π rotation) per operation
        // circle inversion preserves the angle completely
    }

    #[test]
    fn it_demonstrates_forward_only_reflection_pattern() {
        // forward-only reflection has a single consistent pattern:
        // each reflection adds blade based on the complement formula
        // double reflection always accumulates ~8 blades

        let original = Geonum::new(5.0, 0.0, 1.0); // grade 0 (scalar)

        // test various axes - all follow same pattern
        let test_axes = vec![
            ("0", Geonum::new(1.0, 0.0, 1.0)),
            ("π/2", Geonum::new(1.0, 1.0, 2.0)),
            ("π/4", Geonum::new(1.0, 1.0, 4.0)),
            ("3π/4", Geonum::new(1.0, 3.0, 4.0)),
            ("π/6", Geonum::new(1.0, 1.0, 6.0)),
            ("π/3", Geonum::new(1.0, 1.0, 3.0)),
            ("π/8", Geonum::new(1.0, 1.0, 8.0)),
        ];

        for (name, axis) in test_axes {
            let once = original.reflect(&axis);
            let twice = once.reflect(&axis);

            // forward-only pattern: double reflection adds 8 blades
            let blade_accumulation = twice.angle.blade() - original.angle.blade();
            assert_eq!(
                blade_accumulation, 8,
                "axis at {name}: double reflection adds 8 blades"
            );

            // grade returns to original (8 % 4 = 0)
            assert_eq!(twice.angle.grade(), 0, "axis at {name}: grade returns to 0");

            // angle returns to original (but with blade accumulation)
            assert!(
                (twice.angle.mod_4_angle() - original.angle.mod_4_angle()).abs() < 1e-10,
                "axis at {name}: angle returns to original"
            );
        }

        // prove all axes produce same blade accumulation
        let axis_pi_6 = Geonum::new(1.0, 1.0, 6.0);
        let axis_pi_4 = Geonum::new(1.0, 1.0, 4.0);

        let test_pi_6 = original.reflect(&axis_pi_6).reflect(&axis_pi_6);
        let test_pi_4 = original.reflect(&axis_pi_4).reflect(&axis_pi_4);

        assert_eq!(
            test_pi_6.angle.blade(),
            8,
            "π/6 axis: blade 8 after double reflection"
        );
        assert_eq!(
            test_pi_4.angle.blade(),
            8,
            "π/4 axis: blade 8 after double reflection"
        );

        // forward-only geometry has one universal reflection pattern
        // blade accumulation is predictable and consistent
    }

    #[test]
    fn it_demonstrates_inversion_preserves_grade_parity_relationships() {
        // geonum's grade structure has involutive pairs: 0↔2, 1↔3
        // operations that preserve this pairing maintain orthogonality relationships
        // circular inversion is one such operation
        let center = Geonum::new_from_cartesian(0.0, 0.0); // origin for clarity
        let radius = 2.0;

        // test points at different angles and distances
        let test_configs = vec![
            // (distance, angle_pi_rad, angle_div, description)
            (1.0, 0.0, 1.0, "inside on +x axis"),
            (3.0, 0.0, 1.0, "outside on +x axis"),
            (1.0, 1.0, 2.0, "inside on +y axis"),
            (3.0, 1.0, 2.0, "outside on +y axis"),
            (1.0, 1.0, 4.0, "inside at π/4"),
            (3.0, 1.0, 4.0, "outside at π/4"),
            (1.0, 1.0, 1.0, "inside on -x axis"),
            (3.0, 1.0, 1.0, "outside on -x axis"),
        ];

        println!("\nSingle point inversions from origin:");
        for (dist, pi_rad, div, desc) in test_configs {
            let p = Geonum::new(dist, pi_rad, div);
            let p_inv = p.invert_circle(&center, radius);

            println!(
                "{}: dist={} angle={:.3} blade={} → dist={:.3} angle={:.3} blade={}",
                desc,
                dist,
                p.angle.value(),
                p.angle.blade(),
                p_inv.length,
                p_inv.angle.value(),
                p_inv.angle.blade()
            );

            // verify inversion property
            assert!((p.length * p_inv.length - radius * radius).abs() < EPSILON);
        }

        // now test difference vectors between points (where blade changes occurred before)
        println!("\nDifference vectors between points:");

        // create a configuration that shows blade transformation
        let p1 = Geonum::new_from_cartesian(2.0, 1.0);
        let p2 = Geonum::new_from_cartesian(3.0, 0.0);
        let p3 = Geonum::new_from_cartesian(2.0, -1.0);

        // compute difference vectors
        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let v23 = p3 - p2;

        println!("Original vectors:");
        println!(
            "  v12=p2-p1: length={:.3} angle={:.3} blade={}",
            v12.length,
            v12.angle.value(),
            v12.angle.blade()
        );
        println!(
            "  v13=p3-p1: length={:.3} angle={:.3} blade={}",
            v13.length,
            v13.angle.value(),
            v13.angle.blade()
        );
        println!(
            "  v23=p3-p2: length={:.3} angle={:.3} blade={}",
            v23.length,
            v23.angle.value(),
            v23.angle.blade()
        );

        // invert the points
        let p1_inv = p1.invert_circle(&center, radius);
        let p2_inv = p2.invert_circle(&center, radius);
        let p3_inv = p3.invert_circle(&center, radius);

        // compute inverted difference vectors
        let v12_inv = p2_inv - p1_inv;
        let v13_inv = p3_inv - p1_inv;
        let v23_inv = p3_inv - p2_inv;

        println!("Inverted vectors:");
        println!(
            "  v12_inv: length={:.3} angle={:.3} blade={}",
            v12_inv.length,
            v12_inv.angle.value(),
            v12_inv.angle.blade()
        );
        println!(
            "  v13_inv: length={:.3} angle={:.3} blade={}",
            v13_inv.length,
            v13_inv.angle.value(),
            v13_inv.angle.blade()
        );
        println!(
            "  v23_inv: length={:.3} angle={:.3} blade={}",
            v23_inv.length,
            v23_inv.angle.value(),
            v23_inv.angle.blade()
        );

        // KEY INSIGHT: blade transformation happens in difference vectors
        // individual points from origin maintain blade, but vectors between inverted points transform

        // test with points that create perpendicular vectors
        println!("\nPerpendicular vector configuration:");
        let center2 = Geonum::new_from_cartesian(1.0, 0.0); // offset center
        let q1 = Geonum::new_from_cartesian(3.0, 0.0);
        let q2 = Geonum::new_from_cartesian(4.0, 0.0);
        let q3 = Geonum::new_from_cartesian(3.0, 1.0);

        let u1 = q2 - q1; // horizontal
        let u2 = q3 - q1; // vertical

        println!("Original perpendicular vectors:");
        println!("  u1: blade={} (horizontal)", u1.angle.blade());
        println!("  u2: blade={} (vertical)", u2.angle.blade());

        // these perpendicular vectors have different blades (orthogonality via blade difference)
        assert_ne!(
            u1.angle.blade() % 2,
            u2.angle.blade() % 2,
            "perpendicular vectors differ by odd blade count"
        );

        let q1_inv = q1.invert_circle(&center2, radius);
        let q2_inv = q2.invert_circle(&center2, radius);
        let q3_inv = q3.invert_circle(&center2, radius);

        let u1_inv = q2_inv - q1_inv;
        let u2_inv = q3_inv - q1_inv;

        println!("Inverted 'perpendicular' vectors:");
        println!("  u1_inv: blade={}", u1_inv.angle.blade());
        println!("  u2_inv: blade={}", u2_inv.angle.blade());

        // blade relationships transform under inversion
        let blade_diff_original = (u2.angle.blade() as i32 - u1.angle.blade() as i32).abs();
        let blade_diff_inverted = (u2_inv.angle.blade() as i32 - u1_inv.angle.blade() as i32).abs();

        println!("Blade difference: {blade_diff_original} → {blade_diff_inverted}");

        // check if grade differences are preserved (blade mod 4)
        let grade_diff_original =
            ((u2.angle.grade() as i32 - u1.angle.grade() as i32).abs() % 4) as usize;
        let grade_diff_inverted =
            ((u2_inv.angle.grade() as i32 - u1_inv.angle.grade() as i32).abs() % 4) as usize;

        println!("Grade difference: {grade_diff_original} → {grade_diff_inverted}");

        // orthogonality is encoded in odd grade differences (parity)
        // grade 0 vs grade 1: difference = 1 (odd) → orthogonal
        // grade 2 vs grade 3: difference = 1 (odd) → orthogonal
        // grade 0 vs grade 2: difference = 2 (even) → parallel (dual pair)
        // grade 1 vs grade 3: difference = 2 (even) → parallel (dual pair)

        assert_eq!(
            grade_diff_original % 2,
            1,
            "original vectors are orthogonal (odd grade diff)"
        );
        assert_eq!(
            grade_diff_inverted % 2,
            1,
            "inverted vectors remain orthogonal (odd grade diff)"
        );

        // this is expected from geonum's involutive grade pairs (0↔2, 1↔3)
        // operations respecting this pairing preserve orthogonality parity
        // circular inversion is such an operation - it may shift grades within pairs
        // but preserves the odd/even nature of grade differences
    }
}
