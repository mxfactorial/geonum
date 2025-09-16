use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

/// Angle struct that maintains the angle-blade invariant
///
/// encapsulates the fundamental geometric number constraint:
/// - angle value stays within [0, π/2)
/// - blade counts π/2 rotations
#[derive(Debug, Clone, Copy)]
pub struct Angle {
    /// angle within current π/2 segment, always in range [0, π/2)
    value: f64,
    /// rotation count (determines geometric grade)
    /// our substrate doesnt enable lights path so
    /// we keep count of π/2 turns with this
    /// digital prosthetic until its automated:
    /// https://github.com/mxfactorial/holographic-cloud
    blade: usize,
}

impl Angle {
    /// creates a new angle from pi radians and divisor
    /// automatically decomposes total angle into blade count and remainder
    ///
    /// # arguments
    /// * `pi_radians` - number of π radians
    /// * `divisor` - denominator of π (2 means π/2, 4 means π/4, etc)
    ///
    /// # returns
    /// angle struct with blade count and remainder value
    ///
    /// # examples
    /// ```
    /// use geonum::Angle;
    /// use std::f64::consts::PI;
    ///
    /// let angle = Angle::new(3.0, 4.0);  // 3 * π/4 = 135 degrees
    /// assert_eq!(angle.blade(), 1);      // one π/2 rotation
    /// assert_eq!(angle.value(), PI / 4.0); // π/4 remainder
    /// ```
    pub fn new(pi_radians: f64, divisor: f64) -> Self {
        let quarter_pi = PI / 2.0;

        // exact quarter-turns: use clean pi_radians count directly
        if divisor == 2.0 && pi_radians.fract() == 0.0 {
            // handle negative values by normalizing first
            let normalized_quarters = if pi_radians < 0.0 {
                // add enough full rotations to make positive
                let full_rotations = ((-pi_radians + 3.0) / 4.0).ceil() * 4.0;
                (pi_radians + full_rotations) as usize
            } else {
                pi_radians as usize
            };
            return Self {
                value: 0.0,
                blade: normalized_quarters,
            };
        }

        // general case: clone pi_radians for floating point buggery
        let pi_radians_copy = pi_radians;
        let total_angle = pi_radians_copy * PI / divisor;

        // handle negative angles by adding full rotations
        let normalized_total = if total_angle < 0.0 {
            let full_rotations = (total_angle.abs() / (4.0 * quarter_pi)).ceil();
            total_angle + full_rotations * 4.0 * quarter_pi
        } else {
            total_angle
        };

        // count complete π/2 rotations (preserve full count)
        let blade = (normalized_total / quarter_pi) as usize;

        // remainder within current π/2 segment
        let value = normalized_total % quarter_pi;

        let angle = Self { value, blade };
        angle.normalize_boundaries()
    }

    /// creates a new angle with additional blade count
    /// combines normal angle processing with extra blade increments
    ///
    /// # arguments
    /// * `added_blade` - additional blade count to add
    /// * `pi_radians` - number of π radians
    /// * `divisor` - denominator of π (2 means π/2, 4 means π/4, etc)
    ///
    /// # returns
    /// angle struct with processed angle plus additional blade count
    pub fn new_with_blade(added_blade: usize, pi_radians: f64, divisor: f64) -> Self {
        let base_angle = Angle::new(pi_radians, divisor);
        let blade_increment = Angle::new(added_blade as f64, 2.0); // added_blade * π/2
        base_angle + blade_increment
    }

    /// creates a new angle from cartesian coordinates
    /// converts (x, y) to geometric angle representation
    ///
    /// # arguments
    /// * `x` - x coordinate
    /// * `y` - y coordinate
    ///
    /// # returns
    /// angle struct representing the direction from origin to (x, y)
    pub fn new_from_cartesian(x: f64, y: f64) -> Self {
        let angle_radians = y.atan2(x);
        // convert radians to pi_radians for Angle::new
        // which handles all normalization and decomposition
        let pi_radians = angle_radians / PI;
        // Angle::new expects (pi_radians * divisor, divisor)
        // so for direct pi_radians, use divisor = 1
        Self::new(pi_radians, 1.0)
    }

    /// rotates this angle by a given amount
    /// automatically handles π/2 boundary crossings and blade updates
    ///
    /// # arguments  
    /// * `delta` - angle to rotate by
    ///
    /// # returns
    /// new angle with rotation applied
    pub fn rotate(self, delta: Angle) -> Self {
        self + delta
    }

    /// returns the angle value within current π/2 segment
    ///
    /// # returns
    /// angle value in range [0, π/2)
    pub fn value(&self) -> f64 {
        self.value
    }

    /// returns the blade count (rotation count)
    ///
    /// # returns
    /// number of π/2 rotations (0-3 in 4D rotation space)
    pub fn blade(&self) -> usize {
        self.blade
    }

    /// returns the geometric grade based on blade count
    ///
    /// # returns
    /// geometric algebra grade: 0=scalar, 1=vector, 2=bivector, 3=trivector
    ///
    /// all geometric objects exhibit one of 4 behavioral patterns that repeat with π/2 periodicity
    /// * Grade 0 behavior (scalar-like): blades 0, 4, 8, 12, 1000, ...
    /// * Grade 1 behavior (vector-like): blades 1, 5, 9, 13, 1001, ...
    /// * Grade 2 behavior (bivector-like): blades 2, 6, 10, 14, 1002, ...
    /// * Grade 3 behavior (trivector-like): blades 3, 7, 11, 15, 1003, ...
    ///
    /// this 4-fold periodicity emerges from the fundamental quadrature relationship
    /// sin(θ+π/2) = cos(θ), which creates a natural cycle through geometric grades
    /// the blade count preserves full dimensional information while the grade
    /// determines the geometric behavior (how it transforms under operations)
    pub fn grade(&self) -> usize {
        self.mod_4_blade()
    }

    /// tests if this angle represents a scalar (blade = 0)
    pub fn is_scalar(&self) -> bool {
        self.mod_4_blade() == 0
    }

    /// tests if this angle represents a vector (blade = 1)  
    pub fn is_vector(&self) -> bool {
        self.mod_4_blade() == 1
    }

    /// tests if this angle represents a bivector (blade = 2)
    pub fn is_bivector(&self) -> bool {
        self.mod_4_blade() == 2
    }

    /// tests if this angle represents a trivector (blade = 3)
    pub fn is_trivector(&self) -> bool {
        self.mod_4_blade() == 3
    }

    /// normalizes blade count to 4D rotation space [0,3]
    ///
    /// # returns
    /// blade count within [0,3] range representing geometric grades
    fn mod_4_blade(&self) -> usize {
        self.blade % 4
    }

    /// returns this angle with blade count reset to base for its grade
    ///
    /// blade accumulation is geometrically primitive - operations like reflection
    /// fundamentally work through blade arithmetic (2 + 2 = 4 blades for double
    /// reflection). however, control loops and iterative algorithms may need
    /// geometric consistency without unbounded blade growth
    ///
    /// this method preserves the angles grade (blade % 4) while resetting the
    /// blade count to its minimum value for that grade:
    /// - grade 0 (scalar): blade = 0
    /// - grade 1 (vector): blade = 1  
    /// - grade 2 (bivector): blade = 2
    /// - grade 3 (trivector): blade = 3
    ///
    /// # example
    /// ```
    /// use geonum::Angle;
    /// let angle = Angle::new_with_blade(1000, 1.0, 4.0); // blade 1000, grade 0
    /// let normalized = angle.base_angle(); // blade 0, grade 0 (same angle)
    /// ```
    pub fn base_angle(&self) -> Angle {
        Angle {
            blade: self.grade(), // reset to base blade for grade
            value: self.value,   // preserve fractional angle
        }
    }

    /// normalizes this angle when value is at or exceeds π/2 boundaries
    fn normalize_boundaries(&self) -> Self {
        let quarter_pi = PI / 2.0;

        // handle floating point precision near π/2
        const EPSILON: f64 = 1e-10;
        if (self.value - quarter_pi).abs() < EPSILON {
            return Self {
                blade: self.blade + 1,
                value: 0.0,
            };
        }

        // handle values >= π/2
        if self.value >= quarter_pi {
            let additional_blades = (self.value / quarter_pi) as usize;
            let final_value = self.value % quarter_pi;
            // check again for precision at boundary after modulo
            if (final_value - quarter_pi).abs() < EPSILON {
                return Self {
                    blade: self.blade + additional_blades + 1,
                    value: 0.0,
                };
            }
            return Self {
                blade: self.blade + additional_blades,
                value: final_value,
            };
        }

        *self
    }

    /// internal geometric addition preserving blade progression and π/2 boundary invariants
    fn geometric_add(&self, other: &Self) -> Self {
        // step 1: add full blade counts (preserve semantic meaning)
        let total_blade = self.blade + other.blade;

        // step 2: add angle values within current π/2 segments
        let total_value = self.value + other.value;

        // avoid floating point buggery
        let quarter_pi = PI / 2.0;
        if total_value == 0.0 {
            // exact case: no boundary crossing
            return Self {
                blade: total_blade,
                value: 0.0,
            };
        }
        if (total_value - quarter_pi).abs() < 1e-15 {
            // exact case: one boundary crossing
            return Self {
                blade: total_blade + 1,
                value: 0.0,
            };
        }

        // step 3: create angle with combined blade and value, then normalize
        let combined = Self {
            blade: total_blade,
            value: total_value,
        };

        combined.normalize_boundaries()
    }

    /// internal geometric subtraction preserving blade progression and π/2 boundary invariants
    fn geometric_sub(&self, other: &Self) -> Self {
        // subtract blade counts and angle values separately to avoid precision buggery
        let blade_diff = self.blade as i64 - other.blade as i64;
        let value_diff = self.value - other.value;

        // avoid floating point buggery with exact cases
        if value_diff.abs() < 1e-15 {
            // exact case: values are equal, result is just blade difference
            let normalized_blade = if blade_diff < 0 {
                let four_rotations = ((-blade_diff + 3) / 4) * 4; // round up to multiple of 4
                (blade_diff + four_rotations) as usize
            } else {
                blade_diff as usize
            };
            return Self {
                blade: normalized_blade,
                value: 0.0,
            };
        }

        // handle negative value: borrow from blade count
        let (intermediate_blade, intermediate_value) = if value_diff < 0.0 {
            let quarter_pi = PI / 2.0;
            let final_value = value_diff + quarter_pi;
            let final_blade = blade_diff - 1;
            (final_blade, final_value)
        } else {
            (blade_diff, value_diff)
        };

        // handle negative blade count: normalize to positive
        let normalized_blade = if intermediate_blade < 0 {
            let four_rotations = ((-intermediate_blade + 3) / 4) * 4; // round up to multiple of 4
            (intermediate_blade + four_rotations) as usize
        } else {
            intermediate_blade as usize
        };

        // create angle and normalize boundaries
        let result = Self {
            blade: normalized_blade,
            value: intermediate_value,
        };

        result.normalize_boundaries()
    }

    /// tests if this angle is opposite to another angle
    ///
    /// two angles are opposite if they differ by π (blade difference of 2)
    /// and have the same value within their π/2 segment
    ///
    /// # arguments
    /// * `other` - the angle to compare with
    ///
    /// # returns
    /// true if the angles are opposites (π apart)
    pub fn is_opposite(&self, other: &Angle) -> bool {
        let blade_diff = (self.blade as i32 - other.blade as i32).abs();
        let values_match = (self.value - other.value).abs() < 1e-15;
        blade_diff == 2 && values_match
    }

    /// dual operation that adds π rotation
    ///
    /// adds 2 to blade count (π rotation) flattening the traditional GA grade map
    /// from n+1 grade levels (0 through n) to just 2 involutive pairs:
    /// - pair 1: grade 0 ↔ grade 2 (scalar ↔ bivector)
    /// - pair 2: grade 1 ↔ grade 3 (vector ↔ trivector)
    ///
    /// this works in any dimension because grades cycle modulo 4
    /// so grade 1000000 in million-D space is just grade 0 (1000000 % 4 = 0)
    /// eliminating dimension-specific k→(n-k) duality formulas
    pub fn dual(&self) -> Angle {
        // add π rotation (2 blade counts)
        *self + Angle::new_with_blade(2, 0.0, 1.0)
    }

    /// computes the undual operation (inverse of dual)
    ///
    /// in geonum's 4-cycle blade structure, undual is the same as dual
    /// because the grade mapping 0↔0, 1↔3, 2↔2, 3↔1 is self-inverse
    pub fn undual(&self) -> Angle {
        self.dual()
    }

    /// conjugate for complex representation: negates the angle
    /// if angle represents e^(iθ), conjugate represents e^(-iθ)
    /// negation is adding π
    pub fn conjugate(&self) -> Angle {
        *self + Angle::new(1.0, 1.0)
    }

    /// returns the angle in radians within [0, 2π)
    /// by taking blade modulo 4 and adding the value component
    ///
    /// useful for interfacing with external code that expects angles in [0, 2π)
    /// such as orbital mechanics, phase calculations, or visualization
    ///
    /// # returns
    /// angle in radians as f64 within [0, 2π)
    pub fn mod_4_angle(&self) -> f64 {
        self.mod_4_blade() as f64 * PI / 2.0 + self.value
    }

    /// negates this angle by adding π rotation (2 blades)
    ///
    /// negation is forward rotation by 180 degrees, not backwards motion
    /// this fundamental operation appears throughout geometry as sign flips,
    /// vector opposites, and complex conjugation
    pub fn negate(&self) -> Angle {
        *self + Angle::new(1.0, 1.0) // add π (2 blades)
    }

    /// projects this angle onto another angle direction
    /// returns the cosine of the angle difference
    pub fn project(&self, onto: Angle) -> f64 {
        let angle_diff = onto - *self;
        angle_diff.mod_4_angle().cos()
    }
}

impl PartialEq for Angle {
    fn eq(&self, other: &Self) -> bool {
        // exact blade comparison
        if self.blade != other.blade {
            return false;
        }

        // avoid floating point buggery with exact cases
        let value_diff = (self.value - other.value).abs();
        if value_diff < 1e-15 {
            return true; // exact match
        }

        // for non-exact cases, use standard floating point comparison
        self.value == other.value
    }
}

impl Eq for Angle {}

impl Add for Angle {
    type Output = Angle;

    fn add(self, other: Self) -> Angle {
        self.geometric_add(&other)
    }
}

impl Add<&Angle> for Angle {
    type Output = Angle;

    fn add(self, other: &Self) -> Angle {
        self.geometric_add(other)
    }
}

impl Add<Angle> for &Angle {
    type Output = Angle;

    fn add(self, other: Angle) -> Angle {
        self.geometric_add(&other)
    }
}

impl Add<&Angle> for &Angle {
    type Output = Angle;

    fn add(self, other: &Angle) -> Angle {
        self.geometric_add(other)
    }
}

impl Sub for Angle {
    type Output = Angle;

    fn sub(self, other: Self) -> Angle {
        self.geometric_sub(&other)
    }
}

impl Sub<&Angle> for Angle {
    type Output = Angle;

    fn sub(self, other: &Self) -> Angle {
        self.geometric_sub(other)
    }
}

impl Sub<Angle> for &Angle {
    type Output = Angle;

    fn sub(self, other: Angle) -> Angle {
        self.geometric_sub(&other)
    }
}

impl Sub<&Angle> for &Angle {
    type Output = Angle;

    fn sub(self, other: &Angle) -> Angle {
        self.geometric_sub(other)
    }
}

impl Mul for Angle {
    type Output = Angle;

    fn mul(self, other: Self) -> Angle {
        self.geometric_add(&other)
    }
}

impl Mul<&Angle> for Angle {
    type Output = Angle;

    fn mul(self, other: &Self) -> Angle {
        self.geometric_add(other)
    }
}

impl Mul<Angle> for &Angle {
    type Output = Angle;

    fn mul(self, other: Angle) -> Angle {
        self.geometric_add(&other)
    }
}

impl Mul<&Angle> for &Angle {
    type Output = Angle;

    fn mul(self, other: &Angle) -> Angle {
        self.geometric_add(other)
    }
}

impl Div<f64> for Angle {
    type Output = Angle;

    fn div(self, divisor: f64) -> Angle {
        // divide both the total angle and reconstruct
        let total_angle = (self.blade as f64) * (PI / 2.0) + self.value;
        let divided_angle = total_angle / divisor;
        Angle::new(divided_angle, PI)
    }
}

impl Div<f64> for &Angle {
    type Output = Angle;

    fn div(self, divisor: f64) -> Angle {
        let total_angle = (self.blade as f64) * (PI / 2.0) + self.value;
        let divided_angle = total_angle / divisor;
        Angle::new(divided_angle, PI)
    }
}

impl Div<Angle> for Angle {
    type Output = Angle;

    fn div(self, other: Angle) -> Angle {
        self.geometric_sub(&other)
    }
}

impl Div<&Angle> for Angle {
    type Output = Angle;

    fn div(self, other: &Angle) -> Angle {
        self.geometric_sub(other)
    }
}

impl Div<Angle> for &Angle {
    type Output = Angle;

    fn div(self, other: Angle) -> Angle {
        self.geometric_sub(&other)
    }
}

impl Div<&Angle> for &Angle {
    type Output = Angle;

    fn div(self, other: &Angle) -> Angle {
        self.geometric_sub(other)
    }
}

impl PartialOrd for Angle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Angle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.blade.cmp(&other.blade) {
            std::cmp::Ordering::Equal => {
                // only finite values in Angles, so unwrap is safe
                self.value.partial_cmp(&other.value).unwrap()
            }
            other => other,
        }
    }
}

impl std::fmt::Display for Angle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Angle(blade: {}, value: {:.4})", self.blade, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn it_sums_less_than_a_quarter_turn() {
        let angle1 = Angle::new(1.0, 8.0); // π/8
        let angle2 = Angle::new(1.0, 6.0); // π/6

        let sum = angle1 + angle2; // π/8 + π/6 = 7π/24 < π/2

        assert_eq!(sum.blade(), 0); // no boundary crossing
        assert!((sum.value() - (7.0 * PI / 24.0)).abs() < EPSILON);
    }

    #[test]
    fn it_sums_greater_than_a_quarter_turn() {
        let angle1 = Angle::new(1.0, 3.0); // π/3
        let angle2 = Angle::new(1.0, 4.0); // π/4

        let sum = angle1 + angle2; // π/3 + π/4 = 7π/12 > π/2

        assert_eq!(sum.blade(), 1); // crosses π/2 boundary, increments blade
        assert!((sum.value() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON);
    }

    #[test]
    fn it_sums_rotations_to_multiple_blades() {
        let angle1 = Angle::new(3.0, 4.0); // 3π/4, blade 1, value π/4
        let angle2 = Angle::new(5.0, 4.0); // 5π/4, blade 2, value π/4

        let sum = angle1 + angle2; // blade sum: (1+2)%4=3, value sum: π/4+π/4=π/2
                                   // π/2 crosses boundary: final_blade=(3+1)%4=0

        assert_eq!(sum.blade(), 4); // preserves full blade count: 1+2+1=4
        assert!((sum.value()).abs() < EPSILON); // π/2 boundary crossing leaves no remainder
    }

    #[test]
    fn it_constructs_blade_0_from_large_angles() {
        let angle = Angle::new(4.0, 2.0); // 4*(π/2) = 2π

        assert_eq!(angle.blade(), 4); // preserves original blade count
        assert_eq!(angle.grade(), 0); // 4 % 4 = 0 (scalar grade)
        assert!((angle.value()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_1_from_large_angles() {
        let angle = Angle::new(5.0, 2.0); // 5*(π/2)

        assert_eq!(angle.blade(), 5); // preserves original blade count
        assert_eq!(angle.grade(), 1); // 5 % 4 = 1 (vector grade)
        assert!((angle.value()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_2_from_large_angles() {
        let angle = Angle::new(6.0, 2.0); // 6*(π/2)

        assert_eq!(angle.blade(), 6); // preserves original blade count
        assert_eq!(angle.grade(), 2); // 6 % 4 = 2 (bivector grade)
        assert!((angle.value()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_3_from_large_angles() {
        let angle = Angle::new(7.0, 2.0); // 7*(π/2)

        assert_eq!(angle.blade(), 7); // preserves original blade count
        assert_eq!(angle.grade(), 3); // 7 % 4 = 3 (trivector grade)
        assert!((angle.value()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_preserves_blade_1000() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)

        assert_eq!(angle.blade(), 1000); // preserves original blade count
        assert_eq!(angle.grade(), 0); // 1000 % 4 = 0 (scalar grade)
        assert!(angle.is_scalar()); // grade test works
        assert!(!angle.is_vector()); // other grades false
    }

    #[test]
    fn it_sums_high_blades() {
        let angle1 = Angle::new(1000.0, 2.0); // blade 100
        let angle2 = Angle::new(500.0, 2.0); // blade 50

        let sum = angle1 + angle2; // blade 1000 + 500 = 1500

        assert_eq!(sum.blade(), 1500); // preserves full semantic blade count
        assert_eq!(sum.grade(), 0); // 150 % 4 = 2 (bivector grade)
        assert!(sum.is_scalar()); // grade test works
    }

    #[test]
    fn it_subtracts_pi_over_6_from_pi_over_3() {
        let angle1 = Angle::new(1.0, 3.0); // π/3
        let angle2 = Angle::new(1.0, 6.0); // π/6

        let diff = angle1 - angle2; // π/3 - π/6 = π/6

        assert_eq!(diff.blade(), 0); // π/6 is less than π/2, no blade increment
        assert!((diff.value() - PI / 6.0).abs() < EPSILON); // remainder is π/6
    }

    #[test]
    fn it_subtracts_pi_over_3_from_4pi_over_3() {
        let angle1 = Angle::new(4.0, 3.0); // 4π/3
        let angle2 = Angle::new(1.0, 3.0); // π/3

        let diff = angle1 - angle2; // 4π/3 - π/3 = π

        assert_eq!(diff.blade(), 2); // π = 2 * π/2, so blade = 2
        assert!((diff.value()).abs() < EPSILON); // exact multiple of π/2 has no remainder
    }

    #[test]
    fn it_subtracts_11pi_over_6_from_pi_over_6() {
        let angle1 = Angle::new(1.0, 6.0); // π/6
        let angle2 = Angle::new(11.0, 6.0); // 11π/6

        let diff = angle1 - angle2; // π/6 - 11π/6 = -10π/6 = -5π/3

        // negative result should normalize to positive angle
        // -5π/3 = -5π/3 + 2π = π/3
        assert_eq!(diff.blade(), 0); // π/3 is less than π/2, no blade increment
        assert!((diff.value() - PI / 3.0).abs() < EPSILON); // remainder is π/3
    }

    #[test]
    fn it_multiplies_angles_as_addition() {
        let angle1 = Angle::new(1.0, 8.0); // π/8
        let angle2 = Angle::new(1.0, 6.0); // π/6

        let product = angle1 * angle2; // π/8 * π/6 = π/8 + π/6 = 7π/24

        assert_eq!(product.blade(), 0); // 7π/24 < π/2, no blade increment
        assert!((product.value() - (7.0 * PI / 24.0)).abs() < EPSILON);
    }

    #[test]
    fn it_computes_sin_of_1000_blade() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)

        let sin_result = angle.mod_4_angle().sin();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // sin(0 + 0.0) = sin(0) = 0
        assert!((sin_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_sin_with_1003_blade() {
        let angle = Angle::new(1003.0, 2.0); // 1003*(π/2)

        let sin_result = angle.mod_4_angle().sin();

        // 1003 % 4 = 3, so grade is 3 (trivector)
        // sin(3π/2 + 0.0) = sin(3π/2) = -1
        assert!((sin_result - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_of_1000_blade() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)

        let cos_result = angle.mod_4_angle().cos();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // cos(0 + 0.0) = cos(0) = 1
        assert!((cos_result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_with_1001_blade() {
        let angle = Angle::new(1001.0, 2.0); // 1001*(π/2)

        let cos_result = angle.mod_4_angle().cos();

        // 1001 % 4 = 1, so grade is 1 (vector)
        // cos(π/2 + 0.0) = cos(π/2) = 0
        assert!((cos_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_creates_angle_with_additional_blade() {
        // test basic functionality
        let angle = Angle::new_with_blade(2, 1.0, 4.0); // 2 extra blades + π/4
        assert_eq!(angle.blade(), 2); // π/4 gives blade 0, + 2 = blade 2
        assert!((angle.value() - PI / 4.0).abs() < EPSILON);

        // test with angle that causes boundary crossing
        let angle2 = Angle::new_with_blade(1, 3.0, 4.0); // 1 extra blade + 3π/4
        assert_eq!(angle2.blade(), 2); // 3π/4 gives blade 1, + 1 = blade 2
        assert!((angle2.value() - PI / 4.0).abs() < EPSILON); // 3π/4 - π/2 = π/4

        // test with zero additional blades
        let angle3 = Angle::new_with_blade(0, 1.0, 2.0); // 0 extra blades + π/2
        assert_eq!(angle3.blade(), 1); // π/2 gives blade 1, + 0 = blade 1
        assert!(angle3.value().abs() < EPSILON); // exact π/2 leaves no remainder

        // test with large additional blade count
        let angle4 = Angle::new_with_blade(10, 0.0, 1.0); // 10 extra blades + 0
        assert_eq!(angle4.blade(), 10); // 0 gives blade 0, + 10 = blade 10
        assert!(angle4.value().abs() < EPSILON);
    }

    #[test]
    fn it_computes_tan_of_1000_blade() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)

        let tan_result = angle.mod_4_angle().tan();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // tan(0 + 0.0) = tan(0) = 0
        assert!((tan_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_tan_with_1002_blade() {
        let angle = Angle::new(1002.0, 2.0); // 1002*(π/2)

        let tan_result = angle.mod_4_angle().tan();

        // 1002 % 4 = 2, so grade is 2 (bivector)
        // tan(π + 0.0) = tan(π) = 0
        assert!((tan_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_compares_angles_for_equality() {
        let angle1 = Angle::new(1000.0, 2.0); // 1000*(π/2)
        let angle2 = Angle::new(1000.0, 2.0); // same
        let angle3 = Angle::new(1001.0, 2.0); // different blade
        let angle4 = Angle::new(1.0, 4.0); // different value

        // exact equality
        assert_eq!(angle1, angle2);

        // different blades
        assert_ne!(angle1, angle3);

        // different values
        assert_ne!(angle1, angle4);
    }

    #[test]
    fn it_detects_opposite_angles() {
        // test that angles differing by π (blade difference of 2) are opposites

        // case 1: blade 0 and blade 2 with same value
        let scalar = Angle::new(0.0, 1.0); // blade 0, value 0
        let bivector = Angle::new(1.0, 1.0); // blade 2, value 0
        assert!(
            scalar.is_opposite(&bivector),
            "scalar and bivector are opposites"
        );
        assert!(
            bivector.is_opposite(&scalar),
            "opposite detection is symmetric"
        );

        // case 2: blade 1 and blade 3 with same value
        let vector = Angle::new(1.0, 2.0); // blade 1, value 0
        let trivector = Angle::new(3.0, 2.0); // blade 3, value 0
        assert!(
            vector.is_opposite(&trivector),
            "vector and trivector are opposites"
        );

        // case 3: different values - not opposites
        let angle1 = Angle::new(1.0, 4.0); // blade 0, value π/4
        let angle2 = Angle::new(1.0, 1.0); // blade 2, value 0
        assert!(
            !angle1.is_opposite(&angle2),
            "different values are not opposites"
        );

        // case 4: blade difference not 2 - not opposites
        let angle3 = Angle::new(0.0, 1.0); // blade 0
        let angle4 = Angle::new(1.0, 2.0); // blade 1
        assert!(
            !angle3.is_opposite(&angle4),
            "blade difference of 1 is not opposite"
        );

        // case 5: same angle - not opposite
        let angle5 = Angle::new(1.0, 3.0); // blade 0, value π/3
        assert!(!angle5.is_opposite(&angle5), "same angle is not opposite");

        // case 6: high blade counts
        let high1 = Angle::new(10.0, 2.0); // blade 10, value 0
        let high2 = Angle::new(12.0, 2.0); // blade 12, value 0
        assert!(
            high1.is_opposite(&high2),
            "high blades with difference 2 are opposites"
        );
    }

    #[test]
    fn it_creates_angle_from_cartesian_coordinates() {
        // test basic quadrants
        let angle_0 = Angle::new_from_cartesian(1.0, 0.0); // positive x-axis
        assert_eq!(angle_0.blade(), 0);
        assert!(angle_0.value().abs() < EPSILON);

        let angle_90 = Angle::new_from_cartesian(0.0, 1.0); // positive y-axis
        assert_eq!(angle_90.blade(), 1);
        assert!(angle_90.value().abs() < EPSILON);

        let angle_180 = Angle::new_from_cartesian(-1.0, 0.0); // negative x-axis
        assert_eq!(angle_180.blade(), 2);
        assert!(angle_180.value().abs() < EPSILON);

        let angle_270 = Angle::new_from_cartesian(0.0, -1.0); // negative y-axis
        assert_eq!(angle_270.blade(), 3);
        assert!(angle_270.value().abs() < EPSILON);

        // test 45 degree angle
        let angle_45 = Angle::new_from_cartesian(1.0, 1.0);
        assert_eq!(angle_45.blade(), 0);
        assert!((angle_45.value() - PI / 4.0).abs() < EPSILON);

        // test 3-4-5 triangle (arctan(3/4))
        let angle_triangle = Angle::new_from_cartesian(4.0, 3.0);
        let expected_radians = (3.0_f64).atan2(4.0);
        let expected_angle = Angle::new(expected_radians, PI);
        assert_eq!(angle_triangle.blade(), expected_angle.blade());
        assert!((angle_triangle.value() - expected_angle.value()).abs() < EPSILON);
    }

    #[test]
    fn it_preserves_blade_with_signed_angles() {
        // reminder: before creating elaborate "grade-preserving" methods,
        // remember that angles can be negative and this solves most problems naturally

        let positive_angle = Angle::new(1.0, 3.0); // π/3
        let negative_angle = Angle::new(-1.0, 3.0); // -π/3

        // negative angles give opposite sin values (for anti-commutativity)
        assert!(
            (positive_angle.mod_4_angle().sin() + negative_angle.mod_4_angle().sin()).abs()
                < EPSILON
        );

        // but same cos values (preserving geometric relationships)
        assert!(
            (positive_angle.mod_4_angle().cos() - negative_angle.mod_4_angle().cos()).abs()
                < EPSILON
        );

        // when you need anti-commutativity (v ∧ w = -(w ∧ v)),
        // the solution is often just to negate one result
    }

    #[test]
    fn it_handles_negative_angle_on_zero_blade() {
        // test subtracting π/2 from blade=0 (angle 0)
        let zero = Angle::new(0.0, 1.0); // 0 radians (blade=0)
        let half_pi = Angle::new(1.0, 2.0); // π/2
        let result = zero - half_pi;

        // 0 - π/2 = -π/2 wraps around to 3π/2
        // 3π/2 = 3 * (π/2) + 0
        assert_eq!(result.blade(), 3);
        assert_eq!(result.value(), 0.0);
    }

    #[test]
    fn it_handles_negative_angle_on_one_blade() {
        // test subtracting π from blade=1 (angle π/2)
        let one_blade = Angle::new(1.0, 2.0); // π/2 (blade=1)
        let pi = Angle::new(1.0, 1.0); // π
        let result = one_blade - pi;

        // π/2 - π = -π/2 normalizes to 3π/2
        // 3π/2 = 3 * (π/2) + 0
        assert_eq!(result.blade(), 3);
        assert_eq!(result.value(), 0.0);
    }

    #[test]
    fn it_creates_negative_pi_over_2() {
        // test what Angle::new(-1.0, 2.0) actually creates
        let neg_half_pi = Angle::new(-1.0, 2.0);
        println!(
            "Angle::new(-1.0, 2.0) gives blade={}, value={}",
            neg_half_pi.blade(),
            neg_half_pi.value()
        );

        // test adding it to zero
        let zero = Angle::new(0.0, 1.0);
        let result = zero + neg_half_pi;
        println!(
            "0 + Angle::new(-1.0, 2.0) gives blade={}, value={}",
            result.blade(),
            result.value()
        );
    }

    #[test]
    fn it_computes_dual_angle() {
        // test dual operation using π-rotation (adds 2 blades)
        // this creates grade transformations: 0→2, 1→3, 2→0, 3→1

        // scalar (blade 0) → bivector (blade 2)
        let scalar = Angle::new(0.0, 1.0); // blade 0
        let dual_scalar = scalar.dual();
        assert_eq!(dual_scalar.grade(), 2);
        assert_eq!(dual_scalar.blade(), 2); // blade 0 + 2 = blade 2

        // vector (blade 1) → trivector (blade 3)
        let vector = Angle::new(1.0, 2.0); // blade 1
        let dual_vector = vector.dual();
        assert_eq!(dual_vector.grade(), 3);
        assert_eq!(dual_vector.blade(), 3); // blade 1 + 2 = blade 3

        // bivector (blade 2) → scalar (blade 4 = 0 mod 4)
        let bivector = Angle::new(2.0, 2.0); // blade 2
        let dual_bivector = bivector.dual();
        assert_eq!(dual_bivector.grade(), 0);
        assert_eq!(dual_bivector.blade(), 4); // blade 2 + 2 = blade 4

        // trivector (blade 3) → vector (blade 5 = 1 mod 4)
        let trivector = Angle::new(3.0, 2.0); // blade 3
        let dual_trivector = trivector.dual();
        assert_eq!(dual_trivector.grade(), 1);
        assert_eq!(dual_trivector.blade(), 5); // blade 3 + 2 = blade 5

        // prove involution property: dual(dual(x)) returns to original grade
        let scalar_double = scalar.dual().dual();
        assert_eq!(scalar_double.grade(), scalar.grade());
        assert_eq!(scalar_double.value(), scalar.value());

        let vector_double = vector.dual().dual();
        assert_eq!(vector_double.grade(), vector.grade());
        assert_eq!(vector_double.value(), vector.value());

        let bivector_double = bivector.dual().dual();
        assert_eq!(bivector_double.grade(), bivector.grade());
        assert_eq!(bivector_double.value(), bivector.value());

        let trivector_double = trivector.dual().dual();
        assert_eq!(trivector_double.grade(), trivector.grade());
        assert_eq!(trivector_double.value(), trivector.value());

        // test high blade numbers still follow the pattern
        let high_blade = Angle::new_with_blade(1001, 0.0, 1.0); // blade 1001, grade 1
        let dual_high = high_blade.dual();
        assert_eq!(dual_high.grade(), 3); // grade 1 → grade 3
        assert_eq!(dual_high.blade(), 1003); // blade 1001 + 2 = blade 1003

        // π-rotation dual creates grade pairs: 0↔2, 1↔3
        // applying twice adds 4 blades (2π rotation) returning to original grade
    }

    #[test]
    fn it_proves_undual_equals_dual() {
        // undual is the same as dual because the k → (4-k) % 4 mapping is self-inverse
        let test_angles = vec![
            Angle::new(0.0, 1.0),                 // grade 0
            Angle::new(1.0, 2.0),                 // grade 1
            Angle::new(2.0, 2.0),                 // grade 2
            Angle::new(3.0, 2.0),                 // grade 3
            Angle::new_with_blade(100, 1.0, 3.0), // high blade
        ];

        for angle in test_angles {
            let dual = angle.dual();
            let undual = angle.undual();

            // dual and undual are the same operation
            assert_eq!(dual.blade(), undual.blade());
            assert_eq!(dual.value(), undual.value());

            // applying dual twice returns to original grade
            let double_dual = angle.dual().dual();
            assert_eq!(double_dual.grade(), angle.grade());
            assert_eq!(double_dual.value(), angle.value());
        }
    }

    #[test]
    fn it_orders_angles_by_blade_then_value() {
        // test ordering by blade first
        let blade0 = Angle::new(1.0, 3.0); // PI/3 (blade 0)
        let blade1 = Angle::new(3.0, 2.0); // 3PI/2 (blade 3)
        let blade2 = Angle::new(5.0, 2.0); // 5PI/2 (blade 5)

        assert!(blade0 < blade1);
        assert!(blade1 < blade2);
        assert!(blade0 < blade2);

        // test ordering by value when blades are equal
        let angle1 = Angle::new(1.0, 6.0); // PI/6
        let angle2 = Angle::new(1.0, 4.0); // PI/4
        let angle3 = Angle::new(1.0, 3.0); // PI/3

        assert!(angle1 < angle2); // PI/6 < PI/4
        assert!(angle2 < angle3); // PI/4 < PI/3
        assert!(angle1 < angle3); // PI/6 < PI/3

        // test with high blade counts
        let high1 = Angle::new(1000.0, 2.0); // blade 1000
        let high2 = Angle::new(1001.0, 2.0); // blade 1001
        assert!(high1 < high2);

        // test equality in ordering
        let eq1 = Angle::new(2.0, 3.0); // blade 0, value 2PI/3
        let eq2 = Angle::new(2.0, 3.0); // same
        assert!(eq1 <= eq2);
        assert!(eq1 >= eq2);
        assert!(eq1 >= eq2);
        assert!(eq1 <= eq2);

        // test that blade takes precedence over value
        let small_blade_big_value = Angle::new(0.8, 1.0); // 0.8PI (blade 1, value 0.3PI)
        let big_blade_small_value = Angle::new(1.1, 1.0); // 1.1PI (blade 2, value 0.1PI)
                                                          // even though 0.8PI > 1.1PI in terms of raw angle value within their blades,
                                                          // blade 1 < blade 2, so the first angle is less than the second
        assert!(small_blade_big_value < big_blade_small_value);
    }

    #[test]
    fn it_computes_mod_4_angle() {
        // test basic cases within first rotation
        let angle1 = Angle::new(0.5, 4.0); // π/8, blade 0
        assert!((angle1.mod_4_angle() - PI / 8.0).abs() < EPSILON);

        let angle2 = Angle::new(1.0, 2.0); // π/2, blade 1
        assert!((angle2.mod_4_angle() - PI / 2.0).abs() < EPSILON);

        let angle3 = Angle::new(3.0, 2.0); // 3π/2, blade 3
        assert!((angle3.mod_4_angle() - 3.0 * PI / 2.0).abs() < EPSILON);

        // test angles with blade count > 3
        let angle4 = Angle::new(5.0, 2.0); // 5π/2, blade 5 -> blade 1
        assert!((angle4.mod_4_angle() - PI / 2.0).abs() < EPSILON);

        let angle5 = Angle::new(8.0, 2.0); // 8π/2 = 4π, blade 8 -> blade 0
        assert!(angle5.mod_4_angle().abs() < EPSILON);

        let angle6 = Angle::new(10.0, 2.0); // 10π/2 = 5π, blade 10 -> blade 2
        assert!((angle6.mod_4_angle() - PI).abs() < EPSILON);

        // test with non-zero value component
        let angle7 = Angle::new(9.5, 2.0); // 9.5π/2, blade 9, value π/4
                                           // blade 9 % 4 = 1, so result is π/2 + π/4 = 3π/4
        assert!((angle7.mod_4_angle() - 3.0 * PI / 4.0).abs() < EPSILON);

        // test large blade counts
        let angle8 = Angle::new(1000.0, 2.0); // blade 1000 -> blade 0
        assert!(angle8.mod_4_angle().abs() < EPSILON);

        let angle9 = Angle::new(1001.0, 2.0); // blade 1001 -> blade 1
        assert!((angle9.mod_4_angle() - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn it_adds_two_blades_when_dualizing_bivector() {
        // π-rotation dual adds 2 blades

        let bivector_angle = Angle::new_with_blade(2, 0.0, 1.0); // blade 2, grade 2
        let dual_angle = bivector_angle.dual();

        // bivector → scalar (grade 2 → grade 0)
        assert_eq!(dual_angle.grade(), 0);

        // blade 2 + 2 = blade 4
        assert_eq!(dual_angle.blade(), 4);
    }

    #[test]
    fn it_negates_angle() {
        // test angle negation (complex conjugation)
        // if angle represents e^(iθ), negated angle represents e^(-iθ)

        // test π/3 angle
        let angle = Angle::new(1.0, 3.0); // π/3

        // negate by subtracting from zero
        let zero = Angle::new(0.0, 1.0);
        let negated = zero - angle;

        // -π/3 = 2π - π/3 = 5π/3
        let expected_neg = Angle::new(5.0, 3.0); // 5π/3
        assert_eq!(negated, expected_neg);

        // test with π/2 (blade 1)
        let right_angle = Angle::new(1.0, 2.0); // π/2
        let neg_right = zero - right_angle;

        // -π/2 = 2π - π/2 = 3π/2
        let expected_neg_right = Angle::new(3.0, 2.0); // 3π/2
        assert_eq!(neg_right, expected_neg_right);

        // test full rotation
        let full = Angle::new(2.0, 1.0); // 2π
        let neg_full = zero - full;

        // -2π = 0
        assert_eq!(neg_full, zero);

        // test conjugate method adds π (forward rotation)
        let angle_conj = angle.conjugate();
        assert_eq!(
            angle_conj.blade() - angle.blade(),
            2,
            "conjugate adds 2 blades"
        );

        let right_conj = right_angle.conjugate();
        assert_eq!(
            right_conj.blade() - right_angle.blade(),
            2,
            "conjugate adds π"
        );
    }

    #[test]
    fn it_resets_angle_blade_to_minimum() {
        // test grade preservation with blade reset
        let angle_high_blade = Angle::new_with_blade(1000, 1.0, 4.0); // blade 1000, grade 0
        let normalized = angle_high_blade.base_angle();
        assert_eq!(normalized.blade(), 0, "grade 0 resets to blade 0");
        assert_eq!(
            normalized.value(),
            angle_high_blade.value(),
            "angle value preserved"
        );

        // test each grade resets to minimum blade
        let grade1 = Angle::new_with_blade(101, 0.0, 1.0); // blade 101, grade 1
        assert_eq!(grade1.base_angle().blade(), 1, "grade 1 resets to blade 1");

        let grade2 = Angle::new_with_blade(102, 0.0, 1.0); // blade 102, grade 2
        assert_eq!(grade2.base_angle().blade(), 2, "grade 2 resets to blade 2");

        let grade3 = Angle::new_with_blade(103, 0.0, 1.0); // blade 103, grade 3
        assert_eq!(grade3.base_angle().blade(), 3, "grade 3 resets to blade 3");

        // test already minimal blade unchanged
        let minimal = Angle::new(1.0, 2.0); // π/2, blade 1
        assert_eq!(minimal.base_angle(), minimal, "minimal blade unchanged");

        // test that base_angle and dual operations have consistent grade results
        let high_blade = Angle::new_with_blade(101, 1.0, 4.0); // blade 101, grade 1
        let dual_then_base = high_blade.dual().base_angle();
        let base_then_dual = high_blade.base_angle().dual();

        // both should end at same grade (operations dont commute in blade, but do in grade)
        assert_eq!(
            dual_then_base.grade(),
            base_then_dual.grade(),
            "both orders reach same grade"
        );
    }

    #[test]
    fn it_negates_angle_with_forward_rotation() {
        // test basic negation adds π (2 blades)
        let angle = Angle::new(1.0, 4.0); // π/4
        let negated = angle.negate();

        assert_eq!(
            negated.blade() - angle.blade(),
            2,
            "negation adds 2 blades (π rotation)"
        );
        assert_eq!(negated.value(), angle.value(), "fractional angle preserved");

        // test double negation adds 4 blades (2π rotation)
        let double_neg = negated.negate();
        assert_eq!(
            double_neg.blade() - angle.blade(),
            4,
            "double negation adds 4 blades"
        );

        // after base_angle(), double negation returns to original grade
        assert_eq!(
            double_neg.base_angle().grade(),
            angle.grade(),
            "double negation preserves grade after reset"
        );

        // test negation of each grade
        let scalar = Angle::new_with_blade(0, 1.0, 3.0); // grade 0
        let vector = Angle::new_with_blade(1, 1.0, 3.0); // grade 1
        let bivector = Angle::new_with_blade(2, 1.0, 3.0); // grade 2
        let trivector = Angle::new_with_blade(3, 1.0, 3.0); // grade 3

        assert_eq!(scalar.negate().blade(), 2, "scalar → bivector (0+2=2)");
        assert_eq!(vector.negate().blade(), 3, "vector → trivector (1+2=3)");
        assert_eq!(
            bivector.negate().blade(),
            4,
            "bivector → next scalar (2+2=4)"
        );
        assert_eq!(
            trivector.negate().blade(),
            5,
            "trivector → next vector (3+2=5)"
        );

        // test grade transformations under negation
        assert_eq!(
            scalar.negate().grade(),
            2,
            "scalar negates to bivector grade"
        );
        assert_eq!(
            vector.negate().grade(),
            3,
            "vector negates to trivector grade"
        );
        assert_eq!(
            bivector.negate().grade(),
            0,
            "bivector negates to scalar grade"
        );
        assert_eq!(
            trivector.negate().grade(),
            1,
            "trivector negates to vector grade"
        );

        // test negation preserves forward-only rotation principle
        let original = Angle::new(3.0, 4.0); // 3π/4
        let neg_once = original.negate();
        let neg_twice = neg_once.negate();

        // blades only accumulate forward, never backwards
        assert!(
            neg_once.blade() > original.blade(),
            "first negation increases blade"
        );
        assert!(
            neg_twice.blade() > neg_once.blade(),
            "second negation increases blade further"
        );

        // test negation at blade boundaries
        let at_boundary = Angle::new(1.0, 2.0); // exactly π/2, blade 1
        let neg_boundary = at_boundary.negate();
        assert_eq!(
            neg_boundary.blade(),
            3,
            "negation from blade 1 goes to blade 3"
        );
        assert_eq!(
            neg_boundary.value(),
            0.0,
            "value at exact boundary becomes 0"
        );
    }

    #[test]
    fn it_normalizes_angles_exceeding_pi_2_to_next_blade() {
        // geometric_sub now normalizes values >= π/2 to next blade

        let angle1 = Angle::new(0.0, 1.0); // 0 radians
        let angle2 = Angle::new(1.0, 2.0); // π/2 radians (already normalized to blade 1)

        let diff = angle2 - angle1;

        // π/2 difference is blade 1, value 0
        assert_eq!(diff.blade(), 1, "π/2 difference is blade 1");
        assert!(diff.value().abs() < 1e-10, "blade 1 has value ~0");
    }

    #[test]
    fn test_normalize_boundaries() {
        use std::f64::consts::PI;

        // test value very close to π/2 normalizes to next blade
        let angle_near_pi_2 = Angle {
            blade: 2,
            value: PI / 2.0 - 1e-11,
        };
        let normalized = angle_near_pi_2.normalize_boundaries();
        assert_eq!(normalized.blade(), 3);
        assert_eq!(normalized.value(), 0.0);

        // test value exactly π/2 normalizes to next blade
        let angle_at_pi_2 = Angle {
            blade: 1,
            value: PI / 2.0,
        };
        let normalized = angle_at_pi_2.normalize_boundaries();
        assert_eq!(normalized.blade(), 2);
        assert_eq!(normalized.value(), 0.0);

        // test value > π/2 normalizes blade and reduces value
        let angle_over_pi_2 = Angle {
            blade: 0,
            value: PI * 0.75, // 3π/4
        };
        let normalized = angle_over_pi_2.normalize_boundaries();
        assert_eq!(normalized.blade(), 1);
        assert!((normalized.value() - PI / 4.0).abs() < 1e-10);

        // test small positive value stays unchanged
        let angle_small = Angle {
            blade: 5,
            value: 0.1,
        };
        let normalized = angle_small.normalize_boundaries();
        assert_eq!(normalized.blade(), 5);
        assert_eq!(normalized.value(), 0.1);

        // test value that's 2π normalizes to blade 4 value 0
        let angle_2pi = Angle {
            blade: 0,
            value: 2.0 * PI,
        };
        let normalized = angle_2pi.normalize_boundaries();
        assert_eq!(normalized.blade(), 4);
        assert_eq!(normalized.value(), 0.0);
    }

    #[test]
    fn it_projects() {
        // project angle onto itself
        let angle = Angle::new(1.0, 4.0); // π/4
        let self_proj = angle.project(angle);
        assert!((self_proj - 1.0).abs() < 1e-10); // cos(0) = 1

        // project onto perpendicular angle
        let perp = Angle::new(1.0, 2.0); // π/2
        let perp_proj = angle.project(perp);
        assert!((perp_proj - (PI / 4.0).cos()).abs() < 1e-10); // cos(π/2 - π/4) = cos(π/4)

        // project onto opposite angle
        let opposite = angle + Angle::new(1.0, 1.0); // π/4 + π
        let opp_proj = angle.project(opposite);
        assert!((opp_proj + 1.0).abs() < 1e-10); // cos(π) = -1

        // test grade cycling
        let high_blade = Angle::new_with_blade(1000, 1.0, 4.0); // blade 1000, π/4 remainder
        let low_blade = Angle::new(1.0, 4.0); // blade 0, π/4
        let cycle_proj = high_blade.project(low_blade);
        assert!((cycle_proj - 1.0).abs() < 1e-10); // equivalent angles project to 1

        // test angle wrapping via geometric_sub negative blade handling
        let small_angle = Angle::new(1.0, 8.0); // π/8, blade 0, value π/8
        let large_angle = Angle::new_with_blade(5, 1.0, 4.0); // blade 5, π/4
        let wrapped_diff = large_angle - small_angle;
        // large_angle - small_angle = blade 5 - blade 0 = blade 5, value π/4 - π/8 = π/8
        assert_eq!(wrapped_diff.blade(), 5);
        assert!((wrapped_diff.value() - PI / 8.0).abs() < 1e-10);
        assert!((wrapped_diff.mod_4_angle() - 1.9634954084936207).abs() < 1e-10);
    }
}
