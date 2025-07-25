use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

/// Angle struct that maintains the angle-blade invariant
///
/// Encapsulates the fundamental geometric number constraint:
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

        Self { value, blade }
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
        // atan2 returns radians directly, not pi radians
        // need to normalize to positive angle first
        let normalized = if angle_radians < 0.0 {
            angle_radians + 2.0 * PI
        } else {
            angle_radians
        };
        // decompose into blade and value
        let blade = (normalized / (PI / 2.0)) as usize;
        let value = normalized % (PI / 2.0);
        Self { blade, value }
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
        self.geometric_add(&delta)
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

        // step 3: handle π/2 boundary crossings that change geometric grade
        // when value exceeds π/2, we cross into the next geometric grade
        // multiple crossings possible when both angles near π/2 limit
        let additional_crossings = (total_value / quarter_pi) as usize;
        let final_value = total_value % quarter_pi;

        // step 4: compute final blade count including boundary crossings
        // preserves full semantic blade count
        let final_blade = total_blade + additional_crossings;

        Self {
            blade: final_blade,
            value: final_value,
        }
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
        let (final_blade, final_value) = if value_diff < 0.0 {
            let quarter_pi = PI / 2.0;
            let final_value = value_diff + quarter_pi;
            let final_blade = blade_diff - 1;
            (final_blade, final_value)
        } else {
            (blade_diff, value_diff)
        };

        // handle negative blade count: normalize to positive
        let normalized_blade = if final_blade < 0 {
            let four_rotations = ((-final_blade + 3) / 4) * 4; // round up to multiple of 4
            (final_blade + four_rotations) as usize
        } else {
            final_blade as usize
        };

        Self {
            blade: normalized_blade,
            value: final_value,
        }
    }

    /// returns the angle offset for the current grade
    ///
    /// # returns
    /// angle offset: 0, π/2, π, or 3π/2 based on grade
    fn grade_offset(&self) -> f64 {
        match self.mod_4_blade() {
            0 => 0.0,            // 0
            1 => PI / 2.0,       // π/2
            2 => PI,             // π
            3 => 3.0 * PI / 2.0, // 3π/2
            _ => unreachable!(),
        }
    }

    /// computes sine of the total angle
    ///
    /// # returns
    /// sin(total_angle)
    pub fn sin(&self) -> f64 {
        (self.value + self.grade_offset()).sin()
    }

    /// computes cosine of the total angle
    ///
    /// # returns  
    /// cos(total_angle)
    pub fn cos(&self) -> f64 {
        (self.value + self.grade_offset()).cos()
    }

    /// computes tangent of the total angle
    ///
    /// # returns
    /// tan(total_angle)
    pub fn tan(&self) -> f64 {
        (self.value + self.grade_offset()).tan()
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

    /// computes the rotation needed for the dual operation
    ///
    /// # arguments
    /// * `pseudoscalar_blade` - the blade count of the pseudoscalar
    ///
    /// # returns
    /// the angle to rotate by for the dual
    pub fn dual_rotation_for_blade(&self, pseudoscalar_blade: usize) -> Angle {
        // dual maps grade k to grade (pseudo_grade - k) mod 4
        // we need to rotate within the current 4-blade cycle

        let current_grade = self.mod_4_blade();
        let pseudo_grade = pseudoscalar_blade % 4;

        // compute target grade
        let target_grade = (pseudo_grade + 4 - current_grade) % 4;

        // compute minimal rotation within current cycle
        // from current_grade to target_grade
        let grade_diff = (target_grade + 4 - current_grade) % 4;

        // return rotation angle
        Angle::new(grade_diff as f64, 2.0)
    }

    /// computes the rotation needed for the undual operation
    ///
    /// # arguments
    /// * `pseudoscalar_blade` - the blade count of the pseudoscalar
    ///
    /// # returns
    /// the angle to rotate by for the undual (inverse dual)
    pub fn undual_rotation_for_blade(&self, pseudoscalar_blade: usize) -> Angle {
        // undual reverses the dual operation
        // we need to compute the forward rotation that brings us back

        let current_grade = self.mod_4_blade();
        let pseudo_grade = pseudoscalar_blade % 4;

        // find what grade we came from before dual
        // if dual maps original → current via (pseudo - original) % 4 = current
        // then original = (pseudo - current) % 4
        let original_grade = (pseudo_grade + 4 - current_grade) % 4;

        // now compute forward rotation from current back to original
        // we want the minimal positive rotation
        let forward_rotation = if original_grade >= current_grade {
            // simple forward rotation within same cycle
            original_grade - current_grade
        } else {
            // need to go forward through a full cycle
            (4 - current_grade) + original_grade
        };

        Angle::new(forward_rotation as f64, 2.0)
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
                // values are always finite in valid Angles, so unwrap is safe
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

        let sin_result = angle.sin();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // sin(0 + 0.0) = sin(0) = 0
        assert!((sin_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_sin_with_1003_blade() {
        let angle = Angle::new(1003.0, 2.0); // 1003*(π/2)

        let sin_result = angle.sin();

        // 1003 % 4 = 3, so grade is 3 (trivector)
        // sin(3π/2 + 0.0) = sin(3π/2) = -1
        assert!((sin_result - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_of_1000_blade() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)

        let cos_result = angle.cos();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // cos(0 + 0.0) = cos(0) = 1
        assert!((cos_result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_with_1001_blade() {
        let angle = Angle::new(1001.0, 2.0); // 1001*(π/2)

        let cos_result = angle.cos();

        // 1001 % 4 = 1, so grade is 1 (vector)
        // cos(π/2 + 0.0) = cos(π/2) = 0
        assert!((cos_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_grade_offset_for_all_grades() {
        let angle0 = Angle::new(1000.0, 2.0); // grade 0 (scalar)
        let angle1 = Angle::new(1001.0, 2.0); // grade 1 (vector)
        let angle2 = Angle::new(1002.0, 2.0); // grade 2 (bivector)
        let angle3 = Angle::new(1003.0, 2.0); // grade 3 (trivector)

        assert!((angle0.grade_offset() - 0.0).abs() < EPSILON); // 0
        assert!((angle1.grade_offset() - PI / 2.0).abs() < EPSILON); // π/2
        assert!((angle2.grade_offset() - PI).abs() < EPSILON); // π
        assert!((angle3.grade_offset() - 3.0 * PI / 2.0).abs() < EPSILON); // 3π/2
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

        let tan_result = angle.tan();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // tan(0 + 0.0) = tan(0) = 0
        assert!((tan_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_tan_with_1002_blade() {
        let angle = Angle::new(1002.0, 2.0); // 1002*(π/2)

        let tan_result = angle.tan();

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
        assert!((positive_angle.sin() + negative_angle.sin()).abs() < EPSILON);

        // but same cos values (preserving geometric relationships)
        assert!((positive_angle.cos() - negative_angle.cos()).abs() < EPSILON);

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
    fn it_computes_dual_rotation_for_blade() {
        // test dual rotations for 2D (pseudoscalar blade 2)
        let e1 = Angle::new(0.0, 1.0); // blade 0
        let rotation_e1 = e1.dual_rotation_for_blade(2);
        assert_eq!(rotation_e1.blade(), 2); // rotates by π (grade 0 to grade 2)
        assert!(rotation_e1.value().abs() < EPSILON);

        let e2 = Angle::new(1.0, 2.0); // blade 1
        let rotation_e2 = e2.dual_rotation_for_blade(2);
        assert_eq!(rotation_e2.blade(), 0); // no rotation (grade 1 stays at grade 1)
        assert!(rotation_e2.value().abs() < EPSILON);

        let bivector = Angle::new(2.0, 2.0); // blade 2
        let rotation_biv = bivector.dual_rotation_for_blade(2);
        // bivector to scalar: grade 2 to grade 0
        // target = (2 - 2) % 4 = 0, rotation = (0 - 2 + 4) % 4 = 2
        assert_eq!(rotation_biv.blade() % 4, 2); // π rotation
        assert!(rotation_biv.value().abs() < EPSILON);

        // test 3D (pseudoscalar blade 3)
        let e1_3d = Angle::new(0.0, 1.0); // blade 0
        let rotation_3d = e1_3d.dual_rotation_for_blade(3);
        // in 3D: e1 → e2∧e3 (blade 0 → blade 3)
        assert_eq!(rotation_3d.blade(), 3);

        // test higher dimensions - verify full transformation
        let e1_10d = Angle::new(0.0, 1.0); // blade 0 (grade 0)
        let rotation_10d = e1_10d.dual_rotation_for_blade(10); // pseudoscalar blade 10

        // apply the rotation to get the dual
        let dual_e1_10d = e1_10d + rotation_10d;

        // in 10D with pseudoscalar grade 2, e1 (grade 0) maps to grade 2
        assert_eq!(dual_e1_10d.blade(), 2); // blade 0 + 2 = blade 2
        assert_eq!(dual_e1_10d.grade(), 2); // grade 2 (bivector)

        // test with a vector in 10D
        let e3_10d = Angle::new(5.0, 2.0); // blade 5 (grade 1)
        let rotation_e3 = e3_10d.dual_rotation_for_blade(10);
        let dual_e3_10d = e3_10d + rotation_e3;

        // grade 1 with pseudoscalar grade 2 maps to grade 1
        assert_eq!(dual_e3_10d.blade(), 5); // stays at blade 5
        assert_eq!(dual_e3_10d.grade(), 1); // stays at grade 1
    }

    #[test]
    fn it_computes_undual_rotation_for_blade() {
        // test undual rotation computation
        // note: undual in geonum_mod.rs directly computes original blade
        // this tests the angle method which computes forward rotation

        // 2D case: after dual, compute rotation to return
        let after_dual = Angle::new(3.0, 2.0); // blade 3 (grade 3)
        let pseudo_2d = 4; // blade 4 (grade 0 pseudoscalar)

        // undual rotation to get back to grade 1
        let undual_rot = after_dual.undual_rotation_for_blade(pseudo_2d);

        // from grade 3 to grade 1 requires forward rotation
        // original grade = (0 - 3) % 4 = 1
        // forward rotation = (4 - 3) + 1 = 2
        assert_eq!(undual_rot.blade(), 2);
        assert!(undual_rot.value().abs() < EPSILON);

        // test different grades
        let test_cases = vec![
            (0, 0), // grade 0 stays at grade 0
            (1, 2), // grade 1 came from grade 3, needs 2 forward to reach it
            (2, 0), // grade 2 came from grade 2, needs 0 forward (stays same)
            (3, 2), // grade 3 came from grade 1, needs 2 forward to reach it
        ];

        for (start_grade, expected_rotation) in test_cases {
            let angle = Angle::new(start_grade as f64, 2.0);
            let rotation = angle.undual_rotation_for_blade(0); // grade 0 pseudoscalar
            assert_eq!(rotation.blade(), expected_rotation);
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
}
