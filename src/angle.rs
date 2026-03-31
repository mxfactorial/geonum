use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

/// Angle struct: blade + projection ratio
///
/// blade counts which π/2 segment. t encodes position within it
/// as the projection ratio between adjacent π/2 blades:
///   t = opp / (hyp + adj), t ∈ [0, 1)
///
/// cos and sin recover rationally from t — no trig:
///   cos = (1 - t²) / (1 + t²)
///   sin = 2t / (1 + t²)
#[derive(Debug, Clone, Copy)]
pub struct Angle {
    /// projection ratio between blades, tan(θ/2) ∈ [0, 1)
    t: f64,
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
    /// angle struct with blade count and remainder
    ///
    /// # examples
    /// ```
    /// use geonum::Angle;
    /// use std::f64::consts::PI;
    ///
    /// let angle = Angle::new(3.0, 4.0);  // 3 * π/4 = 135 degrees
    /// assert_eq!(angle.blade(), 1);      // one π/2 rotation
    /// assert!((angle.rem() - PI / 4.0).abs() < 1e-10); // π/4 remainder
    /// ```
    pub fn new(pi_radians: f64, divisor: f64) -> Self {
        let quarter_pi = PI / 2.0;

        // exact quarter-turns: t is 0, blade carries everything
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
                t: 0.0,
                blade: normalized_quarters,
            };
        }

        // general case
        let total_angle = pi_radians * PI / divisor;

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
        let rem = normalized_total % quarter_pi;

        // handle boundary precision
        const EPSILON: f64 = 1e-10;
        if (rem - quarter_pi).abs() < EPSILON {
            return Self {
                blade: blade + 1,
                t: 0.0,
            };
        }
        if rem.abs() < EPSILON {
            return Self { blade, t: 0.0 };
        }

        // convert remainder to projection ratio — one tan() call
        let t = (rem / 2.0).tan();

        if t >= 1.0 - EPSILON {
            return Self {
                blade: blade + 1,
                t: 0.0,
            };
        }

        Self { blade, t }
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
        let base = Angle::new(pi_radians, divisor);
        Self {
            blade: base.blade + added_blade,
            t: base.t,
        }
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
        const EPSILON: f64 = 1e-10;
        let ax = x.abs();
        let ay = y.abs();

        if ax < EPSILON && ay < EPSILON {
            return Self { blade: 0, t: 0.0 };
        }

        // axis-aligned: blade from sign, t = 0
        if ax < EPSILON {
            return Self {
                blade: if y > 0.0 { 1 } else { 3 },
                t: 0.0,
            };
        }
        if ay < EPSILON {
            return Self {
                blade: if x > 0.0 { 0 } else { 2 },
                t: 0.0,
            };
        }

        // blade from sign pattern
        let blade = match (x > 0.0, y > 0.0) {
            (true, true) => 0_usize,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };

        // t = opp / (hyp + adj) — projection ratio between adjacent blades
        let hyp = (ax * ax + ay * ay).sqrt();
        let (adj, opp) = match blade {
            0 | 2 => (ax, ay),
            1 | 3 => (ay, ax),
            _ => unreachable!(),
        };

        let t = opp / (hyp + adj);

        if t >= 1.0 - EPSILON {
            return Self {
                blade: blade + 1,
                t: 0.0,
            };
        }

        Self { blade, t }
    }

    /// direct construction from blade and projection ratio
    pub fn from_parts(blade: usize, t: f64) -> Self {
        const EPSILON: f64 = 1e-10;
        if t >= 1.0 - EPSILON {
            return Self {
                blade: blade + 1,
                t: 0.0,
            };
        }
        Self { blade, t }
    }

    /// tests if two angles are within floating point tolerance
    pub fn near(&self, other: &Angle) -> bool {
        self.blade == other.blade && (self.t - other.t).abs() < 1e-10
    }

    /// tests if this angle's grade_angle is within tolerance of a scalar
    pub fn near_rad(&self, radians: f64) -> bool {
        (self.grade_angle() - radians).abs() < 1e-10
    }

    /// tests if this angle's remainder is within tolerance of a scalar
    pub fn near_rem(&self, radians: f64) -> bool {
        (self.rem() - radians).abs() < 1e-10
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

    /// returns the angle remainder in radians within [0, π/2)
    /// derived from t for backward compat
    pub fn rem(&self) -> f64 {
        2.0 * self.t.atan()
    }

    /// returns the projection ratio value
    pub fn t(&self) -> f64 {
        self.t
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
        self.blade % 4
    }

    /// tests if this angle represents a scalar (blade = 0)
    pub fn is_scalar(&self) -> bool {
        self.grade() == 0
    }

    /// tests if this angle represents a vector (blade = 1)  
    pub fn is_vector(&self) -> bool {
        self.grade() == 1
    }

    /// tests if this angle represents a bivector (blade = 2)
    pub fn is_bivector(&self) -> bool {
        self.grade() == 2
    }

    /// tests if this angle represents a trivector (blade = 3)
    pub fn is_trivector(&self) -> bool {
        self.grade() == 3
    }

    /// returns this angle with blade count reset to base for its grade
    ///
    /// blade accumulation is geometrically primitive - operations like reflection
    /// fundamentally work through blade arithmetic (2 + 2 = 4 blades for double
    /// reflection). however, control loops and iterative algorithms may need
    /// geometric consistency without unbounded blade growth
    ///
    /// this method preserves the angles grade (blade % 4) while resetting the
    /// blade count to its minimum for that grade:
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
            t: self.t,           // preserve projection ratio
        }
    }

    /// addition generates the blade lattice as a side effect of overflow
    ///
    /// T = (t1 + t2) / (1 - t1·t2)
    /// denominator always positive for t1, t2 ∈ [0, 1)
    ///
    /// T < 1:  no crossing, result = T
    /// T = 1:  exact boundary, blade += 1, t = 0
    /// T > 1:  crossed, blade += 1, t = (T-1)/(T+1) — thats Q
    /// T > 1 twice: blade += 2 — thats D
    ///
    /// the four-fold grade structure falls out of this overflow arithmetic.
    /// blade isnt defined then used by addition.
    /// addition produces blade as the discrete part of its result
    fn geometric_add(&self, other: &Self) -> Self {
        const EPSILON: f64 = 1e-10;
        let total_blade = self.blade + other.blade;

        // both zero: pure blade addition, t unchanged
        if self.t == 0.0 && other.t == 0.0 {
            return Self {
                blade: total_blade,
                t: 0.0,
            };
        }

        // one zero: result is the other t
        if self.t == 0.0 {
            return Self {
                blade: total_blade,
                t: other.t,
            };
        }
        if other.t == 0.0 {
            return Self {
                blade: total_blade,
                t: self.t,
            };
        }

        // tangent sum on projection ratios
        let n = self.t + other.t;
        let d = 1.0 - self.t * other.t;
        // d > 0 always for t1, t2 ∈ [0, 1)

        let sum = n / d;

        if (sum - 1.0).abs() < EPSILON {
            // exact π/2 boundary
            Self {
                blade: total_blade + 1,
                t: 0.0,
            }
        } else if sum < 1.0 {
            // no crossing
            Self {
                blade: total_blade,
                t: sum,
            }
        } else {
            // crossed — rational correction
            let corrected = (sum - 1.0) / (sum + 1.0);
            if corrected >= 1.0 - EPSILON {
                Self {
                    blade: total_blade + 2,
                    t: 0.0,
                }
            } else {
                Self {
                    blade: total_blade + 1,
                    t: corrected,
                }
            }
        }
    }

    /// tangent difference on projection ratios with rational borrow
    ///
    /// R = (t1 - t2) / (1 + t1·t2)
    /// denominator always positive
    ///
    /// R >= 0: no borrow, result = R
    /// R < 0:  borrow blade, complement = (1 - |R|) / (1 + |R|)
    fn geometric_sub(&self, other: &Self) -> Self {
        const EPSILON: f64 = 1e-10;
        let blade_diff = self.blade as i64 - other.blade as i64;

        // equal t: pure blade difference
        if (self.t - other.t).abs() < EPSILON {
            return Self {
                blade: normalize_blade(blade_diff),
                t: 0.0,
            };
        }

        // other t zero: keep self t
        if other.t == 0.0 {
            return Self {
                blade: normalize_blade(blade_diff),
                t: self.t,
            };
        }

        // self t zero: borrow
        if self.t == 0.0 {
            let complement = (1.0 - other.t) / (1.0 + other.t);
            return Self {
                blade: normalize_blade(blade_diff - 1),
                t: complement,
            };
        }

        // tangent difference on projection ratios
        let n = self.t - other.t;
        let d = 1.0 + self.t * other.t; // always positive
        let r = n / d;

        if r >= -EPSILON {
            Self {
                blade: normalize_blade(blade_diff),
                t: r.max(0.0),
            }
        } else {
            // borrow: complement = (1 - |r|) / (1 + |r|)
            let abs_r = r.abs();
            let complement = (1.0 - abs_r) / (1.0 + abs_r);
            if complement >= 1.0 - EPSILON {
                Self {
                    blade: normalize_blade(blade_diff),
                    t: 0.0,
                }
            } else {
                Self {
                    blade: normalize_blade(blade_diff - 1),
                    t: complement,
                }
            }
        }
    }

    /// tests if this angle is opposite to another angle
    ///
    /// two angles are opposite if they differ by π (blade difference of 2)
    /// and have the same remainder within their π/2 segment
    ///
    /// # arguments
    /// * `other` - the angle to compare with
    ///
    /// # returns
    /// true if the angles are opposites (π apart)
    pub fn is_opposite(&self, other: &Angle) -> bool {
        let blade_diff = (self.blade as i32 - other.blade as i32).abs();
        let t_match = (self.t - other.t).abs() < 1e-15;
        blade_diff == 2 && t_match
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
        Angle {
            blade: self.blade + 2,
            t: self.t,
        }
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
        Angle {
            blade: self.blade + 2,
            t: self.t,
        }
    }

    /// returns the grade-based angle representation in radians within [0, 2π)
    ///
    /// computes the geometric angle by mapping the 4-cycle grade pattern to quarter-turns:
    /// - grade 0 (scalar): 0 radians
    /// - grade 1 (vector): π/2 radians
    /// - grade 2 (bivector): π radians
    /// - grade 3 (trivector): 3π/2 radians
    ///
    /// then adds the fractional angle remainder within the current π/2 segment
    ///
    /// this exposes the fundamental geometric structure where each grade represents
    /// a π/2 rotation from the previous grade, creating the 4-fold symmetry that
    /// underlies geometric algebra's grade behavior patterns
    ///
    /// useful for interfacing with external code expecting standard angles in [0, 2π)
    /// such as trigonometric functions, orbital mechanics, or visualization systems
    ///
    /// # returns
    /// angle in radians as f64 within [0, 2π) representing the grade-angle mapping
    pub fn grade_angle(&self) -> f64 {
        self.grade() as f64 * PI / 2.0 + 2.0 * self.t.atan()
    }

    /// negates this angle by adding π rotation (2 blades)
    ///
    /// negation is forward rotation by 180 degrees, not backwards motion
    /// this fundamental operation appears throughout geometry as sign flips,
    /// vector opposites, and complex conjugation
    pub fn negate(&self) -> Angle {
        Angle {
            blade: self.blade + 2,
            t: self.t,
        }
    }

    /// cos and sin of full angle — rational in t, no sqrt
    ///
    /// cos(rem) = (1 - t²) / (1 + t²)
    /// sin(rem) = 2t / (1 + t²)
    /// blade applies sign and axis swap
    pub fn cos_sin(&self) -> (f64, f64) {
        let t2 = self.t * self.t;
        let denom = 1.0 + t2;
        let cos_rem = (1.0 - t2) / denom;
        let sin_rem = 2.0 * self.t / denom;

        match self.grade() {
            0 => (cos_rem, sin_rem),
            1 => (-sin_rem, cos_rem),
            2 => (-cos_rem, -sin_rem),
            3 => (sin_rem, -cos_rem),
            _ => unreachable!(),
        }
    }

    /// projects this angle onto another angle direction
    /// returns the cosine of the angle difference — rational via cos_sin
    pub fn project(&self, onto: Angle) -> f64 {
        let diff = onto - *self;
        let (cos_val, _) = diff.cos_sin();
        cos_val
    }
}

/// normalize negative blade to positive by adding full rotations
fn normalize_blade(blade: i64) -> usize {
    if blade < 0 {
        let four_rotations = ((-blade + 3) / 4) * 4;
        (blade + four_rotations) as usize
    } else {
        blade as usize
    }
}

impl PartialEq for Angle {
    fn eq(&self, other: &Self) -> bool {
        // exact blade comparison
        if self.blade != other.blade {
            return false;
        }

        // avoid floating point buggery with exact cases
        let t_diff = (self.t - other.t).abs();
        if t_diff < 1e-15 {
            return true; // exact match
        }

        // for non-exact cases, use standard floating point comparison
        self.t == other.t
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

impl Mul<f64> for Angle {
    type Output = Angle;

    fn mul(self, scalar: f64) -> Angle {
        // scale blade count and remainder separately
        // avoids converting large blade to radians (blade * π/2 → huge float)
        let scaled_blade = self.blade as f64 * scalar;
        let blade_whole = scaled_blade.floor();
        let blade_frac_rem = (scaled_blade - blade_whole) * (PI / 2.0);

        let scaled_rem = self.rem() * scalar + blade_frac_rem;

        // remainder overflow adjusts blade
        let quarter_pi = PI / 2.0;
        let extra_blades = (scaled_rem / quarter_pi).floor();
        let final_rem = scaled_rem - extra_blades * quarter_pi;

        let total_blade = blade_whole + extra_blades;
        let normalized_blade = if total_blade < 0.0 {
            let full = ((-total_blade + 3.0) / 4.0).ceil() * 4.0;
            (total_blade + full) as usize
        } else {
            total_blade as usize
        };

        if final_rem.abs() < 1e-10 {
            Angle {
                blade: normalized_blade,
                t: 0.0,
            }
        } else {
            Angle {
                blade: normalized_blade,
                t: (final_rem / 2.0).tan(),
            }
        }
    }
}

impl Mul<f64> for &Angle {
    type Output = Angle;

    fn mul(self, scalar: f64) -> Angle {
        (*self) * scalar
    }
}

impl Div<f64> for Angle {
    type Output = Angle;

    fn div(self, divisor: f64) -> Angle {
        self * (1.0 / divisor)
    }
}

impl Div<f64> for &Angle {
    type Output = Angle;

    fn div(self, divisor: f64) -> Angle {
        *self * (1.0 / divisor)
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
                // t is monotonically increasing in [0, 1)
                self.t.partial_cmp(&other.t).unwrap()
            }
            other => other,
        }
    }
}

impl std::fmt::Display for Angle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Angle(blade: {}, t: {:.4})", self.blade, self.t)
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
        assert!((sum.rem() - (7.0 * PI / 24.0)).abs() < EPSILON);
    }

    #[test]
    fn it_sums_greater_than_a_quarter_turn() {
        let angle1 = Angle::new(1.0, 3.0); // π/3
        let angle2 = Angle::new(1.0, 4.0); // π/4

        let sum = angle1 + angle2; // π/3 + π/4 = 7π/12 > π/2

        assert_eq!(sum.blade(), 1); // crosses π/2 boundary, increments blade
        assert!((sum.rem() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON);
    }

    #[test]
    fn it_sums_rotations_to_multiple_blades() {
        let angle1 = Angle::new(3.0, 4.0); // 3π/4, blade 1, rem π/4
        let angle2 = Angle::new(5.0, 4.0); // 5π/4, blade 2, rem π/4

        let sum = angle1 + angle2; // blade sum: (1+2)%4=3, rem sum: π/4+π/4=π/2
                                   // π/2 crosses boundary: final_blade=(3+1)%4=0

        assert_eq!(sum.blade(), 4); // preserves full blade count: 1+2+1=4
        assert!((sum.rem()).abs() < EPSILON); // π/2 boundary crossing leaves no remainder
    }

    #[test]
    fn it_constructs_blade_0_from_large_angles() {
        let angle = Angle::new(4.0, 2.0); // 4*(π/2) = 2π

        assert_eq!(angle.blade(), 4); // preserves original blade count
        assert_eq!(angle.grade(), 0); // 4 % 4 = 0 (scalar grade)
        assert!((angle.rem()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_1_from_large_angles() {
        let angle = Angle::new(5.0, 2.0); // 5*(π/2)

        assert_eq!(angle.blade(), 5); // preserves original blade count
        assert_eq!(angle.grade(), 1); // 5 % 4 = 1 (vector grade)
        assert!((angle.rem()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_2_from_large_angles() {
        let angle = Angle::new(6.0, 2.0); // 6*(π/2)

        assert_eq!(angle.blade(), 6); // preserves original blade count
        assert_eq!(angle.grade(), 2); // 6 % 4 = 2 (bivector grade)
        assert!((angle.rem()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
    }

    #[test]
    fn it_constructs_blade_3_from_large_angles() {
        let angle = Angle::new(7.0, 2.0); // 7*(π/2)

        assert_eq!(angle.blade(), 7); // preserves original blade count
        assert_eq!(angle.grade(), 3); // 7 % 4 = 3 (trivector grade)
        assert!((angle.rem()).abs() < EPSILON); // exact multiple of π/2 leaves no remainder
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
        assert!((diff.rem() - PI / 6.0).abs() < EPSILON); // remainder is π/6
    }

    #[test]
    fn it_subtracts_pi_over_3_from_4pi_over_3() {
        let angle1 = Angle::new(4.0, 3.0); // 4π/3
        let angle2 = Angle::new(1.0, 3.0); // π/3

        let diff = angle1 - angle2; // 4π/3 - π/3 = π

        assert_eq!(diff.blade(), 2); // π = 2 * π/2, so blade = 2
        assert!((diff.rem()).abs() < EPSILON); // exact multiple of π/2 has no remainder
    }

    #[test]
    fn it_subtracts_11pi_over_6_from_pi_over_6() {
        let angle1 = Angle::new(1.0, 6.0); // π/6
        let angle2 = Angle::new(11.0, 6.0); // 11π/6

        let diff = angle1 - angle2; // π/6 - 11π/6 = -10π/6 = -5π/3

        // negative result should normalize to positive angle
        // -5π/3 = -5π/3 + 2π = π/3
        assert_eq!(diff.blade(), 0); // π/3 is less than π/2, no blade increment
        assert!((diff.rem() - PI / 3.0).abs() < EPSILON); // remainder is π/3
    }

    #[test]
    fn it_multiplies_angles_as_addition() {
        let angle1 = Angle::new(1.0, 8.0); // π/8
        let angle2 = Angle::new(1.0, 6.0); // π/6

        let product = angle1 * angle2; // π/8 * π/6 = π/8 + π/6 = 7π/24

        assert_eq!(product.blade(), 0); // 7π/24 < π/2, no blade increment
        assert!((product.rem() - (7.0 * PI / 24.0)).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_sin_of_1000_blade() {
        let angle = Angle::new(1000.0, 2.0); // 1000*(π/2)
        let (cos_result, sin_result) = angle.cos_sin();

        // 1000 % 4 = 0, so grade is 0 (scalar)
        // cos(0) = 1, sin(0) = 0
        assert!((cos_result - 1.0).abs() < EPSILON);
        assert!((sin_result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_sin_with_1001_blade() {
        let angle = Angle::new(1001.0, 2.0); // 1001*(π/2)
        let (cos_result, sin_result) = angle.cos_sin();

        // 1001 % 4 = 1, so grade is 1 (vector)
        // cos(π/2) = 0, sin(π/2) = 1
        assert!((cos_result - 0.0).abs() < EPSILON);
        assert!((sin_result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_sin_with_1003_blade() {
        let angle = Angle::new(1003.0, 2.0); // 1003*(π/2)
        let (cos_result, sin_result) = angle.cos_sin();

        // 1003 % 4 = 3, so grade is 3 (trivector)
        // cos(3π/2) = 0, sin(3π/2) = -1
        assert!((cos_result - 0.0).abs() < EPSILON);
        assert!((sin_result - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn it_creates_angle_with_additional_blade() {
        // test basic functionality
        let angle = Angle::new_with_blade(2, 1.0, 4.0); // 2 extra blades + π/4
        assert_eq!(angle.blade(), 2); // π/4 gives blade 0, + 2 = blade 2
        assert!((angle.rem() - PI / 4.0).abs() < EPSILON);

        // test with angle that causes boundary crossing
        let angle2 = Angle::new_with_blade(1, 3.0, 4.0); // 1 extra blade + 3π/4
        assert_eq!(angle2.blade(), 2); // 3π/4 gives blade 1, + 1 = blade 2
        assert!((angle2.rem() - PI / 4.0).abs() < EPSILON); // 3π/4 - π/2 = π/4

        // test with zero additional blades
        let angle3 = Angle::new_with_blade(0, 1.0, 2.0); // 0 extra blades + π/2
        assert_eq!(angle3.blade(), 1); // π/2 gives blade 1, + 0 = blade 1
        assert!(angle3.rem().abs() < EPSILON); // exact π/2 leaves no remainder

        // test with large additional blade count
        let angle4 = Angle::new_with_blade(10, 0.0, 1.0); // 10 extra blades + 0
        assert_eq!(angle4.blade(), 10); // 0 gives blade 0, + 10 = blade 10
        assert!(angle4.rem().abs() < EPSILON);
    }

    #[test]
    fn it_computes_cos_sin_at_lattice_points() {
        // at lattice points (t = 0), cos_sin reduces to blade lookup
        // blade 0: (1, 0), blade 1: (0, 1), blade 2: (-1, 0), blade 3: (0, -1)
        let lattice = [
            (0.0, 1.0, (1.0, 0.0)),
            (1.0, 2.0, (0.0, 1.0)),
            (1.0, 1.0, (-1.0, 0.0)),
            (3.0, 2.0, (0.0, -1.0)),
        ];

        for (pi_r, div, (expected_cos, expected_sin)) in lattice {
            let angle = Angle::new(pi_r, div);
            let (c, s) = angle.cos_sin();
            assert!((c - expected_cos).abs() < EPSILON);
            assert!((s - expected_sin).abs() < EPSILON);
        }
    }

    #[test]
    fn it_compares_angles_for_equality() {
        let angle1 = Angle::new(1000.0, 2.0); // 1000*(π/2)
        let angle2 = Angle::new(1000.0, 2.0); // same
        let angle3 = Angle::new(1001.0, 2.0); // different blade
        let angle4 = Angle::new(1.0, 4.0); // different remainder

        // exact equality
        assert_eq!(angle1, angle2);

        // different blades
        assert_ne!(angle1, angle3);

        // different rems
        assert_ne!(angle1, angle4);
    }

    #[test]
    fn it_detects_opposite_angles() {
        // test that angles differing by π (blade difference of 2) are opposites

        // case 1: blade 0 and blade 2 with same remainder
        let scalar = Angle::new(0.0, 1.0); // blade 0, rem 0
        let bivector = Angle::new(1.0, 1.0); // blade 2, rem 0
        assert!(
            scalar.is_opposite(&bivector),
            "scalar and bivector are opposites"
        );
        assert!(
            bivector.is_opposite(&scalar),
            "opposite detection is symmetric"
        );

        // case 2: blade 1 and blade 3 with same remainder
        let vector = Angle::new(1.0, 2.0); // blade 1, rem 0
        let trivector = Angle::new(3.0, 2.0); // blade 3, rem 0
        assert!(
            vector.is_opposite(&trivector),
            "vector and trivector are opposites"
        );

        // case 3: different rems - not opposites
        let angle1 = Angle::new(1.0, 4.0); // blade 0, rem π/4
        let angle2 = Angle::new(1.0, 1.0); // blade 2, rem 0
        assert!(
            !angle1.is_opposite(&angle2),
            "different rems are not opposites"
        );

        // case 4: blade difference not 2 - not opposites
        let angle3 = Angle::new(0.0, 1.0); // blade 0
        let angle4 = Angle::new(1.0, 2.0); // blade 1
        assert!(
            !angle3.is_opposite(&angle4),
            "blade difference of 1 is not opposite"
        );

        // case 5: same angle - not opposite
        let angle5 = Angle::new(1.0, 3.0); // blade 0, rem π/3
        assert!(!angle5.is_opposite(&angle5), "same angle is not opposite");

        // case 6: high blade counts
        let high1 = Angle::new(10.0, 2.0); // blade 10, rem 0
        let high2 = Angle::new(12.0, 2.0); // blade 12, rem 0
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
        assert!(angle_0.rem().abs() < EPSILON);

        let angle_90 = Angle::new_from_cartesian(0.0, 1.0); // positive y-axis
        assert_eq!(angle_90.blade(), 1);
        assert!(angle_90.rem().abs() < EPSILON);

        let angle_180 = Angle::new_from_cartesian(-1.0, 0.0); // negative x-axis
        assert_eq!(angle_180.blade(), 2);
        assert!(angle_180.rem().abs() < EPSILON);

        let angle_270 = Angle::new_from_cartesian(0.0, -1.0); // negative y-axis
        assert_eq!(angle_270.blade(), 3);
        assert!(angle_270.rem().abs() < EPSILON);

        // test 45 degree angle
        let angle_45 = Angle::new_from_cartesian(1.0, 1.0);
        assert_eq!(angle_45.blade(), 0);
        assert!((angle_45.rem() - PI / 4.0).abs() < EPSILON);

        // test 3-4-5 triangle (arctan(3/4))
        let angle_triangle = Angle::new_from_cartesian(4.0, 3.0);
        let expected_radians = (3.0_f64).atan2(4.0);
        let expected_angle = Angle::new(expected_radians, PI);
        assert_eq!(angle_triangle.blade(), expected_angle.blade());
        assert!((angle_triangle.rem() - expected_angle.rem()).abs() < EPSILON);
    }

    #[test]
    fn it_preserves_blade_with_signed_angles() {
        // reminder: before creating elaborate "grade-preserving" methods,
        // remember that angles can be negative and this solves most problems naturally

        let positive_angle = Angle::new(1.0, 3.0); // π/3
        let negative_angle = Angle::new(-1.0, 3.0); // -π/3

        // negative angles give opposite sin values (for anti-commutativity)
        assert!(
            (positive_angle.grade_angle().sin() + negative_angle.grade_angle().sin()).abs()
                < EPSILON
        );

        // but same cos values (preserving geometric relationships)
        assert!(
            (positive_angle.grade_angle().cos() - negative_angle.grade_angle().cos()).abs()
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
        assert_eq!(result.rem(), 0.0);
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
        assert_eq!(result.rem(), 0.0);
    }

    #[test]
    fn it_creates_negative_pi_over_2() {
        // test what Angle::new(-1.0, 2.0) actually creates
        let neg_half_pi = Angle::new(-1.0, 2.0);
        println!(
            "Angle::new(-1.0, 2.0) gives blade={}, rem={}",
            neg_half_pi.blade(),
            neg_half_pi.rem()
        );

        // test adding it to zero
        let zero = Angle::new(0.0, 1.0);
        let result = zero + neg_half_pi;
        println!(
            "0 + Angle::new(-1.0, 2.0) gives blade={}, rem={}",
            result.blade(),
            result.rem()
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
        assert_eq!(scalar_double.rem(), scalar.rem());

        let vector_double = vector.dual().dual();
        assert_eq!(vector_double.grade(), vector.grade());
        assert_eq!(vector_double.rem(), vector.rem());

        let bivector_double = bivector.dual().dual();
        assert_eq!(bivector_double.grade(), bivector.grade());
        assert_eq!(bivector_double.rem(), bivector.rem());

        let trivector_double = trivector.dual().dual();
        assert_eq!(trivector_double.grade(), trivector.grade());
        assert_eq!(trivector_double.rem(), trivector.rem());

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
            assert_eq!(dual.rem(), undual.rem());

            // applying dual twice returns to original grade
            let double_dual = angle.dual().dual();
            assert_eq!(double_dual.grade(), angle.grade());
            assert_eq!(double_dual.rem(), angle.rem());
        }
    }

    #[test]
    fn it_orders_angles_by_blade_then_rem() {
        // test ordering by blade first
        let blade0 = Angle::new(1.0, 3.0); // PI/3 (blade 0)
        let blade1 = Angle::new(3.0, 2.0); // 3PI/2 (blade 3)
        let blade2 = Angle::new(5.0, 2.0); // 5PI/2 (blade 5)

        assert!(blade0 < blade1);
        assert!(blade1 < blade2);
        assert!(blade0 < blade2);

        // test ordering by remainder when blades are equal
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
        let eq1 = Angle::new(2.0, 3.0); // blade 0, rem 2PI/3
        let eq2 = Angle::new(2.0, 3.0); // same
        assert!(eq1 <= eq2);
        assert!(eq1 >= eq2);
        assert!(eq1 >= eq2);
        assert!(eq1 <= eq2);

        // test that blade takes precedence over remainder
        let small_blade_big_rem = Angle::new(0.8, 1.0); // 0.8PI (blade 1, rem 0.3PI)
        let big_blade_small_rem = Angle::new(1.1, 1.0); // 1.1PI (blade 2, rem 0.1PI)
                                                        // even though 0.8PI > 1.1PI in terms of raw angle remainder within their blades,
                                                        // blade 1 < blade 2, so the first angle is less than the second
        assert!(small_blade_big_rem < big_blade_small_rem);
    }

    #[test]
    fn it_computes_grade_angle() {
        // test basic cases within first rotation
        let angle1 = Angle::new(0.5, 4.0); // π/8, blade 0
        assert!((angle1.grade_angle() - PI / 8.0).abs() < EPSILON);

        let angle2 = Angle::new(1.0, 2.0); // π/2, blade 1
        assert!((angle2.grade_angle() - PI / 2.0).abs() < EPSILON);

        let angle3 = Angle::new(3.0, 2.0); // 3π/2, blade 3
        assert!((angle3.grade_angle() - 3.0 * PI / 2.0).abs() < EPSILON);

        // test angles with blade count > 3
        let angle4 = Angle::new(5.0, 2.0); // 5π/2, blade 5 -> blade 1
        assert!((angle4.grade_angle() - PI / 2.0).abs() < EPSILON);

        let angle5 = Angle::new(8.0, 2.0); // 8π/2 = 4π, blade 8 -> blade 0
        assert!(angle5.grade_angle().abs() < EPSILON);

        let angle6 = Angle::new(10.0, 2.0); // 10π/2 = 5π, blade 10 -> blade 2
        assert!((angle6.grade_angle() - PI).abs() < EPSILON);

        // test with non-zero remainder component
        let angle7 = Angle::new(9.5, 2.0); // 9.5π/2, blade 9, rem π/4
                                           // blade 9 % 4 = 1, so result is π/2 + π/4 = 3π/4
        assert!((angle7.grade_angle() - 3.0 * PI / 4.0).abs() < EPSILON);

        // test large blade counts
        let angle8 = Angle::new(1000.0, 2.0); // blade 1000 -> blade 0
        assert!(angle8.grade_angle().abs() < EPSILON);

        let angle9 = Angle::new(1001.0, 2.0); // blade 1001 -> blade 1
        assert!((angle9.grade_angle() - PI / 2.0).abs() < EPSILON);
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
            normalized.rem(),
            angle_high_blade.rem(),
            "angle rem preserved"
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
        assert_eq!(negated.rem(), angle.rem(), "fractional angle preserved");

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
        assert_eq!(neg_boundary.rem(), 0.0, "rem at exact boundary becomes 0");
    }

    #[test]
    fn it_handles_boundary_via_tangent_formula() {
        // boundary logic is now algebraic in the tangent sum formula
        // normalize_boundaries is eliminated
        let angle1 = Angle::new(1.0, 3.0); // π/3
        let angle2 = Angle::new(1.0, 6.0); // π/6
        let sum = angle1 + angle2; // π/3 + π/6 = π/2 → exact boundary

        assert_eq!(sum.blade(), 1);
        assert!(sum.rem().abs() < EPSILON);

        // near-boundary
        let near_pi_2 = Angle::new(0.99, 2.0); // just under π/2
        assert_eq!(near_pi_2.blade(), 0);
        assert!(near_pi_2.t() < 1.0);
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
        let small_angle = Angle::new(1.0, 8.0); // π/8, blade 0, rem π/8
        let large_angle = Angle::new_with_blade(5, 1.0, 4.0); // blade 5, π/4
        let wrapped_diff = large_angle - small_angle;
        // large_angle - small_angle = blade 5 - blade 0 = blade 5, rem π/4 - π/8 = π/8
        assert_eq!(wrapped_diff.blade(), 5);
        assert!((wrapped_diff.rem() - PI / 8.0).abs() < 1e-10);
        assert!((wrapped_diff.grade_angle() - 1.9634954084936207).abs() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // projection ratio (t) properties
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn it_constructs_lattice_points_with_t_zero() {
        let a0 = Angle::new(0.0, 1.0);
        assert_eq!(a0.blade(), 0);
        assert_eq!(a0.t(), 0.0);

        let a1 = Angle::new(1.0, 2.0);
        assert_eq!(a1.blade(), 1);
        assert_eq!(a1.t(), 0.0);

        let a2 = Angle::new(1.0, 1.0);
        assert_eq!(a2.blade(), 2);
        assert_eq!(a2.t(), 0.0);

        let a3 = Angle::new(3.0, 2.0);
        assert_eq!(a3.blade(), 3);
        assert_eq!(a3.t(), 0.0);
    }

    #[test]
    fn it_constructs_common_angles_with_exact_t() {
        // π/6: t = tan(π/12) = 2 - √3
        let a = Angle::new(1.0, 6.0);
        assert_eq!(a.blade(), 0);
        assert!((a.t() - (2.0 - 3.0_f64.sqrt())).abs() < EPSILON);

        // π/4: t = tan(π/8)
        let b = Angle::new(1.0, 4.0);
        assert_eq!(b.blade(), 0);
        assert!((b.t() - (PI / 8.0).tan()).abs() < EPSILON);

        // π/3: t = tan(π/6) = 1/√3
        let c = Angle::new(1.0, 3.0);
        assert_eq!(c.blade(), 0);
        assert!((c.t() - 1.0 / 3.0_f64.sqrt()).abs() < EPSILON);
    }

    #[test]
    fn it_bounds_t_in_zero_to_one() {
        // t = tan(rem/2) where rem ∈ [0, π/2)
        // so rem/2 ∈ [0, π/4) and t ∈ [0, 1)
        let angles = [
            (1.0, 6.0),
            (1.0, 4.0),
            (1.0, 3.0),
            (2.0, 5.0),
            (3.0, 8.0),
            (7.0, 16.0),
        ];

        for (pi_r, div) in angles {
            let a = Angle::new(pi_r, div);
            assert!(
                a.t() >= 0.0 && a.t() < 1.0,
                "t={} out of [0,1) for {}π/{}",
                a.t(),
                pi_r,
                div
            );
        }
    }

    #[test]
    fn it_projects_rationally() {
        // cos_sin uses (1-t²)/(1+t²) and 2t/(1+t²) — no sqrt
        let cases = [
            (1.0, 6.0),
            (1.0, 4.0),
            (1.0, 3.0),
            (2.0, 3.0),
            (5.0, 4.0),
            (5.0, 3.0),
        ];

        for (pi_r, div) in cases {
            let a = Angle::new(pi_r, div);
            let (cos_val, sin_val) = a.cos_sin();
            let rad = pi_r * PI / div;
            assert!(
                (cos_val - rad.cos()).abs() < EPSILON,
                "cos mismatch for {}π/{}",
                pi_r,
                div
            );
            assert!(
                (sin_val - rad.sin()).abs() < EPSILON,
                "sin mismatch for {}π/{}",
                pi_r,
                div
            );
        }
    }

    #[test]
    fn it_adds_without_crossing() {
        // π/8 + π/8 = π/4 (t < 1, no crossing)
        let a = Angle::new(1.0, 8.0);
        let sum = a + a;
        assert_eq!(sum.blade(), 0);
        let expected_t = (PI / 8.0).tan(); // tan(π/4 / 2) = tan(π/8)
        assert!((sum.t() - expected_t).abs() < EPSILON);
    }

    #[test]
    fn it_adds_at_exact_boundary() {
        // π/4 + π/4 = π/2 → exact boundary
        let a = Angle::new(1.0, 4.0);
        let sum = a + a;
        assert_eq!(sum.blade(), 1);
        assert!(sum.t() < EPSILON);
    }

    #[test]
    fn it_adds_across_boundary_with_rational_correction() {
        // π/3 + π/4 = 7π/12 → crosses π/2
        let a = Angle::new(1.0, 3.0);
        let b = Angle::new(1.0, 4.0);
        let sum = a + b;

        assert_eq!(sum.blade(), 1);
        // remainder = 7π/12 - π/2 = π/12
        // t = tan(π/24)
        let expected_t = (PI / 24.0).tan();
        assert!(
            (sum.t() - expected_t).abs() < EPSILON,
            "t after crossing: {} vs {}",
            sum.t(),
            expected_t
        );
    }

    #[test]
    fn it_subtracts_with_rational_borrow() {
        // π/3 - π/6 = π/6 (no borrow)
        let diff1 = Angle::new(1.0, 3.0) - Angle::new(1.0, 6.0);
        assert_eq!(diff1.blade(), 0);
        let expected_t = (PI / 12.0).tan(); // tan(π/6 / 2)
        assert!((diff1.t() - expected_t).abs() < EPSILON);

        // π/6 - π/3 → borrows
        let diff2 = Angle::new(1.0, 6.0) - Angle::new(1.0, 3.0);
        assert_eq!(diff2.blade(), 3); // borrowed: 0-1 → 3 (mod 4)
    }

    #[test]
    fn it_adds_with_always_positive_denominator() {
        // for t1, t2 ∈ [0, 1): t1·t2 < 1, so 1 - t1·t2 > 0
        // no sign branching needed — t1·t2 < 1 when both < 1
        let angles = [
            (1.0, 6.0),
            (1.0, 4.0),
            (1.0, 3.0),
            (2.0, 5.0),
            (3.0, 7.0),
            (3.0, 8.0),
        ];

        for &(p1, d1) in &angles {
            for &(p2, d2) in &angles {
                let a = Angle::new(p1, d1);
                let b = Angle::new(p2, d2);
                let d = 1.0 - a.t() * b.t();
                assert!(
                    d > 0.0,
                    "denominator not positive for {}π/{} + {}π/{}: d={}",
                    p1,
                    d1,
                    p2,
                    d2,
                    d
                );
            }
        }
    }

    #[test]
    fn it_dualizes_and_negates_with_blade_only() {
        let a = Angle::new(1.0, 4.0);
        assert_eq!(a.dual().t(), a.t());
        assert_eq!(a.dual().blade() - a.blade(), 2);
        assert_eq!(a.negate().t(), a.t());
        assert_eq!(a.negate().blade() - a.blade(), 2);
    }

    #[test]
    fn it_multiplies_via_tangent_sum() {
        let a = Angle::new(1.0, 4.0);
        let b = Angle::new(1.0, 6.0);
        let product = a * b;

        // π/4 + π/6 = 5π/12, t = tan(5π/24)
        let expected_t = (5.0 * PI / 24.0).tan();
        assert_eq!(product.blade(), 0);
        assert!((product.t() - expected_t).abs() < EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // near methods
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn it_detects_near_angles() {
        let a = Angle::new(1.0, 4.0); // π/4
        let b = Angle::new(1.0, 4.0); // same

        assert!(a.near(&b));
        assert!(b.near(&a));

        // different blade, same t → not near
        let c = Angle::new_with_blade(4, 1.0, 4.0); // blade 4, same t
        assert!(!a.near(&c));

        // same blade, different t → not near
        let d = Angle::new(1.0, 3.0); // π/3, blade 0 but different t
        assert!(!a.near(&d));

        // near within tolerance
        let tiny = Angle::from_parts(0, a.t() + 1e-12);
        assert!(a.near(&tiny));

        // outside tolerance
        let far = Angle::from_parts(0, a.t() + 1e-8);
        assert!(!a.near(&far));
    }

    #[test]
    fn it_compares_near_rad() {
        // grade 0: grade_angle = 0 + rem
        let a = Angle::new(1.0, 4.0); // π/4
        assert!(a.near_rad(PI / 4.0));
        assert!(!a.near_rad(PI / 3.0));

        // grade 1: grade_angle = π/2 + rem
        let b = Angle::new(3.0, 4.0); // 3π/4, blade 1
        assert!(b.near_rad(3.0 * PI / 4.0));
        assert!(!b.near_rad(PI / 4.0)); // thats the rem, not the grade_angle

        // lattice point
        let c = Angle::new(1.0, 2.0); // π/2
        assert!(c.near_rad(PI / 2.0));
    }

    #[test]
    fn it_compares_near_rem() {
        // rem is the within-quadrant remainder
        let a = Angle::new(1.0, 4.0); // π/4, blade 0, rem ≈ π/4
        assert!(a.near_rem(PI / 4.0));
        assert!(!a.near_rem(PI / 3.0));

        // blade 1 angle: grade_angle = π/2 + rem, but near_rem checks rem only
        let b = Angle::new(3.0, 4.0); // 3π/4, blade 1, rem ≈ π/4
        assert!(b.near_rem(PI / 4.0)); // rem is π/4
        assert!(!b.near_rem(3.0 * PI / 4.0)); // thats grade_angle, not rem

        // lattice: rem = 0
        let c = Angle::new(1.0, 2.0); // π/2
        assert!(c.near_rem(0.0));
    }
}
