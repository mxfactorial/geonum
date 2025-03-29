use std::f64::consts::PI;
use std::ops::{Add, Mul};

const TWO_PI: f64 = 2.0 * PI;

#[derive(Debug, Copy, Clone, PartialOrd)]
struct Angle(f64);

impl PartialEq for Angle {
    fn eq(&self, other: &Self) -> bool {
        let s = self.mod_2pi();
        let o = other.mod_2pi();
        // compare with some epsilon to handle floating-point precision
        const EPSILON: f64 = 1e-10;
        (s.0 - o.0).abs() < EPSILON || (s.0 - o.0).abs() > TWO_PI - EPSILON
    }
}

impl Angle {
    fn mod_2pi(&self) -> Self {
        Angle(self.0 % TWO_PI)
    }
}

impl Add for Angle {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let s = self.mod_2pi();
        let o = other.mod_2pi();
        Angle(s.0 + o.0)
    }
}

impl Mul for Angle {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.add(other)
    }
}

#[derive(Debug)]
struct Gn {
    length: u64,
    angle: Angle,
}

impl PartialEq for Gn {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length && self.angle == other.angle
    }
}

impl Mul for Gn {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Gn {
            length: self.length * other.length,
            angle: self.angle * other.angle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn it_wraps_angles() {
        assert_eq!(Angle(PI * 2.0), Angle(0.0));
    }

    #[test]
    fn it_adds_two_angles() {
        let left = Angle(PI / 4.0);
        let right = Angle(PI / 4.0);
        assert_eq!(left + right, Angle(PI / 2.0));
    }

    #[test]
    fn it_adds_angles_when_multiplied() {
        let left = Angle(PI / 4.0);
        let right = Angle(PI / 4.0);
        assert_eq!(left * right, Angle(PI / 2.0));
    }

    #[test]
    fn it_multiplies_two() {
        let left = Gn {
            length: 2,
            angle: Angle(PI / 4.0),
        };
        let right = Gn {
            length: 2,
            angle: Angle(PI / 4.0),
        };
        assert_eq!(
            left * right,
            Gn {
                length: 4,
                angle: Angle(PI / 2.0),
            }
        );
    }

    #[test]
    fn it_multiplies_three() {
        let first = Gn {
            length: 2,
            angle: Angle(PI / 2.0),
        };
        let second = Gn {
            length: 3,
            angle: Angle(PI / 4.0),
        };
        let third = Gn {
            length: 4,
            angle: Angle(PI / 6.0),
        };
        assert_eq!(
            first * second * third,
            Gn {
                length: 24,
                angle: Angle(PI / 2.0 + PI / 4.0 + PI / 6.0),
            }
        );
    }
}
