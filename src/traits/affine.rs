use crate::Geonum;

pub trait Affine {
    fn translate(&self, displacement: &Self) -> Self;
    fn shear(&self, shear_angle: f64) -> Self;
    fn area_quadrilateral(p1: &Self, p2: &Self, p3: &Self, p4: &Self) -> f64;
}

#[cfg(feature = "affine")]
impl Affine for Geonum {
    fn translate(&self, displacement: &Geonum) -> Geonum {
        *self + *displacement // direct vector addition
    }

    fn shear(&self, shear_angle: f64) -> Geonum {
        Geonum {
            length: self.length,
            angle: self.angle + shear_angle, // uniform angular transformation
            blade: self.blade,
        }
    }

    fn area_quadrilateral(p1: &Geonum, p2: &Geonum, p3: &Geonum, p4: &Geonum) -> f64 {
        // area using wedge products - pure geometric algebra approach
        // triangulate: split quadrilateral into two triangles
        // area of triangle = |edge1 ∧ edge2| / 2

        // triangle 1: p1, p2, p3
        let edge1 = *p2 + p1.negate(); // vector from p1 to p2
        let edge2 = *p3 + p1.negate(); // vector from p1 to p3
        let triangle1_area = edge1.wedge(&edge2).length / 2.0;

        // triangle 2: p1, p3, p4
        let edge3 = *p3 + p1.negate(); // vector from p1 to p3
        let edge4 = *p4 + p1.negate(); // vector from p1 to p4
        let triangle2_area = edge3.wedge(&edge4).length / 2.0;

        triangle1_area + triangle2_area
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Geonum;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn it_preserves_grade_after_translation() {
        let point = Geonum {
            length: 5.0,
            angle: PI / 6.0,
            blade: 1,
        };
        let displacement = Geonum {
            length: 3.0,
            angle: PI / 2.0,
            blade: 1,
        };

        let translated = point.translate(&displacement);
        assert_eq!(translated.blade, point.blade); // grade preserved
    }

    #[test]
    fn it_reverses_translation_with_inverse_displacement() {
        let point = Geonum {
            length: 5.0,
            angle: PI / 6.0,
            blade: 1,
        };
        let displacement = Geonum {
            length: 3.0,
            angle: PI / 2.0,
            blade: 1,
        };
        let inverse = displacement.negate();

        let translated = point.translate(&displacement);
        let back = translated.translate(&inverse);

        assert!(point.length_diff(&back) < EPSILON);
        assert!(point.angle_distance(&back) < EPSILON);
    }

    #[test]
    fn it_preserves_length_and_grade_after_shear() {
        let point = Geonum {
            length: 5.0,
            angle: PI / 3.0,
            blade: 1,
        };
        let sheared = point.shear(PI / 6.0);

        assert!(point.length_diff(&sheared) < EPSILON); // length preserved
        assert!((sheared.angle - (point.angle + PI / 6.0)).abs() < EPSILON); // angle shifted
        assert_eq!(sheared.blade, point.blade); // grade preserved
    }

    #[test]
    fn it_preserves_parallelism_after_shear() {
        let dir1 = Geonum {
            length: 2.0,
            angle: 0.0,
            blade: 1,
        };
        let dir2 = Geonum {
            length: 3.0,
            angle: 0.0,
            blade: 1,
        }; // parallel
        let shear_angle = PI / 4.0;

        let sheared1 = dir1.shear(shear_angle);
        let sheared2 = dir2.shear(shear_angle);

        // parallelism preserved - same angle relationship
        assert!(sheared1.angle_distance(&sheared2) < EPSILON);
    }

    #[test]
    fn it_returns_12_for_4x3_rectangle_area() {
        let v1 = Geonum {
            length: 0.0,
            angle: 0.0,
            blade: 1,
        }; // origin
        let v2 = Geonum {
            length: 4.0,
            angle: 0.0,
            blade: 1,
        }; // (4,0)
        let v3 = Geonum {
            length: 5.0,
            angle: (3.0_f64).atan2(4.0),
            blade: 1,
        }; // (4,3)
        let v4 = Geonum {
            length: 3.0,
            angle: PI / 2.0,
            blade: 1,
        }; // (0,3)

        let area = Geonum::area_quadrilateral(&v1, &v2, &v3, &v4);
        assert!((area - 12.0).abs() < EPSILON); // 4×3 rectangle
    }
}
