use crate::{angle::Angle, Geonum};

pub trait Affine {
    fn translate(&self, displacement: &Self) -> Self;
    fn shear(&self, shear_angle: Angle) -> Self;
    fn area_quadrilateral(p1: &Self, p2: &Self, p3: &Self, p4: &Self) -> f64;
}

#[cfg(feature = "affine")]
impl Affine for Geonum {
    fn translate(&self, displacement: &Geonum) -> Geonum {
        *self + *displacement // direct vector addition
    }

    fn shear(&self, shear_angle: Angle) -> Geonum {
        Geonum {
            length: self.length,
            angle: self.angle + shear_angle, // uniform angular transformation
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
    const EPSILON: f64 = 1e-10;

    #[test]
    fn it_preserves_grade_after_translation() {
        let point = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]
        let displacement = Geonum::new(3.0, 1.0, 2.0); // [3, π/2]

        let translated = point.translate(&displacement);
        assert_eq!(translated.angle.grade(), point.angle.grade()); // grade preserved
    }

    #[test]
    fn it_reverses_translation_with_inverse_displacement() {
        let point = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]
        let displacement = Geonum::new(3.0, 1.0, 2.0); // [3, π/2]
        let inverse = displacement.negate();

        let translated = point.translate(&displacement);
        let back = translated.translate(&inverse);

        assert!(point.length_diff(&back) < EPSILON);
        assert!((point.angle - back.angle).value() < EPSILON);
    }

    #[test]
    fn it_preserves_length_and_transforms_angle_after_shear() {
        let point = Geonum::new(5.0, 1.0, 3.0); // [5, π/3]
        let shear_angle = Angle::new(1.0, 6.0); // π/6
        let sheared = point.shear(shear_angle);

        assert!(point.length_diff(&sheared) < EPSILON); // length preserved
        let expected_angle = point.angle + shear_angle;
        assert!((sheared.angle - expected_angle).value() < EPSILON); // angle shifted by shear amount

        // grade changes when angle sum crosses π/2 boundary
        // π/3 + π/6 = π/2, so grade changes from 0 to 1
        assert_eq!(point.angle.grade(), 0); // original grade
        assert_eq!(sheared.angle.grade(), 1); // grade after shear
    }

    #[test]
    fn it_preserves_parallelism_after_shear() {
        let dir1 = Geonum::new(2.0, 0.0, 1.0); // [2, 0]
        let dir2 = Geonum::new(3.0, 0.0, 1.0); // [3, 0] - parallel

        let shear_angle = Angle::new(1.0, 4.0); // π/4
        let sheared1 = dir1.shear(shear_angle);
        let sheared2 = dir2.shear(shear_angle);

        // parallelism preserved - same angle relationship
        assert!((sheared1.angle - sheared2.angle).value() < EPSILON);
    }

    #[test]
    fn it_returns_12_for_4x3_rectangle_area() {
        let v1 = Geonum::new(0.0, 0.0, 1.0); // origin
        let v2 = Geonum::new(4.0, 0.0, 1.0); // (4,0)
        let v3 = Geonum::new_with_angle(5.0, Angle::new_from_cartesian(4.0, 3.0)); // (4,3)
        let v4 = Geonum::new(3.0, 1.0, 2.0); // (0,3)

        let area = Geonum::area_quadrilateral(&v1, &v2, &v3, &v4);
        assert!((area - 12.0).abs() < EPSILON); // 4×3 rectangle
    }
}
