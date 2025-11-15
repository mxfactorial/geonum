use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_point() {
    // traditional: point requires coordinates [x, y, z]
    // 2D needs 2 floats, 3D needs 3 floats, 100D needs 100 floats
    //
    // geonum: [length, angle] works in any dimension

    let point = Geonum::new_from_cartesian(3.0, 4.0);

    assert_eq!(point.length, 5.0);
    assert!((point.angle.grade_angle().tan() - 4.0 / 3.0).abs() < EPSILON);

    // angle-native construction
    let x_axis = Geonum::new(7.0, 0.0, 1.0); // 7 units at 0 radians
    let y_axis = Geonum::new(7.0, 1.0, 2.0); // 7 units at π/2 radians

    assert_eq!(x_axis.length, 7.0);
    assert!(x_axis.angle.grade_angle().abs() < EPSILON);
    assert!((y_axis.angle.grade_angle() - PI / 2.0).abs() < EPSILON);

    // distance between points
    let p1 = Geonum::new_from_cartesian(1.0, 2.0);
    let p2 = Geonum::new_from_cartesian(4.0, 6.0);
    let distance = (p2 - p1).length;

    assert!((distance - 5.0).abs() < EPSILON);
}

#[test]
fn its_a_line() {
    // traditional: slope-intercept y = mx + b, breaks for vertical lines
    // PGA: plücker coordinates [l:m:n:p:q:r], 6 components for 3D line
    //
    // geonum: wedge two points, π/2 increment creates line element

    let p0 = Geonum::new(1.0, 0.0, 1.0); // 1 unit at 0 radians, grade 0
    let p1 = Geonum::new(1.0, 1.0, 4.0); // 1 unit at π/4 radians, grade 0

    // wedge formula: [|a|*|b|*sin(θb-θa), θa + θb + π/2]
    let line = p0.wedge(&p1);

    // magnitude: 1 * 1 * sin(π/4 - 0) = sin(π/4) = √2/2
    let oriented_area = line.length;
    assert!((oriented_area - 0.7071067811865475).abs() < EPSILON);

    // angle: 0 + π/4 + π/2 = 3π/4, crosses one boundary
    assert_eq!(line.angle.blade(), 1);
    assert_eq!(line.angle.grade(), 1);

    // vertical line - no special case needed
    let v0 = Geonum::new(1.0, 0.0, 1.0); // 0 rad, grade 0
    let v1 = Geonum::new(1.0, 1.0, 2.0); // π/2 rad, grade 1

    let vertical = v0.wedge(&v1);

    // magnitude: 1 * 1 * sin(π/2 - 0) = 1.0
    let area = vertical.length;
    assert!((area - 1.0).abs() < EPSILON);

    // angle: 0 + π/2 + π/2 = π, crosses two boundaries
    assert_eq!(vertical.angle.blade(), 2);
    assert_eq!(vertical.angle.grade(), 2);

    // horizontal line - same pattern
    let h0 = Geonum::new(1.0, 0.0, 1.0); // 0 rad
    let h1 = Geonum::new(1.0, 0.0, 1.0); // same angle

    let horizontal = h0.wedge(&h1);

    // parallel vectors: sin(0) = 0, nilpotent
    assert!(horizontal.length < EPSILON);
}

#[test]
fn its_a_plane() {
    // traditional: implicit equation ax + by + cz = d, 4 coefficients
    // PGA: trivector from P₁ ∧ P₂ ∧ P₃, complex grade manipulations
    //
    // geonum: wedge grade-0 with grade-1, π/2 creates bivector space

    let p0 = Geonum::new(1.0, 0.0, 1.0); // 1 unit at 0 rad, grade 0
    let p1 = Geonum::new(1.0, 1.0, 2.0); // 1 unit at π/2 rad, grade 1

    let plane = p0.wedge(&p1);

    // orthogonal unit vectors: magnitude = 1 * 1 * sin(π/2 - 0) = 1.0
    let bivector_magnitude = plane.length;
    assert!((bivector_magnitude - 1.0).abs() < EPSILON);

    // angle: 0 + π/2 + π/2 = π, crosses two boundaries
    assert_eq!(plane.angle.blade(), 2);
    assert_eq!(plane.angle.grade(), 2);

    // bivector encodes oriented area directly
    // no normal vector computation, no implicit equations
    // the π/2 increment created the perpendicular bivector space

    // anticommutativity: order matters for orientation
    let plane_reversed = p1.wedge(&p0);

    // same magnitude
    let reversed_magnitude = plane_reversed.length;
    assert!((reversed_magnitude - 1.0).abs() < EPSILON);

    // negative sin adds π orientation (2 more blades)
    // angle: π/2 + 0 + π/2 + π = 2π, blade = 4
    assert_eq!(plane_reversed.angle.blade(), 4);
    assert_eq!(plane_reversed.angle.grade(), 0); // 4 % 4 = 0, wraps to scalar

    // different blade but same oriented magnitude
    // orientation encoded in blade count, not separate sign
}
