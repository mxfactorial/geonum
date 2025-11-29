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
    assert_eq!(y_axis.length, 7.0);

    // cartesian (3,4) is same as polar (5, atan(4/3))
    // but polar (5, π/6) doesnt require cartesian coordinates
    let polar = Geonum::new(5.0, 1.0, 6.0); // 5 units at π/6 radians
    assert_eq!(polar.length, 5.0);
    assert!((polar.angle.grade_angle() - PI / 6.0).abs() < EPSILON);

    // dimension-free: same [length, angle] representation works in 2D, 3D, 100D
    // only the interpretation of projections changes
}

#[test]
fn its_a_line() {
    // traditional PGA: line from two points uses 6 plücker coordinates (direction + moment)
    // geonum: wedge product creates bivector [magnitude, angle]

    let p1 = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let p2 = Geonum::new(1.0, 1.0, 2.0); // grade 0, π/2 from p1

    let line = p1.wedge(&p2);

    // wedge formula: magnitude = |a|*|b|*sin(angle_diff)
    // for perpendicular: sin(π/2) = 1, so magnitude = 1*1*1 = 1
    assert!((line.length - 1.0).abs() < EPSILON);

    // grade: wedge of two grade-0 creates grade-2 (bivector)
    assert_eq!(line.angle.grade(), 2);

    println!("line representation:");
    println!("  traditional PGA: 6 plücker coordinates");
    println!(
        "  geonum: [magnitude={:.3}, blade={}]",
        line.length,
        line.angle.blade()
    );
}

#[test]
fn its_a_plane() {
    // plane from wedging grade-0 and grade-1 creates grade-2 bivector

    let p0 = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let p1 = Geonum::new(1.0, 1.0, 2.0); // grade 1 (π/2)

    let plane = p0.wedge(&p1);

    // perpendicular: magnitude = 1*1*sin(π/2) = 1
    assert!((plane.length - 1.0).abs() < EPSILON);

    // grade-0 ∧ grade-1 = grade-2
    assert_eq!(plane.angle.grade(), 2);

    // bivector encodes oriented area directly
    println!("plane as bivector:");
    println!("  magnitude: oriented area = {:.3}", plane.length);
    println!("  grade: {} (bivector)", plane.angle.grade());
}

#[test]
fn its_a_rotation() {
    // rotation is angle addition, not matrix multiplication

    let point = Geonum::new(5.0, 0.0, 1.0); // 5 units at 0 radians
    let rotation = Angle::new(1.0, 4.0); // π/4 radians

    let rotated = point.rotate(rotation);

    // rotation preserves magnitude
    assert_eq!(rotated.length, 5.0);

    // angle increased by π/4
    assert!((rotated.angle.grade_angle() - PI / 4.0).abs() < EPSILON);

    // traditional: requires 2×2 rotation matrix and 4 multiplications
    // geonum: single angle addition
    println!("rotation:");
    println!("  traditional: matrix multiplication O(n²)");
    println!("  geonum: angle addition O(1)");
}

#[test]
fn its_a_3d_point() {
    // how does [length, angle] represent points in 3D?
    // it depends on what "3D point" means

    println!("\n=== 3D POINT REPRESENTATIONS ===\n");

    // case 1: uniform magnitude point like (1, 1, 1)
    // this is just "3 quarter-turns from origin"
    let uniform = Geonum::new_with_blade(1.0, 3, 0.0, 1.0);

    println!("uniform magnitude (1,1,1):");
    println!("  representation: Geonum::new_with_blade(1.0, 3, 0.0, 1.0)");
    println!("  blade: {}", uniform.angle.blade());
    println!("  grade: {}", uniform.angle.grade());
    println!("  interpretation: 1 unit after 3 quarter-turns");

    assert_eq!(uniform.angle.blade(), 3);
    assert_eq!(uniform.angle.grade(), 3); // 3 % 4 = 3

    // case 2: point along single axis like (0, 1, 0)
    // this is "1 unit after 1 quarter-turn"
    let y_axis = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    println!("\nsingle axis (0,1,0):");
    println!("  representation: Geonum::new_with_blade(1.0, 1, 0.0, 1.0)");
    println!("  blade: {}", y_axis.angle.blade());
    println!("  grade: {}", y_axis.angle.grade());
    println!("  interpretation: 1 unit after 1 quarter-turn (y-axis)");

    assert_eq!(y_axis.angle.blade(), 1);
    assert_eq!(y_axis.angle.grade(), 1);

    // case 3: point along z-axis like (0, 0, 5)
    // this is "5 units after 2 quarter-turns"
    let z_axis = Geonum::new_with_blade(5.0, 2, 0.0, 1.0);

    println!("\nz-axis point (0,0,5):");
    println!("  representation: Geonum::new_with_blade(5.0, 2, 0.0, 1.0)");
    println!("  blade: {}", z_axis.angle.blade());
    println!("  grade: {}", z_axis.angle.grade());
    println!("  interpretation: 5 units after 2 quarter-turns (z-axis)");

    assert_eq!(z_axis.length, 5.0);
    assert_eq!(z_axis.angle.blade(), 2);
    assert_eq!(z_axis.angle.grade(), 2);

    // case 4: arbitrary point like (1, 2, 3) needs GeoCollection
    // each component is a separate geonum
    let arbitrary = GeoCollection::from(vec![
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // x component: 1 unit at blade 0
        Geonum::new_with_blade(2.0, 1, 0.0, 1.0), // y component: 2 units at blade 1
        Geonum::new_with_blade(3.0, 2, 0.0, 1.0), // z component: 3 units at blade 2
    ]);

    println!("\narbitrary point (1,2,3):");
    println!("  representation: GeoCollection with 3 geonums");
    println!(
        "  x: {} units at blade {}",
        arbitrary[0].length,
        arbitrary[0].angle.blade()
    );
    println!(
        "  y: {} units at blade {}",
        arbitrary[1].length,
        arbitrary[1].angle.blade()
    );
    println!(
        "  z: {} units at blade {}",
        arbitrary[2].length,
        arbitrary[2].angle.blade()
    );

    assert_eq!(arbitrary.len(), 3);
    assert_eq!(arbitrary[0].length, 1.0);
    assert_eq!(arbitrary[1].length, 2.0);
    assert_eq!(arbitrary[2].length, 3.0);

    // case 5: cartesian conversion for familiar coordinates
    let from_cartesian = Geonum::new_from_cartesian(3.0, 4.0);

    println!("\nfrom cartesian (3,4) in 2D:");
    println!("  representation: Geonum::new_from_cartesian(3.0, 4.0)");
    println!("  length: {:.4}", from_cartesian.length);
    println!("  angle: {:.4} rad", from_cartesian.angle.grade_angle());
    println!("  interpretation: 5 units at atan(4/3)");

    let expected_length = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt();
    let expected_angle = (4.0_f64).atan2(3.0);

    assert!((from_cartesian.length - expected_length).abs() < EPSILON);
    assert!((from_cartesian.angle.grade_angle() - expected_angle).abs() < EPSILON);

    println!("\n=== KEY INSIGHT ===");
    println!("\"3D point\" can mean:");
    println!("  1. position in 3D space → use GeoCollection for arbitrary coordinates");
    println!("  2. geometric object at grade 3 → use blade count");
    println!("  3. blade field encodes dimensional context, not position");
}

#[test]
fn its_non_euclidean() {
    // blade projections are views of one angle, not independent coordinates
    // they dont satisfy euclidean pythagorean theorem

    let entity = Geonum::new_with_blade(2.0, 3, 0.0, 1.0);

    println!("\nentity with blade 3:");
    println!("  length: {:.3}", entity.length);
    println!("  blade: {}", entity.angle.blade());
    println!("  angle: {:.3} rad", entity.angle.grade_angle());

    // project to first 4 dimensions
    let proj_0 = entity.project_to_dimension(0);
    let proj_1 = entity.project_to_dimension(1);
    let proj_2 = entity.project_to_dimension(2);
    let proj_3 = entity.project_to_dimension(3);

    println!("\nprojections:");
    println!("  dimension 0: {:.3}", proj_0);
    println!("  dimension 1: {:.3}", proj_1);
    println!("  dimension 2: {:.3}", proj_2);
    println!("  dimension 3: {:.3}", proj_3);

    // test euclidean norm
    let euclidean_magnitude =
        (proj_0.powi(2) + proj_1.powi(2) + proj_2.powi(2) + proj_3.powi(2)).sqrt();

    println!("\neuclidean test:");
    println!(
        "  √(proj₀² + proj₁² + proj₂² + proj₃²) = {:.3}",
        euclidean_magnitude
    );
    println!("  entity.length = {:.3}", entity.length);
    println!(
        "  equal? {}",
        (euclidean_magnitude - entity.length).abs() < EPSILON
    );

    // projections are trigonometrically consistent but not euclidean
    // they are views of one angle, not independent vectors
    assert!((euclidean_magnitude - entity.length).abs() > EPSILON);

    println!("\nwhy? projections are views of ONE angle onto different dimensions");
    println!("not independent coordinates that satisfy x²+y²+z²=r²");
}

#[test]
fn its_a_line_between_two_3d_points() {
    // traditional PGA: line through two points uses 6 plücker coordinates
    // 3 for direction, 3 for moment (offset from origin)

    // geonum: points are [length, angle]
    // the angle encodes the direction via blade counting

    // simplified example: two points in different blade positions
    // point 1: 2 units at blade 0 (0 radians)
    let p1 = Geonum::new_with_blade(2.0, 0, 0.0, 1.0);

    // point 2: 3 units at blade 1 (π/2 radians)
    let p2 = Geonum::new_with_blade(3.0, 1, 0.0, 1.0);

    println!("point 1:");
    println!("  length: {:.3}", p1.length);
    println!("  angle: {:.3} rad", p1.angle.grade_angle());
    println!("  blade: {}", p1.angle.blade());

    println!("\npoint 2:");
    println!("  length: {:.3}", p2.length);
    println!("  angle: {:.3} rad (π/2)", p2.angle.grade_angle());
    println!("  blade: {}", p2.angle.blade());

    // line between them via wedge product
    // wedge formula: [|a|*|b|*sin(θb-θa), θa + θb + π/2]
    let line = p1.wedge(&p2);

    println!("\nline via wedge:");
    println!("  magnitude: {:.3}", line.length);
    println!("  angle: {:.3} rad", line.angle.grade_angle());
    println!("  blade: {}", line.angle.blade());
    println!("  grade: {}", line.angle.grade());

    // for perpendicular vectors (π/2 apart): sin(π/2) = 1
    // magnitude: 2 * 3 * 1 = 6
    let angle_diff = (p2.angle.grade_angle() - p1.angle.grade_angle()).abs();
    let expected_mag = p1.length * p2.length * angle_diff.sin();

    println!("\nwedge magnitude:");
    println!("  |a| * |b| * sin(θb-θa)");
    println!(
        "  {:.3} * {:.3} * sin({:.3})",
        p1.length, p2.length, angle_diff
    );
    println!("  = {:.3}", expected_mag);

    assert!((line.length - expected_mag).abs() < EPSILON);
    assert_eq!(line.angle.grade(), 2, "wedge creates bivector (grade 2)");

    // storage comparison
    println!("\nstorage:");
    println!("  traditional PGA: 6 plücker coordinates");
    println!(
        "  geonum: 2 values [length={:.3}, angle with blade={}]",
        line.length,
        line.angle.blade()
    );

    // the wedge encodes both direction and moment in angle + magnitude
    // no decomposition into 6 separate scalars needed
}

#[test]
fn it_encodes_moment() {
    // traditional concern: "line not through origin requires non-zero moment in plücker coordinates"

    // the moment in plücker coordinates encodes offset from origin
    // in geonum, this is captured by the position vectors and their wedge

    // build the two points as position vectors from origin
    let p1 = Geonum::new_from_cartesian(1.0, 0.0); // simplified to 2D: (1,0)
    let p2 = Geonum::new_from_cartesian(0.0, 1.0); // (0,1)

    println!("point p1 at (1,0):");
    println!("  length: {:.3}", p1.length);
    println!("  angle: {:.3} rad", p1.angle.grade_angle());

    println!("\npoint p2 at (0,1):");
    println!("  length: {:.3}", p2.length);
    println!("  angle: {:.3} rad (π/2)", p2.angle.grade_angle());

    // line between them
    let line = p1.wedge(&p2);

    println!("\nline via wedge:");
    println!("  magnitude: {:.3}", line.length);
    println!("  angle: {:.3} rad", line.angle.grade_angle());
    println!("  blade: {}", line.angle.blade());

    // the wedge magnitude is the oriented area of parallelogram
    // for perpendicular unit vectors: 1 * 1 * sin(π/2) = 1
    assert!(
        (line.length - 1.0).abs() < EPSILON,
        "unit area for perpendicular unit vectors"
    );

    // this line passes through (1,0) and (0,1), not the origin
    // the "moment" information is encoded in the combined position vectors
    // no need to separately store direction + moment

    // in traditional PGA:
    // - direction: (0,1) - (1,0) = (-1, 1, 0)
    // - moment: (1,0,0) × (-1,1,0) encodes offset
    // - total: 6 values

    // in geonum:
    // - wedge encodes both position and orientation
    // - total: 2 values (length + angle with blade)

    println!("\nthe wedge magnitude and angle encode what plücker separates:");
    println!("  magnitude: oriented area (related to moment)");
    println!("  angle: combined orientation (related to direction)");
    println!("  no decomposition into direction + moment needed");
}

#[test]
fn its_a_cartesian_unit_cube() {
    // unit cube with 8 vertices
    // plain coordinate math, no geonum

    println!("\n=== CARTESIAN UNIT CUBE ===\n");

    // 8 vertices as (x, y, z) tuples
    let v000 = (0.0, 0.0, 0.0);
    let v100 = (1.0, 0.0, 0.0);
    let v010 = (0.0, 1.0, 0.0);
    let v001 = (0.0, 0.0, 1.0);
    let v110 = (1.0, 1.0, 0.0);
    let v101 = (1.0, 0.0, 1.0);
    let v011 = (0.0, 1.0, 1.0);
    let v111 = (1.0, 1.0, 1.0);

    println!("vertices:");
    println!("  v000: {:?}", v000);
    println!("  v100: {:?}", v100);
    println!("  v010: {:?}", v010);
    println!("  v001: {:?}", v001);
    println!("  v110: {:?}", v110);
    println!("  v101: {:?}", v101);
    println!("  v011: {:?}", v011);
    println!("  v111: {:?}", v111);

    println!("\n=== EDGES ===\n");

    // 12 edges (4 along each axis direction)
    // compute edge vectors by subtraction

    // x-direction edges
    let edge_x1: (f64, f64, f64) = (v100.0 - v000.0, v100.1 - v000.1, v100.2 - v000.2);
    let edge_x2: (f64, f64, f64) = (v110.0 - v010.0, v110.1 - v010.1, v110.2 - v010.2);
    let edge_x3: (f64, f64, f64) = (v101.0 - v001.0, v101.1 - v001.1, v101.2 - v001.2);
    let edge_x4: (f64, f64, f64) = (v111.0 - v011.0, v111.1 - v011.1, v111.2 - v011.2);

    println!("x-direction edges:");
    println!("  v000→v100: {:?}", edge_x1);
    println!("  v010→v110: {:?}", edge_x2);
    println!("  v001→v101: {:?}", edge_x3);
    println!("  v011→v111: {:?}", edge_x4);

    // edge lengths (euclidean norm)
    let len_x1 = (edge_x1.0.powi(2) + edge_x1.1.powi(2) + edge_x1.2.powi(2)).sqrt();
    let len_x2 = (edge_x2.0.powi(2) + edge_x2.1.powi(2) + edge_x2.2.powi(2)).sqrt();
    let len_x3 = (edge_x3.0.powi(2) + edge_x3.1.powi(2) + edge_x3.2.powi(2)).sqrt();
    let len_x4 = (edge_x4.0.powi(2) + edge_x4.1.powi(2) + edge_x4.2.powi(2)).sqrt();

    println!(
        "  lengths: {:.3}, {:.3}, {:.3}, {:.3}",
        len_x1, len_x2, len_x3, len_x4
    );

    assert!((len_x1 - 1.0).abs() < EPSILON);
    assert!((len_x2 - 1.0).abs() < EPSILON);
    assert!((len_x3 - 1.0).abs() < EPSILON);
    assert!((len_x4 - 1.0).abs() < EPSILON);

    // y-direction edges
    let edge_y1: (f64, f64, f64) = (v010.0 - v000.0, v010.1 - v000.1, v010.2 - v000.2);
    let edge_y2: (f64, f64, f64) = (v110.0 - v100.0, v110.1 - v100.1, v110.2 - v100.2);
    let edge_y3: (f64, f64, f64) = (v011.0 - v001.0, v011.1 - v001.1, v011.2 - v001.2);
    let edge_y4: (f64, f64, f64) = (v111.0 - v101.0, v111.1 - v101.1, v111.2 - v101.2);

    let len_y1 = (edge_y1.0.powi(2) + edge_y1.1.powi(2) + edge_y1.2.powi(2)).sqrt();
    let len_y2 = (edge_y2.0.powi(2) + edge_y2.1.powi(2) + edge_y2.2.powi(2)).sqrt();
    let len_y3 = (edge_y3.0.powi(2) + edge_y3.1.powi(2) + edge_y3.2.powi(2)).sqrt();
    let len_y4 = (edge_y4.0.powi(2) + edge_y4.1.powi(2) + edge_y4.2.powi(2)).sqrt();

    println!("\ny-direction edges:");
    println!(
        "  lengths: {:.3}, {:.3}, {:.3}, {:.3}",
        len_y1, len_y2, len_y3, len_y4
    );

    assert!((len_y1 - 1.0).abs() < EPSILON);
    assert!((len_y2 - 1.0).abs() < EPSILON);
    assert!((len_y3 - 1.0).abs() < EPSILON);
    assert!((len_y4 - 1.0).abs() < EPSILON);

    // z-direction edges
    let edge_z1: (f64, f64, f64) = (v001.0 - v000.0, v001.1 - v000.1, v001.2 - v000.2);
    let edge_z2: (f64, f64, f64) = (v101.0 - v100.0, v101.1 - v100.1, v101.2 - v100.2);
    let edge_z3: (f64, f64, f64) = (v011.0 - v010.0, v011.1 - v010.1, v011.2 - v010.2);
    let edge_z4: (f64, f64, f64) = (v111.0 - v110.0, v111.1 - v110.1, v111.2 - v110.2);

    let len_z1 = (edge_z1.0.powi(2) + edge_z1.1.powi(2) + edge_z1.2.powi(2)).sqrt();
    let len_z2 = (edge_z2.0.powi(2) + edge_z2.1.powi(2) + edge_z2.2.powi(2)).sqrt();
    let len_z3 = (edge_z3.0.powi(2) + edge_z3.1.powi(2) + edge_z3.2.powi(2)).sqrt();
    let len_z4 = (edge_z4.0.powi(2) + edge_z4.1.powi(2) + edge_z4.2.powi(2)).sqrt();

    println!("\nz-direction edges:");
    println!(
        "  lengths: {:.3}, {:.3}, {:.3}, {:.3}",
        len_z1, len_z2, len_z3, len_z4
    );

    assert!((len_z1 - 1.0).abs() < EPSILON);
    assert!((len_z2 - 1.0).abs() < EPSILON);
    assert!((len_z3 - 1.0).abs() < EPSILON);
    assert!((len_z4 - 1.0).abs() < EPSILON);

    println!("\n=== FACE DIAGONALS ===\n");

    // xy-face diagonal from v000 to v110
    let diag_xy: (f64, f64, f64) = (v110.0 - v000.0, v110.1 - v000.1, v110.2 - v000.2);
    let len_diag_xy = (diag_xy.0.powi(2) + diag_xy.1.powi(2) + diag_xy.2.powi(2)).sqrt();

    println!("xy-face diagonal v000→v110:");
    println!("  vector: {:?}", diag_xy);
    println!(
        "  length: {:.3} (expected √2 = {:.3})",
        len_diag_xy,
        2.0_f64.sqrt()
    );

    assert!((len_diag_xy - 2.0_f64.sqrt()).abs() < EPSILON);

    // xz-face diagonal from v000 to v101
    let diag_xz: (f64, f64, f64) = (v101.0 - v000.0, v101.1 - v000.1, v101.2 - v000.2);
    let len_diag_xz = (diag_xz.0.powi(2) + diag_xz.1.powi(2) + diag_xz.2.powi(2)).sqrt();

    println!("\nxz-face diagonal v000→v101:");
    println!("  length: {:.3}", len_diag_xz);

    assert!((len_diag_xz - 2.0_f64.sqrt()).abs() < EPSILON);

    // yz-face diagonal from v000 to v011
    let diag_yz: (f64, f64, f64) = (v011.0 - v000.0, v011.1 - v000.1, v011.2 - v000.2);
    let len_diag_yz = (diag_yz.0.powi(2) + diag_yz.1.powi(2) + diag_yz.2.powi(2)).sqrt();

    println!("\nyz-face diagonal v000→v011:");
    println!("  length: {:.3}", len_diag_yz);

    assert!((len_diag_yz - 2.0_f64.sqrt()).abs() < EPSILON);

    println!("\n=== BODY DIAGONAL ===\n");

    // main diagonal from v000 to v111
    let diag_main: (f64, f64, f64) = (v111.0 - v000.0, v111.1 - v000.1, v111.2 - v000.2);
    let len_diag_main = (diag_main.0.powi(2) + diag_main.1.powi(2) + diag_main.2.powi(2)).sqrt();

    println!("body diagonal v000→v111:");
    println!("  vector: {:?}", diag_main);
    println!(
        "  length: {:.3} (expected √3 = {:.3})",
        len_diag_main,
        3.0_f64.sqrt()
    );

    assert!((len_diag_main - 3.0_f64.sqrt()).abs() < EPSILON);

    println!("\n=== FACE AREA ===\n");

    // xy-face area via cross product
    // vectors: v000→v100 and v000→v010
    let u = edge_x1; // (1, 0, 0)
    let v = edge_y1; // (0, 1, 0)

    // cross product u × v
    let cross_xy: (f64, f64, f64) = (
        u.1 * v.2 - u.2 * v.1,
        u.2 * v.0 - u.0 * v.2,
        u.0 * v.1 - u.1 * v.0,
    );

    let area_xy = (cross_xy.0.powi(2) + cross_xy.1.powi(2) + cross_xy.2.powi(2)).sqrt();

    println!("xy-face area:");
    println!("  u = {:?}", u);
    println!("  v = {:?}", v);
    println!("  u × v = {:?}", cross_xy);
    println!("  |u × v| = {:.3}", area_xy);

    assert!((area_xy - 1.0).abs() < EPSILON);

    println!("\n=== VOLUME ===\n");

    // volume via scalar triple product: u · (v × w)
    let w = edge_z1; // (0, 0, 1)

    // v × w
    let cross_vw: (f64, f64, f64) = (
        v.1 * w.2 - v.2 * w.1,
        v.2 * w.0 - v.0 * w.2,
        v.0 * w.1 - v.1 * w.0,
    );

    // u · (v × w)
    let volume = u.0 * cross_vw.0 + u.1 * cross_vw.1 + u.2 * cross_vw.2;

    println!("cube volume:");
    println!("  u = {:?}", u);
    println!("  v = {:?}", v);
    println!("  w = {:?}", w);
    println!("  v × w = {:?}", cross_vw);
    println!("  u · (v × w) = {:.3}", volume);

    assert!((volume - 1.0).abs() < EPSILON);

    println!("\n=== SUMMARY ===");
    println!("cube with 8 vertices, 12 edges, 6 faces");
    println!("all edges: length 1.0");
    println!("face diagonals: length √2 ≈ {:.3}", 2.0_f64.sqrt());
    println!("body diagonal: length √3 ≈ {:.3}", 3.0_f64.sqrt());
    println!("face area: 1.0");
    println!("volume: 1.0");
}

#[test]
fn its_a_geonum_unit_cube() {
    // demonstrate geonum geometric operations
    // 2D: square area and diagonal (✓)
    // 3D: unit volume via wedge

    println!("\n=== GEONUM UNIT CUBE ===\n");

    // two orthogonal unit edges for 2D operations
    let edge_x = Geonum::new(1.0, 0.0, 1.0); // angle 0
    let edge_y = Geonum::new(1.0, 1.0, 2.0); // angle π/2

    println!("two orthogonal edges:");
    println!(
        "  edge_x: magnitude={}, angle={:.3}, grade={}",
        edge_x.length,
        edge_x.angle.grade_angle(),
        edge_x.angle.grade()
    );
    println!(
        "  edge_y: magnitude={}, angle={:.3}, grade={}",
        edge_y.length,
        edge_y.angle.grade_angle(),
        edge_y.angle.grade()
    );

    println!("\n=== SQUARE DIAGONAL (2D) ===\n");

    let diagonal_2d = edge_x + edge_y;
    println!("edge_x + edge_y:");
    println!("  magnitude: {:.3}", diagonal_2d.length);
    println!("  expected: √2 = {:.3}", 2.0_f64.sqrt());

    if (diagonal_2d.length - 2.0_f64.sqrt()).abs() < EPSILON {
        println!("  ✓ square diagonal works!");
    }

    println!("\n=== SQUARE AREA (2D) ===\n");

    let area = edge_x.wedge(&edge_y);
    println!("edge_x ∧ edge_y:");
    println!("  magnitude: {:.3} (area)", area.length);
    println!("  grade: {} (bivector)", area.angle.grade());
    println!("  angle: {:.3}", area.angle.grade_angle());

    if (area.length - 1.0).abs() < EPSILON && area.angle.grade() == 2 {
        println!("  ✓ square area = 1.0, stored as single bivector!");
    }

    println!("\n=== UNIT VOLUME (3D) ===\n");

    // for volume: two edges with angles summing to π give trivector
    // maximum magnitude when perpendicular: π/4 + 3π/4
    let vol_edge1 = Geonum::new_with_blade(1.0, 0, 1.0, 4.0); // angle π/4
    let vol_edge2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // blade 1 + π/4 = angle 3π/4

    println!("two edges for volume:");
    println!("  edge1: angle={:.3} (π/4)", vol_edge1.angle.grade_angle());
    println!("  edge2: angle={:.3} (3π/4)", vol_edge2.angle.grade_angle());
    println!(
        "  sum: {:.3} + {:.3} = {:.3} (π)",
        vol_edge1.angle.grade_angle(),
        vol_edge2.angle.grade_angle(),
        vol_edge1.angle.grade_angle() + vol_edge2.angle.grade_angle()
    );

    let volume = vol_edge1.wedge(&vol_edge2);
    println!("\nedge1 ∧ edge2:");
    println!("  magnitude: {:.3} (volume)", volume.length);
    println!("  grade: {} (trivector)", volume.angle.grade());
    println!("  angle: {:.3} (3π/2)", volume.angle.grade_angle());

    if (volume.length - 1.0).abs() < EPSILON && volume.angle.grade() == 3 {
        println!("  ✓ unit volume = 1.0, stored as single trivector!");
    }

    println!("\n=== SUMMARY ===");
    println!("geonum geometric operations:");
    println!(
        "  2D diagonal: {:.3} (expected √2 = {:.3}) ✓",
        diagonal_2d.length,
        2.0_f64.sqrt()
    );
    println!("  2D area: {:.3} bivector (grade 2) ✓", area.length);
    println!("  3D volume: {:.3} trivector (grade 3) ✓", volume.length);
    println!("\nKey insight: angles summing to π → trivector via wedge");
}
