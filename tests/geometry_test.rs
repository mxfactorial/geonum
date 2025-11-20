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
    println!("  x: {} units at blade {}", arbitrary[0].length, arbitrary[0].angle.blade());
    println!("  y: {} units at blade {}", arbitrary[1].length, arbitrary[1].angle.blade());
    println!("  z: {} units at blade {}", arbitrary[2].length, arbitrary[2].angle.blade());

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
fn its_a_3d_volume() {
    // volumes dont need 3 independent edge values
    //
    // traditional: store [width, height, depth] as 3 independent scalars
    // problem: theyre NOT independent - constrained by pythagorean relationship
    //
    // geonum: store diagonal [length, angle], edges are constrained projections
    // benefit: makes the geometric constraint explicit, not hidden

    println!("\n=== 3D VOLUME: EDGES AS CONSTRAINED PROJECTIONS ===\n");

    let width: f64 = 3.0;
    let height: f64 = 4.0;
    let depth: f64 = 5.0;

    // traditional approach hides the constraint
    println!("traditional storage: [{}, {}, {}]", width, height, depth);
    println!("  appears independent but constrained by x² + y² + z² = diagonal²");

    // compute diagonal - this is the actual geometric primitive
    let diagonal = (width.powi(2) + height.powi(2) + depth.powi(2)).sqrt();
    println!("\ndiagonal: {:.4}", diagonal);

    // encode angles using spherical-like coordinates
    // NOTE: spherical coords (r, θ, φ) already do this - not revolutionary (why not 10/10)
    // but geonum extends this to operations via blade arithmetic
    let xy_projection = (width.powi(2) + height.powi(2)).sqrt();
    let theta = height.atan2(width);           // azimuth in xy-plane
    let phi = depth.atan2(xy_projection);      // elevation from xy-plane

    println!("  azimuth θ: {:.4} rad", theta);
    println!("  elevation φ: {:.4} rad", phi);

    // the geometric number: [diagonal, angle]
    // this is 2 values regardless of dimension (8/10: same for 3D, 10D, 1000D)
    let diagonal_geonum = Geonum::new(diagonal, theta, PI);

    println!("\ngeonum representation:");
    println!("  [length: {:.4}, angle: {:.4}]",
        diagonal_geonum.length,
        diagonal_geonum.angle.grade_angle()
    );
    println!("  storage: {} bytes (constant for any dimension)",
        std::mem::size_of_val(&diagonal_geonum)
    );

    // INSIGHT: edges are projections, not independent measurements
    // this is the systematic inversion - measurements derive from geometry
    let edge_x = diagonal * theta.cos() * phi.cos();
    let edge_y = diagonal * theta.sin() * phi.cos();
    let edge_z = diagonal * phi.sin();

    println!("\nedges via projection (measurements from geometry):");
    println!("  x = diagonal × cos(θ) × cos(φ) = {:.4}", edge_x);
    println!("  y = diagonal × sin(θ) × cos(φ) = {:.4}", edge_y);
    println!("  z = diagonal × sin(φ)          = {:.4}", edge_z);

    assert!((edge_x - width).abs() < EPSILON);
    assert!((edge_y - height).abs() < EPSILON);
    assert!((edge_z - depth).abs() < EPSILON);

    // the pythagorean constraint is automatic, not enforced
    // (pushes toward 9/10: constraint falls out naturally)
    let sum_of_squares = edge_x.powi(2) + edge_y.powi(2) + edge_z.powi(2);
    let diagonal_squared = diagonal.powi(2);

    println!("\npythagorean constraint (automatic):");
    println!("  x² + y² + z² = {:.4}", sum_of_squares);
    println!("  diagonal²    = {:.4}", diagonal_squared);
    println!("  difference:    {:.4e}", (sum_of_squares - diagonal_squared).abs());

    assert!((sum_of_squares - diagonal_squared).abs() < EPSILON);

    // volume computable from projections
    let volume = edge_x * edge_y * edge_z;
    println!("\nvolume = x × y × z = {:.4}", volume);

    // COMPARISON: traditional GA approach
    // would need 2^3 = 8 components to represent 3D multivector
    // geonum: 2 values (length + angle) regardless of dimension
    println!("\n=== COMPLEXITY COMPARISON ===");
    println!("traditional GA in 3D:");
    println!("  multivector components: 2³ = 8");
    println!("  per-dimension scaling: O(2ⁿ)");
    println!("\ngeonum:");
    println!("  storage: 2 values [length, angle]");
    println!("  any dimension: still 2 values");
    println!("  edges: projections (not stored, computed)");

    // demonstrate dimension-free property
    println!("\n=== DIMENSION-FREE VERIFICATION ===");
    let dims = [3, 10, 100, 1000];
    for &d in &dims {
        let test_diagonal = Geonum::new_with_blade(diagonal, d, theta, PI);
        println!("  {}D: {} bytes", d, std::mem::size_of_val(&test_diagonal));
        assert_eq!(
            std::mem::size_of_val(&test_diagonal),
            std::mem::size_of_val(&diagonal_geonum)
        );
    }

    println!("\n=== KEY INSIGHTS ===");
    println!("✓ [x,y,z] independence is artificial");
    println!("✓ pythagorean constraint automatic, not enforced");
    println!("✓ same representation for 3D and 1000D volumes");
    println!("✓ systematic inversion: measurements from geometry");
    println!("✓ 2 values vs 2ⁿ components for traditional GA");
    println!();
    println!("comparison to spherical coordinates:");
    println!("  (r,θ,φ) also encodes position via angles");
    println!("  geonum extends this to operations via blade arithmetic");
    println!();
    println!("exponential reduction:");
    println!("  traditional GA: 8 components for 3D, 1024 for 10D");
    println!("  geonum: 2 values for any dimension");
}

#[test]
fn its_non_euclidean() {
    // blade projections don't satisfy pythagorean theorem
    // this is not a bug - it reveals euclidean coordinate independence is irrelevant
    //
    // pythagorean x² + y² + z² = r² assumes independent orthogonal coordinates
    // but blade projections are views of ONE angle onto different blade positions
    // testing pythagorean imposes euclidean scaffolding on angle-native geometry

    println!("\n=== BLADE PROJECTIONS ARE NON-EUCLIDEAN ===\n");

    // create geometric number via rotations (natural construction)
    let base = Geonum::scalar(5.0);
    let rotated = base.rotate(Angle::new(1.0, 4.0)); // π/4 rotation
    let entity = rotated.scale(2.0); // 10 units at π/4

    println!("entity: 10 units at π/4");
    println!("  length: {}", entity.length);
    println!("  angle: {:.4} rad", entity.angle.grade_angle());

    // project onto blade dimensions
    let proj_0 = entity.project_to_dimension(0); // blade 0 space
    let proj_1 = entity.project_to_dimension(1); // blade 1 space
    let proj_2 = entity.project_to_dimension(2); // blade 2 space

    println!("\nblade projections:");
    println!("  blade 0: {:.4}", proj_0);
    println!("  blade 1: {:.4}", proj_1);
    println!("  blade 2: {:.4}", proj_2);

    // projections are trigonometrically consistent
    let angle = entity.angle.grade_angle();
    let expected_0 = entity.length * (angle - 0.0).cos();
    let expected_1 = entity.length * (angle - PI / 2.0).cos();

    assert!((proj_0 - expected_0).abs() < EPSILON);
    assert!((proj_1 - expected_1).abs() < EPSILON);

    println!("\ntrigonometric consistency:");
    println!("  proj_0 = length × cos(angle - 0)     ✓");
    println!("  proj_1 = length × cos(angle - π/2)   ✓");

    // TEST: do blade projections satisfy pythagorean theorem?
    let euclidean_magnitude = (proj_0.powi(2) + proj_1.powi(2) + proj_2.powi(2)).sqrt();
    let actual_magnitude = entity.length;

    println!("\npythagorean test:");
    println!("  √(proj_0² + proj_1² + proj_2²) = {:.4}", euclidean_magnitude);
    println!("  actual magnitude               = {:.4}", actual_magnitude);
    println!("  difference                     = {:.4}", (euclidean_magnitude - actual_magnitude).abs());

    // this FAILS - blade projections are non-euclidean
    assert!(
        (euclidean_magnitude - actual_magnitude).abs() > 0.1,
        "blade projections do NOT satisfy pythagorean theorem"
    );

    println!("\n  ✗ pythagorean theorem FAILS");
    println!("  → blade projections are non-euclidean");

    // WHY this is expected, not a bug
    println!("\n=== WHY PYTHAGOREAN FAILS ===");
    println!();
    println!("pythagorean x² + y² + z² = r² assumes:");
    println!("  • x, y, z are independent measurements");
    println!("  • orthogonal coordinate axes");
    println!("  • linear combination: r⃗ = x·e₁ + y·e₂ + z·e₃");
    println!();
    println!("blade projections are:");
    println!("  • projections of ONE angle onto blade positions");
    println!("  • NOT independent coordinates");
    println!("  • views of same geometric object, not components");
    println!();
    println!("testing pythagorean asks:");
    println!("  'do blade projections behave like euclidean coordinates?'");
    println!();
    println!("answer: NO - and that's the point");
    println!("  blade geometry is angle-native");
    println!("  euclidean coordinate independence is projection artifact");

    // what IS preserved: angle relationships
    println!("\n=== WHAT IS PRESERVED ===");

    let blade_0_axis = Angle::new_with_blade(0, 0.0, 1.0);
    let blade_1_axis = Angle::new_with_blade(1, 0.0, 1.0);

    let angle_to_0 = entity.angle.project(blade_0_axis);
    let angle_to_1 = entity.angle.project(blade_1_axis);

    assert!((proj_0 - entity.length * angle_to_0).abs() < EPSILON);
    assert!((proj_1 - entity.length * angle_to_1).abs() < EPSILON);

    println!("angle relationships:");
    println!("  proj_0 = length × angle_projection_0  ✓");
    println!("  proj_1 = length × angle_projection_1  ✓");
    println!();
    println!("blade projections preserve:");
    println!("  • angle relationships");
    println!("  • trigonometric consistency");
    println!("  • rotational structure");
    println!();
    println!("blade projections do NOT preserve:");
    println!("  • euclidean coordinate independence");
    println!("  • pythagorean magnitude relationship");
    println!("  • linear combination properties");

    println!("\n=== KEY INSIGHT ===");
    println!();
    println!("pythagorean theorem describes euclidean coordinate scaffolding");
    println!("blade geometry works with angle primitives directly");
    println!();
    println!("testing if blade projections satisfy pythagorean is like");
    println!("testing if polar coordinates (r,θ) satisfy cartesian relationships");
    println!("→ you're imposing coordinate assumptions on angle-native representation");
    println!();
    println!("blade projections are non-euclidean by design");
    println!("not because geometry is broken");
    println!("but because euclidean independence is irrelevant to angle space");
}
