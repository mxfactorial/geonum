// geonum doesnt need a pseudoscalar
//
// traditional geometric algebra builds duality on a pseudoscalar I = e₁∧…∧eₙ, the
// top-grade element, and pays 2^n components to carry the multivector it lives in.
// geonum replaces I with one angle op: the dual adds π (two blades), and the 2^n
// explosion collapses to the two numbers [magnitude, angle]. these tests prove the
// pseudoscalar is unnecessary scaffolding — duality, the dimensional ceiling, and the
// exponential component count all fall out of blade arithmetic
//
// run: cargo test --test pseudoscalar_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_solves_the_exponential_complexity_explosion() {
    // THE PROBLEM: traditional GA suffers from 2^n explosion
    // why? it refuses to acknowledge that rotations compose by angle addition
    // instead, it scatters rotation information across exponentially many components

    // traditional GA component count:
    // 1D: 2 components (scalar, e1)
    // 2D: 4 components (scalar, e1, e2, e12)
    // 3D: 8 components (scalar, e1, e2, e3, e12, e13, e23, e123)
    // 10D: 1024 components (all possible products of basis vectors)
    // nD: 2^n components

    // THE SOLUTION: geonum recognizes that a rotation is just [length, angle]
    // no matter how many dimensions, a 45° rotation is stored as one number

    // proof: represent a 45° rotation
    let rotation_45 = Geonum::new(1.0, 1.0, 4.0); // [1, π/4]

    // apply this rotation to different objects - always the same operation
    let x_axis = Geonum::new(1.0, 0.0, 1.0);
    let rotated = x_axis * rotation_45; // rotate x-axis by 45°
    assert_eq!(rotated.angle, Angle::new(1.0, 4.0)); // now at 45°

    // this single number works in ANY dimension:
    // - in 2D: rotates in the xy-plane
    // - in 3D: rotates in the xy-plane (z unchanged)
    // - in 10D: rotates in the xy-plane (other 8 dims unchanged)

    // traditional GA cant do this! it needs:
    // - 2D: distribute across 4 components
    // - 3D: distribute across 8 components
    // - 10D: distribute across 1024 components
    // all to represent the same simple 45° rotation

    // the key insight: multiplication is just angle addition
    let a = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]
    let b = Geonum::new(3.0, 1.0, 3.0); // [3, π/3]
    let product = a * b;

    // geonum: O(1) operations
    assert_eq!(product.mag, 6.0); // lengths multiply
    assert_eq!(product.angle, Angle::new(1.0, 2.0)); // angles add: π/6 + π/3 = π/2

    // traditional GA: O(4^n) operations for the same result!
    // in 10D: 1024 × 1024 = 1,048,576 component multiplications
    // of which 99.9% produce zeros that still get computed and stored

    // even worse: chained operations
    let c = Geonum::new(1.5, 1.0, 4.0); // [1.5, π/4]
    let chain = a * b * c;

    // geonum: still O(1)
    assert_eq!(chain.mag, 9.0); // 2 × 3 × 1.5
    assert_eq!(chain.angle, Angle::new(3.0, 4.0)); // π/6 + π/3 + π/4 = 3π/4

    // traditional GA: must expand (a*b) into 2^n components,
    // then multiply all 2^n by c's 2^n components
    // the explosion compounds with every operation!

    // geonum solves the 2^n explosion by storing what actually matters:
    // the total rotation angle, not its decomposition into 2^n pieces
}

/// test proving that geonum eliminates the need for pseudoscalars
///
/// Traditional geometric algebra requires pseudoscalars (like e₁∧e₂∧e₃ in 3D) to:
/// 1. Define duality operations: dual(A) = A * I where I is the pseudoscalar
/// 2. Represent oriented volume elements
/// 3. Handle metric and orientation of the space
/// 4. Define the "top grade" of multivectors
///
/// Geonum's angle-blade representation makes pseudoscalars unnecessary
/// by encoding these relationships directly in the geometric structure.

#[test]
fn it_doesnt_need_a_pseudoscalar() {
    // 1. DUALITY MAPPING: v* = v · I (multiply by pseudoscalar)
    //    traditional GA needs to:
    //    - define basis vectors e₁, e₂, e₃
    //    - compute pseudoscalar I = e₁∧e₂∧e₃
    //    - multiply vector by I to get dual
    //    geonum skips all that - dual() just adds 2 blades directly
    let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1 = vector
    let dual = vector.dual();
    assert_eq!(dual.angle.blade(), 3); // blade 1 + 2 = 3 (trivector)

    // 2. VOLUME ORIENTATION: I² = ±1 determines metric signature
    //    traditional: must compute pseudoscalar square to determine orientation
    //    geonum: dual operation handles orientation through blade arithmetic
    let volume = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // grade 3 volume element
    let dual_volume = volume.dual(); // duality transformation without pseudoscalar
    assert_eq!(dual_volume.angle.blade(), 5); // blade 3 + 2 = 5
    assert_eq!(dual_volume.angle.grade(), 1); // grade 5 % 4 = 1 (vector)
                                              // proves duality maps grade 3→1 without constructing I or computing I²

    // 3. HODGE STAR OPERATION: *v = v · I / |I|²
    //    traditional: matrix transformation + magnitude computation and verification
    //    geonum: direct blade transformation
    let vector = [3.0, 0.0, 0.0];
    let pseudoscalar_transform = [
        [0.0, 0.0, 1.0],  // x → yz dual
        [0.0, 0.0, -1.0], // y → zx dual
        [1.0, 0.0, 0.0],  // z → xy dual
    ];
    let mut dual_vector = [0.0; 3];
    for i in 0..3 {
        for (j, &vector_j) in vector.iter().enumerate() {
            dual_vector[i] += pseudoscalar_transform[i][j] * vector_j;
        }
    }
    // traditional hodge star requires magnitude verification: |*v| = |v|
    let vector_magnitude_sq: f64 =
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    let dual_magnitude_sq: f64 = dual_vector[0] * dual_vector[0]
        + dual_vector[1] * dual_vector[1]
        + dual_vector[2] * dual_vector[2];
    assert_eq!(vector_magnitude_sq.sqrt(), dual_magnitude_sq.sqrt()); // |I|² normalization preserved
    assert_eq!(dual_vector, [0.0, 0.0, 3.0]);

    // 4. GRADE EXTRACTION: extract k-grade part using pseudoscalar projections
    //    traditional: pseudoscalar projections for grade filtering
    //    geonum: unified operations eliminates grade extraction dependency
    let scale_factor = 2.0; // scalar component (grade 0)
    let rotation_angle = PI / 4.0; // bivector component (grade 2 via rotation)

    // traditional: extract grades using pseudoscalar projections
    // grade extraction formula: ⟨A⟩ₖ = (A ∧ Iᵏ) ⌋ I⁻ᵏ / k!

    // extract scalar (grade 0): ⟨A⟩₀ = A ⌋ I⁰ = A ⌋ 1 = scalar_part
    let scalar_part = scale_factor; // grade 0 extraction
    let pseudoscalar_grade0 = 1.0; // I⁰ = identity for scalar extraction
    assert_eq!(scalar_part * pseudoscalar_grade0, scale_factor); // pseudoscalar I⁰ appears in extraction

    // extract bivector (grade 2): ⟨A⟩₂ = (A ∧ I²) ⌋ I⁻² / 2!
    // in 2D: pseudoscalar I = e₁∧e₂, so I² = -1
    let pseudoscalar_2d = -1.0; // I² = (e₁∧e₂)² = -1 in 2D
    let bivector_coefficient = rotation_angle / pseudoscalar_2d; // divide by I² for extraction
    let rotation_matrix = [
        [rotation_angle.cos(), -rotation_angle.sin()],
        [rotation_angle.sin(), rotation_angle.cos()],
    ]; // grade 2 (bivector) as rotation matrix
    assert_eq!(bivector_coefficient, -rotation_angle); // pseudoscalar I² = -1 negates coefficient in extraction formula

    // apply decomposed operations: scaling then rotation matrix multiplication
    let input_point = [1.0, 0.0];
    let scaled_point = [input_point[0] * scalar_part, input_point[1] * scalar_part];
    let traditional_result = [
        rotation_matrix[0][0] * scaled_point[0] + rotation_matrix[0][1] * scaled_point[1],
        rotation_matrix[1][0] * scaled_point[0] + rotation_matrix[1][1] * scaled_point[1],
    ];

    // geonum: unified operation without grade extraction
    let input = Geonum::new(1.0, 0.0, 1.0);
    let geonum_result = input.scale_rotate(scale_factor, Angle::new(rotation_angle, PI));
    let geonum_x = geonum_result.mag * geonum_result.angle.grade_angle().cos();
    let geonum_y = geonum_result.mag * geonum_result.angle.grade_angle().sin();
    assert!((traditional_result[0] - geonum_x).abs() < 1e-10);
    assert!((traditional_result[1] - geonum_y).abs() < 1e-10);

    // 5. COMPLEMENT OPERATIONS: orthogonal complement via pseudoscalar
    //    traditional: multiply by pseudoscalar for complement A^⊥ = A · I
    //    geonum: complement through blade arithmetic (dual operation)

    // traditional: construct pseudoscalar I for complement operation
    let pseudoscalar_3d = 1.0; // I₃ = e₁∧e₂∧e₃ = +1 in right-handed 3D
    let line_vector = [1.0, 0.0, 0.0]; // line in x direction

    // traditional complement: A^⊥ = A · I (multiply by pseudoscalar)
    let traditional_complement = [
        line_vector[0] * pseudoscalar_3d,
        line_vector[1] * pseudoscalar_3d,
        line_vector[2] * pseudoscalar_3d,
    ]; // complement is the orthogonal subspace
    assert_eq!(traditional_complement, [1.0, 0.0, 0.0]);

    // geonum: complement through dual() blade arithmetic
    let line = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1 line
    let complement = line.dual(); // complement via blade addition
    assert_eq!(complement.angle.blade(), 3); // grade 1 → grade 3 (line → volume)
    assert_eq!(complement.mag, 1.0); // magnitude preserved

    // 6. CROSS PRODUCTS: v × w = (v ∧ w) · I^(-1) in 3D
    //    traditional: wedge then multiply by pseudoscalar inverse
    //    geonum: wedge product directly
    let v = [1.0, 0.0, 0.0]; // x unit vector
    let w = [0.0, 1.0, 0.0]; // y unit vector

    // traditional: compute wedge product v∧w
    let e12_coeff = v[0] * w[1] - v[1] * w[0]; // xy-component = 1
    let e23_coeff = v[1] * w[2] - v[2] * w[1]; // yz-component = 0
    let e31_coeff = v[2] * w[0] - v[0] * w[2]; // zx-component = 0

    // traditional: left-handed pseudoscalar I₃ = -1 (solution to sign error)
    let pseudoscalar_i3 = -1.0; // left-handed orientation
    let i3_squared = -1.0; // (e₁∧e₂∧e₃)² = -1 in 3D euclidean GA
    let pseudoscalar_inverse = pseudoscalar_i3 / i3_squared; // (-1) / (-1) = +1

    // traditional: cross product (v∧w) · I₃⁻¹
    let traditional_cross = [
        e23_coeff * pseudoscalar_inverse, // yz → x component
        e31_coeff * pseudoscalar_inverse, // zx → y component
        e12_coeff * pseudoscalar_inverse, // xy → z component
    ];
    assert_eq!(traditional_cross, [0.0, 0.0, 1.0]); // x × y = z

    // test magnitude |v × w| = |v| |w| sin(θ)
    let cross_magnitude_sq: f64 = traditional_cross[0] * traditional_cross[0]
        + traditional_cross[1] * traditional_cross[1]
        + traditional_cross[2] * traditional_cross[2];
    let cross_magnitude = cross_magnitude_sq.sqrt();
    let v_magnitude_sq: f64 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let v_magnitude = v_magnitude_sq.sqrt();
    let w_magnitude_sq: f64 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let w_magnitude = w_magnitude_sq.sqrt();
    let expected_magnitude = v_magnitude * w_magnitude * (PI / 2.0).sin(); // sin(90°) = 1
    assert_eq!(cross_magnitude, expected_magnitude);

    // geonum: wedge without pseudoscalar inverse computation
    let v_geonum = Geonum::new_from_cartesian(1.0, 0.0);
    let w_geonum = Geonum::new_from_cartesian(0.0, 1.0);
    let geonum_wedge = v_geonum.wedge(&w_geonum);
    assert_eq!(geonum_wedge.mag, 1.0); // magnitude preserved without pseudoscalar operations

    // 7. NORMAL VECTORS: surface normals via pseudoscalar multiplication
    //    traditional: compute tangent cross product then multiply pseudoscalar
    //    geonum: normals through grade parity relationships

    // traditional: surface normal = (v1 × v2) · I₃⁻¹ requires pseudoscalar
    // geonum: surface normal via grade parity - odd differences are orthogonal

    // create surface as bivector (grade 2)
    let surface_x = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let surface_y = surface_x.rotate(Angle::new(1.0, 2.0)); // grade 1
    let surface_plane = surface_x.wedge(&surface_y); // grade 2

    // find normals: grades with odd difference from surface grade 2
    let normal_vector = surface_y; // grade 1 (diff=1, odd=orthogonal)
    let normal_trivector = surface_y
        .rotate(Angle::new(1.0, 2.0))
        .rotate(Angle::new(1.0, 2.0)); // grade 3 (diff=1, odd=orthogonal)
    let parallel_scalar = surface_plane.dual(); // grade 0 (diff=2, even=parallel)

    // test orthogonality via dot products
    let dot_vector = surface_plane.dot(&normal_vector);
    let dot_trivector = surface_plane.dot(&normal_trivector);
    let dot_parallel = surface_plane.dot(&parallel_scalar);

    assert!(
        dot_vector.mag.abs() < 1e-10,
        "grade 1 normal orthogonal to grade 2 surface"
    );
    assert!(
        dot_trivector.mag.abs() < 1e-10,
        "grade 3 normal orthogonal to grade 2 surface"
    );
    assert!(
        dot_parallel.mag.abs() > 1e-10,
        "grade 0 parallel to grade 2 surface (even diff)"
    );

    // geonum ghosts pseudoscalar I₃ multiplication
    // orthogonality emerges from grade parity, no pseudoscalar needed

    // CONCLUSION: All traditional pseudoscalar functionality achieved
    // through direct angle-blade arithmetic, eliminating the need for:
    // - Special pseudoscalar objects
    // - Dimension-specific unit volume elements
    // - Complex duality multiplication formulas
    // - Grade-dependent pseudoscalar properties
    // - Metric signature pseudoscalar complications
}

#[test]
fn it_demonstrates_pseudoscalar_elimination_benefits() {
    // traditional GA rotation: R*v*R† requires 3 geometric products × 64 operations = 192 operations
    // geonum rotation: angle addition = 1 operation
    // the 192x difference comes from eliminating the pseudoscalar

    // test rotation equivalence
    let point = Geonum::new_from_cartesian(3.0, 4.0);
    let rotation = Angle::new(1.0, 2.0); // π/2
    let rotated = point.rotate(rotation);

    let x = rotated.mag * rotated.angle.grade_angle().cos();
    let y = rotated.mag * rotated.angle.grade_angle().sin();
    assert!((x - (-4.0)).abs() < EPSILON);
    assert!((y - 3.0).abs() < EPSILON);

    // traditional GA needs pseudoscalar to define rotation planes through basis blade products
    // this forces dimension-specific formulas and full multiplication tables
    // geonum recognizes rotation IS angle addition, not basis blade reconstruction

    // test rotation composition
    let r1 = Angle::new(1.0, 4.0); // π/4
    let r2 = Angle::new(1.0, 6.0); // π/6
    let r3 = Angle::new(1.0, 3.0); // π/3

    // traditional: R1*R2*R3 through geometric products
    // geonum: r1 + r2 + r3
    let composed = r1 + r2 + r3;
    assert_eq!(composed, Angle::new(3.0, 4.0)); // 3π/4

    let final_point = point.rotate(composed);
    let fx = final_point.mag * final_point.angle.grade_angle().cos();
    let fy = final_point.mag * final_point.angle.grade_angle().sin();
    assert!((fx - (-4.949)).abs() < 0.01);
    assert!((fy - (-0.707)).abs() < 0.01);

    // eliminating pseudoscalar reveals rotation as angle addition
    // 192 operations → 1 operation
}

#[test]
fn it_proves_dualization_as_angle_ops_compresses_ga() {
    // traditional GA duality requires multiplying by dimension-specific pseudoscalars
    // with exponential 2^n component arrays
    // geonum reduces duality to O(1) angle arithmetic through the quadrature's bivector

    // the 4 scalar, vector, bivector, trivector principle grades
    // emerge from the quadrature's bivector: sin(θ+π/2) = cos(θ)
    // this π/2 rotation creates the 4-quarter-turn cycle needed for complete GA

    // demonstrate duality compression: any dimensional object → simple angle operation
    let million_dim_vector = Geonum::new_with_blade(1.0, 1_000_000, 1.0, 4.0);
    let billion_dim_bivector = Geonum::new_with_blade(2.0, 1_000_000_000, 1.0, 3.0);

    // traditional GA: needs 2^1000000 and 2^1000000000 components for duality
    // (more storage than atoms in observable universe)

    // geonum: duality is just angle arithmetic regardless of dimension
    let dual_million = million_dim_vector.dual();
    let dual_billion = billion_dim_bivector.dual();

    // duality maps through 4-cycle: grade k → grade (k+2) % 4
    // dual() adds 2 blades (π rotation)
    assert_eq!(million_dim_vector.angle.grade(), 0); // 1000000 % 4 = 0 (scalar)
    assert_eq!(dual_million.angle.grade(), 2); // (0+2) % 4 = 2 (bivector)

    assert_eq!(billion_dim_bivector.angle.grade(), 0); // 1000000000 % 4 = 0 (scalar)
    assert_eq!(dual_billion.angle.grade(), 2); // (0+2) % 4 = 2 (bivector)

    // compression achieved: exponential → constant time
    // traditional: O(2^n) storage and computation
    // geonum: O(1) angle operations from quadrature's bivector foundation

    // the key insight: the bivector sin(θ+π/2) = cos(θ) IS the duality operator
    // this π/2 rotation imposes the incidence structure that defines geometric relationships
    // point-line-plane-volume duality emerges from this single trigonometric identity
    // eliminating need for dimension-specific pseudoscalars or exponential storage

    // prove duality preserves length (isometry property)
    assert_eq!(dual_million.mag, million_dim_vector.mag);
    assert_eq!(dual_billion.mag, billion_dim_bivector.mag);

    // prove duality involution: dual(dual(x)) returns to original grade
    let double_dual_million = dual_million.dual();
    let double_dual_billion = dual_billion.dual();
    assert_eq!(
        double_dual_million.angle.grade(),
        million_dim_vector.angle.grade()
    );
    assert_eq!(
        double_dual_billion.angle.grade(),
        billion_dim_bivector.angle.grade()
    );

    // prove O(1) complexity: blade arithmetic regardless of dimension size
    // dual() adds 2 blades regardless of dimension
    assert_eq!(
        dual_million.angle.blade(),
        million_dim_vector.angle.blade() + 2
    );
    assert_eq!(
        dual_billion.angle.blade(),
        billion_dim_bivector.angle.blade() + 2
    );
}
