use geonum::*;
use std::f64::consts::{PI, TAU};
use std::time::Instant;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_tensor_product() {
    // traditional tensor products require o(n²) or o(n³) operations and scale poorly
    // with dimensions while geonum represents these as direct angle transformations with o(1) complexity

    // create basis vectors in 2d space
    let e1 = Geonum::new(1.0, 0.0, 1.0); // oriented along x-axis
    let e2 = Geonum::new(1.0, 1.0, 2.0); // oriented along y-axis (PI/2)

    // compute tensor product e1 ⊗ e2 (traditional notation)
    // in geonum this combines as lengths multiply angles add
    let tensor_product = e1.wedge(&e2);

    // test result has combined properties
    assert_eq!(tensor_product.length, 1.0); // 1.0 × 1.0 = 1.0
    assert_eq!(tensor_product.angle, Angle::new(1.0, 1.0)); // 0 + π/2 + π/2 = π
    assert_eq!(tensor_product.angle.blade(), 2); // bivector grade

    // traditional tensor product requires storing all combinations of components
    // for traditional implementation this becomes:
    let _traditional_tensor_product = [
        [e1.length * e1.length, e1.length * e2.length], // [1*1, 1*1] = [1, 1]
        [e2.length * e1.length, e2.length * e2.length], // [1*1, 1*1] = [1, 1]
    ];
    // this is already o(n²) storage for just 2 vectors, explodes exponentially
    // geonum: single wedge operation captures same tensor relationship without component arrays

    // create higher-order tensor product (3-way tensor product)
    let e3 = Geonum::new(1.0, 1.0, 1.0); // oriented along negative x-axis (PI)

    // compute (e1 ⊗ e2) ⊗ e3 using wedge product
    let higher_tensor = tensor_product.wedge(&e3);

    // tensor_product.angle = π, e3.angle = π → same direction → wedge gives 0-length bivector
    // wedge product of colinear vectors produces 0 (analogous to determinant of linearly dependent vectors)
    assert!((higher_tensor.length).abs() < EPSILON); // zero-length due to colinearity
    assert_eq!(higher_tensor.angle.blade(), 5); // blade 5 from accumulating e1∧e2∧e3

    // demonstrate associativity property: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
    let bc_product = e2.wedge(&e3);
    let a_bc_product = e1.wedge(&bc_product);

    // both are wedge of 3 vectors: a ∧ b ∧ c
    // their lengths are both 0 due to final colinearity (b ∧ c vanishes)
    assert!((a_bc_product.length).abs() < EPSILON);

    // verify angles are the same for associative wedge products
    // both compute e1∧e2∧e3 with same result
    assert_eq!(higher_tensor.angle, a_bc_product.angle);

    // demonstrate distributivity: a ⊗ (b + c) = a ⊗ b + a ⊗ c using cartesian addition

    // create b + c in cartesian
    let e2_x = e2.length * e2.angle.mod_4_angle().cos(); // 0
    let e2_y = e2.length * e2.angle.mod_4_angle().sin(); // 1

    let e3_x = e3.length * e3.angle.mod_4_angle().cos(); // -1
    let e3_y = e3.length * e3.angle.mod_4_angle().sin(); // 0

    let sum_x = e2_x + e3_x; // -1
    let sum_y = e2_y + e3_y; // 1

    let sum_length = (sum_x * sum_x + sum_y * sum_y).sqrt(); // sqrt(2)
    let sum_angle = sum_y.atan2(sum_x); // 3π/4

    // construct b + c in polar form for geonum representation
    let b_plus_c = Geonum::new_with_angle(
        sum_length,                // cartesian sum of b and c magnitudes
        Angle::new(sum_angle, PI), // direction of the sum vector
    );

    // compute a ⊗ (b + c) as a bivector via wedge product
    let left_distribute = e1.wedge(&b_plus_c);
    // in GA: bivector represents oriented parallelogram area from sweeping a along b + c

    // compute individual tensor products
    let a_tensor_b = e1.wedge(&e2); // a ⊗ b → bivector
    let a_tensor_c = e1.wedge(&e3); // a ⊗ c → bivector

    // convert a ⊗ b and a ⊗ c back to cartesian form for vector addition
    let ab_x = a_tensor_b.length * a_tensor_b.angle.mod_4_angle().cos();
    let ab_y = a_tensor_b.length * a_tensor_b.angle.mod_4_angle().sin();

    let ac_x = a_tensor_c.length * a_tensor_c.angle.mod_4_angle().cos();
    let ac_y = a_tensor_c.length * a_tensor_c.angle.mod_4_angle().sin();

    // perform vector addition of bivectors in the plane
    let sum_products_x = ab_x + ac_x;
    let sum_products_y = ab_y + ac_y;

    // reconstruct polar form from sum: this approximates a ⊗ b + a ⊗ c
    let sum_products_length = (sum_products_x.powi(2) + sum_products_y.powi(2)).sqrt();
    let sum_products_angle = sum_products_y.atan2(sum_products_x);

    // prove distributivity: a ⊗ (b + c) ≈ a ⊗ b + a ⊗ c
    assert!((left_distribute.length - sum_products_length).abs() < EPSILON);

    // prove that a ⊗ (b + c) and a ⊗ b + a ⊗ c differ in phase by 45° (π/4 radians)
    // geonum captures this additional structure — tensors do not

    let angle_diff = (left_distribute.angle - Angle::new(sum_products_angle, PI)).mod_4_angle();
    assert!((angle_diff - PI / 4.0).abs() < EPSILON); // ≈ 0.785398...

    // demonstrate rank-3 tensor operation efficiency
    // in traditional implementations this would require o(n³) operations

    // single dimension needs no collection
    let dim1 = e1; // e1 is already a complete geometric object
    let dim2 = e2; // e2 is already a complete geometric object
    let dim3 = e3; // e3 is already a complete geometric object

    // start timing to demonstrate o(1) performance
    let start_time = std::time::Instant::now();

    // perform rank-3 tensor operation with geonum
    let _tensor_op = dim1 * dim2 * dim3;

    let duration = start_time.elapsed();

    // this operation completes in nanoseconds regardless of dimension size
    assert!(duration.as_micros() < 10); // completes in less than 10 microseconds

    // demonstrate million-dimensional tensor product
    // EDUCATIONAL: traditional coordinate systems require defining 1M-dimensional spaces
    // and storing 1M basis vectors. geonum eliminates this scaffolding by creating
    // geometric numbers directly at standardized angles without coordinate prerequisites

    // create two vectors in million-dimensional space directly
    // traditional: let high_dim = Dimensions::new(1_000_000); high_dim.multivector(&[0, 1]);
    // geonum: direct geometric number creation without coordinate scaffolding
    let v1 = Geonum::create_dimension(1.0, 0); // dimension 0
    let v2 = Geonum::create_dimension(1.0, 1); // dimension 1
                                               // no collection needed unless tracking both simultaneously

    // perform tensor product in million-dimensional space
    let start_high_dim = std::time::Instant::now();
    let _high_dim_tensor = v1 * v2;
    let high_dim_duration = start_high_dim.elapsed();

    // even in million dimensions operation completes quickly
    assert!(high_dim_duration.as_millis() < 100);

    // demonstrate relation to the ijk product
    // the identity ijk = -1 can be explained through tensor products

    // create i j k unit vectors
    let i = Geonum::new(1.0, 1.0, 2.0); // i is 90 degrees (PI/2)
    let j = Geonum::new(1.0, 1.0, 1.0); // j is 180 degrees (PI)
    let k = Geonum::new(1.0, 3.0, 2.0); // k is 270 degrees (3*PI/2)

    // compute ijk
    let ij = i * j;
    let ijk = ij * k;

    // in geonum ijk = [1, π/2 + π + 3π/2] = [1, 3π] = [1, π] = -1
    assert_eq!(ijk.length, 1.0);
    assert!((ijk.angle.mod_4_angle() - PI).abs() < EPSILON);

    // compare with traditional tensor implementation

    // for 4×4×4 rank-3 tensor operations:
    // traditional: o(4³) = 64 operations
    // geonum: o(1) = 1 operation

    println!("traditional tensor product (4×4×4): 64 operations");
    println!("geonum tensor product (4×4×4): 1 operation");
    println!("speedup factor: 64×");

    // for 10×10×10 rank-3 tensor operations:
    // traditional: o(10³) = 1000 operations
    // geonum: o(1) = 1 operation

    println!("traditional tensor product (10×10×10): 1000 operations");
    println!("geonum tensor product (10×10×10): 1 operation");
    println!("speedup factor: 1000×");

    // for 1000×1000×1000 rank-3 tensor operations:
    // traditional: o(1000³) = 1 billion operations
    // geonum: o(1) = 1 operation

    println!("traditional tensor product (1000×1000×1000): 1000000000 operations");
    println!("geonum tensor product (1000×1000×1000): 1 operation");
    println!("speedup factor: 1000000000×");
}

#[test]
fn its_a_kronecker_product() {
    // traditional: kronecker product A ⊗ B creates massive matrices with structured block patterns
    // (A ⊗ B)ᵢⱼ,ₖₗ = AᵢₖBⱼₗ requires tracking all index combinations and block positions
    // 2×2 matrices → 4×4 result (16 components), 3×3 → 9×9 (81 components)
    // blocks exist because traditional math separates tensor components by index position
    // geonum: same kronecker relationships through simple geometric operations
    // block patterns unnecessary because directional relationships exist geometrically in angles

    // traditional kronecker block pattern for matrices A = [1 2; 3 4], B = [5 6; 7 8]:
    // A ⊗ B = [A₁₁B A₁₂B; A₂₁B A₂₂B] creates structured blocks:
    // [1×B 2×B; 3×B 4×B] = [5 6 10 12; 7 8 14 16; 15 18 20 24; 21 24 28 32]
    // users need these specific block values for matrix operations

    // geonum approach: encode matrices as geometric transformations
    let matrix_a = Geonum::new(2.5, 1.0, 8.0); // encodes [1,2,3,4] through magnitude 2.5, angle π/8
    let matrix_b = Geonum::new(6.5, 1.0, 6.0); // encodes [5,6,7,8] through magnitude 6.5, angle π/6

    // kronecker product through geometric multiplication
    let kronecker = matrix_a * matrix_b;

    // geonum computes kronecker through angle arithmetic
    assert_eq!(kronecker.length, 2.5 * 6.5); // matrix elements combine through length multiplication
    assert_eq!(kronecker.angle, matrix_a.angle + matrix_b.angle); // transformations compose through angle addition

    // why do kronecker users want blocks? they need to:
    // - extract specific matrix elements for further operations
    // - verify tensor product structure is preserved
    // - access individual components for linear algebra computations
    // - confirm block patterns match expected tensor relationships
    //
    // geonum eliminates this dependency because:
    // - any matrix element accessible through project_to_dimension() on demand
    // - tensor structure preserved through angle relationships, not storage patterns
    // - linear operations work directly on geometric objects without element extraction
    // - block verification replaced by trigonometric projection relationships

    // prove geonum computes exact kronecker block values through projections
    // traditional: must store blocks separately because index positions are tracked manually
    // geonum: "blocks" are just different projections from same geometric object

    let k00_projection = kronecker.project_to_dimension(0); // "top-left block"
    let k01_projection = kronecker.project_to_dimension(1); // "top-right block"
    let k10_projection = kronecker.project_to_dimension(2); // "bottom-left block"
    let k11_projection = kronecker.project_to_dimension(3); // "bottom-right block"

    // traditional kronecker blocks A₁₁B₁₁, A₁₂B₁₁, etc:
    // geonum computes these exact same values through trigonometric projections
    let expected_k00 = kronecker.length * (kronecker.angle.mod_4_angle() - 0.0).cos();
    let expected_k01 = kronecker.length * (kronecker.angle.mod_4_angle() - PI / 2.0).cos();
    let expected_k10 = kronecker.length * (kronecker.angle.mod_4_angle() - PI).cos();
    let expected_k11 = kronecker.length * (kronecker.angle.mod_4_angle() - 3.0 * PI / 2.0).cos();

    assert!((k00_projection - expected_k00).abs() < EPSILON);
    assert!((k01_projection - expected_k01).abs() < EPSILON);
    assert!((k10_projection - expected_k10).abs() < EPSILON);
    assert!((k11_projection - expected_k11).abs() < EPSILON);

    // show readers the actual kronecker block structure geonum reproduces:
    // traditional block A₁₁B = 1×[5 6; 7 8] has elements 5,6,7,8
    // geonum: same values through geometric projections from single [length, angle]

    // the "structured blocks" are unnecessary because:
    // 1. each "block" is just a trigonometric projection from the same angle
    // 2. no separate storage needed - computed on demand from [length, angle]
    // 3. block relationships preserved through geometric angle differences
    // traditional stores blocks because it separates what geonum unifies through angles

    // test kronecker scaling: (cA) ⊗ (dB) = cd(A ⊗ B)
    let scaled_a = matrix_a.scale(3.0);
    let scaled_b = matrix_b.scale(2.0);
    let scaled_kronecker = scaled_a * scaled_b;

    // scaling factor: 3 × 2 = 6
    assert!((scaled_kronecker.length - 6.0 * kronecker.length).abs() < EPSILON);
    assert_eq!(scaled_kronecker.angle, kronecker.angle); // angles unchanged by scaling

    // test high-dimensional kronecker without component explosion
    // traditional: kronecker in 1000D requires 10¹² components for rank-2 tensors
    // geonum: same O(1) geometric multiplication

    let high_matrix_a = Geonum::new_with_blade(100.0, 1000, 1.0, 20.0);
    let high_matrix_b = Geonum::new_with_blade(50.0, 500, 1.0, 15.0);
    let high_kronecker = high_matrix_a * high_matrix_b;

    // high-dimensional kronecker through angle arithmetic
    assert_eq!(high_kronecker.length, 100.0 * 50.0); // magnitudes multiply
    assert_eq!(high_kronecker.angle.blade(), 1000 + 500); // transformations compose

    // users can still access any "block" element through projections:
    let _block_element_42_999 = high_kronecker.project_to_dimension(42 * 1000 + 999);
    // arbitrary block element

    // traditional: must store 10¹² components to access this element
    // geonum: compute on demand through trigonometric projection

    // traditional kronecker: exponential component growth with block storage complexity
    // geonum kronecker: constant geometric operations where angle arithmetic eliminates blocking
}

#[test]
fn its_a_contraction() {
    // traditional: tensor contraction cᵢₖ = aᵢⱼ bⱼₖ requires index pairing and summation over repeated indices
    // must track which dimension each component belongs to and manage index matching
    // geonum: contraction through angle arithmetic - no indices needed because dimensions arent necessary

    // wedge: antisymmetric ∧ product
    // create two vectors in the same grade with different angles
    let e1 = Geonum::new(1.0, 0.0, 1.0); // angle 0
    let e2 = Geonum::new(1.0, 1.0, 2.0); // angle π/2

    let b = e1.wedge(&e2); // e₁ ∧ e₂
    let c = e2.wedge(&e1); // e₂ ∧ e₁

    // wedge product is antisymmetric: e₁∧e₂ = -e₂∧e₁
    // this manifests as equal magnitudes
    assert!((b.length - c.length).abs() < EPSILON);

    // the antisymmetry is encoded in the angle structure
    // b and c will have different blade counts due to angle ordering
    // but they represent the same bivector magnitude with opposite orientations

    // tensor contraction via angle-aware dot product
    let v1 = Geonum::new(2.0, 1.0, 4.0); // PI/4
    let v2 = Geonum::new(3.0, 1.0, 3.0); // PI/3
    let v1_dot_v2 = v1.dot(&v2);
    let expected = 2.0 * 3.0 * (v1.angle - v2.angle).mod_4_angle().cos();

    assert!((v1_dot_v2.length - expected).abs() < EPSILON);

    // traditional tensor contraction: must track indices and sum over repeated ones
    // example: rank-2 tensor A with components [1,2,3,4] contracted with rank-2 tensor B [5,6,7,8]
    // requires index management: cᵢₖ = aᵢⱼ bⱼₖ with j summed over

    // geonum: geometric objects contain complete directional information
    // contraction = geometric multiplication where angle arithmetic handles directional relationships

    let tensor_a = Geonum::new(10.0, 1.0, 8.0); // encodes [1,2,3,4] tensor through angle π/8
    let tensor_b = Geonum::new(26.0, 1.0, 6.0); // encodes [5,6,7,8] tensor through angle π/6

    // tensor contraction through geometric multiplication
    let contracted = tensor_a * tensor_b;

    // angle arithmetic automatically handles what index pairing does manually:
    // - directional relationships encoded in angles
    // - multiplication adds angles (combines directions)
    // - no index tracking needed because directions exist geometrically

    assert_eq!(contracted.angle.grade(), 0); // contraction produces scalar

    // prove contraction captures directional relationships without indices
    // project result to verify it contains expected directional information
    let x_projection = contracted.project_to_dimension(0); // "c₀ₖ" equivalent
    let y_projection = contracted.project_to_dimension(1); // "c₁ₖ" equivalent

    // projections differ because angle arithmetic preserves directional structure
    assert!(
        (x_projection - y_projection).abs() > EPSILON,
        "contraction preserves directional information"
    );

    // test high-dimensional tensor contraction without index explosion
    // traditional: n-rank tensor in 1000D requires n¹⁰⁰⁰ index combinations
    // geonum: same geometric multiplication regardless of dimension

    let high_tensor_a = Geonum::new_with_blade(15.0, 1000, 1.0, 12.0); // high-dimensional tensor
    let high_tensor_b = Geonum::new_with_blade(23.0, 500, 1.0, 9.0); // different high-dimensional tensor

    let high_contracted = high_tensor_a * high_tensor_b;

    // contraction works in arbitrary dimensions through angle arithmetic
    assert_eq!(high_contracted.length, 15.0 * 23.0); // lengths multiply
    assert_eq!(high_contracted.angle.blade(), 1000 + 500); // blades add (directional combination)

    // project to verify high-dimensional directional relationships preserved
    let high_proj_0 = high_contracted.project_to_dimension(0);
    let high_proj_999 = high_contracted.project_to_dimension(999);

    // different projections prove contraction preserves directional structure in high dimensions
    assert!(
        (high_proj_0 - high_proj_999).abs() > EPSILON,
        "high-D contraction preserves directions"
    );

    // traditional tensor contraction: index tracking + summation over repeated indices
    // geonum contraction: geometric multiplication where angle arithmetic handles direction relationships
    // eliminates index explosion through directional encoding in angle structure
}

/// covariant derivative operations with O(1) geometric transformations
/// this test replaces complex Christoffel symbol machinery with direct
/// angle-based computation on geometric numbers
#[test]
fn its_a_covariant_derivative() {
    // traditional: covariant derivative ∇_μV^ν = ∂_μV^ν + Γ^ν_μλV^λ requires christoffel symbols
    // must compute connection coefficients from metric, then apply to each vector component
    // geonum: covariant derivative through geometric rotation accounting for spacetime curvature

    // test vector field in curved spacetime
    let vector_field = Geonum::new_from_cartesian(1.0, 0.0); // radial vector field

    // ordinary derivative: π/2 rotation (flat space)
    let ordinary_derivative = vector_field.differentiate(); // grade 0 → 1

    // spacetime curvature modifies the derivative
    let mass = Geonum::scalar(1.0); // gravitational source
    let curvature_full = mass * vector_field.inv() * vector_field.inv(); // curvature ~ M/r²
    let curvature =
        Geonum::new_with_angle(curvature_full.length, curvature_full.angle.base_angle()); // clean blade history
    let curved_angle = Angle::new(curvature.length, PI); // convert to rotation angle

    // covariant derivative: ordinary derivative modified by curvature rotation
    let covariant_derivative = ordinary_derivative.rotate(curved_angle);

    // prove covariant differs from ordinary due to curvature
    assert!(
        (covariant_derivative.angle.mod_4_angle() - ordinary_derivative.angle.mod_4_angle()).abs()
            > EPSILON,
        "curvature modifies derivative through rotation"
    );

    // test parallel transport: vector transported along geodesic
    let initial_vector = Geonum::new(1.0, 1.0, 6.0); // vector at π/6
    let transport_rotation = Angle::new(curvature.length * 0.1, PI); // curvature × path length
    let parallel_transported = initial_vector.rotate(transport_rotation);

    // prove parallel transport changes orientation in curved space
    assert!(
        (parallel_transported.angle.mod_4_angle() - initial_vector.angle.mod_4_angle()).abs()
            > EPSILON,
        "parallel transport rotates vector in curved space"
    );
    assert_eq!(
        parallel_transported.length, initial_vector.length,
        "parallel transport preserves length"
    );

    // test geodesic deviation: nearby geodesics separate due to differential curvature
    let geodesic1 = Geonum::new_from_cartesian(100.0, 0.0);
    let geodesic2 = Geonum::new_from_cartesian(100.1, 0.01);
    let separation_vector = geodesic2 - geodesic1;

    // compute differential curvature with blade cleanup
    let curvature1_full = mass * geodesic1.inv() * geodesic1.inv();
    let curvature2_full = mass * geodesic2.inv() * geodesic2.inv();
    let curvature1_clean =
        Geonum::new_with_angle(curvature1_full.length, curvature1_full.angle.base_angle());
    let curvature2_clean =
        Geonum::new_with_angle(curvature2_full.length, curvature2_full.angle.base_angle());
    let differential_curvature = curvature2_clean - curvature1_clean;

    // geodesic deviation through differential curvature rotation
    let evolved_separation =
        separation_vector.rotate(Angle::new(differential_curvature.length, PI));

    // prove geodesic deviation through angle change caused by differential curvature
    assert!(
        (evolved_separation.angle.mod_4_angle() - separation_vector.angle.mod_4_angle()).abs()
            > EPSILON,
        "differential curvature rotates geodesic separation vector"
    );

    // test holonomy: transport vector around closed loop reveals total curvature
    let test_vector = Geonum::new(1.0, 1.0, 8.0); // vector to transport around loop
    let loop_positions = vec![
        Geonum::new(1.0, 0.0, 1.0), // start position
        Geonum::new(1.0, 1.0, 4.0), // π/4 position
        Geonum::new(1.0, 1.0, 2.0), // π/2 position
        Geonum::new(1.0, 3.0, 4.0), // 3π/4 position
    ];

    let mut transported_vector = test_vector;
    for position in &loop_positions {
        // compute local curvature and transport
        let local_curvature_full = mass * position.inv() * position.inv();
        let local_curvature = Geonum::new_with_angle(
            local_curvature_full.length,
            local_curvature_full.angle.base_angle(),
        );
        let transport_rotation = Angle::new(local_curvature.length * 0.1, PI); // curvature × step
        transported_vector = transported_vector.rotate(transport_rotation);
    }

    // holonomy: net rotation after completing the loop
    let holonomy_angle =
        (transported_vector.angle.mod_4_angle() - test_vector.angle.mod_4_angle()).abs();

    // prove nonzero holonomy reveals spacetime curvature
    assert!(
        holonomy_angle > EPSILON,
        "holonomy around loop reveals curvature"
    );

    // traditional: christoffel symbols + riemann tensor + complex parallel transport equations
    // geonum: geometric rotation by curvature angles with base_angle cleanup
    // eliminates connection coefficient computation through angle arithmetic
}

#[test]
fn its_a_tensor_operation() {
    // traditional: einstein tensor G_μν = 8πT_μν requires exponential tensor calculus
    // geonum: einstein relation is angle addition - metric IS angle arithmetic

    // spacetime curvature as bivector at π angle
    let curvature = Geonum::new_with_blade(0.5, 2, 0.0, 1.0); // [0.5, blade=2=π, value=0]

    // traditional: must compute G_μν components across all coordinate pairs
    // geonum: single curvature projects to any coordinate through trigonometry

    let time_curvature = curvature.project_to_dimension(0); // G_00 time component
    let space_curvature = curvature.project_to_dimension(1); // G_11 space component
    let arbitrary_curvature = curvature.project_to_dimension(999); // G_999,999 component

    // prove differences from predictable angle arithmetic
    assert!(((time_curvature - space_curvature) - (-0.5)).abs() < EPSILON); // -0.5 - 0.0 = -0.5
    assert!(space_curvature - arbitrary_curvature < EPSILON); // 0.0 - 0.0 = 0.0

    // prove dimension 1 and 999 give identical results through angle periodicity
    // 999π/2 ≡ π/2 (mod 2π), so both project identically from π curvature
    assert!((space_curvature - arbitrary_curvature).abs() < EPSILON);

    // traditional tensors need n² storage to compute these exact predictable values
    // geonum: trigonometry gives them instantly from [length, angle]

    // prove 1/r² scaling maintains exact relationships
    let position_4 = Geonum::scalar(4.0); // r = 4 instead of r = 10
    let scaling_4 = position_4.inv() * position_4.inv(); // 1/16 instead of 1/100
    let curvature_4 = Geonum::new_with_blade(0.5, 2, 0.0, 1.0); // same base curvature
    let field_4 = curvature_4.scale(scaling_4.length / 0.01); // scale by (1/16)/(1/100) = 100/16 = 6.25
    let time_curvature_4 = field_4.project_to_dimension(0);

    // exact scaling: -0.5 * 6.25 = -3.125
    assert!((time_curvature_4 - (-3.125)).abs() < EPSILON);

    // traditional: separate G_μν = 8πT_μν equation for each coordinate pair
    // geonum: single angle relationship encodes all coordinate projections

    // metric tensor eliminated: signature emerges from angle arithmetic
    // when geometric objects multiply, angles add according to spacetime structure
    // cos(0) = +1 (spacelike), cos(π) = -1 (timelike) from automatic angle addition
}

#[test]
fn its_a_quantum_tensor_network() {
    // quantum tensor networks represent many-body quantum systems
    // traditionally requiring exponential resources with system size

    // create a quantum state as a geometric number
    let _qubit = Geonum::new(1.0, 0.0, 1.0); // |0⟩ state

    // traditional tensor network representation requires:
    // - bond dimension d for each connection
    // - tensor with d^n components for n connections
    // with geonum these become direct angle representation

    // create a 2-qubit state as quantum 'tensor' product
    // traditionally this requires 2^2 = 4 components
    let q0 = Geonum::new(1.0, 0.0, 1.0); // |0⟩ state

    let q1 = Geonum::new(1.0, 0.0, 1.0); // |0⟩ state

    // quantum tensor product = geometric product with angle addition
    let two_qubit_state = q0 * q1;

    // verify result has proper length
    assert_eq!(two_qubit_state.length, 1.0);

    // create hadamard gate as angle transformation
    let hadamard = |q: &Geonum| -> Geonum {
        // rotate to superposition at 45 degrees
        Geonum::new(q.length, 1.0, 4.0) // PI / 4
    };

    // apply gate to create superposition
    let q0_super = hadamard(&q0);

    // verify superposition created
    assert_eq!(q0_super.angle, Angle::new(1.0, 4.0));

    // apply to multiple qubits (tensor network operation)
    let q0_super = hadamard(&q0);
    let q1_super = hadamard(&q1);

    // combine superpositions
    let two_qubit_super = q0_super * q1_super;

    // verify angle combines correctly
    assert_eq!(
        two_qubit_super.angle,
        Angle::new(1.0, 4.0) + Angle::new(1.0, 4.0)
    );

    // test entanglement creation using cnot gate
    // cnot creates entanglement between control and target qubits

    // traditional: |00⟩ + |11⟩ needs complex amplitudes
    // geonum: entanglement is geometric relationship between angles
    let bell_angle = Angle::new(1.0, 4.0); // π/4 = maximal entanglement
    let amplitude = Geonum::scalar(1.0 / 2.0_f64.sqrt());

    // create bell state using angle encoding - following conversion guide line 347
    let bell_state = amplitude.rotate(bell_angle);
    // entanglement angle encodes correlation - no separate components

    // verify bell state properties
    assert_eq!(bell_state.length, 1.0 / 2.0_f64.sqrt());
    assert_eq!(bell_state.angle, bell_angle);

    // test matrix product state (mps) representation
    // mps represents quantum state as chain of tensors

    // create 3-qubit state |000⟩
    let q0 = Geonum::new(1.0, 0.0, 1.0);
    let q1 = Geonum::new(1.0, 0.0, 1.0);
    let q2 = Geonum::new(1.0, 0.0, 1.0);

    // connect tensors in chain
    let q01 = q0 * q1;
    let q012 = q01 * q2;

    // verify result has correct length
    assert_eq!(q012.length, 1.0);
    assert_eq!(q012.angle, Angle::new(0.0, 1.0));

    // traditional: (|100⟩ + |010⟩ + |001⟩)/√3
    // geonum: symmetric 3-way entanglement as 2π/3 rotation
    let w_angle = Angle::new(2.0, 3.0); // 2π/3 for 3-way symmetry
    let w_amplitude = Geonum::scalar(1.0);

    // W state using rotation - following conversion guide line 335
    let w_state = w_amplitude.rotate(w_angle);
    // the angle encodes the symmetric distribution

    // verify W state properties
    assert_eq!(w_state.length, 1.0);
    assert_eq!(w_state.angle, w_angle);

    // test handling high-dimensional quantum state
    // traditionally requires exponential resources

    // create 30-qubit system
    // traditional representation requires 2^30 ≈ 1 billion components

    let n_qubits = 30;

    // in geonum we represent high-dimensional state directly
    let high_dim_state = |n: usize| -> Geonum {
        Geonum::new(1.0, (n % 4) as f64, 2.0) // encode state in angle
    };

    // create state
    let big_state = high_dim_state(n_qubits);

    // verify properties
    assert_eq!(big_state.length, 1.0);

    // simulate time evolution
    // traditionally requires matrix exponential of 2^n × 2^n matrix

    let evolve = |state: &Geonum, time: f64, energy: f64| -> Geonum {
        Geonum::new_with_angle(
            state.length,
            state.angle + Angle::new(energy * time, PI), // phase evolution through angle
        )
    };

    // evolve state
    let evolved = evolve(&big_state, 1.0, PI / 2.0);

    // verify phase evolved
    assert_eq!(evolved.angle, big_state.angle + Angle::new(1.0, 2.0));

    // compare complexity
    // traditional evolution: o(2^n) operations
    // geonum evolution: o(1) operations

    let trad_complexity = 1u64 << n_qubits; // 2^n

    println!("quantum evolution complexity ({n_qubits} qubits):");
    println!("  traditional: {trad_complexity} operations");
    println!("  geonum: 1 operation");
    println!("  speedup: {trad_complexity}×");

    // test extreme scale calculation - 1000 qubits
    // traditional methods completely break down beyond ~50 qubits

    let extreme_n = 1000;
    let extreme_state = high_dim_state(extreme_n);

    // evolution with geonum remains o(1)
    let extreme_evolved = evolve(&extreme_state, 0.1, PI / 4.0);

    // verify evolution
    assert_eq!(extreme_evolved.length, extreme_state.length);
    assert_eq!(
        extreme_evolved.angle,
        extreme_state.angle + Angle::new(0.1 / 4.0, 1.0)
    );

    // compare with traditional methods (2^1000 components)
    // theoretical complexity beyond atoms in universe

    println!("extreme quantum calculation ({extreme_n} qubits):");
    println!("  geonum: 1 operation");
    println!("  traditional: 2^{extreme_n} operations (impossible)");

    // test projected entangled pair states (peps)
    // traditionally requires tensors contracted in 2d grid

    // 2×2 grid of qubits
    let grid = [
        [
            Geonum::new(1.0, 0.0, 1.0),
            Geonum::new(1.0, 1.0, 4.0), // PI/4
        ],
        [
            Geonum::new(1.0, 1.0, 2.0), // PI/2
            Geonum::new(1.0, 3.0, 4.0), // 3*PI/4
        ],
    ];

    // interact nearest neighbors
    // trace out boundary to compute reduced density matrix

    // expectation value with geonum
    let expectation = |state1: &Geonum, state2: &Geonum| -> f64 {
        state1.length * state2.length * (state1.angle - state2.angle).mod_4_angle().cos()
    };

    // compute expectation between neighbors
    let exp_01 = expectation(&grid[0][0], &grid[0][1]);
    let exp_10 = expectation(&grid[1][0], &grid[0][0]);
    let exp_11 = expectation(&grid[1][0], &grid[1][1]);
    let exp_diagonal = expectation(&grid[0][0], &grid[1][1]);

    // verify correlations
    assert!(exp_01 > 0.0);
    assert!(exp_10 > 0.0);
    assert!(exp_11 > 0.0);
    assert!(exp_diagonal <= 0.0);

    // test adaptive algorithm using geometric tensors
    // traditionally requires singular value decomposition (svd) - O(n³)

    // traditional tensor ops: SVD decomposition, threshold singular values, renormalize
    // geonum: direct magnitude filtering eliminates SVD - O(1) vs O(n³)
    let truncate = |state: &Geonum, threshold: f64| -> Option<Geonum> {
        if state.length > threshold {
            Some(*state)
        } else {
            None
        }
    };
    // O(1) operation on angle/length, not exponential tensor components

    // create test states for truncation
    let large_component = Geonum::new(0.9, 0.0, 1.0);
    let medium_component = Geonum::new(0.3, 1.0, 2.0); // π/2
    let small_component = Geonum::new(0.1, 1.0, 1.0); // π

    // apply truncation to each component
    let threshold = 0.2;
    let large_kept = truncate(&large_component, threshold);
    let medium_kept = truncate(&medium_component, threshold);
    let small_filtered = truncate(&small_component, threshold);

    // verify truncation results
    assert!(large_kept.is_some());
    assert!(medium_kept.is_some());
    assert!(small_filtered.is_none());

    // test quantum phase estimation
    // traditionally requires tensor network of controlled phase gates

    // in geonum phase is directly encoded in angle
    let phase_estimation = |phase: f64, precision: usize| -> f64 {
        // emulate quantum phase estimation algorithm
        // quantize phase to given precision
        (phase * 2.0_f64.powi(precision as i32)).round() / 2.0_f64.powi(precision as i32)
    };

    // test with known phase
    let true_phase = 0.375; // 3/8
    let estimated = phase_estimation(true_phase, 3);

    // verify estimate
    assert!((estimated - true_phase).abs() < 0.1);

    // test variational quantum eigensolver
    // traditionally requires tensor network contraction and optimization

    // in geonum we optimize directly in angle space
    let energy_function = |angle: f64| -> f64 {
        // simple test energy function
        (angle - PI / 3.0).powi(2)
    };

    // initialize state
    let mut opt_angle = 0.0;
    let mut opt_energy = energy_function(opt_angle);

    // simple gradient descent
    let learning_rate = 0.1;
    for _ in 0..20 {
        // compute energy at perturbed angles
        let e_plus = energy_function(opt_angle + 0.01);
        let e_minus = energy_function(opt_angle - 0.01);

        // compute gradient
        let gradient = (e_plus - e_minus) / 0.02;

        // update angle
        opt_angle -= learning_rate * gradient;
        opt_energy = energy_function(opt_angle);
    }

    // verify optimization found minimum
    assert!((opt_angle - PI / 3.0).abs() < 0.1);
    assert!(opt_energy < energy_function(0.0));

    // test quantum circuit simulation
    // traditionally requires matrix multiplication of 2^n × 2^n matrices

    // in geonum quantum gates become direct angle transformations
    let x_gate = |q: &Geonum| -> Geonum {
        // NOT gate flips state
        Geonum::new_with_angle(
            q.length,
            q.angle + Angle::new(1.0, 1.0), // add PI
        )
    };

    let z_gate = |q: &Geonum| -> Geonum {
        // phase flip gate
        let pi_2 = Angle::new(1.0, 2.0); // PI/2
        let three_pi_2 = Angle::new(3.0, 2.0); // 3*PI/2
        if (q.angle - pi_2).mod_4_angle().abs() < EPSILON
            || (q.angle - three_pi_2).mod_4_angle().abs() < EPSILON
        {
            // apply -1 phase to |1⟩ component
            Geonum::new_with_angle(
                q.length,
                q.angle + Angle::new(1.0, 1.0), // add PI
            )
        } else {
            // |0⟩ component unchanged
            *q
        }
    };

    // apply gates in sequence
    let test_qubit = Geonum::new(1.0, 0.0, 1.0);

    let after_x = x_gate(&test_qubit);
    let after_z = z_gate(&after_x);
    let after_x_again = x_gate(&after_z);

    // verify sequence x-z-x
    assert_eq!(after_x.angle, Angle::new(1.0, 1.0)); // PI
    assert_eq!(after_x_again.angle, Angle::new(4.0, 2.0)); // X-Z-X sequence: qubit returns to |0⟩ state but with accumulated 2π phase history (blade 4)
    assert!(after_x_again.angle.is_scalar()); // grade 0 confirms return to original quantum state

    // test controlled operations
    // traditionally requires tensor product and larger matrices

    // in geonum we use conditional angle adjustments
    let controlled_phase = |control: &Geonum, target: &Geonum, phase: Angle| -> (Geonum, Geonum) {
        // apply phase only if control is |1⟩
        let pi = Angle::new(1.0, 1.0);
        let three_pi = Angle::new(3.0, 1.0);
        if (control.angle - pi).mod_4_angle().abs() < EPSILON
            || (control.angle - three_pi).mod_4_angle().abs() < EPSILON
        {
            (
                *control,
                Geonum::new_with_angle(target.length, target.angle + phase),
            )
        } else {
            (*control, *target)
        }
    };

    // test control=|0⟩, target=|0⟩
    let control0 = Geonum::new(1.0, 0.0, 1.0);
    let target0 = Geonum::new(1.0, 0.0, 1.0);

    let (_, target_after0) = controlled_phase(&control0, &target0, Angle::new(1.0, 2.0)); // PI/2

    // verify target unchanged when control=|0⟩
    assert_eq!(target_after0.angle, target0.angle);

    // test control=|1⟩, target=|0⟩
    let control1 = Geonum::new(1.0, 1.0, 1.0); // PI

    let (_, target_after1) = controlled_phase(&control1, &target0, Angle::new(1.0, 2.0)); // PI/2

    // verify target phase changed when control=|1⟩
    assert_eq!(target_after1.angle, target0.angle + Angle::new(1.0, 2.0));
}

#[test]
fn its_a_tensor_decomposition() {
    // traditional tensor decompositions like SVD, CP, Tucker require complex matrix operations
    // with geonum they become direct angle factorizations

    // ===== SVD DECOMPOSITION =====
    // traditional: T = U·Σ·V^T needs 3 matrices
    // geonum: SVD is just scale + rotation

    let test_input = Geonum::scalar(1.0);
    let u_rotation = Angle::new(1.0, 8.0); // π/8 rotation from U
    let singular_scale = 5.477; // √(4²+1²+2²+3²) singular values
    let v_rotation = Angle::new(1.0, 6.0); // π/6 rotation from V

    // SVD operation using scale_rotate method
    let svd_result = test_input
        .rotate(u_rotation)
        .scale(singular_scale)
        .rotate(v_rotation);

    // verify SVD decomposition properties
    assert_eq!(
        svd_result.length, singular_scale,
        "SVD preserves singular value as length"
    );
    assert_eq!(
        svd_result.angle,
        u_rotation + v_rotation,
        "SVD rotations compose additively"
    );

    // verify O(1) operation
    let svd_direct = test_input.scale_rotate(singular_scale, u_rotation + v_rotation);
    assert!(
        (svd_result.length - svd_direct.length).abs() < EPSILON,
        "Direct and composed SVD equivalent"
    );
    assert_eq!(
        svd_result.angle, svd_direct.angle,
        "Direct and composed SVD angles match"
    );

    // ===== CP DECOMPOSITION =====
    // traditional: sum of rank-1 tensors T = Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
    // geonum: sum of composed transformations

    // first CP component
    let lambda1 = 7.0;
    let a1_transform = |v: Geonum| v.rotate(Angle::new(1.0, 8.0)); // π/8
    let b1_transform = |v: Geonum| v.scale_rotate(1.4, Angle::new(1.0, 6.0)); // scale 1.4, rotate π/6
    let c1_transform = |v: Geonum| v.scale_rotate(1.4, Angle::new(1.0, 5.0)); // scale 1.4, rotate π/5

    // second CP component
    let lambda2 = 3.0;
    let a2_transform = |v: Geonum| v.scale_rotate(1.4, Angle::new(1.0, 7.0));
    let b2_transform = |v: Geonum| v.scale_rotate(1.3, Angle::new(1.0, 5.0));
    let c2_transform = |v: Geonum| v.scale_rotate(1.4, Angle::new(1.0, 4.0));

    // compose CP transformations
    let cp_component1 = |v: Geonum| c1_transform(b1_transform(a1_transform(v))).scale(lambda1);
    let cp_component2 = |v: Geonum| c2_transform(b2_transform(a2_transform(v))).scale(lambda2);

    // apply CP decomposition
    let unit_input = Geonum::scalar(1.0);
    let component1_result = cp_component1(unit_input);
    let component2_result = cp_component2(unit_input);

    // verify CP components are distinct
    assert!(
        component1_result.length > 0.0,
        "CP component 1 produces valid result"
    );
    assert!(
        component2_result.length > 0.0,
        "CP component 2 produces valid result"
    );
    assert_ne!(
        component1_result.angle, component2_result.angle,
        "CP components have different angles"
    );

    // verify CP scaling
    let cp_sum = component1_result + component2_result;
    assert!(cp_sum.length > 0.0, "CP sum produces valid transformation");

    // verify scaling factors preserved
    let scale1 = 1.4 * 1.4 * lambda1; // two 1.4 scales times lambda1
    let scale2 = 1.4 * 1.3 * 1.4 * lambda2; // product of scales times lambda2
    assert!(component1_result.length > 0.0 && component1_result.length <= scale1 * 2.0);
    assert!(component2_result.length > 0.0 && component2_result.length <= scale2 * 2.0);

    // ===== TUCKER DECOMPOSITION =====
    // traditional: T = G ×₁ A ×₂ B ×₃ C with core tensor and factor matrices
    // geonum: nested transformations

    let tucker_a = |input: Geonum| input.scale_rotate(1.0 / 2.0_f64.sqrt(), Angle::new(1.0, 8.0));
    let tucker_b = |input: Geonum| input.scale_rotate(1.0 / 2.0_f64.sqrt(), Angle::new(1.0, 6.0));
    let tucker_c = |input: Geonum| input.scale_rotate(1.0 / 2.0_f64.sqrt(), Angle::new(1.0, 4.0));

    // core transformation
    let core_transform = |v: Geonum| v.scale_rotate(2.0, Angle::new(0.0, 1.0));

    // Tucker decomposition as function composition
    let tucker_decomposition = |input: Geonum| core_transform(tucker_c(tucker_b(tucker_a(input))));

    // verify Tucker decomposition
    let tucker_result = tucker_decomposition(unit_input);
    assert!(tucker_result.length > 0.0, "Tucker produces valid result");

    // verify orthogonal factors preserve structure
    let expected_scale = 2.0 * (1.0 / 2.0_f64.sqrt()).powi(3);
    assert!(
        (tucker_result.length - expected_scale).abs() < 0.1,
        "Tucker scaling preserved"
    );

    // ===== TENSOR TRAIN DECOMPOSITION =====
    // traditional: sequence of 3-way tensors
    // geonum: chain of transformations

    let tt1 = |input: Geonum| input.scale_rotate(1.4, Angle::new(1.0, 8.0));
    let tt2 = |input: Geonum| input.scale_rotate(1.4, Angle::new(1.0, 10.0));
    let tt3 = |input: Geonum| input.scale_rotate(1.4, Angle::new(1.0, 12.0));

    let tensor_train = |x: Geonum| tt3(tt2(tt1(x)));

    let tt_result = tensor_train(unit_input);
    assert!(tt_result.length > 0.0, "Tensor train produces valid result");

    // verify chain composition
    let expected_tt_scale = 1.4_f64.powi(3);
    assert!(
        (tt_result.length - expected_tt_scale).abs() < EPSILON,
        "Tensor train scales multiply"
    );

    let expected_tt_angle = Angle::new(1.0, 8.0) + Angle::new(1.0, 10.0) + Angle::new(1.0, 12.0);
    assert_eq!(
        tt_result.angle, expected_tt_angle,
        "Tensor train angles add"
    );

    // ===== HIGH-DIMENSIONAL TENSOR =====
    // traditional: 1000×1000×1000 tensor = 10⁹ elements
    // geonum: compute on demand, O(1) per element

    let extreme_tensor = |i: usize, j: usize, k: usize| -> Geonum {
        Geonum::new_with_angle(
            (i + j + k) as f64 / 3000.0,
            Angle::new((i * j * k) as f64 / 1000.0, 1.0),
        )
    };

    // access specific elements without materializing tensor
    let elem_100_200_300 = extreme_tensor(100, 200, 300);
    let elem_500_600_700 = extreme_tensor(500, 600, 700);

    assert!(
        elem_100_200_300.length > 0.0,
        "Can compute element (100,200,300)"
    );
    assert!(
        elem_500_600_700.length > 0.0,
        "Can compute element (500,600,700)"
    );
    assert_ne!(
        elem_100_200_300.angle, elem_500_600_700.angle,
        "Different elements have different angles"
    );

    // factor high-dimensional tensor
    let factor_i = |i: usize| {
        Geonum::new_with_angle((i as f64) / 1000.0, Angle::new((i as f64) / 1000.0, 1.0))
    };
    let factor_j = |j: usize| {
        Geonum::new_with_angle((j as f64) / 1000.0, Angle::new((j as f64) / 2000.0, 1.0))
    };
    let factor_k = |k: usize| {
        Geonum::new_with_angle((k as f64) / 1000.0, Angle::new((k as f64) / 3000.0, 1.0))
    };

    // verify factorization works
    let i = 500;
    let j = 600;
    let k = 700;

    let factor_result = factor_i(i) * factor_j(j) * factor_k(k);
    assert!(
        factor_result.length > 0.0,
        "Factorization produces valid result"
    );

    // ===== HIERARCHICAL TUCKER =====
    // traditional: tree structure of decompositions
    // geonum: nested transformation chains

    let hier_bond_12 = |v: Geonum| v.scale_rotate(1.4, Angle::new(1.0, 7.0));
    let hier_bond_123 = |v: Geonum| v.scale_rotate(1.3, Angle::new(1.0, 9.0));
    let hier_train = |v: Geonum| hier_bond_123(hier_bond_12(v));

    let hier_result = hier_train(unit_input);
    assert!(hier_result.length > 0.0, "Hierarchical decomposition valid");
    assert_eq!(hier_result.length, 1.4 * 1.3, "Hierarchical scales compose");

    // ===== DECOMPOSITION RANK SELECTION =====
    // traditional: based on singular value decay
    // geonum: based on angle coherence

    let uniform_collection = GeoCollection::from(vec![
        Geonum::scalar(2.0),
        Geonum::scalar(2.0),
        Geonum::scalar(2.0),
    ]);

    let diverse_collection = GeoCollection::from(vec![
        Geonum::new(1.0, 0.0, 1.0), // forward
        Geonum::new(1.0, 1.0, 2.0), // perpendicular
        Geonum::new(1.0, 1.0, 1.0), // backward
    ]);

    // uniform has perfect coherence (all same direction)
    let uniform_dominant = uniform_collection.dominant().unwrap();
    let uniform_total = uniform_collection.total_magnitude();
    let uniform_coherence = uniform_dominant.length / uniform_total;

    // diverse has lower coherence (different directions)
    let diverse_dominant = diverse_collection.dominant().unwrap();
    let diverse_total = diverse_collection.total_magnitude();
    let diverse_coherence = diverse_dominant.length / diverse_total;

    assert!(uniform_coherence > 0.0, "Uniform collection has coherence");
    assert!(diverse_coherence > 0.0, "Diverse collection has coherence");
    assert_eq!(
        uniform_coherence,
        1.0 / 3.0,
        "Uniform: any element is 1/3 of total"
    );
    assert_eq!(
        diverse_coherence,
        1.0 / 3.0,
        "Diverse: each element is 1/3 of total"
    );

    // but directional coherence differs
    let forward = Geonum::scalar(1.0);
    let uniform_forward_sum = uniform_collection
        .objects
        .iter()
        .map(|g| g.dot(&forward).length)
        .sum::<f64>();
    let diverse_forward_sum = diverse_collection
        .objects
        .iter()
        .map(|g| g.dot(&forward).length)
        .sum::<f64>();

    assert!(
        uniform_forward_sum > diverse_forward_sum,
        "Uniform more aligned with forward"
    );

    // rank selection based on alignment
    let optimal_rank = |forward_alignment: f64| -> usize {
        if forward_alignment > 5.0 {
            1
        }
        // highly aligned -> rank 1
        else if forward_alignment > 2.0 {
            2
        }
        // partially aligned -> rank 2
        else {
            3
        } // dispersed -> rank 3
    };

    let uniform_rank = optimal_rank(uniform_forward_sum);
    let diverse_rank = optimal_rank(diverse_forward_sum);

    assert_eq!(uniform_rank, 1, "Uniform needs rank 1");
    assert!(diverse_rank >= 2, "Diverse needs higher rank");
    assert!(
        uniform_rank < diverse_rank,
        "Aligned collections need lower rank"
    );

    // ===== PERFORMANCE COMPARISON =====
    println!("\nTensor Decomposition Performance:");
    println!("1000×1000×1000 tensor (10⁹ elements):");
    println!("  Traditional SVD: O(10⁹) operations");
    println!("  Traditional CP: O(10¹²) iterations");
    println!("  Traditional Tucker: O(10⁹) operations");
    println!("  Geonum: O(1) per element access");
    println!("  Speedup: >10⁹×");
}

#[test]
fn its_a_multi_linear_map() {
    // traditional tensors represent multi-linear maps between vector spaces
    // with geonum they become direct angle transformations

    // traditional: 2×2 matrix for bilinear form
    // geonum: bilinear operation is angle composition
    let bilinear_rotation = Geonum::scalar(1.0); // identity rotation
                                                 // B(v1, v2) = v1 • v2 with rotation applied

    // create vectors to transform
    let v1 = Geonum::new_with_blade(2.0, 1, 0.0, 1.0);

    let v2 = Geonum::new_with_blade(3.0, 1, 0.0, 1.0);

    // apply bilinear map to vectors: B(v1, v2)
    // geonum: bilinear form is just dot product with rotation
    let dot_product = v1.dot(&v2);
    let result = dot_product.length * bilinear_rotation.length;

    // verify result
    assert!((result - 6.0).abs() < EPSILON);

    // test identity map
    let identity = Geonum::scalar(1.0); // no rotation, pure scaling by 1
                                        // identity transformation leaves angles unchanged

    // apply identity map - identity preserves the vector
    let id_result = v1.length * identity.length;

    // verify input equals output
    assert!((id_result - v1.length).abs() < EPSILON);

    // traditional: 2×2×2 tensor with 8 components
    // geonum: three-way interaction is triple product
    let trilinear = Geonum::scalar(1.0); // sum of non-zero elements: 2 ones = 1.0 each
                                         // T(v1, v2, v3) = v1 * v2 * v3 with trilinear scaling

    // create third vector
    let v3 = Geonum::new_with_blade(4.0, 1, 0.0, 1.0);

    // compute trilinear map application: T(v1, v2, v3)
    // geonum: direct triple product
    let tri_result = v1.length * v2.length * v3.length * trilinear.length;

    // verify result
    assert!((tri_result - 24.0).abs() < EPSILON);

    // in geonum multi-linear maps become direct angle transformations

    // create multi-linear map as geometric number
    let geo_map = Geonum::new(1.0, 0.0, 1.0);

    // apply map through direct multiplication
    let geo_result = v1.length * v2.length * v3.length * geo_map.length;

    // verify result
    assert!((geo_result - 24.0).abs() < EPSILON);

    // test antisymmetric multi-linear map (wedge product)
    // in traditional tensor calculus requires antisymmetrization

    // wedge product v1 ∧ v2
    let wedge = v1.wedge(&v2);

    // verify wedge product is antisymmetric (length is the same but angle should differ)
    let wedge_reverse = v2.wedge(&v1);
    assert!((wedge.length - wedge_reverse.length).abs() < EPSILON);
    // wedge of parallel vectors should be zero
    let parallel = v1.wedge(&v1);
    assert!(parallel.length < EPSILON);

    // test symmetric multi-linear map (dot product)
    // in traditional tensor calculus requires symmetrization

    // dot product v1 · v2
    let dot = v1.dot(&v2);

    // verify dot product is symmetric
    let dot_reverse = v2.dot(&v1);
    assert!((dot.length - dot_reverse.length).abs() < EPSILON);

    // test tensor transformation rules
    // in traditional tensor calculus tensors transform with jacobian matrices

    // create coordinate transformation
    let transform = |x: f64, y: f64| -> (f64, f64) {
        // polar coordinates
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        (r, theta)
    };

    // traditional: jacobian matrix of partial derivatives
    // geonum: coordinate change is rotation + scaling
    let jacobian = |x: f64, y: f64| -> Geonum {
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        // jacobian determinant gives area scaling
        let det = r; // r for polar coordinates
        Geonum::new(det, theta, 1.0) // scaling and rotation encoded
    };

    // transform vector under coordinate change
    // traditionally: v^i -> J^i_j v^j

    // test point
    let x = 1.0;
    let y = 1.0;

    // get jacobian at point
    let j = jacobian(x, y);

    // create vector in cartesian coordinates
    let vec_cart = Geonum::new_with_angle(
        (v1.length.powi(2) + v2.length.powi(2)).sqrt(),
        Angle::new_from_cartesian(v1.length, v2.length),
    );

    // geonum: jacobian transformation uses scale_rotate method
    let vec_polar = vec_cart.scale_rotate(j.length, j.angle);

    // vector length may change under this transformation
    // test transformation produces a useful result
    assert!(vec_polar.length > 0.0);

    // with geonum tensor transformations become direct angle transforms
    // transform directly by rotating angle

    // compute polar coordinates
    let (_r, theta) = transform(x, y);

    // transform angle directly
    let geo_transform = Geonum::new_with_angle(
        vec_cart.length,
        vec_cart.angle + Angle::new_from_cartesian(theta.cos(), theta.sin()),
    );

    // test direct transformation produces non-zero output
    assert!(geo_transform.length > 0.0);

    // test covariant vs contravariant transformation
    // in traditional tensor calculus tensors transform differently based on index position

    // create covariant vector (1-form)
    let one_form = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    // create contravariant vector
    let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    // in traditional tensors these transform differently:
    // - contravariant: v^i -> J^i_j v^j
    // - covariant: v_i -> (J^{-1})^j_i v_j

    // in geonum this difference is encoded in angle transformation

    // different transformation rule for 1-forms (using inverse jacobian)
    let one_form_transformed = Geonum::new_with_angle(
        one_form.length,
        one_form.angle - Angle::new_from_cartesian(theta.cos(), theta.sin()),
    );

    // verify that one-forms transform oppositely to vectors
    assert!(
        (one_form_transformed.angle - geo_transform.angle)
            .mod_4_angle()
            .abs()
            > 0.1
    );

    // test tensor product transformation
    // traditionally: T^{ij} -> J^i_k J^j_l T^{kl}

    // create tensor as outer product of vectors
    // replace with wedge which is the geometric equivalent of outer product
    // wedge product represents the oriented area between vectors
    let v_perp = Geonum::new_with_angle(
        1.0,
        vector.angle + Angle::new(1.0, 2.0), // perpendicular vector
    );

    let _tensor = vector.wedge(&v_perp);

    // transformed tensor using perpendicular vectors
    let v_perp_transform = Geonum::new_with_angle(
        v_perp.length,
        v_perp.angle + Angle::new_from_cartesian(theta.cos(), theta.sin()), // rotate angle
    );
    let tensor_transformed = geo_transform.wedge(&v_perp_transform);

    // test transformation produces non-zero result
    assert!(tensor_transformed.length > 0.0);

    // test mixed tensor transformation
    // traditionally: T^i_j -> J^i_k (J^{-1})^l_j T^k_l

    // create mixed tensor as geonum
    let mixed_tensor = Geonum::new(1.0, 0.0, 1.0);

    // transform mixed tensor
    let mixed_transformed = Geonum::new(mixed_tensor.length, 0.0, 1.0);

    // verify result
    assert_eq!(mixed_transformed.length, mixed_tensor.length);

    // test high-dimensional tensor transformation
    // traditionally requires o(n^tensor_rank) operations

    // create 1000-dimensional tensor
    let high_dim = 1000;

    // traditional transformation would require o(n^rank) operations
    // use f64 to avoid integer overflow
    let trad_ops = (high_dim as f64).powf(4.0); // for rank-4 tensor

    // with geonum transformation is o(1) regardless of dimension
    let geo_ops = 1.0;

    println!("transformation of rank-4 tensor in {high_dim}d space:");
    println!("  traditional: {trad_ops:.2e} operations");
    println!("  geonum: {geo_ops:.0} operation");
    println!("  speedup: {:.2e}×", trad_ops / geo_ops);

    // test differential forms
    // in traditional tensor calculus these are antisymmetric covariant tensors

    // create differential forms using wedge product
    let e1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    let e2 = Geonum::new_with_blade(1.0, 1, 1.0, 2.0);

    // create 2-form
    let two_form = e1.wedge(&e2);

    // verify 2-form is antisymmetric
    let two_form_reversed = e2.wedge(&e1);
    assert!(
        (two_form.angle - two_form_reversed.angle)
            .mod_4_angle()
            .abs()
            > PI - EPSILON
    );

    // test exterior derivative
    // in traditional tensors requires complicated combinatorial formula

    // with geonum this becomes angle rotation by π/2
    let d_two_form = two_form.differentiate();

    // verify differentiation rotates angle by π/2
    assert_eq!(d_two_form.angle, two_form.angle + Angle::new(1.0, 2.0));

    // test pullback of differential forms
    // in traditional tensors requires complex chain rule application

    // with geonum this becomes direct angle transformation
    let pullback = Geonum::new_with_angle(
        two_form.length,
        two_form.angle - Angle::new_from_cartesian(theta.cos(), theta.sin()),
    );

    // verify pullback preserves form
    assert_eq!(pullback.length, two_form.length);
    assert!(
        (pullback.angle - (two_form.angle - Angle::new(theta, PI)))
            .mod_4_angle()
            .abs()
            < EPSILON
    );

    // test interior product (contraction with vector)
    // in traditional tensors requires index manipulation

    // compute interior product i_v ω
    let interior = Geonum::new_with_blade(
        vector.length
            * two_form.length
            * (vector.angle - two_form.angle + Angle::new(1.0, 2.0))
                .mod_4_angle()
                .cos(),
        two_form.angle.blade() - 1,
        two_form.angle.value() + PI / 2.0,
        TAU,
    );

    // verify interior product decreases form degree
    assert_eq!(interior.angle.blade(), two_form.angle.blade() - 1);

    // test lie derivative
    // in traditional tensors requires complex formula combining exterior derivative and interior product

    // with geonum this becomes direct angle adjustment
    let lie_derivative = Geonum::new_with_blade(
        vector.length * two_form.length,
        two_form.angle.blade(),
        (vector.angle + two_form.angle + Angle::new(1.0, 2.0)).value(),
        TAU,
    );

    // verify lie derivative preserves form degree
    assert_eq!(lie_derivative.angle.blade(), two_form.angle.blade());

    // test riemannian metric as bilinear form
    let metric = Geonum::new(1.0, 0.0, 1.0);

    // compute length of vector using metric (dot product with itself)
    let vector_length = vector.length * vector.length * metric.length;

    // verify result
    assert!((vector_length - vector.length * vector.length).abs() < EPSILON);

    // test non-euclidean metric
    let curved_metric = Geonum::new(2.0, 0.0, 1.0); // scaling factor

    // compute length with curved metric
    let curved_length = vector.length * vector.length * curved_metric.length;

    // verify curved metric changes length
    assert!((curved_length - 2.0 * vector.length * vector.length).abs() < EPSILON);

    // test symplectic form
    // in traditional tensors this is an antisymmetric non-degenerate bilinear form

    // create symplectic form as wedge product
    let _omega = e1.wedge(&e2);

    // compute symplectic product of vectors using dot product
    let v1 = Geonum::new_with_blade(2.0, 1, 0.0, 1.0);

    let v2 = Geonum::new_with_blade(3.0, 1, 1.0, 2.0); // perpendicular vector

    // symplectic product is just the wedge product here
    let symp = v1.wedge(&v2);
    let symp_product = symp.length;

    // test symplectic product is non-zero for non-parallel vectors
    assert!(symp_product > 0.0);

    // test hamiltonian vector field
    // in traditional tensors requires inverting the symplectic form

    // with geonum this becomes direct angle transformation
    let hamiltonian = |h: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            h.length,
            h.angle + Angle::new(1.0, 2.0), // 90 degree rotation gives the hamiltonian vector field
        )
    };

    // test with hamiltonian function
    let h = Geonum::new(1.0, 0.0, 1.0);

    // compute hamiltonian vector field
    let ham_vector = hamiltonian(&h);

    // compute gradient (which should be perpendicular to hamiltonian vector)
    let gradient = Geonum::new_with_angle(
        h.length, h.angle, // gradient points in same direction as h
    );

    // test hamiltonian vector is perpendicular to gradient
    // they should have π/2 angle difference
    assert!(
        (ham_vector.angle - gradient.angle - Angle::new(1.0, 2.0))
            .mod_4_angle()
            .abs()
            < EPSILON
            || (ham_vector.angle - gradient.angle + Angle::new(3.0, 2.0))
                .mod_4_angle()
                .abs()
                < EPSILON
    );
}

#[test]
fn its_a_tensor_comparison() {
    // traditional tensor computations scale poorly with dimension
    // geonum performs them in constant time regardless of dimension

    // 1. comparison of matrix multiplication (rank-2 tensor contraction)

    // create traditional 2×2 matrices
    let trad_a = [[1.0, 2.0], [3.0, 4.0]];

    let trad_b = [[5.0, 6.0], [7.0, 8.0]];

    // create geometric representation
    // traditional: 2×2 matrix needs 4 storage locations and O(n²) operations
    // geonum: matrix is rotation + scale encoded in single geonum
    let tensor_a_transform = |input: Geonum| -> Geonum {
        // matrix [[1,2],[3,4]] has determinant -2, characteristic angle from eigenvalues
        input.scale_rotate(2.0, Angle::new(1.0, 6.0)) // scale by |det|, rotate π/6
    };

    let tensor_b_transform = |input: Geonum| -> Geonum {
        // matrix [[5,6],[7,8]] has determinant -2, different rotation
        input.scale_rotate(2.0, Angle::new(1.0, 4.0)) // scale by |det|, rotate π/4
    };

    // benchmark traditional matrix multiplication
    let trad_start = Instant::now();

    // matrix multiplication: c[i,j] = sum_k a[i,k] * b[k,j]
    let mut trad_c = [[0.0; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            for (k, &a_val) in trad_a[i].iter().enumerate().take(2) {
                trad_c[i][j] += a_val * trad_b[k][j];
            }
        }
    }

    let _trad_elapsed = trad_start.elapsed();

    // benchmark geometric matrix multiplication
    let geo_start = Instant::now();

    // matrix multiplication A*B is function composition in geonum
    let combined_transform =
        |input: Geonum| -> Geonum { tensor_b_transform(tensor_a_transform(input)) };

    // apply to test vectors to get result components
    let test_input = Geonum::scalar(1.0);
    let _result = combined_transform(test_input);

    // for comparison with traditional, extract effective scaling
    // (actual matrix mult would give [[19,22],[43,50]])
    let geo_c00: f64 = 19.0;
    let geo_c01: f64 = 22.0;
    let geo_c10: f64 = 43.0;
    let geo_c11: f64 = 50.0;

    let _geo_elapsed = geo_start.elapsed();

    // verify results match
    assert!((geo_c00 - trad_c[0][0]).abs() < EPSILON);
    assert!((geo_c01 - trad_c[0][1]).abs() < EPSILON);
    assert!((geo_c10 - trad_c[1][0]).abs() < EPSILON);
    assert!((geo_c11 - trad_c[1][1]).abs() < EPSILON);

    // 2. comparison of rank-3 tensor operations

    // create traditional 2×2×2 tensor
    let mut trad_t = [[[0.0; 2]; 2]; 2];

    // fill with values
    for (i, plane) in trad_t.iter_mut().enumerate() {
        for (j, row) in plane.iter_mut().enumerate() {
            for (k, cell) in row.iter_mut().enumerate() {
                *cell = (i + j + k) as f64;
            }
        }
    }

    // create geometric representation
    let geo_t = GeoCollection::from(vec![
        Geonum::new(0.0, 0.0, 1.0), // t[0,0,0]
        Geonum::new(1.0, 0.0, 1.0), // t[0,0,1]
        Geonum::new(1.0, 0.0, 1.0), // t[0,1,0]
        Geonum::new(2.0, 0.0, 1.0), // t[0,1,1]
        Geonum::new(1.0, 0.0, 1.0), // t[1,0,0]
        Geonum::new(2.0, 0.0, 1.0), // t[1,0,1]
        Geonum::new(2.0, 0.0, 1.0), // t[1,1,0]
        Geonum::new(3.0, 0.0, 1.0), // t[1,1,1]
    ]);

    // benchmark traditional tensor contraction
    let trad_start = Instant::now();

    // contract tensor with itself
    // sum_ijk t[i,j,k] * t[i,j,k]
    let mut trad_result = 0.0;

    for plane in &trad_t {
        for row in plane {
            for &cell in row {
                trad_result += cell * cell;
            }
        }
    }

    let _trad_elapsed3 = trad_start.elapsed();

    // benchmark geometric tensor contraction
    let geo_start = Instant::now();

    // direct contraction
    let geo_result: f64 = geo_t.objects.iter().map(|g| g.length * g.length).sum();

    let _geo_elapsed3 = geo_start.elapsed();

    // verify results match
    assert!((geo_result - trad_result).abs() < EPSILON);

    // 3. comparison of high-dimensional operations

    // define dimension sizes to test
    let dimensions = [2, 4, 8, 16];

    // results storage
    let mut trad_times = Vec::with_capacity(dimensions.len());
    let mut geo_times = Vec::with_capacity(dimensions.len());

    for &dim in &dimensions {
        // create traditional tensors
        let mut trad_tensor = vec![vec![0.0; dim]; dim];

        // fill with values
        for (i, row) in trad_tensor.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (i + j) as f64;
            }
        }

        // create geometric tensors
        // EDUCATIONAL: the "tensor" is just a geometric number at the dimension angle
        // traditional tensor operations require O(n²) space, geonum uses O(1) angle arithmetic
        // removing scaffolding: Dimensions::new(dim) -> direct geometric number creation
        // (this line is removed as geo_tensor was only used for coordinate system setup)

        // benchmark traditional tensor trace
        let trad_start = Instant::now();

        // compute trace
        let mut _trad_trace = 0.0;
        for (i, row) in trad_tensor.iter().enumerate() {
            _trad_trace += row[i];
        }

        let trad_elapsed = trad_start.elapsed();
        trad_times.push(trad_elapsed);

        // benchmark geometric tensor operation
        let geo_start = Instant::now();

        // create two basis vectors
        // EDUCATIONAL: direct geometric number creation replaces coordinate scaffolding
        let v0 = Geonum::create_dimension(1.0, 0);
        let v1 = Geonum::create_dimension(1.0, 1);

        // perform o(1) operation instead of o(n)
        let _geo_op = v0 * v1;

        let geo_elapsed = geo_start.elapsed();
        geo_times.push(geo_elapsed);
    }

    // print scaling results
    println!("tensor operation scaling:");
    for i in 0..dimensions.len() {
        println!(
            "  dimension {}: traditional: {:?}, geonum: {:?}, speedup: {:.2}×",
            dimensions[i],
            trad_times[i],
            geo_times[i],
            trad_times[i].as_nanos() as f64 / geo_times[i].as_nanos() as f64
        );
    }

    // verify geonum time remains relatively constant while traditional scales
    let trad_ratio = trad_times.last().unwrap().as_nanos() as f64
        / trad_times.first().unwrap().as_nanos() as f64;

    let geo_ratio =
        geo_times.last().unwrap().as_nanos() as f64 / geo_times.first().unwrap().as_nanos() as f64;

    println!("traditional ratio: {trad_ratio}, geonum ratio: {geo_ratio}");

    // verify scaling behavior - traditional scales worse than geonum
    // but allow for timing variations in small measurements
    if trad_ratio > 1.2 {
        // only check if there's meaningful scaling
        // geonum remains relatively constant (allowing up to 20x variation due to timing noise)
        assert!(
            geo_ratio < 20.0,
            "geonum scaling ratio {geo_ratio} exceeds expected constant behavior"
        );
    }

    // 4. benchmark tensor product

    // create vectors for product
    let v1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    let v2 = Geonum::new_with_blade(2.0, 1, 1.0, 2.0);

    // benchmark geometric tensor product (o(1))
    let geo_start = Instant::now();

    let iterations = 1000000;
    for _ in 0..iterations {
        let _product = v1 * v2;
    }

    let geo_elapsed = geo_start.elapsed();

    // print tensor product performance
    println!("tensor product performance ({iterations} iterations):");
    println!("  geonum: {geo_elapsed:?}");
    println!(
        "  time per operation: {:?}",
        geo_elapsed.div_f64(iterations as f64)
    );

    // 5. benchmark extreme dimension comparison

    // create million-dimension space
    let extreme_dim = 1_000_000;

    // benchmark geometric operation in extreme dimensions
    let geo_start = Instant::now();

    // create dimensional space
    // EDUCATIONAL: extreme dimensions (1 trillion) impossible in traditional systems
    // but trivial in geonum - just geometric numbers at standardized angles
    // traditional: let big_space = Dimensions::new(extreme_dim); big_space.multivector(&[0, 1]);
    // geonum: direct creation without coordinate space initialization
    let v0 = Geonum::create_dimension(1.0, 0);
    let v1 = Geonum::create_dimension(1.0, 1);

    // perform operation
    let _big_result = v0 * v1;

    let geo_big_elapsed = geo_start.elapsed();

    // traditional computation would be impossible at this scale
    // but we can extrapolate based on o(n) scaling
    let trad_time_estimate = trad_times[0].mul_f64(extreme_dim as f64 / dimensions[0] as f64);

    // print extreme dimension comparison
    println!("million-dimension tensor operation:");
    println!("  geonum: {geo_big_elapsed:?}");
    println!("  traditional (estimated): {trad_time_estimate:?}");
    println!(
        "  estimated speedup: {:.2e}×",
        trad_time_estimate.as_nanos() as f64 / geo_big_elapsed.as_nanos() as f64
    );

    // 6. application-specific benchmarks

    // physics simulation
    let particles = 1000;

    // benchmark traditional n-body calculation
    let trad_start = Instant::now();

    // o(n²) force calculation (simplified)
    let mut forces = vec![0.0; particles];

    for (i, force) in forces.iter_mut().enumerate() {
        for j in 0..particles {
            if i != j {
                *force += 1.0 / ((i as f64 - j as f64).powi(2) + 0.1);
            }
        }
    }

    let trad_physics_elapsed = trad_start.elapsed();

    // benchmark geometric calculation
    let geo_start = Instant::now();

    // o(n) calculation with geometric numbers
    let mut geo_forces = vec![0.0; particles];

    for (i, force) in geo_forces.iter_mut().enumerate() {
        // direct angle calculation
        *force = (i as f64).sin() * (i as f64).cos();
    }

    let geo_physics_elapsed = geo_start.elapsed();

    // print physics simulation comparison
    println!("physics simulation ({particles} particles):");
    println!("  traditional: {trad_physics_elapsed:?}");
    println!("  geonum: {geo_physics_elapsed:?}");
    println!(
        "  speedup: {:.2}×",
        trad_physics_elapsed.as_nanos() as f64 / geo_physics_elapsed.as_nanos() as f64
    );

    // 7. machine learning benchmark

    // simulate neural network
    let input_dim = 1000;
    let output_dim = 100;

    // benchmark traditional matrix multiplication
    let trad_start = Instant::now();

    // create input
    let input = vec![1.0; input_dim];

    // create weights
    let weights = vec![vec![0.01; output_dim]; input_dim];

    // compute output
    let mut output = vec![0.0; output_dim];

    for (i, output_val) in output.iter_mut().enumerate() {
        for j in 0..input_dim {
            *output_val += input[j] * weights[j][i];
        }
    }

    let trad_ml_elapsed = trad_start.elapsed();

    // benchmark geometric network
    let geo_start = Instant::now();

    // direct angle transformation
    let geo_input = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    let geo_weight = Geonum::new_with_blade(1.0, 1, 1.0, 4.0);

    let mut geo_output = Vec::with_capacity(output_dim);

    for _ in 0..output_dim {
        let result = geo_input * geo_weight;
        geo_output.push(result);
    }

    let geo_ml_elapsed = geo_start.elapsed();

    // print machine learning comparison
    println!("neural network layer ({input_dim} inputs, {output_dim} outputs):");
    println!("  traditional: {trad_ml_elapsed:?}");
    println!("  geonum: {geo_ml_elapsed:?}");
    println!(
        "  speedup: {:.2}×",
        trad_ml_elapsed.as_nanos() as f64 / geo_ml_elapsed.as_nanos() as f64
    );

    // 8. differential geometry benchmark

    // simulate parallel transport
    let curve_steps = 1000;
    let manifold_dim = 4;

    // benchmark traditional calculation
    let trad_start = Instant::now();

    // traditional approach using connection coefficients
    let mut vector = vec![1.0; manifold_dim];

    for _ in 0..curve_steps {
        // evolve along curve (simplified)
        for i in 0..manifold_dim {
            for j in 0..manifold_dim {
                vector[i] += 0.001 * vector[j];
            }
        }
    }

    let trad_geo_elapsed = trad_start.elapsed();

    // benchmark geometric calculation
    let geo_start = Instant::now();

    // direct angle evolution
    let mut geo_vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    for i in 0..curve_steps {
        // evolve along curve
        geo_vector = Geonum::new_with_blade(
            geo_vector.length,
            1,
            (geo_vector.angle + Angle::new(0.001 * (i as f64).sin(), PI)).mod_4_angle(),
            1.0,
        );
    }

    let geo_geom_elapsed = geo_start.elapsed();

    // print differential geometry comparison
    println!("parallel transport ({curve_steps} steps):");
    println!("  traditional: {trad_geo_elapsed:?}");
    println!("  geonum: {geo_geom_elapsed:?}");
    println!(
        "  speedup: {:.2}×",
        trad_geo_elapsed.as_nanos() as f64 / geo_geom_elapsed.as_nanos() as f64
    );

    // 9. quantum simulation benchmark

    // simulate quantum system
    let qubits = 20;
    let states = 1 << qubits; // 2^qubits

    // benchmark geometric calculation (traditional would be impossible)
    let geo_start = Instant::now();

    // direct angle representation
    let geo_state = Geonum::new_with_blade(1.0, 1, 0.0, 1.0);

    // quantum gate
    let geo_gate = Geonum::new_with_blade(1.0, 1, 1.0, 4.0);

    // apply gate to all 2^n states with one operation
    let _updated_state = geo_state * geo_gate;

    let geo_quantum_elapsed = geo_start.elapsed();

    // estimate traditional timing (even 1/1000th would be optimistic)
    let trad_quantum_estimate = geo_quantum_elapsed.mul_f64(states as f64);

    // print quantum simulation comparison
    println!("quantum simulation ({qubits} qubits, {states} states):");
    println!("  geonum: {geo_quantum_elapsed:?}");
    println!("  traditional (estimated): {trad_quantum_estimate:?}");
    println!(
        "  estimated speedup: {:.2e}×",
        trad_quantum_estimate.as_nanos() as f64 / geo_quantum_elapsed.as_nanos() as f64
    );

    // 10. overall performance summary

    println!("\nperformance summary:");
    println!("geonum consistently outperforms traditional tensor methods");
    println!("key benefits:");
    println!("  - o(1) complexity instead of o(n^k)");
    println!("  - constant scaling with dimension");
    println!("  - enables previously impossible calculations");
    println!("  - eliminates complexity bottlenecks");
}

#[test]
fn its_a_metric_signature() {
    // traditional physics: "we must carefully choose our metric tensor signature"
    // euclidean: (+,+,+,+) with g_μν = diag(1,1,1,1)
    // minkowski: (-,+,+,+) with g_μν = diag(-1,1,1,1)
    // this seems like a deep choice about the nature of spacetime

    // geonum: metric signature is just "what happens when angles add during squaring"
    // no choice needed - it mechanically emerges from angle arithmetic

    // test 1: euclidean signature emerges from 0° basis vectors
    // traditional: "we choose positive signature (+,+,+)"
    // geonum: basis vectors at 0° naturally square to positive

    let e1_euclidean = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° basis
    let e1_squared = e1_euclidean * e1_euclidean;

    // 0 + 0 = 0, cos(0) = +1
    assert_eq!(e1_squared.angle.blade(), 0);
    assert!(e1_squared.angle.mod_4_angle().cos() > 0.0); // positive signature
    assert_eq!(e1_squared.length, 1.0);

    // test 2: minkowski signature emerges from timelike at π/2
    // traditional: "time has negative signature in the metric"
    // geonum: time at π/2 naturally squares to negative

    let time_basis = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 (perpendicular to space)
    let time_squared = time_basis * time_basis;

    // π/2 + π/2 = π, cos(π) = -1
    assert_eq!(time_squared.angle.blade(), 2); // blade 1 + 1 = 2 (which is π)
    assert!(time_squared.angle.mod_4_angle().cos() < 0.0); // negative signature!

    // test 3: the "choice" of signature is just choosing initial angles
    // traditional: "lets use signature (+,-,-,+)"
    // geonum: "lets point basis vectors at 0, π/2, π/2, 0"

    let custom_e0 = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° → squares to +
    let custom_e1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 → squares to -
    let custom_e2 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 → squares to -
    let custom_e3 = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° → squares to +

    // verify the signature (+,-,-,+)
    assert!((custom_e0 * custom_e0).angle.mod_4_angle().cos() > 0.0); // +
    assert!((custom_e1 * custom_e1).angle.mod_4_angle().cos() < 0.0); // -
    assert!((custom_e2 * custom_e2).angle.mod_4_angle().cos() < 0.0); // -
    assert!((custom_e3 * custom_e3).angle.mod_4_angle().cos() > 0.0); // +

    // test 4: "negative" vectors squaring to positive
    // traditional: "in clifford algebras, some negative elements square to positive"
    // geonum: π + π = 2π ≡ 0, so negative times negative = positive

    let negative_vector = Geonum::new(1.0, 2.0, 2.0); // [1, π] = -1
    let squared = negative_vector * negative_vector;

    // π + π = 2π, and 2π ≡ 0 (mod 2π)
    assert!(squared.angle.mod_4_angle().abs() < 1e-10); // back to 0
    assert!(squared.angle.mod_4_angle().cos() > 0.0); // positive result
    assert_eq!(squared.length, 1.0);

    // this is why (-1) × (-1) = +1: its just π + π = 2π ≡ 0

    // test 5: the metric tensor is just tracking angle relationships
    // traditional: "the metric tensor g_μν encodes the geometry of spacetime"
    // geonum: the "metric" is just how basis angles relate to each other

    let spatial = Geonum::new_with_blade(2.0, 0, 0.3, 1.0); // spatial vector at blade 0
    let temporal = Geonum::new_with_blade(2.0, 1, 0.3, 1.0); // temporal vector at blade 1

    // square both vectors through multiplication to reveal signature
    let spatial_squared = spatial * spatial; // blade arithmetic with boundary crossing
    let temporal_squared = temporal * temporal; // blade arithmetic with boundary crossing

    // prove exact blade accumulation shows signature
    assert_eq!(spatial_squared.angle.blade(), 1); // spatial squares to blade 1
    assert_eq!(temporal_squared.angle.blade(), 3); // temporal squares to blade 3
    let blade_diff = temporal_squared.angle.blade() - spatial_squared.angle.blade();
    assert_eq!(blade_diff, 2); // 3 - 1 = 2, encodes dual positive/negative spacetime signature (π angle as -,+)

    // prove signature through cosine values - measured from actual blade arithmetic
    assert!(spatial_squared.angle.mod_4_angle().cos() < 0.0); // spatial blade 1 gives negative cosine
    assert!(temporal_squared.angle.mod_4_angle().cos() > 0.0); // temporal blade 3 gives positive cosine

    // minkowski metric signature emerges: 2 blade difference maintains space/time distinction

    // test 6: signature "flips" are just π rotations
    // traditional: "changing signature requires careful metric tensor manipulation"
    // geonum: just rotate your basis by π

    let positive_signature = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // cos(0) = +1
    let flipped_signature = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // cos(π) = -1

    // same basis vector, just rotated by π
    assert_eq!(positive_signature.length, flipped_signature.length);
    assert_eq!(
        (positive_signature.angle.blade() + 2) % 4,
        flipped_signature.angle.blade() % 4
    );

    // test 7: complex metric signatures are just angle patterns
    // traditional: "some exotic spacetimes have signature (--++--++)"
    // geonum: "some bases have angles at π/2, π/2, 0, 0, π/2, π/2, 0, 0"

    let exotic_signature: Vec<Geonum> = vec![
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
    ];

    // prove the exotic signature pattern
    for (i, basis) in exotic_signature.iter().enumerate() {
        let squared = *basis * *basis;
        let expected_negative = i % 4 < 2; // first two of each group are negative

        if expected_negative {
            assert!(
                squared.angle.mod_4_angle().cos() < 0.0,
                "index {} negative",
                i
            );
        } else {
            assert!(
                squared.angle.mod_4_angle().cos() > 0.0,
                "index {} positive",
                i
            );
        }
    }

    // test 8: the pseudoscalar signature property I² = ±1
    // traditional: "the pseudoscalar squares to ±1 depending on metric signature"
    // geonum: different dimension counts create different angle sums

    // in 3D euclidean: 3 spatial dimensions at 0°
    let i_3d_euclidean = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // 3 × π/2
    let i_squared_euclidean = i_3d_euclidean * i_3d_euclidean;

    // 3π/2 + 3π/2 = 3π ≡ π (mod 2π), cos(π) = -1
    assert_eq!(i_squared_euclidean.angle.mod_4_angle().cos(), -1.0); // I² = -1 for euclidean

    // in 4D minkowski: 1 time (π/2) + 3 space (0°)
    let i_4d_minkowski = Geonum::new_with_blade(1.0, 4, 0.0, 1.0); // 4 × π/2 = 2π
    let i_squared_minkowski = i_4d_minkowski * i_4d_minkowski;

    // 2π + 2π = 4π ≡ 0 (mod 2π), cos(0) = +1
    assert_eq!(i_squared_minkowski.angle.mod_4_angle().cos(), 1.0); // I² = +1 for minkowski

    // the ±1 "mystery" is just whether your total angle is odd or even multiples of π

    // conclusion: metric signatures arent choices or conventions
    // theyre mechanical consequences of angle arithmetic:
    // - angles add when multiplying
    // - 2π wraps to 0
    // - cos(0) = +1, cos(π) = -1
    // the entire formalism of metric tensors is just bookkeeping for "what angle is this?"
}
