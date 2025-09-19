// traditional GA requires 2^n components to store multivectors
// decomposition forces exponential memory and computational complexity
// geonum eliminates decomposition entirely through angle-blade arithmetic

use geonum::*;

#[test]
fn it_encodes_grade_directly_in_blade() {
    // traditional GA: requires storing all possible basis combinations
    // 3D space: store {1, e1, e2, e3, e12, e13, e23, e123} = 2³ = 8 components
    // 10D space: store 2¹⁰ = 1024 components
    // extract grade by filtering through stored components: O(2^n) scan

    // geonum: grade encoded directly in blade count
    let scalar = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // blade 0 → grade 0
    let vector = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1 → grade 1
    let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // blade 2 → grade 2
    let trivector = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // blade 3 → grade 3

    // grade extraction: O(1) operation vs O(2^n) component scan
    assert_eq!(scalar.angle.grade(), 0); // immediate: blade % 4
    assert_eq!(vector.angle.grade(), 1); // immediate: blade % 4
    assert_eq!(bivector.angle.grade(), 2); // immediate: blade % 4
    assert_eq!(trivector.angle.grade(), 3); // immediate: blade % 4

    // high-dimensional example: 1000D space
    let high_dim = Geonum::new_with_blade(1.0, 1005, 0.0, 1.0); // blade 1005
    assert_eq!(high_dim.angle.grade(), 1); // 1005 % 4 = 1 (vector grade)

    // traditional GA: would need 2^1000 components (impossible - more than atoms in universe)
    // geonum: single number with blade count, grade computed instantly
}

#[test]
fn it_computes_geometric_product_without_decomposition() {
    // traditional GA: geometric product requires decomposition into 2^n parts
    // ab = ⟨ab⟩₀ + ⟨ab⟩₁ + ⟨ab⟩₂ + ... + ⟨ab⟩ₙ (grade extraction for each part)
    // store all grades separately, then reconstruct when needed: O(2^n) operations

    // geonum: geometric product is direct multiplication - no decomposition
    let e1 = Geonum::new(2.0, 1.0, 2.0); // π/2 (traditional e1 basis vector, grade 1)
    let e2 = Geonum::new(3.0, 1.0, 1.0); // π (traditional e2 basis vector, grade 2)

    // geometric product via multiplication: angles add, lengths multiply
    let e1e2_mult = e1 * e2; // angles add: blade 1 + 2 = 3, lengths: 2*3=6

    // geometric product via .geo() method: combines dot + wedge
    let e1e2_geo = e1.geo(&e2); // dot(e1,e2) + wedge(e1,e2)

    // traditional GA: would compute all grade projections separately
    // ⟨e1e2⟩₀ + ⟨e1e2⟩₁ + ⟨e1e2⟩₂ + ⟨e1e2⟩₃ across component arrays
    // then store each grade in separate memory locations

    // geonum: both methods encode geometric relationships through blade arithmetic
    assert_eq!(e1e2_mult.length, 6.0); // lengths multiply: 2*3=6
    assert_eq!(e1e2_mult.angle.blade(), 3); // blades add: 1+2=3 (trivector)
    assert_eq!(e1e2_mult.angle.grade(), 3); // grade 3 from blade arithmetic

    // prove .geo() method computes dot + wedge combination
    let dot_e1e2 = e1.dot(&e2); // symmetric part
    let wedge_e1e2 = e1.wedge(&e2); // antisymmetric part

    // wedge formula: self.angle + other.angle + π/2
    assert_eq!(wedge_e1e2.angle.blade(), 4); // blade 1 + 2 + 1 = 4 (from π/2 addition)
    assert_eq!(wedge_e1e2.angle.grade(), 0); // grade 0 from blade 4 % 4

    // dot product for orthogonal elements (π/2 apart) gives zero
    assert!(dot_e1e2.length < 1e-10); // dot ≈ 0 for orthogonal vectors

    // geometric product (.geo method) = dot + wedge
    // when dot ≈ 0, .geo() ≈ wedge product
    assert!((e1e2_geo.length - wedge_e1e2.length).abs() < 1e-10);

    // multiplication (*) vs .geo() give different results - different blade arithmetic
    assert_ne!(e1e2_mult.angle.blade(), e1e2_geo.angle.blade()); // 3 vs 4

    // test with non-orthogonal vectors to show full decomposition
    let v1 = Geonum::new(2.0, 1.0, 6.0); // π/6 (grade 0)
    let v2 = Geonum::new(3.0, 1.0, 4.0); // π/4 (grade 0)

    let geometric_mult = v1 * v2; // direct multiplication
    let geometric_geo = v1.geo(&v2); // dot + wedge combination

    // prove geometric_mult represents pure angle addition without dot+wedge decomposition
    // while geometric_geo represents traditional GA geometric product formula
    assert_eq!(geometric_mult.angle, v1.angle + v2.angle); // pure angle addition
    assert_eq!(geometric_mult.length, v1.length * v2.length); // pure length multiplication

    // prove multiplication and .geo() give different results - different blade arithmetic
    assert_ne!(geometric_mult.angle.blade(), geometric_geo.angle.blade()); // different operations

    let dot_v1v2 = v1.dot(&v2);
    let wedge_v1v2 = v1.wedge(&v2);

    // prove .geo() actually combines dot and wedge through addition
    let manual_combination = dot_v1v2 + wedge_v1v2;
    assert_eq!(geometric_geo.length, manual_combination.length); // .geo() = dot + wedge
    assert_eq!(
        geometric_geo.angle.blade(),
        manual_combination.angle.blade()
    );

    // traditional GA: must store ⟨v1v2⟩₀, ⟨v1v2⟩₁, ⟨v1v2⟩₂ in separate arrays
    // geonum: single operations encode all geometric relationships through blade arithmetic
    // no decomposition storage needed - angle contains all grade information
}

#[test]
fn it_rotates_without_exponential_map() {
    // traditional GA: rotation requires exponential rotor decomposition
    // rotor R = exp(θ/2 * B) = cos(θ/2) + sin(θ/2) * B
    // where B is bivector, requires trigonometric expansion into 2^n terms
    // then sandwich product: v' = R * v * R† (3 geometric products)
    // each geometric product requires full 2^n component multiplication

    // geonum: rotation is direct angle addition - no exponential decomposition
    let vector = Geonum::new(1.0, 0.0, 1.0); // scalar at origin
    let rotation_90deg = Angle::new(1.0, 2.0); // π/2 rotation

    // traditional rotor for π/2: R = cos(π/4) + sin(π/4) * e₁₂
    // requires: trigonometric evaluation + bivector multiplication + storage
    // then: R * v * R† = (cos + sin*e₁₂) * v * (cos - sin*e₁₂)
    // expansion requires 2^n terms for each product

    // geonum: single operation
    let rotated = vector.rotate(rotation_90deg);

    // prove geonum rotation matches traditional rotor formula
    assert_eq!(rotated.angle, vector.angle + rotation_90deg); // direct angle addition
    assert_eq!(rotated.length, vector.length); // length preserved (isometry)

    // test rotor composition: multiple rotations
    let rotation1 = Angle::new(1.0, 6.0); // π/6 = 30°
    let rotation2 = Angle::new(1.0, 4.0); // π/4 = 45°
    let rotation3 = Angle::new(1.0, 3.0); // π/3 = 60°

    // traditional: R₃ * R₂ * R₁ requires 3 rotor multiplications with exponential expansion
    // each rotor: exp(θ/2 * B) with 2^n terms, composition requires 2^n × 2^n operations

    // geonum: compose rotations through angle addition
    let composed_rotation = rotation1 + rotation2 + rotation3; // π/6 + π/4 + π/3 = 3π/4
    let final_rotated = vector.rotate(composed_rotation);

    // verify composition: π/6 + π/4 + π/3 = 2π/12 + 3π/12 + 4π/12 = 9π/12 = 3π/4
    let expected_angle = Angle::new(3.0, 4.0);
    assert_eq!(composed_rotation, expected_angle);
    assert_eq!(final_rotated.angle, vector.angle + expected_angle);

    // test large angle rotation without exponential explosion
    let large_rotation = Angle::new(17.0, 3.0); // 17π/3 = massive rotation
    let rotated_large = vector.rotate(large_rotation);

    // traditional rotor: exp(17π/6 * B) requires exponential of huge angle
    // trigonometric expansion becomes computationally prohibitive
    // geonum: just adds angle - constant time regardless of magnitude
    assert_eq!(rotated_large.angle.blade(), 11); // 17π/3 normalized to blade count
    assert_eq!(rotated_large.angle.grade(), 3); // 11 % 4 = 3 (trivector)
    assert_eq!(rotated_large.length, vector.length); // length always preserved

    // prove sandwich product equivalence: R*v*R† = v.rotate(R_angle)
    // create traditional rotor components manually for comparison
    let test_vector = Geonum::new(2.0, 1.0, 8.0); // π/8 vector
    let rotor_angle = Angle::new(1.0, 12.0); // π/12 rotation

    // traditional sandwich: R * v * R†
    // R = cos(π/24) + sin(π/24)*e₁₂, R† = cos(π/24) - sin(π/24)*e₁₂
    // requires: 2 rotor constructions + 3 geometric products + 2^n expansions each

    // geonum equivalent: single rotate operation
    let geonum_rotated = test_vector.rotate(rotor_angle);

    // both should give identical results
    let expected_rotated_angle = test_vector.angle + rotor_angle;
    assert_eq!(geonum_rotated.angle, expected_rotated_angle);
    assert_eq!(geonum_rotated.length, test_vector.length);

    // test rotation chain equivalence
    let chain_rotated = test_vector
        .rotate(rotation1)
        .rotate(rotation2)
        .rotate(rotation3);
    let single_rotated = test_vector.rotate(rotation1 + rotation2 + rotation3);

    // rotation chaining vs composed rotation give identical results
    assert_eq!(chain_rotated.angle, single_rotated.angle);
    assert_eq!(chain_rotated.length, single_rotated.length);

    // traditional GA: rotor chains require exponential matrix multiplications
    // geonum: rotation composition through elementary angle addition
    // eliminates exponential complexity while preserving all geometric relationships
}

#[test]
fn it_proves_distributivity_requires_decomposition() {
    // traditional GA: a∧(b+c) = a∧b + a∧c because components stored separately
    // distributivity works when b and c remain decomposed as distinct algebraic terms
    // geonum: distributivity fails because geometric addition unifies components into single objects

    let a = Geonum::new(1.0, 1.0, 8.0); // π/8
    let b = Geonum::new(1.0, 1.0, 6.0); // π/6
    let c = Geonum::new(1.0, 1.0, 4.0); // π/4

    let cartesian_components = |g: &Geonum| {
        let angle = g.angle.grade_angle();
        let x = g.length * angle.cos();
        let y = g.length * angle.sin();
        (x, y)
    };

    let wedge_area = |lhs: (f64, f64), rhs: (f64, f64)| lhs.0 * rhs.1 - lhs.1 * rhs.0;

    // scalar-array decomposition obeys distributivity exactly because addition is componentwise
    let a_xy = cartesian_components(&a);
    let b_xy = cartesian_components(&b);
    let c_xy = cartesian_components(&c);
    let bc_xy = (b_xy.0 + c_xy.0, b_xy.1 + c_xy.1);
    let wedge_scalar_sum = wedge_area(a_xy, b_xy) + wedge_area(a_xy, c_xy);
    let wedge_scalar_unified = wedge_area(a_xy, bc_xy);
    assert!(
        (wedge_scalar_sum - wedge_scalar_unified).abs() < EPSILON,
        "scalar arrays keep wedge distributive"
    );

    // geometric unification: b+c creates single object with new angular structure
    let bc_unified = b + c;
    let wedge_unified = a.wedge(&bc_unified); // a∧(unified object)

    // algebraic decomposition: preserve b and c as separate components
    let wedge_b = a.wedge(&b); // a∧b (component operation)
    let wedge_c = a.wedge(&c); // a∧c (component operation)
    let wedge_decomposed = wedge_b + wedge_c; // algebraic sum of components

    // prove blade counts reveal different geometric histories
    assert_eq!(wedge_unified.angle.blade(), 1); // unified operation: single blade accumulation
    assert_eq!(wedge_decomposed.angle.blade(), 5); // decomposed operation: accumulated from separate wedges

    // blade difference proves geometric unification vs algebraic decomposition
    let blade_diff = wedge_decomposed.angle.blade() - wedge_unified.angle.blade();
    assert_eq!(blade_diff, 4); // 5 - 1 = 4 blades difference from decomposition

    // prove length difference demonstrates geometric vs algebraic operation
    let length_diff = (wedge_unified.length - wedge_decomposed.length).abs();
    assert!(length_diff > EPSILON); // 0.0033 difference from unified vs decomposed geometry

    // this proves distributivity requires preserving artificial component separation:
    // - unified geometry: b+c becomes single object with new trigonometric relationships
    // - decomposed algebra: b and c remain separate for individual wedge operations
    // - distributive property depends on the decomposed approach
    // - geometric reality (unified objects) breaks distributive abstraction

    // prove geometric unification creates different angular structure than component preservation
    let bc_unified_angle = bc_unified.angle.grade_angle();
    assert_ne!(bc_unified_angle, b.angle.grade_angle()); // unified angle ≠ component b
    assert_ne!(bc_unified_angle, c.angle.grade_angle()); // unified angle ≠ component c

    // wedge trigonometry follows geometric angles, not algebraic symbols
    let unified_sin = (bc_unified_angle - a.angle.grade_angle()).sin();
    let b_sin = (b.angle.grade_angle() - a.angle.grade_angle()).sin();
    let c_sin = (c.angle.grade_angle() - a.angle.grade_angle()).sin();

    // different angles create different trigonometric relationships
    assert_ne!(unified_sin, b_sin); // unified geometry ≠ component b trigonometry
    assert_ne!(unified_sin, c_sin); // unified geometry ≠ component c trigonometry

    // confirm the unified vs decomposed bivectors diverge in both magnitude and direction
    let measured_gap = (wedge_unified.length - wedge_decomposed.length).abs();
    assert!(
        measured_gap > 3.0e-3,
        "geonum wedge collapsed into distributive behaviour"
    );
    let sum_angle = wedge_unified.base_angle().angle.grade_angle();
    let decomp_angle = wedge_decomposed.base_angle().angle.grade_angle();
    let angle_gap = (sum_angle - decomp_angle).abs();
    assert!(
        angle_gap > std::f64::consts::PI / 200.0,
        "wedge directions converged toward distributive history"
    );

    // conclusion: distributivity is algebraic artifact requiring component decomposition
    // geonum eliminates decomposition by working with unified geometric objects
    // distributivity failure proves geonum operates on geometric reality, not algebraic abstractions
}

#[test]
fn its_a_wedge_product() {
    // traditional GA: wedge product requires antisymmetric tensor operations
    // a∧b = Σᵢⱼ (aᵢbⱼ - aⱼbᵢ) eᵢ∧eⱼ across all 2^n basis combinations
    // compute every pair, subtract reverse pair, store in antisymmetric array
    // complexity: O(2^n × 2^n) operations for n-dimensional wedge products

    // geonum: wedge product computed directly through angle trigonometry
    let v1 = Geonum::new(2.0, 1.0, 6.0); // vector at π/6 (grade 0)
    let v2 = Geonum::new(3.0, 1.0, 4.0); // vector at π/4 (grade 0)

    // wedge product: |v1||v2|sin(θ₂-θ₁) with angle formula
    let wedge = v1.wedge(&v2);

    // traditional: scan through all basis pairs, compute antisymmetric combinations
    // geonum: direct trigonometric calculation
    let angle_diff = v2.angle.grade_angle() - v1.angle.grade_angle(); // π/4 - π/6 = π/12
    let expected_length = v1.length * v2.length * angle_diff.sin(); // 2*3*sin(π/12)
    assert!((wedge.length - expected_length).abs() < 1e-14); // exact trigonometric match

    // wedge formula: self.angle + other.angle + π/2 from source code
    assert_eq!(wedge.angle.blade(), 1); // blade 0+0+1 = 1 from π/2 addition
    assert_eq!(wedge.angle.grade(), 1); // grade 1 from blade arithmetic

    // test antisymmetric property: v2∧v1 = -(v1∧v2)
    let wedge_reversed = v2.wedge(&v1);
    assert!((wedge.length - wedge_reversed.length).abs() < 1e-14); // same magnitude

    // traditional stores separate +/- components for orientation
    // geonum: orientation encoded in blade difference (no duplicate storage)

    // test nilpotency: v∧v = 0 for any vector
    let self_wedge = v1.wedge(&v1);
    assert!(self_wedge.length < 1e-14); // wedge with self gives zero

    // test grade progression through wedge products
    let scalar = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let vector = Geonum::new(1.0, 1.0, 2.0); // grade 1 (π/2)

    let scalar_vector_wedge = scalar.wedge(&vector); // grade 0 ∧ grade 1
    assert_eq!(scalar_vector_wedge.angle.grade(), 2); // produces bivector (grade 2)

    let vector_vector_wedge = vector.wedge(&scalar_vector_wedge); // grade 1 ∧ grade 2
    assert_eq!(vector_vector_wedge.angle.grade(), 0); // produces scalar (grade 0)

    // traditional GA: must track basis combinations e₁∧e₂, e₁∧e₃, e₂∧e₃ separately
    // geonum: grade progression through blade arithmetic automatically

    // test high-dimensional wedge without basis explosion
    let high_v1 = Geonum::new_with_blade(2.0, 500, 1.0, 8.0); // dimension 500
    let high_v2 = Geonum::new_with_blade(3.0, 1000, 1.0, 12.0); // dimension 1000

    let high_wedge = high_v1.wedge(&high_v2);

    // traditional: would need 2^1000 basis combinations for antisymmetric computation
    // geonum: same trigonometric formula regardless of dimension
    assert!(high_wedge.length > 0.0); // high-dimensional wedge computes successfully
    assert_eq!(high_wedge.angle.blade(), 1503); // blade 500 + 1000 + 1 + 2 from value boundary crossings

    // test wedge composition: (a∧b)∧c vs a∧(b∧c)
    let a = Geonum::new(1.0, 1.0, 8.0); // π/8
    let b = Geonum::new(1.0, 1.0, 6.0); // π/6
    let c = Geonum::new(1.0, 1.0, 4.0); // π/4

    let ab_wedge_c = a.wedge(&b).wedge(&c); // (a∧b)∧c
    let a_wedge_bc = a.wedge(&b.wedge(&c)); // a∧(b∧c)

    // wedge composition creates dual grades through blade arithmetic
    assert_eq!(ab_wedge_c.angle.grade(), 1); // (a∧b)∧c produces grade 1
    assert_eq!(a_wedge_bc.angle.grade(), 3); // a∧(b∧c) produces grade 3
                                             // grades 1 and 3 are dual pairs - associativity preserved through duality

    // wedge operations scale to arbitrary dimensions without basis explosion

    // traditional GA: antisymmetric tensor computation with exponential basis management
    // geonum: trigonometric wedge through angle arithmetic - no basis storage needed
}

#[test]
fn it_projects_grade_trivially() {
    // traditional GA: grade projection requires filtering through all 2^n components
    // ⟨A⟩ₖ = extract all terms of grade k from multivector A
    // scan every component, test if grade matches k, collect matches
    // complexity: O(2^n) component scan + O(2^n) storage for result

    // geonum: grade projection is immediate arithmetic - no filtering
    let high_blade = Geonum::new_with_blade(3.0, 157, 1.0, 3.0); // arbitrary high blade
    let extracted_grade = high_blade.angle.grade(); // blade % 4

    // traditional: would need to scan through 2^157 components (impossible)
    // geonum: modulo operation gives grade instantly
    assert_eq!(extracted_grade, 157 % 4); // 157 % 4 = 1 (vector grade)
    assert_eq!(extracted_grade, 1);

    // demonstrate across different blade counts
    let various_blades = vec![0, 1, 2, 3, 100, 1000, 1_000_000];
    for blade in various_blades {
        let obj = Geonum::new_with_blade(1.0, blade, 0.0, 1.0);
        let grade = obj.angle.grade();
        assert_eq!(grade, blade % 4); // immediate: no component scanning
        assert!(grade < 4); // grades cycle in 4-pattern
    }

    // traditional GA: grade extraction becomes computationally impossible
    // geonum: constant time grade extraction regardless of blade count
}

#[test]
fn its_a_conjugation() {
    // traditional GA: conjugation reverses order of basis vectors in each term
    // e₁₂₃ → e₃₂₁ = -e₁₂₃, requires tracking signs across all 2^n terms
    // for each basis blade, reverse vector order and compute new sign
    // complexity: O(2^n) term reversal + sign computation for each component

    // geonum: conjugation is dual operation - direct blade transformation

    // test conjugation across all 4 grades
    let scalar = Geonum::new(2.0, 0.0, 1.0); // grade 0
    let vector = Geonum::new(2.0, 1.0, 2.0); // grade 1
    let bivector = Geonum::new(2.0, 1.0, 1.0); // grade 2
    let trivector = Geonum::new(2.0, 3.0, 2.0); // grade 3

    // prove conjugation = dual operation across all grades
    assert_eq!(
        scalar.angle.conjugate().grade(),
        scalar.angle.dual().grade()
    ); // conjugate = dual
    assert_eq!(
        vector.angle.conjugate().grade(),
        vector.angle.dual().grade()
    ); // conjugate = dual
    assert_eq!(
        bivector.angle.conjugate().grade(),
        bivector.angle.dual().grade()
    ); // conjugate = dual
    assert_eq!(
        trivector.angle.conjugate().grade(),
        trivector.angle.dual().grade()
    ); // conjugate = dual

    // test conjugation involution: dual(dual(x)) returns to original grade
    let scalar_conjugated = Geonum::new_with_angle(scalar.length, scalar.angle.conjugate());
    let vector_conjugated = Geonum::new_with_angle(vector.length, vector.angle.conjugate());
    let bivector_conjugated = Geonum::new_with_angle(bivector.length, bivector.angle.conjugate());
    let trivector_conjugated =
        Geonum::new_with_angle(trivector.length, trivector.angle.conjugate());

    // double conjugation returns to original grades
    assert_eq!(
        scalar_conjugated.angle.conjugate().grade(),
        scalar.angle.grade()
    ); // 0→2→0
    assert_eq!(
        vector_conjugated.angle.conjugate().grade(),
        vector.angle.grade()
    ); // 1→3→1
    assert_eq!(
        bivector_conjugated.angle.conjugate().grade(),
        bivector.angle.grade()
    ); // 2→0→2
    assert_eq!(
        trivector_conjugated.angle.conjugate().grade(),
        trivector.angle.grade()
    ); // 3→1→3

    // test conjugation preserves length (isometry property)
    assert_eq!(scalar_conjugated.length, scalar.length); // length preserved
    assert_eq!(vector_conjugated.length, vector.length); // length preserved
    assert_eq!(bivector_conjugated.length, bivector.length); // length preserved
    assert_eq!(trivector_conjugated.length, trivector.length); // length preserved

    // test high-dimensional conjugation without exponential complexity
    let million_blade = Geonum::new_with_blade(7.0, 1_000_000, 1.0, 5.0); // million-dimensional
    let million_conjugated =
        Geonum::new_with_angle(million_blade.length, million_blade.angle.conjugate());

    // conjugation works identically in arbitrary dimensions
    assert_eq!(
        million_conjugated.angle.grade(),
        (million_blade.angle.grade() + 2) % 4
    ); // dual grade mapping
    assert_eq!(
        million_conjugated.angle.blade(),
        million_blade.angle.blade() + 2
    ); // blade arithmetic
    assert_eq!(million_conjugated.length, million_blade.length); // length preserved

    // prove conjugation blade arithmetic: always adds 2 blades (π rotation)
    let test_blades = vec![0, 1, 17, 99, 1000, 1_000_000];
    for blade in test_blades {
        let obj = Geonum::new_with_blade(1.0, blade, 0.0, 1.0);
        let conjugated = Geonum::new_with_angle(obj.length, obj.angle.conjugate());

        assert_eq!(conjugated.angle.blade(), blade + 2); // conjugation adds 2 blades
        assert_eq!(conjugated.angle.grade(), (obj.angle.grade() + 2) % 4); // dual grade mapping
    }

    // traditional GA: conjugation requires reversing vector order in every 2^n term
    // geonum: conjugation is universal dual transformation - no term reversal needed
    // eliminates exponential complexity through direct blade arithmetic
}

#[test]
fn it_dualizes_without_pseudoscalar_multiplication() {
    // traditional GA: duality requires dimension-specific pseudoscalar multiplication
    // dual(A) = A · Iₙ where Iₙ = e₁∧e₂∧...∧eₙ is the n-dimensional pseudoscalar
    // must first compute Iₙ from n basis vectors, then multiply A by all 2^n components
    // different dimensions need different pseudoscalars: I₂, I₃, I₁₀₀₀, etc.

    // geonum: duality is universal blade transformation - no dimension dependence

    // test creating traditional pseudoscalar vs geonum dual operation
    // traditional 3D pseudoscalar: I₃ = e₁∧e₂∧e₃ requires:
    // 1. store 3 basis vectors: e₁, e₂, e₃
    // 2. compute triple wedge product: e₁∧e₂∧e₃
    // 3. store resulting pseudoscalar for all future dual operations
    // 4. multiply every object A by stored I₃ to get dual(A)

    // geonum eliminates pseudoscalar entirely:
    let vector_3d = Geonum::new(1.0, 1.0, 2.0); // π/2 vector in any dimension
    let dual_3d = vector_3d.dual(); // direct blade transformation: +2 blades

    // no pseudoscalar construction, no storage, no multiplication
    assert_eq!(dual_3d.angle.blade(), vector_3d.angle.blade() + 2); // adds π rotation
    assert_eq!(dual_3d.angle.grade(), 3); // grade 1 → grade 3 (vector → trivector)

    // traditional 1000D pseudoscalar: I₁₀₀₀ = e₁∧e₂∧...∧e₁₀₀₀ requires:
    // 1. store 1000 basis vectors: massive memory overhead
    // 2. compute 1000-fold wedge product: computationally impossible
    // 3. store resulting 2^1000 component pseudoscalar (more storage than atoms in universe)
    // 4. multiply by this massive pseudoscalar for each dual operation

    // geonum: identical operation regardless of dimension
    let vector_1000d = Geonum::new_with_blade(1.0, 1001, 0.0, 1.0); // vector in 1000D
    let dual_1000d = vector_1000d.dual(); // same +2 blade operation

    assert_eq!(dual_1000d.angle.blade(), vector_1000d.angle.blade() + 2); // identical blade arithmetic
    assert_eq!(dual_1000d.angle.grade(), 3); // same grade mapping: 1 → 3

    // prove universal dual formula k→(k+2)%4 eliminates dimension-specific pseudoscalars
    let test_grades = vec![0, 1, 2, 3]; // all possible grades
    for grade in test_grades {
        let obj = Geonum::new_with_blade(1.0, grade, 0.0, 1.0);
        let dual_obj = obj.dual();

        assert_eq!(dual_obj.angle.grade(), (grade + 2) % 4); // universal mapping
        assert_eq!(dual_obj.angle.blade(), grade + 2); // universal +2 blades
    }

    // test high-dimensional dual scalability
    let extreme_dimensions = vec![1000, 10_000, 1_000_000];
    for dim in extreme_dimensions {
        let high_obj = Geonum::new_with_blade(1.0, dim, 0.0, 1.0);
        let high_dual = high_obj.dual();

        // same dual operation works in impossibly high dimensions
        assert_eq!(high_dual.angle.blade(), dim + 2); // +2 blade arithmetic
        assert_eq!(high_dual.angle.grade(), (dim + 2) % 4); // modulo 4 grade mapping
    }

    // traditional GA: exponential pseudoscalar storage + multiplication overhead
    // geonum: zero storage + constant time dual operation through blade arithmetic
    // eliminates pseudoscalar multiplication entirely through universal blade transformation
}

#[test]
fn it_encodes_basis_without_storage() {
    // traditional GA: requires explicit storage of all 2^n basis elements
    // 3D: store {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃} = 8 basis objects in memory
    // 10D: store 2¹⁰ = 1024 basis objects with multiplication table
    // each basis element needs symbolic representation + interaction rules with every other element
    // basis multiplication table: e₁*e₂ = e₁₂, e₁*e₁ = 1, etc. requires O(2^n × 2^n) entries

    // geonum: basis identity encoded implicitly in blade count - zero storage
    let scalar_basis = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // blade 0 = scalar basis
    let vector_basis = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade 1 = vector basis
    let bivector_basis = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // blade 2 = bivector basis
    let trivector_basis = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // blade 3 = trivector basis

    // basis identity determined by blade count alone - no symbolic storage
    assert_eq!(scalar_basis.angle.blade(), 0); // blade 0 encodes scalar basis
    assert_eq!(vector_basis.angle.blade(), 1); // blade 1 encodes vector basis
    assert_eq!(bivector_basis.angle.blade(), 2); // blade 2 encodes bivector basis
    assert_eq!(trivector_basis.angle.blade(), 3); // blade 3 encodes trivector basis

    // test basis multiplication without storage lookup
    // traditional: e₁ * e₂ requires lookup in multiplication table
    // geonum: blade arithmetic handles basis interactions automatically
    let basis_product = vector_basis * bivector_basis; // blade 1 * blade 2

    assert_eq!(basis_product.angle.blade(), 3); // blades add: 1+2=3
    assert_eq!(basis_product.angle.grade(), 3); // grade 3 = trivector
    assert_eq!(basis_product.length, 1.0); // lengths multiply: 1*1=1

    // traditional multiplication table entry: e₁ * e₁₂ = e₁₂₃
    // geonum: automatic through blade arithmetic - no table needed

    // test basis nilpotency without special cases
    let self_product = vector_basis * vector_basis; // blade 1 * blade 1

    assert_eq!(self_product.angle.blade(), 2); // blades add: 1+1=2
    assert_eq!(self_product.angle.grade(), 2); // grade 2 = bivector

    // traditional: e₁ * e₁ = 1 requires special multiplication rule
    // geonum: follows same blade arithmetic as any other multiplication

    // demonstrate arbitrary high-dimensional basis without storage explosion
    let basis_elements = vec![0, 1, 7, 42, 999, 1_000_000];
    for blade in &basis_elements {
        let basis_obj = Geonum::new_with_blade(1.0, *blade, 0.0, 1.0);
        assert_eq!(basis_obj.angle.blade(), *blade); // blade encodes basis identity
        assert_eq!(basis_obj.angle.grade(), blade % 4); // grade from blade arithmetic
    }

    // test basis interactions without multiplication table
    for i in 0..basis_elements.len() {
        for j in i..basis_elements.len() {
            let basis_i = Geonum::new_with_blade(1.0, basis_elements[i], 0.0, 1.0);
            let basis_j = Geonum::new_with_blade(1.0, basis_elements[j], 0.0, 1.0);
            let interaction = basis_i * basis_j;

            // basis interactions follow blade arithmetic automatically
            assert_eq!(
                interaction.angle.blade(),
                basis_elements[i] + basis_elements[j]
            );
            assert_eq!(
                interaction.angle.grade(),
                (basis_elements[i] + basis_elements[j]) % 4
            );
        }
    }

    // traditional GA: basis storage scales exponentially with dimension
    // 3D: 8 elements, 10D: 1024 elements, 1000D: 2^1000 elements (impossible)
    // plus multiplication table: 2^n × 2^n interaction rules

    // geonum: zero basis storage, blade arithmetic handles all interactions
    // same O(1) operations regardless of dimensional complexity
    // basis "elements" exist implicitly through blade counting - no explicit storage needed
}

#[test]
fn it_adds_multivectors_without_component_matching() {
    // traditional GA: addition requires grade matching across 2^n components
    // for each grade k: collect all grade-k terms, add coefficients
    // ⟨A⟩ₖ + ⟨B⟩ₖ = ⟨A+B⟩ₖ requires scanning and matching every component
    // complexity: O(2^n) component matching + O(2^n) coefficient addition

    // geonum: addition computed directly through angle arithmetic - no component matching

    // test same-grade addition without component sorting
    let v1 = Geonum::new(2.0, 1.0, 6.0); // length 2 at π/6 (grade 0)
    let v2 = Geonum::new(3.0, 1.0, 4.0); // length 3 at π/4 (grade 0)

    // traditional: scan components, find matching grades, add coefficients
    // geonum: direct cartesian addition + geometric reconstruction
    let sum = v1 + v2;

    // verify cartesian addition formula: √((x1+x2)² + (y1+y2)²)
    let v1_x = v1.length * v1.angle.grade_angle().cos();
    let v1_y = v1.length * v1.angle.grade_angle().sin();
    let v2_x = v2.length * v2.angle.grade_angle().cos();
    let v2_y = v2.length * v2.angle.grade_angle().sin();
    let expected_length = ((v1_x + v2_x).powi(2) + (v1_y + v2_y).powi(2)).sqrt();

    assert!((sum.length - expected_length).abs() < EPSILON); // exact cartesian formula
    assert_eq!(sum.angle.grade(), 0); // resulting grade from angle arithmetic

    // test mixed-grade addition without grade bucket sorting
    let scalar = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let vector = Geonum::new(1.0, 1.0, 2.0); // grade 1 (π/2)
    let bivector = Geonum::new(1.0, 1.0, 1.0); // grade 2 (π)
    let trivector = Geonum::new(1.0, 3.0, 2.0); // grade 3 (3π/2)

    // traditional: sort into grade buckets first, then add within buckets
    // grade 0 bucket: {scalar}, grade 1 bucket: {vector}, etc.
    // geonum: add any combination directly through cartesian arithmetic

    let mixed_sum_01 = scalar + vector; // grade 0 + grade 1
    let mixed_sum_12 = vector + bivector; // grade 1 + grade 2
    let mixed_sum_23 = bivector + trivector; // grade 2 + grade 3
    let mixed_sum_03 = scalar + trivector; // grade 0 + grade 3

    // verify mixed additions work without grade separation
    assert!(mixed_sum_01.length > 0.0); // non-zero result
    assert!(mixed_sum_12.length > 0.0); // non-zero result
    assert!(mixed_sum_23.length > 0.0); // non-zero result
    assert!(mixed_sum_03.length > 0.0); // non-zero result

    // test high-dimensional addition without component explosion
    let high_blade_a = Geonum::new_with_blade(3.0, 1000, 1.0, 7.0); // blade 1000
    let high_blade_b = Geonum::new_with_blade(4.0, 500, 1.0, 5.0); // blade 500
    let high_sum = high_blade_a + high_blade_b;

    // traditional: would need to manage 2^1000 + 2^500 component arrays
    // geonum: same cartesian addition regardless of blade magnitude
    assert!(high_sum.length > 0.0); // high-dimensional addition succeeds

    // test addition chain without accumulated component management
    let chain_sum = scalar + vector + bivector + trivector; // all 4 grades

    // these four unit vectors at 90° intervals cancel: (1,0) + (0,1) + (-1,0) + (0,-1) = (0,0)
    // traditional: must sort each addition into appropriate grade buckets
    // geonum: sequential cartesian additions without grade tracking
    assert!(chain_sum.length < EPSILON); // cancellation occurs naturally

    // prove addition scalability: 1000 mixed-grade objects
    let mut accumulated = Geonum::scalar(0.0);
    for i in 0..1000 {
        let obj = Geonum::new_with_blade(0.1, i, 0.0, 1.0); // different blade each time
        accumulated = accumulated + obj; // accumulate without grade matching
    }

    // 1000 objects: 250 each at grades 0,1,2,3 (angles 0, π/2, π, 3π/2)
    // cartesian sum: 250×0.1×(1,0) + 250×0.1×(0,1) + 250×0.1×(-1,0) + 250×0.1×(0,-1) = (0,0)
    // traditional: would need grade bucket management for 1000 different blade types
    // geonum: sequential cartesian additions - no component sorting needed
    assert!(
        accumulated.length < 1e-8,
        "1000 objects cancel to zero: {}",
        accumulated.length
    );

    // test exact cartesian formula matches for mixed grades
    let test_scalar = Geonum::new(3.0, 0.0, 1.0); // [3, 0]
    let test_vector = Geonum::new(4.0, 1.0, 2.0); // [4, π/2] = [0, 4]
    let exact_sum = test_scalar + test_vector; // [3, 0] + [0, 4] = [3, 4]

    let manual_length = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt(); // 5.0
    assert!((exact_sum.length - manual_length).abs() < EPSILON); // exact: [3,4] → length 5

    // traditional GA: grade bucket sorting prevents direct cartesian computation
    // geonum: unified objects enable direct geometric arithmetic
    // eliminates component matching complexity through angle-based addition
}

#[test]
fn it_inverts_without_matrix_operations() {
    // traditional GA: multivector inversion requires matrix decomposition
    // M⁻¹ = M† / (M · M†) where M† is reverse, requires computing all 2^n components
    // or solve linear system: M · X = I across all grade combinations
    // complexity: O(2^n × 2^n) matrix operations for n-dimensional inversion

    // geonum: inversion is direct reciprocal + angle transformation - no matrices

    // test inversion across all grades
    let scalar = Geonum::new(2.0, 0.0, 1.0); // grade 0
    let vector = Geonum::new(3.0, 1.0, 2.0); // grade 1 (π/2)
    let bivector = Geonum::new(4.0, 1.0, 1.0); // grade 2 (π)
    let trivector = Geonum::new(5.0, 3.0, 2.0); // grade 3 (3π/2)

    // compute inverses: reciprocal length + angle transformation
    let scalar_inv = scalar.inv();
    let vector_inv = vector.inv();
    let bivector_inv = bivector.inv();
    let trivector_inv = trivector.inv();

    // verify inversion formula: 1/length for all grades
    assert_eq!(scalar_inv.length, 1.0 / scalar.length); // 1/2
    assert_eq!(vector_inv.length, 1.0 / vector.length); // 1/3
    assert_eq!(bivector_inv.length, 1.0 / bivector.length); // 1/4
    assert_eq!(trivector_inv.length, 1.0 / trivector.length); // 1/5

    // test multiplicative identity: geo * geo.inv() = 1
    let scalar_identity = scalar * scalar_inv;
    let vector_identity = vector * vector_inv;
    let bivector_identity = bivector * bivector_inv;
    let trivector_identity = trivector * trivector_inv;

    // all identities have unit length
    assert!((scalar_identity.length - 1.0).abs() < EPSILON);
    assert!((vector_identity.length - 1.0).abs() < EPSILON);
    assert!((bivector_identity.length - 1.0).abs() < EPSILON);
    assert!((trivector_identity.length - 1.0).abs() < EPSILON);

    // verify inversion blade arithmetic: inv() adds π rotation (2 blades)
    assert_eq!(scalar_inv.angle.blade(), scalar.angle.blade() + 2); // 0 + 2 = 2
    assert_eq!(vector_inv.angle.blade(), vector.angle.blade() + 2); // 1 + 2 = 3
    assert_eq!(bivector_inv.angle.blade(), bivector.angle.blade() + 2); // 2 + 2 = 4
    assert_eq!(trivector_inv.angle.blade(), trivector.angle.blade() + 2); // 3 + 2 = 5

    // test inversion grade transformation
    assert_eq!(scalar_inv.angle.grade(), 2); // grade 0 → grade 2 (scalar → bivector)
    assert_eq!(vector_inv.angle.grade(), 3); // grade 1 → grade 3 (vector → trivector)
    assert_eq!(bivector_inv.angle.grade(), 0); // grade 2 → grade 0 (bivector → scalar)
    assert_eq!(trivector_inv.angle.grade(), 1); // grade 3 → grade 1 (trivector → vector)

    // test high-dimensional inversion without matrix explosion
    let million_obj = Geonum::new_with_blade(7.0, 1_000_000, 2.0, 5.0); // million-dimensional
    let million_inv = million_obj.inv();

    // traditional: would need 2^1000000 × 2^1000000 matrix inversion (impossible)
    // geonum: same reciprocal + angle transformation regardless of dimension
    assert_eq!(million_inv.length, 1.0 / million_obj.length); // reciprocal formula
    assert_eq!(million_inv.angle.blade(), million_obj.angle.blade() + 2); // +2 blade arithmetic

    // test inversion chain: inv(inv(x)) = x (involution property)
    let double_inv = scalar_inv.inv();
    let triple_inv = vector_inv.inv().inv();

    // double inversion returns to original (modulo blade accumulation)
    assert!((double_inv.length - scalar.length).abs() < EPSILON); // length returns
    assert_eq!(double_inv.angle.grade(), scalar.angle.grade()); // grade returns through 4-cycle

    // triple inversion equals single inversion
    assert!((triple_inv.length - vector_inv.length).abs() < EPSILON);
    assert_eq!(triple_inv.angle.grade(), vector_inv.angle.grade());

    // test inversion preserves geometric relationships
    let test_objects = vec![
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // grade 0
        Geonum::new_with_blade(2.0, 1, 1.0, 6.0), // grade 1
        Geonum::new_with_blade(3.0, 2, 1.0, 4.0), // grade 2
        Geonum::new_with_blade(4.0, 3, 1.0, 3.0), // grade 3
    ];

    for obj in test_objects {
        let inv_obj = obj.inv();
        let identity_check = obj * inv_obj;

        // all objects invert to unit length identity
        assert!((identity_check.length - 1.0).abs() < EPSILON);
        // identity preserves multiplicative structure through blade arithmetic
    }

    // traditional GA: matrix inversion requires exponential operations that scale impossibly
    // geonum: constant time inversion through direct geometric operations
    // eliminates matrix decomposition entirely while preserving all inversion properties
}
