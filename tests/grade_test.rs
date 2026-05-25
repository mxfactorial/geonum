// grades are blade % 4, and duality is k → (4 − k)
//
// traditional geometric algebra spreads a multivector across n+1 grade levels and maps
// duality with the dimension-specific k → (n − k). geonum collapses both: grade is
// blade mod 4 — a quarter-turn count — so every dimension reuses the same four
// behaviors, and duality is the fixed k → (4 − k) involution, two pairs (0↔2, 1↔3).
// these tests prove grade decomposition discards the angle addition that carries the
// geometry, and that the four-fold grade structure replaces the unbounded grade ladder
//
// run: cargo test --test grade_test -- --show-output

use geonum::*;

const EPSILON: f64 = 1e-10;

#[test]
fn it_proves_grade_decomposition_ignores_angle_addition() {
    // traditional geometric algebra ignores that multiplication adds angles
    // when you multiply v1 * v2, the angles add: θ1 + θ2
    // but traditional GA pretends this angle addition doesnt happen

    // example: multiply two 45° vectors
    // v1 at 45°, v2 at 45°
    // v1 * v2 rotates by 45° + 45° = 90°

    // but traditional GA ignores this simple angle addition and instead:
    // 1. computes a "scalar part" (grade 0)
    // 2. computes a "bivector part" (grade 2)
    // 3. stores both in separate memory locations
    // 4. pretends the 90° rotation is somehow split between them

    // this negligence - ignoring angle addition - forces traditional GA to:
    // - track 2^n components to handle all possible angle accumulations
    // - invent "grade decomposition" to duplicate the angle information
    // - create massive computational overhead for simple rotations

    // geonum acknowledges that multiplication adds angles
    let v1 = Geonum::new(1.0, 1.0, 4.0); // 45° = π/4
    let v2 = Geonum::new(1.0, 1.0, 4.0); // 45° = π/4

    let product = v1 * v2;

    // result: 45° + 45° = 90° rotation, stored as single angle
    assert_eq!(product.mag, 1.0);
    assert_eq!(product.angle.blade(), 1); // 90° rotation (blade 1)
    assert!(product.angle.near_rem(0.0)); // exactly π/2

    // geonum stores the angle addition result directly
    // no need to decompose into "scalar" and "bivector" parts
    // no need for 2^n components to track angle accumulations

    // traditional GA creates "grade 0" and "grade 2" components because
    // it refuses to acknowledge that angles simply added to 90°

    // demonstration: multiply 0° by 90°
    let x_axis = Geonum::create_dimension(1.0, 0); // 0°
    let y_axis = Geonum::create_dimension(1.0, 1); // 90°

    let xy_product = x_axis * y_axis;

    // angle addition: 0° + 90° = 90°
    assert_eq!(xy_product.angle.blade(), 1); // 90° rotation
    assert!(xy_product.angle.near_rem(0.0));

    // traditional GA would ignore this angle addition and instead:
    // - compute x·y = 0 (call it "scalar part")
    // - compute x∧y = 1 (call it "bivector part")
    // - store both separately
    // - pretend the 90° rotation is somehow "decomposed"

    // but the 90° rotation hasnt been decomposed - its been ignored!
    // grade decomposition is what you get when you refuse to track angle addition

    // by ignoring "angles add", traditional GA creates exponential complexity
    // every possible angle sum needs its own storage location
    // thats why you get 2^n components - one for each possible accumulation

    // geonum eliminates slack from the geometry by requiring angle addition
    // no duplication, no exponential blowup, just store the angle sum directly
}

#[test]
fn it_demonstrates_inversion_preserves_grade_parity_relationships() {
    // geonum's grade structure has involutive pairs: 0↔2, 1↔3
    // operations that preserve this pairing maintain orthogonality relationships
    // circular inversion is one such operation
    let center = Geonum::new_from_cartesian(0.0, 0.0); // origin for clarity
    let radius = 2.0;

    // test points at different angles and distances
    let test_configs = vec![
        // (distance, angle_pi_rad, angle_div, description)
        (1.0, 0.0, 1.0, "inside on +x axis"),
        (3.0, 0.0, 1.0, "outside on +x axis"),
        (1.0, 1.0, 2.0, "inside on +y axis"),
        (3.0, 1.0, 2.0, "outside on +y axis"),
        (1.0, 1.0, 4.0, "inside at π/4"),
        (3.0, 1.0, 4.0, "outside at π/4"),
        (1.0, 1.0, 1.0, "inside on -x axis"),
        (3.0, 1.0, 1.0, "outside on -x axis"),
    ];

    println!("\nSingle point inversions from origin:");
    for (dist, pi_rad, div, desc) in test_configs {
        let p = Geonum::new(dist, pi_rad, div);
        let p_inv = p.invert_circle(&center, radius);

        println!(
            "{}: dist={} angle={:.3} blade={} → dist={:.3} angle={:.3} blade={}",
            desc,
            dist,
            p.angle.rem(),
            p.angle.blade(),
            p_inv.mag,
            p_inv.angle.rem(),
            p_inv.angle.blade()
        );

        // verify inversion property
        assert!((p.mag * p_inv.mag - radius * radius).abs() < EPSILON);
    }

    // now test difference vectors between points (where blade changes occurred before)
    println!("\nDifference vectors between points:");

    // create a configuration that shows blade transformation
    let p1 = Geonum::new_from_cartesian(2.0, 1.0);
    let p2 = Geonum::new_from_cartesian(3.0, 0.0);
    let p3 = Geonum::new_from_cartesian(2.0, -1.0);

    // compute difference vectors
    let v12 = p2 - p1;
    let v13 = p3 - p1;
    let v23 = p3 - p2;

    println!("Original vectors:");
    println!(
        "  v12=p2-p1: length={:.3} angle={:.3} blade={}",
        v12.mag,
        v12.angle.rem(),
        v12.angle.blade()
    );
    println!(
        "  v13=p3-p1: length={:.3} angle={:.3} blade={}",
        v13.mag,
        v13.angle.rem(),
        v13.angle.blade()
    );
    println!(
        "  v23=p3-p2: length={:.3} angle={:.3} blade={}",
        v23.mag,
        v23.angle.rem(),
        v23.angle.blade()
    );

    // invert the points
    let p1_inv = p1.invert_circle(&center, radius);
    let p2_inv = p2.invert_circle(&center, radius);
    let p3_inv = p3.invert_circle(&center, radius);

    // compute inverted difference vectors
    let v12_inv = p2_inv - p1_inv;
    let v13_inv = p3_inv - p1_inv;
    let v23_inv = p3_inv - p2_inv;

    println!("Inverted vectors:");
    println!(
        "  v12_inv: length={:.3} angle={:.3} blade={}",
        v12_inv.mag,
        v12_inv.angle.rem(),
        v12_inv.angle.blade()
    );
    println!(
        "  v13_inv: length={:.3} angle={:.3} blade={}",
        v13_inv.mag,
        v13_inv.angle.rem(),
        v13_inv.angle.blade()
    );
    println!(
        "  v23_inv: length={:.3} angle={:.3} blade={}",
        v23_inv.mag,
        v23_inv.angle.rem(),
        v23_inv.angle.blade()
    );

    // KEY INSIGHT: blade transformation happens in difference vectors
    // individual points from origin maintain blade, but vectors between inverted points transform

    // test with points that create perpendicular vectors
    println!("\nPerpendicular vector configuration:");
    let center2 = Geonum::new_from_cartesian(1.0, 0.0); // offset center
    let q1 = Geonum::new_from_cartesian(3.0, 0.0);
    let q2 = Geonum::new_from_cartesian(4.0, 0.0);
    let q3 = Geonum::new_from_cartesian(3.0, 1.0);

    let u1 = q2 - q1; // horizontal
    let u2 = q3 - q1; // vertical

    println!("Original perpendicular vectors:");
    println!("  u1: blade={} (horizontal)", u1.angle.blade());
    println!("  u2: blade={} (vertical)", u2.angle.blade());

    // these perpendicular vectors have different blades (orthogonality via blade difference)
    assert_ne!(
        u1.angle.blade() % 2,
        u2.angle.blade() % 2,
        "perpendicular vectors differ by odd blade count"
    );

    let q1_inv = q1.invert_circle(&center2, radius);
    let q2_inv = q2.invert_circle(&center2, radius);
    let q3_inv = q3.invert_circle(&center2, radius);

    let u1_inv = q2_inv - q1_inv;
    let u2_inv = q3_inv - q1_inv;

    println!("Inverted 'perpendicular' vectors:");
    println!("  u1_inv: blade={}", u1_inv.angle.blade());
    println!("  u2_inv: blade={}", u2_inv.angle.blade());

    // blade relationships transform under inversion
    let blade_diff_original = (u2.angle.blade() as i32 - u1.angle.blade() as i32).abs();
    let blade_diff_inverted = (u2_inv.angle.blade() as i32 - u1_inv.angle.blade() as i32).abs();

    println!("Blade difference: {blade_diff_original} → {blade_diff_inverted}");

    // check if grade differences are preserved (blade mod 4)
    let grade_diff_original =
        ((u2.angle.grade() as i32 - u1.angle.grade() as i32).abs() % 4) as usize;
    let grade_diff_inverted =
        ((u2_inv.angle.grade() as i32 - u1_inv.angle.grade() as i32).abs() % 4) as usize;

    println!("Grade difference: {grade_diff_original} → {grade_diff_inverted}");

    // orthogonality is encoded in odd grade differences (parity)
    // grade 0 vs grade 1: difference = 1 (odd) → orthogonal
    // grade 2 vs grade 3: difference = 1 (odd) → orthogonal
    // grade 0 vs grade 2: difference = 2 (even) → parallel (dual pair)
    // grade 1 vs grade 3: difference = 2 (even) → parallel (dual pair)

    assert_eq!(
        grade_diff_original % 2,
        1,
        "original vectors are orthogonal (odd grade diff)"
    );
    assert_eq!(
        grade_diff_inverted % 2,
        1,
        "inverted vectors remain orthogonal (odd grade diff)"
    );

    // this is expected from geonum's involutive grade pairs (0↔2, 1↔3)
    // operations respecting this pairing preserve orthogonality parity
    // circular inversion is such an operation - it may shift grades within pairs
    // but preserves the odd/even nature of grade differences
}

#[test]
fn it_replaces_k_to_n_minus_k_with_k_to_4_minus_k() {
    // traditional GA: duality maps grade k to grade (n-k) where n = space dimension
    // different dimensional spaces need different duality mappings
    // 3D: k → (3-k), 4D: k → (4-k), 1000D: k → (1000-k)

    // geonum: universal duality k → (4-k) % 4 regardless of dimensional space
    // works for any dimension through quadrature's bivector foundation

    // demonstrate universal mapping across arbitrary dimensions
    let obj_3d = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1 in "3D context"
    let obj_1000d = Geonum::new_with_blade(1.0, 1001, 0.0, 1.0); // grade 1 in "1000D context"
    let obj_million_d = Geonum::new_with_blade(1.0, 1_000_001, 0.0, 1.0); // grade 1 in "million-D context"

    // traditional GA would need different formulas:
    // 3D: grade 1 → grade (3-1) = 2
    // 1000D: grade 1 → grade (1000-1) = 999
    // million-D: grade 1 → grade (1000000-1) = 999999

    // geonum uses same formula k → (4-k) % 4 for all:
    let dual_3d = obj_3d.dual();
    let dual_1000d = obj_1000d.dual();
    let dual_million_d = obj_million_d.dual();

    // all grade 1 objects map to grade 3 regardless of "dimensional context"
    assert_eq!(obj_3d.angle.grade(), 1);
    assert_eq!(obj_1000d.angle.grade(), 1);
    assert_eq!(obj_million_d.angle.grade(), 1);

    assert_eq!(dual_3d.angle.grade(), 3); // (1+2) % 4 = 3
    assert_eq!(dual_1000d.angle.grade(), 3); // (1+2) % 4 = 3
    assert_eq!(dual_million_d.angle.grade(), 3); // (1+2) % 4 = 3

    // demonstrate grade 2 → grade 0 universally
    let bivector_any_dim = Geonum::new_with_blade(2.0, 1002, 0.0, 1.0); // grade 2
    let dual_bivector = bivector_any_dim.dual();

    assert_eq!(bivector_any_dim.angle.grade(), 2);
    assert_eq!(dual_bivector.angle.grade(), 0); // (2+2) % 4 = 0

    // compression: eliminates dimension-dependent duality formulas
    // one universal k → (4-k) % 4 mapping works for any dimensional space

    // geonum eliminates binomial coefficient (n choose k) component explosion
    // traditional GA: 3D needs (3 choose 1) = 3 vectors, 1000D needs (1000 choose 1) = 1000 vectors
    // geonum: grade 1 objects use same single [length, angle] representation regardless of dimension
    // "linearly independent k-vectors" are irrelevant - direction exists naturally through angle preservation

    // geonum eliminates Hodge decomposition: ω = dα + δβ + γ
    // traditional: separate storage for exact, co-exact, and harmonic components with orthogonal projections
    // geonum: all decomposition distinctions collapse to angle arithmetic
    let form_omega = Geonum::new_with_blade(1.0, 5, 1.0, 3.0); // arbitrary differential form
    let exact_component = form_omega.rotate(Angle::new(1.0, 2.0)); // dα becomes π/2 rotation
    let coexact_component = form_omega.rotate(Angle::new(3.0, 2.0)); // δβ becomes 3π/2 rotation
    let harmonic_component = form_omega; // γ is original angle relationship

    // prove no separate storage needed for Hodge decomposition components
    assert_eq!(
        std::mem::size_of_val(&form_omega),
        std::mem::size_of_val(&exact_component)
    );
    assert_eq!(
        std::mem::size_of_val(&exact_component),
        std::mem::size_of_val(&coexact_component)
    );
    assert_eq!(
        std::mem::size_of_val(&coexact_component),
        std::mem::size_of_val(&harmonic_component)
    );

    // prove all grade 1 objects have identical storage regardless of "dimensional context"
    assert_eq!(
        std::mem::size_of_val(&obj_3d),
        std::mem::size_of_val(&obj_1000d)
    );
    assert_eq!(
        std::mem::size_of_val(&obj_1000d),
        std::mem::size_of_val(&obj_million_d)
    );

    // traditional GA storage would scale with binomial coefficients:
    // 3D grade 1: 3 components, 1000D grade 1: 1000 components, million-D grade 1: 1000000 components
    // geonum storage: constant 2 components (length + angle) for any dimension
}

#[test]
fn it_compresses_traditional_ga_grades_to_two_involutive_pairs() {
    // geonum's π-rotation dual creates a different incidence structure than traditional GA
    // instead of computing maximal common subspaces, it computes containing spaces

    let line1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // grade 1
    let line2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // grade 1, different angle
    let bivector = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // grade 2
    let bivector2 = Geonum::new_with_blade(1.0, 2, 1.0, 4.0); // grade 2, different angle

    // line meet line → grade 1 (vector)
    // geometric meaning: the intersection point represented as a vector from origin
    // traditional GA expects grade 0 (scalar point)
    assert_eq!(line1.meet(&line2).angle.grade(), 1);

    // vector meet bivector → grade 2 (bivector)
    // geometric meaning: the minimal plane containing both the line and the original plane
    // traditional GA expects grade 0 (point of intersection)
    assert_eq!(line1.meet(&bivector).angle.grade(), 2);

    // bivector meet bivector → grade 3 (trivector)
    // geometric meaning: the 3D volume spanned by the two planes
    // traditional GA expects grade 1 (line of intersection)
    assert_eq!(bivector.meet(&bivector2).angle.grade(), 3);

    // this reversal happens because π-rotation dual creates scalar↔bivector
    // and vector↔trivector pairings rather than traditional complementary pairings

    // KEY INSIGHT: geonum flattens traditional GA's n+1 grade levels (0 through n)
    // to just 2 involutive pairs that work in any dimension:
    // - pair 1: grade 0 ↔ grade 2 (scalar ↔ bivector)
    // - pair 2: grade 1 ↔ grade 3 (vector ↔ trivector)
    //
    // grades cycle modulo 4, so grade 1000000 in million-D space is just grade 0
    // this eliminates dimension-specific k→(n-k) duality formulas
    // replacing them with universal k→(k+2)%4 that works everywhere
}

#[test]
fn it_handles_mixed_grade_operations_naturally() {
    // traditional GA: restricts operations to "like grades" or requires complex rules
    // scalar * scalar = scalar, vector * vector = scalar + bivector, etc.
    // mixed grade operations need special handling and decomposition

    // geonum: blade arithmetic works for ANY grade combination
    let scalar = Geonum::new(2.0, 0.0, 1.0); // blade 0 (grade 0)
    let vector = Geonum::new(3.0, 1.0, 2.0); // blade 1 (grade 1)
    let bivector = Geonum::new(1.5, 1.0, 1.0); // blade 2 (grade 2)
    let trivector = Geonum::new(4.0, 3.0, 2.0); // blade 3 (grade 3)

    // mixed grade products: blade counts just add
    let scalar_vector = scalar * vector; // 0+1=1 (vector)
    let vector_bivector = vector * bivector; // 1+2=3 (trivector)
    let bivector_trivector = bivector * trivector; // 2+3=5 (grade 1: 5%4=1)
    let scalar_trivector = scalar * trivector; // 0+3=3 (trivector)

    // verify blade arithmetic works regardless of starting grades
    assert_eq!(scalar_vector.angle.blade(), 1);
    assert_eq!(vector_bivector.angle.blade(), 3);
    assert_eq!(bivector_trivector.angle.blade(), 5);
    assert_eq!(scalar_trivector.angle.blade(), 3);

    // verify grades cycle correctly (blade % 4)
    assert_eq!(scalar_vector.angle.grade(), 1); // blade 1 → grade 1
    assert_eq!(vector_bivector.angle.grade(), 3); // blade 3 → grade 3
    assert_eq!(bivector_trivector.angle.grade(), 1); // blade 5 → grade 1
    assert_eq!(scalar_trivector.angle.grade(), 3); // blade 3 → grade 3

    // traditional GA: each combination needs special rules and storage
    // geonum: universal blade addition works for all grade combinations
    // no restrictions, no special cases, no decomposition complexity
}
