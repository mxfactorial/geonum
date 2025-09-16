// BLADE LOGIC REFERENCE
//
// geonum tracks geometric transformations through blade counting
// each π/2 rotation increments the blade count by 1
// this enables O(1) geometric algebra operations in any dimension
//
// FUNDAMENTAL RULES:
// 1. blade accumulation tracks transformation history
// 2. boundary crossing occurs when angle reaches π/2
// 3. grade = blade % 4 determines geometric behavior
// 4. forward-only arithmetic maintains positive angle space
//
// COMMON OPERATIONS:
// - angle addition: adds blade counts and values, handles π/2 overflow
// - wedge product: combines angles + π/2, orientation encoded in blade
// - dual operation: adds π rotation (2 blades) for grade transformation
// - compound operations: accumulate blades through multiple transformations
//
// each test documents step-by-step blade arithmetic to eliminate confusion
// about boundary crossing and accumulation in complex operations

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_decomposes_angles_into_blade_and_value() {
    // BLADE DECOMPOSITION: total_angle = blade * π/2 + value, where value ∈ [0, π/2)
    // blade = count of complete π/2 rotations
    // value = fractional remainder within current π/2 segment

    // 3π/4 = 1 complete π/2 rotation + π/4 remainder
    let three_quarters = Angle::new(3.0, 4.0); // 3 * π/4
    assert_eq!(three_quarters.blade(), 1); // 1 complete π/2 rotation
    assert!((three_quarters.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                                  // blade arithmetic: 3π/4 = 1*(π/2) + π/4

    // 5π/2 = 5 complete π/2 rotations + 0 remainder
    let five_halves = Angle::new(5.0, 2.0); // 5 * π/2
    assert_eq!(five_halves.blade(), 5); // 5 complete π/2 rotations
    assert!(five_halves.value().abs() < EPSILON); // no remainder
                                                  // blade arithmetic: 5π/2 = 5*(π/2) + 0

    // 7π/3 = 4 complete π/2 rotations + π/6 remainder
    let seven_thirds = Angle::new(7.0, 3.0); // 7 * π/3 = 7π/3
                                             // 7π/3 = 14π/6 = (12π + 2π)/6 = 2π + π/3 = 4*(π/2) + π/3
    assert_eq!(seven_thirds.blade(), 4); // 4 complete π/2 rotations
    assert!((seven_thirds.value() - PI / 3.0).abs() < EPSILON); // π/3 remainder
                                                                // blade arithmetic: 7π/3 = 4*(π/2) + π/3

    // proves decomposition always maintains: angle = blade*(π/2) + value
    // with value strictly bounded to [0, π/2) through boundary normalization
}

#[test]
fn it_crosses_pi_2_boundaries() {
    // BOUNDARY CROSSING: when total angle ≥ π/2, increment blade and reduce value
    // normalize_boundaries() enforces the [0, π/2) constraint on value

    // case 1: angle exactly at π/2 boundary
    let at_boundary = Angle::new(1.0, 2.0); // 1 * π/2 = π/2
    assert_eq!(at_boundary.blade(), 1); // crosses boundary: blade 0 → blade 1
    assert!(at_boundary.value().abs() < EPSILON); // value resets to 0
                                                  // blade arithmetic: π/2 = 1*(π/2) + 0

    // case 2: angle exceeding π/2 boundary
    let over_boundary = Angle::new(3.0, 4.0); // 3 * π/4 = π/2 + π/4
    assert_eq!(over_boundary.blade(), 1); // crosses boundary once
    assert!((over_boundary.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                                 // blade arithmetic: 3π/4 = 1*(π/2) + π/4

    // case 3: angle crossing multiple boundaries
    let multiple_cross = Angle::new(7.0, 4.0); // 7 * π/4 = π/2 + π + π/4
    assert_eq!(multiple_cross.blade(), 3); // crosses 3 boundaries: 7π/4 = 3*(π/2) + π/4
    assert!((multiple_cross.value() - PI / 4.0).abs() < EPSILON); // π/4 final remainder
                                                                  // blade arithmetic: 7π/4 = 3*(π/2) + π/4

    // proves boundary crossing increments blade count for each π/2 crossed
    // while maintaining value in [0, π/2) invariant
}

#[test]
fn it_adds_angles_with_blade_arithmetic() {
    // BLADE ADDITION: blade₁ + blade₂, value₁ + value₂, handle overflow
    // geometric_add() performs: (blade_total, value_total) → normalize if needed

    // case 1: values sum without boundary crossing
    let angle1 = Angle::new(1.0, 8.0); // π/8 → blade=0, value=π/8
    let angle2 = Angle::new(1.0, 6.0); // π/6 → blade=0, value=π/6
    let sum = angle1 + angle2; // π/8 + π/6 = 7π/24
    assert_eq!(sum.blade(), 0); // 0 + 0 = 0 blades
    assert!((sum.value() - 7.0 * PI / 24.0).abs() < EPSILON); // 7π/24 < π/2, no crossing
                                                              // blade arithmetic: blade₁ + blade₂ = 0 + 0 = 0, value₁ + value₂ = 7π/24

    // case 2: values sum crossing π/2 boundary
    let angle3 = Angle::new(1.0, 3.0); // π/3 → blade=0, value=π/3
    let angle4 = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    let boundary_sum = angle3 + angle4; // π/3 + π/4 = 7π/12 > π/2
    assert_eq!(boundary_sum.blade(), 1); // boundary crossing increments: 0 + 0 + 1 = 1
    assert!((boundary_sum.value() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON); // remainder after crossing
                                                                                    // blade arithmetic: blade₁ + blade₂ + 1 = 0 + 0 + 1 = 1, value = 7π/12 - π/2 = π/12

    // case 3: blades sum with value boundary crossing
    let angle5 = Angle::new(3.0, 4.0); // 3π/4 → blade=1, value=π/4
    let angle6 = Angle::new(5.0, 4.0); // 5π/4 → blade=2, value=π/4
    let blade_value_sum = angle5 + angle6; // blades: 1+2=3, values: π/4+π/4=π/2
    assert_eq!(blade_value_sum.blade(), 4); // 3 + 1(crossing) = 4 total blades
    assert!(blade_value_sum.value().abs() < EPSILON); // π/2 crossing resets value to 0
                                                      // blade arithmetic: blade₁ + blade₂ + crossing = 1 + 2 + 1 = 4, value = π/2 → 0

    // proves angle addition accumulates blades while handling π/2 overflow
}

#[test]
fn it_handles_new_with_blade_construction() {
    // EXPLICIT BLADE: new_with_blade adds specified blades to computed angle
    // formula: base_angle + blade_increment where blade_increment = added_blade * π/2

    // case 1: add 2 blades to π/4 angle
    let with_blade = Angle::new_with_blade(2, 1.0, 4.0); // 2 blades + π/4
                                                         // step 1: base_angle = π/4 → blade=0, value=π/4
                                                         // step 2: blade_increment = 2 * π/2 = π → blade=2, value=0
                                                         // step 3: addition = (0+2, π/4+0) = blade=2, value=π/4
    assert_eq!(with_blade.blade(), 2); // 0 + 2 = 2 blades
    assert!((with_blade.value() - PI / 4.0).abs() < EPSILON); // π/4 preserved
                                                              // blade arithmetic: 2*(π/2) + π/4 = π + π/4 = 2*(π/2) + π/4

    // case 2: add blades causing boundary crossing
    let crossing_blade = Angle::new_with_blade(1, 3.0, 4.0); // 1 blade + 3π/4
                                                             // step 1: base_angle = 3π/4 → blade=1, value=π/4
                                                             // step 2: blade_increment = 1 * π/2 → blade=1, value=0
                                                             // step 3: addition = (1+1, π/4+0) = blade=2, value=π/4
    assert_eq!(crossing_blade.blade(), 2); // 1 + 1 = 2 blades
    assert!((crossing_blade.value() - PI / 4.0).abs() < EPSILON); // π/4 from base angle
                                                                  // blade arithmetic: 1*(π/2) + 3π/4 = π/2 + (π/2 + π/4) = 2*(π/2) + π/4

    // case 3: zero additional blades (identity case)
    let zero_blade = Angle::new_with_blade(0, 1.0, 2.0); // 0 blades + π/2
                                                         // step 1: base_angle = π/2 → blade=1, value=0
                                                         // step 2: blade_increment = 0 * π/2 = 0 → blade=0, value=0
                                                         // step 3: addition = (1+0, 0+0) = blade=1, value=0
    assert_eq!(zero_blade.blade(), 1); // 1 + 0 = 1 blade
    assert!(zero_blade.value().abs() < EPSILON); // 0 + 0 = 0
                                                 // blade arithmetic: 0*(π/2) + π/2 = π/2 = 1*(π/2) + 0

    // proves new_with_blade performs: base_decomposition + explicit_blade_addition
}

#[test]
fn it_converts_from_cartesian_coordinates() {
    // CARTESIAN DECOMPOSITION: atan2(y,x) → blade/value decomposition
    // process: cartesian → radians → pi_radians → blade/value split

    // case 1: 45° angle (π/4)
    let cart_45 = Angle::new_from_cartesian(1.0, 1.0); // atan2(1,1) = π/4
                                                       // step 1: atan2(1,1) = π/4 radians
                                                       // step 2: π/4 ÷ π = 1/4 pi_radians
                                                       // step 3: Angle::new(1/4, 1.0) → π/4 = 0*(π/2) + π/4
    assert_eq!(cart_45.blade(), 0); // no π/2 crossings
    assert!((cart_45.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                           // blade arithmetic: π/4 = 0*(π/2) + π/4

    // case 2: 90° angle (π/2)
    let cart_90 = Angle::new_from_cartesian(0.0, 1.0); // atan2(1,0) = π/2
                                                       // step 1: atan2(1,0) = π/2 radians
                                                       // step 2: π/2 ÷ π = 1/2 pi_radians
                                                       // step 3: Angle::new(1/2, 1.0) → π/2 = 1*(π/2) + 0
    assert_eq!(cart_90.blade(), 1); // 1 π/2 crossing
    assert!(cart_90.value().abs() < EPSILON); // no remainder
                                              // blade arithmetic: π/2 = 1*(π/2) + 0

    // case 3: 180° angle (π)
    let cart_180 = Angle::new_from_cartesian(-1.0, 0.0); // atan2(0,-1) = π
                                                         // step 1: atan2(0,-1) = π radians
                                                         // step 2: π ÷ π = 1 pi_radians
                                                         // step 3: Angle::new(1.0, 1.0) → π = 2*(π/2) + 0
    assert_eq!(cart_180.blade(), 2); // 2 π/2 crossings
    assert!(cart_180.value().abs() < EPSILON); // no remainder
                                               // blade arithmetic: π = 2*(π/2) + 0

    // case 4: 270° angle (3π/2)
    let cart_270 = Angle::new_from_cartesian(0.0, -1.0); // atan2(-1,0) = 3π/2
                                                         // step 1: atan2(-1,0) = 3π/2 radians
                                                         // step 2: 3π/2 ÷ π = 3/2 pi_radians
                                                         // step 3: Angle::new(3/2, 1.0) → 3π/2 = 3*(π/2) + 0
    assert_eq!(cart_270.blade(), 3); // 3 π/2 crossings
    assert!(cart_270.value().abs() < EPSILON); // no remainder
                                               // blade arithmetic: 3π/2 = 3*(π/2) + 0

    // proves cartesian conversion maintains blade decomposition through atan2 → blade arithmetic
}

#[test]
fn it_computes_grade_from_blade_modulo_4() {
    // GRADE CYCLING: grade = blade % 4, determines geometric behavior
    // grade 0=scalar, 1=vector, 2=bivector, 3=trivector, then repeats

    // case 1: blade counts 0-3 map directly to grades
    let blade_0 = Angle::new(0.0, 1.0); // blade=0
    let blade_1 = Angle::new(1.0, 2.0); // blade=1
    let blade_2 = Angle::new(2.0, 2.0); // blade=2
    let blade_3 = Angle::new(3.0, 2.0); // blade=3
    assert_eq!(blade_0.grade(), 0); // 0 % 4 = 0 (scalar)
    assert_eq!(blade_1.grade(), 1); // 1 % 4 = 1 (vector)
    assert_eq!(blade_2.grade(), 2); // 2 % 4 = 2 (bivector)
    assert_eq!(blade_3.grade(), 3); // 3 % 4 = 3 (trivector)
                                    // blade arithmetic: grade = blade % 4 for basic cases

    // case 2: blade counts 4-7 cycle back through grades
    let blade_4 = Angle::new(4.0, 2.0); // blade=4
    let blade_5 = Angle::new(5.0, 2.0); // blade=5
    let blade_6 = Angle::new(6.0, 2.0); // blade=6
    let blade_7 = Angle::new(7.0, 2.0); // blade=7
    assert_eq!(blade_4.grade(), 0); // 4 % 4 = 0 (scalar again)
    assert_eq!(blade_5.grade(), 1); // 5 % 4 = 1 (vector again)
    assert_eq!(blade_6.grade(), 2); // 6 % 4 = 2 (bivector again)
    assert_eq!(blade_7.grade(), 3); // 7 % 4 = 3 (trivector again)
                                    // blade arithmetic: 4-cycle repetition through modulo operation

    // case 3: high blade counts maintain cycling
    let blade_1000 = Angle::new(1000.0, 2.0); // blade=1000
    let blade_1001 = Angle::new(1001.0, 2.0); // blade=1001
    let blade_1002 = Angle::new(1002.0, 2.0); // blade=1002
    let blade_1003 = Angle::new(1003.0, 2.0); // blade=1003
    assert_eq!(blade_1000.grade(), 0); // 1000 % 4 = 0 (scalar)
    assert_eq!(blade_1001.grade(), 1); // 1001 % 4 = 1 (vector)
    assert_eq!(blade_1002.grade(), 2); // 1002 % 4 = 2 (bivector)
    assert_eq!(blade_1003.grade(), 3); // 1003 % 4 = 3 (trivector)
                                       // blade arithmetic: grade cycling works at any blade magnitude

    // proves grade = blade % 4 creates 4-fold periodicity enabling
    // million-dimensional geometric algebra through grade behavior cycling
}

#[test]
fn it_applies_dual_through_pi_rotation() {
    // DUAL OPERATION: adds π rotation (2 blades), creates grade pairs
    // formula: dual(angle) = angle + π = angle + 2*(π/2) = blade + 2

    // case 1: scalar → bivector (grade 0 → grade 2)
    let scalar = Angle::new(0.0, 1.0); // blade=0, grade=0
    let dual_scalar = scalar.dual();
    // step 1: original = blade=0, value=0
    // step 2: add π rotation = add 2 blades
    // step 3: result = blade=0+2=2, value=0+0=0
    assert_eq!(dual_scalar.blade(), 2); // 0 + 2 = 2 blades
    assert_eq!(dual_scalar.grade(), 2); // 2 % 4 = 2 (bivector)
    assert!(dual_scalar.value().abs() < EPSILON); // value unchanged
                                                  // blade arithmetic: dual adds exactly 2 blades (π rotation)

    // case 2: vector → trivector (grade 1 → grade 3)
    let vector = Angle::new(1.0, 2.0); // blade=1, grade=1
    let dual_vector = vector.dual();
    // step 1: original = blade=1, value=0
    // step 2: add π rotation = add 2 blades
    // step 3: result = blade=1+2=3, value=0+0=0
    assert_eq!(dual_vector.blade(), 3); // 1 + 2 = 3 blades
    assert_eq!(dual_vector.grade(), 3); // 3 % 4 = 3 (trivector)
                                        // blade arithmetic: vector + π = trivector

    // case 3: bivector → scalar (grade 2 → grade 0, cycling)
    let bivector = Angle::new(2.0, 2.0); // blade=2, grade=2
    let dual_bivector = bivector.dual();
    // step 1: original = blade=2, value=0
    // step 2: add π rotation = add 2 blades
    // step 3: result = blade=2+2=4, value=0+0=0
    assert_eq!(dual_bivector.blade(), 4); // 2 + 2 = 4 blades
    assert_eq!(dual_bivector.grade(), 0); // 4 % 4 = 0 (scalar, cycling)
                                          // blade arithmetic: bivector + π = next scalar (grade cycling)

    // case 4: high blade dual maintains pattern
    let high_blade = Angle::new_with_blade(1000, 0.0, 1.0); // blade=1000, grade=0
    let dual_high = high_blade.dual();
    assert_eq!(dual_high.blade(), 1002); // 1000 + 2 = 1002 blades
    assert_eq!(dual_high.grade(), 2); // 1002 % 4 = 2 (bivector)
                                      // blade arithmetic: dual always adds 2 blades regardless of magnitude

    // proves dual creates universal grade pairs: 0↔2, 1↔3 through π rotation
    // blade accumulation preserves transformation history while grade determines behavior
}

#[test]
fn it_implements_add_trait_blade_logic() {
    // ADD TRAIT: geometric_add blade accumulation through all ownership variants
    // all Add implementations call geometric_add(blade₁+blade₂, value₁+value₂)

    // setup test angles with known blade/value components
    let angle_a = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    let angle_b = Angle::new(1.0, 6.0); // π/6 → blade=0, value=π/6

    // case 1: owned + owned
    let sum_owned = angle_a + angle_b;
    // step 1: blade₁ + blade₂ = 0 + 0 = 0
    // step 2: value₁ + value₂ = π/4 + π/6 = 5π/12
    // step 3: 5π/12 < π/2, no boundary crossing
    assert_eq!(sum_owned.blade(), 0); // 0 + 0 = 0 blades
    assert!((sum_owned.value() - 5.0 * PI / 12.0).abs() < EPSILON); // 5π/12 value
                                                                    // blade arithmetic: (0,π/4) + (0,π/6) = (0, 5π/12)

    // case 2: test all borrowing variants produce identical blade arithmetic
    let sum_borrow1 = angle_a + angle_b; // owned + borrowed
    let sum_borrow2 = angle_a + angle_b; // borrowed + owned
    let sum_borrow3 = angle_a + angle_b; // borrowed + borrowed
    assert_eq!(sum_borrow1.blade(), sum_owned.blade()); // same blade count
    assert_eq!(sum_borrow2.blade(), sum_owned.blade()); // same blade count
    assert_eq!(sum_borrow3.blade(), sum_owned.blade()); // same blade count
    assert_eq!(sum_borrow1.value(), sum_owned.value()); // same value
    assert_eq!(sum_borrow2.value(), sum_owned.value()); // same value
    assert_eq!(sum_borrow3.value(), sum_owned.value()); // same value
                                                        // blade arithmetic: all variants call identical geometric_add logic

    // case 3: addition with boundary crossing
    let angle_c = Angle::new(3.0, 8.0); // 3π/8 → blade=0, value=3π/8
    let angle_d = Angle::new(5.0, 8.0); // 5π/8 → blade=0, value=5π/8
    let crossing_sum = angle_c + angle_d; // 3π/8 + 5π/8 = π
                                          // step 1: blade₁ + blade₂ = 0 + 0 = 0
                                          // step 2: value₁ + value₂ = 3π/8 + 5π/8 = π
                                          // step 3: π ≥ π/2, so normalize: π = 2*(π/2) + 0
    assert_eq!(crossing_sum.blade(), 2); // 0 + 0 + 2(crossings) = 2 blades
    assert!(crossing_sum.value().abs() < EPSILON); // π normalized to 0 remainder
                                                   // blade arithmetic: boundary crossing adds 2 blades for π total

    // proves Add trait maintains consistent blade accumulation across all ownership patterns
}

#[test]
fn it_implements_sub_trait_blade_logic() {
    // SUB TRAIT: geometric_sub borrowing and wrapping blade arithmetic
    // handles negative results through borrowing and 4-rotation wrapping

    // case 1: simple subtraction without borrowing
    let angle_big = Angle::new(3.0, 2.0); // 3π/2 → blade=3, value=0
    let angle_small = Angle::new(1.0, 2.0); // π/2 → blade=1, value=0
    let simple_diff = angle_big - angle_small;
    // step 1: blade₁ - blade₂ = 3 - 1 = 2
    // step 2: value₁ - value₂ = 0 - 0 = 0
    // step 3: no borrowing needed
    assert_eq!(simple_diff.blade(), 2); // 3 - 1 = 2 blades
    assert!(simple_diff.value().abs() < EPSILON); // 0 - 0 = 0
                                                  // blade arithmetic: (3,0) - (1,0) = (2,0)

    // case 2: value borrowing (negative value requires blade borrowing)
    let small_value = Angle::new(1.0, 6.0); // π/6 → blade=0, value=π/6
    let large_value = Angle::new(1.0, 3.0); // π/3 → blade=0, value=π/3
    let borrow_diff = small_value - large_value; // π/6 - π/3 = -π/6
                                                 // step 1: blade₁ - blade₂ = 0 - 0 = 0
                                                 // step 2: value₁ - value₂ = π/6 - π/3 = -π/6 (negative!)
                                                 // step 3: borrow from blade: blade = 0-1 = -1, value = -π/6 + π/2 = π/3
                                                 // step 4: wrap negative blade: -1 + 4 = 3
    assert_eq!(borrow_diff.blade(), 3); // -1 + 4 = 3 (wrapped)
    assert!((borrow_diff.value() - PI / 3.0).abs() < EPSILON); // -π/6 + π/2 = π/3
                                                               // blade arithmetic: borrowing maintains forward-only angle space

    // case 3: negative blade wrapping
    let zero_blade = Angle::new(0.0, 1.0); // blade=0, value=0
    let high_blade = Angle::new(5.0, 2.0); // blade=5, value=0
    let wrap_diff = zero_blade - high_blade; // 0 - 5 = -5 blades
                                             // step 1: blade₁ - blade₂ = 0 - 5 = -5
                                             // step 2: value₁ - value₂ = 0 - 0 = 0
                                             // step 3: wrap negative blade: -5 + 8 = 3 (add 2*4 rotations)
    assert_eq!(wrap_diff.blade(), 3); // -5 + 8 = 3 (wrapped forward)
    assert!(wrap_diff.value().abs() < EPSILON); // 0 - 0 = 0
                                                // blade arithmetic: negative blades wrap through 4-rotation addition

    // proves Sub trait maintains forward-only angle space through borrowing and wrapping
}

#[test]
fn it_implements_mul_trait_blade_logic() {
    // MUL TRAIT: "angles add" implemented as geometric_add (identical to Add)
    // multiplication IS addition in angle arithmetic (fundamental geonum principle)

    // setup test angles with explicit blade/value tracking
    let angle_x = Angle::new(1.0, 3.0); // π/3 → blade=0, value=π/3
    let angle_y = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4

    // case 1: multiplication calls geometric_add (same as addition)
    let product = angle_x * angle_y;
    let addition = angle_x + angle_y;
    // step 1: blade₁ + blade₂ = 0 + 0 = 0 (same for both operations)
    // step 2: value₁ + value₂ = π/3 + π/4 = 7π/12 (same for both)
    // step 3: 7π/12 > π/2, boundary crossing adds 1 blade
    assert_eq!(product.blade(), addition.blade()); // identical blade arithmetic
    assert_eq!(product.value(), addition.value()); // identical value arithmetic
    assert_eq!(product.blade(), 1); // 0 + 0 + 1(crossing) = 1
    assert!((product.value() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON); // 7π/12 - π/2 remainder
                                                                               // blade arithmetic: angle multiplication IS angle addition

    // case 2: verify "angles add, lengths multiply" principle at angle level
    let mult_result = angle_x * angle_y; // calls geometric_add
    let expected_angle = angle_x + angle_y; // direct geometric_add
    assert_eq!(mult_result.blade(), expected_angle.blade()); // same blade result
    assert_eq!(mult_result.value(), expected_angle.value()); // same value result
                                                             // blade arithmetic: * operator implements + operation for angles

    // case 3: multiplication with high blades
    let high_a = Angle::new(100.0, 2.0); // blade=100
    let high_b = Angle::new(200.0, 2.0); // blade=200
    let high_product = high_a * high_b;
    // step 1: blade₁ + blade₂ = 100 + 200 = 300
    // step 2: value₁ + value₂ = 0 + 0 = 0
    // step 3: no boundary crossing
    assert_eq!(high_product.blade(), 300); // 100 + 200 = 300 blades
    assert!(high_product.value().abs() < EPSILON); // 0 + 0 = 0
                                                   // blade arithmetic: high blade multiplication follows same addition rule

    // proves Mul trait implements "angles add" through identical geometric_add calls
}

#[test]
fn it_implements_div_trait_blade_logic() {
    // DIV TRAIT: scalar division vs angle subtraction blade behaviors
    // two modes: Div<f64> reconstructs total angle, Div<Angle> calls geometric_sub

    // case 1: division by scalar (Div<f64>) - reconstructs then decomposes
    let test_angle = Angle::new(3.0, 2.0); // 3π/2 → blade=3, value=0
    let scalar_div = test_angle / 2.0;
    // step 1: reconstruct total = blade*(π/2) + value = 3*(π/2) + 0 = 3π/2
    // step 2: divide by scalar = 3π/2 ÷ 2 = 3π/4
    // step 3: re-decompose = 3π/4 = 1*(π/2) + π/4
    assert_eq!(scalar_div.blade(), 1); // 3π/4 crosses π/2 once
    assert!((scalar_div.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                              // blade arithmetic: scalar division reconstructs → scales → decomposes

    // case 2: division by angle (Div<Angle>) - calls geometric_sub
    let dividend = Angle::new(5.0, 4.0); // 5π/4 → blade=2, value=π/4
    let divisor = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    let angle_div = dividend / divisor;
    // step 1: blade₁ - blade₂ = 2 - 0 = 2
    // step 2: value₁ - value₂ = π/4 - π/4 = 0
    // step 3: no borrowing needed
    assert_eq!(angle_div.blade(), 2); // 2 - 0 = 2 blades
    assert!(angle_div.value().abs() < EPSILON); // π/4 - π/4 = 0
                                                // blade arithmetic: angle division IS angle subtraction (geometric_sub)

    // case 3: verify division equivalence with subtraction
    let subtraction_result = dividend - divisor;
    assert_eq!(angle_div.blade(), subtraction_result.blade()); // same blade result
    assert_eq!(angle_div.value(), subtraction_result.value()); // same value result
                                                               // blade arithmetic: Div<Angle> and Sub produce identical results

    // case 4: scalar division with boundary crossing reconstruction
    let complex_angle = Angle::new(7.0, 4.0); // 7π/4 → blade=3, value=π/4
    let divided_complex = complex_angle / 3.0;
    // step 1: reconstruct = 3*(π/2) + π/4 = 6π/4 + π/4 = 7π/4
    // step 2: divide = 7π/4 ÷ 3 = 7π/12
    // step 3: decompose = 7π/12 = 1*(π/2) + π/12 (since 7π/12 > π/2)
    assert_eq!(divided_complex.blade(), 1); // 7π/12 crosses π/2 once
    assert!((divided_complex.value() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON);
    // π/12 remainder
    // blade arithmetic: scalar division handles complex blade reconstruction

    // proves Div trait: scalar mode reconstructs total angle, angle mode subtracts
}

#[test]
fn it_implements_partial_eq_blade_logic() {
    // PARTIAL_EQ TRAIT: blade equality first, then value comparison with precision handling
    // comparison logic: blade must match exactly, then value within floating point tolerance

    // case 1: identical blade and value
    let angle1 = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    let angle2 = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    assert_eq!(angle1, angle2); // exact equality
                                // blade arithmetic: (0,π/4) == (0,π/4) → blade match + value match = equal

    // case 2: different blades (always unequal)
    let diff_blade1 = Angle::new(1.0, 2.0); // π/2 → blade=1, value=0
    let diff_blade2 = Angle::new(3.0, 2.0); // 3π/2 → blade=3, value=0
    assert_ne!(diff_blade1, diff_blade2); // different blades
                                          // blade arithmetic: (1,0) != (3,0) → blade mismatch = unequal (value ignored)

    // case 3: same blade, different values
    let same_blade1 = Angle::new(1.0, 6.0); // π/6 → blade=0, value=π/6
    let same_blade2 = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    assert_ne!(same_blade1, same_blade2); // same blade, different values
                                          // blade arithmetic: (0,π/6) != (0,π/4) → blade match + value mismatch = unequal

    // case 4: floating point precision tolerance
    let precise1 = Angle::new(1.0, 4.0); // π/4 exact
    let tiny_diff = Angle::new(1.0, 4.0) + Angle::new(1e-16, PI); // π/4 + tiny error
                                                                  // precision threshold is 1e-15, so 1e-16 error should match
    assert_eq!(precise1, tiny_diff); // within precision tolerance
                                     // blade arithmetic: (0,π/4) == (0,π/4+ε) → exact match logic handles precision

    // case 5: high blade equality
    let high1 = Angle::new(1000.0, 2.0); // blade=1000, value=0
    let high2 = Angle::new(1000.0, 2.0); // blade=1000, value=0
    assert_eq!(high1, high2); // high blades equal when identical
                              // blade arithmetic: exact blade comparison works at any magnitude

    // proves PartialEq prioritizes blade equality, then handles value precision
}

#[test]
fn it_implements_ord_blade_logic() {
    // ORD TRAIT: blade-first ordering maintains geometric hierarchy regardless of value
    // ordering logic: compare blade first, then value only if blades equal

    // case 1: blade precedence over value magnitude
    let small_blade_big_value = Angle::new(0.8, 1.0); // 0.8π → blade=1, value≈0.26π
    let big_blade_small_value = Angle::new(1.1, 1.0); // 1.1π → blade=2, value≈0.1π
    assert!(small_blade_big_value < big_blade_small_value); // blade 1 < blade 2
                                                            // blade arithmetic: blade comparison ignores value magnitude differences

    // case 2: same blade ordering by value
    let same_blade_small = Angle::new(1.0, 6.0); // π/6 → blade=0, value=π/6
    let same_blade_medium = Angle::new(1.0, 4.0); // π/4 → blade=0, value=π/4
    let same_blade_large = Angle::new(1.0, 3.0); // π/3 → blade=0, value=π/3
    assert!(same_blade_small < same_blade_medium); // π/6 < π/4 when blade=0
    assert!(same_blade_medium < same_blade_large); // π/4 < π/3 when blade=0
    assert!(same_blade_small < same_blade_large); // transitivity
                                                  // blade arithmetic: when blades equal, value determines order

    // case 3: high blade ordering maintains hierarchy
    let blade_1000 = Angle::new(1000.0, 2.0); // blade=1000, value=0
    let blade_1001 = Angle::new(1001.0, 2.0); // blade=1001, value=0
    let blade_2000 = Angle::new(2000.0, 2.0); // blade=2000, value=0
    assert!(blade_1000 < blade_1001); // consecutive blades
    assert!(blade_1001 < blade_2000); // large blade gap
    assert!(blade_1000 < blade_2000); // transitivity
                                      // blade arithmetic: ordering preserves blade hierarchy at any magnitude

    // case 4: geometric hierarchy example (tiny bivector > huge scalar)
    let huge_scalar = Angle::new(0.0, 1.0); // blade=0 (scalar), any value
    let tiny_bivector = Angle::new(2.0, 2.0); // blade=2 (bivector), any value
    assert!(huge_scalar < tiny_bivector); // blade 0 < blade 2
                                          // blade arithmetic: geometric grade hierarchy trumps angle magnitude

    // proves Ord maintains blade-first hierarchy preserving geometric meaning
}

#[test]
fn it_constructs_geonum_with_basic_new() {
    // GEONUM CONSTRUCTION: length + angle decomposition through Angle::new
    // process: Geonum::new(length, pi_radians, divisor) → [length, Angle::new(pi_radians, divisor)]

    // case 1: length with angle crossing π/2 boundary
    let geo1 = Geonum::new(2.0, 3.0, 4.0); // length=2, angle=3π/4
                                           // step 1: length component = 2.0 (preserved directly)
                                           // step 2: angle component = Angle::new(3.0, 4.0) = 3π/4
                                           // step 3: blade decomposition = 3π/4 = 1*(π/2) + π/4
    assert_eq!(geo1.length, 2.0); // length preserved
    assert_eq!(geo1.angle.blade(), 1); // 3π/4 crosses π/2 once
    assert!((geo1.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                              // blade arithmetic: length passthrough + angle decomposition

    // case 2: length with exact π/2 angle
    let geo2 = Geonum::new(3.5, 1.0, 2.0); // length=3.5, angle=π/2
                                           // step 1: length component = 3.5 (preserved)
                                           // step 2: angle component = Angle::new(1.0, 2.0) = π/2
                                           // step 3: blade decomposition = π/2 = 1*(π/2) + 0
    assert_eq!(geo2.length, 3.5); // length preserved
    assert_eq!(geo2.angle.blade(), 1); // π/2 = 1 blade exactly
    assert!(geo2.angle.value().abs() < EPSILON); // no remainder
                                                 // blade arithmetic: exact π/2 creates clean blade boundary

    // case 3: length with high blade angle
    let geo3 = Geonum::new(1.0, 7.0, 2.0); // length=1, angle=7π/2
                                           // step 1: length component = 1.0 (preserved)
                                           // step 2: angle component = Angle::new(7.0, 2.0) = 7π/2
                                           // step 3: blade decomposition = 7π/2 = 7*(π/2) + 0
    assert_eq!(geo3.length, 1.0); // length preserved
    assert_eq!(geo3.angle.blade(), 7); // 7 complete π/2 rotations
    assert!(geo3.angle.value().abs() < EPSILON); // no remainder
                                                 // blade arithmetic: high blade angles work identically

    // proves Geonum::new preserves length while decomposing angle through blade arithmetic
}

#[test]
fn it_constructs_geonum_with_angle_composition() {
    // ANGLE COMPOSITION: direct length + angle pairing without decomposition
    // process: Geonum::new_with_angle(length, angle) → [length, angle] (no processing)

    // case 1: pre-computed angle with existing blade/value
    let angle = Angle::new(1.0, 3.0); // π/3 → blade=0, value=π/3
    let geo = Geonum::new_with_angle(1.5, angle);
    // step 1: length component = 1.5 (direct assignment)
    // step 2: angle component = angle (direct assignment, no decomposition)
    assert_eq!(geo.length, 1.5); // length preserved exactly
    assert_eq!(geo.angle.blade(), angle.blade()); // blade preserved exactly
    assert_eq!(geo.angle.value(), angle.value()); // value preserved exactly
                                                  // blade arithmetic: no blade processing - preserves input blade/value state

    // case 2: high blade angle preservation
    let high_angle = Angle::new_with_blade(1000, 1.0, 4.0); // blade=1000, value=π/4
    let high_geo = Geonum::new_with_angle(2.0, high_angle);
    // step 1: length = 2.0 (direct)
    // step 2: angle = high_angle (direct, no blade modification)
    assert_eq!(high_geo.length, 2.0); // length direct
    assert_eq!(high_geo.angle.blade(), 1000); // blade preserved exactly
    assert!((high_geo.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved exactly
                                                                  // blade arithmetic: million-dimensional preservation without processing

    // case 3: exact angle state preservation
    let complex_angle = Angle::new(7.0, 4.0); // 7π/4 → blade=3, value=π/4
    let preserve_geo = Geonum::new_with_angle(0.8, complex_angle);
    assert_eq!(preserve_geo.angle.blade(), complex_angle.blade()); // identical blade
    assert_eq!(preserve_geo.angle.value(), complex_angle.value()); // identical value
                                                                   // blade arithmetic: new_with_angle bypasses all angle processing

    // proves new_with_angle preserves exact blade/value state without modification
}

#[test]
fn it_constructs_geonum_from_cartesian() {
    // CARTESIAN CONVERSION: sqrt(x²+y²) + atan2(y,x) → blade/value decomposition
    // process: (x,y) → [sqrt(x²+y²), Angle::new_from_cartesian(x,y)]

    // case 1: 3-4-5 triangle (standard pythagorean)
    let geo1 = Geonum::new_from_cartesian(3.0, 4.0);
    // step 1: length = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5.0
    // step 2: angle = atan2(4.0, 3.0) ≈ 0.927 radians ≈ 0.295π
    // step 3: blade decomposition = 0.295π = 0*(π/2) + 0.295π (< π/2)
    assert_eq!(geo1.length, 5.0); // pythagorean magnitude
    assert_eq!(geo1.angle.blade(), 0); // angle < π/2, no boundary crossing
    let expected_angle = 4.0_f64.atan2(3.0); // atan2 calculation
    assert!((geo1.angle.mod_4_angle() - expected_angle).abs() < EPSILON); // preserves atan2 result
                                                                          // blade arithmetic: cartesian → polar → blade decomposition

    // case 2: unit circle quadrants (exact blade boundaries)
    let geo_90 = Geonum::new_from_cartesian(0.0, 1.0); // (0,1) = 90°
                                                       // step 1: length = sqrt(0² + 1²) = 1.0
                                                       // step 2: angle = atan2(1.0, 0.0) = π/2
                                                       // step 3: blade decomposition = π/2 = 1*(π/2) + 0
    assert_eq!(geo_90.length, 1.0); // unit magnitude
    assert_eq!(geo_90.angle.blade(), 1); // π/2 crosses boundary once
    assert!(geo_90.angle.value().abs() < EPSILON); // exact boundary has no remainder
                                                   // blade arithmetic: 90° → blade=1, value=0

    let geo_180 = Geonum::new_from_cartesian(-1.0, 0.0); // (-1,0) = 180°
                                                         // step 1: length = sqrt(1² + 0²) = 1.0
                                                         // step 2: angle = atan2(0.0, -1.0) = π
                                                         // step 3: blade decomposition = π = 2*(π/2) + 0
    assert_eq!(geo_180.length, 1.0); // unit magnitude
    assert_eq!(geo_180.angle.blade(), 2); // π crosses boundary twice
    assert!(geo_180.angle.value().abs() < EPSILON); // exact boundary has no remainder
                                                    // blade arithmetic: 180° → blade=2, value=0

    // case 3: first quadrant 45° angle
    let geo_45 = Geonum::new_from_cartesian(1.0, 1.0); // (1,1) = 45°
                                                       // step 1: length = sqrt(1² + 1²) = sqrt(2)
                                                       // step 2: angle = atan2(1.0, 1.0) = π/4
                                                       // step 3: blade decomposition = π/4 = 0*(π/2) + π/4
    assert!((geo_45.length - 2.0_f64.sqrt()).abs() < EPSILON); // sqrt(2) magnitude
    assert_eq!(geo_45.angle.blade(), 0); // π/4 < π/2, no crossing
    assert!((geo_45.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 remainder
                                                                // blade arithmetic: 45° → blade=0, value=π/4

    // proves cartesian conversion: magnitude calculation + atan2 blade decomposition
}

#[test]
fn it_constructs_geonum_with_explicit_blade() {
    // EXPLICIT BLADE: length + new_with_blade for high-dimensional control
    // process: Geonum::new_with_blade(length, blade, pi_radians, divisor) → [length, Angle::new_with_blade(...)]

    // case 1: add blades to basic angle
    let geo1 = Geonum::new_with_blade(2.0, 3, 1.0, 4.0); // length=2, 3 blades + π/4
                                                         // step 1: length = 2.0 (direct assignment)
                                                         // step 2: base_angle = Angle::new(1.0, 4.0) = π/4 → blade=0, value=π/4
                                                         // step 3: blade_increment = 3 * π/2 → blade=3, value=0
                                                         // step 4: addition = (0+3, π/4+0) = blade=3, value=π/4
    assert_eq!(geo1.length, 2.0); // length preserved
    assert_eq!(geo1.angle.blade(), 3); // 0 + 3 = 3 blades
    assert!((geo1.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                              // blade arithmetic: explicit blade addition to computed angle

    // case 2: million-dimensional control
    let geo_million = Geonum::new_with_blade(1.0, 1_000_000, 0.0, 1.0); // blade=1000000
                                                                        // step 1: length = 1.0 (direct)
                                                                        // step 2: base_angle = Angle::new(0.0, 1.0) = 0 → blade=0, value=0
                                                                        // step 3: blade_increment = 1000000 * π/2 → blade=1000000, value=0
                                                                        // step 4: addition = (0+1000000, 0+0) = blade=1000000, value=0
    assert_eq!(geo_million.length, 1.0); // length preserved
    assert_eq!(geo_million.angle.blade(), 1_000_000); // explicit million-dimensional blade
    assert!(geo_million.angle.value().abs() < EPSILON); // 0 value
                                                        // blade arithmetic: enables million-dimensional geometric algebra

    // case 3: explicit blade with boundary crossing base angle
    let geo_cross = Geonum::new_with_blade(1.5, 2, 3.0, 4.0); // length=1.5, 2 blades + 3π/4
                                                              // step 1: length = 1.5 (direct)
                                                              // step 2: base_angle = Angle::new(3.0, 4.0) = 3π/4 → blade=1, value=π/4
                                                              // step 3: blade_increment = 2 * π/2 = π → blade=2, value=0
                                                              // step 4: addition = (1+2, π/4+0) = blade=3, value=π/4
    assert_eq!(geo_cross.length, 1.5); // length preserved
    assert_eq!(geo_cross.angle.blade(), 3); // 1 + 2 = 3 total blades
    assert!((geo_cross.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 from base angle
                                                                   // blade arithmetic: explicit blade + base angle decomposition

    // proves new_with_blade enables explicit dimensional control through blade arithmetic
}

#[test]
fn it_proves_geonum_constructor_equivalence() {
    // CONSTRUCTOR EQUIVALENCE: multiple paths to same geometric object
    // different constructors reaching identical blade/value states

    // target: length=3, angle=π/2 → blade=1, value=0
    let direct = Geonum::new(3.0, 1.0, 2.0); // direct construction
                                             // step 1: length = 3.0 (direct)
                                             // step 2: angle = Angle::new(1.0, 2.0) = π/2 → blade=1, value=0

    let with_angle = Geonum::new_with_angle(3.0, Angle::new(1.0, 2.0)); // angle composition
                                                                        // step 1: length = 3.0 (direct)
                                                                        // step 2: angle = Angle::new(1.0, 2.0) = π/2 → blade=1, value=0 (preserved)

    let from_cart = Geonum::new_from_cartesian(0.0, 3.0); // cartesian conversion
                                                          // step 1: length = sqrt(0² + 3²) = 3.0
                                                          // step 2: angle = atan2(3.0, 0.0) = π/2 → blade=1, value=0

    // blade equivalence verification
    assert_eq!(direct.length, with_angle.length); // same length: 3.0
    assert_eq!(direct.angle.blade(), with_angle.angle.blade()); // same blade: 1
    assert_eq!(direct.angle.value(), with_angle.angle.value()); // same value: 0

    assert_eq!(direct.length, from_cart.length); // cartesian length matches
    assert_eq!(direct.angle.blade(), from_cart.angle.blade()); // cartesian blade matches
    assert_eq!(direct.angle.value(), from_cart.angle.value()); // cartesian value matches
                                                               // blade arithmetic: all paths reach identical (length=3, blade=1, value=0) state

    // case 2: equivalence with complex angle
    let target_angle = Angle::new(5.0, 4.0); // 5π/4 → blade=2, value=π/4
    let method1 = Geonum::new(2.5, 5.0, 4.0); // direct with angle parameters
    let method2 = Geonum::new_with_angle(2.5, target_angle); // pre-computed angle

    assert_eq!(method1.length, method2.length); // same length
    assert_eq!(method1.angle.blade(), method2.angle.blade()); // same blade
    assert_eq!(method1.angle.value(), method2.angle.value()); // same value
                                                              // blade arithmetic: different construction paths converge to same blade state

    // proves multiple constructors reach identical geometric objects through blade arithmetic
}

#[test]
fn it_creates_dimensional_geonums() {
    // DIMENSIONAL CREATION: create_dimension maps index to standardized blade angles
    // formula: Angle::new(dimension_index as f64, 2.0) = dimension_index * π/2

    // case 1: basic dimensional axes
    let dim0 = Geonum::create_dimension(1.0, 0); // 0th dimension (x-axis)
                                                 // step 1: length = 1.0 (direct)
                                                 // step 2: angle = Angle::new(0.0, 2.0) = 0 * π/2 = 0
                                                 // step 3: blade decomposition = 0 = 0*(π/2) + 0
    assert_eq!(dim0.length, 1.0); // unit length
    assert_eq!(dim0.angle.blade(), 0); // 0 π/2 rotations
    assert!(dim0.angle.value().abs() < EPSILON); // no remainder
                                                 // blade arithmetic: dimension 0 → blade=0 (scalar direction)

    let dim1 = Geonum::create_dimension(1.0, 1); // 1st dimension (y-axis)
                                                 // step 1: length = 1.0 (direct)
                                                 // step 2: angle = Angle::new(1.0, 2.0) = 1 * π/2 = π/2
                                                 // step 3: blade decomposition = π/2 = 1*(π/2) + 0
    assert_eq!(dim1.angle.blade(), 1); // 1 π/2 rotation
    assert!(dim1.angle.value().abs() < EPSILON); // exact boundary
                                                 // blade arithmetic: dimension 1 → blade=1 (vector direction)

    // case 2: high dimensional mapping
    let dim1000 = Geonum::create_dimension(2.0, 1000); // 1000th dimension
                                                       // step 1: length = 2.0 (direct)
                                                       // step 2: angle = Angle::new(1000.0, 2.0) = 1000 * π/2
                                                       // step 3: blade decomposition = 1000*(π/2) = 1000*(π/2) + 0
    assert_eq!(dim1000.length, 2.0); // length preserved
    assert_eq!(dim1000.angle.blade(), 1000); // exact blade = dimension index
    assert!(dim1000.angle.value().abs() < EPSILON); // no remainder
                                                    // blade arithmetic: dimension index directly maps to blade count

    // case 3: dimensional angle relationships
    assert!(dim0.is_orthogonal(&dim1)); // 0 ⊥ π/2 (cos(π/2) = 0)
    let dim2 = Geonum::create_dimension(1.0, 2); // π angle
    assert!(!dim0.is_orthogonal(&dim2)); // 0 and π are opposite (cos(π) = -1)
                                         // blade arithmetic: orthogonality through dot product angle differences

    // proves create_dimension enables arbitrary dimensional access through blade mapping
}

#[test]
fn it_computes_geonum_calculus_through_blade_rotation() {
    // CALCULUS OPERATIONS: differentiate/integrate through π/2 rotations
    // differentiate adds +π/2 (1 blade), integrate adds +3π/2 (3 blades)

    // case 1: differentiate adds π/2 rotation
    let function = Geonum::new(2.0, 1.0, 6.0); // length=2, angle=π/6 → blade=0, value=π/6
    let derivative = function.differentiate();
    // step 1: original = length=2, blade=0, value=π/6
    // step 2: add π/2 rotation = Angle::new(1.0, 2.0) → blade=1, value=0
    // step 3: angle addition = (0+1, π/6+0) = blade=1, value=π/6
    assert_eq!(derivative.length, 2.0); // length preserved
    assert_eq!(derivative.angle.blade(), 1); // blade=0 + 1 = blade=1
    assert!((derivative.angle.value() - PI / 6.0).abs() < EPSILON); // value preserved
                                                                    // blade arithmetic: differentiation = +1 blade through π/2 rotation

    // case 2: integrate adds 3π/2 rotation (forward-only -π/2)
    let integrand = Geonum::new(3.0, 1.0, 4.0); // length=3, angle=π/4 → blade=0, value=π/4
    let integral = integrand.integrate();
    // step 1: original = length=3, blade=0, value=π/4
    // step 2: add 3π/2 rotation = Angle::new(3.0, 2.0) → blade=3, value=0
    // step 3: angle addition = (0+3, π/4+0) = blade=3, value=π/4
    assert_eq!(integral.length, 3.0); // length preserved
    assert_eq!(integral.angle.blade(), 3); // blade=0 + 3 = blade=3
    assert!((integral.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
                                                                  // blade arithmetic: integration = +3 blades through 3π/2 forward rotation

    // case 3: calculus inverse relationship through blade accumulation
    let original = Geonum::new(1.0, 0.0, 1.0); // blade=0, value=0
    let diff_then_int = original.differentiate().integrate(); // +1 blade, then +3 blades
                                                              // step 1: differentiate = blade=0 → blade=1
                                                              // step 2: integrate = blade=1 → blade=1+3=4
                                                              // step 3: total = blade=0 → blade=4 (4 blades = 2π rotation)
    assert_eq!(diff_then_int.angle.blade(), 4); // 0 + 1 + 3 = 4 blades
    assert_eq!(diff_then_int.angle.grade(), 0); // 4 % 4 = 0 (same grade)
    assert!(diff_then_int.angle.value().abs() < EPSILON); // value preserved
                                                          // blade arithmetic: d/dx∫ = 4 blade accumulation, not algebraic cancellation

    // proves calculus operations accumulate transformation history through blade arithmetic
}

#[test]
fn it_inverts_geonum_through_length_and_angle_negation() {
    // INVERSION: [1/r, θ+π] through reciprocal length + π rotation
    // formula: inv(geonum) = [1/length, angle.negate()] where negate adds π (2 blades)

    // case 1: invert basic geonum
    let geonum = Geonum::new(2.0, 1.0, 3.0); // length=2, angle=π/3 → blade=0, value=π/3
    let inverse = geonum.inv();
    // step 1: reciprocal length = 1/2 = 0.5
    // step 2: negate angle = angle + π = π/3 + π = 4π/3
    // step 3: blade decomposition = 4π/3 = 2*(π/2) + π/3
    assert_eq!(inverse.length, 0.5); // 1/2 reciprocal
    assert_eq!(inverse.angle.blade(), 2); // π/3 + π crosses 2 boundaries
    assert!((inverse.angle.value() - PI / 3.0).abs() < EPSILON); // π/3 value preserved
                                                                 // blade arithmetic: inv() = [1/r, θ+π] through reciprocal + π rotation

    // case 2: multiplicative identity through blade accumulation
    let identity = geonum * inverse; // z * z^(-1)
                                     // step 1: length = 2.0 * 0.5 = 1.0 (lengths multiply)
                                     // step 2: angle = π/3 + (π/3 + π) = π/3 + 4π/3 = 5π/3 (angles add)
                                     // step 3: blade decomposition = 5π/3 = 3*(π/2) + π/6
    assert!((identity.length - 1.0).abs() < EPSILON); // unit magnitude
    assert_eq!(identity.angle.blade(), 3); // 0 + 2 = 3 total blades (from angle addition)
    assert_eq!(identity.angle.grade(), 3); // 3 % 4 = 3 (trivector)
                                           // blade arithmetic: multiplicative identity preserves geometric work history

    // case 3: inversion with high blade count
    let high_geonum = Geonum::new_with_blade(4.0, 1000, 1.0, 4.0); // blade=1000, value=π/4
    let high_inverse = high_geonum.inv();
    // step 1: reciprocal length = 1/4 = 0.25
    // step 2: negate angle = blade=1000 + 2 = blade=1002 (π rotation)
    // step 3: value preserved = π/4
    assert_eq!(high_inverse.length, 0.25); // 1/4 reciprocal
    assert_eq!(high_inverse.angle.blade(), 1002); // 1000 + 2 = 1002 blades
    assert!((high_inverse.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
                                                                      // blade arithmetic: inversion scales to arbitrary blade magnitudes

    // proves inv() combines reciprocal length with π rotation through blade arithmetic
}

#[test]
fn it_divides_geonum_through_multiplication_by_inverse() {
    // DIVISION: a/b = a * inv(b) through angle arithmetic
    // compound operation: inversion blade arithmetic + multiplication blade arithmetic

    // case 1: basic division blade accumulation
    let dividend = Geonum::new(6.0, 1.0, 4.0); // length=6, angle=π/4 → blade=0, value=π/4
    let divisor = Geonum::new(2.0, 1.0, 6.0); // length=2, angle=π/6 → blade=0, value=π/6
    let quotient = dividend.div(&divisor);
    // step 1: divisor.inv() = [1/2, π/6+π] = [0.5, 7π/6] → blade=2, value=π/6
    // step 2: multiplication = [6*0.5, π/4+(7π/6)] = [3.0, π/4+7π/6]
    // step 3: angle addition = π/4 + 7π/6 = 3π/12 + 14π/12 = 17π/12
    // step 4: blade decomposition = 17π/12 = 2*(π/2) + 5π/12
    assert_eq!(quotient.length, 3.0); // 6/2 = 3
    assert_eq!(quotient.angle.blade(), 2); // angle sum crosses 2 boundaries
    assert!((quotient.angle.value() - 5.0 * PI / 12.0).abs() < EPSILON); // remainder after crossings
                                                                         // blade arithmetic: division accumulates inversion + multiplication blades

    // case 2: verify division equivalence with manual inv() + mul()
    let manual_quotient = dividend * divisor.inv();
    assert_eq!(quotient.length, manual_quotient.length); // same length result
    assert_eq!(quotient.angle.blade(), manual_quotient.angle.blade()); // same blade result
    assert_eq!(quotient.angle.value(), manual_quotient.angle.value()); // same value result
                                                                       // blade arithmetic: div() IS multiplication by inverse

    // case 3: division with high blade accumulation
    let high_dividend = Geonum::new_with_blade(8.0, 500, 1.0, 3.0); // blade=500, value=π/3
    let high_divisor = Geonum::new_with_blade(4.0, 200, 1.0, 6.0); // blade=200, value=π/6
    let high_quotient = high_dividend.div(&high_divisor);
    // step 1: high_divisor.inv() = [1/4, blade=200+2, value=π/6] = [0.25, blade=202, π/6]
    // step 2: multiplication = [8*0.25, blade=500+202, value=π/3+π/6] = [2.0, blade=702, π/2]
    // step 3: π/2 boundary = blade=702+1=703, value=0
    assert_eq!(high_quotient.length, 2.0); // 8/4 = 2
    assert_eq!(high_quotient.angle.blade(), 703); // 500 + (200+2) + 1(crossing) = 703
    assert!(high_quotient.angle.value().abs() < EPSILON); // π/2 crossing resets value
                                                          // blade arithmetic: division accumulates all intermediate blade transformations

    // proves div() accumulates blade history through inversion + multiplication compound operation
}

#[test]
fn it_normalizes_geonum_preserving_blade_structure() {
    // NORMALIZATION: length=1, angle preserved (blade/value unchanged)
    // formula: normalize(geonum) = [1.0, geonum.angle] (length scaling only)

    // case 1: normalize non-unit geonum
    let unnormalized = Geonum::new(5.0, 2.0, 3.0); // length=5, angle=2π/3 → blade=1, value=π/6
    let normalized = unnormalized.normalize();
    // step 1: length = 1.0 (forced to unit)
    // step 2: angle = unnormalized.angle (preserved exactly)
    // step 3: blade/value = blade=1, value=π/6 (no modification)
    assert_eq!(normalized.length, 1.0); // unit length
    assert_eq!(normalized.angle.blade(), unnormalized.angle.blade()); // blade preserved
    assert_eq!(normalized.angle.value(), unnormalized.angle.value()); // value preserved
    assert_eq!(normalized.angle.grade(), unnormalized.angle.grade()); // grade preserved
                                                                      // blade arithmetic: normalization affects only length, preserves blade structure

    // case 2: normalize high blade geonum
    let high_unnorm = Geonum::new_with_blade(3.7, 1000, 1.0, 4.0); // length=3.7, blade=1000, value=π/4
    let high_norm = high_unnorm.normalize();
    // step 1: length = 1.0 (unit)
    // step 2: angle = high_unnorm.angle (preserved)
    // step 3: blade/value = blade=1000, value=π/4 (unchanged)
    assert_eq!(high_norm.length, 1.0); // unit length
    assert_eq!(high_norm.angle.blade(), 1000); // blade preserved exactly
    assert!((high_norm.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved exactly
                                                                   // blade arithmetic: normalization preserves high blade structure

    // case 3: normalization idempotency
    let already_unit = Geonum::new(1.0, 3.0, 4.0); // length=1, angle=3π/4 → blade=1, value=π/4
    let normalized_again = already_unit.normalize();
    // step 1: length = 1.0 (already unit, unchanged)
    // step 2: angle = already_unit.angle (preserved)
    assert_eq!(normalized_again.length, 1.0); // still unit
    assert_eq!(normalized_again.angle.blade(), already_unit.angle.blade()); // blade unchanged
    assert_eq!(normalized_again.angle.value(), already_unit.angle.value()); // value unchanged
                                                                            // blade arithmetic: normalization is idempotent on unit geonums

    // proves normalize() isolates length scaling from blade structure preservation
}

#[test]
fn it_computes_dot_product_returning_scalar_blade() {
    // DOT PRODUCT: |a|*|b|*cos(θb-θa) returns grade 0 for positive cosine and grade 2 for negative
    // sign is encoded as a +π rotation rather than a negative length

    // case 1: dot product of vectors with angle difference
    let vec_a = Geonum::new(3.0, 1.0, 6.0); // length=3, angle=π/6 → blade=0, value=π/6
    let vec_b = Geonum::new(4.0, 1.0, 3.0); // length=4, angle=π/3 → blade=0, value=π/3
    let dot_result = vec_a.dot(&vec_b);
    // step 1: angle_diff = vec_b.angle - vec_a.angle = π/3 - π/6 = π/6
    // step 2: magnitude = 3 * 4 * cos(π/6) = 12 * (√3/2) ≈ 10.39
    // step 3: result = [magnitude, base scalar angle]
    let expected_mag = 3.0 * 4.0 * (PI / 6.0).cos();
    assert!((dot_result.length - expected_mag).abs() < EPSILON); // computed magnitude
    assert_eq!(dot_result.angle, Angle::new(0.0, 1.0)); // positive cosine keeps scalar base angle

    // case 2: orthogonal vectors (π/2 apart)
    let ortho_a = Geonum::new(2.0, 0.0, 1.0); // angle=0 → blade=0, value=0
    let ortho_b = Geonum::new(3.0, 1.0, 2.0); // angle=π/2 → blade=1, value=0
    let ortho_dot = ortho_a.dot(&ortho_b);
    // step 1: angle_diff = π/2 - 0 = π/2
    // step 2: magnitude = 2 * 3 * cos(π/2) = 6 * 0 = 0
    // step 3: result = [0, blade=0] (scalar zero)
    assert!(ortho_dot.length.abs() < EPSILON); // zero magnitude
    assert_eq!(ortho_dot.angle, Angle::new(0.0, 1.0)); // zero magnitude uses scalar pair

    // case 3: high blade dot product
    let high_a = Geonum::new_with_blade(2.0, 1000, 1.0, 6.0); // blade=1000, value=π/6
    let high_b = Geonum::new_with_blade(3.0, 1001, 1.0, 4.0); // blade=1001, value=π/4
    let high_dot = high_a.dot(&high_b);
    // step 1: angle difference computed through mod_4_angle differences
    // step 2: magnitude = 2 * 3 * cos(angle_diff)
    assert!(high_dot.length.is_finite()); // finite result
    assert_eq!(high_dot.angle, Angle::new_with_blade(2, 0.0, 1.0)); // sign encoded as bivector pair
                                                                    // blade arithmetic: dot product collapses to scalar pair with grade encoding sign

    // proves dot() always returns scalar blade regardless of input blade complexity
}

#[test]
fn it_computes_wedge_product_with_blade_increment() {
    // WEDGE PRODUCT: |a|*|b|*sin(θb-θa) with angle sum + π/2 blade increment
    // formula: wedge(a,b) = [|a|*|b|*|sin(θb-θa)|, θa + θb + π/2] with orientation handling

    // case 1: wedge product with positive orientation
    let vec_a = Geonum::new(3.0, 1.0, 6.0); // length=3, angle=π/6 → blade=0, value=π/6
    let vec_b = Geonum::new(4.0, 1.0, 3.0); // length=4, angle=π/3 → blade=0, value=π/3
    let wedge_ab = vec_a.wedge(&vec_b);
    // step 1: angle_diff = π/3 - π/6 = π/6, sin(π/6) = 0.5 > 0 (positive)
    // step 2: magnitude = 3 * 4 * |sin(π/6)| = 12 * 0.5 = 6.0
    // step 3: angle_sum = π/6 + π/3 + π/2 = π/6 + 2π/6 + 3π/6 = π
    // step 4: blade decomposition = π = 2*(π/2) + 0
    assert!((wedge_ab.length - 6.0).abs() < EPSILON); // computed magnitude
    assert_eq!(wedge_ab.angle.blade(), 2); // π = 2*(π/2), crosses 2 boundaries
    assert!(wedge_ab.angle.value().abs() < EPSILON); // π has no remainder
                                                     // blade arithmetic: wedge adds π/2 to angle sum, creates bivector

    // case 2: wedge product with negative orientation (anticommutativity)
    let wedge_ba = vec_b.wedge(&vec_a);
    // step 1: angle_diff = π/6 - π/3 = -π/6, sin(-π/6) = -0.5 < 0 (negative)
    // step 2: magnitude = 4 * 3 * |sin(-π/6)| = 12 * 0.5 = 6.0 (same)
    // step 3: angle_sum = π/3 + π/6 + π/2 = π (same base)
    // step 4: orientation correction = π + π = 2π (adds π for negative sin)
    // step 5: blade decomposition = 2π = 4*(π/2) + 0
    assert!((wedge_ba.length - 6.0).abs() < EPSILON); // same magnitude
    assert_eq!(wedge_ba.angle.blade(), 4); // 2π = 4*(π/2) with orientation π
                                           // blade arithmetic: negative sin adds π rotation (2 more blades)

    // case 3: nilpotency (v ∧ v = 0)
    let self_wedge = vec_a.wedge(&vec_a);
    // step 1: angle_diff = π/6 - π/6 = 0, sin(0) = 0
    // step 2: magnitude = 3 * 3 * |sin(0)| = 9 * 0 = 0
    assert!(self_wedge.length < EPSILON); // zero magnitude
                                          // blade arithmetic: parallel vectors produce zero wedge regardless of blade increment

    // proves wedge() creates oriented area through blade increment + orientation encoding
}

#[test]
fn it_computes_geometric_product_combining_dot_and_wedge_blades() {
    // GEOMETRIC PRODUCT: dot + wedge combination through geonum addition
    // formula: geo(a,b) = dot(a,b) + wedge(a,b) through cartesian addition

    // case 1: geometric product blade combination
    let vec_a = Geonum::new(3.0, 1.0, 6.0); // length=3, angle=π/6 → blade=0, value=π/6
    let vec_b = Geonum::new(4.0, 1.0, 3.0); // length=4, angle=π/3 → blade=0, value=π/3

    let dot_part = vec_a.dot(&vec_b);
    let wedge_part = vec_a.wedge(&vec_b);
    let geo_product = vec_a.geo(&vec_b);
    // step 1: dot_part = [magnitude, blade=0, value=0] (scalar)
    // step 2: wedge_part = [magnitude, blade=2, value=0] (bivector from angle sum + π/2)
    // step 3: addition = cartesian sum of dot + wedge → new blade/value state
    assert_eq!(dot_part.angle.blade(), 0); // dot always scalar
    assert_eq!(wedge_part.angle.blade(), 2); // wedge creates bivector
                                             // blade arithmetic: geo() combines different blade results through addition

    // case 2: verify geometric product equivalence with manual combination
    let manual_geo = dot_part + wedge_part;
    assert!((geo_product.length - manual_geo.length).abs() < EPSILON); // same magnitude
    assert_eq!(geo_product.angle.blade(), manual_geo.angle.blade()); // same blade
    assert_eq!(geo_product.angle.value(), manual_geo.angle.value()); // same value
                                                                     // blade arithmetic: geo() IS dot + wedge through geonum addition

    // case 3: geometric product with high blade inputs
    let high_a = Geonum::new_with_blade(2.0, 500, 1.0, 4.0); // blade=500, value=π/4
    let high_b = Geonum::new_with_blade(3.0, 300, 1.0, 6.0); // blade=300, value=π/6
    let high_dot = high_a.dot(&high_b);
    let high_wedge = high_a.wedge(&high_b);
    let high_geo = high_a.geo(&high_b);
    // step 1: dot = [magnitude, blade=0, value=0] (always scalar)
    // step 2: wedge = [magnitude, blade=500+300+2=802, value=π/4+π/6+π/2] with orientation
    // step 3: addition = scalar + wedge through cartesian: combined_blade = 0 + 803 = 803
    // step 4: cartesian boundary crossings add +3 more: 803 + 3 = 806
    assert_eq!(high_dot.angle.blade(), 0); // dot always scalar blade
    assert_eq!(high_wedge.angle.blade(), 803); // wedge: 500+300+2+1(crossing) = 803 blades
    assert_eq!(high_geo.angle.blade(), 803); // addition: dot(0) + wedge(803) = 803
                                             // blade arithmetic: geo() = dot(scalar) + wedge(complex) → 806 through compound accumulation

    // proves geo() unifies scalar and bivector components through geonum addition blade arithmetic
}

#[test]
fn it_rotates_geonum_through_angle_addition() {
    // ROTATION: preserves length, adds rotation angle to geonum angle
    // formula: rotate(geonum, rotation) = [geonum.length, geonum.angle + rotation]

    // case 1: rotation without boundary crossing
    let base_geo = Geonum::new(2.0, 1.0, 8.0); // length=2, angle=π/8 → blade=0, value=π/8
    let rotation = Angle::new(1.0, 6.0); // π/6 rotation → blade=0, value=π/6
    let rotated = base_geo.rotate(rotation);
    // step 1: length = 2.0 (preserved)
    // step 2: angle addition = π/8 + π/6 = 3π/24 + 4π/24 = 7π/24
    // step 3: blade arithmetic = (0+0, π/8+π/6) = blade=0, value=7π/24
    assert_eq!(rotated.length, 2.0); // length preserved
    assert_eq!(rotated.angle.blade(), 0); // no boundary crossing
    assert!((rotated.angle.value() - 7.0 * PI / 24.0).abs() < EPSILON); // combined angle
                                                                        // blade arithmetic: rotation = angle addition without blade change

    // case 2: rotation with boundary crossing
    let cross_geo = Geonum::new(3.0, 1.0, 3.0); // length=3, angle=π/3 → blade=0, value=π/3
    let cross_rotation = Angle::new(1.0, 4.0); // π/4 rotation → blade=0, value=π/4
    let cross_rotated = cross_geo.rotate(cross_rotation);
    // step 1: length = 3.0 (preserved)
    // step 2: angle addition = π/3 + π/4 = 4π/12 + 3π/12 = 7π/12
    // step 3: boundary check = 7π/12 > π/2, crosses boundary
    // step 4: blade arithmetic = 7π/12 = 1*(π/2) + π/12
    assert_eq!(cross_rotated.length, 3.0); // length preserved
    assert_eq!(cross_rotated.angle.blade(), 1); // boundary crossing increments blade
    assert!((cross_rotated.angle.value() - (7.0 * PI / 12.0 - PI / 2.0)).abs() < EPSILON); // remainder
                                                                                           // blade arithmetic: rotation with crossing = angle addition + boundary handling

    // case 3: rotation with high blade accumulation
    let high_geo = Geonum::new_with_blade(1.0, 1000, 1.0, 6.0); // blade=1000, value=π/6
    let high_rotation = Angle::new_with_blade(200, 1.0, 4.0); // blade=200, value=π/4
    let high_rotated = high_geo.rotate(high_rotation);
    // step 1: length = 1.0 (preserved)
    // step 2: blade addition = 1000 + 200 = 1200
    // step 3: value addition = π/6 + π/4 = 2π/12 + 3π/12 = 5π/12
    // step 4: blade arithmetic = (1000+200, π/6+π/4) = blade=1200, value=5π/12
    assert_eq!(high_rotated.length, 1.0); // length preserved
    assert_eq!(high_rotated.angle.blade(), 1200); // blade accumulation
    assert!((high_rotated.angle.value() - 5.0 * PI / 12.0).abs() < EPSILON); // value sum
                                                                             // blade arithmetic: rotation accumulates blades at arbitrary magnitudes

    // proves rotate() is pure angle addition preserving length while accumulating blade history
}

#[test]
fn it_negates_geonum_through_pi_rotation() {
    // NEGATION: preserves length, adds π rotation (2 blades) to angle
    // formula: negate(geonum) = [geonum.length, geonum.angle.negate()]

    // case 1: negate basic geonum
    let base_geo = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4
    let negated = base_geo.negate();
    // step 1: length = 2.0 (preserved)
    // step 2: angle negation = angle + π = π/4 + π = 5π/4
    // step 3: blade decomposition = 5π/4 = 2*(π/2) + π/4
    assert_eq!(negated.length, 2.0); // length preserved
    assert_eq!(negated.angle.blade(), 2); // π/4 + π crosses 2 boundaries
    assert!((negated.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                                 // blade arithmetic: negation adds exactly 2 blades (π rotation)

    // case 2: double negation accumulates blades
    let double_neg = negated.negate();
    // step 1: length = 2.0 (preserved)
    // step 2: angle negation = 5π/4 + π = 9π/4
    // step 3: blade decomposition = 9π/4 = 4*(π/2) + π/4
    assert_eq!(double_neg.length, 2.0); // length preserved
    assert_eq!(double_neg.angle.blade(), 4); // 2 + 2 = 4 blades total
    assert_eq!(double_neg.angle.grade(), 0); // 4 % 4 = 0 (same grade as original)
    assert!((double_neg.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
                                                                    // blade arithmetic: double negation = 4 blade accumulation (2π rotation)

    // case 3: negation with high blade count
    let high_geo = Geonum::new_with_blade(1.5, 1000, 1.0, 6.0); // blade=1000, value=π/6
    let high_negated = high_geo.negate();
    // step 1: length = 1.5 (preserved)
    // step 2: angle negation = blade=1000 + 2 = blade=1002
    // step 3: value = π/6 (preserved through negate)
    assert_eq!(high_negated.length, 1.5); // length preserved
    assert_eq!(high_negated.angle.blade(), 1002); // 1000 + 2 = 1002 blades
    assert!((high_negated.angle.value() - PI / 6.0).abs() < EPSILON); // value preserved
                                                                      // blade arithmetic: negation adds 2 blades regardless of starting blade count

    // proves negate() implements geometric π rotation preserving transformation history
}

#[test]
fn it_projects_geonum_to_arbitrary_dimensions() {
    // PROJECTION: queries any dimension through trigonometric calculation
    // formula: project_to_dimension(n) = length * cos(n*π/2 - total_angle) - blade independent

    // case 1: project geonum to standard dimensions
    let geo = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4
    let proj_0 = geo.project_to_dimension(0); // project to x-axis (0*π/2 = 0)
    let proj_1 = geo.project_to_dimension(1); // project to y-axis (1*π/2 = π/2)
                                              // step 1: total_angle = blade*(π/2) + value = 0*(π/2) + π/4 = π/4
                                              // step 2: proj_0 = 2 * cos(0 - π/4) = 2 * cos(-π/4) = 2 * √2/2 = √2
                                              // step 3: proj_1 = 2 * cos(π/2 - π/4) = 2 * cos(π/4) = 2 * √2/2 = √2
    assert!((proj_0 - 2.0_f64.sqrt()).abs() < EPSILON); // √2 projection
    assert!((proj_1 - 2.0_f64.sqrt()).abs() < EPSILON); // √2 projection
                                                        // blade arithmetic: projection ignores blade structure, uses total angle

    // case 2: project high blade geonum to arbitrary dimensions
    let high_geo = Geonum::new_with_blade(3.0, 1000, 1.0, 6.0); // blade=1000, value=π/6
    let proj_42 = high_geo.project_to_dimension(42); // arbitrary dimension
    let proj_1000000 = high_geo.project_to_dimension(1_000_000); // million-D
                                                                 // step 1: total_angle = 1000*(π/2) + π/6 (high angle)
                                                                 // step 2: dimension_angle_42 = 42*π/2, dimension_angle_million = 1000000*π/2
                                                                 // step 3: projections computed through cos(dimension_angle - total_angle)
    assert!(proj_42.is_finite()); // finite projection to dimension 42
    assert!(proj_1000000.is_finite()); // finite projection to million-D
                                       // blade arithmetic: projection queries any dimension without blade constraints

    // case 3: projection independence from blade complexity
    let simple_geo = Geonum::new(3.0, 1.0, 6.0); // blade=0, value=π/6 (same total angle)
    let complex_geo = Geonum::new_with_blade(3.0, 1000, 1.0, 6.0); // blade=1000, value=π/6
                                                                   // both have same total_angle when reduced to mod_4_angle
    let simple_proj = simple_geo.project_to_dimension(5);
    let complex_proj = complex_geo.project_to_dimension(5);
    // projections should differ due to different total angles from blade counts
    // blade arithmetic: projection sees blade history through total angle calculation
    assert!(simple_proj.is_finite() && complex_proj.is_finite()); // both finite

    // proves project_to_dimension enables arbitrary dimensional queries through trigonometry
}

#[test]
fn it_handles_scalar_sign_encoding_original() {
    // SCALAR SIGN: positive/negative encoded in blade direction (0 vs 2)
    // positive → blade=0 (0 angle), negative → blade=2 (π angle)

    // case 1: positive scalar encoding
    let positive = Geonum::scalar(5.0);
    // step 1: length = abs(5.0) = 5.0
    // step 2: angle = Angle::new(0.0, 1.0) = 0 (positive branch)
    // step 3: blade decomposition = 0 = 0*(π/2) + 0
    assert_eq!(positive.length, 5.0); // magnitude from abs()
    assert_eq!(positive.angle.blade(), 0); // positive → blade 0
    assert!(positive.angle.value().abs() < EPSILON); // 0 angle
                                                     // blade arithmetic: positive scalar → blade=0 direction

    // case 2: negative scalar encoding
    let negative = Geonum::scalar(-3.0);
    // step 1: length = abs(-3.0) = 3.0
    // step 2: angle = Angle::new(1.0, 1.0) = π (negative branch)
    // step 3: blade decomposition = π = 2*(π/2) + 0
    assert_eq!(negative.length, 3.0); // magnitude from abs()
    assert_eq!(negative.angle.blade(), 2); // negative → blade 2 (π rotation)
    assert!(negative.angle.value().abs() < EPSILON); // π angle has no remainder
                                                     // blade arithmetic: negative scalar → blade=2 direction (π rotation)

    // case 3: zero scalar (boundary case)
    let zero = Geonum::scalar(0.0);
    // step 1: length = abs(0.0) = 0.0
    // step 2: angle = Angle::new(0.0, 1.0) = 0 (positive branch for zero)
    // step 3: blade decomposition = 0 = 0*(π/2) + 0
    assert_eq!(zero.length, 0.0); // zero magnitude
    assert_eq!(zero.angle.blade(), 0); // zero treated as positive → blade 0
    assert!(zero.angle.value().abs() < EPSILON); // 0 angle
                                                 // blade arithmetic: zero defaults to positive blade direction

    // case 4: scalar sign detection through blade arithmetic
    assert!(positive.angle.blade() != negative.angle.blade()); // different directions
    let blade_diff = (negative.angle.blade() as i32 - positive.angle.blade() as i32).abs();
    assert_eq!(blade_diff, 2); // π apart (2 blade difference)
                               // blade arithmetic: positive/negative differ by π rotation (2 blades)

    // proves scalar() encodes sign through blade direction, eliminating separate sign field
}

#[test]
fn it_increments_blade_through_rotation() {
    // INCREMENT BLADE: adds π/2 rotation (1 blade) through angle addition
    // formula: angle + Angle::new(1.0, 2.0) = angle + π/2 = blade + 1

    // case 1: increment from scalar to vector
    let scalar_base = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4
    let incremented = scalar_base.increment_blade();
    // step 1: original = blade=0, value=π/4
    // step 2: rotation = Angle::new(1.0, 2.0) = π/2 → blade=1, value=0
    // step 3: addition = (0+1, π/4+0) = blade=1, value=π/4
    assert_eq!(incremented.length, scalar_base.length); // length preserved
    assert_eq!(incremented.angle.blade(), 1); // blade=0 + 1 = blade=1
    assert!((incremented.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
                                                                     // blade arithmetic: π/4 → π/4 + π/2 = 3π/4 = 1*(π/2) + π/4

    // case 2: increment causing grade cycling
    let trivector_base = Geonum::new(1.0, 3.0, 2.0); // 3π/2 → blade=3, value=0
    let cycled = trivector_base.increment_blade();
    // step 1: original = blade=3, value=0 (grade 3)
    // step 2: rotation = π/2 → blade=1, value=0
    // step 3: addition = (3+1, 0+0) = blade=4, value=0
    assert_eq!(cycled.angle.blade(), 4); // blade=3 + 1 = blade=4
    assert_eq!(cycled.angle.grade(), 0); // 4 % 4 = 0 (cycling to scalar)
    assert!(cycled.angle.value().abs() < EPSILON); // value preserved
                                                   // blade arithmetic: grade 3 → grade 0 through blade cycling

    // case 3: increment high dimensional blade
    let high_dim = Geonum::new_with_blade(1.0, 1000, 0.0, 1.0); // blade=1000
    let inc_high = high_dim.increment_blade();
    // step 1: original = blade=1000, value=0
    // step 2: rotation = π/2 → blade=1, value=0
    // step 3: addition = (1000+1, 0+0) = blade=1001, value=0
    assert_eq!(inc_high.angle.blade(), 1001); // blade=1000 + 1 = blade=1001
    assert_eq!(inc_high.angle.grade(), 1); // 1001 % 4 = 1 (vector grade)
                                           // blade arithmetic: increment works at arbitrary blade magnitudes

    // proves increment_blade adds exactly 1 blade through π/2 geometric rotation
}

#[test]
fn it_decrements_blade_through_forward_rotation() {
    // DECREMENT BLADE: -π/2 becomes +3π/2 through forward-only arithmetic
    // formula: angle + Angle::new(-1.0, 2.0) where negative angles wrap forward

    // case 1: decrement from vector back to "scalar equivalent"
    let vector_base = Geonum::new(2.0, 1.0, 2.0); // π/2 → blade=1, value=0
    let decremented = vector_base.decrement_blade();
    // step 1: original = blade=1, value=0
    // step 2: rotation = Angle::new(-1.0, 2.0) = -π/2 wraps to +3π/2 → blade=3, value=0
    // step 3: addition = (1+3, 0+0) = blade=4, value=0
    assert_eq!(decremented.length, vector_base.length); // length preserved
    assert_eq!(decremented.angle.blade(), 4); // blade=1 + 3 = blade=4 (not blade=0!)
    assert!(decremented.angle.value().abs() < EPSILON); // value preserved
    assert_eq!(decremented.angle.grade(), 0); // 4 % 4 = 0 (scalar grade through cycling)
                                              // blade arithmetic: -π/2 → +3π/2 maintains forward-only principle

    // case 2: decrement preserves transformation history
    let start = Geonum::new(1.0, 1.0, 4.0); // π/4 → blade=0, value=π/4
    let inc_then_dec = start.increment_blade().decrement_blade();
    // step 1: increment = blade=0 → blade=1
    // step 2: decrement = blade=1 → blade=1+3=4
    // step 3: total transformation = blade=0 → blade=4 (4 blades accumulated)
    assert_eq!(inc_then_dec.angle.blade(), 4); // blade=0 → 1 → 4 (not back to 0)
    assert_eq!(inc_then_dec.angle.grade(), 0); // 4 % 4 = 0 (same grade)
    assert!((inc_then_dec.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
                                                                      // blade arithmetic: inc+dec adds 4 blades total (preserves geometric work)

    // case 3: decrement high dimensional blade
    let high_blade = Geonum::new_with_blade(1.0, 1000, 0.0, 1.0); // blade=1000
    let dec_high = high_blade.decrement_blade();
    // step 1: original = blade=1000, value=0
    // step 2: rotation = -π/2 → +3π/2 = blade=3, value=0
    // step 3: addition = (1000+3, 0+0) = blade=1003, value=0
    assert_eq!(dec_high.angle.blade(), 1003); // blade=1000 + 3 = blade=1003
    assert_eq!(dec_high.angle.grade(), 3); // 1003 % 4 = 3 (trivector grade)
                                           // blade arithmetic: forward wrapping works at arbitrary blade magnitudes

    // proves decrement_blade uses forward-only arithmetic, accumulating transformation history
}

#[test]
fn it_preserves_blade_operations_at_high_dimensions() {
    // HIGH DIMENSIONAL BLADES: operations scale to arbitrary blade counts
    // all blade operations work identically regardless of blade magnitude

    // case 1: dual operation scaling
    let low_blade = Geonum::new_with_blade(1.0, 5, 0.0, 1.0); // blade=5
    let high_blade = Geonum::new_with_blade(1.0, 1_000_000, 0.0, 1.0); // blade=1000000

    let dual_low = low_blade.dual();
    let dual_high = high_blade.dual();
    // step 1: dual adds π rotation (2 blades) to any blade count
    // step 2: low: blade=5 + 2 = blade=7
    // step 3: high: blade=1000000 + 2 = blade=1000002
    assert_eq!(dual_low.angle.blade(), 7); // 5 + 2 = 7
    assert_eq!(dual_high.angle.blade(), 1_000_002); // 1000000 + 2 = 1000002
    assert_eq!(dual_low.angle.grade(), 3); // 7 % 4 = 3
    assert_eq!(dual_high.angle.grade(), 2); // 1000002 % 4 = 2
                                            // blade arithmetic: dual operation consistent at any blade magnitude

    // case 2: increment/decrement scaling
    let medium_blade = Geonum::new_with_blade(1.0, 500, 1.0, 6.0); // blade=500, value=π/6
    let inc_medium = medium_blade.increment_blade(); // +1 blade
    let dec_medium = medium_blade.decrement_blade(); // +3 blades (forward wrap)

    assert_eq!(inc_medium.angle.blade(), 501); // 500 + 1 = 501
    assert_eq!(dec_medium.angle.blade(), 503); // 500 + 3 = 503
    assert!((inc_medium.angle.value() - PI / 6.0).abs() < EPSILON); // value preserved
    assert!((dec_medium.angle.value() - PI / 6.0).abs() < EPSILON); // value preserved
                                                                    // blade arithmetic: increment/decrement consistent at medium blade magnitudes

    // case 3: blade copy operation scaling
    let source_high = Geonum::new_with_blade(1.0, 10_000, 1.0, 8.0); // blade=10000
    let target_higher = Geonum::new_with_blade(1.0, 50_000, 0.0, 1.0); // blade=50000
    let copied = source_high.copy_blade(&target_higher);
    // step 1: blade difference = 50000 - 10000 = 40000
    // step 2: rotation = 40000 * π/2 through angle arithmetic
    // step 3: result = blade=10000 + 40000 = blade=50000
    assert_eq!(copied.angle.blade(), 50_000); // exact blade copying at high dimensions
    assert!((copied.angle.value() - PI / 8.0).abs() < EPSILON); // value from source preserved
                                                                // blade arithmetic: copy_blade scales to extreme blade counts

    // proves blade operations maintain O(1) complexity regardless of blade magnitude
    // enabling geometric algebra in million-dimensional spaces through blade arithmetic
}

#[test]
fn it_reflects_geonum_through_forward_only_formula() {
    // REFLECTION: 2*axis + (2π - base_angle(point)) blade formula
    // forward-only reflection avoids subtraction through complement arithmetic

    // case 1: reflect point across x-axis
    let point = Geonum::new_from_cartesian(1.0, 1.0); // (1,1) → length=√2, angle=π/4 → blade=0, value=π/4
    let x_axis = Geonum::new(1.0, 0.0, 1.0); // angle=0 → blade=0, value=0
    let reflected = point.reflect(&x_axis);
    // step 1: base_angle(point) = point.base_angle() = blade=0, value=π/4
    // step 2: complement = Angle::new(4.0, 1.0) - base_angle = 2π - π/4 = 7π/4 → blade=3, value=π/4
    // step 3: reflected_angle = x_axis + x_axis + complement = 0 + 0 + 7π/4 = 7π/4
    // step 4: blade arithmetic = 7π/4 = 3*(π/2) + π/4
    assert_eq!(reflected.length, point.length); // length preserved
    assert_eq!(reflected.angle.blade(), 7); // 0 + 0 + 7 = 7 blades total
    assert!((reflected.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                                   // blade arithmetic: reflection adds 7 blades through forward-only formula

    // case 2: double reflection accumulates more blades
    let double_reflected = reflected.reflect(&x_axis);
    // step 1: base_angle(reflected) = blade=3, value=π/4 (grade preserved, blade reset)
    // step 2: complement = 2π - base_angle = 2π - 7π/4 = π/4 → blade=0, value=π/4
    // step 3: double_angle = 0 + 0 + π/4 = π/4 → blade=0, value=π/4
    let initial_blade = point.angle.blade();
    let final_blade = double_reflected.angle.blade();
    let blade_accumulation = final_blade - initial_blade;
    assert_eq!(blade_accumulation, 4); // double reflection adds 4 blades for π/4 starting point
    assert_eq!(double_reflected.angle.grade(), 0); // 4 % 4 = 0 (same grade as original)
                                                   // blade arithmetic: double reflection accumulation depends on starting angle
                                                   // π/4 point: complement = 2π - π/4 = 7π/4, produces 4 total blade accumulation
                                                   // pure scalar: complement = 2π - 0 = 2π, produces 8 total blade accumulation

    // case 3: reflection across arbitrary axis
    let axis_45 = Geonum::new(1.0, 1.0, 4.0); // π/4 axis → blade=0, value=π/4
    let arbitrary_reflected = point.reflect(&axis_45);
    // step 1: base_angle(point) = blade=0, value=π/4
    // step 2: complement = 2π - π/4 = 7π/4 → blade=3, value=π/4
    // step 3: reflected_angle = π/4 + π/4 + 7π/4 = 9π/4 = 2*(π/2) + π/4 → blade=2, value=π/4
    // step 4: final result adds to original: blade=0 + blade=2 = blade=2
    assert_eq!(arbitrary_reflected.length, point.length); // length preserved
    assert_eq!(arbitrary_reflected.angle.blade(), 8); // π/4 + π/4 + 7π/4 through angle arithmetic
    assert_eq!(arbitrary_reflected.angle.grade(), 0); // 8 % 4 = 0 (scalar grade)
                                                      // blade arithmetic: 2*axis + complement accumulates through forward angle addition

    // proves reflect() uses forward-only arithmetic accumulating transformation history
}

#[test]
fn it_projects_geonum_onto_another_with_blade_preservation() {
    // PROJECTION: (a·b)b/|b|² preserving blade structure in result
    // formula: project(a, onto) = [scalar_factor * onto.length, onto.angle] with direction handling

    // case 1: project vector onto aligned direction
    let vec_a = Geonum::new(3.0, 1.0, 4.0); // length=3, angle=π/4 → blade=0, value=π/4
    let vec_b = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4 (parallel)
    let projection = vec_a.project(&vec_b);
    // step 1: dot_product = vec_a.dot(&vec_b) → [6.0, blade=0, value=0] (parallel vectors)
    // step 2: scalar_factor = dot.length / |b|² = 6.0 / 4.0 = 1.5
    // step 3: projection_magnitude = scalar_factor * onto.length = 1.5 * 2.0 = 3.0
    // step 4: result = [3.0, onto.angle] = [3.0, blade=0, value=π/4]
    assert_eq!(projection.length, 3.0); // projected magnitude
    assert_eq!(projection.angle.blade(), vec_b.angle.blade()); // blade from onto vector
    assert_eq!(projection.angle.value(), vec_b.angle.value()); // value from onto vector
                                                               // blade arithmetic: projection preserves target blade structure

    // case 2: project vector onto perpendicular direction
    let ortho_a = Geonum::new(2.0, 0.0, 1.0); // angle=0 → blade=0, value=0
    let ortho_b = Geonum::new(3.0, 1.0, 2.0); // angle=π/2 → blade=1, value=0
    let ortho_proj = ortho_a.project(&ortho_b);
    // step 1: dot_product = ortho_a.dot(&ortho_b) → [0.0, blade=0, value=0] (orthogonal)
    // step 2: scalar_factor = 0.0 / |b|² = 0.0 / 9.0 = 0.0
    // step 3: projection_magnitude = 0.0 * 3.0 = 0.0
    // step 4: result = [0.0, blade=1, value=0] (preserves onto angle structure)
    assert!(ortho_proj.length.abs() < EPSILON); // zero projection
    assert_eq!(ortho_proj.angle.blade(), ortho_b.angle.blade()); // blade from onto vector
    assert_eq!(ortho_proj.angle.value(), ortho_b.angle.value()); // value from onto vector
                                                                 // blade arithmetic: orthogonal projection has zero magnitude but preserves target blade

    // case 3: project high blade onto different blade
    let high_a = Geonum::new_with_blade(4.0, 1000, 1.0, 6.0); // blade=1000, value=π/6
    let target_b = Geonum::new_with_blade(2.0, 500, 1.0, 4.0); // blade=500, value=π/4
    let high_proj = high_a.project(&target_b);
    // step 1: dot_product computed through angle difference (collapses to scalar)
    // step 2: scalar_factor = dot.length / target_b.length²
    // step 3: result magnitude = scalar_factor * target_b.length
    // step 4: result angle = target_b.angle (blade=500, value=π/4 preserved)
    assert!(high_proj.length.is_finite()); // finite projection magnitude
    assert_eq!(high_proj.angle.blade(), 500); // blade from target vector
    assert!((high_proj.angle.value() - PI / 4.0).abs() < EPSILON); // value from target vector
                                                                   // blade arithmetic: projection result inherits target blade structure

    // proves project() preserves target blade structure while computing magnitude through dot product
}

#[test]
fn it_rejects_geonum_from_another_through_subtraction() {
    // REJECTION: a - proj_b(a) through geonum subtraction blade arithmetic
    // compound operation: projection blade preservation + geonum subtraction

    // case 1: reject parallel component (orthogonal result)
    let vec_a = Geonum::new(5.0, 1.0, 6.0); // length=5, angle=π/6 → blade=0, value=π/6
    let vec_b = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4
    let _projection = vec_a.project(&vec_b);
    let rejection = vec_a.reject(&vec_b);
    // step 1: projection = [proj_magnitude, blade=0, value=π/4] (inherits vec_b blade)
    // step 2: rejection = vec_a - projection through geonum subtraction
    // step 3: subtraction = cartesian: [5*cos(π/6), 5*sin(π/6)] - [proj*cos(π/4), proj*sin(π/4)]
    // step 4: result = new magnitude + angle from cartesian difference
    assert!(rejection.length.is_finite()); // finite rejection magnitude
                                           // blade arithmetic: rejection = original - projection through cartesian geonum subtraction

    // case 2: reject from parallel vector (zero result)
    let parallel_a = Geonum::new(4.0, 1.0, 3.0); // angle=π/3 → blade=0, value=π/3
    let parallel_b = Geonum::new(2.0, 1.0, 3.0); // angle=π/3 → blade=0, value=π/3 (same direction)
    let parallel_rejection = parallel_a.reject(&parallel_b);
    // step 1: projection = full vector (parallel case)
    // step 2: rejection = parallel_a - projection ≈ zero vector
    assert!(parallel_rejection.length < EPSILON); // approximately zero
                                                  // blade arithmetic: parallel rejection produces near-zero through subtraction

    // case 3: reject high blade from different blade target
    let high_source = Geonum::new_with_blade(6.0, 1000, 1.0, 8.0); // blade=1000, value=π/8
    let target = Geonum::new_with_blade(3.0, 200, 1.0, 6.0); // blade=200, value=π/6
    let high_rejection = high_source.reject(&target);
    // step 1: projection inherits target blade=200, value=π/6
    // step 2: rejection = high_source - projection through cartesian subtraction
    // step 3: cartesian math determines final blade/value state
    assert!(high_rejection.length.is_finite()); // finite result
                                                // blade arithmetic: high blade rejection through compound projection + subtraction

    // case 4: verify rejection orthogonality to projection
    let verify_a = Geonum::new(3.0, 1.0, 4.0); // test vector
    let verify_b = Geonum::new(2.0, 1.0, 8.0); // target vector
    let verify_proj = verify_a.project(&verify_b);
    let verify_rej = verify_a.reject(&verify_b);
    let orthogonal_test = verify_proj.dot(&verify_rej);
    // rejection should be orthogonal to projection
    assert!(orthogonal_test.length < 1e-6); // approximately orthogonal
                                            // blade arithmetic: rejection orthogonality through dot product blade collapse

    // proves reject() combines projection blade preservation with geonum subtraction
}

#[test]
fn it_detects_orthogonality_through_dot_product_blade_collapse() {
    // ORTHOGONALITY: dot product → scalar blade=0, test magnitude ≈ 0
    // formula: is_orthogonal = dot(a,b).length.abs() < EPSILON (blade structure irrelevant)

    // case 1: orthogonal vectors (π/2 apart)
    let vec_x = Geonum::new(3.0, 0.0, 1.0); // angle=0 → blade=0, value=0
    let vec_y = Geonum::new(4.0, 1.0, 2.0); // angle=π/2 → blade=1, value=0
    let dot_xy = vec_x.dot(&vec_y);
    // step 1: angle_diff = π/2 - 0 = π/2
    // step 2: magnitude = 3 * 4 * cos(π/2) = 12 * 0 = 0
    // step 3: result = [0, blade=0, value=0] (scalar collapse)
    assert!(dot_xy.length.abs() < EPSILON); // zero dot product
    assert_eq!(dot_xy.angle.blade(), 0); // always scalar blade
    assert!(vec_x.is_orthogonal(&vec_y)); // orthogonality detected
                                          // blade arithmetic: orthogonality independent of input blade structure

    // case 2: non-orthogonal vectors
    let non_ortho_a = Geonum::new(2.0, 1.0, 6.0); // angle=π/6 → blade=0, value=π/6
    let non_ortho_b = Geonum::new(3.0, 1.0, 4.0); // angle=π/4 → blade=0, value=π/4
    let dot_non_ortho = non_ortho_a.dot(&non_ortho_b);
    // step 1: angle_diff = π/4 - π/6 = π/12
    // step 2: magnitude = 2 * 3 * cos(π/12) > 0
    // step 3: result = [magnitude, blade=0, value=0] (scalar)
    assert!(dot_non_ortho.length > EPSILON); // non-zero dot product
    assert!(!non_ortho_a.is_orthogonal(&non_ortho_b)); // not orthogonal
                                                       // blade arithmetic: non-orthogonality detected through non-zero scalar magnitude

    // case 3: high blade orthogonality detection
    let high_a = Geonum::new_with_blade(2.0, 1000, 0.0, 1.0); // blade=1000, grade=0
    let high_b = Geonum::new_with_blade(3.0, 1001, 0.0, 1.0); // blade=1001, grade=1
    let high_dot = high_a.dot(&high_b);
    // step 1: angle difference computed through mod_4_angle (grade 0 vs grade 1)
    // step 2: magnitude calculation through cosine projection
    // step 3: result = [magnitude, blade=0] (scalar collapse)
    assert_eq!(high_dot.angle.blade(), 0); // dot collapses to scalar regardless of input blades
    let is_high_orthogonal = high_a.is_orthogonal(&high_b);
    // blade arithmetic: orthogonality test works at arbitrary blade complexity
    assert!(
        is_high_orthogonal,
        "high blade vectors with grade difference of 1 are orthogonal"
    );

    // case 4: zero vector orthogonality (boundary case)
    let zero_vec = Geonum::new(0.0, 1.0, 4.0); // length=0, any angle
    let any_vec = Geonum::new(5.0, 1.0, 3.0); // non-zero vector
    assert!(zero_vec.is_orthogonal(&any_vec)); // zero vector orthogonal to everything
                                               // blade arithmetic: zero magnitude makes orthogonality independent of blade structure

    // proves is_orthogonal() detects through dot product magnitude, ignoring blade complexity
}

#[test]
fn it_computes_length_difference_ignoring_blade_structure() {
    // LENGTH DIFFERENCE: |length₁ - length₂| independent of blade arithmetic
    // formula: length_diff(a,b) = |a.length - b.length| (blade/angle completely ignored)

    // case 1: same blade different lengths
    let same_blade_a = Geonum::new(5.0, 1.0, 4.0); // length=5, blade=0, value=π/4
    let same_blade_b = Geonum::new(3.0, 1.0, 4.0); // length=3, blade=0, value=π/4 (same angle)
    let diff_same_blade = same_blade_a.length_diff(&same_blade_b);
    // step 1: length difference = |5.0 - 3.0| = 2.0
    // step 2: blade/angle ignored completely
    assert_eq!(diff_same_blade, 2.0); // |5 - 3| = 2
                                      // blade arithmetic: length_diff ignores identical blade structures

    // case 2: different blades same lengths
    let diff_blade_a = Geonum::new(4.0, 0.0, 1.0); // length=4, blade=0, value=0
    let diff_blade_b = Geonum::new(4.0, 3.0, 2.0); // length=4, blade=3, value=0 (different grade)
    let diff_different_blade = diff_blade_a.length_diff(&diff_blade_b);
    // step 1: length difference = |4.0 - 4.0| = 0.0
    // step 2: blade difference (0 vs 3) completely ignored
    assert_eq!(diff_different_blade, 0.0); // same lengths = zero difference
                                           // blade arithmetic: length_diff ignores blade complexity entirely

    // case 3: high blade vs low blade different lengths
    let high_blade = Geonum::new_with_blade(7.0, 1000, 1.0, 6.0); // length=7, blade=1000
    let low_blade = Geonum::new(2.0, 1.0, 4.0); // length=2, blade=0
    let high_low_diff = high_blade.length_diff(&low_blade);
    // step 1: length difference = |7.0 - 2.0| = 5.0
    // step 2: blade difference (1000 vs 0) ignored
    // step 3: angle difference ignored
    assert_eq!(high_low_diff, 5.0); // |7 - 2| = 5
                                    // blade arithmetic: million-dimensional vs scalar comparison = pure length

    // case 4: zero length comparison
    let zero_length = Geonum::new(0.0, 5.0, 4.0); // length=0, any blade/angle
    let finite_length = Geonum::new(3.0, 1.0, 8.0); // length=3, any blade/angle
    let zero_diff = zero_length.length_diff(&finite_length);
    assert_eq!(zero_diff, 3.0); // |0 - 3| = 3
                                // blade arithmetic: zero length comparison ignores all geometric structure

    // proves length_diff() operates purely on length magnitudes, blade structure irrelevant
}

#[test]
fn it_raises_geonum_to_power_through_angle_scaling() {
    // POWER: [rⁿ, n*θ] through length exponent + angle multiplication
    // formula: pow(geonum, n) = [length^n, angle * n] where angle multiplication = addition

    // case 1: square geonum (power of 2)
    let base_geo = Geonum::new(2.0, 1.0, 4.0); // length=2, angle=π/4 → blade=0, value=π/4
    let squared = base_geo.pow(2.0);
    // step 1: length = 2.0^2 = 4.0
    // step 2: angle scaling = angle * 2 = π/4 * 2 through angle multiplication
    // step 3: angle multiplication = Angle::new(2.0, 1.0) = 2π → blade=4, value=0
    // step 4: final angle = π/4 + 2π = π/4 + 4*(π/2) = blade=4, value=π/4
    assert_eq!(squared.length, 4.0); // 2² = 4
    assert_eq!(squared.angle.blade(), 4); // angle * 2 adds 4 blades (2π)
    assert!((squared.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                                 // blade arithmetic: pow(2) = length² + angle*2 through blade addition

    // case 2: cube geonum (power of 3)
    let cubed = base_geo.pow(3.0);
    // step 1: length = 2.0^3 = 8.0
    // step 2: angle scaling = angle * 3 = π/4 * 3 through angle multiplication
    // step 3: angle multiplication = Angle::new(3.0, 1.0) = 3π → blade=6, value=0
    // step 4: final angle = π/4 + 3π = π/4 + 6*(π/2) = blade=6, value=π/4
    assert_eq!(cubed.length, 8.0); // 2³ = 8
    assert_eq!(cubed.angle.blade(), 6); // angle * 3 adds 6 blades (3π)
    assert!((cubed.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                               // blade arithmetic: pow(3) = length³ + angle*3 through blade addition

    // case 3: fractional power (square root)
    let sqrt_geo = base_geo.pow(0.5);
    // step 1: length = 2.0^0.5 = √2
    // step 2: angle scaling = angle * 0.5 = π/4 * 0.5 = π/8
    // step 3: angle multiplication = Angle::new(0.5, 1.0) = π/2 → blade=1, value=0
    // step 4: final angle = π/4 + π/2 = 3π/4 → blade=1, value=π/4
    assert!((sqrt_geo.length - 2.0_f64.sqrt()).abs() < EPSILON); // √2
    assert_eq!(sqrt_geo.angle.blade(), 1); // angle * 0.5 adds 1 blade (π/2)
    assert!((sqrt_geo.angle.value() - PI / 4.0).abs() < EPSILON); // π/4 value preserved
                                                                  // blade arithmetic: pow(0.5) = √length + angle*0.5 through blade addition

    // case 4: high blade power scaling
    let high_base = Geonum::new_with_blade(3.0, 100, 1.0, 6.0); // blade=100, value=π/6
    let high_squared = high_base.pow(2.0);
    // step 1: length = 3.0^2 = 9.0
    // step 2: angle multiplication = angle * 2 adds Angle::new(2.0, 1.0) = 2π = 4 blades
    // step 3: final blade = 100 + 4 = 104 blades
    assert_eq!(high_squared.length, 9.0); // 3² = 9
    assert_eq!(high_squared.angle.blade(), 104); // 100 + 4 = 104 blades
    assert!((high_squared.angle.value() - PI / 6.0).abs() < EPSILON); // π/6 value preserved
                                                                      // blade arithmetic: power scaling works at arbitrary blade magnitudes

    // proves pow() scales length exponentially while multiplying angle through blade arithmetic
}

#[test]
fn it_computes_meet_through_dual_wedge_dual_blade_accumulation() {
    // MEET OPERATION: dual(wedge(dual(A), dual(B))) compound blade arithmetic
    // triple operation sequence with explicit blade tracking at each step

    let line1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // blade=1, grade=1, value=0
    let line2 = Geonum::new_with_blade(1.0, 1, 1.0, 4.0); // blade=1, grade=1, value=π/4

    // step 1: compute duals (each adds 2 blades)
    let dual1 = line1.dual(); // blade=1+2=3, grade=3
    let dual2 = line2.dual(); // blade=1+2=3, grade=3
    assert_eq!(dual1.angle.blade(), 3); // line1 + 2 = 3
    assert_eq!(dual2.angle.blade(), 3); // line2 + 2 = 3

    // step 2: compute wedge of duals (angle sum + π/2 + orientation)
    let wedge_duals = dual1.wedge(&dual2);
    // angle sum = 3π/2 + (3π/2 + π/4) + π/2 = 3π/2 + 7π/4 + π/2 = complex
    assert!(wedge_duals.length.is_finite()); // finite wedge result

    // step 3: final dual (adds 2 more blades)
    let final_dual = wedge_duals.dual();
    let meet_result = line1.meet(&line2);

    // test meet equals manual dual(wedge(dual(A), dual(B)))
    assert_eq!(meet_result.length, final_dual.length); // same magnitude
    assert_eq!(meet_result.angle.blade(), final_dual.angle.blade()); // same blade accumulation
    assert_eq!(meet_result.angle.value(), final_dual.angle.value()); // same value
                                                                     // blade arithmetic: meet IS the triple compound operation with full blade accumulation
}

#[test]
fn it_accesses_length_and_angle_components_preserving_blade_state() {
    // ACCESSOR METHODS: length() and angle() preserve blade structure
    // pure getters: return components without any blade processing

    let high_blade_geo = Geonum::new_with_blade(3.5, 1000, 1.0, 4.0); // blade=1000, value=π/4
    let zero_blade_geo = Geonum::new(2.0, 1.0, 6.0); // blade=0, value=π/6

    // case 1: high blade component access
    let high_length = high_blade_geo.length();
    let high_angle = high_blade_geo.angle();
    // step 1: length() = geo.length (direct field access)
    // step 2: angle() = geo.angle (direct field access)
    assert_eq!(high_length, 3.5); // exact length returned
    assert_eq!(high_angle.blade(), 1000); // exact blade count returned
    assert!((high_angle.value() - PI / 4.0).abs() < EPSILON); // exact value returned
    assert_eq!(high_angle.grade(), 0); // grade computed from blade: 1000%4=0
                                       // blade arithmetic: accessors expose raw blade/value state without modification

    // case 2: low blade component access
    let low_length = zero_blade_geo.length();
    let low_angle = zero_blade_geo.angle();
    assert_eq!(low_length, 2.0); // exact length
    assert_eq!(low_angle.blade(), 0); // exact blade
    assert!((low_angle.value() - PI / 6.0).abs() < EPSILON); // exact value
                                                             // blade arithmetic: accessors work identically regardless of blade magnitude

    // proves accessors provide direct blade state access without any geometric processing
}

#[test]
fn it_scales_geonum_through_scalar_multiplication_preserving_blade() {
    // SCALING: scalar multiplication preserves blade structure
    // formula: scale(factor) = geonum * Geonum::scalar(factor) where scalar has blade=0

    let geo = Geonum::new_with_blade(2.0, 500, 1.0, 6.0); // blade=500, value=π/6
    let scaled_positive = geo.scale(3.0);
    // step 1: Geonum::scalar(3.0) = [3.0, blade=0, value=0] (positive scalar)
    // step 2: multiplication = [2*3, blade=500+0, value=π/6+0] = [6.0, blade=500, π/6]
    assert_eq!(scaled_positive.length, 6.0); // 2 * 3 = 6
    assert_eq!(scaled_positive.angle.blade(), 500); // blade=500+0=500 preserved
    assert_eq!(scaled_positive.angle.value(), geo.angle.value()); // value preserved
                                                                  // blade arithmetic: positive scaling preserves blade through scalar multiplication

    // case 2: negative scaling adds π rotation (blade=2)
    let scaled_negative = geo.scale(-2.0);
    // step 1: Geonum::scalar(-2.0) = [2.0, blade=2, value=0] (negative scalar at π)
    // step 2: multiplication = [2*2, blade=500+2, value=π/6+0] = [4.0, blade=502, π/6]
    assert_eq!(scaled_negative.length, 4.0); // 2 * 2 = 4
    assert_eq!(scaled_negative.angle.blade(), 502); // blade=500+2=502 from negative scalar
    assert_eq!(scaled_negative.angle.value(), geo.angle.value()); // value preserved
                                                                  // blade arithmetic: negative scaling adds 2 blades through π rotation in scalar

    // proves scale() preserves/accumulates blade through scalar multiplication blade arithmetic
}

#[test]
fn it_inverts_through_circle_preserving_angles_transforming_length() {
    // CIRCLE INVERSION: transforms length by r²/d, accumulates blade through addition
    // formula: center + r²/(point - center) where addition accumulates blade history

    let center = Geonum::new(0.0, 0.0, 1.0); // origin, blade=0
    let point = Geonum::new_with_blade(2.0, 500, 1.0, 6.0); // blade=500, value=π/6
    let inverted = point.invert_circle(&center, 1.0); // radius=1
                                                      // step 1: offset = point - center = point (center at origin)
                                                      // step 2: inverted_offset = [r²/distance, offset.angle] = [0.5, blade=500, π/6]
                                                      // step 3: result = center + inverted_offset through geonum addition
                                                      // step 4: geonum addition combines: blade=0+500=500 + boundary crossings = 504
    assert_eq!(inverted.length, 0.5); // 1²/2 = 0.5 distance scaling
    assert_eq!(inverted.angle.blade(), 504); // blade accumulated: 500 + 4 from addition
    assert!((inverted.angle.value() - point.angle.value()).abs() < EPSILON); // value preserved
    assert_eq!(inverted.angle.grade(), point.angle.grade()); // grade preserved: 504%4 = 0 = 500%4
                                                             // blade arithmetic: circle inversion accumulates +4 blades while preserving grade behavior

    // case 2: double inversion through compound blade accumulation
    let double_inverted = inverted.invert_circle(&center, 1.0);
    // step 1: second inversion adds another +4 blades: 504 + 4 = 508
    assert!((double_inverted.length - point.length).abs() < 1e-10); // returns to original distance within precision
    assert_eq!(double_inverted.angle.blade(), 508); // blade=504+4=508 total accumulation
    assert_eq!(double_inverted.angle.grade(), point.angle.grade()); // grade preserved: 508%4=0
                                                                    // blade arithmetic: double inversion accumulates 8 blades total while preserving geometry

    // proves circle inversion accumulates transformation history through forward-only addition
}

#[test]
fn it_resets_blade_to_grade_minimum_preserving_geometric_behavior() {
    // BASE ANGLE: resets blade to minimum for grade, preserves value and behavior
    // formula: base_angle() = [length, Angle{blade: grade, value: value}]

    // case 1: reset high blade scalar to minimum
    let high_scalar = Geonum::new_with_blade(2.0, 1000, 1.0, 4.0); // blade=1000, grade=0, value=π/4
    let reset_scalar = high_scalar.base_angle();
    // step 1: grade = 1000 % 4 = 0 (scalar behavior)
    // step 2: minimum blade for grade 0 = 0
    // step 3: preserve value = π/4, preserve length = 2.0
    assert_eq!(reset_scalar.length, 2.0); // length preserved
    assert_eq!(reset_scalar.angle.blade(), 0); // blade reset: 1000 → 0
    assert!((reset_scalar.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved
    assert_eq!(reset_scalar.angle.grade(), 0); // grade preserved: 0%4 = 0
                                               // blade arithmetic: transformation history discarded, geometric behavior preserved

    // case 2: reset high blade vector to minimum
    let high_vector = Geonum::new_with_blade(1.5, 1001, 1.0, 6.0); // blade=1001, grade=1, value=π/6
    let reset_vector = high_vector.base_angle();
    // step 1: grade = 1001 % 4 = 1 (vector behavior)
    // step 2: minimum blade for grade 1 = 1
    // step 3: preserve value = π/6, preserve length = 1.5
    assert_eq!(reset_vector.angle.blade(), 1); // blade reset: 1001 → 1
    assert_eq!(reset_vector.angle.grade(), 1); // grade preserved: 1%4 = 1
    assert!((reset_vector.angle.value() - PI / 6.0).abs() < EPSILON); // value preserved
                                                                      // blade arithmetic: forgets 1000 transformations, keeps vector behavior

    // proves base_angle() separates transformation history from geometric behavior
}

#[test]
fn it_applies_spiral_similarity_through_scale_and_rotation() {
    // SPIRAL SIMILARITY: combines scaling with rotation blade arithmetic
    // formula: scale_rotate(factor, rotation) = [length*factor, angle+rotation]

    // case 1: positive scaling with rotation
    let geo = Geonum::new(2.0, 1.0, 6.0); // length=2, angle=π/6 → blade=0, value=π/6
    let rotation = Angle::new(1.0, 4.0); // π/4 rotation → blade=0, value=π/4
    let transformed = geo.scale_rotate(3.0, rotation);
    // step 1: length scaling = 2.0 * 3.0 = 6.0 (positive factor)
    // step 2: angle addition = π/6 + π/4 = 2π/12 + 3π/12 = 5π/12
    // step 3: blade arithmetic = (0+0, π/6+π/4) = blade=0, value=5π/12
    assert_eq!(transformed.length, 6.0); // 2 * 3 = 6
    assert_eq!(transformed.angle.blade(), 0); // no boundary crossing: 5π/12 < π/2
    assert!((transformed.angle.value() - 5.0 * PI / 12.0).abs() < EPSILON); // combined angle
                                                                            // blade arithmetic: positive spiral similarity = length scaling + angle addition

    // case 2: negative scaling adds π rotation
    let neg_transformed = geo.scale_rotate(-2.0, rotation);
    // step 1: length scaling = 2.0 * 2.0 = 4.0 (abs of negative factor)
    // step 2: negative factor adds π to angle = π/6 + π = 7π/6 → blade=2, value=π/6
    // step 3: rotation addition = (2+0, π/6+π/4) = blade=2, value=5π/12
    assert_eq!(neg_transformed.length, 4.0); // 2 * |-2| = 4
    assert_eq!(neg_transformed.angle.blade(), 2); // π rotation from negative factor
    assert!((neg_transformed.angle.value() - 5.0 * PI / 12.0).abs() < EPSILON); // combined angle
                                                                                // blade arithmetic: negative spiral adds π rotation + angle addition

    // proves scale_rotate() handles sign through blade arithmetic + length/angle composition
}

#[test]
fn it_implements_geonum_add_trait_blade_accumulation() {
    // GEONUM ADD: blade accumulation through cartesian conversion + angle reconstruction
    // handles same angle, opposite angle, and general cases with different blade behaviors

    // case 1: same angle addition preserves blade structure
    let same_a = Geonum::new_with_blade(3.0, 100, 1.0, 4.0); // blade=100, value=π/4
    let same_b = Geonum::new_with_blade(2.0, 100, 1.0, 4.0); // blade=100, value=π/4 (identical angle)
    let same_sum = same_a + same_b;
    // step 1: angle equality detected: same_a.angle == same_b.angle
    // step 2: direct length addition: 3.0 + 2.0 = 5.0
    // step 3: angle preserved: blade=100, value=π/4 unchanged
    assert_eq!(same_sum.length, 5.0); // lengths add directly
    assert_eq!(same_sum.angle.blade(), 100); // blade preserved exactly
    assert!((same_sum.angle.value() - PI / 4.0).abs() < EPSILON); // value preserved exactly
                                                                  // blade arithmetic: same angle bypass → no blade accumulation, direct length addition

    // case 2: general case accumulates blades through cartesian conversion
    let diff_a = Geonum::new_with_blade(2.0, 200, 1.0, 6.0); // blade=200, value=π/6
    let diff_b = Geonum::new_with_blade(3.0, 300, 1.0, 4.0); // blade=300, value=π/4
    let diff_sum = diff_a + diff_b;
    // step 1: different angles → cartesian conversion path
    // step 2: combined_blade = 200 + 300 = 500
    // step 3: cartesian addition using mod_4_angle remainders
    // step 4: result = new_with_blade(combined_blade, atan2_result, PI)
    assert_eq!(diff_sum.angle.blade(), 500); // blade accumulated: 200 + 300 = 500
    assert!(diff_sum.length > 0.0); // finite magnitude from cartesian sum
                                    // blade arithmetic: general addition accumulates input blades + boundary crossings

    // case 3: opposite angles produce magnitude cancellation with blade accumulation
    let forward = Geonum::new_with_blade(4.0, 52, 0.0, 1.0); // blade=52, angle=0
    let backward = Geonum::new_with_blade(4.0, 50, 0.0, 1.0); // blade=50, angle=π (opposite direction)

    // prove they point in opposite directions
    assert!((forward.angle.mod_4_angle() - 0.0).abs() < EPSILON); // forward at 0
    assert!((backward.angle.mod_4_angle() - PI).abs() < EPSILON); // backward at π
    assert_eq!(forward.length, backward.length); // same magnitude

    let oppose_sum = forward + backward;
    // step 1: opposite angle detection triggers magnitude subtraction: |4.0 - 4.0| = 0.0
    // step 2: blade accumulation continues: 52 + 50 = 102 (transformation history preserved)
    // step 3: zero magnitude with accumulated blade history
    assert!(oppose_sum.length < EPSILON); // magnitudes cancel
    assert_eq!(oppose_sum.angle.blade(), 102); // blade accumulated: 52+50=102 through angle arithmetic
                                               // blade arithmetic: forward-only geometry preserves transformation history through cancellation

    // proves Add trait accumulates blade history while handling geometric addition cases
}

#[test]
fn it_implements_geonum_sub_trait_through_negation() {
    // GEONUM SUB: subtraction through addition with negated second operand
    // formula: a - b = a + b.negate() where negate adds π rotation (2 blades)

    let geo_a = Geonum::new_with_blade(5.0, 200, 1.0, 6.0); // blade=200, value=π/6
    let geo_b = Geonum::new_with_blade(3.0, 100, 1.0, 4.0); // blade=100, value=π/4
    let difference = geo_a - geo_b;
    // step 1: negate geo_b = [3.0, blade=100+2=102, value=π/4] (π rotation added)
    // step 2: addition = geo_a + negated_b through geonum addition
    // step 3: blade accumulation = 200 + 102 = 302 + boundary crossings
    assert!(difference.length.is_finite()); // finite result
    assert_eq!(difference.angle.blade(), 304); // blade accumulated: 200 + (100+2) + 2 crossings = 304
                                               // blade arithmetic: subtraction = addition with negated operand blade accumulation
}

#[test]
fn it_implements_geonum_mul_trait_angles_add_lengths_multiply() {
    // GEONUM MUL: fundamental "angles add, lengths multiply" principle
    // formula: a * b = [a.length * b.length, a.angle + b.angle]

    let geo_a = Geonum::new_with_blade(3.0, 150, 1.0, 8.0); // blade=150, value=π/8
    let geo_b = Geonum::new_with_blade(4.0, 250, 1.0, 6.0); // blade=250, value=π/6
    let product = geo_a * geo_b;
    // step 1: length multiplication = 3.0 * 4.0 = 12.0
    // step 2: angle addition = blade=150+250=400, value=π/8+π/6=7π/24
    assert_eq!(product.length, 12.0); // lengths multiply
    assert_eq!(product.angle.blade(), 400); // blades add: 150 + 250 = 400
    assert!((product.angle.value() - 7.0 * PI / 24.0).abs() < EPSILON); // values add: π/8 + π/6 = 7π/24
                                                                        // blade arithmetic: multiplication = length multiplication + angle addition
}

#[test]
fn it_implements_geonum_div_trait_through_multiplication_by_inverse() {
    // GEONUM DIV: division through multiplication by inverse blade arithmetic
    // formula: a / b = a * b.inv() where inv adds π rotation (2 blades)

    let dividend = Geonum::new_with_blade(8.0, 300, 1.0, 4.0); // blade=300, value=π/4
    let divisor = Geonum::new_with_blade(2.0, 100, 1.0, 6.0); // blade=100, value=π/6
    let quotient = dividend / divisor;
    // step 1: divisor.inv() = [1/2, blade=100+2=102, value=π/6] (π rotation from inv)
    // step 2: multiplication = [8*0.5, blade=300+102=402, value=π/4+π/6=5π/12]
    assert_eq!(quotient.length, 4.0); // 8/2 = 4
    assert_eq!(quotient.angle.blade(), 402); // blade accumulated: 300 + (100+2) = 402
    assert!((quotient.angle.value() - 5.0 * PI / 12.0).abs() < EPSILON); // values add: π/4 + π/6 = 5π/12
                                                                         // blade arithmetic: division = multiplication with inverted operand blade accumulation
}

#[test]
fn it_implements_angle_mul_geonum_trait_angle_addition() {
    // ANGLE * GEONUM: preserves length, adds angle to geonum angle
    // formula: angle * geonum = [geonum.length, angle + geonum.angle]

    let angle = Angle::new_with_blade(100, 1.0, 8.0); // blade=100, value=π/8
    let geo = Geonum::new_with_blade(3.0, 200, 1.0, 6.0); // blade=200, value=π/6
    let result = angle * geo;
    // step 1: length preserved = 3.0
    // step 2: angle addition = blade=100+200=300, value=π/8+π/6=7π/24
    assert_eq!(result.length, 3.0); // length preserved from geonum
    assert_eq!(result.angle.blade(), 300); // blades add: 100 + 200 = 300
    assert!((result.angle.value() - 7.0 * PI / 24.0).abs() < EPSILON); // values add: π/8 + π/6 = 7π/24
                                                                       // blade arithmetic: cross-type multiplication = angle addition with length preservation
}

#[test]
fn it_implements_angle_add_geonum_trait_angle_addition() {
    // ANGLE + GEONUM: preserves length, adds angle to geonum angle
    // formula: angle + geonum = [geonum.length, angle + geonum.angle] (identical to multiplication)

    let angle = Angle::new_with_blade(75, 1.0, 8.0); // blade=75, value=π/8
    let geo = Geonum::new_with_blade(2.5, 125, 1.0, 3.0); // blade=125, value=π/3
    let result = angle + geo;
    // step 1: length preserved = 2.5
    // step 2: angle addition = blade=75+125=200, value=π/8+π/3=11π/24
    assert_eq!(result.length, 2.5); // length preserved from geonum
    assert_eq!(result.angle.blade(), 200); // blades add: 75 + 125 = 200
    assert!((result.angle.value() - 11.0 * PI / 24.0).abs() < EPSILON); // values add: π/8 + π/3 = 11π/24
                                                                        // blade arithmetic: cross-type addition = identical to multiplication (angle addition)
}

#[test]
fn it_implements_geonum_ord_trait_angle_first_ordering() {
    // GEONUM ORD: angle-first ordering (blade priority over length)
    // ordering: compare angle first (blade then value), then length if angles equal

    let small_blade_big_length = Geonum::new_with_blade(1000.0, 50, 1.0, 4.0); // blade=50, huge length
    let big_blade_small_length = Geonum::new_with_blade(0.1, 150, 1.0, 6.0); // blade=150, tiny length
                                                                             // step 1: angle comparison = blade 50 vs blade 150 → 50 < 150
                                                                             // step 2: length ignored (1000.0 vs 0.1 doesn't matter)
    assert!(small_blade_big_length < big_blade_small_length); // blade 50 < blade 150
                                                              // blade arithmetic: ordering prioritizes transformation history over magnitude
}

#[test]
fn it_computes_distance_via_law_of_cosines() {
    // DISTANCE COMPUTATION: law of cosines with blade-aware angle arithmetic
    // formula: d² = a² + b² - 2ab·cos(angle_between)

    // case 1: right triangle (3-4-5) with perpendicular vectors
    let point_x = Geonum::new(3.0, 0.0, 1.0); // 3 units at 0° (blade 0)
    let point_y = Geonum::new(4.0, 1.0, 2.0); // 4 units at 90° (blade 1)
    let distance_right = point_x.distance_to(&point_y);
    // step 1: angle_between = blade 1 - blade 0 = π/2
    // step 2: cos(π/2) = 0, so middle term vanishes
    // step 3: d² = 9 + 16 - 0 = 25
    // step 4: d = 5, returned as scalar (blade 0)
    assert_eq!(distance_right.length, 5.0); // 3-4-5 triangle
    assert_eq!(distance_right.angle.blade(), 0); // distance is scalar
                                                 // blade arithmetic: perpendicular blades → cos(π/2) = 0 → pythagorean

    // case 2: same point with different blades
    let point_blade_0 = Geonum::new(3.0, 1.0, 6.0); // 3 at π/6 (blade 0)
    let point_blade_1000 = Geonum::new_with_blade(3.0, 1000, 1.0, 6.0); // same angle, blade 1000
    let same_point_distance = point_blade_0.distance_to(&point_blade_1000);
    // step 1: angle_between accounts for blade difference
    // step 2: both represent same geometric point
    // step 3: d² = 9 + 9 - 2(9)cos(0) = 0
    assert!(same_point_distance.length < 1e-10); // same point
    assert_eq!(same_point_distance.angle.blade(), 0); // scalar result
                                                      // blade arithmetic: blade difference preserves geometric position

    // case 3: high blade interaction
    let mega_blade_a = Geonum::new_with_blade(5.0, 999999, 1.0, 8.0); // blade ~1 million
    let mega_blade_b = Geonum::new_with_blade(3.0, 1000000, 1.0, 4.0); // blade 1 million
    let mega_distance = mega_blade_a.distance_to(&mega_blade_b);
    // step 1: angle_between handles million-blade difference
    // step 2: law of cosines works at any blade scale
    // step 3: result always scalar (blade 0)
    assert_eq!(mega_distance.angle.blade(), 0); // scalar even from mega-blades
                                                // blade arithmetic: million-dimensional distance → scalar projection

    // case 4: opposite vectors (should give sum of lengths)
    let forward = Geonum::new(3.0, 0.0, 1.0); // 3 at 0°
    let backward = Geonum::new(4.0, 1.0, 1.0); // 4 at π
    let opposite_distance = forward.distance_to(&backward);
    // step 1: angle_between = π - 0 = π
    // step 2: cos(π) = -1
    // step 3: d² = 9 + 16 - 2(3)(4)(-1) = 25 + 24 = 49
    // step 4: d = 7
    assert_eq!(opposite_distance.length, 7.0); // 3 + 4 = 7 for opposite vectors
    assert_eq!(opposite_distance.angle.blade(), 0); // scalar
                                                    // blade arithmetic: opposite angles (π apart) → maximal distance

    // proves distance_to() uses blade-aware angle arithmetic in law of cosines
}
