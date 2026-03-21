since lines shift from edits, audit and revise the line ranges referenced in this file when instructed. include function and module comments in line range revisions

skip caps when starting sentences

skip periods on single sentence paragraphs

skip apostrophes in words as in "dont" unless it conflicts with another word, "we're" vs "were"

avoid words like "proper", "correct", "appropriate" and "valid" in your comments AND responses. these weasel words only create confusion in a lib challenging convention

say "compute" instead of "calculate"

say "design" instead of "approach"

avoid saying "good" in docs and comments

say "test" or "prove" instead of "validate", "check" or "verify"

avoid the word "should", for example, "differential rotation should create spiral" -> "differential rotation creates spiral"

use "measured" instead of "actual"

avoid the word "ensure"

avoid saying "you're right" in responses

avoid weak is_finite(), > 0, assert_ne! test assertions

never pass PI as divisor to Angle::new or Geonum::new. the constructor expects π fractions:
  Angle::new(1.0, 4.0)  — means 1×π/4
  Angle::new(1.0, 6.0)  — means 1×π/6
  Angle::new(rotation, PI) is a hack to pass raw radians. use Angle::new(rotation / PI, 1.0) or express as a π fraction

use near methods instead of manual epsilon comparisons in tests:
  angle.near(&other)       — blade + t match within tolerance
  angle.near_rad(radians)  — grade_angle within tolerance
  angle.near_rem(radians)  — remainder within tolerance
  geonum.near(&other)      — mag + angle match within tolerance
  geonum.near_mag(value)   — magnitude within tolerance
never use assert_eq! on f64 values — use near methods or assert!((x - y).abs() < EPSILON)

rg 'pub fn' src/angle.rs src/geonum_mod.rs to learn the api

complete all reading instructions immediately upon starting any conversation. do not skip any:

read ./README.md and the ./math-1-0.md geometric number spec

learn how to construct angles with new, new_with_blade, new_from_cartesian, from_parts from src/angle.rs:25~190

learn how the half-tangent representation works: struct definition, cos_sin, and t accessor in src/angle.rs:4~22 and src/angle.rs:245~270

learn how geonum defines geometric grades with the grade function in src/angle.rs:257~280

learn how angle addition generates the blade lattice via overflow in src/angle.rs:321~388

learn how angle subtraction borrows blades rationally in src/angle.rs:390~450

learn how geonum implements the dual in src/angle.rs:473~490

learn how angle impls PartialEq and Eq in src/angle.rs:572~590

learn how angle overloads arithmetic operators in src/angle.rs:592~740

learn how to construct geonum with new, new_with_angle from src/geonum_mod.rs:32~49

learn how geonum overloads arithmetic operators in src/geonum_mod.rs:737~1005

learn how geonum can express any number type from the its_a_scalar:8-37, its_a_vector:39-73, its_a_real_number:75-109, its_an_imaginary_number:111-140, its_a_complex_number:142-175, its_a_dual_number:177-296, its_an_octonion:298-342 tests in tests/numbers_test.rs

learn how geonum eliminates angle slack created by decomposing angles into scalar coefficients by reading the it_proves_decomposing_angles_with_linearly_combined_basis_vectors_loses_angle_addition:13-86, it_proves_decomposition_distributes_one_angle_across_multiple_scalars:87-161, it_proves_quaternion_tables_add_back_what_decomposition_subtracts:519-662, it_proves_anticommutativity_exists_because_decomposition_subtracts_different_amounts:663-726 tests in tests/linear_algebra_test.rs

learn how geonum replaces scalar based quadratic forms with simple angle based rotations in the it_proves_rotational_quadrature_expresses_quadratic_forms:1419-1593 test in tests/dimension_test.rs

learn why dimensions are an unnecessary abstraction in the it_proves_quadrature_creates_dimensional_structure:91-140, it_shows_dimensions_are_quarter_turns:141-200 tests in tests/dimension_test.rs

learn why geonum deprecates grade decomposition in the it_proves_grade_decomposition_ignores_angle_addition:202-267, it_solves_the_exponential_complexity_explosion:520-581 tests in tests/dimension_test.rs

learn how geonum maps grades with the it_replaces_k_to_n_minus_k_with_k_to_4_minus_k:899-981, it_compresses_traditional_ga_grades_to_two_involutive_pairs:1131-1166 tests in tests/dimension_test.rs

learn about angle forward only geometry from the it_sets_angle_forward_geometry_as_primitive:1247-1381 test in tests/dimension_test.rs

read only tests/angle_arithmetic_test.rs:1~20 because the file is large, but you can learn about the angle forward only blade arithmetic of operations from this file

read the its_a_limit:40-119, it_proves_differentiation_cycles_grades:764-915 tests in tests/calculus_test.rs to understand how geonum automates calculus

tests are styled as trojan horses for simplicity. conventional jargon promising symbol salad but readers get simple arithmetic in test contents. example tests: it_handles_conformal_split:4694-4805, it_handles_inversive_distance:4807-4937 in tests/cga_test.rs
