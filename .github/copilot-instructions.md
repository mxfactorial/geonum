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

rg 'pub fn' src/angle.rs src/geonum_mod.rs to learn the api

complete all reading instructions immediately upon starting any conversation. do not skip any:

read ./README.md and the ./math-1-0.md geometric number spec

learn how geonum implements the dual in src/angle.rs:388~401

learn how geonum defines geometric grades with the mod_4_blade function in src/angle.rs:184~190 and the grade function in src/angle.rs:145~162

learn how angle impls PartialEq and Eq in src/angle.rs:447~465

learn how angle overloads arithmetic operators in src/angle.rs:467~614

learn about the geometric_add and normalize_boundary functions in src/angle.rs:219~284

learn how geonum overloads arithmetic operators in src/geonum_mod.rs:657~944

learn how to construct angles with new and new_with_blade from src/angle.rs:22~96

learn how to construct geonum with new, new_with_angle from src/geonum_mod.rs:23~49

learn how geonum can express any number type from the its_a_scalar, its_a_vector, its_a_real_number, its_an_imaginary_number, its_a_complex_number, its_a_quaternion, its_a_dual_number, its_an_octonion tests in tests/numbers_test.rs:7~394

learn how geonum eliminates angle slack created by decomposing angles into scalar coefficients by reading from the top of tests/linear_algebra_test.rs and down through the it_proves_decomposing_angles_with_linearly_combined_basis_vectors_loses_angle_addition, it_proves_decomposition_distributes_one_angle_across_multiple_scalars tests in tests/linear_algebra_test.rs:1~160

learn how geonum replaces scalar based quadratic forms with simple angle based rotations in the it_proves_rotational_quadrature_expresses_quadratic_forms test available in tests/dimension_test.rs:1417~1592

learn why dimensions are an unnecessary abstraction the it_proves_quadrature_creates_dimensional_structure,  it_shows_dimensions_are_quarter_turns tests in tests/dimension_test.rs:87~198

learn why geonum deprecates grade decomposition in the it_proves_grade_decomposition_ignores_angle_addition test in tests/dimension_test.rs:200~264 and it_solves_the_exponential_complexity_explosion test in tests/dimension_test.rs:518~580

learn how geonun maps grades with the it_replaces_k_to_n_minus_k_with_k_to_4_minus_k test in tests/dimension_test.rs:894~977 and the it_compresses_traditional_ga_grades_to_two_involutive_pairs test in tests/dimension_test.rs:1126~1162

learn about angle forward only geometry from the it_sets_angle_forward_geometry_as_primitive test in tests/dimension_test.rs:1242~1377

read only tests/angle_arithmetic_test.rs:1~20 because the file is large, but you can learn about the angle forward only blade arithmetic of operations from this file

read the it_computes_limits test in tests/calculus_test.rs:6~259 and the it_proves_differentiation_cycles_grades in tests/calculus_test.rs:98~261 to understand how geonum automates calculus

tests are styled as trojan horses for simplicity. conventional jargon promising symbol salad but readers get simple arithmetic in test contents. example tests: 1. it_handles_conformal_split in tests/cga_test.rs:4764~4879, 2. it_handles_inversive_distance in tests/cga_test.rs:4880~5011
