# changelog

## 0.12.0 (2026-03-31)

### fixed
- **BREAKING**: pow() now scales the total angle by n instead of adding nπ — matches repeated multiplication for all n

### added
- Mul<f64> for Angle and &Angle: scalar multiplication of angles
- calculus_test.rs: power rule as angle readout, factorials from angle descent, limits as lossy projections, fundamental theorem as geometric interference, gradient, laplacian, line/surface/volume integrals
- taylor_series_test.rs: taylor coefficients as geometric normalizations, e^x as uniform angle contribution, sin/cos as grade-filtered projections, euler's formula as grade decomposition, convergence as angle descent dominance
- algebra_test.rs: fundamental theorem of algebra via winding numbers — degree = wraps, roots = unwindings, polynomial evaluation on circles, roots of unity as generalized Q lattice

### changed
- replaced old calculus_test.rs (24 tests) with power-rule-anchored suite (23 tests)

## 0.11.0 (2026-03-20)

### breaking

- Angle internal representation changed from radians to stereographic projection ratio `t = tan(θ/2)`
- `rem()` now derived from `t` via `2.0 * t.atan()` — values match within f64 precision but are no longer stored directly
- `normalize_boundaries()` removed — boundary logic is algebraic in the tangent sum formula
- `Display` for Angle now shows `t` instead of `rem`

### fixed
- pow() now scales the total angle by n instead of adding nπ — matches repeated multiplication for all n

### added

- `Angle::t()` — projection ratio between adjacent π/2 blades
- `Angle::from_parts(blade, t)` — direct construction from blade and projection ratio
- `Angle::cos_sin()` — rational cos/sin: `cos = (1-t²)/(1+t²)`, `sin = 2t/(1+t²)`. no trig calls
- `Angle::near(&other)` — floating point comparison within tolerance
- `Angle::near_rad(radians)` — grade_angle comparison within tolerance
- `Angle::near_rem(radians)` — remainder comparison within tolerance
- `Geonum::near(&other)` — magnitude + angle comparison within tolerance
- `Geonum::near_mag(value)` — magnitude comparison within tolerance
- Mul<f64> for Angle and &Angle: scalar multiplication of angles
- calculus_test.rs: power rule as angle readout, factorials from angle descent, limits as lossy projections, fundamental theorem as geometric interference, gradient, laplacian, line/surface/volume integrals
- taylor_series_test.rs: taylor coefficients as geometric normalizations, e^x as uniform angle contribution, sin/cos as grade-filtered projections, euler's formula as grade decomposition, convergence as angle descent dominance
- algebra_test.rs: fundamental theorem of algebra via winding numbers — degree = wraps, roots = unwindings, polynomial evaluation on circles, roots of unity as generalized Q lattice

### changed

- `Angle::new()` converts π fractions to `t` internally — one `tan()` call at construction
- `Angle::new_from_cartesian()` uses `t = opp/(hyp + adj)` — one sqrt, no atan2
- `Angle::geometric_add()` uses tangent sum formula with rational boundary correction `(T-1)/(T+1)`
- `Angle::geometric_sub()` uses tangent difference formula with rational borrow `(1-|R|)/(1+|R|)`
- `Angle::dual()`, `conjugate()`, `negate()` simplified to blade arithmetic
- `Angle::grade_angle()` derives radians from `t` via `atan()`
- `Angle::project()` uses `cos_sin()` instead of `grade_angle().cos()`
- `Geonum::dot()`, `wedge()`, `cos()`, `sin()`, `distance_to()`, `project_to_angle()` use `cos_sin()`
- `Geonum::geo()` computes single `cos_sin()` for both dot and wedge
- `Geonum` addition uses rational projection pipeline: cos_sin (0 sqrts) → sum → magnitude (1 sqrt) → cartesian recovery (0 sqrts)
- replaced old calculus_test.rs (24 tests) with power-rule-anchored suite (23 tests)
- updated angle_arithmetic_test, numbers_test, geonum_mod unit test pow expectations to match corrected angle scaling

### performance

| operation | 0.10.5 | 0.11.0 | speedup |
|---|---|---|---|
| addition | 68.8 ns | 13.9 ns | 5.0× |
| cos | 11.6 ns | 3.5 ns | 3.3× |
| from_cartesian | 24.4 ns | 3.6 ns | 6.7× |
| dot product | 11.0 ns | 8.6 ns | 1.3× |
| wedge product | 12.6 ns | 10.3 ns | 1.2× |
| geometric product | 25.7 ns | 18 ns | 1.4× |
| projection | 11.5 ns | 8.6 ns | 1.3× |
| distance | 21.6 ns | 17.6 ns | 1.2× |
| differentiate | 4.5 ns | 3.4 ns | 1.3× |

## 0.10.5 (2026-03-17)

### added
- ml_attention_test.rs: 22 tests proving attention as interference — dot is forward, wedge is backward, multi-head as angle offsets, positional encoding as rotation, wedge-driven learning, full forward pass with routing and gating
- ml_training_test.rs: 6 tests proving next-token prediction without training — observed step between tokens IS the rotation, board clusters steps by wedge independence, generalizes to unseen sequences, sequence product as context, zero epochs zero gradients

## 0.10.4 (2026-03-16)

### added
- rendering test suite proving polar sweep screen coverage, scanner rotation, perspective projection, ray casting, intersection, reflection, depth ordering, and conformal split through a single triangle pipeline
- rendering benchmark in criterion bench suite comparing per-pixel rotation cost across 2D, 10D, and 100D
- dimension independence test in ML suite proving forward pass, activation, perceptron update, and dot product produce identical results at blade 1 and blade 1_000_000 with blade accumulation traced through even and odd offsets

### changed
- replaced PI divisor pattern with π-fraction constructors in ML test suite
- moved agent instructions from .github/copilot-instructions.md to .agents/onboard.md
- readme example rewritten to show projection chain drawing points and lines from one angle

## 0.10.3 (2026-03-09)

### changed
- reboot chemistry from geometry with wave sum

## 0.10.2 (2026-03-05)

### added
- chemistry test suite

## 0.10.1 (2025-01-31)

### added
- extend qm test suite

## 0.10.0 (2025-12-10)

### changed
- renamed Geonum length struct field to mag (magnitude)
- renamed Angle value struct field to rem (remainder)

## 0.9.3 (2025-11-14)

### added
- arithmetic test suite

### changed
- strengthen calculus test suite

## 0.9.2 (2025-11-08)

### changed
- switch quaternion test coverage from number to linear algebra test suite

## 0.9.1 (2025-10-13)

### added
- optimization test suite proving O(1) angle arithmetic vs O(n³) lagrange multipliers

### changed
- reference individual test names and line ranges in readme and agent instructions

## 0.9.0 (2025-09-19)

### removed
- **BREAKING**: removed `pub fn to_cartesian()` crutch function from `Geonum`
  - function encouraged escaping the geometric domain to raw coordinates
  - all internal usage replaced with inline trigonometry where needed
  - tests updated to use geometric operations (`adj()`, `opp()`) or inline projection

  ### changed
  - `mod_4_blade()` Geonum function changed to `grade_angle()`

## 0.8.1 (2025-09-15)

### added
- angle encoding trigonometric operations (sin, cos, tan) to geonum module returning Geonum instead of raw f64
- addition_test.rs and trigonometry_test.rs comprehensive test coverage

### changed
- switched to standard library trigonometric calls
- moved trigonometric functions from src/angle.rs to src/geonum_mod.rs with angle encoding

### removed
- raw float trigonometric functions from Angle struct

## 0.8.0 (2025-09-13)

### added
- src/geocollection.rs with GeoCollection struct for domain-specific geometric object collections
- Angle::project() method for angle projection operations
- Angle + Geonum operator overloading for direct angle addition to geometric numbers
- angle_arithmetic_test.rs comprehensive angle blade arithmetic test suite
- linear_algebra_test.rs proving geonum eliminates angle slack from scalar decomposition
- mechanics_test.rs classical mechanics without coordinate systems
- geocollection_test.rs test suite for collection operations

### removed
- src/multivector.rs replaced with GeoCollection interface
- Multivector type from lib.rs exports

### changed
- replaced Multivector usage throughout codebase with GeoCollection
- updated all test files to use GeoCollection instead of Multivector
- strengthened test assertions replacing weak is_finite() checks with precise value tests

## 0.7.1 (2025-08-13)

### added
- projective and conformal geometric algebra test suites
- dual and forward only geometry tests in dimensions test suite

### changed
- dual as pi rotation setting involutive grade map
- gravity as angle correlation in astrophysics test suite

## 0.7.0 (2025-07-18)

### added
- Angle struct that maintains angle-blade invariant
- operator overloading for all ownership patterns on Angle and Geonum

### changed
- Geonum now uses Angle struct instead of raw f64 angle and usize blade fields
- geo() geometric product now returns unified Geonum instead of (f64, Geonum) tuple
- timing assertions in economics and finance tests increased for CI compatibility

## 0.6.11 (2025-07-17)

### added
- project_to_dimension method in src/geonum_mod.rs:440~452 enabling on-demand dimensional queries
- proved dimensions are computed via trigonometry not predefined spaces

### removed
- src/dimensions.rs module 
- Dimensions type from lib.rs exports
- coordinate scaffolding pattern across 34+ usage sites

### changed  
- replaced all Dimensions::new() calls with direct geometric number creation
- updated 9 test files to use Multivector::create_dimension() constructor
- updated benches/geonum_benchmarks.rs with 12 replacements
- strengthened dimension_test.rs assertions from weak is_finite() checks to precise trigonometric calculations
- added educational comments explaining transition from coordinate thinking to geometric thinking
- refactored "basis vector" abstractions into direct angle-based geometric operations

## 0.6.10 (2025-06-16)

### added
- add trait for geonum with optimized polar addition
- affine trait module with affine feature flag
- translation shear and area methods eliminating matrix overhead

## 0.6.9 (2025-01-12)

### added
- machine learning trait module with ml feature flag
- electromagnetics trait module with em feature flag  
- waves trait module with waves feature flag
- clifford number test demonstrating million-dimensional support

### changed
- moved domain-specific functions to separate trait modules
- cleaned core geometric algebra from application code
- architecture: technology validation → stable development

## 0.6.8 (2025-04-28)

### added
- eli5 readme section

## 0.6.7 (2025-04-25)

### added
- learn with ai readme section

## 0.6.6 (2025-04-23)

### added
- organized traits into dedicated modules with feature flags
- implemented conditional compilation for optional traits (optics, projection, manifold)
- tensor test suite comparing mathematical representation and performance with geonum

### changed
- refactored codebase for improved maintainability and organization
- moved primary types into separate modules (geonum.rs, multivector.rs, dimensions.rs)
- reduced lib.rs size
- fixed Clippy lints and warnings throughout codebase
- new benchmark numbers in README

## 0.6.5 (2025-04-21)

### changed
- combined coordinate blowup and multivector expansion in readme

## 0.6.4 (2025-04-19)

### added
- robotics test suite demonstrating geonum applications in robotics and automation
- forward and inverse kinematics tests showing O(1) complexity advantage over matrix methods
- path planning and dynamics controller tests for high-performance robotics applications
- SLAM algorithm and sensor fusion tests showcasing geometric number benefits
- computer vision test suite with feature detection, optical flow, and 3D reconstruction
- camera calibration and image registration tests using angle-based geometric transformations
- neural image processing and segmentation tests demonstrating superior computational efficiency
- object detection tests with geometric frame representation for robotics integration

## 0.6.3 (2025-04-18)

### added
- astrophysics test suite demonstrating geonum capabilities in orbital mechanics
- relativistic orbital system tests showing general relativity effects
- million-body simulation test highlighting O(1) complexity advantages
- simulation helpers with orbital constants and velocity calculations
- celestial mechanics primitives for n-body systems
- relativistic helper functions for future gravitational simulations

## 0.6.2 (2025-04-16)

### added
- show blade as a digital prosthetic to angle, a quantized offset destined to be automated by light in an optical medium

## 0.6.1 (2025-04-15)

### added
- link tests directory for convenient access in readme

## 0.6.0 (2025-04-14)

### added
- blade property to Geonum struct to preserve grade information in high dimensions
- blade-aware constructors: from_polar_blade, scalar
- blade helper methods: with_blade, increment_blade, decrement_blade, complement_blade, preserve_blade, with_product_blade
- improved grade extraction using blade property instead of angle-based heuristics
- robust support for million-dimension geometric algebra operations
- explicit blade field assignments in benchmarks
- tests for blade grade preservation in geometric operations
- blade field and optical computing in README

### changed
- updated wedge product to use blade property for grade tracking
- updated geometric product to use blade grade arithmetic
- updated differentiation and integration to increment/decrement blade property
- removed angle normalizations to prevent grade information loss
- updated Multivector grade extraction to use blade property directly
- consolidated multivector and blade tests into a single test file

### fixed
- fixed high-dimension multivector operations that were failing due to angle normalization
- eliminated angle-based grade inference which was unreliable in high dimensions
- fixed geometric product rule for vector*bivector operations to maintain consistent blade grades
- fixed Clifford conjugate to avoid angle normalizations with TAU
- fixed Geonum initialization in Dimensions::multivector to set blade grades by index

## 0.5.0 (2025-04-10)

- Optics trait for physical optics operations with O(1) complexity
- Projection trait for view operations with direct geometric access
- Manifold trait for collections of geometric numbers with angle-based transformations
- impl optical methods on Geonum: refract, aberrate, otf, abcd_transform, magnify
- Manifold methods for direct data structure manipulation: set, over, compose
- O(1) path-based operations vs O(depth) conventional traversal
- angle arithmetic for high-performance data transformations
- impl path access methods on Multivector: find, transform, path_mapper
- direct angle-encoded paths eliminating complex traversal code
- comprehensive test coverage in tests/optics_test.rs proving constant-time operations

## 0.4.5 (2025-04-08)

### fixed
- switch png with gif

## 0.4.4 (2025-04-08)

### fixed
- github blocks crates.io from loading user attachments

## 0.4.3 (2025-04-08)

### added
- project design
- feature tests proving design

## 0.4.2 (2025-04-07)

### added
- electromagnetic field calculation methods (`electric_field`, `poynting_vector`, etc.)
- base methods for field operations (`from_polar`, `from_cartesian`, `to_cartesian`)
- `inverse_field` method for creating fields with inverse power laws
- `Grade` enum for named grades in geometric algebra (Scalar, Vector, Bivector, etc.)
- improved grade handling with new `grade_range` method accepting `[usize; 2]` parameter
- improved pseudoscalar section extraction with improved angle compatibility checks

### fixed
- fixed multivector section extraction to handle components of different grades
- Updated grade-specific component extraction for consistent behavior

 
## 0.4.1 (2025-04-06)

### added
- tests/electromagnetic_field_theory_test.rs demonstrating electromagnetic field operations
- is_orthogonal method for testing perpendicular geometric numbers
- negate method for reversing direction via angle rotation
- length_diff method for magnitude comparisons with O(1) complexity
- propagate method for wave propagation in space and time
- disperse method for creating waves with dispersion relations

## 0.3.2 (2025-04-05)

### added
- machine learning operations with O(1) complexity
- perceptron_update method for geometric perceptron learning
- regression_from method for creating geometric linear regression
- forward_pass method for neural network operations
- activate method supporting relu, sigmoid, and tanh activations
- extensive test suite in tests/machine_learning_test.rs demonstrating tensor replacement
- comprehensive set theory tests in tests/set_theory_test.rs
- quantum mechanics tests in tests/quantum_mechanics_test.rs
- algorithm benchmarking tests in tests/algorithms_test.rs
- category theory tests in tests/category_theory_test.rs
- number theory tests in tests/numbers_test.rs

### changed
- updated readme with machine learning capabilities
- extended internal test coverage to verify ML functionality
- improved angle_distance usage across clustering implementations
- optimized neural network operations with direct angle transformations

## 0.3.0 (2025-04-03)

### added
- Multivector struct for geometric algebra operations
- square root operation for multivectors (important for rotor generation)
- undual operation (complement to the dual operation, mapping (n-k)-vectors back to k-vectors)
- section for pseudoscalar (extracting components for which a given pseudoscalar is the pseudoscalar)
- regressive product (alternative method for computing the meet of subspaces using A ∨ B = (A* ∧ B*)*)
- automatic differentiation through angle rotation (v' = [r, θ + π/2]) (differential geometric calculus)
- left-contraction and right-contraction operations
- anti-commutator product
- grade involution and clifford conjugate
- grade extraction
- comprehensive test coverage for all operations with proper handling of precision issues
- detailed examples in integration tests demonstrating practical applications

### changed
- added new features to readme and moved items from todo to features
- improved documentation with detailed mathematical explanations
- enhanced error handling with edge cases (empty multivectors)
- optimized angle comparison logic for increased precision
- fixed angle comparisons in tests

## 0.2.0 (2025-04-02)

### added
- extreme dimension support with million-dimensional space tests
- benchmarks showing O(1) vs O(n³) computational advantage
- trivector operations and higher-grade multivectors
- expanded test coverage for all public methods
- new methods: dot(), wedge(), geo(), inv(), div(), normalize()

### changed
- enhanced readme with benchmark results and todo list
- improved documentation and test organization
- updated github actions workflow for testing

## 0.1.1 (2025-initial)

### added
- initial implementation of geometric number spec
- core operations for geometric numbers
- basic multivector support
- basic test coverage