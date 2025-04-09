# changelog

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