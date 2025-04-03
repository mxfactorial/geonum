# changelog

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