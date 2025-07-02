<br>
<p align="center"><img width="225" alt="dual" src="shield.gif"></p>
<br>
<p align="center">scaling scientific computing with the <a href="https://gist.github.com/mxfactorial/c151619d22ef6603a557dbf370864085" target="_blank">geometric number</a> spec</p>
<div align="center">

[![build](https://github.com/mxfactorial/geonum/actions/workflows/publish.yaml/badge.svg)](https://github.com/mxfactorial/geonum/actions)
[![Discord](https://img.shields.io/discord/868565277955203122.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KQdC65bG)
[![docs](https://docs.rs/geonum/badge.svg)](https://docs.rs/geonum)
[![crates.io](https://img.shields.io/crates/v/geonum.svg)](https://crates.io/crates/geonum)
[![coverage](https://coveralls.io/repos/github/mxfactorial/geonum/badge.svg?branch=main)](https://coveralls.io/github/mxfactorial/geonum?branch=main)
[![contribute](https://img.shields.io/badge/contribute-paypal-brightgreen.svg)](https://www.paypal.com/paypalme/mxfactorial)
</div>

# geonum

setting a metric with euclidean and squared norms creates a `k^n` component orthogonality search problem for transforming vectors

traditional geometric algebra solutions require `2^n` components to represent multivectors in `n` dimensions

geonum reduces `k^n(2^n)` to 2

geonum dualizes (⋆) components inside algebra's most general form

setting the metric from the quadrature's bivector shields it from entropy with the `log2(4)` bit minimum:

- 1 scalar, `cos(θ)`
- 2 vector, `sin(θ)cos(φ), sin(θ)sin(φ)`
- 1 bivector, `sin(θ+π/2) = cos(θ)`

```rs
/// a geometric number
pub struct Geonum {
    pub length: f64, // multiply
    pub angle: f64,  // add
    pub blade: usize // count π/2 angle turns
}
```

### use

```
cargo add geonum
```

see [tests](https://github.com/mxfactorial/geonum/tree/main/tests) to learn how geometric numbers unify and simplify mathematical foundations including set theory, category theory and algebraic structures

### benches

#### rank-3 tensor comparison

| implementation | size | time |
|----------------|------|------|
| tensor (O(n³)) | 2 | 1.05 µs |
| tensor (O(n³)) | 3 | 2.25 µs |
| tensor (O(n³)) | 4 | 4.20 µs |
| tensor (O(n³)) | 8 | 7.83 µs |
| tensor (O(n³)) | 16 | 66.65 µs |
| geonum (O(1)) | any | 15.52 ns |

geonum achieves constant O(1) time complexity regardless of problem size, 270× faster than tensor operations at size 4 and 4300× faster at size 16, eliminating cubic scaling of traditional tensor implementations

#### extreme dimension comparison

| implementation | dimensions | time | storage complexity |
|----------------|------------|------|-------------------|
| traditional ga | 10 | 545.69 ns (partial) | O(2^n) = 1024 components |
| traditional ga | 30 | theoretical only | O(2^n) = 1 billion+ components |
| traditional ga | 1000 | impossible | O(2^1000) ≈ 10^301 components |
| traditional ga | 1,000,000 | impossible | O(2^1000000) components |
| geonum (O(1)) | 10 | 78.00 ns | O(1) = 2 components |
| geonum (O(1)) | 30 | 79.64 ns | O(1) = 2 components |
| geonum (O(1)) | 1000 | 77.44 ns | O(1) = 2 components |
| geonum (O(1)) | 1,000,000 | 78.79 ns | O(1) = 2 components |

geonum enables geometric algebra in million-dimensional spaces with constant time operations, achieving whats physically impossible with traditional implementations (requires more storage than atoms in the universe)

#### multivector ops

| operation | dimensions | time | traditional ga complexity |
|-----------|------------|------|---------------------------|
| grade extraction | 1,000,000 | 136.46 ns | O(2^n) |
| grade involution | 1,000,000 | 153.37 ns | O(2^n) |
| clifford conjugate | 1,000,000 | 111.39 ns | O(2^n) |
| contractions | 1,000,000 | 292.56 ns | O(2^n) |
| anti-commutator | 1,000,000 | 264.46 ns | O(2^n) |
| all ops combined | 1,000 | 883.74 ns | impossible at high dimensions |

geonum performs all major multivector operations with exceptional efficiency in million-dimensional spaces, maintaining sub-microsecond performance for grade-specific operations that would require exponential time and memory in traditional geometric algebra implementations

### features

- dot product, wedge product, geometric product
- inverse, division, normalization
- million-dimension geometric algebra with O(1) complexity
- multivector support and trivector operations
- rotations, reflections, projections, rejections
- exponential, interior product, dual operations
- meet and join, commutator product, sandwich product
- left-contraction, right-contraction
- anti-commutator product
- grade involution and clifford conjugate
- grade extraction
- section for pseudoscalar (extracting components for which a given pseudoscalar is the pseudoscalar)
- square root operation for multivectors
- undual operation (complement to the dual operation)
- regressive product (alternative method for computing the meet of subspaces)
- automatic differentiation through angle rotation (v' = [r, θ + π/2]) (differential geometric calculus)
- transforms category theory abstractions into simple angle transformations
- unifies discrete and continuous math through a common geometric framework
- provides physical geometric interpretations for abstract mathematical concepts
- automates away unnecessary mathematical formalism using length-angle representation
- enables scaling precision in statistical modeling through direct angle quantization
- supports time evolution via simple angle rotation (angle += energy * time)
- provides statistical methods for angle distributions (arithmetic/circular means, variance, expectation values)
- enables O(1) machine learning operations that would otherwise require O(n²) or O(2^n) complexity
- implements perceptron learning, regression modeling, neural networks and activation functions 
- replaces tensor-based neural network operations with direct angle transformations
- enables scaling to millions of dimensions with constant-time ML computations
- eliminates the "orthogonality search" bottleneck in traditional tensor based machine learning implementations
- angle-encoded data paths for O(1) structure traversal vs O(depth) conventional methods
- optical transformations via direct angle operations (refraction, aberration, OTF)
- Manifold trait for collection operations with lens-like path transformations

### tests
```
cargo check # compile
cargo fmt --check # format
cargo clippy # lint
cargo test --lib # unit
cargo test --test "*" # feature
cargo test --doc # doc
cargo bench # bench
cargo llvm-cov # coverage
```

### eli5

geometric numbers depend on 2 rules:

1. all numbers require a 2 component minimum:
    1. length number
    2. angle radian
2. angles add, lengths multiply

so:

- a 1d number or scalar: `[4, 0]`
    - 4 units long facing 0 radians
- a 2d number or vector: `[[4, 0], [4, pi/2]]`
    - one component 4 units at 0 radians
    - one component 4 units at pi/2 radians
- a 3d number: `[[4, 0], [4, pi/2], [4, pi]]`
    - one component 4 units at 0 radians
    - one component 4 units at pi/2 radians
    - one component 4 units at pi radians

higher dimensions just keep adding components rotated by +pi/2 each time

dimensions are created by rotations and not stacking coordinates

multiplying numbers adds their angles and multiplies their lengths:

- `[2, 0] * [3, pi/2] = [6, pi/2]`

differentiation is just rotating a number by +pi/2:

- `[4, 0]' = [4, pi/2]`
- `[4, pi/2]' = [4, pi]`
- `[4, pi]' = [4, 3pi/2]`
- `[4, 3pi/2]' = [4, 2pi] = [4, 0]`

thats why calculus works automatically and autodiff is o1

and if you spot a blade field in the code, it just counts how many pi/2 turns your angle added

blade = 0 means zero turns  
blade = 1 means one pi/2 turn  
blade = 2 means two pi/2 turns  
etc

blade lets your geometric number index which higher dimensional structure its in without using matrices or tensors:
```
[4, 0]        blade = 0  (initial direction)
    |
    v

[4, pi/2]     blade = 1  (rotated +90 degrees)
    |
    v

[4, pi]       blade = 2  (rotated +180 degrees)
    |
    v

[4, 3pi/2]    blade = 3  (rotated +270 degrees)
    |
    v

[4, 2pi]      blade = 4  (rotated full circle back to start)
```
each +pi/2 turn rotates your geometric number into the next orthogonal direction

geometric numbers build dimensions by rotating—not stacking

### learn with ai

1. install rust: https://www.rust-lang.org/tools/install
1. create an api key with anthropic: https://console.anthropic.com/
1. purchase api credit
1. install [claude code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
1. clone the geonum repo: `git clone https://github.com/mxfactorial/geonum`
1. change your current working directory to geonum: `cd geonum`
1. start claude from the `geonum` directory: `claude`
1. configure claude with your api key
1. supply it this series of prompts:
    ```
    read README.md

    read the math-1-0.md geometric number spec

    read tests/numbers_test.rs

    read tests/multivector_test.rs

    read tests/machine_learning_test.rs

    read tests/astrophysics_test.rs

    read tests/em_field_theory_test.rs

    run 'grep "pub fn" ./src/dimensions.rs' to learn the dimensions module

    run 'grep "pub fn" ./src/geonum_mod.rs' to learn the geonum module

    run 'grep "pub fn" ./src/multivector.rs' to learn the multivector module

    now run 'touch tests/my_test.rs'

    import geonum in tests/my_test.rs with use geonum::*;
    ```
1. describe the test you want the agent to implement for you while using the other test suites and library as a reference
1. execute your test: `cargo test --test my_test -- --show-output`
1. revise and add tests
1. ask the agent to summarize your tests and how they benefit from angle-based complexity
1. ask the agent more questions:
    - what does the math in the leading readme section mean?
    - how does the geometric number spec in math-1-0.md improve computing performance?
    - what is the tests/tensor_test.rs file about?