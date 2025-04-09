<br>
<p align="center"><img width="250" alt="dual" src="shield.gif"></p>
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

setting a metric with euclidean and squared norms creates a `k^n` possible transformation problem for vectors

geonum reduces `k^n` to 2

traditional geometric algebra solutions require `2^n` components to represent multivectors in `n` dimensions

geonum enables all algebras and protects them from entropy by setting `log2(4)` components of the most general form (1 scalar + 2 vector + 1 bivector) as dual (⋆):

```rs
/// a geometric number [length, angle]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geonum {
    pub length: f64, // multiply
    pub angle: f64, // add
}
```

### use

```
cargo add geonum
```

see `tests` to learn how geometric numbers unify and simplify mathematical foundations including set theory, category theory and algebraic structures

### benches

#### rank-3 tensor comparison

| implementation | size | time |
|----------------|------|------|
| tensor (O(n³)) | 2 | 1.06 µs |
| tensor (O(n³)) | 3 | 2.25 µs |
| tensor (O(n³)) | 4 | 3.90 µs |
| tensor (O(n³)) | 8 | 7.92 µs |
| tensor (O(n³)) | 16 | 69.54 µs |
| geonum (O(1)) | any | 9.83 ns |

geonum achieves constant O(1) time complexity regardless of problem size, 400× faster than tensor operations at size 4 and 7000× faster at size 16, eliminating cubic scaling of traditional tensor implementations

#### extreme dimension comparison

| implementation | dimensions | time | storage complexity |
|----------------|------------|------|-------------------|
| traditional ga | 10 | 543.40 ns (partial) | O(2^n) = 1024 components |
| traditional ga | 30 | theoretical only | O(2^n) = 1 billion+ components |
| traditional ga | 1000 | impossible | O(2^1000) ≈ 10^301 components |
| traditional ga | 1,000,000 | impossible | O(2^1000000) components |
| geonum (O(1)) | 10 | 134.12 ns | O(1) = 2 components |
| geonum (O(1)) | 30 | 153.64 ns | O(1) = 2 components |
| geonum (O(1)) | 1000 | 2.08 µs | O(1) = 2 components |
| geonum (O(1)) | 1,000,000 | 2.94 ms | O(1) = 2 components |

geonum enables geometric algebra in million-dimensional spaces with constant time operations, achieving whats mathematically impossible with traditional implementations (requires more storage than atoms in the universe)

#### multivector ops

| operation | dimensions | time | traditional ga complexity |
|-----------|------------|------|---------------------------|
| grade extraction | 1,000,000 | 130.69 ns | O(2^n) |
| grade involution | 1,000,000 | 157.18 ns | O(2^n) |
| clifford conjugate | 1,000,000 | 112.90 ns | O(2^n) |
| contractions | 1,000,000 | 266.31 ns | O(2^n) |
| anti-commutator | 1,000,000 | 246.24 ns | O(2^n) |
| all ops combined | 1,000 | 826.57 ns | impossible at high dimensions |

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

### tests
```
cargo fmt --check # format
cargo clippy # lint
cargo test --lib # unit
cargo test --test "*" # feature
cargo bench # bench
cargo llvm-cov # coverage
```

### docs
```
cargo doc --open
```

### todo

- blade classification (identifying geometric types like points, lines, planes)
- rotor estimation algorithms for transforming between sets of geometric objects
- specialized projective geometric algebra (PGA) support
- broadcasting support for operating on multiple objects (point clouds)
