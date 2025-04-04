[![build](https://github.com/mxfactorial/geonum/actions/workflows/publish.yaml/badge.svg)](https://github.com/mxfactorial/geonum/actions)
[![Discord](https://img.shields.io/discord/868565277955203122.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KQdC65bG)
[![docs](https://docs.rs/geonum/badge.svg)](https://docs.rs/geonum)
[![crates.io](https://img.shields.io/crates/v/geonum.svg)](https://crates.io/crates/geonum)
[![coverage](https://coveralls.io/repos/github/mxfactorial/geonum/badge.svg?branch=main)](https://coveralls.io/github/mxfactorial/geonum?branch=main)
[![contribute](https://img.shields.io/badge/contribute-paypal-brightgreen.svg)](https://www.paypal.com/paypalme/mxfactorial)

# geonum

the [geometric number](https://gist.github.com/mxfactorial/c151619d22ef6603a557dbf370864085) spec in rust

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

### use

```
cargo add geonum
```

see `tests/lib_test.rs` examples for scalable numerical simulation of linear and geometric algebra operations

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
- conformal geometry support
- specialized projective geometric algebra (PGA) support
- broadcasting support for operating on multiple objects (point clouds)
