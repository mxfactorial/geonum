[![build](https://github.com/mxfactorial/geonum/actions/workflows/publish.yaml/badge.svg)](https://github.com/mxfactorial/geonum/actions)
[![Discord](https://img.shields.io/discord/868565277955203122.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KQdC65bG)
[![docs](https://docs.rs/geonum/badge.svg)](https://docs.rs/geonum)
[![crates.io](https://img.shields.io/crates/v/geonum.svg)](https://crates.io/crates/geonum)
[![contribute](https://img.shields.io/badge/contribute-paypal-brightgreen.svg)](https://www.paypal.com/paypalme/mxfactorial)

# geonum

the [geometric number](https://gist.github.com/mxfactorial/c151619d22ef6603a557dbf370864085) spec in rust

### use

`cargo add geonum`

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

### tests
```
cargo fmt --check # format check
cargo clippy # lint
cargo test --lib # unit
cargo test --test lib_test # feature
cargo bench # bench
cargo llvm-cov # coverage
```

### docs
```
cargo doc --open
```

### todo

- [x] dot product, wedge product, geometric product
- [x] inverse, division, normalization
- [x] million-dimension geometric algebra with O(1) complexity
- [x] multivector support and trivector operations
- [ ] rotations, reflections, projections, rejections
- [ ] exponential, interior product, dual operations
- [ ] meet and join, commutator product, sandwich product
