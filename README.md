<br>
<p align="center"><img width="225" alt="dual" src="shield.gif"></p>
<br>
<p align="center">scaling scientific computing with the <a href="https://gist.github.com/mxfactorial/c151619d22ef6603a557dbf370864085" target="_blank">geometric number</a> spec</p>
<div align="center">

[![build](https://github.com/mxfactorial/geonum/actions/workflows/publish.yaml/badge.svg)](https://github.com/mxfactorial/geonum/actions)
[![docs](https://docs.rs/geonum/badge.svg)](https://docs.rs/geonum)
[![dependency status](https://deps.rs/repo/github/mxfactorial/geonum/status.svg)](https://deps.rs/repo/github/mxfactorial/geonum)
[![crates.io](https://img.shields.io/crates/v/geonum.svg)](https://crates.io/crates/geonum)
[![Discord](https://img.shields.io/discord/868565277955203122.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/KQdC65bG)
[![contribute](https://img.shields.io/badge/contribute-paypal-brightgreen.svg)](https://www.paypal.com/paypalme/mxfactorial)
</div>

# geonum

removing an explicit angle from numbers in the name of "pure" math throws away primitive geometric information

once you amputate the angle from a number to create a "scalar", you throw away its compass

computing angles when they deserve to be static forces math into a cave where numbers must be cast from linearly combined shadows

you start by creating a massive artificial "scalar" superstructure of "dimensions" to store **every possible position** where your scalar vector component can appear—and as a "linear combination" of the dimensions it "spans" with other scalars called "basis vectors"

this brute force scalar alchemy explodes into scalars **everywhere**

with most requiring "sparsity" to conceal how many explicit zeros appear declaring *nothing changed*

the omission of geometry is so extreme at this point its suspicious

now your number must hobble through a prison of complicated "matrix" and "tensor" operations computing expensive dot & cross products in a scalar-dimension chain gang with other "linearly independent" scalars—only to reconstruct the simple detail of the direction its facing

and if you want to change its rate of motion, it must freeze all other scalar dimensions in a "partial derivative" with even more zeros

### protect your numbers

setting a metric with euclidean and squared norms between "linearly combined scalars" creates an n-dimensional, rank-k (`n^k`) component orthogonality search problem for transforming vectors

and supporting traditional geometric algebra operations requires `2^n` components to represent multivectors in `n` dimensions

geonum reduces `n^k(2^n)` to 2
geonum dualizes (⋆) components inside algebra's most general form

setting the metric from the quadrature's bivector shields it from entropy with the `log2(4)` bit minimum:

- 1 scalar, `cos(θ)`
- 2 vector, `sin(θ)cos(φ), sin(θ)sin(φ)`
- 1 bivector, `sin(θ+π/2) = cos(θ)`

```rs
/// dimension-free, geometric number
struct Geonum {
    length: f64,      // multiply
    angle: Angle {    // add
        blade: usize,     // counts π/2 rotations
        value: f64        // current [0, π/2) angle
    }
}
```

* dimensions = blade, how many dimensions the angle spans
* project(onto: Angle) -> angle_diff.cos() into any dimension without defining it first
* dual() = blade + 2, duality operation adds π rotation and involutively maps grades (0 ↔ 2, 1 ↔ 3)
* grade() = blade % 4, geometric grade
* replaces "pseudoscalar" with blade arithmetic

### use

```
cargo add geonum
```

### example

compute components and length with angle.project — dimension free

```rust
use geonum::*;

// origin is an angle
let origin = Angle::new(0.0, 1.0);

// endpoint 7 at pi/6 from origin phase
let end_angle = origin + Angle::new(1.0, 6.0);
let end = Geonum::new_with_angle(7.0, end_angle);

// init axes to assert traditional math:
let ex = Angle::new(0.0, 1.0);
let ey = Angle::new(1.0, 2.0); // +pi/2

// compute projections via angle.project
let px = end.length * end_angle.project(ex); // 7·cos
let py = end.length * end_angle.project(ey); // 7·sin

// quadratic identity: px² + py² = L²
assert!(((px * px + py * py) - end.length * end.length).abs() < 1e-12);

// dimension free: blade 1 vs 1_000_001 identical
let p_small = end.length * end_angle.project(Angle::new(1.0, 2.0));
let p_huge = end.length * end_angle.project(Angle::new(1_000_001.0, 2.0));
assert!((p_small - p_huge).abs() < 1e-12);
```

rotation creates dimensional relationships on demand - no coordinate system scaffolding required

see [tests](https://github.com/mxfactorial/geonum/tree/main/tests) to learn how geometric numbers unify and simplify mathematical foundations including set theory, category theory and algebraic structures

### benches

#### tensor operations: O(n³) vs O(1)

| implementation | size | time | speedup |
|----------------|------|------|---------|
| tensor (O(n³)) | 2 | 358 ns | baseline |
| tensor (O(n³)) | 3 | 788 ns | baseline |
| tensor (O(n³)) | 4 | 1.41 µs | baseline |
| tensor (O(n³)) | 8 | 7.95 µs | baseline |
| geonum (O(1)) | all | 17 ns | 21-468× |

geonum achieves constant 17ns regardless of size, while tensor operations scale cubically from 358ns to 7.95µs

#### extreme dimensions

| implementation | dimensions | time | storage |
|----------------|------------|------|---------|
| traditional GA | 10 | 7.95 µs | 2^10 = 1024 components |
| traditional GA | 30+ | impossible | 2^30 = 1B+ components |
| traditional GA | 1000+ | impossible | 2^1000 > atoms in universe |
| geonum | 10 | 31 ns | 2 values |
| geonum | 30 | 35 ns | 2 values |
| geonum | 1000 | 31 ns | 2 values |
| geonum | 1,000,000 | 31 ns | 2 values |

geonum enables million-dimensional geometric algebra with constant-time operations

#### operation benchmarks

| operation | traditional | geonum | speedup |
|-----------|------------|--------|---------|
| jacobian (10×10) | 1.32 µs | 24 ns | 55× |
| jacobian (100×100) | 98.5 µs | 24 ns | 4100× |
| rotation 2D | 4.6 ns | 39 ns | comparable |
| rotation 3D | 21 ns | 21 ns | equivalent |
| rotation 10D | matrix O(n²) | 21 ns | constant |
| geometric product | decomposition | 17 ns | direct |
| wedge product 2D | 1.9 ns | 60 ns | trigonometric |
| wedge product 10D | 45 components | 60 ns | constant |
| dual operation | pseudoscalar mult | 10 ns | universal |
| differentiation | numerical approx | 11 ns | exact π/2 rotation |
| inversion | matrix ops | 10 ns | direct reciprocal |
| projection | dot products | 15 ns | trigonometric |

all geonum operations maintain constant time regardless of dimension, eliminating exponential scaling of traditional approaches

### features

#### core operations
- dot product `.dot()`, wedge product `.wedge()`, geometric product `.geo()` and `*`
- inverse `.inv()`, division `.div()` and `/`, normalization `.normalize()`
- rotations `.rotate()`, reflections `.reflect()`, projections `.project()`, rejections `.reject()`
- scale `.scale()`, scale-rotate `.scale_rotate()`, negate `.negate()`
- differentiation `.differentiate()` via π/2 rotation, integration `.integrate()` via -π/2 rotation
- meet `.meet()` for subspace intersection with geonum's π-rotation incidence structure
- orthogonality test `.is_orthogonal()`, distance `.distance_to()`, length difference `.length_diff()`

#### angle-blade architecture
- blade count tracks π/2 rotations: 0→scalar, 1→vector, 2→bivector, 3→trivector
- grade = blade % 4 determines geometric behavior regardless of dimension
- `.blade()` returns full transformation history, `.grade()` returns geometric grade
- `.base_angle()` resets blade to minimum for grade (memory optimization)
- `.increment_blade()` and `.decrement_blade()` for direct blade manipulation
- `.copy_blade()` transfers blade structure between geonums

#### dimension handling
- million-dimension geometric algebra with O(1) complexity
- `.project_to_dimension(n)` computes projection to any dimension on demand
- `.create_dimension(length, n)` creates standardized n-dimensional basis element
- dimensions emerge from angle arithmetic, no predefined basis vectors needed
- conformal geometric algebra without 32-component storage
- projective geometric algebra without homogeneous coordinates

#### duality without pseudoscalars
- `.dual()` adds π rotation (2 blades), maps grades 0↔2, 1↔3
- `.undual()` identical to dual in 4-cycle structure  
- `.conjugate()` for clifford conjugation
- universal k→(k+2)%4 duality replaces dimension-specific k→(n-k) formulas
- eliminates I = e₁∧...∧eₙ pseudoscalar and its 2^n storage requirement

#### automatic calculus
- differentiation through π/2 rotation eliminates limit computation
- polynomial coefficients emerge from quadrature sin(θ+π/2) = cos(θ)
- grade cycling: f→f'→f''→f'''→f with grades 0→1→2→3→0
- no symbolic manipulation, no numerical approximation

#### constructors
- `Geonum::new(length, pi_radians, divisor)` - basic constructor
- `Geonum::new_with_angle(length, angle)` - from angle struct
- `Geonum::new_from_cartesian(x, y)` - from cartesian coordinates
- `Geonum::new_with_blade(length, blade, pi_radians, divisor)` - explicit blade
- `Geonum::scalar(value)` - scalar at grade 0
- `Angle::new(pi_radians, divisor)` - angle from π fractions
- `Angle::new_with_blade(blade, pi_radians, divisor)` - angle with blade offset
- `Angle::new_from_cartesian(x, y)` - angle from coordinates

#### special operations
- `.pow(n)` for exponentiation preserving angle-length relationship
- `.invert_circle(center, radius)` for conformal inversions
- `.to_cartesian()` converts to (x, y) coordinates
- angle predicates: `.is_scalar()`, `.is_vector()`, `.is_bivector()`, `.is_trivector()`
- angle functions: `.sin()`, `.cos()`, `.tan()`, `.is_opposite()`
- `.mod_4_angle()` returns angle within current π/2 segment

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

    read tests/dimension_test.rs

    read tests/machine_learning_test.rs

    read tests/em_field_theory_test.rs

    run 'grep "pub fn" ./src/angle.rs' to learn the angle module

    run 'grep "pub fn" ./src/geonum_mod.rs' to learn the geonum module

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
