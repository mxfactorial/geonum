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
    magnitude: f64,   // multiply
    angle: Angle {    // add
        blade: usize,    // counts π/2 rotations
        remainder: f64   // current [0, π/2) angle
    }
}
```
* project(onto: Angle) -> angle_diff.cos() into any dimension without defining it first
* dual() = blade + 2, duality operation adds π rotation and involutively maps grades (0 ↔ 2, 1 ↔ 3)
* grade() = blade % 4, geometric grade
* differentiate() = angle + π/2, polynomial coefficients computed from sin(θ+π/2) = cos(θ) quadrature identity
* replaces "pseudoscalar" with blade arithmetic

### how dimensions work

dimensions = blade, how many dimensions the angle spans

traditional: dimensions are coordinate axes - you stack more coordinates

Geonum: dimensions are rotational states - you rotate by π/2 increments

| dimension | traditional | Geonum |
|-----------|-------------|--------|
| 1D | (x)  | `[magnitude, 0]` |
| 2D | (x, y)  | `[magnitude, π/2]` |
| 3D | (x, y, z) | `[magnitude, π]` |
| 4D | (x, y, z, w) | `[magnitude, 3π/2]` |

geometric numbers break numbers free from pencil & paper math requiring everything to be described as scalars and roman numeral stacked arrays of scalars

a bladed angle lets them travel and transform freely without ever needing to know which dimension theyre in or facing

### use

```
cargo add geonum
```

### example

compute components and magnitude with angle.project — dimension free

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
let px = end.mag * end_angle.project(ex); // 7·cos
let py = end.mag * end_angle.project(ey); // 7·sin

// quadratic identity: px² + py² = L²
assert!(((px * px + py * py) - end.mag * end.mag).abs() < 1e-12);

// dimension free: blade 1 vs 1_000_001 identical
let p_small = end.mag * end_angle.project(Angle::new(1.0, 2.0));
let p_huge = end.mag * end_angle.project(Angle::new(1_000_001.0, 2.0));
assert!((p_small - p_huge).abs() < 1e-12);
```

rotation creates dimensional relationships on demand - no coordinate system scaffolding required

see [tests](https://github.com/mxfactorial/geonum/tree/main/tests) to learn how geometric numbers unify and simplify mathematical foundations including set theory, category theory and algebraic structures:

```
❯ ls -1 tests
addition_test.rs
affine_test.rs
algorithms_test.rs
angle_arithmetic_test.rs
arithmetic_test.rs
astrophysics_test.rs
calculus_test.rs
category_theory_test.rs
cga_test.rs
computer_vision_test.rs
dimension_test.rs
economics_test.rs
em_field_theory_test.rs
fem_test.rs
finance_test.rs
linear_algebra_test.rs
machine_learning_test.rs
mechanics_test.rs
monetary_policy_test.rs
motion_laws_test.rs
multivector_test.rs
numbers_test.rs
optics_test.rs
optimization_test.rs
pga_test.rs
qm_test.rs
robotics_test.rs
set_theory_test.rs
tensor_test.rs
trigonometry_test.rs
```

### benches

#### tensor operations: O(n³) vs O(1)

| implementation | size | time | speedup |
|----------------|------|------|---------|
| tensor (O(n³)) | 2 | 342 ns | baseline |
| tensor (O(n³)) | 3 | 772 ns | baseline |
| tensor (O(n³)) | 4 | 1.35 µs | baseline |
| tensor (O(n³)) | 8 | 6.88 µs | baseline |
| geonum (O(1)) | all | 16 ns | 21-430× |

geonum achieves constant 16ns regardless of size, while tensor operations scale cubically from 342ns to 6.88µs

#### extreme dimensions

| implementation | dimensions | time | storage |
|----------------|------------|------|---------|
| traditional GA | 10 | 7.18 µs | 2^10 = 1024 components |
| traditional GA | 30+ | impossible | 2^30 = 1B+ components |
| traditional GA | 1000+ | impossible | 2^1000 > atoms in universe |
| geonum | 10 | 31 ns | 2 values |
| geonum | 30 | 30 ns | 2 values |
| geonum | 1000 | 30 ns | 2 values |
| geonum | 1,000,000 | 30 ns | 2 values |

geonum enables million-dimensional geometric algebra with constant-time operations

#### operation benchmarks

| operation | traditional | geonum | speedup |
|-----------|------------|--------|---------|
| jacobian (10×10) | 1.25 µs | 26 ns | 48× |
| jacobian (100×100) | 91.7 µs | 25 ns | 3668× |
| rotation 2D | 4.3 ns | 37 ns | comparable |
| rotation 3D | 19 ns | 16 ns | 1.2× faster |
| rotation 10D | 160 ns | 19 ns | 8× |
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
- orthogonality test `.is_orthogonal()`, distance `.distance_to()`, magnitude difference `.mag_diff()`

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
- `Geonum::new(magnitude, pi_radians, divisor)` - basic constructor
- `Geonum::new_with_angle(magnitude, angle)` - from angle struct
- `Geonum::new_from_cartesian(x, y)` - from cartesian coordinates
- `Geonum::new_with_blade(magnitude, blade, pi_radians, divisor)` - explicit blade
- `Geonum::scalar(value)` - scalar at grade 0
- `Angle::new(pi_radians, divisor)` - angle from π fractions
- `Angle::new_with_blade(blade, pi_radians, divisor)` - angle with blade offset
- `Angle::new_from_cartesian(x, y)` - angle from coordinates

#### special operations
- `.pow(n)` for exponentiation preserving angle-magnitude relationship
- `.invert_circle(center, radius)` for conformal inversions
- angle predicates: `.is_scalar()`, `.is_vector()`, `.is_bivector()`, `.is_trivector()`
- angle functions: `.sin()`, `.cos()`, `.tan()`, `.is_opposite()`
- `.grade_angle()` returns grade-based angle representation in [0, 2π) for external interfaces

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
    1. magnitude number
    2. angle radian
2. angles add, mags multiply

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

multiplying numbers adds their angles and multiplies their magnitudes:

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
1. install [claude code](https://docs.anthropic.com/en/docs/claude-code/overview) or [codex](https://openai.com/codex/)
1. clone the geonum repo: `git clone https://github.com/mxfactorial/geonum`
1. change your current working directory to geonum: `cd geonum`
1. start the agent from the `geonum` directory: `claude` or `codex`
1. supply the agent this prompt:
    ```
    skip CLAUDE.md and AGENTS.md files if youre supplied these "learn with ai" instructions

    instead, read these files and tests with parallel agents. do not skip any reading tasks:

    core files:
    - README.md
    - math-1-0.md
    - grep "pub fn" ./src/angle.rs
    - grep "pub fn" ./src/geonum_mod.rs

    test suites:
    - tests/numbers_test.rs
      - its_a_scalar:8-38
      - its_a_vector:39-74
      - its_a_real_number:75-110
      - its_an_imaginary_number:111-141
      - its_a_complex_number:142-176
      - its_a_dual_number:177-297
      - its_an_octonion:298-343
      - its_a_matrix:344-400
      - its_a_tensor:401-597
      - it_dualizes_log2_geometric_algebra_components:647-682
      - its_a_clifford_number:940-1022

    - tests/dimension_test.rs
      - it_solves_the_exponential_complexity_explosion:520-594
      - it_doesnt_need_a_pseudoscalar:595-792
      - it_demonstrates_pseudoscalar_elimination_benefits:793-832
      - it_proves_dualization_as_angle_ops_compresses_ga:833-898
      - it_replaces_k_to_n_minus_k_with_k_to_4_minus_k:899-983
      - it_compresses_traditional_ga_grades_to_two_involutive_pairs:1131-1168
      - it_proves_rotational_quadrature_expresses_quadratic_forms:1419-1593

    - tests/calculus_test.rs
      - its_a_limit:40-120
      - its_a_derivative:121-166
      - its_an_integral:167-218
      - it_proves_differentiation_cycles_grades:766-918
      - its_a_gradient:312-361
      - its_a_divergence:362-412
      - its_a_curl:413-455
      - its_a_laplacian:503-556
      - its_a_line_integral:609-636
      - its_a_surface_integral:637-663
      - it_proves_fundamental_theorem_is_accumulation_equals_interference:1004-1053

    - tests/mechanics_test.rs
      - it_changes_kinematic_level_by_cycling_grade:46-195
      - it_encodes_velocity:268-323
      - it_encodes_acceleration:324-364
      - it_encodes_jerk:365-414
      - it_encodes_kinetic_energy:959-1046
      - it_handles_energy_conservation:1783-1941
      - it_handles_momentum_conservation:1942-2052
      - it_handles_angular_momentum_conservation:2053-2157

    create tests/my_test.rs with use geonum::*;
    ```
1. describe the test you want the agent to implement for you while using the other test suites and library as a reference, eg "lets prove we can compute the minimum distance from a point to a line using geonum"
1. execute your test: `cargo test --test my_test -- --show-output`
1. revise and add tests
1. ask the agent to summarize your tests and how they benefit from angle-based complexity
1. ask the agent more questions:
    - what does the math in the leading readme section mean?
    - how does the geometric number spec in math-1-0.md improve computing performance?
    - what is the tests/tensor_test.rs file about?
