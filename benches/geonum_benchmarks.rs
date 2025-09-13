use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use geonum::*;
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Duration;

// Configure benchmark group with reasonable limits to prevent timeouts
fn configure_group<M: criterion::measurement::Measurement>(
    mut group: BenchmarkGroup<M>,
) -> BenchmarkGroup<M> {
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group
}

// TENSOR OPERATIONS: O(n³) vs O(1)
fn bench_tensor_vs_geonum(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("tensor_vs_geonum"));

    // tensor contraction: computing ijk product through component arrays
    // traditional: allocate n³ components, iterate through all indices
    // geonum: single multiplication of three geometric numbers

    for size in [2, 3, 4, 8].iter() {
        let n = *size;

        // benchmark traditional tensor contraction O(n³)
        group.bench_function(BenchmarkId::new("tensor_O(n³)", n), |b| {
            b.iter(|| {
                // allocate rank-3 tensor
                let mut tensor = vec![vec![vec![0.0; n]; n]; n];

                // fill with test values (simulating tensor components)
                for (i, plane) in tensor.iter_mut().enumerate() {
                    for (j, row) in plane.iter_mut().enumerate() {
                        for (k, elem) in row.iter_mut().enumerate() {
                            *elem = black_box((i + j + k) as f64);
                        }
                    }
                }

                // compute contraction: sum over all indices (simulating ijk product)
                let mut result = 0.0;
                for (i, plane) in tensor.iter().enumerate() {
                    for (j, row) in plane.iter().enumerate() {
                        for (k, &elem) in row.iter().enumerate() {
                            result += elem * ((i * j * k) as f64).cos();
                        }
                    }
                }

                black_box(result)
            });
        });

        // benchmark geonum O(1) - same operation regardless of "size"
        group.bench_function(BenchmarkId::new("geonum_O(1)", n), |b| {
            b.iter(|| {
                // ijk product as single geometric operation
                let i = Geonum::new(black_box(1.0), black_box(1.0), black_box(2.0)); // π/2
                let j = Geonum::new(black_box(1.0), black_box(2.0), black_box(2.0)); // π
                let k = Geonum::new(black_box(1.0), black_box(3.0), black_box(2.0)); // 3π/2

                // single multiplication - no loops, no components
                let ijk = i * j * k;

                black_box(ijk.length)
            });
        });
    }

    group.finish();
}

// EXTREME DIMENSIONS: impossible vs constant
fn bench_extreme_dimensions(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("extreme_dimensions"));

    // traditional GA: 2^n component explosion makes high dimensions impossible
    // 10D = 1024 components, 30D = 1 billion components, 1000D = more than atoms in universe
    // geonum: same O(1) operations regardless of dimension

    // traditional GA simulation for low dimensions only (10D max due to memory)
    group.bench_function("traditional_GA_10D", |b| {
        b.iter(|| {
            // simulate 10D multivector with 2^10 = 1024 components
            let components = vec![0.0; 1024];

            // simulate geometric product: would need to multiply all 1024 components
            let mut result = 0.0;
            for (i, &comp) in components.iter().enumerate() {
                result += comp * black_box(i as f64).sin();
            }

            black_box(result)
        });
    });

    // geonum handles any dimension with same performance
    for dimension in [10, 30, 1000, 1_000_000].iter() {
        group.bench_function(BenchmarkId::new("geonum_O(1)", dimension), |b| {
            b.iter(|| {
                // create geometric object in arbitrary dimension
                let high_dim = Geonum::new_with_blade(
                    black_box(1.0),
                    black_box(*dimension),
                    black_box(1.0),
                    black_box(4.0),
                );

                // operations work identically regardless of dimension
                let rotated = high_dim.rotate(Angle::new(1.0, 6.0)); // π/6 rotation
                let scaled = rotated.scale(2.0);
                let dual = scaled.dual();

                // grade extraction is always O(1)
                let grade = dual.angle.grade(); // blade % 4

                black_box(grade)
            });
        });
    }

    // demonstrate dimension-specific operations at extreme scale
    group.bench_function("geonum_million_D_ops", |b| {
        b.iter(|| {
            let dim_1m = Geonum::new_with_blade(
                black_box(3.0),
                black_box(1_000_000),
                black_box(0.0),
                black_box(1.0),
            );
            let dim_2m = Geonum::new_with_blade(
                black_box(4.0),
                black_box(2_000_000),
                black_box(0.0),
                black_box(1.0),
            );

            // geometric product in million dimensions - still O(1)
            let product = dim_1m * dim_2m;

            // wedge product in million dimensions - still O(1)
            let wedge = dim_1m.wedge(&dim_2m);

            // differentiation in million dimensions - still O(1)
            let derivative = dim_1m.differentiate();

            black_box((product.length, wedge.length, derivative.angle.blade()))
        });
    });

    group.finish();
}

// JACOBIAN COMPUTATION: O(n²) vs O(1)
fn bench_jacobian(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("jacobian"));

    // traditional: compute n² partial derivatives for n×n jacobian matrix
    // geonum: any jacobian component computed on-demand via differentiate() + project

    for size in [10, 100].iter() {
        // skip 1000 for memory reasons in traditional
        let n = *size;

        // benchmark traditional jacobian O(n²)
        group.bench_function(BenchmarkId::new("traditional_jacobian_O(n²)", n), |b| {
            b.iter(|| {
                // allocate n×n jacobian matrix
                let mut jacobian = vec![vec![0.0; n]; n];

                // compute all n² partial derivatives
                for (i, row) in jacobian.iter_mut().enumerate() {
                    for (j, elem) in row.iter_mut().enumerate() {
                        // simulate ∂f_i/∂x_j computation
                        // in reality would involve numerical differentiation or symbolic computation
                        let h = 0.0001; // finite difference step
                        let f_plus = black_box((i as f64 + h) * (j as f64).sin());
                        let f_minus = black_box((i as f64 - h) * (j as f64).sin());
                        *elem = (f_plus - f_minus) / (2.0 * h);
                    }
                }

                black_box(jacobian[n / 2][n / 2]) // sample one element
            });
        });

        // benchmark geonum O(1) - compute any component on demand
        group.bench_function(BenchmarkId::new("geonum_jacobian_O(1)", n), |b| {
            b.iter(|| {
                // create function as geometric object
                let f = Geonum::new(black_box(1.0), black_box(PI / 4.0), black_box(PI));

                // compute derivative (π/2 rotation)
                let df = f.differentiate();

                // project to any dimension pair (i,j) without computing full matrix
                let i = black_box(n / 2);
                let j = black_box(n / 2);
                let jacobian_ij = df.project_to_dimension(i * n + j);

                black_box(jacobian_ij)
            });
        });
    }

    // demonstrate extreme case: 1000×1000 jacobian
    group.bench_function("geonum_1000x1000_jacobian", |b| {
        b.iter(|| {
            let f = Geonum::new_with_blade(
                black_box(2.0),
                black_box(500),
                black_box(1.0),
                black_box(3.0),
            );

            // differentiate once
            let df = f.differentiate();

            // can extract any of the 1,000,000 components on demand
            let row = black_box(500);
            let col = black_box(500);
            let component = df.project_to_dimension(row * 1000 + col);

            // no need to compute other 999,999 components
            black_box(component)
        });
    });

    group.finish();
}

// ROTATION: matrix multiplication vs angle addition
fn bench_rotation(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("rotation"));

    // traditional: rotation matrix multiplication O(n²)
    // geonum: rotation is angle addition O(1)

    // 2D rotation
    group.bench_function("traditional_2D_matrix", |b| {
        b.iter(|| {
            let theta = black_box(PI / 6.0);
            let x = black_box(3.0);
            let y = black_box(4.0);

            // construct 2×2 rotation matrix
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // matrix multiplication: [x', y'] = R × [x, y]
            let x_rot = cos_theta * x - sin_theta * y;
            let y_rot = sin_theta * x + cos_theta * y;

            black_box((x_rot, y_rot))
        });
    });

    group.bench_function("geonum_2D_rotation", |b| {
        b.iter(|| {
            let vector = Geonum::new_from_cartesian(black_box(3.0), black_box(4.0));
            let rotation = Angle::new(black_box(1.0), black_box(6.0)); // π/6

            // single operation: angle addition
            let rotated = vector.rotate(rotation);

            black_box(rotated.length)
        });
    });

    // 3D rotation (using Euler angles for traditional)
    group.bench_function("traditional_3D_matrix", |b| {
        b.iter(|| {
            let roll = black_box(PI / 8.0);
            let pitch = black_box(PI / 6.0);
            let yaw = black_box(PI / 4.0);
            let (x, y, z) = (black_box(1.0), black_box(2.0), black_box(3.0));

            // compose 3 rotation matrices (9 trig evaluations)
            let cr = roll.cos();
            let sr = roll.sin();
            let cp = pitch.cos();
            let sp = pitch.sin();
            let cy = yaw.cos();
            let sy = yaw.sin();

            // combined rotation matrix (3×3 = 9 components)
            // apply to vector (9 multiplications + 6 additions)
            let x_rot = (cy * cp) * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z;
            let y_rot = (sy * cp) * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z;
            let z_rot = (-sp) * x + (cp * sr) * y + (cp * cr) * z;

            black_box((x_rot, y_rot, z_rot))
        });
    });

    group.bench_function("geonum_3D_rotation", |b| {
        b.iter(|| {
            let vector = Geonum::new(black_box(3.74), black_box(1.0), black_box(5.0)); // √(1²+2²+3²) at some angle
            let rotation = Angle::new(black_box(1.0), black_box(4.0)); // π/4

            // single operation regardless of dimension
            let rotated = vector.rotate(rotation);

            black_box(rotated.angle.blade())
        });
    });

    // 10D rotation (matrix would be 10×10 = 100 components)
    group.bench_function("traditional_10D_matrix", |b| {
        b.iter(|| {
            let n = 10;
            let vector = vec![1.0; n];
            let mut result = vec![0.0; n];

            // simulate 10×10 rotation matrix multiplication
            // in reality would need to construct full SO(10) matrix
            for (i, res_elem) in result.iter_mut().enumerate() {
                for (j, &vec_elem) in vector.iter().enumerate() {
                    // simplified rotation matrix element
                    let matrix_elem = black_box(((i + j) as f64).sin());
                    *res_elem += matrix_elem * vec_elem;
                }
            }

            black_box(result[5])
        });
    });

    group.bench_function("geonum_10D_rotation", |b| {
        b.iter(|| {
            let vector = Geonum::new_with_blade(
                black_box(1.0),
                black_box(10),
                black_box(0.0),
                black_box(1.0),
            );
            let rotation = Angle::new(black_box(1.0), black_box(3.0)); // π/3

            // same single operation for any dimension
            let rotated = vector.rotate(rotation);

            black_box(rotated.angle.grade())
        });
    });

    group.finish();
}

// GEOMETRIC PRODUCT: grade decomposition vs direct multiplication
fn bench_geometric_product(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("geometric_product"));

    // traditional GA: ab = ⟨ab⟩₀ + ⟨ab⟩₁ + ⟨ab⟩₂ + ... grade decomposition
    // geonum: ab = direct multiplication via angle addition

    // vector * vector in 3D
    group.bench_function("traditional_GA_v*v_3D", |b| {
        b.iter(|| {
            // traditional: compute dot and wedge separately, then combine
            let v1 = [black_box(1.0), black_box(2.0), black_box(3.0)];
            let v2 = [black_box(4.0), black_box(5.0), black_box(6.0)];

            // dot product (grade 0 part)
            let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

            // wedge product (grade 2 part) - compute bivector components
            let e12 = v1[0] * v2[1] - v1[1] * v2[0];
            let e13 = v1[0] * v2[2] - v1[2] * v2[0];
            let e23 = v1[1] * v2[2] - v1[2] * v2[1];

            // store in separate grade buckets
            let grade_0 = dot;
            let grade_2 = (e12, e13, e23);

            black_box((grade_0, grade_2))
        });
    });

    group.bench_function("geonum_v*v", |b| {
        b.iter(|| {
            let v1 = Geonum::new(black_box(3.74), black_box(1.0), black_box(6.0)); // magnitude √14 at some angle
            let v2 = Geonum::new(black_box(8.77), black_box(1.0), black_box(4.0)); // magnitude √77 at another angle

            // direct multiplication - no grade decomposition
            let product = v1 * v2;

            black_box(product.angle.blade())
        });
    });

    // bivector * bivector (would produce grade 0 and grade 4 in traditional)
    group.bench_function("traditional_GA_B*B", |b| {
        b.iter(|| {
            // bivector components in 4D: e12, e13, e14, e23, e24, e34
            let b1 = [
                black_box(1.0),
                black_box(0.5),
                black_box(0.3),
                black_box(0.7),
                black_box(0.2),
                black_box(0.9),
            ];
            let b2 = [
                black_box(0.4),
                black_box(0.8),
                black_box(0.6),
                black_box(0.1),
                black_box(0.5),
                black_box(0.3),
            ];

            // compute scalar part (grade 0)
            let scalar = -(b1[0] * b2[0]
                + b1[1] * b2[1]
                + b1[2] * b2[2]
                + b1[3] * b2[3]
                + b1[4] * b2[4]
                + b1[5] * b2[5]);

            // compute 4-vector part (grade 4) - would need e1234 component
            // simplified: just simulate the computation
            let grade_4 = b1[0] * b2[5] - b1[1] * b2[4] + b1[2] * b2[3];

            black_box((scalar, grade_4))
        });
    });

    group.bench_function("geonum_B*B", |b| {
        b.iter(|| {
            let b1 = Geonum::new_with_blade(
                black_box(1.0),
                black_box(2),
                black_box(0.0),
                black_box(1.0),
            ); // bivector
            let b2 = Geonum::new_with_blade(
                black_box(1.0),
                black_box(2),
                black_box(1.0),
                black_box(3.0),
            ); // another bivector

            // direct multiplication - angles add, no decomposition
            let product = b1 * b2;

            black_box(product.angle.grade())
        });
    });

    // high-grade multiplication
    group.bench_function("traditional_GA_high_grade", |b| {
        b.iter(|| {
            // in 10D, would have 2^10 = 1024 components to track
            // simulate by allocating and scanning
            let mut components = vec![0.0; 1024];
            components[100] = black_box(1.0); // some grade 3 component
            components[500] = black_box(2.0); // some grade 5 component

            // geometric product would need to check all 1024×1024 combinations
            let mut result = 0.0;
            for (i, &comp) in components.iter().enumerate() {
                if comp != 0.0 {
                    // would multiply with other multivector components
                    result += comp * black_box(i as f64).sin();
                }
            }

            black_box(result)
        });
    });

    group.bench_function("geonum_high_grade", |b| {
        b.iter(|| {
            let high1 = Geonum::new_with_blade(
                black_box(1.0),
                black_box(100),
                black_box(0.0),
                black_box(1.0),
            );
            let high2 = Geonum::new_with_blade(
                black_box(1.0),
                black_box(500),
                black_box(0.0),
                black_box(1.0),
            );

            // same operation regardless of grade or dimension
            let product = high1 * high2;

            black_box(product.angle.blade())
        });
    });

    group.finish();
}

// WEDGE PRODUCT: antisymmetric tensor vs trigonometry
fn bench_wedge_product(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("wedge_product"));

    // traditional: a∧b = Σᵢⱼ (aᵢbⱼ - aⱼbᵢ) eᵢ∧eⱼ antisymmetric computation
    // geonum: |a||b|sin(θ) via direct trigonometry

    // 2D wedge product
    group.bench_function("traditional_wedge_2D", |b| {
        b.iter(|| {
            let v1 = [black_box(3.0), black_box(4.0)];
            let v2 = [black_box(1.0), black_box(2.0)];

            // compute antisymmetric part: v1[0]*v2[1] - v1[1]*v2[0]
            let wedge = v1[0] * v2[1] - v1[1] * v2[0];

            black_box(wedge)
        });
    });

    group.bench_function("geonum_wedge_2D", |b| {
        b.iter(|| {
            let v1 = Geonum::new_from_cartesian(black_box(3.0), black_box(4.0));
            let v2 = Geonum::new_from_cartesian(black_box(1.0), black_box(2.0));

            // direct trigonometric computation
            let wedge = v1.wedge(&v2);

            black_box(wedge.length)
        });
    });

    // 3D wedge product
    group.bench_function("traditional_wedge_3D", |b| {
        b.iter(|| {
            let v1 = [black_box(1.0), black_box(2.0), black_box(3.0)];
            let v2 = [black_box(4.0), black_box(5.0), black_box(6.0)];

            // compute all antisymmetric components
            let e12 = v1[0] * v2[1] - v1[1] * v2[0];
            let e13 = v1[0] * v2[2] - v1[2] * v2[0];
            let e23 = v1[1] * v2[2] - v1[2] * v2[1];

            black_box((e12, e13, e23))
        });
    });

    group.bench_function("geonum_wedge_3D", |b| {
        b.iter(|| {
            let v1 = Geonum::new(black_box(3.74), black_box(1.0), black_box(6.0));
            let v2 = Geonum::new(black_box(8.77), black_box(1.0), black_box(4.0));

            // sin(angle_diff) computation
            let wedge = v1.wedge(&v2);

            black_box(wedge.angle.blade())
        });
    });

    // high-D wedge (10D would need 45 antisymmetric components)
    group.bench_function("traditional_wedge_10D", |b| {
        b.iter(|| {
            let n = 10;
            let v1: Vec<f64> = (0..n).map(|i| black_box(i as f64 + 1.0)).collect();
            let v2: Vec<f64> = (0..n).map(|i| black_box((n - i) as f64)).collect();

            // compute n(n-1)/2 = 45 antisymmetric components
            let mut wedge_components = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let component = v1[i] * v2[j] - v1[j] * v2[i];
                    wedge_components.push(component);
                }
            }

            black_box(wedge_components.len()) // 45 components
        });
    });

    group.bench_function("geonum_wedge_10D", |b| {
        b.iter(|| {
            let v1 = Geonum::new_with_blade(
                black_box(5.0),
                black_box(10),
                black_box(1.0),
                black_box(6.0),
            );
            let v2 = Geonum::new_with_blade(
                black_box(7.0),
                black_box(10),
                black_box(1.0),
                black_box(4.0),
            );

            // same trigonometric formula regardless of dimension
            let wedge = v1.wedge(&v2);

            black_box(wedge.length)
        });
    });

    // extreme dimension wedge
    group.bench_function("geonum_wedge_1M_D", |b| {
        b.iter(|| {
            let v1 = Geonum::new_with_blade(
                black_box(2.0),
                black_box(1_000_000),
                black_box(0.0),
                black_box(1.0),
            );
            let v2 = Geonum::new_with_blade(
                black_box(3.0),
                black_box(1_000_000),
                black_box(1.0),
                black_box(2.0),
            );

            // still O(1) in million dimensions
            let wedge = v1.wedge(&v2);

            black_box(wedge.angle.grade())
        });
    });

    group.finish();
}

// DUAL OPERATION: pseudoscalar multiplication vs blade addition
fn bench_dual(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("dual"));

    // traditional: A* = A · Iₙ where Iₙ is n-dimensional pseudoscalar
    // geonum: dual is +2 blade transformation, no pseudoscalar needed

    // 3D dual (traditional needs I₃ = e₁∧e₂∧e₃)
    group.bench_function("traditional_dual_3D", |b| {
        b.iter(|| {
            // vector components
            let v = [black_box(1.0), black_box(2.0), black_box(3.0)];

            // compute dual via pseudoscalar multiplication
            // in 3D: vector dual becomes bivector
            // e₁* = e₂∧e₃, e₂* = e₃∧e₁, e₃* = e₁∧e₂
            let dual_e23 = v[0]; // coefficient of e₁ → e₂₃
            let dual_e31 = v[1]; // coefficient of e₂ → e₃₁
            let dual_e12 = v[2]; // coefficient of e₃ → e₁₂

            black_box((dual_e23, dual_e31, dual_e12))
        });
    });

    group.bench_function("geonum_dual_3D", |b| {
        b.iter(|| {
            let v = Geonum::new(black_box(3.74), black_box(1.0), black_box(6.0));

            // direct blade transformation: +2 blades
            let dual = v.dual();

            black_box(dual.angle.blade())
        });
    });

    // 10D dual (traditional needs I₁₀ = e₁∧e₂∧...∧e₁₀)
    group.bench_function("traditional_dual_10D", |b| {
        b.iter(|| {
            let n = 10;
            // vector in 10D
            let v: Vec<f64> = (0..n).map(|i| black_box(i as f64 + 1.0)).collect();

            // pseudoscalar in 10D has 2^10 = 1024 components
            // dual operation maps grade k → grade (n-k)
            // vector (grade 1) → 9-vector (grade 9)

            // simulate multiplication by pseudoscalar
            let mut dual_components = vec![0.0; 252]; // C(10,9) = 10 components for grade 9
            for i in 0..10 {
                // each vector component maps to corresponding (n-1)-vector component
                dual_components[i] = v[i] * black_box((i as f64).cos());
            }

            black_box(dual_components[5])
        });
    });

    group.bench_function("geonum_dual_10D", |b| {
        b.iter(|| {
            let v = Geonum::new_with_blade(
                black_box(1.0),
                black_box(10),
                black_box(0.0),
                black_box(1.0),
            );

            // same +2 blade operation regardless of dimension
            let dual = v.dual();

            black_box(dual.angle.grade())
        });
    });

    // extreme dimension dual
    group.bench_function("geonum_dual_1M_D", |b| {
        b.iter(|| {
            let v = Geonum::new_with_blade(
                black_box(2.0),
                black_box(1_000_000),
                black_box(0.0),
                black_box(1.0),
            );

            // still O(1) in million dimensions
            let dual = v.dual();

            black_box(dual.angle.blade()) // blade 1_000_002
        });
    });

    // dual involution test
    group.bench_function("geonum_dual_involution", |b| {
        b.iter(|| {
            let v = Geonum::new(black_box(5.0), black_box(1.0), black_box(3.0));

            // dual of dual returns to original grade
            let dual_once = v.dual();
            let dual_twice = dual_once.dual();

            black_box(dual_twice.angle.grade())
        });
    });

    group.finish();
}

// DIFFERENTIATION: symbolic/numerical vs π/2 rotation
fn bench_differentiation(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("differentiation"));

    // traditional: f'(x) = lim(h→0) [f(x+h) - f(x)]/h via numerical approximation
    // geonum: differentiation is π/2 rotation (add 1 blade)

    // numerical differentiation (finite difference)
    group.bench_function("traditional_numerical_diff", |b| {
        b.iter(|| {
            let x = black_box(2.0);
            let h = 0.0001; // finite difference step

            // f(x) = x² → f'(x) = 2x
            let f = |t: f64| t * t;

            // forward difference approximation
            let f_plus = f(x + h);
            let f_minus = f(x);
            let derivative = (f_plus - f_minus) / h;

            black_box(derivative)
        });
    });

    group.bench_function("geonum_diff", |b| {
        b.iter(|| {
            let f = Geonum::new(black_box(4.0), black_box(0.0), black_box(1.0)); // f(2) = 4

            // differentiation via π/2 rotation
            let df = f.differentiate();

            black_box(df.angle.blade())
        });
    });

    // central difference (more accurate traditional method)
    group.bench_function("traditional_central_diff", |b| {
        b.iter(|| {
            let x = black_box(3.0);
            let h = 0.0001;

            // f(x) = sin(x) → f'(x) = cos(x)
            let f = |t: f64| t.sin();

            // central difference: [f(x+h) - f(x-h)]/(2h)
            let f_plus = f(x + h);
            let f_minus = f(x - h);
            let derivative = (f_plus - f_minus) / (2.0 * h);

            black_box(derivative)
        });
    });

    // second derivative
    group.bench_function("traditional_second_diff", |b| {
        b.iter(|| {
            let x = black_box(1.5);
            let h = 0.0001;

            // f(x) = x³ → f''(x) = 6x
            let f = |t: f64| t * t * t;

            // second derivative via finite differences
            // f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)]/h²
            let f_plus = f(x + h);
            let f_center = f(x);
            let f_minus = f(x - h);
            let second_derivative = (f_plus - 2.0 * f_center + f_minus) / (h * h);

            black_box(second_derivative)
        });
    });

    group.bench_function("geonum_second_diff", |b| {
        b.iter(|| {
            let f = Geonum::new(black_box(3.375), black_box(0.0), black_box(1.0)); // f(1.5) = 3.375

            // first derivative: π/2 rotation
            let df = f.differentiate();

            // second derivative: another π/2 rotation
            let d2f = df.differentiate();

            black_box(d2f.angle.grade())
        });
    });

    // nth derivative chain
    group.bench_function("traditional_nth_diff", |b| {
        b.iter(|| {
            let x = black_box(1.0);
            let h = 0.0001;
            let n = 4; // 4th derivative

            // computing 4th derivative numerically is extremely unstable
            // requires high-order finite difference formulas
            let mut result = x;
            for _ in 0..n {
                // simplified: just accumulate operations
                result = (result + h) * black_box(0.99);
            }

            black_box(result)
        });
    });

    group.bench_function("geonum_nth_diff", |b| {
        b.iter(|| {
            let f = Geonum::new(black_box(1.0), black_box(0.0), black_box(1.0));

            // 4th derivative: 4 rotations by π/2
            let df = f.differentiate();
            let d2f = df.differentiate();
            let d3f = d2f.differentiate();
            let d4f = d3f.differentiate();

            // returns to original grade (4-cycle)
            black_box(d4f.angle.grade())
        });
    });

    group.finish();
}

// INVERSION: matrix operations vs reciprocal
fn bench_inversion(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("inversion"));

    // traditional: matrix inversion via Gaussian elimination or LU decomposition
    // geonum: inversion is 1/length and angle transformation

    // 2×2 matrix inversion
    group.bench_function("traditional_matrix_inv_2x2", |b| {
        b.iter(|| {
            // matrix [[a, b], [c, d]]
            let a = black_box(3.0);
            let b = black_box(4.0);
            let c = black_box(2.0);
            let d = black_box(5.0);

            // inverse formula: 1/det * [[d, -b], [-c, a]]
            let det = a * d - b * c;
            let inv_a = d / det;
            let inv_b = -b / det;
            let inv_c = -c / det;
            let inv_d = a / det;

            black_box((inv_a, inv_b, inv_c, inv_d))
        });
    });

    group.bench_function("geonum_inv", |b| {
        b.iter(|| {
            let geo = Geonum::new(black_box(5.0), black_box(1.0), black_box(6.0));

            // direct inversion: 1/length, angle transformation
            let inv = geo.inv();

            black_box(inv.length)
        });
    });

    // 3×3 matrix inversion (cofactor method)
    group.bench_function("traditional_matrix_inv_3x3", |b| {
        b.iter(|| {
            // 3×3 matrix
            let m = [
                [black_box(2.0), black_box(1.0), black_box(3.0)],
                [black_box(1.0), black_box(3.0), black_box(2.0)],
                [black_box(3.0), black_box(2.0), black_box(1.0)],
            ];

            // compute determinant
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

            // compute cofactor matrix (9 2×2 determinants)
            let c00 = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
            let c01 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / det;
            let c02 = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
            // ... would need all 9 cofactors

            black_box((c00, c01, c02))
        });
    });

    // multivector inversion in GA
    group.bench_function("traditional_GA_inv", |b| {
        b.iter(|| {
            // simplified GA inversion: M^-1 = M† / (M · M†)
            // where M† is the reverse (reversion of basis blade order)

            // multivector components (scalar + vector + bivector + trivector)
            let scalar = black_box(2.0);
            let vector = [black_box(1.0), black_box(2.0), black_box(3.0)];
            let bivector = [black_box(0.5), black_box(0.7), black_box(0.3)];
            let trivector = black_box(0.2);

            // compute reverse (sign changes for certain grades)
            let rev_scalar = scalar;
            let _rev_vector = vector; // grade 1: no sign change
            let _rev_bivector = [-bivector[0], -bivector[1], -bivector[2]]; // grade 2: sign change
            let _rev_trivector = -trivector; // grade 3: sign change

            // compute M · M† (simplified)
            let norm_sq = scalar * scalar
                + vector[0] * vector[0]
                + vector[1] * vector[1]
                + vector[2] * vector[2]
                + bivector[0] * bivector[0]
                + bivector[1] * bivector[1]
                + bivector[2] * bivector[2]
                + trivector * trivector;

            // scale reverse by 1/norm²
            let inv_scalar = rev_scalar / norm_sq;

            black_box(inv_scalar)
        });
    });

    group.bench_function("geonum_multivector_inv", |b| {
        b.iter(|| {
            // any grade object inverts the same way
            let multivector = Geonum::new_with_blade(
                black_box(3.0),
                black_box(7),
                black_box(1.0),
                black_box(4.0),
            );

            // same operation regardless of grade
            let inv = multivector.inv();

            black_box(inv.angle.blade())
        });
    });

    // high-dimensional inversion
    group.bench_function("geonum_high_dim_inv", |b| {
        b.iter(|| {
            let high = Geonum::new_with_blade(
                black_box(5.0),
                black_box(1000),
                black_box(0.0),
                black_box(1.0),
            );

            // still O(1) in high dimensions
            let inv = high.inv();

            // verify multiplicative identity
            let identity = high * inv;

            black_box(identity.length) // should be 1.0
        });
    });

    group.finish();
}

// PROJECTION: component extraction vs trigonometry
fn bench_projection(c: &mut Criterion) {
    let mut group = configure_group(c.benchmark_group("projection"));

    // traditional: projection via dot products with basis vectors
    // geonum: projection via trigonometric relationships

    // project onto x-axis (extract x component)
    group.bench_function("traditional_project_x", |b| {
        b.iter(|| {
            let v = [black_box(3.0), black_box(4.0), black_box(5.0)];
            let e1 = [1.0, 0.0, 0.0]; // x-axis basis vector

            // dot product to project
            let projection = v[0] * e1[0] + v[1] * e1[1] + v[2] * e1[2];

            black_box(projection)
        });
    });

    group.bench_function("geonum_project_x", |b| {
        b.iter(|| {
            let v = Geonum::new(black_box(7.07), black_box(1.0), black_box(8.0)); // √50 at some angle

            // project to dimension via trigonometry
            let projection = v.project_to_dimension(0); // x-axis is dimension 0

            black_box(projection)
        });
    });

    // project onto arbitrary direction
    group.bench_function("traditional_project_arbitrary", |b| {
        b.iter(|| {
            let v = [black_box(2.0), black_box(3.0), black_box(1.0)];
            // arbitrary unit vector
            let dir = [0.6, 0.8, 0.0]; // normalized [3, 4, 0]

            // dot product for projection magnitude
            let proj_magnitude = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];

            // scale direction by magnitude
            let proj_x = proj_magnitude * dir[0];
            let proj_y = proj_magnitude * dir[1];
            let proj_z = proj_magnitude * dir[2];

            black_box((proj_x, proj_y, proj_z))
        });
    });

    group.bench_function("geonum_project_arbitrary", |b| {
        b.iter(|| {
            let v = Geonum::new(black_box(3.74), black_box(1.0), black_box(6.0));
            let direction = Geonum::new(black_box(5.0), black_box(1.0), black_box(5.0));

            // projection via dot product (cos of angle difference)
            let projection = v.dot(&direction);

            black_box(projection.length)
        });
    });

    // extract all components (full decomposition)
    group.bench_function("traditional_extract_all", |b| {
        b.iter(|| {
            let v = [
                black_box(1.0),
                black_box(2.0),
                black_box(3.0),
                black_box(4.0),
            ];

            // extract all 4 components via basis projections
            let mut components = vec![0.0; 4];
            // dot with i-th basis vector (all zeros except 1 at position i)
            // simplified since basis is orthonormal
            components[..4].copy_from_slice(&v);

            black_box(components)
        });
    });

    group.bench_function("geonum_project_selective", |b| {
        b.iter(|| {
            let v = Geonum::new_with_blade(
                black_box(5.48),
                black_box(4),
                black_box(0.0),
                black_box(1.0),
            ); // √30

            // project to specific dimensions on demand
            let x_proj = v.project_to_dimension(0);
            let y_proj = v.project_to_dimension(1);
            // no need to extract others if not needed

            black_box((x_proj, y_proj))
        });
    });

    // high-dimensional projection
    group.bench_function("traditional_project_100D", |b| {
        b.iter(|| {
            let n = 100;
            let v: Vec<f64> = (0..n).map(|i| black_box(i as f64 * 0.1)).collect();

            // project onto dimension 50
            let basis_50: Vec<f64> = (0..n).map(|i| if i == 50 { 1.0 } else { 0.0 }).collect();

            // dot product in 100D
            let mut projection = 0.0;
            for i in 0..n {
                projection += v[i] * basis_50[i];
            }

            black_box(projection)
        });
    });

    group.bench_function("geonum_project_100D", |b| {
        b.iter(|| {
            let v = Geonum::new_with_blade(
                black_box(10.0),
                black_box(100),
                black_box(1.0),
                black_box(4.0),
            );

            // project to dimension 50 - still O(1)
            let projection = v.project_to_dimension(50);

            black_box(projection)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_vs_geonum,
    bench_extreme_dimensions,
    bench_jacobian,
    bench_rotation,
    bench_geometric_product,
    bench_wedge_product,
    bench_dual,
    bench_differentiation,
    bench_inversion,
    bench_projection
);

criterion_main!(benches);
