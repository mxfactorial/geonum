use criterion::{black_box, criterion_group, criterion_main, Criterion};
use geonum::*;
use std::f64::consts::PI;

// Simulated tensor-based implementation of geometric algebra
struct Tensor3D {
    // A rank-3 tensor for 3D geometric algebra in a naive implementation
    data: Vec<Vec<Vec<f64>>>,
}

impl Tensor3D {
    // Create a new tensor with n×n×n dimensions
    fn new(n: usize) -> Self {
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let mut matrix = Vec::with_capacity(n);
            for _ in 0..n {
                let row = vec![0.0; n];
                matrix.push(row);
            }
            data.push(matrix);
        }
        Self { data }
    }

    // Initialize tensor with random values for benchmark purposes
    fn initialize(&mut self) {
        let n = self.data.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // Assign some values for benchmarking
                    self.data[i][j][k] = (i * j * k) as f64 % 10.0;
                }
            }
        }
    }

    // Multiply with another tensor (simulating geometric product)
    // This is a simplified version for benchmarking; a real implementation
    // would be more complex and involve proper tensor contraction
    fn multiply(&self, other: &Tensor3D) -> Tensor3D {
        let n = self.data.len();
        let mut result = Tensor3D::new(n);

        // O(n³) complexity operation
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // Perform multiplication for each element
                    // In a real implementation, this would be a proper tensor contraction
                    // which would be even more expensive
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += self.data[i][j][l] * other.data[l][j][k];
                    }
                    result.data[i][j][k] = sum;
                }
            }
        }

        result
    }
}

fn bench_tensor_product(c: &mut Criterion) {
    let sizes = [2, 3, 4];

    let mut group = c.benchmark_group("Tensor vs Geonum");

    for &size in &sizes {
        // Tensor-based approach
        group.bench_function(format!("tensor_product_size_{}", size), |b| {
            b.iter(|| {
                let mut tensor1 = Tensor3D::new(black_box(size));
                let mut tensor2 = Tensor3D::new(black_box(size));
                tensor1.initialize();
                tensor2.initialize();

                // Perform the tensor product - O(n³) complexity
                black_box(tensor1.multiply(&tensor2))
            })
        });
    }

    // Geonum approach - O(1) complexity regardless of size
    group.bench_function("geonum_product", |b| {
        b.iter(|| {
            let i = Geonum {
                length: black_box(1.0),
                angle: black_box(PI / 2.0),
            }; // i = [1, pi/2]
            let j = Geonum {
                length: black_box(1.0),
                angle: black_box(PI),
            }; // j = [1, pi]
            let k = Geonum {
                length: black_box(1.0),
                angle: black_box(3.0 * PI / 2.0),
            }; // k = [1, 3pi/2]

            // compute the ijk product - O(1) complexity
            let ij = i.mul(&j);
            let ijk = ij.mul(&k);

            black_box(ijk)
        })
    });

    group.finish();
}

fn bench_scaling_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling Comparison");

    // For tensor-based approach, we'll simulate increasing size
    // and measure how the performance scales with size
    for size in [2, 4, 8, 16] {
        group.bench_function(format!("tensor_scaling_{}", size), |b| {
            b.iter(|| {
                let mut tensor = Tensor3D::new(black_box(size));
                tensor.initialize();

                // Simulate operations that scale with tensor size
                let mut sum = 0.0;
                for i in 0..size {
                    for j in 0..size {
                        for k in 0..size {
                            sum += tensor.data[i][j][k];
                        }
                    }
                }

                black_box(sum)
            })
        });
    }

    // For geonum approach, the number of operations remains
    // constant regardless of "size" parameter
    for size in [2, 4, 8, 16] {
        group.bench_function(format!("geonum_scaling_{}", size), |b| {
            b.iter(|| {
                // Regardless of the "size" parameter, geonum operations
                // always take constant time - O(1) complexity
                let v1 = Geonum {
                    length: black_box(size as f64),
                    angle: black_box(PI / 4.0),
                };

                let v2 = Geonum {
                    length: black_box(size as f64),
                    angle: black_box(PI / 3.0),
                };

                // Perform geonum operations - always O(1)
                let product = v1.mul(&v2);
                let dot = v1.dot(&v2);
                let wedge = v1.wedge(&v2);

                black_box((product, dot, wedge))
            })
        });
    }

    group.finish();
}

fn bench_ijk_product(c: &mut Criterion) {
    c.bench_function("ijk_product", |b| {
        b.iter(|| {
            let i = Geonum {
                length: black_box(1.0),
                angle: black_box(PI / 2.0),
            }; // i = [1, pi/2]
            let j = Geonum {
                length: black_box(1.0),
                angle: black_box(PI),
            }; // j = [1, pi]
            let k = Geonum {
                length: black_box(1.0),
                angle: black_box(3.0 * PI / 2.0),
            }; // k = [1, 3pi/2]

            // compute the ijk product
            let ij = i.mul(&j);
            let ijk = ij.mul(&k);

            black_box(ijk)
        })
    });
}

fn bench_dimension_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dimension Operations");

    for &dim_size in &[3, 10, 100] {
        group.bench_function(format!("create_dimension_{}", dim_size), |b| {
            b.iter(|| {
                let dims = Dimensions::new(black_box(dim_size));
                black_box(dims)
            })
        });

        group.bench_function(format!("multivector_{}", dim_size), |b| {
            b.iter(|| {
                let dims = Dimensions::new(black_box(dim_size));
                let indices: Vec<usize> = (0..dim_size).collect();
                let mv = dims.multivector(&indices);
                black_box(mv)
            })
        });
    }

    group.finish();
}

// Benchmark massively higher-dimension GA operations
fn bench_extreme_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Extreme Dimension Comparison");

    // In traditional GA libraries, operations on high-dimensional spaces are
    // exponentially expensive: 10D space = 2^10 = 1024 components per multivector

    // Benchmark for traditional GA in 10D space (extremely memory intensive)
    group.bench_function("traditional_ga_10d_product", |b| {
        b.iter(|| {
            // For 10D, we'd need 2^10 = 1024 components per multivector
            // This is already at the edge of practicality for traditional GA

            // Create two grade-1 vectors in 10D space
            // In a real GA library, each vector would need storage for all 1024 components
            let mut v1 = vec![0.0; 1024];
            let mut v2 = vec![0.0; 1024];

            // Set grade-1 components (indices 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            // These are the basis vectors in 10D space
            let grade1_indices = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
            for &idx in &grade1_indices {
                v1[idx] = 1.0;
                v2[idx] = 1.0;
            }

            // Even just calculating the dot product in 10D is intensive
            // This is O(n) where n is dimension, but full geometric product is O(2^n)
            let mut dot_product = 0.0;
            for &idx in &grade1_indices {
                dot_product += v1[idx] * v2[idx];
            }

            // Just to simulate how expensive a full geometric product would be,
            // we'll do a very limited calculation - just computing a small subset
            // of the full 1024×1024 operation
            let mut result = vec![0.0; 1024];
            for i in 1..20 {
                // Doing only 20 components for benchmarking sanity
                for j in 1..20 {
                    if v1[i] != 0.0 && v2[j] != 0.0 {
                        // XOR gives the resulting basis element index
                        let k = i ^ j;

                        // In real GA, we'd need to compute the sign based on the
                        // number of swaps needed (very complex for high dimensions)
                        let sign = if (i & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };

                        result[k] += sign * v1[i] * v2[j];
                    }
                }
            }

            black_box((dot_product, result))
        })
    });

    // Now scale to an absurd level - 30D space
    // In traditional GA this would be 2^30 > 1 billion components!
    // No library could practically handle this in real-time

    // For traditional GA, we have to use a different approach:
    // Simulate the cost by only computing the theoretical operation count
    group.bench_function("traditional_ga_30d_theoretical", |b| {
        b.iter(|| {
            // Dimension of the space
            let dim = 30;

            // Number of components in a full multivector: 2^dim
            let component_count = 1 << dim; // 2^30 > 1 billion

            // Number of operations needed for a geometric product:
            // O(2^2n) = O(2^60) - literally quadrillions of operations

            // For benchmarking, we'll just simulate the counting of operations
            // to avoid crashing the machine or waiting for days
            let operations_count = 1u64 << (dim * 2);

            // For vectors (grade-1 elements), the cost is lower but still O(2^n)
            let vector_ops = 1u64 << dim;

            // Just return the operation counts to show the theoretical complexity
            black_box((component_count, operations_count, vector_ops))
        })
    });

    // Geonum in 10D - CONSTANT TIME regardless of dimension
    group.bench_function("geonum_10d_product", |b| {
        b.iter(|| {
            // Create a 10D space
            let dims = Dimensions::new(10);

            // Get two basis vectors (could be any of the 10 dimensions)
            let vectors = dims.multivector(&[0, 1]);
            let v1 = vectors[0];
            let v2 = vectors[1];

            // Perform geometric operations - all O(1) time
            let dot = v1.dot(&v2);
            let wedge = v1.wedge(&v2);
            let product = v1.mul(&v2);

            black_box((dot, wedge, product))
        })
    });

    // Geonum in 30D - STILL CONSTANT TIME!
    group.bench_function("geonum_30d_product", |b| {
        b.iter(|| {
            // Create a 30D space (would be impossible to represent fully in traditional GA)
            let dims = Dimensions::new(30);

            // Get two basis vectors
            let vectors = dims.multivector(&[0, 1]);
            let v1 = vectors[0];
            let v2 = vectors[1];

            // Perform geometric operations - all O(1) time regardless of dimension
            let dot = v1.dot(&v2);
            let wedge = v1.wedge(&v2);
            let product = v1.mul(&v2);

            black_box((dot, wedge, product))
        })
    });

    // Geonum in 1000D - STILL CONSTANT TIME!
    // This would be completely impossible with traditional GA (2^1000 components!)
    group.bench_function("geonum_1000d_product", |b| {
        b.iter(|| {
            // Create a 1000D space
            let dims = Dimensions::new(1000);

            // Get two basis vectors
            let vectors = dims.multivector(&[0, 1]);
            let v1 = vectors[0];
            let v2 = vectors[1];

            // Perform geometric operations - all O(1) time regardless of dimension
            let dot = v1.dot(&v2);
            let wedge = v1.wedge(&v2);
            let product = v1.mul(&v2);

            black_box((dot, wedge, product))
        })
    });

    // Go TRULY extreme - geometric operations in 1,000,000D space
    // With traditional GA, this would require 2^1000000 components
    // which is astronomically beyond the number of atoms in the universe
    group.bench_function("geonum_million_d_product", |b| {
        b.iter(|| {
            // Create a 1,000,000D space
            let dims = Dimensions::new(1_000_000);

            // Get two basis vectors
            let vectors = dims.multivector(&[0, 1]);
            let v1 = vectors[0];
            let v2 = vectors[1];

            // Perform geometric operations - STILL O(1) time!
            let dot = v1.dot(&v2);
            let wedge = v1.wedge(&v2);
            let product = v1.mul(&v2);

            black_box((dot, wedge, product))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_product,
    bench_scaling_comparison,
    bench_ijk_product,
    bench_dimension_operations,
    bench_extreme_dimensions
);
criterion_main!(benches);
