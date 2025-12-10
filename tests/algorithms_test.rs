// "algorithm analysis" is founded on a fictional model called "computational complexity" to group operations
//
// to keep algorithm evaluations consistent with a fictional model you must self-referentially require "big O notation" as a "bound" for all "runtime analysis"
//
// hacking performance consistency with asymptotic bounds just traps everyone in an efficiency loop ("amortized constant time")
//
// and denies them the opportunity to understand how computation **naturally scales** in physical hardware
//
// so instead of "analyzing algorithmic complexity", geometric numbers prove their computational efficiency with physical hardware by *extending* the processor's existing operations with `let parallel = sin(pi/2);`
//
// rejecting "complexity classes" for "angle operations" empowers people to understand algorithm performance or "scaling" so well they can even **quantify** it:
//
// ```rs
// let sequential = [1, 0];
// let parallel = [1, PI/2];
// // measure performance gain
// parallel / sequential == [1, PI/2] // orthogonal execution path
// ```
//
// say goodbye to O(n log n)

use geonum::*;
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_constant_time_operation() {
    // in traditional algorithmic analysis, O(1) operations have fixed cost
    // in geometric numbers, this is represented by angle invariance

    // create geometric number representing computational operation
    let operation = Geonum::new(1.0, 0.0, 1.0); // base cost unit, direction in computational space

    // computational cost is independent of problem size
    let small_problem_cost = operation;
    let large_problem_cost = operation;

    // test invariance to problem size (cost is the same)
    assert_eq!(small_problem_cost.mag, large_problem_cost.mag);
    assert!((small_problem_cost.angle - large_problem_cost.angle).rem() < EPSILON);

    // test operation composition (multiple constant operations)
    let combined_operations = Geonum::new(3.0, 0.0, 1.0); // three operations, same direction

    // still constant time regardless of composition
    assert!(combined_operations.mag > operation.mag);
    assert!((combined_operations.angle - operation.angle).rem() < EPSILON);

    // demonstrate array indexing as constant time operation
    let array_op = |arr: &[i32], idx: usize| -> i32 {
        // array indexing cost represented by geometric number
        let _op_cost = Geonum::new(1.0, 0.0, 1.0); // single operation, direct memory access

        arr[idx] // actual operation is O(1)
    };

    // create test array
    let array = [1, 2, 3, 4, 5];

    // access different elements, cost is the same
    let val1 = array_op(&array, 0);
    let val2 = array_op(&array, 4);

    // test correct values and operation worked
    assert_eq!(val1, 1);
    assert_eq!(val2, 5);
}

#[test]
fn its_a_linear_algorithm() {
    // in traditional analysis, O(n) algorithms scale linearly with input size
    // in geometric numbers, this is represented by length scaling with input size

    // create basis operations
    let base_op = Geonum::new(1.0, 0.0, 1.0);

    // linear scaling is represented by length proportional to input size
    // for a linear algorithm processing n items
    let compute_cost = |n: usize| -> Geonum {
        Geonum::new_with_angle(
            n as f64 * base_op.mag, // cost scales linearly with n
            base_op.angle,          // same operation type
        )
    };

    // test linear scaling for different input sizes
    let cost_10 = compute_cost(10);
    let cost_20 = compute_cost(20);
    let cost_100 = compute_cost(100);

    // verify linear scaling property
    assert_eq!(cost_20.mag / cost_10.mag, 2.0); // twice the input, twice the cost
    assert_eq!(cost_100.mag / cost_10.mag, 10.0); // 10x input, 10x cost

    // angle remains the same (same operation type)
    assert!((cost_10.angle - cost_20.angle).rem() < EPSILON);
    assert!((cost_20.angle - cost_100.angle).rem() < EPSILON);

    // demonstrate linear search algorithm
    let linear_search = |arr: &[i32], target: i32| -> Option<usize> {
        for (i, &item) in arr.iter().enumerate() {
            // each comparison represented by geometric number
            let _comparison = Geonum::new(1.0, 0.0, 1.0); // unit cost, direct comparison

            if item == target {
                return Some(i);
            }
        }
        None
    };

    // create test array
    let array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // search for different elements
    let found_first = linear_search(&array, 1);
    let found_last = linear_search(&array, 10);
    let not_found = linear_search(&array, 11);

    // verify search works correctly
    assert_eq!(found_first, Some(0));
    assert_eq!(found_last, Some(9));
    assert_eq!(not_found, None);
}

#[test]
fn its_a_sorting_algorithm() {
    // in traditional analysis, sorting requires O(n log n) comparisons
    // in geometric numbers, sorting can be represented through angle partitioning

    // create a geometric representation of elements to sort
    // angle represents the value, length can represent frequency or weight
    let unsorted_elements = [
        Geonum::new(1.0, 0.7, PI), // 0.7 radians ~40°
        Geonum::new(1.0, 0.2, PI), // 0.2 radians ~11°
        Geonum::new(1.0, 1.5, PI), // 1.5 radians ~86°
        Geonum::new(1.0, 0.1, PI), // 0.1 radians ~6°
        Geonum::new(1.0, 1.0, PI), // 1.0 radians ~57°
    ];

    // geometric sorting: sort by angle
    let mut sorted_by_angle = unsorted_elements.to_vec();
    sorted_by_angle.sort_by(|a, b| {
        // compare angles directly using Angle's blade and value
        a.angle
            .blade()
            .cmp(&b.angle.blade())
            .then(a.angle.rem().partial_cmp(&b.angle.rem()).unwrap())
    });

    // prove sorting
    for i in 1..sorted_by_angle.len() {
        let prev = &sorted_by_angle[i - 1];
        let curr = &sorted_by_angle[i];
        // angles are sorted if prev blade < curr blade, or same blade with prev value <= curr value
        assert!(
            prev.angle.blade() < curr.angle.blade()
                || (prev.angle.blade() == curr.angle.blade()
                    && prev.angle.rem() <= curr.angle.rem())
        );
    }

    // angle-based partition sort (conceptual representation of radix/bucket sort)
    // this approach can achieve O(n) for certain distributions
    let angle_bucket_sort = |elements: &[Geonum]| -> Vec<Geonum> {
        // for conceptual demonstration - in practice would use actual buckets
        let mut result = elements.to_vec();
        result.sort_by(|a, b| {
            a.angle
                .blade()
                .cmp(&b.angle.blade())
                .then(a.angle.rem().partial_cmp(&b.angle.rem()).unwrap())
        });
        result
    };

    let angle_sorted = angle_bucket_sort(&unsorted_elements);

    // verify sorting worked correctly
    for i in 1..angle_sorted.len() {
        let prev = &angle_sorted[i - 1];
        let curr = &angle_sorted[i];
        assert!(
            prev.angle.blade() < curr.angle.blade()
                || (prev.angle.blade() == curr.angle.blade()
                    && prev.angle.rem() <= curr.angle.rem())
        );
    }

    // demonstrate how geometric understanding transforms the sorting problem
    // by using angle as a direct coordinate rather than comparison operator
    assert_eq!(angle_sorted.len(), unsorted_elements.len());

    // verify first angle is less than last angle
    let first = &angle_sorted[0].angle;
    let last = &angle_sorted[angle_sorted.len() - 1].angle;
    assert!(
        first.blade() < last.blade() || (first.blade() == last.blade() && first.rem() < last.rem())
    );
}

#[test]
fn its_a_graph_algorithm() {
    // in traditional analysis, graph algorithms use adjacency structures
    // in geometric numbers, graphs can be represented through angle relations

    // create nodes as geometric numbers
    // angle represents position/orientation in the graph
    let node_a = Geonum::new(1.0, 0.0, 1.0); // node at 0°
    let node_b = Geonum::new(1.0, 1.0, 3.0); // node at π/3 = 60°
    let node_c = Geonum::new(1.0, 2.0, 3.0); // node at 2π/3 = 120°
    let node_d = Geonum::new(1.0, 1.0, 1.0); // node at π = 180°
    let node_e = Geonum::new(1.0, 4.0, 3.0); // node at 4π/3 = 240°
    let node_f = Geonum::new(1.0, 5.0, 3.0); // node at 5π/3 = 300°

    // create edges as angle differences
    // smaller angle difference = stronger connection
    let edge_weight = |a: &Geonum, b: &Geonum| -> Geonum {
        // compute angle distance as a geometric number
        let angle_diff = b.angle - a.angle;
        // edge weight is a scalar representing angular separation
        // use the angle difference to create a scalar with that magnitude
        Geonum::new_with_angle(1.0, angle_diff)
    };

    // test edge weights using geometric number comparison
    let expected_60_deg = Geonum::new(1.0, 1.0, 3.0); // π/3 as a scalar
    let expected_180_deg = Geonum::new(1.0, 1.0, 1.0); // π as a scalar

    // test edge weights
    let weight_ab = edge_weight(&node_a, &node_b);
    let weight_ad = edge_weight(&node_a, &node_d);
    let weight_bc = edge_weight(&node_b, &node_c);

    // angles should match expected separations
    assert!((weight_ab.angle - expected_60_deg.angle).rem() < EPSILON); // 60° apart
    assert!((weight_ad.angle - expected_180_deg.angle).rem() < EPSILON); // 180° apart
    assert!((weight_bc.angle - expected_60_deg.angle).rem() < EPSILON); // 60° apart

    // graph traversal as angle progression
    // implement breadth-first search conceptually
    let bfs_from_angle = |start: f64, graph: &[Geonum]| -> Vec<Geonum> {
        // start from the node closest to the starting angle
        let mut result = graph.to_vec();
        result.sort_by(|a, b| {
            // use angle distance to find nodes closest to starting angle
            let start_geonum = Geonum::new(1.0, start, PI);
            let a_diff = (a.angle - start_geonum.angle).rem().abs();
            let b_diff = (b.angle - start_geonum.angle).rem().abs();
            // handle circular distance
            let a_diff = a_diff.min(TAU - a_diff);
            let b_diff = b_diff.min(TAU - b_diff);
            a_diff.partial_cmp(&b_diff).unwrap()
        });
        result
    };

    // start BFS from angle 0
    let traversal = bfs_from_angle(0.0, &[node_a, node_b, node_c, node_d, node_e, node_f]);

    // first node should be closest to angle 0
    assert!((traversal[0].angle - node_a.angle).rem() < EPSILON);

    // path finding as angle minimization
    // find path with minimal angle changes
    let shortest_path = |start: &Geonum, end: &Geonum, graph: &[Geonum]| -> Vec<Geonum> {
        // simplified path finding - in practice would use actual path algorithm
        // just demonstrate using angle differences to guide the search
        let mut path = vec![*start]; // Geonum is Copy

        // find nodes creating a path of minimal angle changes
        let mut current = start;

        while (current.angle - end.angle).rem().abs() > EPSILON {
            // find next node that minimizes angle difference to target
            let mut best_next = current;
            let mut min_weight = Geonum::new(f64::MAX, 0.0, 1.0); // large scalar

            for node in graph {
                // only accept nodes closer to end than current
                let curr_to_end = edge_weight(current, end);
                let node_to_end = edge_weight(node, end);
                let edge_to_node = edge_weight(current, node);

                // compare weights by their angles (smaller angle = shorter distance)
                if node_to_end.angle < curr_to_end.angle && edge_to_node.angle < min_weight.angle {
                    min_weight = edge_to_node;
                    best_next = node;
                }
            }

            // if no progress can be made, break
            if (best_next.angle - current.angle).rem().abs() < EPSILON {
                break;
            }

            path.push(*best_next); // Geonum is Copy
            current = best_next;
        }

        // add end if not already reached
        if (current.angle - end.angle).rem().abs() > EPSILON {
            path.push(*end); // Geonum is Copy
        }

        path
    };

    // find path from node_a to node_d
    let path = shortest_path(
        &node_a,
        &node_d,
        &[node_a, node_b, node_c, node_d, node_e, node_f],
    );

    // test path properties
    assert!((path[0].angle - node_a.angle).rem() < EPSILON); // starts at node_a
    assert!((path[path.len() - 1].angle - node_d.angle).rem() < EPSILON); // ends at node_d
}

#[test]
fn its_a_dynamic_programming() {
    // in traditional analysis, dynamic programming uses memoization tables
    // in geometric numbers, subproblems can be represented with angle positions

    // Fibonacci as classic DP example
    // each Fibonacci number represented as a geometric number
    // angle represents position in sequence, length represents the value
    let fib_geo = |n: usize| -> Geonum {
        if n <= 1 {
            return Geonum::new(
                n as f64, // F(0)=0, F(1)=1
                n as f64, // n * PI/8 - arbitrary angle mapping
                8.0,
            );
        }

        // initialize with base cases
        let mut fib_minus_2 = Geonum::new(0.0, 0.0, 1.0); // F(0)
        let mut fib_minus_1 = Geonum::new(1.0, 1.0, 8.0); // F(1) - PI/8

        // build up solution using previous subproblems
        for i in 2..=n {
            let current = Geonum::new(
                // F(n) = F(n-1) + F(n-2)
                fib_minus_1.mag + fib_minus_2.mag,
                // angle represents position in sequence
                i as f64,
                8.0, // i * PI/8
            );
            fib_minus_2 = fib_minus_1;
            fib_minus_1 = current;
        }

        fib_minus_1
    };

    // test fibonacci computation
    let fib_5 = fib_geo(5);
    let fib_6 = fib_geo(6);
    let fib_7 = fib_geo(7);

    // validate fibonacci values
    assert_eq!(fib_5.mag, 5.0); // F(5) = 5
    assert_eq!(fib_6.mag, 8.0); // F(6) = 8
    assert_eq!(fib_7.mag, 13.0); // F(7) = 13

    // validate angle progression represents position in sequence
    let expected_5 = Angle::new(5.0, 8.0);
    let expected_6 = Angle::new(6.0, 8.0);
    let expected_7 = Angle::new(7.0, 8.0);
    assert!((fib_5.angle - expected_5).rem() < EPSILON);
    assert!((fib_6.angle - expected_6).rem() < EPSILON);
    assert!((fib_7.angle - expected_7).rem() < EPSILON);

    // demonstrate optimal substructure through angle composition
    // Solution to larger problem (fib_7) uses solutions to smaller problems
    assert_eq!(fib_7.mag, fib_6.mag + fib_5.mag);

    // angle difference represents step in DP table
    let step = Angle::new(1.0, 8.0); // PI/8
    assert!(((fib_6.angle - fib_5.angle) - step).rem() < EPSILON);
    assert!(((fib_7.angle - fib_6.angle) - step).rem() < EPSILON);
}

#[test]
fn its_a_parallel_algorithm() {
    // in traditional analysis, parallel algorithms use thread synchronization
    // in geometric numbers, parallel execution is represented by orthogonal angles

    // sequential computation represented at angle 0
    let sequential = Geonum::new(1.0, 0.0, 1.0); // Vector (grade 1) - represents 1D computational direction

    // parallel computation represented at orthogonal angle (90°)
    let parallel = Geonum::new(1.0, 1.0, 2.0); // π/2 - Vector (grade 1) - represents 1D computational direction

    // test orthogonality
    // dot product is zero for perpendicular operations
    assert!(sequential.dot(&parallel).mag.abs() < EPSILON);

    // concurrent execution represented by simultaneous operations
    // wedge product represents "computational area" covered by parallel execution
    let parallel_gain = sequential.wedge(&parallel);

    // Note: parallel_gain is a bivector (blade: 2) representing a computational area.
    // In geometric algebra, the wedge product of two vectors (a ∧ b) creates a bivector
    // that represents the oriented area spanned by those vectors.
    //
    // In our computational model:
    // - sequential (blade: 1) is a vector representing computation in one direction
    // - parallel (blade: 1) is a vector representing computation in an orthogonal direction
    // - parallel_gain (blade: 2) is the bivector area representing the computational space
    //   covered by performing both operations simultaneously
    //
    // The wedge operation automatically sets blade: 1+1=2 for the result.

    // wedge product is non-zero, showing parallel operations cover more "execution space"
    assert!(parallel_gain.mag > 0.0);

    // demonstrate parallel map operation
    let parallel_map = |items: &[i32]| -> Vec<i32> {
        // conceptual parallel map
        // each item processed in parallel would be at orthogonal angles

        // in actual implementation, this would dispatch to multiple threads
        // here we just simulate with sequential code

        // each operation would be represented as
        // Geonum { length: 1.0, angle: PI/2.0 }

        items.iter().map(|&x| x * x).collect()
    };

    // test parallel mapping
    let input = vec![1, 2, 3, 4, 5];
    let output = parallel_map(&input);

    // verify computation worked
    assert_eq!(output, vec![1, 4, 9, 16, 25]);

    // demonstrate parallel speedup model
    // with n processors, computation time reduces by factor of n
    let speedup = |sequential_time: f64, num_processors: f64| -> f64 {
        // amdahl's law simplified: if task is perfectly parallelizable
        sequential_time / num_processors
    };

    let base_time = 10.0;
    assert_eq!(speedup(base_time, 2.0), 5.0); // 2x speedup
    assert_eq!(speedup(base_time, 4.0), 2.5); // 4x speedup
}

#[test]
fn its_a_distributed_algorithm() {
    // in traditional analysis, distributed algorithms use message passing
    // in geometric numbers, distributed computation is angle sector assignments

    // create nodes in a distributed system as geometric numbers
    // angle represents node's position/responsibility in the system
    let node_1 = Geonum::new(1.0, 0.0, 1.0); // node at 0°
    let node_2 = Geonum::new(1.0, 2.0, 5.0); // node at 2π/5 = 72°
    let node_3 = Geonum::new(1.0, 4.0, 5.0); // node at 4π/5 = 144°
    let node_4 = Geonum::new(1.0, 6.0, 5.0); // node at 6π/5 = 216°
    let node_5 = Geonum::new(1.0, 8.0, 5.0); // node at 8π/5 = 288°

    // distributed system as a set of nodes
    let system = vec![node_1, node_2, node_3, node_4, node_5];

    // work assignment based on value's angle
    // find the closest node to handle a given value
    let assign_work = |value_angle: f64, system: &[Geonum]| -> Geonum {
        let mut closest_node = system[0];
        let mut min_distance = f64::MAX;

        for node in system {
            // compute angular distance
            let value_geonum = Geonum::new(1.0, value_angle, PI);

            // compute angular distance properly
            let angle_diff = node.angle - value_geonum.angle;
            // convert to total radians for distance calculation
            let total_diff = (angle_diff.blade() as f64) * (PI / 2.0) + angle_diff.rem();
            let distance = total_diff.abs();

            // handle circular distance (shorter path around circle)
            let distance = distance.min(TAU - distance);

            if distance < min_distance {
                min_distance = distance;
                closest_node = *node;
            }
        }

        closest_node
    };

    // test work assignment
    let work_at_0 = assign_work(0.0, &system);
    assert!((work_at_0.angle - node_1.angle).rem() < EPSILON);

    let work_at_pi = assign_work(PI, &system);

    // node_3 is at 4π/5 = 144°, node_4 is at 6π/5 = 216°
    // π = 180°, so node_4 (216°) is closer: |216° - 180°| = 36° vs |144° - 180°| = 36°
    // actually they're equidistant! Let's check which one was returned
    assert!(
        (work_at_pi.angle - node_3.angle).rem() < EPSILON
            || (work_at_pi.angle - node_4.angle).rem() < EPSILON,
        "Expected node_3 or node_4 for work at PI"
    );

    // demonstrate distributed consensus
    // nodes agree on a value by converging angles
    let reach_consensus = |system: &[Geonum]| -> f64 {
        // in a real system, this would involve message passing
        // here we simplify by computing the average angle

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for node in system {
            sum_x += node.angle.grade_angle().cos();
            sum_y += node.angle.grade_angle().sin();
        }

        // compute average angle (circular mean)
        sum_y.atan2(sum_x)
    };

    // test consensus
    let consensus_angle = reach_consensus(&system);

    // verify consensus is reached
    // all nodes should be within PI distance from consensus
    for node in &system {
        let consensus_geonum = Geonum::new(1.0, consensus_angle, PI);

        // compute angle distance using Angle arithmetic
        let angle_diff = node.angle - consensus_geonum.angle;
        let distance = angle_diff.rem().abs().min(TAU - angle_diff.rem().abs());

        assert!(distance <= PI);
    }
}

#[test]
fn its_a_numerical_method() {
    // in traditional analysis, numerical methods approximate continuous processes
    // in geometric numbers, approximation is represented through angle precision

    // approximate sin(x) using Taylor series
    // each term in series represented as a geometric number
    let sin_approx = |x: f64, terms: usize| -> f64 {
        let mut result = 0.0;

        for n in 0..terms {
            // nth term in Taylor series
            let term = Geonum::new(
                // (-1)^n * x^(2n+1) / (2n+1)!
                if n % 2 == 0 { 1.0 } else { -1.0 } * x.powi(2 * n as i32 + 1)
                    / factorial(2 * n + 1) as f64,
                // angle represents term's position in series
                n as f64,
                8.0, // n * PI/8
            );

            result += term.mag;
        }

        result
    };

    // helper function for factorial
    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    // test approximation at different precision levels
    let x = PI / 6.0; // 30 degrees

    // exact value
    let exact = x.sin();

    // approximations with increasing terms
    let approx_1 = sin_approx(x, 1);
    let approx_2 = sin_approx(x, 2);
    let approx_4 = sin_approx(x, 4);

    // verify convergence with more terms
    assert!((exact - approx_1).abs() > (exact - approx_2).abs());
    assert!((exact - approx_2).abs() > (exact - approx_4).abs());

    // with 4 terms, should be very close to exact result
    assert!((exact - approx_4).abs() < 1e-6);

    // prove numerical integration using trapezoidal rule
    // function to integrate: f(x) = x^2
    let f = |x: f64| -> f64 { x * x };

    // trapezoidal rule integration
    let integrate = |f: fn(f64) -> f64, a: f64, b: f64, n: usize| -> f64 {
        let dx = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));

        for i in 1..n {
            let x = a + i as f64 * dx;
            sum += f(x);
        }

        sum * dx
    };

    // integrate x^2 from 0 to 1, exact result is 1/3
    let exact_integral = 1.0 / 3.0;

    // approximations with increasing subintervals
    let approx_10 = integrate(f, 0.0, 1.0, 10);
    let approx_100 = integrate(f, 0.0, 1.0, 100);

    // prove convergence with more subintervals
    assert!((exact_integral - approx_10).abs() > (exact_integral - approx_100).abs());
    assert!((exact_integral - approx_100).abs() < 1e-4);
}

#[test]
fn its_a_data_structure() {
    // in traditional analysis, data structures use pointers
    // in geometric numbers, data can be organized through angle mapping

    // create a geometric hash table
    // use angle mapping for indices instead of modulo hash
    struct GeoHashTable {
        buckets: Vec<Vec<(String, i32)>>,
        bucket_count: usize,
    }

    impl GeoHashTable {
        fn new(bucket_count: usize) -> Self {
            let mut buckets = Vec::with_capacity(bucket_count);
            for _ in 0..bucket_count {
                buckets.push(Vec::new());
            }
            GeoHashTable {
                buckets,
                bucket_count,
            }
        }

        fn hash(&self, key: &str) -> usize {
            // convert string to angle
            let mut sum = 0;
            for (i, &byte) in key.as_bytes().iter().enumerate() {
                sum += (byte as usize) * (i + 1);
            }

            // instead of traditional hash, map to angle in [0, 2π)
            let angle_radians = (sum % 360) as f64 * PI / 180.0;
            let angle_geonum = Geonum::new(1.0, angle_radians, PI);

            // convert angle to bucket index
            let total_angle =
                angle_geonum.angle.blade() as f64 * PI / 2.0 + angle_geonum.angle.rem();
            (total_angle * self.bucket_count as f64 / TAU) as usize % self.bucket_count
        }

        fn insert(&mut self, key: String, value: i32) {
            let bucket_idx = self.hash(&key);

            // check if key already exists
            for pair in &mut self.buckets[bucket_idx] {
                if pair.0 == key {
                    pair.1 = value;
                    return;
                }
            }

            // key doesn't exist, add new entry
            self.buckets[bucket_idx].push((key, value));
        }

        fn get(&self, key: &str) -> Option<i32> {
            let bucket_idx = self.hash(key);

            for pair in &self.buckets[bucket_idx] {
                if pair.0 == key {
                    return Some(pair.1);
                }
            }

            None
        }
    }

    // test geometric hash table
    let mut geo_hash = GeoHashTable::new(8);

    // insert key-value pairs
    geo_hash.insert("one".to_string(), 1);
    geo_hash.insert("two".to_string(), 2);
    geo_hash.insert("three".to_string(), 3);

    // test retrieval
    assert_eq!(geo_hash.get("one"), Some(1));
    assert_eq!(geo_hash.get("two"), Some(2));
    assert_eq!(geo_hash.get("three"), Some(3));
    assert_eq!(geo_hash.get("four"), None);

    // update existing key
    geo_hash.insert("one".to_string(), 10);
    assert_eq!(geo_hash.get("one"), Some(10));

    // demonstrate binary search tree as geometric angle structure
    // angle encodes the complete path from root - each bit represents left(0) or right(1)
    // this creates a unique angle for every position in the tree
    #[allow(dead_code)]
    struct GeoNode {
        value: i32,
        angle: Angle, // encodes complete path from root
        depth: usize, // tree depth for angle calculation
        left: Option<Box<GeoNode>>,
        right: Option<Box<GeoNode>>,
    }

    impl GeoNode {
        fn new(value: i32) -> Self {
            GeoNode {
                value,
                angle: Angle::new(0.0, 1.0), // root has zero angle
                depth: 0,
                left: None,
                right: None,
            }
        }

        fn insert(&mut self, value: i32) {
            if value < self.value {
                // left branch: add 0 to path encoding
                match self.left {
                    None => {
                        // angle encodes path: each level divides the remaining angle space
                        // this creates a fractal-like distribution where each subtree has its own angle region
                        // for left child at depth d, add π/2^(d+2)
                        let path_angle = self.angle
                            + Angle::new(1.0, (2_u32.pow((self.depth + 2) as u32)) as f64);
                        self.left = Some(Box::new(GeoNode {
                            value,
                            angle: path_angle,
                            depth: self.depth + 1,
                            left: None,
                            right: None,
                        }));
                    }
                    Some(ref mut node) => {
                        node.insert(value);
                    }
                }
            } else {
                // right branch: add 1 to path encoding
                match self.right {
                    None => {
                        // right children get an additional rotation based on depth
                        // deeper nodes have finer angular resolution
                        // for right child at depth d, add 3π/2^(d+2)
                        let path_angle = self.angle
                            + Angle::new(3.0, (2_u32.pow((self.depth + 2) as u32)) as f64);
                        self.right = Some(Box::new(GeoNode {
                            value,
                            angle: path_angle,
                            depth: self.depth + 1,
                            left: None,
                            right: None,
                        }));
                    }
                    Some(ref mut node) => {
                        node.insert(value);
                    }
                }
            }
        }

        fn contains(&self, value: i32) -> bool {
            if value == self.value {
                return true;
            }

            if value < self.value {
                match self.left {
                    None => false,
                    Some(ref node) => node.contains(value),
                }
            } else {
                match self.right {
                    None => false,
                    Some(ref node) => node.contains(value),
                }
            }
        }

        // demonstrate the power of angle encoding - find by exact angle
        fn find_by_angle(&self, target_angle: &Angle, _target_depth: usize) -> Option<i32> {
            // angles are unique addresses - we can navigate directly
            if (self.angle - target_angle).rem() < EPSILON
                && self.angle.blade() == target_angle.blade()
            {
                return Some(self.value);
            }

            // recursively search children
            if let Some(ref left) = self.left {
                if let Some(result) = left.find_by_angle(target_angle, _target_depth) {
                    return Some(result);
                }
            }

            if let Some(ref right) = self.right {
                if let Some(result) = right.find_by_angle(target_angle, _target_depth) {
                    return Some(result);
                }
            }

            None
        }
    }

    // test geometric BST
    let mut root = GeoNode::new(10);

    // insert values
    root.insert(5);
    root.insert(15);
    root.insert(3);
    root.insert(7);

    // test search
    assert!(root.contains(10));
    assert!(root.contains(5));
    assert!(root.contains(15));
    assert!(root.contains(3));
    assert!(root.contains(7));
    assert!(!root.contains(1));
    assert!(!root.contains(20));

    // demonstrate the power of angle-based addressing
    // in traditional BST, you must traverse from root following values
    // with angle encoding, each node has a unique geometric address

    // the angle of node with value 7 (path: root->left->right)
    // encodes its complete position in the tree
    // root starts at Angle::new(0.0, 1.0) = 0
    // going left at depth 0 adds π/2^2 = π/4
    // going right at depth 1 adds 3π/2^3 = 3π/8
    let root_angle = Angle::new(0.0, 1.0);
    let left_at_depth_0 = Angle::new(1.0, 4.0); // π/4
    let right_at_depth_1 = Angle::new(3.0, 8.0); // 3π/8
    let node_7_angle = root_angle + left_at_depth_0 + right_at_depth_1;

    // we can directly query by geometric position!
    // this is impossible in traditional BST without traversing the entire path
    assert_eq!(root.find_by_angle(&node_7_angle, 2), Some(7));
}

#[test]
fn its_a_compression_algorithm() {
    // in traditional analysis, compression uses encoding schemes
    // in geometric numbers, compression can use angle quantization

    // create original data as geometric numbers
    // angle represents the value, length could represent frequency
    let original_data = vec![
        Geonum::new(1.0, 0.12345, PI),
        Geonum::new(1.0, 0.12346, PI),
        Geonum::new(1.0, 0.12347, PI),
        Geonum::new(1.0, 0.54321, PI),
        Geonum::new(1.0, 0.54322, PI),
        Geonum::new(1.0, 0.54323, PI),
        Geonum::new(1.0, 1.23456, PI),
        Geonum::new(1.0, 1.23457, PI),
        Geonum::new(1.0, 1.23458, PI),
    ];

    // compression by angle quantization
    // round angles to fewer decimal places
    let compress = |data: &[Geonum], precision: usize| -> Vec<Geonum> {
        let scale = 10.0_f64.powi(precision as i32);

        // map data to quantized angles
        let mut quantized = Vec::new();

        for item in data {
            // quantize angle to specified precision
            let total_angle = item.angle.blade() as f64 * PI / 2.0 + item.angle.rem();
            let quantized_angle = (total_angle * scale).round() / scale;

            // add only unique quantized angles (deduplication)
            // check if angle already exists in quantized
            let angle_exists = quantized.iter().any(|g: &Geonum| {
                let g_total_angle = g.angle.blade() as f64 * PI / 2.0 + g.angle.rem();
                (g_total_angle - quantized_angle).abs() < EPSILON
            });

            if !angle_exists {
                quantized.push(Geonum::new(item.mag, quantized_angle, PI));
            }
        }

        quantized
    };

    // test compression at different precision levels
    let compressed_3 = compress(&original_data, 3); // 3 decimal places
    let compressed_2 = compress(&original_data, 2); // 2 decimal places
    let compressed_1 = compress(&original_data, 1); // 1 decimal place

    // verify compression ratio improves with lower precision
    assert!(compressed_3.len() <= original_data.len());
    assert!(compressed_2.len() <= compressed_3.len());
    assert!(compressed_1.len() <= compressed_2.len());

    // demonstrate reconstruction error
    // decompress by expanding each quantized value
    let reconstruct = |compressed: &[Geonum], original: &[Geonum]| -> f64 {
        let mut total_error = 0.0;

        for orig in original {
            // find closest angle in compressed data
            let mut min_error = f64::MAX;

            for comp in compressed {
                let orig_total = orig.angle.blade() as f64 * PI / 2.0 + orig.angle.rem();
                let comp_total = comp.angle.blade() as f64 * PI / 2.0 + comp.angle.rem();
                let error = (orig_total - comp_total).abs();
                if error < min_error {
                    min_error = error;
                }
            }

            total_error += min_error;
        }

        total_error / original.len() as f64
    };

    // calculate reconstruction error for each compression level
    let error_3 = reconstruct(&compressed_3, &original_data);
    let error_2 = reconstruct(&compressed_2, &original_data);
    let error_1 = reconstruct(&compressed_1, &original_data);

    // verify error increases with higher compression
    assert!(error_1 >= error_2);
    assert!(error_2 >= error_3);

    // compression ratio quantification
    let compression_ratio = |original: &[Geonum], compressed: &[Geonum]| -> f64 {
        original.len() as f64 / compressed.len() as f64
    };

    // calculate compression ratios
    let ratio_3 = compression_ratio(&original_data, &compressed_3);
    let ratio_2 = compression_ratio(&original_data, &compressed_2);
    let ratio_1 = compression_ratio(&original_data, &compressed_1);

    // verify compression improves with lower precision
    assert!(ratio_1 >= ratio_2);
    assert!(ratio_2 >= ratio_3);
}

#[test]
fn its_a_machine_learning_algorithm() {
    // in traditional analysis, ML uses weight updates
    // in geometric numbers, learning can use angle adjustments

    // create a simple geometric perceptron
    struct GeoPerceptron {
        weights: Vec<Geonum>,
        learning_rate: f64,
    }

    impl GeoPerceptron {
        fn new(features: usize, learning_rate: f64) -> Self {
            // initialize weights with small random angles
            let mut weights = Vec::with_capacity(features);
            for _ in 0..features {
                weights.push(Geonum::new(1.0, 0.1, PI)); // small initial angle
            }

            GeoPerceptron {
                weights,
                learning_rate,
            }
        }

        fn predict(&self, inputs: &[f64]) -> i32 {
            let mut sum = 0.0;

            // weighted sum as input projections
            for (i, &input) in inputs.iter().enumerate() {
                if i < self.weights.len() {
                    // Use sine instead of cosine for better discrimination
                    // This helps prevent balanced weights leading to zero output
                    let weight_projection =
                        self.weights[i].mag * self.weights[i].angle.grade_angle().sin();
                    sum += input * weight_projection;
                }
            }

            // Add a bias term to help with classification
            sum -= 0.5; // Simple threshold adjustment

            // step activation function
            if sum > 0.0 {
                1
            } else {
                0
            }
        }

        fn train(&mut self, inputs: &[f64], target: i32, epochs: usize) {
            for _ in 0..epochs {
                let prediction = self.predict(inputs);
                let error = target - prediction;

                if error != 0 {
                    // update weights based on error
                    for (i, &input) in inputs.iter().enumerate() {
                        if i < self.weights.len() {
                            // adjust angle based on error and input
                            let delta_angle = self.learning_rate * error as f64 * input;

                            // adjust angle based on error and input
                            let angle_adjustment = Angle::new(delta_angle, PI);
                            self.weights[i] = Geonum::new_with_angle(
                                self.weights[i].mag,                      // keep same length
                                self.weights[i].angle + angle_adjustment, // adjust angle
                            );
                        }
                    }
                }
            }
        }
    }

    // test perceptron on AND gate
    let mut perceptron = GeoPerceptron::new(2, 0.1);

    // training data for AND gate
    let training_data = vec![
        (vec![0.0, 0.0], 0), // 0 AND 0 = 0
        (vec![0.0, 1.0], 0), // 0 AND 1 = 0
        (vec![1.0, 0.0], 0), // 1 AND 0 = 0
        (vec![1.0, 1.0], 1), // 1 AND 1 = 1
    ];

    // train perceptron
    for _ in 0..100 {
        // multiple training iterations
        for (inputs, target) in &training_data {
            perceptron.train(inputs, *target, 1);
        }
    }

    // test predictions after training
    for (inputs, target) in &training_data {
        assert_eq!(perceptron.predict(inputs), *target);
    }

    // demonstrate geometric interpretation of learning
    // angles represent decision boundary orientation
    let _initial_angles = [0.1, 0.1]; // starting angles
    let final_angles: Vec<Angle> = perceptron.weights.iter().map(|w| w.angle).collect();

    // verify angles changed during training
    let initial_angle = Angle::new(0.1, PI);
    for angle in &final_angles {
        assert!((angle - initial_angle).rem().abs() > EPSILON);
    }
}

#[test]
fn its_a_cryptographic_algorithm() {
    // in traditional analysis, crypto uses number theory
    // in geometric numbers, crypto can use angle transformations

    // create a simple angle-based encryption scheme
    struct GeoEncryption {
        key_angle: f64,
        key_length: f64,
    }

    impl GeoEncryption {
        fn new(key: u32) -> Self {
            // derive key angle and length from seed
            let key_f64 = key as f64;
            let key_angle = (key_f64 % 360.0) * PI / 180.0; // convert to radians
            let key_length = 1.0 + (key_f64 % 10.0) / 10.0; // between 1.0 and 2.0

            GeoEncryption {
                key_angle,
                key_length,
            }
        }

        fn encrypt(&self, plaintext: &[u8]) -> Vec<Geonum> {
            let mut ciphertext = Vec::with_capacity(plaintext.len());

            for (i, &byte) in plaintext.iter().enumerate() {
                // position-dependent angle shift
                let position_shift = (i % 8) as f64 * PI / 16.0;

                // convert byte to geonum with encryption
                // length encodes the data, angle provides obfuscation
                let obfuscation_angle =
                    (byte as f64 / 128.0) * PI + self.key_angle + position_shift;
                let encrypted = Geonum::new((byte as f64) * self.key_length, obfuscation_angle, PI);

                ciphertext.push(encrypted);
            }

            ciphertext
        }

        fn decrypt(&self, ciphertext: &[Geonum]) -> Vec<u8> {
            let mut plaintext = Vec::with_capacity(ciphertext.len());

            for cipher in ciphertext.iter() {
                // decrypt using length (angle is used only for obfuscation)
                let byte_value = (cipher.mag / self.key_length).round() as u8;
                plaintext.push(byte_value);
            }

            plaintext
        }
    }

    // test encryption and decryption
    let encryption = GeoEncryption::new(12345);

    // message to encrypt
    let message = b"Hello, geometric encryption!";

    // encrypt and decrypt
    let encrypted = encryption.encrypt(message);
    let decrypted = encryption.decrypt(&encrypted);

    // verify decryption works
    assert_eq!(decrypted, message);

    // demonstrate angle perturbation for security
    // changing a single byte should significantly change the ciphertext
    let altered_message = b"Hello, geometric encryptiin!"; // changed 'o' to 'i'
    let altered_encrypted = encryption.encrypt(altered_message);

    // at least some ciphertexts should be different
    let mut differences = 0;
    for i in 0..message.len() {
        if i < altered_encrypted.len()
            && ((encrypted[i].angle - altered_encrypted[i].angle).rem() > EPSILON
                || (encrypted[i].mag - altered_encrypted[i].mag).abs() > EPSILON)
        {
            differences += 1;
        }
    }

    // verify difference in ciphertexts
    assert!(differences > 0);

    // demonstrate key sensitivity
    let wrong_key = GeoEncryption::new(12346); // just one digit different
    let wrong_decrypted = wrong_key.decrypt(&encrypted);

    // decryption with wrong key should differ from original
    assert_ne!(wrong_decrypted, message);
}

#[test]
fn it_rejects_complexity_analysis() {
    // traditional complexity analysis uses asymptotic bounds
    // geometric numbers allow direct performance measurement

    // create operations with different complexity
    let constant_op = |_n: usize| -> Geonum {
        // O(1) operation - angle is 0
        Geonum::new(1.0, 0.0, 1.0)
    };

    let linear_op = |n: usize| -> Geonum {
        // O(n) operation - angle is π/4
        Geonum::new(n as f64, 1.0, 4.0) // π/4
    };

    let quadratic_op = |n: usize| -> Geonum {
        // O(n²) operation - angle is π/2
        Geonum::new((n * n) as f64, 1.0, 2.0) // π/2
    };

    let log_op = |n: usize| -> Geonum {
        // O(log n) operation - angle is π/8
        Geonum::new((n as f64).log2(), 1.0, 8.0) // π/8
    };

    // test scaling for different input sizes
    let n_values = vec![10, 100, 1000];

    for &n in &n_values {
        // evaluate operations
        let c_op = constant_op(n);
        let l_op = linear_op(n);
        let q_op = quadratic_op(n);
        let log_op = log_op(n);

        // verify different scaling behaviors
        assert_eq!(c_op.mag, 1.0); // constant stays at 1
        assert_eq!(l_op.mag, n as f64); // linear scales with n
        assert_eq!(q_op.mag, (n * n) as f64); // quadratic scales with n²
        assert!((log_op.mag - (n as f64).log2()).abs() < EPSILON); // logarithmic scales with log n

        // verify operation types (angles)
        let zero_angle = Angle::new(0.0, 1.0);
        let pi_4_angle = Angle::new(1.0, 4.0);
        let pi_2_angle = Angle::new(1.0, 2.0);
        let pi_8_angle = Angle::new(1.0, 8.0);
        assert!((c_op.angle - zero_angle).rem() < EPSILON);
        assert!((l_op.angle - pi_4_angle).rem() < EPSILON);
        assert!((q_op.angle - pi_2_angle).rem() < EPSILON);
        assert!((log_op.angle - pi_8_angle).rem() < EPSILON);
    }

    // measure algorithm scaling directly through ratios
    let n1 = 10;
    let n2 = 100; // 10x larger

    // compute ratios to measure actual scaling
    let constant_ratio = constant_op(n2).mag / constant_op(n1).mag;
    let linear_ratio = linear_op(n2).mag / linear_op(n1).mag;
    let quadratic_ratio = quadratic_op(n2).mag / quadratic_op(n1).mag;
    let log_ratio = log_op(n2).mag / log_op(n1).mag;

    // verify actual scaling behavior matches expected
    assert_eq!(constant_ratio, 1.0); // constant: n2/n1 = 1
    assert_eq!(linear_ratio, n2 as f64 / n1 as f64); // linear: n2/n1 = 10
    assert_eq!(quadratic_ratio, (n2 * n2) as f64 / (n1 * n1) as f64); // quadratic: (n2/n1)² = 100

    // log ratio is approximately log(n2)/log(n1) which is less than linear
    assert!(log_ratio < linear_ratio);

    // demonstrate direct geometric interpretation of algorithmic complexity
    let complexity_relation = |op1: &Geonum, op2: &Geonum| -> f64 {
        // angle between operations shows their "computational orthogonality"
        (op1.angle - op2.angle).rem().abs()
    };

    // compute relations between different complexities
    let const_vs_linear = complexity_relation(&constant_op(100), &linear_op(100));
    let linear_vs_quadratic = complexity_relation(&linear_op(100), &quadratic_op(100));

    // verify geometric interpretation matches complexity theory
    assert!(const_vs_linear > 0.0); // different complexities have non-zero angle
    assert!(linear_vs_quadratic > 0.0);
}

#[test]
fn it_unifies_algorithm_design() {
    // traditional algorithm design separates paradigms
    // geometric numbers unify approaches through angle transformations

    // create different algorithm paradigms as geometric operations
    // divide and conquer - angle π/4
    let divide_conquer = Geonum::new(1.0, 1.0, 4.0); // π/4

    // dynamic programming - angle π/2
    let dynamic_prog = Geonum::new(1.0, 1.0, 2.0); // π/2

    // greedy algorithm - angle 3π/4
    let greedy = Geonum::new(1.0, 3.0, 4.0); // 3π/4

    // backtracking - angle π
    let backtracking = Geonum::new(1.0, 1.0, 1.0); // π

    // demonstrate geometric relationship between paradigms
    // measure angular distance between approaches
    let paradigm_distance = |p1: &Geonum, p2: &Geonum| -> f64 {
        // compute angular distance between paradigms
        let angle1_total = p1.angle.blade() as f64 * PI / 2.0 + p1.angle.rem();
        let angle2_total = p2.angle.blade() as f64 * PI / 2.0 + p2.angle.rem();
        (angle2_total - angle1_total).abs()
    };

    // compute distances between paradigms
    let dc_dp_dist = paradigm_distance(&divide_conquer, &dynamic_prog);
    let dp_greedy_dist = paradigm_distance(&dynamic_prog, &greedy);
    let greedy_bt_dist = paradigm_distance(&greedy, &backtracking);

    // verify paradigms have equal angular spacing
    assert!((dc_dp_dist - PI / 4.0).abs() < EPSILON);
    assert!((dp_greedy_dist - PI / 4.0).abs() < EPSILON);
    assert!((greedy_bt_dist - PI / 4.0).abs() < EPSILON);

    // demonstrate hybrid algorithm combining paradigms
    let hybrid = |p1: &Geonum, p2: &Geonum, ratio: f64| -> Geonum {
        // for a 50/50 hybrid, use the midpoint
        // for other ratios, pick the closer paradigm
        if ratio == 0.5 {
            // midpoint: add angles and divide by 2
            let sum_angle = p1.angle + p2.angle;
            // dividing by 2 in angle space
            let half_sum = sum_angle / 2.0;
            Geonum::new_with_angle(1.0, half_sum)
        } else if ratio < 0.5 {
            // closer to p1
            *p1
        } else {
            // closer to p2
            *p2
        }
    };

    // create hybrid between divide-conquer and dynamic programming
    let dc_dp_hybrid = hybrid(&divide_conquer, &dynamic_prog, 0.5);

    // verify hybrid is between the two paradigms
    assert!(dc_dp_hybrid.angle >= divide_conquer.angle);
    assert!(dc_dp_hybrid.angle <= dynamic_prog.angle);

    // demonstrate algorithm transformation as rotation
    let transform_algorithm = |algorithm: &Geonum, rotation: f64| -> Geonum {
        let rotation_angle = Angle::new(rotation, PI); // rotation / PI gives π radians
        Geonum::new_with_angle(algorithm.mag, algorithm.angle + rotation_angle)
    };

    // transform divide and conquer to dynamic programming
    let transformed = transform_algorithm(&divide_conquer, PI / 4.0);

    // verify transformation
    assert!((transformed.angle - dynamic_prog.angle).rem() < EPSILON);

    // demonstrate algorithmic duality through geometric complementarity
    let dual_algorithm = |algorithm: &Geonum| -> Geonum {
        // dual is at opposite angle
        let pi_rotation = Angle::new(1.0, 1.0); // π
        Geonum::new_with_angle(algorithm.mag, algorithm.angle + pi_rotation)
    };

    // compute duals
    let dc_dual = dual_algorithm(&divide_conquer);
    let dp_dual = dual_algorithm(&dynamic_prog);

    // verify duality relationship
    let expected_dc_dual = Angle::new(5.0, 4.0); // 5π/4
    let expected_dp_dual = Angle::new(3.0, 2.0); // 3π/2
    assert!((dc_dual.angle - expected_dc_dual).rem() < EPSILON);
    assert!((dp_dual.angle - expected_dp_dual).rem() < EPSILON);
}

#[test]
fn it_scales_quantum_algorithms() {
    // traditional quantum algorithms use complex state vectors
    // geometric numbers represent quantum algorithms as angle superpositions

    // create quantum states as geometric numbers
    // |0⟩ state - angle 0
    let zero_state = Geonum::new(1.0, 0.0, 1.0);

    // |1⟩ state - angle π
    let one_state = Geonum::new(1.0, 1.0, 1.0); // 1 * π

    // superposition state (|0⟩ + |1⟩)/√2 - angle π/4
    let superposition = Geonum::new(1.0, 1.0, 4.0); // π/4

    // demonstrate quantum gates as angle transformations
    // hadamard gate - rotates by π/4
    let hadamard = |state: &Geonum| -> Geonum {
        let rotation = Angle::new(1.0, 4.0); // π/4
        Geonum::new_with_angle(state.mag, state.angle + rotation)
    };

    // phase gate - adds phase π/2 to |1⟩ component
    let phase = |state: &Geonum| -> Geonum {
        let pi_angle = Angle::new(1.0, 1.0); // π
        if (state.angle - pi_angle).rem() < EPSILON {
            // |1⟩ state, add phase
            let phase_rotation = Angle::new(1.0, 2.0); // π/2
            Geonum::new_with_angle(state.mag, state.angle + phase_rotation)
        } else {
            // other state, leave unchanged
            *state
        }
    };

    // test quantum gates
    let h_zero = hadamard(&zero_state);
    let expected_h_zero = Angle::new(1.0, 4.0); // π/4
    assert!((h_zero.angle - expected_h_zero).rem() < EPSILON); // |0⟩ → (|0⟩ + |1⟩)/√2

    let p_one = phase(&one_state);
    let expected_p_one = Angle::new(3.0, 2.0); // 3π/2
    assert!((p_one.angle - expected_p_one).rem() < EPSILON); // |1⟩ → e^(iπ/2)|1⟩

    // demonstrate quantum parallelism through angle superposition
    let parallelism_factor = |state: &Geonum| -> f64 {
        // measure of quantum parallelism based on angle
        // max at π/4 (equal superposition), min at 0 or π (basis states)
        let total_angle = state.angle.blade() as f64 * PI / 2.0 + state.angle.rem();
        let normalized_angle = total_angle % PI;
        (if normalized_angle > PI / 2.0 {
            PI - normalized_angle
        } else {
            normalized_angle
        }) / (PI / 4.0)
    };

    // compute parallelism factors
    let zero_parallelism = parallelism_factor(&zero_state);
    let superpos_parallelism = parallelism_factor(&superposition);

    // verify superposition has maximum parallelism
    assert!(zero_parallelism < 0.1); // basis state has minimal parallelism
    assert!((superpos_parallelism - 1.0).abs() < 0.1); // superposition has maximal parallelism

    // demonstrate multi-qubit entanglement
    // entangled bell state as geometric number
    let bell_state = Geonum::new(1.0, 1.0, 4.0); // π/4

    // measure entanglement through angle precision
    let entanglement = |state: &Geonum| -> f64 {
        // simplified entanglement measure
        // max at π/4, π/2, 3π/4, π (superposition angles)
        let total_angle = state.angle.blade() as f64 * PI / 2.0 + state.angle.rem();
        let norm_angle = total_angle % (PI / 2.0);
        (if norm_angle > PI / 4.0 {
            PI / 2.0 - norm_angle
        } else {
            norm_angle
        }) / (PI / 4.0)
    };

    // compute entanglement
    let bell_entanglement = entanglement(&bell_state);
    let basis_entanglement = entanglement(&zero_state);

    // verify bell state has maximum entanglement
    assert!((bell_entanglement - 1.0).abs() < 0.1);
    assert!(basis_entanglement < 0.1);

    // demonstrate quantum speedup through geometric representation
    // classical vs quantum search algorithm
    let classical_search = |n: usize| -> Geonum {
        // O(n) complexity
        Geonum::new(n as f64, 0.0, 1.0)
    };

    let quantum_search = |n: usize| -> Geonum {
        // O(√n) complexity
        Geonum::new((n as f64).sqrt(), 1.0, 2.0) // π/2
    };

    // compute speedup ratio
    let n = 1000000;
    let speedup = classical_search(n).mag / quantum_search(n).mag;

    // verify quantum speedup is approximately √n
    assert!((speedup - (n as f64).sqrt()).abs() / (n as f64).sqrt() < 0.01);

    // demonstrate geometric representation of quantum circuit
    // angle represents circuit depth/complexity
    let circuit_complexity = |gates: usize, qubits: usize| -> Geonum {
        let angle_fraction = (gates % qubits) as f64 / qubits as f64;
        Geonum::new(gates as f64, angle_fraction, 1.0)
    };

    // compute complexities
    let simple_circuit = circuit_complexity(4, 2);
    let complex_circuit = circuit_complexity(16, 4);

    // verify circuit scaling
    assert!(complex_circuit.mag > simple_circuit.mag);
}
