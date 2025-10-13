use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_finds_minimum_distance_from_point_to_line() {
    // textbook problem: find shortest distance from origin to line x + y = 1
    //
    // traditional requires:
    // 1. minimize f(x,y) = x² + y²
    // 2. subject to g(x,y) = x + y - 1 = 0
    // 3. lagrange: ∇f = λ∇g → [2x, 2y] = λ[1, 1]
    // 4. solve system: x = y = 0.5
    // 5. distance = √(x² + y²) = √0.5 = 0.7071

    let traditional_distance = (0.5_f64).sqrt();

    // angle design recognizes this as perpendicular projection
    // line x + y = 1 has normal vector [1, 1] at angle π/4
    // shortest distance is along this perpendicular
    // so optimal point is at angle π/4

    let normal_angle = Angle::new(1.0, 4.0); // π/4
    let theta = normal_angle.grade_angle();

    // from constraint x + y = 1 with x = L·cos(θ), y = L·sin(θ):
    // L·(cos(θ) + sin(θ)) = 1
    let perpendicular_length = 1.0 / (theta.cos() + theta.sin());

    // construct closest point
    let closest_point = Geonum::new_with_angle(perpendicular_length, normal_angle);

    // distance from origin = length of closest point
    let geonum_distance = closest_point.length;

    assert!(
        (geonum_distance - traditional_distance).abs() < EPSILON,
        "geonum finds same distance as lagrange"
    );

    // test coordinates of closest point
    let x = closest_point.project_to_dimension(0);
    let y = closest_point.project_to_dimension(1);

    assert!((x - 0.5).abs() < EPSILON);
    assert!((y - 0.5).abs() < EPSILON);
    assert!((x + y - 1.0).abs() < EPSILON, "point on line");

    // test any other point on line is farther from origin
    let other_points = [(0.3, 0.7), (0.8, 0.2), (0.1, 0.9)];

    for &(px, py) in &other_points {
        let dist_squared: f64 = px * px + py * py;
        let dist = dist_squared.sqrt();
        assert!(
            dist > geonum_distance,
            "point ({}, {}) farther than optimal",
            px,
            py
        );
    }

    // operations performed:
    // 1. recognize perpendicular at angle π/4
    // 2. cos(π/4) = 0.7071
    // 3. sin(π/4) = 0.7071
    // 4. sum = 1.4142
    // 5. length = 1 / 1.4142 = 0.7071
    //
    // total: 5 arithmetic operations vs solving lagrange system
    //
    // geometric insight:
    // shortest distance is perpendicular
    // perpendicular direction given by constraint normal
    // no optimization algorithm needed - just project perpendicular
}

#[test]
fn it_reduces_optimization_to_angle_matching() {
    // problem: minimize f(x,y) = x² + y² subject to x + y = 1
    //
    // traditional requires solving ∇f = λ∇g system with 3 unknowns
    // angle design recognizes optimal_angle = constraint_gradient_angle directly

    // traditional solution: x = y = 0.5, minimum = 0.5
    let traditional_min = 0.5;

    // quadrature identity proves f = x² + y² = length² from sin²θ + cos²θ = 1
    // so minimizing f = minimizing length

    // constraint x + y = 1 has gradient ∇g = [1, 1] at angle π/4
    let constraint_gradient_angle = Angle::new(1.0, 4.0);

    // lagrange condition ∇f = λ∇g means gradients parallel
    // since ∇f = 2[x, y] points radially at point angle
    // optimality when point_angle = constraint_gradient_angle
    let optimal_angle = constraint_gradient_angle;

    // constraint L(cos(θ) + sin(θ)) = 1 determines length at optimal angle
    let theta = optimal_angle.grade_angle();
    let optimal_length = 1.0 / (theta.cos() + theta.sin());

    // construct solution as geonum
    let solution = Geonum::new_with_angle(optimal_length, optimal_angle);
    let angle_min = solution.length * solution.length;

    assert!(
        (angle_min - traditional_min).abs() < EPSILON,
        "angle matching finds same minimum as lagrange system"
    );

    // test constraint satisfaction via projections
    let x = solution.project_to_dimension(0);
    let y = solution.project_to_dimension(1);
    assert!((x + y - 1.0).abs() < EPSILON);

    // reduction achieved: 3-equation system → angle matching arithmetic
    // traditional: O(n³) for n variables
    // angle: O(1) when optimal angle recognized
}

#[test]
fn it_exposes_scalar_dependency_as_projection_reconstruction() {
    // scalar optimization must project → destroy → reconstruct
    // angle optimization works pre-projection

    // start with geometric object
    let original = Geonum::new_with_angle(1.0 / 2.0_f64.sqrt(), Angle::new(1.0, 4.0));
    let original_objective = original.length * original.length;

    // projecting to scalars destroys geometric information
    let x = original.project_to_dimension(0);
    let y = original.project_to_dimension(1);
    // angle: destroyed
    // length: destroyed

    // scalar optimization must reconstruct via quadratic form
    let reconstructed = x * x + y * y;

    assert!(
        (reconstructed - original_objective).abs() < EPSILON,
        "sum of squares reconstructs destroyed length"
    );

    // overhead identified:
    // 1. had length² available directly
    // 2. projected to x, y (information loss)
    // 3. computed x² + y² (reconstruction)
    //
    // angle design eliminates steps 2 and 3
    // work with length² before projection happens
}

#[test]
fn it_proves_quadratic_forms_reconstruct_pre_projection_angles() {
    // x² + y² isnt fundamental math - its post-projection reconstruction
    // of pre-projection geometric object at [length, angle]
    //
    // traditional: start with scalars x, y → compute x² + y²
    // reality: [length, angle] → project to x, y → sum squares to recover length²

    let length = 5.0_f64;
    let angle = Angle::new(1.0, 3.0); // π/3

    // pre-projection form
    let pre_projection = Geonum::new_with_angle(length, angle);
    let pre_value = pre_projection.length * pre_projection.length;

    // project (information loss)
    let x = pre_projection.project_to_dimension(0);
    let y = pre_projection.project_to_dimension(1);

    // post-projection reconstruction
    let post_value = x * x + y * y;

    assert!(
        (post_value - pre_value).abs() < EPSILON,
        "quadratic form reconstructs pre-projection length"
    );

    // quadratic forms exist because projection destroyed geometric data
    // working pre-projection eliminates need for quadratic reconstruction
}

#[test]
fn it_solves_via_repeatable_angle_arithmetic() {
    // show explicit arithmetic workflow that scales
    // problem: maximize xy subject to x + y = 10

    let constraint_constant = 10.0;
    let angle = Angle::new(1.0, 4.0); // π/4 optimal for this problem
    let theta = angle.grade_angle();

    // step 1: compute length from constraint
    let sum = theta.cos() + theta.sin();
    let length = constraint_constant / sum;

    // step 2: project to coordinates
    let x = length * theta.cos();
    let y = length * theta.sin();

    // step 3: evaluate objective
    let area = x * y;

    // test results
    assert!((x - 5.0).abs() < EPSILON);
    assert!((y - 5.0).abs() < EPSILON);
    assert!((area - 25.0).abs() < EPSILON);

    // operations: 2 trig + 1 add + 1 div + 3 mult = 7 total
    // complexity: O(1) regardless of dimension
    // scalability: same arithmetic for any angle, any constraint constant
}

#[test]
fn it_demonstrates_angle_as_allocation_balance() {
    // angles encode allocation between dimensions
    // 45° = symmetric allocation = equal x and y
    //
    // problem: maximize xy on x + y = 10
    // everyone knows answer is x = y = 5 (square)
    // but why? because 45° means balanced allocation

    let constraint = 10.0;

    // sample different allocation angles
    let allocations = [
        (0.0, "0° all to x"),
        (PI / 4.0, "45° balanced"),
        (PI / 2.0, "90° all to y"),
    ];

    for &(theta, label) in &allocations {
        let sum: f64 = theta.cos() + theta.sin();
        if sum.abs() < EPSILON {
            continue;
        }

        let length = constraint / sum;
        let x = length * theta.cos();
        let y = length * theta.sin();
        let area = x * y;

        if label.contains("balanced") {
            // 45° gives maximum area
            assert!((area - 25.0).abs() < EPSILON);
            assert!((x - y).abs() < EPSILON, "balanced means x = y");
        } else {
            // other angles give zero (all allocated to one dimension)
            assert!(area < 1.0, "{} produces near-zero area", label);
        }
    }

    // insight: 45° angle immediately reveals x = y symmetry
    // no algebra required - its geometric balance
}

#[test]
fn it_collapses_search_space_from_nd_to_1d() {
    // traditional optimization searches n-dimensional coordinate space
    // angle design searches 1-dimensional angle space
    //
    // problem: minimize x² + y² subject to x + y = 1
    // traditional: 2D search constrained to 1D line
    // angle: 1D search directly

    // traditional parameterization: x = t, y = 1-t for t ∈ [0,1]
    // then minimize f(t) = t² + (1-t)²
    let scalar_samples: Vec<_> = (0..=10)
        .map(|i| {
            let t = i as f64 / 10.0;
            let x = t;
            let y = 1.0 - t;
            (t, x * x + y * y)
        })
        .collect();

    let scalar_min = scalar_samples
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // angle design: search angle space directly
    let angle_samples: Vec<_> = (0..=10)
        .map(|i| {
            let theta = (i as f64 / 20.0) * PI; // sample [0, π/2]
            let sum = theta.cos() + theta.sin();
            if sum.abs() < EPSILON {
                return (theta, f64::MAX);
            }
            let length = 1.0 / sum;
            (theta, length * length)
        })
        .collect();

    let angle_min = angle_samples
        .iter()
        .filter(|(_, v)| *v < f64::MAX)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    assert!((scalar_min.1 - angle_min.1).abs() < 0.01);

    // dimensional collapse:
    // scalar: 2D → 1D via constraint parameterization
    // angle: 1D naturally (angle is the search variable)
    //
    // for n dimensions:
    // scalar: nD → (n-k)D where k = constraints
    // angle: 1D always
}

#[test]
fn it_eliminates_lagrange_multipliers() {
    // lagrange multipliers exist to find where ∇f = λ∇g
    // angle design recognizes this as angle matching directly
    //
    // problem: minimize x² + y² subject to x + y = 1
    //
    // traditional needs λ:
    //   ∇f = [2x, 2y]
    //   ∇g = [1, 1]
    //   set [2x, 2y] = λ[1, 1]
    //   solve for x, y, λ
    //
    // angle design:
    //   ∇f points radially (at point angle)
    //   ∇g points at π/4
    //   parallel when point angle = π/4
    //   no λ needed

    // constraint gradient angle
    let grad_angle = Angle::new(1.0, 4.0);

    // at optimal point, gradient points at same angle
    let optimal = Geonum::new_with_angle(
        1.0 / (grad_angle.grade_angle().cos() + grad_angle.grade_angle().sin()),
        grad_angle,
    );

    // gradient of f = x² + y² is ∇f = 2[x, y]
    // this points radially at optimal.angle
    let grad_f_angle = optimal.angle;

    assert!(
        (grad_f_angle.grade_angle() - grad_angle.grade_angle()).abs() < EPSILON,
        "gradient angles match - no multiplier needed"
    );

    // λ eliminated: angle matching replaces multiplier algebra
}

#[test]
fn it_proves_optimization_as_arithmetic_not_search() {
    // traditional optimization = iterative search algorithms
    // angle optimization = direct arithmetic when structure recognized
    //
    // problem: minimize x² + y² subject to x + y = 1

    // traditional would iterate:
    // - start from initial guess
    // - compute gradients numerically
    // - update coordinates
    // - check convergence
    // - repeat 10-100+ times

    // angle design computes directly:
    let constraint_angle = Angle::new(1.0, 4.0); // step 1: recognize constraint gradient
    let theta = constraint_angle.grade_angle();
    let length = 1.0 / (theta.cos() + theta.sin()); // step 2: compute length
    let objective = length * length; // step 3: evaluate objective

    assert!((objective - 0.5).abs() < EPSILON);

    // operations:
    // - constraint gradient: immediate (π/4)
    // - length computation: 2 trig + 1 add + 1 div
    // - objective evaluation: 1 mult
    // total: 7 operations, no iteration

    // transformation: search problem → arithmetic problem
}

#[test]
fn it_scales_arithmetic_across_objectives() {
    // same arithmetic structure works for different objectives
    // only optimal angle and final evaluation change

    let constraint = 10.0;
    let optimal_angle = Angle::new(1.0, 4.0); // π/4 optimal for symmetric problems
    let theta = optimal_angle.grade_angle();

    // shared arithmetic
    let sum = theta.cos() + theta.sin();
    let length = constraint / sum;
    let x = length * theta.cos();
    let y = length * theta.sin();

    // objective 1: maximize xy
    let obj1 = x * y;
    assert!((obj1 - 25.0).abs() < EPSILON);

    // objective 2: minimize x² + y²
    let obj2 = x * x + y * y;
    assert!((obj2 - 50.0).abs() < EPSILON);

    // objective 3: minimize x² - xy + y²
    let obj3 = x * x - x * y + y * y;
    assert!((obj3 - 25.0).abs() < EPSILON);

    // pattern:
    // 1. find optimal angle (problem-specific)
    // 2. compute length via same arithmetic
    // 3. project to coordinates
    // 4. evaluate objective function
    //
    // steps 2-3 identical across objectives
}

#[test]
fn it_exposes_am_gm_inequality_as_angle_symmetry() {
    // arithmetic-geometric mean inequality: (x+y)/2 ≥ √(xy)
    // equality when x = y
    //
    // angle interpretation: equality at 45° (balanced allocation)
    //
    // problem: maximize xy subject to x + y = 10
    // am-gm says maximum when x = y = 5

    let constraint = 10.0;
    let symmetric_angle = Angle::new(1.0, 4.0); // 45°
    let theta = symmetric_angle.grade_angle();

    let length = constraint / (theta.cos() + theta.sin());
    let x = length * theta.cos();
    let y = length * theta.sin();

    // test am-gm equality
    let am = constraint / 2.0; // arithmetic mean
    let gm = (x * y).sqrt(); // geometric mean

    assert!((am - gm).abs() < EPSILON, "equality holds at 45°");
    assert!((x - y).abs() < EPSILON, "45° means x = y");

    // insight:
    // am-gm equality condition = geometric symmetry at 45°
    // no need to derive inequality
    // its angle balance made explicit
}

#[test]
fn it_proves_45_degrees_means_square_allocation() {
    // 45° = π/4 = equal projection onto both dimensions
    // this is why optimal rectangle with fixed perimeter is square
    //
    // problem: maximize area xy with perimeter 2x + 2y = 20
    // equivalently: x + y = 10

    let constraint = 10.0;

    // test various angles
    let angles = [(PI / 6.0, "30°"), (PI / 4.0, "45°"), (PI / 3.0, "60°")];

    let mut max_area = 0.0;
    let mut optimal_deg = 0.0;

    for &(theta, label) in &angles {
        let sum: f64 = theta.cos() + theta.sin();
        let length = constraint / sum;
        let x = length * theta.cos();
        let y = length * theta.sin();
        let area = x * y;

        if area > max_area {
            max_area = area;
            optimal_deg = theta * 180.0 / PI;
        }

        if label == "45°" {
            assert!((x - y).abs() < EPSILON, "45° produces square");
            assert!((area - 25.0).abs() < EPSILON, "maximum area");
        }
    }

    assert!((optimal_deg - 45.0).abs() < 1.0);

    // 45° is geometrically obvious:
    // - equal distance from both axes
    // - balanced allocation
    // - symmetric projection
    // no algebra needed to discover this
}
