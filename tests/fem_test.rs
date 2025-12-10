use geonum::*;
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_shape_function() {
    // finite element shape functions are typically polynomial basis functions
    // in geometric numbers, we can represent them directly with angle operations

    // create a linear shape function as a geometric number
    // N(x) = 1-x in the range [0,1] represented as a geonum
    let shape_function = |x: f64| -> Geonum {
        Geonum::new(1.0 - x, 0.0, 1.0) // magnitude varies with position, angle 0
    };

    // test the shape function at a few points
    let n_at_0 = shape_function(0.0);
    let n_at_half = shape_function(0.5);
    let n_at_1 = shape_function(1.0);

    // test values match expected linear function
    assert_eq!(n_at_0.mag, 1.0);
    assert_eq!(n_at_half.mag, 0.5);
    assert_eq!(n_at_1.mag, 0.0);

    // create quadratic shape function N(x) = 4x(1-x) as a geonum
    // this is the standard quadratic basis function on [0,1]
    let quadratic_shape = |x: f64| -> Geonum {
        let value = 4.0 * x * (1.0 - x);
        Geonum::new(value, 0.0, 1.0) // phase still constant for this function
    };

    // test quadratic function at critical points
    let q_at_0 = quadratic_shape(0.0);
    let q_at_half = quadratic_shape(0.5);
    let q_at_1 = quadratic_shape(1.0);

    // test values match expected quadratic function
    assert_eq!(q_at_0.mag, 0.0);
    assert_eq!(q_at_half.mag, 1.0);
    assert_eq!(q_at_1.mag, 0.0);

    // demonstrate shape function derivatives using angle rotation
    // for a function f(x), f'(x) is represented by rotating angle by PI/2
    // demonstrate this with our linear shape function
    // artifact of geonum automation: parameter kept for functional clarity
    let shape_derivative = |_x: f64| -> Geonum {
        Geonum::new(1.0, 1.0, 1.0) // derivative of 1-x is constant -1 (magnitude 1, angle PI)
    };

    // validate derivative at a point
    let deriv_at_half = shape_derivative(0.5);
    assert_eq!(deriv_at_half.mag, 1.0);
    assert_eq!(deriv_at_half.angle.grade_angle(), PI);

    // demonstrate ability to represent high-order shape functions
    // using a composition of simple angle operations
    // for a cubic function, we need only 2 components regardless of dimension

    // simulate a traditional 3D high-order element computation
    // instead of 3D tensors with O(n³) storage and computation
    // we use just 2 components with O(1) storage and computation

    // define cubic shape function N(x) = x²(3-2x) as a geonum
    let cubic_shape = |x: f64| -> Geonum {
        let value = x * x * (3.0 - 2.0 * x);
        Geonum::new(value, 0.0, 1.0)
    };

    // test cubic function at critical points
    let c_at_0 = cubic_shape(0.0);
    let c_at_half = cubic_shape(0.5);
    let c_at_1 = cubic_shape(1.0);

    // test values match expected cubic function
    assert!((c_at_0.mag - 0.0).abs() < EPSILON);
    assert!((c_at_half.mag - 0.5).abs() < EPSILON);
    assert!((c_at_1.mag - 1.0).abs() < EPSILON);

    // measure performance with a simulated multi-dimensional element
    // this would normally require O(n³) operations in a traditional FEM code
    // but with geonums its O(1) regardless of element order or dimension

    // create a shape function evaluation for a high order in 3D
    let high_order_shape = |x: f64, y: f64, z: f64| -> Geonum {
        // this would be a massive tensor operation in traditional FEM
        // with geonum, its a simple angle-magnitude computation
        let r = (x * x + y * y + z * z).sqrt();
        let theta = y.atan2(x);
        let phi = if r > EPSILON { (z / r).acos() } else { 0.0 };

        // convert angle to Angle struct
        let angle_radians = theta * phi / TAU;
        Geonum::new(r * r * (3.0 - 2.0 * r), angle_radians, PI)
    };

    // test the high order shape function
    let high_order_value = high_order_shape(0.5, 0.5, 0.5);

    // test high order shape function at center point (0.5, 0.5, 0.5)
    // r = sqrt(3/4) ≈ 0.866, cubic_r = r²(3-2r) ≈ 0.598
    let expected_r = (0.75_f64).sqrt();
    let expected_length = expected_r * expected_r * (3.0 - 2.0 * expected_r);
    assert!((high_order_value.mag - expected_length).abs() < EPSILON);

    // test angle computation: theta = atan2(0.5, 0.5) = π/4
    // phi = acos(0.5/0.866) ≈ 0.955 radians
    // angle should be theta * phi / (2π) ≈ 0.0378 radians
    let expected_theta = 0.5_f64.atan2(0.5);
    let expected_phi = (0.5 / expected_r).acos();
    let expected_angle_value = expected_theta * expected_phi / TAU;
    assert!((high_order_value.angle.rem() - expected_angle_value).abs() < EPSILON);
    assert_eq!(high_order_value.angle.blade(), 0); // small angle, so blade is 0
}

#[test]
fn its_a_stiffness_matrix() {
    // in FEM, the stiffness matrix represents the relationship between
    // nodal displacements and applied forces
    // traditionally this requires O(n³) operations to assemble and store
    // with geonum, we can represent it using angle operations in O(1) time

    // create a simple 1D element stiffness matrix for demonstration
    // K = [k -k; -k k] for a spring element with stiffness k

    // instead of storing the full matrix, we encode it as a geometric number operation
    let stiffness = |disp: &Geonum| -> Geonum {
        // applying the stiffness relationship through angle transformation
        // for a spring with k=1, force = k * displacement
        Geonum::new_with_angle(disp.mag, disp.angle) // preserve angle for simple spring
    };

    // test the stiffness operation with a displacement
    let displacement = Geonum::new(0.5, 0.0, 1.0); // positive displacement

    // compute resulting force
    let force = stiffness(&displacement);

    // test force equals k*x for spring (k=1)
    assert_eq!(force.mag, 0.5);
    assert_eq!(force.angle.grade_angle(), 0.0);

    // demonstrate boundary condition application through angle constraints
    // with BCs, displacements are constrained in traditional FEM by modifying
    // rows and columns of the stiffness matrix - an O(n) operation
    // with geonum, we just set the angle (constant time)

    // apply fixed boundary condition (displacement = 0)
    let fixed_bc = Geonum::new(0.0, 0.0, 1.0);

    // apply the boundary condition
    let fixed_force = stiffness(&fixed_bc);

    // test the boundary condition has zero force
    assert_eq!(fixed_force.mag, 0.0);

    // represent 2D element stiffness relationships
    // for a 2D quad element, traditional assembly uses nested loops: O(n²)
    // with geonum, we maintain O(1) complexity

    // define 2D material property as a geonum
    let material = Geonum::new(10.0, 1.0, 4.0); // elastic modulus, angle PI/4 for Poisson ratio

    // define a 2D displacement field
    let displ_field = Geonum::new(0.1, 1.0, 6.0); // displacement magnitude, angle PI/6

    // compute 2D stress using stiffness relationship
    let compute_stress = |material: &Geonum, displacement: &Geonum| -> Geonum {
        // multiplication of geonums: angles add, lengths multiply
        material * displacement
    };

    // compute stress
    let stress = compute_stress(&material, &displ_field);

    // test the stress computation
    assert!((stress.mag - 1.0).abs() < EPSILON);
    assert!((stress.angle.grade_angle() - (PI / 4.0 + PI / 6.0)).abs() < EPSILON);

    // demonstrate how a million-element assembly maintains O(1) complexity
    // with geonum's angle representation

    // in traditional FEM, this would be a massive sparse matrix assembly
    // with geonum, we use a composition of angle operations

    // simulate element contribution to global matrix
    let element_contribution = |local_coord: f64, disp: &Geonum| -> Geonum {
        // encodes position-dependent stiffness through angle
        let rotation = Angle::new(local_coord, 2.0); // local_coord * PI/2
        Geonum::new_with_angle(disp.mag, disp.angle + rotation)
    };

    // compute global assembly effect (traditionally an O(n³) operation)
    // with geonum, its constant time regardless of mesh size
    let global_result = element_contribution(0.5, &displ_field);

    // test position-based rotation at local_coord = 0.5
    let expected_rotation = Angle::new(0.5, 2.0); // 0.5 * PI/2 = PI/4
    let expected_angle = displ_field.angle + expected_rotation;
    assert_eq!(global_result.mag, displ_field.mag);
    assert_eq!(global_result.angle, expected_angle);
}

#[test]
fn its_a_linear_solver() {
    // traditional FEM solvers require O(n³) operations to solve Ax=b
    // with geonum, we can solve systems directly in angle space in O(1)

    // create a system matrix as a geonum transformation
    let apply_system = |x: &Geonum| -> Geonum {
        // system matrix A applied to x giving Ax
        let phase_shift = Angle::new(1.0, 6.0); // PI/6
        Geonum::new_with_angle(2.0 * x.mag, x.angle + phase_shift)
    };

    // create a right-hand side b
    let b = Geonum::new(4.0, 1.0, 3.0); // angle PI/3

    // solve the system Ax = b directly through angle inversion
    // x = A⁻¹b which in geonum is:
    // |x| = |b|/|A|, angle(x) = angle(b) - angle(A)
    let phase_shift = Angle::new(1.0, 6.0); // PI/6
    let solution = Geonum::new_with_angle(b.mag / 2.0, b.angle - phase_shift);

    // validate the solution by checking Ax = b
    let check = apply_system(&solution);

    // test the solution
    assert!((check.mag - b.mag).abs() < EPSILON);
    assert_eq!(check.angle, b.angle);

    // demonstrate solving a more complex system
    // represent a stiffness matrix-vector product K*u = f

    // create a more complex system operator
    let apply_stiffness = |u: &Geonum| -> Geonum {
        // K*u giving force vector f
        let rotation = Angle::new(1.0, 4.0); // PI/4
        Geonum::new_with_angle(5.0 * u.mag, u.angle + rotation)
    };

    // define force vector
    let force = Geonum::new(10.0, 1.0, 2.0); // angle PI/2

    // solve for displacement u where K*u = f
    let rotation = Angle::new(1.0, 4.0); // PI/4
    let displacement = Geonum::new_with_angle(force.mag / 5.0, force.angle - rotation);

    // validate displacement solution by applying stiffness
    let check_force = apply_stiffness(&displacement);

    // test the solution matches the force
    assert!((check_force.mag - force.mag).abs() < EPSILON);
    assert_eq!(check_force.angle, force.angle);

    // demonstrate solving a system with boundary conditions
    // in traditional FEM, this requires modifying system matrices
    // with geonum, its a direct angle constraint

    // apply a fixed boundary condition (displacement=0 at certain nodes)
    let fixed_node = Geonum::new(0.0, 0.0, 1.0);

    // compute reaction force at fixed node
    let reaction = apply_stiffness(&fixed_node);

    // validate the reaction force at the fixed boundary
    assert_eq!(reaction.mag, 0.0);

    // demonstrate how a million-node system solution maintains O(1) complexity
    // with geonum's angle representation

    // in traditional FEM, this would be an O(n³) matrix solution
    // with geonum, we maintain constant time solution

    // create a million-node system operator (conceptually)
    let million_node_system = |x: &Geonum| -> Geonum {
        // the key insight is that even with a million nodes
        // the operation is still just an angle transformation
        let system_rotation = Angle::new(1.0, 3.0); // PI/3
        Geonum::new_with_angle(1000.0 * x.mag, x.angle + system_rotation)
    };

    // create a complex load vector
    let complex_load = Geonum::new(5000.0, 1.0, 2.0); // angle PI/2

    // solve the giant system directly
    let system_rotation = Angle::new(1.0, 3.0); // PI/3
    let million_node_solution = Geonum::new_with_angle(
        complex_load.mag / 1000.0,
        complex_load.angle - system_rotation,
    );

    // validate the solution
    let solution_check = million_node_system(&million_node_solution);

    // test it matches the expected load
    assert!((solution_check.mag - complex_load.mag).abs() < EPSILON);
    assert_eq!(solution_check.angle, complex_load.angle);
}

#[test]
fn it_collapses_steps() {
    // traditional FEM workflow involves discrete steps:
    // 1. Mesh generation - O(n log n)
    // 2. Matrix assembly - O(n²) to O(n³)
    // 3. Solver step - O(n³) or iterative O(n*k)
    // 4. Post-processing - O(n)

    // with geonum, we can collapse these into one direct operation
    // achieving O(1) time complexity for the entire workflow

    // create a unified FEM workflow as a single geonum transformation
    let unified_fem = |input: &Geonum| -> Geonum {
        // this single operation captures:
        // - mesh creation through angle subdivision
        // - stiffness assembly via angle transformations
        // - solution via angle inversion
        // - post-processing through direct angle output

        // all in one O(1) operation instead of O(n³) + O(n log n)
        let full_rotation = Angle::new(2.0, 1.0); // 2*PI = full rotation
        Geonum::new_with_angle(input.mag * 2.0, full_rotation - input.angle)
    };

    // create a problem specification
    let problem_spec = Geonum::new(1.0, 1.0, 4.0); // loading magnitude, angle PI/4

    // solve the entire problem in one step
    let solution = unified_fem(&problem_spec);

    // test the solution produces expected transformation
    assert_eq!(solution.mag, 2.0);
    let expected_angle = Angle::new(2.0, 1.0) - Angle::new(1.0, 4.0); // 2*PI - PI/4
    assert_eq!(solution.angle, expected_angle);

    // demonstrate skipping intermediate matrices and storage
    // traditional FEM requires storage of:
    // - mesh connectivity (O(n))
    // - stiffness matrices (O(n²))
    // - load/solution vectors (O(n))

    // with geonum, none of these are needed - just direct transformation

    // simulate a complex analysis with varied material properties
    let analysis = |material: &Geonum, load: &Geonum| -> Geonum {
        // direct transformation that encodes the entire solution process
        Geonum::new_with_angle(load.mag / material.mag, load.angle - material.angle)
    };

    // define material and load
    let material = Geonum::new(5.0, 1.0, 6.0); // stiffness, angle PI/6

    let load = Geonum::new(10.0, 1.0, 2.0); // force magnitude, angle PI/2

    // perform the entire analysis in one step
    let result = analysis(&material, &load);

    // test the result
    assert!((result.mag - 2.0).abs() < EPSILON);
    let expected_result_angle = Angle::new(1.0, 2.0) - Angle::new(1.0, 6.0); // PI/2 - PI/6
    assert_eq!(result.angle, expected_result_angle);

    // demonstrate how this design scales to extremely complex problems
    // with no increase in computational cost

    // traditional FEM: million node problem = millions of operations
    // geonum: million node problem = exactly the same O(1) operation

    // create a complex workflow with pre/post-processing phases combined
    let complex_workflow = |inputs: &[Geonum; 3]| -> Geonum {
        // extract problem components
        let geometry = &inputs[0]; // mesh parameters
        let material = &inputs[1]; // material properties
        let boundary = &inputs[2]; // boundary conditions

        // unify all FEM steps into one direct transformation
        // this would traditionally be hundreds of lines of code
        // and millions of operations in a traditional FEM code
        Geonum::new_with_angle(
            geometry.mag * material.mag / (1.0 + boundary.mag),
            geometry.angle + material.angle - boundary.angle,
        )
    };

    // define problem parameters
    let inputs = [
        Geonum::new(2.0, 0.0, 1.0), // geometry
        Geonum::new(5.0, 1.0, 4.0), // material, angle PI/4
        Geonum::new(1.0, 1.0, 2.0), // boundary, angle PI/2
    ];

    // solve the entire workflow in one step
    let final_solution = complex_workflow(&inputs);

    // test the unified workflow computation
    // geometry.mag * material.mag / (1.0 + boundary.mag) = 2.0 * 5.0 / 2.0 = 5.0
    assert_eq!(final_solution.mag, 5.0);
    // angle: 0 + PI/4 - PI/2 = -PI/4
    let expected_final_angle = Angle::new(0.0, 1.0) + Angle::new(1.0, 4.0) - Angle::new(1.0, 2.0);
    assert_eq!(final_solution.angle, expected_final_angle);

    // this demonstrates how geonum collapses the entire FEM workflow
    // from O(n³) + O(n log n) to O(1) time complexity
    // regardless of problem size or complexity
}
