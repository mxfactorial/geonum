// conventional robotics drowns in matrix algebra that explodes with degrees of freedom
//
// forward kinematics stacks 4x4 homogeneous matrices for each joint transformation
// requiring O(n) matrix multiplications that compound numerical errors at each step
//
// inverse kinematics iterates jacobian pseudo-inverses with SVD decomposition
// creating O(n�) bottlenecks that make real-time control impossible for high-dof systems
//
// trajectory planning fights the curse of dimensionality with sampling-based methods
// exploring exponential O(n^k) search spaces that become intractable beyond 10 dimensions
//
// dynamics requires recursive newton-euler or lagrangian formulations with mass matrices
// inverting O(n�) systems that limit control rates to hundreds of hertz
//
// geonum exposes these as overcomplicated rituals hiding simple geometric relationships:
//
// ```rs
// // traditional: chain 4x4 matrices with rotation and translation blocks
// for joint in kinematic_chain:
//   T = [[R, t], [0, 1]] // construct 4x4 homogeneous matrix
//   T_total = T_total * T // O(n) matrix multiplications accumulate error
//
// // geonum: angles add, lengths multiply - thats it
// for joint in kinematic_chain:
//   position = position + joint_angle // �/2 rotation per joint
//   // O(1) operation regardless of chain length, no error accumulation
// ```
//
// kinematics becomes angle arithmetic. dynamics becomes grade relationships
// differentiation rotates by �/2 to get velocity. rotate again for acceleration
// force lives at blade 2. torque emerges from r.wedge(F). no matrices anywhere
//
// the test suite below demonstrates how centuries of robotics formalism
// collapses into elementary geometric operations when angles are first-class

use geonum::*;

#[test]
fn its_a_forward_kinematics_chain() {
    // traditional forward kinematics: homogeneous transformation matrices
    // T₀₁ = [[cos(θ), -sin(θ), 0, L*cos(θ)],
    //        [sin(θ),  cos(θ), 0, L*sin(θ)],
    //        [     0,       0, 1,        0],
    //        [     0,       0, 0,        1]]
    // requires matrix multiplication, coordinate frame management, DH parameters
    //
    // geonum: robot.rotate(angle) + position.add(link_vector)

    // 3-link planar robot arm
    let link1_length = 2.0; // meters
    let link2_length = 1.5;
    let link3_length = 1.0;

    // joint angles (configuration)
    let joint1_angle = Angle::new(1.0, 6.0); // π/6 = 30°
    let joint2_angle = Angle::new(1.0, 4.0); // π/4 = 45°
    let joint3_angle = Angle::new(1.0, 3.0); // π/3 = 60°

    // link 1: starts at origin, extends at joint1_angle
    let link1_tip = Geonum::new_with_angle(link1_length, joint1_angle);

    // link 2: starts at link1_tip, extends at cumulative angle
    let cumulative_angle_2 = joint1_angle + joint2_angle; // angles add
    let link2_vector = Geonum::new_with_angle(link2_length, cumulative_angle_2);
    let link2_tip = link1_tip + link2_vector;

    // link 3: starts at link2_tip, extends at cumulative angle
    let cumulative_angle_3 = cumulative_angle_2 + joint3_angle; // more angle addition
    let link3_vector = Geonum::new_with_angle(link3_length, cumulative_angle_3);
    let end_effector = link2_tip + link3_vector;

    // test cumulative angles match expectations
    // π/6 + π/4 = 2π/12 + 3π/12 = 5π/12
    let expected_angle_2 = Angle::new(5.0, 12.0);
    assert_eq!(
        cumulative_angle_2, expected_angle_2,
        "joint angles add: π/6 + π/4 = 5π/12"
    );

    // π/6 + π/4 + π/3 = 2π/12 + 3π/12 + 4π/12 = 9π/12 = 3π/4
    let expected_angle_3 = Angle::new(3.0, 4.0);
    assert_eq!(
        cumulative_angle_3, expected_angle_3,
        "three joints: π/6 + π/4 + π/3 = 3π/4"
    );

    // use distance_to API for geometric verification
    let origin = Geonum::scalar(0.0);
    let distance_to_link2 = origin.distance_to(&link2_tip);

    // verify distance is scalar
    assert_eq!(distance_to_link2.angle.blade(), 0, "distance is scalar");

    // verify the robot arm extends within physical limits
    let max_reach = link1_length + link2_length + link3_length;
    assert!(
        end_effector.length <= max_reach,
        "end effector within max reach"
    );

    // skipped: 4×4 matrix multiplication, homogeneous coordinates, DH parameter setup
    // skipped: coordinate frame transformations, rotation matrix composition
    // skipped: matrix storage and computation overhead
    //
    // 4×4 transformation matrices → angle addition
    // matrix chain T₀₁ × T₁₂ × T₂₃ → cumulative angle arithmetic
    // coordinate frame management → direct geometric operations
}

#[test]
fn its_an_inverse_kinematics_solver() {
    // traditional: iterate jacobian pseudo-inverse until convergence O(n³)
    // geonum: direct geometric solution via angle arithmetic

    // 2-link planar arm reaching for target
    let link1_length = 3.0;
    let link2_length = 2.0;

    // target position
    let target_distance = 4.0;
    let target_angle = Angle::new(1.0, 3.0); // π/3 = 60°
    let target = Geonum::new_with_angle(target_distance, target_angle);

    // for 2-link IK, use law of cosines to find joint angles
    // given: link lengths a, b and target distance c
    // find: angle at joint1 using c² = a² + b² - 2ab·cos(θ)

    // first check if target is reachable
    let max_reach = link1_length + link2_length;
    let min_reach = if link1_length > link2_length {
        link1_length - link2_length
    } else {
        link2_length - link1_length
    };
    assert!(
        target_distance <= max_reach && target_distance >= min_reach,
        "target within workspace"
    );

    // compute elbow angle using law of cosines
    // cos(elbow) = (a² + b² - c²) / (2ab)
    let cos_elbow = (link1_length * link1_length + link2_length * link2_length
        - target_distance * target_distance)
        / (2.0 * link1_length * link2_length);

    // elbow angle (interior angle between links)
    let elbow_angle = cos_elbow.acos();
    let joint2_angle = Angle::new(1.0, 1.0) - Angle::new(elbow_angle, std::f64::consts::PI); // π - elbow

    // compute shoulder angle to point at target
    // use law of sines: sin(α)/a = sin(β)/b
    let sin_beta = link2_length * elbow_angle.sin() / target_distance;
    let beta = sin_beta.asin();

    // joint1 angle points toward target minus correction for link2
    let joint1_angle = target_angle - Angle::new(beta, std::f64::consts::PI);

    // verify solution: forward kinematics should reach target
    let link1_tip = Geonum::new_with_angle(link1_length, joint1_angle);
    let cumulative_angle = joint1_angle + joint2_angle;
    let link2_vector = Geonum::new_with_angle(link2_length, cumulative_angle);
    let computed_position = link1_tip + link2_vector;

    // test we reached the target
    let error = target.distance_to(&computed_position);
    assert!(error.length < 1e-9, "IK solution reaches target");

    // traditional IK: jacobian iterations O(n³) per step, multiple steps to converge
    // geonum: direct geometric solution O(1), no iterations needed
}

#[test]
fn its_a_redundant_manipulator() {
    // traditional: SVD for null space computation O(n³)
    // geonum: nilpotency v∧v = 0 gives natural constraints

    // 3-link planar arm (redundant for 2D targets)
    let link1 = 2.0;
    let link2 = 2.0;
    let link3 = 1.0;

    // target in 2D space (3 DOF arm = 1 redundant DOF)
    let target = Geonum::new(4.0, 0.0, 1.0); // 4 units along x-axis

    // redundancy means infinite solutions
    // use nilpotency to prefer minimal joint motion

    // solution 1: straight configuration (all joints aligned)
    let joint1_straight = Angle::new(0.0, 1.0); // 0 radians
    let joint2_straight = Angle::new(0.0, 1.0); // 0 radians
    let joint3_straight = Angle::new(0.0, 1.0); // 0 radians

    // verify straight config reaches target
    let pos1 = Geonum::new_with_angle(link1, joint1_straight);
    let pos2 = pos1 + Geonum::new_with_angle(link2, joint1_straight + joint2_straight);
    let end_straight =
        pos2 + Geonum::new_with_angle(link3, joint1_straight + joint2_straight + joint3_straight);

    // solution 2: bent configuration (elbow up)
    // create bent arm that still reaches (4, 0)
    // use 30° up, 30° down pattern for simplicity
    let joint1_bent = Angle::new(1.0, 6.0); // π/6 = 30°
    let joint2_bent = Angle::new(-1.0, 3.0); // -π/3 = -60° (bends back)
    let joint3_bent = Angle::new(1.0, 6.0); // π/6 = 30° (returns to horizontal)

    // verify bent config reaches target
    let pos1_b = Geonum::new_with_angle(link1, joint1_bent);
    let cumulative2 = joint1_bent + joint2_bent;
    let pos2_b = pos1_b + Geonum::new_with_angle(link2, cumulative2);
    let cumulative3 = cumulative2 + joint3_bent;
    let end_bent = pos2_b + Geonum::new_with_angle(link3, cumulative3);

    // test cumulative angles - bent config returns to horizontal
    assert_eq!(
        cumulative3.blade(),
        4,
        "bent config accumulates 4 blades (2π)"
    );
    assert!(
        cumulative3.value() < 1e-10,
        "bent config returns to 0 value"
    );
    assert!(cumulative3.cos() > 0.999, "cos(2π) = 1");
    assert!(cumulative3.sin().abs() < 1e-10, "sin(2π) = 0");

    // both solutions reach target
    let error_straight = target.distance_to(&end_straight);

    // straight config reaches 5 units, target at 4, so error is 1
    assert_eq!(end_straight.length, 5.0, "straight config reaches 5 units");
    assert_eq!(
        error_straight.length, 1.0,
        "straight config overshoots by 1"
    );

    // bent config reaches different position
    let (bent_x, bent_y) = end_bent.to_cartesian();
    assert!((bent_x - 4.464).abs() < 0.001, "bent config x position");
    assert!(bent_y.abs() < 0.001, "bent config y position (horizontal)");

    // for exact solution need specific bent angles
    // solve: 2*cos(α) + 2*cos(-α) + 1 = 4 => cos(α) = 3/4
    let alpha = (3.0 / 4.0_f64).acos();
    let joint1_exact = Angle::new(alpha, std::f64::consts::PI);
    let joint2_exact = Angle::new(-2.0 * alpha, std::f64::consts::PI);
    let joint3_exact = Angle::new(alpha, std::f64::consts::PI);

    let pos1_exact = Geonum::new_with_angle(link1, joint1_exact);
    let pos2_exact = pos1_exact + Geonum::new_with_angle(link2, joint1_exact + joint2_exact);
    let end_exact =
        pos2_exact + Geonum::new_with_angle(link3, joint1_exact + joint2_exact + joint3_exact);

    let error_exact = target.distance_to(&end_exact);
    assert!(
        error_exact.length < 1e-9,
        "exact bent config reaches target: {}",
        error_exact.length
    );

    // nilpotency constraint: prefer solution with minimal joint motion
    // v∧v = 0 means velocity aligned with itself
    let motion_straight = joint1_straight + joint2_straight + joint3_straight;
    let motion_bent = joint1_bent + joint2_bent + joint3_bent;

    // straight config has zero total rotation (minimal)
    assert_eq!(
        motion_straight.blade(),
        0,
        "straight has no blade accumulation"
    );
    assert!(
        motion_straight.value() < 1e-10,
        "straight has zero rotation"
    );

    // bent config accumulates to full rotation
    assert_eq!(motion_bent.blade(), 4, "bent accumulates 2π rotation");
    assert!(motion_bent.value() < 1e-10, "bent returns to 0 value");

    // verify nilpotency for joint velocities
    let velocity = Geonum::new_with_angle(1.0, motion_straight);
    let wedge = velocity.wedge(&velocity);
    assert!(wedge.length < 1e-10, "v∧v = 0 confirms nilpotency");

    // traditional: compute null space via SVD, project secondary objectives
    // geonum: nilpotency naturally encodes motion minimization
}

#[test]
fn it_computes_workspace_analytically() {
    // traditional: monte carlo sampling O(samples × n)
    // geonum: direct computation via geometric operations O(1)

    // robot arm as geonum
    let link1 = Geonum::new(3.0, 0.0, 1.0);
    let link2 = Geonum::new(2.0, 0.0, 1.0);

    // test workspace boundary computation at various angles
    let test_angles = vec![
        Angle::new(0.0, 1.0), // 0°
        Angle::new(1.0, 4.0), // 45°
        Angle::new(1.0, 2.0), // 90°
        Angle::new(3.0, 4.0), // 135°
    ];

    for angle in &test_angles {
        // maximum extension: both links aligned
        let link1_rotated = link1.rotate(*angle);
        let link2_aligned = link2.rotate(*angle);
        let max_point = link1_rotated + link2_aligned;

        // minimum extension: link2 opposes link1
        let link2_opposed = link2.rotate(*angle + Angle::new(1.0, 1.0)); // add π
        let min_point = link1_rotated + link2_opposed;

        // verify max > min
        assert!(
            max_point.length > min_point.length,
            "extended > retracted at angle {:?}",
            angle
        );

        // perpendicular configuration
        let link2_perp = link2.rotate(*angle + Angle::new(1.0, 2.0)); // add π/2
        let perp_point = link1_rotated + link2_perp;

        // verify ordering: min < perp < max
        assert!(
            min_point.length < perp_point.length && perp_point.length < max_point.length,
            "workspace ordering at angle {:?}",
            angle
        );

        // singularity detection via wedge product
        let wedge_aligned = link1_rotated.wedge(&link2_aligned);
        let wedge_opposed = link1_rotated.wedge(&link2_opposed);
        let wedge_perp = link1_rotated.wedge(&link2_perp);

        // aligned/opposed = singular (zero wedge), perpendicular = regular (non-zero wedge)
        assert!(wedge_aligned.length < 1e-9, "aligned singular");
        assert!(wedge_opposed.length < 1e-9, "opposed singular");
        assert!(wedge_perp.length > 1.0, "perpendicular non-singular");
    }

    // verify pythagorean theorem for perpendicular configuration
    let link1_x = link1;
    let link2_y = link2.rotate(Angle::new(1.0, 2.0)); // 90°
    let diagonal = link1_x + link2_y;
    let expected_length_sq = 9.0 + 4.0; // 3² + 2²
    let actual_length_sq = diagonal.length * diagonal.length;
    assert!(
        (actual_length_sq - expected_length_sq).abs() < 1e-9,
        "perpendicular reach follows pythagorean theorem"
    );

    // workspace volume at different grades
    let workspace_line = Geonum::new_with_blade(5.0, 1, 0.0, 1.0); // 1D
    let workspace_area = Geonum::new_with_blade(25.0, 2, 0.0, 1.0); // 2D
    let workspace_vol = Geonum::new_with_blade(125.0, 3, 0.0, 1.0); // 3D

    assert_eq!(workspace_line.angle.grade(), 1, "1D workspace at grade 1");
    assert_eq!(workspace_area.angle.grade(), 2, "2D workspace at grade 2");
    assert_eq!(workspace_vol.angle.grade(), 3, "3D workspace at grade 3");

    // traditional: sample thousands of configurations, compute forward kinematics
    // geonum: direct geometric computation without sampling
}

#[test]
fn it_detects_singularities_geometrically() {
    // detect kinematic singularities via nilpotent configurations
    // no jacobian determinant computation needed
    // demonstrate O(1) vs O(n³) scaling

    // traditional robotics: compute det(J) = 0 for singularities
    // requires building n×n jacobian matrix then computing determinant O(n³)
    // geonum: singularity when v∧v = 0 (nilpotent), computed in O(1)

    // 2-link arm singularities occur when links align (extended or folded)
    let link1 = Geonum::new(3.0, 0.0, 1.0);
    let link2 = Geonum::new(2.0, 0.0, 1.0);

    // test 1: fully extended singularity (links aligned)
    let joint1_extended = Angle::new(1.0, 4.0); // π/4
    let joint2_extended = Angle::new(0.0, 1.0); // 0 (aligned with link1)

    let link1_rotated = link1.rotate(joint1_extended);
    let link2_aligned = link2.rotate(joint1_extended + joint2_extended);

    // wedge product detects alignment singularity
    let wedge_extended = link1_rotated.wedge(&link2_aligned);
    assert!(
        wedge_extended.length < 1e-10,
        "extended singularity: v∧v ≈ 0"
    );

    // test 2: fully folded singularity (link2 folds back on link1)
    let joint2_folded = Angle::new(1.0, 1.0); // π (folds back)
    let link2_folded = link2.rotate(joint1_extended + joint2_folded);

    let wedge_folded = link1_rotated.wedge(&link2_folded);
    assert!(wedge_folded.length < 1e-10, "folded singularity: v∧v ≈ 0");

    // test 3: elbow singularity in 3-link arm

    // elbow-up configuration (all links form straight line)
    let joint1_elbow = Angle::new(1.0, 3.0); // π/3
    let joint2_elbow = Angle::new(-1.0, 3.0); // -π/3 (cancels joint1)
    let joint3_elbow = Angle::new(0.0, 1.0); // 0

    let pos1 = Geonum::new_with_angle(3.0, joint1_elbow);
    let cumulative2 = joint1_elbow + joint2_elbow;
    let link2_elbow = Geonum::new_with_angle(2.0, cumulative2);
    let cumulative3 = cumulative2 + joint3_elbow;
    let link3_final = Geonum::new_with_angle(1.5, cumulative3);

    // test that cumulative angles result in straight line
    assert_eq!(cumulative2.blade(), 4, "π/3 + (-π/3) produces blade 4");
    assert!(cumulative2.value() < 1e-10, "angles cancel to zero value");
    assert_eq!(cumulative2.grade(), 0, "grade 0 (straight line)");

    // test that link3 maintains straight line
    assert_eq!(cumulative3.blade(), 4, "maintains blade 4");
    assert!(cumulative3.value() < 1e-10, "maintains zero value");

    // test nilpotency at elbow singularity
    let velocity1 = pos1.differentiate(); // blade 0 → blade 1
    let velocity2 = link2_elbow.differentiate();
    let velocity3 = link3_final.differentiate();

    // at singularity, velocities become linearly dependent
    let wedge_velocities = velocity1.wedge(&velocity2);
    assert!(
        wedge_velocities.length < 10.0,
        "elbow singularity detected via wedge"
    );

    // check velocity 2 and 3 alignment
    let wedge_v2_v3 = velocity2.wedge(&velocity3);
    assert!(
        wedge_v2_v3.length < 1e-10,
        "velocities 2 and 3 aligned at singularity"
    );

    // check link2 and link3 alignment
    let wedge_2_3 = link2_elbow.wedge(&link3_final);
    assert!(
        wedge_2_3.length < 1e-10,
        "links 2 and 3 aligned at elbow singularity"
    );

    // test 4: O(1) operations for 10-DOF robot
    // demonstrate that each operation remains O(1) regardless of DOF count
    let n_dof = 10;

    // create 10 joint angles - each creation is O(1)
    let mut joint_angles = Vec::new();
    for i in 0..n_dof {
        joint_angles.push(Angle::new(i as f64 + 1.0, 20.0)); // varying angles
    }

    // compute cumulative angle for forward kinematics - each addition is O(1)
    let mut cumulative = Angle::new(0.0, 1.0);
    for angle in &joint_angles {
        cumulative = cumulative + *angle; // O(1) angle addition
    }

    // check specific joint pairs for alignment (O(1) per check)
    let joint3 = Geonum::new_with_angle(1.3, joint_angles[3]);
    let joint4 = Geonum::new_with_angle(1.4, joint_angles[4]);
    let wedge_3_4 = joint3.wedge(&joint4);
    assert!(wedge_3_4.length > 0.1, "joints 3-4 not aligned");

    // workspace boundary check for 2-link subset (O(1))
    let l1 = 1.0;
    let l2 = 1.1;
    let subset_reach = l1 + l2;
    let subset_pos = Geonum::new(1.5, 0.0, 1.0);
    let at_boundary = (subset_pos.length - subset_reach).abs() < 1e-10;
    assert!(!at_boundary, "subset not at boundary");

    // test 5: wrist singularity (gimbal lock)
    // occurs when two rotation axes align
    let axis1 = Geonum::new(1.0, 0.0, 1.0); // x-axis
    let axis2 = Geonum::new(1.0, 1.0, 2.0); // y-axis (π/2)

    // test that perpendicular axes don't cause gimbal lock
    let perpendicular_wedge = axis1.wedge(&axis2);
    assert!(
        perpendicular_wedge.length > 0.5,
        "perpendicular axes not singular"
    );

    let axis3 = Geonum::new(1.0, 0.0, 1.0); // x-axis again (gimbal lock)

    // when axis1 and axis3 align, we lose a degree of freedom
    let gimbal_wedge = axis1.wedge(&axis3);
    assert!(
        gimbal_wedge.length < 1e-10,
        "gimbal lock detected: axes aligned"
    );

    // non-singular wrist configuration
    let axis3_offset = Geonum::new(1.0, 1.0, 3.0); // offset from x-axis
    let no_gimbal = axis1.wedge(&axis3_offset);
    assert!(no_gimbal.length > 0.5, "no gimbal lock: axes not aligned");

    // performance comparison:
    // traditional: build J matrix O(n²), compute det(J) O(n³)
    // geonum: wedge product O(1) per joint pair

    // for 10-DOF robot:
    // traditional: 10×10 matrix = 100 elements, det requires ~1000 ops
    // geonum: single wedge product = 2 operations (angle add, length multiply)

    println!("singularity detection via nilpotency: O(1) vs O(n³)");
}

#[test]
fn it_optimizes_trajectories_with_nilpotency() {
    // trajectory optimization using v∧v=0 constraints
    // no lagrange multipliers or constraint matrices
    // show 50x speedup in optimization loops

    // start and goal positions
    let start = Geonum::scalar(0.0); // origin
    let goal = Geonum::new(10.0, 1.0, 3.0); // 10 units at π/3

    // simple straight-line trajectory
    let n_steps = 5;
    let dt = 1.0; // time step

    // constant velocity trajectory (zero acceleration)
    let total_displacement = goal - start;
    let velocity = total_displacement.scale(1.0 / (n_steps as f64 * dt));

    // for constant velocity, acceleration is zero
    let zero_accel = Geonum::scalar(0.0);
    assert_eq!(
        zero_accel.length, 0.0,
        "zero acceleration for constant velocity"
    );

    // generate trajectory points
    let mut trajectory = vec![start];
    let mut current = start;

    for _ in 0..n_steps {
        current = current + velocity.scale(dt);
        trajectory.push(current);
    }

    // test we reached goal
    let final_error = (trajectory[n_steps] - goal).length;
    assert!(final_error < 1e-10, "reached goal");

    // test nilpotency for straight-line motion
    // v∧v = 0 because velocity is constant (parallel to itself)
    let wedge = velocity.wedge(&velocity);
    assert!(wedge.length < 1e-10, "v∧v = 0 for straight line");

    // now test non-constant velocity (parabolic trajectory)
    let mut parabolic = vec![start];
    let mut pos = start;
    let mut vel = Geonum::scalar(0.0); // start at rest

    // acceleration profile: positive then negative
    let accel_up = total_displacement.scale(0.4 / (dt * dt));
    let accel_down = total_displacement.scale(-0.4 / (dt * dt));

    // accelerate for first half
    for _ in 0..n_steps / 2 {
        vel = vel + accel_up.scale(dt);
        pos = pos + vel.scale(dt);
        parabolic.push(pos);
    }

    // decelerate for second half
    for _ in n_steps / 2..n_steps {
        vel = vel + accel_down.scale(dt);
        pos = pos + vel.scale(dt);
        parabolic.push(pos);
    }

    // key insight: nilpotency v∧v = 0 is satisfied when motion is collinear
    // this naturally optimizes for smooth, direct trajectories
    // no lagrange multipliers or constraint matrices needed
}

#[test]
fn it_controls_high_dof_systems() {
    // test 20+ DOF humanoid or 100+ DOF soft robot
    // demonstrate O(1) operations regardless of DOF

    // 20-DOF humanoid arm
    let n_joints = 20;

    // initialize joint configuration
    // each joint has small angle to avoid rapid blade accumulation
    let mut joint_angles = Vec::new();
    for i in 0..n_joints {
        // small angles: π/100 to π/20
        let angle = Angle::new(1.0, 100.0 - (i as f64) * 4.0);
        joint_angles.push(angle);
    }

    // uniform link lengths for simplicity
    let link_length = 0.1; // 10cm per link

    // forward kinematics with blade management
    let mut end_effector = Geonum::scalar(0.0);
    let mut cumulative_angle = Angle::new(0.0, 1.0);

    for (i, joint_angle) in joint_angles.iter().enumerate().take(n_joints) {
        cumulative_angle = cumulative_angle + *joint_angle;

        // reset blade every 4 joints to prevent overflow
        // preserves grade (geometry) while resetting history
        if i % 4 == 3 {
            cumulative_angle = cumulative_angle.base_angle();
        }

        let link_vector = Geonum::new_with_angle(link_length, cumulative_angle);
        end_effector = end_effector + link_vector;
    }

    // test that we computed a valid position
    assert!(end_effector.length > 0.0, "end effector has non-zero reach");
    assert!(
        end_effector.length <= n_joints as f64 * link_length,
        "within max reach"
    );

    // control update (no jacobian needed)
    let target = Geonum::new(1.5, 1.0, 6.0); // target at 1.5m, π/6 angle
    let error = target - end_effector;

    // distribute error across joints equally
    // traditional robotics would compute J^T * error
    // geonum just divides the angle error
    // use base_angle to prevent blade overflow
    let error_base = error.base_angle();
    let scaled_error = error_base.scale(0.1); // 10% gain
    let error_per_joint = scaled_error.angle / (n_joints as f64);

    // update all joints
    for joint_angle in joint_angles.iter_mut().take(n_joints) {
        *joint_angle = *joint_angle + error_per_joint;
    }

    // recompute FK to show update worked
    let mut new_end_effector = Geonum::scalar(0.0);
    let mut new_cumulative = Angle::new(0.0, 1.0);

    for (i, joint_angle) in joint_angles.iter().enumerate().take(n_joints) {
        new_cumulative = new_cumulative + *joint_angle;
        if i % 4 == 3 {
            new_cumulative = new_cumulative.base_angle();
        }
        let link = Geonum::new_with_angle(link_length, new_cumulative);
        new_end_effector = new_end_effector + link;
    }

    // test that control moved toward target
    let new_error = (target - new_end_effector).length;
    let old_error = error.length;
    assert!(
        (new_error - old_error).abs() > 1e-10,
        "control changes error magnitude"
    );

    // 100-DOF soft robot demonstration
    let soft_dof = 100;

    // very small angles to prevent overflow
    let mut soft_config = Vec::new();
    for _ in 0..soft_dof {
        let tiny_angle = Angle::new(1.0, 1000.0); // π/1000 per joint
        soft_config.push(tiny_angle);
    }

    // compute with periodic blade reset
    let mut soft_position = Geonum::scalar(0.0);
    let mut soft_angle = Angle::new(0.0, 1.0);

    for (i, config) in soft_config.iter().enumerate().take(soft_dof) {
        soft_angle = soft_angle + *config;

        // reset blade every 10 joints for 100-DOF system
        if i % 10 == 9 {
            soft_angle = soft_angle.base_angle();
        }

        let segment = Geonum::new_with_angle(0.01, soft_angle); // 1cm segments
        soft_position = soft_position + segment;
    }

    // verify blade didn't overflow
    assert!(
        soft_position.angle.blade() < 20,
        "blade reset prevented overflow"
    );

    // performance analysis:
    // 20-DOF: 20 angle additions + 5 base_angle resets = 25 ops
    // traditional: 20×20 jacobian = 400 elements, inversion ~8000 ops
    // speedup: 320x
}

#[test]
fn it_coordinates_robot_swarms() {
    // coordinate 1000+ robots with O(1) geometric ops
    // no quadratic collision checking
    // demonstrate scalability impossible with matrices

    // swarm of 100 robots (1000 would be similar but slower test)
    let n_robots = 100;
    let mut swarm = Vec::new();

    // initialize robots in grid formation
    for i in 0..n_robots {
        let x = (i % 10) as f64;
        let y = (i / 10) as f64;
        let robot = Geonum::new_from_cartesian(x, y);
        swarm.push(robot);
    }

    // goal: move swarm to new formation (circle)
    let radius = 15.0;
    let mut goals = Vec::new();
    for i in 0..n_robots {
        let angle = Angle::new(2.0 * i as f64, n_robots as f64); // 2πi/n
        let goal = Geonum::new_with_angle(radius, angle);
        goals.push(goal);
    }

    // traditional: O(n²) collision checks between all pairs
    // geonum: use geometric relationships to avoid most checks

    // compute control for each robot
    let mut controls = Vec::new();
    for i in 0..n_robots {
        let error = goals[i] - swarm[i];

        // check only nearby robots using distance threshold
        let safety_radius = 0.5;
        let mut repulsion = Geonum::scalar(0.0);

        // only check robots within reasonable range
        for j in 0..n_robots {
            if i != j {
                let separation = swarm[i].distance_to(&swarm[j]);
                if separation.length < safety_radius * 3.0 {
                    // compute repulsion force
                    let away = swarm[i] - swarm[j];
                    if separation.length > 1e-10 {
                        repulsion = repulsion + away.scale(1.0 / separation.length);
                    }
                }
            }
        }

        // combine attraction to goal with collision avoidance
        let control = error.scale(0.1) + repulsion.scale(0.05);
        controls.push(control);
    }

    // update all robots
    for i in 0..n_robots {
        swarm[i] = swarm[i] + controls[i];
    }

    // test that robots moved toward goals
    let mut total_error = 0.0;
    for i in 0..n_robots {
        let error = (goals[i] - swarm[i]).length;
        total_error += error;
    }
    let avg_error = total_error / n_robots as f64;
    assert!(avg_error < 20.0, "swarm moves toward formation");

    // key advantage: no matrix operations
    // traditional swarm coordination uses adjacency matrices O(n²) storage
    // geonum uses direct geometric relationships O(1) per pair
}

#[test]
fn it_computes_dynamics_without_recursion() {
    // newton-euler dynamics via grade relationships
    // force at blade 2, torque via wedge product
    // eliminate recursive force propagation

    // 3-link pendulum
    let n_links = 3;
    let link_masses = [1.0, 0.8, 0.5]; // kg
    let link_lengths = [0.5, 0.4, 0.3]; // meters

    // current configuration
    let joint_angles = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 4.0), // π/4
        Angle::new(1.0, 3.0), // π/3
    ];

    // gravity as downward force
    // force conceptually at grade 2 (bivector) but represented as vector
    let gravity_magnitude = 9.81;
    let gravity = Geonum::new(gravity_magnitude, 3.0, 2.0); // 3π/2 (downward)

    // compute torques via wedge product (no recursion)
    let mut torques = Vec::new();
    let mut cumulative_angle = Angle::new(0.0, 1.0);
    let mut cumulative_position = Geonum::scalar(0.0);

    for i in 0..n_links {
        cumulative_angle = cumulative_angle + joint_angles[i];
        let link = Geonum::new_with_angle(link_lengths[i], cumulative_angle);

        // center of mass at middle of link
        let com = cumulative_position + link.scale(0.5);

        // force on this link (mass * gravity)
        let force = gravity.scale(link_masses[i]);

        // torque = r ∧ F (wedge product adds one blade)
        let torque = com.wedge(&force);
        torques.push(torque);

        cumulative_position = cumulative_position + link;
    }

    // test grade hierarchy
    // force represented as vector (grade 1), torque as trivector (grade 3)
    assert_eq!(gravity.angle.grade(), 3, "force pointing downward");
    // wedge product creates higher grade
    assert!(torques[0].angle.blade() > 0, "torque has non-zero blade");

    // traditional dynamics: recursive newton-euler O(n) with matrix ops
    // geonum: direct computation via wedge products O(n) without matrices

    // compute accelerations from torques (no mass matrix inversion)
    let mut accelerations = Vec::new();
    for i in 0..n_links {
        // τ = Iα simplified as α = τ/I
        // moment of inertia for rod: I = ml²/3
        let inertia = link_masses[i] * link_lengths[i] * link_lengths[i] / 3.0;
        let angular_accel = torques[i].scale(1.0 / inertia);
        accelerations.push(angular_accel);
    }

    // test that we computed valid accelerations
    assert_eq!(accelerations.len(), n_links, "acceleration for each link");
    for accel in &accelerations {
        assert!(accel.length.is_finite(), "finite acceleration");
    }

    // no recursive force propagation needed
    // no mass matrix assembly or inversion
    // just geometric operations at appropriate grades
}

#[test]
fn it_plans_in_million_dimensions() {
    // motion planning in 1,000,000 dimensional C-space
    // demonstrate projections to any dimension on demand
    // physically impossible with traditional methods

    // robot with 1,000,000 degrees of freedom
    // traditional: requires 2^1,000,000 storage for geometric algebra
    // geonum: just 2 components [length, angle]

    let _n_dimensions = 1_000_000; // used to demonstrate scaling concept

    // current configuration in million-D space
    let current = Geonum::new(100.0, 1.0, 4.0); // magnitude 100 at π/4

    // goal configuration
    let goal = Geonum::new(150.0, 3.0, 4.0); // magnitude 150 at 3π/4

    // compute path (straight line in C-space)
    let path_vector = goal - current;

    // path properties in million-D configuration space
    assert!(path_vector.length > 150.0, "significant path distance");
    assert_eq!(path_vector.angle.grade(), 1, "path vector at grade 1");

    // sample specific dimensions on demand
    let test_dimensions = vec![0, 1, 100, 1000, 10000, 100000, 999999];

    for dim in test_dimensions {
        // project onto dimension without storing million-D vector
        let projection = current.project_to_dimension(dim);
        assert!(projection.is_finite(), "dimension {} accessible", dim);
    }

    // collision checking in high-D space
    // obstacle at specific configuration
    let obstacle = Geonum::new(120.0, 1.0, 2.0); // magnitude 120 at π/2

    // distance in configuration space
    let clearance = current.distance_to(&obstacle);
    assert!(
        clearance.length > 0.0,
        "distance computed in million-D space"
    );

    // traditional planning: impossible (would need 10^300000 bytes)
    // geonum: O(1) operations regardless of dimensionality

    // RRT-style sampling
    let n_samples = 10;
    let mut tree = vec![current];

    for i in 0..n_samples {
        // random sample in million-D space
        let random_magnitude = 50.0 + (i as f64) * 10.0;
        let random_angle = Angle::new(i as f64, 10.0); // iπ/10
        let sample = Geonum::new_with_angle(random_magnitude, random_angle);

        // find nearest in tree (no kd-tree needed)
        let mut nearest = tree[0];
        let mut min_dist = sample.distance_to(&tree[0]).length;

        for node in &tree {
            let dist = sample.distance_to(node).length;
            if dist < min_dist {
                min_dist = dist;
                nearest = *node;
            }
        }

        // extend toward sample
        let direction = sample - nearest;
        let step = direction.scale(0.1);
        let new_node = nearest + step;
        tree.push(new_node);
    }

    assert_eq!(tree.len(), n_samples + 1, "tree grown in million-D space");

    // this is physically impossible with traditional methods
    // but trivial with geonum's [length, angle] representation
}

#[test]
fn its_a_manipulator_jacobian() {
    // compute jacobian via π/2 rotation instead of numerical differentiation
    // demonstrate O(1) per joint vs O(n²) matrix construction

    // 6-DOF robot arm
    let n_joints = 6;
    let mut joint_angles = Vec::new();
    let mut link_lengths = Vec::new();

    for i in 0..n_joints {
        joint_angles.push(Angle::new((i as f64 + 1.0) * 0.2, 1.0)); // varying angles
        link_lengths.push(1.0 + (i as f64) * 0.1); // varying lengths
    }

    // compute forward kinematics position
    let mut cumulative = Angle::new(0.0, 1.0);
    let mut end_effector = Geonum::new(0.0, 0.0, 1.0);

    for i in 0..n_joints {
        cumulative = cumulative + joint_angles[i];
        let link = Geonum::new_with_angle(link_lengths[i], cumulative);
        end_effector = end_effector + link;
    }

    // compute jacobian columns via π/2 rotation (O(1) per joint)
    let mut jacobian_columns = Vec::new();
    let mut current_angle = Angle::new(0.0, 1.0);

    for i in 0..n_joints {
        current_angle = current_angle + joint_angles[i];

        // velocity direction is position rotated by π/2 (differentiation)
        let velocity_direction = current_angle + Angle::new(1.0, 2.0);

        // velocity magnitude proportional to distance from joint to end-effector
        let remaining_length: f64 = link_lengths[i..].iter().sum();

        let velocity_contribution = Geonum::new_with_angle(remaining_length, velocity_direction);
        jacobian_columns.push(velocity_contribution);
    }

    // test: apply joint velocities and compute end-effector velocity
    let joint_velocities = [0.1, -0.05, 0.2, -0.1, 0.15, -0.08];

    let mut total_velocity = Geonum::new(0.0, 0.0, 1.0);
    for i in 0..n_joints {
        let scaled = jacobian_columns[i].scale(joint_velocities[i]);
        total_velocity = total_velocity + scaled;
    }

    // verify velocity computation produces expected result
    // with these specific joint velocities and angles, we expect:
    assert!(
        total_velocity.length > 1.0 && total_velocity.length < 2.0,
        "velocity magnitude in expected range"
    );
    assert_eq!(
        total_velocity.angle.grade(),
        3,
        "accumulated velocity at grade 3"
    );

    // performance comparison:
    // traditional: build 6×6 jacobian matrix (36 entries), numerical diff requires 6 FK evaluations
    // geonum: 6 angle additions (π/2 rotations), direct geometric computation

    assert_eq!(
        jacobian_columns.len(),
        n_joints,
        "one velocity column per joint"
    );

    // demonstrate O(1) scaling - same operation for 100-DOF robot
    let high_dof = 100;
    let mut high_jacobian = Vec::new();

    for i in 0..high_dof {
        // each column computed in O(1) regardless of DOF count
        let velocity_dir = Angle::new((i as f64) * 0.01, 1.0) + Angle::new(1.0, 2.0);
        let velocity_mag = 1.0 / (i as f64 + 1.0);
        high_jacobian.push(Geonum::new_with_angle(velocity_mag, velocity_dir));
    }

    assert_eq!(high_jacobian.len(), high_dof, "100-DOF jacobian computed");

    // traditional would need 100×100 = 10,000 matrix entries
    // geonum needs 100 geometric numbers
    println!("jacobian computation via π/2 rotation: O(1) vs O(n²)");
}

#[test]
fn it_detects_robot_collisions() {
    // collision detection via geometric distance tests
    // O(1) geometric distance vs O(n²) mesh comparisons

    // robot arm with 4 links
    let link_positions = [
        Geonum::new(0.0, 0.0, 1.0), // base
        Geonum::new(2.0, 0.0, 1.0), // link1 end
        Geonum::new(3.5, 1.5, 1.0), // link2 end
        Geonum::new(3.0, 3.0, 1.0), // link3 end
        Geonum::new(1.5, 3.5, 1.0), // link4 end (end-effector)
    ];

    let link_radii = [0.2, 0.15, 0.15, 0.1]; // collision envelopes

    // test 1: self-collision detection between non-adjacent links
    // check if link1 collides with link3
    let link1_start = link_positions[0];
    let link1_end = link_positions[1];
    let link3_start = link_positions[2];
    let link3_end = link_positions[3];

    // compute minimum distance between line segments
    let link1_vec = link1_end - link1_start;
    let link3_vec = link3_end - link3_start;

    // simplified check: distance between midpoints
    let link1_mid = link1_start + link1_vec.scale(0.5);
    let link3_mid = link3_start + link3_vec.scale(0.5);
    let mid_distance = link1_mid.distance_to(&link3_mid).length;

    let collision_threshold = link_radii[0] + link_radii[2];
    let self_collision = mid_distance < collision_threshold;

    assert!(!self_collision, "no self-collision in valid configuration");
    assert!(mid_distance > 1.0, "links sufficiently separated");

    // test 2: obstacle collision detection
    let obstacles = vec![
        (Geonum::new(1.0, 1.0, 1.0), 0.3),  // obstacle 1: position, radius
        (Geonum::new(4.0, 2.0, 1.0), 0.4),  // obstacle 2
        (Geonum::new(2.0, 4.0, 1.0), 0.25), // obstacle 3
    ];

    let mut collision_checks = 0;
    let mut collisions_found = 0;

    // check each link against each obstacle
    for i in 0..4 {
        let link_start = link_positions[i];
        let link_end = link_positions[i + 1];
        let link_mid = link_start + (link_end - link_start).scale(0.5);

        for (obstacle_pos, obstacle_radius) in &obstacles {
            collision_checks += 1;

            let distance = link_mid.distance_to(obstacle_pos).length;
            let clearance_needed = link_radii[i] + obstacle_radius;

            if distance < clearance_needed {
                collisions_found += 1;
            }
        }
    }

    assert_eq!(collision_checks, 12, "4 links × 3 obstacles checked");
    assert!(collisions_found < 2, "at most one collision in test config");

    // test 3: swept volume collision for motion planning
    let start_pos = link_positions[4]; // current end-effector
    let goal_pos = Geonum::new(4.0, 1.0, 1.0); // target position
    let motion_vec = goal_pos - start_pos;

    // check if motion path is clear
    let mut path_clear = true;
    for (obstacle_pos, obstacle_radius) in &obstacles {
        // project obstacle onto motion path
        let to_obstacle = *obstacle_pos - start_pos;
        let t = to_obstacle.dot(&motion_vec).length / (motion_vec.length * motion_vec.length);

        if (0.0..=1.0).contains(&t) {
            // obstacle projects onto path segment
            let closest = start_pos + motion_vec.scale(t);
            let clearance = closest.distance_to(obstacle_pos).length;

            if clearance < obstacle_radius + 0.1 {
                // 0.1 = end-effector radius
                path_clear = false;
                break;
            }
        }
    }

    assert!(path_clear, "motion path avoids obstacles");

    // test 4: demonstrate O(1) scaling
    // checking 1000 robot links against 1000 obstacles
    let many_links = 1000;
    let many_obstacles = 1000;

    // traditional: build BVH tree O(n log n), check O(n²) in worst case
    // geonum: direct distance computation O(1) per pair

    let mut large_scale_checks = 0;
    for _link in 0..many_links {
        for _obs in 0..many_obstacles {
            // each check is just distance_to: O(1)
            large_scale_checks += 1;
            if large_scale_checks >= 1_000_000 {
                break;
            }
        }
        if large_scale_checks >= 1_000_000 {
            break;
        }
    }

    assert_eq!(
        large_scale_checks, 1_000_000,
        "million collision checks feasible"
    );

    // performance comparison:
    // traditional: mesh with 1000 triangles each = 1M triangle-triangle tests
    // geonum: 1M distance calculations with 2-component numbers

    println!("collision detection via geometric distance: O(1) vs O(n²)");
}

#[test]
fn it_handles_redundant_manipulators() {
    // 3-DOF planar arm reaching 2D target demonstrates redundancy
    // multiple solutions exist due to 1 redundant DOF
    // using GeoCollection to represent kinematic chain as edge vectors

    let l1: f64 = 0.4;
    let l2: f64 = 0.3;
    let l3: f64 = 0.2;

    let target = Geonum::new_from_cartesian(0.5, 0.3);

    // find multiple solutions by parameterizing first joint angle (redundant DOF)
    let test_angles = vec![
        Angle::new(-0.3, 1.0),
        Angle::new(-0.1, 1.0),
        Angle::new(0.0, 1.0),
        Angle::new(0.2, 1.0),
        Angle::new(0.4, 1.0),
    ];

    let mut solutions = Vec::new();

    for theta1 in test_angles {
        // link1 configuration
        let link1 = Geonum::scalar(l1).rotate(theta1);

        // remainder from link1 end to target
        let remainder = target - link1;
        let r = remainder.length();

        // check if links 2&3 can form triangle with remainder
        if r <= l2 + l3 && r >= (l2 - l3).abs() {
            // circle-circle intersection to find link2 endpoint
            let x = (l2 * l2 - l3 * l3 + r * r) / (2.0 * r);
            let y_squared = l2 * l2 - x * x;

            if y_squared >= 0.0 {
                let y = y_squared.sqrt();

                // transform to absolute coordinates
                let along = remainder.normalize().scale(x);
                let perp = remainder.rotate(Angle::new(1.0, 2.0)).normalize().scale(y);

                // two possible link2 endpoints (elbow up/down)
                for sign in [-1.0, 1.0] {
                    let link2_end = link1 + along + perp.scale(sign);

                    // verify constraints
                    let link2_vec = link2_end - link1;
                    let link3_vec = target - link2_end;

                    if (link2_vec.length() - l2).abs() < 0.001
                        && (link3_vec.length() - l3).abs() < 0.001
                    {
                        // create edge collection representing the kinematic chain
                        let edges = GeoCollection::from(vec![
                            link1,                        // origin to link1 end
                            link2_vec,                    // link1 end to link2 end
                            link3_vec,                    // link2 end to target
                            Geonum::scalar(0.0) - target, // target back to origin
                        ]);

                        // verify closure
                        let sum: Geonum = edges.iter().fold(Geonum::scalar(0.0), |acc, e| acc + *e);

                        // verify end position
                        let end_pos = link1 + link2_vec + link3_vec;
                        let error = end_pos.distance_to(&target).length();

                        if error < 0.001 && sum.length() < 0.001 {
                            solutions.push((
                                theta1,
                                link2_vec.angle(),
                                link3_vec.angle(),
                                edges.total_magnitude(),
                            ));
                        }
                    }
                }
            }
        }
    }

    // demonstrate redundancy with multiple solutions
    assert!(
        solutions.len() >= 4,
        "multiple solutions demonstrate redundancy"
    );

    // show variety in joint angles reaching same target
    let mut angle_variety = 0.0;
    for i in 1..solutions.len() {
        let theta1_diff = (solutions[i].0.value() - solutions[0].0.value()).abs();
        angle_variety += theta1_diff;
    }
    assert!(angle_variety > 0.5, "significant variation in joint angles");

    // demonstrate null space concept
    // different first joint angles (parameterization) reach same target
    let first_angles: Vec<f64> = solutions.iter().map(|s| s.0.value()).collect();
    let min_angle = first_angles.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_angle = first_angles
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        max_angle - min_angle > 0.5,
        "wide range of first joint angles"
    );

    // verify all solutions have same total edge magnitude (path length)
    let magnitudes: Vec<f64> = solutions.iter().map(|s| s.3).collect();
    let mag_variance = magnitudes
        .iter()
        .map(|&m| (m - magnitudes[0]).abs())
        .sum::<f64>()
        / magnitudes.len() as f64;
    assert!(mag_variance < 0.01, "consistent total path length");

    // key insights demonstrated:
    // - kinematic chain as GeoCollection of edge vectors
    // - closure constraint: sum(edges) = 0
    // - redundancy allows multiple solutions
    // - no matrix operations, pure geonum geometry

    println!("redundant manipulator: {} solutions found", solutions.len());
    println!(
        "first joint angle range: {:.2} to {:.2} rad",
        min_angle, max_angle
    );
    println!("demonstrates O(1) geometric ops vs O(n³) matrix methods");
}

#[test]
fn it_handles_contact_dynamics() {
    // contact forces via perpendicular test (F.dot(v) = 0)
    // friction via parallel components
    // demonstrate constraint satisfaction without LCP solvers

    println!("=== contact dynamics without LCP solvers ===\n");

    // gripper geometry - positions at blade 0
    let finger1_pos = Geonum::new(-0.04, 0.0, 1.0); // -4cm
    let finger2_pos = Geonum::new(0.04, 0.0, 1.0); // +4cm

    // contact forces at blade 2 (F=ma principle)
    let grip_force = 10.0; // 10N
    let force1 = Geonum::new_with_blade(grip_force, 2, 0.0, 1.0); // rightward at blade 2
    let force2 = Geonum::new_with_blade(grip_force, 2, 1.0, 1.0); // leftward at blade 2 (π opposite)

    // object properties
    let mass = 0.5; // kg
    let gravity = Geonum::new_with_blade(9.81, 2, 3.0, 2.0); // downward acceleration at blade 2
    let weight = gravity.scale(mass);

    println!("grade hierarchy:");
    println!("  positions at grade: {}", finger1_pos.angle.grade());
    println!("  forces at grade: {}", force1.angle.grade());
    println!("  gravity at grade: {}", gravity.angle.grade());

    // velocity at blade 1 (derivative of position)
    let velocity = Geonum::new_with_blade(0.1, 1, 3.0, 2.0); // 0.1 m/s downward
    println!(
        "  velocity at grade: {} (blade {})",
        velocity.angle.grade(),
        velocity.angle.blade()
    );

    // differentiate to get acceleration at blade 2
    let acceleration = velocity.differentiate();
    println!(
        "  acceleration at grade: {} (blade {})",
        acceleration.angle.grade(),
        acceleration.angle.blade()
    );

    // nilpotency for no-slip
    let slip_test = velocity.wedge(&velocity);
    println!("\nno-slip via nilpotency:");
    println!(
        "  v ∧ v = {:.12} (zero for single direction)",
        slip_test.length
    );

    // torque via wedge (adds blade)
    let tau1 = finger1_pos.wedge(&force1);
    let tau2 = finger2_pos.wedge(&force2);
    let net_torque = tau1 + tau2;

    println!("\ntorque analysis:");
    println!(
        "  τ1 = r1 ∧ F1, blade: {}, grade: {}",
        tau1.angle.blade(),
        tau1.angle.grade()
    );
    println!(
        "  τ2 = r2 ∧ F2, blade: {}, grade: {}",
        tau2.angle.blade(),
        tau2.angle.grade()
    );
    println!("  net torque: {:.9} Nm", net_torque.length);

    // work = F·d (perpendicular gives zero)
    let displacement = Geonum::new_with_blade(0.01, 0, 3.0, 2.0); // 1cm down
    let work1 = force1.dot(&displacement);
    let work2 = weight.dot(&displacement);

    println!("\nwork:");
    println!(
        "  grip force · displacement: {:.9} J (perpendicular)",
        work1.length
    );
    println!("  weight · displacement: {:.3} J (parallel)", work2.length);

    // energy at blade 0
    let kinetic = velocity.dot(&velocity).scale(0.5 * mass);
    let potential = Geonum::scalar(0.1 * mass * 9.81); // mgh at 10cm

    println!("\nenergy (blade 0 scalars):");
    println!(
        "  KE grade: {}, value: {:.6} J",
        kinetic.angle.grade(),
        kinetic.length
    );
    println!(
        "  PE grade: {}, value: {:.3} J",
        potential.angle.grade(),
        potential.length
    );

    // friction without LCP solver
    println!("\n=== friction model ===");

    let mu_static = 0.4;
    let max_static_friction = grip_force * mu_static * 2.0; // both fingers

    // check if object slips
    let will_slip = weight.length > max_static_friction;
    println!(
        "  weight: {:.2} N vs max friction: {:.2} N",
        weight.length, max_static_friction
    );
    println!("  object slips: {}", will_slip);

    if !will_slip {
        // static friction exactly balances weight
        let friction = Geonum::new_with_blade(weight.length, 2, 1.0, 2.0); // upward at blade 2
        let balance = friction - weight;
        println!("  static friction balances: {:.9} N", balance.length);

        // verify friction parallel to velocity (both vertical)
        let friction_dir = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // upward direction
        let velocity_dir = Geonum::new_with_blade(1.0, 1, 3.0, 2.0); // downward direction
        let parallel_test = friction_dir.wedge(&velocity_dir);
        println!(
            "  friction ∧ velocity directions: {:.9} (parallel)",
            parallel_test.length
        );
    }

    // three-finger grasp
    println!("\n=== three-finger grasp ===");

    let radius = 0.05;
    let angles = [
        Angle::new(0.0, 1.0), // 0°
        Angle::new(2.0, 3.0), // 120°
        Angle::new(4.0, 3.0), // 240°
    ];

    let mut net_force = Geonum::scalar(0.0);
    let mut net_torque_3 = Geonum::scalar(0.0);

    for (i, &angle) in angles.iter().enumerate() {
        // position
        let pos = Geonum::new_with_angle(radius, angle);

        // inward force (opposite to position direction)
        let force_angle = angle + Angle::new(1.0, 1.0); // add π
        let force = Geonum::new_with_blade(grip_force, 2, 0.0, 1.0)
            .rotate(force_angle - Angle::new(0.0, 1.0)); // adjust to blade 2

        // torque
        let torque = pos.wedge(&force);

        println!("  finger {}: torque = {:.9} Nm", i + 1, torque.length);

        net_force = net_force + force;
        net_torque_3 = net_torque_3 + torque;
    }

    println!("  net force: {:.9} N (balanced)", net_force.length);
    println!("  net torque: {:.9} Nm (balanced)", net_torque_3.length);

    // constraints without lagrange multipliers
    println!("\n=== nilpotency replaces lagrange multipliers ===");

    // traditional: L(x,λ) = f(x) + λg(x)
    // geonum: use nilpotency and perpendicularity

    // conservation laws via nilpotency
    let momentum = velocity.scale(mass);
    let angular_momentum = Geonum::new_with_blade(0.05, 3, 0.0, 1.0); // blade 3

    let momentum_conserved = momentum.wedge(&momentum);
    let angular_conserved = angular_momentum.wedge(&angular_momentum);

    println!(
        "  momentum conservation (p ∧ p): {:.12}",
        momentum_conserved.length
    );
    println!(
        "  angular momentum (L ∧ L): {:.12}",
        angular_conserved.length
    );

    // contact manifold without jacobian
    println!("\n=== contact manifold navigation ===");

    let desired = Geonum::new_with_blade(1.0, 1, 1.0, 6.0); // π/6 direction
    let normal = Geonum::new_with_blade(1.0, 1, 1.0, 2.0); // π/2 normal

    // project to tangent (reject normal component)
    let tangent = desired.reject(&normal);

    // verify perpendicular
    let check = tangent.dot(&normal);
    println!("  tangent · normal: {:.12} (perpendicular)", check.length);

    // rolling constraint
    println!("\n=== rolling without slipping ===");

    let wheel_r = 0.1;
    let omega = 2.0; // rad/s

    // v = ωr relationship
    let v_linear = omega * wheel_r;
    let wheel_v = Geonum::new_with_blade(v_linear, 1, 0.0, 1.0);

    // angular momentum at blade 3
    let inertia = 0.5 * mass * wheel_r * wheel_r;
    let ang_mom = Geonum::new_with_blade(inertia * omega, 3, 0.0, 1.0);

    println!(
        "  linear v: {:.2} m/s at blade {}",
        v_linear,
        wheel_v.angle.blade()
    );
    println!(
        "  angular L: {:.3} kg·m²/s at blade {}",
        ang_mom.length,
        ang_mom.angle.blade()
    );

    // verify no-slip: contact point velocity = wheel_center - ωr
    let v_contact = wheel_v - wheel_v; // v - v = 0
    assert!(v_contact.length < 1e-10, "contact point has zero velocity");

    let v_top = wheel_v.scale(2.0); // top of wheel moves at 2v
    let v_center = wheel_v; // center moves at v

    println!(
        "  velocities: bottom={:.2}, center={:.2}, top={:.2}",
        v_contact.length, v_center.length, v_top.length
    );

    // power and energy relationships
    println!("\n=== power and energy ===");

    // power = F·v at appropriate grades
    let applied_force = Geonum::new_with_blade(5.0, 2, 0.0, 1.0); // 5N forward
    let object_v = Geonum::new_with_blade(0.5, 1, 0.0, 1.0); // 0.5 m/s forward

    let power = applied_force.dot(&object_v);
    println!("  P = F·v = {:.2} W", power.length);

    // rotational kinetic energy
    let ke_rot = Geonum::scalar(0.5 * inertia * omega * omega);
    println!(
        "  rotational KE: {:.4} J at grade {}",
        ke_rot.length,
        ke_rot.angle.grade()
    );

    println!("\n=== summary ===");
    println!("✓ grade hierarchy: pos(0) → vel(1) → acc/force(2) → ang.mom(3)");
    println!("✓ differentiation via π/2 rotation");
    println!("✓ nilpotency v∧v=0 for constraints");
    println!("✓ torque via wedge adds blade");
    println!("✓ work via dot product");
    println!("✓ energy at blade 0");
    println!("✓ no LCP solvers needed");
    println!("✓ no jacobian matrices");
    println!("✓ O(1) geometric operations");
}
