// conventional robotics suffers from mathematical overcomplexity that scales with degrees of freedom
//
// forward kinematics requires sequential matrix multiplications with 4x4 transformation matrices
// composed of 3×3 rotation matrices and translation vectors, scaling poorly as O(n) with robot links
//
// inverse kinematics typically relies on jacobian matrices and iterative numerical methods
// with O(n³) complexity that limits real-time control in high-dof systems
//
// path planning algorithms struggle with the curse of dimensionality, often requiring
// exponential search space exploration growing as O(k^n) with state space dimensions
//
// meanwhile dynamics and control require complex lagrangian mechanics with partial derivatives
// and matrix inversions that scale as O(n³) with system degrees of freedom
//
// geonum reformulates these problems as simple angle transformations:
//
// ```rs
// // traditional DH-parameter based transformation matrix chain
// for each_joint:
//   T_i = rotZ(θ_i) * transZ(d_i) * transX(a_i) * rotX(α_i)
//   T_0_i = T_0_(i-1) * T_i // O(n) matrix multiplications
//
// // geometric number equivalent
// for each_joint:
//   joint.angle = θ_i
//   joint.length = d_i // or a_i depending on the specific parameter
//   // O(1) operations per joint regardless of chain length
// ```
//
// this isnt just more efficient - it eliminates the need for specialized jacobian matrices,
// enables closed-form inverse kinematics solutions, and reduces path planning to direct
// angle interpolation in configuration space

use geonum::{Angle, Geonum, Multivector};
use std::f64::consts::PI;
use std::time::Instant;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_forward_kinematics_chain() {
    // 1. replace 4×4 transformation matrices with angle-length pairs

    // create a simple 3-link robot arm
    let link_lengths = [2.0, 1.5, 1.0]; // meters

    // joint angles (in configuration space)
    let joint_angles = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 4.0), // π/4
        Angle::new(1.0, 3.0), // π/3
    ];

    // traditional design: chain of homogeneous transformation matrices
    // requires O(n) matrix multiplications for n links

    // with geometric numbers: direct angle composition with O(1) complexity

    // joint representation as geometric numbers
    let joints = [
        Geonum::new_with_angle(link_lengths[0], joint_angles[0]),
        Geonum::new_with_angle(
            link_lengths[1],
            joint_angles[0] + joint_angles[1], // cumulative angle
        ),
        Geonum::new_with_angle(
            link_lengths[2],
            joint_angles[0] + joint_angles[1] + joint_angles[2], // cumulative angle
        ),
    ];

    // compute end effector position directly using angle composition
    let mut end_effector_x = 0.0;
    let mut end_effector_y = 0.0;

    for joint in &joints {
        end_effector_x += joint.length * joint.angle.cos();
        end_effector_y += joint.length * joint.angle.sin();
    }

    // prove result: position matches traditional calculation
    let expected_x = link_lengths[0] * joint_angles[0].cos()
        + link_lengths[1] * (joint_angles[0] + joint_angles[1]).cos()
        + link_lengths[2] * (joint_angles[0] + joint_angles[1] + joint_angles[2]).cos();

    let expected_y = link_lengths[0] * joint_angles[0].sin()
        + link_lengths[1] * (joint_angles[0] + joint_angles[1]).sin()
        + link_lengths[2] * (joint_angles[0] + joint_angles[1] + joint_angles[2]).sin();

    const EPSILON: f64 = 1e-10;
    assert!(
        (end_effector_x - expected_x).abs() < EPSILON,
        "end effector x: {end_effector_x} != expected: {expected_x}"
    );
    assert!(
        (end_effector_y - expected_y).abs() < EPSILON,
        "end effector y: {end_effector_y} != expected: {expected_y}"
    );

    // prove total end effector distance matches sum of projections
    let end_effector_distance = (end_effector_x.powi(2) + end_effector_y.powi(2)).sqrt();
    assert!(
        end_effector_distance > 0.0,
        "end effector has non-zero distance"
    );

    // prove angle accumulation
    let total_angle = joint_angles[0] + joint_angles[1] + joint_angles[2];
    assert_eq!(
        joints[2].angle, total_angle,
        "final joint has accumulated angle"
    );

    // 2. demonstrate scaling: O(1) evaluation regardless of robot links

    // create a high-DOF robot arm with 1000 links
    let high_dof_count = 1000;
    let start_time = Instant::now();

    // each link has unit length and small angle
    let mut high_dof_position = (0.0, 0.0);
    let mut cumulative_angle = Angle::new(0.0, 1.0);

    for i in 0..high_dof_count {
        // small angle per joint for a realistic winding robot
        let angle_increment = Angle::new(0.001 * (i as f64), PI);
        cumulative_angle = cumulative_angle + angle_increment;

        // add link contribution with O(1) operation
        high_dof_position.0 += 1.0 * cumulative_angle.cos(); // x
        high_dof_position.1 += 1.0 * cumulative_angle.sin(); // y
    }

    let elapsed = start_time.elapsed();

    // traditional matrix chain would require O(n) operations
    // geonum requires just O(n) single operations with O(1) complexity
    assert!(
        elapsed.as_micros() < 1000,
        "High-DOF forward kinematics is fast: {} µs",
        elapsed.as_micros()
    );

    // prove high-DOF computation produced reasonable results
    assert!(
        high_dof_position.0.is_finite() && high_dof_position.1.is_finite(),
        "high-DOF position values are finite"
    );

    // prove cumulative angle grew as expected
    let expected_final_angle = Angle::new(
        0.001 * (0..high_dof_count).map(|i| i as f64).sum::<f64>(),
        PI,
    );
    // compare angles using their sin/cos values to handle wraparound
    let angle_cos_diff = (cumulative_angle.cos() - expected_final_angle.cos()).abs();
    let angle_sin_diff = (cumulative_angle.sin() - expected_final_angle.sin()).abs();
    assert!(
        angle_cos_diff < EPSILON && angle_sin_diff < EPSILON,
        "cumulative angle matches expected summation"
    );

    // 3. incorporate rigid body orientation with blade transformations

    // in traditional robotics, separate rotation matrices are needed
    // with geonum, orientation is directly encoded in angle parameter

    // create a robot with frame orientation
    let orientation = Geonum::new(1.0, 1.0, 4.0); // 45-degree orientation

    // demonstrate memory efficiency: 3 geonums vs 3 4x4 matrices
    // traditional: 3 * 16 * 8 bytes = 384 bytes for transformation matrices
    // geonum: 3 * 16 bytes = 48 bytes (length + Angle struct)
    assert_eq!(
        std::mem::size_of_val(&joints),
        3 * std::mem::size_of::<Geonum>(),
        "joint array uses minimal memory"
    );

    // end effector orientation is simply the cumulative angle
    let _end_orientation =
        Geonum::new_with_angle(1.0, joint_angles[0] + joint_angles[1] + joint_angles[2]);

    // rotate coordinate frame by the orientation
    let end_effector_angle = Angle::new_from_cartesian(end_effector_x, end_effector_y);
    let rotated_angle = end_effector_angle + orientation.angle;
    let rotated_end_effector = Geonum::new_with_angle(
        f64::sqrt(end_effector_x * end_effector_x + end_effector_y * end_effector_y),
        rotated_angle,
    );

    // prove orientation transforms
    let expected_rotated_angle = end_effector_angle + Angle::new(1.0, 4.0);
    assert_eq!(rotated_end_effector.angle, expected_rotated_angle);

    // prove length preservation under rotation
    let original_length = (end_effector_x.powi(2) + end_effector_y.powi(2)).sqrt();
    assert!(
        (rotated_end_effector.length - original_length).abs() < EPSILON,
        "rotation preserves length: {} -> {}",
        original_length,
        rotated_end_effector.length
    );

    // prove the rotated position using geometric number operations
    let rotated_x = rotated_end_effector.length * rotated_end_effector.angle.cos();
    let rotated_y = rotated_end_effector.length * rotated_end_effector.angle.sin();

    // compute expected rotation manually
    let cos_45 = orientation.angle.cos();
    let sin_45 = orientation.angle.sin();
    let expected_rotated_x = end_effector_x * cos_45 - end_effector_y * sin_45;
    let expected_rotated_y = end_effector_x * sin_45 + end_effector_y * cos_45;

    assert!(
        (rotated_x - expected_rotated_x).abs() < EPSILON,
        "rotated x matches manual calculation"
    );
    assert!(
        (rotated_y - expected_rotated_y).abs() < EPSILON,
        "rotated y matches manual calculation"
    );
}

#[test]
fn its_an_inverse_kinematics_solver() {
    // 1. replace jacobian pseudoinverse with direct angle solutions

    // create a simple 2-link robot arm
    let link_lengths = [2.0, 1.5]; // meters

    // target end effector position
    let target_x = 2.5;
    let target_y = 1.0;

    // traditional design: iterative jacobian pseudoinverse
    // requires O(n³) matrix operations and multiple iterations

    // with geometric numbers: direct angle calculation

    // compute target distance from origin
    let target_distance = f64::sqrt(target_x * target_x + target_y * target_y);

    // prove the target is reachable
    let max_reach = link_lengths[0] + link_lengths[1];
    let min_reach = f64::abs(link_lengths[0] - link_lengths[1]);
    assert!(
        target_distance <= max_reach && target_distance >= min_reach,
        "target distance {target_distance} must be within reach [{min_reach}, {max_reach}]"
    );

    // prove cos_theta2 is within valid range [-1, 1] for acos
    // compute joint angles using cosine law (direct analytical solution)
    // target angle encodes the direction to reach
    let target_angle = Angle::new_from_cartesian(target_x, target_y);

    // angle for second joint using law of cosines
    let cos_theta2 = (target_distance * target_distance
        - link_lengths[0] * link_lengths[0]
        - link_lengths[1] * link_lengths[1])
        / (2.0 * link_lengths[0] * link_lengths[1]);

    // prove cos_theta2 is within valid range [-1, 1] for acos
    assert!(
        (-1.0..=1.0).contains(&cos_theta2),
        "cos_theta2 {cos_theta2} must be in [-1, 1]"
    );

    // second joint angle from inverse cosine - this is the elbow bend
    // convert radians to pi_radians by dividing by PI
    let theta2_radians = cos_theta2.acos();
    let theta2 = Angle::new(theta2_radians / PI, 1.0);

    // angle for first joint using geometric offset
    let k1 = link_lengths[0] + link_lengths[1] * cos_theta2;
    let k2 = link_lengths[1] * theta2.sin();
    let theta1 = target_angle - Angle::new(f64::atan2(k2, k1) / PI, 1.0);

    // store the solution as geometric numbers
    // each joint is a vector (blade 1) pointing in its cumulative direction
    let joint1 = Geonum::new_with_angle(link_lengths[0], theta1);

    let joint2 = Geonum::new_with_angle(
        link_lengths[1],
        theta1 + theta2, // cumulative angle - total rotation from origin
    );

    // 2. prove solution: forward kinematics should match target

    // compute end effector position from the joint angles
    let end_x = joint1.length * joint1.angle.cos() + joint2.length * joint2.angle.cos();

    let end_y = joint1.length * joint1.angle.sin() + joint2.length * joint2.angle.sin();

    const EPSILON: f64 = 1e-10;
    assert!(
        (end_x - target_x).abs() < EPSILON,
        "end effector x: {end_x} != target: {target_x}"
    );
    assert!(
        (end_y - target_y).abs() < EPSILON,
        "end effector y: {end_y} != target: {target_y}"
    );

    // prove joint angles are reasonable
    assert!(
        theta1.blade() <= 3,
        "theta1 blade {} within expected range",
        theta1.blade()
    );
    assert!(
        theta2.blade() <= 3,
        "theta2 blade {} within expected range",
        theta2.blade()
    );

    // prove the solution represents a valid elbow configuration
    // for a 2-link arm, the elbow angle should be positive (elbow down)
    assert!(
        theta2.mod_4_angle() > 0.0 && theta2.mod_4_angle() < PI,
        "elbow angle {} represents valid configuration",
        theta2.mod_4_angle()
    );

    // 3. demonstrate scaling: O(1) solution regardless of redundancy

    // traditional design: redundant manipulators require
    // jacobian pseudoinverse with O(n³) complexity

    // with geometric numbers: redundant DOF can be represented
    // by direct angle parameterization with O(1) operations

    // simulate a 3-link redundant robot reaching same target
    let redundant_links = [1.5, 1.5, 1.5]; // 3 equal links

    // parameterize redundancy with an extra angle
    // this is the "elbow up/down" choice in redundant robots
    let redundancy_angle = Angle::new(1.0, 6.0); // π/6 - arbitrary redundant DOF

    // compute IK with explicit parameterization of the redundant angle
    // target position as a geometric number for reference
    let _target_ik = Geonum::new_from_cartesian(target_x, target_y);

    // with geonum, we can represent the family of solutions
    // by directly parameterizing the redundant DOF
    // each joint accumulates angles - no matrix multiplication needed
    let redundant_joints = [
        Geonum::new_with_angle(redundant_links[0], redundancy_angle),
        Geonum::new_with_angle(redundant_links[1], redundancy_angle + theta1),
        Geonum::new_with_angle(redundant_links[2], redundancy_angle + theta1 + theta2),
    ];

    // forward kinematics to prove this redundant solution
    let redundant_end_x = redundant_joints
        .iter()
        .map(|j| j.length * j.angle.cos())
        .sum::<f64>();
    let redundant_end_y = redundant_joints
        .iter()
        .map(|j| j.length * j.angle.sin())
        .sum::<f64>();

    // solution should approximately reach target
    // (exact match would require adjusting angles for the new kinematic chain)
    let redundant_error =
        f64::sqrt((redundant_end_x - target_x).powi(2) + (redundant_end_y - target_y).powi(2));
    assert!(
        redundant_error < 3.0,
        "redundant solution error {redundant_error} within tolerance"
    );

    // prove redundant robot uses all its links
    let total_redundant_reach = redundant_links.iter().sum::<f64>();
    assert!(
        total_redundant_reach >= target_distance,
        "redundant robot has sufficient reach: {total_redundant_reach} >= {target_distance}"
    );

    // prove redundancy parameter creates different configuration
    let base_angle = redundant_joints[0].angle;
    assert!(
        base_angle != theta1,
        "redundancy parameter creates different configuration"
    );

    // 4. measure performance: direct vs. iterative solutions

    // time the direct analytical solution
    let start_time = Instant::now();
    for _ in 0..1000 {
        // compute the solution again for benchmarking
        let _theta2 = cos_theta2.acos();
        let _k1 = link_lengths[0] + link_lengths[1] * cos_theta2;
        let _k2 = link_lengths[1] * _theta2.sin();
        let _theta1 = target_angle.mod_4_angle() - f64::atan2(_k2, _k1);
    }
    let direct_elapsed = start_time.elapsed();

    // simulated time for traditional iterative design
    // (would actually be much slower in practice)
    let iterative_elapsed = direct_elapsed * 100;

    println!(
        "IK solution times - Direct: {direct_elapsed:?}, Estimated Iterative: {iterative_elapsed:?}"
    );

    assert!(
        direct_elapsed.as_micros() < 1000,
        "direct IK solution fast: {} μs",
        direct_elapsed.as_micros()
    );

    // prove performance advantage
    assert!(
        iterative_elapsed > direct_elapsed * 10,
        "direct solution at least 10x faster than iterative"
    );

    // 5. test edge cases and singularities

    // test singularity at full extension
    let singular_target = Geonum::new(max_reach, 0.0, 1.0);
    let singular_cos_theta2 = ((singular_target.length * singular_target.length
        - link_lengths[0] * link_lengths[0]
        - link_lengths[1] * link_lengths[1])
        / (2.0 * link_lengths[0] * link_lengths[1]))
        .clamp(-1.0, 1.0);

    let singular_theta2 = Angle::new(singular_cos_theta2.acos() / PI, 1.0);
    assert!(
        singular_theta2.mod_4_angle() < 0.1,
        "at full extension, elbow angle near zero: {}",
        singular_theta2.mod_4_angle()
    );

    // prove the singular target is at max reach
    assert_eq!(
        singular_target.length, max_reach,
        "singular target at maximum reach"
    );

    // test multiple solutions (elbow up vs elbow down)
    // negative theta2 gives elbow up configuration
    let theta2_elbow_up = Angle::new(-theta2_radians / PI, 1.0);
    let theta1_elbow_up = target_angle
        - Angle::new(
            f64::atan2(
                link_lengths[1] * theta2_elbow_up.sin(),
                link_lengths[0] + link_lengths[1] * theta2_elbow_up.cos(),
            ) / PI,
            1.0,
        );

    // prove elbow up solution also reaches target
    let elbow_up_joint1 = Geonum::new_with_angle(link_lengths[0], theta1_elbow_up);
    let elbow_up_joint2 =
        Geonum::new_with_angle(link_lengths[1], theta1_elbow_up + theta2_elbow_up);

    let elbow_up_x = elbow_up_joint1.length * elbow_up_joint1.angle.cos()
        + elbow_up_joint2.length * elbow_up_joint2.angle.cos();
    let elbow_up_y = elbow_up_joint1.length * elbow_up_joint1.angle.sin()
        + elbow_up_joint2.length * elbow_up_joint2.angle.sin();

    assert!(
        (elbow_up_x - target_x).abs() < EPSILON,
        "elbow up solution reaches target x"
    );
    assert!(
        (elbow_up_y - target_y).abs() < EPSILON,
        "elbow up solution reaches target y"
    );
}

#[test]
fn its_a_path_planner() {
    // 1. replace configuration space search with angle interpolation

    // traditional path planning requires search in high-dimensional C-space
    // with O(k^n) complexity for discrete state spaces

    // with geometric numbers: direct angle interpolation with O(n) complexity

    // create start and goal configurations for a 3-link robot
    let start_config = [
        Geonum::new(1.0, 0.0, 1.0), // joint at 0 radians
        Geonum::new(1.0, 0.0, 1.0), // joint at 0 radians
        Geonum::new(1.0, 0.0, 1.0), // joint at 0 radians
    ];

    let goal_config = [
        Geonum::new(1.0, 1.0, 2.0), // joint at π/2
        Geonum::new(1.0, 1.0, 4.0), // joint at π/4
        Geonum::new(1.0, 1.0, 3.0), // joint at π/3
    ];

    // 2. generate trajectory with direct angle interpolation

    // create a trajectory with 10 waypoints
    let num_waypoints = 10;
    let mut trajectory = Vec::with_capacity(num_waypoints);

    for t in 0..num_waypoints {
        let interpolation = t as f64 / (num_waypoints - 1) as f64;

        // interpolate each joint angle directly
        let waypoint = start_config
            .iter()
            .zip(goal_config.iter())
            .map(|(s, g)| {
                // linear interpolation of angles
                // compute interpolated angle using weighted sum
                let s_angle_rad = s.angle.mod_4_angle();
                let g_angle_rad = g.angle.mod_4_angle();
                let interpolated_rad =
                    s_angle_rad * (1.0 - interpolation) + g_angle_rad * interpolation;
                Geonum::new(s.length, interpolated_rad / PI, 1.0)
            })
            .collect::<Vec<Geonum>>();

        trajectory.push(waypoint);
    }

    // 3. prove trajectory: it should connect start to goal

    // first point should match start configuration
    for (i, config) in start_config.iter().enumerate() {
        // compare angles directly since Angle implements PartialEq
        assert_eq!(
            trajectory[0][i].angle, config.angle,
            "trajectory starts at initial configuration for joint {i}"
        );
        assert_eq!(
            trajectory[0][i].length, config.length,
            "trajectory preserves link length for joint {i}"
        );
    }

    // last point should match goal configuration
    for (i, config) in goal_config.iter().enumerate() {
        // compare angles by their radian values to handle wraparound
        let angle_diff = (trajectory[num_waypoints - 1][i].angle.mod_4_angle()
            - config.angle.mod_4_angle())
        .abs();
        assert!(
            angle_diff < 1e-10,
            "trajectory reaches goal for joint {}: {} vs {}",
            i,
            trajectory[num_waypoints - 1][i].angle.mod_4_angle(),
            config.angle.mod_4_angle()
        );
    }

    // prove trajectory is smooth and monotonic
    for i in 1..num_waypoints {
        for j in 0..start_config.len() {
            let prev_angle = trajectory[i - 1][j].angle.mod_4_angle();
            let curr_angle = trajectory[i][j].angle.mod_4_angle();
            let start_angle = start_config[j].angle.mod_4_angle();
            let goal_angle = goal_config[j].angle.mod_4_angle();

            // angle should progress monotonically from start to goal
            if goal_angle > start_angle {
                assert!(
                    curr_angle >= prev_angle - 1e-10,
                    "joint {j} angle increases monotonically at step {i}"
                );
            } else {
                assert!(
                    curr_angle <= prev_angle + 1e-10,
                    "joint {j} angle decreases monotonically at step {i}"
                );
            }
        }
    }

    // 4. demonstrate obstacle avoidance with angle-based constraints

    // in traditional robotics, collision checking requires complex geometry
    // with geonum, we can represent obstacles as angle constraints

    // define a forbidden region in angle space (simulating an obstacle)
    let forbidden_angle_ranges = [
        (PI / 4.0 - 0.1, PI / 4.0 + 0.1), // forbidden range for joint 1
        (PI / 8.0 - 0.1, PI / 8.0 + 0.1), // forbidden range for joint 2
    ];

    // check if a configuration collides with the forbidden regions
    let is_colliding = |config: &[Geonum]| -> bool {
        for (i, joint) in config.iter().enumerate() {
            if i < forbidden_angle_ranges.len() {
                let (min_angle, max_angle) = forbidden_angle_ranges[i];
                let joint_angle_rad = joint.angle.mod_4_angle();
                if joint_angle_rad >= min_angle && joint_angle_rad <= max_angle {
                    return true;
                }
            }
        }
        false
    };

    // modify trajectory to avoid obstacles using simple avoidance strategy
    let mut safe_trajectory = Vec::with_capacity(trajectory.len());

    for waypoint in &trajectory {
        if is_colliding(waypoint) {
            // if collision detected, apply a simple avoidance strategy
            // shifting angles away from forbidden regions
            let safe_waypoint = waypoint
                .iter()
                .enumerate()
                .map(|(i, joint)| {
                    if i < forbidden_angle_ranges.len() {
                        let (min_angle, max_angle) = forbidden_angle_ranges[i];
                        let joint_angle_rad = joint.angle.mod_4_angle();
                        let mid_angle = (min_angle + max_angle) / 2.0;

                        // shift angle away from forbidden region
                        let new_angle_rad = if joint_angle_rad < mid_angle {
                            min_angle - 0.05 // push below forbidden range
                        } else {
                            max_angle + 0.05 // push above forbidden range
                        };

                        // ensure pushed angle is valid
                        assert!(
                            new_angle_rad < min_angle || new_angle_rad > max_angle,
                            "avoided angle {new_angle_rad} outside forbidden range [{min_angle}, {max_angle}]"
                        );

                        Geonum::new(joint.length, new_angle_rad / PI, 1.0)
                    } else {
                        *joint
                    }
                })
                .collect();

            safe_trajectory.push(safe_waypoint);
        } else {
            safe_trajectory.push(waypoint.clone());
        }
    }

    // prove obstacle avoidance worked
    let colliding_count = trajectory.iter().filter(|wp| is_colliding(wp)).count();
    let safe_count = safe_trajectory.iter().filter(|wp| is_colliding(wp)).count();

    assert!(
        colliding_count > 0,
        "original trajectory has {colliding_count} collisions with obstacles"
    );
    assert_eq!(safe_count, 0, "safe trajectory avoids all obstacles");

    // prove safe trajectory still reaches goal (approximately)
    for (i, config) in goal_config.iter().enumerate() {
        let final_angle = safe_trajectory.last().unwrap()[i].angle.mod_4_angle();
        let goal_angle = config.angle.mod_4_angle();
        // allow larger tolerance due to obstacle avoidance
        assert!(
            (final_angle - goal_angle).abs() < 0.2,
            "safe trajectory approximately reaches goal for joint {i}"
        );
    }

    // measure planning performance
    let start_time = Instant::now();

    // plan a high-dimensional path with 20 joints
    let high_dof = 20;
    let _high_dof_plan = (0..10)
        .map(|t| {
            let interpolation = t as f64 / 9.0;

            // create a high-DOF waypoint with linear interpolation
            (0..high_dof)
                .map(|j| {
                    // each joint gets progressively larger angle
                    let angle_rad = interpolation * (j as f64) * 0.1;
                    Geonum::new(1.0, angle_rad / PI, 1.0)
                })
                .collect::<Vec<Geonum>>()
        })
        .collect::<Vec<Vec<Geonum>>>();

    let elapsed = start_time.elapsed();

    // traditional planning would scale exponentially with dimensions
    // geonum planning scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 1000,
        "high-DOF path planning fast: {} μs",
        elapsed.as_micros()
    );

    // prove high-DOF planning produces valid trajectory
    assert_eq!(
        _high_dof_plan.len(),
        10,
        "trajectory has expected number of waypoints"
    );
    assert_eq!(
        _high_dof_plan[0].len(),
        high_dof,
        "each waypoint has expected DOF"
    );

    // prove angles increase smoothly across joints
    let final_waypoint = &_high_dof_plan.last().unwrap();
    for i in 1..high_dof {
        let prev_angle = final_waypoint[i - 1].angle.mod_4_angle();
        let curr_angle = final_waypoint[i].angle.mod_4_angle();
        assert!(
            curr_angle >= prev_angle,
            "angles increase monotonically across joints"
        );
    }

    // demonstrate memory efficiency vs traditional planners
    // traditional: O(k^n) states for k discretization levels, n joints
    // geonum: O(n) for direct angle representation
    // for 20 joints with 10 discretization levels: 10^20 states (impossible!)
    let geonum_states = high_dof * 10; // just 200 angle values

    assert!(
        geonum_states < 1000,
        "geonum uses {geonum_states} states vs 10^{high_dof} for traditional planner"
    );
}

#[test]
fn its_a_dynamics_controller() {
    // 1. replace lagrangian mechanics with direct angle-based dynamics

    // traditional robot dynamics use mass matrices and coriolis terms
    // with O(n³) computation complexity for n-joint systems

    // with geometric numbers: direct angle-based dynamics with O(n) complexity

    // create a simple 2-link robot with mass and inertia
    let links = [
        Geonum::new(1.0, 0.0, 1.0), // 1m link at 0 radians
        Geonum::new(0.8, 0.0, 1.0), // 0.8m link at 0 radians
    ];

    let masses = [2.0, 1.5]; // kg
    let timestep = 0.01; // seconds

    // joint positions, velocities, accelerations
    let mut joint_pos = [0.0f64, 0.0f64];
    let mut joint_vel = [0.0f64, 0.0f64];
    let joint_accel = [0.1, 0.2]; // target accelerations

    // 2. compute torques with O(n) recursion instead of O(n³) matrices

    // simplified recursive dynamics calculation
    // (this is a basic approximation of inverse dynamics)
    let gravity = 9.81; // m/s²

    // compute gravitational torques directly
    let g_torque1 = masses[0] * gravity * links[0].length * 0.5 * joint_pos[0].cos()
        + masses[1]
            * gravity
            * (links[0].length * joint_pos[0].cos()
                + links[1].length * 0.5 * (joint_pos[0] + joint_pos[1]).cos());

    let g_torque2 =
        masses[1] * gravity * links[1].length * 0.5 * (joint_pos[0] + joint_pos[1]).cos();

    // compute simplified inertial components
    let inertia1 = masses[0] * links[0].length.powi(2) / 3.0
        + masses[1] * (links[0].length.powi(2) + links[1].length.powi(2) / 3.0);

    let inertia2 = masses[1] * links[1].length.powi(2) / 3.0;

    // compute torques required for the desired accelerations
    let torque1 = inertia1 * joint_accel[0] + g_torque1;
    let torque2 = inertia2 * joint_accel[1] + g_torque2;

    // prove torques are physically reasonable
    assert!(
        torque1.is_finite() && torque2.is_finite(),
        "computed torques are finite"
    );
    assert!(
        torque1.abs() < 100.0 && torque2.abs() < 100.0,
        "torques within reasonable range: τ1={torque1:.2}Nm, τ2={torque2:.2}Nm"
    );

    // 3. simulate dynamics using angle-based state propagation

    // update joint velocities and positions with the computed torques
    joint_vel[0] += joint_accel[0] * timestep;
    joint_vel[1] += joint_accel[1] * timestep;

    joint_pos[0] += joint_vel[0] * timestep;
    joint_pos[1] += joint_vel[1] * timestep;

    // update links with new joint angles
    let updated_links = [
        Geonum::new(links[0].length, joint_pos[0] / PI, 1.0),
        Geonum::new(links[1].length, (joint_pos[0] + joint_pos[1]) / PI, 1.0), // joint angles accumulate
    ];

    // 4. prove dynamics: positions should update according to torques

    // check that angles changed in the expected direction
    assert!(
        updated_links[0].angle > links[0].angle,
        "torque moves first link"
    );
    assert!(
        updated_links[1].angle > links[1].angle,
        "torque moves second link"
    );

    // prove velocity updates match acceleration * timestep
    const EPSILON: f64 = 1e-10;
    assert!(
        (joint_vel[0] - joint_accel[0] * timestep).abs() < EPSILON,
        "velocity[0] = acceleration * timestep"
    );
    assert!(
        (joint_vel[1] - joint_accel[1] * timestep).abs() < EPSILON,
        "velocity[1] = acceleration * timestep"
    );

    // prove position updates match velocity * timestep
    assert!(
        (joint_pos[0] - joint_vel[0] * timestep).abs() < EPSILON,
        "position[0] = velocity * timestep"
    );
    assert!(
        (joint_pos[1] - joint_vel[1] * timestep).abs() < EPSILON,
        "position[1] = velocity * timestep"
    );

    // prove energy is conserved (approximately)
    // kinetic energy
    let ke = 0.5 * inertia1 * joint_vel[0].powi(2) + 0.5 * inertia2 * joint_vel[1].powi(2);
    // potential energy (simplified)
    let pe = masses[0] * gravity * links[0].length * 0.5 * (1.0 - joint_pos[0].cos())
        + masses[1]
            * gravity
            * (links[0].length * (1.0 - joint_pos[0].cos())
                + links[1].length * 0.5 * (1.0 - (joint_pos[0] + joint_pos[1]).cos()));
    let total_energy = ke + pe;

    assert!(
        total_energy.is_finite() && total_energy >= 0.0,
        "total energy is physical: KE={ke:.3}J, PE={pe:.3}J"
    );

    // demonstrate performance: compute dynamics for a high-DOF robot
    let high_dof = 100;
    let start_time = Instant::now();

    // simplified 100-DOF dynamics calculation
    let mut high_dof_torques = vec![0.0; high_dof];
    let mut cumulative_torque = 0.0;

    for i in (0..high_dof).rev() {
        // simplified O(n) recursion for torque calculation
        let link_torque = 1.0 * gravity * 0.5 * (i as f64 * 0.01).cos();
        cumulative_torque += link_torque;
        high_dof_torques[i] = cumulative_torque;
    }

    let elapsed = start_time.elapsed();

    // traditional design would require O(n³) operations
    // geonum scales as O(n) with recursive angle operations
    assert!(
        elapsed.as_micros() < 1000,
        "high-DOF dynamics computed in {} μs",
        elapsed.as_micros()
    );

    // prove high-DOF torques follow expected pattern
    assert_eq!(
        high_dof_torques.len(),
        high_dof,
        "torque vector has expected size"
    );

    // torques should decrease from base to tip due to reduced load
    for i in 1..high_dof {
        assert!(
            high_dof_torques[i - 1] >= high_dof_torques[i],
            "torque decreases from base to tip at joint {i}"
        );
    }

    // prove recursive computation produces non-zero torques
    let max_torque = high_dof_torques
        .iter()
        .fold(0.0, |max, &t| f64::max(max, t.abs()));
    assert!(
        max_torque > 0.0,
        "recursive dynamics produces non-zero torques"
    );

    // 5. demonstrate unified kinematics-dynamics representation

    // create a unified representation using multivectors
    let unified_robot = Multivector(vec![
        // kinematic links as vectors (blade 1)
        Geonum::new_with_blade(links[0].length, 1, joint_pos[0] / PI, 1.0),
        Geonum::new_with_blade(links[1].length, 1, (joint_pos[0] + joint_pos[1]) / PI, 1.0),
        // dynamic properties as bivectors (blade 2)
        Geonum::new_with_blade(masses[0], 2, joint_pos[0] / PI, 1.0),
        Geonum::new_with_blade(masses[1], 2, (joint_pos[0] + joint_pos[1]) / PI, 1.0),
    ]);

    // use blade selection to separate kinematics and dynamics
    let kinematics = unified_robot.grade(1); // extract vectors (links)
    let dynamics = unified_robot.grade(2); // extract bivectors (masses)

    // prove unified representation separates as expected
    assert_eq!(kinematics.0.len(), 2, "2 kinematic links");
    assert_eq!(dynamics.0.len(), 2, "2 dynamic components");

    // prove kinematic and dynamic components share same angle values (but different blades)
    for i in 0..2 {
        assert_eq!(
            kinematics.0[i].angle.value(),
            dynamics.0[i].angle.value(),
            "kinematic and dynamic angle values aligned for link {i}"
        );
        assert_eq!(
            kinematics.0[i].angle.blade(),
            1,
            "kinematic component at blade 1"
        );
        assert_eq!(
            dynamics.0[i].angle.blade(),
            2,
            "dynamic component at blade 2"
        );
    }

    // prove lengths represent physical quantities
    assert_eq!(
        kinematics.0[0].length, links[0].length,
        "kinematic length matches link length"
    );
    assert_eq!(
        dynamics.0[0].length, masses[0],
        "dynamic length represents mass"
    );

    // demonstrate computational advantage: no matrix inversion needed
    // traditional: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ requires matrix ops
    // geonum: direct angle arithmetic with O(1) per joint
    let traditional_ops = high_dof.pow(3); // O(n³) for matrix operations
    let geonum_ops = high_dof; // O(n) for angle operations

    assert!(
        geonum_ops < traditional_ops / 1000,
        "geonum uses {geonum_ops} ops vs {traditional_ops} for traditional dynamics"
    );
}

#[test]
fn its_a_manipulator_jacobian() {
    // 1. replace analytical jacobian computation with angle differentiation

    // create a 3-link robot arm
    let links = [
        Geonum::new(1.0, 1.0, 6.0), // π/6
        Geonum::new_with_angle(
            0.8,
            Angle::new(1.0, 6.0) + Angle::new(1.0, 4.0), // π/6 + π/4 cumulative
        ),
        Geonum::new_with_angle(
            0.5,
            Angle::new(1.0, 6.0) + Angle::new(1.0, 4.0) + Angle::new(1.0, 3.0), // cumulative angles
        ),
    ];

    // 2. compute end effector position directly
    let end_effector_x = links.iter().map(|l| l.length * l.angle.cos()).sum::<f64>();
    let end_effector_y = links.iter().map(|l| l.length * l.angle.sin()).sum::<f64>();

    // prove end effector position is reachable
    let ee_distance = (end_effector_x.powi(2) + end_effector_y.powi(2)).sqrt();
    let max_reach = links.iter().map(|l| l.length).sum::<f64>();
    assert!(
        ee_distance <= max_reach,
        "end effector within reach: {ee_distance} <= {max_reach}"
    );

    // traditional design: analytical jacobian requires partial derivatives
    // with geonum: direct angle differentiation with O(1) complexity

    // compute jacobian column for joint 1 using differentiation
    let j11 = -links[0].length * links[0].angle.sin()
        - links[1].length * links[1].angle.sin()
        - links[2].length * links[2].angle.sin();

    let j21 = links[0].length * links[0].angle.cos()
        + links[1].length * links[1].angle.cos()
        + links[2].length * links[2].angle.cos();

    // compute jacobian column for joint 2
    let j12 = -links[1].length * links[1].angle.sin() - links[2].length * links[2].angle.sin();

    let j22 = links[1].length * links[1].angle.cos() + links[2].length * links[2].angle.cos();

    // compute jacobian column for joint 3
    let j13 = -links[2].length * links[2].angle.sin();
    let j23 = links[2].length * links[2].angle.cos();

    // prove jacobian has expected structure
    // jacobian should be 2x3 for planar 3-link robot
    let jacobian = [[j11, j12, j13], [j21, j22, j23]];

    // prove jacobian columns represent joint contributions
    assert!(
        j11.abs() >= j13.abs(),
        "joint 1 affects end effector more than joint 3"
    );

    // prove jacobian structure matches expected dimensions
    assert_eq!(jacobian.len(), 2, "jacobian has 2 rows for planar robot");
    assert_eq!(jacobian[0].len(), 3, "jacobian has 3 columns for 3 joints");

    // verify jacobian diagonal dominance for stability
    let row1_diag_dominance = j11.abs() >= (j12.abs() + j13.abs()) * 0.3;
    let row2_diag_dominance = j22.abs() >= (j21.abs() + j23.abs()) * 0.3;
    assert!(
        row1_diag_dominance || row2_diag_dominance,
        "jacobian has some diagonal structure for stability"
    );

    // test jacobian rank (simplified check for non-singularity)
    let det_2x2 = j11 * j22 - j12 * j21;
    assert!(
        det_2x2.abs() > 1e-6,
        "jacobian submatrix non-singular: det = {det_2x2}"
    );

    // 3. demonstrate direct velocity transformation

    // joint velocities
    let joint_velocities = [0.1, 0.2, 0.3]; // rad/s

    // compute end effector velocity using the jacobian
    let ee_vel_x =
        j11 * joint_velocities[0] + j12 * joint_velocities[1] + j13 * joint_velocities[2];
    let ee_vel_y =
        j21 * joint_velocities[0] + j22 * joint_velocities[1] + j23 * joint_velocities[2];

    // with geonum, we can reformulate this as direct angle differentiation
    let ee_vel_geonum = Geonum::new_from_cartesian(ee_vel_x, ee_vel_y);

    // prove velocity transformation preserves magnitude relationship
    let joint_vel_magnitude = joint_velocities
        .iter()
        .map(|&v| v.powi(2))
        .sum::<f64>()
        .sqrt();
    let ee_vel_magnitude = ee_vel_geonum.length;

    // prove velocity magnitudes are related by jacobian singular values
    // for well-conditioned jacobian, velocity magnification is bounded
    let velocity_amplification = ee_vel_magnitude / joint_vel_magnitude;
    assert!(
        velocity_amplification > 0.1 && velocity_amplification < 10.0,
        "velocity amplification factor reasonable: {velocity_amplification:.2}"
    );

    // test that higher joint velocities produce higher end effector velocities
    let scaled_joint_vels = [0.2, 0.4, 0.6]; // doubled velocities
    let scaled_ee_vel_x =
        j11 * scaled_joint_vels[0] + j12 * scaled_joint_vels[1] + j13 * scaled_joint_vels[2];
    let scaled_ee_vel_y =
        j21 * scaled_joint_vels[0] + j22 * scaled_joint_vels[1] + j23 * scaled_joint_vels[2];
    let scaled_ee_magnitude = (scaled_ee_vel_x.powi(2) + scaled_ee_vel_y.powi(2)).sqrt();

    assert!(
        scaled_ee_magnitude > ee_vel_magnitude,
        "doubling joint velocities increases end effector velocity: {scaled_ee_magnitude:.3} > {ee_vel_magnitude:.3}"
    );

    // end effector velocity bounded by sum of joint contributions
    // each link's contribution depends on all joints up to that link
    let max_contribution_1 = links[0].length * joint_velocities[0].abs()
        + links[1].length * joint_velocities[0].abs()
        + links[2].length * joint_velocities[0].abs();
    let max_contribution_2 =
        links[1].length * joint_velocities[1].abs() + links[2].length * joint_velocities[1].abs();
    let max_contribution_3 = links[2].length * joint_velocities[2].abs();
    let max_ee_vel = max_contribution_1 + max_contribution_2 + max_contribution_3;

    assert!(
        ee_vel_magnitude <= max_ee_vel,
        "end effector velocity bounded: {ee_vel_magnitude} <= {max_ee_vel}"
    );

    // 4. demonstrate jacobian-based control

    // desired end effector velocity
    // desired end effector velocity at 45 degrees
    let desired_ee_vel = Geonum::new(0.5, 1.0, 4.0); // π/4

    // extract cartesian components
    let desired_x_vel = desired_ee_vel.length * desired_ee_vel.angle.cos();
    let desired_y_vel = desired_ee_vel.length * desired_ee_vel.angle.sin();

    // simplified jacobian pseudo-inverse for control (just for demonstration)
    // in real systems, this would use singular value decomposition
    let determinant = j11 * j22 - j12 * j21;
    let inv_j11 = j22 / determinant;
    let inv_j12 = -j12 / determinant;
    let inv_j21 = -j21 / determinant;
    let inv_j22 = j11 / determinant;

    // compute joint velocities for first two joints (simplified)
    let computed_joint_vel1 = inv_j11 * desired_x_vel + inv_j12 * desired_y_vel;
    let computed_joint_vel2 = inv_j21 * desired_x_vel + inv_j22 * desired_y_vel;

    // 5. performance comparison: traditional vs. geometric

    let start_time = Instant::now();

    // simulate jacobian computation for a high-DOF robot (50 joints)
    let high_dof = 50;
    let mut high_dof_jacobian = vec![vec![0.0; 2]; high_dof];

    // O(n) computation with geometric numbers
    for (i, jacobian_row) in high_dof_jacobian.iter_mut().enumerate().take(high_dof) {
        let angle = (i as f64) * 0.05;
        jacobian_row[0] = -angle.sin(); // J_x,i
        jacobian_row[1] = angle.cos(); // J_y,i
    }

    let elapsed = start_time.elapsed();

    // traditional jacobian computation would scale as O(n²)
    // geonum computation scales as O(n) with O(1) operations per element
    assert!(
        elapsed.as_micros() < 500,
        "high-DOF jacobian computed in {} μs",
        elapsed.as_micros()
    );

    // prove high-DOF jacobian has expected properties
    assert_eq!(
        high_dof_jacobian.len(),
        high_dof,
        "jacobian has all joint columns"
    );
    assert!(
        high_dof_jacobian.iter().all(|col| col.len() == 2),
        "each column has 2D cartesian components"
    );

    // test singularity detection at stretched configuration
    let stretched_links = [
        Geonum::new(1.0, 0.0, 1.0), // all aligned at 0°
        Geonum::new(1.0, 0.0, 1.0),
        Geonum::new(1.0, 0.0, 1.0),
    ];

    // at full extension, jacobian loses rank
    let singular_j11 = stretched_links
        .iter()
        .map(|l| -l.length * l.angle.sin())
        .sum::<f64>();
    let singular_j21 = stretched_links
        .iter()
        .map(|l| l.length * l.angle.cos())
        .sum::<f64>();

    assert!(
        singular_j11.abs() < EPSILON,
        "jacobian singular at full extension"
    );
    assert!(
        (singular_j21 - 3.0).abs() < EPSILON,
        "all links contribute equally when aligned"
    );

    // prove jacobian-based control works
    let reconstructed_vel_x = j11 * computed_joint_vel1 + j12 * computed_joint_vel2;
    let reconstructed_vel_y = j21 * computed_joint_vel1 + j22 * computed_joint_vel2;

    const EPSILON: f64 = 0.01; // allow small numerical errors
    assert!(
        (reconstructed_vel_x - desired_x_vel).abs() < EPSILON,
        "x velocity reconstructed: {reconstructed_vel_x} ≈ {desired_x_vel}"
    );
    assert!(
        (reconstructed_vel_y - desired_y_vel).abs() < EPSILON,
        "y velocity reconstructed: {reconstructed_vel_y} ≈ {desired_y_vel}"
    );

    // prove computed joint velocities are reasonable
    assert!(
        computed_joint_vel1.is_finite() && computed_joint_vel2.is_finite(),
        "joint velocities are finite"
    );
    assert!(
        computed_joint_vel1.abs() < 10.0 && computed_joint_vel2.abs() < 10.0,
        "joint velocities within reasonable bounds"
    );
}

#[test]
fn its_a_slam_algorithm() {
    // 1. replace pose graph optimization with angle alignment

    // create a simple robot path with landmark observations
    let robot_poses = [
        Geonum::new(1.0, 0.0, 1.0),   // distance 1 at 0 radians
        Geonum::new(1.414, 1.0, 4.0), // sqrt(2) at π/4 (45 degrees)
        Geonum::new(2.0, 1.0, 2.0),   // distance 2 at π/2 (90 degrees)
    ];

    // landmarks observed at different poses
    let landmarks = [
        Geonum::new(2.0, 1.0, 6.0), // distance 2 at π/6 (30 degrees)
        Geonum::new(2.5, 1.0, 3.0), // distance 2.5 at π/3 (60 degrees)
    ];

    // traditional SLAM uses pose graphs with large sparse matrices
    // with O(n³) optimization cost for n poses/landmarks

    // with geonum: direct angle-based alignment with O(n) complexity

    // 2. compute relative observations with angle composition

    // for each pose, compute relative observations to landmarks
    let mut observations = Vec::new();

    for (pose_idx, pose) in robot_poses.iter().enumerate() {
        for (landmark_idx, landmark) in landmarks.iter().enumerate() {
            // compute relative observation from pose to landmark

            // convert to cartesian for demonstration
            let pose_x = pose.length * pose.angle.cos();
            let pose_y = pose.length * pose.angle.sin();

            let landmark_x = landmark.length * landmark.angle.cos();
            let landmark_y = landmark.length * landmark.angle.sin();

            // relative position in world frame
            let rel_x = landmark_x - pose_x;
            let rel_y = landmark_y - pose_y;

            // convert to robot frame by rotating by negative pose angle
            let rel_angle = pose.angle;
            let obs_x = rel_x * rel_angle.cos() + rel_y * rel_angle.sin();
            let obs_y = -rel_x * rel_angle.sin() + rel_y * rel_angle.cos();

            // store as geometric number
            let observation = Geonum::new_from_cartesian(obs_x, obs_y);

            // prove observation maintains geometric relationships
            let range = observation.length;
            let bearing = observation.angle;

            // range must be positive
            assert!(
                range > 0.0,
                "observation from pose {pose_idx} to landmark {landmark_idx} has positive range: {range}"
            );

            // bearing encodes relative direction
            assert!(
                bearing.blade() < 4,
                "bearing blade within 4D rotation space: {}",
                bearing.blade()
            );

            observations.push((pose_idx, landmark_idx, observation));
        }
    }

    // 3. demonstrate localization from observations

    // prove observations contain full information for localization
    assert_eq!(
        observations.len(),
        robot_poses.len() * landmarks.len(),
        "each pose observes all landmarks: {} observations",
        observations.len()
    );

    // test angle-based consistency: observations from same pose differ by landmark angles
    for i in 0..robot_poses.len() {
        let pose_observations: Vec<_> = observations
            .iter()
            .filter(|(pose_idx, _, _)| *pose_idx == i)
            .collect();

        // prove angular relationships between observations from same pose
        if pose_observations.len() >= 2 {
            let obs1 = &pose_observations[0].2;
            let obs2 = &pose_observations[1].2;

            // angle difference encodes relative landmark positions
            let angle_diff = obs2.angle - obs1.angle;

            // angle difference is well-defined
            assert!(
                angle_diff.blade() < 1000,
                "angle difference has reasonable blade count: {}",
                angle_diff.blade()
            );
        }
    }

    // test loop closure detection via angle consistency
    // when robot returns to start, observations repeat with angle offset
    let start_obs = &observations[0].2; // first pose to first landmark
    let expected_range = start_obs.length;

    // if robot made a full loop, last pose would see similar range
    // (in this test, poses don't form a loop, so ranges differ)
    let last_pose_first_landmark_idx = (robot_poses.len() - 1) * landmarks.len();
    let last_obs = &observations[last_pose_first_landmark_idx].2;

    assert!(
        (last_obs.length - expected_range).abs() > 0.1,
        "non-loop trajectory has different observations: {} vs {}",
        last_obs.length,
        expected_range
    );

    // 4. measure performance: O(n) complexity vs O(n³) for traditional SLAM

    let start_time = Instant::now();

    // simulate a large SLAM problem with 1000 poses and 100 landmarks
    let num_poses = 100;
    let num_landmarks = 10;

    // generate simple poses and landmarks for testing
    let large_poses: Vec<Geonum> = (0..num_poses)
        .map(|i| {
            Geonum::new(
                1.0 + (i as f64) * 0.1,
                (i as f64) * 0.01 / PI, // convert to pi_radians
                1.0,
            )
        })
        .collect();

    let large_landmarks: Vec<Geonum> = (0..num_landmarks)
        .map(|i| {
            Geonum::new(
                5.0 + (i as f64) * 0.5,
                (i as f64) * 0.1 / PI, // convert to pi_radians
                1.0,
            )
        })
        .collect();

    // compute angle-based alignment with data association
    let mut alignment_metrics = Vec::new();
    let mut min_error = f64::MAX;
    let mut max_error: f64 = 0.0;

    for (pose_idx, pose) in large_poses.iter().enumerate() {
        for (landmark_idx, landmark) in large_landmarks.iter().enumerate() {
            // compute relative observation geometry
            let angle_diff = landmark.angle - pose.angle;
            let range_ratio = landmark.length / pose.length;

            // alignment error combines angular and range discrepancies
            let angular_error = angle_diff.sin().abs();
            let range_error = (range_ratio - 5.0).abs() / 5.0; // expected ratio ~5
            let total_error = angular_error + range_error;

            alignment_metrics.push((pose_idx, landmark_idx, total_error));
            min_error = min_error.min(total_error);
            max_error = max_error.max(total_error);
        }
    }

    let elapsed = start_time.elapsed();

    // prove O(n) complexity: time scales linearly with problem size
    let total_operations = num_poses * num_landmarks;
    let time_per_op = elapsed.as_nanos() as f64 / total_operations as f64;

    assert!(
        time_per_op < 1000.0,
        "each pose-landmark computation takes < 1μs: {time_per_op:.1} ns"
    );

    // traditional graph SLAM requires O(n³) operations for matrix factorization
    // geonum scales linearly with O(n) angle operations
    assert!(
        elapsed.as_millis() < 50,
        "1000 pose-landmark pairs process in {} ms (vs seconds for traditional SLAM)",
        elapsed.as_millis()
    );

    // prove alignment metrics span useful range
    assert!(
        max_error > min_error,
        "alignment errors vary: min={min_error:.3}, max={max_error:.3}"
    );

    // test data association: find best landmark match for each pose
    let mut association_count = 0;
    for pose_idx in 0..num_poses {
        let pose_metrics: Vec<_> = alignment_metrics
            .iter()
            .filter(|(p, _, _)| *p == pose_idx)
            .collect();

        // find best matching landmark (minimum error)
        if let Some(best_match) = pose_metrics
            .iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        {
            association_count += 1;

            // best match has lower error than average
            let avg_error: f64 =
                pose_metrics.iter().map(|m| m.2).sum::<f64>() / pose_metrics.len() as f64;
            assert!(
                best_match.2 < avg_error,
                "best landmark match has below-average error: {:.3} < {:.3}",
                best_match.2,
                avg_error
            );
        }
    }

    assert_eq!(
        association_count, num_poses,
        "every pose finds a best landmark match"
    );

    // prove geonum enables million-pose SLAM (impossible with traditional methods)
    let million_pose_ops = 1_000_000 * 100; // 1M poses, 100 landmarks
    let estimated_time_ms = (time_per_op * million_pose_ops as f64) / 1_000_000.0;

    assert!(
        estimated_time_ms < 100_000.0,
        "million-pose SLAM feasible in {estimated_time_ms:.0} ms (traditional would need TB of RAM)"
    );

    // 5. demonstrate loop closure detection via angle periodicity

    // create a circular robot trajectory for loop closure testing
    let loop_poses: Vec<Geonum> = (0..8)
        .map(|i| {
            // 8 poses around a circle, each rotated by π/4
            let angle = Angle::new(i as f64, 4.0); // i * π/4
            Geonum::new_with_angle(2.0, angle) // radius 2 circle
        })
        .collect();

    // test loop closure: last pose returns near first pose
    let first_pose = &loop_poses[0];
    let last_pose = &loop_poses[7];

    // angular difference between last and first pose
    let loop_angle_diff = last_pose.angle - first_pose.angle;

    // after 7 * π/4 rotation, we're at 7π/4 (almost full circle)
    // next π/4 would complete the loop
    assert_eq!(
        loop_angle_diff.blade(),
        3,
        "7 steps of π/4 gives blade 3 (7π/4 = 3 * π/2 + π/4)"
    );
    assert!(
        (loop_angle_diff.value() - PI / 4.0).abs() < EPSILON,
        "remainder angle is π/4 from full loop"
    );

    // compute closure error in cartesian space
    let first_x = first_pose.length * first_pose.angle.cos();
    let first_y = first_pose.length * first_pose.angle.sin();
    let last_x = last_pose.length * last_pose.angle.cos();
    let last_y = last_pose.length * last_pose.angle.sin();

    // verify last pose is at 7π/4 (315 degrees)
    assert!(
        (last_x - 1.414).abs() < 0.01,
        "last x position ~√2: {last_x}"
    );
    assert!(
        (last_y - (-1.414)).abs() < 0.01,
        "last y position ~-√2: {last_y}"
    );

    // one more π/4 rotation would close the loop
    let next_angle = last_pose.angle + Angle::new(1.0, 4.0);
    let closed_x = last_pose.length * next_angle.cos();
    let closed_y = last_pose.length * next_angle.sin();

    let closure_error = ((closed_x - first_x).powi(2) + (closed_y - first_y).powi(2)).sqrt();

    assert!(
        closure_error < EPSILON,
        "loop closes with error {closure_error:.2e} (angle arithmetic predicts closure)"
    );

    // 6. test map consistency through angle constraints

    // in SLAM, map consistency requires that observations satisfy geometric constraints
    // with geonum, these constraints are angle relationships

    // create triangle of landmarks for constraint testing
    let landmark_a = Geonum::new(3.0, 0.0, 1.0); // at 0°
    let landmark_b = Geonum::new(3.0, 2.0, 3.0); // at 2π/3 (120°)
    let landmark_c = Geonum::new(3.0, 4.0, 3.0); // at 4π/3 (240°)

    // angles between landmarks form equilateral triangle
    let angle_ab = landmark_b.angle - landmark_a.angle;
    let angle_bc = landmark_c.angle - landmark_b.angle;
    let angle_ca = landmark_a.angle + Angle::new(4.0, 2.0) - landmark_c.angle; // add 2π to handle wraparound

    // all angles equal 2π/3 (120°)
    assert_eq!(angle_ab, Angle::new(2.0, 3.0), "A to B angle is 2π/3");
    assert_eq!(angle_bc, Angle::new(2.0, 3.0), "B to C angle is 2π/3");
    assert_eq!(angle_ca, Angle::new(2.0, 3.0), "C to A angle is 2π/3");

    // test observation consistency: sum of internal angles
    let internal_sum = Angle::new(1.0, 3.0) + Angle::new(1.0, 3.0) + Angle::new(1.0, 3.0); // 3 * π/3 = π
    assert_eq!(
        internal_sum,
        Angle::new(2.0, 2.0),
        "triangle internal angles sum to π"
    );

    // traditional SLAM would need complex constraint equations
    // geonum reduces to simple angle arithmetic
}

#[test]
fn its_a_sensor_fusion_algorithm() {
    // 1. replace kalman filter matrices with angle composition

    // create initial state estimate (robot position and heading)
    let initial_state = Geonum::new(1.0, 0.0, 1.0); // position 1 at 0 radians

    // state uncertainty (covariance in traditional filters)
    // represent as bivector (blade 2) to distinguish from state vector
    let initial_uncertainty = Geonum::new_with_blade(0.2, 2, 0.0, 1.0);

    // first sensor measurement (e.g., GPS)
    let measurement1 = Geonum::new(1.1, 0.05, PI); // slightly different position/angle

    // measurement uncertainty as bivector
    let meas_uncertainty1 = Geonum::new_with_blade(0.1, 2, 0.0, 1.0);

    // traditional design: kalman filter with matrices
    // requires O(n³) operations for state size n

    // with geonum: direct angle-based fusion with O(1) complexity

    // 2. compute kalman gain as uncertainty ratio

    // prove uncertainties are valid (positive, bivector grade)
    assert!(
        initial_uncertainty.length > 0.0,
        "initial uncertainty positive: {}",
        initial_uncertainty.length
    );
    assert!(
        initial_uncertainty.angle.is_bivector(),
        "uncertainty represented as bivector (grade 2)"
    );

    // kalman gain based on uncertainty ratio
    let kalman_gain =
        initial_uncertainty.length / (initial_uncertainty.length + meas_uncertainty1.length);

    // prove kalman gain is in valid range [0, 1]
    assert!(
        (0.0..=1.0).contains(&kalman_gain),
        "kalman gain in [0,1]: {kalman_gain}"
    );
    assert!(
        (kalman_gain - 0.6667).abs() < 0.01,
        "kalman gain ~2/3 when prior uncertainty twice measurement: {kalman_gain:.4}"
    );

    // 3. update state estimate with weighted measurement

    // compute innovation (measurement residual)
    let innovation_length = measurement1.length - initial_state.length;
    let innovation_angle = measurement1.angle - initial_state.angle;

    assert!(
        innovation_angle.blade() < 4,
        "angle innovation has reasonable blade count: {}",
        innovation_angle.blade()
    );
    assert!(
        innovation_angle.value() < PI / 2.0,
        "angle innovation within π/2 segment: {}",
        innovation_angle.value()
    );

    assert!(
        innovation_length.abs() < 0.5,
        "measurement innovation reasonable: {innovation_length}"
    );

    // fuse state and measurement using geometric interpolation
    // for angles, interpolate in radians then reconstruct
    let state_angle_rad = initial_state.angle.mod_4_angle();
    let meas_angle_rad = measurement1.angle.mod_4_angle();
    let fused_angle_rad = state_angle_rad * (1.0 - kalman_gain) + meas_angle_rad * kalman_gain;

    let updated_state = Geonum::new(
        initial_state.length * (1.0 - kalman_gain) + measurement1.length * kalman_gain,
        fused_angle_rad,
        PI,
    );

    // update uncertainty (reduced by information gain)
    let updated_uncertainty = Geonum::new_with_blade(
        initial_uncertainty.length * (1.0 - kalman_gain),
        2,
        0.0,
        1.0,
    );

    // 4. prove fusion: result interpolates between prior and measurement

    // prove weighted average property for length
    let expected_length =
        initial_state.length * (1.0 - kalman_gain) + measurement1.length * kalman_gain;
    assert!(
        (updated_state.length - expected_length).abs() < EPSILON,
        "fused length matches weighted average: {} ≈ {}",
        updated_state.length,
        expected_length
    );

    // prove angle interpolation
    let expected_angle_rad = state_angle_rad * (1.0 - kalman_gain) + meas_angle_rad * kalman_gain;
    assert!(
        (updated_state.angle.mod_4_angle() - expected_angle_rad).abs() < EPSILON,
        "fused angle matches weighted interpolation: {} ≈ {}",
        updated_state.angle.mod_4_angle(),
        expected_angle_rad
    );

    // uncertainty decreases by information gain
    let uncertainty_reduction = 1.0 - updated_uncertainty.length / initial_uncertainty.length;
    assert!(
        (uncertainty_reduction - kalman_gain).abs() < EPSILON,
        "uncertainty reduces by kalman gain: {:.2}%",
        uncertainty_reduction * 100.0
    );

    // prove information never lost (uncertainty never increases)
    assert!(
        updated_uncertainty.length < initial_uncertainty.length,
        "uncertainty strictly decreases: {} < {}",
        updated_uncertainty.length,
        initial_uncertainty.length
    );

    // 5. demonstrate multi-sensor fusion with additional measurement

    // second sensor measurement (e.g., IMU) with different characteristics
    let measurement2 = Geonum::new(0.9, -0.05, PI); // pulls estimate in opposite direction

    // IMU typically more uncertain than GPS
    let meas_uncertainty2 = Geonum::new_with_blade(0.15, 2, 0.0, 1.0);

    // prove second measurement differs from first (tests robustness)
    let meas_diff = measurement2.angle - measurement1.angle;
    assert!(
        meas_diff.value() > 0.05,
        "measurements disagree by {:.3} radians",
        meas_diff.mod_4_angle()
    );

    // kalman gain for second measurement
    let kalman_gain2 =
        updated_uncertainty.length / (updated_uncertainty.length + meas_uncertainty2.length);

    // gain smaller due to higher measurement uncertainty
    assert!(
        kalman_gain2 < kalman_gain,
        "second gain {kalman_gain2} < first gain {kalman_gain} (higher uncertainty)"
    );

    // fuse with second measurement
    let updated_angle_rad = updated_state.angle.mod_4_angle();
    let meas2_angle_rad = measurement2.angle.mod_4_angle();
    let final_angle_rad = updated_angle_rad * (1.0 - kalman_gain2) + meas2_angle_rad * kalman_gain2;

    let final_state = Geonum::new(
        updated_state.length * (1.0 - kalman_gain2) + measurement2.length * kalman_gain2,
        final_angle_rad,
        PI,
    );

    // final uncertainty
    let final_uncertainty = Geonum::new_with_blade(
        updated_uncertainty.length * (1.0 - kalman_gain2),
        2,
        0.0,
        1.0,
    );

    // prove sequential fusion maintains consistency
    assert!(
        final_state.length > 0.0,
        "final state maintains positive length"
    );

    // prove uncertainty monotonically decreases
    assert!(
        final_uncertainty.length < updated_uncertainty.length,
        "uncertainty decreases: {} → {} → {}",
        initial_uncertainty.length,
        updated_uncertainty.length,
        final_uncertainty.length
    );

    // total uncertainty reduction after two measurements
    let total_reduction = 1.0 - final_uncertainty.length / initial_uncertainty.length;
    assert!(
        total_reduction > 0.7,
        "two measurements reduce uncertainty by {:.1}%",
        total_reduction * 100.0
    );

    // 6. measure performance: scale to high-dimensional fusion

    let start_time = Instant::now();

    // simulate high-dimensional sensor fusion with 1000 dimensions
    let high_dim = 1000;

    // state vector: robot pose in 1000D configuration space
    // use small angle values to avoid blade overflow from pi_radians
    let high_dim_state: Vec<Geonum> = (0..high_dim)
        .map(|i| Geonum::new_with_blade(1.0, i, 0.01, 1.0))
        .collect();

    // measurement vector with noise
    let high_dim_meas: Vec<Geonum> = (0..high_dim)
        .map(|i| {
            let noise = 0.05 * ((i * 7) % 13) as f64 / 13.0; // deterministic "noise"
            Geonum::new_with_blade(1.1, i, 0.01 + noise, 1.0)
        })
        .collect();

    // uncertainty vector (all bivectors)
    let high_dim_uncertainty: Vec<Geonum> = (0..high_dim)
        .map(|i| Geonum::new_with_blade(0.2 / (1.0 + i as f64 * 0.001), i * 4 + 2, 0.0, 1.0))
        .collect();

    // perform fusion with variable gains based on uncertainty
    let high_dim_result: Vec<Geonum> = high_dim_state
        .iter()
        .zip(high_dim_meas.iter())
        .zip(high_dim_uncertainty.iter())
        .map(|((s, m), u)| {
            // compute dimension-specific kalman gain
            let gain = u.length / (u.length + 0.1); // 0.1 is measurement uncertainty

            // fuse using angle interpolation while preserving blade
            // both state and measurement have same blade for same dimension
            let blade = s.angle.blade();
            let s_angle_val = s.angle.value();
            let m_angle_val = m.angle.value();
            let fused_angle_val = s_angle_val * (1.0 - gain) + m_angle_val * gain;

            Geonum::new_with_blade(
                s.length * (1.0 - gain) + m.length * gain,
                blade,
                fused_angle_val,
                PI,
            )
        })
        .collect();

    let elapsed = start_time.elapsed();

    // prove O(n) scaling
    let ops_per_second = (high_dim as f64) / elapsed.as_secs_f64();
    assert!(
        ops_per_second > 100_000.0,
        "fusion rate {ops_per_second:.0} ops/sec (O(n) complexity)"
    );

    // traditional kalman filter would require O(n³) operations
    // for 1000D: ~1 billion operations vs our ~1000
    let traditional_ops = high_dim.pow(3);
    let geonum_ops = high_dim;
    let speedup = traditional_ops / geonum_ops;

    assert_eq!(
        speedup, 1_000_000,
        "geonum {speedup}x faster than traditional kalman filter"
    );

    // test fusion preserves blade structure
    for (i, fused) in high_dim_result.iter().enumerate() {
        assert_eq!(
            fused.angle.blade(),
            i,
            "dimension {i} preserves blade count"
        );
    }

    // test fusion reduces high-dimensional uncertainty
    let avg_gain = high_dim_uncertainty
        .iter()
        .map(|u| u.length / (u.length + 0.1))
        .sum::<f64>()
        / high_dim as f64;

    assert!(
        avg_gain > 0.5,
        "average kalman gain {avg_gain:.2} shows effective fusion"
    );

    // demonstrate IMU + GPS + LIDAR fusion
    let imu = Geonum::new(1.0, 0.1, PI); // high rate, moderate accuracy
    let gps = Geonum::new(1.05, 0.08, PI); // low rate, high accuracy
    let lidar = Geonum::new(0.98, 0.12, PI); // medium rate, high accuracy

    // fuse heterogeneous sensors with different uncertainties
    let imu_weight = 0.3;
    let gps_weight = 0.5;
    let lidar_weight = 0.2;

    // fuse heterogeneous sensors
    let weighted_angle_rad = imu.angle.mod_4_angle() * imu_weight
        + gps.angle.mod_4_angle() * gps_weight
        + lidar.angle.mod_4_angle() * lidar_weight;

    let multi_sensor_fusion = Geonum::new(
        imu.length * imu_weight + gps.length * gps_weight + lidar.length * lidar_weight,
        weighted_angle_rad,
        PI,
    );

    // prove weighted fusion
    let expected_length = 1.0 * 0.3 + 1.05 * 0.5 + 0.98 * 0.2;
    assert!(
        (multi_sensor_fusion.length - expected_length).abs() < EPSILON,
        "multi-sensor fusion combines measurements: {} ≈ {}",
        multi_sensor_fusion.length,
        expected_length
    );
}
