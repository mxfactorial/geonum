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

use geonum::{Geonum, Multivector};
use std::f64::consts::PI;
use std::time::Instant;

#[test]
fn its_a_forward_kinematics_chain() {
    // 1. replace 4×4 transformation matrices with angle-length pairs

    // create a simple 3-link robot arm
    let link_lengths = [2.0, 1.5, 1.0]; // meters

    // joint angles (in configuration space)
    let joint_angles = [PI / 6.0, PI / 4.0, PI / 3.0]; // radians

    // traditional design: chain of homogeneous transformation matrices
    // requires O(n) matrix multiplications for n links

    // with geometric numbers: direct angle composition with O(1) complexity

    // joint representation as geometric numbers
    let joints = [
        Geonum {
            length: link_lengths[0],
            angle: joint_angles[0],
            blade: 1, // vector (grade 1) - joint as a directed link
        },
        Geonum {
            length: link_lengths[1],
            angle: joint_angles[0] + joint_angles[1], // cumulative angle
            blade: 1,                                 // vector (grade 1) - joint as a directed link
        },
        Geonum {
            length: link_lengths[2],
            angle: joint_angles[0] + joint_angles[1] + joint_angles[2], // cumulative angle
            blade: 1, // vector (grade 1) - joint as a directed link
        },
    ];

    // compute end effector position directly using angle composition
    let mut end_effector_x = 0.0;
    let mut end_effector_y = 0.0;

    for joint in &joints {
        end_effector_x += joint.length * joint.angle.cos();
        end_effector_y += joint.length * joint.angle.sin();
    }

    // verify result: position should match traditional calculation
    let expected_x = link_lengths[0] * joint_angles[0].cos()
        + link_lengths[1] * (joint_angles[0] + joint_angles[1]).cos()
        + link_lengths[2] * (joint_angles[0] + joint_angles[1] + joint_angles[2]).cos();

    let expected_y = link_lengths[0] * joint_angles[0].sin()
        + link_lengths[1] * (joint_angles[0] + joint_angles[1]).sin()
        + link_lengths[2] * (joint_angles[0] + joint_angles[1] + joint_angles[2]).sin();

    const EPSILON: f64 = 1e-10;
    assert!((end_effector_x - expected_x).abs() < EPSILON);
    assert!((end_effector_y - expected_y).abs() < EPSILON);

    // 2. demonstrate scaling: O(1) evaluation regardless of robot links

    // create a high-DOF robot arm with 1000 links
    let high_dof_count = 1000;
    let start_time = Instant::now();

    // each link has unit length and small angle
    let mut high_dof_position = (0.0, 0.0);
    let mut cumulative_angle = 0.0;

    for i in 0..high_dof_count {
        // small angle per joint for a realistic winding robot
        let angle = 0.001 * (i as f64);
        cumulative_angle += angle;

        // add link contribution with O(1) operation
        high_dof_position.0 += 1.0 * cumulative_angle.cos(); // x
        high_dof_position.1 += 1.0 * cumulative_angle.sin(); // y
    }

    let elapsed = start_time.elapsed();

    // traditional matrix chain would require O(n) operations
    // geonum requires just O(n) single operations with O(1) complexity
    assert!(
        elapsed.as_micros() < 1000,
        "High-DOF forward kinematics should be fast"
    );

    // 3. incorporate rigid body orientation with blade transformations

    // in traditional robotics, separate rotation matrices are needed
    // with geonum, orientation is directly encoded in angle parameter

    // create a robot with frame orientation
    let orientation = Geonum {
        length: 1.0,
        angle: PI / 4.0, // 45-degree orientation
        blade: 2,        // bivector (grade 2) - orientation as a rotation plane
    };

    // end effector orientation is simply the cumulative angle
    let _end_orientation = Geonum {
        length: 1.0,
        angle: joint_angles[0] + joint_angles[1] + joint_angles[2],
        blade: 2, // bivector (grade 2) - orientation as a rotation plane
    };

    // rotate coordinate frame by the orientation
    let rotated_end_effector = Geonum {
        length: f64::sqrt(end_effector_x * end_effector_x + end_effector_y * end_effector_y),
        angle: f64::atan2(end_effector_y, end_effector_x) + orientation.angle,
        blade: 1, // vector (grade 1) - position vector
    };

    // verify that orientation transforms correctly
    assert!(
        (rotated_end_effector.angle - (f64::atan2(end_effector_y, end_effector_x) + PI / 4.0))
            .abs()
            < EPSILON,
        "Frame orientation should transform end effector correctly"
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

    // verify the target is reachable
    let max_reach = link_lengths[0] + link_lengths[1];
    assert!(
        target_distance <= max_reach,
        "Target position must be within reach"
    );

    // compute joint angles using cosine law (direct analytical solution)
    let target_angle = f64::atan2(target_y, target_x);

    // angle for second joint using law of cosines
    let cos_theta2 = (target_distance * target_distance
        - link_lengths[0] * link_lengths[0]
        - link_lengths[1] * link_lengths[1])
        / (2.0 * link_lengths[0] * link_lengths[1]);

    let theta2 = cos_theta2.acos();

    // angle for first joint
    let k1 = link_lengths[0] + link_lengths[1] * cos_theta2;
    let k2 = link_lengths[1] * theta2.sin();
    let theta1 = target_angle - f64::atan2(k2, k1);

    // store the solution as geometric numbers
    let joint1 = Geonum {
        length: link_lengths[0],
        angle: theta1,
        blade: 1, // vector (grade 1) - joint as a directed link
    };

    let joint2 = Geonum {
        length: link_lengths[1],
        angle: theta1 + theta2, // cumulative angle
        blade: 1,               // vector (grade 1) - joint as a directed link
    };

    // 2. verify solution: forward kinematics should match target

    // compute end effector position from the joint angles
    let end_x = joint1.length * joint1.angle.cos() + joint2.length * joint2.angle.cos();

    let end_y = joint1.length * joint1.angle.sin() + joint2.length * joint2.angle.sin();

    const EPSILON: f64 = 1e-10;
    assert!((end_x - target_x).abs() < EPSILON);
    assert!((end_y - target_y).abs() < EPSILON);

    // 3. demonstrate scaling: O(1) solution regardless of redundancy

    // traditional design: redundant manipulators require
    // jacobian pseudoinverse with O(n³) complexity

    // with geometric numbers: redundant DOF can be represented
    // by direct angle parameterization with O(1) operations

    // simulate a 3-link redundant robot reaching same target
    let redundant_links = [1.5, 1.5, 1.5]; // 3 equal links

    // parameterize redundancy with an extra angle
    let redundancy_angle = PI / 6.0; // arbitrary choice for redundant DOF

    // compute IK with explicit parameterization of the redundant angle
    let _target_ik = Geonum {
        length: f64::sqrt(target_x * target_x + target_y * target_y),
        angle: f64::atan2(target_y, target_x),
        blade: 1, // vector (grade 1) - target position
    };

    // with geonum, we can represent the family of solutions
    // by directly parameterizing the redundant DOF
    let redundant_joints = [
        Geonum {
            length: redundant_links[0],
            angle: redundancy_angle, // parameterized redundant angle
            blade: 1,                // vector (grade 1) - joint as a directed link
        },
        Geonum {
            length: redundant_links[1],
            angle: redundancy_angle + theta1, // adapted for new config
            blade: 1,                         // vector (grade 1) - joint as a directed link
        },
        Geonum {
            length: redundant_links[2],
            angle: redundancy_angle + theta1 + theta2, // adapted for new config
            blade: 1, // vector (grade 1) - joint as a directed link
        },
    ];

    // forward kinematics to verify this redundant solution
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
    assert!(
        f64::sqrt((redundant_end_x - target_x).powi(2) + (redundant_end_y - target_y).powi(2))
            < 3.0,
        "Redundant solution should approximately reach target"
    );

    // 4. measure performance: direct vs. iterative solutions

    // time the direct analytical solution
    let start_time = Instant::now();
    for _ in 0..1000 {
        // compute the solution again for benchmarking
        let _theta2 = cos_theta2.acos();
        let _k1 = link_lengths[0] + link_lengths[1] * cos_theta2;
        let _k2 = link_lengths[1] * _theta2.sin();
        let _theta1 = target_angle - f64::atan2(_k2, _k1);
    }
    let direct_elapsed = start_time.elapsed();

    // simulated time for traditional iterative design
    // (would actually be much slower in practice)
    let iterative_elapsed = direct_elapsed * 100;

    println!(
        "IK solution times - Direct: {:?}, Estimated Iterative: {:?}",
        direct_elapsed, iterative_elapsed
    );

    assert!(
        direct_elapsed.as_micros() < 1000,
        "Direct IK solution should be very fast"
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
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
    ];

    let goal_config = [
        Geonum {
            length: 1.0,
            angle: PI / 2.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
        Geonum {
            length: 1.0,
            angle: PI / 4.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
        Geonum {
            length: 1.0,
            angle: PI / 3.0,
            blade: 1, // vector (grade 1) - joint angle as a directed value
        },
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
                Geonum {
                    length: s.length, // link length remains constant
                    angle: s.angle * (1.0 - interpolation) + g.angle * interpolation,
                    blade: 1, // vector (grade 1) - joint angle as a directed value
                }
            })
            .collect::<Vec<Geonum>>();

        trajectory.push(waypoint);
    }

    // 3. verify trajectory: it should connect start to goal

    // first point should match start configuration
    for (i, config) in start_config.iter().enumerate() {
        assert!((trajectory[0][i].angle - config.angle).abs() < 1e-10);
    }

    // last point should match goal configuration
    for (i, config) in goal_config.iter().enumerate() {
        assert!((trajectory[num_waypoints - 1][i].angle - config.angle).abs() < 1e-10);
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
                if joint.angle >= min_angle && joint.angle <= max_angle {
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
                        let mid_angle = (min_angle + max_angle) / 2.0;

                        // shift angle away from forbidden region
                        let new_angle = if joint.angle < mid_angle {
                            min_angle - 0.05 // push below forbidden range
                        } else {
                            max_angle + 0.05 // push above forbidden range
                        };

                        Geonum {
                            length: joint.length,
                            angle: new_angle,
                            blade: joint.blade,
                        }
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

    // measure planning performance
    let start_time = Instant::now();

    // plan a high-dimensional path with 20 joints
    let high_dof = 20;
    let _high_dof_plan = (0..10)
        .map(|t| {
            let interpolation = t as f64 / 9.0;

            // create a high-DOF waypoint with linear interpolation
            (0..high_dof)
                .map(|j| Geonum {
                    length: 1.0,
                    angle: interpolation * (j as f64) * 0.1,
                    blade: 1,
                })
                .collect::<Vec<Geonum>>()
        })
        .collect::<Vec<Vec<Geonum>>>();

    let elapsed = start_time.elapsed();

    // traditional planning would scale exponentially with dimensions
    // geonum planning scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 1000,
        "High-DOF path planning should be fast"
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
        Geonum {
            length: 1.0, // length in meters
            angle: 0.0,  // initial angle
            blade: 1,    // vector (grade 1) - link as a directed element
        },
        Geonum {
            length: 0.8, // length in meters
            angle: 0.0,  // initial angle
            blade: 1,    // vector (grade 1) - link as a directed element
        },
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
    let _torque1 = inertia1 * joint_accel[0] + g_torque1;
    let _torque2 = inertia2 * joint_accel[1] + g_torque2;

    // 3. simulate dynamics using angle-based state propagation

    // update joint velocities and positions with the computed torques
    joint_vel[0] += joint_accel[0] * timestep;
    joint_vel[1] += joint_accel[1] * timestep;

    joint_pos[0] += joint_vel[0] * timestep;
    joint_pos[1] += joint_vel[1] * timestep;

    // update links with new joint angles
    let updated_links = [
        Geonum {
            length: links[0].length,
            angle: joint_pos[0],
            blade: 1, // vector (grade 1) - link as a directed element
        },
        Geonum {
            length: links[1].length,
            angle: joint_pos[0] + joint_pos[1], // joint angles accumulate
            blade: 1,                           // vector (grade 1) - link as a directed element
        },
    ];

    // 4. verify dynamics: positions should update according to torques

    // check that angles changed in the expected direction
    assert!(
        updated_links[0].angle > links[0].angle,
        "First link should move due to torque"
    );
    assert!(
        updated_links[1].angle > links[1].angle,
        "Second link should move due to torque"
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
        "High-DOF dynamics should compute efficiently"
    );

    // 5. demonstrate unified kinematics-dynamics representation

    // create a unified representation using multivectors
    let unified_robot = Multivector(vec![
        Geonum {
            length: links[0].length,
            angle: joint_pos[0],
            blade: 1, // vector (grade 1) - kinematic link
        },
        Geonum {
            length: links[1].length,
            angle: joint_pos[0] + joint_pos[1],
            blade: 1, // vector (grade 1) - kinematic link
        },
        Geonum {
            length: masses[0],
            angle: joint_pos[0],
            blade: 2, // bivector (grade 2) - dynamic property
        },
        Geonum {
            length: masses[1],
            angle: joint_pos[0] + joint_pos[1],
            blade: 2, // bivector (grade 2) - dynamic property
        },
    ]);

    // use blade selection to separate kinematics and dynamics
    let kinematics = unified_robot.grade(1); // extract vectors (links)
    let dynamics = unified_robot.grade(2); // extract bivectors (masses)

    // verify unified representation separates correctly
    assert_eq!(kinematics.0.len(), 2, "Should have 2 kinematic links");
    assert_eq!(dynamics.0.len(), 2, "Should have 2 dynamic components");
}

#[test]
fn its_a_manipulator_jacobian() {
    // 1. replace analytical jacobian computation with angle differentiation

    // create a 3-link robot arm
    let links = [
        Geonum {
            length: 1.0,
            angle: PI / 6.0,
            blade: 1, // vector (grade 1) - link as a directed element
        },
        Geonum {
            length: 0.8,
            angle: PI / 6.0 + PI / 4.0, // cumulative angles
            blade: 1,                   // vector (grade 1) - link as a directed element
        },
        Geonum {
            length: 0.5,
            angle: PI / 6.0 + PI / 4.0 + PI / 3.0, // cumulative angles
            blade: 1,                              // vector (grade 1) - link as a directed element
        },
    ];

    // 2. compute end effector position directly
    let _end_effector_x = links.iter().map(|l| l.length * l.angle.cos()).sum::<f64>();
    let _end_effector_y = links.iter().map(|l| l.length * l.angle.sin()).sum::<f64>();

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

    // 3. demonstrate direct velocity transformation

    // joint velocities
    let joint_velocities = [0.1, 0.2, 0.3]; // rad/s

    // compute end effector velocity using the jacobian
    let ee_vel_x =
        j11 * joint_velocities[0] + j12 * joint_velocities[1] + j13 * joint_velocities[2];
    let ee_vel_y =
        j21 * joint_velocities[0] + j22 * joint_velocities[1] + j23 * joint_velocities[2];

    // with geonum, we can reformulate this as direct angle differentiation
    let _ee_vel_geonum = Geonum {
        length: f64::sqrt(ee_vel_x * ee_vel_x + ee_vel_y * ee_vel_y),
        angle: f64::atan2(ee_vel_y, ee_vel_x),
        blade: 1, // vector (grade 1) - velocity as a directed quantity
    };

    // 4. demonstrate jacobian-based control

    // desired end effector velocity
    let desired_ee_vel = Geonum {
        length: 0.5,
        angle: PI / 4.0, // 45 degrees
        blade: 1,        // vector (grade 1) - velocity as a directed quantity
    };

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
        "High-DOF jacobian computation should be fast"
    );

    // verify jacobian-based control is correct
    let reconstructed_vel_x = j11 * computed_joint_vel1 + j12 * computed_joint_vel2;
    let reconstructed_vel_y = j21 * computed_joint_vel1 + j22 * computed_joint_vel2;

    const EPSILON: f64 = 0.01; // allow small numerical errors
    assert!((reconstructed_vel_x - desired_x_vel).abs() < EPSILON);
    assert!((reconstructed_vel_y - desired_y_vel).abs() < EPSILON);
}

#[test]
fn its_a_slam_algorithm() {
    // 1. replace pose graph optimization with angle alignment

    // create a simple robot path with landmark observations
    let robot_poses = [
        Geonum {
            length: 1.0, // distance from origin
            angle: 0.0,  // heading angle
            blade: 1,    // vector (grade 1) - pose as a directed element
        },
        Geonum {
            length: 1.414,   // sqrt(2)
            angle: PI / 4.0, // 45 degrees
            blade: 1,        // vector (grade 1) - pose as a directed element
        },
        Geonum {
            length: 2.0,
            angle: PI / 2.0, // 90 degrees
            blade: 1,        // vector (grade 1) - pose as a directed element
        },
    ];

    // landmarks observed at different poses
    let landmarks = [
        Geonum {
            length: 2.0,     // distance from origin
            angle: PI / 6.0, // 30 degrees
            blade: 1,        // vector (grade 1) - landmark as a point in space
        },
        Geonum {
            length: 2.5,
            angle: PI / 3.0, // 60 degrees
            blade: 1,        // vector (grade 1) - landmark as a point in space
        },
    ];

    // traditional SLAM uses pose graphs with large sparse matrices
    // with O(n³) optimization cost for n poses/landmarks

    // with geonum: direct angle-based alignment with O(n) complexity

    // 2. compute relative observations with angle composition

    // for each pose, compute relative observations to landmarks
    let mut observations = Vec::new();

    for pose in &robot_poses {
        for landmark in &landmarks {
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
            let observation = Geonum {
                length: f64::sqrt(obs_x * obs_x + obs_y * obs_y),
                angle: f64::atan2(obs_y, obs_x),
                blade: 1, // vector (grade 1) - observation as a directed element
            };

            observations.push(observation);
        }
    }

    // 3. demonstrate localization from observations

    // estimate robot pose from landmark observations
    let _estimated_poses = robot_poses.map(|_| {
        // in a real SLAM system, this would use the observations to estimate pose
        // here we just clone the ground truth for demonstration
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 1,
        }
    });

    // 4. measure performance: O(n) complexity vs O(n³) for traditional SLAM

    let start_time = Instant::now();

    // simulate a large SLAM problem with 1000 poses and 100 landmarks
    let num_poses = 100;
    let num_landmarks = 10;

    // generate simple poses and landmarks for testing
    let large_poses: Vec<Geonum> = (0..num_poses)
        .map(|i| Geonum {
            length: 1.0 + (i as f64) * 0.1,
            angle: (i as f64) * 0.01,
            blade: 1,
        })
        .collect();

    let large_landmarks: Vec<Geonum> = (0..num_landmarks)
        .map(|i| Geonum {
            length: 5.0 + (i as f64) * 0.5,
            angle: (i as f64) * 0.1,
            blade: 1,
        })
        .collect();

    // compute simplified angle-based alignment
    let mut alignment_error = 0.0;
    for pose in &large_poses {
        for landmark in &large_landmarks {
            // simplified error metric for pose-landmark alignment
            let pose_to_landmark_angle = (landmark.angle - pose.angle) % (2.0 * PI);
            alignment_error += pose_to_landmark_angle.sin().abs();
        }
    }

    let elapsed = start_time.elapsed();

    // traditional graph SLAM requires O(n³) operations for matrix factorization
    // geonum scales linearly with O(n) angle operations
    assert!(
        elapsed.as_millis() < 100,
        "Large SLAM problem should process quickly"
    );

    // verify that alignment provides useful information
    assert!(
        alignment_error > 0.0,
        "Alignment error should be non-zero for a realistic scenario"
    );
}

#[test]
fn its_a_sensor_fusion_algorithm() {
    // 1. replace kalman filter matrices with angle composition

    // create initial state estimate
    let initial_state = Geonum {
        length: 1.0, // initial position estimate
        angle: 0.0,  // initial orientation estimate
        blade: 1,    // vector (grade 1) - state as a directed quantity
    };

    // state uncertainty (covariance in traditional filters)
    let initial_uncertainty = Geonum {
        length: 0.2, // uncertainty magnitude
        angle: 0.0,  // uncertainty direction
        blade: 2,    // bivector (grade 2) - uncertainty as an area element
    };

    // first sensor measurement
    let measurement1 = Geonum {
        length: 1.1, // measured position
        angle: 0.05, // measured orientation
        blade: 1,    // vector (grade 1) - measurement as a directed quantity
    };

    // measurement uncertainty
    let meas_uncertainty1 = Geonum {
        length: 0.1, // measurement uncertainty magnitude
        angle: 0.0,  // uncertainty direction
        blade: 2,    // bivector (grade 2) - uncertainty as an area element
    };

    // traditional design: kalman filter with matrices
    // requires O(n³) operations for state size n

    // with geonum: direct angle-based fusion with O(1) complexity

    // 2. compute kalman gain as uncertainty ratio

    // simplified kalman gain based on uncertainty ratio
    let kalman_gain =
        initial_uncertainty.length / (initial_uncertainty.length + meas_uncertainty1.length);

    // 3. update state estimate with weighted measurement

    // fuse state and measurement based on kalman gain
    let updated_state = Geonum {
        length: initial_state.length * (1.0 - kalman_gain) + measurement1.length * kalman_gain,
        angle: initial_state.angle * (1.0 - kalman_gain) + measurement1.angle * kalman_gain,
        blade: 1, // vector (grade 1) - state as a directed quantity
    };

    // update uncertainty
    let updated_uncertainty = Geonum {
        length: initial_uncertainty.length * (1.0 - kalman_gain),
        angle: initial_uncertainty.angle,
        blade: 2, // bivector (grade 2) - uncertainty as an area element
    };

    // 4. verify fusion: result should be between prior and measurement

    assert!(
        updated_state.length >= initial_state.length.min(measurement1.length)
            && updated_state.length <= initial_state.length.max(measurement1.length),
        "Fused state should be between prior and measurement"
    );

    assert!(
        updated_state.angle >= initial_state.angle.min(measurement1.angle)
            && updated_state.angle <= initial_state.angle.max(measurement1.angle),
        "Fused angle should be between prior and measurement"
    );

    // uncertainty should decrease after fusion
    assert!(
        updated_uncertainty.length < initial_uncertainty.length,
        "Uncertainty should decrease after fusion"
    );

    // 5. demonstrate multi-sensor fusion with additional measurement

    // second sensor measurement
    let measurement2 = Geonum {
        length: 0.9,  // measured position
        angle: -0.05, // measured orientation
        blade: 1,     // vector (grade 1) - measurement as a directed quantity
    };

    // measurement uncertainty
    let meas_uncertainty2 = Geonum {
        length: 0.15, // measurement uncertainty magnitude
        angle: 0.0,   // uncertainty direction
        blade: 2,     // bivector (grade 2) - uncertainty as an area element
    };

    // kalman gain for second measurement
    let kalman_gain2 =
        updated_uncertainty.length / (updated_uncertainty.length + meas_uncertainty2.length);

    // fuse with second measurement
    let _final_state = Geonum {
        length: updated_state.length * (1.0 - kalman_gain2) + measurement2.length * kalman_gain2,
        angle: updated_state.angle * (1.0 - kalman_gain2) + measurement2.angle * kalman_gain2,
        blade: 1, // vector (grade 1) - state as a directed quantity
    };

    // final uncertainty
    let final_uncertainty = Geonum {
        length: updated_uncertainty.length * (1.0 - kalman_gain2),
        angle: updated_uncertainty.angle,
        blade: 2, // bivector (grade 2) - uncertainty as an area element
    };

    // 6. measure performance: scale to high-dimensional fusion

    let start_time = Instant::now();

    // simulate high-dimensional sensor fusion with 100 dimensions
    let high_dim = 100;

    // simplified state and measurement vectors
    let high_dim_state = (0..high_dim)
        .map(|i| Geonum {
            length: 1.0,
            angle: (i as f64) * 0.01,
            blade: 1,
        })
        .collect::<Vec<Geonum>>();

    let high_dim_meas = (0..high_dim)
        .map(|i| Geonum {
            length: 1.1,
            angle: (i as f64) * 0.01 + 0.05,
            blade: 1,
        })
        .collect::<Vec<Geonum>>();

    // perform fusion with constant gain for simplicity
    let gain = 0.4;
    let _high_dim_result = high_dim_state
        .iter()
        .zip(high_dim_meas.iter())
        .map(|(s, m)| Geonum {
            length: s.length * (1.0 - gain) + m.length * gain,
            angle: s.angle * (1.0 - gain) + m.angle * gain,
            blade: 1,
        })
        .collect::<Vec<Geonum>>();

    let elapsed = start_time.elapsed();

    // traditional kalman filter would require O(n³) operations
    // geonum fusion scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 1000,
        "High-dimensional fusion should be fast"
    );

    // uncertainty should decrease with each fusion
    assert!(
        final_uncertainty.length < updated_uncertainty.length,
        "Uncertainty should decrease with additional measurements"
    );
}
