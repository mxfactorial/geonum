// robotics without matrices
//
// the working objects of robotics — homogeneous transforms, jacobians, wind-up
// counters, mass matrices — are coordinate scaffolding around geometric facts
// that compute directly in [magnitude, angle]:
//
//   forward kinematics   cumulative angle addition, one op per joint
//   inverse kinematics   triangle closure, the law-of-cosines angle built from
//                        its own cosine — no acos round-trip
//   manipulability       det(J) = l1·l2·sin θ2 is one wedge magnitude
//   spatial wrists       out-of-plane rotations compose by the half-angle
//                        product (quaternion_test)
//   cable wind-up        the blade counts the full turns a rotation matrix
//                        forgets (R(2π) = I)
//   gravity load         the wedge against a vertical force extracts the
//                        horizontal moment arm
//   redundancy           null-space self-motion is velocity contributions
//                        cancelling at opposite angles
//   clearance            point-to-path distance is one rejection
//   feedback             a proportional step scales the error magnitude exactly
//   planning scale       a million-dimensional c-space is still two numbers
//
// assertions land on values the tests didnt construct: pythagorean triples,
// polygon closure, the textbook det(J), the moment-arm sum, hand geometry
//
// run: cargo test --test robotics_test

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// ---------------------------------------------------------------------------
// forward kinematics: the pose is a running angle sum
// ---------------------------------------------------------------------------
#[test]
fn its_a_kinematic_chain() {
    // each joint adds its angle to the chain's heading; each link extends at
    // the running total. the 4×4 homogeneous transform stack reduces to one
    // angle addition and one vector addition per joint

    // two links at a right angle land on the 3-4-5 triangle
    let shoulder = Angle::new(0.0, 1.0);
    let elbow = Angle::new(1.0, 2.0); // π/2 bend
    let upper = Geonum::new_with_angle(3.0, shoulder);
    let fore = Geonum::new_with_angle(4.0, shoulder + elbow);
    let tip = upper + fore;

    assert!(tip.near_mag(5.0), "3-4-5: the reach is the hypotenuse");
    assert_eq!(tip.angle.grade(), 0, "first-quadrant heading");
    assert!(
        (tip.angle.t() - 0.5).abs() < EPSILON,
        "the 3-4-5 heading is the projection ratio 4/(5+3) = 1/2 exactly"
    );

    // six equal links turning π/3 each close a hexagon: the tip returns to the
    // base because the headings sweep one full turn — polygon closure, a truth
    // the chain doesnt construct
    let mut heading = Angle::new(0.0, 1.0);
    let step = Angle::new(1.0, 3.0); // π/3 per joint
    let mut walk = Geonum::scalar(0.0);
    for _ in 0..6 {
        walk = walk + Geonum::new_with_angle(1.0, heading);
        heading = heading + step;
    }
    assert!(walk.mag < 1e-9, "the hexagon closes: Σ links = 0");
}

// ---------------------------------------------------------------------------
// inverse kinematics: triangle closure, no acos round-trip
// ---------------------------------------------------------------------------
#[test]
fn its_an_inverse_kinematics_solution() {
    // 2-link IK is the law of cosines. the traditional solver computes
    // cos(elbow) then calls acos to get radians back — a scalar round-trip.
    // geonum builds the elbow angle FROM its cosine and sine
    // (new_from_cartesian), and the proof is closure: forward kinematics
    // through the solved joints lands on the target

    let (a, b) = (3.0, 2.0); // link lengths
    let target = Geonum::new(4.0, 1.0, 3.0); // [4, π/3]
    let c = target.mag;

    // reachability: the triangle inequality is the workspace annulus
    assert!(
        c <= a + b && c >= (a - b).abs(),
        "target inside the annulus"
    );

    // interior elbow angle from the law of cosines — built as an angle from
    // (cos, sin), never passing through acos
    let cos_int = (a * a + b * b - c * c) / (2.0 * a * b);
    let sin_int = (1.0 - cos_int * cos_int).sqrt();
    let interior = Angle::new_from_cartesian(cos_int, sin_int);

    // shoulder offset from the law of sines, same construction
    let sin_beta = b * sin_int / c;
    let cos_beta = (1.0 - sin_beta * sin_beta).sqrt();
    let beta = Angle::new_from_cartesian(cos_beta, sin_beta);

    // elbow-up: shoulder aims below the target line, elbow bends π − interior
    let shoulder_up = target.angle - beta;
    let elbow_up = Angle::new(1.0, 1.0) - interior;
    let tip_up =
        Geonum::new_with_angle(a, shoulder_up) + Geonum::new_with_angle(b, shoulder_up + elbow_up);
    assert!(
        tip_up.distance_to(&target).mag < 1e-9,
        "elbow-up closes on the target"
    );

    // elbow-down: the mirror branch — shoulder above the target line, elbow
    // bending the other way, which in forward-only rotation is π + interior
    let shoulder_down = target.angle + beta;
    let elbow_down = Angle::new(1.0, 1.0) + interior;
    let tip_down = Geonum::new_with_angle(a, shoulder_down)
        + Geonum::new_with_angle(b, shoulder_down + elbow_down);
    assert!(
        tip_down.distance_to(&target).mag < 1e-9,
        "elbow-down closes on the target"
    );

    // two solutions, one triangle: the branches are the elbow reflected
    // across the target line
    assert_ne!(
        shoulder_up, shoulder_down,
        "the redundant branch is a distinct configuration"
    );
}

// ---------------------------------------------------------------------------
// manipulability: det(J) is one wedge magnitude
// ---------------------------------------------------------------------------
#[test]
fn it_reads_the_jacobian_determinant_off_the_wedge() {
    // the 2-link jacobian determinant every robotics text derives —
    // det(J) = l1·l2·sin θ2 — is the wedge magnitude of the two link vectors.
    // no 2×2 matrix, no determinant expansion: the manipulability measure is
    // the area the links span
    let (l1, l2) = (3.0, 2.0);
    let shoulder = Angle::new(1.0, 5.0); // π/5 — arbitrary, det(J) ignores it
    let link1 = Geonum::new_with_angle(l1, shoulder);

    for (p, d) in [(1.0, 6.0), (1.0, 4.0), (1.0, 2.0), (5.0, 6.0)] {
        let elbow = Angle::new(p, d);
        let link2 = Geonum::new_with_angle(l2, shoulder + elbow);
        let det_j = l1 * l2 * (p * PI / d).sin(); // the textbook formula
        assert!(
            (link1.wedge(&link2).mag - det_j).abs() < EPSILON,
            "det(J) = l1·l2·sin θ2 read off the wedge at θ2 = {p}π/{d}"
        );
    }

    // the wedge vanishes exactly where the textbook says the arm is singular:
    // full extension and full fold — the workspace boundary
    let extended = Geonum::new_with_angle(l2, shoulder + Angle::new(0.0, 1.0));
    let folded = Geonum::new_with_angle(l2, shoulder + Angle::new(1.0, 1.0));
    assert!(
        link1.wedge(&extended).mag < EPSILON,
        "extension is singular"
    );
    assert!(link1.wedge(&folded).mag < EPSILON, "fold is singular");
    assert!((link1 + extended).near_mag(l1 + l2), "boundary reach: 5");
    assert!((link1 + folded).near_mag(l1 - l2), "boundary reach: 1");

    // gimbal lock is the same vanished wedge one level up: when two wrist AXES
    // align, the plane they spanned is gone and their rotations compose
    // order-blind — same-axis composition is the commutative multiply
    // (quaternion_test::it_twists_the_arm_holding_the_book). the lost DOF IS
    // the anti-symmetric term
    let axis_roll = Geonum::new(1.0, 0.0, 1.0);
    let axis_pitch = Geonum::new(1.0, 1.0, 2.0);
    let axis_roll_again = Geonum::new(1.0, 0.0, 1.0);
    assert!(
        axis_roll.wedge(&axis_pitch).mag > 0.99,
        "perpendicular wrist axes span a full plane"
    );
    assert!(
        axis_roll.wedge(&axis_roll_again).mag < EPSILON,
        "aligned axes: the wedge — and the DOF — is gone"
    );
}

// ---------------------------------------------------------------------------
// spatial wrist: out-of-plane composition by the half-angle product
// ---------------------------------------------------------------------------
#[test]
fn it_composes_a_spatial_wrist() {
    // planar chains compose in one plane, where angles just add. a spatial
    // wrist pitches about one axis and yaws about another, and the composite
    // obeys the half-angle product
    // cos(c/2) = cos(a/2)cos(b/2) − sin(a/2)sin(b/2)cos(Θ), Θ between the axes.
    // quaternion_test::it_rotates_the_book_about_two_axes proves the
    // perpendicular case and locates all order-dependence in the wedge; here
    // the same law prices a real wrist, orthogonal or not

    let half = Angle::new(1.0, 4.0); // quarter-turn commands, θ/2 = π/4
    let sym = Geonum::cos(half) * Geonum::cos(half);

    // orthogonal wrist (Θ = π/2): the correction projection vanishes and two
    // quarter turns compose to 2π/3 — the cube-diagonal fact
    let pitch_axis = Geonum::new(1.0, 0.0, 1.0);
    let yaw_axis = Geonum::new(1.0, 1.0, 2.0);
    let skew = Geonum::sin(half) * Geonum::sin(half) * pitch_axis.dot(&yaw_axis);
    let composite = sym + skew;
    assert!(
        composite.near(&Geonum::cos(Angle::new(1.0, 3.0))),
        "orthogonal wrist: cos(c/2) = 1/2, the composite turn is 2π/3"
    );

    // oblique wrist (Θ = π/3, a non-orthogonal gimbal): the correction term
    // survives with the axes' projection inside it — sin·sin lands blade 2,
    // so geonum addition applies the formula's minus as a position
    let oblique_axis = Geonum::new(1.0, 1.0, 3.0); // π/3 from the pitch axis
    let skew_ob = Geonum::sin(half) * Geonum::sin(half) * pitch_axis.dot(&oblique_axis);
    assert_eq!(skew_ob.angle.grade(), 2, "the − sign is the blade sum");
    let composite_ob = sym + skew_ob;
    let reference = (PI / 4.0).cos().powi(2) - (PI / 4.0).sin().powi(2) * (PI / 3.0).cos();
    assert!(
        (composite_ob.mag - reference).abs() < EPSILON,
        "oblique wrist: cos(c/2) = cos²(π/4) − sin²(π/4)cos(π/3) = 1/4"
    );
    assert_eq!(composite_ob.angle.grade(), 0, "positive cosine: grade 0");
}

// ---------------------------------------------------------------------------
// cable wind-up: the blade is the turn counter SO(3) forgets
// ---------------------------------------------------------------------------
#[test]
fn it_counts_cable_windup_in_the_blade() {
    // a continuously rotating wrist returns to the same orientation every full
    // turn — R(2π) = I, so a rotation matrix cannot represent how twisted the
    // cable harness is, and real controllers bolt a wind-up counter beside the
    // SO(3) state. the angle IS that counter: each full turn arrives four
    // blades on with grade and position unchanged
    // (quaternion_test::it_twists_the_arm_holding_the_book)

    let wrist = Geonum::new(1.0, 1.0, 5.0); // tool heading π/5
    let full_turn = Angle::new(4.0, 2.0); // 2π

    let wound = wrist.rotate(full_turn).rotate(full_turn).rotate(full_turn);
    assert_eq!(
        wound.base_angle(),
        wrist.base_angle(),
        "the tool points the same way after three turns"
    );
    let turns = (wound.angle - wrist.angle).blade() / 4;
    assert_eq!(
        turns, 3,
        "the cable holds three twists — read off the blade"
    );

    // unwinding is angle subtraction: one turn back, two twists left
    let unwound = Geonum::new_with_angle(wound.mag, wound.angle - full_turn);
    assert_eq!((unwound.angle - wrist.angle).blade() / 4, 2);
    assert_eq!(
        unwound.base_angle(),
        wrist.base_angle(),
        "orientation never changed while the count did"
    );

    // a cable budget is a blade budget: the controller compares counts, no
    // separate wind-up state to maintain
    let budget_turns = 2;
    assert!(
        turns > budget_turns,
        "three twists exceed a two-turn budget"
    );
    assert!(
        (unwound.angle - wrist.angle).blade() / 4 <= budget_turns,
        "after unwinding one turn the budget clears"
    );

    // base_angle() drops the count on purpose — the escape hatch for a joint
    // with nothing tethered to it. an untethered joint resets; a cabled one
    // keeps its blade
    assert_eq!(wound.base_angle().angle.blade(), wrist.angle.blade() % 4);
}

// ---------------------------------------------------------------------------
// gravity compensation: the wedge extracts the horizontal moment arm
// ---------------------------------------------------------------------------
#[test]
fn it_compensates_gravity_with_the_wedge() {
    // holding a pose against gravity needs the torque at each joint: the
    // moment of each mass about the pivot. traditionally Jᵀ·F; here τ = r ∧ F,
    // and wedging with a VERTICAL force extracts the HORIZONTAL moment arm on
    // its own — |r ∧ F| = |r|·F·|sin(θ_F − θ_r)| = F·(r·cos θ_r) = F·x

    let (l1, m1) = (0.5, 1.0);
    let (l2, m2) = (0.4, 0.8);
    let g = 9.81;
    let shoulder = Angle::new(1.0, 6.0); // π/6
    let elbow = Angle::new(1.0, 4.0); // π/4 relative

    // centers of mass
    let com1 = Geonum::new_with_angle(l1 / 2.0, shoulder);
    let joint2 = Geonum::new_with_angle(l1, shoulder);
    let com2 = joint2 + Geonum::new_with_angle(l2 / 2.0, shoulder + elbow);

    // weights point straight down
    let w1 = Geonum::new(m1 * g, 3.0, 2.0);
    let w2 = Geonum::new(m2 * g, 3.0, 2.0);

    // shoulder torque: both masses lever about the base. both hang on the same
    // side — same rotational sense — so the moment magnitudes sum
    let tau1 = com1.wedge(&w1);
    let tau2 = com2.wedge(&w2);
    assert_eq!(tau1.angle.grade(), 2, "torque is a bivector");

    let x1 = (l1 / 2.0) * (PI / 6.0).cos();
    let x2 = l1 * (PI / 6.0).cos() + (l2 / 2.0) * (PI / 6.0 + PI / 4.0).cos();
    let shoulder_ref = m1 * g * x1 + m2 * g * x2; // Σ mᵢ·g·xᵢ, the moment sum
    assert!(
        (tau1.mag + tau2.mag - shoulder_ref).abs() < EPSILON,
        "the shoulder holds Σ mᵢgxᵢ — the wedge found every moment arm"
    );

    // elbow torque: only the forearm mass levers about joint 2
    let tau_elbow = (com2 - joint2).wedge(&w2);
    let elbow_ref = m2 * g * (l2 / 2.0) * (PI / 6.0 + PI / 4.0).cos();
    assert!(
        (tau_elbow.mag - elbow_ref).abs() < EPSILON,
        "the elbow holds m2·g·(l2/2)·cos(θ1+θ2)"
    );

    // the moment arm is the adjacent projection the wedge implies:
    // |r ∧ w| = |w| · r.adj().mag
    assert!(
        (tau1.mag - w1.mag * com1.adj().mag).abs() < EPSILON,
        "the wedge against vertical is weight × horizontal arm"
    );
}

// ---------------------------------------------------------------------------
// redundancy: null-space self-motion, velocities cancelling by angle
// ---------------------------------------------------------------------------
#[test]
fn it_moves_joints_while_the_tip_stands_still() {
    // a redundant arm moves its joints with the tip pinned — the null space
    // that traditionally costs an SVD. each joint's velocity contribution is
    // its lever to the tip rotated a quarter turn (circular motion's velocity,
    // the differentiate rotation) scaled by the joint rate; self-motion is
    // those contributions cancelling at opposite angles

    // straight 3-link arm along x: joints at 0, 0.5, 0.9, tip at 1.2
    let joints = [
        Geonum::scalar(0.0),
        Geonum::new(0.5, 0.0, 1.0),
        Geonum::new(0.9, 0.0, 1.0),
    ];
    let tip = Geonum::new(1.2, 0.0, 1.0);

    let quarter = Angle::new(1.0, 2.0);
    let levers: Vec<Geonum> = joints.iter().map(|j| (tip - *j).rotate(quarter)).collect();

    // rates chosen so the weighted levers cancel: 1.0·1.2 + 1.0·0.7 = (1.9/0.3)·0.3
    let null_rates = [1.0, 1.0, -(1.2 + 0.7) / 0.3];
    let tip_velocity = levers
        .iter()
        .zip(null_rates)
        .fold(Geonum::scalar(0.0), |v, (lever, rate)| {
            v + lever.scale(rate)
        });
    assert!(
        tip_velocity.near_mag(0.0),
        "joints spin, the tip stands still — null-space motion, no SVD"
    );

    // the same levers at a uniform rate sweep the tip at the summed radii —
    // the contrast showing the cancellation above is the null space, not a
    // degenerate arm
    let swing = levers
        .iter()
        .fold(Geonum::scalar(0.0), |v, lever| v + lever.scale(1.0));
    assert!(
        swing.near_mag(1.2 + 0.7 + 0.3),
        "uniform rates: the tip sweeps at Σ radii"
    );
}

// ---------------------------------------------------------------------------
// clearance: point-to-path distance is one rejection
// ---------------------------------------------------------------------------
#[test]
fn it_clears_obstacles_with_the_rejection() {
    // collision checking against a straight motion is projection/rejection:
    // the projection places the obstacle along the path, the rejection IS the
    // clearance. no closest-point iteration, no mesh pair tests

    let motion = Geonum::new(4.0, 0.0, 1.0); // path from the origin, 4 along x

    // obstacle beside the path at hand-known geometry (2, 1.5)
    let obstacle = Geonum::new_from_cartesian(2.0, 1.5);
    let along = obstacle.project(&motion);
    let clearance = obstacle.reject(&motion);
    assert!(
        (along.mag - 2.0).abs() < EPSILON,
        "foot of the perpendicular at x = 2"
    );
    assert!(
        (clearance.mag - 1.5).abs() < EPSILON,
        "clearance is the rejection: 1.5"
    );

    // the path parameter is the projection over the path length: abeam
    let t_param = along.mag / motion.mag;
    assert!((0.0..=1.0).contains(&t_param), "obstacle abeam the segment");

    // an obstacle past the end projects beyond the segment — not this leg's
    // problem
    let beyond = Geonum::new_from_cartesian(5.0, 1.0);
    let t_beyond = beyond.project(&motion).mag / motion.mag;
    assert!(t_beyond > 1.0, "projects past the goal");

    // a graze: clearance under the envelope radius blocks the motion
    let graze = Geonum::new_from_cartesian(2.0, 0.2);
    assert!(
        graze.reject(&motion).mag < 0.3,
        "0.2 clearance < 0.3 envelope: replan"
    );
}

// ---------------------------------------------------------------------------
// feedback: the proportional step scales the error magnitude exactly
// ---------------------------------------------------------------------------
#[test]
fn it_steers_a_swarm_by_scaling_the_error() {
    // proportional control x += k·(goal − x) shrinks every error magnitude by
    // exactly (1 − k) per step — the gain acts on the error's magnitude while
    // its angle rides along. convergence is a geometric series read off scale,
    // no gain matrix, no eigenvalue analysis

    let n = 16;
    let gain = 0.1;

    let mut swarm: Vec<Geonum> = (0..n)
        .map(|i| Geonum::new_from_cartesian((i % 4) as f64, (i / 4) as f64))
        .collect();
    let goals: Vec<Geonum> = (0..n)
        .map(|i| Geonum::new_with_angle(6.0, Angle::new(2.0 * i as f64 / n as f64, 1.0)))
        .collect();

    // one step: every robot's error shrinks by the same exact factor
    for (robot, goal) in swarm.iter_mut().zip(&goals) {
        let before = (*goal - *robot).mag;
        *robot = *robot + (*goal - *robot).scale(gain);
        let after = (*goal - *robot).mag;
        assert!(
            (after / before - (1.0 - gain)).abs() < 1e-9,
            "the error scales by exactly 1 − k"
        );
    }

    // twenty more steps: the geometric series, still exact
    let start_error: f64 = swarm.iter().zip(&goals).map(|(r, g)| (*g - *r).mag).sum();
    for _ in 0..20 {
        for (robot, goal) in swarm.iter_mut().zip(&goals) {
            *robot = *robot + (*goal - *robot).scale(gain);
        }
    }
    let end_error: f64 = swarm.iter().zip(&goals).map(|(r, g)| (*g - *r).mag).sum();
    assert!(
        (end_error / start_error - (1.0 - gain).powi(20)).abs() < 1e-6,
        "twenty steps shrink the formation error by (1 − k)^20"
    );
}

// ---------------------------------------------------------------------------
// planning scale: a million-dimensional c-space is still two numbers
// ---------------------------------------------------------------------------
#[test]
fn it_plans_in_a_million_dimensional_configuration_space() {
    // a configuration is [magnitude, angle] whatever the ambient dimension —
    // dimensions are blade addresses, not storage. projections onto any axis
    // and distances between configurations stay O(1) where coordinate planners
    // store n components and decomposed geometric algebras store 2^n

    let config = Geonum::new(100.0, 1.0, 4.0); // [100, π/4]

    // the projection onto dimension k is 100·cos(kπ/2 − π/4) — spot-checked at
    // the near axes and at the millionth dimension, same one-cosine cost
    for (dim, reference) in [
        (0usize, 100.0 * (PI / 4.0).cos()),
        (1, 100.0 * (PI / 4.0).cos()),
        (2, -100.0 * (PI / 4.0).cos()),
        (1_000_000, 100.0 * (PI / 4.0).cos()), // 1M ≡ 0 mod 4
    ] {
        assert!(
            (config.project_to_dimension(dim) - reference).abs() < EPSILON,
            "projection onto dimension {dim} costs one cosine"
        );
    }

    // c-space distance by the law of cosines — one op between configurations,
    // no n-term coordinate sum
    let obstacle = Geonum::new(120.0, 1.0, 2.0); // [120, π/2]
    let clearance = config.distance_to(&obstacle);
    let reference =
        (100.0_f64.powi(2) + 120.0_f64.powi(2) - 2.0 * 100.0 * 120.0 * (PI / 4.0).cos()).sqrt();
    assert!(
        (clearance.mag - reference).abs() < EPSILON,
        "c-space clearance from the law of cosines"
    );

    // a straight-line plan is one subtraction, and halfway along it half the
    // distance remains — segment interpolation with no coordinate vectors
    let goal = Geonum::new(150.0, 3.0, 4.0);
    let segment = goal - config;
    let step = config + segment.scale(0.5);
    assert!(
        (step.distance_to(&goal).mag - segment.mag * 0.5).abs() < 1e-9,
        "halfway along the plan, half the distance remains"
    );
}
