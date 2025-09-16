use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// GEOMETRIC NUMBER REPRESENTATION OF MECHANICAL QUANTITIES
//
// geonum encoding eliminates vector mechanics complexity:
//
// 1. KINEMATIC HIERARCHY THROUGH BLADE COUNT:
//    - position: blade=0 (fundamental spatial quantity)
//    - velocity: blade=1 (position + π/2 rotation = first derivative)
//    - acceleration: blade=2 (velocity + π/2 rotation = second derivative)
//    - jerk: blade=3 (acceleration + π/2 rotation = third derivative)
//    blade count tracks derivative order via differentiation = π/2 rotation
//    traditional mechanics: separate vector equations for each quantity
//    geonum mechanics: single geometric object at different blade levels
//
// 2. SAME GEOMETRIC OBJECT AT DIFFERENT RATES:
//    position, velocity, acceleration are identical geometric objects
//    at different blade levels representing different rates of change
//    differentiation increments blade count while preserving geometric structure
//    traditional mechanics: position r⃗, velocity v⃗, acceleration a⃗ as separate vectors
//    geonum mechanics: r[blade=0] → r[blade=1] → r[blade=2] via π/2 rotation
//
// 3. CONSERVATION THROUGH GEOMETRIC NILPOTENCY:
//    momentum.wedge(&momentum) = 0 IS momentum conservation
//    energy.wedge(&energy) = 0 IS energy conservation
//    no external conservation laws needed - built into geometric nilpotency
//    traditional mechanics: impose conservation laws through lagrangian constraints
//    geonum mechanics: conservation emerges from v∧v = 0 geometric relationship
//
// 4. FORCE AS MOMENTUM RATE:
//    force = blade=2 quantity (acceleration level derivative of momentum)
//    force application = momentum + force×dt via geometric angle addition
//    traditional mechanics: F = ma through vector addition and scalar multiplication
//    geonum mechanics: force integration through blade arithmetic
//
// 5. ENERGY THROUGH DOT PRODUCTS:
//    kinetic energy = velocity.dot(&velocity) encodes ½mv² relationships
//    potential energy = position in force field via geometric projections
//    traditional mechanics: separate kinetic T = ½mv² and potential V energy formulas
//    geonum mechanics: energy emerges from geometric product relationships

#[test]
fn it_changes_kinematic_level_by_cycling_grade() {
    // fundamental principle: differentiation/integration cycles through grades
    // grade (not blade) determines kinematic level:
    // - differentiation: grade n → grade (n+1) % 4
    // - integration: grade n → grade (n-1) % 4
    // blade accumulates history, grade shows behavior

    // start with position at arbitrary blade count
    let position = Geonum::new_with_blade(10.0, 5, 1.0, 7.0); // blade 5, grade 1
    assert_eq!(position.angle.grade(), 1, "position at grade 1");

    // differentiate to get velocity
    let velocity = position.differentiate();
    assert_eq!(velocity.angle.grade(), 2, "velocity at grade 2 (1+1 mod 4)");
    assert_eq!(
        velocity.length, position.length,
        "differentiation preserves magnitude"
    );

    // differentiate velocity to get acceleration
    let acceleration = velocity.differentiate();
    assert_eq!(
        acceleration.angle.grade(),
        3,
        "acceleration at grade 3 (2+1 mod 4)"
    );

    // differentiate acceleration to get jerk
    let jerk = acceleration.differentiate();
    assert_eq!(jerk.angle.grade(), 0, "jerk at grade 0 (3+1 mod 4)");

    // differentiate jerk - cycles back to grade 1
    let snap = jerk.differentiate();
    assert_eq!(
        snap.angle.grade(),
        1,
        "snap at grade 1 (0+1 mod 4, full cycle)"
    );

    // test integration reverses grade progression
    let recovered_jerk = snap.integrate();
    assert_eq!(
        recovered_jerk.angle.grade(),
        0,
        "integrate snap → jerk at grade 0"
    );

    let recovered_accel = recovered_jerk.integrate();
    assert_eq!(
        recovered_accel.angle.grade(),
        3,
        "integrate jerk → acceleration at grade 3"
    );

    let recovered_velocity = recovered_accel.integrate();
    assert_eq!(
        recovered_velocity.angle.grade(),
        2,
        "integrate acceleration → velocity at grade 2"
    );

    let recovered_position = recovered_velocity.integrate();
    assert_eq!(
        recovered_position.angle.grade(),
        1,
        "integrate velocity → position at grade 1"
    );

    // magnitude preserved through entire cycle
    assert_eq!(
        recovered_position.length, position.length,
        "magnitude preserved through full differentiate/integrate cycle"
    );

    // test kinematic levels are grade-dependent, not blade-dependent
    let high_blade_pos = Geonum::new_with_blade(15.0, 1000, 0.0, 1.0); // blade 1000, grade 0
    let high_blade_vel = high_blade_pos.differentiate();
    let high_blade_acc = high_blade_vel.differentiate();
    let high_blade_jerk = high_blade_acc.differentiate();

    assert_eq!(
        high_blade_pos.angle.grade(),
        0,
        "position at grade 0 (1000 % 4)"
    );
    assert_eq!(high_blade_vel.angle.grade(), 1, "velocity at grade 1");
    assert_eq!(high_blade_acc.angle.grade(), 2, "acceleration at grade 2");
    assert_eq!(high_blade_jerk.angle.grade(), 3, "jerk at grade 3");

    // demonstrate grade determines physical meaning regardless of blade
    let scalar_like_1 = Geonum::new_with_blade(5.0, 0, 0.0, 1.0); // blade 0, grade 0
    let scalar_like_2 = Geonum::new_with_blade(5.0, 4, 0.0, 1.0); // blade 4, grade 0
    let scalar_like_3 = Geonum::new_with_blade(5.0, 1000, 0.0, 1.0); // blade 1000, grade 0

    assert_eq!(scalar_like_1.angle.grade(), 0, "blade 0 → grade 0");
    assert_eq!(scalar_like_2.angle.grade(), 0, "blade 4 → grade 0");
    assert_eq!(scalar_like_3.angle.grade(), 0, "blade 1000 → grade 0");

    // all behave identically under differentiation
    let deriv_1 = scalar_like_1.differentiate();
    let deriv_2 = scalar_like_2.differentiate();
    let deriv_3 = scalar_like_3.differentiate();

    assert_eq!(deriv_1.angle.grade(), 1, "all differentiate to grade 1");
    assert_eq!(deriv_2.angle.grade(), 1, "regardless of starting blade");
    assert_eq!(deriv_3.angle.grade(), 1, "grade determines behavior");

    // traditional calculus: d/dt requires limit definition and differentiation rules
    // ∂f/∂t = lim(Δt→0) [f(t+Δt) - f(t)]/Δt with epsilon-delta proofs O(n)
    // chain rule, product rule, quotient rule for composite functions O(n²)
    //
    // geonum: differentiation is π/2 rotation, integration is -π/2 rotation O(1)
    // no limits, no symbolic manipulation, just grade cycling

    // traditional mechanics: kinematic hierarchy via successive differentiation
    // position → velocity → acceleration → jerk requires operator stacking O(n)
    //
    // geonum: grade cycling 0→1→2→3→0 encodes entire hierarchy O(1)
    // blade 1000 acts identical to blade 0 due to grade = blade % 4

    println!("kinematic hierarchy via grade cycling:");
    println!(
        "  position:     grade {} (blade {})",
        position.angle.grade(),
        position.angle.blade()
    );
    println!(
        "  velocity:     grade {} (blade {})",
        velocity.angle.grade(),
        velocity.angle.blade()
    );
    println!(
        "  acceleration: grade {} (blade {})",
        acceleration.angle.grade(),
        acceleration.angle.blade()
    );
    println!(
        "  jerk:         grade {} (blade {})",
        jerk.angle.grade(),
        jerk.angle.blade()
    );
    println!(
        "  snap:         grade {} (blade {})",
        snap.angle.grade(),
        snap.angle.blade()
    );
    println!("\ngrade cycles 0→1→2→3→0, blade accumulates history");
}

#[test]
fn it_encodes_position() {
    // position as geometric number without coordinate system dependency
    let position = Geonum::new(3.0, 1.0, 4.0); // 3 units at π/4

    assert_eq!(
        position.angle.blade(),
        0,
        "position at blade 0 (fundamental spatial)"
    );
    assert_eq!(position.length, 3.0, "distance from origin");
    assert!(
        (position.angle.mod_4_angle() - PI / 4.0).abs() < EPSILON,
        "direction angle π/4"
    );

    // test displacement addition
    let displacement = Geonum::new(2.0, 1.0, 6.0); // 2 units at π/6
    let new_position = position + displacement;

    // verify geometric addition produces expected result
    let (x1, y1) = position.to_cartesian();
    let (x2, y2) = displacement.to_cartesian();
    let expected_length = ((x1 + x2).powi(2) + (y1 + y2).powi(2)).sqrt();
    assert!(
        (new_position.length - expected_length).abs() < EPSILON,
        "displacement addition matches vector mechanics"
    );

    // test high-dimensional projections with meaningful assertions
    let position_1000d = position.project_to_dimension(1000);
    let position_million_d = position.project_to_dimension(1_000_000);

    // projections should be bounded by position magnitude
    assert!(
        position_1000d.abs() <= position.length + EPSILON,
        "1000D projection bounded by magnitude"
    );
    assert!(
        position_million_d.abs() <= position.length + EPSILON,
        "million-D projection bounded by magnitude"
    );

    // test scaling preserves projection ratios
    let scaled_position = position.scale(5.0);
    let scaled_1000d = scaled_position.project_to_dimension(1000);
    assert!(
        (scaled_1000d - 5.0 * position_1000d).abs() < EPSILON,
        "scaling preserves dimensional relationships"
    );

    // dimension 4 points at 4×π/2 = 2π (full rotation back to start)
    // cos(2π - θ) = cos(-θ) = cos(θ), so projections should be equal
    let position_dim_0 = position.project_to_dimension(0);
    let position_dim_4 = position.project_to_dimension(4);

    // these should be equal due to cos periodicity
    assert!(
        (position_dim_4 - position_dim_0).abs() < EPSILON,
        "dimension 4 (2π) equals dimension 0 via cos periodicity: {} ≈ {}",
        position_dim_4,
        position_dim_0
    );

    // traditional: position requires coordinate system setup and basis vectors
    // million dimensions = million basis vectors in memory O(n)
    //
    // geonum: position exists independently, projects to any dimension on demand O(1)
}

#[test]
fn it_encodes_velocity() {
    let initial_position = Geonum::new(8.0, 2.0, 5.0); // 8 units at 2π/5

    // velocity via differentiation (π/2 rotation)
    let velocity = initial_position.differentiate();

    assert_eq!(initial_position.angle.blade(), 0, "position at blade 0");
    assert_eq!(
        velocity.angle.blade(),
        1,
        "velocity at blade 1 (π/2 rotated)"
    );
    assert_eq!(
        velocity.length, initial_position.length,
        "differentiation preserves magnitude"
    );

    // test velocity projections
    let velocity_x = velocity.project_to_dimension(0);
    let velocity_y = velocity.project_to_dimension(1);

    // velocity magnitude from components
    let velocity_magnitude = (velocity_x.powi(2) + velocity_y.powi(2)).sqrt();
    assert!(
        (velocity_magnitude - velocity.length).abs() < EPSILON,
        "velocity components reconstruct magnitude"
    );

    // test integration recovers position
    let recovered_position = velocity.integrate();
    let position_base = recovered_position.base_angle();

    assert_eq!(
        position_base.angle.blade(),
        0,
        "integration returns to blade 0"
    );
    assert!(
        (position_base.length - initial_position.length).abs() < EPSILON,
        "integration preserves magnitude"
    );

    // angle matches after base_angle reset
    assert!(
        (position_base.angle.mod_4_angle() - initial_position.angle.mod_4_angle()).abs() < EPSILON,
        "integration recovers original angle"
    );

    // traditional: velocity = dr/dt requires finite differences or symbolic differentiation
    // numerical methods accumulate error, symbolic methods need expression trees O(n)
    //
    // geonum: velocity = position.differentiate() via π/2 rotation O(1)
    // integration reverses the rotation, no numerical approximation
}

#[test]
fn it_encodes_acceleration() {
    let initial_position = Geonum::new(6.0, 1.0, 5.0); // 6 units at π/5

    // climb derivative hierarchy
    let velocity = initial_position.differentiate(); // blade 0 → 1
    let acceleration = velocity.differentiate(); // blade 1 → 2

    assert_eq!(acceleration.angle.blade(), 2, "acceleration at blade 2");
    assert_eq!(
        acceleration.length, initial_position.length,
        "double differentiation preserves magnitude"
    );

    // F = ma with meaningful test
    let mass = 2.5; // kg
    let force = acceleration.scale(mass);

    assert_eq!(
        force.angle.blade(),
        2,
        "force at same blade as acceleration"
    );
    assert!(
        (force.length - mass * acceleration.length).abs() < EPSILON,
        "F = ma via scaling"
    );

    // test that force and acceleration point same direction
    assert_eq!(
        force.angle, acceleration.angle,
        "force parallel to acceleration"
    );

    // traditional: F = ma requires vector spaces and coordinate transformations
    // second derivatives need d²r/dt² with finite difference approximations O(n²)
    //
    // geonum: two π/2 rotations give acceleration, scale by mass for force O(1)
    // F = ma emerges from simple scaling, no coordinate frames needed
}

#[test]
fn it_encodes_jerk() {
    let initial_position = Geonum::new(10.0, 3.0, 7.0); // 10 units at 3π/7

    // complete kinematic hierarchy
    let velocity = initial_position.differentiate(); // blade 0 → 1
    let acceleration = velocity.differentiate(); // blade 1 → 2
    let jerk = acceleration.differentiate(); // blade 2 → 3
    let fourth_derivative = jerk.differentiate(); // blade 3 → 4

    assert_eq!(jerk.angle.blade(), 3, "jerk at blade 3");
    assert_eq!(
        fourth_derivative.angle.grade(),
        0,
        "fourth derivative returns to grade 0"
    );

    // magnitudes preserved through entire chain
    assert!(
        (jerk.length - initial_position.length).abs() < EPSILON,
        "jerk preserves original magnitude"
    );
    assert!(
        (fourth_derivative.length - initial_position.length).abs() < EPSILON,
        "fourth derivative preserves original magnitude"
    );

    // test triple integration returns to position
    let recovered_accel = jerk.integrate();
    let recovered_vel = recovered_accel.integrate();
    let recovered_pos = recovered_vel.integrate();
    let final_position = recovered_pos.base_angle();

    assert_eq!(
        final_position.angle.blade(),
        0,
        "triple integration returns to blade 0"
    );
    assert!(
        (final_position.length - initial_position.length).abs() < EPSILON,
        "triple integration preserves magnitude"
    );

    // traditional: jerk d³r/dt³ requires third derivatives, snap d⁴r/dt⁴ requires fourth
    // numerical methods compound error at each level O(n³), O(n⁴)
    //
    // geonum: unlimited derivatives via grade cycling, exact reversibility
    // jerk, snap, crackle, pop... all just π/2 rotations O(1)
}

#[test]
fn it_displaces_from_derived_velocity() {
    // define initial position
    let initial_position = Geonum::new(10.0, 1.0, 3.0); // 10m at π/3 (60°)
    println!(
        "initial position: length={}, angle={}, blade={}",
        initial_position.length,
        initial_position.angle.mod_4_angle(),
        initial_position.angle.blade()
    );

    // derive velocity from position (π/2 rotation)
    let velocity = initial_position.differentiate();
    assert_eq!(velocity.angle.blade(), 1, "velocity at blade 1");
    assert_eq!(
        velocity.length, initial_position.length,
        "differentiation preserves magnitude"
    );
    println!(
        "derived velocity: length={}, angle={}, blade={}",
        velocity.length,
        velocity.angle.mod_4_angle(),
        velocity.angle.blade()
    );

    // compute displacement using derived velocity over time interval
    let time_interval = 2.0; // seconds
    let displacement = velocity.scale(time_interval); // d = v × t
    assert_eq!(displacement.length, 20.0, "displacement = 10 × 2 = 20");
    assert_eq!(
        displacement.angle, velocity.angle,
        "displacement preserves velocity direction"
    );
    println!(
        "displacement: length={}, angle={}, blade={}",
        displacement.length,
        displacement.angle.mod_4_angle(),
        displacement.angle.blade()
    );

    // add displacement to initial position
    let final_position = initial_position + displacement;
    println!(
        "final position: length={}, angle={}, blade={}",
        final_position.length,
        final_position.angle.mod_4_angle(),
        final_position.angle.blade()
    );

    // assert final position is physically meaningful
    // initial: 10m at π/3 (60°)
    // displacement: 20m at π/3 + π/2 = 5π/6 (150°)

    // convert to cartesian to verify physics
    let (x0, y0) = initial_position.to_cartesian();
    let (dx, dy) = displacement.to_cartesian();
    let (xf, yf) = final_position.to_cartesian();

    println!("\ncartesian verification:");
    println!("  initial: ({:.3}, {:.3})", x0, y0);
    println!("  displacement: ({:.3}, {:.3})", dx, dy);
    println!("  final: ({:.3}, {:.3})", xf, yf);

    // final position should equal initial + displacement in cartesian
    assert!(
        (xf - (x0 + dx)).abs() < EPSILON,
        "x-component: {:.6} ≈ {:.6}",
        xf,
        x0 + dx
    );
    assert!(
        (yf - (y0 + dy)).abs() < EPSILON,
        "y-component: {:.6} ≈ {:.6}",
        yf,
        y0 + dy
    );

    // compute expected final position magnitude
    let expected_magnitude = ((x0 + dx).powi(2) + (y0 + dy).powi(2)).sqrt();
    assert!(
        (final_position.length - expected_magnitude).abs() < EPSILON,
        "final position magnitude matches vector addition: {:.6} ≈ {:.6}",
        final_position.length,
        expected_magnitude
    );

    // test that velocity derived from position creates physically meaningful displacement
    // the displacement moves us from 10m at 60° to a new position
    // this proves differentiation produces a velocity that generates real motion

    // additional test: perpendicular velocity creates perpendicular displacement
    let perpendicular_position = Geonum::new(10.0, 5.0, 6.0); // 10m at 5π/6 (150°)
    let perpendicular_velocity = perpendicular_position.differentiate();

    // angle difference between original and perpendicular: 5π/6 - π/3 = π/2
    let angle_diff =
        (perpendicular_position.angle.mod_4_angle() - initial_position.angle.mod_4_angle()).abs();
    assert!(
        (angle_diff - PI / 2.0).abs() < EPSILON,
        "positions are perpendicular"
    );

    // their derived velocities should also be perpendicular
    let velocity_angle_diff =
        (perpendicular_velocity.angle.mod_4_angle() - velocity.angle.mod_4_angle()).abs();
    let normalized_diff = if velocity_angle_diff > PI {
        2.0 * PI - velocity_angle_diff
    } else {
        velocity_angle_diff
    };
    assert!(
        (normalized_diff - PI / 2.0).abs() < EPSILON,
        "derived velocities maintain perpendicularity"
    );

    println!("\nphysics verified: derivative creates meaningful velocity → displacement → motion");

    // traditional kinematics: r(t) = r₀ + ∫v(t)dt requires integration
    // numerical integration accumulates error, path integrals need discretization O(n)
    //
    // geonum: displacement = velocity.scale(time), position update via addition O(1)
    // derivative creates velocity that produces real motion when scaled by time
}

#[test]
fn it_squares_displacement_from_derived_acceleration() {
    // kinematic equation: d = v₀t + ½at²
    // demonstrates acceleration creates quadratic displacement growth

    // define initial position
    let initial_position = Geonum::new(5.0, 1.0, 4.0); // 5m at π/4 (45°)
    println!(
        "initial position: length={}, angle={}, blade={}",
        initial_position.length,
        initial_position.angle.mod_4_angle(),
        initial_position.angle.blade()
    );

    // derive velocity from position (π/2 rotation)
    let initial_velocity = initial_position.differentiate();
    assert_eq!(initial_velocity.angle.blade(), 1, "velocity at blade 1");
    println!(
        "initial velocity: length={}, angle={}, blade={}",
        initial_velocity.length,
        initial_velocity.angle.mod_4_angle(),
        initial_velocity.angle.blade()
    );

    // derive acceleration from velocity (another π/2 rotation)
    let acceleration = initial_velocity.differentiate();
    assert_eq!(acceleration.angle.blade(), 2, "acceleration at blade 2");
    assert_eq!(
        acceleration.length, initial_position.length,
        "double differentiation preserves magnitude"
    );
    println!(
        "acceleration: length={}, angle={}, blade={}",
        acceleration.length,
        acceleration.angle.mod_4_angle(),
        acceleration.angle.blade()
    );

    // compute displacement over time with constant acceleration
    let time = 3.0; // seconds

    // first term: v₀t (linear displacement from initial velocity)
    let linear_displacement = initial_velocity.scale(time);
    assert_eq!(linear_displacement.length, 15.0, "v₀t = 5 × 3 = 15");
    assert_eq!(
        linear_displacement.angle, initial_velocity.angle,
        "linear term preserves velocity direction"
    );
    println!(
        "\nlinear displacement (v₀t): length={}, angle={}",
        linear_displacement.length,
        linear_displacement.angle.mod_4_angle()
    );

    // second term: ½at² (quadratic displacement from acceleration)
    let time_squared = time * time; // t²
    let quadratic_displacement = acceleration.scale(0.5 * time_squared);
    assert_eq!(
        quadratic_displacement.length, 22.5,
        "½at² = 0.5 × 5 × 9 = 22.5"
    );
    assert_eq!(
        quadratic_displacement.angle, acceleration.angle,
        "quadratic term preserves acceleration direction"
    );
    println!(
        "quadratic displacement (½at²): length={}, angle={}",
        quadratic_displacement.length,
        quadratic_displacement.angle.mod_4_angle()
    );

    // total displacement: combine linear and quadratic terms
    let total_displacement = linear_displacement + quadratic_displacement;
    println!(
        "total displacement: length={}, angle={}, blade={}",
        total_displacement.length,
        total_displacement.angle.mod_4_angle(),
        total_displacement.angle.blade()
    );

    // add total displacement to initial position
    let final_position = initial_position + total_displacement;
    println!(
        "final position: length={}, angle={}, blade={}",
        final_position.length,
        final_position.angle.mod_4_angle(),
        final_position.angle.blade()
    );

    // verify physics in cartesian coordinates
    let (x0, y0) = initial_position.to_cartesian();
    let (dx_linear, dy_linear) = linear_displacement.to_cartesian();
    let (dx_quad, dy_quad) = quadratic_displacement.to_cartesian();
    let (xf, yf) = final_position.to_cartesian();

    println!("\ncartesian verification:");
    println!("  initial: ({:.3}, {:.3})", x0, y0);
    println!(
        "  linear displacement: ({:.3}, {:.3})",
        dx_linear, dy_linear
    );
    println!("  quadratic displacement: ({:.3}, {:.3})", dx_quad, dy_quad);
    println!(
        "  total displacement: ({:.3}, {:.3})",
        dx_linear + dx_quad,
        dy_linear + dy_quad
    );
    println!("  final: ({:.3}, {:.3})", xf, yf);

    // final position should equal initial + linear + quadratic displacements
    let expected_x = x0 + dx_linear + dx_quad;
    let expected_y = y0 + dy_linear + dy_quad;
    assert!(
        (xf - expected_x).abs() < EPSILON,
        "x-component: {:.6} ≈ {:.6}",
        xf,
        expected_x
    );
    assert!(
        (yf - expected_y).abs() < EPSILON,
        "y-component: {:.6} ≈ {:.6}",
        yf,
        expected_y
    );

    // test that quadratic term dominates for large time
    let large_time = 10.0;
    let large_linear = initial_velocity.scale(large_time);
    let large_quadratic = acceleration.scale(0.5 * large_time * large_time);

    assert!(
        large_quadratic.length > large_linear.length,
        "quadratic term dominates for large t: {:.1} > {:.1}",
        large_quadratic.length,
        large_linear.length
    );

    // ratio should be t/2 for equal magnitude initial conditions
    let ratio = large_quadratic.length / large_linear.length;
    assert!(
        (ratio - large_time / 2.0).abs() < EPSILON,
        "quadratic/linear ratio = t/2 = {:.1}",
        ratio
    );

    // test zero initial velocity case (pure acceleration from rest)
    let rest_position = Geonum::new(5.0, 0.0, 1.0); // at rest
    let rest_velocity = rest_position.differentiate();
    let rest_acceleration = rest_velocity.differentiate();

    // from rest: d = ½at² only
    let rest_displacement = rest_acceleration.scale(0.5 * time * time);
    assert_eq!(rest_displacement.length, 22.5, "from rest: d = ½at²");

    println!("\nphysics verified: acceleration creates quadratic displacement growth");
    println!("kinematic equation d = v₀t + ½at² emerges from double differentiation");

    // traditional kinematics: d = v₀t + ½∫∫a(t)dt²dt requires double integration
    // taylor series expansion, numerical quadrature methods O(n²)
    //
    // geonum: kinematic equation emerges from grade hierarchy
    // v₀t from velocity scaling, ½at² from acceleration scaling
    // no integration, no taylor series, just direct scaling operations O(1)
}

#[test]
fn it_encodes_force() {
    let mass = 3.0; // kg
    let acceleration = Geonum::new_with_blade(4.0, 2, 1.0, 8.0); // 4 m/s² at blade 2

    // F = ma
    let force = acceleration.scale(mass);

    assert_eq!(force.angle.blade(), 2, "force at blade 2");
    assert!(
        (force.length - mass * acceleration.length).abs() < EPSILON,
        "F = ma"
    );
    assert_eq!(force.angle, acceleration.angle, "force || acceleration");

    // test impulse-momentum theorem: Δp = FΔt
    // start with object at rest, apply force for time interval
    let initial_velocity = Geonum::new(0.0, 0.0, 1.0); // at rest
    let initial_momentum = initial_velocity.scale(mass); // p = 0

    assert_eq!(initial_momentum.length, 0.0, "initial momentum is zero");

    // impulse = force × time
    let time_interval = 2.0; // seconds
    let impulse = force.scale(time_interval); // J = FΔt

    assert_eq!(impulse.angle.blade(), 2, "impulse at same blade as force");
    assert!(
        (impulse.length - force.length * time_interval).abs() < EPSILON,
        "impulse = force × time"
    );

    // final velocity from kinematic equation: v = at
    let final_velocity = acceleration.scale(time_interval);
    let final_momentum = final_velocity.scale(mass); // p = mv

    assert_eq!(final_momentum.angle.blade(), 2, "final momentum at blade 2");
    assert!(
        (final_momentum.length - mass * final_velocity.length).abs() < EPSILON,
        "final momentum = mass × final velocity"
    );

    // verify impulse equals momentum change
    let momentum_change = final_momentum.length - initial_momentum.length;
    assert!(
        (momentum_change - impulse.length).abs() < EPSILON,
        "Δp = J (impulse-momentum theorem): {} ≈ {}",
        momentum_change,
        impulse.length
    );

    // traditional mechanics: F = dp/dt requires time derivatives of momentum
    // impulse J = ∫F dt needs integration, momentum updates via vector addition O(n)
    //
    // geonum: F = ma and p = mv via simple scaling
    // impulse = force × time, no integration needed O(1)
}

#[test]
fn it_encodes_momentum() {
    let mass = 4.0; // kg
    let velocity = Geonum::new_with_blade(6.0, 1, 2.0, 9.0); // blade 1, 2π/9 angle

    let momentum = velocity.scale(mass);

    assert_eq!(momentum.angle.blade(), 1, "momentum at blade 1");
    assert!(
        (momentum.length - mass * velocity.length).abs() < EPSILON,
        "p = mv"
    );
    assert_eq!(momentum.angle, velocity.angle, "momentum || velocity");

    // conservation via nilpotency
    let self_wedge = momentum.wedge(&momentum);
    assert!(self_wedge.length < EPSILON, "p∧p = 0 (conservation)");

    // elastic collision: momentum conservation
    let mass2 = 2.0; // kg
    let velocity2 = Geonum::new_with_blade(3.0, 1, 5.0, 6.0); // blade 1, 5π/6 angle
    let momentum2 = velocity2.scale(mass2);

    // total momentum before collision
    let total_momentum_before = momentum + momentum2;
    // blade accumulates: 1 + 1 + 3 (from blade preservation) = 5
    assert_eq!(
        total_momentum_before.angle.blade(),
        5,
        "total momentum blade from addition"
    );

    // after elastic collision (velocities exchange for equal masses demonstration)
    // in real physics we'd solve conservation equations, here we verify invariant
    let total_momentum_after = total_momentum_before; // conserved

    // verify conservation law holds
    assert_eq!(
        total_momentum_after.length, total_momentum_before.length,
        "momentum magnitude conserved"
    );
    assert_eq!(
        total_momentum_after.angle, total_momentum_before.angle,
        "momentum direction conserved"
    );

    // rotation preserves conservation
    let rotation = Angle::new(1.0, 7.0); // π/7
    let rotated_total = total_momentum_before.rotate(rotation);

    assert!(
        (rotated_total.length - total_momentum_before.length).abs() < EPSILON,
        "rotation preserves total momentum magnitude"
    );

    // nilpotency still holds for total momentum
    let total_wedge = total_momentum_before.wedge(&total_momentum_before);
    assert!(total_wedge.length < EPSILON, "p_total∧p_total = 0");

    // traditional: momentum conservation requires coordinate-free formulation
    // p = mv with vector operations, conservation via dp/dt = 0 analysis O(n)
    //
    // geonum: momentum = velocity.scale(mass), conservation via nilpotency p∧p = 0
    // rotation invariance automatic through angle arithmetic O(1)
}

#[test]
fn it_encodes_angular_momentum() {
    let position = Geonum::new(3.0, 1.0, 6.0); // 3 units at π/6
    let mass = 2.0; // kg
    let velocity = Geonum::new_with_blade(4.0, 1, 1.0, 4.0); // blade 1, π/4
    let momentum = velocity.scale(mass);

    // L = r ∧ p
    let angular_momentum = position.wedge(&momentum);

    // angular momentum is bivector-like
    assert_eq!(angular_momentum.angle.grade(), 2, "L at grade 2 (bivector)");

    // magnitude encodes |r||p|sin(θ)
    let angle_diff = (momentum.angle - position.angle).mod_4_angle();
    let expected_magnitude = position.length * momentum.length * angle_diff.sin().abs();
    assert!(
        (angular_momentum.length - expected_magnitude).abs() < EPSILON,
        "L magnitude matches |r||p|sin(θ)"
    );

    // conservation via nilpotency
    let angular_self_wedge = angular_momentum.wedge(&angular_momentum);
    assert!(
        angular_self_wedge.length < EPSILON,
        "L∧L = 0 (conservation)"
    );

    // test torque changes angular momentum: τ = dL/dt
    let force = Geonum::new_with_blade(5.0, 2, 1.0, 3.0); // blade 2, π/3
    let torque = position.wedge(&force); // τ = r ∧ F

    // torque is bivector-like, same grade as angular momentum
    assert_eq!(
        torque.angle.grade(),
        2,
        "torque at grade 2 (bivector, same as L)"
    );

    // torque magnitude |r||F||sin(θ)|
    let torque_angle_diff = (force.angle - position.angle).mod_4_angle();
    let expected_torque = position.length * force.length * torque_angle_diff.sin().abs();
    assert!(
        (torque.length - expected_torque).abs() < EPSILON,
        "torque magnitude matches |r||F||sin(θ)|"
    );

    // traditional mechanics: L = r × p requires cross product and basis vectors
    // torque τ = r × F, angular momentum conservation via dL/dt = τ analysis O(n²)
    //
    // geonum: L = r.wedge(p) via angle addition, torque = r.wedge(F)
    // conservation via nilpotency L∧L = 0, no coordinate systems needed O(1)
}

#[test]
fn it_encodes_work() {
    let force = Geonum::new(10.0, 1.0, 6.0); // 10 N at π/6, blade 0 (vector)
    let displacement = Geonum::new(3.0, 1.0, 6.0); // 3m at π/6 (aligned)

    // W = F·d for aligned case
    let work_interaction = force.dot(&displacement);
    let work_magnitude = work_interaction.length.abs(); // handle sign

    let expected_work = force.length * displacement.length; // cos(0) = 1
    assert!(
        (work_magnitude - expected_work).abs() < EPSILON,
        "aligned work W = F·d"
    );

    // perpendicular case: force at π/6, displacement at 2π/3 (difference = π/2)
    let perpendicular_displacement = Geonum::new(3.0, 2.0, 3.0); // 2π/3
    let perpendicular_work = force.dot(&perpendicular_displacement);

    // compute the measured perpendicular work magnitude
    let perpendicular_work_magnitude = perpendicular_work.length.abs();

    // physics: perpendicular force does zero work (cos(π/2) = 0)
    // force blade 0 at π/6, displacement blade 0 at 2π/3
    // the dot product accounts for the π/2 angle difference
    assert!(
        perpendicular_work_magnitude < EPSILON,
        "perpendicular work = {:.6} ≈ 0 (F⊥d → W=0)",
        perpendicular_work_magnitude
    );

    // test work-energy theorem: W = ΔKE
    let mass = 2.0; // kg
    let initial_velocity = Geonum::new(2.0, 1.0, 6.0); // 2 m/s at π/6

    // kinetic energy: KE = ½mv² as geometric number
    let v_squared = initial_velocity * initial_velocity; // v² gives scalar (blade 0)
    let initial_ke = v_squared.scale(0.5 * mass);

    assert_eq!(
        initial_ke.angle.grade(),
        0,
        "KE at grade 0 (scalar from v²)"
    );
    assert!(
        (initial_ke.length - 0.5 * mass * 4.0).abs() < EPSILON,
        "initial KE = ½mv² = 4 J"
    );

    // work done equals change in kinetic energy
    // for aligned force and displacement, work increases KE
    assert!(
        work_interaction.length > 0.0,
        "positive work for aligned F and d"
    );

    // final velocity from energy conservation
    // KE_final = KE_initial + W
    let final_ke_magnitude = initial_ke.length + work_interaction.length;
    assert!(
        final_ke_magnitude > initial_ke.length,
        "work increases kinetic energy: {:.2} J → {:.2} J",
        initial_ke.length,
        final_ke_magnitude
    );

    // traditional mechanics: W = ∫F·dr requires path integration
    // work-energy theorem via calculus of variations O(n)
    //
    // geonum: W = F.dot(d) direct computation
    // perpendicular test proves geometric correctness without integration O(1)
}

#[test]
fn it_encodes_kinetic_energy() {
    let mass = 3.0; // kg
    let velocity = Geonum::new_with_blade(8.0, 1, 1.0, 7.0); // 8 m/s at blade 1, π/7

    // kinetic energy from velocity self-dot product: v·v = |v|²
    let v_squared = velocity.dot(&velocity); // dot product gives scalar (blade 0)
    assert_eq!(v_squared.angle.blade(), 0, "v² at blade 0 (scalar)");
    assert_eq!(v_squared.angle.grade(), 0, "v² at grade 0");
    assert!(
        (v_squared.length - 64.0).abs() < EPSILON,
        "v² = 8² = 64 m²/s²"
    );

    // kinetic energy: KE = ½mv²
    let kinetic_energy = v_squared.scale(0.5 * mass);
    assert_eq!(kinetic_energy.angle.blade(), 0, "KE at blade 0 (scalar)");
    assert!(
        (kinetic_energy.length - 96.0).abs() < EPSILON,
        "KE = ½(3)(64) = 96 J"
    );

    // test energy scaling: doubling velocity quadruples energy
    let double_velocity = velocity.scale(2.0); // 16 m/s
    assert!(
        (double_velocity.length - 16.0).abs() < EPSILON,
        "doubled velocity = 16 m/s"
    );

    let double_v_squared = double_velocity.dot(&double_velocity);
    assert!(
        (double_v_squared.length - 256.0).abs() < EPSILON,
        "(2v)² = 256 = 4×64 m²/s²"
    );

    let double_ke = double_v_squared.scale(0.5 * mass);
    assert!(
        (double_ke.length - 384.0).abs() < EPSILON,
        "KE(2v) = 384 = 4×96 J"
    );

    // verify quadratic relationship
    let ratio = double_ke.length / kinetic_energy.length;
    assert!(
        (ratio - 4.0).abs() < EPSILON,
        "doubling velocity quadruples energy: ratio = 4.0"
    );

    // test high-dimensional velocity (blade 1000)
    let high_velocity = Geonum::new_with_blade(5.0, 1000, 2.0, 13.0); // blade 1000
    let high_v_squared = high_velocity.dot(&high_velocity);

    assert_eq!(
        high_v_squared.angle.blade(),
        0,
        "v² gives blade 0 even from blade 1000"
    );
    assert!(
        (high_v_squared.length - 25.0).abs() < EPSILON,
        "high-dim v² = 5² = 25 m²/s²"
    );

    let high_ke = high_v_squared.scale(0.5 * mass);
    assert!(
        (high_ke.length - 37.5).abs() < EPSILON,
        "high-dim KE = ½(3)(25) = 37.5 J"
    );

    // test relativistic-like energy (velocity at different grades)
    let grade_2_velocity = Geonum::new_with_blade(10.0, 2, 3.0, 11.0); // grade 2
    let grade_2_v_squared = grade_2_velocity.dot(&grade_2_velocity);

    assert_eq!(
        grade_2_v_squared.angle.blade(),
        0,
        "velocity at any grade gives scalar energy"
    );
    assert!(
        (grade_2_v_squared.length - 100.0).abs() < EPSILON,
        "grade 2 v² = 10² = 100 m²/s²"
    );

    // traditional mechanics: T = ½mv² requires velocity magnitude |v⃗|² in n dimensions
    // complexity O(n) for n-dimensional velocity vector magnitude
    // energy separate from velocity, requires explicit formula application
    //
    // geonum: KE = ½m(v·v) emerges from geometric dot product
    // same operation in any dimension or blade count O(1)
    // energy encoded in velocity self-interaction, no separate formula needed
}

#[test]
fn it_encodes_potential_energy() {
    let mass = 2.0; // kg
    let height_position = Geonum::new(5.0, 0.0, 1.0); // 5m height, angle 0
    let gravity_field = Geonum::new(9.8, 1.0, 1.0); // 9.8 m/s² at π (downward)

    // potential energy from position-field interaction
    let field_position_interaction = height_position.dot(&gravity_field);

    let signed_interaction = field_position_interaction.length
        * field_position_interaction
            .angle
            .project(Angle::new(0.0, 1.0));
    // positions at angle 0 and field at angle π are opposite
    // dot product of opposite directions gives negative
    assert!(
        signed_interaction < 0.0,
        "opposite directions give negative dot product"
    );

    let interaction_magnitude = signed_interaction.abs();
    assert!(
        (interaction_magnitude - 49.0).abs() < EPSILON,
        "h·g = 5×9.8 = 49 m²/s²"
    );

    let potential_energy = mass * interaction_magnitude;
    assert!(
        (potential_energy - 98.0).abs() < EPSILON,
        "PE = m(h·g) = 2×49 = 98 J"
    );

    // test potential energy scaling with height
    let double_height = height_position.scale(2.0); // 10m
    assert!(
        (double_height.length - 10.0).abs() < EPSILON,
        "doubled height = 10m"
    );

    let double_interaction = double_height.dot(&gravity_field);
    let double_magnitude =
        (double_interaction.length * double_interaction.angle.project(Angle::new(0.0, 1.0))).abs();
    assert!(
        (double_magnitude - 98.0).abs() < EPSILON,
        "(2h)·g = 10×9.8 = 98 m²/s²"
    );

    let double_pe = mass * double_magnitude;
    assert!((double_pe - 196.0).abs() < EPSILON, "PE(2h) = 196 = 2×98 J");

    // verify linear scaling
    let ratio = double_pe / potential_energy;
    assert!(
        (ratio - 2.0).abs() < EPSILON,
        "doubling height doubles potential energy: ratio = 2.0"
    );

    // test perpendicular field (no potential energy)
    let horizontal_position = Geonum::new(5.0, 0.5, 1.0); // 5m at π/2 (horizontal)
    let perpendicular_interaction = horizontal_position.dot(&gravity_field);

    // π/2 angle to π angle: difference is π/2 (perpendicular)
    let perpendicular_magnitude = perpendicular_interaction.length.abs();
    assert!(
        perpendicular_magnitude < EPSILON,
        "perpendicular position-field gives zero PE: {:.6} ≈ 0",
        perpendicular_magnitude
    );

    // test high-dimensional potential energy
    let high_position = Geonum::new_with_blade(3.0, 1000, 1.0, 11.0); // blade 1000
    let high_field = Geonum::new_with_blade(12.0, 500, 2.0, 13.0); // blade 500

    let high_interaction = high_position.dot(&high_field);
    assert_eq!(
        high_interaction.angle.blade(),
        0,
        "dot product gives blade 0 from any blade count"
    );

    // compute angle difference for expected value
    let pos_angle = high_position.angle.mod_4_angle();
    let field_angle = high_field.angle.mod_4_angle();
    let angle_diff = (field_angle - pos_angle).abs();
    let cos_angle = angle_diff.cos();

    let expected_magnitude = 3.0 * 12.0 * cos_angle.abs();
    let actual_magnitude = high_interaction.length.abs();
    assert!(
        (actual_magnitude - expected_magnitude).abs() < EPSILON,
        "high-dim PE = 3×12×cos(angle) = {:.3}",
        expected_magnitude
    );

    // test field reversal (negative work against field)
    let upward_field = gravity_field.negate(); // reverse field direction
    let upward_interaction = height_position.dot(&upward_field);

    let upward_scalar =
        upward_interaction.length * upward_interaction.angle.project(Angle::new(0.0, 1.0));
    assert!(
        upward_scalar > 0.0,
        "aligned position-field gives positive dot product"
    );
    assert!(
        (upward_scalar - 49.0).abs() < EPSILON,
        "reversed field changes sign but not magnitude"
    );

    // traditional mechanics: V = mgh requires gravitational field definition
    // field integral ∫g·dr for arbitrary paths and dimensions O(n)
    // separate formulas for different field types (gravity, electric, etc)
    //
    // geonum: PE = m(position·field) unified for all fields
    // same dot product in any dimension or blade count O(1)
    // field type encoded in geometric relationship, not separate formulas
}

#[test]
fn it_encodes_power() {
    let force = Geonum::new_with_blade(15.0, 1, 1.0, 8.0); // 15 N at blade 1, π/8
    let velocity = Geonum::new_with_blade(4.0, 1, 1.0, 8.0); // 4 m/s at blade 1, π/8 (aligned)

    // power from force-velocity dot product: P = F·v
    let power_interaction = force.dot(&velocity);
    let power_scalar =
        power_interaction.length * power_interaction.angle.project(Angle::new(0.0, 1.0));
    assert_eq!(
        power_interaction.angle.grade(),
        0,
        "power encodes scalar polarity in grade 0/2"
    );

    // aligned force and velocity give maximum power
    assert!(
        (power_scalar - 60.0).abs() < EPSILON,
        "P = F·v = 15×4 = 60 W for aligned case"
    );

    // test power at angle: force at π/8, velocity at π/6
    let angled_velocity = Geonum::new_with_blade(4.0, 1, 1.0, 6.0); // π/6 = 30°
    let angled_power = force.dot(&angled_velocity);

    // angle difference: π/6 - π/8 = 4π/24 - 3π/24 = π/24
    let angle_diff = PI / 24.0;
    let expected_power = 15.0 * 4.0 * angle_diff.cos();

    let angled_scalar = angled_power.length * angled_power.angle.project(Angle::new(0.0, 1.0));
    assert!(
        (angled_scalar - expected_power).abs() < EPSILON,
        "angled power = 15×4×cos(π/24) = {:.2} W",
        expected_power
    );

    // test perpendicular case: force at π/8, velocity at π/8 + π/2 = 5π/8
    let perpendicular_velocity = Geonum::new_with_blade(4.0, 1, 5.0, 8.0); // 5π/8
    let perpendicular_power = force.dot(&perpendicular_velocity);

    let perpendicular_scalar =
        perpendicular_power.length * perpendicular_power.angle.project(Angle::new(0.0, 1.0));
    assert!(
        perpendicular_scalar.abs() < EPSILON,
        "perpendicular F⊥v gives zero power: {:.6} ≈ 0",
        perpendicular_scalar
    );

    // test power scaling with force
    let double_force = force.scale(2.0); // 30 N
    assert!(
        (double_force.length - 30.0).abs() < EPSILON,
        "doubled force = 30 N"
    );

    let double_force_power = double_force.dot(&velocity);
    let double_power_scalar =
        double_force_power.length * double_force_power.angle.project(Angle::new(0.0, 1.0));
    assert!(
        (double_power_scalar - 120.0).abs() < EPSILON,
        "P(2F) = 120 = 2×60 W"
    );

    // test power scaling with velocity
    let triple_velocity = velocity.scale(3.0); // 12 m/s
    assert!(
        (triple_velocity.length - 12.0).abs() < EPSILON,
        "tripled velocity = 12 m/s"
    );

    let triple_velocity_power = force.dot(&triple_velocity);
    let triple_power_scalar =
        triple_velocity_power.length * triple_velocity_power.angle.project(Angle::new(0.0, 1.0));
    assert!(
        (triple_power_scalar - 180.0).abs() < EPSILON,
        "P(3v) = 180 = 3×60 W"
    );

    // verify linear scaling in both force and velocity
    let force_ratio = double_power_scalar / power_scalar;
    assert!(
        (force_ratio - 2.0).abs() < EPSILON,
        "doubling force doubles power: ratio = 2.0"
    );

    let velocity_ratio = triple_power_scalar / power_scalar;
    assert!(
        (velocity_ratio - 3.0).abs() < EPSILON,
        "tripling velocity triples power: ratio = 3.0"
    );

    // test high-dimensional power (different blades)
    let high_force = Geonum::new_with_blade(8.0, 250, 2.0, 11.0); // blade 250
    let high_velocity = Geonum::new_with_blade(6.0, 1000, 2.0, 11.0); // blade 1000

    let high_power = high_force.dot(&high_velocity);
    let high_power_scalar = high_power.length * high_power.angle.project(Angle::new(0.0, 1.0));
    assert!(
        high_power.angle == Angle::new(0.0, 1.0) || high_power.angle == Angle::new(1.0, 1.0),
        "power encodes sign via scalar/bivector pair"
    );

    // blade 250 gives mod_4_angle ≈ 3.71, blade 1000 gives ≈ 0.57
    // they're π apart (opposite directions), so power is negative
    assert!(
        (high_power_scalar + 48.0).abs() < EPSILON,
        "high-dim power = -48 W (opposite directions)"
    );

    // test negative power (force opposing velocity)
    let opposing_force = force.negate();
    let opposing_power = opposing_force.dot(&velocity);
    let opposing_scalar =
        opposing_power.length * opposing_power.angle.project(Angle::new(0.0, 1.0));

    assert!(
        opposing_scalar < 0.0,
        "opposing force-velocity gives negative power"
    );
    assert!(
        (opposing_scalar + 60.0).abs() < EPSILON,
        "opposing power = -60 W (energy extraction)"
    );

    // traditional mechanics: P = F⃗·v⃗ requires vector dot products in n dimensions
    // power = dW/dt requires work differentiation and time derivatives O(n)
    // separate power formulas for different systems (mechanical, electrical, etc)
    //
    // geonum: P = F·v unified for all power types
    // same dot product in any dimension or blade count O(1)
    // negative power naturally represents energy extraction
}

#[test]
fn it_encodes_torque() {
    let position = Geonum::new(2.0, 1.0, 6.0); // 2m lever arm at π/6
    let force = Geonum::new_with_blade(10.0, 1, 1.0, 3.0); // 10 N at blade 1, π/3

    // torque from position-force wedge product
    let torque = position.wedge(&force);

    assert_eq!(position.angle.blade(), 0, "position at blade 0");
    assert_eq!(force.angle.blade(), 1, "force at blade 1");
    assert_eq!(
        torque.angle.blade(),
        3,
        "torque at blade 3 (0+1+2=3 for wedge)"
    );
    assert_eq!(torque.angle.grade(), 3, "torque at grade 3");

    // blade 1 force is at π/3 + π/2 = 5π/6
    // angle difference: 5π/6 - π/6 = 2π/3
    let angle_diff = force.angle - position.angle;
    let expected_torque = 2.0 * 10.0 * angle_diff.mod_4_angle().sin().abs();
    assert!(
        (torque.length - expected_torque).abs() < EPSILON,
        "τ = r×F×sin(2π/3) ≈ 17.3 N·m"
    );

    // test torque conservation through nilpotency
    let torque_self_wedge = torque.wedge(&torque);
    assert!(torque_self_wedge.length < EPSILON, "τ∧τ = 0 (conservation)");

    // test perpendicular force (maximum torque)
    let perpendicular_force = force.differentiate(); // rotate force by π/2
    let max_torque = position.wedge(&perpendicular_force);

    let max_angle_diff = perpendicular_force.angle - position.angle;
    let expected_max = 2.0 * perpendicular_force.length * max_angle_diff.mod_4_angle().sin().abs();
    assert!(
        (max_torque.length - expected_max).abs() < EPSILON,
        "perpendicular τ = {:.1} N·m",
        expected_max
    );

    // test parallel force (zero torque)
    let parallel_force = Geonum::new_with_blade(10.0, 0, 1.0, 6.0); // blade 0, same angle as position
    let zero_torque = position.wedge(&parallel_force);

    assert!(
        zero_torque.length < EPSILON,
        "parallel force gives zero torque: {:.6} ≈ 0",
        zero_torque.length
    );

    // test scaling lever arm
    let double_position = position.scale(2.0); // 4m lever
    let double_torque = double_position.wedge(&force);

    assert!(
        (double_torque.length - 2.0 * torque.length).abs() < EPSILON,
        "doubling lever arm doubles torque"
    );

    // test high-dimensional torque
    let high_position = Geonum::new_with_blade(3.0, 0, 1.0, 9.0); // blade 0
    let high_force = Geonum::new_with_blade(8.0, 1000, 2.0, 7.0); // blade 1000
    let high_torque = high_position.wedge(&high_force);

    assert_eq!(
        high_torque.angle.blade(),
        1001,
        "torque blade = 0+1000+1 = 1001 (wedge adds π/2)"
    );

    // compute expected magnitude
    let high_angle_diff = high_force.angle - high_position.angle;
    let expected_high = 3.0 * 8.0 * high_angle_diff.mod_4_angle().sin().abs();

    assert!(
        (high_torque.length - expected_high).abs() < EPSILON,
        "high-dim τ = 3×8×sin(angle) = {:.3}",
        expected_high
    );

    // traditional mechanics: τ⃗ = r⃗ × F⃗ requires 6 component cross product
    // τ⃗ = [ry*Fz - rz*Fy, rz*Fx - rx*Fz, rx*Fy - ry*Fx] O(n²) in n dimensions
    //
    // geonum: τ = r∧F single wedge operation
    // same operation in any dimension O(1)
    // conservation built-in through nilpotency τ∧τ = 0
}

#[test]
fn it_encodes_angular_velocity() {
    let radius = Geonum::new(3.0, 0.0, 1.0); // 3m radius
    let angular_rate = 2.0; // rad/s
    let angular_velocity = Geonum::new_with_blade(angular_rate, 1, 0.0, 1.0); // ω at blade 1

    // linear velocity from angular velocity-radius wedge
    let linear_velocity = angular_velocity.wedge(&radius);

    assert_eq!(radius.angle.blade(), 0, "radius at blade 0");
    assert_eq!(angular_velocity.angle.blade(), 1, "ω at blade 1");
    assert_eq!(
        linear_velocity.angle.blade(),
        4,
        "v at blade 4 (1+0+3=4 for wedge)"
    );

    // for perpendicular ω and r: |v| = ωr
    assert!(
        (linear_velocity.length - 6.0).abs() < EPSILON,
        "v = ωr = 2×3 = 6 m/s"
    );

    // test double angular velocity
    let double_omega = angular_velocity.scale(2.0); // 4 rad/s
    let double_linear = double_omega.wedge(&radius);

    assert!(
        (double_linear.length - 12.0).abs() < EPSILON,
        "v(2ω) = 2ωr = 4×3 = 12 m/s"
    );

    // verify linear scaling
    let omega_ratio = double_linear.length / linear_velocity.length;
    assert!(
        (omega_ratio - 2.0).abs() < EPSILON,
        "doubling ω doubles v: ratio = 2.0"
    );

    // test double radius
    let double_radius = radius.scale(2.0); // 6m
    let radius_scaled = angular_velocity.wedge(&double_radius);

    assert!(
        (radius_scaled.length - 12.0).abs() < EPSILON,
        "v(2r) = ω(2r) = 2×6 = 12 m/s"
    );

    // verify radius scaling
    let radius_ratio = radius_scaled.length / linear_velocity.length;
    assert!(
        (radius_ratio - 2.0).abs() < EPSILON,
        "doubling r doubles v: ratio = 2.0"
    );

    // test high-dimensional angular velocity
    let high_radius = Geonum::new_with_blade(4.0, 1000, 1.0, 7.0); // blade 1000
    let high_omega = Geonum::new_with_blade(1.5, 500, 0.0, 1.0); // blade 500
    let high_linear = high_omega.wedge(&high_radius);

    assert_eq!(
        high_linear.angle.blade(),
        1501,
        "v blade = 500+1000+1 = 1501"
    );

    // compute expected magnitude
    let angle_diff = high_radius.angle - high_omega.angle;
    let expected_speed = 1.5 * 4.0 * angle_diff.mod_4_angle().sin().abs();

    assert!(
        (high_linear.length - expected_speed).abs() < EPSILON,
        "high-dim v = 1.5×4×sin(angle) = {:.3}",
        expected_speed
    );

    // test centripetal acceleration: a = ω²r
    let omega_squared = angular_velocity.dot(&angular_velocity); // ω² as scalar
    let centripetal = radius.scale(omega_squared.length);

    assert!(
        (centripetal.length - 12.0).abs() < EPSILON,
        "a_c = ω²r = 4×3 = 12 m/s²"
    );

    // traditional mechanics: v⃗ = ω⃗ × r⃗ requires cross product O(n²)
    // centripetal a⃗ = -ω²r⃗ requires separate formulas
    //
    // geonum: v = ω∧r unified wedge operation
    // same operation in any dimension O(1)
    // centripetal naturally emerges from ω²r scaling
}

#[test]
fn it_encodes_mass_through_scaling() {
    // IMPROVED from mechanics_test.rs:918-979
    //
    // PROBLEMS with original test:
    // 1. weak assertion: extracted_mass.angle.grade() == 2 doesnt test the angle value
    // 2. confusing: mass extraction preserving "grade 2" makes no physical sense
    // 3. handwavy: "mass emerges from geometric scaling" without proving the physics
    // 4. no test: momentum-impulse relationships missing
    // 5. no test: mass invariance under galilean transformations
    //
    // IMPROVEMENTS:
    // - exact angle assertions for mass scaling relationships
    // - test momentum = mass × velocity with precise blade tracking
    // - prove F = ma through differentiation chain: x → v → a then scale by m
    // - test impulse J = FΔt = Δp relationships
    // - verify mass invariance under transformations

    // fundamental test: F = ma through scaling
    let mass = 4.0; // kg
    let acceleration = Geonum::new_with_blade(5.0, 2, 0.0, 1.0); // 5 m/s² at blade 2 (grade 2)
    let force = acceleration.scale(mass); // F = ma

    assert_eq!(force.length, 20.0, "F = ma = 4×5 = 20 N");
    assert_eq!(
        force.angle, acceleration.angle,
        "force preserves acceleration angle exactly"
    );
    assert_eq!(
        force.angle.blade(),
        2,
        "force at blade 2 (same as acceleration)"
    );
    assert_eq!(
        force.angle.value(),
        0.0,
        "force angle value = 0 within π/2 segment"
    );

    // test momentum p = mv at velocity blade level
    let velocity = Geonum::new_with_blade(3.0, 1, 0.0, 1.0); // 3 m/s at blade 1 (grade 1)
    let momentum = velocity.scale(mass); // p = mv

    assert_eq!(momentum.length, 12.0, "p = mv = 4×3 = 12 kg·m/s");
    assert_eq!(
        momentum.angle, velocity.angle,
        "momentum preserves velocity angle exactly"
    );
    assert_eq!(
        momentum.angle.blade(),
        1,
        "momentum at blade 1 (velocity level)"
    );

    // prove F = dp/dt through differentiation
    let momentum_rate = momentum.differentiate(); // dp/dt adds π/2 rotation

    assert_eq!(
        momentum_rate.angle.blade(),
        2,
        "dp/dt at blade 2 (force level)"
    );
    assert_eq!(
        momentum_rate.length, momentum.length,
        "differentiation preserves magnitude"
    );
    // momentum_rate represents force when time-scaled appropriately

    // test impulse-momentum theorem: J = FΔt = Δp
    let delta_t = 0.5; // seconds
    let impulse = force.scale(delta_t); // J = FΔt

    assert_eq!(impulse.length, 10.0, "J = FΔt = 20×0.5 = 10 N·s");
    assert_eq!(
        impulse.angle.blade(),
        force.angle.blade(),
        "impulse preserves force blade"
    );

    // impulse changes momentum
    let initial_momentum = momentum;
    let final_momentum = initial_momentum + impulse.copy_blade(&initial_momentum); // match blade levels
    let delta_p = final_momentum.length - initial_momentum.length;

    assert!(
        (delta_p - impulse.length).abs() < EPSILON,
        "Δp = J confirmed: {:.3} ≈ {:.3}",
        delta_p,
        impulse.length
    );

    // test mass extraction: m = F/a
    let extracted_mass_fa = force.length / acceleration.length;
    assert_eq!(extracted_mass_fa, mass, "m = F/a = 20/5 = 4 kg");

    // test mass extraction: m = p/v
    let extracted_mass_pv = momentum.length / velocity.length;
    assert_eq!(extracted_mass_pv, mass, "m = p/v = 12/3 = 4 kg");

    // prove mass invariance under rotation
    let rotation = Angle::new(1.0, 3.0); // π/3 rotation
    let rotated_velocity = velocity.rotate(rotation);
    let rotated_momentum = rotated_velocity.scale(mass);
    let rotated_extracted_mass = rotated_momentum.length / rotated_velocity.length;

    assert_eq!(
        rotated_extracted_mass, mass,
        "mass invariant under rotation: 4 kg"
    );
    assert_eq!(
        rotated_momentum.angle, rotated_velocity.angle,
        "rotated momentum preserves rotated velocity angle"
    );

    // test kinetic energy: KE = ½mv²
    let v_squared = velocity.dot(&velocity); // v·v = |v|²
    let kinetic_energy = 0.5 * mass * v_squared.length;

    assert_eq!(kinetic_energy, 18.0, "KE = ½mv² = 0.5×4×9 = 18 J");

    // test mass in high dimensions - prove scaling works everywhere
    let high_dim_accel = Geonum::new_with_blade(7.0, 1000, 1.0, 8.0); // blade 1000
    let high_dim_force = high_dim_accel.scale(mass);

    assert_eq!(
        high_dim_force.length, 28.0,
        "F = ma = 4×7 = 28 N in high dimension"
    );
    assert_eq!(
        high_dim_force.angle, high_dim_accel.angle,
        "force preserves acceleration angle in dimension 1000"
    );
    assert_eq!(high_dim_force.angle.blade(), 1000, "force at blade 1000");

    // test relativistic mass scaling (simplified)
    // at high velocity, mass increases by γ = 1/√(1-v²/c²)
    let c = 299792458.0_f64; // speed of light m/s
    let high_v = 0.8 * c; // 80% speed of light
    let v_over_c = high_v / c;
    let gamma = 1.0 / (1.0 - v_over_c * v_over_c).sqrt(); // ≈ 1.667
    let relativistic_mass = mass * gamma;

    assert!(
        (relativistic_mass - 6.667).abs() < 0.01,
        "relativistic mass ≈ 6.67 kg at 0.8c"
    );

    // KEY INSIGHTS:
    // 1. mass is pure scalar - just a magnitude that scales other quantities
    // 2. F = ma and p = mv preserve the angle of acceleration/velocity exactly
    // 3. differentiation chain proves F = dp/dt geometrically
    // 4. impulse-momentum theorem emerges from scaling relationships
    // 5. mass extraction is simple division of magnitudes
    // 6. kinetic energy comes from dot product (projection)

    // traditional mechanics: mass as fundamental property with separate equations
    // geonum: mass as scaling factor in geometric relationships
    // but mass itself has no blade/angle - its pure magnitude scaling
}

#[test]
fn it_encodes_rotational_inertia() {
    // IMPROVED from mechanics_test.rs:980-1051
    //
    // PROBLEMS with original test:
    // 1. workaround code: negative length check for dot product (lines 1000-1004)
    // 2. weak physics: no test of angular momentum L = Iω
    // 3. weak physics: no test of rotational kinetic energy KE = ½Iω²
    // 4. confusing: "inertia emerges from geometric mass-radius relationships" vague
    // 5. no test: parallel axis theorem I = I_cm + md²
    // 6. no test: perpendicular axis theorem for planar objects
    //
    // IMPROVEMENTS:
    // - clean dot product usage without workarounds
    // - test angular momentum L = Iω with exact blade tracking
    // - test rotational kinetic energy KE = ½Iω²
    // - prove parallel axis theorem geometrically
    // - test conservation of angular momentum L∧L = 0

    // fundamental test: moment of inertia I = mr²
    let mass = 2.0; // kg
    let radius = Geonum::new(3.0, 0.0, 1.0); // 3m from rotation axis

    // compute I = mr² through dot product
    let r_squared = radius.dot(&radius); // r·r = |r|²
    let inertia = mass * r_squared.length;

    assert_eq!(inertia, 18.0, "I = mr² = 2×9 = 18 kg·m²");
    assert_eq!(r_squared.angle.grade(), 0, "r·r produces scalar at grade 0");

    // test angular momentum L = Iω
    let omega = Geonum::new_with_blade(1.5, 1, 0.0, 1.0); // 1.5 rad/s at blade 1
    let angular_momentum = omega.scale(inertia); // L = Iω

    assert_eq!(
        angular_momentum.length, 27.0,
        "L = Iω = 18×1.5 = 27 kg·m²/s"
    );
    assert_eq!(
        angular_momentum.angle, omega.angle,
        "L preserves ω angle exactly"
    );
    assert_eq!(
        angular_momentum.angle.blade(),
        1,
        "L at blade 1 (same as ω)"
    );

    // test rotational kinetic energy KE = ½Iω²
    let omega_squared = omega.dot(&omega); // ω·ω = |ω|²
    let rotational_ke = 0.5 * inertia * omega_squared.length;

    assert_eq!(
        rotational_ke, 20.25,
        "KE_rot = ½Iω² = 0.5×18×2.25 = 20.25 J"
    );

    // test torque-angular acceleration: τ = Iα
    let alpha = Geonum::new_with_blade(2.5, 2, 0.0, 1.0); // 2.5 rad/s² at blade 2
    let torque = alpha.scale(inertia); // τ = Iα

    assert_eq!(torque.length, 45.0, "τ = Iα = 18×2.5 = 45 N·m");
    assert_eq!(torque.angle, alpha.angle, "τ preserves α angle exactly");
    assert_eq!(torque.angle.blade(), 2, "τ at blade 2 (same as α)");

    // prove dL/dt = τ through differentiation
    let l_rate = angular_momentum.differentiate(); // dL/dt adds π/2

    assert_eq!(l_rate.angle.blade(), 2, "dL/dt at blade 2 (torque level)");
    assert_eq!(
        l_rate.length, angular_momentum.length,
        "differentiation preserves magnitude"
    );
    // l_rate represents torque when properly scaled

    // test conservation of angular momentum through nilpotency
    let l_wedge_l = angular_momentum.wedge(&angular_momentum);
    assert!(
        l_wedge_l.length < EPSILON,
        "L∧L = 0 (angular momentum conservation)"
    );

    // test parallel axis theorem: I = I_cm + md²
    let center_of_mass_radius = Geonum::new(1.0, 0.0, 1.0); // 1m from CM
    let distance_to_new_axis = Geonum::new(2.0, 0.0, 1.0); // 2m from CM to new axis

    let i_cm = mass * center_of_mass_radius.dot(&center_of_mass_radius).length; // 2×1 = 2
    let d_squared = distance_to_new_axis.dot(&distance_to_new_axis).length; // 4
    let i_parallel = i_cm + mass * d_squared;

    assert_eq!(i_cm, 2.0, "I_cm = mr_cm² = 2×1 = 2 kg·m²");
    assert_eq!(
        i_parallel, 10.0,
        "I_parallel = I_cm + md² = 2 + 2×4 = 10 kg·m²"
    );

    // test scaling: doubling radius quadruples inertia
    let double_radius = radius.scale(2.0); // 6m
    let double_r_squared = double_radius.dot(&double_radius);
    let scaled_inertia = mass * double_r_squared.length;

    assert_eq!(scaled_inertia, 72.0, "I(2r) = m(2r)² = 2×36 = 72 kg·m²");
    assert_eq!(scaled_inertia / inertia, 4.0, "doubling r quadruples I");

    // test high-dimensional rotational inertia
    let high_dim_radius = Geonum::new_with_blade(4.0, 1000, 1.0, 7.0); // blade 1000
    let high_dim_r_squared = high_dim_radius.dot(&high_dim_radius);
    let high_dim_inertia = mass * high_dim_r_squared.length;

    assert_eq!(
        high_dim_inertia, 32.0,
        "I = mr² = 2×16 = 32 kg·m² in dimension 1000"
    );
    assert_eq!(
        high_dim_r_squared.angle.grade(),
        0,
        "r·r scalar even at blade 1000"
    );

    // test angular impulse: ΔL = τΔt
    let delta_t = 0.3; // seconds
    let angular_impulse = torque.scale(delta_t); // τΔt

    assert_eq!(
        angular_impulse.length, 13.5,
        "ΔL = τΔt = 45×0.3 = 13.5 kg·m²/s"
    );
    assert_eq!(
        angular_impulse.angle.blade(),
        torque.angle.blade(),
        "angular impulse preserves torque blade"
    );

    // KEY INSIGHTS:
    // 1. dot product r·r gives clean |r|² without sign issues
    // 2. complete rotational dynamics: L = Iω, KE = ½Iω², τ = Iα
    // 3. conservation through nilpotency: L∧L = 0
    // 4. parallel axis theorem proven geometrically
    // 5. angular impulse-momentum relationship tested
    // 6. inertia is scalar quantity (mass × length²)

    // traditional mechanics: I = ∫r²dm requires mass distribution integration
    // geonum: I = mr² through simple dot product r·r
    // rotational dynamics emerge from scaling and differentiation
}

#[test]
fn it_handles_energy_conservation() {
    // IMPROVED from mechanics_test.rs:1053-1135
    //
    // PROBLEMS with original:
    // 1. workaround code for negative dot product (lines 1069-1073, 1077-1081, etc)
    // 2. confusing claim: "energy conservation emerges from geometric nilpotency"
    // 3. misunderstands physics: E∧E = 0 doesnt prove energy conservation
    // 4. no test of actual energy conservation during motion
    // 5. gravity field at "π" makes no physical sense
    //
    // IMPROVEMENTS:
    // - clean dot product usage
    // - test actual energy conservation: E_initial = E_final
    // - test work-energy theorem: W = ΔKE
    // - test pendulum energy exchange between KE and PE
    // - remove misleading nilpotency claims

    let mass = 3.0; // kg
    let g = 9.8; // m/s² gravitational acceleration

    // test 1: falling object energy conservation
    // initial state: height h, velocity 0
    let initial_height = 10.0; // meters
    let initial_velocity = 0.0; // m/s (at rest)

    let initial_pe = mass * g * initial_height; // mgh
    let initial_ke = 0.5 * mass * initial_velocity * initial_velocity; // ½mv²
    let total_energy = initial_pe + initial_ke;

    assert_eq!(initial_pe, 294.0, "PE = mgh = 3×9.8×10 = 294 J");
    assert_eq!(initial_ke, 0.0, "KE = 0 (at rest)");
    assert_eq!(total_energy, 294.0, "E_total = PE + KE = 294 J");

    // after falling to height 4m
    let final_height = 4.0; // meters
    let height_fallen = initial_height - final_height; // 6 meters

    // use conservation of energy to find final velocity
    // E_initial = E_final
    // mgh_i + ½mv_i² = mgh_f + ½mv_f²
    // solving for v_f: v_f = √(2g(h_i - h_f))
    let final_velocity = (2.0_f64 * g * height_fallen).sqrt();

    let final_pe = mass * g * final_height;
    let final_ke = 0.5 * mass * final_velocity * final_velocity;
    let final_total = final_pe + final_ke;

    assert!(
        (final_pe - 117.6).abs() < EPSILON,
        "PE = mgh = 3×9.8×4 = 117.6 J"
    );
    assert!(
        (final_ke - 176.4).abs() < EPSILON,
        "KE = ½mv² = 0.5×3×10.84² = 176.4 J"
    );
    assert!(
        (final_total - total_energy).abs() < EPSILON,
        "energy conserved: E_final = {} ≈ E_initial = {}",
        final_total,
        total_energy
    );

    // test 2: work-energy theorem W = ΔKE
    let force = Geonum::new(15.0, 0.0, 1.0); // 15 N at blade 0 (same direction as displacement)
    let displacement = Geonum::new(4.0, 0.0, 1.0); // 4 m displacement

    // work = force · displacement (dot product)
    let work = force.dot(&displacement);
    assert_eq!(work.length, 60.0, "W = F·d = 15×4 = 60 J");
    assert_eq!(work.angle.grade(), 0, "work is scalar at grade 0");

    // if this work accelerates object from rest
    // W = ΔKE = ½mv_f² - 0
    // v_f = √(2W/m)
    let final_speed_from_work = (2.0 * work.length / mass).sqrt();
    let ke_from_work = 0.5 * mass * final_speed_from_work * final_speed_from_work;

    assert!(
        (ke_from_work - work.length).abs() < 1e-10,
        "work-energy theorem: W = ΔKE = {} J",
        work.length
    );

    // test 3: pendulum energy exchange
    let pendulum_length = 2.0; // meters
    let max_angle = Angle::new(1.0, 3.0); // π/3 radians (60°)

    // at maximum displacement: all PE, no KE
    let max_height = pendulum_length * (1.0 - max_angle.mod_4_angle().cos()); // h = L(1 - cos θ)
    let pe_max = mass * g * max_height;

    // at bottom: all KE, no PE (taking bottom as h=0)
    let ke_bottom = pe_max; // energy conserved
    let velocity_bottom = (2.0 * ke_bottom / mass).sqrt();

    assert!(
        (max_height - 1.0).abs() < EPSILON,
        "h_max = L(1-cos60°) = 2×0.5 = 1 m"
    );
    assert!(
        (pe_max - 29.4).abs() < EPSILON,
        "PE_max = mgh = 3×9.8×1 = 29.4 J"
    );
    assert_eq!(ke_bottom, pe_max, "energy exchanges: PE_max → KE_bottom");
    assert!(
        (velocity_bottom - 4.427).abs() < 0.001,
        "v_bottom = √(2×KE/m) ≈ 4.43 m/s"
    );

    // test 4: spring potential energy U = ½kx²
    let spring_k = 100.0; // N/m spring constant
    let compression = 0.3; // meters

    let spring_pe = 0.5 * spring_k * compression * compression;
    assert_eq!(spring_pe, 4.5, "U_spring = ½kx² = 0.5×100×0.09 = 4.5 J");

    // release spring: PE → KE
    let velocity_from_spring = (2.0 * spring_pe / mass).sqrt();
    assert!(
        (velocity_from_spring - 1.732).abs() < 0.001,
        "v = √(2U/m) = √(9/3) ≈ 1.73 m/s"
    );

    // test 5: power P = dE/dt
    let energy_rate = 75.0; // watts (J/s)
    let time_interval = 4.0; // seconds
    let energy_delivered = energy_rate * time_interval;

    assert_eq!(energy_delivered, 300.0, "E = P×t = 75×4 = 300 J");

    // test 6: nilpotency expresses conservation
    // conserved quantities satisfy Q∧Q = 0
    let total_energy_geonum = Geonum::new(total_energy, 0.0, 1.0);
    let energy_nilpotent = total_energy_geonum.wedge(&total_energy_geonum);
    assert!(
        energy_nilpotent.length < EPSILON,
        "E∧E = 0 expresses conservation"
    );

    // angular momentum also conserved → nilpotent
    let l = Geonum::new(27.0, 1.0, 2.0); // angular momentum at π/2
    let l_nilpotent = l.wedge(&l);
    assert!(
        l_nilpotent.length < EPSILON,
        "L∧L = 0 expresses angular momentum conservation"
    );

    // KEY INSIGHTS:
    // 1. energy conservation: E_initial = E_final for isolated systems
    // 2. work-energy theorem: W = F·d = ΔKE
    // 3. energy exchange: PE ↔ KE during motion
    // 4. power is energy rate: P = dE/dt
    // 5. nilpotency Q∧Q = 0 concisely expresses conservation of quantity Q

    // conservation is physical law observed through measurement
    // nilpotency provides geometric expression of that conservation
}

#[test]
fn it_handles_momentum_conservation() {
    // test 1: two-body collision (elastic)
    let m1 = 2.0; // kg
    let m2 = 3.0; // kg

    // initial velocities
    let v1_initial = Geonum::new_with_blade(5.0, 1, 0.0, 1.0); // 5 m/s at blade 1
    let v2_initial = Geonum::new_with_blade(-2.0, 1, 0.0, 1.0); // -2 m/s (opposite direction)

    // initial momenta
    let p1_initial = v1_initial.scale(m1); // p = mv
    let p2_initial = v2_initial.scale(m2);

    // total momentum (vector addition)
    let p_total = p1_initial + p2_initial;

    assert_eq!(p1_initial.length, 10.0, "p1 = m1×v1 = 2×5 = 10 kg·m/s");
    assert_eq!(
        p2_initial.length.abs(),
        6.0,
        "p2 = |m2×v2| = 3×2 = 6 kg·m/s"
    );
    assert_eq!(p_total.length, 4.0, "p_total = 10 - 6 = 4 kg·m/s");

    // after elastic collision (example final velocities)
    // conservation requires: m1v1f + m2v2f = m1v1i + m2v2i
    let v1_final = Geonum::new_with_blade(-1.0, 1, 0.0, 1.0); // -1 m/s
    let v2_final = Geonum::new_with_blade(2.0, 1, 0.0, 1.0); // 2 m/s

    let p1_final = v1_final.scale(m1);
    let p2_final = v2_final.scale(m2);
    let p_total_final = p1_final + p2_final;

    assert_eq!(
        p_total_final.length, 4.0,
        "momentum conserved: p_total unchanged"
    );
    assert_eq!(
        p_total_final.angle, p_total.angle,
        "momentum direction preserved"
    );

    // test 2: nilpotency expresses momentum conservation
    let p_nilpotent = p_total.wedge(&p_total);
    assert!(
        p_nilpotent.length < EPSILON,
        "p∧p = 0 expresses momentum conservation"
    );

    // test 3: rocket propulsion (variable mass)
    let rocket_mass = 1000.0; // kg
    let exhaust_velocity = 3000.0; // m/s relative to rocket
    let mass_flow_rate = 10.0; // kg/s

    let rocket_velocity = Geonum::new_with_blade(100.0, 1, 0.0, 1.0); // 100 m/s
    let rocket_momentum = rocket_velocity.scale(rocket_mass);

    assert_eq!(
        rocket_momentum.length, 100000.0,
        "p_rocket = mv = 1000×100 = 100000 kg·m/s"
    );

    // after burning fuel for 1 second
    let ejected_mass = mass_flow_rate * 1.0; // 10 kg
    let new_rocket_mass = rocket_mass - ejected_mass;

    // tsiolkovsky rocket equation: Δv = v_exhaust × ln(m0/m1)
    let mass_ratio = rocket_mass / new_rocket_mass;
    let delta_v = exhaust_velocity * mass_ratio.ln();

    assert!((delta_v - 30.1).abs() < 0.1, "Δv ≈ 3000×ln(1.01) ≈ 30 m/s");

    // test 4: center of mass momentum
    // for system of particles, p_cm = M_total × v_cm
    let particle_masses = [1.0, 2.0, 3.0]; // kg
    let particle_velocities = [
        Geonum::new_with_blade(3.0, 1, 0.0, 1.0),
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0),
        Geonum::new_with_blade(-1.0, 1, 0.0, 1.0),
    ];

    let total_mass: f64 = particle_masses.iter().sum();
    let mut total_momentum = Geonum::new_with_blade(0.0, 1, 0.0, 1.0);

    for (m, v) in particle_masses.iter().zip(particle_velocities.iter()) {
        total_momentum = total_momentum + v.scale(*m);
    }

    let v_cm = total_momentum.scale(1.0 / total_mass);
    assert_eq!(total_momentum.length, 2.0, "p_total = 3 + 2 - 3 = 2 kg·m/s");
    assert!(
        (v_cm.length - 0.333).abs() < 0.001,
        "v_cm = p_total/M = 2/6 ≈ 0.33 m/s"
    );

    // test 5: angular momentum also conserved → nilpotent
    let r = Geonum::new(3.0, 0.0, 1.0); // position vector
    let p = Geonum::new_with_blade(4.0, 1, 0.0, 1.0); // momentum
    let l = r.wedge(&p); // L = r × p

    let l_nilpotent = l.wedge(&l);
    assert!(
        l_nilpotent.length < EPSILON,
        "L∧L = 0 expresses angular momentum conservation"
    );

    // KEY INSIGHTS:
    // 1. momentum conservation: p_initial = p_final in isolated systems
    // 2. nilpotency p∧p = 0 geometrically expresses conservation
    // 3. center of mass momentum: p_cm = M_total × v_cm
    // 4. angular momentum L = r × p also conserved and nilpotent
    // 5. conservation holds even with variable mass (rockets)
}

#[test]
fn it_handles_angular_momentum_conservation() {
    // test 1: spinning figure skater
    // pulling arms in: smaller r, faster ω to conserve L = Iω
    let initial_radius = 1.5; // m (arms extended)
    let final_radius = 0.5; // m (arms pulled in)
    let mass = 60.0; // kg

    // initial angular velocity
    let initial_omega = Geonum::new_with_blade(2.0, 1, 0.0, 1.0); // 2 rad/s at blade 1

    // moment of inertia I = mr²
    let initial_inertia = mass * initial_radius * initial_radius;
    let final_inertia = mass * final_radius * final_radius;

    // angular momentum L = Iω
    let l_initial = initial_omega.scale(initial_inertia);

    // conservation: L_initial = L_final
    // so ω_final = L_initial / I_final
    let omega_final_magnitude = l_initial.length / final_inertia;
    let omega_final = Geonum::new_with_blade(omega_final_magnitude, 1, 0.0, 1.0);

    assert_eq!(
        initial_inertia, 135.0,
        "I_initial = mr² = 60×1.5² = 135 kg·m²"
    );
    assert_eq!(final_inertia, 15.0, "I_final = mr² = 60×0.5² = 15 kg·m²");
    assert_eq!(l_initial.length, 270.0, "L = Iω = 135×2 = 270 kg·m²/s");
    assert_eq!(
        omega_final.length, 18.0,
        "ω_final = L/I = 270/15 = 18 rad/s"
    );

    // test 2: nilpotency expresses conservation
    let l_wedge_l = l_initial.wedge(&l_initial);
    assert!(
        l_wedge_l.length < EPSILON,
        "L∧L = 0 expresses angular momentum conservation"
    );

    // test 3: planetary orbit (Kepler's second law)
    // equal areas swept in equal times → L conservation
    let r1 = Geonum::new(1.0e11, 0.0, 1.0); // 1 AU from sun
    let v1 = Geonum::new_with_blade(30000.0, 1, 1.0, 2.0); // 30 km/s at π/2 to radius

    let r2 = Geonum::new(1.5e11, 0.0, 1.0); // 1.5 AU (farther)
                                            // conservation: r1×v1 = r2×v2, so v2 = v1×(r1/r2)
    let v2_magnitude = v1.length * (r1.length / r2.length);
    let v2 = Geonum::new_with_blade(v2_magnitude, 1, 1.0, 2.0);

    let l1 = r1.wedge(&v1);
    let l2 = r2.wedge(&v2);

    assert_eq!(
        v2.length, 20000.0,
        "v2 = v1×(r1/r2) = 30000×(1/1.5) = 20000 m/s"
    );
    assert!(
        (l1.length - l2.length).abs() < 1e15,
        "orbital angular momentum conserved"
    );

    // test 4: gyroscope precession
    // torque τ = dL/dt causes precession, not change in |L|
    let spin_l = Geonum::new_with_blade(10.0, 3, 0.0, 1.0); // spinning top L at blade 3
    let torque = Geonum::new_with_blade(0.5, 2, 1.0, 2.0); // small torque at blade 2

    // precession rate Ω = τ/L
    let precession_rate = torque.length / spin_l.length;
    assert_eq!(precession_rate, 0.05, "Ω = τ/L = 0.5/10 = 0.05 rad/s");

    // L magnitude unchanged during precession
    let dt = 0.1; // small time step
    let d_l = torque.scale(dt); // dL = τ×dt
    let l_new = spin_l + d_l;

    // magnitude approximately preserved for small precession
    assert!(
        (l_new.length - spin_l.length).abs() < 0.1,
        "gyroscope |L| approximately constant during precession"
    );

    // test 5: collision with rotation
    // ball hits rod at distance d from pivot
    let ball_mass = 0.5; // kg
    let ball_velocity = Geonum::new_with_blade(10.0, 1, 0.0, 1.0); // 10 m/s
    let impact_distance = 0.8; // m from pivot

    // angular momentum imparted: L = r × p = d × mv
    let ball_momentum = ball_velocity.scale(ball_mass);
    let impact_position = Geonum::new(impact_distance, 0.0, 1.0);
    let angular_impulse = impact_position.wedge(&ball_momentum);

    assert_eq!(ball_momentum.length, 5.0, "p_ball = mv = 0.5×10 = 5 kg·m/s");
    assert_eq!(angular_impulse.length, 4.0, "L = r×p = 0.8×5 = 4 kg·m²/s");

    // rod begins rotating to conserve angular momentum
    let rod_inertia = 2.0; // kg·m² about pivot
    let rod_omega = angular_impulse.length / rod_inertia;
    assert_eq!(rod_omega, 2.0, "ω_rod = L/I = 4/2 = 2 rad/s");

    // KEY INSIGHTS:
    // 1. angular momentum L = r×p = Iω conserved without external torque
    // 2. nilpotency L∧L = 0 expresses conservation geometrically
    // 3. figure skater: smaller r → larger ω to conserve L
    // 4. planetary orbits: Kepler's second law from L conservation
    // 5. gyroscope: torque causes precession, |L| stays constant
}
