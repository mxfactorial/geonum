// mechanics in its native geometry: the kinematic hierarchy is grade cycling,
// the dynamic quantities are wedge and dot
//
// differentiation is a quarter turn, so position, velocity, acceleration, jerk are
// ONE object wound to grades 0, 1, 2, 3 — the derivative order is the blade, read
// mod 4. angular momentum, torque, and rotational velocity are wedges; kinetic
// energy, work, power, and moment of inertia are dots landing on grade 0. mass is a
// pure scalar — it scales the magnitude and touches no angle.
//
// the self-wedge a∧a = 0 is universal: sin(θ−θ) = 0 for every geonum, conserved or
// not. it witnesses that a state cannot repeat, not that a quantity is conserved.
// conservation is expressed by antisymmetry — the wedge — not searched for as a
// symmetry: dL/dt = v∧v + r∧a is a sum of wedges that vanish (self-wedge, parallel
// wedge), and internal forces cancel pairwise as F_ij = −F_ji. no Lagrangian, no
// symmetry hunt, cancellation read straight off the antisymmetric product.
// energy joins in act V: the conserved scalar is the phase point's magnitude,
// and dE/dt = 0 is the kinetic credit F·v cancelling the potential debit kx·ẋ.
//
// run: cargo test --test mechanics_test -- --show-output

use geonum::*;

const EPSILON: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════
// act I: the kinematic hierarchy is grade cycling
// ═══════════════════════════════════════════════════════════

#[test]
fn it_cycles_the_kinematic_hierarchy_through_grades() {
    // conventional mechanics stacks a separate vector equation and a finite-difference
    // scheme at each level — position, velocity, acceleration, jerk. here each is one
    // differentiate(), a quarter turn, and grade cycles 0→1→2→3→0 while the magnitude
    // rides through untouched

    let position = Geonum::new(10.0, 1.0, 3.0); // [10, π/3], grade 0
    let velocity = position.differentiate();
    let acceleration = velocity.differentiate();
    let jerk = acceleration.differentiate();
    let snap = jerk.differentiate();

    assert_eq!(position.angle.grade(), 0, "position grade 0");
    assert_eq!(velocity.angle.grade(), 1, "velocity grade 1");
    assert_eq!(acceleration.angle.grade(), 2, "acceleration grade 2");
    assert_eq!(jerk.angle.grade(), 3, "jerk grade 3");
    assert_eq!(snap.angle.grade(), 0, "snap back to grade 0");

    assert!(
        velocity.near_mag(10.0),
        "the quarter turn costs no magnitude"
    );
    assert!(snap.near_mag(10.0), "still 10 after a full cycle");
}

#[test]
fn it_carries_its_own_motion_in_the_quarter_turn() {
    // a position needs no separately-tracked velocity to move. differentiate() turns it a
    // quarter turn, and that IS the velocity the position is tangent to — same length,
    // perpendicular. because the length survives, |v| = |r| is the tangential speed of
    // circular motion at unit rate. scale by time for a displacement, add it, and one
    // position has stepped its own orbit: rotate, scale, add, no second initial condition
    // supplied. conventional kinematics carries r and v as independent vectors updated by
    // dr/dt; here they are one number at blade n and blade n+1

    let position = Geonum::new(10.0, 1.0, 3.0); // 10 m at π/3

    // the velocity is the position turned a quarter turn, magnitude intact
    let velocity = position.differentiate();
    assert_eq!(
        velocity.angle.grade(),
        1,
        "v is r's quarter turn, grade 0 → 1"
    );
    assert!(
        velocity.near_mag(position.mag),
        "|v| = |r|: the tangential speed of circular motion"
    );

    // displacement = v·t runs straight along the tangent — scale leaves the angle alone
    let dt = 2.0;
    let displacement = velocity.scale(dt);
    assert!(
        displacement.near_mag(20.0),
        "displacement = |v|·t = 10×2 = 20"
    );
    assert_eq!(
        displacement.angle, velocity.angle,
        "the step is along the tangent"
    );

    // the displacement is a quarter turn off the radius: a tangent step, one Euler step
    // of a circle
    assert!(
        (displacement.angle - position.angle).near(&Angle::new(1.0, 2.0)),
        "the step is perpendicular to the radius"
    );

    // add it — the new position is where the motion carried it, √(r² + d²) out
    let moved = position + displacement;
    let expected = (position.mag * position.mag + displacement.mag * displacement.mag).sqrt();
    assert!(
        moved.near_mag(expected),
        "the tangent step landed at √(r² + d²)"
    )
}

#[test]
fn it_grows_displacement_quadratically_from_derived_acceleration() {
    // the kinematic equation d = v₀t + ½at² wants double integration and a Taylor
    // expansion conventionally. here both terms are scalings of the derivative hierarchy:
    // v₀t scales the velocity by t, ½at² scales the acceleration by ½t². the quadratic
    // term outruns the linear as t grows — the ½t² overtaking the t

    let position = Geonum::new(5.0, 1.0, 4.0); // 5 m at π/4
    let velocity = position.differentiate(); // grade 1, |v| = 5
    let acceleration = velocity.differentiate(); // grade 2, |a| = 5

    let t = 3.0;
    let linear = velocity.scale(t); // v₀t
    let quadratic = acceleration.scale(0.5 * t * t); // ½at²
    assert!(linear.near_mag(15.0), "v₀t = 5×3 = 15");
    assert!(quadratic.near_mag(22.5), "½at² = ½×5×9 = 22.5");

    // velocity and acceleration are a quarter turn apart, so the two displacements are
    // perpendicular and the total travel is √((v₀t)² + (½at²)²)
    let displacement = linear + quadratic;
    let expected = (15.0_f64 * 15.0 + 22.5 * 22.5).sqrt();
    assert!(
        displacement.near_mag(expected),
        "total = √((v₀t)² + (½at²)²)"
    );

    // for equal |v₀| and |a|, the quadratic overtakes the linear at t = 2
    let ratio = quadratic.mag / linear.mag;
    assert!(
        (ratio - t / 2.0).abs() < EPSILON,
        "quadratic/linear ratio is t/2"
    );

    // from rest, the travel is pure ½at²
    assert!(
        acceleration.scale(0.5 * t * t).near_mag(22.5),
        "from rest: d = ½at²"
    )
}

#[test]
fn it_reads_kinematic_level_off_grade_not_blade() {
    // kinematic level is grade (blade % 4), not blade. a position wound 1000 quarter
    // turns out differentiates through the same grade sequence as one at blade 0 — the
    // derivative order is dimension-blind

    let low = Geonum::new(5.0, 0.0, 1.0); // blade 0, grade 0
    let high = Geonum::new_with_blade(5.0, 1000, 0.0, 1.0); // blade 1000, grade 0

    let mut a = low;
    let mut b = high;
    for _ in 0..4 {
        a = a.differentiate();
        b = b.differentiate();
        assert_eq!(a.angle.grade(), b.angle.grade(), "same grade at any blade");
    }

    assert_eq!(b.angle.blade(), 1004, "blade accumulates the history");
    assert_eq!(b.angle.grade(), 0, "grade returns to 0 (1004 % 4)");
}

#[test]
fn it_recovers_position_by_integrating_back_down_the_hierarchy() {
    // integration is the inverse quarter turn (−π/2, taken forward as 3π/2). climb to
    // jerk and integrate three times: base_angle recovers the position, same magnitude
    // same direction, no drift because the rotation is exact

    let position = Geonum::new(7.0, 2.0, 5.0); // [7, 2π/5]
    let jerk = position.differentiate().differentiate().differentiate();

    let recovered = jerk.integrate().integrate().integrate().base_angle();

    assert!(
        recovered.near(&position),
        "integrate back to the same position"
    )
}

// ═══════════════════════════════════════════════════════════
// act II: dynamic quantities are wedge and dot, not cross products and norms
// ═══════════════════════════════════════════════════════════

#[test]
fn it_wedges_position_and_momentum_into_angular_momentum() {
    // conventional mechanics builds L = r × p from a cross product over a basis — six
    // component multiplies in 3D. here it is one wedge: a grade-2 bivector whose
    // magnitude is |r||p||sinΔ|, and a radial momentum wedges to nothing

    let r = Geonum::new(3.0, 1.0, 6.0); // 3 m at π/6
    let momentum = Geonum::new(8.0, 2.0, 3.0); // 8 kg·m/s at 2π/3, a quarter turn off r

    let l = r.wedge(&momentum);
    assert_eq!(l.angle.grade(), 2, "angular momentum is a bivector");
    assert!(l.near_mag(24.0), "|L| = 3×8×sin(π/2) = 24");

    let radial = Geonum::new(8.0, 1.0, 6.0); // along r
    assert!(
        r.wedge(&radial).near_mag(0.0),
        "radial momentum sweeps no area"
    )
}

#[test]
fn it_wedges_lever_and_force_into_torque() {
    // torque is τ = r × F, another cross product, and τ = dL/dt by a separate calculus
    // argument. here τ = r∧F is one wedge, and τ = dL/dt is one differentiate() — the
    // quarter turn that advances the angular momentum by a grade

    let r = Geonum::new(2.0, 1.0, 6.0); // 2 m lever at π/6
    let force = Geonum::new_with_blade(10.0, 2, 1.0, 4.0); // 10 N, grade 2

    let torque = r.wedge(&force);
    assert!(
        r.scale(2.0).wedge(&force).near_mag(2.0 * torque.mag),
        "double the lever, double the torque"
    );

    let along = Geonum::new(10.0, 1.0, 6.0); // along r
    assert!(
        r.wedge(&along).near_mag(0.0),
        "parallel force exerts no torque"
    );

    // τ = dL/dt: differentiating the angular momentum advances it one grade
    let momentum = Geonum::new_with_blade(4.0, 1, 1.0, 4.0);
    let l = r.wedge(&momentum);
    assert_eq!(l.angle.grade(), 2, "angular momentum is a bivector");
    assert_eq!(
        l.differentiate().angle.grade(),
        3,
        "dL/dt is one quarter turn past L — that is the torque"
    )
}

#[test]
fn it_relates_angular_and_linear_motion_through_the_wedge() {
    // v = ω × r and centripetal a = ω²r are cross-product and vector-identity results
    // conventionally. here v = ω∧r gives |v| = ωr directly, and a = ω²r is the ω·ω dot
    // scaling the radius

    let omega = Geonum::new_with_blade(2.0, 1, 0.0, 1.0); // 2 rad/s, grade 1
    let r = Geonum::new(3.0, 0.0, 1.0); // 3 m radius

    let v = omega.wedge(&r);
    assert!(v.near_mag(6.0), "|v| = ωr = 2×3 = 6");
    assert!(
        omega.scale(2.0).wedge(&r).near_mag(12.0),
        "double ω doubles v"
    );
    assert!(
        omega.wedge(&r.scale(2.0)).near_mag(12.0),
        "double r doubles v"
    );

    let a_c = r.scale(omega.dot(&omega).mag);
    assert!(a_c.near_mag(12.0), "a = ω²r = 4×3 = 12")
}

#[test]
fn it_dots_velocity_into_kinetic_energy() {
    // KE = ½m|v|² needs the velocity vector's norm in n dimensions conventionally. here
    // the self-dot v·v lands |v|² at grade 0, so energy is the velocity's own
    // interaction — quadratic by construction, at any winding

    let mass = 3.0;
    let velocity = Geonum::new_with_blade(8.0, 1, 1.0, 7.0); // 8 m/s, grade 1

    let ke = velocity.dot(&velocity).scale(0.5 * mass);
    assert_eq!(ke.angle.grade(), 0, "kinetic energy is a scalar");
    assert!(ke.near_mag(96.0), "KE = ½·3·64 = 96 J");

    let ke_double = velocity
        .scale(2.0)
        .dot(&velocity.scale(2.0))
        .scale(0.5 * mass);
    assert!(ke_double.near_mag(384.0), "KE(2v) = 4×96 = 384 J");

    let high = Geonum::new_with_blade(8.0, 1000, 1.0, 7.0);
    assert!(
        high.dot(&high).scale(0.5 * mass).near_mag(96.0),
        "same energy at blade 1000"
    )
}

#[test]
fn it_dots_force_into_work_and_power() {
    // work W = ∫F·dr is a path integral and power P = dW/dt a time derivative
    // conventionally. here both are one dot at the angle between the vectors: W = F·d,
    // P = F·v, full when aligned and gone across a perpendicular — no path, no clock

    let force = Geonum::new(10.0, 1.0, 6.0); // 10 N at π/6

    let displacement = Geonum::new(3.0, 1.0, 6.0); // 3 m, aligned
    assert!(
        force.dot(&displacement).near_mag(30.0),
        "W = F·d = 10×3 = 30 J"
    );

    let across = Geonum::new(3.0, 2.0, 3.0); // 2π/3, a quarter turn off the force
    assert!(force.dot(&across).near_mag(0.0), "F⊥d does no work");

    let velocity = Geonum::new(4.0, 1.0, 6.0); // 4 m/s, aligned
    assert!(force.dot(&velocity).near_mag(40.0), "P = F·v = 10×4 = 40 W");

    // opposing motion extracts energy: the dot lands negative, encoded at grade 2
    let opposing = velocity.negate(); // π apart
    let extracted = force.dot(&opposing);
    assert_eq!(
        extracted.angle.grade(),
        2,
        "opposing motion is negative power"
    );
    assert!(extracted.near_mag(40.0), "energy extracted at 40 W")
}

#[test]
fn it_dots_radius_into_rotational_inertia() {
    // moment of inertia is I = ∫r²dm, a mass-distribution integral conventionally. for a
    // point mass it is m(r·r) — the radius self-dot at grade 0 scaled by mass — and the
    // parallel-axis shift composes from the same dot

    let mass = 2.0;
    let r = Geonum::new(3.0, 0.0, 1.0); // 3 m from the axis

    let inertia = mass * r.dot(&r).mag;
    assert!((inertia - 18.0).abs() < EPSILON, "I = mr² = 2×9 = 18 kg·m²");

    let double = mass * r.scale(2.0).dot(&r.scale(2.0)).mag;
    assert!((double - 72.0).abs() < EPSILON, "I(2r) = 2×36 = 72");

    let r_cm = Geonum::new(1.0, 0.0, 1.0);
    let shift = Geonum::new(2.0, 0.0, 1.0);
    let i_parallel = mass * r_cm.dot(&r_cm).mag + mass * shift.dot(&shift).mag;
    assert!(
        (i_parallel - 10.0).abs() < EPSILON,
        "I_cm + md² = 2 + 8 = 10"
    )
}

// ═══════════════════════════════════════════════════════════
// act III: force, momentum, and mass are scaling and blade arithmetic
// ═══════════════════════════════════════════════════════════

#[test]
fn it_scales_acceleration_into_force_with_mass_as_pure_scalar() {
    // F = ma sits inside vector spaces and coordinate frames conventionally. here it is
    // one scale: the acceleration's angle rides through unchanged (force ∥ acceleration),
    // only the magnitude grows, and mass carries no angle at all

    let mass = 3.0;
    let acceleration = Geonum::new_with_blade(5.0, 2, 1.0, 8.0); // 5 m/s², grade 2

    let force = acceleration.scale(mass);
    assert_eq!(force.angle, acceleration.angle, "force ∥ acceleration");
    assert!(force.near_mag(15.0), "|F| = m|a| = 15 N");

    let m_here = force.mag / acceleration.mag;
    let rotated = acceleration.rotate(Angle::new(1.0, 3.0));
    let m_rotated = rotated.scale(mass).mag / rotated.mag;
    assert!(
        (m_here - m_rotated).abs() < EPSILON,
        "mass is rotation-invariant"
    );
    assert!((m_here - 3.0).abs() < EPSILON, "m = F/a = 3 kg")
}

#[test]
fn it_scales_velocity_into_momentum_and_differentiates_it_to_force() {
    // p = mv and F = dp/dt are separate vector statements conventionally. here p is a
    // scale of the velocity at grade 1, F = dp/dt is one differentiate() to grade 2, and
    // the impulse-momentum theorem Δp = FΔt is a scale by time

    let mass = 4.0;
    let velocity = Geonum::new_with_blade(3.0, 1, 0.0, 1.0); // 3 m/s, grade 1

    let momentum = velocity.scale(mass);
    assert!(momentum.near_mag(12.0), "p = mv = 12 kg·m/s");
    assert_eq!(
        momentum.angle.grade(),
        1,
        "momentum rides the velocity grade"
    );
    assert_eq!(
        momentum.differentiate().angle.grade(),
        2,
        "dp/dt lands the force grade"
    );

    let acceleration = Geonum::new_with_blade(5.0, 2, 0.0, 1.0); // grade 2
    let force = acceleration.scale(mass);
    let dt = 2.0;
    let impulse = force.scale(dt);
    let delta_p = acceleration.scale(dt).scale(mass);
    assert!(
        impulse.near(&delta_p),
        "Δp = FΔt, the momentum the force delivers"
    )
}

// ═══════════════════════════════════════════════════════════
// act IV: the self-wedge is universal — conservation is antisymmetry
// ═══════════════════════════════════════════════════════════

#[test]
fn it_self_wedges_every_quantity_to_zero_conserved_or_not() {
    // "nilpotency expresses conservation" is a tempting reading, but a∧a = 0 holds for
    // every geonum because sin(θ−θ) = 0. fire it on a free momentum and on one changed by
    // an impulse — both self-wedge to zero, so the identity carries no information about
    // conservation. it witnesses that a state cannot repeat, a different fact

    let free = Geonum::new_with_blade(12.0, 1, 1.0, 5.0); // a momentum, no net force
    let driven = free + Geonum::new_with_blade(3.0, 1, 1.0, 5.0); // the same, after an impulse

    assert!(
        free.wedge(&free).near_mag(0.0),
        "free momentum self-wedges to 0"
    );
    assert!(
        driven.wedge(&driven).near_mag(0.0),
        "driven momentum self-wedges to 0 too"
    );

    let position = Geonum::new(4.0, 1.0, 3.0); // never a conserved quantity
    assert!(
        position.wedge(&position).near_mag(0.0),
        "so does anything else"
    )
}

#[test]
fn it_conserves_momentum_because_the_interaction_forces_are_pi_apart() {
    // momentum conservation is the antisymmetry of the interaction: F_ij = −F_ji, the
    // force on 1 from 2 and on 2 from 1 a π rotation apart (Newton's third law). the two
    // impulses cancel pairwise by that antisymmetry, so the total momentum has nothing to
    // move it — no translation symmetry searched, the cancellation is read straight off

    let dt = 0.01;
    let p1 = Geonum::new_with_blade(5.0, 1, 1.0, 5.0);
    let p2 = Geonum::new_with_blade(3.0, 1, 2.0, 7.0);
    let p_total = p1 + p2;

    let force_on_1 = Geonum::new_with_blade(9.0, 1, 1.0, 3.0); // the interaction force
    let force_on_2 = force_on_1.negate(); // third law: equal and opposite, π apart

    let p1_next = p1 + force_on_1.scale(dt); // each body takes its impulse
    let p2_next = p2 + force_on_2.scale(dt);

    assert!(
        force_on_1.scale(dt).near_mag(0.09),
        "body 1 takes a real 0.09 impulse"
    );
    // yet the total does not move — the two impulses interfere to zero
    let change = (p1_next + p2_next) - p_total;
    assert!(
        change.mag < EPSILON,
        "total momentum conserved: Δp_total = 0"
    )
}

#[test]
fn it_conserves_angular_momentum_because_the_wedge_is_antisymmetric() {
    // no symmetry search, no Lagrangian: conservation is the antisymmetry of the wedge.
    // dL/dt = d(r∧v)/dt = v∧v + r∧a. the first is a self-wedge, zero for any v; the second
    // is r∧a, zero whenever a is central (parallel to r). both vanish by antisymmetry, so
    // dL/dt = 0 is read straight off the antisymmetric product

    let r0 = Geonum::new(4.0, 1.0, 5.0);
    let v0 = Geonum::new_with_blade(2.0, 1, 1.0, 3.0);
    let a0 = Geonum::new_with_angle(1.5, r0.angle + Angle::new(1.0, 1.0)); // central: r + π

    let spin = v0.wedge(&v0); // v∧v
    let torque = r0.wedge(&a0); // r∧a
    assert!(spin.near_mag(0.0), "v∧v = 0: the self-wedge vanishes");
    assert!(torque.near_mag(0.0), "r∧a = 0: central a is parallel to r");
    // dL/dt is their sum, so it vanishes term by term

    // integrate a circular orbit and read L across the trajectory — it holds because both
    // wedge terms vanish at every step, which is Kepler's equal areas
    let gm = 1.0;
    let dt = 0.01;
    let mut r = Geonum::new(1.0, 0.0, 1.0); // radius 1
    let mut v = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // √(GM/r) = 1, perpendicular
    let l0 = r.wedge(&v).mag;

    let mut max_drift: f64 = 0.0;
    for _ in 0..300 {
        let a = Geonum::new_with_angle(gm / (r.mag * r.mag), r.angle + Angle::new(1.0, 1.0));
        v = (v + a.scale(dt)).base_angle(); // kick
        r = (r + v.scale(dt)).base_angle(); // drift
        max_drift = max_drift.max((r.wedge(&v).mag - l0).abs());
    }

    assert!(
        max_drift < 1e-9,
        "L held across the orbit: max drift {max_drift:.1e}"
    )
}

// ═══════════════════════════════════════════════════════════
// act V: energy — the conserved scalar is a magnitude
// ═══════════════════════════════════════════════════════════
//
// momentum conservation above is the π pairing of forces; energy gets the same
// geometry. the oscillator state is ONE geonum: the phase point
// [x·√(k/2), v·√(m/2)] built from newton's integrated output. its squared
// magnitude is ½kx² + ½mv², so the conserved energy is a MAGNITUDE — the
// pythagorean readout of one geonum, the quadrature closing with newton doing
// the rotating. the KE ↔ PE exchange is the quarter turn walking the grade
// cycle, dE/dt = 0 is the kinetic credit F·v interfering with the potential
// debit kx·ẋ — one dot placed π apart — and dissipation is the unpartnered
// term: a damper's −cv² has no π partner, so the magnitude drains by exactly
// that dot. no lagrangian, no symmetry search: the energy is a magnitude, the
// dynamics is rotation, and rotation never touches magnitude

const MASS: f64 = 2.0;
const SPRING_K: f64 = 8.0;
const OMEGA: f64 = 2.0; // √(k/m)
const AMPLITUDE: f64 = 1.5; // release displacement, from rest
const E0: f64 = 9.0; // ½k·A²
const DT: f64 = 1e-4;
const STEPS: usize = 31416; // one period T = 2π/ω = π

// the phase point: displacement on the adjacent leg weighted √(k/2), velocity on
// the opposite leg weighted √(m/2). its squared magnitude is ½kx² + ½mv².
// the state rides the 0/π rays, so the dimension-0 projection reads ±mag exactly
fn phase(x: &Geonum, v: &Geonum) -> Geonum {
    Geonum::new_from_cartesian(
        x.project_to_dimension(0) * (SPRING_K / 2.0).sqrt(),
        v.project_to_dimension(0) * (MASS / 2.0).sqrt(),
    )
}

fn spring(x: &Geonum) -> Geonum {
    x.negate().scale(SPRING_K) // hooke: −kx, the negate a π rotation
}

// kick-drift under a supplied force — newton only, no energy constructed
fn kick_drift(
    x: Geonum,
    v: Geonum,
    force: impl Fn(&Geonum, &Geonum) -> Geonum,
) -> (Geonum, Geonum) {
    let a = force(&x, &v).scale(1.0 / MASS);
    let v_next = (v + a.scale(DT)).base_angle();
    let x_next = (x + v_next.scale(DT)).base_angle();
    (x_next, v_next)
}

#[test]
fn it_conserves_energy_as_the_phase_magnitude() {
    let mut x = Geonum::new(AMPLITUDE, 0.0, 1.0); // released at +A
    let mut v = Geonum::scalar(0.0); // from rest

    let mut max_energy_drift: f64 = 0.0;
    let mut min_x = f64::INFINITY;
    let mut max_speed: f64 = 0.0;

    for _ in 0..STEPS {
        (x, v) = kick_drift(x, v, |x, _| spring(x));

        // the energy read two ways: the phase magnitude squared, and the dot-built legs
        let p = phase(&x, &v);
        let pe = x.dot(&x).scale(0.5 * SPRING_K);
        let ke = v.dot(&v).scale(0.5 * MASS);
        let legs = pe + ke; // both grade 0 — magnitudes add

        // the phase magnitude squared IS ½kx² + ½mv² — the pythagorean readout
        // of one geonum, exact at every step
        assert!(p.dot(&p).near(&legs), "phase.mag² = pe + ke");

        max_energy_drift = max_energy_drift.max((legs.mag - E0).abs() / E0);
        min_x = min_x.min(x.project_to_dimension(0));
        max_speed = max_speed.max(v.mag);
    }

    // the magnitude holds still across the whole period. the band is the
    // integrator's, not the algebra's: kick-drift at dt = 1e-4 wobbles the
    // measured energy at ~1e-4 relative, orders above near()'s tolerance
    assert!(
        max_energy_drift < 1e-3,
        "energy rides the magnitude: drift {max_energy_drift:.2e}"
    );

    // while the legs swing full range — x out to −A, speed up to Aω. the motion
    // is all in the angle; the magnitude never moved
    assert!(min_x < -0.99 * AMPLITUDE, "x swings to −A: min {min_x:.3}");
    assert!(
        max_speed > 0.99 * AMPLITUDE * OMEGA,
        "speed reaches Aω: max {max_speed:.3}"
    );
}

#[test]
fn it_exchanges_the_legs_by_the_quarter_turn() {
    let mut x = Geonum::new(AMPLITUDE, 0.0, 1.0);
    let mut v = Geonum::scalar(0.0);

    // the phase point rotates clockwise at ω, so at the odd eighths of the period
    // its angle is 7π/4, 5π/4, 3π/4, π/4 — grades 3, 2, 1, 0, the cycle walked once
    let eighth = STEPS / 8;

    for n in 1..=STEPS {
        (x, v) = kick_drift(x, v, |x, _| spring(x));
        if n % eighth != 0 {
            continue;
        }

        let i = n / eighth;
        let p = phase(&x, &v);
        let pe = x.dot(&x).scale(0.5 * SPRING_K);
        let ke = v.dot(&v).scale(0.5 * MASS);

        match i {
            1 | 3 | 5 | 7 => {
                // eighth turns: the exchange caught halfway, equal energy on each leg
                let expected_grade = match i {
                    1 => 3,
                    3 => 2,
                    5 => 1,
                    _ => 0,
                };
                assert_eq!(
                    p.angle.grade(),
                    expected_grade,
                    "eighth {i}: the phase point walks the grade cycle"
                );
                assert!(
                    (pe.mag - E0 / 2.0).abs() / E0 < 1e-3,
                    "eighth {i}: half the energy on the potential leg"
                );
                assert!(
                    (ke.mag - E0 / 2.0).abs() / E0 < 1e-3,
                    "eighth {i}: half on the kinetic leg"
                );
            }
            2 | 6 => {
                // quarter turns: the potential leg empty, all energy kinetic
                assert!(pe.mag / E0 < 1e-3, "quarter turn {i}: potential leg empty");
                assert!(
                    (ke.mag - E0).abs() / E0 < 1e-3,
                    "quarter turn {i}: all energy on the kinetic leg"
                );
            }
            _ => {
                // half and full period: all energy back on the potential leg
                assert!(ke.mag / E0 < 1e-3, "half turn {i}: kinetic leg empty");
                assert!(
                    (pe.mag - E0).abs() / E0 < 1e-3,
                    "half turn {i}: all energy on the potential leg"
                );
            }
        }
    }
}

#[test]
fn it_cancels_the_kinetic_credit_against_the_potential_debit() {
    let mut x = Geonum::new(AMPLITUDE, 0.0, 1.0);
    let mut v = Geonum::scalar(0.0);

    let mut credit_sample = Geonum::scalar(0.0);
    let mut debit_sample = Geonum::scalar(0.0);

    for n in 1..=STEPS {
        (x, v) = kick_drift(x, v, |x, _| spring(x));

        // the kinetic credit F·v and the potential debit kx·ẋ — the same dot,
        // placed π apart by the force's negate. pointwise annihilation:
        // dE/dt = 0 read as a π pair interfering, the same cancellation the
        // third law gives momentum
        let credit = spring(&x).dot(&v);
        let debit = x.scale(SPRING_K).dot(&v);
        assert!(
            (credit + debit).near_mag(0.0),
            "credit + debit interfere to zero at every step"
        );

        if n == STEPS / 8 {
            credit_sample = credit;
            debit_sample = debit;
        }
    }

    // and not by idleness: at t = T/8 each side carries kA²ω/2 = 18 W
    let expected_power = SPRING_K * AMPLITUDE * AMPLITUDE * OMEGA / 2.0;
    assert!(
        (credit_sample.mag - expected_power).abs() / expected_power < 1e-2,
        "the credit carries kA²ω/2: {:.3}",
        credit_sample.mag
    );
    assert!(
        credit_sample.angle.is_opposite(&debit_sample.angle),
        "credit and debit sit π apart — the pairing is the conservation"
    );
}

#[test]
fn it_drains_energy_only_through_the_unpartnered_power() {
    const DAMP_C: f64 = 0.4; // damping coefficient

    let mut x = Geonum::new(AMPLITUDE, 0.0, 1.0);
    let mut v = Geonum::scalar(0.0);

    let mut drained = 0.0; // ∫ F_damp·v dt — f64 at the boundary, like a loss readout

    for _ in 0..STEPS {
        (x, v) = kick_drift(x, v, |x, v| spring(x) + v.negate().scale(DAMP_C));

        // the damper's power −cv² lands at grade 2 with no π partner
        drained += v.negate().scale(DAMP_C).dot(&v).project_to_dimension(0) * DT;

        // the spring pair still cancels under damping — its conservation is untouched
        let credit = spring(&x).dot(&v);
        let debit = x.scale(SPRING_K).dot(&v);
        assert!(
            (credit + debit).near_mag(0.0),
            "the spring pair cancels under damping too"
        );
    }

    // the budget closes: every joule lost left through the unpartnered dot
    let p = phase(&x, &v);
    let e_final = p.dot(&p).mag;
    assert!(
        (e_final - E0 - drained).abs() / E0 < 1e-2,
        "energy loss = accumulated unpartnered power: {:.4} vs {:.4}",
        e_final - E0,
        drained
    );

    // and the drain follows the light-damping envelope E(T) = E0·e^(−cT/m)
    let envelope = (-DAMP_C / MASS * std::f64::consts::PI).exp();
    assert!(
        (e_final / E0 - envelope).abs() < 0.05,
        "measured decay {:.3} tracks the envelope {:.3}",
        e_final / E0,
        envelope
    );
}
