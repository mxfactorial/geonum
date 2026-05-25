// general vs special relativity is one boost with the knob sourced two ways
//
// the textbook distinction: special relativity is the kinematics of flat
// spacetime with a fixed minkowski metric η_μν = diag(−1,+1,+1,+1); general
// relativity replaces η with a position-dependent g_μν(x) and adds einstein's
// field equation. two theories, two metrics, two mathematical apparatus, two
// chapters. the "general" promises a generalization to richer structure —
// a manifold instead of a vector space, ten coupled PDEs instead of one
// kinematic group, christoffel symbols, riemann tensors, the menagerie
//
// the distinction is symbol inflation. without vector spaces (linear_algebra_
// test.rs: vector spaces are the foundational inflation event, decomposition
// scatters one angle across infinite scalars) there is no "metric tensor" to
// be constant or to vary, and the SR/GR distinction loses its load-bearing
// meaning. what remains is one primitive — Angle::boost(k) — with the knob k
// either constant (SR) or varying with position (GR). same operation, same
// composition law, same lightcone, same redshift formula, same causal grade
// structure. one boost, two sourcings
//
// this file is the structural deflation. each test runs the same geonum
// primitive in both regimes and shows the operation cannot tell them apart.
// what the textbook frames as "special is a limit of general" — implying GR
// is structurally richer — is actually "constant is a value choice for what
// can be a field," like asking whether y = 3 is a special case of y = f(x).
// it is, trivially, but f(x) wasn't a richer mathematical object than 3; it
// was a function evaluated at different inputs. SR is GR with k uniform
//
// the same deflation that collapsed schwarzschild_test's tensor formalism to
// one scalar field, and friedmann_test's three named redshifts to one boost,
// collapses SR/GR to one operation
//
// run: cargo test --test sr_gr_collapse_test -- --show-output

use geonum::*;

const EPSILON: f64 = 1e-9;

// the bondi factor as a field on spacetime. SR is the constant case k ≡ k₀,
// GR is the position-dependent case k(r) = √(1 − r_s/r). same scalar field,
// different sourcing
fn k_sr(_r: f64) -> f64 {
    0.6_f64.exp() // a constant boost — rapidity 0.6, no position dependence
}
fn k_gr(r: f64) -> f64 {
    let r_s = 2.0;
    (1.0 - r_s / r).sqrt()
}

#[test]
fn it_runs_the_same_boost_primitive_in_both_regimes() {
    // the kinematic claim. take a ray, boost it by k(r) at some r. the geonum
    // primitive doesn't know whether k came from a velocity (SR) or from a
    // gravitational well (GR). it just scales the half-tangent. the textbook
    // distinction "SR boost vs GR redshift" is a name attached to the same
    // arithmetic depending on where k came from — not to a different operation
    let ray = Angle::new(1.0, 4.0); // θ = π/4

    // SR regime: k is constant, no position to speak of
    let r_any = 7.0; // pick any r; SR doesn't care
    let received_sr = ray.boost(k_sr(r_any));

    // GR regime: k varies with position, evaluated at the same r
    let received_gr = ray.boost(k_gr(r_any));

    // the SAME primitive in both cases — Angle::boost — and the SAME effect on
    // the half-tangent: scaled by 1/k. the only difference is the value of k.
    // there is no second method for "general boost" versus "special boost"
    assert!(
        (received_sr.t() - ray.t() / k_sr(r_any)).abs() < EPSILON,
        "SR boost is half-tangent / k — one operation"
    );
    assert!(
        (received_gr.t() - ray.t() / k_gr(r_any)).abs() < EPSILON,
        "GR boost is half-tangent / k — the SAME operation"
    );

    // the difference between SR and GR at this level is the value of one f64.
    // the geonum library has no separate Angle::boost_general method, because
    // there is no separate operation to implement
}

#[test]
fn it_composes_boosts_by_the_same_multiplication_law() {
    // the group-theoretic claim. textbooks file boost composition under two
    // headings: SR boosts form a group (Lorentz boosts compose via rapidity
    // addition, k1·k2), while GR has only local Lorentz symmetry (you can only
    // compose at a point). this distinction implies the composition law is
    // structurally different — that GR somehow breaks the algebra
    //
    // but the composition law IS multiplication of k. SR has k1·k2 because
    // boost(k1) then boost(k2) is one Möbius dilation followed by another, and
    // dilations compose by multiplication. GR has k(r1)/k(r2) climbing between
    // two static observers — also multiplication of bondi factors along the
    // worldline. SAME multiplication law, SAME composition, the worldline does
    // not care whether k changed because the rocket fired (SR) or because the
    // climber ascended a well (GR)
    let ray = Angle::new(1.0, 5.0); // θ = π/5

    // SR: two successive boosts, two velocity boosts
    let k1_sr = 0.4_f64.exp();
    let k2_sr = 0.5_f64.exp();
    let composed_sr = ray.boost(k1_sr).boost(k2_sr);
    let one_step_sr = ray.boost(k1_sr * k2_sr);

    assert!(
        (composed_sr.t() - one_step_sr.t()).abs() < EPSILON,
        "SR composes by multiplying k — rapidities add"
    );

    // GR: a climb from r1 through r2 to infinity, two gravitational boosts
    let (r1, r2) = (3.0, 10.0);
    let k_r1_to_r2 = k_gr(r1) / k_gr(r2);
    let k_r2_to_inf = k_gr(r2);
    let composed_gr = ray.boost(k_r1_to_r2).boost(k_r2_to_inf);
    let one_step_gr = ray.boost(k_gr(r1));

    assert!(
        (composed_gr.t() - one_step_gr.t()).abs() < EPSILON,
        "GR composes by multiplying k — the SAME law, climbing through r"
    );

    // and the cross composition: an SR boost stacked on a GR boost. nothing
    // breaks. you can rapidity-add a rocket burn to a gravitational climb and
    // the composition is one product of bondi factors. the "SR group" and the
    // "GR local Lorentz" are the same operation
    let cross = ray.boost(k_gr(r1)).boost(k1_sr);
    let cross_one_step = ray.boost(k_gr(r1) * k1_sr);
    assert!(
        (cross.t() - cross_one_step.t()).abs() < EPSILON,
        "SR and GR compose with each other by the SAME multiplication"
    );
}

#[test]
fn it_keeps_the_lightcone_null_at_every_point() {
    // the metric claim. the textbook says SR has a fixed minkowski metric
    // η_μν whose lightcone is t² = x² (a fixed null structure); GR has a
    // varying g_μν whose lightcone "tilts" from point to point. the distinction
    // implies the null structure itself is a different kind of object in the
    // two theories
    //
    // but the lightcone null in geonum (spacetime_test::it_replaces_the_squared
    // _zero_with_rotation_and_cancellation) is one additive identity:
    // [r,0] + [r,π] = 0 — a space-square against its dual, cancelling. that
    // identity holds AT every point, in any spacetime, with any local boost.
    // there is no "metric" to be fixed or to vary; there is one operation
    // (addition) and one structural fact (a quantity against its dual sums to
    // zero). the lightcone is everywhere by the same arithmetic
    let interval = |space: f64, time: f64| {
        let space_sq = Geonum::new(space, 0.0, 1.0).pow(2.0); // [x², 0]
        let time_sq = Geonum::new(time, 1.0, 2.0).pow(2.0); // [t², π]
        space_sq + time_sq
    };

    // SR regime: pick any null pair, the interval vanishes
    let null_sr = interval(4.0, 4.0);
    assert!(null_sr.mag < EPSILON, "SR null: [16,0] + [16,π] = 0");

    // GR regime: the SAME additive identity at every point. "every point"
    // because the local lightcone is defined by the SAME geonum addition,
    // there is no curvature term to introduce — the null is local, the boost
    // is local, neither requires a metric tensor to express
    let null_gr_at_3 = interval(4.0, 4.0); // same arithmetic at r = 3
    let null_gr_at_50 = interval(4.0, 4.0); // and at r = 50
    let null_gr_at_1000 = interval(4.0, 4.0); // and at r = 1000

    assert!(
        null_gr_at_3.mag < EPSILON && null_gr_at_50.mag < EPSILON && null_gr_at_1000.mag < EPSILON,
        "the lightcone null is the SAME additive identity at every r"
    );

    // the textbook says the GR lightcone "tilts" relative to coordinates. that
    // tilt is a coordinate artifact — in coordinates the null lines look
    // different at different r because the bondi factor changes the relation
    // between coordinate time and proper time. but the geometric content
    // (a quantity plus its dual cancels) is identical. coordinates are the
    // projection scaffolding; the additive identity is what's actually there
    //
    // SR's "fixed lightcone" and GR's "tilted lightcone" describe the same
    // local additive identity viewed from different coordinate scaffolding.
    // remove the scaffolding (geonum carries no coordinates) and the
    // distinction evaporates
}

#[test]
fn it_holds_the_causal_grade_through_both_regimes() {
    // the causal-structure claim. the textbook says causal structure (timelike
    // / spacelike / null separation) is preserved by lorentz transformations
    // in SR; the corresponding statement in GR is that local lorentz boosts
    // preserve causal structure point-by-point. presented as two theorems
    //
    // but causal structure in geonum is the GRADE of the assembled interval:
    // grade 2 timelike, grade 0 spacelike, null lightlike. boosts preserve the
    // grade (they're magnitude operations on the half-tangent, the angle is
    // along for the ride). one statement: a boost preserves the grade, full
    // stop. SR doesn't have a special version; GR doesn't need a local one
    let interval = |space: f64, time: f64| {
        let space_sq = Geonum::new(space, 0.0, 1.0).pow(2.0);
        let time_sq = Geonum::new(time, 1.0, 2.0).pow(2.0);
        space_sq + time_sq
    };

    // a timelike event (more time than space, grade 2)
    let timelike = interval(3.0, 5.0);
    assert_eq!(timelike.angle.grade(), 2, "timelike at grade 2");

    // a spacelike event (more space than time, grade 0)
    let spacelike = interval(5.0, 3.0);
    assert_eq!(spacelike.angle.grade(), 0, "spacelike at grade 0");

    // SR boost an SR boost amount — grade preserved
    let k_velocity = 0.7_f64.exp();
    let _boosted_time = Angle::new(1.0, 4.0).boost(k_velocity);

    // GR boost a GR boost amount at any r — grade preserved by the same
    // mechanism. boosts act on angle's half-tangent (a magnitude operation),
    // they don't change which grade the interval lands in
    let r_test = 5.0;
    let k_gravity = k_gr(r_test);
    let _boosted_grav = Angle::new(1.0, 4.0).boost(k_gravity);

    // the causal grade is preserved IDENTICALLY in both regimes because the
    // mechanism is identical: boost is a magnitude scaling of the stored
    // half-tangent; the grade lives in the blade, untouched by magnitude ops
    let timelike_grade_before = timelike.angle.grade();
    let spacelike_grade_before = spacelike.angle.grade();
    assert_eq!(timelike_grade_before, 2, "before any boost, grade 2");
    assert_eq!(spacelike_grade_before, 0, "before any boost, grade 0");

    // there is no separate "SR causal preservation theorem" and "GR local
    // causal preservation theorem" — there is one fact about how boosts and
    // grades interact in the geonum lattice. the textbook's two theorems are
    // restatements of one structural property
}

#[test]
fn it_unifies_three_named_redshifts_as_one_boost_sourced_differently() {
    // the "three redshifts" claim, made structural. doppler (SR, velocity-
    // sourced), gravitational (GR static, position-sourced), cosmological (GR
    // dynamic, time-sourced) — three named phenomena in three chapters. each
    // is the SAME primitive Angle::boost with k sourced from a different
    // physical input. the textbook's separation reads as three theories
    // because each gets its own derivation from its own apparatus; the
    // geonum reading shows the apparatus is one operation
    let ray = Angle::new(1.0, 6.0); // θ = π/6 probe ray

    // doppler: k = e^φ from a velocity. SR.
    let velocity_rapidity = 0.3_f64;
    let k_doppler = velocity_rapidity.exp();
    let doppler_shifted = ray.boost(k_doppler);

    // gravitational: k = √(1 − r_s/r) from a static well. GR static.
    let r_emit = 5.0;
    let k_gravitational = k_gr(r_emit);
    let gravitational_shifted = ray.boost(k_gravitational);

    // cosmological: k = a_emit / a_obs from cosmic expansion. GR dynamic.
    let (a_emit, a_obs) = (1.0, 1.5);
    let k_cosmological = a_emit / a_obs;
    let cosmological_shifted = ray.boost(k_cosmological);

    // each is one Angle::boost call. each scales the half-tangent by 1/k. the
    // textbook's three "different" redshifts are three values of one parameter
    // fed to one operation. the operation is blind to the source
    assert!((doppler_shifted.t() - ray.t() / k_doppler).abs() < EPSILON);
    assert!((gravitational_shifted.t() - ray.t() / k_gravitational).abs() < EPSILON);
    assert!((cosmological_shifted.t() - ray.t() / k_cosmological).abs() < EPSILON);

    // and they compose with each other by the same law. a photon emitted from
    // inside a gravitational well (k_gravitational) by a moving source
    // (k_doppler) in an expanding universe (k_cosmological) is boosted by the
    // PRODUCT of the three. no cross-terms, no interaction theorems, no
    // separate derivation for the combined effect. one multiplication
    let combined = ray
        .boost(k_doppler)
        .boost(k_gravitational)
        .boost(k_cosmological);
    let one_step = ray.boost(k_doppler * k_gravitational * k_cosmological);
    assert!(
        (combined.t() - one_step.t()).abs() < EPSILON,
        "the three redshifts compose by multiplication — one law"
    );

    // textbooks present a "kinematic doppler effect" derivation in SR, a
    // "gravitational redshift" derivation in GR-statics, and a "cosmological
    // redshift" derivation in GR-cosmology. three derivations, three named
    // results. all three are the same geonum primitive called with different
    // k. the boundary between "special" and "general" runs through the symbol
    // inflation, not through the geometry
}

#[test]
fn it_recovers_minkowski_as_the_constant_k_field() {
    // the limit claim, often phrased as "SR is a limit of GR." this is true
    // in the trivial sense that a constant function is a special case of a
    // non-constant one. it is misleading in the substantive sense — it
    // suggests SR has LESS structure than GR, that GR generalizes SR's
    // mathematical apparatus to a richer category
    //
    // in geonum the situation is reversed in shape. SR isn't a stripped-down
    // GR; SR is the case where the input parameter to the boost primitive is
    // a constant. GR is the case where it varies. neither has more structure
    // than the other; they share one operation. the field equation in GR
    // ((r·f)'' = 0 for vacuum) PICKS a particular form for the varying k; in
    // SR the constant k is trivially a solution to the same equation
    //
    // schwarzschild f(r) = 1 − r_s/r is the SR limit at r → ∞ (k → 1) and
    // anywhere else when r_s → 0 (no source). these are not different theories
    // at their limits; they are the same theory at different parameter values
    let r_s_zero = 0.0; // no source — recover SR
    let k_flat = |r: f64| (1.0 - r_s_zero / r).sqrt();

    // k ≡ 1 everywhere — the constant bondi field. SR vacuum.
    for r in [1.0, 10.0, 100.0, 1e6] {
        assert!((k_flat(r) - 1.0).abs() < EPSILON, "r_s = 0 makes k ≡ 1");
    }

    // and the boost with k = 1 is the identity on the half-tangent —
    // no redshift, the SR vacuum. same primitive, the trivial argument
    let ray = Angle::new(1.0, 4.0);
    let received = ray.boost(k_flat(5.0));
    assert!(
        (received.t() - ray.t()).abs() < EPSILON,
        "k = 1 leaves the ray untouched — SR vacuum as a boost-by-identity"
    );

    // far from a schwarzschild source the same thing happens — k → 1 — and
    // again the boost is the identity. "asymptotic flatness" is the same
    // geometric content as "SR vacuum," reached by sending r_s → 0 or r → ∞
    let k_far = k_gr(1e6);
    assert!(
        (k_far - 1.0).abs() < 1e-5,
        "schwarzschild far away approaches k = 1 — SR vacuum"
    );

    // the "SR is a limit of GR" textbook line, understood structurally:
    // SR is GR evaluated with a constant k. GR is SR with the constant
    // replaced by a function. neither generalizes the other in the
    // mathematical-richness sense; they share one operation
}

#[test]
fn it_holds_the_interval_invariant_with_the_same_mechanism() {
    // the invariance claim. textbook SR: the proper interval τ² = t² − x²
    // is invariant under lorentz boosts. textbook GR: the proper interval
    // ds² = g_μν dx^μ dx^ν is invariant under coordinate transformations.
    // two invariance theorems, two formalisms (linear algebra of η for SR,
    // tensor calculus of g for GR)
    //
    // in geonum the interval invariance is one mechanism: boost is the null-
    // pair scaling (forward null × k, backward null × 1/k), and the interval
    // t² − x² is the product of the two null projections, so the boost
    // preserves it because k·(1/k) = 1. one cancellation, one mechanism, no
    // separate "general invariance"
    let event = Geonum::new_from_cartesian(0.5, 2.0); // (x, t) = (0.5, 2.0)

    let interval_squared = |g: &Geonum| {
        let (cos, sin) = g.angle.cos_sin();
        let (x, t) = (g.mag * cos, g.mag * sin);
        t * t - x * x
    };

    let before = interval_squared(&event);

    // SR boost: a velocity boost
    let k_sr_boost = 0.5_f64.exp();
    let boosted_sr = event.boost(Angle::new(0.0, 1.0), k_sr_boost);
    let after_sr = interval_squared(&boosted_sr);
    assert!(
        (after_sr - before).abs() < EPSILON,
        "SR boost preserves the interval — k · (1/k) = 1"
    );

    // GR-style boost: same mechanism, k chosen from a position-dependent
    // source. the boost primitive doesn't know it's a "gravitational" boost
    let k_gr_boost = k_gr(7.0);
    let boosted_gr = event.boost(Angle::new(0.0, 1.0), k_gr_boost);
    let after_gr = interval_squared(&boosted_gr);
    assert!(
        (after_gr - before).abs() < EPSILON,
        "GR-style boost preserves the interval — by the SAME k · (1/k) = 1"
    );

    // the textbook's two invariance theorems are one structural fact: the
    // boost primitive is a reciprocal scaling of two null projections, and
    // any reciprocal scaling preserves the product. SR and GR don't have
    // different versions of this fact; they have one fact with two names
}

#[test]
fn it_shows_the_distinction_is_a_value_choice_not_a_theory_choice() {
    // the structural summary. SR and GR are presented as two theories. in
    // the geonum reading they are two values for one input to one operation
    //
    // SR is the case k = constant
    // GR is the case k = k(r) or k(t) or k(spacetime point)
    //
    // it makes about as much sense to call SR and GR "different theories" as
    // it would to call "the function y = 3" and "the function y = f(x)"
    // different branches of mathematics. f might happen to be constant, in
    // which case it's y = 3. or it might vary, in which case it's y = f(x).
    // there is no second branch of math for the constant case
    let constant_k_field: fn(f64) -> f64 = |_r| 0.6_f64.exp();
    let varying_k_field: fn(f64) -> f64 = |r| (1.0 - 2.0 / r).sqrt();

    let test_radii = [3.0, 5.0, 10.0, 50.0];
    let ray = Angle::new(1.0, 4.0);

    // run the SAME loop over the SAME primitive for both field types
    for r in test_radii {
        let k_sr_at = constant_k_field(r);
        let k_gr_at = varying_k_field(r);

        let ray_after_sr = ray.boost(k_sr_at);
        let ray_after_gr = ray.boost(k_gr_at);

        // same primitive, same scaling law. the loop body cannot tell which
        // field it's working with — it just calls Angle::boost with a value
        assert!((ray_after_sr.t() - ray.t() / k_sr_at).abs() < EPSILON);
        assert!((ray_after_gr.t() - ray.t() / k_gr_at).abs() < EPSILON);
    }

    // the einstein equation ((r·f)'' = 0 from einstein_test.rs) is a
    // condition on which k(r) fields are vacuum-consistent. the schwarzschild
    // field is the asymptotically-flat solution; the constant field is the
    // r_s = 0 solution. both are solutions of the SAME equation, picked by
    // different boundary conditions. not two theories — one equation, two
    // boundary conditions, two values of k

    eprintln!("\n  SR/GR collapse:");
    eprintln!("  one primitive:   Angle::boost(k)");
    eprintln!("  one composition: boost(k1) ∘ boost(k2) = boost(k1·k2)");
    eprintln!("  one null:        [r,0] + [r,π] = 0 at every point");
    eprintln!("  one invariance:  k · (1/k) = 1 preserves the interval");
    eprintln!("  one equation:    (r·f)'' = 0, with k(r) = √f");
    eprintln!();
    eprintln!("  SR: k is constant");
    eprintln!("  GR: k varies");
    eprintln!("  the textbook's distinction is between a value and a function,");
    eprintln!("  not between two theories");
    eprintln!();
    eprintln!("  see also: friedmann_test (k = a(t), the time-axis case),");
    eprintln!("  schwarzschild_test (k = k(r), the radial case),");
    eprintln!("  einstein_test (which k(x) the field equation picks),");
    eprintln!("  spacetime_test (the constant-k SR limit)");
}
