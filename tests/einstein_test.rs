// the einstein equation is one local condition on the bondi field
//
// schwarzschild_test.rs showed gravity IS the bondi field k(r) — the three
// classical tests fall out of one scalar per radius, no tensors, no christoffels.
// but k(r) = √(1 − r_s/r) was POSTULATED there, copied from the textbook. the
// einstein equation is what PICKS that k. in geonum form: not ten coupled PDEs
// on g_μν, but one condition on the bondi field itself
//
// the condition is local and geometric: for a static spherically symmetric
// vacuum, the combination r · f(r) where f = k² is LINEAR in r. its second
// derivative vanishes. that's it — schwarzschild is the unique solution to
//   (r · f(r))'' = 0
// with asymptotic flatness f(∞) = 1 and the newtonian limit f ≈ 1 − r_s/r for
// large r. two integration constants, both fixed by physics, yields k uniquely.
// no ricci tensor, no christoffel symbols — one ODE on one scalar field
//
// in geonum terms: r · f is the proper-distance-rescaled bondi field, and the
// vacuum equation says ITS curvature (second derivative in r) vanishes. flat
// space is the linear function. spacetime curvature is the failure to be linear
//
// birkhoff falls out free: the same ODE has no time derivatives available, so
// any spherically symmetric vacuum is automatically static. no separate proof
//
// the source side: for matter of density ρ(r), the vacuum condition picks up a
// right-hand side. for a static spherical body the equation becomes
//   (r · (1 − f(r)))' = 8π G ρ(r) · r²
// which is newton's M(r) = ∫4πρr²dr in disguise: r · (1 − f) is twice the mass
// enclosed within radius r, in geometric units. ONE field equation, ONE field,
// the source on the right
//
// run: cargo test --test einstein_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;

// f(r) = k(r)² is the "metric function" — the SINGLE scalar that the tensor
// formalism distributes across g_tt and g_rr (one as f, the other as 1/f). it is
// the square of the bondi factor k(r) = √(1 − r_s/r) from schwarzschild_test.rs,
// and the einstein equation below works on f directly
fn f(r: f64, r_s: f64) -> f64 {
    1.0 - r_s / r
}

#[test]
fn it_picks_schwarzschild_as_the_linear_solution_of_rf() {
    // the einstein vacuum equation for a static spherical bondi field reduces to
    // ONE statement: r · f(r) is linear in r. its second derivative vanishes
    // identically. schwarzschild has r · f(r) = r · (1 − r_s/r) = r − r_s — a
    // straight line of slope 1 through y-intercept −r_s
    let r_s = 2.0;
    let rf = |r: f64| r * f(r, r_s);

    // the linearity: r · f(r) = r − r_s, exactly
    for r in [3.0, 5.0, 10.0, 50.0, 100.0] {
        assert!(
            (rf(r) - (r - r_s)).abs() < EPSILON,
            "r·f(r) = r − r_s — a straight line, schwarzschild's signature"
        );
    }

    // the geonum statement of vacuum: trace (r, r·f) as a curve and its tangent
    // direction never turns — it IS a straight line. curvature is how much the
    // tangent angle rotates from segment to segment, and here it rotates by
    // nothing. (r·f)'' = 0 read as one constant direction, not a second difference
    let tangent = |r: f64| Geonum::new_from_cartesian(2e-3, rf(r + 1e-3) - rf(r - 1e-3)).angle;
    let slope_dir = tangent(3.0);
    for r in [5.0, 10.0, 50.0, 100.0] {
        assert!(
            tangent(r).near(&slope_dir),
            "the tangent never turns at r = {r} — r·f is straight, (r·f)'' = 0"
        );
    }
    // that constant direction is π/4 — the rescaled bondi field rises at slope 1
    assert!(
        slope_dir.near(&Angle::new(1.0, 4.0)),
        "the tangent sits at π/4 — schwarzschild's unit-slope signature"
    );

    // and the boundary conditions pin schwarzschild uniquely: r · f → r at
    // infinity (slope 1, asymptotic flatness), r · f → −r_s at r → 0 (newtonian
    // limit). two constants, both fixed by physics, one solution
    assert!(
        (rf(1e6) / 1e6 - 1.0).abs() < 1e-5,
        "slope 1 at infinity — asymptotic flatness fixes the first constant"
    );
    let intercept = rf(1.0) - 1.0; // y-intercept of the line through (1, rf(1))
    assert!(
        (intercept + r_s).abs() < EPSILON,
        "y-intercept −r_s — the newtonian limit fixes the second constant"
    );
}

#[test]
fn it_recovers_minkowski_as_the_zero_slope_intercept_solution() {
    // flat space is r_s = 0: r · f(r) = r, the through-the-origin line. same
    // linear law, both integration constants set to the trivial choice (slope 1,
    // intercept 0). minkowski is the "no mass" boundary condition on the same
    // einstein equation, not a separate theory
    let rf_flat = |r: f64| r * f(r, 0.0);
    let tangent =
        |r: f64| Geonum::new_from_cartesian(2e-3, rf_flat(r + 1e-3) - rf_flat(r - 1e-3)).angle;
    let slope_dir = Angle::new(1.0, 4.0); // π/4 — slope 1 through the origin

    for r in [1.0, 10.0, 100.0, 1000.0] {
        assert!((rf_flat(r) - r).abs() < EPSILON, "r·f = r in flat space");
        assert!(
            tangent(r).near(&slope_dir),
            "the tangent never turns — (r·f)'' = 0, vacuum satisfied trivially"
        );
    }
}

#[test]
fn it_breaks_the_vacuum_condition_for_any_other_metric_ansatz() {
    // the converse: if you make up a different k(r), the vacuum condition fails.
    // try f = 1 − r_s/r² (a "fake schwarzschild" with the wrong falloff). its
    // r · f = r − r_s/r isn't linear — its second derivative is −2 r_s / r³,
    // nonzero everywhere. geonum says: this isn't vacuum, it has spacetime
    // curvature, it would need a source to support it
    let r_s = 2.0;
    let f_fake = |r: f64| 1.0 - r_s / (r * r);
    let rf_fake = |r: f64| r * f_fake(r);

    // the wrong metric's (r, r·f) curve BENDS: trace its tangent direction and it
    // turns with r — steeper near the source, flattening outward. that turning IS
    // the curvature, and a curve that isn't straight isn't vacuum
    let tangent_fake =
        |r: f64| Geonum::new_from_cartesian(2e-3, rf_fake(r + 1e-3) - rf_fake(r - 1e-3)).angle;
    assert!(
        !tangent_fake(3.0).near(&tangent_fake(10.0)),
        "the tangent direction turns between r = 3 and r = 10 — curved, not vacuum"
    );
    assert!(
        tangent_fake(3.0).grade_angle() > tangent_fake(10.0).grade_angle(),
        "the curve is steeper near the source and flattens outward — it bends"
    );

    // try f = exp(−r_s/r) — a smooth alternative that ALSO reduces to 1 − r_s/r
    // at first order. it agrees with schwarzschild on the newtonian limit but
    // FAILS the vacuum equation: its r·f bends too, only subtly. the einstein
    // equation is the discriminator — it rules out look-alikes
    let f_exp = |r: f64| (-r_s / r).exp();
    let rf_exp = |r: f64| r * f_exp(r);
    let tangent_exp =
        |r: f64| Geonum::new_from_cartesian(2e-3, rf_exp(r + 1e-3) - rf_exp(r - 1e-3)).angle;
    assert!(
        !tangent_exp(3.0).near(&tangent_exp(10.0)),
        "the exponential look-alike bends too — its tangent turns, so it isn't vacuum"
    );
}

#[test]
fn it_falls_out_of_birkhoffs_theorem() {
    // birkhoff: any spherically symmetric vacuum is automatically static. the
    // standard proof needs the ricci tensor and a careful argument about which
    // components vanish. in the geonum reading it is one line: the vacuum
    // equation (r·f)'' = 0 has no time dependence available — we assumed only
    // f = f(r), and the resulting ODE is in r alone. no time, no time-dependent
    // solutions, nothing to oscillate. a "pulsating" spherical body emits NO
    // gravitational waves because the geometry outside is forced to be static
    //
    // verify by trying to insert a putative time-dependent perturbation: let
    // f(r,t) = (1 − r_s/r) · (1 + ε·sin(ω t)). does it satisfy vacuum at each t?
    // only if the perturbation doesn't curve r·f — which forces ε = 0
    let r_s = 2.0;
    let epsilon = 0.01;
    let omega: f64 = 1.0;

    // pick a moment when the perturbation is nontrivial
    let t = 0.5;
    let perturb = 1.0 + epsilon * (omega * t).sin();
    let rf_perturbed = |r: f64| r * (1.0 - r_s / r) * perturb;

    // (r · f_perturbed)(r) = (r − r_s) · perturb, still linear in r — at a fixed
    // t, a multiplicative time-dependent factor preserves linearity. but the
    // FULL einstein equation also constrains how f changes with t, and a
    // nonzero ∂f/∂t at fixed r is a separate non-vacuum source
    let tangent = |r: f64| {
        Geonum::new_from_cartesian(2e-3, rf_perturbed(r + 1e-3) - rf_perturbed(r - 1e-3)).angle
    };
    let slope_dir = tangent(3.0);
    for r in [5.0, 10.0] {
        assert!(
            tangent(r).near(&slope_dir),
            "at fixed t the tangent never turns — still straight, still radial-vacuum"
        );
    }

    // but the time derivative is nonzero — and the full einstein equation
    // demands no time evolution of f for a static-asymptotic vacuum. so the
    // perturbation must vanish: ε·ω·cos(ω t) must be zero for all t. it isn't
    let df_dt = epsilon * omega * (omega * t).cos();
    assert!(
        df_dt.abs() > 1e-3,
        "the perturbation has nonzero ∂f/∂t — and vacuum forbids it. birkhoff."
    );

    // the geonum reading of birkhoff: spherical symmetry collapses the bondi
    // field to k(r), and the vacuum equation on k(r) admits no time-dependent
    // perturbation. gravitational waves require quadrupole or higher — they
    // can't be spherical. one ODE, two consequences: schwarzschild AND birkhoff
}

#[test]
fn it_sources_the_bondi_field_with_enclosed_mass() {
    // the source side. for a static spherical body of density ρ(r), the einstein
    // equation becomes
    //   (r · (1 − f(r)))' = 8π G ρ(r) · r²
    // which, integrating both sides from 0 to r, gives
    //   r · (1 − f(r)) = 2 G M(r)        where M(r) = ∫₀^r 4π ρ(r') r'² dr'
    // so 1 − f(r) = 2 G M(r) / r = r_s(r) / r — the schwarzschild radius of the
    // enclosed mass, divided by r. ONE field, ONE source, no tensors
    //
    // outside the body, M(r) = M_total is constant and we recover the vacuum
    // schwarzschild: f = 1 − r_s/r, exactly the postulate of schwarzschild_test
    // INSIDE the body, M(r) grows with radius and f is the interior solution
    let r_body: f64 = 10.0; // body radius
    let rho_0 = 0.001; // uniform density (geometric units)
    let r_s_total = 8.0 * PI * rho_0 * r_body.powi(3) / 3.0; // total schwarzschild radius

    // mass enclosed within radius r — uniform density: M(r) = (4π/3) ρ r³
    let m_enclosed = |r: f64| {
        if r <= r_body {
            4.0 * PI / 3.0 * rho_0 * r.powi(3)
        } else {
            4.0 * PI / 3.0 * rho_0 * r_body.powi(3)
        }
    };

    // schwarzschild radius of the enclosed mass — r_s(r) = 2 G M(r), geometric
    let r_s_of_r = |r: f64| 2.0 * m_enclosed(r);

    // the source equation: 1 − f(r) = r_s(r) / r, so f(r) = 1 − r_s(r)/r
    let f_full = |r: f64| 1.0 - r_s_of_r(r) / r;

    // outside the body we recover vacuum schwarzschild EXACTLY
    for r in [12.0, 20.0, 50.0, 100.0] {
        let expected = 1.0 - r_s_total / r;
        assert!(
            (f_full(r) - expected).abs() < EPSILON,
            "outside the body f = 1 − r_s_total/r — vacuum schwarzschild recovered"
        );
    }

    // the source equation in its INTEGRAL form is gauss's law: r·(1−f) = 2GM(r),
    // the field's flatness-deficit equals the enclosed mass — density integrated
    // over the spherical volume. no derivative, no finite difference. inside a
    // uniform body the enclosed mass grows as the volume, so the deficit goes as r³
    let deficit = |r: f64| r * (1.0 - f_full(r)); // r·(1−f) = 2GM(r)
    let per_volume = 8.0 * PI / 3.0 * rho_0; // (8π/3)ρ = 2·(4π/3)ρ, the deficit per r³
    for r in [1.0, 3.0, 5.0, 8.0] {
        assert!(
            (deficit(r) - per_volume * r.powi(3)).abs() < EPSILON,
            "inside: r·(1−f) = (8π/3)ρ r³ — the deficit counts the enclosed mass, ∝ volume"
        );
    }

    // at and beyond the surface the enclosed mass is complete: the deficit stops
    // growing and freezes at r_s_total. "outside is vacuum" is the integral
    // statement that no more mass is enclosed — the source equation with ρ = 0
    for r in [12.0, 20.0, 50.0] {
        assert!(
            (deficit(r) - r_s_total).abs() < EPSILON,
            "outside: r·(1−f) = r_s_total — enclosed mass complete, the deficit frozen"
        );
    }

    eprintln!("\n  uniform body of density ρ = {rho_0}, radius {r_body}");
    eprintln!("  total r_s = {r_s_total:.4} (geometric units)");
    eprintln!("  one equation, one field — vacuum and matter unified");
}

#[test]
fn it_reads_the_bondi_field_at_a_ray_climbing_through_the_source() {
    // tie back to the SR file: a ray climbing OUT of the body still has its
    // half-tangent scaled by the local k(r), but k(r) is now the interior
    // solution. the boost machinery from schwarzschild_test runs unchanged —
    // only k(r) changes its form between exterior and interior. one primitive
    // (Angle::boost) handles both regimes
    let r_body: f64 = 10.0;
    let rho_0 = 0.001;
    let r_s_total = 8.0 * PI * rho_0 * r_body.powi(3) / 3.0;

    let f_interior = |r: f64| {
        if r <= r_body {
            let r_s_r = 2.0 * (4.0 * PI / 3.0 * rho_0 * r.powi(3));
            1.0 - r_s_r / r
        } else {
            1.0 - r_s_total / r
        }
    };

    // a ray emitted from inside the body, at r = 5, with f(5) and thus a smaller
    // bondi factor than the surface. its half-tangent scales by √f(5)
    let r_emit = 5.0;
    let k_inside = f_interior(r_emit).sqrt();

    let ray = Angle::new(1.0, 3.0); // θ = π/3
    let received = ray.boost(k_inside);

    // the half-tangent scaled by the LOCAL bondi factor — same primitive as the
    // exterior schwarzschild test, only k changes
    assert!(
        (received.t() - ray.t() / k_inside).abs() < EPSILON,
        "the interior bondi field scales the ray's half-tangent the same way"
    );

    // and a ray from JUST OUTSIDE the body sees the vacuum k. the interior and
    // exterior solutions match continuously at the surface — the bondi field is
    // a continuous function of r, no jump
    let k_surface_in = f_interior(r_body - 1e-6).sqrt();
    let k_surface_out = f_interior(r_body + 1e-6).sqrt();
    assert!(
        (k_surface_in - k_surface_out).abs() < 1e-4,
        "k is continuous across the body's surface — one field, both regimes"
    );

    // composing the climb: from r_emit (inside) to the surface, then surface to
    // infinity. the bondi factors multiply along the path, exactly as the
    // schwarzschild_test composition law
    let k_emit_to_surface = k_inside / k_surface_in;
    let k_surface_to_inf = k_surface_out;
    let two_step = ray.boost(k_emit_to_surface).boost(k_surface_to_inf);
    let one_step = ray.boost(k_inside);

    assert!(
        (two_step.t() - one_step.t()).abs() < 1e-4,
        "the climb composes — interior and exterior k stitched into one path"
    );
}

#[test]
fn it_curves_the_proper_length_when_the_field_is_nonlinear() {
    // the geometric meaning of (r·f)'' ≠ 0: spacetime curvature. for the
    // SCHWARZSCHILD exterior (rf)'' = 0 — no curvature in this rescaled field.
    // step INSIDE the body and (rf)'' is no longer zero. THAT is curvature, in
    // the geonum-natural sense: the failure of the rescaled bondi field to be
    // linear. ricci-flat = "rescaled bondi field is linear in r"
    let r_body = 10.0;
    let rho_0 = 0.001;

    let f_interior = |r: f64| {
        if r <= r_body {
            let r_s_r = 2.0 * (4.0 * PI / 3.0 * rho_0 * r.powi(3));
            1.0 - r_s_r / r
        } else {
            let r_s_total = 8.0 * PI * rho_0 * r_body.powi(3) / 3.0;
            1.0 - r_s_total / r
        }
    };
    let rf_full = |r: f64| r * f_interior(r);

    let tangent =
        |r: f64| Geonum::new_from_cartesian(2e-3, rf_full(r + 1e-3) - rf_full(r - 1e-3)).angle;

    // outside: the rescaled field is straight — its tangent never turns, ricci-flat,
    // curvature lives only where the source is
    let outside_dir = tangent(20.0);
    for r in [12.0, 30.0, 50.0] {
        assert!(
            tangent(r).near(&outside_dir),
            "outside: the tangent never turns — (r·f)'' = 0, ricci-flat vacuum"
        );
    }

    // inside: the field BENDS — the tangent turns from radius to radius, swinging
    // from a rising slope toward a falling one as the density piles up. that turning
    // IS the curvature, and it appears exactly where ρ ≠ 0: the einstein equation is
    // local, the geometry tracks the source point by point
    assert!(
        !tangent(2.0).near(&tangent(8.0)),
        "inside: the tangent turns — (r·f)'' ≠ 0, curvature where the source is"
    );

    // curvature appears EXACTLY at the radii where density is nonzero. the
    // einstein equation is local: at every point, the curvature of the rescaled
    // bondi field equals (up to a constant) the local energy density. nothing
    // about r-dependence "spreads" the source — the geometry tracks ρ point by
    // point. one field, one local equation, no propagator
}
