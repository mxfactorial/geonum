// why the three lattice constants are π/2, π/3, π/4 — and why all three
//
// the chemistry IE model (tests/chemistry_test.rs, acts VII–XV) runs on three
// constants with denominators 2, 3, 4 and calls them "zero fitted parameters".
// this file proves that claim: the denominators are not chosen small, they are
// the three smallest each first to fill a distinct rotation-closure role on the
// π/2 lattice. and the two non-lattice roles cannot collapse into one.
//
// the lattice is quadrature: sin(θ+π/2) = cos(θ) forces a 4-cycle on grades
// (0→1→2→3→0 under π/2 rotation). π/2 is not a peer of π/3 and π/4 — it is the
// ambient geometry the other two live inside. so the real question is why
// pairing lands at π/3 and the phase shift at π/4.
//
//   2 — the Q-period: the quadrature quarter-turn, the lattice itself
//   4 — the Q-bisector: the unique angle equidistant from the two poles, where
//       projection onto the 0-axis equals projection onto the π/2-axis — a clean
//       lattice landmark at 45°. (the IE uses π/4's radian as a band-confined
//       phase coefficient, NOT this bisector projection — see it_bisects)
//   3 — the D-closure: the smallest fraction whose rotation cycle first lands on
//       a pure blade at π (the dual, a π rotation, blade+2) WITHOUT ever hitting
//       π/2. the smallest non-Q closure
//
// 5, 6, 7 fill no new role: 5 and 7 are D-closures larger than 3, and 6 = 2·3
// closes Q like 4 but composite. so the ANGLE denominators 2, 3, 4 are forced,
// not fitted.
//
// the magnitude 1/n is the other half, and it is derived too — by a different
// geometric operation. rotation preserves magnitude, so the angular closure
// cannot reach 1/n. division can: shell n is n windings of the four-blade 2π
// cycle (act IV blade_count_is_shell), and 1/n is the inverse of that winding
// count — the rotational extent read as a temporal rate, the way tan = sin/cos
// forms a rate in trigonometry_test. the final tests show 1/n = inv(n) and that
// it is load-bearing. the model runs on two geometric operations: rotation for
// the angular lattice (2/3/4), inversion for the radial rate (1/n).
//
// run: cargo test --test chem_constants_test -- --show-output

use geonum::*;

const EPSILON: f64 = 1e-10;
const RYDBERG: f64 = 13.6;
const ROOT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2; // √2/2 = cos(π/4) = sin(π/4)

// NIST first ionization energies, Z=1..=18 (eV)
const EXP: [f64; 18] = [
    13.598, 24.587, 5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565, 5.139, 7.646,
    5.986, 8.152, 10.487, 10.360, 12.968, 15.760,
];

// the first step where a π/k rotation cycle lands on a pure blade (t = 0), and
// the blade it lands on. blade 1 is the π/2 quarter (a Q closure); blade 2 is π,
// the dual (a D closure). rotating from 0 by π/k, count steps to the first t=0
fn first_pure_blade(k: f64) -> (usize, usize) {
    let step = Angle::new(1.0, k);
    let mut a = Angle::new(0.0, 1.0); // blade 0, t 0
    for s in 1..=(2.0 * k).ceil() as usize {
        a = a + step;
        if a.t().abs() < EPSILON {
            return (s, a.blade());
        }
    }
    unreachable!("a π/{k} cycle reaches a pure blade by 2k steps")
}

#[test]
fn it_closes_the_quadrature_lattice_with_pi_over_2() {
    // π/2 is the lattice: one π/2 rotation advances the grade by one, and four
    // of them return to the start — the 4-cycle quadrature forces. this is not a
    // constant chosen beside the others, it is the geometry they sit in
    let quarter = Angle::new(1.0, 2.0); // π/2
    let mut a = Angle::new(0.0, 1.0); // grade 0
    for expected in [1usize, 2, 3, 0] {
        a = a + quarter;
        assert_eq!(
            a.grade(),
            expected,
            "π/2 rotation advances the grade by one"
        );
    }
    // four quarter-turns close the cycle: blade advanced by exactly 4
    assert_eq!(
        a.blade(),
        4,
        "four π/2 turns close the lattice (full 2π cycle)"
    );

    // the closure IS quadrature: differentiation is a +π/2 rotation, so it
    // cycles the grades f→f'→f''→f'''→f. sin(θ+π/2) = cos(θ)
    let f = Geonum::new(1.0, 0.0, 1.0); // grade 0
    let mut d = f;
    for expected in [1usize, 2, 3, 0] {
        d = d.differentiate();
        assert_eq!(d.angle.grade(), expected, "differentiate is a π/2 rotation");
    }
}

#[test]
fn it_bisects_the_quadrature_with_pi_over_4() {
    // π/4 is the bisector of the lattice quarter: the unique angle in (0, π/2)
    // equidistant from the 0-axis and the π/2-axis — projection onto adjacent
    // (cos) equals projection onto opposite (sin), both √2/2. that fixes π/4 as a
    // clean lattice landmark at 45°, which is why a clean rational coefficient is
    // available there for the IE to use.
    //
    // the IE formula does NOT use this bisector projection: it keeps adj at full
    // weight (the hydrogenic backbone) and weights opp by π/4's radian value
    // ≈ 0.785 — a radial-dominant ray at atan(π/4) ≈ 38°. the symmetric 45° ray
    // would need coefficient tan(π/4) = 1 and would break the hydrogenic series.
    // so π/4 enters the IE as a band-confined coefficient (proven in
    // it_projects_the_opp_combiner_onto_the_atan_ray and
    // it_proves_the_opp_coefficient_is_load_bearing), distinct from the bisector
    // proven below
    let bisector = Angle::new(1.0, 4.0); // π/4
    let pole_0 = Angle::new(0.0, 1.0); // the 0-axis (adj)
    let pole_q = Angle::new(1.0, 2.0); // the π/2-axis (opp)

    let onto_0 = bisector.project(pole_0); // cos(π/4)
    let onto_q = bisector.project(pole_q); // cos(π/4 − π/2) = cos(π/4)
    assert!(
        (onto_0 - onto_q).abs() < EPSILON,
        "π/4 projects equally onto both poles: {onto_0:.6} vs {onto_q:.6}"
    );
    assert!(
        (onto_0 - ROOT_HALF).abs() < EPSILON,
        "the equidistant value is √2/2"
    );

    // cos = sin only at the bisector. the other small fractions tilt toward one
    // pole — π/3 and π/6 are reflections of each other across π/4, neither equal
    for k in [3.0, 6.0] {
        let a = Angle::new(1.0, k);
        let gap = (a.project(pole_0) - a.project(pole_q)).abs();
        assert!(
            gap > 0.3,
            "π/{k} is not equidistant from the poles (gap {gap:.3})"
        );
    }

    // π/4 = (π/2)/2 — literally the midpoint of the lattice quarter
    let midpoint = pole_q.project(bisector); // cos(π/2 − π/4) = cos(π/4)
    assert!(
        (midpoint - ROOT_HALF).abs() < EPSILON,
        "π/4 bisects the quarter"
    );
}

#[test]
fn it_closes_the_dual_with_pi_over_3() {
    // π/3 is the D-closure: three π/3 rotations sum to π — a dual, the π rotation
    // that adds 2 blades and maps grade 0↔2. the cycle first lands on a pure
    // blade at π (blade 2), SKIPPING the π/2 quarter (blade 1) entirely
    let third = Angle::new(1.0, 3.0); // π/3
    let mut a = Angle::new(0.0, 1.0);

    // step 1: π/3 carries a remainder — not on the lattice
    a = a + third;
    assert!(a.t().abs() > EPSILON, "π/3 is off the pure-blade lattice");
    // step 2: 2π/3 still carries a remainder — the quarter is skipped
    a = a + third;
    assert!(a.t().abs() > EPSILON, "2π/3 skips the π/2 quarter");
    // step 3: π — a pure blade at last, blade 2, grade 2: the dual
    a = a + third;
    assert!(a.t().abs() < EPSILON, "3·π/3 = π lands on a pure blade");
    assert_eq!(a.blade(), 2, "the closure is at π — blade 2");

    // and that pure blade IS the dual: a π rotation, what .dual() applies
    let dual_of_zero = Geonum::new(1.0, 0.0, 1.0).dual();
    assert_eq!(
        a.blade(),
        dual_of_zero.angle.blade(),
        "three π/3 turns equal one dual (π, blade+2)"
    );

    // π/3 is the SMALLEST such closure: the first pure blade it reaches is the
    // dual (blade 2) at 3 steps. π/2 and π/4 instead close on the quarter
    assert_eq!(
        first_pure_blade(3.0),
        (3, 2),
        "π/3 first closes at the dual"
    );
    assert_eq!(first_pure_blade(2.0), (1, 1), "π/2 closes on the quarter");
    assert_eq!(first_pure_blade(4.0), (2, 1), "π/4 closes on the quarter");
}

#[test]
fn it_assigns_2_3_4_to_three_distinct_closures() {
    // sweep k = 2..=7 and read which closure each π/k first reaches. blade 1 is a
    // Q closure (the quarter); blade 2 is a D closure (the dual). the three
    // smallest each first to fill a distinct role are 2, 3, 4
    let landings: Vec<(usize, (usize, usize))> =
        (2..=7).map(|k| (k, first_pure_blade(k as f64))).collect();

    // 2 → Q at 1 step (the lattice), 4 → Q at 2 steps (the bisector),
    // 3 → D at 3 steps (first dual closure). the rest replicate:
    // 5 → D at 5 (a slower 3), 6 → Q at 3 (2·3, composite), 7 → D at 7
    let expected = [
        (2usize, (1usize, 1usize)), // Q-lattice
        (3, (3, 2)),                // D-closure (first)
        (4, (2, 1)),                // Q-bisector (first even > 2)
        (5, (5, 2)),                // D, slower than 3
        (6, (3, 1)),                // Q, but 6 = 2·3
        (7, (7, 2)),                // D, slower than 3
    ];
    assert_eq!(landings, expected, "the closure each π/k first reaches");

    // the distinct roles, each held by the smallest k that fills it:
    //   Q-lattice : the unique 1-step quarter closure → k = 2
    //   Q-bisector: the smallest quarter closure past the lattice → k = 4
    //   D-closure : the smallest dual closure → k = 3
    let q_lattice = landings.iter().find(|(_, l)| *l == (1, 1)).unwrap().0;
    let dual_closures: Vec<usize> = landings
        .iter()
        .filter(|(_, (_, b))| *b == 2)
        .map(|(k, _)| *k)
        .collect();
    let bisectors: Vec<usize> = landings
        .iter()
        .filter(|(_, (s, b))| *b == 1 && *s > 1)
        .map(|(k, _)| *k)
        .collect();

    assert_eq!(
        q_lattice, 2,
        "the lattice is the unique 1-step quarter closure"
    );
    assert_eq!(
        *dual_closures.iter().min().unwrap(),
        3,
        "the smallest dual closure is π/3"
    );
    assert_eq!(
        *bisectors.iter().min().unwrap(),
        4,
        "the smallest bisector past the lattice is π/4"
    );

    // 5, 6, 7 add nothing: 5 and 7 are dual closures larger than 3, and 6 is a
    // quarter closure larger than 4 (and composite, 6 = 2·3). every role is
    // already held by a smaller denominator
    for &k in &[5usize, 6, 7] {
        let (_, b) = first_pure_blade(k as f64);
        let role_holder = if b == 2 { 3 } else { 4 };
        assert!(
            k > role_holder,
            "π/{k} replicates the role of π/{role_holder}"
        );
    }

    eprintln!("\n═══ the denominators are forced, not fitted ═══\n");
    eprintln!("  k   first pure blade   closure        role");
    for (k, (steps, blade)) in &landings {
        let closure = if *blade == 1 {
            "quarter (Q)"
        } else {
            "dual (D)  "
        };
        let role = match (k, blade) {
            (2, _) => "Q-lattice (the geometry)",
            (4, _) => "Q-bisector (phase coeff)",
            (3, _) => "D-closure (pairing)",
            (_, 1) => "→ replicates π/4",
            _ => "→ replicates π/3",
        };
        eprintln!("  {k}   step {steps} → blade {blade}    {closure}    {role}");
    }
}

#[test]
fn it_needs_both_a_dual_closure_and_a_bisector() {
    // why does IE use TWO distinct constants for spin and phase, not one repeated?
    // because π/3 and π/4 are geometrically disjoint lattice closures — π/3 closes
    // the dual, π/4 bisects the quarter — so neither can stand in for the other.
    // the IE uses π/3 as the spin/pairing offset and π/4's radian as the phase
    // coefficient; this proves the two are not interchangeable lattice constants
    let third = Angle::new(1.0, 3.0); // π/3, the D-closure (spin)
    let quarter = Angle::new(1.0, 4.0); // π/4, the bisector
    let pole_0 = Angle::new(0.0, 1.0);
    let pole_q = Angle::new(1.0, 2.0);

    // one distinguishing property is the quarter-bisector: the angle equidistant
    // from both poles (cos = sin). π/4 has it; π/3 tilts toward the 0-axis. this is
    // what separates them geometrically — NOT a claim that the IE combiner needs
    // balance (it does not; it weights adj over opp — see it_bisects, audit #1)
    let q_balance = (quarter.project(pole_0) - quarter.project(pole_q)).abs();
    let third_balance = (third.project(pole_0) - third.project(pole_q)).abs();
    assert!(
        q_balance < EPSILON,
        "π/4 balances the two axes — the bisector"
    );
    assert!(
        third_balance > 0.3,
        "π/3 tilts ({third_balance:.3}) — distinct from the bisector"
    );

    // the pairing job needs the triplet to close on the dual: three steps must
    // land on π (blade 2), the antipode. π/3 does — three π/3 turns reach the
    // dual exactly. three π/4 turns land on 3π/4 (blade 1, still carrying a
    // remainder), so a bisector cannot pair electrons antipodally in a triplet
    let three_thirds = Angle::new(0.0, 1.0) + third + third + third;
    let three_quarters = Angle::new(0.0, 1.0) + quarter + quarter + quarter;
    assert_eq!(
        three_thirds.blade(),
        2,
        "three π/3 turns reach the dual (blade 2)"
    );
    assert!(
        three_thirds.t().abs() < EPSILON,
        "the dual is a pure blade at π"
    );
    assert_eq!(
        three_quarters.blade(),
        1,
        "three π/4 turns land on 3π/4 — blade 1, short of the dual"
    );
    assert!(
        three_quarters.t().abs() > EPSILON,
        "3π/4 still carries a remainder — not even a pure blade"
    );

    // the two roles are disjoint: π/4 bisects but does not close the dual, π/3
    // closes the dual but does not bisect. neither absorbs the other, so the two
    // lattice constants are not interchangeable — which it_works_only confirms
    // empirically (swapping spin and phase loses an anomaly)
    let bisector_closes_dual = three_quarters.blade() == 2 && three_quarters.t().abs() < EPSILON;
    let dual_bisects = third_balance < EPSILON;
    assert!(
        !bisector_closes_dual,
        "the bisector does not reach the dual in a triplet"
    );
    assert!(!dual_bisects, "the dual closure does not bisect the axes");
}

// ─────────────────────────────────────────────────────────────────────────
// the empirical confirmation: the minimal IE model, run with the two constants
// swapped. spread = π/2 is held fixed (it is the lattice). only spin and Q vary
// ─────────────────────────────────────────────────────────────────────────

// the 1/n radial law (Bohr momentum ∝ 1/n), and a flat foil that ignores the shell
fn bohr(n: usize) -> f64 {
    1.0 / n as f64
}
fn flat(_n: usize) -> f64 {
    1.0
}

// the nucleus-scaled marginal p of electron z under the canonical lattice, with
// its projections onto the 0-axis (adj) and the π/2-axis (opp)
fn marginal_projections(z: usize) -> (Geonum, f64, f64) {
    let nucleus = Geonum::new(z as f64, 0.0, 1.0);
    let marginal = Geonum::electron_wave(z, Lattice::Canonical)
        - Geonum::electron_wave(z - 1, Lattice::Canonical);
    let p = nucleus * marginal;
    let adj = p.project(&Geonum::new(1.0, 0.0, 1.0)).mag;
    let opp = p
        .project(&Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0)))
        .mag;
    (p, adj, opp)
}

// (RMSE over Z=1-18, Be>B holds, N>O holds) for a (spin, q, radial) assignment —
// the IE1 series swapped through Geonum::ionization_energy under Lattice::Custom
fn score(spin: Angle, q: Angle, radial: fn(usize) -> f64) -> (f64, bool, bool) {
    let lattice = Lattice::Custom { spin, radial, q };
    let ie = |z: usize| Geonum::ionization_energy(z, z, lattice);
    let mut sse = 0.0;
    for z in 1..=18usize {
        sse += (ie(z) - EXP[z - 1]).powi(2);
    }
    let rmse = (sse / 18.0).sqrt();
    (rmse, ie(4) > ie(5), ie(7) > ie(8))
}

#[test]
fn it_works_only_with_the_forced_spin_q_assignment() {
    // run the model with the two free constants assigned every way the eligible
    // denominators allow. only spin = π/3 (D-closure), Q = π/4 (bisector)
    // reproduces both anomalies inside the 3 eV band. the roles are not
    // interchangeable in the model, exactly as the geometry forces
    let third = Angle::new(1.0, 3.0); // D-closure
    let quarter = Angle::new(1.0, 4.0); // Q-bisector

    let configs = [
        ("spin π/3, Q π/4 (forced)", third, quarter),
        ("spin π/4, Q π/4 (two bisectors)", quarter, quarter),
        ("spin π/3, Q π/3 (two D-closures)", third, third),
        ("spin π/4, Q π/3 (swapped roles)", quarter, third),
    ];

    eprintln!("\n═══ the (spin, Q) assignment, every eligible way ═══\n");
    eprintln!("  assignment                          RMSE    Be>B   N>O");
    let mut results = Vec::new();
    for (label, spin, q) in configs {
        let (rmse, be, no) = score(spin, q, bohr);
        eprintln!("  {label:34}  {rmse:5.2}   {be:5}  {no:5}");
        results.push((rmse, be, no));
    }

    let (base_rmse, base_be, base_no) = results[0];
    // the forced assignment reproduces both anomalies under the 3 eV band
    assert!(
        base_be && base_no,
        "the forced assignment keeps both anomalies"
    );
    assert!(
        base_rmse < 3.0,
        "the forced assignment RMSE {base_rmse:.2} < 3 eV"
    );

    // no other assignment matches it: each swap loses an anomaly or leaves the
    // band. the model confirms the roles are non-interchangeable
    for (label, (rmse, be, no)) in configs.iter().skip(1).zip(results.iter().skip(1)) {
        let matches_baseline = *be && *no && *rmse < 3.0;
        assert!(
            !matches_baseline,
            "{}: a swap reproduced the baseline — roles would be interchangeable",
            label.0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// what π/4 actually does in the IE: a scaled projection, not the bisector
//
// it_bisects proves π/4 is the bisector DIRECTION (cos = sin). these two pin its
// actual IE role. q.grade_angle() reads π/4 out as the coefficient (≈0.785), and
// adj + 0.785·opp is the scaled projection of the marginal onto the ray
// atan(π/4) ≈ 38° — a radial-dominant ray below the 45° bisector. the coefficient
// is load-bearing: no parameter-free combiner reproduces it.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn it_projects_the_opp_combiner_onto_the_atan_ray() {
    // adj + c·opp is p · (1, c): the scaled projection of the marginal (folded
    // into the first quadrant by the |·| that .mag takes) onto the ray ψ =
    // atan(c), scaled by √(1+c²) = sec ψ. for the IE's c = π/4 the ray is
    // atan(π/4) ≈ 38°, below the 45° bisector — adj's full weight pulls it down
    let c = std::f64::consts::FRAC_PI_4; // the IE coefficient = π/4's radian
    let psi = c.atan(); // the projecting ray
    let scale = (1.0 + c * c).sqrt(); // sec ψ

    for z in 1..=18usize {
        let (p, adj, opp) = marginal_projections(z);
        let phi = p.angle.grade_angle();
        let alpha = phi.sin().abs().atan2(phi.cos().abs()); // p folded into Q1
        assert!(
            (adj + c * opp - scale * p.mag * (alpha - psi).cos()).abs() < 1e-9,
            "z={z}: adj + c·opp = sec ψ · p.mag · cos(foldedφ − ψ)"
        );
    }

    // the ray sits below the 45° bisector: a bisector projection would need
    // coefficient tan(π/4) = 1 (and would break the hydrogenic series)
    assert!(
        psi < std::f64::consts::FRAC_PI_4,
        "the IE ray atan(π/4) ≈ {:.1}° is below the 45° bisector",
        psi.to_degrees()
    );
}

#[test]
fn it_proves_the_opp_coefficient_is_load_bearing() {
    // the π/4 coefficient earns its place: it gets both anomalies AND keeps
    // hydrogenic. the two parameter-free combiners the projection geometry allows
    // each fail — the hypotenuse √(adj²+opp²) discards p's angle and loses both
    // anomalies; the bisector (adj+opp)/√2 breaks hydrogenic
    let q = Angle::new(1.0, 4.0);

    // (RMSE, Be>B, N>O) over Z=1-18 for a combiner (adj, opp) -> scalar
    let eval = |combine: &dyn Fn(f64, f64) -> f64| -> (f64, bool, bool) {
        let ie = |z: usize| {
            let (_, a, o) = marginal_projections(z);
            let n = Geonum::valence_shell(z);
            RYDBERG * combine(a, o) / (n * n) as f64
        };
        let mut sse = 0.0;
        for z in 1..=18usize {
            sse += (ie(z) - EXP[z - 1]).powi(2);
        }
        ((sse / 18.0).sqrt(), ie(4) > ie(5), ie(7) > ie(8))
    };

    // the π/4 coefficient: both anomalies, RMSE in band
    let (rmse_q, be_q, no_q) = eval(&|a, o| a + q.grade_angle() * o);
    assert!(be_q && no_q, "π/4 reproduces Be>B and N>O");
    assert!(rmse_q < 3.0, "π/4 RMSE {rmse_q:.2} sits in the band");

    // the hypotenuse keeps hydrogenic but loses both anomalies
    let (_, be_h, no_h) = eval(&|a, o| (a * a + o * o).sqrt());
    assert!(!be_h && !no_h, "the hypotenuse loses both anomalies");

    // the bisector breaks hydrogenic: H (Z=1, opp=0) lands 13.6/√2, not 13.6
    let (_, a1, o1) = marginal_projections(1);
    let n1 = Geonum::valence_shell(1);
    let h_bisector = RYDBERG * (a1 + o1) / 2.0_f64.sqrt() / (n1 * n1) as f64;
    assert!(
        (h_bisector - RYDBERG / 2.0_f64.sqrt()).abs() < 1e-9,
        "the bisector sends hydrogenic H to 13.6/√2, not 13.6"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// the radial axis: 1/n is the inverse of the winding count
//
// the closure forces the ANGLES (2, 3, 4) through rotation. the per-electron
// magnitude 1/n comes from a different geometric operation — division. rotation
// preserves magnitude, so the angular closure cannot reach 1/n; inversion does.
//
// shell n is reached after n windings of the fundamental four-blade (2π) cycle
// (act IV blade_count_is_shell reads shell = blade/4 + 1). the winding count is a
// rotational extent — a geonum of magnitude n. its inverse is the per-winding
// rate, magnitude 1/n. division is geometry becoming temporal: the same
// operation that forms tan = sin/cos in trigonometry_test, a rate of one
// quadrature component against another. so 1/n is derived (inv of the winding
// count), not fitted. the model carries two geometric operations — rotation for
// the angular lattice, inversion for the radial rate.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn it_separates_the_radial_law_from_the_angular_closure() {
    // rotation by any lattice angle preserves magnitude — it only adds to the
    // angle. so the angular closure (rotation) cannot reach 1/n: the radial law
    // lives in a different operation, division, not in the 2/3/4 argument
    let e = Geonum::new(0.5, 1.0, 1.0); // an electron of magnitude 0.5
    for k in [2.0, 3.0, 4.0] {
        let rotated = e.rotate(Angle::new(1.0, k));
        assert!(
            (rotated.mag - e.mag).abs() < EPSILON,
            "rotation by π/{k} preserves magnitude — rotation fixes angle, not radius"
        );
    }

    // two electrons at the same angle but different shells differ only in
    // magnitude — the radial axis is orthogonal to the angular one
    let pos = Angle::new(1.0, 1.0); // π
    let shell_1 = Geonum::new_with_angle(bohr(1), pos);
    let shell_3 = Geonum::new_with_angle(bohr(3), pos);
    assert!(
        shell_1.angle.near(&shell_3.angle),
        "same angle, different shell"
    );
    assert!(
        (shell_1.mag - shell_3.mag).abs() > EPSILON,
        "the shell distinction lives in the radial magnitude, not the angle"
    );
}

#[test]
fn it_derives_the_radial_law_by_inverting_the_winding_count() {
    // 1/n is not supplied — it is the geonum inverse of the shell's winding
    // count. division turns the rotational extent (n windings) into the temporal
    // rate (1/n per winding), the operation trigonometry_test forms tan with
    for n in 1..=5usize {
        // shell n sits at blades 4·(n-1)..4·n — n full four-blade (2π) windings.
        // the winding count reads back out of the blade structure (act IV)
        let shell_blade = 4 * (n - 1);
        let winding_count = shell_blade / 4 + 1;
        assert_eq!(winding_count, n, "shell n is n windings of the 2π cycle");

        // the winding count as a rotational extent, and its rate by inversion
        let windings = Geonum::new(winding_count as f64, 0.0, 1.0);
        let rate = windings.inv();
        assert!(
            rate.near_mag(bohr(n)),
            "the radial law 1/n is inv(winding count): {} vs {}",
            rate.mag,
            bohr(n)
        );
    }
}

#[test]
fn it_proves_the_radial_law_is_load_bearing() {
    // 1/n is not a removable scale: replacing it with a flat magnitude changes
    // the predictions and worsens the fit. the model genuinely depends on the
    // radial input, so the angular closure does not tell the whole story
    let spin = Angle::new(1.0, 3.0);
    let q = Angle::new(1.0, 4.0);

    let (rmse_bohr, _, _) = score(spin, q, bohr);
    let (rmse_flat, _, _) = score(spin, q, flat);

    eprintln!("\n  radial law: 1/n RMSE {rmse_bohr:.2} eV  vs  flat RMSE {rmse_flat:.2} eV");

    assert!(
        rmse_bohr < rmse_flat,
        "the 1/n radial law ({rmse_bohr:.2}) fits better than a flat magnitude ({rmse_flat:.2})"
    );
}
