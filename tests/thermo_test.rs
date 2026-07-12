// cycle work is a swept wedge
//
// thermodynamics reads a heat engine's work off the area its cycle encloses in
// the PV plane — then computes that area with path integrals. the area is a
// swept wedge sum (integral_test's primitive): put the origin at the cycle's
// center and the state vector sweeps the enclosure triangle by triangle, every
// segment carrying the same orientation
//
//   - engine vs refrigerator is v ∧ w = −w ∧ v: reverse the traversal and
//     every segment wedge negates — same machine, same area, opposite
//     thermodynamic arrow
//   - on the TS plane the carnot cycle is an exact rectangle: the heats are
//     isotherm strips (wedges), the work is their gap, and the efficiency
//     1 − T_c/T_h is a magnitude ratio — the first law closes geometrically
//   - the wick rotation β = it: the same energy gap drives quantum evolution
//     as pure rotation (magnitude untouched — unitarity) and thermal
//     weighting as pure scale (angle untouched — boltzmann). temperature sits
//     a grade off time, and the swap exchanges wave_sum for total_magnitude
//
// run: cargo test --test thermo_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// the state points of a rectangular cycle, as vectors from the cycle's center
fn corners_from_center(states: &[(f64, f64)], center: (f64, f64)) -> Vec<Geonum> {
    states
        .iter()
        .map(|&(x, y)| {
            Geonum::new_from_cartesian(x, y) - Geonum::new_from_cartesian(center.0, center.1)
        })
        .collect()
}

// the swept area: half the wedge magnitudes of consecutive state vectors,
// with every segment's orientation read off the wedge's angle offset
fn swept_area(corners: &[Geonum]) -> (f64, Vec<usize>) {
    let mut wedges = Vec::new();
    let mut orientations = Vec::new();
    for k in 0..corners.len() {
        let a = corners[k];
        let b = corners[(k + 1) % corners.len()];
        let w = a.wedge(&b);
        // the wedge lands at a + b + π/2, plus π when the sweep is negative:
        // the offset's grade is the orientation — 1 forward, 3 backward
        orientations.push((w.angle - a.angle - b.angle).grade());
        wedges.push(w);
    }
    let total = GeoCollection::from(wedges).total_magnitude() * 0.5;
    (total, orientations)
}

#[test]
fn it_sweeps_cycle_work_as_an_oriented_wedge() {
    // a rectangular cycle: expand at high pressure, drop pressure, compress
    // at low pressure, rise again. V ∈ [1, 3], P ∈ [1, 2] — enclosed area 2
    let cycle = [(1.0, 2.0), (3.0, 2.0), (3.0, 1.0), (1.0, 1.0)]; // (V, P), engine order
    let corners = corners_from_center(&cycle, (2.0, 1.5));

    let (work, orientations) = swept_area(&corners);
    assert!(
        (work - 2.0).abs() < 1e-12,
        "the cycle's work is the enclosed area: (ΔV)(ΔP) = 2"
    );

    // every segment sweeps the same way — the cycle turns coherently around
    // its center, and the state vector's winding is the enclosure
    assert_eq!(
        orientations,
        vec![3, 3, 3, 3],
        "all four sweeps share one orientation — the engine's arrow"
    );
}

#[test]
fn it_flips_engine_to_refrigerator_by_orientation() {
    let cycle = [(1.0, 2.0), (3.0, 2.0), (3.0, 1.0), (1.0, 1.0)];
    let corners = corners_from_center(&cycle, (2.0, 1.5));

    // reverse the traversal: every segment wedge lands π away — v ∧ w = −w ∧ v
    // read as thermodynamics, the negation a position
    for k in 0..corners.len() {
        let a = corners[k];
        let b = corners[(k + 1) % corners.len()];
        let forward = a.wedge(&b);
        let reversed = b.wedge(&a);
        assert!(
            reversed.angle.is_opposite(&forward.angle),
            "segment {k}: reversing the sweep rotates the wedge π"
        );
        assert!(
            reversed.near_mag(forward.mag),
            "segment {k}: and costs no area"
        );
    }

    // same machine, same area, opposite arrow: the refrigerator encloses the
    // identical 2 units with every orientation flipped
    let reversed: Vec<Geonum> = corners.iter().rev().cloned().collect();
    let (work, orientations) = swept_area(&reversed);
    assert!(
        (work - 2.0).abs() < 1e-12,
        "the reversed cycle encloses the same area"
    );
    assert_eq!(
        orientations,
        vec![1, 1, 1, 1],
        "with the opposite orientation — work consumed, not delivered"
    );
}

#[test]
fn it_prices_carnot_efficiency_as_a_magnitude_ratio() {
    // on the TS plane the carnot cycle is an exact rectangle: isotherms at
    // T_h and T_c, adiabats at constant S. the heats are isotherm strips —
    // wedges of ΔS with the temperature axis
    let (t_hot, t_cold, delta_s) = (500.0, 300.0, 2.0);

    let q_hot = Geonum::new(delta_s, 0.0, 1.0).wedge(&Geonum::new(t_hot, 1.0, 2.0));
    let q_cold = Geonum::new(delta_s, 0.0, 1.0).wedge(&Geonum::new(t_cold, 1.0, 2.0));
    assert!(q_hot.near_mag(1000.0), "Q_h = T_h·ΔS — the hot strip");
    assert!(q_cold.near_mag(600.0), "Q_c = T_c·ΔS — the cold strip");

    // the work is the enclosed TS rectangle, swept around its center
    let cycle = [(1.0, t_cold), (1.0, t_hot), (3.0, t_hot), (3.0, t_cold)]; // (S, T), engine order: adiabat up, hot isotherm, adiabat down, cold isotherm
    let corners = corners_from_center(&cycle, (2.0, 400.0));
    let (work, orientations) = swept_area(&corners);

    assert!(
        (work - (t_hot - t_cold) * delta_s).abs() < 1e-9,
        "W = (T_h − T_c)·ΔS — the rectangle between the strips"
    );
    assert_eq!(orientations, vec![3, 3, 3, 3], "one coherent engine sweep");

    // the first law closes geometrically: the enclosed area is the strip gap
    assert!(
        (work - (q_hot.mag - q_cold.mag)).abs() < 1e-9,
        "W = Q_h − Q_c — no energy invented by the geometry"
    );

    // and carnot's bound is a magnitude ratio, no entropy calculus run
    let efficiency = work / q_hot.mag;
    assert!(
        (efficiency - (1.0 - t_cold / t_hot)).abs() < 1e-12,
        "η = W/Q_h = 1 − T_c/T_h = 0.4"
    );
}

#[test]
fn it_swaps_wave_sum_for_total_magnitude_under_wick() {
    // one two-level system, energy gap ε. quantum evolution applies the gap as
    // ROTATION — scale_rotate with the magnitude knob at unity, so the state
    // stays unit (unitarity). thermal weighting applies the same gap as SCALE —
    // the angle knob at zero, so the weight never turns (boltzmann)
    let epsilon = 1.3;
    let (t, beta) = (0.9, 0.7);

    let ground = Geonum::new(1.0, 0.0, 1.0);
    let excited_phase = ground.scale_rotate(1.0, Angle::new(epsilon * t / PI, 1.0));
    let excited_weight = ground.scale_rotate((-beta * epsilon).exp(), Angle::new(0.0, 1.0));

    assert!(
        excited_phase.near_mag(1.0),
        "rotation never touches magnitude — unitarity"
    );
    assert_eq!(
        excited_weight.angle, ground.angle,
        "scale never touches angle — boltzmann weighting"
    );

    // the wick rotation β = it exchanges which collection observable you
    // compute: the propagator trace is the WAVE_SUM of the phases
    // (interference), the partition function the TOTAL_MAGNITUDE of the
    // weights (no interference — nothing left to interfere)
    let quantum: GeoCollection = vec![ground, excited_phase].into();
    let thermal: GeoCollection = vec![ground, excited_weight].into();

    assert!(
        quantum
            .wave_sum()
            .near_mag(2.0 * (epsilon * t / 2.0).cos().abs()),
        "the trace interferes: |1 + e^(iεt)| = 2|cos(εt/2)|"
    );
    assert!(
        (thermal.total_magnitude() - (1.0 + (-beta * epsilon).exp())).abs() < 1e-12,
        "the partition function sums plain: Z = 1 + e^(−βε)"
    );

    // temperature sits a grade off time: same gap, same machinery, the
    // quarter turn between the two knobs is the whole difference
}
