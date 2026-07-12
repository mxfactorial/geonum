// quantum search is a quarter-turn walk
//
// grover's algorithm lives in the 2-plane spanned by |target⟩ and |rest⟩, and
// everything in it is angle arithmetic geonum performs directly:
//
//   - the state is [1, θ] with sin θ the target amplitude — one geonum where
//     the conventional register holds 2^n complex amplitudes
//   - the oracle is a reflection across |rest⟩, the diffusion a reflection
//     across the start state — and two reflections compose to a rotation by
//     twice the angle between the axes (quaternion_test), so one grover
//     iteration IS rotate(2θ)
//   - the O(√N) headline: the walk needs π/(4θ) steps with θ = asin(1/√N) —
//     the quadratic speedup is an angle budget, π/4 of arc paid 2θ at a time
//   - rotations compose, so the whole walk collapses to one multiplication:
//     search 2^100 items in O(1). the conventional simulation stores 2^100
//     amplitudes — more than atoms — to track two numbers
//
// phase estimation gets the same treatment: QPE builds an ancilla register and
// an inverse QFT to squeeze n bits out of an eigenvalue's angle, because
// measurement collapses blade to grade. storage reads the angle whole
//
// run: cargo test --test quantum_computing_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

// the search plane: |rest⟩ along angle 0, |target⟩ a quarter turn up.
// the state starts at θs = asin(1/√N) — almost all amplitude on |rest⟩
fn rest_axis() -> Geonum {
    Geonum::new(1.0, 0.0, 1.0)
}

#[test]
fn it_flips_the_target_amplitude_with_the_oracle_reflection() {
    // the oracle marks the target by flipping its amplitude's sign. as a
    // reflection across |rest⟩ the target component lands π away — the sign
    // flip is a position, read off the opp projection's angle
    let theta = (1.0 / 32.0_f64).asin(); // N = 1024
    let state = Geonum::new_with_angle(1.0, Angle::new(theta / PI, 1.0));

    let marked = state.reflect(&rest_axis());

    let before = state.opp(); // target amplitude, |sin θ| with grade-encoded sign
    let after = marked.opp();

    assert!(before.near_mag(after.mag), "the oracle costs no amplitude");
    assert!(
        after.angle.is_opposite(&before.angle),
        "the flip is a π rotation of the target component, not a sign bit"
    );
}

#[test]
fn it_composes_oracle_and_diffusion_into_one_rotation() {
    // oracle = reflect across |rest⟩ (angle 0), diffusion = reflect across the
    // start state (angle θs). two reflections compose to a rotation by twice
    // the angle between the axes: one grover iteration is rotate(2θs)
    let theta = (1.0 / 32.0_f64).asin();
    let start = Geonum::new_with_angle(1.0, Angle::new(theta / PI, 1.0));

    let state = Geonum::new_with_angle(1.0, Angle::new(2.0 * theta / PI, 1.0)); // mid-walk
    let iterated = state.reflect(&rest_axis()).reflect(&start);
    let rotated = state.rotate(start.angle + start.angle); // rotate by 2θs

    assert!(
        iterated.near_mag(rotated.mag),
        "reflections cost no amplitude"
    );
    assert_eq!(
        iterated.angle.base_angle(),
        rotated.angle.base_angle(),
        "oracle then diffusion IS the 2θ rotation"
    );
}

#[test]
fn it_walks_grover_to_the_target_in_pi_over_4_root_n_steps() {
    // N = 1024: θ = asin(1/32). the state needs to travel from θ to π/2, paying
    // 2θ per iteration — π/(4θ) ≈ (π/4)√N steps, the quadratic speedup counted
    // as an angle budget. walk it with the actual double reflection
    let n: f64 = 1024.0;
    let theta = (1.0 / n.sqrt()).asin();
    let start = Geonum::new_with_angle(1.0, Angle::new(theta / PI, 1.0));

    let optimal = (PI / (4.0 * theta) - 0.5).round() as usize;
    assert_eq!(
        optimal, 25,
        "⌊π/4·√1024⌋ steps: 25 iterations, not 512 probes"
    );

    let mut state = start;
    for _ in 0..optimal {
        state = state.reflect(&rest_axis()).reflect(&start).base_angle();
    }

    // success probability is the squared opp projection — and it matches the
    // textbook amplitude sin((2j+1)θ) the walk never computed
    let success = state.opp().mag * state.opp().mag;
    let textbook = ((2.0 * optimal as f64 + 1.0) * theta).sin().powi(2);

    assert!(
        (success - textbook).abs() < 1e-9,
        "the reflection walk lands the textbook amplitude"
    );
    assert!(
        success > 0.999,
        "25 angle additions find 1 item in 1024: p = {success:.6}"
    );
}

#[test]
fn it_searches_two_to_the_hundred_items_in_one_multiplication() {
    // N = 2^100. the conventional state vector needs 2^100 amplitudes — more
    // than atoms in the observable universe — and even the counted walk needs
    // ~9·10^14 iterations. but rotations compose: j iterations of rotate(2θ)
    // is one angle (2j+1)·θ, computed in one multiplication
    let theta = 2.0_f64.powi(-50); // asin(2^-50) = 2^-50 to f64 precision
    let optimal = (PI / (4.0 * theta)).floor() as u64;

    assert_eq!(
        optimal, 884_279_719_003_555,
        "⌊(π/4)·2^50⌋ iterations — the √N budget, counted without walking"
    );

    // the whole walk in one construction: the final angle lands the quarter
    // turn exactly (the residual mis-rotation ~1e-15 rad sits below the angle
    // lattice's boundary snap)
    let final_angle = Angle::new((2.0 * optimal as f64 + 1.0) * theta / PI, 1.0);
    assert!(
        final_angle.near(&Angle::new(1.0, 2.0)),
        "the state arrives at |target⟩ — the quarter turn, exactly"
    );

    let state = Geonum::new_with_angle(1.0, final_angle);
    assert!(
        state.opp().near_mag(1.0),
        "success amplitude 1 to machine precision, for a search space no state vector can hold"
    );
}

#[test]
fn it_reads_the_eigenphase_directly() {
    // phase estimation: U|ψ⟩ = e^(2πiφ)|ψ⟩ and QPE spends n ancilla qubits plus
    // an inverse QFT to estimate φ to n bits — ceremony that exists because
    // measurement collapses blade to grade. apply U as the rotation it is and
    // the eigenphase sits in the angle, whole, at full f64 precision. qubit
    // control firmware already ships this bookkeeping as the virtual-Z gate —
    // a software phase register the pulse hardware never sees
    let phi = 1.0 / 2.0_f64.sqrt(); // an irrational eigenphase
    let u_rotation = Angle::new(2.0 * phi, 1.0); // 2πφ

    let eigenstate = Geonum::new(1.0, 1.0, 7.0); // any eigenstate direction
    let applied = eigenstate.rotate(u_rotation);

    let recovered = (applied.angle - eigenstate.angle).grade_angle() / (2.0 * PI);
    assert!(
        (recovered - phi).abs() < 1e-15,
        "the eigenphase read whole: {recovered} vs {phi} — no ancilla register"
    );
}
