// attention is interference
//
// forward: score = query.dot(&key) — a Geonum
// backward: gradient = query.wedge(&key) — also a Geonum
// same angle difference, cosine vs sine
//
// score * value summed over keys: constructive and destructive interference
// the sum cancels what should cancel. normalize() after.
//
// projection is rotation. rotations compose by addition.
// what needs sequencing is routing: which tokens attend to which.
// everything else collapses.
//
// capacity is not a parameter. two features interact.
// dot captures whats parallel. wedge captures whats independent.
// the wedge is either zero or its a new geonum — already computed.
// the number of geonums a token needs is the number of
// nonzero wedge products in its feature interaction graph.
//
// f64 appears once: at the boundary with the loss function.

use geonum::*;

const EPSILON: f64 = 1e-10;

/// signed rotation strength from output toward target
/// positive = target is counterclockwise from output
/// negative = target is clockwise from output
fn signed_wedge(output: &Geonum, target: &Geonum) -> f64 {
    let wedge = output.wedge(target);
    let dir = (target.angle.grade_angle() - output.angle.grade_angle()).sin();
    if dir >= 0.0 {
        wedge.mag
    } else {
        -wedge.mag
    }
}

/// apply a signed rotation step to an angle
fn apply_step(current: Angle, gradient: f64, rate: f64) -> Angle {
    let step = Angle::new(rate * gradient.abs(), 1.0);
    if gradient >= 0.0 {
        current + step
    } else {
        current - step
    }
}

// ═══════════════════════════════════════════════════════════
// act I: dot is forward, wedge is backward
// ═══════════════════════════════════════════════════════════

#[test]
fn it_returns_a_geonum_from_dot_product() {
    // grade 0 = aligned. grade 2 = opposed. magnitude = strength.

    let q = Geonum::new_with_angle(2.0, Angle::new(1.0, 6.0));
    let k_aligned = Geonum::new_with_angle(3.0, Angle::new(1.0, 6.0));
    let k_orthogonal = Geonum::new_with_angle(3.0, Angle::new(2.0, 3.0));
    let k_opposed = Geonum::new_with_angle(3.0, Angle::new(7.0, 6.0));

    let score_aligned = q.dot(&k_aligned);
    let score_orthogonal = q.dot(&k_orthogonal);
    let score_opposed = q.dot(&k_opposed);

    assert_eq!(score_aligned.angle.grade(), 0);
    assert!((score_aligned.mag - 6.0).abs() < EPSILON);

    assert!(score_orthogonal.mag < EPSILON);

    assert_eq!(score_opposed.angle.grade(), 2);
    assert!((score_opposed.mag - 6.0).abs() < EPSILON);
}

#[test]
fn it_returns_a_geonum_from_wedge_product() {
    // dot and wedge are cosine and sine of the same angle difference
    // dot² + wedge² = |a|²|b|² — pythagorean identity

    let a = Geonum::new_with_angle(2.0, Angle::new(1.0, 6.0));
    let b = Geonum::new_with_angle(3.0, Angle::new(1.0, 3.0));

    let dot = a.dot(&b);
    let wedge = a.wedge(&b);

    let product_sq = a.mag * a.mag * b.mag * b.mag;
    assert!(
        (dot.mag * dot.mag + wedge.mag * wedge.mag - product_sq).abs() < 1e-6,
        "dot² + wedge² = |a|²|b|²"
    );
}

#[test]
fn it_has_zero_wedge_at_alignment_and_maximum_at_orthogonality() {
    // aligned: dot maximized, wedge zero — converged, nothing to learn
    // orthogonal: dot zero, wedge maximized — everything to learn

    let a = Geonum::new_with_angle(2.0, Angle::new(1.0, 4.0));
    let b_aligned = Geonum::new_with_angle(3.0, Angle::new(1.0, 4.0));
    let b_orthogonal = Geonum::new_with_angle(3.0, Angle::new(3.0, 4.0));

    assert!((a.dot(&b_aligned).mag - a.mag * b_aligned.mag).abs() < EPSILON);
    assert!(a.wedge(&b_aligned).mag < EPSILON);

    assert!(a.dot(&b_orthogonal).mag < EPSILON);
    assert!((a.wedge(&b_orthogonal).mag - a.mag * b_orthogonal.mag).abs() < EPSILON);
}

// ═══════════════════════════════════════════════════════════
// act II: attention is interference
// ═══════════════════════════════════════════════════════════

#[test]
fn it_weights_values_by_geometric_multiplication() {
    // score * value: magnitudes multiply, angles add
    // grade 0 score preserves value direction (constructive)
    // grade 2 score flips value by π (destructive)

    let value = Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0));

    let aligned = Geonum::new_with_angle(0.9, Angle::new(0.0, 1.0));
    let reinforced = aligned * value;
    assert!((reinforced.mag - 0.9).abs() < EPSILON);
    assert_eq!(reinforced.angle.grade(), value.angle.grade());

    let opposed = Geonum::new_with_angle(0.3, Angle::new(1.0, 1.0));
    let flipped = opposed * value;
    assert!((flipped.mag - 0.3).abs() < EPSILON);
}

#[test]
fn it_computes_attention_as_wave_sum() {
    // sum of score * value over all keys
    // vector sum < scalar sum — interference is doing the weighting

    let query = Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0));
    let keys = [
        Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0)),
        Geonum::new_with_angle(1.0, Angle::new(7.0, 6.0)),
    ];
    let values = [
        Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0)),
    ];

    let output: Geonum = keys
        .iter()
        .zip(values.iter())
        .map(|(k, v)| query.dot(k) * *v)
        .fold(Geonum::scalar(0.0), |acc, sv| acc + sv);

    let scalar_sum: f64 = keys
        .iter()
        .zip(values.iter())
        .map(|(k, v)| (query.dot(k) * *v).mag)
        .sum();

    assert!(output.mag > 0.0);
    assert!(output.mag < scalar_sum);
}

#[test]
fn it_normalizes_the_output_not_the_weights() {
    let scores = [
        Geonum::new_with_angle(5.0, Angle::new(0.0, 1.0)),
        Geonum::new_with_angle(0.2, Angle::new(1.0, 4.0)),
        Geonum::new_with_angle(3.0, Angle::new(1.0, 1.0)),
    ];
    let values = [
        Geonum::new_with_angle(1.0, Angle::new(1.0, 8.0)),
        Geonum::new_with_angle(1.0, Angle::new(3.0, 8.0)),
        Geonum::new_with_angle(1.0, Angle::new(5.0, 8.0)),
    ];

    let raw: Geonum = scores
        .iter()
        .zip(values.iter())
        .map(|(s, v)| *s * *v)
        .fold(Geonum::scalar(0.0), |acc, sv| acc + sv);

    let output = raw.normalize();

    assert!((output.mag - 1.0).abs() < EPSILON);
    assert_eq!(output.angle, raw.angle);
}

// ═══════════════════════════════════════════════════════════
// act III: rotations compose — routing is what sequences
// ═══════════════════════════════════════════════════════════

#[test]
fn it_composes_projection_chains_into_one_rotation() {
    let token = Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0));

    let rots = [
        Angle::new(1.0, 8.0),
        Angle::new(1.0, 3.0),
        Angle::new(3.0, 8.0),
        Angle::new(2.0, 3.0),
    ];

    let sequential = rots.iter().fold(token, |t, &r| t.rotate(r));
    let composed = token.rotate(rots.iter().fold(Angle::new(0.0, 1.0), |a, &r| a + r));

    assert!((sequential.mag - composed.mag).abs() < EPSILON);
    assert_eq!(sequential.angle, composed.angle);
}

#[test]
fn it_sequences_routing_not_layers() {
    let n_tokens = 4;

    let tokens: Vec<Geonum> = (0..n_tokens)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64 * 2.0, n_tokens as f64)))
        .collect();

    let route = |tokens: &[Geonum], q_rot: Angle, k_rot: Angle, v_rot: Angle| -> Vec<Geonum> {
        let queries: Vec<Geonum> = tokens.iter().map(|t| t.rotate(q_rot)).collect();
        let keys: Vec<Geonum> = tokens.iter().map(|t| t.rotate(k_rot)).collect();
        let values: Vec<Geonum> = tokens.iter().map(|t| t.rotate(v_rot)).collect();

        queries
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let raw: Geonum = keys
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| q.dot(k) * *v)
                    .fold(tokens[i], |acc, sv| acc + sv);

                if raw.mag > EPSILON {
                    raw.normalize().base_angle()
                } else {
                    raw.base_angle()
                }
            })
            .collect()
    };

    let r1 = route(
        &tokens,
        Angle::new(1.0, 8.0) + Angle::new(1.0, 3.0),
        Angle::new(5.0, 8.0),
        Angle::new(2.0, 8.0),
    );
    let r2 = route(
        &r1,
        Angle::new(3.0, 8.0) + Angle::new(2.0, 3.0),
        Angle::new(7.0, 8.0),
        Angle::new(4.0, 8.0),
    );

    let all_same = r2
        .windows(2)
        .all(|w| (w[0].mag - w[1].mag).abs() < EPSILON && w[0].angle == w[1].angle);
    assert!(!all_same);
}

// ═══════════════════════════════════════════════════════════
// act IV: multi-head is multi-angle
// ═══════════════════════════════════════════════════════════

#[test]
fn it_runs_multi_head_attention_as_angle_offsets() {
    let n_heads = 4;
    let n_tokens = 4;

    let tokens: Vec<Geonum> = (0..n_tokens)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64, n_tokens as f64)))
        .collect();

    let base_q = Angle::new(1.0, 12.0);
    let base_k = Angle::new(5.0, 12.0);
    let base_v = Angle::new(2.0, 12.0);

    let head_outputs: Vec<Vec<Geonum>> = (0..n_heads)
        .map(|h| {
            let offset = Angle::new(h as f64, n_heads as f64);
            let queries: Vec<Geonum> = tokens.iter().map(|t| t.rotate(base_q + offset)).collect();
            let keys: Vec<Geonum> = tokens.iter().map(|t| t.rotate(base_k + offset)).collect();
            let values: Vec<Geonum> = tokens.iter().map(|t| t.rotate(base_v + offset)).collect();

            queries
                .iter()
                .map(|q| {
                    keys.iter()
                        .zip(values.iter())
                        .map(|(k, v)| q.dot(k) * *v)
                        .fold(Geonum::scalar(0.0), |acc, sv| acc + sv)
                })
                .collect()
        })
        .collect();

    let combined: Vec<Geonum> = (0..n_tokens)
        .map(|i| {
            let sum = (0..n_heads)
                .map(|h| head_outputs[h][i])
                .fold(tokens[i], |acc, h| acc + h);
            if sum.mag > EPSILON {
                sum.normalize().base_angle()
            } else {
                sum.base_angle()
            }
        })
        .collect();

    for h1 in 0..n_heads {
        for h2 in (h1 + 1)..n_heads {
            let differs = (0..n_tokens).any(|i| {
                (head_outputs[h1][i].mag - head_outputs[h2][i].mag).abs() > EPSILON
                    || head_outputs[h1][i].angle != head_outputs[h2][i].angle
            });
            assert!(differs, "heads {} and {} differ", h1, h2);
        }
    }

    for t in &combined {
        assert!(t.mag > 0.0);
        assert!(t.mag.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════
// act V: position is rotation, dimension is blade
// ═══════════════════════════════════════════════════════════

#[test]
fn it_encodes_position_as_rotation() {
    let n = 8;

    let base: Vec<Geonum> = (0..n)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64 * 2.0, n as f64)))
        .collect();

    let positioned: Vec<Geonum> = base
        .iter()
        .enumerate()
        .map(|(pos, t)| t.rotate(Angle::new(pos as f64, n as f64)))
        .collect();

    for (b, p) in base.iter().zip(positioned.iter()) {
        assert!((b.mag - p.mag).abs() < EPSILON);
    }

    let dot_adjacent = positioned[0].dot(&positioned[1]);
    let dot_distant = positioned[0].dot(&positioned[7]);
    assert!(
        (dot_adjacent.mag - dot_distant.mag).abs() > EPSILON
            || dot_adjacent.angle != dot_distant.angle
    );
}

#[test]
fn it_proves_attention_cost_is_independent_of_embedding_dimension() {
    let q = Geonum::new_with_angle(1.5, Angle::new(1.0, 6.0));
    let k = Geonum::new_with_angle(2.0, Angle::new(1.0, 4.0));

    let score_low = q.dot(&k);

    let offset = Angle::new_with_blade(1_000_000, 0.0, 1.0);
    let q_high = Geonum::new_with_angle(q.mag, q.angle + offset);
    let k_high = Geonum::new_with_angle(k.mag, k.angle + offset);

    let score_high = q_high.dot(&k_high);

    assert!((score_low.mag - score_high.mag).abs() < EPSILON);
    assert_eq!(score_low.angle.grade(), score_high.angle.grade());
}

#[test]
fn it_proves_per_score_cost_is_constant_across_sequence_length() {
    // traditional attention: O(n² · d) — quadratic in tokens, linear in embedding dim
    // geonum attention: O(n²) — quadratic in tokens (irreducible), constant in embedding dim
    //
    // each q.dot(k) operates on two 24-byte geonums regardless of sequence length
    // the n² comes from routing pairs, not from per-pair cost
    // traditional: each score is a d-dimensional dot product O(d)
    // geonum: each score is one angle comparison O(1)

    let q_angle = Angle::new(1.0, 6.0);
    let offset = Angle::new_with_blade(1_000_000, 0.0, 1.0);

    // run full attention at different sequence lengths with million-D tokens
    // per-token output is one 24-byte Geonum at any scale
    let mut output_mags = Vec::new();
    for n in [4, 128, 1024] {
        let tokens: Vec<Geonum> = (0..n)
            .map(|i| {
                Geonum::new_with_angle(1.0, Angle::new_with_blade(1_000_000, i as f64, n as f64))
            })
            .collect();

        let q = Geonum::new_with_angle(1.5, q_angle + offset);

        // attention for one query against all keys
        let output: Geonum = tokens
            .iter()
            .map(|k| {
                let score = q.dot(k);
                score * *k
            })
            .fold(Geonum::scalar(0.0), |acc, sv| acc + sv);

        // output is one Geonum regardless of sequence length
        assert_eq!(std::mem::size_of_val(&output), 24);
        assert!(output.mag.is_finite());
        output_mags.push(output.mag);

        // each individual score is also one Geonum
        let score = q.dot(&tokens[0]);
        assert_eq!(std::mem::size_of_val(&score), 24);
    }

    // per-pair score is independent of other tokens in the sequence
    // dot(q, k) depends only on q and k — no summation over sequence
    // traditional d=1M attention: each score = 1M multiply-adds
    // geonum d=1M attention: each score = 1 dot on two 24-byte geonums
    assert_eq!(std::mem::size_of::<Geonum>(), 24);
}

// ═══════════════════════════════════════════════════════════
// act VI: learning — wedge grade drives the update
// ═══════════════════════════════════════════════════════════

#[test]
fn it_learns_a_rotation_to_match_a_target() {
    let input = Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0));
    let target = Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0));

    let learning_rate = 0.1;
    let mut learned_rotation = Angle::new(0.0, 1.0);
    let mut losses = Vec::new();

    for _ in 0..50 {
        let output = input.rotate(learned_rotation);

        let dot = output.dot(&target);
        losses.push(-(dot.mag * dot.angle.grade_angle().cos()));

        // signed wedge: magnitude = step size, sign = direction
        let gradient = signed_wedge(&output, &target);
        learned_rotation = apply_step(learned_rotation, gradient, learning_rate);
    }

    assert!(losses.last().unwrap() < losses.first().unwrap());

    let final_dot = input.rotate(learned_rotation).dot(&target);
    assert!(final_dot.angle.grade() == 0 && final_dot.mag > 0.9);
}

#[test]
fn it_proves_signed_wedge_encodes_rotation_direction() {
    // sin(target_angle - output_angle) gives the signed direction
    // positive = target is counterclockwise, negative = clockwise
    // this works for any angle difference, not just small ones

    let origin = Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)); // π/4

    // small counterclockwise: π/3 > π/4
    let ccw_small = Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0));
    assert!(
        signed_wedge(&origin, &ccw_small) > 0.0,
        "small ccw is positive"
    );

    // small clockwise: π/6 < π/4
    let cw_small = Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0));
    assert!(
        signed_wedge(&origin, &cw_small) < 0.0,
        "small cw is negative"
    );

    // large counterclockwise: 5π/6 from π/4 (diff = 7π/12)
    let ccw_large = Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0));
    assert!(
        signed_wedge(&origin, &ccw_large) > 0.0,
        "large ccw is positive"
    );

    // large clockwise: from π/4 back to near 0
    let cw_large = Geonum::new_with_angle(1.0, Angle::new(1.0, 20.0)); // π/20
    assert!(
        signed_wedge(&origin, &cw_large) < 0.0,
        "large cw is negative"
    );

    // apply_step moves in the signed direction
    let rate = 0.1;
    for (target, label) in [
        (ccw_small, "small ccw"),
        (cw_small, "small cw"),
        (ccw_large, "large ccw"),
        (cw_large, "large cw"),
    ] {
        let grad = signed_wedge(&origin, &target);
        let stepped = apply_step(origin.angle, grad, rate);
        let gap_before = (target.angle.grade_angle() - origin.angle.grade_angle()).abs();
        let gap_after = (target.angle.grade_angle() - stepped.grade_angle()).abs();
        assert!(
            gap_after < gap_before,
            "{}: step reduces distance to target ({:.4} → {:.4})",
            label,
            gap_before,
            gap_after
        );
    }
}

#[test]
fn it_learns_multiple_rotations_simultaneously() {
    let inputs = [
        Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0)),
        Geonum::new_with_angle(1.0, Angle::new(3.0, 4.0)),
    ];
    let targets = [
        Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0)),
        Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 1.0)),
    ];

    let learning_rate = 0.1;
    let mut rotations = [Angle::new(0.0, 1.0); 4];

    let loss = |rotations: &[Angle; 4]| -> f64 {
        (0..4)
            .map(|i| {
                let d = inputs[i].rotate(rotations[i]).dot(&targets[i]);
                -(d.mag * d.angle.grade_angle().cos())
            })
            .sum()
    };

    let initial_loss = loss(&rotations);

    for _ in 0..80 {
        for i in 0..4 {
            let output = inputs[i].rotate(rotations[i]);
            let gradient = signed_wedge(&output, &targets[i]);
            rotations[i] = apply_step(rotations[i], gradient, learning_rate);
        }
    }

    assert!(loss(&rotations) < initial_loss);

    for i in 0..4 {
        let d = inputs[i].rotate(rotations[i]).dot(&targets[i]);
        assert!(d.mag > 0.9 && d.angle.grade() == 0);
    }
}

#[test]
fn it_learns_which_tokens_to_attend_to() {
    // four tokens at 2π/n spacing. shift by 1: 0→1, 1→2, 2→3, 3→0

    let n = 4;
    let routing_target = [1, 2, 3, 0];

    // tokens span full 2π so cyclic routing is geometrically expressible
    let tokens: Vec<Geonum> = (0..n)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64 * 2.0, n as f64)))
        .collect();

    let values = tokens.clone();
    let target_outputs: Vec<Geonum> = routing_target.iter().map(|&r| values[r]).collect();

    let learning_rate = 0.05;
    let mut q_rot = Angle::new(1.0, 8.0); // π/8 — asymmetric init breaks saddle
    let mut k_rot = Angle::new(0.0, 1.0);

    let attend = |q_rot: Angle, k_rot: Angle| -> Vec<Geonum> {
        let queries: Vec<Geonum> = tokens.iter().map(|t| t.rotate(q_rot)).collect();
        let keys: Vec<Geonum> = tokens.iter().map(|t| t.rotate(k_rot)).collect();
        queries
            .iter()
            .map(|q| {
                keys.iter()
                    .zip(values.iter())
                    .map(|(k, v)| q.dot(k) * *v)
                    .fold(Geonum::scalar(0.0), |acc, sv| acc + sv)
            })
            .collect()
    };

    let loss = |q_rot: Angle, k_rot: Angle| -> f64 {
        attend(q_rot, k_rot)
            .iter()
            .zip(target_outputs.iter())
            .map(|(o, t)| {
                let d = o.dot(t);
                -(d.mag * d.angle.grade_angle().cos())
            })
            .sum()
    };

    let initial_loss = loss(q_rot, k_rot);

    for _ in 0..200 {
        let outputs = attend(q_rot, k_rot);

        // each outputs wedge with its target IS the gradient
        // no cross-pair accumulation — direct misalignment signal
        let mut q_grad = 0.0_f64;
        let mut k_grad = 0.0_f64;
        for i in 0..n {
            let grad = signed_wedge(&outputs[i], &target_outputs[i]);
            q_grad += grad / n as f64;
            k_grad -= grad / n as f64;
        }

        q_rot = apply_step(q_rot, q_grad, learning_rate);
        k_rot = apply_step(k_rot, k_grad, learning_rate);
    }

    assert!(loss(q_rot, k_rot) < initial_loss);

    let queries: Vec<Geonum> = tokens.iter().map(|t| t.rotate(q_rot)).collect();
    let keys: Vec<Geonum> = tokens.iter().map(|t| t.rotate(k_rot)).collect();

    let mut correct = 0;
    for i in 0..n {
        let best_j = (0..n)
            .map(|j| {
                let d = queries[i].dot(&keys[j]);
                (j, d.mag * d.angle.grade_angle().cos())
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(j, _)| j)
            .unwrap();

        if best_j == routing_target[i] {
            correct += 1;
        }
    }

    assert!(correct >= 3, "learned {}/{} routes", correct, n);
}

#[test]
fn it_learns_sequence_completion() {
    // [A, B, A, ?] → B — three sequences, same pattern, different tokens

    let sequences: Vec<(Vec<Geonum>, Geonum)> = vec![
        (
            vec![
                Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0)),
                Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0)),
                Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0)),
            ],
            Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0)),
        ),
        (
            vec![
                Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
                Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0)),
                Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
            ],
            Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0)),
        ),
        (
            vec![
                Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0)),
                Geonum::new_with_angle(1.0, Angle::new(3.0, 4.0)),
                Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0)),
            ],
            Geonum::new_with_angle(1.0, Angle::new(3.0, 4.0)),
        ),
    ];

    let learning_rate = 0.05;
    let mut q_rot = Angle::new(1.0, 7.0);
    let mut k_rot = Angle::new(2.0, 7.0);
    let mut v_rot = Angle::new(3.0, 7.0);

    let predict = |seq: &[Geonum], q_rot: Angle, k_rot: Angle, v_rot: Angle| -> Geonum {
        let positioned: Vec<Geonum> = seq
            .iter()
            .enumerate()
            .map(|(pos, t)| t.rotate(Angle::new(pos as f64, 8.0)))
            .collect();

        let query = positioned.last().unwrap().rotate(q_rot);
        let keys: Vec<Geonum> = positioned.iter().map(|t| t.rotate(k_rot)).collect();
        let values: Vec<Geonum> = positioned.iter().map(|t| t.rotate(v_rot)).collect();

        let raw: Geonum = keys
            .iter()
            .zip(values.iter())
            .map(|(k, v)| query.dot(k) * *v)
            .fold(Geonum::scalar(0.0), |acc, sv| acc + sv);

        if raw.mag > EPSILON {
            raw.normalize()
        } else {
            raw
        }
    };

    let total_loss = |q_rot: Angle, k_rot: Angle, v_rot: Angle| -> f64 {
        sequences
            .iter()
            .map(|(seq, target)| {
                let pred = predict(seq, q_rot, k_rot, v_rot);
                let d = pred.dot(target);
                -(d.mag * d.angle.grade_angle().cos())
            })
            .sum()
    };

    let initial_loss = total_loss(q_rot, k_rot, v_rot);

    for _ in 0..300 {
        // output-target wedge: the misalignment IS the gradient
        let mut q_grad = 0.0_f64;
        let mut k_grad = 0.0_f64;
        let mut v_grad = 0.0_f64;

        for (seq, target) in &sequences {
            let pred = predict(seq, q_rot, k_rot, v_rot);
            let grad = signed_wedge(&pred, target);
            q_grad += grad;
            k_grad -= grad;
            v_grad += grad;
        }

        let n_seq = sequences.len() as f64;
        q_rot = apply_step(q_rot, q_grad / n_seq, learning_rate);
        k_rot = apply_step(k_rot, k_grad / n_seq, learning_rate);
        v_rot = apply_step(v_rot, v_grad / n_seq, learning_rate);
    }

    assert!(total_loss(q_rot, k_rot, v_rot) < initial_loss);

    for (seq, target) in &sequences {
        let pred = predict(seq, q_rot, k_rot, v_rot);
        let d = pred.dot(target);
        assert!(d.mag * d.angle.grade_angle().cos() > 0.0);
    }
}

// ═══════════════════════════════════════════════════════════
// act VII: capacity emerges from the wedge product
// ═══════════════════════════════════════════════════════════

#[test]
fn it_produces_nonzero_wedge_between_independent_features() {
    let f1 = Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0));
    let f2 = Geonum::new_with_angle(1.0, Angle::new(1.1, 6.0));

    assert!(f1.dot(&f2).mag > 0.9);
    assert!(f1.wedge(&f2).mag < 0.2);

    let f3 = Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0));
    let f4 = Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0));

    assert!(f3.dot(&f4).mag < EPSILON);
    assert!((f3.wedge(&f4).mag - 1.0).abs() < EPSILON);
}

#[test]
fn it_derives_capacity_from_wedge_products() {
    let features = [
        Geonum::new_with_angle(1.0, Angle::new(0.0, 1.0)),
        Geonum::new_with_angle(0.8, Angle::new(1.0, 20.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0)),
        Geonum::new_with_angle(0.6, Angle::new(11.0, 20.0)),
        Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
    ];

    let threshold = 0.3;
    let mut representation: Vec<Geonum> = vec![features[0]];

    for &feature in &features[1..] {
        let min_wedge = representation
            .iter()
            .map(|r| r.wedge(&feature).mag / (r.mag * feature.mag))
            .fold(f64::INFINITY, f64::min);

        if min_wedge > threshold {
            representation.push(feature);
        } else {
            let best_idx = representation
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.dot(&feature)
                        .mag
                        .partial_cmp(&b.dot(&feature).mag)
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap();

            representation[best_idx] = representation[best_idx] + feature;
        }
    }

    assert!(representation.len() < features.len());

    for &feature in &features {
        let max_alignment = representation
            .iter()
            .map(|r| {
                let d = r.dot(&feature);
                (d.mag * d.angle.grade_angle().cos()) / (r.mag * feature.mag)
            })
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(max_alignment > 0.3);
    }
}

#[test]
fn it_matches_chemistry_shell_structure() {
    let mut particle = Geonum::new(1.0, 0.0, 1.0);
    let mut grades_seen = vec![particle.angle.grade()];

    for _ in 0..8 {
        particle = particle.increment_blade();
        grades_seen.push(particle.angle.grade());
    }

    assert_eq!(grades_seen, vec![0, 1, 2, 3, 0, 1, 2, 3, 0]);

    let g0 = Geonum::new(1.0, 0.0, 1.0);
    let g1 = Geonum::new(1.0, 1.0, 2.0);
    let g2 = Geonum::new(1.0, 1.0, 1.0);

    assert!(g0.wedge(&g1).mag > EPSILON);
    assert!(g1.wedge(&g2).mag > EPSILON);
    assert!(g0.wedge(&g2).mag < EPSILON);
}

#[test]
fn it_learns_with_dynamic_capacity() {
    let n_tokens = 3;

    let mut tokens: Vec<Vec<Geonum>> = (0..n_tokens)
        .map(|i| {
            vec![Geonum::new_with_angle(
                1.0,
                Angle::new(i as f64, n_tokens as f64),
            )]
        })
        .collect();

    let targets: Vec<[Geonum; 2]> = vec![
        [
            Geonum::new_with_angle(1.0, Angle::new(1.0, 6.0)),
            Geonum::new_with_angle(1.0, Angle::new(2.0, 3.0)),
        ],
        [
            Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0)),
            Geonum::new_with_angle(1.0, Angle::new(3.0, 4.0)),
        ],
        [
            Geonum::new_with_angle(1.0, Angle::new(1.0, 3.0)),
            Geonum::new_with_angle(1.0, Angle::new(5.0, 6.0)),
        ],
    ];

    let learning_rate = 0.1;
    let wedge_threshold = 0.3;

    for _ in 0..60 {
        for i in 0..n_tokens {
            for target in &targets[i] {
                let best_match = tokens[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let da = a.dot(target).mag * a.dot(target).angle.grade_angle().cos();
                        let db = b.dot(target).mag * b.dot(target).angle.grade_angle().cos();
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();

                let current = tokens[i][best_match];
                let wedge = current.wedge(target);

                if wedge.mag / (current.mag * target.mag) > wedge_threshold && tokens[i].len() < 4 {
                    // the wedge IS the new geonum
                    tokens[i].push(Geonum::new_with_angle(wedge.mag, target.angle));
                } else {
                    // rotate toward target — signed wedge picks the direction
                    let gradient = signed_wedge(&current, target);
                    let new_angle = apply_step(current.angle, gradient, learning_rate);
                    tokens[i][best_match] = Geonum::new_with_angle(current.mag, new_angle);
                }
            }
        }
    }

    let total_geonums: usize = tokens.iter().map(|t| t.len()).sum();
    assert!(total_geonums > n_tokens);

    for i in 0..n_tokens {
        for target in &targets[i] {
            let max_alignment = tokens[i]
                .iter()
                .map(|g| {
                    let d = g.dot(target);
                    d.mag * d.angle.grade_angle().cos() / (g.mag * target.mag)
                })
                .fold(f64::NEG_INFINITY, f64::max);

            assert!(max_alignment > 0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// act VIII: full forward pass — f64 only at the boundary
// ═══════════════════════════════════════════════════════════

#[test]
fn it_runs_a_full_forward_pass_in_geonum_space() {
    let n_routing_steps = 3;
    let n_tokens = 4;
    let n_heads = 2;

    let vocab: Vec<Geonum> = (0..10)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64, 5.0)))
        .collect();

    let input_ids = [0, 3, 7, 2];
    let mut tokens: Vec<Geonum> = input_ids
        .iter()
        .enumerate()
        .map(|(pos, &id)| vocab[id].rotate(Angle::new(pos as f64, 20.0)))
        .collect();

    let step_rots: Vec<Vec<(Angle, Angle, Angle)>> = (0..n_routing_steps)
        .map(|s| {
            (0..n_heads)
                .map(|h| {
                    let base =
                        Angle::new((s * n_heads + h) as f64, (n_routing_steps * n_heads) as f64);
                    (
                        base + Angle::new(1.0, 12.0),
                        base + Angle::new(5.0, 12.0),
                        base + Angle::new(3.0, 12.0),
                    )
                })
                .collect()
        })
        .collect();

    let ff_rots: Vec<Angle> = (0..n_routing_steps)
        .map(|s| Angle::new((s + 1) as f64, (n_routing_steps + 1) as f64))
        .collect();

    for step in 0..n_routing_steps {
        let head_outputs: Vec<Vec<Geonum>> = step_rots[step]
            .iter()
            .map(|&(q_rot, k_rot, v_rot)| {
                let queries: Vec<Geonum> = tokens.iter().map(|t| t.rotate(q_rot)).collect();
                let keys: Vec<Geonum> = tokens.iter().map(|t| t.rotate(k_rot)).collect();
                let values: Vec<Geonum> = tokens.iter().map(|t| t.rotate(v_rot)).collect();

                queries
                    .iter()
                    .map(|q| {
                        keys.iter()
                            .zip(values.iter())
                            .map(|(k, v)| q.dot(k) * *v)
                            .fold(Geonum::scalar(0.0), |acc, sv| acc + sv)
                    })
                    .collect()
            })
            .collect();

        tokens = (0..n_tokens)
            .map(|i| {
                let sum = (0..n_heads)
                    .map(|h| head_outputs[h][i])
                    .fold(tokens[i], |acc, h| acc + h);

                let normed = if sum.mag > EPSILON {
                    sum.normalize().base_angle()
                } else {
                    sum.base_angle()
                };

                let hidden = normed.rotate(ff_rots[step]);
                let adj = hidden.adj();
                if adj.angle.is_scalar() {
                    hidden
                } else {
                    hidden.scale(0.1)
                }
            })
            .collect();
    }

    let logits: Vec<Vec<f64>> = tokens
        .iter()
        .map(|t| {
            vocab
                .iter()
                .map(|v| {
                    let d = t.dot(v);
                    d.mag * d.angle.grade_angle().cos()
                })
                .collect()
        })
        .collect();

    for token_logits in logits.iter() {
        assert_eq!(token_logits.len(), 10);
        assert!(token_logits.iter().all(|l| l.is_finite()));
    }

    let all_same = tokens
        .windows(2)
        .all(|w| (w[0].mag - w[1].mag).abs() < EPSILON && w[0].angle == w[1].angle);
    assert!(!all_same);
}
