// next token prediction without training
//
// the step between consecutive tokens is a rotation
// observe it, cluster it, apply it
//
// sequences in, predictions out, generalizes to sequences never seen
// the board discovers how many step patterns exist
// zero epochs. zero gradients. zero layers.

use geonum::*;

const VOCAB_SIZE: usize = 8;
const EPSILON: f64 = 1e-10;

fn vocab() -> Vec<Geonum> {
    (0..VOCAB_SIZE)
        .map(|i| Geonum::new_with_angle(1.0, Angle::new(i as f64 * 2.0, VOCAB_SIZE as f64)))
        .collect()
}

fn argmax(logits: &[f64]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn logits_for(output: &Geonum, v: &[Geonum]) -> Vec<f64> {
    v.iter()
        .map(|vi| {
            let d = output.dot(vi);
            d.mag * d.angle.grade_angle().cos()
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════
// the step is the rotation
// ═══════════════════════════════════════════════════════════

/// the rotation between two consecutive tokens
fn observed_step(prev: &Geonum, current: &Geonum) -> Geonum {
    let diff = current.angle.grade_angle() - prev.angle.grade_angle();
    Geonum::new_from_cartesian(diff.cos(), diff.sin())
}

struct Board {
    nodes: Vec<(Geonum, Angle)>,
}

impl Board {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// observe a step. merge if it matches an existing node, otherwise malloc.
    fn observe(&mut self, step: Geonum, threshold: f64) {
        for (center, rotation) in &mut self.nodes {
            if center.mag > EPSILON && center.normalize().wedge(&step).mag < threshold {
                *center = *center + step;
                *rotation = center.angle.base_angle();
                return;
            }
        }
        self.nodes.push((step, step.angle.base_angle()));
    }

    /// predict: match the observed step to a node, rotate the last token
    fn predict(&self, last_step: &Geonum, last_token: &Geonum, v: &[Geonum]) -> usize {
        let mut best_node = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (ni, (center, _)) in self.nodes.iter().enumerate() {
            if center.mag > EPSILON {
                let d = center.normalize().dot(last_step);
                let score = d.mag * d.angle.grade_angle().cos();
                if score > best_score {
                    best_score = score;
                    best_node = ni;
                }
            }
        }
        let output = last_token.rotate(self.nodes[best_node].1);
        argmax(&logits_for(&output, v))
    }
}

// ═══════════════════════════════════════════════════════════
// act I: one step pattern, one node
// ═══════════════════════════════════════════════════════════

#[test]
fn it_predicts_next_token_with_one_step_pattern() {
    let v = vocab();

    // step +1: every sequence increments by 1
    let train: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![3, 4, 5, 6], vec![5, 6, 7, 0]];

    let mut board = Board::new();
    for seq in &train {
        for w in seq.windows(2) {
            board.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    assert_eq!(board.nodes.len(), 1, "one step pattern → one node");

    // train predictions: predict last from prefix
    for seq in &train {
        let n = seq.len();
        let step = observed_step(&v[seq[n - 3]], &v[seq[n - 2]]);
        let pred = board.predict(&step, &v[seq[n - 2]], &v);
        assert_eq!(
            pred,
            seq[n - 1],
            "train: {:?} → {}",
            &seq[..n - 1],
            seq[n - 1]
        );
    }

    // unseen sequences
    let test: Vec<(Vec<usize>, usize)> =
        vec![(vec![2, 3, 4], 5), (vec![7, 0, 1], 2), (vec![4, 5, 6], 7)];
    for (seq, target) in &test {
        let n = seq.len();
        let step = observed_step(&v[seq[n - 2]], &v[seq[n - 1]]);
        let pred = board.predict(&step, &v[seq[n - 1]], &v);
        assert_eq!(pred, *target, "test: {:?} → {}", seq, target);
    }
}

// ═══════════════════════════════════════════════════════════
// act II: three step patterns, three nodes
// ═══════════════════════════════════════════════════════════

#[test]
fn it_discovers_multiple_step_patterns() {
    let v = vocab();

    let step1: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![5, 6, 7, 0]];
    let step2: Vec<Vec<usize>> = vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]];
    let step3: Vec<Vec<usize>> = vec![vec![0, 3, 6, 1], vec![2, 5, 0, 3]];

    let mut board = Board::new();
    for seq in step1.iter().chain(step2.iter()).chain(step3.iter()) {
        for w in seq.windows(2) {
            board.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    assert_eq!(board.nodes.len(), 3, "three step patterns → three nodes");

    // the three rotations correspond to +1, +2, +3 vocab steps
    let mut rotations: Vec<f64> = board.nodes.iter().map(|&(_, r)| r.grade_angle()).collect();
    rotations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quarter = std::f64::consts::PI / 4.0;
    assert!((rotations[0] - quarter).abs() < 0.01); // π/4 = step +1
    assert!((rotations[1] - 2.0 * quarter).abs() < 0.01); // π/2 = step +2
    assert!((rotations[2] - 3.0 * quarter).abs() < 0.01); // 3π/4 = step +3
}

// ═══════════════════════════════════════════════════════════
// act III: generalizes to unseen sequences
// ═══════════════════════════════════════════════════════════

#[test]
fn it_generalizes_across_step_patterns() {
    let v = vocab();

    // train on some sequences per step
    let train: Vec<Vec<usize>> = vec![
        vec![0, 1, 2, 3],
        vec![3, 4, 5, 6], // step +1
        vec![0, 2, 4, 6],
        vec![1, 3, 5, 7], // step +2
        vec![0, 3, 6, 1],
        vec![4, 7, 2, 5], // step +3
    ];

    let mut board = Board::new();
    for seq in &train {
        for w in seq.windows(2) {
            board.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    // test on sequences the board never saw
    let test: Vec<(Vec<usize>, usize)> = vec![
        (vec![7, 0, 1], 2), // step +1
        (vec![4, 5, 6], 7), // step +1
        (vec![3, 5, 7], 1), // step +2
        (vec![6, 0, 2], 4), // step +2
        (vec![1, 4, 7], 2), // step +3
        (vec![7, 2, 5], 0), // step +3
    ];

    let mut correct = 0;
    for (seq, target) in &test {
        let n = seq.len();
        let step = observed_step(&v[seq[n - 2]], &v[seq[n - 1]]);
        let pred = board.predict(&step, &v[seq[n - 1]], &v);
        if pred == *target {
            correct += 1;
        }
    }
    assert_eq!(
        correct,
        test.len(),
        "generalizes to {}/{} unseen sequences",
        correct,
        test.len()
    );
}

// ═══════════════════════════════════════════════════════════
// act IV: board size IS task complexity
// ═══════════════════════════════════════════════════════════

#[test]
fn it_proves_board_size_matches_task_complexity() {
    let v = vocab();

    // 1 step pattern
    let s1: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
    let mut b1 = Board::new();
    for seq in &s1 {
        for w in seq.windows(2) {
            b1.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    // 2 step patterns
    let s2: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![0, 2, 4, 6]];
    let mut b2 = Board::new();
    for seq in &s2 {
        for w in seq.windows(2) {
            b2.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    // 4 step patterns
    let s4: Vec<Vec<usize>> = vec![
        vec![0, 1, 2, 3],
        vec![0, 2, 4, 6],
        vec![0, 3, 6, 1],
        vec![0, 4, 0, 4],
    ];
    let mut b4 = Board::new();
    for seq in &s4 {
        for w in seq.windows(2) {
            b4.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
        }
    }

    assert_eq!(b1.nodes.len(), 1);
    assert_eq!(b2.nodes.len(), 2);
    assert_eq!(b4.nodes.len(), 4);
}

// ═══════════════════════════════════════════════════════════
// act V: context via multiplication
// ═══════════════════════════════════════════════════════════

#[test]
fn it_uses_sequence_product_as_context() {
    // for sequences where the pattern isnt constant-step,
    // the geometric product of the whole sequence encodes the context
    // residual from context to target gives the rotation

    let v = vocab();

    // pattern: target = product of all input tokens (mod the circle)
    let train: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 1], 1), // v[0]*v[1] at angle 0+π/4=π/4 → target at π/4 = token 1
        (vec![1, 2], 3), // angle π/4+π/2=3π/4 → token 3
        (vec![0, 3], 3), // angle 0+3π/4=3π/4 → token 3
        (vec![2, 2], 4), // angle π/2+π/2=π → token 4
        (vec![1, 1], 2), // angle π/4+π/4=π/2 → token 2
    ];

    let mut board = Board::new();
    for (seq, target) in &train {
        let ctx = seq
            .iter()
            .map(|&id| v[id])
            .fold(Geonum::scalar(1.0), |acc, t| acc * t);
        let diff = v[*target].angle.grade_angle() - ctx.angle.grade_angle();
        let residual = Geonum::new_from_cartesian(diff.cos(), diff.sin());
        board.observe(residual, 0.3);
    }

    // all residuals point to the same rotation (identity — product already IS the target)
    assert_eq!(board.nodes.len(), 1);

    // generalize
    let test: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 2], 2), // 0+π/2=π/2 → token 2
        (vec![3, 3], 6), // 3π/4+3π/4=3π/2 → token 6
        (vec![1, 3], 4), // π/4+3π/4=π → token 4
    ];

    for (seq, target) in &test {
        let ctx = seq
            .iter()
            .map(|&id| v[id])
            .fold(Geonum::scalar(1.0), |acc, t| acc * t);
        let output = ctx.rotate(board.nodes[0].1);
        let pred = argmax(&logits_for(&output, &v));
        assert_eq!(pred, *target, "product context: {:?} → {}", seq, target);
    }
}

// ═══════════════════════════════════════════════════════════
// act VI: dimension independence
// ═══════════════════════════════════════════════════════════

#[test]
fn it_proves_prediction_is_dimension_independent() {
    let v = vocab();
    let offset = Angle::new_with_blade(1_000_000, 0.0, 1.0);
    let v_high: Vec<Geonum> = v
        .iter()
        .map(|g| Geonum::new_with_angle(g.mag, g.angle + offset))
        .collect();

    let sequences: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![0, 2, 4, 6]];

    // build boards at blade 0 and blade 1M
    let mut board_low = Board::new();
    let mut board_high = Board::new();
    for seq in &sequences {
        for w in seq.windows(2) {
            board_low.observe(observed_step(&v[w[0]], &v[w[1]]), 0.3);
            board_high.observe(observed_step(&v_high[w[0]], &v_high[w[1]]), 0.3);
        }
    }

    assert_eq!(board_low.nodes.len(), board_high.nodes.len());

    // same predictions
    let step_low = observed_step(&v[2], &v[3]);
    let step_high = observed_step(&v_high[2], &v_high[3]);
    let pred_low = board_low.predict(&step_low, &v[3], &v);
    let pred_high = board_high.predict(&step_high, &v_high[3], &v_high);
    assert_eq!(pred_low, pred_high);

    assert_eq!(std::mem::size_of::<Geonum>(), 24);
}
