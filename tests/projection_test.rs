//! a line is a projected angle, an area is weighted angles summed, the integral is a
//! boundary read — geometry is angle-first, projection is the afterthought
//!
//! the conventional picture is projection-first: you live in x and y, store points as
//! coordinate pairs, draw a line by looping a parameter, and measure area by summing a mesh
//! of rectangles. geonum inverts it. the primitive is the angle (carrying a magnitude);
//! line, area, shape, and the coordinates themselves are figures CAST from angles —
//! projection space, computed only when a scalar is demanded.
//!
//! - a line through the origin is one angle; it PROJECTS to the figure, and carries no shape
//!   (the same angle at any magnitude). a line of length r is the geonum [r, θ], one op from
//!   the origin. position and coordinates fall out of subtraction and projection
//! - projection is not a separate primitive: it is multiply (angle adds, magnitude
//!   multiplies) with the factor pinned to the cosine of the angle gap
//! - area is rotations weighted by their radii, summed — each wedge a weighted angle. a
//!   straight run is a redundant sequence that collapses to one; a polyline keeps one term
//!   per bend; the spacing→0 limit is the integral
//! - the weighted-angle sum is the primitive; it always computes the area, exact in the
//!   limit. where the angles are redundant it collapses to a boundary read — collinear (the
//!   line), constant weight (the sector), an antiderivative (∫cos = sin, the spiral's r³/3,
//!   the exponential's eˣ). where they arent you sum them directly. "non-elementary"
//!   (r = e^(θ²/2)) means no O(1) shortcut, not no answer — there is no swept area geonum
//!   cant measure; the weighted-angle sum just runs directly
//! - the shape lives in the weights (a GeoCollection of weighted angles); the angle is the
//!   shapeless line. line and area are both projection-space figures; the angle is primitive

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-9;

// ═══════════════════════════════════════════════════════════════════════════
// part 1 — the angle is the primitive; line and coordinates are projected
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_projects_an_origin_geonum_into_a_positioned_unit_vector() {
    // Q=(1,1) at the origin is [√2, π/4]. project it onto the axes and it casts two unit
    // shadows: onto +x it lands on P=(1,0), onto +y it lands on [1, π/2] — the vector from
    // P to Q. those two shadows reassemble Q. position and the unit vector come out of one
    // origin geonum by projection, no coordinates pulled apart
    let q = Geonum::new_from_cartesian(1.0, 1.0);
    let p = q.project(&Geonum::new(1.0, 0.0, 1.0)); // x-shadow = P
    let pq = q.project(&Geonum::new(1.0, 1.0, 2.0)); // y-shadow = the unit vector P→Q

    assert!(
        p.near_mag(1.0) && p.angle.grade() == 0,
        "Q's x-shadow is P = [1, 0]"
    );
    assert!(
        pq.near_mag(1.0) && pq.angle.grade() == 1,
        "the y-shadow is a unit vector at π/2"
    );
    assert!(pq.angle.near_rad(PI / 2.0), "its angle is π/2");

    let head = p + pq;
    assert!(
        head.near_mag(q.mag) && head.angle.grade() == q.angle.grade(),
        "P + unit reaches Q — the shadows reassemble the point"
    );
}

#[test]
fn it_creates_a_line_from_a_single_angle() {
    // one angle, given unit magnitude, IS the line — y=x is [1, π/4], no point, no anchor.
    // the whole line is that one geonum scaled: every point [r, θ] is line·r, on the line
    let theta = Angle::new(1.0, 4.0);
    let line = Geonum::new_with_angle(1.0, theta);
    assert!(line.near_mag(1.0));
    assert_eq!(line.angle, theta, "from the single angle θ alone");

    for r in [2.0, 0.5, -3.0, 1000.0] {
        let pt = line.scale(r);
        assert!(pt.near_mag(r.abs()), "point at r is one op");
        assert!(
            pt.reject(&line).mag < EPSILON,
            "every [r, θ] is on the line"
        );
    }
}

#[test]
fn it_gives_the_line_its_magnitude_as_one_op() {
    // the bare angle is the direction; give it a magnitude and it is a SEGMENT — a directed
    // extent from the origin, the geonum [r, θ], one scale_rotate from an origin reference.
    // a line carries magnitude because a geonum is a magnitude in a direction
    let theta = Angle::new(1.0, 4.0);
    let line = Geonum::new(1.0, 0.0, 1.0).scale_rotate(2.0, theta);
    assert!(
        line.near_mag(2.0) && line.angle == theta,
        "[2, π/4], length and direction"
    );

    // its tip is (√2, √2) — but those coordinates are the afterthought, projected only now
    let s = 2.0_f64.sqrt();
    let (x, y) = coords(line);
    assert!(
        (x - s).abs() < EPSILON && (y - s).abs() < EPSILON,
        "tip (√2, √2), projected last"
    );
}

#[test]
fn it_keeps_the_angle_shapeless() {
    // the same angle is the line at any magnitude — a point near the origin and one far out
    // on y=x share it exactly. the angle carries direction, not length, not shape
    let near = Geonum::new(1.0, 1.0, 4.0);
    let far = Geonum::new(1000.0, 1.0, 4.0);
    assert_eq!(
        near.angle, far.angle,
        "same angle at any distance — the angle is the line"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// part 2 — projection is multiply
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_shows_projection_is_a_cosine_scaled_multiply() {
    // scale_rotate(f, ρ) = self * [f, ρ]: scale the magnitude, add the angle — multiply.
    // projection is the SAME op with the factor pinned to the cosine of the gap and the
    // rotation landing on the target. project ⊂ scale_rotate ⊂ multiply
    let g = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]
    let onto = Angle::new(1.0, 4.0); // π/4 — cosine of the gap positive

    let projected = g.project(&Geonum::new_with_angle(1.0, onto));
    let cos_gap = g.angle.project(onto); // cos(π/12), read off project
    let rebuilt = g.scale_rotate(cos_gap, onto - g.angle);
    let via_multiply = g * Geonum::new_with_angle(cos_gap, onto - g.angle);

    assert!(
        projected.near(&rebuilt),
        "project == scale_rotate(cos(gap), → target)"
    );
    assert!(
        projected.near(&via_multiply),
        "and both are g * [cos(gap), gap] — multiply"
    );
}

#[test]
fn it_projects_a_point_against_the_single_angle() {
    // relate any point to the line through one projection onto the ONE angle — foot and
    // offset, no range walked. (2,0) drops onto y=x at (1,1), √2 away
    let line = Geonum::new_with_angle(1.0, Angle::new(1.0, 4.0));
    let x = Geonum::new_from_cartesian(2.0, 0.0);
    let (fx, fy) = coords(x.project(&line));
    assert!(
        (fx - 1.0).abs() < EPSILON && (fy - 1.0).abs() < EPSILON,
        "foot of (2,0) on y=x is (1,1)"
    );
    assert!(
        (x.reject(&line).mag - 2.0_f64.sqrt()).abs() < EPSILON,
        "offset = √2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// part 3 — area is weighted angles, summed
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_reads_the_swept_area_from_the_summed_rotation() {
    // traversing a chord from the origin is a sequence of rotations; they ADD to one net
    // angle (the boundary difference), and the wedge reads the swept triangle from that one
    // angle and the two radii — O(1), no loop
    let p1 = Geonum::new_from_cartesian(0.0, 1.0); // (0,1)
    let p2 = Geonum::new_from_cartesian(1.0, 1.0); // (1,1)

    let net = (p1.angle.grade_angle() - p2.angle.grade_angle()).abs();
    assert!(
        (net - PI / 4.0).abs() < EPSILON,
        "the rotations add to a single angle, π/4"
    );
    assert!(
        (0.5 * p1.wedge(&p2).mag - 0.5).abs() < EPSILON,
        "the wedge reads the swept area = ½"
    );
}

#[test]
fn it_accumulates_area_per_edge_over_a_polyline() {
    // a polyline's area ACCUMULATES — one wedge per edge, summed — and does NOT telescope to
    // the boundary: the bend at (2,1) carries area the direct triangle never sees
    let p0 = Geonum::new_from_cartesian(0.0, 1.0);
    let p1 = Geonum::new_from_cartesian(2.0, 1.0);
    let p2 = Geonum::new_from_cartesian(2.0, 0.0);

    let swept = 0.5 * (p0.wedge(&p1).mag + p1.wedge(&p2).mag);
    let boundary = 0.5 * p0.wedge(&p2).mag;
    assert!(
        (swept - 2.0).abs() < EPSILON,
        "area accumulates per edge = 2"
    );
    assert!(
        (swept - boundary).abs() > 0.5,
        "it does not telescope — the bend carries real area"
    );
}

#[test]
fn it_unifies_line_and_polyline_as_one_weighted_angle_sum() {
    // each wedge is a weighted angle: the rotation sin(Δθ) weighted by the radii. a straight
    // line is a REDUNDANT sequence of them — collinear, so they collapse to one term. a
    // polyline is the SAME sum with the angles non-redundant: one weighted term per edge
    let a = Geonum::new_from_cartesian(0.0, 1.0);
    let b = Geonum::new_from_cartesian(2.0, 0.0);
    let samples: Vec<Geonum> = (0..=10)
        .map(|k| a + (b - a).scale(k as f64 / 10.0))
        .collect();
    assert!(
        (weighted_angle_area(&samples) - 0.5 * a.wedge(&b).mag).abs() < EPSILON,
        "the line's redundant angles collapse to the single boundary wedge"
    );

    let path = [
        Geonum::new_from_cartesian(0.0, 1.0),
        Geonum::new_from_cartesian(2.0, 1.0),
        Geonum::new_from_cartesian(2.0, 0.0),
    ];
    assert!(
        (weighted_angle_area(&path) - 2.0).abs() < EPSILON,
        "the polyline keeps one weighted angle per bend — same operation, more terms"
    );
}

#[test]
fn it_reads_the_shape_from_the_weighted_collection() {
    // the shape a path traces lives in the WEIGHTS, not the angle: a GeoCollection of the
    // per-edge wedges (each a weighted angle) carries it, and ½ their summed magnitude is
    // the area. the angle is the shapeless line; the weights are its shape
    let path = [
        Geonum::new_from_cartesian(0.0, 1.0),
        Geonum::new_from_cartesian(2.0, 1.0),
        Geonum::new_from_cartesian(2.0, 0.0),
    ];
    let weighted: GeoCollection = path.windows(2).map(|w| w[0].wedge(&w[1])).collect();
    assert!(
        (0.5 * weighted.total_magnitude() - 2.0).abs() < EPSILON,
        "the area is ½·Σ of the weighted entries — read from the collection, not an angle"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// part 4 — the boundary read is the shortcut; the weighted sum is always there
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_collapses_the_constant_weight_integral_to_a_boundary_read() {
    // let the spacing go to zero and the weighted-angle sum becomes the integral ½∫r²dθ. on
    // the unit circle the inscribed chords converge to the sector from below — the riemann
    // limit. but the weight ½r² is CONSTANT, so the integral collapses to a boundary read:
    // ½r²θ, the antiderivative at the boundary. the chords merely approach what it states
    let theta = PI / 2.0;
    let sector = 0.5 * theta; // ½r²θ with r = 1 → π/4

    let coarse = inscribed_area(10, theta);
    let fine = inscribed_area(100_000, theta);
    assert!(
        coarse < fine && fine < sector,
        "shrinking the spacing closes the gap from below"
    );
    assert!(
        (sector - PI / 4.0).abs() < EPSILON,
        "½r²θ = π/4, the boundary the chords approach"
    );
    assert!(
        (sector - fine).abs() < 1e-7,
        "the limit closes on the boundary read it collapses to"
    );
}

#[test]
fn it_collapses_a_varying_radius_when_its_square_is_integrable_in_theta() {
    // the weight need NOT be constant. on the archimedean spiral r = θ the swept area is
    // ½∫r²dθ = ½∫θ²dθ, and θ² has an antiderivative — θ³/3, which is r³/3 because r = θ. so
    // the integral collapses to a boundary read ½·r³/3 even though the radius varies. a
    // varying radius loses its O(1) shortcut only when its square isnt integrable in θ
    // (r = e^(θ²)) — and geonum still sums it. r³/3 is the antiderivative in θ, not ∫r²dr in r
    let theta = 2.0_f64; // sweep θ ∈ [0, 2]; the spiral point at angle t is [t, t]
    let boundary = 0.5 * theta.powi(3) / 3.0; // ½ · r³/3 = θ³/6
    assert!(
        (boundary - theta.powi(3) / 6.0).abs() < EPSILON,
        "the boundary read is θ³/6 via r³/3"
    );

    // the spacing→0 chords on the spiral close on it — no loop, just a varying weight
    let n = 200_000;
    let pts: Vec<Geonum> = (0..=n)
        .map(|k| {
            let t = theta * k as f64 / n as f64;
            Geonum::new_with_angle(t, Angle::new(t, PI)) // [t, t] — radius and angle both t
        })
        .collect();
    assert!(
        (boundary - weighted_angle_area(&pts)).abs() < 1e-3,
        "the chords converge to ½·r³/3 — the spiral collapses, varying radius and all"
    );
}

#[test]
fn it_collapses_the_exponential_spiral() {
    // the exponential is the most native radius. in geonum it is rotation (e^(iθ) = [1, θ])
    // or growth (eˣ, its own antiderivative from the boundary magnitudes — exponential_test).
    // the equiangular spiral r = e^θ is growth: r² = e^(2θ), antiderivative e^(2θ)/2, so the
    // swept area collapses to (r_b² − r_a²)/4 — a boundary read of the two magnitudes
    let big = 1.0_f64; // sweep θ ∈ [0, 1]; the spiral point at angle t is [e^t, t]
    let (r_a, r_b) = (0.0_f64.exp(), big.exp()); // boundary magnitudes e^0 = 1, e^1
    let boundary = (r_b * r_b - r_a * r_a) / 4.0; // (e² − 1)/4

    let n = 200_000;
    let pts: Vec<Geonum> = (0..=n)
        .map(|k| {
            let t = big * k as f64 / n as f64;
            Geonum::new_with_angle(t.exp(), Angle::new(t, PI)) // [e^t, t] — exponential growth
        })
        .collect();
    assert!(
        (boundary - weighted_angle_area(&pts)).abs() < 1e-3,
        "r = e^θ collapses to (r_b² − r_a²)/4 — read from the boundary magnitudes, eˣ its own antiderivative"
    );
}

#[test]
fn it_squares_an_angle_as_a_rotation() {
    // squaring the sweep parameter is a pure ANGLE operation: scale the angle by its own
    // measure, θ·θ = θ², a rotation — the Angle carries no magnitude for a radius to enter.
    // pow(2) is the number-square instead: on a geonum it squares the magnitude (mag²) and
    // DOUBLES the angle (2θ). so e^(θ²) is the angle's rotation, never a hard radius
    let theta = Angle::new(1.0, 2.0); // θ = π/2

    // θ² lands in the angle, crossing a grade boundary — (π/2)² ≈ 2.47 rad is past π/2
    let squared = theta * (PI / 2.0);
    assert!(
        squared.near_rad((PI / 2.0) * (PI / 2.0)),
        "θ scaled by θ is θ² — the square is a rotation living in the angle"
    );

    // pow(2) squares the magnitude and doubles the angle: [mag², 2θ], the number-square
    let powered = Geonum::new_with_angle(3.0, theta).pow(2.0);
    assert!(powered.near_mag(9.0), "pow squares the magnitude: 3² = 9");
    assert!(
        powered.angle.near_rad(PI),
        "pow doubles the angle: 2·π/2 = π, not (π/2)²"
    );
}

#[test]
fn it_derives_the_definite_integral_of_cos_from_the_boundary() {
    // where the integrand has an antiderivative the integral is a boundary read, no loop.
    // geonum's
    // integrate tick (the +3π/2 grade rotation) turns cos's readout into sin — sin is never
    // hand-fed — and ∫cos = sin(b) − sin(a) falls out of two endpoint reads
    let antideriv = |x: Angle| unit(x).integrate().angle.cos_sin().0;
    for &(a, b) in &[
        (Angle::new(0.0, 1.0), Angle::new(1.0, 2.0)), // [0, π/2]
        (Angle::new(1.0, 6.0), Angle::new(2.0, 3.0)), // [π/6, 2π/3]
    ] {
        let derived = antideriv(b) - antideriv(a);
        let want = b.grade_angle().sin() - a.grade_angle().sin();
        assert!(
            (derived - want).abs() < EPSILON,
            "∫cos = sin(b) − sin(a), a boundary read"
        );
    }
}

// ───────────────────────────────────────────────────────────────────────────
// helpers — the projection-space readouts, cast from angle space only when asked
// ───────────────────────────────────────────────────────────────────────────

// read a geonum's cartesian coordinates by projecting onto the two axes (the afterthought)
fn coords(g: Geonum) -> (f64, f64) {
    (
        g.mag * g.angle.project(Angle::new(0.0, 1.0)),
        g.mag * g.angle.project(Angle::new(1.0, 2.0)),
    )
}

// the unit object at angle x — x lives in the angle, read out by cos_sin
fn unit(x: Angle) -> Geonum {
    Geonum::new_with_angle(1.0, x)
}

// the swept area: each edge a weighted angle (radii × the rotation), summed
fn weighted_angle_area(points: &[Geonum]) -> f64 {
    points.windows(2).map(|w| 0.5 * w[0].wedge(&w[1]).mag).sum()
}

// inscribe n chords on the unit-circle arc [0, θ] and sum their weighted angles
fn inscribed_area(n: usize, theta: f64) -> f64 {
    let pts: Vec<Geonum> = (0..=n)
        .map(|k| Geonum::new(1.0, theta * k as f64 / n as f64, PI)) // [1, (k/n)·θ]
        .collect();
    weighted_angle_area(&pts)
}
