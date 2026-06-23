// the geometry under the geometry. point, line, metric — the primitives every coordinate system is
// built ON — are not bedrock; they are rotations and projections, and these tests show it. a point
// is a located rotation, [r, θ]. a line is a projection, r·cos, its sign a π turn, formed from
// points. the metric is that π turn itself: self-cancelling, pre-projection, the one coordinate-free
// fact a metric carries — its signature. keep the angle and nothing is left to compute: the length
// is the magnitude, handed over directly. drop it and the bills come due — √(x²+y²) to buy the
// length back, a tensor to buy duality back. each test charges that bill to Cartesian and asserts
// against the gap, never a value it builds.

use geonum::*;

#[test]
fn its_a_metric() {
    // the metric is the π rotation, and it emerges rather than being asserted: square a unit vector
    // through the geometric product and it lands on the dual of the scalar — e² = −1. the metric's
    // minus sign IS the π dual. and the dual is self-cancelling: e⁴ folds back to the scalar grade
    let e = Geonum::new(1.0, 1.0, 2.0); // a unit vector, [1, π/2]

    let e_squared = e * e; // mags multiply, angles add → [1, π]
    let minus_one = Geonum::scalar(1.0).dual(); // the −1: the scalar rotated π
    assert!(
        e_squared.near(&minus_one),
        "e² = the dual of 1 = −1 — the metric's minus sign emerges as the π rotation"
    );

    let e_fourth = e_squared * e_squared; // (−1)² → [1, 2π]
    assert_eq!(
        e_fourth.angle.grade(),
        Geonum::scalar(1.0).angle.grade(),
        "self-cancelling: e⁴ folds back to the scalar — π + π returns to grade 0"
    );
}

#[test]
fn its_a_point() {
    // a point is a location — a magnitude reached by a rotation, [r, θ]. lines form from points:
    // the segment between two points is their difference, a geonum whose magnitude is the line's
    // length and whose angle is its direction. a point and the line it spans are one kind of object
    let p1 = Geonum::new_from_cartesian(1.0, 2.0); // a point
    let p2 = Geonum::new_from_cartesian(4.0, 6.0); // another point

    let segment = p2 - p1; // the line from p1 to p2 — formed by subtracting points
    assert!(
        segment.near_mag(5.0),
        "the segment is the 3-4-5 between the points"
    );
    assert_eq!(
        std::mem::size_of_val(&segment),
        std::mem::size_of_val(&p1),
        "the line is the same two numbers as a point — formed from points, not a new type"
    );
}

#[test]
fn its_a_line() {
    // a line is not a primitive drawn in + and − directions — it is a projection: a magnitude cast
    // through an angle onto a reference. [r, θ].project(φ) is r·cos(θ−φ) lying along φ, and the sign
    // is that cosine. there is no stored line; project produces it, and "negative" returns as a π turn
    let x_axis = Geonum::new(1.0, 0.0, 1.0); // the reference direction, +x

    // the line of [r, θ] along the reference is its magnitude scaled by cos(θ−φ)
    let v = Geonum::new(4.0, 1.0, 3.0); // [4, π/3]
    let line = v.project(&x_axis);
    assert!(
        line.near_mag(2.0),
        "the line is r·cos(θ−φ) = 4·cos(π/3) = 2"
    );
    assert_eq!(
        line.angle, x_axis.angle,
        "it lies along the reference direction"
    );

    // the sign is the cosine sweeping: full at Δ=0, gone at Δ=π/2, and at Δ=π it comes back as
    // positive magnitude a half-turn around — "negative" is a π rotation, not a second direction
    let along = Geonum::new(2.0, 0.0, 1.0); // Δ = 0
    let across = Geonum::new(2.0, 1.0, 2.0); // Δ = π/2
    let opposite = Geonum::new(2.0, 1.0, 1.0); // Δ = π

    assert!(
        along.project(&x_axis).near_mag(2.0),
        "Δ=0: cos 0 = 1, the full line"
    );
    assert!(
        across.project(&x_axis).near_mag(0.0),
        "Δ=π/2: cos = 0, the line vanishes"
    );

    let neg = opposite.project(&x_axis);
    assert!(neg.near_mag(2.0), "Δ=π: the full magnitude is back");
    assert!(
        neg.angle.is_opposite(&x_axis.angle),
        "but a half-turn around — the negative line is +x rotated π, not a drawn −direction"
    );
}

#[test]
fn it_overshoots_the_length_when_it_sums_the_legs() {
    // the Cartesian legs r cos φ + r sin φ do NOT sum to the straight length r. at 45° their sum is
    // √2·r — reading the point off the axes costs the length the projection threw away
    let point = Geonum::new(3.0, 1.0, 4.0); // [r = 3, φ = π/4]
    let x = point.mag * point.angle.project(Angle::new(0.0, 1.0)); // r cos φ
    let y = point.mag * point.angle.project(Angle::new(1.0, 2.0)); // r sin φ

    let leg_sum = x + y;
    assert!(
        leg_sum > point.mag,
        "the legs sum to more than the straight length r"
    );
    assert!(
        (leg_sum - point.mag * 2.0_f64.sqrt()).abs() < 1e-10,
        "at 45° the leg-sum is exactly √2·r, not r"
    );
}

#[test]
fn it_recovers_the_length_only_by_squaring() {
    // the legs alone dont give the length back — only the squared norm does. √(x²+y²) = r works
    // because x²+y² = r²(cos²φ+sin²φ) and cos²φ+sin²φ = 1. the square root is the cost of the angle
    let point = Geonum::new(5.0, 1.0, 7.0); // [r = 5, φ = π/7]
    let x = point.mag * point.angle.project(Angle::new(0.0, 1.0));
    let y = point.mag * point.angle.project(Angle::new(1.0, 2.0));

    assert!(
        (x + y - point.mag).abs() > 1.0,
        "the raw legs dont equal the length"
    );
    assert!(
        ((x * x + y * y).sqrt() - point.mag).abs() < 1e-10,
        "√(x²+y²) = r — the length reconstructed only by squaring"
    );
}

#[test]
fn it_carries_the_length_independent_of_the_angle() {
    // the length is not built from the legs: rotate the point and both Cartesian legs swing, while
    // the magnitude is untouched. geonum holds the length as a primitive — nothing to recompute
    let point = Geonum::new(5.0, 1.0, 6.0); // [r = 5, φ = π/6]
    let x_axis = Angle::new(0.0, 1.0);

    let leg_before = point.mag * point.angle.project(x_axis);
    let rotated = point.rotate(Angle::new(2.0, 5.0)); // rotate by 2π/5
    let leg_after = rotated.mag * rotated.angle.project(x_axis);

    assert!(
        (leg_after - leg_before).abs() > 1.0,
        "the Cartesian x-leg swings under the rotation"
    );
    assert!(
        rotated.near_mag(point.mag),
        "the length is unchanged — it never depended on the legs"
    );
}

#[test]
fn it_measures_length_in_any_dimension_without_a_squared_sum() {
    // Cartesian length in n dimensions is √(x₁²+…+xₙ²) — an n-term squared sum that grows with the
    // dimension. geonum reads the length off a computed segment as one number, the same whether the
    // segment lies in the plane or a thousand dimensions out
    let p1 = Geonum::new_from_cartesian(1.0, 2.0);
    let p2 = Geonum::new_from_cartesian(4.0, 6.0);
    let plane_length = (p2 - p1).mag; // the 3-4-5 segment, computed
    assert!(
        (plane_length - 5.0).abs() < 1e-10,
        "the segment is 3-4-5 → length 5 in the plane"
    );

    // the same two points swept 1000 quarter-turns out, then subtracted — length read the same way
    let out = Angle::new_with_blade(1000, 0.0, 1.0);
    let p1_hi = Geonum::new_with_angle(p1.mag, p1.angle + out);
    let p2_hi = Geonum::new_with_angle(p2.mag, p2.angle + out);
    let hi_length = (p2_hi - p1_hi).mag;
    assert!(
        (hi_length - plane_length).abs() < 1e-10,
        "same length 1000 dimensions out — one number, no 1000-term √(Σxᵢ²)"
    );
}
