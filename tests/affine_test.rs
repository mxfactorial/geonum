//! affine geometry takedown
//!
//! affine geometry erects a coordinate superstructure — homogeneous coordinates,
//! augmented matrices, translation vectors — to do what geonum already does with
//! three core operations. this file dismantles each affine primitive by rebuilding
//! it from the core:
//!
//! - translation is `+` — and underneath, a conserved projection, not a stored move
//! - shear is `rotate` — affine "shear" is a uniform angle add, not even a shear
//! - quadrilateral area is one `wedge` — no vertices, no coordinates
//! - the blade's grade is live rotation; its winding (blade / 4) is inert — a full
//!   turn moves nothing and no projection sees it. winding is rotation's
//!   bookkeeping, not a translation; the operation that moves a point is `+`
//!
//! the `Affine` trait and its cargo feature were removed once these held: the layer
//! carried nothing the core didnt
//!
//! the last test closes the riff this file grew out of. y = mx + b is a single
//! geonum — a scalar leg and a vector leg one blade apart — which is why
//! ∫(mx + b) dx never appears in integral_test: the affine line is the
//! zero-curvature degenerate case and adds no integration content

use geonum::{Angle, Geonum};

const EPSILON: f64 = 1e-10;

// ---------------------------------------------------------------------------
// translation is addition
// ---------------------------------------------------------------------------
#[test]
fn it_dissolves_translation_into_addition() {
    // the matrix story needs homogeneous coordinates and a 3×3 augmented matrix to
    // translate a 2D point. geonum needs `+`
    let point = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]
    let displacement = Geonum::new(3.0, 1.0, 2.0); // [3, π/2]

    let translated = point + displacement; // the whole operation

    // grade survives — nothing affine-specific happened, only addition
    assert_eq!(translated.angle.grade(), point.angle.grade());

    // reversal is `+` with the negated displacement. geonum `+` records its path in
    // the blade, so the point returns in magnitude and direction
    let back = translated + displacement.negate();
    assert!(back.mag_diff(&point) < EPSILON);
    assert!((back.angle - point.angle).rem() < EPSILON);
}

// ---------------------------------------------------------------------------
// translation is a conserved projection, so there is nothing to store
// ---------------------------------------------------------------------------
#[test]
fn it_exposes_translation_as_a_conserved_projection() {
    // affine geometry materializes a translated copy of every point. but translation
    // moves only ONE shadow: the projection along the displacement shifts by a
    // supplied value, the orthogonal projection never moves. the conserved shadow
    // doesnt change across its axis, so it needs no computing at all
    let point = Geonum::new(5.0, 1.0, 6.0);
    let displacement = Geonum::new(3.0, 1.0, 2.0);
    let moved = point + displacement;

    let across = displacement.angle + Angle::new(1.0, 2.0); // a quarter turn off the displacement
    let across_before = point.mag * point.angle.project(across);
    let across_after = moved.mag * moved.angle.project(across);
    assert!(
        (across_before - across_after).abs() < EPSILON,
        "the orthogonal shadow is conserved through the translation"
    );

    let along_before = point.mag * point.angle.project(displacement.angle);
    let along_after = moved.mag * moved.angle.project(displacement.angle);
    assert!(
        (along_after - along_before - displacement.mag).abs() < EPSILON,
        "the parallel shadow shifts by exactly the displacement magnitude"
    );
}

// ---------------------------------------------------------------------------
// the blade's grade is live rotation; its winding is inert — a full turn moves
// nothing. winding is rotation's bookkeeping, not a translation; the live
// translation is `+`
// ---------------------------------------------------------------------------
#[test]
fn it_keeps_the_winding_inert_while_the_grade_stays_live() {
    // affine bolts rotation and translation together as separate machinery. geonum
    // keeps one blade — but its two halves are not symmetric affine motions. the
    // grade (blade % 4) is LIVE rotation: it drives every cos_sin-based readout. the
    // winding (blade / 4) is INERT: a full turn records +4 yet relocates nothing,
    // and no projection can see it
    let p = Geonum::new(5.0, 1.0, 6.0); // [5, π/6]
    let wound = p.rotate(Angle::new(2.0, 1.0)); // +2π → blade +4, one full turn
    let turned = p.rotate(Angle::new(1.0, 2.0)); // +π/2 → blade +1, grade 0 → 1

    // the turn is recorded, but the point never left: same magnitude, same
    // direction, zero displacement — winding is not a translation
    assert_eq!(
        wound.angle.blade(),
        p.angle.blade() + 4,
        "the full turn lands as +4"
    );
    assert!(
        wound.near_mag(p.mag),
        "winding leaves the magnitude untouched"
    );
    assert_eq!(
        wound.angle.grade(),
        p.angle.grade(),
        "winding leaves the direction untouched"
    );
    assert!(
        (wound - p).mag < EPSILON,
        "a full turn produces zero displacement — it moves nothing to translate"
    );

    // projection is blind to the winding but swings under one blade of rotation —
    // the asymmetry: q invisible, r live
    let onto = Angle::new(0.0, 1.0); // +x
    assert!(
        (wound.angle.project(onto) - p.angle.project(onto)).abs() < EPSILON,
        "the winding is invisible to projection"
    );
    assert!(
        (turned.angle.project(onto) - (-0.5)).abs() < EPSILON,
        "rotation is live: the +x projection swings to cos(2π/3) = -0.5"
    );
}

// ---------------------------------------------------------------------------
// shear is rotation
// ---------------------------------------------------------------------------
#[test]
fn it_dissolves_shear_into_rotation() {
    // affine "shear" here adds the same angle to everything, which is a rotation —
    // a real shear is non-uniform. geonum spells it `rotate`
    let point = Geonum::new(5.0, 1.0, 3.0); // [5, π/3]
    let amount = Angle::new(1.0, 6.0); // π/6

    let sheared = point.rotate(amount);

    assert!(sheared.near_mag(point.mag)); // magnitude untouched
    assert!(sheared.angle.near(&(point.angle + amount))); // angle gains the amount
                                                          // π/3 + π/6 = π/2 crosses a grade boundary, 0 → 1
    assert_eq!(point.angle.grade(), 0);
    assert_eq!(sheared.angle.grade(), 1);

    // the "preserves parallelism" claim is trivial under a uniform angle add: two
    // parallel directions take the same rotation and stay parallel
    let dir1 = Geonum::new(2.0, 0.0, 1.0);
    let dir2 = Geonum::new(3.0, 0.0, 1.0);
    let quarter = Angle::new(1.0, 4.0);
    assert!(dir1.rotate(quarter).angle.near(&dir2.rotate(quarter).angle));
}

// ---------------------------------------------------------------------------
// quadrilateral area is one wedge
// ---------------------------------------------------------------------------
#[test]
fn it_dissolves_quadrilateral_area_into_one_wedge() {
    // affine area triangulates four coordinate vertices. a parallelogram's area is
    // one wedge of its two edge geonums — two [mag, angle] numbers, no vertices
    let base = Geonum::new(4.0, 0.0, 1.0); // edge [4, 0]
    let side = Geonum::new(3.0, 1.0, 2.0); // edge [3, π/2]
    assert!(
        base.wedge(&side).near_mag(12.0),
        "the 4×3 rectangle is one wedge"
    );

    // any side angle, the same primitive — the wedge carries |base||side|sin(Δθ)
    let slanted = Geonum::new(3.0, 1.0, 3.0); // [3, π/3]
    let expected = 4.0 * 3.0 * Angle::new(1.0, 3.0).cos_sin().1; // |base||side| sin(π/3)
    assert!(base.wedge(&slanted).near_mag(expected));
}

// ---------------------------------------------------------------------------
// y = mx + b is a single geonum — which is why integral_test ignores it
// ---------------------------------------------------------------------------
#[test]
fn it_collapses_the_affine_line_into_a_single_geonum() {
    // y = mx + b is not a number plus a translation. it is ONE geonum: a scalar leg
    // (b, blade 0) and a vector leg (mx, blade 1) a single π/2 turn apart. the "+"
    // is the orthogonal combination of two grades, the way b + i(mx) is one number
    let (m, b) = (2.0, 3.0);

    for &x in &[0.0, 1.0, 2.5, 10.0] {
        let scalar_leg = Geonum::new(b, 0.0, 1.0); // b at blade 0
        let vector_leg = Geonum::new(m * x, 1.0, 2.0); // mx at blade 1
        let y = scalar_leg + vector_leg; // one geonum

        // the intercept is the conserved projection: adj == b for every x, the shadow
        // that never moves as the line runs
        assert!(
            y.adj().near_mag(b),
            "the intercept b is the conserved scalar leg"
        );
        // the slope term is the other leg, linear in x
        assert!(y.opp().near_mag(m * x), "mx is the vector leg");
    }

    // the two legs sit exactly one blade apart — the line is a single rotation
    // between its scalar and vector grades, not a coordinate pair
    let scalar_leg = Geonum::new(b, 0.0, 1.0);
    let vector_leg = Geonum::new(m, 1.0, 2.0);
    assert_eq!(vector_leg.angle.blade() - scalar_leg.angle.blade(), 1);

    // this is why ∫(mx + b) dx is absent from integral_test: the affine line is the
    // zero-curvature case — it turns 0 (constant angle), so its area is a single
    // wedge, and +b is a conserved projection adding nothing to integrate. the line
    // carries no integration content the trig, telescoping, and swept-curve tests
    // dont already cover
}
