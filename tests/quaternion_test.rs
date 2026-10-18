//! quaternions, factored
//!
//! a unit quaternion is one rotation flattened into four scalars on a leash,
//! (w, x, y, z) = (cos θ/2, sin θ/2 · axis) with w² + x² + y² + z² = 1, and the
//! hamilton product fuses two operations into one non-commutative multiply:
//!   - the composition of rotations, which commutes: geonum multiply, angles add
//!   - the plane the turn from one axis to the other draws, which flips with
//!     order: geonum wedge, a ∧ b = −(b ∧ a)
//!
//! the two parts are one angle read twice. the dot is cos of the gap between the
//! factors and the wedge is sin of the same gap, and sin is cos read a quarter turn
//! on. cos is even in the gap, cos shifted by π/2 is odd, so the dot cannot tell a
//! from b and the wedge can. the anti-commutativity quaternions build a table for
//! is the parity of a cosine moved by the quadrature
//!
//! the half-angle quaternions introduce as a device is geonum's stored coordinate,
//! t = tan(θ/2). the four slots are (1, t)/√(1+t²) spread over an axis, the sqrt
//! being what flattening t into two slots costs, and the leash is that
//! denominator. cos_sin reads the double angle rationally, (1−t²)/(1+t²), with
//! no sqrt
//!
//! the two demos every quaternion text opens with each switch one part off:
//!   - the book: quarter turns about two perpendicular axes. the axes' gap is π/2,
//!     so its even reading (the dot) is zero and its odd reading (the wedge) is one.
//!     every order-sensitive slot of the composition is the wedge term; the angle
//!     both orders share, 2π/3 about a cube diagonal, is a multiply of half-angle
//!     projections
//!   - the belt: a full turn about one axis twists the arm, a second untwists it.
//!     one axis, so the wedge is identically zero, and the −1 is the winding read
//!     at half rate. the winding story continues in spinor_test
//!
//! i·j = k and i·j·k = −1 as blade addition are in lib_test::it_computes_ijk_product;
//! the table's i·j = −j·i as the correction decomposition needs is in
//! linear_algebra_test::it_proves_quaternion_tables_add_back_what_decomposition_subtracts
//!
//! run: cargo test --test quaternion_test

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

// ---------------------------------------------------------------------------
// a quaternion is one half-angle flattened into four constrained scalars
// ---------------------------------------------------------------------------
#[test]
fn its_a_quaternion() {
    // a rotation by θ = π/3, stored once. its angle field holds t = tan(θ/2):
    // the half-angle quaternions introduce as a device is the stored coordinate
    let rotation = Angle::new(1.0, 3.0);
    let t = rotation.t();
    assert!((t - (PI / 6.0).tan()).abs() < EPSILON, "t = tan(θ/2)");

    // the quaternion reads the rotation at half rate: w = cos θ/2, |v| = sin θ/2
    let half = rotation / 2.0;
    let w = Geonum::cos(half); // the scalar slot
    let v = Geonum::sin(half); // the vector slots' shared magnitude
    let q: Quat = (w.mag, v.mag, 0.0, 0.0); // flattened onto the x axis

    // the slots are (1, t) normalized: w = 1/√(1+t²), |v| = t/√(1+t²). the sqrt
    // is what flattening t into two slots costs, and the leash w² + |v|² = 1 is
    // its denominator
    let norm = (1.0_f64 + t * t).sqrt();
    assert!((q.0 - 1.0 / norm).abs() < EPSILON, "w = 1/√(1+t²)");
    assert!((q.1 - t / norm).abs() < EPSILON, "|v| = t/√(1+t²)");
    assert!(
        (q.0 * q.0 + q.1 * q.1 - 1.0).abs() < EPSILON,
        "the leash is the denominator"
    );

    // squared as geonums the slots keep their grades: cos² lands blade 0, sin²
    // lands blade 2 — π/2 + π/2, the odd slot squared is a half turn
    let w2 = w * w;
    let v2 = v * v;
    assert_eq!(w2.angle.grade(), 0);
    assert_eq!(v2.angle.grade(), 2, "sin² is a half turn");

    // grades kept, adding the squares subtracts the magnitudes (blade 2 is the π
    // ray): cos² − sin² = cos θ, the DOUBLED angle — the readout the sandwich
    // RvR† exists to perform
    assert!(
        (w2 + v2).near(&Geonum::cos(rotation)),
        "w² + |v|² with grades kept reads cos θ — the double-angle readout"
    );

    // cos_sin reads the same double angle straight off t, (1−t²)/(1+t²), no sqrt
    let (cos_theta, _) = rotation.cos_sin();
    assert!((cos_theta - (1.0 - t * t) / (1.0 + t * t)).abs() < EPSILON);
    assert!((w2 + v2).near_mag(cos_theta));

    // one pair of squares, two readouts: grades kept reads the rotation, grades
    // dropped reads the leash. the constraint quaternions maintain is the
    // flattened shadow of their own double-angle readout
}

// ---------------------------------------------------------------------------
// the anti-commutativity is the parity of a cosine moved a quarter turn
// ---------------------------------------------------------------------------
#[test]
fn it_reads_the_anticommutativity_off_the_quadrature() {
    // two directions at a generic gap Δ = 4π/7 − π/5
    let a = Geonum::new(2.0, 1.0, 5.0); // [2, π/5]
    let b = Geonum::new(3.0, 4.0, 7.0); // [3, 4π/7]

    // the composition of rotations: angles add, so it commutes
    assert!(
        (a * b).near(&(b * a)),
        "magnitudes multiply, angles add, order-blind"
    );

    // the dot is cos Δ — even in Δ, so swapping the factors changes nothing
    assert!(a.dot(&b).near(&b.dot(&a)), "cos(−Δ) = cos Δ");

    // the wedge is sin Δ — odd in Δ, so swapping the factors adds π
    assert!(
        b.wedge(&a).near(&a.wedge(&b).negate()),
        "b ∧ a = −(a ∧ b): the quaternion i·j = −j·i"
    );

    // and sin Δ is cos Δ read a quarter turn on. turn b three quarters forward
    // (−π/2 in forward-only arithmetic) and dot: the magnitude is the wedge's
    let quarter_on = Angle::new(3.0, 2.0);
    let dot_shifted = a.dot(&b.rotate(quarter_on));
    assert!(
        dot_shifted.near_mag(a.wedge(&b).mag),
        "sin Δ = cos(Δ − π/2): the wedge is the dot read a quarter turn on"
    );
    assert_eq!(dot_shifted.angle.grade(), 0);

    // swap the factors under the same shift and the dot lands on the π ray:
    // cos(−Δ − π/2) = −sin Δ. the even cosine, moved by the quadrature, is odd,
    // and that parity is the whole anti-commutativity
    let dot_shifted_swapped = b.dot(&a.rotate(quarter_on));
    assert!(dot_shifted_swapped.near_mag(a.wedge(&b).mag));
    assert_eq!(
        dot_shifted_swapped.angle.grade(),
        2,
        "the swap is a half turn, read off the grade"
    );

    // the wedge writes the quarter turn into its own angle
    assert_eq!(
        a.wedge(&b).angle,
        a.angle + b.angle + Angle::new(1.0, 2.0),
        "θa + θb + π/2"
    );
}

// ---------------------------------------------------------------------------
// composing two rotations is order-dependent by exactly the angle between them
// ---------------------------------------------------------------------------
#[test]
fn it_makes_rotation_composition_order_dependent() {
    // a rotation is two reflections. reflecting across axis a then axis b turns
    // by 2(b − a); the other order turns by 2(a − b). the two outcomes differ by
    // 4(b − a), the geometry of the axes and nothing else
    let v = Geonum::new(1.0, 1.0, 6.0); // [1, π/6]
    let a = Geonum::new(1.0, 1.0, 4.0); // axis at π/4
    let b = Geonum::new(1.0, 5.0, 12.0); // axis at 5π/12, π/6 past a

    let a_then_b = v.reflect(&a).reflect(&b);
    let b_then_a = v.reflect(&b).reflect(&a);

    // a then b: v turned forward by 2(b − a) = π/3, to π/2
    assert!(a_then_b.angle.near_rad(PI / 2.0), "v + 2(b − a)");
    // b then a: v turned back by π/3, to −π/6 — the 11π/6 ray
    assert!(b_then_a.angle.near_rad(11.0 * PI / 6.0), "v − 2(b − a)");

    // the gap is 4(b − a) = 2π/3, the same angle the book turns through
    assert!(
        (a_then_b.angle - b_then_a.angle).near_rad(2.0 * PI / 3.0),
        "the order-dependence is the geometric angle between the axes, four times"
    );
}

// ---------------------------------------------------------------------------
// the book demo: two axes — the entire order-dependence is the wedge term
// ---------------------------------------------------------------------------
#[test]
fn it_rotates_the_book_about_two_axes() {
    // the demo: book flat on the table. quarter turn about x, then quarter turn
    // about y. restart, same turns, other order. the book lands in two different
    // orientations — the textbook motivation for a non-commutative product

    // a quarter turn at half rate, and two axes a quarter turn apart
    let half = Angle::new(1.0, 4.0); // π/4
    let axis_x = Geonum::new(1.0, 0.0, 1.0);
    let axis_y = Geonum::new(1.0, 1.0, 2.0);

    // the axes' gap Θ = π/2 read twice: the even reading is zero, the odd is one.
    // the demo picks the one gap where the dot switches off and the wedge sits at
    // its maximum — the sweep is the quadrature with the cosine gone
    let even = axis_x.dot(&axis_y);
    let odd = axis_x.wedge(&axis_y);
    assert!(
        even.near_mag(0.0),
        "cos(π/2) = 0: nothing symmetric to share"
    );
    assert!(
        odd.near_mag(1.0),
        "sin(π/2) = 1: a quarter turn draws the full plane"
    );

    // what both orders share: the composite half-angle cosine
    //   cos(c/2) = cos(a/2)cos(b/2) − sin(a/2)sin(b/2)cos Θ
    // as geonum multiplies of half-angle projections
    let sym = Geonum::cos(half) * Geonum::cos(half);
    let skew = Geonum::sin(half) * Geonum::sin(half) * even;

    // no hand-applied minus: sin·sin lands two blades up, π/2 + π/2 = π, and
    // adding at opposite angles subtracts — the formula's − sign is a position
    assert_eq!(
        skew.angle.grade(),
        2,
        "the − in the formula is the blade sum"
    );
    assert!(
        skew.near_mag(0.0),
        "perpendicular axes: the correction is a vanished projection"
    );

    let composite_cos = sym + skew;
    assert!(composite_cos.near_mag(0.5));
    assert_eq!(composite_cos.angle.grade(), 0);

    // cos(π/3) = 1/2: the composite half-angle is π/3, the full turn 2π/3 — both
    // orders turn the book 120° about a cube diagonal, (1,1,1) one way and
    // (1,1,−1) the other
    let composite_half = Angle::new(1.0, 3.0);
    assert!(Geonum::cos(composite_half).near(&composite_cos));
    assert!((composite_half * 2.0).near(&Angle::new(2.0, 3.0)));

    // what the orders do not share: the odd term sin(a/2)sin(b/2) · (axis ∧ axis),
    // the plane drawn by turning from one axis to the other. applying x then y
    // composes as y after x, so its turn is y ∧ x; the other order's is x ∧ y.
    // same magnitude, a half turn apart
    let odd_xy = Geonum::sin(half) * Geonum::sin(half) * axis_y.wedge(&axis_x);
    let odd_yx = Geonum::sin(half) * Geonum::sin(half) * axis_x.wedge(&axis_y);
    assert!(odd_xy.near_mag(0.5));
    assert!(odd_yx.near_mag(0.5));
    assert_eq!(odd_xy.angle.grade(), 2);
    assert_eq!(
        odd_yx.angle.grade(),
        0,
        "reverse the turn and the plane it draws flips a half turn"
    );

    // the entire order gap is that one term, taken twice. everything celebrated
    // as "3D rotations don't commute" is one odd reading of one angle
    let gap = odd_xy - odd_yx;
    assert!(
        gap.near_mag(1.0),
        "q₁q₂ − q₂q₁ = 2·v₁∧v₂, magnitude 2 sin²(π/4) sin(π/2) = 1"
    );

    // scaffolding corroborates: the hamilton product, slot by slot
    let c = (PI / 4.0).cos();
    let s = (PI / 4.0).sin();
    let q_x: Quat = (c, s, 0.0, 0.0);
    let q_y: Quat = (c, 0.0, s, 0.0);
    let x_then_y = hamilton(q_y, q_x); // apply x first: q_y ⊗ q_x
    let y_then_x = hamilton(q_x, q_y);

    // the scalar and both axis slots are order-blind, and the scalar is the
    // composite cosine
    assert!(
        (x_then_y.0 - y_then_x.0).abs() < EPSILON,
        "scalar slot: order-blind"
    );
    assert!(
        (x_then_y.1 - y_then_x.1).abs() < EPSILON,
        "x slot: order-blind"
    );
    assert!(
        (x_then_y.2 - y_then_x.2).abs() < EPSILON,
        "y slot: order-blind"
    );
    assert!((x_then_y.0 - composite_cos.mag).abs() < EPSILON);

    // the z slot is the odd term, and it is the only slot that flips
    assert!(
        (x_then_y.3 + odd_xy.mag).abs() < EPSILON,
        "z = −sin²(π/4): the grade-2 landing"
    );
    assert!(
        (y_then_x.3 - odd_yx.mag).abs() < EPSILON,
        "z = +sin²(π/4): the grade-0 landing"
    );
    assert!(
        ((x_then_y.3 - y_then_x.3).abs() - gap.mag).abs() < EPSILON,
        "the whole order gap is the z slot"
    );

    // the gap between the two outcomes is itself a rotation, q_xy ⊗ q_yx⁻¹, and
    // its scalar slot is again 1/2: the orders disagree by another 2π/3. the
    // composite, its reverse-order twin, and their gap share one angle, the
    // three-fold symmetry of the cube corner the book pivots around
    let between = hamilton(x_then_y, conj(y_then_x));
    assert!(
        (between.0 - composite_cos.mag).abs() < EPSILON,
        "the order gap is the same 2π/3 the composites turn through"
    );
}

// ---------------------------------------------------------------------------
// the belt trick: one axis — zero wedge, the −1 is the winding at half rate
// ---------------------------------------------------------------------------
#[test]
fn it_twists_the_arm_holding_the_book() {
    // the other classic: hold the book in your palm, rotate it a full 360° —
    // the arm is twisted. 360° more — untwisted. quaternions carry the demo as
    // q(2π) = −1, q(4π) = +1, the double cover

    // one axis: there is no turn from the axis to itself, so the odd term
    // x ∧ x = 0 draws no plane. the book demo was all wedge and no dot; this
    // demo is all dot and no wedge — the other factor
    let axis_x = Geonum::new(1.0, 0.0, 1.0);
    assert!(
        axis_x.wedge(&axis_x).near_mag(0.0),
        "no turn from an axis to itself, no plane drawn"
    );

    // with the odd term gone nothing is order-sensitive: same-axis turns are
    // the commuting multiply (it_reads_the_anticommutativity_off_the_quadrature),
    // and the demo's whole content is winding. a full turn arrives four blades
    // on with the orientation unchanged
    let book = Geonum::new(1.0, 1.0, 6.0);
    let turn = Angle::new(2.0, 1.0); // 2π

    let once = book.rotate(turn);
    assert!(
        once.base_angle().near(&book.base_angle()),
        "the book looks the same at 2π"
    );
    assert_eq!(
        once.angle.blade(),
        book.angle.blade() + 4,
        "the arm holds four more quarter turns"
    );

    // the quaternion is the rotation at half rate: halve the winding and read
    // the grade. 2π halves to π — grade 2, the −1. a second turn halves to 2π —
    // grade 0, the +1: the arm untwists
    let spinor_once = (once.angle - book.angle) / 2.0;
    assert_eq!(
        spinor_once.grade(),
        2,
        "2π halves to π: −q, the twisted arm"
    );

    let twice = once.rotate(turn);
    let spinor_twice = (twice.angle - book.angle) / 2.0;
    assert_eq!(spinor_twice.grade(), 0, "4π halves to 2π: +q, untwisted");

    // scaffolding corroborates: q(π)² = −1 with every cross slot zero, the −1
    // coming from the symmetric −v·v term, and same-axis products order-blind
    let half_turn: Quat = ((PI / 2.0).cos(), (PI / 2.0).sin(), 0.0, 0.0); // q(π) = (0, x)
    let full_turn_q = hamilton(half_turn, half_turn); // i² = −1
    assert!(
        (full_turn_q.0 + 1.0).abs() < EPSILON,
        "q(2π) = −1: the sign the belt trick makes physical"
    );
    for cross in [full_turn_q.1, full_turn_q.2, full_turn_q.3] {
        assert!(cross.abs() < EPSILON, "zero odd term: the −1 is −v·v");
    }
    let quarter_turn: Quat = ((PI / 4.0).cos(), (PI / 4.0).sin(), 0.0, 0.0);
    let ab = hamilton(half_turn, quarter_turn);
    let ba = hamilton(quarter_turn, half_turn);
    for (l, r) in [(ab.0, ba.0), (ab.1, ba.1), (ab.2, ba.2), (ab.3, ba.3)] {
        assert!((l - r).abs() < EPSILON, "one axis: order-blind");
    }

    // the −1 read as a position: cos of the halved winding is magnitude 1 at
    // grade 2, the scaffolding's scalar slot as a landing
    let readout = Geonum::cos(spinor_once);
    assert!(readout.near_mag(full_turn_q.0.abs()));
    assert_eq!(
        readout.angle.grade(),
        2,
        "the scaffolding's −1, a grade-2 landing"
    );

    // quaternions stop counting at the cover, q(4π) = q(0). the angle is the
    // count: blade 8 is a different address from blade 0 at the same grade.
    // spinor_test carries this to the belt parity class
    assert_eq!(
        twice.angle.blade(),
        book.angle.blade() + 8,
        "4π is +q to the quaternion, eight blades to the angle"
    );
    assert_eq!(twice.angle.grade(), book.angle.grade());
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// scaffolding: quaternion components (w, x, y, z) = (cos θ/2, sin θ/2 · axis),
// the decomposed algebra the demos are traditionally read through.
// assertion-side reference only — the geonum side computes with angle arithmetic
type Quat = (f64, f64, f64, f64);

// (w₁, v₁)(w₂, v₂) = (w₁w₂ − v₁·v₂, w₁v₂ + w₂v₁ + v₁×v₂)
fn hamilton(a: Quat, b: Quat) -> Quat {
    (
        a.0 * b.0 - a.1 * b.1 - a.2 * b.2 - a.3 * b.3,
        a.0 * b.1 + a.1 * b.0 + a.2 * b.3 - a.3 * b.2,
        a.0 * b.2 - a.1 * b.3 + a.2 * b.0 + a.3 * b.1,
        a.0 * b.3 + a.1 * b.2 - a.2 * b.1 + a.3 * b.0,
    )
}

fn conj(q: Quat) -> Quat {
    (q.0, -q.1, -q.2, -q.3)
}
