//! quaternions, factored
//!
//! quaternion multiplication packs three things into one non-commutative product: the
//! composition of rotations, the oriented plane they turn in, and the closure to −1. geonum
//! keeps them as separate operations and lets the blade carry the structure:
//!   - the rotor is MULTIPLY — angles add, commutative (same-plane rotations compose this way)
//!   - the oriented plane is the WEDGE — anti-symmetric, a ∧ b = −(b ∧ a): this is i·j = −j·i
//!   - the blade carries grade and winding: i·j = k and i·j·k = −1 are blade arithmetic
//!
//! the non-commutativity quaternions need is in the wedge, not the multiply: reading k·i off
//! the commutative multiply finds none, because the anti-symmetry lives in the wedge.
//! composing two rotations is order-dependent by exactly the angle between them — the
//! geometry, read off the blade, never collapsed to a scalar shadow
//!
//! its_a_quaternion names the object — one half-angle flattened into four scalars riding a
//! norm constraint — and splits its product into a symmetric (order-blind) and an
//! anti-symmetric (order-sensitive) part. the two demonstrations every quaternion text
//! opens with each showcase exactly one:
//!   - the book demo — quarter turn about x then y, versus y then x: every order-sensitive
//!     component of the composition is the cross (wedge) term; the composite angle both
//!     orders share (2π/3, the cube-diagonal turn) is a geonum multiply of half-angle
//!     projections
//!   - the belt trick — a full turn twists the arm, a second untwists it: one axis, so the
//!     wedge is identically zero, and the celebrated −1 is the winding read at half rate.
//!     the winding story is proven in spinor_test (−1 at 2π, rauch interferometry, the belt
//!     parity class); here it reads as the wedge-free complement of the book demo
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
    // a unit quaternion is four scalars (w, x, y, z) = (cos θ/2, sin θ/2 · axis)
    // riding one constraint w² + x² + y² + z² = 1: four slots and a leash to
    // store one rotation. the exponential form texts write, q = e^(n̂θ/2), is
    // exponential_test's e^(iθ) = [1, θ] read at half-angle — geonum stores the
    // rotation, the half-angle is the angle field itself
    // (spinor_test::it_stores_the_spinor_half_angle_as_t: t IS tan(θ/2)) —
    // and the slots are its two projections

    // a rotation by θ = π/3, stored at half-angle
    let half = Angle::new(1.0, 6.0); // θ/2 = π/6
    let w = Geonum::cos(half); // the scalar slot
    let v = Geonum::sin(half); // the vector slots' shared magnitude
    let q: Quat = (w.mag, v.mag, 0.0, 0.0); // flattened onto the x axis
    assert!((q.0 - (PI / 6.0).cos()).abs() < EPSILON, "w = cos θ/2");
    assert!((q.1 - (PI / 6.0).sin()).abs() < EPSILON, "|v| = sin θ/2");

    // squared, the slots keep their grades: cos² lands blade 0, sin² lands
    // blade 2 — π/2 + π/2, the odd slot squared is a half turn. this is the
    // blade arithmetic geometry_test::its_a_metric proves as e² = −1
    let w2 = w * w;
    let v2 = v * v;
    assert_eq!(w2.angle.grade(), 0);
    assert_eq!(
        v2.angle.grade(),
        2,
        "sin² is a half turn, not a positive scalar"
    );

    // grades kept, adding the squares subtracts the magnitudes (blade 2 is the
    // π ray): cos² − sin² = cos θ — the DOUBLED angle, the readout the sandwich
    // RvR† exists to perform
    assert!(
        (w2 + v2).near(&Geonum::cos(Angle::new(1.0, 3.0))),
        "w² + |v|² with grades kept reads cos θ — the double-angle readout"
    );

    // grades dropped — the |·|² the four slots take — the same squares sum to
    // the constraint: w² + |v|² = 1. the identity itself is proven as a
    // projection in trigonometry_test::it_derives_pythagorean_identity_from_quadrature;
    // the point here is the contrast: one pair of squares, two readouts. the
    // leash quaternions maintain is the flattened shadow of their own
    // double-angle readout
    let (ch, sh) = half.cos_sin();
    assert!(
        (ch * ch + sh * sh - 1.0).abs() < EPSILON,
        "the unit norm is the projection identity, not a maintained invariant"
    );

    // composing two quaternions, the hamilton product mixes a symmetric part
    // (w₁w₂ − v₁·v₂ and the w·v terms — order-blind) with an anti-symmetric
    // part (v₁×v₂ — flips with order). generic half-angles, skew axes:
    let (ca, sa) = ((PI / 5.0).cos(), (PI / 5.0).sin());
    let (cb, sb) = ((PI / 7.0).cos(), (PI / 7.0).sin());
    let q1: Quat = (ca, sa, 0.0, 0.0); // about x
    let q2: Quat = (cb, 0.0, 0.6 * sb, 0.8 * sb); // about (0, 0.6, 0.8)
    let ab = hamilton(q1, q2);
    let ba = hamilton(q2, q1);

    // swapping the order moves exactly 2·v₁×v₂ and nothing else
    let cross = (0.0, -0.8 * sa * sb, 0.6 * sa * sb); // v₁×v₂ by hand
    assert!(
        (ab.0 - ba.0).abs() < EPSILON,
        "the symmetric part is order-blind"
    );
    assert!((ab.1 - ba.1 - 2.0 * cross.0).abs() < EPSILON);
    assert!((ab.2 - ba.2 - 2.0 * cross.1).abs() < EPSILON);
    assert!((ab.3 - ba.3 - 2.0 * cross.2).abs() < EPSILON);

    // the demos below each shut one part off: perpendicular axes put the whole
    // order-dependence in the anti-symmetric term (the book), one axis kills
    // it and leaves the winding (the belt)
}

// ---------------------------------------------------------------------------
// a geonum is a plane — the wedge of two directions is a grade-2 bivector
// ---------------------------------------------------------------------------
#[test]
fn its_a_plane() {
    let a = Geonum::new(1.0, 0.0, 1.0); // [1, 0]
    let b = Geonum::new(1.0, 1.0, 2.0); // [1, π/2]
    let plane = a.wedge(&b);
    assert_eq!(
        plane.angle.grade(),
        2,
        "the wedge of two directions is a bivector — a plane"
    );
    assert!(plane.near_mag(1.0), "unit area: |a||b|sin(π/2) = 1");
}

// ---------------------------------------------------------------------------
// the anti-commutativity i·j = −j·i lives in the WEDGE: a ∧ b = −(b ∧ a)
// ---------------------------------------------------------------------------
#[test]
fn it_keeps_the_anticommutativity_in_the_wedge() {
    let a = Geonum::new(1.0, 0.0, 1.0);
    let b = Geonum::new(1.0, 1.0, 2.0);
    assert!(
        b.wedge(&a).near(&a.wedge(&b).negate()),
        "b ∧ a = −(a ∧ b): reversing order flips orientation — the quaternion anti-commutativity"
    );
}

// ---------------------------------------------------------------------------
// the rotor is MULTIPLY — angles add, so it commutes (same-plane rotations do)
// ---------------------------------------------------------------------------
#[test]
fn it_composes_the_rotor_commutatively() {
    let a = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
    let b = Geonum::new(3.0, 1.0, 4.0); // [3, π/4]
    assert!(
        (a * b).near(&(b * a)),
        "the rotor multiply commutes: magnitudes multiply, angles add"
    );
}

// ---------------------------------------------------------------------------
// geonum factors what quaternions fuse: multiply (commuting rotor) and wedge
// (anti-symmetric plane) are distinct operations, bundled into one product by ℍ
// ---------------------------------------------------------------------------
#[test]
fn it_factors_what_the_quaternion_product_fuses() {
    let a = Geonum::new(1.0, 0.0, 1.0);
    let b = Geonum::new(1.0, 1.0, 2.0);
    assert!(
        !(a * b).near(&a.wedge(&b)),
        "multiply and wedge are distinct operators"
    );
    assert!((a * b).near(&(b * a)), "multiply commutes");
    assert!(
        !a.wedge(&b).near(&b.wedge(&a)),
        "wedge does not — the property the quaternion product fuses into its one multiply"
    );
}

// ---------------------------------------------------------------------------
// i·j = k and i·j·k = −1 are blade arithmetic, the winding kept in the blade
// ---------------------------------------------------------------------------
#[test]
fn it_carries_ijk_to_negative_one() {
    let i = Geonum::create_dimension(1.0, 1); // blade 1
    let j = Geonum::create_dimension(1.0, 2); // blade 2
    let k = Geonum::create_dimension(1.0, 3); // blade 3

    assert!((i * j).near(&k), "i·j = k: blades add, 1 + 2 = 3");

    let ijk = i * j * k; // blade 6
    assert_eq!(
        ijk.angle.grade(),
        2,
        "i·j·k = −1: blade 6, grade 2, the negative real ray"
    );
    assert!(
        ijk.near_mag(1.0),
        "magnitude 1 — a unit, not a scalar collapse"
    );
    assert_eq!(
        ijk.angle.blade(),
        6,
        "blade 6 keeps the winding, not reduced to grade 2"
    );
}

// ---------------------------------------------------------------------------
// composing two rotations is order-dependent by exactly the angle between them.
// a rotation is two reflections; reflecting across axis a then b is a rotation by
// 2(b−a), the reverse order the reverse rotation — the gap is 4(b−a), the geometry
// ---------------------------------------------------------------------------
#[test]
fn it_makes_rotation_composition_order_dependent() {
    let v = Geonum::new(1.0, 1.0, 6.0); // [1, π/6]
    let a = Geonum::new(1.0, 1.0, 4.0); // axis at π/4
    let b = Geonum::new(1.0, 5.0, 12.0); // axis at 5π/12

    let a_then_b = v.reflect(&a).reflect(&b);
    let b_then_a = v.reflect(&b).reflect(&a);

    assert!(
        !a_then_b.near(&b_then_a),
        "reflect-a-then-b ≠ reflect-b-then-a: composing rotations does not commute"
    );

    let gap = (a_then_b.angle - b_then_a.angle).grade_angle();
    assert!(
        (gap - 4.0 * (5.0 / 12.0 - 1.0 / 4.0) * PI).abs() < 1e-9,
        "the gap is exactly 4·(b−a) = 2π/3 — the order-dependence is the geometric angle, not noise"
    );
}

// ---------------------------------------------------------------------------
// not a cross-product cycle. create_dimension walks the GRADE cycle, so blades
// 0,1,2 are a scalar, a vector, a bivector — not three basis vectors. expecting
// e1∧e2=e3, e2∧e3=e1, e3∧e1=e2 to close treats them as cartesian axes and judges
// the wrap-around by its projected angle, dropping the blade that distinguishes them
// ---------------------------------------------------------------------------
#[test]
fn it_keeps_the_blade_where_the_cross_product_cycle_looks_broken() {
    let e1 = Geonum::create_dimension(1.0, 0); // blade 0
    let e2 = Geonum::create_dimension(1.0, 1); // blade 1
    let e3 = Geonum::create_dimension(1.0, 2); // blade 2

    // these are three GRADES, not three vectors
    assert_eq!(e1.angle.grade(), 0, "blade 0 — a scalar");
    assert_eq!(e2.angle.grade(), 1, "blade 1 — a vector");
    assert_eq!(e3.angle.grade(), 2, "blade 2 — a bivector");

    // the wrap-around e3 ∧ e1 reads magnitude 0 — but that is the SHADOW. the wedge magnitude
    // is sin(projected gap), and e1 (angle 0) and e3 (angle π) are π apart, so sin(π) = 0. the
    // blade never entered the sine
    assert!(
        e3.wedge(&e1).near_mag(0.0),
        "the projected gap is π, sin(π) = 0 — no area in the shadow"
    );

    // but the blade keeps e1 and e3 apart: two blades, distinct grades. calling the cycle
    // broken collapses them to their projected direction and forgets the winding
    assert_ne!(
        e1.angle.blade(),
        e3.angle.blade(),
        "blade 0 ≠ blade 2 — the winding distinguishes them; they are not anti-parallel vectors"
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

    // a quaternion carries the rotation at half-angle: a quarter turn about an
    // axis is (cos π/4, sin π/4 · axis)
    let c = (PI / 4.0).cos();
    let s = (PI / 4.0).sin();

    let q_x = (c, s, 0.0, 0.0);
    let q_y = (c, 0.0, s, 0.0);

    // composing under q v q⁻¹: apply x first means q_y ⊗ q_x
    let x_then_y = hamilton(q_y, q_x);
    let y_then_x = hamilton(q_x, q_y);

    // the hand-derived reference: (c², cs, cs, ∓s²)
    for (got, want) in [
        (x_then_y.0, c * c),
        (x_then_y.1, c * s),
        (x_then_y.2, c * s),
        (x_then_y.3, -s * s),
        (y_then_x.3, s * s),
    ] {
        assert!((got - want).abs() < EPSILON);
    }

    // locate the order-dependence: subtract the orders component by component.
    // the scalar and both axis terms are order-blind
    assert!(
        (x_then_y.0 - y_then_x.0).abs() < EPSILON,
        "scalar part: order-blind"
    );
    assert!(
        (x_then_y.1 - y_then_x.1).abs() < EPSILON,
        "x term: order-blind"
    );
    assert!(
        (x_then_y.2 - y_then_x.2).abs() < EPSILON,
        "y term: order-blind"
    );

    // the ONLY order-sensitive component is the cross term v₁×v₂ — the plane
    // the two axes span. q₁q₂ − q₂q₁ = 2·v₁∧v₂: the scalar·scalar term, both
    // scalar·vector terms, and −v₁·v₂ are symmetric under exchange, so
    // everything celebrated as "3D rotations don't commute" is the exchange
    // sign of one anti-symmetric product of two odd-graded objects — the
    // (−1)^(rs) parity table evaluated at r = s = 1. the standard physical
    // proof of deep non-commutativity is a book demonstrating a single table
    // entry
    //
    // and the sign is not an axiom about odd grades: the wedge adds π when
    // exchange reverses orientation — a position, not a bit
    // (it_keeps_the_anticommutativity_in_the_wedge).
    // (−1)^(rs) is the flattened 2θ readout of that half turn
    assert!(
        (x_then_y.3 + y_then_x.3).abs() < EPSILON,
        "the cross term flips: a ∧ b = −(b ∧ a)"
    );
    let flip = x_then_y.3 - y_then_x.3;
    assert!(
        (flip.abs() - 2.0 * s * s).abs() < EPSILON,
        "the whole order gap is twice the wedge coefficient"
    );

    // what both orders share: the composite angle. two quarter turns about
    // perpendicular axes compose to a 2π/3 rotation — the cube-diagonal fact —
    // and that angle is geonum multiplication of half-angle projections:
    // cos(c/2) = cos(a/2)cos(b/2) − sin(a/2)sin(b/2)cos(Θ), Θ between the axes.
    // geonum stores rotation position as t = tan(θ/2), so the half-angle
    // quaternions introduce as a device is the native coordinate
    let half = Angle::new(1.0, 4.0); // π/4, half the book's quarter turn
    let sym = Geonum::cos(half) * Geonum::cos(half); // magnitudes multiply, angles add

    // the correction term carries the projection of one axis on the other —
    // a quarter turn apart, the projection vanishes: geometry_test::its_a_line
    // at Δ = π/2, the line gone
    let axis_x = Geonum::new(1.0, 0.0, 1.0);
    let axis_y = Geonum::new(1.0, 1.0, 2.0);
    let skew = Geonum::sin(half) * Geonum::sin(half) * axis_x.dot(&axis_y);
    assert!(
        skew.near_mag(0.0),
        "perpendicular axes: the correction is a vanished projection"
    );

    // no hand-applied minus: sin·sin lands two blades up (π/2 + π/2 = π), and
    // adding at opposite angles subtracts — the formula's − sign is a position
    assert_eq!(
        skew.angle.grade(),
        2,
        "the − in cos(a/2)cos(b/2) − sin(a/2)sin(b/2) is the blade sum"
    );
    let composite_cos = sym + skew;
    assert!(composite_cos.near_mag(0.5));
    assert_eq!(composite_cos.angle.grade(), 0);

    // scaffolding corroborates: the quaternion scalar part is the same number
    assert!((x_then_y.0 - composite_cos.mag).abs() < EPSILON);

    // name the angle: cos(π/3) = 1/2, so the composite half-angle is π/3 and
    // the full composite is 2π/3 — both orders turn the book 120° about a cube
    // diagonal, (1,1,1) one way and (1,1,−1) the other
    let composite_half = Angle::new(1.0, 3.0);
    assert!(Geonum::cos(composite_half).near(&composite_cos));
    assert!((composite_half * 2.0).near(&Angle::new(2.0, 3.0)));

    // the gap between the two outcomes is itself a rotation: q_xy ⊗ q_yx⁻¹.
    // its scalar part is again 1/2 — the orders disagree by another 2π/3. the
    // composite, its reverse-order twin, and their gap share one angle: the
    // three-fold symmetry of the cube corner the book pivots around. the
    // order-dependence is this geometry, not a sign conjured by a table
    let gap = hamilton(x_then_y, conj(y_then_x));
    assert!(
        (gap.0 - composite_cos.mag).abs() < EPSILON,
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

    // scaffolding: one full turn about x, built from two half turns. the cross
    // term is x×x = 0 — the wedge is identically zero on one axis — so the −1
    // comes from the symmetric −v·v term. the book demo's order-dependence was
    // all wedge and no dot; this demo is all dot and no wedge. two demos, the
    // two factors the product fuses
    let half_turn = ((PI / 2.0).cos(), (PI / 2.0).sin(), 0.0, 0.0); // q(π) = (0, x)
    let full_turn_q = hamilton(half_turn, half_turn); // i² = −1
    assert!(
        (full_turn_q.0 + 1.0).abs() < EPSILON,
        "q(2π) = −1: the sign the belt trick makes physical"
    );
    for cross in [full_turn_q.1, full_turn_q.2, full_turn_q.3] {
        assert!(
            cross.abs() < EPSILON,
            "zero wedge: the −1 is not anti-symmetry"
        );
    }

    // with the wedge gone, nothing is order-sensitive: same-axis turns commute
    let quarter_turn = ((PI / 4.0).cos(), (PI / 4.0).sin(), 0.0, 0.0);
    let ab = hamilton(half_turn, quarter_turn);
    let ba = hamilton(quarter_turn, half_turn);
    for (l, r) in [(ab.0, ba.0), (ab.1, ba.1), (ab.2, ba.2), (ab.3, ba.3)] {
        assert!((l - r).abs() < EPSILON, "one axis: order-blind");
    }

    // geonum: same-axis composition is the commutative multiply
    // (it_composes_the_rotor_commutatively), so the demo's
    // whole content is winding — a full turn arrives four blades on with the
    // orientation unchanged
    let book = Geonum::new(1.0, 1.0, 6.0);
    let turn = Angle::new(2.0, 1.0); // 2π

    let once = book.rotate(turn);
    assert_eq!(
        once.base_angle(),
        book.base_angle(),
        "the book looks the same at 2π"
    );
    assert_eq!(
        once.angle.blade(),
        book.angle.blade() + 4,
        "the arm holds four more quarter turns"
    );

    // the quaternion is the rotation at half rate: halve the winding and read
    // the grade. 2π halves to π — grade 2, the −1 the scaffolding computed.
    // a second turn halves to 2π — grade 0, the +1: the arm untwists
    let spinor_once = (once.angle - book.angle) / 2.0;
    assert_eq!(
        spinor_once.grade(),
        2,
        "2π halves to π: −q, the twisted arm"
    );

    let twice = once.rotate(turn);
    let spinor_twice = (twice.angle - book.angle) / 2.0;
    assert_eq!(spinor_twice.grade(), 0, "4π halves to 2π: +q, untwisted");

    // corroborate the grade readout against the scaffolding's scalar part:
    // cos of the halved winding is [1, π] — magnitude 1 at the grade-2 landing,
    // the −1 as a position
    let readout = Geonum::cos(spinor_once);
    assert!(readout.near_mag(full_turn_q.0.abs()));
    assert_eq!(
        readout.angle.grade(),
        2,
        "the scaffolding's −1, read as a grade-2 landing"
    );

    // quaternions stop counting at the cover: q(4π) = q(0), the count gone.
    // the angle is the count — blade 8 is a different address than blade 0,
    // same grade. spinor_test carries this to the belt parity class
    assert_ne!(
        twice.angle, book.angle,
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
