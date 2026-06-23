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
//! run: cargo test --test quaternion_test

use geonum::*;
use std::f64::consts::PI;

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
