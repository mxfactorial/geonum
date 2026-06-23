// electromagnetism, as geometry — and as a demonstration of what two numbers do that the tensor
// machinery cannot. the cross product behind the Poynting vector exists ONLY in 3D and 7D; the
// field tensor F_μν grows as n²; ∇× returns a vector only in 3D. the wedge, the rotation, and the
// grade carry Maxwell's structure in any dimension, O(1).

use geonum::*;

#[test]
fn it_holds_the_whole_field_in_two_numbers() {
    // the electromagnetic field tensor F_μν is antisymmetric: n(n−1)/2 independent components — 6 in
    // spacetime's 4D, 1_999_000 in 2000D. geonum carries the field in two numbers, and its Poynting
    // flux comes out identical no matter how many dimensions the fields are embedded in
    let e = Geonum::new(3.0, 0.0, 1.0);
    let b = e.rotate(Angle::new(1.0, 2.0)).scale(2.0 / 3.0); // |B| = 2, ⊥ E
    let flux = e.wedge(&b);

    // the same fields embedded 2000 quarter-turns out — geonum's "2000 dimensions"
    let e_hi = Geonum::new_with_angle(3.0, Angle::new_with_blade(2000, 0.0, 1.0));
    let b_hi = e_hi.rotate(Angle::new(1.0, 2.0)).scale(2.0 / 3.0);
    assert!(
        e_hi.wedge(&b_hi).near_mag(flux.mag),
        "the flux is identical in 4D and 2000D — O(1), two numbers, no F_μν"
    );

    let f_components = |n: u64| n * (n - 1) / 2; // what the antisymmetric tensor must carry
    assert_eq!(f_components(4), 6);
    assert_eq!(
        f_components(2000),
        1_999_000,
        "the tensor explodes as n²; geonum holds at 2"
    );
}

#[test]
fn it_takes_the_poynting_flux_where_no_cross_product_exists() {
    // the Poynting vector S = E × B is built on the cross product — which exists ONLY in 3D and 7D.
    // the real operation is the wedge E ∧ B: it equals |E × B| in 3D and keeps computing in any
    // dimension, where × is undefined
    let e = Geonum::new(3.0, 0.0, 1.0);
    let b = Geonum::new(2.0, 1.0, 2.0); // ⊥ E
    assert!(e.wedge(&b).near_mag(6.0), "|E ∧ B| = |E × B| = 6 in 3D");

    let e_hi = Geonum::new_with_angle(3.0, Angle::new_with_blade(1000, 0.0, 1.0));
    let b_hi = Geonum::new_with_angle(2.0, Angle::new_with_blade(1000, 1.0, 2.0)); // ⊥ E, |B| = 2
    assert!(
        e_hi.wedge(&b_hi).near_mag(6.0),
        "and the same flux in 1000D, where E × B has no meaning"
    );
}

#[test]
fn it_curls_into_a_bivector_in_any_dimension() {
    // ∇×E in Faraday's law returns a vector only in 3D — in general the curl of a vector is a 2-form
    // (a bivector). geonum's curl is a π/2 rotation that raises the grade to that bivector, and it
    // runs in any dimension where the 3-vector ∇× cannot even be written
    let e = Geonum::new(2.0, 1.0, 2.0); // E as a grade-1 vector
    let curl = e.differentiate();
    assert_eq!(
        curl.angle.grade(),
        2,
        "curl of a vector is a bivector — a 2-form"
    );

    let e_hi = Geonum::new_with_angle(2.0, Angle::new_with_blade(1000, 1.0, 2.0));
    let curl_hi = e_hi.differentiate();
    assert!(
        curl_hi.near_mag(e.mag),
        "the curl computes in 1000D, where ∇× has no vector form"
    );
    assert_eq!(
        curl_hi.angle.grade(),
        2,
        "still a bivector — the operation is dimension-free"
    );
}
