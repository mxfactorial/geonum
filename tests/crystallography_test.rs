// the crystallographic restriction is an angle-lattice test
//
// only 1-, 2-, 3-, 4- and 6-fold rotations can be symmetries of a periodic
// lattice — the theorem that made shechtman's 1984 five-fold diffraction
// pattern a scandal. the standard proof rotates two neighboring lattice
// points toward each other and demands the result land back on the lattice:
// the produced vector has length |1 − 2cos θ| times the spacing, so the
// symmetry survives only when 2cos θ is an integer. that is an angle-lattice
// compatibility test, run here as geonum rotations:
//
//   - n = 1, 2, 3, 4, 6 pass — the only angles whose doubled cosine is whole
//   - every other n produces a lattice vector SHORTER than the minimal
//     spacing: the contradiction, constructed
//   - five-fold misses by golden-ratio amounts: 2cos 72° = 1/φ. penrose
//     tilings and quasicrystals are golden because that is the residue the
//     integer lattice refuses
//   - closure is not tiling: the pentagon's turning closes at 2π like any
//     polygon (curve_test) — the restriction is the extra gate translation
//     symmetry adds
//
// run: cargo test --test crystallography_test -- --show-output

use geonum::*;

// the restriction construction for an n-fold candidate: take neighboring
// lattice points a and b one spacing apart, rotate a about b by θ and b about
// a by −θ. the images' separation is parallel to ab; symmetry demands it be a
// whole number of spacings
fn separation(n: usize) -> Geonum {
    let theta = Angle::new(2.0, n as f64); // 2π/n
    let a = Geonum::scalar(0.0);
    let b = Geonum::new(1.0, 0.0, 1.0);

    let a_rotated = b + (a - b).rotate(theta); // a swung about b
    let b_rotated = a + (b - a).rotate(Angle::new(-2.0, n as f64)); // b swung about a

    a_rotated - b_rotated
}

#[test]
fn it_restricts_lattice_rotations_to_the_crystallographic_set() {
    let mut allowed = Vec::new();

    for n in 1..=12usize {
        let s = separation(n);

        // the separation is parallel to the lattice row — its quadrature
        // component cancels by the rotation symmetry itself
        assert!(
            s.opp().mag < 1e-9,
            "{n}-fold: the construction stays on the row"
        );

        // lattice compatibility: the length is a whole number of spacings
        if (s.mag - s.mag.round()).abs() < 1e-9 {
            allowed.push(n);
        }
    }

    assert_eq!(
        allowed,
        vec![1, 2, 3, 4, 6],
        "the crystallographic set — the only n with 2cos(2π/n) whole"
    );

    // the forbidden n produce a lattice vector strictly shorter than the
    // minimal spacing — the contradiction that kills the symmetry
    for n in [5usize, 7, 8, 9, 10, 12] {
        let s = separation(n);
        assert!(
            s.mag > 1e-6 && s.mag < 1.0 - 1e-6,
            "{n}-fold: a vector shorter than the shortest — impossible on a lattice"
        );
    }
}

#[test]
fn it_forbids_five_fold_by_the_golden_ratio() {
    // 2cos 72° = φ − 1 = 1/φ: the five-fold angle's doubled cosine is the
    // golden ratio's reciprocal — irrational, so the lattice test fails by
    // exactly the amount that structures every quasicrystal
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let (cos_72, _) = Angle::new(2.0, 5.0).cos_sin();

    assert!(
        (2.0 * cos_72 - (phi - 1.0)).abs() < 1e-12,
        "2cos 72° = 1/φ — the five-fold residue is golden"
    );

    // the construction's illegal vector has length 2 − φ = 1/φ² of a spacing
    assert!(
        (separation(5).mag - (2.0 - phi)).abs() < 1e-9,
        "the five-fold contradiction is 1/φ² spacings long"
    );
}

#[test]
fn it_closes_the_pentagon_but_cannot_tile_with_it() {
    // closure is cheap: five exterior turns of 2π/5 close the pentagon the
    // same way curve_test's hexagon closes — every polygon's turning totals
    // one full turn
    let mut turning = Angle::new(0.0, 1.0);
    for _ in 0..5 {
        turning = turning + Angle::new(2.0, 5.0);
    }
    assert!(
        turning.near(&Angle::new(2.0, 1.0)),
        "the pentagon closes — turning is no obstacle"
    );

    // tiling is the harder gate: closure needs the turning to total 2π,
    // periodicity needs 2cos θ whole. the hexagon passes both; the pentagon
    // passes one — which is why bathroom floors have hexagons and quasicrystals
    // have penrose rhombs
    let hexagon = separation(6);
    assert!(
        hexagon.near_mag(0.0),
        "6-fold: the rotated neighbors coincide — the lattice absorbs the turn"
    );
    let pentagon = separation(5);
    assert!(
        pentagon.mag > 0.38 && pentagon.mag < 0.39,
        "5-fold: the golden residue survives — no lattice absorbs it"
    );
}
