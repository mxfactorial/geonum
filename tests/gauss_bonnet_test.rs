// total curvature is counted turns
//
// gauss-bonnet — ∫K dA + ∮k_g ds = 2πχ — is presented as a deep bridge between
// analysis and topology, proved with connections and pullbacks. the content is
// curve_test's closure argument gone global: a path closes when its turning
// completes full turns, and on a curved surface the interior takes a share of
// the turning. everything below is blade arithmetic:
//
//   - descartes 1630: the angle deficits of a convex polyhedron total 4π —
//     blade 8, exactly, for all five platonic solids. the euler characteristic
//     is the winding count: total = 2πχ
//   - a spherical triangle's area IS its angle excess — the sphere pays area
//     for every radian the angles overshoot π
//   - boundary turning plus enclosed curvature is one conserved 2π: the flat
//     hexagon pays it all in turning, the spherical octant splits it 3π/2
//     turning + π/2 curvature
//   - the scalar reading of 4π is 0 — grade_angle wraps the theorem away. the
//     result lives in the blade
//
// run: cargo test --test gauss_bonnet_test -- --show-output

use geonum::*;

#[test]
fn it_sums_polyhedron_deficits_to_four_pi() {
    // (vertices, edges, faces, faces meeting per vertex, face corner angle)
    let platonic = [
        (
            "tetrahedron",
            4usize,
            6usize,
            4usize,
            3.0,
            Angle::new(1.0, 3.0),
        ),
        ("cube", 8, 12, 6, 3.0, Angle::new(1.0, 2.0)),
        ("octahedron", 6, 12, 8, 4.0, Angle::new(1.0, 3.0)),
        ("dodecahedron", 20, 30, 12, 3.0, Angle::new(3.0, 5.0)),
        ("icosahedron", 12, 30, 20, 5.0, Angle::new(1.0, 3.0)),
    ];

    for (name, v, e, f, meeting, corner) in platonic {
        // one vertex's deficit: the full turn minus what the faces fill
        let deficit = Angle::new(2.0, 1.0) - corner * meeting;

        // total over the solid — angle addition accumulating blade
        let mut total = Angle::new(0.0, 1.0);
        for _ in 0..v {
            total = total + deficit;
        }

        // descartes: 4π, always — blade 8 with nothing left over
        assert!(
            total.near(&Angle::new(4.0, 1.0)),
            "{name}: deficits total 4π — blade {}, rem {:.2e}",
            total.blade(),
            total.rem()
        );

        // and 4π = 2πχ: the euler characteristic of the sphere, counted by
        // the winding. the scalar reading is blind — grade_angle wraps 4π to 0
        let chi = (v + f) as i64 - e as i64;
        assert_eq!(chi, 2, "{name}: V − E + F = 2");
        assert!(
            total.near(&Angle::new(2.0 * chi as f64, 1.0)),
            "{name}: total deficit = 2πχ"
        );
        assert!(
            total.grade_angle() < 1e-9,
            "{name}: the scalar reads 0 — the theorem lives in the blade"
        );
    }
}

#[test]
fn it_measures_spherical_area_as_angle_excess() {
    // the octant triangle — three right angles: excess = 3·π/2 − π = π/2,
    // which is exactly one eighth of the unit sphere's 4π
    let octant_excess =
        Angle::new(1.0, 2.0) + Angle::new(1.0, 2.0) + Angle::new(1.0, 2.0) - Angle::new(1.0, 1.0);
    assert!(
        octant_excess.near(&Angle::new(1.0, 2.0)),
        "excess π/2 — the octant's area, read off angle arithmetic"
    );

    // a birectangular triangle with apex α: two right angles pin it to a lune
    // of angle α, and its area is α itself — the excess again
    let apex = Angle::new(1.0, 3.0);
    let excess = Angle::new(1.0, 2.0) + Angle::new(1.0, 2.0) + apex - Angle::new(1.0, 1.0);
    assert!(
        excess.near(&apex),
        "the birectangular triangle's area IS its apex angle"
    );
}

#[test]
fn it_splits_one_full_turn_between_turning_and_curvature() {
    // flat closure: a hexagon pays the whole 2π in boundary turning — six
    // exterior angles of π/3, no curvature to share with
    let mut flat_turning = Angle::new(0.0, 1.0);
    for _ in 0..6 {
        flat_turning = flat_turning + Angle::new(1.0, 3.0);
    }
    assert!(
        flat_turning.near(&Angle::new(2.0, 1.0)),
        "the flat polygon closes on turning alone"
    );

    // spherical closure: the octant's boundary turns only 3·π/2 — the
    // enclosed curvature (area π/2, K = 1) supplies the rest. same 2π,
    // split between boundary and interior
    let exterior = Angle::new(1.0, 2.0); // π − π/2 at each corner
    let sphere_turning = exterior + exterior + exterior;
    let enclosed_curvature = Angle::new(1.0, 2.0); // ∫K dA over the octant
    assert!(
        (sphere_turning + enclosed_curvature).near(&Angle::new(2.0, 1.0)),
        "turning + curvature = 2π — the surface takes its share of the closure"
    );
}

#[test]
fn it_zeroes_the_torus_deficits() {
    // a flat square torus: four squares meet at every vertex, filling the full
    // turn exactly — deficit zero, everywhere
    let deficit = Angle::new(2.0, 1.0) - Angle::new(1.0, 2.0) * 4.0;
    assert!(
        deficit.near(&Angle::new(0.0, 1.0)),
        "four right angles fill the vertex — no deficit"
    );

    // and 0 = 2πχ: the torus's euler characteristic, counted on a 3×3 grid
    let (v, e, f) = (9i64, 18i64, 9i64);
    assert_eq!(v - e + f, 0, "χ(torus) = 0 — the donut pays no curvature");
}
