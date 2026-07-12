// u(xᵢ) = Σⱼ K(xᵢ, yⱼ) ρⱼ is scalar summation — it adds magnitudes and throws away the
// influence direction from each yⱼ to xᵢ. the geometric version keeps the angle:
//
//   U(xᵢ) = Σⱼ [K(rᵢⱼ) · ρⱼ, θᵢⱼ]
//
// where θᵢⱼ = angle from source yⱼ to target xᵢ. accumulating geonum instead of f64
// lets the field carry both net strength AND net direction of influence.
//
// the scalar u(xᵢ) is then the grade-0 projection of U(xᵢ) — what you get when you
// collapse the direction back to a number. the geometric version is strictly richer

use geonum::*;

const EPSILON: f64 = 1e-10;

#[test]
fn it_replaces_kernel_weighted_sum_with_directed_accumulation() {
    // u(xᵢ) = Σⱼ K(rᵢⱼ) ρⱼ adds scalars — no direction, each term is K(r)·ρ
    // U(xᵢ) = Σⱼ source_j.spread(boundary_j) accumulates directed quantities
    //
    // the kernel K is not a function here — boundary = [r, θᵢⱼ] is the kernel
    // spread IS the kernel application: source.spread(boundary) = [ρ/r, φ + θᵢⱼ]
    // each radial kernel expresses as a spread over a boundary, and the whole family is one
    // op — only the boundary magnitude changes, r for 1/r and r² for 1/r²; the "formula"
    // K(r) is just which magnitude sits in the boundary

    // grade 0 source, density ρ=2, kernel K(r) = 1/r
    let rho = Geonum::new(2.0, 0.0, 1.0);

    // case 1: two sources at r=2, both from the east (θ=0)
    let east = Geonum::new(2.0, 0.0, 1.0); // boundary [r=2, θ=0]

    let geo_collinear = rho.spread(east) + rho.spread(east); // [1,0] + [1,0] = [2,0]
    let scalar_collinear = rho.mag / east.mag + rho.mag / east.mag; // 1 + 1 = 2

    // collinear: geometric magnitude agrees with scalar sum
    assert!(geo_collinear.near_mag(scalar_collinear));
    assert!(geo_collinear.angle.near_rad(0.0)); // net influence points east

    // case 2: one source east, one source west — same r=2, same ρ=2
    let west = Geonum::new(2.0, 1.0, 1.0); // boundary [r=2, θ=π]

    let geo_opposite = rho.spread(east) + rho.spread(west); // [1,0] + [1,π] → cancel
    let scalar_opposite = rho.mag / east.mag + rho.mag / west.mag; // still 1 + 1 = 2

    // scalar sum still 2 — no direction, no cancellation
    assert!((scalar_opposite - scalar_collinear).abs() < EPSILON);

    // geometric sum cancels — equal sources from opposite directions annihilate
    assert!(geo_opposite.near_mag(0.0));

    // scalar sum cannot distinguish collinear from opposite; geometric sum can
    assert!(geo_collinear.mag > geo_opposite.mag);
}

#[test]
fn it_proves_scalar_sum_is_grade_0_projection_of_geometric_field() {
    // u = Σⱼ K·ρⱼ sums contribution magnitudes — it adds legs, not vectors
    // U = Σⱼ source_j.spread(boundary_j) accumulates with direction — it computes the hypotenuse
    //
    // geometry_test::it_overshoots_the_length_when_it_sums_the_legs: leg₁ + leg₂ > √(leg₁²+leg₂²)
    // the scalar convolution has been computing that overshooting sum all along

    // source 1: density 3, kernel K(r)=1/r at r=1, from the east
    let rho1 = Geonum::new(3.0, 0.0, 1.0);
    let b1 = Geonum::new(1.0, 0.0, 1.0); // boundary [r=1, θ=0]

    // source 2: density 4, kernel K(r)=1/r at r=1, from the north
    let rho2 = Geonum::new(4.0, 0.0, 1.0);
    let b2 = Geonum::new(1.0, 1.0, 2.0); // boundary [r=1, θ=π/2]

    let field = rho1.spread(b1) + rho2.spread(b2); // [3,east] + [4,north]
    let scalar_sum = rho1.mag / b1.mag + rho2.mag / b2.mag; // 3 + 4 = 7

    // geometric field is the 3-4-5 hypotenuse
    assert!(field.near_mag(5.0));

    // scalar sum is the legs added: 3 + 4 = 7 > 5
    assert!((scalar_sum - 7.0).abs() < EPSILON);
    assert!(scalar_sum > field.mag);

    // project_to_dimension recovers each source leg from the geometric field
    let x_component = field.project_to_dimension(0); // east leg: 3
    let y_component = field.project_to_dimension(1); // north leg: 4

    assert!((x_component - rho1.mag).abs() < EPSILON);
    assert!((y_component - rho2.mag).abs() < EPSILON);

    // the legs reconstruct the hypotenuse via √(3²+4²) — not by summing them
    let reconstructed = (x_component.powi(2) + y_component.powi(2)).sqrt();
    assert!((reconstructed - field.mag).abs() < EPSILON);
}

#[test]
fn it_accumulates_sources_with_influence_direction_intact() {
    // two equal influences a π turn apart hand off to length 0 — but a length-0 geonum is
    // not the scalar's bare 0. the opposite-angle sum's blade is the combined winding of
    // the two rays (geonum_mod.rs Add), so two opposed-pair cancellations land on two
    // distinct length-0 positions. the scalar sum collapses every cancellation to the one
    // bare number 0; the geometric sum lands each at its own position in angle space

    let rho = Geonum::new(2.0, 0.0, 1.0); // density ρ=2, kernel K(r)=1/r at r=2

    // east + west: equal sources on the horizontal axis, influence from θ=0 and θ=π
    let east = Geonum::new(2.0, 0.0, 1.0); // boundary [r=2, θ=0]
    let west = Geonum::new(2.0, 1.0, 1.0); // boundary [r=2, θ=π]
    let ew = rho.spread(east) + rho.spread(west);

    // north + south: equal sources on the vertical axis, influence from π/2 and 3π/2
    let north = Geonum::new(2.0, 1.0, 2.0); // [r=2, θ=π/2]
    let south = Geonum::new(2.0, 3.0, 2.0); // [r=2, θ=3π/2]
    let ns = rho.spread(north) + rho.spread(south);

    // both vanish in magnitude — the scalar sum's only outcome, and it reports the
    // same 0 for both: K·ρ + K·ρ from opposite directions is 0 either way
    assert!(ew.near_mag(0.0));
    assert!(ns.near_mag(0.0));

    // but the two length-0 geonums are distinct: the horizontal pair's combined winding
    // is 0 + 2 = 2, the vertical pair's is 1 + 3 = 4. the blade is winding — the
    // quarter-turns the two rays accumulated (algebra_test) — not an axis the zero stores.
    // two different cancellations, two different positions
    assert_eq!(ew.angle.blade(), 2);
    assert_eq!(ns.angle.blade(), 4);

    // the scalar collapses both to the one number 0; these land at two positions. winding
    // 2 sits at bivector grade; winding 4 folds home to scalar grade — a full turn is four
    // blades, winding home (algebra_test)
    assert_eq!(ew.angle.grade(), 2);
    assert_eq!(ns.angle.grade(), 0);
}

#[test]
fn it_computes_convolution_integral_as_angle_quadrature() {
    // the continuous convolution u(x) = ∫_B K(x,y) ρ(y) dy is a sum over a sweep of
    // source directions. geonum integrates it as ANGLE QUADRATURE: each source angle
    // contributes a directed [K(r)·ρ, θ], and the accumulation is exact — not a riemann
    // limit (integral_test earns that refusal). a uniform ring of sources around the
    // target makes the quadrature close on itself: the directed contributions sweep a
    // full turn and cancel, so the field at the center is exactly zero. that is Newton's
    // shell theorem — and it holds at EVERY sampling resolution, not in the limit. the
    // scalar magnitude-sum, blind to direction, can never vanish; it only grows

    let rho = 2.0; // uniform source density ρ
    let r = 2.0; // ring radius; kernel K(r) = 1/r, so the boundary magnitude is r
    let source = Geonum::new(rho, 0.0, 1.0); // a source of density ρ at grade 0

    // integrate the ring at several resolutions — exact at each one, never converging
    for n in [2usize, 3, 6, 12] {
        let mut field = Geonum::new(0.0, 0.0, 1.0); // directed accumulator
        let mut scalar_sum = 0.0; // the scalar convolution, magnitudes only

        for j in 0..n {
            // jth source sits a fraction j/n of a full turn around the ring; its kernel
            // is the boundary [r, θ_j] and spread applies it: [ρ/r, θ_j]
            let theta = Angle::new(2.0 * j as f64 / n as f64, 1.0); // 2πj/n
            let boundary = Geonum::new_with_angle(r, theta);
            field = field + source.spread(boundary); // directed quadrature term
            scalar_sum += source.mag / boundary.mag; // |K·ρ|, direction discarded
        }

        // the geometric quadrature vanishes — the ring's directed influence cancels and
        // the field at the center is zero, the shell theorem, exact at this n
        assert!(
            field.near_mag(0.0),
            "n={n}: directed ring quadrature cancels to zero field at the center"
        );

        // the scalar convolution sums magnitudes only: it grows as n·ρ/r, reporting
        // "more sources, more signal" — it cannot represent the cancellation the field saw
        assert!((scalar_sum - n as f64 * rho / r).abs() < EPSILON);
    }
}

#[test]
fn it_factors_kernel_into_magnitude_and_angle_components() {
    // K(xᵢ, yⱼ) is conventionally a scalar function of two points, but it only ever
    // reaches the displacement Δ = xᵢ − yⱼ through its length |Δ|. that collapse is the
    // projection: K factors into a magnitude K(|Δ|) and an angle dir(Δ), and the scalar
    // form keeps only the magnitude. the geometric kernel keeps both, so it is the
    // faithful lift the scalar kernel is a shadow of
    //
    // K_geo(xᵢ, yⱼ) = [K(|xᵢ - yⱼ|), Angle::new_from_cartesian(xᵢ - yⱼ)]

    // radial kernel K(r) = 1/r — a function of the distance alone
    let k = |r: f64| 1.0 / r;

    // three displacements Δ = xᵢ − yⱼ that share length 5 but point three different ways
    let a = Geonum::new_from_cartesian(3.0, 4.0); // [5, atan2(4,3)]
    let b = Geonum::new_from_cartesian(4.0, 3.0); // [5, atan2(3,4)]
    let c = Geonum::new_from_cartesian(-3.0, -4.0); // [5, the opposite bearing]

    // same length → the scalar kernel hands back one number for all three, blind to
    // direction. K(|Δ|) is a many-to-one map: it cannot tell these displacements apart
    assert!(a.near_mag(5.0) && b.near_mag(5.0) && c.near_mag(5.0));
    let ks = k(a.mag); // 1/5 = 0.2
    assert!((ks - 0.2).abs() < EPSILON);
    assert!((k(b.mag) - ks).abs() < EPSILON);
    assert!((k(c.mag) - ks).abs() < EPSILON);

    // the geometric kernel factors K into [K(|Δ|), dir(Δ)] — applying K to the
    // magnitude while carrying the displacement's angle through untouched
    let ka = Geonum::new_with_angle(k(a.mag), a.angle);
    let kb = Geonum::new_with_angle(k(b.mag), b.angle);
    let kc = Geonum::new_with_angle(k(c.mag), c.angle);

    // the magnitude component IS the scalar kernel — nothing added there, the scalar
    // form is exactly this projection onto grade-0 strength
    assert!(ka.near_mag(ks) && kb.near_mag(ks) && kc.near_mag(ks));

    // but the angle component separates what the scalar collapsed: three equal
    // magnitudes, three distinct bearings — each the direction the scalar kernel dropped.
    // the geometric kernel is one-to-one where the scalar kernel was many-to-one
    assert!((ka.angle.grade_angle() - 4.0_f64.atan2(3.0)).abs() < EPSILON); // (3,4) → atan2(4,3)
    assert!((kb.angle.grade_angle() - 3.0_f64.atan2(4.0)).abs() < EPSILON); // (4,3) → atan2(3,4)

    // (3,4) vs (−3,−4): identical scalar kernel, exactly opposite geometric kernel —
    // the half-turn the scalar form folds onto the same number
    assert!(ka.angle.is_opposite(&kc.angle));
}

#[test]
fn it_is_gauss_law_in_any_dimension_as_one_spread() {
    // radially symmetric kernels — the ones constant on spheres — are the tractable class,
    // and a sphere is why: it is a trivial object, a magnitude with the angle free, the
    // orbit of one radius. its whole d-dimensional surface is the single number r^(d-1)
    // (the unit-sphere constant folded into the source), so the inverse-power field is ONE
    // spread and Gauss's law is that number cancelling itself. the dimension is an exponent,
    // never a grid: the conventional method integrates over the d-ball; geonum divides by a
    // number and multiplies it back.

    let flux = 7.0; // source strength Φ, a grade-0 scalar
    let source = Geonum::new(flux, 0.0, 1.0);
    let r = 1.5_f64; // sphere radius

    // sweep the dimension d — each field is one spread, the dimension only the exponent
    let mut prev = f64::INFINITY;
    for d in [2usize, 3, 4, 5, 7, 10, 15, 25, 40] {
        let measure = r.powi((d - 1) as i32); // the sphere's whole surface, one number
        let surface = Geonum::new(measure, 0.0, 1.0);

        // the d-dimensional inverse-power field, exact: Φ / r^(d-1) — the Laplacian
        // Green's-function falloff in every dimension, no √(Σ d terms), no integral
        let field = source.spread(surface);
        assert!(field.near_mag(flux / measure));

        // higher dimension falls faster at r > 1 — strictly, no plateau
        assert!(
            field.mag < prev,
            "d={d}: the field falls faster as the dimension climbs"
        );
        prev = field.mag;

        // Gauss's law: gather the field back through the same sphere and the source returns,
        // because the surface's r^(d-1) annihilates the field's 1/r^(d-1) — in any dimension
        assert!(
            field.spread(surface.inv()).near_mag(flux),
            "d={d}: flux conserved through the sphere, one op not an integral"
        );
    }
}

#[test]
fn it_keeps_the_sphere_a_single_number_a_million_dimensions_out() {
    // push the dimension to a million. a coordinate kernel needs a 10^6-term √(Σ xᵢ²) for
    // the distance and 10^6 components for the field. the sphere is still one number; the
    // dimension lives in the blade (project_to_dimension is blade mod 4), read and projected
    // in O(1) at the millionth axis exactly as at the lowest

    let source = Geonum::new(5.0, 0.0, 1.0);

    // a unit sphere whose surface direction is swept a million quarter-turns out
    let surface = Geonum::new_with_angle(1.0, Angle::new_with_blade(1_000_000, 1.0, 4.0));
    let field = source.spread(surface); // one op, a million dimensions deep

    // a unit sphere dilutes nothing — the flux is unchanged, in any dimension
    assert!(field.near_mag(5.0));

    // and the field reads back at the millionth axis in O(1), the same as the lowest:
    // 1_000_000 ≡ 0 (mod 4), so the millionth-dimension projection IS the 0th
    let deep = field.project_to_dimension(1_000_000);
    let shallow = field.project_to_dimension(0);
    assert!(
        (deep - shallow).abs() < EPSILON,
        "the millionth dimension costs exactly what the lowest costs"
    );
}

#[test]
fn it_steps_an_n_body_gravitational_acceleration() {
    // a live application: one acceleration evaluation in an N-body gravitational step — the
    // Newtonian / Poisson inverse-square field. each source pulls the body toward it with
    // G·m/r²; spreading G·m over [r², direction] gives
    // that pull as a directed geonum, and they accumulate into the force vector the
    // integrator needs — one [magnitude, angle] per body, the distance read as the
    // magnitude, in any dimension (a cartesian step carries d components and a d-term √Σ)

    let g = 1.0; // gravitational constant, natural units
    let m = 3.0; // each source mass
    let r = 2.0; // range to each source
    let strength = Geonum::new(g * m, 0.0, 1.0); // the source charge G·m

    // the pull toward a source in a given direction: G·m spread over the 1/r² boundary
    let pull = |dir: Angle| strength.spread(Geonum::new_with_angle(r * r, dir));

    // a body flanked by equal masses due north and due south, with one due east
    let accel = pull(Angle::new(1.0, 2.0)) // north
        + pull(Angle::new(3.0, 2.0)) // south
        + pull(Angle::new(0.0, 1.0)); // east

    // north and south annihilate; the body accelerates toward the east mass alone, at
    // exactly G·m/r² due east — the closed form, not a value the test built
    assert!(accel.near_mag(g * m / (r * r))); // 0.75
    assert!(accel.angle.near_rad(0.0)); // due east

    // the same three-body step a million dimensions out — the acceleration is the same one
    // op, the dimension only a blade the accumulation never reads, no d-term distance
    let out = Angle::new_with_blade(1_000_000, 0.0, 1.0);
    let accel_hi = pull(Angle::new(1.0, 2.0) + out)
        + pull(Angle::new(3.0, 2.0) + out)
        + pull(Angle::new(0.0, 1.0) + out);
    assert!(accel_hi.near_mag(g * m / (r * r))); // 0.75, a million dimensions deep
}
