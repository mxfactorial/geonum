// the minkowski minus sign is a π rotation, not a scalar
//
// algebra_test.rs shows the geonum lattice is z⁴ = 1: grades 0, 1, 2, 3 ARE the
// four fourth-roots of unity, the Q lattice. so −1 is [1, π], grade 2, the dual
// — a π rotation, a geometric position. the metric signature s² = −t²+x²+y²+z²
// rests entirely on that one minus, and time is space rotated a quarter turn
// (the Wick rotation t → it), so the minus is the quarter turn squared:
// (i·t)² = [t², π] = −t².
//
// the tensor formalism buries this. a metric tensor g_μν is an n×n grid of
// scalar inner products, and "choosing a signature" gets dressed up as a deep
// decision about the nature of spacetime. it is none of that — it is which angle
// each basis squares to: 0 squares to +, π/2 squares to −. the matrix
// bookkeeping (components, raised and lowered indices, scalar combinations)
// obscures that one fact. the signature proof below is relocated here out of the
// tensor suite, where it sat under the scalar-combination machinery.
//
// the causal structure of spacetime (https://en.wikipedia.org/wiki/Causal_structure)
// — whether two events are timelike, lightlike, or spacelike separated — then
// reads directly off the grade of the assembled interval:
//   timelike  s² < 0  → grade 2  (causally connected, a sub-light worldline)
//   lightlike s² = 0  → null     (the light cone, a light ray)
//   spacelike s² > 0  → grade 0  (causally disconnected, no signal connects them)
// no metric tensor, no index gymnastics — the trichotomy is one geonum's grade.
//
// run: cargo test --test spacetime_test -- --show-output

use geonum::*;
use std::f64::consts::FRAC_PI_2;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_metric_signature() {
    // relocated from tensor_test.rs. the tensor framing — g_μν matrices, a
    // "choice" of signature — is bookkeeping over the one geometric fact this
    // proves: a basis squares to + or − by the angle it sits at, no choice

    // traditional physics: "we must carefully choose our metric tensor signature"
    // euclidean: (+,+,+,+) with g_μν = diag(1,1,1,1)
    // minkowski: (-,+,+,+) with g_μν = diag(-1,1,1,1)
    // this seems like a deep choice about the nature of spacetime

    // geonum: metric signature is just "what happens when angles add during squaring"
    // no choice needed - it mechanically emerges from angle arithmetic

    // test 1: euclidean signature emerges from 0° basis vectors
    // traditional: "we choose positive signature (+,+,+)"
    // geonum: basis vectors at 0° naturally square to positive

    let e1_euclidean = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° basis
    let e1_squared = e1_euclidean * e1_euclidean;

    // 0 + 0 = 0, cos(0) = +1
    assert_eq!(e1_squared.angle.blade(), 0);
    assert!(e1_squared.angle.grade_angle().cos() > 0.0); // positive signature
    assert_eq!(e1_squared.mag, 1.0);

    // test 2: minkowski signature emerges from timelike at π/2
    // traditional: "time has negative signature in the metric"
    // geonum: time at π/2 naturally squares to negative

    let time_basis = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 (perpendicular to space)
    let time_squared = time_basis * time_basis;

    // π/2 + π/2 = π, cos(π) = -1
    assert_eq!(time_squared.angle.blade(), 2); // blade 1 + 1 = 2 (which is π)
    assert!(time_squared.angle.grade_angle().cos() < 0.0); // negative signature!

    // test 3: the "choice" of signature is just choosing initial angles
    // traditional: "lets use signature (+,-,-,+)"
    // geonum: "lets point basis vectors at 0, π/2, π/2, 0"

    let custom_e0 = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° → squares to +
    let custom_e1 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 → squares to -
    let custom_e2 = Geonum::new_with_blade(1.0, 1, 0.0, 1.0); // π/2 → squares to -
    let custom_e3 = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // 0° → squares to +

    // verify the signature (+,-,-,+)
    assert!((custom_e0 * custom_e0).angle.grade_angle().cos() > 0.0); // +
    assert!((custom_e1 * custom_e1).angle.grade_angle().cos() < 0.0); // -
    assert!((custom_e2 * custom_e2).angle.grade_angle().cos() < 0.0); // -
    assert!((custom_e3 * custom_e3).angle.grade_angle().cos() > 0.0); // +

    // test 4: "negative" vectors squaring to positive
    // traditional: "in clifford algebras, some negative elements square to positive"
    // geonum: π + π = 2π ≡ 0, so negative times negative = positive

    let negative_vector = Geonum::new(1.0, 2.0, 2.0); // [1, π] = -1
    let squared = negative_vector * negative_vector;

    // π + π = 2π, and 2π ≡ 0 (mod 2π)
    assert!(squared.angle.grade_angle().abs() < 1e-10); // back to 0
    assert!(squared.angle.grade_angle().cos() > 0.0); // positive result
    assert_eq!(squared.mag, 1.0);

    // this is why (-1) × (-1) = +1: its just π + π = 2π ≡ 0

    // test 5: the metric tensor is just tracking angle relationships
    // traditional: "the metric tensor g_μν encodes the geometry of spacetime"
    // geonum: the "metric" is just how basis angles relate to each other

    let spatial = Geonum::new_with_blade(2.0, 0, 0.3, 1.0); // spatial vector at blade 0
    let temporal = Geonum::new_with_blade(2.0, 1, 0.3, 1.0); // temporal vector at blade 1

    // square both vectors through multiplication to reveal signature
    let spatial_squared = spatial * spatial; // blade arithmetic with boundary crossing
    let temporal_squared = temporal * temporal; // blade arithmetic with boundary crossing

    // prove exact blade accumulation shows signature
    assert_eq!(spatial_squared.angle.blade(), 1); // spatial squares to blade 1
    assert_eq!(temporal_squared.angle.blade(), 3); // temporal squares to blade 3
    let blade_diff = temporal_squared.angle.blade() - spatial_squared.angle.blade();
    assert_eq!(blade_diff, 2); // 3 - 1 = 2, encodes dual positive/negative spacetime signature (π angle as -,+)

    // prove signature through cosine values - measured from actual blade arithmetic
    assert!(spatial_squared.angle.grade_angle().cos() < 0.0); // spatial blade 1 gives negative cosine
    assert!(temporal_squared.angle.grade_angle().cos() > 0.0); // temporal blade 3 gives positive cosine

    // minkowski metric signature emerges: 2 blade difference maintains space/time distinction

    // test 6: signature "flips" are just π rotations
    // traditional: "changing signature requires careful metric tensor manipulation"
    // geonum: just rotate your basis by π

    let positive_signature = Geonum::new_with_blade(1.0, 0, 0.0, 1.0); // cos(0) = +1
    let flipped_signature = Geonum::new_with_blade(1.0, 2, 0.0, 1.0); // cos(π) = -1

    // same basis vector, just rotated by π
    assert_eq!(positive_signature.mag, flipped_signature.mag);
    assert_eq!(
        (positive_signature.angle.blade() + 2) % 4,
        flipped_signature.angle.blade() % 4
    );

    // test 7: complex metric signatures are just angle patterns
    // traditional: "some exotic spacetimes have signature (--++--++)"
    // geonum: "some bases have angles at π/2, π/2, 0, 0, π/2, π/2, 0, 0"

    let exotic_signature: Vec<Geonum> = vec![
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 1, 0.0, 1.0), // π/2 → -
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
        Geonum::new_with_blade(1.0, 0, 0.0, 1.0), // 0 → +
    ];

    // prove the exotic signature pattern
    for (i, basis) in exotic_signature.iter().enumerate() {
        let squared = *basis * *basis;
        let expected_negative = i % 4 < 2; // first two of each group are negative

        if expected_negative {
            assert!(
                squared.angle.grade_angle().cos() < 0.0,
                "index {} negative",
                i
            );
        } else {
            assert!(
                squared.angle.grade_angle().cos() > 0.0,
                "index {} positive",
                i
            );
        }
    }

    // test 8: the pseudoscalar signature property I² = ±1
    // traditional: "the pseudoscalar squares to ±1 depending on metric signature"
    // geonum: different dimension counts create different angle sums

    // in 3D euclidean: 3 spatial dimensions at 0°
    let i_3d_euclidean = Geonum::new_with_blade(1.0, 3, 0.0, 1.0); // 3 × π/2
    let i_squared_euclidean = i_3d_euclidean * i_3d_euclidean;

    // 3π/2 + 3π/2 = 3π ≡ π (mod 2π), cos(π) = -1
    assert_eq!(i_squared_euclidean.angle.grade_angle().cos(), -1.0); // I² = -1 for euclidean

    // in 4D minkowski: 1 time (π/2) + 3 space (0°)
    let i_4d_minkowski = Geonum::new_with_blade(1.0, 4, 0.0, 1.0); // 4 × π/2 = 2π
    let i_squared_minkowski = i_4d_minkowski * i_4d_minkowski;

    // 2π + 2π = 4π ≡ 0 (mod 2π), cos(0) = +1
    assert_eq!(i_squared_minkowski.angle.grade_angle().cos(), 1.0); // I² = +1 for minkowski

    // the ±1 "mystery" is just whether your total angle is odd or even multiples of π

    // conclusion: metric signatures arent choices or conventions
    // theyre mechanical consequences of angle arithmetic:
    // - angles add when multiplying
    // - 2π wraps to 0
    // - cos(0) = +1, cos(π) = -1
    // the entire formalism of metric tensors is just bookkeeping for "what angle is this?"
}

#[test]
fn it_replaces_the_squared_zero_with_rotation_and_cancellation() {
    // the conventional eye carries two "squares to zero" devices — the dual unit
    // ε² = 0 and the null vector v·v = 0, each its own algebra. geonum replaces
    // both with angle arithmetic: products rotate, and the single zero is additive
    //
    // a square rotates: [r, θ]² = [r², 2θ] doubles the angle and SQUARES the
    // magnitude. so a product can never vanish for r ≠ 0 — squaring is a rotation
    // with a growing magnitude, not an annihilation. this replaces the dual-number
    // ε² = 0: squaring [1, π] rotates it to [1, 2π] = +1, there is no nilpotent
    let sq = Geonum::new(1.0, 2.0, 2.0).pow(2.0); // [1, π]² = [1, 2π] = +1
    assert!(
        sq.near_mag(1.0),
        "squaring rotates — the magnitude survives, never zero"
    );
    assert_eq!(sq.angle.grade(), 0, "[1,π]² lands back at +1, not at 0");

    // the single zero is additive: a quantity against its own dual, [r,θ] + [r,θ+π],
    // two equal magnitudes a π rotation apart. this replaces the null vector — no
    // indefinite metric, the lightcone null is just a sum of opposites cancelling
    let cancel = Geonum::new(1.0, 0.0, 1.0) + Geonum::new(1.0, 1.0, 1.0); // [1,0] + [1,π]
    assert!(
        cancel.mag < EPSILON,
        "cancellation is additive — a sum of opposites"
    );

    // so the lightcone null comes from the additive branch (summing the grade-0
    // space square against the grade-2 time square), never from squaring. two
    // conventional squared-zero algebras replaced by one rotation and one sum
}

#[test]
fn it_reads_the_metric_as_the_dual_inverting_the_half_tangent() {
    // the metric is not a grid of squared inner products — it is the dual: a π
    // rotation, blade + 2 (src/angle.rs::dual). on the half-tangent S = tan(θ/2)
    // the dual is the inversion S → −1/S, since dualizing sends θ → θ+π and
    // tan((θ+π)/2) = −cot(θ/2) = −1/S. the same involution carries the metric
    // signature (it_replaces_the_squared_zero_with_rotation_and_cancellation), and
    // its fixed point S = ±i is the isotropic vector e₁ ± i·e₂ — the light cone

    let half_tangent = |a: Angle| (a.grade_angle() / 2.0).tan();

    for (p, q) in [(1.0, 3.0), (1.0, 4.0), (2.0, 5.0), (1.0, 6.0)] {
        let a = Angle::new(p, q);
        let s = half_tangent(a);

        // the dual inverts the half-tangent: S → −1/S
        assert!(
            (half_tangent(a.dual()) - (-1.0 / s)).abs() < 1e-9,
            "the dual inverts the half-tangent: tan((θ+π)/2) = −1/tan(θ/2)"
        );

        // and it always moves the grade by two, so no real direction is its own
        // dual — the inversion has no fixed point on the real lattice
        assert_ne!(
            a.dual().grade(),
            a.grade(),
            "no real direction is self-dual"
        );

        // the rational metric reads cos θ off that same half-tangent:
        // cos θ = (1−S²)/(1+S²). its denominator 1 + S² is the very polynomial the
        // dual's fixed point solves — S = −1/S ⟺ S² + 1 = 0 ⟺ S = ±i. so the
        // isotropic vector is the dual's fixed point AND the metric's pole: one
        // imaginary balance, which 1 + S² ≥ 1 holds off the real line
        let cos_from_s = (1.0 - s * s) / (1.0 + s * s);
        assert!(
            (cos_from_s - a.grade_angle().cos()).abs() < 1e-9,
            "cos θ = (1−S²)/(1+S²) — the metric read from the half-tangent"
        );
    }

    // so the light cone never appears as a real self-dual ray, only as the real
    // shadow of S = ±i: a direction and its dual, a π apart, cancelling additively
    // (the null of it_replaces_the_squared_zero...). the metric is the involution;
    // the isotropic vector is where it would hold still
}

#[test]
fn it_spans_the_three_conics_with_one_half_tangent() {
    // the three generalized complex units are the three conics, and one rational
    // half-tangent t parametrizes all of them through a single curvature κ:
    //   cos_κ(t) = (1 − κt²)/(1 + κt²)      sin_κ(t) = 2t/(1 + κt²)
    // the unit conic is cos_κ² + κ·sin_κ² = 1, and the generalized unit squares
    // to −κ:
    //   κ = +1  elliptic    i² = −1   the circle      cos²+sin²=1
    //   κ =  0  parabolic    ε² =  0   the light cone  the dual number, s²=0
    //   κ = −1  hyperbolic   j² = +1   the boost       cosh²−sinh²=1
    // geonum carries one (blade, t); t says which conic. the same κ is the unit's
    // square, the conic's curvature, and the denominator sign — and the light cone
    // is the κ=0 seam between an imaginary null (t=±i) and a real one (t=±1)

    let cos_k = |kappa: f64, t: f64| (1.0 - kappa * t * t) / (1.0 + kappa * t * t);
    let sin_k = |kappa: f64, t: f64| 2.0 * t / (1.0 + kappa * t * t);

    // the unit-conic identity holds for every curvature, one rational form
    for t in [0.2_f64, 0.5, 0.9] {
        for kappa in [1.0_f64, 0.0, -1.0] {
            let (c, s) = (cos_k(kappa, t), sin_k(kappa, t));
            assert!(
                (c * c + kappa * s * s - 1.0).abs() < 1e-12,
                "cos_κ² + κ·sin_κ² = 1 at κ = {kappa}"
            );
        }
    }

    // κ = +1 is not a foil — it IS geonum's circle: the rational cos/sin geonum
    // recovers from the stored half-tangent (src/angle.rs) is exactly cos_{+1}/sin_{+1}
    for (p, q) in [(1.0, 3.0), (1.0, 4.0), (2.0, 5.0)] {
        let a = Angle::new(p, q);
        let t = (a.grade_angle() / 2.0).tan();
        let (cos, sin) = a.cos_sin();
        assert!((cos_k(1.0, t) - cos).abs() < 1e-9, "κ=+1 is geonum's cos");
        assert!((sin_k(1.0, t) - sin).abs() < 1e-9, "κ=+1 is geonum's sin");
    }

    // κ = −1 is the boost: one sign flip from the circle (1+t² → 1−t²) yields the
    // rational hyperbola, and the rapidity reads back as s = tanh(φ/2)
    let s = 0.5_f64;
    let (cosh, sinh) = (cos_k(-1.0, s), sin_k(-1.0, s));
    assert!(
        (cosh * cosh - sinh * sinh - 1.0).abs() < 1e-12,
        "cosh² − sinh² = 1"
    );
    let phi = (sinh / cosh).atanh();
    assert!((s - (phi / 2.0).tanh()).abs() < 1e-12, "s = tanh(φ/2)");

    // κ = 0 is the parabolic seam — the light cone, the dual number ε²=0. cos_0=1,
    // sin_0=2t: the norm carries no contribution from sin, so length is blind to
    // the t direction — exactly s²=0, the nilpotent
    for t in [0.2_f64, 0.5, 0.9] {
        assert!((cos_k(0.0, t) - 1.0).abs() < 1e-12, "cos_0 = 1");
        assert!((sin_k(0.0, t) - 2.0 * t).abs() < 1e-12, "sin_0 = 2t");
    }

    // the null is the pole of cos_κ where 1 + κt² = 0: real for the hyperbola
    // (s → 1, the asymptote, the light cone head-on) so cos_{−1} blows up, but
    // imaginary for the circle (t = ±i) so cos_{+1} stays bounded. the κ=0 dual
    // number is the seam between them. geonum never squares to zero
    // (it_replaces_the_squared_zero) — the light cone is this boundary, reached
    // additively, the fixed-point shadow of the dual (it_reads_the_metric...)
    assert!(
        cos_k(-1.0, 0.99).abs() > 50.0,
        "hyperbolic cos blows up at the real null s → 1 — the light cone asymptote"
    );
    assert!(
        cos_k(1.0, 0.99).abs() <= 1.0,
        "elliptic cos stays bounded — its null t = ±i is off the real line"
    );
}

// the spacetime interval s² = (space)² + (time)² as a geonum vector sum. space
// sits on the real axis (grade 0), time on the i-axis (grade 1), so the time
// square lands at grade 2 (−t²) and subtracts. the causal character is the grade
// of the result — no metric tensor needed to assemble it
fn interval(space: f64, time: f64) -> Geonum {
    let space_sq = Geonum::new(space, 0.0, 1.0).pow(2.0); // [x², 0]
    let time_sq = Geonum::new(time, 1.0, 2.0).pow(2.0); // [t², π]
    space_sq + time_sq
}

#[test]
fn its_timelike() {
    // timelike separation: more time than space, |Δt| > |Δx|. the grade-2 time
    // square outweighs the grade-0 space square, so the interval lands grade 2
    // (s² < 0). the events are causally connected — a massive worldline slower
    // than light passes through both, and every frame agrees on their order
    let s = interval(3.0, 5.0); // Δx = 3, Δt = 5
    assert_eq!(s.angle.grade(), 2, "timelike interval is grade 2 (s² < 0)");
    assert!(s.near_mag(25.0 - 9.0), "|s²| = |9 − 25| = 16");
    // Δx/Δt = 3/5 < 1: a sub-light worldline connects timelike events

    // a 3+1 timelike interval: x²+y²+z² − t² with t dominating (1+4+4 − 16 = −7)
    let s4 = Geonum::new(1.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(2.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(2.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(4.0, 1.0, 2.0).pow(2.0);
    assert_eq!(s4.angle.grade(), 2, "3+1 timelike at grade 2");
    assert!(s4.near_mag(7.0), "|s²| = |9 − 16| = 7");
}

#[test]
fn its_spacelike() {
    // spacelike separation: more space than time, |Δx| > |Δt|. the grade-0 space
    // square outweighs the grade-2 time square, so the interval stays grade 0
    // (s² > 0). the events are causally disconnected — no signal at or below light
    // speed connects them, and their time order is frame-dependent
    let s = interval(5.0, 3.0); // Δx = 5, Δt = 3
    assert_eq!(s.angle.grade(), 0, "spacelike interval is grade 0 (s² > 0)");
    assert!(s.near_mag(25.0 - 9.0), "|s²| = 25 − 9 = 16");
    // Δx/Δt = 5/3 > 1: connecting them would need a faster-than-light signal

    // a 3+1 spacelike interval: x²+y²+z² − t² with space dominating (4+4+1 − 4 = +5)
    let s4 = Geonum::new(2.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(2.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(1.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(2.0, 1.0, 2.0).pow(2.0);
    assert_eq!(s4.angle.grade(), 0, "3+1 spacelike at grade 0");
    assert!(s4.near_mag(5.0), "|s²| = 9 − 4 = 5");
}

#[test]
fn its_lightlike() {
    // lightlike (null) separation: space equals time, |Δx| = |Δt|. the grade-2
    // time square exactly cancels the grade-0 space square — destructive
    // interference of a quantity against its own dual, [r,0] + [r,π] = 0. the
    // interval is null, the light cone itself, reachable only by a light ray
    let s = interval(4.0, 4.0); // Δx = Δt = 4
    assert!(s.mag < EPSILON, "lightlike interval is null — s² = 0");
    // Δx/Δt = 4/4 = 1: a light ray connects null-separated events

    // a 3+1 null interval: 3² + 4² + 0² = 5², a photon in the xy-plane on the cone
    let s4 = Geonum::new(3.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(4.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(0.0, 0.0, 1.0).pow(2.0)
        + Geonum::new(5.0, 1.0, 2.0).pow(2.0);
    assert!(s4.mag < EPSILON, "3+1 null: 3²+4² = 5², on the light cone");
}

#[test]
fn it_finds_the_light_cone_where_the_dual_cancels() {
    // the light cone is the boundary between the causal regions, and the grade
    // flips across it. exactly on the cone (x = t) the grade-2 time square cancels
    // the grade-0 space square — a quantity against its own dual, [r,0]+[r,π] = 0 —
    // so the interval is null. step off the cone and the grade reappears: more
    // space lands grade 0 (spacelike exterior), more time lands grade 2 (timelike
    // interior). the null cone separates the two
    let on_cone = interval(4.0, 4.0);
    assert!(
        on_cone.mag < EPSILON,
        "on the cone the dual cancels — s² = 0"
    );

    // step out toward space: the interval reappears spacelike, grade 0
    let exterior = interval(5.0, 4.0);
    assert!(
        exterior.mag > EPSILON,
        "off the cone the interval is nonzero"
    );
    assert_eq!(
        exterior.angle.grade(),
        0,
        "more space than time is spacelike"
    );

    // step in toward time: the interval reappears timelike, grade 2
    let interior = interval(4.0, 5.0);
    assert!(
        interior.mag > EPSILON,
        "off the cone the interval is nonzero"
    );
    assert_eq!(
        interior.angle.grade(),
        2,
        "more time than space is timelike"
    );
}

#[test]
fn it_shows_a_scalar_interval_discards_causal_structure() {
    // the lesson algebra_test draws for winding numbers, drawn here for causality:
    // a scalar |s²| cant tell timelike from spacelike. two events with s² = +9
    // and s² = −9 share the same scalar magnitude — the causal character is the
    // grade, an angle a scalar metric discards and then re-smuggles as a sign
    let spacelike = interval(5.0, 4.0); // 25 − 16 = +9
    let timelike = interval(4.0, 5.0); // 16 − 25 = −9

    // the scalar a metric reports is identical — it has thrown the angle away
    assert!(spacelike.near_mag(timelike.mag), "same scalar |s²| = 9");

    // the causal structure survives only in the grade: grade 0 vs grade 2
    assert_eq!(spacelike.angle.grade(), 0, "spacelike lives at grade 0");
    assert_eq!(timelike.angle.grade(), 2, "timelike lives at grade 2");

    eprintln!("\n  scalar |s²| = {:.0} for both events", spacelike.mag);
    eprintln!("  causality is the grade: spacelike grade 0, timelike grade 2");
    eprintln!("  a scalar metric discards it, then re-smuggles it as the −+++ sign");
}

// the scalar-coordinate callers below boost an event (t, x) along the x-axis and
// read the pair back. this thin adapter wraps Geonum::boost so the suite has ONE
// boost implementation. a boost is the SCALE half of scale_rotate: the method
// projects onto the two light-cone nulls and scales them reciprocally — the
// forward null t+x stretches by the Bondi factor k = e^α, the backward t−x
// shrinks by 1/k, zero rotation. magnitude = boost, angle = rotation
fn boost_xt(t: f64, x: f64, rapidity: f64) -> (f64, f64) {
    let boosted = Geonum::new_from_cartesian(x, t).boost(Angle::new(0.0, 1.0), rapidity.exp());
    let (cos, sin) = boosted.angle.cos_sin();
    (boosted.mag * sin, boosted.mag * cos) // (t', x')
}

#[test]
fn it_boosts_an_event_by_scaling_the_null_cone() {
    // Geonum::boost reproduces the standard hyperbolic boost: scaling the null
    // rays by e^±α gives back t' = t cosh α + x sinh α exactly, no cosh/sinh in
    // the method — just the reciprocal scaling of the two null projections
    let (t, x) = (5.0, 3.0); // a timelike event
    let alpha = 0.5;
    let (tp, xp) = boost_xt(t, x, alpha);

    assert!(
        (tp - (t * alpha.cosh() + x * alpha.sinh())).abs() < EPSILON,
        "t' = t cosh α + x sinh α"
    );
    assert!(
        (xp - (t * alpha.sinh() + x * alpha.cosh())).abs() < EPSILON,
        "x' = t sinh α + x cosh α"
    );

    // boosts compose by multiplying the scale factors, so rapidity is ADDITIVE:
    // e^α · e^β = e^(α+β). the geometric product turns boost composition into
    // magnitude multiplication, the way angle addition composes rotations
    let compose = Geonum::new(0.5_f64.exp(), 0.0, 1.0) * Geonum::new(0.9_f64.exp(), 0.0, 1.0);
    assert!(
        compose.near_mag((0.5_f64 + 0.9).exp()),
        "two boosts compose to one of summed rapidity"
    );
}

#[test]
fn it_boosts_an_event_by_projecting_onto_the_asymptotes_a_quarter_turn_apart() {
    // Geonum::boost is the conjugate-hyperbola picture: t²−x²=1 and its conjugate
    // x²−t²=1 share the asymptotes t = ±x — the light cone — one quarter turn
    // apart (π/4 and 3π/4). a boost is not a separate squeeze: the method projects
    // the event onto the two asymptotes and scales them oppositely by the Bondi
    // factor. projection + the quarter turn + a scale, no (t±x) touched by hand
    let alpha = 0.6_f64;
    let k = alpha.exp(); // the Bondi / Doppler factor

    // the event as one geonum in the (x, t) plane, boosted along x (axis = 0)
    let (t, x) = (2.0, 0.5);
    let event = Geonum::new_from_cartesian(x, t);
    let boosted = event.boost(Angle::new(0.0, 1.0), k);
    let (cos, sin) = boosted.angle.cos_sin();
    let (xb, tb) = (boosted.mag * cos, boosted.mag * sin);

    // it preserves the interval — it IS the lorentz boost, built from projections
    assert!(
        (tb * tb - xb * xb - (t * t - x * x)).abs() < 1e-9,
        "the asymptote-projection boost preserves t²−x²"
    );

    // the axis is a free parameter: boosting along a tilted spatial direction
    // moves the nulls with it (axis ± π/4) and still preserves the interval — the
    // squeeze is the same geometry pointed any way, not hard-coded to x
    let tilt = Angle::new(1.0, 5.0); // π/5
    let tilted = event.boost(tilt, k);
    let n = Geonum::new_with_angle(1.0, tilt);
    let along = event.mag * event.angle.project(tilt); // signed component on the axis
    let perp = event.reject(&n).mag; // perpendicular (the tilted-frame "time")
    let along_b = tilted.mag * tilted.angle.project(tilt);
    let perp_b = tilted.reject(&n).mag;
    assert!(
        ((perp_b * perp_b - along_b * along_b) - (perp * perp - along * along)).abs() < 1e-9,
        "a boost along any axis preserves the interval in that axis's frame"
    );

    // the conjugate hyperbola is the timelike sector turned that same quarter: a
    // point on t²−x²=1, rotated π/2, lands on the spacelike conjugate x²−t²=1.
    // the grade-0/grade-2 causal split is this quarter turn seen on the curves
    let on_timelike = Geonum::new_from_cartesian(0.5, (1.0 + 0.25_f64).sqrt());
    let turned = on_timelike.rotate(Angle::new(1.0, 2.0));
    let cx = turned.mag * turned.angle.grade_angle().cos();
    let ct = turned.mag * turned.angle.grade_angle().sin();
    assert!(
        (cx * cx - ct * ct - 1.0).abs() < EPSILON,
        "a quarter turn carries the timelike hyperbola onto its spacelike conjugate"
    );
}

#[test]
fn it_holds_the_causal_grade_invariant_under_a_boost() {
    // the trichotomy extends to boosts: a boost preserves the interval, so it
    // preserves the grade. timelike stays grade 2, spacelike stays grade 0, and a
    // null event stays on the cone. the causal class is the lorentz invariant
    let alpha = 0.7;

    for (t, x, grade, label) in [(5.0, 3.0, 2usize, "timelike"), (3.0, 5.0, 0, "spacelike")] {
        let before = interval(x, t);
        let (tp, xp) = boost_xt(t, x, alpha);
        let after = interval(xp.abs(), tp.abs()); // interval squares its inputs

        assert_eq!(
            after.angle.grade(),
            grade,
            "{label} stays {label} (grade {grade})"
        );
        assert!(
            after.near_mag(before.mag),
            "{label}: |s²| invariant under the boost"
        );
    }

    // a null event stays null — scaling fixes zero, so the light cone is
    // boost-invariant. this is the geonum statement of light-speed invariance
    let (tp, xp) = boost_xt(4.0, 4.0, alpha);
    assert!(
        interval(xp.abs(), tp.abs()).mag < EPSILON,
        "the light cone is invariant — a null event stays null under any boost"
    );
}

// an event as (time, space): time the magnitude on the boost-orthogonal axis,
// space a geonum [ρ, ψ] carrying its DIRECTION in the angle. a boost keeps the
// directional work in angle space — project the spatial geonum onto the boost
// line n (the cos of the angle difference), apply the one hyperbolic step to the
// (time, parallel) magnitudes, then reassemble by adding the boosted parallel
// back along n to the untouched perpendicular. the spatial direction rotates as
// a result, and that rotation is read straight off the geonum angle. no (x,y)
// component arithmetic — the boost is magnitude, the direction is angle
fn boost_event(time: f64, space: Geonum, n: Angle, rapidity: f64) -> (f64, Geonum) {
    let along = Geonum::new_with_angle(1.0, n);
    let perp = space.reject(&along); // perpendicular part, untouched by the boost
    let par = space.mag * space.angle.project(n); // signed length along n (cos of the diff)

    // the one hyperbolic step — scaling the (time ± parallel) null pair, magnitude
    let (t2, par2) = boost_xt(time, par, rapidity);

    // reassemble in angle space: boosted parallel back along n, plus the perp
    let pi = Angle::new(1.0, 1.0);
    let par_vec = Geonum::new_with_angle(par2.abs(), if par2 >= 0.0 { n } else { n + pi });
    (t2, par_vec + perp)
}

#[test]
fn it_keeps_the_wigner_rotation_in_angle_space() {
    // the same non-commutative case, done geometrically: the spatial direction is
    // a geonum angle throughout, decomposition is projection (cos of the angle
    // difference), the boost is the magnitude step, and the wigner rotation comes
    // out as a geonum ANGLE — no (x,y) components, no atan2
    let (alpha, beta) = (0.9, 0.7);
    let x_axis = Angle::new(0.0, 1.0); // 0
    let y_axis = Angle::new(1.0, 2.0); // π/2
    let at_rest = Geonum::new(0.0, 0.0, 1.0); // no spatial part

    // the net boost reached by x then y
    let (t1, s1) = boost_event(1.0, at_rest, x_axis, alpha);
    let (t2, s2) = boost_event(t1, s1, y_axis, beta);
    let net_dir = s2.angle;
    let net_rapidity = (s2.mag / t2).atanh();

    // R = undo the net boost ∘ the two boosts — it fixes the rest frame
    let r = |time: f64, space: Geonum| {
        let (a, b) = boost_event(time, space, x_axis, alpha);
        let (c, d) = boost_event(a, b, y_axis, beta);
        boost_event(c, d, net_dir, -net_rapidity)
    };
    let (tf, sf) = r(1.0, at_rest);
    assert!(
        (tf - 1.0).abs() < 1e-9 && sf.mag < 1e-9,
        "R returns rest to rest"
    );

    // R on a spatial probe: it fixes time and turns in the spatial plane. the
    // wigner angle is the geonum angle of the turned x-axis
    let (tx, sx) = r(0.0, Geonum::new(1.0, 0.0, 1.0)); // x-axis probe
    assert!(
        tx.abs() < 1e-9,
        "the residual fixes time — a rotation, not a boost"
    );
    assert!(sx.near_mag(1.0), "and preserves spatial length");
    let omega = sx.angle.grade_angle();
    assert!(
        (1e-2..1.0).contains(&omega),
        "two non-collinear boosts leave a geonum-angle rotation: Ω = {omega:.4}"
    );

    // R acts on the probe exactly as scale_rotate(1, Ω) — geonum's spiral with
    // the boost knob at unity. the boosts were scale_rotate(k, no turn) (pure
    // scale); the residual is scale_rotate(1, Ω) (pure rotation). one primitive,
    // two knobs: scale = boost, angle = rotation
    let spiral = Geonum::new(1.0, 0.0, 1.0).scale_rotate(1.0, sx.angle);
    assert!(
        spiral.near(&sx),
        "R = scale_rotate(1, Ω) on space — the boost undone, the wigner rotation left"
    );

    // the y-axis probe turns by the SAME geonum angle — a rigid rotation, read
    // entirely in angle space (the y-axis at π/2 lands at π/2 + Ω)
    let (_, sy) = r(0.0, Geonum::new(1.0, 1.0, 2.0)); // y-axis probe (π/2)
    assert!(
        (sy.angle.grade_angle() - (omega + FRAC_PI_2)).abs() < 1e-9,
        "both axes turn by Ω — a rotation, not a shear"
    );

    eprintln!("\n  wigner rotation as a geonum angle: Ω = {omega:.4} rad");

    // collinear control: a single boost leaves no rotation — the x-probe returns
    // unturned, Ω = 0. the rotation above is born of non-collinearity
    let solo = |time: f64, space: Geonum| {
        let (a, b) = boost_event(time, space, x_axis, alpha);
        let (sg, ss) = boost_event(1.0, at_rest, x_axis, alpha);
        boost_event(a, b, ss.angle, -(ss.mag / sg).atanh())
    };
    let (_, sx0) = solo(0.0, Geonum::new(1.0, 0.0, 1.0));
    assert!(
        sx0.angle.grade_angle() < 1e-9,
        "a single boost leaves no rotation — Ω = 0 collinear"
    );

    // so the [magnitude, angle] split survives the non-commutative case WITHOUT
    // leaving angle space: the boost is the magnitude step, the spatial direction
    // and the wigner rotation it leaves behind are geonum angles throughout
}

// the boosts above act on spacetime POINTS (t, x). the Angle::boost method acts
// on a DIRECTION on the celestial sphere — a light ray's polar angle θ from the
// boost axis. a unit direction has stereographic coordinate tan(θ/2), and a
// lorentz boost is the Möbius dilation that scales it by 1/k, the Bondi factor.
// geonum stores tan(θ/2) as its own half-tangent t, rational in (grade, t)
// across all four quadrants, so the boost is one rational scale — relativistic
// aberration with no cosh/sinh, the Penrose celestial-sphere picture

#[test]
fn it_aberrates_a_light_ray_by_scaling_the_half_tangent() {
    let ray = Angle::new(1.0, 3.0); // θ = π/3, 60° off the boost axis
    let k = 0.6_f64.exp(); // Bondi factor for rapidity 0.6

    // in the forward hemisphere the boost is just t → t/k: one division of the
    // stored half-tangent, no transcendentals
    let aberrated = ray.boost(k);
    assert!(
        (aberrated.t() - ray.t() / k).abs() < EPSILON,
        "forward hemisphere: t' = t/k"
    );

    // the boosted direction obeys the relativistic aberration formula
    // cos θ' = (cos θ + β)/(1 + β cos θ) — scaling the half-tangent IS stellar
    // aberration, recovered rationally from the stored ratio
    let beta = (k * k - 1.0) / (k * k + 1.0); // β = tanh φ, from the Bondi factor
    let (cos, _) = ray.cos_sin();
    let (cos_prime, _) = aberrated.cos_sin();
    assert!(
        (cos_prime - (cos + beta) / (1.0 + beta * cos)).abs() < EPSILON,
        "cos θ' = (cos θ + β)/(1 + β cos θ) — relativistic aberration"
    );

    // the forward axis is a fixed point: boosting θ = 0 (t = 0) leaves it on the
    // axis. (the backward pole θ = π is the other fixed point)
    let fixed = Angle::new(0.0, 1.0).boost(k);
    assert!(
        fixed.t().abs() < EPSILON,
        "the forward axis is fixed — t = 0 stays 0"
    );

    // boosts compose: boost by k1 then k2 = boost by k1·k2. the dilations multiply
    // (rapidity adds), the same composition law as the event boost
    let (k1, k2) = (0.4_f64.exp(), 0.5_f64.exp());
    let twice = ray.boost(k1).boost(k2);
    let once = ray.boost(k1 * k2);
    assert!(
        (twice.t() - once.t()).abs() < EPSILON,
        "boosts compose — dilations multiply, rapidity adds"
    );

    // headlight effect: a stronger boost crowds the ray toward the forward axis
    // (smaller t') — relativistic beaming, as a smaller stored ratio
    let strong = ray.boost(3.0_f64.exp());
    assert!(
        strong.t() < aberrated.t(),
        "a stronger boost pulls the ray toward the forward axis — the headlight effect"
    );
}

#[test]
fn it_boosts_a_backward_ray_across_the_blade_boundary() {
    let k = 0.6_f64.exp();
    let beta = (k * k - 1.0) / (k * k + 1.0);

    // a backward-hemisphere ray, θ = 2π/3 — past the π/2 boundary, blade 1, where
    // the stereographic coordinate is (1+t)/(1−t), still rational
    let ray = Angle::new(2.0, 3.0);
    assert_eq!(
        ray.blade(),
        1,
        "θ = 2π/3 sits in the backward hemisphere, blade 1"
    );

    let aberrated = ray.boost(k);
    let (cos, _) = ray.cos_sin();
    let (cos_prime, _) = aberrated.cos_sin();
    assert!(
        (cos_prime - (cos + beta) / (1.0 + beta * cos)).abs() < EPSILON,
        "cos θ' = (cos θ + β)/(1 + β cos θ) holds across the blade boundary"
    );

    // and the boost crosses the boundary: a strong enough boost swings the
    // backward ray (blade 1) into the forward hemisphere (blade 0) — relativistic
    // beaming pulling a rear ray to the front
    assert_eq!(
        aberrated.blade(),
        0,
        "the backward ray boosts forward, crossing into blade 0"
    );
}

#[test]
fn it_boosts_any_blade_via_the_grade() {
    let k = 0.6_f64.exp();
    let beta = (k * k - 1.0) / (k * k + 1.0);

    // a grade-2 direction, θ = 5π/4 — into the third quadrant, where the
    // stereographic coordinate is S = −1/t
    let ray = Angle::new(5.0, 4.0);
    assert_eq!(ray.grade(), 2, "θ = 5π/4 is grade 2");
    let aberrated = ray.boost(k);
    let (cos, _) = ray.cos_sin();
    let (cos_prime, _) = aberrated.cos_sin();
    assert!(
        (cos_prime - (cos + beta) / (1.0 + beta * cos)).abs() < EPSILON,
        "the aberration formula holds at grade 2 too"
    );

    // keyed on grade, not the literal blade: a direction one full turn on
    // (blade +4, same grade 0) boosts to the same place. any accumulated blade works
    let wound = Angle::new_with_blade(4, 1.0, 3.0); // θ = π/3 + 2π, grade 0
    let plain = Angle::new(1.0, 3.0); // θ = π/3
    assert!(
        (wound.boost(k).t() - plain.boost(k).t()).abs() < EPSILON,
        "a full turn of accumulated blade boosts identically — keyed on grade"
    );

    // the backward pole θ = π (grade 2, t = 0, the stereographic point at ∞) is a
    // fixed point — like the forward pole, the boost leaves it put
    let back_pole = Angle::new(1.0, 1.0); // π
    assert!(
        back_pole.boost(k).near(&back_pole),
        "the backward pole is fixed"
    );
}
