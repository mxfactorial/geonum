use geonum::*;

// minkowski space is the stereographic model flattened to scalars
//
// minkowski builds spacetime from five pieces: coordinates for an event, a
// time coordinate to translate along, a metric tensor to give them a sign,
// a linear group to preserve it, and s² as the invariant. every piece is a scalar, and the direction of
// anything has to be reconstructed by dividing coordinates
//
// the stereographic model starts from the direction and never leaves it.
// a direction is its half-tangent from the backward pole — the t geonum
// stores. a boost is that coordinate scaled by 1/k with the past-future
// dual as its only fixed points. causal class is the grade. the interval
// survives as a shadow, the product of the two null coordinates, which the
// scale leaves alone because k·(1/k) = 1
//
// each test runs one minkowski piece at full weight — the transcendental
// calls, the table, the matrix, the square — lands it on the geonum op it
// flattens, and ends with what fell off on the way. parity is the license
// to replace; the ledger of what was left behind is the reason. the
// physics the model carries lives elsewhere: the signature and the causal
// trichotomy in spacetime_test, the crossing height and the fixed pair in
// stereographic_test, the pair itself in cgt_test03
//
// run: cargo test --test minkowski_space_test -- --show-output

#[test]
fn it_stores_the_direction_minkowski_reconstructs_from_coordinates() {
    // minkowski: an event is the pair (t, x). its direction is derived: an
    // arctangent over the coordinates, a halving, a tangent — two
    // transcendental calls to reach the stereographic coordinate, and two
    // more, cos and sin, to get the coordinates back out of the angle
    let (x, t) = (0.5_f64, 2.0_f64);
    let theta = t.atan2(x);
    let half_tangent = (theta / 2.0).tan();
    let (x_back, t_back) = (theta.cos(), theta.sin());

    // geonum: the direction is the primitive, stored as the crossing height
    // of the ray from the backward pole — opp over hyp plus adj, one
    // division, no arctangent. it is the same number
    let event = Geonum::new_from_cartesian(x, t);
    let r = (x * x + t * t).sqrt();
    let crossing_height = t / (r + x);
    assert!(
        Geonum::scalar(event.angle.t()).near_mag(crossing_height),
        "the stored t is the pole ray's crossing height"
    );
    assert!(
        Geonum::scalar(crossing_height).near_mag(half_tangent),
        "and it is tan(θ/2), reached without the arctangent"
    );

    // the coordinates come back rationally from the stored ratio — the
    // same values cos and sin return, with no cos and no sin
    let (cos, sin) = event.angle.cos_sin();
    assert!(
        Geonum::scalar(cos).near_mag(x_back),
        "(1 − t²)/(1 + t²) is cos θ"
    );
    assert!(Geonum::scalar(sin).near_mag(t_back), "2t/(1 + t²) is sin θ");
    assert!(Geonum::scalar(event.mag * cos).near_mag(x), "x back");
    assert!(Geonum::scalar(event.mag * sin).near_mag(t), "t back");

    // a minkowski 4-vector is one geonum read four times. the shadows of
    // the event at the four blade addresses are x, t, −x, −t: the two
    // coordinates and the two signs minkowski carries as separate data,
    // each one cosine of an angle difference
    let shadows: Vec<f64> = (0..4).map(|d| event.project_to_dimension(d)).collect();
    assert!(
        Geonum::scalar(shadows[0] - x).near_mag(0.0),
        "blade 0 reads x"
    );
    assert!(
        Geonum::scalar(shadows[1] - t).near_mag(0.0),
        "blade 1 reads t"
    );
    assert!(
        Geonum::scalar(shadows[2] + x).near_mag(0.0),
        "blade 2 reads −x, the dual"
    );
    assert!(
        Geonum::scalar(shadows[3] + t).near_mag(0.0),
        "blade 3 reads −t, the dual"
    );

    // the readout at blade 1000 is the same call and the same x: a 1001st
    // coordinate with no storage added, since 1000 % 4 = 0. the count of
    // coordinates is a count of readouts, not a fact about space
    assert!(
        Geonum::scalar(event.project_to_dimension(1000) - x).near_mag(0.0),
        "the 1001st coordinate is x again"
    );

    // y and z are not two more shadows of this geonum. they are deeper
    // angles in a chain, passengers a boost of the leading angle leaves
    // untouched, and minkowski reconstructs each with its own arctangent
    // (stereographic_test::it_unwinds_any_sphere_one_angle_per_projection)
    eprintln!("left behind: atan2, tan, cos, sin — four transcendental calls per event — and four stored numbers for what one geonum reads four times. one division kept");
}

#[test]
fn it_reads_the_metric_tensor_as_one_dual() {
    // minkowski: ⟨u, v⟩ = η_μν u^μ v^ν with η = diag(−1, +1, +1, +1). the
    // −1 is installed by hand, "choosing a signature" is which slot gets
    // it, and the double sum runs over all sixteen entries
    let eta = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let inner = |u: [f64; 4], v: [f64; 4]| {
        let mut sum = 0.0;
        let mut zero_products = 0;
        for (mu, row) in eta.iter().enumerate() {
            for (nu, entry) in row.iter().enumerate() {
                if *entry == 0.0 {
                    zero_products += 1;
                }
                sum += entry * u[mu] * v[nu];
            }
        }
        (sum, zero_products)
    };

    // geonum: time sits a quarter turn from space. multiply two time
    // components and the angles add to π — the −1 is the product of two
    // quarter turns, computed, not installed. four products, no table
    let time = |t: f64| Geonum::new(t, 1.0, 2.0);
    let space = |x: f64| Geonum::new(x, 0.0, 1.0);
    let space_axis = Angle::new(0.0, 1.0);

    for (u, v) in [
        ([2.0, 0.5, 1.0, 1.5], [1.0, 1.5, 0.5, 2.0]),
        ([3.0, 1.0, 0.5, 0.5], [0.5, 2.0, 1.0, 1.0]),
        ([1.0, 2.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]),
    ] {
        let (table, zero_products) = inner(u, v);
        assert_eq!(
            zero_products, 12,
            "twelve of the sixteen entries multiply zero"
        );

        let angles = time(u[0]) * time(v[0])
            + space(u[1]) * space(v[1])
            + space(u[2]) * space(v[2])
            + space(u[3]) * space(v[3]);
        let signed = angles.mag * angles.angle.project(space_axis);
        assert!(
            Geonum::scalar(signed - table).near_mag(0.0),
            "the π between the two time products reads as η's −1"
        );
    }

    // η_tt = −1 is [1, π], the dual of the unit. the whole tensor is one
    // dual and three identities — the relative π of the pair
    // (spacetime_test::its_a_metric_signature, test 5)
    assert!(
        (time(1.0) * time(1.0)).near(&Geonum::scalar(1.0).dual()),
        "two quarter turns multiply to the dual of one"
    );
    assert_eq!(
        (time(1.0) * time(1.0)).angle.grade(),
        2,
        "η's minus is blade 1 + blade 1"
    );
    eprintln!("left behind: a 4×4 table, twelve zero entries multiplied per product, and the signature choice — one dual kept");
}

#[test]
fn it_diagonalizes_the_lorentz_matrix_into_the_dilation() {
    // minkowski: a boost is the matrix Λ = [[cosh φ, sinh φ], [sinh φ,
    // cosh φ]] acting on (t, x), a member of the linear group preserving η.
    // two hyperbolic calls to build it, four multiplies and two adds to
    // apply it
    let lambda = |phi: f64| [[phi.cosh(), phi.sinh()], [phi.sinh(), phi.cosh()]];
    let apply = |m: [[f64; 2]; 2], (t, x): (f64, f64)| {
        (m[0][0] * t + m[0][1] * x, m[1][0] * t + m[1][1] * x)
    };
    let phi = 0.6_f64;
    let k = phi.exp();
    let boost = lambda(phi);

    // its eigenvectors are the two null rays and its eigenvalues are k and
    // 1/k: the matrix is a dilation written in the wrong basis
    let (tf, xf) = apply(boost, (1.0, 1.0));
    assert!(
        Geonum::scalar(tf).near_mag(k),
        "the forward null stretches by k"
    );
    assert!(Geonum::scalar(xf).near_mag(k), "along itself");
    let (tb, xb) = apply(boost, (1.0, -1.0));
    assert!(
        Geonum::scalar(tb).near_mag(1.0 / k),
        "the backward null shrinks by 1/k"
    );
    assert!(Geonum::scalar(xb + 1.0 / k).near_mag(0.0), "along itself");
    assert!(
        Geonum::scalar(boost[0][0] * boost[1][1] - boost[0][1] * boost[1][0])
            .near_mag(k * (1.0 / k)),
        "det Λ = cosh² − sinh² = 1 = k·(1/k)"
    );

    // geonum: the dilation is the primitive. Geonum::boost projects onto
    // the two nulls and scales them by k and 1/k, and lands where Λ lands
    let (x, t) = (0.5_f64, 2.0_f64);
    let axis = Angle::new(0.0, 1.0);
    let boosted = Geonum::new_from_cartesian(x, t).boost(axis, k);
    let (cos, sin) = boosted.angle.cos_sin();
    let (tl, xl) = apply(boost, (t, x));
    assert!(
        Geonum::scalar(boosted.mag * sin).near_mag(tl),
        "t' agrees with Λ"
    );
    assert!(
        Geonum::scalar(boosted.mag * cos).near_mag(xl),
        "x' agrees with Λ"
    );

    // composition at full weight: Λ(φ₁)Λ(φ₂) is a matrix product, eight
    // multiplies and four adds, and it equals Λ(φ₁ + φ₂) only by the
    // hyperbolic addition identities. geonum: k₁·k₂, one multiply — mags
    // multiply, so rapidities add with nothing to prove
    let (phi1, phi2) = (0.4_f64, 0.5_f64);
    let (a, b) = (lambda(phi1), lambda(phi2));
    let product = [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ];
    let composed = lambda(phi1 + phi2);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                Geonum::scalar(product[i][j]).near_mag(composed[i][j]),
                "Λ(φ₁)Λ(φ₂) = Λ(φ₁ + φ₂) entry by entry"
            );
        }
    }
    let ray = Angle::new(1.0, 3.0);
    let (k1, k2) = (phi1.exp(), phi2.exp());
    assert!(
        Geonum::scalar(ray.boost(k).t()).near_mag(ray.t() / k),
        "on a direction the boost is t/k, one division"
    );
    assert!(
        ray.boost(k1).boost(k2).near(&ray.boost(k1 * k2)),
        "two dilations compose to one of the product factor"
    );

    // dimension at full weight: in n dimensions Λ is (n+1)×(n+1) and every
    // passenger rides through it — a boost in the 1001st dimension is a
    // million-entry matrix with four working entries. the dilation is the
    // same call on the leading angle at any blade, and the interval it
    // preserves is read the same way
    let far = Angle::new_with_blade(1000, 0.0, 1.0); // the 1001st dimension as an axis
    let n = Geonum::new_with_angle(1.0, far);
    let event = Geonum::new_with_angle(
        (x * x + t * t).sqrt(),
        far + Angle::new_from_cartesian(x, t),
    );
    let along = event.mag * event.angle.project(far);
    let perp = event.reject(&n).mag;
    let moved = event.boost(far, k);
    let along_b = moved.mag * moved.angle.project(far);
    let perp_b = moved.reject(&n).mag;
    assert!(
        Geonum::scalar(perp_b * perp_b - along_b * along_b).near_mag(perp * perp - along * along),
        "the boost at blade 1000 preserves the interval with the same call"
    );
    assert!(
        Geonum::scalar(along_b).near_mag(xl),
        "and lands the same x' the 2×2 lands — no 1001×1001 built"
    );
    eprintln!("left behind: cosh, sinh, a 2×2 per boost, an 8-multiply composition proof, and an (n+1)×(n+1) matrix at n dimensions — one scale kept");
}

#[test]
fn it_reads_the_interval_as_the_product_of_the_null_coordinates() {
    // minkowski: s² = η_μν u^μ u^ν = −t² + x² = −(t + x)(t − x). the scalar
    // invariant is minus the product of the two null coordinates, and it
    // is squared because a scalar cannot hold a sign any other way
    let (x, t) = (0.5_f64, 2.0_f64);
    let s_squared = |t: f64, x: f64| -t * t + x * x;
    let (u, v) = (t + x, t - x);
    assert!(
        Geonum::scalar(-u * v - s_squared(t, x)).near_mag(0.0),
        "s² = −u·v"
    );

    // geonum: the null coordinates are the event's projections onto the
    // rays a quarter turn either side of the axis, π/4 and 3π/4
    let fwd = Angle::new(1.0, 4.0);
    let bwd = Angle::new(3.0, 4.0);
    let root_two = 2.0_f64.sqrt();
    let nulls = |g: &Geonum| {
        (
            g.mag * g.angle.project(fwd) * root_two,
            g.mag * g.angle.project(bwd) * root_two,
        )
    };
    let event = Geonum::new_from_cartesian(x, t);
    let (u0, v0) = nulls(&event);
    assert!(
        Geonum::scalar(u0).near_mag(u),
        "t + x is the forward null projection"
    );
    assert!(
        Geonum::scalar(v0).near_mag(v),
        "t − x is the backward null projection"
    );

    // the boost scales them reciprocally, so their product is what it
    // leaves alone: the invariant is k·(1/k) = 1 read on the event
    let k = 0.6_f64.exp();
    let axis = Angle::new(0.0, 1.0);
    let boosted = event.boost(axis, k);
    let (u1, v1) = nulls(&boosted);
    assert!(
        Geonum::scalar(u1).near_mag(k * u),
        "the forward coordinate stretches"
    );
    assert!(
        Geonum::scalar(v1).near_mag(v / k),
        "the backward coordinate shrinks"
    );
    assert!(
        Geonum::scalar(u1 * v1).near_mag(u * v),
        "their product is the interval"
    );

    // the light cone is where one null coordinate is zero. as a direction
    // it is a fixed point of the dilation, and as a magnitude it scales by
    // k — the doppler factor. minkowski reads the same ray as s² = 0
    // (stereographic_test::it_fixes_exactly_one_dual_pair)
    let on_cone = Geonum::new_from_cartesian(1.0, 1.0);
    let (_, v_cone) = nulls(&on_cone);
    assert!(
        Geonum::scalar(v_cone).near_mag(0.0),
        "on the cone the backward coordinate is zero"
    );
    let moved = on_cone.boost(axis, k);
    assert!(
        moved.angle.base_angle().near(&on_cone.angle.base_angle()),
        "the null ray is fixed as a direction"
    );
    assert!(
        moved.near_mag(k * on_cone.mag),
        "and scaled by k as a magnitude"
    );

    // the square at full weight: it erases orientation. the event ahead
    // (t = 2) and the event behind (t = −2) have the same s², so minkowski
    // cannot tell the future cone from the past cone and adds a
    // time-orientation axiom by hand — a chosen future. the event as a
    // vector holds it: its time shadow sits on the dual pair, grade 0
    // ahead, grade 2 behind, and no axiom is consulted
    let ahead = Geonum::new_from_cartesian(x, t);
    let behind = Geonum::new_from_cartesian(x, -t);
    assert!(
        Geonum::scalar(s_squared(t, x) - s_squared(-t, x)).near_mag(0.0),
        "s² is blind to the sign of t"
    );
    let time_axis = Angle::new(1.0, 2.0);
    assert_eq!(
        ahead.project_to_angle(time_axis).angle.grade(),
        0,
        "ahead: the time shadow sits on the future ray"
    );
    assert_eq!(
        behind.project_to_angle(time_axis).angle.grade(),
        2,
        "behind: the time shadow sits on the past ray, its dual"
    );
    assert_eq!(
        ahead.angle.grade(),
        0,
        "the event ahead is on the lower half turn"
    );
    assert_eq!(
        behind.angle.grade(),
        3,
        "the event behind is on the upper — the blade is the orientation"
    );

    // in minkowski the sign of u·v sorts the causal classes. in geonum the
    // classes are the grade of the assembled interval, no square taken
    // (spacetime_test::its_timelike, its_spacelike, its_lightlike)
    eprintln!("left behind: the square, and the time-orientation axiom that patches what the square erased — one grade kept");
}

#[test]
fn it_counts_time_as_winding_where_minkowski_translates_along_t() {
    // minkowski: t is a coordinate. time passing is translation along the
    // t axis, the present is the hyperplane t = const picked by a value,
    // and the future differs from the past by the sign of a float. at full
    // weight the coordinate is a float accumulator: one second is t += 1.0
    let mut t_axis = 0.0_f64;

    // geonum: time is the count of turns. one second is one full turn,
    // blade + 4, the rate is the quarter turn, and the moment is the
    // cancelled sum of the event ahead and behind — no hyperplane chosen
    let tick = Angle::new(4.0, 2.0);
    let mut present = Geonum::new(1.0, 0.0, 1.0);
    for second in 0..=5usize {
        assert_eq!(
            present.angle.blade(),
            4 * second,
            "the clock reads the blade"
        );
        assert_eq!(present.angle.grade(), 0, "and the position never moves");
        let ahead = present.differentiate();
        let behind = present.integrate();
        assert_eq!(ahead.angle.grade(), 1, "ahead is the quarter turn on");
        assert_eq!(behind.angle.grade(), 3, "behind is its dual");
        let moment = ahead + behind;
        assert!(moment.near_mag(0.0), "the moment is the cancelled sum");
        assert_eq!(
            moment.angle.blade(),
            8 * second + 4,
            "and its record winds with the clock"
        );
        assert!(
            Geonum::scalar(t_axis).near_mag(second as f64),
            "the float keeps pace for now"
        );
        t_axis += 1.0;
        present = present.rotate(tick);
    }

    // the rate: minkowski differentiates along t with a limit; the model
    // rotates by a quarter turn
    // (calculus_test::it_shows_limits_discard_what_angles_preserve)
    assert!(
        present
            .differentiate()
            .near(&present.rotate(Angle::new(1.0, 2.0))),
        "d/dt is the quarter turn"
    );

    // the ledger, at scale: a float time coordinate absorbs a tick once the
    // count passes 2^53 — t + 1 is t, bit for bit — while the blade adds
    // four exactly. the timing industry ships the fix by hand as a 96-bit
    // second-plus-fraction field (atomic_clock_test)
    let far_future = 2.0_f64.powi(53);
    assert!(
        far_future + 1.0 == far_future,
        "the float clock swallows the second"
    );
    let wound = Angle::new_with_blade(4 * (1usize << 53), 0.0, 1.0);
    let next = wound + tick;
    assert_eq!(next.blade() - wound.blade(), 4, "the blade clock counts it");
    eprintln!("left behind: a t axis to translate along, a hyperplane to pick, a sign to orient, a limit to take, and a float that drops seconds past 2^53 — one winding count kept");
}

#[test]
fn it_reads_two_positions_on_one_tick_as_a_vector_with_no_time_shadow() {
    // minkowski: two events at the same t differ by (Δt, Δx) = (0, Δx), a
    // spacelike separation on the hyperplane of simultaneity. apply Λ to
    // each and subtract, eight multiplies and four adds, and Δt' comes out
    // as Δx sinh φ: simultaneity was this frame's reading. the textbook
    // route is γ(Δt − vΔx), with the sign convention argued
    let phi = 0.6_f64;
    let k = phi.exp();
    let lambda = |(t, x): (f64, f64)| {
        (
            t * phi.cosh() + x * phi.sinh(),
            t * phi.sinh() + x * phi.cosh(),
        )
    };
    let (a_t, a_x) = (2.0_f64, 0.5_f64);
    let (b_t, b_x) = (2.0_f64, 2.0_f64);
    let dx = b_x - a_x;
    let (a1, b1) = (lambda((a_t, a_x)), lambda((b_t, b_x)));
    let (dt_prime, dx_prime) = (b1.0 - a1.0, b1.1 - a1.1);
    assert!(
        Geonum::scalar(dt_prime).near_mag(dx * phi.sinh()),
        "Δt' = Δx sinh φ"
    );
    assert!(
        Geonum::scalar(dx_prime).near_mag(dx * phi.cosh()),
        "Δx' = Δx cosh φ"
    );

    // geonum: the two positions are two geonums on one tick, and their
    // difference is one subtraction. it has no shadow on the time axis and
    // a full shadow on the present, grade 0 — a vector lying in the present
    // plane. its interval is Δx², spacelike
    let time_axis = Angle::new(1.0, 2.0);
    let present_axis = Angle::new(0.0, 1.0);
    let a = Geonum::new_from_cartesian(a_x, a_t);
    let b = Geonum::new_from_cartesian(b_x, b_t);
    let separation = b - a;
    assert!(
        separation.project_to_angle(time_axis).near_mag(0.0),
        "no time shadow: the pair shares the tick"
    );
    let along = separation.project_to_angle(present_axis);
    assert!(
        along.near_mag(dx),
        "the separation lies in the present plane"
    );
    assert_eq!(along.angle.grade(), 0, "pointing from a to b");
    let interval = |g: &Geonum| {
        let s = g.project_to_angle(present_axis).mag;
        let t = g.project_to_angle(time_axis).mag;
        Geonum::new(s, 0.0, 1.0).pow(2.0) + Geonum::new(t, 1.0, 2.0).pow(2.0)
    };
    assert_eq!(interval(&separation).angle.grade(), 0, "spacelike");
    assert!(interval(&separation).near_mag(dx * dx), "Δx²");

    // boost both and subtract: the separation tilts off the present plane
    // and picks up a time shadow of Δx sinh φ — the numbers Λ gave, read
    // as a vector rotating rather than a hyperplane failing to hold
    let axis = Angle::new(0.0, 1.0);
    let moving = b.boost(axis, k) - a.boost(axis, k);
    assert!(
        moving.project_to_angle(time_axis).near_mag(dx * phi.sinh()),
        "a time shadow appears"
    );
    assert_eq!(
        moving.project_to_angle(time_axis).angle.grade(),
        0,
        "b now reads later than a"
    );
    assert!(
        moving
            .project_to_angle(present_axis)
            .near_mag(dx * phi.cosh()),
        "the present shadow stretches"
    );
    assert!(
        interval(&moving).near_mag(dx * dx),
        "the interval is the same Δx² from the moving frame"
    );
    assert_eq!(interval(&moving).angle.grade(), 0, "still spacelike");

    // the boost is linear, so the separation boosts as one vector: one
    // scale on the difference, not Λ twice and a subtraction. the t shadow
    // is the frame's readout; the separation is what both frames boosted
    let once = separation.boost(axis, k);
    assert!(
        once.near_mag(moving.mag),
        "boosting the separation is the difference of the boosts"
    );
    assert!(
        once.angle.base_angle().near(&moving.angle.base_angle()),
        "same direction"
    );

    // the boost advances the record. each event went through two
    // projections and a sum, and its blade counts them: from blade 0 to
    // blade 4. build a fresh event at the boosted coordinates and it reads
    // the same t and the same x, indistinguishable to minkowski, at blade
    // 0. the t shadow is the readout that forgets the boost; the winding
    // is the coordinate that counts it
    let a_moving = a.boost(axis, k);
    let (cos, sin) = a_moving.angle.cos_sin();
    let fresh = Geonum::new_from_cartesian(a_moving.mag * cos, a_moving.mag * sin);
    assert!(
        fresh
            .project_to_angle(time_axis)
            .near_mag(a_moving.project_to_angle(time_axis).mag),
        "the same t"
    );
    assert!(
        fresh.angle.base_angle().near(&a_moving.angle.base_angle()),
        "the same direction"
    );
    assert_eq!(a.angle.blade(), 0, "before the boost the record reads 0");
    assert_eq!(a_moving.angle.blade(), 4, "the boost wound the record to 4");
    assert_eq!(fresh.angle.blade(), 0, "the event that never moved reads 0");
    assert_eq!(
        b.boost(axis, k).angle.blade(),
        4,
        "b's record wound the same"
    );
    eprintln!("left behind: a hyperplane of simultaneity, Λ applied twice and subtracted, γ(Δt − vΔx) with its sign convention, and a t that cannot tell a boosted event from one that never moved — one subtraction, one scale, and the record of both kept");
}
