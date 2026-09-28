use geonum::*;

// the stored t is a drawn construction, not a formula: draw the ray from the
// backward pole of the circle through the direction's point, and t is the
// height where that ray crosses the perpendicular diameter. this suite proves
// the drawing — the crossing height is the stored ratio, the inscribed angle
// theorem is why it's the half-tangent, a quarter turn advances the height by
// the order-4 möbius map m(s) = (1+s)/(1−s), and Angle::boost dilates the
// drawn line by 1/k at every grade with one dual pair as its whole fixed set
//
// the dilation's physics lives elsewhere: stellar aberration and the headlight
// effect in spacetime_test, kepler's anomaly conversion in anomaly_test. here
// the subject is the line those suites compute on — and the riemann sphere the
// traditional account wraps around it collapses into one geonum whose boost is
// a scalar multiplication

#[test]
fn it_draws_t_as_the_pole_rays_crossing_height() {
    // directions across blade 0: rebuilding the direction from the drawn
    // crossing height recovers the angle — the height is the stored ratio
    let angles = [
        Angle::new(1.0, 6.0),
        Angle::new(1.0, 4.0),
        Angle::new(1.0, 3.0),
        Angle::new(2.0, 5.0),
    ];
    for a in angles {
        let rebuilt = Angle::from_parts(0, crossing_height(a));
        assert!(
            rebuilt.near(&a),
            "the drawn crossing height rebuilds the direction"
        );
    }

    // the 3-4-5 triangle lands the construction on a rational: t = 3/(5+4) = 1/3
    let a = Angle::new_from_cartesian(4.0, 3.0);
    let third = Angle::from_parts(0, 1.0 / 3.0);
    assert!(a.near(&third), "opp/(hyp + adj) = 3/9 = 1/3");
    assert!(
        Angle::from_parts(0, crossing_height(a)).near(&third),
        "and the drawn crossing height is the same 1/3"
    );

    // the denominator hyp + adj is the drawn ray's run: on the circle of
    // radius 5 the pole sits at −5 and the point at (4, 3), so the ray runs
    // 5 + 4 = 9 and rises 3 — new_from_cartesian's formula is this ray's slope
    let run = 5.0 + 4.0;
    let rise = 3.0;
    assert!(
        Angle::from_parts(0, rise / run).near(&a),
        "the projection ratio is rise over run of the pole ray"
    );
}

#[test]
fn it_halves_the_central_angle_at_the_backward_pole() {
    // inscribed angle theorem: the arc from the forward pole to the direction
    // subtends the central angle θ at the center and θ/2 at the backward pole,
    // so the chord from the pole points at half the central angle — the pole
    // ray's slope is tan(θ/2), and that is why t is the half-tangent
    let angles = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 3.0), // π/3
        Angle::new(2.0, 3.0), // 2π/3 — blade 1
        Angle::new(5.0, 6.0), // 5π/6
    ];
    for a in angles {
        let (x, y) = a.cos_sin();
        let chord = Angle::new_from_cartesian(1.0 + x, y);
        assert!(
            chord.near(&(a * 0.5)),
            "the chord from the backward pole points at half the central angle"
        );
    }

    // past the diameter the chord picks up the D event: the point sits below
    // the axis, so the chord direction is θ/2 plus a half turn
    let reflex = Angle::new(5.0, 4.0); // 5π/4
    let (x, y) = reflex.cos_sin();
    let chord = Angle::new_from_cartesian(1.0 + x, y);
    assert!(
        chord.near(&(reflex * 0.5 + Angle::new(1.0, 1.0))),
        "past the diameter the half-angle chord carries a half turn"
    );
}

#[test]
fn it_iterates_the_quarter_turn_as_an_order_4_mobius_map() {
    // on the circle a quarter turn is Q; on the drawn line it is the möbius
    // map m(s) = (1+s)/(1−s). the per-grade coordinate table is m iterated on
    // the stored t, and m returns after four applications — the blade lattice
    // mod 4, acting on the line
    let m = |s: f64| (1.0 + s) / (1.0 - s);
    let a = Angle::new(1.0, 5.0); // π/5, blade 0
    let q = Angle::new(1.0, 2.0);
    let t = a.t();

    let h0 = crossing_height(a);
    let h1 = crossing_height(a + q);
    let h2 = crossing_height(a + q + q);
    let h3 = crossing_height(a + q + q + q);

    // each quarter turn of the direction advances the drawn height by m
    assert!(drawn(h1).near(&drawn(m(h0))), "Q acts on the line as m");
    assert!(drawn(h2).near(&drawn(m(h1))), "a second Q is m again");
    assert!(drawn(h3).near(&drawn(m(h2))), "and a third");

    // the four heights are the per-grade coordinate table, generated from t
    assert!(drawn(h0).near(&drawn(t)), "grade 0: s = t");
    assert!(
        drawn(h1).near(&drawn((1.0 + t) / (1.0 - t))),
        "grade 1: s = (1+t)/(1−t) = m(t)"
    );
    assert!(
        drawn(h2).near(&drawn(-1.0 / t)),
        "grade 2: s = −1/t = m²(t)"
    );
    assert!(
        drawn(h3).near(&drawn((t - 1.0) / (t + 1.0))),
        "grade 3: s = (t−1)/(t+1) = m³(t)"
    );

    // a fourth application closes the cycle: m has order 4, the full turn
    assert!(
        drawn(m(h3)).near(&drawn(h0)),
        "m⁴ returns the height — the quadrature on the drawn line"
    );
}

#[test]
fn it_boosts_every_grade_by_one_dilation_of_the_drawn_line() {
    // Angle::boost divides the drawn height by k — a pure dilation of the
    // line, at every grade, whichever branch of the coordinate table the
    // direction sits on. spacetime_test asserts this through the aberration
    // formula; here it is read straight off the drawing
    let rays = [
        Angle::new(1.0, 5.0), // grade 0
        Angle::new(3.0, 4.0), // grade 1
        Angle::new(6.0, 5.0), // grade 2
        Angle::new(9.0, 5.0), // grade 3
    ];
    for k in [2.0, 0.5, 0.6_f64.exp()] {
        for r in rays {
            let before = drawn(crossing_height(r));
            let after = drawn(crossing_height(r.boost(k)));
            assert!(
                after.near(&before.scale(1.0 / k)),
                "the boost divides the drawn height by k at every grade"
            );
        }
    }
}

#[test]
fn it_fixes_exactly_one_dual_pair() {
    let k = 0.6_f64.exp();

    // a fixed direction solves h = h/k on the line: h = 0 (the forward pole)
    // or no finite height at all (the backward pole, whose pole ray never
    // crosses the diameter). those two directions are one dual pair — the
    // second pole is dual() of the first, not a separate special case
    let forward = Angle::new(0.0, 1.0);
    let backward = forward.dual();
    assert!(
        backward.is_opposite(&forward),
        "the two poles are one dual pair, a diameter apart"
    );
    assert!(forward.boost(k).near(&forward), "height 0 stays put");
    assert!(
        backward.boost(k).near(&backward),
        "the poleward ray with no crossing stays put"
    );

    // off the pair, every direction moves — one witness per grade
    let movers = [
        Angle::new(1.0, 5.0), // grade 0
        Angle::new(3.0, 4.0), // grade 1
        Angle::new(6.0, 5.0), // grade 2
        Angle::new(9.0, 5.0), // grade 3
    ];
    for r in movers {
        assert!(
            !r.boost(k).near(&r),
            "off the dual pair every direction moves"
        );
    }

    // k → 0 dilates every finite height to ∞: the whole circle collapses onto
    // the dual pole, the dilation's other fixed point
    for r in movers {
        assert_eq!(
            r.boost(0.0),
            backward,
            "at k = 0 every ray lands on the dual pole"
        );
    }
}

#[test]
fn it_boosts_about_any_axis_by_conjugating_the_dilation() {
    // every boost in the repo dilates toward θ = 0, but the forward pole is
    // wherever you put it: rotate the axis home, dilate, rotate back —
    // (θ − axis).boost(k) + axis. no new primitive, a composition of three
    // existing ops that carries the whole fixed-pair structure to the new pole
    let tilted = |dir: Angle, axis: Angle, k: f64| (dir - axis).boost(k) + axis;

    let axis = Angle::new(1.0, 3.0); // π/3, an axis off every lattice point
    let k = 0.6_f64.exp();

    // the fixed pair moves with the axis: the axis and its dual stay put
    assert!(
        tilted(axis, axis, k).near(&axis),
        "the chosen axis is the forward pole of its own boost"
    );
    assert!(
        tilted(axis.dual(), axis, k).near(&axis.dual()),
        "and its dual is the backward pole"
    );

    // relative to the axis, the drawn height dilates by 1/k — conjugation
    // transports the dilation, not an approximation of it
    let rays = [
        Angle::new(1.0, 5.0), // grade 0
        Angle::new(3.0, 4.0), // grade 1
        Angle::new(6.0, 5.0), // grade 2
        Angle::new(9.0, 5.0), // grade 3
    ];
    for r in rays {
        let before = drawn(crossing_height(r - axis));
        let after = drawn(crossing_height(tilted(r, axis, k) - axis));
        assert!(
            after.near(&before.scale(1.0 / k)),
            "the height measured from the tilted axis divides by k"
        );
    }

    // off the tilted pair every direction moves — measured at base angle so
    // the movement is geometric, not blade bookkeeping from the conjugation
    for r in rays {
        assert!(
            !tilted(r, axis, k).base_angle().near(&r.base_angle()),
            "off the tilted dual pair every direction moves"
        );
    }

    // conjugating by the forward pole itself is the plain boost — axis 0
    // recovers the dilation this suite began with
    let home = Angle::new(0.0, 1.0);
    for r in rays {
        assert!(
            tilted(r, home, k).near(&r.boost(k)),
            "axis 0 recovers the untilted boost"
        );
    }
}

#[test]
fn it_doesnt_need_the_riemann_sphere() {
    // the traditional account puts every light ray on a sphere with complex
    // stereographic coordinate ζ = e^{iφ}·tan(θ/2) and boosts by ζ → ζ/k.
    // that coordinate factors as [tan(θ/2), φ] — magnitude and angle, a
    // geonum. the factorization is exponential_test's split of e across the
    // two numbers: e^{iφ} is rotation living in the angle, k = e^rapidity is
    // growth living in the magnitude. so the axial boost is a real-exponential
    // step — multiplication by the scalar 1/k, azimuth invariance is
    // angles-add computing 0 + φ — and the meridian magnitude is the drawn
    // height the rest of this suite dilates. the foil grinds the same
    // aberration out of a boosted null 4-vector, component by component, and
    // its transverse pair is discovered to be a passenger
    let k = 0.6_f64.exp();
    let beta = (k * k - 1.0) / (k * k + 1.0);
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();

    let polars = [Angle::new(1.0, 5.0), Angle::new(3.0, 4.0)]; // blades 0 and 1
    let azimuths = [Angle::new(2.0, 7.0), Angle::new(6.0, 5.0)]; // grades 0 and 2

    for a in polars {
        for phi in azimuths {
            // foil: the null ray as 4 components, boosted entry by entry
            let theta = a.grade_angle();
            let px = theta.sin() * phi.grade_angle().cos();
            let py = theta.sin() * phi.grade_angle().sin();
            let pz = theta.cos();
            let e_prime = gamma * (1.0 + beta * pz);
            let pz_prime = gamma * (pz + beta);
            // the boost matrix's transverse block is identity: px, py ride
            let cos_prime = pz_prime / e_prime;
            let sin_prime = (px * px + py * py).sqrt() / e_prime;
            let foil_half_tangent = sin_prime / (1.0 + cos_prime);
            let foil_theta_prime = cos_prime.acos();
            let foil_azimuth = Angle::new_from_cartesian(px, py);

            // the sphere coordinate is a geonum: the drawn height as
            // magnitude, the azimuth as angle. the boost is one product
            let zeta = Geonum::new_with_angle(crossing_height(a), phi);
            let boosted = zeta * Geonum::scalar(1.0 / k);

            // azimuth invariance computed through the op: angles add 0 + φ
            assert_eq!(boosted.angle, phi, "the boost adds zero azimuth");
            // and the foil discovers the same invariance in its own output
            assert!(
                foil_azimuth.near(&phi),
                "the foil's transverse components rebuild the input azimuth"
            );

            // the product's magnitude is the foil's boosted half-tangent
            assert!(
                boosted.near_mag(foil_half_tangent),
                "one multiplication lands the 4-component result"
            );

            // Angle::boost on the meridian alone reaches the polar angle the
            // foil ground out of four components
            assert!(
                a.boost(k).near_rad(foil_theta_prime),
                "the meridian angle carries the whole aberration"
            );

            // and the geonum's magnitude is the drawn height of that meridian
            assert!(
                drawn(boosted.mag).near(&drawn(crossing_height(a.boost(k)))),
                "the sphere coordinate's magnitude is the suite's drawn line"
            );
        }
    }

    // rotation about the axis is multiplication by [1, δφ] — an imaginary-
    // exponential step, all turn; the boost is multiplication by [1/k, 0] — a
    // real-exponential step, all growth. one product each, so they commute —
    // the subgroup the sphere account writes as complex multiplication is the
    // exponential distributing its step between the two slots
    let zeta = Geonum::new_with_angle(crossing_height(Angle::new(1.0, 5.0)), Angle::new(2.0, 7.0));
    let spin = Geonum::new(1.0, 1.0, 6.0);
    let squeeze = Geonum::scalar(1.0 / k);
    assert!(
        ((zeta * spin) * squeeze).near(&((zeta * squeeze) * spin)),
        "axial rotation and boost commute as products"
    );
}

#[test]
fn it_unwinds_any_sphere_one_angle_per_projection() {
    // the projective geometry between the n-sphere and its plane is the
    // circle→line drawing, unchanged: a point on the traditional n-sphere is
    // a chain of angles, and projecting from the backward pole converts the
    // LEADING angle into a magnitude — the same crossing height the suite
    // drew on the circle — while every deeper angle rides untouched. the
    // dimension never parameterizes the op; it only counts the passengers.
    // so the projection iterates, and the named family is the recursion's
    // trace: five angles down, each level IS the next rung — glome→space at
    // three angles left, sphere→plane at two (the textbook picture), and
    // circle→line at one, where the recursion bottoms out on the drawing
    // this suite opened with. one op, distinguished by nothing but how many
    // angles are still waiting
    let chain = [
        Angle::new(1.0, 5.0), // π/5
        Angle::new(1.0, 3.0), // π/3
        Angle::new(2.0, 3.0), // 2π/3
        Angle::new(3.0, 4.0), // 3π/4
        Angle::new(6.0, 5.0), // 6π/5 — the last angle sweeps the full circle
    ];

    let mut comps = sphere_point(&chain);
    for (level, a) in chain.iter().enumerate() {
        let landing = project_from_pole(&comps);

        if landing.len() == 1 {
            // the base of the recursion is the circle→line drawing itself:
            // one component, the signed crossing height
            assert!(
                drawn(landing[0]).near(&drawn(crossing_height(*a))),
                "the recursion bottoms out at the circle's crossing height"
            );
            continue;
        }

        // the landing radius at every level is the leading angle's
        // circle→line height — the same drawing, no dimension consulted
        let radius = landing.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(
            drawn(radius).near(&drawn(crossing_height(*a))),
            "level {level}: the landing radius is the leading angle's crossing height"
        );

        // the passengers: the landing direction is the sphere one angle
        // shorter, its leading angle the next in the chain, unmoved
        comps = landing.iter().map(|c| c / radius).collect();
        let read = if comps.len() == 2 {
            Angle::new_from_cartesian(comps[0], comps[1])
        } else {
            leading_angle(&comps)
        };
        assert!(
            read.near(&chain[level + 1]),
            "level {level}: the projection leaves the deeper angles unmoved"
        );
    }

    // Angle::boost on the leading angle is the plane's radial dilation at any
    // count of passengers: the landing radius divides by k, the landing
    // direction — every deeper angle — rides
    let k = 0.6_f64.exp();
    let mut boosted_chain = chain;
    boosted_chain[0] = chain[0].boost(k);
    let plain = project_from_pole(&sphere_point(&chain));
    let boosted = project_from_pole(&sphere_point(&boosted_chain));
    let plain_r = plain.iter().map(|c| c * c).sum::<f64>().sqrt();
    let boosted_r = boosted.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!(
        drawn(boosted_r).near(&drawn(plain_r).scale(1.0 / k)),
        "boosting the leading angle dilates the landing radius by 1/k"
    );
    for (b, p) in boosted.iter().zip(&plain) {
        assert!(
            drawn(b / boosted_r).near(&drawn(p / plain_r)),
            "while the landing direction — every passenger — rides unmoved"
        );
    }
}

#[test]
fn it_writes_the_sphere_into_the_flat_planes_radial_law() {
    // the landing target of the drawing is flat by construction, and the
    // unwinding test showed every angle but the leading one rides untouched —
    // so an observer reading the plane measures every local angle undistorted
    // and can detect the sphere only through one channel: the radial
    // magnitude law. this test writes that law. locally the flat ruler is
    // blind to it (the disagreement opens at third order); at any separation
    // the law is one rational identity in drawn heights; and toward the
    // backward pole the flat reading runs away from the arc without bound —
    // the far-field residual, the one fingerprint the projection leaves

    // local flatness: an arc θ near the forward pole lands at height
    // tan(θ/2), so the calibrated flat reading 2·s matches the arc through
    // second order — 2·tan(θ/2) = θ + θ³/12 + … the observer's nearby
    // experiments read flat because the sphere's first signature is cubic
    for divisor in [32.0, 64.0, 128.0] {
        let a = Angle::new(1.0, divisor); // arc π/divisor from the forward pole
        let arc = a.grade_angle();
        let flat = 2.0 * crossing_height(a);
        assert!(
            (flat - arc).abs() < arc * arc * arc,
            "the flat ruler disagrees with the arc only at third order"
        );
    }

    // the radial law at any separation, no limit taken: two directions on one
    // meridian land at s₁ and s₂, and their flat separation is the arc's own
    // crossing height scaled by the conformal coupling —
    //   s₂ − s₁ = h(Δ)·(1 + s₁s₂)
    // where h(Δ) = tan(Δ/2) is the separation angle's circle→line height. as
    // Δ shrinks this reads ds/dθ = (1 + s²)/2, the conformal factor of
    // stereographic projection; at the origin the coupling is 1 and the flat
    // ruler is calibrated. the sphere's whole curvature lives in this one
    // magnitude law — the flattened angle's law, written out
    let bases = [
        Angle::new(1.0, 6.0), // π/6
        Angle::new(1.0, 3.0), // π/3
        Angle::new(2.0, 5.0), // 2π/5
    ];
    let steps = [Angle::new(1.0, 12.0), Angle::new(1.0, 5.0)];
    for a in bases {
        for d in steps {
            let s1 = crossing_height(a);
            let s2 = crossing_height(a + d);
            let coupling = 1.0 + s1 * s2;
            assert!(
                drawn(s2 - s1).near(&drawn(crossing_height(d) * coupling)),
                "flat separation = arc half-tangent × (1 + s₁s₂)"
            );
        }
    }

    // the far field: toward the backward pole the coupling compounds and the
    // flat reading runs away from the arc — the arc approaches π while the
    // landing height grows without bound (the pole itself never lands,
    // it_fixes_exactly_one_dual_pair). an observer holding the flat ruler to
    // the far field reads a magnitude anomaly: not new content in the data,
    // the radial law of the projection they are standing in
    let near_pole = Angle::new(49.0, 50.0); // 49π/50
    let arc = near_pole.grade_angle();
    let flat = 2.0 * crossing_height(near_pole);
    assert!(
        flat > 10.0 * arc,
        "toward the backward pole the flat reading runs away from the arc"
    );
}

#[test]
fn it_outputs_the_null_pair_cga_inputs() {
    // conformal geometric algebra adjoins two directions e₊² = +1, e₋² = −1,
    // spans their plane, and takes the null pair n₀ = (e₋ − e₊)/2 and
    // n∞ = e₋ + e₊ as INPUT — basis structure fixed before any point exists.
    // the drawing runs the other way: the pair is OUTPUT, the two
    // degeneracies of a chosen pole ray (it_fixes_exactly_one_dual_pair).
    // this foil runs cga's own arithmetic in (e₁, e₊, e₋) components under
    // the (+, +, −) signature and watches each adjoined piece reduce to a
    // suite fact

    // the adjoined "null" is a cancellation event, not a new kind of vector:
    // e₊² lands grade 0, e₋² lands grade 2 — the sign of a square is a
    // position — and n∞² = e₊² + e₋² is [1, 0] + [1, π] = 0, the additive
    // null of spacetime_test
    let e_plus_sq = Geonum::new(1.0, 0.0, 1.0);
    let e_minus_sq = Geonum::new(1.0, 1.0, 1.0);
    assert!(
        (e_plus_sq + e_minus_sq).near_mag(0.0),
        "the null basis squares to a cancellation, not to a new zero"
    );

    // cga component arithmetic: the signature inner product, the null basis,
    // and the embedding P(s) = n₀ + s·e₁ + ½s²·n∞ onto the cone the plane
    // makes possible
    let sig_dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] - a[2] * b[2];
    let n0 = [0.0, -0.5, 0.5];
    let n_inf = [0.0, 1.0, 1.0];
    let embed = |s: f64| [s, -0.5 + 0.5 * s * s, 0.5 + 0.5 * s * s];

    assert!(
        drawn(sig_dot(n0, n0)).near_mag(0.0),
        "n₀ is null by signature"
    );
    assert!(
        drawn(sig_dot(n_inf, n_inf)).near_mag(0.0),
        "n∞ is null by signature"
    );
    assert!(
        drawn(sig_dot(n0, n_inf)).near(&drawn(-1.0)),
        "n₀·n∞ = −1, the pair's coupling"
    );

    // every embedded landing is null — cone membership the representation
    // carries point by point — and the inner product stores the SQUARE of
    // the flat separation the radial law computes rationally:
    // P(s)·P(t) = −½(s−t)² = −½·h(Δ)²·(1 + st)²
    let bases = [
        Angle::new(1.0, 6.0),
        Angle::new(1.0, 3.0),
        Angle::new(2.0, 5.0),
    ];
    let step = Angle::new(1.0, 12.0);
    for a in bases {
        let (s, t) = (crossing_height(a), crossing_height(a + step));
        let h = crossing_height(step);
        assert!(
            drawn(sig_dot(embed(s), embed(s))).near_mag(0.0),
            "every embedded point sits on the cone"
        );
        assert!(
            drawn(sig_dot(embed(s), embed(t))).near(&drawn(-0.5 * (h * (1.0 + s * t)).powi(2))),
            "the embedding's inner product is the radial law, squared"
        );
    }

    // cga's dilation is a linear map of the adjoined plane — n₀ → k·n₀,
    // n∞ → n∞/k, e₁ fixed: the reciprocal scaling the account calls a
    // rotation in e₊∧e₋. track the (n₀, e₁, n∞) weights, apply the map,
    // dehomogenize by the n₀ weight — the landing that comes back is s/k,
    // the one division Angle::boost already is
    let k = 0.6_f64.exp();
    for a in [Angle::new(1.0, 5.0), Angle::new(3.0, 4.0)] {
        let s = crossing_height(a);
        let weights = [1.0, s, 0.5 * s * s]; // (n₀, e₁, n∞) weights of P(s)
        let dilated = [k * weights[0], weights[1], weights[2] / k];
        // the cone condition w₁² = 2·w₀·w∞ survives the reciprocal scaling
        assert!(
            drawn(dilated[1] * dilated[1] - 2.0 * dilated[0] * dilated[2]).near_mag(0.0),
            "the reciprocal scaling holds the cone"
        );
        let landed = dilated[1] / dilated[0]; // dehomogenize
        assert!(
            drawn(landed).near(&drawn(crossing_height(a.boost(k)))),
            "three weights and a dehomogenization compute boost's one division"
        );
    }

    // the map's fixed rays are n₀ and n∞ themselves — the pair cga adjoined.
    // dehomogenized, n₀ sits at landing 0, the forward pole's crossing
    // height; n∞ has no n₀ weight to divide by — the backward pole, whose
    // pole ray never crosses: its run 1 + cos θ vanishes. the plane's two
    // basis nulls are the drawing's two degeneracies, adopted as axioms
    let forward = Angle::new(0.0, 1.0);
    assert!(
        drawn(crossing_height(forward)).near_mag(0.0),
        "n₀ dehomogenizes to the forward pole's landing"
    );
    let (x, _) = forward.dual().cos_sin();
    assert!(
        drawn(1.0 + x).near_mag(0.0),
        "n∞ is the dual pole: the pole ray's run vanishes, no landing"
    );
}

// the height where the ray from the backward pole (−1, 0) through the
// direction's point (cos θ, sin θ) crosses the perpendicular diameter x = 0 —
// computed by drawing, from the cartesian point alone, never reading t
fn crossing_height(a: Angle) -> f64 {
    let (x, y) = a.cos_sin();
    // the ray runs 1 + x horizontally and rises y; it reaches the diameter
    // after 1/(1 + x) of its run, at height y/(1 + x)
    y / (1.0 + x)
}

// the drawn height as a geonum: a signed reading with the sign as a position,
// negative heights on the π ray
fn drawn(height: f64) -> Geonum {
    Geonum::scalar(height)
}

// a point on the traditional n-sphere as a chain of angles: each component is
// a cos peeled off the running product of sines — the decomposed
// representation the projection foil computes in
fn sphere_point(chain: &[Angle]) -> Vec<f64> {
    let mut comps = Vec::with_capacity(chain.len() + 1);
    let mut sines = 1.0;
    for a in chain {
        let (c, s) = a.cos_sin();
        comps.push(sines * c);
        sines *= s;
    }
    comps.push(sines);
    comps
}

// the pole ray at any count of components: from the backward pole
// (−1, 0, …, 0) through the point, landing on the hyperplane through the
// center — the circle→line drawing with more passengers aboard
fn project_from_pole(comps: &[f64]) -> Vec<f64> {
    comps[1..].iter().map(|c| c / (1.0 + comps[0])).collect()
}

// read the leading angle off a component list: the cos leg against the
// magnitude of everything deeper
fn leading_angle(comps: &[f64]) -> Angle {
    let rest = comps[1..].iter().map(|c| c * c).sum::<f64>().sqrt();
    Angle::new_from_cartesian(comps[0], rest)
}
