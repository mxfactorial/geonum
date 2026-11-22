use geonum::*;
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_shape_function() {
    // finite element shape functions are typically polynomial basis functions
    // in geometric numbers, we can represent them directly with angle operations

    // create a linear shape function as a geometric number
    // N(x) = 1-x in the range [0,1] represented as a geonum
    let shape_function = |x: f64| -> Geonum {
        Geonum::new(1.0 - x, 0.0, 1.0) // magnitude varies with position, angle 0
    };

    // test the shape function at a few points
    let n_at_0 = shape_function(0.0);
    let n_at_half = shape_function(0.5);
    let n_at_1 = shape_function(1.0);

    // test values match expected linear function
    assert!(n_at_0.near_mag(1.0));
    assert!(n_at_half.near_mag(0.5));
    assert!(n_at_1.near_mag(0.0));

    // create quadratic shape function N(x) = 4x(1-x) as a geonum
    // this is the standard quadratic basis function on [0,1]
    let quadratic_shape = |x: f64| -> Geonum {
        let value = 4.0 * x * (1.0 - x);
        Geonum::new(value, 0.0, 1.0) // phase still constant for this function
    };

    // test quadratic function at critical points
    let q_at_0 = quadratic_shape(0.0);
    let q_at_half = quadratic_shape(0.5);
    let q_at_1 = quadratic_shape(1.0);

    // test values match expected quadratic function
    assert!(q_at_0.near_mag(0.0));
    assert!(q_at_half.near_mag(1.0));
    assert!(q_at_1.near_mag(0.0));

    // demonstrate shape function derivatives using angle rotation
    // for a function f(x), f'(x) is the quarter turn of f, counted once for each turn of x it
    // carries, over the quarter turn of x
    // demonstrate this with our linear shape function N(x) = 1 - x: the 1 carries no angle and
    // leaves nothing, the x carries itself once and leaves 1, and the minus sign is a half
    // turn applied after
    let shape_derivative = |x: f64| -> Geonum {
        let x = Geonum::new(x, 1.0, 12.0); // [x, π/12], given an angle so its count can be proven
        turn_and_divide(&x, &x, 1.0).negate() // derivative of 1-x is constant -1 (magnitude 1, angle PI)
    };

    // test the derivative at a point
    let deriv_at_half = shape_derivative(0.5);
    assert!(deriv_at_half.near(&Geonum::new(1.0, 1.0, 1.0))); // [1, π]

    // demonstrate ability to represent high-order shape functions
    // using a composition of simple angle operations
    // for a cubic function, we need only 2 components regardless of dimension

    // simulate a traditional 3D high-order element computation
    // instead of 3D tensors with O(n³) storage and computation
    // we use just 2 components with O(1) storage and computation

    // define cubic shape function N(x) = x²(3-2x) as a geonum
    let cubic_shape = |x: f64| -> Geonum {
        let value = x * x * (3.0 - 2.0 * x);
        Geonum::new(value, 0.0, 1.0)
    };

    // test cubic function at critical points
    let c_at_0 = cubic_shape(0.0);
    let c_at_half = cubic_shape(0.5);
    let c_at_1 = cubic_shape(1.0);

    // test values match expected cubic function
    assert!(c_at_0.near_mag(0.0));
    assert!(c_at_half.near_mag(0.5));
    assert!(c_at_1.near_mag(1.0));

    // the quadratic and cubic shape functions get their derivatives the way the linear one
    // did, term by term: each power of x rotated a quarter turn, counted once for each
    // angle of x it carries, over the quarter turn of x. coefficients scale after, and a
    // minus sign is a half turn applied after
    let rate_of_power = |x: f64, n: f64| -> Geonum {
        let x = Geonum::new(x, 1.0, 12.0); // [x, π/12]
        Geonum::scalar(turn_and_divide(&x.pow(n), &x, n).mag)
    };

    // N(x) = 4x(1-x) = 4x - 4x², so N'(x) = 4 - 8x
    let quadratic_derivative =
        |x: f64| rate_of_power(x, 1.0).scale(4.0) + rate_of_power(x, 2.0).scale(4.0).negate();
    assert!(quadratic_derivative(0.5).near_mag(0.0)); // flat at the peak
    assert!(quadratic_derivative(0.25).near_mag(2.0)); // rising
    let falling = quadratic_derivative(0.75); // -2, the minus sign a half turn
    assert!(falling.near_mag(2.0) && falling.angle.grade() == 2);

    // N(x) = x²(3-2x) = 3x² - 2x³, so N'(x) = 6x - 6x²
    let cubic_derivative =
        |x: f64| rate_of_power(x, 2.0).scale(3.0) + rate_of_power(x, 3.0).scale(2.0).negate();
    assert!(cubic_derivative(0.5).near_mag(1.5)); // steepest at the middle
    assert!(cubic_derivative(1.0).near_mag(0.0)); // flat where it reaches 1

    // measure performance with a simulated multi-dimensional element
    // this would normally require O(n³) operations in a traditional FEM code
    // but with geonums its O(1) regardless of element order or dimension

    // create a shape function evaluation for a high order in 3D
    let high_order_shape = |x: f64, y: f64, z: f64| -> Geonum {
        // this would be a massive tensor operation in traditional FEM
        // with geonum, its a simple angle-magnitude computation
        let r = (x * x + y * y + z * z).sqrt();
        let theta = y.atan2(x);
        let phi = if r > EPSILON { (z / r).acos() } else { 0.0 };

        // convert angle to Angle struct
        let angle_radians = theta * phi / TAU;
        Geonum::new(r * r * (3.0 - 2.0 * r), angle_radians / PI, 1.0)
    };

    // test the high order shape function
    let high_order_value = high_order_shape(0.5, 0.5, 0.5);

    // test high order shape function at center point (0.5, 0.5, 0.5)
    // r = sqrt(3/4) ≈ 0.866, cubic_r = r²(3-2r) ≈ 0.598
    let expected_r = (0.75_f64).sqrt();
    let expected_length = expected_r * expected_r * (3.0 - 2.0 * expected_r);
    assert!(high_order_value.near_mag(expected_length));

    // test angle computation: theta = atan2(0.5, 0.5) = π/4
    // phi = acos(0.5/0.866) ≈ 0.955 radians
    // angle is theta * phi / (2π) ≈ 0.1194 radians
    let expected_theta = 0.5_f64.atan2(0.5);
    let expected_phi = (0.5 / expected_r).acos();
    let expected_angle_value = expected_theta * expected_phi / TAU;
    assert!(high_order_value.angle.near_rem(expected_angle_value));
    assert_eq!(high_order_value.angle.blade(), 0); // small angle, so blade is 0
}

#[test]
fn its_a_stiffness_matrix() {
    // in FEM, the stiffness matrix represents the relationship between
    // nodal displacements and applied forces
    // traditionally it is formed entry by entry from products of cosines and sines, then
    // assembled and stored
    // with geonum nothing is formed or stored. a bar is one number and its stiffness is one
    // read: the displacement projected onto the bar and scaled

    // --- one spring is a projection onto the stretch axis ---

    // K = [k -k; -k k] for a spring element with stiffness k. here [3 −3; −3 3]
    let k = 3.0;

    // the two nodal displacements are one geonum, and the spring only feels the stretch:
    // the part of it along (1, -1), the line at 3π/4
    let stretch_axis = Geonum::new(1.0, 3.0, 4.0);
    let spring = |u: &Geonum| u.project(&stretch_axis).scale(2.0 * k);

    // the projection lands on what the matrix gives for each displacement
    for (u, force) in [
        ((0.5, 0.0), (1.5, -1.5)), // node 1 moves, node 2 held
        ((0.2, -0.7), (2.7, -2.7)),
        ((-0.4, 0.3), (-2.1, 2.1)),
    ] {
        let u = Geonum::new_from_cartesian(u.0, u.1);
        assert!(reads(&spring(&u), force));
    }

    // both nodes moving together is the direction a quarter turn from the stretch axis. the
    // projection has nothing of it, so there is no force. the matrix is singular because a
    // projection keeps one direction and drops the one a quarter turn on
    let together = Geonum::new(0.8, 1.0, 4.0);
    assert!(spring(&together).near_mag(0.0));

    // moving apart is along the axis and comes back scaled by 2k
    let apart = Geonum::new(0.8, 3.0, 4.0);
    assert!(spring(&apart).near_mag(2.0 * k * 0.8));

    // a fixed boundary condition, no displacement at either node, has zero force
    assert!(spring(&Geonum::scalar(0.0)).near_mag(0.0));

    // --- one bar in the plane is a projection onto the bar ---

    // traditional: the block k[c² cs; cs s²], three products of a cosine and a sine. for
    // k = 100 at π/3 it is [25 43.3013; 43.3013 75], and times (0.02, 0.01) it gives
    // (0.933013, 1.616025)
    let k = 100.0;
    let alpha = Angle::new(1.0, 3.0); // the bar sits at π/3
    let (c, s) = alpha.cos_sin();

    // geonum: project the displacement onto the bar and scale by k
    let bar = Geonum::new_with_angle(1.0, alpha);
    let moved = Geonum::new_from_cartesian(0.02, 0.01);
    assert!(reads(&moved.project(&bar).scale(k), (0.933013, 1.616025)));

    // the three matrix entries are not three facts. they are the bar squared, read on the
    // two axes: squaring doubles the angle, and c², cs, s² are ½(1 + cos 2α), ½ sin 2α,
    // ½(1 − cos 2α)
    let squared = bar * bar; // [1, 2α]
    let (cos_2a, sin_2a) = components(&squared);
    assert!((c * c - 0.5 * (1.0 + cos_2a)).abs() < EPSILON);
    assert!((c * s - 0.5 * sin_2a).abs() < EPSILON);
    assert!((s * s - 0.5 * (1.0 - cos_2a)).abs() < EPSILON);

    // --- it never rotates ---

    // a projection scales what it reads and turns nothing. so the force from one
    // displacement, read along a second, is the force from the second read along the first.
    // that is what the matrix being symmetric says
    let read_along = |force: &Geonum, d: &Geonum| force.mag * d.mag * force.angle.project(d.angle);
    let (u, v) = (Geonum::new(0.02, 1.0, 12.0), Geonum::new(0.03, 3.0, 5.0));
    let bar_force = |d: &Geonum| d.project(&bar).scale(k);
    let (u_along_v, v_along_u) = (
        read_along(&bar_force(&u), &v),
        read_along(&bar_force(&v), &u),
    );
    assert!((u_along_v - v_along_u).abs() < EPSILON);

    // both are k × the first read on the bar × the second read on the bar
    assert!((u_along_v - k * read_on(&u, &bar) * read_on(&v, &bar)).abs() < EPSILON);

    // --- where k comes from ---

    // a bar 2 long with EA = 210, pulled by 21 at its end. the same pull runs through all of
    // it, so its strain is one number, the pull over EA
    let (ea, length, end_load) = (210.0, 2.0, 21.0);
    let strain = Geonum::scalar(end_load / ea);

    // the end moves by the strain read back along the bar, at its two ends. no elements, and
    // no shape functions to fill in between nodes
    let read_back = |x: f64| turn_back_and_divide(&strain, &Geonum::new(x, 1.0, 12.0), 1.0).mag;
    let end_moved = read_back(length) - read_back(0.0);
    assert!((end_moved - 0.2).abs() < EPSILON, "21 × 2 / 210");

    // k is the pull over how far the end moved: EA/L, the 105 a textbook reaches by
    // integrating a product of two shape function rates over the element
    assert!((end_load / end_moved - 105.0).abs() < EPSILON);
}

#[test]
fn its_a_linear_solver() {
    // traditional FEM solvers assemble Ku = f and factor the matrix to solve it: unknown
    // components on fixed axes, and an equation for each
    // with geonum there are bars and loads, each a magnitude at an angle. where a joint goes
    // is one number, and every stretch and bar force is a reading of it

    // --- a joint is its bars added ---

    // two bars hold a free joint: 100 along 0 and 100 along π/3. each bar is one geonum, its
    // stiffness under a square root so that the bar squared carries k
    let bars = [Geonum::new(10.0, 0.0, 1.0), Geonum::new(10.0, 1.0, 3.0)];

    // the joint keeps two sums of its bars: half the stiffnesses, and half the bars squared
    let (scale, squares) = joint(&bars);
    assert!(
        (scale - 100.0).abs() < EPSILON,
        "half the stiffnesses summed"
    );
    assert!(squares.near(&Geonum::new(50.0, 1.0, 3.0)), "[50, π/3]");

    // squaring doubled the angles, so half the angle of the squares is the line the joint is
    // stiffest along: π/6, between the two bars, with no eigenvalue problem
    let stiffest = Geonum::new_with_angle(1.0, squares.angle * 0.5);
    assert!(stiffest.angle.near(&Angle::new(1.0, 6.0)));

    // move the joint 0.01 along that line and the bars pull straight back with 100 + 50
    // times the move. a quarter turn on they pull back with 100 − 50
    let across = stiffest.differentiate();
    let pulled = |u: &Geonum| pull(&bars[0], u) + pull(&bars[1], u);
    let (along_it, across_it) = (stiffest.scale(0.01), across.scale(0.01));
    assert!((pulled(&along_it) + along_it.scale(150.0)).near_mag(0.0));
    assert!((pulled(&across_it) + across_it.scale(50.0)).near_mag(0.0));

    // --- where it goes under a load ---

    // a load of 10 straight up. the textbook assembles [125 43.3013; 43.3013 75] and inverts
    // it for (−0.057735, 0.166667)
    let load = Geonum::new(10.0, 1.0, 2.0);

    // geonum: read the load along the stiffest line and a quarter turn on, and divide each
    // reading by what it meets there, 150 and 50. two reads and two divides
    let moved = give(&bars, &load);
    assert!(reads(&moved, (-0.057735, 0.166667)));

    // each bar pulls back along itself by its stiffness times the joints reading on it. the
    // load and the pulls, laid head to tail, close
    let closed = bars.iter().fold(load, |sum, bar| sum + pull(bar, &moved));
    assert!(closed.near_mag(0.0));

    // --- more bars than balance can settle ---

    // three bars hold one joint: 100 straight up, and 50 each along π/6 and 5π/6. the
    // textbook counts three unknown bar forces against two balance equations, calls the
    // joint indeterminate and adds an equation
    let bars = [
        Geonum::new(100.0_f64.sqrt(), 1.0, 2.0),
        Geonum::new(50.0_f64.sqrt(), 1.0, 6.0),
        Geonum::new(50.0_f64.sqrt(), 5.0, 6.0),
    ];
    let load = Geonum::new(10.0, 3.0, 2.0); // 10 straight down

    // the joint goes to one number by the same two reads and two divides
    let moved = give(&bars, &load);
    assert!(moved.near(&Geonum::new(0.08, 3.0, 2.0)), "[0.08, 3π/2]");

    // what each bar feels is that one number read on it: 0.08, 0.04 and 0.04, away from
    // where the bars are held
    let felt = bars.map(|bar| read_on(&moved, &bar));
    assert!((felt[0] + 0.08).abs() < EPSILON);
    assert!((felt[1] + 0.04).abs() < EPSILON);
    assert!((felt[2] + 0.04).abs() < EPSILON);

    // a number has two independent readings, so the third was never unknown. the equation
    // the textbook adds says so: the two outer bars add to the middle one, and their
    // readings add to its
    assert!((felt[0] - (felt[1] + felt[2])).abs() < EPSILON);

    // so the bars pull 8, 2 and 2, the textbooks answer, and with the load they close
    let pulls = bars.map(|bar| pull(&bar, &moved));
    assert!(pulls[0].near_mag(8.0) && pulls[1].near_mag(2.0) && pulls[2].near_mag(2.0));
    let closed = pulls.iter().fold(load, |sum, bar_pull| sum + *bar_pull);
    assert!(closed.near_mag(0.0));

    // --- two joints that share a bar ---

    // joint a is held by two bars, joint b by two more, and a fifth bar of stiffness 200
    // runs from a to b. the textbook assembles a 4×4 table for four unknown components and
    // eliminates
    let at_a = [
        Geonum::new(100.0_f64.sqrt(), 3.0, 4.0),
        Geonum::new(150.0_f64.sqrt(), 1.0, 2.0),
    ];
    let at_b = [
        Geonum::new(100.0_f64.sqrt(), 1.0, 4.0),
        Geonum::new(150.0_f64.sqrt(), 1.0, 2.0),
    ];
    let shared = Geonum::new(200.0_f64.sqrt(), 0.0, 1.0);
    let line = Geonum::new_with_angle(1.0, shared.angle); // from a to b
    let (load_a, load_b) = (Geonum::new(10.0, 3.0, 2.0), Geonum::new(20.0, 5.0, 3.0));

    // with the shared bar gone each joint goes where its own load takes it. read along the
    // bar, that leaves them 0.448803 further apart
    let apart = read_on(&give(&at_b, &load_b), &line) - read_on(&give(&at_a, &load_a), &line);
    assert!((apart - 0.448803).abs() < 1e-6);

    // a push of 1 along the bar moves each joint 0.026667 along it, and stretches the bar
    // itself by 1/200
    let give_a = read_on(&give(&at_a, &line), &line);
    let give_b = read_on(&give(&at_b, &line), &line);
    let bar_gives = 1.0 / (shared.mag * shared.mag);
    assert!((give_a - 0.026667).abs() < 1e-6 && (give_b - 0.026667).abs() < 1e-6);

    // the bar and the two joints it joins give one after the other, so what they give adds,
    // and the force in the bar is one divide
    let force = apart / (bar_gives + give_a + give_b);
    assert!(
        (force - 7.693772).abs() < 1e-6,
        "the textbooks number from its 4×4 table"
    );

    // the bar pulls a toward b and b toward a, a half turn apart. each joint goes where its
    // load and that pull take it, and lands on the textbooks displacements
    let on_a = load_a + line.scale(force);
    let on_b = load_b + line.scale(force).negate();
    let (moved_a, moved_b) = (give(&at_a, &on_a), give(&at_b, &on_b));
    assert!(reads(&moved_a, (0.138501, -0.015375)));
    assert!(reads(&moved_b, (0.176969, -0.130845)));

    // the shared bar is stretched by what it carries
    let stretched = read_on(&moved_b, &line) - read_on(&moved_a, &line);
    assert!((stretched - force * bar_gives).abs() < EPSILON);

    // and head to tail, both joints close
    let closed_a = at_a.iter().fold(on_a, |sum, bar| sum + pull(bar, &moved_a));
    let closed_b = at_b.iter().fold(on_b, |sum, bar| sum + pull(bar, &moved_b));
    assert!(closed_a.near_mag(0.0) && closed_b.near_mag(0.0));

    // --- when there is nothing to divide by ---

    // two bars along the same line. their squares point the same way, so the summed squares
    // are as large as the scale, and a quarter turn from the bars the joint meets scale − size,
    // nothing: a mechanism
    let lined_up = [Geonum::new(10.0, 1.0, 5.0), Geonum::new(7.0, 6.0, 5.0)]; // π/5 and π/5 + π
    let (scale, squares) = joint(&lined_up);
    assert!((scale - squares.mag).abs() < EPSILON, "scale − size = 0");

    // a push a quarter turn from the bars reads nothing on either, so neither pulls back
    let sideways = Geonum::new(0.01, 1.0, 5.0).differentiate();
    assert!(lined_up
        .iter()
        .all(|bar| pull(bar, &sideways).near_mag(0.0)));
}

#[test]
fn it_collapses_steps() {
    // traditional FEM workflow involves discrete steps:
    // 1. Mesh generation
    // 2. Matrix assembly
    // 3. Solver step
    // 4. Post-processing
    //
    // the mesh comes first: it keeps a body as scalars at nodes, and the other three steps
    // are spent getting back what that dropped
    // with geonum the bar is not cut up. there is nothing to assemble or solve, and an
    // answer is read where it is wanted: O(1), with no n to grow

    // a bar fixed to a wall, 3 long, EA = 210, carrying 10 for each unit of its length
    let (ea, length, q) = (210.0, 3.0, 10.0);
    let at = |x: f64| Geonum::new(x, 1.0, 12.0); // [x, π/12], a length along the bar

    // --- no mesh, no assembly, no solve ---

    // the force in the bar at x is the load beyond x: the load read back at the two ends
    let load_up_to = |x: f64| turn_back_and_divide(&Geonum::scalar(q), &at(x), 1.0).mag;
    let force = |x: f64| load_up_to(length) - load_up_to(x);
    assert!(
        (force(0.0) - 30.0).abs() < EPSILON,
        "the wall holds all of it: qL"
    );
    assert!(force(length).abs() < EPSILON, "the free end holds none");

    // the strain is the force over EA. the displacement at x is the strain read back from
    // the wall to x, term by term: the whole load, less the load behind x
    let strain = |x: f64| force(x) / ea;
    let strain_up_to = |x: f64| {
        let whole_load = turn_back_and_divide(&Geonum::scalar(q * length), &at(x), 1.0).mag; // qL × x
        let behind = turn_back_and_divide(&at(x).scale(q), &at(x), 2.0).mag; // q x² / 2
        (whole_load - behind) / ea
    };
    let displacement = |x: f64| strain_up_to(x) - strain_up_to(0.0);

    // --- post-processing is the read ---

    // the textbooks closed forms, typed: qL/EA at the wall and qL²/2EA at the tip
    assert!((strain(0.0) - 0.142857).abs() < 1e-6);
    assert!((displacement(length) - 0.214286).abs() < 1e-6);

    // and anywhere between, with nothing stored at nodes to fill in from
    assert!((displacement(length / 2.0) - 0.160714).abs() < 1e-6);

    // --- the mesh, beside it ---

    // the textbook route cuts the bar into elements, assembles them and solves. typed from
    // that route, the strain it reports in the element at the wall
    for (elements, at_the_wall) in [(4.0, 0.125), (40.0, 0.141071), (4000.0, 0.142839)] {
        // an element keeps one strain for its whole length, so at the wall it is short by
        // the load on half an element, over EA
        let short = q * (length / elements) / 2.0 / ea;
        assert!((strain(0.0) - at_the_wall - short).abs() < 1e-6);
    }

    // 4000 equations are still short by 0.000018. the mesh pays for every digit, O(n) on
    // this bar and up to O(n³) in a solid, and never arrives. the read is O(1): the bar has
    // a length, a stiffness and a load, and no n. n came in with the mesh
}

// ---------------------------------------------------------------------------
// helpers — the readings the tests compare with textbook numbers, what a joint keeps of
// its bars, and the rate and the read with no strips that the calculus suites use
// ---------------------------------------------------------------------------

// the two readings of a geonum on the fixed axes
fn components(g: &Geonum) -> (f64, f64) {
    (g.project_to_dimension(0), g.project_to_dimension(1))
}

// a geonum read on the two axes, against a textbook pair typed to six places
fn reads(g: &Geonum, textbook: (f64, f64)) -> bool {
    let (x, y) = components(g);
    (x - textbook.0).abs() < 1e-6 && (y - textbook.1).abs() < 1e-6
}

// one number read on anothers line
fn read_on(u: &Geonum, line: &Geonum) -> f64 {
    u.mag * u.angle.project(line.angle)
}

// what a joint keeps of its bars: half the sum of their stiffnesses, and half the sum of
// the bars squared. a bar is [√k, α], so its square is [k, 2α]
fn joint(bars: &[Geonum]) -> (f64, Geonum) {
    let scale = bars.iter().map(|bar| bar.mag * bar.mag).sum::<f64>() / 2.0;
    let squares = bars
        .iter()
        .map(|bar| *bar * *bar)
        .reduce(|sum, square| sum + square)
        .map(|sum| sum.scale(0.5).base_angle())
        .unwrap_or(Geonum::scalar(0.0));
    (scale, squares)
}

// where a joint goes under a push: the push read along the line the joint is stiffest
// along and a quarter turn on, each reading over what it meets there. the stiffest line
// sits at half the angle of the summed squares and meets the scale plus their size. a
// quarter turn on meets the scale less their size
fn give(bars: &[Geonum], push: &Geonum) -> Geonum {
    let (scale, squares) = joint(bars);
    let stiffest = Geonum::new_with_angle(1.0, squares.angle * 0.5);
    let across = stiffest.differentiate();
    let along_it = stiffest.scale(read_on(push, &stiffest) / (scale + squares.mag));
    let across_it = across.scale(read_on(push, &across) / (scale - squares.mag));
    (along_it + across_it).base_angle()
}

// what a bar pulls its joint with once the joint has gone to u: along itself, by its
// stiffness times the reading of u on it, and back against that reading
fn pull(bar: &Geonum, u: &Geonum) -> Geonum {
    let along = Geonum::new_with_angle(1.0, bar.angle);
    along.scale(-bar.mag * bar.mag * read_on(u, &along))
}

// the rate of f against x: turn both a quarter turn, count, and divide. magnitudes divide,
// angles subtract, the quarter turns cancel. as differentiate_test has it
//
// the count is how many times x went into f. whoever made f has it and hands it over, and
// it is proven here in angle space: the angle of x taken that many times is the angle of
// f. angles add, and no angle is read out as a float to find it
fn turn_and_divide(f: &Geonum, x: &Geonum, count: f64) -> Geonum {
    assert!(
        f.angle.near(&(x.angle * count)),
        "x went into f {count} times"
    );
    let df = f.differentiate().scale(count);
    let dx = x.differentiate();
    Geonum::new_with_angle(df.mag / dx.mag, df.angle - dx.angle)
}

// the antiderivative of rate against x: turn_and_divide run backward. multiply by the
// quarter turn of x, rotate the other three quarter turns with integrate(), drop the full
// turn that makes with base_angle(), then divide by the count. as integral_test has it. read at two ends it is the
// integral, with no strips
//
// the count is how many times x has gone into the result, one more than into the rate. it
// is handed over and proven the same way, in angle space
fn turn_back_and_divide(rate: &Geonum, x: &Geonum, count: f64) -> Geonum {
    let around = (*rate * x.differentiate()).integrate().base_angle();
    assert!(
        around.angle.near(&(x.angle * count).base_angle()),
        "x went into the result {count} times"
    );
    around.scale(1.0 / count)
}
