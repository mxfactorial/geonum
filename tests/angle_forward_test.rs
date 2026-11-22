// angles only go forward
//
// a geometric number is a magnitude and an angle. multiplying multiplies the magnitudes
// and adds the angles, and that is complete as it stands: nothing in angle space
// subtracts
//
//   a negative        a half turn on
//   a - b             a + b a half turn on
//   a quarter turn back    three quarter turns on
//   an inverse        the magnitude turned over, a half turn on
//   a turn undone     a whole turn on, the same grade and the same readings
//
// subtraction appears when a number is projected. the methods that take one angle from
// another are the ones that read a number against another: dot, wedge, geo, project,
// reflect, distance_to. the minus sign is in the reading, where a half turn projects
//
// blades keep every turn made. grade, blade % 4, is the geometry, and base_angle() drops
// the turns when the history is not wanted
//
// this suite holds that line until a use case shows the semiring breaks the geometry.
// until then it runs defense against a symbol that only appears after projection
//
// run: cargo test --test angle_forward_test

use geonum::*;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_semiring() {
    // a semiring adds and multiplies and never subtracts. a geonum adds in its angle and
    // multiplies in its magnitude
    let a = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
    let b = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]
    assert!((a * b).near(&Geonum::new(6.0, 1.0, 2.0)), "[6, π/2]");

    // each has one value that changes nothing: an angle of 0 added, a magnitude of 1
    // multiplied in
    assert!((Geonum::scalar(1.0) * a).near(&a));
    assert!(a.rotate(Angle::new(0.0, 1.0)).near(&a));

    // no operation hands back a minus sign. a scale of −2 comes back as a magnitude of 6 a
    // half turn on
    assert!(
        b.scale(-2.0).near(&Geonum::new(6.0, 7.0, 6.0)),
        "[6, π/6 + π]"
    );

    // and every operation that stays in angle space adds blades. none takes one away
    let blade = a.angle.blade();
    assert_eq!(a.differentiate().angle.blade(), blade + 1);
    assert_eq!(a.negate().angle.blade(), blade + 2);
    assert_eq!(a.dual().angle.blade(), blade + 2);
    assert_eq!(a.inv().angle.blade(), blade + 2);
    assert_eq!(a.integrate().angle.blade(), blade + 3);
    assert_eq!(a.decrement_blade().angle.blade(), blade + 3);
    assert_eq!(a.pow(3.0).angle.blade(), blade + 2, "three times π/3 is π");

    // seven multiplies by [1, 3π/2] are 21 quarter turns, all of them kept
    let mut turned = Geonum::scalar(1.0);
    for _ in 0..7 {
        turned = turned * Geonum::new(1.0, 3.0, 2.0);
    }
    assert_eq!(turned.angle.blade(), 21);
}

#[test]
fn it_negates_by_adding_a_half_turn() {
    // a negative is a number a half turn on: two blades
    let a = Geonum::new(2.0, 1.0, 3.0);
    assert!(a.negate().near(&Geonum::new(2.0, 4.0, 3.0)), "[2, π/3 + π]");
    assert!(a.negate().near(&a.dual()));

    // so a - b is a + b a half turn on. 5 − 3 is [2, 0]
    let (five, three) = (Geonum::scalar(5.0), Geonum::scalar(3.0));
    assert!((five - three).near(&(five + three.negate())));
    assert!((five - three).near(&Geonum::scalar(2.0)));

    // and 3 − 5 is [2, π]: a magnitude of 2, a half turn on. the −2 is what it reads on
    // dimension 0
    let below = three - five;
    assert!(below.near(&Geonum::new(2.0, 1.0, 1.0)));
    assert!((below.project_to_dimension(0) + 2.0).abs() < EPSILON);

    // two numbers two blades apart are each the others negative, at any blade. neither
    // one is the negative one
    let at_460 = Geonum::new_with_blade(3.0, 460, 0.0, 1.0);
    let at_462 = Geonum::new_with_blade(3.0, 462, 0.0, 1.0);
    assert!((at_460.project_to_dimension(0) - 3.0).abs() < EPSILON);
    assert!((at_462.project_to_dimension(0) + 3.0).abs() < EPSILON);
    assert!((at_460 + at_462).near_mag(0.0));

    // negated, each has the grade of the other
    assert!(at_460.negate().near(&at_462));
    assert_eq!(at_462.negate().angle.grade(), at_460.angle.grade());

    // and each is two blades on from the other, whichever way round it is asked
    assert_eq!((at_462.angle - at_460.angle).blade(), 2);
    assert_eq!((at_460.angle - at_462.angle).blade(), 2);
}

#[test]
fn it_goes_back_by_going_on() {
    // a quarter turn back is three quarter turns on
    let a = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
    assert!(
        a.integrate().near(&Geonum::new(2.0, 11.0, 6.0)),
        "[2, π/3 + 3π/2]"
    );
    assert!(a.decrement_blade().near(&a.integrate()));

    // an angle made a quarter turn back is made three on
    assert!(Angle::new(-1.0, 2.0).near(&Angle::new(3.0, 2.0)));

    // from blade 5, one blade down is blade 8: three on, at the grade one below
    let at_5 = Geonum::new_with_blade(2.0, 5, 1.0, 3.0);
    assert_eq!(at_5.decrement_blade().angle.blade(), 8);
    assert_eq!(at_5.decrement_blade().angle.grade(), 0);

    // a quarter turn undone is a whole turn made: four blades on, the same grade, and the
    // same reading on every dimension
    let round_trip = a.differentiate().integrate();
    assert_eq!(round_trip.angle.blade(), a.angle.blade() + 4);
    assert_eq!(round_trip.angle.grade(), a.angle.grade());
    for dimension in 0..4 {
        let (there, here) = (
            round_trip.project_to_dimension(dimension),
            a.project_to_dimension(dimension),
        );
        assert!((there - here).abs() < EPSILON);
    }

    // an inverse turns the magnitude over and goes a half turn on: [1/2, π/3 + π]
    assert!(a.inv().near(&Geonum::new(0.5, 4.0, 3.0)));

    // twice, and the magnitude is back, a whole turn on
    assert!(
        a.inv().inv().near(&Geonum::new(2.0, 7.0, 3.0)),
        "[2, π/3 + 2π]"
    );

    // so a number times its inverse has magnitude 1 and keeps going: [1, 2θ + π]
    assert!((a * a.inv()).near(&Geonum::new(1.0, 5.0, 3.0)), "[1, 5π/3]");

    // and / is a multiply by the inverse. the angles add, with the half turn
    let b = Geonum::new(3.0, 1.0, 6.0); // [3, π/6]
    assert!((a / b).near(&(a * b.inv())));
    assert!(
        (a / b).near(&Geonum::new(2.0 / 3.0, 3.0, 2.0)),
        "[2/3, π/3 + π/6 + π]"
    );
}

#[test]
fn it_keeps_every_turn_in_the_blade() {
    // [3, 2π] reads the same 3 as [3, 0] and is a different number: it has made a turn
    let turned = Geonum::new(3.0, 2.0, 1.0);
    let unturned = Geonum::scalar(3.0);
    assert!((turned.project_to_dimension(0) - unturned.project_to_dimension(0)).abs() < EPSILON);
    assert_eq!(turned.angle.blade(), 4);
    assert_eq!(unturned.angle.blade(), 0);

    // the grade is the geometry, and it is the same
    assert_eq!(turned.angle.grade(), unturned.angle.grade());

    // base_angle() drops the turns and keeps the grade
    assert!(turned.base_angle().near(&unturned));

    // at blade 1000 it is the same: grade 0, and blade 0 once the turns are dropped
    let far = Geonum::new_with_blade(3.0, 1000, 0.0, 1.0);
    assert_eq!(far.angle.grade(), 0);
    assert!(far.base_angle().near(&unturned));
}

#[test]
fn it_subtracts_only_to_project() {
    // reading one number against another is where one angle is taken from another. the dot
    // of [2, π/3] and [3, π/6] is 6 cos(π/6), 5.1962
    let a = Geonum::new(2.0, 1.0, 3.0);
    let b = Geonum::new(3.0, 1.0, 6.0);
    assert!((a.dot(&b).mag - 5.1962).abs() < 1e-4);

    // it reads the gap between the two angles and nothing else. turn both by 3π/4 and it
    // is the same reading
    let turn = Angle::new(3.0, 4.0);
    assert!(a.rotate(turn).dot(&b.rotate(turn)).near(&a.dot(&b)));

    // put both at blade 400 and it is the same reading, with the blades gone from it
    let far_a = Geonum::new_with_blade(2.0, 400, 1.0, 3.0);
    let far_b = Geonum::new_with_blade(3.0, 400, 1.0, 6.0);
    assert!(far_a.dot(&far_b).near(&a.dot(&b)));
    assert_eq!(far_a.dot(&far_b).angle.blade(), 0);

    // the minus sign belongs to the reading. [3, π] has a magnitude of 3 and reads −3 on
    // dimension 0
    let half_turn_on = Geonum::new(3.0, 1.0, 1.0);
    assert!(half_turn_on.near_mag(3.0));
    assert!((half_turn_on.project_to_dimension(0) + 3.0).abs() < EPSILON);

    // and when the reading is handed back as a geonum the sign is a half turn again: the
    // dot of [2, 0] and [3, π] is [6, π]
    let opposed = Geonum::scalar(2.0).dot(&half_turn_on);
    assert!(opposed.near(&Geonum::new(6.0, 1.0, 1.0)));

    // the diff of two angles is itself an angle, so it goes forward too. π/12 less π/6
    // is 23π/12
    let diff = Angle::new(1.0, 12.0) - Angle::new(1.0, 6.0);
    assert!(diff.near(&Angle::new(23.0, 12.0)));
}

#[test]
fn it_measures_a_turn_from_the_blades_it_was_given() {
    // to measure a turn, give the number its blades before asking. three numbers at blades
    // 6, 8 and 10
    let lower = Geonum::new_with_blade(3.0, 6, 0.0, 1.0);
    let start = Geonum::new_with_blade(3.0, 8, 0.0, 1.0);
    let upper = Geonum::new_with_blade(3.0, 10, 0.0, 1.0);

    // the lower and the upper are both a half turn from the start. they have the same
    // grade and the same reading, −3 where the start reads 3
    assert_eq!(lower.angle.grade(), upper.angle.grade());
    assert!((lower.project_to_dimension(0) + 3.0).abs() < EPSILON);
    assert!((upper.project_to_dimension(0) + 3.0).abs() < EPSILON);
    assert!((start.project_to_dimension(0) - 3.0).abs() < EPSILON);

    // the blades say which is which. the start is two on from the lower, and the upper is
    // two on from the start
    assert!(lower.dual().near(&start));
    assert!(start.dual().near(&upper));
    assert_eq!((start.angle - lower.angle).blade(), 2);
    assert_eq!((upper.angle - start.angle).blade(), 2);

    // a number given no blades has nowhere below to be measured from. one blade down from
    // blade 0 is blade 3
    assert_eq!(Geonum::scalar(3.0).decrement_blade().angle.blade(), 3);
}

#[test]
fn it_keeps_the_minus_sign_out_of_angle_space() {
    // the minus sign is a symbol of the projected side. no operation hands one back: every
    // magnitude that comes out of angle space is 0 or more, and whatever a projection would
    // call negative is a half turn in the angle
    //
    // six numbers across the grades, one of them at blade 460, taken two at a time through
    // every operation on a geonum
    let numbers = [
        Geonum::new(2.0, 1.0, 3.0),
        Geonum::new(3.0, 5.0, 4.0),
        Geonum::new_with_blade(0.5, 7, 1.0, 8.0),
        Geonum::new_with_blade(4.0, 460, 0.0, 1.0),
        Geonum::new(1.5, 3.0, 2.0),
        Geonum::scalar(1.0),
    ];
    let center = Geonum::new(0.25, 2.0, 7.0);

    for a in &numbers {
        for b in &numbers {
            let made = [
                // the ones that stay in angle space
                ("*", *a * *b),
                ("/", *a / *b),
                ("+", *a + *b),
                ("-", *a - *b),
                ("negate", a.negate()),
                ("inv", a.inv()),
                ("dual", a.dual()),
                ("undual", a.undual()),
                ("differentiate", a.differentiate()),
                ("integrate", a.integrate()),
                ("rotate", a.rotate(b.angle)),
                ("scale_rotate", a.scale_rotate(b.mag, b.angle)),
                ("scale by −2.5", a.scale(-2.5)),
                ("scale by 2.5", a.scale(2.5)),
                ("pow 3", a.pow(3.0)),
                ("pow 0.5", a.pow(0.5)),
                ("pow −2", a.pow(-2.0)),
                ("powi 3", a.powi(3)),
                ("increment_blade", a.increment_blade()),
                ("decrement_blade", a.decrement_blade()),
                ("base_angle", a.base_angle()),
                ("normalize", a.normalize()),
                ("copy_blade", a.copy_blade(b)),
                ("boost", a.boost(b.angle, 2.0)),
                // the ones that read one number against another and hand a geonum back
                ("dot", a.dot(b)),
                ("wedge", a.wedge(b)),
                ("geo", a.geo(b)),
                ("project", a.project(b)),
                ("reject", a.reject(b)),
                ("reflect", a.reflect(b)),
                ("meet", a.meet(b)),
                ("distance_to", a.distance_to(b)),
                ("project_to_angle", a.project_to_angle(b.angle)),
                ("adj", a.adj()),
                ("opp", a.opp()),
                ("invert_circle", a.invert_circle(&center, 2.0)),
            ];
            for (operation, result) in made {
                assert!(result.mag >= 0.0, "{operation} handed back a minus sign");
            }
        }
    }
}

#[test]
fn it_lifts_a_negative_at_the_door() {
    // a negative handed in from the projected side is stored as a magnitude a half turn on
    assert!(
        Geonum::scalar(-3.0).near(&Geonum::new(3.0, 1.0, 1.0)),
        "[3, π]"
    );
    assert!(Geonum::new_from_cartesian(-3.0, 0.0).near(&Geonum::new(3.0, 1.0, 1.0)));

    // the trig readings come back the same way. cos π is −1 to a projection, and here it
    // is a magnitude of 1 a half turn on. sin 3π/2 is a magnitude of 1 three quarter turns on
    assert!(Geonum::cos(Angle::new(1.0, 1.0)).near(&Geonum::new(1.0, 1.0, 1.0)));
    assert!(Geonum::sin(Angle::new(3.0, 2.0)).near(&Geonum::new(1.0, 3.0, 2.0)));

    // four constructors take a magnitude as it is given and do not lift it: new,
    // new_with_angle, new_with_blade and create_dimension. this suite does not assert on
    // what they do with a negative
}

#[test]
fn it_finds_the_same_place_turning_back_or_on() {
    // a quarter turn to the power −0.5 turns back half of one. to the power 3.5 it turns on
    // three and a half. the two are the same number, the complex i^(−1/2), which reads
    // 0.7071 and −0.7071. this is turning itself, on any circle
    let quarter_turn = Geonum::new(1.0, 1.0, 2.0);
    let back_half = quarter_turn.pow(-0.5);
    let on_three_and_a_half = quarter_turn.pow(3.5);
    assert!(back_half
        .base_angle()
        .near(&on_three_and_a_half.base_angle()));
    assert!(
        on_three_and_a_half.near(&Geonum::new(1.0, 7.0, 4.0)),
        "[1, 7π/4]"
    );
    let reads = std::f64::consts::FRAC_1_SQRT_2; // 0.7071
    assert!((back_half.project_to_dimension(0) - reads).abs() < EPSILON);
    assert!((back_half.project_to_dimension(1) + reads).abs() < EPSILON);

    // so two finished angles do not say which way the turn between them went. Angle - Angle
    // answers how far on from one to the other. take a = [3, π/6] to a power, turn a a
    // quarter turn, take it to the power again, and diff the two angles: −2 reads as 2
    // quarter turns on, and −0.5 as 3.5 on
    let a = Geonum::new(3.0, 1.0, 6.0);
    for (power, on) in [
        (-2.0, 2.0),
        (-0.5, 3.5),
        (0.5, 0.5),
        (2.0, 2.0),
        (3.5, 3.5),
        (7.0, 7.0),
    ] {
        let diff = a.differentiate().pow(power).angle - a.pow(power).angle;
        assert!(diff.near(&Angle::new(on, 2.0)));
    }
}
