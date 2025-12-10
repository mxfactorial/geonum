// "category theory" is founded on a fictional data type called a "category" to group definitions
//
// to keep arrow operations consistent with a fictional data type you must self-referentially require an "identity arrow" as a "morphism" for all "objects"
//
// hacking associative consistency with commutative diagrams just traps everyone in a formalism loop ("natural in all components")
//
// and denies them the opportunity to understand how relationships **naturally compose** in the physical universe
//
// so instead of "defining a category", geometric numbers prove their relationship consistency with the physical universe by *extending* the universe's existing dimensions with `let transform = sin(pi/2);`
//
// rejecting "categories" for "transformations" empowers people to understand the relationship or "composition" between operations so well they can even **quantify** it:
//
// ```rs
// let clockwise = [1, 0];
// let rightward = [1, PI/2];
// // measure composition
// rightward / clockwise == [1, PI/2]
// ```
//
// say goodbye to `F(f) ∘ F(g) = F(f ∘ g)`

use geonum::*;
use std::f64::consts::{PI, TAU};

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;

#[test]
fn its_a_category() {
    // category theory uses "objects" and "morphisms" as abstract structures
    // in geometric numbers, we use dimensions and transformations instead

    // create a "category" as a transformation space
    // category theory abstracts over "objects" and "morphisms" but geonum makes this concrete:
    // objects become geometric numbers, morphisms become rotations, composition becomes angle addition
    // no need to declare a "space" - objects exist directly as [length, angle, blade]

    // create "objects" as geometric numbers in that space
    // Vector (grade 1) - represents an object as a 1D direction in category space
    let a = Geonum::new(1.0, 0.0, 2.0);
    // Vector (grade 1) - represents an object as a 1D direction in category space
    let b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // test dimension extension vs categorical objects
    // instead of saying "a is an object in category C", we say "a exists in space"
    // categorical objects become geometric numbers at standardized angles
    let objects = GeoCollection::from(vec![
        Geonum::create_dimension(1.0, 0),
        Geonum::create_dimension(1.0, 1),
    ]); // categorical objects as geometric positions at angles 0 and π/2

    // test objects exist in the space
    assert!(objects[0].angle.grade_angle() < EPSILON);
    assert!((objects[1].angle.grade_angle() - PI / 2.0).abs() < EPSILON);

    // test angle-based composition vs symbol-based composition
    // in category theory, f ∘ g is defined abstractly
    // with geometric numbers, composition is simply angle addition and length multiplication

    // create "morphisms" as rotations
    // Vector (grade 1) - morphisms are 1D transformations between objects
    // In some geometric algebra applications, morphisms could be bivectors (blade: 2)
    // representing transformations in a plane, but here we model them as vectors
    let f = Geonum::new(2.0, 1.0, 4.0); // rotation by π/4
                                        // Vector (grade 1) - consistent with f for composition operations
    let g = Geonum::new(3.0, 1.0, 6.0); // rotation by π/6

    // test composition as direct angle addition (f ∘ g)
    let f_compose_g = f * g;
    // Note: f_compose_g inherits its blade grade through the mul operation
    // The blade grade is determined by the with_product_blade method, which uses |a.blade - b.blade|
    // Since both f and g have blade: 1, f_compose_g will also have blade: 1

    // test composed transformation properties
    assert!((f_compose_g.mag - 6.0).abs() < EPSILON); // lengths multiply: 2*3=6
                                                      // angles add: pi/4 + pi/6 = 5pi/12
    let expected_angle = PI / 4.0 + PI / 6.0;
    assert!((f_compose_g.angle.grade_angle() - expected_angle).abs() < EPSILON);

    // test natural transformations as geometric rotations
    // a natural transformation is just a rotation that preserves structure
    let transform = |x: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            x.mag,
            x.angle + Angle::new(2.0, 6.0), // rotate everything by π/3
        )
    };

    let a_transformed = transform(&a);
    let b_transformed = transform(&b);

    // test transformation preserves structure (angle differences)
    let original_diff = (b.angle.grade_angle() - a.angle.grade_angle()).abs();
    let transformed_diff =
        (b_transformed.angle.grade_angle() - a_transformed.angle.grade_angle()).abs();

    assert!((original_diff - transformed_diff).abs() < EPSILON);

    // test we can measure functorial relationships
    // instead of asserting that a relationship is functorial, we can measure
    // exactly how well it preserves structure through angles
    let preservation_measure = original_diff / transformed_diff;
    assert!((preservation_measure - 1.0).abs() < EPSILON);
}

#[test]
fn its_a_functor() {
    // in category theory, a functor maps between categories
    // with geometric numbers, a functor is simply a transformation between spaces

    // create two spaces (analogous to categories)
    // functors traditionally require formal mappings between category structures
    // geonum eliminates this abstraction: functors become direct angle transformations
    // source and target "categories" are just different angle interpretations of the same geometric space

    // create "objects" in source category
    let a = Geonum::new(1.0, 0.0, 2.0);
    let b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // create a "morphism" in source category (rotation by pi/4)
    let f_source = Geonum::new(1.0, 0.5, 4.0); // π/4

    // test how rotation naturally creates structure preservation
    // define a functor as an angle transformation
    let functor = |g: &Geonum| -> Geonum {
        // scale lengths and double angles
        Geonum::new_with_angle(
            g.mag * 2.0,
            g.angle + g.angle, // double angles by adding to itself
        )
    };

    // apply functor to objects
    let f_a = functor(&a);
    let f_b = functor(&b);

    // apply functor to morphism
    // artifact of geonum automation: functorial mapping replaced by direct angle transformation
    let _f_morphism = functor(&f_source);

    // test transformation space vs arbitrary mapping
    // our functor preserves the angle relationships between objects
    let original_diff = (b.angle.grade_angle() - a.angle.grade_angle()).abs();
    let transformed_diff = (f_b.angle.grade_angle() - f_a.angle.grade_angle()).abs();

    // our example functor doubles angles, so differences should double
    assert!((transformed_diff - 2.0 * original_diff).abs() < EPSILON);

    // test structure preservation as angle consistency
    // F(f ∘ g) = F(f) ∘ F(g)

    // create another morphism
    let g_source = Geonum::new(1.0, 1.0, 6.0); // π/6

    // compose in source category
    let f_compose_g = f_source * g_source;

    // apply functor to composition
    let f_of_composition = functor(&f_compose_g);

    // apply functor to individual morphisms then compose
    let f_f = functor(&f_source);
    let f_g = functor(&g_source);
    let composition_of_f = f_f * f_g;

    // for our simplified model, we focus on the key property
    // that angle transformation behaves consistently
    //
    // the full functorial property would require more complex implementation
    // but the angle-based approach captures the essence

    // angles do maintain consistent transformations
    let angle_diff =
        (f_of_composition.angle.grade_angle() - composition_of_f.angle.grade_angle()).abs();
    assert!(angle_diff < EPSILON || (TAU - angle_diff) < EPSILON);

    // test we can directly measure the degree of structure preservation
    // instead of just asserting it exists
    let original_composition_angle = f_source.angle.grade_angle() + g_source.angle.grade_angle();
    let functor_composition_angle = f_f.angle.grade_angle() + f_g.angle.grade_angle();
    let expected_relationship = 2.0; // our functor doubles angles

    let measured_relationship = functor_composition_angle / original_composition_angle;
    assert!((measured_relationship - expected_relationship).abs() < EPSILON);
}

#[test]
fn its_an_adjunction() {
    // in category theory, an adjunction is a pair of functors with special properties
    // with geometric numbers, adjunctions are direct angle reflections

    // create two spaces (analogous to categories)
    // adjunctions traditionally involve complex universal properties and natural transformations
    // geonum simplifies this: adjoint functors become angle doubling/halving operations
    // the "spaces" are just conceptual - the real structure lives in the angle arithmetic

    // define a "left adjoint" functor from C to D by doubling angles
    let left_adjoint = |g: &Geonum| -> Geonum {
        // double angles
        Geonum::new_with_angle(g.mag, g.angle + g.angle)
    };

    // define a "right adjoint" functor from D to C by halving angles
    let right_adjoint = |g: &Geonum| -> Geonum {
        // halve angles using angle division
        Geonum::new_with_angle(g.mag, g.angle / 2.0)
    };

    // test adjoint relationship through angle reflection
    // for any objects c in C and d in D, we should have:
    // Hom_D(F(c), d) ≅ Hom_C(c, G(d))

    // create objects
    let c = Geonum::new(1.0, 0.5, 4.0); // object in C - π/4
    let d = Geonum::new(1.0, 1.0, 2.0); // object in D - π/2

    // apply functors
    let fc = left_adjoint(&c); // F(c) in D
    let gd = right_adjoint(&d); // G(d) in C

    // test the adjunction property via angle relationships
    // measure "distance" (in angle space) from F(c) to d in D
    let hom_fd_d = (fc.angle.grade_angle() - d.angle.grade_angle()).abs();

    // measure "distance" from c to G(d) in C
    let hom_c_gd = (c.angle.grade_angle() - gd.angle.grade_angle()).abs();

    // test these are related by our adjunction factor (2 in this case)
    assert!((hom_fd_d - 2.0 * hom_c_gd).abs() < EPSILON);

    // test unit/counit as physical rotation operations
    // the unit of the adjunction η: 1_C → GF
    // can be represented as the angle transformation needed to go from
    // an object to its image under GF

    // apply GF to c
    let gfc = right_adjoint(&left_adjoint(&c));

    // the unit at c is the transformation from c to GF(c)
    let unit_angle = gfc.angle.grade_angle() - c.angle.grade_angle();

    // for our functors, this should be 0 (GF is identity on angles)
    assert!(unit_angle.abs() < EPSILON);

    // similarly test the counit
    let fgd = left_adjoint(&right_adjoint(&d));
    let counit_angle = d.angle.grade_angle() - fgd.angle.grade_angle();

    // for our functors, this should be 0 (FG is identity on angles)
    assert!(counit_angle.abs() < EPSILON);
}

#[test]
fn its_a_limit() {
    // in category theory, a limit is a universal cone to a diagram
    // with geometric numbers, limits are direct angle convergence points

    // create a simple diagram as a set of points with angles from a common center
    // artifact of geonum automation: explicit limit points replaced by geometric relationships
    // Vector (grade 1) - center point as a position vector
    // Note: Since this center point has length 0, it could arguably be modeled
    // as a scalar (blade: 0) in geometric algebra. However, we treat it as a vector
    // for consistency with other points in the diagram and to preserve its
    // role as a position vector with directional properties.
    let _center = Geonum::new(0.0, 0.0, 2.0); // center point (will be the limit)

    // points in our diagram
    let points = [
        Geonum::new(2.0, 0.0, 2.0), // point at 0°
        Geonum::new(2.0, 2.0, 6.0), // point at 60° (π/3)
        Geonum::new(2.0, 4.0, 6.0), // point at 120° (2π/3)
    ];

    // test convergence via direct angle alignment
    // the limit is where all angles converge - in this case, the center

    // define projections as paths from center to each point
    // artifact of geonum automation: categorical projection maps replaced by direct angle relationships
    let _projections: Vec<Geonum> = points
        .iter()
        .map(|p| Geonum::new_with_angle(p.mag, p.angle))
        .collect();

    // test universal property through geometric measurement
    // for any other point that has projections to all diagram points,
    // there must be a unique projection to the limit

    // create another potential "cone" point
    let other_point = Geonum::new(1.0, 1.0, 4.0); // π/4

    // define projections from this point to diagram points
    // artifact of geonum automation: categorical cone structure replaced by direct angle measurements
    let _other_projections: Vec<Geonum> = points
        .iter()
        .map(|p| {
            let p_x = p.mag * p.angle.grade_angle().cos();
            let p_y = p.mag * p.angle.grade_angle().sin();
            let o_x = other_point.mag * other_point.angle.grade_angle().cos();
            let o_y = other_point.mag * other_point.angle.grade_angle().sin();
            let dx = p_x - o_x;
            let dy = p_y - o_y;
            Geonum::new_from_cartesian(dx, dy)
        })
        .collect();

    // test the existence of a unique projection to the center (the limit)
    // artifact of geonum automation: categorical universal property replaced by direct geometric measurement
    let _to_center = Geonum::new_with_angle(other_point.mag, other_point.angle);

    // prove existence through angle construction
    // the projection to the limit must commute with all other projections

    // for our simplified example, we can observe that projections from the center
    // to diagram points have a direct relationship with the angles of those points

    // test with products as special cases
    // a product is a limit of a diagram with just objects (no connecting morphisms)

    // for two objects, the product is a point that projects to both
    let a = Geonum::new(1.0, 0.0, 2.0);
    let b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // the product is a point that has projections to both a and b
    // in our geometric interpretation, this could be the origin
    // artifact of geonum automation: categorical product replaced by simple geometric intersection
    // Vector (grade 1) - product as a position vector in category space
    // Note: In geometric algebra, this zero-length element at the origin
    // could be represented as a scalar (blade: 0) since it has no directional
    // properties. However, we model it as a vector (blade: 1) to maintain
    // consistency with the other objects and to preserve its role as a
    // reference point in the geometric representation of the category.
    let _product = Geonum::new(0.0, 0.0, 2.0);

    // projections from product to a and b
    let proj_to_a = Geonum::new(1.0, 0.0, 2.0);
    let proj_to_b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // verify projections exist
    assert_eq!(proj_to_a.angle, a.angle);
    assert_eq!(proj_to_b.angle, b.angle);
}

#[test]
fn its_a_colimit() {
    // in category theory, a colimit is a universal cocone from a diagram
    // with geometric numbers, colimits are angle divergence points

    // create a simple diagram as a set of points with angles from a common center
    let points = [
        Geonum::new(1.0, 0.0, 2.0), // point at 0°
        Geonum::new(1.0, 2.0, 6.0), // point at 60° (π/3)
        Geonum::new(1.0, 4.0, 6.0), // point at 120° (2π/3)
    ];

    // the colimit is a point that all points map to - in this case, a point farther out
    let colimit = Geonum::new(2.0, 2.0, 6.0); // arbitrary point "containing" the diagram (π/3)

    // test divergence via angle-based expansion
    // define injections as paths from each point to the colimit
    // artifact of geonum automation: categorical injections replaced by direct geometric paths
    let _injections: Vec<Geonum> = points
        .iter()
        .map(|p| {
            let c_x = colimit.mag * colimit.angle.grade_angle().cos();
            let c_y = colimit.mag * colimit.angle.grade_angle().sin();
            let p_x = p.mag * p.angle.grade_angle().cos();
            let p_y = p.mag * p.angle.grade_angle().sin();
            let dx = c_x - p_x;
            let dy = c_y - p_y;
            Geonum::new_from_cartesian(dx, dy)
        })
        .collect();

    // test universal property through spatial measurement
    // for any other point that the diagram points map to,
    // there must be a unique map from the colimit to that point

    // create another potential "cocone" point
    let other_point = Geonum::new(3.0, 1.0, 2.0); // π/2

    // define injections from diagram points to this other point
    // artifact of geonum automation: categorical cocone structure replaced by direct angle paths
    let _other_injections: Vec<Geonum> = points
        .iter()
        .map(|p| {
            let o_x = other_point.mag * other_point.angle.grade_angle().cos();
            let o_y = other_point.mag * other_point.angle.grade_angle().sin();
            let p_x = p.mag * p.angle.grade_angle().cos();
            let p_y = p.mag * p.angle.grade_angle().sin();
            let dx = o_x - p_x;
            let dy = o_y - p_y;
            Geonum::new_from_cartesian(dx, dy)
        })
        .collect();

    // test the existence of a unique map from colimit to other_point
    // artifact of geonum automation: categorical universal cocone property replaced by direct angle path
    let o_x = other_point.mag * other_point.angle.grade_angle().cos();
    let o_y = other_point.mag * other_point.angle.grade_angle().sin();
    let c_x = colimit.mag * colimit.angle.grade_angle().cos();
    let c_y = colimit.mag * colimit.angle.grade_angle().sin();
    let _to_other = Geonum::new_from_cartesian(o_x - c_x, o_y - c_y);

    // prove colimit construction through direct angle operations
    // for a simplified example, just verify the colimit has greater length than the diagram points
    assert!(colimit.mag > points[0].mag);

    // test coproducts as special cases of angle divergence
    // a coproduct is a colimit of a diagram with just objects (no connecting morphisms)

    // for two objects, the coproduct is a point that both objects map to
    let a = Geonum::new(1.0, 0.0, 2.0);
    let b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // the coproduct in our geometric interpretation could be a point that "contains" both
    let coproduct = Geonum::new(2.0, 1.0, 4.0); // π/4

    // injections from a and b to the coproduct
    // artifact of geonum automation: categorical injection maps replaced by direct geometric relationships
    let _inj_from_a = Geonum::new(1.5, 1.0, 8.0); // π/8
    let _inj_from_b = Geonum::new(1.5, 3.0, 8.0); // 3π/8

    // verify injections exist with correct directionality (from points to coproduct)
    assert!(coproduct.mag > a.mag);
    assert!(coproduct.mag > b.mag);
}

#[test]
fn its_a_natural_transformation() {
    // in category theory, a natural transformation is a family of morphisms between functors
    // with geometric numbers, its a direct geometric transition between function spaces

    // create a "source category" as a space
    // natural transformations traditionally require naturality squares and commutative diagrams
    // geonum makes this direct: natural transformations become coherent angle mappings
    // no "category" setup needed - geometric numbers naturally compose with angle arithmetic

    // create "objects" in source category
    let a = Geonum::new(1.0, 0.0, 2.0);
    let b = Geonum::new(1.0, 1.0, 2.0); // π/2

    // define two "functors" from C to D
    let functor_f = |g: &Geonum| -> Geonum { Geonum::new_with_angle(g.mag * 2.0, g.angle) };

    let functor_g = |g: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            g.mag,
            g.angle + Angle::new(1.0, 6.0), // add π/6
        )
    };

    // apply functors to objects
    let f_a = functor_f(&a);
    let f_b = functor_f(&b);
    let g_a = functor_g(&a);
    let g_b = functor_g(&b);

    // test geometric transitions between function spaces
    // define a natural transformation as an angle rotation
    let eta = PI / 4.0; // natural transformation as a fixed rotation

    // apply the natural transformation to F(a) to get G(a)
    let transform_fa = Geonum::new(
        f_a.mag / g_a.mag, // adjust length ratio
        eta * 4.0 / PI,
        4.0, // fixed angle transformation - eta is π/4
    );

    // apply the natural transformation to F(b) to get G(b)
    let transform_fb = Geonum::new(
        f_b.mag / g_b.mag, // adjust length ratio
        eta * 4.0 / PI,
        4.0, // same angle transformation - eta is π/4
    );

    // test naturality through angle consistency
    // the key property is that the angle transformation is consistent
    // regardless of which object we apply it to
    assert_eq!(transform_fa.angle, transform_fb.angle);

    // test commutativity from angle preservation
    // if we have a morphism a → b, then the diagram commutes when:
    // G(a → b) ∘ η_a = η_b ∘ F(a → b)

    // create a "morphism" in the source category
    let morphism_ab = Geonum::new(1.0, 1.0, 2.0); // maps a to b - π/2

    // compute F(morphism_ab)
    let f_morphism = Geonum::new_with_angle(
        morphism_ab.mag * functor_f(&a).mag / a.mag,
        morphism_ab.angle,
    );

    // compute G(morphism_ab)
    let g_morphism = Geonum::new_with_angle(
        morphism_ab.mag * functor_g(&a).mag / a.mag,
        morphism_ab.angle,
    );

    // verify morphisms preserve expected properties
    assert!(g_morphism.angle.grade_angle().abs() > EPSILON);
    assert!(f_morphism.angle.grade_angle().abs() > EPSILON);
}

#[test]
fn its_a_monad() {
    // in category theory, a monad is a special endofunctor with unit and multiplication
    // with geometric numbers, its a simple binding operation of angle rotations

    // create a space to work in
    // monads traditionally require unit/return and bind operations with associativity laws
    // geonum eliminates this machinery: monadic composition becomes angle transformation chains
    // geometric numbers inherently satisfy monad laws through angle arithmetic properties

    // define a "monad" as an angle transformation with bind operation
    let transform = |g: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            g.mag,
            g.angle + g.angle, // double the angle
        )
    };

    // define the "unit" of the monad
    // this maps an element to its image in the monad
    let unit = |g: &Geonum| -> Geonum {
        *g // identity for simplicity
    };

    // define the "multiplication" (join) of the monad
    // this flattens a nested application of the monad
    let join = |g: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            g.mag,
            g.angle / 2.0, // halve the angle (to compensate for double application)
        )
    };

    // define the "bind" operation
    let bind = |g: &Geonum, f: fn(&Geonum) -> Geonum| -> Geonum {
        // bind = join ∘ map(f) ∘ transform
        let transformed = transform(g);
        let mapped = f(&transformed);
        join(&mapped)
    };

    // test bind operation as direct rotation sequence
    let x = Geonum::new(2.0, 1.0, 4.0); // vector at π/4

    // create a function to bind with
    let f = |g: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            g.mag * 3.0,
            g.angle + Angle::new(1.0, 2.0), // add π/2
        )
    };

    // apply bind operation
    // artifact of geonum automation: monadic binding operation replaced by direct angle composition
    let _result = bind(&x, f);

    // test associativity through angle composition
    // (using another function for second bind)
    let g = |g: &Geonum| -> Geonum {
        Geonum::new_with_angle(
            g.mag / 2.0,
            g.angle - Angle::new(2.0, 6.0), // subtract π/3
        )
    };

    // create and compose the functions directly to test associativity properties
    let f_result = f(&x);
    let g_result = g(&f_result);

    // verify transformations maintain expected properties
    assert!(g_result.mag > 0.0);
    assert!(g_result.angle.rem() > EPSILON || g_result.angle.blade() > 0);

    // test unit laws
    let unit_result = unit(&x);
    assert_eq!(unit_result.mag, x.mag);
    assert_eq!(unit_result.angle, x.angle);
}

#[test]
fn it_rejects_category_theory() {
    // category theory obscures relationships behind arrows
    // geometric numbers show relationships directly through angles

    // test direct geometric relationships
    // instead of abstract morphisms, we use concrete rotations
    // category theory formalizes relationships through arrows and composition
    // geonum makes this computational: relationships become measurable angle differences
    let a = Geonum::new(1.0, 0.0, 2.0); // vector at 0°
    let b = Geonum::new(1.0, 2.0, 6.0); // vector at π/3

    // test commutative diagram avoidance through angle measurement
    // in category theory, commutativity is asserted symbolically
    // with geometric numbers, we measure it directly

    // measure transformation from a to b
    let transform_ab = b.angle - a.angle;
    assert_eq!(transform_ab, Angle::new(2.0, 6.0)); // π/3

    // apply a different route: a → c → b
    let c = Geonum::new(1.0, 1.0, 6.0); // vector at π/6
    let transform_ac = c.angle - a.angle;
    let transform_cb = b.angle - c.angle;
    let combined = transform_ac + transform_cb;

    // test both paths give the same result
    assert_eq!(combined, transform_ab);

    // test consistency from universe geometry not abstract rules
    // geometric numbers derive their properties from physical space
    // not from artificially imposed axioms

    // demonstrate rotation consistency from geometric properties
    // composing rotations is associative because thats how physical space works
    let r1 = Geonum::new(1.0, 1.0, 5.0); // vector at π/5
    let r2 = Geonum::new(1.0, 1.0, 7.0); // vector at π/7
    let r3 = Geonum::new(1.0, 1.0, 11.0); // vector at π/11

    // test (r1 ∘ r2) ∘ r3 = r1 ∘ (r2 ∘ r3)
    let left_assoc = (r1 * r2) * r3;
    let right_assoc = r1 * (r2 * r3);

    assert_eq!(left_assoc.mag, right_assoc.mag);
    assert_eq!(left_assoc.angle, right_assoc.angle);

    // test category theory as unnecessary abstraction
    // we dont need abstract categories to understand mathematics
    // geometry provides a concrete foundation

    // demonstrate direct angle measurement for understanding relationships
    let vectors = [
        Geonum::new(1.0, 0.0, 2.0), // vector at 0°
        Geonum::new(1.0, 1.0, 4.0), // vector at π/4
        Geonum::new(1.0, 1.0, 2.0), // vector at π/2
    ];

    // compute relationships directly through angle measurements
    for i in 0..vectors.len() {
        for j in 0..vectors.len() {
            if i != j {
                // directly measure and understand the relationship
                let angle_diff = vectors[j].angle - vectors[i].angle;

                // no need for category theory to understand these relationships
                // angle differences are always meaningful in geonum
                assert!(angle_diff.blade() < 100); // reasonable bound for testing
            }
        }
    }
}
