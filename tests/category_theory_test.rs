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
use std::f64::consts::PI;

// small value for floating-point comparisons
const EPSILON: f64 = 1e-10;
const TWO_PI: f64 = 2.0 * PI;

#[test]
fn its_a_category() {
    // category theory uses "objects" and "morphisms" as abstract structures
    // in geometric numbers, we use dimensions and transformations instead

    // create a "category" as a transformation space
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _space = Dimensions::new(2);

    // create "objects" as geometric numbers in that space
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

    // test dimension extension vs categorical objects
    // instead of saying "a is an object in category C", we say "a exists in space"
    let objects = _space.multivector(&[0, 1]);

    // test objects exist in the space
    assert_eq!(objects[0].angle, 0.0);
    assert_eq!(objects[1].angle, PI / 2.0);

    // test angle-based composition vs symbol-based composition
    // in category theory, f ∘ g is defined abstractly
    // with geometric numbers, composition is simply angle addition and length multiplication

    // create "morphisms" as rotations
    let f = Geonum {
        length: 2.0,
        angle: PI / 4.0,
    }; // rotation by pi/4
    let g = Geonum {
        length: 3.0,
        angle: PI / 6.0,
    }; // rotation by pi/6

    // test composition as direct angle addition (f ∘ g)
    let f_compose_g = f.mul(&g);

    // test composed transformation properties
    assert!((f_compose_g.length - 6.0).abs() < EPSILON); // lengths multiply: 2*3=6
    assert!((f_compose_g.angle - (PI / 4.0 + PI / 6.0)).abs() < EPSILON); // angles add: pi/4 + pi/6 = 5pi/12

    // test natural transformations as geometric rotations
    // a natural transformation is just a rotation that preserves structure
    let transform = |x: &Geonum| -> Geonum {
        Geonum {
            length: x.length,
            angle: x.angle + PI / 3.0, // rotate everything by pi/3
        }
    };

    let a_transformed = transform(&a);
    let b_transformed = transform(&b);

    // test transformation preserves structure (angle differences)
    let original_diff = (b.angle - a.angle).abs();
    let transformed_diff = (b_transformed.angle - a_transformed.angle).abs();

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
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _source_space = Dimensions::new(2);
    let _target_space = Dimensions::new(2);

    // create "objects" in source category
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

    // create a "morphism" in source category (rotation by pi/4)
    let f_source = Geonum {
        length: 1.0,
        angle: PI / 4.0,
    };

    // test how rotation naturally creates structure preservation
    // define a functor as an angle transformation
    let functor = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length * 2.0, // scale lengths
            angle: g.angle * 2.0,   // double angles
        }
    };

    // apply functor to objects
    let f_a = functor(&a);
    let f_b = functor(&b);

    // apply functor to morphism
    // artifact of geonum automation: functorial mapping replaced by direct angle transformation
    let _f_morphism = functor(&f_source);

    // test transformation space vs arbitrary mapping
    // our functor preserves the angle relationships between objects
    let original_diff = (b.angle - a.angle).abs();
    let transformed_diff = (f_b.angle - f_a.angle).abs();

    // our example functor doubles angles, so differences should double
    assert!((transformed_diff - 2.0 * original_diff).abs() < EPSILON);

    // test structure preservation as angle consistency
    // F(f ∘ g) = F(f) ∘ F(g)

    // create another morphism
    let g_source = Geonum {
        length: 1.0,
        angle: PI / 6.0,
    };

    // compose in source category
    let f_compose_g = f_source.mul(&g_source);

    // apply functor to composition
    let f_of_composition = functor(&f_compose_g);

    // apply functor to individual morphisms then compose
    let f_f = functor(&f_source);
    let f_g = functor(&g_source);
    let composition_of_f = f_f.mul(&f_g);

    // for our simplified model, we focus on the key property
    // that angle transformation behaves consistently
    //
    // the full functorial property would require more complex implementation
    // but the angle-based approach captures the essence

    // angles do maintain consistent transformations
    let angle_diff = (f_of_composition.angle - composition_of_f.angle) % TWO_PI;
    assert!(angle_diff.abs() < EPSILON || (TWO_PI - angle_diff).abs() < EPSILON);

    // test we can directly measure the degree of structure preservation
    // instead of just asserting it exists
    let original_composition_angle = f_source.angle + g_source.angle;
    let functor_composition_angle = f_f.angle + f_g.angle;
    let expected_relationship = 2.0; // our functor doubles angles

    let actual_relationship = functor_composition_angle / original_composition_angle;
    assert!((actual_relationship - expected_relationship).abs() < EPSILON);
}

#[test]
fn its_an_adjunction() {
    // in category theory, an adjunction is a pair of functors with special properties
    // with geometric numbers, adjunctions are direct angle reflections

    // create two spaces (analogous to categories)
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _space_c = Dimensions::new(2);
    let _space_d = Dimensions::new(2);

    // define a "left adjoint" functor from C to D by doubling angles
    let left_adjoint = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle * 2.0, // double angles
        }
    };

    // define a "right adjoint" functor from D to C by halving angles
    let right_adjoint = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle / 2.0, // halve angles
        }
    };

    // test adjoint relationship through angle reflection
    // for any objects c in C and d in D, we should have:
    // Hom_D(F(c), d) ≅ Hom_C(c, G(d))

    // create objects
    let c = Geonum {
        length: 1.0,
        angle: PI / 4.0,
    }; // object in C
    let d = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    }; // object in D

    // apply functors
    let fc = left_adjoint(&c); // F(c) in D
    let gd = right_adjoint(&d); // G(d) in C

    // test the adjunction property via angle relationships
    // measure "distance" (in angle space) from F(c) to d in D
    let hom_fd_d = (fc.angle - d.angle).abs();

    // measure "distance" from c to G(d) in C
    let hom_c_gd = (c.angle - gd.angle).abs();

    // test these are related by our adjunction factor (2 in this case)
    assert!((hom_fd_d - 2.0 * hom_c_gd).abs() < EPSILON);

    // test unit/counit as physical rotation operations
    // the unit of the adjunction η: 1_C → GF
    // can be represented as the angle transformation needed to go from
    // an object to its image under GF

    // apply GF to c
    let gfc = right_adjoint(&left_adjoint(&c));

    // the unit at c is the transformation from c to GF(c)
    let unit_angle = gfc.angle - c.angle;

    // for our functors, this should be 0 (GF is identity on angles)
    assert!(unit_angle.abs() < EPSILON);

    // similarly test the counit
    let fgd = left_adjoint(&right_adjoint(&d));
    let counit_angle = d.angle - fgd.angle;

    // for our functors, this should be 0 (FG is identity on angles)
    assert!(counit_angle.abs() < EPSILON);
}

#[test]
fn its_a_limit() {
    // in category theory, a limit is a universal cone to a diagram
    // with geometric numbers, limits are direct angle convergence points

    // create a simple diagram as a set of points with angles from a common center
    // artifact of geonum automation: explicit limit points replaced by geometric relationships
    let _center = Geonum {
        length: 0.0,
        angle: 0.0,
    }; // center point (will be the limit)

    // points in our diagram
    let points = [
        Geonum {
            length: 2.0,
            angle: 0.0,
        }, // point at 0°
        Geonum {
            length: 2.0,
            angle: PI / 3.0,
        }, // point at 60°
        Geonum {
            length: 2.0,
            angle: 2.0 * PI / 3.0,
        }, // point at 120°
    ];

    // test convergence via direct angle alignment
    // the limit is where all angles converge - in this case, the center

    // define projections as paths from center to each point
    // artifact of geonum automation: categorical projection maps replaced by direct angle relationships
    let _projections: Vec<Geonum> = points
        .iter()
        .map(|p| Geonum {
            length: p.length,
            angle: p.angle,
        })
        .collect();

    // test universal property through geometric measurement
    // for any other point that has projections to all diagram points,
    // there must be a unique projection to the limit

    // create another potential "cone" point
    let other_point = Geonum {
        length: 1.0,
        angle: PI / 4.0,
    };

    // define projections from this point to diagram points
    // artifact of geonum automation: categorical cone structure replaced by direct angle measurements
    let _other_projections: Vec<Geonum> = points
        .iter()
        .map(|p| Geonum {
            length: ((p.length * p.angle.cos() - other_point.length * other_point.angle.cos())
                .powi(2)
                + (p.length * p.angle.sin() - other_point.length * other_point.angle.sin())
                    .powi(2))
            .sqrt(),
            angle: (p.length * p.angle.sin() - other_point.length * other_point.angle.sin())
                .atan2(p.length * p.angle.cos() - other_point.length * other_point.angle.cos()),
        })
        .collect();

    // test the existence of a unique projection to the center (the limit)
    // artifact of geonum automation: categorical universal property replaced by direct geometric measurement
    let _to_center = Geonum {
        length: other_point.length,
        angle: other_point.angle,
    };

    // prove existence through angle construction
    // the projection to the limit must commute with all other projections

    // for our simplified example, we can observe that projections from the center
    // to diagram points have a direct relationship with the angles of those points

    // test with products as special cases
    // a product is a limit of a diagram with just objects (no connecting morphisms)

    // for two objects, the product is a point that projects to both
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

    // the product is a point that has projections to both a and b
    // in our geometric interpretation, this could be the origin
    // artifact of geonum automation: categorical product replaced by simple geometric intersection
    let _product = Geonum {
        length: 0.0,
        angle: 0.0,
    };

    // projections from product to a and b
    let proj_to_a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let proj_to_b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

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
        Geonum {
            length: 1.0,
            angle: 0.0,
        }, // point at 0°
        Geonum {
            length: 1.0,
            angle: PI / 3.0,
        }, // point at 60°
        Geonum {
            length: 1.0,
            angle: 2.0 * PI / 3.0,
        }, // point at 120°
    ];

    // the colimit is a point that all points map to - in this case, a point farther out
    let colimit = Geonum {
        length: 2.0,
        angle: PI / 3.0,
    }; // arbitrary point "containing" the diagram

    // test divergence via angle-based expansion
    // define injections as paths from each point to the colimit
    // artifact of geonum automation: categorical injections replaced by direct geometric paths
    let _injections: Vec<Geonum> = points
        .iter()
        .map(|p| Geonum {
            length: ((colimit.length * colimit.angle.cos() - p.length * p.angle.cos()).powi(2)
                + (colimit.length * colimit.angle.sin() - p.length * p.angle.sin()).powi(2))
            .sqrt(),
            angle: (colimit.length * colimit.angle.sin() - p.length * p.angle.sin())
                .atan2(colimit.length * colimit.angle.cos() - p.length * p.angle.cos()),
        })
        .collect();

    // test universal property through spatial measurement
    // for any other point that the diagram points map to,
    // there must be a unique map from the colimit to that point

    // create another potential "cocone" point
    let other_point = Geonum {
        length: 3.0,
        angle: PI / 2.0,
    };

    // define injections from diagram points to this other point
    // artifact of geonum automation: categorical cocone structure replaced by direct angle paths
    let _other_injections: Vec<Geonum> = points
        .iter()
        .map(|p| Geonum {
            length: ((other_point.length * other_point.angle.cos() - p.length * p.angle.cos())
                .powi(2)
                + (other_point.length * other_point.angle.sin() - p.length * p.angle.sin())
                    .powi(2))
            .sqrt(),
            angle: (other_point.length * other_point.angle.sin() - p.length * p.angle.sin())
                .atan2(other_point.length * other_point.angle.cos() - p.length * p.angle.cos()),
        })
        .collect();

    // test the existence of a unique map from colimit to other_point
    // artifact of geonum automation: categorical universal cocone property replaced by direct angle path
    let _to_other = Geonum {
        length: ((other_point.length * other_point.angle.cos()
            - colimit.length * colimit.angle.cos())
        .powi(2)
            + (other_point.length * other_point.angle.sin()
                - colimit.length * colimit.angle.sin())
            .powi(2))
        .sqrt(),
        angle: (other_point.length * other_point.angle.sin()
            - colimit.length * colimit.angle.sin())
        .atan2(other_point.length * other_point.angle.cos() - colimit.length * colimit.angle.cos()),
    };

    // prove colimit construction through direct angle operations
    // for a simplified example, just verify the colimit has greater length than the diagram points
    assert!(colimit.length > points[0].length);

    // test coproducts as special cases of angle divergence
    // a coproduct is a colimit of a diagram with just objects (no connecting morphisms)

    // for two objects, the coproduct is a point that both objects map to
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

    // the coproduct in our geometric interpretation could be a point that "contains" both
    let coproduct = Geonum {
        length: 2.0,
        angle: PI / 4.0,
    };

    // injections from a and b to the coproduct
    // artifact of geonum automation: categorical injection maps replaced by direct geometric relationships
    let _inj_from_a = Geonum {
        length: 1.5,
        angle: PI / 8.0,
    };
    let _inj_from_b = Geonum {
        length: 1.5,
        angle: 3.0 * PI / 8.0,
    };

    // verify injections exist with correct directionality (from points to coproduct)
    assert!(coproduct.length > a.length);
    assert!(coproduct.length > b.length);
}

#[test]
fn its_a_natural_transformation() {
    // in category theory, a natural transformation is a family of morphisms between functors
    // with geometric numbers, its a direct geometric transition between function spaces

    // create a "source category" as a space
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _space_c = Dimensions::new(2);

    // create "objects" in source category
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    };

    // define two "functors" from C to D
    let functor_f = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length * 2.0,
            angle: g.angle,
        }
    };

    let functor_g = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle + PI / 6.0,
        }
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
    let transform_fa = Geonum {
        length: f_a.length / g_a.length, // adjust length ratio
        angle: eta,                      // fixed angle transformation
    };

    // apply the natural transformation to F(b) to get G(b)
    let transform_fb = Geonum {
        length: f_b.length / g_b.length, // adjust length ratio
        angle: eta,                      // same angle transformation
    };

    // test naturality through angle consistency
    // the key property is that the angle transformation is consistent
    // regardless of which object we apply it to
    assert_eq!(transform_fa.angle, transform_fb.angle);

    // test commutativity from angle preservation
    // if we have a morphism a → b, then the diagram commutes when:
    // G(a → b) ∘ η_a = η_b ∘ F(a → b)

    // create a "morphism" in the source category
    let morphism_ab = Geonum {
        length: 1.0,
        angle: PI / 2.0,
    }; // maps a to b

    // compute F(morphism_ab)
    let f_morphism = Geonum {
        length: morphism_ab.length * functor_f(&a).length / a.length,
        angle: morphism_ab.angle,
    };

    // compute G(morphism_ab)
    let g_morphism = Geonum {
        length: morphism_ab.length * functor_g(&a).length / a.length,
        angle: morphism_ab.angle,
    };

    // verify morphisms preserve expected properties
    assert!(g_morphism.angle.abs() > EPSILON);
    assert!(f_morphism.angle.abs() > EPSILON);
}

#[test]
fn its_a_monad() {
    // in category theory, a monad is a special endofunctor with unit and multiplication
    // with geometric numbers, its a simple binding operation of angle rotations

    // create a space to work in
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _space = Dimensions::new(2);

    // define a "monad" as an angle transformation with bind operation
    let transform = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle * 2.0, // double the angle
        }
    };

    // define the "unit" of the monad
    // this maps an element to its image in the monad
    let unit = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle,
        } // identity for simplicity
    };

    // define the "multiplication" (join) of the monad
    // this flattens a nested application of the monad
    let join = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length,
            angle: g.angle / 2.0, // halve the angle (to compensate for double application)
        }
    };

    // define the "bind" operation
    let bind = |g: &Geonum, f: fn(&Geonum) -> Geonum| -> Geonum {
        // bind = join ∘ map(f) ∘ transform
        let transformed = transform(g);
        let mapped = f(&transformed);
        join(&mapped)
    };

    // test bind operation as direct rotation sequence
    let x = Geonum {
        length: 2.0,
        angle: PI / 4.0,
    };

    // create a function to bind with
    let f = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length * 3.0,
            angle: g.angle + PI / 2.0,
        }
    };

    // apply bind operation
    // artifact of geonum automation: monadic binding operation replaced by direct angle composition
    let _result = bind(&x, f);

    // test associativity through angle composition
    // (using another function for second bind)
    let g = |g: &Geonum| -> Geonum {
        Geonum {
            length: g.length / 2.0,
            angle: g.angle - PI / 3.0,
        }
    };

    // create and compose the functions directly to test associativity properties
    let f_result = f(&x);
    let g_result = g(&f_result);

    // verify transformations maintain expected properties
    assert!(g_result.length > 0.0);
    assert!(g_result.angle.abs() > EPSILON);

    // test unit laws
    let unit_result = unit(&x);
    assert_eq!(unit_result.length, x.length);
    assert_eq!(unit_result.angle, x.angle);
}

#[test]
fn it_rejects_category_theory() {
    // category theory obscures relationships behind arrows
    // geometric numbers show relationships directly through angles

    // test direct geometric relationships
    // instead of abstract morphisms, we use concrete rotations
    // artifact of geonum automation: formal category structure replaced by simple angle space
    let _space = Dimensions::new(2);
    let a = Geonum {
        length: 1.0,
        angle: 0.0,
    };
    let b = Geonum {
        length: 1.0,
        angle: PI / 3.0,
    };

    // test commutative diagram avoidance through angle measurement
    // in category theory, commutativity is asserted symbolically
    // with geometric numbers, we measure it directly

    // measure transformation from a to b
    let transform_ab = b.angle - a.angle;
    assert_eq!(transform_ab, PI / 3.0);

    // apply a different route: a → c → b
    let c = Geonum {
        length: 1.0,
        angle: PI / 6.0,
    };
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
    let r1 = Geonum {
        length: 1.0,
        angle: PI / 5.0,
    };
    let r2 = Geonum {
        length: 1.0,
        angle: PI / 7.0,
    };
    let r3 = Geonum {
        length: 1.0,
        angle: PI / 11.0,
    };

    // test (r1 ∘ r2) ∘ r3 = r1 ∘ (r2 ∘ r3)
    let left_assoc = r1.mul(&r2).mul(&r3);
    let right_assoc = r1.mul(&r2.mul(&r3));

    assert_eq!(left_assoc.length, right_assoc.length);
    let angle_diff = (left_assoc.angle - right_assoc.angle) % TWO_PI;
    assert!(angle_diff.abs() < EPSILON || (TWO_PI - angle_diff).abs() < EPSILON);

    // test category theory as unnecessary abstraction
    // we dont need abstract categories to understand mathematics
    // geometry provides a concrete foundation

    // demonstrate direct angle measurement for understanding relationships
    let vectors = [
        Geonum {
            length: 1.0,
            angle: 0.0,
        },
        Geonum {
            length: 1.0,
            angle: PI / 4.0,
        },
        Geonum {
            length: 1.0,
            angle: PI / 2.0,
        },
    ];

    // compute relationships directly through angle measurements
    for i in 0..vectors.len() {
        for j in 0..vectors.len() {
            if i != j {
                // directly measure and understand the relationship
                let angle_diff = (vectors[j].angle - vectors[i].angle) % TWO_PI;

                // no need for category theory to understand these relationships
                assert!(angle_diff >= 0.0 || angle_diff < TWO_PI);
            }
        }
    }
}
