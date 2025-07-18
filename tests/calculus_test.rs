use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;
const TWO_PI: f64 = 2.0 * PI;

#[test]
fn it_computes_limits() {
    // this test demonstrates that "limits" are unnecessary when using geometric numbers

    // differentiation is simply a pi/2 rotation and the foundation of
    // calculus emerges directly from this geometric structure

    // let v = [[1, 0], [1, pi/2]] # 2d

    // everything can be a 1d "derivative" or projection of the base 2d v
    // so long as the difference between their angles is pi/2 and they
    // follow the "angles add, lengths multiply" rule

    // v'       = [1, pi/2]  # first derivative (rotate v by pi/2)
    // v''      = [1, pi]    # second derivative (rotate v' by pi/2) = -v
    // v'''     = [1, 3pi/2] # third derivative (rotate v'' by pi/2) = -v'
    // v''''    = [1, 2pi]   # fourth derivative (rotate v''' by pi/2) = v
    // v'''''   = [1, 5pi/2] # fifth derivative (rotate v'''' by pi/2) = v'
    // v''''''  = [1, 3pi]   # sixth derivative (rotate v''''' by pi/2) = -v
    // v''''''' = [1, 7pi/2] # seventh derivative (rotate v'''''' by pi/2) = -v'

    // this geometric space enables continuous rotation as an
    // incrementing pi/2 angle, which is the essence of differentiation,
    // and sets the period of the "derive" function to 4

    // the wedge product between vectors AND their derivatives is nilpotent

    let v = [
        Geonum::new(1.0, 0.0, 2.0), // [1, 0]
        Geonum::new(1.0, 1.0, 2.0), // [1, pi/2]
    ];

    // extract the components
    let v0 = v[0]; // [1, 0]
    let v1 = v[1]; // [1, pi/2]

    // the derivative v' is directly represented by the second basis vector
    // this demonstrates how differentiation emerges from the initial pair
    // without requiring limits
    let v_prime = v1;

    // prove v' = [1, pi/2]
    assert_eq!(v_prime.length, 1.0);
    assert!((v_prime.angle.mod_4_angle() - PI / 2.0).abs() < EPSILON);

    // prove nilpotency using wedge product
    let self_wedge = v0.wedge(&v0);
    assert!(self_wedge.length < EPSILON);

    // prove differentiating twice returns negative of original
    // v'' = v' rotated by pi/2 = [1, pi/2 + pi/2] = [1, pi] = -v
    let v_double_prime = Geonum::new_with_angle(
        v_prime.length,
        v_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // prove v'' = -v
    assert_eq!(v_double_prime.length, v0.length);
    assert!((v_double_prime.angle.mod_4_angle() - PI).abs() < EPSILON);

    // prove the 4-cycle property by computing v''' and v''''
    let v_triple_prime = Geonum::new_with_angle(
        v_double_prime.length,
        v_double_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v''' = [1, 3pi/2] = -v'
    assert_eq!(v_triple_prime.length, v_prime.length);
    assert!((v_triple_prime.angle.mod_4_angle() - 3.0 * PI / 2.0).abs() < EPSILON);

    let v_quadruple_prime = Geonum::new_with_angle(
        v_triple_prime.length,
        v_triple_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v'''' = [1, 0] = original v
    assert_eq!(v_quadruple_prime.length, v0.length);
    let angle_rad = v_quadruple_prime.angle.mod_4_angle();
    assert!(angle_rad < EPSILON || (TWO_PI - angle_rad) < EPSILON);

    // extend the demonstration with fifth derivative
    let v_quintuple_prime = Geonum::new_with_angle(
        v_quadruple_prime.length,
        v_quadruple_prime.angle + Angle::new(1.0, 2.0), // add π/2
    );

    // v''''' = [1, pi/2] = v'
    assert_eq!(v_quintuple_prime.length, v_prime.length);
    assert!((v_quintuple_prime.angle.mod_4_angle() - v_prime.angle.mod_4_angle()).abs() < EPSILON);
}
