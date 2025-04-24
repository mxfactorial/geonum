use crate::geonum_mod::{Geonum, EPSILON, TWO_PI};
use crate::multivector::Multivector;

pub trait Manifold {
    /// find a component matching the given path angle
    /// conventional: tree traversal or hash lookup O(log n) or O(1) with overhead
    /// geonum: direct angle-based component lookup O(n) but with small n and no tree overhead
    fn find(&self, path_angle: f64) -> Option<&Geonum>;

    /// apply a transformation to all components through angle rotation
    /// conventional: traverse and transform each element individually O(n)
    /// geonum: single unified transformation through angle arithmetic O(n) with minimal operations
    fn transform(&self, angle_rotation: f64) -> Self;

    /// create path mapping function for use with complex data structures
    /// conventional: complex path traversal functions with nested references O(depth)
    /// geonum: angle-encoded path functions for direct geometric access O(1) per lookup
    fn path_mapper<T>(&self, path_generator: fn(&T) -> f64) -> impl Fn(&T) -> Vec<Geonum>;

    /// set value at a specific path angle
    /// conventional: recursive traversal with mutation O(depth)
    /// geonum: direct angle-based transformation O(1)
    fn set(&self, path_angle: f64, new_value: f64) -> Self;

    /// apply function to value at a specific path angle
    /// conventional: complex functor composition O(depth)
    /// geonum: direct angle-path function application O(1)
    fn over<F>(&self, path_angle: f64, f: F) -> Self
    where
        F: Fn(f64) -> f64;

    /// compose paths through angle addition
    /// conventional: nested higher-order functions O(n)
    /// geonum: direct angle arithmetic O(1)
    fn compose(&self, other_angle: f64) -> Self;
}

impl Manifold for Multivector {
    fn find(&self, path_angle: f64) -> Option<&Geonum> {
        self.0
            .iter()
            .find(|g| (g.angle - path_angle).abs() < EPSILON)
    }

    fn transform(&self, angle_rotation: f64) -> Self {
        Self(
            self.0
                .iter()
                .map(|g| Geonum {
                    length: g.length,
                    angle: g.angle + angle_rotation,
                    blade: g.blade, // preserve blade grade
                })
                .collect(),
        )
    }

    fn path_mapper<T>(&self, path_generator: fn(&T) -> f64) -> impl Fn(&T) -> Vec<Geonum> {
        move |data: &T| {
            let path = path_generator(data);
            self.0
                .iter()
                .filter(|g| (g.angle - path).abs() < EPSILON)
                .cloned()
                .collect()
        }
    }

    fn set(&self, path_angle: f64, new_value: f64) -> Self {
        // create a new multivector with the updated value
        Self(
            self.0
                .iter()
                .map(|g| {
                    // if this is the component at the target path, update its value
                    if (g.angle - path_angle).abs() < EPSILON {
                        Geonum {
                            length: new_value,
                            angle: g.angle,
                            blade: g.blade, // preserve blade grade
                        }
                    } else {
                        // otherwise keep the original component
                        *g
                    }
                })
                .collect(),
        )
    }

    fn over<F>(&self, path_angle: f64, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        // create a new multivector by applying the function to the target component
        Self(
            self.0
                .iter()
                .map(|g| {
                    // if this is the component at the target path, apply the function
                    if (g.angle - path_angle).abs() < EPSILON {
                        Geonum {
                            length: f(g.length),
                            angle: g.angle,
                            blade: g.blade, // preserve blade grade
                        }
                    } else {
                        // otherwise keep the original component
                        *g
                    }
                })
                .collect(),
        )
    }

    fn compose(&self, other_angle: f64) -> Self {
        // create a new multivector with composed paths
        Self(
            self.0
                .iter()
                .map(|g| Geonum {
                    length: g.length,
                    angle: (g.angle + other_angle) % TWO_PI,
                    blade: g.blade, // preserve blade grade
                })
                .collect(),
        )
    }
}
