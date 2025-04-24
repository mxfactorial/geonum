use crate::geonum_mod::Geonum;

pub trait Projection: Sized {
    /// view data through a specific projection path
    /// conventional: complex lens functions with higher-order abstractions
    /// geonum: encode projection paths as angles for direct access
    fn view<T>(&self, data: &T, path_encoder: fn(&T) -> f64) -> Self;

    /// compose two projections into a single new projection
    /// conventional: nested higher-order functions
    /// geonum: direct angle addition
    fn compose(&self, other: &Self) -> Self;
}

impl Projection for Geonum {
    fn view<T>(&self, data: &T, path_encoder: fn(&T) -> f64) -> Self {
        let path = path_encoder(data);

        // create a projection by encoding the path as an angle
        Self {
            length: self.length,
            angle: self.angle + path,
            blade: self.blade, // preserve blade grade
        }
    }

    fn compose(&self, other: &Self) -> Self {
        // compose projections by angle addition
        Self {
            length: self.length * other.length,
            angle: self.angle + other.angle,
            blade: self.blade, // preserve blade grade
        }
    }
}
