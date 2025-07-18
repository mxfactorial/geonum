use crate::{angle::Angle, geonum_mod::Geonum};

pub trait Projection: Sized {
    /// view data through a specific projection path
    /// conventional: complex lens functions with higher-order abstractions
    /// geonum: encode projection paths as angles for direct access
    fn view<T>(&self, data: &T, path_encoder: fn(&T) -> Angle) -> Self;

    /// compose two projections into a single new projection
    /// conventional: nested higher-order functions
    /// geonum: direct angle addition
    fn compose(&self, other: &Self) -> Self;
}

impl Projection for Geonum {
    fn view<T>(&self, data: &T, path_encoder: fn(&T) -> Angle) -> Self {
        let path_angle = path_encoder(data);

        // create a projection by encoding the path as an angle
        Geonum::new_with_angle(self.length, self.angle + path_angle)
    }

    fn compose(&self, other: &Self) -> Self {
        // compose projections by angle addition
        Geonum::new_with_angle(self.length * other.length, self.angle + other.angle)
    }
}
