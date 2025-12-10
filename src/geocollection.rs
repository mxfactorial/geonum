use crate::{Angle, Geonum};

/// collection of geometric objects for domain-specific operations
///
/// provides base functionality for geometric object collections
/// intended for extension with domain-specific traits
pub struct GeoCollection {
    pub objects: Vec<Geonum>,
}

impl GeoCollection {
    /// creates an empty collection
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// returns the number of geometric objects in the collection
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// returns true if the collection contains no objects
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// returns an iterator over references to the geometric objects
    pub fn iter(&self) -> impl Iterator<Item = &Geonum> {
        self.objects.iter()
    }
}

impl Default for GeoCollection {
    fn default() -> Self {
        Self::new()
    }
}

// implement From trait for creating from Vec
impl From<Vec<Geonum>> for GeoCollection {
    fn from(objects: Vec<Geonum>) -> Self {
        Self { objects }
    }
}

// implement FromIterator for collecting from iterators
impl FromIterator<Geonum> for GeoCollection {
    fn from_iter<T: IntoIterator<Item = Geonum>>(iter: T) -> Self {
        Self {
            objects: iter.into_iter().collect(),
        }
    }
}

// enable indexing into the collection
impl std::ops::Index<usize> for GeoCollection {
    type Output = Geonum;

    fn index(&self, index: usize) -> &Self::Output {
        &self.objects[index]
    }
}

// enable iteration
impl IntoIterator for GeoCollection {
    type Item = Geonum;
    type IntoIter = std::vec::IntoIter<Geonum>;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.into_iter()
    }
}

impl<'a> IntoIterator for &'a GeoCollection {
    type Item = &'a Geonum;
    type IntoIter = std::slice::Iter<'a, Geonum>;

    fn into_iter(self) -> Self::IntoIter {
        self.objects.iter()
    }
}

// implement AsRef for borrowing as Vec
impl AsRef<Vec<Geonum>> for GeoCollection {
    fn as_ref(&self) -> &Vec<Geonum> {
        &self.objects
    }
}

// implement AsRef for borrowing as slice
impl AsRef<[Geonum]> for GeoCollection {
    fn as_ref(&self) -> &[Geonum] {
        &self.objects
    }
}

impl GeoCollection {
    /// removes geometric objects with magnitude below threshold
    ///
    /// useful for filtering out noise, negligible contributions, or
    /// implementing level-of-detail systems
    pub fn truncate(&self, threshold: f64) -> Self {
        Self::from(
            self.objects
                .iter()
                .filter(|g| g.mag > threshold)
                .cloned()
                .collect::<Vec<_>>(),
        )
    }

    /// selects objects within a cone defined by direction and half-angle
    ///
    /// useful for spatial queries, visibility tests, sensor field-of-view
    /// calculations, and directional selection in robotics/graphics
    pub fn select_cone(&self, direction: &Geonum, half_angle: f64) -> Self {
        Self::from(
            self.objects
                .iter()
                .filter(|g| {
                    let magnitude = g.mag * direction.mag;
                    if magnitude == 0.0 {
                        return false;
                    }

                    // cos(θ) = (v1·v2) / (|v1||v2|) with sign encoded in dot.angle
                    let dot = g.dot(direction);
                    let signed_cos = dot.mag / magnitude * dot.angle.project(Angle::new(0.0, 1.0));

                    let angle_between = signed_cos.clamp(-1.0, 1.0).acos();
                    angle_between <= half_angle
                })
                .cloned()
                .collect::<Vec<_>>(),
        )
    }

    /// computes the total magnitude of all objects in the collection
    ///
    /// useful for energy calculations, total force/field strength,
    /// or any domain where magnitudes sum meaningfully
    pub fn total_magnitude(&self) -> f64 {
        self.objects.iter().map(|g| g.mag).sum()
    }

    /// finds the dominant object (largest magnitude) in the collection
    ///
    /// useful for identifying primary contributions, maximum forces,
    /// or most significant elements in physical simulations
    pub fn dominant(&self) -> Option<&Geonum> {
        self.objects
            .iter()
            .max_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap())
    }

    /// scales all objects in the collection by a uniform factor
    ///
    /// useful for unit conversions, coordinate transformations,
    /// or adjusting overall magnitude while preserving relationships
    pub fn scale_all(&self, factor: f64) -> Self {
        Self::from(
            self.objects
                .iter()
                .map(|g| g.scale(factor))
                .collect::<Vec<_>>(),
        )
    }

    /// rotates all objects in the collection by the same angle
    ///
    /// useful for coordinate frame transformations, rotating entire
    /// systems, or applying uniform phase shifts
    pub fn rotate_all(&self, rotation: crate::Angle) -> Self {
        Self::from(
            self.objects
                .iter()
                .map(|g| g.rotate(rotation))
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Angle, Geonum};
    use std::f64::consts::PI;

    #[test]
    fn it_creates_empty_collection() {
        let collection = GeoCollection::new();
        assert!(collection.is_empty());
        assert_eq!(collection.len(), 0);
    }

    #[test]
    fn it_creates_from_vec() {
        let objects = vec![
            Geonum::scalar(1.0),
            Geonum::scalar(2.0),
            Geonum::scalar(3.0),
        ];
        let collection = GeoCollection::from(objects);
        assert_eq!(collection.len(), 3);
    }

    #[test]
    fn it_truncates_small_magnitudes() {
        let collection = GeoCollection::from(vec![
            Geonum::new(0.9, 0.0, 1.0),  // kept
            Geonum::new(0.05, 0.0, 1.0), // filtered out
            Geonum::new(0.3, 1.0, 2.0),  // filtered out
            Geonum::new(0.7, 1.0, 1.0),  // kept
        ]);

        let filtered = collection.truncate(0.5);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.objects[0].mag > 0.5);
        assert!(filtered.objects[1].mag > 0.5);
    }

    #[test]
    fn it_selects_objects_within_cone() {
        let forward = Geonum::new(1.0, 0.0, 1.0); // 0 angle
        let collection = GeoCollection::from(vec![
            Geonum::new(1.0, 0.0, 1.0), // 0 angle - selected
            Geonum::new(1.0, 1.0, 8.0), // π/8 - selected
            Geonum::new(1.0, 1.0, 2.0), // π/2 - not selected
            Geonum::new(1.0, 1.0, 1.0), // π - not selected
        ]);

        let cone = collection.select_cone(&forward, PI / 4.0); // 45° cone
        assert_eq!(cone.len(), 2);
    }

    #[test]
    fn it_computes_total_magnitude() {
        let collection = GeoCollection::from(vec![
            Geonum::new(2.0, 0.0, 1.0),
            Geonum::new(3.0, 1.0, 2.0),
            Geonum::new(5.0, 1.0, 1.0),
        ]);

        let total = collection.total_magnitude();
        assert_eq!(total, 10.0); // 2 + 3 + 5
    }

    #[test]
    fn it_finds_dominant_object() {
        let collection = GeoCollection::from(vec![
            Geonum::new(2.0, 0.0, 1.0),
            Geonum::new(7.0, 1.0, 2.0), // largest
            Geonum::new(3.0, 1.0, 1.0),
        ]);

        let dominant = collection.dominant().unwrap();
        assert_eq!(dominant.mag, 7.0);
    }

    #[test]
    fn it_returns_none_for_empty_dominant() {
        let collection = GeoCollection::new();
        assert!(collection.dominant().is_none());
    }

    #[test]
    fn it_scales_all_objects() {
        let collection = GeoCollection::from(vec![
            Geonum::new(1.0, 0.0, 1.0),
            Geonum::new(2.0, 1.0, 2.0),
            Geonum::new(3.0, 1.0, 1.0),
        ]);

        let scaled = collection.scale_all(2.0);
        assert_eq!(scaled.objects[0].mag, 2.0);
        assert_eq!(scaled.objects[1].mag, 4.0);
        assert_eq!(scaled.objects[2].mag, 6.0);
    }

    #[test]
    fn it_rotates_all_objects() {
        let collection = GeoCollection::from(vec![
            Geonum::new(1.0, 0.0, 1.0), // 0 angle
            Geonum::new(1.0, 1.0, 4.0), // π/4 angle
        ]);

        let rotation = Angle::new(1.0, 2.0); // π/2
        let rotated = collection.rotate_all(rotation);

        // angles should be increased by π/2
        assert_eq!(rotated.objects[0].angle, Angle::new(1.0, 2.0));
        assert_eq!(rotated.objects[1].angle, Angle::new(3.0, 4.0)); // π/4 + π/2 = 3π/4
    }

    #[test]
    fn it_normalizes_collection() {
        let collection =
            GeoCollection::from(vec![Geonum::new(3.0, 0.0, 1.0), Geonum::new(4.0, 1.0, 2.0)]);

        let total = collection.total_magnitude(); // 7.0
        let normalized = collection.scale_all(1.0 / total);

        assert!((normalized.total_magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn it_handles_cone_selection_with_wrapped_angles() {
        let forward = Geonum::new(1.0, 0.0, 1.0); // 0 angle
        let collection = GeoCollection::from(vec![
            Geonum::new(1.0, 0.0, 1.0),               // 0 angle - selected
            Geonum::new_with_blade(1.0, 8, 0.0, 1.0), // 2π (blade 8, same as 0) - selected
            Geonum::new(1.0, 1.0, 1.0),               // π - not selected
        ]);

        let cone = collection.select_cone(&forward, PI / 4.0);
        assert_eq!(cone.len(), 2); // both 0 and 2π angles selected
    }

    #[test]
    fn it_iterates_over_collection() {
        let collection = GeoCollection::from(vec![
            Geonum::scalar(1.0),
            Geonum::scalar(2.0),
            Geonum::scalar(3.0),
        ]);

        let sum: f64 = collection.iter().map(|g| g.mag).sum();
        assert_eq!(sum, 6.0);
    }

    #[test]
    fn it_indexes_into_collection() {
        let collection = GeoCollection::from(vec![Geonum::scalar(10.0), Geonum::scalar(20.0)]);

        assert_eq!(collection[0].mag, 10.0);
        assert_eq!(collection[1].mag, 20.0);
    }

    #[test]
    fn it_converts_to_vec_reference() {
        let collection = GeoCollection::from(vec![Geonum::scalar(1.0)]);
        let vec_ref: &Vec<Geonum> = collection.as_ref();
        assert_eq!(vec_ref.len(), 1);
    }

    #[test]
    fn it_converts_to_slice() {
        let collection = GeoCollection::from(vec![Geonum::scalar(1.0), Geonum::scalar(2.0)]);
        let slice: &[Geonum] = collection.as_ref();
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn it_collects_from_iterator() {
        let collection: GeoCollection = (0..5).map(|i| Geonum::scalar(i as f64)).collect();

        assert_eq!(collection.len(), 5);
        assert_eq!(collection[2].mag, 2.0);
    }

    #[test]
    fn it_preserves_angles_during_scaling() {
        let angle = Angle::new(1.0, 3.0); // π/3
        let collection = GeoCollection::from(vec![Geonum::new_with_angle(2.0, angle)]);

        let scaled = collection.scale_all(3.0);
        assert_eq!(scaled.objects[0].angle, angle); // angle unchanged
        assert_eq!(scaled.objects[0].mag, 6.0); // magnitude scaled
    }

    #[test]
    fn it_handles_empty_collection_operations() {
        let empty = GeoCollection::new();

        assert_eq!(empty.total_magnitude(), 0.0);
        assert_eq!(empty.truncate(1.0).len(), 0);
        assert_eq!(empty.scale_all(2.0).len(), 0);
        assert_eq!(empty.rotate_all(Angle::new(1.0, 2.0)).len(), 0);
    }
}
