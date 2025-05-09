//! # geonum
//!
//! geometric number library supporting unlimited dimensions with O(1) complexity
//!
//! ## features
//!
//! this crate provides optional features:
//!
//! - `optics` - ray tracing, lens operations and optical transformations
//! - `projection` - view transformations and projections
//! - `manifold` - manifold operations and transformations
//!
//! no features are enabled by default, only core functionality

// mod declarations first - these tell Rust about the module structure
mod dimensions;
mod geonum_mod; // avoids name collision with crate
mod multivector;
pub mod traits;

// re-export all primary types
pub use dimensions::Dimensions;
pub use geonum_mod::{Activation, Geonum, EPSILON, TWO_PI, VACUUM_IMPEDANCE};
pub use multivector::{Grade, Multivector};

// re-export all traits based on features
#[cfg(feature = "manifold")]
pub use traits::Manifold;
#[cfg(feature = "optics")]
pub use traits::Optics;
#[cfg(feature = "projection")]
pub use traits::Projection;
