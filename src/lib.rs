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
//! - `ml` - machine learning operations and neural network functions
//! - `em` - electromagnetic field calculations and operations
//! - `waves` - wave propagation and dispersion modeling
//!
//! no features are enabled by default, only core functionality

// mod declarations first - these tell Rust about the module structure
mod dimensions;
mod geonum_mod; // avoids name collision with crate
mod multivector;
pub mod traits;

// re-export all primary types
pub use dimensions::Dimensions;
pub use geonum_mod::{Geonum, EPSILON, TWO_PI};
pub use multivector::{Grade, Multivector};

// re-export all traits based on features
#[cfg(feature = "em")]
pub use traits::Electromagnetics;
#[cfg(feature = "manifold")]
pub use traits::Manifold;
#[cfg(feature = "optics")]
pub use traits::Optics;
#[cfg(feature = "projection")]
pub use traits::Projection;
#[cfg(feature = "waves")]
pub use traits::Waves;
#[cfg(feature = "ml")]
pub use traits::{Activation, MachineLearning};
