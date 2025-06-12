//! trait definitions for the geonum crate
//!
//! defines the traits that define various geometric algebra operations

// conditionally re-export trait modules based on features

#[cfg(feature = "optics")]
pub mod optics;
#[cfg(feature = "optics")]
pub use optics::Optics;

#[cfg(feature = "projection")]
pub mod projection;
#[cfg(feature = "projection")]
pub use projection::Projection;

#[cfg(feature = "manifold")]
pub mod manifold;
#[cfg(feature = "manifold")]
pub use manifold::Manifold;

#[cfg(feature = "ml")]
pub mod machine_learning;
#[cfg(feature = "ml")]
pub use machine_learning::{Activation, MachineLearning};
