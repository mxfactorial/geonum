//! machine learning trait implementation
//!
//! defines the MachineLearning trait and related functionality for ML modeling

use crate::{geonum_mod::Geonum, Angle};
use std::f64::consts::PI;

/// activation functions for neural networks
///
/// represents different activation functions used in neural networks
/// when applied to a geometric number, these functions transform the magnitude
/// component while preserving the angle component
///
/// # examples
///
/// ```
/// use geonum::{Geonum, Activation, MachineLearning};
///
/// let num = Geonum::new(2.0, 1.0, 4.0); // magnitude 2.0, angle π/4
///
/// // apply relu activation
/// let relu_output = num.activate(Activation::ReLU);
///
/// // apply sigmoid activation
/// let sigmoid_output = num.activate(Activation::Sigmoid);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// rectified linear unit: f(x) = max(0, x)
    ReLU,
    /// sigmoid function: f(x) = 1/(1+e^(-x))
    Sigmoid,
    /// hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// identity function: f(x) = x
    Identity,
}

pub trait MachineLearning: Sized {
    /// creates a geometric number representing a regression line
    /// conventional: least squares matrix operations O(n³)
    /// geonum: direct covariance-to-angle encoding O(1)
    fn regression_from(cov_xy: f64, var_x: f64) -> Self;

    /// updates a weight vector for perceptron learning
    /// conventional: vector arithmetic with multiple operations O(n)
    /// geonum: direct angle-based weight update O(1)
    fn perceptron_update(&self, learning_rate: f64, error: f64, input: &Self) -> Self;

    /// performs a neural network forward pass
    /// conventional: matrix-vector multiplication O(n²)
    /// geonum: geometric product with angle addition O(1)
    fn forward_pass(&self, weight: &Self, bias: &Self) -> Self;

    /// applies an activation function to a geometric number
    /// conventional: element-wise function application O(n)
    /// geonum: single geometric transformation O(1)
    fn activate(&self, activation: Activation) -> Self;
}

impl MachineLearning for Geonum {
    fn regression_from(cov_xy: f64, var_x: f64) -> Self {
        Geonum {
            mag: (cov_xy.powi(2) / var_x).sqrt(),
            angle: Angle::new(cov_xy.atan2(var_x), PI), // convert radians to geometric angle
        }
    }

    fn perceptron_update(&self, learning_rate: f64, error: f64, input: &Geonum) -> Self {
        let input_grade = input.angle.grade();
        let sign_x = if input_grade > 2 { -1.0 } else { 1.0 };
        let angle_update = Angle::new(-learning_rate * error * sign_x / PI, 1.0);

        Geonum {
            mag: self.mag + learning_rate * error * input.mag,
            angle: self.angle + angle_update,
        }
    }

    fn forward_pass(&self, weight: &Geonum, bias: &Geonum) -> Self {
        Geonum {
            mag: self.mag * weight.mag + bias.mag,
            angle: self.angle + weight.angle,
        }
    }

    fn activate(&self, activation: Activation) -> Self {
        match activation {
            Activation::ReLU => Geonum {
                mag: if self.angle.grade_angle().cos() > 0.0 {
                    self.mag
                } else {
                    0.0
                },
                angle: self.angle,
            },
            Activation::Sigmoid => Geonum {
                mag: self.mag / (1.0 + (-self.angle.grade_angle().cos()).exp()),
                angle: self.angle,
            },
            Activation::Tanh => Geonum {
                mag: self.mag * self.angle.grade_angle().cos().tanh(),
                angle: self.angle,
            },
            Activation::Identity => *self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geonum_mod::EPSILON;

    #[test]
    fn it_computes_regression_from_covariance() {
        // test regression computation from covariance and variance
        let cov_xy = 2.0;
        let var_x = 4.0;

        let regression = Geonum::regression_from(cov_xy, var_x);

        // prove magnitude encodes the correlation strength
        let expected_mag = (cov_xy.powi(2) / var_x).sqrt();
        assert!((regression.mag - expected_mag).abs() < EPSILON);

        // prove angle encodes the slope direction
        let expected_angle = Angle::new(cov_xy.atan2(var_x), PI);
        assert_eq!(regression.angle, expected_angle);

        // regression represents the intrinsic angle of the (x,y) relationship
        // not a 2D position but the single geometric angle between variables
        // "no directionless numbers" means no naked x-axis - you start with (x,y)
        // blade=0 indicates this relationship angle (≈0.464 rad) is less than π/2
        assert_eq!(regression.angle.blade(), 0);
    }

    #[test]
    fn it_updates_perceptron_weights() {
        // create initial weight
        let weight = Geonum::new(1.0, 1.0, 4.0); // [1, π/4]

        // create input
        let input = Geonum::new(2.0, 1.0, 6.0); // [2, π/6]

        // apply perceptron update
        let learning_rate = 0.1;
        let error = 0.5;
        let updated_weight = weight.perceptron_update(learning_rate, error, &input);

        // verify weight update follows perceptron rule
        // magnitude is updated by learning_rate * error * input.mag
        let expected_mag = weight.mag + learning_rate * error * input.mag;
        assert!((updated_weight.mag - expected_mag).abs() < EPSILON);

        // angle is updated by learning rule
        let input_grade = input.angle.grade();
        let sign_x = if input_grade > 2 { -1.0 } else { 1.0 };
        let angle_update = Angle::new(-learning_rate * error * sign_x / PI, 1.0);
        let expected_angle = weight.angle + angle_update;
        assert_eq!(updated_weight.angle, expected_angle);

        // grade is preserved
        assert_eq!(updated_weight.angle.grade(), weight.angle.grade());
    }

    #[test]
    fn it_performs_neural_network_operations() {
        // create input, weight, and bias
        let input = Geonum::new(2.0, 1.0, 3.0); // [2, π/3]
        let weight = Geonum::new(1.5, 1.0, 6.0); // [1.5, π/6]
        let bias = Geonum::new(0.5, 0.0, 1.0); // scalar bias

        // forward pass
        let forward_result = input.forward_pass(&weight, &bias);

        // test forward pass computation
        let expected_mag = input.mag * weight.mag + bias.mag;
        assert!((forward_result.mag - expected_mag).abs() < EPSILON);

        let expected_angle = input.angle + weight.angle;
        assert_eq!(forward_result.angle, expected_angle);

        // test activation functions
        let test_input = Geonum::new(1.0, 1.0, 4.0); // [1, π/4]

        // test ReLU activation
        let relu_result = test_input.activate(Activation::ReLU);
        assert!(relu_result.mag > 0.0); // positive input should remain positive

        // test sigmoid activation
        let sigmoid_result = test_input.activate(Activation::Sigmoid);
        assert!(sigmoid_result.mag > 0.0 && sigmoid_result.mag < test_input.mag);

        // test tanh activation
        let tanh_result = test_input.activate(Activation::Tanh);
        assert!(tanh_result.mag.abs() <= test_input.mag);

        // test identity activation
        let identity_result = test_input.activate(Activation::Identity);
        assert_eq!(identity_result.mag, test_input.mag);
        assert_eq!(identity_result.angle, test_input.angle);
        assert_eq!(identity_result.angle.grade(), test_input.angle.grade());
    }
}
