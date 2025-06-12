//! machine learning trait implementation
//!
//! defines the MachineLearning trait and related functionality for ML modeling

use crate::geonum_mod::Geonum;
use std::f64::consts::PI;

/// activation functions for neural networks
///
/// represents different activation functions used in neural networks
/// when applied to a geometric number, these functions transform the length
/// component while preserving the angle component
///
/// # examples
///
/// ```
/// use geonum::{Geonum, Activation, MachineLearning};
///
/// let num = Geonum { length: 2.0, angle: 0.5, blade: 1 };
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
            length: (cov_xy.powi(2) / var_x).sqrt(),
            angle: cov_xy.atan2(var_x),
            blade: 1, // regression line is a vector (grade 1)
        }
    }

    fn perceptron_update(&self, learning_rate: f64, error: f64, input: &Geonum) -> Self {
        let sign_x = if input.angle > PI { -1.0 } else { 1.0 };

        Geonum {
            length: self.length + learning_rate * error * input.length,
            angle: self.angle - learning_rate * error * sign_x,
            blade: self.blade, // preserve blade grade for weight vector
        }
    }

    fn forward_pass(&self, weight: &Geonum, bias: &Geonum) -> Self {
        Geonum {
            length: self.length * weight.length + bias.length,
            angle: self.angle + weight.angle,
            blade: self.with_product_blade(weight).blade, // use product blade rules
        }
    }

    fn activate(&self, activation: Activation) -> Self {
        match activation {
            Activation::ReLU => Geonum {
                length: if self.angle.cos() > 0.0 {
                    self.length
                } else {
                    0.0
                },
                angle: self.angle,
                blade: self.blade, // preserve blade grade
            },
            Activation::Sigmoid => Geonum {
                length: self.length / (1.0 + (-self.angle.cos()).exp()),
                angle: self.angle,
                blade: self.blade, // preserve blade grade
            },
            Activation::Tanh => Geonum {
                length: self.length * self.angle.cos().tanh(),
                angle: self.angle,
                blade: self.blade, // preserve blade grade
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

        // verify length encodes the correlation strength
        let expected_length = (cov_xy.powi(2) / var_x).sqrt();
        assert!((regression.length - expected_length).abs() < EPSILON);

        // verify angle encodes the slope direction
        let expected_angle = cov_xy.atan2(var_x);
        assert!((regression.angle - expected_angle).abs() < EPSILON);

        // verify blade indicates vector nature of regression line
        assert_eq!(regression.blade, 1);
    }

    #[test]
    fn it_updates_perceptron_weights() {
        // create initial weight
        let weight = Geonum {
            length: 1.0,
            angle: PI / 4.0,
            blade: 1,
        };

        // create input
        let input = Geonum {
            length: 2.0,
            angle: PI / 6.0,
            blade: 1,
        };

        // apply perceptron update
        let learning_rate = 0.1;
        let error = 0.5;
        let updated_weight = weight.perceptron_update(learning_rate, error, &input);

        // verify weight update follows perceptron rule
        // length should be updated by learning_rate * error * input.length
        let expected_length = weight.length + learning_rate * error * input.length;
        assert!((updated_weight.length - expected_length).abs() < EPSILON);

        // angle should be updated by learning rule
        let sign_x = if input.angle > PI { -1.0 } else { 1.0 };
        let expected_angle = weight.angle - learning_rate * error * sign_x;
        assert!((updated_weight.angle - expected_angle).abs() < EPSILON);

        // blade should be preserved
        assert_eq!(updated_weight.blade, weight.blade);
    }

    #[test]
    fn it_performs_neural_network_operations() {
        // create input, weight, and bias
        let input = Geonum {
            length: 2.0,
            angle: PI / 3.0,
            blade: 1,
        };
        let weight = Geonum {
            length: 1.5,
            angle: PI / 6.0,
            blade: 1,
        };
        let bias = Geonum {
            length: 0.5,
            angle: 0.0,
            blade: 0,
        };

        // forward pass
        let forward_result = input.forward_pass(&weight, &bias);

        // verify forward pass computation
        let expected_length = input.length * weight.length + bias.length;
        assert!((forward_result.length - expected_length).abs() < EPSILON);

        let expected_angle = input.angle + weight.angle;
        assert!((forward_result.angle - expected_angle).abs() < EPSILON);

        // test activation functions
        let test_input = Geonum {
            length: 1.0,
            angle: PI / 4.0,
            blade: 1,
        };

        // test ReLU activation
        let relu_result = test_input.activate(Activation::ReLU);
        assert!(relu_result.length > 0.0); // positive input should remain positive

        // test sigmoid activation
        let sigmoid_result = test_input.activate(Activation::Sigmoid);
        assert!(sigmoid_result.length > 0.0 && sigmoid_result.length < test_input.length);

        // test tanh activation
        let tanh_result = test_input.activate(Activation::Tanh);
        assert!(tanh_result.length.abs() <= test_input.length);

        // test identity activation
        let identity_result = test_input.activate(Activation::Identity);
        assert_eq!(identity_result.length, test_input.length);
        assert_eq!(identity_result.angle, test_input.angle);
        assert_eq!(identity_result.blade, test_input.blade);
    }
}
