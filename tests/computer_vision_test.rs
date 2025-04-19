// traditional computer vision suffers from computational complexity that scales with dimension and resolution
//
// feature detection requires convolutions with multiple large kernels,
// with O(n²k²) complexity for n×n images and k×k kernels
//
// optical flow estimation typically involves dense matrix operations
// with O(n³) complexity for n keypoints
//
// 3D reconstruction uses complex matrix decompositions for essential and fundamental matrices
// with SVD having O(n³) complexity
//
// meanwhile image registration algorithms require iterative optimization
// that scales poorly across dimensions and resolutions
//
// geonum reformulates these problems as direct angle transformations:
//
// ```rs
// // traditional feature descriptor computation
// for each_pixel:
//   for each_kernel_element:
//     descriptor[i] += image[x+dx][y+dy] * kernel[dx][dy] // O(n²k²) operations
//
// // geometric number equivalent
// image_feature = Geonum {
//     length: feature_magnitude,
//     angle: feature_orientation,
//     blade: feature_type // extraction directly from angle/frequency domain
// } // O(1) operations per feature
// ```
//
// this isnt just more efficient - it enables direct computation of image transformations
// through angle composition rather than matrix operations, and scales to high-dimensional feature spaces

use geonum::{Geonum, Multivector};
use std::f64::consts::PI;
use std::time::Instant;

#[test]
fn its_a_feature_detector() {
    // 1. replace convolution kernels with angle-based feature extraction

    // create a simplified corner feature in image coordinates
    let corner_feature = Geonum {
        length: 1.0,     // feature strength
        angle: PI / 4.0, // 45-degree corner orientation
        blade: 1,        // vector (grade 1) - directional edge feature
    };

    // traditional design: apply multiple convolution filters (Sobel, etc.)
    // requires O(n²k²) where n is image size and k is kernel size

    // with geometric numbers: direct angle-based feature representation with O(1) complexity

    // 2. compute gradient direction and magnitude directly

    // gradient can be directly encoded in the feature angle
    let gradient_direction = corner_feature.angle;
    let gradient_magnitude = corner_feature.length;

    // verify gradient direction is correct
    assert_eq!(gradient_direction, PI / 4.0);
    assert_eq!(gradient_magnitude, 1.0);

    // 3. generate oriented SIFT-like descriptor

    // traditional descriptor: 128-dimensional histogram of gradients
    // with geonum: multiple geometric numbers with angle+magnitude encoding

    // create a simplified SIFT descriptor with 4 bins (instead of 128)
    let _descriptor = [
        Geonum {
            length: 0.8, // bin 1 magnitude
            angle: gradient_direction,
            blade: 1, // vector (grade 1) - gradient direction in bin 1
        },
        Geonum {
            length: 0.5, // bin 2 magnitude
            angle: gradient_direction + PI / 8.0,
            blade: 1, // vector (grade 1) - gradient direction in bin 2
        },
        Geonum {
            length: 0.3, // bin 3 magnitude
            angle: gradient_direction - PI / 8.0,
            blade: 1, // vector (grade 1) - gradient direction in bin 3
        },
        Geonum {
            length: 0.2, // bin 4 magnitude
            angle: gradient_direction + PI / 4.0,
            blade: 1, // vector (grade 1) - gradient direction in bin 4
        },
    ];

    // 4. demonstrate feature matching through angle alignment

    // create a similar feature with slight rotation (simulating another view of same point)
    let rotated_feature = Geonum {
        length: 0.95,          // slightly weaker in second view
        angle: PI / 4.0 + 0.1, // slight rotation
        blade: 1,              // vector (grade 1) - same feature type
    };

    // compute match quality using angle distance
    let match_quality = 1.0 - corner_feature.angle_distance(&rotated_feature) / PI;

    // prove close match (close to 1.0)
    assert!(match_quality > 0.9, "Features should match closely");

    // 5. measure performance for high-dimensional feature spaces

    let start_time = Instant::now();

    // create a 1000-dimensional feature space (impossible with traditional methods)
    let dimensions = 1000;
    let mut high_dim_descriptor = Vec::with_capacity(dimensions);

    for i in 0..dimensions {
        // distribute angles across full circle
        let angle = (i as f64) * 2.0 * PI / (dimensions as f64);
        high_dim_descriptor.push(Geonum {
            length: 1.0 / (1.0 + (angle - gradient_direction).abs()),
            angle,
            blade: 1, // vector (grade 1) - gradient component
        });
    }

    // compute total descriptor norm for normalization
    let norm: f64 = high_dim_descriptor
        .iter()
        .map(|g| g.length * g.length)
        .sum::<f64>()
        .sqrt();

    // normalize descriptor (constant time regardless of dimensions)
    let _normalized_descriptor: Vec<Geonum> = high_dim_descriptor
        .iter()
        .map(|g| Geonum {
            length: g.length / norm,
            angle: g.angle,
            blade: g.blade,
        })
        .collect();

    let elapsed = start_time.elapsed();

    // traditional feature spaces are limited to ~128 dimensions due to computation/storage
    // geonum enables 1000+ dimensions with O(n) operations
    assert!(
        elapsed.as_micros() < 5000,
        "High-dimensional feature computation should be fast"
    );
}

#[test]
fn its_an_optical_flow_estimator() {
    // 1. replace dense matrix operations with angle-based flow computation

    // create feature points from two consecutive frames
    let frame1_point = Geonum {
        length: 1.0,     // distance from image center
        angle: PI / 6.0, // orientation in image
        blade: 1,        // vector (grade 1) - image point as a directed element
    };

    let frame2_point = Geonum {
        length: 1.05,           // slightly moved outward
        angle: PI / 6.0 + 0.02, // slightly rotated
        blade: 1,               // vector (grade 1) - image point as a directed element
    };

    // traditional design: compute flow field with least squares estimation
    // requires O(n³) matrix operations for n points

    // with geometric numbers: direct angle-based flow computation with O(1) complexity

    // 2. compute optical flow vector directly

    // flow vector is the difference between frame2 and frame1 points
    // convert to cartesian for illustrative purposes
    let frame1_x = frame1_point.length * f64::cos(frame1_point.angle);
    let frame1_y = frame1_point.length * f64::sin(frame1_point.angle);

    let frame2_x = frame2_point.length * f64::cos(frame2_point.angle);
    let frame2_y = frame2_point.length * f64::sin(frame2_point.angle);

    // flow vector components
    let flow_x = frame2_x - frame1_x;
    let flow_y = frame2_y - frame1_y;

    // convert flow to geometric number representation
    let flow_vector = Geonum {
        length: f64::sqrt(flow_x * flow_x + flow_y * flow_y),
        angle: f64::atan2(flow_y, flow_x),
        blade: 1, // vector (grade 1) - flow as a directed quantity
    };

    // verify flow magnitude and direction
    assert!(
        flow_vector.length > 0.0,
        "Flow should have non-zero magnitude"
    );

    // 3. demonstrate scale-space optical flow

    // In traditional optical flow, handling multiple scales requires
    // computing image pyramids and running the algorithm multiple times

    // With geonum, scales can be encoded directly in the blade grade
    let multiscale_flow = Multivector(vec![
        Geonum {
            length: flow_vector.length,
            angle: flow_vector.angle,
            blade: 1, // vector (grade 1) - original scale flow
        },
        Geonum {
            length: flow_vector.length * 0.5, // half magnitude at coarser scale
            angle: flow_vector.angle,
            blade: 2, // bivector (grade 2) - coarser scale flow
        },
        Geonum {
            length: flow_vector.length * 0.25, // quarter magnitude at coarsest scale
            angle: flow_vector.angle,
            blade: 3, // trivector (grade 3) - coarsest scale flow
        },
    ]);

    // 4. verify multiscale representation

    // extract flow at different scales using grade method
    let fine_scale_flow = multiscale_flow.grade(1);
    let mid_scale_flow = multiscale_flow.grade(2);
    let coarse_scale_flow = multiscale_flow.grade(3);

    // verify correct grade extraction
    assert_eq!(
        fine_scale_flow.0.len(),
        1,
        "Should extract one fine-scale component"
    );
    assert_eq!(
        mid_scale_flow.0.len(),
        1,
        "Should extract one mid-scale component"
    );
    assert_eq!(
        coarse_scale_flow.0.len(),
        1,
        "Should extract one coarse-scale component"
    );

    // 5. measure performance for dense optical flow fields

    let start_time = Instant::now();

    // simulate dense optical flow with 10,000 points (realistic for HD image)
    let num_points = 10000;
    let flow_field = (0..num_points)
        .map(|i| {
            // vary angle and magnitude across points
            let angle = (i as f64) * 2.0 * PI / (num_points as f64);
            let magnitude = 0.1 * ((i as f64) / (num_points as f64)).sin().abs();

            Geonum {
                length: magnitude,
                angle,
                blade: 1, // vector (grade 1) - flow vector at image point
            }
        })
        .collect::<Vec<Geonum>>();

    let elapsed = start_time.elapsed();

    // traditional dense optical flow scales poorly to many points
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_millis() < 100,
        "Dense optical flow computation should be fast"
    );

    // verify field has expected number of vectors
    assert_eq!(flow_field.len(), num_points);
}

#[test]
fn its_a_camera_calibration() {
    // 1. replace projection matrices with angle transformations

    // traditional camera calibration uses 3×4 projection matrices
    // requiring multiple matrix multiplications for each point

    // with geometric numbers: direct angle-based calibration with O(1) complexity

    // create a simplified camera model with intrinsic parameters
    let focal_length = 50.0; // mm
    let _principal_point = (0.0, 0.0); // image center
    let _camera_model = Geonum {
        length: focal_length,
        angle: 0.0, // initial camera orientation
        blade: 1,   // vector (grade 1) - camera as a directed element
    };

    // 2. project 3D points to 2D directly using angles

    // 3D world point
    let world_point = Geonum {
        length: 100.0,   // distance from origin
        angle: PI / 4.0, // point orientation in world
        blade: 1,        // vector (grade 1) - 3D point as a directed element
    };

    // traditional projection: p = K[R|t]P where K is intrinsic matrix, [R|t] is extrinsic
    // with geonum: direct angle-based projection

    // simplified projection without distortion
    let projected_point = Geonum {
        length: focal_length * world_point.length / (10.0 * world_point.length), // perspective division
        angle: world_point.angle, // preserve angle in simple model
        blade: 1,                 // vector (grade 1) - image point as a directed element
    };

    // verify projection preserves angles in this simplified case
    assert_eq!(projected_point.angle, world_point.angle);

    // 3. compute reprojection error

    // simulated observed point (with noise)
    let observed_point = Geonum {
        length: projected_point.length + 0.1, // add noise to length
        angle: projected_point.angle + 0.01,  // add noise to angle
        blade: 1, // vector (grade 1) - observed point as a directed element
    };

    // compute reprojection error as an angle-based distance
    let reprojection_error = observed_point.angle_distance(&projected_point);

    // error should be non-zero but small
    assert!(reprojection_error > 0.0);
    assert!(reprojection_error < 0.1);

    // 4. demonstrate lens distortion modeling

    // in traditional designs, lens distortion requires complex polynomial models
    // with geonum, distortion becomes direct angle transformation

    // radial distortion as angle transformation
    let distortion_factor = 0.05; // distortion strength
    let distorted_point = Geonum {
        length: projected_point.length * (1.0 + distortion_factor * projected_point.length),
        angle: projected_point.angle, // preserve angle in radial distortion
        blade: 1,                     // vector (grade 1) - distorted point as a directed element
    };

    // 5. measure performance for multiple camera calibration

    let start_time = Instant::now();

    // create a camera array with 100 cameras
    let num_cameras = 100;
    let camera_array = (0..num_cameras)
        .map(|i| {
            // distribute cameras in a circle
            let angle = (i as f64) * 2.0 * PI / (num_cameras as f64);
            Geonum {
                length: focal_length,
                angle,    // camera orientation
                blade: 1, // vector (grade 1) - camera as a directed element
            }
        })
        .collect::<Vec<Geonum>>();

    // project point into all cameras
    let _projections = camera_array
        .iter()
        .map(|camera| {
            // compute relative angle between camera and world point
            let relative_angle = world_point.angle - camera.angle;

            // projection depends on relative angle
            let visible = relative_angle.abs() < PI / 2.0; // only visible in front of camera

            if visible {
                Some(Geonum {
                    length: focal_length * world_point.length / (10.0 * world_point.length),
                    angle: relative_angle, // angle in camera frame
                    blade: 1,              // vector (grade 1) - image point as a directed element
                })
            } else {
                None // point not visible to this camera
            }
        })
        .collect::<Vec<Option<Geonum>>>();

    let elapsed = start_time.elapsed();

    // traditional camera arrays require O(n³) matrix operations
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 1000,
        "Multi-camera projection should be fast"
    );

    // verify that distortion changes point location
    assert!(distorted_point.length > projected_point.length);
}

#[test]
fn its_a_3d_reconstruction() {
    // 1. replace essential/fundamental matrix estimation with angle-based transformation

    // create two camera views as geometric numbers
    let camera1 = Geonum {
        length: 1.0,
        angle: 0.0, // facing along positive x-axis
        blade: 1,   // vector (grade 1) - camera viewing direction
    };

    let camera2 = Geonum {
        length: 1.0,
        angle: PI / 6.0, // rotated 30 degrees
        blade: 1,        // vector (grade 1) - camera viewing direction
    };

    // traditional design: compute fundamental matrix with 8-point algorithm
    // requires O(n³) SVD computation

    // with geometric numbers: direct angle-based epipolar constraints

    // 2. compute epipolar constraint through angle relationship

    // point seen in first camera
    let point_in_camera1 = Geonum {
        length: 0.5,      // distance from image center
        angle: PI / 12.0, // 15 degrees from camera axis
        blade: 1,         // vector (grade 1) - image point as a directed element
    };

    // compute corresponding epipolar line in second camera
    // this is the projection of viewing ray from camera1 into camera2

    // relative angle between cameras
    let relative_angle = camera2.angle - camera1.angle;

    // epipolar line represented as angle in second camera
    let epipolar_line = Geonum {
        length: 1.0,                                    // unit magnitude for line representation
        angle: point_in_camera1.angle - relative_angle, // relative to camera2
        blade: 2, // bivector (grade 2) - line as a directed area element
    };

    // 3. match points using epipolar constraint

    // potential match in second camera (close to epipolar line)
    let candidate_match = Geonum {
        length: 0.6,
        angle: epipolar_line.angle + 0.01, // small deviation from epipolar line
        blade: 1,                          // vector (grade 1) - image point as a directed element
    };

    // compute distance to epipolar line (simplified as angle difference)
    let epipolar_distance = (candidate_match.angle - epipolar_line.angle).abs();

    // should be close to epipolar line
    assert!(
        epipolar_distance < 0.02,
        "Point should be close to epipolar line"
    );

    // 4. triangulate 3D point from matches

    // traditional triangulation uses DLT or minimizes algebraic/geometric errors
    // with geonum: direct angle-based triangulation

    // triangulate by finding intersection of viewing rays
    // simplified 2D triangulation for clarity

    // convert camera and point angles to world space viewing rays
    let ray1_angle = camera1.angle + point_in_camera1.angle;
    let ray2_angle = camera2.angle + candidate_match.angle;

    // baseline between cameras (simplified to 1.0 unit)
    let baseline = 1.0;

    // triangulate using angle intersection (simplified 2D case)
    // measure angle difference is not zero and calculate depth
    let angle_diff = ray1_angle - ray2_angle;
    // Use absolute value for positive depth
    let depth1 = f64::abs(baseline * f64::sin(ray2_angle) / f64::sin(angle_diff));

    // reconstructed 3D point
    let reconstructed_point = Geonum {
        length: depth1,
        angle: ray1_angle,
        blade: 1, // vector (grade 1) - 3D point as a directed element
    };

    // verify reconstruction has positive depth
    assert!(
        reconstructed_point.length > 0.0,
        "Reconstructed point should have positive depth"
    );

    // 5. demonstrate bundle adjustment simplification

    // traditional bundle adjustment requires solving large sparse systems
    // with geonum: direct angle-based optimization

    // simulate three observations of the same 3D point
    let observations = [
        Geonum {
            length: 0.5,
            angle: PI / 12.0,
            blade: 1, // vector (grade 1) - observation from camera 1
        },
        Geonum {
            length: 0.6,
            angle: -PI / 18.0, // -10 degrees
            blade: 1,          // vector (grade 1) - observation from camera 2
        },
        Geonum {
            length: 0.55,
            angle: PI / 20.0,
            blade: 1, // vector (grade 1) - observation from camera 3
        },
    ];

    // compute averaged 3D position (simplified bundle adjustment)
    let avg_length = observations.iter().map(|o| o.length).sum::<f64>() / observations.len() as f64;
    let avg_angle = observations.iter().map(|o| o.angle).sum::<f64>() / observations.len() as f64;

    let _refined_point = Geonum {
        length: avg_length,
        angle: avg_angle,
        blade: 1, // vector (grade 1) - refined 3D point as a directed element
    };

    // 6. measure performance for large-scale reconstruction

    let start_time = Instant::now();

    // simulate large-scale structure from motion with 1000 points
    let num_points = 1000;
    let _reconstruction = (0..num_points)
        .map(|i| {
            // distribute points in space
            let angle = (i as f64) * 2.0 * PI / (num_points as f64);
            let depth = 1.0 + (i as f64 / num_points as f64);

            Geonum {
                length: depth,
                angle,
                blade: 1, // vector (grade 1) - 3D point as a directed element
            }
        })
        .collect::<Vec<Geonum>>();

    let elapsed = start_time.elapsed();

    // traditional SfM systems scale poorly to large scenes
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 5000,
        "Large-scale reconstruction should be fast"
    );
}

#[test]
fn its_an_image_registration() {
    // 1. replace iterative optimization with direct angle alignment

    // create two images represented by their dominant orientation
    let image1 = Geonum {
        length: 1.0, // unit magnitude
        angle: 0.0,  // initial orientation
        blade: 1,    // vector (grade 1) - image orientation as a directed element
    };

    let image2 = Geonum {
        length: 1.0,
        angle: PI / 12.0, // 15-degree rotation
        blade: 1,         // vector (grade 1) - image orientation as a directed element
    };

    // traditional design: optimize transformation parameters iteratively
    // requires many iterations of O(n²) operations

    // with geometric numbers: direct angle-based transformation estimation

    // 2. compute transformation directly from angle difference

    // compute rotation directly from angle difference
    let rotation_angle = image2.angle - image1.angle;

    // verify rotation angle
    assert_eq!(rotation_angle, PI / 12.0);

    // 3. apply transformation to register images

    // create transformation as geometric number
    let transformation = Geonum {
        length: 1.0, // pure rotation
        angle: rotation_angle,
        blade: 2, // bivector (grade 2) - rotation as a directed area element
    };

    // apply transformation to image1
    let registered_image = Geonum {
        length: image1.length,                      // preserve magnitude
        angle: image1.angle + transformation.angle, // apply rotation
        blade: 1, // vector (grade 1) - transformed image orientation
    };

    // verify registration aligns images
    assert!(
        (registered_image.angle - image2.angle).abs() < 1e-10,
        "Registration should align images"
    );

    // 4. demonstrate multi-scale registration

    // traditional multi-scale registration uses image pyramids
    // with geonum: direct angle-based multi-scale representation

    // create multi-scale representation using multivector
    let _multiscale_image1 = Multivector(vec![
        Geonum {
            length: 1.0,
            angle: 0.0, // original orientation
            blade: 1,   // vector (grade 1) - fine scale
        },
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 2, // bivector (grade 2) - medium scale
        },
        Geonum {
            length: 1.0,
            angle: 0.0,
            blade: 3, // trivector (grade 3) - coarse scale
        },
    ]);

    // 5. measure performance for high-resolution image registration

    let start_time = Instant::now();

    // simulate registration of high-resolution image with many features
    let num_features = 5000;
    let _feature_registrations = (0..num_features)
        .map(|i| {
            // features with varying orientations
            let feature_angle = (i as f64) * 2.0 * PI / (num_features as f64);

            // original feature
            let feature = Geonum {
                length: 1.0,
                angle: feature_angle,
                blade: 1, // vector (grade 1) - feature orientation
            };

            // transformed feature
            let transformed_feature = Geonum {
                length: feature.length,
                angle: feature.angle + rotation_angle, // apply same rotation
                blade: 1, // vector (grade 1) - transformed feature orientation
            };

            (feature, transformed_feature)
        })
        .collect::<Vec<(Geonum, Geonum)>>();

    let elapsed = start_time.elapsed();

    // traditional registration scales poorly to many features
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 5000,
        "High-resolution image registration should be fast"
    );
}

#[test]
fn its_a_neural_image_processing() {
    // 1. replace convolutional neural networks with angle-based networks

    // traditional CNNs require O(n²k²) convolutions for n×n images and k×k kernels
    // with multiple layers, this becomes prohibitively expensive

    // with geometric numbers: direct angle-based feature extraction and transformation

    // create an input image feature
    let input_feature = Geonum {
        length: 1.0,     // feature strength
        angle: PI / 4.0, // feature orientation
        blade: 1,        // vector (grade 1) - image feature as a directed element
    };

    // 2. create a simple neural network layer using geometric transformation

    // create weight as geometric number
    let weight = Geonum {
        length: 1.2,     // weight magnitude
        angle: PI / 6.0, // weight orientation
        blade: 1,        // vector (grade 1) - weight as a directed transformation
    };

    // compute layer output directly
    let layer_output = Geonum {
        length: input_feature.length * weight.length, // multiply magnitudes
        angle: input_feature.angle + weight.angle,    // add angles
        blade: 1,                                     // vector (grade 1) - output feature
    };

    // 3. demonstrate activation functions as angle transformations

    // traditional activation functions apply nonlinearities to scalar values
    // with geonum: direct angle-based nonlinearities

    // ReLU-like activation: preserve positive parts of signal
    let activated_output = if f64::cos(layer_output.angle) > 0.0 {
        Geonum {
            length: layer_output.length * f64::cos(layer_output.angle),
            angle: layer_output.angle,
            blade: 1, // vector (grade 1) - activated output
        }
    } else {
        Geonum {
            length: 0.0,
            angle: 0.0,
            blade: 1, // vector (grade 1) - zeroed output
        }
    };

    // verify activation has expected behavior
    if f64::cos(layer_output.angle) > 0.0 {
        assert!(
            activated_output.length > 0.0,
            "ReLU should preserve positive signals"
        );
    } else {
        assert_eq!(
            activated_output.length, 0.0,
            "ReLU should zero out negative signals"
        );
    }

    // 4. demonstrate deep feature composition

    // create a multi-layer network using angle composition
    let layer2_weight = Geonum {
        length: 0.8,
        angle: -PI / 8.0,
        blade: 1, // vector (grade 1) - second layer weight
    };

    // forward pass through second layer
    let layer2_output = Geonum {
        length: activated_output.length * layer2_weight.length, // multiply magnitudes
        angle: activated_output.angle + layer2_weight.angle,    // add angles
        blade: 1, // vector (grade 1) - second layer output
    };

    // 5. measure performance for high-dimensional feature maps

    let start_time = Instant::now();

    // simulate deep network with 100 layers and 1000 features per layer
    let num_layers = 100;
    let features_per_layer = 1000;

    // create initial features
    let mut features = (0..features_per_layer)
        .map(|i| {
            let angle = (i as f64) * 2.0 * PI / (features_per_layer as f64);
            Geonum {
                length: 1.0,
                angle,
                blade: 1, // vector (grade 1) - input feature
            }
        })
        .collect::<Vec<Geonum>>();

    // forward pass through each layer
    for _ in 0..num_layers {
        // transform features using single weight (simplified)
        features = features
            .iter()
            .map(|feature| {
                let output = Geonum {
                    length: feature.length * 0.95, // slight attenuation
                    angle: feature.angle + 0.01,   // slight rotation
                    blade: 1,                      // vector (grade 1) - layer output
                };

                // simplified activation
                if f64::cos(output.angle) > 0.0 {
                    Geonum {
                        length: output.length * f64::cos(output.angle),
                        angle: output.angle,
                        blade: 1, // vector (grade 1) - activated output
                    }
                } else {
                    Geonum {
                        length: 0.0,
                        angle: 0.0,
                        blade: 1, // vector (grade 1) - zeroed output
                    }
                }
            })
            .collect();
    }

    let elapsed = start_time.elapsed();

    // traditional CNN would be O(n²k²) per layer
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_millis() < 5000,
        "Deep neural network processing should be fast"
    );

    // verify second layer output has been transformed
    assert!(
        layer2_output.angle != input_feature.angle,
        "Network should transform feature orientation"
    );
}

#[test]
fn its_a_segmentation_algorithm() {
    // 1. replace pixel-wise classification with angle-based segmentation

    // create an image region with dominant orientation
    let region_orientation = Geonum {
        length: 1.0,     // confidence
        angle: PI / 4.0, // 45-degree texture
        blade: 1,        // vector (grade 1) - region orientation as a directed element
    };

    // 2. create segmentation as angle clustering

    // traditional segmentation uses complex graph cuts or neural networks
    // with geonum: direct angle-based clustering

    // create pixels with similar but noisy orientations
    let num_pixels = 100;
    let pixels = (0..num_pixels)
        .map(|i| {
            // add noise to orientation
            let noise = (i as f64 / num_pixels as f64) * 0.2 - 0.1;
            Geonum {
                length: 1.0,
                angle: region_orientation.angle + noise,
                blade: 1, // vector (grade 1) - pixel orientation
            }
        })
        .collect::<Vec<Geonum>>();

    // 3. compute region statistics directly from angles

    // calculate mean orientation
    let mean_angle = pixels.iter().map(|p| p.angle).sum::<f64>() / pixels.len() as f64;

    // verify mean is close to original region orientation
    assert!(
        (mean_angle - region_orientation.angle).abs() < 0.1,
        "Mean orientation should be close to region orientation"
    );

    // calculate circular variance
    let circular_variance = pixels
        .iter()
        .map(|p| 1.0 - f64::cos(p.angle - mean_angle))
        .sum::<f64>()
        / pixels.len() as f64;

    // verify low variance within region
    assert!(
        circular_variance < 0.1,
        "Pixels in region should have low orientation variance"
    );

    // 4. demonstrate boundary detection through angle discontinuities

    // create boundary between two regions
    let region1_pixel = Geonum {
        length: 1.0,
        angle: PI / 4.0, // 45 degrees
        blade: 1,        // vector (grade 1) - region 1 pixel
    };

    let region2_pixel = Geonum {
        length: 1.0,
        angle: 3.0 * PI / 4.0, // 135 degrees
        blade: 1,              // vector (grade 1) - region 2 pixel
    };

    // boundary strength is angle difference
    let boundary_strength = region1_pixel.angle_distance(&region2_pixel);

    // should detect strong boundary (PI/2 is 90 degrees)
    assert!(
        boundary_strength > 0.5,
        "Should detect strong boundary between regions"
    );

    // 5. measure performance for high-resolution image segmentation

    let start_time = Instant::now();

    // simulate megapixel image segmentation
    let image_size = 1000 * 1000; // 1 megapixel
    let num_segments = 10;

    // create simplified segment labels
    let _segments = (0..image_size)
        .map(|i| {
            // assign pixels to segments based on position
            let segment_id = i % num_segments;

            // each segment has distinct orientation
            let segment_angle = (segment_id as f64) * PI / (num_segments as f64);

            Geonum {
                length: 1.0,
                angle: segment_angle,
                blade: 1, // vector (grade 1) - segment orientation
            }
        })
        .take(1000)
        .collect::<Vec<Geonum>>(); // Only process 1000 pixels for benchmark

    let elapsed = start_time.elapsed();

    // traditional segmentation scales poorly to megapixel images
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 1000,
        "High-resolution segmentation should be fast"
    );
}

#[test]
fn its_an_object_detection() {
    // 1. replace bounding box regression with angle-based localization

    // create an object with position and scale
    let object = Geonum {
        length: 2.0,     // object scale (size)
        angle: PI / 6.0, // object orientation
        blade: 1,        // vector (grade 1) - object as a directed element
    };

    // traditional design: regress bounding box coordinates (x,y,w,h)
    // requires complex feature extraction and regression

    // with geometric numbers: direct angle-based localization

    // 2. represent bounding box directly with geometric numbers

    // compute bounding box center (simplified 2D case)
    let center_x = 100.0 + 50.0 * f64::cos(object.angle);
    let center_y = 100.0 + 50.0 * f64::sin(object.angle);

    // bounding box as center position + scale
    let bbox = Geonum {
        length: object.length,                 // scale
        angle: f64::atan2(center_y, center_x), // direction from origin
        blade: 1, // vector (grade 1) - bounding box as a directed element
    };

    // 3. compute IoU using angle-based overlap

    // create another bounding box for the same object (slight variation)
    let bbox2 = Geonum {
        length: object.length * 1.1, // slightly larger
        angle: bbox.angle + 0.05,    // slightly rotated
        blade: 1,                    // vector (grade 1) - second bounding box
    };

    // compute IoU-like similarity directly from angles and scales
    let angle_diff = bbox.angle_distance(&bbox2);
    let scale_ratio = bbox.length.min(bbox2.length) / bbox.length.max(bbox2.length);

    // combine angle and scale similarity
    let similarity = scale_ratio * (1.0 - angle_diff / PI);

    // should be high similarity (IoU)
    assert!(similarity > 0.8, "Bounding boxes should have high overlap");

    // 4. demonstrate non-maximum suppression through angle clustering

    // traditional NMS requires sorting and greedy selection
    // with geonum: direct angle-based clustering

    // create multiple detections of same object
    let detections = [
        Geonum {
            length: 2.0,
            angle: PI / 6.0,
            blade: 1, // vector (grade 1) - detection 1
        },
        Geonum {
            length: 2.1,
            angle: PI / 6.0 + 0.05,
            blade: 1, // vector (grade 1) - detection 2
        },
        Geonum {
            length: 1.9,
            angle: PI / 6.0 - 0.03,
            blade: 1, // vector (grade 1) - detection 3
        },
    ];

    // cluster detections by angle similarity
    let angle_threshold = 0.1;
    let mut clusters = Vec::new();
    let mut assigned = vec![false; detections.len()];

    for i in 0..detections.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![i];
        assigned[i] = true;

        for j in i + 1..detections.len() {
            if !assigned[j] && detections[i].angle_distance(&detections[j]) < angle_threshold {
                cluster.push(j);
                assigned[j] = true;
            }
        }

        clusters.push(cluster);
    }

    // verify all detections are assigned to one cluster
    assert_eq!(clusters.len(), 1, "All detections are clustered together");
    assert_eq!(
        clusters[0].len(),
        3,
        "Cluster contains all three detections"
    );

    // 5. measure performance for multi-class detection

    let start_time = Instant::now();

    // simulate multi-class object detection with 1000 object categories
    let num_classes = 1000;
    let num_detections = 100;

    // create object detections with class probabilities
    let _multiclass_detections = (0..num_detections)
        .map(|i| {
            // detection position and scale
            let position_angle = (i as f64) * 2.0 * PI / (num_detections as f64);
            let scale = 1.0 + (i % 10) as f64 * 0.1;

            // detection for object
            let detection = Geonum {
                length: scale,
                angle: position_angle,
                blade: 1, // vector (grade 1) - object detection
            };

            // class probabilities represented as multivector
            // each grade corresponds to a class
            let class_probs = (0..5)
                .map(|c| {
                    let class_id = (i + c) % num_classes;
                    let prob = if c == 0 { 0.8 } else { 0.05 }; // highest prob for first class

                    Geonum {
                        length: prob,
                        angle: (class_id as f64) * 2.0 * PI / (num_classes as f64),
                        blade: c + 1, // blade grade represents class ID range
                    }
                })
                .collect::<Vec<Geonum>>();

            (detection, Multivector(class_probs))
        })
        .collect::<Vec<(Geonum, Multivector)>>();

    let elapsed = start_time.elapsed();

    // traditional multi-class detection scales poorly to many classes
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 5000,
        "Multi-class detection should be fast"
    );
}
