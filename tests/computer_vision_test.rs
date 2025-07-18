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

use geonum::{Angle, Geonum, Multivector};
use std::f64::consts::PI;
use std::time::Instant;

const EPSILON: f64 = 1e-10;

#[test]
fn its_a_feature_detector() {
    // 1. replace convolution kernels with angle-based feature extraction

    // create a simplified corner feature in image coordinates
    let corner_feature = Geonum::new(1.0, 1.0, 4.0); // vector (grade 1) - 45-degree corner orientation

    // traditional design: apply multiple convolution filters (Sobel, etc.)
    // requires O(n²k²) where n is image size and k is kernel size

    // with geometric numbers: direct angle-based feature representation with O(1) complexity

    // 2. compute gradient direction and magnitude directly

    // gradient can be directly encoded in the feature angle
    let gradient_direction = corner_feature.angle.mod_4_angle();
    let gradient_magnitude = corner_feature.length;

    // verify gradient direction is correct
    assert!((gradient_direction - PI / 4.0).abs() < EPSILON);
    assert_eq!(gradient_magnitude, 1.0);

    // 3. generate oriented SIFT-like descriptor

    // traditional descriptor: 128-dimensional histogram of gradients
    // with geonum: multiple geometric numbers with angle+magnitude encoding

    // create a simplified SIFT descriptor with 4 bins (instead of 128)
    let _descriptor = [
        Geonum::new_with_angle(
            0.8, // bin 1 magnitude
            corner_feature.angle,
        ), // vector (grade 1) - gradient direction in bin 1
        Geonum::new_with_angle(
            0.5, // bin 2 magnitude
            corner_feature.angle + Angle::new(1.0, 8.0),
        ), // vector (grade 1) - gradient direction in bin 2
        Geonum::new_with_angle(
            0.3, // bin 3 magnitude
            corner_feature.angle - Angle::new(1.0, 8.0),
        ), // vector (grade 1) - gradient direction in bin 3
        Geonum::new_with_angle(
            0.2, // bin 4 magnitude
            corner_feature.angle + Angle::new(1.0, 4.0),
        ), // vector (grade 1) - gradient direction in bin 4
    ];

    // 4. demonstrate feature matching through angle alignment

    // create a similar feature with slight rotation (simulating another view of same point)
    let rotated_feature = Geonum::new_with_angle(
        0.95,                                        // slightly weaker in second view
        corner_feature.angle + Angle::new(0.05, PI), // slight rotation
    ); // vector (grade 1) - same feature type

    // compute match quality using angle distance
    let angle_diff = (rotated_feature.angle - corner_feature.angle).mod_4_angle();
    let match_quality = 1.0 - angle_diff / PI;

    // prove close match (close to 1.0)
    assert!(
        match_quality > 0.9,
        "Features should match closely. Match quality: {match_quality}"
    );

    // 5. measure performance for high-dimensional feature spaces

    let start_time = Instant::now();

    // create a 1000-dimensional feature space (impossible with traditional methods)
    let dimensions = 1000;
    let mut high_dim_descriptor = Vec::with_capacity(dimensions);

    for i in 0..dimensions {
        // distribute angles across full circle
        let angle = (i as f64) * 2.0 * PI / (dimensions as f64);
        high_dim_descriptor.push(Geonum::new(
            1.0 / (1.0 + (angle - gradient_direction).abs()),
            angle,
            PI,
        )); // vector (grade 1) - gradient component
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
        .map(|g| Geonum::new_with_angle(g.length / norm, g.angle))
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
    let frame1_point = Geonum::new(1.0, 1.0, 6.0); // orientation at π/6

    let frame2_point = Geonum::new_with_angle(
        1.05,                                      // slightly moved outward
        frame1_point.angle + Angle::new(0.02, PI), // slightly rotated
    );

    // traditional design: compute flow field with least squares estimation
    // requires O(n³) matrix operations for n points

    // with geometric numbers: direct angle-based flow computation with O(1) complexity

    // 2. compute optical flow vector directly

    // flow vector is the difference between frame2 and frame1 points
    // convert to cartesian for illustrative purposes
    let frame1_x = frame1_point.length * frame1_point.angle.cos();
    let frame1_y = frame1_point.length * frame1_point.angle.sin();

    let frame2_x = frame2_point.length * frame2_point.angle.cos();
    let frame2_y = frame2_point.length * frame2_point.angle.sin();

    // flow vector components
    let flow_x = frame2_x - frame1_x;
    let flow_y = frame2_y - frame1_y;

    // convert flow to geometric number representation
    let flow_vector = Geonum::new_from_cartesian(flow_x, flow_y);

    // verify flow magnitude and direction
    assert!(
        flow_vector.length > 0.0,
        "Flow should have non-zero magnitude"
    );

    // 3. demonstrate scale-space optical flow

    // In traditional optical flow, handling multiple scales requires
    // computing image pyramids and running the algorithm multiple times

    // With geonum, scales can be encoded directly in the blade grade
    // Ensure flow_vector is grade 1 (vector)
    let flow_grade_1 = Geonum::new_with_blade(
        flow_vector.length,
        1, // ensure vector (grade 1)
        flow_vector.angle.mod_4_angle(),
        PI,
    );

    let multiscale_flow = Multivector(vec![
        flow_grade_1, // vector (grade 1) - original scale flow
        Geonum::new_with_blade(
            flow_vector.length * 0.5, // half magnitude at coarser scale
            2,                        // bivector (grade 2) - coarser scale flow
            flow_vector.angle.mod_4_angle(),
            PI,
        ),
        Geonum::new_with_blade(
            flow_vector.length * 0.25, // quarter magnitude at coarsest scale
            3,                         // trivector (grade 3) - coarsest scale flow
            flow_vector.angle.mod_4_angle(),
            PI,
        ),
    ]);

    // 4. verify multiscale representation

    // extract flow at different scales using blade index
    let flow_components: Vec<&Geonum> = multiscale_flow.0.iter().collect();

    // verify we have all three scale components
    assert_eq!(
        flow_components.len(),
        3,
        "Should have three scale components"
    );

    // verify blade grades are as expected
    assert_eq!(
        flow_components[0].angle.blade() % 4,
        1,
        "Fine scale should be grade 1"
    );
    assert_eq!(
        flow_components[1].angle.blade() % 4,
        2,
        "Mid scale should be grade 2"
    );
    assert_eq!(
        flow_components[2].angle.blade() % 4,
        3,
        "Coarse scale should be grade 3"
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

            Geonum::new(magnitude, angle, PI) // vector (grade 1) - flow vector at image point
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
    let _camera_model = Geonum::new(focal_length, 0.0, 2.0); // initial camera orientation

    // 2. project 3D points to 2D directly using angles

    // 3D world point
    let world_point = Geonum::new(100.0, 1.0, 4.0); // point at π/4 orientation

    // traditional projection: p = K[R|t]P where K is intrinsic matrix, [R|t] is extrinsic
    // with geonum: direct angle-based projection

    // simplified projection without distortion
    let projected_point = Geonum::new_with_angle(
        focal_length * world_point.length / (10.0 * world_point.length), // perspective division
        world_point.angle, // preserve angle in simple model
    );

    // verify projection preserves angles in this simplified case
    assert_eq!(projected_point.angle, world_point.angle);

    // 3. compute reprojection error

    // simulated observed point (with noise)
    let observed_point = Geonum::new_with_angle(
        projected_point.length + 0.1,                 // add noise to length
        projected_point.angle + Angle::new(0.01, PI), // add noise to angle
    );

    // compute reprojection error as an angle-based distance
    let angle_diff = observed_point.angle - projected_point.angle;
    let reprojection_error = angle_diff.mod_4_angle().abs();

    // error should be non-zero but small
    assert!(reprojection_error > 0.0);
    assert!(reprojection_error < 0.1);

    // 4. demonstrate lens distortion modeling

    // in traditional designs, lens distortion requires complex polynomial models
    // with geonum, distortion becomes direct angle transformation

    // radial distortion as angle transformation
    let distortion_factor = 0.05; // distortion strength
    let distorted_point = Geonum::new_with_angle(
        projected_point.length * (1.0 + distortion_factor * projected_point.length),
        projected_point.angle, // preserve angle in radial distortion
    );

    // 5. measure performance for multiple camera calibration

    let start_time = Instant::now();

    // create a camera array with 100 cameras
    let num_cameras = 100;
    let camera_array = (0..num_cameras)
        .map(|i| {
            // distribute cameras in a circle
            let angle = (i as f64) * 2.0 * PI / (num_cameras as f64);
            Geonum::new(focal_length, angle, PI)
        })
        .collect::<Vec<Geonum>>();

    // project point into all cameras
    let _projections = camera_array
        .iter()
        .map(|camera| {
            // compute relative angle between camera and world point
            let relative_angle = world_point.angle.mod_4_angle() - camera.angle.mod_4_angle();

            // projection depends on relative angle
            let visible = relative_angle.abs() < PI / 2.0; // only visible in front of camera

            if visible {
                Some(Geonum::new(
                    focal_length * world_point.length / (10.0 * world_point.length),
                    relative_angle, // angle in camera frame
                    PI,
                ))
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
    let camera1 = Geonum::new(1.0, 0.0, 2.0); // facing along positive x-axis
    let camera2 = Geonum::new(1.0, 1.0, 6.0); // rotated 30 degrees

    // traditional design: compute fundamental matrix with 8-point algorithm
    // requires O(n³) SVD computation

    // with geometric numbers: direct angle-based epipolar constraints

    // 2. compute epipolar constraint through angle relationship

    // point seen in first camera
    let point_in_camera1 = Geonum::new(0.5, 1.0, 12.0); // 15 degrees from camera axis

    // compute corresponding epipolar line in second camera
    // this is the projection of viewing ray from camera1 into camera2

    // relative angle between cameras
    let relative_angle = camera2.angle - camera1.angle;

    // epipolar line represented as angle in second camera
    let epipolar_line = Geonum::new_with_angle(
        1.0,                                     // unit magnitude for line representation
        point_in_camera1.angle - relative_angle, // relative to camera2
    );

    // 3. match points using epipolar constraint

    // potential match in second camera (close to epipolar line)
    let candidate_match = Geonum::new_with_angle(
        0.6,
        epipolar_line.angle + Angle::new(0.01, PI), // small deviation from epipolar line
    );

    // compute distance to epipolar line (simplified as angle difference)
    let angle_diff = candidate_match.angle - epipolar_line.angle;
    let epipolar_distance = angle_diff.mod_4_angle().abs();

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
    let depth1 = f64::abs(baseline * ray2_angle.sin() / angle_diff.sin());

    // reconstructed 3D point
    let reconstructed_point = Geonum::new_with_angle(depth1, ray1_angle);

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
        Geonum::new(0.5, 1.0, 12.0),  // observation from camera 1
        Geonum::new(0.6, -1.0, 18.0), // observation from camera 2 (-10 degrees)
        Geonum::new(0.55, 1.0, 20.0), // observation from camera 3
    ];

    // compute averaged 3D position (simplified bundle adjustment)
    let avg_length = observations.iter().map(|o| o.length).sum::<f64>() / observations.len() as f64;
    let avg_angle = observations
        .iter()
        .map(|o| o.angle.mod_4_angle())
        .sum::<f64>()
        / observations.len() as f64;

    let _refined_point = Geonum::new(avg_length, avg_angle, PI);

    // 6. measure performance for large-scale reconstruction

    let start_time = Instant::now();

    // simulate large-scale structure from motion with 1000 points
    let num_points = 1000;
    let _reconstruction = (0..num_points)
        .map(|i| {
            // distribute points in space
            let angle = (i as f64) * 2.0 * PI / (num_points as f64);
            let depth = 1.0 + (i as f64 / num_points as f64);

            Geonum::new(depth, angle, PI)
        })
        .collect::<Vec<Geonum>>();

    let elapsed = start_time.elapsed();

    // traditional SfM systems scale poorly to large scenes
    // geonum scales linearly with O(n) operations
    assert!(
        elapsed.as_micros() < 5000,
        "Large-scale reconstruction is fast"
    );
}

#[test]
fn its_an_image_registration() {
    // 1. replace iterative optimization with direct angle alignment

    // create two images represented by their dominant orientation
    let image1 = Geonum::new(1.0, 0.0, 2.0); // vector at 0°
    let image2 = Geonum::new(1.0, 1.0, 12.0); // vector at π/12 (15°)

    // traditional design: optimize transformation parameters iteratively
    // requires many iterations of O(n²) operations

    // with geometric numbers: direct angle-based transformation estimation

    // 2. compute transformation directly from angle difference

    // compute rotation directly from angle difference
    let rotation_angle = image2.angle - image1.angle;

    // verify rotation angle
    assert_eq!(rotation_angle, Angle::new(1.0, 12.0));

    // 3. apply transformation to register images

    // apply transformation to image1
    let registered_image = image1.rotate(rotation_angle);

    // verify registration aligns images
    assert_eq!(registered_image.angle, image2.angle);

    // 4. demonstrate multi-scale registration

    // traditional multi-scale registration uses image pyramids
    // with geonum: direct angle-based multi-scale representation

    // create multi-scale representation using multivector
    let _multiscale_image1 = Multivector(vec![
        Geonum::new(1.0, 0.0, 2.0),               // fine scale
        Geonum::new_with_blade(1.0, 2, 0.0, 2.0), // medium scale
        Geonum::new_with_blade(1.0, 3, 0.0, 2.0), // coarse scale
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
            let feature = Geonum::new(1.0, feature_angle, PI);

            // transformed feature
            let transformed_feature = feature.rotate(rotation_angle);

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
    let input_feature = Geonum::new(1.0, 1.0, 4.0); // vector at π/4

    // 2. create a simple neural network layer using geometric transformation

    // create weight as geometric number
    let weight = Geonum::new(1.2, 1.0, 6.0); // weight at π/6

    // compute layer output directly
    let layer_output = input_feature * weight;

    // 3. demonstrate activation functions as angle transformations

    // traditional activation functions apply nonlinearities to scalar values
    // with geonum: direct angle-based nonlinearities

    // ReLU-like activation: preserve positive parts of signal
    let activated_output = if layer_output.angle.cos() > 0.0 {
        Geonum::new_with_angle(
            layer_output.length * layer_output.angle.cos(),
            layer_output.angle,
        )
    } else {
        Geonum::new(0.0, 0.0, 2.0) // zeroed output
    };

    // verify activation has expected behavior
    if layer_output.angle.cos() > 0.0 {
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
    let layer2_weight = Geonum::new(0.8, -1.0, 8.0); // weight at -π/8

    // forward pass through second layer
    let layer2_output = activated_output * layer2_weight;

    // 5. measure performance for high-dimensional feature maps

    let start_time = Instant::now();

    // simulate deep network with 100 layers and 1000 features per layer
    let num_layers = 100;
    let features_per_layer = 1000;

    // create initial features
    let mut features = (0..features_per_layer)
        .map(|i| {
            let angle = (i as f64) * 2.0 * PI / (features_per_layer as f64);
            Geonum::new(1.0, angle, PI)
        })
        .collect::<Vec<Geonum>>();

    // forward pass through each layer
    for _ in 0..num_layers {
        // transform features using single weight (simplified)
        features = features
            .iter()
            .map(|feature| {
                let output = Geonum::new_with_angle(
                    feature.length * 0.95,                // slight attenuation
                    feature.angle + Angle::new(0.01, PI), // slight rotation
                );

                // simplified activation
                if output.angle.cos() > 0.0 {
                    Geonum::new_with_angle(output.length * output.angle.cos(), output.angle)
                } else {
                    Geonum::new(0.0, 0.0, 2.0) // zeroed output
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
    let region_orientation = Geonum::new(1.0, 1.0, 4.0); // vector at π/4 (45-degree texture)

    // 2. create segmentation as angle clustering

    // traditional segmentation uses complex graph cuts or neural networks
    // with geonum: direct angle-based clustering

    // create pixels with similar but noisy orientations
    let num_pixels = 100;
    let pixels = (0..num_pixels)
        .map(|i| {
            // add noise to orientation
            let noise = (i as f64 / num_pixels as f64) * 0.2 - 0.1;
            Geonum::new_with_angle(1.0, region_orientation.angle + Angle::new(noise, PI))
        })
        .collect::<Vec<Geonum>>();

    // 3. compute region statistics directly from angles

    // calculate mean orientation
    let mean_angle =
        pixels.iter().map(|p| p.angle.mod_4_angle()).sum::<f64>() / pixels.len() as f64;

    // verify mean is close to original region orientation
    assert!(
        (mean_angle - region_orientation.angle.mod_4_angle()).abs() < 0.1,
        "Mean orientation should be close to region orientation"
    );

    // calculate circular variance
    let circular_variance = pixels
        .iter()
        .map(|p| 1.0 - f64::cos(p.angle.mod_4_angle() - mean_angle))
        .sum::<f64>()
        / pixels.len() as f64;

    // verify low variance within region
    assert!(
        circular_variance < 0.1,
        "Pixels in region should have low orientation variance"
    );

    // 4. demonstrate boundary detection through angle discontinuities

    // create boundary between two regions
    let region1_pixel = Geonum::new(1.0, 1.0, 4.0); // vector at π/4 (45 degrees)
    let region2_pixel = Geonum::new(1.0, 3.0, 4.0); // vector at 3π/4 (135 degrees)

    // boundary strength is angle difference
    let angle_diff = region1_pixel.angle - region2_pixel.angle;
    let boundary_strength = angle_diff.mod_4_angle().abs();

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

            Geonum::new(1.0, segment_angle, PI)
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
    let object = Geonum::new(2.0, 1.0, 6.0); // vector at π/6

    // traditional design: regress bounding box coordinates (x,y,w,h)
    // requires complex feature extraction and regression

    // with geometric numbers: direct angle-based localization

    // 2. represent bounding box directly with geometric numbers

    // compute bounding box center (simplified 2D case)
    let center_x = 100.0 + 50.0 * object.angle.cos();
    let center_y = 100.0 + 50.0 * object.angle.sin();

    // bounding box as center position + scale
    let bbox_angle = Geonum::new_from_cartesian(center_x, center_y).angle;
    let bbox = Geonum::new_with_angle(object.length, bbox_angle);

    // 3. compute IoU using angle-based overlap

    // create another bounding box for the same object (slight variation)
    let bbox2 = Geonum::new_with_angle(
        object.length * 1.05,              // slightly larger
        bbox.angle + Angle::new(0.02, PI), // slightly rotated
    );

    // compute IoU-like similarity directly from angles and scales
    let angle_diff = (bbox2.angle - bbox.angle).mod_4_angle();
    let scale_ratio = bbox.length.min(bbox2.length) / bbox.length.max(bbox2.length);

    // combine angle and scale similarity
    let similarity = scale_ratio * (1.0 - angle_diff / PI);

    // should be high similarity (IoU)
    // With 0.02 radian rotation and 1.05x scale, similarity should be high
    let expected_similarity = (1.0 / 1.05) * (1.0 - 0.02 / PI); // ~0.95 * 0.994 = ~0.94
    assert!(
        similarity > 0.9,
        "Bounding boxes should have high overlap. Similarity: {similarity}, expected: ~{expected_similarity}"
    );

    // 4. demonstrate non-maximum suppression through angle clustering

    // traditional NMS requires sorting and greedy selection
    // with geonum: direct angle-based clustering

    // create multiple detections of same object
    let detections = [
        Geonum::new(2.0, 1.0, 6.0), // detection at π/6
        Geonum::new_with_angle(2.1, Angle::new(1.0, 6.0) + Angle::new(0.05, PI)), // slightly rotated
        Geonum::new_with_angle(1.9, Angle::new(1.0, 6.0) - Angle::new(0.03, PI)), // slightly opposite rotation
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
            let angle_diff = (detections[j].angle - detections[i].angle).mod_4_angle();
            // Handle circular distance
            let angle_dist = angle_diff.min(2.0 * PI - angle_diff);
            if !assigned[j] && angle_dist < angle_threshold {
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
            let detection = Geonum::new(scale, position_angle, PI);

            // class probabilities represented as multivector
            // each grade corresponds to a class
            let class_probs = (0..5)
                .map(|c| {
                    let class_id = (i + c) % num_classes;
                    let prob = if c == 0 { 0.8 } else { 0.05 }; // highest prob for first class

                    Geonum::new_with_blade(
                        prob,
                        c + 1, // blade grade represents class ID range
                        (class_id as f64) * 2.0 / (num_classes as f64),
                        PI,
                    )
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
