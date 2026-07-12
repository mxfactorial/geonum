// circular statistics is the patch, geonum is the fix
//
// statistics needed a special subfield — circular statistics, with its own
// journals, its own mean, its own variance — because scalar statistics breaks
// on angles: average the bearings 359° and 1° arithmetically and the answer
// points due south. the subfield exists to repair what dropping the angle
// broke. store the angle and nothing needs repair:
//
//   - the circular mean is wave_sum normalized — interference does the
//     averaging, and there is no wraparound to patch because there is no wrap
//   - the mean resultant length R is wave_sum over total_magnitude — the
//     interference gap 1 − R IS the dispersion, the same gap geocollection
//     reads between coherent and incoherent superposition
//   - the winding scalar records lose (the np.unwrap ceremony) never left:
//     a vehicle that circled twice reads the same compass heading with blade 8
//     stored — the odometer is the blade
//
// run: cargo test --test directional_stats_test -- --show-output

use geonum::*;

#[test]
fn it_averages_359_and_1_to_north_not_south() {
    // the textbook failure case: bearings 359° and 1°, both a degree off north
    let bearings: GeoCollection = [359.0, 1.0]
        .iter()
        .map(|&deg| Geonum::new_with_angle(1.0, Angle::new(deg, 180.0)))
        .collect();

    let mean = bearings.wave_sum();

    // the interference average points north, magnitude 2·cos(1°) — barely
    // shy of full coherence
    assert!(
        mean.angle.base_angle().near(&Angle::new(0.0, 1.0)),
        "the circular mean is north"
    );
    assert!(
        mean.near_mag(2.0 * Angle::new(1.0, 180.0).cos_sin().0),
        "resultant = 2·cos(1°)"
    );

    // the scalar mean (359 + 1)/2 = 180 points due south — not slightly
    // wrong, exactly backwards. the arithmetic mean of two angles a degree
    // apart lands a half turn away
    let scalar_mean = Angle::new((359.0 + 1.0) / 2.0, 180.0);
    assert!(
        scalar_mean.is_opposite(&mean.angle.base_angle()),
        "the scalar method points exactly backwards"
    );
}

#[test]
fn it_reads_dispersion_off_the_interference_gap() {
    // the mean resultant length R = |wave_sum| / total_magnitude measures
    // concentration: R = 1 fully coherent, R = 0 fully dispersed. the
    // circular variance 1 − R is the interference gap — for a tight cluster
    // of half-width scale δ it computes to δ² (the small-angle limit the
    // von mises distribution linearizes to)
    let delta = 0.02;
    let heading = 1.0; // π/4 base heading — the cluster center
    let cluster: GeoCollection = (-2..=2)
        .map(|k| {
            Geonum::new_with_angle(
                1.0,
                Angle::new(heading / 4.0 + k as f64 * delta / std::f64::consts::PI, 1.0),
            )
        })
        .collect();

    let r = cluster.wave_sum().mag / cluster.total_magnitude();
    assert!(
        (1.0 - r - delta * delta).abs() < 1e-7,
        "circular variance 1 − R = δ² for the {{−2δ..2δ}} cluster: {:.2e}",
        1.0 - r
    );

    // full dispersion: the four cardinal directions interfere to nothing —
    // R = 0, maximum circular variance, no mean direction exists
    let dispersed: GeoCollection = (0..4)
        .map(|k| Geonum::new_with_angle(1.0, Angle::new(k as f64, 2.0)))
        .collect();
    assert!(
        dispersed.wave_sum().near_mag(0.0),
        "uniform directions cancel — R = 0, dispersion total"
    );
}

#[test]
fn it_keeps_the_winding_the_compass_wraps_away() {
    // a vehicle turns through two full laps in π/8 increments. every compass
    // reading along the way lives in [0°, 360°) — the time series has to be
    // "unwrapped" (the np.unwrap ceremony) to recover total rotation, and one
    // missed sample aliases a whole lap away. the blade never wrapped, so
    // there is nothing to unwrap
    let start = Angle::new(1.0, 6.0); // heading π/6
    let step = Angle::new(1.0, 8.0); // π/8 per turn increment

    let mut heading = start;
    for _ in 0..32 {
        heading = heading + step; // 32 × π/8 = 4π — two laps
    }

    assert_eq!(
        heading.base_angle(),
        start.base_angle(),
        "the compass reads the same heading it started with"
    );
    assert_eq!(
        heading.blade() - start.blade(),
        8,
        "while the blade stored both laps"
    );
    assert_eq!(
        (heading.blade() - start.blade()) / 4,
        2,
        "the lap count is the winding — the odometer is the blade"
    );
}
