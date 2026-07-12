// timekeeping is winding counting
//
// the SI second is DEFINED as 9,192,631,770 turns of cesium's hyperfine
// phase — time's base unit is a winding count. no float phase register can
// hold what that definition demands: a year of atomic time is 2.9×10^17
// turns, a count past f64's exact-integer range before the year even starts,
// and at 1.8×10^18 radians the float's next representable instant sits ~40
// cycles away — four and a half nanoseconds that cannot be expressed, 1.3
// meters of GPS, a whole tick vanishing into rounding bit-identically. the
// accumulated float clock ends the year ~87 μs adrift — ~60× the telecom
// PTP sync budget — while the blade count is exact. the industry hand-rolls the
// fix: NTP eras, PTP's 96-bit second+fraction registers — an integer count
// beside a fraction, blade and t shipped as a protocol instead of arithmetic
//
//   - a year added one minute at a time (525,600 additions) lands the blade
//     on the exact integer 4 × 31,536,000 × 9,192,631,770 — every cycle of
//     the year counted, none rounded
//   - two clocks one tick apart: four blades apart as geonums, bit-identical
//     as floats
//   - after a 10^17-turn year the sub-cycle readout is as sharp as at t = 0:
//     the angle's resolution never ages, while a float's degrades with every
//     order of magnitude it climbs
//
// run: cargo test --test atomic_clock_test -- --show-output

use geonum::*;
use std::f64::consts::PI;

const CESIUM_HZ: u64 = 9_192_631_770; // the SI second, by definition
const MINUTES_PER_YEAR: u64 = 525_600; // 365 days, measured in minutes
const SECONDS_PER_YEAR: u64 = 31_536_000;

// the year's true cycle count, exact in integer arithmetic
fn year_cycles() -> u128 {
    SECONDS_PER_YEAR as u128 * CESIUM_HZ as u128
}

#[test]
fn it_counts_every_cesium_cycle_of_a_year() {
    // one minute of atomic time: 60 × 9,192,631,770 cycles, four blades each —
    // an exact integer handed to the blade, no radians constructed
    let one_minute = Angle::new_with_blade((4 * 60 * CESIUM_HZ) as usize, 0.0, 1.0);

    let mut clock = Angle::new(0.0, 1.0);
    let mut float_phase = 0.0_f64; // the conventional register, run alongside
    for _ in 0..MINUTES_PER_YEAR {
        clock = clock + one_minute;
        float_phase += 2.0 * PI * (60 * CESIUM_HZ) as f64;
    }

    // the blade lands the year exactly: 1.16×10^18 quarter-turns, an integer
    // the test verifies against u128 arithmetic — a count too large for f64
    // to even state (2.9×10^17 cycles > 2^53)
    assert_eq!(
        clock.blade() as u128,
        4 * year_cycles(),
        "every cycle of the year counted, none rounded"
    );
    assert!(
        clock.near_rem(0.0),
        "and nothing spilled into the fraction — whole cycles stay whole"
    );

    // the float register fogged: its next representable instant is dozens of
    // cycles away, and its accumulated count has drifted off the true one
    let fog_cycles = (float_phase.next_up() - float_phase) / (2.0 * PI);
    assert!(
        fog_cycles > 10.0,
        "the float cannot resolve its own tick: next instant {fog_cycles:.0} cycles away"
    );
    let drift_cycles = (float_phase / (2.0 * PI) - year_cycles() as f64).abs();
    assert!(
        drift_cycles > 1.0,
        "the accumulated float count lost whole cycles: {drift_cycles:.0} adrift"
    );

    // in the timing industry's units: the float clock ends the year tens of
    // microseconds adrift — the telecom PTP budget is 1.5 μs, GPS wants
    // nanoseconds — while the blade clock is exact
    let drift_seconds = drift_cycles / CESIUM_HZ as f64;
    assert!(
        drift_seconds > 1e-5,
        "the float clock broke microsecond sync: {:.1} μs adrift",
        drift_seconds * 1e6
    );
    eprintln!(
        "year end: blade exact; float fog {fog_cycles:.0} cycles ({:.1} ns), drift {drift_cycles:.0} cycles ({:.1} μs)",
        fog_cycles / CESIUM_HZ as f64 * 1e9,
        drift_seconds * 1e6
    );
}

#[test]
fn it_distinguishes_clocks_one_tick_apart() {
    // two cesium clocks at year end, clock b one cycle ahead — the smallest
    // disagreement two atomic clocks can have
    let year_blade = (4 * year_cycles()) as usize;
    let clock_a = Angle::new_with_blade(year_blade, 0.0, 1.0);
    let clock_b = clock_a + Angle::new_with_blade(4, 0.0, 1.0);

    assert_eq!(
        clock_b.blade() - clock_a.blade(),
        4,
        "one tick resolved exactly — four quarter-turns of blade"
    );

    // the float registers absorb the tick: adding one full cycle to the
    // year-end phase rounds back to the same bits. the two clocks read as
    // one clock — the disagreement is unrepresentable
    let phase_a = year_cycles() as f64 * 2.0 * PI;
    let phase_b = phase_a + 2.0 * PI;
    assert!(
        phase_b == phase_a,
        "the float clocks are bit-identical — the tick vanished into rounding"
    );
}

#[test]
fn it_keeps_subcycle_resolution_after_a_petaturn_year() {
    // advance the year-end clock by 3/8 of a cycle and read the sub-cycle
    // position: it matches a newborn clock's reading exactly — same base
    // angle, same t, bit for bit. the winding never taxed the fraction
    let year_blade = (4 * year_cycles()) as usize;
    let aged = Angle::new_with_blade(year_blade, 0.0, 1.0) + Angle::new(3.0, 4.0);
    let newborn = Angle::new(3.0, 4.0);

    assert_eq!(
        aged.base_angle(),
        newborn.base_angle(),
        "the sub-cycle readout after 10^17 turns matches the newborn clock exactly"
    );

    // the float's resolution aged eighteen orders of magnitude over the same
    // year: its representable step grew from femtoradians to hundreds of
    // radians. the angle's step never moved
    let phase_old = year_cycles() as f64 * 2.0 * PI;
    let phase_new = 0.75 * PI;
    let aging = (phase_old.next_up() - phase_old) / (phase_new.next_up() - phase_new);
    assert!(
        aging > 1e15,
        "float resolution degraded {aging:.1e}× over the year — the blade+t register did not age"
    );
}
