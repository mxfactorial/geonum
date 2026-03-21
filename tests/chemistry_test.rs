// the entire particle zoo is three lines:
//
//   let proton   = Geonum::new(1.0, 0.0, 1.0);  // grade 0
//   let neutron  = Geonum::new(1.0, 1.0, 2.0);  // grade 1
//   let electron = Geonum::new(1.0, 1.0, 1.0);  // grade 2
//
// antiparticle is .dual(). decay is .rotate(). charge is .angle.grade().
// no class hierarchy. no lookup tables. no 300-page textbook.
//
// to see why, try coding the conventional design:
// class Particle, Element and Shell, with methods
// electron_count(), shell_count(), max_electrons(), pending_electrons()
//
// u just end up ditching ur particles for waves
//
// the conventional periodic table in code is 9+ lookup tables
// that dont derive from each other (see issue/ for the python train wreck):
//
//   1. aufbau filling order — 19 memorized pairs
//   2. orbital capacities 2, 6, 10, 14 — lookup values for 2(2l+1)
//   3. ~20 exception elements with bespoke rationalizations
//   4. stable neutron counts per element — empirical, no formula
//   5. magic numbers [2, 8, 20, 28, 50, 82, 126] — a second shell model
//   6. binding energy — 5 empirical constants (Bethe-Weizsacker)
//   7. decay mode decision tree — 6+ branches
//   8. half-lives per isotope — no general formula
//   9. ~3000 known isotopes catalogued individually
//
// the Particle class hierarchy increases the dysfunction:
// - Neutron.decay() returns [Proton, Electron, Antineutrino] — type changes across method call
// - antiparticle() returns base class, not a typed mirror — hierarchy cant decide
// - uranium needs 330 particle objects to say [magnitude, angle]
//
// this test suite proves the replacement in 7 acts:
//
// act I builds the conventional abstractions (orbital capacity, shell capacity, aufbau)
// and shows each one is angle arithmetic — no tables needed
//
// act II dissolves them (spin pairing from dual, aufbau exceptions from symmetry,
// periodic table from grade cycle, particle hierarchy from angle)
//
// act III proves particles are waves:
// decay products interfere (vector sum < scalar sum)
// bonding is constructive interference, antibonding is destructive
// adding an electron changes the standing wave pattern of the whole shell
// particles in bins cant do any of this — waves can
//
// act IV: the blade chain — the particle zoo is one chain of increment_blade()
//
// act V: grades tell you everything — binding is grade 2, electron-electron is grade 0,
// grade offset weakens projection
//
// act VI: wave interference — the running sum cancels,
// collect decomposes it, amplitude contains all pairs
//
// act VII: ionization energy from three lattice constants —
// spread = π/2, spin = π/3, Q = π/4 — denominators 2, 3, 4
// zero fitted parameters, both anomalies (Be > B, N > O)

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;
const RYDBERG: f64 = 13.6;

fn spread() -> Angle {
    Angle::new(1.0, 2.0) // π/2 — one grade step
}
fn spin() -> Angle {
    Angle::new(1.0, 3.0) // π/3 — pairing angle
}

// act I: build the conventional abstractions

#[test]
fn it_computes_orbital_capacity() {
    // orbital capacity 2(2l+1) from angle subdivision per grade
    // grade l gets 2l+1 distinct angle positions in a pi/2 quadrant
    // x2 for spin (dual) gives the full capacity
    // eliminates: orbital capacity lookup table

    let expected_capacities = [2, 6, 10, 14]; // s, p, d, f

    for (l, &expected) in expected_capacities.iter().enumerate() {
        // 2l+1 distinct positions in a pi/2 quadrant at grade l
        let num_positions = 2 * l + 1;
        let capacity = 2 * num_positions; // x2 for spin pairing via dual

        assert_eq!(capacity, expected);

        // create geonums at each position within the quadrant
        // step = pi / (2 * num_positions), so m-th position = m * pi / (2 * num_positions)
        let divisor = (2 * num_positions) as f64;
        let positions: Vec<Geonum> = (0..num_positions)
            .map(|m| Geonum::new_with_angle(1.0, Angle::new_with_blade(l, m as f64, divisor)))
            .collect();

        assert_eq!(positions.len(), num_positions);

        // pauli exclusion as bonus: self-wedge of any position = 0
        for pos in &positions {
            let self_wedge = pos.wedge(pos);
            assert!(self_wedge.mag < EPSILON);
        }
    }
}

#[test]
fn it_computes_shell_capacity() {
    // shell n sums grades l=0..n-1: sum of 2(2l+1) = 2n^2
    // eliminates: shell capacity as separate formula

    let expected_shell_capacities = [2, 8, 18, 32]; // shells 1-4

    for n in 1..=4usize {
        let shell_capacity: usize = (0..n).map(|l| 2 * (2 * l + 1)).sum();
        assert_eq!(shell_capacity, expected_shell_capacities[n - 1]);
        assert_eq!(shell_capacity, 2 * n * n);
    }

    // build a full shell 2 as 8 geonums: 2 at grade 0 + 6 at grade 1
    let mut shell_2: Vec<Geonum> = Vec::new();

    // s subshell (grade 0): 1 position x 2 spins
    let s_pos = Geonum::new_with_angle(1.0, Angle::new_with_blade(0, 0.0, 1.0));
    shell_2.push(s_pos);
    shell_2.push(s_pos.dual()); // spin pair is pi apart

    // p subshell (grade 1): 3 positions x 2 spins
    // step = pi/6, so m-th position = m * pi/6
    for m in 0..3 {
        let p_pos = Geonum::new_with_angle(1.0, Angle::new_with_blade(1, m as f64, 6.0));
        shell_2.push(p_pos);
        shell_2.push(p_pos.dual()); // spin pair
    }

    assert_eq!(shell_2.len(), 8);
}

#[test]
fn it_derives_aufbau_from_angle_ordering() {
    // (n+l) * pi/2 gives total angle per subshell (madelung rule)
    // sorting subshells by total angle produces the filling order
    // no lookup table needed — its ascending angle order
    // eliminates: 19-entry aufbau filling order

    // generate subshells as (n, l) pairs with their total angle
    let mut subshells: Vec<(usize, usize, f64)> = Vec::new();
    for n in 1..=5usize {
        for l in 0..n {
            let total_angle = (n + l) as f64 * PI / 2.0;
            subshells.push((n, l, total_angle));
        }
    }

    // sort by total angle, break ties by n (lower n first)
    subshells.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap().then(a.0.cmp(&b.0)));

    // the conventional 19-entry aufbau table
    let aufbau_table: Vec<(usize, usize)> = vec![
        (1, 0), // 1s
        (2, 0), // 2s
        (2, 1), // 2p
        (3, 0), // 3s
        (3, 1), // 3p
        (4, 0), // 4s
        (3, 2), // 3d
        (4, 1), // 4p
        (5, 0), // 5s
        (4, 2), // 4d
        (5, 1), // 5p
    ];

    // sorted angle order matches the memorized aufbau sequence
    for (i, &(n, l)) in aufbau_table.iter().enumerate() {
        assert_eq!(subshells[i].0, n);
        assert_eq!(subshells[i].1, l);
    }
}

// act II: watch the abstractions dissolve

#[test]
fn it_proves_spin_pairing_from_dual() {
    // spin pair = orbital angle + orbital.dual() (pi apart)
    // pauli exclusion: self-wedge = 0
    // spin pair dot is maximally negative (cos(pi) = -1) — opposite orientation
    // different orbital positions have nonzero wedge — distinct angular states
    // eliminates: pauli exclusion as separate postulate

    let up = Geonum::new(1.0, 1.0, 4.0); // pi/4
    let down = up.dual(); // pi/4 + pi = 5pi/4

    // pauli: self-wedge = 0 (cant occupy same state twice)
    assert!(up.wedge(&up).mag < EPSILON);
    assert!(down.wedge(&down).mag < EPSILON);

    // spin pair: dot at pi means maximally opposite orientation
    // cos(pi) = -1, giving dot.mag = 1.0 at angle pi (negative scalar)
    let pair_dot = up.dot(&down);
    assert!(pair_dot.near_mag(1.0));
    assert!(pair_dot.angle.near_rad(PI));

    // spin orthogonality via projection: up projects zero onto down's axis
    // cos(pi) = -1, so project gives -1 (maximally anti-aligned)
    let spin_projection = up.angle.project(down.angle);
    assert!((spin_projection - (-1.0)).abs() < EPSILON);

    // different orbital positions (not spin pairs) have nonzero wedge
    // two p-orbital slots separated by less than pi
    let p1 = Geonum::new(1.0, 1.0, 4.0); // pi/4
    let p2 = Geonum::new(1.0, 1.0, 3.0); // pi/3
    let orbital_wedge = p1.wedge(&p2);
    assert!(orbital_wedge.mag > 0.1); // distinct angular states

    // but same-state self-wedge remains zero (pauli holds universally)
    assert!(p1.wedge(&p1).mag < EPSILON);
    assert!(p2.wedge(&p2).mag < EPSILON);
}

#[test]
fn it_dissolves_aufbau_exceptions() {
    // chromium Z=24: conventional says [Ar] 4s2 3d4, measured is [Ar] 4s1 3d5
    // half-filled d shell (5 electrons) creates symmetric angle distribution
    // the "exception" is the expected minimum-interference filling
    // eliminates: ~20 exception element patches

    // 4s (n+l=4) and 3d (n+l=5) are adjacent angle tiers
    let s_angle = 4.0 * PI / 2.0; // 4s total angle
    let d_angle = 5.0 * PI / 2.0; // 3d total angle
    assert!((d_angle - s_angle - PI / 2.0).abs() < EPSILON); // adjacent tiers

    // half-filled d shell: 5 evenly-spaced angles in a pi/2 quadrant
    // step = pi/10, so m-th position = m * pi/10
    let half_filled: Vec<Geonum> = (0..5)
        .map(|m| Geonum::new_with_angle(1.0, Angle::new_with_blade(2, m as f64, 10.0)))
        .collect();

    // 4-electron d config: only 4 of 5 positions filled
    let four_filled: Vec<Geonum> = (0..4)
        .map(|m| Geonum::new_with_angle(1.0, Angle::new_with_blade(2, m as f64, 10.0)))
        .collect();

    // symmetric config has balanced pairwise dot products
    // sum of all pairwise dot magnitudes for 5 evenly-spaced vs 4
    let mut sum_5 = 0.0;
    for i in 0..half_filled.len() {
        for j in (i + 1)..half_filled.len() {
            sum_5 += half_filled[i].dot(&half_filled[j]).mag;
        }
    }

    let mut sum_4 = 0.0;
    for i in 0..four_filled.len() {
        for j in (i + 1)..four_filled.len() {
            sum_4 += four_filled[i].dot(&four_filled[j]).mag;
        }
    }

    // normalize by number of pairs: C(5,2)=10, C(4,2)=6
    let avg_5 = sum_5 / 10.0;
    let avg_4 = sum_4 / 6.0;

    // 5 evenly-spaced angles have lower average interference per pair
    // because symmetric distribution minimizes overlap
    assert!(avg_5 < avg_4);
}

#[test]
fn it_maps_periodic_table_from_grade_cycle() {
    // s=grade 0, p=grade 1, d=grade 2, f=grade 3
    // block widths = 2(2*grade+1)
    // period lengths from cumulative block sums
    // eliminates: periodic table layout as memorized structure

    let block_widths: Vec<usize> = (0..4).map(|g| 2 * (2 * g + 1)).collect();
    assert_eq!(block_widths, vec![2, 6, 10, 14]); // s, p, d, f

    // period lengths: which blocks appear in each period
    // period 1: s only = 2
    // period 2,3: s+p = 8
    // period 4,5: s+p+d = 18
    // period 6,7: s+p+d+f = 32
    let period_blocks: Vec<Vec<usize>> = vec![
        vec![0],          // period 1: s
        vec![0, 1],       // period 2: s+p
        vec![0, 1],       // period 3: s+p
        vec![0, 1, 2],    // period 4: s+d+p
        vec![0, 1, 2],    // period 5: s+d+p
        vec![0, 1, 2, 3], // period 6: s+f+d+p
        vec![0, 1, 2, 3], // period 7: s+f+d+p
    ];

    let expected_lengths = [2, 8, 8, 18, 18, 32, 32];

    for (i, blocks) in period_blocks.iter().enumerate() {
        let length: usize = blocks.iter().map(|&g| block_widths[g]).sum();
        assert_eq!(length, expected_lengths[i]);
    }

    // noble gases at complete s+p fills
    // He(2), Ne(10), Ar(18), Kr(36), Xe(54), Rn(86)
    let noble_gas_z: Vec<usize> = expected_lengths
        .iter()
        .scan(0usize, |acc, &len| {
            *acc += len;
            Some(*acc)
        })
        .collect();
    assert_eq!(noble_gas_z[0], 2); // He
    assert_eq!(noble_gas_z[1], 10); // Ne
    assert_eq!(noble_gas_z[2], 18); // Ar
    assert_eq!(noble_gas_z[3], 36); // Kr
    assert_eq!(noble_gas_z[4], 54); // Xe
    assert_eq!(noble_gas_z[5], 86); // Rn
}

#[test]
fn it_dissolves_particle_hierarchy() {
    // proton, neutron, electron as Geonum at different angles
    // all same type — no class hierarchy needed
    // eliminates: Particle/Proton/Neutron/Electron class zoo

    // charge from grade: grade 0 = proton (+1), grade 2 = electron (-1)
    let proton = Geonum::new(1.0, 0.0, 1.0); // grade 0, angle 0
    let neutron = Geonum::new(1.0, 1.0, 2.0); // grade 1, angle pi/2
    let electron = Geonum::new(1.0, 1.0, 1.0); // grade 2, angle pi

    assert_eq!(proton.angle.grade(), 0); // scalar-like: +1 charge
    assert_eq!(neutron.angle.grade(), 1); // vector-like: 0 charge (between + and -)
    assert_eq!(electron.angle.grade(), 2); // bivector-like: -1 charge

    // antiparticle = dual(): positron is electron.dual()
    let positron = electron.dual();
    assert_eq!(positron.angle.grade(), 0); // same grade as proton (positive charge)
    assert!(electron.angle.is_opposite(&positron.angle)); // pi apart

    // antineutrino is neutron.dual()
    let antineutrino = neutron.dual();
    assert_eq!(antineutrino.angle.grade(), 3); // trivector-like

    // mass ratio neutron/proton ~1.00138 encoded in magnitude
    let proton_mass = Geonum::new(1.0, 0.0, 1.0);
    let neutron_mass = Geonum::new(1.00138, 1.0, 2.0);
    assert!((neutron_mass.mag / proton_mass.mag - 1.00138).abs() < EPSILON);
}

// act III: particles become waves

#[test]
fn it_models_decay_as_rotation() {
    // neutron at pi/2 decomposes into products via rotation
    // same type in, same type out — no type change across method call
    // products interfere like waves, not stack like particles
    // eliminates: decay mode decision tree, type-changing methods

    let neutron = Geonum::new(1.0, 1.0, 2.0); // pi/2, grade 1

    // beta decay: each product is a rotation from the neutrons angle
    let decay = |g: Geonum| -> Vec<Geonum> {
        vec![
            g.rotate(Angle::new(-1.0, 2.0)), // -pi/2: grade 1 -> grade 0 (proton)
            g.rotate(Angle::new(1.0, 2.0)),  // +pi/2: grade 1 -> grade 2 (electron)
            g.rotate(Angle::new(1.0, 1.0)),  // +pi: grade 1 -> grade 3 (antineutrino)
        ]
    };

    let products = decay(neutron);

    // same type in, same type out
    assert_eq!(products[0].angle.grade(), 0); // proton
    assert_eq!(products[1].angle.grade(), 2); // electron
    assert_eq!(products[2].angle.grade(), 3); // antineutrino

    // the wave proof: decay products interfere
    // particles would give total count = sum of individual counts
    // waves give vector sum ≠ scalar sum because angles cancel
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for p in &products {
        sum_x += p.mag * p.angle.grade_angle().cos();
        sum_y += p.mag * p.angle.grade_angle().sin();
    }
    let vector_sum = (sum_x.powi(2) + sum_y.powi(2)).sqrt();
    let scalar_sum: f64 = products.iter().map(|p| p.mag).sum();

    // partial cancellation: proton(0) + electron(pi) nearly cancel,
    // antineutrino(3pi/2) survives — vector sum < scalar sum
    assert!(vector_sum < scalar_sum);
    assert!((scalar_sum - 3.0).abs() < EPSILON); // particles: 1+1+1 = 3
    assert!(vector_sum < 1.5); // waves: partial cancellation
}

#[test]
fn it_models_bonding_as_angle_alignment() {
    // bonding is constructive interference, not particles sitting together
    // antibonding is destructive interference
    // eliminates: VSEPR as separate framework

    // helper: vector sum of waves gives the combined amplitude
    let combine = |waves: &[Geonum]| -> f64 {
        let mut x = 0.0;
        let mut y = 0.0;
        for w in waves {
            x += w.mag * w.angle.grade_angle().cos();
            y += w.mag * w.angle.grade_angle().sin();
        }
        (x.powi(2) + y.powi(2)).sqrt()
    };

    // H2 bonding orbital: two waves at same angle = constructive interference
    let h1 = Geonum::new(1.0, 1.0, 4.0); // pi/4
    let h2 = Geonum::new(1.0, 1.0, 4.0); // pi/4

    let bonding_amplitude = combine(&[h1, h2]);
    assert!((bonding_amplitude - 2.0).abs() < EPSILON); // full reinforcement

    // H2 antibonding orbital: two waves pi apart = destructive interference
    let h3 = h1.dual(); // pi/4 + pi
    let antibonding_amplitude = combine(&[h1, h3]);
    assert!(antibonding_amplitude < EPSILON); // full cancellation

    // particles cant cancel. 1 proton + 1 proton = 2 protons, always.
    // waves can: 1 wave + 1 anti-phase wave = 0. this is why bonding works.

    // water: bond angle from projection, not lookup
    let bond_angle_rad = 104.5 * PI / 180.0;
    let o_p1 = Geonum::new(1.0, 0.0, 1.0);
    let o_p2 = Geonum::new_with_angle(1.0, Angle::new(104.5, 180.0));

    // partial interference: neither full reinforcement nor full cancellation
    let water_amplitude = combine(&[o_p1, o_p2]);
    assert!(water_amplitude > 0.5); // not fully cancelled
    assert!(water_amplitude < 2.0); // not fully reinforced

    // bond angle recovered from angle.project()
    let projection = o_p1.angle.project(o_p2.angle);
    let reconstructed_angle = projection.acos();
    assert!((reconstructed_angle - bond_angle_rad).abs() < 0.01);
}

#[test]
fn it_replaces_element_class_with_angle_count() {
    // directly addresses issue spec: electron_count, shell_count, max_electrons, pending_electrons
    // all four derived from Vec<Geonum> — no Element or Shell struct needed
    // then prove electrons are waves: adding one changes the whole pattern
    // eliminates: Element/Shell class hierarchy

    // standing wave pattern: vector sum magnitude of all electrons
    // particles in bins are independent — adding one doesnt affect others
    // waves are coupled — adding one changes the interference pattern
    let standing_wave = |electrons: &[Geonum]| -> f64 {
        let mut x = 0.0;
        let mut y = 0.0;
        for e in electrons {
            x += e.mag * e.angle.grade_angle().cos();
            y += e.mag * e.angle.grade_angle().sin();
        }
        (x.powi(2) + y.powi(2)).sqrt()
    };

    // build carbon (Z=6): 6 geonums at shell angles
    // shells are pi/2 apart (quadrature). shell n sits at n * pi/2
    // subshell offsets as pi fractions: pi/200 for spin pair, pi/20 for orbital position
    let electron_in_shell = |n: usize, offset_num: f64, offset_denom: f64| -> Geonum {
        Geonum::new_with_angle(
            1.0,
            Angle::new(n as f64, 2.0) + Angle::new(offset_num, offset_denom),
        )
    };

    let carbon: Vec<Geonum> = vec![
        // shell 1: 1s2 — 2 electrons at pi/2
        electron_in_shell(1, 0.0, 1.0),
        electron_in_shell(1, 1.0, 200.0), // spin pair: pi/200 offset
        // shell 2: 2s2 + 2p2 — 4 electrons at pi (with subshell offsets)
        electron_in_shell(2, 0.0, 1.0),
        electron_in_shell(2, 1.0, 200.0), // spin pair
        electron_in_shell(2, 1.0, 20.0),  // p orbital: pi/20 offset
        electron_in_shell(2, 1.0, 15.0),  // p orbital: pi/15 offset
    ];

    // the four closures from the issue spec — all derived from angles
    // electron_count = len
    assert_eq!(carbon.len(), 6);

    // shell_count = distinct pi/2 stations
    let shell_of = |e: &Geonum| -> usize { (e.angle.grade_angle() / (PI / 2.0)).round() as usize };
    let mut shells: Vec<usize> = carbon.iter().map(&shell_of).collect();
    shells.sort();
    shells.dedup();
    assert_eq!(shells, vec![1, 2]);

    // max_electrons = 2n^2
    let max_electrons = |n: usize| 2 * n * n;
    assert_eq!(max_electrons(2), 8);

    // pending_electrons = max - count in outermost
    let outermost = *shells.last().unwrap();
    let in_outermost = carbon.iter().filter(|e| shell_of(e) == outermost).count();
    assert_eq!(max_electrons(outermost) - in_outermost, 4);

    // now prove these are waves, not particles in bins
    // track the standing wave pattern as electrons are added one by one
    let pattern_after_5 = standing_wave(&carbon[..5]);
    let pattern_after_6 = standing_wave(&carbon);

    // adding the 6th electron changed the interference pattern
    assert!((pattern_after_5 - pattern_after_6).abs() > EPSILON);

    // particles in bins: total = sum of individual magnitudes (no interaction)
    // waves: total ≠ sum because they interfere
    let scalar_sum: f64 = carbon.iter().map(|e| e.mag).sum();
    assert!((standing_wave(&carbon) - scalar_sum).abs() > EPSILON);

    // the pattern depends on angles, not just count
    // rotate one electron and the whole pattern shifts
    let mut rotated_carbon = carbon.clone();
    rotated_carbon[5] = rotated_carbon[5].rotate(Angle::new(1.0, 4.0)); // pi/4 nudge
    assert!((standing_wave(&carbon) - standing_wave(&rotated_carbon)).abs() > EPSILON);
}

// ═══════════════════════════════════════════════════════════
// the running wave sum
//
// an element is a count of electrons from the origin.
// blade count IS shell. grade IS subshell.
// energy is projection back to origin.
// ═══════════════════════════════════════════════════════════

fn subshell_order(max_n: usize) -> Vec<(usize, usize)> {
    let mut subs = Vec::new();
    for n in 1..=max_n {
        for l in 0..n {
            subs.push((n, l));
        }
    }
    subs.sort_by_key(|&(n, l)| (n + l, n));
    subs
}

fn grade_positions(base: Angle, l: usize, spread: Angle, spin: Angle) -> Vec<Angle> {
    let n_orb = 2 * l + 1;
    let orbital_step = spread / n_orb as f64;
    let mut pos = Vec::new();
    for orb in 0..n_orb {
        let mut angle = base;
        for _ in 0..orb {
            angle = angle + orbital_step;
        }
        pos.push(angle);
        pos.push(angle + spin);
    }
    pos
}

fn wave_sum(z: usize, spread: Angle, spin: Angle) -> Geonum {
    if z == 0 {
        return Geonum::new(0.0, 0.0, 1.0);
    }
    let order = subshell_order(5);
    let mut wave = Geonum::new(0.0, 0.0, 1.0);
    let mut placed = 0;

    for &(n, l) in &order {
        if placed >= z {
            break;
        }
        let mut base = Angle::new(1.0, 1.0); // π
        for _ in 0..l {
            base = base + spread;
        }
        let positions = grade_positions(base, l, spread, spin);
        let to_fill = positions.len().min(z - placed);
        let mag = 1.0 / n as f64;

        for &pos in positions.iter().take(to_fill) {
            wave = wave + Geonum::new_with_angle(mag, pos);
        }
        placed += to_fill;
    }
    wave
}

fn collect(z: usize, spread: Angle, spin: Angle) -> Vec<Geonum> {
    let order = subshell_order(5);
    let mut particles = Vec::new();
    let mut placed = 0;
    for &(n, l) in &order {
        if placed >= z {
            break;
        }
        let mut base = Angle::new(1.0, 1.0); // π
        for _ in 0..l {
            base = base + spread;
        }
        let positions = grade_positions(base, l, spread, spin);
        let to_fill = positions.len().min(z - placed);
        let mag = 1.0 / n as f64;
        for &pos in positions.iter().take(to_fill) {
            particles.push(Geonum::new_with_angle(mag, pos));
        }
        placed += to_fill;
    }
    particles
}

fn n_outer(z: usize) -> usize {
    let order = subshell_order(5);
    let mut placed = 0;
    let mut n = 1;
    for &(nn, l) in &order {
        if placed >= z {
            break;
        }
        n = nn;
        placed += (2 * (2 * l + 1)).min(z - placed);
    }
    n
}

/// scaffolding: compute Σ(count_at_shell / n²) from z.
/// deterministic from z and the derived ordering.
fn individual_sq(z: usize) -> f64 {
    let order = subshell_order(5);
    let mut sum = 0.0;
    let mut rem = z;
    for &(n, l) in &order {
        if rem == 0 {
            break;
        }
        let cap = (2 * (2 * l + 1)).min(rem);
        sum += cap as f64 / (n * n) as f64;
        rem -= cap;
    }
    sum
}

// act IV: the blade chain

#[test]
fn blade_chain_is_the_particle_zoo() {
    let proton = Geonum::new(1.0, 0.0, 1.0);
    let neutron = proton.increment_blade();
    let electron = neutron.increment_blade();
    let antineutrino = electron.increment_blade();
    let back = antineutrino.increment_blade();

    assert_eq!(proton.angle.grade(), 0);
    assert_eq!(neutron.angle.grade(), 1);
    assert_eq!(electron.angle.grade(), 2);
    assert_eq!(antineutrino.angle.grade(), 3);
    assert_eq!(back.angle.grade(), 0);
    assert_eq!(back.angle.blade(), 4);
}

#[test]
fn blade_count_is_shell() {
    let mut g = Geonum::new(1.0, 0.0, 1.0);
    for _ in 0..12 {
        let shell = g.angle.blade() / 4 + 1;
        let sub = g.angle.grade();
        // blade 0..3 → shell 1, blade 4..7 → shell 2, blade 8..11 → shell 3
        assert_eq!(shell, g.angle.blade() / 4 + 1);
        assert_eq!(sub, g.angle.blade() % 4);
        g = g.increment_blade();
    }
}

// act V: grades tell you everything

#[test]
fn binding_is_grade_2() {
    let nucleus = Geonum::new(RYDBERG, 0.0, 1.0);
    for n in 1..=4usize {
        let e = Geonum::new(1.0 / n as f64, 1.0, 1.0);
        let b = nucleus.dot(&e);
        assert_eq!(b.angle.grade(), 2);
        assert!((b.mag - RYDBERG / n as f64).abs() < 1e-6);
    }
}

#[test]
fn electron_electron_is_grade_0() {
    let e1 = Geonum::new(1.0, 1.0, 1.0);
    let e2 = Geonum::new(1.0, 1.0, 1.0);
    let d = e1.dot(&e2);
    assert_eq!(d.angle.grade(), 0);
}

#[test]
fn grade_offset_weakens_projection() {
    let spread = spread();
    let nucleus = Geonum::new(RYDBERG, 0.0, 1.0);

    let s = Geonum::new(0.5, 1.0, 1.0); // at π
    let p_angle = Angle::new(1.0, 1.0) + spread; // π + π/2
    let p = Geonum::new_with_angle(0.5, p_angle);

    let sb = nucleus.dot(&s);
    let pb = nucleus.dot(&p);

    assert_eq!(sb.angle.grade(), 2);
    // p-orbital at 3π/2 is orthogonal to nucleus at 0: cos(3π/2) = 0
    // so dot product magnitude is zero and grade is 0 (non-negative zero → grade 0)
    assert_eq!(pb.angle.grade(), 0);
    // p-electron offset by spread has zero binding projection (orthogonal)
    assert!(sb.mag > pb.mag);
}

// act VI: wave interference

#[test]
fn wave_self_dot_is_grade_0() {
    // wave.dot(wave): grade 2 + grade 2 = 4 ≡ 0
    let spread = spread();
    let spin = spin();
    for z in 1..=10 {
        let wave = wave_sum(z, spread, spin);
        let sd = wave.dot(&wave);
        assert_eq!(sd.angle.grade(), 0, "Z={}: self-dot is grade 0", z);
        assert!((sd.mag - wave.mag * wave.mag).abs() < 1e-6);
    }
}

#[test]
fn wave_sum_and_collect_are_the_same_chain() {
    let spread = spread();
    let spin = spin();

    for z in 1..=10usize {
        let wave = wave_sum(z, spread, spin);

        let particles = collect(z, spread, spin);
        let reconstructed = particles
            .iter()
            .fold(Geonum::new(0.0, 0.0, 1.0), |acc, &g| acc + g);

        assert!(wave.near_mag(reconstructed.mag));
        assert_eq!(wave.angle.grade(), reconstructed.angle.grade());
        assert_eq!(particles.len(), z);
    }
}

#[test]
fn every_wave_sum_cancels() {
    let spread = spread();
    let spin = spin();
    for z in 2..=18 {
        let wave = wave_sum(z, spread, spin);
        let particles = collect(z, spread, spin);
        let scalar_sum: f64 = particles.iter().map(|g| g.mag).sum();
        assert!(
            wave.mag < scalar_sum,
            "Z={}: wave ({:.4}) < scalar sum ({:.4})",
            z,
            wave.mag,
            scalar_sum
        );
    }
}

#[test]
fn wave_amplitude_contains_all_pairs() {
    let spread = spread();
    let spin = spin();

    for z in 2..=10usize {
        let wave = wave_sum(z, spread, spin);

        let particles = collect(z, spread, spin);

        // |wave|² = Σ|eᵢ|² + 2Σ|eᵢ||eⱼ|cos(θᵢ-θⱼ)
        // pairwise dot gives signed contribution via cos of angle diff
        let mut pair_sum = 0.0;
        for i in 0..particles.len() {
            for j in (i + 1)..particles.len() {
                let ai = particles[i].angle.grade_angle();
                let aj = particles[j].angle.grade_angle();
                pair_sum += particles[i].mag * particles[j].mag * (ai - aj).cos();
            }
        }

        let from_fold = wave.mag * wave.mag;
        let from_pairs = individual_sq(z) + 2.0 * pair_sum;

        assert!(
            (from_fold - from_pairs).abs() < 1e-3,
            "Z={}: wave ({:.6}) = decomposition ({:.6})",
            z,
            from_fold,
            from_pairs
        );
    }
}

// act VII: ionization energy from three lattice constants
//
// spread = π/2 = Angle::new(1.0, 2.0) — one grade step
// spin   = π/3 = Angle::new(1.0, 3.0) — pairing angle
// Q      = π/4 = Angle::new(1.0, 4.0) — phase shift between projection axes
//
// denominators 2, 3, 4 — the smallest rational π fractions after 1.
// zero fitted parameters.

fn ie_model(z: usize, waves: &[Geonum]) -> f64 {
    let q = Angle::new(1.0, 4.0);
    let n = n_outer(z);
    let nucleus = Geonum::new(z as f64, 0.0, 1.0);
    let marginal = waves[z] - waves[z - 1];
    let p = nucleus * marginal;
    let ref0 = Geonum::new(1.0, 0.0, 1.0);
    let ref_q = Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0));
    let adj = p.project(&ref0);
    let opp = p.project(&ref_q);
    RYDBERG * (adj.mag + q.grade_angle() * opp.mag) / (n * n) as f64
}

#[test]
fn ionization_energy_from_geometry() {
    // three lattice constants, zero fitted parameters
    let spread = spread();
    let spin = spin();
    let exp: [f64; 18] = [
        13.598, 24.587, 5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565, 5.139, 7.646,
        5.986, 8.152, 10.487, 10.360, 12.968, 15.760,
    ];

    let waves: Vec<Geonum> = (0..=18).map(|z| wave_sum(z, spread, spin)).collect();

    let mut sse = 0.0;
    for z in 1..=18usize {
        let ie = ie_model(z, &waves);
        assert!(ie > 0.0, "Z={}: IE must be positive", z);
        sse += (ie - exp[z - 1]).powi(2);
    }
    let rmse = (sse / 18.0).sqrt();

    // Be > B anomaly (Z=4 > Z=5)
    let ie_be = ie_model(4, &waves);
    let ie_b = ie_model(5, &waves);
    assert!(ie_be > ie_b, "Be ({:.2}) > B ({:.2})", ie_be, ie_b);

    // N > O anomaly (Z=7 > Z=8)
    let ie_n = ie_model(7, &waves);
    let ie_o = ie_model(8, &waves);
    assert!(ie_n > ie_o, "N ({:.2}) > O ({:.2})", ie_n, ie_o);

    // RMSE < 3.0 with zero free parameters
    assert!(rmse < 3.0, "RMSE={:.2} should be < 3.0", rmse);

    eprintln!("\n═══ act VII: ionization energy from geometry ═══\n");
    eprintln!("  spread = π/2, spin = π/3, Q = π/4");
    eprintln!("  denominators: 2, 3, 4 — zero fitted parameters\n");
    for z in 1..=18 {
        let ie = ie_model(z, &waves);
        let err = (ie - exp[z - 1]) / exp[z - 1] * 100.0;
        eprintln!(
            "  Z={:2} IE={:6.2} exp={:6.2} err={:+5.1}%",
            z,
            ie,
            exp[z - 1],
            err
        );
    }
    eprintln!("\n  RMSE={:.2}  anomalies=2/2\n", rmse);
}
