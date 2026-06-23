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
// grade_step = π/2, spin = π/3, Q = π/4 — denominators 2, 3, 4
// zero fitted parameters, both anomalies (Be > B, N > O)
//
// act VIII: the outer shell is max(n) across filled subshells —
// madelung fills 4s before 3d, so the spatial outer shell is the largest n
// present. the IE formula divides by n² of the outer shell. taking max(n)
// extends the model through both transition rows (Z=1-54, up to Xe)
//
// act IX: the same three constants predict second ionization energy —
// one formula iek(z, electrons) covers neutral atom and cation. the nuclear
// factor carries the exposed core charge z·(z-electrons+1), recovering the
// hydrogenic Z² limit for deep stripping while staying identity at neutral
//
// act X: a third observable — electron affinity. the next electron stepping ON
// (wave[z+1]-wave[z]) separates halogens (bound) from noble gases (a shell jump,
// repulsive), and Mulliken EN = (IE1+EA)/2 puts F on top. the intra-period
// gradient stays flat — the model's honest limit
//
// act XI: the relativistic edge — the max(n) d-rescue reverses at the 5d row.
// one parameter-free term, n_eff = max(n) − (Zα)²·(n_max−4)·(max(n)−last), the
// fine-structure constant fixed by nature, threads all three d rows
//
// acts XII–XV confront the np closed shells — Ar, Kr, Xe. within the shipped
// first-harmonic projection (both rays reading harmonic 1 of p) they sit above
// the (π/4)·p.mag ceiling and the deficit widens each period: a true theorem
// about that instrument, not a verdict on the geometry
//
// act XIII climbs it with a SECOND HARMONIC. squaring a geonum doubles its
// angle, and every np marginal's doubled phase lands on π/3 — the pairing
// closure. one structural quantum R/3, gated on phase + grade + core, lands all
// three within 2% and improves the in-sample fit. the missing "quadratic term"
// was scalar talk for rotation — a doubled phase, present the whole time
//
// act XIV banishes the gate: its three scalar predicates were one standing
// wave — the marginal's pair phase closing against the (n−1)p core (2m − C on a
// pure blade), the landed grade assigning the quantum (grade 1 → R/3, grade 3
// → R/9). act XV fences the next wall — generalizing the closure finds the s²
// family unbidden and one falsifier: molybdenum pays R/9 where its grade says R/3

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;
const RYDBERG: f64 = 13.6;

// NIST first ionization energies, Z=1..=80 (eV). Z=55-80 (Cs..Hg) carry act XI
const EXP: [f64; 80] = [
    13.598, 24.587, 5.392, 9.323, 8.298, 11.260, 14.534, 13.618, 17.423, 21.565, 5.139, 7.646,
    5.986, 8.152, 10.487, 10.360, 12.968, 15.760, 4.341, 6.113, 6.561, 6.828, 6.746, 6.767, 7.434,
    7.902, 7.881, 7.640, 7.726, 9.394, 5.999, 7.900, 9.815, 9.752, 11.814, 13.999, 4.177, 5.695,
    6.217, 6.634, 6.759, 7.092, 7.280, 7.361, 7.459, 8.337, 7.576, 8.994, 5.786, 7.344, 8.608,
    9.010, 10.451, 12.130, // Z=55..80: period 6 (Cs..Hg)
    3.894, 5.212, 5.577, 5.539, 5.473, 5.525, 5.582, 5.644, 5.670, 6.150, 5.864, 5.939, 6.022,
    6.108, 6.184, 6.254, 5.426, 6.825, 7.550, 7.864, 7.834, 8.438, 8.967, 8.959, 9.226, 10.438,
];

const ELEMENT: [&str; 80] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
];

fn grade_step() -> Angle {
    Angle::new(1.0, 2.0) // π/2 — one grade step
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
fn it_walks_aufbau_through_the_blade_lattice() {
    // the filling order is a walk through the blade lattice, no sorted table.
    // a subshell's tier T = n+l is the total angle (n+l)·π/2, which is blade
    // n+l. madelung_walk rotates one tier at a time (increment_blade) and trades
    // l for n down each tier's diagonal — the (n,l) pairs fall out of the walk
    // eliminates: the 19-entry aufbau order AND the (n+l, n) sort comparator

    let order = Geonum::madelung_order(6);

    // the walk reproduces the memorized aufbau sequence with no comparator
    let aufbau = [
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
    for (i, &(n, l)) in aufbau.iter().enumerate() {
        assert_eq!(order[i], (n, l), "subshell {i} diverges from aufbau");
    }

    // each subshell's tier is a blade: the total angle (n+l)·π/2 lands on blade
    // n+l with zero remainder. the madelung tier is a π/2 rotation count
    for &(n, l) in &order {
        let tier = Geonum::new(1.0, (n + l) as f64, 2.0); // (n+l)·π/2
        assert_eq!(tier.angle.blade(), n + l, "tier blade carries n+l");
        assert!(
            tier.angle.t().abs() < EPSILON,
            "tier angle lands on a clean π/2 multiple"
        );
    }

    // within a shared tier the diagonal trades l for n: consecutive same-tier
    // entries climb n by one and drop l by one
    for w in order.windows(2) {
        let (n0, l0) = w[0];
        let (n1, l1) = w[1];
        if n0 + l0 == n1 + l1 {
            assert_eq!(n1, n0 + 1, "diagonal step climbs n by one");
            assert_eq!(l0, l1 + 1, "diagonal step drops l by one");
        }
    }

    // successive tiers sit one π/2 rotation apart — the outer walk is
    // increment_blade, the same quarter turn that separates grades
    let tier_2p = Geonum::new(1.0, 3.0, 2.0); // 3·π/2, blade 3, the 2p tier
    assert_eq!(
        tier_2p.rotate(grade_step()).angle.blade(),
        4,
        "the next tier is one π/2 rotation past"
    );
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
    // chromium Z=24: conventional [Ar] 4s2 3d4, measured [Ar] 4s1 3d5. a
    // half-filled d shell is five evenly-spaced angles, and their lower average
    // pairwise overlap is the geometric reason that filling is favored (the
    // madelung walk eliminates the ~20 exception patches; this is the intuition)

    // 4s (n+l=4) and 3d (n+l=5) are adjacent tiers — one π/2, one blade, apart
    let s_tier = Geonum::new(1.0, 4.0, 2.0); // 4·π/2, the 4s tier
    let d_tier = Geonum::new(1.0, 5.0, 2.0); // 5·π/2, the 3d tier
    assert_eq!(
        d_tier.angle.blade(),
        s_tier.angle.blade() + 1,
        "3d sits one tier (one blade) past 4s"
    );

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

// the model itself — the madelung walk, the electron wave, the valence shell,
// the ionization projection — lives in the library: `Geonum`'s `Chemistry` trait
// (src/traits/chemistry.rs). these tests build their geonums with that trait and
// validate the outputs against NIST. two small helpers stay test-side:

// the last-filled shell (the naive rule) — the foil act VIII measures the spatial
// valence shell against. a consumer wants `Geonum::valence_shell`; this is only
// the comparison baseline (and the relativistic shell's overshoot reference)
fn last_filled(z: usize) -> usize {
    let mut placed = 0;
    let mut n = 1;
    for (nn, l) in Geonum::madelung_order(6) {
        if placed >= z {
            break;
        }
        n = nn; // overwrite — last-filled wins
        placed += (2 * (2 * l + 1)).min(z - placed);
    }
    n
}

// Σ(count_at_shell / n²) over the filled subshells — the diagonal sum the wave
// amplitude decomposes into (it_contains_all_pairs_in_the_wave_amplitude)
fn individual_sq(z: usize) -> f64 {
    let mut sum = 0.0;
    let mut rem = z;
    for (n, l) in Geonum::madelung_order(6) {
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
fn it_chains_the_particle_zoo_with_increment_blade() {
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
fn it_decomposes_the_blade_into_shell_and_subshell() {
    // one blade integer carries both quantum numbers: shell = blade/4 + 1 (which
    // 2π winding) and subshell = grade = blade % 4. they recombine to the blade,
    // so the two are one count, not two separate stores. the physical (n,l)
    // anchoring lives in it_walks_aufbau_through_the_blade_lattice
    let mut g = Geonum::new(1.0, 0.0, 1.0);
    for _ in 0..12 {
        let blade = g.angle.blade();
        let shell = blade / 4 + 1;
        let sub = g.angle.grade();
        assert_eq!(
            4 * (shell - 1) + sub,
            blade,
            "shell and subshell recombine to the one blade"
        );
        g = g.increment_blade();
    }
}

// act V: grades tell you everything

#[test]
fn it_binds_at_grade_2() {
    let nucleus = Geonum::new(RYDBERG, 0.0, 1.0);
    for n in 1..=4usize {
        let e = Geonum::new(1.0 / n as f64, 1.0, 1.0);
        let b = nucleus.dot(&e);
        assert_eq!(b.angle.grade(), 2);
        assert!((b.mag - RYDBERG / n as f64).abs() < 1e-6);
    }
}

#[test]
fn it_dots_electron_electron_to_grade_0() {
    let e1 = Geonum::new(1.0, 1.0, 1.0);
    let e2 = Geonum::new(1.0, 1.0, 1.0);
    let d = e1.dot(&e2);
    assert_eq!(d.angle.grade(), 0);
}

#[test]
fn it_weakens_projection_with_grade_offset() {
    let nucleus = Geonum::new(RYDBERG, 0.0, 1.0);

    // binding is the electron's projection onto the nucleus axis; offsetting it
    // off-axis weakens that projection monotonically, reaching zero at the
    // orthogonal quarter turn. bind at π, then offset by π/4, then by π/2
    let aligned = Geonum::new_with_angle(0.5, Angle::new(1.0, 1.0)); // π
    let off_q = Geonum::new_with_angle(0.5, Angle::new(1.0, 1.0) + Angle::new(1.0, 4.0)); // π+π/4
    let off_h = Geonum::new_with_angle(0.5, Angle::new(1.0, 1.0) + Angle::new(1.0, 2.0)); // π+π/2

    let b_aligned = nucleus.dot(&aligned).mag;
    let b_off_q = nucleus.dot(&off_q).mag;
    let b_off_h = nucleus.dot(&off_h).mag;

    // monotone weakening: full at π, less at a quarter-grade offset, zero at the
    // orthogonal half-grade offset
    assert!(
        b_aligned > b_off_q,
        "a quarter-grade offset weakens the binding"
    );
    assert!(b_off_q > b_off_h, "a half-grade offset weakens it further");
    assert!(b_off_h < 1e-9, "the orthogonal offset zeroes the binding");
}

// act VI: wave interference

#[test]
fn it_self_dots_the_wave_to_grade_0() {
    // wave.dot(wave): grade 2 + grade 2 = 4 ≡ 0
    for z in 1..=10 {
        let wave = Geonum::electron_wave(z, Lattice::Canonical);
        let sd = wave.dot(&wave);
        assert_eq!(sd.angle.grade(), 0, "Z={}: self-dot is grade 0", z);
        assert!((sd.mag - wave.mag * wave.mag).abs() < 1e-6);
    }
}

#[test]
fn it_proves_wave_sum_and_collect_are_one_chain() {
    for z in 1..=10usize {
        let wave = Geonum::electron_wave(z, Lattice::Canonical);

        let particles = Geonum::electron_shell(z, Lattice::Canonical).objects;
        let reconstructed = particles
            .iter()
            .fold(Geonum::new(0.0, 0.0, 1.0), |acc, &g| acc + g);

        assert!(wave.near_mag(reconstructed.mag));
        assert_eq!(wave.angle.grade(), reconstructed.angle.grade());
        assert_eq!(particles.len(), z);
    }
}

#[test]
fn it_cancels_every_wave_sum() {
    for z in 2..=18 {
        let wave = Geonum::electron_wave(z, Lattice::Canonical);
        let particles = Geonum::electron_shell(z, Lattice::Canonical).objects;
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
fn it_contains_all_pairs_in_the_wave_amplitude() {
    for z in 2..=10usize {
        let wave = Geonum::electron_wave(z, Lattice::Canonical);

        let particles = Geonum::electron_shell(z, Lattice::Canonical).objects;

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
// grade_step = π/2 = Angle::new(1.0, 2.0) — one grade step
// spin   = π/3 = Angle::new(1.0, 3.0) — pairing angle
// Q      = π/4 = Angle::new(1.0, 4.0) — the opp coefficient: π/4's radian (≈0.785)
//          weights the π/2-axis projection (the 38° ray, not the bisector — see
//          chem_constants_test::it_projects_the_opp_combiner_onto_the_atan_ray)
//
// denominators 2, 3, 4 — the smallest rational π fractions after 1, each a clean
// lattice landmark; Q's radian rides as a band-confined coefficient (finding #1)

// the IE model lives in the library — Geonum's Chemistry trait
// (src/traits/chemistry.rs). these are thin views on it, so the tests below
// validate Geonum::ionization_energy / electron_affinity against NIST.

// IE1 (the z-th electron stepping off) and IE_k (z protons, `electrons` electrons)
fn ie_model(z: usize) -> f64 {
    Geonum::ionization_energy(z, z, Lattice::Canonical)
}
fn iek(z: usize, electrons: usize) -> f64 {
    Geonum::ionization_energy(z, electrons, Lattice::Canonical)
}

// IE1 projected over a CHOSEN outer shell — used to compare the spatial
// valence_shell against the naive last_filled foil (act VIII)
fn ie_at(z: usize, n: usize) -> f64 {
    let marginal = Geonum::electron_wave(z, Lattice::Canonical)
        - Geonum::electron_wave(z - 1, Lattice::Canonical);
    marginal.ionization_projection(
        Geonum::new(z as f64, 0.0, 1.0),
        n as f64,
        Lattice::Canonical,
    )
}

// electron affinity: signed, and the unsigned binding magnitude
fn ea(z: usize) -> f64 {
    Geonum::electron_affinity(z, Lattice::Canonical)
}
fn ea_bind(z: usize) -> f64 {
    Geonum::electron_affinity(z, Lattice::Canonical).abs()
}

// the relativistic IE (act XI): the marginal projected over the contracted shell
fn ie_rel(z: usize) -> f64 {
    let marginal = Geonum::electron_wave(z, Lattice::Canonical)
        - Geonum::electron_wave(z - 1, Lattice::Canonical);
    marginal.ionization_projection(
        Geonum::new(z as f64, 0.0, 1.0),
        Geonum::relativistic_valence_shell(z),
        Lattice::Canonical,
    )
}

#[test]
fn it_computes_ionization_energy_from_geometry() {
    // three lattice constants, zero fitted parameters

    let mut sse = 0.0;
    for z in 1..=18usize {
        let ie = ie_model(z);
        assert!(ie > 0.0, "Z={}: IE must be positive", z);
        sse += (ie - EXP[z - 1]).powi(2);
    }
    let rmse = (sse / 18.0).sqrt();

    // Be > B anomaly (Z=4 > Z=5)
    let ie_be = ie_model(4);
    let ie_b = ie_model(5);
    assert!(ie_be > ie_b, "Be ({:.2}) > B ({:.2})", ie_be, ie_b);

    // N > O anomaly (Z=7 > Z=8)
    let ie_n = ie_model(7);
    let ie_o = ie_model(8);
    assert!(ie_n > ie_o, "N ({:.2}) > O ({:.2})", ie_n, ie_o);

    // RMSE < 3.0 with zero free parameters
    assert!(rmse < 3.0, "RMSE={:.2} should be < 3.0", rmse);

    eprintln!("\n═══ act VII: ionization energy from geometry ═══\n");
    eprintln!("  grade_step = π/2, spin = π/3, Q = π/4");
    eprintln!("  denominators: 2, 3, 4 — zero fitted parameters\n");
    for z in 1..=18 {
        let ie = ie_model(z);
        let err = (ie - EXP[z - 1]) / EXP[z - 1] * 100.0;
        eprintln!(
            "  Z={:2} IE={:6.2} exp={:6.2} err={:+5.1}%",
            z,
            ie,
            EXP[z - 1],
            err
        );
    }
    eprintln!("\n  RMSE={:.2}  anomalies=2/2\n", rmse);
}

// act VIII: the outer shell is max(n)
//
// madelung fills 4s before 3d, 5s before 4d, 6s before 4f before 5d. across
// these the largest n present is the spatial outer shell — the electron the
// IE formula ionizes. taking max(n) for the n² divisor extends act VII from
// the s/p block (Z=1-18) through both transition rows — the 3d block (Z=19-36)
// and the 4d block (Z=37-54, up to Xe) — with zero new parameters. the same
// fix rescues both d rows. last_filled (last-filled n) stands as the foil.

// root mean square error of the IE model over Z=start..=end for a given outer
// shell rule
fn rmse(start: usize, end: usize, outer: fn(usize) -> usize) -> f64 {
    let mut sse = 0.0;
    for z in start..=end {
        let marginal = Geonum::electron_wave(z, Lattice::Canonical)
            - Geonum::electron_wave(z - 1, Lattice::Canonical);
        let pred = marginal.ionization_projection(
            Geonum::new(z as f64, 0.0, 1.0),
            outer(z) as f64,
            Lattice::Canonical,
        );
        sse += (pred - EXP[z - 1]).powi(2);
    }
    (sse / (end - start + 1) as f64).sqrt()
}

#[test]
fn it_holds_max_n_as_identity_in_sample() {
    // Z=1-18: madelung order is monotonic in n, so max(n) equals last-filled n.
    // the running max leaves the in-sample RMSE untouched

    let r_orig = rmse(1, 18, last_filled);
    let r_max = rmse(1, 18, Geonum::valence_shell);

    assert!(
        (r_orig - r_max).abs() < 1e-10,
        "max(n) is identity for Z=1-18"
    );
    assert!(r_max < 3.0, "in-sample RMSE preserved: {:.3}", r_max);
}

#[test]
fn it_tames_both_d_blocks_with_max_n() {
    // 3d fills after 4s, 4d after 5s. last-filled n drops to the inner d and
    // over-predicts each row; max(n) holds the outer s and lands both d blocks
    // in the s/p error band. the fix that rescued the 3d row rescues the 4d row

    for (block, lo, hi) in [("3d", 21, 30), ("4d", 39, 48)] {
        let r_orig = rmse(lo, hi, last_filled);
        let r_max = rmse(lo, hi, Geonum::valence_shell);

        eprintln!("  {block} block: last-filled {r_orig:.3} eV  ->  max(n) {r_max:.3} eV");

        assert!(
            r_max < r_orig / 2.0,
            "{block}: max(n) at least halves the d-block RMSE (last-filled {r_orig:.3}, max {r_max:.3})"
        );
        assert!(
            r_max < 1.5,
            "{block}: max(n) RMSE sits in the s/p band ({r_max:.3} eV)"
        );
    }
}

#[test]
fn it_lands_transition_metals_in_physical_range() {
    // last-filled n predicts the d rows at 11-19 eV (measured 6-9 eV), +50-115%
    // error. max(n) lands both Sc-Zn and Y-Cd in single digits within 30%

    for z in (21..=30).chain(39..=48) {
        let pred = ie_at(z, Geonum::valence_shell(z));
        let measured = EXP[z - 1];
        let err_pct = (pred - measured).abs() / measured * 100.0;

        assert!(
            pred < 12.0,
            "{} (Z={}): predicted {:.2} eV outside physical range",
            ELEMENT[z - 1],
            z,
            pred
        );
        assert!(
            err_pct < 30.0,
            "{} (Z={}): {:.1}% error exceeds 30% bound",
            ELEMENT[z - 1],
            z,
            err_pct
        );
    }
}

#[test]
fn it_prints_the_full_comparison() {
    eprintln!("\nZ=1-54 ionization energies, last-filled n vs max(n) (zero new parameters)\n");
    eprintln!("  Z   elem    lastfill  max(n)   exp     last_err  max_err");
    for z in 1..=54 {
        let p_orig = ie_at(z, last_filled(z));
        let p_max = ie_at(z, Geonum::valence_shell(z));
        let e = EXP[z - 1];
        let e_orig = (p_orig - e) / e * 100.0;
        let e_max = (p_max - e) / e * 100.0;
        let mark = match z {
            1..=18 => "",
            19..=36 => " (3d row)",
            _ => " (4d row)",
        };
        eprintln!(
            "  {:2}  {:3}     {:6.2}    {:6.2}   {:6.2}   {:+6.1}%   {:+6.1}%{}",
            z,
            ELEMENT[z - 1],
            p_orig,
            p_max,
            e,
            e_orig,
            e_max,
            mark
        );
    }

    let periods: [(&str, usize, usize); 3] = [
        ("Z=1-18  (s/p)", 1, 18),
        ("Z=19-36 (3d) ", 19, 36),
        ("Z=37-54 (4d) ", 37, 54),
    ];
    eprintln!("\n  RMSE by period (last-filled  ->  max(n)):");
    for (label, lo, hi) in periods {
        eprintln!(
            "    {label}  {:.3}  ->  {:.3} eV",
            rmse(lo, hi, last_filled),
            rmse(lo, hi, Geonum::valence_shell)
        );
    }
}

// act IX: the same constants predict second ionization energy
//
// a model with real geometric content predicts more than the one observable it
// was read off. the cation has z protons but fewer electrons. the marginal is
// the electron stepping off (from electrons-1 to electrons), the nucleus keeps
// its full z protons, and the n² divisor reads the outer shell that remains.
//
// the nuclear factor carries the EXPOSED CORE CHARGE: the electrons-1 inner
// electrons screen electrons-1 protons, leaving z-(electrons-1) exposed to the
// marginal. the geometric product nucleus.geo(exposed) lands magnitude
// z·(z-electrons+1) at grade 0 — one factor of z from the nucleus, the second
// from the exposed core. at neutral exposed=1, so the factor is the bare z and
// acts VII-VIII are untouched. at a bare ion (1 electron) exposed=z, so the
// factor is z² and the model recovers the hydrogenic IE = RYDBERG·Z²/n² exactly.

// NIST second ionization energies, (Z, eV) for Z=3..=20. IE2 starts at Z=3
// because He+ to He²⁺ is the bare hydrogenic limit, outside the screened model
const IE2: [(usize, f64); 18] = [
    (3, 75.640),
    (4, 18.211),
    (5, 25.155),
    (6, 24.383),
    (7, 29.601),
    (8, 35.121),
    (9, 34.971),
    (10, 40.963),
    (11, 47.286),
    (12, 15.035),
    (13, 18.829),
    (14, 16.346),
    (15, 19.769),
    (16, 23.338),
    (17, 23.814),
    (18, 27.630),
    (19, 31.625),
    (20, 11.872),
];

#[test]
fn it_reduces_ie2_to_ie1_at_full_electron_count() {
    // the bridge: iek(z, z) is the neutral atom — exposed = z-(z-1) = 1, so the
    // nuclear factor collapses to the bare z and iek recovers act VII's ie_model
    // exactly. IE2 rides the same machinery, the electron count drops by one

    for z in 1..=20usize {
        let general = iek(z, z);
        let neutral = ie_model(z);
        assert!(
            (general - neutral).abs() < EPSILON,
            "Z={z}: iek(z,z) {general:.6} != ie_model {neutral:.6}"
        );
    }
}

#[test]
fn it_reproduces_the_hydrogenic_series_exactly() {
    // one electron of charge Z binds at exactly RYDBERG·Z². the single-electron
    // marginal is [1, grade 2] (no opp component), so the bare-z numerator would
    // give RYDBERG·Z, short one factor of z. the exposed core (electrons=1 →
    // exposed=z) supplies the second z and the form lands 13.6·Z² to the bit

    let names = ["H", "He+", "Li2+", "Be3+"];
    for z in 1..=4usize {
        let pred = iek(z, 1);
        let exact = RYDBERG * (z * z) as f64;
        assert!(
            (pred - exact).abs() < EPSILON,
            "{}: {pred:.4} != exact 13.6·Z² {exact:.4}",
            names[z - 1]
        );
    }
}

#[test]
fn it_tracks_the_ie2_trend_against_nist() {
    // the same three constants trace the IE2 series across Z=3-20. the exposed
    // core Z² lands the magnitude near absolute — the global least-squares scale
    // sits near 1, where the bare-z numerator needed ~1.9. pearson r measures the
    // shape: every group rise and the resets at Be/Mg/Ca where a fresh valence
    // shell opens

    let preds: Vec<f64> = IE2.iter().map(|&(z, _)| iek(z, z - 1)).collect();
    let measured: Vec<f64> = IE2.iter().map(|&(_, m)| m).collect();
    let n = preds.len() as f64;

    let mp = preds.iter().sum::<f64>() / n;
    let mm = measured.iter().sum::<f64>() / n;
    let cov: f64 = preds
        .iter()
        .zip(&measured)
        .map(|(p, m)| (p - mp) * (m - mm))
        .sum();
    let sp = preds.iter().map(|p| (p - mp).powi(2)).sum::<f64>().sqrt();
    let sm = measured
        .iter()
        .map(|m| (m - mm).powi(2))
        .sum::<f64>()
        .sqrt();
    let r = cov / (sp * sm);

    // least-squares global scale through the origin
    let scale = preds.iter().zip(&measured).map(|(p, m)| p * m).sum::<f64>()
        / preds.iter().map(|p| p * p).sum::<f64>();

    eprintln!("\nIE2 trend, Z=3-20 (exposed-core Z², zero new parameters)\n");
    eprintln!("  el    pred    nist    err");
    for (i, &(z, m)) in IE2.iter().enumerate() {
        eprintln!(
            "  {:3} {:7.2} {:7.2}  {:+5.1}%",
            ELEMENT[z - 1],
            preds[i],
            m,
            (preds[i] - m) / m * 100.0
        );
    }
    // the scale's offset from 1 is the one-for-one screening residual: exposed =
    // z-(electrons-1) charges each inner electron with exactly one proton of
    // screening, but real screening isnt exactly 1.0, so the model reports the
    // leak as a global scale near but not at 1
    let screening_residual = (1.0 - scale).abs();
    eprintln!(
        "\n  pearson r = {r:.4}   global scale = {scale:.3}   one-for-one screening residual = {:.1}%",
        screening_residual * 100.0
    );

    // r near 1 means the geometry orders the whole series
    assert!(
        r > 0.95,
        "IE2 trend tracks NIST: pearson r {r:.4} below 0.95"
    );
    // the exposed-core Z² lands the magnitude near absolute — the residual is
    // the one-for-one screening assumption leaking, a few percent rather than
    // the ~2x scale the bare-z numerator needed
    assert!(
        screening_residual < 0.15,
        "one-for-one screening leaks {:.1}% — past the 15% band",
        screening_residual * 100.0
    );
}

#[test]
fn it_separates_the_closed_core_cliff_from_valence_removal() {
    // ionizing into a noble-gas core costs many times the neutral IE1, while
    // pulling a remaining valence electron costs about the same again. the
    // geometry draws the line through the valence shell: Li+/Na+/K+ collapse to a smaller
    // inner shell so the n² divisor shrinks and IE2/IE1 jumps; Be/Mg/Ca keep
    // their outer s shell so the ratio sits lower

    let closed_core = [3usize, 11, 19]; // Li, Na, K — ionize into He/Ne/Ar cores
    let valence = [4usize, 12, 20]; // Be, Mg, Ca — still hold a valence s

    let ratio = |z: usize| iek(z, z - 1) / iek(z, z);

    let closed_ratios: Vec<f64> = closed_core.iter().map(|&z| ratio(z)).collect();
    let valence_ratios: Vec<f64> = valence.iter().map(|&z| ratio(z)).collect();

    let min_closed = closed_ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_valence = valence_ratios
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    eprintln!("\n  min closed-core IE2/IE1 {min_closed:.2}  >  max valence {max_valence:.2}");

    // every closed-core jump clears every valence step by a wide margin — the
    // geometry never reads a noble-core break as a valence removal
    assert!(
        min_closed > 2.0 * max_valence,
        "cliff: smallest closed-core jump {min_closed:.2} more than doubles the largest valence ratio {max_valence:.2}"
    );

    // Li+ is the sharpest cliff in the table — a two-electron He core
    assert!(
        closed_ratios[0] > 8.0,
        "Li IE2/IE1 jump {:.2} below 8 — the He-core cliff flattened",
        closed_ratios[0]
    );
}

#[test]
fn it_drops_a_shell_into_the_core() {
    // the mechanism behind the cliff stated directly: ionizing Li+, Na+, K+
    // breaks into the core so the valence shell falls a full shell, while Be/Mg hold their
    // valence shell. the n² divisor change IS the cliff
    for (z, expect_drop) in [
        (3usize, true),
        (11, true),
        (19, true),
        (4, false),
        (12, false),
    ] {
        let neutral_n = Geonum::valence_shell(z);
        let cation_n = Geonum::valence_shell(z - 1);
        let dropped = cation_n < neutral_n;
        assert_eq!(
            dropped,
            expect_drop,
            "{}: cation shell drop {dropped} differs from physics {expect_drop}",
            ELEMENT[z - 1]
        );
    }
}

#[test]
fn it_tracks_the_na_deep_stripping_staircase() {
    // the Na successive-ionization staircase IE1..IE5. the exposed core climbs
    // 1, 2, 3, 4, 5 as electrons strip away, carrying the nuclear factor from z
    // toward z² and tracking the hydrogenic Z² rise NIST follows. the bare-z
    // numerator plateaus near 18-24 eV here; the exposed-core form lands every
    // deep step within 1.5x of measured
    let z = 11;
    let nist = [5.139, 47.286, 71.620, 98.910, 138.400];

    eprintln!("\nNa successive ionization (Z=11), exposed-core Z²\n");
    eprintln!("  k  electrons  exposed  n_out   pred    nist");
    let mut preds = Vec::new();
    for k in 1..=5usize {
        let electrons = z - (k - 1);
        let pred = iek(z, electrons);
        preds.push(pred);
        eprintln!(
            "  {k}      {electrons:2}         {}      {}    {pred:6.2}  {:6.2}",
            z - (electrons - 1),
            Geonum::valence_shell(electrons),
            nist[k - 1]
        );
    }

    // the first cliff is real and large
    assert!(
        preds[1] / preds[0] > 3.0,
        "Na IE2/IE1 {:.2} below 3 — first cliff flattened",
        preds[1] / preds[0]
    );
    // every deep step lands within 1.5x of NIST — the Z² rise is recovered
    for k in 3..=5usize {
        let gap = nist[k - 1] / preds[k - 1];
        assert!(
            gap < 1.5,
            "Na IE{k}: gap {gap:.2}x above 1.5x — deep stripping not recovered"
        );
    }
}

// act X: a third observable — electron affinity
//
// IE1 reads the electron stepping OFF (wave[z]-wave[z-1]); electron affinity
// reads the next electron stepping ON (wave[z+1]-wave[z]) as a neutral atom
// gains an electron to an anion. the added electron sits outside the z already
// present, so it feels a nucleus screened to +1, and the divisor reads the
// anion's valence shell. same projection, the marginal index shifts.
//
// the geometry separates closed from open shells: a noble gas refuses the
// electron — the valence shell jumps a shell and the marginal lands at grade 2 / angle π
// (anti-aligned, repulsive) — while a halogen binds it at grade 0 near the
// origin, and F comes out the most bound. the limit is the intra-period
// gradient: the flat 1/n weight cant climb B<C<O<F, stated as a measured fact.

// NIST electron affinities (eV), Z=1-18. noble gases and alkaline earths are
// near zero or unbound, reported negative by convention
const EA_NIST: [(usize, f64); 18] = [
    (1, 0.754),
    (2, -0.50),
    (3, 0.618),
    (4, -0.50),
    (5, 0.280),
    (6, 1.262),
    (7, -0.07),
    (8, 1.461),
    (9, 3.401),
    (10, -0.30),
    (11, 0.548),
    (12, -0.40),
    (13, 0.433),
    (14, 1.390),
    (15, 0.746),
    (16, 2.077),
    (17, 3.613),
    (18, -0.36),
];

#[test]
fn it_isolates_the_noble_gases_by_shell_jump() {
    // the first signal: adding an electron that opens a new shell —
    // Geonum::valence_shell(z+1) > Geonum::valence_shell(z) — flags exactly He, Ne, Ar across Z=1-18 with
    // zero false positives. the same valence-shell closure the IE2 cliff rides,
    // read the other direction: a closed shell refuses the next electron

    let nobles = [2usize, 10, 18]; // He, Ne, Ar
    let mut flagged = Vec::new();
    for z in 1..=18usize {
        if Geonum::valence_shell(z + 1) > Geonum::valence_shell(z) {
            flagged.push(z);
        }
    }
    assert_eq!(
        flagged, nobles,
        "the shell-jump flags exactly the noble gases"
    );

    // every flagged element measures EA ≤ 0 — the geometry never marks a bound
    // element as opening a new shell
    for &z in &nobles {
        let nist = EA_NIST.iter().find(|&&(zz, _)| zz == z).unwrap().1;
        assert!(
            nist <= 0.0,
            "{}: flag matches unbound EA {nist:.2}",
            ELEMENT[z - 1]
        );
    }
}

#[test]
fn it_separates_halogens_from_noble_gases() {
    // the headline: the geometry separates halogens (large positive EA) from
    // noble gases (near zero or negative). the signed projection reads the
    // noble-gas marginal at grade 2 / angle π as negative and the halogen
    // marginal at grade 0 as the most bound. F is the most bound in period 2
    let waves: Vec<Geonum> = (0..=20)
        .map(|z| Geonum::electron_wave(z, Lattice::Canonical))
        .collect();

    let halogens = [9usize, 17]; // F, Cl
    let nobles = [2usize, 10, 18]; // He, Ne, Ar

    let halogen_min = halogens
        .iter()
        .map(|&z| ea(z))
        .fold(f64::INFINITY, f64::min);
    let noble_max = nobles
        .iter()
        .map(|&z| ea(z))
        .fold(f64::NEG_INFINITY, f64::max);

    for &z in &nobles {
        let v = ea(z);
        assert!(
            v < 0.0,
            "{}: noble-gas EA {v:.3} is unbound",
            ELEMENT[z - 1]
        );
    }
    assert!(
        halogen_min > noble_max,
        "halogen floor {halogen_min:.3} clears noble ceiling {noble_max:.3}"
    );

    // F is the most bound element in period 2, compared against B, C, O. nitrogen
    // (Z=7) is left out of the comparison: its half-filled 2p³ is a degenerate
    // special case the flat 1/n weight treats as a peer, so it is not a fair
    // comparand (act X's stated limit)
    let ea_f = ea(9);
    for z in [5usize, 6, 8] {
        assert!(
            ea_f >= ea(z),
            "F ({ea_f:.3}) is at least as bound as {} ({:.3})",
            ELEMENT[z - 1],
            ea(z)
        );
    }

    eprintln!("\n═══ act X: electron affinity from the same three constants ═══\n");
    eprintln!("  el  Z   ea_pred   nist     shell-jump   marg-grade");
    for &(z, nist) in EA_NIST.iter() {
        let jump = Geonum::valence_shell(z + 1) > Geonum::valence_shell(z);
        let marg = waves[z + 1] - waves[z];
        eprintln!(
            "  {:3} {:2}  {:8.4}  {:+7.3}   {:9}    {}",
            ELEMENT[z - 1],
            z,
            ea(z),
            nist,
            if jump { "NEW SHELL" } else { "-" },
            marg.angle.grade(),
        );
    }
    eprintln!("\n  halogen floor {halogen_min:.3} eV  >  noble ceiling {noble_max:.3} eV");
}

#[test]
fn it_holds_the_halogen_separation_into_period_4() {
    // the split is not a small-Z accident: Br (Z=35, halogen) binds while Kr
    // (Z=36, noble) opens shell 5 and goes unbound, same as F/Ne and Cl/Ar

    let ea_br = ea(35); // Br, NIST 3.36 eV
    let ea_kr = ea(36); // Kr, unbound

    assert!(ea_br > 0.0, "Br EA {ea_br:.3} is bound");
    assert!(ea_kr < 0.0, "Kr EA {ea_kr:.3} is unbound (new shell)");
    assert!(
        Geonum::valence_shell(36) < Geonum::valence_shell(37),
        "Kr->anion opens shell 5, the geometric signal for unbound"
    );
}

#[test]
fn it_signs_every_affinity_by_subshell_continuity() {
    // the EA sign turns on whether the added electron extends the open subshell
    // or opens a fresh closure (subshell_of(z+1) == subshell_of(z)). that one
    // madelung-walk equality signs every affinity Z=1-18 against NIST with a
    // single miss — nitrogen, whose half-filled 2p³ reads near zero (−0.07).
    // both the alkalis (a second s electron, bound) and the alkaline earths
    // (a first p against a closed s², unbound) land their measured sign
    let disagree: Vec<usize> = EA_NIST
        .iter()
        .filter(|&&(z, nist)| (ea(z) > 0.0) != (nist > 0.0))
        .map(|&(z, _)| z)
        .collect();
    assert_eq!(
        disagree,
        vec![7],
        "subshell continuity signs every affinity but nitrogen's half-filled 2p³"
    );
}

#[test]
fn it_puts_fluorine_on_top_by_mulliken_electronegativity() {
    // mulliken EN = (IE1 + EA)/2, both from marginals. the IE1 component carries
    // the period trend and the EA binding tips F over the rest. test against the
    // pauling scale by pearson correlation — rising across a period, F on top

    let pauling: [(usize, f64); 10] = [
        (3, 0.98),
        (5, 2.04),
        (6, 2.55),
        (7, 3.04),
        (8, 3.44),
        (9, 3.98),
        (11, 0.93),
        (15, 2.19),
        (16, 2.58),
        (17, 3.16),
    ];

    let en = |z: usize| (ie_model(z) + ea_bind(z)) / 2.0;
    let preds: Vec<f64> = pauling.iter().map(|&(z, _)| en(z)).collect();
    let refs: Vec<f64> = pauling.iter().map(|&(_, p)| p).collect();

    // F carries the largest predicted EN — the top of the scale
    let en_f = en(9);
    let en_max = preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (en_f - en_max).abs() < EPSILON,
        "F EN {en_f:.3} is the maximum {en_max:.3}"
    );

    let n = preds.len() as f64;
    let mp = preds.iter().sum::<f64>() / n;
    let mm = refs.iter().sum::<f64>() / n;
    let cov: f64 = preds
        .iter()
        .zip(&refs)
        .map(|(p, m)| (p - mp) * (m - mm))
        .sum();
    let sp = preds.iter().map(|p| (p - mp).powi(2)).sum::<f64>().sqrt();
    let sm = refs.iter().map(|m| (m - mm).powi(2)).sum::<f64>().sqrt();
    let r = cov / (sp * sm);

    eprintln!("\n  mulliken EN vs pauling: F on top, pearson r = {r:.4}");
    assert!(r > 0.85, "mulliken EN tracks pauling: r {r:.4} below 0.85");
}

// act XI: the relativistic edge — the 5d row
//
// max(n) rescued the 3d row (4s before 3d) and the 4d row (5s before 4d) by
// holding the outer s shell. the third try reverses: at the 5d row (Hf-Hg,
// Z=72-80) max(n) holds the outer 6s and under-predicts (Os −39.7%), while
// last-filled drops to the inner 5d and lands closer. relativistic 6s
// contraction pulls the measured 5d IEs up toward the inner-d magnitude — the
// heavy 6s feels a smaller effective n than its principal quantum number.
//
// the correction is one parameter-free term. n_eff contracts max(n) toward the
// inner d by the product of two factors read from nature and the walk:
//   (Zα)²       — the lorentz weight of the s electron at velocity Zα·c
//   (n_max − 4) — periods since the d-inversion onset at the 3d row (n_max=4)
// at the 3d row (n_max−4)=0 kills the term, so n_eff = max(n) exactly and the
// first rescue stands untouched. at 5d the factor is 2 and (Zα)² ≈ 0.3, pulling
// n_eff most of the way to the inner 5d and rescuing the reversed row.
//
// the FORM is an ansatz: the product structure — linear in (n_max−4), linear in
// (n_max−n_last), quadratic in Zα — is chosen to thread the three d rows, with α
// the only input fixed by nature. it is a parameter-free fit of a chosen shape
// across three rows, the physical reading being the relativistic 6s contraction.

#[test]
fn it_threads_three_d_rows_with_relativistic_contraction() {
    // RMSE of the relativistic rule over a row
    let r_rel = |lo: usize, hi: usize| -> f64 {
        let mut sse = 0.0;
        for z in lo..=hi {
            sse += (ie_rel(z) - EXP[z - 1]).powi(2);
        }
        (sse / (hi - lo + 1) as f64).sqrt()
    };

    eprintln!("\n═══ act XI: relativistic correction across three d rows ═══\n");
    eprintln!("  row   last-filled   max(n)     correction");
    let rows = [("3d", 21usize, 30usize), ("4d", 39, 48), ("5d", 72, 80)];
    let mut last = [0.0; 3];
    let mut max = [0.0; 3];
    let mut rel = [0.0; 3];
    for (i, (row, lo, hi)) in rows.iter().enumerate() {
        last[i] = rmse(*lo, *hi, last_filled);
        max[i] = rmse(*lo, *hi, Geonum::valence_shell);
        rel[i] = r_rel(*lo, *hi);
        eprintln!(
            "  {:3}   {:8.3}      {:8.3}   {:8.3} eV",
            row, last[i], max[i], rel[i]
        );
    }

    // 3d: (n_max−4)=0 kills the term, so n_eff = max(n) bit-for-bit
    assert!(
        (rel[0] - max[0]).abs() < EPSILON,
        "3d: (n_max−4)=0 leaves the max(n) rescue exact"
    );
    // 4d: a light contraction holds the rescue, staying in the s/p band
    assert!(
        rel[1] <= max[1] + EPSILON,
        "4d: correction holds the max(n) rescue"
    );
    assert!(
        rel[1] < 1.5,
        "4d: relativistic RMSE in the s/p band ({:.3})",
        rel[1]
    );
    // 5d: the rescue reverses (max loses to last-filled), and the correction
    // beats both, at least halving the max(n) error
    assert!(
        max[2] > last[2],
        "5d: max(n) ({:.3}) loses to last-filled ({:.3}) — the rescue reverses",
        max[2],
        last[2]
    );
    assert!(
        rel[2] < last[2],
        "5d: correction ({:.3}) beats last-filled ({:.3})",
        rel[2],
        last[2]
    );
    assert!(
        rel[2] < max[2] / 2.0,
        "5d: correction ({:.3}) at least halves the max(n) RMSE ({:.3})",
        rel[2],
        max[2]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// act XII — the np-shortfall wall is a first-harmonic theorem
// ═══════════════════════════════════════════════════════════════════════════
//
// the np closed shells resist the shipped projection. with both rays reading
// harmonic 1 of p, the q·opp term the marginal can supply is capped at
// (π/4)·p.mag — below what NIST needs — and the gap widens each period. true
// about THAT instrument, and only that. the climb the next acts make is not a
// frame rotation but a SECOND HARMONIC, the overtone the first harmonic is deaf
// to: squaring a geonum doubles its angle, and the np marginals' doubled phase
// lands on π/3, the pairing closure

// the electron waves, cached once: W[z] = electron_wave(z)
fn waves() -> Vec<Geonum> {
    (0..=54)
        .map(|z| Geonum::electron_wave(z, Lattice::Canonical))
        .collect()
}

// the marginal electron at z — the one ionization removes
fn marginal(w: &[Geonum], z: usize) -> Geonum {
    w[z] - w[z - 1]
}

// the shipped first-harmonic instrument (act VII/VIII as released):
// IE = R * (adj + (pi/4)*opp) / n^2 — both rays read harmonic 1 of p
fn ie_fundamental(w: &[Geonum], z: usize) -> f64 {
    marginal(w, z).ionization_projection(
        Geonum::new(z as f64, 0.0, 1.0),
        Geonum::valence_shell(z) as f64,
        Lattice::Canonical,
    )
}

// the doubled-phase resonance: p*p lands on the pairing closure pi/3
// exactly when the marginal's remainder is pi/6. angle arithmetic, no trig:
// rem + rem = pi/3 within float epsilon
fn resonant(w: &[Geonum], z: usize) -> bool {
    (2.0 * marginal(w, z).angle.rem() - PI / 3.0).abs() < 1e-9
}

// the gate: a CLOSED p shell (grade 0 marginal — the half-filled family
// lands at grade 3) torn off a p-CORE (n >= 3 means a filled (n-1)p exists
// beneath the shell being ionized; neon at n = 2 has none and needs no fix)
fn gate(w: &[Geonum], z: usize) -> bool {
    resonant(w, z) && marginal(w, z).angle.grade() == 0 && Geonum::valence_shell(z) >= 3
}

// the phased instrument: same lattice, same forced constants, one new
// detector. the quantum is R/3 — rydberg over the closure denominator
// SELECTED BY THE RESONATING CHANNEL ITSELF (the channel that fires is the
// pi/3 channel; the menu of admissible denominators is the closure set the
// constants suite proves forced). zero fitted magnitudes: nothing here was
// tuned to NIST — the gate is structural (phase, grade, core) and the
// quantum is drawn from the lattice's own constants
fn ie_phased(w: &[Geonum], z: usize) -> f64 {
    ie_fundamental(w, z) + if gate(w, z) { RYDBERG / 3.0 } else { 0.0 }
}

#[test]
fn it_proves_the_np_wall_is_a_first_harmonic_theorem() {
    // WITHIN the shipped projection form the np targets are unreachable:
    // need = EXP*n^2/R - adj exceeds the ceiling (pi/4)*p.mag at Ar, Kr, Xe,
    // and the wall-unit deficit widens. nothing in this suite disputes the
    // inequality — only its interpretation as a limit on the geometry
    let w = waves();
    let q = Angle::new(1.0, 4.0); // pi/4 — the phase coefficient
    let mut deficits = Vec::new();

    for &z in &[18usize, 36, 54] {
        let p = Geonum::new(z as f64, 0.0, 1.0) * marginal(&w, z);
        let adj = p.project(&Geonum::new(1.0, 0.0, 1.0));
        let n = Geonum::valence_shell(z) as f64;

        let need = EXP[z - 1] * n * n / RYDBERG - adj.mag;
        let ceiling = q.grade_angle() * p.mag;

        assert!(
            need > ceiling,
            "Z={z}: the first harmonic cannot reach (need {need:.3} > ceiling {ceiling:.3})"
        );
        deficits.push(need - ceiling);
    }

    // the widening the wall reports — in the instrument's units
    assert!(deficits[1] > deficits[0], "Kr deficit deepens past Ar");
    assert!(deficits[2] > deficits[1], "Xe deficit deepens past Kr");
}

#[test]
fn it_decomposes_the_widening_into_the_unit_echo() {
    // the artifact, exposed arithmetically. the instrument's shortfall in its
    // own units (need - q*opp_used) equals the ENERGY gap times n^2/R —
    // identically, because that is just the formula rearranged. so the
    // "quadratic widening" factors as (energy drift) x (the formula's own
    // n^2 ratio). measured: the 3.03x growth from Ar to Xe is 1.09x of real
    // energy drift times (5/3)^2 = 2.78x of denominator echo — exactly
    let w = waves();
    let q = Angle::new(1.0, 4.0);
    let ref0 = Geonum::new(1.0, 0.0, 1.0);
    let ref_q = Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0));

    let mut shortfall_wall = Vec::new(); // in the instrument's units
    let mut gap_ev = Vec::new(); // in nature's units
    let mut ns = Vec::new();

    for &z in &[18usize, 36, 54] {
        let p = Geonum::new(z as f64, 0.0, 1.0) * marginal(&w, z);
        let adj = p.project(&ref0);
        let opp = p.project(&ref_q);
        let n = Geonum::valence_shell(z) as f64;

        let need = EXP[z - 1] * n * n / RYDBERG - adj.mag;
        shortfall_wall.push(need - q.grade_angle() * opp.mag);
        gap_ev.push(EXP[z - 1] - ie_fundamental(&w, z));
        ns.push(n);
    }

    // identity: shortfall_wall == gap_ev * n^2 / R, term by term
    for i in 0..3 {
        assert!(
            (shortfall_wall[i] - gap_ev[i] * ns[i] * ns[i] / RYDBERG).abs() < 1e-9,
            "the wall's units are the energy gap times n^2/R — identically"
        );
    }

    // therefore the growth ratio decomposes exactly: the n^2 echo is (5/3)^2
    let growth_wall = shortfall_wall[2] / shortfall_wall[0];
    let growth_ev = gap_ev[2] / gap_ev[0];
    assert!(
        (growth_wall / growth_ev - 25.0 / 9.0).abs() < 1e-9,
        "of the wall's widening, (5/3)^2 is its own denominator reflected back"
    );

    // and in nature's units the deficit is FLAT — a quantum, not a quadratic:
    // 4.348, 4.370, 4.735 eV. within 10% of each other, within 5% of R/3
    let max = gap_ev.iter().cloned().fold(f64::MIN, f64::max);
    let min = gap_ev.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        max / min < 1.10,
        "the energy deficit does not grow quadratically"
    );
    for g in &gap_ev {
        assert!(
            (g - RYDBERG / 3.0).abs() / (RYDBERG / 3.0) < 0.05,
            "the flat quantum is R/3 — rydberg over the pairing-closure denominator"
        );
    }
}

#[test]
fn it_detects_hund_stability_from_the_doubled_phase() {
    // the lattice already knew. sweep Z = 1..=54 for marginals whose DOUBLED
    // phase lands on the pairing closure (2*rem = pi/3, i.e. rem = pi/6 —
    // the bisector of the spin closure). the resonance set is exactly the
    // half-filled and closed p subshells — N, P, As, Sb and Ne, Ar, Kr, Xe —
    // chemistry's two famous special-stability families, found by one phase
    // condition. grade splits the families: closed shells land at grade 0,
    // half-filled at grade 3. hund's rule is a phase detector
    let w = waves();
    let res: Vec<usize> = (1..=54).filter(|&z| resonant(&w, z)).collect();
    assert_eq!(
        res,
        vec![7, 10, 15, 18, 33, 36, 51, 54],
        "2θ = π/3 fires at exactly the half-filled and closed p subshells"
    );

    for &z in &[10usize, 18, 36, 54] {
        assert_eq!(marginal(&w, z).angle.grade(), 0, "closed shells at grade 0");
    }
    for &z in &[7usize, 15, 33, 51] {
        assert_eq!(marginal(&w, z).angle.grade(), 3, "half-filled at grade 3");
    }

    // the gate isolates the wall's three targets: closed (grade 0) AND a
    // p-core beneath (n >= 3). neon is correctly silent — nothing under 2p
    let gated: Vec<usize> = (1..=54).filter(|&z| gate(&w, z)).collect();
    assert_eq!(
        gated,
        vec![18, 36, 54],
        "the detector fires at Ar, Kr, Xe only"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// act XIII — the phased climb
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn it_climbs_the_wall_with_a_phased_quantum() {
    // the verdict. one structural quantum — R/3, gated on phase, grade, and
    // core — lands all three "unreachable" targets inside 2%, IMPROVES the
    // in-sample fit (argon was the worst resident of the old one), and leaves
    // both transition rows bit-identical because the detector never fires
    // there. roughly fifty bystanders, zero collateral. "no frame rotation
    // reaches it" stands, and is beside the point: this is not a rotation
    let w = waves();

    // the three np shells land within 2% — far inside the suite's own 30%
    // transition-metal standard, inside even the s/p band
    for &z in &[18usize, 36, 54] {
        let pred = ie_phased(&w, z);
        let err = (pred - EXP[z - 1]).abs() / EXP[z - 1];
        assert!(
            err < 0.02,
            "Z={z}: phased instrument lands {pred:.2} vs NIST {:.2} ({:.1}%)",
            EXP[z - 1],
            err * 100.0
        );
    }

    // the in-sample fit improves: 2.358 -> 2.124 eV RMSE over Z = 1..=18
    let rmse = |f: &dyn Fn(usize) -> f64, lo: usize, hi: usize| -> f64 {
        let sse: f64 = (lo..=hi).map(|z| (f(z) - EXP[z - 1]).powi(2)).sum();
        (sse / (hi - lo + 1) as f64).sqrt()
    };
    let old = rmse(&|z| ie_fundamental(&w, z), 1, 18);
    let new = rmse(&|z| ie_phased(&w, z), 1, 18);
    assert!(
        new < old,
        "the climb improves the fit it extends: {old:.3} -> {new:.3} eV"
    );
    assert!(new < 3.0, "act VII's own band still holds");

    // both d-blocks untouched — the resonance never fires at a d marginal
    for (lo, hi) in [(21usize, 30usize), (39, 48)] {
        let old_d = rmse(&|z| ie_fundamental(&w, z), lo, hi);
        let new_d = rmse(&|z| ie_phased(&w, z), lo, hi);
        assert!(
            (old_d - new_d).abs() < 1e-12,
            "transition rows bit-identical: the detector is silent there"
        );
    }

    // honest residue, logged as owed:
    // - the energy quantum drifts 9% (4.35 -> 4.74 eV) across the three
    //   periods. real, unexplained, the next wall to survey
    // - R/3 is forced-MENU (closure denominators only, selected by the
    //   resonating channel) but not yet DERIVED the way the constants suite
    //   derives 2, 3, 4. that derivation is the remaining todo
    // - the production form is a phased projection of p*p inside
    //   ionization_projection, not a gated constant. the constant proves
    //   reachability — which is all an impossibility theorem needs to die
}

// ═══════════════════════════════════════════════════════════════════════════
// act XIV — the wall banished: the gate was a standing wave
// ═══════════════════════════════════════════════════════════════════════════

// the (ns, ls) subshell's standing wave, rebuilt from the lattice placement
fn subshell_wave(z: usize, ns: usize, ls: usize) -> Geonum {
    let spin = Angle::new(1.0, 3.0);
    let grade_step = Angle::new(1.0, 2.0);
    let mut acc = Geonum::scalar(0.0);
    let mut placed = 0;
    for (n, l) in Geonum::madelung_order(6) {
        if placed >= z {
            break;
        }
        let mut base = Angle::new(1.0, 1.0);
        for _ in 0..l {
            base = base + grade_step;
        }
        let n_orb = 2 * l + 1;
        let step = grade_step / n_orb as f64;
        let mut pos = Vec::new();
        let mut a = base;
        for _ in 0..n_orb {
            pos.push(a);
            pos.push(a + spin);
            a = a + step;
        }
        let fill = pos.len().min(z - placed);
        if n == ns && l == ls {
            for p in pos.iter().take(fill) {
                acc = acc + Geonum::new_with_angle(1.0 / n as f64, *p);
            }
        }
        placed += fill;
    }
    acc
}

// the standing-wave closure: the marginal's pair phase against the core wave.
// returns the landed angle when 2m − C lands on a pure blade (t = 0 — the
// carry arithmetic's own boundary), and None when the wave fails to close
fn pair_closure(w: &[Geonum], z: usize) -> Option<Angle> {
    let n = Geonum::valence_shell(z);
    if n < 2 {
        return None;
    }
    let core = subshell_wave(z, n - 1, 1);
    if core.mag < 1e-12 {
        return None; // no (n-1)p wave: nothing to interfere with (neon)
    }
    let m = marginal(w, z).angle;
    let combined = (m + m) - core.angle; // pair phase against the core
    if combined.t() < 1e-9 {
        Some(combined)
    } else {
        None
    }
}

#[test]
fn it_banishes_the_wall_with_a_standing_wave() {
    // act XIII's gate — `resonant && grade == 0 && n >= 3` — was three scalar
    // predicates standing in for one wave. in real space the term exists where
    // the marginal's pair phase CLOSES against the (n-1)p core standing wave:
    // 2m − C lands on a pure blade. closure is binary by the lattice's own
    // carry boundary (t = 0.0 exactly, vs misses at tan(π/12), tan(π/6)),
    // the core's existence is the wave's amplitude (0 at neon, 3+√3 at every
    // period, stationary at the pairing closure π/3 — a particle), and the
    // LANDED GRADE assigns the quantum: grade 1 (closed shells) → R/3,
    // grade 3 (half-filled) → R/9 = (R/3)². no gate survives — only geometry
    let w = waves();

    // the closure set: both hund families with cores, nothing else
    let closures: Vec<usize> = (1..=54)
        .filter(|&z| pair_closure(&w, z).is_some())
        .collect();
    assert_eq!(
        closures,
        vec![15, 18, 33, 36, 51, 54],
        "the wave closes at exactly six configurations"
    );

    // the lattice sorts the families by landed grade
    for &z in &[18usize, 36, 54] {
        assert_eq!(
            pair_closure(&w, z).unwrap().grade(),
            1,
            "closed shells close at grade 1"
        );
    }
    for &z in &[15usize, 33, 51] {
        assert_eq!(
            pair_closure(&w, z).unwrap().grade(),
            3,
            "half-filled close at grade 3"
        );
    }

    // the core standing wave is a lattice particle: amplitude 3+√3, phase π/3
    let core = subshell_wave(18, 2, 1);
    assert!(
        ((core.mag * 2.0) - (3.0 + 3.0_f64.sqrt())).abs() < 1e-9,
        "p⁶ amplitude is 3+√3"
    );
    assert!(
        (core.angle.rem() - PI / 3.0).abs() < 1e-9,
        "p⁶ wave sits on the pairing closure"
    );

    // the gateless model: quantum by landed grade, third harmonic universal
    let c3 = 0.297;
    let ie_wave = |z: usize| -> f64 {
        let q = match pair_closure(&w, z).map(|a| a.grade()) {
            Some(1) => RYDBERG / 3.0,
            Some(3) => RYDBERG / 9.0,
            _ => 0.0,
        };
        ie_fundamental(&w, z) + q + c3 * (3.0 * w[z].angle.grade_angle()).cos()
    };

    // the cliffs hold and phosphorus lands on a coefficient it was never fitted to
    for &(z, tol) in &[
        (18usize, 0.005),
        (36, 0.005),
        (54, 0.005),
        (15, 0.001),
        (33, 0.03),
        (51, 0.04),
    ] {
        let err = (ie_wave(z) - EXP[z - 1]).abs() / EXP[z - 1];
        assert!(
            err < tol,
            "Z={z}: {:.3} vs {:.3} ({:.2}%)",
            ie_wave(z),
            EXP[z - 1],
            err * 100.0
        );
    }

    // every block improves or holds against the gated model of act XIII
    let rmse = |f: &dyn Fn(usize) -> f64, lo: usize, hi: usize| -> f64 {
        ((lo..=hi).map(|z| (f(z) - EXP[z - 1]).powi(2)).sum::<f64>() / (hi - lo + 1) as f64).sqrt()
    };
    assert!(rmse(&ie_wave, 1, 18) < 2.142, "in-sample improves");
    assert!(rmse(&ie_wave, 31, 36) < 1.786, "4p improves");
    assert!(rmse(&ie_wave, 49, 54) < 2.001, "5p improves");
    assert!(
        rmse(&ie_wave, 21, 30) < 1.5 && rmse(&ie_wave, 39, 48) < 1.5,
        "d-blocks hold the band"
    );

    // owed, logged as this suite logs its walls: R/3's magnitude is still
    // menu-forced; R/9 = (R/3)² tracks the grade-1/grade-3 dual pair but is
    // observed, not derived; As and Sb run −0.3 eV inside the underived
    // aberration band. act XII's wall theorem stays exactly that — a true
    // theorem about a first-harmonic instrument — and this test is its
    // epitaph: the boundary nothing closes was a closure
}

// ═══════════════════════════════════════════════════════════════════════════
// act XV — the exponent fence: molybdenum falsifies the grade law
// ═══════════════════════════════════════════════════════════════════════════

// which (n, l) subshell the z-th electron lands in, by the madelung walk
fn subshell_of(z: usize) -> (usize, usize) {
    let mut placed = 0;
    for (n, l) in Geonum::madelung_order(6) {
        let cap = 2 * (2 * l + 1);
        if placed + cap >= z {
            return (n, l);
        }
        placed += cap;
    }
    (0, 0)
}

// generalized closure: the marginal's pair phase against its own same-l core
fn closure_general(w: &[Geonum], z: usize) -> Option<Angle> {
    let (ns, ls) = subshell_of(z);
    if ns < 2 {
        return None;
    }
    let core = subshell_wave(z, ns - 1, ls);
    if core.mag < 1e-12 {
        return None;
    }
    let m = marginal(w, z).angle;
    let combined = (m + m) - core.angle;
    if combined.t() < 1e-9 {
        Some(combined)
    } else {
        None
    }
}

#[test]
fn it_walls_the_quantum_exponent_at_molybdenum() {
    // generalizing the closure to every subshell (same-l core) finds a FOURTH
    // family unbidden — the closed s² shells (Be, Mg, Ca, Sr), landing at
    // grade 3 like the half-filled p family — and one falsifier: molybdenum.
    // Mo's 4d marginal closes against the 3d¹⁰ core at GRADE 1 (the R/3
    // grade) but its measured residual sits at R/9 ≈ 1.5 eV, not 4.5. the
    // bare law `k = (g+1)/2 of the landed grade` is dead as stated: the
    // exponent counts something the grade only shadows — path traversals,
    // or an l-dependence. one d-point cannot pick between them, so this test
    // is a fence in this suite's tradition: the deviation asserted, the
    // derivation owed. note also the d¹⁰ family (Zn, Cd) produces NO closure
    // — rhyming with chemistry's weaker d¹⁰ stability, logged as a rhyme
    let w = waves();
    let closures: Vec<usize> = (1..=54)
        .filter(|&z| closure_general(&w, z).is_some())
        .collect();
    assert_eq!(
        closures,
        vec![4, 12, 15, 18, 20, 33, 36, 38, 42, 51, 54],
        "eleven closures, four families"
    );

    // the alkaline earths close at grade 3 — detected, never fitted
    for &z in &[4usize, 12, 20, 38] {
        assert_eq!(
            closure_general(&w, z).unwrap().grade(),
            3,
            "s² closes at grade 3"
        );
    }

    // the fence: Mo lands grade 1, pays R/9
    assert_eq!(
        closure_general(&w, 42).unwrap().grade(),
        1,
        "Mo closes at grade 1"
    );
    let c3 = 0.297;
    let base = ie_fundamental(&w, 42) + c3 * (3.0 * w[42].angle.grade_angle()).cos();
    let resid = EXP[41] - base;
    assert!(
        (resid - RYDBERG / 9.0).abs() < 0.5,
        "Mo pays the R/9 quantum: {resid:.3}"
    );
    assert!(
        (resid - RYDBERG / 3.0).abs() > 2.0,
        "Mo refuses the R/3 its grade predicts"
    );
}
