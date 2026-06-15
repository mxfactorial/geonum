//! chemistry trait implementation
//!
//! the periodic table from blade arithmetic. the madelung filling order is a
//! blade-chain walk (tier T = n+l is a blade); the electron shell is the
//! [`wave_sum`](crate::GeoCollection::wave_sum) of electron geonums placed at the
//! lattice angles; ionization energy, electron affinity, and electronegativity
//! are projections of the marginal electron over the valence shell. validated
//! against NIST in the chemistry test suites.

use crate::{Angle, GeoCollection, Geonum};

/// rydberg energy (eV) — the hydrogenic binding scale
const RYDBERG: f64 = 13.6;

/// fine-structure constant — fixed by nature, drives the relativistic contraction
const ALPHA: f64 = 1.0 / 137.035_999_084;

/// the 1/n radial law (Bohr momentum ∝ 1/n)
fn bohr(n: usize) -> f64 {
    1.0 / n as f64
}

/// the constants the model runs on. `Canonical` is the assignment the suite proves
/// forced: spin = π/3 (the pairing closure), radial = 1/n, q = π/4 (the phase
/// coefficient). `Custom` varies them to probe why the canonical one is forced
/// (tests/chem_constants_test.rs). spread = π/2 is the lattice itself and is
/// never varied, so it is not part of the configuration.
#[derive(Clone, Copy)]
pub enum Lattice {
    Canonical,
    Custom {
        spin: Angle,
        radial: fn(usize) -> f64,
        q: Angle,
    },
}

impl Lattice {
    fn constants(self) -> (Angle, fn(usize) -> f64, Angle) {
        match self {
            Lattice::Canonical => (Angle::new(1.0, 3.0), bohr, Angle::new(1.0, 4.0)),
            Lattice::Custom { spin, radial, q } => (spin, radial, q),
        }
    }
}

/// the orbital positions of a subshell at grade l: 2l+1 orbitals stepping across
/// the π/2 quadrant from `base`, each with its spin pair one `spin` offset away
fn grade_positions(base: Angle, l: usize, spread: Angle, spin: Angle) -> Vec<Angle> {
    let n_orb = 2 * l + 1;
    let orbital_step = spread / n_orb as f64;
    let mut pos = Vec::new();
    let mut angle = base;
    for _ in 0..n_orb {
        pos.push(angle);
        pos.push(angle + spin);
        angle = angle + orbital_step;
    }
    pos
}

/// the last-filled shell — the naive outer-shell rule the d-block madelung
/// inversion breaks (4s fills before 3d). internal: feeds the relativistic
/// contraction's overshoot term. the spatial rule consumers want is `valence_shell`
fn last_filled(z: usize) -> usize {
    let mut placed = 0;
    let mut n = 1;
    for (nn, l) in Geonum::madelung_order(6) {
        if placed >= z {
            break;
        }
        n = nn;
        placed += (2 * (2 * l + 1)).min(z - placed);
    }
    n
}

/// which (n, l) subshell the z-th electron lands in, by the madelung walk.
/// the affinity sign turns on whether the added electron continues a subshell
/// or opens a new one — `subshell_of(z + 1) == subshell_of(z)`
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

/// the unsigned binding of the (z+1)th electron stepping on — a screened (+1)
/// nucleus, projected over the anion's valence shell. shared by `electron_affinity`
/// (signed) and `electronegativity`
fn affinity_binding(z: usize, lattice: Lattice) -> f64 {
    let screened = Geonum::new(1.0, 0.0, 1.0);
    let marginal = Geonum::electron_wave(z + 1, lattice) - Geonum::electron_wave(z, lattice);
    marginal.ionization_projection(screened, Geonum::valence_shell(z + 1) as f64, lattice)
}

/// the periodic table as blade arithmetic — an extension trait on [`Geonum`]
pub trait Chemistry: Sized {
    /// the madelung filling order as (n, l) pairs up to principal shell `max_n`,
    /// walked as a blade chain: tier T = n+l is a blade, each tier's diagonal
    /// trades l for n
    fn madelung_order(max_n: usize) -> Vec<(usize, usize)>;

    /// the `z` electron geonums, placed in madelung order at the lattice angles
    /// with the radial law — the shell as a collection
    fn electron_shell(z: usize, lattice: Lattice) -> GeoCollection;

    /// the electron shell summed into one wave — `electron_shell(z).wave_sum()`
    fn electron_wave(z: usize, lattice: Lattice) -> Self;

    /// the spatial valence shell — the largest n across filled subshells
    fn valence_shell(z: usize) -> usize;

    /// the relativistic effective valence shell: `valence_shell` contracted toward
    /// the inner d by (Zα)² × periods-since-the-d-inversion × the overshoot. for
    /// the heavy 5d row where the 6s contraction reverses the max(n) rule
    fn relativistic_valence_shell(z: usize) -> f64;

    /// the low-level projection: `self` (a marginal electron) scaled by `nuclear`,
    /// projected over `n²` through the phase coefficient — the building block of
    /// every ionization observable
    fn ionization_projection(&self, nuclear: Self, n: f64, lattice: Lattice) -> f64;

    /// ionization energy in eV of the species with `z` protons and `electrons`
    /// electrons. IE1 = `ionization_energy(z, z)`, IE2 = `(z, z-1)`, and successive
    /// ionization follows. the nuclear factor carries the exposed core charge, so
    /// deep stripping recovers the hydrogenic Z² limit
    fn ionization_energy(z: usize, electrons: usize, lattice: Lattice) -> f64;

    /// signed electron affinity in eV: the next electron stepping on. bound
    /// (positive) when it extends the open subshell, repulsive (negative) when it
    /// opens a fresh closure — the sign is `subshell_of(z + 1) == subshell_of(z)`
    fn electron_affinity(z: usize, lattice: Lattice) -> f64;

    /// Mulliken electronegativity, (IE1 + EA binding)/2
    fn electronegativity(z: usize, lattice: Lattice) -> f64;
}

impl Chemistry for Geonum {
    fn madelung_order(max_n: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut tier = Geonum::new(1.0, 1.0, 2.0); // π/2, blade 1 — the 1s tier
        while tier.angle.blade() < 2 * max_n {
            let t = tier.angle.blade(); // tier T = n+l, the rotation count
            let l_start = (t - 1) / 2; // diagonal start: the largest l with l < n
            let mut n = t - l_start;
            let mut l = l_start;
            loop {
                if n <= max_n {
                    out.push((n, l));
                }
                if l == 0 {
                    break;
                }
                n += 1; // the diagonal step: trade one l for one n
                l -= 1;
            }
            tier = tier.increment_blade(); // π/2 rotation to the next tier
        }
        out
    }

    fn electron_shell(z: usize, lattice: Lattice) -> GeoCollection {
        let (spin, radial, _) = lattice.constants();
        let spread = Angle::new(1.0, 2.0); // π/2 — the lattice
        let mut electrons = Vec::new();
        let mut placed = 0;
        for (n, l) in Geonum::madelung_order(6) {
            if placed >= z {
                break;
            }
            let mut base = Angle::new(1.0, 1.0); // π
            for _ in 0..l {
                base = base + spread;
            }
            let positions = grade_positions(base, l, spread, spin);
            let to_fill = positions.len().min(z - placed);
            let mag = radial(n);
            for &p in positions.iter().take(to_fill) {
                electrons.push(Geonum::new_with_angle(mag, p));
            }
            placed += to_fill;
        }
        GeoCollection::from(electrons)
    }

    fn electron_wave(z: usize, lattice: Lattice) -> Geonum {
        Geonum::electron_shell(z, lattice).wave_sum()
    }

    fn valence_shell(z: usize) -> usize {
        let mut placed = 0;
        let mut n = 1;
        for (nn, l) in Geonum::madelung_order(6) {
            if placed >= z {
                break;
            }
            if nn > n {
                n = nn; // running max — the spatial valence shell
            }
            placed += (2 * (2 * l + 1)).min(z - placed);
        }
        n
    }

    fn relativistic_valence_shell(z: usize) -> f64 {
        let n_max = Geonum::valence_shell(z) as f64;
        let n_last = last_filled(z) as f64;
        let lorentz = (z as f64 * ALPHA).powi(2);
        let periods_since_inversion = (n_max - 4.0).max(0.0);
        n_max - lorentz * periods_since_inversion * (n_max - n_last)
    }

    fn ionization_projection(&self, nuclear: Self, n: f64, lattice: Lattice) -> f64 {
        let (_, _, q) = lattice.constants();
        let p = nuclear * *self; // self = the marginal electron
        let ref0 = Geonum::new(1.0, 0.0, 1.0);
        let ref_q = Geonum::new_with_angle(1.0, Angle::new(1.0, 2.0));
        let adj = p.project(&ref0);
        let opp = p.project(&ref_q);
        RYDBERG * (adj.mag + q.grade_angle() * opp.mag) / (n * n)
    }

    fn ionization_energy(z: usize, electrons: usize, lattice: Lattice) -> f64 {
        // the exposed core charge: the electrons-1 inner electrons screen
        // electrons-1 protons, leaving z-(electrons-1) exposed to the marginal
        let nucleus = Geonum::new(z as f64, 0.0, 1.0);
        let exposed = Geonum::new((z - (electrons - 1)) as f64, 0.0, 1.0);
        let marginal = Geonum::electron_wave(electrons, lattice)
            - Geonum::electron_wave(electrons - 1, lattice);
        marginal.ionization_projection(
            nucleus.geo(&exposed),
            Geonum::valence_shell(electrons) as f64,
            lattice,
        )
    }

    fn electron_affinity(z: usize, lattice: Lattice) -> f64 {
        let bind = affinity_binding(z, lattice);
        // bound iff the added (z+1)th electron extends the open subshell; unbound
        // iff it is the first occupant of a fresh closure that repels it. grade
        // cannot sign this — the alkali spin-pair (bound) and the noble shell
        // jump (unbound) both land grade 2 — but subshell continuity can
        if subshell_of(z + 1) == subshell_of(z) {
            bind
        } else {
            -bind
        }
    }

    fn electronegativity(z: usize, lattice: Lattice) -> f64 {
        (Geonum::ionization_energy(z, z, lattice) + affinity_binding(z, lattice)) / 2.0
    }
}
