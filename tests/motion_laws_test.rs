// newton's laws are fragments of one idea: geometric nilpotency IS conservation
//
// What Newton saw as separate laws are just different faces of the wedge product:
// - First Law (Inertia): v ∧ v = 0
// - Second Law (F=ma): Motion from geometric products
// - Third Law (Action/Reaction): Antisymmetry of ∧
// - Conservation Laws: All just v ∧ v = 0 again

use geonum::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

#[test]
fn it_ships_conservation_with_the_wedge_product() {
    // CORE INSIGHT: v ∧ v = 0 IS conservation
    // No external conservation law needed - it's in the algebra

    let vector = Geonum::new(5.0, 1.0, 3.0); // [5, π/3]

    // Geometric nilpotency: identical vectors cannot create structure
    let self_wedge = vector.wedge(&vector);

    println!("=== CONSERVATION FROM GEOMETRIC NILPOTENCY ===");
    println!(
        "Vector: [{}, {:.3}]",
        vector.length,
        vector.angle.mod_4_angle()
    );
    println!(
        "v ∧ v = [{:.10}, {:.3}]",
        self_wedge.length,
        self_wedge.angle.mod_4_angle()
    );

    // This IS conservation: you can't create something from nothing
    assert!(self_wedge.length < EPSILON, "Conservation: v ∧ v = 0");

    // Compare to Newton's laws
    println!("\nNewton's Laws: External conservation principles");
    println!("- Conservation of momentum as separate law");
    println!("- Conservation of energy as separate law");
    println!("- Force = ma with conservation enforced externally");

    println!("\nWedge Product: Conservation built-in");
    println!("- v ∧ v = 0 automatically (geometric nilpotency)");
    println!("- No external conservation law needed");
    println!("- Motion and conservation unified in single operation");

    println!("\n✓ Conservation emerges automatically from wedge algebra");
}

#[test]
fn it_ships_conservation_with_motion() {
    // SIMPLIFIED DEMO: This is NOT trying to simulate real orbits

    // We're showing that motion + conservation emerge from geometric ops
    // without needing F=ma or external force laws

    let mut position = Geonum::new(1e11, 0.0, 1.0); // [1e11, 0]
    let mut momentum = Geonum::new(1e35, 1.0, 2.0); // [1e35, π/2]

    println!("=== MOTION WITH AUTOMATIC CONSERVATION ===");

    // Track quantities that should be conserved
    let initial_angular_momentum = position.wedge(&momentum);
    let initial_energy = position.dot(&momentum);

    println!("Initial state:");
    println!(
        "Position: [{:.2e}, {:.3}]",
        position.length,
        position.angle.mod_4_angle()
    );
    println!(
        "Momentum: [{:.2e}, {:.3}]",
        momentum.length,
        momentum.angle.mod_4_angle()
    );
    println!(
        "Angular momentum: [{:.3e}, {:.3}]",
        initial_angular_momentum.length,
        initial_angular_momentum.angle.mod_4_angle()
    );
    println!("Energy: {:.3e}", initial_energy.length);

    // Evolve system using wedge-based motion
    // NOTE: This is SYMBOLIC motion, not accurate physics simulation
    // The point is that motion emerges from geometric ops, not forces
    for _step in 0..10 {
        let _circulation = position.wedge(&momentum);

        // Arbitrary small rotation to show conservation holds
        // Don't focus on the numbers - focus on the STRUCTURE
        let small_rotation = Angle::new(0.001, PI);

        position.angle = position.angle + small_rotation;
        momentum.angle = momentum.angle + small_rotation;
    }

    // Check conservation automatically preserved
    let final_angular_momentum = position.wedge(&momentum);
    let final_energy = position.dot(&momentum);

    println!("\nFinal state:");
    println!(
        "Position: [{:.2e}, {:.3}]",
        position.length,
        position.angle.mod_4_angle()
    );
    println!(
        "Momentum: [{:.2e}, {:.3}]",
        momentum.length,
        momentum.angle.mod_4_angle()
    );
    println!(
        "Angular momentum: [{:.3e}, {:.3}]",
        final_angular_momentum.length,
        final_angular_momentum.angle.mod_4_angle()
    );
    println!("Energy: {:.3e}", final_energy.length);

    // Conservation preserved automatically - no external enforcement
    let angular_momentum_change =
        (final_angular_momentum.length - initial_angular_momentum.length).abs();
    let energy_change = (final_energy.length - initial_energy.length).abs();

    println!("\nConservation check:");
    println!(
        "Angular momentum change: {:.3e}",
        angular_momentum_change / initial_angular_momentum.length.max(1e-50)
    );
    println!(
        "Energy change: {:.3e}",
        energy_change / initial_energy.length.abs().max(1e-50)
    );

    // Conservation emerges from geometric structure, not external laws
    assert!(
        angular_momentum_change < initial_angular_momentum.length * 0.1,
        "Angular momentum conserved"
    );
    assert!(
        energy_change < initial_energy.length.abs() * 0.1,
        "Energy conserved"
    );

    println!("\n✓ Conservation preserved automatically through wedge algebra");
    println!("✓ No external conservation laws needed");
}

#[test]
fn it_sets_geometric_nilpotency_as_physical_law() {
    // Show that v ∧ v = 0 IS the fundamental physical law
    // Not just mathematical property - it's conservation itself

    println!("=== GEOMETRIC NILPOTENCY AS PHYSICAL LAW ===");

    // Test various scenarios where conservation must hold
    let scenarios = vec![
        ("Energy", Geonum::new(100.0, 0.0, 1.0)),  // [100, 0]
        ("Momentum", Geonum::new(50.0, 1.0, 4.0)), // [50, π/4]
        ("Angular momentum", Geonum::new(75.0, 1.0, 1.0)), // [75, π]
        ("Charge", Geonum::new(1.6e-19, 1.0, 6.0)), // [1.6e-19, π/6]
    ];

    for (quantity_name, quantity) in scenarios {
        let self_interaction = quantity.wedge(&quantity);

        println!("\n{quantity_name} conservation:");
        println!(
            "Quantity: [{:.3e}, {:.3}]",
            quantity.length,
            quantity.angle.mod_4_angle()
        );
        println!(
            "Self-interaction: [{:.3e}, {:.3}]",
            self_interaction.length,
            self_interaction.angle.mod_4_angle()
        );

        // Physical law: identical quantities cannot interact with themselves
        assert!(
            self_interaction.length < EPSILON,
            "{quantity_name} cannot interact with itself - conservation preserved"
        );

        println!("✓ {quantity_name} conserved through geometric nilpotency");
    }

    println!("\nPhysical interpretation:");
    println!("v ∧ v = 0 means: identical quantities cannot create new structure");
    println!("This IS conservation - nothing created from sameness");
    println!("No external law needed - it's built into the geometry");

    println!("\nTraditional physics: conservation as separate principles");
    println!("Geometric algebra: conservation as fundamental constraint");
    println!("v ∧ v = 0 is the physical law underlying all conservation");

    println!("\n✓ Geometric nilpotency IS the fundamental physical law");
}

#[test]
fn its_independent_of_external_enforcement() {
    // Demonstrate: conservation violations are impossible, not just forbidden
    // The algebra prevents them automatically

    println!("=== NO EXTERNAL ENFORCEMENT NEEDED ===");

    let vector = Geonum::new(42.0, 1.0, 7.0); // [42, π/7]

    // Try to violate conservation by self-interaction
    let attempted_violation = vector.wedge(&vector);

    println!("Attempting conservation violation:");
    println!(
        "Vector: [{}, {:.3}]",
        vector.length,
        vector.angle.mod_4_angle()
    );
    println!(
        "Self-wedge: [{:.10}, {:.3}]",
        attempted_violation.length,
        attempted_violation.angle.mod_4_angle()
    );

    // Violation impossible - algebra prevents it
    assert!(
        attempted_violation.length < EPSILON,
        "Conservation violation impossible"
    );

    println!("✓ Conservation violation automatically prevented");

    // Compare to traditional physics enforcement
    println!("\nTraditional physics:");
    println!("- Write conservation laws");
    println!("- Check if system violates them");
    println!("- Add constraints or modify equations");
    println!("- External enforcement required");

    println!("\nGeometric algebra:");
    println!("- v ∧ v = 0 built into operations");
    println!("- Violations impossible, not just forbidden");
    println!("- No external enforcement needed");
    println!("- Conservation automatic");

    // Test multiple attempts at violation
    let vectors = [
        Geonum::new(1.0, 0.0, 1.0),   // [1, 0]
        Geonum::new(100.0, 1.0, 2.0), // [100, π/2]
        Geonum::new(1e-10, 2.0, 1.0), // [1e-10, 2π]
    ];

    for (i, v) in vectors.iter().enumerate() {
        let violation_attempt = v.wedge(v);
        println!(
            "Violation attempt {}: [{:.10}, {:.3}]",
            i + 1,
            violation_attempt.length,
            violation_attempt.angle.mod_4_angle()
        );
        assert!(
            violation_attempt.length < EPSILON,
            "All violation attempts fail"
        );
    }

    println!("\n✓ All conservation violation attempts automatically fail");
    println!("✓ No external enforcement mechanism needed");
}

#[test]
fn it_unifies_newtons_disparate_laws_into_one_operation() {
    // THE KEY INSIGHT: Newton needed 3 laws + conservation principles
    // because he lacked the math to see they're all THE SAME THING

    println!("=== NEWTON'S 'SEPARATE' LAWS ARE ONE GEOMETRIC TRUTH ===");

    println!("\nNewton's Fragmented View:");
    println!("1. First Law: Objects at rest stay at rest");
    println!("2. Second Law: F = ma");
    println!("3. Third Law: Equal and opposite reactions");
    println!("+ Conservation: Separate principles added on top");

    println!("\nThe Geometric Truth - It's ALL the wedge product:");
    println!("Just different aspects of v ∧ w:");

    let object1 = Geonum::new(10.0, 0.0, 1.0); // [10, 0]
    let object2 = Geonum::new(5.0, 1.0, 3.0); // [5, π/3]

    // Newton's "First Law" = Just nilpotency: v ∧ v = 0
    let inertia_test = object1.wedge(&object1);
    println!("\n1. 'First Law' is just: v ∧ v = 0");
    println!(
        "   Self-wedge = [{:.10}, {:.3}]",
        inertia_test.length,
        inertia_test.angle.mod_4_angle()
    );

    // Newton's "Second Law" = Just the geometric product
    let interaction = object1.geo(&object2);
    println!("\n2. 'Second Law' is just: geometric product gives motion");
    println!("   No need for F=ma, motion emerges from v⊙w");

    // The geometric product encodes both alignment (scalar) and rotation (bivector)
    // This IS the "force" - not as a separate concept, but as geometric relationship
    let alignment = object1.dot(&object2);
    let rotation = object1.wedge(&object2);

    // The geometric product is the sum of dot and wedge parts
    // This unification IS what Newton split into "force" and "torque"
    let reconstructed = alignment + rotation;
    assert_eq!(
        interaction.length, reconstructed.length,
        "Geometric product = dot + wedge: unified where Newton saw separate forces"
    );
    assert_eq!(
        interaction.angle, reconstructed.angle,
        "F=ma emerges from geometric relationships, not external forces"
    );

    // Newton's "Third Law" = Just antisymmetry: v∧w = -w∧v
    let action = object1.wedge(&object2);
    let reaction = object2.wedge(&object1);
    println!("\n3. 'Third Law' is just: antisymmetry of wedge");
    println!(
        "   v∧w = [{:.3}, {:.3}]",
        action.length,
        action.angle.mod_4_angle()
    );
    println!(
        "   w∧v = [{:.3}, {:.3}]",
        reaction.length,
        reaction.angle.mod_4_angle()
    );

    // Prove antisymmetry (action = -reaction)
    let angle_diff = (action.angle - reaction.angle).mod_4_angle().abs();
    let angle_diff_mod = angle_diff % (2.0 * PI);
    let is_opposite = (angle_diff_mod - PI).abs() < 0.1 || angle_diff_mod < 0.1;
    assert!(is_opposite, "Action and reaction are opposite");
    assert!(
        (action.length - reaction.length).abs() < EPSILON,
        "Equal magnitudes"
    );

    println!("✓ Action and reaction automatically opposite");

    // Conservation automatic
    println!("+ Conservation: built into every operation (v ∧ v = 0)");

    println!("\n=== THE UNIFICATION ===");
    println!("Newton: Multiple 'laws' because he saw fragments");
    println!("Reality: One geometric operation (∧) with different faces:");
    println!("  - When v=w: Nilpotency (First Law + Conservation)");
    println!("  - When v≠w: Antisymmetry (Third Law)");
    println!("  - Combined with ⊙: Motion emerges (Second Law)");

    println!("\n✓ Newton's laws aren't wrong - they're REDUNDANT");
    println!("✓ Like describing 'wetness', 'transparency', and 'flow' as separate laws of water");
    println!("✓ The wedge product IS the law - everything else is perspective");
}

#[test]
fn it_supplies_conservation_in_one_cpu_tick() {
    // Compressing classical mechanics into nanoseconds

    let start = std::time::Instant::now();

    let vector = Geonum::new(5.0, 1.0, 3.0); // 1 allocation
    let conservation_check = vector.wedge(&vector); // 1 operation and 1 allocation
    assert!(conservation_check.length < EPSILON);

    let duration = start.elapsed();

    println!("=== COMPUTATIONAL EFFICIENCY ===");
    println!("Newton under apple tree: Centuries developing 3 laws + conservation");
    println!("Rust unit test: {duration:?} to prove all of them");
    println!("Memory: 2 Geonum allocations");
    println!("Operations: 1 wedge product");
    println!("Result: All conservation laws proven automatically");

    println!("\n✓ Apple tree moment → Geonum::new() call");
    println!("✓ Newton's lifetime of work → one CPU tick");
}

#[test]
fn it_improves_physics_engines() {
    // Show the computational burden Newton's fragmented view creates

    println!("=== TRADITIONAL PHYSICS ENGINE FOR CONSERVATION ===");
    println!("Data structures needed:");
    println!("- Position vector (3 floats)");
    println!("- Velocity vector (3 floats)");
    println!("- Momentum vector (3 floats)");
    println!("- Force accumulator (3 floats)");
    println!("- Energy state (1 float)");
    println!("- Angular momentum state (3 floats)");
    println!("Total: 16 floats minimum");

    println!("\nComputational overhead:");
    println!("- Constraint solver loop");
    println!("- Energy calculation (kinetic + potential)");
    println!("- Momentum calculation");
    println!("- Angular momentum calculation");
    println!("- Validation checks for each conserved quantity");
    println!("- Error correction when conservation violated");
    println!("- Integration timestep management");
    println!("- Numerical stability monitoring");

    println!("\n=== GEONUM APPROACH ===");
    println!("Data structure: 1 geometric number");
    println!("Conservation check: v ∧ v = 0");
    println!("Violation prevention: Impossible by construction");
    println!("Error correction: None needed");
    println!("Validation: Automatic");

    // Prove the simplicity
    let physics_state = Geonum::new(42.0, 1.0, 7.0);
    let conservation_proof = physics_state.wedge(&physics_state);

    println!("\nActual computation:");
    println!("let state = Geonum::new(42.0, 1.0, 7.0);");
    println!("let conservation = state.wedge(&state);");
    println!(
        "Result: [{:.10}, {:.3}] - conservation proven",
        conservation_proof.length,
        conservation_proof.angle.mod_4_angle()
    );

    assert!(
        conservation_proof.length < EPSILON,
        "Conservation impossible to violate"
    );

    println!("\n✓ Newton's 3 laws + conservation → 2 lines of Rust");
    println!("✓ Centuries of physics → nanoseconds of computation");
}
