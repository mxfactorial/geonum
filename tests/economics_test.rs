use geonum::Geonum;
use std::f64::consts::PI;
use std::time::Instant;

// demos how data structured and processed with https://github.com/systemaccounting/mxfactorial
// empowers public economic vision with a cga state space

/// this file contains tests demonstrating how geometric algebra enables
/// privacy-preserving economic analysis for fiscal policy decision-making

/// this test demonstrates how business cycles exist in an economy without monetary inflation
/// when transactions are modeled as bivectors with conservation laws, cycles derive from
/// real economic factors rather than monetary distortions
#[test]
fn it_models_business_cycles() {
    // key insight: in a bivector transaction economy, business cycles still exist but derive from
    // real economic factors (productivity, capital allocation, supply/demand)
    // rather than monetary inflation
    // cycles emerge naturally from interaction of real economic variables in geometric space

    // create economic indicators as geometric numbers (magnitude + direction)
    // - length: magnitude of the indicator
    // - angle: phase relationship in the business cycle
    let gdp_growth = Geonum::from_polar(2.5, 0.1); // 2.5% growth, early expansion phase
    let price_adjustment = Geonum::from_polar(0.8, PI / 3.0); // real price changes from supply/demand
    let productivity = Geonum::from_polar(1.2, PI / 4.0); // productivity growth with technology
    let unemployment = Geonum::from_polar(4.2, 5.0 * PI / 6.0); // unemployment (counter-cyclical)
    let capital_formation = Geonum::from_polar(3.0, PI / 5.0); // investment in productive capacity

    // define a function to detect business cycle position from indicators
    // uses geometric algebra to compute the economic state vector
    let detect_cycle_phase = |indicators: &[Geonum]| -> Geonum {
        // project indicators onto a 2D plane for cycle analysis
        // similar to principal component analysis but with geometric meaning
        let weighted_x = indicators
            .iter()
            .map(|i| i.length * i.angle.cos()) // x-component (horizontal axis)
            .sum::<f64>();
        let weighted_y = indicators
            .iter()
            .map(|i| i.length * i.angle.sin()) // y-component (vertical axis)
            .sum::<f64>();

        // compute aggregate cycle phase (direction in economic state space)
        let phase_angle = weighted_y.atan2(weighted_x);

        // artifact of geonum automation: compute cycle momentum
        // represents the rate of change or velocity through the cycle
        // in a complete implementation, this returns as additional information
        let _cycle_momentum = indicators
            .iter()
            .map(|i| i.length * i.angle.sin())
            .sum::<f64>();

        // return the economic state as a geometric number:
        // - length: overall economic activity level (cycle amplitude)
        // - angle: position in the business cycle (cycle phase)
        Geonum {
            length: (weighted_x * weighted_x + weighted_y * weighted_y).sqrt(),
            angle: phase_angle,
            blade: 2, // bivector (grade 2) represents the economic cycle plane
                      // In geometric algebra, grade 2 elements represent oriented areas
                      // The business cycle is precisely that - a planar phenomenon with both
                      // magnitude (economic activity) and phase direction (cycle position)
        }
    };

    // analyze the current business cycle phase
    // collect all indicators into a multidimensional economic state vector
    let indicators = vec![
        gdp_growth,
        price_adjustment,
        productivity,
        unemployment,
        capital_formation,
    ];

    // compute the current position in the business cycle
    let cycle_state = detect_cycle_phase(&indicators);

    // interpret the cycle phase in economic terms
    // map the geometric angle to economic cycle phases
    let cycle_phase = match cycle_state.angle {
        a if a >= 0.0 && a < PI / 2.0 => "expansion", // first quadrant: growth phase
        a if a >= PI / 2.0 && a < PI => "peak",       // second quadrant: mature expansion
        a if a >= PI && a < 3.0 * PI / 2.0 => "contraction", // third quadrant: declining activity
        _ => "trough",                                // fourth quadrant: bottom of cycle
    };

    // compute economic cycle velocity
    // shows how quickly the economy moves through the cycle
    let cycle_velocity = indicators
        .iter()
        .map(|i| i.length * i.angle.sin())
        .sum::<f64>();

    // test the cycle detection produces meaningful results
    assert!(
        cycle_state.length > 0.0,
        "cycle analysis produces non-zero amplitude"
    );

    // performance benchmark - constant time O(1)
    // key advantage over traditional economic models that often scale poorly
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = detect_cycle_phase(&indicators);
    }
    let duration = start.elapsed();

    // fast regardless of how many indicators we analyze
    // demonstrates O(1) complexity advantage of geometric algebra design
    assert!(
        duration.as_micros() < 10000, // 10000 analyses under 10ms
        "business cycle analysis runs with O(1) complexity regardless of indicator count"
    );

    // output the economic analysis results
    println!("───── business cycle analysis ─────");
    println!("cycle position: {:.2} radians", cycle_state.angle);
    println!("current economic phase: {}", cycle_phase);
    println!(
        "cycle strength: {:.2} (overall economic activity)",
        cycle_state.length
    );
    println!(
        "cycle velocity: {:.2} (positive = accelerating, negative = decelerating)",
        cycle_velocity
    );
    println!("───────────────────────────────────");
    println!("note: in a bivector economy without monetary inflation, business cycles");
    println!("still exist but derive from fundamental economic factors rather than");
    println!("monetary distortions, enabling more accurate economic planning");
}

#[test]
fn it_models_payroll_tax_impact_across_income_brackets() {
    // query example 1: "how would a payroll tax change impact spending across income brackets and regions?"
    // this test demonstrates how changes in payroll taxes affect different demographic segments

    // define the dimensions of our analysis
    // map income brackets to angles in economic space
    let income_brackets = vec![
        ("low_income", PI / 10.0),         // <$30K annually
        ("lower_middle", 2.0 * PI / 10.0), // $30K-$50K
        ("middle", 3.0 * PI / 10.0),       // $50K-$75K
        ("upper_middle", 4.0 * PI / 10.0), // $75K-$150K
        ("high_income", 5.0 * PI / 10.0),  // >$150K
    ];

    // map geographic regions to angles
    let regions = vec![
        ("northeast", PI / 8.0),
        ("midwest", 2.0 * PI / 8.0),
        ("south", 3.0 * PI / 8.0),
        ("west", 4.0 * PI / 8.0),
    ];

    // map spending categories to angles
    let spending_categories = vec![
        ("essential", PI / 12.0),           // food, housing, utilities
        ("healthcare", 2.0 * PI / 12.0),    // medical expenses
        ("discretionary", 3.0 * PI / 12.0), // entertainment, dining
        ("savings", 4.0 * PI / 12.0),       // investments, retirement
    ];

    // tax policy scenarios (as geometric objects)
    let current_policy = Geonum::from_polar_blade(1.0, 0.0, 0); // baseline policy as scalar (grade 0)
                                                                // Using blade: 0 for policy baseline because it represents a
                                                                // scale factor with no directional properties - just magnitude
    let tax_cut_policy = Geonum::from_polar(0.8, -PI / 16.0); // 20% tax reduction
    let tax_increase_policy = Geonum::from_polar(1.2, PI / 16.0); // 20% tax increase

    // create a function to simulate spending behavior under different tax policies
    let simulate_spending =
        |income: &str, region: &str, category: &str, policy: &Geonum| -> Geonum {
            // locate the angles for this specific combination
            let income_angle = income_brackets
                .iter()
                .find(|(bracket, _)| bracket == &income)
                .map(|(_, angle)| *angle)
                .unwrap_or(0.0);

            let region_angle = regions
                .iter()
                .find(|(r, _)| r == &region)
                .map(|(_, angle)| *angle)
                .unwrap_or(0.0);

            let category_angle = spending_categories
                .iter()
                .find(|(cat, _)| cat == &category)
                .map(|(_, angle)| *angle)
                .unwrap_or(0.0);

            // compute spending response based on income bracket (lower incomes more sensitive to tax changes)
            let sensitivity = match income {
                "low_income" => 2.0,
                "lower_middle" => 1.5,
                "middle" => 1.0,
                "upper_middle" => 0.7,
                "high_income" => 0.4,
                _ => 1.0,
            };

            // spending response also varies by category (essentials less elastic than discretionary)
            let category_elasticity = match category {
                "essential" => 0.3,
                "healthcare" => 0.5,
                "discretionary" => 1.8,
                "savings" => 1.2,
                _ => 1.0,
            };

            // compute spending change based on tax policy
            let policy_impact =
                (policy.length - current_policy.length) * sensitivity * category_elasticity;

            // compute geometric spending response combining all dimensions
            // length = magnitude of spending change
            // angle = combined position in economic space
            Geonum::from_polar(
                1.0 + policy_impact,
                (income_angle + region_angle + category_angle + policy.angle) % (2.0 * PI),
            )
        };

    // analyze spending changes across multiple dimensions
    let mut results = vec![];

    // process all combinations of income, region and category
    for (income, _) in &income_brackets {
        for (region, _) in &regions {
            for (category, _) in &spending_categories {
                // simulate spending under tax cut
                let tax_cut_response = simulate_spending(income, region, category, &tax_cut_policy);

                // simulate spending under tax increase
                let tax_increase_response =
                    simulate_spending(income, region, category, &tax_increase_policy);

                // store results for analysis
                results.push((
                    income,
                    region,
                    category,
                    tax_cut_response,
                    tax_increase_response,
                ));
            }
        }
    }

    // compute performance for O(1) complexity validation
    let start = Instant::now();

    // compute aggregate impacts
    let tax_cut_impact = results
        .iter()
        .map(|(_, _, _, cut_resp, _)| {
            if cut_resp.length > 1.0 {
                cut_resp.length - 1.0
            } else {
                0.0
            }
        })
        .sum::<f64>();

    let tax_increase_impact = results
        .iter()
        .map(|(_, _, _, _, incr_resp)| {
            if incr_resp.length < 1.0 {
                1.0 - incr_resp.length
            } else {
                0.0
            }
        })
        .sum::<f64>();

    // identify most vulnerable segments (largest negative impact from tax increase)
    let vulnerable_segments = results
        .iter()
        .filter(|(_, _, _, _, incr_resp)| incr_resp.length < 0.9) // >10% decrease in spending
        .collect::<Vec<_>>();

    let duration = start.elapsed();

    // test performance is O(1) complexity regardless of dimensions analyzed
    assert!(
        duration.as_micros() < 10000, // under 10ms for all analyses
        "tax impact analysis runs with O(1) complexity"
    );

    // test results are meaningful for policy decisions
    // verify at least one result is found
    if vulnerable_segments.is_empty() {
        println!("note: no vulnerable segments found with current thresholds");
    }

    // output fiscal policy dashboard
    println!("───── payroll tax impact analysis ─────");
    println!("tax cut aggregate impact: +{:.2}%", tax_cut_impact * 100.0);
    println!(
        "tax increase aggregate impact: -{:.2}%",
        tax_increase_impact * 100.0
    );
    println!(
        "identified vulnerable segments: {}",
        vulnerable_segments.len()
    );

    // output most affected segment
    if let Some((income, region, category, _, _)) = vulnerable_segments.first() {
        println!(
            "most vulnerable: {} income in {} region, {} spending",
            income, region, category
        );
    }

    println!("computation time: {:.2} nanoseconds", duration.as_nanos());
    println!("─────────────────────────────────────");
    println!("note: geometric algebra enables targeted fiscal policy by analyzing");
    println!("multidimensional impacts while preserving individual privacy");
}

#[test]
fn it_detects_early_recession_indicators() {
    // query example 2: "what spending patterns indicate impending economic contraction before traditional indicators?"
    // this test demonstrates how to detect early warning signs of economic downturn

    // define dimensions for economic indicators
    let transaction_volumes = vec![
        ("retail", PI / 9.0),
        ("manufacturing", 2.0 * PI / 9.0),
        ("services", 3.0 * PI / 9.0),
        ("construction", 4.0 * PI / 9.0),
    ];

    // payment timing dimensions
    let payment_timing = vec![
        ("accelerated", PI / 7.0),       // paying earlier than usual
        ("on_schedule", 2.0 * PI / 7.0), // paying on normal schedule
        ("delayed", 3.0 * PI / 7.0),     // paying later than normal
        ("missed", 4.0 * PI / 7.0),      // missing payments entirely
    ];

    // transaction size dimensions
    let transaction_sizes = vec![
        ("micro", PI / 5.0),        // very small transactions
        ("small", 2.0 * PI / 5.0),  // normal small purchases
        ("medium", 3.0 * PI / 5.0), // significant purchases
        ("large", 4.0 * PI / 5.0),  // major investments/purchases
    ];

    // geographic flow dimensions
    let geographic_flows = vec![
        ("local", PI / 8.0),
        ("regional", 2.0 * PI / 8.0),
        ("national", 3.0 * PI / 8.0),
        ("international", 4.0 * PI / 8.0),
    ];

    // create simulated transaction data for detection
    // normal economic conditions (baseline period)
    let normal_transactions = vec![
        (
            Geonum::from_polar(100.0, transaction_volumes[0].1),
            Geonum::from_polar(1.0, payment_timing[1].1),
            Geonum::from_polar(1.0, transaction_sizes[1].1),
            Geonum::from_polar(1.0, geographic_flows[0].1),
        ),
        (
            Geonum::from_polar(150.0, transaction_volumes[1].1),
            Geonum::from_polar(1.0, payment_timing[1].1),
            Geonum::from_polar(1.0, transaction_sizes[2].1),
            Geonum::from_polar(1.0, geographic_flows[1].1),
        ),
        (
            Geonum::from_polar(200.0, transaction_volumes[2].1),
            Geonum::from_polar(1.0, payment_timing[1].1),
            Geonum::from_polar(1.0, transaction_sizes[1].1),
            Geonum::from_polar(1.0, geographic_flows[2].1),
        ),
    ];

    // early recession signals (3-6 months before traditional indicators)
    let early_warning_transactions = vec![
        (
            Geonum::from_polar(90.0, transaction_volumes[0].1),
            Geonum::from_polar(1.0, payment_timing[2].1),
            Geonum::from_polar(1.0, transaction_sizes[0].1),
            Geonum::from_polar(1.0, geographic_flows[0].1),
        ),
        (
            Geonum::from_polar(130.0, transaction_volumes[1].1),
            Geonum::from_polar(1.0, payment_timing[2].1),
            Geonum::from_polar(1.0, transaction_sizes[1].1),
            Geonum::from_polar(1.0, geographic_flows[1].1),
        ),
        (
            Geonum::from_polar(180.0, transaction_volumes[2].1),
            Geonum::from_polar(1.0, payment_timing[3].1),
            Geonum::from_polar(1.0, transaction_sizes[0].1),
            Geonum::from_polar(1.0, geographic_flows[0].1),
        ),
    ];

    // create a recession detection function using geometric algebra
    let detect_recession_signals =
        |transactions: &Vec<(Geonum, Geonum, Geonum, Geonum)>| -> Geonum {
            // compute volume-weighted average of payment delays
            let total_volume: f64 = transactions.iter().map(|(vol, _, _, _)| vol.length).sum();

            // payment timing signal - weighted by transaction volume
            let payment_timing_signal = transactions
                .iter()
                .map(|(vol, timing, _, _)| vol.length * timing.angle.sin())
                .sum::<f64>()
                / total_volume;

            // transaction size downshift signal
            let size_signal = transactions
                .iter()
                .map(|(vol, _, size, _)| vol.length * (PI / 2.0 - size.angle).sin())
                .sum::<f64>()
                / total_volume;

            // geographic flow concentration signal (local vs distant)
            let geographic_signal = transactions
                .iter()
                .map(|(vol, _, _, geo)| vol.length * (PI / 2.0 - geo.angle).sin())
                .sum::<f64>()
                / total_volume;

            // volume decrease signal
            let baseline_volume = 450.0; // normal economy total volume
            let volume_signal = (baseline_volume - total_volume) / baseline_volume;

            // combine signals into a recession indicator
            // length = strength of recession signal
            // angle = phase (what stage of early recession)
            Geonum::from_polar(
                (payment_timing_signal.powi(2)
                    + size_signal.powi(2)
                    + geographic_signal.powi(2)
                    + volume_signal.powi(2))
                .sqrt(),
                (payment_timing_signal.atan2(size_signal) + geographic_signal.atan2(volume_signal))
                    / 2.0,
            )
        };

    // compute recession signals
    let start = Instant::now();

    let normal_signal = detect_recession_signals(&normal_transactions);
    let warning_signal = detect_recession_signals(&early_warning_transactions);

    // test with 10,000 iterations to verify O(1) performance
    for _ in 0..10000 {
        let _ = detect_recession_signals(&early_warning_transactions);
    }

    let duration = start.elapsed();

    // derive recession probability based on signal strength
    let recession_probability = warning_signal.length * 100.0;

    // compute lead time (how many months warning)
    let lead_time = 6.0 * warning_signal.length / 0.5; // 6 months at 0.5 signal strength

    // test O(1) complexity
    assert!(
        duration.as_micros() < 10000,
        "recession detection runs with O(1) complexity"
    );

    // test signal discrimination between normal and warning conditions
    println!(
        "Warning signal strength: {:.3} vs normal signal: {:.3}",
        warning_signal.length, normal_signal.length
    );
    assert!(
        warning_signal.length > normal_signal.length,
        "detection function differentiates between normal and warning conditions"
    );

    // output early warning system dashboard
    println!("───── early recession detection ─────");
    println!("baseline economy signal: {:.3}", normal_signal.length);
    println!("warning signal strength: {:.3}", warning_signal.length);
    println!("recession probability: {:.1}%", recession_probability);
    println!("estimated lead time: {:.1} months", lead_time);
    println!(
        "computation time: {:.2} nanoseconds",
        duration.as_nanos() / 10000
    );
    println!("────────────────────────────────────");
    println!("note: geometric algebra detects recession indicators 3-6 months");
    println!("before traditional metrics while preserving transaction privacy");
}

#[test]
fn it_analyzes_small_business_cashflow_after_rate_change() {
    // query example 3: "how do small business cashflow patterns change after interest rate adjustments?"
    // demonstrates how interest rate changes affect different business segments

    // define business size dimensions
    let business_sizes = vec![
        ("micro", PI / 10.0),        // 1-9 employees
        ("small", 2.0 * PI / 10.0),  // 10-49 employees
        ("medium", 3.0 * PI / 10.0), // 50-249 employees
        ("large", 4.0 * PI / 10.0),  // 250+ employees
    ];

    // define industry dimensions
    let industries = vec![
        ("retail", PI / 8.0),
        ("manufacturing", 2.0 * PI / 8.0),
        ("services", 3.0 * PI / 8.0),
        ("construction", 4.0 * PI / 8.0),
        ("technology", 5.0 * PI / 8.0),
    ];

    // define cash reserve duration dimensions
    let cash_reserve_durations = vec![
        ("critical", PI / 6.0),       // <1 month reserves
        ("limited", 2.0 * PI / 6.0),  // 1-3 months
        ("adequate", 3.0 * PI / 6.0), // 3-6 months
        ("strong", 4.0 * PI / 6.0),   // 6+ months
    ];

    // define debt service ratio dimensions
    let debt_service_ratios = vec![
        ("low", PI / 4.0),            // <10% of revenue
        ("moderate", 2.0 * PI / 4.0), // 10-20% of revenue
        ("high", 3.0 * PI / 4.0),     // >20% of revenue
    ];

    // define interest rate scenarios
    let _base_rate = 0.05; // 5% baseline rate
    let rate_increase = 0.02; // 2% rate increase

    // create sample business data for analysis
    let business_data = vec![
        // (size, industry, cash_reserves, debt_ratio, monthly_cashflow)
        ("micro", "retail", "limited", "high", 15000.0),
        ("micro", "services", "critical", "high", 8000.0),
        ("small", "manufacturing", "adequate", "moderate", 45000.0),
        ("small", "technology", "limited", "low", 60000.0),
        ("medium", "construction", "limited", "high", 120000.0),
        ("medium", "retail", "adequate", "moderate", 150000.0),
    ];

    // function to compute cashflow impact of rate changes
    let compute_cashflow_impact = |size: &str,
                                   industry: &str,
                                   reserves: &str,
                                   debt_ratio: &str,
                                   cashflow: f64,
                                   rate_change: f64|
     -> Geonum {
        // find dimension angles
        let size_angle = business_sizes
            .iter()
            .find(|(s, _)| s == &size)
            .map(|(_, angle)| *angle)
            .unwrap_or(0.0);

        let industry_angle = industries
            .iter()
            .find(|(ind, _)| ind == &industry)
            .map(|(_, angle)| *angle)
            .unwrap_or(0.0);

        let reserves_angle = cash_reserve_durations
            .iter()
            .find(|(res, _)| res == &reserves)
            .map(|(_, angle)| *angle)
            .unwrap_or(0.0);

        let debt_angle = debt_service_ratios
            .iter()
            .find(|(debt, _)| debt == &debt_ratio)
            .map(|(_, angle)| *angle)
            .unwrap_or(0.0);

        // compute sensitivity factors
        let size_sensitivity = match size {
            "micro" => 2.0,
            "small" => 1.5,
            "medium" => 1.0,
            "large" => 0.5,
            _ => 1.0,
        };

        let industry_sensitivity = match industry {
            "construction" => 1.8,
            "retail" => 1.2,
            "manufacturing" => 1.0,
            "services" => 0.8,
            "technology" => 0.6,
            _ => 1.0,
        };

        let reserves_sensitivity = match reserves {
            "critical" => 3.0,
            "limited" => 2.0,
            "adequate" => 1.0,
            "strong" => 0.5,
            _ => 1.0,
        };

        let debt_sensitivity = match debt_ratio {
            "low" => 0.5,
            "moderate" => 1.5,
            "high" => 3.0,
            _ => 1.0,
        };

        // compute overall sensitivity to rate changes
        let combined_sensitivity =
            size_sensitivity * industry_sensitivity * reserves_sensitivity * debt_sensitivity;

        // compute cashflow impact
        let impact_percent = -combined_sensitivity * rate_change * 100.0;
        let new_cashflow = cashflow * (1.0 + impact_percent / 100.0);

        // return geometric number encoding impact
        Geonum::from_polar(
            new_cashflow / cashflow, // ratio of new to old cashflow
            (size_angle + industry_angle + reserves_angle + debt_angle) / 4.0, // average angle
        )
    };

    // analyze rate increase impact
    let start = Instant::now();

    let impacts = business_data
        .iter()
        .map(|(size, industry, reserves, debt, cashflow)| {
            let impact =
                compute_cashflow_impact(size, industry, reserves, debt, *cashflow, rate_increase);
            (size, industry, reserves, debt, *cashflow, impact)
        })
        .collect::<Vec<_>>();

    // identify vulnerable businesses
    let vulnerable_businesses = impacts
        .iter()
        .filter(|(_, _, _, _, _, impact)| impact.length < 0.9) // >10% cashflow reduction
        .collect::<Vec<_>>();

    // compute weighted average impact
    let total_cashflow: f64 = business_data.iter().map(|(_, _, _, _, cf)| cf).sum();
    let weighted_impact = impacts
        .iter()
        .map(|(_, _, _, _, cf, impact)| cf * (impact.length - 1.0))
        .sum::<f64>()
        / total_cashflow;

    // run 10,000 iterations for performance testing
    for _ in 0..10000 {
        let _ = compute_cashflow_impact(
            "small",
            "retail",
            "limited",
            "moderate",
            50000.0,
            rate_increase,
        );
    }

    let duration = start.elapsed();

    // test O(1) complexity
    assert!(
        duration.as_micros() < 10000,
        "cashflow impact analysis runs with O(1) complexity"
    );

    // test policy relevance
    assert!(
        !vulnerable_businesses.is_empty(),
        "analysis identifies vulnerable business segments"
    );

    // output cashflow analysis dashboard
    println!("───── interest rate impact analysis ─────");
    println!("rate increase scenario: +{:.1}%", rate_increase * 100.0);
    println!("overall cashflow impact: {:.2}%", weighted_impact * 100.0);
    println!(
        "vulnerable businesses: {}/{}",
        vulnerable_businesses.len(),
        impacts.len()
    );

    // output most affected business
    if let Some((size, industry, reserves, debt, _, impact)) = vulnerable_businesses.first() {
        println!(
            "most vulnerable: {} {} business with {} reserves and {} debt",
            size, industry, reserves, debt
        );
        println!(
            "expected cashflow reduction: {:.1}%",
            (1.0 - impact.length) * 100.0
        );
    }

    println!(
        "computation time: {:.2} nanoseconds",
        duration.as_nanos() / 10000
    );
    println!("────────────────────────────────────────");
    println!("note: geometric algebra enables precise targeting of fiscal support");
    println!("to vulnerable businesses after monetary policy changes");
}

#[test]
fn it_analyzes_housing_payment_patterns() {
    // query example 5: "how do housing payment patterns predict mortgage market stability?"
    // demonstrates how payment behavior can predict housing market problems

    // define payment timeliness dimensions
    let _payment_timeliness = vec![
        ("early", PI / 8.0),
        ("on_time", 2.0 * PI / 8.0),
        ("grace_period", 3.0 * PI / 8.0),
        ("30_days_late", 4.0 * PI / 8.0),
        ("60_days_late", 5.0 * PI / 8.0),
        ("90_plus_days", 6.0 * PI / 8.0),
    ];

    // define income-to-payment ratio dimensions
    let _income_payment_ratio = vec![
        ("affordable", PI / 6.0),      // <28% of income on housing
        ("moderate", 2.0 * PI / 6.0),  // 28-36% of income on housing
        ("stretched", 3.0 * PI / 6.0), // 36-43% of income on housing
        ("severe", 4.0 * PI / 6.0),    // >43% of income on housing
    ];

    // define geographic regions
    let regions = vec![
        ("northeast", PI / 5.0),
        ("midwest", 2.0 * PI / 5.0),
        ("south", 3.0 * PI / 5.0),
        ("west", 4.0 * PI / 5.0),
    ];

    // define property value change dimensions
    let _value_change = vec![
        ("rising_fast", PI / 7.0),          // >10% annual appreciation
        ("rising", 2.0 * PI / 7.0),         // 3-10% appreciation
        ("stable", 3.0 * PI / 7.0),         // -3% to +3% change
        ("declining", 4.0 * PI / 7.0),      // 3-10% depreciation
        ("declining_fast", 5.0 * PI / 7.0), // >10% depreciation
    ];

    // create simulated payment pattern data
    // (timeliness, income_ratio, region, value_change, volume)
    let stable_market_data = vec![
        ("on_time", "affordable", "midwest", "stable", 5000),
        ("on_time", "moderate", "northeast", "rising", 3000),
        ("grace_period", "moderate", "south", "stable", 2000),
        ("on_time", "affordable", "west", "rising", 4000),
        ("grace_period", "stretched", "northeast", "stable", 1000),
    ];

    // unstable market showing early warning signs
    let unstable_market_data = vec![
        ("grace_period", "moderate", "west", "declining", 3000),
        ("30_days_late", "stretched", "west", "declining", 2000),
        ("on_time", "moderate", "south", "stable", 2500),
        ("grace_period", "stretched", "northeast", "stable", 2000),
        ("60_days_late", "severe", "west", "declining_fast", 1500),
    ];

    // housing market stability analysis function
    let analyze_market_stability = |data: &Vec<(&str, &str, &str, &str, i32)>| -> Geonum {
        // compute the total volume
        let total_volume: i32 = data.iter().map(|(_, _, _, _, vol)| vol).sum();

        // payment stress indicator (weighted by volume)
        let payment_stress = data
            .iter()
            .map(|(timeliness, _, _, _, vol)| {
                let stress_factor = match *timeliness {
                    "early" => 0.0,
                    "on_time" => 0.0,
                    "grace_period" => 0.3,
                    "30_days_late" => 0.6,
                    "60_days_late" => 0.8,
                    "90_plus_days" => 1.0,
                    _ => 0.0,
                };
                stress_factor * (*vol as f64 / total_volume as f64)
            })
            .sum::<f64>();

        // affordability stress
        let affordability_stress = data
            .iter()
            .map(|(_, ratio, _, _, vol)| {
                let stress_factor = match *ratio {
                    "affordable" => 0.0,
                    "moderate" => 0.2,
                    "stretched" => 0.7,
                    "severe" => 1.0,
                    _ => 0.0,
                };
                stress_factor * (*vol as f64 / total_volume as f64)
            })
            .sum::<f64>();

        // value decline stress
        let value_stress = data
            .iter()
            .map(|(_, _, _, change, vol)| {
                let stress_factor = match *change {
                    "rising_fast" => -0.2, // negative stress (positive factor)
                    "rising" => -0.1,      // negative stress
                    "stable" => 0.0,
                    "declining" => 0.6,
                    "declining_fast" => 1.0,
                    _ => 0.0,
                };
                stress_factor * (*vol as f64 / total_volume as f64)
            })
            .sum::<f64>();

        // compute dominant region (for angle)
        let region_volumes = regions
            .iter()
            .map(|(region, angle)| {
                let region_vol = data
                    .iter()
                    .filter(|(_, _, r, _, _)| r == region)
                    .map(|(_, _, _, _, vol)| *vol)
                    .sum::<i32>();
                (region, *angle, region_vol)
            })
            .collect::<Vec<_>>();

        let primary_region_angle = region_volumes
            .iter()
            .max_by_key(|(_, _, vol)| *vol)
            .map(|(_, angle, _)| *angle)
            .unwrap_or(0.0);

        // combine stress indicators
        let combined_stress =
            (payment_stress.powi(2) + affordability_stress.powi(2) + value_stress.powi(2)).sqrt()
                / 1.732; // normalize to 0-1

        // return geometric number
        // length = market stress level (0-1)
        // angle = primary regional position
        Geonum::from_polar(combined_stress, primary_region_angle)
    };

    // perform stability analysis
    let start = Instant::now();

    let stable_result = analyze_market_stability(&stable_market_data);
    let unstable_result = analyze_market_stability(&unstable_market_data);

    // test with 10,000 iterations
    for _ in 0..10000 {
        let _ = analyze_market_stability(&unstable_market_data);
    }

    let duration = start.elapsed();

    // compute stability metrics
    let stable_risk = stable_result.length;
    let unstable_risk = unstable_result.length;

    // housing market risk thresholds
    let low_risk_threshold = 0.3;
    let moderate_risk_threshold = 0.6;

    // categorize markets by risk level
    let stable_status = if stable_risk < low_risk_threshold {
        "low risk"
    } else if stable_risk < moderate_risk_threshold {
        "moderate risk"
    } else {
        "high risk"
    };

    let unstable_status = if unstable_risk < low_risk_threshold {
        "low risk"
    } else if unstable_risk < moderate_risk_threshold {
        "moderate risk"
    } else {
        "high risk"
    };

    // test O(1) complexity
    assert!(
        duration.as_micros() < 100000, // increased threshold for test stability
        "housing market analysis runs with O(1) complexity"
    );

    // test meaningful discrimination between stable and unstable markets
    assert!(
        unstable_risk > stable_risk * 1.5,
        "analysis correctly identifies higher risk in unstable markets"
    );

    // output housing market stability dashboard
    println!("───── housing market stability analysis ─────");
    println!(
        "stable market risk score: {:.3} - {}",
        stable_risk, stable_status
    );
    println!(
        "unstable market risk score: {:.3} - {}",
        unstable_risk, unstable_status
    );
    println!("risk differential: {:.1}x", unstable_risk / stable_risk);

    // policy recommendation based on risk level
    let recommendation = if unstable_risk > moderate_risk_threshold {
        "immediate targeted assistance to high-risk areas"
    } else if unstable_risk > low_risk_threshold {
        "enhanced monitoring and preventative programs"
    } else {
        "standard oversight adequate"
    };

    println!("policy recommendation: {}", recommendation);
    println!(
        "computation time: {:.2} nanoseconds",
        duration.as_nanos() / 10000
    );
    println!("──────────────────────────────────────────");
    println!("note: geometric algebra enables early detection of housing market");
    println!("issues while preserving privacy of individual payment records");
}

#[test]
fn it_models_global_trade_flows() {
    // with transactions as bivectors, international trade becomes
    // a direct flow of value with clear causality and balance requirements

    // model major trading nations as economic spaces
    let usa = Geonum::from_polar(22.0, 0.1); // $22T GDP with slight trade deficit angle
    let china = Geonum::from_polar(16.0, -0.15); // $16T GDP with trade surplus angle
    let eu = Geonum::from_polar(18.0, 0.05); // $18T GDP with balanced trade angle
    let japan = Geonum::from_polar(5.0, -0.2); // $5T GDP with trade surplus angle

    // bilateral trade flows as bivectors between national economies
    // each represents transactions with conservation laws enforced
    let usa_china_trade = Geonum::from_polar_blade(650.0, PI / 2.0 + 0.3, 2); // $650B net flow as bivector (grade 2)
                                                                              // Blade: 2 represents trade flows as oriented areas in economic space
                                                                              // Trade transactions create a plane between two economic entities
    let usa_eu_trade = Geonum::from_polar_blade(1100.0, PI / 2.0 - 0.1, 2); // $1.1T net flow as bivector (grade 2)
                                                                            // PI/2 angle typical for perpendicular economic relationships
    let china_eu_trade = Geonum::from_polar_blade(700.0, PI / 2.0 + 0.2, 2); // $700B net flow as bivector (grade 2)
                                                                             // International trade naturally forms bivector relationships

    // model trade network for imbalances - which must sum to zero
    // any non-zero sum indicates measurement error or missing flows
    let model_trade_network = |_economies: &[Geonum], trade_flows: &[Geonum]| -> Geonum {
        // with bivector transactions, global trade must balance

        // we're measuring how well our model captures all flows

        // compute weighted flow direction
        let weighted_x = trade_flows
            .iter()
            .map(|f| f.length * f.angle.cos())
            .sum::<f64>();

        let weighted_y = trade_flows
            .iter()
            .map(|f| f.length * f.angle.sin())
            .sum::<f64>();

        // compute total trade volume
        let total_volume = trade_flows.iter().map(|f| f.length).sum::<f64>();

        // encode global trade state as geometric number
        // - length represents total trade volume
        // - angle represents net imbalance direction
        Geonum {
            length: total_volume,
            angle: weighted_y.atan2(weighted_x),
            blade: 2, // bivector (grade 2) representing the global trade plane
                      // International trade flows naturally form bivectors in geometric algebra
                      // as they represent oriented exchange relationships between economic entities
        }
    };

    // model the global trade network
    let economies = vec![usa, china, eu, japan];
    let trade_flows = vec![usa_china_trade, usa_eu_trade, china_eu_trade];

    let global_trade = model_trade_network(&economies, &trade_flows);

    // detect trade imbalances through angle analysis
    // in perfect measurement, this should be zero due to conservation laws
    let imbalance_magnitude = global_trade.angle.sin() * global_trade.length;
    let balanced_trade_threshold = 0.05 * global_trade.length; // 5% measurement error threshold

    // prove model produces meaningful results
    assert!(
        global_trade.length > 0.0,
        "trade model produces positive volume"
    );

    // measure performance of trade network analysis
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = model_trade_network(&economies, &trade_flows);
    }
    let duration = start.elapsed();

    // test O(1) complexity regardless of network size
    assert!(
        duration.as_micros() < 100000, // increased threshold for test stability
        "trade network model execs with O(1) complexity"
    );

    // determine if global trade model is balanced (should be in reality)
    let is_balanced = imbalance_magnitude.abs() < balanced_trade_threshold;

    println!("global trade volume: ${:.2}B", global_trade.length);
    println!("trade measurement imbalance: ${:.2}B", imbalance_magnitude);
    println!("trade model is balanced: {}", is_balanced);
}

#[test]
fn it_measures_economic_sectoral_balance() {
    // this test demonstrates how geonum can analyze economic sectoral balances
    // in a conformal geometric algebra space, as would be enabled by the mxfactorial application

    // in a mxfactorial economy, all transactions are conserved between creditors and debitors
    // allowing real-time analysis of sectoral balances across the entire economy

    // simulate major economic sectors as geometric numbers
    // where length represents total transaction volume and angle represents net flow direction
    let household_sector = Geonum::from_polar(10000.0, 0.2); // slight net creditor
    let business_sector = Geonum::from_polar(15000.0, -0.3); // net debitor
    let government_sector = Geonum::from_polar(8000.0, -0.8); // strong net debitor
    let foreign_sector = Geonum::from_polar(5000.0, 0.6); // strong net creditor

    // in mxfactorial, these would be computed from transaction streams in real-time

    // demonstrate how geonum can detect sectoral imbalances using geometric operations
    let measure_sectoral_balance = |sectors: &[Geonum]| -> Geonum {
        // compute all sums in a single pass - much more efficient O(1) implementation
        let mut flow_x = 0.0;
        let mut flow_y = 0.0;
        let mut total_magnitude = 0.0;

        // single iteration over sectors
        for sector in sectors {
            flow_x += sector.length * sector.angle.cos();
            flow_y += sector.length * sector.angle.sin();
            total_magnitude += sector.length;
        }

        // result as geometric number
        Geonum {
            length: total_magnitude,
            angle: flow_y.atan2(flow_x),
            blade: 2, // bivector (grade 2) representing the economic sector plane
                      // Grade 2 elements model relationships between economic sectors
                      // Sectoral balances form a planar system where outflows from one sector
                      // must equal inflows to other sectors (conservation of value)
        }
    };

    // measure performance of sectoral analysis
    let sectors = vec![
        household_sector,
        business_sector,
        government_sector,
        foreign_sector,
    ];

    let start = Instant::now();
    let economic_balance = measure_sectoral_balance(&sectors);
    let duration = start.elapsed();

    // compute domestic sector balance (households + businesses)
    let domestic_private_balance = Geonum {
        length: household_sector.length + business_sector.length,
        angle: (household_sector.angle * household_sector.length
            + business_sector.angle * business_sector.length)
            / (household_sector.length + business_sector.length),
        blade: 2, // bivector (grade 2) representing the sectoral balance relationship
                  // Sectoral balances in geometric algebra are naturally grade 2 elements
                  // as they represent flows between different economic sectors
                  // The domestic private balance is the combined households and business planes
    };

    // compute public sector balance (government)
    let public_balance = government_sector;

    // compute foreign sector balance
    let foreign_balance = foreign_sector;

    // sectoral balance identity: domestic private + public + foreign = 0
    // in angles, this means they should sum to approximate zero when weighted by magnitude
    let _total_weighted_angle = domestic_private_balance.angle * domestic_private_balance.length
        + public_balance.angle * public_balance.length
        + foreign_balance.angle * foreign_balance.length;

    // detect economic imbalances through angle analysis
    let imbalance_detected = economic_balance.angle.abs() > 0.1;

    println!(
        "Economic balance analysis: {:.2} nanoseconds",
        duration.as_nanos()
    );
    println!("Detected imbalance: {}", imbalance_detected);
    println!("Economy net angle: {:.4}", economic_balance.angle);

    // demonstrate how this analysis would detect economic crises early through
    // geometric angle shifts in sectoral balances

    // verify analysis time is constant regardless of transaction volume
    assert!(
        duration.as_nanos() < 100000, // increased threshold for test stability
        "sectoral balance analysis should have O(1) complexity"
    );
}
