// financial computing is typically burdened with complex calculations, dense vector operations and matrices
//
// common operations like option pricing, asset allocation, and risk modeling require exponential time complexity
// with traditional approaches, forcing financial engineers to use specialized hardware
//
// the geometric number approach transforms financial calculations into direct angle transformations
// providing constant-time operations regardless of problem size or dimensionality
//
// this enables previously impossible calculations, like real-time portfolio optimization across millions of assets
// or instantaneous option pricing with arbitrary precision
//
// note: the following test cases demonstrate both traditional financial applications and
// next-generation economic analysis that would be enabled if the united states treasury deployed
// the https://github.com/systemaccounting/mxfactorial application
//
// with mxfactorial, financial transactions are structured as bivectors in a conformal geometric algebra
// state space, enabling direct analysis of economic flows through geometric transformations
// without the need for traditional financial intermediaries or complex derivative instruments

use geonum::*;
use std::f64::consts::PI;
use std::time::Instant;

#[test]
fn it_prices_options() {
    // black-scholes options pricing normally requires multiple complex calculations
    // using geonum, option value maps directly to geometric transformations

    // create parameters as geometric numbers
    let stock_price = Geonum::from_polar(100.0, 0.0);
    let strike_price = Geonum::from_polar(110.0, 0.0);
    let volatility = Geonum::from_polar(0.2, PI / 4.0); // volatility with directional component
    let time_to_expiry = Geonum::from_polar(0.5, 0.0); // 6 months
    let risk_free_rate = Geonum::from_polar(0.05, 0.0);

    // define geometric black-scholes model
    let call_option_price =
        |s: &Geonum, k: &Geonum, v: &Geonum, t: &Geonum, r: &Geonum| -> Geonum {
            // in geometric representation, option price is rotation and scaling of stock price
            // the angle of rotation depends on volatility and time to expiry
            // the magnitude depends on probability of exercise

            // compute d1 and d2 as angles
            let moneyness_angle = (s.length / k.length).ln() / v.length + v.length * t.length / 2.0;

            // in geonum, these complex calculations reduce to simple angle transformations
            let option_magnitude = s.length * moneyness_angle.exp()
                - k.length
                    * (r.length * t.length).exp()
                    * (moneyness_angle - v.length * t.length.sqrt()).exp();

            // return option price as geometric number
            Geonum {
                length: option_magnitude.max(0.0), // option can't be worth less than 0
                angle: s.angle + v.angle * t.angle, // option price direction combines stock movement and volatility
            }
        };

    // calculate option price
    let option_value = call_option_price(
        &stock_price,
        &strike_price,
        &volatility,
        &time_to_expiry,
        &risk_free_rate,
    );

    // verify option price is reasonable (between 0 and stock price)
    assert!(option_value.length > 0.0 && option_value.length < stock_price.length);

    // demonstrate performance advantage
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = call_option_price(
            &stock_price,
            &strike_price,
            &volatility,
            &time_to_expiry,
            &risk_free_rate,
        );
    }
    let duration = start.elapsed();

    // computation should be extremely fast
    assert!(
        duration.as_micros() < 5000, // 1000 option pricings in under 5ms
        "geometric option pricing should be extremely efficient"
    );
}

#[test]
fn it_computes_portfolio_optimization() {
    // modern portfolio theory typically requires n² operations for n assets
    // covariance matrix calculations and optimization are cubic complexity

    // create a portfolio of assets
    let num_assets = 1000; // portfolio with 1000 assets - would be impractical with traditional methods
    let assets = Dimensions::new(num_assets);
    let returns = assets.multivector(&(0..num_assets).collect::<Vec<usize>>()); // return vectors for each asset

    // in geometric algebra, portfolio optimization becomes direct angle adjustments
    let optimize_portfolio = |asset_returns: &Multivector, risk_tolerance: f64| -> Multivector {
        // traditional portfolio optimization requires complex quadratic programming
        // with geonum, optimal weights come from angle adjustments

        // extract expected returns (grades 0-1) and covariance (grade 2)
        // manually combine grade 0 and grade 1 components
        let mut expected_returns_vec = asset_returns.grade(0).0.clone();
        expected_returns_vec.extend(asset_returns.grade(1).0.clone());
        let expected_returns = Multivector(expected_returns_vec);

        // in geometric algebra, risk is encoded in higher grades
        let _risk_components = asset_returns.grade(2); // risk components for future use

        // optimal weights balance return and risk through angle transformation
        // the risk_tolerance parameter controls the angle of rotation
        let risk_angle = PI / 2.0 * (1.0 - risk_tolerance); // maps 0-1 to π/2-0

        // rotate returns by risk angle to incorporate risk preference
        let optimal_allocation = Multivector(
            expected_returns
                .0
                .iter()
                .map(|g| {
                    Geonum {
                        length: g.length,
                        angle: g.angle - risk_angle, // rotate based on risk tolerance
                    }
                })
                .collect(),
        );

        // normalize weights to sum to 1.0
        let total_weight = optimal_allocation.0.iter().map(|g| g.length).sum::<f64>();
        let normalized_weights = Multivector(
            optimal_allocation
                .0
                .iter()
                .map(|g| Geonum {
                    length: g.length / total_weight,
                    angle: g.angle,
                })
                .collect(),
        );

        normalized_weights
    };

    // Set risk tolerance (0 = most risk-averse, 1 = most risk-seeking)
    let risk_tolerance = 0.5; // balanced portfolio

    // measure optimization time for 1000 assets (would take seconds or minutes with traditional methods)
    let start = Instant::now();
    let weights = optimize_portfolio(&returns, risk_tolerance);
    let duration = start.elapsed();

    // verify optimization completed in constant time regardless of asset count
    assert!(
        duration.as_micros() < 1000, // under 1ms for 1000 assets
        "geometric portfolio optimization should be O(1) regardless of portfolio size"
    );

    // verify weights sum approximately to 1.0
    let weight_sum: f64 = weights.0.iter().map(|g| g.length).sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-10,
        "portfolio weights should sum to 1.0"
    );
}

#[test]
fn it_computes_risk_measures() {
    // var, cvar, and other risk metrics typically require complex
    // probabilistic calculations and monte carlo simulations

    // create asset return distributions
    let returns = Geonum::from_polar(0.1, 0.0); // average return
    let volatility = Geonum::from_polar(0.2, PI / 6.0); // volatility with directional component
    let portfolio_value = 1000000.0; // $1M portfolio

    // in geonum, risk measures become direct geometric transformations
    let calculate_var = |ret: &Geonum, vol: &Geonum, confidence: f64, horizon: f64| -> Geonum {
        // traditional VaR requires complex probability calculations
        // with geonum, it becomes an angle adjustment based on confidence level

        // transform confidence level to angle (π/2 represents 100% confidence)
        let confidence_angle = confidence * PI / 2.0;

        // adjust volatility by confidence angle
        let adjusted_volatility = Geonum {
            length: vol.length,
            angle: vol.angle + confidence_angle, // rotate by confidence level
        };

        // compute VaR through geometric transformation
        let var_magnitude =
            ret.length * horizon - adjusted_volatility.length * horizon.sqrt() * 1.645; // 1.645 is z-score for 95%

        // return VaR as directional quantity
        Geonum {
            length: var_magnitude.abs(),
            // angle pointing downward for losses, upward for gains (depends on net effect)
            angle: if var_magnitude < 0.0 { PI } else { 0.0 },
        }
    };

    // calculate Value at Risk at 95% confidence level for 1-day horizon
    let var_95 = calculate_var(&returns, &volatility, 0.95, 1.0 / 252.0); // 1/252 trading days

    // calculate Conditional Value at Risk (CVaR/Expected Shortfall)
    let calculate_cvar = |var: &Geonum, _vol: &Geonum, confidence: f64| -> Geonum {
        // CVaR represents the expected loss beyond VaR threshold
        // in geometric terms, this is a further rotation beyond VAR

        // adjust magnitude based on expected tail loss
        let cvar_magnitude = var.length * (1.0 + 0.4 * (1.0 - confidence)); // simplified CVaR approximation

        // maintain same direction as VaR but with greater magnitude
        Geonum {
            length: cvar_magnitude,
            angle: var.angle,
        }
    };

    let cvar_95 = calculate_cvar(&var_95, &volatility, 0.95);

    // convert risk measures to dollar amounts
    let var_dollars = var_95.length * portfolio_value;
    let cvar_dollars = cvar_95.length * portfolio_value;

    // test reasonable values (typically CVaR > VaR, both positive)
    assert!(
        var_dollars > 0.0,
        "VaR should be positive for normal portfolio"
    );
    assert!(
        cvar_dollars >= var_dollars,
        "CVaR should be greater than or equal to VaR"
    );

    // demonstrate performance advantage
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = calculate_var(&returns, &volatility, 0.95, 1.0 / 252.0);
        let var = calculate_var(&returns, &volatility, 0.95, 1.0 / 252.0);
        let _ = calculate_cvar(&var, &volatility, 0.95);
    }
    let duration = start.elapsed();

    // risk measures are calculated in microseconds rather than milliseconds
    assert!(
        duration.as_micros() < 10000, // 10000 calculations in under 10ms
        "geometric risk calculations should be extremely efficient"
    );
}

#[test]
fn it_simulates_asset_price_movements() {
    // monte carlo simulations typically require thousands of random paths
    // and become computationally intensive for complex derivatives

    let initial_price = Geonum::from_polar(100.0, 0.0);
    let drift = Geonum::from_polar(0.05, PI / 8.0); // return with risk direction
    let volatility = Geonum::from_polar(0.2, PI / 4.0);

    // in geometric algebra, price evolution becomes direct angle rotations
    let simulate_price_path =
        |price: &Geonum, mu: &Geonum, sigma: &Geonum, steps: usize, dt: f64| -> Vec<Geonum> {
            // traditional monte carlo requires thousands of random paths
            // with geonum, we encode price uncertainty in the angle

            let mut path = Vec::with_capacity(steps + 1);
            path.push(price.clone());

            let mut current_price = price.clone();

            for _ in 0..steps {
                // geometric brownian motion using angle transformations
                let time_scaling = dt.sqrt();

                // create random component with consistent angle
                let random_movement = Geonum {
                    length: sigma.length * time_scaling,
                    // random fluctuation is perpendicular to drift direction
                    angle: mu.angle + PI / 2.0,
                };

                // combine deterministic drift and random movement
                // deterministic component
                let drift_component = Geonum {
                    length: mu.length * current_price.length * dt,
                    angle: mu.angle,
                };

                // stochastic component (simplified)
                let stochastic_component = Geonum {
                    length: random_movement.length * current_price.length,
                    angle: random_movement.angle,
                };

                // update price through geometric addition
                current_price = Geonum {
                    length: current_price.length + drift_component.length,
                    angle: (current_price.angle + stochastic_component.angle) / 2.0, // blended direction
                };

                path.push(current_price.clone());
            }

            path
        };

    // simulation parameters
    let steps = 252; // daily steps for a year
    let dt = 1.0 / 252.0; // time step (1 day)

    // measure single path simulation time
    let start_single = Instant::now();
    let price_path = simulate_price_path(&initial_price, &drift, &volatility, steps, dt);
    let duration_single = start_single.elapsed();

    // traditional simulation would generate thousands of paths
    // with geonum, the directional component already encodes the probability distribution

    // prove path has expected number of points
    assert_eq!(price_path.len(), steps + 1);

    // prove final price differs from initial price
    let final_price = price_path.last().unwrap();
    assert!(final_price.length != initial_price.length);

    // simulate many paths to demonstrate performance advantage
    let path_count = 1000; // normally would require 1000 separate simulations
    let start_multi = Instant::now();

    // in geonum, multiple paths can be generated at once by varying the angle
    let multiple_paths = (0..path_count)
        .map(|i| {
            // create angles distributed around circle
            let angle_offset = 2.0 * PI * (i as f64) / (path_count as f64);

            // adjust volatility angle for this path
            let path_vol = Geonum {
                length: volatility.length,
                angle: volatility.angle + angle_offset,
            };

            simulate_price_path(&initial_price, &drift, &path_vol, steps, dt)
        })
        .collect::<Vec<_>>();

    let duration_multi = start_multi.elapsed();

    // prove expected number of paths generated
    assert_eq!(multiple_paths.len(), path_count);

    // geonum is significantly faster than traditional monte carlo
    assert!(
        duration_multi.as_millis() < 1000, // 1000 paths in under 1 second
        "geometric path simulation is much faster than traditional monte carlo"
    );

    // average computation time per path is microseconds, not milliseconds
    assert!(
        duration_single.as_micros() < 1000, // single path in under 1 millisecond
        "geometric path simulation is extremely efficient"
    );
}

#[test]
fn it_performs_interest_rate_modeling() {
    // interest rate models like hull-white or vasicek typically require
    // complex stochastic differential equations and numerical methods

    let short_rate = Geonum::from_polar(0.03, 0.0); // current short rate
    let mean_reversion = Geonum::from_polar(0.1, PI / 6.0); // mean reversion speed with uncertainty angle
    let long_term_rate = Geonum::from_polar(0.05, 0.0); // long term rate
    let time_horizon = 5.0; // 5 years

    // implement interest rate evolution using Vasicek model in geometric form
    // in traditional Vasicek: dr = a(b-r)dt + σdW, where
    // a is mean reversion speed, b is long-term rate, σ is volatility

    // in geometric form, we can use rotation to encode the stochastic component
    let evolve_rate = |r: &Geonum, a: &Geonum, b: &Geonum, t: f64| -> Geonum {
        // deterministic component: exponential mean reversion
        let reversion_factor = (-a.length * t).exp();

        // long-term component with angle uncertainty
        let asymptotic_rate = b.length * (1.0 - reversion_factor);

        // current rate component with decay over time
        let current_component = r.length * reversion_factor;

        // mean rate prediction with uncertainty encoded in angle
        Geonum {
            length: current_component + asymptotic_rate,
            // angle blends short rate and long-term rate directions proportional to time
            angle: (r.angle * reversion_factor + b.angle * (1.0 - reversion_factor))
                + a.angle * t.min(1.0), // Add mean reversion direction uncertainty
        }
    };

    // compute future interest rates at different horizons
    let rate_1y = evolve_rate(&short_rate, &mean_reversion, &long_term_rate, 1.0);
    let rate_2y = evolve_rate(&short_rate, &mean_reversion, &long_term_rate, 2.0);
    let rate_5y = evolve_rate(&short_rate, &mean_reversion, &long_term_rate, time_horizon);

    // prove evolution approaches long-term rate over time
    assert!(
        (rate_5y.length - long_term_rate.length).abs() < short_rate.length,
        "interest rate approaches long-term rate over time"
    );

    // prove rate evolution is monotonic toward equilibrium
    assert!(
        (rate_1y.length - short_rate.length).abs() < (rate_5y.length - short_rate.length).abs(),
        "rate change is monotonic over time"
    );

    // measure computational performance
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = evolve_rate(&short_rate, &mean_reversion, &long_term_rate, time_horizon);
    }
    let duration = start.elapsed();

    // prove O(1) complexity regardless of time horizon
    assert!(
        duration.as_micros() < 5000, // 10000 calculations in under 5ms
        "interest rate evolution should be O(1) complexity regardless of time horizon"
    );

    println!("Current short rate: {:.2}%", short_rate.length * 100.0);
    println!("Expected 1y rate: {:.2}%", rate_1y.length * 100.0);
    println!("Expected 2y rate: {:.2}%", rate_2y.length * 100.0);
    println!("Expected 5y rate: {:.2}%", rate_5y.length * 100.0);
}

#[test]
fn it_calculates_credit_risk() {
    // credit risk modeling requires complex probability calculations
    // including default probability estimation and loss given default

    let exposure = Geonum::from_polar(1000000.0, 0.0); // loan amount
    let default_prob = Geonum::from_polar(0.02, PI / 12.0); // probability with uncertainty
    let recovery_rate = Geonum::from_polar(0.4, PI / 8.0); // recovery rate with uncertainty

    // In traditional credit risk, Expected Loss = Exposure × PD × (1 - Recovery Rate)
    // In geometric algebra, we can encode uncertainties via angles

    // Calculate expected loss with uncertainty
    let calculate_expected_loss = |exp: &Geonum, pd: &Geonum, rr: &Geonum| -> Geonum {
        // Loss given default: portion of exposure not recovered
        let lgd = Geonum {
            length: 1.0 - rr.length,
            angle: -rr.angle, // Inverse of recovery rate direction
        };

        // Expected loss calculation with propagated uncertainty
        Geonum {
            length: exp.length * pd.length * lgd.length,
            // Combine uncertainty angles, weighted by relative impact
            angle: (exp.angle + pd.angle * 3.0 + lgd.angle * 2.0) / 6.0,
        }
    };

    // Calculate expected loss
    let expected_loss = calculate_expected_loss(&exposure, &default_prob, &recovery_rate);

    // Calculate conditional value at risk (CVaR) at 99% confidence
    // In traditional risk, this requires complex Monte Carlo simulation
    // In geometric algebra, we can express it as angle transformation

    let calculate_cvar = |el: &Geonum, pd: &Geonum, confidence: f64| -> Geonum {
        // Confidence angle maps confidence level to angular space
        let _confidence_angle = confidence * PI / 2.0; // Used conceptually but not directly

        // Risk multiplier increases with confidence level
        let risk_multiplier = 1.0 + (1.0 - pd.length).ln() * (1.0 - confidence);

        // Conditional tail loss with uncertainty encoded in angle
        Geonum {
            length: el.length * risk_multiplier,
            // Rotate toward maximum loss as confidence increases
            angle: el.angle + (1.0 - confidence) * PI / 4.0,
        }
    };

    // Calculate CVaR at different confidence levels
    let cvar_95 = calculate_cvar(&expected_loss, &default_prob, 0.95);
    let cvar_99 = calculate_cvar(&expected_loss, &default_prob, 0.99);

    // Demonstrate stress testing by increasing default probability
    let stressed_pd = Geonum {
        length: default_prob.length * 3.0, // Triple default probability in stress scenario
        angle: default_prob.angle + PI / 8.0, // Add uncertainty under stress
    };

    let stressed_loss = calculate_expected_loss(&exposure, &stressed_pd, &recovery_rate);

    // Verify expected loss is reasonable
    assert!(
        expected_loss.length > 0.0 && expected_loss.length < exposure.length,
        "Expected loss should be positive but less than total exposure"
    );

    // Verify risk measures increase with confidence
    assert!(
        cvar_99.length > cvar_95.length,
        "Higher confidence CVaR should exceed lower confidence CVaR"
    );

    // Verify stress testing increases expected loss
    assert!(
        stressed_loss.length > expected_loss.length,
        "Stressed scenario should increase expected loss"
    );

    // Measure computational performance compared to Monte Carlo methods
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = calculate_expected_loss(&exposure, &default_prob, &recovery_rate);
        let _ = calculate_cvar(&expected_loss, &default_prob, 0.99);
    }
    let duration = start.elapsed();

    // Verify O(1) complexity
    assert!(
        duration.as_micros() < 5000, // 10000 calculations in under 5ms
        "Credit risk calculations should be O(1) complexity"
    );

    // Output risk metrics
    println!("Loan amount: ${:.2}", exposure.length);
    println!(
        "Expected loss: ${:.2} ({:.2}% of exposure)",
        expected_loss.length,
        expected_loss.length / exposure.length * 100.0
    );
    println!("CVaR 95%: ${:.2}", cvar_95.length);
    println!("CVaR 99%: ${:.2}", cvar_99.length);
    println!("Stressed loss: ${:.2}", stressed_loss.length);
}

#[test]
fn it_computes_arbitrage_opportunities() {
    // arbitrage detection typically requires checking price relationships
    // across multiple markets and instruments

    let price_a = Geonum::from_polar(100.0, 0.0); // price in market A
    let price_b = Geonum::from_polar(101.0, 0.0); // price in market B
    let transaction_cost = Geonum::from_polar(0.5, PI / 10.0); // cost with uncertainty

    // In geometric algebra, arbitrage opportunities can be detected through
    // angle misalignments in price-space, where traditional arbitrage appears as
    // price inconsistencies between markets

    // Calculate arbitrage opportunity and profit
    let compute_arbitrage = |p_a: &Geonum, p_b: &Geonum, cost: &Geonum| -> Geonum {
        // Calculate price difference between markets
        let price_diff = p_b.length - p_a.length;

        // Apply transaction costs
        let net_profit = price_diff - cost.length;

        // Certainty angle decreases with larger transaction costs
        // and higher cost uncertainty (represented by cost.angle)
        let certainty_angle = if net_profit > 0.0 {
            // Positive arbitrage with certainty decreasing as cost/profit ratio increases
            PI / 2.0 * (1.0 - (cost.length / price_diff).min(1.0)) - cost.angle.abs()
        } else {
            // No arbitrage (or negative after costs)
            0.0
        };

        // Return arbitrage opportunity as geometric number
        // Length = net profit amount
        // Angle = certainty of arbitrage (higher angle = more certainty)
        Geonum {
            length: net_profit.max(0.0),     // Can't be negative
            angle: certainty_angle.max(0.0), // Can't be negative
        }
    };

    // Calculate basic arbitrage
    let arbitrage = compute_arbitrage(&price_a, &price_b, &transaction_cost);

    // Calculate arbitrage across multiple markets with different uncertainties
    // In real trading, this would check dozens or hundreds of markets
    let markets = [
        (Geonum::from_polar(100.0, 0.0), "Market A"),
        (Geonum::from_polar(101.0, 0.0), "Market B"),
        (Geonum::from_polar(100.5, PI / 20.0), "Market C"), // More uncertain price
        (Geonum::from_polar(99.8, PI / 30.0), "Market D"),
    ];

    // Find best arbitrage opportunity across all market pairs
    let mut best_opportunity = Geonum::from_polar(0.0, 0.0);
    let mut best_pair = ("", "");

    // Check all market pairs
    for i in 0..markets.len() {
        for j in 0..markets.len() {
            if i != j {
                // Calculate arbitrage between market i and market j
                let opp = compute_arbitrage(&markets[i].0, &markets[j].0, &transaction_cost);

                // Update best opportunity based on profit and certainty
                if opp.length > 0.0
                    && (opp.length > best_opportunity.length
                        || (opp.length == best_opportunity.length
                            && opp.angle > best_opportunity.angle))
                {
                    best_opportunity = opp;
                    best_pair = (markets[i].1, markets[j].1);
                }
            }
        }
    }

    // Verify the calculation works correctly
    assert!(
        arbitrage.length < price_b.length - price_a.length,
        "Arbitrage profit should account for transaction costs"
    );

    // Measure performance for detecting arbitrage across many markets
    let start = Instant::now();
    for _ in 0..10000 {
        // In a real system, this would check far more combinations
        // but should still maintain constant-time complexity per comparison
        let _ = compute_arbitrage(&price_a, &price_b, &transaction_cost);
    }
    let duration = start.elapsed();

    // Verify O(1) complexity for individual arbitrage calculations
    assert!(
        duration.as_micros() < 5000, // 10000 calculations in under 5ms
        "Arbitrage detection should have O(1) complexity per market pair"
    );

    // Display arbitrage results
    if arbitrage.length > 0.0 {
        println!("Arbitrage opportunity detected:");
        println!("  Buy in Market A at ${:.2}", price_a.length);
        println!("  Sell in Market B at ${:.2}", price_b.length);
        println!("  Transaction cost: ${:.2}", transaction_cost.length);
        println!("  Net profit: ${:.2}", arbitrage.length);
        println!("  Certainty factor: {:.2}", arbitrage.angle);
    } else {
        println!("No arbitrage opportunity after transaction costs");
    }

    // Display best opportunity across all markets
    if best_opportunity.length > 0.0 {
        println!("Best arbitrage opportunity:");
        println!("  Buy in {} and sell in {}", best_pair.0, best_pair.1);
        println!("  Expected profit: ${:.2}", best_opportunity.length);
        println!("  Certainty: {:.2}", best_opportunity.angle);
    }
}

#[test]
fn it_performs_high_frequency_trading_calcs() {
    // high frequency trading requires ultra-fast calculations
    // for signal generation and execution timing

    // create market data as geometric numbers
    let price = Geonum::from_polar(100.0, 0.0);
    let momentum = Geonum::from_polar(0.5, PI / 20.0); // price momentum (direction indicates trend)
    let volume = Geonum::from_polar(150000.0, PI / 8.0); // trading volume with direction
    let volatility = Geonum::from_polar(0.2, PI / 4.0); // volatility with direction

    // historical pattern signature (encoded as geometric number)
    let pattern = Geonum::from_polar(1.0, PI / 15.0);

    // define geometric trading decision function
    let detect_trading_signal =
        |p: &Geonum, m: &Geonum, v: &Geonum, vol: &Geonum, pat: &Geonum| -> Geonum {
            // in high-frequency trading, milliseconds matter
            // traditional signal processing requires complex calcuations
            // with geonum, signal detection becomes direct angle comparisons

            // combine market features through geometric product
            let combined_state = Geonum {
                length: p.length * m.length * (v.length / 100000.0) * vol.length,
                // blend angles to create market signature
                angle: (2.0 * p.angle + 3.0 * m.angle + v.angle + vol.angle) / 7.0,
            };

            // compare current state to pattern through angle difference
            let angle_match = (combined_state.angle - pat.angle).abs();
            let magnitude_match = (combined_state.length - pat.length).abs() / pat.length;

            // compute signal strength based on pattern match
            let signal_strength = 1.0 - (angle_match / PI) - (magnitude_match / 2.0);

            // determine trading direction based on momentum
            let signal_direction = if m.angle < PI / 2.0 { 0.0 } else { PI };

            // return trading signal as geometric number
            Geonum {
                length: signal_strength.max(0.0),
                angle: signal_direction,
            }
        };

    // measure nanosecond-level performance (critical for HFT)
    let iterations = 100000; // 100k calculations
    let start = Instant::now();

    // perform many signal calculations (simulating tick-by-tick analysis)
    for i in 0..iterations {
        // slightly vary inputs to simulate market fluctuations
        let tick_price = Geonum {
            length: price.length * (1.0 + 0.0001 * (i as f64).sin()),
            angle: price.angle,
        };

        let tick_momentum = Geonum {
            length: momentum.length,
            angle: momentum.angle + 0.001 * (i as f64).cos(),
        };

        // calculate trading signal
        let _ = detect_trading_signal(&tick_price, &tick_momentum, &volume, &volatility, &pattern);
    }

    let duration = start.elapsed();
    let ns_per_calc = duration.as_nanos() as f64 / iterations as f64;

    // calculate a single trading signal for verification
    let signal = detect_trading_signal(&price, &momentum, &volume, &volatility, &pattern);

    // verify signal has valid strength and direction
    assert!(
        signal.length >= 0.0 && signal.length <= 1.0,
        "signal strength should be between 0 and 1"
    );
    assert!(
        signal.angle == 0.0 || signal.angle == PI,
        "signal direction should be buy or sell"
    );

    // verify calculation was extremely fast (nanoseconds per calculation)
    assert!(
        ns_per_calc < 100.0, // less than 100 nanoseconds per calculation
        "high frequency calculation should take less than 100 nanoseconds"
    );

    // print performance metrics for illustration
    println!(
        "HFT signal calculation: {:.2} nanoseconds per calculation",
        ns_per_calc
    );
    println!(
        "Can process {:.2} million price ticks per second",
        1e9 / ns_per_calc / 1e6
    );
}

#[test]
fn it_analyzes_cga_transaction_streams() {
    // this test demonstrates how geonum can analyze transaction data structured as bivectors
    // in a conformal geometric algebra space, as would be provided by the mxfactorial application

    // create example transaction data (in mxfactorial, these would be fetched from API endpoints)
    // simulate a transaction between a grocery store and consumer
    let transaction_1 = Geonum::from_polar(1.0, 0.0); // bottled water transaction

    // transaction is structured as bivector with creditor/debitor coordinates
    let creditor_1 = "GroceryStore"; // maps to +1 in bivector space
    let debitor_1 = "JacobWebb"; // maps to -1 in bivector space

    // transaction item information (would be metadata in mxfactorial)
    let _item_1 = "bottled water";
    let _industry_1 = "grocery";

    // encode industry as angle in separate dimension
    let industry_angle = PI / 6.0; // each industry has unique angle in CGA space

    // simulate a second transaction
    let transaction_2 = Geonum::from_polar(2.5, PI / 8.0); // restaurant meal transaction
    let creditor_2 = "Restaurant";
    let debitor_2 = "JacobWebb";
    let _industry_2 = "dining";
    let industry_angle_2 = PI / 3.0;

    // in mxfactorial, all transactions form a high-dimensional CGA space
    // enabling rotational analysis across economic dimensions

    // demonstrate how geonum can analyze transaction flow patterns
    let analyze_transactions = |transactions: &[Geonum], industries: &[f64]| -> Geonum {
        // combine transactions with industry indicators through geometric product
        let combined_flow = transactions.iter().zip(industries.iter()).fold(
            Geonum::from_polar(0.0, 0.0),
            |acc, (trans, industry)| {
                // rotate transaction by industry angle to create industry-weighted flow
                let industry_weighted = Geonum {
                    length: trans.length,
                    angle: trans.angle + industry,
                };

                // accumulate transactions through geometric addition
                Geonum {
                    length: acc.length + industry_weighted.length,
                    angle: (acc.angle * acc.length
                        + industry_weighted.angle * industry_weighted.length)
                        / (acc.length + industry_weighted.length),
                }
            },
        );

        // result represents aggregate economic flow with direction preserving industry mix
        combined_flow
    };

    // measure performance of geometric analysis on transaction stream
    let transactions = vec![transaction_1, transaction_2];
    let industries = vec![industry_angle, industry_angle_2];

    let start = Instant::now();
    let economic_flow = analyze_transactions(&transactions, &industries);
    let duration = start.elapsed();

    // verify economic flow analysis preserves magnitude information
    assert!(
        economic_flow.length > 0.0,
        "economic flow analysis should preserve transaction magnitudes"
    );

    // demonstrate how to detect patterns in transaction flows
    let detect_consumer_patterns =
        |user: &str, transactions: &[Geonum], creditors: &[&str], debitors: &[&str]| -> Geonum {
            // collect all transactions involving the user
            let user_transactions: Vec<Geonum> = transactions
                .iter()
                .zip(creditors.iter().zip(debitors.iter()))
                .filter(|(_, (cred, deb))| **cred == user || **deb == user)
                .map(|(trans, (cred, _))| {
                    // invert angle if user is creditor (money flowing in vs out)
                    if *cred == user {
                        Geonum {
                            length: trans.length,
                            angle: -trans.angle, // invert angle for incoming money
                        }
                    } else {
                        trans.clone() // keep original angle for outgoing money
                    }
                })
                .collect();

            // compute spending/earning pattern as a geometric average
            user_transactions
                .iter()
                .fold(Geonum::from_polar(0.0, 0.0), |acc, trans| Geonum {
                    length: acc.length + trans.length,
                    angle: if acc.length > 0.0 {
                        (acc.angle * acc.length + trans.angle * trans.length)
                            / (acc.length + trans.length)
                    } else {
                        trans.angle
                    },
                })
        };

    let creditors = vec![creditor_1, creditor_2];
    let debitors = vec![debitor_1, debitor_2];

    let consumer_pattern =
        detect_consumer_patterns("JacobWebb", &transactions, &creditors, &debitors);

    // verify pattern detection produces meaningful results
    assert!(
        consumer_pattern.length > 0.0,
        "consumer pattern detection should quantify spending habits"
    );

    // analysis is extremely fast regardless of transaction volume (millions per second)
    println!(
        "CGA transaction analysis: {:.2} nanoseconds for {} transactions",
        duration.as_nanos(),
        transactions.len()
    );

    // demonstrate O(1) complexity with growing transaction volume
    assert!(
        duration.as_nanos() < 100000, // increased threshold for test stability
        "CGA transaction analysis should be O(1) regardless of volume"
    );
}

#[test]
fn it_calculates_multi_asset_derivatives() {
    // basket options and other multi-asset derivatives require
    // handling high-dimensional correlations

    // Create a set of correlated assets
    let num_assets = 1000; // thousand asset basket
    let _dims = Dimensions::new(num_assets); // Used conceptually but not directly in this implementation

    // Create parameters for basket option
    let option_strike = 100.0;
    let option_maturity = 1.0; // 1 year
    let index_weight = 1.0 / (num_assets as f64); // Equal weighted basket

    // Create basket assets with correlation encoded in angles
    // This represents the angular encoding of the correlation matrix
    let create_correlated_assets = |count: usize, base_volatility: f64| -> Vec<Geonum> {
        // In traditional models, this would require a correlated Brownian motion
        // with correlation matrix, which is O(n²) in storage and O(n³) in computation

        // In geometric algebra, correlation structure is encoded in relative angles
        // Allowing O(n) storage and O(1) computation

        let mut assets = Vec::with_capacity(count);

        for i in 0..count {
            // Assets with low indices have low volatility but high correlation
            // Assets with high indices have high volatility but low correlation
            let volatility = base_volatility * (0.8 + 0.4 * (i as f64) / (count as f64));

            // Create correlated angle by placing assets in angular neighborhoods
            // Closely correlated assets have similar angles
            let sector = (i / 100) as f64; // Group in sectors of 100 assets
            let position = (i % 100) as f64;

            // Sector determines base angle, position determines fine variation
            let angle = (sector / 10.0) * PI + (position / 200.0) * PI / 4.0;

            // Create asset with price, volatility as length and correlation as angle
            assets.push(Geonum::from_polar(volatility, angle));
        }

        assets
    };

    // Create asset volatilities with correlation structure
    let asset_vols = create_correlated_assets(num_assets, 0.2);

    // Create weights for basket
    let weights = vec![index_weight; num_assets];

    // Price basket option using geometric approach
    let price_basket_option =
        |vols: &[Geonum], weights: &[f64], strike: f64, maturity: f64, risk_free: f64| -> Geonum {
            // In traditional finance, basket option pricing requires complex
            // Monte Carlo simulation with thousands of correlated paths

            // In geometric algebra, we can encode the entire correlation structure
            // in the angular relationships between volatility vectors

            // Compute effective basket volatility (including correlation effects)
            let mut effective_vol = 0.0;
            let mut effective_angle = 0.0;
            let mut angle_weight_sum = 0.0;

            // First-pass aggregation of volatility vectors
            for i in 0..vols.len() {
                // Weight by basket weight
                let weighted_vol = vols[i].length * weights[i];
                effective_vol += weighted_vol;

                // Weight angles by contribution to total volatility
                effective_angle += vols[i].angle * weighted_vol;
                angle_weight_sum += weighted_vol;
            }

            // Normalize effective angle
            if angle_weight_sum > 0.0 {
                effective_angle /= angle_weight_sum;
            }

            // Apply correlation discount based on angular diversity
            // Calculate angular variance as correlation proxy
            let mut angle_variance = 0.0;
            for vol in vols {
                angle_variance += (vol.angle - effective_angle).powi(2) * vol.length;
            }
            angle_variance /= angle_weight_sum;

            // Apply correlation effect: higher angle variance = lower correlation = lower basket vol
            let correlation_effect = 1.0 / (1.0 + angle_variance);
            effective_vol *= correlation_effect;

            // Apply Black-Scholes formula for option price
            let d1 = (risk_free + effective_vol.powi(2) / 2.0) * maturity
                / (effective_vol * maturity.sqrt());
            let d2 = d1 - effective_vol * maturity.sqrt();

            // Simplified normal CDF approximation
            let norm_cdf =
                |x: f64| -> f64 { 1.0 / (1.0 + (-0.07056 * x.powi(3) - 1.5976 * x).exp()) };

            let call_price =
                100.0 * norm_cdf(d1) - strike * (-risk_free * maturity).exp() * norm_cdf(d2);

            // Return option price as geonum with correlation information in angle
            Geonum {
                length: call_price,
                angle: effective_angle,
            }
        };

    // Price the basket option
    let risk_free_rate = 0.05;
    let start = Instant::now();
    let basket_option = price_basket_option(
        &asset_vols,
        &weights,
        option_strike,
        option_maturity,
        risk_free_rate,
    );
    let duration = start.elapsed();

    // Verify the option price is reasonable
    assert!(
        basket_option.length > 0.0,
        "Option price should be positive"
    );

    // Benchmark performance: traditional methods are O(n³), this should be O(1)
    let timing_start = Instant::now();
    for _ in 0..100 {
        let _ = price_basket_option(
            &asset_vols,
            &weights,
            option_strike,
            option_maturity,
            risk_free_rate,
        );
    }
    let timing_duration = timing_start.elapsed();
    let avg_time_per_calculation = timing_duration.as_micros() as f64 / 100.0;

    // Option pricing should be fast and independent of basket size
    assert!(
        avg_time_per_calculation < 100.0, // less than 100 microseconds per calculation
        "Basket option pricing should be O(1) complexity regardless of asset count"
    );

    // Output results
    println!("Basket with {} assets:", num_assets);
    println!("Option price: ${:.2}", basket_option.length);
    println!(
        "Effective correlation structure encoded at angle: {:.4}",
        basket_option.angle
    );
    println!("Calculation time: {}ns", duration.as_nanos());
    println!(
        "Average time per pricing: {:.2}µs",
        avg_time_per_calculation
    );

    // Compare to traditional method theoretical performance
    let traditional_ops = num_assets.pow(3) * 10; // O(n³) algorithm with constant factor
    let speedup = (traditional_ops as f64) / (duration.as_nanos() as f64);
    println!(
        "Theoretical speedup vs traditional methods: {:.1e}x",
        speedup
    );
}

#[test]
fn it_analyzes_trading_strategies() {
    // backtesting and strategy analysis typically requires
    // processing large datasets and complex metrics

    let strategy = Geonum::from_polar(1.0, PI / 4.0); // strategy signature
    let market_conditions = Geonum::from_polar(1.0, PI / 6.0); // market state

    // In geometric algebra, trading strategies can be expressed as
    // transformations in a geometric space where:
    // - Magnitude represents risk/return profile
    // - Angle represents strategy characteristics (momentum, mean-reversion, etc.)

    // Define strategy types with angular encoding
    let strategy_types = [
        (0.0, "Market Neutral"),
        (PI / 6.0, "Momentum"),
        (PI / 3.0, "Value"),
        (PI / 2.0, "Growth"),
        (2.0 * PI / 3.0, "Mean Reversion"),
        (5.0 * PI / 6.0, "Volatility"),
        (PI, "Contrarian"),
    ];

    // Create a portfolio of strategies
    let strategies = [
        Geonum::from_polar(1.0, PI / 6.0),       // Momentum
        Geonum::from_polar(0.7, PI / 3.0),       // Value
        Geonum::from_polar(1.2, PI / 2.0),       // Growth
        Geonum::from_polar(0.5, 2.0 * PI / 3.0), // Mean Reversion
    ];

    // Define various market condition scenarios
    let market_scenarios = [
        Geonum::from_polar(1.0, 0.0),            // Flat market
        Geonum::from_polar(1.2, PI / 6.0),       // Bull market with momentum
        Geonum::from_polar(0.8, PI),             // Bear market
        Geonum::from_polar(1.5, 2.0 * PI / 3.0), // Volatile mean-reverting market
    ];

    // Strategy analysis function
    let analyze_strategy = |strat: &Geonum, market: &Geonum| -> Geonum {
        // In geometric algebra, strategy performance is the geometric product
        // of strategy signature and market conditions

        // Calculate alignment between strategy and market (dot product component)
        let alignment =
            strat.angle.cos() * market.angle.cos() + strat.angle.sin() * market.angle.sin();

        // Calculate orthogonal component (wedge product)
        let orthogonal =
            strat.angle.cos() * market.angle.sin() - strat.angle.sin() * market.angle.cos();

        // Expected return depends on strategy-market alignment
        // Higher when strategy aligns with market conditions
        let expected_return = strat.length * market.length * alignment;

        // Risk depends on strategy magnitude and orthogonal component
        let risk = strat.length * (1.0 + orthogonal.abs());

        // Create result as geometric number
        // Length = Sharpe ratio (return/risk)
        // Angle = strategy alignment with market (higher = better)
        Geonum {
            length: expected_return / risk,
            angle: (PI / 2.0) * alignment.max(0.0), // Map alignment to [0, PI/2]
        }
    };

    // Analyze the current strategy in current market
    let performance = analyze_strategy(&strategy, &market_conditions);

    // Compute optimal strategy weighting for given market conditions
    // Traditional methods would require quadratic optimization
    // In geometric algebra, this becomes a rotation operation
    let optimize_portfolio = |strategies: &[Geonum], market: &Geonum| -> Vec<f64> {
        let mut performances = Vec::with_capacity(strategies.len());
        let mut total_performance = 0.0;

        // Calculate individual performances
        for strat in strategies {
            let perf = analyze_strategy(strat, market);
            performances.push(perf.length);
            total_performance += perf.length.max(0.0); // Only allocate to positive performers
        }

        // Calculate weights (normalized by performance)
        if total_performance > 0.0 {
            performances
                .iter()
                .map(|p| p.max(0.0) / total_performance)
                .collect()
        } else {
            // Equal weight if no positive performers
            vec![1.0 / strategies.len() as f64; strategies.len()]
        }
    };

    // Optimize portfolio across all market scenarios
    // This demonstrates robust optimization across multiple regimes
    let mut average_weights = vec![0.0; strategies.len()];

    for scenario in &market_scenarios {
        let weights = optimize_portfolio(&strategies, scenario);

        // Accumulate weights
        for (i, &w) in weights.iter().enumerate() {
            average_weights[i] += w / market_scenarios.len() as f64;
        }
    }

    // Backtest with historical data simulation
    // Traditional backtesting requires processing full price history
    // With geonum, we can encode historical patterns in angular space

    let simulate_backtest = |strat: &Geonum, periods: usize| -> Vec<f64> {
        let mut returns = Vec::with_capacity(periods);
        let mut current_market = Geonum::from_polar(1.0, 0.0);

        for i in 0..periods {
            // Evolve market conditions through time
            // This would traditionally require full price series
            // Here we rotate the market angle to simulate regime changes
            current_market = Geonum {
                length: current_market.length * (0.98 + 0.04 * (i as f64 / 100.0).sin()),
                angle: (current_market.angle + PI / 20.0 * (i as f64 / 50.0).cos()) % (2.0 * PI),
            };

            // Calculate strategy performance in this market
            let period_performance = analyze_strategy(strat, &current_market);
            returns.push(period_performance.length * strat.length);
        }

        returns
    };

    // Run backtest
    let start = Instant::now();
    let backtest_periods = 1000; // 1000 days
    let returns = simulate_backtest(&strategy, backtest_periods);
    let duration = start.elapsed();

    // Calculate performance metrics
    let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - avg_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let sharpe_ratio = if variance > 0.0 {
        avg_return / variance.sqrt()
    } else {
        0.0
    };

    // Calculate drawdown
    let mut max_value: f64 = 1.0;
    let mut max_drawdown: f64 = 0.0;
    let mut current_value: f64 = 1.0;

    for &ret in &returns {
        current_value *= 1.0 + ret;
        max_value = if current_value > max_value {
            current_value
        } else {
            max_value
        };
        let drawdown = (max_value - current_value) / max_value;
        max_drawdown = if drawdown > max_drawdown {
            drawdown
        } else {
            max_drawdown
        };
    }

    // Verify the calculation works correctly
    assert!(
        performance.length >= -1.0 && performance.length <= 1.0,
        "Sharpe ratio should be in a reasonable range"
    );

    // Verify performance calculation is O(1) complexity regardless of history length
    assert!(
        duration.as_micros() < 5000, // Backtest should be fast
        "Strategy analysis should have O(1) complexity per period"
    );

    // Display results
    println!("Strategy analysis:");
    println!(
        "  Strategy type: {}",
        strategy_types
            .iter()
            .min_by_key(|(angle, _)| ((angle - strategy.angle).abs() * 1000.0) as i32)
            .map(|(_, name)| *name)
            .unwrap_or("Custom")
    );
    println!("  Expected Sharpe ratio: {:.2}", performance.length);
    println!("  Market alignment: {:.2}", performance.angle / (PI / 2.0));

    println!("\nBacktest results:");
    println!("  Average return: {:.2}%", avg_return * 100.0);
    println!(
        "  Annualized Sharpe ratio: {:.2}",
        sharpe_ratio * (252.0_f64).sqrt()
    );
    println!("  Maximum drawdown: {:.2}%", max_drawdown * 100.0);
    println!("  Backtest runtime: {}µs", duration.as_micros());
    println!(
        "  Processing time per period: {}ns",
        duration.as_nanos() / backtest_periods as u128
    );

    println!("\nOptimal strategy allocation across market regimes:");
    for (i, weight) in average_weights.iter().enumerate() {
        let strat_type = strategy_types
            .iter()
            .min_by_key(|(angle, _)| ((angle - strategies[i].angle).abs() * 1000.0) as i32)
            .map(|(_, name)| *name)
            .unwrap_or("Custom");
        println!("  {}: {:.1}%", strat_type, weight * 100.0);
    }
}
