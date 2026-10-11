use geonum::*;

// causal economics: value moves on the null
//
// a transaction is two prices a half turn apart, the producer's at θ and
// the consumer's at θ + π, and it settles when they cancel: [p, θ] +
// [p, θ + π] = 0. two conditions close it, the half turn and the equal
// magnitudes, and each is read off the sum: any other gap or any other
// price leaves a residual. the cancelled sum keeps the blades of both legs,
// and that is the record. the causal order is per account: what an account
// spends must have reached it through a pair that already closed, which is
// a grade read off its running sum, each leg read on its own pair's axis. a
// leg with no partner leaves the book's sum on the consumer's ray, and that
// residual is the one thing the order forbids
//
// run: cargo test --test causal_economics_test -- --show-output

// --- the kernel

#[test]
fn it_settles_a_transaction_as_the_null_interval() {
    // two prices cancelling, the producer's at θ, the consumer's a half turn
    // on. the record carries both windings
    let price = 1.000_f64;
    let pole = Angle::new(2.0, 7.0); // the producer's pole, off every lattice point
    let producer = Geonum::new_with_angle(price, pole);
    let consumer = Geonum::new_with_angle(price, pole.dual());
    let trade = producer + consumer;
    assert!(trade.near_mag(0.0), "settled: the prices cancel");
    assert_eq!(
        trade.angle.blade(),
        producer.angle.blade() + consumer.angle.blade(),
        "the record carries both windings"
    );

    // the order of the two measurements is free: a buyer can timestamp a
    // lower price first and the seller approve it two turns later, or the
    // seller can post first. either way the pair closes at the later
    // measurement and the record sums the same two windings
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let ask_first = Geonum::new_with_angle(price, pole + turn(0))
        + Geonum::new_with_angle(price, pole.dual() + turn(2));
    let bid_first = Geonum::new_with_angle(price, pole.dual() + turn(0))
        + Geonum::new_with_angle(price, pole + turn(2));
    assert!(
        ask_first.near_mag(0.0) && bid_first.near_mag(0.0),
        "settled either way"
    );
    assert_eq!(
        ask_first.angle.blade(),
        bid_first.angle.blade(),
        "the record does not care who measured first"
    );
}

// --- how to harden a soft science

#[test]
fn it_settles_at_the_half_turn_and_at_no_other_gap() {
    // sweep the consumer around the producer. the residual of the sum is
    // 2p·|cos(gap/2)|, the law of cosines read off Add, and it is the null at
    // the half turn alone
    let price = 2.5_f64;
    let pole = Angle::new(2.0, 7.0);
    let producer = Geonum::new_with_angle(price, pole);
    for gap in [
        Angle::new(1.0, 2.0),
        Angle::new(3.0, 4.0),
        Angle::new(1.0, 1.0),
        Angle::new(5.0, 4.0),
        Angle::new(3.0, 2.0),
    ] {
        let off = Geonum::new_with_angle(price, pole + gap);
        let expected = 2.0 * price * (gap.grade_angle() / 2.0).cos().abs();
        assert!(
            (producer + off).near_mag(expected),
            "at gap {gap} the residual is 2p·|cos(gap/2)|"
        );
    }
}

#[test]
fn it_settles_at_equal_measurements_and_at_no_other_price() {
    // sweep the buyer's measurement against the seller's. the residual of
    // the sum is |p_s − p_b| on the ray of whoever measured more, and it is
    // the null at equality alone. the pair closes only when two measurements
    // neither party controls agree; until then there is no record, only a
    // leg left standing on one ray
    let pole = Angle::new(2.0, 7.0);
    let seller = Geonum::new_with_angle(1.000, pole);
    let buyer = |measured: f64| Geonum::new_with_angle(measured, pole.dual());

    assert!(
        (seller + buyer(1.000)).near_mag(0.0),
        "1.000 against 1.000 is the null"
    );

    let under = seller + buyer(0.900);
    assert!(under.near_mag(0.100), "the buyer measured 0.100 less");
    assert_eq!(
        under.angle.grade(),
        pole.grade(),
        "and the residual stands on the seller's ray"
    );

    let over = seller + buyer(1.100);
    assert!(over.near_mag(0.100), "the buyer measured 0.100 more");
    assert_eq!(
        over.angle.grade(),
        pole.dual().grade(),
        "and the residual stands on the buyer's ray"
    );
}

#[test]
fn it_lets_an_account_spend_only_what_reached_it_through_a_closed_pair() {
    // an account's receipt leg sits on its pole, its spend leg on the dual,
    // and its running sum, read at every winding, sits on the producer's
    // ray or the null, never on the consumer's ray
    let price = 2.5_f64;
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let account = Angle::new(3.0, 5.0);

    // credited at turn 2 when the pair with its customer closed, debited at
    // turn 3 when it paid its own supplier
    let received = Geonum::new_with_angle(price, account + turn(2));
    let spent = Geonum::new_with_angle(price, account.dual() + turn(3));
    let at_turn_2: GeoCollection = [received].into_iter().collect();
    assert_eq!(
        at_turn_2.wave_sum().angle.grade(),
        account.grade(),
        "turn 2: the balance sits on the producer's ray"
    );
    assert!(at_turn_2.wave_sum().near_mag(price), "by what was received");
    let at_turn_3: GeoCollection = [received, spent].into_iter().collect();
    assert!(
        at_turn_3.wave_sum().near_mag(0.0),
        "turn 3: spent what was received, the balance is the null"
    );

    // the violation as a quantity: debit at turn 1 before anything arrived
    // and the running sum sits on the consumer's ray by the full amount
    let spent_early = Geonum::new_with_angle(price, account.dual() + turn(1));
    let at_turn_1: GeoCollection = [spent_early].into_iter().collect();
    assert_eq!(
        at_turn_1.wave_sum().angle.grade(),
        account.dual().grade(),
        "turn 1: the balance sits on the consumer's ray"
    );
    assert!(
        at_turn_1.wave_sum().near_mag(price),
        "by the full amount spent"
    );
}

#[test]
fn it_orders_the_records_so_the_receipt_precedes_the_spend() {
    // "through a pair that already closed": a closed pair's record sums both
    // legs' blades, two of the pole's own plus the half turn plus eight per
    // turn. read the turn count back off the record with the pole removed,
    // so the order compares time and not which pole the pair sat on.
    // follow value A → B → C
    let price = 2.5_f64;
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let closed_at =
        |record: &Geonum, pole: Angle| (record.angle.blade() - 2 * pole.blade() - 2) / 8;
    let (a, b) = (Angle::new(2.0, 7.0), Angle::new_with_blade(3, 3.0, 5.0));

    // A → B closes at turn 2, B → C at turn 3. the poles sit at different
    // grades on purpose; the turn count reads through them
    let ab = Geonum::new_with_angle(price, a + turn(2))
        + Geonum::new_with_angle(price, a.dual() + turn(2));
    let bc = Geonum::new_with_angle(price, b + turn(3))
        + Geonum::new_with_angle(price, b.dual() + turn(3));
    assert!(ab.near_mag(0.0) && bc.near_mag(0.0), "both pairs settled");
    assert_eq!(closed_at(&ab, a), 2, "the receipt closed at turn 2");
    assert_eq!(closed_at(&bc, b), 3, "the spend closed at turn 3");
    assert!(
        closed_at(&ab, a) < closed_at(&bc, b),
        "the receipt's record precedes the spend's: already closed"
    );

    // close B → C at turn 1, before A → B at turn 2, and the comparison
    // flips with nothing else in the records differing
    let bc_early = Geonum::new_with_angle(price, b + turn(1))
        + Geonum::new_with_angle(price, b.dual() + turn(1));
    assert!(bc_early.near_mag(0.0), "settled between B and C");
    assert!(
        closed_at(&bc_early, b) < closed_at(&ab, a),
        "but recorded before B had received: the chain runs backward"
    );
}

#[test]
fn it_reads_one_accounts_book_across_two_poles_on_each_pairs_own_axis() {
    // B receives from A in a pair at pole a and spends to C in a pair at
    // pole b, the poles at different grades as above. B's book cannot sum
    // the two raw legs: they sit a gap apart that is no half turn, and the
    // sum is the 2p·|cos(gap/2)| residual of the sweep, neither a balance
    // nor a null. the board's move (cgt_test): project each leg onto its own
    // pair's axis. on its own axis a leg projects whole, p on the producer's
    // side and p on the consumer's, and those cells sum as a book. the when
    // stays on the legs' blades, where the record above reads it
    let price = 2.5_f64;
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let (a, b) = (Angle::new(2.0, 7.0), Angle::new_with_blade(3, 3.0, 5.0));
    let received = Geonum::new_with_angle(price, a + turn(2)); // B's leg in A → B
    let spent = Geonum::new_with_angle(price, b.dual() + turn(3)); // B's leg in B → C

    // raw, the two legs interfere at their gap
    let gap = (b.dual() - a).grade_angle();
    assert!(
        (received + spent).near_mag(2.0 * price * (gap / 2.0).cos().abs()),
        "two poles do not make a book: the raw sum is the sweep's residual"
    );

    // on its own pair's axis each leg is a whole cell with its side
    let cell_in = received.project_to_angle(a);
    let cell_out = spent.project_to_angle(b);
    assert!(
        cell_in.near_mag(price) && cell_in.angle.grade() == 0,
        "the receipt projects whole, producer's side"
    );
    assert!(
        cell_out.near_mag(price) && cell_out.angle.grade() == 2,
        "the spend projects whole, consumer's side"
    );
    // on the other pair's axis it would shrink by the cosine of the gap, so
    // the book reads each pair on the axis the pair closed on
    assert!(
        received
            .project_to_angle(b)
            .near_mag(price * (b - a).grade_angle().cos().abs()),
        "a leg on another pair's axis is a shadow, not a cell"
    );

    // B's book, turn by turn: received at 2, the balance on the producer's
    // side by p; spent at 3, the null
    let at_turn_2: GeoCollection = [cell_in].into_iter().collect();
    assert_eq!(
        at_turn_2.wave_sum().angle.grade(),
        0,
        "turn 2: balance on the producer's side"
    );
    assert!(at_turn_2.wave_sum().near_mag(price), "by what was received");
    let at_turn_3: GeoCollection = [cell_in, cell_out].into_iter().collect();
    assert!(
        at_turn_3.wave_sum().near_mag(0.0),
        "turn 3: spent what was received"
    );

    // the same two pairs with B spending at turn 1 before receiving at turn
    // 2: at turn 1 the book is one cell on the consumer's side. refused
    let spent_early = Geonum::new_with_angle(price, b.dual() + turn(1));
    let at_turn_1: GeoCollection = [spent_early.project_to_angle(b)].into_iter().collect();
    assert_eq!(
        at_turn_1.wave_sum().angle.grade(),
        2,
        "turn 1: on the consumer's side"
    );
    assert!(at_turn_1.wave_sum().near_mag(price), "by the full amount");
}

// --- counterfeit

#[test]
fn it_refuses_a_deposit_written_against_a_note_instead_of_erased_from_a_balance() {
    // a bank buys mary's note by writing her a deposit. the deposit is a
    // debit on the bank's own balance, and the bank has 50 earned
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let bank = Angle::new(4.0, 9.0);
    let earned = Geonum::new_with_angle(50.0, bank + turn(1));
    let deposit = Geonum::new_with_angle(100.0, bank.dual() + turn(2));

    // the rule reads the bank's running sum: 50 earned, 100 written, the
    // sum lands on the consumer's ray by 50. refused
    let trial: GeoCollection = [earned, deposit].into_iter().collect();
    let balance = trial.wave_sum();
    assert_eq!(
        balance.angle.grade(),
        bank.dual().grade(),
        "the bank is overdrawn"
    );
    assert!(balance.near_mag(50.0), "by the 50 it did not have");

    // the move that lets it through: the note, an item the bank just
    // acquired, booked on the bank's producer ray as if it were money. the
    // sum then reads as a surplus and nothing has been received from anyone
    let note_as_equity = Geonum::new_with_angle(100.0, bank + turn(2));
    let dressed: GeoCollection = [earned, deposit, note_as_equity].into_iter().collect();
    assert_eq!(
        dressed.wave_sum().angle.grade(),
        bank.grade(),
        "with the note booked as equity the bank's sum reads as a surplus"
    );
    assert!(
        dressed.wave_sum().near_mag(50.0),
        "of 50, the deposit's uncovered half wearing the note's face"
    );
}

// --- the measure

#[test]
fn it_refuses_the_leg_where_the_soft_science_manages_the_angle() {
    // one account, a sequence of proposed legs, each a geonum whose ray says
    // which end of the pair it is. the hard book admits a leg only if the
    // running sum stays off the consumer's ray. the soft book admits every
    // leg and inherits the residual
    let account = Angle::new(3.0, 5.0);
    let turn = |n: usize| Angle::new_with_blade(4 * n, 0.0, 1.0);
    let proposed = [
        Geonum::new_with_angle(10.0, account + turn(1)), // 10 arrives
        Geonum::new_with_angle(4.0, account.dual() + turn(2)), // spends 4
        Geonum::new_with_angle(8.0, account.dual() + turn(3)), // tries 8 with 6 on hand
        Geonum::new_with_angle(5.0, account + turn(4)),  // 5 arrives
        Geonum::new_with_angle(3.0, account.dual() + turn(5)), // spends 3
        Geonum::new_with_angle(20.0, account.dual() + turn(6)), // tries 20 with 8 on hand
    ];

    let mut hard: Vec<Geonum> = Vec::new();
    let mut refused = 0;
    for &leg in &proposed {
        let trial: GeoCollection = hard.iter().copied().chain([leg]).collect();
        let balance = trial.wave_sum();
        let overdrawn = balance.angle.grade() == account.dual().grade() && !balance.near_mag(0.0);
        if overdrawn {
            refused += 1;
        } else {
            hard.push(leg);
        }
    }
    assert_eq!(
        refused, 2,
        "the two spends with nothing behind them are refused"
    );

    // the hard book ends on the producer's ray by what remains unspent, and
    // one unit of it buys one unit of what arrived
    let hard_book: GeoCollection = hard.iter().copied().collect();
    let hard_balance = hard_book.wave_sum();
    assert!(hard_balance.near_mag(10.0 - 4.0 + 5.0 - 3.0), "8 unspent");
    assert_eq!(
        hard_balance.angle.grade(),
        account.grade(),
        "on the producer's ray"
    );
    let arrived = 10.0 + 5.0;
    assert!(
        Geonum::scalar(measure(&hard_book, arrived, account)).near_mag(1.0),
        "a surplus on the producer's ray is not a stretch"
    );

    // the soft book ends on the consumer's ray by the two admitted legs net
    // of what was on hand, and its unit buys less by that residual over
    // what arrived. the angle it then manages is the one it admitted
    let soft_book: GeoCollection = proposed.iter().copied().collect();
    let soft_balance = soft_book.wave_sum();
    assert!(soft_balance.near_mag(20.0), "overdrawn by 20");
    assert_eq!(
        soft_balance.angle.grade(),
        account.dual().grade(),
        "on the consumer's ray"
    );
    assert!(
        Geonum::scalar(measure(&soft_book, arrived, account)).near_mag((arrived + 20.0) / arrived),
        "the soft measure stretched by the residual it admitted"
    );
}

// --- the record

#[test]
fn it_holds_the_angle_while_production_takes_the_magnitude_to_zero() {
    // a settled pair at a pole, then production tick after tick: each turn
    // scales the magnitude down and winds the record on. the magnitude goes
    // to zero. the pole, which says who produced, and the blade, which says
    // when, do not
    let pole = Angle::new(2.0, 7.0);
    let turn = Angle::new(4.0, 2.0);
    let mut value = Geonum::new_with_angle(100.0, pole);
    let start = value;
    for gain in [0.5_f64, 0.5, 0.5, 0.0] {
        value = value.rotate(turn).scale(gain);
    }
    assert!(value.near_mag(0.0), "production took the magnitude to zero");
    assert_eq!(
        value.angle.base_angle(),
        start.angle.base_angle(),
        "the pole is intact: who produced is still on the record"
    );
    assert_eq!(
        value.angle.blade(),
        start.angle.blade() + 16,
        "four turns of when"
    );

    // the pair at zero magnitude still settles, and its record still sums
    // both windings
    let settled = value + value.dual();
    assert!(
        settled.near_mag(0.0),
        "a zero-magnitude pair is still a pair"
    );
    assert_eq!(
        settled.angle.blade(),
        value.angle.blade() + value.dual().angle.blade(),
        "and the record carries it"
    );
}

// --- the book

// what one unit buys, inverted to read as a stretch: what arrived plus
// whatever the book left unmatched on the consumer's ray, over what arrived.
// a null book measures one, and so does a surplus on the producer's ray
fn measure(book: &GeoCollection, arrived: f64, pole: Angle) -> f64 {
    let sum = book.wave_sum();
    let unpartnered = if sum.angle.grade() == pole.dual().grade() {
        sum.mag
    } else {
        0.0
    };
    (arrived + unpartnered) / arrived
}
