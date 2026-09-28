use geonum::*;

// combinatorial game theory is what humans can see in spacetime
//
// conway understands a game by building a toy: a position is its option
// sets, G = {G^L | G^R}, every future board enumerated before the first
// move. the cast under the toy is a buyer and a seller, two vectors a half
// turn apart, and their trade is one addition: matched magnitudes at dual
// angles cancel to the null, and the vanished magnitude carries the pair's
// summed blade, the winding that timestamps the trade. the board is not
// underneath the pair. it falls out of projecting the pair onto its own
// axis, cell by cell
//
// what is left of the theory once the spacetime is in place is the sensory
// layer: the pipeline that delivers the present to a human. prices are
// directions, optimization is a reader taking the smallest angle that has
// reached them, the seat is a saddle point, and the present flow is one
// SELECT over what arrived
//
// the suite runs in that order: conway's engine as the foil, the whole
// board dropping out of one projection, the thermograph read as a light
// cone, settlement with its winding, and then the human layer. the
// spacetime it stands on is proven elsewhere and is not repeated here —
// the causal trichotomy, boosts and aberration in spacetime_test, the
// crossing height and the fixed pair in stereographic_test, the scalar
// layer in minkowski_space_test, transactions as pairs of events with
// places and times in economic_spacetime_test
//
// run: cargo test --test cgt_test -- --show-output

// earth's radius in light-seconds: 6 371 000 m over 299 792 458 m/s
const EARTH_LS: f64 = 6_371_000.0 / 299_792_458.0;

// ------------------------------------------------------------------ the toy

#[test]
fn it_projects_every_future_board_before_the_first_move() {
    // conway's game is nothing but its option sets: G = {G^L | G^R} — a
    // position is the enumeration of all movement projected from it,
    // boards drawn before any board is played. the matched size-2 trade,
    // as the demand requires it built:
    let matched = disjunctive_sum(&int(2), &neg(&int(2)));

    // the engine settles by exhausting the stack: neither starter wins —
    // the balanced trade is a P-position, conway's zero
    assert!(
        second_player_wins(&matched),
        "the balanced trade is a P-position"
    );

    // the demand is countable: 19 boards drawn to settle a size-2 trade
    assert_eq!(positions(&matched), 19, "19 boards to settle size 2");

    // the same settlement without the board: one addition
    let buyer = Geonum::scalar(2.0);
    let settled = buyer + buyer.dual();
    assert!(
        settled.near_mag(0.0),
        "one addition settles what 19 boards project"
    );

    // the stack is the 2^n maze in game form: the boards for a size-n trade
    // number Σ C(i+j, i) over i, j ≤ n — every lattice path to every
    // position, drawn once per path. 19, 69, 251 — the count more than
    // triples per unit of trade while the addition stays one op
    for n in [2usize, 3, 4] {
        let trade = disjunctive_sum(&int(n), &neg(&int(n)));
        assert!(second_player_wins(&trade), "size {n} settles the same");
        assert_eq!(
            positions(&trade),
            pascal_stack(n),
            "size {n} draws the binomial stack"
        );
        let size = Geonum::scalar(n as f64);
        assert!(
            (size + size.dual()).near_mag(0.0),
            "still one addition at size {n}"
        );
    }
}

#[test]
fn it_drops_the_gameboard_out_of_one_projection() {
    // conway's board for a size-n trade is the (n+1)×(n+1) grid of
    // positions (i, j): Left holding i squares, Right holding j. the engine
    // sorts every cell by exhaustive search, and to reach a cell it draws
    // every path to it. the board is not underneath the pair. it falls out
    // of projecting the pair onto its own axis: a cell is the buyer at size
    // i against the seller at size j, and its class is the grade of that
    // residual's shadow on the payoff axis — L on the axis, R a half turn
    // on, P at zero extent
    let pole = Angle::new(2.0, 7.0); // the buyer's pole, off every lattice point
    let n = 3;
    let mut boards_drawn = 0;
    for i in 0..=n {
        for j in 0..=n {
            let cell = disjunctive_sum(&int(i), &neg(&int(j)));
            boards_drawn += positions(&cell);

            let residual = Geonum::new_with_angle(i as f64, pole)
                + Geonum::new_with_angle(j as f64, pole.dual());
            assert_eq!(
                outcome(&cell),
                read(&residual, pole),
                "cell ({i}, {j}) is the residual's shadow on the pair's axis"
            );
        }
    }

    // the engine drew 226 boards to sort 16 cells; the projection read one
    // cosine per cell. the grid is a line of residuals seen edge-on
    assert_eq!(boards_drawn, 226, "226 boards for a 4×4 grid");

    // drop an open move onto every cell — a star token — and the diagonal
    // turns N while the rest of the board holds. the token is a quarter
    // turn off the axis, so the shadow cannot see it unless the shadow was
    // already zero: the fourth class is the time axis showing through
    let open = Geonum::new_with_angle(1.0, pole + Angle::new(1.0, 2.0));
    for i in 0..=n {
        for j in 0..=n {
            let cell = disjunctive_sum(&disjunctive_sum(&int(i), &neg(&int(j))), &star());
            let residual = Geonum::new_with_angle(i as f64, pole)
                + Geonum::new_with_angle(j as f64, pole.dual())
                + open;
            assert_eq!(
                outcome(&cell),
                read(&residual, pole),
                "cell ({i}, {j}) with an open move reads the same shadow"
            );
        }
    }
}

// ------------------------------------------------------------ leaving the board

#[test]
fn it_reads_the_thermograph_as_a_light_cone() {
    // a hot game is a switch {a | b}, a > b: whoever moves first takes the
    // better number. conway's thermograph draws it in a (value, temperature)
    // plane — a left wall at a − τ, a right wall at b + τ, slopes ∓1,
    // meeting at the freezing temperature (a − b)/2 above the mean
    // (a + b)/2, then a vertical mast. the outcome table cannot see any of
    // this: {3 | 1} files with the number 2 as class L, yet {3 | 1} − 2 is a
    // first-player win — the hot game is confused with its own mean, and
    // temperature is the width the class sort dropped
    let switch = game(vec![int(3)], vec![int(1)]);
    assert_eq!(outcome(&switch), Outcome::L, "{{3 | 1}} sorts as L");
    assert_eq!(outcome(&int(2)), Outcome::L, "so does its mean, 2");
    assert_eq!(
        outcome(&disjunctive_sum(&switch, &neg(&int(2)))),
        Outcome::N,
        "{{3 | 1}} − 2 is a first-player win — confused with its mean"
    );

    // read the drawing as spacetime: put the mean on the time axis and the
    // temperature on the space axis. the left and right stops are then the
    // event's null coordinates, t + x and t − x — the two walls are the
    // light cone, and the stops are where it crosses
    let (a, b) = (3.0_f64, 1.0_f64);
    let (mean, temp) = ((a + b) / 2.0, (a - b) / 2.0);
    let event = Geonum::new_from_cartesian(temp, mean); // (x, t) = (temperature, mean)
    let (left_stop, right_stop) = stops(&event);
    assert!(
        drawn(left_stop).near(&drawn(a)),
        "the left stop is the forward null coordinate t + x"
    );
    assert!(
        drawn(right_stop).near(&drawn(b)),
        "the right stop is the backward null coordinate t − x"
    );

    // cooling by τ taxes each move: the stops walk in to a − τ and b + τ.
    // on the cone that is the two measurements travelling toward each
    // other along the null walls — at light speed, the one speed
    // information has — the temperature shrinking, the mean holding
    let tau = 0.4_f64;
    let cooled = Geonum::new_from_cartesian(temp - tau, mean);
    let (cl, cr) = stops(&cooled);
    assert!(drawn(cl).near(&drawn(a - tau)), "left wall: a − τ");
    assert!(drawn(cr).near(&drawn(b + tau)), "right wall: b + τ");

    // frozen at τ = (a − b)/2 the walls meet: both measurements have
    // arrived, both stops read the mean, the game is a number, and the
    // event sits on the time axis — the mast is a worldline at rest.
    // perfect information is this closed cone, not an axiom. conway's
    // figure is this one with the axes exchanged
    let frozen = Geonum::new_from_cartesian(0.0, mean);
    assert!(
        frozen.angle.near(&Angle::new(1.0, 2.0)),
        "the frozen game sits on the time axis"
    );
    let (fl, fr) = stops(&frozen);
    assert!(drawn(fl).near(&drawn(mean)), "left stop at the mean");
    assert!(drawn(fr).near(&drawn(mean)), "right stop at the mean");
}

#[test]
fn it_settles_to_the_additive_null_and_timestamps_it_with_winding() {
    // the settlement: matched magnitudes at dual angles cancel in one
    // addition — conservation is dual-pair cancellation, the double entry
    // written as geometry. this is the light cone of spacetime_test::
    // its_lightlike, a quantity against its own dual, [r, θ] + [r, θ+π] = 0
    let size = 2.0_f64.sqrt(); // an irrational size — no tick, no lattice
    let buyer = Geonum::new_with_angle(size, Angle::new(2.0, 7.0)); // blade 0
    let seller = Geonum::new_with_angle(size, buyer.angle.dual()); // blade 2
    let settled = buyer + seller;
    assert!(
        settled.near_mag(0.0),
        "a matched trade settles to the additive null"
    );

    // the vanished magnitude still carries when: the null's angle is the
    // pair's summed winding — blade 0 from the buyer, blade 2 from the
    // seller, blade 2 on the record
    assert_eq!(
        settled.angle.blade(),
        buyer.angle.blade() + seller.angle.blade(),
        "the settlement timestamps itself with the pair's summed blade"
    );

    // the pair on the time axis — the item ahead of the buyer at π/2,
    // behind the seller at 3π/2 — settles at blade 4, grade 0: future plus
    // past lands a zero-extent record on the present, the flat crest
    let ahead = Geonum::new_with_angle(size, Angle::new(1.0, 2.0)); // blade 1
    let behind = Geonum::new_with_angle(size, ahead.angle.dual()); // blade 3
    let present = ahead + behind;
    assert!(present.near_mag(0.0), "future and past meet at zero extent");
    assert_eq!(present.angle.blade(), 4, "blade 1 + blade 3 on the record");
    assert_eq!(
        present.angle.grade(),
        0,
        "and the record sits at the present"
    );

    // the same pair trades again after a day of motion — each party a full
    // turn along its history. the new settlement lands at the same place
    // with a later winding: space is the position, time is the count
    // (atomic_clock_test: timekeeping is winding counting)
    let day = Angle::new(4.0, 2.0); // one full turn, four quarter turns
    let buyer_later = buyer.rotate(day);
    let seller_later = Geonum::new_with_angle(size, buyer_later.angle.dual());
    let settled_later = buyer_later + seller_later;
    assert!(
        settled_later.near_mag(0.0),
        "the later trade settles the same"
    );
    assert_eq!(
        settled_later.angle.base_angle(),
        settled.angle.base_angle(),
        "same place on the board"
    );
    assert!(
        settled.angle < settled_later.angle,
        "a later tick on the clock — the records order by winding"
    );
}

// ------------------------------------------------ what humans can see

#[test]
fn it_optimizes_by_taking_the_smallest_angle_that_has_reached_the_reader() {
    // a reader at home at NOW, wanting the best flat white. three cafes have
    // posted prices at three moments, each through an api with its own
    // latency, and one runs privately and publishes nothing. what the
    // reader can choose from is what has arrived: the interval from the
    // post, delayed by the wire, to the reader is not spacelike, and the
    // seller is openly operated
    let home = latlng(40.7484, -73.9857);
    let now = 60.0_f64;
    let cost = 1.71_f64;
    let mut cafes = vec![
        Seller {
            name: "north",
            price: 4.50,
            place: latlng(40.7580, -73.9855), // 1.1 km
            posted: 0.0,                      // a minute ago
            wire: 0.050,                      // 50 ms api latency
            open: true,
        },
        Seller {
            name: "corner",
            price: 3.80,
            place: latlng(40.7527, -73.9772), // 0.9 km
            posted: 59.995,                   // 5 ms ago
            wire: 0.050,
            open: true,
        },
        Seller {
            name: "south",
            price: 3.20,
            place: latlng(40.7359, -73.9911), // 1.5 km
            posted: 30.0,                     // half a minute ago
            wire: 0.050,
            open: false, // private: publishes nothing
        },
    ];

    // prices are directions: the cost is a price's shadow on the cost axis
    // and the markup its rejection, so cheaper is a smaller angle and the
    // reader's minimum is a minimum over angles. Angle orders by blade then t
    let (north, corner, south) = (
        price_direction(cost, 4.50).angle,
        price_direction(cost, 3.80).angle,
        price_direction(cost, 3.20).angle,
    );
    assert!(
        south < corner && corner < north,
        "cheaper is a smaller angle"
    );

    // light crosses the mile in microseconds. the wire is the cone the
    // reader lives in: 50 ms of api latency is ten thousand times the light
    // time, and it is the latency that sets what is visible
    let light_time = light_seconds(separation(home, cafes[1].place));
    assert!(
        cafes[1].wire > 1e3 * light_time,
        "the api wire is over a thousand light-times long"
    );
    eprintln!(
        "  light to the corner cafe: {:.1} µs; api latency: {:.0} ms",
        light_time * 1e6,
        cafes[1].wire * 1e3
    );

    // at NOW the option set is one direction: north, posted a minute ago.
    // corner posted 5 ms ago and its wire needs 50, south is dark. the
    // best move is the minimum over what arrived, $4.50
    let best_now = best(&cafes, home, now, cost).expect("something has arrived");
    assert_eq!(best_now.name, "north", "only north has reached the reader");
    assert!(
        Geonum::scalar(best_now.price).near_mag(4.50),
        "the best visible price is $4.50"
    );
    assert_eq!(visible(&cafes, home, now).len(), 1, "one direction at NOW");

    // the gap between the best that arrived and the best that exists is the
    // spatial arbitrage of the moment: a smaller angle the reader cannot
    // see yet. as directions, south is about 10° inside north
    let (existing_name, existing_price) = {
        let s = cafes
            .iter()
            .min_by_key(|s| price_direction(cost, s.price).angle)
            .expect("three cafes");
        (s.name, s.price)
    };
    assert_eq!(existing_name, "south");
    let gap =
        price_direction(cost, best_now.price).angle - price_direction(cost, existing_price).angle;
    assert!(
        gap.near_rad((cost / 4.50_f64).acos() - (cost / 3.20_f64).acos()),
        "the arbitrage gap is the angle between the two price directions"
    );

    // 60 ms later the corner cafe's record has arrived and the minimum drops
    // to $3.80. nothing about the prices changed; the cone grew
    let best_later = best(&cafes, home, now + 0.060, cost).expect("two arrived");
    assert_eq!(best_later.name, "corner", "corner arrives with its wire");
    assert!(
        price_direction(cost, best_later.price).angle < price_direction(cost, best_now.price).angle,
        "the minimum angle fell as a record arrived"
    );

    // south switches to openly operated. its record posted half a minute
    // ago and is already inside the cone; it was the flag that hid it. the
    // minimum drops to $3.20 and the arbitrage gap closes to zero
    cafes[2].open = true;
    let best_open = best(&cafes, home, now + 0.060, cost).expect("all three");
    assert_eq!(
        best_open.name, "south",
        "the private seller becomes a direction"
    );
    assert!(
        Geonum::scalar(best_open.price).near_mag(3.20),
        "the reader now sees the smallest angle that exists"
    );
    let closed =
        price_direction(cost, best_open.price).angle - price_direction(cost, existing_price).angle;
    assert!(
        closed.near_rad(0.0),
        "no gap between what arrived and what exists"
    );

    // the option set, counted. with south switched on, NOW already holds
    // two directions — south was inside the cone the whole time, the flag
    // hid it — and 60 ms later the corner record makes three. the board a
    // reader plays on is the number of directions in their cone, and the
    // game is the minimum over it
    assert_eq!(
        visible(&cafes, home, now).len(),
        2,
        "south was always in the cone"
    );
    assert_eq!(
        visible(&cafes, home, now + 0.060).len(),
        3,
        "corner arrives with its wire"
    );
    for at in [now, now + 0.060] {
        let names: Vec<&str> = visible(&cafes, home, at).iter().map(|s| s.name).collect();
        eprintln!("  t = {at:.3}: visible {names:?}");
    }
}

#[test]
fn it_seats_the_trade_at_a_saddle_point() {
    // a saddle is a stationary point whose curvatures split: a minimum along
    // one direction, a maximum along the direction a quarter turn from it.
    // the interval surface is one — z = x² − t² is the space square against
    // its dual, its zero set the light cone, its stationary point the apex,
    // the flat present. spacetime_test walks it. here the saddle is the seat
    // in a zero-sum game the saddle is minimax: the row player's best
    // minimum meets the column player's best maximum and neither can
    // improve alone. the price game after the surface flattened: the
    // seller asks, the buyer bids, a sale clears at the ask when the bid
    // covers it, and the bids run from the visible minimum up since no
    // seller sells below it. payoff to the seller, rows ask, columns bid
    let asks = [3.20_f64, 3.80, 4.50];
    let bids = [3.20_f64, 3.80, 4.50];
    let payoff = |ask: f64, bid: f64| if bid >= ask { ask } else { 0.0 };

    // maximin: the seller's best worst case. minimax: the buyer's best
    // worst case. they meet at $3.20 — the seat
    let maximin = asks
        .iter()
        .map(|&a| {
            bids.iter()
                .map(|&b| payoff(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let minimax = bids
        .iter()
        .map(|&b| {
            asks.iter()
                .map(|&a| payoff(a, b))
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        (Geonum::scalar(maximin) + Geonum::scalar(minimax).dual()).near_mag(0.0),
        "maximin against minimax is the null: the saddle exists"
    );
    assert!(
        Geonum::scalar(maximin).near_mag(3.20),
        "and its value is the visible minimum"
    );

    // the stationary condition is ∂V/∂ownership = 0: at the seat the
    // seller's take and the buyer's payment are the dual pair cancelling
    let seat = Geonum::scalar(3.20) + Geonum::scalar(3.20).dual();
    assert!(seat.near_mag(0.0), "the seat is the settled pair");

    // the curvatures at the seat. along the seller's axis, raising the ask
    // against the settled bid loses the sale: the payoff falls from 3.20 to
    // 0, a residual on the dual — a maximum. along the buyer's axis,
    // raising the bid against the settled ask changes nothing: the sale
    // still clears at the ask, the residual is zero — flat. once the
    // surface has flattened the saddle has one curved axis left, the
    // seller's, and the buyer's axis lies level along the cut
    let sellers_move =
        Geonum::scalar(payoff(3.80, 3.20)) + Geonum::scalar(payoff(3.20, 3.20)).dual();
    assert!(
        sellers_move.near_mag(3.20),
        "the seller's move costs the whole sale"
    );
    assert_eq!(
        sellers_move.angle.grade(),
        2,
        "a fall: the seller sits at a maximum"
    );
    let buyers_move =
        Geonum::scalar(payoff(3.20, 3.80)) + Geonum::scalar(payoff(3.20, 3.20)).dual();
    assert!(
        buyers_move.near_mag(0.0),
        "the buyer's move changes nothing: their axis is flat"
    );
}

#[test]
fn it_draws_the_present_flow_with_one_select() {
    // systemaccounting's transaction is the pair as a record — bivector
    // json, two poles and one magnitude:
    //
    //   {
    //     "item": "bottled water",
    //     "price": "1.000",           // measured by seller − measured by buyer = 0
    //     "quantity": "1",
    //     "creditor": "GroceryStore", // seller (producer) — the dual pole
    //     "debitor": "JacobWebb",     // buyer (consumer) — the pole
    //     "creditor_approval_time": "2023-03-20T04:58:27.771Z",
    //     "debitor_approval_time": "2023-03-20T04:58:32.001Z"
    //   }
    //
    // read natively: price × quantity is the magnitude, the debitor names
    // the pole and the creditor its dual — the other fixed pole of the same
    // trade — and the transaction time, the later approval, when the pair
    // closes, is the winding. the account names label π/2 turns; the poles
    // below carry none. the present flow of value is
    //
    //   SELECT SUM(price*quantity) FROM transactions WHERE transaction_time=NOW();
    //
    // a ledger of five closed rows, three at the present winding
    let now = 3;
    let ledger = [
        pair(1.000 * 1.0, Angle::new(2.0, 7.0), 3), // bottled water
        pair(2.50 * 2.0, Angle::new(3.0, 5.0), 3),  // bread
        pair(20.0 * 3.0, Angle::new(4.0, 9.0), 3),  // flour
        pair(900.0 * 1.0, Angle::new(1.0, 5.0), 2), // rent, a turn ago
        pair(4.0 * 1.0, Angle::new(2.0, 7.0), 4),   // coffee, a turn from now
    ];

    // WHERE transaction_time = NOW(): the flat present as a predicate, a
    // cut one winding thick, read off the blade
    let present: Vec<&[Geonum; 2]> = ledger
        .iter()
        .filter(|row| row[0].angle.blade() / 4 == now)
        .collect();
    assert_eq!(present.len(), 3, "three rows on the present cut");

    // SUM(price*quantity): total_magnitude over the cut, r per row
    let flow: GeoCollection = present.iter().map(|row| row[0]).collect();
    assert!(
        drawn(flow.total_magnitude()).near(&drawn(1.0 + 5.0 + 60.0)),
        "the present flow is 66"
    );

    // the other sum on the same cut is the null: every pair on it settled,
    // so ∂V/∂ownership = 0 for the whole present at once — the flow is the
    // magnitude of what cancels at the crest
    let both_legs: GeoCollection = present.iter().flat_map(|row| row.iter().copied()).collect();
    assert!(
        both_legs.wave_sum().near_mag(0.0),
        "the cut's vector sum is the null — the ledger conserves"
    );
    // other windings are other boards: each has its own SELECT
    let select = |at: usize| -> f64 {
        ledger
            .iter()
            .filter(|row| row[0].angle.blade() / 4 == at)
            .map(|row| row[0].mag)
            .sum()
    };
    assert!(
        drawn(select(now - 1)).near(&drawn(900.0)),
        "yesterday: rent"
    );
    assert!(drawn(select(now + 1)).near(&drawn(4.0)), "tomorrow: coffee");

    // perfect information is a closed cone, not an axiom. a guitar lesson
    // with the teacher's measurement landed and the student's in flight is
    // one leg with no dual on the cut: the WHERE clause has no row to see,
    // and what the network has delivered is a residual on the seller's ray,
    // supply waiting for its demand. wire the student in and the pair
    // closes, cancels, and joins the present board — "incomplete
    // information" was the wire, not the game
    let lesson_pole = Angle::new(3.0, 5.0);
    let teacher_measured = pair(40.0, lesson_pole, now)[1]; // the credit leg alone
    let delivered: GeoCollection = both_legs
        .iter()
        .copied()
        .chain([teacher_measured])
        .collect();
    let residual = delivered.wave_sum();
    assert!(
        residual.near_mag(40.0),
        "the open row leaves its one measurement as the residual"
    );
    assert_eq!(
        residual.angle.grade(),
        lesson_pole.dual().grade(),
        "on the seller's ray — supply waiting for the demand side to arrive"
    );
    let [student_measured, teacher_again] = pair(40.0, lesson_pole, now);
    assert!(
        (student_measured + teacher_again).near_mag(0.0),
        "wired, the lesson settles like every pair"
    );
    let wired: GeoCollection = flow.iter().copied().chain([student_measured]).collect();
    assert!(
        drawn(wired.total_magnitude()).near(&drawn(66.0 + 40.0)),
        "and the SELECT at NOW sees it: 106"
    );
}

// ------------------------------------------------------------------ the foil
//
// conway's construction, verbatim: a game is nothing but its two option
// sets — all possible movement projected before any movement happens. the
// engine below is the flat spacetimer's demand, run for real: boards as
// data, plays as recursion, answers by exhaustive search

const LEFT: bool = true;
const RIGHT: bool = false;

struct Game {
    left: Vec<Game>,
    right: Vec<Game>,
}

fn game(left: Vec<Game>, right: Vec<Game>) -> Game {
    Game { left, right }
}

// day 0: no options for either seat — the settled board
fn zero() -> Game {
    game(vec![], vec![])
}

// the integer ladder n = {n−1 | }, one square per unit of value
fn int(n: usize) -> Game {
    (0..n).fold(zero(), |g, _| game(vec![g], vec![]))
}

// star = {0|0}: one open move either seat may take
fn star() -> Game {
    game(vec![zero()], vec![zero()])
}

// negation is the role swap recursed over the whole tree — dual() spelled
// out as structural recursion
fn neg(g: &Game) -> Game {
    Game {
        left: g.right.iter().map(neg).collect(),
        right: g.left.iter().map(neg).collect(),
    }
}

// the disjunctive sum: move in either component while the other rides
fn disjunctive_sum(g: &Game, h: &Game) -> Game {
    Game {
        left: g
            .left
            .iter()
            .map(|gl| disjunctive_sum(gl, h))
            .chain(h.left.iter().map(|hl| disjunctive_sum(g, hl)))
            .collect(),
        right: g
            .right
            .iter()
            .map(|gr| disjunctive_sum(gr, h))
            .chain(h.right.iter().map(|hr| disjunctive_sum(g, hr)))
            .collect(),
    }
}

// normal play by exhaustive search: the mover wins when some option lands
// the opponent, moving first, on a lost board. the mover argument every
// question on the board grows
fn wins_moving_first(g: &Game, mover: bool) -> bool {
    let options = if mover { &g.left } else { &g.right };
    options.iter().any(|o| !wins_moving_first(o, !mover))
}

// conway's G = 0: neither starter wins
fn second_player_wins(g: &Game) -> bool {
    !wins_moving_first(g, LEFT) && !wins_moving_first(g, RIGHT)
}

// the count of boards the projection lays out — every position in the tree
fn positions(g: &Game) -> usize {
    1 + g
        .left
        .iter()
        .chain(g.right.iter())
        .map(positions)
        .sum::<usize>()
}

// the same count in closed form: a size-n trade's tree visits position
// (i, j) once per lattice path to it, C(i+j, i) times, for i, j ≤ n
fn pascal_stack(n: usize) -> usize {
    let binomial = |top: usize, k: usize| (1..=k).fold(1usize, |acc, i| acc * (top - k + i) / i);
    (0..=n)
        .flat_map(|i| (0..=n).map(move |j| binomial(i + j, i)))
        .sum()
}

// the four outcome classes, sorted by who wins under each starting order
#[derive(Debug, PartialEq)]
enum Outcome {
    L,
    R,
    P,
    N,
}

fn outcome(g: &Game) -> Outcome {
    match (wins_moving_first(g, LEFT), wins_moving_first(g, RIGHT)) {
        (false, false) => Outcome::P,
        (true, true) => Outcome::N,
        (true, false) => Outcome::L,
        (false, true) => Outcome::R,
    }
}

// --------------------------------------------------------------- the drawing

// the drawn height as a geonum: sign as a position on the π ray
fn drawn(height: f64) -> Geonum {
    Geonum::scalar(height)
}

// a game's two stops read off an event as its null coordinates: the signed
// projections onto the forward null (π/4) and the backward null (3π/4),
// scaled back from the unit nulls — t + x and t − x
fn stops(event: &Geonum) -> (f64, f64) {
    let root_two = 2.0_f64.sqrt();
    let forward = Angle::new(1.0, 4.0);
    let backward = Angle::new(3.0, 4.0);
    (
        event.mag * event.angle.project(forward) * root_two,
        event.mag * event.angle.project(backward) * root_two,
    )
}

// a cell of the board read off a residual: its shadow on the pair's axis
// carries the sign as grade 0 (L) or grade 2 (R); no residual is settled
// (P); a residual the axis cannot see is a position in motion (N)
fn read(residual: &Geonum, axis: Angle) -> Outcome {
    let shadow = residual.project_to_angle(axis);
    if residual.mag < EPSILON {
        Outcome::P
    } else if shadow.mag < EPSILON {
        Outcome::N
    } else if shadow.angle.grade() == 0 {
        Outcome::L
    } else {
        Outcome::R
    }
}

// a row of the ledger as geometry: price × quantity at the debitor's pole,
// wound to the transaction time, and the same magnitude at the dual pole
// for the creditor. the turn count reads back as blade / 4
fn pair(size: f64, pole: Angle, turn: usize) -> [Geonum; 2] {
    let winding = Angle::new_with_blade(4 * turn, 0.0, 1.0);
    let debit = Geonum::new_with_angle(size, pole + winding);
    [debit, debit.dual()]
}

// ----------------------------------------------------------------- sellers
// a posted price: who, how much, where, when, through what wire, and
// whether the business publishes at all
struct Seller {
    name: &'static str,
    price: f64,
    place: (Angle, Angle),
    posted: f64,
    wire: f64,
    open: bool,
}

// the sellers whose price has reached a reader by a moment: openly
// operated, and the interval from the post, delayed by the wire, to the
// reader is timelike or null
fn visible(sellers: &[Seller], reader: (Angle, Angle), at: f64) -> Vec<&Seller> {
    sellers
        .iter()
        .filter(|s| {
            if !s.open {
                return false;
            }
            let dt = at - s.posted - s.wire;
            if dt < 0.0 {
                return false;
            }
            let between = interval(light_seconds(separation(s.place, reader)), dt);
            between.mag < EPSILON || between.angle.grade() == 2
        })
        .collect()
}

// the best move: the smallest price angle in the reader's cone
fn best(sellers: &[Seller], reader: (Angle, Angle), at: f64, cost: f64) -> Option<&Seller> {
    visible(sellers, reader, at)
        .into_iter()
        .min_by_key(|s| price_direction(cost, s.price).angle)
}

// a price as a direction: the cost is its shadow on the cost axis and the
// markup its rejection, so the angle has cosine cost/price and tangent ε
fn price_direction(cost: f64, price: f64) -> Geonum {
    let markup = (price * price - cost * cost).max(0.0).sqrt();
    Geonum::new_from_cartesian(cost, markup)
}

// ------------------------------------------------------------- the geometry

// a latlng is two angles: degrees are π fractions with divisor 180
fn latlng(lat_deg: f64, lng_deg: f64) -> (Angle, Angle) {
    (Angle::new(lat_deg, 180.0), Angle::new(lng_deg, 180.0))
}

// the unit chain of a place, rational in the stored angles
fn unit((lat, lng): (Angle, Angle)) -> [f64; 3] {
    let (cos_lat, sin_lat) = lat.cos_sin();
    let (cos_lng, sin_lng) = lng.cos_sin();
    [cos_lat * cos_lng, cos_lat * sin_lng, sin_lat]
}

// the great-circle angle between two places, read from their chains
fn separation(a: (Angle, Angle), b: (Angle, Angle)) -> Angle {
    let (u, v) = (unit(a), unit(b));
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = (1.0 - dot * dot).max(0.0).sqrt();
    Angle::new_from_cartesian(dot, cross)
}

// arc length on earth in light-seconds
fn light_seconds(sep: Angle) -> f64 {
    EARTH_LS * sep.grade_angle()
}

// the interval between two events with c = 1: the space square at grade 0
// against the time square on the dual (spacetime_test)
fn interval(space_ls: f64, seconds: f64) -> Geonum {
    Geonum::new(space_ls, 0.0, 1.0).pow(2.0) + Geonum::new(seconds, 1.0, 2.0).pow(2.0)
}
