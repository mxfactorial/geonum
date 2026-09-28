use geonum::*;

// a transaction is a pair of events in spacetime
//
// systemaccounting records a trade as one item with two independent
// measurements of its price, each stamped by who measured and when.
// extend the record with where — debitor_latlng, creditor_latlng — and
// the row is two events: the seller's measurement at (creditor_latlng,
// creditor_approval_time) and the buyer's at (debitor_latlng,
// debitor_approval_time)
//
// a latlng is already two angles on a sphere, so nothing is converted:
// the angles are stored, their components come back rationally, and the
// map is the stereographic projection of the sphere onto the plane — the
// map point of a place is a geonum whose magnitude is the stored
// half-tangent of the colatitude and whose angle is the longitude
//
// with two events per row the interval between them has a grade. c·Δt
// against the great-circle separation: timelike and the second
// measurement sat inside the first one's cone, spacelike and the two
// measurements could not have informed each other — independence as
// geometry rather than a word in the schema. between rows the same grade
// sorts spatial arbitrage, a price difference light has not yet carried,
// from temporal arbitrage, a difference only production closes
//
// and the present a reader can know is their past cone. no cells, no
// NOW() equality: a row is visible from a place at a time when the
// interval from the row's event to the reader is timelike or null. the
// flow of value a reader can sum is the total magnitude of the rows the
// cone has delivered. as every seat receives the cut, the price surface
// over it flattens: every reader's minimum meets, every seller falls to
// it, and what production leaves behind is the winding
//
// distances are in light-seconds so c = 1 and every interval is an
// addition of two squares, [d, 0]² + [Δt, π/2]², the time square landing
// on the dual (spacetime_test). the cgt suites read the human layers over
// this — the board, the SELECT — and minkowski_space_test reads the
// scalar layer under it
//
// run: cargo test --test economic_spacetime_test -- --show-output

// earth's radius in light-seconds: 6 371 000 m over 299 792 458 m/s
const EARTH_LS: f64 = 6_371_000.0 / 299_792_458.0;

#[test]
fn it_stores_a_latlng_as_two_angles_and_draws_the_map_by_projection() {
    // GroceryStore at 40.7128 N, 74.0060 W — the JSON names two angles and
    // the constructor stores them as π fractions, degrees over 180
    let (lat, lng) = latlng(40.7128, -74.0060);
    assert_eq!(
        lat.blade(),
        0,
        "a northern latitude sits in the first quarter"
    );
    assert!(
        lat.near_rad(40.7128_f64.to_radians()),
        "the latitude is the angle the degrees named"
    );

    // the unit components come back rationally from the stored angles: no
    // sin, no cos, the same three numbers the trig route produces
    let [ux, uy, uz] = unit((lat, lng));
    let (phi, lam) = (40.7128_f64.to_radians(), (-74.0060_f64).to_radians());
    assert!(
        Geonum::scalar(ux).near_mag((phi.cos() * lam.cos()).abs()),
        "x component"
    );
    assert!(
        Geonum::scalar(uy - phi.cos() * lam.sin()).near_mag(0.0),
        "y component, sign and all"
    );
    assert!(Geonum::scalar(uz).near_mag(phi.sin()), "z component");

    // the map: project from the south pole onto the equatorial plane. a
    // place at colatitude θ lands at radius tan(θ/2) — which is the stored
    // t of the colatitude angle — at azimuth equal to its longitude, since
    // the projection is conformal and the longitude passes through
    let colatitude = Angle::new(1.0, 2.0) - lat;
    let map_point = Geonum::new_with_angle(colatitude.t(), lng);
    let classic_radius = ((90.0_f64 - 40.7128).to_radians() / 2.0).tan();
    assert!(
        map_point.near_mag(classic_radius),
        "the map radius is the colatitude's half-tangent, stored, not computed"
    );
    assert!(
        map_point.angle.near(&lng),
        "the map azimuth is the longitude itself"
    );
    let (cos, sin) = map_point.angle.cos_sin();
    assert!(
        Geonum::scalar(map_point.mag * cos - classic_radius * lam.cos()).near_mag(0.0),
        "map x agrees with tan(θ/2)·cos λ"
    );
    assert!(
        Geonum::scalar(map_point.mag * sin - classic_radius * lam.sin()).near_mag(0.0),
        "map y agrees with tan(θ/2)·sin λ"
    );

    // the great-circle separation between two places is one angle read
    // from the two unit chains, and it agrees with the haversine formula
    // to within its own tolerance
    let buyer = latlng(40.7580, -73.9855); // JacobWebb, uptown
    let sep = separation((lat, lng), buyer);
    assert!(
        Geonum::scalar(light_seconds(sep))
            .near_mag(haversine_ls((40.7128, -74.0060), (40.7580, -73.9855))),
        "the separation matches haversine"
    );
    eprintln!(
        "  GroceryStore -> JacobWebb: {:.3} km, {:.2e} light-seconds",
        light_seconds(sep) * 299_792.458,
        light_seconds(sep)
    );
}

#[test]
fn it_places_each_approval_as_an_event_and_grades_the_interval_between_them() {
    // the row as given, extended with where each party measured
    let row = TransactionItem {
        item: "bottled water",
        price: 1.000,
        quantity: 1.0,
        creditor: "GroceryStore",
        debitor: "JacobWebb",
        creditor_latlng: (40.7128, -74.0060),
        debitor_latlng: (40.7580, -73.9855),
        creditor_approval_time: 27.771, // 2023-03-20T04:58:27.771Z, seconds past the minute
        debitor_approval_time: 32.001,  // 2023-03-20T04:58:32.001Z
    };

    // the seller's measurement and the buyer's are two events. their
    // separation in space is the great-circle angle scaled by the radius,
    // in light-seconds; their separation in time is 4.230 seconds
    let (seller, buyer) = row.events();
    let d = light_seconds(separation(seller.place, buyer.place));
    let dt = buyer.seconds - seller.seconds;
    assert!(
        Geonum::scalar(dt).near_mag(4.230),
        "4.230 s between the measurements"
    );

    // the interval between the two measurements is timelike, grade 2: light
    // crosses the city in microseconds and had 4.23 seconds. the buyer's
    // measurement sat inside the seller's cone, the price could have
    // travelled, and the pair closes causally connected
    let between = interval(d, dt);
    assert_eq!(
        between.angle.grade(),
        2,
        "timelike: the buyer measured inside the seller's cone"
    );
    assert!(between.near_mag(dt * dt - d * d), "|s²| = Δt² − d²");

    // the time between them as winding: one second is one turn, so 4.230
    // seconds is 16 quarter turns and a remainder — the record's clock
    let wound = Angle::new(2.0 * dt, 1.0); // 4.23 turns = 8.46 π
    assert_eq!(wound.blade(), 16, "four whole seconds are 16 quarter turns");
    assert!(
        wound.near_rem(0.23 * 2.0 * std::f64::consts::PI),
        "the 230 ms is the remainder within the last quarter"
    );

    // the trade itself is the pair cancelling, price against price, the
    // same additive null whichever event is read first
    let settled =
        Geonum::scalar(row.price * row.quantity) + Geonum::scalar(row.price * row.quantity).dual();
    assert!(
        settled.near_mag(0.0),
        "seller's 1.000 against buyer's 1.000 is zero"
    );
    eprintln!(
        "  {}: {} <- {}, d = {:.2e} ls, Δt = {:.3} s, interval grade {}",
        row.item,
        row.creditor,
        row.debitor,
        d,
        dt,
        between.angle.grade()
    );
}

#[test]
fn it_tells_independent_measurements_from_informed_ones_by_the_grade() {
    // a seller in Tokyo and a buyer in New York, 10 850 km apart: light
    // needs about 36 ms to cross. the grade of the interval between their
    // two measurements says whether either could have seen the other's
    let tokyo = latlng(35.6762, 139.6503);
    let new_york = latlng(40.7128, -74.0060);
    let d = light_seconds(separation(tokyo, new_york));
    assert!(
        Geonum::scalar(d).near_mag(haversine_ls((35.6762, 139.6503), (40.7128, -74.0060))),
        "Tokyo to New York in light-seconds"
    );

    // approvals 10 ms apart: spacelike, grade 0. neither measurement was in
    // the other's cone, so the two prices are independent by geometry, not
    // by declaration
    let independent = interval(d, 0.010);
    assert_eq!(
        independent.angle.grade(),
        0,
        "spacelike: the measurements could not have met"
    );
    assert!(
        independent.near_mag(d * d - 0.010 * 0.010),
        "|s²| = d² − Δt²"
    );

    // approvals exactly one light-crossing apart: null. the seller's price
    // arrives as the buyer measures — the cone's edge
    let edge = interval(d, d);
    assert!(
        edge.near_mag(0.0),
        "null: light arrives as the second measurement is taken"
    );

    // approvals 4.23 s apart: timelike, grade 2. the second measurement had
    // the first available for over four seconds
    let informed = interval(d, 4.230);
    assert_eq!(
        informed.angle.grade(),
        2,
        "timelike: the second measurement could know the first"
    );

    // on earth the independent case is narrow: the widest separation is a
    // half circumference, about 67 ms of light. any two approvals further
    // apart in time than that are connected wherever they were taken
    let antipodal = EARTH_LS * std::f64::consts::PI;
    assert_eq!(
        interval(antipodal, 0.070).angle.grade(),
        2,
        "70 ms connects any two places on earth"
    );
    eprintln!(
        "  Tokyo -> New York: {:.1} ms of light; earth's widest gap: {:.1} ms",
        d * 1e3,
        antipodal * 1e3
    );
}

#[test]
fn it_sorts_arbitrage_by_the_grade_between_two_rows() {
    // the same item sold in two places at two prices. the price difference
    // is a residual, 0.200 on the higher side. whether it can be arbitraged
    // depends on the interval between the two sales
    let spread = Geonum::scalar(1.200) + Geonum::scalar(1.000).dual();
    assert!(spread.near_mag(0.200), "a 0.200 spread");
    assert_eq!(spread.angle.grade(), 0, "on the higher side");

    let d = light_seconds(separation(
        latlng(35.6762, 139.6503),
        latlng(40.7128, -74.0060),
    ));

    // sales 10 ms apart: spacelike. the cheaper price has not reached the
    // dearer market, and the spread is exploitable only by whoever is the
    // wire — spatial arbitrage, which lazily depends on prices not
    // travelling at the speed of light
    assert_eq!(
        interval(d, 0.010).angle.grade(),
        0,
        "spacelike: a spread light has not carried yet"
    );

    // sales one light-crossing apart: null. light has caught up, and with
    // it every seat that reads prices at light speed
    assert!(interval(d, d).near_mag(0.0), "null: light catches up");

    // sales a minute apart: timelike. both markets have had the other's
    // price; what remains of the spread is what production has not
    // collapsed — temporal arbitrage, which depends on production
    // collapsing the difference at the speed of light
    assert_eq!(
        interval(d, 60.0).angle.grade(),
        2,
        "timelike: only production closes what is left"
    );

    // the spread itself does not change with the grade; the grade says who
    // can act on it. the residual is the same 0.200 in every case
    assert!(
        spread.near_mag(0.200),
        "the residual is a fact about the prices"
    );
}

#[test]
fn it_reads_the_present_a_place_can_know_as_its_past_cone() {
    // a reader in London at NOW. three rows settled at three places, at
    // three moments before NOW. which of them the reader can know is not
    // a time cut and not a cell: it is whether the row's event lies inside
    // the reader's past cone — interval timelike or null
    let london = latlng(51.5074, -0.1278);
    let now = 32.001_f64;
    let rows = [
        TransactionItem {
            item: "bottled water",
            price: 1.000,
            quantity: 1.0,
            creditor: "GroceryStore",
            debitor: "JacobWebb",
            creditor_latlng: (40.7128, -74.0060),
            debitor_latlng: (40.7580, -73.9855),
            creditor_approval_time: 27.771,
            debitor_approval_time: 28.001, // closed 4.000 s before NOW
        },
        TransactionItem {
            item: "onigiri",
            price: 1.200,
            quantity: 1.0,
            creditor: "Konbini",
            debitor: "Aiko",
            creditor_latlng: (35.6762, 139.6503),
            debitor_latlng: (35.6895, 139.6917),
            creditor_approval_time: 31.990,
            debitor_approval_time: 31.996, // closed 5 ms before NOW
        },
        TransactionItem {
            item: "flat white",
            price: 3.500,
            quantity: 1.0,
            creditor: "Cafe",
            debitor: "Mia",
            creditor_latlng: (-33.8688, 151.2093),
            debitor_latlng: (-33.8650, 151.2094),
            creditor_approval_time: 31.800,
            debitor_approval_time: 31.901, // closed 100 ms before NOW
        },
    ];

    // visible from London at a given moment: the row closed, and the
    // interval from its closing event to the reader is not spacelike
    let visible = |at: f64| -> Vec<&TransactionItem> {
        rows.iter()
            .filter(|row| {
                let (_, buyer) = row.events();
                let dt = at - buyer.seconds;
                if dt < 0.0 {
                    return false;
                }
                let between = interval(light_seconds(separation(buyer.place, london)), dt);
                between.mag < EPSILON || between.angle.grade() == 2
            })
            .collect()
    };

    // at NOW: New York closed 4 s ago and 18.6 ms of light away, visible.
    // Sydney closed 100 ms ago and 56.7 ms away, visible. Tokyo closed 5 ms
    // ago and 31.9 ms away — the cone has not arrived
    let seen = visible(now);
    assert_eq!(seen.len(), 2, "two of three rows have reached London");
    assert!(
        seen.iter().all(|row| row.item != "onigiri"),
        "Tokyo is still in flight"
    );

    // the flow of value London can sum is the total magnitude of what the
    // cone delivered: 1.000 + 3.500
    let flow: GeoCollection = seen
        .iter()
        .map(|row| Geonum::scalar(row.price * row.quantity))
        .collect();
    assert!(
        Geonum::scalar(flow.total_magnitude()).near_mag(4.500),
        "4.500 delivered"
    );

    // 30 ms later the Tokyo cone has arrived: 35 ms elapsed against 31.9 ms
    // of light. the same predicate now returns three rows and 5.700
    let later = visible(now + 0.030);
    assert_eq!(later.len(), 3, "the third row arrives with light");
    let flow_later: GeoCollection = later
        .iter()
        .map(|row| Geonum::scalar(row.price * row.quantity))
        .collect();
    assert!(
        Geonum::scalar(flow_later.total_magnitude()).near_mag(5.700),
        "5.700 once light has caught up"
    );

    // and a reader in Tokyo at NOW sees its own row at once and London's
    // view a light-crossing later: the present is a place's cone, not a
    // shared hyperplane
    let tokyo = latlng(35.6762, 139.6503);
    let (_, onigiri_buyer) = rows[1].events();
    let at_home = interval(
        light_seconds(separation(onigiri_buyer.place, tokyo)),
        now - onigiri_buyer.seconds,
    );
    assert_eq!(at_home.angle.grade(), 2, "Tokyo knows its own trade at NOW");
    for row in &seen {
        eprintln!(
            "  London at NOW sees: {} {:.3}",
            row.item,
            row.price * row.quantity
        );
    }
}

#[test]
fn it_flattens_the_price_surface_as_every_seat_receives_the_cut_and_leaves_the_winding() {
    // the hypersurface is the present cut: the field of price directions
    // over places at one winding. three sellers posted at three moments,
    // and three readers reach the cut through three different accesses —
    // one wired, one behind a half-second cache, one reading a monthly
    // report. each sees a different piece of the surface, so the minimum
    // differs from seat to seat. that difference is the surface's slope,
    // and it is the whole of aggressive commerce
    let cost = 1.71_f64;
    let now = 60.0_f64;
    let mut sellers = vec![
        Seller {
            name: "north",
            price: 4.50,
            place: latlng(40.7580, -73.9855),
            posted: now - 10.0,
            wire: 0.050,
            open: true,
        },
        Seller {
            name: "corner",
            price: 3.80,
            place: latlng(40.7527, -73.9772),
            posted: now - 0.300,
            wire: 0.050,
            open: true,
        },
        Seller {
            name: "south",
            price: 3.20,
            place: latlng(40.7359, -73.9911),
            posted: now - 0.020,
            wire: 0.050,
            open: true,
        },
    ];
    let readers = [
        Reader {
            name: "wired",
            place: latlng(40.7484, -73.9857),
            access: 0.0,
        },
        Reader {
            name: "cached",
            place: latlng(40.7614, -73.9776),
            access: 0.500,
        },
        Reader {
            name: "monthly",
            place: latlng(40.7061, -74.0087),
            access: 5.0,
        },
    ];

    // the surface has slope: the wired reader's minimum is $3.80, the other
    // two see only north and pay $4.50. south, posted 20 ms ago, has
    // reached no one yet
    let minima: Vec<(&str, f64)> = readers
        .iter()
        .map(|r| {
            let b = best_for(&sellers, r, now, cost).expect("north reached everyone");
            (r.name, b.price)
        })
        .collect();
    assert!(Geonum::scalar(minima[0].1).near_mag(3.80), "wired: corner");
    assert!(Geonum::scalar(minima[1].1).near_mag(4.50), "cached: north");
    assert!(Geonum::scalar(minima[2].1).near_mag(4.50), "monthly: north");
    let slope = price_direction(cost, 4.50).angle - price_direction(cost, 3.80).angle;
    assert!(
        slope.near_rad((cost / 4.50_f64).acos() - (cost / 3.80_f64).acos()),
        "the slope across seats is the angle between their minima"
    );
    for (name, price) in &minima {
        eprintln!("  before access: {name} pays ${price:.2}");
    }

    // everyone receives access: the wires drop to zero and every seat reads
    // the cut itself. the cones coincide, perfect information as a fact
    // about latency, and every reader's minimum is the same $3.20. the
    // slope across seats is zero
    for seller in sellers.iter_mut() {
        seller.wire = 0.0;
    }
    let wired: Vec<Reader> = readers
        .iter()
        .map(|r| Reader {
            name: r.name,
            place: r.place,
            access: 0.0,
        })
        .collect();
    for r in &wired {
        let b = best_for(&sellers, r, now, cost).expect("everyone reached");
        assert_eq!(
            b.name, "south",
            "{}: sees the smallest angle that exists",
            r.name
        );
    }
    let flat = price_direction(cost, 3.20).angle - price_direction(cost, 3.20).angle;
    assert!(flat.near_rad(0.0), "no slope across seats");

    // on a surface everyone can see, a seller above the minimum loses every
    // reader, so every angle falls to the minimum. the field flattens: the
    // residual between any two sellers is the null, every cell of the board
    // is P, and the game is solved — no move changes the outcome
    for seller in sellers.iter_mut() {
        seller.price = 3.20;
    }
    let directions: Vec<Geonum> = sellers
        .iter()
        .map(|s| price_direction(cost, s.price))
        .collect();
    let mut cells = Vec::new();
    for i in 0..directions.len() {
        for j in 0..directions.len() {
            if i != j {
                cells.push(directions[i] - directions[j]);
            }
        }
    }
    let field: GeoCollection = cells.into_iter().collect();
    assert_eq!(field.len(), 6, "six ordered pairs on the board");
    assert!(
        Geonum::scalar(field.total_magnitude()).near_mag(0.0),
        "every cell is P: no residual between any two sellers"
    );
    assert!(
        field.wave_sum().near_mag(0.0),
        "the surface has no slope left"
    );

    // the only slope left is in time. competition takes the angle to the
    // cost axis, and production takes the cost axis down: each tick is a
    // full turn on the record and a scaling of the magnitude, until the
    // price is the null with nothing on either side. the record's blade
    // climbs the whole way
    let tick = Angle::new(4.0, 2.0);
    let mut value = price_direction(cost, cost); // price at cost: on the axis
    assert!(
        value.angle.near(&Angle::new(0.0, 1.0)),
        "the angle has collapsed onto the axis"
    );
    let start = value;
    let efficiencies = [0.8_f64, 0.75, 2.0 / 3.0, 0.5, 0.0];
    for (k, gain) in efficiencies.iter().enumerate() {
        let before = value.mag;
        value = value.rotate(tick).scale(*gain);
        assert!(value.mag <= before, "tick {}: the cost fell", k + 1);
        assert_eq!(
            value.angle.blade(),
            4 * (k + 1),
            "tick {}: the record wound on",
            k + 1
        );
        eprintln!(
            "  tick {}: cost ${:.3}, record blade {}",
            k + 1,
            value.mag,
            value.angle.blade()
        );
    }
    assert!(value.near_mag(0.0), "production reached zero cost");
    assert_eq!(value.angle.blade(), 20, "five ticks on the record");
    assert_eq!(
        value.angle.base_angle(),
        start.angle.base_angle(),
        "same position, the winding is what changed"
    );

    // the trade still happens at zero magnitude: the pair cancels, and the
    // record carries both legs' windings — the robot hands the twenty back
    // as origami and the ledger still counts the turn
    let origami = value + value.dual();
    assert!(origami.near_mag(0.0), "the zero-magnitude pair settles");
    assert_eq!(
        origami.angle.blade(),
        value.angle.blade() + value.dual().angle.blade(),
        "and the record sums the pair's windings"
    );

    // p = mv: momentum is the value at its rate, the quarter turn. at the
    // start its magnitude is the cost; at the end it is zero, and what
    // remains is the winding — the information the scalar could not hold,
    // conserved in the angle. the knowledge-accumulating currency is the
    // record's blade
    assert!(
        start.differentiate().near_mag(cost),
        "momentum at the start: the cost at its rate"
    );
    let momentum = value.differentiate();
    assert!(
        momentum.near_mag(0.0),
        "momentum at the end: zero magnitude"
    );
    assert_eq!(
        momentum.angle.blade(),
        21,
        "and twenty-one quarter turns of record"
    );
}

// ------------------------------------------------------------------ the row

// systemaccounting's transaction item, extended with where each party
// measured. times are seconds past the minute of the example timestamps
struct TransactionItem {
    item: &'static str,
    price: f64,
    quantity: f64,
    creditor: &'static str,
    debitor: &'static str,
    creditor_latlng: (f64, f64),
    debitor_latlng: (f64, f64),
    creditor_approval_time: f64,
    debitor_approval_time: f64,
}

// one measurement: a place on the sphere and a moment
struct Event {
    place: (Angle, Angle),
    seconds: f64,
}

impl TransactionItem {
    // the row as two events, the seller's measurement and the buyer's
    fn events(&self) -> (Event, Event) {
        (
            Event {
                place: latlng(self.creditor_latlng.0, self.creditor_latlng.1),
                seconds: self.creditor_approval_time,
            },
            Event {
                place: latlng(self.debitor_latlng.0, self.debitor_latlng.1),
                seconds: self.debitor_approval_time,
            },
        )
    }
}

// ------------------------------------------------------ sellers and readers
// a reader at a place with an access latency of their own: zero for a
// wired seat, longer for a cache or a report
struct Reader {
    name: &'static str,
    place: (Angle, Angle),
    access: f64,
}

// the sellers whose price has reached a reader through their own access
// on top of the seller's wire
fn visible_to<'a>(sellers: &'a [Seller], reader: &Reader, at: f64) -> Vec<&'a Seller> {
    sellers
        .iter()
        .filter(|s| {
            if !s.open {
                return false;
            }
            let dt = at - s.posted - s.wire - reader.access;
            if dt < 0.0 {
                return false;
            }
            let between = interval(light_seconds(separation(s.place, reader.place)), dt);
            between.mag < EPSILON || between.angle.grade() == 2
        })
        .collect()
}

// the best move for a reader: the smallest price angle in their cone
fn best_for<'a>(sellers: &'a [Seller], reader: &Reader, at: f64, cost: f64) -> Option<&'a Seller> {
    visible_to(sellers, reader, at)
        .into_iter()
        .min_by_key(|s| price_direction(cost, s.price).angle)
}

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

// the great-circle angle between two places, read from their chains as
// one angle: its cosine is the dot, its sine the cross
fn separation(a: (Angle, Angle), b: (Angle, Angle)) -> Angle {
    let (u, v) = (unit(a), unit(b));
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = (1.0 - dot * dot).max(0.0).sqrt();
    Angle::new_from_cartesian(dot, cross)
}

// arc length on earth in light-seconds: the radius times the angle
fn light_seconds(sep: Angle) -> f64 {
    EARTH_LS * sep.grade_angle()
}

// the interval between two events with c = 1: the space square at grade 0
// against the time square on the dual (spacetime_test)
fn interval(space_ls: f64, seconds: f64) -> Geonum {
    Geonum::new(space_ls, 0.0, 1.0).pow(2.0) + Geonum::new(seconds, 1.0, 2.0).pow(2.0)
}

// the scalar foil for the separation: haversine, in light-seconds
fn haversine_ls((lat1, lng1): (f64, f64), (lat2, lng2): (f64, f64)) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let (dp, dl) = ((lat2 - lat1).to_radians(), (lng2 - lng1).to_radians());
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    EARTH_LS * 2.0 * h.sqrt().asin()
}
