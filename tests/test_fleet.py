import numpy as np

from fleet_balance.city import make_city
from fleet_balance.demand import destination_probs, origin_rates
from fleet_balance.rebalance import (apply_moves, cap_moves, greedy_moves,
                                     hungarian_moves, min_cost_moves, move_cost, targets_from_demand)
from fleet_balance.sim import Config, run

city = make_city(8)
dist = city.distance_km()


def test_origin_rates_sum():
    assert abs(origin_rates(city, 5000).sum() - 5000) < 1e-6


def test_destination_rows_sum_to_one():
    for h in (3, 8, 13, 18):
        assert np.allclose(destination_probs(city, h).sum(axis=1), 1)


def test_targets_sum_to_fleet():
    t = targets_from_demand(np.random.default_rng(1).random(64), 500)
    assert t.sum() == 500 and (t >= 0).all()


def test_lp_reaches_target_and_beats_greedy():
    rng = np.random.default_rng(3)
    target = targets_from_demand(rng.random(64), 400)
    stock = targets_from_demand(rng.random(64), 400)
    lp = min_cost_moves(stock, target, dist)
    gr = greedy_moves(stock, target, dist)
    assert (apply_moves(stock, lp) == target).all()
    assert (apply_moves(stock, gr) == target).all()
    assert move_cost(lp, dist, 1)[1] <= move_cost(gr, dist, 1)[1] + 1e-9


def test_cap_moves_respects_budget():
    rng = np.random.default_rng(4)
    target = targets_from_demand(rng.random(64), 400)
    stock = targets_from_demand(rng.random(64), 400)
    capped = cap_moves(min_cost_moves(stock, target, dist), dist, 20)
    assert sum(k for _, _, k in capped) <= 20


def test_fleet_is_conserved():
    cfg = Config(fleet=300, trips_per_day=1500, days=2, policy="hourly_reactive")
    r = run(city, cfg)
    # available at the start of the hour, minus those leaving on trips, plus those on the road = fleet
    served = (r.demand - r.lost).sum(axis=1)
    assert (r.stock.sum(axis=1) - served + r.in_use == 300).all()
    assert (r.lost <= r.demand).all()


def test_rebalancing_helps_when_fleet_is_short():
    base = Config(fleet=250, trips_per_day=2500, days=3)
    none = run(city, Config(**{**base.__dict__, "policy": "none"}))
    react = run(city, Config(**{**base.__dict__, "policy": "hourly_reactive"}))
    assert react.served_share > none.served_share


def test_same_seed_same_result():
    cfg = Config(fleet=300, trips_per_day=1500, days=2)
    assert run(city, cfg).lost_trips == run(city, cfg).lost_trips


def test_hungarian_matches_lp_cost():
    rng = np.random.default_rng(9)
    for _ in range(5):
        target = targets_from_demand(rng.random(64) + 0.1, 300)
        stock = targets_from_demand(rng.random(64) + 0.1, 300)
        h = hungarian_moves(stock, target, dist)
        lp = min_cost_moves(stock, target, dist)
        assert (apply_moves(stock, h) == target).all()
        assert abs(move_cost(h, dist, 1)[1] - move_cost(lp, dist, 1)[1]) < 1e-6


def test_all_policies_face_identical_demand():
    base = Config(fleet=300, trips_per_day=1500, days=2)
    runs = [run(city, Config(**{**base.__dict__, "policy": p})) for p in ("none", "night_reset", "hourly_reactive")]
    assert (runs[0].demand == runs[1].demand).all() and (runs[0].demand == runs[2].demand).all()


def test_moves_are_counted_and_priced():
    cfg = Config(fleet=300, trips_per_day=1500, days=2, policy="hourly_reactive", cost_per_vehicle_km=2.0)
    r = run(city, cfg)
    assert r.moved_vehicles > 0 and r.move_cost > 0
    none = run(city, Config(**{**cfg.__dict__, "policy": "none"}))
    assert none.moved_vehicles == 0 and none.move_cost == 0
