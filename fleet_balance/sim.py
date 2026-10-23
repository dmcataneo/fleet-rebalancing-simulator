"""Hourly simulation of a shared fleet.

Each hour: (1) the policy may move vehicles, (2) trips are drawn per zone, (3)
a trip is served only if a vehicle is in that zone, otherwise it is lost
demand, (4) served vehicles arrive at their destination the next hour.

Simplifications: every trip lasts one hour, lost riders do not walk to a
neighbouring zone, battery and repairs are ignored, demand does not react to
availability. All of these affect real results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .city import City
from .demand import destination_probs, origin_rates
from .rebalance import (apply_moves, cap_moves, greedy_moves, hungarian_moves,
                        min_cost_moves, move_cost, targets_from_demand)

MATCHERS = {"lp": min_cost_moves, "hungarian": hungarian_moves, "greedy": greedy_moves}

POLICIES = ["none", "night_reset", "hourly_reactive"]


@dataclass
class Config:
    fleet: int = 1500
    trips_per_day: float = 6000
    days: int = 5
    weekend: bool = False
    policy: str = "none"
    hourly_budget: int = 60        # vehicles movable per hour (reactive policy)
    night_budget: int = 600        # vehicles movable at night reset
    lookahead_hours: int = 3
    cost_per_vehicle_km: float = 0.9  # fictional currency units
    matcher: str = "lp"  # lp, hungarian or greedy
    seed: int = 7


@dataclass
class Result:
    demand: np.ndarray            # (steps, zones) requested trips
    lost: np.ndarray              # (steps, zones) unserved trips
    stock: np.ndarray             # (steps, zones) vehicles available at start of hour, after arrivals and moves, before trips
    in_use: np.ndarray = None     # (steps,) vehicles on a trip at the end of the hour (leave this hour, arrive next)
    moved_vehicles: int = 0
    move_cost: float = 0.0
    log: list = field(default_factory=list)

    @property
    def served_share(self) -> float:
        d = self.demand.sum()
        return float(1 - self.lost.sum() / d) if d else 1.0

    @property
    def lost_trips(self) -> float:
        return float(self.lost.sum())


def initial_stock(city: City, fleet: int, rates: np.ndarray) -> np.ndarray:
    # start the first morning stocked like the residential areas (people park at home)
    t = targets_from_demand(city.residential, fleet)
    return t


def run(city: City, cfg: Config) -> Result:
    # Demand is drawn up front from its own stream, so every policy faces
    # exactly the same trip requests for a given seed. Only destinations of
    # served trips depend on what the policy did.
    rates = origin_rates(city, cfg.trips_per_day, cfg.weekend)  # (24, Z)
    dist = city.distance_km()
    Z = city.zones
    probs = [destination_probs(city, h, cfg.weekend) for h in range(24)]
    stock = initial_stock(city, cfg.fleet, rates)
    in_transit = np.zeros(Z, dtype=int)
    steps = 24 * cfg.days
    demand_log = np.zeros((steps, Z))
    lost_log = np.zeros((steps, Z))
    stock_log = np.zeros((steps, Z))
    demand_all = np.random.default_rng([cfg.seed, 0]).poisson(
        np.tile(rates, (cfg.days, 1)))
    in_use_log = np.zeros(steps, dtype=int)
    res = Result(demand_log, lost_log, stock_log, in_use_log)
    matcher = MATCHERS[cfg.matcher]

    for step in range(steps):
        h = step % 24
        stock = stock + in_transit
        in_transit = np.zeros(Z, dtype=int)

        moves = []
        if cfg.policy == "night_reset" and h == 4:
            target = targets_from_demand(_window(rates, 5, 24), int(stock.sum()))
            moves = cap_moves(matcher(stock, target, dist), dist, cfg.night_budget)
        elif cfg.policy == "hourly_reactive":
            target = targets_from_demand(_window(rates, h, cfg.lookahead_hours), int(stock.sum()))
            moves = cap_moves(matcher(stock, target, dist), dist, cfg.hourly_budget)
        if moves:
            n, c = move_cost(moves, dist, cfg.cost_per_vehicle_km)
            res.moved_vehicles += n
            res.move_cost += c
            stock = apply_moves(stock, moves)
        stock_log[step] = stock

        d = demand_all[step]
        served = np.minimum(d, stock)
        demand_log[step], lost_log[step] = d, d - served
        stock = stock - served
        dest_rng = np.random.default_rng([cfg.seed, 1, step])
        for o in np.flatnonzero(served):
            dest = dest_rng.choice(Z, size=served[o], p=probs[h][o])
            np.add.at(in_transit, dest, 1)
        in_use_log[step] = in_transit.sum()
        assert stock.sum() + in_transit.sum() == cfg.fleet, "vehicle count changed"
        
    return res


def _window(rates: np.ndarray, start: int, hours: int) -> np.ndarray:
    idx = [(start + k) % 24 for k in range(hours)]
    return rates[idx].sum(axis=0)


def compare(city: City, cfg: Config, seeds: int = 5) -> dict[str, dict[str, float]]:
    """Run every policy on the same seeds and average. Returns metrics per policy."""
    out = {}
    for pol in POLICIES:
        served, lost, cost, moved = [], [], [], []
        for s in range(seeds):
            c = Config(**{**cfg.__dict__, "policy": pol, "seed": cfg.seed + s})
            r = run(city, c)
            served.append(r.served_share)
            lost.append(r.lost_trips / cfg.days)
            cost.append(r.move_cost / cfg.days)
            moved.append(r.moved_vehicles / cfg.days)
        out[pol] = {
            "served_share": float(np.mean(served)),
            "lost_trips_per_day": float(np.mean(lost)),
            "moved_vehicles_per_day": float(np.mean(moved)),
            "move_cost_per_day": float(np.mean(cost)),
            "served_share_sd": float(np.std(served)),
        }
    return out


def cost_per_extra_trip(metrics: dict[str, dict[str, float]]) -> dict[str, float | None]:
    """Extra move cost per additional trip served, relative to doing nothing."""
    base = metrics["none"]
    out = {}
    for pol, m in metrics.items():
        extra = base["lost_trips_per_day"] - m["lost_trips_per_day"]
        out[pol] = None if pol == "none" or extra <= 0 else m["move_cost_per_day"] / extra
    return out


def hourly_profile(res: Result, days: int) -> dict[str, np.ndarray]:
    """Average over days: demand, lost trips and stock per hour of day, summed over zones."""
    d = res.demand.sum(axis=1).reshape(days, 24).mean(axis=0)
    l = res.lost.sum(axis=1).reshape(days, 24).mean(axis=0)
    s = res.stock.sum(axis=1).reshape(days, 24).mean(axis=0)
    return {"demand": d, "lost": l, "stock": s}
