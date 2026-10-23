"""Rebalancing policies.

A policy looks at current stock per zone and returns moves (from_zone,
to_zone, count). It never sees future random demand, only the expected demand.

Theory used here
----------------
Given surplus zones (stock above target) and deficit zones (stock below
target), moving vehicles at least total distance is a transportation problem
(Hitchcock, 1941), a minimum-cost flow that can be solved exactly as a linear
program. Real operators also have to route trucks and respect their capacity,
which is much harder (Raviv, Tzur and Forma, 2013; Schuijbroek, Hampshire and
van Hoeve, 2017). This project ignores truck routing and charges a flat cost
per vehicle-km moved. That is a simplification, and the cost figures are
fictional.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def targets_from_demand(expected_demand: np.ndarray, fleet: int) -> np.ndarray:
    """Integer stock targets proportional to expected demand, summing to fleet.

    Proportional to demand is a simple base-stock rule. It is not optimal: zones
    with volatile demand deserve a larger buffer (the newsvendor logic), but it
    is a clear baseline.
    """
    w = expected_demand + 1e-9
    raw = w / w.sum() * fleet
    t = np.floor(raw).astype(int)
    short = fleet - t.sum()
    order = np.argsort(-(raw - t))
    t[order[:short]] += 1
    return t


def min_cost_moves(stock: np.ndarray, target: np.ndarray, dist_km: np.ndarray) -> list[tuple[int, int, int]]:
    """Exact minimum-distance moves that turn `stock` into `target` (LP)."""
    surplus = np.maximum(stock - target, 0)
    deficit = np.maximum(target - stock, 0)
    src, dst = np.flatnonzero(surplus), np.flatnonzero(deficit)
    if len(src) == 0 or len(dst) == 0:
        return []
    total = min(surplus.sum(), deficit.sum())
    cost = dist_km[np.ix_(src, dst)].ravel()
    ns, nd = len(src), len(dst)
    a_ub, b_ub = [], []
    for i in range(ns):
        row = np.zeros(ns * nd)
        row[i * nd:(i + 1) * nd] = 1
        a_ub.append(row)
        b_ub.append(surplus[src[i]])
    for j in range(nd):
        row = np.zeros(ns * nd)
        row[j::nd] = 1
        a_ub.append(row)
        b_ub.append(deficit[dst[j]])
    a_eq, b_eq = np.ones((1, ns * nd)), [total]
    res = linprog(cost, A_ub=np.array(a_ub), b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        return greedy_moves(stock, target, dist_km)
    flow = np.rint(res.x).reshape(ns, nd).astype(int)
    return [(int(src[i]), int(dst[j]), int(flow[i, j]))
            for i in range(ns) for j in range(nd) if flow[i, j] > 0]


def greedy_moves(stock: np.ndarray, target: np.ndarray, dist_km: np.ndarray) -> list[tuple[int, int, int]]:
    """Nearest-first matching. Fast, not optimal. Kept as a comparison."""
    surplus = np.maximum(stock - target, 0).astype(int)
    deficit = np.maximum(target - stock, 0).astype(int)
    pairs = [(dist_km[i, j], i, j) for i in np.flatnonzero(surplus) for j in np.flatnonzero(deficit)]
    pairs.sort()
    moves = []
    for _, i, j in pairs:
        k = min(surplus[i], deficit[j])
        if k > 0:
            moves.append((int(i), int(j), int(k)))
            surplus[i] -= k
            deficit[j] -= k
    return moves


def cap_moves(moves, dist_km, max_vehicles: int):
    """Keep the shortest moves first until the vehicle budget is used."""
    out, used = [], 0
    for i, j, k in sorted(moves, key=lambda m: dist_km[m[0], m[1]]):
        take = min(k, max_vehicles - used)
        if take <= 0:
            break
        out.append((i, j, take))
        used += take
    return out


def apply_moves(stock: np.ndarray, moves) -> np.ndarray:
    s = stock.copy()
    for i, j, k in moves:
        k = min(k, s[i])
        s[i] -= k
        s[j] += k
    return s


def move_cost(moves, dist_km, cost_per_vehicle_km: float) -> tuple[int, float]:
    n = sum(k for _, _, k in moves)
    km = sum(k * dist_km[i, j] for i, j, k in moves)
    return n, km * cost_per_vehicle_km


def hungarian_moves(stock: np.ndarray, target: np.ndarray, dist_km: np.ndarray) -> list[tuple[int, int, int]]:
    """Unit-level assignment: each surplus vehicle is assigned to one empty slot.

    The Hungarian method solves the one-to-one assignment problem. Our problem
    has zones holding many vehicles, so we first expand every zone into single
    vehicles and slots. The optimum cost equals the transportation LP, but the
    expanded matrix grows with the number of vehicles to move, so it only makes
    sense when moves are few or when the real question is one-to-one (a specific
    vehicle, driver or courier to a specific task).
    Uses scipy's linear_sum_assignment, a modern implementation of this problem.
    """
    from scipy.optimize import linear_sum_assignment

    surplus = np.maximum(stock - target, 0).astype(int)
    deficit = np.maximum(target - stock, 0).astype(int)
    n = int(min(surplus.sum(), deficit.sum()))
    if n == 0:
        return []
    src = np.repeat(np.arange(len(stock)), surplus)
    dst = np.repeat(np.arange(len(stock)), deficit)
    cost = dist_km[np.ix_(src, dst)]
    r, c = linear_sum_assignment(cost)
    # more units than needed on one side: keep the n cheapest pairs
    pairs = sorted(zip(r, c), key=lambda rc: cost[rc])[:n]
    flow: dict[tuple[int, int], int] = {}
    for i, j in pairs:
        key = (int(src[i]), int(dst[j]))
        flow[key] = flow.get(key, 0) + 1
    return [(a, b, k) for (a, b), k in flow.items()]
