"""Demand model: where trips start and end, hour by hour.

Origin intensity in zone z at hour h is a mix of three activity types whose
weights change through the day (commute out in the morning, commute back in the
evening, leisure at midday and weekends). Destinations follow a gravity rule:
more attractive and closer zones get more trips. The shapes are assumptions for
illustration, not measurements.
"""
from __future__ import annotations

import numpy as np

from .city import City


def _bump(h, centre, width):
    return np.exp(-((h - centre) ** 2) / (2 * width**2))


def hourly_profiles(weekend: bool = False) -> dict[str, np.ndarray]:
    h = np.arange(24)
    if weekend:
        res = 0.4 * _bump(h, 11, 3) + 0.2 * _bump(h, 17, 3)
        biz = 0.05 * _bump(h, 12, 4)
        leis = 1.0 * _bump(h, 14, 3.5) + 0.3 * _bump(h, 19, 2)
    else:
        res = 1.0 * _bump(h, 8, 1.3) + 0.25 * _bump(h, 13, 2) + 0.2 * _bump(h, 20, 2)
        biz = 0.15 * _bump(h, 12.5, 1.5) + 1.0 * _bump(h, 17.5, 1.3)
        leis = 0.25 * _bump(h, 13, 2.5) + 0.45 * _bump(h, 19, 2)
    return {"res": res, "biz": biz, "leis": leis}


def origin_rates(city: City, total_trips: float, weekend: bool = False) -> np.ndarray:
    """Expected trip starts, shape (24, zones), summing to total_trips."""
    p = hourly_profiles(weekend)
    rate = (
        p["res"][:, None] * city.residential[None, :]
        + p["biz"][:, None] * city.business[None, :]
        + p["leis"][:, None] * city.leisure[None, :]
    )
    return rate / rate.sum() * total_trips


def destination_probs(city: City, hour: int, weekend: bool = False, distance_scale_km: float = 2.0) -> np.ndarray:
    """Row-stochastic matrix P[o, d] for trips starting at `hour`."""
    h = hour
    p = hourly_profiles(weekend)
    # people head where the activity of that period is: mornings to work,
    # evenings home, midday leisure. Shares are assumptions.
    morning = 6 <= h < 11
    evening = 16 <= h < 22
    if weekend:
        attract = 0.7 * city.leisure + 0.3 * city.residential
    elif morning:
        attract = 0.75 * city.business + 0.15 * city.leisure + 0.10 * city.residential
    elif evening:
        attract = 0.65 * city.residential + 0.25 * city.leisure + 0.10 * city.business
    else:
        attract = 0.4 * city.leisure + 0.3 * city.business + 0.3 * city.residential
    dist = city.distance_km()
    w = attract[None, :] * np.exp(-dist / distance_scale_km)
    np.fill_diagonal(w, w.diagonal() * 0.2)  # few trips end where they started
    return w / w.sum(axis=1, keepdims=True)
