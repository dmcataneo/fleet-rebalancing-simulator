"""A fictional city: an N x N grid of zones with three kinds of activity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ZONE_KM = 0.6  # distance between neighbouring zone centres, in km (fictional)


@dataclass
class City:
    n: int
    residential: np.ndarray  # weight per zone, flattened (n*n)
    business: np.ndarray
    leisure: np.ndarray

    @property
    def zones(self) -> int:
        return self.n * self.n

    def coords(self) -> np.ndarray:
        idx = np.arange(self.zones)
        return np.column_stack([idx // self.n, idx % self.n])

    def distance_km(self) -> np.ndarray:
        """Manhattan distance between all zone pairs, in km."""
        c = self.coords()
        d = np.abs(c[:, None, :] - c[None, :, :]).sum(axis=2)
        return d * ZONE_KM


def make_city(n: int = 12) -> City:
    """Business district near the middle-north, a waterfront park along the
    south edge, housing spread around with a denser ring. All made up."""
    r, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    def blob(cr, cc, s):
        return np.exp(-((r - cr) ** 2 + (c - cc) ** 2) / (2 * s**2))

    business = blob(n * 0.3, n * 0.5, n * 0.12)
    leisure = blob(n - 1, n * 0.45, n * 0.14) + 0.4 * blob(n * 0.5, n * 0.1, n * 0.08)
    res = 0.25 + blob(n * 0.55, n * 0.25, n * 0.2) + blob(n * 0.6, n * 0.8, n * 0.2)
    res = res * (1 - 0.7 * business)
    norm = lambda x: (x / x.sum()).ravel()
    return City(n, norm(res), norm(business), norm(leisure))
