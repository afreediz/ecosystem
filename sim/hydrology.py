"""Ocean / river / lake generation on the elevation field (§8.2 of v1.md).

Pure NumPy + standard-library BFS (no scipy dependency). Produces boolean maps:
``ocean``, ``river``, ``lake``, ``beach``, plus derived ``freshwater`` and ``water_any``,
and a freshwater-distance field used to boost moisture.

Coordinate convention used throughout: arrays are indexed [y, x] (row, col).
"""
from __future__ import annotations

from collections import deque
import numpy as np

from config import WorldConfig

# 8-neighborhood offsets (dy, dx)
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _ocean_floodfill(elevation: np.ndarray, sea_level: float) -> np.ndarray:
    """Cells below sea level connected to the map border are ocean (BFS from edges)."""
    h, w = elevation.shape
    below = elevation < sea_level
    ocean = np.zeros((h, w), dtype=bool)
    dq = deque()
    # seed from all border cells that are below sea level
    for x in range(w):
        for y in (0, h - 1):
            if below[y, x] and not ocean[y, x]:
                ocean[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if below[y, x] and not ocean[y, x]:
                ocean[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in _NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and below[ny, nx] and not ocean[ny, nx]:
                ocean[ny, nx] = True
                dq.append((ny, nx))
    return ocean


def _lowest_neighbor(elevation: np.ndarray, y: int, x: int):
    h, w = elevation.shape
    best = None
    best_e = np.inf
    for dy, dx in _NB8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and elevation[ny, nx] < best_e:
            best_e = elevation[ny, nx]
            best = (ny, nx)
    return best, best_e


def _carve_rivers(elevation: np.ndarray, ocean: np.ndarray, cfg: WorldConfig,
                  rng: np.random.Generator):
    """Carve rivers downhill from high sources to the sea; pool into lakes at minima."""
    h, w = elevation.shape
    river = np.zeros((h, w), dtype=bool)
    lake = np.zeros((h, w), dtype=bool)

    # pick sources among high, non-ocean cells
    land = ~ocean
    land_idx = np.argwhere(land & (elevation > np.quantile(elevation[land], 0.75)))
    if land_idx.shape[0] == 0:
        return river, lake
    n_sources = min(cfg.n_river_sources, land_idx.shape[0])
    chosen = rng.choice(land_idx.shape[0], size=n_sources, replace=False)

    for ci in chosen:
        y, x = int(land_idx[ci, 0]), int(land_idx[ci, 1])
        steps = 0
        max_steps = h * w  # safety bound
        while steps < max_steps:
            steps += 1
            river[y, x] = True
            if ocean[y, x]:
                break
            (ny, nx), ne = _lowest_neighbor(elevation, y, x)
            if (ny, nx) == (y, x):
                break
            if ne >= elevation[y, x]:
                # local minimum -> form a small lake by flooding up to a spill level
                spill = elevation[y, x] + 0.015
                basin = _floodfill_basin(elevation, y, x, spill, ocean)
                lake |= basin
                # try to find a spill cell on the basin rim lower than current
                spill_cell = _find_spill(elevation, basin, ocean)
                if spill_cell is None:
                    break
                y, x = spill_cell
            else:
                if river[ny, nx]:        # merged into an existing river
                    y, x = ny, nx
                    river[y, x] = True
                    if ocean[y, x]:
                        break
                    # continue a few more downhill steps from the junction
                y, x = ny, nx
    return river, lake


def _floodfill_basin(elevation: np.ndarray, sy: int, sx: int, spill: float,
                     ocean: np.ndarray) -> np.ndarray:
    """Flood cells reachable from (sy,sx) with elevation <= spill (BFS)."""
    h, w = elevation.shape
    basin = np.zeros((h, w), dtype=bool)
    if ocean[sy, sx]:
        return basin
    dq = deque([(sy, sx)])
    basin[sy, sx] = True
    count = 0
    while dq and count < 400:                # cap lake size for v1
        y, x = dq.popleft()
        count += 1
        for dy, dx in _NB8:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and not basin[ny, nx]
                    and not ocean[ny, nx] and elevation[ny, nx] <= spill):
                basin[ny, nx] = True
                dq.append((ny, nx))
    return basin


def _find_spill(elevation: np.ndarray, basin: np.ndarray, ocean: np.ndarray):
    """Lowest cell on the rim adjacent to the basin (the overflow point)."""
    h, w = elevation.shape
    best = None
    best_e = np.inf
    ys, xs = np.nonzero(basin)
    for y, x in zip(ys.tolist(), xs.tolist()):
        for dy, dx in _NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not basin[ny, nx]:
                if elevation[ny, nx] < best_e:
                    best_e = elevation[ny, nx]
                    best = (ny, nx)
    return best


def _distance_to(mask: np.ndarray, max_dist: float) -> np.ndarray:
    """Approx euclidean distance to nearest True cell via multi-source BFS (capped)."""
    h, w = mask.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    dq = deque()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0.0
        dq.append((y, x))
    cap = float(max_dist)
    while dq:
        y, x = dq.popleft()
        if dist[y, x] >= cap:
            continue
        for dy, dx in _NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                step = 1.41421356 if (dy and dx) else 1.0
                nd = dist[y, x] + step
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    dq.append((ny, nx))
    return dist


def generate(elevation: np.ndarray, cfg: WorldConfig, rng: np.random.Generator) -> dict:
    """Build all hydrology maps. Returns a dict of boolean arrays + the moisture boost."""
    sea_level = cfg.sea_level_threshold
    ocean = _ocean_floodfill(elevation, sea_level)
    river, lake = _carve_rivers(elevation, ocean, cfg, rng)
    river &= ~ocean
    lake &= ~ocean

    # beach: land cells within 1 step of ocean
    h, w = elevation.shape
    beach = np.zeros((h, w), dtype=bool)
    oy, ox = np.nonzero(ocean)
    for y, x in zip(oy.tolist(), ox.tolist()):
        for dy, dx in _NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not ocean[ny, nx]:
                beach[ny, nx] = True

    freshwater = (river | lake) & ~ocean
    water_any = freshwater | ocean

    # moisture boost from freshwater with distance falloff
    fw_dist = _distance_to(freshwater, cfg.moisture_boost_radius)
    boost = np.clip(1.0 - fw_dist / cfg.moisture_boost_radius, 0.0, 1.0).astype(np.float32)
    boost[np.isinf(fw_dist)] = 0.0

    return {
        "ocean": ocean,
        "river": river,
        "lake": lake,
        "beach": beach,
        "freshwater": freshwater,
        "water_any": water_any,
        "moisture_boost": boost,
    }
