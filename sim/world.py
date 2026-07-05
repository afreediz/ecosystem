"""World generation: continuous noise fields, hydrology, biomes (§8 of v1.md).

Generated once at startup from the master seed. Stores spatial-only fields; the
time-varying temperature offsets are added at runtime by ``environment`` (§9).

Arrays are indexed [y, x]. ``world_to_cell`` clamps continuous entity positions to
valid cell indices.
"""
from __future__ import annotations

import numpy as np
from opensimplex import OpenSimplex

from config import WorldConfig
from sim import hydrology

# biome ids
OCEAN, BEACH, MOUNTAIN, COLD, DESERT, FOREST, PLAINS = range(7)
BIOME_NAMES = {
    OCEAN: "ocean", BEACH: "beach", MOUNTAIN: "mountain", COLD: "cold",
    DESERT: "desert", FOREST: "forest", PLAINS: "plains",
}
# RGB colors for the renderer (sim never uses these for logic)
BIOME_COLORS = {
    OCEAN: (28, 52, 120),
    BEACH: (214, 196, 140),
    MOUNTAIN: (120, 120, 128),
    COLD: (228, 236, 244),
    DESERT: (214, 192, 120),
    FOREST: (40, 110, 56),
    PLAINS: (104, 158, 72),
}


def _fractal_noise(gen: OpenSimplex, w: int, h: int, scale: float, octaves: int) -> np.ndarray:
    """Layered OpenSimplex noise normalized to [0,1]."""
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    field = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    freq = scale
    total_amp = 0.0
    for _ in range(octaves):
        # OpenSimplex.noise2 is scalar; use the vectorized grid helper for speed
        layer = gen.noise2array(xs * freq, ys * freq)
        field += amp * layer
        total_amp += amp
        amp *= 0.5
        freq *= 2.0
    field /= max(total_amp, 1e-9)
    field = (field - field.min()) / (np.ptp(field) + 1e-9) # normalises to [0, 1]
    return field.astype(np.float32)


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.w = cfg.width
        self.h = cfg.height

        # World generation depends ONLY on the world seed, so the same world seed always
        # yields the same terrain + rivers regardless of the run/determinism seed. The noise
        # generators take the seed directly; hydrology (which needs randomness for river
        # sources) gets a generator derived solely from the world seed -- never the shared
        # run RNG, which would otherwise make the map change between runs of the same world.
        elev_gen = OpenSimplex(seed=cfg.seed)
        moist_gen = OpenSimplex(seed=cfg.seed + 9973)
        world_rng = np.random.default_rng(cfg.seed)

        # --- continuous fields ---
        self.elevation = _fractal_noise(elev_gen, self.w, self.h, cfg.noise_scale,
                                        cfg.noise_octaves)
        moisture = _fractal_noise(moist_gen, self.w, self.h, cfg.moisture_scale,
                                  max(2, cfg.noise_octaves - 1))

        # --- hydrology (ocean / rivers / lakes) ---
        hydro = hydrology.generate(self.elevation, cfg, world_rng)
        self.ocean = hydro["ocean"]
        self.river = hydro["river"]
        self.lake = hydro["lake"]
        self.beach = hydro["beach"]
        self.freshwater = hydro["freshwater"]
        self.water_any = hydro["water_any"]

        # moisture boosted near freshwater, then re-clamped
        self.moisture = np.clip(moisture + 0.5 * hydro["moisture_boost"], 0.0, 1.0)

        # --- static temperature: latitude gradient minus lapse rate * elevation ---
        lat = np.linspace(1.0, 0.0, self.h, dtype=np.float32)[:, None]  # top edge warm
        base = np.broadcast_to(lat, (self.h, self.w)).astype(np.float32)
        temp = base - self.elevation * cfg.lapse_rate
        self.static_temp = ((temp - temp.min()) / (np.ptp(temp) + 1e-9)).astype(np.float32)

        # --- biome labels (render + base nutrient/plant suitability) ---
        self.biome = self._classify_biomes()

        # --- soil nutrients: per-cell pool [0,1]; richer in lowland/moist land ---
        nut = 0.4 + 0.4 * self.moisture - 0.3 * self.elevation
        nut[self.water_any] = 0.0
        self.nutrients = np.clip(nut, 0.0, 1.0).astype(np.float32)

        # plant suitability multiplier per biome (carrying capacity factor)
        self.plant_suitability = self._plant_suitability()

        # cover: dense FOREST conceals prey from predators (a spatial refuge). This is the
        # mechanism that stabilizes predator-prey coexistence -- without a refuge, foxes
        # drive sheep to local extinction and then starve. Sheep in cover cannot be seen or
        # caught by foxes; they must leave for the richer open grassland (plains) to forage,
        # where they become vulnerable. NOTE: bare, rocky MOUNTAIN is deliberately NOT cover.
        # When it was, forest+mountain made ~40% of the map an untouchable prey reservoir:
        # foxes could never crop enough sheep, the predator starved to extinction, and the
        # prey then exploded. Forest-only refuge (~30% of land) still protects prey from
        # total collapse while leaving foxes enough huntable range to persist (see v1.md §18).
        self.cover = (self.biome == FOREST)

        # passability: open water (ocean + rivers/lakes) and very high mountain block
        # movement. Animals can't walk across water; they drink from an adjacent cell
        # (consumption.py allows drinking when within eat range of freshwater).
        self.passable = ~self.water_any & (self.elevation <= 0.97)

        # nearest-freshwater fields for perception (direction + distance to drink)
        (self.fw_dist, self.fw_nearest_x, self.fw_nearest_y) = self._nearest_source_fields(self.freshwater)

        # nearest-cover fields: lets the sleep system steer animals toward the closest safe
        # spot (forest cover) to bed down at night without a per-agent search each tick.
        (self.cover_dist, self.cover_nearest_x, self.cover_nearest_y) = self._nearest_source_fields(self.cover)

    # ------------------------------------------------------------------ biomes
    def _classify_biomes(self) -> np.ndarray:
        c = self.cfg
        b = np.full((self.h, self.w), PLAINS, dtype=np.int8)
        elev, moist, temp = self.elevation, self.moisture, self.static_temp
        # priority order from §8.3
        b[(elev > c.mountain_threshold) & ~self.water_any] = MOUNTAIN
        cold = (temp < c.cold_threshold) & ~self.water_any & (elev <= c.mountain_threshold)
        b[cold] = COLD
        desert = ((moist < c.desert_moisture) & (temp > c.warm_threshold)
                  & ~self.water_any & (elev <= c.mountain_threshold) & ~cold)
        b[desert] = DESERT
        forest = ((moist > c.forest_moisture) & ~self.water_any
                  & (elev <= c.mountain_threshold) & ~cold & ~desert)
        b[forest] = FOREST
        b[self.beach] = BEACH
        b[self.ocean] = OCEAN
        return b

    def _plant_suitability(self) -> np.ndarray:
        s = np.zeros((self.h, self.w), dtype=np.float32)
        s[self.biome == PLAINS] = 1.0
        s[self.biome == FOREST] = 0.85
        s[self.biome == BEACH] = 0.2
        s[self.biome == DESERT] = 0.15
        s[self.biome == COLD] = 0.25
        s[self.biome == MOUNTAIN] = 0.1
        s[self.water_any] = 0.0
        return s

    # ------------------------------------------------------------------ nearest-source fields
    def _nearest_source_fields(self, source: np.ndarray):
        """Multi-source BFS over a boolean ``source`` mask.

        Returns ``(dist, nearest_x, nearest_y)``: for every cell, the distance (in cell
        units; inf where no source is reachable) to the closest source cell and that
        cell's center coordinates. Used for both freshwater (drinking) and cover (sleep).
        """
        from collections import deque
        h, w = self.h, self.w
        INF = np.float32(np.inf)
        dist = np.full((h, w), INF, dtype=np.float32)
        nx_arr = np.full((h, w), -1, dtype=np.int32)
        ny_arr = np.full((h, w), -1, dtype=np.int32)
        dq = deque()
        ys, xs = np.nonzero(source)
        for y, x in zip(ys.tolist(), xs.tolist()):
            dist[y, x] = 0.0
            nx_arr[y, x] = x
            ny_arr[y, x] = y
            dq.append((y, x))
        nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while dq:
            y, x = dq.popleft()
            base = dist[y, x]
            for dy, dx in nb8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    step = 1.41421356 if (dy and dx) else 1.0
                    nd = base + step
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        nx_arr[ny, nx] = nx_arr[y, x]
                        ny_arr[ny, nx] = ny_arr[y, x]
                        dq.append((ny, nx))
        # store nearest as float cell-center coords for direction math
        return dist, (nx_arr.astype(np.float32) + 0.5), (ny_arr.astype(np.float32) + 0.5)

    # ------------------------------------------------------------------ sampling
    def world_to_cell(self, x, y):
        cx = np.clip(np.asarray(x, dtype=np.intp), 0, self.w - 1)
        cy = np.clip(np.asarray(y, dtype=np.intp), 0, self.h - 1)
        return cx, cy

    def sample(self, field: np.ndarray, x, y):
        cx, cy = self.world_to_cell(x, y)
        return field[cy, cx]

    def is_freshwater(self, x, y) -> np.ndarray:
        cx, cy = self.world_to_cell(x, y)
        return self.freshwater[cy, cx]

    def is_passable(self, x, y) -> np.ndarray:
        cx, cy = self.world_to_cell(x, y)
        return self.passable[cy, cx]

    def in_cover(self, x, y) -> np.ndarray:
        """True where prey is concealed from predators (forest / mountain refuge)."""
        cx, cy = self.world_to_cell(x, y)
        return self.cover[cy, cx]

    def random_land_positions(self, n: int, rng: np.random.Generator,
                              near_freshwater: bool = False) -> np.ndarray:
        """Draw ``n`` random passable land positions (optionally biased near water)."""
        if near_freshwater:
            # prefer cells with high moisture boost (proxy for water proximity)
            land = self.passable & ~self.water_any
            weight = (self.moisture * land).ravel()
        else:
            land = self.passable & ~self.water_any
            weight = land.ravel().astype(np.float64)
        weight = weight / weight.sum()
        flat = rng.choice(self.w * self.h, size=n, p=weight)
        ys, xs = np.divmod(flat, self.w)
        jitter = rng.uniform(0.1, 0.9, size=(n, 2))
        pos = np.stack([xs + jitter[:, 0], ys + jitter[:, 1]], axis=1).astype(np.float32)
        return pos

    def clustered_land_positions(self, n: int, rng: np.random.Generator,
                                 n_clusters: int, spread: float,
                                 near_freshwater: bool = False) -> np.ndarray:
        """Spawn ``n`` individuals in ``n_clusters`` tight groups (herds / packs).

        Animals live in groups, not scattered uniformly; clustered starts bootstrap
        mate-finding (a lone disperser can't breed) and create persistent breeding demes.
        Members land on passable, non-water land near their cluster centre.
        """
        n_clusters = max(1, min(n_clusters, n))
        centers = self.random_land_positions(n_clusters, rng, near_freshwater)
        out = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            c = centers[i % n_clusters]
            for _ in range(20):                      # rejection-sample a valid spot
                p = c + rng.normal(0.0, spread, size=2).astype(np.float32)
                p[0] = np.clip(p[0], 0, self.w - 1e-3)
                p[1] = np.clip(p[1], 0, self.h - 1e-3)
                if self.is_passable(p[0], p[1]) and not self.water_any[int(p[1]), int(p[0])]:
                    break
            out[i] = p
        return out
