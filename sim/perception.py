"""Local, egocentric **grid** perception (§7.2, §12 of v1.md).

Each tick this builds, for every alive agent, a stack of egocentric perception grids
plus a small vector of internal/global scalars. The grids are the agent's *raw* local
view of the world -- a square window of side ``K = 2*R + 1`` cells centred on the agent,
where ``R`` is the largest sensory range across all species. Cells beyond the agent's
OWN heritable ``sensory_range`` (Euclidean, in cells) or outside the world are zeroed, so
each individual still only perceives what its eyes can reach -- the window is just a
fixed, batchable canvas.

This replaces the old "nearest food / nearest threat / nearest mate / nearest water"
feature blocks. Those scalars threw away the spatial layout of the surroundings; the
grids keep it, which is exactly what a future convolutional brain needs (each grid is a
CNN input channel). The current ``RuleBrain`` *decodes* these grids back into the
targets it needs (see ``sim/brain.py``), so the rule logic is unchanged in spirit -- the
decoding that used to live here now lives in the consumer, just as a CNN would consume
the channels directly.

GRID CHANNELS  (obs.grids: (N, N_CHANNELS, K, K), float32):
  0 CH_TERRAIN  biome label, (biome_id + 1) / NUM_BIOMES   (0 = unseen / out of bounds)
  1 CH_WATER    drinkable freshwater present (1/0)
  2 CH_VEG      vegetation density [0,1]
  3 CH_FOOD     edible *entities* present (carnivore: exposed prey; herbivore: none)
  4 CH_THREAT   predator entities present (prey: foxes; predator: none)
  5 CH_MATE     eligible mates present (same species, opposite sex, adult)

SCALARS  (obs.scalars: (N, SCALAR_DIM), float32):
  0 hunger | 1 thirst | 2 energy | 3 health | 4 age/max_age | 5 sex
  6 temperature(own cell) | 7 time_of_day | 8 season
  9 sensory_range (cells; for distance normalization + CNN) | 10 diet (1=carnivore)
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import Config, SHEEP, FOX
from sim import genome as gn

# --- grid channels ---
CH_TERRAIN, CH_WATER, CH_VEG, CH_FOOD, CH_THREAT, CH_MATE = range(6)
N_CHANNELS = 6
NUM_BIOMES = 7                      # see sim/world.py biome ids (OCEAN..PLAINS)

# --- scalar layout ---
(S_HUNGER, S_THIRST, S_ENERGY, S_HEALTH, S_AGE, S_SEX,
 S_TEMP, S_TIME, S_SEASON, S_SENSORY, S_DIET) = range(11)
SCALAR_DIM = 11


class Observation:
    """The batched perception handed to ``Brain.decide``.

    ``grids``   : (N, N_CHANNELS, K, K) float32 -- egocentric channels (CNN-ready).
    ``scalars`` : (N, SCALAR_DIM)       float32 -- internal state + global env.
    ``radius``  : int                            -- the window half-width R (K = 2R+1).
    """
    __slots__ = ("grids", "scalars", "radius")

    def __init__(self, grids: np.ndarray, scalars: np.ndarray, radius: int):
        self.grids = grids
        self.scalars = scalars
        self.radius = radius


class Perception:
    def __init__(self, cfg: Config, world, entities, grid, env):
        self.cfg = cfg
        self.world = world
        self.ent = entities
        self.grid = grid
        self.env = env
        self.veg = None                       # wired in per-tick by Simulation
        self._species_grids = {}              # species_id -> SpatialGrid (rebuilt each tick)

        # window half-width = ceil(largest sensory_range across all species). One fixed K
        # so a single (N, C, K, K) batch covers both species; smaller-eyed individuals just
        # see a masked sub-disc of the same canvas.
        max_sens = max(s.gene_ranges["sensory_range"].hi for s in cfg.species.values())
        self.R = int(np.ceil(max_sens))
        self.K = 2 * self.R + 1

        # egocentric distance-from-centre stencil (K,K)
        offs = np.arange(-self.R, self.R + 1)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")
        self._d_cell = np.sqrt((ox * ox + oy * oy).astype(np.float32))   # dist from centre

        # circular eye masks cached by integer radius r=0..R (m[r] = cells within r). An
        # agent uses the mask for round(its sensory_range), avoiding a per-agent recompute.
        self._mask_cache = np.stack(
            [(self._d_cell <= r).astype(np.float32) for r in range(self.R + 1)])  # (R+1,K,K)

        # zero-padded (border R) static world fields, so each agent's KxK window is a plain
        # slice of the padded array (a sliding-window view; no per-agent index math). Terrain
        # and water never change; vegetation is re-padded each tick.
        biome = world.biome.astype(np.float32)
        terrain = (biome + 1.0) * (1.0 / NUM_BIOMES)
        self._terr_pad = np.pad(terrain, self.R)
        self._water_pad = np.pad(world.freshwater.astype(np.float32), self.R)

        # lazily-grown output buffers (grow to the peak alive count, not max_entities)
        self._grids = None
        self._scalars = None

    # ------------------------------------------------------------------ buffers
    def _ensure_buffers(self, n: int) -> None:
        if self._grids is None or self._grids.shape[0] < n:
            cap = max(n, 1)
            self._grids = np.zeros((cap, N_CHANNELS, self.K, self.K), dtype=np.float32)
            self._scalars = np.zeros((cap, SCALAR_DIM), dtype=np.float32)

    # ------------------------------------------------------------------ public
    def build(self, temp_field: np.ndarray) -> tuple[Observation, np.ndarray]:
        """Return (obs, idx) for the N alive agents in ``idx``."""
        ent = self.ent
        idx = ent.alive_indices()
        n = idx.shape[0]
        self._ensure_buffers(n)
        grids = self._grids
        scalars = self._scalars
        if n == 0:
            return Observation(grids[:0], scalars[:0], self.R), idx

        px = ent.pos_x[idx]
        py = ent.pos_y[idx]
        spec_arr = ent.species[idx]
        sens = gn.gene(ent.genome[idx], "sensory_range")
        max_age = gn.gene(ent.genome[idx], "max_age")

        cx, cy = self.world.world_to_cell(px, py)        # clamped cell indices (n,)
        cx = cx.astype(np.int32)
        cy = cy.astype(np.int32)

        # entity channels are sparse -> zero this tick's slice before scattering into them.
        # field channels (terrain/water/veg) are fully overwritten below, no zeroing needed.
        grids[:n, CH_FOOD:CH_MATE + 1] = 0.0

        self._fill_field_channels(grids, n, cx, cy, sens)
        self._fill_entity_channels(grids, idx, px, py, cx, cy, spec_arr, sens)

        # --- scalars ---
        s = scalars
        s[:n, S_HUNGER] = ent.hunger[idx]
        s[:n, S_THIRST] = ent.thirst[idx]
        s[:n, S_ENERGY] = ent.energy[idx]
        s[:n, S_HEALTH] = ent.health[idx]
        s[:n, S_AGE] = np.clip(ent.age[idx] / np.maximum(max_age, 1e-6), 0.0, 1.0)
        s[:n, S_SEX] = ent.sex[idx].astype(np.float32)
        s[:n, S_TEMP] = temp_field[cy, cx]
        s[:n, S_TIME] = self.env.time_of_day
        s[:n, S_SEASON] = self.env.season
        s[:n, S_SENSORY] = sens
        s[:n, S_DIET] = (spec_arr == FOX).astype(np.float32)   # foxes are carnivores

        return Observation(grids[:n], scalars[:n], self.R), idx

    # ------------------------------------------------------------------ field channels
    def _fill_field_channels(self, grids, n, cx, cy, sens) -> None:
        """Terrain / water / vegetation sampled over each agent's egocentric window.

        Each agent's KxK window is the slice ``pad[cy:cy+K, cx:cx+K]`` of the R-padded world
        field; ``sliding_window_view`` exposes all those windows as one zero-copy array that
        we gather with the agents' cell indices. Off-world cells fall in the zero pad border;
        cells beyond the agent's own sensory range are removed by the cached circular mask.
        """
        K = self.K
        veg_pad = np.pad(self.veg, self.R)                       # veg changes every tick
        masks = self._mask_cache[np.clip(np.rint(sens).astype(np.intp), 0, self.R)]  # (n,K,K)

        terr_win = sliding_window_view(self._terr_pad, (K, K))[cy, cx]   # (n,K,K)
        water_win = sliding_window_view(self._water_pad, (K, K))[cy, cx]
        veg_win = sliding_window_view(veg_pad, (K, K))[cy, cx]

        grids[:n, CH_TERRAIN] = terr_win * masks
        grids[:n, CH_WATER] = water_win * masks
        grids[:n, CH_VEG] = veg_win * masks

    # ------------------------------------------------------------------ entity channels
    def _fill_entity_channels(self, grids, idx, px, py, cx, cy, spec_arr, sens) -> None:
        ent = self.ent
        cfg = self.cfg
        is_sheep = spec_arr == SHEEP
        is_fox = spec_arr == FOX

        # THREAT: prey see foxes (predators see no threats). Skip when no foxes are alive.
        fox_grid = self._species_grids.get(FOX)
        if fox_grid is not None and int(ent.count_species(FOX)) > 0:
            for k in np.nonzero(is_sheep)[0]:
                cand, cpx, cpy = fox_grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
                if cand.shape[0]:
                    self._scatter(grids[k, CH_THREAT], int(cx[k]), int(cy[k]), cpx, cpy)

        # FOOD (entities): foxes see EXPOSED prey -- sheep hidden in cover are invisible to
        # predators (the prey refuge, v1.md §18). Herbivores leave this channel empty; their
        # food is the vegetation channel.
        sheep_grid = self._species_grids.get(SHEEP)
        if sheep_grid is not None and is_fox.any():
            for k in np.nonzero(is_fox)[0]:
                cand, cpx, cpy = sheep_grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
                if cand.shape[0] == 0:
                    continue
                keep = ~self.world.in_cover(cpx, cpy)
                if keep.any():
                    self._scatter(grids[k, CH_FOOD], int(cx[k]), int(cy[k]), cpx[keep], cpy[keep])

        # MATE: adults see in-range conspecifics of the opposite sex who are also adult.
        # Juveniles can't mate, so they skip the (expensive) query entirely.
        for k in range(idx.shape[0]):
            sp = int(spec_arr[k])
            slot = idx[k]
            if ent.age[slot] < cfg.species[sp].maturity_age:
                continue
            grid = self._species_grids.get(sp)
            if grid is None:
                continue
            cand, cpx, cpy = grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
            if cand.shape[0] == 0:
                continue
            mat = cfg.species[sp].maturity_age
            valid = (ent.sex[cand] != ent.sex[slot]) & (ent.age[cand] >= mat) & (cand != slot)
            if valid.any():
                self._scatter(grids[k, CH_MATE], int(cx[k]), int(cy[k]), cpx[valid], cpy[valid])

    def _scatter(self, chan, ocx: int, ocy: int, cpx, cpy) -> None:
        """Mark candidate world positions as present cells in agent ``k``'s (K,K) window."""
        R = self.R
        ox = cpx.astype(np.int32) - ocx
        oy = cpy.astype(np.int32) - ocy
        m = (ox >= -R) & (ox <= R) & (oy >= -R) & (oy <= R)
        if m.any():
            chan[oy[m] + R, ox[m] + R] = 1.0
