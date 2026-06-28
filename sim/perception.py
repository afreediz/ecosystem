"""Local, egocentric **per-species** grid perception (§7.2, §12 of v1.md).

Each tick this builds, for every alive agent, a stack of egocentric perception grids plus
a small vector of internal/global scalars. The grids are the agent's *raw* local view of
the world -- a square window of side ``K = 2*R + 1`` cells centred on the agent, where
``R`` is the largest sensory range across all species. Cells beyond the agent's OWN
heritable ``sensory_range`` (Euclidean, in cells) or outside the world are zeroed, so each
individual only perceives what its eyes can reach -- the window is just a fixed, batchable
canvas. Each grid channel is CNN-ready (the whole point of the grid design).

Perception is **separated by species**: a species only carries the channels it actually
uses, so there are no dead inputs for a future per-species CNN.

  Sheep channels (5):  terrain | water | food (=grass field) | threat (=foxes) | mate
  Fox   channels (4):  terrain | water | food (=exposed prey) | mate

The ``food`` channel is unified in position but species-specific in content: a herbivore's
food is the vegetation field, a carnivore's food is prey entities. Foxes have no predators,
so they carry no ``threat`` channel.

``build`` returns ``(obs_by_species, idx)``:
  * ``obs_by_species`` -- ``{species_id: Observation}``, each with that species' layout.
  * ``idx`` -- the GLOBAL alive indices, so all downstream systems still operate on one
    aligned set. Each ``Observation`` also carries its own ``idx`` (the global slot ids of
    its rows) so callers can scatter per-species results back into the global ordering.

SCALARS (obs.scalars: (N, SCALAR_DIM), float32) -- identical layout for both species:
  0 hunger | 1 thirst | 2 energy | 3 health | 4 age/max_age | 5 sex
  6 temperature(own cell) | 7 time_of_day | 8 season
  9 sensory_range (cells; for distance normalization + CNN)
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import Config, SHEEP, FOX
from sim import genome as gn

# --- per-species grid channel layouts ---
# Sheep:
SH_TERRAIN, SH_WATER, SH_FOOD, SH_THREAT, SH_MATE = range(5)
SHEEP_N_CHANNELS = 5
# Fox (no threat, food is prey entities):
FX_TERRAIN, FX_WATER, FX_FOOD, FX_MATE = range(4)
FOX_N_CHANNELS = 4

# channel names per species (index = position in that species' grid stack); used by the
# viewer's perception inspector and as the single source of truth for the layouts.
CHANNEL_NAMES = {
    SHEEP: ("terrain", "water", "food", "threat", "mate"),
    FOX:   ("terrain", "water", "food", "mate"),
}
SPECIES_N_CHANNELS = {SHEEP: SHEEP_N_CHANNELS, FOX: FOX_N_CHANNELS}

NUM_BIOMES = 7                      # see sim/world.py biome ids (OCEAN..PLAINS)

# --- scalar layout (shared by both species) ---
(S_HUNGER, S_THIRST, S_ENERGY, S_HEALTH, S_AGE, S_SEX,
 S_TEMP, S_TIME, S_SEASON, S_SENSORY) = range(10)
SCALAR_DIM = 10


class Observation:
    """The batched per-species perception handed to ``Brain.decide``.

    ``grids``   : (N, C, K, K) float32 -- egocentric channels for this species (CNN-ready).
    ``scalars`` : (N, SCALAR_DIM) float32 -- internal state + global env.
    ``radius``  : int -- the window half-width R (K = 2R+1).
    ``idx``     : (N,) global entity slot ids for the rows (sorted), for scatter-back.
    ``species`` : int -- the species id this observation describes.
    """
    __slots__ = ("grids", "scalars", "radius", "idx", "species")

    def __init__(self, grids, scalars, radius, idx, species):
        self.grids = grids
        self.scalars = scalars
        self.radius = radius
        self.idx = idx
        self.species = species


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
        # so each species' batch shares the same canvas; smaller-eyed individuals just see a
        # masked sub-disc of it.
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

        # lazily-grown output buffers, one (grids, scalars) pair per species
        self._buf = {}                        # species_id -> [grids, scalars]

    # ------------------------------------------------------------------ buffers
    def _ensure_buffers(self, sid: int, n: int):
        buf = self._buf.get(sid)
        if buf is None or buf[0].shape[0] < n:
            cap = max(n, 1)
            buf = [np.zeros((cap, SPECIES_N_CHANNELS[sid], self.K, self.K), dtype=np.float32),
                   np.zeros((cap, SCALAR_DIM), dtype=np.float32)]
            self._buf[sid] = buf
        return buf

    # ------------------------------------------------------------------ public
    def build(self, temp_field: np.ndarray):
        """Return (obs_by_species, idx). ``idx`` is the global alive ordering."""
        ent = self.ent
        idx = ent.alive_indices()
        species_of_idx = ent.species[idx]
        veg_pad = np.pad(self.veg, self.R)            # one veg pad per tick (sheep food)
        obs_by_species = {}
        for sid in (SHEEP, FOX):
            sp_idx = idx[species_of_idx == sid]       # sorted subset of the global idx
            obs_by_species[sid] = self._build_species(sid, sp_idx, temp_field, veg_pad)
        return obs_by_species, idx

    def _build_species(self, sid, sp_idx, temp_field, veg_pad) -> Observation:
        n = sp_idx.shape[0]
        grids, scalars = self._ensure_buffers(sid, n)
        if n == 0:
            return Observation(grids[:0], scalars[:0], self.R, sp_idx, sid)

        ent = self.ent
        px = ent.pos_x[sp_idx]
        py = ent.pos_y[sp_idx]
        sens = gn.gene(ent.genome[sp_idx], "sensory_range")
        max_age = gn.gene(ent.genome[sp_idx], "max_age")
        cx, cy = self.world.world_to_cell(px, py)
        cx = cx.astype(np.int32)
        cy = cy.astype(np.int32)
        masks = self._mask_cache[np.clip(np.rint(sens).astype(np.intp), 0, self.R)]  # (n,K,K)

        # --- field channels: terrain + water (both species) ---
        grids[:n, 0] = self._field(self._terr_pad, cx, cy, masks)    # terrain is index 0
        grids[:n, 1] = self._field(self._water_pad, cx, cy, masks)   # water is index 1

        # --- food + entity channels (species-specific) ---
        if sid == SHEEP:
            # food = vegetation field; threat = foxes; mate = conspecifics
            grids[:n, SH_FOOD] = self._field(veg_pad, cx, cy, masks)
            grids[:n, SH_THREAT:SH_MATE + 1] = 0.0   # entity channels: zero before scatter
            self._scatter_predators(grids, n, px, py, cx, cy, sens, SH_THREAT)
            self._scatter_mates(grids, n, sp_idx, px, py, cx, cy, sens, sid, SH_MATE)
        else:  # FOX
            # food = exposed prey entities; mate = conspecifics (no threat channel)
            grids[:n, FX_FOOD:FX_MATE + 1] = 0.0     # entity channels: zero before scatter
            self._scatter_prey(grids, n, px, py, cx, cy, sens, FX_FOOD)
            self._scatter_mates(grids, n, sp_idx, px, py, cx, cy, sens, sid, FX_MATE)

        # --- scalars ---
        s = scalars
        s[:n, S_HUNGER] = ent.hunger[sp_idx]
        s[:n, S_THIRST] = ent.thirst[sp_idx]
        s[:n, S_ENERGY] = ent.energy[sp_idx]
        s[:n, S_HEALTH] = ent.health[sp_idx]
        s[:n, S_AGE] = np.clip(ent.age[sp_idx] / np.maximum(max_age, 1e-6), 0.0, 1.0)
        s[:n, S_SEX] = ent.sex[sp_idx].astype(np.float32)
        s[:n, S_TEMP] = temp_field[cy, cx]
        s[:n, S_TIME] = self.env.time_of_day
        s[:n, S_SEASON] = self.env.season
        s[:n, S_SENSORY] = sens

        return Observation(grids[:n], scalars[:n], self.R, sp_idx, sid)

    # ------------------------------------------------------------------ fill helpers
    def _field(self, src_pad, cx, cy, masks):
        """Egocentric KxK window of a padded world field, masked by each agent's eye disc."""
        K = self.K
        return sliding_window_view(src_pad, (K, K))[cy, cx] * masks   # (n,K,K)

    def _scatter_predators(self, grids, n, px, py, cx, cy, sens, ch):
        """Prey-only: mark in-range foxes (skip entirely when none are alive)."""
        fox_grid = self._species_grids.get(FOX)
        if fox_grid is None or int(self.ent.count_species(FOX)) == 0:
            return
        for k in range(n):
            cand, cpx, cpy = fox_grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
            if cand.shape[0]:
                self._scatter(grids[k, ch], int(cx[k]), int(cy[k]), cpx, cpy)

    def _scatter_prey(self, grids, n, px, py, cx, cy, sens, ch):
        """Predator-only: mark in-range EXPOSED prey -- sheep hidden in cover are invisible
        to predators (the prey refuge, v1.md §18)."""
        sheep_grid = self._species_grids.get(SHEEP)
        if sheep_grid is None:
            return
        for k in range(n):
            cand, cpx, cpy = sheep_grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
            if cand.shape[0] == 0:
                continue
            keep = ~self.world.in_cover(cpx, cpy)
            if keep.any():
                self._scatter(grids[k, ch], int(cx[k]), int(cy[k]), cpx[keep], cpy[keep])

    def _scatter_mates(self, grids, n, sp_idx, px, py, cx, cy, sens, sid, ch):
        """Adults see in-range conspecifics of the opposite sex who are also adult. Juveniles
        can't mate, so they skip the (expensive) query entirely."""
        ent = self.ent
        grid = self._species_grids.get(sid)
        if grid is None:
            return
        mat = self.cfg.species[sid].maturity_age
        for k in range(n):
            slot = sp_idx[k]
            if ent.age[slot] < mat:
                continue
            cand, cpx, cpy = grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
            if cand.shape[0] == 0:
                continue
            valid = (ent.sex[cand] != ent.sex[slot]) & (ent.age[cand] >= mat) & (cand != slot)
            if valid.any():
                self._scatter(grids[k, ch], int(cx[k]), int(cy[k]), cpx[valid], cpy[valid])

    def _scatter(self, chan, ocx: int, ocy: int, cpx, cpy) -> None:
        """Mark candidate world positions as present cells in one agent's (K,K) window."""
        R = self.R
        ox = cpx.astype(np.int32) - ocx
        oy = cpy.astype(np.int32) - ocy
        m = (ox >= -R) & (ox <= R) & (oy >= -R) & (oy <= R)
        if m.any():
            chan[oy[m] + R, ox[m] + R] = 1.0
