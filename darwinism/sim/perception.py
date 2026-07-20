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

  Sheep channels (6):  terrain | water | food (=grass field) | threat (=foxes) | mate | dist
  Fox   channels (5):  terrain | water | food (=exposed prey) | mate | dist

The ``food`` channel is unified in position but species-specific in content: a herbivore's
food is the vegetation field, a carnivore's food is prey entities. Foxes have no predators,
so they carry no ``threat`` channel.

The trailing POSITIONAL channel (``dist``) is common to every species: each cell's radial
distance from the agent, masked by the agent's OWN vision disc exactly like the content
channels and normalized. A translation-equivariant conv otherwise has no way to know how far
a perceived cell lies -- which is exactly what "nearest"/"best" targeting needs -- so this
channel supplies it. (Direction to a target is recovered downstream by the soft-argmax head's
own coordinate readout, and the RuleBrain is isotropic, so distance is the whole positional
signal; no x/y channel is carried.) The RuleBrain itself ignores it.

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

from darwinism.config import FOX, SHEEP, Config, FieldFood, predators_of, prey_of
from darwinism.sim import genome as gn

# --- per-species grid channel layouts ---
# Content channels (species-specific) come first, then a common POSITIONAL block is appended
# AFTER them (see below). Because the content indices never shift, the RuleBrain -- which
# reads channels by fixed index -- is unaffected by the positional block.
# Sheep content (5):
SH_TERRAIN, SH_WATER, SH_FOOD, SH_THREAT, SH_MATE = range(5)
SHEEP_N_CONTENT = 5
# Fox content (4) (no threat, food is prey entities):
FX_TERRAIN, FX_WATER, FX_FOOD, FX_MATE = range(4)
FOX_N_CONTENT = 4

# POSITIONAL channel -- common to every species, appended after the content channels: the
# radial DISTANCE of each cell from the agent, masked by the agent's OWN vision disc (the same
# mask as the content channels) and normalized. A plain conv is translation-equivariant and
# otherwise cannot know how far a perceived cell is -- exactly what "nearest"/"best" targeting
# needs (both are functions of distance alone). No x/y channel is carried: the soft-argmax head
# recovers the *direction* to a target from its own coordinate readout, and the RuleBrain is
# isotropic (nearest/best regardless of direction), so distance is the whole positional signal
# to clone. The RuleBrain ignores this channel (it decodes its own targets).
POS_DIST = 0                               # offset WITHIN the positional block
N_POS_CHANNELS = 1
SHEEP_N_CHANNELS = SHEEP_N_CONTENT + N_POS_CHANNELS      # 6
FOX_N_CHANNELS = FOX_N_CONTENT + N_POS_CHANNELS          # 5

_POS_NAMES = ("dist",)
# channel names per species (index = position in that species' grid stack); used by the
# viewer's perception inspector and as the single source of truth for the layouts.
CHANNEL_NAMES = {
    SHEEP: ("terrain", "water", "food", "threat", "mate") + _POS_NAMES,
    FOX:   ("terrain", "water", "food", "mate") + _POS_NAMES,
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
    ``channels``: dict role -> grid channel index for THIS species (e.g. {"terrain":0,
      "water":1, "food":2, "mate":3, "dist":4}). Makes the grid self-describing so a brain
      can read a channel by role without hardcoded indices (see RuleBrain).
    ``food_reduction``: how a rule brain should reduce the food channel to a target -- "best"
      (richest cell, for grazers eating a continuous field) or "nearest" (closest marker, for
      hunters eating discrete prey). A neural brain ignores this and learns off the raw channel.
    """
    __slots__ = ("grids", "scalars", "radius", "idx", "species", "channels", "food_reduction")

    def __init__(self, grids, scalars, radius, idx, species, channels=None,
                 food_reduction="nearest"):
        self.grids = grids
        self.scalars = scalars
        self.radius = radius
        self.idx = idx
        self.species = species
        self.channels = channels
        self.food_reduction = food_reduction


class Perception:
    def __init__(self, cfg: Config, world, entities, env):
        self.cfg = cfg
        self.world = world
        self.ent = entities
        self.env = env
        self.veg = None                       # wired in per-tick by Simulation
        self.temp_field = None                # wired in per-tick by Simulation
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
        # normalized positional stencil fed as the common POSITIONAL channel: radial distance
        # in [0,1], kept ~[0,1]-scaled to match the content channels' range.
        Rf = float(self.R)
        self._pos_d = (self._d_cell / (float(np.sqrt(2.0)) * Rf)).astype(np.float32)

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

        # --- per-species channel SCHEMA, derived from diet + predation relationships (data,
        # not hardcoded). Canonical order: terrain, water, food, [threat iff the species has
        # predators], mate, then the common positional 'dist'. For the default sheep+fox set
        # this reproduces the historical layouts exactly (sheep=6, fox=5). ---
        self._prey_of = prey_of(cfg.species)          # sid -> prey species ids (nearest-entity food)
        self._predators_of = predators_of(cfg.species)  # sid -> predator species ids (threat)
        self._schema = {}          # sid -> tuple of content role names (no positional)
        self._chan_index = {}      # sid -> {role: channel index}  (includes 'dist')
        self._n_channels = {}      # sid -> total channel count (content + positional)
        self._channel_names = {}   # sid -> tuple of channel names (content + positional)
        self._food_fields = {}     # sid -> list of world-field names this species grazes
        self._food_reduction = {}  # sid -> "best" (grazer) | "nearest" (hunter), for RuleBrain
        for sid in sorted(cfg.species):
            spec = cfg.species[sid]
            self._food_fields[sid] = [s.field for s in spec.diet if isinstance(s, FieldFood)]
            # a species that grazes a continuous field reduces food to the BEST cell; a pure
            # hunter reduces the (binary) prey markers to the NEAREST one.
            self._food_reduction[sid] = "best" if self._food_fields[sid] else "nearest"
            roles = ["terrain", "water", "food"]
            if self._predators_of.get(sid):
                roles.append("threat")
            roles.append("mate")
            names = tuple(roles) + _POS_NAMES
            self._schema[sid] = tuple(roles)
            self._channel_names[sid] = names
            self._chan_index[sid] = {r: i for i, r in enumerate(names)}
            self._n_channels[sid] = len(names)

        # lazily-grown output buffers, one (grids, scalars) pair per species
        self._buf = {}                        # species_id -> [grids, scalars]

    # ------------------------------------------------------------------ buffers
    def _ensure_buffers(self, sid: int, n: int):
        buf = self._buf.get(sid)
        if buf is None or buf[0].shape[0] < n:
            cap = max(n, 1)
            buf = [np.zeros((cap, self._n_channels[sid], self.K, self.K), dtype=np.float32),
                   np.zeros((cap, SCALAR_DIM), dtype=np.float32)]
            self._buf[sid] = buf
        return buf

    # ------------------------------------------------------------------ public
    def build(self):
        """Return (obs_by_species, idx). ``idx`` is the global alive ordering."""
        ent = self.ent
        idx = ent.alive_indices()
        species_of_idx = ent.species[idx]
        veg_pad = np.pad(self.veg, self.R)            # one veg pad per tick (grazing field)
        obs_by_species = {}
        for sid in sorted(self.cfg.species):          # ascending id (determinism)
            sp_idx = idx[species_of_idx == sid]       # sorted subset of the global idx
            obs_by_species[sid] = self._build_species(sid, sp_idx, veg_pad)
        return obs_by_species, idx

    def _build_species(self, sid, sp_idx, veg_pad) -> Observation:
        n = sp_idx.shape[0]
        grids, scalars = self._ensure_buffers(sid, n)
        ci = self._chan_index[sid]                   # role -> channel index (schema)
        fr = self._food_reduction[sid]
        if n == 0:
            return Observation(grids[:0], scalars[:0], self.R, sp_idx, sid, ci, fr)

        ent = self.ent
        px = ent.pos_x[sp_idx]
        py = ent.pos_y[sp_idx]
        sens = gn.gene(ent.genome[sp_idx], "sensory_range")
        max_age = gn.gene(ent.genome[sp_idx], "max_age")
        cx, cy = self.world.world_to_cell(px, py)
        cx = cx.astype(np.int32)
        cy = cy.astype(np.int32)
        masks = self._mask_cache[np.clip(np.rint(sens).astype(np.intp), 0, self.R)]  # (n,K,K)

        # --- field channels: terrain + water (every species; assignment overwrites) ---
        grids[:n, ci["terrain"]] = self._field(self._terr_pad, cx, cy, masks)
        grids[:n, ci["water"]] = self._field(self._water_pad, cx, cy, masks)

        # --- zero the food + entity channels (buffer is reused across ticks; scatter only
        # SETS present cells) before filling them ---
        grids[:n, ci["food"]] = 0.0
        if "threat" in ci:
            grids[:n, ci["threat"]] = 0.0
        grids[:n, ci["mate"]] = 0.0

        # --- food (species-specific by diet): graze world field(s) and/or hunt prey species ---
        for _fld in self._food_fields[sid]:
            # the vegetation per-cell field is the only grazeable field today
            grids[:n, ci["food"]] = self._field(veg_pad, cx, cy, masks)
        if self._prey_of.get(sid):
            # exposed prey only -- prey hidden in cover are invisible to predators (refuge)
            self._scatter_from_species(grids, n, px, py, cx, cy, sens, ci["food"],
                                       self._prey_of[sid], cover_filter=True)

        # --- threat: nearby predator species (only present when this species has predators) ---
        if "threat" in ci:
            self._scatter_from_species(grids, n, px, py, cx, cy, sens, ci["threat"],
                                       self._predators_of[sid], cover_filter=False)

        # --- mate: conspecific opposite-sex adults ---
        self._scatter_mates(grids, n, sp_idx, px, py, cx, cy, sens, sid, ci["mate"])

        # --- positional channel (common): radial distance, masked to each agent's own vision
        # disc exactly like the content channels above ---
        grids[:n, ci["dist"]] = self._pos_d[None] * masks

        # --- scalars ---
        s = scalars
        s[:n, S_HUNGER] = ent.hunger[sp_idx]
        s[:n, S_THIRST] = ent.thirst[sp_idx]
        s[:n, S_ENERGY] = ent.energy[sp_idx]
        s[:n, S_HEALTH] = ent.health[sp_idx]
        s[:n, S_AGE] = np.clip(ent.age[sp_idx] / np.maximum(max_age, 1e-6), 0.0, 1.0)
        s[:n, S_SEX] = ent.sex[sp_idx].astype(np.float32)
        s[:n, S_TEMP] = self.temp_field[cy, cx]
        s[:n, S_TIME] = self.env.time_of_day
        s[:n, S_SEASON] = self.env.season
        s[:n, S_SENSORY] = sens

        return Observation(grids[:n], scalars[:n], self.R, sp_idx, sid, ci, fr)

    # ------------------------------------------------------------------ fill helpers
    def _field(self, src_pad, cx, cy, masks):
        """Egocentric KxK window of a padded world field, masked by each agent's eye disc."""
        K = self.K
        return sliding_window_view(src_pad, (K, K))[cy, cx] * masks   # (n,K,K)

    def _scatter_from_species(self, grids, n, px, py, cx, cy, sens, ch, species_ids,
                              cover_filter):
        """Mark in-range members of the given (ascending-id) species into channel ``ch``.

        Used for both the ``food`` channel (prey species, ``cover_filter=True`` so prey hidden
        in cover are invisible -- the refuge, v1.md §18) and the ``threat`` channel (predator
        species, no cover filter). Draws no RNG; scatter is idempotent (sets cells to 1.0), so
        the order among ``species_ids`` does not affect the result -- for the default config
        each list is a singleton, byte-identical to the old per-role helpers."""
        ent = self.ent
        for other in species_ids:
            grid = self._species_grids.get(other)
            if grid is None or int(ent.count_species(other)) == 0:
                continue
            for k in range(n):
                cand, cpx, cpy = grid.query_radius(float(px[k]), float(py[k]), float(sens[k]))
                if cand.shape[0] == 0:
                    continue
                if cover_filter:
                    keep = ~self.world.in_cover(cpx, cpy)
                    if not keep.any():
                        continue
                    cpx, cpy = cpx[keep], cpy[keep]
                self._scatter(grids[k, ch], int(cx[k]), int(cy[k]), cpx, cpy)

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
