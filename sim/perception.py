"""Local, egocentric observation builder (§7.2, §12 of v1.md).

Produces the (N, OBS_DIM) matrix each tick. Every external block (food/threat/mate/
water) is gated by the agent's heritable ``sensory_range``: if nothing of a category is
within range the block is zeroed and ``present = 0``. Relative offsets are normalized by
sensory_range so the vector is scale-free. There is NEVER a global "nearest" fallback.

OBS layout (OBS_DIM = 29):
  0 hunger | 1 thirst | 2 energy | 3 health | 4 age/max_age | 5 sex
  6-9   nearest food   : dx/r, dy/r, dist/r, present
  10-13 nearest threat : dx/r, dy/r, dist/r, present
  14-17 nearest mate   : dx/r, dy/r, dist/r, present
  18-21 nearest water  : dx/r, dy/r, dist/r, present
  22 temperature | 23 nutrients | 24 elevation | 25 moisture
  26 on_water | 27 time_of_day | 28 season
"""
from __future__ import annotations

import numpy as np

from config import Config, SHEEP, FOX
from sim import genome as gn

OBS_DIM = 29


class Perception:
    def __init__(self, cfg: Config, world, entities, grid, env):
        self.cfg = cfg
        self.world = world
        self.ent = entities
        self.grid = grid
        self.env = env
        self._species_grids = {}  # species_id -> SpatialGrid (rebuilt each tick)

    # ------------------------------------------------------------------ public
    def build(self, temp_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs, idx) where obs is (N,OBS_DIM) for the N alive agents in ``idx``."""
        ent = self.ent
        idx = ent.alive_indices()
        n = idx.shape[0]
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        if n == 0:
            return obs, idx

        px = ent.pos_x[idx]
        py = ent.pos_y[idx]
        spec_arr = ent.species[idx]
        sens = gn.gene(ent.genome[idx], "sensory_range")
        max_age = gn.gene(ent.genome[idx], "max_age")

        # --- internal block 0-5 ---
        obs[:, 0] = ent.hunger[idx]
        obs[:, 1] = ent.thirst[idx]
        obs[:, 2] = ent.energy[idx]
        obs[:, 3] = ent.health[idx]
        obs[:, 4] = np.clip(ent.age[idx] / np.maximum(max_age, 1e-6), 0.0, 1.0)
        obs[:, 5] = ent.sex[idx].astype(np.float32)

        # --- external blocks (per category) ---
        self._fill_water(obs, px, py, sens)
        self._fill_food(obs, idx, px, py, spec_arr, sens)
        self._fill_threat_and_mate(obs, idx, px, py, spec_arr, sens)

        # --- local env block 22-28 ---
        cx, cy = self.world.world_to_cell(px, py)
        obs[:, 22] = temp_field[cy, cx]
        obs[:, 23] = self.world.nutrients[cy, cx]
        obs[:, 24] = self.world.elevation[cy, cx]
        obs[:, 25] = self.world.moisture[cy, cx]
        obs[:, 26] = self.world.freshwater[cy, cx].astype(np.float32)
        obs[:, 27] = self.env.time_of_day
        obs[:, 28] = self.env.season
        return obs, idx

    # ------------------------------------------------------------------ water
    def _fill_water(self, obs, px, py, sens):
        """Nearest freshwater from the precomputed world distance/direction fields."""
        w = self.world
        cx, cy = w.world_to_cell(px, py)
        dist = w.fw_dist[cy, cx]                # world units (cells)
        tx = w.fw_nearest_x[cy, cx].astype(np.float32)
        ty = w.fw_nearest_y[cy, cx].astype(np.float32)
        in_range = (dist <= sens) & np.isfinite(dist)
        dx = (tx - px)
        dy = (ty - py)
        r = np.maximum(sens, 1e-6)
        obs[:, 18] = np.where(in_range, np.clip(dx / r, -1, 1), 0.0)
        obs[:, 19] = np.where(in_range, np.clip(dy / r, -1, 1), 0.0)
        obs[:, 20] = np.where(in_range, np.clip(dist / r, 0, 1), 0.0)
        obs[:, 21] = in_range.astype(np.float32)

    # ------------------------------------------------------------------ food
    def _fill_food(self, obs, idx, px, py, spec_arr, sens):
        """Sheep food = best vegetation cell in range; fox food = nearest sheep in range."""
        w = self.world
        veg = self.veg
        thr = self.cfg.sim.food_eat_threshold
        is_sheep = spec_arr == SHEEP
        is_fox = spec_arr == FOX

        # --- sheep: faithful local forage perception -- each sheep sees the
        # highest-vegetation cell within ITS OWN sensory_range. Batched across all sheep
        # with a per-agent-masked offset stencil (vectorized, but semantics are exactly
        # "best grass cell I can actually see"). ---
        sheep_rows = np.nonzero(is_sheep)[0]
        if sheep_rows.shape[0]:
            self._sheep_food_vectorized(obs, sheep_rows, px, py, sens, veg, thr)

        # --- fox: nearest EXPOSED sheep via species grid (sheep in cover are hidden) ---
        if is_fox.any():
            sheep_grid = self._species_grids.get(SHEEP)
            for k in np.nonzero(is_fox)[0]:
                self._nearest_from_grid(obs, k, px[k], py[k], sens[k], sheep_grid,
                                        base=6, exclude_slot=None, exclude_cover=True)

    def _sheep_food_vectorized(self, obs, rows, px, py, sens, veg, thr):
        w = self.world
        R = int(np.ceil(self.cfg.species[SHEEP].gene_ranges["sensory_range"].hi))
        K = 2 * R + 1
        offs = np.arange(-R, R + 1)
        oy, ox = np.meshgrid(offs, offs, indexing="ij")        # (K,K)
        d2_stencil = (ox.astype(np.float32) ** 2 + oy.astype(np.float32) ** 2)  # (K,K)

        sx = px[rows]
        sy = py[rows]
        scx = np.clip(sx.astype(np.intp), 0, w.w - 1)
        scy = np.clip(sy.astype(np.intp), 0, w.h - 1)
        srange = sens[rows]

        # target cell indices for every (agent, stencil) pair
        tx = scx[:, None, None] + ox[None]                      # (S,K,K)
        ty = scy[:, None, None] + oy[None]
        in_bounds = (tx >= 0) & (tx < w.w) & (ty >= 0) & (ty < w.h)
        txc = np.clip(tx, 0, w.w - 1)
        tyc = np.clip(ty, 0, w.h - 1)
        veg_vals = veg[tyc, txc]                                # (S,K,K)

        within = d2_stencil[None] <= (srange[:, None, None] ** 2)
        valid = within & in_bounds & (veg_vals > thr)
        score = np.where(valid, veg_vals - 0.02 * np.sqrt(d2_stencil)[None], -np.inf)

        flat = score.reshape(score.shape[0], -1)
        best = np.argmax(flat, axis=1)
        has_food = np.isfinite(flat[np.arange(flat.shape[0]), best])
        by, bx = np.unravel_index(best, (K, K))

        # target cell center relative to the sheep's continuous position
        target_x = scx + ox.reshape(-1)[best] + 0.5
        target_y = scy + oy.reshape(-1)[best] + 0.5
        tdx = target_x - sx
        tdy = target_y - sy
        dist = np.sqrt(tdx * tdx + tdy * tdy)
        rr = np.maximum(srange, 1e-6)

        present = has_food.astype(np.float32)
        obs[rows, 6] = np.where(has_food, np.clip(tdx / rr, -1, 1), 0.0)
        obs[rows, 7] = np.where(has_food, np.clip(tdy / rr, -1, 1), 0.0)
        obs[rows, 8] = np.where(has_food, np.clip(dist / rr, 0, 1), 0.0)
        obs[rows, 9] = present

    # ------------------------------------------------------------------ threat / mate
    def _fill_threat_and_mate(self, obs, idx, px, py, spec_arr, sens):
        ent = self.ent
        is_sheep = spec_arr == SHEEP
        fox_grid = self._species_grids.get(FOX)
        age = ent.age[idx]

        # threat: sheep see nearest fox (block 10-13). foxes: none. Skip entirely when
        # there are no foxes alive (common; avoids N pointless grid queries).
        n_fox = int(self.ent.count_species(FOX))
        if n_fox > 0 and is_sheep.any() and fox_grid is not None:
            for k in np.nonzero(is_sheep)[0]:
                self._nearest_from_grid(obs, k, px[k], py[k], sens[k], fox_grid,
                                        base=10, exclude_slot=None)

        # mate: nearest in-range conspecific, opposite sex, adult (block 14-17).
        # Only ADULTS can mate, so juveniles skip the (expensive) query -- a big win in a
        # growing population dominated by young animals.
        for k in range(idx.shape[0]):
            sp = int(spec_arr[k])
            if age[k] < self.cfg.species[sp].maturity_age:
                continue
            grid = self._species_grids.get(sp)
            if grid is None:
                continue
            self._nearest_mate(obs, k, idx[k], px[k], py[k], sens[k], grid, sp)

    # ------------------------------------------------------------------ helpers
    def _nearest_from_grid(self, obs, k, x, y, r, grid, base, exclude_slot,
                           exclude_cover=False):
        if grid is None:
            return
        cand, cpx, cpy = grid.query_radius(float(x), float(y), float(r))
        if cand.shape[0] == 0:
            return
        if exclude_slot is not None:
            keep = cand != exclude_slot
            cand, cpx, cpy = cand[keep], cpx[keep], cpy[keep]
            if cand.shape[0] == 0:
                return
        if exclude_cover:                       # predators can't see prey hidden in cover
            keep = ~self.world.in_cover(cpx, cpy)
            cand, cpx, cpy = cand[keep], cpx[keep], cpy[keep]
            if cand.shape[0] == 0:
                return
        d2 = (cpx - x) ** 2 + (cpy - y) ** 2
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        rr = max(float(r), 1e-6)
        obs[k, base + 0] = np.clip((cpx[j] - x) / rr, -1, 1)
        obs[k, base + 1] = np.clip((cpy[j] - y) / rr, -1, 1)
        obs[k, base + 2] = np.clip(dist / rr, 0, 1)
        obs[k, base + 3] = 1.0

    def _nearest_mate(self, obs, k, slot, x, y, r, grid, species_id):
        ent = self.ent
        cand, cpx, cpy = grid.query_radius(float(x), float(y), float(r))
        if cand.shape[0] == 0:
            return
        my_sex = ent.sex[slot]
        spec = self.cfg.species[species_id]
        opp = ent.sex[cand] != my_sex
        adult = ent.age[cand] >= spec.maturity_age
        valid = opp & adult & (cand != slot)
        if not valid.any():
            return
        cand, cpx, cpy = cand[valid], cpx[valid], cpy[valid]
        d2 = (cpx - x) ** 2 + (cpy - y) ** 2
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        rr = max(float(r), 1e-6)
        obs[k, 14] = np.clip((cpx[j] - x) / rr, -1, 1)
        obs[k, 15] = np.clip((cpy[j] - y) / rr, -1, 1)
        obs[k, 16] = np.clip(dist / rr, 0, 1)
        obs[k, 17] = 1.0
