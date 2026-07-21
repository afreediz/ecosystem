"""Structure-of-Arrays entity store (§5, §10 of v1.md).

All animal state lives in parallel NumPy arrays indexed by *slot*, sized to a fixed
max-capacity pool. An ``alive`` boolean mask marks live slots; dead slots are recycled
via a free list. Vegetation is NOT here -- it is a per-cell field on the World.

Iteration order for determinism is always ascending slot index.
"""
from __future__ import annotations

import numpy as np

from darwinism.config import Config, SpeciesConfig
from darwinism.sim import genome as gn

# Sex labels for the ``sex`` array (random 50/50 at birth; non-heritable). The choice of
# which integer is male is arbitrary -- it only drives the viewer's male marker.
FEMALE = 0
MALE = 1


class Entities:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cap = cfg.sim.max_entities
        self.cap = cap

        # --- live state arrays (one row per slot) ---
        self.pos_x = np.zeros(cap, dtype=np.float32)
        self.pos_y = np.zeros(cap, dtype=np.float32)
        self.heading_x = np.zeros(cap, dtype=np.float32)
        self.heading_y = np.zeros(cap, dtype=np.float32)
        self.energy = np.zeros(cap, dtype=np.float32)
        self.hunger = np.zeros(cap, dtype=np.float32)
        self.thirst = np.zeros(cap, dtype=np.float32)
        self.health = np.zeros(cap, dtype=np.float32)
        self.age = np.zeros(cap, dtype=np.float32)
        self.sex = np.zeros(cap, dtype=np.int8)          # 0 / 1
        self.species = np.full(cap, -1, dtype=np.int8)
        self.genome = np.zeros((cap, gn.N_GENES), dtype=np.float32)
        self.repro_cooldown = np.zeros(cap, dtype=np.float32)
        # Monotonic per-animal identity token. A slot is recycled by the free list, so its
        # index alone cannot tell one animal from the next occupant. ``birth_id`` gets a fresh
        # unique value on every spawn, so anything that keeps per-agent state across ticks
        # (e.g. a neural brain's LSTM memory) can detect "this slot now holds a DIFFERENT
        # animal" by a changed id and reset. 0 == a slot that never held an animal. This is
        # bookkeeping only -- it draws no RNG, so it does not affect run determinism.
        self.birth_id = np.zeros(cap, dtype=np.int64)
        # cosmetic countdown (ticks): >0 means "recently bred" -> viewer tints it rose.
        # Never read by any decision/system, so it cannot affect determinism.
        self.mating_glow = np.zeros(cap, dtype=np.float32)
        # circadian rest state: True while the animal is sleeping (set by the sleep system).
        # Sleepers hold position, suppress eat/drink/mate, and burn energy slowly. Affects
        # the sim (movement/metabolism), so it is real state, not a viewer-only flag.
        self.asleep = np.zeros(cap, dtype=bool)
        # set True the tick the sleep system OVERRODE this agent's action (asleep OR dashing to
        # cover): the brain's emitted action was discarded, so it drove no outcome. Pure
        # diagnostic state -- no system reads it, so it cannot affect dynamics/determinism; the
        # RL trainer uses it to exclude action-overridden steps from the policy gradient.
        self.action_overridden = np.zeros(cap, dtype=bool)
        self.alive = np.zeros(cap, dtype=bool)

        # free list of available slots (stack; pop from the end)
        self._free = list(range(cap - 1, -1, -1))
        # next identity token to hand out (see ``birth_id`` above); starts at 1 so 0 stays
        # reserved for "never spawned".
        self._next_birth_id = 1

    # ------------------------------------------------------------------ helpers
    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    def alive_indices(self) -> np.ndarray:
        return np.nonzero(self.alive)[0]

    def species_mask(self, species_id: int) -> np.ndarray:
        return self.alive & (self.species == species_id)

    def count_species(self, species_id: int) -> int:
        return int(self.species_mask(species_id).sum())

    # ------------------------------------------------------------------ spawn / kill
    def _take_slots(self, n: int) -> np.ndarray:
        n = min(n, len(self._free))
        if n <= 0:
            return np.empty(0, dtype=np.intp)
        slots = np.array([self._free.pop() for _ in range(n)], dtype=np.intp)
        return slots

    def spawn(self, spec: SpeciesConfig, genomes: np.ndarray, pos: np.ndarray,
              rng: np.random.Generator, energy: float = 0.7, age=0.0) -> np.ndarray:
        """Create entities of one species at given positions with given genomes.

        ``genomes`` is (n, N_GENES); ``pos`` is (n, 2). ``age`` may be a scalar or a
        per-entity array (used to seed founders as adults). Returns the slot indices used.
        Silently truncates if the pool is full.
        """
        n = genomes.shape[0]
        slots = self._take_slots(n)
        k = slots.shape[0]
        if k == 0:
            return slots
        slots_k = slots
        self.pos_x[slots_k] = pos[:k, 0]
        self.pos_y[slots_k] = pos[:k, 1]
        ang = rng.uniform(0.0, 2 * np.pi, size=k).astype(np.float32)
        self.heading_x[slots_k] = np.cos(ang)
        self.heading_y[slots_k] = np.sin(ang)
        self.energy[slots_k] = energy
        self.hunger[slots_k] = 0.1
        self.thirst[slots_k] = 0.1
        self.health[slots_k] = 1.0
        age_arr = np.asarray(age, dtype=np.float32)
        self.age[slots_k] = age_arr[:k] if age_arr.ndim else age_arr
        self.sex[slots_k] = rng.integers(0, 2, size=k).astype(np.int8)
        self.species[slots_k] = spec.species_id
        self.genome[slots_k] = genomes[:k]
        # stamp each new animal with a unique identity token (see ``birth_id``)
        self.birth_id[slots_k] = np.arange(self._next_birth_id, self._next_birth_id + k,
                                           dtype=np.int64)
        self._next_birth_id += k
        self.repro_cooldown[slots_k] = 0.0
        self.mating_glow[slots_k] = 0.0    # recycled slots must not inherit a stale glow
        self.asleep[slots_k] = False       # newborns / recycled slots start awake
        self.action_overridden[slots_k] = False
        self.alive[slots_k] = True
        return slots_k

    def kill(self, slots: np.ndarray) -> None:
        slots = np.asarray(slots, dtype=np.intp)
        slots = slots[self.alive[slots]]
        if slots.shape[0] == 0:
            return
        self.alive[slots] = False
        self.species[slots] = -1
        # return slots to the free list (sorted desc so low indices are reused first,
        # which keeps iteration order stable and deterministic)
        for s in sorted(slots.tolist(), reverse=True):
            self._free.append(s)
