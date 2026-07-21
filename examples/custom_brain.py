"""Drive one species with a CUSTOM Brain, leaving the others on the built-in RuleBrain.

A ``Brain`` implements ``decide(obs_by_species, idx) -> act`` where ``act`` is a
``(len(idx), ACT_DIM)`` float32 matrix aligned to the global alive ordering ``idx``. Read a
channel by ROLE from ``obs.channels`` (the grid is self-describing); the consumption /
reproduction systems enforce true world adjacency, so a brain only needs to point roughly at
a target and raise a gate. This example is a deterministic hand-written forager -- no RNG, so
the run stays reproducible.

    venv/Scripts/python.exe examples/custom_brain.py
"""
import numpy as np

import darwinism as dw


class GreedyForagerBrain(dw.Brain):
    """Head toward the best food cell in view and always try to eat/drink."""

    def decide(self, obs_by_species, idx):
        act = np.zeros((len(idx), dw.ACT_DIM), dtype=np.float32)
        for obs in obs_by_species.values():
            if obs.grids.shape[0] == 0:
                continue
            # reduce the food channel to a target direction (nearest/best cell)
            present, dx, dy, _dist = dw.best_in_channel(obs.grids[:, obs.channels["food"]])
            mag = np.sqrt(dx * dx + dy * dy)
            safe = mag > 1e-6
            hx = np.where(safe, dx / np.where(safe, mag, 1.0), 1.0)   # else drift east
            hy = np.where(safe, dy / np.where(safe, mag, 1.0), 0.0)
            rows = np.searchsorted(idx, obs.idx)     # this species' rows in the global ordering
            act[rows, dw.A_DX] = hx
            act[rows, dw.A_DY] = hy
            act[rows, dw.A_EAT] = 1.0
            act[rows, dw.A_DRINK] = 1.0
            act[rows, dw.A_SPEED] = 1.0
        return act


cfg = dw.make_config(world_seed=12345, seed=7)
# sheep -> custom brain; fox -> None means "fall back to the shared RuleBrain"
sim = dw.Simulation(cfg, brain={dw.SHEEP: GreedyForagerBrain(), dw.FOX: None})

for tick in range(1500):
    sim.step()
    if (tick + 1) % 500 == 0:
        print(f"tick {tick + 1:>5}  {sim.populations}")

print("\nfinal:", sim.populations)
