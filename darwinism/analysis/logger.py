"""CSV logging of per-tick simulation stats + per-species trait means (§17 of v1.md).

Appends one row every ``log_every`` ticks. Columns include population counts, vegetation
biomass, births, deaths-by-cause, and mean heritable traits per species (the evolution
signal). Headless runs are the primary producer.
"""
from __future__ import annotations

import csv
from pathlib import Path

from darwinism.sim import genome as gn


class Logger:
    def __init__(self, path: str, sim, log_every: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sim = sim
        self.log_every = log_every if log_every is not None else sim.cfg.sim.log_every
        self._fh = None
        self._writer = None
        self._fields = self._build_fields()

    def _build_fields(self):
        species = self.sim.cfg.species
        names = [species[sid].name for sid in sorted(species)]     # ascending id (n_sheep, n_fox, ..)
        base = ["tick"] + [f"n_{name}" for name in names]
        base += ["veg_biomass", "births", "deaths", "death_starve", "death_thirst",
                 "death_age", "death_health", "death_predation"]
        for name in names:
            for g in gn.GENE_NAMES:
                base.append(f"{name}_{g}")
        return base

    def open(self):
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
        self._writer.writeheader()

    def record(self):
        if self._writer is None:
            self.open()
        sim = self.sim
        if sim.tick % self.log_every != 0:
            return
        row = {k: sim.stats.get(k, 0) for k in self._fields if k in sim.stats}
        species = sim.cfg.species
        for sid in sorted(species):
            name = species[sid].name
            traits = sim.trait_means(sid)
            for g in gn.GENE_NAMES:
                row[f"{name}_{g}"] = traits[g]
        # ensure all fields present
        for k in self._fields:
            row.setdefault(k, sim.stats.get(k, 0))
        self._writer.writerow(row)

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
