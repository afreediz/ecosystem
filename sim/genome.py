"""Genome layout, mutation, and sexual crossover (§11 of v1.md).

The gene vector is fixed-length and the gene *order* is global so every system can
index genes by name without per-entity lookups. Each species clamps its genes to its
own ``GeneRange`` bounds (config), but all species share the same physical layout --
genes a species does not use are simply left at their init value and ignored.
"""
from __future__ import annotations

import numpy as np

from config import SHEEP, FOX, SpeciesConfig

# In future, make the GENES specific to species
# Fixed global gene order. Indices here are used across perception/movement/metabolism.
GENE_NAMES = [
    "max_speed",
    "sensory_range",
    "metabolism_rate",
    "size",
    "max_age",
    "repro_threshold",
    "flee_distance",   # sheep behavioral gene
    "aggression",      # fox behavioral gene
    "chronotype",      # circadian gene: per-individual sleep-time offset (both species)
]
GENE_INDEX = {name: i for i, name in enumerate(GENE_NAMES)}
N_GENES = len(GENE_NAMES)

# Default (neutral) values for genes a species does not draw / use.
_NEUTRAL = {
    "max_speed": 1.0,
    "sensory_range": 12.0,
    "metabolism_rate": 1.0,
    "size": 1.0,
    "max_age": 2000.0,
    "repro_threshold": 0.6,
    "flee_distance": 0.7,
    "aggression": 0.7,
    "chronotype": 0.0,
}


def gene(genomes: np.ndarray, name: str) -> np.ndarray:
    """Column view of one gene across all entities. ``genomes`` is (N, N_GENES)."""
    return genomes[:, GENE_INDEX[name]]


def _bounds(spec: SpeciesConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene (lo, hi) clamp arrays for a species; neutral genes pin to their value."""
    lo = np.empty(N_GENES, dtype=np.float32)
    hi = np.empty(N_GENES, dtype=np.float32)
    for i, name in enumerate(GENE_NAMES):
        rng = spec.gene_ranges.get(name)
        if rng is None:
            lo[i] = hi[i] = _NEUTRAL[name]
        else:
            lo[i] = rng.lo
            hi[i] = rng.hi
    return lo, hi


def random_genomes(spec: SpeciesConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` fresh genomes uniformly within the species' gene ranges."""
    lo, hi = _bounds(spec)
    out = rng.uniform(lo, hi, size=(n, N_GENES)).astype(np.float32)
    return out


def mutate(genomes: np.ndarray, spec: SpeciesConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-gene gaussian mutation with probability ``mutation_rate``; clamp to range.

    Operates in place on a copy and returns it. ``genomes`` is (M, N_GENES).
    """
    g = genomes.astype(np.float32, copy=True)
    if g.shape[0] == 0:
        return g
    lo, hi = _bounds(spec)
    span = np.maximum(hi - lo, 1e-6)
    mask = rng.random(g.shape) < spec.mutation_rate
    deltas = rng.normal(0.0, spec.mutation_strength, size=g.shape).astype(np.float32)
    g += mask * deltas * span                    # mutation strength scaled by gene span
    np.clip(g, lo, hi, out=g)
    return g


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, spec: SpeciesConfig,
              rng: np.random.Generator) -> np.ndarray:
    """Uniform per-gene crossover of paired parents, then mutate.

    ``parent_a`` / ``parent_b`` are (M, N_GENES) aligned arrays (one row per child).
    Returns (M, N_GENES) child genomes.
    """
    m = parent_a.shape[0]
    if m == 0:
        return np.empty((0, N_GENES), dtype=np.float32)
    pick_a = rng.random((m, N_GENES)) < 0.5
    child = np.where(pick_a, parent_a, parent_b).astype(np.float32)
    return mutate(child, spec, rng)
