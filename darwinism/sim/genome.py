"""Genome layout, mutation, and sexual crossover (§11 of v1.md).

The gene vector is fixed-length and the gene *order* is global so every system can index
genes by name without per-entity lookups. Each species clamps its genes to its own
``GeneRange`` bounds (config); a gene a species does not declare is pinned to a neutral value
and ignored -- but it still occupies a column, so every species shares one physical layout.

**Runtime registry (framework).** The layout is no longer a hardcoded list: it is BUILT from
the registered species via ``build_registry(cfg.species)``, called once at ``Simulation``
construction (before the entity store is sized). This lets a new species introduce a novel
trait without editing this module. To keep the default (sheep + fox) run byte-identical, the
registry uses a fixed CANONICAL order for the built-in genes and reproduces the exact 9-name
list the codebase has always used; genuinely novel genes are appended after them, in
species-id then declaration order. Determinism note: the registry lives in module globals, so
build ONE ``Simulation`` at a time per process (or reuse the same species set) -- constructing
two simulations with different gene sets in the same process shares this one layout.
"""
from __future__ import annotations

import numpy as np

from darwinism.config import SpeciesConfig

# Canonical order + neutral value of every BUILT-IN gene. The registry emits these first, in
# this order, so the default species set yields the historical 9-gene layout unchanged.
_CANONICAL_ORDER = [
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

# Active layout (rebuilt by ``build_registry``). Initialised to the full built-in set so the
# module is usable before any Simulation is constructed (e.g. introspection, tests).
GENE_NAMES = list(_CANONICAL_ORDER)
GENE_INDEX = {name: i for i, name in enumerate(GENE_NAMES)}
N_GENES = len(GENE_NAMES)


def build_registry(species: dict) -> list[str]:
    """(Re)build the global gene layout from the registered species and return it.

    Called once at ``Simulation`` construction. Emits every built-in gene ANY species uses in
    canonical order, then any novel (non-built-in) gene, ordered by species id then the order
    it appears in that species' ``gene_ranges``. For the default sheep+fox set this reproduces
    the historical 9-gene list exactly, so the RNG stream (draws are shaped by ``N_GENES``)
    is unchanged.
    """
    global GENE_NAMES, GENE_INDEX, N_GENES
    used = set()
    novel: list[str] = []
    for sid in sorted(species):
        for name in species[sid].gene_ranges:          # dict preserves declaration order
            used.add(name)
            if name not in _CANONICAL_ORDER and name not in novel:
                novel.append(name)
    names = [n for n in _CANONICAL_ORDER if n in used] + novel
    GENE_NAMES = names
    GENE_INDEX = {n: i for i, n in enumerate(names)}
    N_GENES = len(names)
    return names


def gene(genomes: np.ndarray, name: str) -> np.ndarray:
    """Column view of one gene across all entities. ``genomes`` is (N, N_GENES)."""
    return genomes[:, GENE_INDEX[name]]


def _bounds(spec: SpeciesConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene (lo, hi) clamp arrays for a species; genes it does not declare pin to their
    neutral value (built-in default, or 0.0 for a novel gene another species introduced)."""
    lo = np.empty(N_GENES, dtype=np.float32)
    hi = np.empty(N_GENES, dtype=np.float32)
    for i, name in enumerate(GENE_NAMES):
        rng = spec.gene_ranges.get(name)
        if rng is None:
            lo[i] = hi[i] = _NEUTRAL.get(name, 0.0)
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
