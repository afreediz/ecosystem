"""darwinism -- a headless, deterministic ecosystem + evolution simulation framework.

Plants (a per-cell field), sheep, and foxes live on a noise-generated world with biomes,
hydrology, weather and seasons. Animals act through a ``Brain.decide(obs) -> act`` contract,
carry a heritable genome, and perceive only their local surroundings. The research output is
data: population curves, predator-prey oscillations, and trait drift over generations.

Build around it by importing the public surface below:

    import darwinism as dw

    cfg = dw.make_config(world_seed=12345, seed=7)
    sim = dw.Simulation(cfg)                 # default RuleBrain for every species
    for _ in range(9000):
        stats = sim.step()
    print(sim.populations)

Extend it by declaring new species (``SpeciesConfig`` + ``diet``), plugging in tick-systems
(``System`` / ``StepContext`` / ``default_pipeline``), or subclassing ``Brain``. See
EXTENDING.md. Determinism contract: same ``world_seed`` + ``Config`` + run ``seed`` => an
identical run.
"""
from __future__ import annotations

from darwinism.config import (
    FOX,
    PLANT,
    SHEEP,
    SPECIES_NAMES,
    Config,
    EnvConfig,
    FieldFood,
    GeneRange,
    PreyFood,
    SimConfig,
    SpeciesConfig,
    WorldConfig,
    default_species,
    make_config,
    predators_of,
    prey_of,
)
from darwinism.sim.brain import (
    A_DRINK,
    A_DX,
    A_DY,
    A_EAT,
    A_REPRO,
    A_SPEED,
    ACT_DIM,
    Brain,
    CompositeBrain,
    RuleBrain,
    best_in_channel,
    nearest_in_channel,
)
from darwinism.sim.perception import SCALAR_DIM, Observation
from darwinism.sim.simulation import Simulation
from darwinism.sim.systems import StepContext, System, default_pipeline

__version__ = "1.0.0"

__all__ = [
    # config / world building
    "Config", "make_config", "WorldConfig", "EnvConfig", "SimConfig",
    "SpeciesConfig", "GeneRange", "FieldFood", "PreyFood", "default_species",
    "prey_of", "predators_of",
    "PLANT", "SHEEP", "FOX", "SPECIES_NAMES",
    # simulation
    "Simulation",
    # brain contract
    "Brain", "RuleBrain", "CompositeBrain", "PolicyBrain",
    "ACT_DIM", "A_DX", "A_DY", "A_EAT", "A_DRINK", "A_REPRO", "A_SPEED",
    "nearest_in_channel", "best_in_channel",
    # perception contract
    "Observation", "SCALAR_DIM",
    # tick-system registry (extension point)
    "System", "StepContext", "default_pipeline",
    "__version__",
]


def __getattr__(name):
    """Lazily expose the learned brain so ``import darwinism`` never requires torch.

    ``PolicyBrain`` lives behind the optional ``[torch]`` extra; accessing it imports the
    module (and torch) on demand, with the module's own ImportError surfaced if torch is
    missing.
    """
    if name in ("PolicyBrain", "policy_brain_from_path"):
        from darwinism.sim import policy_brain as _pb
        return getattr(_pb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
