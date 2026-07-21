"""Simulation systems. Each operates on the SoA entity store + world fields.

The tick pipeline is a list of ``System`` objects sharing a per-tick ``StepContext``; see
``darwinism.sim.systems.pipeline``. Import the registry surface from here."""

from darwinism.sim.systems.pipeline import (
    BrainSystem,
    ConsumptionSystem,
    EnvironmentSystem,
    GridSystem,
    MetabolismSystem,
    MovementSystem,
    PerceptionSystem,
    ReproductionSystem,
    SleepSystem,
    StatsSystem,
    StepContext,
    System,
    VegetationSystem,
    default_pipeline,
)

__all__ = [
    "StepContext", "System", "default_pipeline",
    "EnvironmentSystem", "GridSystem", "PerceptionSystem", "BrainSystem", "SleepSystem",
    "MovementSystem", "ConsumptionSystem", "MetabolismSystem", "ReproductionSystem",
    "VegetationSystem", "StatsSystem",
]
