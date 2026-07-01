"""All tunable parameters for the ecosystem simulation, grouped into dataclasses.

This is the single source of truth for every magic number, plus the master seed.
A single ``numpy.random.Generator`` is created here and passed into every system
that needs randomness (see §15 of v1.md) -- there is no global ``np.random`` use.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

# Species ids (also used as indices into per-species config lists).
PLANT = -1  # vegetation lives in a per-cell field, not the entity pool; id for clarity
SHEEP = 0
FOX = 1

SPECIES_NAMES = {SHEEP: "sheep", FOX: "fox"}


@dataclass
class WorldConfig:
    width: int = 208                    # 16:9 (208x117) to match a widescreen display;
    height: int = 117                   # area ~= the old 160x160 so dynamics stay similar
    seed: int = 12345
    noise_octaves: int = 5
    noise_scale: float = 0.012          # base frequency for elevation noise
    moisture_scale: float = 0.02
    sea_level_threshold: float = 0.38   # on normalized elevation [0,1]
    n_river_sources: int = 14
    mountain_threshold: float = 0.80
    cold_threshold: float = 0.30        # on normalized temperature [0,1]
    warm_threshold: float = 0.65
    desert_moisture: float = 0.30
    forest_moisture: float = 0.60
    lapse_rate: float = 0.5             # temp drop per unit elevation
    moisture_boost_radius: float = 12.0  # cells; freshwater raises nearby moisture


@dataclass
class EnvConfig:
    day_length: float = 240.0           # sim-time units per day
    year_length: float = 4800.0         # sim-time units per year (~20 days)
    weather_change_rate: float = 0.02   # prob per tick of a weather transition roll
    diurnal_amp: float = 0.12           # temperature swing amplitude over a day
    seasonal_amp: float = 0.20          # temperature swing amplitude over a year
    nutrient_regen_rate: float = 0.0008  # per dt, toward 1.0
    rain_moisture_boost: float = 0.25
    heat_thirst_factor: float = 1.8     # multiplier on thirst during "heat" weather

    # --- sleep / circadian rhythm (§ diurnal behavior) ---
    # Animals rest at night: as dusk falls they head for a safe spot (cover) and sleep,
    # waking near dawn. Onset/wake are the *population mean* times-of-day; each individual
    # is shifted by its heritable ``chronotype`` gene so they don't all drop at once.
    sleep_onset: float = 0.80           # mean time_of_day when night rest begins (~dusk)
    sleep_wake: float = 0.26            # mean time_of_day when animals wake (~dawn)
    # grace window (fraction of a day) after onset to reach cover before collapsing where
    # they stand -- past this point a straggler sleeps wherever it is (exhaustion).
    sleep_shelter_window: float = 0.06
    sleep_burn_factor: float = 0.45     # metabolic burn multiplier while asleep (resting)
    sleep_need_factor: float = 0.6      # hunger/thirst accumulate slower while asleep


@dataclass
class GeneRange:
    """Inclusive [lo, hi] range a gene is clamped to for a species, plus init spread."""
    lo: float
    hi: float


@dataclass
class SpeciesConfig:
    name: str
    species_id: int
    init_count: int
    # gene ranges (clamp bounds); init values are drawn uniformly within these
    gene_ranges: dict = field(default_factory=dict)
    maturity_age: float = 120.0
    repro_cost: float = 0.30            # energy each parent pays (fraction of capacity)
    repro_cooldown: float = 80.0
    litter_size: int = 1
    hunger_rate: float = 0.006          # hunger increase per dt
    thirst_rate: float = 0.009          # thirst increase per dt (heat-scaled)
    base_burn: float = 0.004            # energy burn per dt at rest
    move_cost: float = 0.020            # extra energy per (speed*size) per dt
    population_cap: int = 1200
    mutation_rate: float = 0.15
    mutation_strength: float = 0.08
    # need thresholds used by the RuleBrain / reproduction
    repro_max_hunger: float = 0.55
    repro_max_thirst: float = 0.55
    eat_value: float = 0.45             # energy gained eating a full veg cell (sheep)
    predation_gain: float = 0.6         # fraction of prey size -> fox energy
    hunt_success: float = 0.2           # base per-tick kill prob (x aggression gene)
    hunt_halfsat: float = 70.0          # Type III response: prey count at half hunt success


def default_species() -> dict:
    sheep_genes = {
        "max_speed":       GeneRange(0.6, 1.8),
        "sensory_range":   GeneRange(8.0, 22.0),
        "metabolism_rate": GeneRange(0.7, 1.3),
        "size":            GeneRange(0.7, 1.4),
        "max_age":         GeneRange(1400.0, 2600.0),
        "repro_threshold": GeneRange(0.5, 0.8),
        "flee_distance":   GeneRange(0.4, 1.0),  # behavioral gene (fraction of range)
        "chronotype":      GeneRange(-0.06, 0.06),  # per-individual sleep-time offset
    }
    fox_genes = {
        "max_speed":       GeneRange(1.0, 2.4),
        "sensory_range":   GeneRange(10.0, 28.0),
        "metabolism_rate": GeneRange(0.7, 1.3),
        "size":            GeneRange(0.9, 1.8),
        "max_age":         GeneRange(1600.0, 3000.0),
        "repro_threshold": GeneRange(0.62, 0.82),
        "aggression":      GeneRange(0.4, 1.0),  # behavioral gene (predation prob)
        "chronotype":      GeneRange(-0.06, 0.06),  # per-individual sleep-time offset
    }
    sheep = SpeciesConfig(
        name="sheep", species_id=SHEEP, init_count=240, gene_ranges=sheep_genes,
        maturity_age=110.0, repro_cost=0.25, repro_cooldown=90.0, litter_size=1,
        hunger_rate=0.0040, thirst_rate=0.0060, base_burn=0.0020, move_cost=0.0045,
        population_cap=1400, mutation_rate=0.18, mutation_strength=0.08,
        repro_max_hunger=0.6, repro_max_thirst=0.6, eat_value=0.9,
    )
    fox = SpeciesConfig(
        name="fox", species_id=FOX, init_count=24, gene_ranges=fox_genes,
        maturity_age=100.0, repro_cost=0.35, repro_cooldown=100.0, litter_size=2,
        # fox metabolism runs leaner than prey (lower burn/hunger) so the predator can ride
        # out prey troughs instead of starving to extinction at every dip -- the key to
        # keeping fox numbers persistent rather than crashing (see v1.md §18). base_burn was
        # eased 0.0012->0.0010 when perception became egocentric GRIDS: the grid's inherent
        # cell-quantization adds small noise to predator pursuit / prey fleeing that tipped
        # the fragile balance to fox extinction (~t3000) on the default seed; the slightly
        # leaner burn gives foxes the endurance to ride it out (verified seeds 12345/7/99).
        hunger_rate=0.0020, thirst_rate=0.0050, base_burn=0.0010, move_cost=0.0010,
        population_cap=430, mutation_rate=0.28, mutation_strength=0.18,
        repro_max_hunger=0.55, repro_max_thirst=0.6, predation_gain=0.72,
        hunt_success=0.5, hunt_halfsat=90.0,
    )
    return {SHEEP: sheep, FOX: fox}


@dataclass
class SimConfig:
    dt: float = 1.0
    grid_cell_size: float = 28.0        # ~ max sensory_range; one bucket per query ring
    max_entities: int = 4000
    log_every: int = 10
    # gene order is fixed across the codebase; see genome.py
    veg_regrow_rate: float = 0.010      # vegetation growth toward carrying capacity per dt
    veg_graze_amount: float = 0.8       # fraction of a cell's veg a sheep takes per bite
    eat_radius: float = 1.6             # adjacency distance for eat/drink/attack
    repro_radius: float = 2.0           # adjacency distance for mating
    food_eat_threshold: float = 0.15    # min vegetation in a cell to be worth eating
    mating_glow_duration: float = 12.0  # ticks a pair stays "rose"-tinted after breeding
                                        # (cosmetic only; read by the viewer, not the sim)


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    species: dict = field(default_factory=default_species)
    # Run / determinism seed: seeds ALL stochastic *dynamics* (population spawn, weather,
    # per-tick decisions / predation / reproduction). It does NOT affect world generation --
    # that depends only on ``world.seed`` (the world seed). ``None`` => a fresh random seed is
    # drawn (and recorded back here) by ``make_rng`` so each run differs; an explicit value
    # makes the run reproducible. So: same world.seed + same config + same seed => identical
    # run; same world.seed but different seed => a different run on the SAME world.
    seed: int | None = None

    def make_rng(self) -> np.random.Generator:
        if self.seed is None:                       # resolve + record a random run seed
            self.seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        return np.random.default_rng(self.seed)


def make_config(world_seed: int | None = None, seed: int | None = None,
                **world_overrides) -> Config:
    """Convenience builder.

    ``world_seed`` -- seeds terrain + hydrology (same world_seed => identical world).
    ``seed``       -- run/determinism seed (see ``Config.seed``); ``None`` => random per run.
    """
    cfg = Config()
    cfg.seed = seed
    if world_seed is not None:
        cfg.world = replace(cfg.world, seed=world_seed)
    if world_overrides:
        cfg.world = replace(cfg.world, **world_overrides)
    return cfg
