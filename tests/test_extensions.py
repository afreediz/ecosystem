"""Extension-path tests: prove a developer can add a NEW species (with a novel heritable
trait) purely as config, and that perception / genome / systems / stats all adapt with no
core edits. A new species is a NEW config, so it is exempt from the golden-master baseline --
these tests instead assert it runs, is self-consistent, and is reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import darwinism as dw                      # noqa: E402
from darwinism.sim import genome as gn      # noqa: E402
from determinism_util import state_hash     # noqa: E402

RABBIT = 2


def _three_species_cfg(seed: int) -> dw.Config:
    """Default sheep + fox, plus a rabbit: a herbivore with a NOVEL 'burrow_depth' gene that
    foxes also hunt (so it exercises multi-prey predation and the new species' threat channel)."""
    cfg = dw.make_config(world_seed=12345, seed=seed)
    rabbit = dw.SpeciesConfig(
        name="rabbit", species_id=RABBIT, init_count=80,
        diet=[dw.FieldFood(field="vegetation", eat_value=0.7)],
        cluster=(5, 5.0),
        gene_ranges={
            "max_speed": dw.GeneRange(0.8, 2.2),
            "sensory_range": dw.GeneRange(6.0, 18.0),
            "metabolism_rate": dw.GeneRange(0.7, 1.3),
            "size": dw.GeneRange(0.4, 0.9),
            "max_age": dw.GeneRange(1000.0, 2000.0),
            "repro_threshold": dw.GeneRange(0.45, 0.75),
            "burrow_depth": dw.GeneRange(0.0, 1.0),          # NOVEL trait, not built-in
            "chronotype": dw.GeneRange(-0.06, 0.06),
        },
        maturity_age=80.0, repro_cost=0.2, repro_cooldown=70.0, litter_size=3,
        hunger_rate=0.0045, thirst_rate=0.0022, base_burn=0.0022, move_cost=0.005,
        population_cap=800, mutation_rate=0.2, mutation_strength=0.08,
        repro_max_hunger=0.6, repro_max_thirst=0.6,
    )
    cfg.species[RABBIT] = rabbit
    cfg.species[dw.FOX].diet[0].prey.append(RABBIT)     # foxes hunt sheep AND rabbits
    return cfg


def test_new_species_runs_and_is_self_consistent():
    cfg = _three_species_cfg(7)
    sim = dw.Simulation(cfg)

    # the novel gene joined the runtime registry, appended after the built-in genes
    assert "burrow_depth" in gn.GENE_NAMES
    assert gn.GENE_NAMES.index("burrow_depth") >= 9

    # relationships derived from diet: fox now preys on sheep + rabbit; rabbit fears fox
    assert set(cfg.prey_of()[dw.FOX]) == {dw.SHEEP, RABBIT}
    assert cfg.predators_of()[RABBIT] == [dw.FOX]

    for _ in range(150):
        sim.step()

    # stats + populations gained the new species automatically
    assert set(sim.populations) == {"sheep", "fox", "rabbit"}
    assert "n_rabbit" in sim.stats
    assert sim.populations["rabbit"] > 0                # founders survive 150 ticks

    # perception built a schema for the rabbit; being hunted, it carries a threat channel
    rabbit_obs = sim.last_obs[RABBIT]
    assert "threat" in rabbit_obs.channels
    assert rabbit_obs.food_reduction == "best"          # grazer
    # the fox (apex) still has no threat channel
    assert "threat" not in sim.last_obs[dw.FOX].channels


def test_new_species_reproducible():
    def run(seed):
        sim = dw.Simulation(_three_species_cfg(seed))
        for _ in range(60):
            sim.step()
        return state_hash(sim)

    assert run(7) == run(7)                              # same config => identical run
    assert run(7) != run(99)                             # different seed => different run
