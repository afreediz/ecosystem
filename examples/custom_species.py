"""Add a THIRD species with a novel heritable trait -- purely as config, no core edits.

A ``rabbit`` is a small, fast herbivore (grazes the vegetation field) with a novel
``burrow_depth`` gene, and foxes are extended to hunt rabbits as well as sheep. Perception
channels, the genome layout, the consumption/reproduction systems, and the stats/logger all
adapt automatically from the species declaration.

    venv/Scripts/python.exe examples/custom_species.py
"""
import darwinism as dw
from darwinism.sim import genome as gn

RABBIT = 2

rabbit = dw.SpeciesConfig(
    name="rabbit", species_id=RABBIT, init_count=90,
    # DIET: grazing a per-cell world field (herbivore). PreyFood would make it a hunter.
    diet=[dw.FieldFood(field="vegetation", eat_value=0.7)],
    cluster=(5, 5.0),                        # founder herds: (n_clusters, spread)
    gene_ranges={
        "max_speed":       dw.GeneRange(0.8, 2.2),
        "sensory_range":   dw.GeneRange(6.0, 18.0),
        "metabolism_rate": dw.GeneRange(0.7, 1.3),
        "size":            dw.GeneRange(0.4, 0.9),
        "max_age":         dw.GeneRange(1000.0, 2000.0),
        "repro_threshold": dw.GeneRange(0.45, 0.75),
        "burrow_depth":    dw.GeneRange(0.0, 1.0),   # <-- NOVEL trait; joins the genome registry
        "chronotype":      dw.GeneRange(-0.06, 0.06),
    },
    maturity_age=80.0, repro_cost=0.2, repro_cooldown=70.0, litter_size=3,
    hunger_rate=0.0045, thirst_rate=0.0022, base_burn=0.0022, move_cost=0.005,
    population_cap=800, mutation_rate=0.2, mutation_strength=0.08,
    repro_max_hunger=0.6, repro_max_thirst=0.6,
)

cfg = dw.make_config(world_seed=12345, seed=7)
cfg.species[RABBIT] = rabbit
cfg.species[dw.FOX].diet[0].prey.append(RABBIT)     # foxes now hunt sheep AND rabbits

sim = dw.Simulation(cfg)     # building the Simulation finalises the gene registry
print("gene registry:", ", ".join(gn.GENE_NAMES))   # note 'burrow_depth' appended at the end
print("prey_of fox:", cfg.prey_of()[dw.FOX], " predators_of rabbit:", cfg.predators_of()[RABBIT])

for tick in range(3000):
    sim.step()
    if (tick + 1) % 1000 == 0:
        print(f"tick {tick + 1:>5}  {sim.populations}")

print("\nfinal:", sim.populations)
