"""Minimal headless run: build a world, step the default sheep+fox ecosystem, read stats.

    venv/Scripts/python.exe examples/headless_run.py
"""
import darwinism as dw

# same world_seed + config + run seed => byte-identical run
cfg = dw.make_config(world_seed=12345, seed=7)
sim = dw.Simulation(cfg)                      # default RuleBrain drives every species

for tick in range(2000):
    stats = sim.step()
    if (tick + 1) % 500 == 0:
        print(f"tick {tick + 1:>5}  populations={sim.populations}  "
              f"veg={stats['veg_biomass']:.0f}")

print("\nfinal:", sim.populations)
# mean heritable traits of the surviving sheep (the evolution signal)
print("sheep trait means:", {k: round(v, 3) for k, v in sim.trait_means(dw.SHEEP).items()})
