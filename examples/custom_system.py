"""Add a custom tick-SYSTEM to the pipeline -- no core edits.

A ``System`` reads/writes the shared ``StepContext`` each tick. Here a toy "drought" system
periodically knocks down the vegetation field, tightening the food supply. It is inserted
right after vegetation growth so its effect is visible the same tick.

Systems can read anything on the context: entity SoA arrays (ctx.ent), the world, the run
RNG (ctx.rng), per-tick observations/actions (ctx.obs / ctx.act / ctx.idx), etc. This is also
how you'd read a heritable gene a new species declares (via ctx.ent.genome + genome.gene).

    venv/Scripts/python.exe examples/custom_system.py
"""
import darwinism as dw


class DroughtSystem(dw.System):
    """Every ``period`` ticks, scale the whole vegetation field down by ``severity``."""

    def __init__(self, period=400, severity=0.6):
        self.period = period
        self.severity = severity

    def apply(self, ctx):
        if ctx.tick % self.period == 0:
            ctx.veg *= self.severity            # ctx.veg is the live per-cell field (mutated in place)


cfg = dw.make_config(world_seed=12345, seed=7)

# default pipeline + our system appended after VegetationSystem (index -1 is StatsSystem, so
# insert before it). You can also replace or reorder entries; just keep the RNG-drawing
# systems (movement/consumption/metabolism/reproduction) in their relative order.
pipeline = dw.default_pipeline(cfg)
pipeline.insert(-1, DroughtSystem(period=400, severity=0.6))

sim = dw.Simulation(cfg, systems=pipeline)
print("pipeline:", [type(s).__name__ for s in sim.systems])

for tick in range(2000):
    stats = sim.step()
    if (tick + 1) % 400 == 0:
        print(f"tick {tick + 1:>5}  veg={stats['veg_biomass']:>7.0f}  {sim.populations}")
