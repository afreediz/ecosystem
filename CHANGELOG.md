# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) (0.x = the public API may still change).

## [0.1.0] — 2026-07-21

First framework release: the simulation is now an installable, importable package with
documented extension points, while the default sheep + fox world stays **byte-identical** to
the pre-framework version (verified to fox extinction at 6905 ticks on seed 7, and via a
golden-master determinism suite across seeds 7/99/12345).

### Added
- **Packaging.** `pip install`-able `darwinism` package (flat layout, hatchling backend) with a
  curated public API (`darwinism/__init__.py`), `__version__`, and PEP 561 `py.typed`. Optional
  extras: `[analysis]`, `[render]`, `[torch]`, `[dev]`, `[all]`. Console scripts `darwinism-run`
  / `darwinism-live` (plus `python -m darwinism` and back-compat root shims).
- **Declarative species.** `SpeciesConfig` now carries a `diet` (`FieldFood` / `PreyFood`) and
  `cluster`; predation relationships (`prey_of` / `predators_of`) are derived from diet. Add a
  new species as pure config — perception, genome, systems, and stats all adapt.
- **Runtime genome registry.** The gene layout is built from the registered species
  (`genome.build_registry`), so a new species can introduce a novel heritable trait without
  editing the core.
- **Systems registry.** The tick is an ordered list of `System` objects over a shared
  `StepContext` (`darwinism.sim.systems.pipeline`, `default_pipeline`); insert/replace/reorder
  tick-systems via `Simulation(systems=...)`.
- **Self-describing observations.** `Observation.channels` maps role → grid channel index, so a
  brain reads channels by role rather than hardcoded indices. `nearest_in_channel` /
  `best_in_channel` decode helpers are public.
- **Determinism test suite** (`tests/`): golden-master (CSV + entity-state hashes) and an
  extension smoke test (new species with a novel trait). `import-linter` enforces the
  `sim`-must-not-import-`render` invariant.
- **Docs**: `EXTENDING.md`, runnable `examples/` (species / system / brain), `CONTRIBUTING.md`,
  and a reworked `README.md`.

### Changed
- Internal modules moved under the `darwinism/` package; all imports updated. The two-species
  `(SHEEP, FOX)` hardcoding throughout perception, brain, consumption, metabolism,
  reproduction, seeding, grids, stats, and the logger was replaced with iteration over
  `sorted(cfg.species)` — behaviour-preserving for the default config.

### Notes
- Predator–prey coexistence remains fragile and seed-dependent (see `CLAUDE.md` §Calibration);
  on some seeds foxes still go extinct in a deep trough. This is unchanged from before the
  refactor.
