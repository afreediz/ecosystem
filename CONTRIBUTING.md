# Contributing to darwinism

Thanks for your interest! This is a small, deterministic simulation framework; a few
conventions keep it fast to work in and keep runs reproducible.

## Dev setup

```bash
python -m venv venv
venv/Scripts/activate                 # Windows;  source venv/bin/activate on Unix
pip install -e ".[all,dev]"
```

This installs the package in editable mode plus the dev tools (`pytest`, `ruff`,
`import-linter`). Use `venv/Scripts/python.exe` on Windows.

## Tests

```bash
python -m pytest                                     # full suite
python -m pytest tests/test_determinism.py           # golden-master (determinism)
python -m pytest "tests/test_determinism.py::test_matches_golden[7]"   # fast single seed
```

The suite's backbone is a **golden-master determinism** test: the default (sheep + fox) config
must stay **byte-identical** to a frozen baseline (`tests/baselines/golden.json`, CSV +
entity-state hashes across seeds 7/99/12345). Run it after every change. If you deliberately,
knowingly change the default-config dynamics, re-capture the baseline
(`python tests/capture_baselines.py`) — the diff in the committed JSON is the audit trail.

`tests/test_extensions.py` proves the extension path (a new species with a novel trait runs,
is self-consistent, and is reproducible).

## Lint

```bash
ruff check .
ruff format .
lint-imports          # enforces the architecture invariant below
```

## Architecture invariants (please don't break)

1. **`darwinism.sim` never imports `darwinism.render`.** The sim core is pure numbers; the
   renderer is an optional, read-only observer. This is machine-enforced by `import-linter`
   (config in `pyproject.toml`) — run `lint-imports`.
2. **Determinism, two seeds.** `world.seed` drives world generation only; `Config.seed` drives
   all stochastic dynamics via one `numpy.random.Generator`. No global `np.random`. Same
   `world_seed` + `Config` + run `seed` ⇒ byte-identical run.
3. **Iterate species in `sorted(cfg.species)` order** wherever order is observable, and don't
   reorder the RNG-drawing systems (movement/consumption/metabolism/reproduction).
4. **Structure-of-Arrays.** Entity state is parallel NumPy arrays indexed by slot
   (`darwinism/sim/entities.py`) — never one object per entity.
5. **The `Brain.decide(obs_by_species, idx) -> act` contract is the spine.** The brain sees
   only observations; systems enforce the authoritative world conditions.

See **[EXTENDING.md](EXTENDING.md)** for how the extension points work, and **[CLAUDE.md](CLAUDE.md)**
for the (fragile) predator–prey calibration notes — retune those constants gently and always
re-run a long (8000-tick) simulation before trusting a change.

## Pull requests

- Keep changes focused; match the surrounding code's style and comment density.
- Run `pytest` (determinism must stay green) and `ruff check` before opening a PR.
- If you touched calibration-sensitive constants, note the long-run population effect.
