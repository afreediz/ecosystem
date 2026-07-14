# Archived: recurrent neural-brain (RL) stack

These files were the **recurrent CNN+MLP+LSTM actor-critic** brain and its reinforcement-learning
trainer. They were detached from the live codebase when deployment moved to the memoryless
imitation-learning policy (`sim/policy_brain.py` + `notebooks/imitation_learning/`).

- **`neural_brain.py`** — `NeuralBrain` + `SpeciesActorCritic` (per-species CNN+MLP+LSTM +
  critic). Originally `sim/neural_brain.py`.
- **`train_neural_brain.py`** — the RL trainer (imitation warm-start → recurrent PPO). Originally
  at the repo root.

Nothing in the live codebase imports these anymore. `run_experiment.py` / `run_live.py` now only
load imitation-learning `PolicyBrain` checkpoints (a `state_dict` key), so the old recurrent
`runs/brain.pt` checkpoints can no longer be deployed while these are archived.

## To restore

1. `git mv backup/neural_brain.py sim/neural_brain.py`
2. `git mv backup/train_neural_brain.py train_neural_brain.py`
3. Re-add the recurrent-checkpoint branch in `run_experiment.py._load_species_brain` (it loaded a
   `NeuralBrain` when the blob had `sheep` + `fox` keys) — see git history for the exact code.

The as-archived files still `import` from their original paths (e.g. `from sim.neural_brain import
NeuralBrain`), so they run only after step 1–2 put them back.
