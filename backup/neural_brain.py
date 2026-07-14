"""Neural network brain: CNN perception + scalar sensing + LSTM memory (§7.1, §13, §21).

A drop-in replacement for ``RuleBrain`` behind the exact same contract
(``decide(obs_by_species, idx) -> act``). Where the rule brain *decodes* the perception
grids back into hand-picked targets and arbitrates them with if/else priorities, this
brain learns straight off the raw channels:

  grids (N, C, K, K)  --CNN-->  spatial feature
  scalars (N, 10)     --MLP-->  interoceptive feature   (health, hunger, thirst, energy,
                                                         age, sex, temperature, time, ...)
  [spatial | intero]  --LSTMCell-->  recurrent memory   (per-agent, carried across ticks)
  memory              --actor heads-->  action  (heading gaussian, eat/drink/repro
                                                  bernoulli gates, speed beta throttle)
  memory              --critic head-->  state value      (used only by the RL trainer)

Two independent networks are held, one per species, because sheep and fox perception
layouts differ (5 vs 4 channels, see ``sim/perception.py``) -- exactly the per-species
CNN the grid design was built for.

MEMORY LIFECYCLE. The decision itself still reads ONLY the observation, honouring the
brain<->world contract. The LSTM hidden state is the brain's *own* internal memory, kept
in a per-slot table (``h``/``c`` of shape ``(cap, hidden)``). Because entity slots are
recycled by the free list, the brain resets a slot's memory the moment a *different* animal
occupies it, detected via ``entities.birth_id`` (bound once by the Simulation). Reading the
identity token is pure lifecycle bookkeeping -- it never enters the policy input.

DETERMINISM. In eval mode (``training=False``) the brain acts by the distribution *mode*
(no sampling), so it draws no randomness at all: a run is reproducible from
(world seed, run seed, weights) just like the rule brain, and it does not perturb the numpy
run-RNG stream the other systems consume. In training mode it samples (via the global torch
RNG, which the trainer seeds) and records each decision for PPO.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta, Normal

from config import SHEEP, FOX
from sim.brain import ACT_DIM, Brain
from sim.perception import SCALAR_DIM, SPECIES_N_CHANNELS

# action-vector layout is shared with the rule brain; imported for clarity in the heads
# (columns: A_DX, A_DY, A_EAT, A_DRINK, A_REPRO, A_SPEED -- see sim/brain.py)

_SPEED_EPS = 1e-4        # keep the Beta speed sample strictly inside (0, 1) for a finite log-prob


# --------------------------------------------------------------------------- network
class SpeciesActorCritic(nn.Module):
    """Per-species recurrent actor-critic.

    Forward consumes one tick of a batch of agents plus their carried LSTM state and returns
    the action-distribution parameters, the state value, and the next LSTM state. The CNN
    ends in an adaptive pool so the same architecture accepts any window size ``K`` (which is
    set by the largest sensory range in the config, so it can differ between runs).
    """

    def __init__(self, n_channels: int, scalar_dim: int = SCALAR_DIM,
                 hidden: int = 128, cnn_feat: int = 128, scalar_feat: int = 32):
        super().__init__()
        self.hidden = hidden

        # --- CNN over the egocentric perception grids ---
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),      # -> (32, 4, 4) regardless of input K
        )
        self.cnn_fc = nn.Linear(32 * 4 * 4, cnn_feat)

        # --- MLP over the interoceptive / global scalar vector ---
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_feat), nn.ReLU(inplace=True),
            nn.Linear(scalar_feat, scalar_feat), nn.ReLU(inplace=True),
        )

        # --- recurrent memory ---
        self.lstm = nn.LSTMCell(cnn_feat + scalar_feat, hidden)

        # --- actor heads ---
        self.head_mean = nn.Linear(hidden, 2)              # heading gaussian mean (dx, dy)
        self.head_logstd = nn.Parameter(torch.full((2,), -0.5))  # state-independent log std
        self.gate_logits = nn.Linear(hidden, 3)            # eat / drink / repro bernoulli
        self.speed_ab = nn.Linear(hidden, 2)               # speed beta (alpha, beta) pre-softplus

        # --- critic head ---
        self.value_head = nn.Linear(hidden, 1)

    def features(self, grids: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        c = self.conv(grids).flatten(1)
        c = F.relu(self.cnn_fc(c))
        s = self.scalar_mlp(scalars)
        return torch.cat([c, s], dim=1)

    def forward(self, grids, scalars, h, c):
        """Returns (params, value, h_next, c_next).

        ``params`` = (mean, logstd, gate_logits, speed_ab) with speed_ab already mapped to
        concentrations >= 1 (a unimodal Beta). ``value`` is (N,).
        """
        x = self.features(grids, scalars)
        h_next, c_next = self.lstm(x, (h, c))
        mean = self.head_mean(h_next)
        logstd = self.head_logstd.clamp(-2.0, 1.0).expand_as(mean)
        gate_logits = self.gate_logits(h_next)
        speed_ab = F.softplus(self.speed_ab(h_next)) + 1.0     # alpha, beta >= 1
        value = self.value_head(h_next).squeeze(-1)
        return (mean, logstd, gate_logits, speed_ab), value, h_next, c_next


# --------------------------------------------------------------- action distribution helpers
# The action space is hybrid: a 2-D gaussian heading (movement normalizes it to a unit
# direction, so the magnitude is a free exploration dimension), 3 independent bernoulli gates,
# and a Beta throttle in (0, 1). Log-probs and entropies sum across the three factors.

def _dists(params):
    mean, logstd, gate_logits, speed_ab = params
    heading = Normal(mean, logstd.exp())
    gates = Bernoulli(logits=gate_logits)
    speed = Beta(speed_ab[:, 0], speed_ab[:, 1])
    return heading, gates, speed


def action_logp_entropy(params, actions):
    """Log-prob and entropy of ``actions`` (N, ACT_DIM) under the policy ``params``."""
    heading, gates, speed = _dists(params)
    lp = heading.log_prob(actions[:, 0:2]).sum(-1)
    lp = lp + gates.log_prob(actions[:, 2:5]).sum(-1)
    sp = actions[:, 5].clamp(_SPEED_EPS, 1.0 - _SPEED_EPS)
    lp = lp + speed.log_prob(sp)
    ent = heading.entropy().sum(-1) + gates.entropy().sum(-1) + speed.entropy()
    return lp, ent


def sample_action(params):
    """Stochastic action (for training rollouts). Returns (actions (N,6), logp (N,))."""
    heading, gates, speed = _dists(params)
    head = heading.sample()
    gate = gates.sample()
    sp = speed.sample().clamp(_SPEED_EPS, 1.0 - _SPEED_EPS)
    actions = torch.cat([head, gate, sp.unsqueeze(-1)], dim=1)
    lp, _ = action_logp_entropy(params, actions)
    return actions, lp


def mode_action(params):
    """Deterministic action (for eval / deployment). No randomness is drawn."""
    mean, logstd, gate_logits, speed_ab = params
    gate = (gate_logits > 0.0).float()                     # sigmoid > 0.5  <=>  logit > 0
    speed = speed_ab[:, 0] / (speed_ab[:, 0] + speed_ab[:, 1])   # Beta mean (always finite)
    return torch.cat([mean, gate, speed.unsqueeze(-1)], dim=1)


# --------------------------------------------------------------------------- the brain
class NeuralBrain(Brain):
    """Learned brain implementing the ``Brain`` contract with per-agent LSTM memory."""

    def __init__(self, cfg, device: str = "cpu", hidden: int = 128,
                 training: bool = False):
        self.cfg = cfg
        self.device = torch.device(device)
        self.hidden = hidden
        self.training = training           # sample + record when True; act by mode when False
        self.recorder = None               # set by the RL trainer to capture rollouts

        self.nets = {
            sid: SpeciesActorCritic(SPECIES_N_CHANNELS[sid], hidden=hidden).to(self.device)
            for sid in (SHEEP, FOX)
        }

        cap = cfg.sim.max_entities
        # per-slot LSTM state tables (one per species; a slot only ever holds one species at a
        # time, so the tables never collide)
        self.h = {sid: torch.zeros(cap, hidden, device=self.device) for sid in (SHEEP, FOX)}
        self.c = {sid: torch.zeros(cap, hidden, device=self.device) for sid in (SHEEP, FOX)}
        # identity token whose memory currently lives in each slot; a mismatch vs
        # entities.birth_id means the slot was recycled -> reset that memory.
        self._occupant = np.zeros(cap, dtype=np.int64)
        self.entities = None               # bound by Simulation for identity-based resets

    # ------------------------------------------------------------------ wiring
    def bind(self, entities) -> None:
        """Give the brain a handle on the entity store, used ONLY to detect slot recycling
        (via ``birth_id``) so an animal never inherits a dead predecessor's LSTM memory."""
        self.entities = entities

    def eval(self) -> "NeuralBrain":
        self.training = False
        for net in self.nets.values():
            net.eval()
        return self

    def train_mode(self) -> "NeuralBrain":
        self.training = True
        for net in self.nets.values():
            net.train()
        return self

    def parameters(self, sid: int):
        return self.nets[sid].parameters()

    # ------------------------------------------------------------------ memory lifecycle
    def _sync_memory(self, sid: int, slots: np.ndarray) -> None:
        """Zero the LSTM state of any slot now occupied by a new animal (or first use)."""
        if self.entities is not None:
            cur = self.entities.birth_id[slots]
            changed = cur != self._occupant[slots]
            if changed.any():
                ch = slots[changed]
                ch_t = torch.as_tensor(ch, device=self.device, dtype=torch.long)
                self.h[sid][ch_t] = 0.0
                self.c[sid][ch_t] = 0.0
                self._occupant[ch] = cur[changed]
        # if no entity handle is bound we simply keep whatever memory the slot holds; the
        # Simulation always binds, so this fallback only matters for isolated unit tests.

    def reset(self) -> None:
        """Wipe ALL memory (e.g. when a training episode restarts the world)."""
        for sid in (SHEEP, FOX):
            self.h[sid].zero_()
            self.c[sid].zero_()
        self._occupant[:] = 0

    # ------------------------------------------------------------------ the contract
    def decide(self, obs_by_species, idx) -> np.ndarray:
        n_global = idx.shape[0]
        act = np.zeros((n_global, ACT_DIM), dtype=np.float32)
        if n_global == 0:
            return act
        for sid in (SHEEP, FOX):
            obs = obs_by_species.get(sid)
            if obs is None or obs.grids.shape[0] == 0:
                continue
            self._decide_species(sid, obs, idx, act)
        return act

    def _decide_species(self, sid: int, obs, idx: np.ndarray, act: np.ndarray) -> None:
        net = self.nets[sid]
        slots = obs.idx                                   # global slot ids of this species
        self._sync_memory(sid, slots)

        grids = torch.from_numpy(np.ascontiguousarray(obs.grids)).to(self.device)
        scalars = torch.from_numpy(np.ascontiguousarray(obs.scalars)).to(self.device)
        st = torch.as_tensor(slots, device=self.device, dtype=torch.long)
        h0, c0 = self.h[sid][st], self.c[sid][st]

        with torch.no_grad():
            params, value, h1, c1 = net(grids, scalars, h0, c0)
            if self.training:
                actions, logp = sample_action(params)
            else:
                actions, logp = mode_action(params), None

        # carry the recurrent state forward for these agents
        self.h[sid][st] = h1
        self.c[sid][st] = c1

        a = actions.cpu().numpy().astype(np.float32)
        pos = np.searchsorted(idx, slots)                 # rows of this species in global act
        act[pos] = a

        if self.training and self.recorder is not None:
            # hand over the LSTM state that ENTERED this tick (h0/c0) too: the trainer stores
            # it as the initial state of each truncated-BPTT window so the update replays from
            # the memory the agent actually had, not from zero (correct recurrent PPO).
            self.recorder.record(sid, slots, obs.grids, obs.scalars, a,
                                 logp.cpu().numpy(), value.cpu().numpy(),
                                 h0.cpu().numpy(), c0.cpu().numpy())

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {
            "hidden": self.hidden,
            "sheep": self.nets[SHEEP].state_dict(),
            "fox": self.nets[FOX].state_dict(),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> "NeuralBrain":
        blob = torch.load(path, map_location=self.device)
        self.nets[SHEEP].load_state_dict(blob["sheep"], strict=strict)
        self.nets[FOX].load_state_dict(blob["fox"], strict=strict)
        return self
