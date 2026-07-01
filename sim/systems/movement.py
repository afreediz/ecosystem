"""Movement system (§14 of v1.md).

Converts desired headings (action dims 0,1) into new positions with turn-rate-limited
steering (this is where exploration momentum lives -- the stateless brain emits a fresh
random heading each tick, and the limited turn rate turns that into a smooth wander).
Blocks movement into water (ocean + rivers/lakes) / impassable terrain; animals drink
from an adjacent land cell rather than walking onto the water.
Terrain slope scales effective speed. Updates heading arrays for next tick.
"""
from __future__ import annotations

import numpy as np

from sim import genome as gn
from sim.brain import A_DX, A_DY, A_SPEED

_MAX_TURN = 0.7   # radians an agent can rotate toward its desired heading per tick


def _normalize(vx, vy):
    mag = np.sqrt(vx * vx + vy * vy)
    safe = mag > 1e-6
    return (np.where(safe, vx / np.where(safe, mag, 1.0), vx),
            np.where(safe, vy / np.where(safe, mag, 1.0), vy))


def apply(cfg, world, ent, idx, act, rng):
    if idx.shape[0] == 0:
        return
    dt = cfg.sim.dt
    desired_x = act[:, A_DX]
    desired_y = act[:, A_DY]

    cur_x = ent.heading_x[idx]
    cur_y = ent.heading_y[idx]
    cur_x, cur_y = _normalize(cur_x, cur_y)
    dx, dy = _normalize(desired_x, desired_y)

    # turn-rate-limited steering: rotate current heading toward desired by <= _MAX_TURN
    cur_ang = np.arctan2(cur_y, cur_x)
    des_ang = np.arctan2(dy, dx)
    diff = (des_ang - cur_ang + np.pi) % (2 * np.pi) - np.pi   # wrapped to [-pi, pi]
    step = np.clip(diff, -_MAX_TURN, _MAX_TURN)
    new_ang = cur_ang + step
    nhx = np.cos(new_ang).astype(np.float32)
    nhy = np.sin(new_ang).astype(np.float32)

    # speed: gene max_speed with a slight size penalty, scaled by terrain slope
    genome = ent.genome[idx]
    max_speed = gn.gene(genome, "max_speed")
    size = gn.gene(genome, "size")
    speed = max_speed * (1.0 - 0.15 * (size - 1.0))
    speed = np.maximum(speed, 0.05)

    px = ent.pos_x[idx]
    py = ent.pos_y[idx]
    # terrain cost: steeper / higher elevation slows movement
    elev = world.sample(world.elevation, px, py)
    terrain_factor = np.clip(1.0 - 0.5 * elev, 0.3, 1.0)
    # locomotion throttle from the brain (0 = hold position, 1 = full speed). Metabolism
    # charges the move cost in proportion, so easing off genuinely conserves energy.
    throttle = np.clip(act[:, A_SPEED], 0.0, 1.0)
    move = speed * terrain_factor * throttle * dt
    # sleeping animals hold their position (the sleep system already cleared their gates)
    move = np.where(ent.asleep[idx], 0.0, move)

    new_x = px + nhx * move
    new_y = py + nhy * move

    # clamp to world bounds
    new_x = np.clip(new_x, 0.0, world.w - 1e-3)
    new_y = np.clip(new_y, 0.0, world.h - 1e-3)

    # block impassable cells (ocean / high mountain): stay put + reflect heading
    passable = world.is_passable(new_x, new_y)
    blocked = ~passable
    if blocked.any():
        new_x = np.where(blocked, px, new_x)
        new_y = np.where(blocked, py, new_y)
        nhx = np.where(blocked, -nhx, nhx)   # reflect so they wander away from the wall
        nhy = np.where(blocked, -nhy, nhy)

    ent.pos_x[idx] = new_x
    ent.pos_y[idx] = new_y
    ent.heading_x[idx] = nhx
    ent.heading_y[idx] = nhy
