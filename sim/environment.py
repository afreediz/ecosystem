"""Global time-varying environment: clock, season, weather, temperature (§9 of v1.md).

Holds only scalar/global state. The per-cell static temperature lives on the World;
this module supplies the additive offsets (season + diurnal + weather) and a helper to
produce the full temperature field at the current time.
"""
from __future__ import annotations

import numpy as np

from config import EnvConfig

# weather states
CLEAR, RAIN, HEAT = range(3)
WEATHER_NAMES = {CLEAR: "clear", RAIN: "rain", HEAT: "heat"}


class Environment:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.t = 0.0
        self.weather = CLEAR
        self.time_of_day = 0.0      # [0,1)
        self.season = 0.0           # [0,1)
        self._season_offset = 0.0
        self._diurnal_offset = 0.0
        self._weather_offset = 0.0

    def update(self, dt: float) -> None:
        c = self.cfg
        self.t += dt
        self.time_of_day = (self.t % c.day_length) / c.day_length
        self.season = (self.t % c.year_length) / c.year_length

        # diurnal: coldest before dawn (~0.0), warmest mid-afternoon (~0.6)
        self._diurnal_offset = c.diurnal_amp * np.sin(2 * np.pi * (self.time_of_day - 0.25))
        # seasonal: peak summer at season 0.5
        self._season_offset = c.seasonal_amp * np.sin(2 * np.pi * (self.season - 0.25))

        # weather: stochastic transitions
        if self.rng.random() < c.weather_change_rate:
            self.weather = int(self.rng.integers(0, 3))
        if self.weather == HEAT:
            self._weather_offset = 0.10
        elif self.weather == RAIN:
            self._weather_offset = -0.05
        else:
            self._weather_offset = 0.0

    @property
    def temp_offset(self) -> float:
        return float(self._season_offset + self._diurnal_offset + self._weather_offset)

    def temperature_field(self, static_temp: np.ndarray) -> np.ndarray:
        """Full temperature field at current time = static + global offset."""
        return np.clip(static_temp + self.temp_offset, 0.0, 1.0)

    def thirst_multiplier(self) -> float:
        """Heat (weather + summer) raises thirst rate."""
        m = 1.0 + max(0.0, self.temp_offset) * 1.5
        if self.weather == HEAT:
            m *= self.cfg.heat_thirst_factor
        return float(m)

    def growth_multiplier(self) -> float:
        """Season + weather scaling for plant growth (winter low, rain boosts)."""
        seasonal = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * (self.season - 0.25)))
        if self.weather == RAIN:
            seasonal *= 1.25
        elif self.weather == HEAT:
            seasonal *= 0.85
        return float(seasonal)
