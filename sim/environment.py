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


def season_name(season: float) -> str:
    """Season label, anchored to the temperature model (summer warmest at 0.5,
    winter coldest at 0.0/1.0, spring rising at 0.25, autumn falling at 0.75)."""
    s = season % 1.0
    if s < 0.125 or s >= 0.875:
        return "winter"
    if s < 0.375:
        return "spring"
    if s < 0.625:
        return "summer"
    return "autumn"


def daytime_name(time_of_day: float) -> str:
    """Time-of-day label. t=0.0 is pre-dawn (coldest), 0.25 sunrise, ~0.55 midday,
    ~0.75 sunset (see the diurnal temperature curve below)."""
    t = time_of_day % 1.0
    if t < 0.20:
        return "night"
    if t < 0.30:
        return "dawn"
    if t < 0.45:
        return "morning"
    if t < 0.55:
        return "noon"
    if t < 0.75:
        return "afternoon"
    if t < 0.85:
        return "dusk"
    return "night"


def _smoothstep(a: float, b: float, x: float) -> float:
    """Hermite smoothstep: 0 below ``a``, 1 above ``b``, smooth in between."""
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def light_level(time_of_day: float) -> float:
    """Daylight in [0,1]: ~0 at deep night, 1 at midday, smooth dawn/dusk transitions.

    Cosmetic only (the viewer dims the scene by ``1 - light_level``); it is NOT read by
    the sim. Kept here so the visual darkening lines up with the diurnal temperature curve
    and the animals' sleep window (dusk ~0.78, dawn ~0.26)."""
    t = time_of_day % 1.0
    dawn = _smoothstep(0.16, 0.30, t)        # brighten through sunrise
    dusk = 1.0 - _smoothstep(0.74, 0.88, t)  # darken through sunset
    return float(dawn * dusk)


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

        # season is normally driven by ``t`` but can be paused or nudged forward at
        # runtime (live viewer controls). With no manual input these stay neutral, so a
        # headless run is byte-for-byte identical to the original t-derived season.
        self.season_paused = False
        self._season_phase = 0.0    # auto-advancing fraction of a year
        self._season_shift = 0.0    # manual forward offset

    def update(self, dt: float) -> None:
        c = self.cfg
        self.t += dt
        self.time_of_day = (self.t % c.day_length) / c.day_length
        if not self.season_paused:
            self._season_phase += dt / c.year_length
        self.season = (self._season_phase + self._season_shift) % 1.0

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

    # ------------------------------------------------------------------ season control
    def advance_season(self, amount: float = 0.1) -> None:
        """Push the season forward by ``amount`` of a year (wraps at 1.0)."""
        self._season_shift = (self._season_shift + amount) % 1.0

    def toggle_season_pause(self) -> bool:
        """Freeze/unfreeze seasonal progression (day & weather keep running)."""
        self.season_paused = not self.season_paused
        return self.season_paused

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
