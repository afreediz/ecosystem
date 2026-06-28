"""Arcade observer window (§16 of v1.md).

Holds a Simulation, calls ``sim.step(dt)`` in ``on_update``, and draws it. It is an
OBSERVER ONLY -- it never mutates sim state. The world is rendered once into a background
texture; entities are drawn each frame as batched point clouds (one call per species).

Controls:
  SPACE        pause/resume            UP/DOWN     steps-per-frame +/-
  +/- or wheel zoom in/out             drag mouse  pan
  WASD         pan                     0           reset view (fit whole map)
  V            toggle vegetation overlay tint
  ESC          quit
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import arcade

from config import Config, SHEEP, FOX
from sim.simulation import Simulation
from sim.world import BIOME_COLORS


class EcosystemViewer(arcade.Window):
    scale = 4

    def __init__(self, cfg: Config | None = None, scale: int = 4, steps_per_frame: int = 1):
        cfg = cfg or Config()
        self.sim = Simulation(cfg)
        self.scale = scale

        # Full world size in content pixels. The window is capped to fit the display so
        # the whole map is never larger than the screen; the camera zoom (below) handles
        # showing more or less of it.
        self._content_w = self.sim.world.w * scale
        self._content_h = self.sim.world.h * scale

        disp_w, disp_h = arcade.get_display_size()
        max_w, max_h = int(disp_w * 0.9), int(disp_h * 0.9)
        win_w = min(self._content_w, max_w)
        win_h = min(self._content_h, max_h)
        super().__init__(win_w, win_h, "Ecosystem + Evolution (v1)", resizable=True)
        self.background_color = arcade.color.BLACK

        self.paused = False
        self.steps_per_frame = steps_per_frame
        self.show_veg = True

        # World camera (zoom/pan) for terrain + entities; GUI camera (fixed) for the HUD.
        self.world_camera = arcade.camera.Camera2D()
        self.gui_camera = arcade.camera.Camera2D()
        self._reset_view()

        self._bg_texture = self._build_background_texture()
        self._veg_dirty = True

    # ------------------------------------------------------------------ camera
    def _fit_zoom(self) -> float:
        """Zoom at which the entire world fits inside the current window."""
        return min(self.width / self._content_w, self.height / self._content_h)

    def _reset_view(self):
        self.world_camera.zoom = self._fit_zoom()
        self.world_camera.position = (self._content_w / 2, self._content_h / 2)

    def _zoom_at(self, screen_x: float, screen_y: float, factor: float):
        fit = self._fit_zoom()
        before = self.world_camera.unproject((screen_x, screen_y))
        new_zoom = max(fit * 0.5, min(fit * 40.0, self.world_camera.zoom * factor))
        self.world_camera.zoom = new_zoom
        after = self.world_camera.unproject((screen_x, screen_y))
        px, py = self.world_camera.position
        self.world_camera.position = (px + (before.x - after.x), py + (before.y - after.y))

    # ------------------------------------------------------------------ background
    def _build_background_texture(self) -> arcade.Texture:
        world = self.sim.world
        h, w = world.h, world.w

        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for bid, color in BIOME_COLORS.items():
            rgb[world.biome == bid] = color

        rgb[world.freshwater] = (60, 130, 220)

        img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        return arcade.Texture(img)

    # ------------------------------------------------------------------ loop
    def on_update(self, dt: float):
        if self.paused:
            return
        for _ in range(self.steps_per_frame):
            self.sim.step()

    def _world_to_screen(self, x: np.ndarray, y: np.ndarray):
        sx = x * self.scale
        sy = (self.sim.world.h - y) * self.scale   # flip Y: world row 0 at top
        return sx, sy

    def on_draw(self):
        self.clear()

        self.world_camera.use()
        # background terrain (drawn in content pixel space; the camera scales it)
        arcade.draw_texture_rect(
            self._bg_texture, arcade.LBWH(0, 0, self._content_w, self._content_h))

        ent = self.sim.entities
        # sheep
        sheep = np.nonzero(ent.species_mask(SHEEP))[0]
        if sheep.shape[0]:
            sx, sy = self._world_to_screen(ent.pos_x[sheep], ent.pos_y[sheep])
            pts = list(zip(sx.tolist(), sy.tolist()))
            arcade.draw_points(pts, (245, 245, 245), max(2.0, self.scale * 0.8))
        # fox
        fox = np.nonzero(ent.species_mask(FOX))[0]
        if fox.shape[0]:
            fx, fy = self._world_to_screen(ent.pos_x[fox], ent.pos_y[fox])
            pts = list(zip(fx.tolist(), fy.tolist()))
            arcade.draw_points(pts, (220, 70, 40), max(3.0, self.scale * 1.1))

        self.gui_camera.use()
        self._draw_hud()

    def _draw_hud(self):
        s = self.sim.stats
        env = self.sim.env
        from sim.environment import WEATHER_NAMES
        lines = [
            f"tick {self.sim.tick}   {'PAUSED' if self.paused else f'x{self.steps_per_frame}'}"
            f"   zoom {self.world_camera.zoom / self._fit_zoom():.1f}x",
            f"sheep {s.get('n_sheep', 0)}   fox {s.get('n_fox', 0)}   veg {s.get('veg_biomass', 0):.0f}",
            f"day {env.time_of_day:.2f}  season {env.season:.2f}  weather {WEATHER_NAMES[env.weather]}",
            f"births {s.get('births', 0)}  deaths {s.get('deaths', 0)} "
            f"(pred {s.get('death_predation', 0)})",
        ]
        y = self.height - 18
        for ln in lines:
            arcade.draw_text(ln, 8, y, (255, 255, 255), 12,
                             font_name=("consolas", "monospace"))
            y -= 16

    # ------------------------------------------------------------------ input
    def on_resize(self, width: int, height: int):
        super().on_resize(width, height)
        self.world_camera.match_window()
        self.gui_camera.match_window()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self._zoom_at(x, y, 1.15 ** scroll_y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # drag the world along with the cursor
        z = self.world_camera.zoom
        px, py = self.world_camera.position
        self.world_camera.position = (px - dx / z, py - dy / z)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.UP:
            self.steps_per_frame = min(50, self.steps_per_frame + 1)
        elif key == arcade.key.DOWN:
            self.steps_per_frame = max(1, self.steps_per_frame - 1)
        elif key in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_at(self.width / 2, self.height / 2, 1.25)
        elif key in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_at(self.width / 2, self.height / 2, 1 / 1.25)
        elif key in (arcade.key.KEY_0, arcade.key.NUM_0):
            self._reset_view()
        elif key in (arcade.key.W, arcade.key.A, arcade.key.S, arcade.key.D):
            step = 40 / self.world_camera.zoom
            px, py = self.world_camera.position
            dx = (key == arcade.key.D) - (key == arcade.key.A)
            dy = (key == arcade.key.W) - (key == arcade.key.S)
            self.world_camera.position = (px + dx * step, py + dy * step)
        elif key == arcade.key.V:
            self.show_veg = not self.show_veg
        elif key == arcade.key.ESCAPE:
            self.close()


def run(cfg: Config | None = None, scale: int = 4, steps_per_frame: int = 1):
    EcosystemViewer(cfg, scale=scale, steps_per_frame=steps_per_frame)
    arcade.run()
