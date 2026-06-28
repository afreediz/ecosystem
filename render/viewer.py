"""Arcade observer window (§16 of v1.md).

Holds a Simulation, calls ``sim.step(dt)`` in ``on_update``, and draws it. It is an
OBSERVER ONLY -- it never mutates sim state. The world is rendered once into a background
texture; entities are drawn each frame as batched point clouds (one call per species).

Controls:
  SPACE        pause/resume            UP/DOWN     speed up / slow down (x2 / /2)
  +/- or wheel zoom in/out             right-drag  pan
  0            reset view (fit map)    V           toggle vegetation overlay tint
  S            fast-forward season     Ctrl+S      pause/resume season
  Shift+S      spawn sheep at cursor   Shift+F     spawn fox at cursor
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

    def __init__(self, cfg: Config | None = None, scale: int = 4, steps_per_frame: float = 1.0):
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
        self.steps_per_frame = float(steps_per_frame)
        self._step_accum = 0.0   # carries fractional steps across frames
        self.show_veg = True
        self._mouse_x = win_w / 2     # latest cursor pos (screen px) for spawn-at-mouse
        self._mouse_y = win_h / 2
        self._pan_anchor = None       # world point grabbed at right-mouse-press (for panning)

        # World camera (zoom/pan) for terrain + entities; GUI camera (fixed) for the HUD.
        self.world_camera = arcade.camera.Camera2D()
        self.gui_camera = arcade.camera.Camera2D()
        self._reset_view()

        self._bg_texture = self._build_background_texture()
        self._build_veg_overlay()

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

    # ------------------------------------------------------------------ vegetation overlay
    def _build_veg_overlay(self) -> None:
        """Create a live RGBA overlay tinting each grazeable cell by its vegetation level.

        Only plains + forest (the cells that grow meaningful food) are tinted; other
        biomes keep their natural background. The texture lives in the default atlas and
        is re-uploaded in place each frame (``update_texture_image``) -- no per-frame
        allocation, so it stays cheap even at 160x160.
        """
        world = self.sim.world
        self._veg_mask = world.plant_suitability >= 0.5          # plains + forest
        self._veg_rgba = np.zeros((world.h, world.w, 4), dtype=np.uint8)
        self._veg_img = Image.fromarray(self._veg_rgba, mode="RGBA")
        self._veg_texture = arcade.Texture(self._veg_img)
        self.ctx.default_atlas.add(self._veg_texture)            # realize atlas region + hash
        self._refresh_veg_texture()

    def _refresh_veg_texture(self) -> None:
        # level 0 -> bare (tan), level 1 -> lush green; normalized so ~0.6 biomass reads full
        level = np.clip(self.sim.veg / 0.6, 0.0, 1.0)
        m = self._veg_mask
        rgba = self._veg_rgba
        rgba[..., 0] = np.where(m, 150.0 - 110.0 * level, 0).astype(np.uint8)
        rgba[..., 1] = np.where(m, 110.0 + 70.0 * level, 0).astype(np.uint8)
        rgba[..., 2] = np.where(m, 60.0 - 20.0 * level, 0).astype(np.uint8)
        rgba[..., 3] = np.where(m, 220, 0).astype(np.uint8)      # transparent off-mask
        self._veg_img.frombytes(rgba.tobytes())
        self.ctx.default_atlas.update_texture_image(self._veg_texture)

    # ------------------------------------------------------------------ loop
    def on_update(self, dt: float):
        if self.paused:
            return
        self._step_accum += self.steps_per_frame
        n = int(self._step_accum)
        self._step_accum -= n
        for _ in range(n):
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

        # live vegetation tint over grazeable cells (depletes when grazed, regrows seasonally)
        if self.show_veg:
            self._refresh_veg_texture()
            arcade.draw_texture_rect(
                self._veg_texture, arcade.LBWH(0, 0, self._content_w, self._content_h))

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
        from sim.environment import WEATHER_NAMES, season_name, daytime_name
        season_tag = season_name(env.season) + (" PAUSED" if env.season_paused else "")
        lines = [
            f"tick {self.sim.tick}   {'PAUSED' if self.paused else f'x{self.steps_per_frame:g}'}"
            f"   zoom {self.world_camera.zoom / self._fit_zoom():.1f}x",
            f"sheep {s.get('n_sheep', 0)}   fox {s.get('n_fox', 0)}   veg {s.get('veg_biomass', 0):.0f}",
            f"{daytime_name(env.time_of_day)} ({env.time_of_day:.2f})   "
            f"{season_tag} ({env.season:.2f})   weather {WEATHER_NAMES[env.weather]}",
            f"births {s.get('births', 0)}  deaths {s.get('deaths', 0)} "
            f"(pred {s.get('death_predation', 0)})",
        ]
        # dark translucent backing so white text stays readable over light terrain
        # (snow / beach / grazed grass) -- without it the HUD "disappears" on bright cells.
        panel_w = min(self.width, 430)
        panel_h = len(lines) * 16 + 12
        arcade.draw_rect_filled(
            arcade.LBWH(0, self.height - panel_h, panel_w, panel_h), (0, 0, 0, 150))

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

    def on_mouse_motion(self, x, y, dx, dy):
        self._mouse_x, self._mouse_y = x, y

    def on_mouse_press(self, x, y, button, modifiers):
        self._mouse_x, self._mouse_y = x, y
        # remember the world point grabbed under the cursor so right-drag can keep it there
        self._pan_anchor = self.world_camera.unproject((x, y))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self._mouse_x, self._mouse_y = x, y
        self._zoom_at(x, y, 1.15 ** scroll_y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._mouse_x, self._mouse_y = x, y
        # pan only while RIGHT (or middle) is held; left-drag does nothing so a plain
        # click never shifts the map.
        if not (buttons & (arcade.MOUSE_BUTTON_RIGHT | arcade.MOUSE_BUTTON_MIDDLE)):
            return
        if self._pan_anchor is None:
            self._pan_anchor = self.world_camera.unproject((x, y))
            return
        # Keep the grabbed world point pinned under the cursor. Shifting the camera by
        # (anchor - current) is exact at any zoom and never drifts, unlike accumulating
        # raw pixel deltas.
        cur = self.world_camera.unproject((x, y))
        px, py = self.world_camera.position
        self.world_camera.position = (px + (self._pan_anchor.x - cur.x),
                                      py + (self._pan_anchor.y - cur.y))

    def _spawn_at_mouse(self, species_id: int) -> None:
        """Spawn one animal at the cell under the cursor."""
        v = self.world_camera.unproject((self._mouse_x, self._mouse_y))
        wx = v.x / self.scale
        wy = self.sim.world.h - v.y / self.scale     # invert the draw-time Y flip
        self.sim.spawn_at(species_id, wx, wy)

    def on_key_press(self, key, modifiers):
        ctrl = modifiers & arcade.key.MOD_CTRL
        shift = modifiers & arcade.key.MOD_SHIFT
        if key == arcade.key.SPACE:
            self.paused = not self.paused
        elif key == arcade.key.UP:
            self.steps_per_frame = min(64.0, self.steps_per_frame * 2)
        elif key == arcade.key.DOWN:
            self.steps_per_frame = max(1 / 64, self.steps_per_frame / 2)
        elif key in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._zoom_at(self.width / 2, self.height / 2, 1.25)
        elif key in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._zoom_at(self.width / 2, self.height / 2, 1 / 1.25)
        elif key in (arcade.key.KEY_0, arcade.key.NUM_0):
            self._reset_view()
        elif key == arcade.key.S:
            if ctrl:
                self.sim.env.toggle_season_pause()       # Ctrl+S: pause/resume season
            elif shift:
                self._spawn_at_mouse(SHEEP)              # Shift+S: spawn sheep at cursor
            else:
                self.sim.env.advance_season(0.1)         # S: fast-forward the season
        elif key == arcade.key.F and shift:
            self._spawn_at_mouse(FOX)                    # Shift+F: spawn fox at cursor
        elif key == arcade.key.V:
            self.show_veg = not self.show_veg
        elif key == arcade.key.ESCAPE:
            self.close()


def run(cfg: Config | None = None, scale: int = 4, steps_per_frame: float = 1.0):
    EcosystemViewer(cfg, scale=scale, steps_per_frame=steps_per_frame)
    arcade.run()
