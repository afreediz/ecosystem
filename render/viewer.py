"""Arcade observer window (§16 of v1.md).

Holds a Simulation, calls ``sim.step(dt)`` in ``on_update``, and draws it. It is an
OBSERVER ONLY -- it never mutates sim state. The world is rendered once into a background
texture; entities are drawn each frame as batched point clouds (one call per species).

Controls:
  SPACE        pause/resume            UP/DOWN     speed up / slow down (x2 / /2)
  +/- or wheel zoom in/out             right-drag  pan
  0            reset view (fit map)    middle-drag pan
  V            toggle veg overlay      Ctrl+V      freeze/unfreeze veg regrowth
  S            fast-forward season     Ctrl+S      pause/resume season
  Shift+S      spawn sheep at cursor   Shift+F     spawn fox at cursor
  ESC          quit
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import arcade

from config import Config, SHEEP, FOX
from sim.entities import MALE
from sim.simulation import Simulation
from sim.world import BIOME_COLORS
from sim.perception import (
    SH_TERRAIN, SH_WATER, SH_FOOD, SH_THREAT, SH_MATE,
    FX_TERRAIN, FX_WATER, FX_FOOD, FX_MATE)

# Both species flash this rose hue for a few ticks right after they breed.
MATING_COLOR = (255, 80, 150)
# Small black marker drawn at the top-right of every male.
MALE_MARK_COLOR = (0, 0, 0)
# Ring drawn around the currently-selected (inspected) entity.
SELECT_COLOR = (255, 230, 60)

# Perception-grid inspector: per-species (label, channel, base colour) lists -- each species
# carries only the channels it uses, so the panel adapts to the selected agent. The base
# colour is the hue a fully-present cell glows; intermediate values fade toward black.
GRID_CHANNELS_BY_SPECIES = {
    SHEEP: [
        ("terrain", SH_TERRAIN, (150, 150, 150)),
        ("water",   SH_WATER,   (60, 130, 220)),
        ("food",    SH_FOOD,    (90, 190, 80)),     # food = grass field
        ("threat",  SH_THREAT,  (230, 60, 40)),
        ("mate",    SH_MATE,    (255, 90, 160)),
    ],
    FOX: [
        ("terrain", FX_TERRAIN, (150, 150, 150)),
        ("water",   FX_WATER,   (60, 130, 220)),
        ("food",    FX_FOOD,    (250, 200, 50)),    # food = prey entities
        ("mate",    FX_MATE,    (255, 90, 160)),
    ],
}


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

        # entity inspector: slot of the clicked agent (None = nothing selected) + the lazily
        # built per-channel textures used to draw its egocentric perception grids.
        self._selected_slot = None
        self._grid_textures = None     # list[(arcade.Texture, np.ndarray rgba, PIL.Image)]
        self._grid_k = None            # K (window side) the textures were sized for

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

        # sheep then fox; each gets a male marker + a rose tint while mating
        self._draw_species(SHEEP, (245, 245, 245), max(2.0, self.scale * 0.8))
        self._draw_species(FOX, (220, 70, 40), max(3.0, self.scale * 1.1))

        # ring the inspected entity so it's easy to keep track of as it moves
        self._draw_selection_ring()

        # nightfall: dim the whole scene as daylight fades (cosmetic, driven by the same
        # diurnal clock the animals sleep by). Drawn over terrain + entities, under the HUD.
        self._draw_night_overlay()

        self.gui_camera.use()
        self._draw_hud()
        self._draw_perception_inspector()

    def _draw_selection_ring(self) -> None:
        slot = self._selected_slot
        if slot is None:
            return
        ent = self.sim.entities
        if not bool(ent.alive[slot]):
            self._selected_slot = None       # the inspected animal died; drop the selection
            return
        sx, sy = self._world_to_screen(ent.pos_x[slot:slot + 1], ent.pos_y[slot:slot + 1])
        r = max(4.0, self.scale * 1.6)
        arcade.draw_circle_outline(float(sx[0]), float(sy[0]), r, SELECT_COLOR, 1.5)

    def _draw_night_overlay(self) -> None:
        from sim.environment import light_level
        darkness = 1.0 - light_level(self.sim.env.time_of_day)
        if darkness <= 0.01:
            return
        alpha = int(165 * darkness)              # deep night ~ a heavy dusk-blue veil
        arcade.draw_rect_filled(
            arcade.LBWH(0, 0, self._content_w, self._content_h), (12, 18, 48, alpha))

    def _draw_species(self, species_id: int, base_color, size: float) -> None:
        """Draw one species as a point cloud, with per-entity overlays:

        * animals that bred in the last ``mating_glow_duration`` ticks are tinted
          ``MATING_COLOR`` instead of their species color;
        * sleeping animals are drawn dimmed (resting in the dark);
        * every male gets a small black dot offset to its top-right.
        """
        ent = self.sim.entities
        slots = np.nonzero(ent.species_mask(species_id))[0]
        if slots.shape[0] == 0:
            return
        sx, sy = self._world_to_screen(ent.pos_x[slots], ent.pos_y[slots])

        mating = ent.mating_glow[slots] > 0.0
        asleep = ent.asleep[slots]
        dim_color = tuple(int(ch * 0.45) for ch in base_color)
        # three disjoint groups: awake, asleep (dimmed), and mating (tint wins either way)
        for mask, color in ((~mating & ~asleep, base_color),
                            (~mating & asleep, dim_color),
                            (mating, MATING_COLOR)):
            if mask.any():
                arcade.draw_points(list(zip(sx[mask].tolist(), sy[mask].tolist())),
                                   color, size)

        # male marker: a small black dot nudged toward the top-right of the body
        male = ent.sex[slots] == MALE
        if male.any():
            off = max(1.0, size * 0.55)
            mx = (sx[male] + off).tolist()
            my = (sy[male] + off).tolist()
            arcade.draw_points(list(zip(mx, my)), MALE_MARK_COLOR, max(1.5, size * 0.5))

    def _draw_hud(self):
        s = self.sim.stats
        env = self.sim.env
        from sim.environment import WEATHER_NAMES, season_name, daytime_name
        season_tag = season_name(env.season) + (" PAUSED" if env.season_paused else "")
        lines = [
            f"tick {self.sim.tick}   {'PAUSED' if self.paused else f'x{self.steps_per_frame:g}'}"
            f"   zoom {self.world_camera.zoom / self._fit_zoom():.1f}x",
            f"sheep {s.get('n_sheep', 0)}   fox {s.get('n_fox', 0)}   veg {s.get('veg_biomass', 0):.0f}"
            f"{'  [veg frozen]' if self.sim.veg_growth_paused else ''}",
            f"{daytime_name(env.time_of_day)} ({env.time_of_day:.2f})   "
            f"{season_tag} ({env.season:.2f})   weather {WEATHER_NAMES[env.weather]}",
            f"births {s.get('births', 0)}  deaths {s.get('deaths', 0)} "
            f"(pred {s.get('death_predation', 0)})   asleep {s.get('n_asleep', 0)}",
            "male: black dot   mating: rose tint   asleep: dimmed",
            "left-click an animal to inspect its perception grids",
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

    # ------------------------------------------------------------------ perception inspector
    def _selected_obs_grids(self):
        """The (C, K, K) perception stack of the selected agent, or None.

        Looks the entity's slot up in its species' latest observation. Returns None when
        nothing is selected, the agent died, or it isn't in this tick's obs yet.
        """
        slot = self._selected_slot
        if slot is None:
            return None
        if not bool(self.sim.entities.alive[slot]):
            self._selected_slot = None
            return None
        obs_by_species = self.sim.last_obs
        if obs_by_species is None:
            return None
        species_id = int(self.sim.entities.species[slot])
        obs = obs_by_species.get(species_id)
        if obs is None:
            return None
        rows = np.nonzero(obs.idx == slot)[0]
        if rows.shape[0] == 0 or int(rows[0]) >= obs.grids.shape[0]:
            return None
        return obs.grids[int(rows[0])]

    def _ensure_grid_textures(self, k: int, n_ch: int) -> None:
        """Lazily (re)build ``n_ch`` per-channel RGBA textures for a window side ``k``."""
        if self._grid_textures is not None and self._grid_k == (k, n_ch):
            return
        self._grid_textures = []
        for _ in range(n_ch):
            rgba = np.zeros((k, k, 4), dtype=np.uint8)
            img = Image.fromarray(rgba, mode="RGBA")
            tex = arcade.Texture(img)
            self.ctx.default_atlas.add(tex)
            self._grid_textures.append([tex, rgba, img])
        self._grid_k = (k, n_ch)

    def _draw_perception_inspector(self) -> None:
        grids = self._selected_obs_grids()
        if grids is None:
            return
        ent = self.sim.entities
        slot = self._selected_slot
        species_id = int(ent.species[slot])
        channels = GRID_CHANNELS_BY_SPECIES[species_id]
        n_ch = len(channels)
        k = grids.shape[-1]
        self._ensure_grid_textures(k, n_ch)

        species = "fox" if species_id == FOX else "sheep"

        # layout: two columns of channel tiles tucked under the top-right corner
        chan, gap, label_h, margin = 92, 6, 14, 8
        cols, rows = 2, (n_ch + 1) // 2
        header_h = 18
        panel_w = cols * chan + (cols + 1) * gap
        panel_h = header_h + rows * (chan + label_h) + (rows + 1) * gap
        x0 = self.width - panel_w - margin
        y_top = self.height - margin

        arcade.draw_rect_filled(
            arcade.LBWH(x0, y_top - panel_h, panel_w, panel_h), (0, 0, 0, 170))
        arcade.draw_text(f"perception  {species} #{slot}", x0 + gap, y_top - header_h + 2,
                         SELECT_COLOR, 12, font_name=("consolas", "monospace"))

        grid_top = y_top - header_h
        for i, (label, ch, color) in enumerate(channels):
            col, row = i % cols, i // cols
            cell_x = x0 + gap + col * (chan + gap)
            cell_top = grid_top - gap - row * (chan + label_h + gap)
            self._update_channel_texture(i, grids[ch], color)
            tex = self._grid_textures[i][0]
            arcade.draw_text(label, cell_x, cell_top - label_h + 2, color, 10,
                             font_name=("consolas", "monospace"))
            arcade.draw_texture_rect(
                tex, arcade.LBWH(cell_x, cell_top - label_h - chan, chan, chan))

    def _update_channel_texture(self, i: int, chan_vals: np.ndarray, color) -> None:
        """Paint one channel into its texture: brightness ~ value, marker at the centre."""
        tex, rgba, img = self._grid_textures[i]
        # no flip: channel row 0 is the smallest world-y, which the map also draws at the
        # top (see _world_to_screen), so the tile orientation already matches the map.
        v = np.clip(chan_vals, 0.0, 1.0)
        rgba[..., 0] = (color[0] * v).astype(np.uint8)
        rgba[..., 1] = (color[1] * v).astype(np.uint8)
        rgba[..., 2] = (color[2] * v).astype(np.uint8)
        rgba[..., 3] = np.where(v > 0.0, 235, 60).astype(np.uint8)   # faint where unseen/empty
        c = v.shape[0] // 2                                          # agent sits at the centre
        rgba[c, c] = (255, 255, 255, 255)
        img.frombytes(rgba.tobytes())
        self.ctx.default_atlas.update_texture_image(tex)

    # ------------------------------------------------------------------ input
    def on_resize(self, width: int, height: int):
        super().on_resize(width, height)
        if not hasattr(self, "gui_camera"):
            return   # a resize event can fire during window construction, before setup
        # GUI camera must keep (0,0) at the bottom-left after a resize (position=True),
        # else the HUD's screen coords fall outside the view and it vanishes in fullscreen.
        self.gui_camera.match_window(position=True)
        self.world_camera.match_window()
        self._reset_view()   # refit the whole map into the new window size

    def on_mouse_motion(self, x, y, dx, dy):
        self._mouse_x, self._mouse_y = x, y

    def on_mouse_press(self, x, y, button, modifiers):
        self._mouse_x, self._mouse_y = x, y
        # remember the world point grabbed under the cursor so right-drag can keep it there
        self._pan_anchor = self.world_camera.unproject((x, y))
        # left-click selects the nearest animal under the cursor for the perception inspector
        # (left-drag never pans, so this can't be confused with a pan gesture)
        if button == arcade.MOUSE_BUTTON_LEFT:
            self._selected_slot = self._pick_entity(x, y)

    def _pick_entity(self, screen_x, screen_y):
        """Slot of the nearest alive animal within a small screen-radius of the cursor.

        Returns the entity slot index, or None if the click landed on empty ground.
        """
        ent = self.sim.entities
        slots = np.nonzero(ent.alive)[0]
        if slots.shape[0] == 0:
            return None
        sx, sy = self._world_to_screen(ent.pos_x[slots], ent.pos_y[slots])
        # entity content-pixel coords -> screen coords via the world camera projection
        proj = [self.world_camera.project((px, py))
                for px, py in zip(sx.tolist(), sy.tolist())]
        scr = np.array([(p.x, p.y) for p in proj], dtype=np.float32)
        d2 = (scr[:, 0] - screen_x) ** 2 + (scr[:, 1] - screen_y) ** 2
        i = int(np.argmin(d2))
        # accept only a reasonably close click (~14 px) so empty space deselects
        return int(slots[i]) if d2[i] <= 14.0 ** 2 else None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self._mouse_x, self._mouse_y = x, y
        self._zoom_at(x, y, 1.15 ** scroll_y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self._mouse_x, self._mouse_y = x, y
        # pan only while RIGHT (or middle) is held; left-drag does nothing so a plain
        # click never shifts the map.
        if not (buttons & (arcade.MOUSE_BUTTON_MIDDLE)):
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
            if ctrl:
                self.sim.veg_growth_paused = not self.sim.veg_growth_paused  # Ctrl+V
            else:
                self.show_veg = not self.show_veg                            # V: overlay
        elif key == arcade.key.ESCAPE:
            self.close()


def run(cfg: Config | None = None, scale: int = 4, steps_per_frame: float = 1.0):
    EcosystemViewer(cfg, scale=scale, steps_per_frame=steps_per_frame)
    arcade.run()
