"""Uniform spatial hash for local radius queries (§12 of v1.md).

Rebuilt once per tick from current alive positions. Supports per-category neighbor
queries within a radius so perception stays near O(N) instead of O(N^2).
"""
from __future__ import annotations

import numpy as np


class SpatialGrid:
    def __init__(self, width: float, height: float, cell_size: float):
        self.width = width
        self.height = height
        self.cell = float(cell_size)
        self.nx = max(1, int(np.ceil(width / self.cell)))
        self.ny = max(1, int(np.ceil(height / self.cell)))
        # filled by rebuild()
        self._cell_of = None        # (M,) flat cell id per indexed entity
        self._indices = None        # (M,) entity slot ids, sorted by cell
        self._cell_start = None     # CSR-style start offsets per flat cell
        self._px = None
        self._py = None

    def _flat(self, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        return cy * self.nx + cx

    def rebuild(self, indices: np.ndarray, pos_x: np.ndarray, pos_y: np.ndarray) -> None:
        """Index the given entity slots at their positions (sorted into buckets)."""
        if indices.shape[0] == 0:
            self._indices = np.empty(0, dtype=np.intp)
            self._cell_start = np.zeros(self.nx * self.ny + 1, dtype=np.intp)
            self._px = np.empty(0, dtype=np.float32)
            self._py = np.empty(0, dtype=np.float32)
            return
        px = pos_x[indices]
        py = pos_y[indices]
        cx = np.clip((px / self.cell).astype(np.intp), 0, self.nx - 1)
        cy = np.clip((py / self.cell).astype(np.intp), 0, self.ny - 1)
        flat = self._flat(cx, cy)
        order = np.argsort(flat, kind="stable")
        self._indices = indices[order]
        self._px = px[order]
        self._py = py[order]
        sorted_flat = flat[order]
        # CSR offsets via bincount cumulative
        counts = np.bincount(sorted_flat, minlength=self.nx * self.ny)
        self._cell_start = np.zeros(self.nx * self.ny + 1, dtype=np.intp)
        np.cumsum(counts, out=self._cell_start[1:])

    def query_radius(self, x: float, y: float, radius: float):
        """Return (slot_indices, px, py) for all indexed entities within ``radius``.

        Scans the block of grid cells overlapping the query disk, then does an exact
        distance filter. Returns arrays aligned to the surviving candidates.
        """
        if self._indices is None or self._indices.shape[0] == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty(0, dtype=np.float32)
            return empty_i, empty_f, empty_f
        r = float(radius)
        cx0 = max(0, int((x - r) / self.cell))
        cx1 = min(self.nx - 1, int((x + r) / self.cell))
        cy0 = max(0, int((y - r) / self.cell))
        cy1 = min(self.ny - 1, int((y + r) / self.cell))
        parts = []
        for cy in range(cy0, cy1 + 1):
            base = cy * self.nx
            s = self._cell_start[base + cx0]
            e = self._cell_start[base + cx1 + 1]
            if e > s:
                parts.append((s, e))
        if not parts:
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty(0, dtype=np.float32)
            return empty_i, empty_f, empty_f
        # gather candidate ranges
        idx_parts = [self._indices[s:e] for s, e in parts]
        px_parts = [self._px[s:e] for s, e in parts]
        py_parts = [self._py[s:e] for s, e in parts]
        cand = np.concatenate(idx_parts)
        cpx = np.concatenate(px_parts)
        cpy = np.concatenate(py_parts)
        d2 = (cpx - x) ** 2 + (cpy - y) ** 2
        keep = d2 <= r * r
        return cand[keep], cpx[keep], cpy[keep]
