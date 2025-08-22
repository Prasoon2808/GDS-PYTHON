"""UI overlay helpers for canvas-based views."""

from typing import List, Tuple


class ColumnGridOverlay:
    """Render column grid labels and dots on a Tkinter canvas."""

    GRID_COLOR = "#dddddd"

    def __init__(self, canvas):
        self.canvas = canvas
        self.coords: List[Tuple[str, float, float, str]] = []
        self._r = 8

    def _build_coords(self, cg, ox: float, oy: float, scale: float) -> None:
        """Pre-compute drawing coordinates for the column grid."""
        r = max(8, scale * 0.3)
        label_gap = r * 2.5
        coords: List[Tuple[str, float, float, str]] = []
        for i in range(cg.gw + 1):
            x = ox + i * scale
            coords.append(("col", x, oy - label_gap, cg.col_label(i)))
            for j in range(cg.gh + 1):
                y = oy + j * scale
                if i == 0:
                    coords.append(("row", ox - label_gap, y, cg.row_label(cg.gh - j)))
                coords.append(("dot", x, y, ""))
        self.coords = coords
        self._r = r

    def redraw(self, cg, ox: float, oy: float, scale: float) -> None:
        """Draw the grid overlay using cached coordinates."""
        self._build_coords(cg, ox, oy, scale)
        r = self._r
        for kind, x, y, text in self.coords:
            if kind == "dot":
                self.canvas.create_oval(
                    x - 2,
                    y - 2,
                    x + 2,
                    y + 2,
                    fill=self.GRID_COLOR,
                    outline="",
                    tags=("grid",),
                )
            else:
                self.canvas.create_oval(
                    x - r,
                    y - r,
                    x + r,
                    y + r,
                    outline=self.GRID_COLOR,
                    fill=self.GRID_COLOR,
                    width=1,
                )
                self.canvas.create_text(x, y, text=text, fill="#555")
        self.canvas.tag_lower("grid")
