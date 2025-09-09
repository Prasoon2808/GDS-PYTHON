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
        self.canvas.delete('overlay')
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
                    tags=("overlay", "grid"),
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
                    tags=("overlay",),
                )
                self.canvas.create_text(x, y, text=text, fill="#555", tags=("overlay",))
        self.canvas.tag_lower("grid")


class DoorLegendOverlay:
    """Simple legend showing the color used to draw doors."""

    def __init__(self, canvas, color: str, label: str = "Door"):
        self.canvas = canvas
        self.color = color
        self.label = label

    def redraw(self) -> None:
        self.canvas.delete("legend")
        size = 20
        x0, y0 = 10, 10
        self.canvas.create_rectangle(
            x0,
            y0,
            x0 + size,
            y0 + size,
            fill=self.color,
            outline="#000",
            tags=("legend",),
        )
        self.canvas.create_text(
            x0 + size + 6,
            y0 + size / 2,
            text=self.label,
            anchor="w",
            fill="#000",
            tags=("legend",),
        )
