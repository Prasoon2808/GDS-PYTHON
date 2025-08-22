import os
import sys

# Ensure repository root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GenerateView, GridPlan, ColumnGrid, Openings
from ui.overlays import ColumnGridOverlay


class CountingCanvas:
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.items = []

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, _):
        self.items.clear()

    def create_line(self, x0, y0, x1, y1, **kwargs):
        self.items.append(("line", x0, y0, x1, y1))
        return len(self.items)

    def create_rectangle(self, x0, y0, x1, y1, **kwargs):
        self.items.append(("rect", x0, y0, x1, y1))
        return len(self.items)

    def create_oval(self, x0, y0, x1, y1, **kwargs):
        self.items.append(("oval", x0, y0, x1, y1))
        return len(self.items)

    def create_text(self, x, y, **kwargs):
        self.items.append(("text", x, y, kwargs.get("text")))
        return len(self.items)

    def tag_bind(self, *args, **kwargs):
        pass

    def tag_lower(self, *args, **kwargs):
        pass

    def coords(self, *args, **kwargs):
        pass

    def itemconfigure(self, *args, **kwargs):
        pass



def test_grid_labels_persist_across_redraws():
    plan = GridPlan(4.0, 4.0, column_grid=ColumnGrid(4, 4))
    gv = GenerateView.__new__(GenerateView)
    gv.bed_plan = plan
    gv.bath_plan = None
    gv.plan = plan
    gv.bed_openings = Openings(plan)
    gv.bath_openings = None
    gv.canvas = CountingCanvas()
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv.zoom_factor = 1.0

    gv._draw()
    first_text = [i for i in gv.canvas.items if i[0] == "text"]
    assert first_text

    gv._draw()
    second_text = [i for i in gv.canvas.items if i[0] == "text"]
    assert len(second_text) == len(first_text)
