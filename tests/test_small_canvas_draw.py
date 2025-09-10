import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gds import GenerateView, GridPlan

class TinyCanvas:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.after_idle_called_with = None
    def winfo_width(self):
        return self.w
    def winfo_height(self):
        return self.h
    def delete(self, *args, **kwargs):
        pass
    def update_idletasks(self):
        pass
    def after_idle(self, func):
        self.after_idle_called_with = func


def test_draw_reschedules_on_too_small_canvas_and_draws_after_resize():
    gv = GenerateView.__new__(GenerateView)
    gv.canvas = TinyCanvas(10, 10)
    gv.zoom_factor = 1
    gv.bed_plan = GridPlan(1.0, 1.0)
    gv.bed_openings = object()
    gv.bath_plan = gv.liv_plan = gv.kitch_plan = None
    gv.plan = gv.bed_plan
    gv.sim_path = gv.sim_poly = gv.sim2_path = gv.sim2_poly = []
    gv.grid_overlay = type('G', (), {'redraw': lambda *a, **k: None})()
    gv.popover = type('P', (), {'hide': lambda self: None})()
    gv._draw_all_layers = lambda *a, **k: None

    gv._draw()
    assert gv.canvas.after_idle_called_with == gv._draw

    gv.canvas.w = gv.canvas.h = 400
    called = {'count': 0}
    gv._draw_all_layers = lambda *a, **k: called.update(count=called['count'] + 1)
    gv._draw()
    assert called['count'] == 1
