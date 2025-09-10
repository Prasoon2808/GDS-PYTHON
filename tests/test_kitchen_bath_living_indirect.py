import os
import os
import sys
import tkinter as tk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GenerateView,
    GridPlan,
    Openings,
    CELL_M,
    shares_edge,
    WALL_LEFT,
    WALL_RIGHT,
    WALL_BOTTOM,
    WALL_TOP,
)
from test_generate_view import DummyStatus, DummyRoot


def make_gv():
    master = tk.Tcl()
    tk._default_root = master
    gv = GenerateView.__new__(GenerateView)
    gv.status = DummyStatus()
    gv.root = DummyRoot()
    gv.sim_timer = gv.sim2_timer = None
    gv.sim_path = gv.sim_poly = gv.sim2_path = gv.sim2_poly = []
    gv._draw = lambda: None
    gv._log_run = lambda meta: None
    gv.bed_key = None
    gv.mlp = gv.transformer = None
    gv.force_bst_pair = type("V", (), {"get": lambda self: False})()
    return gv


def test_indirect_living_connection(monkeypatch):
    gv = make_gv()
    cell = CELL_M
    gv.bed_Wm = 2 * cell
    gv.bed_Hm = cell
    gv.bath_dims = (cell, cell)
    gv.bath_Wm = gv.bath_Hm = cell
    gv.liv_dims = (cell, cell)
    gv.liv_Wm = gv.liv_Hm = cell
    gv.kitch_dims = (cell, cell)
    gv.kitch_Wm = gv.kitch_Hm = cell

    gv.bed_openings = Openings(GridPlan(gv.bed_Wm, gv.bed_Hm))
    gv.bed_openings.swing_depth = cell
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = cell / 2
    gv.bed_openings.door_width = cell
    gv.bath_openings = Openings(GridPlan(*gv.bath_dims))
    gv.bath_openings.swing_depth = cell
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = cell / 2
    gv.bath_openings.door_width = cell
    gv.liv_openings = Openings(GridPlan(*gv.liv_dims))
    gv.liv_openings.swing_depth = cell
    gv.liv_openings.door_wall = WALL_TOP
    gv.liv_openings.door_center = cell / 2
    gv.liv_openings.door_width = cell
    gv.kitch_openings = None
    gv.bath_liv_openings = Openings(GridPlan(*gv.bath_dims))
    gv.bath_liv_openings.swing_depth = cell
    gv.bath_liv_openings.door_wall = WALL_BOTTOM
    gv.bath_liv_openings.door_center = cell / 2
    gv.bath_liv_openings.door_width = cell
    gv.liv_bath_openings = Openings(GridPlan(*gv.liv_dims))
    gv.liv_bath_openings.swing_depth = cell
    gv.liv_bath_openings.door_wall = WALL_TOP
    gv.liv_bath_openings.door_center = cell / 2
    gv.liv_bath_openings.door_width = cell

    gv._apply_openings_from_ui = lambda: True

    class DummyBedroomSolver:
        def __init__(self, plan, *a, **k):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {"score": 1.0}

    class DummyKitchenSolver:
        def __init__(self, plan, *a, **k):
            self.plan = plan
        def run(self, appliance_sets=None):
            self.plan.place(0, 0, 1, 1, 'SINK')
            return self.plan, None

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None):
        p = GridPlan(w, h)
        p.place(0, 0, 1, 1, 'WC')
        return p

    def dummy_arrange_livingroom(w, h, rules, openings=None):
        p = GridPlan(w, h)
        p.place(0, 0, 1, 1, 'SOFA')
        return p

    import vastu_all_in_one
    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'KitchenSolver', DummyKitchenSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)
    monkeypatch.setattr(vastu_all_in_one.GenerateView, '_add_door_clearance', lambda *a, **k: None, raising=False)

    gv._solve_and_draw()

    assert "Kitchen must share" not in gv.status.msg
    assert shares_edge(gv.kitch_plan, gv.bath_plan)
    assert not shares_edge(gv.kitch_plan, gv.liv_plan)
