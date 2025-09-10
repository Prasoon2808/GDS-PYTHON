import os
import sys
import tkinter as tk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import solver
from solver import GenerateView, GridPlan, Openings
from rules import CELL_M


class DummyStatus:
    def __init__(self):
        self.msg = None

    def set(self, msg):
        self.msg = msg


class DummyRoot:
    def after_cancel(self, *a, **k):
        pass


def test_kitchen_adjacency_failure_sets_status(monkeypatch):
    master = tk.Tcl()
    tk._default_root = master

    gv = GenerateView.__new__(GenerateView)
    gv.status = DummyStatus()
    gv.root = DummyRoot()
    gv.sim_timer = gv.sim2_timer = None
    gv.sim_path = gv.sim_poly = gv.sim2_path = gv.sim2_poly = []
    gv._draw = lambda: None
    gv._log_run = lambda meta: None

    cell = CELL_M
    gv.bed_Wm = gv.bed_Hm = cell * 4
    gv.bath_dims = (cell, cell)
    gv.bath_Wm = gv.bath_Hm = cell
    gv.liv_dims = (cell * 2, cell)
    gv.liv_Wm = cell * 2
    gv.liv_Hm = cell
    gv.kitch_dims = (cell * 2, cell)
    gv.kitch_Wm = cell * 2
    gv.kitch_Hm = cell

    for attr, w, h in [
        ("bed_openings", gv.bed_Wm, gv.bed_Hm),
        ("bath_openings", *gv.bath_dims),
        ("liv_openings", *gv.liv_dims),
        ("bath_liv_openings", *gv.bath_dims),
        ("liv_bath_openings", *gv.liv_dims),
        ("kitch_openings", *gv.kitch_dims),
    ]:
        op = Openings(GridPlan(w, h))
        op.swing_depth = cell
        op.door_width = cell
        op.door_center = cell / 2
        setattr(gv, attr, op)

    gv.plan = GridPlan(gv.bed_Wm, gv.bed_Hm)
    gv._apply_openings_from_ui = lambda: True
    monkeypatch.setattr(
        GenerateView, "_add_door_clearance", lambda self, *a, **k: None
    )
    gv.bed_key = None
    gv.mlp = gv.transformer = None
    gv.force_bst_pair = type("V", (), {"get": lambda self: False})()

    class DummyBedroomSolver:
        def __init__(self, plan, *a, **k):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {"score": 1.0}

    class DummyKitchenSolver:
        def __init__(self, plan, *a, **k):
            self.plan = plan

        def run(self, appliance_sets=None, required=()):
            self.plan.place(1, 0, 1, 1, 'SINK')
            return self.plan, None

    monkeypatch.setattr(solver, "BedroomSolver", DummyBedroomSolver)
    monkeypatch.setattr(solver, "KitchenSolver", DummyKitchenSolver)
    def dummy_arrange_bathroom(*a, **k):
        p = GridPlan(cell, cell)
        return p

    def dummy_arrange_livingroom(*a, **k):
        p = GridPlan(cell * 2, cell)
        p.place(1, 0, 1, 1, 'SOFA')
        return p

    monkeypatch.setattr(solver, "arrange_bathroom", dummy_arrange_bathroom)
    monkeypatch.setattr(solver, "arrange_livingroom", dummy_arrange_livingroom)
    monkeypatch.setattr(solver, "shares_edge", lambda a, b: False)

    gv._solve_and_draw()
    assert gv.status.msg

