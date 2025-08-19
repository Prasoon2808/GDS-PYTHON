import os
import sys
import pytest
import tkinter as tk

# Ensure the repository root is importable when tests are executed from the
# ``tests`` directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GenerateView,
    Openings,
    GridPlan,
    components_by_code,
    WALL_RIGHT,
    WALL_LEFT,
    WALL_TOP,
    WALL_BOTTOM,
)


class DummyStatus:
    def __init__(self):
        self.msg = ''

    def set(self, msg: str):
        self.msg = msg


class DummyRoot:
    def after_cancel(self, *_):
        pass


def make_generate_view(bath_dims=(2.0, 2.0)):
    master = tk.Tcl()
    tk._default_root = master
    gv = GenerateView.__new__(GenerateView)
    gv.bath_dims = bath_dims
    gv.bed_openings = Openings(GridPlan(4.0, 4.0))
    gv.bath_openings = Openings(GridPlan(*bath_dims)) if bath_dims else None
    gv.status = DummyStatus()
    gv.sim_timer = None
    gv.sim2_timer = None
    gv.root = DummyRoot()
    gv.sim_path = gv.sim_poly = gv.sim2_path = gv.sim2_poly = []
    gv._apply_openings_from_ui = lambda: None
    gv.bed_Wm = gv.bed_Hm = 4.0
    if bath_dims:
        gv.bath_Wm, gv.bath_Hm = bath_dims
    else:
        gv.bath_Wm = gv.bath_Hm = 0.0
    gv._draw = lambda: None
    gv._log_run = lambda meta: None
    gv.bed_key = None
    gv.mlp = gv.transformer = None
    gv.force_bst_pair = type('V', (), {'get': lambda self: False})()
    return gv


def test_bedroom_door_on_shared_wall_sets_status(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9

    gv._solve_and_draw()

    assert gv.status.msg == 'Bedroom door cannot be on shared wall.'
    assert getattr(gv, 'bath_plan', None) is None
    assert getattr(gv, 'bed_plan', None) is None


def test_bathroom_door_not_on_shared_wall_skips_bath(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_TOP
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9

    gv._solve_and_draw()

    assert gv.status.msg == 'Bathroom door must be on shared wall.'
    assert getattr(gv, 'bath_plan', None) is None
    assert isinstance(gv.bed_plan, GridPlan)


def test_valid_shared_wall_bathroom_door_generates_furniture(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules):
        plan = GridPlan(w, h)
        plan.place(0, 0, 1, 1, 'WC')
        return plan

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bed_openings.windows = [[WALL_TOP, 0.5, 0.5], [-1, 0.0, 0.0]]

    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    gv.bath_openings.windows = [[WALL_TOP, 0.5, 0.5], [-1, 0.0, 0.0]]

    gv._solve_and_draw()

    assert components_by_code(gv.bed_plan, 'BED'), 'Bedroom furniture missing'
    assert components_by_code(gv.bath_plan, 'WC'), 'Bathroom furniture missing'


def test_apply_batch_and_generate_reruns_bed_and_bath(monkeypatch):
    import vastu_all_in_one

    def fake_solve_and_draw(self):
        plan = GridPlan(self.bed_Wm, self.bed_Hm)
        plan.place(0, 0, 1, 1, 'BED')
        self.bed_plan = self.plan = plan
        if self.bath_dims:
            bath = GridPlan(self.bath_dims[0], self.bath_dims[1])
            bath.place(0, 0, 1, 1, 'WC')
            self.bath_plan = bath
        self._draw()

    monkeypatch.setattr(GenerateView, '_solve_and_draw', fake_solve_and_draw, raising=False)

    gv = make_generate_view((2.0, 2.0))
    gv._draw = lambda: None

    gv._apply_batch_and_generate()

    assert components_by_code(gv.bed_plan, 'BED'), 'Bedroom furniture missing after regenerate'
    assert components_by_code(gv.bath_plan, 'WC'), 'Bathroom furniture missing after regenerate'


def test_furniture_controls_present_for_generator_label(monkeypatch):
    import types
    import tkinter as tk
    import vastu_all_in_one

    master = tk.Tcl()
    tk._default_root = master

    class FakeWidget:
        def __init__(self, master=None, **kwargs):
            self.children = []
            self.kwargs = kwargs
            if master is not None and hasattr(master, 'children'):
                master.children.append(self)
        def pack(self, *args, **kwargs):
            return self
        def grid(self, *args, **kwargs):
            return self
        def destroy(self):
            pass
        def winfo_children(self):
            return self.children
        def grid_columnconfigure(self, *args, **kwargs):
            pass

    fake_ttk = types.SimpleNamespace(
        Frame=FakeWidget,
        Label=FakeWidget,
        Combobox=FakeWidget,
        Scale=FakeWidget,
        Button=FakeWidget,
        Checkbutton=FakeWidget,
    )
    monkeypatch.setattr(vastu_all_in_one, 'ttk', fake_ttk)

    gv = GenerateView.__new__(GenerateView)
    gv.room_label = 'Bedroom (Generator)'
    gv.sidebar = FakeWidget()
    gv.bed_Hm = 4.0
    gv.bed_Wm = 4.0
    gv.bath_dims = None

    for name in [
        '_solve_and_draw', '_apply_batch_and_generate', '_add_furniture',
        '_remove_furniture', '_simulate_one', '_simulate_two',
        'simulate_circulation', '_export_png']:
        setattr(gv, name, lambda *a, **kw: None)

    GenerateView._build_sidebar(gv)
    assert hasattr(gv, 'furn_kind'), 'Furniture controls missing'


def test_opening_control_limits(monkeypatch):
    import types
    import tkinter as tk
    import vastu_all_in_one

    master = tk.Tcl()
    tk._default_root = master

    comboboxes = []

    class FakeWidget:
        def __init__(self, master=None, **kwargs):
            self.children = []
            self.kwargs = kwargs
            if master is not None and hasattr(master, 'children'):
                master.children.append(self)
        def pack(self, *args, **kwargs):
            return self
        def grid(self, *args, **kwargs):
            return self
        def destroy(self):
            pass
        def winfo_children(self):
            return self.children
        def grid_columnconfigure(self, *args, **kwargs):
            pass

    class FakeCombobox(FakeWidget):
        def __init__(self, master=None, **kwargs):
            super().__init__(master, **kwargs)
            comboboxes.append(self)

    fake_ttk = types.SimpleNamespace(
        Frame=FakeWidget,
        Label=FakeWidget,
        Combobox=FakeCombobox,
        Scale=FakeWidget,
        Button=FakeWidget,
        Checkbutton=FakeWidget,
    )
    monkeypatch.setattr(vastu_all_in_one, 'ttk', fake_ttk)

    gv = GenerateView.__new__(GenerateView)
    gv.room_label = 'Bedroom'
    gv.sidebar = FakeWidget()
    gv.bed_Hm = 4.0
    gv.bed_Wm = 4.0
    gv.bath_dims = (2.0, 2.0)

    for name in [
        '_solve_and_draw', '_apply_batch_and_generate', '_add_furniture',
        '_remove_furniture', '_simulate_one', '_simulate_two',
        'simulate_circulation', '_export_png']:
        setattr(gv, name, lambda *a, **kw: None)

    GenerateView._build_sidebar(gv)

    assert 'Right' not in comboboxes[0].kwargs['values']
    assert gv.bed_w1_wall.get() == 'Bottom'
    assert 'Right' not in comboboxes[1].kwargs['values']
    assert 'Right' not in comboboxes[2].kwargs['values']

    assert comboboxes[3].kwargs['values'] == ['Left']
    assert 'Left' not in comboboxes[4].kwargs['values']
    assert 'Left' not in comboboxes[5].kwargs['values']
