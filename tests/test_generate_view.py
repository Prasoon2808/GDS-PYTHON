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
    ColumnGrid,
    components_by_code,
    add_door_clearance,
    WALL_RIGHT,
    WALL_LEFT,
    WALL_TOP,
    WALL_BOTTOM,
    opposite_wall,
    CELL_M,
    DOOR_FILL,
    WINDOW_FILL,
)
from ui.overlays import ColumnGridOverlay


class DummyStatus:
    def __init__(self):
        self.msg = ''

    def set(self, msg: str):
        self.msg = msg


class DummyRoot:
    def after_cancel(self, *_):
        pass


class BoundingCanvas:
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.items = []

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, _):
        tag = _
        if tag in ("all", None):
            self.items.clear()
        else:
            self.items = [i for i in self.items if tag not in i.get('tags', ())]

    def create_line(self, x0, y0, x1, y1, **kwargs):
        self.items.append({
            "type": "line",
            "bbox": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            **kwargs,
        })

    def create_rectangle(self, x0, y0, x1, y1, **kwargs):
        self.items.append({
            "type": "rect",
            "bbox": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            **kwargs,
        })

    def create_oval(self, x0, y0, x1, y1, **kwargs):
        self.items.append({
            "type": "oval",
            "bbox": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
            **kwargs,
        })

    def create_text(self, x, y, **kwargs):
        self.items.append({"type": "text", "bbox": (x, y, x, y), **kwargs})

    def tag_bind(self, *args, **kwargs):
        pass

    def tag_lower(self, *args, **kwargs):
        pass

    def bbox(self, tag):
        if not self.items:
            return None
        xs = [b["bbox"][0] for b in self.items] + [b["bbox"][2] for b in self.items]
        ys = [b["bbox"][1] for b in self.items] + [b["bbox"][3] for b in self.items]
        return min(xs), min(ys), max(xs), max(ys)


def make_generate_view(bath_dims=(2.0, 2.0), living_dims=None):
    master = tk.Tcl()
    tk._default_root = master
    gv = GenerateView.__new__(GenerateView)
    gv.bath_dims = bath_dims
    gv.liv_dims = living_dims
    gv.bed_openings = Openings(GridPlan(4.0, 4.0))
    gv.bed_openings.swing_depth = 0.60
    gv.bath_openings = Openings(GridPlan(*bath_dims)) if bath_dims else None
    if gv.bath_openings:
        gv.bath_openings.swing_depth = CELL_M
    gv.bath_liv_openings = Openings(GridPlan(*bath_dims)) if bath_dims and living_dims else None
    if gv.bath_liv_openings:
        gv.bath_liv_openings.swing_depth = CELL_M
    gv.liv_bath_openings = Openings(GridPlan(*living_dims)) if bath_dims and living_dims else None
    if gv.liv_bath_openings:
        gv.liv_bath_openings.swing_depth = CELL_M
    gv.liv_openings = Openings(GridPlan(*living_dims)) if living_dims else None
    if gv.liv_openings:
        gv.liv_openings.swing_depth = 0.60
    gv.status = DummyStatus()
    gv.sim_timer = None
    gv.sim2_timer = None
    gv.root = DummyRoot()
    gv.sim_path = gv.sim_poly = gv.sim2_path = gv.sim2_poly = []

    def _apply_stub():
        if gv.liv_dims and gv.bed_openings.door_wall != WALL_RIGHT:
            gv.status.set('Bedroom door must be on shared wall.')
            return False
        return True

    gv._apply_openings_from_ui = _apply_stub
    gv.bed_Wm = gv.bed_Hm = 4.0
    if bath_dims:
        gv.bath_Wm, gv.bath_Hm = bath_dims
    else:
        gv.bath_Wm = gv.bath_Hm = 0.0
    if living_dims:
        gv.liv_Wm, gv.liv_Hm = living_dims
    else:
        gv.liv_Wm = gv.liv_Hm = 0.0
    gv.liv_plan = GridPlan(gv.liv_Wm, gv.liv_Hm) if living_dims else None
    gv._draw = lambda: None
    gv._log_run = lambda meta: None
    gv.bed_key = None
    gv.mlp = gv.transformer = None
    gv.force_bst_pair = type('V', (), {'get': lambda self: False})()
    return gv


def test_bedroom_door_on_shared_wall_allows_generation(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {
                'score': 1.0,
                'coverage': 0.5,
                'paths_ok': True,
                'reach_windows': True,
            }

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        p = GridPlan(w, h)
        if openings:
            add_door_clearance(p, openings, 'DOOR')
        if secondary_openings:
            secondary_openings.ext_rect = add_door_clearance(p, secondary_openings, 'LIVING_DOOR')
        return p

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    gv.liv_openings.door_wall = WALL_BOTTOM
    gv.liv_openings.door_center = 1.0
    gv.liv_openings.door_width = 0.9

    gv._apply_openings_from_ui = lambda: True

    gv._solve_and_draw()

    assert 'Bedroom door must be on shared wall.' not in gv.status.msg
    assert isinstance(gv.bed_plan, GridPlan)
    assert isinstance(gv.bath_plan, GridPlan)


def test_bedroom_door_not_on_shared_wall_rejects(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {
                'score': 1.0,
                'coverage': 0.5,
                'paths_ok': True,
                'reach_windows': True,
            }

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        p = GridPlan(w, h)
        if openings:
            add_door_clearance(p, openings, 'DOOR')
        if secondary_openings:
            secondary_openings.ext_rect = add_door_clearance(p, secondary_openings, 'LIVING_DOOR')
        return p

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    gv.liv_openings.door_wall = WALL_BOTTOM
    gv.liv_openings.door_center = 1.0
    gv.liv_openings.door_width = 0.9

    gv._solve_and_draw()

    assert gv.status.msg == 'Bedroom door must be on shared wall.'
    assert getattr(gv, 'bed_plan', None) is None


def test_bathroom_door_not_on_shared_wall_skips_bath(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        p = GridPlan(w, h)
        if openings:
            add_door_clearance(p, openings, 'DOOR')
        if secondary_openings:
            secondary_openings.ext_rect = add_door_clearance(p, secondary_openings, 'LIVING_DOOR')
        return p

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

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
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


def test_apply_batch_and_generate_updates_status(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.0, 'paths_ok': True, 'reach_windows': True}

    seen = {}
    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        seen['rng'] = rng
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv._apply_batch_and_generate()

    assert seen.get('rng') is None, 'arrange_bathroom should be deterministic'


def test_solver_failure_keeps_previous_plan(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            pass
        def run(self):
            return None, {}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9

    prev = GridPlan(4.0, 4.0)
    prev.place(0, 0, 1, 1, 'BED')
    gv.plan = prev

    gv._solve_and_draw()

    assert gv.plan is prev
    assert gv.status.msg == 'No arrangement found (adjust door/windows).'


def test_solver_rejects_plan_without_bed(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = GridPlan(plan.Wm, plan.Hm)

        def run(self):
            # Simulate solver failing to place a bed
            return None, {'status': 'no_bed'}

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)

    draw_flag = {'called': False}

    def fake_draw():
        draw_flag['called'] = True

    gv = make_generate_view(None)
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv._draw = fake_draw

    gv._solve_and_draw()

    assert getattr(gv, 'plan', None) is None
    assert 'No bed placed' in gv.status.msg
    assert draw_flag['called'] is False


def test_arrange_bathroom_failure_warns_user(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan
        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def failing_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return None

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', failing_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9

    gv._solve_and_draw()

    assert gv.bath_plan is None
    assert gv.plan is gv.bed_plan
    assert gv.status.msg == 'Bathroom generation failed; bedroom only.'


def test_generate_view_combines_all_rooms_and_aligns_doors(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        plan = GridPlan(w, h)
        plan.place(0, 0, 1, 1, 'WC')
        return plan

    def dummy_arrange_livingroom(w, h, rules, openings=None, rng=None):
        plan = GridPlan(w, h)
        plan.place(0, 0, 1, 1, 'SOFA')
        return plan

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    # shared bedroom/bathroom door
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    # living room exterior door
    gv.liv_openings.door_wall = WALL_BOTTOM
    gv.liv_openings.door_center = 1.0
    gv.liv_openings.door_width = 0.9

    gv._apply_openings_from_ui = lambda: True

    gv._solve_and_draw()

    assert isinstance(gv.bed_plan, GridPlan)
    assert isinstance(gv.bath_plan, GridPlan)
    assert isinstance(gv.liv_plan, GridPlan)
    assert gv.plan.gw == max(gv.bed_plan.gw + gv.bath_plan.gw, gv.liv_plan.gw)

    bath_clear = next(
        (x, y, w, h)
        for x, y, w, h, kind, owner in gv.bath_plan.clearzones
        if kind == 'DOOR_CLEAR' and owner == 'BATHROOM_DOOR'
    )
    shared_op = Openings(gv.bed_plan)
    shared_op.door_wall = WALL_RIGHT
    shared_op.door_center = gv.bath_openings.door_center
    shared_op.door_width = gv.bath_openings.door_width
    shared_op.swing_depth = gv.bath_openings.swing_depth
    _, start, _ = shared_op.door_span_cells()
    depth = gv.bed_plan.meters_to_cells(shared_op.swing_depth)
    outside_x = gv.bed_plan.gw + depth
    outside_y = start
    bed_label = gv.bed_plan.coord_to_label(outside_x, outside_y)
    bath_label = gv.bath_plan.coord_to_label(bath_clear[0], bath_clear[1])
    assert bed_label == bath_label


def test_bathroom_has_second_door_shared_with_living(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        p = GridPlan(w, h)
        if openings:
            add_door_clearance(p, openings, 'DOOR')
        if secondary_openings:
            secondary_openings.ext_rect = add_door_clearance(p, secondary_openings, 'LIVING_DOOR')
        return p

    def dummy_arrange_livingroom(w, h, rules, openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    gv.bath_liv_openings.door_wall = WALL_BOTTOM
    gv.bath_liv_openings.door_center = 1.2
    gv.bath_liv_openings.door_width = 0.9
    gv.liv_bath_openings.door_wall = WALL_TOP
    gv.liv_bath_openings.door_center = 1.2
    gv.liv_bath_openings.door_width = 0.9
    gv.liv_openings.door_wall = WALL_BOTTOM
    gv.liv_openings.door_center = 1.0
    gv.liv_openings.door_width = 0.9

    gv._apply_openings_from_ui = lambda: True

    gv._solve_and_draw()

    bx1, by1, bw1, bh1 = gv.bath_openings.door_rect_cells()
    for j in range(by1, by1 + bh1):
        for i in range(bx1, bx1 + bw1):
            assert gv.bath_plan.occ[j][i] == 'DOOR'
    bx2, by2, bw2, bh2 = gv.bath_liv_openings.door_rect_cells()
    for j in range(by2, by2 + bh2):
        for i in range(bx2, bx2 + bw2):
            assert gv.bath_plan.occ[j][i] == 'DOOR'

    gv.liv_bath_openings.p = gv.liv_plan
    lx, ly, lw, lh = gv.liv_bath_openings.door_rect_cells()
    for j in range(ly, ly + lh):
        for i in range(lx, lx + lw):
            assert gv.liv_plan.occ[j][i] == 'DOOR'

    assert any(
        owner == 'LIVING_DOOR' and kind == 'DOOR_CLEAR'
        for *_, kind, owner in gv.liv_plan.clearzones
    )
    assert gv.liv_bath_openings.door_wall == WALL_TOP


def test_bedroom_door_drawn_with_bathroom(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    def dummy_arrange_livingroom(w, h, rules, openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.canvas = BoundingCanvas(200, 200)
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv._draw = GenerateView._draw.__get__(gv)
    gv.zoom_factor = 1.0
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9

    gv._apply_openings_from_ui = lambda: True

    calls = {}
    orig = gv._draw_room_openings

    def spy(cv, openings, ox, oy, scale, wall_width, open_width, draw_door, room_name='bed'):
        if openings is gv.bed_openings:
            calls['bed'] = draw_door
        return orig(cv, openings, ox, oy, scale, wall_width, open_width, draw_door, room_name)

    gv._draw_room_openings = spy
    gv._solve_and_draw()

    assert calls.get('bed') is True


def test_door_rectangle_present(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            return self.plan, {
                'score': 1.0,
                'coverage': 0.5,
                'paths_ok': True,
                'reach_windows': True,
            }

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    def dummy_arrange_livingroom(w, h, rules, openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.canvas = BoundingCanvas(200, 200)
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv._draw = GenerateView._draw.__get__(gv)
    gv.zoom_factor = 1.0
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9

    gv._apply_openings_from_ui = lambda: True
    gv._solve_and_draw()

    door_items = [
        i
        for i in gv.canvas.items
        if i.get('fill') == DOOR_FILL and 'opening' in i.get('tags', ())
    ]
    assert door_items, 'Door rectangle not found'
    assert all(i.get('outline') == 'black' for i in door_items)


def test_opening_click_opens_dialog(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            return self.plan, {
                'score': 1.0,
                'coverage': 0.5,
                'paths_ok': True,
                'reach_windows': True,
            }

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', lambda *a, **k: GridPlan(1, 1))

    calls = []

    class DummyDialog:
        def __init__(self, root, info, apply_func):
            calls.append(info)

    monkeypatch.setattr(vastu_all_in_one, 'OpeningDialog', DummyDialog)

    class TagCanvas(BoundingCanvas):
        def __init__(self, width=200, height=200):
            super().__init__(width, height)
            self._id = 0

        def create_line(self, *args, **kwargs):
            self._id += 1
            self.items.append({'id': self._id, 'type': 'line', 'tags': kwargs.get('tags', ()), **{'bbox': (min(args[0], args[2]), min(args[1], args[3]), max(args[0], args[2]), max(args[1], args[3]))}})
            return self._id

        def create_rectangle(self, x0, y0, x1, y1, **kwargs):
            self._id += 1
            self.items.append({'id': self._id, 'type': 'rect', 'bbox': (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), 'tags': kwargs.get('tags', ()), **kwargs})
            return self._id

        def create_oval(self, x0, y0, x1, y1, **kwargs):
            self._id += 1
            self.items.append({'id': self._id, 'type': 'oval', 'bbox': (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), 'tags': kwargs.get('tags', ()), **kwargs})
            return self._id

        def create_text(self, x, y, **kwargs):
            self._id += 1
            self.items.append({'id': self._id, 'type': 'text', 'bbox': (x, y, x, y), 'tags': kwargs.get('tags', ()), **kwargs})
            return self._id

        def find_withtag(self, tag):
            return tuple(i['id'] for i in self.items if tag in i.get('tags', ()))

        def itemcget(self, item_id, attr):
            for i in self.items:
                if i['id'] == item_id:
                    return i.get(attr, '')
            return ''

        def type(self, item_id):
            for i in self.items:
                if i['id'] == item_id:
                    return i.get('type')
            return ''

        def addtag_withtag(self, tag, item_id):
            for i in self.items:
                if i['id'] == item_id and tag not in i.get('tags', ()): 
                    i['tags'] = i.get('tags', ()) + (tag,)

        def dtag(self, item_id, tag):
            for i in self.items:
                if i['id'] == item_id:
                    i['tags'] = tuple(t for t in i.get('tags', ()) if t != tag)

        def tag_raise(self, *args, **kwargs):
            pass

    gv = make_generate_view((2.0, 2.0))
    gv.canvas = TagCanvas(200, 200)
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv._draw = GenerateView._draw.__get__(gv)
    gv.zoom_factor = 1.0
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv._apply_openings_from_ui = lambda: True
    gv._solve_and_draw()

    door_id = None
    window_id = None
    for item in gv.canvas.find_withtag('opening'):
        fill = gv.canvas.itemcget(item, 'fill')
        outline = gv.canvas.itemcget(item, 'outline')
        assert outline == 'black'
        if fill == DOOR_FILL:
            door_id = item
        else:
            assert fill == WINDOW_FILL
            window_id = item

    assert door_id is not None and window_id is not None

    event = type('E', (), {'x': 0, 'y': 0})()

    gv.canvas.addtag_withtag('current', door_id)
    gv._on_canvas_click(event)
    gv.canvas.dtag(door_id, 'current')

    gv.canvas.addtag_withtag('current', window_id)
    gv._on_canvas_click(event)

    assert len(calls) == 2
    assert [info['type'] for info in calls] == ['door', 'window']


def test_bath_living_door_drawn_and_mirrored(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        p = GridPlan(w, h)
        return p

    def dummy_arrange_livingroom(w, h, rules, openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_livingroom', dummy_arrange_livingroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.canvas = BoundingCanvas(200, 200)
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv._draw = GenerateView._draw.__get__(gv)
    gv.zoom_factor = 1.0
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv.bath_liv_openings = None
    gv.liv_bath_openings = None

    gv.bed_openings.door_wall = WALL_RIGHT

    gv.bath_plan = GridPlan(*gv.bath_dims)
    gv.liv_plan = GridPlan(*gv.liv_dims)
    gv.bath_plan.x_offset = 0
    gv.bath_plan.y_offset = 0
    gv.liv_plan.x_offset = gv.bath_plan.gw
    gv.liv_plan.y_offset = 0

    gv._apply_openings_from_ui = GenerateView._apply_openings_from_ui.__get__(gv)
    assert gv._apply_openings_from_ui()
    assert gv.bath_liv_openings is not None
    assert gv.liv_bath_openings is not None

    gv._apply_openings_from_ui = lambda: True

    calls = {}
    orig = gv._draw_room_openings

    def spy(cv, openings, ox, oy, scale, wall_width, open_width, draw_door, room_name='bed'):
        if openings in (gv.bath_liv_openings, gv.liv_bath_openings):
            calls[openings] = draw_door
        return orig(cv, openings, ox, oy, scale, wall_width, open_width, draw_door, room_name)

    gv._draw_room_openings = spy
    gv._solve_and_draw()

    assert calls.get(gv.bath_liv_openings) is True
    assert calls.get(gv.liv_bath_openings) is True

def test_init_schedules_solver(monkeypatch):
    import vastu_all_in_one

    class DummyWidget:
        def __init__(self, *a, **k):
            pass
        def pack(self, *a, **k):
            return self
        def bind(self, *a, **k):
            pass

    monkeypatch.setattr(vastu_all_in_one.ttk, 'Frame', DummyWidget)
    monkeypatch.setattr(vastu_all_in_one.ttk, 'Button', DummyWidget)
    monkeypatch.setattr(vastu_all_in_one.ttk, 'Label', DummyWidget)
    monkeypatch.setattr(vastu_all_in_one.tk, 'Canvas', DummyWidget)
    monkeypatch.setattr(vastu_all_in_one.GenerateView, '_build_sidebar', lambda self: None)

    class DummyRoot:
        def __init__(self):
            self.after_idle_called_with = None
        def after_idle(self, func):
            self.after_idle_called_with = func
        def bind_all(self, *a, **k):
            pass

    root = DummyRoot()
    gv = vastu_all_in_one.GenerateView(root, 4.0, 4.0, None, bath_dims=None, liv_dims=None)
    assert root.after_idle_called_with == gv._solve_and_draw


class DummyCanvas:
    def __init__(self):
        self.next_id = 1
    def create_rectangle(self, *args, **kwargs):
        rid = self.next_id; self.next_id += 1; return rid
    def coords(self, *args, **kwargs):
        pass
    def delete(self, *args, **kwargs):
        pass
    def bind(self, *args, **kwargs):
        pass
    def focus_set(self):
        pass


def setup_drag_view(include_liv=False):
    gv = GenerateView.__new__(GenerateView)
    gv.bed_Wm = gv.bed_Hm = 3.0
    gv.bath_Wm = gv.bath_Hm = 3.0
    gv.bath_dims = (3.0, 3.0)
    if include_liv:
        gv.liv_Wm = gv.liv_Hm = 2.0
        gv.liv_dims = (2.0, 2.0)
    else:
        gv.liv_Wm = gv.liv_Hm = 0.0
        gv.liv_dims = None
    gv.bed_plan = GridPlan(gv.bed_Wm, gv.bed_Hm)
    gv.bath_plan = GridPlan(gv.bath_Wm, gv.bath_Hm)
    gv.liv_plan = GridPlan(gv.liv_Wm, gv.liv_Hm) if include_liv else None
    gv.Wm = max(gv.bed_Wm + gv.bath_Wm, gv.liv_Wm if include_liv else 0)
    gv.Hm = max(gv.bed_Hm, gv.bath_Hm) + (gv.liv_Hm if include_liv else 0)
    gv.plan = GridPlan(gv.Wm, gv.Hm)
    GenerateView._combine_plans(gv)
    gv.canvas = DummyCanvas()
    gv._draw = lambda: None
    gv._log_event = lambda *a, **k: None
    gv.selected = None
    gv.selected_locked = False
    gv.ox = gv.oy = 0
    gv.scale = 1
    gv.undo_stack = []
    gv.redo_stack = []
    return gv


def test_on_up_updates_only_bed_plan():
    gv = setup_drag_view()
    gv.bed_plan.place(0, 0, 1, 1, 'BED')
    GenerateView._combine_plans(gv)
    gv.drag_item = {
        'orig': [0, 0, 1, 1],
        'live': [1, 0, 1, 1],
        'code': 'BED',
        'room': 'bed',
        'ghost': None,
    }
    GenerateView._on_up(gv, type('E', (), {})())
    assert gv.bed_plan.occ[0][1] == 'BED'
    assert all(cell is None for row in gv.bath_plan.occ for cell in row)


def test_on_up_updates_only_bath_plan():
    gv = setup_drag_view()
    gv.bath_plan.place(0, 0, 1, 1, 'WC')
    GenerateView._combine_plans(gv)
    xoff = gv.bed_plan.gw
    gv.drag_item = {
        'orig': [xoff, 0, 1, 1],
        'live': [xoff + 1, 0, 1, 1],
        'code': 'WC',
        'room': 'bath',
        'ghost': None,
    }
    GenerateView._on_up(gv, type('E', (), {})())
    assert gv.bath_plan.occ[0][1] == 'WC'
    assert all('WC' not in row for row in gv.bed_plan.occ)


def test_on_up_updates_only_liv_plan():
    gv = setup_drag_view(include_liv=True)
    gv.liv_plan.place(0, 0, 1, 1, 'SOFA')
    GenerateView._combine_plans(gv)
    yoff = max(gv.bed_plan.gh, gv.bath_plan.gh)
    gv.drag_item = {
        'orig': [0, yoff, 1, 1],
        'live': [1, yoff, 1, 1],
        'code': 'SOFA',
        'room': 'living',
        'ghost': None,
    }
    GenerateView._on_up(gv, type('E', (), {})())
    assert gv.liv_plan.occ[0][1] == 'SOFA'
    assert all('SOFA' not in row for row in gv.bed_plan.occ)


def test_locked_bath_item_reapplied_after_generate(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.0, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0))

    bed = GridPlan(gv.bed_Wm, gv.bed_Hm)
    bath = GridPlan(gv.bath_dims[0], gv.bath_dims[1])
    bath.place(0, 0, 1, 1, 'WC')

    combined = GridPlan(bed.gw + bath.gw, max(bed.gh, bath.gh))
    for j in range(bed.gh):
        for i in range(bed.gw):
            combined.occ[j][i] = bed.occ[j][i]
    for j in range(bath.gh):
        for i in range(bath.gw):
            combined.occ[j][i + bed.gw] = bath.occ[j][i]

    gv.plan = combined
    gv.bed_plan = bed
    gv.bath_plan = bath
    gv.selected = {'rect': [bed.gw, 0, 1, 1], 'code': 'WC'}
    gv.selected_locked = True

    gv._apply_batch_and_generate()

    assert components_by_code(gv.bath_plan, 'WC'), 'Locked bathroom item missing after regenerate'


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
    gv.liv_dims = None
    gv.liv_plan = None
    gv.zoom_factor = tk.DoubleVar(value=1.0)

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
    gv.liv_dims = None
    gv.liv_plan = None
    gv.zoom_factor = tk.DoubleVar(value=1.0)

    for name in [
        '_solve_and_draw', '_apply_batch_and_generate', '_add_furniture',
        '_remove_furniture', '_simulate_one', '_simulate_two',
        'simulate_circulation', '_export_png']:
        setattr(gv, name, lambda *a, **kw: None)

    GenerateView._build_sidebar(gv)

    assert comboboxes[0].kwargs['values'] == ['Right']
    assert gv.bed_w1_wall.get() == 'Top'
    assert 'Right' not in comboboxes[1].kwargs['values']
    assert 'Right' not in comboboxes[2].kwargs['values']

    assert comboboxes[3].kwargs['values'] == ['Left']
    assert 'Left' not in comboboxes[4].kwargs['values']
    assert 'Left' not in comboboxes[5].kwargs['values']


def test_mirrored_clearances_align_by_label(monkeypatch):
    import vastu_all_in_one

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {'score': 1.0, 'coverage': 0.5, 'paths_ok': True, 'reach_windows': True}

    def dummy_arrange_bathroom(w, h, rules, openings=None, secondary_openings=None, rng=None):
        return GridPlan(w, h)

    monkeypatch.setattr(vastu_all_in_one, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(vastu_all_in_one, 'arrange_bathroom', dummy_arrange_bathroom)

    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 3.0))
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_center = 1.0
    gv.bath_openings.door_width = 0.9
    gv.liv_openings.door_wall = WALL_BOTTOM
    gv.liv_openings.door_center = 1.0
    gv.liv_openings.door_width = 0.9

    gv._solve_and_draw()

    bath_clear = next(
        (x, y, w, h)
        for x, y, w, h, kind, owner in gv.bath_plan.clearzones
        if kind == 'DOOR_CLEAR' and owner == 'BATHROOM_DOOR'
    )
    shared_op = Openings(gv.bed_plan)
    shared_op.door_wall = WALL_RIGHT
    shared_op.door_center = gv.bath_openings.door_center
    shared_op.door_width = gv.bath_openings.door_width
    shared_op.swing_depth = gv.bath_openings.swing_depth
    _, start, _ = shared_op.door_span_cells()
    depth = gv.bed_plan.meters_to_cells(shared_op.swing_depth)
    outside_x = gv.bed_plan.gw + depth
    outside_y = start
    bed_label = gv.bed_plan.coord_to_label(outside_x, outside_y)
    bath_label = gv.bath_plan.coord_to_label(bath_clear[0], bath_clear[1])

    assert bed_label == bath_label


def test_place_rejects_door_clear_overlap():
    plan = GridPlan(2.0, 2.0)
    plan.mark_clear(0, 0, 1, 1, 'DOOR_CLEAR', 'TEST')
    with pytest.raises(ValueError):
        plan.place(0, 0, 1, 1, 'BED')


def test_mark_clear_removes_occupied_region():
    plan = GridPlan(2.0, 2.0)
    plan.place(0, 0, 1, 1, 'BED')
    plan.mark_clear(0, 0, 1, 1, 'DOOR_CLEAR', 'TEST')
    assert plan.occ[0][0] is None


def test_grid_labels_fully_visible():
    plan = GridPlan(4.0, 4.0, column_grid=ColumnGrid(4, 4))
    gv = GenerateView.__new__(GenerateView)
    gv.bed_plan = plan
    gv.bath_plan = None
    gv.liv_plan = None
    gv.plan = plan
    gv.bed_openings = Openings(plan)
    gv.bath_openings = None
    gv.canvas = BoundingCanvas(200, 200)
    gv.grid_overlay = ColumnGridOverlay(gv.canvas)
    gv.sim_poly = gv.sim2_poly = []
    gv.sim_path = gv.sim2_path = []
    gv.sim_index = gv.sim2_index = 0
    gv.zoom_factor = 1.0
    gv._draw()
    bbox = gv.canvas.bbox('all')
    assert bbox[0] >= 0 and bbox[1] >= 0
    assert bbox[2] <= gv.canvas.width and bbox[3] <= gv.canvas.height
