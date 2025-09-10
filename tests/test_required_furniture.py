import os
import sys
import pytest

# Ensure repository root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gds import (
    GenerateView,
    GridPlan,
    WALL_RIGHT,
)
from test_generate_view import setup_drag_view, make_generate_view


def test_generated_plan_contains_required_furniture():
    gv = setup_drag_view(include_liv=True, include_kitch=True)
    gv.bed_plan.place(0, 0, 1, 1, 'BED')
    gv.bath_plan.place(0, 0, 1, 1, 'WC')
    gv.liv_plan.place(0, 0, 1, 1, 'SOFA')
    gv.kitch_plan.place(0, 0, 1, 1, 'SINK')
    GenerateView._combine_plans(gv)
    codes = {c for row in gv.plan.occ for c in row if c}
    assert {'BED', 'SOFA', 'SINK', 'WC'} <= codes


def test_missing_bed_raises(monkeypatch):
    import gds

    class DummyBedroomSolver:
        def __init__(self, plan, *args, **kwargs):
            self.plan = plan

        def run(self):
            return self.plan, {}

    monkeypatch.setattr(gds, 'BedroomSolver', DummyBedroomSolver)

    gv = make_generate_view()
    gv.bed_openings.door_wall = WALL_RIGHT
    gv._apply_openings_from_ui = lambda: True

    gv._solve_and_draw()
    codes = {c for row in gv.plan.occ for c in row if c}
    assert 'BED' not in codes
