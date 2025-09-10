import os
import sys
import pytest
import tkinter as tk

# Ensure repository root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gds import (
    GenerateView,
    GridPlan,
    Openings,
    WALL_LEFT,
    WALL_RIGHT,
    WALL_TOP,
    WALL_BOTTOM,
    opposite_wall,
    CELL_M,
)

from test_generate_view import make_generate_view


def _alignment_stub_factory(gv):
    def _apply_stub():
        if gv.bed_openings.door_wall != WALL_BOTTOM:
            gv.status.set('Bedroom door must open to living room.')
            return False
        if gv.bath_openings.door_wall != WALL_LEFT:
            gv.status.set('Bathroom must expose door to bedroom.')
            return False
        if gv.bath_liv_openings is None or gv.bath_liv_openings.door_wall != WALL_BOTTOM:
            gv.status.set('Bathroom must expose door to living room.')
            return False
        return True

    return _apply_stub


def test_combined_plan_living_contact():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv.bed_plan = GridPlan(gv.bed_Wm, gv.bed_Hm)
    gv.bed_plan.place(0, 0, 1, 1, 'BED')
    gv.bath_plan = GridPlan(gv.bath_Wm, gv.bath_Hm)
    gv.liv_plan.place(0, 0, 1, 1, 'SOFA')
    gv.plan = GridPlan(gv.bed_Wm + gv.bath_Wm + gv.liv_Wm, max(gv.bed_Hm, gv.bath_Hm) + gv.liv_Hm)
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_liv_openings.door_wall = WALL_BOTTOM

    assert gv._apply_openings_from_ui()

    pre_wall = gv.bath_liv_openings.door_wall
    GenerateView._combine_plans(gv)

    assert gv.liv_plan.y_offset == max(gv.bed_plan.gh, gv.bath_plan.gh)
    assert gv.liv_plan.x_offset == 0
    assert gv.liv_plan.gw >= gv.bed_plan.gw
    assert gv.bed_openings.door_wall == WALL_BOTTOM
    assert gv.bath_openings.door_wall == WALL_LEFT
    assert gv.bath_liv_openings.door_wall == pre_wall
    assert opposite_wall(pre_wall) == WALL_TOP


def test_bedroom_door_misaligned():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_LEFT
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_liv_openings.door_wall = WALL_BOTTOM

    assert not gv._apply_openings_from_ui()
    assert gv.status.msg == 'Bedroom door must open to living room.'


def test_bathroom_doors_misaligned():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_BOTTOM
    gv.bath_liv_openings.door_wall = WALL_BOTTOM

    assert not gv._apply_openings_from_ui()
    assert gv.status.msg == 'Bathroom must expose door to bedroom.'

    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_liv_openings.door_wall = WALL_LEFT

    assert not gv._apply_openings_from_ui()
    assert gv.status.msg == 'Bathroom must expose door to living room.'



def test_living_room_too_narrow():
    gv = make_generate_view((2.0, 2.0), living_dims=(3.0, 1.0))
    gv._validate_living_dims()
    assert gv.liv_Wm == pytest.approx(3.0)
    assert not gv.liv_auto_adjusted


def test_living_room_too_shallow():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 0.2))
    gv._validate_living_dims()
    required = max(0.60, CELL_M)
    assert gv.liv_Hm == pytest.approx(required)
    assert gv.liv_auto_adjusted


def test_living_room_dims_ok():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 1.0))
    gv._validate_living_dims()  # should not raise
    assert not gv.liv_auto_adjusted


def test_narrow_living_room_abuts_kitchen():
    gv = make_generate_view((2.0, 2.0), living_dims=(2.0, 2.0))
    gv._validate_living_dims()
    gv.bed_plan = GridPlan(gv.bed_Wm, gv.bed_Hm)
    gv.bed_plan.place(0, 0, 1, 1, "BED")
    gv.liv_plan = GridPlan(gv.liv_Wm, gv.liv_Hm)
    gv.liv_plan.place(0, 0, 1, 1, "SOFA")
    gv.kitch_plan = GridPlan(2.0, 2.0)
    gv.kitch_plan.place(0, 0, 1, 1, "SINK")
    gv.kitch_Wm = gv.kitch_Hm = 2.0

    GenerateView._combine_plans(gv)

    assert gv.liv_plan.x_offset == gv.bed_plan.gw - gv.liv_plan.gw
    assert gv.liv_plan.x_offset + gv.liv_plan.gw == gv.kitch_plan.x_offset
    assert gv.plan.occ[gv.liv_plan.y_offset][gv.liv_plan.x_offset] == "SOFA"
    assert gv.plan.occ[gv.kitch_plan.y_offset][gv.kitch_plan.x_offset] == "SINK"

def test_missing_bathroom_living_door():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv.bath_liv_openings = None
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_LEFT

    assert not gv._apply_openings_from_ui()
    assert gv.status.msg == 'Bathroom must expose door to living room.'


def test_living_invalid_without_shared_door(monkeypatch):
    import gds

    gv = make_generate_view((2.0, 5.0), living_dims=(6.0, 2.0))
    gv._apply_openings_from_ui = lambda: True
    gv.bath_liv_openings = None
    gv.liv_bath_openings = None

    class DummyBedroomSolver:
        def __init__(self, plan, *a, **k):
            self.plan = plan

        def run(self):
            self.plan.place(0, 0, 1, 1, 'BED')
            return self.plan, {}

    monkeypatch.setattr(gds, 'BedroomSolver', DummyBedroomSolver)
    monkeypatch.setattr(
        gds, 'arrange_bathroom', lambda *a, **k: GridPlan(*gv.bath_dims)
    )
    monkeypatch.setattr(
        gds, 'arrange_livingroom', lambda *a, **k: GridPlan(*gv.liv_dims)
    )
    monkeypatch.setattr(gds, 'shares_edge', lambda a, b: True)

    gv._solve_and_draw()
    assert gv.liv_plan is None
