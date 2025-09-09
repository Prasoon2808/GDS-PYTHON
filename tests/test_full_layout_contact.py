import os
import sys
import pytest
import tkinter as tk

# Ensure repository root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GenerateView,
    GridPlan,
    Openings,
    WALL_LEFT,
    WALL_RIGHT,
    WALL_TOP,
    WALL_BOTTOM,
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
    gv.bath_plan = GridPlan(gv.bath_Wm, gv.bath_Hm)
    gv.plan = GridPlan(gv.bed_Wm + gv.bath_Wm + gv.liv_Wm, max(gv.bed_Hm, gv.bath_Hm) + gv.liv_Hm)
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bath_liv_openings.door_wall = WALL_BOTTOM

    assert gv._apply_openings_from_ui()

    GenerateView._combine_plans(gv)

    assert gv.liv_plan.y_offset == max(gv.bed_plan.gh, gv.bath_plan.gh)
    assert gv.liv_plan.x_offset == 0
    assert gv.liv_plan.gw >= gv.bed_plan.gw + gv.bath_plan.gw
    assert gv.bed_openings.door_wall == WALL_BOTTOM
    assert gv.bath_openings.door_wall == WALL_LEFT
    assert gv.bath_liv_openings.door_wall == WALL_BOTTOM


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


def test_missing_bathroom_living_door():
    gv = make_generate_view((2.0, 2.0), living_dims=(6.0, 2.0))
    gv.bath_liv_openings = None
    gv._apply_openings_from_ui = _alignment_stub_factory(gv)

    gv.bed_openings.door_wall = WALL_BOTTOM
    gv.bath_openings.door_wall = WALL_LEFT

    assert not gv._apply_openings_from_ui()
    assert gv.status.msg == 'Bathroom must expose door to living room.'
