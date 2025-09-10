import os
import sys
import pytest

# Ensure repository root is importable when running tests directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver import GenerateView, GridPlan
from rules import CELL_M


def test_combine_plans_handles_offsets_without_index_error():
    side = CELL_M * 1.5  # produces a 2x2 grid with rounding
    gv = GenerateView.__new__(GenerateView)

    # Per-room plans with non-integral metre dimensions
    gv.bed_plan = GridPlan(side, side)
    gv.bath_plan = GridPlan(side, side)
    gv.liv_plan = GridPlan(side, side)
    gv.kitch_plan = GridPlan(side, side)

    # Place items at the furthest cells to ensure indexing occurs at offsets
    gv.bed_plan.place(gv.bed_plan.gw - 1, 0, 1, 1, "BED")
    gv.bath_plan.place(gv.bath_plan.gw - 1, 0, 1, 1, "WC")
    gv.liv_plan.place(0, gv.liv_plan.gh - 1, 1, 1, "SOFA")
    gv.kitch_plan.place(gv.kitch_plan.gw - 1, gv.kitch_plan.gh - 1, 1, 1, "SINK")

    try:
        GenerateView._combine_plans(gv)
    except IndexError:
        pytest.fail("_combine_plans raised IndexError")


def test_living_beside_bedroom_layout():
    gv = GenerateView.__new__(GenerateView)
    gv.bed_plan = GridPlan(4.0, 4.0)
    gv.bath_plan = GridPlan(2.0, 2.0)
    gv.liv_plan = GridPlan(3.0, 4.0)  # as tall as bedroom -> placed to the right
    gv.kitch_plan = GridPlan(2.0, 2.0)

    GenerateView._combine_plans(gv)

    assert gv.liv_plan.x_offset == gv.bed_plan.gw
    assert gv.bath_plan.y_offset == gv.bed_plan.gh
    assert gv.kitch_plan.x_offset == gv.liv_plan.x_offset
    assert gv.bath_plan.x_offset + gv.bath_plan.gw == gv.kitch_plan.x_offset


def test_kitchen_below_bath_in_corridor_layout():
    gv = GenerateView.__new__(GenerateView)
    gv.bed_plan = GridPlan(4.0, 4.0)
    gv.bath_plan = GridPlan(2.0, 2.0)
    gv.liv_plan = GridPlan(6.0, 2.0)  # shallow living -> corridor below
    gv.kitch_plan = GridPlan(2.0, 2.0)

    GenerateView._combine_plans(gv)

    assert gv.liv_plan.y_offset == gv.bed_plan.gh
    assert gv.kitch_plan.y_offset == gv.liv_plan.y_offset
    assert gv.bath_plan.y_offset == 0
    assert gv.kitch_plan.x_offset == gv.bath_plan.x_offset
