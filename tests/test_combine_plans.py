import pytest

from vastu_all_in_one import GenerateView, GridPlan, CELL_M


def test_combine_plans_handles_offsets_without_index_error():
    side = CELL_M * 1.5  # produces a 2x2 grid with rounding
    gv = GenerateView.__new__(GenerateView)

    # Per-room plans with non-integral metre dimensions
    gv.bed_plan = GridPlan(side, side)
    gv.bath_plan = GridPlan(side, side)
    gv.liv_plan = GridPlan(side, side)

    # Place items at the furthest cells to ensure indexing occurs at offsets
    gv.bed_plan.place(gv.bed_plan.gw - 1, 0, 1, 1, "BED")
    gv.bath_plan.place(gv.bath_plan.gw - 1, 0, 1, 1, "WC")
    gv.liv_plan.place(0, gv.liv_plan.gh - 1, 1, 1, "SOFA")

    try:
        GenerateView._combine_plans(gv)
    except IndexError:
        pytest.fail("_combine_plans raised IndexError")
