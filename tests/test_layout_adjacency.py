import itertools
import os
import sys

# Ensure repository root importable when tests run from this directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GenerateView, GridPlan, shares_edge, overlaps, CELL_M
from test_generate_view import make_generate_view


def layout_and_check(gv):
    """Apply overlap and adjacency checks similar to ``_solve_and_draw``.

    Returns ``True`` when plans are valid and merged, otherwise ``False`` and
    ``gv.status.msg`` holds the failure reason.
    """
    room_plans = [
        (gv.bed_plan, "Bedroom"),
        (gv.bath_plan, "Bathroom"),
        (gv.liv_plan, "Living"),
        (gv.kitch_plan, "Kitchen"),
    ]
    for (plan_a, name_a), (plan_b, name_b) in itertools.combinations(
        [rp for rp in room_plans if rp[0]], 2
    ):
        if overlaps(plan_a, plan_b):
            gv.status.set(f"Rooms {name_a} and {name_b} overlap")
            return False

    if gv.kitch_plan and gv.bath_plan and gv.liv_plan:
        if not (
            shares_edge(gv.kitch_plan, gv.bath_plan)
            and shares_edge(gv.kitch_plan, gv.liv_plan)
        ):
            gv.status.set(
                "Kitchen must share an edge with BOTH Living and Bathroom. Currently it does not."
            )
            return False

    GenerateView._combine_plans(gv)
    return True


def test_positive_merge():
    cell = CELL_M
    gv = make_generate_view((cell, cell), living_dims=(2 * cell, cell))
    gv.bed_plan = GridPlan(cell, cell)
    gv.bath_plan = GridPlan(cell, cell)
    gv.liv_plan = GridPlan(2 * cell, cell)
    gv.kitch_plan = GridPlan(cell, cell)

    gv.bed_plan.x_offset = 0
    gv.bed_plan.y_offset = 0
    gv.bath_plan.x_offset = 2
    gv.bath_plan.y_offset = 0
    gv.liv_plan.x_offset = 0
    gv.liv_plan.y_offset = 1
    gv.kitch_plan.x_offset = 2
    gv.kitch_plan.y_offset = 1

    assert layout_and_check(gv)
    assert gv.status.msg == ""


def test_kitchen_shift_breaks_adjacency():
    cell = CELL_M
    gv = make_generate_view((cell, cell), living_dims=(2 * cell, cell))
    gv.bed_plan = GridPlan(cell, cell)
    gv.bath_plan = GridPlan(cell, cell)
    gv.liv_plan = GridPlan(2 * cell, cell)
    gv.kitch_plan = GridPlan(cell, cell)

    gv.bed_plan.x_offset = 0
    gv.bed_plan.y_offset = 0
    gv.bath_plan.x_offset = 2
    gv.bath_plan.y_offset = 0
    gv.liv_plan.x_offset = 0
    gv.liv_plan.y_offset = 1
    gv.kitch_plan.x_offset = 3  # shifted one cell to the right
    gv.kitch_plan.y_offset = 1

    assert not layout_and_check(gv)
    assert (
        gv.status.msg
        == "Kitchen must share an edge with BOTH Living and Bathroom. Currently it does not."
    )


def test_living_room_overlap_raises_error():
    cell = CELL_M
    gv = make_generate_view((cell, cell), living_dims=(2 * cell, cell))
    gv.bed_plan = GridPlan(cell, cell)
    gv.bath_plan = GridPlan(cell, cell)
    gv.liv_plan = GridPlan(3 * cell, cell)
    gv.kitch_plan = GridPlan(cell, cell)

    gv.bed_plan.x_offset = 0
    gv.bed_plan.y_offset = 2
    gv.bath_plan.x_offset = 2
    gv.bath_plan.y_offset = 0
    gv.liv_plan.x_offset = 0
    gv.liv_plan.y_offset = 0  # intrudes into bathroom space horizontally
    gv.kitch_plan.x_offset = 2
    gv.kitch_plan.y_offset = 1

    assert not layout_and_check(gv)
    assert gv.status.msg == "Rooms Bathroom and Living overlap"
