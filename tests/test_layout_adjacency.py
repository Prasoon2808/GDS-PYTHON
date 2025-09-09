import itertools
import os
import sys

# Ensure repository root importable when tests run from this directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GenerateView,
    GridPlan,
    shares_edge,
    overlaps,
    CELL_M,
    Openings,
    WALL_RIGHT,
    WALL_LEFT,
    WALL_BOTTOM,
    WALL_TOP,
)
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

    if gv.liv_plan and gv.kitch_plan and getattr(gv, "liv_kitch_openings", None):
        def _shared_wall(a: GridPlan, b: GridPlan) -> int:
            ax0, ay0 = a.x_offset, a.y_offset
            ax1, ay1 = ax0 + a.gw, ay0 + a.gh
            bx0, by0 = b.x_offset, b.y_offset
            bx1, by1 = bx0 + b.gw, by0 + b.gh
            if ax1 == bx0 and max(ay0, by0) < min(ay1, by1):
                return WALL_RIGHT
            if bx1 == ax0 and max(ay0, by0) < min(ay1, by1):
                return WALL_LEFT
            if ay1 == by0 and max(ax0, bx0) < min(ax1, bx1):
                return WALL_TOP
            if by1 == ay0 and max(ax0, bx0) < min(ax1, bx1):
                return WALL_BOTTOM
            return WALL_BOTTOM

        def _has_door(p: GridPlan, wall: int) -> bool:
            if wall == WALL_LEFT:
                return any(p.occ[j][0] == "DOOR" for j in range(p.gh))
            if wall == WALL_RIGHT:
                return any(p.occ[j][p.gw - 1] == "DOOR" for j in range(p.gh))
            if wall == WALL_BOTTOM:
                return any(p.occ[0][i] == "DOOR" for i in range(p.gw))
            if wall == WALL_TOP:
                return any(p.occ[p.gh - 1][i] == "DOOR" for i in range(p.gw))
            return False

        shared_wall = _shared_wall(gv.kitch_plan, gv.liv_plan)
        if not _has_door(gv.liv_plan, shared_wall):
            gv.status.set("Living and Kitchen must share a door")
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


def test_deleting_liv_kitch_door_triggers_failure():
    cell = CELL_M
    gv = make_generate_view((cell, cell), living_dims=(cell, cell))
    gv.bed_plan = GridPlan(cell, cell)
    gv.bath_plan = GridPlan(cell, cell)
    gv.liv_plan = GridPlan(cell, cell)
    gv.kitch_plan = GridPlan(cell, cell)

    gv.bed_plan.x_offset = 0
    gv.bed_plan.y_offset = 0
    gv.bath_plan.x_offset = 1
    gv.bath_plan.y_offset = 0
    gv.liv_plan.x_offset = 0
    gv.liv_plan.y_offset = 1
    gv.kitch_plan.x_offset = 1
    gv.kitch_plan.y_offset = 1

    gv.liv_kitch_openings = Openings(gv.liv_plan)
    gv.liv_kitch_openings.door_wall = WALL_RIGHT
    gv.liv_kitch_openings.door_width = cell
    gv.kitch_liv_openings = Openings(gv.kitch_plan)
    gv.kitch_liv_openings.door_wall = WALL_LEFT
    gv.kitch_liv_openings.door_width = cell

    for op, plan in (
        (gv.liv_kitch_openings, gv.liv_plan),
        (gv.kitch_liv_openings, gv.kitch_plan),
    ):
        dx, dy, dw, dh = op.door_rect_cells()
        for j in range(dy, dy + dh):
            for i in range(dx, dx + dw):
                plan.occ[j][i] = "DOOR"

    assert layout_and_check(gv)

    for plan in (gv.liv_plan, gv.kitch_plan):
        for j in range(plan.gh):
            for i in range(plan.gw):
                if plan.occ[j][i] == "DOOR":
                    plan.occ[j][i] = None

    assert not layout_and_check(gv)
    assert gv.status.msg == "Living and Kitchen must share a door"
