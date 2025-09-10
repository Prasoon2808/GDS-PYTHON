import os
import sys

# Ensure repository root importable when tests run from this directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gds import GridPlan, CELL_M, WALL_LEFT, WALL_RIGHT
from multiroom import MultiRectRoom


def test_multi_segment_room_doors_and_constraints():
    cell = CELL_M
    seg1 = GridPlan(2 * cell, cell)
    seg1.x_offset = 1
    seg1.y_offset = 0
    seg2 = GridPlan(2 * cell, cell)
    seg2.x_offset = 4
    seg2.y_offset = 0
    room = MultiRectRoom([seg1, seg2])

    assert room.meets_constraints(
        min_area=4 * cell * cell, min_width=cell, min_height=cell
    )

    left_neighbor = GridPlan(cell, cell)
    left_neighbor.x_offset = 0
    left_neighbor.y_offset = 0
    right_neighbor = GridPlan(cell, cell)
    right_neighbor.x_offset = 6
    right_neighbor.y_offset = 0

    seg, wall = room.segment_for_neighbor(left_neighbor)
    assert seg is seg1 and wall == WALL_LEFT
    seg, wall = room.segment_for_neighbor(right_neighbor)
    assert seg is seg2 and wall == WALL_RIGHT

    room.place_door_to(left_neighbor, width=cell)
    room.place_door_to(right_neighbor, width=cell)

    assert seg1.occ[0][0] == "DOOR"
    assert seg2.occ[0][seg2.gw - 1] == "DOOR"
    assert left_neighbor.occ[0][left_neighbor.gw - 1] == "DOOR"
    assert right_neighbor.occ[0][0] == "DOOR"
