from dataclasses import dataclass
from typing import List, Optional, Tuple

from gds import (
    GridPlan,
    Openings,
    CELL_M,
    WALL_LEFT,
    WALL_RIGHT,
    WALL_TOP,
    WALL_BOTTOM,
    opposite_wall,
    shared_wall,
)


@dataclass
class MultiRectRoom:
    """A room composed of multiple rectangular ``GridPlan`` segments."""

    segments: List[GridPlan]

    def total_area(self) -> float:
        """Return the combined area of all segments in square metres."""
        return sum(seg.Wm * seg.Hm for seg in self.segments)

    def meets_constraints(
        self, min_area: float, min_width: float, min_height: float
    ) -> bool:
        """Check total area and per-segment dimension constraints.

        ``min_area`` applies to the sum of all segments while ``min_width`` and
        ``min_height`` must be satisfied by each individual segment."""

        if self.total_area() < min_area:
            return False
        for seg in self.segments:
            if seg.Wm < min_width or seg.Hm < min_height:
                return False
        return True

    def segment_for_neighbor(self, neighbor: GridPlan) -> Tuple[Optional[GridPlan], int]:
        """Return the segment and wall that touches ``neighbor``."""
        for seg in self.segments:
            wall = shared_wall(seg, neighbor)
            if wall != -1:
                return seg, wall
        return None, -1

    def place_door_to(self, neighbor: GridPlan, width: float = CELL_M) -> None:
        """Place a door between this room and ``neighbor``.

        The method selects the segment that borders ``neighbor`` and carves out a
        door of ``width`` metres on the shared wall.  A matching door is also
        carved on the neighbour's opposing wall."""

        segment, wall = self.segment_for_neighbor(neighbor)
        if segment is None:
            raise ValueError("No adjacent segment available for door placement")
        op = Openings(segment)
        op.door_wall = wall
        op.door_width = width
        dx, dy, dw, dh = op.door_rect_cells()
        for j in range(dy, dy + dh):
            for i in range(dx, dx + dw):
                segment.occ[j][i] = "DOOR"
        # mirror door on neighbour
        op2 = Openings(neighbor)
        op2.door_wall = opposite_wall(wall)
        op2.door_width = width
        dx, dy, dw, dh = op2.door_rect_cells()
        for j in range(dy, dy + dh):
            for i in range(dx, dx + dw):
                neighbor.occ[j][i] = "DOOR"
