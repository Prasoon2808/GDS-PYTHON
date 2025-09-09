import os
import sys

# Ensure repository root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GenerateView,
    PALETTE,
    ITEM_LABELS,
    BedroomSolver,
    GridPlan,
)
from test_generate_view import setup_drag_view


def test_kitchen_codes_and_selection():
    codes = {
        'SINK', 'COOK', 'REF', 'DW', 'ISLN',
        'BASE', 'WALL', 'HOOD', 'OVEN', 'MICRO'
    }
    assert GenerateView.KITCH_CODES == codes
    for code in codes:
        assert code in PALETTE
        assert code in ITEM_LABELS

    gv = setup_drag_view()
    gv.plan.place(0, 0, 1, 1, 'SINK')
    x0, y0, x1, y1 = GenerateView._cell_rect(gv, 0, 0)
    event = type('E', (), {'x': (x0 + x1) / 2, 'y': (y0 + y1) / 2})()
    GenerateView._on_down(gv, event)
    assert gv.selected['code'] == 'SINK'


def test_grid_snapshot_assigns_unique_ints_to_kitchen_codes():
    solver = BedroomSolver.__new__(BedroomSolver)
    expected = {
        'SINK': 7,
        'COOK': 8,
        'REF': 9,
        'DW': 10,
        'ISLN': 11,
        'BASE': 12,
        'WALL': 13,
        'HOOD': 14,
        'OVEN': 15,
        'MICRO': 16,
    }
    for code, val in expected.items():
        plan = GridPlan(1.0, 1.0)
        plan.place(0, 0, 1, 1, code)
        grid = solver._grid_snapshot(plan, max_hw=1)
        assert grid[0, 0] == val

