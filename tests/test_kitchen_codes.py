import os
import sys

# Ensure repository root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver import (
    GenerateView,
    PALETTE,
    ITEM_LABELS,
    BedroomSolver,
    GridPlan,
    GRID_CODE_MAPPING,
)
from test_generate_view import setup_drag_view


def test_kitchen_codes_and_selection():
    codes = {
        'SINK', 'COOK', 'SLAB', 'REF', 'DW', 'ISLN',
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


def test_grid_code_mapping_and_snapshots_consistent():
    solver = BedroomSolver.__new__(BedroomSolver)
    view = GenerateView.__new__(GenerateView)
    expected = {
        'SINK': 7,
        'COOK': 8,
        'SLAB': 9,
        'REF': 10,
        'DW': 11,
        'ISLN': 12,
        'BASE': 13,
        'WALL': 14,
        'HOOD': 15,
        'OVEN': 16,
        'MICRO': 17,
    }
    # Ensure the module-level mapping contains all kitchen codes with expected values
    for code, val in expected.items():
        assert GRID_CODE_MAPPING[code] == val

    # Both snapshot implementations should reference the same mapping
    for code, val in expected.items():
        plan = GridPlan(1.0, 1.0)
        plan.place(0, 0, 1, 1, code)
        grid_solver = solver._grid_snapshot(plan, max_hw=1)
        grid_view = view._grid_snapshot(plan, max_hw=1)
        assert grid_solver[0, 0] == val
        assert grid_view[0, 0] == val
        assert grid_solver[0, 0] == grid_view[0, 0]

