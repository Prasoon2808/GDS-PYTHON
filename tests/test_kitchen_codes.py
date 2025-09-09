import os
import sys

# Ensure repository root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GenerateView, PALETTE, ITEM_LABELS
from test_generate_view import setup_drag_view


def test_kitchen_codes_and_selection():
    codes = {
        'SINK', 'COOK', 'REFR', 'DW', 'ISLN',
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

