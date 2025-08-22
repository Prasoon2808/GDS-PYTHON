import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import arrange_livingroom, LIV_RULES


def _find_rect(plan, code):
    rect = None
    for y, row in enumerate(plan.occ):
        for x, c in enumerate(row):
            if c == code:
                if rect is None:
                    rect = [x, y, x, y]
                else:
                    rect[0] = min(rect[0], x)
                    rect[1] = min(rect[1], y)
                    rect[2] = max(rect[2], x)
                    rect[3] = max(rect[3], y)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def test_arrange_livingroom_respects_clearances():
    plan = arrange_livingroom(5.0, 4.0, LIV_RULES)

    # core furniture should be present
    assert _find_rect(plan, "SOFA") is not None
    assert _find_rect(plan, "CTAB") is not None
    assert _find_rect(plan, "STAB") is not None

    # clearance rectangles (except rug) must remain empty
    for x, y, w, h, kind, _ in plan.clearzones:
        if kind == 'RUG':
            continue
        for j in range(y, y + h):
            for i in range(x, x + w):
                assert plan.occ[j][i] is None

    # ensure coffee table front clearance exists
    assert any(kind == 'FRONT' and owner == 'CTAB' for x, y, w, h, kind, owner in plan.clearzones)


def test_arrange_livingroom_deterministic_and_uses_rule_sizes():
    plan1 = arrange_livingroom(5.0, 4.0, LIV_RULES)
    plan2 = arrange_livingroom(5.0, 4.0, LIV_RULES)

    assert plan1.occ == plan2.occ
    assert plan1.clearzones == plan2.clearzones

    sofa = _find_rect(plan1, "SOFA")
    expected_sw = plan1.meters_to_cells(min(LIV_RULES["furniture_size_ranges"]["sofas"]["length_m_range"]))
    assert sofa[2] == expected_sw


def test_arrange_livingroom_partial_plan_when_space_limited():
    plan = arrange_livingroom(1.5, 3.5, LIV_RULES)

    assert _find_rect(plan, "SOFA") is None
    assert _find_rect(plan, "CTAB") is not None

