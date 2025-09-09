import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import arrange_bathroom, BATH_RULES, CELL_M


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


def test_arrange_bathroom_respects_clearances():
    plan = arrange_bathroom(3.0, 3.5, BATH_RULES)

    # fixtures should be present
    for code in ("WC", "LAV", "TUB", "SHR"):
        assert _find_rect(plan, code) is not None

    # clearance rectangles must remain empty
    for x, y, w, h, _, _ in plan.clearzones:
        for j in range(y, y + h):
            for i in range(x, x + w):
                assert plan.occ[j][i] is None

    wc = _find_rect(plan, "WC")
    lav = _find_rect(plan, "LAV")
    gap_cells = lav[0] - (wc[0] + wc[2])
    required = BATH_RULES["fixtures"]["lavatory"]["to_adjacent_fixture_edge_m"]["min"]
    assert gap_cells * CELL_M >= required


def test_arrange_bathroom_deterministic_and_uses_rule_sizes():
    plan1 = arrange_bathroom(3.0, 3.5, BATH_RULES)
    plan2 = arrange_bathroom(3.0, 3.5, BATH_RULES)

    assert plan1.occ == plan2.occ
    assert plan1.clearzones == plan2.clearzones

    tub = _find_rect(plan1, "TUB")
    shr = _find_rect(plan1, "SHR")

    fx = BATH_RULES["fixtures"]
    in_m = BATH_RULES.get("units", {}).get("IN_M", 0.0254)
    expected_tub = plan1.meters_to_cells(min(fx["bathtub"]["common_lengths_m"]))
    expected_shr = plan1.meters_to_cells(min(s["w"] * in_m for s in fx["shower"]["stall_nominal_sizes_in"]))

    assert tub[2] == expected_tub
    assert shr[2] == expected_shr


def test_arrange_bathroom_partial_plan_when_space_limited():
    """All core fixtures are instantiated even when space is tight."""
    with pytest.warns(UserWarning):
        plan = arrange_bathroom(1.2, 2.0, BATH_RULES)

    # Tub is wider than the room; it should be omitted
    assert _find_rect(plan, "TUB") is None

    # Core fixtures should still appear
    for code in ("WC", "LAV", "SHR"):
        assert _find_rect(plan, code) is not None

