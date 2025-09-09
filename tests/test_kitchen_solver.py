import random
import os, sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import (
    GridPlan,
    Openings,
    KitchenSolver,
    default_kitchen_sets,
    components_by_code,
    dot_score,
)


def test_default_sets_include_work_triangle():
    sets = default_kitchen_sets()
    assert sets[0] == ('SINK',)
    assert sets[-1] == ('SINK', 'COOK', 'REF')


def test_solver_reports_triangle_bonus():
    plan = GridPlan(8.0, 8.0)
    plan.place(0, 0, 1, 1, 'SINK')
    plan.place(8, 0, 1, 1, 'COOK')
    plan.place(0, 8, 1, 1, 'REF')
    openings = Openings(plan)
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights={})
    result, meta = solver.run(appliance_sets=[('SINK', 'COOK', 'REF')])
    assert result is not None
    feats = meta.get('features', {})
    assert feats.get('work_triangle_bonus', 0.0) == 1.0


def test_work_triangle_bonus_drops_when_appliances_far():
    plan = GridPlan(8.0, 8.0)
    plan.place(0, 0, 1, 1, 'SINK')
    plan.place(8, 0, 1, 1, 'COOK')
    plan.place(0, 20, 1, 1, 'REF')
    openings = Openings(plan)
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights={})
    result, meta = solver.run(appliance_sets=[('SINK', 'COOK', 'REF')])
    assert result is not None
    feats = meta.get('features', {})
    assert feats.get('work_triangle_bonus', 1.0) == 0.0


def test_solver_fills_missing_appliances():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights={})
    result, meta = solver.run(appliance_sets=[('SINK', 'COOK', 'REF')])
    assert result is not None, 'solver failed to place appliances'
    for code in ('SINK', 'COOK', 'REF'):
        assert list(components_by_code(result, code)), f'{code} not placed'


def test_solver_score_uses_dot_product_only():
    plan = GridPlan(3.0, 3.0)
    plan.place(0, 0, 1, 1, 'SINK')
    plan.place(1, 0, 1, 1, 'COOK')
    plan.place(2, 0, 1, 1, 'REF')
    openings = Openings(plan)
    weights = {'adjacency': 0.5, 'work_triangle_bonus': 2.0}
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights=weights)
    result, meta = solver.run(appliance_sets=[('SINK', 'COOK', 'REF')])
    assert result is not None, 'solver failed to place appliances'
    feats = meta.get('features', {})
    expected = dot_score(weights, feats)
    assert math.isclose(meta.get('score'), expected)


def test_custom_book_clear_override():
    plan = GridPlan(2.0, 2.0)
    openings = Openings(plan)
    custom_book = {'CLEAR': {'front_min': 0.123}}
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights={}, book=custom_book)
    assert solver.c.get('front_min') == 0.123
