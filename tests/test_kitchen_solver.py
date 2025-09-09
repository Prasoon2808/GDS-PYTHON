import random
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GridPlan, Openings, KitchenSolver, default_kitchen_sets


def test_default_sets_include_work_triangle():
    sets = default_kitchen_sets()
    assert sets[0] == ('SINK',)
    assert sets[-1] == ('SINK', 'COOK', 'REF')


def test_solver_reports_triangle_bonus():
    plan = GridPlan(3.0, 3.0)
    plan.place(0, 0, 1, 1, 'SINK')
    plan.place(2, 0, 1, 1, 'COOK')
    plan.place(0, 2, 1, 1, 'REF')
    openings = Openings(plan)
    solver = KitchenSolver(plan, openings, rng=random.Random(0), weights={})
    result, meta = solver.run(appliance_sets=[('SINK', 'COOK', 'REF')])
    assert result is not None
    feats = meta.get('features', {})
    assert feats.get('work_triangle_bonus', 0.0) == 1.0
