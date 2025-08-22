import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GridPlan, Openings, BedroomSolver, components_by_code


def test_solver_signals_no_bed():
    plan = GridPlan(1.5, 1.5)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    result, meta = solver.run(iters=10, time_budget_ms=100, max_attempts=2)
    assert result is None
    assert meta.get('status') == 'no_bed'


def test_solver_falls_back_when_dresser_missing():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    result, _ = solver.run(iters=80, time_budget_ms=200, max_attempts=2)
    assert result is not None
    counts = {code: len(list(components_by_code(result, code))) for code in ['BED', 'BST', 'DRS', 'WRD']}
    assert counts['BED'] == 1
    assert counts['DRS'] == 0  # dresser omitted due to space


def test_solver_custom_set_fallback():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    sets = [
        ('BED', 'BST', 'DRS'),  # requires dresser
        ('BED', 'BST'),         # fallback without dresser
    ]
    result, _ = solver.run(iters=80, time_budget_ms=200, max_attempts=2, furniture_sets=sets)
    assert result is not None
    counts = {code: len(list(components_by_code(result, code))) for code in ['BED', 'BST', 'DRS']}
    assert counts['BED'] == 1
    assert counts['BST'] >= 1
    assert counts['DRS'] == 0
