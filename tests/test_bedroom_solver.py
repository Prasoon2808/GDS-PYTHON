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


def test_solver_custom_set_fallback():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    sets = [
        ('BED', 'BST', 'DRS'),  # requires dresser
        ('BED', 'BST'),         # fallback without dresser
        ('BED',),               # final fallback to just bed
    ]
    result, _ = solver.run(iters=80, time_budget_ms=200, max_attempts=2, furniture_sets=sets)
    assert result is not None
    counts = {code: len(list(components_by_code(result, code))) for code in ['BED', 'BST', 'DRS']}
    assert counts['BED'] == 1


def test_solver_respects_min_adjacency_threshold():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    result, meta = solver.run(iters=80, time_budget_ms=200, max_attempts=2, min_adjacency=10.0)
    assert result is None
    assert meta.get('status') == 'adjacency_below_threshold'
    assert meta.get('features', {}).get('adjacency', 0.0) < 10.0


def test_mark_clear_removes_entire_bed():
    plan = GridPlan(3.0, 3.0)
    plan.place(0, 0, 2, 1, 'BED:1')
    # Overlap clearance with the second cell of the bed
    plan.mark_clear(1, 0, 0.5, 1, 'SIDE', 'TEST')
    assert not components_by_code(plan, 'BED')


class ClearingBedroomSolver(BedroomSolver):
    def _add_window_clearances(self, p):
        super()._add_window_clearances(p)
        # Remove a single cell from the placed bed to simulate partial overlap
        p.clear(0, 0, 1, 1)


def test_solver_rejects_truncated_bed():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    seed = {'wall': 0, 'beds': [(0, 0, 2, 1)]}

    solver = ClearingBedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    result, _, _ = solver._try_seed(seed)
    assert result is None
