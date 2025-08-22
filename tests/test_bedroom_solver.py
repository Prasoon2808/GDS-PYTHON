import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GridPlan, Openings, BedroomSolver


def test_solver_signals_no_bed():
    plan = GridPlan(1.5, 1.5)
    openings = Openings(plan)
    solver = BedroomSolver(plan, openings, bed_key=None, rng=random.Random(0), weights={})
    result, meta = solver.run(iters=10, time_budget_ms=100, max_attempts=2)
    assert result is None
    assert meta.get('status') == 'no_bed'
