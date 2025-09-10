import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from solver import overlaps


class Dummy:
    def __init__(self, x, y, w, h):
        self.x_offset = x
        self.y_offset = y
        self.gw = w
        self.gh = h


def test_overlaps_true_and_false():
    a = Dummy(0, 0, 2, 2)
    b = Dummy(1, 1, 2, 2)
    c = Dummy(2, 2, 2, 2)
    assert overlaps(a, b)
    assert not overlaps(a, c)
