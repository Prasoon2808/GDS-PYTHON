import os
import sys
import pytest

# Ensure the repository root is importable when tests are executed from the
# ``tests`` directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vastu_all_in_one import GenerateView, Openings, GridPlan, WALL_RIGHT, WALL_LEFT


class DummyStatus:
    def __init__(self):
        self.msg = ''

    def set(self, msg: str):
        self.msg = msg


def make_generate_view(bath_dims=(2.0, 2.0)):
    gv = GenerateView.__new__(GenerateView)
    gv.bath_dims = bath_dims
    gv.bed_openings = Openings(GridPlan(4.0, 4.0))
    gv.bath_openings = Openings(GridPlan(*bath_dims)) if bath_dims else None
    gv.status = DummyStatus()
    return gv


def test_shared_wall_door_alignment_passes():
    gv = make_generate_view((2.0, 2.0))
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bath_openings.door_wall = WALL_LEFT
    gv.bed_openings.door_center = gv.bath_openings.door_center = 1.0
    gv.bed_openings.door_width = gv.bath_openings.door_width = 0.9

    assert gv._validate_shared_wall_door() is True
    assert gv.status.msg == ''


@pytest.mark.parametrize(
    'bath_dims,bath_center,expected_msg',
    [
        (
            (2.0, 2.0),
            1.25,
            'Door must align on shared wall between bedroom and bathroom.',
        ),
        (
            None,
            None,
            'Door on right wall requires adjacent bathroom.',
        ),
    ],
)
def test_misaligned_or_nonshared_door_sets_status(bath_dims, bath_center, expected_msg):
    gv = make_generate_view(bath_dims)
    gv.bed_openings.door_wall = WALL_RIGHT
    gv.bed_openings.door_center = 1.0
    gv.bed_openings.door_width = 0.9

    if bath_dims:
        gv.bath_openings.door_wall = WALL_LEFT
        gv.bath_openings.door_center = bath_center
        gv.bath_openings.door_width = 0.9

    result = gv._validate_shared_wall_door()
    assert result is False
    assert gv.status.msg == expected_msg


def test_furniture_controls_present_for_generator_label(monkeypatch):
    import types
    import tkinter as tk
    import vastu_all_in_one

    master = tk.Tcl()
    tk._default_root = master

    class FakeWidget:
        def __init__(self, master=None, **kwargs):
            self.children = []
            if master is not None and hasattr(master, 'children'):
                master.children.append(self)
        def pack(self, *args, **kwargs):
            return self
        def grid(self, *args, **kwargs):
            return self
        def destroy(self):
            pass
        def winfo_children(self):
            return self.children
        def grid_columnconfigure(self, *args, **kwargs):
            pass

    fake_ttk = types.SimpleNamespace(
        Frame=FakeWidget,
        Label=FakeWidget,
        Combobox=FakeWidget,
        Scale=FakeWidget,
        Button=FakeWidget,
        Checkbutton=FakeWidget,
    )
    monkeypatch.setattr(vastu_all_in_one, 'ttk', fake_ttk)

    gv = GenerateView.__new__(GenerateView)
    gv.room_label = 'Bedroom (Generator)'
    gv.sidebar = FakeWidget()
    gv.bed_Hm = 4.0
    gv.bed_Wm = 4.0
    gv.bath_dims = None

    for name in [
        '_solve_and_draw', '_apply_batch_and_generate', '_add_furniture',
        '_remove_furniture', '_simulate_one', '_simulate_two',
        'simulate_circulation', '_export_png']:
        setattr(gv, name, lambda *a, **kw: None)

    GenerateView._build_sidebar(gv)
    assert hasattr(gv, 'furn_kind'), 'Furniture controls missing'
