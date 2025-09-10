import pytest

from gds import GenerateView, GridPlan, Openings, WALL_LEFT


class LayeredCanvas:
    """Minimal canvas emulating stacking order for tests."""

    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.items = []
        self._id = 0

    # Geometry helpers -------------------------------------------------
    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, tag):
        if tag in ("all", None):
            self.items.clear()
        else:
            self.items = [i for i in self.items if tag not in i.get("tags", ())]

    # Item creation -----------------------------------------------------
    def _create_item(self, tags):
        self._id += 1
        item = {"id": self._id, "tags": tags}
        self.items.append(item)
        return self._id

    def create_line(self, *args, **kwargs):
        return self._create_item(kwargs.get("tags", ()))

    def create_rectangle(self, *args, **kwargs):
        return self._create_item(kwargs.get("tags", ()))

    def create_oval(self, *args, **kwargs):
        return self._create_item(kwargs.get("tags", ()))

    def create_text(self, *args, **kwargs):
        return self._create_item(kwargs.get("tags", ()))

    def tag_bind(self, *args, **kwargs):
        pass

    # Stacking operations ----------------------------------------------
    def _resolve(self, spec):
        if isinstance(spec, int):
            for it in self.items:
                if it["id"] == spec:
                    return [it]
            return []
        return [it for it in self.items if spec in it.get("tags", ())]

    def tag_lower(self, spec, below=None):
        targets = self._resolve(spec)
        for t in targets:
            self.items.remove(t)
        if below is not None:
            ref = self._resolve(below)
            idx = self.items.index(ref[0]) if ref else 0
            for t in reversed(targets):
                self.items.insert(idx, t)
        else:
            self.items = targets + self.items

    def tag_raise(self, spec, above=None):
        targets = self._resolve(spec)
        for t in targets:
            self.items.remove(t)
        if above is not None:
            ref = self._resolve(above)
            idx = self.items.index(ref[-1]) + 1 if ref else len(self.items)
            for t in targets:
                self.items.insert(idx, t)
                idx += 1
        else:
            self.items.extend(targets)

    # Queries -----------------------------------------------------------
    def find_all(self):
        return tuple(i["id"] for i in self.items)

    def find_withtag(self, tag):
        return tuple(i["id"] for i in self.items if tag in i.get("tags", ()))

    def bbox(self, tag):
        return 0, 0, 0, 0


def test_clearance_below_furniture_and_walls():
    plan = GridPlan(2.0, 2.0)
    plan.place(0, 0, 1, 1, "BED")
    plan.clearzones.append((0, 0, 1, 1, "CLR", "BED"))
    openings = Openings(plan)
    openings.door_wall = WALL_LEFT

    gv = GenerateView.__new__(GenerateView)
    gv.canvas = LayeredCanvas()
    gv.opening_item_info = {}
    gv._draw_all_layers(plan, openings, 0, 0, 20, 2, 1, True, "bed")

    order = gv.canvas.find_all()
    clear_id = gv.canvas.find_withtag("clear")[0]
    furn_id = gv.canvas.find_withtag("furn")[0]
    room_id = gv.canvas.find_withtag("room")[0]
    opening_id = gv.canvas.find_withtag("opening")[0]

    assert order.index(clear_id) < order.index(furn_id) < order.index(room_id)
    assert order.index(furn_id) < order.index(opening_id)


def test_window_clearance_drawn():
    plan = GridPlan(3.0, 3.0)
    openings = Openings(plan)
    openings.door_wall = -1  # suppress door clearance
    openings.windows = [(0, 0.5, 1.0)]

    gv = GenerateView.__new__(GenerateView)
    gv.canvas = LayeredCanvas()
    gv.opening_item_info = {}
    gv._draw_all_layers(plan, openings, 0, 0, 20, 2, 1, False, "bed")

    clear_items = gv.canvas.find_withtag("clear")
    assert len(clear_items) == 1


def test_door_clearance_mirrored():
    plan = GridPlan(2.0, 2.0)
    openings = Openings(plan)
    openings.windows = []

    gv = GenerateView.__new__(GenerateView)
    gv.canvas = LayeredCanvas()
    gv.opening_item_info = {}
    gv._draw_all_layers(plan, openings, 0, 0, 20, 2, 1, True, "bed")

    clear_items = gv.canvas.find_withtag("clear")
    assert len(clear_items) == 2

