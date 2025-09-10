"""Compatibility wrapper for the main application module.

This project stores the primary implementation in a file whose name contains
spaces.  Python modules cannot normally be imported from such filenames, so we
load that file dynamically and re-export its public attributes under the more
conventional ``vastu_all_in_one`` module name used by the tests.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


_path = pathlib.Path(__file__).with_name("GDS last stable version.py")
_spec = importlib.util.spec_from_file_location(__name__, _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def overlaps(a, b) -> bool:
    """Return ``True`` if axis-aligned plans ``a`` and ``b`` overlap."""
    ax0, ay0 = a.x_offset, a.y_offset
    ax1, ay1 = ax0 + a.gw, ay0 + a.gh
    bx0, by0 = b.x_offset, b.y_offset
    bx1, by1 = bx0 + b.gw, by0 + b.gh
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def shares_edge(a, b) -> bool:
    """Return ``True`` when plans ``a`` and ``b`` share a boundary edge."""
    ax0, ay0 = a.x_offset, a.y_offset
    ax1, ay1 = ax0 + a.gw, ay0 + a.gh
    bx0, by0 = b.x_offset, b.y_offset
    bx1, by1 = bx0 + b.gw, by0 + b.gh
    return (
        (ax1 == bx0 and max(ay0, by0) < min(ay1, by1))
        or (bx1 == ax0 and max(ay0, by0) < min(ay1, by1))
        or (ay1 == by0 and max(ax0, bx0) < min(ax1, bx1))
        or (by1 == ay0 and max(ax0, bx0) < min(ax1, bx1))
    )


setattr(_module, "overlaps", overlaps)
setattr(_module, "shares_edge", shares_edge)

# Replace the current module with the loaded implementation so ``import
# vastu_all_in_one`` exposes the full API of the underlying file.
sys.modules[__name__] = _module

