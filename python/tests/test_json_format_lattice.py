"""Tests for the JSON format example files for libcasm.xtal.Lattice.

These tests validate that:
1. Each JSON example file is valid and can be read with Lattice.from_dict().
2. The constructed Lattice has the expected geometric properties.
3. Round-trip JSON serialization is consistent (to_dict / from_dict).
"""

import json
import math
import pathlib

import numpy as np
import pytest

import libcasm.xtal as xtal

JSON_DIR = (
    pathlib.Path(__file__).parent.parent / "doc" / "reference" / "json_format" / "json"
)


def load_lattice(filename):
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    return xtal.Lattice.from_dict(data)


# ---------------------------------------------------------------------------
# Helper checks
# ---------------------------------------------------------------------------


def column_norms(lattice):
    """Return the lengths of the three lattice vectors."""
    L = lattice.column_vector_matrix()
    return [np.linalg.norm(L[:, i]) for i in range(3)]


def angle_between_columns(lattice, i, j):
    """Return the angle (degrees) between lattice vectors i and j."""
    L = lattice.column_vector_matrix()
    vi, vj = L[:, i], L[:, j]
    cos_theta = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
    return math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Round-trip test (applies to all examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "xtal_lattice_simple_cubic.json",
        "xtal_lattice_fcc.json",
        "xtal_lattice_bcc.json",
        "xtal_lattice_hcp.json",
    ],
)
def test_roundtrip(filename):
    """Lattice.from_dict followed by to_dict reproduces the lattice vectors."""
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    lattice = xtal.Lattice.from_dict(data)
    data2 = lattice.to_dict()
    lv_in = np.array(data["lattice_vectors"])
    lv_out = np.array(data2["lattice_vectors"])
    np.testing.assert_allclose(lv_out, lv_in, atol=1e-10)


# ---------------------------------------------------------------------------
# Per-example geometric checks
# ---------------------------------------------------------------------------


def test_simple_cubic():
    """Simple cubic lattice: a = 1.0 Å, all angles 90°, all vectors equal.

    The lattice point group is Oh with 48 operations.
    """
    lattice = load_lattice("xtal_lattice_simple_cubic.json")
    norms = column_norms(lattice)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-10)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        np.testing.assert_allclose(
            angle_between_columns(lattice, i, j), 90.0, atol=1e-8
        )
    assert len(xtal.make_point_group(lattice)) == 48


def test_fcc():
    """FCC Al primitive cell: a = 4.046 Å, all vectors length a/sqrt(2), all angles 60°.

    The lattice point group is Oh with 48 operations.
    """
    lattice = load_lattice("xtal_lattice_fcc.json")
    a = 4.046
    expected_norm = a / math.sqrt(2)
    norms = column_norms(lattice)
    np.testing.assert_allclose(norms, [expected_norm] * 3, atol=1e-3)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        np.testing.assert_allclose(
            angle_between_columns(lattice, i, j), 60.0, atol=1e-3
        )
    assert len(xtal.make_point_group(lattice)) == 48


def test_bcc():
    """BCC Fe primitive cell: a = 2.87 Å, all vectors length a*sqrt(3)/2, all angles
    ~109.47°.

    The lattice point group is Oh with 48 operations.
    """
    lattice = load_lattice("xtal_lattice_bcc.json")
    a = 2.87
    expected_norm = a * math.sqrt(3) / 2
    norms = column_norms(lattice)
    np.testing.assert_allclose(norms, [expected_norm] * 3, atol=1e-3)
    expected_angle = math.degrees(math.acos(-1.0 / 3.0))  # ~109.47°
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        np.testing.assert_allclose(
            angle_between_columns(lattice, i, j), expected_angle, atol=1e-3
        )
    assert len(xtal.make_point_group(lattice)) == 48


def test_hcp():
    """HCP Mg: a = 3.2094 Å, c = 5.2105 Å, in-plane angle 120°, c perpendicular.

    The lattice point group is D6h with 24 operations.
    """
    lattice = load_lattice("xtal_lattice_hcp.json")
    a = 3.20940
    c = 5.21050
    norms = column_norms(lattice)
    np.testing.assert_allclose(norms[0], a, atol=1e-5)
    np.testing.assert_allclose(norms[1], a, atol=1e-5)
    np.testing.assert_allclose(norms[2], c, atol=1e-5)
    # In-plane vectors form 120° angle
    np.testing.assert_allclose(angle_between_columns(lattice, 0, 1), 120.0, atol=1e-8)
    # c-axis is perpendicular to both in-plane vectors
    np.testing.assert_allclose(angle_between_columns(lattice, 0, 2), 90.0, atol=1e-8)
    np.testing.assert_allclose(angle_between_columns(lattice, 1, 2), 90.0, atol=1e-8)
    assert len(xtal.make_point_group(lattice)) == 24
