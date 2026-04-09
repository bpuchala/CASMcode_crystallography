"""Tests for the JSON format example files for libcasm.xtal.Occupant.

These tests validate that:
1. Each JSON example file is valid and can be read with Occupant.from_dict().
2. The constructed Occupant has the expected properties.
3. Round-trip JSON serialization is consistent (to_dict / from_dict).
"""

import json
import pathlib

import numpy as np
import pytest

import libcasm.xtal as xtal

JSON_DIR = (
    pathlib.Path(__file__).parent.parent / "doc" / "reference" / "json_format" / "json"
)


def load_occupant(filename):
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    return xtal.Occupant.from_dict(data)


# ---------------------------------------------------------------------------
# Round-trip test (applies to all examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "xtal_occupant_atom.json",
        "xtal_occupant_magspin.json",
        "xtal_occupant_dimer_x.json",
    ],
)
def test_roundtrip(filename):
    """Occupant.from_dict followed by to_dict preserves the occupant name."""
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    occupant = xtal.Occupant.from_dict(data)
    data2 = occupant.to_dict()
    assert data2["name"] == data["name"]


# ---------------------------------------------------------------------------
# Per-example checks
# ---------------------------------------------------------------------------


def test_atom():
    """Simple aluminum atom: name 'Al', one atom component."""
    occupant = load_occupant("xtal_occupant_atom.json")
    assert occupant.name() == "Al"
    atoms = occupant.atoms()
    assert len(atoms) == 1
    assert atoms[0].name() == "Al"


def test_magspin():
    """Collinear magnetic spin occupant: name 'A', atom at origin."""
    occupant = load_occupant("xtal_occupant_magspin.json")
    assert occupant.name() == "A"
    atoms = occupant.atoms()
    assert len(atoms) == 1
    assert atoms[0].name() == "A"
    np.testing.assert_allclose(atoms[0].coordinate(), [0.0, 0.0, 0.0], atol=1e-10)


def test_dimer():
    """Homonuclear dimer A2: name 'A2', two atoms 'A' at ±0.1 along the given axis."""
    occupant = load_occupant("xtal_occupant_dimer_x.json")
    axis = 0
    assert occupant.name() == "A2"
    atoms = occupant.atoms()
    assert len(atoms) == 2
    for atom in atoms:
        assert atom.name() == "A"

    coords = np.array([atom.coordinate() for atom in atoms])
    # Both off-axis components should be zero
    off_axes = [i for i in range(3) if i != axis]
    np.testing.assert_allclose(coords[:, off_axes], 0.0, atol=1e-10)
    # On-axis components should be ±0.1, center of mass at origin
    np.testing.assert_allclose(np.sum(coords[:, axis]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.abs(coords[:, axis]), 0.1, atol=1e-10)
