"""Tests for the JSON format example files for libcasm.xtal.AtomComponent.

These tests validate that:
1. Each JSON example file is valid and can be read with AtomComponent.from_dict().
2. The constructed AtomComponent has the expected properties.
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


def load_atomcomponent(filename):
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    return xtal.AtomComponent.from_dict(data)


# ---------------------------------------------------------------------------
# Round-trip test (applies to all examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "xtal_atomcomponent_atom.json",
        "xtal_atomcomponent_magspin.json",
    ],
)
def test_roundtrip(filename):
    """AtomComponent.from_dict followed by to_dict preserves name and coordinate."""
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    atom = xtal.AtomComponent.from_dict(data)
    data2 = atom.to_dict()
    assert data2["name"] == data["name"]
    np.testing.assert_allclose(data2["coordinate"], data["coordinate"], atol=1e-10)


# ---------------------------------------------------------------------------
# Per-example checks
# ---------------------------------------------------------------------------


def test_atom():
    """Simple aluminum atom: name 'Al', coordinate at origin."""
    atom = load_atomcomponent("xtal_atomcomponent_atom.json")
    assert atom.name() == "Al"
    np.testing.assert_allclose(atom.coordinate(), [0.0, 0.0, 0.0], atol=1e-10)


def test_magspin():
    """Atom with Cmagspin: name 'A', coordinate at origin, Cmagspin value 1.0."""
    atom = load_atomcomponent("xtal_atomcomponent_magspin.json")
    assert atom.name() == "A"
    np.testing.assert_allclose(atom.coordinate(), [0.0, 0.0, 0.0], atol=1e-10)
    properties = atom.properties()
    assert "Cmagspin" in properties
    np.testing.assert_allclose(properties["Cmagspin"].ravel(), [1.0], atol=1e-10)
