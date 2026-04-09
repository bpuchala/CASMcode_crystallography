"""Tests for the JSON format example files for libcasm.xtal.DoFSetBasis.

These tests validate that:
1. Each JSON example file is valid and can be read with DoFSetBasis.from_dict().
2. The constructed DoFSetBasis has the expected properties.
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


def load_dofsetbasis(filename):
    """Load the first DoFSetBasis from a JSON file.

    DoFSetBasis.from_dict returns list[DoFSetBasis], one per key in the dict.
    """
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    result = xtal.DoFSetBasis.from_dict(data)
    return result[0]


# ---------------------------------------------------------------------------
# Round-trip test (applies to all examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "xtal_dofsetbasis_disp.json",
        "xtal_dofsetbasis_disp_xy.json",
        "xtal_dofsetbasis_Hstrain.json",
        "xtal_dofsetbasis_Hstrain_diagonal.json",
    ],
)
def test_roundtrip(filename):
    """DoFSetBasis.from_dict followed by to_dict preserves basis and axis_names."""
    with open(JSON_DIR / filename) as f:
        data = json.load(f)
    dof_list = xtal.DoFSetBasis.from_dict(data)
    assert len(dof_list) == 1
    dof = dof_list[0]
    dofname = dof.dofname()
    data2 = {}
    dof.to_dict(data2)
    np.testing.assert_allclose(
        np.array(data2[dofname]["basis"]),
        np.array(data[dofname]["basis"]),
        atol=1e-10,
    )
    assert data2[dofname]["axis_names"] == data[dofname]["axis_names"]


# ---------------------------------------------------------------------------
# Per-example checks
# ---------------------------------------------------------------------------


def test_disp():
    """Displacement DoF: dofname 'disp', 3 axes (dx, dy, dz), identity basis."""
    dof = load_dofsetbasis("xtal_dofsetbasis_disp.json")
    assert dof.dofname() == "disp"
    assert dof.axis_names() == ["dx", "dy", "dz"]
    assert dof.basis().shape == (3, 3)
    np.testing.assert_allclose(dof.basis(), np.eye(3), atol=1e-10)


def test_disp_xy():
    """Displacement DoF restricted to xy-plane: 2 axes (dx, dy), basis shape (3, 2)."""
    dof = load_dofsetbasis("xtal_dofsetbasis_disp_xy.json")
    assert dof.dofname() == "disp"
    assert dof.axis_names() == ["dx", "dy"]
    assert dof.basis().shape == (3, 2)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
    np.testing.assert_allclose(dof.basis(), expected, atol=1e-10)


def test_Hstrain():
    """Hencky strain DoF: dofname 'Hstrain', 6 axes (E_1..E_6), identity basis."""
    dof = load_dofsetbasis("xtal_dofsetbasis_Hstrain.json")
    assert dof.dofname() == "Hstrain"
    assert dof.axis_names() == ["E_1", "E_2", "E_3", "E_4", "E_5", "E_6"]
    assert dof.basis().shape == (6, 6)
    np.testing.assert_allclose(dof.basis(), np.eye(6), atol=1e-10)


def test_Hstrain_diagonal():
    """Hencky strain DoF restricted to diagonal — E_1, E_2, E_3: basis shape (6, 3)."""
    dof = load_dofsetbasis("xtal_dofsetbasis_Hstrain_diagonal.json")
    assert dof.dofname() == "Hstrain"
    assert dof.axis_names() == ["E_1", "E_2", "E_3"]
    assert dof.basis().shape == (6, 3)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    ).T
    np.testing.assert_allclose(dof.basis(), expected, atol=1e-10)
