"""Tests for the JSON format example files for libcasm.xtal.Prim.

These tests validate that:
1. Each JSON example file is valid and can be read with Prim.from_dict().
2. The constructed Prim has the expected structural properties.
3. Round-trip JSON serialization is consistent (to_dict / from_dict).
"""

import json
import pathlib

import numpy as np
import pytest

import libcasm.xtal as xtal

PRIM_JSON_DIR = (
    pathlib.Path(__file__).parent.parent / "doc" / "examples" / "prim" / "json"
)


def load_prim(filename):
    with open(PRIM_JSON_DIR / filename) as f:
        data = json.load(f)
    return xtal.Prim.from_dict(data)


def local_dof_names(prim):
    """Return the set of DoF type names present across all sites."""
    names = set()
    for site_dofs in prim.local_dof():
        for dof in site_dofs:
            names.add(dof.dofname())
    return names


def global_dof_names(prim):
    """Return the set of global DoF type names."""
    return {dof.dofname() for dof in prim.global_dof()}


# ---------------------------------------------------------------------------
# Round-trip test (applies to all examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "simple_cubic_binary.json",
        "ZrO_prim.json",
        "simple_cubic_disp.json",
        "simple_cubic_Hstrain.json",
        "simple_cubic_ising.json",
    ],
)
def test_roundtrip(filename):
    """Prim.from_dict followed by to_dict reproduces the lattice vectors."""
    with open(PRIM_JSON_DIR / filename) as f:
        data = json.load(f)
    prim = xtal.Prim.from_dict(data)
    data2 = prim.to_dict()
    lv_in = np.array(data["lattice_vectors"])
    lv_out = np.array(data2["lattice_vectors"])
    np.testing.assert_allclose(lv_out, lv_in, atol=1e-10)


# ---------------------------------------------------------------------------
# Per-example structural checks
# ---------------------------------------------------------------------------


def test_simple_cubic_binary():
    """Simple cubic binary: 1 site, occupants A and B, Oh factor group (48 ops)."""
    prim = load_prim("simple_cubic_binary.json")
    occ_dof = prim.occ_dof()
    assert len(occ_dof) == 1
    assert set(occ_dof[0]) == {"A", "B"}
    assert global_dof_names(prim) == set()
    assert len(xtal.make_factor_group(prim)) == 48


def test_ZrO():
    """HCP ZrO: 4 sites, Zr on sites 0-1, Va/O on sites 2-3, D6h factor group
    (24 ops)."""
    prim = load_prim("ZrO_prim.json")
    occ_dof = prim.occ_dof()
    assert len(occ_dof) == 4
    assert occ_dof[0] == ["Zr"]
    assert occ_dof[1] == ["Zr"]
    assert set(occ_dof[2]) == {"Va", "O"}
    assert set(occ_dof[3]) == {"Va", "O"}
    assert global_dof_names(prim) == set()
    assert len(xtal.make_factor_group(prim)) == 24


def test_simple_cubic_disp():
    """Simple cubic with displacement DoF: 1 site, disp local DoF present."""
    prim = load_prim("simple_cubic_disp.json")
    assert len(prim.occ_dof()) == 1
    assert "disp" in local_dof_names(prim)
    assert global_dof_names(prim) == set()


def test_simple_cubic_Hstrain():
    """Simple cubic with Hencky strain global DoF."""
    prim = load_prim("simple_cubic_Hstrain.json")
    assert "Hstrain" in global_dof_names(prim)


def test_simple_cubic_ising():
    """Simple cubic Ising model: 1 site, occupants A.up and A.down."""
    prim = load_prim("simple_cubic_ising.json")
    occ_dof = prim.occ_dof()
    assert len(occ_dof) == 1
    assert set(occ_dof[0]) == {"A.up", "A.down"}
