"""Tests for Structure.neighborhood"""

import math

import numpy as np
import pytest

import libcasm.xtal as xtal


def make_simple_cubic(a=1.0):
    lat = xtal.Lattice(np.diag([a, a, a]))
    return xtal.Structure(
        lattice=lat,
        atom_coordinate_frac=np.zeros((3, 1)),
        atom_type=["A"],
    )


def make_fcc_primitive(a=4.0):
    lat = xtal.Lattice(
        np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]).T
    )
    return xtal.Structure(
        lattice=lat,
        atom_coordinate_frac=np.zeros((3, 1)),
        atom_type=["A"],
    )


def make_bcc_conventional(a=2.0):
    """BCC conventional cell: 2 atoms at (0,0,0) and (0.5,0.5,0.5)."""
    lat = xtal.Lattice(np.diag([a, a, a]))
    return xtal.Structure(
        lattice=lat,
        atom_coordinate_frac=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]).T,
        atom_type=["A", "A"],
    )


def make_site(b, i, j, k):
    return xtal.IntegralSiteCoordinate.from_list([b, i, j, k])


def as_tuples(sites):
    return {tuple(n.to_list()) for n in sites}


def site_cart(structure, n):
    L = structure.lattice().column_vector_matrix()
    return structure.atom_coordinate_cart()[:, n.sublattice()] + L @ np.array(
        n.unitcell(), dtype=float
    )


def distances(structure, sites):
    return sorted(np.linalg.norm(site_cart(structure, n)) for n in sites)


# --- phenomenal_sites=None ---


def test_none_includes_origin_sites():
    """phenomenal_sites=None always includes the origin unit cell sites."""
    s = make_simple_cubic(a=1.0)
    nbrs = s.neighborhood(1.5)
    assert make_site(0, 0, 0, 0) in nbrs


def test_none_simple_cubic_count():
    """phenomenal_sites=None: 6 NN + self for simple cubic a=1, cutoff=1.1.

    cutoff=1.1 selects only the 1st shell (dist=1.0); 2nd shell is at sqrt(2)~1.414.
    """
    s = make_simple_cubic(a=1.0)
    nbrs = s.neighborhood(1.1)
    assert len(nbrs) == 7


def test_none_simple_cubic_distances():
    """All returned sites are within cutoff; only two unique distances."""
    s = make_simple_cubic(a=1.0)
    nbrs = s.neighborhood(1.1)
    dists = distances(s, nbrs)
    assert all(d < 1.1 for d in dists)
    # Two unique distances: 0.0 (self) and 1.0 (NN)
    assert dists[0] == pytest.approx(0.0)
    assert all(pytest.approx(1.0) == d for d in dists[1:])


def test_none_fcc_primitive():
    """phenomenal_sites=None: 12 FCC NN + self at cutoff just below 2nd shell."""
    s = make_fcc_primitive(a=4.0)
    first_shell = 4.0 / math.sqrt(2)  # ~2.828
    nbrs = s.neighborhood(3.5)
    assert len(nbrs) == 13  # 12 NN + self
    dists = distances(s, nbrs)
    unique = sorted({round(d, 6) for d in dists})
    assert len(unique) == 2
    assert unique[0] == pytest.approx(0.0)
    assert unique[1] == pytest.approx(first_shell, rel=1e-5)


def test_none_bcc_both_origin_sites_included():
    """phenomenal_sites=None with 2-atom basis: both origin sites always included."""
    s = make_bcc_conventional(a=2.0)
    nbrs = s.neighborhood(2.0)
    assert make_site(0, 0, 0, 0) in nbrs
    assert make_site(1, 0, 0, 0) in nbrs


# --- phenomenal_sites as int ---


def test_int_exclude_phenomenal():
    """phenomenal_sites=int, include_phenomenal_sites=False: origin site excluded."""
    s = make_simple_cubic(a=1.0)
    nbrs = s.neighborhood(1.1, phenomenal_sites=0, include_phenomenal_sites=False)
    assert len(nbrs) == 6
    assert make_site(0, 0, 0, 0) not in nbrs
    dists = distances(s, nbrs)
    assert all(pytest.approx(1.0) == d for d in dists)


def test_int_include_phenomenal():
    """phenomenal_sites=int, include_phenomenal_sites=True: origin site included."""
    s = make_simple_cubic(a=1.0)
    nbrs = s.neighborhood(1.1, phenomenal_sites=0, include_phenomenal_sites=True)
    assert len(nbrs) == 7
    assert make_site(0, 0, 0, 0) in nbrs


# --- phenomenal_sites as list[int] ---


def test_list_int_same_as_single_int():
    """phenomenal_sites=[int] gives same result as phenomenal_sites=int."""
    s = make_simple_cubic(a=1.0)
    single = s.neighborhood(1.1, phenomenal_sites=0)
    listed = s.neighborhood(1.1, phenomenal_sites=[0])
    assert as_tuples(single) == as_tuples(listed)


def test_list_int_two_sites():
    """phenomenal_sites=[int, int] with 2-atom basis."""
    s = make_bcc_conventional(a=2.0)
    # Both origin sites as phenomenal, exclude them
    nbrs = s.neighborhood(2.0, phenomenal_sites=[0, 1], include_phenomenal_sites=False)
    assert make_site(0, 0, 0, 0) not in nbrs
    assert make_site(1, 0, 0, 0) not in nbrs


# --- phenomenal_sites as IntegralSiteCoordinate ---


def test_site_at_origin():
    """phenomenal_sites=IntegralSiteCoordinate at (0,0,0): same as int=0."""
    s = make_simple_cubic(a=1.0)
    nbrs_int = s.neighborhood(1.1, phenomenal_sites=0)
    nbrs_site = s.neighborhood(1.1, phenomenal_sites=make_site(0, 0, 0, 0))
    assert as_tuples(nbrs_int) == as_tuples(nbrs_site)


def test_site_off_origin_exclude():
    """phenomenal_sites off-origin: correct neighbors, phenomenal site excluded."""
    s = make_simple_cubic(a=1.0)
    # Phenomenal at unit cell (1,0,0) = Cartesian (1,0,0)
    phenom = make_site(0, 1, 0, 0)
    nbrs = s.neighborhood(1.1, phenomenal_sites=phenom, include_phenomenal_sites=False)
    expected = {
        (0, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 1, 1, 0),
        (0, 1, -1, 0),
        (0, 1, 0, 1),
        (0, 1, 0, -1),
    }
    assert as_tuples(nbrs) == expected
    assert phenom not in nbrs


def test_site_off_origin_include():
    """phenomenal_sites off-origin, include_phenomenal_sites=True."""
    s = make_simple_cubic(a=1.0)
    phenom = make_site(0, 1, 0, 0)
    nbrs = s.neighborhood(1.1, phenomenal_sites=phenom, include_phenomenal_sites=True)
    assert phenom in nbrs
    assert len(nbrs) == 7


# --- phenomenal_sites as list[IntegralSiteCoordinate] ---


def test_list_site_single_same_as_single_make_site():
    """phenomenal_sites=[site] same as phenomenal_sites=site."""
    s = make_simple_cubic(a=1.0)
    phenom = make_site(0, 1, 0, 0)
    single = s.neighborhood(1.1, phenomenal_sites=phenom)
    listed = s.neighborhood(1.1, phenomenal_sites=[phenom])
    assert as_tuples(single) == as_tuples(listed)


def test_list_site_two_adjacent_sites():
    """phenomenal_sites=[site, site]: union of neighborhoods, phenomenal sites excluded.

    Simple cubic a=1, cutoff=1.1 selects only 1st shell (dist=1.0).
    Phenomenal: (0,0,0) and (1,0,0).
    NN of (0,0,0) minus (1,0,0): (-1,0,0),(0,±1,0),(0,0,±1) = 5
    NN of (1,0,0) minus (0,0,0): (2,0,0),(1,±1,0),(1,0,±1) = 5
    Total: 10 sites.
    """
    s = make_simple_cubic(a=1.0)
    phenoms = [make_site(0, 0, 0, 0), make_site(0, 1, 0, 0)]
    nbrs = s.neighborhood(1.1, phenomenal_sites=phenoms, include_phenomenal_sites=False)
    expected = {
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 1, 0),
        (0, 0, 0, -1),
        (0, 0, 0, 1),
        (0, 2, 0, 0),
        (0, 1, 1, 0),
        (0, 1, -1, 0),
        (0, 1, 0, 1),
        (0, 1, 0, -1),
    }
    assert as_tuples(nbrs) == expected
    assert make_site(0, 0, 0, 0) not in nbrs
    assert make_site(0, 1, 0, 0) not in nbrs


def test_list_site_two_adjacent_include():
    """phenomenal_sites=[site, site], include_phenomenal_sites=True."""
    s = make_simple_cubic(a=1.0)
    phenoms = [make_site(0, 0, 0, 0), make_site(0, 1, 0, 0)]
    nbrs = s.neighborhood(1.1, phenomenal_sites=phenoms, include_phenomenal_sites=True)
    assert make_site(0, 0, 0, 0) in nbrs
    assert make_site(0, 1, 0, 0) in nbrs
    assert len(nbrs) == 12  # 10 non-phenomenal + 2 phenomenal
