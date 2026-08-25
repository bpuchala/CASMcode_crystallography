import math

import numpy as np
import pytest

import libcasm.xtal as xtal
import libcasm.xtal.lattices as xtal_lattices


def test_miller_bravais_direction_roundtrip():
    uvw_list = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, -1.0, 3.0],
        [-1.0, -1.0, -1.0],
    ]
    for uvw in uvw_list:
        uvtw = xtal.miller_to_miller_bravais_direction(uvw)
        assert uvtw.shape == (4,)
        # U + V + T = 0
        assert math.isclose(uvtw[0] + uvtw[1] + uvtw[2], 0.0, abs_tol=1e-10)
        assert np.allclose(xtal.miller_bravais_to_miller_direction(uvtw), uvw)


def test_miller_bravais_plane_roundtrip():
    hkl_list = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, -1.0, 3.0],
        [-1.0, -1.0, -1.0],
    ]
    for hkl in hkl_list:
        hkil = xtal.miller_to_miller_bravais_plane(hkl)
        assert hkil.shape == (4,)
        # h + k + i = 0
        assert math.isclose(hkil[0] + hkil[1] + hkil[2], 0.0, abs_tol=1e-10)
        assert np.allclose(xtal.miller_bravais_to_miller_plane(hkil), hkl)


def test_miller_bravais_direction_reference_values():
    # Standard hcp direction indices:
    # [uvw] -> [UVTW] (after scaling to integers)
    reference = [
        ([1, 1, 0], [1, 1, -2, 0]),  # <11-20>, close-packed direction
        ([1, 0, 0], [2, -1, -1, 0]),  # <2-1-10>
        ([0, 1, 0], [-1, 2, -1, 0]),  # <-12-10>
        ([1, -1, 0], [1, -1, 0, 0]),  # <1-100>, prismatic direction
        ([0, 0, 1], [0, 0, 0, 1]),  # [0001], c-axis
        ([2, 1, 0], [1, 0, -1, 0]),  # <10-10>, prismatic direction
    ]
    for uvw, uvtw_expected in reference:
        uvtw = xtal.scale_to_int(xtal.miller_to_miller_bravais_direction(uvw))
        assert np.array_equal(uvtw, np.array(uvtw_expected))
        # and the reverse, up to a scale factor
        uvw_result = xtal.scale_to_int(
            xtal.miller_bravais_to_miller_direction(uvtw_expected)
        )
        assert np.array_equal(uvw_result, xtal.scale_to_int(uvw))


def test_miller_bravais_plane_reference_values():
    # Standard hcp plane indices: (hkl) -> (hkil)
    reference = [
        ([0, 0, 1], [0, 0, 0, 1]),  # basal, (0001)
        ([1, 0, 0], [1, 0, -1, 0]),  # prismatic, {10-10}
        ([1, 1, 0], [1, 1, -2, 0]),  # second-order prismatic, {11-20}
        ([1, 0, 1], [1, 0, -1, 1]),  # first-order pyramidal, {10-11}
        ([1, 1, 2], [1, 1, -2, 2]),  # second-order pyramidal, {11-22}
        ([1, 0, 2], [1, 0, -1, 2]),  # {10-12}, tension twin plane
    ]
    for hkl, hkil_expected in reference:
        hkil = xtal.miller_to_miller_bravais_plane(hkl)
        assert np.array_equal(hkil, np.array(hkil_expected, dtype=float))
        assert np.array_equal(
            xtal.miller_bravais_to_miller_plane(hkil_expected),
            np.array(hkl, dtype=float),
        )


def test_miller_bravais_direction_and_plane_differ():
    # The classic bug: the plane and direction conventions are not the same
    hkl = uvw = [1.0, 1.0, 0.0]
    hkil = xtal.miller_to_miller_bravais_plane(hkl)
    uvtw = xtal.miller_to_miller_bravais_direction(uvw)
    assert np.array_equal(hkil, np.array([1.0, 1.0, -2.0, 0.0]))
    assert np.allclose(uvtw, np.array([1.0, 1.0, -2.0, 0.0]) / 3.0)
    assert not np.allclose(hkil, uvtw)


def test_miller_bravais_2d_columns():
    uvw_columns = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).transpose()
    assert uvw_columns.shape == (3, 3)

    uvtw_columns = xtal.miller_to_miller_bravais_direction(uvw_columns)
    assert uvtw_columns.shape == (4, 3)
    hkil_columns = xtal.miller_to_miller_bravais_plane(uvw_columns)
    assert hkil_columns.shape == (4, 3)

    # U + V + T = 0, h + k + i = 0, for every column
    assert np.allclose(uvtw_columns[0:3, :].sum(axis=0), np.zeros((3,)))
    assert np.allclose(hkil_columns[0:3, :].sum(axis=0), np.zeros((3,)))

    # 1d and 2d give consistent results
    for i in range(uvw_columns.shape[1]):
        assert np.allclose(
            uvtw_columns[:, i],
            xtal.miller_to_miller_bravais_direction(uvw_columns[:, i]),
        )
        assert np.allclose(
            hkil_columns[:, i],
            xtal.miller_to_miller_bravais_plane(uvw_columns[:, i]),
        )

    # round trip, as columns
    assert np.allclose(
        xtal.miller_bravais_to_miller_direction(uvtw_columns), uvw_columns
    )
    assert np.allclose(xtal.miller_bravais_to_miller_plane(hkil_columns), uvw_columns)


def test_miller_bravais_input_validation():
    with pytest.raises(ValueError, match="must have shape"):
        xtal.miller_to_miller_bravais_direction([1.0, 0.0])
    with pytest.raises(ValueError, match="must have shape"):
        xtal.miller_to_miller_bravais_plane([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must have shape"):
        xtal.miller_bravais_to_miller_direction([1.0, 0.0, 0.0])

    # U + V + T != 0
    with pytest.raises(ValueError, match=r"U \+ V \+ T = 0"):
        xtal.miller_bravais_to_miller_direction([1.0, 1.0, 1.0, 0.0])

    # h + k + i != 0
    with pytest.raises(ValueError, match=r"h \+ k \+ i = 0"):
        xtal.miller_bravais_to_miller_plane([1.0, 1.0, 1.0, 0.0])


def test_scale_to_int_rational():
    assert np.array_equal(xtal.scale_to_int([1.0, 1.0, 0.0]), np.array([1, 1, 0]))
    assert np.array_equal(xtal.scale_to_int([2.0, 2.0, 0.0]), np.array([1, 1, 0]))
    assert np.array_equal(xtal.scale_to_int([6.0, 4.0, 2.0]), np.array([3, 2, 1]))
    assert np.array_equal(
        xtal.scale_to_int([1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 0.0]),
        np.array([1, 1, -2, 0]),
    )
    # sign is preserved
    assert np.array_equal(xtal.scale_to_int([-1.0, -0.5, 0.0]), np.array([-2, -1, 0]))
    # result dtype is integer
    assert xtal.scale_to_int([1.0, 1.0, 0.0]).dtype == np.int64


def test_scale_to_int_failure():
    # irrational: cannot be scaled to integers with max_element=10
    irrational = [1.0, math.sqrt(2.0), 0.0]
    assert xtal.scale_to_int_if_possible(irrational) is None
    with pytest.raises(ValueError, match="could not scale to integers"):
        xtal.scale_to_int(irrational)

    # rational, but requires a scale factor larger than max_element
    v = [1.0, 1.0 / 3.0, 1.0 / 7.0]
    assert xtal.scale_to_int_if_possible(v, max_element=10) is None
    assert np.array_equal(xtal.scale_to_int(v, max_element=21), np.array([21, 7, 3]))

    # zero vector raises
    with pytest.raises(ValueError, match="zero vector"):
        xtal.scale_to_int([0.0, 0.0, 0.0])

    # max_element < 1 raises
    with pytest.raises(ValueError, match="max_element must be >= 1"):
        xtal.scale_to_int([1.0, 1.0, 0.0], max_element=0)

    # all elements of the result are <= max_element
    result = xtal.scale_to_int([1.0, 1.0 / 3.0, 0.0])
    assert np.max(np.abs(result)) <= 10


def test_scale_columns_to_int_if_possible():
    M = np.array(
        [
            [1.0, 1.0 / 3.0, 0.0],  # scalable -> [3, 1, 0]
            [1.0, math.sqrt(2.0), 0.0],  # not scalable
            [0.0, 0.0, 0.0],  # zero vector
        ]
    ).transpose()
    result = xtal.scale_columns_to_int_if_possible(M)
    assert result.shape == (3, 3)
    assert np.allclose(result[:, 0], np.array([3.0, 1.0, 0.0]))
    assert np.allclose(result[:, 1], M[:, 1])
    assert np.allclose(result[:, 2], M[:, 2])

    with pytest.raises(ValueError, match="must have shape"):
        xtal.scale_columns_to_int_if_possible([1.0, 0.0, 0.0])


def test_cartesian_to_miller_hexagonal():
    a = 3.23
    c = 5.17
    lattice = xtal_lattices.hexagonal(a=a, c=c)

    # Round trip: hkl -> Cartesian normal -> hkl
    for hkl in [[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 2], [1, 0, 2]]:
        n_cart = xtal.miller_plane_to_cartesian(lattice, hkl)
        assert n_cart.shape == (3,)
        result = xtal.cartesian_to_miller_plane(lattice, n_cart)
        assert np.array_equal(result, np.array(hkl))

    # Round trip: uvw -> Cartesian direction -> uvw
    for uvw in [[0, 0, 1], [1, 0, 0], [1, 1, 0], [2, 1, 0], [1, -1, 1]]:
        d_cart = xtal.miller_direction_to_cartesian(lattice, uvw)
        assert d_cart.shape == (3,)
        result = xtal.cartesian_to_miller_direction(lattice, d_cart)
        assert np.array_equal(result, np.array(uvw))

    # The basal plane normal is along z, and the c-axis direction is along z
    assert np.allclose(
        xtal.miller_plane_to_cartesian(lattice, [0, 0, 1]),
        np.array([0.0, 0.0, 2.0 * math.pi / c]),
    )
    assert np.allclose(
        xtal.miller_direction_to_cartesian(lattice, [0, 0, 1]),
        np.array([0.0, 0.0, c]),
    )

    # The magnitude of the input normal does not matter
    n_cart = xtal.miller_plane_to_cartesian(lattice, [1, 0, 2])
    assert np.array_equal(
        xtal.cartesian_to_miller_plane(lattice, 100.0 * n_cart),
        np.array([1, 0, 2]),
    )
    assert np.array_equal(
        xtal.cartesian_to_miller_plane(lattice, 1e-4 * n_cart),
        np.array([1, 0, 2]),
    )


def test_cartesian_to_miller_bravais_hexagonal():
    lattice = xtal_lattices.hexagonal(a=3.23, c=5.17)

    # {10-12} tension twinning plane, from its Cartesian normal
    n_cart = xtal.miller_plane_to_cartesian(lattice, [1, 0, 2])
    hkl = xtal.cartesian_to_miller_plane(lattice, n_cart)
    hkil = xtal.miller_to_miller_bravais_plane(hkl)
    assert np.array_equal(hkil, np.array([1.0, 0.0, -1.0, 2.0]))

    # <11-20> close-packed direction, from its Cartesian direction
    d_cart = xtal.miller_direction_to_cartesian(lattice, [1, 1, 0])
    uvw = xtal.cartesian_to_miller_direction(lattice, d_cart)
    uvtw = xtal.scale_to_int(xtal.miller_to_miller_bravais_direction(uvw))
    assert np.array_equal(uvtw, np.array([1, 1, -2, 0]))


def test_cartesian_to_miller_conventional_lattice():
    # The conventional cell indices use case: pass a different lattice.
    # Use a primitive rhombohedral lattice and its conventional hexagonal cell.
    a_hex = 4.0
    c_hex = 12.0
    conventional_lattice = xtal_lattices.hexagonal(a=a_hex, c=c_hex)
    C = conventional_lattice.column_vector_matrix()

    # Obverse setting: the primitive rhombohedral lattice vectors, in terms of the
    # conventional hexagonal lattice vectors
    T = np.array(
        [
            [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
        ]
    ).transpose()
    primitive_lattice = xtal.Lattice(C @ T)

    # The primitive lattice really is a sublattice of the conventional one
    assert math.isclose(
        conventional_lattice.volume(), 3.0 * primitive_lattice.volume(), abs_tol=1e-8
    )

    # The basal plane: (0001) in the conventional hexagonal cell.
    n_cart = xtal.miller_plane_to_cartesian(conventional_lattice, [0, 0, 1])
    assert np.array_equal(
        xtal.cartesian_to_miller_plane(conventional_lattice, n_cart),
        np.array([0, 0, 1]),
    )
    # Same plane, different indices in the primitive rhombohedral cell: (111)
    assert np.array_equal(
        xtal.cartesian_to_miller_plane(primitive_lattice, n_cart),
        np.array([1, 1, 1]),
    )

    # The c-axis direction: [0001] conventional, [111] primitive
    d_cart = xtal.miller_direction_to_cartesian(conventional_lattice, [0, 0, 1])
    assert np.array_equal(
        xtal.cartesian_to_miller_direction(conventional_lattice, d_cart),
        np.array([0, 0, 1]),
    )
    assert np.array_equal(
        xtal.cartesian_to_miller_direction(primitive_lattice, d_cart),
        np.array([1, 1, 1]),
    )


def test_cartesian_to_miller_irrational():
    lattice = xtal_lattices.hexagonal(a=3.23, c=5.17)
    n_cart = np.array([1.0, math.sqrt(2.0), math.pi])
    assert xtal.cartesian_to_miller_plane(lattice, n_cart) is None
    assert xtal.cartesian_to_miller_direction(lattice, n_cart) is None

    with pytest.raises(ValueError, match="must have shape"):
        xtal.cartesian_to_miller_plane(lattice, [1.0, 0.0])
    with pytest.raises(ValueError, match="must have shape"):
        xtal.cartesian_to_miller_direction(lattice, [1.0, 0.0])


def test_is_hexagonal_or_trigonal():
    # hexagonal
    assert xtal.is_hexagonal_or_trigonal(xtal_lattices.hexagonal(a=3.23, c=5.17))
    assert xtal.is_hexagonal_or_trigonal(xtal_lattices.HCP(a=3.23))

    # trigonal / rhombohedral
    assert xtal.is_hexagonal_or_trigonal(xtal_lattices.rhombohedral(a=4.0, alpha=70.0))

    # cubic: has four three-fold axes
    assert not xtal.is_hexagonal_or_trigonal(xtal_lattices.cubic(4.0))
    assert not xtal.is_hexagonal_or_trigonal(xtal_lattices.BCC(a=4.0))
    assert not xtal.is_hexagonal_or_trigonal(xtal_lattices.FCC(a=4.0))

    # other crystal families: no three-fold axis
    assert not xtal.is_hexagonal_or_trigonal(xtal_lattices.tetragonal(a=3.0, c=5.0))
    assert not xtal.is_hexagonal_or_trigonal(
        xtal_lattices.orthorhombic(a=3.0, b=4.0, c=5.0)
    )
    assert not xtal.is_hexagonal_or_trigonal(
        xtal_lattices.monoclinic(a=3.0, b=4.0, c=5.0, beta=100.0)
    )
    assert not xtal.is_hexagonal_or_trigonal(
        xtal_lattices.triclinic(a=3.0, b=4.0, c=5.0, alpha=88.0, beta=100.0, gamma=93.0)
    )

    # a rhombohedral lattice with alpha=60 degrees is actually FCC
    assert not xtal.is_hexagonal_or_trigonal(
        xtal_lattices.rhombohedral(a=4.0, alpha=60.0)
    )


def test_is_hexagonal_or_trigonal_point_group():
    lattice = xtal_lattices.hexagonal(a=3.23, c=5.17)
    point_group = xtal.make_point_group(lattice)
    assert len(point_group) == 24
    assert xtal.is_hexagonal_or_trigonal(point_group=point_group)

    # `point_group` takes precedence over `lattice`
    assert xtal.is_hexagonal_or_trigonal(
        lattice=xtal_lattices.cubic(4.0),
        point_group=point_group,
    )

    with pytest.raises(ValueError, match="is required"):
        xtal.is_hexagonal_or_trigonal()
