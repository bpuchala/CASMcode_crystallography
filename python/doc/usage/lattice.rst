Lattice construction and symmetry analysis
==========================================

Lattice construction
--------------------

The :class:`~casm.xtal.Lattice` class represents a three-dimensional lattice. It is constructed by providing the lattice vectors as columns of a shape=(3,3) array.

.. code-block:: Python

    import math
    import numpy as np
    import casm.xtal as xtal

    # Lattice vectors
    a = 3.23398686
    c = 5.16867834
    lattice_column_vector_matrix = np.array([
        [a, 0., 0.],  # a, along x
        [-a / 2., a * math.sqrt(3.) / 2., 0.],  # a
        [0., 0., c],  # c
    ]).transpose()  # <--- note transpose
    lattice = xtal.Lattice(lattice_column_vector_matrix)


Coordinate conversions
----------------------

This column-vector convention is used throughout CASM to represent basis vectors and values because it allows easily transforming values between different bases. For instance, coordinates stored as columns in shape=(3,n) arrays can be transformed between fractional and Cartesian coordinates using:

.. code-block:: Python

    coordinate_cart = lattice.column_vector_matrix() @ coordinate_frac
    coordinate_frac = np.linalg.pinv(lattice.column_vector_matrix()) @ coordinate_cart

For clarity and ease of use, casm-xtal also includes equivalent methods, :func:`~casm.xtal.fractional_to_cartesian()` and :func:`~casm.xtal.cartesian_to_fractional()`, for performing these transformations:

.. code-block:: Python

    coordinate_cart = xtal.fractional_to_cartesian(lattice, coordinate_frac)
    coordinate_frac = xtal.cartesian_to_fractional(lattice, coordinate_cart)

Additionally, the :func:`~casm.xtal.fractional_within()` can be used to set fractional coordinates with values less than 0. or greater than or equal to 1. to the equivalent values within the lattice unit cell.


Symmetry operations
-------------------

A symmetry operation transforms a spatial coordinate according to :math:`\vec{r}_{cart}\rightarrow A \vec{r}_{cart}+\vec{\tau}`, where :math:`A` is the shape=(3,3) `operation matrix` and :math:`\vec{\tau}` is the `translation vector`.

An instance of the :class:`~casm.xtal.SymOp` class, op, is used to represent a symmetry operation that transforms Cartesian coordinates according to:

.. code-block:: Python

    r_after = op.matrix() @ r_before + op.translation()

where r_before and r_after are shape=(3,n) arrays with the Cartesian coordinates as columns of the matrices before and after transformation, respectively.

Additionally, for magnetic materials there may be time reversal symmetry. A symmetry operation transforms magnetic spins according to:

.. code-block:: Python

    if op.time_reversal() is True:
        s_after = -s_before

where s_before and s_after are the spins before and after transformation, respectively.


Lattice point group generation
------------------------------

The point group is the set of symmetry operations that transform the lattice vectors but leave all the lattice points (the points that are integer multiples of the lattice vectors) invariant. The lattice point group can be generated using the :func:`~casm.xtal.make_point_group()` method. For the example of a simple cubic lattice, the lattice point group has 48 operations:

.. code-block:: Python

    >>> lattice = xtal.Lattice(np.eye(3))
    >>> point_group = xtal.make_point_group(lattice)
    >>> len(point_group)
    48


.. _lattice-symmetry-operation-information:

Symmetry operation information
------------------------------

The :class:`~casm.xtal.SymInfo` class is used to determine information about a :class:`~casm.xtal.SymOp`, such as:

- The type of symmetry operation
- The axis of rotation or mirror plane normal
- The angle of rotation
- The location of an invariant point
- The screw or glide translation component

The symmetry information for the point group operations can be constructed from the :class:`~casm.xtal.SymOp` and the :class:`~casm.xtal.Lattice`:

.. code-block:: Python

    >>> syminfo = xtal.SymInfo(point_group[1], lattice)
    >>> print("op_type:", syminfo.op_type())
    op_type: rotation
    >>> print("axis:", syminfo.axis())
    axis: [1. 0. 0.]
    >>> print("angle:", syminfo.angle())
    angle: 180.0
    >>> print("location:", syminfo.location())
    location: [0. 0. 0.]

A brief description can also be printed following the conventions of International Tables for Crystallography, and using either fractional or Cartesian coordinates, using the :func:`~casm.xtal.SymInfo.brief_frac()` or :func:`~casm.xtal.SymInfo.brief_cart()` methods of :class:`~casm.xtal.SymInfo`:

.. code-block:: Python

    >>> i = 1
    >>> for op in point_group:
    ...     syminfo = xtal.SymInfo(op, lattice)
    ...     print(str(i) + ":", syminfo.brief_cart())
    ...     i += 1
    ...
    1: -1 0.0000000 0.0000000 0.0000000
    2: 2 x, 0, 0
    3: 2 0.7071068*x, -0.7071068*x, 0
    4: -4??? 0, 0, z; 0.0000000 0.0000000 0.0000000
    5: -4??? 0, 0, z; 0.0000000 0.0000000 0.0000000
    6: 2 0.7071068*x, 0.7071068*x, 0
    ...
    44: 4??? 0, 0, z
    45: 4??? 0, 0, z
    46: m 0.7071068*x, 0.7071068*x, z
    47: m 0, y, z
    48: 1


Lattice equivalence
-------------------

A lattice can be represented using any choice of lattice vectors that results in the same lattice points. The :func:`~casm.xtal.is_equivalent_to` method checks for the equivalence lattices that do not have identical lattice vectors by determining if one choice of lattice vectors can be formed by linear combination of the others according to :math:`L_1 = L_2 U`, where :math:`L_1` and :math:`L_2` are the lattice vectors as columns of matrices, and :math:`U` is an integer matrix with :math:`\det(U) = \pm 1`:

.. code-block:: Python

    >>> lattice1 = xtal.Lattice(np.array([
    ...     [1., 0., 0.], # 'a'
    ...     [0., 1., 0.], # 'b'
    ...     [0., 0., 1.]  # 'c'
    ... ]).transpose())
    >>> lattice2 = xtal.Lattice(np.array([
    ...     [1., 1., 0.], # 'a' + 'b'
    ...     [0., 1., 0.], # 'b'
    ...     [0., 0., 1.]  # 'c'
    ... ]).transpose())
    >>> print(lattice1 == lattice2) # checks if lattice vectors are ~equal
    False
    >>> print(xtal.is_equivalent_to(lattice1, lattice2)) # checks if lattice points are ~equal
    True


Lattice canonical form
----------------------

For clarity and comparison purposes it useful to have a canonical choice of equivalent lattice vectors. The :func:`~casm.xtal.make_canonical` method finds the canonical right-handed Niggli cell of the lattice, by applying lattice point group operations to find the equivalent lattice in the orientiation which compares greatest.

.. code-block:: Python

    >>> noncanonical_lattice = xtal.Lattice(
    ...     np.array([
    ...             [0., 0., 2.], # c (along z)
    ...             [1., 0., 0.], # a (along x)
    ...             [0., 1., 0.]] # a (along y)
    ...     ).transpose())
    >>> canonical_lattice = xtal.make_canonical(noncanonical_lattice)
    >>> print(canonical_lattice.column_vector_matrix().transpose())
    [[1. 0. 0.]  # a (along x)
     [0. 1. 0.]  # a (along y)
     [0. 0. 2.]] # c (along z)
    >>> print(canonical_lattice > noncanonical_lattice)
    True

The lattice comparison method prefers lattice vectors that form symmetric matrices with large positive values on the diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

.. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/
