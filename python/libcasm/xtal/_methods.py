import functools
import math
from collections import namedtuple
from typing import Any, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

import libcasm.casmglobal
import libcasm.xtal._xtal as _xtal


def make_primitive(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> Any:
    """Make the primitive cell of a Prim or atomic Structure

    Notes
    -----
    Currently, for Structure this method only considers atom coordinates and types.
    Molecular coordinates and types are not considered. Properties are not considered.
    The default CASM tolerance is used for comparisons. To consider molecules
    or properties, or to use a different tolerance, use a Prim.

    Parameters
    ----------
    obj: Union[ _xtal.Prim, _xtal.Structure]
        A Prim or an atomic Structure, which determines whether
        :func:`~libcasm.xtal.make_primitive_prim`, or
        :func:`~libcasm.xtal.make_primitive_structure` is called.

    Returns
    -------
    canonical_obj : Union[_xtal.Prim, _xtal.Structure]
        The primitive equivalent Prim or atomic Structure.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_primitive_prim(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_primitive_structure(obj)
    else:
        raise TypeError(f"TypeError in make_primitive: received {type(obj).__name__}")


def make_canonical(
    obj: Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure],
) -> Any:
    """Make an equivalent Lattice, Prim, or Structure with the canonical form
    of the lattice

    Parameters
    ----------
    obj: Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure]
        A Lattice, Prim, or Structure, which determines whether
        :func:`~libcasm.xtal.make_canonical_lattice`, or
        :func:`~libcasm.xtal.make_canonical_prim`,
        :func:`~libcasm.xtal.make_canonical_structure` is called.

    Returns
    -------
    canonical_obj : Union[_xtal.Lattice, _xtal.Prim, _xtal.Structure]
        The equivalent Lattice, Prim, or Structure with canonical form of the lattice.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_canonical_prim(obj)
    elif isinstance(obj, _xtal.Lattice):
        return _xtal.make_canonical_lattice(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_canonical_structure(obj)
    else:
        raise TypeError(f"TypeError in make_canonical: received {type(obj).__name__}")


def make_crystal_point_group(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> list[_xtal.SymOp]:
    """Make the crystal point group of a Prim or Structure

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_crystal_point_group` or
        :func:`~libcasm.xtal.make_structure_crystal_point_group` is called.

    Returns
    -------
    crystal_point_group : list[:class:`~libcasm.xtal.SymOp`]
        The crystal point group is the group constructed from the factor
        group operations with translation vector set to zero.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_crystal_point_group(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_crystal_point_group(obj)
    else:
        raise TypeError(
            f"TypeError in make_crystal_point_group: received {type(obj).__name__}"
        )


def make_factor_group(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> list[_xtal.SymOp]:
    """Make the factor group of a Prim or Structure

    Notes
    -----
    For :class:`~libcasm.xtal.Structure`, this method only considers atom coordinates
    and types. Molecular coordinates and types are not considered. Properties are not
    considered. The default CASM tolerance is used for comparisons. To consider
    molecules or properties, or to use a different tolerance, use a
    :class:`~libcasm.xtal.Prim` with :class:`~libcasm.xtal.Occupant` that have
    properties.

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_factor_group` or
        :func:`~libcasm.xtal.make_structure_factor_group` is called.

    Returns
    -------
    factor_group : list[:class:`~libcasm.xtal.SymOp`]
        The set of symmery operations, with translation lying within the
        primitive unit cell, that leave the lattice vectors, global DoF
        (for :class:`~libcasm.xtal.Prim`), and basis site coordinates and local DoF
        (for :class:`~libcasm.xtal.Prim`) or atom coordinates and atom types
        (for :class:`~libcasm.xtal.Structure`) invariant.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_factor_group(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_factor_group(obj)
    else:
        raise TypeError(
            f"TypeError in make_factor_group: received {type(obj).__name__}"
        )


def make_within(
    obj: Union[_xtal.Prim, _xtal.Structure],
) -> Any:
    """Returns an equivalent Prim or Structure with all site coordinates within the \
    unit cell

    Parameters
    ----------
    obj: Union[_xtal.Prim, _xtal.Structure]
        A Prim or Structure, which determines whether
        :func:`~libcasm.xtal.make_prim_within` or
        :func:`~libcasm.xtal.make_structure_within` is called.

    Returns
    -------
    obj_within : Any
        An equivalent Prim or Structure with all site coordinates within the \
        unit cell.
    """
    if isinstance(obj, _xtal.Prim):
        return _xtal.make_prim_within(obj)
    elif isinstance(obj, _xtal.Structure):
        return _xtal.make_structure_within(obj)
    else:
        raise TypeError(f"TypeError in make_within: received {type(obj).__name__}")


@functools.total_ordering
class ApproximateFloatArray:
    def __init__(
        self,
        arr: np.ndarray,
        abs_tol: float = libcasm.casmglobal.TOL,
    ):
        """Store an array that will be compared lexicographically up to a given
        absolute tolerance using math.isclose

        Parameters
        ----------
        arr: numpy.ndarray
            The array to be compared

        abs_tol: float = :data:`~libcasm.casmglobal.TOL`
            The absolute tolerance
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                "Error in ApproximateFloatArray: arr must be a numpy.ndarray"
            )
        self.arr = arr
        self.abs_tol = abs_tol

    def __eq__(self, other):
        if len(self.arr) != len(other.arr):
            return False
        for i in range(len(self.arr)):
            if not math.isclose(self.arr[i], other.arr[i], abs_tol=self.abs_tol):
                return False
        return True

    def __lt__(self, other):
        if len(self.arr) != len(other.arr):
            return len(self.arr) < len(other.arr)
        for i in range(len(self.arr)):
            if not math.isclose(self.arr[i], other.arr[i], abs_tol=self.abs_tol):
                return self.arr[i] < other.arr[i]
        return False


StructureAtomInfo = namedtuple(
    "StructureAtomInfo",
    ["atom_type", "atom_coordinate_frac", "atom_coordinate_cart", "atom_properties"],
)
""" A namedtuple, used to hold atom info when sorting, filtering, etc. atoms in a 
:class:`~_xtal.Structure`.

.. rubric:: Constructor

Parameters
----------
atom_type: str
    The atom type, from :func:`~_xtal.Structure.atom_type`.
atom_coordinate_frac: numpy.ndarray[numpy.float64[3]]
    The fractional coordinate of the atom, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_frac`.
atom_coordinate_cart: numpy.ndarray[numpy.float64[3]]
    The Cartesian coordinate of the atom, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_cart`.
atom_properties: dict[str, numpy.ndarray[numpy.float64[m]]]
    The continuous properties associated with the atoms, if present, from 
    :func:`~_xtal.Structure.atom_type.atom_coordinate_frac`. All atoms
    must have the same properties with values of the same dimension.
"""


def make_structure_atom_info(
    structure: _xtal.Structure,
) -> list[StructureAtomInfo]:
    """Create a list of StructureAtomInfo from a Structure

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted, filtered, etc. by atom info.

    Returns
    -------
    structure_atom_info: list[StructureAtomInfo]
        A list of StructureAtomInfo.

    """

    atom_type = structure.atom_type()
    atom_coordinate_frac = structure.atom_coordinate_frac()
    atom_coordinate_cart = structure.atom_coordinate_cart()
    atom_properties = structure.atom_properties()

    atoms = []
    import copy

    for i in range(len(atom_type)):
        atoms.append(
            StructureAtomInfo(
                copy.copy(atom_type[i]),
                atom_coordinate_frac[:, i].copy(),
                atom_coordinate_cart[:, i].copy(),
                {key: atom_properties[key][:, i].copy() for key in atom_properties},
            )
        )

    return atoms


def make_structure_from_atom_info(
    lattice: _xtal.Lattice,
    atoms: list[StructureAtomInfo],
    global_properties: dict[str, np.ndarray[np.float64]] = {},
) -> _xtal.Structure:
    """Create a Structure from a list of StructureAtomInfo

    Parameters
    ----------
    lattice: _xtal.Lattice]
        The lattice for the resulting structure.
    atoms: list[StructureAtomInfo]
        A list of StructureAtomInfo. The Cartesian coordinates are used when setting
        atom coordinates in the resulting structure.
    global_properties: dict[str, numpy.ndarray[numpy.float64[m, n]]] = {}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    structure: _xtal.Structure
        The resulting structure
    """

    n_atoms = len(atoms)

    atom_type = [atom.atom_type for atom in atoms]
    atom_coordinate_cart = np.zeros((3, n_atoms))
    atom_properties = {}

    for i, atom in enumerate(atoms):
        if i == 0:
            for key, value in atom.atom_properties.items():
                dim = value.shape[0]
                atom_properties[key] = np.zeros((dim, n_atoms))

        atom_coordinate_cart[:, i] = atom.atom_coordinate_cart
        for key, value in atom.atom_properties.items():
            atom_properties[key][:, i] = atom.atom_properties[key]

    atom_coordinate_frac = _xtal.cartesian_to_fractional(
        lattice=lattice,
        coordinate_cart=atom_coordinate_cart,
    )

    return _xtal.Structure(
        lattice=lattice,
        atom_type=atom_type,
        atom_coordinate_frac=atom_coordinate_frac,
        atom_properties=atom_properties,
        global_properties=global_properties,
    )


def sort_structure_by_atom_info(
    structure: _xtal.Structure,
    key: Callable[[StructureAtomInfo], Any],
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    key: Callable[[StructureAtomInfo], Any]
        The function used to return a value which is sorted. This is passed to the
        `key` parameter of `list.sort()` to sort a `list[StructureAtomInfo]`.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted as specified.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    if len(structure.mol_type()) != 0:
        raise ValueError(
            "Error: only atomic structures may be sorted using sort_by_atom_info"
        )

    atoms = make_structure_atom_info(structure)
    atoms.sort(key=key, reverse=reverse)

    return make_structure_from_atom_info(
        lattice=structure.lattice(),
        atoms=atoms,
        global_properties=structure.global_properties(),
    )


def sort_structure_by_atom_type(
    structure: _xtal.Structure,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by atom type

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by atom type.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """
    return sort_structure_by_atom_info(
        structure,
        key=lambda atom_info: atom_info.atom_type,
        reverse=reverse,
    )


def sort_structure_by_atom_coordinate_frac(
    structure: _xtal.Structure,
    order: str = "cba",
    abs_tol: float = libcasm.casmglobal.TOL,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by fractional coordinates

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    order: str = "cba"
        Sort order of fractional coordinate components. Default "cba" sorts by
        fractional coordinate along the "c" (third) lattice vector first, "b" (second)
        lattice vector second, and "a" (first) lattice vector third.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by fractional coordinates.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    def compare_f(atom_info):
        values = []
        for i in range(len(order)):
            if order[i] == "a":
                values.append(atom_info.atom_coordinate_frac[0])
            elif order[i] == "b":
                values.append(atom_info.atom_coordinate_frac[1])
            elif order[i] == "c":
                values.append(atom_info.atom_coordinate_frac[2])

        return ApproximateFloatArray(
            arr=np.array(values),
            abs_tol=abs_tol,
        )

    return sort_structure_by_atom_info(
        structure,
        key=compare_f,
        reverse=reverse,
    )


def sort_structure_by_atom_coordinate_cart(
    structure: _xtal.Structure,
    order: str = "zyx",
    abs_tol: float = libcasm.casmglobal.TOL,
    reverse: bool = False,
) -> _xtal.Structure:
    """Sort an atomic structure by Cartesian coordinates

    Parameters
    ----------
    structure: _xtal.Structure
        The structure to be sorted. Must be an atomic structure only.
    order: str = "zyx"
        Sort order of Cartesian coordinate components. Default "zyx" sorts by
        "z" Cartesian coordinate first, "y" Cartesian coordinate second, and "x"
        Cartesian coordinate third.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        Floating point tolerance for coordinate comparisons.
    reverse: bool = False
        By default, sort in ascending order. If ``reverse==True``, then sort in
        descending order.

    Returns
    -------
    sorted_structure: _xtal.Structure
        An equivalent structure with atoms sorted by Cartesian coordinates.

    Raises
    ------
    ValueError
        For non-atomic structure, if ``len(structure.mol_type()) != 0``.
    """

    def compare_f(atom_info):
        values = []
        for i in range(len(order)):
            if order[i] == "x":
                values.append(atom_info.atom_coordinate_frac[0])
            elif order[i] == "y":
                values.append(atom_info.atom_coordinate_frac[1])
            elif order[i] == "z":
                values.append(atom_info.atom_coordinate_frac[2])

        return ApproximateFloatArray(
            arr=np.array(values),
            abs_tol=abs_tol,
        )

    return sort_structure_by_atom_info(
        structure,
        key=compare_f,
        reverse=reverse,
    )


def substitute_structure_species(
    structure: _xtal.Structure,
    substitutions: dict[str, str],
) -> _xtal.Structure:
    """Create a copy of a structure with renamed atomic and molecular species

    Parameters
    ----------
    structure: _xtal.Structure
        The initial structure
    substitutions: dict[str, str]
        The substitutions to make, using the convention key->value. For example, using
        ``substitutions = { "B": "C"}`` results in all `atom_type` and `mol_type`
        equal to "B" in the input structure being changed to "C" in the output
        structure.

    Returns
    -------
    structure_with_substitutions: _xtal.Structure
        A copy of `structure`, with substitutions of `atom_type` and `mol_type`.
    """
    return _xtal.Structure(
        lattice=structure.lattice(),
        atom_coordinate_frac=structure.atom_coordinate_frac(),
        atom_type=[substitutions.get(x, x) for x in structure.atom_type()],
        atom_properties=structure.atom_properties(),
        mol_coordinate_frac=structure.mol_coordinate_frac(),
        mol_type=[substitutions.get(x, x) for x in structure.mol_type()],
        mol_properties=structure.mol_properties(),
        global_properties=structure.global_properties(),
    )


def filter_structure_by_atom_info(
    structure: _xtal.Structure,
    filter: Callable[[StructureAtomInfo], Any],
) -> _xtal.Structure:
    """Return a copy of a structure with atoms passing a filter function

    .. rubric:: Example usage

    .. code-block:: Python

        # Remove all atoms with z coordinate >= 2.0
        structure_without_Al = filter_structure_by_atom_info(
            input_structure,
            lambda atom_info: atom_info.atom_coordinate_cart[2] < 2.0,
        )

    Parameters
    ----------
    structure: _xtal.Structure
        The initial structure
    filter: Callable[[StructureAtomInfo], Any]
        A function of `StructureAtomInfo` which returns True for atoms that should be
        kept and False for atoms that should be removed.

    Returns
    -------
    structure_after_removals: _xtal.Structure
        A copy of `structure` excluding atoms for which the `filter` function returns
        False.
    """
    return make_structure_from_atom_info(
        lattice=structure.lattice(),
        atoms=[x for x in make_structure_atom_info(structure) if filter(x)],
        global_properties=structure.global_properties(),
    )


def combine_structures(
    structures: list[_xtal.Structure],
    lattice: Optional[_xtal.Lattice] = None,
    global_properties: dict[str, np.ndarray[np.float64]] = {},
) -> _xtal.Structure:
    """Return a new structure which combines the atomic species of all input structures

    Parameters
    ----------
    structures: list[_xtal.Structure]
        The structures to be combined.
    lattice: Optional[_xtal.Lattice] = None
        If not None, the lattice of the resulting structure. If None, the resulting
        structure will use the lattice of the first structure.
    global_properties: dict[str, numpy.ndarray[numpy.float64[m, n]]] = {}
        Continuous properties associated with entire crystal, if present. Keys must be
        the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.

    Returns
    -------
    combined_structure: _xtal.Structure
        The resulting structure, which contains the species of all input structures
        with Cartesian coordinates fixed to the same values as in the input structures.
        All atom or molecule properties remain the same as in the input structures.
    """
    atoms = []
    for structure in structures:
        if lattice is None:
            lattice = structure.lattice()
        atoms += make_structure_atom_info(structure)
    return make_structure_from_atom_info(
        lattice=lattice,
        atoms=atoms,
        global_properties=global_properties,
    )


### Miller and Miller-Bravais indices ###

# (u,v,w) -> (U,V,T,W), as a matrix acting on Miller direction indices
_UVTW_FROM_UVW = np.array(
    [
        [2.0 / 3.0, -1.0 / 3.0, 0.0],
        [-1.0 / 3.0, 2.0 / 3.0, 0.0],
        [-1.0 / 3.0, -1.0 / 3.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# (U,V,T,W) -> (u,v,w), as a matrix acting on Miller-Bravais direction indices
_UVW_FROM_UVTW = np.array(
    [
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# (h,k,l) -> (h,k,i,l), as a matrix acting on Miller plane indices
_HKIL_FROM_HKL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# (h,k,i,l) -> (h,k,l), as a matrix acting on Miller-Bravais plane indices
_HKL_FROM_HKIL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _as_indices_array(
    indices: npt.ArrayLike,
    dim: int,
    what: str,
    method: str,
) -> np.ndarray:
    """Validate and convert index input to a shape=(dim,) or shape=(dim,n) array"""
    arr = np.asarray(indices, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.shape[0] != dim:
        raise ValueError(
            f"Error in {method}: {what} must have shape=({dim},) or shape=({dim},n); "
            f"received shape={arr.shape}"
        )
    return arr


def _validate_sum_is_zero(
    arr: np.ndarray,
    constraint: str,
    method: str,
    abs_tol: float,
) -> None:
    """Validate the first three components of 4-index input sum to zero"""
    total = arr[0] + arr[1] + arr[2]
    if not np.all(np.abs(total) < abs_tol):
        raise ValueError(
            f"Error in {method}: the Miller-Bravais constraint {constraint} "
            f"is not satisfied (to within abs_tol={abs_tol})"
        )


def miller_to_miller_bravais_direction(
    uvw_indices: npt.ArrayLike,
) -> np.ndarray:
    r"""Convert Miller direction indices, :math:`[uvw]`, to Miller-Bravais direction
    indices, :math:`[UVTW]`

    Notes
    -----
    This is the *direction* (lattice vector) convention, in which

    .. math::

        U = (2u - v)/3, \quad V = (2v - u)/3, \quad T = -(U + V), \quad W = w,

    so that :math:`U + V + T = 0` and
    :math:`U \vec{a}_1 + V \vec{a}_2 + T \vec{a}_3 + W \vec{c}` is the same lattice
    vector as :math:`u \vec{a}_1 + v \vec{a}_2 + w \vec{c}`, where
    :math:`\vec{a}_3 = -(\vec{a}_1 + \vec{a}_2)`.

    This is *not* the same transformation used for plane indices. For planes, use
    :func:`~libcasm.xtal.miller_to_miller_bravais_plane`.

    Integer :math:`[uvw]` generally maps to :math:`[UVTW]` with thirds. The
    conventional integer form can be obtained using
    :func:`~libcasm.xtal.scale_to_int`. For example:

    .. code-block:: Python

        >>> uvtw = xtal.miller_to_miller_bravais_direction([1.0, 1.0, 0.0])
        >>> print(uvtw)
        [ 0.33333333  0.33333333 -0.66666667  0.        ]
        >>> print(xtal.scale_to_int(uvtw))
        [ 1  1 -2  0]

    Parameters
    ----------
    uvw_indices: array_like
        The Miller direction indices, :math:`(u, v, w)`, either as a single direction,
        with shape=(3,), or as columns of a shape=(3,n) array.

    Returns
    -------
    uvtw_indices: numpy.ndarray[numpy.float64]
        The Miller-Bravais direction indices, :math:`(U, V, T, W)`, with shape=(4,) or
        shape=(4,n), matching the shape of `uvw_indices`. The output satisfies
        :math:`U + V + T = 0`.

    Raises
    ------
    ValueError
        If `uvw_indices` does not have shape=(3,) or shape=(3,n).
    """
    arr = _as_indices_array(
        uvw_indices, 3, "uvw_indices", "miller_to_miller_bravais_direction"
    )
    return _UVTW_FROM_UVW @ arr


def miller_bravais_to_miller_direction(
    uvtw_indices: npt.ArrayLike,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> np.ndarray:
    r"""Convert Miller-Bravais direction indices, :math:`[UVTW]`, to Miller direction
    indices, :math:`[uvw]`

    Notes
    -----
    This is the *direction* (lattice vector) convention, in which

    .. math::

        u = U - T, \quad v = V - T, \quad w = W.

    The input must satisfy the Miller-Bravais direction constraint,
    :math:`U + V + T = 0`.

    This is *not* the same transformation used for plane indices. For planes, use
    :func:`~libcasm.xtal.miller_bravais_to_miller_plane`.

    Parameters
    ----------
    uvtw_indices: array_like
        The Miller-Bravais direction indices, :math:`(U, V, T, W)`, either as a single
        direction, with shape=(4,), or as columns of a shape=(4,n) array. Must satisfy
        :math:`U + V + T = 0`.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check the :math:`U + V + T = 0` constraint.

    Returns
    -------
    uvw_indices: numpy.ndarray[numpy.float64]
        The Miller direction indices, :math:`(u, v, w)`, with shape=(3,) or
        shape=(3,n), matching the shape of `uvtw_indices`.

    Raises
    ------
    ValueError
        If `uvtw_indices` does not have shape=(4,) or shape=(4,n), or if the
        :math:`U + V + T = 0` constraint is not satisfied to within `abs_tol`.
    """
    arr = _as_indices_array(
        uvtw_indices, 4, "uvtw_indices", "miller_bravais_to_miller_direction"
    )
    _validate_sum_is_zero(
        arr, "U + V + T = 0", "miller_bravais_to_miller_direction", abs_tol
    )
    return _UVW_FROM_UVTW @ arr


def miller_to_miller_bravais_plane(
    hkl_indices: npt.ArrayLike,
) -> np.ndarray:
    r"""Convert Miller plane indices, :math:`(hkl)`, to Miller-Bravais plane indices,
    :math:`(hkil)`

    Notes
    -----
    This is the *plane* (reciprocal lattice vector) convention, in which

    .. math::

        i = -(h + k),

    with :math:`h`, :math:`k`, and :math:`l` unchanged, so that
    :math:`h + k + i = 0`. The fourth index is redundant: it is the intercept index
    for the :math:`\vec{a}_3 = -(\vec{a}_1 + \vec{a}_2)` axis.

    This is *not* the same transformation used for direction indices. For directions,
    use :func:`~libcasm.xtal.miller_to_miller_bravais_direction`.

    Parameters
    ----------
    hkl_indices: array_like
        The Miller plane indices, :math:`(h, k, l)`, either as a single plane, with
        shape=(3,), or as columns of a shape=(3,n) array.

    Returns
    -------
    hkil_indices: numpy.ndarray[numpy.float64]
        The Miller-Bravais plane indices, :math:`(h, k, i, l)`, with shape=(4,) or
        shape=(4,n), matching the shape of `hkl_indices`. The output satisfies
        :math:`h + k + i = 0`.

    Raises
    ------
    ValueError
        If `hkl_indices` does not have shape=(3,) or shape=(3,n).
    """
    arr = _as_indices_array(
        hkl_indices, 3, "hkl_indices", "miller_to_miller_bravais_plane"
    )
    return _HKIL_FROM_HKL @ arr


def miller_bravais_to_miller_plane(
    hkil_indices: npt.ArrayLike,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> np.ndarray:
    r"""Convert Miller-Bravais plane indices, :math:`(hkil)`, to Miller plane indices,
    :math:`(hkl)`

    Notes
    -----
    This is the *plane* (reciprocal lattice vector) convention, in which the redundant
    third index, :math:`i = -(h + k)`, is simply dropped.

    The input must satisfy the Miller-Bravais plane constraint,
    :math:`h + k + i = 0`.

    This is *not* the same transformation used for direction indices. For directions,
    use :func:`~libcasm.xtal.miller_bravais_to_miller_direction`.

    Parameters
    ----------
    hkil_indices: array_like
        The Miller-Bravais plane indices, :math:`(h, k, i, l)`, either as a single
        plane, with shape=(4,), or as columns of a shape=(4,n) array. Must satisfy
        :math:`h + k + i = 0`.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check the :math:`h + k + i = 0` constraint.

    Returns
    -------
    hkl_indices: numpy.ndarray[numpy.float64]
        The Miller plane indices, :math:`(h, k, l)`, with shape=(3,) or shape=(3,n),
        matching the shape of `hkil_indices`.

    Raises
    ------
    ValueError
        If `hkil_indices` does not have shape=(4,) or shape=(4,n), or if the
        :math:`h + k + i = 0` constraint is not satisfied to within `abs_tol`.
    """
    arr = _as_indices_array(
        hkil_indices, 4, "hkil_indices", "miller_bravais_to_miller_plane"
    )
    _validate_sum_is_zero(
        arr, "h + k + i = 0", "miller_bravais_to_miller_plane", abs_tol
    )
    return _HKL_FROM_HKIL @ arr


### Rationalization of indices ###


def scale_to_int_if_possible(
    v: npt.ArrayLike,
    max_element: int = 10,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> Optional[np.ndarray]:
    """Scale a vector to the smallest parallel integer vector, if possible

    Notes
    -----
    This method finds the smallest positive integer :math:`s \\le` `max_element` such
    that :math:`s \\vec{v} / \\max_i |v_i|` has all components within `abs_tol` of an
    integer, and returns those integers. Because the input is first normalized so that
    its largest component has magnitude 1, the result always satisfies
    :math:`\\max_i |v_i| \\le` `max_element`, and it has no common integer factor.

    The sign of the input is preserved. This is the standard way to obtain integer
    Miller or Miller-Bravais indices from a vector of fractional coordinates.

    Parameters
    ----------
    v: array_like
        A vector, with shape=(n,).
    max_element: int = 10
        The maximum allowed magnitude of any element of the result. If no scaling with
        all elements of magnitude less than or equal to `max_element` gives integer
        values (to within `abs_tol`), then the vector is treated as irrational and
        None is returned.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check whether a scaled element is an integer.

    Returns
    -------
    scaled_v: Optional[numpy.ndarray[numpy.int64]]
        The smallest integer vector parallel to `v`, with shape=(n,), or None if `v`
        cannot be scaled to integers with all elements of magnitude less than or equal
        to `max_element`.

    Raises
    ------
    ValueError
        If `v` does not have shape=(n,), if `max_element` is less than 1, or if `v`
        is the zero vector (all elements with magnitude less than `abs_tol`).
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            "Error in scale_to_int_if_possible: v must have shape=(n,); "
            f"received shape={arr.shape}"
        )
    if max_element < 1:
        raise ValueError("Error in scale_to_int_if_possible: max_element must be >= 1")
    max_abs = np.max(np.abs(arr))
    if max_abs < abs_tol:
        raise ValueError("Error in scale_to_int_if_possible: v is the zero vector")

    unit = arr / max_abs
    for scale in range(1, int(max_element) + 1):
        scaled = scale * unit
        if np.all(np.abs(scaled - np.round(scaled)) < abs_tol):
            return np.round(scaled).astype(np.int64)
    return None


def scale_to_int(
    v: npt.ArrayLike,
    max_element: int = 10,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> np.ndarray:
    """Scale a vector to the smallest parallel integer vector

    Notes
    -----
    This is equivalent to :func:`~libcasm.xtal.scale_to_int_if_possible`, except that
    it raises instead of returning None when the vector cannot be scaled to integers.

    Parameters
    ----------
    v: array_like
        A vector, with shape=(n,).
    max_element: int = 10
        The maximum allowed magnitude of any element of the result.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check whether a scaled element is an integer.

    Returns
    -------
    scaled_v: numpy.ndarray[numpy.int64]
        The smallest integer vector parallel to `v`, with shape=(n,).

    Raises
    ------
    ValueError
        If `v` does not have shape=(n,), if `max_element` is less than 1, if `v` is
        the zero vector, or if `v` cannot be scaled to integers with all elements of
        magnitude less than or equal to `max_element`.
    """
    scaled_v = scale_to_int_if_possible(v, max_element=max_element, abs_tol=abs_tol)
    if scaled_v is None:
        raise ValueError(
            "Error in scale_to_int: could not scale to integers with all elements "
            f"of magnitude <= max_element={max_element}"
        )
    return scaled_v


def scale_columns_to_int_if_possible(
    M: npt.ArrayLike,
    max_element: int = 10,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> np.ndarray:
    """Scale the columns of a matrix to integer vectors, where possible

    Notes
    -----
    Each column is scaled independently, using
    :func:`~libcasm.xtal.scale_to_int_if_possible`. Columns which cannot be scaled to
    integers, and columns which are the zero vector, are copied unchanged. To
    determine which columns were successfully scaled, use
    :func:`~libcasm.xtal.scale_to_int_if_possible` column by column.

    Parameters
    ----------
    M: array_like
        A matrix, with shape=(m,n), with the vectors to be scaled as columns.
    max_element: int = 10
        The maximum allowed magnitude of any element of a scaled column.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check whether a scaled element is an integer.

    Returns
    -------
    scaled_M: numpy.ndarray[numpy.float64]
        A copy of `M`, with shape=(m,n), in which each column that can be scaled to a
        parallel integer vector with all elements of magnitude less than or equal to
        `max_element` is replaced by that integer vector. The result has floating
        point type because unscaled columns are preserved.

    Raises
    ------
    ValueError
        If `M` does not have shape=(m,n), or if `max_element` is less than 1.
    """
    arr = np.asarray(M, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "Error in scale_columns_to_int_if_possible: M must have shape=(m,n); "
            f"received shape={arr.shape}"
        )
    if max_element < 1:
        raise ValueError(
            "Error in scale_columns_to_int_if_possible: max_element must be >= 1"
        )
    result = arr.copy()
    for i in range(arr.shape[1]):
        column = arr[:, i]
        if np.max(np.abs(column)) < abs_tol:
            continue
        scaled_column = scale_to_int_if_possible(
            column, max_element=max_element, abs_tol=abs_tol
        )
        if scaled_column is not None:
            result[:, i] = scaled_column
    return result


### Cartesian coordinates <-> Miller indices ###


def miller_direction_to_cartesian(
    lattice: _xtal.Lattice,
    uvw_indices: npt.ArrayLike,
) -> np.ndarray:
    r"""Convert Miller direction indices, :math:`[uvw]`, to a Cartesian vector

    Notes
    -----
    The Cartesian vector is
    :math:`\vec{d} = u \vec{a} + v \vec{b} + w \vec{c} = L \vec{x}`, where :math:`L`
    is the lattice column vector matrix and
    :math:`\vec{x} = (u, v, w)`. In other words, Miller direction indices are the
    fractional coordinates of a lattice vector, so this is equivalent to
    :func:`~libcasm.xtal.fractional_to_cartesian`.

    Parameters
    ----------
    lattice: ~libcasm.xtal.Lattice
        The lattice that the indices are relative to. To obtain conventional cell
        indices, pass the conventional cell lattice.
    uvw_indices: array_like
        The Miller direction indices, :math:`(u, v, w)`, either as a single direction,
        with shape=(3,), or as columns of a shape=(3,n) array.

    Returns
    -------
    direction_cart: numpy.ndarray[numpy.float64]
        The direction in Cartesian coordinates, with shape=(3,) or shape=(3,n),
        matching the shape of `uvw_indices`. The vector is not normalized: its length
        is the length of the lattice vector with the given indices.

    Raises
    ------
    ValueError
        If `uvw_indices` does not have shape=(3,) or shape=(3,n).
    """
    arr = _as_indices_array(
        uvw_indices, 3, "uvw_indices", "miller_direction_to_cartesian"
    )
    return lattice.column_vector_matrix() @ arr


def miller_plane_to_cartesian(
    lattice: _xtal.Lattice,
    hkl_indices: npt.ArrayLike,
) -> np.ndarray:
    r"""Convert Miller plane indices, :math:`(hkl)`, to a Cartesian plane normal

    Notes
    -----
    Miller plane indices are the fractional coordinates of a reciprocal lattice
    vector, :math:`\vec{G} = h \vec{a}^{*} + k \vec{b}^{*} + l \vec{c}^{*} =
    R \vec{x}`, where :math:`R` is the reciprocal lattice column vector matrix,
    from :func:`~libcasm.xtal.Lattice.reciprocal`, and :math:`\vec{x} = (h, k, l)`.
    The reciprocal lattice vector :math:`\vec{G}` is normal to the :math:`(hkl)`
    plane.

    Parameters
    ----------
    lattice: ~libcasm.xtal.Lattice
        The lattice that the indices are relative to. To use conventional cell
        indices, pass the conventional cell lattice.
    hkl_indices: array_like
        The Miller plane indices, :math:`(h, k, l)`, either as a single plane, with
        shape=(3,), or as columns of a shape=(3,n) array.

    Returns
    -------
    plane_normal_cart: numpy.ndarray[numpy.float64]
        The plane normal in Cartesian coordinates, with shape=(3,) or shape=(3,n),
        matching the shape of `hkl_indices`. The vector is not normalized: because
        CASM's reciprocal lattice includes the factor :math:`2\pi`, its length is
        :math:`2\pi / d_{hkl}`, where :math:`d_{hkl}` is the interplanar spacing.

    Raises
    ------
    ValueError
        If `hkl_indices` does not have shape=(3,) or shape=(3,n).
    """
    arr = _as_indices_array(hkl_indices, 3, "hkl_indices", "miller_plane_to_cartesian")
    return lattice.reciprocal().column_vector_matrix() @ arr


def cartesian_to_miller_direction(
    lattice: _xtal.Lattice,
    direction_cart: npt.ArrayLike,
    max_element: int = 10,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> Optional[np.ndarray]:
    r"""Convert a Cartesian direction to Miller direction indices, :math:`[uvw]`

    Notes
    -----
    The Cartesian direction is expressed in fractional coordinates by solving
    :math:`L \vec{x} = \vec{d}`, where :math:`L` is the lattice column vector matrix,
    and then :math:`\vec{x}` is scaled to the smallest parallel integer vector using
    :func:`~libcasm.xtal.scale_to_int_if_possible`.

    Only the direction of `direction_cart` matters; its magnitude does not.

    To obtain conventional cell indices, pass the conventional cell lattice. There is
    no separate conventional cell code path.

    To obtain Miller-Bravais direction indices, :math:`[UVTW]`, for hexagonal or
    trigonal lattices, pass the result to
    :func:`~libcasm.xtal.miller_to_miller_bravais_direction` and then to
    :func:`~libcasm.xtal.scale_to_int`.

    Parameters
    ----------
    lattice: ~libcasm.xtal.Lattice
        The lattice that the resulting indices are relative to.
    direction_cart: array_like
        A direction in Cartesian coordinates, with shape=(3,).
    max_element: int = 10
        The maximum allowed magnitude of any resulting index. If the direction cannot
        be expressed with indices of magnitude less than or equal to `max_element`, it
        is treated as irrational and None is returned.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check whether a scaled element is an integer.

    Returns
    -------
    uvw_indices: Optional[numpy.ndarray[numpy.int64]]
        The Miller direction indices, :math:`(u, v, w)`, with shape=(3,), or None if
        the direction is irrational with respect to `lattice` (i.e. it cannot be
        expressed with indices of magnitude less than or equal to `max_element`).

    Raises
    ------
    ValueError
        If `direction_cart` does not have shape=(3,), if `max_element` is less than 1,
        or if `direction_cart` is the zero vector.
    """
    arr = np.asarray(direction_cart, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            "Error in cartesian_to_miller_direction: direction_cart must have "
            f"shape=(3,); received shape={arr.shape}"
        )
    direction_frac = np.linalg.solve(lattice.column_vector_matrix(), arr)
    return scale_to_int_if_possible(
        direction_frac, max_element=max_element, abs_tol=abs_tol
    )


def cartesian_to_miller_plane(
    lattice: _xtal.Lattice,
    plane_normal_cart: npt.ArrayLike,
    max_element: int = 10,
    abs_tol: float = libcasm.casmglobal.TOL,
) -> Optional[np.ndarray]:
    r"""Convert a Cartesian plane normal to Miller plane indices, :math:`(hkl)`

    Notes
    -----
    The Cartesian plane normal is expressed in fractional coordinates with respect to
    the reciprocal lattice by solving :math:`R \vec{x} = \vec{n}`, where :math:`R` is
    the reciprocal lattice column vector matrix, from
    :func:`~libcasm.xtal.Lattice.reciprocal`, and then :math:`\vec{x}` is scaled to
    the smallest parallel integer vector using
    :func:`~libcasm.xtal.scale_to_int_if_possible`.

    Only the direction of `plane_normal_cart` matters; its magnitude does not, so the
    factor :math:`2\pi` in CASM's reciprocal lattice has no effect on the result.

    To obtain conventional cell indices, pass the conventional cell lattice. There is
    no separate conventional cell code path.

    To obtain Miller-Bravais plane indices, :math:`(hkil)`, for hexagonal or trigonal
    lattices, pass the result to
    :func:`~libcasm.xtal.miller_to_miller_bravais_plane`.

    Parameters
    ----------
    lattice: ~libcasm.xtal.Lattice
        The lattice that the resulting indices are relative to.
    plane_normal_cart: array_like
        A plane normal in Cartesian coordinates, with shape=(3,).
    max_element: int = 10
        The maximum allowed magnitude of any resulting index. If the plane normal
        cannot be expressed with indices of magnitude less than or equal to
        `max_element`, it is treated as irrational and None is returned.
    abs_tol: float = :data:`~libcasm.casmglobal.TOL`
        The absolute tolerance used to check whether a scaled element is an integer.

    Returns
    -------
    hkl_indices: Optional[numpy.ndarray[numpy.int64]]
        The Miller plane indices, :math:`(h, k, l)`, with shape=(3,), or None if the
        plane normal is irrational with respect to `lattice` (i.e. it cannot be
        expressed with indices of magnitude less than or equal to `max_element`).

    Raises
    ------
    ValueError
        If `plane_normal_cart` does not have shape=(3,), if `max_element` is less than
        1, or if `plane_normal_cart` is the zero vector.
    """
    arr = np.asarray(plane_normal_cart, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            "Error in cartesian_to_miller_plane: plane_normal_cart must have "
            f"shape=(3,); received shape={arr.shape}"
        )
    reciprocal_column_vector_matrix = lattice.reciprocal().column_vector_matrix()
    plane_frac_recip = np.linalg.solve(reciprocal_column_vector_matrix, arr)
    return scale_to_int_if_possible(
        plane_frac_recip, max_element=max_element, abs_tol=abs_tol
    )


### Lattice classification ###

# Proper rotation angles, in degrees, of the operations generated by a
# three-fold or six-fold rotation axis (excluding the two-fold rotation, which
# is not unique to three-fold and six-fold axes)
_THREEFOLD_AND_SIXFOLD_ANGLES = [60.0, 120.0, 240.0, 300.0]


def is_hexagonal_or_trigonal(
    lattice: Optional[_xtal.Lattice] = None,
    point_group: Optional[list[_xtal.SymOp]] = None,
    angle_tol: float = 1e-3,
) -> bool:
    """Check if a lattice or point group is hexagonal or trigonal

    Notes
    -----
    A lattice belongs to the hexagonal or trigonal (rhombohedral) crystal family if
    and only if it has exactly one three-fold or six-fold rotation axis. Cubic
    lattices also have three-fold rotations, but they have four distinct three-fold
    axes, and no other crystal family has any.

    This method therefore collects the axes, up to sign, of all proper rotations by
    60, 120, 240, or 300 degrees, and returns True if and only if there is exactly one
    such axis. Improper operations (rotoinversions, mirrors) are ignored. Screw
    operations are treated as proper rotations, so a factor group may also be given.

    This is the standard test for whether Miller-Bravais (four-index) notation
    applies. See :func:`~libcasm.xtal.miller_to_miller_bravais_direction` and
    :func:`~libcasm.xtal.miller_to_miller_bravais_plane`.

    Parameters
    ----------
    lattice: Optional[~libcasm.xtal.Lattice] = None
        A lattice. If `point_group` is None, the lattice point group, from
        :func:`~libcasm.xtal.make_point_group`, is used. One of `lattice` or
        `point_group` is required.
    point_group: Optional[list[~libcasm.xtal.SymOp]] = None
        A point group, factor group, or crystal point group to check directly. If
        provided, `lattice` is ignored. This allows checking the symmetry of a
        structure or prim, which may be lower than the symmetry of its lattice.
    angle_tol: float = 1e-3
        The absolute tolerance, in degrees, used when comparing rotation angles.

    Returns
    -------
    is_hexagonal_or_trigonal: bool
        True if there is exactly one three-fold or six-fold proper rotation axis;
        otherwise False.

    Raises
    ------
    ValueError
        If both `lattice` and `point_group` are None.
    """
    if point_group is None:
        if lattice is None:
            raise ValueError(
                "Error in is_hexagonal_or_trigonal: one of `lattice` or "
                "`point_group` is required"
            )
        point_group = _xtal.make_point_group(lattice)

    axes: list[np.ndarray] = []
    for op in point_group:
        if op.op_type() not in ("rotation_or_screw", "rotation", "screw"):
            continue
        angle = op.angle()
        if not any(
            math.isclose(angle, x, abs_tol=angle_tol)
            for x in _THREEFOLD_AND_SIXFOLD_ANGLES
        ):
            continue
        axis = np.asarray(op.axis(), dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < libcasm.casmglobal.TOL:
            continue
        axis = axis / norm
        if not any(
            np.allclose(axis, x, atol=libcasm.casmglobal.TOL)
            or np.allclose(axis, -x, atol=libcasm.casmglobal.TOL)
            for x in axes
        ):
            axes.append(axis)

    return len(axes) == 1
