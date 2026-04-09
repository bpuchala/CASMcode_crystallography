.. _xtal_DoFSetBasis:

DoFSetBasis
===========

JSON format for :class:`~libcasm.xtal.DoFSetBasis`.

The basis for a set of degrees of freedom (DoF). In Prim JSON, this appears as a dict mapping DoF type names to their specifications.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.DoFSetBasis</code></dd>
    </div>
    <dt class="casm-part">DoFSetBasis specification attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>axis_names</code></td>
            <td>Names for the individual DoF basis vectors. Required if basis is present. Size must match the number of rows in basis.</td>
            <td>array of string, shape=(dim,)</td>
        </tr>
        <tr>
            <td><code>basis</code></td>
            <td>Row-vector matrix defining the user-specified basis in terms of the standard basis (optional, default is the standard basis).</td>
            <td>2d array of number, shape=(dim, standard_dim)</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE


.. rubric:: Examples

The following JSON files provide example DoF specifications that can be read using
:func:`~libcasm.xtal.DoFSetBasis.from_dict`.

- Atomic displacement DoF (3D: standard basis): :download:`xtal_dofsetbasis_disp.json <json/xtal_dofsetbasis_disp.json>`
- Atomic displacement DoF (2D: :math:`dx, dy` only): :download:`xtal_dofsetbasis_disp_xy.json <json/xtal_dofsetbasis_disp_xy.json>`
- Hencky strain global DoF (6D: standard basis): :download:`xtal_dofsetbasis_Hstrain.json <json/xtal_dofsetbasis_Hstrain.json>`
- Hencky strain global DoF (3D: :math:`E_{xx}, E_{yy}, E_{zz}` only): :download:`xtal_dofsetbasis_Hstrain_diagonal.json <json/xtal_dofsetbasis_Hstrain_diagonal.json>`

Example: reading a DoFSetBasis from a JSON file:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal

    # Load DoFSetBasis from JSON file
    # The JSON contains a dict mapping dof type name to specification
    with open("xtal_dofsetbasis_disp.json", "r") as f:
        data = json.load(f)

    # Construct the DoFSetBasis (reads the first entry in the dict)
    dof = xtal.DoFSetBasis.from_dict(data)

    # Access the DoF type name, axis names, and basis matrix
    dofname = dof.dofname()       # "disp"
    axis_names = dof.axis_names() # ["dx", "dy", "dz"]
    basis = dof.basis()           # numpy array, shape=(3, 3)

    # Convert back to dict (updates an existing dict in-place)
    data2 = {}
    dof.to_dict(data2)

