.. _xtal_Lattice:

Lattice
=======

JSON format for :class:`~libcasm.xtal.Lattice`.

A crystal lattice, specified by three lattice vectors.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.Lattice</code></dd>
    </div>
    <dt class="casm-part">Lattice attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>lattice_vectors</code></td>
            <td>Lattice vectors (as row vectors), in Angstroms.</td>
            <td>2d array of number, shape=(3,3)</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE


.. rubric:: Examples

The following JSON files provide example lattices that can be read using
:func:`~libcasm.xtal.Lattice.from_dict`.

- Simple cubic (a = 1.0 Å): :download:`xtal_lattice_simple_cubic.json <json/xtal_lattice_simple_cubic.json>`
- FCC Al (a = 4.046 Å): :download:`xtal_lattice_fcc.json <json/xtal_lattice_fcc.json>`
- BCC Fe (a = 2.87 Å): :download:`xtal_lattice_bcc.json <json/xtal_lattice_bcc.json>`
- HCP Mg (a = 3.2094 Å, c = 5.2105 Å): :download:`xtal_lattice_hcp.json <json/xtal_lattice_hcp.json>`

Example: reading a Lattice from a JSON file:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal

    # Load lattice from JSON file
    with open("xtal_lattice_fcc.json", "r") as f:
        data = json.load(f)

    # Construct the Lattice
    lattice = xtal.Lattice.from_dict(data)

    print(lattice)
    # {
    #     "lattice_vectors" : [
    #         [ 0.000, 2.023, 2.023],
    #         [ 2.023, 0.000, 2.023],
    #         [ 2.023, 2.023, 0.000]
    #     ]
    # }

    # The lattice vectors are stored as columns internally
    L = lattice.column_vector_matrix()  # shape (3, 3)

    # Convert back to dict (lattice_vectors are rows in JSON)
    data2 = lattice.to_dict()

