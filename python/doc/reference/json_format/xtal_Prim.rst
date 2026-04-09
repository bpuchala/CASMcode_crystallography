.. _xtal_Prim:

Prim
====

JSON format for :class:`~libcasm.xtal.Prim`.

Primitive crystal structure and allowed degrees of freedom (DoF).

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.Prim</code></dd>
    </div>
    <dt class="casm-part">Prim attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>basis</code></td>
            <td>Basis site descriptions.</td>
            <td><a href="#xtal-prim-site">array of Site</a></td>
        </tr>
        <tr>
            <td><code>coordinate_mode</code></td>
            <td>Basis site coordinate type. One of: "Fractional", "Direct", or "Cartesian".</td>
            <td>string</td>
        </tr>
        <tr>
            <td><code>description</code></td>
            <td>Project description (optional, not used by CASM).</td>
            <td>string</td>
        </tr>
        <tr>
            <td><code>dofs</code></td>
            <td>Continuous global DoF (optional).</td>
            <td><a href="xtal_DoFSetBasis.html">dict of DoFSetBasis</a></td>
        </tr>
        <tr>
            <td><code>lattice_vectors</code></td>
            <td>Lattice vectors (as row vectors), in Angstroms.</td>
            <td>2d array of number, shape=(3,3)</td>
        </tr>
        <tr>
            <td><code>species</code></td>
            <td>Fixed atom properties and molecule definitions (optional).</td>
            <td><a href="xtal_Occupant.html">dict of Occupant</a></td>
        </tr>
        <tr>
            <td><code>title</code></td>
            <td>Project title. Must consist of alphanumeric characters and underscores only. The first character may not be a number.</td>
            <td>string</td>
        </tr>
    </table>
    </dd>
    <dt class="casm-part"><a id="xtal-prim-site"></a>Site attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>coordinate</code></td>
            <td>Site coordinate with units as specified by coordinate_mode.</td>
            <td>array of number, shape=(3,)</td>
        </tr>
        <tr>
            <td><code>dofs</code></td>
            <td>Continuous site DoF (optional).</td>
            <td><a href="xtal_DoFSetBasis.html">dict of DoFSetBasis</a></td>
        </tr>
        <tr>
            <td><code>label</code></td>
            <td>An integer, &gt;= 0, that distinguishes otherwise identical sites (optional).</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>occupants</code></td>
            <td>Allowed occupant species at this site (optional, default=["UNKNOWN"]).</td>
            <td>array of string</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE


.. rubric:: Examples

The following JSON files provide example prim structures that can be read using
:func:`~libcasm.xtal.Prim.from_dict`.

- Simple cubic, binary occupation (A/B): :download:`simple_cubic_binary.json <../../examples/prim/json/simple_cubic_binary.json>`
- HCP ZrO with vacancies: :download:`ZrO_prim.json <../../examples/prim/json/ZrO_prim.json>`
- Simple cubic with atomic displacement DoF: :download:`simple_cubic_disp.json <../../examples/prim/json/simple_cubic_disp.json>`
- Simple cubic with Hencky strain global DoF: :download:`simple_cubic_Hstrain.json <../../examples/prim/json/simple_cubic_Hstrain.json>`
- Simple cubic with collinear magnetic spin (Ising): :download:`simple_cubic_ising.json <../../examples/prim/json/simple_cubic_ising.json>`

Example: reading a Prim from a JSON file:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal

    # Load prim from JSON file
    with open("simple_cubic_binary.json", "r") as f:
        data = json.load(f)

    # Construct the Prim
    prim = xtal.Prim.from_dict(data)

    # Access the lattice
    lattice = prim.lattice()

    # Access occupant DoF: list of lists of allowed occupant names per site
    occ_dof = prim.occ_dof()
    # e.g. [["A", "B"]] for a single-site binary prim

    # Convert back to dict
    data2 = prim.to_dict()

