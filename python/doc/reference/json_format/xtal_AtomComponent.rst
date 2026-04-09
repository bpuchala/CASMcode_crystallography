.. _xtal_AtomComponent:

AtomComponent
=============

JSON format for :class:`~libcasm.xtal.AtomComponent`.

An atomic component of a molecular occupant.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.AtomComponent</code></dd>
    </div>
    <dt class="casm-part">AtomComponent attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>coordinate</code></td>
            <td>Position of the atom, relative to the basis site. Coordinate mode is the same as the enclosing Prim or Structure.</td>
            <td>array of number, shape=(3,)</td>
        </tr>
        <tr>
            <td><code>name</code></td>
            <td>Atom type name.</td>
            <td>string</td>
        </tr>
        <tr>
            <td><code>properties</code></td>
            <td>Fixed atomic properties, such as magnetic moment or selective dynamics flags (optional).</td>
            <td><a href="#xtal-atomcomponent-speciesproperty">dict of SpeciesProperty</a></td>
        </tr>
    </table>
    </dd>
    <dt class="casm-part"><a id="xtal-atomcomponent-speciesproperty"></a>SpeciesProperty attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>value</code></td>
            <td>Species attribute value. Dimension must match the standard dimension of the CASM property type.</td>
            <td>array of number</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE


.. rubric:: Examples

The following JSON files provide example atom components that can be read using
:func:`~libcasm.xtal.AtomComponent.from_dict`.

- Simple aluminum atom (no properties): :download:`xtal_atomcomponent_atom.json <json/xtal_atomcomponent_atom.json>`
- Atom with collinear magnetic spin (Cmagspin): :download:`xtal_atomcomponent_magspin.json <json/xtal_atomcomponent_magspin.json>`

Example: reading an AtomComponent from a JSON file:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal

    # Load atom component from JSON file
    with open("xtal_atomcomponent_atom.json", "r") as f:
        data = json.load(f)

    # Construct the AtomComponent
    atom = xtal.AtomComponent.from_dict(data)

    # Access the atom name and coordinate
    name = atom.name()
    coord = atom.coordinate()

    # Convert back to dict
    data2 = atom.to_dict()

