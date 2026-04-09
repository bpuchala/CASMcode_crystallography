.. _xtal_Occupant:

Occupant
========

JSON format for :class:`~libcasm.xtal.Occupant`.

A site occupant, which may be a vacancy, single atom, or molecule.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.Occupant</code></dd>
    </div>
    <dt class="casm-part">Occupant attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>atoms</code></td>
            <td>List of atoms that comprise a molecule (optional, may be excluded for single-atom occupants).</td>
            <td><a href="xtal_AtomComponent.html">array of AtomComponent</a></td>
        </tr>
        <tr>
            <td><code>name</code></td>
            <td>Chemical name of the species (optional, used to override the key in the species dict).</td>
            <td>string</td>
        </tr>
        <tr>
            <td><code>properties</code></td>
            <td>Fixed molecular properties, such as magnetic spin or selective dynamics flags (optional).</td>
            <td><a href="#xtal-occupant-speciesproperty">dict of SpeciesProperty</a></td>
        </tr>
    </table>
    </dd>
    <dt class="casm-part"><a id="xtal-occupant-speciesproperty"></a>SpeciesProperty attributes:</dt>
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

The following JSON files provide example occupants that can be read using
:func:`~libcasm.xtal.Occupant.from_dict`.

- Simple aluminum atom: :download:`xtal_occupant_atom.json <json/xtal_occupant_atom.json>`
- Atom with discrete collinear magnetic spin (Cmagspin): :download:`xtal_occupant_magspin.json <json/xtal_occupant_magspin.json>`
- Dimer, oriented along x: :download:`xtal_occupant_dimer_x.json <json/xtal_occupant_dimer_x.json>`

Example: reading an Occupant from a JSON file:

.. code-block:: Python

    import json
    import libcasm.xtal as xtal

    # Load occupant from JSON file
    with open("xtal_occupant_atom.json", "r") as f:
        data = json.load(f)

    # Construct the Occupant
    occupant = xtal.Occupant.from_dict(data)

    # Access the occupant name
    name = occupant.name()

    # Access atom components
    atoms = occupant.atoms()

    # Convert back to dict
    data2 = occupant.to_dict()

