.. _xtal_Structure:

Structure
=========

JSON format for :class:`~libcasm.xtal.Structure`.

A crystal structure of atomic or molecular occupants with associated properties.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.Structure</code></dd>
    </div>
    <dt class="casm-part">Structure attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>atom_coords</code></td>
            <td>Atom coordinates.</td>
            <td>2d array of number, shape=(n_atoms, 3)</td>
        </tr>
        <tr>
            <td><code>atom_type</code></td>
            <td>Atom type names.</td>
            <td>array of string, shape=(n_atoms,)</td>
        </tr>
        <tr>
            <td><code>atom_properties</code></td>
            <td>Continuous properties associated with individual atoms (optional).</td>
            <td><a href="#xtal-structure-property">dict of Property</a></td>
        </tr>
        <tr>
            <td><code>coordinate_mode</code></td>
            <td>Coordinate type for atom and molecule coordinates. One of: "Fractional", "Direct", or "Cartesian".</td>
            <td>string</td>
        </tr>
        <tr>
            <td><code>lattice_vectors</code></td>
            <td>Lattice vectors (as row vectors), in Angstroms.</td>
            <td>2d array of number, shape=(3,3)</td>
        </tr>
        <tr>
            <td><code>mol_coords</code></td>
            <td>Molecule coordinates (optional).</td>
            <td>2d array of number, shape=(n_molecules, 3)</td>
        </tr>
        <tr>
            <td><code>mol_type</code></td>
            <td>Molecule type names (optional).</td>
            <td>array of string, shape=(n_molecules,)</td>
        </tr>
        <tr>
            <td><code>mol_properties</code></td>
            <td>Continuous properties associated with individual molecules (optional).</td>
            <td><a href="#xtal-structure-property">dict of Property</a></td>
        </tr>
        <tr>
            <td><code>global_properties</code></td>
            <td>Continuous properties associated with the entire crystal (optional).</td>
            <td><a href="#xtal-structure-property">dict of Property</a></td>
        </tr>
    </table>
    </dd>
    <dt class="casm-part"><a id="xtal-structure-property"></a>Property attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>value</code></td>
            <td>Property value. For atom and molecule properties, an array of vectors, one per atom or molecule. For global properties, a scalar or vector. Dimensions must match the standard dimension of the CASM property type.</td>
            <td>number or array of number</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE

