.. _xtal_SymInfo:

SymInfo
=======

JSON format for :class:`~libcasm.xtal.SymInfo`.

Symmetry operation type, axis, invariant point, and other descriptive information.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.SymInfo</code></dd>
    </div>
    <dt class="casm-part">SymInfo attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>angle</code></td>
            <td>Rotation angle in degrees.</td>
            <td>number</td>
        </tr>
        <tr>
            <td><code>brief</code></td>
            <td>Brief string descriptions of the symmetry operation in Cartesian (CART) and fractional (FRAC) coordinates.</td>
            <td>dict of string</td>
        </tr>
        <tr>
            <td><code>conjugacy_class</code></td>
            <td>Index of the conjugacy class containing this operation.</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>invariant_point</code></td>
            <td>A Cartesian coordinate that is invariant under the operation (does not exist for identity).</td>
            <td>dict</td>
        </tr>
        <tr>
            <td><code>inverse_operation</code></td>
            <td>Index in the group of the inverse operation.</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>mirror_normal</code></td>
            <td>Unit normal vector of mirror and glide planes (Cartesian).</td>
            <td>dict</td>
        </tr>
        <tr>
            <td><code>rotation_angle</code></td>
            <td>Rotation angle of the operation (degrees).</td>
            <td>number</td>
        </tr>
        <tr>
            <td><code>rotation_axis</code></td>
            <td>Rotation axis (Cartesian, normalized to length 1).</td>
            <td>dict</td>
        </tr>
        <tr>
            <td><code>shift</code></td>
            <td>Screw or glide shift vector (Cartesian).</td>
            <td>dict</td>
        </tr>
        <tr>
            <td><code>type</code></td>
            <td>Symmetry operation type: "identity", "mirror", "glide", "rotation", "screw", "inversion", "rotoinversion", or "invalid".</td>
            <td>string</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE

