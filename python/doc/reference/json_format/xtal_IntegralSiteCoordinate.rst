.. _xtal_IntegralSiteCoordinate:

IntegralSiteCoordinate
======================

JSON format for :class:`~libcasm.xtal.IntegralSiteCoordinate`.

An integral site coordinate [b, i, j, k], where b is the sublattice index and i, j, k are unit cell indices. Serialized as a list of 4 integers.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.IntegralSiteCoordinate</code></dd>
    </div>
    <dt class="casm-part">IntegralSiteCoordinate list elements:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Index</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>[0]</code></td>
            <td>Sublattice index (b).</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>[1]</code></td>
            <td>Unit cell index (i).</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>[2]</code></td>
            <td>Unit cell index (j).</td>
            <td>int</td>
        </tr>
        <tr>
            <td><code>[3]</code></td>
            <td>Unit cell index (k).</td>
            <td>int</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE

