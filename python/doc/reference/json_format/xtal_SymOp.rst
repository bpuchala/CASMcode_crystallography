.. _xtal_SymOp:

SymOp
=====

JSON format for :class:`~libcasm.xtal.SymOp`.

A symmetry operation, represented as a transformation matrix, translation vector, and time reversal flag.

.. BEGIN GENERATED TABLE - updated by generate_json_format_tables.py

.. raw:: html

    <dl class="casm-list">
    <div style="display: flex; flex-direction: row; align-items: center; gap: 5px;">
        <dt class="casm-part">Python type:</dt>
        <dd class="casm-part"><code>libcasm.xtal.SymOp</code></dd>
    </div>
    <dt class="casm-part">SymOp attributes:</dt>
    <dd class="casm-part">
    <table class="casm-table">
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Format</th>
        </tr>
        <tr>
            <td><code>matrix</code></td>
            <td>Coordinate transformation matrix.</td>
            <td>2d array of number, shape=(3,3)</td>
        </tr>
        <tr>
            <td><code>tau</code></td>
            <td>Coordinate translation vector.</td>
            <td>array of number, shape=(3,)</td>
        </tr>
        <tr>
            <td><code>time_reversal</code></td>
            <td>Whether the symmetry operation includes time reversal.</td>
            <td>bool</td>
        </tr>
    </table>
    </dd>
    </dl>

.. END GENERATED TABLE

