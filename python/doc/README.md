# Sphinx documentation source

This directory contains the Sphinx documentation source for `libcasm-xtal`.

## Building the docs

    sphinx-build -b html python/doc $LIBCASM_PYDOCS/xtal/2.0

## Generating JSON format reference pages

The JSON format reference pages are generated from `json_format_tables.json`:

    python python/doc/generate_json_format_tables.py

This script:
- Generates one RST page per type in `reference/json_format/`
- Writes a toctree snippet `reference/json_format_toctree.rst` for use in CASMcode_pydocs
- Updates the toctree in `reference/json_format_reference.rst`

To add or update JSON format documentation, edit `json_format_tables.json` and re-run the script.
