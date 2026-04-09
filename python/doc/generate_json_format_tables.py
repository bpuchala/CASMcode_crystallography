# Use the data in "python/doc/json_format_tables.json" to generate:
#   - Individual per-type RST pages in python/doc/reference/json_format/
#   - A toctree snippet python/doc/reference/json_format_toctree.rst (for
#     CASMcode_pydocs)
#   - An in-place update of python/doc/reference/json_format_reference.rst
#
# Per-type RST pages use BEGIN/END GENERATED TABLE markers. If a page already
# exists and contains those markers, only the table section is updated and any
# manually authored content (examples, descriptions) is preserved. If the file
# does not exist, a skeleton page is written.

import json
import os
import re


def escape_html_chars(input_string):
    """
    Escapes special HTML characters in a string.

    Args:
        input_string (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    return input_string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_attribute_row(row, section, href_transform=None):
    """
    Generates a row for an attribute in a JSON format table.

    Args:
        row (dict): A dictionary containing row information with keys
            'name', 'description', and 'format' (with 'href' and 'text').
        section (list[str]): The list to append the row to.
        href_transform (callable, optional): Function to transform href values.
            Signature: (href: str) -> str.
    """
    href = row["format"]["href"]
    if href_transform is not None:
        href = href_transform(href)

    section.append("        <tr>")
    section.append(
        f"            <td><code>{escape_html_chars(row['name'])}</code></td>"
    )
    section.append(f"            <td>{escape_html_chars(row['description'])}</td>")
    if href:
        section.append(
            f'            <td><a href="{escape_html_chars(href)}">'
            f"{escape_html_chars(row['format']['text'])}</a></td>"
        )
    else:
        section.append(
            f"            <td>{escape_html_chars(row['format']['text'])}</td>"
        )
    section.append("        </tr>")


def _generate_attributes_table(table_data, section, href_transform=None):
    """Generate an attributes table HTML and append to section list."""
    title = table_data["title"]
    col_names = table_data.get("column_names", ["Name", "Description", "Format"])

    # Table title with optional anchor
    if "anchor" in table_data:
        anchor = table_data["anchor"]
        section.append(
            f'    <dt class="casm-part">'
            f'<a id="{escape_html_chars(anchor)}"></a>'
            f"{escape_html_chars(title)}:</dt>"
        )
    else:
        section.append(f'    <dt class="casm-part">{escape_html_chars(title)}:</dt>')

    section.append('    <dd class="casm-part">')
    section.append('    <table class="casm-table">')
    section.append("        <tr>")
    for col in col_names:
        section.append(f"            <th>{escape_html_chars(col)}</th>")
    section.append("        </tr>")

    for row in table_data["rows"]:
        generate_attribute_row(row, section, href_transform=href_transform)

    section.append("    </table>")
    section.append("    </dd>")


def build_table_block(data):
    """Build the raw HTML table block for a JSON format type.

    Args:
        data (dict): Section data from json_format_tables.json.

    Returns:
        str: RST ``.. raw:: html`` block containing the attributes tables.
    """
    lines = []
    python_type = data["python_type"]

    lines.append(".. raw:: html\n")
    lines.append('    <dl class="casm-list">')

    # Python type div
    lines.append(
        '    <div style="display: flex; flex-direction: row; '
        'align-items: center; gap: 5px;">'
    )
    lines.append('        <dt class="casm-part">Python type:</dt>')
    lines.append(
        f'        <dd class="casm-part">'
        f"<code>{escape_html_chars(python_type)}</code></dd>"
    )
    lines.append("    </div>")

    for table_data in data.get("tables", []):
        _generate_attributes_table(table_data, lines)

    lines.append("    </dl>")

    return "\n".join(lines)


def write_or_update_section_page(data, out_path):
    """Write a new skeleton RST page or update the generated table in an existing one.

    If the file does not exist (or exists but lacks the BEGIN/END markers), a
    full skeleton page is written.  If the file already contains the markers,
    only the content between them is replaced, preserving any manually authored
    content outside the markers (e.g. examples, extended descriptions).

    Args:
        data (dict): Section data from json_format_tables.json.
        out_path (str): Destination file path.
    """
    begin_marker = ".. BEGIN GENERATED TABLE"
    end_marker = ".. END GENERATED TABLE"

    table_block = build_table_block(data)
    generated_section = (
        f"{begin_marker} - updated by generate_json_format_tables.py\n\n"
        f"{table_block}\n\n"
        f"{end_marker}"
    )

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = f.read()
        pattern = re.compile(
            rf"({re.escape(begin_marker)}.*?\n)(.*?)({re.escape(end_marker)})",
            re.DOTALL,
        )
        new_content, n = pattern.subn(generated_section, existing)
        if n > 0:
            with open(out_path, "w") as f:
                f.write(new_content)
            return

    # File doesn't exist or has no markers: write a full skeleton.
    anchor = data["anchor"]
    section_title = data["section"]
    python_type = data["python_type"]
    description = data.get("description", "")

    lines = []
    lines.append(f".. _{escape_html_chars(anchor)}:\n")
    lines.append(escape_html_chars(section_title))
    lines.append("=" * len(section_title) + "\n")
    lines.append(f"JSON format for :class:`~{python_type}`.\n")
    if description:
        lines.append(escape_html_chars(description) + "\n")
    lines.append(generated_section)
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "json_format_tables.json")
with open(json_path, "r") as f:
    json_data = json.load(f)

# ---------------------------------------------------------------------------
# Multi-page output: one RST page per type
# ---------------------------------------------------------------------------

out_dir = os.path.join(script_dir, "reference", "json_format")
os.makedirs(out_dir, exist_ok=True)

for section_data in json_data:
    anchor = section_data["anchor"]
    out_path = os.path.join(out_dir, f"{anchor}.rst")
    write_or_update_section_page(section_data, out_path)
    print(f"Wrote/updated {out_path}")

# ---------------------------------------------------------------------------
# Toctree snippet for CASMcode_pydocs
# ---------------------------------------------------------------------------

toctree_lines = [
    ".. toctree::",
    "   :hidden:",
    "   :caption: JSON Format Reference",
    "",
]
for section_data in json_data:
    toctree_lines.append(f"   json_format/{section_data['anchor']}")

toctree_path = os.path.join(script_dir, "reference", "json_format_toctree.rst")
with open(toctree_path, "w") as f:
    f.write("\n".join(toctree_lines) + "\n")
print(f"Wrote {toctree_path}")

# ---------------------------------------------------------------------------
# Update toctree in json_format_reference.rst
# ---------------------------------------------------------------------------

begin_marker = ".. BEGIN GENERATED TOCTREE"
end_marker = ".. END GENERATED TOCTREE"

local_toctree_lines = [
    ".. toctree::",
    "    :hidden:",
    "    :caption: JSON Format Reference",
    "",
]
for section_data in json_data:
    local_toctree_lines.append(f"    json_format/{section_data['anchor']}")
local_toctree = "\n".join(local_toctree_lines)

rst_path = os.path.join(script_dir, "reference", "json_format_reference.rst")
with open(rst_path, "r") as f:
    content = f.read()

pattern = re.compile(
    rf"({re.escape(begin_marker)}.*?\n)(.*?)({re.escape(end_marker)})",
    re.DOTALL,
)
replacement = (
    f"{begin_marker} - updated by generate_json_format_tables.py\n\n"
    f"{local_toctree}\n\n"
    f"{end_marker}"
)
new_content, n = pattern.subn(replacement, content)
if n == 0:
    raise RuntimeError(
        f"Markers '{begin_marker}' / '{end_marker}' not found in {rst_path}."
    )
with open(rst_path, "w") as f:
    f.write(new_content)
print(f"Updated toctree in {rst_path}")
