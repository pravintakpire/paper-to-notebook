"""Construct .ipynb notebooks from a list of cell dicts using nbformat."""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def build_notebook(cells_json: list) -> nbformat.NotebookNode:
    """Convert a list of cell dicts into a proper .ipynb notebook.

    Args:
        cells_json: List of {"cell_type": "code"|"markdown", "source": "..."}

    Returns:
        A NotebookNode ready to be written to disk.
    """
    nb = new_notebook()

    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.9",
    }

    for cell_data in cells_json:
        cell_type = cell_data["cell_type"]
        source = cell_data["source"]

        if cell_type == "markdown":
            nb.cells.append(new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(new_code_cell(source))
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    return nb


def save_notebook(nb: nbformat.NotebookNode, output_path: str) -> None:
    """Write a notebook to disk as .ipynb."""
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
