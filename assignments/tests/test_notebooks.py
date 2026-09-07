from pathlib import Path
import os
import subprocess
import sys
import copy

import nbformat
import pytest
from IPython.core.interactiveshell import InteractiveShell

from scripts.migrate_notebooks import migrate_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("assignment*/**/*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_is_local_and_compiles(path):
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assignment = path.relative_to(ROOT).parts[0]
    assert "from cs231n_local import setup_notebook" in code_cells[0].source
    assert f'setup_notebook("{assignment}")' in code_cells[0].source
    shell = InteractiveShell.instance()
    for cell in code_cells:
        for legacy in ("google.colab", "/content/drive", "My Drive", "FOLDERNAME", "!pip install", "%pip install", "wget "):
            assert legacy not in cell.source, (path, legacy)
        compile(shell.input_transformer_manager.transform_cell(cell.source), str(path), "exec")


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2", "assignment3"])
def test_notebook_bootstrap_from_its_directory_and_external_cwd(assignment, tmp_path):
    path = next(ROOT.glob(f"{assignment}/**/01 *.ipynb"))
    source = next(c.source for c in nbformat.read(path, 4).cells if c.cell_type == "code")
    for cwd in (path.parent, tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", source + "\nimport cs231n\nassert str(CONFIG.assignment_root) in cs231n.__file__"],
            cwd=cwd,
            env={**os.environ, "CS231N_ROOT": str(ROOT), "CS231N_DEVICE": "cpu"},
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr


def test_migration_preserves_exercises_metadata_and_outputs_and_is_idempotent(tmp_path):
    path = tmp_path / "01 knn.ipynb"
    original = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell("from google.colab import drive"),
        nbformat.v4.new_markdown_cell("Question and answer remain here."),
        nbformat.v4.new_code_cell("# TODO: exercise\nanswer = 42", execution_count=7,
                                 outputs=[nbformat.v4.new_output("stream", name="stdout", text="saved answer")]),
    ], metadata={"custom": "preserve"})
    nbformat.write(original, path)
    untouched = copy.deepcopy(original)
    assert migrate_notebook(path, "assignment1")
    migrated = nbformat.read(path, 4)
    assert migrated.cells[1:] == untouched.cells[1:]
    assert migrated.metadata == untouched.metadata
    first_bytes = path.read_bytes()
    assert not migrate_notebook(path, "assignment1")
    assert path.read_bytes() == first_bytes
