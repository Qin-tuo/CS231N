import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = {
    "assignment1": (
        "assignment1",
        [
            "01 knn.ipynb",
            "02 softmax.ipynb",
            "03 two_layer_net.ipynb",
            "04 features.ipynb",
            "05 FullyConnectedNets.ipynb",
        ],
        [
            "cs231n/classifiers/k_nearest_neighbor.py",
            "cs231n/classifiers/linear_classifier.py",
            "cs231n/classifiers/softmax.py",
            "cs231n/classifiers/fc_net.py",
            "cs231n/optim.py",
            "cs231n/solver.py",
            "cs231n/layers.py",
        ],
    ),
    "assignment2": (
        "assignment2/assignment2",
        [
            "01 BatchNormalization.ipynb",
            "02 Dropout.ipynb",
            "03 ConvolutionalNetworks.ipynb",
            "04 PyTorch.ipynb",
            "05 RNN_Captioning_pytorch.ipynb",
        ],
        [
            "cs231n/layers.py",
            "cs231n/classifiers/fc_net.py",
            "cs231n/optim.py",
            "cs231n/solver.py",
            "cs231n/classifiers/cnn.py",
            "cs231n/im2col_cython.pyx",
            "cs231n/classifiers/rnn_pytorch.py",
        ],
    ),
    "assignment3": (
        "assignment3/assignment3",
        [
            "01 Transformer_Captioning.ipynb",
            "02 Self_Supervised_Learning.ipynb",
            "03 DDPM.ipynb",
            "04 CLIP_DINO.ipynb",
        ],
        [
            "cs231n/transformer_layers.py",
            "cs231n/classifiers/transformer.py",
            "cs231n/captioning_solver_transformer.py",
            "cs231n/simclr/contrastive_loss.py",
            "cs231n/simclr/data_utils.py",
            "cs231n/simclr/utils.py",
            "cs231n/simclr/model.py",
            "cs231n/unet.py",
            "cs231n/gaussian_diffusion.py",
            "cs231n/ddpm_trainer.py",
            "cs231n/emoji_dataset.py",
            "cs231n/clip_dino.py",
        ],
    ),
}
SAVED_ARTIFACTS = (
    "cs231n/saved/best_softmax.npy",
    "cs231n/saved/best_two_layer_net.npy",
    "cs231n/saved/checkpoints/epoch_1.pkl",
)


def collector_module():
    return importlib.import_module("scripts.collect_submission")


@pytest.fixture
def tiny_repo(tmp_path):
    repo = tmp_path / "course with spaces"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    for relative_root, notebooks, code in ASSIGNMENTS.values():
        assignment = repo / relative_root
        for relative in [*code, "cs231n/helper.py", "my helpers.py"]:
            path = assignment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# student code\n", encoding="utf-8")
        for name in notebooks:
            notebook = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "id": "title",
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": f"# {name}",
                    },
                    {
                        "id": "student-code",
                        "cell_type": "code",
                        "metadata": {},
                        "source": "raise AssertionError('Do not execute submissions')",
                        "execution_count": 1,
                        "outputs": [{"output_type": "stream", "name": "stdout", "text": "saved output\n"}],
                    },
                ],
            }
            (assignment / name).write_text(json.dumps(notebook), encoding="utf-8")
        for relative in [
            "collect_submission.ipynb",
            "scratch.ipynb",
            "makepdf.py",
            "collect_submission.py",
            "cs231n/datasets/downloaded.py",
            "cs231n/checkpoints/state.py",
            "cs231n/pretrained/downloaded.py",
            "cs231n/pretrained/model.pth",
            "cs231n/saved/pretrained/model.pth",
            "cs231n/saved/.cache/model.pth",
            ".ipynb_checkpoints/01 knn-checkpoint.ipynb",
            ".venv/lib/site-packages/installed.py",
        ]:
            path = assignment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("excluded", encoding="utf-8")
        for relative in SAVED_ARTIFACTS:
            path = assignment / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"student-trained artifact\n\x00\xff")
    return repo


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_code_only_archive_contains_teaching_notebooks_and_sources(
    assignment, tiny_repo, tmp_path, monkeypatch
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    output = tmp_path / "submission files"
    result = collector.main([
        "--assignment", assignment, "--output-dir", str(output), "--code-only"
    ])
    assert result == 0
    number = assignment[-1]
    archive = output / f"a{number}_code_submission.zip"
    relative_root, notebooks, code = ASSIGNMENTS[assignment]
    saved = SAVED_ARTIFACTS if assignment in {"assignment1", "assignment2"} else ()
    with zipfile.ZipFile(archive) as submission:
        assert set(submission.namelist()) == set([
            *notebooks, *code, "cs231n/helper.py", "my helpers.py", *saved
        ])
        assert submission.read(notebooks[0]) == (tiny_repo / relative_root / notebooks[0]).read_bytes()
    assert list(output.iterdir()) == [archive]


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2"])
def test_saved_training_artifacts_are_preserved_in_submission(
    assignment, tiny_repo, tmp_path, monkeypatch
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    output = tmp_path / "submission"
    result = collector.main([
        "--assignment", assignment, "--output-dir", str(output), "--code-only"
    ])
    assert result == 0
    with zipfile.ZipFile(output / f"a{assignment[-1]}_code_submission.zip") as submission:
        for relative in SAVED_ARTIFACTS:
            assert relative in submission.namelist()
            assert submission.read(relative) == (tiny_repo / ASSIGNMENTS[assignment][0] / relative).read_bytes()


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2"])
def test_saved_directory_is_optional_before_training(
    assignment, tiny_repo, tmp_path, monkeypatch
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    shutil.rmtree(tiny_repo / ASSIGNMENTS[assignment][0] / "cs231n/saved")
    output = tmp_path / "submission"
    result = collector.main([
        "--assignment", assignment, "--output-dir", str(output), "--code-only"
    ])
    assert result == 0
    assert zipfile.is_zipfile(output / f"a{assignment[-1]}_code_submission.zip")


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_repository_file_lists_match_assignment_order(assignment):
    collector = collector_module()
    relative_root, expected_notebooks, required_code = ASSIGNMENTS[assignment]
    assignment_root = ROOT / relative_root
    notebooks, files = collector.submission_files(assignment, assignment_root)
    assert [path.name for path in notebooks] == expected_notebooks
    included = {path.relative_to(assignment_root).as_posix() for path in files}
    assert set(required_code) <= included
    assert {name for name in included if name.endswith(".ipynb")} == set(expected_notebooks)
    assert "makepdf.py" not in included


def test_missing_required_notebook_does_not_create_incomplete_archive(
    tiny_repo, tmp_path, monkeypatch, capsys
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    (tiny_repo / "assignment1/02 softmax.ipynb").unlink()
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment1", "--output-dir", str(output), "--code-only"])
    assert result != 0
    assert "02 softmax.ipynb" in capsys.readouterr().err
    assert not (output / "a1_code_submission.zip").exists()


def test_missing_required_source_does_not_create_incomplete_archive(
    tiny_repo, tmp_path, monkeypatch, capsys
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    (tiny_repo / "assignment2/assignment2/cs231n/im2col_cython.pyx").unlink()
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment2", "--output-dir", str(output), "--code-only"])
    assert result != 0
    assert "im2col_cython.pyx" in capsys.readouterr().err
    assert not (output / "a2_code_submission.zip").exists()


@pytest.mark.parametrize("dependency", ["pandoc", "xelatex"])
def test_missing_pdf_executable_names_dependency_and_keeps_zip(
    dependency, tiny_repo, tmp_path, monkeypatch, capsys
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    monkeypatch.setattr(collector.shutil, "which", lambda name: None if name == dependency else f"/bin/{name}")
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment1", "--output-dir", str(output)])
    assert result != 0
    error = capsys.readouterr().err
    assert dependency in error
    assert "Install" in error and "PATH" in error
    assert "--code-only" in error
    assert zipfile.is_zipfile(output / "a1_code_submission.zip")
    assert not (output / "a1_inline_submission.pdf").exists()


def test_missing_nbconvert_names_environment_setup_and_keeps_zip(
    tiny_repo, tmp_path, monkeypatch, capsys
):
    collector = collector_module()
    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    monkeypatch.setitem(sys.modules, "nbconvert", None)
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment1", "--output-dir", str(output)])
    assert result != 0
    error = capsys.readouterr().err
    assert "nbconvert" in error and "uv sync" in error
    assert zipfile.is_zipfile(output / "a1_code_submission.zip")


def test_pdf_export_preserves_saved_cells_in_question_order(
    tiny_repo, tmp_path, monkeypatch
):
    collector = collector_module()
    import nbconvert

    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    monkeypatch.setattr(collector.shutil, "which", lambda name: f"/bin/{name}")
    exported = []
    pdf_content = b"%PDF-1.4\nfixture export\n%%EOF\n"

    class PDFExporter:
        def from_notebook_node(self, notebook, resources):
            exported.append((notebook, resources))
            return pdf_content, resources

    monkeypatch.setattr(nbconvert, "PDFExporter", PDFExporter)
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment1", "--output-dir", str(output)])
    assert result == 0
    assert (output / "a1_inline_submission.pdf").read_bytes() == pdf_content
    notebook, resources = exported[0]
    assert [cell.source for cell in notebook.cells if cell.cell_type == "markdown"] == [
        f"# {name}" for name in ASSIGNMENTS["assignment1"][1]
    ]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 5
    assert all(cell.outputs[0].text == "saved output\n" for cell in code_cells)
    assert all(cell.execution_count == 1 for cell in code_cells)
    assert resources["metadata"]["path"] == str(tiny_repo / "assignment1")


def test_pdf_conversion_failure_reports_error_without_losing_archive(
    tiny_repo, tmp_path, monkeypatch, capsys
):
    collector = collector_module()
    import nbconvert

    monkeypatch.setattr(collector, "ROOT", tiny_repo)
    monkeypatch.setattr(collector.shutil, "which", lambda name: f"/bin/{name}")

    class PDFExporter:
        def from_notebook_node(self, notebook, resources):
            raise RuntimeError("LaTeX Error: File `tcolorbox.sty' not found")

    monkeypatch.setattr(nbconvert, "PDFExporter", PDFExporter)
    output = tmp_path / "submission"
    result = collector.main(["--assignment", "assignment1", "--output-dir", str(output)])
    assert result != 0
    assert "tcolorbox.sty" in capsys.readouterr().err
    assert zipfile.is_zipfile(output / "a1_code_submission.zip")
    assert not (output / "a1_inline_submission.pdf").exists()


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_shell_wrapper_uses_own_location_from_unrelated_directory(
    assignment, tiny_repo, tmp_path
):
    collector_module()
    (tiny_repo / "scripts").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts/collect_submission.py", tiny_repo / "scripts")
    shutil.copytree(ROOT / "cs231n_local", tiny_repo / "cs231n_local", dirs_exist_ok=True)
    relative_root = ASSIGNMENTS[assignment][0]
    wrapper = tiny_repo / relative_root / "collectSubmission.sh"
    shutil.copy2(ROOT / relative_root / "collectSubmission.sh", wrapper)
    caller = tmp_path / "caller directory"
    caller.mkdir()
    environment = dict(os.environ, PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    result = subprocess.run(
        ["bash", str(wrapper), "--output-dir", "submission files", "--code-only"],
        cwd=caller, env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert zipfile.is_zipfile(caller / "submission files" / f"a{assignment[-1]}_code_submission.zip")


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_shell_wrapper_uses_repository_python_with_clean_path(assignment):
    wrapper = ROOT / ASSIGNMENTS[assignment][0] / "collectSubmission.sh"
    result = subprocess.run(
        ["/bin/bash", str(wrapper), "--help"],
        cwd="/tmp", env={"PATH": "/usr/bin:/bin"}, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--assignment" in result.stdout
    assert "--output-dir" in result.stdout


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
@pytest.mark.parametrize("selection", ["override", "active", "path"])
def test_shell_wrapper_honors_selected_python(assignment, selection, tmp_path):
    interpreters = {}
    for name in ["override", "active", "path"]:
        interpreter = tmp_path / f"{name} environment" / "bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text(f"#!/bin/sh\nprintf '%s\\n' '{name}'\n")
        interpreter.chmod(0o755)
        interpreters[name] = interpreter
    environment = {"PATH": f"{interpreters['path'].parent}:/usr/bin:/bin"}
    if selection in {"override", "active"}:
        environment["VIRTUAL_ENV"] = str(interpreters["active"].parent.parent)
    if selection == "override":
        environment["CS231N_PYTHON"] = str(interpreters["override"])
    wrapper = ROOT / ASSIGNMENTS[assignment][0] / "collectSubmission.sh"
    result = subprocess.run(
        ["/bin/bash", str(wrapper), "--help"],
        cwd="/tmp", env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"{selection}\n"
