from pathlib import Path

import pytest

from types import SimpleNamespace

from cs231n_local import config
from cs231n_local.config import assignment_dir, data_dir, find_repo_root, resolve_device, setup_notebook


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2", "assignment3"])
def test_real_assignment_root_contains_course_package(assignment):
    assert (assignment_dir(assignment, ROOT) / "cs231n" / "__init__.py").is_file()
    assert data_dir(assignment, ROOT).is_dir()


@pytest.mark.parametrize("name", ["../", "tests", "/tmp"])
def test_assignment_name_cannot_select_arbitrary_directory(name):
    with pytest.raises(ValueError, match="Unknown assignment"):
        assignment_dir(name, ROOT)


def test_setup_honors_root_override_from_outside_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("CS231N_ROOT", str(ROOT))
    monkeypatch.setenv("CS231N_DEVICE", "cpu")
    result = setup_notebook("assignment2", cwd=tmp_path)
    assert result.assignment_root == ROOT / "assignment2" / "assignment2"


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    backend = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 2),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setattr(config, "_load_torch", lambda: backend)
    assert resolve_device("auto") == expected


def test_cuda_device_index_and_unavailable_backends(monkeypatch):
    backend = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(config, "_load_torch", lambda: backend)
    assert resolve_device("cuda:1") == "cuda:1"
    for name in ["cuda:2", "cuda:-1", "cuda:x", "mps"]:
        with pytest.raises((RuntimeError, ValueError), match="CS231N_DEVICE"):
            resolve_device(name)


def test_missing_data_error_gives_preparation_command(monkeypatch, tmp_path):
    monkeypatch.setenv("CS231N_DEVICE", "cpu")
    monkeypatch.setenv("CS231N_DATA_ROOT", str(tmp_path))
    result = setup_notebook("assignment2", cwd=ROOT)
    with pytest.raises(FileNotFoundError, match="prepare_data.py --assignment assignment2 --dataset coco"):
        result.require_data("coco_captioning", "coco")


def test_find_repo_root_from_nested_path(tmp_path):
    repo = tmp_path / "repo"
    (repo / "assignment1").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = repo / "assignment1" / "cs231n"
    nested.mkdir()
    assert find_repo_root(nested) == repo


def test_assignment_dir_rejects_unknown_assignment(tmp_path):
    repo = tmp_path / "repo"
    (repo / "assignment1").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown assignment"):
        assignment_dir("assignment9", repo)


def test_external_data_root_is_assignment_scoped(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "assignment2" / "assignment2").mkdir(parents=True)
    external = tmp_path / "datasets"
    monkeypatch.setenv("CS231N_DATA_ROOT", str(external))
    assert data_dir("assignment2", repo) == external / "assignment2"


def test_explicit_cpu_does_not_require_torch(monkeypatch):
    monkeypatch.setenv("CS231N_DEVICE", "cpu")
    assert resolve_device() == "cpu"
