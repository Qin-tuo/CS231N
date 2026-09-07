"""Repository paths and runtime device selection shared by all notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class NotebookConfig:
    """Resolved paths and device for one assignment notebook."""

    repo_root: Path
    assignment_root: Path
    data_root: Path
    device: str

    def require_data(self, relative_path: str, dataset: str) -> Path:
        """Locate a prepared resource and explain how to obtain missing data."""
        path = self.data_root / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Missing data: {path}\nFrom {self.repo_root}, run:\n"
                f"uv run python scripts/prepare_data.py --assignment "
                f"{self.assignment_root.name} --dataset {dataset}"
            )
        return path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find the repository root from a notebook or module working directory."""

    start_path = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "assignment1").is_dir():
            return candidate
    raise RuntimeError(f"Cannot find CS231n repository from {start_path}")


def assignment_dir(assignment: str, repo_root: Path | str | None = None) -> Path:
    """Return an assignment directory and validate its name."""

    if assignment not in {"assignment1", "assignment2", "assignment3"}:
        raise ValueError(f"Unknown assignment: {assignment}")
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else find_repo_root()
    path = root / assignment
    if (path / assignment).is_dir():
        path = path / assignment
    if not path.is_dir():
        raise ValueError(f"Unknown assignment: {assignment}")
    return path


def data_dir(
    assignment: str,
    repo_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> Path:
    """Return the dataset directory for an assignment.

    ``data_root`` is useful for CLI callers; the environment variable provides
    the same override to notebooks and shell commands.
    """

    assignment_root = assignment_dir(assignment, repo_root)
    external = data_root or os.environ.get("CS231N_DATA_ROOT")
    if external:
        return Path(external).expanduser().resolve() / assignment
    return assignment_root / "cs231n" / "datasets"


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for device selection; run `uv sync` first"
        ) from exc


def resolve_device(requested: str | None = None) -> str:
    """Resolve ``auto``, CPU, MPS, or CUDA into a usable device string."""

    value = (requested or os.environ.get("CS231N_DEVICE") or "auto").strip().lower()
    valid = {"auto", "cpu", "mps", "cuda"}
    if value not in valid and not value.startswith("cuda:"):
        raise ValueError(
            f"Unsupported CS231N_DEVICE={value!r}; use auto, cpu, mps, cuda, or cuda:N"
        )

    if value == "cpu":
        return value

    torch = _load_torch()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if value == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if value == "mps" and mps_available:
        return value
    if (value == "cuda" or value.startswith("cuda:")) and cuda_available:
        if value.startswith("cuda:"):
            index = value.partition(":")[2]
            if not index.isdigit() or int(index) >= torch.cuda.device_count():
                raise RuntimeError(f"CS231N_DEVICE={value} is unavailable")
        return value
    raise RuntimeError(f"CS231N_DEVICE={value} is unavailable")


def setup_notebook(assignment: str, cwd: Path | str | None = None) -> NotebookConfig:
    """Resolve paths, expose the assignment package, and select a device."""

    configured_root = os.environ.get("CS231N_ROOT")
    root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else find_repo_root(cwd)
    )
    assignment_root = assignment_dir(assignment, root)
    loaded_package = sys.modules.get("cs231n")
    loaded_file = getattr(loaded_package, "__file__", None)
    if loaded_file and not Path(loaded_file).resolve().is_relative_to(assignment_root):
        raise RuntimeError(
            f"A different assignment's cs231n package is already loaded: {loaded_file}. "
            "Restart the notebook kernel before switching assignments."
        )
    assignment_path = str(assignment_root)
    if assignment_path in sys.path:
        sys.path.remove(assignment_path)
    sys.path.insert(0, assignment_path)
    return NotebookConfig(
        repo_root=root,
        assignment_root=assignment_root,
        data_root=data_dir(assignment, root),
        device=resolve_device("cpu" if assignment == "assignment1" else None),
    )
