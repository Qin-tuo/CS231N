# CS231n Local and Cloud GPU Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all three CS231n assignments runnable from a local Jupyter environment while keeping a single, configurable path and device workflow for cloud GPU runs.

**Architecture:** Add a small root-level `cs231n_local` package that discovers the repository, resolves assignment/data directories, and selects `cuda`, `mps`, or `cpu`. Keep each assignment's existing `cs231n` package intact. Replace Colab-only notebook setup cells through a deterministic migration script and validate the resulting notebooks statically and with lightweight imports.

**Tech Stack:** Python 3.10-3.13, `uv`, pathlib, pytest, Jupyter, NumPy, SciPy, Matplotlib, Pillow, h5py, PyTorch, torchvision, nbformat, nbconvert, shell download scripts.

---

## Resume Status (2026-09-07)

The steps below are the original plan. This checklist records the resumed implementation; no Git commits were requested or created.

- [x] Task 1: Configuration, nested assignment roots, device selection, and regression tests.
- [x] Task 2: Python 3.12 environment, editable helper package, dependencies, and device checks.
- [x] Task 3: Dataset preparation, staged extraction, complete-data checks, and portable shell wrappers.
- [x] Task 4: All 17 notebooks migrated; startup/import checks and content-preservation checks passed.
- [x] Task 5: Local collectors verified, including original Assignment 1/2 saved training artifacts; missing PDF tooling handled.
- [x] Task 6: Final integrated tests, CIFAR loading checks, and verification record in `docs/LOCAL_WORKFLOW_STATUS.md`.

The supported Python range is 3.11-3.13, retaining the existing project's 3.11 minimum. Work continues in the original directory on `local-runtime`; the source before this session is backed up at `/tmp/cs231n-local-before-20260907.tar.gz`.

### Task 1: Add repository and device configuration

**Files:**
- Create: `cs231n_local/__init__.py`
- Create: `cs231n_local/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for root, assignment, data, and device resolution**

```python
import pytest
import torch

from cs231n_local.config import data_dir, find_repo_root, resolve_device


def test_find_repo_root_from_nested_path(tmp_path):
    repo = tmp_path / "repo"
    (repo / "assignment1").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    nested = repo / "assignment1" / "cs231n"
    nested.mkdir()
    assert find_repo_root(nested) == repo


def test_external_data_root_is_assignment_scoped(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "assignment2" / "assignment2").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    external = tmp_path / "datasets"
    monkeypatch.setenv("CS231N_DATA_ROOT", str(external))
    assert data_dir("assignment2", repo) == external / "assignment2"


def test_device_request_rejects_unavailable_backend(monkeypatch):
    monkeypatch.setenv("CS231N_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CS231N_DEVICE=mps"):
        resolve_device()
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_config.py`

Expected: FAIL because `cs231n_local.config` does not exist.

- [ ] **Step 3: Implement the minimal configuration API**

Implement these functions in `cs231n_local/config.py`:

```python
@dataclass(frozen=True)
class NotebookConfig:
    repo_root: Path
    assignment_root: Path
    data_root: Path
    device: str


def find_repo_root(start: Path | str | None = None) -> Path:
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "assignment1").is_dir():
            return candidate
    raise RuntimeError(f"Cannot find CS231n repository from {start_path}")


def assignment_dir(assignment: str, repo_root: Path | str | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    path = root / assignment
    if not path.is_dir():
        raise ValueError(f"Unknown assignment: {assignment}")
    return path


def data_dir(
    assignment: str,
    repo_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> Path:
    external = data_root or os.environ.get("CS231N_DATA_ROOT")
    if external:
        return Path(external).expanduser().resolve() / assignment
    return assignment_dir(assignment, repo_root) / "cs231n" / "datasets"


def resolve_device(requested: str | None = None) -> str:
    value = (requested or os.environ.get("CS231N_DEVICE") or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cpu":
        return value
    if value == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return value
    if (value == "cuda" or value.startswith("cuda:")) and torch.cuda.is_available():
        return value
    raise RuntimeError(f"CS231N_DEVICE={value} is unavailable")


def setup_notebook(assignment: str, cwd: Path | str | None = None) -> NotebookConfig:
    root = Path(os.environ["CS231N_ROOT"]).expanduser().resolve() if os.environ.get("CS231N_ROOT") else find_repo_root(cwd)
    assignment_root = assignment_dir(assignment, root)
    if str(assignment_root) not in sys.path:
        sys.path.insert(0, str(assignment_root))
    return NotebookConfig(root, assignment_root, data_dir(assignment, root), resolve_device())
```

Import `dataclass`, `os`, `sys`, `Path`, and `torch` at the top of the module. `find_repo_root` walks the current path and parents until it finds both `pyproject.toml` and `assignment1`; `CS231N_ROOT` overrides that search. `data_dir` uses an explicit `data_root` argument or `CS231N_DATA_ROOT / assignment` and otherwise falls back to `<assignment>/cs231n/datasets`. `resolve_device` accepts `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`, checks backend availability, and raises `RuntimeError` containing the requested value when an explicit backend is unavailable.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_config.py`

Expected: PASS for root discovery, external data routing, automatic device selection, explicit CPU selection, and unavailable-device errors.

- [ ] **Step 5: Record the change**

Run: `git add cs231n_local tests/test_config.py && git commit -m "feat: add local repository and device configuration"`

If the managed workspace rejects git index writes, retain the files and report the permission limitation.

### Task 2: Define the local environment and user commands

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `README.md`
- Create: `.env.example`
- Create: `scripts/check_environment.py`

- [ ] **Step 1: Add supported Python range and dependencies**

Set `requires-python = ">=3.10,<3.14"` and add runtime dependencies for the notebooks: `numpy`, `scipy`, `matplotlib`, `pillow`, `imageio`, `h5py`, `scikit-learn`, `pandas`, `requests`, `tqdm`, `cython`, `jupyterlab`, `nbconvert`, `pytest`, `torch`, and `torchvision`. Keep CLIP and profiling dependencies in an optional `assignment3` extra so Assignment 1 does not require them.

- [ ] **Step 2: Regenerate the lockfile and add environment checks**

Run: `uv lock`

Implement `scripts/check_environment.py` to import NumPy, SciPy, Matplotlib, PIL, h5py, torch, and torchvision, print their versions, and print `resolve_device()`; exit nonzero with the missing package name if an import fails.

- [ ] **Step 3: Document local and cloud GPU setup**

Document:

```bash
uv sync
uv run python scripts/check_environment.py
uv run jupyter lab
CS231N_DEVICE=cuda uv run jupyter lab
```

Explain that `CS231N_ROOT`, `CS231N_DATA_ROOT`, and `CS231N_DEVICE` are optional, and show a cloud GPU example that points `CS231N_DATA_ROOT` at persistent storage.

- [ ] **Step 4: Verify the environment command without datasets**

Run: `uv run python scripts/check_environment.py`

Expected: dependency versions and one of `cuda`, `mps`, or `cpu` are printed; missing optional CLIP packages do not prevent Assignment 1/2 checks.

### Task 3: Make dataset preparation local and idempotent

**Files:**
- Create: `scripts/prepare_data.py`
- Modify: `assignment1/cs231n/datasets/get_datasets.sh`
- Modify: `assignment2/assignment2/cs231n/datasets/get_datasets.sh`
- Modify: `assignment3/assignment3/cs231n/datasets/get_datasets.sh`
- Modify: `assignment2/assignment2/cs231n/datasets/get_coco_dataset.sh`
- Modify: `assignment3/assignment3/cs231n/datasets/get_coco_dataset.sh`

- [ ] **Step 1: Add a dry-run test for dataset destinations**

Create `tests/test_prepare_data.py` that invokes `python scripts/prepare_data.py --assignment assignment1 --dry-run --data-root <tmp>` and asserts the output contains `<tmp>/assignment1` and the CIFAR-10 archive URL, without creating files.

- [ ] **Step 2: Implement the data preparation CLI**

Implement `--assignment {assignment1,assignment2,assignment3}`, `--dataset {cifar10,imagenet_val,coco,all}`, `--data-root`, and `--dry-run`. Pass the CLI `--data-root` directly to `cs231n_local.config.data_dir(..., data_root=args.data_root)`; use `urllib`/`tarfile` or invoke the existing scripts with an explicit working directory; download to a temporary `.part` file and skip a dataset when its expected extracted marker already exists.

- [ ] **Step 3: Remove hard-coded working-directory assumptions from shell scripts**

Each shell script must compute its own directory with `SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"`, write into that directory, use `curl -fL` or `wget` with `set -eu`, and never require the caller to `cd` first. Preserve the existing archive URLs and extracted directory names.

- [ ] **Step 4: Verify dry-run and shell syntax**

Run: `pytest -q tests/test_prepare_data.py` and `bash -n assignment1/cs231n/datasets/get_datasets.sh assignment2/assignment2/cs231n/datasets/get_datasets.sh assignment3/assignment3/cs231n/datasets/get_datasets.sh`

Expected: dry-run PASS and all shell scripts parse successfully.

### Task 4: Replace Colab-only notebook setup cells

**Files:**
- Create: `scripts/migrate_notebooks.py`
- Modify: all teaching and submission notebooks under `assignment1`, `assignment2/assignment2`, and `assignment3/assignment3`
- Create: `tests/test_notebooks.py`

- [ ] **Step 1: Write static checks for migrated notebooks**

`tests/test_notebooks.py` must load every `.ipynb` with `nbformat` and assert no code cell contains an unconditional `from google.colab import drive`, `/content/drive`, `My Drive`, or `FOLDERNAME`. It must also assert that the first code cell imports `cs231n_local` and calls `setup_notebook` with the matching assignment name.

- [ ] **Step 2: Implement a deterministic Notebook JSON migration script**

`migrate_notebooks.py` must:

1. Replace each first Colab mount/path cell with a local setup cell that computes `CONFIG = setup_notebook("assignmentN")`, prints `CONFIG.assignment_root` and `CONFIG.device`, and changes into `CONFIG.assignment_root` with `os.chdir`.
2. Replace hard-coded relative dataset paths such as `cs231n/datasets/...` with `str(CONFIG.data_root / "...")` where the surrounding code requires a string.
3. Replace `device = "cuda" if ...` and `torch.device("cuda...")` patterns with `CONFIG.device` while preserving explicit CPU fallbacks in helper functions.
4. Preserve markdown, student TODO cells, execution outputs, and notebook ordering.
5. Be idempotent: running the script twice produces byte-identical notebooks.

- [ ] **Step 3: Run the migration and static checks**

Run: `python scripts/migrate_notebooks.py --root .` followed by `pytest -q tests/test_notebooks.py`

Expected: all notebooks are rewritten once, then static checks PASS on a second run with no diff.

- [ ] **Step 4: Check local imports from each assignment**

Run:

```bash
uv run python -c 'from cs231n_local import setup_notebook; print(setup_notebook("assignment1"))'
uv run python -c 'import sys; sys.path.insert(0, "assignment2/assignment2"); import cs231n'
uv run python -c 'import sys; sys.path.insert(0, "assignment3/assignment3"); import cs231n'
```

Expected: configuration prints a valid repository/data path and both assignment packages import without Colab modules.

### Task 5: Make submission collection runnable locally

**Files:**
- Create: `scripts/collect_submission.py`
- Modify: `assignment1/collectSubmission.sh`
- Modify: `assignment2/assignment2/collectSubmission.sh`
- Modify: `assignment3/assignment3/collect_submission.ipynb`
- Modify: `assignment2/assignment2/collect_submission.ipynb`
- Modify: `assignment1/collect_submission.ipynb`

- [ ] **Step 1: Add a local CLI with the existing output contract**

Implement `scripts/collect_submission.py --assignment assignmentN --output-dir <dir>` to gather that assignment's `.py` and `.ipynb` files and invoke `jupyter nbconvert --to pdf` (or the existing `makepdf.py`) to produce `aN_code_submission.zip` and `aN_inline_submission.pdf` in the selected output directory.

- [ ] **Step 2: Replace submission Notebook mount cells**

Use the same `setup_notebook` initialization as teaching notebooks and remove Drive-specific `FOLDERNAME`/`%cd` cells. Keep the existing collection file lists and output names.

- [ ] **Step 3: Verify collection on a tiny temporary copy**

Run: `uv run python scripts/collect_submission.py --assignment assignment1 --output-dir /tmp/cs231n-submission-check`

Expected: a zip containing `.py`/`.ipynb` files is created; PDF conversion either succeeds or prints the exact missing LaTeX dependency and leaves the zip intact.

### Task 6: Run the full migration verification

**Files:**
- Modify: `README.md` if verification commands expose a documentation mismatch
- Create: `tests/test_smoke_imports.py`

- [ ] **Step 1: Add smoke imports and device tests**

Test that each assignment package imports after temporarily adding its assignment root to `sys.path`, that `resolve_device("cpu")` returns CPU, and that no test imports `google.colab`.

- [ ] **Step 2: Run the complete checks**

Run: `uv run pytest -q`

Expected: configuration, data dry-run, notebook static, and smoke tests PASS.

- [ ] **Step 3: Run lightweight notebook checks**

Run `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120` only for the first setup/shape cells or use the repository's existing test tags; skip COCO downloads and long training when their data markers are absent.

- [ ] **Step 4: Inspect the final diff and record limitations**

Run: `git diff --stat` and `git diff --check`.

Confirm no student TODO or model code changed, all Colab paths are gone from executable cells, and the README names the cloud GPU data preparation commands.
