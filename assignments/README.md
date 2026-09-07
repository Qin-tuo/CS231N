# CS231n Assignments: Local and Cloud GPU Workflow

Run all commands below from this `assignments` directory. The notebooks use local Jupyter kernels and preserve the course exercises, student TODOs, written answers, and saved outputs. Runtime migration does not mean the exercises or training runs are complete.

## Setup

Use `uv`; `.python-version` selects Python 3.12. The supported range is Python 3.11-3.13, and `uv.lock` records dependency versions. Python 3.14 is not part of this environment.

```bash
uv sync --locked
uv run python scripts/check_environment.py --assignment assignment1
uv run jupyter lab --no-browser --ip=127.0.0.1
```

Open `assignment1/01 knn.ipynb` and run its first cell. Assignment 2 notebooks are under `assignment2/assignment2`; Assignment 3 notebooks are under `assignment3/assignment3`. Each notebook should have its own kernel because the three assignments contain different packages all named `cs231n`.

Prepare CIFAR-10 before loading data (about 163 MiB downloaded; NumPy loading needs several GiB of RAM):

```bash
uv run python scripts/prepare_data.py --assignment assignment1 --dataset cifar10
uv run python scripts/prepare_data.py --assignment assignment2 --dataset cifar10
```

Downloads skip complete datasets and use temporary files and staged extraction. Interrupted or incomplete datasets are retried. `--dry-run` prints URLs and destinations without downloading; the legacy dataset shell scripts also work from any directory.

Captioning notebooks need COCO features. Prepare these larger resources where the notebook will run:

```bash
uv run python scripts/prepare_data.py --assignment assignment2 --dataset coco
uv run python scripts/prepare_data.py --assignment assignment3 --dataset coco
```

Assignment 3 adds optional dependencies and pretrained resources:

```bash
uv sync --locked --extra assignment3
uv run --extra assignment3 python scripts/check_environment.py --assignment assignment3
uv run --extra assignment3 python scripts/prepare_data.py --assignment assignment3 --dataset cifar10
uv run --extra assignment3 python scripts/prepare_data.py --assignment assignment3 --dataset simclr
uv run --extra assignment3 python scripts/prepare_data.py --assignment assignment3 --dataset ddpm
uv run --extra assignment3 jupyter lab --no-browser --ip=127.0.0.1
```

`ddpm` prepares emoji images, text embeddings, and the course `model-70000.pt` checkpoint. SimCLR and DDPM resources live below the configured data root. CLIP and DINO model loaders download their weights on first use. Notebook CLIP calls cache weights in `<data root>/clip`; set `TORCH_HOME` to persistent storage to retain DINO's torch.hub cache on a GPU VM.

The DAVIS section of `04 CLIP_DINO.ipynb` also needs TensorFlow and TensorFlow Datasets:

```bash
uv sync --locked --extra assignment3 --extra davis
uv run --extra assignment3 --extra davis python scripts/check_environment.py --assignment assignment3 --davis
uv run --extra assignment3 --extra davis jupyter lab --no-browser --ip=127.0.0.1
```

DAVIS downloads on the first `DavisDataset` call into `<data root>/tensorflow_datasets`. Keep extras on subsequent `uv run` commands because `uv` synchronizes the requested environment. No notebook installs packages or mounts Google Drive.

## Paths and devices

The notebooks use these optional environment variables:

- `CS231N_ROOT`: repository root when the notebook is launched outside the checkout.
- `CS231N_DATA_ROOT`: external data disk. Data is stored below `$CS231N_DATA_ROOT/assignmentN`. Set the same value during preparation and notebook execution. If set, it is used even when empty; there is no silent fallback to another dataset directory.
- `CS231N_DEVICE`: `auto` (default), `cpu`, `mps`, `cuda`, or `cuda:N`.

For example, on an Apple Silicon Mac:

```bash
CS231N_DEVICE=mps uv run --extra assignment3 jupyter lab --no-browser --ip=127.0.0.1
```

On a cloud GPU VM:

```bash
export CS231N_ROOT=/workspace/assignments
export CS231N_DATA_ROOT=/workspace/datasets
export CS231N_DEVICE=cuda
export TORCH_HOME=/workspace/datasets/model-cache/torch
uv sync --locked --extra assignment3
uv run --extra assignment3 python scripts/check_environment.py --assignment assignment3
uv run --extra assignment3 jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Copy the source and lockfile to the VM, then create its environment there; `.venv` and compiled Cython extensions are specific to the operating system. The VM needs a working NVIDIA driver; the environment check performs tensor forward/backward on the selected device. From your local terminal, forward the Jupyter port with `ssh -L 8888:127.0.0.1:8888 user@gpu-host`, then open the token URL printed by Jupyter. Use another local port if 8888 is occupied.

`auto` selects CUDA, then Apple MPS, then CPU. Assignment 1 uses CPU. NumPy and the course's CPU reference/gradient checks stay on CPU, including the RNN and captioning reference solvers. PyTorch image classification, ViT, SimCLR, DDPM, CLIP, and DINO use the selected device where supported. A successful device check does not guarantee every model operator works on MPS; use `CS231N_DEVICE=cpu` for a problematic local check, or CUDA for long runs. Training has not been launched by the setup scripts.

For the fast convolution section in Assignment 2, execute the notebook's build cell or run:

```bash
uv run python -c 'import subprocess, sys; from cs231n_local import assignment_dir; subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=assignment_dir("assignment2") / "cs231n", check=True)'
```

Compilation requires a C compiler (Xcode Command Line Tools on macOS or a build toolchain on Linux). Restart an already-running kernel after rebuilding for a different Python environment.

## Verification

```bash
uv run --extra assignment3 pytest -q
uv run --extra assignment3 python scripts/check_notebooks.py
uv run --extra assignment3 python scripts/check_notebooks.py --assignment assignment1 --with-cifar
uv run --extra assignment3 python scripts/check_notebooks.py --assignment assignment2 --with-cifar
```

The notebook check executes only the specified setup/import cells in fresh kernels. `--with-cifar` additionally loads prepared CIFAR-10. Neither mode runs student TODOs, downloads COCO/models, or starts training. Original notebook files and saved outputs are not overwritten. Use `--assignment assignment1` or `assignment2` with the base environment if Assignment 3 extras are not installed.

`scripts/migrate_notebooks.py` is the idempotent migration used for the bundled notebooks. It preserves cell ordering, non-code content, metadata, and outputs. Migration has already been applied; normal study does not require running it again.

## Submission files

Run the local collector from the repository root after executing the notebooks:

```bash
uv run --no-sync python scripts/collect_submission.py --assignment assignment1
uv run --no-sync python scripts/collect_submission.py --assignment assignment2
uv run --no-sync python scripts/collect_submission.py --assignment assignment3
```

The collector uses the numbered notebook names present here and writes `aN_code_submission.zip` and `aN_inline_submission.pdf` in the assignment directory. Override it with `--output-dir <directory>`. It includes saved notebook contents without executing exercises, plus the course's `cs231n/saved` training artifacts for Assignment 1/2 when present. Downloaded datasets and pretrained caches are excluded. `--no-sync` keeps the environment you already installed, including optional extras.

PDF export requires Pandoc and XeLaTeX (MacTeX on macOS or TeX Live on Linux) on `PATH`. If they are absent or conversion fails, the command returns nonzero and retains the ZIP. Collect just the ZIP with:

```bash
uv run --no-sync python scripts/collect_submission.py --assignment assignment1 --code-only
```
