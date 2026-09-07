# Local Workflow Verification

Verified on 2026-09-07 on an Apple Silicon Mac with Python 3.12.14 and MPS.

## Scope

The local/cloud runtime migration is implemented. Course exercises and training are separate work: student algorithm TODOs, written answers, notebook ordering, metadata, execution counts, and saved outputs were preserved. No GPU VM was rented and no training job was started.

## Checks Completed

- `uv run --extra assignment3 pytest -q`: 90 tests passed. One upstream CLIP `pkg_resources` deprecation warning remains; the compatible setuptools version is constrained in `pyproject.toml`.
- All 17 notebooks passed startup/import execution in fresh Jupyter kernels.
- All nine CIFAR-loading notebook cells across Assignment 1/2 passed with real data.
- MPS tensor forward/backward passed; a SimCLR model forward and THOP profile also passed on MPS with random inputs.
- Assignment 2 Cython extension compiled; fast-convolution forward/backward output shapes passed.
- All shell scripts passed syntax checks; all seven dataset wrappers ran from `/tmp`.
- Migration is idempotent, and snapshot comparison confirmed preservation of non-code cells, metadata, execution counts, and outputs.
- Actual submission ZIPs for all three assignments passed archive integrity checks. Collector tests verify saved training artifacts for Assignment 1/2 are preserved.
- Independent code review completed with no unresolved findings.

## Data Prepared

CIFAR-10 is installed at `assignment1/cs231n/datasets/cifar-10-batches-py`. Assignment 2/3 use relative symlinks to this same copy. The downloaded archive's MD5 matches torchvision's recorded value: `c58f30108f718f92721af3b95e74349a`.

COCO, SimCLR/DDPM checkpoints, CLIP/DINO weights, and DAVIS were not downloaded. Their preparation commands and persistent-cache paths are documented in `README.md`. The `assignment3` dependencies are installed locally; the optional `davis` dependencies are not installed.

## Limits

- CUDA hardware and full training runs were not exercised. Explicit CUDA selection fails clearly on this Mac; test device availability again on a GPU VM.
- Pandoc and XeLaTeX are not installed, so actual PDF rendering was not exercised. The collector reports these dependencies, returns nonzero, and retains its ZIP. `--code-only` works without PDF tools.
- The bundled notebooks retain their original older metadata. Current Jupyter can report a missing-cell-ID compatibility warning while reading some notebooks.
- Git has no commits. Changes are present in the original workspace on `local-runtime`; the source snapshot from before this session is `/tmp/cs231n-local-before-20260907.tar.gz`.

## Resume Study

From `assignments`:

```bash
uv run --extra assignment3 jupyter lab --no-browser --ip=127.0.0.1
```

Open `assignment1/01 knn.ipynb` to start Assignment 1. Use a separate kernel for each notebook, and retain the appropriate `uv` extras when running later assignments. See `README.md` for cloud GPU, data preparation, verification, and collection commands.
