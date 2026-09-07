#!/usr/bin/env python3
"""Collect assignment source and saved notebooks into the course ZIP and PDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs231n_local.config import assignment_dir


NOTEBOOKS = {
    "assignment1": (
        "01 knn.ipynb",
        "02 softmax.ipynb",
        "03 two_layer_net.ipynb",
        "04 features.ipynb",
        "05 FullyConnectedNets.ipynb",
    ),
    "assignment2": (
        "01 BatchNormalization.ipynb",
        "02 Dropout.ipynb",
        "03 ConvolutionalNetworks.ipynb",
        "04 PyTorch.ipynb",
        "05 RNN_Captioning_pytorch.ipynb",
    ),
    "assignment3": (
        "01 Transformer_Captioning.ipynb",
        "02 Self_Supervised_Learning.ipynb",
        "03 DDPM.ipynb",
        "04 CLIP_DINO.ipynb",
    ),
}

REQUIRED_CODE = {
    "assignment1": (
        "cs231n/classifiers/k_nearest_neighbor.py",
        "cs231n/classifiers/linear_classifier.py",
        "cs231n/classifiers/softmax.py",
        "cs231n/classifiers/fc_net.py",
        "cs231n/optim.py",
        "cs231n/solver.py",
        "cs231n/layers.py",
    ),
    "assignment2": (
        "cs231n/layers.py",
        "cs231n/classifiers/fc_net.py",
        "cs231n/optim.py",
        "cs231n/solver.py",
        "cs231n/classifiers/cnn.py",
        "cs231n/im2col_cython.pyx",
        "cs231n/classifiers/rnn_pytorch.py",
    ),
    "assignment3": (
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
    ),
}

EXCLUDED_DIRS = {"datasets", "pretrained", "checkpoints", "saved", "__pycache__"}
EXCLUDED_FILES = {"makepdf.py", "collect_submission.py"}


def submission_files(assignment: str, assignment_root: Path) -> tuple[list[Path], list[Path]]:
    """Return ordered teaching notebooks and the full set of submission files."""
    if assignment not in NOTEBOOKS:
        raise ValueError(f"Unknown assignment: {assignment}")
    required = (*NOTEBOOKS[assignment], *REQUIRED_CODE[assignment])
    missing = [name for name in required if not (assignment_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required submission files are missing from {assignment_root}: "
            + ", ".join(missing)
        )
    notebooks = [assignment_root / name for name in NOTEBOOKS[assignment]]
    files = list(notebooks)
    for directory, directories, filenames in os.walk(assignment_root):
        directories[:] = sorted(
            name for name in directories
            if name not in EXCLUDED_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix in {".py", ".pyx"} and name not in EXCLUDED_FILES:
                files.append(path)
    # The original A1/A2 archives also include trained models in cs231n/saved.
    if assignment in {"assignment1", "assignment2"}:
        saved_exclusions = EXCLUDED_DIRS - {"saved", "checkpoints"}
        for directory, directories, filenames in os.walk(assignment_root / "cs231n/saved"):
            directories[:] = sorted(
                name for name in directories
                if name not in saved_exclusions and not name.startswith(".")
            )
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.is_file():
                    files.append(path)
    return notebooks, files


def export_pdf(notebooks: list[Path], assignment_root: Path, destination: Path) -> None:
    """Render saved notebook contents without executing student code."""
    try:
        import nbformat
        from nbconvert import PDFExporter
    except ImportError as exc:
        raise RuntimeError(
            f"Missing PDF Python dependency: {exc.name or 'nbformat/nbconvert'}. "
            f"Run `uv sync` from {ROOT}, then rerun with `uv run python`. "
            "Use --code-only to collect the ZIP without a PDF."
        ) from exc

    missing = [name for name in ("pandoc", "xelatex") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing PDF dependencies: {', '.join(missing)}. "
            "Install Pandoc and a TeX distribution containing XeLaTeX "
            "(MacTeX on macOS or TeX Live on Linux), ensure `pandoc` and "
            "`xelatex` are on PATH, then rerun this command. "
            "Use --code-only to collect the ZIP without a PDF."
        )

    try:
        combined = nbformat.v4.new_notebook()
        for notebook_index, path in enumerate(notebooks):
            notebook = nbformat.read(path, as_version=4)
            if notebook_index == 0:
                combined.metadata = notebook.metadata
            else:
                combined.cells.append(nbformat.v4.new_raw_cell(
                    r"\newpage", metadata={"raw_mimetype": "text/latex"}
                ))
            combined.cells.extend(notebook.cells)
        # Separate notebooks can reuse cell IDs; IDs must be unique after merging.
        for index, cell in enumerate(combined.cells):
            cell.id = f"submission-cell-{index}"
        pdf, _ = PDFExporter().from_notebook_node(
            combined, resources={"metadata": {"path": str(assignment_root)}}
        )
        destination.write_bytes(pdf)
    except Exception as exc:
        raise RuntimeError(
            f"PDF conversion failed: {exc}\n"
            "Check the reported notebook or LaTeX dependency and rerun. "
            "Use --code-only to collect the ZIP without a PDF."
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, choices=tuple(NOTEBOOKS))
    parser.add_argument(
        "--output-dir", type=Path,
        help="Artifact directory (defaults to the assignment directory)",
    )
    parser.add_argument(
        "--code-only", action="store_true",
        help="Create the code/notebook ZIP without PDF dependencies",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive = None
    try:
        assignment_root = assignment_dir(args.assignment, ROOT)
        notebooks, files = submission_files(args.assignment, assignment_root)
        output_dir = (args.output_dir or assignment_root).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"a{args.assignment[-1]}"
        archive = output_dir / f"{prefix}_code_submission.zip"
        pdf = output_dir / f"{prefix}_inline_submission.pdf"
        with tempfile.TemporaryDirectory(prefix=".submission-", dir=output_dir) as temporary:
            temporary_root = Path(temporary)
            zip_temporary = temporary_root / archive.name
            with zipfile.ZipFile(zip_temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in files:
                    bundle.write(path, arcname=path.relative_to(assignment_root).as_posix())
            zip_temporary.replace(archive)
            print(f"Created {archive}", flush=True)
            if not args.code_only:
                pdf_temporary = temporary_root / pdf.name
                export_pdf(notebooks, assignment_root, pdf_temporary)
                pdf_temporary.replace(pdf)
                print(f"Created {pdf}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"collect_submission: {exc}", file=sys.stderr)
        if archive is not None and archive.is_file():
            print(f"ZIP retained at {archive}; PDF was not updated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
