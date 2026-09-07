#!/usr/bin/env python3
"""Execute notebook startup/import cells in fresh Jupyter kernels, without training."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs231n_local import assignment_dir


# Explicit cell selections avoid running student TODOs, downloads, or training.
STARTUP_CELLS = {
    "01 knn.ipynb": [0, 2],
    "02 softmax.ipynb": [0, 2],
    "03 two_layer_net.ipynb": [0, 2],
    "04 features.ipynb": [0, 2],
    "05 FullyConnectedNets.ipynb": [0, 4],
    "01 BatchNormalization.ipynb": [0, 2],
    "02 Dropout.ipynb": [0, 2],
    "03 ConvolutionalNetworks.ipynb": [0, 2],
    "04 PyTorch.ipynb": [0, 5],
    "05 RNN_Captioning_pytorch.ipynb": [0, 2],
    "01 Transformer_Captioning.ipynb": [0, 2],
    "02 Self_Supervised_Learning.ipynb": [0, 7, 11],
    "03 DDPM.ipynb": [0, 4],
    "04 CLIP_DINO.ipynb": [0, 8, 9],
    "collect_submission.ipynb": [0],
}
CIFAR_CELLS = {
    "01 knn.ipynb": 3, "02 softmax.ipynb": 4, "03 two_layer_net.ipynb": 3,
    "04 features.ipynb": 4, "05 FullyConnectedNets.ipynb": 5,
    "01 BatchNormalization.ipynb": 3, "02 Dropout.ipynb": 3,
    "03 ConvolutionalNetworks.ipynb": 3, "04 PyTorch.ipynb": 7,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", choices=["assignment1", "assignment2", "assignment3", "all"], default="all")
    parser.add_argument("--with-cifar", action="store_true", help="Also load prepared CIFAR-10 in the relevant notebooks")
    args = parser.parse_args()
    assignments = [args.assignment] if args.assignment != "all" else ["assignment1", "assignment2", "assignment3"]
    failed = 0
    for assignment in assignments:
        for path in sorted(assignment_dir(assignment, ROOT).glob("*.ipynb")):
            try:
                original = nbformat.read(path, as_version=4)
                indices = list(STARTUP_CELLS[path.name])
                if args.with_cifar and path.name in CIFAR_CELLS:
                    indices.append(CIFAR_CELLS[path.name])
                selected = nbformat.v4.new_notebook(
                    cells=[copy.deepcopy(original.cells[i]) for i in indices]
                )
                NotebookClient(
                    selected, kernel_name="python3", timeout=120,
                    resources={"metadata": {"path": str(path.parent)}},
                ).execute()
                print(f"PASS {assignment}/{path.name} ({len(indices)} cells)", flush=True)
            except Exception as exc:
                failed += 1
                print(f"FAIL {assignment}/{path.name}: {exc}", file=sys.stderr, flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
