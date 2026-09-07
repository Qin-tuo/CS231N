#!/usr/bin/env python3
"""Print the installed runtime versions and the selected compute device."""

from __future__ import annotations

import importlib
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs231n_local import assignment_dir, data_dir, resolve_device


REQUIRED = ("numpy", "scipy", "matplotlib", "PIL", "h5py", "future", "torch", "torchvision", "nbformat", "nbconvert", "ipykernel")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", choices=["assignment1", "assignment2", "assignment3"])
    parser.add_argument("--davis", action="store_true", help="Also check optional DAVIS data dependencies")
    args = parser.parse_args()
    failed = False
    modules = list(REQUIRED)
    if args.assignment == "assignment3":
        modules += ["clip", "cv2", "thop", "einops", "joblib"]
    if args.davis:
        modules += ["tensorflow", "tensorflow_datasets"]
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failed = True
            print(f"{name}: MISSING ({exc})", file=sys.stderr)
            continue
        version = getattr(module, "__version__", "installed")
        print(f"{name}: {version}")

    if failed:
        extras = " --extra assignment3" if args.assignment == "assignment3" else ""
        if args.davis:
            extras += " --extra davis"
        print(f"Install the missing dependencies with `uv sync{extras}`.", file=sys.stderr)
        return 1

    try:
        device = resolve_device("cpu" if args.assignment == "assignment1" else None)
        import torch
        x = torch.arange(4, device=device, dtype=torch.float32, requires_grad=True)
        x.square().sum().backward()
        assert torch.equal(x.grad.cpu(), torch.tensor([0., 2., 4., 6.]))
        print(f"device: {device} (tensor forward/backward passed)")
    except (RuntimeError, ValueError) as exc:
        print(f"device: ERROR ({exc})", file=sys.stderr)
        return 1
    for assignment in [args.assignment] if args.assignment else ["assignment1", "assignment2", "assignment3"]:
        print(f"{assignment}: {assignment_dir(assignment, ROOT)}")
        print(f"  data: {data_dir(assignment, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
