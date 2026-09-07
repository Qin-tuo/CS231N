#!/usr/bin/env python3
"""Migrate the bundled Colab notebooks without changing exercises or outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def setup_source(assignment: str) -> str:
    return f'''# Local CS231n setup.
import os
import sys
from pathlib import Path

_start = Path(os.environ.get("CS231N_ROOT", Path.cwd())).expanduser().resolve()
for _root in (_start, *_start.parents):
    if (_root / "cs231n_local").is_dir() and (_root / "pyproject.toml").is_file():
        sys.path.insert(0, str(_root))
        break
else:
    raise RuntimeError("Open this notebook inside assignments, or set CS231N_ROOT to that directory.")

from cs231n_local import setup_notebook
CONFIG = setup_notebook("{assignment}")
os.chdir(CONFIG.assignment_root)
print("Assignment:", CONFIG.assignment_root)
print("Data:", CONFIG.data_root)
print("Device:", CONFIG.device)
'''


def migrate_source(source: str, name: str, assignment: str) -> str:
    if name == "collect_submission.ipynb" and "collectSubmission.sh" in source:
        return f'''import subprocess
subprocess.run(
    [sys.executable, str(CONFIG.repo_root / "scripts" / "collect_submission.py"),
     "--assignment", "{assignment}", "--output-dir", str(CONFIG.assignment_root)],
    check=True,
)
'''

    if "setup.py build_ext --inplace" in source:
        return '''import subprocess
subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=CONFIG.assignment_root / "cs231n", check=True,
)
import importlib
import cs231n.fast_layers
importlib.invalidate_caches()
importlib.reload(cs231n.fast_layers)
'''

    if "%%bash" in source and "pretrained_simclr_model.pth" in source:
        return '''pretrained_path = CONFIG.require_data(
    "pretrained_model/pretrained_simclr_model.pth", "simclr"
)
'''

    source = "\n".join(
        line for line in source.split("\n")
        if not re.match(r"\s*[!%]\s*pip install", line)
        and not (line.lstrip().startswith("#") and any(
            token in line for token in ("google.colab", "/content/drive", "FOLDERNAME")
        ))
    )
    source = source.replace(
        "if USE_GPU and torch.cuda.is_available():\n    device = torch.device('cuda')\nelse:\n    device = torch.device('cpu')",
        "device = torch.device(CONFIG.device if USE_GPU else 'cpu')",
    )
    for expression in (
        'torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        'torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
    ):
        source = source.replace(expression, "torch.device(CONFIG.device)")
    source = source.replace('"cuda" if torch.cuda.is_available() else "cpu"', "CONFIG.device")
    source = source.replace("'cuda' if torch.cuda.is_available() else 'cpu'", "CONFIG.device")
    source = source.replace("device='cuda'", "device=device")
    for call in (
        "train_val(model, train_loader, optimizer, epoch, epochs)",
        "train_val(model, test_loader, None, epoch, epochs)",
    ):
        source = source.replace(call, call[:-1] + ", device=device)")

    source = source.replace("'cs231n/datasets/cifar-10-batches-py'", 'str(CONFIG.require_data("cifar-10-batches-py", "cifar10"))')
    source = source.replace("'./cs231n/datasets'", "str(CONFIG.data_root)")
    for old in ("root='data'", "root='./data'"):
        source = source.replace(old, "root=str(CONFIG.data_root)")
    if name != "04 features.ipynb":
        source = source.replace("get_CIFAR10_data()", "get_CIFAR10_data(data_root=CONFIG.require_data(\"cifar-10-batches-py\", \"cifar10\").parent)")
    source = re.sub(
        r"\bload_coco_data\((?!base_dir=)",
        'load_coco_data(base_dir=CONFIG.require_data("coco_captioning", "coco"), ',
        source,
    )

    for filename in ("unet.png", "CLIP.png", "dino.gif", "dino_res.mp4"):
        for quote in ("'", '"'):
            old = f"f{quote}/content/drive/My Drive/{{FOLDERNAME}}/{filename}{quote}"
            source = source.replace(old, f'str(CONFIG.assignment_root / "{filename}")')
    source = source.replace(
        'f"/content/drive/My Drive/{FOLDERNAME}/cs231n/exp/pretrained"',
        'str(CONFIG.data_root / "pretrained_model" / "ddpm")',
    )
    source = source.replace(
        "trainer.load(70000)",
        'CONFIG.require_data("pretrained_model/ddpm/model-70000.pt", "ddpm")\ntrainer.load(70000)',
    ) if 'CONFIG.require_data("pretrained_model/ddpm/model-70000.pt"' not in source else source
    source = source.replace(
        "EmojiDataset(image_size)",
        'EmojiDataset(image_size, data_path=CONFIG.require_data("emoji_data.npz", "ddpm"), '
        'text_emb_path=CONFIG.require_data("text_embeddings.pt", "ddpm"))',
    )
    source = source.replace("DavisDataset()", 'DavisDataset(data_dir=str(CONFIG.data_root / "tensorflow_datasets"))')
    source = source.replace('clip.load("ViT-B/32", device=device)', 'clip.load("ViT-B/32", device=device, download_root=str(CONFIG.data_root / "clip"))')
    source = source.replace('ClipEmbed(device)', 'ClipEmbed(device, download_root=str(CONFIG.data_root / "clip"))')
    source = source.replace("torch.load('simclr_sanity_check.key')", "torch.load('simclr_sanity_check.key', map_location='cpu', weights_only=False)")
    source = source.replace("'./pretrained_model/pretrained_simclr_model.pth'", 'str(CONFIG.require_data("pretrained_model/pretrained_simclr_model.pth", "simclr"))')
    source = source.replace("'./pretrained_model/trained_simclr_model.pth'", 'str(CONFIG.data_root / "pretrained_model" / "trained_simclr_model.pth")')
    source = source.replace("num_workers=16", "num_workers=0")
    source = source.replace("pin_memory=True", 'pin_memory=str(device).startswith("cuda")')
    return source


def migrate_notebook(path: Path, assignment: str) -> bool:
    notebook = json.loads(path.read_text())
    changed = False
    first_code = True
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        old = "".join(cell["source"])
        new = setup_source(assignment) if first_code else migrate_source(old, path.name, assignment)
        first_code = False
        if old != new:
            cell["source"] = new.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    changed = 0
    for assignment in ("assignment1", "assignment2", "assignment3"):
        for path in sorted((args.root / assignment).rglob("*.ipynb")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            if migrate_notebook(path, assignment):
                print(path)
                changed += 1
    print(f"Migrated {changed} notebooks.")


if __name__ == "__main__":
    main()
