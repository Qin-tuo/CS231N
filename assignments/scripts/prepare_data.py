#!/usr/bin/env python3
"""Download and unpack the datasets used by the CS231n assignments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.client import HTTPException
import shutil
from pathlib import Path
import sys
import tarfile
import tempfile
import stat
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs231n_local.config import data_dir


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive_name: str
    marker: str
    archive_type: str
    required_files: tuple[str, ...] = ()

    def complete(self, root: Path) -> bool:
        return all((root / name).is_file() and (root / name).stat().st_size > 0
                   for name in (self.marker, *self.required_files))


DATASETS = {
    "cifar10": Dataset(
        "cifar10",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "cifar-10-python.tar.gz",
        "cifar-10-batches-py/batches.meta",
        "tar",
        tuple(f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6))
        + ("cifar-10-batches-py/test_batch",),
    ),
    "imagenet_val": Dataset(
        "imagenet_val",
        "https://cs231n.stanford.edu/imagenet_val_25.npz",
        "imagenet_val_25.npz",
        "imagenet_val_25.npz",
        "file",
    ),
    "coco": Dataset(
        "coco",
        "https://cs231n.stanford.edu/coco_captioning.zip",
        "coco_captioning.zip",
        "coco_captioning/coco2014_captions.h5",
        "zip",
        tuple("coco_captioning/" + name for name in (
            "coco2014_vocab.json", "train2014_urls.txt", "val2014_urls.txt",
            "train2014_vgg16_fc7_pca.h5", "val2014_vgg16_fc7_pca.h5",
            "train2014_vgg16_fc7.h5", "val2014_vgg16_fc7.h5",
        )),
    ),
    "simclr": Dataset(
        "simclr", "https://cs231n.stanford.edu/2025/storage/a3/pretrained_simclr_model.pth",
        "pretrained_model/pretrained_simclr_model.pth",
        "pretrained_model/pretrained_simclr_model.pth", "file",
    ),
    "emoji": Dataset(
        "emoji", "https://cs231n.stanford.edu/2025/storage/a3/emoji_data.npz",
        "emoji_data.npz", "emoji_data.npz", "file",
    ),
    "text_embeddings": Dataset(
        "text_embeddings", "https://cs231n.stanford.edu/2025/storage/a3/text_embeddings.pt",
        "text_embeddings.pt", "text_embeddings.pt", "file",
    ),
    "ddpm_weights": Dataset(
        "ddpm_weights", "https://cs231n.stanford.edu/2025/storage/a3/model-70000.pt",
        "pretrained_model/ddpm/model-70000.pt", "pretrained_model/ddpm/model-70000.pt", "file",
    ),
}


def datasets_for(assignment: str, requested: str) -> list[Dataset]:
    if assignment not in {"assignment1", "assignment2", "assignment3"}:
        raise ValueError(f"Unknown assignment: {assignment}")
    if requested == "all":
        names = ["cifar10", "imagenet_val"] if assignment == "assignment1" else ["cifar10", "coco"]
        if assignment == "assignment3":
            names += ["simclr", "emoji", "text_embeddings", "ddpm_weights"]
    elif requested == "ddpm":
        names = ["emoji", "text_embeddings", "ddpm_weights"]
    else:
        names = [requested]
    if "coco" in names and assignment == "assignment1":
        raise ValueError("COCO data is only used by assignment2 and assignment3")
    if any(name in names for name in ("simclr", "emoji", "text_embeddings", "ddpm_weights")) and assignment != "assignment3":
        raise ValueError("SimCLR and DDPM resources are only used by assignment3")
    if any(name not in DATASETS for name in names):
        raise ValueError(f"Unknown dataset: {requested}")
    return [DATASETS[name] for name in names]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
        expected_size = response.headers.get("Content-Length")
        if expected_size is not None and output.tell() != int(expected_size):
            raise RuntimeError(f"Incomplete download: {url} ({output.tell()}/{expected_size} bytes)")


def prepare_dataset(dataset: Dataset, destination: Path, dry_run: bool = False) -> None:
    marker = destination / dataset.marker
    if dataset.complete(destination):
        print(f"skip {dataset.name}: {marker}")
        return
    print(f"prepare {dataset.name}: {dataset.url} -> {destination}")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{dataset.name}-", suffix=".part", dir=destination, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        download(dataset.url, temporary)
        # Validate a full extraction before publishing any completion markers.
        with tempfile.TemporaryDirectory(prefix=f"{dataset.name}-", dir=destination) as staging_dir:
            staging = Path(staging_dir)
            if dataset.archive_type == "tar":
                with tarfile.open(temporary, "r:gz") as archive_file:
                    archive_file.extractall(staging, filter="data")
            elif dataset.archive_type == "zip":
                with zipfile.ZipFile(temporary) as archive_file:
                    for member in archive_file.infolist():
                        target = (staging / member.filename).resolve()
                        if not target.is_relative_to(staging.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError(f"Unsafe archive entry: {member.filename}")
                    archive_file.extractall(staging)
            elif dataset.archive_type == "file":
                target = staging / dataset.archive_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temporary), str(target))
            else:
                raise ValueError(f"Unknown archive type: {dataset.archive_type}")
            if not dataset.complete(staging):
                raise RuntimeError(f"Download incomplete; expected dataset files are missing: {dataset.name}")
            # Publish the marker last so an interrupted copy is retried.
            with tempfile.TemporaryDirectory(prefix=f".{dataset.name}-marker-", dir=destination) as marker_dir:
                pending_marker = Path(marker_dir) / "complete"
                (staging / dataset.marker).replace(pending_marker)
                marker.unlink(missing_ok=True)
                for source in staging.iterdir():
                    target = destination / source.name
                    if source.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    else:
                        source.replace(target)
                marker.parent.mkdir(parents=True, exist_ok=True)
                pending_marker.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
    if not dataset.complete(destination):
        raise RuntimeError(f"Download completed but expected marker is missing: {marker}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, choices=["assignment1", "assignment2", "assignment3"])
    parser.add_argument("--dataset", default="all", choices=["cifar10", "imagenet_val", "coco", "simclr", "ddpm", "all"])
    parser.add_argument("--data-root", type=Path, help="External directory containing assignment data")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        datasets = datasets_for(args.assignment, args.dataset)
        destination = data_dir(args.assignment, ROOT, data_root=args.data_root)
        for dataset in datasets:
            prepare_dataset(dataset, destination, dry_run=args.dry_run)
    except (OSError, RuntimeError, ValueError, HTTPException, zipfile.BadZipFile, tarfile.TarError) as exc:
        print(f"prepare_data: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
