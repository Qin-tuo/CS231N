from pathlib import Path
import subprocess
import sys
import io
import shutil
import tarfile
import zipfile

import pytest

from scripts import prepare_data


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_data_dry_run_reports_assignment_destination(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--assignment",
            "assignment1",
            "--dataset",
            "cifar10",
            "--data-root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(tmp_path / "assignment1") in result.stdout
    assert "cifar-10-python.tar.gz" in result.stdout


def test_prepare_data_rejects_invalid_assignment():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--assignment",
            "assignment9",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "assignment" in result.stderr.lower()


def test_preparation_runs_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--assignment", "assignment2", "--dataset", "cifar10", "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(ROOT / "assignment2/assignment2/cs231n/datasets") in result.stdout


def test_partial_dataset_is_not_treated_as_complete(tmp_path, monkeypatch):
    destination = tmp_path / "datasets"
    marker = destination / prepare_data.DATASETS["cifar10"].marker
    marker.parent.mkdir(parents=True)
    marker.write_text("partial")
    calls = []

    def fail_download(*args):
        calls.append(args)
        raise OSError("offline")

    monkeypatch.setattr(prepare_data, "download", fail_download)
    with pytest.raises(OSError, match="offline"):
        prepare_data.prepare_dataset(prepare_data.DATASETS["cifar10"], destination)
    assert calls
    assert marker.read_text() == "partial"
    assert not list(destination.glob("*.part"))


@pytest.mark.parametrize("archive_type", ["tar", "zip"])
def test_archive_preparation_is_idempotent(tmp_path, monkeypatch, archive_type):
    archive = tmp_path / "fixture"
    content = b"dataset content"
    if archive_type == "tar":
        with tarfile.open(archive, "w:gz") as output:
            info = tarfile.TarInfo("tiny/data.txt")
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    else:
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("tiny/data.txt", content)
    dataset = prepare_data.Dataset("tiny", "https://example.invalid/tiny", "tiny.archive", "tiny/data.txt", archive_type)
    calls = []

    def download_fixture(url, path):
        calls.append(url)
        shutil.copyfile(archive, path)

    monkeypatch.setattr(prepare_data, "download", download_fixture)
    destination = tmp_path / "output"
    prepare_data.prepare_dataset(dataset, destination)
    prepare_data.prepare_dataset(dataset, destination)
    assert (destination / "tiny/data.txt").read_bytes() == content
    assert len(calls) == 1
    assert not list(destination.glob("*.part"))


def test_rejects_tar_path_traversal_before_writing(tmp_path, monkeypatch):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 4
        output.addfile(info, io.BytesIO(b"oops"))
    monkeypatch.setattr(prepare_data, "download", lambda url, path: shutil.copyfile(archive, path))
    dataset = prepare_data.Dataset("unsafe", "fixture", "unsafe.tar.gz", "missing", "tar")
    with pytest.raises((ValueError, tarfile.TarError, RuntimeError)):
        prepare_data.prepare_dataset(dataset, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_failed_archive_does_not_leave_a_success_marker(tmp_path, monkeypatch):
    archive = tmp_path / "partial.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("cifar-10-batches-py/batches.meta", "partial")
    monkeypatch.setattr(prepare_data, "download", lambda url, path: shutil.copyfile(archive, path))
    original = prepare_data.DATASETS["cifar10"]
    from dataclasses import replace
    dataset = replace(original, archive_type="zip")
    with pytest.raises(RuntimeError, match="missing"):
        prepare_data.prepare_dataset(dataset, tmp_path / "output")
    assert not (tmp_path / "output" / original.marker).exists()


def test_assignment3_resources_include_data_and_pretrained_checkpoint():
    resources = prepare_data.datasets_for("assignment3", "ddpm")
    assert {item.marker for item in resources} == {
        "emoji_data.npz", "text_embeddings.pt", "pretrained_model/ddpm/model-70000.pt"
    }
    assert prepare_data.datasets_for("assignment3", "simclr")[0].marker == "pretrained_model/pretrained_simclr_model.pth"


def test_download_rejects_truncated_response(tmp_path, monkeypatch):
    response = io.BytesIO(b"short")
    response.headers = {"Content-Length": "20"}
    monkeypatch.setattr(prepare_data, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="Incomplete download"):
        prepare_data.download("https://example.invalid/model.pt", tmp_path / "model.part")


def test_empty_pretrained_file_is_not_complete(tmp_path):
    (tmp_path / "emoji_data.npz").touch()
    assert not prepare_data.DATASETS["emoji"].complete(tmp_path)


def test_interrupted_publish_is_not_treated_as_complete(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("tiny/marker", "complete")
        output.writestr("tiny/data", "full contents")
    dataset = prepare_data.Dataset("tiny", "fixture", "tiny.zip", "tiny/marker", "zip", ("tiny/data",))
    monkeypatch.setattr(prepare_data, "download", lambda url, path: shutil.copyfile(archive, path))
    real_copytree = shutil.copytree

    def interrupted_copytree(source, target, **kwargs):
        real_copytree(source, target, **kwargs)
        (target / "data").write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(prepare_data.shutil, "copytree", interrupted_copytree)
    with pytest.raises(OSError, match="disk full"):
        prepare_data.prepare_dataset(dataset, tmp_path / "output")
    assert not dataset.complete(tmp_path / "output")
