import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from cs231n_local import assignment_dir


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2", "assignment3"])
def test_course_imports_use_the_correct_package(assignment, tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", f"""
from cs231n_local import setup_notebook
config = setup_notebook({assignment!r})
import cs231n
import cs231n.data_utils
import cs231n.gradient_check
import cs231n.optim
assert str(config.assignment_root) in cs231n.__file__
if {assignment!r} == 'assignment2':
    import cs231n.rnn_layers_pytorch
    import cs231n.fast_layers
if {assignment!r} == 'assignment3':
    import cs231n.transformer_layers
    import cs231n.gaussian_diffusion
    import cs231n.simclr.data_utils
assert 'google.colab' not in __import__('sys').modules
"""],
        cwd=tmp_path,
        env={**os.environ, "CS231N_ROOT": str(ROOT), "CS231N_DEVICE": "cpu"},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("assignment", ["assignment1", "assignment2", "assignment3"])
def test_numpy_loader_accepts_external_data_root(assignment, monkeypatch, tmp_path):
    path = assignment_dir(assignment, ROOT) / "cs231n/data_utils.py"
    spec = importlib.util.spec_from_file_location("course_data_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = []

    def load_tiny(root):
        captured.append(Path(root))
        return np.zeros((3, 4, 4, 3)), np.arange(3), np.zeros((1, 4, 4, 3)), np.zeros(1)

    monkeypatch.setattr(module, "load_CIFAR10", load_tiny)
    result = module.get_CIFAR10_data(num_training=2, num_validation=1, num_test=1, data_root=tmp_path)
    assert captured == [tmp_path / "cifar-10-batches-py"]
    assert result["X_train"].shape == (2, 3, 4, 4)


def load_assignment3_module(name):
    path = assignment_dir("assignment3", ROOT) / f"cs231n/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clip_helpers_do_not_require_tensorflow_until_davis():
    if importlib.util.find_spec("clip") is None or importlib.util.find_spec("cv2") is None:
        pytest.skip("Install assignment3 extra to check CLIP helpers")
    load_assignment3_module("clip_dino")


def test_emoji_dataset_respects_explicit_paths_without_downloading(tmp_path, monkeypatch):
    if importlib.util.find_spec("clip") is None:
        pytest.skip("Install assignment3 extra to check emoji data")
    import torch
    module = load_assignment3_module("emoji_dataset")
    data_path = tmp_path / "tiny.npz"
    embed_path = tmp_path / "tiny.pt"
    np.savez(data_path, item={"images": np.zeros((1, 4, 4, 3), dtype=np.uint8), "texts": ["hello"]})
    torch.save({"idx_mapping": {"hello": 0}, "embs": torch.zeros((1, 8))}, embed_path)

    def existing_only(path):
        assert Path(path) in (data_path, embed_path)
        assert Path(path).is_file()

    monkeypatch.setattr(module, "download_data", existing_only)
    dataset = module.EmojiDataset(4, data_path=data_path, text_emb_path=embed_path)
    assert len(dataset) == 1


def test_clip_embed_honors_explicit_weight_cache(tmp_path, monkeypatch):
    if importlib.util.find_spec("clip") is None:
        pytest.skip("Install assignment3 extra to check CLIP cache")
    import torch
    module = load_assignment3_module("emoji_dataset")
    requested = []

    def load_model(name, **kwargs):
        requested.append(kwargs)
        return torch.nn.Identity(), None

    monkeypatch.setattr(module.clip, "load", load_model)
    module.ClipEmbed("cpu", download_root=tmp_path)
    assert requested == [{"device": "cpu", "download_root": tmp_path}]
