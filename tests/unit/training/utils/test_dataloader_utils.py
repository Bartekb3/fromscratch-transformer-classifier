import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.training.utils.dataloader_utils import (
    get_data_loader_from_cfg,
    load_dataset,
    make_collate_trim_to_longest,
)


def test_collate_trims_to_longest_with_pad_true_keeps_labels():
    collate = make_collate_trim_to_longest(pad_is_true_mask=True, drop_labels_if_present=False)
    # Two sequences of length 5 with different numbers of PADs (True = PAD)
    batch = [
        (torch.tensor([1, 2, 3, 0, 0]), torch.tensor([False, False, False, True, True]), torch.tensor(9)),
        (torch.tensor([4, 5, 0, 0, 0]), torch.tensor([False, False, True, True, True]), torch.tensor(8)),
    ]

    ids, mask, labels = collate(batch)

    assert ids.shape == (2, 3)  # trim to max real length (3 tokens)
    assert mask.shape == (2, 3)
    assert torch.equal(labels, torch.tensor([9, 8]))
    assert torch.equal(ids[0], torch.tensor([1, 2, 3]))
    assert torch.equal(ids[1], torch.tensor([4, 5, 0]))


def test_collate_supports_inverse_mask_and_drops_labels_with_max_len():
    collate = make_collate_trim_to_longest(
        pad_is_true_mask=False, drop_labels_if_present=True, max_seq_len=2
    )
    # Here True marks real tokens; longest real len is 3 but capped to 2
    batch = [
        (torch.tensor([7, 7, 7]), torch.tensor([True, True, False]), torch.tensor(1)),
        (torch.tensor([9, 9, 9]), torch.tensor([True, True, True]), torch.tensor(2)),
    ]

    ids, mask = collate(batch)

    assert ids.shape == (2, 2)
    assert mask.shape == (2, 2)
    assert torch.equal(ids[0], torch.tensor([7, 7]))
    assert torch.equal(mask[0], torch.tensor([True, True]))


def test_get_data_loader_from_cfg_builds_loader_for_modes(tmp_path: Path):
    # Create and save a small TensorDataset (input_ids, attention_mask, labels)
    dataset = TensorDataset(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        torch.tensor([[False, False, True], [False, False, False]]),
        torch.tensor([0, 1]),
    )
    ds_path = tmp_path / "ds.pt"
    torch.save(dataset, ds_path)

    cfg = {
        "data": {
            "train": {"dataset_path": str(ds_path)},
        },
        "tokenizer": {"max_length": 4},
        "training": {"batch_size": 2},
    }

    loader_finetune = get_data_loader_from_cfg(cfg, kind_ds="train", mode="finetuning")
    loader_pretrain = get_data_loader_from_cfg(cfg, kind_ds="train", mode="pretraining")

    assert loader_finetune is not None and loader_pretrain is not None

    batch_ft = next(iter(loader_finetune))
    assert len(batch_ft) == 3  # ids, mask, labels
    assert batch_ft[0].shape[1] == 3  # trimmed to longest real len (no max cap)

    batch_pt = next(iter(loader_pretrain))
    assert len(batch_pt) == 2  # labels dropped for pretraining mode
    assert batch_pt[0].shape[1] == 3

    # Missing dataset path should return None
    cfg["data"]["val"] = {}
    assert get_data_loader_from_cfg(cfg, kind_ds="val", mode="finetuning") is None


def test_load_dataset_restores_tensor_dataset(tmp_path: Path):
    original = TensorDataset(torch.tensor([1, 2]), torch.tensor([3, 4]))
    path = tmp_path / "tensor_ds.pt"
    torch.save(original, path)

    loaded = load_dataset(path)

    assert isinstance(loaded, TensorDataset)
    assert len(loaded.tensors) == 2
    assert torch.equal(loaded.tensors[0], original.tensors[0])
