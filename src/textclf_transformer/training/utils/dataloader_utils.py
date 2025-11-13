import torch
from typing import Optional
from torch.serialization import add_safe_globals
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Any, Mapping, Literal


def make_collate_trim_to_longest(
    pad_is_true_mask: bool = True,
    drop_labels_if_present: bool = False,
    max_seq_len: Optional[int] = None,
):
    """Create a collate function that trims each batch to its longest real sequence.

    The returned callable expects batch items shaped as:
    - ``(input_ids, attention_mask[, labels])`` where
      ``attention_mask`` is boolean. By default, ``True`` denotes PAD tokens
      and ``False`` denotes real tokens (set ``pad_is_true_mask=False`` if the
      mask semantics are inverted).
    - If labels are present in the dataset but ``drop_labels_if_present`` is
      ``True``, labels are dropped from the returned batch.

    Behavior:
        1) Stack items along the batch dimension.
        2) Compute real sequence lengths by counting non-PAD positions according to
           ``pad_is_true_mask``.
        3) Trim ``input_ids`` and ``attention_mask`` to the maximum real length
           found in the batch.
        4) Optionally cap the length to ``max_seq_len`` when provided.
        5) Return ``(input_ids, attention_mask[, labels])`` where labels are
           included only when present and not dropped.

    Args:
        pad_is_true_mask: If ``True``, ``attention_mask=True`` marks PAD tokens.
            If ``False``, ``True`` marks real tokens.
        drop_labels_if_present: If ``True``, omit labels even if present in items.
        max_seq_len: Optional hard cap on the trimmed sequence length.

    Returns:
        Callable: ``collate(batch)`` that outputs either ``(ids, mask)`` or
        ``(ids, mask, labels)`` depending on ``drop_labels_if_present`` and the
        presence of labels in the input batch.
    """
    def collate(batch):
        has_labels = len(batch[0]) == 3

        input_ids = torch.stack([b[0] for b in batch], dim=0)
        attention_mask = torch.stack([b[1] for b in batch], dim=0)

        if pad_is_true_mask:
            real_lens = input_ids.size(1) - attention_mask.sum(dim=1)
        else:
            real_lens = attention_mask.sum(dim=1)

        max_len = int(real_lens.max().item()) if input_ids.size(1) > 0 else 0
        if max_seq_len is not None:
            max_len = min(max_len, int(max_seq_len))
        if max_len <= 0:
            max_len = 1

        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        if has_labels and not drop_labels_if_present:
            labels = torch.stack([b[2] for b in batch], dim=0)
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask

    return collate


def load_dataset(pt_path: str | Path):
    """Load a serialized PyTorch dataset (e.g., ``TensorDataset``) with safe globals."""
    add_safe_globals([TensorDataset])
    return torch.load(pt_path, weights_only=False)


def get_data_loader_from_cfg(cfg: dict[str, Any], kind_ds: Literal["train", "val", "test"], mode: Literal['pretraining', 'finetuning']):
    """Instantiate a ``DataLoader`` for the dataset described under ``cfg['data'][kind_ds]["dataset_path"]``. """
    dataset_path = cfg["data"].get(f"{kind_ds}", {}).get("dataset_path", None)
    if not dataset_path:
        return None
    ds = load_dataset(dataset_path)
    max_seq_len = cfg["architecture"]["max_sequence_length"]
    batch_size = cfg["training"]["batch_size"]

    if mode == 'finetuning':
        collate_fn = make_collate_trim_to_longest(
            drop_labels_if_present=False, max_seq_len=max_seq_len)
    else:
        collate_fn = make_collate_trim_to_longest(
            drop_labels_if_present=True, max_seq_len=max_seq_len)

    shuffle = (kind_ds == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
