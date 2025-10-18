import torch
from typing import Optional


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


def collate_for_pretraining(pad_is_true_mask: bool = True, max_seq_len: Optional[int] = None):
    """Factory for a pretraining collate that trims to longest and drops labels.

    Args:
        pad_is_true_mask: See ``make_collate_trim_to_longest``.
        max_seq_len: Optional hard cap on the trimmed sequence length.

    Returns:
        Callable configured as ``drop_labels_if_present=True``.
    """
    return make_collate_trim_to_longest(
        pad_is_true_mask=pad_is_true_mask,
        drop_labels_if_present=True,
        max_seq_len=max_seq_len,
    )


def collate_for_classification(pad_is_true_mask: bool = True, max_seq_len: Optional[int] = None):
    """Factory for a classification collate that trims to longest and keeps labels.

    Args:
        pad_is_true_mask: See ``make_collate_trim_to_longest``.
        max_seq_len: Optional hard cap on the trimmed sequence length.

    Returns:
        Callable configured as ``drop_labels_if_present=False``.
    """
    return make_collate_trim_to_longest(
        pad_is_true_mask=pad_is_true_mask,
        drop_labels_if_present=False,
        max_seq_len=max_seq_len,
    )
