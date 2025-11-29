import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from textclf_transformer.models.pooling.pooling import (
    ClsTokenPooling,
    MeanPooling,
    MaxPooling,
    MinPooling,
)


def test_cls_token_pooling_returns_first_token_and_ignores_mask():
    """Ensures CLS pooling always returns the first token embedding and that provided masks have no effect, so pooled vectors equal x[:,0,:] regardless of padding."""
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, True, False, True], [True, True, False, False]])
    pooling = ClsTokenPooling()

    out_nomask = pooling(x)
    out_mask = pooling(x, key_padding_mask=mask)

    expected = x[:, 0, :]
    assert torch.allclose(out_nomask, expected)
    assert torch.allclose(out_mask, expected)


def test_mean_pooling_without_mask_matches_torch_mean():
    """When no mask is provided, mean pooling should exactly equal torch.mean over the sequence dimension, so outputs mirror a straightforward average."""
    x = torch.randn(3, 5, 2)
    pooling = MeanPooling()

    out = pooling(x)
    expected = x.mean(dim=1)

    assert torch.allclose(out, expected)


def test_mean_pooling_excludes_masked_positions_and_clamps_denominator():
    """Checks that masked tokens are dropped from the mean and fully masked batches clamp the denominator to avoid NaNs, producing intuitive averages and zeros where everything is padded."""
    x = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ]
    )
    mask = torch.tensor([[False, True, False], [True, True, False]])
    pooling = MeanPooling()

    out = pooling(x, key_padding_mask=mask)
    expected = torch.tensor(
        [
            [(1.0 + 3.0) / 2, (1.0 + 3.0) / 2],  # positions 0 and 2 valid
            [6.0, 6.0],  # only last position valid; denom clamps to 1
        ]
    )

    assert torch.allclose(out, expected)

    all_mask = torch.ones(1, 3, dtype=torch.bool)
    out_all_mask = pooling(x[:1], key_padding_mask=all_mask)
    assert torch.allclose(out_all_mask, torch.zeros_like(out_all_mask))


def test_max_pooling_masks_out_padded_tokens():
    """Verifies max pooling ignores masked positions (treated as -inf) while matching vanilla max when unmasked and returning -inf surrogates when everything is masked."""
    x = torch.tensor([[[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]]])
    mask = torch.tensor([[False, True, False]])  # mask out the largest value
    pooling = MaxPooling()

    out = pooling(x, key_padding_mask=mask)
    expected = torch.tensor([[3.0, 3.0]])

    assert torch.allclose(out, expected)

    out_nomask = pooling(x, key_padding_mask=None)
    assert torch.allclose(out_nomask, x.max(dim=1).values)

    all_mask = torch.ones(1, 3, dtype=torch.bool)
    out_all_mask = pooling(x, key_padding_mask=all_mask)
    neg_inf = torch.full_like(out_all_mask, torch.finfo(x.dtype).min)
    assert torch.allclose(out_all_mask, neg_inf)


def test_min_pooling_masks_out_padded_tokens():
    """Verifies min pooling ignores masked positions (treated as +inf) and behaves like torch.min without masks, returning +inf surrogates for all-mask cases."""
    x = torch.tensor([[[1.0, 1.0], [-5.0, -5.0], [3.0, 3.0]]])
    mask = torch.tensor([[False, True, False]])  # mask out the smallest value
    pooling = MinPooling()

    out = pooling(x, key_padding_mask=mask)
    expected = torch.tensor([[1.0, 1.0]])

    assert torch.allclose(out, expected)

    out_nomask = pooling(x, key_padding_mask=None)
    assert torch.allclose(out_nomask, x.min(dim=1).values)

    all_mask = torch.ones(1, 3, dtype=torch.bool)
    out_all_mask = pooling(x, key_padding_mask=all_mask)
    pos_inf = torch.full_like(out_all_mask, torch.finfo(x.dtype).max)
    assert torch.allclose(out_all_mask, pos_inf)
