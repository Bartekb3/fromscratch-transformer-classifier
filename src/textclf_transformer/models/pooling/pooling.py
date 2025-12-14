import torch
from torch import nn

# class ClsTokenPooling(nn.Module):
#     """
#     Pooling layer that returns the representation of the first token in the sequence
#     (conventionally the [CLS] token).

#     This module is mask-agnostic: the optional key padding mask is accepted for API
#     consistency but ignored.

#     Forward:
#         Input:
#             - x (Tensor): Encoder outputs of shape (B, N, D)
#             - key_padding_mask (Tensor, optional): Boolean mask (B, N); unused
#         Output:
#             - Tensor: Pooled sequence representation of shape (B, D), taken from x[:, 0, :]
#     """
#     def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
#         """
#         Args:
        
#             x (Tensor): Encoder outputs of shape (B, N, D).
#             key_padding_mask (Tensor, optional): Boolean mask of shape (B, N);
#                 accepted but unused in this pooling variant.

#         Returns:
#             Tensor: Tensor of shape (B, D) corresponding to the first token representation.
#         """
#         return x[:, 0, :]

class MeanPooling(nn.Module):
    """
    Mean pooling across the sequence length, with optional masking of PAD positions.

    If a key padding mask is provided, PAD positions (True in the mask) are excluded
    from the mean by zeroing their contributions and dividing by the number of valid
    tokens per example.
    """
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            x (Tensor): Encoder outputs of shape (B, N, D).
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N) where
                True marks PAD tokens to be excluded from the mean.

        Returns:
            Tensor: Mean-pooled representation of shape (B, D).
        """
        if key_padding_mask is None:
            return x.mean(dim=1)
        mask = ~key_padding_mask  # True for valid (non-PAD)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)  # (B,1)
        masked = x * mask.unsqueeze(-1).to(x.dtype)
        return masked.sum(dim=1) / denom

class ClsTokenPooling(nn.Module):
    """
    Pooling layer that calculates the mean of embeddings at positions divisible by 512
    (0, 512, 1024, ...). This strategy is useful for long sequences formed by
    concatenating multiple chunks, where each chunk (e.g., 512 tokens) starts with a
    [CLS] token.

    If a key padding mask is provided, PAD positions (True in the mask) at these
    indices are excluded from the mean.
    """
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            x (Tensor): Encoder outputs of shape (B, N, D).
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N) where
                True marks PAD tokens to be excluded.

        Returns:
            Tensor: Pooled representation of shape (B, D).
        """
        # Select tokens at indices 0, 512, 1024, ...
        x_sub = x[:, ::512]  # (B, N_sub, D)

        if key_padding_mask is None:
            return x_sub.mean(dim=1)

        # Select mask values at the same indices
        mask_sub = key_padding_mask[:, ::512]  # (B, N_sub)
        
        # Logic mirroring MeanPooling for the subset
        valid_mask = ~mask_sub  # True for valid (non-PAD)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        masked_x = x_sub * valid_mask.unsqueeze(-1).to(x.dtype)
        
        return masked_x.sum(dim=1) / denom

class MaxPooling(nn.Module):
    """
    Max pooling across the sequence length with optional PAD masking.

    When a key padding mask is provided, PAD positions (True in the mask) are set to
    the minimum finite value representable by the tensor dtype (i.e., -inf surrogate)
    prior to the max reduction along the sequence dimension.
    """
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            x (Tensor): Encoder outputs of shape (B, N, D).
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N) where
                True marks PAD tokens to be ignored in the max.

        Returns:
            Tensor: Max-pooled representation of shape (B, D).
        """
        if key_padding_mask is None:
            return x.max(dim=1).values
        neg_inf = torch.finfo(x.dtype).min
        masked = x.masked_fill(key_padding_mask.unsqueeze(-1), neg_inf)
        return masked.max(dim=1).values

class MinPooling(nn.Module):
    """
    Min pooling across the sequence length with optional PAD masking.

    When a key padding mask is provided, PAD positions (True in the mask) are set to
    the maximum finite value representable by the tensor dtype (i.e., +inf surrogate)
    prior to the min reduction along the sequence dimension.
    """
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            x (Tensor): Encoder outputs of shape (B, N, D).
            key_padding_mask (Tensor, optional): Boolean mask of shape (B, N) where
                True marks PAD tokens to be ignored in the min.

        Returns:
            Tensor: Min-pooled representation of shape (B, D).
        """
        if key_padding_mask is None:
            return x.min(dim=1).values
        pos_inf = torch.finfo(x.dtype).max
        masked = x.masked_fill(key_padding_mask.unsqueeze(-1), pos_inf)
        return masked.min(dim=1).values
