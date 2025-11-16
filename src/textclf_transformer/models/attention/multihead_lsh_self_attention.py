import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..embeddings.rotary import apply_rope


BUCKET_PAD_ID = -1


class LSHAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,
        out_dropout: float = 0.0,
        attention_embed_dim: int | None = None,
        num_hashes: int = 4,
        chunk_size: int = 64,
        mask_within_chunks: bool = True,
    ):
        """
        Args:
            embed_dim: Hidden size of the transformer model.
            num_heads: Number of attention heads the hidden size is split into.
            bias: Whether the projection layers include biases.
            attn_dropout: Dropout probability applied to attention weights.
            out_dropout: Dropout probability applied after the output projection.
            attention_embed_dim: Optional projection size for Q/K/V/out. Defaults to
                ``embed_dim`` and must be divisible by ``num_heads``.
            num_hashes: Count of independent LSH rounds used for bucketization.
            chunk_size: Bucket length (tokens) used when performing local attention.
            mask_within_chunks (bool):
                If **True**, queries in a chunk may only attend to keys from the
                same LSH bucket within the 3 x window (intra-bucket attention).
                If **False**, queries may attend to any key within the 3 x window.
        """
        super().__init__()
        assert attention_embed_dim % num_heads == 0, "attention_embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_hashes = num_hashes
        self.chunk_size = chunk_size
        self.dk = embed_dim // num_heads
        self.proj_bias = bias
        self.mask_within_chunks = mask_within_chunks

        # In Reformer, Q and K share the same projection (Q = K)
        self.Uqv = nn.Linear(embed_dim, 2 * attention_embed_dim, bias=self.proj_bias)
        self.Uout = nn.Linear(attention_embed_dim, embed_dim, bias=self.proj_bias)

        self.out_drop = nn.Dropout(out_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for Uqv projection, same as in Pythorch implementation
        nn.init.xavier_uniform_(self.Uqv.weight)
        nn.init.xavier_uniform_(self.Uout.weight)
        if self.proj_bias:
            nn.init.zeros_(self.Uqv.bias)
            nn.init.zeros_(self.Uout.bias)

    def pad_to_even_buckets(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Pads the input so that:
        - sequence length N is a multiple of chunk_size
        - number of buckets (N // chunk_size) is even

        Args:
            x:    [B, N, D]
            mask: [B, N] boolean tensor indicating which tokens are [PAD] (True)

        Returns:
            x_padded:   [B, N', D]
            mask_padded:[B, N'] boolean tensor (True for [PAD] tokens)
            n_pad:      number of padded tokens
        """

        B, N, D = x.shape
        S = self.chunk_size

        # pad to multiple of chunk_size
        pad_to_multiple = (S - (N % S)) % S
        N1 = N + pad_to_multiple

        # pad so number of buckets (N1 // S) is even
        n_buckets = N1 // S
        if n_buckets % 2 != 0:
            pad_to_even_buckets = S
        else:
            pad_to_even_buckets = 0

        total_pad = pad_to_multiple + pad_to_even_buckets

        if total_pad > 0:
            pad_tensor = x.new_zeros(B, total_pad, D)
            x = torch.cat([x, pad_tensor], dim=1)

            mask_pad = torch.ones(
                B, total_pad, device=x.device, dtype=torch.bool)
            mask = torch.cat([mask, mask_pad], dim=1)

        return x, mask, total_pad

    def random_hash(self, x: Tensor, n_buckets: int) -> Tensor:
        B, num_hashes, num_heads, N, dk = x.shape

        # Random hashing matrix R: (H#, H, dk, nb/2)
        R = torch.randn(
            num_hashes, num_heads, dk, n_buckets // 2,
            device=x.device, dtype=x.dtype
        )
        R = R / torch.norm(R, dim=2, keepdim=True)

        # x: (B,H#,H,N,dk)
        # R: (H#,H,dk,n_buckets)
        projections = torch.einsum(
            'BhHNd, hHdn -> BhHNn', x, R)  # x@R

        hash_values = torch.argmax(
            torch.cat([projections, -projections], dim=-1), dim=-1
        )
        # hash: (B,H#,H,N)
        return hash_values

    def get_permutation_from_hash(self, hash_codes: torch.Tensor):
        B, Hh, H, N = hash_codes.shape
        device = hash_codes.device

        # sorting by key keeps original order of hashes
        pos = torch.arange(N, device=device).view(1, 1, 1, N)
        key = hash_codes * (N + 1) + pos

        # perm: (B,H#,H,N) - permutation sorting hashes, used for sorting sequence along N dimension
        perm = torch.argsort(key, dim=-1)

        # used to restore original sequence order
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(dim=-1, index=perm,
                          src=pos.expand(B, Hh, H, N))

        return perm, inv_perm

    def chunk_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        buckets: Tensor,
        valid_mask: Tensor,
        mask_within_chunks: bool
    ) -> Tensor:
        """
        Compute local self-attention over sorted LSH chunks, with optional masking
        between different buckets and lookback/lookahead to neighbour chunks.

        Each query chunk attends to its own keys/values, as well as to keys/values
        from the previous and next chunks. This enables limited context sharing
        across adjacent buckets after LSH sorting.

        Args:
            q (Tensor): (B, Hh, H, N, dk) - sorted 
            k (Tensor): (B, Hh, H, N, dk) - sorted 
            v (Tensor): (B, Hh, H, N, dk) - sorted 
            buckets (Tensor):
                Sorted LSH bucket indices: (B, Hh, H, N).
                Each integer label corresponds to a hash bucket ID assigned to a token.
            valid_mask (Tensor):
                Boolean mask (B, Hh, H, N) indicating which tokens are valid (True).
                When provided, attention scores to padded tokens are masked out and
                outputs for padded queries are zeroed.
            mask_within_chunks (bool):
                If True, mask attention between tokens belonging to different buckets
                within the same chunk (enforces intra-bucket attention only).
                If False, allow full attention within each 3xchunk window.

        Returns:
            Tensor:
                Attention output of shape (B, Hh, H, n_chunks, chunk_size, dk),
                where each chunk has attended to its own, previous, and next chunks.
        """

        B, num_hashes, num_heads, N, dk = q.shape
        chunk_size = self.chunk_size
        n_chunks = N // chunk_size

        q_chunks = q.view(B, num_hashes, num_heads, n_chunks, chunk_size, dk)
        k_chunks = k.view(B, num_hashes, num_heads, n_chunks, chunk_size, dk)
        v_chunks = v.view(B, num_hashes, num_heads, n_chunks, chunk_size, dk)
        b_chunks = buckets.view(B, num_hashes, num_heads, n_chunks, chunk_size)
        valid_chunks = valid_mask.view(
            B, num_hashes, num_heads, n_chunks, chunk_size)

        def _get_next_and_previous_chunk(x_chunks: Tensor, padding_value: int | float | bool) -> Tensor:
            """
            Concatenate each chunk with its previous and next chunks along the sequence dimension.

            Args:
                x_chunks (Tensor): 
                    Input tensor of shape (B, Hh, H, n_chunks, chunk_size, ...),
                padding_value (int | float): 
                    Value that should be used to pad the first and last chunks.
                    • 0.0 for q, k, v tensors (float)
                    • -1  for bucket ID tensors (int)
                    • False for mask tensors (bool)

            Returns:
                Tensor:
                    Concatenated tensor of shape (B, Hh, H, n_chunks, 3 * chunk_size, ...),
                    where each chunk now includes [previous, current, next] chunks.
            """
            shape = list(x_chunks.shape)
            shape[3] = 1

            x_prev = torch.cat([
                x_chunks.new_full(shape, padding_value),
                x_chunks[:, :, :, :-1]
            ], dim=3)
            x_next = torch.cat([
                x_chunks[:, :, :, 1:],
                x_chunks.new_full(shape, padding_value)
            ], dim=3)

            x_concat = torch.cat([x_prev, x_chunks, x_next], dim=4)
            return x_concat

        # look back to previous and next chunk
        k_both = _get_next_and_previous_chunk(k_chunks, 0.0)
        v_both = _get_next_and_previous_chunk(v_chunks, 0.0)
        b_both = _get_next_and_previous_chunk(b_chunks, BUCKET_PAD_ID)
        valid_both = _get_next_and_previous_chunk(valid_chunks, False)

        # Compute attention scores: (B,H#,H, n_chunks, chunk_size, 3*chunk_size)
        scores = torch.matmul(
            q_chunks, k_both.transpose(-2, -1)) / (dk ** 0.5)

        pad_keys = (b_both == BUCKET_PAD_ID).unsqueeze(4).expand_as(scores)
        key_invalid = (~valid_both).unsqueeze(4).expand_as(scores)
        if mask_within_chunks:
            diff_mask = (b_chunks.unsqueeze(-1) != b_both.unsqueeze(4))
            mask = diff_mask | pad_keys | key_invalid
        else:
            mask = pad_keys | key_invalid

        # prevent each token from attending to itself TODO only if lot to attend
        idx = torch.arange(chunk_size, device=mask.device)
        mask[:, :, :, :, idx, chunk_size + idx] = True

        # make mask additive
        dtype = scores.dtype
        neg_large = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
        add_mask = mask.to(dtype=dtype) * neg_large
        scores = scores + add_mask

        # calculate attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        out = torch.matmul(attn_weights, v_both)

        # force zeros for padded tokens
        out = out * valid_chunks.unsqueeze(-1).to(out.dtype)

        return out  # [B, H#, H, n_chunks, chunk_size, dk]

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        rope: dict | None = None
    ) -> Tensor:
        """
        Compute Reformer-style LSH self-attention with chunked local windows
        and 1-back/1-forward context.

        This module:
        1) Pads the sequence so its length is a multiple of `chunk_size` and the
            number of chunks is even (extra tokens are treated as padding and
            removed at the end).
        2) Projects inputs with a shared Q/K projection and a separate V
            projection (Q = K ≠ V), replicated across `num_hashes` hash rounds.
        3) Generates LSH bucket ids per (round, head) and sorts tokens by bucket.
        4) Splits the sorted sequence into fixed-size chunks. Each chunk attends
            to its [previous, current, next] chunks (3 x window). If
            `mask_within_chunks=True`, attention inside the window is further
            restricted to tokens that share the same bucket as the query.
        5) Masks out padded keys and (via row masking & post-mul) produces exact
            zeros for padded queries. Self-attention on the exact diagonal is also
            masked inside the center chunk.
        6) Unsorts the outputs back to the original order, averages across hash
            rounds, applies the output projection and dropout, and finally removes
            any padding that was added in step (1).

        Args:
            x (Tensor):
                Input embeddings of shape **[B, N, D]**, where
                *B* = batch size, *N* = sequence length, *D* = embed_dim.
            padding_mask (Tensor):
                Boolean tensor of shape **[B, N]** with **True** at padded
                positions and **False** at real tokens. Padded tokens are ignored
                as keys and produce zero outputs as queries.
            rope (dict | None): Optional rotary positional cache. Provide ``rope_cos``/``rope_sin``
                tables (broadcastable to (B, H, N, dk)) and optionally ``rope_position_ids`` (B, N)
                to apply RoPE to Q/K before hashing.

        Returns:
            Tensor:
                out vectors of shape **[B, N, D]** in the original token order.
        """

        # additional pad to make number of buckets (N/chunk_size) even
        x, padding_mask_bool, n_pad = self.pad_to_even_buckets(x, padding_mask)
        B, N, D = x.shape
        dk = self.dk

        qv = self.Uqv(x)  # [B, N, 2*D]
        q, v = qv.chunk(2, dim=-1)

        q = q.view(B, N, self.num_heads, dk).transpose(
            1, 2).contiguous()
        v = v.view(B, N, self.num_heads, dk).transpose(
            1, 2).contiguous()

        # Apply RoPE if provided
        if rope is None:
            rope = {}
        rope_cos = rope.get('rope_cos', None)
        rope_sin = rope.get('rope_sin', None)
        rope_position_ids = rope.get('rope_position_ids', None)
        if (rope_cos is not None) and (rope_sin is not None):
            q, _ = apply_rope(q, q, rope_cos, rope_sin, rope_position_ids)

        q = q.unsqueeze(1).expand(B, self.num_hashes,
                                  self.num_heads, N, dk).contiguous()
        v = v.unsqueeze(1).expand(B, self.num_hashes,
                                  self.num_heads, N, dk).contiguous()

        # valid mask - True on 'real' tokens: (B,H#,H,N)
        valid_mask = ~padding_mask_bool[:, None, None, :].expand(
            B, self.num_hashes, self.num_heads, N).contiguous()
        # set pad queries/vals to 0.0
        mask_float = valid_mask.to(dtype=q.dtype).unsqueeze(-1)
        q = q * mask_float
        v = v * mask_float

        # hashing queries
        n_buckets = N // self.chunk_size
        hash_codes = self.random_hash(q, n_buckets)  # (B,H#,H,N)
        # set padding bucket to -1
        hash_codes = hash_codes.masked_fill(~valid_mask, BUCKET_PAD_ID)

        # sort q, v, mask, and buckets according to the order of hash codes
        sort_indices, undo_sort = self.get_permutation_from_hash(hash_codes)
        idx = sort_indices.unsqueeze(-1).expand(B,
                                                self.num_hashes, self.num_heads, N, dk)
        q_sorted = torch.gather(q, dim=3, index=idx).contiguous()
        v_sorted = torch.gather(v, dim=3, index=idx).contiguous()
        buckets = torch.gather(hash_codes, dim=-1, index=sort_indices)
        valid_mask_sorted = torch.gather(
            valid_mask.to(dtype=torch.long), dim=3, index=sort_indices
        ).to(dtype=torch.bool)

        # chunked attention : (B, H#, H, n_chunks, chunk_size, dk)
        ctx = self.chunk_attention(
            q_sorted, q_sorted, v_sorted, buckets,
            valid_mask_sorted,
            mask_within_chunks=self.mask_within_chunks)

        ctx = ctx.contiguous().view(B, self.num_hashes, self.num_heads, N, dk)

        # unsort the outputs
        idx = undo_sort.unsqueeze(-1).expand(B,
                                             self.num_hashes, self.num_heads, N, dk)
        ctx_unsorted = torch.gather(ctx, dim=3, index=idx)

        # mean over hash rounds
        ctx_unsorted = ctx_unsorted.transpose(
            2, 3).reshape(B, self.num_hashes, N, D)
        ctx_mean = ctx_unsorted.mean(dim=1)

        # Final output projection
        out = self.Uout(ctx_mean)
        out = self.out_drop(out)

        if n_pad > 0:
            out = out[:, :-n_pad, :]

        return out  # (B,N,D)
