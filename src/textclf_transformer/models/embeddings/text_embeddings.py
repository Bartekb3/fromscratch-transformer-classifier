import torch
from torch import nn
from ..consts import LN_EPS
from .positional_encodings import LearnedPositionalEmbedding, SinusoidalPositionalEncoding
from .rotary import build_rope_cache 


class TransformerTextEmbeddings(nn.Module):
    """
    Token + position (+ optional token-type) embeddings with LayerNorm and Dropout.

    This module composes token embeddings, absolute positional information
    (either learned or sinusoidal), and optionally token-type (segment) embeddings
    into a single representation. The summed embeddings are then normalized with
    LayerNorm and regularized with Dropout. Optionally, representations at PAD
    positions are zeroed out for cosmetic cleanliness (attention masking should
    still be applied elsewhere).

    Initialization:
        - `word_embeddings.weight` ~ Xavier uniform.
        - `token_type_embeddings.weight` ~ Xavier uniform (if enabled).
        - Learned positional embeddings perform their own initialization within
          `LearnedPositionalEmbedding`.
        - If `pad_token_id` is provided, the PAD row in `word_embeddings` is zeroed.

    Args:
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Model/embedding dimension `D`.
        max_position_embeddings (int): Maximum supported sequence length.
        type_vocab_size (int): Number of token-type (segment) IDs. If 0, no segments are used.
        pos_encoding (str): Either `"learned"` (BERT-style; default) or `"sinusoidal"`.
        embedding_dropout (float): Dropout probability applied after LayerNorm.
        pad_token_id (int | None): PAD token id; if provided, its embedding row is zeroed.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pos_encoding: str = "learned",
        embedding_dropout: float = 0.1,
        pad_token_id: int | None = 0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)

        # Token type embeddings (opcjonalnie)
        self.use_token_type = bool(type_vocab_size)
        if self.use_token_type:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_dim)
        else:
            self.token_type_embeddings = None

        # Positional embeddings/encodings
        pos_encoding = pos_encoding.lower()
        self.pos_kind = pos_encoding
        if pos_encoding == "learned":
            self.position = LearnedPositionalEmbedding(max_position_embeddings, embedding_dim)
        elif pos_encoding == "sinusoidal":
            self.position = SinusoidalPositionalEncoding(embedding_dim, max_position_embeddings)
        elif pos_encoding == "rope":
            # RoPE uses rotary on (Q, K) inside attention; no absolute positions are added here.
            self.position = None
        else:
            raise ValueError(f"Unknown pos_encoding '{pos_encoding}'. Use 'learned' or 'sinusoidal'.")

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=LN_EPS)
        self.dropout = nn.Dropout(embedding_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize word/segment embeddings and zero the PAD row (if provided)."""
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        if self.use_token_type:
            nn.init.xavier_uniform_(self.token_type_embeddings.weight)
        pad_idx = self.word_embeddings.padding_idx
        if pad_idx is not None:
            with torch.no_grad():
                self.word_embeddings.weight[pad_idx].zero_()


    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        """
        Compute text embeddings by summing token, positional, and (optionally) token-type embeddings,
        followed by LayerNorm and Dropout. Optionally zero-out PAD positions in the final output.

        Args:
            input_ids (LongTensor): Tensor of shape `(B, N)` with token ids.
            token_type_ids (LongTensor, optional): Tensor of shape `(B, N)` with segment ids
                (e.g., 0/1/...). If `None` and segments are enabled, a zero tensor is used.
            position_ids (LongTensor, optional): Tensor of shape `(B, N)` with absolute positions.
                For `'learned'` encodings, if `None`, positions default to `arange(N)` broadcast to `(B, N)`.
                For `'sinusoidal'`, `position_ids` is ignored.

        Returns:
            Tensor: Embeddings of shape `(B, N, D)`, layer-normalized and dropout-applied.
        """
        B, N = input_ids.shape
        device = input_ids.device

        word_emb = self.word_embeddings(input_ids)  # (B, N, D)

        # Positions
        if self.pos_kind == "learned":
            if position_ids is None:
                position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
            pos_emb = self.position(position_ids)  # (B, N, D)
            x = word_emb + pos_emb
        elif self.pos_kind == "sinusoidal":
            pos = self.position(seq_len=N, device=device)  # (N, D)
            pos_emb = pos.unsqueeze(0).expand(B, N, -1)
            x = word_emb + pos_emb
        elif self.pos_kind == "rope":
            # No absolute positions added here (RoPE will be applied to Q/K in attention).
            x = word_emb
        else:
            raise RuntimeError("Unsupported positional encoding kind.")



        if self.use_token_type:
            if token_type_ids is None:
                token_type_ids = torch.zeros((B, N), dtype=torch.long, device=device)
            else:
                token_type_ids = token_type_ids.to(device=device, dtype=torch.long)
            x = x + self.token_type_embeddings(token_type_ids)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
