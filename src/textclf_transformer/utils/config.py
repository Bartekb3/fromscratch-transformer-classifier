import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple
import random
import os
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for reproducibility across Python, NumPy, and PyTorch.

    This function seeds:
        - Python's ``random`` module,
        - NumPy,
        - PyTorch CPU and all available CUDA devices,
        - the ``PYTHONHASHSEED`` environment variable.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _import_module_from_path(py_path: str):
    """Dynamically import a Python module from a ``.py`` file path.

    Args:
        py_path: Filesystem path to a Python source file.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module spec or loader cannot be created.
    """
    py_path = str(py_path)
    spec = importlib.util.spec_from_file_location(
        "user_tokenizer_wrapper", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie mogę załadować modułu z pliku: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tokenizer_wrapper_from_cfg(tok_cfg: Dict[str, Any]):
    """Load a tokenizer wrapper class from a file and initialize it from disk.

    The configuration must provide:
        - ``wrapper_path``: path to a Python file defining ``WordPieceTokenizerWrapper``,
        - ``vocab_dir``: directory with tokenizer assets to be passed to ``wrapper.load(...)``.

    After loading, the function returns both the wrapper instance and the underlying
    Hugging Face tokenizer exposed at ``wrapper.tokenizer``.

    Args:
        tok_cfg: Tokenizer configuration dictionary.

    Returns:
        Tuple of ``(wrapper, hf_tokenizer)``.

    Raises:
        AttributeError: If the module does not define ``WordPieceTokenizerWrapper``.
        RuntimeError: If the wrapper does not set ``tokenizer`` after ``load()``.
    """
    wrapper_path = tok_cfg["wrapper_path"]
    vocab_dir = tok_cfg["vocab_dir"]

    module = _import_module_from_path(wrapper_path)
    if not hasattr(module, "WordPieceTokenizerWrapper"):
        raise AttributeError(
            f"Plik {wrapper_path} nie definiuje klasy WordPieceTokenizerWrapper."
        )
    WrapperCls = getattr(module, "WordPieceTokenizerWrapper")
    wrapper = WrapperCls()
    wrapper.load(vocab_dir)
    hf_tok = wrapper.tokenizer
    if hf_tok is None:
        raise RuntimeError(
            "Wrapper nie ustawił atrybutu `tokenizer` po `load()`.")
    return wrapper, hf_tok


def vocab_and_pad_from_tokenizer(hf_tok) -> Tuple[int, int]:
    """Extract vocabulary size and PAD token id from a Hugging Face tokenizer.

    Args:
        hf_tok: A tokenizer object exposing ``vocab_size`` and ``pad_token_id``.

    Returns:
        A tuple ``(vocab_size, pad_id)``; if ``pad_token_id`` is ``None``, ``0`` is used.
    """
    vocab_size = hf_tok.vocab_size
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else 0
    return vocab_size, pad_id


def arch_kwargs_from_cfg(arch_cfg: Dict[str, Any], hf_tok) -> Dict[str, Any]:
    """Build model constructor keyword arguments from architecture config and tokenizer.

    This helper reads required fields from ``arch_cfg`` and augments them with
    derived values from the tokenizer (vocab size and default PAD id). It also
    flattens attention-related options for ease of model construction.

    Expected keys in ``arch_cfg`` include:
        - ``max_sequence_length``, ``embedding_dim``, ``num_layers``,
          ``mlp_size``, ``mlp_dropout``, ``pos_encoding``,
          ``embedding_dropout``, optionally ``pad_token_id``.
        - ``attention``: a dict with keys:
            - ``kind`` (defaults to ``"mha"``),
            - ``mha`` (dict) with ``num_heads``, ``attn_dropout``,
              ``mha_out_dropout``, ``projection_bias``.

    Args:
        arch_cfg: Architecture configuration dictionary.
        hf_tok: Tokenizer used to obtain ``vocab_size`` and default ``pad_token_id``.

    Returns:
        A dictionary of keyword arguments suitable for initializing the model.
    """
    vocab_size, pad_id_from_tok = vocab_and_pad_from_tokenizer(hf_tok)
    attn = arch_cfg['attention']
    kind = attn['kind']

    pad_token_id = arch_cfg.get("pad_token_id", pad_id_from_tok)

    kw = dict(
        vocab_size=vocab_size,
        max_sequence_length=arch_cfg["max_sequence_length"],
        embedding_dim=arch_cfg["embedding_dim"],
        num_layers=arch_cfg["num_layers"],
        mlp_size=arch_cfg["mlp_size"],
        mlp_dropout=arch_cfg["mlp_dropout"],
        pos_encoding=arch_cfg["pos_encoding"],
        embedding_dropout=arch_cfg["embedding_dropout"],
        pad_token_id=pad_token_id,
        attention_kind=kind,
        num_heads=attn["num_heads"],
        attn_dropout=attn["attn_dropout"],
        mha_out_dropout=attn["attn_out_dropout"],
        mha_projection_bias=attn["projection_bias"]
    )

    kw['attention_params'] = attn[f"{kind}"]

    return kw
