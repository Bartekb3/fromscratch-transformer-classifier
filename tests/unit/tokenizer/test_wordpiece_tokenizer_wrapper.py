from pathlib import Path

import pandas as pd
import pytest
import torch

from textclf_transformer.tokenizer.wordpiece_tokenizer_wrapper import (
    WordPieceTokenizerWrapper,
)


@pytest.fixture(scope="module")
def trained_wrapper(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("wordpiece")
    training_path = base_dir / "train.txt"
    training_path.write_text(
        "hello world\nthis is a test\nanother line\n", encoding="utf-8"
    )

    tok_dir = base_dir / "tok"
    wrapper = WordPieceTokenizerWrapper()
    wrapper.train(
        tokenizer_dir=str(tok_dir),
        input=str(training_path),
        vocab_size=50,
        min_frequency=1,
    )
    wrapper.load(str(tok_dir))
    return wrapper, tok_dir, training_path


def test_load_missing_directory_raises():
    wrapper = WordPieceTokenizerWrapper()
    with pytest.raises(FileNotFoundError):
        wrapper.load("nonexistent_tokenizer_dir")


def test_train_and_encode_from_file(trained_wrapper):
    wrapper, tok_dir, training_path = trained_wrapper

    assert (tok_dir / "vocab.txt").exists()
    assert (tok_dir / "tokenizer.json").exists()

    ds = wrapper.encode(
        input=str(training_path),
        labels=[0, 1, 2],
        max_length=10,
    )
    input_ids, attention_mask, labels = ds.tensors

    assert input_ids.shape == (3, 10)
    assert attention_mask.shape == (3, 10)
    assert attention_mask.dtype == torch.bool
    assert labels.dtype == torch.long
    assert labels.tolist() == [0, 1, 2]
    assert attention_mask.any()  # padded positions are marked as True


def test_encode_requires_loaded_tokenizer():
    wrapper = WordPieceTokenizerWrapper()
    with pytest.raises(AssertionError):
        wrapper.encode("hello world", max_length=4)


def test_encode_returns_dict_for_raw_texts(trained_wrapper):
    wrapper, _, _ = trained_wrapper

    encoded = wrapper.encode(
        input=["custom sentence"], max_length=6, return_type="dict"
    )

    assert set(encoded.keys()) == {"input_ids", "attention_mask"}
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["attention_mask"].dtype == torch.bool


def test_encode_pandas_validations_and_labels(trained_wrapper):
    wrapper, _, _ = trained_wrapper

    with pytest.raises(TypeError):
        wrapper.encode_pandas(["bad"], text_col="text", max_length=4)

    df_missing_text = pd.DataFrame({"other": ["x"]})
    with pytest.raises(ValueError):
        wrapper.encode_pandas(df_missing_text, text_col="text", max_length=4)

    df = pd.DataFrame({"text": ["hi", "there"], "label": [1, 0]})
    ds = wrapper.encode_pandas(df, text_col="text", label_col="label", max_length=8)
    input_ids, attention_mask, labels = ds.tensors
    assert input_ids.shape == (2, 8)
    assert attention_mask.dtype == torch.bool
    assert labels.tolist() == [1, 0]

    with pytest.raises(ValueError):
        wrapper.encode_pandas(df, text_col="text", label_col="missing", max_length=4)


def test_mask_input_for_mlm_masks_non_special_tokens(trained_wrapper):
    wrapper, _, _ = trained_wrapper

    encoded = wrapper.tokenizer(
        "hello world",
        padding="max_length",
        max_length=8,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    special_mask = wrapper.tokenizer.get_special_tokens_mask(
        input_ids[0].tolist(), already_has_special_tokens=True
    )

    torch.manual_seed(0)
    masked, labels = wrapper.mask_input_for_mlm(
        input_ids, mask_p=1.0, mask_token_p=0.9, random_token_p=0.0
    )
    mask_token_id = wrapper.tokenizer.mask_token_id

    for idx, is_special in enumerate(special_mask):
        if is_special:
            assert labels[0, idx].item() == -100
            assert masked[0, idx].item() == input_ids[0, idx].item()
        else:
            assert labels[0, idx].item() == input_ids[0, idx].item()
            assert masked[0, idx].item() == mask_token_id

    with pytest.raises(AssertionError):
        wrapper.mask_input_for_mlm(
            input_ids, mask_token_p=0.9, random_token_p=0.2
        )
