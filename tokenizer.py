from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
from typing import Sequence
from tokenizers.processors import TemplateProcessing


class WordPieceTokenizerWrapper:
    """
    A wrapper around Hugging Face's `BertWordPieceTokenizer` and `BertTokenizerFast`
    for training, saving, loading, and encoding datasets.

    Example: 
    >>> tokenizer = WordPieceTokenizerWrapper()
    >>> tokenizer.train(tokenizer_dir='my_tokenizer', input='text1.txt')
    >>>
    >>> num_lines = sum(1 for _ in open("text2.txt"))
    >>> labels = np.random.randint(0, num_classes, size=num_lines)
    >>> ds = tokenizer.encode(
    ...     tokenizer_dir='my_tokenizer',
    ...     input='text2.txt',
    ...     labels=labels,
    ...     max_length=8
    ... )
    >>> print(ds[0])
    (tensor([ 2,  42, 142, 24,  37,  3, 0, 0]),
        tensor([False, False, False, False, False, False, True, True]),
        tensor(1))
    """

    def __init__(self):
        self.tokenizer = None

    def train(self,
              tokenizer_dir: str,
              input: str | list[str],
              vocab_size: int = 30000,
              min_frequency: int = 2,
              special_tokens: list[str] = [
                  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
              include_cls: bool = True,
              include_sep: bool = True
              ) -> None:
        """
        Train a WordPiece tokenizer on the given text corpus and save it to disk.

        Args:
            tokenizer_dir (str): Directory where the tokenizer files will be saved.
            input (str | list[str]): Path or list of paths to training text files.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 30,000.
            min_frequency (int, optional): Minimum frequency for tokens to be included in vocabulary. Defaults to 2.
            special_tokens (list[str], optional): Special tokens to include. Defaults to
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].
            include_cls (bool): Whether to include the [CLS] token at the beginning of each encoded sequence.
            include_sep (bool): Whether to include the [SEP] token at the end of each encoded sequence (before padding).
        """

        tokenizer = BertWordPieceTokenizer(
            lowercase=True,
            strip_accents=True
        )

        tokenizer.train(
            files=input,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )

        if include_cls and include_sep:
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                                ("[SEP]", tokenizer.token_to_id("[SEP]"))],
            )
        elif include_sep:
            tokenizer.post_processor = TemplateProcessing(
                single="$A [SEP]",
                special_tokens=[("[SEP]", tokenizer.token_to_id("[SEP]"))],
            )
        elif include_cls:
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A",
                special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]"))],
            )


        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(str(tokenizer_dir))
        tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    def load(self, tokenizer_dir: str) -> None:
        tokenizer_dir = Path(tokenizer_dir)
        if not tokenizer_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer directory not found: {tokenizer_dir}")
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(tokenizer_dir), use_fast=True)

    def encode(self,
               tokenizer_dir: str,
               input: str | list[str],
               max_length: int,
               labels: Sequence[int] | None = None) -> TensorDataset:
        """
        Encode text files into a `TensorDataset` suitable for PyTorch models.

        Args:
            tokenizer_dir (str): Path to the tokenizer directory including `vocab.txt` and `tokenizer.json`.
            input (str | list[str]): Path or list of paths to text files. Each line is treated as one example.
            labels (Sequence[int]): Labels aligned with the input texts.
            max_length (int): Maximum sequence length for padding/truncation.

        Returns:
            TensorDataset: A dataset of 3 tensors, each with length equal to the number of text examples:
                - input_ids: (N, max_length)
                    Token IDs for each example, padded/truncated to max_length.
                - attention_mask: (N, max_length)
                    Boolean mask where True marks a padding position and False
                    marks a real token. ( Note: this is different from Hugging Face's
                    default convention where 1 = token, 0 = padding).
                - labels: (N,) (if provided in args)
                    The provided labels (used in fine-tuning), converted to a tensor.
        """

        if self.tokenizer is None:
            self.load(tokenizer_dir)
        tok = self.tokenizer

        if isinstance(input, str):
            input = [input]

        texts = []
        for fname in input:
            with open(fname, "r", encoding="utf-8") as f:
                texts.extend([line.strip() for line in f if line.strip()])

        encoded = tok(texts, padding="max_length",
                      max_length=max_length,
                      truncation=True,
                      return_tensors="pt",
                      return_token_type_ids=False,
                      add_special_tokens=True)
        encoded['attention_mask'] = encoded['attention_mask'] == 0

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if labels is not None:  # labels is optional
            labels = torch.tensor(labels)
            ds = TensorDataset(input_ids, attention_mask, labels)
        else:
            ds = TensorDataset(input_ids, attention_mask)
        return ds

    def mask_input_for_mlm(self,
                   input_ids: torch.LongTensor,
                   mask_p: float = 0.15,
                   mask_token_p: float = 0.8,
                   random_token_p: float = 0.1
                   ):
        """
        Create masked inputs and labels for Masked Language Modeling (MLM).
        Applies the standard BERT masking recipe: draw `mask_p` of the tokens to
        predict, replace a fraction of the selected positions with `[MASK]`, swap a
        smaller portion with random vocabulary tokens, and leave the rest unchanged.
        Args:
            input_ids (torch.LongTensor): Tensor of token ids shaped `(B, N)` where
                `B` is the batch size and `N` is the sequence length.
            mask_p (float): Overall probability of selecting a token for prediction.
            mask_token_p (float): Probability of replacing a selected token with
                the `[MASK]` token.
            random_token_p (float): Probability of replacing a selected token with a
                random token sampled from the vocabulary. The remainder is left as-is.
        Returns:
            tuple[torch.LongTensor, torch.LongTensor]:
                - Masked `input_ids` tensor shaped `(B, N)` to feed into the model.
                - Labels tensor shaped `(B, N)` where tokens not chosen for masking
                  are set to `-100` to be ignored by the loss function.
        ## Note:
            The tokenizer must be loaded - meaning you have to use `self.encode()` fuction first.
            Otherwise, the tokenizer will not know which input ids correspond to which tokens.
        """
        assert mask_token_p + random_token_p <= 1.0
        assert self.tokenizer is not None, "`self.encode()` must be used before masking input."

        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, mask_p)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        input_ids_masked = input_ids.clone()

        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, mask_token_p)).bool() & masked_indices
        input_ids_masked[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        p = random_token_p / (1 - mask_token_p)
        indices_random = torch.bernoulli(torch.full(
            labels.shape, p)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids_masked[indices_random] = random_words[indices_random]

        return input_ids_masked, labels
