from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
from typing import Literal, Sequence, Dict, Union, List, Optional
from tokenizers.processors import TemplateProcessing


def _find_project_root(start: Path) -> Optional[Path]:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "textclf_transformer").exists():
            return candidate
    return None


def _resolve_existing_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path.cwd() / path)

        cwd_root = _find_project_root(Path.cwd())
        if cwd_root is not None:
            candidates.append(cwd_root / path)

        module_root = _find_project_root(Path(__file__).resolve())
        if module_root is not None and module_root != cwd_root:
            candidates.append(module_root / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    attempted = "\n".join(f"- {c}" for c in candidates) if candidates else f"- {path}"
    raise FileNotFoundError(
        f"Tokenizer directory not found: {path}\n"
        f"Attempted:\n{attempted}\n"
        "Tip: if running from a notebook in a subfolder, pass an absolute path or a path relative to the repo root."
    )


class WordPieceTokenizerWrapper:
    """
    A wrapper around Hugging Face's `BertWordPieceTokenizer` and `BertTokenizerFast`
    for training, saving, loading, and encoding datasets.

    Example: 
    >>> tokenizer = WordPieceTokenizerWrapper()
    >>> tokenizer.train(tokenizer_dir='my_tokenizer', input='text1.txt')
    >>> # training part can be skipped if you already have 'my_tokenizer' dir with vocab.txt file
    >>> tokenizer.load(tokenizer_dir='my_tokenizer')
    >>>
    >>> num_lines = sum(1 for _ in open("text2.txt"))
    >>> labels = np.random.randint(0, num_classes, size=num_lines)
    >>> ds = tokenizer.encode(
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
        self.tokenizer_dir = None

    def train(self,
              tokenizer_dir: str,
              input: Union[str, List[str]],
              vocab_size: int = 30000,
              min_frequency: int = 2,
              special_tokens: List[str] = [
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

    def load(self, tokenizer_dir: Union[str, Path] = "./BERT_original") -> None:
        tokenizer_dir = _resolve_existing_dir(tokenizer_dir)
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(tokenizer_dir), use_fast=True)

    def encode(self,
               input: Union[str, List[str]],
               max_length: int,
               labels: Optional[Sequence[int]] = None,
               return_type: Literal['dict', 'tensordataset'] = 'tensordataset') -> Union[TensorDataset, Dict]:
        """
        Encode text data from file paths or raw strings and produce tensors ready
        for model consumption.

        Args:
            tokenizer_dir (str): Path to the tokenizer directory including `vocab.txt` and `tokenizer.json`.
            input (str | list[str]): Path or list of paths to text files, or a list of raw text examples.
                Every non-empty line is treated as an individual example.
            labels (Sequence[int], optional): Label sequence aligned with the text examples.
            max_length (int): Maximum sequence length to pad and/or truncate to.
            return_type (Literal['dict', 'tensordataset']): Controls the return format. Use `'dict'` to
                receive the Hugging Face style tokenization dictionary, or `'tensordataset'` for a PyTorch
                `TensorDataset`. Defaults to `'tensordataset'`.

        Returns:
            dict | TensorDataset: When `return_type='dict'`, a dictionary with keys `input_ids`,
            `attention_mask`, and `labels` (present only if `labels` is supplied), each mapped to tensors.
            When `return_type='tensordataset'`, a `TensorDataset` containing `(input_ids, attention_mask)`
            and `(labels,)` appended if provided.
        """

        assert return_type in ['dict', 'tensordataset']
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load() first."

        is_input_str = isinstance(input, str)
        if is_input_str:
            input = [input]

        texts = []
        first_file = Path(input[0])
        try:
            first_file.exists()
            if is_input_str:
                print('[INFO] input is treated as a path to text file')
            else:
                print('[INFO] input is treated as a list of paths to text files')
            for fname in input:
                with open(fname, "r", encoding="utf-8") as f:
                    texts.extend([line.strip() for line in f if line.strip()])
        except:
            print('[INFO] input is treated as a list of input texts')
            texts = input

        encoded = self.tokenizer(texts, padding="max_length",
                                 max_length=max_length,
                                 truncation=True,
                                 return_tensors="pt",
                                 return_token_type_ids=False,
                                 add_special_tokens=True)
        encoded['attention_mask'] = encoded['attention_mask'] == 0

        if labels is not None:
            encoded['labels'] = torch.tensor(labels, dtype=torch.long)

        if return_type == 'dict':
            return encoded

        if 'labels' in encoded:
            return TensorDataset(encoded['input_ids'], encoded['attention_mask'], encoded['labels'])
        else:
            return TensorDataset(encoded['input_ids'], encoded['attention_mask'])

    def encode_pandas(self,
                      df,
                      text_col: str,
                      max_length: int,
                      label_col: Optional[str] = None) -> TensorDataset:
        """
        Encode a pandas DataFrame into a TensorDataset for PyTorch models.

        Args:
            df (pandas.DataFrame): DataFrame containing the data.
            text_col (str): Name of the column containing text sequences.
            max_length (int): Maximum sequence length for padding/truncation.
            label_col (str, optional): Name of the column containing labels.

        Returns:
            TensorDataset: A dataset containing:
                - input_ids: (N, max_length)
                - attention_mask: (N, max_length)
                - labels: (N,) if label_col is provided
        """
        import pandas as pd
        assert self.tokenizer is not None, "Tokenizer not loaded. Call load() first."

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame for argument `df`.")

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        texts = df[text_col].astype(str).tolist()

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True
        )
        encoded['attention_mask'] = encoded['attention_mask'] == 0

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(
                    f"Column '{label_col}' not found in DataFrame.")
            labels = torch.tensor(df[label_col].tolist(), dtype=torch.long)
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
        Note:
            Call ``self.load()`` (directly or indirectly via ``encode``/``encode_pandas``)
            beforehand so that ``self.tokenizer`` is populated and masking can resolve
            token ids such as ``[MASK]``.
        """
        assert mask_token_p + random_token_p <= 1.0

        device = input_ids.device

        labels = input_ids.clone()
        shape = input_ids.shape

        probability_matrix = torch.full(shape, mask_p, device=device)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool, device=device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        input_ids_masked = input_ids.clone()

        indices_replaced = torch.bernoulli(torch.full(
            shape, mask_token_p, device=device)).bool() & masked_indices
        input_ids_masked[indices_replaced] = self.tokenizer.mask_token_id

        p = random_token_p / (1 - mask_token_p)
        indices_random = torch.bernoulli(torch.full(
            shape, p, device=device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.tokenizer.vocab_size, shape, dtype=torch.long, device=device)
        input_ids_masked[indices_random] = random_words[indices_random]

        return input_ids_masked, labels
