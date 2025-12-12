from pathlib import Path
from types import SimpleNamespace

from textclf_transformer.training.utils import config as config_module


def test_resolve_project_path_prefers_repo_root(monkeypatch, tmp_path):
    """Relative paths should resolve against the repository root, not CWD."""
    monkeypatch.chdir(tmp_path)
    rel_path = Path("src/textclf_transformer/tokenizer/wordpiece_tokenizer_wrapper.py")
    resolved = config_module._resolve_project_path(rel_path)
    assert resolved == config_module.PROJECT_ROOT / rel_path


def test_load_tokenizer_wrapper_handles_repo_relative_paths(monkeypatch):
    """Tokenizer loader should convert config paths to absolute ones."""
    called = {}

    class DummyWrapper:
        def __init__(self):
            self.loaded = None

        def load(self, vocab_dir: str):
            called["vocab_dir"] = Path(vocab_dir)
            self.loaded = called["vocab_dir"]

    dummy_module = SimpleNamespace(WordPieceTokenizerWrapper=DummyWrapper)

    def fake_import(path: str):
        called["wrapper_path"] = Path(path)
        return dummy_module

    monkeypatch.setattr(config_module, "_import_module_from_path", fake_import)

    tok_cfg = {
        "wrapper_path": "src/textclf_transformer/tokenizer/wordpiece_tokenizer_wrapper.py",
        "vocab_dir": "src/textclf_transformer/tokenizer/BERT_original",
    }

    wrapper = config_module.load_tokenizer_wrapper_from_cfg(tok_cfg)

    assert isinstance(wrapper, DummyWrapper)
    expected_wrapper = config_module.PROJECT_ROOT / Path(tok_cfg["wrapper_path"])
    expected_vocab = config_module.PROJECT_ROOT / Path(tok_cfg["vocab_dir"])
    assert called["wrapper_path"] == expected_wrapper
    assert called["vocab_dir"] == expected_vocab
