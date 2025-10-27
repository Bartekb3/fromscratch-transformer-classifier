Tests

Purpose:
- Simple unit tests for core components (e.g., attention, dropout behavior, tensor shapes).

How to run:
- Requires `pytest`. Install project dependencies (e.g., `pip install -r requirements.txt`) and run:
  - `pytest -q`

Contents:
- unit/ — module-level tests (e.g., `test_multihead_self_attention.py`, `test_lsh.py`).

Notes:
- Tests do not download any data. A standard PyTorch environment is sufficient.
