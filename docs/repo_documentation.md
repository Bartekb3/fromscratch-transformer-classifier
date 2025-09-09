# Repository Structure

This project is a from-scratch, modular Transformer for text classification experiments.  
The structure is designed to keep **experiments reproducible**, **runs organized**, and **evaluation results separate**.

---

## Top level

- **README.md** – Overview of the project and quickstart instructions.
- **LICENSE** – License under which the project is released.
- **requirements.txt** – Python dependencies required to run training and evaluation.
- **.gitignore / .gitattributes** – Git housekeeping: ignore generated files, enable Git LFS later for datasets or checkpoints.

---

## experiments/ – Experiment definitions

Each subfolder inside `experiments/` defines a single experiment, which is also one training run.  
The folder name is the canonical name of the experiment and will be mirrored in `runs/`.

Contents of an experiment folder:

- **config.yaml** – The single configuration file describing dataset, tokenizer, model, training settings, and logging.
- **evals/** _(optional)_ – Reusable evaluation profiles. Each YAML file describes an evaluation setup (e.g., baseline test, threshold sweep, checkpoint comparison).  
  The filename determines the evaluation folder name under `runs/<experiment>/evaluations/`.

Example:  
`experiments/imdb_performer_l4_d256/` contains the training config and optional eval configs for that specific run.

---

## runs/ – Training and evaluation outputs

Each subfolder in `runs/` has the same name as its counterpart in `experiments/`.  
It is created automatically when training starts and holds **all outputs** for that experiment.

Contents of a run folder:

- **checkpoints/** – Model weights saved during training (`last.pt`, `best_f1.pt`, etc.).
- **metrics.csv / metrics.jsonl** – Training and validation metrics logged over time.
- **config_locked.yaml** – A frozen copy of the training config used.
- **manifest.json** – Metadata about the environment (e.g., timestamp, git commit, Python/PyTorch versions, device, seed).
- **evaluations/** – Subfolders created for each evaluation run.  
  Each evaluation folder contains:
  - `eval_config_locked.yaml` – Frozen copy of the evaluation parameters.
  - `manifest.json` – Copied metadata for reproducibility.
  - `split_<split>/` – Results for the evaluated dataset split (metrics, predictions).
  - `figures/` – Optional plots such as ROC or confusion matrices.

Example:  
`runs/imdb_performer_l4_d256/evaluations/test_baseline/` stores results of the baseline test evaluation.

---

## data/ – Dataset storage

This folder is for local datasets and their processed forms. It is **not versioned** unless small.

- **raw/** – Original downloads (e.g., IMDB dataset).
- **processed/** – Preprocessed or tokenized versions cached for faster loading.
- **README.md** – Instructions on where to obtain datasets, how to store them here, and any privacy considerations.

---

## reports/ – Visuals and analysis

Used for reporting and deeper analysis beyond the automated logs.

- **figures/** – Plots generated manually or from notebooks.
- **tables/** – Tabular summaries of results.
- **notebooks/** – Jupyter notebooks for exploratory analysis, such as comparing multiple runs or visualizing metrics.

---

## src/ – Source code

All importable Python modules for the project.

- **cli/** – Command-line interfaces:

  - `train.py` – Runs training using a config file from `experiments/`.
  - `evaluate.py` – Runs evaluation using a trained run and either an eval config or command-line flags.

- **data/** – Dataset handling: loaders, tokenization, and collators.

- **models/** – Modular Transformer implementation:

  - `embeddings.py` – Token and positional embeddings.
  - `attention/` – Different attention mechanisms (traditional, Performer, Reformer).
  - `blocks/` – Encoder block definitions.
  - `pooling/` – Sequence-to-single embedding strategies (CLS, average, max).
  - `heads/` – Classification heads.
  - `transformer.py` – Assembles the full encoder-only Transformer.

- **training/** – Training logic: loop, optimizer setup, scheduler.

- **eval/** – Evaluation logic: metrics, reporting, and optional analyzers for robustness or subgroup checks.

- **logging/** – Logging and artifact management: metrics writing, checkpoint saving, locked configs.

- **utils/** – Shared utilities:
  - `config.py` – Load and validate configs.
  - `registry.py` – Small registries mapping names in configs (e.g., `"performer"`) to implementation classes.

---

## tests/ – Manual tests

Contains lightweight tests to check core functionality.  
These are **run manually** (no automation yet) but are structured so CI can be added later.

- **unit/** – Tests for specific components (attention shapes, pooling, forward pass).
- **data/** – Tiny test fixtures.
- **README.md** – Instructions for running tests manually (e.g., `pytest tests -q`).

---

## docs/ – Documentation

Human-friendly documentation for collaborators and reviewers.

- **index.md** – How to navigate the repo.
- **architecture.md** – Diagrams and explanations of the Transformer architecture.
- **experiments.md** – How to create a new experiment folder, run training, and evaluate results.
