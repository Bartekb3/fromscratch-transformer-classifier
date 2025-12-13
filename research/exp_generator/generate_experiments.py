from itertools import product
from pathlib import Path
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(
    description="Generate experiment configurations")
parser.add_argument("--exp", "-e", required=True, help="Experiment key to run")
exp = parser.parse_args().exp

if exp == 'E1':

    ARCHITECTURES = {
        "bertsmall": {
            "num_layers": 4,
            "embedding_dim": 512,
            "mlp_size": 2048,
            "num_heads": 8,
            "attention_embedding_dim": 512,
        },
    }

    ATTENTIONS = ["mha"]

    FREEZE_N_LAYERS = [0, 1, 2]
    CLF_DROPOUT = [0.1, 0.2]
    POOLING = ['cls', 'mean']
    GRID = list(product(FREEZE_N_LAYERS, CLF_DROPOUT, POOLING))

elif exp == 'E2':

    ARCHITECTURES = {
        "bertsmall": {
            "num_layers": 4,
            "embedding_dim": 512,
            "mlp_size": 2048,
            "num_heads": 8,
            "attention_embedding_dim": 512,
        },
    }

    ATTENTIONS = ["lsh", "favor"]

    NUM_HASHES = [4, 8]
    CHUNK_SIZE = [64, 128]
    LSH_GRID = list(product(NUM_HASHES, CHUNK_SIZE))

    NB_FEATURES = [1, 0.5, 0.25, 0.125]  # fraction of D
    FAVOR_GRID = NB_FEATURES

elif exp == 'E3':

    ARCHITECTURES = {
        "autotinys1": {
            "num_layers": 5,
            "embedding_dim": 564,
            "mlp_size": 1054,
            "num_heads": 8,
            "attention_embedding_dim": 512,
        },
        "autotinys2": {
            "num_layers": 4,
            "embedding_dim": 396,
            "mlp_size": 624,
            "num_heads": 6,
            "attention_embedding_dim": 384,
        },
        "autotinys3": {
            "num_layers": 4,
            "embedding_dim": 432,
            "mlp_size": 384,
            "num_heads": 4,
            "attention_embedding_dim": 256,
        },
        "autotinys4": {
            "num_layers": 3,
            "embedding_dim": 320,
            "mlp_size": 608,
            "num_heads": 4,
            "attention_embedding_dim": 256,
        },
    }

    ATTENTIONS = ["mha", "lsh", "favor"]

else:
    raise ValueError(f"Unknown experiment key: {exp}")

TEMPLATES = {
    "pretraining": f"{ROOT}/research/exp_generator/config_tmpl_pretraining.yaml",
    "finetuning": f"{ROOT}/research/exp_generator/config_tmpl_finetuning.yaml",
}

MAX_LENGTH = {
    "wikipedia": 512,
    "imdb": 512,
    "hyperpartisan": 4096,
    "arxiv": 16384,
}

NUM_LABELS = {
    "imdb": 2,
    "arxiv": 11,
    "hyperpartisan": 2
}


# TODO - exp1
BEST_FT_PARAMS = {  
    "imdb": {
        "freeze_n_layers": 1,
        "clf_dropout": 0.1,
        "pooling": "cls"
    },
    "hyperpartisan": {
        "freeze_n_layers": 1,
        "clf_dropout": 0.1,
        "pooling": "cls"
    },
    "arxiv": {
        "freeze_n_layers": 1,
        "clf_dropout": 0.1,
        "pooling": "cls"
    },
}


MODES = ["pretraining", "finetuning"]
DATASETS = ["wikipedia", "imdb", "hyperpartisan", "arxiv"]


def build_data(mode: str, dataset_name: str) -> dict:
    train_path = f"data/tokenized/{dataset_name}_train.pt"
    data = {"train": {"dataset_path": train_path}}
    if dataset_name == "wikipedia":
        return data
    val_path = f"data/tokenized/{dataset_name}_val.pt"
    data["val"] = {"dataset_path": val_path}
    if mode != "pretraining":
        data["test"] = {
            "dataset_path": f"data/tokenized/{dataset_name}_test.pt"}
    return data


def generate_exp(mode,
                 dataset_name,
                 architecture,
                 attention_mechanism,
                 folder_name,
                 clf_dropout=None,
                 freeze_n_layers=None,
                 pooling=None,
                 num_hashes=8,
                 chunk_size=128,
                 nb_features=0.5):

    cfg = yaml.safe_load(Path(TEMPLATES[mode]).read_text(encoding="utf-8"))

    # EXP NAME
    cfg["experiment"]["name"] = folder_name
    cfg["experiment"]["output_dir"] = f"experiments/{mode}/{folder_name}"
    cfg["logging"]["wandb"]["run_name"] = folder_name

    # ARCH
    max_length = MAX_LENGTH[dataset_name]
    cfg["tokenizer"]["max_length"] = max_length
    arch = ARCHITECTURES[architecture]
    cfg_arch = cfg["architecture"]
    cfg_arch["embedding_dim"] = arch["embedding_dim"]
    cfg_arch["num_layers"] = arch["num_layers"]
    cfg_arch["mlp_size"] = arch["mlp_size"]
    cfg_arch["attention"]["attention_embedding_dim"] = arch["attention_embedding_dim"]
    cfg_arch["attention"]["num_heads"] = arch["num_heads"]
    cfg_arch["attention"]["kind"] = attention_mechanism

    if attention_mechanism == 'lsh':
        cfg["architecture"]['attention']['lsh']['num_hashes'] = num_hashes
        cfg["architecture"]['attention']['lsh']['chunk_size'] = chunk_size
    elif attention_mechanism == 'favor':
        cfg["architecture"]['attention']['favor']['nb_features'] = nb_features * \
            arch["attention_embedding_dim"]

    # TRAIN
    batch_size = int(32768/max_length)
    cfg['training']['batch_size'] = batch_size
    if mode == 'finetuning':
        cfg['classification_head']['classifier_dropout'] = clf_dropout
        cfg["classification_head"]["pooling"] = pooling

        # freeze_n_layers = 0 means we dont freeze anything , > 0 we freeze embeddings as well
        cfg['training']['freeze'] = freeze_n_layers > 0
        cfg['training']['freeze_n_layers'] = freeze_n_layers

        cfg["classification_head"]["num_labels"] = NUM_LABELS[dataset_name]

        cfg['training']['learning_rate'] = 3e-5
    else:
        if dataset_name == 'wikipedia': 
            cfg['training']['learning_rate'] = 5e-4
        else:
            cfg['training']['learning_rate'] = 2e-4


    # DATA
    cfg["data"] = build_data(mode, dataset_name)

    # PRE NAME
    if mode == "pretraining":
        if dataset_name != 'wikipedia':

            pre_name = folder_name.split("_")
            pre_name[2] = "wikipedia"
            pre_name = "_".join(pre_name)

            resume = cfg["training"]["resume"]
            resume["resume_pretrainig_name"] = pre_name
            resume["checkpoint_path"] = f"../{pre_name}/checkpoints/model.ckpt"
            resume["is_resume"] = True
    else:
        if exp == 'E1':
            pre_name = f"E1_pretraining_{dataset_name}_{architecture}_{attention_mechanism}"
        else:
            pre_name = folder_name.split("_")
            pre_name[1] = "pretraining"
            pre_name = "_".join(pre_name)

        cfg["pretrained_experiment"]["name"] = pre_name
        cfg["pretrained_experiment"]["path"] = f"experiments/pretraining/{pre_name}"

    exp_dir = ROOT / "experiments" / mode / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    out_cfg = exp_dir / "config.yaml"
    out_cfg.write_text(yaml.dump(cfg, sort_keys=False,
                       allow_unicode=True), encoding="utf-8")
    
def main():

    p = product(MODES, ARCHITECTURES, ATTENTIONS, DATASETS)

    if exp == "E1":  # grid finetuning test

        for mode, architecture, attention_mechanism, dataset_name in p:
            if mode == "finetuning":
                if dataset_name == "wikipedia":
                    continue
                for grid in GRID:
                    freeze_n_layers, clf_dropout, pooling = grid
                    folder_name = f"E1_{mode}_{dataset_name}_{architecture}_{attention_mechanism}_f{freeze_n_layers}_d{clf_dropout}_{pooling}"

                    generate_exp(mode,
                                 dataset_name,
                                 architecture,
                                 attention_mechanism,
                                 folder_name,
                                 clf_dropout,
                                 freeze_n_layers,
                                 pooling)
            else:
                folder_name = f"E1_{mode}_{dataset_name}_{architecture}_{attention_mechanism}"
                generate_exp(mode,
                             dataset_name,
                             architecture,
                             attention_mechanism,
                             folder_name)

    elif exp == "E2":  # attn hyperparam test

        for mode, architecture, attention_mechanism, dataset_name in p:
            if mode == "finetuning" and dataset_name == "wikipedia":
                continue

            if mode == 'finetuning':
                freeze_n_layers, clf_dropout, pooling = BEST_FT_PARAMS[dataset_name].values()
            else:
                freeze_n_layers, clf_dropout, pooling = None, None, None

            if attention_mechanism == 'lsh':
                for lsh_grid in LSH_GRID:
                    num_hashes, chunk_size = lsh_grid
                    folder_name = f"E2_{mode}_{dataset_name}_{architecture}_{attention_mechanism}_h{num_hashes}_c{chunk_size}"
                    generate_exp(mode,
                                 dataset_name,
                                 architecture,
                                 attention_mechanism,
                                 folder_name,
                                 clf_dropout,
                                 freeze_n_layers,
                                 pooling,
                                 num_hashes=num_hashes,
                                 chunk_size=chunk_size,
                                 )
            else:
                for favor_grid in FAVOR_GRID:
                    nb_features = favor_grid
                    folder_name = f"E2_{mode}_{dataset_name}_{architecture}_{attention_mechanism}_nb{nb_features}"
                    generate_exp(mode,
                                 dataset_name,
                                 architecture,
                                 attention_mechanism,
                                 folder_name,
                                 clf_dropout,
                                 freeze_n_layers,
                                 pooling,
                                 nb_features=nb_features)

    elif exp == "E3":  # architecture test

        for mode, architecture, attention_mechanism, dataset_name in p:
            if mode == "finetuning" and dataset_name == "wikipedia":
                continue

            if mode == 'finetuning':
                freeze_n_layers, clf_dropout, pooling = BEST_FT_PARAMS[dataset_name].values()
            else:
                freeze_n_layers, clf_dropout, pooling = None, None, None

            folder_name = f"E3_{mode}_{dataset_name}_{architecture}_{attention_mechanism}"
            generate_exp(mode,
                         dataset_name,
                         architecture,
                         attention_mechanism,
                         folder_name,
                         clf_dropout,
                         freeze_n_layers,
                         pooling)


if __name__ == "__main__":
    main()
