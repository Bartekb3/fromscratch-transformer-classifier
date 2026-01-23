import pandas as pd 
import wandb
import matplotlib.pyplot as plt

entity = "praca-inzynierska"
project = "final-experiments" 


def get_experiments_dict(entity=entity, project=project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    d = {}
    for run in runs:
        d[run.id] = run.name
    return d




def get_plot_data_by_run_id(
    run_id,
    keys,
    samples=100000,
):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    required_keys = set(keys) | {"_step", "_timestamp"}

    history = run.history(
        keys=list(required_keys),
        samples=samples
    )

    df = pd.DataFrame(history)

    df = (
        df
        .sort_values("_step")
        .reset_index(drop=True)
    )

    return df


def get_runtime_by_run_id(run_id, entity=entity, project=project):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    runtime = run.summary.get("_runtime")

    if runtime is None:
        raise ValueError(f"No runtime found for run_id={run_id}")

    return runtime  


def add_metric_runtime_row(
    df: pd.DataFrame,
    run_id: str,
    name: str,
    metric_name: str,
    metric_scope: str,   # "eval" | "test" | "train"
    agg: str,            # "min" | "max" | "avg"
    ):
    """
    Adds a single record to the DataFrame:
    - name: experiment name
    - x: runtime in seconds
    - y: aggregated metric value
    """


    if agg not in {"min", "max", "avg"}:
        raise ValueError("agg must be one of: 'min', 'max', 'avg'")

    if metric_scope not in {"eval", "test", "train"}:
        raise ValueError("metric_scope must be one of: 'eval', 'test', 'train'")

    metric_key = f"{metric_scope}/{metric_name}"

    runtime_sec = get_runtime_by_run_id(run_id)

    hist_df = get_plot_data_by_run_id(
        run_id=run_id,
        keys=[metric_key],
    )

    if metric_key not in hist_df.columns:
        raise ValueError(f"Metric '{metric_key}' not found in run history")

    values = hist_df[metric_key].dropna()

    if values.empty:
        raise ValueError(f"No values for metric '{metric_key}'")

    if agg == "min":
        y_val = values.min()
    elif agg == "max":
        y_val = values.max()
    elif agg == "avg":
        y_val = values.mean()

    new_row = pd.DataFrame(
        [{
            "name": name,
            "x": runtime_sec,
            "y": y_val,
        }]
    )

    df = pd.concat([df, new_row], ignore_index=True)

    return df


# def build_metric_runtime_df(
#     run_ids: list[str],
#     run_names: list[str],
#     metric_name: str,
#     metric_scope: str,   # "eval" | "test" | "train"
#     agg: str,            # "min" | "max" | "avg"
# ):
#     """
#     Builds a DataFrame with runtime vs aggregated metric values
#     for multiple runs.
#     """

#     if len(run_ids) != len(run_names):
#         raise ValueError("run_ids and run_names must have the same length")

#     df = pd.DataFrame(columns=["name", "x", "y"])

#     for run_id, name in zip(run_ids, run_names):
#         df = add_metric_runtime_row(
#             df=df,
#             run_id=run_id,
#             name=name,
#             metric_name=metric_name,
#             metric_scope=metric_scope,
#             agg=agg,
#         )

#     return df



def plot_runtime_metric_scatter(
    df,
    xlabel: str,
    ylabel: str,
    title: str,
):
    """
    Creates a scatter plot from a DataFrame with columns:
    ['name', 'x', 'y'] and adds a legend based on experiment names.
    """

    plt.figure(figsize=(8, 6))

    for name, group in df.groupby("name"):
        plt.scatter(
            group["x"],
            group["y"],
            label=name,
            s=60,
            alpha=0.8,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Experiment", loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

import pandas as pd

def load_run_metrics_df(run_id: str) -> pd.DataFrame:
    """
    Zwraca DataFrame w formacie:
    metric_name | metric_scope | value | duration_s | avg_epoch_time_s
    dla pojedynczego run_id.
    avg_epoch_time_s = duration_s / liczba wierszy metryki train/avg_epoch_loss
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # runtime w sekundach
    duration_s = get_runtime_by_run_id(run_id)

    # Pobranie historii
    hist = run.history(keys=None, samples=100000)
    df = pd.DataFrame(hist).sort_values("_step").reset_index(drop=True)

    # Liczymy avg_epoch_time_s na podstawie train/avg_epoch_loss
    if "train/avg_epoch_loss" in df.columns:
        n_epochs = df["train/avg_epoch_loss"].dropna().shape[0]
        if n_epochs > 0:
            avg_epoch_time_s = duration_s / n_epochs
        else:
            avg_epoch_time_s = duration_s
    else:
        avg_epoch_time_s = duration_s

    # Rozbijamy kolumny w formacie "scope/metric_name" na metric_scope i metric_name
    records = []
    for col in df.columns:
        if col.startswith(("train/", "eval/", "test/")):
            scope, metric_name = col.split("/", 1)
            for val in df[col]:
                records.append({
                    "metric_name": metric_name,
                    "metric_scope": scope,
                    "value": val,
                    "duration_s": duration_s,
                    "avg_epoch_time_s": avg_epoch_time_s,
                })

    return pd.DataFrame(records)


def build_metric_runtime_df(
    run_ids: list[str],
    run_names: list[str],
    metrics: list[dict],
):
    if len(run_ids) != len(run_names):
        raise ValueError("run_ids and run_names must have the same length")

    if not metrics:
        raise ValueError("metrics list cannot be empty")

    rows = []

    for run_id, name in zip(run_ids, run_names):
        run_df = load_run_metrics_df(run_id)

        row = {
            "run_id": run_id,
            "run_name": name,
        }

        # runtime i avg_epoch_time_s już w run_df
        duration_s = run_df["duration_s"].iloc[0]
        avg_epoch_time_s = run_df["avg_epoch_time_s"].iloc[0]

        row["duration_s"] = duration_s
        row["avg_epoch_time_s"] = avg_epoch_time_s

        metric_cache = {}

        def get_metric_df(scope: str, metric_name: str) -> pd.DataFrame:
            key = (scope, metric_name)
            if key not in metric_cache:
                metric_cache[key] = (
                    run_df[
                        (run_df["metric_scope"] == scope) &
                        (run_df["metric_name"] == metric_name)
                    ]
                    .reset_index(drop=True)
                )
            return metric_cache[key]

        # agregacja metryk
        for m in metrics:
            scope = m["metric_scope"]
            metric_name = m["metric_name"]
            agg = m["agg"]

            metric_df = get_metric_df(scope, metric_name)
            value = float("nan")

            if not metric_df.empty:
                if agg in {"min", "max", "mean", "sum"}:
                    value = getattr(metric_df["value"], agg)()

                elif agg == "last":
                    value = metric_df["value"].iloc[-1]

                elif agg == "at_index":
                    selector = m.get("select_at")
                    if selector is None:
                        raise ValueError(
                            "agg='at_index' requires select_at"
                        )

                    sel_df = get_metric_df(
                        selector["metric_scope"],
                        selector["metric_name"],
                    )

                    if not sel_df.empty:
                        sel_agg = selector["agg"]

                        if sel_agg == "idxmax":
                            idx = sel_df["value"].idxmax()
                        elif sel_agg == "idxmin":
                            idx = sel_df["value"].idxmin()
                        else:
                            raise ValueError(
                                f"Unknown selector agg: {sel_agg}"
                            )

                        if idx < len(metric_df):
                            value = metric_df.loc[idx, "value"]

                else:
                    raise ValueError(f"Unknown agg: {agg}")

            col_name = f"{scope}/{metric_name}"
            row[col_name] = value

        rows.append(row)

    return pd.DataFrame(rows)

