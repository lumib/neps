import neps
import pandas as pd
from pathlib import Path
from pathlib import Path
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

APPEARANCE = {
    "new__priorband+asha": ["PriorBand (ASHA)","cornflowerblue"],
    "new__priorband+sh": ["PriorBand (SH)","darkviolet"],
    "new__priorband+hb": ["PriorBand (HB)","darkcyan"],
    "new__priorband+asynhb": ["PriorBand (AsyncHB)","darkslategray"],
    "new__RandomSearch": ["RandomSearch","olive"],
}

def extract_results(path, replace_inf=True):
    """
    Extract results from a NEPS experiment directory.
    """
    fulldf, _ = neps.status(path, print_summary=False)
    is_multi_fidelity = fulldf.columns[-13] == "learning_curve"
    fulldf=fulldf[fulldf.columns[-15:]] if is_multi_fidelity else fulldf[fulldf.columns[-13:]]
    fulldf.sort_values("time_sampled",inplace=True)
    result_frame = pd.DataFrame(columns=["cost", "objective_to_minimize"])
    result_frame["cost"] = fulldf["cost"].cumsum()
    result_frame["objective_to_minimize"] = fulldf["objective_to_minimize"].cummin()
    if replace_inf:
        result_frame["objective_to_minimize"].replace(np.inf, np.nan)
    result_frame.set_index("cost",inplace=True)
    return result_frame

def plot_experiment_loss(path:Path, ylabel, xlabel, title, ax = plt.subplots()[1], y_axis:Literal["evaluation", "cost", "fidelity"]="evaluations", max_fidelity:float=20):
    # Assume the path and its subdirectories consists of experiment_name/group_or_algorithm/seed/experiment_files
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    results_dict = {}
    metadata_dict = {}
    for group_dir in Path(path).iterdir():
        results_dict[group_dir.name] = {}
        metadata_dict[group_dir.name] = {}
        with open(group_dir /next(group_dir.iterdir()).name/ "optimizer_info.yaml", 'r') as f:
            metadata_dict[group_dir.name]["optimizer_name"] = yaml.safe_load(f)["name"]
        for seed_dir in group_dir.iterdir():
            results_dict[group_dir.name][seed_dir.name] = extract_results(seed_dir)

    for group, group_dict in results_dict.items():
        all_curves = []
        all_xs = []
        for _, seed_df in group_dict.items():
            if y_axis == "evaluation":
                seed_df.index = seed_df.index / max_fidelity
            all_curves.append(seed_df["objective_to_minimize"])
            all_xs.append(seed_df.index)

        agg_group_df = pd.DataFrame(all_curves).transpose().sort_index().ffill(limit_area='inside')
        means = agg_group_df.mean(axis=1, skipna=True)
        sem = agg_group_df.sem(axis=1,skipna=True).replace(np.nan, 0.0)
        ax.plot(agg_group_df.index, means, label=path.name + " - " + APPEARANCE.get(metadata_dict[group]["optimizer_name"], [group, None])[0], color=APPEARANCE.get(metadata_dict[group]["optimizer_name"], [group, None])[1])
        ax.fill_between(agg_group_df.index, means - sem, means + sem, alpha=0.15, color=APPEARANCE.get(metadata_dict[group]["optimizer_name"], [group, None])[1])
    if y_axis == "fidelity":
        x_tick_labels = ax.get_xticklabels()
        if y_axis == "fidelity":
            ax.set_xticklabels([str(label.get_text())+"x" for label in x_tick_labels])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_experiments_ranks(path:Path, ylabel, xlabel, title, ax = plt.subplots()[1], y_axis:Literal["evaluation", "cost", "fidelity"]="evaluations", max_fidelity:float=20):
    # Assume the path and its subdirectories consists of experiment_name/group_or_algorithm/seed/experiment_files
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    results_dict = {}
    metadata_dict = {}
    for group_dir in Path(path).iterdir():
        results_dict[group_dir.name] = {}
        metadata_dict[group_dir.name] = {}
        with open(group_dir /next(group_dir.iterdir()).name/ "optimizer_info.yaml", 'r') as f:
            metadata_dict[group_dir.name]["optimizer_name"] = yaml.safe_load(f)["name"]
        for seed_dir in group_dir.iterdir():
            results_dict[group_dir.name][seed_dir.name] = extract_results(seed_dir)

    group_dfs = []
    for group_dict in results_dict.values():
        all_curves = []
        all_xs = []
        for _, seed_df in group_dict.items():
            if y_axis == "evaluation":
                seed_df.index = seed_df.index / max_fidelity
            all_curves.append(seed_df["objective_to_minimize"])
            all_xs.append(seed_df.index)

        group_df = pd.DataFrame(all_curves).transpose().sort_index().ffill(limit_area='inside')
        group_df.columns = group_dict.keys()
        group_dfs.append(group_df)
    groups = list(results_dict.keys())
    unique_fidelities = list(set(sum([df.index.tolist() for df in group_dfs],[])))
    unique_fidelities.sort()
    min_fid = max([df.index.min() for df in group_dfs])
    max_fid = unique_fidelities[-1]
    min_seeds = min([len(df.columns) for df in group_dfs])
    for i, group_df in enumerate(group_dfs):
        group_df = group_df.reindex([fid for fid in unique_fidelities if fid>= min_fid and fid<= max_fid])
        group_df = group_df.ffill()
        group_df = group_df.iloc[:, range(min_seeds)]
        group_dfs[i] = group_df
    for seed in range(min_seeds):
        seed_df = pd.concat([df.iloc[:, seed] for df in group_dfs], axis=1)
        for group in range(len(group_dfs)):
            group_dfs[group].iloc[:, seed] = seed_df.rank(1).iloc[:, group]

    for group, group_df in enumerate(group_dfs):
        all_curves = []
        all_xs = []
        for seed in group_df.columns:
            seed_df = group_df[seed]
            if y_axis == "evaluation":
                seed_df.index = seed_df.index / max_fidelity
            all_curves.append(seed_df)
            all_xs.append(seed_df.index)

        agg_group_df = pd.DataFrame(all_curves).transpose().sort_index().ffill(limit_area='inside')
        means = agg_group_df.mean(axis=1, skipna=True)
        sem = agg_group_df.sem(axis=1,skipna=True).replace(np.nan, 0.0)
        ax.plot(agg_group_df.index, means, label=groups[group] + " - " + APPEARANCE.get(metadata_dict[groups[group]]["optimizer_name"], [groups[group], None])[0], color=APPEARANCE.get(metadata_dict[groups[group]]["optimizer_name"], [groups[group], None])[1])
        ax.fill_between(agg_group_df.index, means - sem, means + sem, alpha=0.15, color=APPEARANCE.get(metadata_dict[groups[group]]["optimizer_name"], [groups[group], None])[1])
    if y_axis == "fidelity":
        x_tick_labels = ax.get_xticklabels()
        if y_axis == "fidelity":
            ax.set_xticklabels([str(label.get_text())+"x" for label in x_tick_labels])
    ax.set_yticks(range(len(groups)+1))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    return ax
