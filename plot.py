import json
import math
import os
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

model_types = {
    "ERNIE-4.5-21B-A3B": "MoE",
    "Qwen3-30B-A3B": "MoE",
    "OLMoE-1B-7B": "MoE",
    "OLMo-1B": "Dense",
    "Llama-3.2-3B": "Dense",
    "gpt-oss-20b": "MoE",
    "OLMo-7B": "Dense",
    "Qwen3-4B": "Dense",
    "Ministral-3-3B": "Dense",
    "GLM-4.7-Flash": "MoE",
    "Mixtral-8x7B": "MoE",
    "pythia-12b": "Dense",
}

model_specs = {
    "Qwen3-4B": 1.0,
    "OLMo-1B": 1.0,
    "OLMo-7B": 1.0,
    "Ministral-3-3B": 1.0,
    "Llama-3.2-3B": 1.0,
    "pythia-12b": 1.0,
    "OLMoE-1B-7B": 8 / 64,
    "Qwen3-30B-A3B": 8 / 128,
    "ERNIE-4.5-21B-A3B": (6 + 2) / (64 + 2),  # Two shared experts
    "gpt-oss-20b": 4 / 32,
    "GLM-4.7-Flash": (4 + 1) / (64 + 1),  # One shared expert
    "Mixtral-8x7B": 2 / 8,
}

model_depths = {
    "Qwen3-4B": 0,
    "OLMo-1B": 16,
    "OLMo-7B": 0,
    "Ministral-3-3B": 0,
    "Llama-3.2-3B": 0,
    "OLMoE-1B-7B": 16,
    "Qwen3-30B-A3B": 48,
    "ERNIE-4.5-21B-A3B": 28,
    "gpt-oss-20b": 0,
    "GLM-4.7-Flash": 0,
    "Mixtral-8x7B": 0,
    "pythia-12b": 0,
}

concept_to_category = {
    "adjective": "pos",
    "adposition": "pos",
    "adverb": "pos",
    "auxiliary": "pos",
    "coordinating_conjunction": "pos",
    "determiner": "pos",
    "noun": "pos",
    "numeral": "pos",
    "particle": "pos",
    "pronoun": "pos",
    "proper_noun": "pos",
    "punctuation": "pos",
    "subordinating_conjunction": "pos",
    "symbol": "pos",
    "verb": "pos",
    "other": "pos",
    "is_superscript": "latex",
    "is_subscript": "latex",
    "is_inline_math": "latex",
    "is_display_math": "latex",
    "is_math": "latex",
    "is_denominator": "latex",
    "is_numerator": "latex",
    "is_frac": "latex",
    "is_author": "latex",
    "is_title": "latex",
    "is_reference": "latex",
    "is_abstract": "latex",
    "is_function_def": "code",
    "is_function_call": "code",
    "is_assignment": "code",
    "is_class_def": "code",
    "is_import": "code",
    "is_comment": "code",
    "is_string_literal": "code",
    "is_control_flow": "code",
    "is_loop": "code",
    "is_conditional": "code",
    "is_exception_handling": "code",
    "is_array_literal": "code",
    "is_method_call": "code",
    "is_lambda": "code",
    "is_operator": "code",
    "is_constant": "code",
    "is_boolean": "code",
    "is_null": "code",
    "is_decorator": "code",
    "is_async": "code",
    "leading_capital": "text",
    "leading_loweralpha": "text",
    "all_digits": "text",
    "is_not_ascii": "text",
    "contains_all_whitespace": "text",
    "all_capitals": "text",
    "is_not_alphanumeric": "text",
    "contains_whitespace": "text",
    "contains_capital": "text",
    "contains_digit": "text",
}


def _get_model_name(filepath: str, suffix: str = ".csv"):
    filename = filepath.split("/")[-1]
    filename = filename.removesuffix(suffix)
    while not filename.endswith(("B", "b", "Flash")):
        filename = filename.rsplit("-", 1)[0]

    return filename


def _get_best_probes(df: pd.DataFrame, metric="f1"):
    """
    Calculates the best possible performance for each k,
    allowing the 'best layer/expert' to change as k changes.
    """

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    best_expert_per_layer = (
        df.groupby(["concept", "layer", "k"])[metric].max().reset_index()
    )

    best_layer_per_k = (
        best_expert_per_layer.groupby(["concept", "k"])[metric].max().reset_index()
    )

    return best_layer_per_k


def _prepare_scatter_data(df, metric: str, k_val: int):
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    df_k = df[df["k"] == k_val]
    df_max = df_k.groupby(["name", "concept", "category"])[metric].max().reset_index()

    df_pivot = df_max.pivot(
        index=["concept", "category"], columns="name", values=metric
    ).reset_index()

    return df_pivot.dropna()


def plot_probe_scatter(filenames: list[tuple[str, str]], k=1):
    """Figure 2 annd Figure 9"""
    model_names = [
        (_get_model_name(filename1), _get_model_name(filename2))
        for filename1, filename2 in filenames
    ]

    dfs = [
        (pd.read_csv(filename1), pd.read_csv(filename2))
        for filename1, filename2 in filenames
    ]

    best = [
        (_get_best_probes(df1, metric="f1"), _get_best_probes(df2, metric="f1"))
        for df1, df2 in dfs
    ]

    dfs_labeled = [
        (df1.assign(name=name1), df2.assign(name=name2))
        for (df1, df2), (name1, name2) in zip(best, model_names)
    ]

    fig, axs = plt.subplots(
        1,
        len(model_names),
        figsize=(6.75, 2.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axs = axs.flat

    for i, ((df1, df2), ax, (m1, m2)) in enumerate(zip(dfs_labeled, axs, model_names)):
        df = pd.concat([df1, df2], ignore_index=True)
        df["Architecture"] = df["name"].map(model_types)
        df["category"] = df["concept"].map(concept_to_category)
        df = _prepare_scatter_data(df, "f1", k_val=k)
        sns.scatterplot(
            data=df, x=m2, y=m1, hue="category", style="category", s=30, ax=ax
        )

        ax.plot(
            [0, 1.05],
            [0, 1.05],
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Equal",
            zorder=0,
        )
        ax.set_xlabel(r"Best-layer F1")
        ax.set_title(rf"{m1} $|$ {m2}", fontsize=7)

        if i == 0 or i == 3:
            ax.set_ylabel(r"Best-layer F1")
        else:
            ax.set_ylabel("")

        if i != 0:
            ax.get_legend().remove()
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, fontsize=7, loc="lower right")

        ax.set_xlim(0.6, 1.05)
        ax.set_ylim(0.6, 1.05)

    fig.savefig("all_concepts.pdf", bbox_inches="tight")


def plot_probe_perf_olmo_family(filenames: list[str], seed=42):
    """Figure 3"""
    model_names = [_get_model_name(filename) for filename in filenames]

    dfs = [pd.read_csv(filename) for filename in filenames]
    best = [_get_best_probes(df, metric="f1") for df in dfs]

    dfs_labeled = [df.assign(name=name) for df, name in zip(best, model_names)]
    df = pd.concat(dfs_labeled, ignore_index=True)

    fig = plt.figure(figsize=(3.25, 2.5))

    ax = sns.lineplot(
        df,
        y="f1",
        x="k",
        hue="name",
        errorbar=("ci", 95),
        marker="o",
        markers=True,
        dashes=False,
        estimator="mean",
        err_style="band",
        seed=seed,
    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.xlabel(r"$k$")
    plt.ylabel(r"Best-layer F1")
    fig.tight_layout()
    fig.savefig("poly_olmoe.pdf", bbox_inches="tight")


def plot_probe_perf_model_vs_model(filenames: list[tuple[str, str]], seed=42):
    """Figure 1 and Figure 8"""
    model_names = [
        (_get_model_name(filename1), _get_model_name(filename2))
        for filename1, filename2 in filenames
    ]

    dfs = [
        (pd.read_csv(filename1), pd.read_csv(filename2))
        for filename1, filename2 in filenames
    ]

    best = [
        (_get_best_probes(df1, metric="f1"), _get_best_probes(df2, metric="f1"))
        for df1, df2 in dfs
    ]

    dfs_labeled = [
        (df1.assign(model=name1), df2.assign(model=name2))
        for (df1, df2), (name1, name2) in zip(best, model_names)
    ]

    fig, axs = plt.subplots(
        1,
        len(model_names),
        figsize=(6.75, 2.0),
        sharex=False,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axs = axs.flat

    for i, ((df1, df2), ax) in enumerate(zip(dfs_labeled, axs)):
        df = pd.concat([df1, df2], ignore_index=True)
        df["architecture"] = df["model"].map(model_types)

        sns.lineplot(
            data=df,
            y="f1",
            x="k",
            hue="model",
            style="architecture",
            errorbar=("ci", 95),
            dashes=False,
            markers=True,
            estimator="mean",
            err_style="band",
            seed=seed,
            ax=ax,
        )

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(round(x))}")
        )
        ax.set_xlabel(r"$k$")
        if i == 0 or i == 3:
            ax.set_ylabel(r"Best-layer F1")
        else:
            ax.set_ylabel("")

        handles, labels = ax.get_legend_handles_labels()
        model_names = df["model"].unique()

        new_handles = []
        new_labels = []
        for h, lab in zip(handles, labels):
            if lab in model_names:
                new_handles.append(h)
                new_labels.append(lab)

        ax.legend(new_handles, new_labels, title=None, loc="lower right", fontsize=7)

    fig.savefig("poly_all.pdf", bbox_inches="tight")


def plot_network_sparsity(filenames: list[str]):
    """Figure 4"""
    model_names = [_get_model_name(filename) for filename in filenames]
    dfs = [pd.read_csv(filename) for filename in filenames]

    dfs = [_get_best_probes(df) for df in dfs]

    dfs_labeled = [df.assign(name=name) for df, name in zip(dfs, model_names)]
    df = pd.concat(dfs_labeled, ignore_index=True)
    df["active_ratio"] = df["name"].map(model_specs)

    df = df[df["active_ratio"] < 1.0]
    df["active_ratio"] = df["active_ratio"].map(lambda x: f"{x:.2f}")
    df.sort_values("active_ratio", inplace=True)

    fig = plt.figure(figsize=(3.25, 2.5))

    ax = sns.lineplot(
        data=df, x="k", y="f1", hue="active_ratio", marker="o", palette="rocket"
    )
    ax.legend(
        title=r"$N_\mathrm{A}/N$", loc="lower right", fontsize=6, title_fontsize=7
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))

    plt.xlabel(r"$k$")
    plt.ylabel("Best-layer F1")

    plt.ylim(0.9, 1.01)

    fig.tight_layout()
    fig.savefig("network_sparsity.pdf", bbox_inches="tight")


def plot_spec_hist(filenames: list[str]):
    """Figure 10"""
    dfs = []
    for filename in filenames:
        name = _get_model_name(filename)
        df = pd.read_csv(filename)
        df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        df = df.dropna(subset=["f1"])
        df["name"] = name
        df["arch"] = df["name"].map(model_types)
        if df["arch"].iloc[0] == "MoE":
            dfs.append(df)

    num_models = len(dfs)
    ncols = math.ceil(math.sqrt(num_models))
    nrows = math.ceil(num_models / ncols)

    if num_models == 1:
        figsize = (3.25, 2.5)
    else:
        figsize = (6.75, 3.5)

    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True
    )
    axs = axs.flatten()

    cmap = sns.color_palette(palette="rocket", as_cmap=True)

    for i, df in enumerate(dfs):
        df_best = df.sort_values("f1", ascending=False).drop_duplicates(
            ["concept", "layer", "expert"]
        )

        best_scores = df_best.groupby(["layer", "concept"])["f1"].max().reset_index()
        best_scores.rename(columns={"f1": "max_f1"}, inplace=True)

        merged = pd.merge(df_best, best_scores, on=["layer", "concept"])

        df_active = merged[merged["f1"] >= (merged["max_f1"] * 0.95)]

        concept_counts = (
            df_active.groupby(["layer", "concept"])["expert"].nunique().reset_index()
        )

        ax = axs[i]

        sns.histplot(
            data=concept_counts,
            y="expert",
            hue="layer",
            multiple="stack",
            discrete=True,
            legend=False,
            palette=cmap,
            ax=ax,
        )
        for p in ax.patches:
            p.set_edgecolor("none")
            p.set_alpha(0.95)

        norm = mpl.colors.Normalize(  # type: ignore
            vmin=df_active["layer"].min(), vmax=df_active["layer"].max()
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.set_title(df["name"].iloc[0])
        ax.set_ylabel("Number of Experts")
        ax.set_xlabel("Count")

    for j in range(num_models, len(axs)):
        fig.delaxes(axs[j])

    if num_models == 1:
        cbar = fig.colorbar(
            sm, ax=axs[0], orientation="vertical", pad=0.02, fraction=0.05
        )
    else:
        cbar = fig.colorbar(sm, ax=axs[:num_models], orientation="vertical", pad=0.02)

    cbar.set_label("Layer Depth", rotation=270, labelpad=10)

    filename = dfs[0]["name"].iloc[0] if num_models == 1 else "spec"
    fig.savefig(f"{filename}_hist.pdf", bbox_inches="tight")


def plot_domain_spec(filename: str):
    """Figure 7"""
    data = np.load(filename)
    k_vals = data["k_values"]
    _, L, E = data["input"].shape

    records = []
    for s_idx, k in enumerate(k_vals):
        for layer in range(L):
            for e in range(E):
                input_spec = (
                    data["input"][s_idx, layer, e]
                    - data["input_baseline"][s_idx, layer, e]
                )
                output_spec = (
                    data["output"][s_idx, layer, e]
                    - data["output_baseline"][s_idx, layer, e]
                )

                records.append(
                    {
                        "Layer": layer,
                        "Specialization": input_spec,
                        "Number of Clusters": str(k),
                        "Mode": "Input",
                    }
                )
                records.append(
                    {
                        "Layer": layer,
                        "Specialization": output_spec,
                        "Number of Clusters": str(k),
                        "Mode": "Output",
                    }
                )

    df = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.75, 3.0), sharey=True, constrained_layout=True
    )
    sns.lineplot(
        data=df[df["Mode"] == "Input"],
        x="Layer",
        y="Specialization",
        hue="Number of Clusters",
        palette="flare",
        ax=ax1,
        legend=False,
    )
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))
    sns.lineplot(
        data=df[df["Mode"] == "Output"],
        x="Layer",
        y="Specialization",
        hue="Number of Clusters",
        palette="flare",
        ax=ax2,
        # legend=False,
    )
    ax2.legend(title=None)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(4, integer=True))

    fig.savefig("domain_spec.pdf", bbox_inches="tight")


def plot_interp_scores(filenames: list[str]):
    """Figure 5"""
    all_rows = []
    for filename in filenames:
        file = filename.split("/")[-1]

        model_name = file.removesuffix("_auto_interp.json")
        model_name = model_name.split("_")[0]
        while not model_name.endswith("B"):
            model_name = model_name.rsplit("-", 1)[0]

        layer = file.split("_")[1].removeprefix("L")
        depth = model_depths[model_name]

        with open(filename, "r") as f:
            data = json.load(f)

        rows = [
            {
                "F1": e_data["metrics"]["f1_score"],
                "Layer": int(layer),
                "Model": model_name,
                "Total_Layers": depth,
            }
            for _, e_data in data.items()
            if e_data["hypothesis"]
            is not None  # Special case when API blocked an expert
        ]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["Relative_Depth"] = df["Layer"] / (df["Total_Layers"] - 1)
    df["Relative_Depth_Rounded"] = df["Relative_Depth"].round(1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.25, 2.5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 2]},
    )

    sns.boxplot(
        ax=axes[0],
        data=df,
        x="Model",
        y="F1",
        hue="Model",
        showfliers=False,
        width=0.5,
    )
    axes[0].set_xticklabels([])
    axes[0].set_xlabel("")

    axes[0].set_ylabel("F1 Score")

    sns.lineplot(
        ax=axes[1],
        data=df,
        x="Relative_Depth",
        y="F1",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        errorbar=("ci", 95),
    )
    axes[1].set_xlabel(r"Relative Layer Depth")
    axes[1].legend(loc="lower right")

    fig.savefig("interp_scores.pdf", bbox_inches="tight")


def plot_causal_importance_over_layers(filepaths: list[str], total_prompts=20):
    """Figure 6"""
    layer_groups = defaultdict(list)

    for fp in filepaths:
        fname = os.path.basename(fp)
        layer = int(fname.split("_")[0].removeprefix("L"))
        layer_groups[layer].append(fp)

    rank_bins = {
        "Rank 1": [0],
        "Rank 2-3": [1, 2],
        "Rank 4-8": list(range(3, 8)),
    }
    rank_order = ["Rank 1", "Rank 2-3", "Rank 4-8", "Not Routed"]

    rows = []

    for layer, layer_files in sorted(layer_groups.items()):
        ranks = [np.load(fp)["ranks"] for fp in layer_files]
        experts = [
            int(os.path.basename(fp).split("_")[1].removeprefix("E"))
            for fp in layer_files
        ]

        for i, (e, mat) in enumerate(zip(experts, ranks)):
            k = mat.shape[1]

            matched_row = mat[e]

            matched_bins = {}
            for name, cols in rank_bins.items():
                cols = [c for c in cols if c < k]
                matched_bins[name] = matched_row[cols].sum()

            matched_routed = matched_row.sum()
            matched_bins["Not Routed"] = max(total_prompts - matched_routed, 0)

            for bin_name, count in matched_bins.items():
                rows.append(
                    {
                        "Layer": layer,
                        "Expert": e,
                        "Rank Bin": bin_name,
                        "Condition": "Matched",
                        "Percentage": 100 * count / total_prompts,
                    }
                )

            control_rows = []
            for j, mat_other in enumerate(ranks):
                if j == i:
                    continue
                if e < mat_other.shape[0]:
                    control_rows.append(mat_other[e])

            if len(control_rows) == 0:
                continue

            control_mean = np.mean(control_rows, axis=0)

            control_bins = {}
            for name, cols in rank_bins.items():
                cols = [c for c in cols if c < k]
                control_bins[name] = control_mean[cols].sum()

            control_routed = control_mean.sum()
            control_bins["Not Routed"] = max(total_prompts - control_routed, 0)

            for bin_name, count in control_bins.items():
                rows.append(
                    {
                        "Layer": layer,
                        "Expert": e,
                        "Rank Bin": bin_name,
                        "Condition": "Control",
                        "Percentage": 100 * count / total_prompts,
                    }
                )

    df = pd.DataFrame(rows)

    df_sum = df.groupby(["Rank Bin", "Condition"], as_index=False)["Percentage"].mean()

    df_sum["Rank Bin"] = pd.Categorical(
        df_sum["Rank Bin"], categories=rank_order, ordered=True
    )
    df_sum = df_sum.sort_values("Rank Bin")  # type: ignore

    fig = plt.figure(figsize=(3.25, 2.5))

    ax = sns.barplot(
        data=df_sum,
        x="Rank Bin",
        y="Percentage",
        hue="Condition",
        order=rank_order,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.legend(title=None, loc="upper left")

    plt.ylabel(r"Prompts ($\%$)")
    plt.xlabel("")
    fig.tight_layout()
    fig.savefig("causal_importance_all_layers.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plotting results. Pass in the figure number from the paper to plot that exact Figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--figures",
        nargs="+",
        type=int,
        default=[1],
        help="Figure numbers to plot",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        default=None,
        help="File paths to data",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        type=str,
        default=None,
        help="File paths to data as pairs. Figures 1, 2, 8 and 9 need pairs. Usually pairs are MoE/Dense models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    assert args.pair is not None or args.files is not None
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=1.0,
        rc={"text.usetex": True},
    )
    plt.rcParams["text.usetex"] = True

    fig_numbers = args.figures

    for number in fig_numbers:
        match number:
            case 1 | 8:
                assert args.pair is not None, (
                    f"Figure {number} needs file paths as pairs. Use with --pairs"
                )
                plot_probe_perf_model_vs_model(args.pair, args.seed)
            case 2 | 9:
                assert args.pair is not None, (
                    f"Figure {number} needs file paths as pairs. Use with --pairs"
                )
                plot_probe_scatter(args.pair)
            case 3:
                plot_probe_perf_olmo_family(args.files, args.seed)
            case 4:
                plot_network_sparsity(args.files)
            case 5:
                plot_interp_scores(args.files)
            case 6:
                plot_causal_importance_over_layers(args.files)
            case 7:
                plot_domain_spec(args.files[0])
            case 10:
                plot_spec_hist(args.files)
            case _:
                raise ValueError(
                    f"Invalid Figure number. Figure number was {number}. Valid Figure numbers are 1-10."
                )
