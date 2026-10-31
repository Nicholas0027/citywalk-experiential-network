from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE2_OUT = ROOT / "data" / "analysis"
OUT_DIR = ROOT / "results" / "network"

N_PERM = 5000
SEED = 20260630


def gini(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    return float((2 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def rank_series(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(dtype=float)


def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    xr = rank_series(x)
    yr = rank_series(y)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def permutation_pvalue(x: pd.Series, y: pd.Series, observed: float, rng: np.random.Generator) -> float:
    xr = rank_series(x)
    yr = rank_series(y)
    if np.std(xr) == 0 or np.std(yr) == 0 or not np.isfinite(observed):
        return float("nan")
    centered_x = xr - xr.mean()
    centered_y = yr - yr.mean()
    denom = math.sqrt(float((centered_x @ centered_x) * (centered_y @ centered_y)))
    if denom == 0:
        return float("nan")
    extreme = 0
    for _ in range(N_PERM):
        shuffled = rng.permutation(centered_y)
        rho = float(centered_x @ shuffled / denom)
        if abs(rho) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (N_PERM + 1))


def p_label(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def build_node_metrics() -> pd.DataFrame:
    nodes = pd.read_csv(STAGE2_OUT / "reference_nodes_100m_1500m.csv")
    edges = pd.read_csv(STAGE2_OUT / "reference_edges_100m_1500m.csv")

    graph = nx.Graph()
    for cluster_id in nodes["cluster_id"].astype(int):
        graph.add_node(cluster_id)
    for _, row in edges.iterrows():
        start, end = int(row["start"]), int(row["end"])
        weight = float(row["line_weight"])
        if graph.has_edge(start, end):
            graph[start][end]["weight"] += weight
        else:
            graph.add_edge(start, end, weight=weight)
    for _, _, data in graph.edges(data=True):
        data["distance"] = 1.0 / data["weight"] if data["weight"] > 0 else 1.0

    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="distance")
    closeness = nx.closeness_centrality(graph, distance="distance")

    metrics = nodes[
        [
            "cluster_id",
            "location_POI",
            "Primary_Theme",
            "point_weight",
            "mean_P",
            "mean_A",
            "mean_D",
        ]
    ].copy()
    metrics["degree"] = metrics["cluster_id"].map(degree).fillna(0.0)
    metrics["strength"] = metrics["cluster_id"].map(strength).fillna(0.0)
    metrics["betweenness"] = metrics["cluster_id"].map(betweenness).fillna(0.0)
    metrics["closeness"] = metrics["cluster_id"].map(closeness).fillna(0.0)
    metrics["active_node"] = metrics["degree"] > 0
    return metrics


def pad_distribution(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = {"mean_P": "Pleasure", "mean_A": "Arousal", "mean_D": "Dominance"}
    for subset_name, subset in [("All nodes", metrics), ("Active nodes", metrics[metrics["active_node"]])]:
        for col, label in names.items():
            vals = subset[col].astype(float)
            rows.append(
                {
                    "subset": subset_name,
                    "dimension": label,
                    "n": len(vals),
                    "mean": vals.mean(),
                    "sd": vals.std(),
                    "median": vals.median(),
                    "q25": vals.quantile(0.25),
                    "q75": vals.quantile(0.75),
                    "share_negative": (vals < 0).mean(),
                    "share_zero": (vals == 0).mean(),
                    "share_positive": (vals > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def correlation_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    variables_x = ["point_weight", "degree", "strength", "betweenness", "closeness"]
    variables_y = ["mean_P", "mean_A", "mean_D"]
    labels = {
        "point_weight": "Featured frequency",
        "degree": "Degree",
        "strength": "Weighted degree",
        "betweenness": "Betweenness",
        "closeness": "Closeness",
        "mean_P": "Pleasure",
        "mean_A": "Arousal",
        "mean_D": "Dominance",
    }
    rows = []
    for subset_name, subset in [("All nodes", metrics), ("Active nodes", metrics[metrics["active_node"]])]:
        for x in variables_x:
            for y in variables_y:
                rho = spearman_rho(subset[x], subset[y])
                p = permutation_pvalue(subset[x], subset[y], rho, rng)
                rows.append(
                    {
                        "subset": subset_name,
                        "x": labels[x],
                        "y": labels[y],
                        "spearman_rho": rho,
                        "permutation_p": p,
                        "p_label": p_label(p),
                        "n": len(subset),
                    }
                )
    return pd.DataFrame(rows)


def topology_coupling_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 7)
    pairs = [
        ("point_weight", "degree"),
        ("point_weight", "strength"),
        ("point_weight", "betweenness"),
        ("point_weight", "closeness"),
        ("degree", "betweenness"),
        ("strength", "betweenness"),
    ]
    labels = {
        "point_weight": "Featured frequency",
        "degree": "Degree",
        "strength": "Weighted degree",
        "betweenness": "Betweenness",
        "closeness": "Closeness",
    }
    rows = []
    for subset_name, subset in [("All nodes", metrics), ("Active nodes", metrics[metrics["active_node"]])]:
        for x, y in pairs:
            rho = spearman_rho(subset[x], subset[y])
            p = permutation_pvalue(subset[x], subset[y], rho, rng)
            rows.append(
                {
                    "subset": subset_name,
                    "x": labels[x],
                    "y": labels[y],
                    "spearman_rho": rho,
                    "permutation_p": p,
                    "p_label": p_label(p),
                    "n": len(subset),
                }
            )
    return pd.DataFrame(rows)


def theme_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("Primary_Theme")
        .agg(
            nodes=("cluster_id", "count"),
            active_nodes=("active_node", "sum"),
            mean_weight=("point_weight", "mean"),
            median_weight=("point_weight", "median"),
            mean_degree=("degree", "mean"),
            mean_betweenness=("betweenness", "mean"),
            mean_P=("mean_P", "mean"),
            mean_A=("mean_A", "mean"),
            mean_D=("mean_D", "mean"),
        )
        .reset_index()
        .sort_values("nodes", ascending=False)
    )


def manuscript_tables(
    pad: pd.DataFrame, topo_corr: pd.DataFrame, pad_corr: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pad_all = pad[pad["subset"] == "All nodes"].copy()
    pad_all["Median [IQR]"] = pad_all.apply(
        lambda r: f"{r['median']:.2f} [{r['q25']:.2f}, {r['q75']:.2f}]", axis=1
    )
    pad_all["Positive / zero / negative (%)"] = pad_all.apply(
        lambda r: f"{100*r['share_positive']:.1f} / {100*r['share_zero']:.1f} / {100*r['share_negative']:.1f}",
        axis=1,
    )
    pad_table = pad_all[["dimension", "mean", "sd", "Median [IQR]", "Positive / zero / negative (%)"]].copy()
    pad_table["mean"] = pad_table["mean"].map(lambda v: f"{v:.2f}")
    pad_table["sd"] = pad_table["sd"].map(lambda v: f"{v:.2f}")
    pad_table.columns = ["PAD dimension", "Mean", "SD", "Median [IQR]", "Positive / zero / negative (%)"]

    topo_sel = topo_corr[
        (topo_corr["subset"] == "All nodes")
        & (topo_corr["x"] == "Featured frequency")
        & (topo_corr["y"].isin(["Degree", "Weighted degree", "Betweenness", "Closeness"]))
    ].copy()
    topo_sel["rho (p)"] = topo_sel.apply(
        lambda r: f"{r['spearman_rho']:.3f} ({r['p_label']})", axis=1
    )
    topo_table = topo_sel[["x", "y", "rho (p)", "n"]].copy()
    topo_table.columns = ["Variable 1", "Variable 2", "Spearman rho (permutation p)", "n"]

    pad_sel = pad_corr[
        (pad_corr["subset"] == "All nodes")
        & (pad_corr["x"].isin(["Featured frequency", "Betweenness", "Closeness"]))
    ].copy()
    pad_sel["rho (p)"] = pad_sel.apply(
        lambda r: f"{r['spearman_rho']:.3f} ({r['p_label']})", axis=1
    )
    pad_coupling_table = pad_sel[["x", "y", "rho (p)", "n"]].copy()
    pad_coupling_table.columns = ["Network variable", "PAD dimension", "Spearman rho (permutation p)", "n"]
    return pad_table, topo_table, pad_coupling_table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_node_metrics()
    pad = pad_distribution(metrics)
    pad_corr = correlation_tests(metrics)
    topo_corr = topology_coupling_tests(metrics)
    themes = theme_summary(metrics)
    inequality = pd.DataFrame(
        [
            {"variable": "Featured frequency", "gini": gini(metrics["point_weight"])},
            {"variable": "Degree", "gini": gini(metrics["degree"])},
            {"variable": "Weighted degree", "gini": gini(metrics["strength"])},
            {"variable": "Betweenness", "gini": gini(metrics["betweenness"])},
            {"variable": "Closeness", "gini": gini(metrics["closeness"])},
        ]
    )
    pad_table, topo_table, pad_coupling_table = manuscript_tables(pad, topo_corr, pad_corr)

    metrics.to_csv(OUT_DIR / "stage3_node_coupling_metrics.csv", index=False, encoding="utf-8-sig")
    pad.to_csv(OUT_DIR / "stage3_pad_distribution.csv", index=False, encoding="utf-8-sig")
    pad_corr.to_csv(OUT_DIR / "stage3_pad_network_correlations.csv", index=False, encoding="utf-8-sig")
    topo_corr.to_csv(OUT_DIR / "stage3_topology_coupling_correlations.csv", index=False, encoding="utf-8-sig")
    themes.to_csv(OUT_DIR / "stage3_theme_node_summary.csv", index=False, encoding="utf-8-sig")
    inequality.to_csv(OUT_DIR / "stage3_network_inequality_summary.csv", index=False, encoding="utf-8-sig")
    pad_table.to_csv(OUT_DIR / "manuscript_pad_distribution_table.csv", index=False, encoding="utf-8-sig")
    topo_table.to_csv(OUT_DIR / "manuscript_topology_coupling_table.csv", index=False, encoding="utf-8-sig")
    pad_coupling_table.to_csv(OUT_DIR / "manuscript_pad_coupling_table.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_DIR / "stage3_network_coupling_results.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="node_metrics", index=False)
        pad.to_excel(writer, sheet_name="pad_distribution", index=False)
        pad_corr.to_excel(writer, sheet_name="pad_correlations", index=False)
        topo_corr.to_excel(writer, sheet_name="topology_correlations", index=False)
        themes.to_excel(writer, sheet_name="theme_summary", index=False)
        inequality.to_excel(writer, sheet_name="inequality", index=False)
        pad_table.to_excel(writer, sheet_name="ms_pad_distribution", index=False)
        topo_table.to_excel(writer, sheet_name="ms_topology_coupling", index=False)
        pad_coupling_table.to_excel(writer, sheet_name="ms_pad_coupling", index=False)

    print("Node metrics:", metrics.shape)
    print("\nPAD distribution")
    print(pad_table.to_string(index=False))
    print("\nTopology coupling")
    print(topo_table.to_string(index=False))
    print("\nPAD coupling")
    print(pad_coupling_table.to_string(index=False))
    print("\nInequality")
    print(inequality.to_string(index=False))


if __name__ == "__main__":
    main()
