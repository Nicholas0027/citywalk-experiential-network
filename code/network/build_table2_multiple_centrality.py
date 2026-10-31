from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
NODE_PATH = ROOT / "data" / "analysis" / "reference_nodes_100m_1500m.csv"
EDGE_PATH = ROOT / "data" / "analysis" / "reference_edges_100m_1500m.csv"
REPRESENTATIVE_PATH = ROOT / "results" / "network" / "figure2_representative_nodes.csv"
OUT_DIR = ROOT / "results" / "network"

N_PERMUTATIONS = 5_000
SEED = 20260722


def build_directed_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    directed_edges = (
        edges.groupby(["start", "end"], as_index=False)["line_weight"].sum()
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["cluster_id"].astype(int))
    graph.add_weighted_edges_from(
        (
            int(row.start),
            int(row.end),
            float(row.line_weight),
        )
        for row in directed_edges.itertuples()
    )
    return graph


def calculate_metrics(nodes: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    metrics = nodes[["cluster_id", "location_POI", "point_weight"]].copy()
    metrics["weighted_in_degree"] = metrics["cluster_id"].map(
        dict(graph.in_degree(weight="weight"))
    )
    metrics["weighted_out_degree"] = metrics["cluster_id"].map(
        dict(graph.out_degree(weight="weight"))
    )
    metrics["weighted_total_degree"] = (
        metrics["weighted_in_degree"] + metrics["weighted_out_degree"]
    )
    metrics["betweenness"] = metrics["cluster_id"].map(
        nx.betweenness_centrality(graph, normalized=True, weight=None)
    )
    metrics["closeness"] = metrics["cluster_id"].map(
        nx.closeness_centrality(graph)
    )
    metrics["pagerank"] = metrics["cluster_id"].map(
        nx.pagerank(graph, weight="weight")
    )
    metrics = metrics.fillna(0.0)

    rank_columns = [
        "point_weight",
        "weighted_in_degree",
        "weighted_out_degree",
        "weighted_total_degree",
        "betweenness",
        "closeness",
        "pagerank",
    ]
    for column in rank_columns:
        metrics[f"{column}_rank"] = metrics[column].rank(
            ascending=False, method="min"
        ).astype(int)
    return metrics


def permutation_spearman(
    x: pd.Series, y: pd.Series, rng: np.random.Generator
) -> tuple[float, float]:
    x_rank = rankdata(x.to_numpy(dtype=float))
    y_rank = rankdata(y.to_numpy(dtype=float))
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    observed = float(np.dot(x_centered, y_centered) / denominator)
    exceedances = 0
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(y_centered)
        simulated = float(np.dot(x_centered, permuted) / denominator)
        exceedances += abs(simulated) >= abs(observed)
    p_value = (exceedances + 1) / (N_PERMUTATIONS + 1)
    return observed, p_value


def format_value(value: float, rank: int, decimals: int) -> str:
    return f"{value:.{decimals}f} (rank {rank})"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(NODE_PATH)
    edges = pd.read_csv(EDGE_PATH)
    representatives = pd.read_csv(REPRESENTATIVE_PATH).sort_values(
        "representative_order"
    )

    graph = build_directed_network(nodes, edges)
    metrics = calculate_metrics(nodes, graph)

    selected = representatives[["cluster_id", "poi_en", "representative_order"]].merge(
        metrics, on="cluster_id", how="left", validate="one_to_one"
    )

    table_rows: list[dict[str, object]] = []
    for row in selected.itertuples():
        table_rows.append(
            {
                "Featured-frequency rank": int(row.point_weight_rank),
                "Spot": row.poi_en,
                "Featured frequency": int(row.point_weight),
                "Weighted in-degree": format_value(
                    row.weighted_in_degree, row.weighted_in_degree_rank, 0
                ),
                "Weighted out-degree": format_value(
                    row.weighted_out_degree, row.weighted_out_degree_rank, 0
                ),
                "Weighted total degree": format_value(
                    row.weighted_total_degree, row.weighted_total_degree_rank, 0
                ),
                "Betweenness": format_value(
                    row.betweenness, row.betweenness_rank, 6
                ),
                "Closeness": format_value(row.closeness, row.closeness_rank, 6),
                "PageRank": format_value(row.pagerank, row.pagerank_rank, 6),
            }
        )

    rng = np.random.default_rng(SEED)
    correlation_columns = [
        "weighted_in_degree",
        "weighted_out_degree",
        "weighted_total_degree",
        "betweenness",
        "closeness",
        "pagerank",
    ]
    correlation_rows = []
    correlation_display = {}
    for column in correlation_columns:
        rho, p_value = permutation_spearman(
            metrics["point_weight"], metrics[column], rng
        )
        correlation_rows.append(
            {
                "Network metric": column,
                "Spearman rho with featured frequency": rho,
                "Permutation p": p_value,
                "Permutations": N_PERMUTATIONS,
                "Nodes": len(metrics),
            }
        )
        p_label = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        correlation_display[column] = f"rho = {rho:.3f} ({p_label})"

    table_rows.append(
        {
            "Featured-frequency rank": "",
            "Spot": f"All-node association with featured frequency (n = {len(metrics):,})",
            "Featured frequency": "--",
            "Weighted in-degree": correlation_display["weighted_in_degree"],
            "Weighted out-degree": correlation_display["weighted_out_degree"],
            "Weighted total degree": correlation_display["weighted_total_degree"],
            "Betweenness": correlation_display["betweenness"],
            "Closeness": correlation_display["closeness"],
            "PageRank": correlation_display["pagerank"],
        }
    )

    table = pd.DataFrame(table_rows)
    correlations = pd.DataFrame(correlation_rows)
    table.to_csv(
        OUT_DIR / "table2_multiple_centrality_assessment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        OUT_DIR / "table2_all_node_frequency_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUT_DIR / "directed_node_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(table.to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
