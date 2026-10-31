from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "data" / "analysis"
STAGE19 = ROOT / "results" / "subnetworks"
DATA = ROOT / "data" / "spatial"
OUT = ROOT / "results" / "communities"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_NODE_FILE = STAGE2 / "reference_nodes_100m_1500m.csv"
THEME_NODE_FILE = STAGE19 / "theme_subnetwork_nodes.csv"
THEME_EDGE_FILE = STAGE19 / "theme_subnetwork_edges.csv"
RECORD_FILE = STAGE19 / "theme_subnetwork_records.csv"
BOUNDARY_FILE = DATA / "xiamen_district" / "xiamen_district.shp"

THEME_ORDER = [
    "Commercial & Tourism",
    "Nature",
    "Historical & Art",
    "Local Life",
]
NETWORK_ORDER = ["Overall", *THEME_ORDER]
N_STARTS = 100
RANDOM_SEED = 20260727
MIN_PAD_COMMUNITY_NODES = 10


def clean_theme(value: object) -> str:
    return (
        str(value)
        .replace("Commericial", "Commercial")
        .replace("Local_Life", "Local Life")
    )


def entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if not len(values):
        return np.nan
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    nonzero = np.count_nonzero(values > 0)
    if nonzero <= 1:
        return 0.0
    return entropy(values) / math.log(len(values))


def cramers_v(table: pd.DataFrame) -> float:
    if table.empty or min(table.shape) <= 1:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    total = table.to_numpy().sum()
    return float(np.sqrt((chi2 / total) / min(table.shape[0] - 1, table.shape[1] - 1)))


def build_undirected_graph(
    node_ids: pd.Series,
    edges: pd.DataFrame,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(node_ids.astype(int))
    for row in edges.itertuples():
        start = int(row.start)
        end = int(row.end)
        weight = float(row.path_frequency)
        if graph.has_edge(start, end):
            graph[start][end]["weight"] += weight
        else:
            graph.add_edge(start, end, weight=weight)
    return graph


def best_louvain_partition(
    graph: nx.Graph,
    network: str,
) -> tuple[dict[int, str], dict[str, float | int], pd.DataFrame]:
    linked_nodes = [node for node, degree in graph.degree() if degree > 0]
    linked_graph = graph.subgraph(linked_nodes).copy()
    largest_nodes = max(nx.connected_components(linked_graph), key=len)
    largest_graph = linked_graph.subgraph(largest_nodes).copy()

    partitions: list[list[set[int]]] = []
    modularities: list[float] = []
    node_order = sorted(largest_graph.nodes())
    encoded_partitions: list[np.ndarray] = []

    for index in range(N_STARTS):
        communities = nx.community.louvain_communities(
            largest_graph,
            weight="weight",
            resolution=1.0,
            seed=RANDOM_SEED + index,
        )
        partitions.append(communities)
        modularities.append(
            nx.community.modularity(
                largest_graph,
                communities,
                weight="weight",
                resolution=1.0,
            )
        )
        lookup = {
            node: community_index
            for community_index, community in enumerate(communities)
            for node in community
        }
        encoded_partitions.append(
            np.asarray([lookup[node] for node in node_order], dtype=int)
        )

    best_index = int(np.argmax(modularities))
    best_partition = partitions[best_index]
    best_labels = encoded_partitions[best_index]
    stability = np.asarray(
        [
            adjusted_rand_score(best_labels, labels)
            for labels in encoded_partitions
        ],
        dtype=float,
    )

    ordered_communities = sorted(
        best_partition,
        key=lambda community: (-len(community), min(community)),
    )
    assignment = {
        int(node): f"C{community_index}"
        for community_index, community in enumerate(ordered_communities, start=1)
        for node in community
    }

    run_table = pd.DataFrame(
        {
            "network": network,
            "run": np.arange(1, N_STARTS + 1),
            "modularity": modularities,
            "communities": [len(partition) for partition in partitions],
            "ari_to_best_partition": stability,
            "selected": np.arange(N_STARTS) == best_index,
        }
    )
    diagnostics = {
        "linked_nodes": linked_graph.number_of_nodes(),
        "linked_edges": linked_graph.number_of_edges(),
        "lcc_nodes": largest_graph.number_of_nodes(),
        "lcc_edges": largest_graph.number_of_edges(),
        "lcc_share_linked": largest_graph.number_of_nodes()
        / linked_graph.number_of_nodes(),
        "communities": len(best_partition),
        "weighted_modularity": modularities[best_index],
        "mean_ari_to_best": float(stability.mean()),
        "minimum_ari_to_best": float(stability.min()),
        "modularity_run_sd": float(np.std(modularities, ddof=1)),
    }
    return assignment, diagnostics, run_table


def assign_districts(nodes: pd.DataFrame) -> pd.DataFrame:
    boundaries = gpd.read_file(BOUNDARY_FILE).to_crs("EPSG:4490")
    points = gpd.GeoDataFrame(
        nodes[["cluster_id"]].copy(),
        geometry=gpd.points_from_xy(nodes["lng"], nodes["lat"]),
        crs="EPSG:4490",
    )
    joined = gpd.sjoin(
        points,
        boundaries[["NAME", "geometry"]],
        how="left",
        predicate="within",
    )
    missing = joined["NAME"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            points.loc[missing].to_crs("EPSG:32650"),
            boundaries[["NAME", "geometry"]].to_crs("EPSG:32650"),
            how="left",
            max_distance=750,
            distance_col="district_distance_m",
        )
        joined.loc[missing, "NAME"] = nearest["NAME"].to_numpy()
    joined["district"] = joined["NAME"].fillna("Outside/near boundary")
    return nodes.merge(
        joined[["cluster_id", "district"]],
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )


def add_direct_pad(
    nodes: pd.DataFrame,
    records: pd.DataFrame,
    network: str,
) -> pd.DataFrame:
    relevant = records.copy()
    if network != "Overall":
        relevant = relevant[relevant["theme"] == network]
    direct = (
        relevant[
            relevant["is_inherited"].astype(str).str.casefold() == "no"
        ]
        .groupby("cluster_id", as_index=False)
        .agg(
            direct_P=("P_score", "mean"),
            direct_A=("A_score", "mean"),
            direct_D=("D_score", "mean"),
            direct_records=("P_score", "size"),
        )
    )
    return nodes.merge(
        direct,
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )


def add_theme_profile(
    nodes: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        records.groupby(["cluster_id", "theme"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=THEME_ORDER, fill_value=0)
    )
    counts.columns = [f"records_{theme}" for theme in counts.columns]
    counts["dominant_theme"] = (
        counts.rename(columns=lambda value: value.replace("records_", ""))
        .idxmax(axis=1)
    )
    return nodes.merge(
        counts.reset_index(),
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )


def node_centralities(
    nodes: pd.DataFrame,
    graph: nx.Graph,
) -> pd.DataFrame:
    linked = graph.subgraph(
        [node for node, degree in graph.degree() if degree > 0]
    ).copy()
    degree = dict(linked.degree(weight="weight"))
    pagerank = nx.pagerank(linked, weight="weight")
    result = nodes.copy()
    result["community_weighted_degree"] = result["cluster_id"].map(degree).fillna(0.0)
    result["community_pagerank"] = result["cluster_id"].map(pagerank).fillna(0.0)
    return result


def community_profiles(
    network: str,
    nodes: pd.DataFrame,
    records: pd.DataFrame,
) -> pd.DataFrame:
    assigned = nodes[nodes["community_status"] == "LCC community"].copy()
    rows: list[dict[str, object]] = []
    lcc_size = len(assigned)

    for community, frame in assigned.groupby("community_id", sort=False):
        district_counts = frame["district"].value_counts()
        theme_counts = (
            frame["dominant_theme"].value_counts().reindex(THEME_ORDER, fill_value=0)
            if "dominant_theme" in frame
            else pd.Series(dtype=int)
        )
        top = frame.nlargest(
            min(5, len(frame)),
            ["community_weighted_degree", "featured_records"],
        )

        projected = gpd.GeoDataFrame(
            frame[["cluster_id"]].copy(),
            geometry=gpd.points_from_xy(frame["lng"], frame["lat"]),
            crs="EPSG:4490",
        ).to_crs("EPSG:32650")
        centroid = projected.geometry.union_all().centroid
        radius_km = projected.geometry.distance(centroid).median() / 1_000

        row: dict[str, object] = {
            "network": network,
            "community_id": community,
            "nodes": len(frame),
            "share_of_lcc_nodes": len(frame) / lcc_size,
            "featured_records": int(frame["featured_records"].sum()),
            "median_spatial_radius_km": radius_km,
            "dominant_district": district_counts.index[0],
            "dominant_district_share": district_counts.iloc[0] / len(frame),
            "district_entropy_normalized": normalized_entropy(
                district_counts.to_numpy()
            ),
            "top_nodes": "; ".join(top["location_POI"].astype(str)),
            "direct_pad_nodes": int(frame["direct_P"].notna().sum()),
            "mean_P": frame["direct_P"].mean(),
            "mean_A": frame["direct_A"].mean(),
            "mean_D": frame["direct_D"].mean(),
        }
        if network == "Overall":
            dominant_theme = theme_counts.idxmax()
            row.update(
                {
                    "dominant_theme": dominant_theme,
                    "dominant_theme_share": theme_counts.loc[dominant_theme]
                    / len(frame),
                    "theme_entropy_normalized": normalized_entropy(
                        theme_counts.to_numpy()
                    ),
                    **{
                        f"nodes_{theme}": int(theme_counts.loc[theme])
                        for theme in THEME_ORDER
                    },
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pad_heterogeneity_tests(
    network: str,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    assigned = nodes[nodes["community_status"] == "LCC community"].copy()
    eligible = (
        assigned.groupby("community_id")
        .size()
        .loc[lambda values: values >= MIN_PAD_COMMUNITY_NODES]
        .index
    )
    assigned = assigned[assigned["community_id"].isin(eligible)]
    rows = []
    for pad in ["P", "A", "D"]:
        column = f"direct_{pad}"
        groups = [
            frame[column].dropna().to_numpy(dtype=float)
            for _, frame in assigned.groupby("community_id")
            if frame[column].notna().sum() >= 5
        ]
        if len(groups) < 2:
            continue
        statistic, p_value = kruskal(*groups)
        n_values = sum(len(values) for values in groups)
        n_groups = len(groups)
        effect = max(
            0.0,
            (statistic - n_groups + 1) / (n_values - n_groups),
        )
        rows.append(
            {
                "network": network,
                "pad_dimension": pad,
                "communities_tested": n_groups,
                "node_observations": n_values,
                "kruskal_h": statistic,
                "p_value": p_value,
                "epsilon_squared": effect,
            }
        )
    return pd.DataFrame(rows)


def network_association_summary(
    network: str,
    nodes: pd.DataFrame,
) -> dict[str, object]:
    assigned = nodes[nodes["community_status"] == "LCC community"].copy()
    district_valid = assigned["district"] != "Outside/near boundary"
    district_nmi = normalized_mutual_info_score(
        assigned.loc[district_valid, "district"],
        assigned.loc[district_valid, "community_id"],
    )
    profile = community_profiles(network, nodes, pd.DataFrame())
    summary: dict[str, object] = {
        "network": network,
        "district_community_nmi": district_nmi,
        "weighted_mean_dominant_district_share": np.average(
            profile["dominant_district_share"],
            weights=profile["nodes"],
        ),
        "weighted_mean_community_radius_km": np.average(
            profile["median_spatial_radius_km"],
            weights=profile["nodes"],
        ),
    }
    if network == "Overall":
        contingency = pd.crosstab(
            assigned["community_id"],
            assigned["dominant_theme"],
        )
        summary.update(
            {
                "theme_community_nmi": normalized_mutual_info_score(
                    assigned["dominant_theme"],
                    assigned["community_id"],
                ),
                "theme_community_cramers_v": cramers_v(contingency),
                "weighted_mean_dominant_theme_share": np.average(
                    profile["dominant_theme_share"],
                    weights=profile["nodes"],
                ),
            }
        )
    return summary


def main() -> None:
    records = pd.read_csv(RECORD_FILE)
    records["theme"] = records["theme"].map(clean_theme)
    baseline_nodes = pd.read_csv(BASELINE_NODE_FILE).rename(
        columns={"point_weight": "featured_records"}
    )
    baseline_nodes = assign_districts(baseline_nodes)
    theme_nodes = pd.read_csv(THEME_NODE_FILE)
    theme_edges = pd.read_csv(THEME_EDGE_FILE)
    theme_nodes["theme"] = theme_nodes["theme"].map(clean_theme)
    theme_edges["theme"] = theme_edges["theme"].map(clean_theme)

    overall_edges = (
        theme_edges.groupby(["start", "end"], as_index=False)
        .agg(
            path_frequency=("path_frequency", "sum"),
            routes=("routes", "sum"),
            mean_distance_m=("mean_distance_m", "mean"),
        )
        .assign(network="Overall")
    )

    assignment_frames = []
    profile_frames = []
    diagnostic_rows = []
    run_frames = []
    pad_test_frames = []
    association_rows = []

    for network in NETWORK_ORDER:
        if network == "Overall":
            nodes = baseline_nodes.copy()
            edges = overall_edges.copy()
            nodes = add_theme_profile(nodes, records)
        else:
            nodes = theme_nodes[theme_nodes["theme"] == network].copy()
            nodes = assign_districts(nodes)
            edges = theme_edges[theme_edges["theme"] == network].copy()

        nodes = add_direct_pad(nodes, records, network)
        graph = build_undirected_graph(nodes["cluster_id"], edges)
        nodes = node_centralities(nodes, graph)
        assignment, diagnostics, run_table = best_louvain_partition(
            graph,
            network,
        )
        linked_nodes = {
            node for node, degree in graph.degree() if degree > 0
        }
        lcc_nodes = set(assignment)
        nodes["network"] = network
        nodes["community_id"] = nodes["cluster_id"].map(assignment)
        nodes["community_status"] = np.select(
            [
                nodes["cluster_id"].isin(lcc_nodes),
                nodes["cluster_id"].isin(linked_nodes),
            ],
            ["LCC community", "Other linked component"],
            default="Isolated",
        )
        assignment_frames.append(nodes)
        profiles = community_profiles(network, nodes, records)
        profile_frames.append(profiles)
        diagnostic_rows.append(
            {
                "network": network,
                "all_nodes": len(nodes),
                "isolated_nodes": int(
                    (nodes["community_status"] == "Isolated").sum()
                ),
                "other_linked_component_nodes": int(
                    (
                        nodes["community_status"]
                        == "Other linked component"
                    ).sum()
                ),
                "largest_community_nodes": int(profiles["nodes"].max()),
                "largest_community_share_lcc": float(
                    profiles["share_of_lcc_nodes"].max()
                ),
                "median_community_nodes": float(profiles["nodes"].median()),
                **diagnostics,
            }
        )
        run_frames.append(run_table)
        pad_test_frames.append(pad_heterogeneity_tests(network, nodes))
        association_rows.append(network_association_summary(network, nodes))

    assignments = pd.concat(assignment_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    runs = pd.concat(run_frames, ignore_index=True)
    pad_tests = pd.concat(pad_test_frames, ignore_index=True)
    pad_tests["q_global"] = multipletests(
        pad_tests["p_value"],
        method="fdr_bh",
    )[1]
    associations = pd.DataFrame(association_rows)

    assignments.to_csv(OUT / "network_community_node_assignments.csv", index=False)
    profiles.to_csv(OUT / "network_community_profiles.csv", index=False)
    diagnostics.to_csv(OUT / "network_community_diagnostics.csv", index=False)
    runs.to_csv(OUT / "network_community_louvain_runs.csv", index=False)
    pad_tests.to_csv(OUT / "network_community_pad_tests.csv", index=False)
    associations.to_csv(
        OUT / "network_community_spatial_theme_associations.csv",
        index=False,
    )
    overall_edges.to_csv(OUT / "overall_network_edges.csv", index=False)

    print("\nCommunity diagnostics")
    print(diagnostics.to_string(index=False))
    print("\nSpatial and theme associations")
    print(associations.to_string(index=False))
    print("\nPAD heterogeneity")
    print(pad_tests.to_string(index=False))
    print("\nLargest community profiles")
    print(
        profiles.sort_values(["network", "nodes"], ascending=[True, False])
        .groupby("network")
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
