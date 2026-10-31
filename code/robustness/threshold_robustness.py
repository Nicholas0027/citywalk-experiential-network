from __future__ import annotations

import math
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "analysis"
OUT_DIR = ROOT / "results" / "robustness"

INPUT_FILE = DATA_DIR / "final_poi_records.csv"
BOUNDARY_SHP = ROOT / "data" / "spatial" / "xiamen_district" / "xiamen_district.shp"

EARTH_RADIUS_M = 6_371_008.8
BOUNDARY_TOLERANCE_M = 500.0
BASE_CLUSTER_M = 100
BASE_SPLIT_M = 1500
CLUSTER_THRESHOLDS_M = [50, 100, 200]
SPLIT_THRESHOLDS_M = [1000, 1500, 2000, 3000]

WGS84_TO_UTM50N = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)


def read_shape_polygons(path: Path) -> list[list[tuple[float, float]]]:
    """Read Polygon/MultiPolygon coordinates from an ESRI shapefile without geopandas."""
    data = path.read_bytes()
    offset = 100
    polygons: list[list[tuple[float, float]]] = []
    while offset + 8 <= len(data):
        _, content_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_bytes = content_words * 2
        record = data[offset : offset + content_bytes]
        offset += content_bytes
        if len(record) < 44:
            continue
        shape_type = struct.unpack("<i", record[:4])[0]
        if shape_type not in (5, 15, 25):
            continue
        num_parts, num_points = struct.unpack("<2i", record[36:44])
        parts_start = 44
        points_start = parts_start + 4 * num_parts
        parts = list(struct.unpack(f"<{num_parts}i", record[parts_start:points_start]))
        points = [
            struct.unpack("<2d", record[points_start + i * 16 : points_start + (i + 1) * 16])
            for i in range(num_points)
        ]
        parts.append(num_points)
        for i in range(num_parts):
            ring = points[parts[i] : parts[i + 1]]
            if len(ring) >= 3:
                polygons.append(ring)
    return polygons


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def in_any_polygon(x: object, y: object, polygons: list[list[tuple[float, float]]]) -> bool:
    try:
        xx, yy = float(x), float(y)
    except Exception:
        return False
    return any(point_in_ring(xx, yy, ring) for ring in polygons)


def point_to_segment_distance_m(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Approximate point-to-segment distance in metres within the Xiamen extent."""
    lat0 = math.radians(py)

    def project(lon: float, lat: float) -> tuple[float, float]:
        return (
            math.radians(lon) * EARTH_RADIUS_M * math.cos(lat0),
            math.radians(lat) * EARTH_RADIUS_M,
        )

    p = np.array(project(px, py))
    a = np.array(project(x1, y1))
    b = np.array(project(x2, y2))
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def distance_to_polygons_m(x: object, y: object, polygons: list[list[tuple[float, float]]]) -> float:
    try:
        xx, yy = float(x), float(y)
    except Exception:
        return float("inf")
    best = float("inf")
    for ring in polygons:
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            best = min(best, point_to_segment_distance_m(xx, yy, x1, y1, x2, y2))
    return best


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def projected_xy(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    x, y = WGS84_TO_UTM50N.transform(df["lng_num"].astype(float).to_numpy(), df["lat_num"].astype(float).to_numpy())
    return pd.Series(x, index=df.index), pd.Series(y, index=df.index)


def mode_or_first(values: pd.Series):
    clean = values.dropna()
    if clean.empty:
        return np.nan
    counts = clean.value_counts()
    return counts.index[0]


def safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def prepare_clean_records() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df["lng_num"] = pd.to_numeric(df["lng"], errors="coerce")
    df["lat_num"] = pd.to_numeric(df["lat"], errors="coerce")
    polygons = read_shape_polygons(BOUNDARY_SHP)
    df["inside_xiamen_boundary"] = [
        in_any_polygon(lng, lat, polygons) for lng, lat in zip(df["lng_num"], df["lat_num"])
    ]
    df["distance_to_boundary_m"] = [
        distance_to_polygons_m(lng, lat, polygons) for lng, lat in zip(df["lng_num"], df["lat_num"])
    ]
    df["inside_or_near_xiamen_500m"] = df["inside_xiamen_boundary"] | (
        df["distance_to_boundary_m"] <= BOUNDARY_TOLERANCE_M
    )
    clean = df[
        df["inside_or_near_xiamen_500m"]
        & df["lng_num"].notna()
        & df["lat_num"].notna()
        & df["note_index"].notna()
        & df["Primary_Theme"].notna()
    ].copy()
    clean["location_index"] = pd.to_numeric(clean["location_index"], errors="coerce")
    clean = clean.dropna(subset=["location_index"]).copy()
    clean["location_index"] = clean["location_index"].astype(int)
    return clean


def assign_grid_clusters(records: pd.DataFrame, cluster_m: int) -> pd.DataFrame:
    df = records.copy()
    x, y = projected_xy(df)
    df["cluster_x"] = (x / cluster_m).round() * cluster_m
    df["cluster_y"] = (y / cluster_m).round() * cluster_m
    cluster_keys = pd.Series(list(zip(df["cluster_x"].astype(int), df["cluster_y"].astype(int))), index=df.index)
    df["cluster_id"] = pd.factorize(cluster_keys, sort=True)[0]
    return df


def build_network(records: pd.DataFrame, cluster_m: int, split_m: int):
    clustered = assign_grid_clusters(records, cluster_m)
    nodes = (
        clustered.groupby("cluster_id")
        .agg(
            lng=("lng_num", "mean"),
            lat=("lat_num", "mean"),
            location_POI=("location_POI", mode_or_first),
            Primary_Theme=("Primary_Theme", mode_or_first),
            point_weight=("cluster_id", "size"),
            mean_P=("P_score", safe_mean),
            mean_A=("A_score", safe_mean),
            mean_D=("D_score", safe_mean),
        )
        .reset_index()
    )
    node_lookup = nodes.set_index("cluster_id")[["lat", "lng", "location_POI", "Primary_Theme"]].to_dict("index")

    raw_edges = []
    considered = 0
    same_cluster = 0
    dropped_long = 0
    distances = []
    sorted_df = clustered.sort_values(["note_index", "location_index"])
    for note_id, group in sorted_df.groupby("note_index", sort=False):
        rows = group[["cluster_id", "Primary_Theme"]].to_records(index=False)
        for i in range(len(rows) - 1):
            start = int(rows[i][0])
            end = int(rows[i + 1][0])
            if start == end:
                same_cluster += 1
                continue
            s = node_lookup[start]
            e = node_lookup[end]
            dist_m = haversine_m(s["lat"], s["lng"], e["lat"], e["lng"])
            distances.append(dist_m)
            considered += 1
            if dist_m <= split_m:
                raw_edges.append(
                    {
                        "start": start,
                        "end": end,
                        "Theme": rows[i][1],
                        "note_index": note_id,
                        "distance_m": dist_m,
                    }
                )
            else:
                dropped_long += 1

    raw_edges_df = pd.DataFrame(raw_edges)
    if raw_edges_df.empty:
        edges = pd.DataFrame(
            columns=[
                "start",
                "end",
                "line_weight",
                "Theme",
                "mean_distance_m",
                "start_POI",
                "end_POI",
                "edge_label",
            ]
        )
    else:
        edges = (
            raw_edges_df.groupby(["start", "end"])
            .agg(
                line_weight=("note_index", "count"),
                Theme=("Theme", mode_or_first),
                mean_distance_m=("distance_m", "mean"),
            )
            .reset_index()
        )
        meta = nodes.set_index("cluster_id")
        edges["start_POI"] = edges["start"].map(meta["location_POI"])
        edges["end_POI"] = edges["end"].map(meta["location_POI"])
        edges["edge_label"] = edges["start_POI"].astype(str) + " -> " + edges["end_POI"].astype(str)

    diagnostics = {
        "cluster_m": cluster_m,
        "split_m": split_m,
        "records_used": len(clustered),
        "notes_used": clustered["note_index"].nunique(),
        "nodes": len(nodes),
        "unique_edges": len(edges),
        "raw_edges_retained": len(raw_edges_df),
        "candidate_transitions": considered,
        "same_cluster_transitions": same_cluster,
        "dropped_long_transitions": dropped_long,
        "retained_transition_share": len(raw_edges_df) / considered if considered else float("nan"),
        "top10_node_weight_share": nodes.nlargest(10, "point_weight")["point_weight"].sum() / len(clustered),
        "top20_node_weight_share": nodes.nlargest(20, "point_weight")["point_weight"].sum() / len(clustered),
    }
    return clustered, nodes, edges, diagnostics, distances


def rank_overlap(current: set[str], baseline: set[str]) -> tuple[int, float]:
    if not current and not baseline:
        return 0, float("nan")
    inter = len(current & baseline)
    union = len(current | baseline)
    return inter, inter / union if union else float("nan")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = prepare_clean_records()
    clean.to_csv(OUT_DIR / "stage2_clean_records_inside_or_near_xiamen.csv", index=False, encoding="utf-8-sig")

    baseline_clustered, baseline_nodes, baseline_edges, baseline_diag, baseline_distances = build_network(
        clean, BASE_CLUSTER_M, BASE_SPLIT_M
    )
    baseline_nodes.to_csv(OUT_DIR / "baseline_clean_nodes_100m_1500m.csv", index=False, encoding="utf-8-sig")
    baseline_edges.to_csv(OUT_DIR / "baseline_clean_edges_100m_1500m.csv", index=False, encoding="utf-8-sig")

    base_top10_nodes = set(baseline_nodes.nlargest(10, "point_weight")["location_POI"].astype(str))
    base_top20_nodes = set(baseline_nodes.nlargest(20, "point_weight")["location_POI"].astype(str))
    base_top10_edges = set(baseline_edges.nlargest(10, "line_weight")["edge_label"].astype(str))
    base_top20_edges = set(baseline_edges.nlargest(20, "line_weight")["edge_label"].astype(str))

    summary_rows = []
    top_rows = []
    all_dist_rows = []
    for cluster_m in CLUSTER_THRESHOLDS_M:
        clustered_for_dist, _, _, _, distances_no_filter = build_network(clean, cluster_m, 10**9)
        for dist in distances_no_filter:
            all_dist_rows.append({"cluster_m": cluster_m, "distance_m": dist})
        for split_m in SPLIT_THRESHOLDS_M:
            _, nodes, edges, diag, _ = build_network(clean, cluster_m, split_m)
            top10_nodes = set(nodes.nlargest(10, "point_weight")["location_POI"].astype(str))
            top20_nodes = set(nodes.nlargest(20, "point_weight")["location_POI"].astype(str))
            top10_edges = set(edges.nlargest(10, "line_weight")["edge_label"].astype(str))
            top20_edges = set(edges.nlargest(20, "line_weight")["edge_label"].astype(str))
            n10, n10j = rank_overlap(top10_nodes, base_top10_nodes)
            n20, n20j = rank_overlap(top20_nodes, base_top20_nodes)
            e10, e10j = rank_overlap(top10_edges, base_top10_edges)
            e20, e20j = rank_overlap(top20_edges, base_top20_edges)
            diag.update(
                {
                    "top10_node_overlap_with_baseline": n10,
                    "top10_node_jaccard_with_baseline": n10j,
                    "top20_node_overlap_with_baseline": n20,
                    "top20_node_jaccard_with_baseline": n20j,
                    "top10_edge_overlap_with_baseline": e10,
                    "top10_edge_jaccard_with_baseline": e10j,
                    "top20_edge_overlap_with_baseline": e20,
                    "top20_edge_jaccard_with_baseline": e20j,
                }
            )
            summary_rows.append(diag)
            top_rows.append(
                {
                    "cluster_m": cluster_m,
                    "split_m": split_m,
                    "top10_nodes": "; ".join(nodes.nlargest(10, "point_weight")["location_POI"].astype(str)),
                    "top10_edges": "; ".join(edges.nlargest(10, "line_weight")["edge_label"].astype(str)),
                }
            )

    summary = pd.DataFrame(summary_rows)
    top_lists = pd.DataFrame(top_rows)
    distances_df = pd.DataFrame(all_dist_rows)
    distance_summary = (
        distances_df.groupby("cluster_m")["distance_m"]
        .quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        .unstack()
        .rename(columns={0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90", 0.95: "p95"})
        .reset_index()
    )
    for threshold in SPLIT_THRESHOLDS_M:
        share = (
            distances_df.assign(within=distances_df["distance_m"] <= threshold)
            .groupby("cluster_m")["within"]
            .mean()
            .rename(f"share_within_{threshold}m")
        )
        distance_summary = distance_summary.merge(share, on="cluster_m")

    baseline_top_nodes = baseline_nodes.nlargest(30, "point_weight").copy()
    baseline_top_edges = baseline_edges.nlargest(30, "line_weight").copy()

    summary.to_csv(OUT_DIR / "threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    top_lists.to_csv(OUT_DIR / "threshold_top10_lists.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(OUT_DIR / "adjacent_edge_distance_summary.csv", index=False, encoding="utf-8-sig")
    baseline_top_nodes.to_csv(OUT_DIR / "baseline_top30_nodes.csv", index=False, encoding="utf-8-sig")
    baseline_top_edges.to_csv(OUT_DIR / "baseline_top30_edges.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_DIR / "threshold_robustness_results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="scenario_summary", index=False)
        distance_summary.to_excel(writer, sheet_name="distance_summary", index=False)
        top_lists.to_excel(writer, sheet_name="top10_lists", index=False)
        baseline_top_nodes.to_excel(writer, sheet_name="baseline_top_nodes", index=False)
        baseline_top_edges.to_excel(writer, sheet_name="baseline_top_edges", index=False)

    print("Clean records:", len(clean), "notes:", clean["note_index"].nunique())
    print("Baseline:", baseline_diag)
    print(summary.to_string(index=False))
    print(distance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
