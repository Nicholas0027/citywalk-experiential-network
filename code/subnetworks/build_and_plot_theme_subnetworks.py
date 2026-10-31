from __future__ import annotations

import math
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "data" / "analysis"
DATA = ROOT / "data" / "spatial"
OUT = ROOT / "results" / "subnetworks"
OUT.mkdir(parents=True, exist_ok=True)

RECORD_FILE = STAGE2 / "clean_poi_records.csv"
BASELINE_NODE_FILE = STAGE2 / "reference_nodes_100m_1500m.csv"
BOUNDARY_FILE = DATA / "xiamen_district" / "xiamen_district.shp"

THEME_ORDER = [
    "Commercial & Tourism",
    "Nature",
    "Historical & Art",
    "Local Life",
]
THEME_COLORS = {
    "Commercial & Tourism": "#2CA7C9",
    "Nature": "#7CB342",
    "Historical & Art": "#D66BA8",
    "Local Life": "#E76F35",
}
THEME_LABELS = {
    "Commercial & Tourism": "Commercial & tourism",
    "Nature": "Nature",
    "Historical & Art": "Historical & art",
    "Local Life": "Local life",
}
POI_LABELS = {
    "幸福路": "Xingfu Road area",
    "黄厝沙滩": "Island Ring Boulevard",
    "华新路": "Huaxin Road",
    "八市": "The Eighth Market",
    "南普陀寺": "Nanputuo Temple",
    "集美学村": "Jimei School Village",
    "十里长堤": "Shili Long Causeway",
    "溪岸路花鸟市场": "Xi'an Road Flower and Bird Market",
    "百家村": "Baijia Village",
    "陈嘉庚故居": "Former Residence of Tan Kah Kee",
}

WGS84 = "EPSG:4326"
MAP_CRS = "EPSG:32650"
TRANSFORMER = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)
EARTH_RADIUS_M = 6_371_008.8

CORE_LONLAT = (118.00, 118.20, 24.415, 24.635)
MUNICIPAL_LONLAT = (117.903, 118.349, 24.420, 24.860)
MAP_BG = "#11171B"
MAP_TEXT = "#F3F5F6"
MAP_MUTED = "#AEB8BE"
MAP_BOUNDARY = "#CBD2D6"


def clean_theme(value: object) -> str:
    return (
        str(value)
        .replace("Commericial", "Commercial")
        .replace("Local_Life", "Local Life")
    )


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": MAP_BG,
        }
    )


def assign_cluster_ids(records: pd.DataFrame, cluster_m: int = 100) -> pd.DataFrame:
    x, y = TRANSFORMER.transform(
        records["lng_num"].to_numpy(),
        records["lat_num"].to_numpy(),
    )
    result = records.copy()
    result["cluster_x"] = (
        np.asarray(x) / cluster_m
    ).round().astype(int) * cluster_m
    result["cluster_y"] = (
        np.asarray(y) / cluster_m
    ).round().astype(int) * cluster_m
    keys = pd.Series(
        list(zip(result["cluster_x"], result["cluster_y"])),
        index=result.index,
    )
    result["cluster_id"] = pd.factorize(keys, sort=True)[0]
    return result


def representative_name(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    return counts.index[0]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def build_edges(
    records: pd.DataFrame,
    baseline_nodes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    node_lookup = baseline_nodes.set_index("cluster_id")[["lat", "lng"]]
    sorted_records = records.sort_values(["note_index", "location_index"])
    for note_index, route in sorted_records.groupby("note_index", sort=False):
        if len(route) < 2:
            continue
        themes = route["theme"].dropna().unique()
        if len(themes) != 1:
            raise RuntimeError(f"Route {note_index} has {len(themes)} theme labels.")
        route_rows = route[["cluster_id"]].to_records(index=False)
        for index in range(len(route_rows) - 1):
            start = int(route_rows[index][0])
            end = int(route_rows[index + 1][0])
            start_node = node_lookup.loc[start]
            end_node = node_lookup.loc[end]
            distance = haversine_m(
                float(start_node["lat"]),
                float(start_node["lng"]),
                float(end_node["lat"]),
                float(end_node["lng"]),
            )
            if start == end or distance > 1_500:
                continue
            rows.append(
                {
                    "theme": themes[0],
                    "note_index": note_index,
                    "start": start,
                    "end": end,
                    "distance_m": float(distance),
                }
            )
    transitions = pd.DataFrame(rows)
    edges = (
        transitions.groupby(["theme", "start", "end"], as_index=False)
        .agg(
            path_frequency=("note_index", "size"),
            routes=("note_index", "nunique"),
            mean_distance_m=("distance_m", "mean"),
        )
    )
    return edges


def add_centralities(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for theme in THEME_ORDER:
        theme_nodes = nodes[nodes["theme"] == theme].copy()
        theme_edges = edges[edges["theme"] == theme]
        graph = nx.DiGraph()
        graph.add_nodes_from(theme_nodes["cluster_id"].astype(int))
        graph.add_weighted_edges_from(
            (
                int(row.start),
                int(row.end),
                float(row.path_frequency),
            )
            for row in theme_edges.itertuples()
        )
        theme_nodes["weighted_in_degree"] = theme_nodes["cluster_id"].map(
            dict(graph.in_degree(weight="weight"))
        )
        theme_nodes["weighted_out_degree"] = theme_nodes["cluster_id"].map(
            dict(graph.out_degree(weight="weight"))
        )
        theme_nodes["weighted_total_degree"] = (
            theme_nodes["weighted_in_degree"] + theme_nodes["weighted_out_degree"]
        )
        theme_nodes["betweenness"] = theme_nodes["cluster_id"].map(
            nx.betweenness_centrality(graph, normalized=True, weight=None)
        )
        theme_nodes["closeness"] = theme_nodes["cluster_id"].map(
            nx.closeness_centrality(graph)
        )
        theme_nodes["pagerank"] = theme_nodes["cluster_id"].map(
            nx.pagerank(graph, weight="weight")
        )
        theme_nodes["active_node"] = theme_nodes["weighted_total_degree"] > 0
        frames.append(theme_nodes.fillna(0.0))
    return pd.concat(frames, ignore_index=True)


def build_subnetworks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(RECORD_FILE)
    records["theme"] = records["Primary_Theme"].map(clean_theme)
    records = assign_cluster_ids(records)
    baseline_nodes = pd.read_csv(BASELINE_NODE_FILE)[
        ["cluster_id", "lng", "lat"]
    ]
    if records["cluster_id"].nunique() != len(baseline_nodes):
        raise RuntimeError("Cluster reconstruction does not match the baseline network.")

    node_names = (
        records.groupby(["theme", "cluster_id"])["location_POI"]
        .agg(representative_name)
        .rename("location_POI")
    )
    nodes = (
        records.groupby(["theme", "cluster_id"], as_index=False)
        .agg(
            featured_records=("location_POI", "size"),
            route_count=("note_index", "nunique"),
            mean_P=("P_score", "mean"),
            mean_A=("A_score", "mean"),
            mean_D=("D_score", "mean"),
        )
        .merge(node_names.reset_index(), on=["theme", "cluster_id"])
        .merge(baseline_nodes, on="cluster_id", how="left", validate="many_to_one")
    )
    edges = build_edges(records, baseline_nodes)
    nodes = add_centralities(nodes, edges)

    membership = (
        nodes.assign(present=1)
        .pivot(index="cluster_id", columns="theme", values="present")
        .fillna(0)
        .astype(int)
        .reindex(columns=THEME_ORDER, fill_value=0)
    )
    membership["n_themes"] = membership.sum(axis=1)
    membership = membership.reset_index()

    membership_summary = (
        membership.groupby("n_themes", as_index=False)
        .agg(nodes=("cluster_id", "size"))
        .sort_values("n_themes")
    )
    membership_summary["share"] = (
        membership_summary["nodes"] / membership_summary["nodes"].sum()
    )

    overlap_rows = []
    for i, theme_a in enumerate(THEME_ORDER):
        members_a = set(
            membership.loc[membership[theme_a] == 1, "cluster_id"].astype(int)
        )
        for theme_b in THEME_ORDER[i + 1 :]:
            members_b = set(
                membership.loc[membership[theme_b] == 1, "cluster_id"].astype(int)
            )
            intersection = len(members_a & members_b)
            union = len(members_a | members_b)
            overlap_rows.append(
                {
                    "theme_a": theme_a,
                    "theme_b": theme_b,
                    "shared_nodes": intersection,
                    "share_of_theme_a": intersection / len(members_a),
                    "share_of_theme_b": intersection / len(members_b),
                    "jaccard_similarity": intersection / union,
                }
            )
    pairwise_overlap = pd.DataFrame(overlap_rows)

    summary_rows = []
    for theme in THEME_ORDER:
        route_subset = records[records["theme"] == theme]
        node_subset = nodes[nodes["theme"] == theme]
        edge_subset = edges[edges["theme"] == theme]
        active = node_subset[node_subset["active_node"]]
        graph = nx.DiGraph()
        graph.add_nodes_from(node_subset["cluster_id"].astype(int))
        graph.add_edges_from(
            (int(row.start), int(row.end)) for row in edge_subset.itertuples()
        )
        weak_components = list(nx.weakly_connected_components(graph))
        largest_component = max((len(component) for component in weak_components), default=0)
        summary_rows.append(
            {
                "theme": theme,
                "routes": route_subset["note_index"].nunique(),
                "poi_records": len(route_subset),
                "nodes": len(node_subset),
                "active_nodes": int(node_subset["active_node"].sum()),
                "isolated_nodes": int((~node_subset["active_node"]).sum()),
                "isolated_node_share": float((~node_subset["active_node"]).mean()),
                "directed_edges": len(edge_subset),
                "transition_records": int(edge_subset["path_frequency"].sum()),
                "weak_components": len(weak_components),
                "largest_component_nodes": largest_component,
                "largest_component_share": largest_component / len(node_subset),
                "reciprocity": nx.reciprocity(graph) or 0.0,
                "mean_P": np.average(
                    node_subset["mean_P"],
                    weights=node_subset["featured_records"],
                ),
                "mean_A": np.average(
                    node_subset["mean_A"],
                    weights=node_subset["featured_records"],
                ),
                "mean_D": np.average(
                    node_subset["mean_D"],
                    weights=node_subset["featured_records"],
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    records.to_csv(OUT / "theme_subnetwork_records.csv", index=False)
    nodes.to_csv(OUT / "theme_subnetwork_nodes.csv", index=False)
    edges.to_csv(OUT / "theme_subnetwork_edges.csv", index=False)
    membership.to_csv(OUT / "node_theme_membership_overlap.csv", index=False)
    membership_summary.to_csv(
        OUT / "node_theme_membership_summary.csv", index=False
    )
    pairwise_overlap.to_csv(OUT / "theme_pairwise_node_overlap.csv", index=False)
    summary.to_csv(OUT / "theme_subnetwork_summary.csv", index=False)
    return records, nodes, edges, summary


def projected_extent(
    bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    xmin, ymin = TRANSFORMER.transform(bounds[0], bounds[2])
    xmax, ymax = TRANSFORMER.transform(bounds[1], bounds[3])
    return xmin, xmax, ymin, ymax


def node_sizes(values: pd.Series) -> np.ndarray:
    return 2.0 + 2.15 * np.sqrt(values.to_numpy(dtype=float))


def edge_widths(values: pd.Series) -> np.ndarray:
    return 0.14 + 0.52 * np.log1p(values.to_numpy(dtype=float))


def add_basemap(ax: plt.Axes, zoom: int) -> None:
    try:
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.DarkMatterNoLabels,
            zoom=zoom,
            crs=MAP_CRS,
            attribution=False,
            reset_extent=False,
            interpolation="bilinear",
        )
    except Exception as exc:
        print(f"Basemap unavailable; boundary-only map retained: {exc}")


def theme_glow(
    ax: plt.Axes,
    nodes: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    color: str,
) -> None:
    xmin, xmax, ymin, ymax = extent
    visible = nodes[
        nodes.geometry.x.between(xmin, xmax)
        & nodes.geometry.y.between(ymin, ymax)
    ]
    if visible.empty:
        return
    histogram, _, _ = np.histogram2d(
        visible.geometry.x,
        visible.geometry.y,
        bins=(260, 260),
        range=((xmin, xmax), (ymin, ymax)),
        weights=np.sqrt(visible["featured_records"]),
    )
    density = gaussian_filter(histogram.T, sigma=6.5)
    positive = density[density > 0]
    if not len(positive):
        return
    normalized = np.clip(density / np.percentile(positive, 99.2), 0, 1) ** 0.75
    red, green, blue = to_rgb(color)
    cmap = LinearSegmentedColormap.from_list(
        f"{color}_glow",
        [
            (0.0, (red, green, blue, 0.0)),
            (0.5, (red, green, blue, 0.08)),
            (1.0, (red, green, blue, 0.34)),
        ],
    )
    ax.imshow(
        normalized,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="bilinear",
        zorder=4,
        aspect="auto",
    )


def geographic_ticks(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    show_x: bool,
    show_y: bool,
) -> None:
    lon_step = 0.05 if bounds == CORE_LONLAT else 0.10
    lat_step = 0.05 if bounds == CORE_LONLAT else 0.10
    lon_ticks = np.arange(
        np.ceil(bounds[0] / lon_step) * lon_step,
        bounds[1] + lon_step / 2,
        lon_step,
    )
    lat_ticks = np.arange(
        np.ceil(bounds[2] / lat_step) * lat_step,
        bounds[3] + lat_step / 2,
        lat_step,
    )
    x_ticks = [TRANSFORMER.transform(lon, np.mean(bounds[2:]))[0] for lon in lon_ticks]
    y_ticks = [TRANSFORMER.transform(np.mean(bounds[:2]), lat)[1] for lat in lat_ticks]
    ax.set_xticks(
        x_ticks,
        [f"{lon:.2f}°E" for lon in lon_ticks] if show_x else [],
    )
    ax.set_yticks(
        y_ticks,
        [f"{lat:.2f}°N" for lat in lat_ticks] if show_y else [],
    )
    ax.tick_params(
        length=2.0,
        width=0.5,
        color=MAP_MUTED,
        labelcolor="#3E464B",
        pad=1.5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(MAP_BOUNDARY)


def label_top_nodes(
    ax: plt.Axes,
    nodes: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
) -> None:
    visible = nodes[
        nodes.geometry.x.between(extent[0], extent[1])
        & nodes.geometry.y.between(extent[2], extent[3])
    ].sort_values("featured_records", ascending=False)
    labelled = visible[visible["location_POI"].isin(POI_LABELS)].head(3)
    local_life_offsets = {
        "溪岸路花鸟市场": (-8, 13),
        "华新路": (-8, -13),
        "幸福路": (8, 4),
    }
    default_offsets = [(-6, 9), (-6, -10), (7, 8)]
    for label_index, (_, row) in enumerate(labelled.iterrows()):
        if row["theme"] == "Local Life":
            dx, dy = local_life_offsets.get(
                row["location_POI"],
                default_offsets[label_index],
            )
        else:
            dx, dy = default_offsets[label_index]
        ax.annotate(
            POI_LABELS[row["location_POI"]],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            fontsize=4.7,
            color=MAP_TEXT,
            bbox={
                "boxstyle": "square,pad=0.08",
                "facecolor": MAP_BG,
                "edgecolor": "none",
                "alpha": 0.78,
            },
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.38,
                "color": MAP_MUTED,
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
            zorder=20,
        )


def draw_panel(
    ax: plt.Axes,
    theme: str,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    summary: pd.Series,
    extent: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    zoom: int,
    show_x: bool,
    show_y: bool,
    annotate: bool,
) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    add_basemap(ax, zoom)
    districts.plot(
        ax=ax,
        facecolor="none",
        edgecolor=MAP_BOUNDARY,
        linewidth=0.42,
        alpha=0.70,
        zorder=3,
    )
    theme_glow(ax, nodes, extent, THEME_COLORS[theme])

    segments = [np.asarray(geometry.coords) for geometry in edges.geometry]
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors=THEME_COLORS[theme],
                linewidths=edge_widths(edges["path_frequency"]),
                alpha=0.42,
                capstyle="round",
                zorder=6,
            )
        )
    ax.scatter(
        nodes.geometry.x,
        nodes.geometry.y,
        s=node_sizes(nodes["featured_records"]),
        c=THEME_COLORS[theme],
        alpha=0.78,
        edgecolors="#F7F9FA",
        linewidths=0.25,
        zorder=9,
    )
    if annotate:
        label_top_nodes(ax, nodes, extent)
    ax.text(
        0.025,
        0.965,
        THEME_LABELS[theme],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=MAP_TEXT,
        bbox={
            "boxstyle": "square,pad=0.20",
            "facecolor": MAP_BG,
            "edgecolor": THEME_COLORS[theme],
            "linewidth": 0.7,
            "alpha": 0.88,
        },
        zorder=25,
    )
    ax.text(
        0.025,
        0.885,
        (
            f"{int(summary.routes):,} routes  |  "
            f"{int(summary.nodes):,} nodes  |  "
            f"{int(summary.directed_edges):,} edges"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.9,
        color=MAP_MUTED,
        zorder=25,
    )
    geographic_ticks(ax, bounds, show_x=show_x, show_y=show_y)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def make_four_panel_map(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    summary: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    stem: str,
    zoom: int,
    annotate: bool,
) -> None:
    extent = projected_extent(bounds)
    district_gdf = gpd.read_file(BOUNDARY_FILE).to_crs(MAP_CRS)
    node_gdf = gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes.lng, nodes.lat),
        crs=WGS84,
    ).to_crs(MAP_CRS)

    node_lookup = node_gdf.drop_duplicates("cluster_id").set_index("cluster_id")
    edge_rows = []
    geometries = []
    for row in edges.itertuples():
        if row.start not in node_lookup.index or row.end not in node_lookup.index:
            continue
        start = node_lookup.loc[row.start].geometry
        end = node_lookup.loc[row.end].geometry
        edge_rows.append(row._asdict())
        geometries.append(LineString([(start.x, start.y), (end.x, end.y)]))
    edge_gdf = gpd.GeoDataFrame(edge_rows, geometry=geometries, crs=MAP_CRS)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.85))
    for index, (ax, theme) in enumerate(zip(axes.flat, THEME_ORDER)):
        draw_panel(
            ax,
            theme,
            node_gdf[node_gdf["theme"] == theme],
            edge_gdf[edge_gdf["theme"] == theme],
            district_gdf,
            summary.set_index("theme").loc[theme],
            extent,
            bounds,
            zoom,
            show_x=index >= 2,
            show_y=index % 2 == 0,
            annotate=annotate,
        )
        ax.text(
            -0.075,
            1.015,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="bottom",
        )

    node_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="#89969D",
            markeredgecolor="#F7F9FA",
            markeredgewidth=0.4,
            markersize=np.sqrt(node_sizes(pd.Series([value]))[0]),
            label=str(value),
        )
        for value in [10, 100, 500]
    ]
    edge_handles = [
        Line2D(
            [],
            [],
            color="#68777F",
            lw=edge_widths(pd.Series([value]))[0],
            solid_capstyle="round",
            label=str(value),
        )
        for value in [1, 8, 32]
    ]
    node_legend = fig.legend(
        handles=node_handles,
        labels=["10", "100", "500"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.29, 0.010),
        frameon=False,
        title="Theme-specific node frequency",
        title_fontsize=5.7,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    edge_legend = fig.legend(
        handles=edge_handles,
        labels=["1", "8", "32"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.73, 0.010),
        frameon=False,
        title="Path frequency",
        title_fontsize=5.7,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    node_legend.get_title().set_fontweight("bold")
    edge_legend.get_title().set_fontweight("bold")
    node_legend.set_zorder(30)
    edge_legend.set_zorder(30)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.985,
        bottom=0.095,
        wspace=0.08,
        hspace=0.08,
    )
    save_figure(fig, stem)


def main() -> None:
    apply_style()
    _, nodes, edges, summary = build_subnetworks()
    make_four_panel_map(
        nodes,
        edges,
        summary,
        CORE_LONLAT,
        "fig_theme_subnetworks_core_nature_style",
        zoom=12,
        annotate=True,
    )
    make_four_panel_map(
        nodes,
        edges,
        summary,
        MUNICIPAL_LONLAT,
        "figS_theme_subnetworks_municipality_nature_style",
        zoom=10,
        annotate=False,
    )
    print(summary.to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
