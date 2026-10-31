from __future__ import annotations

import colorsys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.text import Text
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
STAGE19 = ROOT / "results" / "subnetworks"
STAGE21 = ROOT / "results" / "communities"
DATA = ROOT / "data" / "spatial"
STAGE21.mkdir(parents=True, exist_ok=True)

ASSIGNMENT_FILE = STAGE21 / "network_community_node_assignments.csv"
PROFILE_FILE = STAGE21 / "network_community_profiles.csv"
DIAGNOSTIC_FILE = STAGE21 / "network_community_diagnostics.csv"
OVERALL_EDGE_FILE = STAGE21 / "overall_network_edges.csv"
THEME_EDGE_FILE = STAGE19 / "theme_subnetwork_edges.csv"
KEY_NODE_FILE = STAGE19 / "theme_subnetwork_key_nodes_for_pad_callouts.csv"
BOUNDARY_FILE = DATA / "xiamen_district" / "xiamen_district.shp"
LAYOUT_FILE = STAGE21 / "network_community_forceatlas2_layouts.csv"

THEME_ORDER = [
    "Commercial & Tourism",
    "Nature",
    "Historical & Art",
    "Local Life",
]
NETWORK_ORDER = ["Overall", *THEME_ORDER]
THEME_LABELS = {
    "Commercial & Tourism": "Commercial & tourism",
    "Nature": "Nature",
    "Historical & Art": "Historical & art",
    "Local Life": "Local life",
}
THEME_COLORS = {
    "Commercial & Tourism": "#2B9DBD",
    "Nature": "#6FA83A",
    "Historical & Art": "#C95E9A",
    "Local Life": "#E36B32",
}
PAD_COLORS = {
    "P": "#C65D50",
    "A": "#D39C31",
    "D": "#4E79A7",
}
COMMUNITY_COLORS = {
    "C1": "#287D98",
    "C2": "#D06E48",
    "C3": "#699A51",
    "C4": "#B55F8C",
    "C5": "#776DAA",
    "C6": "#C29434",
    "C7": "#469486",
    "C8": "#9B6A50",
    "C9": "#6684A7",
    "C10": "#929344",
}
OTHER_LINKED_COLOR = "#9CA4A8"
ISOLATE_COLOR = "#D9DDDF"
MINOR_COMMUNITY_COLOR = "#767D81"
MAP_CRS = "EPSG:32650"
CGCS2000 = "EPSG:4490"
OVERALL_MAP_BOUNDS = (118.04, 118.19, 24.415, 24.525)
THEME_MAP_BOUNDS = (117.98, 118.21, 24.415, 24.66)
TO_MAP = Transformer.from_crs(CGCS2000, MAP_CRS, always_xy=True)
PAD_CALLOUT_X = 1.16
PAD_CALLOUT_Y = (0.60, 0.16, -0.28)
PAD_CALLOUT_WIDTH = 0.70
PAD_CALLOUT_HEIGHT = 0.28

POI_LABELS = {
    "八市": "Eighth Market",
    "中山路步行街": "Zhongshan Road",
    "华新路": "Huaxin Road",
    "幸福路": "Xingfu Road",
    "溪岸路花鸟市场": "Xi'an Road Market",
    "南普陀寺": "Nanputuo Temple",
    "厦门大学": "Xiamen University",
    "菽庄花园": "Shuzhuang Garden",
    "日光岩": "Sunlight Rock",
    "音乐广场": "Music Square",
    "白城沙滩": "Baicheng Beach",
    "白鹭洲公园": "Bailuzhou Park",
    "海湾公园": "Haiwan Park",
    "百家村": "Baijia Village",
    "沙坡尾艺术西区": "Shapowei Art Zone",
    "黄厝沙滩": "Island Ring Boulevard",
    "莲花公园": "Lianhua Park",
    "东荣广场": "Dongrong Square",
    "万石沧海": "Wanshi Canghai",
    "张万森家门口的炮仗花": "Huaxin neighborhood",
    "最美转角": "Beautiful Corner",
    "三丘田码头": "Sanqiutian Pier",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 6.5,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.6,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def overall_community_color(community_id: object) -> str:
    if pd.isna(community_id):
        return OTHER_LINKED_COLOR
    return COMMUNITY_COLORS.get(str(community_id), MINOR_COMMUNITY_COLOR)


def mix_color(color: str, target: str, fraction: float) -> str:
    source_rgb = np.asarray(to_rgb(color))
    target_rgb = np.asarray(to_rgb(target))
    return to_hex((1 - fraction) * source_rgb + fraction * target_rgb)


def theme_community_palette(
    theme: str,
    communities: int,
) -> dict[str, str]:
    hue, _, _ = colorsys.rgb_to_hls(*to_rgb(THEME_COLORS[theme]))
    lightness_saturation = [
        (0.27, 0.75),
        (0.80, 0.45),
        (0.42, 0.78),
        (0.67, 0.62),
        (0.32, 0.40),
        (0.75, 0.72),
        (0.52, 0.80),
        (0.87, 0.48),
        (0.47, 0.38),
        (0.61, 0.75),
        (0.36, 0.58),
        (0.72, 0.38),
        (0.56, 0.45),
        (0.85, 0.65),
    ]
    return {
        f"C{index}": to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))
        for index, (lightness, saturation) in enumerate(
            lightness_saturation,
            start=1,
        )
        if index <= communities
    }


def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes["cluster_id"].astype(int))
    for row in edges.itertuples():
        start = int(row.start)
        end = int(row.end)
        weight = float(row.path_frequency)
        if graph.has_edge(start, end):
            graph[start][end]["weight"] += weight
        else:
            graph.add_edge(start, end, weight=weight)
    return graph


def normalize_positions(
    positions: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    node_ids = list(positions)
    coordinates = np.asarray([positions[node] for node in node_ids], dtype=float)
    coordinates -= np.nanmedian(coordinates, axis=0)
    span = np.nanpercentile(np.abs(coordinates), 99.5, axis=0)
    span[span == 0] = 1.0
    coordinates /= span
    coordinates = np.clip(coordinates, -1.08, 1.08)
    return {
        node: coordinates[index]
        for index, node in enumerate(node_ids)
    }


def make_layouts(
    assignments: pd.DataFrame,
    overall_edges: pd.DataFrame,
    theme_edges: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for index, network in enumerate(NETWORK_ORDER):
        nodes = assignments[
            (assignments["network"] == network)
            & (assignments["community_status"] == "LCC community")
        ].copy()
        edges = (
            overall_edges.copy()
            if network == "Overall"
            else theme_edges[theme_edges["theme"] == network].copy()
        )
        valid = set(nodes["cluster_id"].astype(int))
        edges = edges[
            edges["start"].isin(valid) & edges["end"].isin(valid)
        ]
        graph = build_graph(nodes, edges)
        node_mass = {
            int(row.cluster_id): 1.0 + np.log1p(row.community_weighted_degree)
            for row in nodes.itertuples()
        }
        node_size = {
            int(row.cluster_id): 0.5 + 0.08 * np.sqrt(row.featured_records)
            for row in nodes.itertuples()
        }
        positions = nx.forceatlas2_layout(
            graph,
            max_iter=220 if network == "Overall" else 180,
            scaling_ratio=4.0,
            gravity=1.2,
            strong_gravity=True,
            node_mass=node_mass,
            node_size=node_size,
            weight="weight",
            linlog=False,
            seed=20260727 + index,
        )
        positions = normalize_positions(positions)
        frames.append(
            pd.DataFrame(
                {
                    "network": network,
                    "cluster_id": list(positions),
                    "layout_x": [positions[node][0] for node in positions],
                    "layout_y": [positions[node][1] for node in positions],
                }
            )
        )
    layout = pd.concat(frames, ignore_index=True)
    layout.to_csv(LAYOUT_FILE, index=False)
    return layout


def project_nodes(nodes: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        nodes.copy(),
        geometry=gpd.points_from_xy(nodes["lng"], nodes["lat"]),
        crs=CGCS2000,
    ).to_crs(MAP_CRS)


def geographic_ticks(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    show_x: bool = True,
) -> None:
    if bounds[1] - bounds[0] > 0.20:
        lon_ticks = np.asarray([118.00, 118.10, 118.20])
        lat_ticks = np.asarray([24.45, 24.55, 24.65])
    else:
        lon_ticks = np.asarray([118.05, 118.10, 118.15])
        lat_ticks = np.asarray([24.45, 24.50])
    center_lat = (bounds[2] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[1]) / 2
    x_ticks = [TO_MAP.transform(lon, center_lat)[0] for lon in lon_ticks]
    y_ticks = [TO_MAP.transform(center_lon, lat)[1] for lat in lat_ticks]
    ax.set_xticks(
        x_ticks,
        [f"{value:.2f}°E" for value in lon_ticks] if show_x else [],
    )
    ax.set_yticks(y_ticks, [f"{value:.2f}°N" for value in lat_ticks])
    ax.tick_params(length=2.0, width=0.45, pad=1.2, color="#40464A")
    for spine in ax.spines.values():
        spine.set_color("#22282B")
        spine.set_linewidth(0.55)


def projected_limits(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    west, south = TO_MAP.transform(bounds[0], bounds[2])
    east, north = TO_MAP.transform(bounds[1], bounds[3])
    return west, east, south, north


def edge_segments_map(
    nodes: gpd.GeoDataFrame,
    edges: pd.DataFrame,
) -> tuple[list[list[tuple[float, float]]], np.ndarray]:
    lookup = nodes.set_index("cluster_id").geometry
    segments = []
    weights = []
    for row in edges.itertuples():
        start = int(row.start)
        end = int(row.end)
        if start not in lookup.index or end not in lookup.index:
            continue
        a = lookup.loc[start]
        b = lookup.loc[end]
        segments.append([(a.x, a.y), (b.x, b.y)])
        weights.append(float(row.path_frequency))
    return segments, np.asarray(weights, dtype=float)


def draw_map(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    community_colors: dict[str, str],
    edge_color: str,
    show_x: bool,
    compact: bool = False,
    edge_width_multiplier: float = 1.0,
    edge_alpha: float = 0.30,
) -> None:
    projected = project_nodes(nodes)
    west, east, south, north = projected_limits(bounds)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    boundaries.plot(
        ax=ax,
        facecolor="#FAFAF8",
        edgecolor="#171B1D",
        linewidth=0.48,
        zorder=0,
    )
    segments, edge_weights = edge_segments_map(projected, edges)
    if len(segments):
        widths = edge_width_multiplier * (
            0.14 + 0.24 * np.log1p(edge_weights)
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_color,
                linewidths=widths,
                alpha=edge_alpha,
                zorder=1,
            )
        )

    isolate = projected[projected["community_status"] == "Isolated"]
    other = projected[
        projected["community_status"] == "Other linked component"
    ]
    lcc = projected[projected["community_status"] == "LCC community"]
    ax.scatter(
        isolate.geometry.x,
        isolate.geometry.y,
        s=2.0 if compact else 2.5,
        c=ISOLATE_COLOR,
        alpha=0.55,
        linewidth=0,
        zorder=2,
    )
    ax.scatter(
        other.geometry.x,
        other.geometry.y,
        s=3.0 if compact else 3.8,
        c=OTHER_LINKED_COLOR,
        alpha=0.60,
        linewidth=0,
        zorder=3,
    )
    for community_id, frame in lcc.groupby("community_id"):
        sizes = (
            3.0 + 2.1 * np.sqrt(frame["featured_records"])
            if compact
            else 4.0 + 2.7 * np.sqrt(frame["featured_records"])
        )
        ax.scatter(
            frame.geometry.x,
            frame.geometry.y,
            s=sizes,
            c=community_colors.get(str(community_id), MINOR_COMMUNITY_COLOR),
            edgecolor="white",
            linewidth=0.22,
            alpha=0.90,
            zorder=4,
        )
    geographic_ticks(ax, bounds=bounds, show_x=show_x)
    ax.set_aspect("equal")


def draw_pad_callout(
    ax: plt.Axes,
    row: pd.Series,
    anchor: tuple[float, float],
) -> None:
    box_x, box_y = anchor
    width = PAD_CALLOUT_WIDTH
    height = PAD_CALLOUT_HEIGHT
    ax.annotate(
        "",
        xy=(float(row["plot_x"]), float(row["plot_y"])),
        xytext=(box_x, box_y + height / 2),
        arrowprops={
            "arrowstyle": "-",
            "color": "#697277",
            "lw": 0.48,
            "shrinkA": 0,
            "shrinkB": 2,
        },
        zorder=8,
    )
    ax.add_patch(
        Rectangle(
            (box_x, box_y),
            width,
            height,
            facecolor="white",
            edgecolor="#7A8489",
            linewidth=0.48,
            zorder=9,
        )
    )
    label = POI_LABELS.get(
        str(row["location_POI"]),
        str(row["location_POI"]),
    )
    if len(label) > 23:
        label = f"{label[:21]}..."
    ax.text(
        box_x + 0.025,
        box_y + height - 0.035,
        label,
        ha="left",
        va="top",
        fontsize=4.25,
        fontweight="bold",
        color="#202629",
        zorder=11,
    )
    baseline = box_y + 0.100
    chart_height = 0.075
    bar_width = 0.058
    bar_x = [box_x + 0.17, box_x + 0.34, box_x + 0.51]
    values = [
        float(row["direct_P"]),
        float(row["direct_A"]),
        float(row["direct_D"]),
    ]
    ax.plot(
        [box_x + 0.095, box_x + 0.625],
        [baseline, baseline],
        color="#7D8589",
        linewidth=0.35,
        zorder=10,
    )
    for x, value, pad in zip(bar_x, values, ["P", "A", "D"]):
        value = np.clip(value, -1, 1)
        y = baseline if value >= 0 else baseline + value * chart_height
        ax.add_patch(
            Rectangle(
                (x, y),
                bar_width,
                abs(value) * chart_height,
                facecolor=PAD_COLORS[pad],
                edgecolor="none",
                zorder=11,
            )
        )
        ax.text(
            x + bar_width / 2,
            box_y + 0.018,
            pad,
            ha="center",
            va="bottom",
            fontsize=3.7,
            color=PAD_COLORS[pad],
            zorder=11,
        )


def highlight_key_nodes_on_map(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    key_nodes: pd.DataFrame,
    community_colors: dict[str, str],
) -> gpd.GeoDataFrame:
    projected = project_nodes(nodes)
    selected = key_nodes[["cluster_id"]].merge(
        projected[
            [
                "cluster_id",
                "community_id",
                "featured_records",
                "geometry",
            ]
        ],
        on="cluster_id",
        how="inner",
        validate="one_to_one",
    )
    for row in selected.itertuples():
        ax.scatter(
            [row.geometry.x],
            [row.geometry.y],
            s=10 + 2.6 * np.sqrt(row.featured_records),
            facecolor=community_colors.get(
                str(row.community_id),
                MINOR_COMMUNITY_COLOR,
            ),
            edgecolor="#171B1D",
            linewidth=0.8,
            zorder=7,
        )
    return gpd.GeoDataFrame(selected, geometry="geometry", crs=MAP_CRS)


def connect_map_nodes_to_pad(
    figure: plt.Figure,
    map_ax: plt.Axes,
    network_ax: plt.Axes,
    selected: gpd.GeoDataFrame,
) -> None:
    for (_, row), box_y in zip(selected.iterrows(), PAD_CALLOUT_Y):
        connector = ConnectionPatch(
            xyA=(float(row.geometry.x), float(row.geometry.y)),
            coordsA=map_ax.transData,
            xyB=(
                PAD_CALLOUT_X,
                box_y + 0.72 * PAD_CALLOUT_HEIGHT,
            ),
            coordsB=network_ax.transData,
            arrowstyle="-",
            color="#697277",
            linewidth=0.42,
            alpha=0.58,
            shrinkA=3,
            shrinkB=0,
            clip_on=False,
            zorder=0.8,
        )
        figure.add_artist(connector)


def draw_network(
    ax: plt.Axes,
    network: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    layouts: pd.DataFrame,
    label_communities: int,
    community_colors: dict[str, str],
    edge_width_multiplier: float = 1.0,
    edge_alpha: float = 0.32,
    key_nodes: pd.DataFrame | None = None,
) -> None:
    layout = layouts[layouts["network"] == network]
    frame = (
        nodes[nodes["community_status"] == "LCC community"]
        .merge(layout, on=["network", "cluster_id"], validate="one_to_one")
        .copy()
    )
    frame["plot_x"] = frame["layout_x"]
    frame["plot_y"] = frame["layout_y"]
    valid = set(frame["cluster_id"].astype(int))
    edges = edges[edges["start"].isin(valid) & edges["end"].isin(valid)]
    positions = frame.set_index("cluster_id")[["plot_x", "plot_y"]]
    segments = [
        [
            tuple(positions.loc[int(row.start)]),
            tuple(positions.loc[int(row.end)]),
        ]
        for row in edges.itertuples()
    ]
    weights = edges["path_frequency"].to_numpy(dtype=float)
    if len(segments):
        ax.add_collection(
            LineCollection(
                segments,
                colors="#A9AFB2",
                linewidths=edge_width_multiplier
                * (0.14 + 0.28 * np.log1p(weights)),
                alpha=edge_alpha,
                zorder=1,
            )
        )
    for community_id, community in frame.groupby("community_id"):
        sizes = 5.0 + 2.8 * np.sqrt(community["featured_records"])
        ax.scatter(
            community["plot_x"],
            community["plot_y"],
            s=sizes,
            c=community_colors.get(
                str(community_id),
                MINOR_COMMUNITY_COLOR,
            ),
            edgecolor="white",
            linewidth=0.25,
            alpha=0.92,
            zorder=2,
        )
    top_communities = (
        frame.groupby("community_id")
        .size()
        .sort_values(ascending=False)
        .head(label_communities)
        .index
    )
    for community_id in top_communities:
        community = frame[frame["community_id"] == community_id]
        anchor = community.nlargest(
            1,
            ["community_weighted_degree", "featured_records"],
        ).iloc[0]
        label = ax.text(
            anchor["plot_x"],
            anchor["plot_y"],
            community_id,
            ha="center",
            va="center",
            color="white",
            fontsize=4.8,
            fontweight="bold",
            zorder=5,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.4, foreground="#202427")]
        )

    if key_nodes is not None:
        selected = key_nodes.merge(
            frame[
                [
                    "cluster_id",
                    "plot_x",
                    "plot_y",
                    "community_id",
                ]
            ],
            on="cluster_id",
            how="inner",
            validate="one_to_one",
        )
        for (_, row), box_y in zip(selected.iterrows(), PAD_CALLOUT_Y):
            color = community_colors.get(
                str(row["community_id"]),
                MINOR_COMMUNITY_COLOR,
            )
            ax.scatter(
                [row["plot_x"]],
                [row["plot_y"]],
                s=18 + 1.8 * np.sqrt(row["featured_records"]),
                facecolor=color,
                edgecolor="#171B1D",
                linewidth=0.8,
                zorder=7,
            )
            draw_pad_callout(ax, row, (PAD_CALLOUT_X, box_y))

    ax.set_xlim(-1.12, 1.90 if key_nodes is not None else 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.03,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def scale_figure_text(figure: plt.Figure, factor: float) -> None:
    for text_artist in figure.findobj(Text):
        text_artist.set_fontsize(text_artist.get_fontsize() * factor)


def plot_overall(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    overall_edges: pd.DataFrame,
    layouts: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
) -> None:
    nodes = assignments[assignments["network"] == "Overall"].copy()
    profile = (
        profiles[profiles["network"] == "Overall"]
        .sort_values("nodes", ascending=False)
        .head(10)
        .copy()
    )
    diagnostic = diagnostics.set_index("network").loc["Overall"]

    figure = plt.figure(figsize=(8.5, 6.5), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 0.92],
        left=0.070,
        right=0.975,
        bottom=0.075,
        top=0.955,
        wspace=0.20,
        hspace=0.25,
    )
    map_ax = figure.add_subplot(grid[0, 0])
    network_ax = figure.add_subplot(grid[0, 1])
    theme_ax = figure.add_subplot(grid[1, 0])
    pad_ax = figure.add_subplot(grid[1, 1])

    draw_map(
        map_ax,
        nodes,
        overall_edges,
        boundaries,
        bounds=OVERALL_MAP_BOUNDS,
        community_colors=COMMUNITY_COLORS,
        edge_color="#9AA1A5",
        show_x=True,
        compact=False,
    )
    draw_network(
        network_ax,
        "Overall",
        nodes,
        overall_edges,
        layouts,
        label_communities=10,
        community_colors=COMMUNITY_COLORS,
    )
    map_ax.set_title(
        "Geographic distribution",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    network_ax.set_title(
        "Network and community structure",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    network_ax.text(
        0.5,
        -0.035,
        (
            f"Largest connected component: {int(diagnostic.lcc_nodes)} nodes"
            f"\n{int(diagnostic.communities)} communities"
            f"   |   weighted modularity Q = "
            f"{diagnostic.weighted_modularity:.3f}"
        ),
        transform=network_ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        fontweight="normal",
        color="#283237",
        linespacing=1.25,
    )

    context_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COMMUNITY_COLORS["C1"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.3,
            label="Largest connected component",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OTHER_LINKED_COLOR,
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=4.0,
            label="Other connected components",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ISOLATE_COLOR,
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=4.0,
            label="Isolated nodes",
        ),
    ]
    map_ax.legend(
        handles=context_handles,
        loc="upper left",
        bbox_to_anchor=(0.018, 0.985),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.25,
        labelspacing=0.28,
        fontsize=5.2,
    )

    y = np.arange(len(profile))
    left = np.zeros(len(profile))
    for theme in THEME_ORDER:
        values = profile[f"nodes_{theme}"].to_numpy(dtype=float)
        shares = values / profile["nodes"].to_numpy(dtype=float)
        theme_ax.barh(
            y,
            shares,
            left=left,
            height=0.66,
            color=THEME_COLORS[theme],
            edgecolor="white",
            linewidth=0.3,
            label=THEME_LABELS[theme],
        )
        left += shares
    theme_ax.set_yticks(y, profile["community_id"])
    theme_ax.set_ylim(len(profile) - 0.5, -1.45)
    theme_ax.set_xlim(0, 1)
    theme_ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    theme_ax.set_xlabel("Node composition (%)", labelpad=2)
    theme_ax.set_title(
        "Route-theme composition",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    for index, count in enumerate(profile["nodes"]):
        theme_ax.text(
            1.015,
            index,
            f"n={int(count)}",
            transform=theme_ax.get_yaxis_transform(),
            va="center",
            fontsize=5.0,
            color="#454B4F",
        )
    theme_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.995),
        ncol=2,
        borderaxespad=0,
        handlelength=1.0,
        columnspacing=0.8,
        labelspacing=0.3,
    )
    theme_ax.spines[["top", "right"]].set_visible(False)

    pad_rows = []
    pad_offsets = {"P": -0.20, "A": 0.0, "D": 0.20}
    for pad, column in [
        ("P", "direct_P"),
        ("A", "direct_A"),
        ("D", "direct_D"),
    ]:
        summary = (
            nodes[
                (nodes["community_status"] == "LCC community")
                & nodes["community_id"].isin(profile["community_id"])
            ]
            .groupby("community_id")[column]
            .agg(["count", "mean", "std"])
            .reindex(profile["community_id"])
        )
        means = summary["mean"].to_numpy(dtype=float)
        standard_deviations = summary["std"].fillna(0).to_numpy(dtype=float)
        pad_ax.errorbar(
            means,
            y + pad_offsets[pad],
            xerr=standard_deviations,
            fmt="o",
            color=PAD_COLORS[pad],
            ecolor=PAD_COLORS[pad],
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.35,
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
            label={"P": "Pleasure", "A": "Arousal", "D": "Dominance"}[pad],
            zorder=3,
        )
        for community_id, row in summary.iterrows():
            pad_rows.append(
                {
                    "community_id": community_id,
                    "PAD_dimension": pad,
                    "nodes_with_direct_evidence": int(row["count"]),
                    "mean": float(row["mean"]),
                    "standard_deviation": float(row["std"]),
                }
            )
    pd.DataFrame(pad_rows).to_csv(
        STAGE21 / "overall_community_pad_mean_sd.csv",
        index=False,
    )
    pad_ax.axvline(
        0,
        color="#646B6F",
        linestyle=(0, (3, 2)),
        linewidth=0.65,
        zorder=1,
    )
    pad_ax.set_xlim(-0.72, 1.28)
    pad_ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    pad_ax.set_yticks(y, profile["community_id"])
    pad_ax.set_ylim(len(profile) - 0.5, -1.15)
    pad_ax.set_xlabel("PAD score (mean +/- SD)", labelpad=2)
    pad_ax.set_title(
        "Community-level PAD profiles",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    pad_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.985),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    pad_ax.grid(
        axis="x",
        color="#D8DCDE",
        linewidth=0.45,
        alpha=0.75,
        zorder=0,
    )
    pad_ax.tick_params(axis="y", length=0)
    pad_ax.spines[["top", "right"]].set_visible(False)

    add_panel_label(map_ax, "a")
    add_panel_label(network_ax, "b")
    add_panel_label(theme_ax, "c")
    add_panel_label(pad_ax, "d")
    scale_figure_text(figure, 1.08)

    base = STAGE21 / "fig_overall_network_communities_nature_style"
    figure.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def plot_subnetworks(
    assignments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    theme_edges: pd.DataFrame,
    layouts: pd.DataFrame,
    boundaries: gpd.GeoDataFrame,
    key_nodes: pd.DataFrame,
) -> None:
    diagnostic_lookup = diagnostics.set_index("network")
    figure = plt.figure(figsize=(10.0, 4.0), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        2,
        left=0.052,
        right=0.992,
        top=0.975,
        bottom=0.105,
        wspace=0.075,
        hspace=0.13,
    )
    panel_labels = ["a", "b", "c", "d"]
    aligned_annotations = []

    for index, theme in enumerate(THEME_ORDER):
        row, column = divmod(index, 2)
        inner = outer[row, column].subgridspec(
            1,
            2,
            width_ratios=[1.0, 1.65],
            wspace=0.018,
        )
        map_ax = figure.add_subplot(inner[0, 0])
        network_ax = figure.add_subplot(inner[0, 1])
        nodes = assignments[assignments["network"] == theme].copy()
        edges = theme_edges[theme_edges["theme"] == theme].copy()
        diagnostic = diagnostic_lookup.loc[theme]
        community_colors = theme_community_palette(
            theme,
            int(diagnostic.communities),
        )

        theme_key_nodes = key_nodes[key_nodes["theme"] == theme]
        draw_map(
            map_ax,
            nodes,
            edges,
            boundaries,
            bounds=OVERALL_MAP_BOUNDS,
            community_colors=community_colors,
            edge_color=THEME_COLORS[theme],
            show_x=True,
            compact=True,
            edge_width_multiplier=1.25,
            edge_alpha=0.38,
        )
        selected_map_nodes = highlight_key_nodes_on_map(
            map_ax,
            nodes,
            theme_key_nodes,
            community_colors,
        )
        draw_network(
            network_ax,
            theme,
            nodes,
            edges,
            layouts,
            label_communities=min(10, int(diagnostic.communities)),
            community_colors=community_colors,
            edge_width_multiplier=1.25,
            edge_alpha=0.40,
            key_nodes=theme_key_nodes,
        )
        connect_map_nodes_to_pad(
            figure,
            map_ax,
            network_ax,
            selected_map_nodes,
        )
        map_ax.text(
            0.018,
            0.968,
            THEME_LABELS[theme],
            transform=map_ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            fontweight="bold",
            color="#171B1D",
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": "white",
                "edgecolor": THEME_COLORS[theme],
                "linewidth": 0.85,
                "alpha": 0.94,
            },
            zorder=10,
        )
        aligned_annotations.append(
            (
                map_ax,
                network_ax,
                panel_labels[index],
                int(diagnostic.communities),
                float(diagnostic.weighted_modularity),
            )
        )

    figure.canvas.draw()
    for (
        map_ax,
        network_ax,
        panel_label,
        community_count,
        modularity,
    ) in aligned_annotations:
        map_box = map_ax.get_position()
        network_box = network_ax.get_position()
        baseline_y = map_box.y1 + 0.006
        figure.text(
            map_box.x0 - 0.018,
            baseline_y,
            panel_label,
            ha="right",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#111111",
        )
        figure.text(
            network_box.x0,
            baseline_y,
            f"{community_count} communities   |   Q={modularity:.3f}",
            ha="left",
            va="bottom",
            fontsize=6.3,
            fontweight="bold",
            color="#283237",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4F5960",
            markersize=4.3,
            label="Largest connected component communities",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OTHER_LINKED_COLOR,
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=4.0,
            label="Other connected components",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ISOLATE_COLOR,
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=4.0,
            label="Isolated nodes",
        ),
        *[
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=PAD_COLORS[pad],
                markeredgecolor="none",
                markersize=4.0,
                label=label,
            )
            for pad, label in [
                ("P", "Pleasure"),
                ("A", "Arousal"),
                ("D", "Dominance"),
            ]
        ],
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.0,
        fontsize=5.2,
    )
    base = STAGE21 / "fig_theme_subnetwork_communities_nature_style"
    figure.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(base.with_suffix(".pdf"), facecolor="white")
    figure.savefig(base.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def make_tables(
    profiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    summary = diagnostics.merge(associations, on="network", how="left")
    columns = [
        "network",
        "all_nodes",
        "linked_nodes",
        "lcc_nodes",
        "lcc_share_linked",
        "communities",
        "weighted_modularity",
        "largest_community_share_lcc",
        "mean_ari_to_best",
        "weighted_mean_community_radius_km",
    ]
    summary[columns].to_csv(
        STAGE21 / "table_network_community_summary.csv",
        index=False,
    )

    overall_columns = [
        "community_id",
        "nodes",
        "share_of_lcc_nodes",
        "featured_records",
        "dominant_theme",
        "dominant_theme_share",
        "dominant_district",
        "median_spatial_radius_km",
        "mean_P",
        "mean_A",
        "mean_D",
        "top_nodes",
    ]
    profiles[profiles["network"] == "Overall"][overall_columns].to_csv(
        STAGE21 / "table_overall_community_profiles.csv",
        index=False,
    )
    profiles[profiles["network"] != "Overall"].to_csv(
        STAGE21 / "table_theme_subnetwork_community_profiles.csv",
        index=False,
    )


def main() -> None:
    apply_style()
    assignments = pd.read_csv(ASSIGNMENT_FILE)
    profiles = pd.read_csv(PROFILE_FILE)
    diagnostics = pd.read_csv(DIAGNOSTIC_FILE)
    associations = pd.read_csv(
        STAGE21 / "network_community_spatial_theme_associations.csv"
    )
    overall_edges = pd.read_csv(OVERALL_EDGE_FILE)
    theme_edges = pd.read_csv(THEME_EDGE_FILE)
    key_nodes = pd.read_csv(KEY_NODE_FILE)
    boundaries = gpd.read_file(BOUNDARY_FILE).to_crs(MAP_CRS)
    layouts = make_layouts(assignments, overall_edges, theme_edges)
    plot_overall(
        assignments,
        profiles,
        diagnostics,
        overall_edges,
        layouts,
        boundaries,
    )
    plot_subnetworks(
        assignments,
        diagnostics,
        theme_edges,
        layouts,
        boundaries,
        key_nodes,
    )
    make_tables(profiles, diagnostics, associations)
    print("Saved community figures, layouts, and summary tables.")


if __name__ == "__main__":
    main()
