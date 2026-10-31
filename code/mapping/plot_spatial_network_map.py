from __future__ import annotations

from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "data" / "analysis"
DATA = ROOT / "data" / "spatial"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NODE_FILE = STAGE2 / "reference_nodes_100m_1500m.csv"
EDGE_FILE = STAGE2 / "reference_edges_100m_1500m.csv"
BOUNDARY_FILE = DATA / "xiamen_district" / "xiamen_district.shp"

THEME_ORDER = ["Commercial & Tourism", "Nature", "Historical & Art", "Local Life"]
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
DISTRICT_LABELS = {
    "SiMing": "Siming",
    "HuLi": "Huli",
    "JiMei": "Jimei",
    "HaiCang": "Haicang",
    "TongAn": "Tong'an",
    "XiangAn": "Xiang'an",
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
LABEL_OFFSETS = {
    "幸福路": (-72, -17),
    "黄厝沙滩": (10, -7),
    "华新路": (-72, 12),
    "八市": (-78, -3),
    "南普陀寺": (10, -10),
    "集美学村": (11, 14),
    "十里长堤": (11, -7),
    "溪岸路花鸟市场": (24, 23),
    "百家村": (31, 7),
    "陈嘉庚故居": (32, 23),
}

WGS84 = "EPSG:4326"
MAP_CRS = "EPSG:32650"  # WGS 84 / UTM zone 50N; distances are represented in metres.
TRANSFORMER = Transformer.from_crs(WGS84, MAP_CRS, always_xy=True)
MAIN_LONLAT = (118.00, 118.20, 24.415, 24.635)
CITY_LONLAT = (117.903, 118.349, 24.420, 24.860)

MAP_BG = "#11171B"
MAP_TEXT = "#F2F4F5"
MAP_MUTED = "#AEB8BE"
MAP_BOUNDARY = "#C8D0D4"
LEGEND_BG = "#151D22"


def clean_theme(value: object) -> str:
    return str(value).replace("Commericial", "Commercial").replace("Local_Life", "Local Life")


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": MAP_BG,
        }
    )


def projected_extent(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin = TRANSFORMER.transform(bounds[0], bounds[2])
    xmax, ymax = TRANSFORMER.transform(bounds[1], bounds[3])
    return xmin, xmax, ymin, ymax


def node_sizes(weights: pd.Series, main: bool = True) -> np.ndarray:
    scale = 3.25 if main else 1.05
    base = 2.5 if main else 0.7
    return base + scale * np.sqrt(weights.to_numpy(dtype=float))


def edge_widths(weights: pd.Series, main: bool = True) -> np.ndarray:
    scale = 0.76 if main else 0.20
    base = 0.22 if main else 0.05
    return base + scale * np.log1p(weights.to_numpy(dtype=float))


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


def add_scale_bar(ax: plt.Axes, length_km: int, x: float, y: float) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000
    x0 = xmin + x * (xmax - xmin)
    y0 = ymin + y * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color=MAP_TEXT, lw=1.4, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x0], [y0 - 250, y0 + 250], color=MAP_TEXT, lw=0.8, zorder=20)
    ax.plot([x0 + length, x0 + length], [y0 - 250, y0 + 250], color=MAP_TEXT, lw=0.8, zorder=20)
    ax.text(x0 + length / 2, y0 + 520, f"{length_km} km", ha="center", va="bottom", fontsize=5.8, color=MAP_TEXT, zorder=20)


def add_north_arrow(ax: plt.Axes, x: float = 0.955, y: float = 0.91) -> None:
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.065),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=MAP_TEXT,
        arrowprops=dict(arrowstyle="-|>", color=MAP_TEXT, lw=0.9, mutation_scale=8),
        zorder=30,
    )


def geographic_ticks(ax: plt.Axes) -> None:
    lon_ticks = np.arange(118.00, 118.201, 0.05)
    lat_ticks = np.arange(24.45, 24.636, 0.05)
    x_ticks = [TRANSFORMER.transform(lon, 24.50)[0] for lon in lon_ticks]
    y_ticks = [TRANSFORMER.transform(118.10, lat)[1] for lat in lat_ticks]
    ax.set_xticks(x_ticks, [f"{lon:.2f}°E" for lon in lon_ticks])
    ax.set_yticks(y_ticks, [f"{lat:.2f}°N" for lat in lat_ticks])
    ax.tick_params(length=2.2, width=0.55, color=MAP_MUTED, labelcolor="#3E464B", pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(MAP_BOUNDARY)


def build_spatial_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = pd.read_csv(NODE_FILE, encoding="utf-8-sig")
    edges = pd.read_csv(EDGE_FILE, encoding="utf-8-sig")
    nodes["theme"] = nodes["Primary_Theme"].map(clean_theme)
    node_lookup = nodes.set_index("cluster_id")[["lng", "lat"]]

    line_geometries = []
    valid_rows = []
    for index, row in edges.iterrows():
        if row["start"] not in node_lookup.index or row["end"] not in node_lookup.index:
            continue
        start = node_lookup.loc[row["start"]]
        end = node_lookup.loc[row["end"]]
        line_geometries.append(LineString([(start.lng, start.lat), (end.lng, end.lat)]))
        valid_rows.append(index)
    edges = edges.loc[valid_rows].copy().reset_index(drop=True)
    edges["theme"] = edges["Theme"].map(clean_theme)

    node_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lng, nodes.lat), crs=WGS84).to_crs(MAP_CRS)
    edge_gdf = gpd.GeoDataFrame(edges, geometry=line_geometries, crs=WGS84).to_crs(MAP_CRS)
    districts = gpd.read_file(BOUNDARY_FILE).to_crs(MAP_CRS)
    return node_gdf, edge_gdf, districts


def draw_edges(ax: plt.Axes, edges: gpd.GeoDataFrame, main: bool) -> None:
    for theme in THEME_ORDER:
        subset = edges[edges["theme"] == theme]
        if subset.empty:
            continue
        segments = [np.asarray(geom.coords) for geom in subset.geometry]
        collection = LineCollection(
            segments,
            colors=THEME_COLORS[theme],
            linewidths=edge_widths(subset["line_weight"], main=main),
            alpha=0.43 if main else 0.13,
            zorder=5,
            capstyle="round",
        )
        ax.add_collection(collection)



def draw_direction_arrows(ax: plt.Axes, edges: gpd.GeoDataFrame) -> None:
    cutoff = edges["line_weight"].quantile(0.965)
    strong = edges[edges["line_weight"] >= cutoff]
    for _, row in strong.iterrows():
        coords = np.asarray(row.geometry.coords)
        start, end = coords[0], coords[-1]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=4.2,
            linewidth=0.35 + 0.36 * np.log1p(row.line_weight),
            color=THEME_COLORS.get(row.theme, "#777777"),
            alpha=0.68,
            shrinkA=2.5,
            shrinkB=2.5,
            zorder=6,
        )
        ax.add_patch(arrow)


def draw_nodes(ax: plt.Axes, nodes: gpd.GeoDataFrame, main: bool) -> None:
    for theme in THEME_ORDER:
        subset = nodes[nodes["theme"] == theme]
        if subset.empty:
            continue
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=node_sizes(subset["point_weight"], main=main),
            c=THEME_COLORS[theme],
            alpha=0.76 if main else 0.67,
            edgecolors="#F7F9FA",
            linewidths=0.35 if main else 0.12,
            zorder=9,
            rasterized=False,
        )


def draw_density_glow(
    ax: plt.Axes,
    nodes: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    main: bool,
) -> None:
    xmin, xmax, ymin, ymax = extent
    visible = nodes[
        nodes.geometry.x.between(xmin, xmax) & nodes.geometry.y.between(ymin, ymax)
    ]
    histogram, _, _ = np.histogram2d(
        visible.geometry.x,
        visible.geometry.y,
        bins=(360 if main else 220, 360 if main else 220),
        range=((xmin, xmax), (ymin, ymax)),
    )
    density = gaussian_filter(histogram.T, sigma=8.5 if main else 5.5)
    upper = np.percentile(density[density > 0], 99.2)
    normalized = np.clip(density / upper, 0, 1) ** 0.72
    glow = LinearSegmentedColormap.from_list(
        "node_density_glow",
        [
            (0.00, (0.02, 0.05, 0.07, 0.00)),
            (0.20, (0.10, 0.55, 0.70, 0.03)),
            (0.55, (0.20, 0.78, 0.88, 0.16)),
            (0.82, (0.52, 0.92, 0.94, 0.32)),
            (1.00, (0.94, 1.00, 1.00, 0.58)),
        ],
    )
    ax.imshow(
        normalized,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=glow,
        interpolation="bilinear",
        zorder=4,
        aspect="auto",
    )


def number_pois(ax: plt.Axes, nodes: gpd.GeoDataFrame) -> None:
    for number, poi in enumerate(POI_LABELS, start=1):
        subset = nodes[nodes["location_POI"] == poi]
        if subset.empty:
            continue
        row = subset.sort_values("point_weight", ascending=False).iloc[0]
        ax.annotate(
            str(number),
            xy=(row.geometry.x, row.geometry.y),
            xytext=LABEL_OFFSETS[poi],
            textcoords="offset points",
            fontsize=5.7,
            fontweight="bold",
            color=MAP_TEXT,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.22", facecolor=LEGEND_BG, edgecolor=MAP_TEXT, linewidth=0.55, alpha=0.96),
            arrowprops=dict(arrowstyle="-", color=MAP_MUTED, lw=0.55, shrinkA=1.5, shrinkB=2.5),
            zorder=25,
        )


def style_info_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(LEGEND_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#536069")


def draw_place_list(ax: plt.Axes) -> None:
    style_info_axis(ax)
    ax.text(0.08, 0.91, "Selected locations", color=MAP_TEXT, fontsize=6.6, fontweight="bold", transform=ax.transAxes)
    for number, label in enumerate(POI_LABELS.values(), start=1):
        y = 0.80 - (number - 1) * 0.077
        ax.text(
            0.11,
            y,
            str(number),
            color=MAP_TEXT,
            fontsize=5.5,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.18", facecolor="#273239", edgecolor=MAP_TEXT, linewidth=0.45),
            transform=ax.transAxes,
        )
        ax.text(0.21, y, label, color=MAP_TEXT, fontsize=5.05, ha="left", va="center", transform=ax.transAxes)


def draw_encoding_panel(ax: plt.Axes) -> None:
    style_info_axis(ax)
    ax.text(0.07, 0.92, "Legends", color=MAP_TEXT, fontsize=6.6, fontweight="bold", transform=ax.transAxes)
    ax.text(0.07, 0.80, "Node fill · primary route theme", color=MAP_MUTED, fontsize=5.15, transform=ax.transAxes)
    positions = [(0.10, 0.68), (0.54, 0.68), (0.10, 0.56), (0.54, 0.56)]
    for theme, (x, y) in zip(THEME_ORDER, positions):
        ax.scatter([x], [y], s=24, c=THEME_COLORS[theme], edgecolors=MAP_TEXT, linewidths=0.35, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.07, y, THEME_LABELS[theme].replace("Commercial & tourism", "Commercial").replace("Historical & art", "Historical"), color=MAP_TEXT, fontsize=4.9, va="center", transform=ax.transAxes)

    ax.text(0.07, 0.40, "Node size · spots frequency", color=MAP_MUTED, fontsize=5.15, transform=ax.transAxes)
    for x, value in zip([0.20, 0.48, 0.78], [10, 100, 500]):
        ax.scatter([x], [0.29], s=node_sizes(pd.Series([value]), main=True)[0], c="#A8B3B9", edgecolors=MAP_TEXT, linewidths=0.35, transform=ax.transAxes)
        ax.text(x, 0.205, str(value), color=MAP_TEXT, fontsize=4.9, ha="center", transform=ax.transAxes)

    ax.text(0.07, 0.135, "Edge width · path frequency", color=MAP_MUTED, fontsize=5.15, transform=ax.transAxes)
    for x, value in zip([0.20, 0.49, 0.79], [1, 8, 32]):
        ax.plot([x - 0.07, x + 0.07], [0.075, 0.075], color=MAP_TEXT, lw=edge_widths(pd.Series([value]), main=True)[0], solid_capstyle="round", transform=ax.transAxes)
        ax.text(x, 0.018, str(value), color=MAP_TEXT, fontsize=4.9, ha="center", transform=ax.transAxes)


def make_figure() -> None:
    style()
    nodes, edges, districts = build_spatial_data()
    main_extent = projected_extent(MAIN_LONLAT)
    city_extent = projected_extent(CITY_LONLAT)

    fig = plt.figure(figsize=(7.2, 5.9))
    ax = fig.add_axes([0.065, 0.075, 0.69, 0.88])
    ax.set_xlim(main_extent[0], main_extent[1])
    ax.set_ylim(main_extent[2], main_extent[3])
    add_basemap(ax, zoom=12)
    districts.plot(ax=ax, facecolor="none", edgecolor=MAP_BOUNDARY, linewidth=0.62, alpha=0.78, zorder=3)
    draw_density_glow(ax, nodes, main_extent, main=True)
    draw_edges(ax, edges, main=True)
    draw_nodes(ax, nodes, main=True)
    draw_direction_arrows(ax, edges)
    number_pois(ax, nodes)
    ax.set_xlim(main_extent[0], main_extent[1])
    ax.set_ylim(main_extent[2], main_extent[3])
    geographic_ticks(ax)
    add_north_arrow(ax, x=0.865, y=0.91)
    add_scale_bar(ax, length_km=5, x=0.055, y=0.055)
    ax.text(0.012, 0.015, "Basemap: CARTO; boundaries: Xiamen municipal data", transform=ax.transAxes, fontsize=4.8, color=MAP_MUTED, zorder=30)

    inset = fig.add_axes([0.765, 0.635, 0.22, 0.32])
    inset.set_xlim(city_extent[0], city_extent[1])
    inset.set_ylim(city_extent[2], city_extent[3])
    add_basemap(inset, zoom=10)
    districts.plot(ax=inset, facecolor=(1, 1, 1, 0.03), edgecolor=MAP_BOUNDARY, linewidth=0.5, zorder=3)
    draw_density_glow(inset, nodes, city_extent, main=False)
    draw_edges(inset, edges, main=False)
    draw_nodes(inset, nodes, main=False)

    for _, row in districts.iterrows():
        point = row.geometry.representative_point()
        label = DISTRICT_LABELS.get(row["NAME"], row["NAME"])
        inset.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=4.3,
            color=MAP_TEXT,
            bbox=dict(facecolor=LEGEND_BG, edgecolor="none", alpha=0.55, pad=0.08),
            zorder=15,
        )

    rect = Rectangle(
        (main_extent[0], main_extent[2]),
        main_extent[1] - main_extent[0],
        main_extent[3] - main_extent[2],
        facecolor="none",
        edgecolor=MAP_TEXT,
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=20,
    )
    inset.add_patch(rect)
    inset.set_xlim(city_extent[0], city_extent[1])
    inset.set_ylim(city_extent[2], city_extent[3])
    inset.text(0.03, 0.965, "Xiamen municipality", transform=inset.transAxes, ha="left", va="top", fontsize=5.6, fontweight="bold", color=MAP_TEXT, zorder=25)
    add_scale_bar(inset, length_km=10, x=0.07, y=0.07)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color(MAP_BOUNDARY)

    labels_ax = fig.add_axes([0.765, 0.335, 0.22, 0.275])
    draw_place_list(labels_ax)

    encoding_ax = fig.add_axes([0.765, 0.075, 0.22, 0.23])
    draw_encoding_panel(encoding_ax)

    output_stem = OUT / "fig_spatial_citywalk_network_dark_layered_nature_style"
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    nodes.drop(columns="geometry").to_csv(OUT / "fig_spatial_network_nodes_source.csv", index=False, encoding="utf-8-sig")
    edges.drop(columns="geometry").to_csv(OUT / "fig_spatial_network_edges_source.csv", index=False, encoding="utf-8-sig")
    print(f"Saved figure outputs to {OUT}")
    print(f"Nodes: {len(nodes):,}; edges: {len(edges):,}")


if __name__ == "__main__":
    make_figure()
