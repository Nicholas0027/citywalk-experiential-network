from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "analysis"
STAGE2_DIR = DATA_DIR
STAGE3_DIR = ROOT / "results" / "network"
OUT_DIR = ROOT / "results" / "network"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260721
N_BOOT = 1000
N_STABILITY = 100
FINAL_WIDTH_IN = 7.2

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
THEME_MARKERS = {
    "Commercial & Tourism": "o",
    "Nature": "^",
    "Historical & Art": "s",
    "Local Life": "D",
}
PAD_COLORS = {
    "Pleasure": "#D55E32",
    "Arousal": "#4A83C6",
    "Dominance": "#3C9D6F",
}
ROUTE_PAD_COLORS = {
    "Pleasure": "#C65D50",
    "Arousal": "#D39C31",
    "Dominance": "#4E79A7",
}
CLUSTER_COLORS = ["#3F6C9E", "#D28C45", "#5E9C76", "#9A6BA0", "#777777", "#C45A5A"]

ARCHETYPE_LABELS = {
    1: "Mixed structural hubs",
    2: "Calm local-life nodes",
    3: "Stimulating commercial nodes",
    4: "Calm heritage nodes",
    5: "Calm nature nodes",
    6: "Weakly appraised peripheral nodes",
}

ORIGINAL_REPRESENTATIVE_POIS = [
    "幸福路",
    "黄厝沙滩",
    "华新路",
    "八市",
    "南普陀寺",
    "集美学村",
    "十里长堤",
    "溪岸路花鸟市场",
    "百家村",
    "陈嘉庚故居",
]

POI_TRANSLATIONS = {
    "环岛路": "Island Ring Boulevard",
    "环岛南路": "Island Ring Boulevard",
    "黄厝沙滩": "Island Ring Boulevard",
    "华新路": "Huaxin Road",
    "八市": "The Eighth Market",
    "第八市场": "The Eighth Market",
    "中山路": "Zhongshan Road",
    "中山路步行街": "Zhongshan Road",
    "南普陀寺": "Nanputuo Temple",
    "集美学村": "Jimei School Village",
    "十里长堤": "Shili Long Causeway",
    "溪岸路花鸟市场": "Xi'an Road Flower and Bird Market",
    "溪岸花鸟市场": "Xi'an Road Flower and Bird Market",
    "幸福路": "Xingfu Road area",
    "百家村": "Baijia Village",
    "陈嘉庚故居": "Former Residence of Tan Kah Kee",
    "张万森家门口的炮仗花": "Zhang Wansen House",
    "1号张万森家门口的炮仗花": "Zhang Wansen House",
    "鼓浪屿": "Gulangyu Island",
    "厦门大学": "Xiamen University",
    "沙坡尾": "Shapowei",
    "白鹭洲公园": "Bailuzhou Park",
    "海湾公园": "Haiwan Park",
}


def clean_theme(value: object) -> str:
    text = str(value).strip().replace("Commericial", "Commercial")
    text = text.replace("Local_Life", "Local Life")
    return text


def poi_label(value: object) -> str:
    text = str(value).strip()
    return POI_TRANSLATIONS.get(text, text)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def epsilon_squared(h_value: float, n: int, k: int) -> float:
    return max(0.0, float((h_value - k + 1) / (n - k)))


def dunn_posthoc(data: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    valid = data[[value_col, group_col]].dropna().copy()
    valid["rank"] = stats.rankdata(valid[value_col].to_numpy())
    n_total = len(valid)
    _, tie_counts = np.unique(valid[value_col].to_numpy(), return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts)
    variance = (n_total * (n_total + 1) / 12.0) - tie_term / (12.0 * (n_total - 1))
    summaries = valid.groupby(group_col)["rank"].agg(["mean", "size"])
    rows: list[dict[str, object]] = []
    for i, group_a in enumerate(THEME_ORDER):
        for group_b in THEME_ORDER[i + 1 :]:
            if group_a not in summaries.index or group_b not in summaries.index:
                continue
            a = summaries.loc[group_a]
            b = summaries.loc[group_b]
            se = math.sqrt(variance * (1 / a["size"] + 1 / b["size"]))
            z_value = (a["mean"] - b["mean"]) / se
            p_raw = 2 * stats.norm.sf(abs(z_value))
            rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "z": z_value,
                    "p_raw": p_raw,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = multipletests(result["p_raw"], method="holm")[1]
    return result


def prepare_route_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.read_csv(DATA_DIR / "final_poi_records.csv", encoding="utf-8-sig")
    records["Primary_Theme"] = records["Primary_Theme"].map(clean_theme)
    for col in ["P_score", "A_score", "D_score"]:
        records[col] = pd.to_numeric(records[col], errors="coerce")

    route = (
        records.groupby("note_index")
        .agg(
            n_locations=("location_POI", "nunique"),
            theme=("Primary_Theme", lambda x: x.mode().iloc[0]),
            Pleasure=("P_score", "mean"),
            Arousal=("A_score", "mean"),
            Dominance=("D_score", "mean"),
        )
        .reset_index()
    )
    route = route[route["theme"].isin(THEME_ORDER)].copy()

    kw_rows: list[dict[str, object]] = []
    dunn_frames: list[pd.DataFrame] = []
    for dimension in ["Pleasure", "Arousal", "Dominance"]:
        groups = [
            route.loc[route["theme"] == theme, dimension].dropna().to_numpy()
            for theme in THEME_ORDER
        ]
        h_value, p_value = stats.kruskal(*groups)
        n_total = sum(len(group) for group in groups)
        kw_rows.append(
            {
                "dimension": dimension,
                "n_routes": n_total,
                "H": h_value,
                "df": len(groups) - 1,
                "p_raw": p_value,
                "epsilon_squared": epsilon_squared(h_value, n_total, len(groups)),
            }
        )
        dunn = dunn_posthoc(route, dimension, "theme")
        dunn.insert(0, "dimension", dimension)
        dunn_frames.append(dunn)

    kw = pd.DataFrame(kw_rows)
    kw["p_holm_across_PAD"] = multipletests(kw["p_raw"], method="holm")[1]
    dunn_all = pd.concat(dunn_frames, ignore_index=True)
    route.to_csv(OUT_DIR / "figure1_route_source_data.csv", index=False, encoding="utf-8-sig")
    kw.to_csv(OUT_DIR / "theme_pad_kruskal_wallis.csv", index=False, encoding="utf-8-sig")
    dunn_all.to_csv(OUT_DIR / "theme_pad_dunn_holm.csv", index=False, encoding="utf-8-sig")
    return route, kw, dunn_all


def assign_cluster_ids(records: pd.DataFrame, cluster_m: int = 100) -> pd.DataFrame:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    x, y = transformer.transform(records["lng_num"].to_numpy(), records["lat_num"].to_numpy())
    result = records.copy()
    result["cluster_x"] = (np.asarray(x) / cluster_m).round().astype(int) * cluster_m
    result["cluster_y"] = (np.asarray(y) / cluster_m).round().astype(int) * cluster_m
    keys = pd.Series(list(zip(result["cluster_x"], result["cluster_y"])), index=result.index)
    result["cluster_id"] = pd.factorize(keys, sort=True)[0]
    return result


def prepare_node_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(STAGE2_DIR / "reference_nodes_100m_1500m.csv")
    metrics = pd.read_csv(STAGE3_DIR / "stage3_node_coupling_metrics.csv")
    records = pd.read_csv(STAGE2_DIR / "clean_poi_records.csv")
    records["Primary_Theme"] = records["Primary_Theme"].map(clean_theme)
    clustered = assign_cluster_ids(records)

    if clustered["cluster_id"].nunique() != len(nodes):
        raise RuntimeError("Cluster reconstruction does not match the 100 m baseline nodes.")

    theme_counts = (
        clustered.groupby(["cluster_id", "Primary_Theme"]).size().unstack(fill_value=0)
    )
    for theme in THEME_ORDER:
        if theme not in theme_counts.columns:
            theme_counts[theme] = 0
    theme_props = theme_counts[THEME_ORDER].div(theme_counts[THEME_ORDER].sum(axis=1), axis=0)
    theme_props.columns = [f"theme_{theme}" for theme in THEME_ORDER]

    node_data = metrics.merge(nodes[["cluster_id", "lng", "lat"]], on="cluster_id", how="left")
    node_data["Primary_Theme"] = node_data["Primary_Theme"].map(clean_theme)
    node_data = node_data.merge(theme_props.reset_index(), on="cluster_id", how="left")
    node_data["poi_en"] = node_data["location_POI"].map(poi_label)
    node_data.to_csv(OUT_DIR / "figure2_3_node_source_data.csv", index=False, encoding="utf-8-sig")
    return node_data, clustered


def evaluate_clusters(node_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = node_data[node_data["active_node"]].copy().reset_index(drop=True)
    active["log_frequency"] = np.log1p(active["point_weight"])
    active["log_degree"] = np.log1p(active["degree"])
    active["log_strength"] = np.log1p(active["strength"])
    active["log_betweenness"] = np.log1p(active["betweenness"] * 100000)

    domain_features = {
        "mobility_topology": [
            "log_frequency",
            "log_degree",
            "log_strength",
            "log_betweenness",
            "closeness",
        ],
        "sentiment": ["mean_P", "mean_A", "mean_D"],
        "theme": [f"theme_{theme}" for theme in THEME_ORDER],
    }
    raw_features = [feature for group in domain_features.values() for feature in group]
    x_raw = active[raw_features].fillna(0.0).to_numpy()
    x_scaled = StandardScaler().fit_transform(x_raw)

    start = 0
    for features in domain_features.values():
        width = len(features)
        x_scaled[:, start : start + width] /= math.sqrt(width)
        start += width

    rng = np.random.default_rng(SEED)
    diagnostic_rows: list[dict[str, float]] = []
    fitted: dict[int, KMeans] = {}
    for k in range(2, 8):
        model = KMeans(n_clusters=k, n_init=100, random_state=SEED)
        labels = model.fit_predict(x_scaled)
        fitted[k] = model
        stability: list[float] = []
        for _ in range(N_STABILITY):
            sample_idx = rng.choice(len(active), size=int(0.8 * len(active)), replace=False)
            boot_model = KMeans(n_clusters=k, n_init=30, random_state=int(rng.integers(1_000_000)))
            boot_model.fit(x_scaled[sample_idx])
            stability.append(adjusted_rand_score(labels, boot_model.predict(x_scaled)))
        diagnostic_rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(x_scaled, labels),
                "davies_bouldin": davies_bouldin_score(x_scaled, labels),
                "mean_stability_ARI": float(np.mean(stability)),
                "sd_stability_ARI": float(np.std(stability, ddof=1)),
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    sil = (diagnostics["silhouette"] - diagnostics["silhouette"].min()) / (
        diagnostics["silhouette"].max() - diagnostics["silhouette"].min()
    )
    stab = (diagnostics["mean_stability_ARI"] - diagnostics["mean_stability_ARI"].min()) / (
        diagnostics["mean_stability_ARI"].max() - diagnostics["mean_stability_ARI"].min()
    )
    diagnostics["selection_score"] = 0.6 * sil + 0.4 * stab
    selected_k = int(diagnostics.loc[diagnostics["selection_score"].idxmax(), "k"])

    labels = fitted[selected_k].labels_
    active["cluster_raw"] = labels
    ordering_score = (
        active.groupby("cluster_raw")[["log_frequency", "log_degree", "log_strength", "log_betweenness"]]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=False)
    )
    label_map = {old: new + 1 for new, old in enumerate(ordering_score.index)}
    active["archetype"] = active["cluster_raw"].map(label_map).astype(int)
    active["archetype_name"] = active["archetype"].map(ARCHETYPE_LABELS)

    profile_features = {
        "Visibility": "log_frequency",
        "Connectivity": "log_degree",
        "Flow strength": "log_strength",
        "Bridging": "log_betweenness",
        "Accessibility": "closeness",
        "Pleasure": "mean_P",
        "Arousal": "mean_A",
        "Dominance": "mean_D",
        "Commercial": "theme_Commercial & Tourism",
        "Nature": "theme_Nature",
        "Heritage": "theme_Historical & Art",
        "Local life": "theme_Local Life",
    }
    profile_raw = active.groupby("archetype")[list(profile_features.values())].mean()
    profile_raw.columns = list(profile_features.keys())
    profile_z = pd.DataFrame(
        StandardScaler().fit_transform(profile_raw),
        index=profile_raw.index,
        columns=profile_raw.columns,
    )
    sizes = active["archetype"].value_counts().sort_index()
    profile_raw.insert(0, "n_nodes", sizes)
    profile_z.insert(0, "n_nodes", sizes)

    diagnostics.to_csv(OUT_DIR / "cluster_number_diagnostics.csv", index=False)
    active.to_csv(OUT_DIR / "node_archetype_assignments.csv", index=False, encoding="utf-8-sig")
    profile_raw.to_csv(OUT_DIR / "node_archetype_profiles_raw.csv")
    profile_z.to_csv(OUT_DIR / "node_archetype_profiles_z.csv")
    return active, diagnostics, profile_z


def spearman_bootstrap(active: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    variables = {
        "Featured frequency": "point_weight",
        "Betweenness": "betweenness",
        "Closeness": "closeness",
    }
    pads = {"Pleasure": "mean_P", "Arousal": "mean_A", "Dominance": "mean_D"}
    rng = np.random.default_rng(SEED + 19)
    n = len(active)
    for x_name, x_col in variables.items():
        for y_name, y_col in pads.items():
            x = active[x_col].to_numpy(dtype=float)
            y = active[y_col].to_numpy(dtype=float)
            observed = stats.spearmanr(x, y).statistic
            boot = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = rng.integers(0, n, n)
                boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
            pairs.append(
                {
                    "network_variable": x_name,
                    "PAD_dimension": y_name,
                    "n_nodes": n,
                    "rho": observed,
                    "ci_low": np.nanquantile(boot, 0.025),
                    "ci_high": np.nanquantile(boot, 0.975),
                    "p_raw": stats.spearmanr(x, y).pvalue,
                }
            )
    result = pd.DataFrame(pairs)
    result["p_holm"] = multipletests(result["p_raw"], method="holm")[1]
    result.to_csv(OUT_DIR / "active_node_pad_coupling_bootstrap.csv", index=False)
    return result


def plot_overall_patterns(route: pd.DataFrame, kw: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 3.45), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.75], wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    upper = int(min(25, route["n_locations"].quantile(0.995)))
    clipped = route["n_locations"].clip(upper=upper)
    bins = np.arange(0.5, upper + 1.5, 1)
    ax_a.hist(clipped, bins=bins, color="#6E91B8", edgecolor="white", linewidth=0.35)
    median = route["n_locations"].median()
    mean = route["n_locations"].mean()
    ax_a.axvline(median, color="#B64342", lw=1.1, ls="--", label=f"Median {median:.0f}")
    ax_a.axvline(mean, color="#3F3F3F", lw=1.0, ls=":", label=f"Mean {mean:.1f}")
    ax_a.set_xlabel("Unique POIs per route")
    ax_a.set_ylabel("Routes")
    ax_a.set_xlim(0.5, upper + 0.5)
    ax_a.legend(loc="upper right")
    ax_a.grid(axis="y", color="#E8E8E8", lw=0.5)
    panel_label(ax_a, "a")

    counts = route["theme"].value_counts().reindex(THEME_ORDER)
    y = np.arange(len(THEME_ORDER))
    ax_b.barh(y, counts, color=[THEME_COLORS[t] for t in THEME_ORDER], height=0.62)
    for yi, value in zip(y, counts):
        ax_b.text(value + counts.max() * 0.025, yi, f"{int(value):,}", va="center", fontsize=6.2)
    ax_b.set_yticks(y, ["Commercial", "Nature", "Historical & art", "Local life"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Routes")
    ax_b.set_xlim(0, counts.max() * 1.18)
    ax_b.grid(axis="x", color="#E8E8E8", lw=0.5)
    panel_label(ax_b, "b")

    offsets = {"Pleasure": -0.18, "Arousal": 0.0, "Dominance": 0.18}
    for dimension in ["Pleasure", "Arousal", "Dominance"]:
        for i, theme in enumerate(THEME_ORDER):
            values = route.loc[route["theme"] == theme, dimension].dropna().to_numpy()
            mean_value = float(np.mean(values))
            sd_value = float(np.std(values, ddof=1))
            yy = i + offsets[dimension]
            ax_c.errorbar(
                mean_value,
                yy,
                xerr=sd_value,
                fmt="o",
                ms=3.8,
                color=ROUTE_PAD_COLORS[dimension],
                ecolor=ROUTE_PAD_COLORS[dimension],
                elinewidth=0.9,
                capsize=2,
                label=dimension if i == 0 else None,
            )
    ax_c.axvline(0, color="#888888", lw=0.7, ls="--")
    ax_c.set_yticks(np.arange(len(THEME_ORDER)), ["Commercial", "Nature", "Historical & art", "Local life"])
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.80, 1.23)
    ax_c.set_xlabel("Route-level PAD score")
    ax_c.grid(axis="x", color="#E8E8E8", lw=0.5)
    ax_c.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.28), columnspacing=0.9, handletextpad=0.35)
    panel_label(ax_c, "c", x=-0.10)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.21)
    save_figure(fig, "fig_overall_route_patterns_nature_style")


def select_representative_nodes(node_data: pd.DataFrame) -> pd.DataFrame:
    order = {poi: index for index, poi in enumerate(ORIGINAL_REPRESENTATIVE_POIS)}
    selected = node_data[node_data["location_POI"].isin(ORIGINAL_REPRESENTATIVE_POIS)].copy()
    selected["representative_order"] = selected["location_POI"].map(order)
    selected = selected.sort_values("representative_order").drop_duplicates("location_POI")
    selected["selection_basis"] = np.where(
        selected["representative_order"] < 8,
        "Audited high frequency",
        "Theme coverage",
    )
    if len(selected) != len(ORIGINAL_REPRESENTATIVE_POIS):
        missing = set(ORIGINAL_REPRESENTATIVE_POIS) - set(selected["location_POI"])
        raise RuntimeError(f"Missing representative POIs: {sorted(missing)}")
    return selected.reset_index(drop=True)


def plot_featured_nodes(node_data: pd.DataFrame) -> None:
    selected = select_representative_nodes(node_data)
    selected.to_csv(OUT_DIR / "figure2_representative_nodes.csv", index=False, encoding="utf-8-sig")
    labels = selected["poi_en"].tolist()
    y = np.arange(len(selected))

    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 4.45))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.4, 2.0], hspace=0.34)
    top_gs = outer[0].subgridspec(1, 5, wspace=0.34)
    pie_gs = outer[1].subgridspec(2, 5, hspace=0.08, wspace=0.34)
    ax_a = fig.add_subplot(top_gs[0, :3])
    ax_c = fig.add_subplot(top_gs[0, 3:])
    ax_b_group = fig.add_subplot(outer[1], frameon=False)
    ax_b_group.patch.set_alpha(0)
    ax_b_group.set_xticks([])
    ax_b_group.set_yticks([])
    ax_b_group.set_zorder(-10)

    colors = [THEME_COLORS[t] for t in selected["Primary_Theme"]]
    bars = ax_a.barh(y, selected["point_weight"], color=colors, height=0.64)
    for bar, basis in zip(bars, selected["selection_basis"]):
        if basis == "Theme coverage":
            bar.set_hatch("////")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.45)
    for yi, value in zip(y, selected["point_weight"]):
        ax_a.text(value + selected["point_weight"].max() * 0.018, yi, f"{int(value)}", va="center", fontsize=6.1)
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Featured records in 100 m node", labelpad=1)
    ax_a.set_xlim(0, selected["point_weight"].max() * 1.16)
    ax_a.grid(axis="x", color="#E8E8E8", lw=0.5)
    ax_a.tick_params(axis="y", length=0)

    short_labels = ["Commercial", "Nature", "Historical & art", "Local life"]

    pad = selected[["mean_P", "mean_A", "mean_D"]].to_numpy()
    cmap = mcolors.LinearSegmentedColormap.from_list("pad", ["#4A83C6", "#F7F7F7", "#D55E32"])
    image = ax_c.imshow(pad, aspect="auto", cmap=cmap, norm=mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax_c.set_xticks([0, 1, 2], ["P", "A", "D"])
    ax_c.set_yticks(y, [f"({idx + 1})" for idx in range(len(selected))])
    ax_c.tick_params(axis="y", length=0, pad=2)
    for i in range(len(selected)):
        for j in range(3):
            value = pad[i, j]
            ax_c.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(value) > 0.58 else "#222222")
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_c, orientation="vertical", fraction=0.055, pad=0.04, ticks=[-1, 0, 1])
    cbar.set_label("PAD score", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)

    for idx, row in selected.reset_index(drop=True).iterrows():
        pie_ax = fig.add_subplot(pie_gs[idx // 5, idx % 5])
        values = np.array([row[f"theme_{theme}"] for theme in THEME_ORDER], dtype=float)
        values = np.nan_to_num(values, nan=0.0)

        def autopct(pct: float) -> str:
            return f"{pct:.0f}%" if pct >= 7 else ""

        pie_ax.pie(
            values,
            colors=[THEME_COLORS[t] for t in THEME_ORDER],
            startangle=90,
            counterclock=False,
            autopct=autopct,
            pctdistance=0.73,
            textprops={"fontsize": 5.0, "color": "#222222"},
            wedgeprops={"width": 0.72, "edgecolor": "white", "linewidth": 0.5},
        )
        title = "\n".join(textwrap.wrap(f"({idx + 1}) {row['poi_en']}", width=22))
        pie_ax.set_title(title, fontsize=5.4, pad=1.0)
        pie_ax.set_aspect("equal")
        pie_ax.set_xticks([])
        pie_ax.set_yticks([])

    panel_label(ax_a, "a", x=-0.4607, y=1.02)
    panel_label(ax_c, "c", x=-0.20, y=1.02)
    panel_label(ax_b_group, "b", x=-0.2666, y=1.02)

    handles = [mpl.patches.Patch(color=THEME_COLORS[t], label=s) for t, s in zip(THEME_ORDER, short_labels)]
    hatch_handle = mpl.patches.Patch(facecolor="#B7B7B7", edgecolor="white", hatch="////", label="Added for theme coverage")
    fig.legend(
        handles=[*handles, hatch_handle],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.55, 0.005),
        columnspacing=0.8,
        handlelength=1.1,
    )
    fig.subplots_adjust(left=0.285, right=0.985, top=0.94, bottom=0.11)
    save_figure(fig, "fig_featured_node_profiles_nature_style")


def plot_integrated_coupling(node_data: pd.DataFrame) -> None:
    data = node_data.copy()
    data["log_frequency_plot"] = np.log10(data["point_weight"])
    data["log_betweenness_plot"] = np.log10(1 + data["betweenness"] * 100000)

    tests = []
    for x_name, x_col, y_name, y_col in [
        ("Featured frequency", "point_weight", "Betweenness", "betweenness"),
        ("Featured frequency", "point_weight", "Arousal", "mean_A"),
        ("Betweenness", "betweenness", "Arousal", "mean_A"),
    ]:
        result = stats.spearmanr(data[x_col], data[y_col])
        tests.append({"x": x_name, "y": y_name, "rho": result.statistic, "p_raw": result.pvalue})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = multipletests(tests["p_raw"], method="holm")[1]
    tests.to_csv(OUT_DIR / "figure3_pairwise_spearman_tests.csv", index=False)

    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for theme in THEME_ORDER:
        subset = data[data["Primary_Theme"] == theme]
        common = dict(
            color=THEME_COLORS[theme],
            marker=THEME_MARKERS[theme],
            alpha=0.48,
            edgecolors="white",
            linewidths=0.2,
        )
        ax_a.scatter(
            subset["log_frequency_plot"],
            subset["log_betweenness_plot"],
            subset["mean_A"],
            s=7,
            depthshade=False,
            **common,
        )
        ax_b.scatter(subset["point_weight"], subset["betweenness"], s=9, **common)
        ax_c.scatter(subset["point_weight"], subset["mean_A"], s=9, **common)
        ax_d.scatter(subset["betweenness"], subset["mean_A"], s=9, **common)

    x_grid = np.linspace(data["log_frequency_plot"].min(), data["log_frequency_plot"].max(), 2)
    y_grid = np.linspace(data["log_betweenness_plot"].min(), data["log_betweenness_plot"].max(), 2)
    xx, yy = np.meshgrid(x_grid, y_grid)
    ax_a.plot_surface(xx, yy, np.zeros_like(xx), color="#B8B8B8", alpha=0.12, shade=False)
    ax_a.set_xlabel("Featured records", labelpad=2)
    ax_a.set_ylabel("Betweenness", labelpad=3)
    ax_a.set_zlabel("Arousal", labelpad=2)
    ax_a.set_xticks(np.log10([1, 10, 100, 600]), ["1", "10", "100", "600"])
    bet_ticks = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2])
    ax_a.set_yticks(np.log10(1 + bet_ticks * 100000), ["0", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
    ax_a.set_zticks([-1, 0, 1])
    ax_a.view_init(elev=22, azim=-58)
    ax_a.grid(True)
    ax_a.text2D(-0.12, 1.03, "a", transform=ax_a.transAxes, fontsize=8, fontweight="bold")

    ax_b.set_xscale("log")
    ax_b.set_yscale("symlog", linthresh=1e-6)
    ax_b.set_xlabel("Featured records in 100 m node")
    ax_b.set_ylabel("Betweenness centrality")
    ax_b.set_ylim(bottom=0)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Featured records in 100 m node")
    ax_c.set_ylabel("Mean Arousal")
    ax_d.set_xscale("symlog", linthresh=1e-6)
    ax_d.set_xlabel("Betweenness centrality")
    ax_d.set_ylabel("Mean Arousal")
    ax_d.set_xlim(left=0)
    for ax in [ax_c, ax_d]:
        ax.axhline(0, color="#777777", lw=0.7, ls="--")
        ax.set_ylim(-1.08, 1.08)
    for ax in [ax_b, ax_c, ax_d]:
        ax.grid(color="#E8E8E8", lw=0.45)

    for ax, label, row in zip([ax_b, ax_c, ax_d], ["b", "c", "d"], tests.itertuples()):
        panel_label(ax, label)
        p_text = "$p_{Holm}<0.001$" if row.p_holm < 0.001 else f"$p_{{Holm}}={row.p_holm:.3f}$"
        ax.text(
            0.03,
            0.96,
            f"Spearman $\\rho$={row.rho:.3f}, {p_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
        )

    theme_handles = [
        Line2D([0], [0], marker=THEME_MARKERS[t], linestyle="", markerfacecolor=THEME_COLORS[t], markeredgecolor="white", markersize=5, label=label)
        for t, label in zip(THEME_ORDER, ["Commercial", "Nature", "Historical & art", "Local life"])
    ]
    fig.legend(handles=theme_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.52, 0.005), columnspacing=0.9, handletextpad=0.3)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.97, bottom=0.14)
    save_figure(fig, "fig_integrated_node_coupling_nature_style")


def plot_cluster_diagnostics(active: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FINAL_WIDTH_IN, 2.7))
    ax_a, ax_b = axes
    ax_a.plot(diagnostics["k"], diagnostics["silhouette"], "o-", color="#3F6C9E", label="Silhouette")
    ax_a.plot(diagnostics["k"], diagnostics["mean_stability_ARI"], "s-", color="#D28C45", label="Stability ARI")
    selected_k = int(diagnostics.loc[diagnostics["selection_score"].idxmax(), "k"])
    ax_a.axvline(selected_k, color="#777777", lw=0.8, ls="--")
    ax_a.text(selected_k + 0.08, ax_a.get_ylim()[1] * 0.96, f"Selected k={selected_k}", fontsize=6.2, va="top")
    ax_a.set_xlabel("Number of archetypes (k)")
    ax_a.set_ylabel("Diagnostic score")
    ax_a.legend()
    ax_a.grid(color="#E8E8E8", lw=0.45)
    panel_label(ax_a, "a")

    feature_cols = [
        "log_frequency",
        "log_degree",
        "log_strength",
        "log_betweenness",
        "closeness",
        "mean_P",
        "mean_A",
        "mean_D",
    ] + [f"theme_{theme}" for theme in THEME_ORDER]
    x = StandardScaler().fit_transform(active[feature_cols].fillna(0))
    coords = PCA(n_components=2, random_state=SEED).fit_transform(x)
    for archetype in sorted(active["archetype"].unique()):
        mask = active["archetype"].to_numpy() == archetype
        ax_b.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.55, color=CLUSTER_COLORS[archetype - 1], label=f"A{archetype}")
    ax_b.set_xlabel("PCA 1")
    ax_b.set_ylabel("PCA 2")
    ax_b.legend(ncol=2)
    ax_b.grid(color="#E8E8E8", lw=0.45)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.20, wspace=0.32)
    save_figure(fig, "fig_node_archetype_diagnostics_nature_style")


def write_summary(kw: pd.DataFrame, coupling: pd.DataFrame) -> None:
    lines = [
        "# Stage 14 exploratory results",
        "",
        "- Theme-PAD Kruskal-Wallis results:",
    ]
    for row in kw.itertuples():
        lines.append(
            f"  - {row.dimension}: H={row.H:.2f}, adjusted p={row.p_holm_across_PAD:.3g}, epsilon-squared={row.epsilon_squared:.4f}."
        )
    lines.extend(["", "- Active-node coupling estimates for all PAD dimensions:"])
    for row in coupling.itertuples():
        lines.append(
            f"  - {row.network_variable} vs {row.PAD_dimension}: rho={row.rho:.3f}, 95% CI [{row.ci_low:.3f}, {row.ci_high:.3f}], Holm p={row.p_holm:.3g}."
        )
    (OUT_DIR / "stage14_exploratory_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    route, kw, _ = prepare_route_data()
    node_data, _ = prepare_node_data()
    active = node_data[node_data["active_node"]].copy().reset_index(drop=True)
    coupling = spearman_bootstrap(active)
    plot_overall_patterns(route, kw)
    plot_featured_nodes(node_data)
    plot_integrated_coupling(node_data)
    write_summary(kw, coupling)
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
