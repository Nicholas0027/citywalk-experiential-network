#!/usr/bin/env python3
"""Build the standalone Xiamen Citywalk Explorer page."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd


FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent

NODES_CSV = ROOT / "data" / "analysis" / "spatial_figure_nodes.csv"
EDGES_CSV = ROOT / "data" / "analysis" / "spatial_figure_edges.csv"
COMMUNITIES_CSV = ROOT / "results" / "communities" / "network_community_node_assignments.csv"
RECORDS_CSV = ROOT / "results" / "subnetworks" / "theme_subnetwork_records.csv"
TRANSLATIONS_CSV = FRONTEND / "place_name_translations.csv"
DISTRICTS_SHP = ROOT / "data" / "spatial" / "xiamen_district" / "xiamen_district.shp"
OUTPUT_HTML = FRONTEND / "citywalk_visual_analytics.html"

THEME_CODE = {
    "Commericial & Tourism": "C",
    "Commercial & Tourism": "C",
    "Nature": "N",
    "Historical & Art": "H",
    "Local_Life": "L",
    "Local Life": "L",
}

THEME_LABEL = {
    "C": "Commercial & Tourism",
    "N": "Nature",
    "H": "Historical & Art",
    "L": "Local Life",
}

THEME_COLOR = {
    "C": "#2CA7C9",
    "N": "#7CB342",
    "H": "#D66BA8",
    "L": "#E76F35",
}

REPRESENTATIVE_LABELS = {
    "幸福路": "Xingfu Road",
    "黄厝沙滩": "Huangcuo Beach",
    "华新路": "Huaxin Road",
    "八市": "The Eighth Market",
    "南普陀寺": "Nanputuo Temple",
    "集美学村": "Jimei School Village",
    "十里长堤": "Shili Long Causeway",
    "溪岸路花鸟市场": "Xi'an Road Flower and Bird Market",
    "百家村": "Baijia Village",
    "陈嘉庚故居": "Former Residence of Tan Kah Kee",
}


def load_data() -> dict[str, object]:
    nodes = pd.read_csv(NODES_CSV, encoding="utf-8-sig")
    edges = pd.read_csv(EDGES_CSV, encoding="utf-8-sig")
    assignments = pd.read_csv(COMMUNITIES_CSV, encoding="utf-8-sig")
    translations = pd.read_csv(TRANSLATIONS_CSV, encoding="utf-8-sig")
    name_translations = dict(zip(translations["name_zh"].astype(str), translations["name_en"].astype(str)))
    source_records = pd.read_csv(
        RECORDS_CSV,
        encoding="utf-8-sig",
        usecols=[
            "cluster_id",
            "location_POI",
            "text_evidence",
            "note_index",
            "original_date",
            "is_inherited",
            "P_score",
            "A_score",
            "D_score",
            "Primary_Theme",
            "lng",
            "lat",
        ],
    )
    required_names = set(nodes["location_POI"].dropna().astype(str).str.strip())
    required_names.update(source_records["location_POI"].dropna().astype(str).str.strip())
    missing_translations = sorted(name for name in required_names if name and name not in name_translations)
    if missing_translations:
        raise ValueError(
            f"Missing {len(missing_translations)} place-name translations; run "
            "visual_analytics_frontend/build_place_name_translations.py"
        )
    overall = assignments[assignments["network"] == "Overall"][
        [
            "cluster_id",
            "records_Commercial & Tourism",
            "records_Nature",
            "records_Historical & Art",
            "records_Local Life",
            "dominant_theme",
            "direct_records",
            "community_id",
            "community_status",
            "community_weighted_degree",
            "community_pagerank",
        ]
    ].copy()
    overall["community_num"] = overall["community_id"].map(
        lambda value: int(str(value)[1:]) if pd.notna(value) and str(value).startswith("C") else None
    )
    overall = overall.set_index("cluster_id")

    def clean_text(value: object) -> str:
        return "" if pd.isna(value) else str(value).strip()

    def clean_number(value: object) -> float | None:
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def english_name(value: object) -> str:
        source = clean_text(value)
        return name_translations.get(source, source or "Unnamed Place")

    members_by_cluster: dict[int, list[list[object]]] = {}
    records_by_cluster: dict[int, int] = {}
    excerpt_candidates: dict[int, list[dict[str, object]]] = {}
    notes_by_cluster: dict[int, set[str]] = {}
    excerpts_by_note: dict[str, list[dict[str, object]]] = {}
    member_accumulators: dict[int, dict[str, dict[str, object]]] = {}

    for row in source_records.itertuples(index=False):
        cluster_value = clean_number(row.cluster_id)
        if cluster_value is None:
            continue
        cluster_id = int(cluster_value)
        records_by_cluster[cluster_id] = records_by_cluster.get(cluster_id, 0) + 1
        name = clean_text(row.location_POI) or "Unnamed source place"
        note_index = clean_text(row.note_index)
        if note_index:
            notes_by_cluster.setdefault(cluster_id, set()).add(note_index)
        theme = THEME_CODE.get(clean_text(row.Primary_Theme), "C")
        inherited = clean_text(row.is_inherited).lower() != "no"
        longitude = clean_number(row.lng)
        latitude = clean_number(row.lat)
        p_score = clean_number(row.P_score)
        a_score = clean_number(row.A_score)
        d_score = clean_number(row.D_score)

        cluster_members = member_accumulators.setdefault(cluster_id, {})
        member = cluster_members.setdefault(
            name,
            {
                "count": 0,
                "x_sum": 0.0,
                "y_sum": 0.0,
                "xy_count": 0,
                "themes": {code: 0 for code in THEME_LABEL},
                "direct_pad": [0.0, 0.0, 0.0, 0],
                "all_pad": [0.0, 0.0, 0.0, 0],
            },
        )
        member["count"] = int(member["count"]) + 1
        member["themes"][theme] += 1
        if longitude is not None and latitude is not None:
            member["x_sum"] = float(member["x_sum"]) + longitude
            member["y_sum"] = float(member["y_sum"]) + latitude
            member["xy_count"] = int(member["xy_count"]) + 1
        if p_score is not None and a_score is not None and d_score is not None:
            member["all_pad"][0] += p_score
            member["all_pad"][1] += a_score
            member["all_pad"][2] += d_score
            member["all_pad"][3] += 1
            if not inherited:
                member["direct_pad"][0] += p_score
                member["direct_pad"][1] += a_score
                member["direct_pad"][2] += d_score
                member["direct_pad"][3] += 1

        evidence = clean_text(row.text_evidence)
        if evidence:
            candidate = {
                "q": evidence,
                "n": note_index,
                "dt": clean_text(row.original_date),
                "pl": name,
                "t": theme,
                "p": p_score if p_score is not None else 0.0,
                "a": a_score if a_score is not None else 0.0,
                "d": d_score if d_score is not None else 0.0,
                "h": inherited,
            }
            excerpt_candidates.setdefault(cluster_id, []).append(candidate)
            if note_index:
                excerpts_by_note.setdefault(note_index, []).append(candidate)

    for cluster_id, member_group in member_accumulators.items():
        compact_members: list[list[object]] = []
        for name, member in member_group.items():
            coordinate_count = int(member["xy_count"])
            direct_pad = member["direct_pad"]
            all_pad = member["all_pad"]
            pad = direct_pad if int(direct_pad[3]) else all_pad
            pad_count = int(pad[3])
            theme = max(member["themes"], key=member["themes"].get)
            compact_members.append(
                [
                    english_name(name),
                    int(member["count"]),
                    round(float(member["x_sum"]) / coordinate_count, 6) if coordinate_count else None,
                    round(float(member["y_sum"]) / coordinate_count, 6) if coordinate_count else None,
                    theme,
                    round(float(pad[0]) / pad_count, 2) if pad_count else 0.0,
                    round(float(pad[1]) / pad_count, 2) if pad_count else 0.0,
                    round(float(pad[2]) / pad_count, 2) if pad_count else 0.0,
                    name,
                ]
            )
        members_by_cluster[cluster_id] = sorted(compact_members, key=lambda member: (-int(member[1]), str(member[0])))

    def select_excerpts(cluster_id: int, canonical_name: str, limit: int = 4) -> list[list[object]]:
        deduplicated: dict[str, dict[str, object]] = {}
        candidates = list(excerpt_candidates.get(cluster_id, []))
        if not candidates:
            for note_index in notes_by_cluster.get(cluster_id, set()):
                candidates.extend({**candidate, "h": True} for candidate in excerpts_by_note.get(note_index, []))
        for candidate in candidates:
            text = str(candidate["q"])
            previous = deduplicated.get(text)
            if previous is None or (bool(previous["h"]) and not bool(candidate["h"])):
                deduplicated[text] = candidate
        pool = list(deduplicated.values())
        chosen: list[dict[str, object]] = []
        used_notes: set[str] = set()
        used_themes: set[str] = set()
        used_places: set[str] = set()
        while pool and len(chosen) < limit:
            def candidate_score(candidate: dict[str, object]) -> float:
                text_length = len(str(candidate["q"]))
                score = 180.0 if not bool(candidate["h"]) else 0.0
                score += min(text_length, 220) * 0.35
                score += 28.0 if 28 <= text_length <= 220 else 0.0
                score += 24.0 if str(candidate["n"]) not in used_notes else 0.0
                score += 18.0 if str(candidate["t"]) not in used_themes else 0.0
                score += 12.0 if str(candidate["pl"]) not in used_places else 0.0
                score += 14.0 if str(candidate["pl"]) == canonical_name else 0.0
                return score

            selected = max(pool, key=candidate_score)
            pool.remove(selected)
            chosen.append(selected)
            used_notes.add(str(selected["n"]))
            used_themes.add(str(selected["t"]))
            used_places.add(str(selected["pl"]))
        return [
            [
                candidate["q"],
                candidate["n"],
                candidate["dt"],
                english_name(candidate["pl"]),
                candidate["t"],
                round(float(candidate["p"]), 2),
                round(float(candidate["a"]), 2),
                round(float(candidate["d"]), 2),
                1 if candidate["h"] else 0,
            ]
            for candidate in chosen
        ]

    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(int(row.start), int(row.end), weight=int(row.line_weight))
    undirected = graph.to_undirected()
    pagerank = nx.pagerank(graph, weight="weight")
    betweenness = nx.betweenness_centrality(undirected, weight=None)
    closeness = nx.closeness_centrality(undirected)
    in_weight = dict(graph.in_degree(weight="weight"))
    out_weight = dict(graph.out_degree(weight="weight"))
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    featured_rank = nodes["point_weight"].rank(method="min", ascending=False).astype(int)
    nodes_payload: list[dict[str, object]] = []
    for row, rank in zip(nodes.itertuples(index=False), featured_rank):
        node_id = int(row.cluster_id)
        node_name = str(row.location_POI)
        community = overall.loc[node_id] if node_id in overall.index else None
        community_num = None
        community_status = "Isolated"
        community_weighted_degree = 0
        community_pagerank = 0.0
        theme_mix = [0, 0, 0, 0]
        direct_records = 0
        if community is not None:
            if pd.notna(community["community_num"]):
                community_num = int(community["community_num"])
            community_status = str(community["community_status"])
            if pd.notna(community["community_weighted_degree"]):
                community_weighted_degree = int(community["community_weighted_degree"])
            if pd.notna(community["community_pagerank"]):
                community_pagerank = round(float(community["community_pagerank"]), 7)
            theme_mix = [
                int(round(float(community.get(column, 0)))) if pd.notna(community.get(column, 0)) else 0
                for column in [
                    "records_Commercial & Tourism",
                    "records_Nature",
                    "records_Historical & Art",
                    "records_Local Life",
                ]
            ]
            if pd.notna(community.get("direct_records")):
                direct_records = int(round(float(community["direct_records"])))
        members = members_by_cluster.get(node_id, [])
        for member in members:
            if member[2] is None or member[3] is None:
                member[2] = round(float(row.lng), 6)
                member[3] = round(float(row.lat), 6)
        nodes_payload.append(
            {
                "i": node_id,
                "n": english_name(node_name),
                "zh": node_name,
                "x": round(float(row.lng), 6),
                "y": round(float(row.lat), 6),
                "t": THEME_CODE.get(str(row.Primary_Theme), "C"),
                "w": int(row.point_weight),
                "r": int(rank),
                "mt": theme_mix,
                "mc": len(members),
                "rc": records_by_cluster.get(node_id, int(row.point_weight)),
                "m": members,
                "ex": select_excerpts(node_id, node_name),
                "p": round(float(row.mean_P), 3),
                "a": round(float(row.mean_A), 3),
                "d": round(float(row.mean_D), 3),
                "dr": direct_records,
                "pr": round(float(pagerank.get(node_id, 0)), 7),
                "bt": round(float(betweenness.get(node_id, 0)), 7),
                "cl": round(float(closeness.get(node_id, 0)), 7),
                "iw": int(in_weight.get(node_id, 0)),
                "ow": int(out_weight.get(node_id, 0)),
                "ic": int(in_degree.get(node_id, 0)),
                "oc": int(out_degree.get(node_id, 0)),
                "cm": community_num,
                "cs": community_status,
                "cwd": community_weighted_degree,
                "cpr": community_pagerank,
            }
        )

    edges_payload = [
        {
            "s": int(row.start),
            "t": int(row.end),
            "w": int(row.line_weight),
            "th": THEME_CODE.get(str(row.Theme), "C"),
            "d": round(float(row.mean_distance_m), 1),
        }
        for row in edges.itertuples(index=False)
    ]

    districts = gpd.read_file(DISTRICTS_SHP)
    districts["geometry"] = districts.geometry.simplify(0.001, preserve_topology=True)
    districts_geojson = json.loads(districts.to_json(drop_id=True))
    district_names = {
        "SiMing": "Siming",
        "HuLi": "Huli",
        "JiMei": "Jimei",
        "HaiCang": "Haicang",
        "TongAn": "Tong'an",
        "XiangAn": "Xiang'an",
    }
    for feature in districts_geojson["features"]:
        name = str(feature["properties"].get("NAME", ""))
        feature["properties"] = {"name": district_names.get(name, name)}

    representative: list[dict[str, object]] = []
    for chinese_name, fallback_english_name in REPRESENTATIVE_LABELS.items():
        match = nodes[nodes["location_POI"] == chinese_name]
        if match.empty:
            continue
        row = match.iloc[0]
        representative.append(
            {
                "i": int(row["cluster_id"]),
                "n": name_translations.get(chinese_name, fallback_english_name),
                "w": int(row["point_weight"]),
                "t": THEME_CODE.get(str(row["Primary_Theme"]), "C"),
                "a": round(float(row["mean_A"]), 2),
            }
        )

    sentiment_summary = {
        dimension: {
            "negative": int((nodes[column] < 0).sum()),
            "neutral": int((nodes[column] == 0).sum()),
            "positive": int((nodes[column] > 0).sum()),
        }
        for dimension, column in {"P": "mean_P", "A": "mean_A", "D": "mean_D"}.items()
    }
    summary = {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "pathRecords": int(edges["line_weight"].sum()),
        "poiRecords": 26131,
        "routes": 4528,
        "communities": int(overall["community_num"].dropna().nunique()),
        "lccNodes": int((overall["community_status"] == "LCC community").sum()),
        "isolatedNodes": int((overall["community_status"] == "Isolated").sum()),
        "modularity": 0.708,
        "meanAri": 0.923,
        "themeCounts": {
            label: int((nodes["Primary_Theme"].map(THEME_CODE) == code).sum())
            for code, label in THEME_LABEL.items()
        },
        "sentiment": sentiment_summary,
    }
    return {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "districts": districts_geojson,
        "representative": representative,
        "summary": summary,
        "themeLabel": THEME_LABEL,
        "themeColor": THEME_COLOR,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Explore Xiamen's popular walking connections, neighbourhood character and citywide Citywalk routes.">
  <title>Xiamen Citywalk · Explore the City on Foot</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root {
      --bg:#090e10; --map:#0f1719; --panel:#121a1c; --panel-2:#0d1416;
      --line:#2d383a; --line-soft:#202b2d; --text:#f6f0e6; --muted:#c8beaf;
      --quiet:#9a9083; --accent:#d7b77f; --accent-soft:rgba(215,183,127,.14);
      --cyan:#2ca7c9; --green:#7cb342; --pink:#d66ba8; --orange:#e76f35;
      --font-serif:"Cormorant Garamond","Noto Serif SC","Songti SC",Georgia,serif;
      --shadow:0 20px 60px rgba(0,0,0,.42); --shadow-soft:0 10px 28px rgba(0,0,0,.24);
      --ease:cubic-bezier(.22,1,.36,1);
    }
    * { box-sizing:border-box; }
    html, body, #app { width:100%; height:100%; margin:0; }
    body { overflow:hidden; background:var(--bg); color:var(--text); font-family:var(--font-serif); font-size:16px; font-weight:500; letter-spacing:.008em; }
    button, input { font:inherit; }
    button { color:inherit; -webkit-tap-highlight-color:transparent; }
    button, input, [role="button"] { outline:none; }
    button:focus-visible, input:focus-visible, [role="button"]:focus-visible, .leaflet-control-zoom a:focus-visible {
      outline:2px solid #f0cb8c; outline-offset:2px; box-shadow:0 0 0 4px rgba(215,183,127,.16);
    }
    ::selection { background:rgba(215,183,127,.25); color:#fffaf1; }
    #app { display:grid; grid-template-columns:minmax(0,1fr) clamp(380px,31vw,430px); transition:grid-template-columns .34s var(--ease); }
    #app.sidebar-collapsed { grid-template-columns:minmax(0,1fr) 0; }
    #map-shell { position:relative; min-width:0; background:var(--map); }
    #map { position:absolute; inset:0; background:var(--map); }
    .leaflet-container { background:var(--map); font:inherit; }
    .leaflet-control-zoom a { display:grid; place-items:center; width:34px!important; height:34px!important; background:#121a1f; color:var(--text); border-color:var(--line); font-size:21px; transition:background .18s ease,color .18s ease; }
    .leaflet-control-zoom a:hover { background:#202b2d; color:#fffaf1; }
    .leaflet-bar { border-color:var(--line)!important; box-shadow:var(--shadow)!important; }
    .leaflet-control-attribution { background:rgba(11,16,19,.72)!important; color:#89979f!important; }
    .leaflet-control-attribution a { color:#b8c7cf!important; }
    #map-title { position:absolute; z-index:700; left:62px; right:136px; top:18px; max-width:580px; padding:15px 18px 16px; overflow:hidden; border:1px solid rgba(215,183,127,.26); border-radius:14px; background:linear-gradient(135deg,rgba(13,21,23,.93),rgba(13,20,22,.78)); box-shadow:var(--shadow); backdrop-filter:blur(16px) saturate(120%); pointer-events:none; }
    #map-title::before { content:""; position:absolute; inset:0 0 auto; height:2px; background:linear-gradient(90deg,var(--accent),rgba(215,183,127,0)); }
    #map-title .eyebrow { margin-bottom:5px; color:#e5c48b; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    #map-title h1 { margin:0; max-width:520px; font-size:27px; font-weight:600; line-height:1.08; letter-spacing:-.025em; }
    #map-title p { margin:7px 0 0; max-width:520px; color:var(--muted); font-size:13px; line-height:1.45; }
    #sidebar-toggle { position:absolute; z-index:780; top:18px; right:16px; display:inline-flex; align-items:center; gap:8px; min-height:38px; padding:8px 12px; border:1px solid rgba(215,183,127,.25); border-radius:999px; background:rgba(13,21,23,.88); color:var(--muted); box-shadow:var(--shadow-soft); backdrop-filter:blur(12px); cursor:pointer; transition:color .18s ease,border-color .18s ease,background .18s ease,transform .18s var(--ease); }
    #sidebar-toggle:hover { border-color:rgba(215,183,127,.52); background:rgba(24,34,36,.94); color:var(--text); transform:translateY(-1px); }
    #sidebar-toggle:active { transform:translateY(0); }
    #sidebar-toggle .toggle-icon { color:var(--accent); font-size:17px; line-height:1; transition:transform .28s var(--ease); }
    #sidebar-toggle .toggle-label { font-size:12px; font-weight:600; letter-spacing:.03em; }
    #app.sidebar-collapsed #sidebar-toggle .toggle-icon { transform:rotate(180deg); }
    #map-legend { position:absolute; z-index:700; left:16px; bottom:52px; min-width:220px; max-width:310px; padding:13px 14px; border:1px solid rgba(215,183,127,.19); border-radius:12px; background:rgba(12,19,21,.9); box-shadow:var(--shadow); backdrop-filter:blur(14px); }
    .legend-title { margin-bottom:8px; color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .legend-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px 12px; }
    .legend-item { display:flex; align-items:center; gap:7px; min-width:0; color:#e1dacf; font-size:11px; line-height:1.25; }
    .swatch { flex:0 0 auto; width:9px; height:9px; border:1px solid rgba(255,255,255,.48); border-radius:50%; }
    .gradient { height:8px; margin:3px 0 5px; border:1px solid rgba(255,255,255,.2); border-radius:5px; background:linear-gradient(90deg,#3b73b9 0%,#d8dcde 50%,#e8892f 100%); }
    .gradient-labels { display:flex; justify-content:space-between; color:var(--quiet); font-size:11px; }
    .legend-note { margin-top:7px; color:var(--quiet); font-size:11px; line-height:1.4; }
    .sentiment-legend { display:grid; grid-template-columns:44px 1fr; align-items:center; gap:9px; }
    .sentiment-legend svg { width:42px; height:42px; overflow:visible; }
    .sentiment-channel { display:grid; grid-template-columns:14px 1fr; gap:4px; margin:2px 0; color:#e1dacf; font-size:11px; }
    .sentiment-channel b { color:var(--text); }
    .bridge-legend { display:flex; align-items:center; gap:8px; color:#e1dacf; font-size:11px; line-height:1.35; }
    .bridge-symbol { position:relative; flex:0 0 auto; width:25px; height:25px; border:2px solid #72d7e8; border-radius:50%; }
    .bridge-symbol::after { content:""; position:absolute; inset:6px; border-radius:50%; background:#d6dde1; }
    .size-legend { display:flex; align-items:flex-end; gap:18px; height:27px; }
    .size-example { display:flex; align-items:center; gap:5px; color:var(--quiet); font-size:11px; }
    .size-circle { display:inline-block; border:1px solid #fff; border-radius:50%; background:#839198; opacity:.9; }
    #map-status { position:absolute; z-index:700; right:16px; bottom:16px; display:flex; gap:7px; align-items:center; padding:8px 11px; border:1px solid rgba(215,183,127,.16); border-radius:999px; background:rgba(12,19,21,.86); color:var(--muted); font-size:12px; box-shadow:var(--shadow-soft); backdrop-filter:blur(10px); }
    #map-status b { color:var(--text); font-variant-numeric:tabular-nums; }
    #tooltip { position:fixed; z-index:2000; visibility:hidden; opacity:0; min-width:220px; max-width:min(300px,calc(100vw - 24px)); padding:12px 13px; border:1px solid rgba(215,183,127,.26); border-radius:12px; background:rgba(12,19,21,.97); box-shadow:var(--shadow); pointer-events:none; transform:translateY(5px); transition:opacity .14s ease,transform .14s var(--ease),visibility 0s linear .14s; }
    #tooltip.visible { visibility:visible; opacity:1; transform:translateY(0); transition-delay:0s; }
    #tooltip h4 { margin:0 0 6px; font-size:15px; font-weight:600; }
    #tooltip .tip-row { display:flex; justify-content:space-between; gap:18px; margin-top:4px; color:var(--muted); font-size:12px; }
    #tooltip .tip-row b { color:var(--text); font-weight:600; font-variant-numeric:tabular-nums; }
    #sidebar { min-width:0; height:100%; display:flex; flex-direction:column; overflow:hidden; border-left:1px solid var(--line); background:linear-gradient(180deg,#141d1f,#11191b); box-shadow:-12px 0 38px rgba(0,0,0,.27); opacity:1; transform:translateX(0); transition:opacity .22s ease,transform .34s var(--ease); }
    #app.sidebar-collapsed #sidebar { opacity:0; transform:translateX(26px); pointer-events:none; border-left-color:transparent; }
    #sidebar-header { position:relative; padding:23px 22px 19px; overflow:hidden; border-bottom:1px solid var(--line); }
    #sidebar-header::after { content:""; position:absolute; right:-38px; top:-54px; width:150px; height:150px; border-radius:50%; background:radial-gradient(circle,rgba(215,183,127,.09),transparent 68%); pointer-events:none; }
    .product-kicker { display:flex; align-items:center; gap:8px; margin-bottom:8px; color:#e5c48b; font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .product-kicker::before { content:""; width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 0 4px var(--accent-soft); }
    #sidebar-header h2 { margin:0; font-size:28px; font-weight:600; line-height:1.08; letter-spacing:-.025em; }
    #sidebar-header p { margin:9px 0 0; max-width:360px; color:var(--muted); font-size:13px; line-height:1.48; }
    #tabs { display:grid; grid-template-columns:repeat(2,1fr); gap:5px; margin:10px 12px 0; padding:4px; border:1px solid var(--line-soft); border-radius:999px; background:#0c1416; }
    .tab-button { min-height:36px; padding:8px 10px; border:1px solid transparent; border-radius:999px; background:transparent; color:var(--muted); font-size:13px; font-weight:600; cursor:pointer; transition:color .18s ease,border-color .18s ease,background .18s ease,box-shadow .18s ease; }
    .tab-button:hover { color:var(--text); background:#172123; }
    .tab-button.active { color:#fff8ed; border-color:rgba(215,183,127,.3); background:linear-gradient(180deg,#263033,#1e292b); box-shadow:0 4px 14px rgba(0,0,0,.22); }
    #sidebar-body { flex:1; min-height:0; overflow:auto; overscroll-behavior:contain; scrollbar-gutter:stable; scrollbar-width:thin; scrollbar-color:#455254 transparent; }
    #sidebar-body::-webkit-scrollbar, #search-results::-webkit-scrollbar { width:8px; }
    #sidebar-body::-webkit-scrollbar-thumb, #search-results::-webkit-scrollbar-thumb { border:2px solid transparent; border-radius:999px; background:#455254; background-clip:padding-box; }
    .tab-panel { display:none; padding:18px 18px max(28px,env(safe-area-inset-bottom)); }
    .tab-panel.active { display:block; animation:panel-reveal .28s var(--ease); }
    @keyframes panel-reveal { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    .section { margin-bottom:20px; }
    .section-title { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:9px; color:#eee7dc; font-size:14px; font-weight:600; }
    .section-title small { color:var(--quiet); font-size:11px; font-weight:500; }
    .search-wrap { position:relative; }
    #node-search { width:100%; min-height:42px; padding:10px 12px; border:1px solid #3b484a; border-radius:10px; background:#0b1214; color:var(--text); font-size:13px; caret-color:var(--accent); transition:border-color .18s ease,box-shadow .18s ease,background .18s ease; }
    #node-search::placeholder, #source-place-filter::placeholder { color:#887f74; }
    #node-search:focus { border-color:#b69561; background:#0e1618; box-shadow:0 0 0 3px rgba(215,183,127,.12); }
    #search-results { position:absolute; z-index:20; left:0; right:0; top:calc(100% + 6px); display:none; max-height:248px; overflow:auto; border:1px solid #455254; border-radius:11px; background:#111a1c; box-shadow:var(--shadow); transform-origin:top center; }
    #search-results.open { animation:search-reveal .18s var(--ease); }
    @keyframes search-reveal { from { opacity:0; transform:translateY(-4px) scale(.985); } to { opacity:1; transform:translateY(0) scale(1); } }
    .search-result { display:flex; align-items:center; justify-content:space-between; gap:12px; width:100%; min-height:42px; padding:9px 11px; border:0; border-bottom:1px solid var(--line-soft); background:transparent; color:var(--text); text-align:left; cursor:pointer; transition:background .15s ease,color .15s ease; }
    .search-result:hover, .search-result:focus-visible { background:#202b2d; }
    .search-result:last-child { border-bottom:0; }
    .search-result span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:13px; }
    .search-result span:last-child { flex:0 0 auto; color:var(--quiet); font-size:11px; }
    .segment { display:flex; flex-wrap:wrap; gap:7px; }
    .chip { min-height:36px; padding:7px 10px; border:1px solid #354345; border-radius:999px; background:#0c1416; color:var(--muted); font-size:12px; font-weight:600; cursor:pointer; transition:color .18s ease,border-color .18s ease,background .18s ease,transform .18s var(--ease),box-shadow .18s ease; }
    .chip:hover { border-color:#647477; color:var(--text); transform:translateY(-1px); }
    .chip:active { transform:translateY(0); }
    .chip.active { border-color:var(--chip-color,#4e6570); background:color-mix(in srgb,var(--chip-color,#4e6570) 18%,#0d1418); color:var(--text); box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--chip-color,#4e6570) 22%,transparent); }
    .mode-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:5px; }
    .mode-button { min-height:40px; padding:8px 6px; border:1px solid #354345; border-radius:9px; background:#0c1416; color:var(--muted); font-size:12px; font-weight:600; cursor:pointer; transition:color .18s ease,border-color .18s ease,background .18s ease,transform .18s var(--ease); }
    .mode-button:hover { border-color:#5a696c; color:var(--text); transform:translateY(-1px); }
    .mode-button:active { transform:translateY(0); }
    .mode-button.active { border-color:#3da5c0; background:linear-gradient(180deg,#1c2e33,#17262b); color:#effbfe; box-shadow:inset 0 -2px 0 rgba(44,167,201,.36); }
    .slider-row { display:grid; grid-template-columns:1fr auto; align-items:center; gap:10px; margin-top:8px; }
    input[type="range"] { width:100%; height:24px; accent-color:var(--accent); cursor:pointer; }
    input[type="range"]::-webkit-slider-runnable-track { height:4px; border-radius:999px; background:#354244; }
    input[type="range"]::-webkit-slider-thumb { width:18px; height:18px; margin-top:-7px; border:2px solid #0e1517; border-radius:50%; background:var(--accent); box-shadow:0 2px 8px rgba(0,0,0,.45); -webkit-appearance:none; }
    .slider-value { min-width:84px; color:#ece4d9; font-size:12px; font-weight:600; font-variant-numeric:tabular-nums; text-align:right; }
    .toggle-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; margin-top:10px; }
    .toggle { min-height:39px; padding:8px 7px; border:1px solid #354345; border-radius:9px; background:#0c1416; color:var(--muted); font-size:12px; font-weight:600; cursor:pointer; transition:color .18s ease,border-color .18s ease,background .18s ease,transform .18s var(--ease); }
    .toggle:hover { border-color:#5a696c; color:var(--text); transform:translateY(-1px); }
    .toggle:active { transform:translateY(0); }
    .toggle.active { border-color:#5f7c82; background:#1b292c; color:var(--text); box-shadow:inset 0 -2px 0 rgba(114,215,232,.18); }
    .metric-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
    .metric { min-width:0; padding:11px 9px; border:1px solid var(--line-soft); border-radius:10px; background:linear-gradient(145deg,#10191b,#0c1315); }
    .metric strong { display:block; overflow:hidden; color:var(--text); font-size:19px; font-weight:600; line-height:1.05; font-variant-numeric:tabular-nums; text-overflow:ellipsis; white-space:nowrap; }
    .metric span { display:block; margin-top:4px; color:var(--quiet); font-size:11px; font-weight:600; line-height:1.2; text-transform:uppercase; letter-spacing:.06em; }
    #selected-card { padding:14px; border:1px solid #334143; border-radius:12px; background:linear-gradient(145deg,#111b1d,#0c1315); box-shadow:inset 0 1px 0 rgba(255,255,255,.025); }
    #selected-card.is-updating { animation:card-reveal .3s var(--ease); }
    @keyframes card-reveal { from { opacity:.4; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }
    .empty-state { padding:17px 10px; color:var(--quiet); font-size:12px; line-height:1.5; text-align:center; }
    .node-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .node-heading h3 { margin:0; font-size:18px; font-weight:600; line-height:1.18; }
    .node-subtitle { margin-top:4px; color:var(--muted); font-size:11px; }
    .rank-badge { flex:0 0 auto; padding:5px 8px; border:1px solid #4b5a5d; border-radius:999px; color:#e8e0d5; font-size:11px; font-weight:600; }
    .node-kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:5px; margin-top:10px; }
    .node-kpi { padding:8px 6px; border-radius:8px; background:#091113; text-align:center; }
    .node-kpi b { display:block; font-size:15px; font-weight:600; font-variant-numeric:tabular-nums; }
    .node-kpi span { color:var(--quiet); font-size:11px; }
    .pad-block { margin-top:11px; padding-top:10px; border-top:1px solid var(--line-soft); }
    .pad-row { display:grid; grid-template-columns:14px 1fr 34px; align-items:center; gap:7px; margin:7px 0; }
    .pad-row > span { color:#e5ddd2; font-size:11px; font-weight:700; }
    .pad-row > b { color:#e5ddd2; font-size:11px; font-weight:600; font-variant-numeric:tabular-nums; text-align:right; }
    .pad-track { position:relative; height:7px; border-radius:4px; background:#27333a; overflow:hidden; }
    .pad-track::after { content:""; position:absolute; top:0; bottom:0; left:50%; width:1px; background:rgba(255,255,255,.5); }
    .pad-fill { position:absolute; top:0; bottom:0; border-radius:4px; }
    .theme-block { margin-top:11px; padding-top:10px; border-top:1px solid var(--line-soft); }
    .theme-stack { display:flex; width:100%; height:10px; margin:7px 0 8px; overflow:hidden; border:1px solid rgba(255,255,255,.18); border-radius:5px; background:#263238; }
    .theme-stack span { display:block; min-width:0; }
    .theme-breakdown { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px 9px; }
    .theme-breakdown-item { display:grid; grid-template-columns:8px minmax(0,1fr) auto; align-items:center; gap:5px; min-width:0; color:var(--muted); font-size:11px; }
    .theme-breakdown-item .dot { width:7px; height:7px; border-radius:50%; }
    .theme-breakdown-item span:nth-child(2) { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .theme-breakdown-item b { color:#e8e0d5; font-weight:600; font-variant-numeric:tabular-nums; }
    .aggregate-note { margin:0 0 7px; color:var(--quiet); font-size:11px; line-height:1.45; }
    .excerpt-list { display:grid; gap:7px; }
    .excerpt-card { padding:10px 11px; border-left:2px solid var(--excerpt-color); border-radius:0 8px 8px 0; background:#091113; }
    .excerpt-card blockquote { margin:0; color:#ede6dc; font-size:12px; line-height:1.5; }
    .excerpt-meta { display:flex; flex-wrap:wrap; justify-content:space-between; gap:4px 8px; margin-top:7px; color:var(--quiet); font-size:11px; }
    .excerpt-meta b { color:#bdc8cd; font-weight:600; }
    .view-findings { width:100%; margin-top:8px; }
    .links-title { margin:13px 0 7px; color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .link-row { display:grid; grid-template-columns:18px minmax(0,1fr) auto; align-items:center; gap:6px; width:100%; min-height:38px; padding:7px 6px; border:0; border-top:1px solid var(--line-soft); border-radius:5px; background:transparent; color:var(--text); text-align:left; cursor:pointer; transition:background .15s ease,padding-left .18s var(--ease); }
    .link-row:hover { padding-left:9px; background:#1a2527; }
    .link-row .arrow { color:#76cada; text-align:center; }
    .link-row .destination { overflow:hidden; font-size:12px; text-overflow:ellipsis; white-space:nowrap; }
    .link-row .weight { color:var(--quiet); font-size:11px; font-variant-numeric:tabular-nums; }
    .representative-list { display:grid; gap:4px; }
    .representative { display:grid; grid-template-columns:24px minmax(0,1fr) auto; align-items:center; gap:8px; width:100%; min-height:44px; padding:8px; border:1px solid transparent; border-radius:9px; background:transparent; color:var(--text); text-align:left; cursor:pointer; transition:border-color .17s ease,background .17s ease,transform .17s var(--ease); }
    .representative:hover { border-color:#3d4b4e; background:#182225; transform:translateX(2px); }
    .representative .number { display:grid; place-items:center; width:24px; height:24px; border:1px solid #59696c; border-radius:50%; color:#eee6db; font-size:11px; font-variant-numeric:tabular-nums; }
    .representative .names { min-width:0; }
    .representative .names b, .representative .names span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .representative .names b { font-size:12px; font-weight:600; }
    .representative .names span { margin-top:2px; color:var(--quiet); font-size:11px; }
    .sentiment-badge { padding:4px 6px; border:1px solid #455356; border-radius:999px; color:#d8d0c5; font-size:11px; font-variant-numeric:tabular-nums; }
    .source-context { padding:13px; border:1px solid var(--line-soft); border-radius:11px; background:var(--panel-2); }
    .source-context h3 { margin:0; font-size:18px; font-weight:600; }
    .source-context p { margin:6px 0 0; color:var(--muted); font-size:12px; line-height:1.5; }
    .source-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:5px; margin-top:9px; }
    .source-summary div { padding:8px 5px; border-radius:8px; background:#091113; text-align:center; }
    .source-summary b { display:block; font-size:15px; font-weight:600; font-variant-numeric:tabular-nums; }
    .source-summary span { color:var(--quiet); font-size:11px; }
    #source-place-filter { width:100%; min-height:40px; padding:9px 11px; border:1px solid #3b484a; border-radius:9px; background:#0b1214; color:var(--text); font-size:12px; transition:border-color .18s ease,box-shadow .18s ease; }
    #source-place-filter:focus { border-color:#b69561; box-shadow:0 0 0 3px rgba(215,183,127,.12); }
    #source-place-filter:disabled { opacity:.5; }
    .source-list-note { margin:8px 1px 10px; color:var(--quiet); font-size:11px; line-height:1.45; }
    .source-place-list { display:grid; gap:3px; }
    .source-place-row { display:grid; grid-template-columns:24px 8px minmax(0,1fr) auto; align-items:center; gap:7px; padding:8px 5px; border-top:1px solid var(--line-soft); }
    .source-place-row:first-child { border-top:0; }
    .source-index { color:#9aa6a7; font-size:11px; font-variant-numeric:tabular-nums; text-align:right; }
    .source-place-row .dot { width:7px; height:7px; border-radius:50%; }
    .source-place-copy { min-width:0; }
    .source-place-copy b { display:block; overflow:hidden; color:#ede6dc; font-size:12px; font-weight:600; text-overflow:ellipsis; white-space:nowrap; }
    .source-place-copy span { display:block; margin-top:2px; color:var(--quiet); font-size:11px; }
    .source-place-count { color:#d7cec2; font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .district-label { border:0!important; background:transparent!important; box-shadow:none!important; color:rgba(238,232,222,.62)!important; font-size:11px!important; font-weight:600!important; text-shadow:0 1px 3px #000; }
    .district-label::before { display:none!important; }
    .map-node { cursor:pointer; }
    .map-node.is-selected { filter:drop-shadow(0 0 6px rgba(255,245,226,.82)); }
    .bridge-ring { fill:none; stroke:#72d7e8; pointer-events:none; }
    .member-halo { fill:rgba(44,167,201,.035); stroke:#64bfd3; stroke-dasharray:3 3; pointer-events:none; }
    .member-link { stroke:#8cd6e6; stroke-width:.35; pointer-events:none; }
    .member-point { cursor:help; stroke:#0b1013; }
    .member-caption { fill:#d2edf3; font-family:var(--font-serif); font-size:12px; font-weight:600; paint-order:stroke; stroke:#0b1013; stroke-width:2.5px; pointer-events:none; }
    .tip-composition { display:flex; height:6px; margin:7px 0 4px; overflow:hidden; border-radius:3px; background:#263238; }
    .tip-composition span { display:block; min-width:0; }
    .tip-note { color:var(--quiet); font-size:11px; line-height:1.35; }
    @media (pointer:coarse) {
      .node-hit { r:14px; }
      .chip, .mode-button, .toggle, .tab-button { min-height:44px; }
    }
    @media (max-width:900px) {
      html, body, #app { height:100dvh; }
      #app, #app.sidebar-collapsed { grid-template-columns:1fr; grid-template-rows:minmax(250px,56dvh) minmax(0,1fr); }
      #sidebar, #app.sidebar-collapsed #sidebar { opacity:1; transform:none; pointer-events:auto; border-left:0; border-top:1px solid var(--line); }
      #sidebar-header { display:none; }
      #sidebar-toggle { display:none; }
      #map-title { right:14px; max-width:none; }
      #map-title p { display:none; }
      #tabs { margin-top:8px; }
    }
    @media (max-width:640px) {
      #map-title { left:52px; right:10px; top:10px; padding:11px 13px 12px; border-radius:11px; }
      #map-title .eyebrow { font-size:9px; }
      #map-title h1 { font-size:21px; }
      #map-legend { left:10px; bottom:46px; min-width:0; max-width:min(270px,calc(100vw - 20px)); padding:10px 11px; }
      #map-status { right:10px; bottom:10px; max-width:calc(100vw - 20px); overflow:hidden; font-size:10px; white-space:nowrap; text-overflow:ellipsis; }
      .tab-panel { padding-inline:14px; }
      .theme-breakdown { grid-template-columns:1fr; }
    }
    @media (max-height:680px) and (min-width:901px) {
      #sidebar-header { padding-block:15px 12px; }
      #sidebar-header p { display:none; }
      #map-title p { display:none; }
    }
    @media (prefers-reduced-motion:reduce) {
      *, *::before, *::after { scroll-behavior:auto!important; animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; }
    }
  </style>
</head>
<body>
  <div id="app">
    <main id="map-shell" aria-label="Interactive map of Citywalk routes across Xiamen">
      <div id="map" role="region" aria-label="Explore places and walking connections on the Xiamen map"></div>
      <div id="map-title">
        <div class="eyebrow">Xiamen · Citywalk Guide</div>
        <h1>Discover Xiamen, one walk at a time</h1>
        <p>Follow popular connections through hidden streets, coastal paths and the character of every stop.</p>
      </div>
      <button id="sidebar-toggle" type="button" aria-controls="sidebar" aria-expanded="true"><span class="toggle-icon" aria-hidden="true">›</span><span class="toggle-label">Hide panel</span></button>
      <div id="map-legend"></div>
      <div id="map-status" role="status"></div>
    </main>
    <aside id="sidebar">
      <header id="sidebar-header">
        <div class="product-kicker">Xiamen Citywalk</div>
        <h2>Explore the city on foot</h2>
        <p>Search a place, follow popular connections and uncover what gives each neighbourhood its own rhythm.</p>
      </header>
      <nav id="tabs" role="tablist" aria-label="Citywalk explorer sections">
        <button class="tab-button active" id="tab-button-explore" data-tab="explore" role="tab" aria-selected="true" aria-controls="tab-explore">Explore</button>
        <button class="tab-button" id="tab-button-findings" data-tab="findings" role="tab" aria-selected="false" aria-controls="tab-findings" tabindex="-1">Place details</button>
      </nav>
      <div id="sidebar-body">
        <section class="tab-panel active" id="tab-explore" role="tabpanel" aria-labelledby="tab-button-explore">
          <div class="section">
            <label class="section-title" for="node-search">Find a place <small>English or Chinese</small></label>
            <div class="search-wrap">
              <input id="node-search" type="search" autocomplete="off" placeholder="Search 2,413 places…" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="search-results">
              <div id="search-results" role="listbox" aria-label="Matching places"></div>
            </div>
          </div>
          <div class="section">
            <div class="section-title">Map style</div>
            <div class="mode-grid">
              <button class="mode-button active" data-mode="theme" aria-pressed="true">Experience</button>
              <button class="mode-button" data-mode="sentiment" aria-pressed="false">Mood</button>
              <button class="mode-button" data-mode="community" aria-pressed="false">Clusters</button>
            </div>
          </div>
          <div class="section">
            <div class="section-title">Experience types <small id="theme-count-label"></small></div>
            <div class="segment" id="theme-chips"></div>
          </div>
          <div class="section">
            <label class="section-title" for="edge-threshold">Route visibility <small>show stronger connections</small></label>
            <div class="slider-row">
              <input id="edge-threshold" type="range" min="0" max="9" step="1" value="3" aria-describedby="edge-threshold-label">
              <output class="slider-value" id="edge-threshold-label"></output>
            </div>
            <label class="section-title" for="node-threshold" style="margin-top:12px">Place visibility <small>focus on popular stops</small></label>
            <div class="slider-row">
              <input id="node-threshold" type="range" min="0" max="8" step="1" value="0" aria-describedby="node-threshold-label">
              <output class="slider-value" id="node-threshold-label"></output>
            </div>
            <div class="toggle-grid">
              <button class="toggle" id="toggle-arrows" aria-pressed="false">Direction arrows</button>
              <button class="toggle active" id="toggle-labels" aria-pressed="true">Place labels</button>
              <button class="toggle" id="toggle-bridges" aria-pressed="false">Key connectors</button>
              <button class="toggle" id="reset-view">Reset view</button>
            </div>
          </div>
          <div class="section">
            <div class="section-title">On the map</div>
            <div class="metric-grid" id="visible-metrics"></div>
          </div>
          <div class="section">
            <div class="section-title">Place insights <small>choose a point or search result</small></div>
            <div id="selected-card"><div class="empty-state">Choose a place to see its character, mood, nearby names and strongest onward routes.</div></div>
          </div>
          <div class="section">
            <div class="section-title">Popular places <small>choose one to explore nearby</small></div>
            <div class="representative-list" id="representative-list"></div>
          </div>
        </section>
        <section class="tab-panel" id="tab-findings" role="tabpanel" aria-labelledby="tab-button-findings" hidden>
          <div class="section">
            <div class="section-title">Places in this 100 m cluster <small id="source-place-count-label">featured example</small></div>
            <div class="source-context" id="source-context"></div>
          </div>
          <div class="section">
            <label class="section-title" for="source-place-filter">Nearby place list <small>frequency and mood</small></label>
            <input id="source-place-filter" type="search" autocomplete="off" placeholder="Filter nearby place names…">
            <p class="source-list-note" id="source-list-note"></p>
            <div class="source-place-list" id="source-place-list"></div>
          </div>
        </section>
      </div>
    </aside>
  </div>
  <div id="tooltip" role="tooltip" aria-hidden="true"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    const DATA = __DATA__;
    const THEME_ORDER = ["C","N","H","L"];
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const COMMUNITY_COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC","#2F8F9D","#A0CBE8","#FFBE7D","#8CD17D"];
    const SENTIMENT_SCALE = d3.scaleLinear().domain([-1,0,1]).range(["#3B73B9","#D8DCDE","#E8892F"]).clamp(true);
    const EDGE_LEVELS = [1,2,3,4,5,8,12,16,24,32];
    const NODE_LEVELS = [1,2,3,5,10,20,50,100,250];
    const nodeMap = new Map(DATA.nodes.map(node => [node.i,node]));
    const bridgeRankedNodes = [...DATA.nodes].sort((left,right) => right.bt-left.bt);
    const bridgeNodeCount = Math.max(1,Math.ceil(DATA.nodes.length*.02));
    const bridgeNodeIds = new Set(bridgeRankedNodes.slice(0,bridgeNodeCount).map(node => node.i));
    const bridgeRankMap = new Map(bridgeRankedNodes.map((node,index) => [node.i,index+1]));
    const maxBetweenness = d3.max(DATA.nodes,node => node.bt) || 1;
    const incidentMap = new Map(DATA.nodes.map(node => [node.i,[]]));
    DATA.edges.forEach(edge => {
      incidentMap.get(edge.s)?.push({...edge,direction:"out"});
      incidentMap.get(edge.t)?.push({...edge,direction:"in"});
    });

    const state = {
      colorMode:"theme", activeThemes:new Set(THEME_ORDER), edgeLevelIndex:3,
      nodeLevelIndex:0, showArrows:false, showLabels:true, showBridges:false,
      selectedId:null, findingsNodeId:DATA.representative[0]?.i ?? DATA.nodes[0].i,
    };

    const map = L.map("map", {zoomControl:true,minZoom:9,maxZoom:17,attributionControl:true,preferCanvas:false});
    map.fitBounds([[24.415,118.00],[24.635,118.20]], {padding:[18,18]});
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png", {
      subdomains:"abcd", maxZoom:20, attribution:"© OpenStreetMap © CARTO"
    }).addTo(map);
    L.geoJSON(DATA.districts, {style:{color:"#BBC5CA",weight:.75,opacity:.55,fillColor:"#ffffff",fillOpacity:.018}}).addTo(map);

    const svg = d3.select(map.getPanes().overlayPane).append("svg").attr("class","leaflet-zoom-hide").style("pointer-events","all");
    const root = svg.append("g");
    const defs = root.append("defs");
    THEME_ORDER.forEach(theme => {
      defs.append("marker").attr("id",`arrow-${theme}`).attr("viewBox","0 -4 8 8").attr("refX",8).attr("refY",0)
        .attr("markerWidth",4).attr("markerHeight",4).attr("orient","auto")
        .append("path").attr("d","M0,-3.3L8,0L0,3.3Z").attr("fill",DATA.themeColor[theme]);
    });
    defs.append("marker").attr("id","arrow-selected").attr("viewBox","0 -4 8 8").attr("refX",8).attr("refY",0)
      .attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto")
      .append("path").attr("d","M0,-3.3L8,0L0,3.3Z").attr("fill","#ffffff");
    defs.append("marker").attr("id","arrow-bridge").attr("viewBox","0 -4 8 8").attr("refX",8).attr("refY",0)
      .attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto")
      .append("path").attr("d","M0,-3.3L8,0L0,3.3Z").attr("fill","#72D7E8");
    const edgeLayer = root.append("g").attr("class","edge-layer");
    const bridgeLayer = root.append("g").attr("class","bridge-layer");
    const memberLayer = root.append("g").attr("class","member-layer");
    const nodeLayer = root.append("g").attr("class","node-layer");
    const labelLayer = root.append("g").attr("class","label-layer");

    const tooltip = document.getElementById("tooltip");
    function positionTooltip(event) {
      const x = Math.max(12,Math.min(window.innerWidth - tooltip.offsetWidth - 12,event.clientX + 13));
      const y = Math.max(12,Math.min(window.innerHeight - tooltip.offsetHeight - 12,event.clientY + 13));
      tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`;
    }
    function showTooltip(event,html) {
      tooltip.innerHTML = html;
      positionTooltip(event);
      tooltip.classList.add("visible");
      tooltip.setAttribute("aria-hidden","false");
    }
    function moveTooltip(event) { if (tooltip.classList.contains("visible")) positionTooltip(event); }
    function hideTooltip() { tooltip.classList.remove("visible"); tooltip.setAttribute("aria-hidden","true"); }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g,character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[character]);
    }
    function themeMix(node) {
      const values = THEME_ORDER.map((theme,index) => ({theme,count:Number(node.mt?.[index] || 0)}));
      if (d3.sum(values,value => value.count) === 0) values.find(value => value.theme === node.t).count = Math.max(1,node.w);
      return values;
    }
    function themeComposition(node) {
      const values = themeMix(node);
      const total = d3.sum(values,value => value.count) || 1;
      return values.map(value => ({...value,total,share:value.count / total}));
    }
    function compositionBar(node,className="theme-stack") {
      return `<div class="${className}">${themeComposition(node).map(value => `<span style="width:${value.share*100}%;background:${DATA.themeColor[value.theme]}"></span>`).join("")}</div>`;
    }
    function nodeTooltip(node) {
      const compactMix = themeComposition(node).filter(value => value.count > 0).map(value => `${value.theme} ${Math.round(value.share*100)}%`).join(" · ");
      const bridgeRank = bridgeRankMap.get(node.i);
      return `<h4>${escapeHtml(node.n)}</h4><div class="tip-row"><span>Experience</span><b>${escapeHtml(DATA.themeLabel[node.t])}</b></div>${compositionBar(node,"tip-composition")}<div class="tip-note">${compactMix}</div><div class="tip-row"><span>Popularity</span><b>${node.w} · rank ${node.r}</b></div><div class="tip-row"><span>Connector position</span><b>${node.bt > 0 ? `rank ${bridgeRank} · ${node.bt.toFixed(4)}` : "not a connector"}</b></div><div class="tip-row"><span>Nearby place names</span><b>${node.mc}</b></div><div class="tip-row"><span>Mood P / A / D</span><b>${node.p.toFixed(2)} / ${node.a.toFixed(2)} / ${node.d.toFixed(2)}</b></div>`;
    }

    function nodeRadius(node) {
      const zoomScale = Math.max(.72,Math.min(1.16,.78 + (map.getZoom() - 11) * .055));
      return Math.max(.9,Math.min(7.8,(.7 + .27 * Math.sqrt(node.w)) * zoomScale));
    }
    function edgeWidth(edge) { return Math.max(.34,Math.min(2.15,.22 + .31 * Math.log1p(edge.w))); }
    function currentVisibleData() {
      const nodeThreshold = NODE_LEVELS[state.nodeLevelIndex];
      const edgeThreshold = EDGE_LEVELS[state.edgeLevelIndex];
      const nodes = DATA.nodes.filter(node => state.activeThemes.has(node.t) && node.w >= nodeThreshold);
      if (state.selectedId !== null && !nodes.some(node => node.i === state.selectedId)) nodes.push(nodeMap.get(state.selectedId));
      const nodeIds = new Set(nodes.map(node => node.i));
      const edges = DATA.edges.filter(edge => edge.w >= edgeThreshold && state.activeThemes.has(edge.th) && nodeIds.has(edge.s) && nodeIds.has(edge.t));
      return {nodes,nodeIds,edges,nodeThreshold,edgeThreshold};
    }
    function point(node) { return map.latLngToLayerPoint([node.y,node.x]); }
    function selectedNeighbors() {
      if (state.selectedId === null) return new Set();
      const neighbors = new Set([state.selectedId]);
      incidentMap.get(state.selectedId)?.forEach(edge => neighbors.add(edge.direction === "out" ? edge.t : edge.s));
      return neighbors;
    }

    function updateOverlayFrame() {
      const bounds = map.getBounds();
      const topLeft = map.latLngToLayerPoint(bounds.getNorthWest());
      const bottomRight = map.latLngToLayerPoint(bounds.getSouthEast());
      svg.attr("width",bottomRight.x-topLeft.x).attr("height",bottomRight.y-topLeft.y)
        .style("left",`${topLeft.x}px`).style("top",`${topLeft.y}px`);
      root.attr("transform",`translate(${-topLeft.x},${-topLeft.y})`);
    }

    function expandedMemberData(node) {
      if (!node?.m?.length) return [];
      const collisionRanks = new Map();
      return node.m.map((member,index) => {
        const base = map.latLngToLayerPoint([member[3],member[2]]);
        const collisionKey = `${Math.round(base.x/3)}:${Math.round(base.y/3)}`;
        const collisionRank = collisionRanks.get(collisionKey) || 0;
        collisionRanks.set(collisionKey,collisionRank + 1);
        const angle = collisionRank * 2.399963;
        const offset = collisionRank === 0 ? 0 : 2.2 * Math.sqrt(collisionRank);
        return {
          key:`${index}-${member[0]}`, name:member[0], count:member[1], theme:member[4],
          p:member[5], a:member[6], d:member[7], base,
          display:{x:base.x + Math.cos(angle)*offset,y:base.y + Math.sin(angle)*offset},
          offset,
        };
      });
    }

    function renderExpandedMembers() {
      const node = state.selectedId === null ? null : nodeMap.get(state.selectedId);
      const members = expandedMemberData(node);
      const center = node ? point(node) : null;
      const radius100m = node ? Math.abs(map.latLngToLayerPoint([
        node.y,
        node.x + 100 / (111320 * Math.cos(node.y * Math.PI / 180)),
      ]).x - center.x) : 0;
      memberLayer.selectAll("circle.member-halo").data(node ? [node] : []).join("circle")
        .attr("class","member-halo").attr("cx",center?.x || 0).attr("cy",center?.y || 0)
        .attr("r",Math.max(2,radius100m)).attr("stroke-width",.7);
      memberLayer.selectAll("line.member-link").data(members.filter(member => member.offset > .25),member => member.key).join("line")
        .attr("class","member-link").attr("x1",member => member.base.x).attr("y1",member => member.base.y)
        .attr("x2",member => member.display.x).attr("y2",member => member.display.y).attr("stroke-opacity",.42);
      memberLayer.selectAll("circle.member-point").data(members,member => member.key).join(
        enter => enter.append("circle").attr("class","member-point")
          .on("mouseenter",(event,member) => showTooltip(event,`<h4>${escapeHtml(member.name)}</h4><div class="tip-row"><span>Mentions</span><b>${member.count}</b></div><div class="tip-row"><span>Experience</span><b>${escapeHtml(DATA.themeLabel[member.theme])}</b></div><div class="tip-row"><span>Mood P / A / D</span><b>${member.p.toFixed(2)} / ${member.a.toFixed(2)} / ${member.d.toFixed(2)}</b></div><div class="tip-note">Original place retained inside the selected 100 m cluster.</div>`))
          .on("mousemove",moveTooltip)
          .on("mouseleave",hideTooltip),
        update => update,
        exit => exit.remove()
      ).attr("cx",member => member.display.x).attr("cy",member => member.display.y)
       .attr("r",member => Math.min(3.1,.8 + .38*Math.sqrt(member.count)))
       .attr("fill",member => DATA.themeColor[member.theme]).attr("fill-opacity",.9).attr("stroke-width",.45);
      const labels = [...members].sort((a,b) => b.count-a.count).slice(0,members.length <= 20 ? 12 : 7);
      memberLayer.selectAll("text.member-caption").data(labels,member => member.key).join("text")
        .attr("class","member-caption").attr("x",member => member.display.x+4).attr("y",member => member.display.y-3)
        .text(member => member.name.length > 14 ? `${member.name.slice(0,14)}…` : member.name);
    }

    function drawNodeGlyph(element,node,radius) {
      const effectiveRadius = state.colorMode === "sentiment" ? Math.max(2.15,radius) : radius;
      const signature = `${state.colorMode}|${effectiveRadius.toFixed(2)}|${node.i === state.selectedId ? 1 : 0}`;
      if (element.dataset.glyphSignature === signature) return;
      element.dataset.glyphSignature = signature;
      radius = effectiveRadius;
      const glyph = d3.select(element);
      glyph.selectAll("*").remove();
      if (state.colorMode === "theme") {
        const slices = d3.pie().sort(null).value(value => value.count)(themeMix(node).filter(value => value.count > 0));
        const arc = d3.arc().innerRadius(0).outerRadius(radius);
        glyph.selectAll("path.theme-slice").data(slices).join("path").attr("class","theme-slice")
          .attr("d",arc).attr("fill",slice => DATA.themeColor[slice.data.theme])
          .attr("stroke","#0B1013").attr("stroke-width",.18);
      } else if (state.colorMode === "sentiment") {
        glyph.append("circle").attr("r",Math.max(.5,radius*.34)).attr("fill",SENTIMENT_SCALE(node.p));
        glyph.append("path").attr("d",d3.arc()({innerRadius:radius*.40,outerRadius:radius*.66,startAngle:0,endAngle:Math.PI*2}))
          .attr("fill",SENTIMENT_SCALE(node.a));
        glyph.append("path").attr("d",d3.arc()({innerRadius:radius*.72,outerRadius:radius,startAngle:0,endAngle:Math.PI*2}))
          .attr("fill",SENTIMENT_SCALE(node.d));
      } else {
        glyph.append("circle").attr("r",radius)
          .attr("fill",node.cm ? COMMUNITY_COLORS[(node.cm - 1) % COMMUNITY_COLORS.length] : "#59656B");
      }
      glyph.append("circle").attr("r",radius+.25).attr("fill","none").attr("stroke","#F7F9FA")
        .attr("stroke-width",node.i === state.selectedId ? 1.45 : .32).attr("stroke-opacity",node.i === state.selectedId ? 1 : .64);
      glyph.append("circle").attr("r",Math.max(4,radius+1.4)).attr("fill","transparent").attr("class","node-hit");
    }

    function render() {
      updateOverlayFrame();
      const visible = currentVisibleData();
      const position = new Map(visible.nodes.map(node => [node.i,point(node)]));
      const neighbors = selectedNeighbors();
      const hasSelection = state.selectedId !== null;
      const bridgeVisibleIds = new Set(visible.nodes.filter(node => bridgeNodeIds.has(node.i)).map(node => node.i));

      edgeLayer.selectAll("line.network-edge").data(visible.edges,edge => `${edge.s}-${edge.t}-${edge.th}`).join(
        enter => enter.append("line").attr("class","network-edge").attr("stroke-linecap","round")
          .on("mouseenter",(event,edge) => {
            const source = nodeMap.get(edge.s), target = nodeMap.get(edge.t);
            showTooltip(event,`<h4>${escapeHtml(source.n)} → ${escapeHtml(target.n)}</h4><div class="tip-row"><span>Route frequency</span><b>${edge.w}</b></div><div class="tip-row"><span>Average distance</span><b>${Math.round(edge.d)} m</b></div><div class="tip-row"><span>Experience</span><b>${escapeHtml(DATA.themeLabel[edge.th])}</b></div>`);
          }).on("mousemove",moveTooltip).on("mouseleave",hideTooltip),
        update => update,
        exit => exit.remove()
      ).attr("x1",edge => position.get(edge.s).x).attr("y1",edge => position.get(edge.s).y)
       .attr("x2",edge => position.get(edge.t).x).attr("y2",edge => position.get(edge.t).y)
       .attr("stroke",edge => hasSelection && (edge.s === state.selectedId || edge.t === state.selectedId) ? "#ffffff" : (state.showBridges && (bridgeVisibleIds.has(edge.s) || bridgeVisibleIds.has(edge.t)) ? "#72D7E8" : DATA.themeColor[edge.th]))
       .attr("stroke-width",edge => hasSelection && (edge.s === state.selectedId || edge.t === state.selectedId) ? edgeWidth(edge)+.65 : (state.showBridges && (bridgeVisibleIds.has(edge.s) || bridgeVisibleIds.has(edge.t)) ? edgeWidth(edge)+.5 : edgeWidth(edge)))
       .attr("stroke-opacity",edge => {
         if (hasSelection) return edge.s === state.selectedId || edge.t === state.selectedId ? .92 : .055;
         if (state.showBridges) return bridgeVisibleIds.has(edge.s) || bridgeVisibleIds.has(edge.t) ? .78 : .045;
         return .34;
       })
       .attr("marker-end",edge => {
         const selectedEdge = edge.s === state.selectedId || edge.t === state.selectedId;
         const bridgeEdge = state.showBridges && (bridgeVisibleIds.has(edge.s) || bridgeVisibleIds.has(edge.t));
         if (!state.showArrows || (!selectedEdge && edge.w < Math.max(8,visible.edgeThreshold))) return null;
         return `url(#${selectedEdge ? "arrow-selected" : (bridgeEdge ? "arrow-bridge" : `arrow-${edge.th}`)})`;
       });

      bridgeLayer.selectAll("circle.bridge-ring").data(state.showBridges ? visible.nodes.filter(node => bridgeNodeIds.has(node.i)) : [],node => node.i).join("circle")
        .attr("class","bridge-ring")
        .attr("cx",node => position.get(node.i).x).attr("cy",node => position.get(node.i).y)
        .attr("r",node => nodeRadius(node)+3.2+3.4*Math.sqrt(node.bt/maxBetweenness))
        .attr("stroke-width",node => .7+1.5*Math.sqrt(node.bt/maxBetweenness))
        .attr("stroke-opacity",node => .48+.46*Math.sqrt(node.bt/maxBetweenness));

      renderExpandedMembers();
      nodeLayer.selectAll("g.map-node").data(visible.nodes,node => node.i).join(
        enter => enter.append("g").attr("class","map-node")
          .on("mouseenter",(event,node) => showTooltip(event,nodeTooltip(node)))
          .on("mousemove",moveTooltip)
          .on("mouseleave",hideTooltip)
          .on("click",(event,node) => { event.stopPropagation(); selectNode(node.i,true); }),
        update => update,
        exit => exit.remove()
      ).classed("is-selected",node => node.i === state.selectedId)
       .attr("transform",node => `translate(${position.get(node.i).x},${position.get(node.i).y})`)
       .attr("opacity",node => {
         if (hasSelection) return neighbors.has(node.i) ? 1 : .12;
         if (state.showBridges) return bridgeNodeIds.has(node.i) ? 1 : .14;
         return .9;
       })
       .each(function(node) { drawNodeGlyph(this,node,nodeRadius(node) + (node.i === state.selectedId ? 1.15 : 0)); });

      let labelNodes = [];
      if (state.showLabels && state.showBridges) labelNodes = visible.nodes.filter(node => bridgeNodeIds.has(node.i) || node.i === state.selectedId).sort((a,b) => b.bt-a.bt).slice(0,26);
      else if (state.showLabels) labelNodes = visible.nodes.filter(node => node.w >= 100 || node.i === state.selectedId).sort((a,b) => b.w-a.w).slice(0,32);
      else if (state.selectedId !== null && visible.nodeIds.has(state.selectedId)) labelNodes = [nodeMap.get(state.selectedId)];
      labelLayer.selectAll("text.node-label").data(labelNodes,node => node.i).join("text")
        .attr("class","node-label").attr("x",node => position.get(node.i).x).attr("y",node => position.get(node.i).y-nodeRadius(node)-(state.showBridges && bridgeNodeIds.has(node.i) ? 10 : 4))
        .attr("text-anchor","middle").attr("fill","#F7FAFB").attr("font-size",12).attr("font-weight",700)
        .attr("paint-order","stroke").attr("stroke","#0B1013").attr("stroke-width",4).attr("stroke-linejoin","round")
        .style("pointer-events","none").text(node => node.n);

      updateLegend(); updateVisibleMetrics(visible);
    }

    function updateVisibleMetrics(visible) {
      const records = d3.sum(visible.edges,edge => edge.w);
      const meanPleasure = d3.mean(visible.nodes,node => node.p) ?? 0;
      const meanArousal = d3.mean(visible.nodes,node => node.a) ?? 0;
      const meanDominance = d3.mean(visible.nodes,node => node.d) ?? 0;
      const visibleBridges = visible.nodes.filter(node => bridgeNodeIds.has(node.i));
      const strongestBridge = [...visibleBridges].sort((left,right) => right.bt-left.bt)[0];
      const analysisMetrics = state.showBridges ? `
        <div class="metric"><strong>${visibleBridges.length}</strong><span>top-2% bridges</span></div>
        <div class="metric"><strong>${strongestBridge ? strongestBridge.bt.toFixed(3) : "0.000"}</strong><span>max betweenness</span></div>
        <div class="metric"><strong>${strongestBridge ? `#${bridgeRankMap.get(strongestBridge.i)}` : "—"}</strong><span>global bridge rank</span></div>` : `
        <div class="metric"><strong>${meanPleasure.toFixed(2)}</strong><span>mean pleasure</span></div>
        <div class="metric"><strong>${meanArousal.toFixed(2)}</strong><span>mean arousal</span></div>
        <div class="metric"><strong>${meanDominance.toFixed(2)}</strong><span>mean dominance</span></div>`;
      document.getElementById("visible-metrics").innerHTML = `
        <div class="metric"><strong>${visible.nodes.length.toLocaleString()}</strong><span>places</span></div>
        <div class="metric"><strong>${visible.edges.length.toLocaleString()}</strong><span>transitions</span></div>
        <div class="metric"><strong>${records.toLocaleString()}</strong><span>path records</span></div>
        ${analysisMetrics}`;
      document.getElementById("map-status").innerHTML = `<b>${visible.nodes.length.toLocaleString()}</b> visible places <span>·</span> <b>${visible.edges.length.toLocaleString()}</b> directed edges <span>·</span> <b>${records.toLocaleString()}</b> path records`;
      document.getElementById("theme-count-label").textContent = `${state.activeThemes.size} of 4 active`;
      document.getElementById("edge-threshold-label").textContent = `frequency ≥ ${visible.edgeThreshold}`;
      document.getElementById("node-threshold-label").textContent = `frequency ≥ ${visible.nodeThreshold}`;
      document.getElementById("edge-threshold").setAttribute("aria-valuetext",`Show routes with frequency ${visible.edgeThreshold} or higher`);
      document.getElementById("node-threshold").setAttribute("aria-valuetext",`Show places with frequency ${visible.nodeThreshold} or higher`);
    }

    function updateLegend() {
      const legend = document.getElementById("map-legend");
      if (state.colorMode === "sentiment") {
        legend.innerHTML = `<div class="legend-title">Places · Mood profile</div><div class="sentiment-legend"><svg viewBox="0 0 42 42" aria-label="Three concentric mood channels"><circle cx="21" cy="21" r="7" fill="#E8892F"></circle><circle cx="21" cy="21" r="12" fill="none" stroke="#D8DCDE" stroke-width="5"></circle><circle cx="21" cy="21" r="18" fill="none" stroke="#3B73B9" stroke-width="4"></circle></svg><div><div class="sentiment-channel"><b>P</b><span>inner · Pleasure</span></div><div class="sentiment-channel"><b>A</b><span>middle · Arousal</span></div><div class="sentiment-channel"><b>D</b><span>outer · Dominance</span></div></div></div><div class="gradient"></div><div class="gradient-labels"><span>−1 lower</span><span>0 neutral</span><span>+1 higher</span></div><div class="legend-title" style="margin-top:10px">Place size · Popularity</div>${sizeLegend()}`;
      } else if (state.colorMode === "community") {
        legend.innerHTML = `<div class="legend-title">Places · Route communities</div><div class="legend-grid">${Array.from({length:14},(_,index) => `<div class="legend-item"><span class="swatch" style="background:${COMMUNITY_COLORS[index]}"></span>C${index+1}</div>`).join("")}</div><div class="legend-item" style="margin-top:8px"><span class="swatch" style="background:#59656B"></span>Other groups / unconnected</div>`;
      } else {
        legend.innerHTML = `<div class="legend-title">Places · Experience mix</div><div class="legend-grid">${THEME_ORDER.map(theme => `<div class="legend-item"><span class="swatch" style="background:${DATA.themeColor[theme]}"></span>${DATA.themeLabel[theme]}</div>`).join("")}</div><div class="legend-note">Each place can blend four experience types. Connections keep the colour of their route type.</div><div class="legend-title" style="margin-top:10px">Place size · Popularity</div>${sizeLegend()}`;
      }
      if (state.showBridges) legend.innerHTML += `<div class="legend-title" style="margin-top:11px">Key connectors</div><div class="bridge-legend"><span class="bridge-symbol"></span><span>Cyan rings mark the top 2% of places that connect otherwise separate walking routes.</span></div>`;
    }
    function sizeLegend() {
      return `<div class="size-legend">${[10,100,500].map(value => `<div class="size-example"><span class="size-circle" style="width:${Math.max(4,nodeRadius({w:value}))*2}px;height:${Math.max(4,nodeRadius({w:value}))*2}px"></span>${value}</div>`).join("")}</div>`;
    }

    function padRow(label,value,color) {
      const magnitude = Math.abs(value) * 50;
      const left = value < 0 ? 50 - magnitude : 50;
      return `<div class="pad-row"><span>${label}</span><div class="pad-track"><div class="pad-fill" style="left:${left}%;width:${magnitude}%;background:${color}"></div></div><b>${value.toFixed(2)}</b></div>`;
    }
    function selectNode(nodeId,fly=true) {
      state.selectedId = nodeId;
      state.findingsNodeId = nodeId;
      const node = nodeMap.get(nodeId);
      if (fly && prefersReducedMotion) map.setView([node.y,node.x],Math.max(map.getZoom(),16),{animate:false});
      else if (fly) map.flyTo([node.y,node.x],Math.max(map.getZoom(),16),{duration:.55});
      const sourceFilter = document.getElementById("source-place-filter");
      if (sourceFilter) sourceFilter.value = "";
      renderSelectedCard(node); renderSourcePlaceList(node); render();
    }
    function themeDetails(node) {
      return themeComposition(node).map(value => `<div class="theme-breakdown-item"><span class="dot" style="background:${DATA.themeColor[value.theme]}"></span><span>${escapeHtml(DATA.themeLabel[value.theme])}</span><b>${value.count.toLocaleString()} · ${Math.round(value.share*100)}%</b></div>`).join("");
    }
    function excerptCard(excerpt) {
      const [quote,note,date,place,theme,p,a,d,inherited] = excerpt;
      return `<article class="excerpt-card" style="--excerpt-color:${DATA.themeColor[theme]}"><blockquote>“${escapeHtml(quote)}”</blockquote><div class="excerpt-meta"><span><b>${escapeHtml(place)}</b> · ${escapeHtml(DATA.themeLabel[theme])}</span><span>${escapeHtml(note)}${date ? ` · ${escapeHtml(date)}` : ""} · P ${Number(p).toFixed(1)} / A ${Number(a).toFixed(1)} / D ${Number(d).toFixed(1)}${inherited ? " · inherited context" : " · direct evidence"}</span></div></article>`;
    }
    function activateTab(tabName) {
      document.querySelectorAll(".tab-button").forEach(button => {
        const active = button.dataset.tab === tabName;
        button.classList.toggle("active",active);
        button.setAttribute("aria-selected",String(active));
        button.tabIndex = active ? 0 : -1;
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        const active = panel.id === `tab-${tabName}`;
        panel.classList.toggle("active",active);
        panel.hidden = !active;
      });
      document.getElementById("sidebar-body").scrollTo({top:0,behavior:prefersReducedMotion ? "auto" : "smooth"});
      if (tabName === "findings") renderSourcePlaceList(nodeMap.get(state.findingsNodeId));
    }
    function renderSourcePlaceList(node,query="") {
      if (!node) return;
      const normalizedQuery = query.trim().toLowerCase();
      const members = node.m.filter(member => String(member[0]).toLowerCase().includes(normalizedQuery));
      const totalRecords = d3.sum(node.m,member => Number(member[1]));
      document.getElementById("source-place-count-label").textContent = `${members.length.toLocaleString()} of ${node.mc.toLocaleString()} places`;
      document.getElementById("source-context").innerHTML = `<h3>${escapeHtml(node.n)}</h3><p>Explore the place names grouped within 100 m of this point. Repeated names are combined and ranked by how often they appear.</p><div class="source-summary"><div><b>${node.mc.toLocaleString()}</b><span>nearby names</span></div><div><b>${totalRecords.toLocaleString()}</b><span>mentions</span></div><div><b>${escapeHtml(DATA.themeLabel[node.t])}</b><span>top experience</span></div></div>`;
      document.getElementById("source-list-note").textContent = normalizedQuery ? `${members.length.toLocaleString()} matching names. The colour shows the leading experience type for each place.` : `Sorted by mention frequency. Colours show each place's leading experience type; P/A/D values summarize its mood profile where available.`;
      document.getElementById("source-place-list").innerHTML = members.length ? members.map((member,index) => `<div class="source-place-row"><span class="source-index">${index+1}</span><span class="dot" style="background:${DATA.themeColor[member[4]]}"></span><span class="source-place-copy"><b>${escapeHtml(member[0])}</b><span>${escapeHtml(DATA.themeLabel[member[4]])} · P ${Number(member[5]).toFixed(2)} / A ${Number(member[6]).toFixed(2)} / D ${Number(member[7]).toFixed(2)}</span></span><span class="source-place-count">× ${Number(member[1]).toLocaleString()}</span></div>`).join("") : `<div class="empty-state">No original place name matches this filter.</div>`;
    }
    function renderSelectedCard(node) {
      const links = [...(incidentMap.get(node.i) || [])].sort((a,b) => b.w-a.w).slice(0,8);
      const communityLabel = node.cm ? `C${node.cm}` : node.cs;
      const selectedCard = document.getElementById("selected-card");
      selectedCard.classList.remove("is-updating");
      void selectedCard.offsetWidth;
      selectedCard.classList.add("is-updating");
      selectedCard.innerHTML = `<div class="node-heading"><div><h3>${escapeHtml(node.n)}</h3><div class="node-subtitle"><span class="swatch" style="display:inline-block;vertical-align:-1px;margin-right:5px;background:${DATA.themeColor[node.t]}"></span>Experience: ${escapeHtml(DATA.themeLabel[node.t])} · ${escapeHtml(communityLabel)}</div></div><div class="rank-badge">popularity #${node.r}</div></div>
        <div class="node-kpis"><div class="node-kpi"><b>${node.w.toLocaleString()}</b><span>place mentions</span></div><div class="node-kpi"><b>${node.mc.toLocaleString()}</b><span>nearby names</span></div><div class="node-kpi"><b>${node.iw} / ${node.ow}</b><span>route flow in / out</span></div><div class="node-kpi"><b>${Math.round(Math.max(...node.mt)/(d3.sum(node.mt)||1)*100)}%</b><span>top experience</span></div><div class="node-kpi"><b>#${bridgeRankMap.get(node.i)}</b><span>connector rank</span></div><div class="node-kpi"><b>${(node.pr*1000).toFixed(2)}</b><span>route reach ×10³</span></div></div>
        <div class="theme-block"><div class="section-title" style="margin-bottom:2px">Experience mix <small>${d3.sum(node.mt).toLocaleString()} classified mentions</small></div>${compositionBar(node)}<div class="theme-breakdown">${themeDetails(node)}</div><div class="aggregate-note" style="margin-top:7px">The leading experience is the largest share; many places blend several types.</div></div>
        <div class="pad-block"><div class="section-title" style="margin-bottom:2px">Mood profile <small>Pleasure · Arousal · Dominance</small></div>${padRow("P",node.p,SENTIMENT_SCALE(node.p))}${padRow("A",node.a,SENTIMENT_SCALE(node.a))}${padRow("D",node.d,SENTIMENT_SCALE(node.d))}</div>
        <div class="theme-block"><div class="section-title" style="margin-bottom:3px">Nearby place cluster <small>${node.mc.toLocaleString()} original names</small></div><div class="aggregate-note">Place details lists every original name grouped into this point, together with frequency, experience type and P/A/D mood profile.</div><button class="toggle view-findings" type="button">View all nearby place names</button></div>
        <div class="links-title">What walkers say · Xiaohongshu</div>
        <div class="excerpt-list">${node.ex.length ? node.ex.map(excerptCard).join("") : `<div class="empty-state">No non-empty source excerpt is retained for this aggregate.</div>`}</div>
        <div class="links-title">Popular onward connections</div>
        <div>${links.length ? links.map(edge => {
          const targetId = edge.direction === "out" ? edge.t : edge.s;
          const target = nodeMap.get(targetId);
          return `<button class="link-row" data-node-id="${targetId}"><span class="arrow">${edge.direction === "out" ? "→" : "←"}</span><span class="destination">${escapeHtml(target.n)}</span><span class="weight">× ${edge.w}</span></button>`;
        }).join("") : `<div class="empty-state">No retained directed transition.</div>`}</div>`;
      selectedCard.querySelectorAll(".link-row").forEach(button => button.addEventListener("click",() => selectNode(Number(button.dataset.nodeId),true)));
      selectedCard.querySelector(".view-findings").addEventListener("click",() => {
        activateTab("findings");
        document.getElementById("tab-button-findings").focus();
      });
    }

    function buildControls() {
      const themeChips = document.getElementById("theme-chips");
      THEME_ORDER.forEach(theme => {
        const button = document.createElement("button"); button.type = "button"; button.className = "chip active"; button.style.setProperty("--chip-color",DATA.themeColor[theme]);
        button.setAttribute("aria-pressed","true");
        button.innerHTML = `<span class="swatch" style="display:inline-block;vertical-align:-1px;margin-right:5px;background:${DATA.themeColor[theme]}"></span>${DATA.themeLabel[theme]}`;
        button.addEventListener("click",() => {
          state.activeThemes.has(theme) ? state.activeThemes.delete(theme) : state.activeThemes.add(theme);
          const active = state.activeThemes.has(theme);
          button.classList.toggle("active",active);
          button.setAttribute("aria-pressed",String(active));
          render();
        });
        themeChips.appendChild(button);
      });
      document.querySelectorAll(".mode-button").forEach(button => button.addEventListener("click",() => {
        document.querySelectorAll(".mode-button").forEach(item => {
          const active = item === button;
          item.classList.toggle("active",active);
          item.setAttribute("aria-pressed",String(active));
        });
        state.colorMode = button.dataset.mode; render();
      }));
      document.getElementById("edge-threshold").addEventListener("input",event => { state.edgeLevelIndex = Number(event.target.value); scheduleRender(); });
      document.getElementById("node-threshold").addEventListener("input",event => { state.nodeLevelIndex = Number(event.target.value); if (state.selectedId !== null && !currentVisibleData().nodeIds.has(state.selectedId)) state.selectedId = null; scheduleRender(); });
      const bindToggle = (id,getValue,setValue) => document.getElementById(id).addEventListener("click",event => {
        const active = !getValue(); setValue(active); event.currentTarget.classList.toggle("active",active); event.currentTarget.setAttribute("aria-pressed",String(active)); render();
      });
      bindToggle("toggle-arrows",() => state.showArrows,value => { state.showArrows = value; });
      bindToggle("toggle-labels",() => state.showLabels,value => { state.showLabels = value; });
      bindToggle("toggle-bridges",() => state.showBridges,value => { state.showBridges = value; });
      document.getElementById("reset-view").addEventListener("click",() => { clearSelectedPlace(); map.fitBounds([[24.415,118.00],[24.635,118.20]],{padding:[18,18]}); render(); });
      const tabButtons = [...document.querySelectorAll(".tab-button")];
      tabButtons.forEach((button,index) => {
        button.addEventListener("click",() => activateTab(button.dataset.tab));
        button.addEventListener("keydown",event => {
          if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
          event.preventDefault();
          const targetIndex = event.key === "Home" ? 0 : event.key === "End" ? tabButtons.length-1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabButtons.length) % tabButtons.length;
          activateTab(tabButtons[targetIndex].dataset.tab); tabButtons[targetIndex].focus();
        });
      });
      document.getElementById("source-place-filter").addEventListener("input",event => renderSourcePlaceList(nodeMap.get(state.findingsNodeId),event.target.value));
      const search = document.getElementById("node-search"), results = document.getElementById("search-results");
      const setSearchOpen = open => {
        results.style.display = open ? "block" : "none";
        results.classList.toggle("open",open);
        search.setAttribute("aria-expanded",String(open));
      };
      const chooseSearchResult = button => {
        const nodeId = Number(button.dataset.nodeId);
        search.value = nodeMap.get(nodeId).n;
        setSearchOpen(false);
        search.focus();
        selectNode(nodeId,true);
      };
      search.addEventListener("input",() => {
        const query = search.value.trim().toLowerCase();
        if (!query) { setSearchOpen(false); return; }
        const matches = DATA.nodes.filter(node => node.n.toLowerCase().includes(query) || node.zh.toLowerCase().includes(query) || node.m.some(member => String(member[0]).toLowerCase().includes(query) || String(member[8]).toLowerCase().includes(query))).sort((a,b) => b.w-a.w).slice(0,9);
        results.innerHTML = matches.map(node => `<button class="search-result" type="button" role="option" data-node-id="${node.i}"><span>${escapeHtml(node.n)}</span><span>${node.w} mentions</span></button>`).join("") || `<div class="empty-state">No matching place.</div>`;
        setSearchOpen(true);
        results.querySelectorAll(".search-result").forEach(button => button.addEventListener("click",() => chooseSearchResult(button)));
      });
      search.addEventListener("keydown",event => {
        if (event.key === "Escape") { setSearchOpen(false); return; }
        if (event.key === "ArrowDown") { const first = results.querySelector(".search-result"); if (first) { event.preventDefault(); first.focus(); } }
      });
      results.addEventListener("keydown",event => {
        const options = [...results.querySelectorAll(".search-result")];
        const index = options.indexOf(document.activeElement);
        if (event.key === "Escape") { event.preventDefault(); setSearchOpen(false); search.focus(); }
        if (event.key === "ArrowDown" && index >= 0) { event.preventDefault(); options[Math.min(index+1,options.length-1)].focus(); }
        if (event.key === "ArrowUp" && index >= 0) { event.preventDefault(); if (index === 0) search.focus(); else options[index-1].focus(); }
      });
      document.addEventListener("click",event => { if (!event.target.closest(".search-wrap")) setSearchOpen(false); });
      const app = document.getElementById("app"), sidebarToggle = document.getElementById("sidebar-toggle");
      sidebarToggle.addEventListener("click",() => {
        const collapsed = app.classList.toggle("sidebar-collapsed");
        sidebarToggle.setAttribute("aria-expanded",String(!collapsed));
        sidebarToggle.querySelector(".toggle-label").textContent = collapsed ? "Show panel" : "Hide panel";
        window.setTimeout(() => { map.invalidateSize({pan:false}); scheduleRender(); },prefersReducedMotion ? 0 : 350);
      });
      const representativeList = document.getElementById("representative-list");
      representativeList.innerHTML = DATA.representative.map((featured,index) => {
        const node = nodeMap.get(featured.i);
        return `<button class="representative" data-node-id="${node.i}"><span class="number">${index+1}</span><span class="names"><b>${escapeHtml(featured.n)}</b><span>${node.mc} source places · ${node.w} records</span></span><span class="sentiment-badge">P ${node.p.toFixed(1)} · A ${node.a.toFixed(1)} · D ${node.d.toFixed(1)}</span></button>`;
      }).join("");
      representativeList.querySelectorAll(".representative").forEach(button => button.addEventListener("click",() => selectNode(Number(button.dataset.nodeId),true)));
      renderSourcePlaceList(nodeMap.get(state.findingsNodeId));
    }

    let renderFrame = null;
    function scheduleRender() {
      if (renderFrame !== null) return;
      renderFrame = window.requestAnimationFrame(() => { renderFrame = null; render(); });
    }
    function clearSelectedPlace() {
      state.selectedId = null;
      hideTooltip();
      document.getElementById("selected-card").innerHTML = `<div class="empty-state">Choose a place to see its character, mood, nearby names and strongest onward routes.</div>`;
    }
    map.on("zoom move resize",scheduleRender);
    map.on("click",() => { clearSelectedPlace(); render(); });
    buildControls(); render();
  </script>
</body>
</html>'''


def load_embedded_payload() -> dict[str, object]:
    """Reuse the last generated data payload when upstream research files are absent."""
    html = OUTPUT_HTML.read_text(encoding="utf-8")
    data_marker = "    const DATA = "
    end_marker = ";\n    const THEME_ORDER"
    data_start = html.index(data_marker) + len(data_marker)
    data_end = html.index(end_marker, data_start)
    return json.loads(html[data_start:data_end])


def main() -> None:
    try:
        payload = load_data()
    except FileNotFoundError as error:
        if not OUTPUT_HTML.exists():
            raise
        payload = load_embedded_payload()
        print(f"Source data unavailable ({error.filename}); reused the embedded page payload.")
    data_json = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_HTML}")
    print(f"Size: {OUTPUT_HTML.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
