# Citywalk Experiential Network

Code, processed data, analyses, and an interactive explorer for a Citywalk network derived
from social-media accounts of visits in Xiamen, China.

The project links ordered place visits with route themes, Pleasure–Arousal–Dominance
(PAD) appraisals, and supporting descriptions. Its reference network contains
**2,413 nodes and 2,879 directed connections**, using 100 m spatial aggregation
and a 1,500 m guard between consecutive reported places.

## Explore the network

The interactive page is prebuilt and includes its network data. From this
repository's root, run:

```sh
python3 -m http.server 8000
```

Open **http://localhost:8000/visualization/citywalk_visual_analytics.html**.
The page requires internet access for Leaflet, D3, map tiles, and web fonts;
no API key or application server is needed to inspect the included network.

The explorer supports geographic search, theme and frequency filters, sentiment
and community views, English place labels with Chinese search, and expansion of
the original place names grouped into a network node. Bridge focus highlights
nodes in the top 2% by betweenness centrality and their connections. Place cards
connect the aggregate network to constituent places. The public export omits
source-post quotations, so quotation panels contain no excerpts.

## Repository layout

```text
visualization/    Interactive page, Python builders, English place-name table
notebooks/        Original research workflow with credentials and outputs removed
code/
  validation/    Source-text audit and single-prompt comparison
  robustness/    Spatial construction and threshold analysis
  network/       Route patterns, centrality, sentiment, and network coupling
  mapping/       Geographic network visualization
  communities/   Overall and theme-specific community analysis
  subnetworks/   Theme-specific network construction and figures
  workflow/      Formal workflow description
data/
  raw/           Place-exclusion vocabulary
  analysis/      POI records, reference network, and plotting inputs
  validation/    Sampled posts, POIs, and audit annotations
  spatial/       Xiamen district boundary
results/         Archived analytical tables, network outputs, and figures
```

## Install the analysis environment

Python 3.11 or 3.12 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For visualization builds alone, install
`visualization/requirements.txt` instead. The included HTML works without a
Python package installation.

## Rebuild the interactive page

```sh
python visualization/build_citywalk_visualisation.py
```

This reads the repository's network, community assignments, POI records,
boundary, and place-name translations. The output is
`visualization/citywalk_visual_analytics.html`.

The translation table is included. To regenerate its labels:

```sh
python visualization/build_place_name_translations.py
python visualization/build_citywalk_visualisation.py
```

## Data and workflow

`data/analysis/final_poi_records.csv` contains the linked POI, sequence, theme,
appraisal, and coordinate records, with internal post identifiers. Geographic screening
produces the reference network stored as
`reference_nodes_100m_1500m.csv` and `reference_edges_100m_1500m.csv`. Original
post text, titles, source quotations, and model-generated textual reasoning
are omitted. Corresponding text columns are empty to preserve input schemas.

`notebooks/citywalk_research_workflow.ipynb` preserves the original exploratory
analysis, prompts, and extraction/geocoding functions. It uses environment
variables `LLM_API_KEY` and `AMAP_API_KEY` in place of credentials. Run it from
`notebooks/`; intermediate files belong in `data/processing/`, and source-text
inference requires separately obtained local inputs. Its cells also
contain exploratory plots with optional dependencies, so use the scripts below
for the packaged analyses. Repeating external model or geocoding calls requires
your own credentials and can incur service charges.

## Reproduce core analyses

Run the following from the repository root. Results are written under `results/`.

| Analysis | Command | Main output |
| --- | --- | --- |
| Spatial construction and threshold robustness | `python code/robustness/threshold_robustness.py` | `results/robustness/threshold_sensitivity_summary.csv` |
| Theme-specific subnetworks | `python code/subnetworks/build_and_plot_theme_subnetworks.py` | `results/subnetworks/theme_subnetwork_summary.csv` |
| Node coupling | `python code/network/analyze_network_coupling.py` | `results/network/stage3_node_coupling_metrics.csv` |
| Multiple centrality measures | `python code/network/build_table2_multiple_centrality.py` | `results/network/table2_multiple_centrality_assessment.csv` |
| Overall and thematic communities | `python code/communities/analyze_network_communities.py` | `results/communities/network_community_diagnostics.csv` |
| Community figures | `python code/communities/plot_network_communities.py` | `results/figures/` |
| Spatial map | `python code/mapping/plot_spatial_network_map.py` | `results/figures/` |
| Route and node profiles | `python code/network/plot_route_and_node_profiles.py` | `results/figures/` |

Archived validation metrics are in `results/validation/`, with structured
annotations for the 100-post sample in `data/validation/route/`. The archived
single-prompt comparison uses GPT-5.4-mini. To repeat source-text inference,
provide a local input CSV through the script's `--input` option. The inference
script reads `ZETA_API_KEY` or `OPENAI_API_KEY`, and an optional `LLM_BASE_URL`.
Source post bodies and raw model responses are not distributed.

## Interpretation and reuse

Connections represent transitions reported in posts. Distances are geographic
approximations between aggregate places. Appraisals summarize descriptions in
the collected posts. Place names retain their original language, while the
explorer supplies an English label table for navigation.

Place data and third-party cartographic resources remain subject to their
respective rights and terms. The repository does not grant a new license
over those materials. Map attribution is retained in the interactive page.
