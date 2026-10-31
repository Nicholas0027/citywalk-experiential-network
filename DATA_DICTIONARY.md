# Data dictionary

## POI record tables

`data/poi_records.csv` contains the integrated place records. Each row represents
a place record associated with one post. `data/network_input_records.csv` is the
geographically screened subset used for network construction and has the same
columns. `(note_index, location_index)` locates a record within its reported
outing.

| Column | Type | Meaning |
| --- | --- | --- |
| `note_index` | string | Internal post identifier shared by records from the same post. |
| `original_date` | string | Posting date as retained in the processing output. |
| `location_index` | number | Position of the place record within the post's ordered place sequence. |
| `location_POI` | string | Place name retained from the processing output. |
| `is_inherited` | string | Indicator of whether appraisal was inherited from the wider post context. |
| `P_score` | number | Pleasure appraisal on the −1 to +1 scale. |
| `A_score` | number | Arousal appraisal on the −1 to +1 scale. |
| `D_score` | number | Dominance appraisal on the −1 to +1 scale. |
| `Primary_Theme` | string | Primary route theme assigned to the post. |
| `lng` | number | Place longitude in decimal degrees, WGS 84. |
| `lat` | number | Place latitude in decimal degrees, WGS 84. |

Pleasure describes positive-to-negative appraisal, Arousal describes calm-to-active
appraisal, and Dominance describes perceived freedom or control. Some retained
scores are fractional because earlier processing combined records. The released
values have not been rounded or recoded.

## Network nodes

Each row of `data/network_nodes.csv` represents one occupied 100 m grid cell.
Locations are means of the contributing records. Representative names and themes
are selected by their frequency within the cell.

| Column | Type | Meaning |
| --- | --- | --- |
| `cluster_id` | integer | Unique node identifier, referenced by the edge table. |
| `lng` | number | Mean longitude of the contributing place records. |
| `lat` | number | Mean latitude of the contributing place records. |
| `location_POI` | string | Representative place name for the node. |
| `Primary_Theme` | string | Most frequent theme among the contributing records. |
| `point_weight` | integer | Number of place records aggregated into the node. |
| `mean_P` | number | Mean Pleasure appraisal of the contributing records. |
| `mean_A` | number | Mean Arousal appraisal of the contributing records. |
| `mean_D` | number | Mean Dominance appraisal of the contributing records. |

Node weights sum to 25,380, the number of network-input records.

## Network edges

Each row of `data/network_edges.csv` represents a distinct ordered pair of nodes.
The edge table contains 2,879 directed connections and 7,734 retained transition
occurrences in total.

| Column | Type | Meaning |
| --- | --- | --- |
| `start` | integer | Origin node identifier, joining to `cluster_id`. |
| `end` | integer | Destination node identifier, joining to `cluster_id`. |
| `line_weight` | integer | Number of retained transition occurrences for this ordered pair. |
| `Theme` | string | Most frequent route theme among contributing transitions. |
| `mean_distance_m` | number | Mean geodesic distance in meters between the aggregate endpoint locations. |
| `start_POI` | string | Representative place name of the origin node. |
| `end_POI` | string | Representative place name of the destination node. |
| `edge_label` | string | Readable origin-to-destination label. |

Connections join consecutive records in a post after ordering by
`location_index`. Adjacent records assigned to the same node do not create an
edge. Retained connections have distances at most 1,500 m.

## Theme values

Original category strings are retained consistently across the tables:

| Stored value | Label used in the paper |
| --- | --- |
| `Commericial & Tourism` | Commercial & Tourism |
| `Historical & Art` | Historical & Art |
| `Local_Life` | Local Life |
| `Nature` | Nature |

## Metadata

`data/metadata.json` records the study area, source platform, coordinate system,
construction parameters, column lists, row counts, post counts where applicable,
and a SHA-256 checksum for each CSV. The checksum paths are relative to the
repository root. Post counts apply to record tables; node and edge tables have
a null post-count value.
