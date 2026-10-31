# Citywalk Experiential Network Data

Processed place records and the derived experiential urban network for Citywalk
in Xiamen, China. The data support analysis of reported visit sequences, route
themes, and Pleasure–Arousal–Dominance (PAD) appraisals.

## Data source and preparation

The source material comprises public Xiaohongshu (RedNote) Citywalk posts
published between May 2020 and January 2025. Processing identified ordered
places, assigned route themes, coded affective appraisals, and geocoded places.
The integrated table contains 26,131 POI records from 4,528 posts. Geographic
and required-field screening retained 25,380 records from 4,489 posts for
network construction.

The network aggregates records on a 100 m projected grid. Consecutively reported
places in different cells are linked when the distance between their aggregate
locations is at most 1,500 m. It contains 2,413 nodes and 2,879 directed
connections. Connection weights count retained transitions reported across posts.

## Files

| File | Content | Rows |
| --- | --- | ---: |
| [poi_records.csv](data/poi_records.csv) | Integrated POI records from 4,528 posts | 26,131 |
| [network_input_records.csv](data/network_input_records.csv) | Records from 4,489 posts used to construct the network | 25,380 |
| [network_nodes.csv](data/network_nodes.csv) | Aggregate place locations, themes, mention counts, and mean appraisals | 2,413 |
| [network_edges.csv](data/network_edges.csv) | Directed connections, frequencies, themes, and distances | 2,879 |
| [metadata.json](data/metadata.json) | File schemas, counts, construction parameters, and SHA-256 checksums | — |

[DATA_DICTIONARY.md](DATA_DICTIONARY.md) explains every column and how the tables
relate. All CSV files use UTF-8 with a byte-order mark and can be opened in
spreadsheet software or read using `encoding="utf-8-sig"`. Coordinates use
WGS 84 (EPSG:4326), with longitude before latitude.

## Interpretation

Post identifiers are internal record identifiers. Theme categories describe
the purpose of an outing; node and edge themes summarize the associated records.
PAD scores range from −1 to +1, and fractional values may occur after aggregation.
Connections represent consecutively reported places. Their distances are
geodesic estimates between aggregate locations. Appraisals summarize descriptions
in the source posts.

The public tables contain structured place and network attributes. Original
post bodies, titles, account information, source quotations, and model-generated
textual explanations are excluded. Retained record values and network statistics
are preserved from the analysis tables.

## Access and citation

Download individual files above or use GitHub's **Code → Download ZIP** option.
Please cite the associated paper and this repository when using the data:

https://github.com/Nicholas0027/citywalk-experiential-network

A data availability statement for the associated article is:

> The processed POI records and derived Citywalk network data are publicly
> available at https://github.com/Nicholas0027/citywalk-experiential-network,
> together with a data dictionary and dataset metadata.
