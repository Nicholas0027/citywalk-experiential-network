from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "validation" / "route"

THEMES = [
    "Commericial & Tourism",
    "Historical & Art",
    "Nature",
    "Local_Life",
]


def case_to_row_index(case_id: str) -> int | None:
    m = re.match(r"Case_(\d+)$", str(case_id))
    return int(m.group(1)) - 1 if m else None


def compact(text: str, limit: int = 5000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def read_csv(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(DATA_DIR / "analysis" / "final_poi_records.csv", "utf-8-sig")
    xhs_rows = read_csv(DATA_DIR / "raw" / "source_notes_anonymized.csv", "utf-8-sig")

    by_note: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in final_rows:
        by_note[row["note_index"]].append(row)

    note_rows = []
    for note_index, pois in by_note.items():
        theme = pois[0]["Primary_Theme"]
        xhs_i = case_to_row_index(note_index)
        title = xhs_rows[xhs_i].get("笔记标题", "") if xhs_i is not None and 0 <= xhs_i < len(xhs_rows) else ""
        content = xhs_rows[xhs_i].get("笔记内容", "") if xhs_i is not None and 0 <= xhs_i < len(xhs_rows) else ""
        note_rows.append(
            {
                "note_index": note_index,
                "Primary_Theme": theme,
                "n_poi": str(len(pois)),
                "title": compact(title, 600),
                "content": compact(content, 5000),
            }
        )

    rng = random.Random(20260719)
    sample_notes = []
    for theme in THEMES:
        candidates = [r for r in note_rows if r["Primary_Theme"] == theme and len(r["content"]) > 20]
        candidates.sort(key=lambda r: r["note_index"])
        sample_notes.extend(rng.sample(candidates, 25))

    sample_note_ids = {r["note_index"] for r in sample_notes}
    sample_pois = []
    for note in sample_notes:
        pois = sorted(by_note[note["note_index"]], key=lambda r: int(float(r["location_index"] or 0)))
        for poi in pois:
            sample_pois.append(
                {
                    "note_index": poi["note_index"],
                    "Primary_Theme": poi["Primary_Theme"],
                    "location_index": poi["location_index"],
                    "location_POI": poi["location_POI"],
                    "text_evidence": compact(poi["text_evidence"], 800),
                    "is_inherited": poi["is_inherited"],
                    "P_score": poi["P_score"],
                    "A_score": poi["A_score"],
                    "D_score": poi["D_score"],
                    "lng": poi["lng"],
                    "lat": poi["lat"],
                    "reasoning": compact(poi["reasoning"], 800),
                }
            )

    with (OUT_DIR / "route_validation_sample_notes.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_notes[0].keys()))
        writer.writeheader()
        writer.writerows(sample_notes)

    with (OUT_DIR / "route_validation_sample_pois.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_pois[0].keys()))
        writer.writeheader()
        writer.writerows(sample_pois)

    by_theme_counts = defaultdict(lambda: {"notes": 0, "pois": 0})
    for r in sample_notes:
        by_theme_counts[r["Primary_Theme"]]["notes"] += 1
    for r in sample_pois:
        by_theme_counts[r["Primary_Theme"]]["pois"] += 1

    with (OUT_DIR / "route_validation_sample_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Primary_Theme", "sampled_notes", "sampled_pois"])
        writer.writeheader()
        for theme in THEMES:
            writer.writerow(
                {
                    "Primary_Theme": theme,
                    "sampled_notes": by_theme_counts[theme]["notes"],
                    "sampled_pois": by_theme_counts[theme]["pois"],
                }
            )

    print(f"Wrote {len(sample_notes)} sampled notes and {len(sample_pois)} POI records.")
    for theme in THEMES:
        print(theme, by_theme_counts[theme])


if __name__ == "__main__":
    main()
