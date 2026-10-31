#!/usr/bin/env python3
"""Run a single-prompt baseline on the 100-note validation sample.

The script reads API credentials from environment variables. Outputs are saved
incrementally so the run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "validation" / "route" / "route_validation_sample_notes.csv"
DEFAULT_JSONL = ROOT / "results" / "validation" / "single_prompt_outputs.jsonl"
DEFAULT_CSV = ROOT / "results" / "validation" / "single_prompt_outputs.csv"
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.zetatechs.com/v1")

THEMES = ["Commercial & Tourism", "Historical & Art", "Nature", "Local Life"]


def read_api_key() -> str:
    for name in ("ZETA_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    raise RuntimeError("Set ZETA_API_KEY or OPENAI_API_KEY before running inference.")


def post_chat_completion(api_key: str, model: str, messages: list[dict[str, str]], temperature: float, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{DEFAULT_BASE_URL}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    return parsed["choices"][0]["message"]["content"]


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Baseline response was not a JSON object.")
    return obj


def normalize_theme(value: Any) -> str:
    text = str(value or "").strip()
    compact = re.sub(r"[\s_&/\\-]+", "", text.lower())
    if "commercial" in compact or "tourism" in compact or "shop" in compact or "food" in compact:
        return "Commercial & Tourism"
    if "historical" in compact or "history" in compact or "art" in compact or "heritage" in compact:
        return "Historical & Art"
    if "nature" in compact or "coast" in compact or "park" in compact or "beach" in compact:
        return "Nature"
    if "local" in compact or "life" in compact or "neighborhood" in compact or "community" in compact:
        return "Local Life"
    return text


def normalize_pad(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "+1", "+", "positive", "high", "p+", "a+", "d+"}:
        return "1"
    if text in {"-1", "-", "negative", "low", "p-", "a-", "d-"}:
        return "-1"
    if text in {"0", "neutral", "none", "na", "n/a", "not applicable", ""}:
        return "0"
    try:
        number = float(text)
        if number > 0:
            return "1"
        if number < 0:
            return "-1"
        return "0"
    except ValueError:
        return text


SYSTEM_PROMPT = """You are a baseline information-extraction model for a Citywalk study.

Read one Xiaohongshu Citywalk note and complete all tasks in one pass.

Allowed route themes:
- Commercial & Tourism: food, cafes, shopping, iconic check-in spots, tourist consumption.
- Historical & Art: heritage buildings, old streets, temples, museums, galleries, artistic/cultural spaces.
- Nature: coast, sea, beaches, parks, mountains, gardens, scenic outdoor walking.
- Local Life: residential alleys, markets, everyday neighborhood spaces, ordinary local routines.

PAD coding:
- Pleasure: 1 enjoyable/beautiful/delicious/healing; 0 neutral or unclear; -1 disappointing/unpleasant.
- Arousal: 1 lively/crowded/noisy/exciting; 0 neutral or unclear; -1 quiet/slow/relaxed.
- Dominance: 1 free/easy/autonomous; 0 neutral or unclear; -1 constrained/crowded/difficult/forced.

Rules:
- Extract only places or place-like stops grounded in the text. Keep the route order implied by the note.
- Do not invent places outside the note.
- If the note gives a list or itinerary order, preserve that order.
- Choose one dominant route theme.
- Assign PAD to each extracted POI from local text evidence when possible; otherwise use route-level affective cues only when clearly applicable.
- Return strict JSON only.

Required JSON schema:
{
  "route_theme": "Commercial & Tourism | Historical & Art | Nature | Local Life",
  "route_sequence_available": true,
  "pois": [
    {
      "location_index": 1,
      "location_POI": "place name",
      "text_evidence": "short source phrase",
      "P_score": 1,
      "A_score": 0,
      "D_score": 1
    }
  ]
}
"""


def build_user_prompt(row: dict[str, str]) -> str:
    return (
        f"Note index: {row.get('note_index', '')}\n"
        f"Title: {row.get('title', '')}\n"
        f"Body: {row.get('content', '')}\n"
        "Please perform POI extraction, route-theme classification, and PAD coding in one response."
    )


def load_done(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            note_index = record.get("note_index")
            if note_index and not record.get("error"):
                done.add(note_index)
    return done


def flatten_to_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            output = record.get("output") or {}
            pois = output.get("pois") if isinstance(output, dict) else []
            if not isinstance(pois, list):
                pois = []
            if not pois:
                rows.append(
                    {
                        "note_index": record.get("note_index", ""),
                        "baseline_theme": normalize_theme(output.get("route_theme") if isinstance(output, dict) else ""),
                        "baseline_sequence_available": output.get("route_sequence_available", "") if isinstance(output, dict) else "",
                        "baseline_location_index": "",
                        "baseline_poi": "",
                        "baseline_text_evidence": "",
                        "baseline_P": "",
                        "baseline_A": "",
                        "baseline_D": "",
                        "error": record.get("error", ""),
                    }
                )
                continue
            for i, poi in enumerate(pois, 1):
                if not isinstance(poi, dict):
                    continue
                rows.append(
                    {
                        "note_index": record.get("note_index", ""),
                        "baseline_theme": normalize_theme(output.get("route_theme")),
                        "baseline_sequence_available": output.get("route_sequence_available", ""),
                        "baseline_location_index": poi.get("location_index", i),
                        "baseline_poi": poi.get("location_POI", ""),
                        "baseline_text_evidence": poi.get("text_evidence", ""),
                        "baseline_P": normalize_pad(poi.get("P_score", "")),
                        "baseline_A": normalize_pad(poi.get("A_score", "")),
                        "baseline_D": normalize_pad(poi.get("D_score", "")),
                        "error": record.get("error", ""),
                    }
                )
    fieldnames = [
        "note_index",
        "baseline_theme",
        "baseline_sequence_available",
        "baseline_location_index",
        "baseline_poi",
        "baseline_text_evidence",
        "baseline_P",
        "baseline_A",
        "baseline_D",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model", default="5.4mini")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    api_key = read_api_key()
    rows: list[dict[str, str]]
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    done = load_done(args.jsonl)
    to_run = [row for row in rows if row.get("note_index") not in done]
    if args.limit:
        to_run = to_run[: args.limit]

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baseline model: {args.model}")
    print(f"Remaining notes to process: {len(to_run)}")

    with args.jsonl.open("a", encoding="utf-8") as out:
        for idx, row in enumerate(to_run, 1):
            note_index = row.get("note_index", "")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ]
            record: dict[str, Any] = {
                "note_index": note_index,
                "model": args.model,
                "temperature": args.temperature,
                "output": None,
                "raw": "",
                "error": "",
            }
            try:
                raw = post_chat_completion(api_key, args.model, messages, args.temperature, args.timeout)
                record["raw"] = raw
                record["output"] = extract_json_object(raw)
            except Exception as exc:
                record["error"] = str(exc)
                print(f"[{idx}/{len(to_run)}] {note_index}: ERROR {str(exc)[:180]}")
            else:
                n_pois = len(record["output"].get("pois", []) or []) if isinstance(record["output"], dict) else 0
                print(f"[{idx}/{len(to_run)}] {note_index}: ok, {n_pois} POIs")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(args.sleep)

    flatten_to_csv(args.jsonl, args.csv)
    print(f"Wrote {args.jsonl}")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
