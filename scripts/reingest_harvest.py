#!/usr/bin/env python3
"""
Edge Harvest Re-Ingest Script
Apple Quality Recognition Engine

Scans ``dataset/edge_harvest/`` for harvested frames (low-confidence
detections and operator overrides) and promotes them into
``dataset/raw_ingest/`` for re-annotation in the next active-learning
cycle.

A manifest CSV is generated at ``dataset/raw_ingest/manifest_<timestamp>.csv``
with columns: timestamp, source_path, operator_override, frame_hash.

Usage:
    python scripts/reingest_harvest.py [--since 2026-07-01] [--limit 50]

Outputs:
    - Copied JPG files in dataset/raw_ingest/
    - Manifest CSV at dataset/raw_ingest/manifest_<timestamp>.csv
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from typing import List, Dict, Optional

# Allow running both as `python scripts/reingest_harvest.py` from the repo
# root and as a direct module invocation. Insert the repo root so that
# edge_harvest_schema is importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from edge_harvest_schema import validate_telemetry  # noqa: E402


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def _scan_harvest_dir(
    harvest_dir: str,
    since: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Scan the harvest directory for telemetry JSON + JPG pairs.

    Walks date-organized subdirectories (``<harvest_dir>/<YYYY-MM-DD>/``)
    and collects matching telemetry/frame pairs. When ``since`` is given,
    only date directories >= that date are included.

    Args:
        harvest_dir: Root directory for harvested frames.
        since: Optional date string (``YYYY-MM-DD``) to filter from.

    Returns:
        List of dicts with keys: ``date``, ``telemetry_path``,
        ``frame_path``, ``timestamp``, ``operator_override``,
        ``frame_hash``.
    """
    results: List[Dict[str, str]] = []

    if not os.path.isdir(harvest_dir):
        return results

    for entry in sorted(os.listdir(harvest_dir)):
        day_dir = os.path.join(harvest_dir, entry)
        if not os.path.isdir(day_dir):
            continue

        # Filter by date if --since was provided.
        if since and entry < since:
            continue

        # Match telemetry_*.json files to their frame_*.jpg counterparts.
        for fname in sorted(os.listdir(day_dir)):
            if not (fname.startswith("telemetry_") and fname.endswith(".json")):
                continue

            telemetry_path = os.path.join(day_dir, fname)
            frame_fname = fname.replace("telemetry_", "frame_").replace(".json", ".jpg")
            frame_path = os.path.join(day_dir, frame_fname)

            if not os.path.isfile(frame_path):
                continue

            # Load telemetry to extract metadata for the manifest.
            try:
                import json
                with open(telemetry_path, "r") as fh:
                    payload = json.load(fh)
                validate_telemetry(payload)
            except Exception as exc:
                print(f"[reingest] WARNING: skipping malformed telemetry "
                      f"{telemetry_path}: {exc}")
                continue

            results.append({
                "date": entry,
                "telemetry_path": telemetry_path,
                "frame_path": frame_path,
                "timestamp": payload.get("timestamp", ""),
                "operator_override": str(payload.get("operator_override", False)),
                "frame_hash": payload.get("frame_hash", ""),
            })

    return results


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote_frames(
    entries: List[Dict[str, str]],
    dest_dir: str,
    limit: Optional[int] = None,
) -> str:
    """Copy harvested frames into the raw_ingest directory and write a manifest.

    Args:
        entries: Output of :func:`_scan_harvest_dir`.
        dest_dir: Destination directory (e.g. ``dataset/raw_ingest``).
        limit: Maximum number of frames to promote. ``None`` = no limit.

    Returns:
        Path to the written manifest CSV file.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if limit is not None:
        entries = entries[:limit]

    manifest_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = os.path.join(dest_dir, f"manifest_{manifest_stamp}.csv")

    promoted = 0
    with open(manifest_path, "w", newline="") as csv_fh:
        writer = csv.DictWriter(
            csv_fh,
            fieldnames=["timestamp", "source_path", "operator_override", "frame_hash"],
        )
        writer.writeheader()

        for entry in entries:
            src = entry["frame_path"]
            dest_fname = os.path.basename(src)
            dest_path = os.path.join(dest_dir, dest_fname)

            # Avoid overwriting if a frame with the same name already exists.
            if os.path.isfile(dest_path):
                continue

            shutil.copy2(src, dest_path)
            writer.writerow({
                "timestamp": entry["timestamp"],
                "source_path": src,
                "operator_override": entry["operator_override"],
                "frame_hash": entry["frame_hash"],
            })
            promoted += 1

    print(f"[reingest] Promoted {promoted} frame(s) to {dest_dir}")
    print(f"[reingest] Manifest written to {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse CLI args, scan for harvested frames, and promote them.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Re-ingest edge-harvested frames into dataset/raw_ingest for re-annotation."
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only promote frames from this date onward (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of frames to promote.",
    )
    parser.add_argument(
        "--harvest-dir",
        default="dataset/edge_harvest",
        help="Root directory for harvested frames.",
    )
    parser.add_argument(
        "--dest-dir",
        default="dataset/raw_ingest",
        help="Destination directory for promoted frames.",
    )
    args = parser.parse_args()

    entries = _scan_harvest_dir(args.harvest_dir, since=args.since)

    if not entries:
        print("[reingest] No harvested frames found.")
        if args.since:
            print(f"[reingest] (filtered since {args.since})")
        return 0

    print(f"[reingest] Found {len(entries)} harvested frame(s).")
    promote_frames(entries, args.dest_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
