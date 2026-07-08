#!/usr/bin/env python3
"""
Daily Operator Override Summary Report Generator
Apple Quality Recognition Engine

Loads all operator grading disagreements for a given date and prints a
summary: total overrides, breakdown by detected grade (G1/G2/G3/DISCARD),
and the most common disagreement patterns.

Usage:
    python scripts/override_report.py [--date 2026-07-08] [--markdown]

Outputs:
    - Console summary (always).
    - Markdown report at reports/override_report_<date>.md (with --markdown).
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from typing import List, Tuple

# Allow running both as `python scripts/override_report.py` from the repo
# root and as a direct module invocation. Insert the repo root so that
# override_persistence is importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from override_persistence import load_overrides, OperatorOverride  # noqa: E402


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _grade_counts(overrides: List[OperatorOverride]) -> Counter:
    """Tally every grade emitted across all override records.

    Args:
        overrides: List of OperatorOverride records.

    Returns:
        collections.Counter mapping grade string -> occurrence count.
    """
    counts: Counter = Counter()
    for record in overrides:
        for apple in record.grading_result:
            grade = apple.get("grade", "UNKNOWN")
            counts[grade] += 1
    return counts


def _disagreement_patterns(
    overrides: List[OperatorOverride],
) -> List[Tuple[str, int]]:
    """Identify the most common disagreement defect patterns.

    A "pattern" is the sorted tuple of defect class names present on an
    overridden apple. This surfaces which defect combinations operators
    most frequently disagree with.

    Args:
        overrides: List of OperatorOverride records.

    Returns:
        List of (pattern_string, count) tuples sorted by count descending.
    """
    patterns: Counter = Counter()
    for record in overrides:
        for apple in record.grading_result:
            defects = apple.get("defects", [])
            names = sorted(
                d.get("name", d.get("class_name", "unknown"))
                for d in defects
            ) if isinstance(defects, list) else []
            pattern_str = ", ".join(names) if names else "(no defects)"
            patterns[pattern_str] += 1
    return patterns.most_common()


def build_report_text(
    date_str: str,
    overrides: List[OperatorOverride],
) -> str:
    """Build the human-readable summary report as plain text.

    Args:
        date_str: The date the report covers (YYYY-MM-DD).
        overrides: List of OperatorOverride records for that date.

    Returns:
        Multi-line plain-text report string.
    """
    total = len(overrides)
    grade_counts = _grade_counts(overrides)
    patterns = _disagreement_patterns(overrides)

    facility_id = overrides[0].facility_id if overrides else "N/A"

    lines: List[str] = []
    lines.append("=" * 64)
    lines.append(f"  OPERATOR OVERRIDE DAILY REPORT — {date_str}")
    lines.append(f"  Facility: {facility_id}")
    lines.append("=" * 64)
    lines.append("")
    lines.append(f"Total override events: {total}")
    lines.append("")

    lines.append("Grade breakdown (computed grades overridden):")
    if grade_counts:
        for grade in ("G1", "G2", "G3", "DISCARD"):
            count = grade_counts.get(grade, 0)
            lines.append(f"  {grade:<10} {count}")
        # Include any unexpected grades.
        for grade, count in grade_counts.items():
            if grade not in ("G1", "G2", "G3", "DISCARD"):
                lines.append(f"  {grade:<10} {count}")
    else:
        lines.append("  (no graded apples recorded)")
    lines.append("")

    lines.append("Most common disagreement patterns (defects on overridden apples):")
    if patterns:
        for pattern, count in patterns[:10]:
            lines.append(f"  [{count:>3}x]  {pattern}")
    else:
        lines.append("  (no disagreement patterns recorded)")
    lines.append("")
    lines.append("=" * 64)
    return "\n".join(lines)


def build_report_markdown(
    date_str: str,
    overrides: List[OperatorOverride],
) -> str:
    """Build the summary report as a markdown document.

    Args:
        date_str: The date the report covers (YYYY-MM-DD).
        overrides: List of OperatorOverride records for that date.

    Returns:
        Markdown-formatted report string.
    """
    total = len(overrides)
    grade_counts = _grade_counts(overrides)
    patterns = _disagreement_patterns(overrides)
    facility_id = overrides[0].facility_id if overrides else "N/A"

    lines: List[str] = []
    lines.append(f"# Operator Override Daily Report — {date_str}")
    lines.append("")
    lines.append(f"**Facility:** `{facility_id}`  ")
    lines.append(f"**Total override events:** {total}")
    lines.append("")

    lines.append("## Grade Breakdown")
    lines.append("")
    lines.append("| Grade | Count |")
    lines.append("|-------|-------|")
    if grade_counts:
        for grade in ("G1", "G2", "G3", "DISCARD"):
            count = grade_counts.get(grade, 0)
            lines.append(f"| {grade} | {count} |")
        for grade, count in grade_counts.items():
            if grade not in ("G1", "G2", "G3", "DISCARD"):
                lines.append(f"| {grade} | {count} |")
    else:
        lines.append("| _none_ | 0 |")
    lines.append("")

    lines.append("## Most Common Disagreement Patterns")
    lines.append("")
    if patterns:
        lines.append("| Count | Defect Pattern |")
        lines.append("|-------|----------------|")
        for pattern, count in patterns[:10]:
            lines.append(f"| {count} | {pattern} |")
    else:
        lines.append("_No disagreement patterns recorded._")
    lines.append("")

    return "\n".join(lines)


def write_markdown_report(
    date_str: str,
    overrides: List[OperatorOverride],
    reports_dir: str = "reports",
) -> str:
    """Write the markdown report to ``reports/override_report_<date>.md``.

    Args:
        date_str: The date the report covers (YYYY-MM-DD).
        overrides: List of OperatorOverride records for that date.
        reports_dir: Directory to write the report into.

    Returns:
        The path to the written markdown file.
    """
    os.makedirs(reports_dir, exist_ok=True)
    md_path = os.path.join(reports_dir, f"override_report_{date_str}.md")
    markdown = build_report_markdown(date_str, overrides)
    with open(md_path, "w") as fh:
        fh.write(markdown)
    return md_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse CLI args, load overrides, and emit the daily summary report.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Daily operator override summary report generator."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to report (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write the report to reports/override_report_<date>.md",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/operator_overrides",
        help="Root directory for override storage.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for markdown report output.",
    )
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    overrides = load_overrides(date_str=date_str, output_dir=args.output_dir)

    report_text = build_report_text(date_str, overrides)
    print(report_text)

    if args.markdown:
        md_path = write_markdown_report(
            date_str, overrides, reports_dir=args.reports_dir
        )
        print(f"\n[REPORT] Markdown written to: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
