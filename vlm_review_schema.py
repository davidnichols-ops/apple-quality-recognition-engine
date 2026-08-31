from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from grading_engine import VALID_GRADES

REVIEW_STATUSES = ("pending_human_review", "approved", "rejected")
DEFAULT_ADVISORY_MODEL = "gemini-3.7-flash"
_PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass(frozen=True)
class VLMReviewProposal:
    profile_id: str
    deterministic_grade: str
    policy_version: str
    suggested_grade: str
    confidence: float
    rationale: str
    model: str = DEFAULT_ADVISORY_MODEL
    proposed_policy_changes: Mapping[str, Any] = field(default_factory=dict)
    proposed_annotations: tuple[Mapping[str, Any], ...] = ()
    status: str = "pending_human_review"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviewed_by: str | None = None
    reviewed_at: str | None = None

    def validate(self) -> None:
        if not _PROFILE_ID_PATTERN.fullmatch(self.profile_id):
            raise ValueError("profile_id contains unsafe characters")
        if self.deterministic_grade not in VALID_GRADES:
            raise ValueError("deterministic_grade is invalid")
        if not self.policy_version.strip():
            raise ValueError("policy_version is required")
        if self.suggested_grade not in VALID_GRADES:
            raise ValueError("suggested_grade is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.rationale.strip():
            raise ValueError("rationale is required")
        if self.status not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(REVIEW_STATUSES)}")
        if self.status != "pending_human_review" and (
            not self.reviewed_by or not self.reviewed_at
        ):
            raise ValueError(
                "reviewed_by and reviewed_at are required for resolved proposals"
            )


def write_review_proposal(proposal: VLMReviewProposal, output_dir: str | Path) -> Path:
    proposal.validate()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = (
        destination
        / f"{proposal.profile_id}_{proposal.created_at.replace(':', '-')}.json"
    )
    path.write_text(
        json.dumps(asdict(proposal), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
