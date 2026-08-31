import json

import pytest

from vlm_review_schema import VLMReviewProposal, write_review_proposal


def proposal(**overrides) -> VLMReviewProposal:
    values = {
        "profile_id": "batch-a-00001",
        "deterministic_grade": "G2",
        "policy_version": "2.0.0-candidate",
        "suggested_grade": "G3",
        "confidence": 0.82,
        "rationale": "Possible missed defect on the calyx view.",
    }
    values.update(overrides)
    return VLMReviewProposal(**values)


def test_pending_proposal_is_valid_but_not_approved() -> None:
    candidate = proposal()
    candidate.validate()
    assert candidate.status == "pending_human_review"


def test_resolved_proposal_requires_human_reviewer() -> None:
    with pytest.raises(ValueError, match="reviewed_by"):
        proposal(status="approved").validate()


def test_invalid_grade_is_rejected() -> None:
    with pytest.raises(ValueError, match="suggested_grade"):
        proposal(suggested_grade="PREMIUM").validate()


def test_profile_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        proposal(profile_id="../../outside").validate()


def test_write_proposal_preserves_advisory_status(tmp_path) -> None:
    path = write_review_proposal(proposal(), tmp_path)
    payload = json.loads(path.read_text())
    assert payload["model"] == "gemini-3.7-flash"
    assert payload["status"] == "pending_human_review"
    assert payload["deterministic_grade"] == "G2"
