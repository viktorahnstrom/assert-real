"""
Unit tests for ``_attach_region_comments`` in ``app.routers.analyses``.

Coverage:
- Backwards compat — the legacy region_comments-only path still attaches
  ``explanation`` to evidence regions and leaves the new fields unset.
- Structured-regions path — when present, evidence_type / evidence_ref /
  claim_confidence are attached and ``explanation`` is overridden by the
  structured ``observation``.
- Label match is whitespace- and case-insensitive so minor casing drift
  between provider output and our region labels does not silently drop
  the tag.
"""

from __future__ import annotations

from app.routers.analyses import _attach_region_comments


def _evidence(label: str = "Skin Texture") -> dict:
    return {"url": "/x.png", "label": label, "activation_score": 0.6}


class TestAttachRegionComments:
    def test_legacy_only_attaches_explanation(self):
        out = _attach_region_comments(
            [_evidence()],
            {"Skin Texture": "Looks waxy."},
        )
        assert out[0]["explanation"] == "Looks waxy."
        assert "evidence_type" not in out[0]
        assert "evidence_ref" not in out[0]
        assert "claim_confidence" not in out[0]

    def test_missing_legacy_label_yields_none_explanation(self):
        out = _attach_region_comments([_evidence()], {})
        assert out[0]["explanation"] is None

    def test_structured_regions_attach_evidence_fields(self):
        structured = [
            {
                "region": "Skin Texture",
                "observation": "No visible pores across the cheeks.",
                "evidence_type": "metric",
                "evidence_ref": "sharpness_z=-3.8",
                "confidence": 0.85,
            }
        ]
        out = _attach_region_comments(
            [_evidence()],
            {"Skin Texture": "legacy text"},
            structured_regions=structured,
        )
        # observation overrides legacy text.
        assert out[0]["explanation"] == "No visible pores across the cheeks."
        assert out[0]["evidence_type"] == "metric"
        assert out[0]["evidence_ref"] == "sharpness_z=-3.8"
        assert out[0]["claim_confidence"] == 0.85

    def test_label_match_is_case_insensitive(self):
        structured = [
            {
                "region": "skin texture",
                "observation": "Different casing.",
                "evidence_type": "visual",
                "evidence_ref": "left cheek crop",
                "confidence": 0.5,
            }
        ]
        out = _attach_region_comments(
            [_evidence("Skin Texture")],
            {},
            structured_regions=structured,
        )
        assert out[0]["evidence_type"] == "visual"
        assert out[0]["evidence_ref"] == "left cheek crop"

    def test_unmatched_structured_region_does_not_break_others(self):
        """A structured record for a region we don't show should be ignored."""
        structured = [
            {
                "region": "Mouth & Teeth",
                "observation": "Different region.",
                "evidence_type": "metric",
                "evidence_ref": "ela_z=+2.9",
                "confidence": 0.9,
            }
        ]
        out = _attach_region_comments(
            [_evidence("Skin Texture")],
            {"Skin Texture": "legacy"},
            structured_regions=structured,
        )
        # Skin Texture gets the legacy explanation, no evidence fields.
        assert out[0]["explanation"] == "legacy"
        assert "evidence_type" not in out[0]

    def test_invalid_confidence_does_not_break_attachment(self):
        structured = [
            {
                "region": "Skin Texture",
                "observation": "Has observation.",
                "evidence_type": "visual",
                "evidence_ref": "left cheek",
                "confidence": "not-a-number",
            }
        ]
        out = _attach_region_comments([_evidence()], {}, structured_regions=structured)
        assert out[0]["explanation"] == "Has observation."
        assert out[0]["evidence_type"] == "visual"
        assert "claim_confidence" not in out[0]
