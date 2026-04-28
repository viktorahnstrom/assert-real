"""
Unit tests for the structured VLM output schema and validator.

Coverage:
- ``parse_structured_response`` — accepts dicts and JSON strings, rejects
  malformed inputs, and clamps soft errors instead of failing.
- ``structured_to_legacy`` — produces the flat shape consumed by today's
  callers.
- Provider schema adapters — return dicts in the shapes each SDK expects.
"""

from __future__ import annotations

import json

from app.services.vlm.prompt_builder import (
    parse_structured_response,
    structured_to_legacy,
)
from app.services.vlm.structured_schema import (
    EVIDENCE_TYPES,
    EXPLANATION_SCHEMA,
    REGION_SCHEMA,
    anthropic_tool_definition,
    gemini_response_schema,
    openai_response_format,
)


def _valid_payload() -> dict:
    return {
        "summary": "The face looks fake.",
        "detailed_analysis": "Skin is too smooth across the cheeks.",
        "technical_notes": "Sharpness z=-3.8 in skin region.",
        "regions": [
            {
                "region": "Skin Texture",
                "observation": "Skin across the cheeks has no visible pores.",
                "evidence_type": "metric",
                "evidence_ref": "sharpness_z=-3.8",
                "confidence": 0.85,
            }
        ],
    }


class TestParseStructuredResponse:
    def test_accepts_dict_payload(self):
        parsed = parse_structured_response(_valid_payload())
        assert parsed is not None
        assert parsed["summary"] == "The face looks fake."
        assert len(parsed["regions"]) == 1
        assert parsed["regions"][0]["region"] == "Skin Texture"

    def test_accepts_json_string(self):
        payload = json.dumps(_valid_payload())
        parsed = parse_structured_response(payload)
        assert parsed is not None
        assert parsed["regions"][0]["evidence_type"] == "metric"

    def test_rejects_none(self):
        assert parse_structured_response(None) is None

    def test_rejects_malformed_json(self):
        assert parse_structured_response("{ not json") is None

    def test_rejects_non_object_payload(self):
        assert parse_structured_response([1, 2, 3]) is None

    def test_rejects_missing_top_level_key(self):
        payload = _valid_payload()
        del payload["summary"]
        assert parse_structured_response(payload) is None

    def test_rejects_regions_not_list(self):
        payload = _valid_payload()
        payload["regions"] = {"not": "a list"}
        assert parse_structured_response(payload) is None

    def test_rejects_unknown_evidence_type(self):
        payload = _valid_payload()
        payload["regions"][0]["evidence_type"] = "guess"
        assert parse_structured_response(payload) is None

    def test_rejects_region_missing_field(self):
        payload = _valid_payload()
        del payload["regions"][0]["evidence_ref"]
        assert parse_structured_response(payload) is None

    def test_rejects_non_numeric_confidence(self):
        payload = _valid_payload()
        payload["regions"][0]["confidence"] = "high"
        assert parse_structured_response(payload) is None

    def test_clamps_out_of_range_confidence(self):
        """Soft failure: clamp rather than burn a retry."""
        payload = _valid_payload()
        payload["regions"][0]["confidence"] = 1.4
        parsed = parse_structured_response(payload)
        assert parsed is not None
        assert parsed["regions"][0]["confidence"] == 1.0

        payload["regions"][0]["confidence"] = -0.2
        parsed = parse_structured_response(payload)
        assert parsed is not None
        assert parsed["regions"][0]["confidence"] == 0.0

    def test_high_confidence_with_empty_ref_is_logged_not_failed(self):
        """The grounding rule is enforced upstream by the prompt — at parse
        time we keep the claim and let the frontend flag it."""
        payload = _valid_payload()
        payload["regions"][0]["evidence_ref"] = ""
        payload["regions"][0]["confidence"] = 0.9
        parsed = parse_structured_response(payload)
        assert parsed is not None
        assert parsed["regions"][0]["evidence_ref"] == ""
        assert parsed["regions"][0]["confidence"] == 0.9

    def test_normalises_markdown_region_label(self):
        """Reuses the existing label normaliser so providers that wrap
        labels in bold (`**Skin Texture**`) do not break attachment."""
        payload = _valid_payload()
        payload["regions"][0]["region"] = "**Skin Texture**"
        parsed = parse_structured_response(payload)
        assert parsed is not None
        assert parsed["regions"][0]["region"] == "Skin Texture"


class TestStructuredToLegacy:
    def test_round_trip(self):
        parsed = parse_structured_response(_valid_payload())
        legacy = structured_to_legacy(parsed)
        assert legacy["summary"] == "The face looks fake."
        assert legacy["region_comments"] == {
            "Skin Texture": "Skin across the cheeks has no visible pores."
        }


class TestProviderAdapters:
    def test_anthropic_tool_definition_shape(self):
        tool = anthropic_tool_definition()
        assert tool["name"] == "submit_explanation"
        assert tool["input_schema"] is EXPLANATION_SCHEMA

    def test_openai_response_format_strict_json_schema(self):
        fmt = openai_response_format()
        assert fmt["format"]["type"] == "json_schema"
        assert fmt["format"]["strict"] is True
        assert fmt["format"]["schema"]["properties"]["regions"]["items"] is REGION_SCHEMA

    def test_gemini_strips_additional_properties(self):
        """Gemini rejects ``additionalProperties`` so the adapter strips it."""
        schema = gemini_response_schema()

        def _walk(node):
            if isinstance(node, dict):
                assert "additionalProperties" not in node
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(schema)
        # Spot-check the structure is otherwise intact.
        assert schema["properties"]["regions"]["items"]["properties"]["evidence_type"][
            "enum"
        ] == list(EVIDENCE_TYPES)
