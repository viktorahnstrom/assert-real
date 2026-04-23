"""
Unit tests for :mod:`app.services.vlm.prompt_builder` parsing.

Focused on ``_parse_region_comments`` — the function is the last stop
between VLM output and the API response, so any label mismatch here
silently drops per-region explanations in the UI.
"""

from __future__ import annotations

from app.services.vlm.prompt_builder import (
    _parse_region_comments,
    parse_explanation_response,
)


class TestParseRegionComments:
    def test_plain_labels(self):
        text = "Skin Texture | Looks waxy.\nMouth & Teeth | Too uniform."
        result = _parse_region_comments(text)
        assert result == {
            "Skin Texture": "Looks waxy.",
            "Mouth & Teeth": "Too uniform.",
        }

    def test_markdown_bold_labels_stripped(self):
        """VLMs often wrap labels in ``**...**`` regardless of prompt style."""
        text = "**Skin Texture** | Looks waxy.\n**Mouth & Teeth** | Too uniform."
        result = _parse_region_comments(text)
        assert "Skin Texture" in result
        assert "Mouth & Teeth" in result
        assert "**Skin Texture**" not in result

    def test_italic_labels_stripped(self):
        text = "*Hair* | Too smooth.\n_Eyes_ | Glassy reflections."
        result = _parse_region_comments(text)
        assert result == {"Hair": "Too smooth.", "Eyes": "Glassy reflections."}

    def test_bullet_prefix_stripped(self):
        text = "- Skin Texture | Looks waxy.\n* Mouth | Too uniform."
        result = _parse_region_comments(text)
        assert result == {"Skin Texture": "Looks waxy.", "Mouth": "Too uniform."}

    def test_trailing_colon_stripped(self):
        text = "Skin Texture: | Looks waxy."
        result = _parse_region_comments(text)
        assert result == {"Skin Texture": "Looks waxy."}

    def test_empty_comment_dropped(self):
        text = "Skin Texture |   \nMouth & Teeth | Too uniform."
        result = _parse_region_comments(text)
        assert result == {"Mouth & Teeth": "Too uniform."}

    def test_lines_without_pipe_ignored(self):
        text = "Some intro text\nSkin Texture | Looks waxy.\nMore text"
        result = _parse_region_comments(text)
        assert result == {"Skin Texture": "Looks waxy."}


class TestParseExplanationResponse:
    def test_full_response_with_markdown_region_labels(self):
        """End-to-end parse of a real VLM response shape (Gemini-style bold)."""
        raw = (
            "[SUMMARY]\nThe face looks fake.\n\n"
            "[DETAILED]\nThe skin is waxy.\n\n"
            "[TECHNICAL]\nHigh confidence.\n\n"
            "[REGIONS]\n"
            "**Skin Texture** | Waxy and plastic-like.\n"
            "**Mouth & Teeth** | Uniform spacing looks unnatural."
        )
        result = parse_explanation_response(raw)
        assert result["summary"] == "The face looks fake."
        assert "waxy" in result["detailed_analysis"]
        assert result["region_comments"] == {
            "Skin Texture": "Waxy and plastic-like.",
            "Mouth & Teeth": "Uniform spacing looks unnatural.",
        }
