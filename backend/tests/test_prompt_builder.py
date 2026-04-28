"""
Unit tests for :mod:`app.services.vlm.prompt_builder` parsing.

Focused on ``_parse_region_comments`` — the function is the last stop
between VLM output and the API response, so any label mismatch here
silently drops per-region explanations in the UI.
"""

from __future__ import annotations

from unittest.mock import patch

from app.services.forensics.report import ForensicsReport, RegionForensics
from app.services.vlm.base import DetectionContext
from app.services.vlm.prompt_builder import (
    _parse_region_comments,
    build_explanation_prompt,
    build_forensic_evidence_block,
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


def _make_detection(
    forensics_report: ForensicsReport | None = None,
) -> DetectionContext:
    """Build a minimal DetectionContext for prompt-builder tests."""
    return DetectionContext(
        classification="fake",
        confidence=0.92,
        model_used="EfficientNet-B4",
        probabilities={"fake": 0.92, "real": 0.08},
        region_labels=[],
        region_categories=[],
        forensics_report=forensics_report,
    )


def _z_score_stub(region: str, metric: str, value: float) -> float:
    """Map raw metric values into z-scores deterministically for tests.

    The stub treats the raw value AS the z-score so each test case can
    construct a ``RegionForensics`` with the desired z directly.
    """
    return float(value)


class TestBuildForensicEvidenceBlock:
    def test_empty_when_no_report(self):
        assert build_forensic_evidence_block(_make_detection(None)) == ""

    def test_empty_when_report_has_no_regions(self):
        report = ForensicsReport(regions={}, image_size=(0, 0))
        assert build_forensic_evidence_block(_make_detection(report)) == ""

    def test_block_renders_notable_metrics(self):
        report = ForensicsReport(
            regions={
                "skin_texture": RegionForensics(
                    laplacian_variance=-3.8,
                    hf_energy=2.1,
                    ela_intensity=0.2,
                ),
                "mouth_teeth": RegionForensics(
                    laplacian_variance=0.1,
                    hf_energy=0.0,
                    ela_intensity=2.9,
                ),
                "eyes_pupils": RegionForensics(
                    laplacian_variance=-0.2,
                    hf_energy=0.3,
                    ela_intensity=-0.1,
                ),
            },
            image_size=(512, 512),
        )

        with patch(
            "app.services.forensics.z_score",
            side_effect=_z_score_stub,
        ):
            block = build_forensic_evidence_block(_make_detection(report))

        assert block.startswith("[FORENSIC EVIDENCE]")
        assert "Skin Texture" in block
        assert "sharpness z=-3.8 (unusually smooth)" in block
        assert "high-freq energy z=+2.1 (upsampling artifacts likely)" in block
        assert "Mouth & Teeth" in block
        assert "ELA peak 2.9σ above face mean" in block
        # Eyes were entirely within real-face range — should be summarised.
        assert "Eyes & Pupils — within real-face range across all metrics" in block

    def test_rows_sorted_by_max_abs_z(self):
        """The most anomalous region must appear first so the VLM sees it."""
        report = ForensicsReport(
            regions={
                "skin_texture": RegionForensics(
                    laplacian_variance=-1.2,
                    hf_energy=0.0,
                    ela_intensity=0.0,
                ),
                "mouth_teeth": RegionForensics(
                    laplacian_variance=0.0,
                    hf_energy=0.0,
                    ela_intensity=3.5,
                ),
            },
            image_size=(512, 512),
        )

        with patch(
            "app.services.forensics.z_score",
            side_effect=_z_score_stub,
        ):
            block = build_forensic_evidence_block(_make_detection(report))

        skin_idx = block.index("Skin Texture")
        mouth_idx = block.index("Mouth & Teeth")
        assert mouth_idx < skin_idx, "Most anomalous region should rank first"

    def test_block_returns_empty_when_distribution_missing(self):
        """Graceful fallback so unit envs without real_distribution.json work."""
        report = ForensicsReport(
            regions={
                "skin_texture": RegionForensics(
                    laplacian_variance=-3.0, hf_energy=0.0, ela_intensity=0.0
                ),
            },
            image_size=(512, 512),
        )

        def raises(*_a, **_k):
            raise RuntimeError("real_distribution.json not found")

        with patch("app.services.forensics.z_score", side_effect=raises):
            assert build_forensic_evidence_block(_make_detection(report)) == ""


class TestBuildExplanationPromptForensicWiring:
    def test_no_forensic_block_when_report_is_none(self):
        prompt = build_explanation_prompt(
            _make_detection(None),
            gradcam_available=True,
            region_count=2,
            ela_available=False,
        )
        assert "[FORENSIC EVIDENCE]" not in prompt

    def test_image_layout_lists_ela_when_available(self):
        prompt = build_explanation_prompt(
            _make_detection(None),
            gradcam_available=True,
            region_count=2,
            ela_available=True,
        )
        # Expect: Image 1 = original, Image 2 = GradCAM, Image 3 = ELA, Images 4, 5 = crops
        assert "Image 1: the original photo." in prompt
        assert "Image 2: the GradCAM heatmap overlay." in prompt
        assert "Image 3: the ELA" in prompt
        assert "Images 4, 5:" in prompt

    def test_image_layout_omits_ela_when_unavailable(self):
        prompt = build_explanation_prompt(
            _make_detection(None),
            gradcam_available=True,
            region_count=2,
            ela_available=False,
        )
        # Crops should follow GradCAM directly.
        assert "the ELA" not in prompt
        assert "Images 3, 4:" in prompt

    def test_forensic_block_appears_when_report_provided(self):
        report = ForensicsReport(
            regions={
                "skin_texture": RegionForensics(
                    laplacian_variance=-3.8, hf_energy=0.0, ela_intensity=0.0
                ),
            },
            image_size=(512, 512),
        )
        with patch(
            "app.services.forensics.z_score",
            side_effect=_z_score_stub,
        ):
            prompt = build_explanation_prompt(
                _make_detection(report),
                gradcam_available=True,
                region_count=0,
                ela_available=True,
            )
        assert "[FORENSIC EVIDENCE]" in prompt
        assert "sharpness z=-3.8" in prompt


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
