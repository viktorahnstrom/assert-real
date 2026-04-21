"""
Rule-Based Template Explanation Provider

Generates structured, data-driven explanations from detection metadata alone —
no API calls, no image inspection, zero cost.

The key design principle is specificity: every sentence references a concrete
number (activation score, confidence, region rank) from the actual detection
result. Generic artifact lists are avoided; instead, one targeted artifact is
selected per region based on the category and the relative activation level.

Intended use:
- User-study control condition: compare against VLM explanations to measure
  whether visual AI reasoning adds value beyond structured numeric analysis.
- Fallback when no VLM API key is configured.
- Offline / cost-free operation.

Research note: this provider is intentionally *different* from VLM, not a
worse imitation. It offers transparent, reproducible, number-grounded
reasoning. A VLM offers visually-grounded but opaque inference.
"""

import time

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation

# ---------------------------------------------------------------------------
# Activation pattern inference
# ---------------------------------------------------------------------------

def _infer_manipulation_type(regions: list) -> str:
    """
    Infer the likely manipulation type from the activation pattern across regions.

    - Single dominant region (top activation >> rest) → localised edit / face-swap
    - Multiple high-activation regions (top ~= second) → full-face generation (GAN/diffusion)
    - No clear regions → ambiguous
    """
    if len(regions) == 0:
        return "ambiguous"

    scores = sorted([r.activation_score for r in regions], reverse=True)

    if len(scores) == 1:
        return "localised"

    top, second = scores[0], scores[1]
    ratio = top / second if second > 0 else float("inf")

    if ratio >= 2.0:
        return "localised"   # one region dominates strongly
    if top >= 0.55 and second >= 0.45:
        return "distributed"  # multiple strong regions → generative artefacts
    return "ambiguous"


# ---------------------------------------------------------------------------
# Per-region artifact selection
# ---------------------------------------------------------------------------

# For each category, pick the single most discriminative artifact to mention.
# Order matters: the first entry is used when confidence is high, the second
# when confidence is moderate, so the language stays specific.
_CATEGORY_PRIMARY_ARTIFACT: dict[str, list[str]] = {
    "eyes_pupils": [
        "asymmetric catchlight positioning",
        "unnatural iris texture smoothness",
    ],
    "skin_texture": [
        "pore-pattern discontinuity at the region boundary",
        "overly uniform skin-tone distribution",
    ],
    "hair_hairline": [
        "unnaturally smooth hairline-to-forehead transition",
        "absence of fine stray hairs along the hairline",
    ],
    "nose_midface": [
        "subtle geometric distortion around the nasal bridge",
        "mid-face blending seam",
    ],
    "mouth_lips": [
        "tooth-boundary sharpness inconsistency",
        "lip-corner blending artefact",
    ],
    "jawline_chin": [
        "face-to-neck boundary blending seam",
        "jawline contour smoothing typical of face-swap composites",
    ],
}


def _pick_artifact(category_id: str, confidence: float) -> str | None:
    options = _CATEGORY_PRIMARY_ARTIFACT.get(category_id)
    if not options:
        return None
    return options[0] if confidence >= 0.80 else options[1] if len(options) > 1 else options[0]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(detection: DetectionContext, top_region=None) -> str:
    pct = round(detection.confidence * 100)
    is_fake = detection.classification == "fake"

    if is_fake:
        if detection.confidence >= 0.85:
            opening = (
                f"The detection model classified this image as a deepfake with {pct}% confidence."
            )
        elif detection.confidence >= 0.65:
            opening = (
                f"The detection model flagged this image as likely manipulated ({pct}% confidence)."
            )
        else:
            opening = (
                f"The detection model returned a weak deepfake signal ({pct}% confidence). "
                f"The result is uncertain."
            )
    else:
        if detection.confidence >= 0.85:
            opening = (
                f"The detection model classified this image as authentic with {pct}% confidence."
            )
        elif detection.confidence >= 0.65:
            opening = f"The detection model found no strong signs of manipulation ({pct}% confidence)."
        else:
            opening = (
                f"The detection model returned a weak authenticity signal ({pct}% confidence). "
                f"Minor editing or compression may be present."
            )

    if top_region is not None:
        region_name = getattr(top_region, "category_label", None) or getattr(
            top_region, "label", "an unidentified region"
        )
        act_pct = round(top_region.activation_score * 100)
        return f"{opening} The strongest signal came from the {region_name} area ({act_pct}% activation)."

    return opening


# ---------------------------------------------------------------------------
# Detailed analysis
# ---------------------------------------------------------------------------

def _build_detailed_analysis(detection: DetectionContext) -> str:
    is_fake = detection.classification == "fake"
    pct = round(detection.confidence * 100)
    paragraphs: list[str] = []

    regions = sorted(
        detection.region_categories or [], key=lambda r: r.activation_score, reverse=True
    )

    # --- Opening: confidence + pattern inference ---
    manipulation_type = _infer_manipulation_type(regions)

    if is_fake:
        if manipulation_type == "localised":
            pattern_note = (
                "The activation is concentrated in one dominant region, which is characteristic "
                "of localised face-swap edits rather than fully AI-generated faces."
            )
        elif manipulation_type == "distributed":
            pattern_note = (
                "High activation is spread across multiple distinct regions simultaneously, "
                "a pattern more typical of fully generative models (GANs or diffusion) "
                "than targeted local edits."
            )
        else:
            pattern_note = (
                "The activation pattern did not clearly indicate a single manipulation type."
            )

        paragraphs.append(
            f"The model assigned {pct}% probability to this image being manipulated. "
            f"{pattern_note}"
        )
    else:
        paragraphs.append(
            f"The model assigned {pct}% probability to this image being authentic. "
            f"The attention pattern showed no localised anomalies consistent with known "
            f"deepfake manipulation signatures."
        )

    # --- Per-region paragraphs ---
    if regions:
        top_score = regions[0].activation_score

        for i, region in enumerate(regions[:3]):
            act_pct = round(region.activation_score * 100)
            cat_label = region.category_label
            region_label = region.label
            artifact = _pick_artifact(region.category_id, detection.confidence)

            if i == 0:
                rank_phrase = "The highest-activation region"
            elif i == 1:
                ratio = round(top_score / region.activation_score, 1) if region.activation_score > 0 else "—"
                rank_phrase = f"The second-ranked region ({ratio}× lower activation than the top)"
            else:
                rank_phrase = "A third region"

            if is_fake:
                if artifact:
                    para = (
                        f"{rank_phrase} was the **{cat_label}** ({region_label}) "
                        f"with {act_pct}% activation. "
                        f"At this confidence level, the most likely indicator in this area is "
                        f"{artifact}."
                    )
                else:
                    para = (
                        f"{rank_phrase} was the **{cat_label}** ({region_label}) "
                        f"with {act_pct}% activation, suggesting anomalies in this area."
                    )
            else:
                if artifact:
                    para = (
                        f"{rank_phrase} was the **{cat_label}** ({region_label}) "
                        f"with {act_pct}% activation. "
                        f"No evidence of {artifact} was detected here."
                    )
                else:
                    para = (
                        f"{rank_phrase} was the **{cat_label}** ({region_label}) "
                        f"with {act_pct}% activation — within the expected range for authentic imagery."
                    )

            paragraphs.append(para)

    elif detection.region_labels:
        label_list = ", ".join(detection.region_labels[:3])
        if is_fake:
            paragraphs.append(
                f"The model's attention was distributed across: {label_list}. "
                f"These regions exhibited activation patterns inconsistent with unmanipulated photographs."
            )
        else:
            paragraphs.append(
                f"The model examined: {label_list}. "
                f"None showed activation patterns indicative of manipulation."
            )

    # --- Closing ---
    if is_fake:
        paragraphs.append(
            "This analysis is based on the model's internal spatial attention and reflects "
            "where in the image the detection signal is strongest. It does not constitute "
            "a pixel-level forensic inspection."
        )
    else:
        paragraphs.append(
            "The absence of localised high-activation anomalies supports the authenticity "
            "classification. Note that sophisticated manipulations targeting regions outside "
            "the model's attention may not be detected."
        )

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Per-region comments  (label → one-sentence explanation)
# ---------------------------------------------------------------------------

def _build_region_comments(detection: DetectionContext) -> dict[str, str]:
    """Return one focused sentence per region, keyed by region label.

    These are attached to evidence_regions[].explanation in the API response
    and displayed directly in the region cards on the frontend.
    """
    is_fake = detection.classification == "fake"
    regions = sorted(
        detection.region_categories or [], key=lambda r: r.activation_score, reverse=True
    )

    comments: dict[str, str] = {}
    for region in regions[:3]:
        act_pct = round(region.activation_score * 100)
        artifact = _pick_artifact(region.category_id, detection.confidence)

        if is_fake:
            if artifact:
                sentence = (
                    f"The model's attention ({act_pct}% activation) in this area is consistent "
                    f"with {artifact}, a known indicator in the {region.category_label} region."
                )
            else:
                sentence = (
                    f"This region drew {act_pct}% activation, suggesting the model detected "
                    f"anomalies in the {region.category_label} area."
                )
        else:
            if artifact:
                sentence = (
                    f"This region showed {act_pct}% activation and appeared natural — "
                    f"no evidence of {artifact} was detected in the {region.category_label} area."
                )
            else:
                sentence = (
                    f"With {act_pct}% activation, this {region.category_label} region "
                    f"showed no anomalies consistent with manipulation."
                )

        comments[region.label] = sentence

    return comments


# ---------------------------------------------------------------------------
# Technical notes
# ---------------------------------------------------------------------------

def _build_technical_notes(detection: DetectionContext) -> str:
    fake_pct = round(detection.probabilities.get("fake", 0) * 100, 1)
    real_pct = round(detection.probabilities.get("real", 0) * 100, 1)
    log_odds = None
    p_fake = detection.probabilities.get("fake", 0)
    if 0 < p_fake < 1:
        import math
        log_odds = round(math.log(p_fake / (1 - p_fake)), 3)

    regions = sorted(
        detection.region_categories or [], key=lambda r: r.activation_score, reverse=True
    )

    lines = [
        f"Architecture: {detection.model_used}",
        f"Softmax output — P(fake): {fake_pct}%, P(real): {real_pct}%"
        + (f"  |  log-odds: {log_odds:+.3f}" if log_odds is not None else ""),
    ]

    if regions:
        region_summary = "; ".join(
            f"{r.category_label} α={round(r.activation_score, 3)}" for r in regions[:3]
        )
        lines.append(f"GradCAM spatial attention (α, mean activation per region): {region_summary}")

        manipulation_type = _infer_manipulation_type(regions)
        pattern_descriptions = {
            "localised": "localised edit / face-swap composite (single-region dominance, α-ratio ≥ 2.0)",
            "distributed": "full-face generative synthesis (multi-region co-activation, GANs or diffusion)",
            "ambiguous": "ambiguous — insufficient spatial separation to classify manipulation modality",
        }
        lines.append(
            f"Inferred manipulation modality: {pattern_descriptions.get(manipulation_type, manipulation_type)}"
        )

        if len(regions) >= 2:
            ratio = round(regions[0].activation_score / regions[1].activation_score, 2) if regions[1].activation_score > 0 else float("inf")
            lines.append(
                f"Activation dominance ratio (rank-1 / rank-2): {ratio:.2f}"
            )

    lines.append(
        "Saliency method: Gradient-weighted Class Activation Mapping (GradCAM) "
        "on final convolutional feature map of EfficientNet-B4."
    )
    lines.append(
        "Explanation pipeline: deterministic rule-based inference — "
        "no vision-language model invoked. Output is fully reproducible given identical inputs."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class RuleBasedProvider(BaseVLMProvider):
    """
    Deterministic, data-driven explanation provider.

    Every statement is grounded in a concrete number from the detection result.
    Output is fully reproducible: same input → same explanation.
    """

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
        region_image_bytes: list[bytes] | None = None,
    ) -> VLMExplanation:
        start = time.time()

        regions = sorted(
            detection.region_categories or [], key=lambda r: r.activation_score, reverse=True
        )
        top_region = regions[0] if regions else None

        summary = _build_summary(detection, top_region)
        detailed = _build_detailed_analysis(detection)
        notes = _build_technical_notes(detection)
        region_comments = _build_region_comments(detection)

        return VLMExplanation(
            summary=summary,
            detailed_analysis=detailed,
            technical_notes=notes,
            region_comments=region_comments if region_comments else None,
            provider="rule_based",
            model="rule-based-v1",
            processing_time_ms=int((time.time() - start) * 1000),
            estimated_cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            raw_response=None,
        )

    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="rule_based",
            name="Rule-Based Templates",
            model="rule-based-v1",
            available=True,
            latency_estimate_ms=1,
            cost_per_1m_input_tokens=0.0,
            cost_per_1m_output_tokens=0.0,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0
