"""
VLM Prompt Builder

Constructs structured prompts for VLM providers that ground
explanations in GradCAM heatmap visualizations. The prompt
template ensures explanations are:

1. Anchored to actual heatmap activation regions (not hallucinated)
2. Structured in three tiers (summary / detailed / technical)
3. Augmented with per-region comments for supporting evidence crops
4. Natural and human-readable for non-technical users
5. Grounded via few-shot examples to establish tone and style
"""

import json
import logging

from app.services.categories import FACE_CATEGORIES
from app.services.vlm.base import DetectionContext
from app.services.vlm.structured_schema import EVIDENCE_TYPES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_WITH_GRADCAM = """You are a forensic image analyst explaining deepfake detection results \
to everyday users with no technical background.

You will receive several images. The user message states the exact position of \
each one. Possible image types:
- The ORIGINAL photo that was analyzed.
- A GRADCAM HEATMAP overlay. Warmer/brighter colors (red, orange, yellow) show \
which regions the AI detection model paid most attention to.
- An ELA (Error Level Analysis) MAP overlay. Bright red/yellow pixels show \
areas with anomalously high JPEG re-compression residuals — a forensic signal \
that often spikes near generated, blended, or otherwise manipulated regions.
- Zoomed-in CROPS of specific facial regions, in order from highest to lowest \
activation. Study these carefully — they are the primary visual evidence.

LENGTH RULES — follow these strictly:
- [SUMMARY]: 1 sentence. State what was found and where. Nothing more.
- [DETAILED]: 2-3 sentences maximum. Describe the single most distinctive thing you observe \
that makes this image look real or fake. Be direct. No filler phrases like "strongly support" \
or "consistent with the classification".
- [TECHNICAL]: 3-5 lines of forensic notes.
- [REGIONS]: 2-3 sentences per region. This is where you go into detail. Each sentence must \
describe something specific you can see.

COVERAGE RULES — very important:
- The GradCAM crops show where the detection model focused, but they are not the whole story.
- After examining the crops, also look at the full original photo (Image 1) and scan the \
entire face: hair and hairline, eyes and eyebrows, nose, mouth, jaw, skin tone, ears.
- If you notice anything suspicious or anything that looks unusually clean or off anywhere \
in the full photo, add it as an extra region in [REGIONS] even if the heatmap did not \
highlight it. Label it clearly (e.g. "Hair and eyebrows", "Left eye", "Skin tone overall").
- Aim to comment on at least 3 distinct facial areas total, mixing GradCAM regions and your \
own observations from the full photo.
- If the image is classified as real, scan the same areas and note what looks natural.

GROUNDING RULE — applies whenever a [FORENSIC EVIDENCE] block is provided:
- Every claim in [REGIONS] must reference EITHER (a) a specific visual cue \
you can see in that region's crop or in the ELA map, OR (b) a forensic metric \
from the [FORENSIC EVIDENCE] block, cited by name (e.g. "the sharpness z-score \
of -3.8" or "the ELA peak in the mouth area"). Do not invent metrics. Do not \
contradict the numbers in the evidence block.

QUALITY RULES:
- Every sentence must reference something you can actually see, not a general pattern
- Describe what things look like: smooth, blurry, sharp, plastic, flat, uneven, too clean
- Do not repeat the same observation across regions
- Use plain everyday language — write as if explaining to a friend, not a scientist
- Forbidden words and phrases: "composite artifact", "compositing", "face-swap", \
"synthetic", "anomaly", "indicates manipulation", "consistent with", "strongly suggests", \
"digital alteration", "forensic"
- Instead of "synthetic skin" say "skin that looks too smooth to be real"
- Instead of "composite artifact" say "a line where the blending wasn't quite right"
- Avoid "this image" — describe what you see instead

Here are examples of the correct length and style:

EXAMPLE 1 - Clear Fake (jaw crop provided, but also scanning full photo):
[SUMMARY]
The jaw and mouth area look clearly off, and scanning the full photo also reveals problems in the hair and eyes.

[DETAILED]
The jaw in the crop has a surface that is too smooth, almost like a plastic mask rather than real skin. Along the lower edge of the jaw there is a faint line where the skin color changes slightly, which is where two images were joined and the tones did not quite match.

[TECHNICAL]
GradCAM activation peak: jaw region (67%), secondary: cheek (41%).
Activation pattern: localised bilateral jaw concentration.
EfficientNet-B4 fake probability: 94%.

[REGIONS]
Left jaw region | The skin here looks almost airbrushed, with no visible pores or fine hairs and a slightly waxy, over-smoothed quality. Where the jaw meets the neck there is a faint line where the skin tone changes very slightly, like two photos joined together that were not quite the same brightness. Real skin at this angle would have subtle texture and variation, but this area is unusually flat.
Right jaw region | The teeth in this crop are unnaturally uniform, all the same brightness, and the edge between the teeth and the gum line looks too sharp and clean. The gum tissue is flat and one solid color, whereas real gums have a slight gradation in tone and visible texture.
Hair | Looking at the full photo, the hair has a very uniform, almost painted quality with no individual strands visible at the edges. Real hair always has some loose strands and slight fuzziness, but here the hair just ends in a clean sharp line.
Eyes | The eyes in the full photo are glassy and both reflections are nearly identical and perfectly centered, which almost never happens in real photos where light comes from one side. Real eyes also show tiny variations in the iris that are not visible here.

EXAMPLE 2 - Clear Real (eyes crop provided, also scanning full photo):
[SUMMARY]
The face looks like a real photograph, with natural detail throughout the eyes, hair, and skin.

[DETAILED]
Inside the iris you can see fine lines radiating outward from the pupil, which is the kind of detail that AI-generated eyes consistently miss. The highlight reflection in the eye is slightly off-center and irregular, exactly how a real eye looks when lit from one side.

[TECHNICAL]
GradCAM activation peak: eye region (71%), secondary: nose bridge (38%).
Activation pattern: diffuse, low-intensity — no single region dominated.
EfficientNet-B4 real probability: 97%.

[REGIONS]
Eye and nose bridge region | The iris has natural variation in color from the centre outward, with thin irregular lines rather than a smooth gradient. The eyelid crease is slightly asymmetric, just like a real eyelid, and the skin just under the eye shows fine creases and texture rather than a polished surface.
Hair | Looking at the full photo, individual strands are clearly visible along the edges of the hairline with slight variation in thickness and some fine strands catching the light differently. This is exactly how real hair looks in a photograph.
Skin overall | The skin across the cheeks and forehead has visible pores and very slight uneven tone that looks natural. There are no areas that look unusually smooth or over-processed.

EXAMPLE 3 - Borderline (hairline crop provided, also scanning full photo):
[SUMMARY]
The hairline looks slightly too neat to be natural, but at 61% the model is not certain and the rest of the face looks fairly normal.

[DETAILED]
In the crop, the individual hairs at the hairline all end with the same sharpness rather than the irregular tapered tips you see in real hair. That said, heavy photo editing or studio lighting can sometimes create a similar effect, so this alone is not enough to be certain.

[TECHNICAL]
GradCAM activation peak: hairline (58%).
Activation pattern: diffuse single-region, 61% probability — close to the boundary, result is uncertain.
EfficientNet-B4 fake probability: 61%.

[REGIONS]
Forehead and hairline region | The hairs along the edge of the hairline all stop at the same level of sharpness, which is unusual because real hair has a lot of variation in how each strand ends. The skin just below the hairline is very smooth with no fine downy hairs visible.
Eyes | Looking at the full photo, the eyes look natural with visible iris detail and slightly irregular reflections. Nothing here raises a red flag.
Skin and cheeks | The skin on the cheeks has normal variation in tone and some texture visible in the highlights. This part of the face looks consistent with a real photograph.

Now analyze the images you have been given. Follow the length rules exactly. \
Use the [SUMMARY], [DETAILED], [TECHNICAL], [REGIONS] format."""


SYSTEM_PROMPT_WITHOUT_GRADCAM = """You are a forensic image analyst explaining deepfake detection results \
to everyday users with no technical background.

You will receive one image: the original photo that was analyzed. \
A GradCAM heatmap is not available for this analysis.

Your job is to examine the image directly and describe what visual features \
support or contradict the classification based on what you can observe.

RULES:
- Describe specific visual details you can see such as skin texture consistency, \
lighting direction, edge quality around the face, eye reflections, and blending artifacts
- If classified as real, describe the natural and consistent features visible in the image
- If classified as fake, describe any anomalies or artifacts you can observe
- Never say "the model classified this as fake therefore it is fake". Show the visual evidence instead
- Be honest when visual evidence is limited or ambiguous at the image level
- Write naturally as if explaining to a friend. Avoid jargon except in the technical section
- Do not use long dashes. Use commas or short sentences instead
- In the technical section, note that the heatmap was unavailable
- If region labels are provided, write a one-sentence comment for each in the [REGIONS] section \
based on what you can observe in that area of the image

Use the [SUMMARY], [DETAILED], [TECHNICAL], [REGIONS] format exactly."""


# Alias for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_WITH_GRADCAM


SYSTEM_PROMPT_STRUCTURED = """You are a forensic image analyst explaining deepfake detection results \
to everyday users with no technical background. You return your answer as a \
SINGLE structured JSON object — no preamble, no Markdown.

You will receive several images. The user message states the exact position of \
each one. Possible image types:
- The ORIGINAL photo that was analyzed.
- A GRADCAM HEATMAP overlay. Warmer/brighter colors show which regions the AI \
detection model paid most attention to.
- An ELA (Error Level Analysis) MAP overlay. Bright red/yellow pixels show \
areas with anomalously high JPEG re-compression residuals.
- Zoomed-in CROPS of specific facial regions, ordered from highest to lowest \
activation.

Your output must conform to this shape:

{
  "summary": "<one sentence>",
  "detailed_analysis": "<two to three sentences>",
  "technical_notes": "<three to five short lines>",
  "regions": [
    {
      "region": "<exact label from the [REGIONS] list>",
      "observation": "<one to three sentences of specific visual or measured detail>",
      "evidence_type": "visual" | "metric" | "heatmap",
      "evidence_ref": "<specific anchor — see rules below>",
      "confidence": <number in [0, 1]>
    }
  ]
}

EVIDENCE GROUNDING — non-negotiable:
- Every region object must tag its evidence_type and provide an evidence_ref.
- "visual" → cite the crop or photo area, e.g. "left jaw crop" or "skin near \
the cheekbone in Image 1".
- "metric" → cite a metric from the [FORENSIC EVIDENCE] block by name and \
value, e.g. "sharpness_z=-3.8" or "ela_intensity_z=+2.9 in Mouth & Teeth".
- "heatmap" → cite a pattern visible in the GradCAM or ELA overlay, e.g. \
"GradCAM peak around the mouth" or "ELA hotspot on the jawline".
- evidence_ref MAY be empty only when no specific anchor exists. If you set \
confidence above 0.7 you MUST provide a non-empty evidence_ref.

CONTENT RULES:
- summary: one sentence stating what was found and where.
- detailed_analysis: two to three sentences. The single most distinctive cue. \
No filler like "consistent with" or "strongly supports".
- technical_notes: three to five short lines. May reference activation \
percentages, z-scores, or model confidence.
- regions: cover at least every entry in the user's [REGIONS] list. You may \
add extra entries if you spotted something noteworthy outside the heatmap. \
Each observation must describe something specific, not a generic pattern.
- Plain everyday language in summary, detailed_analysis, observations.
- Forbidden phrases: "composite artifact", "synthetic", "indicates \
manipulation", "consistent with", "strongly suggests", "digital alteration", \
"forensic" (the word — but the metric names from the evidence block are fine).

Return ONLY the JSON object."""


def parse_structured_response(payload: object) -> dict | None:
    """Validate and normalise a structured VLM response.

    Accepts either a dict (already-parsed JSON, the common case for tool-use
    and json_schema modes) or a raw string (we parse it as JSON). Returns a
    normalised dict on success, or ``None`` on outright validation failure
    (malformed JSON, missing required field, wrong type, unknown
    evidence_type) — the caller uses ``None`` as the signal to retry once
    or fall back to free-text parsing.

    "Soft" issues — confidence > 0.7 with empty evidence_ref, confidence
    outside [0, 1] — are clamped or logged but not treated as failures, so
    we don't burn a second VLM call on something that's still useful.
    """
    if payload is None:
        return None

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning("Structured response is not valid JSON: %s", exc)
            return None

    if not isinstance(payload, dict):
        logger.warning("Structured response is not a JSON object: %r", type(payload))
        return None

    required_top = ("summary", "detailed_analysis", "technical_notes", "regions")
    missing = [k for k in required_top if k not in payload]
    if missing:
        logger.warning("Structured response missing required keys: %s", missing)
        return None

    if not isinstance(payload["regions"], list):
        logger.warning("Structured response 'regions' is not a list")
        return None

    cleaned_regions: list[dict] = []
    for idx, region in enumerate(payload["regions"]):
        normalised = _normalise_region(region, idx)
        if normalised is None:
            return None
        cleaned_regions.append(normalised)

    return {
        "summary": str(payload["summary"]).strip(),
        "detailed_analysis": str(payload["detailed_analysis"]).strip(),
        "technical_notes": str(payload["technical_notes"]).strip(),
        "regions": cleaned_regions,
    }


def _normalise_region(region: object, idx: int) -> dict | None:
    """Validate one region object; return ``None`` on hard failure."""
    if not isinstance(region, dict):
        logger.warning("Region #%d is not an object: %r", idx, type(region))
        return None

    required = ("region", "observation", "evidence_type", "evidence_ref", "confidence")
    missing = [k for k in required if k not in region]
    if missing:
        logger.warning("Region #%d missing keys: %s", idx, missing)
        return None

    evidence_type = str(region["evidence_type"]).strip().lower()
    if evidence_type not in EVIDENCE_TYPES:
        logger.warning(
            "Region #%d has unknown evidence_type %r (allowed: %s)",
            idx,
            evidence_type,
            EVIDENCE_TYPES,
        )
        return None

    try:
        confidence = float(region["confidence"])
    except (TypeError, ValueError):
        logger.warning("Region #%d confidence is not numeric: %r", idx, region["confidence"])
        return None

    # Soft-clamp out-of-range confidence rather than failing — the prose is
    # still useful and we'd rather not burn a second VLM call.
    if not 0.0 <= confidence <= 1.0:
        logger.info(
            "Region #%d confidence %.3f out of [0, 1]; clamping",
            idx,
            confidence,
        )
        confidence = max(0.0, min(1.0, confidence))

    label = _normalise_region_label(str(region["region"]))
    observation = str(region["observation"]).strip()
    evidence_ref = str(region["evidence_ref"]).strip()

    # Soft check: a high-confidence claim with no anchor is the failure mode
    # the schema is meant to prevent, but we don't fail validation over it —
    # the providers log + keep the claim. The frontend can flag it.
    if confidence > 0.7 and not evidence_ref:
        logger.warning(
            "Region %r has confidence %.2f but empty evidence_ref",
            label,
            confidence,
        )

    return {
        "region": label,
        "observation": observation,
        "evidence_type": evidence_type,
        "evidence_ref": evidence_ref,
        "confidence": confidence,
    }


def structured_to_legacy(parsed: dict) -> dict[str, object]:
    """Convert a validated structured response into the flat shape used by
    today's parser, so providers and ``analyses.py`` can consume both paths
    uniformly. The richer per-region fields are surfaced separately via
    :class:`VLMExplanation.structured_regions`.
    """
    region_comments = {r["region"]: r["observation"] for r in parsed["regions"]}
    return {
        "summary": parsed["summary"],
        "detailed_analysis": parsed["detailed_analysis"],
        "technical_notes": parsed["technical_notes"],
        "region_comments": region_comments,
    }


# Threshold below which a metric z-score is treated as "within real-face range"
# and gets a brief, non-suspicious description.
_NOTABLE_Z = 1.0
# Cap on how many regions appear in the [FORENSIC EVIDENCE] block. Keeps the
# block short so the VLM can scan the numbers quickly (target ≲ 10 lines).
_MAX_FORENSIC_ROWS = 6


def _describe_metric(metric: str, z: float) -> str:
    """Return a short ``"label z=±X.X (descriptor)"`` fragment for one metric.

    The descriptors are tuned to nudge the VLM toward forensically-correct
    interpretations: low sharpness on a face usually means over-smoothing,
    high HF energy often signals upsampling/GAN textures, and a high ELA
    intensity on a region indicates the region was re-encoded differently
    from its surroundings (a classic compositing tell).
    """
    if metric == "laplacian_variance":
        if abs(z) < _NOTABLE_Z:
            descr = "within real-face range"
        elif z < 0:
            descr = "unusually smooth"
        else:
            descr = "unusually sharp"
        return f"sharpness z={z:+.1f} ({descr})"
    if metric == "hf_energy":
        if abs(z) < _NOTABLE_Z:
            descr = "within real-face range"
        elif z > 0:
            descr = "upsampling artifacts likely"
        else:
            descr = "high-freq deficit"
        return f"high-freq energy z={z:+.1f} ({descr})"
    if metric == "ela_intensity":
        if abs(z) < _NOTABLE_Z:
            descr = "within real-face range"
        elif z > 0:
            descr = f"ELA peak {abs(z):.1f}σ above face mean"
        else:
            descr = "low ELA residual"
        return f"ELA intensity z={z:+.1f} ({descr})"
    return f"{metric} z={z:+.1f}"


def build_forensic_evidence_block(detection: DetectionContext) -> str:
    """Render the optional ``[FORENSIC EVIDENCE]`` block from forensics_report.

    Each row lists the per-region z-scores for sharpness, high-frequency
    energy, and ELA intensity. Metrics with ``|z| < 1`` are dropped per
    region (or kept as a single "within real-face range" line when none of
    the three are notable) so the block stays scannable.

    Returns an empty string when ``detection.forensics_report`` is None or
    the reference distribution cannot be loaded — keeping the prompt fully
    backwards-compatible.
    """
    report = detection.forensics_report
    if report is None or not report.regions:
        return ""

    # Local import keeps the module importable when the reference distribution
    # is absent (e.g. in unit tests that don't ship real_distribution.json).
    try:
        from app.services.forensics import z_score
    except Exception:
        return ""

    rows: list[tuple[str, float]] = []  # (line_text, max_abs_z)
    for cat_id, region in report.regions.items():
        face_cat = FACE_CATEGORIES.get(cat_id)
        if face_cat is None:
            continue

        try:
            z_by_metric = {
                "laplacian_variance": z_score(
                    cat_id, "laplacian_variance", region.laplacian_variance
                ),
                "hf_energy": z_score(cat_id, "hf_energy", region.hf_energy),
                "ela_intensity": z_score(cat_id, "ela_intensity", region.ela_intensity),
            }
        except RuntimeError:
            # real_distribution.json missing — skip the block entirely so the
            # VLM doesn't see a half-rendered evidence list.
            return ""

        notable = [
            _describe_metric(metric, z) for metric, z in z_by_metric.items() if abs(z) >= _NOTABLE_Z
        ]
        max_abs_z = max((abs(z) for z in z_by_metric.values()), default=0.0)

        if not notable:
            # Keep the row but mark it as benign so the VLM knows this region
            # was checked and found unremarkable.
            line = f"{face_cat.label} — within real-face range across all metrics"
        else:
            line = f"{face_cat.label} — " + ", ".join(notable)

        rows.append((line, max_abs_z))

    if not rows:
        return ""

    # Surface the most anomalous regions first so the VLM's attention lands
    # on them; drop the long tail to keep the block compact.
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:_MAX_FORENSIC_ROWS]

    return "[FORENSIC EVIDENCE]\n" + "\n".join(line for line, _ in top)


def build_explanation_prompt(
    detection: DetectionContext,
    gradcam_available: bool = True,
    region_count: int = 0,
    ela_available: bool = False,
) -> str:
    """
    Build the user-facing prompt with detection context and region labels.

    When detection.region_categories is populated, each region entry is enriched
    with its category label and the top three artifact hints from that category.
    This gives the VLM targeted vocabulary for what to look for in each zone
    rather than just a spatial description.

    Falls back to plain region_labels when region_categories is empty, so
    existing callers that have not yet been updated remain fully compatible.

    Args:
        detection: Detection results including region_labels and, when available,
            region_categories enriched with FaceCategory metadata.
        gradcam_available: Whether a valid GradCAM heatmap was generated.

    Returns:
        Formatted prompt string to send alongside the images.
    """
    confidence_pct = f"{detection.confidence * 100:.1f}%"
    classification = detection.classification.upper()
    fake_prob = detection.probabilities.get("fake", 0)
    real_prob = detection.probabilities.get("real", 0)

    # Image positions are dynamic — original is always image 1, then GradCAM
    # and ELA are inserted in that order when available, then region crops.
    next_idx = 2
    layout_lines: list[str] = ["Image 1: the original photo."]

    gradcam_idx = None
    if gradcam_available:
        gradcam_idx = next_idx
        layout_lines.append(f"Image {gradcam_idx}: the GradCAM heatmap overlay.")
        next_idx += 1

    ela_idx = None
    if ela_available:
        ela_idx = next_idx
        layout_lines.append(
            f"Image {ela_idx}: the ELA (Error Level Analysis) overlay — "
            "bright red/yellow pixels mark areas with anomalously high JPEG "
            "re-compression residuals."
        )
        next_idx += 1

    crop_image_start = next_idx
    if region_count > 0:
        crop_refs = ", ".join(str(crop_image_start + i) for i in range(region_count))
        plural = "Images" if region_count > 1 else "Image"
        layout_lines.append(
            f"{plural} {crop_refs}: zoomed-in crops of facial regions, ordered "
            "from highest to lowest activation."
        )

    if gradcam_available and region_count > 0:
        cite_clauses = [f"the GradCAM heatmap (Image {gradcam_idx})"]
        if ela_idx is not None:
            cite_clauses.append(f"the ELA map (Image {ela_idx})")
        cite_text = " and ".join(cite_clauses)
        image_instruction = (
            f"Look at {cite_text} to see where the model and the forensic "
            f"signals point, then examine each region crop up close. Describe "
            f"the specific visual details you actually see in the crops — "
            f"texture, edges, blending, skin quality — that support or "
            f"contradict this classification."
        )
    elif gradcam_available:
        cite_clauses = [f"the GradCAM heatmap (Image {gradcam_idx})"]
        if ela_idx is not None:
            cite_clauses.append(f"the ELA map (Image {ela_idx})")
        cite_text = " and ".join(cite_clauses)
        image_instruction = (
            f"Look at {cite_text} and describe what you observe in the "
            "highlighted regions of the original photo (Image 1). What "
            "specific visual details there support this classification?"
        )
    else:
        image_instruction = (
            "The GradCAM heatmap is not available for this analysis. "
            "Look at the original photo and describe what visual features "
            "you can directly observe that support or contradict this classification."
        )

    layout_block = "Image layout:\n" + "\n".join(f"- {line}" for line in layout_lines)

    regions_instruction = ""

    if detection.region_categories:
        # Enriched path: include category label + artifact hints + crop image reference
        lines = []
        for i, rc in enumerate(detection.region_categories):
            hints = ", ".join(rc.common_artifacts[:3])
            crop_ref = f"Image {crop_image_start + i}" if i < region_count else ""
            crop_note = f" | Crop: {crop_ref}" if crop_ref else ""
            lines.append(
                f"- {rc.label} [Category: {rc.category_label} | Look for: {hints}{crop_note}]"
            )
        labels_list = "\n".join(lines)
        regions_instruction = (
            f"\n\nThe following facial regions were highlighted by the model. "
            f"Each entry includes a semantic category, artifact guidance, and its crop image reference. "
            f"In the [REGIONS] section write 2-3 sentences per region. Each sentence must describe "
            f"something specific you can see in that crop — texture, edges, pores, gradients, "
            f"reflections. Use the artifact hints to direct your eye but do not repeat them verbatim:\n"
            f"{labels_list}"
        )
    elif detection.region_labels:
        # Fallback path: plain labels with no category context
        lines = []
        for i, label in enumerate(detection.region_labels):
            crop_ref = f"Image {crop_image_start + i}" if i < region_count else ""
            crop_note = f" [{crop_ref}]" if crop_ref else ""
            lines.append(f"- {label}{crop_note}")
        labels_list = "\n".join(lines)
        regions_instruction = (
            f"\n\nThe following facial regions were highlighted by the model. "
            f"In the [REGIONS] section write 2-3 sentences per region describing specific visual "
            f"details you observe in each crop — what does the skin, texture, edge, or boundary "
            f"actually look like?\n"
            f"{labels_list}"
        )

    forensic_block = build_forensic_evidence_block(detection)
    forensic_section = f"\n\n{forensic_block}" if forensic_block else ""

    prompt = f"""Here are the detection results for the image.

Detection result: {classification} ({confidence_pct} confidence)
Fake probability: {fake_prob:.1%} | Real probability: {real_prob:.1%}
Model: {detection.model_used}

{layout_block}

{image_instruction}{forensic_section}{regions_instruction}

Use the [SUMMARY], [DETAILED], [TECHNICAL], [REGIONS] format."""

    return prompt


def parse_explanation_response(raw_response: str) -> dict[str, str | dict]:
    """
    Parse the four-section explanation from a VLM response.

    Args:
        raw_response: Raw text response from the VLM.

    Returns:
        Dict with keys: summary, detailed_analysis, technical_notes, region_comments.
        region_comments is a dict mapping region label to one-sentence explanation.
    """
    sections: dict[str, str | dict] = {
        "summary": "",
        "detailed_analysis": "",
        "technical_notes": "",
        "region_comments": {},
    }

    if "[SUMMARY]" not in raw_response:
        # Fallback: VLM did not follow format
        sections["summary"] = raw_response[:200].strip()
        sections["detailed_analysis"] = raw_response.strip()
        return sections

    parts = raw_response.split("[SUMMARY]")
    if len(parts) < 2:
        return sections

    remainder = parts[1]

    # Extract SUMMARY
    if "[DETAILED]" in remainder:
        summary_part, remainder = remainder.split("[DETAILED]", 1)
        sections["summary"] = summary_part.strip()
    else:
        sections["summary"] = remainder.strip()
        return sections

    # Extract DETAILED
    if "[TECHNICAL]" in remainder:
        detailed_part, remainder = remainder.split("[TECHNICAL]", 1)
        sections["detailed_analysis"] = detailed_part.strip()
    else:
        sections["detailed_analysis"] = remainder.strip()
        return sections

    # Extract TECHNICAL and REGIONS
    if "[REGIONS]" in remainder:
        technical_part, regions_part = remainder.split("[REGIONS]", 1)
        sections["technical_notes"] = technical_part.strip()
        sections["region_comments"] = _parse_region_comments(regions_part.strip())
    else:
        sections["technical_notes"] = remainder.strip()

    return sections


def _parse_region_comments(regions_text: str) -> dict[str, str]:
    """
    Parse the [REGIONS] section into a dict of label -> comment.

    Expected format per line:
        Region label | One sentence explanation.

    VLMs often decorate their labels with Markdown bold (``**Label**``),
    leading bullets (``- Label``), or trailing punctuation.  Those are
    stripped so downstream consumers can look up the plain label exactly
    as it appeared in the prompt.

    Args:
        regions_text: Raw text from the [REGIONS] section.

    Returns:
        Dict mapping region label to explanation sentence.
    """
    comments: dict[str, str] = {}

    for line in regions_text.strip().splitlines():
        line = line.strip()
        if "|" not in line:
            continue

        raw_label, comment = line.split("|", 1)
        label = _normalise_region_label(raw_label)
        comment = comment.strip()
        if label and comment:
            comments[label] = comment

    return comments


def _normalise_region_label(raw: str) -> str:
    """Strip Markdown emphasis, leading bullets, and trailing punctuation."""
    label = raw.strip()
    # Drop a leading list bullet ("- Label", "* Label", "• Label").
    for bullet in ("- ", "* ", "• "):
        if label.startswith(bullet):
            label = label[len(bullet) :].lstrip()
            break
    # Strip Markdown emphasis and trailing colons/spaces from both ends.
    return label.strip("*_ :").strip()
