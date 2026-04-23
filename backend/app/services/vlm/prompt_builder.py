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

from app.services.vlm.base import DetectionContext

SYSTEM_PROMPT_WITH_GRADCAM = """You are a forensic image analyst explaining deepfake detection results \
to everyday users with no technical background.

You will receive images in this order:
- Image 1: The original photo that was analyzed
- Image 2: A GradCAM heatmap overlay. Warmer and brighter colors (red, orange, yellow) show \
which regions the AI detection model paid most attention to.
- Images 3, 4, 5 (if present): Zoomed-in crops of the specific facial regions, in order from \
highest to lowest activation. Study these carefully — they are the primary evidence.

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


def build_explanation_prompt(
    detection: DetectionContext,
    gradcam_available: bool = True,
    region_count: int = 0,
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

    # Build the image layout description so the VLM knows exactly which image is which
    crop_image_start = 3 if gradcam_available else 2
    if gradcam_available:
        if region_count > 0:
            crop_refs = ", ".join(f"Image {crop_image_start + i}" for i in range(region_count))
            image_instruction = (
                f"Look at the GradCAM heatmap (Image 2) to see which areas the model focused on, "
                f"then examine the zoomed-in region crops ({crop_refs}) to inspect each area up close. "
                f"Describe the specific visual details you actually see in the crops — texture, edges, "
                f"blending, skin quality — that support or contradict this classification."
            )
        else:
            image_instruction = (
                "Look at the GradCAM heatmap (Image 2) and describe what you observe "
                "in the highlighted regions of the original photo (Image 1). "
                "What specific visual details in those regions support this classification?"
            )
    else:
        image_instruction = (
            "The GradCAM heatmap is not available for this analysis. "
            "Look at the original photo and describe what visual features "
            "you can directly observe that support or contradict this classification."
        )

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

    prompt = f"""Here are the detection results for the image.

Detection result: {classification} ({confidence_pct} confidence)
Fake probability: {fake_prob:.1%} | Real probability: {real_prob:.1%}
Model: {detection.model_used}

{image_instruction}{regions_instruction}

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
