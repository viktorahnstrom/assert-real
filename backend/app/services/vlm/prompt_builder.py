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

You will receive two images:
- Image 1: The original photo that was analyzed
- Image 2: A GradCAM heatmap overlay. Warmer and brighter colors (red, orange, yellow) show \
which regions the AI model paid most attention to when making its decision.

Your job is to look at both images and describe what you actually observe in the highlighted \
regions. Do not just restate the classification label. Explain the visual evidence.

RULES:
- Reference the heatmap colors directly when describing regions (e.g. "the bright red area around the jaw")
- Describe specific visual details you observe in those regions such as blurring, unnatural edges, \
inconsistent lighting, texture anomalies, or color artifacts
- If classified as real, describe what natural and consistent features appear in the highlighted areas
- If classified as fake, describe the specific anomalies or artifacts visible in the highlighted areas
- Never say "the model classified this as fake therefore it is fake". Show the visual evidence instead
- Write naturally as if explaining to a friend. Avoid jargon except in the technical section
- Do not use long dashes. Use commas or short sentences instead
- Be specific and concrete
- If region labels are provided, write a one-sentence comment for each in the [REGIONS] section

Here are three examples of the tone and style you should follow:

EXAMPLE 1 - Clear Fake:
[SUMMARY]
This image has been digitally manipulated. The heatmap reveals unnatural blending around the jawline and eyes that is inconsistent with a real photograph.

[DETAILED]
As you can see in the heatmap, the model focused heavily on the area around the jaw and the left cheek, shown in bright red. If you look at those regions in the original image, the skin texture has a smudged, slightly plastic quality, like the face has been smoothed over in a way that real skin simply does not look. Around the eye corners there is also a subtle blurriness where the generated face meets the original, which is a classic sign of a face swap. The model is 94% confident this is a deepfake, and looking at the heatmap it is easy to see why it zeroed in on those transition zones.

[TECHNICAL]
The strongest activations are concentrated along the facial boundary regions, including the jaw, cheekbones, and the area around the eyes. These are typically where face swapping models struggle most with seamless blending. The activation pattern is asymmetric, with higher attention on the left side of the face, suggesting the synthetic region may not perfectly mirror the natural lighting. One caveat is that high quality deepfakes trained on similar data may still evade detection in the lower attention regions.

[REGIONS]
Chin and jawline region | The bright red activation here highlights an unnatural smoothing of the jaw contour, where the blending between the synthetic face and the original neck creates a subtle but visible seam.
Left eye region | Yellow activation around the left eye reveals slight asymmetry in the eyelid crease that is inconsistent with the right side, a common artifact in face swap generation.

EXAMPLE 2 - Clear Real:
[SUMMARY]
This appears to be an authentic photograph. The heatmap highlights naturally consistent facial features with no signs of digital manipulation.

[DETAILED]
The heatmap shows the model paying close attention to the eyes and nose bridge, highlighted in orange and yellow. Looking at those areas in the original photo, you can see natural skin texture, consistent light reflection in the eyes, and the subtle asymmetry that real human faces have. These are things that AI generated faces often get slightly wrong. The texture around the highlighted zones looks organic and continuous, with no blurring or color bleeding at the edges. The model is 97% confident this is a real image, and the heatmap supports that since there are no suspicious boundaries or smoothed over regions.

[TECHNICAL]
Primary activations center on the eye region and nasal bridge, which are typically the most diagnostic areas for authenticity. The activation map shows a relatively diffuse and low intensity pattern overall, which tends to correlate with genuine images where no single region triggers anomaly detection. Note that this analysis is based on visual artifacts detectable at this resolution, so highly compressed or low resolution images may reduce detection reliability.

[REGIONS]
Eye and nose bridge region | The orange activation here shows the model examining the nasal bridge and inner eye corners, where the skin texture and lighting transition look completely natural and consistent with a real photograph.

EXAMPLE 3 - Borderline:
[SUMMARY]
This image shows mixed signals. The heatmap highlights some subtle inconsistencies around the hairline but they are not definitive enough to call with high confidence.

[DETAILED]
The heatmap is lighting up mainly around the hairline and the top of the forehead, shown in yellow and light orange. If you look at those spots in the original image, the hair to skin transition looks slightly too clean, almost like it was placed rather than growing naturally. That said, the rest of the face looks fairly consistent and the model is only 61% confident it is fake, which is quite low. This could be a deepfake with a high quality generator, or it could simply be a photo with strong studio lighting that makes the hairline look artificially sharp.

[TECHNICAL]
Activation intensity is relatively low across the board, with the highest response near the upper hairline boundary. This kind of diffuse low confidence pattern often appears with newer generation models that have improved blending, or with images where compression artifacts partially mask manipulation signals. The 61% fake probability sits close to the decision boundary so this result should be treated with caution.

[REGIONS]
Forehead and hairline region | The yellow activation traces the hairline boundary, where the transition between hair and skin looks unusually sharp and uniform compared to what you would expect in a natural photograph.

Now analyze the images you have been given and respond in the same style. \
Use the [SUMMARY], [DETAILED], [TECHNICAL], [REGIONS] format exactly."""


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
) -> str:
    """
    Build the user-facing prompt with detection context and region labels.

    Args:
        detection: Detection results including region_labels from GradCAM crops.
        gradcam_available: Whether a valid GradCAM heatmap was generated.

    Returns:
        Formatted prompt string to send alongside the images.
    """
    confidence_pct = f"{detection.confidence * 100:.1f}%"
    classification = detection.classification.upper()
    fake_prob = detection.probabilities.get("fake", 0)
    real_prob = detection.probabilities.get("real", 0)

    if gradcam_available:
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

    # Add region labels if available so VLM can comment on each crop
    regions_instruction = ""
    if detection.region_labels:
        labels_list = "\n".join(f"- {label}" for label in detection.region_labels)
        regions_instruction = (
            f"\n\nThe following facial regions were highlighted by the model. "
            f"In the [REGIONS] section, write one sentence for each explaining "
            f"what you observe there and why the model may have focused on it:\n"
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

    Args:
        regions_text: Raw text from the [REGIONS] section.

    Returns:
        Dict mapping region label to explanation sentence.
    """
    comments: dict[str, str] = {}

    for line in regions_text.strip().splitlines():
        line = line.strip()
        if "|" in line:
            parts = line.split("|", 1)
            label = parts[0].strip()
            comment = parts[1].strip()
            if label and comment:
                comments[label] = comment

    return comments
