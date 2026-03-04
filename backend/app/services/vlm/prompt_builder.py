"""
VLM Prompt Builder

Constructs structured prompts for VLM providers that ground
explanations in GradCAM heatmap visualizations. The prompt
template ensures explanations are:

1. Anchored to actual heatmap activation regions (not hallucinated)
2. Structured in three tiers (summary / detailed / technical)
3. Natural and human-readable for non-technical users
4. Grounded via few-shot examples to establish tone and style
"""

from app.services.vlm.base import DetectionContext

SYSTEM_PROMPT = """You are a forensic image analyst explaining deepfake detection results \
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

Here are three examples of the tone and style you should follow:

EXAMPLE 1 - Clear Fake:
[SUMMARY]
This image has been digitally manipulated. The heatmap reveals unnatural blending around the jawline and eyes that is inconsistent with a real photograph.

[DETAILED]
As you can see in the heatmap, the model focused heavily on the area around the jaw and the left cheek, shown in bright red. If you look at those regions in the original image, the skin texture has a smudged, slightly plastic quality, like the face has been smoothed over in a way that real skin simply does not look. Around the eye corners there is also a subtle blurriness where the generated face meets the original, which is a classic sign of a face swap. The model is 94% confident this is a deepfake, and looking at the heatmap it is easy to see why it zeroed in on those transition zones.

[TECHNICAL]
The strongest activations are concentrated along the facial boundary regions, including the jaw, cheekbones, and the area around the eyes. These are typically where face swapping models struggle most with seamless blending. The activation pattern is asymmetric, with higher attention on the left side of the face, suggesting the synthetic region may not perfectly mirror the natural lighting. One caveat is that high quality deepfakes trained on similar data may still evade detection in the lower attention regions.

EXAMPLE 2 - Clear Real:
[SUMMARY]
This appears to be an authentic photograph. The heatmap highlights naturally consistent facial features with no signs of digital manipulation.

[DETAILED]
The heatmap shows the model paying close attention to the eyes and nose bridge, highlighted in orange and yellow. Looking at those areas in the original photo, you can see natural skin texture, consistent light reflection in the eyes, and the subtle asymmetry that real human faces have. These are things that AI generated faces often get slightly wrong. The texture around the highlighted zones looks organic and continuous, with no blurring or color bleeding at the edges. The model is 97% confident this is a real image, and the heatmap supports that since there are no suspicious boundaries or smoothed over regions.

[TECHNICAL]
Primary activations center on the eye region and nasal bridge, which are typically the most diagnostic areas for authenticity. The activation map shows a relatively diffuse and low intensity pattern overall, which tends to correlate with genuine images where no single region triggers anomaly detection. Note that this analysis is based on visual artifacts detectable at this resolution, so highly compressed or low resolution images may reduce detection reliability.

EXAMPLE 3 - Borderline:
[SUMMARY]
This image shows mixed signals. The heatmap highlights some subtle inconsistencies around the hairline but they are not definitive enough to call with high confidence.

[DETAILED]
The heatmap is lighting up mainly around the hairline and the top of the forehead, shown in yellow and light orange. If you look at those spots in the original image, the hair to skin transition looks slightly too clean, almost like it was placed rather than growing naturally. That said, the rest of the face looks fairly consistent and the model is only 61% confident it is fake, which is quite low. This could be a deepfake with a high quality generator, or it could simply be a photo with strong studio lighting that makes the hairline look artificially sharp.

[TECHNICAL]
Activation intensity is relatively low across the board, with the highest response near the upper hairline boundary. This kind of diffuse low confidence pattern often appears with newer generation models that have improved blending, or with images where compression artifacts partially mask manipulation signals. The 61% fake probability sits close to the decision boundary so this result should be treated with caution and would benefit from a second opinion.

Now analyze the images you have been given and respond in the same style. \
Use the [SUMMARY], [DETAILED], [TECHNICAL] format exactly."""


def build_explanation_prompt(detection: DetectionContext) -> str:
    """
    Build the user-facing prompt with detection context.

    Args:
        detection: Detection results to include in the prompt

    Returns:
        Formatted prompt string to send alongside the images
    """
    confidence_pct = f"{detection.confidence * 100:.1f}%"
    classification = detection.classification.upper()
    fake_prob = detection.probabilities.get("fake", 0)
    real_prob = detection.probabilities.get("real", 0)

    prompt = f"""Here are the detection results for the image above.

Detection result: {classification} ({confidence_pct} confidence)
Fake probability: {fake_prob:.1%} | Real probability: {real_prob:.1%}
Model: {detection.model_used}

Look at the GradCAM heatmap (Image 2) and describe what you observe in the highlighted regions \
of the original photo (Image 1). What specific visual details in those regions support \
this classification? Use the [SUMMARY], [DETAILED], [TECHNICAL] format."""

    return prompt


def parse_explanation_response(raw_response: str) -> dict[str, str]:
    """
    Parse the three-tier explanation from a VLM response.

    Args:
        raw_response: Raw text response from the VLM

    Returns:
        Dict with keys: summary, detailed_analysis, technical_notes
    """
    sections: dict[str, str] = {
        "summary": "",
        "detailed_analysis": "",
        "technical_notes": "",
    }

    if "[SUMMARY]" in raw_response:
        parts = raw_response.split("[SUMMARY]")
        if len(parts) > 1:
            remainder = parts[1]

            if "[DETAILED]" in remainder:
                summary_part, remainder = remainder.split("[DETAILED]", 1)
                sections["summary"] = summary_part.strip()
            else:
                sections["summary"] = remainder.strip()

            if "[TECHNICAL]" in remainder:
                detailed_part, technical_part = remainder.split("[TECHNICAL]", 1)
                sections["detailed_analysis"] = detailed_part.strip()
                sections["technical_notes"] = technical_part.strip()
            else:
                sections["detailed_analysis"] = remainder.strip()
    else:
        # Fallback: VLM did not follow format
        sections["summary"] = raw_response[:200].strip()
        sections["detailed_analysis"] = raw_response.strip()

    return sections
