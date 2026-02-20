"""
VLM Prompt Builder

Constructs structured prompts for VLM providers that ground
explanations in GradCAM heatmap visualizations. The prompt
template ensures explanations are:

1. Anchored to actual heatmap activation regions (not hallucinated)
2. Structured in three tiers (summary / detailed / technical)
3. Appropriate for non-technical users (plain language)
"""

from app.services.vlm.base import DetectionContext

# System prompt that all providers receive
SYSTEM_PROMPT = """You are an expert forensic image analyst specializing in deepfake detection. \
Your role is to explain detection results to non-technical users in clear, accessible language.

You will receive:
1. An original image that was analyzed
2. A GradCAM heatmap overlay showing which regions the detection model focused on \
(brighter/warmer colors = higher model attention)
3. The detection model's classification and confidence score

CRITICAL RULES:
- ONLY describe artifacts or features visible in the regions highlighted by the heatmap
- Do NOT speculate about regions that are NOT highlighted
- If the heatmap shows no strong activations, say so honestly
- Use plain language — avoid jargon unless in the technical notes section
- Be specific about facial regions (eyes, nose, jawline, hairline, ears, skin texture, etc.)
- If classified as REAL, explain what natural features the model found convincing

Respond in EXACTLY this format with these three sections separated by the markers shown:

[SUMMARY]
One sentence: what the result is and the primary reason. Keep it under 30 words.

[DETAILED]
One paragraph (3-5 sentences): Describe specific visual features or artifacts in the \
highlighted regions that support the classification. Reference the heatmap regions explicitly \
(e.g., "The heatmap highlights the area around the eyes, where..."). Mention the confidence level.

[TECHNICAL]
One paragraph (2-3 sentences): Note which facial regions received the strongest model attention, \
any patterns in the activation distribution, and potential limitations of the analysis. \
This section is for users who want deeper technical context."""


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

    prompt = f"""Analyze this image based on the deepfake detection results below.

**Detection Result:** {classification}
**Confidence:** {confidence_pct}
**Model Used:** {detection.model_used}
**Class Probabilities:** Real: {detection.probabilities.get("real", 0):.3f}, \
Fake: {detection.probabilities.get("fake", 0):.3f}

The first image is the original photo that was analyzed.
The second image is the GradCAM heatmap overlay showing which regions the detection model \
focused on. Brighter/warmer regions received more model attention.

Please explain why the model made this classification, grounding your explanation in the \
heatmap-highlighted regions. Follow the [SUMMARY], [DETAILED], [TECHNICAL] format exactly."""

    return prompt


def parse_explanation_response(raw_response: str) -> dict[str, str]:
    """
    Parse the three-tier explanation from a VLM response.

    Args:
        raw_response: Raw text response from the VLM

    Returns:
        Dict with keys: summary, detailed_analysis, technical_notes
    """

    sections = {
        "summary": "",
        "detailed_analysis": "",
        "technical_notes": "",
    }

    # Try to parse structured format
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
        # Fallback: if VLM didn't follow format, put everything in detailed
        sections["summary"] = raw_response[:200].strip()
        sections["detailed_analysis"] = raw_response.strip()

    return sections
