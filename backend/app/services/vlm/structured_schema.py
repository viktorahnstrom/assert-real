"""
Structured VLM output schema.

Defines the JSON schema shared by every provider that supports schema-enforced
output (Anthropic tool-use, OpenAI ``response_format=json_schema``, Gemini
``response_schema``). Free-text providers (mock / rule_based) emit objects of
this same shape so downstream code can treat all providers uniformly.

The per-region object carries an ``evidence_type`` tag — ``"visual"``,
``"metric"``, or ``"heatmap"`` — and an ``evidence_ref`` string that names the
specific cue. The frontend uses these to highlight the cited metric strip,
heatmap region, or crop alongside the prose claim, turning the explanation
from "VLM with nice prose" into "auditable evidence with a paper trail".
"""

from __future__ import annotations

from typing import Final

# The allowed evidence-type tags. Kept as a tuple so it can be reused as a
# JSON Schema enum, a Python validator set, and a Pydantic ``Literal[...]``.
EVIDENCE_TYPES: Final[tuple[str, ...]] = ("visual", "metric", "heatmap")


# Canonical JSON Schema for one region claim. Provider adapters reuse this and
# wrap or transform it as their API requires.
REGION_SCHEMA: Final[dict] = {
    "type": "object",
    "properties": {
        "region": {
            "type": "string",
            "description": (
                "Human-readable region label exactly as it appeared in the "
                "[REGIONS] list of the user prompt (e.g. 'Skin Texture')."
            ),
        },
        "observation": {
            "type": "string",
            "description": (
                "One to three sentences describing what you actually see or "
                "measure for this region. No filler."
            ),
        },
        "evidence_type": {
            "type": "string",
            "enum": list(EVIDENCE_TYPES),
            "description": (
                "What grounds the observation: 'visual' = a cue you see in a "
                "crop or in the original photo; 'metric' = a value cited from "
                "the [FORENSIC EVIDENCE] block; 'heatmap' = a pattern visible "
                "in the GradCAM or ELA overlay."
            ),
        },
        "evidence_ref": {
            "type": "string",
            "description": (
                "Specific reference for the evidence: when 'metric', cite the "
                "metric name and value (e.g. 'sharpness_z=-3.8'); when "
                "'visual' or 'heatmap', name the area (e.g. 'left jaw crop' "
                "or 'GradCAM peak around the mouth'). Empty string allowed "
                "only when no specific anchor exists."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Your confidence in this observation in [0, 1]. Use lower "
                "values when the cue is subtle or ambiguous."
            ),
        },
    },
    "required": ["region", "observation", "evidence_type", "evidence_ref", "confidence"],
    "additionalProperties": False,
}


# Top-level schema returned by the VLM.
EXPLANATION_SCHEMA: Final[dict] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One sentence stating what was found and where.",
        },
        "detailed_analysis": {
            "type": "string",
            "description": (
                "Two to three sentences describing the most distinctive thing "
                "that makes this image look real or fake."
            ),
        },
        "technical_notes": {
            "type": "string",
            "description": ("Three to five lines of forensic notes for a technical reader."),
        },
        "regions": {
            "type": "array",
            "items": REGION_SCHEMA,
            "description": (
                "One entry per facial region you commented on. Cover at least "
                "the regions listed in the user prompt; you may add more if "
                "you spotted something noteworthy outside the heatmap."
            ),
        },
    },
    "required": ["summary", "detailed_analysis", "technical_notes", "regions"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Provider-specific adapters
# ---------------------------------------------------------------------------


def anthropic_tool_definition() -> dict:
    """Return the Anthropic tool-use tool definition for structured output.

    Anthropic enforces JSON shape via tool_use: the assistant must call this
    tool with arguments matching ``input_schema``. We pin tool_choice to this
    tool so the model cannot reply with free text.
    """
    return {
        "name": "submit_explanation",
        "description": (
            "Submit the structured deepfake explanation. Every region claim "
            "must have a non-empty evidence_ref unless evidence_type implies "
            "no specific anchor exists."
        ),
        "input_schema": EXPLANATION_SCHEMA,
    }


def openai_response_format() -> dict:
    """Return the OpenAI Responses API ``text.format`` block.

    Uses strict json_schema mode so the API rejects malformed output before
    it reaches us.
    """
    return {
        "format": {
            "type": "json_schema",
            "name": "DeepfakeExplanation",
            "schema": EXPLANATION_SCHEMA,
            "strict": True,
        }
    }


def gemini_response_schema() -> dict:
    """Return the Gemini-compatible response_schema dict.

    The genai SDK accepts a plain Python dict mirroring the JSON Schema with
    a few extensions (``propertyOrdering`` etc.). We keep it minimal — the
    canonical schema is enough.
    """

    # Gemini does not support additionalProperties=False; strip it so the
    # SDK does not reject the schema.
    def _strip_additional_properties(node: dict) -> dict:
        out = {k: v for k, v in node.items() if k != "additionalProperties"}
        if "properties" in out:
            out["properties"] = {
                name: _strip_additional_properties(prop) if isinstance(prop, dict) else prop
                for name, prop in out["properties"].items()
            }
        if "items" in out and isinstance(out["items"], dict):
            out["items"] = _strip_additional_properties(out["items"])
        return out

    return _strip_additional_properties(EXPLANATION_SCHEMA)
