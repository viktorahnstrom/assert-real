"""
Face Category Taxonomy for Deepfake Detection

Defines a structured artifact taxonomy for face-specific deepfake detection.
Taxonomic categories are adapted from the anatomical and stylistic categories
introduced in Kamali et al. (2024) "How to Distinguish AI-Generated Images from
Authentic Photographs", which proposes five broad categories across full images:
anatomical, stylistic, functional, physics-based, and sociocultural.

**Scope of adaptation:**
We retain only the anatomical and stylistic categories, further specialised to
the face region as cropped by our detection pipeline.  Functional (object
coherence), physics-based (lighting/shadow consistency), and sociocultural
(contextual plausibility) categories are out of scope for this face-only
deepfake detector and are documented as a limitation of the current system.

Each :class:`FaceCategory` maps a logical face region to:
- A list of MediaPipe Face Mesh landmark indices that delimit that region.
- A list of common deepfake artifact types observed in that region.

These are the shared contract consumed by:
1. The MediaPipe region mapper (future)
2. All VLM explanation providers (:mod:`app.services.vlm`)
3. The GradCAM region labeller (:mod:`app.services.gradcam_service`)
4. The frontend display layer

Landmark index conventions follow the MediaPipe Face Mesh 468-point model
(canonical topology, *not* the refined iris model).  Indices are from the
person's perspective: "right" means the subject's anatomical right (appears on
the left side of the image).

References:
    Kamali, S., Momeny, M., Rabbani, H., & Akbarizadeh, G. (2024).
    How to distinguish AI-generated images from authentic photographs.
    *IEEE Access*, 12, 78823-78843.
    https://doi.org/10.1109/ACCESS.2024.3409217
    Cite as: \\parencite{kamali2024distinguish}
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FaceCategory:
    """A named facial region with associated MediaPipe landmarks and artifact types.

    Attributes:
        id: Stable snake_case identifier used as dictionary key and API field.
        label: Human-readable display label for UI and VLM prompts.
        landmark_indices: Tuple of MediaPipe Face Mesh landmark indices (0-467)
            that delimit this region.  Indices must not overlap with any other
            category — enforced by the unit tests.
        common_artifacts: Tuple of artifact descriptions typical for this region
            in deepfake imagery, ordered roughly by observed frequency.
    """

    id: str
    label: str
    landmark_indices: tuple[int, ...]
    common_artifacts: tuple[str, ...]


# ---------------------------------------------------------------------------
# Landmark index sets — grouped here for readability before assignment.
# All indices reference MediaPipe Face Mesh canonical topology (468 points).
# ---------------------------------------------------------------------------

# Eyes & Pupils
# Right eye (subject's right, appears on the *left* side of the image)
_RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
# Left eye (subject's left, appears on the *right* side of the image)
_LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)

# Eyebrows & Eyelashes
_RIGHT_EYEBROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
_LEFT_EYEBROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)

# Mouth & Teeth — outer lip contour followed by inner lip contour
_OUTER_LIP = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
_INNER_LIP = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)

# Skin Texture — cheek and mid-face interior points, excluding face boundary
_RIGHT_CHEEK = (36, 100, 101, 50, 205, 206, 207, 187, 116, 117, 118, 119, 120, 121, 126, 142, 203, 192)
_LEFT_CHEEK = (266, 329, 330, 280, 425, 426, 427, 411, 345, 346, 347, 348, 349, 350, 355, 371, 423, 416)

# Hairline & Ears — upper face oval and ear-adjacent (preauricular) landmarks
_HAIRLINE = (10, 338, 297, 332, 284, 251, 389, 21, 54, 103, 67, 109)
_EAR_ADJACENT = (127, 162, 234, 93, 356, 454)  # temple/tragus-area approximation

# Facial Boundaries — lower jaw oval, nose bridge, and nose tip/sides
_JAWLINE = (152, 148, 176, 149, 150, 136, 172, 58, 132, 323, 361, 288, 397, 365, 379, 378, 400, 377)
_NOSE = (168, 6, 197, 195, 5, 4, 1, 2, 19, 94, 129, 358, 122, 351, 48, 278, 64, 294)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

FACE_CATEGORIES: dict[str, FaceCategory] = {
    "eyes_pupils": FaceCategory(
        id="eyes_pupils",
        label="Eyes & Pupils",
        landmark_indices=_RIGHT_EYE + _LEFT_EYE,
        common_artifacts=(
            "pupil shape asymmetry between left and right eye",
            "unnatural glossiness or over-bright specular highlights",
            "missing or blurred iris texture and limbal ring",
            "hollow or unfocused gaze with flat reflections",
            "eyelid crease inconsistency or asymmetric aperture",
        ),
    ),
    "eyebrows_eyelashes": FaceCategory(
        id="eyebrows_eyelashes",
        label="Eyebrows & Eyelashes",
        landmark_indices=_RIGHT_EYEBROW + _LEFT_EYEBROW,
        common_artifacts=(
            "individual strands merged into a uniform block",
            "left-right shape or density asymmetry",
            "unnaturally uniform thickness along the brow arc",
            "floating or disconnected lash artefacts at eyelid margin",
            "blurred boundary between brow hair and surrounding skin",
        ),
    ),
    "mouth_teeth": FaceCategory(
        id="mouth_teeth",
        label="Mouth & Teeth",
        landmark_indices=_OUTER_LIP + _INNER_LIP,
        common_artifacts=(
            "overlapping or fused tooth geometry",
            "blurred or missing gum-to-tooth boundary",
            "inconsistent tooth count or irregular spacing",
            "lip colour discontinuity at the commissures",
            "unnatural lip texture or over-smoothed vermilion border",
        ),
    ),
    "skin_texture": FaceCategory(
        id="skin_texture",
        label="Skin Texture",
        landmark_indices=_RIGHT_CHEEK + _LEFT_CHEEK,
        common_artifacts=(
            "plastic or waxy surface appearance",
            "absent or overly uniform pore pattern",
            "unnatural smoothness inconsistent with subject age",
            "blotchy colour gradients not following facial contours",
            "texture discontinuity between cheek and surrounding regions",
        ),
    ),
    "hairline_ears": FaceCategory(
        id="hairline_ears",
        label="Hairline & Ears",
        landmark_indices=_HAIRLINE + _EAR_ADJACENT,
        common_artifacts=(
            "blending artefacts at the hair-to-skin boundary",
            "wispy or floating stray hair strands with no follicle origin",
            "asymmetric ear geometry or missing helix detail",
            "inconsistent hair density or direction near the temples",
            "hard or unnaturally sharp hairline edge",
        ),
    ),
    "facial_boundaries": FaceCategory(
        id="facial_boundaries",
        label="Facial Boundaries",
        landmark_indices=_JAWLINE + _NOSE,
        common_artifacts=(
            "jawline smoothing or blending seam at face-neck junction",
            "asymmetric jaw contour relative to midline",
            "nose bridge width or curvature proportion anomaly",
            "nostril shape asymmetry or missing shadow under the nose",
            "colour or texture discontinuity along the face oval boundary",
        ),
    ),
}
