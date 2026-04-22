"""
Face Category Mapper

Maps GradCAM evidence regions to semantic face categories using MediaPipe
Face Mesh landmark detection.

For each evidence region bounding box the mapper:
1. Runs MediaPipe Face Mesh (468 landmarks, CPU-only) on the full image.
2. Finds which landmarks fall inside the bounding box.
3. Counts landmarks per category (using the landmark → category mapping
   derived from FACE_CATEGORIES) and picks the category with the highest count.
4. Computes overlap_confidence as winning_count / total_in_bbox.

When MediaPipe cannot detect a face, or when no categorised landmarks fall
inside a bounding box, the mapper falls back to get_category_for_label() which
uses the GradCAM region label string to look up the category.  Regions that
cannot be resolved either way are assigned the "unknown" category with
overlap_confidence 0.0.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from app.services.categories import FACE_CATEGORIES, get_category_for_label
from app.services.face_parser import FaceParsingResult
from app.services.vlm.base import RegionWithCategory

# Resolve model path relative to this file so it works both locally
# (backend/models/) and inside Docker (/app/models/).
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"

logger = logging.getLogger(__name__)

_UNKNOWN_CATEGORY_ID = "unknown"
_UNKNOWN_CATEGORY_LABEL = "Unknown Region"

# Number of common_artifacts forwarded to RegionWithCategory for VLM prompts.
_ARTIFACT_HINT_COUNT = 3


@dataclass
class CategorizedRegion:
    """An evidence region enriched with face category information.

    Attributes:
        image: Cropped PIL image of the evidence region.
        label: Original GradCAM free-text label (e.g. "Left eye region").
        activation_score: Peak GradCAM activation score within the crop.
        bbox: Bounding box (x1, y1, x2, y2) in the original image.
        category_id: Resolved category key (e.g. "eyes_pupils"),
            or "unknown" when resolution failed.
        category_label: Human-readable label (e.g. "Eyes & Pupils").
        overlap_confidence: Fraction of landmarks in the bbox that belong to
            the winning category.  0.0 when the fallback path was used.
    """

    image: Image.Image
    label: str
    activation_score: float
    bbox: tuple[int, int, int, int]
    category_id: str
    category_label: str
    overlap_confidence: float

    def to_region_with_category(self) -> RegionWithCategory:
        """Convert to RegionWithCategory for use in DetectionContext.

        Returns:
            RegionWithCategory carrying category metadata and the top artifact
            hints for use in VLM prompt enrichment.
        """
        category = FACE_CATEGORIES.get(self.category_id)
        artifacts = category.common_artifacts[:_ARTIFACT_HINT_COUNT] if category else ()
        return RegionWithCategory(
            label=self.label,
            category_id=self.category_id,
            category_label=self.category_label,
            common_artifacts=artifacts,
        )


class FaceCategoryMapper:
    """Maps GradCAM bounding boxes to face categories via MediaPipe landmarks.

    Initialises a single MediaPipe Face Mesh instance (static_image_mode,
    CPU-only) that is reused across all calls to map_regions().  Call close()
    — or use the instance as a context manager — to release resources.

    Usage::

        mapper = FaceCategoryMapper()
        categorized = mapper.map_regions(pil_image, evidence_regions)
        detection_context.region_categories = [
            r.to_region_with_category() for r in categorized
        ]
        mapper.close()
    """

    def __init__(self) -> None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {_MODEL_PATH}. "
                "Download it with: curl -L -o backend/models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._landmark_to_category_id: dict[int, str] = self._build_landmark_map()
        logger.info(
            "FaceCategoryMapper initialised (%d landmark mappings)",
            len(self._landmark_to_category_id),
        )

    @staticmethod
    def _build_landmark_map() -> dict[int, str]:
        """Build reverse mapping: landmark index → category id.

        Derived from FACE_CATEGORIES so it stays in sync automatically when
        categories are added or updated.
        """
        mapping: dict[int, str] = {}
        for cat_id, cat in FACE_CATEGORIES.items():
            for idx in cat.landmark_indices:
                mapping[idx] = cat_id
        return mapping

    def detect_face_bbox(
        self,
        image: Image.Image,
        padding_pct: float = 0.05,
    ) -> tuple[int, int, int, int] | None:
        """Return a padded bounding box around the detected face.

        Args:
            image: PIL image to run landmark detection on.
            padding_pct: Fraction of image dimensions to pad outward from the
                tight landmark bbox (default 5%).

        Returns:
            (x1, y1, x2, y2) in pixel coordinates, clipped to image bounds,
            or None if no face was detected.
        """
        landmarks_px = self._detect_landmarks(image)
        if landmarks_px is None:
            return None

        width, height = image.size
        xs = [lm[0] for lm in landmarks_px]
        ys = [lm[1] for lm in landmarks_px]

        pad_x = int(width * padding_pct)
        pad_y = int(height * padding_pct)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(width, max(xs) + pad_x)
        y2 = min(height, max(ys) + pad_y)

        return (x1, y1, x2, y2)

    def map_regions(
        self,
        image: Image.Image,
        evidence_regions: list[dict],
        parsing_result: FaceParsingResult | None = None,
    ) -> list[CategorizedRegion]:
        """Map each GradCAM evidence region to its dominant face category.

        Args:
            image: The original PIL image used for landmark detection.
            evidence_regions: Output of GradCAMGenerator.extract_evidence_regions(),
                a list of dicts with keys: image, label, activation_score, bbox.
            parsing_result: Optional BiSeNet face-parsing masks.  When provided,
                assignment uses pixel-area overlap between each bbox and each
                UI category mask — more accurate than landmark counting near
                boundaries.  Falls back to landmarks when no category has
                overlapping pixels.

        Returns:
            List of CategorizedRegion in the same order as evidence_regions.
            Never raises — failures fall back to label-based mapping or "unknown".
        """
        if not evidence_regions:
            return []

        landmarks_px = self._detect_landmarks(image)

        if landmarks_px is None:
            logger.debug("MediaPipe found no face — using label fallback for all regions")

        return [
            self._categorize_region(region, landmarks_px, parsing_result)
            for region in evidence_regions
        ]

    def _detect_landmarks(
        self,
        image: Image.Image,
    ) -> list[tuple[int, int]] | None:
        """Run MediaPipe Face Landmarker and return pixel-space landmark coordinates.

        Args:
            image: Input PIL image (any mode; converted to RGB internally).

        Returns:
            List of (x_px, y_px) indexed by landmark index (0–467), or None
            when no face is detected.
        """
        rgb = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        width, height = image.size
        face = result.face_landmarks[0]
        return [(int(lm.x * width), int(lm.y * height)) for lm in face]

    def _categorize_region(
        self,
        region: dict,
        landmarks_px: list[tuple[int, int]] | None,
        parsing_result: FaceParsingResult | None = None,
    ) -> CategorizedRegion:
        """Assign a face category to one evidence region.

        Priority order:
        1. Pixel-area overlap with BiSeNet parsing masks (when provided).
        2. MediaPipe landmark counting inside the bbox.
        3. get_category_for_label() string lookup on the GradCAM label.
        """
        if parsing_result is not None:
            result = self._assign_by_parsing(region, parsing_result)
            if result is not None:
                return result

        if landmarks_px is not None:
            result = self._assign_by_landmarks(region, landmarks_px)
            if result is not None:
                return result

        return self._fallback_region(region)

    @staticmethod
    def _assign_by_parsing(
        region: dict,
        parsing_result: FaceParsingResult,
    ) -> CategorizedRegion | None:
        """Spatial assignment: pick the UI category with the most pixels in the bbox.

        Confidence is the winning category's pixel count divided by the total
        number of categorised pixels inside the bbox (excluding background and
        classes not in the UI merge, so fully-background bboxes return None).
        """
        x1, y1, x2, y2 = region["bbox"]
        width, height = parsing_result.image_size
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None

        area_per_category: dict[str, int] = {}
        total = 0
        for cat_id, mask in parsing_result.masks_ui.items():
            count = int(mask[y1:y2, x1:x2].sum())
            if count > 0:
                area_per_category[cat_id] = count
                total += count

        if not area_per_category:
            return None

        winning_id = max(area_per_category, key=lambda k: area_per_category[k])
        confidence = area_per_category[winning_id] / total if total > 0 else 0.0
        winning_category = FACE_CATEGORIES[winning_id]

        return CategorizedRegion(
            image=region["image"],
            label=region["label"],
            activation_score=region["activation_score"],
            bbox=region["bbox"],
            category_id=winning_id,
            category_label=winning_category.label,
            overlap_confidence=round(confidence, 4),
        )

    def _assign_by_landmarks(
        self,
        region: dict,
        landmarks_px: list[tuple[int, int]],
    ) -> CategorizedRegion | None:
        """Spatial assignment: count category landmarks inside the bbox.

        Returns:
            CategorizedRegion if a winning category is found, else None.
        """
        x1, y1, x2, y2 = region["bbox"]
        category_counts: dict[str, int] = {}
        total_in_bbox = 0

        for idx, (lx, ly) in enumerate(landmarks_px):
            if x1 <= lx <= x2 and y1 <= ly <= y2:
                total_in_bbox += 1
                cat_id = self._landmark_to_category_id.get(idx)
                if cat_id:
                    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        if not category_counts:
            return None

        winning_id = max(category_counts, key=lambda k: category_counts[k])
        winning_count = category_counts[winning_id]
        confidence = winning_count / total_in_bbox if total_in_bbox > 0 else 0.0
        winning_category = FACE_CATEGORIES[winning_id]

        return CategorizedRegion(
            image=region["image"],
            label=region["label"],
            activation_score=region["activation_score"],
            bbox=region["bbox"],
            category_id=winning_id,
            category_label=winning_category.label,
            overlap_confidence=round(confidence, 4),
        )

    def _fallback_region(self, region: dict) -> CategorizedRegion:
        """Label-based fallback: use get_category_for_label() string lookup."""
        label = region["label"]
        category = get_category_for_label(label)

        if category:
            cat_id = category.id
            cat_label = category.label
        else:
            cat_id = _UNKNOWN_CATEGORY_ID
            cat_label = _UNKNOWN_CATEGORY_LABEL

        return CategorizedRegion(
            image=region["image"],
            label=label,
            activation_score=region["activation_score"],
            bbox=region["bbox"],
            category_id=cat_id,
            category_label=cat_label,
            overlap_confidence=0.0,
        )

    def close(self) -> None:
        """Release MediaPipe Face Landmarker resources."""
        self._landmarker.close()

    def __enter__(self) -> "FaceCategoryMapper":
        return self

    def __exit__(self, *_) -> None:
        self.close()
