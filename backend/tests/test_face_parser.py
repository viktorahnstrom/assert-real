"""
Smoke tests for :mod:`app.services.face_parser`.

Downloads the BiSeNet weights on first run (~50 MB) and runs CPU inference on
one quiz-set real face.  Skipped automatically when facexlib is not installed
or when the test image is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("facexlib", reason="facexlib required for face_parser tests")

from app.services.face_parser import (  # noqa: E402
    BISENET_CLASSES,
    FaceParser,
    FaceParsingResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = REPO_ROOT / "desktop" / "public" / "quiz-images" / "Real1.jpg"


@pytest.fixture(scope="module")
def parsing_result() -> FaceParsingResult:
    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found at {TEST_IMAGE}")
    image = Image.open(TEST_IMAGE).convert("RGB")
    parser = FaceParser()
    return parser.parse(image)


class TestFaceParserStructure:
    """Shape and dtype invariants of FaceParsingResult."""

    def test_image_size_matches_input(self, parsing_result: FaceParsingResult) -> None:
        assert parsing_result.image_size == Image.open(TEST_IMAGE).size

    def test_fine_masks_have_all_19_classes(self, parsing_result: FaceParsingResult) -> None:
        assert set(parsing_result.masks_fine) == set(BISENET_CLASSES)

    def test_fine_masks_have_image_shape(self, parsing_result: FaceParsingResult) -> None:
        width, height = parsing_result.image_size
        for name, mask in parsing_result.masks_fine.items():
            assert mask.shape == (height, width), f"{name} has wrong shape {mask.shape}"
            assert mask.dtype == bool, f"{name} is not boolean (got {mask.dtype})"

    def test_ui_masks_cover_expected_categories(self, parsing_result: FaceParsingResult) -> None:
        expected = {
            "eyes_pupils",
            "eyebrows_eyelashes",
            "mouth_teeth",
            "skin_texture",
            "hairline_ears",
            "facial_boundaries",
        }
        assert set(parsing_result.masks_ui) == expected


class TestFaceParserContent:
    """Masks should be non-trivial for a clear real-face photo."""

    @pytest.mark.parametrize("class_name", ["skin", "l_eye", "r_eye", "nose", "hair"])
    def test_fine_mask_non_empty(self, parsing_result: FaceParsingResult, class_name: str) -> None:
        mask = parsing_result.masks_fine[class_name]
        assert mask.sum() > 0, f"BiSeNet produced empty mask for {class_name} on Real1.jpg"

    def test_mouth_region_non_empty_via_ui_merge(self, parsing_result: FaceParsingResult) -> None:
        """`mouth` fine class can be 0 on closed-mouth photos; the mouth_teeth
        UI merge (mouth ∪ u_lip ∪ l_lip) is the contract downstream consumers
        depend on, so assert that instead."""
        assert parsing_result.masks_ui["mouth_teeth"].sum() > 0

    @pytest.mark.parametrize(
        "ui_id",
        [
            "eyes_pupils",
            "eyebrows_eyelashes",
            "mouth_teeth",
            "skin_texture",
            "hairline_ears",
        ],
    )
    def test_ui_mask_non_empty(self, parsing_result: FaceParsingResult, ui_id: str) -> None:
        assert parsing_result.masks_ui[ui_id].sum() > 0
