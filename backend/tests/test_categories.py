"""
Unit tests for app.services.categories

Validates structural invariants of FACE_CATEGORIES that are required for the
MediaPipe mapper, VLM prompt builder, and frontend display to work correctly.
"""

from app.services.categories import FACE_CATEGORIES, FaceCategory


class TestFaceCategoryStructure:
    """FaceCategory dataclass integrity."""

    def test_all_categories_are_face_category_instances(self):
        for key, cat in FACE_CATEGORIES.items():
            assert isinstance(cat, FaceCategory), f"{key!r} is not a FaceCategory"

    def test_id_matches_dictionary_key(self):
        for key, cat in FACE_CATEGORIES.items():
            assert cat.id == key, f"id {cat.id!r} does not match dict key {key!r}"

    def test_all_categories_have_non_empty_label(self):
        for key, cat in FACE_CATEGORIES.items():
            assert cat.label.strip(), f"{key!r} has an empty label"

    def test_all_categories_have_landmark_indices(self):
        for key, cat in FACE_CATEGORIES.items():
            assert len(cat.landmark_indices) > 0, f"{key!r} has no landmark indices"

    def test_all_categories_have_common_artifacts(self):
        for key, cat in FACE_CATEGORIES.items():
            assert len(cat.common_artifacts) > 0, f"{key!r} has no common_artifacts"

    def test_landmark_indices_are_valid_mediapipe_range(self):
        """All indices must be in the MediaPipe Face Mesh range [0, 467]."""
        for key, cat in FACE_CATEGORIES.items():
            for idx in cat.landmark_indices:
                assert 0 <= idx <= 467, (
                    f"{key!r} contains out-of-range landmark index {idx} "
                    f"(MediaPipe Face Mesh supports 0-467)"
                )

    def test_landmark_indices_have_no_duplicates_within_category(self):
        for key, cat in FACE_CATEGORIES.items():
            seen = set()
            for idx in cat.landmark_indices:
                assert idx not in seen, f"{key!r} contains duplicate landmark index {idx}"
                seen.add(idx)


class TestNoOverlappingLandmarks:
    """Landmark indices must be disjoint across all categories.

    This is a hard contract: the MediaPipe mapper and GradCAM region labeller
    both rely on each landmark belonging to exactly one category.
    """

    def test_no_overlapping_landmark_indices_between_categories(self):
        category_ids = list(FACE_CATEGORIES.keys())
        for i in range(len(category_ids)):
            for j in range(i + 1, len(category_ids)):
                id_a = category_ids[i]
                id_b = category_ids[j]
                set_a = set(FACE_CATEGORIES[id_a].landmark_indices)
                set_b = set(FACE_CATEGORIES[id_b].landmark_indices)
                overlap = set_a & set_b
                assert not overlap, (
                    f"Categories {id_a!r} and {id_b!r} share landmark indices: {sorted(overlap)}"
                )

    def test_all_landmark_indices_across_categories_are_unique(self):
        """Aggregate uniqueness check as a fast single-pass complement."""
        all_indices: list[int] = []
        for cat in FACE_CATEGORIES.values():
            all_indices.extend(cat.landmark_indices)
        assert len(all_indices) == len(set(all_indices)), (
            "Duplicate landmark indices found across FACE_CATEGORIES — "
            "run test_no_overlapping_landmark_indices_between_categories for details"
        )


class TestExpectedCategories:
    """Smoke-test that the six required categories are present."""

    REQUIRED_IDS = {
        "eyes_pupils",
        "eyebrows_eyelashes",
        "mouth_teeth",
        "skin_texture",
        "hairline_ears",
        "facial_boundaries",
    }

    def test_all_required_category_ids_present(self):
        assert self.REQUIRED_IDS <= set(FACE_CATEGORIES.keys()), (
            f"Missing categories: {self.REQUIRED_IDS - set(FACE_CATEGORIES.keys())}"
        )
