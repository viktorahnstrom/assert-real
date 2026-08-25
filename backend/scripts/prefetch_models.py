"""Download every model weight at image build time.

Runs during ``docker build`` so the container starts with a warm cache
and never needs network access to serve a request.  Exercises the same
loader functions the application uses so weights land in the paths the
app reads from.
"""

from __future__ import annotations

import sys


def prefetch_detection_model() -> None:
    """EfficientNet-B4 deepfake detection checkpoint from HuggingFace Hub."""
    from app.utils.model_loader import load_model_checkpoint

    load_model_checkpoint()
    print("  detection model: ok")


def prefetch_face_parser() -> None:
    """BiSeNet face parsing weights (CelebAMask-HQ, 19 classes)."""
    from facexlib.parsing import init_parsing_model

    init_parsing_model(model_name="bisenet", device="cpu")
    print("  face parser: ok")


def prefetch_face_category_mapper() -> None:
    """MediaPipe Face Mesh solution used by FaceCategoryMapper."""
    from app.services.face_category_mapper import FaceCategoryMapper

    mapper = FaceCategoryMapper()
    mapper.close()
    print("  face category mapper (MediaPipe): ok")


def main() -> int:
    steps = [
        ("detection model", prefetch_detection_model),
        ("face parser", prefetch_face_parser),
        ("face category mapper", prefetch_face_category_mapper),
    ]

    failed = False
    for name, fn in steps:
        print(f"Prefetching {name}...")
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed = True

    if failed:
        print(
            "\nOne or more models failed to prefetch. Failing the build.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
