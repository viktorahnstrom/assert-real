"""
Qualitative side-by-side comparison of Grad-CAM vs LayerCAM on XADE fakes.

Runs both CAM methods through the existing GradCAMGenerator against the
deepfake detector checkpoint used in production, and saves a single PNG per
input showing: original | Grad-CAM overlay | LayerCAM overlay.

Usage:
    python -m backend.scripts.compare_cams
    python -m backend.scripts.compare_cams --input desktop/public/quiz-images --output backend/tests/fixtures/cam_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.detect import DeepfakeDetector  # noqa: E402
from app.services.face_category_mapper import FaceCategoryMapper  # noqa: E402
from app.services.gradcam_service import GradCAMGenerator  # noqa: E402
from app.utils.model_loader import load_model_checkpoint  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "desktop" / "public" / "quiz-images"
DEFAULT_OUTPUT = BACKEND_ROOT / "tests" / "fixtures" / "cam_comparison"
DEFAULT_IMAGES = ["Fake1.jpg", "Fake2.jpg", "Fake3.jpg", "Fake4.jpg", "Fake5.jpg"]

_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _load_model(device: torch.device) -> torch.nn.Module:
    checkpoint = load_model_checkpoint()
    model = DeepfakeDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _scale_bbox_to_tensor(
    bbox: tuple[int, int, int, int], orig_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    orig_w, orig_h = orig_size
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * 224 / orig_w),
        int(y1 * 224 / orig_h),
        int(x2 * 224 / orig_w),
        int(y2 * 224 / orig_h),
    )


def _compose_triptych(
    original: Image.Image,
    gradcam_overlay: Image.Image,
    layercam_overlay: Image.Image,
    title_height: int = 28,
) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    tile_w, tile_h = original.size
    canvas = Image.new("RGB", (tile_w * 3, tile_h + title_height), "white")
    canvas.paste(original, (0, title_height))
    canvas.paste(gradcam_overlay, (tile_w, title_height))
    canvas.paste(layercam_overlay, (tile_w * 2, title_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, label in enumerate(["Original", "Grad-CAM (features[-1])", "LayerCAM (fused -2, -1)"]):
        draw.text((tile_w * idx + 8, 4), label, fill="black", font=font)

    return canvas


def compare_image(
    model: torch.nn.Module,
    mapper: FaceCategoryMapper,
    device: torch.device,
    image_path: Path,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    tensor = _preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        target_class = int(torch.argmax(probs).item())

    face_bbox = mapper.detect_face_bbox(image)
    tensor_bbox = _scale_bbox_to_tensor(face_bbox, image.size) if face_bbox else None

    gen_gradcam = GradCAMGenerator(model, method="gradcam")
    gen_layercam = GradCAMGenerator(model, method="layercam")

    heatmap_gradcam = gen_gradcam.generate(tensor, target_class=target_class, face_bbox=tensor_bbox)
    heatmap_layercam = gen_layercam.generate(
        tensor, target_class=target_class, face_bbox=tensor_bbox
    )

    overlay_gradcam = gen_gradcam.create_overlay(image, heatmap_gradcam)
    overlay_layercam = gen_layercam.create_overlay(image, heatmap_layercam)

    triptych = _compose_triptych(image, overlay_gradcam, overlay_layercam)
    triptych.save(output_path, format="PNG")

    gradcam_peak_area = float(np.sum(heatmap_gradcam > 0.5)) / heatmap_gradcam.size
    layercam_peak_area = float(np.sum(heatmap_layercam > 0.5)) / heatmap_layercam.size
    print(
        f"  {image_path.name}: "
        f"pred={'fake' if target_class == 0 else 'real'} "
        f"p={probs[target_class].item():.3f}  "
        f"gradcam>0.5 area={gradcam_peak_area:.3f}  "
        f"layercam>0.5 area={layercam_peak_area:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = _load_model(device)
    mapper = FaceCategoryMapper()

    print(f"Rendering {len(args.images)} comparison(s) into {args.output}")
    for name in args.images:
        image_path = args.input / name
        if not image_path.exists():
            print(f"  SKIP {name}: not found at {image_path}")
            continue
        output_path = args.output / f"{image_path.stem}_cam_comparison.png"
        compare_image(model, mapper, device, image_path, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
