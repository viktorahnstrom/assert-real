"""
Pixel-accurate face parsing using BiSeNet trained on CelebAMask-HQ.

Produces per-pixel semantic masks for 19 face classes (plus background) at the
resolution of the input image.  Used by the face_category_mapper as a more
accurate alternative to MediaPipe-landmark–based region assignment: instead of
counting landmarks that fall inside a GradCAM bbox, the mapper can measure
pixel-area overlap between the bbox and each UI category mask.

The underlying model is the BiSeNet from zllrunning/face-parsing.PyTorch,
ported and distributed by xinntao/facexlib.  Weights auto-download on first
use into ``.venv/Lib/site-packages/facexlib/weights/parsing_bisenet.pth``
(~50 MB).

Output class ordering follows the CelebAMask-HQ convention:

    0: background  1: skin      2: l_brow  3: r_brow   4: l_eye    5: r_eye
    6: eye_g       7: l_ear     8: r_ear   9: ear_r   10: nose    11: mouth
   12: u_lip      13: l_lip    14: neck   15: neck_l  16: cloth   17: hair
   18: hat

References:
    Yu et al. (2018) "BiSeNet: Bilateral Segmentation Network for Real-time
    Semantic Segmentation". Lee et al. (2020) "MaskGAN: Towards Diverse and
    Interactive Facial Image Manipulation" (CelebAMask-HQ dataset).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


# BiSeNet / CelebAMask-HQ 19-class label ordering.
BISENET_CLASSES: tuple[str, ...] = (
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
)


# Merge rule: fine BiSeNet class name → UI category id (see FACE_CATEGORIES).
# Classes not listed (background, hat, neck_l) are intentionally dropped.
_FINE_TO_UI: dict[str, str] = {
    "l_eye": "eyes_pupils",
    "r_eye": "eyes_pupils",
    "eye_g": "eyes_pupils",
    "l_brow": "eyebrows_eyelashes",
    "r_brow": "eyebrows_eyelashes",
    "u_lip": "mouth_teeth",
    "l_lip": "mouth_teeth",
    "mouth": "mouth_teeth",
    "skin": "skin_texture",
    "hair": "hairline_ears",
    "l_ear": "hairline_ears",
    "r_ear": "hairline_ears",
    "ear_r": "hairline_ears",
    "nose": "facial_boundaries",
    "neck": "facial_boundaries",
    "cloth": "facial_boundaries",
}


@dataclass
class FaceParsingResult:
    """Per-pixel face-parsing masks at the original image resolution.

    Attributes:
        masks_fine: Mapping from BiSeNet class name (e.g. ``"l_eye"``) to a
            boolean ``np.ndarray`` of shape ``(H, W)`` in the original image's
            pixel coordinates.  Exactly one mask per :data:`BISENET_CLASSES`
            entry; masks that are entirely absent still appear as all-False.
        masks_ui: Mapping from UI category id (``"eyes_pupils"``, etc.) to the
            union of the corresponding fine masks, same shape and dtype.
        image_size: ``(width, height)`` of the original image in pixels.
    """

    masks_fine: dict[str, np.ndarray]
    masks_ui: dict[str, np.ndarray]
    image_size: tuple[int, int]


class FaceParser:
    """Lazy-initialised wrapper around facexlib's BiSeNet face parser.

    The model is downloaded and loaded on the first call to :meth:`parse`,
    then cached for subsequent calls.  Inference runs on CPU by default; set
    ``FACE_PARSER_DEVICE=cuda`` to use GPU when available.

    Inference is performed at :data:`inference_size` × :data:`inference_size`
    (default 512) and the argmax mask is upsampled with nearest-neighbour to
    the original image resolution.  Lowering the inference size to 256 via
    ``FACE_PARSER_INFERENCE_SIZE`` roughly quarters runtime at the cost of
    mask sharpness near boundaries.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        inference_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.inference_size = inference_size or int(os.getenv("FACE_PARSER_INFERENCE_SIZE", "512"))
        self.device = torch.device(device or os.getenv("FACE_PARSER_DEVICE", "cpu"))
        self._model: torch.nn.Module | None = None
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.inference_size, self.inference_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(self._IMAGENET_MEAN, self._IMAGENET_STD),
            ]
        )

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            from facexlib.parsing import init_parsing_model

            logger.info(
                "Loading BiSeNet face parser (device=%s, inference_size=%d)",
                self.device,
                self.inference_size,
            )
            self._model = init_parsing_model(model_name="bisenet", device=str(self.device))
            self._model.eval()
        return self._model

    def parse(self, image: Image.Image) -> FaceParsingResult:
        """Run face parsing on a PIL image.

        Args:
            image: Any PIL image mode; converted to RGB internally.

        Returns:
            A :class:`FaceParsingResult` whose masks are in the original
            image's pixel coordinates.
        """
        model = self._ensure_model()
        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        input_tensor = self._transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = model(input_tensor)[0]  # (1, 19, H, W) at inference_size
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
            class_map = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        masks_fine: dict[str, np.ndarray] = {
            name: (class_map == idx) for idx, name in enumerate(BISENET_CLASSES)
        }

        masks_ui: dict[str, np.ndarray] = {}
        for fine_name, ui_id in _FINE_TO_UI.items():
            mask = masks_fine[fine_name]
            if ui_id in masks_ui:
                masks_ui[ui_id] = masks_ui[ui_id] | mask
            else:
                masks_ui[ui_id] = mask.copy()

        return FaceParsingResult(
            masks_fine=masks_fine,
            masks_ui=masks_ui,
            image_size=(orig_w, orig_h),
        )
