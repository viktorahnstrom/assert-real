# CAM method qualitative comparison

Side-by-side renders of the XADE deepfake detector's attention under two CAM
methods, both run through `backend/app/services/gradcam_service.py`:

- **Grad-CAM** — Selvaraju et al. 2017, hooked on
  `model.model.features[-1]` (the final conv block of EfficientNet-B4).
- **LayerCAM** — Jiang et al. 2021, fused over `features[-2]` and
  `features[-1]` via `pytorch_grad_cam.LayerCAM` with multi-layer targets.

All five inputs are from `desktop/public/quiz-images/` (current study set,
140k Real/Fake Faces, ~256²). The detector predicts `fake` with p=1.000 on
every one, so CAM differences here are about **localization quality**, not
about correctness. Face-bbox masking (applied before renormalization) is on
for both methods to keep the comparison fair.

Regenerate with:

```bash
python -m backend.scripts.compare_cams
```

## Per-image notes

| Image       | Grad-CAM peak area (>0.5) | LayerCAM peak area (>0.5) | Which looks sharper |
|-------------|---------------------------|---------------------------|---------------------|
| Fake1.jpg   | 0.233                     | 0.152                     | **LayerCAM** — noticeably tighter; Grad-CAM bleeds onto the sky and shoulder. |
| Fake2.jpg   | 0.078                     | 0.074                     | Roughly tied — both focus cleanly on the mouth/chin. |
| Fake3.jpg   | 0.094                     | 0.077                     | **LayerCAM** — shifts peak off the beard onto the cheek, a more forensic-relevant region. |
| Fake4.jpg   | 0.079                     | 0.071                     | Roughly tied — both center on the mouth. |
| Fake5.jpg   | 0.104                     | 0.096                     | **LayerCAM** — marginally tighter around the mouth and nose. |

## Caveats

Both target layers used by LayerCAM (`features[-2]` and `features[-1]`) are
already at the final 7×7 spatial resolution on EfficientNet-B4 with a 224²
input, so multi-layer fusion here combines different channel perspectives
rather than multiple resolutions. Expect modest gains on the existing
256×256 Kaggle images; the improvement should be larger once study images
are upgraded to 1024² FFHQ/StyleGAN3 (Issue 9).

The old Grad-CAM path stays available via `CAM_METHOD=gradcam` for
reproducibility of the thesis baseline.
