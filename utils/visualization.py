"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Reference image / GT overlay color (RGB 0–255).
REF_GT_COLOR = (55, 192, 208)
# Predicted mask overlay color (RGB 0–255).
PRED_COLOR = (225, 122, 203)


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _load_mask(
    mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    size: tuple[int, int],
) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().to("cpu")
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()
        mask_array = mask_array.numpy()
    elif isinstance(mask, np.ndarray):
        mask_array = mask
    elif isinstance(mask, (str, Path)):
        mask_array = np.array(Image.open(mask))
    else:
        mask_array = np.array(mask)

    mask_array = mask_array.squeeze()
    mask_array = mask_array > 0

    if mask_array.shape != size:
        mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255)
        mask_array = np.array(
            mask_image.resize((size[1], size[0]), resample=Image.NEAREST)
        ) > 0

    return mask_array


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_episode_visualizations(
    reference_image: str | Path | Image.Image,
    reference_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    target_image: str | Path | Image.Image,
    target_gt_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    predicted_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    output_dir: str | Path,
    *,
    stem: str,
    iou: float,
    alpha: float = 0.45,
) -> dict[str, Path]:
    """Save ref+mask, target+pred, and target+GT overlays under ``output_dir/vis/``.

    Filenames include the per-episode IoU, e.g.
    ``000042_iou72.35_ref_overlay.png``.
    """
    reference_pil = _load_image(reference_image)
    target_pil = _load_image(target_image)

    reference_np = np.array(reference_pil)
    target_np = np.array(target_pil)

    reference_mask_np = _load_mask(reference_mask, reference_np.shape[:2])
    predicted_mask_np = _load_mask(predicted_mask, target_np.shape[:2])
    target_gt_mask_np = _load_mask(target_gt_mask, target_np.shape[:2])

    reference_overlay = _overlay_mask(
        reference_np, reference_mask_np, REF_GT_COLOR, alpha=alpha
    )
    target_pred_overlay = _overlay_mask(
        target_np, predicted_mask_np, PRED_COLOR, alpha=alpha
    )
    target_gt_overlay = _overlay_mask(
        target_np, target_gt_mask_np, REF_GT_COLOR, alpha=alpha
    )

    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    iou_tag = f"iou{iou:.2f}"
    paths = {
        "ref_overlay": vis_dir / f"{stem}_{iou_tag}_ref_overlay.png",
        "pred_overlay": vis_dir / f"{stem}_{iou_tag}_pred_overlay.png",
        "tgt_gt_overlay": vis_dir / f"{stem}_{iou_tag}_tgt_gt_overlay.png",
    }

    Image.fromarray(reference_overlay).save(paths["ref_overlay"])
    Image.fromarray(target_pred_overlay).save(paths["pred_overlay"])
    Image.fromarray(target_gt_overlay).save(paths["tgt_gt_overlay"])

    return paths


def visualize_prediction(
    reference_image: str | Path | Image.Image,
    reference_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    target_image: str | Path | Image.Image,
    predicted_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    output_path: str | Path,
    *,
    alpha: float = 0.45,
) -> None:
    """Save a side-by-side visualization of reference and target overlays."""
    reference_pil = _load_image(reference_image)
    target_pil = _load_image(target_image)

    reference_np = np.array(reference_pil)
    target_np = np.array(target_pil)

    reference_mask_np = _load_mask(reference_mask, reference_np.shape[:2])
    predicted_mask_np = _load_mask(predicted_mask, target_np.shape[:2])

    reference_overlay = _overlay_mask(
        reference_np, reference_mask_np, REF_GT_COLOR, alpha=alpha
    )
    target_overlay = _overlay_mask(
        target_np, predicted_mask_np, PRED_COLOR, alpha=alpha
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(reference_overlay)
    axes[0].set_title("Reference + Mask")
    axes[1].imshow(target_overlay)
    axes[1].set_title("Target + Prediction")

    for axis in axes:
        axis.axis("off")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
