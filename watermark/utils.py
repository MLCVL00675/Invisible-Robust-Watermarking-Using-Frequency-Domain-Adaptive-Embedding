"""
Utilities Module: Synthetic image generators, visualization, and plotting helpers.
"""

from typing import Tuple, Dict, Any, List, Optional
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_synthetic_cover_image(
    size: Tuple[int, int] = (512, 512), pattern_type: str = "textured"
) -> np.ndarray:
    """
    Generates synthetic high-resolution test cover images with diverse texture profiles.

    Args:
        size: (height, width) tuple.
        pattern_type: 'textured', 'medical_phantom', or 'geometric'.
    """
    h, w = size
    y, x = np.mgrid[0:h, 0:w]

    if pattern_type == "textured":
        # Smooth background gradient
        base = 128.0 + 40.0 * np.sin(x / 40.0) + 30.0 * np.cos(y / 40.0)

        # High-frequency texture zones (simulating hair, fabric, foliage)
        hf_texture = 25.0 * np.sin(x * 0.3) * np.sin(y * 0.3)
        hf_mask = (np.sin(x / 60.0) * np.cos(y / 60.0) > 0.0).astype(np.float64)

        # Concentric rings / edges
        cx, cy = w // 2, h // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rings = 35.0 * np.sin(r / 12.0)

        # Smooth flat patch in top-left
        flat_mask = np.zeros((h, w), dtype=np.float64)
        flat_mask[50:150, 50:150] = 1.0

        img = base + (hf_texture * hf_mask) + (rings * 0.4)
        img[flat_mask > 0] = 180.0

    elif pattern_type == "medical_phantom":
        # Synthetic MRI / CT brain phantom with skull, ventricles, and soft tissue
        cx, cy = w // 2, h // 2
        img = np.zeros((h, w), dtype=np.float64)

        # Outer skull ellipse
        skull = (((x - cx) / (w * 0.42)) ** 2 + ((y - cy) / (h * 0.46)) ** 2) <= 1.0
        img[skull] = 100.0

        # Brain parenchyma
        brain = (((x - cx) / (w * 0.38)) ** 2 + ((y - cy) / (h * 0.42)) ** 2) <= 1.0
        # Parenchyma texture
        tissue_texture = 160.0 + 20.0 * np.sin(x / 8.0) * np.cos(y / 8.0)
        img[brain] = tissue_texture[brain]

        # Ventricles (dark core)
        ventricle1 = (((x - (cx - 30)) / 25.0) ** 2 + ((y - cy) / 60.0) ** 2) <= 1.0
        ventricle2 = (((x - (cx + 30)) / 25.0) ** 2 + ((y - cy) / 60.0) ** 2) <= 1.0
        img[ventricle1 | ventricle2] = 40.0

        # Hyper-intense lesion / pathology
        lesion = (((x - (cx + 70)) / 20.0) ** 2 + ((y - (cy - 70)) / 20.0) ** 2) <= 1.0
        img[lesion] = 245.0

    else:
        # Geometric shapes with sharp step edges and smooth ramps
        img = 128.0 * np.ones((h, w), dtype=np.float64)
        cv2.rectangle(img, (w // 8, h // 8), (w // 2, h // 2), 220.0, -1)
        cv2.circle(img, (3 * w // 4, 3 * h // 4), min(h, w) // 5, 40.0, -1)
        cv2.circle(img, (3 * w // 4, h // 4), min(h, w) // 6, 170.0, -1)

    return np.clip(img, 0.0, 255.0).astype(np.float64)


def generate_synthetic_watermark(
    size: Tuple[int, int] = (32, 32), pattern_type: str = "logo"
) -> np.ndarray:
    """
    Generates a crisp binary watermark emblem or logo.

    Args:
        size: (height, width) tuple (e.g. 32x32).
        pattern_type: 'logo', 'qr', or 'checker'.
    """
    h, w = size
    wm = np.zeros((h, w), dtype=np.uint8)

    if pattern_type == "logo":
        # Outer border
        cv2.rectangle(wm, (1, 1), (w - 2, h - 2), 1, 1)
        # Inner 'C' / 'W' emblem
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        cv2.circle(wm, (cx, cy), r, 1, 2)
        cv2.putText(
            wm,
            "W",
            (cx - 7, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
            1,
            cv2.LINE_AA,
        )
    elif pattern_type == "qr":
        # QR-like checkerboard with alignment boxes
        np.random.seed(42)
        wm = (np.random.rand(h, w) > 0.5).astype(np.uint8)
        # Corner position tags (7x7)
        for r_pos, c_pos in [(0, 0), (0, w - 7), (h - 7, 0)]:
            wm[r_pos : r_pos + 7, c_pos : c_pos + 7] = 1
            wm[r_pos + 1 : r_pos + 6, c_pos + 1 : c_pos + 6] = 0
            wm[r_pos + 2 : r_pos + 5, c_pos + 2 : c_pos + 5] = 1
    else:
        # Standard 4x4 checkerboard tiles
        tile_h, tile_w = h // 4, w // 4
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0:
                    wm[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = 1

    return wm


def save_benchmark_plots(
    cover_image: np.ndarray,
    watermarked_image: np.ndarray,
    spatial_mask: np.ndarray,
    alpha_matrix: np.ndarray,
    original_watermark: np.ndarray,
    attack_results: List[Dict[str, Any]],
    attacked_images: Dict[str, np.ndarray],
    extracted_watermarks: Dict[str, np.ndarray],
    output_filepath: str,
):
    """
    Generates and saves a publication-quality multi-panel visualization of the watermarking
    embedding, masking diagnostics, and attack robustness performance.
    """
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    # Row 0: Embedding Diagnostics
    axes[0, 0].imshow(cover_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("1. Original Cover", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")

    psnr_val = attack_results[0]["psnr_watermarked_db"]
    ssim_val = attack_results[0]["ssim_watermarked"]
    axes[0, 1].imshow(watermarked_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"2. Watermarked\nPSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f}", fontsize=8, fontweight="bold")
    axes[0, 1].axis("off")

    diff = np.abs(watermarked_image.astype(np.float64) - cover_image.astype(np.float64)) * 10.0
    axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title("3. Residual (x10)", fontsize=9, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(spatial_mask, cmap="viridis")
    axes[0, 3].set_title("4. Spatial Activity", fontsize=9, fontweight="bold")
    axes[0, 3].axis("off")

    axes[0, 4].imshow(alpha_matrix, cmap="magma")
    axes[0, 4].set_title("5. Block Alpha Map", fontsize=9, fontweight="bold")
    axes[0, 4].axis("off")

    axes[0, 5].imshow(original_watermark, cmap="gray", vmin=0, vmax=1)
    axes[0, 5].set_title("6. Original Watermark", fontsize=9, fontweight="bold")
    axes[0, 5].axis("off")

    # Rows 1 to 3: Attack simulations & Extracted Watermarks
    selected_attacks = [
        "No Attack",
        "JPEG Compression (Q=90)",
        "JPEG Compression (Q=70)",
        "JPEG Compression (Q=50)",
        "Gaussian Noise (var=0.01)",
        "Speckle Noise (var=0.01)",
        "Median Filter (3x3)",
        "Border Cropping (10%)",
        "Rotation (2 deg)",
    ]

    for idx, att_name in enumerate(selected_attacks):
        if idx >= 9:
            break
        row = 1 + (idx // 3)
        col_pair = (idx % 3)

        rec = next((r for r in attack_results if r["attack_name"] == att_name or att_name in r["attack_name"]), None)
        matched_key = next((k for k in attacked_images.keys() if att_name == k or att_name in k), att_name)
        att_img = attacked_images.get(matched_key, None)
        ext_wm = extracted_watermarks.get(matched_key, None)

        if att_img is not None and rec is not None:
            # Attacked Image
            axes[row, col_pair * 2].imshow(att_img, cmap="gray", vmin=0, vmax=255)
            axes[row, col_pair * 2].set_title(f"{att_name}\nPSNR: {rec['psnr_attacked_db']:.1f}dB", fontsize=7.5)
            axes[row, col_pair * 2].axis("off")

            # Extracted Watermark
            axes[row, col_pair * 2 + 1].imshow(ext_wm, cmap="gray", vmin=0, vmax=1)
            axes[row, col_pair * 2 + 1].set_title(f"Extracted WM\nNC: {rec['nc']:.3f} | BER: {rec['ber']:.2f}", fontsize=7.5, fontweight="bold")
            axes[row, col_pair * 2 + 1].axis("off")
        else:
            axes[row, col_pair * 2].axis("off")
            axes[row, col_pair * 2 + 1].axis("off")

    plt.suptitle(
        "Invisible Robust Watermarking: Cascaded DWT-DCT-SVD with Deterministic Texture Masking",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    plt.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
