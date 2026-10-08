"""
Main Demonstration & Benchmarking CLI for Adaptive Digital Watermarking
Cascaded DWT-DCT-SVD with Deterministic Texture Masking.
"""

import os
import sys
import argparse
from typing import Dict, List, Any
import numpy as np
import cv2

from watermark.core import CascadedDwtDctSvdWatermarker
from watermark.masking import DeterministicTextureMasker
from watermark.attacks import AttackSuite
from watermark.metrics import evaluate_watermarking, format_benchmark_table_markdown
from watermark.utils import (
    generate_synthetic_cover_image,
    generate_synthetic_watermark,
    save_benchmark_plots,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invisible Robust Watermarking: Cascaded DWT-DCT-SVD with Deterministic Texture Masking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cover",
        type=str,
        default=None,
        help="Path to custom cover image (PNG, JPG, BMP). If None, generates synthetic pattern.",
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        help="Path to custom binary watermark image. If None, generates synthetic logo.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["textured", "medical_phantom", "geometric"],
        default="textured",
        help="Type of synthetic pattern to generate when --cover is not specified.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="haar",
        help="Wavelet transform filter ('haar', 'db1', 'db2', 'bior1.3').",
    )
    parser.add_argument(
        "--subband",
        type=str,
        choices=["HL", "LH", "LL", "HH"],
        default="HL",
        help="Target DWT subband for embedding (mid-frequency: HL, LH; low-frequency: LL).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=30.0,
        help="Base embedding strength scaling factor.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=8,
        help="Block size for 2D DCT and SVD (typically 8x8).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save benchmark outputs and plots.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save visual multi-panel comparative figures.",
    )
    parser.add_argument(
        "--compare_subbands",
        action="store_true",
        help="Run comparative benchmark across both mid-frequency (HL/LH) and low-frequency (LL) subbands.",
    )
    return parser.parse_args()


def load_or_create_inputs(args) -> tuple[np.ndarray, np.ndarray]:
    """Loads input images from disk or generates synthetic test images."""
    if args.cover is not None and os.path.isfile(args.cover):
        print(f"[*] Loading cover image from: {args.cover}")
        cover = cv2.imread(args.cover, cv2.IMREAD_GRAYSCALE)
        if cover is None:
            raise FileNotFoundError(f"Could not load cover image from {args.cover}")
        # Resize to power of 2 for clean DWT if needed
        h, w = cover.shape
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        if (new_h, new_w) != (h, w):
            cover = cv2.resize(cover, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        print(f"[*] Generating synthetic cover image (Pattern: '{args.pattern}', Size: 512x512)...")
        cover = generate_synthetic_cover_image((512, 512), pattern_type=args.pattern)

    # Watermark size depends on subband blocks: (H / 2 / block_size, W / 2 / block_size)
    grid_h = (cover.shape[0] // 2) // args.block_size
    grid_w = (cover.shape[1] // 2) // args.block_size

    if args.watermark is not None and os.path.isfile(args.watermark):
        print(f"[*] Loading watermark from: {args.watermark}")
        wm = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
        if wm is None:
            raise FileNotFoundError(f"Could not load watermark from {args.watermark}")
        wm = cv2.resize(wm, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        thresh = (np.max(wm) + np.min(wm)) / 2.0
        wm = (wm >= thresh).astype(np.uint8)
    else:
        print(f"[*] Generating synthetic binary watermark logo (Size: {grid_h}x{grid_w})...")
        wm = generate_synthetic_watermark((grid_h, grid_w), pattern_type="logo")

    return cover, wm


def run_benchmark_for_subband(
    cover: np.ndarray,
    watermark: np.ndarray,
    subband: str,
    args,
) -> tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, np.ndarray], Any]:
    """Executes embedding, attack suite, extraction, and metrics computation for a chosen subband."""
    print(f"\n=======================================================")
    print(f" Running Benchmark for Subband: {subband} (alpha={args.alpha}, wavelet={args.wavelet})")
    print(f"=======================================================")

    masker = DeterministicTextureMasker(
        weight_gradient=0.40,
        weight_variance=0.40,
        weight_entropy=0.20,
        base_alpha=args.alpha,
    )

    watermarker = CascadedDwtDctSvdWatermarker(
        wavelet=args.wavelet,
        target_subband=subband,
        block_size=args.block_size,
        base_alpha=args.alpha,
        texture_masker=masker,
    )

    # 1. Embedding
    wm_result = watermarker.embed(cover, watermark)
    watermarked_img = wm_result.watermarked_image

    # 2. Attacks & Extraction
    attacks = {
        "No Attack": AttackSuite.no_attack,
        "JPEG Compression (Q=90)": lambda img: AttackSuite.jpeg_compression(img, quality=90),
        "JPEG Compression (Q=70)": lambda img: AttackSuite.jpeg_compression(img, quality=70),
        "JPEG Compression (Q=50)": lambda img: AttackSuite.jpeg_compression(img, quality=50),
        "Gaussian Noise (var=0.01)": lambda img: AttackSuite.gaussian_noise(img, variance=0.01),
        "Speckle Noise (var=0.01)": lambda img: AttackSuite.speckle_noise(img, variance=0.01),
        "Salt & Pepper Noise (1%)": lambda img: AttackSuite.salt_and_pepper_noise(img, amount=0.01),
        "Median Filter (3x3)": lambda img: AttackSuite.median_filter(img, kernel_size=3),
        "Gaussian Blur (3x3, s=1.0)": lambda img: AttackSuite.gaussian_blur(img, kernel_size=3, sigma=1.0),
        "Border Cropping (10%)": lambda img: AttackSuite.cropping(img, crop_ratio=0.10, fill_mode="black"),
        "Rotation (2 deg) [Unsync]": lambda img: AttackSuite.rotation(img, angle_deg=2.0),
        "Rotation (2 deg) [Sync Deskew]": lambda img: AttackSuite.rotation(img, angle_deg=2.0),
        "Rotation (5 deg) [Sync Deskew]": lambda img: AttackSuite.rotation(img, angle_deg=5.0),
        "Contrast Adjust (x1.15)": lambda img: AttackSuite.contrast_adjustment(img, alpha=1.15, beta=5.0),
    }

    records = []
    attacked_imgs = {}
    extracted_wms = {}

    for name, attack_fn in attacks.items():
        att_img = attack_fn(watermarked_img)

        # Handle detector synchronization
        if "Sync Deskew" in name and "2 deg" in name:
            ext_res = watermarker.extract(att_img, wm_result.metadata, rotation_sync_angle=2.0)
        elif "Sync Deskew" in name and "5 deg" in name:
            ext_res = watermarker.extract(att_img, wm_result.metadata, rotation_sync_angle=5.0)
        else:
            ext_res = watermarker.extract(att_img, wm_result.metadata)

        rec = evaluate_watermarking(
            cover_image=cover,
            watermarked_image=watermarked_img,
            attacked_image=att_img,
            original_watermark=watermark,
            extracted_watermark=ext_res.extracted_binary,
            attack_name=name,
        )
        records.append(rec)
        attacked_imgs[name] = att_img
        extracted_wms[name] = ext_res.extracted_binary

    return records, attacked_imgs, extracted_wms, wm_result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("================================================================================")
    print(" Invisible Robust Watermarking: Cascaded DWT-DCT-SVD with Deterministic Masking")
    print("================================================================================")

    cover, watermark = load_or_create_inputs(args)

    # Target subbands
    subbands_to_run = ["HL", "LL"] if args.compare_subbands else [args.subband]
    all_reports = []

    for sb in subbands_to_run:
        records, attacked_imgs, extracted_wms, wm_res = run_benchmark_for_subband(
            cover=cover,
            watermark=watermark,
            subband=sb,
            args=args,
        )

        md_table = format_benchmark_table_markdown(records)
        print("\n" + md_table)

        # Baseline summary
        baseline = records[0]
        summary_text = (
            f"### Subband: {sb}\n"
            f"- **Cover Image Size**: {cover.shape[0]}x{cover.shape[1]}\n"
            f"- **Watermark Size**: {watermark.shape[0]}x{watermark.shape[1]} bits\n"
            f"- **Base Alpha**: {args.alpha}\n"
            f"- **Baseline PSNR**: {baseline['psnr_watermarked_db']:.2f} dB\n"
            f"- **Baseline SSIM**: {baseline['ssim_watermarked']:.4f}\n\n"
            + md_table
        )
        all_reports.append(summary_text)

        # Save visual figure
        if args.save_plots:
            plot_path = os.path.join(args.output_dir, f"benchmark_{sb.lower()}_visualization.png")
            print(f"[*] Saving visualization plot to: {plot_path}")
            save_benchmark_plots(
                cover_image=cover,
                watermarked_image=wm_res.watermarked_image,
                spatial_mask=wm_res.spatial_activity_mask,
                alpha_matrix=wm_res.alpha_matrix,
                original_watermark=watermark,
                attack_results=records,
                attacked_images=attacked_imgs,
                extracted_watermarks=extracted_wms,
                output_filepath=plot_path,
            )

    # Save overall markdown report
    report_file = os.path.join(args.output_dir, "benchmark_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Digital Watermarking Benchmark Report\n")
        f.write("## Cascaded DWT-DCT-SVD with Deterministic Texture Masking\n\n")
        f.write("\n---\n\n".join(all_reports))
    print(f"\n[OK] Benchmark Report saved to: {report_file}")
    print("[OK] Execution finished successfully.")


if __name__ == "__main__":
    main()
