"""
Quantitative Benchmarking Metrics Module
Computes Imperceptibility (PSNR, SSIM, MSE) and Robustness (BER, NC, BCR) metrics.
"""

from typing import Dict, Any, List
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def compute_mse(img_orig: np.ndarray, img_mod: np.ndarray) -> float:
    """Computes Mean Squared Error between two images."""
    o = img_orig.astype(np.float64)
    m = img_mod.astype(np.float64)
    return float(np.mean((o - m) ** 2))


def compute_psnr(
    img_orig: np.ndarray, img_mod: np.ndarray, max_val: float = 255.0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = compute_mse(img_orig, img_mod)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / mse))


def compute_ssim(
    img_orig: np.ndarray, img_mod: np.ndarray, max_val: float = 255.0
) -> float:
    """
    Computes Structural Similarity Index (SSIM).
    """
    o = img_orig.astype(np.float64)
    m = img_mod.astype(np.float64)
    val = ssim_fn(o, m, data_range=max_val)
    return float(val)


def compute_ber(watermark_orig: np.ndarray, watermark_ext: np.ndarray) -> float:
    """
    Computes Bit Error Rate (BER) between binary watermarks.

    Formula:
        BER = sum(|W_orig - W_ext|) / N_total
    """
    wo = (watermark_orig >= 0.5).astype(np.int32)
    we = (watermark_ext >= 0.5).astype(np.int32)
    diff = np.abs(wo - we)
    return float(np.mean(diff))


def compute_bcr(watermark_orig: np.ndarray, watermark_ext: np.ndarray) -> float:
    """
    Computes Bit Correct Rate (BCR) as a percentage: (1 - BER) * 100%.
    """
    ber = compute_ber(watermark_orig, watermark_ext)
    return float((1.0 - ber) * 100.0)


def compute_nc(watermark_orig: np.ndarray, watermark_ext: np.ndarray) -> float:
    """
    Computes Normalized Cross-Correlation (NC).

    Formula:
        NC = sum(W_orig * W_ext) / sqrt(sum(W_orig^2) * sum(W_ext^2))
    """
    wo = watermark_orig.astype(np.float64).ravel()
    we = watermark_ext.astype(np.float64).ravel()

    numerator = np.sum(wo * we)
    denominator = np.sqrt(np.sum(wo**2) * np.sum(we**2))

    if denominator < 1e-12:
        return 1.0 if np.array_equal(wo, we) else 0.0

    return float(np.clip(numerator / denominator, -1.0, 1.0))


def evaluate_watermarking(
    cover_image: np.ndarray,
    watermarked_image: np.ndarray,
    attacked_image: np.ndarray,
    original_watermark: np.ndarray,
    extracted_watermark: np.ndarray,
    attack_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Evaluates complete set of imperceptibility and robustness metrics for an experiment.
    """
    # Imperceptibility (Cover vs Watermarked)
    psnr_wm = compute_psnr(cover_image, watermarked_image)
    ssim_wm = compute_ssim(cover_image, watermarked_image)
    mse_wm = compute_mse(cover_image, watermarked_image)

    # Attack Distortion (Cover vs Attacked)
    psnr_att = compute_psnr(cover_image, attacked_image)
    ssim_att = compute_ssim(cover_image, attacked_image)

    # Robustness (Original Watermark vs Extracted Watermark)
    ber = compute_ber(original_watermark, extracted_watermark)
    bcr = compute_bcr(original_watermark, extracted_watermark)
    nc = compute_nc(original_watermark, extracted_watermark)

    return {
        "attack_name": attack_name,
        "psnr_watermarked_db": psnr_wm,
        "ssim_watermarked": ssim_wm,
        "mse_watermarked": mse_wm,
        "psnr_attacked_db": psnr_att,
        "ssim_attacked": ssim_att,
        "ber": ber,
        "bcr_pct": bcr,
        "nc": nc,
    }


def format_benchmark_table_markdown(results: List[Dict[str, Any]]) -> str:
    """
    Generates a clean GitHub-flavored Markdown table from benchmark evaluation records.
    """
    header = (
        "| Attack Simulation | Image PSNR (dB) | Image SSIM | Watermark NC | Watermark BER | BCR (%) | Robustness Status |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    rows = []
    for r in results:
        att = r["attack_name"]
        psnr = f"{r['psnr_attacked_db']:.2f}" if r['psnr_attacked_db'] != float("inf") else "Inf"
        ssim = f"{r['ssim_attacked']:.4f}"
        nc = f"{r['nc']:.4f}"
        ber = f"{r['ber']:.4f}"
        bcr = f"{r['bcr_pct']:.2f}%"

        # Determine robustness rating
        nc_val = r["nc"]
        if nc_val >= 0.95:
            status = "Excellent"
        elif nc_val >= 0.85:
            status = "High"
        elif nc_val >= 0.70:
            status = "Moderate"
        else:
            status = "Degraded"

        rows.append(f"| **{att}** | {psnr} | {ssim} | **{nc}** | {ber} | {bcr} | `{status}` |")

    return header + "\n".join(rows) + "\n"
