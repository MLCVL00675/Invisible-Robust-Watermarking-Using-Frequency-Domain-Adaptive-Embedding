"""
Invisible Robust Watermarking Framework
Cascaded DWT-DCT-SVD with Deterministic Texture Masking
"""

from .masking import DeterministicTextureMasker
from .core import CascadedDwtDctSvdWatermarker, WatermarkResult, ExtractionResult
from .attacks import AttackSuite
from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_mse,
    compute_ber,
    compute_nc,
    compute_bcr,
    evaluate_watermarking,
    format_benchmark_table_markdown,
)
from .utils import (
    generate_synthetic_cover_image,
    generate_synthetic_watermark,
    save_benchmark_plots,
)

__all__ = [
    "DeterministicTextureMasker",
    "CascadedDwtDctSvdWatermarker",
    "WatermarkResult",
    "ExtractionResult",
    "AttackSuite",
    "compute_psnr",
    "compute_ssim",
    "compute_mse",
    "compute_ber",
    "compute_nc",
    "compute_bcr",
    "evaluate_watermarking",
    "format_benchmark_table_markdown",
    "generate_synthetic_cover_image",
    "generate_synthetic_watermark",
    "save_benchmark_plots",
]
