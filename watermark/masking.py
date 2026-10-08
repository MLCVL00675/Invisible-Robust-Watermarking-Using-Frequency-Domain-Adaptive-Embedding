"""
Deterministic Texture and Activity Masking Module
Calculates spatial-domain Human Visual System (HVS) activity weights using
Sobel gradient magnitude, local variance, and local Shannon entropy.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter


class DeterministicTextureMasker:
    """
    Computes a deterministic, lightweight spatial texture and activity mask
    for adaptive frequency-domain watermark embedding.
    """

    def __init__(
        self,
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        window_size: int = 7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
        base_alpha: float = 0.15,
    ):
        """
        Args:
            weight_gradient: Importance weight for Sobel gradient magnitude.
            weight_variance: Importance weight for local statistical variance.
            weight_entropy: Importance weight for local Shannon entropy.
            window_size: Neighborhood window size for statistical filters.
            alpha_min: Minimum embedding strength multiplier.
            alpha_max: Maximum embedding strength multiplier.
            base_alpha: Global baseline embedding scaling parameter.
        """
        total_w = weight_gradient + weight_variance + weight_entropy
        self.w_grad = weight_gradient / total_w
        self.w_var = weight_variance / total_w
        self.w_ent = weight_entropy / total_w
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.base_alpha = base_alpha

    def compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """Computes normalized Sobel gradient magnitude."""
        # Ensure float32 in [0, 1]
        img_f = image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Normalize to [0, 1]
        g_min, g_max = np.min(grad_mag), np.max(grad_mag)
        if g_max > g_min:
            return (grad_mag - g_min) / (g_max - g_min)
        return np.zeros_like(grad_mag)

    def compute_local_variance(self, image: np.ndarray) -> np.ndarray:
        """Computes normalized local variance via sliding uniform filter."""
        img_f = image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0

        mean = uniform_filter(img_f, size=self.window_size, mode="reflect")
        mean_sq = uniform_filter(img_f**2, size=self.window_size, mode="reflect")
        variance = np.maximum(mean_sq - mean**2, 0.0)
        std_dev = np.sqrt(variance)

        # Normalize to [0, 1]
        s_min, s_max = np.min(std_dev), np.max(std_dev)
        if s_max > s_min:
            return (std_dev - s_min) / (s_max - s_min)
        return np.zeros_like(std_dev)

    def compute_local_entropy(self, image: np.ndarray, num_bins: int = 16) -> np.ndarray:
        """
        Computes fast, deterministic local Shannon entropy map using block histogram analysis.
        """
        img_u8 = (
            (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        )
        h, w = img_u8.shape
        quantized = (img_u8 // (256 // num_bins)).astype(np.uint8)

        # Compute entropy per spatial patch to maintain fast deterministic execution
        patch_size = max(4, self.window_size)
        entropy_map = np.zeros((h, w), dtype=np.float32)

        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        padded = np.pad(quantized, ((0, pad_h), (0, pad_w)), mode="reflect")

        ph, pw = padded.shape
        for r in range(0, ph, patch_size):
            for c in range(0, pw, patch_size):
                patch = padded[r : r + patch_size, c : c + patch_size]
                counts = np.bincount(patch.ravel(), minlength=num_bins)
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                ent = -np.sum(probs * np.log2(probs))
                # Fill local window
                r_end = min(r + patch_size, h)
                c_end = min(c + patch_size, w)
                if r < h and c < w:
                    entropy_map[r:r_end, c:c_end] = ent

        e_min, e_max = np.min(entropy_map), np.max(entropy_map)
        if e_max > e_min:
            return (entropy_map - e_min) / (e_max - e_min)
        return np.zeros_like(entropy_map)

    def generate_spatial_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Fuses Sobel gradient, local variance, and local entropy into a unified
        spatial activity mask in [0, 1].
        """
        grad = self.compute_gradient_magnitude(image)
        var = self.compute_local_variance(image)
        ent = self.compute_local_entropy(image)

        combined = self.w_grad * grad + self.w_var * var + self.w_ent * ent
        c_min, c_max = np.min(combined), np.max(combined)
        if c_max > c_min:
            return (combined - c_min) / (c_max - c_min)
        return combined

    def generate_block_alpha_matrix(
        self,
        image: np.ndarray,
        target_subband_shape: Tuple[int, int],
        block_size: int = 8,
        base_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Pools spatial activity mask into block-level adaptive embedding weights alpha_k.

        Args:
            image: Cover image in spatial domain (H, W).
            target_subband_shape: Shape of the target DWT sub-band (H/2, W/2).
            block_size: DCT block size (typically 8x8).
            base_alpha: Optional override for base embedding strength.

        Returns:
            2D numpy array of shape (num_blocks_h, num_blocks_w) representing alpha per block.
        """
        alpha_base = base_alpha if base_alpha is not None else self.base_alpha
        spatial_mask = self.generate_spatial_mask(image)

        sub_h, sub_w = target_subband_shape
        num_blocks_h = sub_h // block_size
        num_blocks_w = sub_w // block_size

        # In 1-level DWT, subband is H/2 x W/2. Each 8x8 subband block corresponds to a 16x16 spatial patch.
        spatial_block_h = image.shape[0] / num_blocks_h
        spatial_block_w = image.shape[1] / num_blocks_w

        alpha_matrix = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float32)

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                r0 = int(round(i * spatial_block_h))
                r1 = int(round((i + 1) * spatial_block_h))
                c0 = int(round(j * spatial_block_w))
                c1 = int(round((j + 1) * spatial_block_w))

                patch = spatial_mask[r0:r1, c0:c1]
                mean_activity = float(np.mean(patch)) if patch.size > 0 else 0.5

                # Linear mapping to [alpha_min, alpha_max] scaled by alpha_base
                # alpha_k = alpha_base * (scale_factor)
                # Map mean_activity in [0, 1] to dynamic range:
                scaled_weight = self.alpha_min + (self.alpha_max - self.alpha_min) * mean_activity
                alpha_matrix[i, j] = scaled_weight * (alpha_base / ((self.alpha_min + self.alpha_max) / 2.0))

        return alpha_matrix
