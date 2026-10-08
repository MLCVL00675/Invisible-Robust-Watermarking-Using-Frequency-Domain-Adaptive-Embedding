"""
Core Cascaded DWT-DCT-SVD Watermarking Pipeline
Implements frequency-domain adaptive embedding and mirrored extraction.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pywt
from scipy.fftpack import dctn, idctn

from .masking import DeterministicTextureMasker


@dataclass
class WatermarkMetadata:
    """Stores decomposition and embedding metadata for non-blind/informed extraction."""
    subband: str
    wavelet: str
    block_size: int
    cover_shape: Tuple[int, int]
    subband_shape: Tuple[int, int]
    grid_shape: Tuple[int, int]
    watermark_shape: Tuple[int, int]
    alpha_matrix: np.ndarray
    original_singular_values: np.ndarray  # Shape: (num_blocks_h, num_blocks_w)
    watermark_binary: np.ndarray


@dataclass
class WatermarkResult:
    """Holds watermarked image and embedding diagnostics."""
    watermarked_image: np.ndarray
    metadata: WatermarkMetadata
    spatial_activity_mask: np.ndarray
    alpha_matrix: np.ndarray


@dataclass
class ExtractionResult:
    """Holds recovered watermark images and continuous estimates."""
    extracted_binary: np.ndarray
    extracted_continuous: np.ndarray
    threshold_used: float


class CascadedDwtDctSvdWatermarker:
    """
    Unified Cascaded DWT-DCT-SVD Watermarker with Deterministic Texture Masking.
    """

    def __init__(
        self,
        wavelet: str = "haar",
        target_subband: str = "HL",
        block_size: int = 8,
        base_alpha: float = 0.20,
        texture_masker: Optional[DeterministicTextureMasker] = None,
    ):
        """
        Args:
            wavelet: Wavelet type for 1-level 2D DWT ('haar', 'db1', 'db2', 'bior1.3', etc.).
            target_subband: Subband for embedding: 'HL' (horizontal high/vertical low) or 'LH'.
            block_size: Block size for DCT/SVD decomposition (default: 8x8).
            base_alpha: Base embedding strength parameter.
            texture_masker: DeterministicTextureMasker instance (or defaults created).
        """
        self.wavelet = wavelet
        self.target_subband = target_subband.upper()
        if self.target_subband not in ("HL", "LH", "HH", "LL"):
            raise ValueError(f"Invalid subband '{target_subband}'. Expected one of ('HL', 'LH', 'HH', 'LL').")
        self.block_size = block_size
        self.base_alpha = base_alpha
        self.masker = texture_masker or DeterministicTextureMasker(base_alpha=base_alpha)

    def _block_dct2(self, block: np.ndarray) -> np.ndarray:
        """Computes 2D DCT with orthonormal scaling."""
        return dctn(block, type=2, norm="ortho")

    def _block_idct2(self, block_dct: np.ndarray) -> np.ndarray:
        """Computes 2D Inverse DCT with orthonormal scaling."""
        return idctn(block_dct, type=2, norm="ortho")

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Validates and converts image to 2D float64 array."""
        if image.ndim == 3:
            # If 3-channel BGR/RGB, convert to grayscale
            import cv2
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        elif image.ndim == 2:
            img_gray = image.copy()
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")

        return img_gray.astype(np.float64)

    def _prepare_watermark(
        self, watermark: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Validates, resizes, and binarizes the watermark payload to match block grid shape.
        """
        import cv2

        wm = watermark.copy()
        if wm.ndim == 3:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)

        # Resize if dimensions do not match
        if wm.shape != target_shape:
            wm = cv2.resize(
                wm.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Binarize to {0, 1}
        if wm.dtype == bool:
            wm_bin = wm.astype(np.float64)
        else:
            thresh = (np.max(wm) + np.min(wm)) / 2.0 if np.max(wm) > np.min(wm) else 0.5
            wm_bin = (wm >= thresh).astype(np.float64)

        return wm_bin

    def embed(
        self,
        cover_image: np.ndarray,
        watermark_payload: np.ndarray,
        base_alpha: Optional[float] = None,
    ) -> WatermarkResult:
        """
        Embeds a binary watermark payload into the cover image using cascaded DWT-DCT-SVD
        and deterministic spatial texture masking.

        Args:
            cover_image: Grayscale or color cover image (H, W).
            watermark_payload: Binary watermark image/matrix.
            base_alpha: Optional override for base embedding strength.

        Returns:
            WatermarkResult containing watermarked image and metadata for extraction.
        """
        alpha_base = base_alpha if base_alpha is not None else self.base_alpha
        img = self._prepare_image(cover_image)
        h, w = img.shape

        # Step 1: 1-level 2D Discrete Wavelet Transform
        coeffs = pywt.dwt2(img, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        # Map subband
        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[self.target_subband].copy()
        sb_h, sb_w = target_sb.shape

        # Step 2: Block Partitioning Dimensions
        bs = self.block_size
        num_blocks_h = sb_h // bs
        num_blocks_w = sb_w // bs
        grid_shape = (num_blocks_h, num_blocks_w)

        if num_blocks_h == 0 or num_blocks_w == 0:
            raise ValueError(f"Subband shape {target_sb.shape} is too small for block size {bs}")

        # Step 3: Deterministic Spatial Activity & Adaptive Alpha Map
        spatial_mask = self.masker.generate_spatial_mask(img)
        alpha_matrix = self.masker.generate_block_alpha_matrix(
            image=img,
            target_subband_shape=(sb_h, sb_w),
            block_size=bs,
            base_alpha=alpha_base,
        )

        # Step 4: Prepare Binary Watermark
        w_bin = self._prepare_watermark(watermark_payload, target_shape=grid_shape)

        # Step 5: Cascaded Block-wise DCT -> SVD -> Adaptive Embedding -> Inverse SVD -> IDCT
        orig_s0_matrix = np.zeros(grid_shape, dtype=np.float64)
        modified_sb = target_sb.copy()

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                r0, r1 = i * bs, (i + 1) * bs
                c0, c1 = j * bs, (j + 1) * bs

                block = target_sb[r0:r1, c0:c1]

                # 2D DCT
                dct_block = self._block_dct2(block)

                # SVD on DCT coefficients: D = U * S * Vt
                U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)
                orig_s0_matrix[i, j] = S[0]

                # Adaptive additive modification of principal singular value:
                # S_wm = S + alpha_k * W_k
                alpha_k = alpha_matrix[i, j]
                bit = w_bin[i, j]

                S_wm = S.copy()
                S_wm[0] = S[0] + (alpha_k * bit)

                # Inverse SVD: D_wm = U * diag(S_wm) * Vt
                dct_wm = np.dot(U * S_wm, Vt)

                # Inverse 2D DCT
                block_wm = self._block_idct2(dct_wm)
                modified_sb[r0:r1, c0:c1] = block_wm

        # Step 6: Inverse 2D DWT Synthesis
        subband_dict[self.target_subband] = modified_sb
        recon_coeffs = (
            subband_dict["LL"],
            (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"]),
        )
        watermarked_img = pywt.idwt2(recon_coeffs, self.wavelet)

        # Crop/match exact original shape (handling odd dimension boundary padding)
        watermarked_img = watermarked_img[:h, :w]
        watermarked_img = np.clip(watermarked_img, 0.0, 255.0)

        metadata = WatermarkMetadata(
            subband=self.target_subband,
            wavelet=self.wavelet,
            block_size=bs,
            cover_shape=(h, w),
            subband_shape=(sb_h, sb_w),
            grid_shape=grid_shape,
            watermark_shape=watermark_payload.shape[:2],
            alpha_matrix=alpha_matrix,
            original_singular_values=orig_s0_matrix,
            watermark_binary=w_bin,
        )

        return WatermarkResult(
            watermarked_image=watermarked_img,
            metadata=metadata,
            spatial_activity_mask=spatial_mask,
            alpha_matrix=alpha_matrix,
        )

    @staticmethod
    def deskew_image(
        image: np.ndarray,
        angle_deg: float,
        border_mode: int = 4,  # cv2.BORDER_REFLECT
    ) -> np.ndarray:
        """
        Applies inverse geometric rotation (deskewing) to synchronize an attacked image.
        """
        import cv2
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        inv_rot_mat = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        deskewed = cv2.warpAffine(
            image.astype(np.float32),
            inv_rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        return deskewed.astype(np.float64)

    def extract(
        self,
        attacked_image: np.ndarray,
        metadata: WatermarkMetadata,
        threshold: float = 0.5,
        rotation_sync_angle: Optional[float] = None,
    ) -> ExtractionResult:
        """
        Extracts watermark from a (potentially attacked) watermarked image.

        Extraction Rule:
            W_extracted = (S_attacked - S_original) / alpha

        Args:
            attacked_image: Received watermarked image under possible attack.
            metadata: WatermarkMetadata generated during embedding.
            threshold: Decision threshold for binarization (default: 0.5).
            rotation_sync_angle: Optional known rotation angle to deskew at the detector.

        Returns:
            ExtractionResult containing binarized and continuous extracted watermarks.
        """
        import cv2

        img_att = self._prepare_image(attacked_image)
        h, w = metadata.cover_shape

        # If geometric rotation synchronization is specified, deskew first
        if rotation_sync_angle is not None and abs(rotation_sync_angle) > 1e-4:
            img_att = self.deskew_image(img_att, angle_deg=rotation_sync_angle)

        # Ensure matching shape
        if img_att.shape != (h, w):
            img_att = cv2.resize(img_att, (w, h), interpolation=cv2.INTER_LINEAR)

        # 1-level 2D DWT
        coeffs = pywt.dwt2(img_att, metadata.wavelet)
        LL, (LH, HL, HH) = coeffs

        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[metadata.subband]

        bs = metadata.block_size
        num_blocks_h, num_blocks_w = metadata.grid_shape

        w_extracted_raw = np.zeros(metadata.grid_shape, dtype=np.float64)

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                r0, r1 = i * bs, (i + 1) * bs
                c0, c1 = j * bs, (j + 1) * bs

                block = target_sb[r0:r1, c0:c1]

                # 2D DCT
                dct_block = self._block_dct2(block)

                # SVD: D_att = U_att * S_att * Vt_att
                _, S_att, _ = np.linalg.svd(dct_block, full_matrices=True)

                s_att_0 = S_att[0]
                s_orig_0 = metadata.original_singular_values[i, j]
                alpha_k = metadata.alpha_matrix[i, j]

                if alpha_k > 1e-7:
                    # Inversion rule: W_raw = (S_att - S_orig) / alpha
                    w_raw = (s_att_0 - s_orig_0) / alpha_k
                else:
                    w_raw = 0.0

                w_extracted_raw[i, j] = w_raw

        # Optimal Binarization
        w_bin = (w_extracted_raw >= threshold).astype(np.uint8)

        # Resize to original watermark resolution if it differed from block grid
        if metadata.watermark_shape != metadata.grid_shape:
            w_bin_final = cv2.resize(
                w_bin.astype(np.float32),
                (metadata.watermark_shape[1], metadata.watermark_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            w_cont_final = cv2.resize(
                w_extracted_raw.astype(np.float32),
                (metadata.watermark_shape[1], metadata.watermark_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            w_bin_final = w_bin
            w_cont_final = w_extracted_raw

        return ExtractionResult(
            extracted_binary=w_bin_final,
            extracted_continuous=w_cont_final,
            threshold_used=threshold,
        )

    def extract_with_rotation_search(
        self,
        attacked_image: np.ndarray,
        metadata: WatermarkMetadata,
        angle_range: Tuple[float, float] = (-10.0, 10.0),
        step: float = 0.5,
        threshold: float = 0.5,
    ) -> Tuple[ExtractionResult, float, float]:
        """
        Performs blind geometric synchronization search over candidate rotation angles.
        Returns the best extraction result along with the detected angle and correlation score.
        """
        from .metrics import compute_nc

        best_nc = -1.0
        best_angle = 0.0
        best_res = None

        candidate_angles = np.arange(angle_range[0], angle_range[1] + step / 2.0, step)
        for angle in candidate_angles:
            res = self.extract(
                attacked_image,
                metadata,
                threshold=threshold,
                rotation_sync_angle=float(angle),
            )
            nc = compute_nc(metadata.watermark_binary, res.extracted_binary)
            if nc > best_nc:
                best_nc = nc
                best_angle = float(angle)
                best_res = res

        return best_res, best_angle, best_nc
