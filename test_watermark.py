"""
Unit Tests for Cascaded DWT-DCT-SVD Digital Watermarking Framework
"""

import unittest
import numpy as np

from watermark.masking import DeterministicTextureMasker
from watermark.core import CascadedDwtDctSvdWatermarker
from watermark.attacks import AttackSuite
from watermark.metrics import (
    compute_psnr,
    compute_ssim,
    compute_mse,
    compute_ber,
    compute_nc,
    compute_bcr,
    evaluate_watermarking,
)
from watermark.utils import (
    generate_synthetic_cover_image,
    generate_synthetic_watermark,
)


class TestDeterministicTextureMasker(unittest.TestCase):
    def setUp(self):
        self.masker = DeterministicTextureMasker(
            weight_gradient=0.4,
            weight_variance=0.4,
            weight_entropy=0.2,
            base_alpha=20.0,
        )
        self.image = generate_synthetic_cover_image((256, 256), "textured")

    def test_spatial_mask_bounds(self):
        mask = self.masker.generate_spatial_mask(self.image)
        self.assertEqual(mask.shape, (256, 256))
        self.assertTrue(np.all(mask >= 0.0))
        self.assertTrue(np.all(mask <= 1.0))

    def test_block_alpha_matrix(self):
        # Target subband 128x128 with 8x8 blocks -> 16x16 alpha matrix
        alpha_mat = self.masker.generate_block_alpha_matrix(
            image=self.image,
            target_subband_shape=(128, 128),
            block_size=8,
            base_alpha=25.0,
        )
        self.assertEqual(alpha_mat.shape, (16, 16))
        self.assertTrue(np.all(alpha_mat > 0.0))
        # High texture areas should have higher alpha than smooth areas
        self.assertGreater(np.max(alpha_mat), np.min(alpha_mat))


class TestCascadedDwtDctSvdWatermarker(unittest.TestCase):
    def setUp(self):
        self.cover = generate_synthetic_cover_image((256, 256), "textured")
        # For 256x256 image, subband is 128x128, with 8x8 blocks -> 16x16 watermark
        self.watermark = generate_synthetic_watermark((16, 16), "logo")

    def test_roundtrip_zero_attack_hl(self):
        watermarker = CascadedDwtDctSvdWatermarker(
            wavelet="haar", target_subband="HL", block_size=8, base_alpha=30.0
        )
        res = watermarker.embed(self.cover, self.watermark)
        self.assertEqual(res.watermarked_image.shape, (256, 256))

        # Imperceptibility check
        psnr = compute_psnr(self.cover, res.watermarked_image)
        ssim = compute_ssim(self.cover, res.watermarked_image)
        self.assertGreater(psnr, 40.0)
        self.assertGreater(ssim, 0.98)

        # Extraction check
        ext = watermarker.extract(res.watermarked_image, res.metadata)
        ber = compute_ber(self.watermark, ext.extracted_binary)
        nc = compute_nc(self.watermark, ext.extracted_binary)

        self.assertEqual(ber, 0.0)
        self.assertAlmostEqual(nc, 1.0, places=4)

    def test_roundtrip_zero_attack_ll(self):
        watermarker = CascadedDwtDctSvdWatermarker(
            wavelet="haar", target_subband="LL", block_size=8, base_alpha=30.0
        )
        res = watermarker.embed(self.cover, self.watermark)
        ext = watermarker.extract(res.watermarked_image, res.metadata)

        ber = compute_ber(self.watermark, ext.extracted_binary)
        nc = compute_nc(self.watermark, ext.extracted_binary)

        self.assertEqual(ber, 0.0)
        self.assertAlmostEqual(nc, 1.0, places=4)

    def test_wavelet_compatibility(self):
        for wname in ["db1", "bior1.3"]:
            watermarker = CascadedDwtDctSvdWatermarker(
                wavelet=wname, target_subband="HL", block_size=8, base_alpha=25.0
            )
            res = watermarker.embed(self.cover, self.watermark)
            ext = watermarker.extract(res.watermarked_image, res.metadata)
            nc = compute_nc(self.watermark, ext.extracted_binary)
            self.assertGreaterEqual(nc, 0.99)


class TestAttackSuiteAndRobustness(unittest.TestCase):
    def setUp(self):
        self.cover = generate_synthetic_cover_image((256, 256), "textured")
        self.watermark = generate_synthetic_watermark((16, 16), "logo")
        self.watermarker = CascadedDwtDctSvdWatermarker(
            wavelet="haar", target_subband="LL", block_size=8, base_alpha=40.0
        )
        self.wm_res = self.watermarker.embed(self.cover, self.watermark)

    def test_jpeg_attack_resilience(self):
        att_img = AttackSuite.jpeg_compression(self.wm_res.watermarked_image, quality=90)
        ext = self.watermarker.extract(att_img, self.wm_res.metadata)
        nc = compute_nc(self.watermark, ext.extracted_binary)
        self.assertGreater(nc, 0.80)

    def test_median_filter_resilience(self):
        att_img = AttackSuite.median_filter(self.wm_res.watermarked_image, kernel_size=3)
        ext = self.watermarker.extract(att_img, self.wm_res.metadata)
        nc = compute_nc(self.watermark, ext.extracted_binary)
        self.assertGreater(nc, 0.80)

    def test_cropping_resilience(self):
        att_img = AttackSuite.cropping(self.wm_res.watermarked_image, crop_ratio=0.10)
        ext = self.watermarker.extract(att_img, self.wm_res.metadata)
        nc = compute_nc(self.watermark, ext.extracted_binary)
        self.assertGreater(nc, 0.80)

    def test_rotation_resilience(self):
        # Genuine rotation attack produces a tilted image
        att_img = AttackSuite.rotation(self.wm_res.watermarked_image, angle_deg=2.0)
        self.assertEqual(att_img.shape, (256, 256))

        # Detector-side synchronization deskewing
        ext_sync = self.watermarker.extract(att_img, self.wm_res.metadata, rotation_sync_angle=2.0)
        nc_sync = compute_nc(self.watermark, ext_sync.extracted_binary)
        self.assertGreater(nc_sync, 0.90)

        # Blind auto-rotation search
        ext_auto, detected_angle, auto_nc = self.watermarker.extract_with_rotation_search(
            att_img, self.wm_res.metadata, angle_range=(-5.0, 5.0), step=0.5
        )
        self.assertAlmostEqual(detected_angle, 2.0, delta=0.5)
        self.assertGreater(auto_nc, 0.90)


class TestMetrics(unittest.TestCase):
    def test_identical_metrics(self):
        arr = np.random.randint(0, 256, (64, 64)).astype(np.float64)
        self.assertEqual(compute_mse(arr, arr), 0.0)
        self.assertEqual(compute_psnr(arr, arr), float("inf"))
        self.assertAlmostEqual(compute_ssim(arr, arr), 1.0, places=4)

    def test_binary_metrics(self):
        w1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        w2 = np.array([[1, 0], [1, 1]], dtype=np.uint8)  # 1 bit error out of 4
        self.assertEqual(compute_ber(w1, w2), 0.25)
        self.assertEqual(compute_bcr(w1, w2), 75.0)
        self.assertGreater(compute_nc(w1, w2), 0.70)


if __name__ == "__main__":
    unittest.main()
