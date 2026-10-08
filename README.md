# Invisible Robust Watermarking Using Frequency-Domain Adaptive Embedding

A modular digital image watermarking framework combining Discrete Wavelet Transform (DWT), Discrete Cosine Transform (DCT), Singular Value Decomposition (SVD), and deterministic spatial texture analysis.

---

## 📌 Architecture Overview

The embedding and extraction pipeline operates through the following stages:

1. **Discrete Wavelet Transform (DWT)**: 1-level 2D Haar DWT decomposes the cover image into four subbands: $LL$, $LH$, $HL$, and $HH$.
2. **Deterministic Spatial Texture Masking**: Computes a local Human Visual System (HVS) activity score $T$ using normalized multimodal features:
   $$\text{Texture Score} (T) = 0.40 \cdot G_{\text{norm}} + 0.40 \cdot V_{\text{norm}} + 0.20 \cdot E_{\text{norm}}$$
   where $G$ is the Sobel gradient magnitude, $V$ is local statistical variance, and $E$ is local Shannon entropy.
3. **Block-wise 2D DCT**: Target subband (e.g., $HL$ or $LL$) is partitioned into non-overlapping $8 \times 8$ blocks and transformed via 2D orthonormal DCT.
4. **Singular Value Decomposition (SVD)**: SVD is applied to each DCT block ($D = U S V^T$).
5. **Adaptive Watermark Embedding**: The principal singular value $S(1,1)$ is modified additively according to the block-adaptive strength $\alpha_k$:
   $$S_{\text{wm}}(1,1) = S(1,1) + (\alpha_k \cdot W_k)$$
6. **Reconstruction**: Inverse SVD $\rightarrow$ Inverse 2D Block DCT $\rightarrow$ Inverse 2D DWT.
7. **Extraction**:
   $$W_{\text{raw}} = \frac{S_{\text{att}}(1,1) - S_{\text{orig}}(1,1)}{\alpha_k} \ge 0.5$$

---

## 📂 Repository Structure

```
├── watermark/
│   ├── __init__.py      # Package interface
│   ├── core.py          # Cascaded DWT-DCT-SVD embedding & extraction engine
│   ├── masking.py       # Deterministic multimodal texture analyzer
│   ├── metrics.py       # Quantitative metrics (PSNR, SSIM, NC, BER, BCR)
│   └── utils.py         # Test pattern generators & visualization helpers
├── main.py              # Main CLI execution tool
├── test_watermark.py    # Unit test suite
├── requirements.txt     # Dependency specifications
└── README.md            # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MLCVL00675/Invisible-Robust-Watermarking-Using-Frequency-Domain-Adaptive-Embedding.git
   cd Invisible-Robust-Watermarking-Using-Frequency-Domain-Adaptive-Embedding
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Run Watermarking Demonstration (Synthetic Pattern)
```bash
python main.py --subband HL --alpha 30.0
```

### 2. Run with Custom Cover & Watermark
```bash
python main.py --cover path/to/cover.png --watermark path/to/logo.png --subband HL --alpha 30.0
```

### 3. Run Unit Tests
```bash
python -m unittest test_watermark.py
```

---

## 📊 Evaluation Metrics

- **Imperceptibility**: Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM).
- **Robustness**: Normalized Cross-Correlation (NC), Bit Error Rate (BER), and Bit Correct Rate (BCR).
