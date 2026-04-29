import cv2
import numpy as np
from typing import Optional, Tuple


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize any float or uint8 image to uint8 [0, 255]."""
    if image.dtype == np.uint8:
        return image.copy()
    if np.max(image) <= 1.0:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


class PreprocessingFilters:

    @staticmethod
    def original(image: np.ndarray) -> np.ndarray:
        """Baseline: scale to float32 [0, 1] with no spatial processing."""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: int = 8,
    ) -> np.ndarray:
        """CLAHE in LAB space — boosts local contrast without hue shift."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def gaussian_sharpen(
        image: np.ndarray,
        strength: float = 1.5,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Laplacian-based sharpening — subtracts second-order edges from the image."""
        img = _to_uint8(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel_size)
        laplacian_3ch = np.stack([laplacian] * 3, axis=-1)
        result = img.astype(np.float32) - strength * laplacian_3ch
        return np.clip(result, 0, 255).astype(np.uint8).astype(np.float32) / 255.0

    @staticmethod
    def bilateral(
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        """Edge-preserving smoothing; reduces microscopy shot noise without blurring membranes."""
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def unsharp_mask(
        image: np.ndarray,
        radius: int = 9,
        amount: float = 0.5,
        sigma: float = 10.0,
        threshold: int = 0,
    ) -> np.ndarray:
        """
        Classic unsharp mask with optional edge threshold.
        Only pixels where |original - blurred| > threshold get sharpened,
        preventing noise amplification in flat background regions.
        """
        img = _to_uint8(image)
        blurred = cv2.GaussianBlur(img, (radius, radius), sigma)
        diff = img.astype(np.int16) - blurred.astype(np.int16)
        if threshold > 0:
            low_contrast = np.abs(diff) < threshold
        result = img.astype(np.float32) + amount * diff.astype(np.float32)
        if threshold > 0:
            result[low_contrast] = img[low_contrast].astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8).astype(np.float32) / 255.0

    @staticmethod
    def reinhard_normalize(
        image: np.ndarray,
        target_mean: Tuple[float, float, float] = (148.60, 169.30, 105.97),
        target_std: Tuple[float, float, float] = (41.13, 9.01, 6.67),
    ) -> np.ndarray:
        """
        Reinhard stain-transfer normalization.
        Matches mean/std of each LAB channel to a fixed reference, reducing
        slide-to-slide variability caused by different staining protocols or microscopes.
        Default reference stats derived from a well-stained MGG blood smear.
        """
        img = _to_uint8(image)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        channels = cv2.split(lab)
        normalized = []
        for ch, tgt_mean, tgt_std in zip(channels, target_mean, target_std):
            src_mean = ch.mean()
            src_std = ch.std() + 1e-6
            normalized.append(np.clip((ch - src_mean) * (tgt_std / src_std) + tgt_mean, 0, 255))
        lab_norm = cv2.merge(normalized).astype(np.uint8)
        result = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Power-law intensity transform via LUT.
        gamma < 1 brightens underexposed smears; gamma > 1 compresses highlights.
        Useful for normalizing slides captured under varying illumination levels.
        """
        img = _to_uint8(image)
        lut = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0,
            0, 255,
        ).astype(np.uint8)
        result = cv2.LUT(img, lut)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def morphological_tophat(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        White top-hat minus black bottom-hat enhancement.
        Top-hat recovers bright granules (eosinophilic granules, cytoplasm detail);
        bottom-hat recovers dark nuclei features. Together they separate cell
        substructures from a slowly varying background more cleanly than linear sharpening.
        """
        img = _to_uint8(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        result = np.clip(img.astype(np.int16) + tophat.astype(np.int16) - blackhat.astype(np.int16), 0, 255).astype(np.uint8)
        return result.astype(np.float32) / 255.0

    @staticmethod
    def color_balance(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
        """
        Per-channel percentile stretch (robust gray-world variant).
        Compensates for white-balance drift and dye batch differences across slides
        by remapping each channel's dynamic range independently.
        """
        img = _to_uint8(image).astype(np.float32)
        for i in range(3):
            p_lo = np.percentile(img[:, :, i], low_pct)
            p_hi = np.percentile(img[:, :, i], high_pct)
            img[:, :, i] = np.clip((img[:, :, i] - p_lo) / (p_hi - p_lo + 1e-6) * 255.0, 0, 255)
        return img.astype(np.float32) / 255.0

    @staticmethod
    def medical_enhanced(image: np.ndarray) -> np.ndarray:
        """Apply robust contrast normalization and edge-aware enhancement for smear images."""
        # Percentile clipping stabilizes exposure differences across microscopes.
        normalized = image.copy().astype(np.float32)
        for i in range(3):
            channel = normalized[:, :, i]
            p2, p98 = np.percentile(channel, (2, 98))
            normalized[:, :, i] = np.clip(
                (channel - p2) / (p98 - p2 + 1e-6) * 255, 0, 255
            )
        normalized = normalized.astype(np.uint8)

        # CLAHE in LAB space improves local contrast without shifting hue too much.
        lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Bilateral filter denoises while preserving boundaries.
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpening_kernel)

        # Edge-weighted blending avoids over-sharpening smooth regions.
        mask = edges.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.stack([mask] * 3, axis=-1)

        result = (enhanced * (1 - mask * 0.3) + sharpened * mask * 0.3).astype(np.uint8)

        return result.astype(np.float32) / 255.0

    @staticmethod
    def estimate_foreground_mask(image: np.ndarray) -> np.ndarray:
        """
        Estimate a soft foreground mask for the leukocyte region.
        The mask is only for XAI visualization and does not affect model predictions.
        """

        if image is None or image.size == 0:
            return np.ones((224, 224), dtype=np.float32)

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_u8 = image.copy()

        hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]

        sat_thr = max(20, int(np.percentile(saturation, 60)))
        val_thr = min(245, int(np.percentile(value, 85)))
        non_white = ((saturation > sat_thr) | (value < val_thr)).astype(np.uint8) * 255

        a_thr = int(np.percentile(a_channel, 60))
        b_thr = int(np.percentile(b_channel, 45))
        nucleus_hint = ((a_channel > a_thr) & (b_channel < b_thr)).astype(np.uint8) * 255

        mask = cv2.bitwise_or(non_white, nucleus_hint)

        # Morphological cleanup removes small background speckles.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            h, w = mask.shape[:2]
            if area > 0.01 * h * w:
                clean = np.zeros_like(mask)
                cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
                mask = clean

        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=6, sigmaY=6)
        max_mask = np.max(soft_mask)
        if max_mask > 0:
            soft_mask = soft_mask / max_mask

        # If foreground estimation fails, return an all-ones mask to avoid hiding XAI.
        coverage = float(np.mean(soft_mask))
        if coverage < 0.05:
            return np.ones(mask.shape[:2], dtype=np.float32)

        return np.clip(soft_mask, 0.0, 1.0).astype(np.float32)
