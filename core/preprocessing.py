"""OpenCV preprocessing for microscopy images. Independent of the detection
models, so implemented now rather than left as an M0/M1 stub."""

import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)


def equalize(image: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel only, to avoid color shift on
    (near-)grayscale microscopy captures."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def enhance_contrast(image: np.ndarray, alpha: float = 1.2, beta: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Open then close a binary mask to drop speckle noise and fill small
    holes, e.g. after thresholding or before contour extraction."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def preprocess(image: np.ndarray, denoise_on: bool = True, equalize_on: bool = True, contrast_on: bool = False) -> np.ndarray:
    """One-click pipeline wired to the UI's preprocessing toggles."""
    out = image
    if denoise_on:
        out = denoise(out)
    if equalize_on:
        out = equalize(out)
    if contrast_on:
        out = enhance_contrast(out)
    return out
