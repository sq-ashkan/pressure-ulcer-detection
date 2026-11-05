"""Aspect-preserving resize + pad to a square canvas, and the exact inverse.

Both models see the photograph the same way: scaled so the long side is `size`,
centred, padded with grey. Predictions are cropped and resized back to the
original resolution before any metric is computed."""
import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(img: np.ndarray, size: int, interp=cv2.INTER_LINEAR, pad_value=PAD_VALUE):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    shape = (size, size) + img.shape[2:]
    canvas = np.full(shape, pad_value, dtype=img.dtype)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (top, left, nh, nw)


def unletterbox(arr: np.ndarray, box, orig_hw, interp=cv2.INTER_LINEAR):
    """Inverse of letterbox for an (S,S) or (S,S,C) array; returns (H,W[,C])."""
    top, left, nh, nw = box
    crop = arr[top:top + nh, left:left + nw]
    h, w = orig_hw
    out = cv2.resize(crop, (w, h), interpolation=interp)
    if arr.ndim == 3 and out.ndim == 2:      # cv2 drops a singleton channel
        out = out[..., None]
    return out
