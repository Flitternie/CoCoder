"""Hybrid image utilities: cross-correlation, convolution, Gaussian blur,
low/high pass filters and hybrid image creation.

Implements the required functions used by the tests:
- cross_correlation_2d
- convolve_2d
- gaussian_blur_kernel_2d
- low_pass
- high_pass
- create_hybrid_image

Only numpy (and optionally cv2) are used.
"""

import numpy as np

def _ensure_3d(img):
    """Return image as HxWxC and a flag if it was originally grayscale."""
    if img.ndim == 2:
        return img[:, :, None], True
    elif img.ndim == 3:
        return img, False
    else:
        raise ValueError("Unsupported image shape: {}".format(img.shape))


def cross_correlation_2d(img, kernel):
    """Compute 2D cross-correlation between image and kernel using reflect padding.

    Supports grayscale (H x W) and color images (H x W x C).
    Kernel must be 2D with odd dimensions.

    Returns an array of the same shape as the input image (dtype float32).
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")

    m, n = kernel.shape
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    # Work in float32 for numerical stability
    img_in = np.asarray(img)
    img_f = img_in.astype(np.float32)

    img3, was_gray = _ensure_3d(img_f)
    H, W, C = img3.shape

    pad_h = m // 2
    pad_w = n // 2

    # Reflect padding
    padded = np.pad(img3, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')

    out = np.empty((H, W, C), dtype=np.float32)

    # Precompute kernel as float32 and with channel axis for broadcasting
    k = np.asarray(kernel, dtype=np.float32)
    k_reshaped = k[:, :, None]

    # Slide kernel over image
    for y in range(H):
        for x in range(W):
            patch = padded[y:y + m, x:x + n, :]
            # elementwise multiply and sum over spatial dims
            # result shape (C,)
            out[y, x, :] = np.sum(patch * k_reshaped, axis=(0, 1))

    if was_gray:
        return out[:, :, 0]
    return out


def convolve_2d(img, kernel):
    """Perform 2D convolution by flipping the kernel and calling cross_correlation_2d."""
    k_flipped = np.flipud(np.fliplr(kernel))
    return cross_correlation_2d(img, k_flipped)


def gaussian_blur_kernel_2d(sigma, width, height):
    """Return a normalized 2D Gaussian kernel of shape (height x width).

    width and height are the number of columns and rows respectively. The
    kernel is normalized so that its entries sum to 1.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    # Create centered coordinate grids
    wx = int(width)
    hy = int(height)
    cx = (wx - 1) / 2.0
    cy = (hy - 1) / 2.0

    x = np.arange(wx) - cx
    y = np.arange(hy) - cy
    xx, yy = np.meshgrid(x, y)

    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    g_sum = g.sum()
    if g_sum == 0:
        # fallback: delta
        g[hy // 2, wx // 2] = 1.0
        g_sum = 1.0
    return (g / g_sum).astype(np.float32)


def low_pass(img, sigma, size):
    """Apply a Gaussian low-pass filter using a square kernel of given size.

    Returns filtered image as float32 with same shape as input.
    """
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel).astype(np.float32)


def high_pass(img, sigma, size):
    """Return the high-pass filtered image: original - low_pass(img)."""
    img_f = np.asarray(img).astype(np.float32)
    low = low_pass(img_f, sigma, size)
    return (img_f - low).astype(np.float32)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    """Create a hybrid image by filtering two images and combining them.

    Parameters:
    - img1, img2: input images (grayscale HxW or color HxWxC). Can be uint8 or
      floating point. If uint8 they are treated as [0,255] and converted to
      float32 for processing.
    - sigma*, size*: Gaussian parameters for filtering.
    - high_low*: 'high' or 'low' specifying which filter to apply to the
      corresponding image.
    - mixin_ratio: float in [0,1] controlling combination: result = mixin_ratio * proc1 + (1-mixin_ratio) * proc2

    Returns the combined image as float32. If inputs were uint8, returned
    values are in range [0,255]. If inputs are float and appear to be in [0,1]
    they are kept in [0,1]. Otherwise float inputs are assumed to be in
    [0,255].
    """
    # Normalize inputs to float32. Detect ranges to choose final clipping.
    a1 = np.asarray(img1)
    a2 = np.asarray(img2)

    if a1.dtype == np.uint8 or a2.dtype == np.uint8:
        # Convert both to [0,1] float32
        f1 = a1.astype(np.float32) / 255.0
        f2 = a2.astype(np.float32) / 255.0
        out_scale = 255.0
        out_clip = (0.0, 255.0)
    else:
        # Floating inputs: check if in [0,1]
        f1 = a1.astype(np.float32)
        f2 = a2.astype(np.float32)
        if f1.min() >= 0.0 and f1.max() <= 1.0 and f2.min() >= 0.0 and f2.max() <= 1.0:
            out_scale = 1.0
            out_clip = (0.0, 1.0)
        else:
            out_scale = 255.0
            out_clip = (0.0, 255.0)

    hl1 = str(high_low1).lower()
    hl2 = str(high_low2).lower()

    if hl1 not in ('low', 'high') or hl2 not in ('low', 'high'):
        raise ValueError("high_low1 and high_low2 must be 'low' or 'high'")

    # Apply filters. Inputs currently are either in [0,1] float32 or arbitrary float.
    proc1 = low_pass(f1, sigma1, size1) if hl1 == 'low' else high_pass(f1, sigma1, size1)
    proc2 = low_pass(f2, sigma2, size2) if hl2 == 'low' else high_pass(f2, sigma2, size2)

    # Combine according to the required formula
    mix = float(mixin_ratio)
    combined = mix * proc1 + (1.0 - mix) * proc2

    # If inputs were originally uint8 we had scaled to [0,1]; produce output in [0,255]
    if out_scale == 255.0:
        combined = combined * 255.0

    # Clip to valid range and return float32
    low_clip, high_clip = out_clip
    combined = np.clip(combined, low_clip, high_clip).astype(np.float32)
    return combined
