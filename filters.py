"""
Test HSV and other filters on image1, image2, image3.
Applies gamma, gaussian, edge detection, and blue-only masking;
shows HSV histograms next to filtered images to tune blue light detection.
"""

import cv2
import numpy as np
from pathlib import Path

# Image paths
IMAGE_NAMES = ["image1.png", "image2.png", "image3.png"]
IMAGE_ENV_NAME = "image_env.png"

# Blue LED HSV — base range (used by apply_blue_mask)
HUE_MIN, HUE_MAX = 78, 128
SAT_MIN, VAL_MIN = 55, 175

# Best filter: tuned to ignore env blues (trash cans) and whites (lights, reflections)
# Tighter hue = emissive cyan–blue only; high sat = no white/gray; high val = no matte blue
BEST_HUE_MIN, BEST_HUE_MAX = 85, 118
BEST_SAT_MIN, BEST_VAL_MIN = 90, 190
# Pixels must be this much brighter than env (or above VAL_ABSOLUTE) to count as a light
ENV_V_DELTA = 18
VAL_ABSOLUTE = 248


def load_images():
    """Load image1, image2, image3 from current dir."""
    base = Path(__file__).resolve().parent
    images = []
    for name in IMAGE_NAMES:
        path = base / name
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: could not read {path}, skipping.")
            continue
        images.append((name, img))
    return images


def load_env():
    """Load image_env.png for background suppression (same scene without / with different lighting)."""
    base = Path(__file__).resolve().parent
    path = base / IMAGE_ENV_NAME
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img


def apply_gamma(img_bgr, gamma=1.2):
    """Gamma correction: brighten midtones to help blue glow stand out."""
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_gaussian(img_bgr, ksize=(5, 5), sigma=1.0):
    """Gaussian blur to reduce noise before masking/edge detection."""
    return cv2.GaussianBlur(img_bgr, ksize, sigma)


def apply_edge_canny(img_bgr, low=50, high=150):
    """Canny edge detection on grayscale."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def apply_blue_mask(img_bgr, hue_min=HUE_MIN, hue_max=HUE_MAX, sat_min=SAT_MIN, val_min=VAL_MIN):
    """HSV inRange mask: only blue/teal bright regions; return BGR with only those visible."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Optional: small morph to clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Black background, only blue regions in color
    out = np.zeros_like(img_bgr)
    out[mask > 0] = img_bgr[mask > 0]
    return out, mask, hsv


def apply_blue_mask_best(img_bgr, img_env_bgr=None):
    """
    Best filter for blue lights: tight HSV (emissive only) + optionally require brighter than env.
    Rejects: matte blue (trash cans), whites/grays (ceiling, floor, walls), faint internal blues.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_current = hsv[:, :, 2]
    lower = np.array([BEST_HUE_MIN, BEST_SAT_MIN, BEST_VAL_MIN])
    upper = np.array([BEST_HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if img_env_bgr is not None:
        if img_env_bgr.shape[:2] != img_bgr.shape[:2]:
            img_env_bgr = cv2.resize(img_env_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        hsv_env = cv2.cvtColor(img_env_bgr, cv2.COLOR_BGR2HSV)
        v_env = hsv_env[:, :, 2].astype(np.int32)
        v_cur = v_current.astype(np.int32)
        # Keep only if noticeably brighter than env (emissive) or extremely bright
        brighter_than_env = (v_cur > v_env + ENV_V_DELTA) | (v_current >= VAL_ABSOLUTE)
        mask = cv2.bitwise_and(mask, np.where(brighter_than_env, 255, 0).astype(np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    out = np.zeros_like(img_bgr)
    out[mask > 0] = img_bgr[mask > 0]
    return out, mask, hsv


def compute_hsv_histograms(hsv_img, mask=None):
    """Compute histograms for H, S, V channels (for display)."""
    h_hist = cv2.calcHist([hsv_img], [0], mask, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_img], [1], mask, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_img], [2], mask, [256], [0, 256])
    return h_hist, s_hist, v_hist


def build_hist_image(h_hist, s_hist, v_hist, height=80):
    """Build a small RGB image showing H, S, V histograms side by side (for cv2 display)."""
    w = 200
    img = np.ones((height, w * 3, 3), dtype=np.uint8) * 255
    h_flat = np.ravel(h_hist)
    s_flat = np.ravel(s_hist)
    v_flat = np.ravel(v_hist)
    h_n = (h_flat / (h_flat.max() + 1e-6) * (height - 4)).astype(np.int32)
    s_n = (s_flat / (s_flat.max() + 1e-6) * (height - 4)).astype(np.int32)
    v_n = (v_flat / (v_flat.max() + 1e-6) * (height - 4)).astype(np.int32)
    for i in range(180):
        x0, x1 = int(i * (w - 1) / 180), int((i + 1) * (w - 1) / 180)
        y0 = int(height - 1 - min(int(h_n.flat[i]), height - 2))
        cv2.rectangle(img, (x0, y0), (x1, height - 1), (0, 0, 255), -1)
    for i in range(256):
        x0 = w + int(i * (w - 1) / 256)
        x1 = w + int((i + 1) * (w - 1) / 256)
        y0 = int(height - 1 - min(int(s_n.flat[i]), height - 2))
        cv2.rectangle(img, (x0, y0), (x1, height - 1), (0, 255, 0), -1)
    for i in range(256):
        x0 = 2 * w + int(i * (w - 1) / 256)
        x1 = 2 * w + int((i + 1) * (w - 1) / 256)
        y0 = int(height - 1 - min(int(v_n.flat[i]), height - 2))
        cv2.rectangle(img, (x0, y0), (x1, height - 1), (255, 0, 0), -1)
    cv2.putText(img, "H", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "S", (w + 5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "V", (2 * w + 5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def run_blue_gamma_single_frame(use_best=True):
    """Apply gamma then blue-only (best filter when use_best=True) to image1,2,3; show in one frame."""
    data = load_images()
    if not data:
        print("No images found. Place image1.png, image2.png, image3.png in the script directory.")
        return
    img_env = load_env()
    if img_env is None and use_best:
        print("image_env.png not found; using best filter without env suppression.")
    max_side = 480
    target_h = 360
    panels = []
    for name, img_bgr in data:
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gamma_img = apply_gamma(img_bgr, gamma=1.2)
        if use_best:
            blue_bgr, _, _ = apply_blue_mask_best(gamma_img, img_env)
        else:
            blue_bgr, _, _ = apply_blue_mask(gamma_img)
        # Resize to common height for a clean row
        ph, pw = blue_bgr.shape[:2]
        scale_h = target_h / ph
        new_w = int(pw * scale_h)
        panel = cv2.resize(blue_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(panel)
    if not panels:
        return
    frame = np.hstack(panels)
    title = "Blue lights (best filter + env)" if use_best else "Blue only (gamma)"
    cv2.imshow(title, frame)
    cv2.resizeWindow(title, min(frame.shape[1], 1920), frame.shape[0])
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run():
    data = load_images()
    if not data:
        print("No images found. Place image1.png, image2.png, image3.png in the script directory.")
        return
    img_env = load_env()

    for name, img_bgr in data:
        if img_bgr is None:
            continue

        # Resize for display if very large
        h, w = img_bgr.shape[:2]
        max_side = 640
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Pipelines
        gamma_img = apply_gamma(img_bgr, gamma=1.2)
        gauss_img = apply_gaussian(img_bgr)
        edge_img = apply_edge_canny(gauss_img)
        blue_only_bgr, blue_mask, hsv_img = apply_blue_mask(img_bgr)
        blue_from_gamma_bgr, mask_g, _ = apply_blue_mask(gamma_img)
        best_bgr, best_mask, _ = apply_blue_mask_best(gamma_img, img_env)

        # HSV histograms: full image and mask-only (best mask for blue)
        h_hist, s_hist, v_hist = compute_hsv_histograms(hsv_img, None)
        h_hist_m, s_hist_m, v_hist_m = compute_hsv_histograms(hsv_img, best_mask)

        hist_full = build_hist_image(h_hist, s_hist, v_hist)
        hist_mask = build_hist_image(h_hist_m, s_hist_m, v_hist_m)

        edge_bgr = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)

        # Row 1: original | gamma | gaussian | edges | blue (gamma) | best (env)
        row1 = np.hstack([
            img_bgr,
            gamma_img,
            gauss_img,
            edge_bgr,
            blue_from_gamma_bgr,
            best_bgr,
        ])
        # Row 2: best mask | HSV hist (full) | HSV hist (blue) | empty tiles
        iw, ih = img_bgr.shape[1], img_bgr.shape[0]
        pad = np.zeros((ih, iw, 3), dtype=np.uint8)
        pad[:] = 255
        hist_full_resized = cv2.resize(hist_full, (iw, ih), interpolation=cv2.INTER_NEAREST)
        hist_mask_resized = cv2.resize(hist_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
        row2 = np.hstack([mask_bgr, hist_full_resized, hist_mask_resized, pad, pad, pad])
        combined = np.vstack([row1, row2])

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["Original", "Gamma", "Gaussian", "Edges", "Blue (gamma)", "Best (env)"]
        for i, label in enumerate(labels):
            x = i * img_bgr.shape[1] + 8
            cv2.putText(combined, label, (x, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, "Best mask", (8, row1.shape[0] + 24), font, 0.5, (0, 0, 0), 1)
        cv2.putText(combined, "HSV hist (full)", (iw + 8, row1.shape[0] + 24), font, 0.5, (0, 0, 0), 1)
        cv2.putText(combined, "HSV hist (blue)", (2 * iw + 8, row1.shape[0] + 24), font, 0.5, (0, 0, 0), 1)

        win_name = f"Filters: {name}"
        cv2.imshow(win_name, combined)
        cv2.resizeWindow(win_name, min(combined.shape[1], 1920), min(combined.shape[0], 800))

    print("Close windows or press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_blue_gamma_single_frame()
