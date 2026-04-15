"""
Live camera: same red pipeline as camera_red_filter.py, then finds two vertical
red rectangles and draws an X whose endpoints are the inner corners (facing each
other). Prints the intersection of the two X lines to the terminal.

Depth (cm) from camera to the X center uses a pinhole model on the horizontal gap
between inner vertical edges (no extra image processing):

    Z_cm = (f_x [px] * B [cm]) / gap_px

  Tune constants at the top of this file: INNER_EDGE_SEPARATION_CM, optional
  DEPTH_CALIB_Z_CM + DEPTH_CALIB_GAP_PX, and DEPTH_FX_PIXELS or
  DEPTH_HORIZONTAL_FOV_DEG.

Press 'q' to quit.
"""

import ctypes
import math
import sys
import time

import cv2
import numpy as np

# camera_red_filter configures DLL path and HIK camera imports
from camera_red_filter import (
    apply_red_mask_best,
    apply_gamma,
    cam_config,
    load_env,
)
from MVS.MvCameraControl_class import *


# --- depth calibration (edit for your camera / marker layout) ----------------

# Real-world distance (cm) between the inner vertical edges of the two markers.
INNER_EDGE_SEPARATION_CM = 13.72

# Optional one-shot calibration: at distance DEPTH_CALIB_Z_CM (cm), inner gap was
# DEPTH_CALIB_GAP_PX (px). Set both numbers, or both None to use f_x * B instead.
DEPTH_CALIB_Z_CM = 73.66  # e.g. 100.0
DEPTH_CALIB_GAP_PX = 403  # e.g. 320.0

# Horizontal focal length (pixels). If None, computed from DEPTH_HORIZONTAL_FOV_DEG.
DEPTH_FX_PIXELS = None  # e.g. 850.0

# Full horizontal field of view (degrees); used only when DEPTH_FX_PIXELS is None.
DEPTH_HORIZONTAL_FOV_DEG = 60.0


# --- geometry -----------------------------------------------------------------

def line_intersection(p1, p2, p3, p4):
    """
    Intersection of infinite line through p1-p2 with line through p3-p4.
    Each p is (x, y) float. Returns (x, y) or None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom
    return (px, py)


def _vertical_overlap_fraction(ly1, lh1, ly2, lh2):
    """Overlap of two y-intervals as a fraction of the shorter height."""
    y1_lo, y1_hi = ly1, ly1 + lh1
    y2_lo, y2_hi = ly2, ly2 + lh2
    overlap = max(0.0, min(y1_hi, y2_hi) - max(y1_lo, y2_lo))
    span = min(lh1, lh2)
    return overlap / max(span, 1e-6)


def find_two_vertical_rectangles(
    bgr_filtered,
    min_area=400,
    min_aspect=1.25,
    min_solidity=0.92,
    max_candidates=12,
    min_pair_height_ratio=0.55,
    min_vertical_overlap=0.45,
):
    """
    From red-filter output (black background), find two vertical bar markers.

    Corner / L-shaped blobs (one vertical edge fused with an adjacent horizontal
    edge) still look tall but are not calibrated vertical bars; they have lower
    solidity (area / convex hull area) than a single bar, so they are dropped.

    Among remaining candidates, picks the left–right pair with best combined area
    that also has similar bar heights and substantial vertical overlap (same
    physical target), instead of blindly taking the two largest blobs.
    """
    gray = cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h <= 0 or w <= 0:
            continue
        if h / float(w) < min_aspect:
            continue
        hull = cv2.convexHull(c)
        hull_a = cv2.contourArea(hull)
        if hull_a < 1e-6:
            continue
        if a / hull_a < min_solidity:
            continue
        cx = x + w * 0.5
        candidates.append({"area": a, "cx": cx, "xywh": (x, y, w, h)})

    if len(candidates) < 2:
        return None, None

    candidates.sort(key=lambda d: d["area"], reverse=True)
    candidates = candidates[:max_candidates]

    best_left, best_right = None, None
    best_score = -1.0
    n = len(candidates)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A, B = candidates[i], candidates[j]
            if A["cx"] >= B["cx"]:
                continue
            left, right = A, B
            _, ly, _, lh = left["xywh"]
            _, ry, _, rh = right["xywh"]
            hratio = min(lh, rh) / max(float(lh), float(rh))
            if hratio < min_pair_height_ratio:
                continue
            ov = _vertical_overlap_fraction(ly, lh, ry, rh)
            if ov < min_vertical_overlap:
                continue
            score = float(left["area"] + right["area"]) + 2000.0 * ov
            if score > best_score:
                best_score = score
                best_left, best_right = left["xywh"], right["xywh"]

    if best_left is not None:
        return best_left, best_right

    # No pair passed geometry: do not fall back to ambiguous largest blobs (bad depth).
    return None, None


def inner_corners_and_x(left, right):
    """
    Inner vertical edge: right side of left rect, left side of right rect.
    X: diagonal from left top-inner to right bottom-inner, and left bottom-inner
    to right top-inner.
    Returns (pt for line A, pt for line B, pt for line C, pt for line D) and
    intersection of the two diagonals, or None if degenerate.
    """
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    # Inner edge pixel coordinates (integer); use consistent inner vertical lines
    x_left_inner = lx + lw - 1
    x_right_inner = rx

    l_top = (float(x_left_inner), float(ly))
    l_bot = (float(x_left_inner), float(ly + lh - 1))
    r_top = (float(x_right_inner), float(ry))
    r_bot = (float(x_right_inner), float(ry + rh - 1))

    # X: l_top -- r_bot and l_bot -- r_top
    inter = line_intersection(l_top, r_bot, l_bot, r_top)
    return l_top, r_bot, l_bot, r_top, inter


def inner_edge_gap_px(left_xywh, right_xywh):
    """Horizontal pixel distance between inner vertical edges (same geometry as inner_corners_and_x)."""
    lx, ly, lw, lh = left_xywh
    rx, ry, rw, rh = right_xywh
    inner_left = lx + lw - 1
    inner_right = rx
    return max(float(inner_right - inner_left), 1.0)


def focal_x_pixels(frame_width: int) -> float:
    """Horizontal focal length in pixels from DEPTH_FX_PIXELS or DEPTH_HORIZONTAL_FOV_DEG."""
    if DEPTH_FX_PIXELS is not None:
        return float(DEPTH_FX_PIXELS)
    h = math.radians(float(DEPTH_HORIZONTAL_FOV_DEG))
    return (0.5 * float(frame_width)) / math.tan(0.5 * h)


def depth_cm_to_x_center(frame_width: int, gap_px: float) -> float:
    """
    Z [cm] from camera optical center to the plane of the markers, using inner-edge gap.
    Uses K = DEPTH_CALIB_Z_CM * DEPTH_CALIB_GAP_PX if both set; else K = f_x * B.
    """
    if DEPTH_CALIB_Z_CM is not None and DEPTH_CALIB_GAP_PX is not None:
        k = float(DEPTH_CALIB_Z_CM) * float(DEPTH_CALIB_GAP_PX)
    else:
        k = focal_x_pixels(frame_width) * float(INNER_EDGE_SEPARATION_CM)
    return k / gap_px


def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("No USB camera found.", file=sys.stderr)
        sys.exit(1)

    stDevInfo = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    cam = MvCamera()
    FPS = 60.0
    cam_config(cam, stDevInfo, FPS)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)
    data_buf = (ctypes.c_ubyte * payload_size)()

    img_env = load_env()
    if img_env is None:
        print("image_env1.png not found; using best filter without env suppression.")

    t_prev = time.perf_counter()
    print_interval = 0.25
    last_print = 0.0

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight
        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gamma_img = apply_gamma(img_bgr)
        out = apply_red_mask_best(gamma_img, img_env)

        left_r, right_r = find_two_vertical_rectangles(out)
        display = out.copy()

        if left_r is not None and right_r is not None:
            l_top, r_bot, l_bot, r_top, inter = inner_corners_and_x(left_r, right_r)
            # X lines (cyan), slightly thick
            col = (255, 255, 0)
            cv2.line(
                display,
                (int(round(l_top[0])), int(round(l_top[1]))),
                (int(round(r_bot[0])), int(round(r_bot[1]))),
                col,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                display,
                (int(round(l_bot[0])), int(round(l_bot[1]))),
                (int(round(r_top[0])), int(round(r_top[1]))),
                col,
                2,
                cv2.LINE_AA,
            )
            if inter is not None:
                ix, iy = inter
                cv2.circle(
                    display,
                    (int(round(ix)), int(round(iy))),
                    6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                gap_px = inner_edge_gap_px(left_r, right_r)
                z_cm = depth_cm_to_x_center(w, gap_px)
                cv2.putText(
                    display,
                    f"({ix:.1f},{iy:.1f}) Z~{z_cm:.0f}cm",
                    (int(round(ix)) + 10, int(round(iy)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                now = time.perf_counter()
                if now - last_print >= print_interval:
                    print(
                        f"X center (px x y): {ix:.2f} {iy:.2f}  |  "
                        f"depth (cm, pinhole inner-gap): {z_cm:.2f}  |  "
                        f"inner gap (px): {gap_px:.1f}"
                    )
                    last_print = now
            # Optional: rectangle outlines for debugging
            lx, ly, lw, lh = left_r
            rx, ry, rw, rh = right_r
            cv2.rectangle(display, (lx, ly), (lx + lw - 1, ly + lh - 1), (0, 200, 0), 1)
            cv2.rectangle(display, (rx, ry), (rx + rw - 1, ry + rh - 1), (0, 200, 0), 1)

        t_now = time.perf_counter()
        fps = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now
        cv2.putText(display, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Red filter + inner X — Live", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
