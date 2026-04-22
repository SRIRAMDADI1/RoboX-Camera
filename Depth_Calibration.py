"""
depth_calibrate.py  —  Interactive calibration tool for camera_red_inner_x.py
=============================================================================

HOW TO USE
----------
1.  Place your two red vertical markers at a known distance from the camera.
2.  Run this script.  The live feed shows the red filter + detected inner edges.
3.  When the reading looks stable, press SPACE to record a sample.
    The terminal prints the current gap_px and the Z you entered.
4.  Move the markers to several different distances (cover your working range).
    Aim for 6-10 samples spanning the full range you care about.
5.  Press 'f' to finish.  The script:
      - Fits  K = Z * gap_px  (should be constant for a pinhole camera)
      - Prints the mean K, its standard deviation, and the resulting
        DEPTH_CALIB_Z_CM / DEPTH_CALIB_GAP_PX pair to paste into your script.
      - Saves a residual plot  depth_calibration.png  next to this file.

CONSTANTS TO SET BELOW
-----------------------
  KNOWN_DISTANCES_CM   — if you always use the same distances, list them here
                         and the script will auto-prompt them in order.
                         Leave as [] to type the distance interactively.
"""

import ctypes
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── import shared camera/filter helpers ──────────────────────────────────────
# Adjust the path if depth_calibrate.py is not in the same directory as
# camera_red_filter.py and camera_red_inner_x.py.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from camera_red_filter import apply_gamma, apply_red_mask_best, cam_config, load_env
from camera_red_inner_x import (
    find_two_vertical_rectangles,
    inner_corners_and_x,
    inner_edge_gap_px,
)
from MVS.MvCameraControl_class import *

# ── optional: list known distances (cm) in the order you will place markers ──
KNOWN_DISTANCES_CM: list[float] = []   # e.g. [30, 50, 75, 100, 125, 150]

FPS = 30.0   # lower than live script; easier to run calibration

# ─────────────────────────────────────────────────────────────────────────────

def fit_calibration(samples: list[tuple[float, float]]):
    """
    samples: list of (z_cm, gap_px)
    Fits  K = z_cm * gap_px  (pinhole model → should be constant).
    Returns mean_K, std_K, residual errors.
    """
    ks = [z * g for z, g in samples]
    mean_k = float(np.mean(ks))
    std_k  = float(np.std(ks))
    residuals = []
    for z_meas, gap_px in samples:
        z_pred = mean_k / gap_px
        residuals.append(z_pred - z_meas)
    return mean_k, std_k, residuals


def save_plot(samples, mean_k, residuals):
    """Save a simple calibration plot using OpenCV (no matplotlib dependency)."""
    if not samples:
        return
    W, H = 700, 420
    img = np.ones((H, W, 3), dtype=np.uint8) * 30
    zs      = np.array([s[0] for s in samples])
    gaps    = np.array([s[1] for s in samples])
    z_pred  = mean_k / gaps

    z_min, z_max = zs.min(), zs.max()
    span = max(z_max - z_min, 1.0)
    pad  = 50

    def tx(z):
        return int(pad + (z - z_min) / span * (W - 2 * pad))

    def ty(z):
        return int(H - pad - (z - z_min) / span * (H - 2 * pad))

    # Draw axes
    cv2.line(img, (pad, H - pad), (W - pad, H - pad), (120, 120, 120), 1)
    cv2.line(img, (pad, pad), (pad, H - pad), (120, 120, 120), 1)

    # Ideal line (z_pred = z_meas)
    cv2.line(img, (tx(z_min), ty(z_min)), (tx(z_max), ty(z_max)), (80, 160, 80), 1)

    # Scatter: measured (yellow) vs predicted (cyan)
    for z_m, z_p in zip(zs, z_pred):
        cv2.circle(img, (tx(z_m), ty(z_m)), 5, (0, 220, 220), -1)   # measured
        cv2.circle(img, (tx(z_m), ty(z_p)), 4, (0, 80, 255),  -1)   # predicted

    cv2.putText(img, "Calibration: measured Z (cyan) vs predicted (orange)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"K = {mean_k:.1f}  (ideal: same for every sample)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    out_path = SCRIPT_DIR / "depth_calibration.png"
    cv2.imwrite(str(out_path), img)
    print(f"\nPlot saved to: {out_path}")


def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("No USB camera found.", file=sys.stderr)
        sys.exit(1)

    stDevInfo = ctypes.cast(
        deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)
    ).contents
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    cam = MvCamera()
    cam_config(cam, stDevInfo, FPS)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)
    data_buf = (ctypes.c_ubyte * payload_size)()

    img_env = load_env()
    if img_env is None:
        print("image_env1.png not found — calibrating without env suppression.")

    samples: list[tuple[float, float]] = []   # (z_cm, gap_px)
    known_idx = 0
    last_gap: float | None = None
    t_prev = time.perf_counter()

    print("\n=== Depth Calibration Tool ===")
    print("SPACE = record sample at current distance")
    print("'f'   = finish and compute calibration constants")
    print("'q'   = quit without saving\n")

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight
        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        gamma_img = apply_gamma(img_bgr)
        filtered  = apply_red_mask_best(gamma_img, img_env)
        display   = filtered.copy()

        left_r, right_r = find_two_vertical_rectangles(filtered)
        gap_px = None

        if left_r is not None and right_r is not None:
            l_top, r_bot, l_bot, r_top, inter = inner_corners_and_x(left_r, right_r)
            gap_px = inner_edge_gap_px(left_r, right_r)
            last_gap = gap_px

            col = (255, 255, 0)
            cv2.line(display,
                     (int(round(l_top[0])), int(round(l_top[1]))),
                     (int(round(r_bot[0])), int(round(r_bot[1]))), col, 2, cv2.LINE_AA)
            cv2.line(display,
                     (int(round(l_bot[0])), int(round(l_bot[1]))),
                     (int(round(r_top[0])), int(round(r_top[1]))), col, 2, cv2.LINE_AA)

            lx, ly, lw, lh = left_r
            rx, ry, rw, rh = right_r
            cv2.rectangle(display, (lx, ly), (lx + lw, ly + lh), (0, 200, 0), 1)
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 200, 0), 1)

            if inter is not None:
                ix, iy = inter
                cv2.circle(display, (int(round(ix)), int(round(iy))), 6,
                           (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(display, f"gap = {gap_px:.1f} px",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display, "No markers detected",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        t_now = time.perf_counter()
        fps_disp = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now
        cv2.putText(display, f"{fps_disp:.1f} FPS  |  samples: {len(samples)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display,
                    "SPACE=record  f=finish  q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Depth Calibration — Red Filter", display)
        key = cv2.waitKey(1) & 0xFF

        # ── record sample ──────────────────────────────────────────────────
        if key == ord(" "):
            if last_gap is None:
                print("[!] No markers visible — cannot record sample.")
            else:
                if KNOWN_DISTANCES_CM and known_idx < len(KNOWN_DISTANCES_CM):
                    z_cm = KNOWN_DISTANCES_CM[known_idx]
                    known_idx += 1
                    print(f"    Auto-distance: {z_cm} cm")
                else:
                    try:
                        z_cm = float(input(f"    Enter actual distance to markers (cm): "))
                    except ValueError:
                        print("[!] Invalid number — sample skipped.")
                        continue
                samples.append((z_cm, last_gap))
                k = z_cm * last_gap
                print(f"    Recorded: Z={z_cm:.2f} cm, gap={last_gap:.1f} px, K={k:.1f}")

        # ── finish ─────────────────────────────────────────────────────────
        elif key == ord("f"):
            break

        elif key == ord("q"):
            print("Aborted — no calibration saved.")
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            cv2.destroyAllWindows()
            return

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()

    # ── compute calibration ────────────────────────────────────────────────
    if len(samples) < 2:
        print("[!] Need at least 2 samples to calibrate. Exiting.")
        return

    mean_k, std_k, residuals = fit_calibration(samples)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Samples       : {len(samples)}")
    print(f"  Mean K        : {mean_k:.2f}  (Z_cm × gap_px)")
    print(f"  Std-dev K     : {std_k:.2f}  ({100*std_k/mean_k:.1f}% of mean)")
    print()
    print("  Per-sample breakdown:")
    print(f"  {'Z_meas (cm)':>12}  {'gap (px)':>10}  {'K':>10}  {'Z_pred (cm)':>12}  {'err (cm)':>9}")
    print("  " + "-" * 60)
    for (z, g), r in zip(samples, residuals):
        k_i = z * g
        print(f"  {z:>12.2f}  {g:>10.1f}  {k_i:>10.1f}  {z+r:>12.2f}  {r:>+9.2f}")
    print()
    print("  ── Paste these two lines into camera_red_inner_x.py ────────")
    # Pick the first sample as the "reference" for the two-constant form
    ref_z, ref_g = samples[0]
    ref_k = mean_k   # use fitted K, not raw sample
    # Back-solve a consistent (Z_calib, gap_calib) pair from mean K
    ref_gap_calib = ref_k / ref_z   # gap at ref_z implied by mean K
    print(f"  DEPTH_CALIB_Z_CM   = {ref_z:.2f}")
    print(f"  DEPTH_CALIB_GAP_PX = {ref_gap_calib:.1f}")
    print()
    print("  Alternative — set K directly (most accurate):")
    print(f"  # Replace depth_cm_to_x_center body's K with: K = {mean_k:.2f}")
    print("=" * 60)

    save_plot(samples, mean_k, residuals)
    print("\nDone.")


if __name__ == "__main__":
    main()