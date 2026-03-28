"""
Live camera: same blue pipeline as camera_blue_filter.py, then finds two vertical
blue rectangles and draws an X whose endpoints are the inner corners (facing each
other). Prints the intersection of the two X lines to the terminal.

Press 'q' to quit.
"""

import ctypes
import os
import sys
import time

import cv2
import numpy as np

# camera_blue_filter configures DLL path and HIK camera imports
from camera_blue_filter import (
    apply_blue_mask_best,
    apply_gamma,
    cam_config,
    load_env,
)
from MVS.MvCameraControl_class import *


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


def find_two_vertical_rectangles(bgr_filtered, min_area=400, min_aspect=1.25):
    """
    From blue-filter output (black background), find the two largest vertical
    blobs. Returns (left_xywh, right_xywh) as (x, y, w, h) or (None, None).
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
        cx = x + w * 0.5
        candidates.append((a, cx, (x, y, w, h)))

    if len(candidates) < 2:
        return None, None

    candidates.sort(key=lambda t: t[0], reverse=True)
    # Two largest vertical blobs; if more than two, take top two by area
    top = candidates[:2]
    top.sort(key=lambda t: t[1])
    _, _, left = top[0]
    _, _, right = top[1]
    return left, right


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
        print("image_env not found; using best filter without env suppression.")

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
        out = apply_blue_mask_best(gamma_img, img_env)

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
                cv2.putText(
                    display,
                    f"({ix:.1f},{iy:.1f})",
                    (int(round(ix)) + 10, int(round(iy)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                now = time.perf_counter()
                if now - last_print >= print_interval:
                    print(f"X intersection (image coords, x y): {ix:.2f} {iy:.2f}")
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

        cv2.imshow("Blue filter + inner X — Live", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
