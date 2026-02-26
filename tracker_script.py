"""
Industrialized blue-rectangle tracker for the target robot.
- Detects ALL blue rectangles in the frame (tolerant to size, color, brightness, rotation).
- Clusters them by proximity (target has "similar blue rectangles next to them").
- Ranks clusters and tracks exactly ONE rectangle: the best candidate (highest score).
- That rectangle encloses the cluster of blue lights we want.
Press 'q' to quit. Press 'd' to toggle debug (show all blue rects).
"""

import os
import time
import ctypes
import cv2
import numpy as np

# Same DLL setup as camera_blue_rectangles / camera.py
DLL_DIRS = [
    r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
    r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
]

dll_dir = next((d for d in DLL_DIRS if os.path.exists(os.path.join(d, "MvCameraControl.dll"))), None)
if dll_dir is None:
    raise FileNotFoundError("MvCameraControl.dll not found. Install HIKRobot MVS (Windows x64).")

os.add_dll_directory(dll_dir)
os.environ["PATH"] = dll_dir + ";" + os.environ.get("PATH", "")

from MVS.MvCameraControl_class import *


def cam_config(cam: MvCamera, stDevInfo, FPS: float):
    cam.MV_CC_CreateHandle(stDevInfo)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
    cam.MV_CC_SetEnumValue("ExposureAuto", 2)
    cam.MV_CC_SetEnumValue("GainAuto", 2)
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 2)
    cam.MV_CC_StartGrabbing()
    cam.MV_CC_SetEnumValue("TriggerMode", 0)
    cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
    cam.MV_CC_SetFloatValue("AcquisitionFrameRate", FPS)


# --- Blue detection: tolerant to size, color, brightness, rotation ---
# Teal/cyan–blue (same as camera_blue_rectangles); slightly relaxed for variation
HUE_MIN, HUE_MAX = 76, 132
SAT_MIN, VAL_MIN = 50, 140
LED_MEAN_MIN = 160
MIN_BLUE_AREA = 150
CLOSE_K, ERODE_K, DILATE_K = 5, 2, 3


def find_all_blue_rectangles(img_bgr):
    """
    Detect all blue/teal rectangular regions. No aspect-ratio filter so rotation is OK.
    Returns list of dicts: {bbox: (x,y,w,h), area, mean_v, cx, cy}.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    low = np.array([HUE_MIN, SAT_MIN, VAL_MIN])
    high = np.array([HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    mask = cv2.erode(mask, np.ones((ERODE_K, ERODE_K), np.uint8))
    mask = cv2.dilate(mask, np.ones((DILATE_K, DILATE_K), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLUE_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue
        region_mask = np.zeros_like(v_channel)
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        mean_v = cv2.mean(v_channel, mask=region_mask)[0]
        if mean_v < LED_MEAN_MIN:
            continue
        cx, cy = x + w / 2, y + h / 2
        rects.append({
            "bbox": (x, y, w, h),
            "area": area,
            "mean_v": mean_v,
            "cx": cx,
            "cy": cy,
        })
    return rects


def _distance(r1, r2):
    return np.hypot(r1["cx"] - r2["cx"], r1["cy"] - r2["cy"])


def cluster_rectangles(rects, max_dist_px=180):
    """
    Group rectangles that are near each other (same robot).
    max_dist_px: two rects in same cluster if center distance < this.
    Returns list of clusters; each cluster is a list of rect dicts.
    """
    if not rects:
        return []
    n = len(rects)
    parent = list(range(n))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            if _distance(rects[i], rects[j]) <= max_dist_px:
                union(i, j)

    groups = {}
    for i in range(n):
        p = find(i)
        if p not in groups:
            groups[p] = []
        groups[p].append(rects[i])

    return list(groups.values())


def rank_clusters(clusters):
    """
    Rank clusters by likelihood of being the target robot.
    Target has "similar blue rectangles next to them" -> prefer more rects, similar sizes, brighter.
    Returns list of (cluster, score) sorted by score descending; higher = better.
    """
    scored = []
    for cluster in clusters:
        n = len(cluster)
        if n == 0:
            continue
        # Primary: number of blue rects (robot has several similar blue lights)
        count_score = n * 1000.0
        # Secondary: total brightness (brighter cluster = more emissive)
        mean_brightness = np.mean([r["mean_v"] for r in cluster])
        brightness_score = mean_brightness * 0.5
        # Tertiary: size consistency (similar areas = same type of LED)
        areas = [r["area"] for r in cluster]
        if len(areas) > 1:
            std_area = np.std(areas)
            mean_area = np.mean(areas)
            consistency = 1.0 / (1.0 + std_area / max(mean_area, 1))  # 0..1
        else:
            consistency = 0.5
        consistency_score = consistency * 200.0
        total = count_score + brightness_score + consistency_score
        scored.append((cluster, total))
    scored.sort(key=lambda x: -x[1])
    return scored


def bbox_around_cluster(cluster, frame_shape, padding_ratio=0.2):
    """
    One bounding box enclosing all rects in the cluster, with padding.
    Returns (x, y, w, h) clamped to frame.
    """
    if not cluster:
        return None
    fh, fw = frame_shape[:2]
    xs = []
    ys = []
    for r in cluster:
        x, y, w, h = r["bbox"]
        xs.extend([x, x + w])
        ys.extend([y, y + h])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    pad_w = max(8, int(width * padding_ratio))
    pad_h = max(8, int(height * padding_ratio))
    x_min = max(0, x_min - pad_w)
    y_min = max(0, y_min - pad_h)
    x_max = min(fw, x_max + pad_w)
    y_max = min(fh, y_max + pad_h)
    w = x_max - x_min
    h = y_max - y_min
    return (x_min, y_min, w, h)


def smooth_bbox(prev, curr, alpha=0.75):
    """Exponential smoothing to reduce jitter."""
    if prev is None or curr is None:
        return curr
    x = int(alpha * prev[0] + (1 - alpha) * curr[0])
    y = int(alpha * prev[1] + (1 - alpha) * curr[1])
    w = int(alpha * prev[2] + (1 - alpha) * curr[2])
    h = int(alpha * prev[3] + (1 - alpha) * curr[3])
    return (max(0, x), max(0, y), max(20, w), max(20, h))


WINDOW_NAME = "Blue rectangle tracker - 1 target"


def main():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError("No USB camera found.")

    stDevInfo = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    cam = MvCamera()
    FPS = 60.0
    cam_config(cam, stDevInfo, FPS)

    payload = MVCC_INTVALUE()
    cam.MV_CC_GetIntValue("PayloadSize", payload)
    payload_size = int(payload.nCurValue)
    data_buf = (ctypes.c_ubyte * payload_size)()

    cv2.namedWindow(WINDOW_NAME)

    track_bbox_smoothed = None
    show_debug = False
    t_prev = time.perf_counter()

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, stFrameInfo, 1000)
        if ret != 0:
            continue

        t_now = time.perf_counter()
        fps = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight
        img_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out = img_bgr.copy()

        # 1) Detect all blue rectangles
        rects = find_all_blue_rectangles(img_bgr)

        # 2) Cluster by proximity (same robot = rects next to each other)
        clusters = cluster_rectangles(rects, max_dist_px=180)

        # 3) Rank; take the single best cluster
        ranked = rank_clusters(clusters)
        best_bbox = None
        if ranked:
            best_cluster = ranked[0][0]
            best_bbox = bbox_around_cluster(best_cluster, img_bgr.shape, padding_ratio=0.2)

        # 4) Smooth and draw the one tracked rectangle
        if best_bbox is not None:
            track_bbox_smoothed = smooth_bbox(track_bbox_smoothed, best_bbox, alpha=0.75)
            tx, ty, tw, th = track_bbox_smoothed
            cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 3)
            cv2.putText(out, "target", (tx, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            track_bbox_smoothed = None

        # Debug: show all detected blue rects
        if show_debug:
            for r in rects:
                x, y, rw, rh = r["bbox"]
                cv2.rectangle(out, (x, y), (x + rw, y + rh), (255, 255, 0), 1)
            cv2.putText(out, f"blue rects: {len(rects)} clusters: {len(clusters)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(out, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(out, "1 tracked box (best cluster) | 'd' debug 'q' quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_debug = not show_debug

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
