"""
Live camera view with detection and highlighting of 2 vertical blue LED lights
(robot flashlights / emissive blue). Tuned for bright luminous teal–cyan–blue
glow (central bright area); ignores darker blue surround and dull blue objects.
Uses the same HIKRobot MVS capture pipeline as camera.py.
Press 'q' to quit.
"""

import os
import time
import ctypes
import cv2
import numpy as np

# Same DLL setup as camera.py
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


# Bright luminous teal/cyan–blue (HSV): hue spans teal through blue
# Matches glowing central area: light blue, pale teal, cyan
HUE_MIN, HUE_MAX = 78, 128   # teal/cyan (~85) through blue (~120)
SATURATION_MIN = 55          # allow pale teal; luminous center often high sat
VALUE_MIN = 175              # bright glow (luminous center)
# Require mean brightness inside region (filters darker blue surround)
LED_MEAN_VALUE_MIN = 195
CLOSE_SIZE, ERODE_SIZE, DILATE_SIZE = 3, 2, 2


def find_vertical_blue_rectangles(img_bgr, min_area=800, max_candidates=2):
    """
    Detect up to max_candidates vertical blue rectangles from LED-type emissions
    (bright blue lights on robots), not general blue-colored objects.
    Returns list of (x, y, w, h) bounding rects, sorted by x (left to right).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    lower_blue = np.array([HUE_MIN, SATURATION_MIN, VALUE_MIN])
    upper_blue = np.array([HUE_MAX, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    close_k = np.ones((CLOSE_SIZE, CLOSE_SIZE), np.uint8)
    erode_k = np.ones((ERODE_SIZE, ERODE_SIZE), np.uint8)
    dilate_k = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.erode(mask, erode_k)
    mask = cv2.dilate(mask, dilate_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h < w:
            continue
        # Only keep regions that are genuinely bright (emissive), not just blue-colored
        region_mask = np.zeros_like(v_channel)
        cv2.drawContours(region_mask, [c], -1, 255, -1)
        mean_v = cv2.mean(v_channel, mask=region_mask)[0]
        if mean_v < LED_MEAN_VALUE_MIN:
            continue
        candidates.append((x, y, w, h, area))

    candidates.sort(key=lambda r: r[0])
    return [c[:4] for c in candidates[:max_candidates]]


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

        cv2.putText(out, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        rects = find_vertical_blue_rectangles(img_bgr, min_area=800, max_candidates=2)

        for (x, y, rw, rh) in rects:
            cv2.rectangle(out, (x, y), (x + rw, y + rh), (0, 255, 0), 3)
            cv2.putText(
                out, "blue", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.imshow("Blue vertical rectangles", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
