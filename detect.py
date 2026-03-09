"""Green-dot detection in BGR frames via HSV thresholding."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectConfig:
    # Hue range (OpenCV: 0-180).  Pure green ≈ 60, so 35-85 covers yellow-green to cyan-green.
    h_low: int = 35
    h_high: int = 85
    s_low: int = 80   # minimum saturation – rejects grey/white
    s_high: int = 255
    v_low: int = 40   # minimum value – rejects black
    v_high: int = 255
    min_area: float = 30.0    # px²
    max_area: float = 15000.0  # px²
    morph_kernel: int = 5      # morphological cleanup kernel size


def detect_green_dots(frame: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Return (N, 2) float array of (x, y) centroids of green blobs in *frame* (BGR)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([cfg.h_low,  cfg.s_low,  cfg.v_low],  dtype=np.uint8),
        np.array([cfg.h_high, cfg.s_high, cfg.v_high], dtype=np.uint8),
    )
    k = cfg.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts: list[list[float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if cfg.min_area <= area <= cfg.max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                pts.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])

    return np.array(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def tune_hsv(frame: np.ndarray, cfg: DetectConfig | None = None) -> DetectConfig:
    """
    Interactive HSV tuner. Opens two windows: the raw frame and the mask + detected blobs.
    Drag the trackbars until the dots look clean, then press any key to confirm.

    Returns a DetectConfig with the chosen values.
    """
    cfg = cfg or DetectConfig()

    WIN_CTRL = "HSV Tuner – controls"
    WIN_MASK = "HSV Tuner – mask (green=detected)"
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)

    def tb(name, val, maxv, win=WIN_CTRL):
        cv2.createTrackbar(name, win, val, maxv, lambda _: None)

    tb("H low",     cfg.h_low,  180)
    tb("H high",    cfg.h_high, 180)
    tb("S low",     cfg.s_low,  255)
    tb("V low",     cfg.v_low,  255)
    tb("Min area",  int(cfg.min_area),  2000)
    tb("Max area",  int(cfg.max_area) // 100, 500)  # stored ×100

    print("[tuner] Adjust trackbars, press any key to confirm.")

    while True:
        c = DetectConfig(
            h_low    = cv2.getTrackbarPos("H low",    WIN_CTRL),
            h_high   = cv2.getTrackbarPos("H high",   WIN_CTRL),
            s_low    = cv2.getTrackbarPos("S low",    WIN_CTRL),
            v_low    = cv2.getTrackbarPos("V low",    WIN_CTRL),
            min_area = float(cv2.getTrackbarPos("Min area", WIN_CTRL)),
            max_area = float(cv2.getTrackbarPos("Max area", WIN_CTRL) * 100),
        )
        dots = detect_green_dots(frame, c)

        # Build visual: mask coloured + detected centroids
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([c.h_low,  c.s_low,  c.v_low],  dtype=np.uint8),
            np.array([c.h_high, 255, 255], dtype=np.uint8),
        )
        vis = frame.copy()
        vis[mask == 0] = (vis[mask == 0] * 0.35).astype(np.uint8)  # dim non-masked regions
        for pt in dots:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)

        cv2.putText(vis, f"Detected: {len(dots)} dots", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(WIN_MASK, vis)

        if cv2.waitKey(30) != -1:
            break

    cv2.destroyWindow(WIN_CTRL)
    cv2.destroyWindow(WIN_MASK)
    return c
