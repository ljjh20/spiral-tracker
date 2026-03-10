"""Blob detection in BGR frames via HSV thresholding."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectConfig:
    # Hue range (OpenCV: 0-180).  Pure green ≈ 60, so 35-85 covers yellow-green to cyan-green.
    h_low: int = 35
    h_high: int = 85
    s_low: int = 34   # minimum saturation – rejects grey/white
    s_high: int = 255
    v_low: int = 40   # minimum value – rejects black
    v_high: int = 255
    min_area: float = 30.0    # px²
    max_area: float = 15000.0  # px²
    morph_kernel: int = 5      # morphological cleanup kernel size


def _hsv_mask(hsv: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Binary mask for *cfg*, with wrapped hue support for colours such as red."""
    lower_sv = np.array([cfg.s_low, cfg.v_low], dtype=np.uint8)
    upper_sv = np.array([cfg.s_high, cfg.v_high], dtype=np.uint8)

    if cfg.h_low <= cfg.h_high:
        mask = cv2.inRange(
            hsv,
            np.array([cfg.h_low, *lower_sv], dtype=np.uint8),
            np.array([cfg.h_high, *upper_sv], dtype=np.uint8),
        )
    else:
        mask_lo = cv2.inRange(
            hsv,
            np.array([0, *lower_sv], dtype=np.uint8),
            np.array([cfg.h_high, *upper_sv], dtype=np.uint8),
        )
        mask_hi = cv2.inRange(
            hsv,
            np.array([cfg.h_low, *lower_sv], dtype=np.uint8),
            np.array([179, *upper_sv], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask_lo, mask_hi)

    k = cfg.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask


def hsv_mask(frame: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Return the cleaned HSV mask for *frame* and *cfg*."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return _hsv_mask(hsv, cfg)


def _blob_stats(frame: np.ndarray, cfg: DetectConfig) -> tuple[np.ndarray, list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return (mask, accepted, rejected) blob stats as (cx, cy, area)."""
    mask = hsv_mask(frame, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted: list[tuple[float, float, float]] = []
    rejected: list[tuple[float, float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        if cfg.min_area <= area <= cfg.max_area:
            accepted.append((cx, cy, area))
        else:
            rejected.append((cx, cy, area))

    return mask, accepted, rejected


def detect_green_dots(frame: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Return (N, 2) float array of (x, y) centroids of accepted blobs in *frame*."""
    _, accepted, _ = _blob_stats(frame, cfg)
    if not accepted:
        return np.empty((0, 2), dtype=float)
    return np.array([[cx, cy] for cx, cy, _area in accepted], dtype=float)


def detect_primary_blob_center(
    frame: np.ndarray,
    cfg: DetectConfig,
    preferred_center: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Return the dominant accepted blob centroid.

    If *preferred_center* is supplied, choose the accepted blob nearest to that
    point; otherwise choose the largest accepted blob.
    """
    _, accepted, _ = _blob_stats(frame, cfg)
    if not accepted:
        return None

    if preferred_center is None:
        cx, cy, _area = max(accepted, key=lambda item: item[2])
    else:
        pref = np.asarray(preferred_center, dtype=float)
        cx, cy, _area = min(
            accepted,
            key=lambda item: (
                float(np.hypot(item[0] - pref[0], item[1] - pref[1])),
                -item[2],
            ),
        )
    return np.array([cx, cy], dtype=float)


def tune_hsv(
    frame: np.ndarray,
    cfg: DetectConfig | None = None,
    *,
    subject: str = "blobs",
) -> DetectConfig:
    """
    Interactive HSV tuner.

    Shows a single resizable window with three panels side-by-side:
      LEFT   – original frame (dimmed outside mask)
      CENTRE – binary HSV mask (white = passes threshold)
      RIGHT  – detected blobs with centroids, IDs, and areas labelled

    Trackbars control all six HSV bounds plus area limits and morph kernel.
    Press any key (or Q) to confirm and return the current DetectConfig.
    """
    cfg = cfg or DetectConfig()

    WIN = f"HSV Tuner ({subject})  |  original · mask · detections  |  press any key to confirm"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Trackbar helpers — all live on the same window as the image
    def tb(name: str, val: int, maxv: int) -> None:
        cv2.createTrackbar(name, WIN, val, maxv, lambda _: None)

    tb("H low",      cfg.h_low,              179)
    tb("H high",     cfg.h_high,             179)
    tb("S low",      cfg.s_low,              255)
    tb("S high",     cfg.s_high,             255)
    tb("V low",      cfg.v_low,              255)
    tb("V high",     cfg.v_high,             255)
    tb("Min area",   int(cfg.min_area),     5000)   # direct pixels²
    tb("Max area x100", int(cfg.max_area // 100), 5000)  # ×100 px²
    tb("Morph k",    cfg.morph_kernel,        21)

    print(f"[tuner] Drag sliders until {subject} look right, then press any key.")

    # Downscale factor so the 3-panel strip fits on screen
    MAX_PANEL_W = 640

    while True:
        h_low  = cv2.getTrackbarPos("H low",         WIN)
        h_high = cv2.getTrackbarPos("H high",        WIN)
        s_low  = cv2.getTrackbarPos("S low",         WIN)
        s_high = cv2.getTrackbarPos("S high",        WIN)
        v_low  = cv2.getTrackbarPos("V low",         WIN)
        v_high = cv2.getTrackbarPos("V high",        WIN)
        min_a  = float(cv2.getTrackbarPos("Min area",       WIN))
        max_a  = float(cv2.getTrackbarPos("Max area x100",  WIN)) * 100.0
        mk     = cv2.getTrackbarPos("Morph k", WIN)
        mk     = max(mk, 1) | 1   # force odd ≥ 1

        c = DetectConfig(
            h_low=h_low, h_high=h_high,
            s_low=s_low, s_high=s_high,
            v_low=v_low, v_high=v_high,
            min_area=min_a, max_area=max(max_a, min_a + 1),
            morph_kernel=mk,
        )

        mask, accepted_f, rejected_f = _blob_stats(frame, c)
        accepted = [(int(cx), int(cy), int(area)) for cx, cy, area in accepted_f]
        rejected = [(int(cx), int(cy), int(area)) for cx, cy, area in rejected_f]

        # ── panel 1: original dimmed outside mask ──────────────────────────
        p1 = frame.copy()
        dim = (p1 * 0.3).astype(np.uint8)
        p1[mask == 0] = dim[mask == 0]

        # ── panel 2: mask as BGR ───────────────────────────────────────────
        p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ── panel 3: detections annotated ─────────────────────────────────
        p3 = frame.copy()
        ORANGE = (255, 100, 0)   # BGR – blue
        for i, (cx, cy, area) in enumerate(accepted):
            cv2.circle(p3, (cx, cy), 10, ORANGE, 2)
            cv2.putText(p3, f"#{i} {area}px", (cx + 12, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
        for (cx, cy, area) in rejected:
            cv2.circle(p3, (cx, cy), 8, (0, 0, 200), 1)
            cv2.putText(p3, f"{area}px", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1, cv2.LINE_AA)

        h_status = f"{h_low}-{h_high}" if h_low <= h_high else f"{h_low}-{h_high} wrap"
        status = (f"ACCEPTED: {len(accepted)}   REJECTED: {len(rejected)}   "
                  f"H[{h_status}] S[{s_low}-{s_high}] V[{v_low}-{v_high}]  "
                  f"area[{int(min_a)}-{int(max_a)}]  k={mk}")
        for panel in (p1, p2, p3):
            cv2.putText(panel, status, (8, panel.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # ── resize panels to a common height then concatenate ──────────────
        ph, pw = frame.shape[:2]
        scale  = min(1.0, MAX_PANEL_W / pw)
        nw, nh = int(pw * scale), int(ph * scale)
        panels = [cv2.resize(p, (nw, nh)) for p in (p1, p2, p3)]
        strip  = np.concatenate(panels, axis=1)

        cv2.imshow(WIN, strip)
        key = cv2.waitKey(30)
        if key != -1:   # any key → confirm
            break

    cv2.destroyWindow(WIN)
    h_status = f"{c.h_low}-{c.h_high}" if c.h_low <= c.h_high else f"{c.h_low}-{c.h_high} wrap"
    print(f"[tuner] Confirmed: H[{h_status}]  S[{c.s_low}-{c.s_high}]  "
          f"V[{c.v_low}-{c.v_high}]  area[{int(c.min_area)}-{int(c.max_area)}]  k={c.morph_kernel}")
    return c
