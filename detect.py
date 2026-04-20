"""Blob detection in BGR frames via HSV thresholding."""

from __future__ import annotations

import json
import re
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

#NOTE: April tags w=1.226in. DO NOT DELETE

_TUNER_DEFAULTS_PATH = Path(__file__).with_name("hsv_tuner_defaults.json")

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


def _subject_key(subject: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", subject.strip().lower()).strip("_")
    return key or "default"


def _load_tuner_defaults_payload(path: Path = _TUNER_DEFAULTS_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def load_saved_detect_config(
    subject: str,
    fallback: DetectConfig,
    *,
    path: Path = _TUNER_DEFAULTS_PATH,
) -> DetectConfig:
    payload = _load_tuner_defaults_payload(path)
    raw = payload.get(_subject_key(subject))
    if not isinstance(raw, dict):
        return fallback

    values = {
        "h_low": int(raw.get("h_low", fallback.h_low)),
        "h_high": int(raw.get("h_high", fallback.h_high)),
        "s_low": int(raw.get("s_low", fallback.s_low)),
        "s_high": int(raw.get("s_high", fallback.s_high)),
        "v_low": int(raw.get("v_low", fallback.v_low)),
        "v_high": int(raw.get("v_high", fallback.v_high)),
        "min_area": float(raw.get("min_area", fallback.min_area)),
        "max_area": float(raw.get("max_area", fallback.max_area)),
        "morph_kernel": int(raw.get("morph_kernel", fallback.morph_kernel)),
    }
    values["morph_kernel"] = max(values["morph_kernel"], 1) | 1
    values["max_area"] = max(values["max_area"], values["min_area"] + 1.0)
    return DetectConfig(**values)


def save_detect_config_defaults(
    subject: str,
    cfg: DetectConfig,
    *,
    path: Path = _TUNER_DEFAULTS_PATH,
) -> Path:
    morph_kernel = max(int(cfg.morph_kernel), 1) | 1
    payload = _load_tuner_defaults_payload(path)
    payload[_subject_key(subject)] = {
        "h_low": int(cfg.h_low),
        "h_high": int(cfg.h_high),
        "s_low": int(cfg.s_low),
        "s_high": int(cfg.s_high),
        "v_low": int(cfg.v_low),
        "v_high": int(cfg.v_high),
        "min_area": float(cfg.min_area),
        "max_area": float(cfg.max_area),
        "morph_kernel": morph_kernel,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


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


def _select_primary_blob(
    accepted: list[tuple[float, float, float]],
    preferred_center: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Choose the dominant accepted blob, optionally preferring proximity to a point."""
    if not accepted:
        return None
    if preferred_center is None:
        return max(accepted, key=lambda item: item[2])

    pref = np.asarray(preferred_center, dtype=float)
    return min(
        accepted,
        key=lambda item: (
            float(np.hypot(item[0] - pref[0], item[1] - pref[1])),
            -item[2],
        ),
    )


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
    primary = _select_primary_blob(accepted, preferred_center)
    if primary is None:
        return None

    cx, cy, _area = primary
    return np.array([cx, cy], dtype=float)


def _close_cv_window(window_title: str) -> None:
    """Close an OpenCV window and flush a few GUI events on macOS."""
    try:
        cv2.destroyWindow(window_title)
    except cv2.error:
        return
    for _ in range(6):
        try:
            cv2.waitKey(1)
        except cv2.error:
            break


def _marker_count_badge(actual_count: int, expected_count: int) -> tuple[str, tuple[int, int, int]]:
    """Return a short badge label and color for live marker-count feedback."""
    if actual_count == expected_count:
        return f"baseline matched {actual_count}/{expected_count}", (70, 205, 70)
    if actual_count < expected_count:
        return f"baseline low {actual_count}/{expected_count}", (0, 180, 255)
    return f"baseline high {actual_count}/{expected_count}", (0, 140, 255)


def _draw_status_badge(
    img: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    *,
    origin: tuple[int, int],
    scale: float = 0.52,
) -> None:
    """Draw a compact badge with a colored indicator dot and label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    radius = 6
    pad_x = 10
    pad_y = 7
    gap = 8
    x0, y0 = origin

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    badge_h = th + baseline + 2 * pad_y
    badge_w = 2 * pad_x + 2 * radius + gap + tw

    cv2.rectangle(img, (x0, y0), (x0 + badge_w, y0 + badge_h), (24, 24, 24), -1)
    cv2.rectangle(img, (x0, y0), (x0 + badge_w, y0 + badge_h), color, 1)

    cy = y0 + badge_h // 2
    cx = x0 + pad_x + radius
    cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.putText(
        img,
        text,
        (cx + radius + gap, y0 + pad_y + th),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def tune_hsv(
    frame: np.ndarray,
    cfg: DetectConfig | None = None,
    *,
    subject: str = "blobs",
    preferred_center: np.ndarray | None = None,
    expected_count: int | None = None,
) -> DetectConfig:
    """
    Interactive HSV tuner.

    Shows a single resizable window with three panels side-by-side:
      LEFT   – original frame (dimmed outside mask)
      CENTRE – binary HSV mask (white = passes threshold)
      RIGHT  – detected blobs with centroids, IDs, and areas labelled

    Trackbars control all six HSV bounds plus area limits and morph kernel.
    Hovering the preview shows a zoom inset. For arbor-marker tuning, the
    predicted centre is drawn live. Press `D` to save the current settings as
    this subject's new defaults. When *expected_count* is provided, a live
    badge shows whether the accepted-marker count matches that baseline. Press
    Enter/Space/Q/Esc to confirm.
    """
    cfg = cfg or DetectConfig()
    subject_key = _subject_key(subject)
    show_primary_center = subject_key == "arbor_marker"

    WIN = (
        f"HSV Tuner ({subject})  |  original · mask · detections  |  "
        "hover zoom  |  D = save defaults  |  Enter/Space/Q/Esc = confirm"
    )
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    mouse_state: dict[str, int | None] = {"x": None, "y": None}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_state["x"] = int(x)
            mouse_state["y"] = int(y)

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
    cv2.setMouseCallback(WIN, on_mouse)

    print(f"[tuner] Drag sliders until {subject} look right. Hover the preview for zoom. Press D to save defaults.")

    MAX_PANEL_W = 640
    HOVER_VIEW_FRAC = 0.20
    HOVER_INSET_FRAC = 0.85
    c = cfg

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
        primary_blob = _select_primary_blob(accepted_f, preferred_center) if show_primary_center else None

        # ── panel 1: original dimmed outside mask ──────────────────────────
        p1 = frame.copy()
        dim = (p1 * 0.3).astype(np.uint8)
        p1[mask == 0] = dim[mask == 0]

        # ── panel 2: mask as BGR ───────────────────────────────────────────
        p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ── panel 3: detections annotated ─────────────────────────────────
        p3 = frame.copy()
        ORANGE = (255, 100, 0)
        for i, (cx, cy, area) in enumerate(accepted):
            cv2.circle(p3, (cx, cy), 10, ORANGE, 2)
            cv2.putText(p3, f"#{i} {area}px", (cx + 12, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
        for (cx, cy, area) in rejected:
            cv2.circle(p3, (cx, cy), 8, (0, 0, 200), 1)
            cv2.putText(p3, f"{area}px", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1, cv2.LINE_AA)

        center_status = "primary center: n/a"
        if primary_blob is not None:
            pcx = int(round(primary_blob[0]))
            pcy = int(round(primary_blob[1]))
            par = int(round(primary_blob[2]))
            center_status = f"primary center: ({pcx}, {pcy})  area={par}px"
            for panel in (p1, p2, p3):
                cv2.drawMarker(panel, (pcx, pcy), (255, 0, 255), cv2.MARKER_STAR, 24, 2)
                cv2.circle(panel, (pcx, pcy), 14, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(
                p1,
                "predicted arbor center",
                (pcx + 14, pcy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                p3,
                "predicted arbor center",
                (pcx + 14, pcy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        h_status = f"{h_low}-{h_high}" if h_low <= h_high else f"{h_low}-{h_high} wrap"
        status = (f"ACCEPTED: {len(accepted)}   REJECTED: {len(rejected)}   "
                  f"H[{h_status}] S[{s_low}-{s_high}] V[{v_low}-{v_high}]  "
                  f"area[{int(min_a)}-{int(max_a)}]  k={mk}")
        for panel in (p1, p2, p3):
            cv2.putText(panel, status, (8, panel.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            if show_primary_center:
                cv2.putText(panel, center_status, (8, panel.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        ph, pw = frame.shape[:2]
        scale = min(1.0, MAX_PANEL_W / max(pw, 1))
        nw, nh = int(pw * scale), int(ph * scale)
        panel_sources = [p1, p2, p3]
        panels = [cv2.resize(p, (nw, nh)) for p in panel_sources]
        strip = np.concatenate(panels, axis=1)

        banner_h = 28
        cv2.rectangle(strip, (0, 0), (strip.shape[1], banner_h), (24, 24, 24), -1)
        cv2.putText(
            strip,
            (
                "Hover preview for zoom  |  "
                + ("magenta star = predicted arbor center  |  " if show_primary_center else "")
                + "D = save defaults  |  Enter/Space/Q/Esc = confirm"
            ),
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        if expected_count is not None:
            badge_text, badge_color = _marker_count_badge(len(accepted), int(expected_count))
            _draw_status_badge(strip, badge_text, badge_color, origin=(10, banner_h + 8))

        hover_x = mouse_state["x"]
        hover_y = mouse_state["y"]
        if hover_x is not None and hover_y is not None and 0 <= hover_y < nh and 0 <= hover_x < 3 * nw:
            panel_idx = min(int(hover_x // max(nw, 1)), 2)
            panel_labels = ["original", "mask", "detections"]
            panel_x0 = panel_idx * nw
            src_x = int(np.clip(round((hover_x - panel_x0) / max(scale, 1e-9)), 0, pw - 1))
            src_y = int(np.clip(round(hover_y / max(scale, 1e-9)), 0, ph - 1))
            src_panel = panel_sources[panel_idx]

            inset_w = max(96, int(round(HOVER_INSET_FRAC * nw)))
            inset_h = max(96, int(round(HOVER_INSET_FRAC * nh)))
            crop_w = max(32, int(round(HOVER_VIEW_FRAC * pw)))
            crop_h = max(32, int(round(HOVER_VIEW_FRAC * ph)))
            half_w = max(8, crop_w // 2)
            half_h = max(8, crop_h // 2)
            x0 = max(0, src_x - half_w)
            y0 = max(0, src_y - half_h)
            x1 = min(src_panel.shape[1], x0 + crop_w)
            y1 = min(src_panel.shape[0], y0 + crop_h)
            x0 = max(0, x1 - crop_w)
            y0 = max(0, y1 - crop_h)
            patch = src_panel[y0:y1, x0:x1]

            zoom = cv2.resize(
                patch,
                (inset_w, inset_h),
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.drawMarker(
                zoom,
                (inset_w // 2, inset_h // 2),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                26,
                1,
            )

            disp_x0 = panel_x0 + int(round(x0 * scale))
            disp_y0 = int(round(y0 * scale))
            disp_x1 = panel_x0 + int(round(x1 * scale))
            disp_y1 = int(round(y1 * scale))
            cv2.rectangle(strip, (disp_x0, disp_y0), (disp_x1, disp_y1), (0, 255, 255), 1)
            cv2.drawMarker(strip, (int(hover_x), int(hover_y)), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)

            inset_x1 = strip.shape[1] - 10
            inset_y0 = 36
            inset_x0 = inset_x1 - inset_w
            inset_y1 = inset_y0 + inset_h
            strip[inset_y0:inset_y1, inset_x0:inset_x1] = zoom
            cv2.rectangle(strip, (inset_x0 - 1, inset_y0 - 1), (inset_x1 + 1, inset_y1 + 20), (255, 255, 255), 1)
            cv2.rectangle(strip, (inset_x0, inset_y0 - 18), (inset_x1, inset_y0), (24, 24, 24), -1)
            cv2.putText(
                strip,
                f"zoom: {panel_labels[panel_idx]}",
                (inset_x0 + 8, inset_y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(WIN, strip)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("d"), ord("D")):
            saved_path = save_detect_config_defaults(subject, c)
            print(f"[tuner] Saved defaults for '{subject}' to {saved_path}")
            continue
        if key in (13, 10, 32, 27, ord("q"), ord("Q")):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    _close_cv_window(WIN)
    h_status = f"{c.h_low}-{c.h_high}" if c.h_low <= c.h_high else f"{c.h_low}-{c.h_high} wrap"
    print(f"[tuner] Confirmed: H[{h_status}]  S[{c.s_low}-{c.s_high}]  "
          f"V[{c.v_low}-{c.v_high}]  area[{int(c.min_area)}-{int(c.max_area)}]  k={c.morph_kernel}")
    return c
