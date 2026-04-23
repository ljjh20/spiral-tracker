#!/usr/bin/env python3
"""
Spiral torsion spring green-dot tracker.

Produces:
  • results/reference_frame.png   – annotated first analyzed frame
  • results/final_frame.png       – annotated last analyzed frame
  • results/tracked.mp4           – full video with overlaid dot IDs
  • results/displacements.csv     – per-dot displacement vs time
  • results/frame_measurements.csv – per-frame summary incl. force-meter reads when enabled
  • results/displacement_ures.png – final-frame URES-style arc-length plot
  • results/displacement_ures_all_frames.png – overlaid URES-style curves for all tracked frames
  • results/displacement_ures_peak_vs_pkl.png – tracker peak-frame URES vs Spring .pkl URES
  • results/displacement_time.png – displacement-vs-time per dot
  • results/force_time.png / force_tip_displacement.png – force-meter plots when enabled

Usage
-----
# Bundle directory with capture + Spring pickle(s):
    python run_tracker.py --input-dir path/to/bundle

# Interactive AprilTag/intrinsics/depth-calibration capture:
    python run_tracker.py --capture-dir mov_intrinsics/capture-2026-03-18T22-11-50.657Z

# Interactive legacy/manual calibration:
    python run_tracker.py --video recordings/IMG_7036.mov --outer-radius-mm 39.5

# Known centre, skip tuner:
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --outer-radius-mm 39.5 --center-x 960 --center-y 540 --no-tune --no-range-gui

# Every 3rd frame only:
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --outer-radius-mm 39.5 --frame-step 3
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import configure_matplotlib_defaults
from bundle_discovery import (
    discover_sweep_candidates,
    load_pickle,
    resolve_unique_capture_bundle,
    resolve_unique_reference_spring,
    resolve_unique_video,
)
from detect import (
    DetectConfig,
    detect_green_dots,
    detect_primary_blob_center,
    hsv_mask,
    load_saved_detect_config,
    tune_hsv,
)
from force_meter import (
    ForceMeterReading,
    calibrate_force_segments,
    detect_force_screen,
    read_force_meter,
)
from planar_calibration import (
    APRILTAG_FAMILY_IDS,
    CalibrationError,
    DepthCalibration,
    FrameUndistorter,
    IntrinsicsSequence,
    PlaneCalibration,
    build_frame_undistorter,
    build_plane_calibration,
    detect_apriltags,
    estimate_frame_pose,
    image_points_to_plane,
    load_depth_calibration,
    load_intrinsics_sequence,
    plane_points_to_image,
)
from spiral_fit import (
    arc_lengths_from_min,
    fit_spiral,
    match_dots,
    refine_center,
    spiral_fit_rmse,
)

configure_matplotlib_defaults()

URES_NOISE_FLOOR_M = 1e-8


# ── Frame / display helpers ───────────────────────────────────────────────────

def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _rotate_points(
    points: np.ndarray,
    width: int,
    height: int,
    degrees: int,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    pts_2d = np.atleast_2d(pts)
    out = pts_2d.copy()
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    if degrees == 90:
        out[:, 0] = float(height - 1) - y
        out[:, 1] = x
    elif degrees == 180:
        out[:, 0] = float(width - 1) - x
        out[:, 1] = float(height - 1) - y
    elif degrees == 270:
        out[:, 0] = y
        out[:, 1] = float(width - 1) - x
    if pts.ndim == 1:
        return out[0]
    return out


def _unrotate_points(
    points: np.ndarray,
    width: int,
    height: int,
    degrees: int,
) -> np.ndarray:
    inverse = {0: 0, 90: 270, 180: 180, 270: 90}[int(degrees)]
    rot_width = width if degrees in (0, 180) else height
    rot_height = height if degrees in (0, 180) else width
    return _rotate_points(points, rot_width, rot_height, inverse)


def _prepare_frame_pair(
    frame: np.ndarray,
    rotate: int,
    frame_undistorter: FrameUndistorter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_geom = frame if frame_undistorter is None else frame_undistorter.undistort(frame)
    return frame_geom, _rotate(frame_geom, rotate)


def _read_frame_at(
    cap: cv2.VideoCapture,
    frame_idx: int,
    rotate: int,
    frame_undistorter: FrameUndistorter | None = None,
) -> np.ndarray | None:
    """Read exactly *frame_idx* (if decodable) and apply rotation."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    _frame_geom, frame_display = _prepare_frame_pair(frame, rotate, frame_undistorter)
    return frame_display


def _read_frame_pair_at(
    cap: cv2.VideoCapture,
    frame_idx: int,
    rotate: int,
    frame_undistorter: FrameUndistorter | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read exactly *frame_idx* and return (geometry_frame, display_frame)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None, None
    return _prepare_frame_pair(frame, rotate, frame_undistorter)


def _read_last_frame(
    cap: cv2.VideoCapture,
    n_frames: int,
    rotate: int,
    frame_undistorter: FrameUndistorter | None = None,
) -> tuple[np.ndarray | None, int]:
    """Seek backwards from the end until a decodable frame is found.
    Returns (frame, frame_idx). frame_idx is -1 if nothing readable."""
    for offset in range(min(60, n_frames)):
        idx = n_frames - 1 - offset
        frame = _read_frame_at(cap, idx, rotate, frame_undistorter)
        if frame is not None:
            return frame, idx
    return None, -1


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


@dataclass
class _ScaleMode:
    key: str
    label: str
    reference_mm: float


def _pick_frame_range(
    cap: cv2.VideoCapture,
    n_frames: int,
    rotate: int,
    frame_undistorter: FrameUndistorter | None = None,
    *,
    fps: float,
    initial_start_idx: int,
    initial_end_idx: int,
) -> tuple[int, int]:
    """Interactive frame-range picker shown before any analysis UI."""
    WIN = "Choose Analysis Range  |  drag sliders, press Enter/Space to accept"
    sample_frame = _read_frame_at(cap, initial_start_idx, rotate, frame_undistorter)
    if sample_frame is None:
        sample_frame = _read_frame_at(cap, initial_end_idx, rotate, frame_undistorter)
    if sample_frame is None:
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        frame_h, frame_w = (raw_h, raw_w) if rotate in (0, 180) else (raw_w, raw_h)
    else:
        frame_h, frame_w = sample_frame.shape[:2]
    scale = min(1.0, 760.0 / max(frame_w, 1), 820.0 / max(frame_h, 1))
    disp_w = max(1, int(round(frame_w * scale)))
    disp_h = max(1, int(round(frame_h * scale)))

    start_idx = int(np.clip(initial_start_idx, 0, max(n_frames - 1, 0)))
    end_idx = int(np.clip(initial_end_idx, start_idx, max(n_frames - 1, 0)))
    initial_pair = (start_idx, end_idx)
    cached_pair: tuple[int, int] | None = None
    start_frame: np.ndarray | None = None
    end_frame: np.ndarray | None = None

    def _frame_panel(frame: np.ndarray | None, label: str, frame_idx: int) -> np.ndarray:
        if frame is None:
            panel = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            cv2.putText(panel, f"Frame {frame_idx} is not decodable", (30, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 92), (20, 20, 20), -1)
        cv2.putText(panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame = {frame_idx}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"time = {frame_idx / max(fps, 1e-9):.2f} s", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return panel

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Start frame", WIN, start_idx, max(n_frames - 1, 0), lambda _: None)
    cv2.createTrackbar("End frame", WIN, end_idx, max(n_frames - 1, 0), lambda _: None)
    print(
        "[range] Choose the first and last frame to analyze.\n"
        "        The selected frame numbers stay absolute in the CSV/plots.\n"
        "        Press Enter/Space to accept, or Esc to keep the initial range."
    )

    while True:
        start_idx = cv2.getTrackbarPos("Start frame", WIN)
        end_idx = cv2.getTrackbarPos("End frame", WIN)
        if start_idx > end_idx:
            end_idx = start_idx
            cv2.setTrackbarPos("End frame", WIN, end_idx)

        current_pair = (start_idx, end_idx)
        if current_pair != cached_pair:
            start_frame = _read_frame_at(cap, start_idx, rotate, frame_undistorter)
            end_frame = _read_frame_at(cap, end_idx, rotate, frame_undistorter)
            cached_pair = current_pair

        start_panel = _frame_panel(start_frame, "Analysis start", start_idx)
        end_panel = _frame_panel(end_frame, "Analysis end", end_idx)
        strip = np.concatenate(
            [cv2.resize(panel, (disp_w, disp_h)) for panel in (start_panel, end_panel)],
            axis=1,
        )

        footer_h = 86
        canvas = np.zeros((disp_h + footer_h, strip.shape[1], 3), dtype=np.uint8)
        canvas[:disp_h] = strip
        range_text = f"Selected analysis window: frames {start_idx} .. {end_idx}  ({end_idx - start_idx + 1} frame(s))"
        cv2.putText(canvas, range_text, (12, disp_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Outputs keep these absolute frame indices; only the analyzed window is cropped.",
                    (12, disp_h + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(30)
        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                start_idx, end_idx = initial_pair
                break
            continue

        key &= 0xFF
        if key in (13, 10, 32):
            if start_frame is not None and end_frame is not None:
                break
            print("[range] Both selected frames must be decodable before accepting.")
        elif key == 27:
            start_idx, end_idx = initial_pair
            break

    _close_cv_window(WIN)
    print(f"[range] Using frames {start_idx} .. {end_idx}")
    return start_idx, end_idx


def _pick_center(
    frame: np.ndarray,
    detected_center: np.ndarray | None = None,
) -> np.ndarray | None:
    """Interactive click to confirm or adjust the spring arbor centre."""
    WIN = "Confirm spring centre (click to adjust, press any key to confirm)"
    selected = None if detected_center is None else np.array(detected_center, dtype=float)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal selected
            selected = np.array([x, y], dtype=float)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    if detected_center is None:
        print("[center] Click the arbor (centre pin) in the window, then press any key.")
    else:
        print("[center] Auto-detected the arbor marker. Click to shift it, or press any key to accept.")

    while True:
        vis = frame.copy()
        if detected_center is not None:
            auto_pt = tuple(np.round(detected_center).astype(int))
            cv2.circle(vis, auto_pt, 16, (0, 200, 255), 2)
            cv2.putText(vis, "auto arbor center", (auto_pt[0] + 16, auto_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        if selected is not None:
            sel_pt = tuple(np.round(selected).astype(int))
            cv2.drawMarker(vis, sel_pt, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            if detected_center is not None:
                auto_pt = tuple(np.round(detected_center).astype(int))
                cv2.line(vis, auto_pt, sel_pt, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, vis)
        if cv2.waitKey(30) != -1 and selected is not None:
            break

    _close_cv_window(WIN)
    if selected is None:
        return None
    c = tuple(np.round(selected).astype(int))
    print(f"[center] Selected: ({c[0]}, {c[1]})")
    return np.array(c, dtype=float)


def _nearest_point(
    point: np.ndarray,
    candidates: np.ndarray,
    *,
    snap_radius_px: float = 28.0,
) -> np.ndarray:
    """Snap *point* to the nearest candidate if it is close enough."""
    if len(candidates) == 0:
        return np.asarray(point, dtype=float)
    point = np.asarray(point, dtype=float)
    dists = np.linalg.norm(candidates - point, axis=1)
    idx = int(np.argmin(dists))
    if float(dists[idx]) <= snap_radius_px:
        return np.asarray(candidates[idx], dtype=float)
    return point


def _scale_reference_px(
    center: np.ndarray,
    outer_point: np.ndarray,
    inner_radius_px: float,
    mode_key: str,
) -> float:
    """Reference pixel length for the selected scale mode."""
    outer_radius_px = float(np.linalg.norm(np.asarray(outer_point, dtype=float) - np.asarray(center, dtype=float)))
    if mode_key == "outer_radius":
        return outer_radius_px
    if mode_key == "radial_extent":
        return outer_radius_px - float(inner_radius_px)
    raise ValueError(f"Unsupported scale mode: {mode_key}")


def _format_panel_label(base: str, count: int, *, stabilized: bool | None = None) -> str:
    count_text = f"{count} dot{'s' if count != 1 else ''}"
    if stabilized is None:
        return f"{base}  |  {count_text}"
    return f"{base}  |  {count_text}  |  {'stabilized' if stabilized else 'raw'}"


def _calibrate_scale_gui(
    frame0: np.ndarray,
    frame_last: np.ndarray,
    frame_last_label: str,
    center: np.ndarray,
    ref_dots: np.ndarray,
    last_dots: np.ndarray,
    *,
    mode: _ScaleMode,
    inner_radius_px: float,
    inner_dot: np.ndarray,
    initial_outer_point: np.ndarray,
    show_inner_radius: bool,
    last_frame_stabilized: bool,
) -> tuple[float, np.ndarray]:
    """
    Interactive scale-calibration helper.

    The selected first frame defines the actual scale reference. The last frame is displayed so the
    user can probe distances there before the main analysis starts.
    """
    WIN = (
        "Scale Calibration  |  left-click reference frame to set reference  |  "
        "right-drag either panel to probe  |  Enter/Space to accept"
    )
    h, w = frame0.shape[:2]
    scale = min(1.0, 760.0 / max(w, 1), 820.0 / max(h, 1))
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))

    reference_mm = float(mode.reference_mm)
    initial_reference_mm = float(mode.reference_mm)
    reference_outer_point = np.asarray(initial_outer_point, dtype=float)
    initial_outer_point = np.asarray(initial_outer_point, dtype=float).copy()
    inner_dot = np.asarray(inner_dot, dtype=float)

    probe_panel: int | None = None
    probe_start: np.ndarray | None = None
    probe_end: np.ndarray | None = None
    probe_dragging = False
    accepted = False

    def from_display_xy(x: int, y: int) -> tuple[int | None, np.ndarray | None]:
        if y < 0 or y >= disp_h or x < 0 or x >= 2 * disp_w:
            return None, None
        panel_idx = 0 if x < disp_w else 1
        x_local = (x - panel_idx * disp_w) / scale
        y_local = y / scale
        point = np.array([
            float(np.clip(x_local, 0, w - 1)),
            float(np.clip(y_local, 0, h - 1)),
        ], dtype=float)
        return panel_idx, point

    def current_reference_px() -> float:
        return _scale_reference_px(center, reference_outer_point, inner_radius_px, mode.key)

    def current_px_per_mm() -> float | None:
        ref_px = current_reference_px()
        if reference_mm <= 0.0 or ref_px <= 1e-9:
            return None
        return ref_px / reference_mm

    def on_mouse(event, x, y, _flags, _param):
        nonlocal probe_panel, probe_start, probe_end, probe_dragging, reference_outer_point
        panel_idx, point = from_display_xy(x, y)
        if point is None or panel_idx is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN and panel_idx == 0:
            reference_outer_point = _nearest_point(point, ref_dots)
        elif event == cv2.EVENT_RBUTTONDOWN:
            probe_panel = panel_idx
            probe_start = point
            probe_end = point
            probe_dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and probe_dragging and panel_idx == probe_panel:
            probe_end = point
        elif event == cv2.EVENT_RBUTTONUP and probe_dragging and panel_idx == probe_panel:
            probe_end = point
            probe_dragging = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    print(
        "[scale] Calibration window:\n"
        "        left-click on the REFERENCE FRAME to choose the scale reference point\n"
        "        right-click and drag on either panel to probe distances\n"
        "        keys: [ ] = -/+1.00 mm, -/+ = -/+0.10 mm, ,/. = -/+0.01 mm,\n"
        "              r = reset reference, c = clear probe, Enter/Space = accept, Esc = keep original"
    )

    while True:
        px_per_mm = current_px_per_mm()
        reference_px = current_reference_px()
        panel_frames = [frame0.copy(), frame_last.copy()]
        panel_dots = [ref_dots, last_dots]
        panel_labels = [
            _format_panel_label("Reference frame (scale reference)", len(ref_dots)),
            _format_panel_label(frame_last_label, len(last_dots), stabilized=last_frame_stabilized),
        ]

        for panel_idx, (panel, dots, label) in enumerate(zip(panel_frames, panel_dots, panel_labels)):
            if len(dots):
                for dot in dots:
                    pt = tuple(np.round(dot).astype(int))
                    cv2.circle(panel, pt, 10, (0, 220, 0), 2, cv2.LINE_AA)

            show_center = panel_idx == 0 or last_frame_stabilized
            if show_center:
                center_pt = tuple(np.round(center).astype(int))
                cv2.drawMarker(panel, center_pt, (0, 0, 255), cv2.MARKER_STAR, 26, 2)
                cv2.putText(panel, "center", (center_pt[0] + 16, center_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

            if panel_idx == 0:
                center_pt = tuple(np.round(center).astype(int))
                ref_pt = tuple(np.round(reference_outer_point).astype(int))
                cv2.line(panel, center_pt, ref_pt, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.circle(panel, ref_pt, 12, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, mode.label, (ref_pt[0] + 14, ref_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)
                if show_inner_radius:
                    inner_pt = tuple(np.round(inner_dot).astype(int))
                    cv2.line(panel, center_pt, inner_pt, (255, 120, 0), 2, cv2.LINE_AA)
                    cv2.circle(panel, inner_pt, 10, (255, 120, 0), 2, cv2.LINE_AA)
                    cv2.putText(panel, "inner radius", (inner_pt[0] + 14, inner_pt[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2, cv2.LINE_AA)

            if probe_panel == panel_idx and probe_start is not None and probe_end is not None:
                p0 = tuple(np.round(probe_start).astype(int))
                p1 = tuple(np.round(probe_end).astype(int))
                cv2.line(panel, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(panel, p0, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, p1, 6, (255, 255, 255), -1, cv2.LINE_AA)
                probe_px = float(np.linalg.norm(probe_end - probe_start))
                probe_text = f"probe: {probe_px:.2f} px"
                if px_per_mm is not None:
                    probe_text += f"  |  {probe_px / px_per_mm:.3f} mm"
                mid = ((p0[0] + p1[0]) // 2 + 10, (p0[1] + p1[1]) // 2 - 10)
                (tw, th), baseline = cv2.getTextSize(
                    probe_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    3,
                )
                box_tl = (max(0, mid[0] - 8), max(0, mid[1] - th - 10))
                box_br = (
                    min(panel.shape[1] - 1, mid[0] + tw + 8),
                    min(panel.shape[0] - 1, mid[1] + baseline + 8),
                )
                cv2.rectangle(panel, box_tl, box_br, (20, 20, 20), -1)
                cv2.rectangle(panel, box_tl, box_br, (255, 255, 255), 1)
                cv2.putText(panel, probe_text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 3, cv2.LINE_AA)

            cv2.rectangle(panel, (0, 0), (panel.shape[1], 84), (20, 20, 20), -1)
            cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "Green circles = detected dots", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA)
            if panel_idx == 0:
                cv2.putText(panel, "Left-click here to move the scale reference", (12, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, "Right-drag here to check a distance before tracking", (12, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        strip = np.concatenate(
            [cv2.resize(panel, (disp_w, disp_h)) for panel in panel_frames],
            axis=1,
        )

        footer_h = 106
        canvas = np.zeros((disp_h + footer_h, strip.shape[1], 3), dtype=np.uint8)
        canvas[:disp_h] = strip

        if reference_px <= 1e-9:
            scale_line = f"{mode.label}: invalid reference length ({reference_px:.3f} px)"
            color = (0, 0, 255)
        else:
            scale_line = f"{mode.label}: {reference_px:.3f} px = {reference_mm:.3f} mm"
            if px_per_mm is not None:
                scale_line += f"  |  {px_per_mm:.5f} px/mm  |  {1.0 / px_per_mm:.6f} mm/px"
            color = (255, 255, 255)

        hint_line = (
            "Keys: [ ] = +/-1.00 mm   -/+ = +/-0.10 mm   ,/. = +/-0.01 mm   "
            "r reset   c clear probe   Enter/Space accept   Esc keep original"
        )
        cv2.putText(canvas, scale_line, (12, disp_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    color, 2, cv2.LINE_AA)
        cv2.putText(canvas, hint_line, (12, disp_h + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Reference scaling always comes from the REFERENCE FRAME; the last frame is a probe/check view.",
                    (12, disp_h + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (160, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30)
        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                reference_mm = initial_reference_mm
                reference_outer_point = initial_outer_point.copy()
                break
            continue

        key &= 0xFF
        if key in (13, 10, 32):
            if current_reference_px() > 1e-9 and reference_mm > 0.0:
                accepted = True
                break
            print("[scale] Reference is invalid; choose a longer reference-frame segment before accepting.")
        elif key == 27:
            reference_mm = initial_reference_mm
            reference_outer_point = initial_outer_point.copy()
            break
        elif key in (ord("["), ord("{")):
            reference_mm = max(0.01, reference_mm - 1.0)
        elif key in (ord("]"), ord("}")):
            reference_mm += 1.0
        elif key in (ord("-"), ord("_")):
            reference_mm = max(0.01, reference_mm - 0.1)
        elif key in (ord("="), ord("+")):
            reference_mm += 0.1
        elif key == ord(","):
            reference_mm = max(0.01, reference_mm - 0.01)
        elif key == ord("."):
            reference_mm += 0.01
        elif key in (ord("r"), ord("R")):
            reference_mm = initial_reference_mm
            reference_outer_point = initial_outer_point.copy()
        elif key in (ord("c"), ord("C")):
            probe_panel = None
            probe_start = None
            probe_end = None
            probe_dragging = False

    _close_cv_window(WIN)
    final_reference_px = _scale_reference_px(center, reference_outer_point, inner_radius_px, mode.key)
    status = "accepted" if accepted else "kept original"
    print(
        f"[scale] Calibration {status}: {mode.label} = "
        f"{final_reference_px:.3f} px / {reference_mm:.3f} mm"
    )
    return reference_mm, reference_outer_point


def _expand_rect(
    rect_xywh: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    """Expand *rect_xywh* about its centre and clip it to *frame_shape*."""
    h, w = frame_shape[:2]
    x, y, rw, rh = rect_xywh
    cx = x + 0.5 * rw
    cy = y + 0.5 * rh
    new_w = max(1.0, float(rw) * scale_x)
    new_h = max(1.0, float(rh) * scale_y)
    x0 = int(np.clip(round(cx - 0.5 * new_w), 0, max(w - 1, 0)))
    y0 = int(np.clip(round(cy - 0.5 * new_h), 0, max(h - 1, 0)))
    x1 = int(np.clip(round(cx + 0.5 * new_w), x0 + 1, w))
    y1 = int(np.clip(round(cy + 0.5 * new_h), y0 + 1, h))
    return x0, y0, x1 - x0, y1 - y0


def _pick_rectangle_gui(
    frame: np.ndarray,
    *,
    window_title: str,
    prompt_lines: tuple[str, ...],
    initial_rect: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pick or adjust a rectangle on *frame* with a simple drag GUI."""
    h, w = frame.shape[:2]
    rect = (
        initial_rect
        if initial_rect is not None
        else (0, 0, w, h)
    )
    rect = _expand_rect(rect, frame.shape, scale_x=1.0, scale_y=1.0)
    display_scale = min(1.0, 1500.0 / max(w, 1), 920.0 / max(h, 1))
    disp_w = max(1, int(round(w * display_scale)))
    disp_h = max(1, int(round(h * display_scale)))
    preview_base = (
        cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        if display_scale < 0.999
        else frame.copy()
    )
    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None
    drag_active = False

    def _clamp_point(x: int, y: int) -> tuple[int, int]:
        return (
            int(np.clip(x, 0, max(w - 1, 0))),
            int(np.clip(y, 0, max(h - 1, 0))),
        )

    def _display_to_frame_point(x: int, y: int) -> tuple[int, int]:
        if display_scale < 0.999:
            xf = int(round(float(x) / display_scale))
            yf = int(round(float(y) / display_scale))
        else:
            xf = int(x)
            yf = int(y)
        return _clamp_point(xf, yf)

    def _frame_to_display_point(point_xy: tuple[int, int]) -> tuple[int, int]:
        x, y = point_xy
        if display_scale < 0.999:
            xd = int(round(float(x) * display_scale))
            yd = int(round(float(y) * display_scale))
        else:
            xd = int(x)
            yd = int(y)
        return (
            int(np.clip(xd, 0, max(disp_w - 1, 0))),
            int(np.clip(yd, 0, max(disp_h - 1, 0))),
        )

    def _normalize_box(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[int, int, int, int]:
        x0 = min(p0[0], p1[0])
        y0 = min(p0[1], p1[1])
        x1 = max(p0[0], p1[0])
        y1 = max(p0[1], p1[1])
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _clear_drag() -> None:
        nonlocal drag_start, drag_current, drag_active
        drag_start = None
        drag_current = None
        drag_active = False

    def _commit_drag(point_xy: tuple[int, int]) -> None:
        nonlocal rect
        if drag_start is None:
            _clear_drag()
            return
        candidate = _normalize_box(drag_start, point_xy)
        if candidate[2] > 4 and candidate[3] > 4:
            rect = candidate
        _clear_drag()

    def _rect_to_display(rect_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, rw, rh = rect_xywh
        p0 = _frame_to_display_point((x, y))
        p1 = _frame_to_display_point((x + rw, y + rh))
        x0 = min(p0[0], p1[0])
        y0 = min(p0[1], p1[1])
        x1 = max(p0[0], p1[0])
        y1 = max(p0[1], p1[1])
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drag_start, drag_current, drag_active
        point = _display_to_frame_point(int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            # Always start a fresh drag, even if a previous release was missed.
            drag_start = point
            drag_current = point
            drag_active = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag_active:
                drag_current = point
        elif event == cv2.EVENT_LBUTTONUP:
            if drag_active:
                drag_current = point
                _commit_drag(point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            _clear_drag()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, disp_w, disp_h)
    cv2.setMouseCallback(window_title, on_mouse)
    print("\n".join(prompt_lines))

    while True:
        vis = preview_base.copy()
        draw_rect = rect
        if drag_active and drag_start is not None and drag_current is not None:
            draw_rect = _normalize_box(drag_start, drag_current)
        x, y, rw, rh = _rect_to_display(draw_rect)
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), (0, 215, 255), 3, cv2.LINE_AA)
        if drag_active and drag_start is not None and drag_current is not None:
            start_pt = _frame_to_display_point(drag_start)
            cur_pt = _frame_to_display_point(drag_current)
            cv2.circle(vis, start_pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, cur_pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
        vis = _put_info_box(
            vis,
            [
                "Force-meter search area",
                "Drag a coarse box around the luggage scale / screen region",
                "Enter/Space = accept   R = reset full frame   C/right-click = cancel current drag",
                (
                    f"display scaled to {display_scale * 100:.0f}% for smoother interaction"
                    if display_scale < 0.999
                    else "display shown at full resolution"
                ),
            ],
            origin=(18, 54),
        )
        cv2.imshow(window_title, vis)
        key = cv2.waitKey(30)
        if key == -1:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        key &= 0xFF
        if key in (13, 10, 32):
            if drag_active and drag_current is not None:
                _commit_drag(drag_current)
            break
        if key in (ord("r"), ord("R")):
            rect = (0, 0, w, h)
            _clear_drag()
        elif key in (ord("c"), ord("C")):
            _clear_drag()
        elif key == 27:
            _clear_drag()
            break

    _close_cv_window(window_title)
    return rect


def _translate_frame(frame: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
    """Translate *frame* by (dx, dy) while preserving the original canvas size."""
    h, w = frame.shape[:2]
    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _stabilize_frame(
    frame: np.ndarray,
    arbor_anchor: np.ndarray,
    arbor_cfg: DetectConfig,
    prev_arbor_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Translate *frame* so the detected arbor-marker centroid aligns to *arbor_anchor*.

    Returns (stabilized_frame, raw_arbor_center, shift_xy).
    """
    arbor_center = detect_primary_blob_center(
        frame,
        arbor_cfg,
        preferred_center=prev_arbor_center,
    )
    if arbor_center is None:
        return frame, None, np.zeros(2, dtype=float)

    shift_xy = arbor_anchor - arbor_center
    stabilized = _translate_frame(frame, shift_xy)
    return stabilized, arbor_center, shift_xy


def _center_from_arbor_offset(
    arbor_center: np.ndarray | None,
    center_offset: np.ndarray | None,
) -> np.ndarray | None:
    """Convert an arbor-marker centroid to the corresponding spring centre."""
    if arbor_center is None or center_offset is None:
        return None
    return np.asarray(arbor_center, dtype=float) + np.asarray(center_offset, dtype=float)


def _unwrap_angles(
    current_wrapped: np.ndarray,
    previous_unwrapped: np.ndarray,
) -> np.ndarray:
    """Unwrap current angles against the previous unwrapped frame angles."""
    delta = (current_wrapped - previous_unwrapped + np.pi) % (2.0 * np.pi) - np.pi
    return previous_unwrapped + delta


def _camera_ures_curve(
    rows: pd.DataFrame,
    arc_col: str,
    disp_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return camera-tracked URES curve data with the lower-displacement endpoint at s=0."""
    ordered = rows.sort_values("dot_id").copy()
    x = (ordered[arc_col] / ordered[arc_col].max()).to_numpy(dtype=float)
    y = ordered[disp_col].to_numpy(dtype=float)
    if len(y) >= 2 and y[-1] < y[0]:
        x = 1.0 - x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        ordered = ordered.iloc[sort_idx].reset_index(drop=True)
    return x, y, ordered


def _moving_endpoint_row(
    rows: pd.DataFrame,
    disp_col: str,
) -> pd.Series | None:
    """Return the endpoint row with the larger displacement."""
    ordered = rows.sort_values("dot_id")
    if ordered.empty:
        return None
    if len(ordered) == 1:
        return ordered.iloc[0]
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    return first if float(first[disp_col]) >= float(last[disp_col]) else last


def _unwrap_scalar_angle(
    current_wrapped: float,
    previous_unwrapped: float,
) -> float:
    """Scalar counterpart to _unwrap_angles for a single tracked angle."""
    delta = (float(current_wrapped) - float(previous_unwrapped) + np.pi) % (2.0 * np.pi) - np.pi
    return float(previous_unwrapped + delta)


def _curve_endpoint_indices(dot_arc_ids: np.ndarray) -> tuple[int, int]:
    """Return the reference-dot indices at the two ends of the fitted spiral."""
    order = np.asarray(dot_arc_ids, dtype=int).reshape(-1)
    return int(np.argmin(order)), int(np.argmax(order))


def _select_arbor_rigid_endpoint_index(
    ref_points: np.ndarray,
    arbor_marker_ref: np.ndarray,
    endpoint_indices: tuple[int, int],
) -> int:
    """
    Choose the spiral endpoint rigidly attached to the arbor.

    The arbor-attached endpoint is the curve end nearest to the arbor marker in the
    reference frame.
    """
    ref_pts = np.asarray(ref_points, dtype=float)
    arbor_ref = np.asarray(arbor_marker_ref, dtype=float).reshape(2)
    best_idx = int(endpoint_indices[0])
    best_dist = float("inf")
    for idx in endpoint_indices:
        endpoint = np.asarray(ref_pts[int(idx)], dtype=float).reshape(2)
        dist = float(np.linalg.norm(endpoint - arbor_ref))
        if dist < best_dist:
            best_dist = dist
            best_idx = int(idx)
    return best_idx


def _vector_angle_rad(
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
) -> float | None:
    """Polar angle of the vector from origin_xy to target_xy, in radians."""
    vec = np.asarray(target_xy, dtype=float).reshape(2) - np.asarray(origin_xy, dtype=float).reshape(2)
    if not np.isfinite(vec).all() or float(np.linalg.norm(vec)) <= 1e-12:
        return None
    return float(np.arctan2(vec[1], vec[0]))


def _arbor_rotation_fields(
    *,
    rotation_deg: float | None,
    rotation_signed_deg: float | None,
    rigid_dot_id: int | None,
    method: str | None,
) -> dict[str, object]:
    return {
        "arbor_rotation_deg": (
            float(rotation_deg)
            if rotation_deg is not None and np.isfinite(float(rotation_deg))
            else np.nan
        ),
        "arbor_rotation_signed_deg": (
            float(rotation_signed_deg)
            if rotation_signed_deg is not None and np.isfinite(float(rotation_signed_deg))
            else np.nan
        ),
        "arbor_rigid_dot_id": (
            float(rigid_dot_id)
            if rigid_dot_id is not None
            else np.nan
        ),
        "arbor_rotation_method": "" if method is None else str(method),
    }


def _ures_tip_angle_deg(u_tip: float, r_tip: float) -> float:
    """Tip angle implied by the URES/chord relation."""
    return float(2.0 * np.degrees(np.arcsin(np.clip(u_tip / max(2.0 * r_tip, 1e-12), -1.0, 1.0))))


def _optimizer_rotation_deg(spring) -> float | None:
    """Applied optimizer rotation from the Spring object, in degrees."""
    rom = getattr(getattr(spring, "specs", None), "rom", None)
    if rom is None:
        return None
    return float(np.degrees(rom))


def _annotate_endpoint_displacement(
    ax,
    y_value: float,
    label: str,
    color: str | tuple[float, float, float] | tuple[int, int, int],
    *,
    linestyle: str = "--",
    text_offset_pts: float = 0.0,
) -> None:
    """Draw a horizontal guide plus an axis marker for an endpoint displacement."""
    ax.axhline(y_value, color=color, lw=1.0, ls=linestyle, alpha=0.65, zorder=1)
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(
        [0.0],
        [y_value],
        marker=">",
        ms=6,
        color=color,
        transform=trans,
        clip_on=False,
        zorder=6,
    )
    ax.annotate(
        label,
        xy=(1.0, y_value),
        xycoords=trans,
        xytext=(-6, text_offset_pts),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color=color,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85),
    )


def _apply_ures_noise_floor(values: np.ndarray, floor: float = URES_NOISE_FLOOR_M) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[np.abs(out) < float(floor)] = 0.0
    return out


def _choose_reference_pkl(pkls: list[Path]) -> Path:
    preferred = [p for p in pkls if not p.stem.endswith("_sweep")]
    candidates = preferred or pkls
    if len(candidates) > 1:
        print(f"[warn] Multiple Spring .pkl files found; using {candidates[0].name}")
    return candidates[0]


def _find_reference_spring_pkl(pkl_dir: Path, legacy_dir: Path | None = None) -> Path | None:
    pkls = sorted(pkl_dir.glob("*.pkl"))
    if pkls:
        return _choose_reference_pkl(pkls)

    if legacy_dir is not None:
        legacy_pkls = sorted(legacy_dir.glob("*.pkl"))
        if legacy_pkls:
            print(f"[warn] No Spring .pkl found in {pkl_dir}; falling back to {legacy_dir}")
            return _choose_reference_pkl(legacy_pkls)
    return None


def _load_reference_spring(path: Path):
    return load_pickle(path)


def _sample_spiral_points(
    center: np.ndarray,
    a: float,
    b: float,
    thetas: np.ndarray,
    n_pts: int = 600,
) -> np.ndarray:
    """Sample the fitted spiral in the working coordinate system."""
    th_range = np.linspace(float(thetas.min()), float(thetas.max()), n_pts)
    r_range = a + b * th_range
    return np.column_stack((
        center[0] + r_range * np.cos(th_range),
        center[1] + r_range * np.sin(th_range),
    ))


def _draw_polyline(
    img: np.ndarray,
    pts_xy: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 220, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a polyline through a sequence of 2D points."""
    out = img.copy()
    pts = np.asarray(pts_xy, dtype=float)
    if len(pts) < 2:
        return out

    h, w = out.shape[:2]
    valid = [
        (int(round(pt[0])), int(round(pt[1])))
        for pt in pts
        if np.isfinite(pt[0]) and np.isfinite(pt[1]) and 0 <= pt[0] < w and 0 <= pt[1] < h
    ]
    for p0, p1 in zip(valid, valid[1:]):
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    return out


def _draw_spiral(
    img: np.ndarray,
    center: np.ndarray,
    a: float,
    b: float,
    thetas: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 255),
    n_pts: int = 600,
) -> np.ndarray:
    """Overlay the fitted Archimedean spiral r = a + b·θ on *img* (in-place copy)."""
    spiral_pts = _sample_spiral_points(center, a, b, thetas, n_pts=n_pts)
    return _draw_polyline(img, spiral_pts, color=color, thickness=2)


def _colormap(n: int) -> list[tuple[int, int, int]]:
    cmap = plt.colormaps["rainbow"].resampled(max(n, 2))
    return [(int(cmap(i)[2] * 255), int(cmap(i)[1] * 255), int(cmap(i)[0] * 255))
            for i in range(n)]


def _draw_overlay(
    frame: np.ndarray,
    center: np.ndarray,
    dot_positions: np.ndarray,   # (M, 2)
    dot_arc_ids: np.ndarray,     # (M,) int – arc-length rank
    colors: list[tuple[int, int, int]],
    ref_positions: np.ndarray,   # (N, 2) – grey ghost crosses
    skipped: bool = False,
    skip_reason: str = "",
    used_fallback: bool = False,
    frame_label: str = "",
    info_lines: list[str] | None = None,
) -> np.ndarray:
    out = frame.copy()
    # Ghost reference positions
    for rp in ref_positions:
        cv2.drawMarker(out, (int(rp[0]), int(rp[1])), (90, 90, 90),
                       cv2.MARKER_CROSS, 14, 1)
    # Centre
    cv2.drawMarker(out, (int(center[0]), int(center[1])), (0, 0, 255),
                   cv2.MARKER_STAR, 22, 2)
    # Dot positions
    for pos, did in zip(dot_positions, dot_arc_ids):
        x, y = int(pos[0]), int(pos[1])
        color = colors[did % len(colors)]
        cv2.circle(out, (x, y), 12, color, -1)
        cv2.putText(out, str(did), (x + 14, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    if skipped:
        cv2.putText(out, f"SKIP  {skip_reason}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4, cv2.LINE_AA)
    if used_fallback:
        cv2.putText(out, "fallback cfg", (20, out.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2, cv2.LINE_AA)
    if frame_label:
        (tw, th), baseline = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(out, (14, 12), (30 + tw, 28 + th + baseline), (20, 20, 20), -1)
        cv2.rectangle(out, (14, 12), (30 + tw, 28 + th + baseline), (255, 255, 255), 1)
        cv2.putText(out, frame_label, (22, 22 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
    if info_lines:
        out = _put_top_right_info_box(out, info_lines)
    return out


def _draw_apriltag_boxes(
    img: np.ndarray,
    detections: dict[int, np.ndarray],
    *,
    color: tuple[int, int, int] = (80, 255, 80),
    labels: dict[int, str] | None = None,
) -> np.ndarray:
    out = img.copy()
    for tag_id, corners in sorted(detections.items()):
        pts = np.round(np.asarray(corners, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 4:
            continue
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        anchor = tuple(int(v) for v in pts[0, 0])
        label = f"tag {tag_id}"
        if labels is not None and tag_id in labels:
            label = f"{label}  {labels[tag_id]}"
        cv2.putText(
            out,
            label,
            (anchor[0] + 8, anchor[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_labeled_segment(
    img: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    label: str,
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = img.copy()
    pt0 = tuple(np.round(np.asarray(p0, dtype=float)).astype(int))
    pt1 = tuple(np.round(np.asarray(p1, dtype=float)).astype(int))
    cv2.line(out, pt0, pt1, color, 2, cv2.LINE_AA)
    cv2.circle(out, pt0, 7, color, -1, cv2.LINE_AA)
    cv2.circle(out, pt1, 7, color, -1, cv2.LINE_AA)
    mid = ((pt0[0] + pt1[0]) // 2 + 10, (pt0[1] + pt1[1]) // 2 - 10)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (mid[0] - 6, mid[1] - th - 8), (mid[0] + tw + 6, mid[1] + baseline + 6), (24, 24, 24), -1)
    cv2.rectangle(out, (mid[0] - 6, mid[1] - th - 8), (mid[0] + tw + 6, mid[1] + baseline + 6), color, 1)
    cv2.putText(out, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


def _put_info_box(
    img: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (18, 56),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    pad = 10
    line_gap = 10
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max((w for w, _ in sizes), default=0)
    text_height = sum(h for _, h in sizes) + max(0, len(lines) - 1) * line_gap
    top_left = (origin[0] - pad, origin[1] - sizes[0][1] - pad if sizes else origin[1] - pad)
    bottom_right = (origin[0] + width + pad, top_left[1] + text_height + 2 * pad)
    cv2.rectangle(out, top_left, bottom_right, (24, 24, 24), -1)
    cv2.rectangle(out, top_left, bottom_right, (255, 255, 255), 1)
    y = origin[1]
    for line, (_w, h) in zip(lines, sizes):
        cv2.putText(out, line, (origin[0], y), font, scale, text_color, thickness, cv2.LINE_AA)
        y += h + line_gap
    return out


def _put_top_right_info_box(
    img: np.ndarray,
    lines: list[str],
    *,
    margin: int = 18,
    top: int = 54,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Place `_put_info_box` content in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    pad = 10
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_width = max(widths, default=0) + 2 * pad
    origin_x = max(margin, img.shape[1] - box_width - margin + pad)
    return _put_info_box(img, lines, origin=(origin_x, top), text_color=text_color)


def _prompt_retune_gui(
    frame: np.ndarray,
    *,
    first_bad_idx: int,
    first_bad_reason: str,
    bad_count: int,
    fps: float,
) -> bool:
    """Ask whether to re-tune using an OpenCV window instead of terminal stdin."""
    win = "Scan Retune Prompt  |  Y retune  |  N finalize"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        try:
            answer = input(_RETUNE_PROMPT).strip().lower()
        except EOFError:
            print("[scan] No interactive stdin available; finalizing with the current parameters.")
            return False
        return answer == "y"

    print("[scan] Retune prompt opened in the OpenCV window. Press Y to re-tune or N to finalize.")
    time_s = first_bad_idx / fps
    lines = [
        "Problematic frames found during scan",
        f"first bad frame = {first_bad_idx}  (t = {time_s:.2f} s)",
        f"reason = {first_bad_reason}",
        f"bad frames pending = {bad_count}",
        "Y / Enter / Space = open tuner on this frame",
        "N / Esc / Q = finalize and skip bad frames",
    ]

    while True:
        vis = frame.copy()
        vis = _put_info_box(vis, lines, origin=(18, 54))
        vis = _put_top_right_info_box(
            vis,
            [
                "Retune decision",
                "Stay in this window",
                "No terminal focus needed",
            ],
            text_color=(170, 220, 255),
        )
        cv2.imshow(win, vis)
        key = cv2.waitKey(30)
        if key == -1:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    _close_cv_window(win)
                    return False
            except cv2.error:
                return False
            continue
        key &= 0xFF
        if key in (ord("y"), ord("Y"), 13, 10, 32):
            _close_cv_window(win)
            return True
        if key in (ord("n"), ord("N"), ord("q"), ord("Q"), 27):
            _close_cv_window(win)
            return False


def _make_lens_distortion_comparison(
    raw_frame: np.ndarray,
    undistorted_frame: np.ndarray,
    raw_tag_detections: dict[int, np.ndarray],
    undistorted_tag_detections: dict[int, np.ndarray],
    distorted_dot_positions: np.ndarray,
    undistorted_dot_positions: np.ndarray,
    distortion_shift_map_px: np.ndarray,
    reference_px_per_mm: float | None,
    *,
    frame_label: str,
) -> np.ndarray:
    _ = raw_frame
    panel_overlay = _draw_apriltag_boxes(undistorted_frame, undistorted_tag_detections, color=(80, 255, 80))

    all_shifts: list[float] = []
    for raw_pt, undist_pt in zip(np.asarray(distorted_dot_positions, dtype=float), np.asarray(undistorted_dot_positions, dtype=float)):
        if not (np.isfinite(raw_pt).all() and np.isfinite(undist_pt).all()):
            continue
        all_shifts.append(float(np.linalg.norm(undist_pt - raw_pt)))
        p0 = tuple(np.round(raw_pt).astype(int))
        p1 = tuple(np.round(undist_pt).astype(int))
        cv2.arrowedLine(panel_overlay, p0, p1, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.22)
        cv2.circle(panel_overlay, p0, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel_overlay, p1, 4, (255, 0, 255), -1, cv2.LINE_AA)

    for tag_id in sorted(set(raw_tag_detections) & set(undistorted_tag_detections)):
        raw_corners = np.asarray(raw_tag_detections[tag_id], dtype=float)
        undist_corners = np.asarray(undistorted_tag_detections[tag_id], dtype=float)
        raw_poly = np.round(raw_corners).astype(np.int32).reshape(-1, 1, 2)
        if len(raw_poly) >= 4:
            cv2.polylines(panel_overlay, [raw_poly], True, (0, 255, 255), 1, cv2.LINE_AA)
        for raw_pt, undist_pt in zip(raw_corners, undist_corners):
            if not (np.isfinite(raw_pt).all() and np.isfinite(undist_pt).all()):
                continue
            all_shifts.append(float(np.linalg.norm(undist_pt - raw_pt)))
            p0 = tuple(np.round(raw_pt).astype(int))
            p1 = tuple(np.round(undist_pt).astype(int))
            cv2.arrowedLine(panel_overlay, p0, p1, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.20)
            cv2.circle(panel_overlay, p0, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel_overlay, p1, 4, (255, 0, 255), -1, cv2.LINE_AA)

    panel_overlay = _put_info_box(
        panel_overlay,
        (
            [
                "Raw marker positions drawn on corrected image",
                "cyan = distorted positions   magenta = corrected positions",
            ]
            + (
                [
                    f"mean correction shift = {np.mean(all_shifts):.2f} px",
                    f"max correction shift = {np.max(all_shifts):.2f} px",
                ]
                if all_shifts
                else []
            )
            + [frame_label]
        ),
        origin=(18, 44),
    )

    shift_map = np.asarray(distortion_shift_map_px, dtype=np.float32)
    finite = np.isfinite(shift_map)
    shift_min = float(np.min(shift_map[finite])) if np.any(finite) else 0.0
    shift_max = float(np.max(shift_map[finite])) if np.any(finite) else 0.0
    shift_mean = float(np.mean(shift_map[finite])) if np.any(finite) else 0.0
    if shift_max > shift_min + 1e-9:
        normalized = np.clip((shift_map - shift_min) / (shift_max - shift_min), 0.0, 1.0)
    else:
        normalized = np.zeros_like(shift_map, dtype=np.float32)
    heat_u8 = np.zeros_like(shift_map, dtype=np.uint8)
    heat_u8[finite] = np.asarray(np.round(normalized[finite] * 255.0), dtype=np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    panel_heatmap = cv2.addWeighted(undistorted_frame, 0.40, heat_color, 0.60, 0.0)
    frame_diag_px = float(np.hypot(panel_heatmap.shape[1], panel_heatmap.shape[0]))

    bar_h = 320
    bar_w = 36
    bar_x1 = panel_heatmap.shape[1] - 24
    bar_x0 = bar_x1 - bar_w
    bar_y0 = 72
    bar_y1 = min(panel_heatmap.shape[0] - 24, bar_y0 + bar_h)
    grad = np.linspace(255, 0, bar_y1 - bar_y0, dtype=np.uint8).reshape(-1, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    panel_heatmap[bar_y0:bar_y1, bar_x0:bar_x1] = grad_color
    cv2.rectangle(panel_heatmap, (bar_x0 - 1, bar_y0 - 1), (bar_x1 + 1, bar_y1 + 1), (255, 255, 255), 1)
    mm_scale_valid = reference_px_per_mm is not None and float(reference_px_per_mm) > 1e-9
    shift_max_mm = shift_max / float(reference_px_per_mm) if mm_scale_valid else np.nan
    shift_min_mm = shift_min / float(reference_px_per_mm) if mm_scale_valid else np.nan
    top_label = f"{shift_max:.2f}px"
    bot_label = f"{shift_min:.2f}px"
    if mm_scale_valid:
        top_label += f" / {shift_max_mm:.3f}mm"
        bot_label += f" / {shift_min_mm:.3f}mm"
    cv2.putText(panel_heatmap, top_label, (bar_x0 - 170, bar_y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel_heatmap, bot_label, (bar_x0 - 170, bar_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    panel_heatmap = _put_info_box(
        panel_heatmap,
        [
            "Warp magnitude heatmap",
            "color = correction displacement in pixels",
            *(["mm labels use reference-frame px/mm"] if mm_scale_valid else []),
            f"mean shift = {shift_mean:.2f} px ({100.0 * shift_mean / max(frame_diag_px, 1.0):.3f}% diag)",
            f"max shift = {shift_max:.2f} px ({100.0 * shift_max / max(frame_diag_px, 1.0):.3f}% diag)",
            frame_label,
        ],
        origin=(18, 44),
    )

    canvas = np.concatenate((panel_overlay, panel_heatmap), axis=1)
    cv2.putText(
        canvas,
        "Lens distortion adjustment comparison",
        (24, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _make_metric_reference_panel(
    frame: np.ndarray,
    center_px: np.ndarray,
    inner_dot_px: np.ndarray,
    outer_dot_px: np.ndarray,
    inner_radius_mm: float,
    outer_radius_mm: float,
    radial_span_mm: float,
    tag_size_mm: float,
    tag_detections: dict[int, np.ndarray],
    tag_width_labels: dict[int, str],
    tag_width_stats: tuple[float, float] | None,
    *,
    frame_label: str,
) -> np.ndarray:
    out = _draw_apriltag_boxes(frame, tag_detections, color=(80, 255, 80), labels=tag_width_labels)
    out = _draw_labeled_segment(
        out,
        center_px,
        inner_dot_px,
        f"inner radius = {inner_radius_mm:.2f} mm",
        color=(0, 170, 255),
    )
    out = _draw_labeled_segment(
        out,
        center_px,
        outer_dot_px,
        f"outer radius = {outer_radius_mm:.2f} mm",
        color=(255, 220, 0),
    )
    cv2.drawMarker(out, tuple(np.round(center_px).astype(int)), (0, 0, 255), cv2.MARKER_STAR, 24, 2)
    cv2.circle(out, tuple(np.round(inner_dot_px).astype(int)), 10, (0, 170, 255), 2, cv2.LINE_AA)
    cv2.circle(out, tuple(np.round(outer_dot_px).astype(int)), 10, (255, 220, 0), 2, cv2.LINE_AA)
    out = _put_info_box(
        out,
        (
            [
                "Metric reference panel",
                f"AprilTag black-square width = {tag_size_mm:.2f} mm",
            ]
            + (
                [
                    f"mean measured tag width = {tag_width_stats[0]:.2f} mm",
                    f"max abs tag-width error = {tag_width_stats[1]:.2f} mm",
                ]
                if tag_width_stats is not None
                else []
            )
            + [
                f"radial span (outer - inner) = {radial_span_mm:.2f} mm",
                frame_label,
            ]
        ),
        origin=(18, 44),
    )
    return out


# ── Two-config detection ──────────────────────────────────────────────────────

def _detect_best(
    frame: np.ndarray,
    cfg_primary: DetectConfig,
    cfg_fallback: DetectConfig | None,
    n_expected: int,
) -> tuple[np.ndarray, bool]:
    """
    Try *cfg_primary* first. If dot count is wrong and *cfg_fallback* is set,
    try that too. Returns (dots, used_fallback).
    """
    dots = detect_green_dots(frame, cfg_primary)
    if len(dots) == n_expected or cfg_fallback is None:
        return dots, False
    dots2 = detect_green_dots(frame, cfg_fallback)
    if len(dots2) == n_expected:
        return dots2, True
    # Both wrong — return whichever is closer to expected count
    if abs(len(dots2) - n_expected) < abs(len(dots) - n_expected):
        return dots2, True
    return dots, False


def _estimate_max_jump(frame: np.ndarray, cfg: DetectConfig) -> float:
    """2.5× the equivalent-circle diameter of the average accepted blob."""
    mask = hsv_mask(frame, cfg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts
             if cfg.min_area <= cv2.contourArea(c) <= cfg.max_area]
    if not areas:
        return 60.0
    return float(2.5 * 2.0 * np.sqrt(np.mean(areas) / np.pi))


def _find_force_setup_frame(
    cap: cv2.VideoCapture,
    cfg: DetectConfig,
    *,
    rotate: int,
    frame_undistorter: FrameUndistorter | None,
    start_frame_idx: int,
    end_frame_idx: int,
    frame_step: int,
) -> tuple[int, np.ndarray, tuple[int, int, int, int]] | None:
    """Pick the frame with the strongest detected force-meter screen."""
    sample_step = max(1, frame_step * 5)
    best: tuple[float, int, np.ndarray, tuple[int, int, int, int]] | None = None
    for frame_idx in range(start_frame_idx, end_frame_idx + 1, sample_step):
        frame = _read_frame_at(cap, frame_idx, rotate, frame_undistorter)
        if frame is None:
            continue
        detection = detect_force_screen(frame, cfg)
        if detection is None:
            continue
        score = float(detection.contour_area_px)
        if best is None or score > best[0]:
            best = (score, frame_idx, frame.copy(), detection.bbox_xywh)
    if best is None:
        return None
    _, frame_idx, frame, bbox = best
    return frame_idx, frame, bbox


def _force_info_lines(reading: ForceMeterReading | None) -> list[str] | None:
    if reading is None:
        return None
    if reading.mass_kg is not None:
        return [
            "Force meter",
            f"mass = {reading.mass_kg:.3f} kg",
            f"read conf = {reading.confidence:.2f}",
        ]
    if reading.screen_visible:
        return [
            "Force meter",
            "mass = unreadable",
            (
                f"{reading.reason}"
                if reading.reason
                else f"read conf = {reading.confidence:.2f}"
            ),
        ]
    return [
        "Force meter",
        "screen not visible",
    ]


def _force_record_fields(reading: ForceMeterReading | None) -> dict[str, object]:
    if reading is None:
        return {
            "mass_kg": np.nan,
            "force_n": np.nan,
            "force_confidence": np.nan,
            "force_screen_visible": False,
            "force_digits": "",
            "force_reason": "",
            "force_bbox_x": np.nan,
            "force_bbox_y": np.nan,
            "force_bbox_w": np.nan,
            "force_bbox_h": np.nan,
            "force_center_x": np.nan,
            "force_center_y": np.nan,
        }

    mass_kg = np.nan if reading.mass_kg is None else float(reading.mass_kg)
    force_n = np.nan if reading.mass_kg is None else float(reading.mass_kg) * 9.80665
    if reading.screen_bbox_xywh is None:
        bbox_x = bbox_y = bbox_w = bbox_h = np.nan
    else:
        bbox_x = float(reading.screen_bbox_xywh[0])
        bbox_y = float(reading.screen_bbox_xywh[1])
        bbox_w = float(reading.screen_bbox_xywh[2])
        bbox_h = float(reading.screen_bbox_xywh[3])
    if reading.screen_center_xy is None:
        center_x = center_y = np.nan
    else:
        center_x = float(reading.screen_center_xy[0])
        center_y = float(reading.screen_center_xy[1])

    return {
        "mass_kg": mass_kg,
        "force_n": force_n,
        "force_confidence": float(reading.confidence),
        "force_screen_visible": bool(reading.screen_visible),
        "force_digits": "" if reading.digits is None else reading.digits,
        "force_reason": reading.reason,
        "force_bbox_x": bbox_x,
        "force_bbox_y": bbox_y,
        "force_bbox_w": bbox_w,
        "force_bbox_h": bbox_h,
        "force_center_x": center_x,
        "force_center_y": center_y,
    }


def _force_plot_keep_mask(mass_kg: pd.Series) -> np.ndarray:
    values = np.asarray(mass_kg, dtype=float)
    n = len(values)
    if n < 7:
        return np.ones(n, dtype=bool)

    window = min(31, max(5, (n // 25) | 1))
    if window % 2 == 0:
        window += 1

    series = pd.Series(values)
    rolling_med = series.rolling(window, center=True, min_periods=1).median()
    residual = (series - rolling_med).abs()
    rolling_mad = residual.rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * rolling_mad.to_numpy()
    threshold = np.maximum(0.15, 4.5 * sigma)
    keep = residual.to_numpy() <= threshold
    if int(np.count_nonzero(keep)) < max(4, int(np.ceil(0.7 * n))):
        return np.ones(n, dtype=bool)
    return keep


def _force_smoothing_window(n: int) -> int:
    if n <= 1:
        return 1
    window = min(11, max(3, (n // 40) | 1))
    if window % 2 == 0:
        window += 1
    return max(window, 1)


def _postprocess_force_measurements(
    frame_df: pd.DataFrame,
    point_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    stats = {
        "readable_frames": 0,
        "outlier_frames": 0,
        "postprocessed_frames": 0,
    }
    if frame_df.empty or "mass_kg" not in frame_df.columns:
        return frame_df, point_df, stats

    out_frame_df = frame_df.copy()
    out_point_df = point_df.copy()
    for raw_col, src_col in (
        ("mass_kg_raw", "mass_kg"),
        ("force_n_raw", "force_n"),
        ("force_confidence_raw", "force_confidence"),
    ):
        if raw_col not in out_frame_df.columns:
            out_frame_df[raw_col] = out_frame_df[src_col]

    out_frame_df["force_outlier_rejected"] = False
    out_frame_df["force_postprocessed"] = False

    valid_force = out_frame_df[out_frame_df["mass_kg"].notna()].copy().sort_values("time_s")
    stats["readable_frames"] = int(len(valid_force))
    if len(valid_force) < 3:
        frame_sync = out_frame_df.set_index("frame")
        for raw_col, src_col in (
            ("mass_kg_raw", "mass_kg"),
            ("force_n_raw", "force_n"),
            ("force_confidence_raw", "force_confidence"),
            ("force_outlier_rejected", "force_outlier_rejected"),
            ("force_postprocessed", "force_postprocessed"),
        ):
            out_point_df[raw_col] = out_point_df["frame"].map(frame_sync[raw_col])
        return out_frame_df, out_point_df, stats

    keep_mask = _force_plot_keep_mask(valid_force["mass_kg"])
    valid_force["force_outlier_rejected"] = ~keep_mask
    filtered_mass = valid_force["mass_kg"].where(keep_mask, np.nan)
    interpolated_mass = filtered_mass.interpolate(limit_direction="both")
    smooth_window = _force_smoothing_window(len(valid_force))
    smoothed_mass = interpolated_mass.rolling(smooth_window, center=True, min_periods=1).median()

    valid_force["mass_kg"] = smoothed_mass.to_numpy(dtype=float)
    valid_force["force_n"] = valid_force["mass_kg"] * 9.80665
    valid_force["force_postprocessed"] = (
        valid_force["force_outlier_rejected"]
        | (valid_force["mass_kg"] - valid_force["mass_kg_raw"]).abs().gt(1e-9)
    )

    stats["outlier_frames"] = int(valid_force["force_outlier_rejected"].sum())
    stats["postprocessed_frames"] = int(valid_force["force_postprocessed"].sum())

    processed_cols = [
        "frame",
        "mass_kg",
        "force_n",
        "mass_kg_raw",
        "force_n_raw",
        "force_confidence_raw",
        "force_outlier_rejected",
        "force_postprocessed",
    ]
    processed_by_frame = valid_force[processed_cols]
    out_frame_df = out_frame_df.drop(
        columns=[
            col
            for col in (
                "mass_kg",
                "force_n",
                "mass_kg_raw",
                "force_n_raw",
                "force_confidence_raw",
                "force_outlier_rejected",
                "force_postprocessed",
            )
            if col in out_frame_df.columns
        ]
    ).merge(processed_by_frame, on="frame", how="left")

    for col, default_value in (
        ("force_outlier_rejected", False),
        ("force_postprocessed", False),
    ):
        out_frame_df[col] = out_frame_df[col].astype("boolean").fillna(default_value).astype(bool)

    frame_sync = out_frame_df.set_index("frame")
    for col in (
        "mass_kg",
        "force_n",
        "mass_kg_raw",
        "force_n_raw",
        "force_confidence_raw",
        "force_outlier_rejected",
        "force_postprocessed",
    ):
        out_point_df[col] = out_point_df["frame"].map(frame_sync[col])

    return out_frame_df, out_point_df, stats


# ── Diagnostic scan ───────────────────────────────────────────────────────────


@dataclass
class _ScanFrameState:
    ok: bool
    reason: str
    matched_after: np.ndarray
    matched_display_after: np.ndarray | None = None
    arbor_center_after: np.ndarray | None = None
    arbor_marker_plane_after: np.ndarray | None = None
    frame_shift_xy: np.ndarray | None = None
    used_fallback: bool = False
    center_draw_after: np.ndarray | None = None
    ref_positions_draw_after: np.ndarray | None = None
    apriltag_count: int = 0
    apriltag_reprojection_rmse_px: float = np.nan


def _copy_scan_state(state: _ScanFrameState) -> _ScanFrameState:
    return _ScanFrameState(
        ok=bool(state.ok),
        reason=str(state.reason),
        matched_after=np.asarray(state.matched_after, dtype=float).copy(),
        matched_display_after=(
            None
            if state.matched_display_after is None
            else np.asarray(state.matched_display_after, dtype=float).copy()
        ),
        arbor_center_after=(
            None
            if state.arbor_center_after is None
            else np.asarray(state.arbor_center_after, dtype=float).copy()
        ),
        arbor_marker_plane_after=(
            None
            if state.arbor_marker_plane_after is None
            else np.asarray(state.arbor_marker_plane_after, dtype=float).copy()
        ),
        frame_shift_xy=(
            None
            if state.frame_shift_xy is None
            else np.asarray(state.frame_shift_xy, dtype=float).copy()
        ),
        used_fallback=bool(state.used_fallback),
        center_draw_after=(
            None
            if state.center_draw_after is None
            else np.asarray(state.center_draw_after, dtype=float).copy()
        ),
        ref_positions_draw_after=(
            None
            if state.ref_positions_draw_after is None
            else np.asarray(state.ref_positions_draw_after, dtype=float).copy()
        ),
        apriltag_count=int(state.apriltag_count),
        apriltag_reprojection_rmse_px=float(state.apriltag_reprojection_rmse_px),
    )

def _scan_frames(
    cap: cv2.VideoCapture,
    cfg_primary: DetectConfig,
    cfg_fallback: DetectConfig | None,
    arbor_cfg: DetectConfig,
    arbor_anchor: np.ndarray,
    n_ref: int,
    ref_dots: np.ndarray,
    max_jump_px: float,
    frame_step: int,
    rotate: int,
    start_frame_idx: int,
    end_frame_idx: int,
    locked_results: dict[int, _ScanFrameState] | None = None,
    rescan_frames: set[int] | None = None,
) -> tuple[list[tuple[int, str]], int, dict[int, _ScanFrameState], int]:
    """
    Quick read-only pass.

    When *locked_results* and *rescan_frames* are supplied, only the frames in
    *rescan_frames* are re-evaluated. Previously-good frames are replayed from
    the cached state transitions instead of being rescanned with the new HSV
    settings.

    Returns (bad_frames, n_scanned, results_by_frame, n_locked).
    bad_frames: list of (frame_idx, reason_string) for frames that would be skipped
    even after trying both configs.
    """
    bad: list[tuple[int, str]] = []
    results_by_frame: dict[int, _ScanFrameState] = {}
    prev_matched = ref_dots.copy()
    prev_arbor_center = arbor_anchor.copy()
    n_scanned = 0
    n_locked = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    idx = start_frame_idx - 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx > end_frame_idx:
            break
        if idx == start_frame_idx:
            continue
        if (idx - start_frame_idx) % frame_step != 0:
            continue

        if (
            locked_results is not None
            and rescan_frames is not None
            and idx in locked_results
            and idx not in rescan_frames
        ):
            cached = _copy_scan_state(locked_results[idx])
            results_by_frame[idx] = cached
            prev_matched = cached.matched_after.copy()
            if cached.arbor_center_after is not None:
                prev_arbor_center = cached.arbor_center_after.copy()
            n_locked += 1
            continue

        n_scanned += 1
        frame = _rotate(frame, rotate)

        frame_stable, arbor_center, _shift = _stabilize_frame(
            frame,
            arbor_anchor,
            arbor_cfg,
            prev_arbor_center,
        )
        if arbor_center is None:
            reason = "arbor missing"
            bad.append((idx, reason))
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=prev_arbor_center.copy(),
            )
            continue
        prev_arbor_center = arbor_center

        cur, fallback_used = _detect_best(frame_stable, cfg_primary, cfg_fallback, n_ref)

        if len(cur) != n_ref:
            reason = f"dot count {len(cur)} ≠ {n_ref}"
            bad.append((idx, reason))
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=prev_arbor_center.copy(),
            )
            continue

        asgn = match_dots(prev_matched, cur, max_dist=max_jump_px)
        if np.any(asgn < 0):
            reason = f"jump too large — {int(np.sum(asgn < 0))} dot(s) moved > {max_jump_px:.0f} px"
            bad.append((idx, reason))
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=prev_arbor_center.copy(),
            )
            continue

        cur_matched = cur[asgn]

        prev_matched = cur_matched
        results_by_frame[idx] = _ScanFrameState(
            ok=True,
            reason="",
            matched_after=prev_matched.copy(),
            matched_display_after=cur_matched.copy(),
            arbor_center_after=prev_arbor_center.copy(),
            frame_shift_xy=np.asarray(_shift, dtype=float).copy(),
            used_fallback=bool(fallback_used),
        )

    return bad, n_scanned, results_by_frame, n_locked


def _scan_frames_apriltag(
    cap: cv2.VideoCapture,
    cfg_primary: DetectConfig,
    cfg_fallback: DetectConfig | None,
    arbor_cfg: DetectConfig,
    arbor_anchor: np.ndarray | None,
    center_offset: np.ndarray | None,
    frame_undistorter: FrameUndistorter,
    intrinsics_seq: IntrinsicsSequence,
    plane_calibration: PlaneCalibration,
    n_ref: int,
    center_plane: np.ndarray,
    ref_dots_plane: np.ndarray,
    max_jump_mm: float,
    frame_step: int,
    rotate: int,
    start_frame_idx: int,
    end_frame_idx: int,
    locked_results: dict[int, _ScanFrameState] | None = None,
    rescan_frames: set[int] | None = None,
) -> tuple[list[tuple[int, str]], int, dict[int, _ScanFrameState], int]:
    """Read-only scan for the AprilTag-plane workflow."""
    bad: list[tuple[int, str]] = []
    results_by_frame: dict[int, _ScanFrameState] = {}
    prev_matched = ref_dots_plane.copy()
    prev_arbor_center = (
        None
        if arbor_anchor is None
        else np.asarray(arbor_anchor, dtype=float).copy()
    )
    n_scanned = 0
    n_locked = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    idx = start_frame_idx - 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx > end_frame_idx:
            break
        if idx == start_frame_idx:
            continue
        if (idx - start_frame_idx) % frame_step != 0:
            continue

        if (
            locked_results is not None
            and rescan_frames is not None
            and idx in locked_results
            and idx not in rescan_frames
        ):
            cached = _copy_scan_state(locked_results[idx])
            results_by_frame[idx] = cached
            prev_matched = cached.matched_after.copy()
            if cached.arbor_center_after is not None:
                prev_arbor_center = cached.arbor_center_after.copy()
            n_locked += 1
            continue

        n_scanned += 1
        frame_geom, frame = _prepare_frame_pair(frame, rotate, frame_undistorter)

        intrinsics = intrinsics_seq.get(idx)
        pose = estimate_frame_pose(
            frame_geom,
            intrinsics,
            idx,
            plane_calibration.plane_points_by_corner,
            apriltag_family=plane_calibration.apriltag_family,
        )
        arbor_marker_plane = None
        red_centroid_raw = None
        if arbor_anchor is not None and center_offset is not None:
            red_centroid_raw = detect_primary_blob_center(
                frame,
                arbor_cfg,
                preferred_center=prev_arbor_center,
            )
            if red_centroid_raw is None:
                reason = "arbor missing"
                bad.append((idx, reason))
                results_by_frame[idx] = _ScanFrameState(
                    ok=False,
                    reason=reason,
                    matched_after=prev_matched.copy(),
                    arbor_center_after=(
                        None
                        if prev_arbor_center is None
                        else prev_arbor_center.copy()
                    ),
                )
                continue
            prev_arbor_center = np.asarray(red_centroid_raw, dtype=float).copy()

        if pose is None:
            reason = "apriltag pose missing"
            bad.append((idx, reason))
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=(
                    None
                    if red_centroid_raw is None
                    else np.asarray(red_centroid_raw, dtype=float).copy()
                ),
                arbor_marker_plane_after=None,
            )
            continue

        if red_centroid_raw is not None:
            arbor_marker_plane = image_points_to_plane(
                _unrotate_points(
                    np.asarray([red_centroid_raw], dtype=float),
                    frame_geom.shape[1],
                    frame_geom.shape[0],
                    rotate,
                ),
                pose,
            )[0]

        cur_px, fallback_used = _detect_best(frame, cfg_primary, cfg_fallback, n_ref)
        if len(cur_px) != n_ref:
            reason = f"dot count {len(cur_px)} ≠ {n_ref}"
            bad.append((idx, reason))
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=(
                    None
                    if red_centroid_raw is None
                    else np.asarray(red_centroid_raw, dtype=float).copy()
                ),
                arbor_marker_plane_after=(
                    None
                    if arbor_marker_plane is None
                    else np.asarray(arbor_marker_plane, dtype=float).copy()
                ),
            )
            continue

        cur_plane = image_points_to_plane(
            _unrotate_points(cur_px, frame_geom.shape[1], frame_geom.shape[0], rotate),
            pose,
        )
        asgn = match_dots(prev_matched, cur_plane, max_dist=max_jump_mm)
        if np.any(asgn < 0):
            reason = (
                f"jump too large — {int(np.sum(asgn < 0))} dot(s) moved > "
                f"{max_jump_mm:.2f} mm"
            )
            bad.append(
                (idx, reason)
            )
            results_by_frame[idx] = _ScanFrameState(
                ok=False,
                reason=reason,
                matched_after=prev_matched.copy(),
                arbor_center_after=(
                    None
                    if red_centroid_raw is None
                    else np.asarray(red_centroid_raw, dtype=float).copy()
                ),
                arbor_marker_plane_after=(
                    None
                    if arbor_marker_plane is None
                    else np.asarray(arbor_marker_plane, dtype=float).copy()
                ),
            )
            continue

        prev_matched = cur_plane[asgn]
        projected_center_draw = np.asarray(
            _rotate_points(
                plane_points_to_image(center_plane, pose),
                frame_geom.shape[1],
                frame_geom.shape[0],
                rotate,
            ),
            dtype=float,
        )
        center_draw = _center_from_arbor_offset(red_centroid_raw, center_offset)
        if center_draw is None:
            center_draw = projected_center_draw
        ref_positions_draw = _rotate_points(
            plane_points_to_image(ref_dots_plane, pose),
            frame_geom.shape[1],
            frame_geom.shape[0],
            rotate,
        )
        results_by_frame[idx] = _ScanFrameState(
            ok=True,
            reason="",
            matched_after=prev_matched.copy(),
            matched_display_after=cur_px[asgn].copy(),
            used_fallback=bool(fallback_used),
            arbor_center_after=(
                None
                if red_centroid_raw is None
                else np.asarray(red_centroid_raw, dtype=float).copy()
            ),
            arbor_marker_plane_after=(
                None
                if arbor_marker_plane is None
                else np.asarray(arbor_marker_plane, dtype=float).copy()
            ),
            center_draw_after=center_draw,
            ref_positions_draw_after=np.asarray(ref_positions_draw, dtype=float).copy(),
            apriltag_count=len(pose.visible_tag_ids),
            apriltag_reprojection_rmse_px=float(pose.reprojection_rmse_px),
        )

    return bad, n_scanned, results_by_frame, n_locked


def _print_scan_report(
    bad: list[tuple[int, str]],
    n_scanned: int,
    n_locked: int,
    fps: float,
    prev_bad_set: set[int] | None = None,
) -> None:
    n_accounted = n_scanned + n_locked
    pct = 100 * len(bad) / max(n_accounted, 1)
    if n_locked > 0:
        print(
            f"\n[scan] {n_scanned} unresolved frame(s) rescanned  |  "
            f"{n_locked} previously-good frame(s) kept"
        )
        print(f"[scan] {len(bad)} would be skipped  ({pct:.1f} % of analyzed frames)")
    else:
        print(f"\n[scan] {n_scanned} frames examined  →  "
              f"{len(bad)} would be skipped  ({pct:.1f} %)")
    if prev_bad_set is not None:
        cur_bad_set = {fi for fi, _ in bad}
        n_fixed = len(prev_bad_set - cur_bad_set)
        n_new   = len(cur_bad_set - prev_bad_set)
        print(f"[scan] ✓ {n_fixed} frame(s) fixed by re-tune"
              + (f"  |  ⚠ {n_new} newly bad" if n_new else ""))
    if not bad:
        print("[scan] No problematic frames — all frames will be used.")
        return
    reasons = Counter(r.split()[0] for _, r in bad)
    for rtype, cnt in reasons.most_common():
        print(f"         {cnt:4d}×  {rtype} …")
    label = "Still-bad frames:" if prev_bad_set is not None else "First bad frames:"
    print(f"[scan] {label}")
    show = bad if prev_bad_set is None else [(fi, r) for fi, r in bad if fi in prev_bad_set][:8] or bad[:8]
    for fi, reason in show[:8]:
        print(f"         frame {fi:5d}  (t = {fi/fps:6.2f} s)  —  {reason}")
    if len(bad) > 8:
        print(f"         … and {len(bad) - 8} more")


_RETUNE_PROMPT = """
┌─────────────────────────────────────────────────────────────────┐
│  What would you like to do?                                     │
│                                                                 │
│  [y]  Re-tune HSV parameters  (opens tuner on first bad frame)  │
│  [n]  Finalize  — run full analysis now, skipping bad frames    │
│                                                                 │
│  TIP: Press [n] at any time to stop re-tuning and proceed.      │
└─────────────────────────────────────────────────────────────────┘
Choice [y/N]: """


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    green_cli_defaults = load_saved_detect_config(
        "green dots",
        DetectConfig(
            h_low=35,
            h_high=85,
            s_low=34,
            v_low=193,
            min_area=300.0,
            max_area=15000.0,
        ),
    )
    arbor_cli_defaults = load_saved_detect_config(
        "arbor marker",
        DetectConfig(
            h_low=18,
            h_high=34,
            s_low=140,
            s_high=255,
            v_low=140,
            v_high=255,
            min_area=5.0,
            max_area=5000.0,
            morph_kernel=3,
        ),
    )
    force_screen_cli_defaults = load_saved_detect_config(
        "force meter screen",
        DetectConfig(
            h_low=90,
            h_high=140,
            s_low=80,
            s_high=255,
            v_low=80,
            v_high=255,
            min_area=500.0,
            max_area=120000.0,
            morph_kernel=7,
        ),
    )

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=None,
                   help="Directory searched recursively for a capture bundle (video + intrinsics/depth) plus Spring .pkl files.")
    p.add_argument("--video", default=None,
                   help="Path to the input video. Optional if --capture-dir is supplied.")
    p.add_argument("--capture-dir", default=None,
                   help="Folder containing capture.mov, intrinsics.json, and depth_calibration.json for AprilTag-plane tracking.")
    p.add_argument("--intrinsics-json", default=None,
                   help="Per-frame intrinsics JSON captured alongside the video. Enables AprilTag-plane tracking.")
    p.add_argument("--depth-calibration-json", default=None,
                   help="Capture-wide depth calibration JSON used to undistort frames in AprilTag mode.")
    p.add_argument("--reference-pkl", default=None,
                   help="Path to a single-Spring .pkl used for displacement_ures_peak_vs_pkl.png. If omitted, auto-discover from --input-dir or the legacy pkl folders.")
    p.add_argument("--start-frame", type=int, default=None,
                   help="Absolute frame index to use as the first analyzed frame")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Absolute frame index to use as the last analyzed frame")
    p.add_argument("--radial-extent-mm", type=float, default=None,
                   help="Physical distance from the innermost to outermost reference radii in the selected first frame")
    p.add_argument("--outer-radius-mm", type=float, default=None,
                   help="Physical distance from the arbor centre to the chosen outer reference point in the selected first frame")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Defaults to <input-dir>/results, <capture-dir>/results, or <video-dir>/results.")
    p.add_argument("--center-x", type=float, default=None)
    p.add_argument("--center-y", type=float, default=None)
    p.add_argument("--refine-center", action="store_true")
    p.add_argument("--apriltag-size-in", type=float, default=1.226,
                   help="Outer black-square width of each AprilTag in inches (used with intrinsics/AprilTag mode)")
    p.add_argument("--apriltag-size-mm", type=float, default=None,
                   help="Outer black-square width of each AprilTag in mm (overrides --apriltag-size-in)")
    p.add_argument("--apriltag-family", default="36h11", choices=sorted(APRILTAG_FAMILY_IDS),
                   help="AprilTag family used in the capture")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip interactive HSV tuner entirely")
    p.add_argument("--no-range-gui", action="store_true",
                   help="Skip the initial frame-range selection window")
    p.add_argument("--no-scale-gui", action="store_true",
                   help="Skip the interactive pre-processing scale calibration window")
    p.add_argument("--track-force", action="store_true",
                   help="Read the blue-backlit luggage scale screen when it appears in frame.")
    p.add_argument("--force-calibration-frame", type=int, default=None,
                   help="Absolute frame index used for force-meter setup. Default: auto-pick the strongest screen frame.")
    p.add_argument("--no-force-gui", action="store_true",
                   help="Skip the force-meter setup GUI and use the auto-detected region / saved defaults.")
    p.add_argument("--force-read-method", default="bruteforce", choices=["bruteforce", "segments"],
                   help="Force-meter digit decoder: the original template matcher or the faster direct seven-segment sampler.")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-jump-px", type=float, default=0.0,
                   help="Inter-frame jump threshold (px). 0 = auto from blob size.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    # HSV defaults
    p.add_argument("--h-low",    type=int,   default=green_cli_defaults.h_low)
    p.add_argument("--h-high",   type=int,   default=green_cli_defaults.h_high)
    p.add_argument("--s-low",    type=int,   default=green_cli_defaults.s_low)
    p.add_argument("--s-high",   type=int,   default=green_cli_defaults.s_high)
    p.add_argument("--v-low",    type=int,   default=green_cli_defaults.v_low)
    p.add_argument("--v-high",   type=int,   default=green_cli_defaults.v_high)
    p.add_argument("--min-area", type=float, default=green_cli_defaults.min_area)
    p.add_argument("--max-area", type=float, default=green_cli_defaults.max_area)
    p.add_argument("--morph-k",  type=int,   default=green_cli_defaults.morph_kernel)
    # Arbor marker HSV defaults (interactive tuning is expected when the marker colour changes)
    p.add_argument("--arbor-h-low",   type=int,   default=arbor_cli_defaults.h_low)
    p.add_argument("--arbor-h-high",  type=int,   default=arbor_cli_defaults.h_high)
    p.add_argument("--arbor-s-low",   type=int,   default=arbor_cli_defaults.s_low)
    p.add_argument("--arbor-s-high",  type=int,   default=arbor_cli_defaults.s_high)
    p.add_argument("--arbor-v-low",   type=int,   default=arbor_cli_defaults.v_low)
    p.add_argument("--arbor-v-high",  type=int,   default=arbor_cli_defaults.v_high)
    p.add_argument("--arbor-min-area", type=float, default=arbor_cli_defaults.min_area)
    p.add_argument("--arbor-max-area", type=float, default=arbor_cli_defaults.max_area)
    p.add_argument("--arbor-morph-k",  type=int,   default=arbor_cli_defaults.morph_kernel)
    args = p.parse_args()

    if args.input_dir is None and args.capture_dir is None and args.video is None:
        sys.exit("[error] Supply --video, --capture-dir, or --input-dir.")
    if args.radial_extent_mm is not None and args.outer_radius_mm is not None:
        sys.exit("[error] Supply only one of --radial-extent-mm or --outer-radius-mm.")
    if args.radial_extent_mm is not None and args.radial_extent_mm <= 0:
        sys.exit("[error] --radial-extent-mm must be positive.")
    if args.outer_radius_mm is not None and args.outer_radius_mm <= 0:
        sys.exit("[error] --outer-radius-mm must be positive.")
    if args.apriltag_size_mm is not None and args.apriltag_size_mm <= 0:
        sys.exit("[error] --apriltag-size-mm must be positive.")
    if args.apriltag_size_in is not None and args.apriltag_size_in <= 0:
        sys.exit("[error] --apriltag-size-in must be positive.")

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir is not None else None
    capture_dir = Path(args.capture_dir).expanduser().resolve() if args.capture_dir is not None else None
    video_path = Path(args.video).expanduser().resolve() if args.video is not None else None
    intrinsics_path = Path(args.intrinsics_json).expanduser().resolve() if args.intrinsics_json is not None else None
    depth_calibration_path = (
        Path(args.depth_calibration_json).expanduser().resolve()
        if args.depth_calibration_json is not None
        else None
    )
    reference_pkl_path = (
        Path(args.reference_pkl).expanduser().resolve()
        if args.reference_pkl is not None
        else None
    )
    if reference_pkl_path is not None and not reference_pkl_path.exists():
        sys.exit(f"[error] Reference Spring .pkl not found: {reference_pkl_path}")

    if input_dir is not None:
        if not input_dir.is_dir():
            sys.exit(f"[error] Input directory does not exist: {input_dir}")

        if capture_dir is None and video_path is None:
            try:
                discovered_capture = resolve_unique_capture_bundle(input_dir)
            except ValueError as exc:
                sys.exit(f"[error] {exc}")
            if discovered_capture is not None:
                capture_dir = discovered_capture.directory
                video_path = discovered_capture.video_path
                if intrinsics_path is None:
                    intrinsics_path = discovered_capture.intrinsics_json
                if depth_calibration_path is None:
                    depth_calibration_path = discovered_capture.depth_calibration_json
                print(f"[bundle] Capture bundle: {capture_dir}")
            else:
                try:
                    discovered_video = resolve_unique_video(input_dir)
                except ValueError as exc:
                    sys.exit(f"[error] {exc}")
                if discovered_video is not None:
                    video_path = discovered_video
                    print(f"[bundle] Video: {video_path}")

        if reference_pkl_path is None:
            try:
                reference_pkl_path = resolve_unique_reference_spring(input_dir)
            except ValueError as exc:
                sys.exit(f"[error] {exc}")
            if reference_pkl_path is not None:
                print(f"[bundle] Reference Spring .pkl: {reference_pkl_path}")

        sweep_candidates = discover_sweep_candidates(input_dir)
        if len(sweep_candidates) == 1:
            print(f"[bundle] Sweep .pkl available for sweep_vs_tracker.py: {sweep_candidates[0].path}")
        elif len(sweep_candidates) > 1:
            print(
                "[warn] Multiple sweep .pkl files found under "
                f"{input_dir}; run_tracker.py does not choose between them automatically."
            )

    if capture_dir is not None:
        if not capture_dir.is_dir():
            sys.exit(f"[error] Capture directory does not exist: {capture_dir}")
        if video_path is None:
            candidate = capture_dir / "capture.mov"
            if candidate.exists():
                video_path = candidate
            else:
                movs = sorted(capture_dir.glob("*.mov")) + sorted(capture_dir.glob("*.MOV"))
                if len(movs) != 1:
                    sys.exit(
                        f"[error] Could not infer a unique .mov in {capture_dir}. "
                        "Supply --video explicitly."
                    )
                video_path = movs[0]
        if intrinsics_path is None:
            intrinsics_path = capture_dir / "intrinsics.json"
        if depth_calibration_path is None:
            depth_calibration_path = capture_dir / "depth_calibration.json"

    if video_path is None:
        sys.exit("[error] No input video could be resolved.")

    use_apriltag_mode = intrinsics_path is not None or depth_calibration_path is not None
    if use_apriltag_mode and (intrinsics_path is None or depth_calibration_path is None):
        sys.exit(
            "[error] AprilTag mode requires both --intrinsics-json and "
            "--depth-calibration-json."
        )
    if intrinsics_path is not None and not intrinsics_path.exists():
        sys.exit(f"[error] Intrinsics JSON not found: {intrinsics_path}")
    if depth_calibration_path is not None and not depth_calibration_path.exists():
        sys.exit(f"[error] Depth calibration JSON not found: {depth_calibration_path}")
    if use_apriltag_mode and (args.outer_radius_mm is not None or args.radial_extent_mm is not None):
        sys.exit(
            "[error] AprilTag/intrinsics/depth mode cannot be combined with "
            "--outer-radius-mm or --radial-extent-mm."
        )
    tag_size_mm = (
        float(args.apriltag_size_mm)
        if args.apriltag_size_mm is not None
        else float(args.apriltag_size_in) * 25.4
    )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif input_dir is not None:
        out_dir = (input_dir / "results").resolve()
    elif capture_dir is not None:
        out_dir = (capture_dir / "results").resolve()
    else:
        out_dir = (video_path.parent / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_default = DetectConfig(
        h_low=args.h_low, h_high=args.h_high,
        s_low=args.s_low, s_high=args.s_high,
        v_low=args.v_low, v_high=args.v_high,
        min_area=args.min_area, max_area=args.max_area,
        morph_kernel=max(args.morph_k, 1) | 1,
    )
    arbor_cfg_default = DetectConfig(
        h_low=args.arbor_h_low, h_high=args.arbor_h_high,
        s_low=args.arbor_s_low, s_high=args.arbor_s_high,
        v_low=args.arbor_v_low, v_high=args.arbor_v_high,
        min_area=args.arbor_min_area, max_area=args.arbor_max_area,
        morph_kernel=max(args.arbor_morph_k, 1) | 1,
    )
    force_cfg_default = DetectConfig(
        h_low=force_screen_cli_defaults.h_low,
        h_high=force_screen_cli_defaults.h_high,
        s_low=force_screen_cli_defaults.s_low,
        s_high=force_screen_cli_defaults.s_high,
        v_low=force_screen_cli_defaults.v_low,
        v_high=force_screen_cli_defaults.v_high,
        min_area=force_screen_cli_defaults.min_area,
        max_area=force_screen_cli_defaults.max_area,
        morph_kernel=max(force_screen_cli_defaults.morph_kernel, 1) | 1,
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if video_width <= 0 or video_height <= 0:
        ret_probe, frame_probe = cap.read()
        if ret_probe and frame_probe is not None:
            video_height, video_width = frame_probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[video] {video_path}  |  {fps:.1f} fps  |  {n_frames} frames")
    if use_apriltag_mode:
        print(f"[intrinsics] {intrinsics_path}")
        print(f"[depth] {depth_calibration_path}")

    if n_frames <= 0:
        sys.exit("[error] Video contains no frames.")

    intrinsics_seq: IntrinsicsSequence | None = None
    depth_calibration: DepthCalibration | None = None
    frame_undistorter: FrameUndistorter | None = None
    if use_apriltag_mode:
        try:
            intrinsics_seq = load_intrinsics_sequence(intrinsics_path)
        except CalibrationError as exc:
            sys.exit(f"[error] Could not load intrinsics: {exc}")
        try:
            depth_calibration = load_depth_calibration(depth_calibration_path)
        except CalibrationError as exc:
            sys.exit(f"[error] Could not load depth calibration: {exc}")
        try:
            intrinsics_seq.validate_image_size(video_width, video_height)
            depth_calibration.validate_image_size(video_width, video_height)
            frame_undistorter = build_frame_undistorter(
                depth_calibration,
                video_width,
                video_height,
            )
        except CalibrationError as exc:
            sys.exit(f"[error] Calibration/video mismatch: {exc}")
        print(f"[intrinsics] Loaded {intrinsics_seq.frame_count} frame record(s)")
        if "heuristic-partial" in intrinsics_seq.parse_modes:
            print(
                "[intrinsics] The JSON did not contain a complete 3x3 camera matrix, "
                "so principal-point gaps were filled heuristically."
            )
        print(
            "[depth] Loaded capture-wide calibration  "
            f"|  distortion center = ({depth_calibration.distortion_center[0]:.2f}, "
            f"{depth_calibration.distortion_center[1]:.2f})  "
            f"|  LUT sizes = {len(depth_calibration.lens_distortion_lookup_table)} / "
            f"{len(depth_calibration.inverse_lens_distortion_lookup_table)}"
        )
        print(
            f"[apriltag] Family {args.apriltag_family}  |  tag size {tag_size_mm:.3f} mm "
            f"({tag_size_mm / 25.4:.3f} in)"
        )

    _last_readable_frame, last_readable_idx = _read_last_frame(
        cap,
        n_frames,
        args.rotate,
        frame_undistorter,
    )
    if last_readable_idx < 0:
        sys.exit("[error] Could not decode any readable frame from the video.")

    start_frame_idx = args.start_frame if args.start_frame is not None else 0
    end_frame_idx = args.end_frame if args.end_frame is not None else last_readable_idx
    if start_frame_idx < 0 or start_frame_idx >= n_frames:
        sys.exit(f"[error] --start-frame must be between 0 and {n_frames - 1}.")
    if end_frame_idx < 0 or end_frame_idx >= n_frames:
        sys.exit(f"[error] --end-frame must be between 0 and {n_frames - 1}.")
    if start_frame_idx > end_frame_idx:
        sys.exit("[error] --start-frame must be <= --end-frame.")

    if not args.no_range_gui:
        start_frame_idx, end_frame_idx = _pick_frame_range(
            cap,
            n_frames,
            args.rotate,
            frame_undistorter,
            fps=fps,
            initial_start_idx=start_frame_idx,
            initial_end_idx=end_frame_idx,
        )

    frame0_geom, frame0 = _read_frame_pair_at(cap, start_frame_idx, args.rotate, frame_undistorter)
    if frame0 is None or frame0_geom is None:
        sys.exit(f"[error] Cannot read selected start frame {start_frame_idx}.")
    frame0_raw_geom: np.ndarray | None = None
    frame0_raw_display: np.ndarray | None = None
    if use_apriltag_mode:
        frame0_raw_geom, frame0_raw_display = _read_frame_pair_at(cap, start_frame_idx, args.rotate, None)
        if frame0_raw_geom is None or frame0_raw_display is None:
            sys.exit(f"[error] Cannot read the raw reference frame {start_frame_idx}.")

    frame_final_geom, frame_final_raw = _read_frame_pair_at(
        cap,
        end_frame_idx,
        args.rotate,
        frame_undistorter,
    )
    if frame_final_raw is None or frame_final_geom is None:
        sys.exit(f"[error] Cannot read selected end frame {end_frame_idx}.")
    frame_final_idx = end_frame_idx

    print(f"[range] Analysis window: frames {start_frame_idx} .. {end_frame_idx} "
          f"({end_frame_idx - start_frame_idx + 1} total)")
    print(f"[range] Reference frame = {start_frame_idx}  |  Final frame = {frame_final_idx}")

    if args.force_calibration_frame is not None:
        if args.force_calibration_frame < start_frame_idx or args.force_calibration_frame > end_frame_idx:
            sys.exit(
                "[error] --force-calibration-frame must lie within the selected "
                "analysis range."
            )

    force_cfg = force_cfg_default
    force_read_method = args.force_read_method
    force_setup_frame_idx: int | None = None
    force_setup_frame: np.ndarray | None = None
    force_search_roi: tuple[int, int, int, int] | None = None
    force_segment_calibration = None
    force_tracking_enabled = False
    force_setup_reading: ForceMeterReading | None = None

    def _read_force(
        frame_for_force: np.ndarray,
        *,
        preferred_center: np.ndarray | None = None,
    ) -> ForceMeterReading:
        return read_force_meter(
            frame_for_force,
            force_cfg,
            search_roi=force_search_roi,
            preferred_center=preferred_center,
            read_method=force_read_method,
            segment_calibration=force_segment_calibration,
        )

    # ── HSV tuning — green final frame, green reference frame, arbor marker ─
    if not args.no_tune:
        print("\n[tuner] ━━━ SESSION 1 of 3 ━━━  Tuning green dots on the FINAL frame "
              "(most deformed — this config is used for the URES plot) …")
        cfg_final = tune_hsv(frame_final_raw, cfg_default, subject="green dots")

        print(f"\n[tuner] ━━━ SESSION 2 of 3 ━━━  Tuning green dots on the REFERENCE frame "
              f"(frame {start_frame_idx}) "
              "(unloaded reference — used as fallback for difficult frames) …")
        cfg_ref = tune_hsv(frame0, cfg_final, subject="green dots")

        print(f"\n[tuner] ━━━ SESSION 3 of 3 ━━━  Tuning the arbor-marker mask on the REFERENCE frame "
              f"(frame {start_frame_idx}) "
              "(used to stabilise every frame before tracking) …")
        arbor_cfg = tune_hsv(frame0, arbor_cfg_default, subject="arbor marker", expected_count=1)
    else:
        cfg_final = cfg_default
        cfg_ref   = cfg_default
        arbor_cfg = arbor_cfg_default

    if args.track_force:
        if args.force_calibration_frame is not None:
            force_setup_frame_idx = int(args.force_calibration_frame)
            force_setup_frame = _read_frame_at(
                cap,
                force_setup_frame_idx,
                args.rotate,
                frame_undistorter,
            )
            if force_setup_frame is None:
                sys.exit(
                    f"[error] Could not decode --force-calibration-frame "
                    f"{force_setup_frame_idx}."
                )
            force_auto_detection = detect_force_screen(force_setup_frame, force_cfg_default)
            force_auto_bbox = None if force_auto_detection is None else force_auto_detection.bbox_xywh
        else:
            force_setup = _find_force_setup_frame(
                cap,
                force_cfg_default,
                rotate=args.rotate,
                frame_undistorter=frame_undistorter,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                frame_step=args.frame_step,
            )
            if force_setup is None:
                force_auto_bbox = None
            else:
                force_setup_frame_idx, force_setup_frame, force_auto_bbox = force_setup

        if force_setup_frame is None or force_setup_frame_idx is None:
            print(
                "[force] No likely force-meter screen found in the selected "
                "analysis range; force tracking disabled."
            )
        else:
            print(
                f"[force] Setup frame = {force_setup_frame_idx}  "
                "(picked from the selected analysis range)"
            )
            print(f"[force] Read method = {force_read_method}")
            if not args.no_tune:
                print(
                    "\n[tuner] ━━━ SESSION 4 of 4 ━━━  Tuning the force-meter "
                    f"screen mask on frame {force_setup_frame_idx} …"
                )
                force_cfg = tune_hsv(
                    force_setup_frame,
                    force_cfg_default,
                    subject="force meter screen",
                )

            force_detection = detect_force_screen(force_setup_frame, force_cfg)
            if force_detection is not None:
                force_auto_bbox = force_detection.bbox_xywh
                suggested_roi = _expand_rect(
                    force_detection.bbox_xywh,
                    force_setup_frame.shape,
                    scale_x=8.0,
                    scale_y=6.0,
                )
            else:
                suggested_roi = (0, 0, force_setup_frame.shape[1], force_setup_frame.shape[0])
                print(
                    "[force] Could not auto-detect the screen on the setup frame "
                    "after tuning; using a broad search area."
                )

            if not args.no_force_gui:
                force_search_roi = _pick_rectangle_gui(
                    force_setup_frame,
                    window_title=(
                        "Force Meter Search Area  |  drag box  |  "
                        "Enter/Space accept  |  R reset"
                    ),
                    prompt_lines=(
                        "[force] Draw a coarse search box around the luggage scale or screen area.",
                        "        Press Enter/Space to accept, or R to reset to the full frame.",
                    ),
                    initial_rect=suggested_roi,
                )
            else:
                force_search_roi = suggested_roi

            if force_read_method == "segments":
                force_segment_calibration = calibrate_force_segments(
                    force_setup_frame,
                    force_cfg,
                    search_roi=force_search_roi,
                )
                if force_segment_calibration is None:
                    print("[force] Segment calibration failed on the setup frame; using broad direct sampling.")
                else:
                    print("[force] Segment calibration locked from the setup frame.")

            force_setup_reading = _read_force(force_setup_frame)
            force_tracking_enabled = True
            print(
                "[force] Search ROI: "
                f"x={force_search_roi[0]}  y={force_search_roi[1]}  "
                f"w={force_search_roi[2]}  h={force_search_roi[3]}"
            )
            if force_setup_reading.mass_kg is not None:
                print(
                    f"[force] Setup-frame read: {force_setup_reading.mass_kg:.3f} kg  "
                    f"(conf = {force_setup_reading.confidence:.2f})"
                )
            elif force_setup_reading.screen_visible:
                print(
                    "[force] Setup-frame screen detected but not decoded  "
                    f"(conf = {force_setup_reading.confidence:.2f}; {force_setup_reading.reason or 'unreadable'})"
                )
            else:
                print("[force] Setup-frame screen was not visible inside the selected ROI.")

            setup_vis = force_setup_frame.copy()
            if force_search_roi is not None:
                rx, ry, rw, rh = force_search_roi
                cv2.rectangle(setup_vis, (rx, ry), (rx + rw, ry + rh), (0, 215, 255), 3, cv2.LINE_AA)
            if force_setup_reading.screen_bbox_xywh is not None:
                bx, by, bw, bh = force_setup_reading.screen_bbox_xywh
                cv2.rectangle(setup_vis, (bx, by), (bx + bw, by + bh), (80, 255, 80), 2, cv2.LINE_AA)
            setup_vis = _put_top_right_info_box(
                setup_vis,
                _force_info_lines(force_setup_reading) or ["Force meter"],
            )
            cv2.imwrite(str(out_dir / "force_meter_setup.png"), setup_vis)
            print(f"[out] {out_dir/'force_meter_setup.png'}")
            if force_setup_reading.screen_crop_bgr is not None:
                cv2.imwrite(
                    str(out_dir / "force_meter_setup_screen.png"),
                    force_setup_reading.screen_crop_bgr,
                )
                print(f"[out] {out_dir/'force_meter_setup_screen.png'}")

    # cfg_final is PRIMARY (used for all frames); cfg_ref is FALLBACK
    cfg_primary  = cfg_final
    cfg_fallback = cfg_ref

    arbor_anchor_raw = detect_primary_blob_center(frame0, arbor_cfg)
    if arbor_anchor_raw is None:
        if use_apriltag_mode:
            sys.exit(
                f"[error] Could not detect the arbor marker in reference frame {start_frame_idx}. "
                "Re-tune the arbor marker mask; AprilTag mode now also uses it for per-frame drift correction."
            )
        print(f"[warn] Could not detect the arbor marker in reference frame {start_frame_idx}.")
    else:
        print(f"[arbor] Reference frame {start_frame_idx} marker centroid: ({arbor_anchor_raw[0]:.1f}, {arbor_anchor_raw[1]:.1f})")

    # ── Spring centre ─────────────────────────────────────────────────────────
    if args.center_x is not None and args.center_y is not None:
        center_px = np.array([args.center_x, args.center_y], dtype=float)
        print(f"[center] CLI centre: {center_px}")
    else:
        center_px = _pick_center(frame0, arbor_anchor_raw)
        if center_px is None:
            sys.exit("[error] No centre selected.")

    center_offset = None if arbor_anchor_raw is None else center_px - arbor_anchor_raw
    if center_offset is not None:
        print(f"[center] Offset from arbor centroid: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
    elif not use_apriltag_mode:
        sys.exit(
            f"[error] Could not detect the arbor marker in reference frame {start_frame_idx}. "
            "Legacy/manual mode still needs it for stabilization."
        )

    # ── Detect reference dots on the selected first frame ────────────────────
    ref_dots_px = detect_green_dots(frame0, cfg_primary)
    if len(ref_dots_px) < 3:
        ref_dots_px = detect_green_dots(frame0, cfg_fallback)
    print(f"[detect] Reference frame {start_frame_idx}: {len(ref_dots_px)} green dots found.")
    if len(ref_dots_px) < 3:
        sys.exit(f"[error] Too few dots in reference frame {start_frame_idx}. Adjust detection parameters.")
    n_dots = len(ref_dots_px)
    ref_dots_px_geom = (
        _unrotate_points(ref_dots_px, frame0_geom.shape[1], frame0_geom.shape[0], args.rotate)
        if use_apriltag_mode
        else ref_dots_px
    )
    center_px_geom = (
        np.asarray(
            _unrotate_points(center_px, frame0_geom.shape[1], frame0_geom.shape[0], args.rotate),
            dtype=float,
        )
        if use_apriltag_mode
        else np.asarray(center_px, dtype=float)
    )

    plane_calibration: PlaneCalibration | None = None
    reference_pose = None
    scale_mode: _ScaleMode | None = None
    reference_px_per_mm = np.nan

    if use_apriltag_mode:
        ref_intrinsics = intrinsics_seq.get(start_frame_idx)
        try:
            plane_calibration = build_plane_calibration(
                frame0_geom,
                ref_intrinsics,
                reference_frame_idx=start_frame_idx,
                tag_size_mm=tag_size_mm,
                apriltag_family=args.apriltag_family,
            )
        except CalibrationError as exc:
            sys.exit(f"[error] Could not build the AprilTag plane calibration: {exc}")

        reference_pose = plane_calibration.reference_pose
        ref_tag_detections = detect_apriltags(frame0_geom, args.apriltag_family)
        ref_tag_edges_px: list[float] = []
        for corners_px in ref_tag_detections.values():
            edge_vecs = np.roll(corners_px, -1, axis=0) - corners_px
            ref_tag_edges_px.extend(np.linalg.norm(edge_vecs, axis=1).tolist())
        reference_px_per_mm = float(np.mean(ref_tag_edges_px) / tag_size_mm) if ref_tag_edges_px else 1.0

        print(
            f"[apriltag] Reference frame {start_frame_idx}: {len(ref_tag_detections)} tag(s)  "
            f"|  anchor id = {plane_calibration.anchor_tag_id}  "
            f"|  pose RMSE = {reference_pose.reprojection_rmse_px:.3f} px"
        )

        center = image_points_to_plane(center_px_geom, reference_pose)
        ref_dots = image_points_to_plane(ref_dots_px_geom, reference_pose)

        if args.refine_center:
            print("[center] Refining numerically on the AprilTag plane …")
            center, a, b, thetas = refine_center(center, ref_dots)
            center_px_geom = plane_points_to_image(center, reference_pose)
            center_px = _rotate_points(
                center_px_geom,
                frame0_geom.shape[1],
                frame0_geom.shape[0],
                args.rotate,
            )
            print(f"[center] Refined plane centre: ({center[0]:.3f}, {center[1]:.3f}) mm")
            if arbor_anchor_raw is not None:
                center_offset = center_px - arbor_anchor_raw
                print(f"[center] Updated offset: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
        else:
            a, b, thetas = fit_spiral(center, ref_dots)

        rmse = spiral_fit_rmse(center, a, b, thetas, ref_dots)
        print(f"[spiral] a={a:.2f} mm  b={b:.4f} mm/rad  "
              f"(≈ {abs(b)*2*np.pi:.1f} mm/turn)")
        print(f"[spiral] geometric fit RMSE = {rmse:.3f} mm")

        arc_mm = arc_lengths_from_min(a, b, thetas)
        arc_px = np.full_like(arc_mm, np.nan)
        unit = "mm"
        calibration_mode = "apriltag-depth"
        px_per_mm = reference_px_per_mm
        print(f"[scale] AprilTag plane calibration active  |  reference scale ≈ {reference_px_per_mm:.3f} px/mm")
    else:
        center = center_px.copy()
        ref_dots = ref_dots_px.copy()

        if args.refine_center:
            print("[center] Refining numerically …")
            center, a, b, thetas = refine_center(center, ref_dots)
            center_px = center.copy()
            if arbor_anchor_raw is not None:
                center_offset = center - arbor_anchor_raw
                print(f"[center] Refined: {center}")
                print(f"[center] Updated offset: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
        else:
            a, b, thetas = fit_spiral(center, ref_dots)

        rmse = spiral_fit_rmse(center, a, b, thetas, ref_dots)
        print(f"[spiral] a={a:.2f} px  b={b:.4f} px/rad  "
              f"(≈ {abs(b)*2*np.pi:.1f} px/turn)")
        print(f"[spiral] geometric fit RMSE = {rmse:.2f} px")

        arc_px = arc_lengths_from_min(a, b, thetas)

        # ── Scale reference geometry + calibration ───────────────────────────
        r_vals = np.linalg.norm(ref_dots - center, axis=1)
        inner_dot = ref_dots[int(np.argmin(r_vals))]
        outer_dot = ref_dots[int(np.argmax(r_vals))]
        inner_radius_px = float(r_vals.min())
        selected_outer_point = np.asarray(outer_dot, dtype=float)

        if args.outer_radius_mm is not None:
            scale_mode = _ScaleMode("outer_radius", "outer radius", float(args.outer_radius_mm))
        elif args.radial_extent_mm is not None:
            scale_mode = _ScaleMode("radial_extent", "radial extent", float(args.radial_extent_mm))

        if scale_mode is not None and not args.no_scale_gui:
            last_probe_frame = frame_final_raw if frame_final_raw is not None else frame0
            last_probe_label = f"Final readable frame ({frame_final_idx})" if frame_final_idx >= 0 else "Last frame fallback"
            last_probe_stabilized = False
            last_probe_dots = np.empty((0, 2), dtype=float)

            if frame_final_raw is not None and arbor_anchor_raw is not None:
                last_probe_frame_stable, last_probe_arbor, _ = _stabilize_frame(
                    frame_final_raw,
                    arbor_anchor_raw,
                    arbor_cfg,
                    arbor_anchor_raw,
                )
                if last_probe_arbor is not None:
                    last_probe_frame = last_probe_frame_stable
                    last_probe_stabilized = True
                last_probe_dots, _ = _detect_best(last_probe_frame, cfg_primary, cfg_fallback, n_dots)
            else:
                last_probe_dots, _ = _detect_best(last_probe_frame, cfg_primary, cfg_fallback, n_dots)

            adjusted_mm, selected_outer_point = _calibrate_scale_gui(
                frame0,
                last_probe_frame,
                last_probe_label,
                center,
                ref_dots,
                last_probe_dots,
                mode=scale_mode,
                inner_radius_px=inner_radius_px,
                inner_dot=inner_dot,
                initial_outer_point=selected_outer_point,
                show_inner_radius=(scale_mode.key == "radial_extent"),
                last_frame_stabilized=last_probe_stabilized,
            )
            scale_mode = _ScaleMode(scale_mode.key, scale_mode.label, adjusted_mm)

        selected_outer_radius_px = float(np.linalg.norm(selected_outer_point - center))
        radial_extent_px = selected_outer_radius_px - inner_radius_px

        if scale_mode is not None:
            scale_reference_px = _scale_reference_px(center, selected_outer_point, inner_radius_px, scale_mode.key)
            px_per_mm = scale_reference_px / scale_mode.reference_mm
            arc_mm = arc_px / px_per_mm
            if scale_mode.key == "outer_radius":
                print(f"[scale] {px_per_mm:.3f} px/mm  "
                      f"(outer radius: {selected_outer_radius_px:.1f} px = {scale_mode.reference_mm} mm)")
            else:
                print(f"[scale] {px_per_mm:.3f} px/mm  "
                      f"(radial extent: {radial_extent_px:.1f} px = {scale_mode.reference_mm} mm)")
        else:
            px_per_mm = 1.0
            arc_mm = arc_px.copy()
            print("[scale] No physical scale supplied; units stay in pixels.")

        unit = "mm" if scale_mode is not None else "px"
        calibration_mode = "manual" if scale_mode is not None else "pixel"

    arc_metric = arc_mm if unit == "mm" else arc_px
    arc_order = np.argsort(arc_metric)
    arc_rank = np.argsort(arc_order)
    arc_col = f"arc_length_{unit}"
    disp_col = f"displacement_{unit}"
    colors = _colormap(n_dots)

    # ── Jump threshold ────────────────────────────────────────────────────────
    if use_apriltag_mode:
        if args.max_jump_px > 0:
            max_jump = args.max_jump_px / max(reference_px_per_mm, 1e-9)
        else:
            max_jump = _estimate_max_jump(frame0, cfg_primary) / max(reference_px_per_mm, 1e-9)
        print(f"[filter] Inter-frame jump threshold: {max_jump:.3f} mm  (~{max_jump * reference_px_per_mm:.1f} px at reference scale)")
    else:
        if args.max_jump_px > 0:
            max_jump = args.max_jump_px
        else:
            max_jump = _estimate_max_jump(frame0, cfg_primary)
        print(f"[filter] Inter-frame jump threshold: {max_jump:.1f} px")

    # ── Diagnostic scan + optional re-tune loop ───────────────────────────────
    prev_bad_set: set[int] | None = None
    scan_cache: dict[int, _ScanFrameState] = {}
    rescan_frames: set[int] | None = None
    while True:
        locked_scan_results = (
            {fi: state for fi, state in scan_cache.items() if state.ok}
            if scan_cache and rescan_frames is not None
            else None
        )
        if use_apriltag_mode:
            print("\n[scan] Scanning all frames (estimating the AprilTag plane pose first) …")
            bad_frames, n_scanned, scan_cache, n_locked = _scan_frames_apriltag(
                cap,
                cfg_primary,
                cfg_fallback,
                arbor_cfg,
                arbor_anchor_raw,
                center_offset,
                frame_undistorter,
                intrinsics_seq,
                plane_calibration,
                n_dots,
                center,
                ref_dots,
                max_jump,
                args.frame_step,
                args.rotate,
                start_frame_idx,
                end_frame_idx,
                locked_results=locked_scan_results,
                rescan_frames=rescan_frames,
            )
        else:
            if rescan_frames is None:
                print("\n[scan] Scanning all frames (stabilising on the arbor marker first) …")
            else:
                print("\n[scan] Rescanning only the previously bad frames (keeping matched frames locked) …")
            bad_frames, n_scanned, scan_cache, n_locked = _scan_frames(
                cap, cfg_primary, cfg_fallback,
                arbor_cfg, arbor_anchor_raw,
                n_dots, ref_dots, max_jump,
                args.frame_step, args.rotate,
                start_frame_idx, end_frame_idx,
                locked_results=locked_scan_results,
                rescan_frames=rescan_frames,
            )
        _print_scan_report(bad_frames, n_scanned, n_locked, fps, prev_bad_set)

        if not bad_frames:
            print("[scan] All frames look good — proceeding to full analysis.")
            break

        if use_apriltag_mode and bad_frames[0][1].startswith("apriltag"):
            print(
                "[scan] Some frames are failing AprilTag pose estimation rather than HSV detection. "
                "Those frames will be skipped."
            )
            break

        first_bad_idx, first_bad_reason = bad_frames[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_bad_idx)
        ret, bad_frame = cap.read()
        if not ret:
            print("[warn] Could not read the first bad frame — finalizing with current parameters.")
            break
        _bad_frame_geom, bad_frame = _prepare_frame_pair(
            bad_frame,
            args.rotate,
            frame_undistorter if use_apriltag_mode else None,
        )

        prompt_frame = bad_frame
        if not use_apriltag_mode and not first_bad_reason.startswith("arbor"):
            prompt_frame_stable, prompt_arbor_center, _ = _stabilize_frame(
                bad_frame,
                arbor_anchor_raw,
                arbor_cfg,
                arbor_anchor_raw,
            )
            if prompt_arbor_center is not None:
                prompt_frame = prompt_frame_stable

        if not _prompt_retune_gui(
            prompt_frame,
            first_bad_idx=first_bad_idx,
            first_bad_reason=first_bad_reason,
            bad_count=len(bad_frames),
            fps=fps,
        ):
            print("[scan] Finalizing with current parameters. "
                  f"{len(bad_frames)} bad frames will be skipped in the output.")
            break

        prev_bad_set = {fi for fi, _ in bad_frames}
        rescan_frames = prev_bad_set.copy()

        if first_bad_reason.startswith("arbor"):
            print(f"\n[tuner] Opening arbor-marker tuner on bad frame #{first_bad_idx} "
                  f"(t = {first_bad_idx/fps:.2f} s)  reason: {first_bad_reason}")
            new_arbor_cfg = tune_hsv(
                bad_frame,
                arbor_cfg,
                subject="arbor marker",
                expected_count=1,
            )
            new_anchor = detect_primary_blob_center(frame0, new_arbor_cfg, preferred_center=arbor_anchor_raw)
            if new_anchor is None:
                print(f"[warn] New arbor settings lost the reference-frame ({start_frame_idx}) arbor marker — keeping previous settings.")
                continue
            arbor_cfg = new_arbor_cfg
            arbor_anchor_raw = new_anchor
            center_display = np.asarray(center_px if use_apriltag_mode else center, dtype=float)
            center_offset = center_display - arbor_anchor_raw
            print(f"[arbor] Updated reference-frame ({start_frame_idx}) marker centroid: ({arbor_anchor_raw[0]:.1f}, {arbor_anchor_raw[1]:.1f})")
            print(f"[center] Updated offset: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
            continue

        print(f"\n[tuner] Opening GREEN-dot tuner on bad frame #{first_bad_idx} "
              f"(t = {first_bad_idx/fps:.2f} s)  reason: {first_bad_reason}")
        if use_apriltag_mode:
            cfg_primary = tune_hsv(
                bad_frame,
                cfg_primary,
                subject="green dots",
                expected_count=n_dots,
            )
        else:
            bad_frame_stable, bad_arbor_center, _ = _stabilize_frame(
                bad_frame,
                arbor_anchor_raw,
                arbor_cfg,
                arbor_anchor_raw,
            )
            if bad_arbor_center is None:
                print("[warn] Could not detect the arbor marker on that frame — re-tune the arbor marker first.")
                continue
            cfg_primary = tune_hsv(
                bad_frame_stable,
                cfg_primary,
                subject="green dots",
                expected_count=n_dots,
            )
        if args.max_jump_px <= 0:
            if use_apriltag_mode:
                max_jump = _estimate_max_jump(frame0, cfg_primary) / max(reference_px_per_mm, 1e-9)
            else:
                max_jump = _estimate_max_jump(frame0, cfg_primary)

    # Always try to process the final frame even if it appeared in the pre-scan
    skip_map = {fi: reason for fi, reason in bad_frames}
    if frame_final_idx >= 0:
        skip_map.pop(frame_final_idx, None)

    ref_phi = np.arctan2(ref_dots[:, 1] - center[1], ref_dots[:, 0] - center[0])
    ref_radius_px_by_dot_id = np.empty(n_dots, dtype=float)
    ref_radius_mm_by_dot_id = np.empty(n_dots, dtype=float)
    for dot_i in range(n_dots):
        dot_id = int(arc_rank[dot_i])
        ref_radius_px_by_dot_id[dot_id] = float(np.linalg.norm(ref_dots_px[dot_i] - center_px))
        if use_apriltag_mode:
            ref_radius_mm_by_dot_id[dot_id] = float(np.linalg.norm(ref_dots[dot_i] - center))
        else:
            ref_radius_mm_by_dot_id[dot_id] = ref_radius_px_by_dot_id[dot_id] / px_per_mm
    inner_ref_idx = int(np.argmin(ref_radius_mm_by_dot_id[arc_rank]))
    outer_ref_idx = int(np.argmax(ref_radius_mm_by_dot_id[arc_rank]))

    reference_frame_label = f"frame {start_frame_idx}  |  range {start_frame_idx}-{end_frame_idx}"

    # ── Save annotated reference frame ────────────────────────────────────────
    if use_apriltag_mode:
        center_ref_draw = _rotate_points(
            plane_points_to_image(center, reference_pose),
            frame0_geom.shape[1],
            frame0_geom.shape[0],
            args.rotate,
        )
        ref_positions_draw = _rotate_points(
            plane_points_to_image(ref_dots, reference_pose),
            frame0_geom.shape[1],
            frame0_geom.shape[0],
            args.rotate,
        )
        frame0_ann = _draw_overlay(
            frame0,
            center_ref_draw,
            ref_dots_px,
            arc_rank,
            colors,
            ref_positions_draw,
            frame_label=reference_frame_label,
        )
        spiral_img = plane_points_to_image(
            _sample_spiral_points(center, a, b, thetas),
            reference_pose,
        )
        spiral_img = _rotate_points(
            spiral_img,
            frame0_geom.shape[1],
            frame0_geom.shape[0],
            args.rotate,
        )
        frame0_ann = _draw_polyline(frame0_ann, spiral_img)
    else:
        frame0_ann = _draw_overlay(
            frame0,
            center,
            ref_dots,
            arc_rank,
            colors,
            ref_dots,
            frame_label=reference_frame_label,
        )
        frame0_ann = _draw_spiral(frame0_ann, center, a, b, thetas)
    cv2.imwrite(str(out_dir / "reference_frame.png"), frame0_ann)
    print(f"\n[out] {out_dir/'reference_frame.png'}  (source frame {start_frame_idx})")

    if use_apriltag_mode:
        tag_detections_display = {
            int(tag_id): _rotate_points(
                corners,
                frame0_geom.shape[1],
                frame0_geom.shape[0],
                args.rotate,
            )
            for tag_id, corners in ref_tag_detections.items()
        }
        raw_tag_detections_display = {
            int(tag_id): _rotate_points(
                depth_calibration.distort_points(corners),
                frame0_raw_geom.shape[1],
                frame0_raw_geom.shape[0],
                args.rotate,
            )
            for tag_id, corners in ref_tag_detections.items()
        }
        tag_width_measurements_mm: dict[int, float] = {}
        for tag_id, corners_px in ref_tag_detections.items():
            corners_plane = image_points_to_plane(corners_px, reference_pose)
            edge_vecs_mm = np.roll(corners_plane, -1, axis=0) - corners_plane
            edge_lengths_mm = np.linalg.norm(edge_vecs_mm, axis=1)
            if len(edge_lengths_mm):
                tag_width_measurements_mm[int(tag_id)] = float(np.mean(edge_lengths_mm))
        tag_width_labels = {
            int(tag_id): f"{width_mm:.2f} mm"
            for tag_id, width_mm in sorted(tag_width_measurements_mm.items())
        }
        if tag_width_measurements_mm:
            tag_width_values_mm = np.asarray(list(tag_width_measurements_mm.values()), dtype=float)
            tag_width_stats = (
                float(np.mean(tag_width_values_mm)),
                float(np.max(np.abs(tag_width_values_mm - tag_size_mm))),
            )
        else:
            tag_width_stats = None
        grid_x, grid_y = np.meshgrid(
            np.arange(frame0_geom.shape[1], dtype=np.float32),
            np.arange(frame0_geom.shape[0], dtype=np.float32),
        )
        shift_map_geom = np.hypot(frame_undistorter.map_x - grid_x, frame_undistorter.map_y - grid_y)
        shift_map_display = _rotate(shift_map_geom, args.rotate)
        distortion_panel = _make_lens_distortion_comparison(
            frame0_raw_display,
            frame0,
            raw_tag_detections_display,
            tag_detections_display,
            _rotate_points(
                depth_calibration.distort_points(ref_dots_px_geom),
                frame0_raw_geom.shape[1],
                frame0_raw_geom.shape[0],
                args.rotate,
            ) if len(ref_dots_px_geom) else np.empty((0, 2), dtype=float),
            ref_dots_px if len(ref_dots_px) else np.empty((0, 2), dtype=float),
            shift_map_display,
            reference_px_per_mm,
            frame_label=reference_frame_label,
        )
        distortion_path = out_dir / "lens_distortion_adjustment.png"
        cv2.imwrite(str(distortion_path), distortion_panel)
        print(f"[out] {distortion_path}")

        inner_radius_mm = float(np.linalg.norm(ref_dots[inner_ref_idx] - center))
        outer_radius_mm = float(np.linalg.norm(ref_dots[outer_ref_idx] - center))
        metric_panel = _make_metric_reference_panel(
            frame0,
            center_ref_draw,
            ref_dots_px[inner_ref_idx],
            ref_dots_px[outer_ref_idx],
            inner_radius_mm,
            outer_radius_mm,
            outer_radius_mm - inner_radius_mm,
            tag_size_mm,
            tag_detections_display,
            tag_width_labels,
            tag_width_stats,
            frame_label=reference_frame_label,
        )
        metric_path = out_dir / "metric_reference_panel.png"
        cv2.imwrite(str(metric_path), metric_panel)
        print(f"[out] {metric_path}")

    # ── Video writer ──────────────────────────────────────────────────────────
    h_px, w_px = frame0.shape[:2]
    out_fps = max(1.0, fps / args.frame_step)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    vout = cv2.VideoWriter(str(out_dir / "tracked.mp4"), fourcc, out_fps, (w_px, h_px))
    vout.write(frame0_ann)

    # ── Main tracking pass ────────────────────────────────────────────────────
    reference_center_raw = (
        (
            _center_from_arbor_offset(arbor_anchor_raw, center_offset)
            if arbor_anchor_raw is not None and center_offset is not None
            else np.asarray(
                _rotate_points(
                    plane_points_to_image(center, reference_pose),
                    frame0_geom.shape[1],
                    frame0_geom.shape[0],
                    args.rotate,
                ),
                dtype=float,
            )
        )
        if use_apriltag_mode
        else np.asarray(center, dtype=float)
    )
    reference_red_centroid = (
        np.asarray(arbor_anchor_raw, dtype=float)
        if arbor_anchor_raw is not None
        else np.full(2, np.nan, dtype=float)
    )
    reference_apriltag_count = len(reference_pose.visible_tag_ids) if use_apriltag_mode else 0
    reference_apriltag_rmse = reference_pose.reprojection_rmse_px if use_apriltag_mode else np.nan
    reference_force_reading = (
        _read_force(
            frame0,
            preferred_center=(
                None
                if force_setup_reading is None
                else force_setup_reading.screen_center_xy
            ),
        )
        if force_tracking_enabled
        else None
    )
    force_prev_center = (
        np.asarray(reference_force_reading.screen_center_xy, dtype=float)
        if reference_force_reading is not None and reference_force_reading.screen_center_xy is not None
        else (
            np.asarray(force_setup_reading.screen_center_xy, dtype=float)
            if force_setup_reading is not None and force_setup_reading.screen_center_xy is not None
            else None
        )
    )
    arbor_endpoint_indices = _curve_endpoint_indices(arc_rank)
    if not np.isfinite(reference_red_centroid).all():
        sys.exit(
            "[error] Could not establish the reference arbor marker position for "
            "rigid-body arbor rotation tracking."
        )
    if use_apriltag_mode:
        reference_arbor_marker_metric = image_points_to_plane(
            _unrotate_points(
                np.asarray([reference_red_centroid], dtype=float),
                frame0_geom.shape[1],
                frame0_geom.shape[0],
                args.rotate,
            ),
            reference_pose,
        )[0]
        arbor_rotation_method = "rigid_plane"
        arbor_reference_points = ref_dots
    else:
        reference_arbor_marker_metric = np.asarray(reference_red_centroid, dtype=float)
        arbor_rotation_method = "rigid_pixel"
        arbor_reference_points = ref_dots
    arbor_rigid_idx = _select_arbor_rigid_endpoint_index(
        arbor_reference_points,
        reference_arbor_marker_metric,
        arbor_endpoint_indices,
    )
    arbor_rigid_dot_id = int(arc_rank[arbor_rigid_idx])
    reference_arbor_phi = _vector_angle_rad(
        reference_arbor_marker_metric,
        arbor_reference_points[arbor_rigid_idx],
    )
    if reference_arbor_phi is None:
        sys.exit(
            "[error] The reference arbor marker and rigid spring endpoint are "
            "degenerate; cannot compute a rigid-body arbor rotation."
        )
    prev_arbor_phi_unwrapped = float(reference_arbor_phi)
    print(
        "[angle] Arbor rotation uses the rigid set "
        f"(arbor marker + dot {arbor_rigid_dot_id}) in {arbor_rotation_method.replace('_', ' ')} coordinates."
    )

    records: list[dict] = []
    frame_records: list[dict] = []
    force_ref_fields = _force_record_fields(reference_force_reading)
    reference_arbor_fields = _arbor_rotation_fields(
        rotation_deg=0.0,
        rotation_signed_deg=0.0,
        rigid_dot_id=arbor_rigid_dot_id,
        method=arbor_rotation_method,
    )
    frame_records.append(dict(
        frame=start_frame_idx,
        frame_in_window=0,
        analysis_start_frame=start_frame_idx,
        analysis_end_frame=end_frame_idx,
        time_s=start_frame_idx / fps,
        tracked=True,
        skipped=False,
        skip_reason="",
        used_fallback=False,
        **reference_arbor_fields,
        **force_ref_fields,
    ))
    for dot_i in range(n_dots):
        records.append(dict(
            frame=start_frame_idx,
            frame_in_window=0,
            analysis_start_frame=start_frame_idx,
            analysis_end_frame=end_frame_idx,
            time_s=start_frame_idx / fps,
            dot_id=int(arc_rank[dot_i]),
            arc_length_px=float(arc_px[dot_i]), arc_length_mm=float(arc_mm[dot_i]),
            x_ref=float(ref_dots_px[dot_i, 0]), y_ref=float(ref_dots_px[dot_i, 1]),
            x_cur=float(ref_dots_px[dot_i, 0]), y_cur=float(ref_dots_px[dot_i, 1]),
            x_ref_plane_mm=float(ref_dots[dot_i, 0]) if use_apriltag_mode else np.nan,
            y_ref_plane_mm=float(ref_dots[dot_i, 1]) if use_apriltag_mode else np.nan,
            x_cur_plane_mm=float(ref_dots[dot_i, 0]) if use_apriltag_mode else np.nan,
            y_cur_plane_mm=float(ref_dots[dot_i, 1]) if use_apriltag_mode else np.nan,
            displacement_px=0.0, displacement_mm=0.0,
            angle_change_deg=0.0, used_fallback=False,
            calibration_mode=calibration_mode,
            apriltag_count=reference_apriltag_count,
            apriltag_reprojection_rmse_px=float(reference_apriltag_rmse),
            arbor_center_x_raw=float(reference_center_raw[0]),
            arbor_center_y_raw=float(reference_center_raw[1]),
            red_centroid_x_raw=float(reference_red_centroid[0]),
            red_centroid_y_raw=float(reference_red_centroid[1]),
            frame_shift_x=0.0, frame_shift_y=0.0,
            **reference_arbor_fields,
            **force_ref_fields,
        ))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    frame_idx    = start_frame_idx - 1
    prev_matched = ref_dots.copy()
    prev_phi_unwrapped = ref_phi.copy()
    prev_arbor_center_raw = None if arbor_anchor_raw is None else arbor_anchor_raw.copy()
    n_skipped    = 0

    print("[track] Processing frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx > end_frame_idx:
            break
        if frame_idx == start_frame_idx:
            continue
        if (frame_idx - start_frame_idx) % args.frame_step != 0:
            continue

        t = frame_idx / fps
        cached_scan_state = scan_cache.get(frame_idx)
        if use_apriltag_mode:
            frame_geom, frame = _prepare_frame_pair(frame, args.rotate, frame_undistorter)
            replay_state = (
                cached_scan_state
                if cached_scan_state is not None and cached_scan_state.ok
                else None
            )
            pose = None
            red_centroid_raw = (
                None
                if replay_state is None or replay_state.arbor_center_after is None
                else np.asarray(replay_state.arbor_center_after, dtype=float)
            )
            center_draw = np.asarray(center_px, dtype=float)
            ref_positions_draw = ref_dots_px
            if replay_state is not None:
                if replay_state.center_draw_after is not None:
                    center_draw = np.asarray(replay_state.center_draw_after, dtype=float)
                if replay_state.ref_positions_draw_after is not None:
                    ref_positions_draw = np.asarray(replay_state.ref_positions_draw_after, dtype=float)
            else:
                pose = estimate_frame_pose(
                    frame_geom,
                    intrinsics_seq.get(frame_idx),
                    frame_idx,
                    plane_calibration.plane_points_by_corner,
                    apriltag_family=plane_calibration.apriltag_family,
                )
                red_centroid_raw = detect_primary_blob_center(
                    frame,
                    arbor_cfg,
                    preferred_center=prev_arbor_center_raw,
                )
            if pose is not None:
                projected_center_draw = np.asarray(
                    _rotate_points(
                        plane_points_to_image(center, pose),
                        frame_geom.shape[1],
                        frame_geom.shape[0],
                        args.rotate,
                    ),
                    dtype=float,
                )
                center_draw = projected_center_draw
                ref_positions_draw = _rotate_points(
                    plane_points_to_image(ref_dots, pose),
                    frame_geom.shape[1],
                    frame_geom.shape[0],
                    args.rotate,
                )
            arbor_center_raw = _center_from_arbor_offset(red_centroid_raw, center_offset)
            if arbor_center_raw is not None:
                center_draw = arbor_center_raw
                prev_arbor_center_raw = np.asarray(red_centroid_raw, dtype=float)
            display_frame = frame
            force_reading = (
                _read_force(
                    display_frame,
                    preferred_center=force_prev_center,
                )
                if force_tracking_enabled
                else None
            )
            if force_reading is not None and force_reading.screen_center_xy is not None:
                force_prev_center = np.asarray(force_reading.screen_center_xy, dtype=float)
            force_fields = _force_record_fields(force_reading)
            force_info_lines = _force_info_lines(force_reading)
            fallback_used = False

            if frame_idx in skip_map:
                ann = _draw_overlay(
                    display_frame,
                    center_draw,
                    np.empty((0, 2)),
                    np.empty(0, int),
                    colors,
                    ref_positions_draw,
                    skipped=True,
                    skip_reason=skip_map[frame_idx],
                    frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                    info_lines=force_info_lines,
                )
                vout.write(ann)
                frame_records.append(dict(
                    frame=frame_idx,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    time_s=t,
                    tracked=False,
                    skipped=True,
                    skip_reason=skip_map[frame_idx],
                    used_fallback=False,
                    **_arbor_rotation_fields(
                        rotation_deg=None,
                        rotation_signed_deg=None,
                        rigid_dot_id=arbor_rigid_dot_id,
                        method=arbor_rotation_method,
                    ),
                    **force_fields,
                ))
                n_skipped += 1
                continue

            def _skip(reason: str) -> None:
                ann = _draw_overlay(
                    display_frame,
                    center_draw,
                    np.empty((0, 2)),
                    np.empty(0, int),
                    colors,
                    ref_positions_draw,
                    skipped=True,
                    skip_reason=reason,
                    frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                    info_lines=force_info_lines,
                )
                vout.write(ann)
                frame_records.append(dict(
                    frame=frame_idx,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    time_s=t,
                    tracked=False,
                    skipped=True,
                    skip_reason=reason,
                    used_fallback=False,
                    **_arbor_rotation_fields(
                        rotation_deg=None,
                        rotation_signed_deg=None,
                        rigid_dot_id=arbor_rigid_dot_id,
                        method=arbor_rotation_method,
                    ),
                    **force_fields,
                ))

            if replay_state is None:
                if pose is None:
                    _skip("apriltag pose missing")
                    n_skipped += 1
                    continue
                if red_centroid_raw is None:
                    _skip("arbor missing")
                    n_skipped += 1
                    continue

                cur_px, fallback_used = _detect_best(display_frame, cfg_primary, cfg_fallback, n_dots)
                if len(cur_px) != n_dots:
                    _skip(f"count={len(cur_px)}")
                    n_skipped += 1
                    continue

                cur_plane = image_points_to_plane(
                    _unrotate_points(cur_px, frame_geom.shape[1], frame_geom.shape[0], args.rotate),
                    pose,
                )
                asgn = match_dots(prev_matched, cur_plane, max_dist=max_jump)
                if np.any(asgn < 0):
                    _skip(f"jump>{max_jump:.2f}mm")
                    n_skipped += 1
                    continue

                cur_matched = cur_plane[asgn]
                draw_positions_px = cur_px[asgn]
                apriltag_count = len(pose.visible_tag_ids)
                apriltag_rmse = float(pose.reprojection_rmse_px)
            else:
                cur_matched = np.asarray(replay_state.matched_after, dtype=float)
                draw_positions_px = (
                    np.asarray(replay_state.matched_display_after, dtype=float)
                    if replay_state.matched_display_after is not None
                    else np.asarray(ref_positions_draw, dtype=float)
                )
                fallback_used = bool(replay_state.used_fallback)
                apriltag_count = int(replay_state.apriltag_count)
                apriltag_rmse = float(replay_state.apriltag_reprojection_rmse_px)

            arbor_marker_metric_current = (
                replay_state.arbor_marker_plane_after
                if replay_state is not None
                else image_points_to_plane(
                    _unrotate_points(
                        np.asarray([red_centroid_raw], dtype=float),
                        frame_geom.shape[1],
                        frame_geom.shape[0],
                        args.rotate,
                    ),
                    pose,
                )[0]
            )
            if arbor_marker_metric_current is None:
                _skip("arbor marker pose missing")
                n_skipped += 1
                continue
            current_arbor_phi_wrapped = _vector_angle_rad(
                arbor_marker_metric_current,
                cur_matched[arbor_rigid_idx],
            )
            if current_arbor_phi_wrapped is None:
                _skip("rigid arbor vector degenerate")
                n_skipped += 1
                continue
            current_arbor_phi_unwrapped = _unwrap_scalar_angle(
                current_arbor_phi_wrapped,
                prev_arbor_phi_unwrapped,
            )
            prev_arbor_phi_unwrapped = current_arbor_phi_unwrapped
            arbor_rotation_signed_deg = float(
                np.degrees(current_arbor_phi_unwrapped - reference_arbor_phi)
            )
            arbor_rotation_deg = float(abs(arbor_rotation_signed_deg))
            arbor_rotation_fields = _arbor_rotation_fields(
                rotation_deg=arbor_rotation_deg,
                rotation_signed_deg=arbor_rotation_signed_deg,
                rigid_dot_id=arbor_rigid_dot_id,
                method=arbor_rotation_method,
            )

            cur_phi_wrapped = np.arctan2(cur_matched[:, 1] - center[1], cur_matched[:, 0] - center[0])
            cur_phi_unwrapped = _unwrap_angles(cur_phi_wrapped, prev_phi_unwrapped)
            prev_matched = cur_matched
            prev_phi_unwrapped = cur_phi_unwrapped

            draw_pos_px, draw_ids = [], []
            for dot_i in range(n_dots):
                cp_plane = cur_matched[dot_i]
                cp_px = draw_positions_px[dot_i]
                disp_px = float(np.linalg.norm(cp_px - ref_dots_px[dot_i]))
                disp_mm = float(np.linalg.norm(cp_plane - ref_dots[dot_i]))
                angle_deg = float(abs(np.degrees(cur_phi_unwrapped[dot_i] - ref_phi[dot_i])))
                records.append(dict(
                    frame=frame_idx, time_s=t,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    dot_id=int(arc_rank[dot_i]),
                    arc_length_px=float(arc_px[dot_i]), arc_length_mm=float(arc_mm[dot_i]),
                    x_ref=float(ref_dots_px[dot_i, 0]), y_ref=float(ref_dots_px[dot_i, 1]),
                    x_cur=float(cp_px[0]), y_cur=float(cp_px[1]),
                    x_ref_plane_mm=float(ref_dots[dot_i, 0]),
                    y_ref_plane_mm=float(ref_dots[dot_i, 1]),
                    x_cur_plane_mm=float(cp_plane[0]),
                    y_cur_plane_mm=float(cp_plane[1]),
                    displacement_px=disp_px, displacement_mm=disp_mm,
                    angle_change_deg=angle_deg, used_fallback=fallback_used,
                    calibration_mode=calibration_mode,
                    apriltag_count=apriltag_count,
                    apriltag_reprojection_rmse_px=apriltag_rmse,
                    arbor_center_x_raw=float(center_draw[0]),
                    arbor_center_y_raw=float(center_draw[1]),
                    red_centroid_x_raw=(
                        np.nan if red_centroid_raw is None else float(red_centroid_raw[0])
                    ),
                    red_centroid_y_raw=(
                        np.nan if red_centroid_raw is None else float(red_centroid_raw[1])
                    ),
                    frame_shift_x=0.0,
                    frame_shift_y=0.0,
                    **arbor_rotation_fields,
                    **force_fields,
                ))
                draw_pos_px.append(cp_px)
                draw_ids.append(int(arc_rank[dot_i]))

            frame_records.append(dict(
                frame=frame_idx,
                frame_in_window=frame_idx - start_frame_idx,
                analysis_start_frame=start_frame_idx,
                analysis_end_frame=end_frame_idx,
                time_s=t,
                tracked=True,
                skipped=False,
                skip_reason="",
                used_fallback=bool(fallback_used),
                **arbor_rotation_fields,
                **force_fields,
            ))
            ann = _draw_overlay(
                display_frame,
                center_draw,
                np.array(draw_pos_px, dtype=float),
                np.array(draw_ids, int),
                colors,
                ref_positions_draw,
                used_fallback=fallback_used,
                frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                info_lines=force_info_lines,
            )
            vout.write(ann)
        else:
            frame = _rotate(frame, args.rotate)
            force_reading = (
                _read_force(
                    frame,
                    preferred_center=force_prev_center,
                )
                if force_tracking_enabled
                else None
            )
            if force_reading is not None and force_reading.screen_center_xy is not None:
                force_prev_center = np.asarray(force_reading.screen_center_xy, dtype=float)
            force_fields = _force_record_fields(force_reading)
            force_info_lines = _force_info_lines(force_reading)
            replay_state = (
                cached_scan_state
                if cached_scan_state is not None and cached_scan_state.ok
                else None
            )
            if replay_state is not None and replay_state.frame_shift_xy is not None:
                shift_xy = np.asarray(replay_state.frame_shift_xy, dtype=float)
                frame_stable = _translate_frame(frame, shift_xy)
                red_centroid_raw = (
                    None
                    if replay_state.arbor_center_after is None
                    else np.asarray(replay_state.arbor_center_after, dtype=float)
                )
                fallback_used = bool(replay_state.used_fallback)
            else:
                frame_stable, red_centroid_raw, shift_xy = _stabilize_frame(
                    frame,
                    arbor_anchor_raw,
                    arbor_cfg,
                    prev_arbor_center_raw,
                )
                fallback_used = False
            display_frame = frame_stable if red_centroid_raw is not None else frame

            if frame_idx in skip_map:
                ann = _draw_overlay(display_frame, center, np.empty((0, 2)), np.empty(0, int),
                                    colors, ref_dots, skipped=True, skip_reason=skip_map[frame_idx],
                                    frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                                    info_lines=force_info_lines)
                vout.write(ann)
                frame_records.append(dict(
                    frame=frame_idx,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    time_s=t,
                    tracked=False,
                    skipped=True,
                    skip_reason=skip_map[frame_idx],
                    used_fallback=False,
                    **_arbor_rotation_fields(
                        rotation_deg=None,
                        rotation_signed_deg=None,
                        rigid_dot_id=arbor_rigid_dot_id,
                        method=arbor_rotation_method,
                    ),
                    **force_fields,
                ))
                n_skipped += 1
                if red_centroid_raw is not None:
                    prev_arbor_center_raw = red_centroid_raw
                continue

            def _skip(reason: str) -> None:
                ann = _draw_overlay(display_frame, center, np.empty((0, 2)), np.empty(0, int),
                                    colors, ref_dots, skipped=True, skip_reason=reason,
                                    frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                                    info_lines=force_info_lines)
                vout.write(ann)
                frame_records.append(dict(
                    frame=frame_idx,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    time_s=t,
                    tracked=False,
                    skipped=True,
                    skip_reason=reason,
                    used_fallback=False,
                    **_arbor_rotation_fields(
                        rotation_deg=None,
                        rotation_signed_deg=None,
                        rigid_dot_id=arbor_rigid_dot_id,
                        method=arbor_rotation_method,
                    ),
                    **force_fields,
                ))

            if red_centroid_raw is None:
                _skip("arbor missing")
                n_skipped += 1
                continue

            prev_arbor_center_raw = red_centroid_raw

            if replay_state is None:
                cur, fallback_used = _detect_best(frame_stable, cfg_primary, cfg_fallback, n_dots)
                if len(cur) != n_dots:
                    _skip(f"count={len(cur)}")
                    n_skipped += 1
                    continue

                asgn = match_dots(prev_matched, cur, max_dist=max_jump)
                if np.any(asgn < 0):
                    _skip(f"jump>{max_jump:.0f}px")
                    n_skipped += 1
                    continue

                cur_matched = cur[asgn]
                draw_positions = cur_matched
            else:
                cur_matched = np.asarray(replay_state.matched_after, dtype=float)
                draw_positions = (
                    np.asarray(replay_state.matched_display_after, dtype=float)
                    if replay_state.matched_display_after is not None
                    else cur_matched
                )
            current_arbor_phi_wrapped = _vector_angle_rad(
                red_centroid_raw,
                cur_matched[arbor_rigid_idx],
            )
            if current_arbor_phi_wrapped is None:
                _skip("rigid arbor vector degenerate")
                n_skipped += 1
                continue
            current_arbor_phi_unwrapped = _unwrap_scalar_angle(
                current_arbor_phi_wrapped,
                prev_arbor_phi_unwrapped,
            )
            prev_arbor_phi_unwrapped = current_arbor_phi_unwrapped
            arbor_rotation_signed_deg = float(
                np.degrees(current_arbor_phi_unwrapped - reference_arbor_phi)
            )
            arbor_rotation_deg = float(abs(arbor_rotation_signed_deg))
            arbor_rotation_fields = _arbor_rotation_fields(
                rotation_deg=arbor_rotation_deg,
                rotation_signed_deg=arbor_rotation_signed_deg,
                rigid_dot_id=arbor_rigid_dot_id,
                method=arbor_rotation_method,
            )
            cur_phi_wrapped = np.arctan2(cur_matched[:, 1] - center[1], cur_matched[:, 0] - center[0])
            cur_phi_unwrapped = _unwrap_angles(cur_phi_wrapped, prev_phi_unwrapped)

            prev_matched = cur_matched
            prev_phi_unwrapped = cur_phi_unwrapped

            arbor_center_raw = red_centroid_raw + center_offset
            draw_pos, draw_ids = [], []
            for dot_i in range(n_dots):
                cp = draw_positions[dot_i]
                disp_px = float(np.linalg.norm(cp - ref_dots[dot_i]))
                disp_mm = disp_px / px_per_mm
                angle_deg = float(abs(np.degrees(cur_phi_unwrapped[dot_i] - ref_phi[dot_i])))
                records.append(dict(
                    frame=frame_idx, time_s=t,
                    frame_in_window=frame_idx - start_frame_idx,
                    analysis_start_frame=start_frame_idx,
                    analysis_end_frame=end_frame_idx,
                    dot_id=int(arc_rank[dot_i]),
                    arc_length_px=float(arc_px[dot_i]), arc_length_mm=float(arc_mm[dot_i]),
                    x_ref=float(ref_dots_px[dot_i, 0]), y_ref=float(ref_dots_px[dot_i, 1]),
                    x_cur=float(cp[0]), y_cur=float(cp[1]),
                    x_ref_plane_mm=np.nan,
                    y_ref_plane_mm=np.nan,
                    x_cur_plane_mm=np.nan,
                    y_cur_plane_mm=np.nan,
                    displacement_px=disp_px, displacement_mm=disp_mm,
                    angle_change_deg=angle_deg, used_fallback=fallback_used,
                    calibration_mode=calibration_mode,
                    apriltag_count=0,
                    apriltag_reprojection_rmse_px=np.nan,
                    arbor_center_x_raw=float(arbor_center_raw[0]),
                    arbor_center_y_raw=float(arbor_center_raw[1]),
                    red_centroid_x_raw=float(red_centroid_raw[0]),
                    red_centroid_y_raw=float(red_centroid_raw[1]),
                    frame_shift_x=float(shift_xy[0]),
                    frame_shift_y=float(shift_xy[1]),
                    **arbor_rotation_fields,
                    **force_fields,
                ))
                draw_pos.append(cp)
                draw_ids.append(int(arc_rank[dot_i]))

            frame_records.append(dict(
                frame=frame_idx,
                frame_in_window=frame_idx - start_frame_idx,
                analysis_start_frame=start_frame_idx,
                analysis_end_frame=end_frame_idx,
                time_s=t,
                tracked=True,
                skipped=False,
                skip_reason="",
                used_fallback=bool(fallback_used),
                **arbor_rotation_fields,
                **force_fields,
            ))
            ann = _draw_overlay(frame_stable, center,
                                np.array(draw_pos), np.array(draw_ids, int),
                                colors, ref_dots, used_fallback=fallback_used,
                                frame_label=f"frame {frame_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                                info_lines=force_info_lines)
            vout.write(ann)

        if (frame_idx - start_frame_idx) % 60 == 0:
            fb_tag = " [fallback]" if fallback_used else ""
            print(f"  frame {frame_idx}/{end_frame_idx}{fb_tag}  skipped_so_far={n_skipped}")

    cap.release()
    vout.release()
    print(f"[out] {out_dir/'tracked.mp4'}  (skipped {n_skipped} frames)")

    # ── Save annotated final frame ────────────────────────────────────────────
    if frame_final_raw is not None:
        final_force_reading = (
            _read_force(
                frame_final_raw,
                preferred_center=force_prev_center,
            )
            if force_tracking_enabled
            else None
        )
        final_force_info_lines = _force_info_lines(final_force_reading)
        if use_apriltag_mode:
            final_pose = estimate_frame_pose(
                frame_final_geom,
                intrinsics_seq.get(frame_final_idx),
                frame_final_idx,
                plane_calibration.plane_points_by_corner,
                apriltag_family=plane_calibration.apriltag_family,
            )
            if final_pose is None:
                print("[warn] Could not estimate the AprilTag plane pose in the final frame — skipping final_frame.png")
            else:
                fd_px, fb = _detect_best(frame_final_raw, cfg_primary, cfg_fallback, n_dots)
                if len(fd_px) == n_dots:
                    fd_plane = image_points_to_plane(
                        _unrotate_points(
                            fd_px,
                            frame_final_geom.shape[1],
                            frame_final_geom.shape[0],
                            args.rotate,
                        ),
                        final_pose,
                    )
                    asgn_f = match_dots(ref_dots, fd_plane, max_dist=max_jump)
                    if not np.any(asgn_f < 0):
                        final_ann = _draw_overlay(
                            frame_final_raw,
                            _rotate_points(
                                plane_points_to_image(center, final_pose),
                                frame_final_geom.shape[1],
                                frame_final_geom.shape[0],
                                args.rotate,
                            ),
                            fd_px[asgn_f],
                            arc_rank,
                            colors,
                            _rotate_points(
                                plane_points_to_image(ref_dots, final_pose),
                                frame_final_geom.shape[1],
                                frame_final_geom.shape[0],
                                args.rotate,
                            ),
                            used_fallback=fb,
                            frame_label=f"frame {frame_final_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                            info_lines=final_force_info_lines,
                        )
                        cv2.imwrite(str(out_dir / "final_frame.png"), final_ann)
                        print(f"[out] {out_dir/'final_frame.png'}  (source frame {frame_final_idx})")
        else:
            frame_final_stable, final_red_centroid_raw, _ = _stabilize_frame(
                frame_final_raw,
                arbor_anchor_raw,
                arbor_cfg,
                prev_arbor_center_raw,
            )
            if final_red_centroid_raw is None:
                print("[warn] Could not detect the arbor marker in the final frame — skipping final_frame.png")
            else:
                fd, fb = _detect_best(frame_final_stable, cfg_primary, cfg_fallback, n_dots)
                if len(fd) == n_dots:
                    asgn_f = match_dots(ref_dots, fd, max_dist=max_jump)
                    if not np.any(asgn_f < 0):
                        final_ann = _draw_overlay(frame_final_stable, center,
                                                  fd[asgn_f], arc_rank,
                                                  colors, ref_dots, used_fallback=fb,
                                                  frame_label=f"frame {frame_final_idx}  |  range {start_frame_idx}-{end_frame_idx}",
                                                  info_lines=final_force_info_lines)
                        cv2.imwrite(str(out_dir / "final_frame.png"), final_ann)
                        print(f"[out] {out_dir/'final_frame.png'}  (source frame {frame_final_idx})")

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    frame_df = pd.DataFrame(frame_records)
    if not frame_df.empty:
        frame_df = (
            frame_df.sort_values(["frame", "tracked", "used_fallback"], ascending=[True, False, True])
            .drop_duplicates(subset=["frame"], keep="first")
            .sort_values("frame")
            .reset_index(drop=True)
        )
        frame_tip_rows: list[dict[str, object]] = []
        for frame_id, grp in df.groupby("frame"):
            tip_row = _moving_endpoint_row(grp, disp_col)
            if tip_row is None:
                continue
            frame_tip_rows.append({
                "frame": int(frame_id),
                f"tip_displacement_{unit}": float(tip_row[disp_col]),
                f"max_displacement_{unit}": float(grp[disp_col].max()),
                "tip_dot_id": int(tip_row["dot_id"]),
            })
        if frame_tip_rows:
            frame_df = frame_df.merge(pd.DataFrame(frame_tip_rows), on="frame", how="left")

    force_postprocess_stats = {
        "readable_frames": 0,
        "outlier_frames": 0,
        "postprocessed_frames": 0,
    }
    if not frame_df.empty and "mass_kg" in frame_df.columns:
        frame_df, df, force_postprocess_stats = _postprocess_force_measurements(frame_df, df)
        if force_postprocess_stats["postprocessed_frames"] > 0:
            print(
                "[force] "
                f"Post-processed {force_postprocess_stats['postprocessed_frames']} readable frame(s) "
                f"at tracker export; rejected {force_postprocess_stats['outlier_frames']} outlier read(s)."
            )

    csv_path = out_dir / "displacements.csv"
    df.to_csv(csv_path, index=False)
    if not frame_df.empty:
        frame_csv_path = out_dir / "frame_measurements.csv"
        frame_df.to_csv(frame_csv_path, index=False)
        n_force_rows = int(frame_df["mass_kg"].notna().sum())
        print(f"[out] {frame_csv_path}  ({len(frame_df)} frames, {n_force_rows} readable force values)")
    n_fallback_frames = df[df["used_fallback"]]["frame"].nunique()
    print(f"[out] {csv_path}  ({len(df)} rows, "
          f"{df['frame'].nunique()} frames, "
          f"{n_fallback_frames} used fallback config)")

    # ── URES-style plot frame selection ──────────────────────────────────────
    # Prefer the actual last video frame if it was tracked; fall back to latest
    last_tracked    = int(df["frame"].max())
    final_frame_idx = frame_final_idx if frame_final_idx in df["frame"].values else last_tracked
    peak_frame_idx  = int(df.groupby("frame")[disp_col].max().idxmax())

    # ── Tip rotation (mirrors ures_plotter_clone formula) ────────────────────
    tip_summary_text = None
    tip_summary_sections: list[str] = []
    peak_tip_id: int | None = None
    peak_tip_radius: float | None = None
    peak_tip_chord_ang: float | None = None
    peak_tip_direct_ang: float | None = None

    final_tip_row = _moving_endpoint_row(df[df["frame"] == final_frame_idx], disp_col)
    if final_tip_row is not None:
        final_tip_id = int(final_tip_row["dot_id"])
        tip_radius = (
            float(ref_radius_mm_by_dot_id[final_tip_id])
            if unit == "mm"
            else float(ref_radius_px_by_dot_id[final_tip_id])
        )
        final_u_tip = float(final_tip_row[disp_col])
        final_tip_ang = float(final_tip_row["angle_change_deg"])
        print(f"[tip ] Final frame {final_frame_idx}  tip endpoint (id={final_tip_id})  "
              f"r={tip_radius:.2f} {unit}  u={final_u_tip:.3f} {unit}  "
              f"→  angle change = {final_tip_ang:.3f} °")
        tip_summary_sections.extend([
            f"Final frame {final_frame_idx} tip (id={final_tip_id})",
            f"r = {tip_radius:.2f} {unit}",
            f"u = {final_u_tip:.3f} {unit}",
            f"angle change = {final_tip_ang:.3f}°",
        ])

    peak_tip_row = _moving_endpoint_row(df[df["frame"] == peak_frame_idx], disp_col)
    if peak_tip_row is not None:
        peak_tip_id = int(peak_tip_row["dot_id"])
        peak_tip_radius = (
            float(ref_radius_mm_by_dot_id[peak_tip_id])
            if unit == "mm"
            else float(ref_radius_px_by_dot_id[peak_tip_id])
        )
        peak_u_tip = float(peak_tip_row[disp_col])
        peak_tip_direct_ang = float(peak_tip_row["angle_change_deg"])
        peak_tip_chord_ang = _ures_tip_angle_deg(peak_u_tip, peak_tip_radius)
        if peak_frame_idx == final_frame_idx:
            print(f"[peak] Peak URES occurs at final frame {peak_frame_idx} "
                  f"with the same tip metrics shown above.")
            tip_summary_sections.extend([
                "",
                f"Peak URES frame = final frame ({peak_frame_idx})",
            ])
        else:
            print(f"[peak] Peak URES frame {peak_frame_idx}  tip endpoint (id={peak_tip_id})  "
                  f"r={peak_tip_radius:.2f} {unit}  u={peak_u_tip:.3f} {unit}  "
                  f"→  angle change = {peak_tip_direct_ang:.3f} °")
            tip_summary_sections.extend([
                "",
                f"Peak URES frame {peak_frame_idx} tip (id={peak_tip_id})",
                f"r = {peak_tip_radius:.2f} {unit}",
                f"u = {peak_u_tip:.3f} {unit}",
                f"angle change = {peak_tip_direct_ang:.3f}°",
            ])

    if tip_summary_sections:
        tip_summary_text = "\n".join(tip_summary_sections)

    # ── URES-style plot ───────────────────────────────────────────────────────
    final = df[df["frame"] == final_frame_idx].sort_values("dot_id")

    fig, ax = plt.subplots(figsize=(10, 5))
    final_x, final_y, final_plot_rows = _camera_ures_curve(final, arc_col, disp_col)
    ax.plot(final_x, final_y, "o-", lw=2, color="steelblue")
    if len(final_y):
        _annotate_endpoint_displacement(
            ax,
            float(final_y[-1]),
            f"tracker final node = {float(final_y[-1]):.3f} {unit}",
            "steelblue",
        )
    for x_i, (_, row) in zip(final_x, final_plot_rows.iterrows()):
        ax.annotate(str(int(row["dot_id"])),
                    (x_i, row[disp_col]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Normalised arc length  s / s_total")
    ax.set_ylabel(f"Displacement  ({unit})")
    ax.set_title(f"URES-style displacement  —  frame {final_frame_idx}  "
                 f"(t = {final_frame_idx/fps:.2f} s)")
    if tip_summary_text is not None:
        ax.text(
            0.02,
            0.98,
            tip_summary_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out_dir / "displacement_ures.png"), dpi=150)
    print(f"[out] {out_dir/'displacement_ures.png'}")
    plt.close(fig)

    # ── URES-style plot, all tracked frames overlaid ─────────────────────────
    frames_sorted = np.array(sorted(df["frame"].unique()), dtype=int)
    cmap_frames = plt.colormaps["viridis"]
    frame_norm = matplotlib.colors.Normalize(
        vmin=int(frames_sorted.min()),
        vmax=int(frames_sorted.max()) if len(frames_sorted) > 1 else int(frames_sorted.min()) + 1,
    )

    fig_all, ax_all = plt.subplots(figsize=(10, 6))
    for frame_i in frames_sorted:
        grp = df[df["frame"] == frame_i].sort_values("dot_id")
        arc_norm_i, disp_i, _ = _camera_ures_curve(grp, arc_col, disp_col)
        color = cmap_frames(frame_norm(frame_i))
        lw = 2.4 if frame_i == frames_sorted[-1] else 1.0
        alpha = 1.0 if frame_i == frames_sorted[-1] else 0.28
        zorder = 3 if frame_i == frames_sorted[-1] else 2
        ax_all.plot(arc_norm_i, disp_i, "-", color=color, lw=lw, alpha=alpha, zorder=zorder)

    first_grp = df[df["frame"] == frames_sorted[0]].sort_values("dot_id")
    first_arc_norm, first_disp, _ = _camera_ures_curve(first_grp, arc_col, disp_col)
    ax_all.plot(first_arc_norm, first_disp, "--", color="black", lw=1.2, alpha=0.8,
                label=f"first tracked frame ({frames_sorted[0]})", zorder=4)

    if len(frames_sorted) > 1:
        last_grp = df[df["frame"] == frames_sorted[-1]].sort_values("dot_id")
        last_arc_norm, last_disp, _ = _camera_ures_curve(last_grp, arc_col, disp_col)
        ax_all.plot(last_arc_norm, last_disp, "-", color=cmap_frames(frame_norm(frames_sorted[-1])),
                    lw=2.6, alpha=1.0, label=f"last tracked frame ({frames_sorted[-1]})", zorder=5)

    peak_grp = df[df["frame"] == peak_frame_idx].sort_values("dot_id")
    peak_arc_norm, peak_disp, _ = _camera_ures_curve(peak_grp, arc_col, disp_col)
    ax_all.plot(
        peak_arc_norm,
        peak_disp,
        "-",
        color="0.55",
        lw=1.0,
        alpha=0.95,
        label=f"peak displacement frame ({peak_frame_idx})",
        zorder=4,
    )

    sm = plt.cm.ScalarMappable(norm=frame_norm, cmap=cmap_frames)
    sm.set_array([])
    cbar = fig_all.colorbar(sm, ax=ax_all, pad=0.02)
    cbar.set_label("Frame index (earlier → later)")

    ax_all.set_xlabel("Normalised arc length  s / s_total")
    ax_all.set_ylabel(f"Displacement  ({unit})")
    ax_all.set_title("URES-style displacement  —  all tracked frames")
    ax_all.grid(True, alpha=0.35)
    ax_all.legend(fontsize=8, loc="best")
    fig_all.tight_layout()
    fig_all.savefig(str(out_dir / "displacement_ures_all_frames.png"), dpi=150)
    print(f"[out] {out_dir/'displacement_ures_all_frames.png'}")
    plt.close(fig_all)

    # ── Peak URES frame vs reference Spring .pkl ─────────────────────────────
    ref_pkl = reference_pkl_path
    if ref_pkl is None and input_dir is None:
        ref_pkl = _find_reference_spring_pkl(Path("individual_pkl_files"), legacy_dir=Path("pkl_to_plot"))
    if ref_pkl is None:
        if input_dir is not None:
            print(f"[warn] No single-Spring .pkl found under {input_dir} — skipping displacement_ures_peak_vs_pkl.png")
        else:
            print("[warn] No Spring .pkl found in individual_pkl_files (or legacy pkl_to_plot) — skipping displacement_ures_peak_vs_pkl.png")
    elif unit != "mm":
        print("[warn] No radial scale supplied; skipping Spring .pkl overlay because tracker displacements are in pixels.")
    else:
        try:
            spring_ref = _load_reference_spring(ref_pkl)
            spring_ures_m, spring_s_norm = spring_ref.construct_ures_representation()
            spring_ures_m = _apply_ures_noise_floor(np.asarray(spring_ures_m, dtype=float))
            spring_ures_mm = spring_ures_m * 1e3
            spring_s_norm = np.asarray(spring_s_norm, dtype=float)
            optimizer_tip_chord_ang = _ures_tip_angle_deg(
                float(abs(spring_ures_m[-1])),
                float(abs(spring_ref.constraints.r[0])),
            )
            optimizer_rot_deg = _optimizer_rotation_deg(spring_ref)

            peak = df[df["frame"] == peak_frame_idx].sort_values("dot_id")
            peak_arc_norm, peak_disp, _ = _camera_ures_curve(peak, arc_col, disp_col)
            peak_max_disp = float(peak[disp_col].max())
            angle_box_lines: list[str] = []
            if peak_tip_id is not None and peak_tip_chord_ang is not None and peak_tip_direct_ang is not None:
                angle_box_lines.extend([
                    f"Tracker peak tip id = {peak_tip_id}",
                    f"Tracker URES/chord angle = {peak_tip_chord_ang:.3f}°",
                    f"Tracker direct angle = {peak_tip_direct_ang:.3f}°",
                ])
            angle_box_lines.append(f"Optimizer URES/chord angle = {optimizer_tip_chord_ang:.3f}°")
            if optimizer_rot_deg is not None:
                angle_box_lines.append(
                    f"Optimizer ROM = {optimizer_rot_deg:.3f}°  (|ROM| = {abs(optimizer_rot_deg):.3f}°)"
                )

            fig_peak, ax_peak = plt.subplots(figsize=(10, 5))
            ax_peak.plot(
                spring_s_norm,
                spring_ures_mm,
                "-",
                lw=2.4,
                color="black",
                label=f"{ref_pkl.stem} (.pkl)",
            )
            ax_peak.plot(
                peak_arc_norm,
                peak_disp,
                "o-",
                lw=2.0,
                color="crimson",
                label=f"tracker peak frame {peak_frame_idx}  (max = {peak_max_disp:.3f} mm)",
            )
            if len(spring_ures_mm):
                _annotate_endpoint_displacement(
                    ax_peak,
                    float(spring_ures_mm[-1]),
                    f"pkl final node = {float(spring_ures_mm[-1]):.3f} mm",
                    "black",
                    linestyle=":",
                    text_offset_pts=8.0,
                )
            if len(peak_disp):
                _annotate_endpoint_displacement(
                    ax_peak,
                    float(peak_disp[-1]),
                    f"tracker final node = {float(peak_disp[-1]):.3f} mm",
                    "crimson",
                    linestyle="--",
                    text_offset_pts=-8.0,
                )
            ax_peak.set_xlabel("Normalised arc length  s / s_total")
            ax_peak.set_ylabel("Displacement  (mm)")
            ax_peak.set_title("URES comparison  —  tracker peak frame vs Spring .pkl")
            if angle_box_lines:
                ax_peak.text(
                    0.02,
                    0.98,
                    "\n".join(angle_box_lines),
                    transform=ax_peak.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.9),
                )
            ax_peak.grid(True, alpha=0.35)
            ax_peak.legend(fontsize=8, loc="best")
            fig_peak.tight_layout()
            fig_peak.savefig(str(out_dir / "displacement_ures_peak_vs_pkl.png"), dpi=150)
            print(f"[out] {out_dir/'displacement_ures_peak_vs_pkl.png'}  (frame {peak_frame_idx} vs {ref_pkl.name})")
            plt.close(fig_peak)
        except Exception as exc:
            print(f"[warn] Could not generate displacement_ures_peak_vs_pkl.png from {ref_pkl.name}: {exc}")

    # ── Displacement vs time ──────────────────────────────────────────────────
    cmap_mpl = plt.colormaps["rainbow"].resampled(max(n_dots, 2))
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for dot_id, grp in df.groupby("dot_id"):
        grp_s   = grp.sort_values("time_s")
        s_label = f"{grp_s[arc_col].iloc[0]:.1f} {unit}"
        color   = cmap_mpl(dot_id / max(n_dots - 1, 1))
        ax2.plot(grp_s["time_s"], grp_s[disp_col],
                 color=color, lw=1.5, label=f"dot {dot_id}  ({s_label})")
    ax2.set_xlabel("Time  (s)")
    ax2.set_ylabel(f"Displacement  ({unit})")
    ax2.set_title("Dot displacement over time")
    ax2.legend(fontsize=7, ncol=max(1, n_dots // 8))
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(str(out_dir / "displacement_time.png"), dpi=150)
    print(f"[out] {out_dir/'displacement_time.png'}")
    plt.close(fig2)

    if not frame_df.empty and frame_df["mass_kg"].notna().any():
        valid_force_plot = frame_df[frame_df["mass_kg"].notna()].sort_values("time_s").copy()

        fig_force_t, ax_force_t = plt.subplots(figsize=(10, 5))
        ax_force_t.plot(valid_force_plot["time_s"], valid_force_plot["mass_kg"], "o-", color="royalblue", lw=1.8, ms=4)
        ax_force_t.set_xlabel("Time  (s)")
        ax_force_t.set_ylabel("Mass reading  (kg)")
        ax_force_t.set_title("Force-meter mass over time")
        ax_force_t.grid(True, alpha=0.35)
        fig_force_t.tight_layout()
        fig_force_t.savefig(str(out_dir / "force_time.png"), dpi=150)
        print(f"[out] {out_dir/'force_time.png'}")
        plt.close(fig_force_t)

        tip_disp_col = f"tip_displacement_{unit}"
        if tip_disp_col in frame_df.columns:
            valid_force_disp = frame_df[
                frame_df["mass_kg"].notna() & frame_df[tip_disp_col].notna()
            ].sort_values("time_s")
            if len(valid_force_disp) >= 2:
                fig_force_disp, ax_force_disp = plt.subplots(figsize=(10, 5))
                ax_force_disp.plot(
                    valid_force_disp[tip_disp_col],
                    valid_force_disp["mass_kg"],
                    "o-",
                    color="teal",
                    lw=1.8,
                    ms=4,
                )
                ax_force_disp.set_xlabel(f"Tip displacement  ({unit})")
                ax_force_disp.set_ylabel("Mass reading  (kg)")
                ax_force_disp.set_title("Force-meter mass vs tip displacement")
                ax_force_disp.grid(True, alpha=0.35)
                fig_force_disp.tight_layout()
                fig_force_disp.savefig(str(out_dir / "force_tip_displacement.png"), dpi=150)
                print(f"[out] {out_dir/'force_tip_displacement.png'}")
                plt.close(fig_force_disp)

    print(f"\n[done] Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
