#!/usr/bin/env python3
"""
Spiral torsion spring green-dot tracker.

Produces:
  • results/reference_frame.png   – annotated frame 0 (unloaded)
  • results/final_frame.png       – annotated last frame (most deformed)
  • results/tracked.mp4           – full video with overlaid dot IDs
  • results/displacements.csv     – per-dot displacement vs time
  • results/displacement_ures.png – final-frame URES-style arc-length plot
  • results/displacement_ures_all_frames.png – overlaid URES-style curves for all tracked frames
  • results/displacement_ures_peak_vs_pkl.png – tracker peak-frame URES vs Spring .pkl URES
  • results/displacement_time.png – displacement-vs-time per dot

Usage
-----
# Interactive (tunes on final frame, then frame 0, then click centre):
    python run_tracker.py --video recordings/IMG_7036.mov --radial-extent-mm 39.5

# Known centre, skip tuner:
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --radial-extent-mm 39.5 --center-x 960 --center-y 540 --no-tune

# Every 3rd frame only:
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --radial-extent-mm 39.5 --frame-step 3
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detect import (
    DetectConfig,
    detect_green_dots,
    detect_primary_blob_center,
    hsv_mask,
    tune_hsv,
)
from spiral_fit import (
    arc_lengths_from_min,
    fit_spiral,
    match_dots,
    refine_center,
    spiral_fit_rmse,
)


# ── Frame / display helpers ───────────────────────────────────────────────────

def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _read_last_frame(
    cap: cv2.VideoCapture, n_frames: int, rotate: int
) -> tuple[np.ndarray | None, int]:
    """Seek backwards from the end until a decodable frame is found.
    Returns (frame, frame_idx). frame_idx is -1 if nothing readable."""
    for offset in range(min(60, n_frames)):
        idx = n_frames - 1 - offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            return _rotate(frame, rotate), idx
    return None, -1


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

    cv2.destroyWindow(WIN)
    if selected is None:
        return None
    c = tuple(np.round(selected).astype(int))
    print(f"[center] Selected: ({c[0]}, {c[1]})")
    return np.array(c, dtype=float)


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


def _ures_tip_angle_deg(u_tip: float, r_tip: float) -> float:
    """Tip angle implied by the URES/chord relation."""
    return float(2.0 * np.degrees(np.arcsin(np.clip(u_tip / max(2.0 * r_tip, 1e-12), -1.0, 1.0))))


def _optimizer_rotation_deg(spring) -> float | None:
    """Applied optimizer rotation from the Spring object, in degrees."""
    rom = getattr(getattr(spring, "specs", None), "rom", None)
    if rom is None:
        return None
    return float(np.degrees(rom))


def _find_reference_spring_pkl(pkl_dir: Path) -> Path | None:
    pkls = sorted(pkl_dir.glob("*.pkl"))
    if not pkls:
        return None
    if len(pkls) > 1:
        print(f"[warn] Multiple Spring .pkl files found in {pkl_dir}; using {pkls[0].name}")
    return pkls[0]


def _load_reference_spring(path: Path):
    from geometry_clone import Constraints, Material, NodeConfig, ParameterSpace, Specs, Spring

    mapping = {
        ("src.geometry", "Spring"): Spring,
        ("src.geometry", "Material"): Material,
        ("src.geometry", "Constraints"): Constraints,
        ("src.geometry", "Specs"): Specs,
        ("src.geometry", "NodeConfig"): NodeConfig,
        ("src.parameter_space", "ParameterSpace"): ParameterSpace,
    }

    class _SpringUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            mapped = mapping.get((module, name))
            if mapped is not None:
                return mapped
            return super().find_class(module, name)

    with path.open("rb") as f:
        return _SpringUnpickler(f).load()


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
    out = img.copy()
    th_min, th_max = thetas.min(), thetas.max()
    th_range = np.linspace(th_min, th_max, n_pts)
    r_range  = a + b * th_range
    xs = (center[0] + r_range * np.cos(th_range)).astype(int)
    ys = (center[1] + r_range * np.sin(th_range)).astype(int)
    h, w = img.shape[:2]
    pts = [(int(xs[i]), int(ys[i])) for i in range(n_pts)
           if 0 <= xs[i] < w and 0 <= ys[i] < h]
    for i in range(len(pts) - 1):
        cv2.line(out, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
    return out


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


# ── Diagnostic scan ───────────────────────────────────────────────────────────

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
) -> tuple[list[tuple[int, str]], int]:
    """
    Quick read-only pass.  Returns (bad_frames, n_scanned).
    bad_frames: list of (frame_idx, reason_string) for frames that would be skipped
    even after trying both configs.
    """
    bad: list[tuple[int, str]] = []
    prev_matched = ref_dots.copy()
    prev_arbor_center = arbor_anchor.copy()
    n_scanned = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx == 0:
            continue
        if idx % frame_step != 0:
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
            bad.append((idx, "arbor missing"))
            continue
        prev_arbor_center = arbor_center

        cur, _ = _detect_best(frame_stable, cfg_primary, cfg_fallback, n_ref)

        if len(cur) != n_ref:
            bad.append((idx, f"dot count {len(cur)} ≠ {n_ref}"))
            continue

        asgn = match_dots(prev_matched, cur, max_dist=max_jump_px)
        if np.any(asgn < 0):
            bad.append((idx, f"jump too large — {int(np.sum(asgn < 0))} dot(s) moved > {max_jump_px:.0f} px"))
            continue

        cur_matched = cur[asgn]

        prev_matched = cur_matched

    return bad, n_scanned


def _print_scan_report(
    bad: list[tuple[int, str]],
    n_scanned: int,
    fps: float,
    prev_bad_set: set[int] | None = None,
) -> None:
    pct = 100 * len(bad) / max(n_scanned, 1)
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--radial-extent-mm", type=float, default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--center-x", type=float, default=None)
    p.add_argument("--center-y", type=float, default=None)
    p.add_argument("--refine-center", action="store_true")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip interactive HSV tuner entirely")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-jump-px", type=float, default=0.0,
                   help="Inter-frame jump threshold (px). 0 = auto from blob size.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    # HSV defaults
    p.add_argument("--h-low",    type=int,   default=35)
    p.add_argument("--h-high",   type=int,   default=85)
    p.add_argument("--s-low",    type=int,   default=34)
    p.add_argument("--v-low",    type=int,   default=193)
    p.add_argument("--min-area", type=float, default=300.0)
    p.add_argument("--max-area", type=float, default=15000.0)
    # Arbor marker HSV defaults (tuned for a blue dot)
    p.add_argument("--arbor-h-low",   type=int,   default=70)
    p.add_argument("--arbor-h-high",  type=int,   default=135)
    p.add_argument("--arbor-s-low",   type=int,   default=80)
    p.add_argument("--arbor-s-high",  type=int,   default=255)
    p.add_argument("--arbor-v-low",   type=int,   default=158)
    p.add_argument("--arbor-v-high",  type=int,   default=255)
    p.add_argument("--arbor-min-area", type=float, default=20.0)
    p.add_argument("--arbor-max-area", type=float, default=20000.0)
    p.add_argument("--arbor-morph-k",  type=int,   default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_default = DetectConfig(
        h_low=args.h_low, h_high=args.h_high,
        s_low=args.s_low, v_low=args.v_low,
        min_area=args.min_area, max_area=args.max_area,
    )
    arbor_cfg_default = DetectConfig(
        h_low=args.arbor_h_low, h_high=args.arbor_h_high,
        s_low=args.arbor_s_low, s_high=args.arbor_s_high,
        v_low=args.arbor_v_low, v_high=args.arbor_v_high,
        min_area=args.arbor_min_area, max_area=args.arbor_max_area,
        morph_kernel=max(args.arbor_morph_k, 1) | 1,
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open: {args.video}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video}  |  {fps:.1f} fps  |  {n_frames} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    if not ret:
        sys.exit("[error] Cannot read frame 0.")
    frame0 = _rotate(frame0, args.rotate)

    frame_final_raw, frame_final_idx = _read_last_frame(cap, n_frames, args.rotate)
    if frame_final_raw is None:
        print("[warn] Could not read final frame; using frame 0 for both tuning sessions.")
        frame_final_raw = frame0
        frame_final_idx = 0

    # ── HSV tuning — green final, green frame 0, arbor marker ───────────────
    if not args.no_tune:
        print("\n[tuner] ━━━ SESSION 1 of 3 ━━━  Tuning green dots on the FINAL frame "
              "(most deformed — this config is used for the URES plot) …")
        cfg_final = tune_hsv(frame_final_raw, cfg_default, subject="green dots")

        print("\n[tuner] ━━━ SESSION 2 of 3 ━━━  Tuning green dots on FRAME 0 "
              "(unloaded reference — used as fallback for difficult frames) …")
        cfg_ref = tune_hsv(frame0, cfg_final, subject="green dots")

        print("\n[tuner] ━━━ SESSION 3 of 3 ━━━  Tuning the arbor-marker mask on FRAME 0 "
              "(used to stabilise every frame before tracking) …")
        arbor_cfg = tune_hsv(frame0, arbor_cfg_default, subject="arbor marker")
    else:
        cfg_final = cfg_default
        cfg_ref   = cfg_default
        arbor_cfg = arbor_cfg_default

    # cfg_final is PRIMARY (used for all frames); cfg_ref is FALLBACK
    cfg_primary  = cfg_final
    cfg_fallback = cfg_ref

    arbor_anchor_raw = detect_primary_blob_center(frame0, arbor_cfg)
    if arbor_anchor_raw is None:
        sys.exit("[error] Could not detect the arbor marker in frame 0. Adjust the arbor HSV parameters.")
    print(f"[arbor] Frame 0 marker centroid: ({arbor_anchor_raw[0]:.1f}, {arbor_anchor_raw[1]:.1f})")

    # ── Spring centre ─────────────────────────────────────────────────────────
    if args.center_x is not None and args.center_y is not None:
        center = np.array([args.center_x, args.center_y], dtype=float)
        print(f"[center] CLI centre: {center}")
    else:
        center = _pick_center(frame0, arbor_anchor_raw)
        if center is None:
            sys.exit("[error] No centre selected.")

    center_offset = center - arbor_anchor_raw
    print(f"[center] Offset from arbor centroid: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")

    # ── Detect reference dots on frame 0 ─────────────────────────────────────
    ref_dots = detect_green_dots(frame0, cfg_primary)
    if len(ref_dots) < 3:
        ref_dots = detect_green_dots(frame0, cfg_fallback)
    print(f"[detect] Frame 0: {len(ref_dots)} green dots found.")
    if len(ref_dots) < 3:
        sys.exit("[error] Too few dots in frame 0. Adjust detection parameters.")
    n_dots = len(ref_dots)

    if args.refine_center:
        print("[center] Refining numerically …")
        center, a, b, thetas = refine_center(center, ref_dots)
        print(f"[center] Refined: {center}")
        center_offset = center - arbor_anchor_raw
        print(f"[center] Updated offset: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
    else:
        # ── Spiral fit on frame 0 ─────────────────────────────────────────────
        a, b, thetas = fit_spiral(center, ref_dots)

    rmse_px = spiral_fit_rmse(center, a, b, thetas, ref_dots)
    print(f"[spiral] a={a:.2f} px  b={b:.4f} px/rad  "
          f"(≈ {abs(b)*2*np.pi:.1f} px/turn)")
    print(f"[spiral] geometric fit RMSE = {rmse_px:.2f} px")

    arc_px    = arc_lengths_from_min(a, b, thetas)
    arc_order = np.argsort(arc_px)
    arc_rank  = np.argsort(arc_order)

    # ── Scale ─────────────────────────────────────────────────────────────────
    r_vals = np.linalg.norm(ref_dots - center, axis=1)
    radial_extent_px = r_vals.max() - r_vals.min()
    if args.radial_extent_mm is not None:
        px_per_mm = radial_extent_px / args.radial_extent_mm
        arc_mm    = arc_px / px_per_mm
        print(f"[scale] {px_per_mm:.3f} px/mm  "
              f"(radial: {radial_extent_px:.1f} px = {args.radial_extent_mm} mm)")
    else:
        px_per_mm = 1.0
        arc_mm    = arc_px.copy()
        print("[scale] No --radial-extent-mm; units stay in pixels.")

    unit     = "mm" if args.radial_extent_mm else "px"
    arc_col  = f"arc_length_{unit}"
    disp_col = f"displacement_{unit}"
    colors   = _colormap(n_dots)

    # ── Jump threshold ────────────────────────────────────────────────────────
    if args.max_jump_px > 0:
        max_jump_px = args.max_jump_px
    else:
        max_jump_px = _estimate_max_jump(frame0, cfg_primary)
    print(f"[filter] Inter-frame jump threshold: {max_jump_px:.1f} px")

    # ── Diagnostic scan + optional re-tune loop ───────────────────────────────
    prev_bad_set: set[int] | None = None
    while True:
        print("\n[scan] Scanning all frames (stabilising on the arbor marker first) …")
        bad_frames, n_scanned = _scan_frames(
            cap, cfg_primary, cfg_fallback,
            arbor_cfg, arbor_anchor_raw,
            n_dots, ref_dots, max_jump_px,
            args.frame_step, args.rotate,
        )
        _print_scan_report(bad_frames, n_scanned, fps, prev_bad_set)

        if not bad_frames:
            print("[scan] All frames look good — proceeding to full analysis.")
            break

        answer = input(_RETUNE_PROMPT).strip().lower()
        if answer != "y":
            print("[scan] Finalizing with current parameters. "
                  f"{len(bad_frames)} bad frames will be skipped in the output.")
            break

        prev_bad_set = {fi for fi, _ in bad_frames}
        first_bad_idx, first_bad_reason = bad_frames[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_bad_idx)
        ret, bad_frame = cap.read()
        if not ret:
            print("[warn] Could not read that frame — skipping re-tune.")
            continue
        bad_frame = _rotate(bad_frame, args.rotate)

        if first_bad_reason.startswith("arbor"):
            print(f"\n[tuner] Opening arbor-marker tuner on bad frame #{first_bad_idx} "
                  f"(t = {first_bad_idx/fps:.2f} s)  reason: {first_bad_reason}")
            new_arbor_cfg = tune_hsv(bad_frame, arbor_cfg, subject="arbor marker")
            new_anchor = detect_primary_blob_center(frame0, new_arbor_cfg, preferred_center=arbor_anchor_raw)
            if new_anchor is None:
                print("[warn] New arbor settings lost the frame 0 arbor marker — keeping previous settings.")
                continue
            arbor_cfg = new_arbor_cfg
            arbor_anchor_raw = new_anchor
            center_offset = center - arbor_anchor_raw
            print(f"[arbor] Updated frame 0 marker centroid: ({arbor_anchor_raw[0]:.1f}, {arbor_anchor_raw[1]:.1f})")
            print(f"[center] Updated offset: dx={center_offset[0]:+.1f} px  dy={center_offset[1]:+.1f} px")
            continue

        print(f"\n[tuner] Opening GREEN-dot tuner on bad frame #{first_bad_idx} "
              f"(t = {first_bad_idx/fps:.2f} s)  reason: {first_bad_reason}")
        bad_frame_stable, bad_arbor_center, _ = _stabilize_frame(
            bad_frame,
            arbor_anchor_raw,
            arbor_cfg,
            arbor_anchor_raw,
        )
        if bad_arbor_center is None:
            print("[warn] Could not detect the arbor marker on that frame — re-tune the arbor marker first.")
            continue
        cfg_primary = tune_hsv(bad_frame_stable, cfg_primary, subject="green dots")
        if args.max_jump_px <= 0:
            max_jump_px = _estimate_max_jump(frame0, cfg_primary)

    # Always try to process the final frame even if it appeared in the pre-scan
    skip_map = {fi: reason for fi, reason in bad_frames}
    if frame_final_idx >= 0:
        skip_map.pop(frame_final_idx, None)

    ref_phi = np.arctan2(ref_dots[:, 1] - center[1], ref_dots[:, 0] - center[0])
    ref_radius_px_by_dot_id = np.empty(n_dots, dtype=float)
    for dot_i in range(n_dots):
        ref_radius_px_by_dot_id[int(arc_rank[dot_i])] = float(np.linalg.norm(ref_dots[dot_i] - center))
    ref_radius_mm_by_dot_id = ref_radius_px_by_dot_id / px_per_mm

    # ── Save annotated reference frame ────────────────────────────────────────
    frame0_ann = _draw_overlay(frame0, center, ref_dots, arc_rank, colors, ref_dots)
    frame0_ann = _draw_spiral(frame0_ann, center, a, b, thetas)
    cv2.imwrite(str(out_dir / "reference_frame.png"), frame0_ann)
    print(f"\n[out] {out_dir/'reference_frame.png'}")

    # ── Video writer ──────────────────────────────────────────────────────────
    h_px, w_px = frame0.shape[:2]
    out_fps = max(1.0, fps / args.frame_step)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # type: ignore[attr-defined]
    vout = cv2.VideoWriter(str(out_dir / "tracked.mp4"), fourcc, out_fps, (w_px, h_px))
    vout.write(frame0_ann)

    # ── Main tracking pass ────────────────────────────────────────────────────
    records: list[dict] = []
    for dot_i in range(n_dots):
        records.append(dict(
            frame=0, time_s=0.0,
            dot_id=int(arc_rank[dot_i]),
            arc_length_px=float(arc_px[dot_i]), arc_length_mm=float(arc_mm[dot_i]),
            x_ref=float(ref_dots[dot_i, 0]),    y_ref=float(ref_dots[dot_i, 1]),
            x_cur=float(ref_dots[dot_i, 0]),    y_cur=float(ref_dots[dot_i, 1]),
            displacement_px=0.0, displacement_mm=0.0,
            angle_change_deg=0.0, used_fallback=False,
            arbor_center_x_raw=float(center[0]),
            arbor_center_y_raw=float(center[1]),
            red_centroid_x_raw=float(arbor_anchor_raw[0]),
            red_centroid_y_raw=float(arbor_anchor_raw[1]),
            frame_shift_x=0.0, frame_shift_y=0.0,
        ))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx    = -1
    prev_matched = ref_dots.copy()
    prev_phi_unwrapped = ref_phi.copy()
    prev_arbor_center_raw = arbor_anchor_raw.copy()
    n_skipped    = 0

    print("[track] Processing frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx == 0:
            continue
        if frame_idx % args.frame_step != 0:
            continue

        frame = _rotate(frame, args.rotate)
        frame_stable, red_centroid_raw, shift_xy = _stabilize_frame(
            frame,
            arbor_anchor_raw,
            arbor_cfg,
            prev_arbor_center_raw,
        )
        display_frame = frame_stable if red_centroid_raw is not None else frame
        t = frame_idx / fps

        if frame_idx in skip_map:
            ann = _draw_overlay(display_frame, center, np.empty((0, 2)), np.empty(0, int),
                                colors, ref_dots, skipped=True, skip_reason=skip_map[frame_idx])
            vout.write(ann)
            n_skipped += 1
            if red_centroid_raw is not None:
                prev_arbor_center_raw = red_centroid_raw
            continue

        def _skip(reason: str) -> None:
            ann = _draw_overlay(display_frame, center, np.empty((0, 2)), np.empty(0, int),
                                colors, ref_dots, skipped=True, skip_reason=reason)
            vout.write(ann)

        if red_centroid_raw is None:
            _skip("arbor missing")
            n_skipped += 1
            continue

        prev_arbor_center_raw = red_centroid_raw

        # ── Detect with fallback ──────────────────────────────────────────
        cur, fallback_used = _detect_best(frame_stable, cfg_primary, cfg_fallback, n_dots)

        if len(cur) != n_dots:
            _skip(f"count={len(cur)}")
            n_skipped += 1
            continue

        asgn = match_dots(prev_matched, cur, max_dist=max_jump_px)
        if np.any(asgn < 0):
            _skip(f"jump>{max_jump_px:.0f}px")
            n_skipped += 1
            continue

        cur_matched = cur[asgn]
        cur_phi_wrapped = np.arctan2(cur_matched[:, 1] - center[1], cur_matched[:, 0] - center[0])
        cur_phi_unwrapped = _unwrap_angles(cur_phi_wrapped, prev_phi_unwrapped)

        prev_matched = cur_matched
        prev_phi_unwrapped = cur_phi_unwrapped

        arbor_center_raw = red_centroid_raw + center_offset
        draw_pos, draw_ids = [], []
        for dot_i in range(n_dots):
            cp      = cur[asgn[dot_i]]
            disp_px = float(np.linalg.norm(cp - ref_dots[dot_i]))
            disp_mm = disp_px / px_per_mm
            angle_deg = float(abs(np.degrees(cur_phi_unwrapped[dot_i] - ref_phi[dot_i])))
            records.append(dict(
                frame=frame_idx, time_s=t,
                dot_id=int(arc_rank[dot_i]),
                arc_length_px=float(arc_px[dot_i]), arc_length_mm=float(arc_mm[dot_i]),
                x_ref=float(ref_dots[dot_i, 0]),    y_ref=float(ref_dots[dot_i, 1]),
                x_cur=float(cp[0]),                 y_cur=float(cp[1]),
                displacement_px=disp_px, displacement_mm=disp_mm,
                angle_change_deg=angle_deg, used_fallback=fallback_used,
                arbor_center_x_raw=float(arbor_center_raw[0]),
                arbor_center_y_raw=float(arbor_center_raw[1]),
                red_centroid_x_raw=float(red_centroid_raw[0]),
                red_centroid_y_raw=float(red_centroid_raw[1]),
                frame_shift_x=float(shift_xy[0]),
                frame_shift_y=float(shift_xy[1]),
            ))
            draw_pos.append(cp)
            draw_ids.append(int(arc_rank[dot_i]))

        ann = _draw_overlay(frame_stable, center,
                            np.array(draw_pos), np.array(draw_ids, int),
                            colors, ref_dots, used_fallback=fallback_used)
        vout.write(ann)

        if frame_idx % 60 == 0:
            fb_tag = " [fallback]" if fallback_used else ""
            print(f"  frame {frame_idx}/{n_frames}{fb_tag}  skipped_so_far={n_skipped}")

    cap.release()
    vout.release()
    print(f"[out] {out_dir/'tracked.mp4'}  (skipped {n_skipped} frames)")

    # ── Save annotated final frame ────────────────────────────────────────────
    if frame_final_raw is not None:
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
                asgn_f = match_dots(ref_dots, fd, max_dist=max_jump_px)
                if not np.any(asgn_f < 0):
                    final_ann = _draw_overlay(frame_final_stable, center,
                                              fd[asgn_f], arc_rank,
                                              colors, ref_dots, used_fallback=fb)
                    cv2.imwrite(str(out_dir / "final_frame.png"), final_ann)
                    print(f"[out] {out_dir/'final_frame.png'}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = out_dir / "displacements.csv"
    df.to_csv(csv_path, index=False)
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
    ref_pkl = _find_reference_spring_pkl(Path("pkl_to_plot"))
    if ref_pkl is None:
        print("[warn] No Spring .pkl found in pkl_to_plot — skipping displacement_ures_peak_vs_pkl.png")
    elif unit != "mm":
        print("[warn] No radial scale supplied; skipping Spring .pkl overlay because tracker displacements are in pixels.")
    else:
        try:
            spring_ref = _load_reference_spring(ref_pkl)
            spring_ures_m, spring_s_norm = spring_ref.construct_ures_representation()
            spring_ures_mm = np.asarray(spring_ures_m, dtype=float) * 1e3
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

    print(f"\n[done] Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
