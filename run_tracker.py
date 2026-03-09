#!/usr/bin/env python3
"""
Spiral torsion spring green-dot tracker.

Detects green dots painted on a physical spring, orders them by arc length
along a fitted Archimedean spiral, tracks them across all video frames, and
produces:
  • results/reference_frame.png   – annotated first frame
  • results/tracked.mp4           – full video with overlaid dot IDs
  • results/displacements.csv     – per-dot displacement vs time
  • results/displacement_ures.png – final-frame URES-style arc-length plot
  • results/displacement_time.png – displacement-vs-time per dot

Usage
-----
# Interactive center click + HSV tuning:
    python run_tracker.py --video recordings/IMG_7036.mov --radial-extent-mm 45.0

# Fully automated (known center):
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --radial-extent-mm 45.0 --center-x 960 --center-y 540 --no-tune

# Every 5th frame only (faster):
    python run_tracker.py --video recordings/IMG_7036.mov \\
        --radial-extent-mm 45.0 --frame-step 5

Notes
-----
• Scale is derived from the pixel distance between the innermost and
  outermost green dots (radial extent), mapped to --radial-extent-mm.
• iPhone .mov files are sometimes encoded rotated; pass --rotate 90 / 180 / 270
  if the preview appears sideways.
• If the spiral has b < 0 (clockwise winding with decreasing radius), the code
  still works – arc lengths are computed as absolute values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # headless-safe; show() still works if a display exists
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detect import DetectConfig, detect_green_dots, tune_hsv
from spiral_fit import (
    arc_lengths_from_min,
    fit_spiral,
    match_dots,
    refine_center,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _pick_center(frame: np.ndarray) -> np.ndarray | None:
    """Show frame; user left-clicks the spring arbor (centre). Press any key to confirm."""
    WIN = "Click the spring centre (arbor), then press any key"
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    print("[center] Click the spring arbor (centre pin) in the window, then press any key.")

    while True:
        vis = frame.copy()
        for pt in clicked:
            cv2.drawMarker(vis, pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(WIN, vis)
        if cv2.waitKey(30) != -1 and clicked:
            break

    cv2.destroyWindow(WIN)
    if not clicked:
        return None
    c = clicked[-1]
    print(f"[center] Selected: ({c[0]}, {c[1]})")
    return np.array(c, dtype=float)


def _colormap(n: int) -> list[tuple[int, int, int]]:
    """Return n BGR colours evenly spaced across the rainbow."""
    cmap = plt.cm.get_cmap("rainbow", n)
    return [(int(cmap(i)[2] * 255), int(cmap(i)[1] * 255), int(cmap(i)[0] * 255))
            for i in range(n)]


def _draw_overlay(
    frame: np.ndarray,
    center: np.ndarray,
    dot_positions: np.ndarray,      # (M, 2) – detected dots this frame
    dot_arc_ids: np.ndarray,        # (M,) int – arc-length rank for each dot
    colors: list[tuple[int, int, int]],
    ref_positions: np.ndarray,      # (N, 2) – reference dot positions (drawn as crosses)
) -> np.ndarray:
    out = frame.copy()
    # Reference ghost positions
    for rp in ref_positions:
        cv2.drawMarker(out, (int(rp[0]), int(rp[1])), (80, 80, 80),
                       cv2.MARKER_CROSS, 12, 1)
    # Spring centre
    cv2.drawMarker(out, (int(center[0]), int(center[1])), (0, 0, 255),
                   cv2.MARKER_STAR, 18, 2)
    # Current dot positions
    for pos, did in zip(dot_positions, dot_arc_ids):
        x, y = int(pos[0]), int(pos[1])
        color = colors[did % len(colors)]
        cv2.circle(out, (x, y), 7, color, -1)
        cv2.putText(out, str(did), (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to .mov / .mp4 video")
    p.add_argument("--radial-extent-mm", type=float, default=None,
                   help="Real distance (mm) between innermost and outermost green dots")
    p.add_argument("--out-dir", default="results", help="Output directory")
    p.add_argument("--center-x", type=float, default=None, help="Spring centre x (px)")
    p.add_argument("--center-y", type=float, default=None, help="Spring centre y (px)")
    p.add_argument("--refine-center", action="store_true",
                   help="Numerically refine the spiral centre after clicking")
    p.add_argument("--no-tune", action="store_true",
                   help="Skip interactive HSV tuner (use default/CLI thresholds)")
    p.add_argument("--frame-step", type=int, default=1,
                   help="Process every Nth frame (1 = every frame)")
    p.add_argument("--max-match-dist", type=float, default=120.0,
                   help="Max pixel distance for dot matching across frames")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate video frames by this many degrees (for iPhone orientation)")
    # HSV knobs (ignored if --no-tune is not set)
    p.add_argument("--h-low",    type=int,   default=35)
    p.add_argument("--h-high",   type=int,   default=85)
    p.add_argument("--s-low",    type=int,   default=80)
    p.add_argument("--v-low",    type=int,   default=40)
    p.add_argument("--min-area", type=float, default=30.0)
    p.add_argument("--max-area", type=float, default=15000.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DetectConfig(
        h_low=args.h_low, h_high=args.h_high,
        s_low=args.s_low, v_low=args.v_low,
        min_area=args.min_area, max_area=args.max_area,
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {args.video}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video}  |  {fps:.1f} fps  |  {n_frames} frames")

    ret, frame0 = cap.read()
    if not ret:
        sys.exit("[error] Could not read first frame.")
    frame0 = _rotate_frame(frame0, args.rotate)

    # ── HSV tuning ────────────────────────────────────────────────────────────
    if not args.no_tune:
        print("[tuner] Opening HSV tuner on the first frame …")
        cfg = tune_hsv(frame0, cfg)

    ref_dots = detect_green_dots(frame0, cfg)
    print(f"[detect] Reference frame: {len(ref_dots)} green dots found.")
    if len(ref_dots) < 3:
        sys.exit("[error] Fewer than 3 dots detected. Adjust HSV params (try --no-tune and tweak --h-low etc.).")

    # ── Spring centre ─────────────────────────────────────────────────────────
    if args.center_x is not None and args.center_y is not None:
        center = np.array([args.center_x, args.center_y], dtype=float)
        print(f"[center] Using CLI-supplied centre: {center}")
    else:
        center = _pick_center(frame0)
        if center is None:
            sys.exit("[error] No centre selected.")

    if args.refine_center:
        print("[center] Refining centre numerically …")
        center, *_ = refine_center(center, ref_dots)
        print(f"[center] Refined centre: {center}")

    # ── Spiral fit on reference frame ─────────────────────────────────────────
    a, b, thetas = fit_spiral(center, ref_dots)
    print(f"[spiral] a = {a:.2f} px,  b = {b:.4f} px/rad  "
          f"(≈ {abs(b) * 2 * np.pi:.1f} px per turn)")

    arc_px = arc_lengths_from_min(a, b, thetas)

    # Arc-length rank: rank 0 = innermost dot
    arc_order = np.argsort(arc_px)           # arc_order[rank] = dot_index
    arc_rank  = np.argsort(arc_order)        # arc_rank[dot_index] = rank

    # ── Scale factor ──────────────────────────────────────────────────────────
    r_vals = np.linalg.norm(ref_dots - center, axis=1)
    radial_extent_px = r_vals.max() - r_vals.min()

    if args.radial_extent_mm is not None:
        px_per_mm = radial_extent_px / args.radial_extent_mm
        arc_mm = arc_px / px_per_mm
        print(f"[scale] {px_per_mm:.3f} px/mm  "
              f"(radial extent: {radial_extent_px:.1f} px = {args.radial_extent_mm} mm)")
    else:
        px_per_mm = 1.0
        arc_mm = arc_px.copy()
        print("[scale] No --radial-extent-mm supplied; units remain in pixels.")

    # ── Reference frame overlay ───────────────────────────────────────────────
    n_dots  = len(ref_dots)
    colors  = _colormap(n_dots)

    ref_draw_pos = ref_dots                    # all ref dots in dot-index order
    ref_draw_ids = arc_rank                    # colour by arc-length rank

    frame0_ann = _draw_overlay(frame0, center,
                               ref_draw_pos, ref_draw_ids, colors, ref_dots)
    cv2.imwrite(str(out_dir / "reference_frame.png"), frame0_ann)
    print(f"[out] {out_dir/'reference_frame.png'}")

    # ── Video writer ──────────────────────────────────────────────────────────
    h_px, w_px = frame0.shape[:2]
    out_fps = max(1.0, fps / args.frame_step)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    vout    = cv2.VideoWriter(str(out_dir / "tracked.mp4"), fourcc, out_fps, (w_px, h_px))

    # ── Frame-by-frame tracking ───────────────────────────────────────────────
    records: list[dict] = []

    # Record frame 0
    for dot_i in range(n_dots):
        records.append(dict(
            frame          = 0,
            time_s         = 0.0,
            dot_id         = int(arc_rank[dot_i]),
            arc_length_px  = float(arc_px[dot_i]),
            arc_length_mm  = float(arc_mm[dot_i]),
            x_ref          = float(ref_dots[dot_i, 0]),
            y_ref          = float(ref_dots[dot_i, 1]),
            x_cur          = float(ref_dots[dot_i, 0]),
            y_cur          = float(ref_dots[dot_i, 1]),
            displacement_px= 0.0,
            displacement_mm= 0.0,
        ))

    vout.write(frame0_ann)

    # Re-open from start for iteration (avoids re-reading frame 0 separately)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = -1

    print("[track] Processing frames …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx == 0:          # already handled above
            continue
        if frame_idx % args.frame_step != 0:
            continue

        frame = _rotate_frame(frame, args.rotate)
        t     = frame_idx / fps

        cur_dots = detect_green_dots(frame, cfg)
        if len(cur_dots) == 0:
            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx}/{n_frames}: no dots detected, skipping")
            continue

        # assignment[dot_i] = index into cur_dots (or -1)
        assignment = match_dots(ref_dots, cur_dots, max_dist=args.max_match_dist)

        draw_pos: list[np.ndarray] = []
        draw_ids: list[int]        = []

        for dot_i, cur_j in enumerate(assignment):
            if cur_j < 0:
                continue
            cp = cur_dots[cur_j]
            disp_px = float(np.linalg.norm(cp - ref_dots[dot_i]))
            records.append(dict(
                frame          = frame_idx,
                time_s         = t,
                dot_id         = int(arc_rank[dot_i]),
                arc_length_px  = float(arc_px[dot_i]),
                arc_length_mm  = float(arc_mm[dot_i]),
                x_ref          = float(ref_dots[dot_i, 0]),
                y_ref          = float(ref_dots[dot_i, 1]),
                x_cur          = float(cp[0]),
                y_cur          = float(cp[1]),
                displacement_px= disp_px,
                displacement_mm= disp_px / px_per_mm,
            ))
            draw_pos.append(cp)
            draw_ids.append(int(arc_rank[dot_i]))

        ann = _draw_overlay(
            frame, center,
            np.array(draw_pos) if draw_pos else np.empty((0, 2)),
            np.array(draw_ids, dtype=int),
            colors, ref_dots,
        )
        vout.write(ann)

        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx}/{n_frames}  matched {sum(a >= 0 for a in assignment)}/{n_dots} dots")

    cap.release()
    vout.release()
    print(f"[out] {out_dir/'tracked.mp4'}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = out_dir / "displacements.csv"
    df.to_csv(csv_path, index=False)
    print(f"[out] {csv_path}  ({len(df)} rows)")

    unit = "mm" if args.radial_extent_mm else "px"
    arc_col  = f"arc_length_{unit}"
    disp_col = f"displacement_{unit}"

    # ── URES-style displacement plot (final frame) ────────────────────────────
    final_frame_idx = df["frame"].max()
    final = df[df["frame"] == final_frame_idx].sort_values("dot_id")

    fig, ax = plt.subplots(figsize=(10, 5))
    arc_norm = final[arc_col] / final[arc_col].max()   # 0→1 like geometry_clone s_norm
    ax.plot(arc_norm, final[disp_col], "o-", lw=2, color="steelblue")
    for _, row in final.iterrows():
        ax.annotate(str(int(row["dot_id"])),
                    (row[arc_col] / final[arc_col].max(), row[disp_col]),
                    fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Normalised arc length  s / s_total")
    ax.set_ylabel(f"Displacement  ({unit})")
    ax.set_title(f"URES-style displacement  –  frame {final_frame_idx}  (t = {final_frame_idx/fps:.2f} s)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    ures_path = out_dir / "displacement_ures.png"
    fig.savefig(ures_path, dpi=150)
    print(f"[out] {ures_path}")
    plt.close(fig)

    # ── Displacement-vs-time per dot ─────────────────────────────────────────
    cmap_mpl = plt.cm.get_cmap("rainbow", n_dots)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for dot_id, grp in df.groupby("dot_id"):
        grp_s = grp.sort_values("time_s")
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
    time_path = out_dir / "displacement_time.png"
    fig2.savefig(time_path, dpi=150)
    print(f"[out] {time_path}")
    plt.close(fig2)

    print("\n[done] All outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
