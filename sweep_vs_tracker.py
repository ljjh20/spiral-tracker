#!/usr/bin/env python3
"""Compare a swept Spring pickle against tracker outputs.

Outputs include:
  - arbor-rotation- and moment-matched URES comparison figures
  - a reference-frame camera-tracking apparatus snapshot with distortion and AprilTag overlays
  - paired arbor-center snapshot figures for selected arbor-rotation/moment matches
  - arbor-rotation- and moment-matched GIF/MP4 sweep walkthroughs
  - summary overlay plots for moment-vs-arbor-rotation and final-node-URES-vs-moment
  - a viridis sweep plot of per-node stress for every swept Spring state
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "spiral_tracker_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from bundle_discovery import load_pickle, resolve_unique_sweep_pkl, resolve_unique_video
from plot_style import configure_matplotlib_defaults
from planar_calibration import (
    CalibrationError,
    build_frame_undistorter,
    build_plane_calibration,
    detect_apriltags,
    image_points_to_plane,
    load_depth_calibration,
    load_intrinsics_sequence,
    plane_points_to_image,
)
from reference_clones.geometry_clone import Spring
from reference_clones.ures_plotter_clone import stress_solidworks_2025, ures_solidworks_2025
from spiral_fit import fit_spiral

configure_matplotlib_defaults()

URES_NOISE_FLOOR_M = 1e-8
STRESS_NOISE_FLOOR_PA = 1e-8
PAPER_PANEL_DPI = 300
PAPER_PANEL_SIZE_PX = 630
PAPER_PANEL_SIZE_IN = PAPER_PANEL_SIZE_PX / PAPER_PANEL_DPI
PAPER_CURVE_AXES_SIZE_MM = 53.0
PAPER_CURVE_LEFT_MARGIN_MM = 14.0
PAPER_CURVE_RIGHT_MARGIN_MM = 3.0
PAPER_CURVE_TOP_MARGIN_MM = 11.0
PAPER_CURVE_BOTTOM_MARGIN_MM = 36.0
PAPER_CURVE_LEGEND_BOTTOM_MM = 3.0
PAPER_SNAPSHOT_WIDTH_MM = 53.0
PAPER_SNAPSHOT_SIZE_PX = int(round(PAPER_SNAPSHOT_WIDTH_MM / 25.4 * PAPER_PANEL_DPI))
PAPER_SNAPSHOT_SIZE_IN = PAPER_SNAPSHOT_SIZE_PX / PAPER_PANEL_DPI
PAPER_TITLE_FONTSIZE = 11
PAPER_AXIS_LABEL_FONTSIZE = 7
PAPER_TICK_FONTSIZE = 5
PAPER_LEGEND_FONTSIZE = 5
PAPER_ANNOTATION_FONTSIZE = 5
ANGLE_MEASUREMENT_LABEL = "Arbor rotation"
ANGLE_IMAGE_DIAGNOSTIC_LABEL = "Image-space rigid-set angle"
PAPER_SNAPSHOT_TITLE = "Camera Tracking Apparatus"
PAPER_SNAPSHOT_OUTPUT_FILENAME = "camera_tracking_apparatus.png"
LARGEST_CLOSE_ARBOR_ROTATION_PLOT_FILENAME = "largest_close_arbor_rotation_match_ures.png"
LARGEST_CLOSE_ARBOR_ROTATION_SQUARE_FILENAME = "largest_close_arbor_rotation_match_ures_53mm.png"
LARGEST_CLOSE_ARBOR_ROTATION_SNAPSHOT_FILENAME = "largest_close_arbor_rotation_match_snapshots.png"
LARGEST_CLOSE_ARBOR_ROTATION_CURVES_CACHE_FILENAME = (
    "largest_close_arbor_rotation_match_ures_curves.csv"
)
LARGEST_CLOSE_ARBOR_ROTATION_SUMMARY_CACHE_FILENAME = (
    "largest_close_arbor_rotation_match_summary.csv"
)
LARGEST_CLOSE_ARBOR_ROTATION_LEGACY_CURVES_CACHE_FILENAMES = (
    "largest_close_angle_match_ures_curves.csv",
)
LARGEST_CLOSE_ARBOR_ROTATION_LEGACY_SUMMARY_CACHE_FILENAMES = (
    "largest_close_angle_match_summary.csv",
)
MOMENT_VS_ARBOR_ROTATION_PLOT_FILENAME = "moment_vs_arbor_rotation.png"
MOMENT_VS_ARBOR_ROTATION_SQUARE_FILENAME = "moment_vs_arbor_rotation_53mm.png"
MOMENT_VS_ARBOR_ROTATION_CACHE_FILENAME = "moment_vs_arbor_rotation_series.csv"
MOMENT_VS_ARBOR_ROTATION_LEGACY_CACHE_FILENAMES = ("moment_vs_visual_angle_series.csv",)
PAPER_CURVE_TITLE = "Displacement Test\nRig vs Model"
PAPER_SNAPSHOT_MARGIN_PX = 180
PAPER_APRILTAG_MARGIN_PX = 60
SQUARE_PAPER_FIG_MM = 53.0
SQUARE_PAPER_FIG_IN = SQUARE_PAPER_FIG_MM / 25.4
SQUARE_PAPER_TEXT_PT = 8.0
SQUARE_PAPER_LEGEND_PT = 7.0
SQUARE_PAPER_TRACKER_MARKER_SIZE = 2.8
SQUARE_PAPER_TRACKER_SCATTER_SIZE = 8.0
SQUARE_PAPER_AXES_MM = 40.0
SQUARE_PAPER_LEFT_MARGIN_MM = 12.0
SQUARE_PAPER_RIGHT_MARGIN_MM = 2.0
SQUARE_PAPER_TOP_MARGIN_MM = 6.0
SQUARE_PAPER_BOTTOM_LEGEND_MM = 16.0
SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM = 28.0
SQUARE_PAPER_BOTTOM_FIT_LABELS_MM = 22.0
SQUARE_PAPER_LEGEND_LOC = "lower right"
SQUARE_PAPER_LEGEND_BBOX_TO_ANCHOR = (0.98, 0.02)
SQUARE_PAPER_LEGEND_NCOL = 1
LARGEST_CLOSE_ANGLE_COMPACT_BOX_Y = 0.012
LARGEST_CLOSE_ANGLE_COMPACT_BOX_HEIGHT_MM = 18.0
LARGEST_CLOSE_ANGLE_BOTTOM_EXTRA_MM = (
    SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM + LARGEST_CLOSE_ANGLE_COMPACT_BOX_HEIGHT_MM
)
FINAL_SWEEP_FINAL_NODE_COMPACT_BOX_Y = 0.010
FINAL_SWEEP_FINAL_NODE_BOTTOM_EXTRA_MM = SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM + 8.0
SQUARE_PAPER_URES_OVERLAY_STYLES = [
    ("#2D6A8A", (0, (4.0, 1.6))),
    ("#BC6C25", (0, (6.0, 1.8, 1.2, 1.8))),
    ("#5B8E55", (0, (1.2, 1.2))),
    ("#7F5539", (0, (5.0, 1.5))),
]
SQUARE_PAPER_URES_PRIMARY_STROKE = [pe.Stroke(linewidth=3.0, foreground="white", alpha=0.92), pe.Normal()]
SQUARE_PAPER_URES_OVERLAY_STROKE = [pe.Stroke(linewidth=1.8, foreground="white", alpha=0.68), pe.Normal()]
SQUARE_PAPER_URES_BAND_COLOR = "#6EA0C8"
SQUARE_PAPER_URES_BAND_ALPHA = 0.22
SQUARE_PAPER_URES_SWEEP_COLOR = "#000000"
SQUARE_PAPER_URES_TRACKED_COLOR = "#C1121F"
SQUARE_PAPER_URES_SWEEP_LINEWIDTH = 1.05
SQUARE_PAPER_URES_TRACKED_LINEWIDTH = 1.95
SQUARE_PAPER_URES_TRACKED_ZORDER = 5.8
SQUARE_PAPER_URES_SWEEP_ZORDER = 6.0
SQUARE_PAPER_URES_BAND_ZORDER = 6.6
SQUARE_PAPER_URES_Y_PADDING_FRAC = 0.06
SQUARE_PAPER_FIT_SWEEP_COLOR = "#B8B8B8"
SQUARE_PAPER_FIT_TRACKED_COLOR = "crimson"
FINAL_SWEEP_MOMENT_BOTTOM_EXTRA_MM = SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM + 8.0
SQUARE_PAPER_STRESS_OPT_INNER_COLOR = "#1F4E79"
SQUARE_PAPER_STRESS_OPT_OUTER_COLOR = "#B8653B"
SQUARE_PAPER_STRESS_FEA_INNER_COLOR = "#8FB7D7"
SQUARE_PAPER_STRESS_FEA_OUTER_COLOR = "#E0B07A"
SQUARE_PAPER_STRESS_OPT_LINEWIDTH = 1.55
SQUARE_PAPER_STRESS_FEA_LINEWIDTH = 1.0
SQUARE_PAPER_STRESS_FEA_ALPHA = 0.85
SQUARE_PAPER_STRESS_FEA_OUTLIER_MAX_TRIM = 8
SQUARE_PAPER_STRESS_FEA_OUTLIER_STEP_FACTOR = 4.0
SQUARE_PAPER_STRESS_FEA_OUTLIER_OFFSET_FACTOR = 1.4
SQUARE_PAPER_STRESS_FEA_OUTLIER_SLOPE_FACTOR = 3.0
SQUARE_PAPER_STRESS_FEA_OUTLIER_NEIGHBORHOOD = 4
SQUARE_PAPER_STRESS_TOP_PADDING_FRAC = 0.05
SQUARE_PAPER_STRESS_BOTTOM_PADDING_FRAC = 0.18
DEFAULT_APRILTAG_SIZE_MM = 1.226 * 25.4
APRILTAG_OVERLAY_COLOR_BGR = (80, 255, 80)
APRILTAG_LABEL_FONT_SCALE = 1.00
APRILTAG_LABEL_OUTLINE_THICKNESS = 5
APRILTAG_LABEL_THICKNESS = 2
LENS_DISTORTION_HEATMAP_ALPHA = 0.52
LENS_DISTORTION_HEATMAP_BASE_ALPHA = 0.10
LENS_DISTORTION_HEATMAP_GAMMA = 0.90
SNAPSHOT_SPIRAL_COLOR_BGR = (0, 220, 255)
SNAPSHOT_SPIRAL_OUTLINE_COLOR_BGR = (24, 24, 24)
SNAPSHOT_MARKER_RADIUS_PX = 9
SNAPSHOT_MARKER_OUTLINE_COLOR_BGR = (245, 245, 245)
SNAPSHOT_MARKER_LABEL_SCALE = 0.38
ARBOR_MARKER_COLOR_BGR = (0, 0, 255)
ARBOR_MARKER_OUTLINE_COLOR_BGR = (245, 245, 245)
DISTORTION_KEY_FOOTER_HEIGHT_IN = 0.62
DISTORTION_KEY_BAR_HEIGHT_FRAC = 0.24
DISTORTION_KEY_BAR_WIDTH_FRAC = 0.72
DISTORTION_KEY_COLOR_BGR = (90, 210, 90)
DISTORTION_KEY_LABEL_POINTSIZE = 12.0
DISTORTION_KEY_TICK_POINTSIZE = 12.0
STANDARD_GRAVITY_M_S2 = 9.80665
CAPSTAN_RADIUS_M = 0.0625
VECTOR_REFERENCE_COLOR_BGR = (255, 255, 0)
VECTOR_DEFORMED_COLOR_BGR = (0, 0, 255)
VECTOR_LINE_OUTLINE_COLOR_BGR = (24, 24, 24)
VECTOR_LINE_THICKNESS_PX = 2
VECTOR_LINE_OUTLINE_THICKNESS_PX = 4
VECTOR_LINE_TIP_LENGTH = 0.06
VECTOR_SNAPSHOT_MARGIN_PX = 48
VECTOR_SNAPSHOT_MIN_HALF_EXTENT_PX = 96
VECTOR_ANNOTATION_FONT_SCALE = 0.58
VECTOR_ANNOTATION_LINE_SPACING_PX = 22
VECTOR_ANNOTATION_PADDING_PX = 10
VECTOR_ANNOTATION_MARGIN_PX = 12
VECTOR_HISTORY_MIN_FPS = 10.0
FORCE_SCREEN_CROP_WIDTH_PX = 200
FORCE_SCREEN_CROP_HEIGHT_PX = 110
TRIPTYCH_PANEL_SIZE_PX = PAPER_SNAPSHOT_SIZE_PX
TRIPTYCH_BG_RGB = (246, 246, 244)
TRIPTYCH_TEXT_RGB = (28, 28, 28)
TRIPTYCH_MUTED_TEXT_RGB = (90, 90, 90)
TRIPTYCH_ACCENT_RGB = (34, 102, 184)
TRIPTYCH_BORDER_RGB = (188, 188, 188)
TRIPTYCH_MARGIN_PX = 20
TRIPTYCH_GAP_PX = 20
TRIPTYCH_HEADER_HEIGHT_PX = 70
TRIPTYCH_LABEL_HEIGHT_PX = 30
FORCE_PANEL_MARGIN_PX = 22
FORCE_PANEL_SCREEN_BOX_HEIGHT_PX = 250
FORCE_PANEL_SCREEN_BG_RGB = (34, 38, 48)
FORCE_PANEL_TITLE_SCALE = 0.78
FORCE_PANEL_VALUE_SCALE = 0.96
FORCE_PANEL_TEXT_SCALE = 0.62
FORCE_PANEL_TEXT_LINE_SPACING_PX = 28
LIVE_PLOT_DPI = 150


def moving_endpoint_row(rows: pd.DataFrame, disp_col: str) -> pd.Series:
    ordered = rows.sort_values("dot_id")
    if ordered.empty:
        raise ValueError("Tracker frame has no rows.")
    if len(ordered) == 1:
        return ordered.iloc[0]
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    return first if float(first[disp_col]) >= float(last[disp_col]) else last


def camera_ures_curve(rows: pd.DataFrame, arc_col: str, disp_col: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows.sort_values("dot_id").copy()
    x = (ordered[arc_col] / ordered[arc_col].max()).to_numpy(dtype=float)
    y = ordered[disp_col].to_numpy(dtype=float)
    return orient_curve_low_endpoint_to_zero(x, y)


def apply_ures_noise_floor(values: np.ndarray, floor: float = URES_NOISE_FLOOR_M) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[np.abs(out) < float(floor)] = 0.0
    return out


def apply_stress_noise_floor(values: np.ndarray, floor: float = STRESS_NOISE_FLOOR_PA) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[np.abs(out) < float(floor)] = 0.0
    return out


def orient_curve_low_endpoint_to_zero(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float).copy()
    y = np.asarray(y_values, dtype=float).copy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) == 0:
        return x, y
    span = float(x[-1] - x[0])
    if abs(span) > 1e-12:
        x = (x - x[0]) / span
    else:
        x = np.linspace(0.0, 1.0, len(x), dtype=float)
    min_idx = int(np.argmin(y))
    left_distance = min_idx
    right_distance = (len(y) - 1) - min_idx
    if right_distance < left_distance:
        x = 1.0 - x[::-1]
        y = y[::-1]
    return x, y


def spring_ures_curve_mm(spring: Spring) -> tuple[np.ndarray, np.ndarray]:
    th_abs, x_abs, y_abs = spring.construct_global_spring()
    def_x = np.asarray(spring.nodes[:, 9], dtype=float)
    def_y = np.asarray(spring.nodes[:, 10], dtype=float)
    lengths = np.asarray(spring.nodes[:, 2], dtype=float)

    se = math.sin(float(th_abs[-1]))
    ce = math.cos(float(th_abs[-1]))
    def_x_tip = ce * (lengths[-1] + def_x[-1]) - se * def_y[-1]
    def_y_tip = se * (lengths[-1] + def_x[-1]) + ce * def_y[-1]

    x_tip = float(x_abs[-1]) + def_x_tip
    y_tip = float(y_abs[-1]) + def_y_tip

    x_def = np.append(np.asarray(x_abs, dtype=float), x_tip)
    y_def = np.append(np.asarray(y_abs, dtype=float), y_tip)

    x_root, y_root = spring.construct_root_geometry()
    ures_m = apply_ures_noise_floor(np.hypot(x_def - x_root, y_def - y_root))
    ures_mm = ures_m * 1e3

    s_abs = np.concatenate(([0.0], np.cumsum(lengths)))
    s_norm = s_abs / max(float(s_abs[-1]), 1e-12)
    return np.asarray(ures_mm, dtype=float), np.asarray(s_norm, dtype=float)


def infer_ures_csv_side(path: Path) -> str | None:
    stem = path.stem.lower()
    if "inner" in stem:
        return "inner"
    if "outer" in stem:
        return "outer"
    return None


def build_ures_overlay_label(index: int) -> str:
    return f"URES {index + 1}"


def build_stress_overlay_label(side: str | None, index: int = 0) -> str:
    if side == "inner":
        return "FEA inner stress" if index == 0 else f"FEA inner stress {index + 1}"
    if side == "outer":
        return "FEA outer stress" if index == 0 else f"FEA outer stress {index + 1}"
    return f"FEA stress {index + 1}"


def resolve_overlay_ures_csvs(
    requested_paths: list[Path] | None,
    input_dir: Path | None,
) -> list[Path]:
    if requested_paths is None:
        return []

    if requested_paths:
        resolved_paths: list[Path] = []
        for csv_path in requested_paths:
            resolved_path = csv_path.expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"SolidWorks URES CSV does not exist: {resolved_path}")
            resolved_paths.append(resolved_path)
        return resolved_paths

    if input_dir is None:
        raise ValueError(
            "--overlay-ures-csv without explicit paths requires --input-dir for auto-discovery."
        )

    discovered_paths: list[Path] = []
    for search_dir in (input_dir / "results", input_dir):
        if not search_dir.is_dir():
            continue
        discovered_paths.extend(sorted(search_dir.glob("ures_*.csv")))

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for csv_path in discovered_paths:
        resolved_path = csv_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        unique_paths.append(resolved_path)

    if unique_paths:
        print(f"[overlay] Auto-discovered {len(unique_paths)} SolidWorks URES CSV file(s).")
        for csv_path in unique_paths:
            print(f"[overlay]   {csv_path}")
    else:
        print(f"[warn] No ures_*.csv files found under {input_dir/'results'} or {input_dir}")
    return unique_paths


def resolve_stress_overlay_csvs(input_dir: Path | None) -> list[Path]:
    if input_dir is None:
        return []

    discovered_paths: list[Path] = []
    for search_dir in (input_dir / "results", input_dir):
        if not search_dir.is_dir():
            continue
        discovered_paths.extend(sorted(search_dir.glob("stress_*.csv")))

    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for csv_path in discovered_paths:
        resolved_path = csv_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        unique_paths.append(resolved_path)

    if unique_paths:
        print(f"[overlay] Auto-discovered {len(unique_paths)} SolidWorks stress CSV file(s).")
        for csv_path in unique_paths:
            print(f"[overlay]   {csv_path}")
    return unique_paths


def load_ures_overlay_curves(csv_paths: list[Path], spring: Spring) -> list[dict[str, object]]:
    if not csv_paths:
        return []

    styles = [
        ("tab:blue", "--"),
        ("tab:orange", "-."),
        ("tab:green", ":"),
        ("tab:brown", "--"),
    ]
    overlays: list[dict[str, object]] = []
    for idx, csv_path in enumerate(csv_paths):
        side = infer_ures_csv_side(csv_path)
        try:
            df = ures_solidworks_2025(
                csv_path,
                spring,
                type="ures",
                side=side,
                report_tip_rotation=False,
            )
        except Exception as exc:
            print(f"[warn] Could not parse SolidWorks URES CSV {csv_path.name}: {exc}")
            continue

        if df.empty:
            print(f"[warn] SolidWorks URES CSV {csv_path.name} produced no rows.")
            continue

        color, linestyle = styles[idx % len(styles)]
        s_norm, curve_mm = orient_curve_low_endpoint_to_zero(
            df["s"].to_numpy(dtype=float),
            df["ures"].to_numpy(dtype=float) * 1e3,
        )
        label = build_ures_overlay_label(len(overlays))
        overlays.append(
            {
                "path": csv_path,
                "label": label,
                "s_norm": s_norm,
                "curve_mm": curve_mm,
                "color": color,
                "linestyle": linestyle,
                "linewidth": 1.0,
            }
        )
        print(f"[overlay] Loaded {csv_path.name} as {label}")
    return overlays


def spring_inner_outer_stress_curves_mpa(spring: Spring) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.asarray(spring.nodes[:, 2], dtype=float)
    s_abs = np.concatenate(([0.0], np.cumsum(lengths)))
    s_norm = s_abs / max(float(s_abs[-1]), 1e-12)

    sigma_inner_pa = apply_stress_noise_floor(np.abs(np.asarray(spring.nodes[:, 15], dtype=float)))
    sigma_outer_pa = apply_stress_noise_floor(np.abs(np.asarray(spring.nodes[:, 16], dtype=float)))
    sigma_inner_mpa = sigma_inner_pa / 1e6
    sigma_outer_mpa = sigma_outer_pa / 1e6

    sigma_inner_mpa = np.concatenate(([float(sigma_inner_mpa[0])], sigma_inner_mpa))
    sigma_outer_mpa = np.concatenate(([float(sigma_outer_mpa[0])], sigma_outer_mpa))
    return np.asarray(s_norm, dtype=float), np.asarray(sigma_inner_mpa, dtype=float), np.asarray(sigma_outer_mpa, dtype=float)


def load_stress_overlay_curves(csv_paths: list[Path]) -> list[dict[str, object]]:
    if not csv_paths:
        return []

    side_counts: dict[str | None, int] = {}
    overlays: list[dict[str, object]] = []
    for csv_path in csv_paths:
        try:
            df = stress_solidworks_2025(csv_path)
        except Exception as exc:
            print(f"[warn] Could not parse SolidWorks stress CSV {csv_path.name}: {exc}")
            continue

        if df.empty:
            print(f"[warn] SolidWorks stress CSV {csv_path.name} produced no rows.")
            continue

        side = infer_ures_csv_side(csv_path)
        side_counts[side] = side_counts.get(side, 0) + 1
        variant_idx = side_counts[side] - 1
        if side == "inner":
            color = SQUARE_PAPER_STRESS_FEA_INNER_COLOR
            linestyles = [(0, (1.4, 1.2)), ":"]
        elif side == "outer":
            color = SQUARE_PAPER_STRESS_FEA_OUTER_COLOR
            linestyles = [(0, (4.0, 1.4)), (0, (3.0, 1.2, 1.1, 1.2))]
        else:
            color = "0.55"
            linestyles = [":"]

        label = build_stress_overlay_label(side, variant_idx)
        overlays.append(
            {
                "path": csv_path,
                "label": label,
                "side": side,
                "s_norm": df["s"].to_numpy(dtype=float),
                "stress_mpa": np.abs(df["stress"].to_numpy(dtype=float)) / 1e6,
                "color": color,
                "linestyle": linestyles[variant_idx % len(linestyles)],
                "linewidth": SQUARE_PAPER_STRESS_FEA_LINEWIDTH,
            }
        )
        print(f"[overlay] Loaded {csv_path.name} as {label}")
    return overlays


def spring_stress_curve_mpa(spring: Spring) -> tuple[np.ndarray, np.ndarray]:
    node_idx = np.asarray(spring.nodes[:, 0], dtype=float)
    stress_pa = apply_stress_noise_floor(np.asarray(spring.nodes[:, 12], dtype=float))
    stress_mpa = stress_pa / 1e6
    return node_idx, stress_mpa


def render_sweep_stress_plot(sweep_records: list[dict], output_path: Path) -> None:
    if not sweep_records:
        raise ValueError("No sweep records available for stress plotting.")

    fig, ax = plt.subplots(figsize=(PAPER_PANEL_SIZE_IN, PAPER_PANEL_SIZE_IN), dpi=PAPER_PANEL_DPI)
    n_records = len(sweep_records)
    rom_label_trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for idx, record in enumerate(sweep_records):
        spring = record["spring"]
        node_idx, stress_mpa = spring_stress_curve_mpa(spring)
        rom_deg = float(record["rom_deg"])
        if n_records == 1:
            gray_level = 0.0
        else:
            frac = idx / float(n_records - 1)
            gray_level = 0.78 * (1.0 - frac)
        color = (gray_level, gray_level, gray_level)
        lw = 1.2
        alpha = 1.0 if idx in (0, len(sweep_records) - 1) else 0.7
        label = None
        if idx == 0:
            label = "undeformed"
        elif idx == len(sweep_records) - 1:
            label = "target deformation"
        ax.plot(node_idx, stress_mpa, color=color, lw=lw, alpha=alpha, label=label)
        if idx == 0 or idx == n_records - 1 or idx % 2 == 0:
            ax.text(
                1.008,
                float(stress_mpa[-1]),
                f"{rom_deg:.1f}°",
                transform=rom_label_trans,
                ha="left",
                va="center",
                fontsize=5,
                color="black",
                clip_on=False,
            )

    ax.set_xlabel("Node index", fontsize=7, labelpad=2)
    ax.set_ylabel("Stress (MPa)", fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.margins(x=0.05, y=0.08)
    ax.set_box_aspect(1)
    ax.grid(False)
    if len(sweep_records) > 1:
        handles, labels = ax.get_legend_handles_labels()
        legend_order = ["target deformation", "undeformed"]
        ordered_pairs = [(h, l) for target in legend_order for h, l in zip(handles, labels) if l == target]
        legend_kwargs = dict(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            fontsize=5,
            ncol=1,
            borderaxespad=0.0,
        )
        if ordered_pairs:
            ordered_handles, ordered_labels = zip(*ordered_pairs)
            ax.legend(ordered_handles, ordered_labels, **legend_kwargs)
        else:
            ax.legend(**legend_kwargs)
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.33, top=0.96)
    fig.savefig(output_path, dpi=PAPER_PANEL_DPI)
    plt.close(fig)


def render_paper_stress_panel(
    *,
    spring: Spring,
    solidworks_stress_overlays: list[dict[str, object]] | None = None,
    figure_title: str = "Stress Test\nRig vs Model",
) -> np.ndarray:
    axes_size_in = PAPER_CURVE_AXES_SIZE_MM / 25.4
    left_margin_in = PAPER_CURVE_LEFT_MARGIN_MM / 25.4
    right_margin_in = PAPER_CURVE_RIGHT_MARGIN_MM / 25.4
    top_margin_in = PAPER_CURVE_TOP_MARGIN_MM / 25.4
    bottom_margin_in = PAPER_CURVE_BOTTOM_MARGIN_MM / 25.4
    legend_bottom_in = PAPER_CURVE_LEGEND_BOTTOM_MM / 25.4

    fig_width_in = left_margin_in + axes_size_in + right_margin_in
    fig_height_in = top_margin_in + axes_size_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=PAPER_PANEL_DPI)
    ax_left = left_margin_in / fig_width_in
    ax_bottom = bottom_margin_in / fig_height_in
    ax_width = axes_size_in / fig_width_in
    ax_height = axes_size_in / fig_height_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    spring_s_norm, spring_inner_mpa, spring_outer_mpa = spring_inner_outer_stress_curves_mpa(spring)
    ax.plot(
        spring_s_norm,
        spring_inner_mpa,
        "-",
        color="tab:blue",
        lw=1.0,
        label="optimizer inner stress",
    )
    ax.plot(
        spring_s_norm,
        spring_outer_mpa,
        "-",
        color="tab:orange",
        lw=1.0,
        label="optimizer outer stress",
    )

    for overlay in solidworks_stress_overlays or []:
        ax.plot(
            np.asarray(overlay["s_norm"], dtype=float),
            np.asarray(overlay["stress_mpa"], dtype=float),
            linestyle=str(overlay["linestyle"]),
            color=str(overlay["color"]),
            lw=float(overlay["linewidth"]),
            label=str(overlay["label"]),
        )

    ax.set_xlabel("Normalized Arc Length", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Stress (MPa)", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.set_title(figure_title, fontsize=PAPER_TITLE_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=PAPER_TICK_FONTSIZE)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.35)
    ax.set_box_aspect(1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=((2.0 * left_margin_in + axes_size_in) / (2.0 * fig_width_in), legend_bottom_in / fig_height_in),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
        fontsize=PAPER_LEGEND_FONTSIZE,
        ncol=1,
    )

    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def annotate_endpoint(
    ax,
    y_value: float,
    label: str,
    color: str,
    linestyle: str = "--",
    text_offset_pts: float = 0.0,
    fontsize: float = 8,
) -> None:
    ax.axhline(y_value, color=color, lw=1.0, ls=linestyle, alpha=0.65, zorder=1)
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([0.0], [y_value], marker=">", ms=6, color=color, transform=trans, clip_on=False, zorder=6)
    ax.annotate(
        label,
        xy=(1.0, y_value),
        xycoords=trans,
        xytext=(-6, text_offset_pts),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize,
        color=color,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85),
    )


def force_to_moment_nm(force_n: float) -> float:
    return float(force_n) * CAPSTAN_RADIUS_M


def moment_to_force_n(moment_nm: float) -> float:
    return float(moment_nm) / CAPSTAN_RADIUS_M


def force_to_mass_kg(force_n: float) -> float:
    return float(force_n) / STANDARD_GRAVITY_M_S2


def _first_finite_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return None
    return float(finite[0])


def _first_nonempty_string(values: pd.Series) -> str | None:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        return text
    return None


def _first_boolish_value(values: pd.Series) -> bool | None:
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _series_float_value(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _series_string_value(row: pd.Series, key: str) -> str | None:
    if key not in row.index or pd.isna(row[key]):
        return None
    text = str(row[key]).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _series_bool_value(row: pd.Series, key: str) -> bool | None:
    if key not in row.index or pd.isna(row[key]):
        return None
    value = row[key]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _nearest_row_by_value(table: pd.DataFrame, column: str, target: float) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    valid = table.loc[values.notna()]
    if valid.empty:
        raise ValueError(f"No finite values available in '{column}' for nearest-match lookup.")
    idx = int((pd.to_numeric(valid[column], errors="coerce") - float(target)).abs().idxmin())
    return table.loc[idx]


def _curve_overlap_error_samples(
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    *,
    min_samples: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    optimizer_x = np.asarray(optimizer_s_norm, dtype=float)
    optimizer_y = np.asarray(optimizer_curve_mm, dtype=float)
    tracker_x = np.asarray(tracker_s_norm, dtype=float)
    tracker_y = np.asarray(tracker_curve_mm, dtype=float)

    optimizer_mask = np.isfinite(optimizer_x) & np.isfinite(optimizer_y)
    tracker_mask = np.isfinite(tracker_x) & np.isfinite(tracker_y)
    if not optimizer_mask.any() or not tracker_mask.any():
        return None

    optimizer_x = optimizer_x[optimizer_mask]
    optimizer_y = optimizer_y[optimizer_mask]
    tracker_x = tracker_x[tracker_mask]
    tracker_y = tracker_y[tracker_mask]

    optimizer_order = np.argsort(optimizer_x)
    tracker_order = np.argsort(tracker_x)
    optimizer_x = optimizer_x[optimizer_order]
    optimizer_y = optimizer_y[optimizer_order]
    tracker_x = tracker_x[tracker_order]
    tracker_y = tracker_y[tracker_order]

    optimizer_x, optimizer_unique_idx = np.unique(optimizer_x, return_index=True)
    optimizer_y = optimizer_y[optimizer_unique_idx]
    tracker_x, tracker_unique_idx = np.unique(tracker_x, return_index=True)
    tracker_y = tracker_y[tracker_unique_idx]

    if optimizer_x.size < 2 or tracker_x.size < 2:
        return None

    overlap_min = max(float(optimizer_x[0]), float(tracker_x[0]))
    overlap_max = min(float(optimizer_x[-1]), float(tracker_x[-1]))
    if not (math.isfinite(overlap_min) and math.isfinite(overlap_max) and overlap_max > overlap_min):
        return None

    sample_count = max(min_samples, optimizer_x.size, tracker_x.size)
    sample_x = np.linspace(overlap_min, overlap_max, sample_count, dtype=float)
    optimizer_interp = np.interp(sample_x, optimizer_x, optimizer_y)
    tracker_interp = np.interp(sample_x, tracker_x, tracker_y)
    return sample_x, np.abs(optimizer_interp - tracker_interp)


def tracker_curve_scale_profile_mm_per_px(
    tracker_rows: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if tracker_rows is None or tracker_rows.empty:
        return None

    coordinate_sets = [
        ("x_cur", "y_cur", "x_cur_plane_mm", "y_cur_plane_mm"),
        ("x_ref", "y_ref", "x_ref_plane_mm", "y_ref_plane_mm"),
    ]
    if "arc_length_mm" not in tracker_rows.columns:
        return None

    arc_length_mm = pd.to_numeric(tracker_rows["arc_length_mm"], errors="coerce")
    finite_arc_rows = tracker_rows.loc[arc_length_mm.notna()].copy()
    if finite_arc_rows.empty:
        return None
    finite_arc_rows["arc_length_mm"] = pd.to_numeric(finite_arc_rows["arc_length_mm"], errors="coerce")
    finite_arc_rows = finite_arc_rows.sort_values("arc_length_mm")
    max_arc_mm = float(finite_arc_rows["arc_length_mm"].max())
    if not math.isfinite(max_arc_mm) or max_arc_mm <= 1e-12:
        return None

    for x_px_col, y_px_col, x_mm_col, y_mm_col in coordinate_sets:
        required = {x_px_col, y_px_col, x_mm_col, y_mm_col}
        if not required.issubset(finite_arc_rows.columns):
            continue
        rows = finite_arc_rows[
            finite_arc_rows[list(required)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        ].copy()
        if len(rows) < 2:
            continue
        rows = rows.sort_values("arc_length_mm")
        s_norm = rows["arc_length_mm"].to_numpy(dtype=float) / max_arc_mm
        image_xy = rows[[x_px_col, y_px_col]].to_numpy(dtype=float)
        plane_xy = rows[[x_mm_col, y_mm_col]].to_numpy(dtype=float)

        s_samples: list[float] = []
        scale_samples: list[float] = []
        n_rows = len(rows)
        for idx in range(n_rows):
            prev_idx = idx - 1 if idx > 0 else idx
            next_idx = idx + 1 if idx + 1 < n_rows else idx
            if prev_idx == next_idx:
                continue
            plane_dist_mm = float(np.linalg.norm(plane_xy[next_idx] - plane_xy[prev_idx]))
            image_dist_px = float(np.linalg.norm(image_xy[next_idx] - image_xy[prev_idx]))
            if not (
                math.isfinite(plane_dist_mm)
                and math.isfinite(image_dist_px)
                and plane_dist_mm > 1e-12
                and image_dist_px > 1e-12
            ):
                continue
            s_samples.append(float(s_norm[idx]))
            scale_samples.append(plane_dist_mm / image_dist_px)

        if scale_samples:
            return np.asarray(s_samples, dtype=float), np.asarray(scale_samples, dtype=float)

    return None


def curve_mean_absolute_error_metrics(
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    *,
    tracker_rows: pd.DataFrame | None = None,
    min_samples: int = 32,
) -> dict[str, float | None]:
    overlap_samples = _curve_overlap_error_samples(
        optimizer_s_norm,
        optimizer_curve_mm,
        tracker_s_norm,
        tracker_curve_mm,
        min_samples=min_samples,
    )
    if overlap_samples is None:
        return {
            "mean_abs_error_mm": None,
            "mean_abs_error_m": None,
            "mean_abs_error_px": None,
        }

    sample_s_norm, abs_error_mm = overlap_samples
    mean_abs_error_mm = float(np.mean(abs_error_mm))
    mean_abs_error_px: float | None = None

    scale_profile = tracker_curve_scale_profile_mm_per_px(tracker_rows)
    if scale_profile is not None:
        scale_s_norm, scale_mm_per_px = scale_profile
        finite_mask = np.isfinite(scale_s_norm) & np.isfinite(scale_mm_per_px) & (scale_mm_per_px > 1e-12)
        if np.any(finite_mask):
            scale_s_norm = scale_s_norm[finite_mask]
            scale_mm_per_px = scale_mm_per_px[finite_mask]
            order = np.argsort(scale_s_norm)
            scale_s_norm = scale_s_norm[order]
            scale_mm_per_px = scale_mm_per_px[order]
            scale_s_norm, unique_idx = np.unique(scale_s_norm, return_index=True)
            scale_mm_per_px = scale_mm_per_px[unique_idx]
            if scale_s_norm.size == 1:
                scale_interp = np.full_like(sample_s_norm, float(scale_mm_per_px[0]), dtype=float)
            elif scale_s_norm.size >= 2:
                scale_interp = np.interp(
                    sample_s_norm,
                    scale_s_norm,
                    scale_mm_per_px,
                    left=float(scale_mm_per_px[0]),
                    right=float(scale_mm_per_px[-1]),
                )
            else:
                scale_interp = None
            if scale_interp is not None and np.all(np.isfinite(scale_interp)) and np.all(scale_interp > 1e-12):
                mean_abs_error_px = float(np.mean(abs_error_mm / scale_interp))

    return {
        "mean_abs_error_mm": mean_abs_error_mm,
        "mean_abs_error_m": mean_abs_error_mm / 1000.0,
        "mean_abs_error_px": mean_abs_error_px,
    }


def format_mean_ures_error_line(metrics: dict[str, float | None] | None) -> str | None:
    if not metrics:
        return None
    mean_abs_error_m = metrics.get("mean_abs_error_m")
    if mean_abs_error_m is None or not math.isfinite(float(mean_abs_error_m)):
        return None
    mean_abs_error_px = metrics.get("mean_abs_error_px")
    if mean_abs_error_px is not None and math.isfinite(float(mean_abs_error_px)):
        return f"Mean |URES error|: {float(mean_abs_error_m):.6f} m | {float(mean_abs_error_px):.2f} px"
    return f"Mean |URES error|: {float(mean_abs_error_m):.6f} m"


def curve_mean_absolute_error_mm(
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    *,
    min_samples: int = 32,
) -> float | None:
    metrics = curve_mean_absolute_error_metrics(
        optimizer_s_norm,
        optimizer_curve_mm,
        tracker_s_norm,
        tracker_curve_mm,
        tracker_rows=None,
        min_samples=min_samples,
    )
    return metrics["mean_abs_error_mm"]


def _format_force_value(force_n: float | None) -> str:
    if force_n is None or not math.isfinite(force_n):
        return "n/a"
    return f"{force_n:.3f} N ({force_to_mass_kg(force_n):.3f} kg)"


def _format_mass_value(mass_kg: float | None) -> str:
    if mass_kg is None or not math.isfinite(mass_kg):
        return "n/a"
    return f"{mass_kg:.3f} kg"


def _format_confidence_value(confidence: float | None) -> str:
    if confidence is None or not math.isfinite(confidence):
        return "n/a"
    return f"{confidence:.2f}"


def _format_force_digits_value(force_digits: str | None) -> str:
    if force_digits is None:
        return "n/a"
    digits_only = "".join(ch for ch in str(force_digits) if ch.isdigit())
    if not digits_only:
        text = str(force_digits).strip()
        return text or "n/a"
    digits_only = digits_only[-4:].zfill(4)
    return f"{digits_only[0]}.{digits_only[1:]} kg"


def _format_moment_value(moment_nm: float | None) -> str:
    if moment_nm is None or not math.isfinite(moment_nm):
        return "n/a"
    return f"{moment_nm:.3f} N*m"


def _format_stiffness_value(stiffness_nm_per_rad: float | None) -> str:
    if stiffness_nm_per_rad is None or not math.isfinite(stiffness_nm_per_rad):
        return "n/a"
    return f"{stiffness_nm_per_rad:.3f} N*m/rad"


def rotational_stiffness_nm_per_rad(moment_nm: float | None, angle_deg: float | None) -> float | None:
    if moment_nm is None or angle_deg is None:
        return None
    if not (math.isfinite(float(moment_nm)) and math.isfinite(float(angle_deg))):
        return None
    angle_rad = math.radians(abs(float(angle_deg)))
    if angle_rad <= 1e-12:
        return None
    return float(moment_nm) / angle_rad


def stiffness_metrics(
    *,
    optimizer_moment_nm: float | None,
    optimizer_angle_deg: float | None,
    tracker_moment_nm: float | None,
    tracker_angle_deg: float | None,
) -> dict[str, float | None]:
    optimizer_stiffness = rotational_stiffness_nm_per_rad(optimizer_moment_nm, optimizer_angle_deg)
    tracker_stiffness = rotational_stiffness_nm_per_rad(tracker_moment_nm, tracker_angle_deg)
    if optimizer_stiffness is None or tracker_stiffness is None:
        return {
            "optimizer_stiffness_nm_per_rad": optimizer_stiffness,
            "tracker_stiffness_nm_per_rad": tracker_stiffness,
            "stiffness_error_nm_per_rad": None,
            "stiffness_error_pct": None,
        }
    error_nm_per_rad = tracker_stiffness - optimizer_stiffness
    error_pct = None
    if abs(optimizer_stiffness) > 1e-12:
        error_pct = 100.0 * error_nm_per_rad / optimizer_stiffness
    return {
        "optimizer_stiffness_nm_per_rad": optimizer_stiffness,
        "tracker_stiffness_nm_per_rad": tracker_stiffness,
        "stiffness_error_nm_per_rad": error_nm_per_rad,
        "stiffness_error_pct": error_pct,
    }


def build_stiffness_summary_lines(
    *,
    optimizer_moment_nm: float | None,
    optimizer_angle_deg: float | None,
    tracker_moment_nm: float | None,
    tracker_angle_deg: float | None,
) -> list[str]:
    metrics = stiffness_metrics(
        optimizer_moment_nm=optimizer_moment_nm,
        optimizer_angle_deg=optimizer_angle_deg,
        tracker_moment_nm=tracker_moment_nm,
        tracker_angle_deg=tracker_angle_deg,
    )
    if (
        metrics["optimizer_stiffness_nm_per_rad"] is None
        or metrics["tracker_stiffness_nm_per_rad"] is None
    ):
        return []
    error_pct = metrics["stiffness_error_pct"]
    error_nm_per_rad = metrics["stiffness_error_nm_per_rad"]
    error_text = _format_stiffness_value(error_nm_per_rad)
    if error_pct is not None and math.isfinite(float(error_pct)):
        error_text = f"{error_text} ({error_pct:+.1f}%)"
    return [
        f"Sweep stiffness: {_format_stiffness_value(metrics['optimizer_stiffness_nm_per_rad'])}",
        f"Tracker stiffness: {_format_stiffness_value(metrics['tracker_stiffness_nm_per_rad'])}",
        f"Stiffness error: {error_text}",
    ]


def load_tracker_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"frame", "dot_id", "arc_length_mm", "displacement_mm", "angle_change_deg", "time_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Tracker CSV is missing required columns: {sorted(missing)}")

    preferred_visual_dot_id: int | None = None
    if "arbor_rigid_dot_id" in df.columns:
        rigid_dot_id = _first_finite_numeric(df["arbor_rigid_dot_id"])
        if rigid_dot_id is not None:
            preferred_visual_dot_id = int(rigid_dot_id)
    if "dot_id" in df.columns and not df.empty:
        if preferred_visual_dot_id is None:
            reference_frame = int(df["frame"].min())
            reference_rows = df[df["frame"] == reference_frame]
            reference_dot_ids = pd.to_numeric(reference_rows["dot_id"], errors="coerce")
            finite_dot_ids = reference_dot_ids[np.isfinite(reference_dot_ids)]
            if not finite_dot_ids.empty:
                preferred_visual_dot_id = int(finite_dot_ids.max())

    frame_records: list[dict[str, float | int]] = []
    for frame, grp in df.groupby("frame", sort=True):
        tip_row = moving_endpoint_row(grp, "displacement_mm")
        image_angle_deg = math.nan
        try:
            visual_angle = included_vector_angle_deg(
                *tracker_first_node_vectors_px(
                    grp,
                    preferred_dot_id=preferred_visual_dot_id,
                )
            )
        except ValueError:
            visual_angle = None
        if visual_angle is not None and math.isfinite(float(visual_angle)):
            image_angle_deg = float(visual_angle)
        arbor_rotation_deg = (
            _first_finite_numeric(grp["arbor_rotation_deg"])
            if "arbor_rotation_deg" in grp.columns
            else None
        )
        arbor_rotation_signed_deg = (
            _first_finite_numeric(grp["arbor_rotation_signed_deg"])
            if "arbor_rotation_signed_deg" in grp.columns
            else None
        )
        arbor_rigid_dot_id = (
            _first_finite_numeric(grp["arbor_rigid_dot_id"])
            if "arbor_rigid_dot_id" in grp.columns
            else None
        )
        visual_angle_deg = (
            float(arbor_rotation_deg)
            if arbor_rotation_deg is not None
            else (
                float(image_angle_deg)
                if math.isfinite(float(image_angle_deg))
                else float(tip_row["angle_change_deg"])
            )
        )
        frame_in_window = (
            int(grp["frame_in_window"].iloc[0])
            if "frame_in_window" in grp.columns
            else int(frame) - int(df["frame"].min())
        )
        tracker_force_n = _first_finite_numeric(grp["force_n"]) if "force_n" in grp.columns else None
        tracker_mass_kg = _first_finite_numeric(grp["mass_kg"]) if "mass_kg" in grp.columns else None
        tracker_force_n_raw = _first_finite_numeric(grp["force_n_raw"]) if "force_n_raw" in grp.columns else None
        tracker_mass_kg_raw = _first_finite_numeric(grp["mass_kg_raw"]) if "mass_kg_raw" in grp.columns else None
        if tracker_force_n is None and tracker_mass_kg is not None:
            tracker_force_n = tracker_mass_kg * STANDARD_GRAVITY_M_S2
        if tracker_mass_kg is None and tracker_force_n is not None:
            tracker_mass_kg = force_to_mass_kg(tracker_force_n)
        tracker_moment_nm = force_to_moment_nm(tracker_force_n) if tracker_force_n is not None else math.nan
        tracker_force_confidence = (
            _first_finite_numeric(grp["force_confidence"]) if "force_confidence" in grp.columns else None
        )
        tracker_force_confidence_raw = (
            _first_finite_numeric(grp["force_confidence_raw"]) if "force_confidence_raw" in grp.columns else None
        )
        force_screen_visible = (
            _first_boolish_value(grp["force_screen_visible"]) if "force_screen_visible" in grp.columns else None
        )
        force_digits = _first_nonempty_string(grp["force_digits"]) if "force_digits" in grp.columns else None
        force_reason = _first_nonempty_string(grp["force_reason"]) if "force_reason" in grp.columns else None
        force_bbox_x = _first_finite_numeric(grp["force_bbox_x"]) if "force_bbox_x" in grp.columns else None
        force_bbox_y = _first_finite_numeric(grp["force_bbox_y"]) if "force_bbox_y" in grp.columns else None
        force_bbox_w = _first_finite_numeric(grp["force_bbox_w"]) if "force_bbox_w" in grp.columns else None
        force_bbox_h = _first_finite_numeric(grp["force_bbox_h"]) if "force_bbox_h" in grp.columns else None
        force_center_x = _first_finite_numeric(grp["force_center_x"]) if "force_center_x" in grp.columns else None
        force_center_y = _first_finite_numeric(grp["force_center_y"]) if "force_center_y" in grp.columns else None
        frame_records.append(
            {
                "frame": int(frame),
                "frame_in_window": frame_in_window,
                "time_s": float(tip_row["time_s"]),
                "tip_dot_id": int(tip_row["dot_id"]),
                "visual_angle_deg": visual_angle_deg,
                "image_angle_deg": image_angle_deg,
                "arbor_rotation_deg": (
                    float(arbor_rotation_deg) if arbor_rotation_deg is not None else math.nan
                ),
                "arbor_rotation_signed_deg": (
                    float(arbor_rotation_signed_deg) if arbor_rotation_signed_deg is not None else math.nan
                ),
                "arbor_rigid_dot_id": (
                    float(arbor_rigid_dot_id) if arbor_rigid_dot_id is not None else math.nan
                ),
                "direct_angle_deg": float(tip_row["angle_change_deg"]),
                "tip_displacement_mm": float(tip_row["displacement_mm"]),
                "tracker_mass_kg": float(tracker_mass_kg) if tracker_mass_kg is not None else math.nan,
                "tracker_force_n": float(tracker_force_n) if tracker_force_n is not None else math.nan,
                "tracker_moment_nm": float(tracker_moment_nm),
                "tracker_mass_kg_raw": (
                    float(tracker_mass_kg_raw) if tracker_mass_kg_raw is not None else math.nan
                ),
                "tracker_force_n_raw": (
                    float(tracker_force_n_raw) if tracker_force_n_raw is not None else math.nan
                ),
                "tracker_force_confidence": (
                    float(tracker_force_confidence) if tracker_force_confidence is not None else math.nan
                ),
                "tracker_force_confidence_raw": (
                    float(tracker_force_confidence_raw)
                    if tracker_force_confidence_raw is not None
                    else math.nan
                ),
                "force_screen_visible": (
                    bool(force_screen_visible) if force_screen_visible is not None else False
                ),
                "force_digits": force_digits or "",
                "force_reason": force_reason or "",
                "force_bbox_x": float(force_bbox_x) if force_bbox_x is not None else math.nan,
                "force_bbox_y": float(force_bbox_y) if force_bbox_y is not None else math.nan,
                "force_bbox_w": float(force_bbox_w) if force_bbox_w is not None else math.nan,
                "force_bbox_h": float(force_bbox_h) if force_bbox_h is not None else math.nan,
                "force_center_x": float(force_center_x) if force_center_x is not None else math.nan,
                "force_center_y": float(force_center_y) if force_center_y is not None else math.nan,
            }
        )

    frame_table = pd.DataFrame(frame_records).sort_values("frame").reset_index(drop=True)
    return df, frame_table


def build_matches(sweep_records: list[dict], frame_table: pd.DataFrame) -> pd.DataFrame:
    visual_angles = pd.to_numeric(frame_table["visual_angle_deg"], errors="coerce")
    valid_frames = frame_table.loc[visual_angles.notna()]
    if valid_frames.empty:
        raise ValueError("Tracker frame table has no readable per-frame arbor rotations.")

    rows: list[dict[str, float | int | str]] = []
    for record in sweep_records:
        rom_deg = abs(float(record["rom_deg"]))
        idx = int((pd.to_numeric(valid_frames["visual_angle_deg"], errors="coerce") - rom_deg).abs().idxmin())
        best = valid_frames.loc[idx]
        rows.append(
            {
                "sweep_index": int(record["index"]),
                "sweep_name": str(record["name"]),
                "optimizer_rom_deg": rom_deg,
                "matched_frame": int(best["frame"]),
                "matched_frame_in_window": int(best["frame_in_window"]),
                "matched_time_s": float(best["time_s"]),
                "matched_tip_dot_id": int(best["tip_dot_id"]),
                "matched_tracker_visual_angle_deg": float(best["visual_angle_deg"]),
                "matched_tracker_direct_angle_deg": float(best["direct_angle_deg"]),
                "matched_tracker_tip_displacement_mm": float(best["tip_displacement_mm"]),
                "abs_angle_error_deg": float(abs(float(best["visual_angle_deg"]) - rom_deg)),
            }
        )
    return pd.DataFrame(rows).sort_values("sweep_index").reset_index(drop=True)


def build_matches_by_moment(sweep_records: list[dict], frame_table: pd.DataFrame) -> pd.DataFrame:
    if "tracker_moment_nm" not in frame_table.columns:
        raise ValueError("Tracker frame table does not include per-frame moment values.")
    tracker_moments = pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce")
    valid_frames = frame_table.loc[tracker_moments.notna()]
    if valid_frames.empty:
        raise ValueError("Tracker frame table has no readable per-frame force/moment values.")

    rows: list[dict[str, float | int | str]] = []
    for record in sweep_records:
        optimizer_moment_nm = optimizer_moment_value(record)
        if optimizer_moment_nm is None:
            raise ValueError(
                "Sweep record is missing an applied moment; moment-matched outputs need this for every entry."
            )
        best = _nearest_row_by_value(valid_frames, "tracker_moment_nm", optimizer_moment_nm)
        tracker_force_n = _series_float_value(best, "tracker_force_n")
        tracker_mass_kg = _series_float_value(best, "tracker_mass_kg")
        tracker_moment_nm = _series_float_value(best, "tracker_moment_nm")
        rows.append(
            {
                "sweep_index": int(record["index"]),
                "sweep_name": str(record["name"]),
                "optimizer_rom_deg": abs(float(record["rom_deg"])),
                "optimizer_moment_nm": float(optimizer_moment_nm),
                "optimizer_force_n": float(moment_to_force_n(optimizer_moment_nm)),
                "optimizer_force_kg": float(force_to_mass_kg(moment_to_force_n(optimizer_moment_nm))),
                "matched_frame": int(best["frame"]),
                "matched_frame_in_window": int(best["frame_in_window"]),
                "matched_time_s": float(best["time_s"]),
                "matched_tip_dot_id": int(best["tip_dot_id"]),
                "matched_tracker_visual_angle_deg": float(best["visual_angle_deg"]),
                "matched_tracker_direct_angle_deg": float(best["direct_angle_deg"]),
                "matched_tracker_tip_displacement_mm": float(best["tip_displacement_mm"]),
                "matched_tracker_force_n": float(tracker_force_n) if tracker_force_n is not None else math.nan,
                "matched_tracker_force_kg": float(tracker_mass_kg) if tracker_mass_kg is not None else math.nan,
                "matched_tracker_moment_nm": float(tracker_moment_nm) if tracker_moment_nm is not None else math.nan,
                "matched_tracker_force_confidence": float(best["tracker_force_confidence"])
                if "tracker_force_confidence" in best.index
                else math.nan,
                "abs_moment_error_nm": (
                    float(abs(tracker_moment_nm - optimizer_moment_nm))
                    if tracker_moment_nm is not None
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("sweep_index").reset_index(drop=True)


def build_matches_by_tip_displacement(
    sweep_summary: pd.DataFrame,
    frame_table: pd.DataFrame,
) -> pd.DataFrame:
    tracker_tip = pd.to_numeric(frame_table["tip_displacement_mm"], errors="coerce")
    valid_frames = frame_table.loc[tracker_tip.notna()]
    if valid_frames.empty:
        raise ValueError("Tracker frame table has no readable per-frame tip displacement values.")

    sweep_tip = pd.to_numeric(sweep_summary["optimizer_final_node_ures_mm"], errors="coerce")
    valid_sweeps = sweep_summary.loc[sweep_tip.notna()]
    if valid_sweeps.empty:
        raise ValueError("Sweep summary table has no readable optimizer final-node URES values.")

    rows: list[dict[str, float | int | str]] = []
    for _, sweep_row in valid_sweeps.iterrows():
        optimizer_tip_mm = float(sweep_row["optimizer_final_node_ures_mm"])
        best = _nearest_row_by_value(valid_frames, "tip_displacement_mm", optimizer_tip_mm)
        rows.append(
            {
                "sweep_index": int(sweep_row["sweep_index"]),
                "sweep_name": str(sweep_row["sweep_name"]),
                "optimizer_rom_deg": float(sweep_row["optimizer_rom_deg"]),
                "optimizer_moment_nm": float(sweep_row["optimizer_moment_nm"]),
                "optimizer_final_node_ures_mm": optimizer_tip_mm,
                "matched_frame": int(best["frame"]),
                "matched_frame_in_window": int(best["frame_in_window"]),
                "matched_time_s": float(best["time_s"]),
                "matched_tip_dot_id": int(best["tip_dot_id"]),
                "matched_tracker_visual_angle_deg": float(best["visual_angle_deg"]),
                "matched_tracker_direct_angle_deg": float(best["direct_angle_deg"]),
                "matched_tracker_tip_displacement_mm": float(best["tip_displacement_mm"]),
                "abs_tip_displacement_error_mm": float(
                    abs(float(best["tip_displacement_mm"]) - optimizer_tip_mm)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("sweep_index").reset_index(drop=True)


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    frame_idx = int(frame_idx)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open tracked video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame

        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen tracked video: {video_path}")

        frame = None
        for _ in range(frame_idx + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_idx}.")
        return frame
    finally:
        cap.release()


def video_path_is_readable(video_path: Path | None) -> bool:
    if video_path is None or not video_path.is_file():
        return False
    cap = cv2.VideoCapture(str(video_path))
    try:
        return bool(cap.isOpened())
    finally:
        cap.release()


def video_path_fps(video_path: Path | None) -> float | None:
    if video_path is None or not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return fps if math.isfinite(fps) and fps > 1e-6 else None
    finally:
        cap.release()


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, :3])


def optimizer_torque_value(record: dict) -> float | None:
    for key in ("loading_m", "torque_spec"):
        value = record.get(key)
        if value is None:
            continue
        try:
            torque = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(torque):
            return torque

    spring = record.get("spring")
    if spring is None:
        return None

    loading = getattr(spring, "loading", None)
    if loading is not None:
        loading_values = np.asarray(loading, dtype=float).ravel()
        if loading_values.size >= 3 and math.isfinite(float(loading_values[2])):
            return float(loading_values[2])

    specs = getattr(spring, "specs", None)
    torque = None if specs is None else getattr(specs, "torque", None)
    if torque is None:
        return None
    try:
        torque_value = float(torque)
    except (TypeError, ValueError):
        return None
    return torque_value if math.isfinite(torque_value) else None


def optimizer_moment_value(record: dict) -> float | None:
    torque_value = optimizer_torque_value(record)
    if torque_value is None:
        return None
    return abs(float(torque_value))


def optimizer_key_label(record: dict, base_label: str) -> str:
    moment_nm = optimizer_moment_value(record)
    if moment_nm is None:
        return base_label
    return f"{base_label} (moment={moment_nm:.4f} N*m)"


def _comparison_visual_angle_deg(comparison_frame: pd.Series) -> float | None:
    return _series_float_value(comparison_frame, "visual_angle_deg")


def _comparison_direct_angle_deg(comparison_frame: pd.Series) -> float | None:
    return _series_float_value(comparison_frame, "direct_angle_deg")


def build_overlay_text(
    *,
    record: dict,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    comparison_frame: pd.Series,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    tracker_rows: pd.DataFrame | None = None,
    angle_label: str,
    summary_lines: list[str] | None = None,
    include_force_lines: bool = False,
) -> str:
    optimizer_moment_nm = optimizer_moment_value(record)
    tracker_visual_angle_deg = _comparison_visual_angle_deg(comparison_frame)
    tracker_direct_angle_deg = _comparison_direct_angle_deg(comparison_frame)
    lines = [
        f"Sweep index: {int(record['index']) + 1}",
        f"Optimizer ROM: {abs(float(record['rom_deg'])):.3f} deg",
        f"Tracker frame: {int(comparison_frame['frame'])} (window {int(comparison_frame['frame_in_window'])})",
        f"Tracker time: {float(comparison_frame['time_s']):.3f} s",
        f"Optimizer peak URES: {float(np.max(optimizer_curve_mm)):.3f} mm",
        f"Tracker peak URES: {float(np.max(tracker_curve_mm)):.3f} mm",
        f"Optimizer final node: {float(optimizer_curve_mm[-1]):.3f} mm",
        f"Tracker final node: {float(tracker_curve_mm[-1]):.3f} mm",
    ]
    mean_curve_error_line = format_mean_ures_error_line(
        curve_mean_absolute_error_metrics(
            optimizer_s_norm,
            optimizer_curve_mm,
            tracker_s_norm,
            tracker_curve_mm,
            tracker_rows=tracker_rows,
        )
    )
    if mean_curve_error_line is not None:
        lines.append(mean_curve_error_line)
    match_lines = summary_lines if summary_lines is not None else [
        (
            f"{angle_label}: {tracker_visual_angle_deg:.3f} deg"
            if tracker_visual_angle_deg is not None
            else f"{angle_label}: n/a"
        ),
        (
            f"Angle error: {abs(tracker_visual_angle_deg - abs(float(record['rom_deg']))):.3f} deg"
            if tracker_visual_angle_deg is not None
            else "Angle error: n/a"
        ),
    ]
    if tracker_direct_angle_deg is not None:
        match_lines.append(f"Tracker direct angle metadata: {tracker_direct_angle_deg:.3f} deg")
    if optimizer_moment_nm is not None:
        lines.insert(2, f"Optimizer moment: {_format_moment_value(optimizer_moment_nm)}")
    insertion_idx = 3 if optimizer_moment_nm is not None else 2
    lines[insertion_idx:insertion_idx] = match_lines
    if include_force_lines:
        tracker_force_n = _series_float_value(comparison_frame, "tracker_force_n")
        tracker_moment_nm = _series_float_value(comparison_frame, "tracker_moment_nm")
        force_lines: list[str] = []
        if tracker_moment_nm is not None:
            force_lines.append(f"Tracker moment: {_format_moment_value(tracker_moment_nm)}")
        if optimizer_moment_nm is not None:
            force_lines.append(
                f"Optimizer tangent force: {_format_force_value(moment_to_force_n(optimizer_moment_nm))}"
            )
        if tracker_force_n is not None:
            force_lines.append(f"Tracker tangent force: {_format_force_value(tracker_force_n)}")
        lines[insertion_idx + len(match_lines):insertion_idx + len(match_lines)] = force_lines
    return "\n".join(lines)


def build_moment_match_summary_lines(record: dict, comparison_frame: pd.Series) -> list[str]:
    optimizer_moment_nm = optimizer_moment_value(record)
    tracker_moment_nm = _series_float_value(comparison_frame, "tracker_moment_nm")
    tracker_visual_angle_deg = _comparison_visual_angle_deg(comparison_frame)
    tracker_direct_angle_deg = _comparison_direct_angle_deg(comparison_frame)
    lines = [
        (
            f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {tracker_visual_angle_deg:.3f} deg"
            if tracker_visual_angle_deg is not None
            else f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: n/a"
        )
    ]
    if optimizer_moment_nm is not None and tracker_moment_nm is not None:
        lines.insert(0, f"Moment error: {abs(tracker_moment_nm - optimizer_moment_nm):.4f} N*m")
    if tracker_direct_angle_deg is not None:
        lines.append(f"Tracker direct angle metadata: {tracker_direct_angle_deg:.3f} deg")
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=optimizer_moment_nm,
            optimizer_angle_deg=abs(float(record["rom_deg"])),
            tracker_moment_nm=tracker_moment_nm,
            tracker_angle_deg=tracker_visual_angle_deg,
        )
    )
    return lines


def build_tip_displacement_match_summary_lines(record: dict, comparison_frame: pd.Series) -> list[str]:
    optimizer_curve_mm, _ = spring_ures_curve_mm(record["spring"])
    optimizer_tip_mm = float(optimizer_curve_mm[-1]) if optimizer_curve_mm.size else math.nan
    tracker_tip_mm = _series_float_value(comparison_frame, "tip_displacement_mm")
    tracker_visual_angle_deg = _comparison_visual_angle_deg(comparison_frame)
    tracker_direct_angle_deg = _comparison_direct_angle_deg(comparison_frame)
    lines = [
        f"Target final-node URES: {optimizer_tip_mm:.3f} mm",
        (
            f"Tracker final-node URES: {tracker_tip_mm:.3f} mm"
            if tracker_tip_mm is not None
            else "Tracker final-node URES: n/a"
        ),
    ]
    if tracker_tip_mm is not None:
        lines.append(f"URES error: {abs(tracker_tip_mm - optimizer_tip_mm):.3f} mm")
    lines.append(
        (
            f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {tracker_visual_angle_deg:.3f} deg"
            if tracker_visual_angle_deg is not None
            else f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: n/a"
        )
    )
    if tracker_direct_angle_deg is not None:
        lines.append(f"Tracker direct angle metadata: {tracker_direct_angle_deg:.3f} deg")
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=optimizer_moment_value(record),
            optimizer_angle_deg=abs(float(record["rom_deg"])),
            tracker_moment_nm=_series_float_value(comparison_frame, "tracker_moment_nm"),
            tracker_angle_deg=tracker_visual_angle_deg,
        )
    )
    return lines


def build_target_angle_summary_lines(comparison_frame: pd.Series, target_angle_deg: float) -> list[str]:
    tracker_visual_angle_deg = _comparison_visual_angle_deg(comparison_frame)
    tracker_direct_angle_deg = _comparison_direct_angle_deg(comparison_frame)
    return [
        f"Target arbor rotation: {float(target_angle_deg):.3f} deg",
        (
            f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {tracker_visual_angle_deg:.3f} deg"
            if tracker_visual_angle_deg is not None
            else f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: n/a"
        ),
        (
            f"Angle error to target: {abs(tracker_visual_angle_deg - float(target_angle_deg)):.3f} deg"
            if tracker_visual_angle_deg is not None
            else "Angle error to target: n/a"
        ),
        (
            f"Tracker direct angle metadata: {tracker_direct_angle_deg:.3f} deg"
            if tracker_direct_angle_deg is not None
            else "Tracker direct angle metadata: n/a"
        ),
    ]


def build_match_context(
    match: pd.Series,
    sweep_records: list[dict],
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
) -> dict[str, object]:
    record = sweep_records[int(match["sweep_index"])]
    spring = record["spring"]
    optimizer_curve_mm, optimizer_s_norm = spring_ures_curve_mm(spring)

    frame_idx = int(match["matched_frame"])
    tracker_rows = tracker_df[tracker_df["frame"] == frame_idx].sort_values("dot_id")
    if tracker_rows.empty:
        raise ValueError(f"Tracker frame {frame_idx} has no rows.")
    tracker_s_norm, tracker_curve_mm = camera_ures_curve(tracker_rows, "arc_length_mm", "displacement_mm")
    comparison_frame = frame_table[frame_table["frame"] == frame_idx].iloc[0]

    return {
        "match": match.copy(),
        "record": record,
        "comparison_frame": comparison_frame,
        "tracker_rows": tracker_rows.copy(),
        "optimizer_curve_mm": optimizer_curve_mm,
        "optimizer_s_norm": optimizer_s_norm,
        "tracker_curve_mm": tracker_curve_mm,
        "tracker_s_norm": tracker_s_norm,
        "tracker_peak_ures_mm": float(np.max(tracker_curve_mm)),
    }


def build_sweep_summary_table(sweep_records: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for record in sweep_records:
        optimizer_curve_mm, optimizer_s_norm = spring_ures_curve_mm(record["spring"])
        optimizer_moment_nm = optimizer_moment_value(record)
        optimizer_force_n = moment_to_force_n(optimizer_moment_nm) if optimizer_moment_nm is not None else math.nan
        rows.append(
            {
                "sweep_index": int(record["index"]),
                "sweep_name": str(record["name"]),
                "optimizer_rom_deg": abs(float(record["rom_deg"])),
                "optimizer_moment_nm": float(optimizer_moment_nm) if optimizer_moment_nm is not None else math.nan,
                "optimizer_force_n": float(optimizer_force_n) if math.isfinite(float(optimizer_force_n)) else math.nan,
                "optimizer_force_kg": (
                    float(force_to_mass_kg(optimizer_force_n)) if math.isfinite(float(optimizer_force_n)) else math.nan
                ),
                "optimizer_final_node_ures_mm": float(optimizer_curve_mm[-1]),
                "optimizer_peak_ures_mm": float(np.max(optimizer_curve_mm)),
                "optimizer_curve_node_count": int(len(optimizer_curve_mm)),
                "optimizer_arc_node_count": int(len(optimizer_s_norm)),
            }
        )
    return pd.DataFrame(rows).sort_values("sweep_index").reset_index(drop=True)


def _match_tolerance_for_sorted_values(sorted_values: np.ndarray, idx: int) -> float:
    values = np.asarray(sorted_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return math.inf
    gaps: list[float] = []
    if idx > 0:
        gaps.append(abs(float(values[idx]) - float(values[idx - 1])))
    if idx + 1 < values.size:
        gaps.append(abs(float(values[idx + 1]) - float(values[idx])))
    finite_positive = [gap for gap in gaps if math.isfinite(gap) and gap > 1e-12]
    if not finite_positive:
        return math.inf
    return 0.5 * min(finite_positive) + 1e-9


def build_tracker_to_sweep_matches(
    frame_table: pd.DataFrame,
    sweep_table: pd.DataFrame,
    *,
    tracker_value_col: str,
    sweep_value_col: str,
    error_col: str,
) -> pd.DataFrame:
    tracker_values = pd.to_numeric(frame_table[tracker_value_col], errors="coerce")
    valid_frames = frame_table.loc[tracker_values.notna()].copy()
    if valid_frames.empty:
        return pd.DataFrame()

    sweep_values = pd.to_numeric(sweep_table[sweep_value_col], errors="coerce")
    valid_sweeps = sweep_table.loc[sweep_values.notna()].copy()
    if valid_sweeps.empty:
        return pd.DataFrame()
    valid_sweeps = valid_sweeps.sort_values(sweep_value_col).reset_index(drop=True)
    sweep_value_array = pd.to_numeric(valid_sweeps[sweep_value_col], errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, float | int | str | bool]] = []
    for _, frame in valid_frames.iterrows():
        tracker_value = float(frame[tracker_value_col])
        idx = int(np.argmin(np.abs(sweep_value_array - tracker_value)))
        matched = valid_sweeps.iloc[idx]
        error_value = abs(tracker_value - float(matched[sweep_value_col]))
        tolerance = _match_tolerance_for_sorted_values(sweep_value_array, idx)
        rows.append(
            {
                "frame": int(frame["frame"]),
                "frame_in_window": int(frame["frame_in_window"]),
                "time_s": float(frame["time_s"]),
                "tracker_match_value": float(tracker_value),
                "tracker_visual_angle_deg": (
                    float(frame["visual_angle_deg"]) if "visual_angle_deg" in frame.index else math.nan
                ),
                "tracker_direct_angle_deg": float(frame["direct_angle_deg"]),
                "tracker_tip_displacement_mm": float(frame["tip_displacement_mm"]),
                "tracker_force_n": (
                    float(frame["tracker_force_n"]) if "tracker_force_n" in frame.index else math.nan
                ),
                "tracker_moment_nm": (
                    float(frame["tracker_moment_nm"]) if "tracker_moment_nm" in frame.index else math.nan
                ),
                "matched_sweep_index": int(matched["sweep_index"]),
                "matched_sweep_name": str(matched["sweep_name"]),
                "matched_sweep_value": float(matched[sweep_value_col]),
                "matched_optimizer_rom_deg": float(matched["optimizer_rom_deg"]),
                "matched_optimizer_moment_nm": float(matched["optimizer_moment_nm"]),
                "matched_optimizer_final_node_ures_mm": float(matched["optimizer_final_node_ures_mm"]),
                error_col: float(error_value),
                "match_tolerance": float(tolerance),
                "within_tolerance": bool(error_value <= tolerance),
            }
        )
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def choose_largest_close_tracker_match(
    matches: pd.DataFrame,
    *,
    tracker_value_col: str,
    error_col: str,
) -> pd.Series | None:
    if matches.empty:
        return None
    close_matches = matches[matches["within_tolerance"]].copy()
    candidate_pool = close_matches if not close_matches.empty else matches.copy()
    ordered = candidate_pool.sort_values(
        by=[tracker_value_col, error_col, "frame"],
        ascending=[False, True, False],
    )
    if ordered.empty:
        return None
    return ordered.iloc[0].copy()


def build_selected_match_context(
    selected_match: pd.Series,
    sweep_records: list[dict],
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
) -> dict[str, object]:
    return build_match_context(
        pd.Series(
            {
                "sweep_index": int(selected_match["matched_sweep_index"]),
                "matched_frame": int(selected_match["frame"]),
            }
        ),
        sweep_records,
        tracker_df,
        frame_table,
    )


def build_record_frame_context(
    record: dict,
    comparison_frame: pd.Series,
    tracker_df: pd.DataFrame,
) -> dict[str, object]:
    spring = record["spring"]
    optimizer_curve_mm, optimizer_s_norm = spring_ures_curve_mm(spring)

    frame_idx = int(comparison_frame["frame"])
    tracker_rows = tracker_df[tracker_df["frame"] == frame_idx].sort_values("dot_id")
    if tracker_rows.empty:
        raise ValueError(f"Tracker frame {frame_idx} has no rows.")
    tracker_s_norm, tracker_curve_mm = camera_ures_curve(tracker_rows, "arc_length_mm", "displacement_mm")

    return {
        "record": record,
        "comparison_frame": comparison_frame.copy(),
        "tracker_rows": tracker_rows.copy(),
        "optimizer_curve_mm": optimizer_curve_mm,
        "optimizer_s_norm": optimizer_s_norm,
        "tracker_curve_mm": tracker_curve_mm,
        "tracker_s_norm": tracker_s_norm,
        "tracker_peak_ures_mm": float(np.max(tracker_curve_mm)),
    }


def build_final_sweep_angle_summary_lines(match: pd.Series) -> list[str]:
    lines = [
        "Selection: tracker frame closest to final sweep ROM by arbor rotation",
        f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {float(match['matched_tracker_visual_angle_deg']):.3f} deg",
        f"Final sweep ROM: {float(match['optimizer_rom_deg']):.3f} deg",
        f"Angle error: {float(match['abs_angle_error_deg']):.3f} deg",
        f"Tracker direct angle metadata: {float(match['matched_tracker_direct_angle_deg']):.3f} deg",
    ]
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=_series_float_value(match, "matched_optimizer_moment_nm"),
            optimizer_angle_deg=_series_float_value(match, "optimizer_rom_deg"),
            tracker_moment_nm=_series_float_value(match, "matched_tracker_moment_nm"),
            tracker_angle_deg=_series_float_value(match, "matched_tracker_visual_angle_deg"),
        )
    )
    return lines


def build_largest_close_moment_summary_lines(selected_match: pd.Series) -> list[str]:
    lines = [
        "Selection: largest tracker moment with close sweep match",
        f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {float(selected_match['tracker_visual_angle_deg']):.3f} deg",
        f"Tracker moment: {_format_moment_value(float(selected_match['tracker_moment_nm']))}",
        f"Moment error: {float(selected_match['abs_moment_error_nm']):.4f} N*m",
        f"Tracker direct angle metadata: {float(selected_match['tracker_direct_angle_deg']):.3f} deg",
    ]
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=_series_float_value(selected_match, "matched_optimizer_moment_nm"),
            optimizer_angle_deg=_series_float_value(selected_match, "matched_optimizer_rom_deg"),
            tracker_moment_nm=_series_float_value(selected_match, "tracker_moment_nm"),
            tracker_angle_deg=_series_float_value(selected_match, "tracker_visual_angle_deg"),
        )
    )
    return lines


def build_largest_close_angle_summary_lines(selected_match: pd.Series) -> list[str]:
    lines = [
        "Selection: largest tracker arbor rotation with close sweep match",
        f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {float(selected_match['tracker_visual_angle_deg']):.3f} deg",
        f"Optimizer ROM: {float(selected_match['matched_optimizer_rom_deg']):.3f} deg",
        f"Angle error: {float(selected_match['abs_angle_error_deg']):.3f} deg",
        f"Tracker direct angle metadata: {float(selected_match['tracker_direct_angle_deg']):.3f} deg",
    ]
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=_series_float_value(selected_match, "matched_optimizer_moment_nm"),
            optimizer_angle_deg=_series_float_value(selected_match, "matched_optimizer_rom_deg"),
            tracker_moment_nm=_series_float_value(selected_match, "tracker_moment_nm"),
            tracker_angle_deg=_series_float_value(selected_match, "tracker_visual_angle_deg"),
        )
    )
    return lines


def build_compact_largest_close_angle_lines(selected_match: pd.Series) -> list[str]:
    lines = [
        f"{ANGLE_MEASUREMENT_LABEL}: {float(selected_match['tracker_visual_angle_deg']):.3f} deg",
        f"Sweep ROM: {float(selected_match['matched_optimizer_rom_deg']):.3f} deg",
        f"Angle error: {float(selected_match['abs_angle_error_deg']):.3f} deg",
    ]
    sweep_reference_stiffness = _series_float_value(
        selected_match,
        "sweep_reference_stiffness_nm_per_rad",
    )
    tracker_reference_stiffness = _series_float_value(
        selected_match,
        "tracker_reference_stiffness_nm_per_rad",
    )
    if sweep_reference_stiffness is None:
        sweep_reference_stiffness = rotational_stiffness_nm_per_rad(1.0, 90.0)
    if sweep_reference_stiffness is not None:
        lines.append(
            f"k sweep (1 N*m / 90 deg): {_format_stiffness_value(sweep_reference_stiffness)}"
        )
    if tracker_reference_stiffness is not None:
        lines.append(
            f"k track (near 1 N*m): {_format_stiffness_value(tracker_reference_stiffness)}"
        )
    if (
        sweep_reference_stiffness is not None
        and tracker_reference_stiffness is not None
        and abs(sweep_reference_stiffness) > 1e-12
    ):
        stiffness_error_pct = 100.0 * (
            tracker_reference_stiffness - sweep_reference_stiffness
        ) / sweep_reference_stiffness
        lines.append(f"Stiffness error: {stiffness_error_pct:+.1f}%")
    return lines


def build_compact_target_moment_lines(
    selected_match: pd.Series,
    *,
    target_moment_nm: float,
) -> list[str]:
    tracker_moment_nm = _series_float_value(selected_match, "tracker_moment_nm")
    optimizer_moment_nm = _series_float_value(selected_match, "matched_optimizer_moment_nm")
    tracker_visual_angle_deg = _series_float_value(selected_match, "tracker_visual_angle_deg")
    lines = [f"Target moment: {_format_moment_value(target_moment_nm)}"]
    if tracker_moment_nm is not None:
        lines.append(f"Tracker moment: {_format_moment_value(tracker_moment_nm)}")
        lines.append(f"Moment error: {abs(tracker_moment_nm - target_moment_nm):.4f} N*m")
    if optimizer_moment_nm is not None:
        lines.append(f"Sweep moment: {_format_moment_value(optimizer_moment_nm)}")
    if tracker_visual_angle_deg is not None:
        lines.append(f"{ANGLE_MEASUREMENT_LABEL}: {tracker_visual_angle_deg:.3f} deg")
    metrics = stiffness_metrics(
        optimizer_moment_nm=optimizer_moment_nm,
        optimizer_angle_deg=_series_float_value(selected_match, "matched_optimizer_rom_deg"),
        tracker_moment_nm=tracker_moment_nm,
        tracker_angle_deg=tracker_visual_angle_deg,
    )
    if metrics["optimizer_stiffness_nm_per_rad"] is not None:
        lines.append(
            f"k sweep: {_format_stiffness_value(metrics['optimizer_stiffness_nm_per_rad'])}"
        )
    if metrics["tracker_stiffness_nm_per_rad"] is not None:
        lines.append(
            f"k track: {_format_stiffness_value(metrics['tracker_stiffness_nm_per_rad'])}"
        )
    stiffness_error_pct = metrics["stiffness_error_pct"]
    if stiffness_error_pct is not None and math.isfinite(float(stiffness_error_pct)):
        lines.append(f"Stiffness error: {stiffness_error_pct:+.1f}%")
    return lines


def build_final_sweep_final_node_ures_summary_lines(match: pd.Series) -> list[str]:
    lines = [
        "Selection: tracker frame closest to final sweep final-node URES",
        f"Target final-node URES: {float(match['matched_optimizer_final_node_ures_mm']):.3f} mm",
        f"Tracker final-node URES: {float(match['tracker_tip_displacement_mm']):.3f} mm",
        f"Final-node URES error: {float(match['abs_final_node_ures_error_mm']):.3f} mm",
        f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}: {float(match['tracker_visual_angle_deg']):.3f} deg",
        f"Tracker direct angle metadata: {float(match['tracker_direct_angle_deg']):.3f} deg",
    ]
    lines.extend(
        build_stiffness_summary_lines(
            optimizer_moment_nm=_series_float_value(match, "matched_optimizer_moment_nm"),
            optimizer_angle_deg=_series_float_value(match, "matched_optimizer_rom_deg"),
            tracker_moment_nm=_series_float_value(match, "tracker_moment_nm"),
            tracker_angle_deg=_series_float_value(match, "tracker_visual_angle_deg"),
        )
    )
    return lines


def build_compact_final_sweep_final_node_ures_lines(match: pd.Series) -> list[str]:
    lines = [
        f"Optimizer angle: {float(match['matched_optimizer_rom_deg']):.3f} deg",
        f"{ANGLE_MEASUREMENT_LABEL}: {float(match['tracker_visual_angle_deg']):.3f} deg",
    ]
    metrics = stiffness_metrics(
        optimizer_moment_nm=_series_float_value(match, "matched_optimizer_moment_nm"),
        optimizer_angle_deg=_series_float_value(match, "matched_optimizer_rom_deg"),
        tracker_moment_nm=_series_float_value(match, "tracker_moment_nm"),
        tracker_angle_deg=_series_float_value(match, "tracker_visual_angle_deg"),
    )
    if metrics["optimizer_stiffness_nm_per_rad"] is not None:
        lines.append(
            f"k sweep: {_format_stiffness_value(metrics['optimizer_stiffness_nm_per_rad'])}"
        )
    if metrics["tracker_stiffness_nm_per_rad"] is not None:
        lines.append(
            f"k track: {_format_stiffness_value(metrics['tracker_stiffness_nm_per_rad'])}"
        )
    stiffness_error_pct = metrics["stiffness_error_pct"]
    if stiffness_error_pct is not None and math.isfinite(float(stiffness_error_pct)):
        lines.append(f"Stiffness error: {stiffness_error_pct:+.1f}%")
    return lines


def choose_reference_frame(frame_table: pd.DataFrame, *, mode: str) -> pd.Series:
    if mode == "angle":
        table = frame_table.assign(abs_visual_angle_deg=frame_table["visual_angle_deg"].abs())
        return _nearest_row_by_value(table, "abs_visual_angle_deg", 0.0)
    if mode == "moment":
        if "tracker_moment_nm" in frame_table.columns:
            moment_values = pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce")
            valid = frame_table.loc[moment_values.notna()].copy()
            if not valid.empty:
                valid = valid.assign(abs_tracker_moment_nm=valid["tracker_moment_nm"].abs())
                return _nearest_row_by_value(valid, "abs_tracker_moment_nm", 0.0)
        table = frame_table.assign(abs_visual_angle_deg=frame_table["visual_angle_deg"].abs())
        return _nearest_row_by_value(table, "abs_visual_angle_deg", 0.0)
    raise ValueError(f"Unknown reference-frame mode: {mode}")


def tracker_arbor_center_px(rows: pd.DataFrame, image_shape: tuple[int, ...]) -> np.ndarray:
    if {"arbor_center_x_raw", "arbor_center_y_raw"}.issubset(rows.columns):
        centers = rows[["arbor_center_x_raw", "arbor_center_y_raw"]].dropna()
        if not centers.empty:
            return centers.mean(axis=0).to_numpy(dtype=float)
    height, width = image_shape[:2]
    return np.array([0.5 * (width - 1), 0.5 * (height - 1)], dtype=float)


def tracker_panel_points_px(rows: pd.DataFrame) -> np.ndarray:
    points: list[np.ndarray] = []
    for x_col, y_col in (("x_ref", "y_ref"), ("x_cur", "y_cur")):
        if {x_col, y_col}.issubset(rows.columns):
            xy = rows[[x_col, y_col]].dropna().to_numpy(dtype=float)
            if xy.size:
                points.append(xy)
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(points)


def apriltag_panel_points_px(frame_bgr: np.ndarray) -> np.ndarray:
    detections = detect_apriltags(frame_bgr)
    if not detections:
        return np.empty((0, 2), dtype=float)
    return np.vstack([np.asarray(corners, dtype=float) for _, corners in sorted(detections.items())])


def apriltag_points_from_detections(detections: dict[int, np.ndarray] | None) -> np.ndarray:
    if not detections:
        return np.empty((0, 2), dtype=float)
    return np.vstack([np.asarray(corners, dtype=float) for _, corners in sorted(detections.items())])


def tracker_reference_points_px(rows: pd.DataFrame) -> np.ndarray:
    if not {"x_ref", "y_ref"}.issubset(rows.columns):
        return np.empty((0, 2), dtype=float)
    return rows.sort_values("dot_id")[["x_ref", "y_ref"]].dropna().to_numpy(dtype=float)


def tracker_reference_points_plane_mm(rows: pd.DataFrame) -> np.ndarray:
    columns = {"x_ref_plane_mm", "y_ref_plane_mm"}
    if not columns.issubset(rows.columns):
        return np.empty((0, 2), dtype=float)
    points = rows.sort_values("dot_id")[["x_ref_plane_mm", "y_ref_plane_mm"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def tracker_red_centroid_px(rows: pd.DataFrame) -> np.ndarray | None:
    columns = {"red_centroid_x_raw", "red_centroid_y_raw"}
    if not columns.issubset(rows.columns):
        return None
    centroids = rows[["red_centroid_x_raw", "red_centroid_y_raw"]].dropna().to_numpy(dtype=float)
    if not len(centroids):
        return None
    return np.mean(centroids, axis=0)


def tracker_vector_origin_px(rows: pd.DataFrame, image_shape: tuple[int, ...]) -> np.ndarray:
    marker_xy = tracker_red_centroid_px(rows)
    if marker_xy is not None:
        return marker_xy
    return tracker_arbor_center_px(rows, image_shape)


def _point_radius_px(point_xy: np.ndarray, center_xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(point_xy, dtype=float) - np.asarray(center_xy, dtype=float)))


def _series_point_xy(row: pd.Series, x_col: str, y_col: str) -> np.ndarray | None:
    if x_col not in row.index or y_col not in row.index:
        return None
    try:
        x_val = float(row[x_col])
        y_val = float(row[y_col])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x_val) and math.isfinite(y_val)):
        return None
    return np.array([x_val, y_val], dtype=float)


def tracker_first_node_row(rows: pd.DataFrame, preferred_dot_id: int | None = None) -> pd.Series:
    if rows.empty:
        raise ValueError("Tracker frame has no rows.")
    if preferred_dot_id is None and "arbor_rigid_dot_id" in rows.columns:
        rigid_dot_id = _first_finite_numeric(rows["arbor_rigid_dot_id"])
        if rigid_dot_id is not None:
            preferred_dot_id = int(rigid_dot_id)
    if preferred_dot_id is not None and "dot_id" in rows.columns:
        preferred = rows[rows["dot_id"] == int(preferred_dot_id)]
        if not preferred.empty:
            for _, row in preferred.iterrows():
                reference_xy = _series_point_xy(row, "x_ref", "y_ref")
                current_xy = _series_point_xy(row, "x_cur", "y_cur")
                if reference_xy is not None or current_xy is not None:
                    return row.copy()
    order_cols: list[str] = []
    ascending: list[bool] = []
    if "dot_id" in rows.columns:
        order_cols.append("dot_id")
        ascending.append(False)
    elif "arc_length_mm" in rows.columns:
        order_cols.append("arc_length_mm")
        ascending.append(False)
    elif "arc_length_px" in rows.columns:
        order_cols.append("arc_length_px")
        ascending.append(False)
    ordered_rows = rows.sort_values(order_cols, ascending=ascending) if order_cols else rows
    for _, row in ordered_rows.iterrows():
        reference_xy = _series_point_xy(row, "x_ref", "y_ref")
        current_xy = _series_point_xy(row, "x_cur", "y_cur")
        if reference_xy is not None or current_xy is not None:
            return row.copy()
    frame_idx = int(rows["frame"].iloc[0]) if "frame" in rows.columns else -1
    raise ValueError(f"Tracker frame {frame_idx} is missing usable first-node coordinates.")


def tracker_first_node_vectors_px(
    rows: pd.DataFrame,
    preferred_dot_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    arbor_xy = tracker_vector_origin_px(rows, (1, 1, 3))
    row = tracker_first_node_row(rows, preferred_dot_id=preferred_dot_id)
    reference_xy = _series_point_xy(row, "x_ref", "y_ref")
    current_xy = _series_point_xy(row, "x_cur", "y_cur")
    if reference_xy is None:
        if current_xy is None:
            frame_idx = int(rows["frame"].iloc[0]) if "frame" in rows.columns else -1
            raise ValueError(f"Tracker frame {frame_idx} is missing usable first-node coordinates.")
        reference_xy = current_xy.copy()
    return arbor_xy, reference_xy, current_xy


def tracker_vector_crop_half_extent_px(
    rows: pd.DataFrame,
    *,
    include_deformed_vector: bool,
    preferred_dot_id: int | None = None,
) -> int:
    arbor_xy, reference_xy, current_xy = tracker_first_node_vectors_px(rows, preferred_dot_id=preferred_dot_id)
    relevant_points = [reference_xy]
    if include_deformed_vector and current_xy is not None:
        relevant_points.append(current_xy)
    radii_px = [float(np.linalg.norm(point_xy - arbor_xy)) for point_xy in relevant_points]
    if not radii_px:
        return VECTOR_SNAPSHOT_MIN_HALF_EXTENT_PX
    max_radius_px = max(radii_px)
    return max(VECTOR_SNAPSHOT_MIN_HALF_EXTENT_PX, int(math.ceil(max_radius_px + VECTOR_SNAPSHOT_MARGIN_PX)))


def vector_angle_deg(start_xy: np.ndarray, end_xy: np.ndarray) -> float | None:
    vec = np.asarray(end_xy, dtype=float) - np.asarray(start_xy, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return float(math.degrees(math.atan2(vec[1], vec[0])))


def included_vector_angle_deg(
    origin_xy: np.ndarray,
    first_xy: np.ndarray | None,
    second_xy: np.ndarray | None,
) -> float | None:
    if first_xy is None or second_xy is None:
        return None
    vec_a = np.asarray(first_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    vec_b = np.asarray(second_xy, dtype=float) - np.asarray(origin_xy, dtype=float)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None
    cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.degrees(math.acos(cos_theta)))


def build_snapshot_angle_annotation_lines(
    comparison_frame: pd.Series,
    tracker_rows: pd.DataFrame,
    preferred_dot_id: int | None = None,
) -> list[str]:
    tracker_row = tracker_first_node_row(tracker_rows, preferred_dot_id=preferred_dot_id)
    direct_angle_deg = float(tracker_row["angle_change_deg"])
    arbor_rotation_deg = _series_float_value(comparison_frame, "arbor_rotation_deg")
    if arbor_rotation_deg is None:
        arbor_rotation_deg = _series_float_value(comparison_frame, "visual_angle_deg")
    arbor_xy, reference_xy, current_xy = tracker_first_node_vectors_px(
        tracker_rows,
        preferred_dot_id=preferred_dot_id,
    )
    image_angle_deg = included_vector_angle_deg(arbor_xy, reference_xy, current_xy)
    lines = [
        (
            f"{ANGLE_IMAGE_DIAGNOSTIC_LABEL}: {image_angle_deg:.3f} deg"
            if image_angle_deg is not None
            else f"{ANGLE_IMAGE_DIAGNOSTIC_LABEL}: n/a"
        ),
        (
            f"{ANGLE_MEASUREMENT_LABEL}: {arbor_rotation_deg:.3f} deg"
            if arbor_rotation_deg is not None
            else f"{ANGLE_MEASUREMENT_LABEL}: n/a"
        ),
        f"Endpoint direct-angle metadata: {direct_angle_deg:.3f} deg",
    ]
    return lines


def build_node_vector_history_annotation_lines(
    tracker_row: pd.Series,
    tracker_rows: pd.DataFrame,
    *,
    preferred_dot_id: int,
) -> list[str]:
    direct_angle_deg = float(tracker_row["angle_change_deg"])
    arbor_rotation_deg = _series_float_value(tracker_row, "arbor_rotation_deg")
    if arbor_rotation_deg is None:
        arbor_rotation_deg = _series_float_value(tracker_row, "visual_angle_deg")
    image_angle_deg = included_vector_angle_deg(*tracker_first_node_vectors_px(
        tracker_rows,
        preferred_dot_id=preferred_dot_id,
    ))
    return [
        f"Frame {int(tracker_row['frame'])}  |  t = {float(tracker_row['time_s']):.3f} s",
        (
            f"{ANGLE_IMAGE_DIAGNOSTIC_LABEL}: n/a"
            if image_angle_deg is None
            else f"{ANGLE_IMAGE_DIAGNOSTIC_LABEL}: {image_angle_deg:.3f} deg"
        ),
        (
            f"{ANGLE_MEASUREMENT_LABEL}: {arbor_rotation_deg:.3f} deg"
            if arbor_rotation_deg is not None
            else f"{ANGLE_MEASUREMENT_LABEL}: n/a"
        ),
        f"Endpoint direct-angle metadata: {direct_angle_deg:.3f} deg",
    ]


def sample_spiral_points(
    center: np.ndarray,
    a: float,
    b: float,
    thetas: np.ndarray,
    *,
    n_pts: int = 600,
) -> np.ndarray:
    th_range = np.linspace(float(np.min(thetas)), float(np.max(thetas)), n_pts)
    r_range = a + b * th_range
    return np.column_stack(
        (
            center[0] + r_range * np.cos(th_range),
            center[1] + r_range * np.sin(th_range),
        )
    )


def draw_polyline_overlay(
    frame_bgr: np.ndarray,
    points_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    outline_thickness: int | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    points = np.asarray(points_xy, dtype=float)
    if len(points) < 2:
        return out

    rounded = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    if outline_color is not None:
        cv2.polylines(
            out,
            [rounded],
            False,
            outline_color,
            outline_thickness if outline_thickness is not None else thickness + 2,
            cv2.LINE_AA,
        )
    cv2.polylines(out, [rounded], False, color, thickness, cv2.LINE_AA)
    return out


def fit_snapshot_spiral_profile_px(
    rows: pd.DataFrame,
    image_shape: tuple[int, ...],
    reference_pose=None,
) -> np.ndarray:
    ordered = rows.sort_values("dot_id")
    if reference_pose is not None:
        plane_points = tracker_reference_points_plane_mm(ordered)
        if len(plane_points) >= 2:
            center_px = tracker_arbor_center_px(ordered, image_shape)
            center_plane = image_points_to_plane(np.asarray([center_px], dtype=float), reference_pose)[0]
            a, b, thetas = fit_spiral(center_plane, plane_points)
            spiral_plane = sample_spiral_points(center_plane, a, b, thetas)
            return np.asarray(plane_points_to_image(spiral_plane, reference_pose), dtype=float)

    pixel_points = tracker_reference_points_px(ordered)
    if len(pixel_points) < 2:
        return np.empty((0, 2), dtype=float)
    center_px = tracker_arbor_center_px(ordered, image_shape)
    a, b, thetas = fit_spiral(center_px, pixel_points)
    return sample_spiral_points(center_px, a, b, thetas)


def draw_tracker_marker_overlay(frame_bgr: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    out = frame_bgr.copy()
    ordered = rows.sort_values("dot_id")
    points = tracker_reference_points_px(ordered)
    if not len(points):
        return out

    cmap = matplotlib.colormaps["rainbow"].resampled(max(len(points), 2))
    dot_ids = ordered["dot_id"].to_numpy(dtype=int)
    for idx, ((x_pos, y_pos), dot_id) in enumerate(zip(points, dot_ids)):
        color_rgb = np.asarray(cmap(idx)[:3]) * 255.0
        color_bgr = tuple(int(v) for v in color_rgb[::-1])
        center = (int(round(float(x_pos))), int(round(float(y_pos))))
        cv2.circle(out, center, SNAPSHOT_MARKER_RADIUS_PX + 2, SNAPSHOT_MARKER_OUTLINE_COLOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(out, center, SNAPSHOT_MARKER_RADIUS_PX, color_bgr, -1, cv2.LINE_AA)
        label_anchor = (center[0] + SNAPSHOT_MARKER_RADIUS_PX + 5, center[1] - SNAPSHOT_MARKER_RADIUS_PX + 2)
        cv2.putText(
            out,
            str(dot_id),
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            SNAPSHOT_MARKER_LABEL_SCALE,
            (24, 24, 24),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(dot_id),
            label_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            SNAPSHOT_MARKER_LABEL_SCALE,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_arbor_marker_overlay(frame_bgr: np.ndarray, marker_xy: np.ndarray | None) -> np.ndarray:
    out = frame_bgr.copy()
    if marker_xy is None or not np.all(np.isfinite(marker_xy)):
        return out

    center = tuple(np.round(np.asarray(marker_xy, dtype=float)).astype(int))
    cv2.drawMarker(out, center, ARBOR_MARKER_OUTLINE_COLOR_BGR, cv2.MARKER_CROSS, 24, 5, cv2.LINE_AA)
    cv2.drawMarker(out, center, ARBOR_MARKER_COLOR_BGR, cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
    return out


def _format_distortion_value(value_px: float) -> str:
    magnitude = abs(float(value_px))
    if magnitude >= 10.0:
        return f"{value_px:.1f} px"
    return f"{value_px:.2f} px"


def draw_distortion_key(
    ax,
    *,
    min_distortion_px: float | None,
    max_distortion_px: float | None,
    label: str = "camera distortion",
) -> None:
    bar_width = DISTORTION_KEY_BAR_WIDTH_FRAC
    bar_left = 0.5 * (1.0 - bar_width)
    bar_right = bar_left + bar_width
    bar_height = DISTORTION_KEY_BAR_HEIGHT_FRAC
    bar_bottom = 0.38
    bar_top = bar_bottom + bar_height

    gradient = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    start_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    end_rgb = np.asarray(DISTORTION_KEY_COLOR_BGR[::-1], dtype=np.float32) / 255.0
    bar_rgb = start_rgb[None, :] * (1.0 - gradient[:, None]) + end_rgb[None, :] * gradient[:, None]
    ax.imshow(
        bar_rgb[None, :, :],
        extent=(bar_left, bar_right, bar_bottom, bar_top),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=1,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (bar_left, bar_bottom),
            bar_width,
            bar_height,
            fill=False,
            edgecolor=(0.6, 0.6, 0.6),
            linewidth=0.8,
            zorder=2,
        )
    )

    if min_distortion_px is not None and max_distortion_px is not None:
        tick_kwargs = dict(
            fontsize=DISTORTION_KEY_TICK_POINTSIZE,
            color=(0.12, 0.12, 0.12),
            va="top",
        )
        ax.text(
            bar_left,
            0.97,
            _format_distortion_value(min_distortion_px),
            ha="left",
            **tick_kwargs,
        )
        ax.text(
            bar_right,
            0.97,
            _format_distortion_value(max_distortion_px),
            ha="right",
            **tick_kwargs,
        )

    ax.text(
        0.5,
        0.06,
        label,
        ha="center",
        va="bottom",
        fontsize=DISTORTION_KEY_LABEL_POINTSIZE,
        color=(0.12, 0.12, 0.12),
    )


def draw_apriltag_scale_overlay(
    frame_bgr: np.ndarray,
    detections: dict[int, np.ndarray],
    labels: dict[int, str] | None = None,
    *,
    color: tuple[int, int, int] = APRILTAG_OVERLAY_COLOR_BGR,
) -> np.ndarray:
    out = frame_bgr.copy()
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
            (anchor[0] + 8, anchor[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            APRILTAG_LABEL_FONT_SCALE,
            (24, 24, 24),
            APRILTAG_LABEL_OUTLINE_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (anchor[0] + 8, anchor[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            APRILTAG_LABEL_FONT_SCALE,
            color,
            APRILTAG_LABEL_THICKNESS,
            cv2.LINE_AA,
        )
    return out


def apply_lens_distortion_heatmap(
    frame_bgr: np.ndarray,
    distortion_shift_map_px: np.ndarray | None,
) -> np.ndarray:
    if distortion_shift_map_px is None:
        return frame_bgr
    shift_map = np.asarray(distortion_shift_map_px, dtype=np.float32)
    finite = np.isfinite(shift_map)
    if not np.any(finite):
        return frame_bgr
    shift_min = float(np.min(shift_map[finite]))
    shift_max = float(np.max(shift_map[finite]))
    if shift_max > shift_min + 1e-9:
        normalized = np.clip((shift_map - shift_min) / (shift_max - shift_min), 0.0, 1.0)
    else:
        normalized = np.zeros_like(shift_map, dtype=np.float32)
    heat_u8 = np.zeros_like(shift_map, dtype=np.uint8)
    heat_u8[finite] = np.asarray(np.round(normalized[finite] * 255.0), dtype=np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_VIRIDIS)
    alpha = np.zeros_like(shift_map, dtype=np.float32)
    alpha[finite] = (
        LENS_DISTORTION_HEATMAP_BASE_ALPHA
        + (LENS_DISTORTION_HEATMAP_ALPHA - LENS_DISTORTION_HEATMAP_BASE_ALPHA)
        * np.power(normalized[finite], LENS_DISTORTION_HEATMAP_GAMMA)
    )
    alpha_3ch = alpha[..., None]
    frame_float = np.asarray(frame_bgr, dtype=np.float32)
    heat_float = np.asarray(heat_color, dtype=np.float32)
    blended = frame_float * (1.0 - alpha_3ch) + heat_float * alpha_3ch
    return np.asarray(np.round(np.clip(blended, 0.0, 255.0)), dtype=np.uint8)


def build_snapshot_overlay_context(
    *,
    input_dir: Path | None,
    tracked_video: Path | None,
    tracker_rows: pd.DataFrame,
    reference_frame_idx: int,
    reference_frame_in_window_idx: int,
) -> dict[str, object]:
    snapshot_frame_bgr: np.ndarray | None = None
    detections: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}
    distortion_shift_map_px: np.ndarray | None = None
    reference_pose = None
    arbor_marker_px = tracker_red_centroid_px(tracker_rows)
    calibration = None

    source_video: Path | None = None
    source_frame_idx: int | None = None
    if input_dir is not None:
        try:
            source_video = resolve_unique_video(input_dir)
            source_frame_idx = int(reference_frame_idx)
        except ValueError as exc:
            print(f"[warn] Could not resolve a unique capture video for snapshot panel: {exc}")

    if source_video is None and tracked_video is not None:
        source_video = tracked_video
        source_frame_idx = int(reference_frame_in_window_idx)

    if source_video is not None and source_frame_idx is not None:
        try:
            snapshot_frame_bgr = read_video_frame(source_video, source_frame_idx)
        except RuntimeError as exc:
            print(f"[warn] Could not read snapshot-panel source frame {source_frame_idx} from {source_video}: {exc}")

    if input_dir is None:
        if snapshot_frame_bgr is not None:
            detections = detect_apriltags(snapshot_frame_bgr)
            labels = {
                int(tag_id): f"{DEFAULT_APRILTAG_SIZE_MM:.2f} mm"
                for tag_id in sorted(detections)
            }
        return {
            "snapshot_frame_bgr": snapshot_frame_bgr,
            "apriltag_detections": detections,
            "apriltag_labels": labels,
            "distortion_shift_map_px": distortion_shift_map_px,
            "arbor_marker_px": arbor_marker_px,
            "spiral_points_px": (
                fit_snapshot_spiral_profile_px(tracker_rows, snapshot_frame_bgr.shape)
                if snapshot_frame_bgr is not None
                else np.empty((0, 2), dtype=float)
            ),
        }

    intrinsics_path = input_dir / "intrinsics.json"
    depth_calibration_path = input_dir / "depth_calibration.json"

    if snapshot_frame_bgr is not None and depth_calibration_path.exists():
        try:
            calibration = load_depth_calibration(depth_calibration_path)
            height, width = snapshot_frame_bgr.shape[:2]
            undistorter = build_frame_undistorter(calibration, width, height)
            snapshot_frame_bgr = undistorter.undistort(snapshot_frame_bgr)
            if arbor_marker_px is not None:
                arbor_marker_px = calibration.undistort_points(np.asarray([arbor_marker_px], dtype=float))[0]
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            distortion_shift_map_px = np.hypot(undistorter.map_x - grid_x, undistorter.map_y - grid_y)
        except (CalibrationError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"[warn] Could not build lens-distortion overlay for snapshot panel: {exc}")

    if snapshot_frame_bgr is not None:
        detections = detect_apriltags(snapshot_frame_bgr)
        labels = {
            int(tag_id): f"{DEFAULT_APRILTAG_SIZE_MM:.2f} mm"
            for tag_id in sorted(detections)
        }

    if detections and snapshot_frame_bgr is not None and intrinsics_path.exists():
        try:
            intrinsics = load_intrinsics_sequence(intrinsics_path)
            plane_calibration = build_plane_calibration(
                snapshot_frame_bgr,
                intrinsics.get(int(reference_frame_idx)),
                reference_frame_idx=int(reference_frame_idx),
                tag_size_mm=DEFAULT_APRILTAG_SIZE_MM,
            )
            reference_pose = plane_calibration.reference_pose
            measured_labels: dict[int, str] = {}
            for tag_id, corners_px in sorted(detections.items()):
                corners_plane = image_points_to_plane(corners_px, reference_pose)
                edge_vecs_mm = np.roll(corners_plane, -1, axis=0) - corners_plane
                edge_lengths_mm = np.linalg.norm(edge_vecs_mm, axis=1)
                if len(edge_lengths_mm):
                    measured_labels[int(tag_id)] = f"{float(np.mean(edge_lengths_mm)):.2f} mm"
            labels.update(measured_labels)
        except (CalibrationError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"[warn] Could not build AprilTag scale labels for snapshot panel: {exc}")

    spiral_points_px = (
        fit_snapshot_spiral_profile_px(tracker_rows, snapshot_frame_bgr.shape, reference_pose=reference_pose)
        if snapshot_frame_bgr is not None
        else np.empty((0, 2), dtype=float)
    )

    return {
        "snapshot_frame_bgr": snapshot_frame_bgr,
        "apriltag_detections": detections,
        "apriltag_labels": labels,
        "distortion_shift_map_px": distortion_shift_map_px,
        "arbor_marker_px": arbor_marker_px,
        "spiral_points_px": spiral_points_px,
    }


def tracker_crop_half_extents_px(
    rows: pd.DataFrame,
    center_xy: np.ndarray,
    image_shape: tuple[int, ...],
    apriltag_points: np.ndarray | None = None,
) -> tuple[int, int]:
    half_width_px = 0
    half_height_px = 0

    panel_points = tracker_panel_points_px(rows)
    if panel_points.size:
        max_dx = float(np.max(np.abs(panel_points[:, 0] - float(center_xy[0]))))
        max_dy = float(np.max(np.abs(panel_points[:, 1] - float(center_xy[1]))))
        half_width_px = max(half_width_px, int(math.ceil(max_dx + PAPER_SNAPSHOT_MARGIN_PX)))
        half_height_px = max(half_height_px, int(math.ceil(max_dy + PAPER_SNAPSHOT_MARGIN_PX)))

    if apriltag_points is not None and apriltag_points.size:
        max_dx = float(np.max(np.abs(apriltag_points[:, 0] - float(center_xy[0]))))
        max_dy = float(np.max(np.abs(apriltag_points[:, 1] - float(center_xy[1]))))
        half_width_px = max(half_width_px, int(math.ceil(max_dx + PAPER_APRILTAG_MARGIN_PX)))
        half_height_px = max(half_height_px, int(math.ceil(max_dy + PAPER_APRILTAG_MARGIN_PX)))

    if half_width_px == 0 or half_height_px == 0:
        height, width = image_shape[:2]
        half_width_px = int(math.ceil(0.5 * width))
        half_height_px = int(math.ceil(0.5 * height))
    return max(half_width_px, PAPER_PANEL_SIZE_PX // 2), max(half_height_px, PAPER_PANEL_SIZE_PX // 2)


def crop_rect_about_center(image_rgb: np.ndarray, center_xy: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    width_px = max(int(width_px), 2)
    height_px = max(int(height_px), 2)
    center_x = int(round(float(center_xy[0])))
    center_y = int(round(float(center_xy[1])))
    half_width = width_px // 2
    half_height = height_px // 2
    left = center_x - half_width
    top = center_y - half_height
    right = left + width_px
    bottom = top + height_px

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image_rgb.shape[1])
    pad_bottom = max(0, bottom - image_rgb.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    return image_rgb[top:bottom, left:right]


def resolve_snapshot_source_frame(
    *,
    input_dir: Path | None,
    tracked_video: Path | None,
    comparison_frame: pd.Series,
) -> np.ndarray:
    frame_idx = int(comparison_frame["frame"])
    frame_in_window_idx = int(comparison_frame["frame_in_window"])
    source_video: Path | None = None
    source_frame_idx: int | None = None
    if input_dir is not None:
        try:
            source_video = resolve_unique_video(input_dir)
            source_frame_idx = frame_idx
        except ValueError as exc:
            print(f"[warn] Could not resolve a unique capture video for snapshot panel: {exc}")
    if source_video is None and tracked_video is not None:
        source_video = tracked_video
        source_frame_idx = frame_in_window_idx
    if source_video is None or source_frame_idx is None:
        raise RuntimeError("A source video is required for snapshot rendering.")
    return read_video_frame(source_video, source_frame_idx)


def draw_vector_arrow_overlay(
    frame_bgr: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = frame_bgr.copy()
    start = tuple(np.round(np.asarray(start_xy, dtype=float)).astype(int))
    end = tuple(np.round(np.asarray(end_xy, dtype=float)).astype(int))
    cv2.arrowedLine(
        out,
        start,
        end,
        VECTOR_LINE_OUTLINE_COLOR_BGR,
        VECTOR_LINE_OUTLINE_THICKNESS_PX,
        cv2.LINE_AA,
        tipLength=VECTOR_LINE_TIP_LENGTH,
    )
    cv2.arrowedLine(
        out,
        start,
        end,
        color,
        VECTOR_LINE_THICKNESS_PX,
        cv2.LINE_AA,
        tipLength=VECTOR_LINE_TIP_LENGTH,
    )
    return out


def draw_vector_annotation_box(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame_bgr
    out = frame_bgr.copy()
    text_lines = [str(line) for line in lines if str(line).strip()]
    if not text_lines:
        return out
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, VECTOR_ANNOTATION_FONT_SCALE, 1)[0]
        for line in text_lines
    ]
    text_width = max((size[0] for size in sizes), default=0)
    text_height = sum(size[1] for size in sizes) + VECTOR_ANNOTATION_LINE_SPACING_PX * max(len(sizes) - 1, 0)
    box_left = VECTOR_ANNOTATION_MARGIN_PX
    box_top = VECTOR_ANNOTATION_MARGIN_PX
    box_right = box_left + text_width + 2 * VECTOR_ANNOTATION_PADDING_PX
    box_bottom = box_top + text_height + 2 * VECTOR_ANNOTATION_PADDING_PX
    cv2.rectangle(
        out,
        (box_left, box_top),
        (box_right, box_bottom),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        out,
        (box_left, box_top),
        (box_right, box_bottom),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    baseline_y = box_top + VECTOR_ANNOTATION_PADDING_PX
    for line, size in zip(text_lines, sizes):
        baseline_y += size[1]
        origin = (box_left + VECTOR_ANNOTATION_PADDING_PX, baseline_y)
        cv2.putText(
            out,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            VECTOR_ANNOTATION_FONT_SCALE,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            VECTOR_ANNOTATION_FONT_SCALE,
            (24, 24, 24),
            1,
            cv2.LINE_AA,
        )
        baseline_y += VECTOR_ANNOTATION_LINE_SPACING_PX
    return out


def render_arbor_vector_snapshot_panel(
    *,
    video_frame_bgr: np.ndarray,
    tracker_rows: pd.DataFrame,
    include_deformed_vector: bool,
    half_extent_px: int | None = None,
    annotation_lines: list[str] | None = None,
    preferred_dot_id: int | None = None,
) -> np.ndarray:
    arbor_xy, reference_xy, current_xy = tracker_first_node_vectors_px(
        tracker_rows,
        preferred_dot_id=preferred_dot_id,
    )
    panel_bgr = draw_vector_arrow_overlay(
        video_frame_bgr,
        arbor_xy,
        reference_xy,
        color=VECTOR_REFERENCE_COLOR_BGR,
    )
    if include_deformed_vector and current_xy is not None:
        panel_bgr = draw_vector_arrow_overlay(
            panel_bgr,
            arbor_xy,
            current_xy,
            color=VECTOR_DEFORMED_COLOR_BGR,
        )
    crop_half_extent_px = (
        int(half_extent_px)
        if half_extent_px is not None
        else tracker_vector_crop_half_extent_px(
            tracker_rows,
            include_deformed_vector=include_deformed_vector,
            preferred_dot_id=preferred_dot_id,
        )
    )
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    cropped_rgb = crop_rect_about_center(
        panel_rgb,
        arbor_xy,
        2 * crop_half_extent_px,
        2 * crop_half_extent_px,
    )
    if cropped_rgb.shape[1] != PAPER_SNAPSHOT_SIZE_PX or cropped_rgb.shape[0] != PAPER_SNAPSHOT_SIZE_PX:
        cropped_rgb = cv2.resize(
            cropped_rgb,
            (PAPER_SNAPSHOT_SIZE_PX, PAPER_SNAPSHOT_SIZE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    if annotation_lines:
        cropped_bgr = draw_vector_annotation_box(cropped_bgr, annotation_lines)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return cropped_rgb


def draw_comparison_curve(
    ax,
    *,
    record: dict,
    comparison_frame: pd.Series,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    tracker_rows: pd.DataFrame | None = None,
    figure_title: str,
    angle_label: str,
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_fontsize: float,
    legend_fontsize: float,
    annotation_fontsize: float,
    info_fontsize: float,
    show_info_text: bool,
    optimizer_label: str = "optimizer sweep state",
    tracker_label: str = "tracker frame",
    x_label: str = "Normalised arc length  s / s_total",
    y_label: str = "Displacement  (mm)",
    title_fontweight: str | float = "normal",
    tracker_linewidth: float = 2.0,
    tracker_markersize: float = 3.0,
    solidworks_overlays: list[dict[str, object]] | None = None,
    legend_loc: str = "lower right",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_bbox_transform: matplotlib.transforms.Transform | None = None,
    legend_ncol: int = 1,
    summary_lines: list[str] | None = None,
    include_force_lines: bool = False,
) -> None:
    ax.plot(
        optimizer_s_norm,
        optimizer_curve_mm,
        "-",
        lw=1.0,
        color="black",
        label=optimizer_label,
        zorder=4,
    )
    for overlay in solidworks_overlays or []:
        overlay_s_norm = np.asarray(overlay["s_norm"], dtype=float)
        overlay_curve_mm = np.asarray(overlay["curve_mm"], dtype=float)
        if overlay_s_norm.size == 0 or overlay_curve_mm.size == 0:
            continue
        ax.plot(
            overlay_s_norm,
            overlay_curve_mm,
            linestyle=str(overlay["linestyle"]),
            lw=float(overlay["linewidth"]),
            color=str(overlay["color"]),
            label=str(overlay["label"]),
            zorder=4.5,
        )
    ax.plot(
        tracker_s_norm,
        tracker_curve_mm,
        "o-",
        lw=tracker_linewidth,
        color="crimson",
        ms=tracker_markersize,
        label=tracker_label,
        zorder=5,
    )

    annotate_endpoint(
        ax,
        float(optimizer_curve_mm[-1]),
        f"optimizer final node = {float(optimizer_curve_mm[-1]):.3f} mm",
        "black",
        linestyle=":",
        text_offset_pts=8.0,
        fontsize=annotation_fontsize,
    )
    annotate_endpoint(
        ax,
        float(tracker_curve_mm[-1]),
        f"tracker final node = {float(tracker_curve_mm[-1]):.3f} mm",
        "crimson",
        linestyle="--",
        text_offset_pts=-8.0,
        fontsize=annotation_fontsize,
    )

    if show_info_text:
        info_text = build_overlay_text(
            record=record,
            optimizer_s_norm=optimizer_s_norm,
            optimizer_curve_mm=optimizer_curve_mm,
            comparison_frame=comparison_frame,
            tracker_s_norm=tracker_s_norm,
            tracker_curve_mm=tracker_curve_mm,
            tracker_rows=tracker_rows,
            angle_label=angle_label,
            summary_lines=summary_lines,
            include_force_lines=include_force_lines,
        )
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=info_fontsize,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
        )

    ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
    ax.set_title(figure_title, fontsize=title_fontsize, fontweight=title_fontweight)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.35)
    legend_kwargs: dict[str, object] = {
        "loc": legend_loc,
        "fontsize": legend_fontsize,
        "ncol": legend_ncol,
    }
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        legend_kwargs["borderaxespad"] = 0.0
    if legend_bbox_transform is not None:
        legend_kwargs["bbox_transform"] = legend_bbox_transform
    ax.legend(**legend_kwargs)


def render_paper_snapshot_panel(
    *,
    video_frame_bgr: np.ndarray,
    tracker_rows: pd.DataFrame,
    apriltag_detections: dict[int, np.ndarray] | None = None,
    apriltag_labels: dict[int, str] | None = None,
    distortion_shift_map_px: np.ndarray | None = None,
    arbor_marker_px: np.ndarray | None = None,
    spiral_points_px: np.ndarray | None = None,
    title: str = PAPER_SNAPSHOT_TITLE,
) -> np.ndarray:
    frame_bgr = apply_lens_distortion_heatmap(video_frame_bgr, distortion_shift_map_px)
    if spiral_points_px is not None and len(spiral_points_px):
        frame_bgr = draw_polyline_overlay(
            frame_bgr,
            spiral_points_px,
            color=SNAPSHOT_SPIRAL_COLOR_BGR,
            outline_color=SNAPSHOT_SPIRAL_OUTLINE_COLOR_BGR,
            thickness=2,
            outline_thickness=4,
        )
    frame_bgr = draw_arbor_marker_overlay(frame_bgr, arbor_marker_px)
    frame_bgr = draw_tracker_marker_overlay(frame_bgr, tracker_rows)
    if apriltag_detections:
        frame_bgr = draw_apriltag_scale_overlay(frame_bgr, apriltag_detections, apriltag_labels)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    center_xy = tracker_arbor_center_px(tracker_rows, rgb.shape)
    apriltag_points = (
        apriltag_points_from_detections(apriltag_detections)
        if apriltag_detections is not None
        else apriltag_panel_points_px(video_frame_bgr)
    )
    half_width_px, half_height_px = tracker_crop_half_extents_px(
        tracker_rows,
        center_xy,
        rgb.shape,
        apriltag_points=apriltag_points,
    )
    cropped_rgb = crop_rect_about_center(rgb, center_xy, 2 * half_width_px, 2 * half_height_px)
    min_distortion_px: float | None = None
    max_distortion_px: float | None = None
    if distortion_shift_map_px is not None:
        cropped_shift_map = crop_rect_about_center(distortion_shift_map_px, center_xy, 2 * half_width_px, 2 * half_height_px)
        finite = np.isfinite(cropped_shift_map)
        if np.any(finite):
            min_distortion_px = float(np.min(cropped_shift_map[finite]))
            max_distortion_px = float(np.max(cropped_shift_map[finite]))

    show_distortion_key = min_distortion_px is not None and max_distortion_px is not None
    fig = plt.figure(figsize=(PAPER_SNAPSHOT_SIZE_IN, PAPER_SNAPSHOT_SIZE_IN), dpi=PAPER_PANEL_DPI)
    if show_distortion_key:
        footer_height_in = DISTORTION_KEY_FOOTER_HEIGHT_IN
        image_height_in = max(PAPER_SNAPSHOT_SIZE_IN - footer_height_in, 0.5)
        gs = fig.add_gridspec(2, 1, height_ratios=[image_height_in, footer_height_in], hspace=0.02)
        ax = fig.add_subplot(gs[0, 0])
        ax_key = fig.add_subplot(gs[1, 0])
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax_key = None

    ax.imshow(cropped_rgb, aspect="equal")
    if title:
        ax.set_title(title, fontsize=PAPER_TITLE_FONTSIZE, pad=4)
    ax.axis("off")
    if ax_key is not None:
        draw_distortion_key(
            ax_key,
            min_distortion_px=min_distortion_px,
            max_distortion_px=max_distortion_px,
        )
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.89 if title else 0.99)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.89 if title else 0.99)
    image = figure_to_rgb(fig)
    if image.shape[1] != PAPER_SNAPSHOT_SIZE_PX or image.shape[0] != PAPER_SNAPSHOT_SIZE_PX:
        image = cv2.resize(
            image,
            (PAPER_SNAPSHOT_SIZE_PX, PAPER_SNAPSHOT_SIZE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
    plt.close(fig)
    return image


def render_reference_apparatus_snapshot(
    *,
    input_dir: Path | None,
    tracked_video: Path | None,
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
) -> np.ndarray | None:
    if frame_table.empty:
        return None

    reference_frame = choose_reference_frame(frame_table, mode="angle")
    reference_frame_rows = tracker_df[
        tracker_df["frame"] == int(reference_frame["frame"])
    ].sort_values("dot_id")
    if reference_frame_rows.empty:
        print(
            f"[warn] Reference frame {int(reference_frame['frame'])} has no tracker rows; "
            "skipping camera-tracking apparatus snapshot."
        )
        return None

    overlay_context = build_snapshot_overlay_context(
        input_dir=input_dir,
        tracked_video=tracked_video,
        tracker_rows=reference_frame_rows,
        reference_frame_idx=int(reference_frame["frame"]),
        reference_frame_in_window_idx=int(reference_frame["frame_in_window"]),
    )
    snapshot_frame_bgr = overlay_context["snapshot_frame_bgr"]
    if snapshot_frame_bgr is None:
        return None

    return render_paper_snapshot_panel(
        video_frame_bgr=snapshot_frame_bgr,
        tracker_rows=reference_frame_rows,
        apriltag_detections=overlay_context["apriltag_detections"],
        apriltag_labels=overlay_context["apriltag_labels"],
        distortion_shift_map_px=overlay_context["distortion_shift_map_px"],
        arbor_marker_px=overlay_context["arbor_marker_px"],
        spiral_points_px=overlay_context["spiral_points_px"],
        title=PAPER_SNAPSHOT_TITLE,
    )


def render_paper_curve_panel(
    *,
    record: dict,
    comparison_frame: pd.Series,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    solidworks_overlays: list[dict[str, object]] | None = None,
    figure_title: str = PAPER_CURVE_TITLE,
    angle_label: str,
    summary_lines: list[str] | None = None,
    include_force_lines: bool = False,
) -> np.ndarray:
    axes_size_in = PAPER_CURVE_AXES_SIZE_MM / 25.4
    left_margin_in = PAPER_CURVE_LEFT_MARGIN_MM / 25.4
    right_margin_in = PAPER_CURVE_RIGHT_MARGIN_MM / 25.4
    top_margin_in = PAPER_CURVE_TOP_MARGIN_MM / 25.4
    bottom_margin_in = PAPER_CURVE_BOTTOM_MARGIN_MM / 25.4
    legend_bottom_in = PAPER_CURVE_LEGEND_BOTTOM_MM / 25.4

    fig_width_in = left_margin_in + axes_size_in + right_margin_in
    fig_height_in = top_margin_in + axes_size_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=PAPER_PANEL_DPI)
    ax_left = left_margin_in / fig_width_in
    ax_bottom = bottom_margin_in / fig_height_in
    ax_width = axes_size_in / fig_width_in
    ax_height = axes_size_in / fig_height_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    draw_comparison_curve(
        ax,
        record=record,
        comparison_frame=comparison_frame,
        optimizer_s_norm=optimizer_s_norm,
        optimizer_curve_mm=optimizer_curve_mm,
        tracker_s_norm=tracker_s_norm,
        tracker_curve_mm=tracker_curve_mm,
        figure_title=figure_title,
        angle_label=angle_label,
        title_fontsize=PAPER_TITLE_FONTSIZE,
        axis_label_fontsize=PAPER_AXIS_LABEL_FONTSIZE,
        tick_fontsize=PAPER_TICK_FONTSIZE,
        legend_fontsize=PAPER_LEGEND_FONTSIZE,
        annotation_fontsize=PAPER_ANNOTATION_FONTSIZE,
        info_fontsize=PAPER_ANNOTATION_FONTSIZE,
        show_info_text=False,
        optimizer_label="optimizer URES",
        tracker_label="physical URES",
        x_label="Normalized Arc Length",
        title_fontweight="bold",
        tracker_linewidth=0.5,
        tracker_markersize=1.0,
        solidworks_overlays=solidworks_overlays,
        legend_loc="lower center",
        legend_bbox_to_anchor=((2.0 * left_margin_in + axes_size_in) / (2.0 * fig_width_in), legend_bottom_in / fig_height_in),
        legend_bbox_transform=fig.transFigure,
        legend_ncol=1,
        summary_lines=summary_lines,
        include_force_lines=include_force_lines,
    )
    ax.set_box_aspect(1)
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def _axes_sized_paper_figure(
    *,
    title: str | None = None,
    bottom_extra_mm: float,
) -> tuple[plt.Figure, plt.Axes]:
    left_in = SQUARE_PAPER_LEFT_MARGIN_MM / 25.4
    right_in = SQUARE_PAPER_RIGHT_MARGIN_MM / 25.4
    top_in = SQUARE_PAPER_TOP_MARGIN_MM / 25.4
    bottom_in = float(bottom_extra_mm) / 25.4
    axes_in = SQUARE_PAPER_AXES_MM / 25.4
    fig_width_in = left_in + axes_in + right_in
    fig_height_in = top_in + axes_in + bottom_in
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=PAPER_PANEL_DPI)
    ax = fig.add_axes(
        [
            left_in / fig_width_in,
            bottom_in / fig_height_in,
            axes_in / fig_width_in,
            axes_in / fig_height_in,
        ]
    )
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=SQUARE_PAPER_TEXT_PT, pad=3)
    return fig, ax


def draw_compact_summary_box(
    target,
    lines: list[str],
    *,
    fontsize: float = SQUARE_PAPER_TEXT_PT,
    x: float = 0.02,
    y: float = 0.98,
    ha: str = "left",
    va: str = "top",
) -> None:
    text_lines = [str(line) for line in lines if str(line).strip()]
    if not text_lines:
        return
    target.text(
        x,
        y,
        "\n".join(text_lines),
        transform=target.transFigure,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.92),
    )


def endpoint_guide(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    linestyle: object = (0, (1.0, 1.2)),
    linewidth: float = 0.6,
    alpha: float = 0.6,
    zorder: float = 4.6,
) -> dict[str, object] | None:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite_mask.any():
        return None
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    return {
        "x_end": float(x_arr[-1]),
        "y": float(y_arr[-1]),
        "color": color,
        "label": label if label is not None else f"{float(y_arr[-1]):.1f}",
        "linestyle": linestyle,
        "linewidth": float(linewidth),
        "alpha": float(alpha),
        "zorder": float(zorder),
    }


def reverse_curve_over_domain(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite_mask.any():
        return x_arr, y_arr
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    x_min = float(x_arr[0])
    x_max = float(x_arr[-1])
    return x_min + x_max - x_arr[::-1], y_arr[::-1]


def crop_significant_fea_stress_endpoint_outliers(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    max_trim: int = SQUARE_PAPER_STRESS_FEA_OUTLIER_MAX_TRIM,
    step_factor: float = SQUARE_PAPER_STRESS_FEA_OUTLIER_STEP_FACTOR,
    offset_factor: float = SQUARE_PAPER_STRESS_FEA_OUTLIER_OFFSET_FACTOR,
    slope_factor: float = SQUARE_PAPER_STRESS_FEA_OUTLIER_SLOPE_FACTOR,
    neighborhood: int = SQUARE_PAPER_STRESS_FEA_OUTLIER_NEIGHBORHOOD,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite_mask.any():
        return x_arr, y_arr
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if x_arr.size <= 6:
        return x_arr, y_arr

    trim_start = 0
    trim_end = 0
    max_trim = max(0, min(int(max_trim), max(0, x_arr.size // 6)))

    def _robust_scale(values: np.ndarray) -> float:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return float("nan")
        scale = float(np.median(finite_values))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = float(np.mean(finite_values))
        return scale

    def should_trim_start(x_view: np.ndarray, y_view: np.ndarray) -> bool:
        if y_view.size < max(6, neighborhood + 3):
            return False
        diffs = np.abs(np.diff(y_view))
        dx = np.abs(np.diff(x_view))
        if diffs.size < 4 or dx.size != diffs.size:
            return False
        slopes = diffs / np.clip(dx, 1e-12, None)
        interior_slice = slice(neighborhood - 1, diffs.size - (neighborhood - 1))
        interior_steps = diffs[interior_slice]
        interior_slopes = slopes[interior_slice]
        if interior_steps.size == 0:
            interior_steps = diffs
        if interior_slopes.size == 0:
            interior_slopes = slopes
        typical_step = _robust_scale(interior_steps)
        typical_slope = _robust_scale(interior_slopes)
        if (
            not math.isfinite(typical_step)
            or typical_step <= 1e-12
            or not math.isfinite(typical_slope)
            or typical_slope <= 1e-12
        ):
            return False
        local_window = y_view[1:min(1 + neighborhood, y_view.size)]
        if local_window.size == 0:
            return False
        local_ref = float(np.median(local_window))
        first_step = float(diffs[0])
        first_slope = float(slopes[0])
        local_offset = abs(float(y_view[0]) - local_ref)
        high_slope = first_slope > slope_factor * typical_slope
        high_offset = local_offset > offset_factor * typical_step
        high_step = first_step > step_factor * typical_step
        return high_slope and (high_offset or high_step)

    def should_trim_end(x_view: np.ndarray, y_view: np.ndarray) -> bool:
        if y_view.size < max(6, neighborhood + 3):
            return False
        return should_trim_start(x_view[::-1], y_view[::-1])

    while trim_start < max_trim:
        end_idx = len(y_arr) - trim_end if trim_end > 0 else len(y_arr)
        x_view = x_arr[trim_start:end_idx]
        y_view = y_arr[trim_start:end_idx]
        if not should_trim_start(x_view, y_view):
            break
        trim_start += 1
    while trim_end < max_trim:
        end_idx = len(y_arr) - trim_end if trim_end > 0 else len(y_arr)
        x_view = x_arr[trim_start:end_idx]
        y_view = y_arr[trim_start:end_idx]
        if not should_trim_end(x_view, y_view):
            break
        trim_end += 1

    if trim_start == 0 and trim_end == 0:
        return x_arr, y_arr
    end_idx = len(x_arr) - trim_end if trim_end > 0 else len(x_arr)
    return x_arr[trim_start:end_idx], y_arr[trim_start:end_idx]


def draw_endpoint_guides(
    ax,
    guides: list[dict[str, object]],
    *,
    fontsize: float = SQUARE_PAPER_LEGEND_PT - 1.0,
) -> None:
    if not guides:
        return
    x_anchor = 0.0
    for guide in guides:
        x_end = float(guide["x_end"])
        y_value = float(guide["y"])
        if not (math.isfinite(x_end) and math.isfinite(y_value)):
            continue
        ax.plot(
            [x_anchor, x_end],
            [y_value, y_value],
            linestyle=guide.get("linestyle", (0, (1.0, 1.2))),
            color=str(guide.get("color", "black")),
            lw=float(guide.get("linewidth", 0.6)),
            alpha=float(guide.get("alpha", 0.6)),
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=float(guide.get("zorder", 4.6)),
            label="_nolegend_",
        )
        ax.annotate(
            str(guide.get("label", f"{y_value:.3f}")),
            xy=(x_anchor, y_value),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=fontsize,
            color=str(guide.get("color", "black")),
            clip_on=False,
        )


def render_square_xy_overlay_plot(
    *,
    optimizer_x: np.ndarray,
    optimizer_y: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
    figure_title: str,
    x_label: str,
    y_label: str,
    optimizer_label: str = "optimizer sweep",
    tracker_label: str = "tracker",
    fit_linear_models: bool = False,
    bottom_extra_mm: float = SQUARE_PAPER_BOTTOM_LEGEND_MM,
    legend_inside_axes: bool = False,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 0.035),
    legend_fontsize_pt: float = SQUARE_PAPER_LEGEND_PT,
    legend_ncol: int = 2,
) -> np.ndarray:
    fig, ax = _axes_sized_paper_figure(
        title=figure_title,
        bottom_extra_mm=bottom_extra_mm,
    )
    optimizer_x_arr = np.asarray(optimizer_x, dtype=float)
    optimizer_y_arr = np.asarray(optimizer_y, dtype=float)
    tracker_x_arr = np.asarray(tracker_x, dtype=float)
    tracker_y_arr = np.asarray(tracker_y, dtype=float)
    optimizer_order = np.argsort(optimizer_x_arr)
    ax.plot(
        optimizer_x_arr[optimizer_order],
        optimizer_y_arr[optimizer_order],
        "-",
        lw=1.2,
        color="black",
        label=optimizer_label,
        zorder=4,
    )
    finite_tracker = np.isfinite(tracker_x_arr) & np.isfinite(tracker_y_arr)
    ax.scatter(
        tracker_x_arr[finite_tracker],
        tracker_y_arr[finite_tracker],
        s=SQUARE_PAPER_TRACKER_SCATTER_SIZE,
        color="crimson",
        alpha=0.28,
        edgecolors="none",
        label=tracker_label if fit_linear_models else "_nolegend_",
        zorder=4.7,
    )
    tracker_curve_x, tracker_curve_y = _binned_median_curve(
        tracker_x_arr[finite_tracker],
        tracker_y_arr[finite_tracker],
    )
    if not fit_linear_models:
        ax.plot(
            tracker_curve_x,
            tracker_curve_y,
            "-",
            lw=1.2,
            color="crimson",
            label=tracker_label,
            zorder=5.1,
        )
    fit_label_specs: list[tuple[str, str]] = []
    if fit_linear_models:
        optimizer_fit = _linear_fit_params(
            optimizer_x_arr[optimizer_order],
            optimizer_y_arr[optimizer_order],
        )
        if optimizer_fit is not None:
            optimizer_fit_x = np.asarray(optimizer_fit["x"], dtype=float)
            optimizer_fit_y = np.asarray(optimizer_fit["y_fit"], dtype=float)
            ax.plot(
                optimizer_fit_x,
                optimizer_fit_y,
                "--",
                lw=1.0,
                color=SQUARE_PAPER_FIT_SWEEP_COLOR,
                alpha=0.95,
                label="_nolegend_",
                zorder=5.4,
            )
            fit_label_specs.append(
                (
                    f"Sweep fit: {float(optimizer_fit['slope_per_deg']):.4f} N*m/deg",
                    SQUARE_PAPER_FIT_SWEEP_COLOR,
                )
            )
        tracked_fit = _detect_linear_tail_fit(tracker_curve_x, tracker_curve_y)
        if tracked_fit is not None:
            tracked_fit_x = np.asarray(tracked_fit["x"], dtype=float)
            tracked_fit_y = np.asarray(tracked_fit["y_fit"], dtype=float)
            ax.plot(
                tracked_fit_x,
                tracked_fit_y,
                "-",
                lw=1.25,
                color=SQUARE_PAPER_FIT_TRACKED_COLOR,
                alpha=0.95,
                label="_nolegend_",
                zorder=5.5,
            )
            ax.plot(
                [float(tracked_fit["start_x"])],
                [float(tracked_fit["start_y"])],
                marker="o",
                ms=4.0,
                markerfacecolor="white",
                markeredgecolor=SQUARE_PAPER_FIT_TRACKED_COLOR,
                markeredgewidth=0.9,
                linestyle="none",
                zorder=5.6,
            )
            fit_label_specs.append(
                (
                    f"Tracked fit: {float(tracked_fit['slope_per_deg']):.4f} N*m/deg",
                    SQUARE_PAPER_FIT_TRACKED_COLOR,
                )
            )
    ax.set_xlabel(x_label, fontsize=SQUARE_PAPER_TEXT_PT)
    ax.set_ylabel(y_label, fontsize=SQUARE_PAPER_TEXT_PT)
    ax.tick_params(axis="both", labelsize=SQUARE_PAPER_TEXT_PT)
    ax.grid(True, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_kwargs = dict(
            loc=legend_loc,
            fontsize=legend_fontsize_pt,
            ncol=legend_ncol,
            framealpha=0.92,
        )
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        if legend_inside_axes:
            legend = ax.legend(handles, labels, **legend_kwargs)
        else:
            legend = fig.legend(handles, labels, **legend_kwargs)
        legend.get_frame().set_edgecolor("0.72")
        legend.get_frame().set_linewidth(0.55)
    if fit_label_specs:
        axes_bbox = ax.get_position()
        center_x = float(axes_bbox.x0 + 0.5 * axes_bbox.width)
        margin_height = float(axes_bbox.y0)
        line_gap = max(0.028, min(0.045, margin_height * 0.18))
        base_y = max(0.035, margin_height * 0.23)
        start_y = base_y + 0.5 * line_gap * (len(fit_label_specs) - 1)
        for idx, (fit_text, fit_color) in enumerate(fit_label_specs):
            fig.text(
                center_x,
                start_y - idx * line_gap,
                fit_text,
                ha="center",
                va="center",
                fontsize=SQUARE_PAPER_LEGEND_PT,
                color=fit_color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=fit_color if fit_color != "black" else "0.5",
                    alpha=0.9,
                ),
            )
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def render_square_ures_comparison_plot(
    *,
    record: dict,
    comparison_frame: pd.Series,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    tracker_rows: pd.DataFrame | None = None,
    solidworks_overlays: list[dict[str, object]] | None = None,
    figure_title: str,
    compact_lines: list[str] | None = None,
    endpoint_guides: list[dict[str, object]] | None = None,
    compact_box_y: float = 0.03,
    bottom_extra_mm: float = SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM,
    legend_inside_axes: bool = False,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 0.11),
    legend_fontsize_pt: float = SQUARE_PAPER_LEGEND_PT,
    legend_ncol: int = 2,
) -> np.ndarray:
    fig, ax = _axes_sized_paper_figure(
        title=figure_title,
        bottom_extra_mm=bottom_extra_mm,
    )
    compact_lines_to_draw = list(compact_lines) if compact_lines is not None else None
    mean_curve_error_line = format_mean_ures_error_line(
        curve_mean_absolute_error_metrics(
            optimizer_s_norm,
            optimizer_curve_mm,
            tracker_s_norm,
            tracker_curve_mm,
            tracker_rows=tracker_rows,
        )
    )
    if mean_curve_error_line is not None:
        if compact_lines_to_draw is None:
            compact_lines_to_draw = []
        compact_lines_to_draw.append(mean_curve_error_line)
    sweep_line, = ax.plot(
        optimizer_s_norm,
        optimizer_curve_mm,
        "-",
        lw=SQUARE_PAPER_URES_SWEEP_LINEWIDTH,
        color=SQUARE_PAPER_URES_SWEEP_COLOR,
        label="Sweep URES",
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=SQUARE_PAPER_URES_SWEEP_ZORDER,
    )
    overlay_curves = []
    for overlay in solidworks_overlays or []:
        overlay_s_norm = np.asarray(overlay["s_norm"], dtype=float)
        overlay_curve_mm = np.asarray(overlay["curve_mm"], dtype=float)
        finite_mask = np.isfinite(overlay_s_norm) & np.isfinite(overlay_curve_mm)
        if not finite_mask.any():
            continue
        overlay_s_norm = overlay_s_norm[finite_mask]
        overlay_curve_mm = overlay_curve_mm[finite_mask]
        order = np.argsort(overlay_s_norm)
        overlay_curves.append((overlay_s_norm[order], overlay_curve_mm[order], overlay))
    if len(overlay_curves) >= 2:
        x_min = min(float(np.min(x_vals)) for x_vals, _, _ in overlay_curves)
        x_max = max(float(np.max(x_vals)) for x_vals, _, _ in overlay_curves)
        if math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min:
            sample_count = max(240, max(len(x_vals) for x_vals, _, _ in overlay_curves))
            envelope_x = np.linspace(x_min, x_max, sample_count, dtype=float)
            envelope_samples: list[np.ndarray] = []
            for x_vals, y_vals, _ in overlay_curves:
                y_interp = np.interp(
                    envelope_x,
                    x_vals,
                    y_vals,
                    left=np.nan,
                    right=np.nan,
                )
                envelope_samples.append(y_interp)
            envelope_stack = np.vstack(envelope_samples)
            valid_columns = np.isfinite(envelope_stack).sum(axis=0) >= 2
            if np.any(valid_columns):
                band_lower = np.nanmin(envelope_stack[:, valid_columns], axis=0)
                band_upper = np.nanmax(envelope_stack[:, valid_columns], axis=0)
                ax.fill_between(
                    envelope_x[valid_columns],
                    band_lower,
                    band_upper,
                    facecolor=SQUARE_PAPER_URES_BAND_COLOR,
                    alpha=SQUARE_PAPER_URES_BAND_ALPHA,
                    edgecolor="none",
                    linewidth=0.0,
                    label="FEA URES band",
                    zorder=SQUARE_PAPER_URES_BAND_ZORDER,
                )
    elif len(overlay_curves) == 1:
        overlay_s_norm, overlay_curve_mm, _ = overlay_curves[0]
        overlay_color, overlay_linestyle = SQUARE_PAPER_URES_OVERLAY_STYLES[0]
        overlay_line, = ax.plot(
            overlay_s_norm,
            overlay_curve_mm,
            linestyle=overlay_linestyle,
            lw=1.1,
            color=overlay_color,
            alpha=0.85,
            label="FEA URES",
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4.0,
        )
        overlay_line.set_path_effects(SQUARE_PAPER_URES_OVERLAY_STROKE)
    tracker_line, = ax.plot(
        tracker_s_norm,
        tracker_curve_mm,
        "-",
        lw=SQUARE_PAPER_URES_TRACKED_LINEWIDTH,
        color=SQUARE_PAPER_URES_TRACKED_COLOR,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Tracked URES",
        zorder=SQUARE_PAPER_URES_TRACKED_ZORDER,
    )
    optimizer_curve_arr = np.asarray(optimizer_curve_mm, dtype=float)
    tracker_curve_arr = np.asarray(tracker_curve_mm, dtype=float)
    overlay_arrays = [
        curve_mm
        for _, curve_mm, _ in overlay_curves
        if curve_mm.size > 0
    ]
    y_arrays = [optimizer_curve_arr, tracker_curve_arr, *overlay_arrays]
    finite_y = np.concatenate([arr[np.isfinite(arr)] for arr in y_arrays if np.isfinite(arr).any()])
    if finite_y.size > 0:
        y_min = float(np.min(finite_y))
        y_max = float(np.max(finite_y))
        y_span = y_max - y_min
        y_pad = max(0.25, SQUARE_PAPER_URES_Y_PADDING_FRAC * max(y_span, 1.0))
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(-0.015, 1.015)
    ax.set_xlabel("Normalized Arc Length", fontsize=SQUARE_PAPER_TEXT_PT)
    ax.set_ylabel("Displacement (mm)", fontsize=SQUARE_PAPER_TEXT_PT)
    ax.tick_params(axis="both", labelsize=SQUARE_PAPER_TEXT_PT)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.22, linestyle=":", linewidth=0.55)
    if endpoint_guides:
        draw_endpoint_guides(ax, endpoint_guides)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_kwargs = dict(
            loc=legend_loc,
            fontsize=legend_fontsize_pt,
            ncol=legend_ncol,
            framealpha=0.92,
        )
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        if legend_inside_axes:
            legend = ax.legend(handles, labels, **legend_kwargs)
        else:
            legend = fig.legend(handles, labels, **legend_kwargs)
        legend.get_frame().set_edgecolor("0.72")
        legend.get_frame().set_linewidth(0.55)
    if compact_lines_to_draw:
        draw_compact_summary_box(
            fig,
            compact_lines_to_draw,
            x=0.5,
            y=compact_box_y,
            ha="center",
            va="bottom",
        )
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def build_square_stress_curve_bundle(
    *,
    spring: Spring,
    solidworks_stress_overlays: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    spring_s_norm, spring_inner_mpa, spring_outer_mpa = spring_inner_outer_stress_curves_mpa(spring)
    curve_bundles: list[dict[str, object]] = [
        {
            "series": "sweep_inner",
            "label": "Optimizer inner stress",
            "role": "optimizer",
            "side": "inner",
            "s_norm": spring_s_norm,
            "stress_mpa": spring_inner_mpa,
            "color": SQUARE_PAPER_STRESS_OPT_INNER_COLOR,
            "linestyle": "-",
            "linewidth": SQUARE_PAPER_STRESS_OPT_LINEWIDTH,
        },
        {
            "series": "sweep_outer",
            "label": "Optimizer outer stress",
            "role": "optimizer",
            "side": "outer",
            "s_norm": spring_s_norm,
            "stress_mpa": spring_outer_mpa,
            "color": SQUARE_PAPER_STRESS_OPT_OUTER_COLOR,
            "linestyle": "-",
            "linewidth": SQUARE_PAPER_STRESS_OPT_LINEWIDTH,
        },
    ]
    for overlay in solidworks_stress_overlays or []:
        curve_bundles.append(
            {
                "series": str(overlay.get("series", str(overlay["label"]).lower().replace(" ", "_"))),
                "label": str(overlay["label"]),
                "role": "fea",
                "side": overlay.get("side"),
                "s_norm": np.asarray(overlay["s_norm"], dtype=float),
                "stress_mpa": np.asarray(overlay["stress_mpa"], dtype=float),
                "color": str(overlay["color"]),
                "linestyle": str(overlay["linestyle"]),
                "linewidth": float(overlay["linewidth"]),
            }
        )
    return curve_bundles


def render_square_stress_curve_bundle_plot(
    *,
    curve_bundles: list[dict[str, object]],
    figure_title: str,
    draw_final_value_guides: bool = False,
    reverse_fea_sides: set[str] | None = None,
    legend_inside_axes: bool = False,
    legend_loc: str = "lower center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 0.035),
    legend_fontsize_pt: float = SQUARE_PAPER_LEGEND_PT,
    legend_ncol: int = 2,
) -> np.ndarray:
    fig, ax = _axes_sized_paper_figure(
        title=figure_title,
        bottom_extra_mm=SQUARE_PAPER_BOTTOM_LEGEND_MM,
    )
    styled_curves: list[dict[str, object]] = []
    reverse_side_set = {str(side).strip().lower() for side in (reverse_fea_sides or set()) if str(side).strip()}
    for curve in curve_bundles:
        series = str(curve.get("series", ""))
        side = curve.get("side")
        if side not in {"inner", "outer"}:
            if "inner" in str(curve.get("label", "")).lower() or series.endswith("_1"):
                side = "inner"
            elif "outer" in str(curve.get("label", "")).lower() or series.endswith("_2"):
                side = "outer"
        role = str(curve.get("role", "optimizer" if series.startswith("sweep_") else "fea"))
        s_norm = np.asarray(curve["s_norm"], dtype=float)
        stress_mpa = np.asarray(curve["stress_mpa"], dtype=float)
        if role == "fea" and side in reverse_side_set:
            s_norm, stress_mpa = reverse_curve_over_domain(s_norm, stress_mpa)
        if role == "fea":
            s_norm, stress_mpa = crop_significant_fea_stress_endpoint_outliers(s_norm, stress_mpa)
        if role == "optimizer":
            display_label = "Optimizer inner stress" if side == "inner" else "Optimizer outer stress"
            color = (
                SQUARE_PAPER_STRESS_OPT_INNER_COLOR
                if side == "inner"
                else SQUARE_PAPER_STRESS_OPT_OUTER_COLOR
            )
            linestyle = "-"
            linewidth = SQUARE_PAPER_STRESS_OPT_LINEWIDTH
            alpha = 0.98
            zorder = 5.2 if side == "inner" else 5.3
        else:
            display_label = "FEA inner stress" if side == "inner" else "FEA outer stress"
            color = (
                SQUARE_PAPER_STRESS_FEA_INNER_COLOR
                if side == "inner"
                else SQUARE_PAPER_STRESS_FEA_OUTER_COLOR
            )
            linestyle = (0, (1.4, 1.2)) if side == "inner" else (0, (4.0, 1.4))
            linewidth = SQUARE_PAPER_STRESS_FEA_LINEWIDTH
            alpha = SQUARE_PAPER_STRESS_FEA_ALPHA
            zorder = 4.0 if side == "inner" else 4.1
        styled_curves.append(
            {
                "series": series,
                "side": side,
                "role": role,
                "label": display_label,
                "s_norm": s_norm,
                "stress_mpa": stress_mpa,
                "color": color,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "alpha": alpha,
                "zorder": zorder,
            }
        )
    for curve in sorted(styled_curves, key=lambda item: float(item["zorder"])):
        ax.plot(
            curve["s_norm"],
            curve["stress_mpa"],
            linestyle=curve["linestyle"],
            color=str(curve["color"]),
            lw=float(curve["linewidth"]),
            alpha=float(curve["alpha"]),
            solid_capstyle="round",
            dash_capstyle="round",
            label=str(curve["label"]),
            zorder=float(curve["zorder"]),
        )
    ax.set_xlabel("Normalized Arc Length", fontsize=SQUARE_PAPER_TEXT_PT)
    ax.set_ylabel("Stress (MPa)", fontsize=SQUARE_PAPER_TEXT_PT)
    ax.tick_params(axis="both", labelsize=SQUARE_PAPER_TEXT_PT)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
    stress_arrays = [
        np.asarray(curve["stress_mpa"], dtype=float)
        for curve in styled_curves
        if np.asarray(curve["stress_mpa"], dtype=float).size > 0
    ]
    if stress_arrays:
        finite_stress = np.concatenate([arr[np.isfinite(arr)] for arr in stress_arrays if np.isfinite(arr).any()])
        if finite_stress.size > 0:
            y_max = float(np.max(finite_stress))
            y_top_pad = max(1.5, SQUARE_PAPER_STRESS_TOP_PADDING_FRAC * max(y_max, 10.0))
            ax.set_ylim(0.0, y_max + y_top_pad)
    ax.set_xlim(-0.015, 1.015)
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    if draw_final_value_guides:
        endpoint_guides = [
            guide
            for guide in [
                endpoint_guide(
                    curve["s_norm"],
                    curve["stress_mpa"],
                    color=str(curve["color"]),
                    linewidth=0.55,
                    alpha=0.55,
                    zorder=float(curve["zorder"]) - 0.05,
                )
                for curve in styled_curves
            ]
            if guide is not None
        ]
        draw_endpoint_guides(ax, endpoint_guides)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        label_to_handle: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            label_to_handle[label] = handle
        legend_order = [
            "Optimizer inner stress",
            "Optimizer outer stress",
            "FEA outer stress",
            "FEA inner stress",
        ]
        ordered_labels = [label for label in legend_order if label in label_to_handle]
        ordered_handles = [label_to_handle[label] for label in ordered_labels]
        legend_kwargs = dict(
            loc=legend_loc,
            fontsize=legend_fontsize_pt,
            ncol=legend_ncol,
            framealpha=0.92,
        )
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        if legend_inside_axes:
            legend = ax.legend(ordered_handles, ordered_labels, **legend_kwargs)
        else:
            legend = fig.legend(ordered_handles, ordered_labels, **legend_kwargs)
        legend.get_frame().set_edgecolor("0.72")
        legend.get_frame().set_linewidth(0.55)
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def render_square_stress_comparison_plot(
    *,
    spring: Spring,
    solidworks_stress_overlays: list[dict[str, object]] | None = None,
    figure_title: str,
) -> np.ndarray:
    return render_square_stress_curve_bundle_plot(
        curve_bundles=build_square_stress_curve_bundle(
            spring=spring,
            solidworks_stress_overlays=solidworks_stress_overlays,
        ),
        figure_title=figure_title,
    )


def render_comparison(
    *,
    video_frame_bgr: np.ndarray,
    record: dict,
    comparison_frame: pd.Series,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    solidworks_overlays: list[dict[str, object]] | None = None,
    figure_title: str,
    video_title: str,
    angle_label: str,
    summary_lines: list[str] | None = None,
    include_force_lines: bool = False,
) -> np.ndarray:
    rgb = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(14, 8), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    ax_img.imshow(rgb)
    ax_img.set_title(video_title, fontsize=12)
    ax_img.axis("off")

    draw_comparison_curve(
        ax_plot,
        record=record,
        comparison_frame=comparison_frame,
        optimizer_s_norm=optimizer_s_norm,
        optimizer_curve_mm=optimizer_curve_mm,
        tracker_s_norm=tracker_s_norm,
        tracker_curve_mm=tracker_curve_mm,
        figure_title=figure_title,
        angle_label=angle_label,
        title_fontsize=12,
        axis_label_fontsize=10,
        tick_fontsize=10,
        legend_fontsize=9,
        annotation_fontsize=8,
        info_fontsize=9,
        show_info_text=True,
        optimizer_label=optimizer_key_label(record, "optimizer sweep state"),
        solidworks_overlays=solidworks_overlays,
        summary_lines=summary_lines,
        include_force_lines=include_force_lines,
    )

    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def render_curve_only_comparison(
    *,
    record: dict,
    comparison_frame: pd.Series,
    optimizer_s_norm: np.ndarray,
    optimizer_curve_mm: np.ndarray,
    tracker_s_norm: np.ndarray,
    tracker_curve_mm: np.ndarray,
    tracker_rows: pd.DataFrame | None = None,
    solidworks_overlays: list[dict[str, object]] | None = None,
    figure_title: str,
    angle_label: str,
    summary_lines: list[str] | None = None,
    include_force_lines: bool = False,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=150, constrained_layout=True)
    draw_comparison_curve(
        ax,
        record=record,
        comparison_frame=comparison_frame,
        optimizer_s_norm=optimizer_s_norm,
        optimizer_curve_mm=optimizer_curve_mm,
        tracker_s_norm=tracker_s_norm,
        tracker_curve_mm=tracker_curve_mm,
        tracker_rows=tracker_rows,
        figure_title=figure_title,
        angle_label=angle_label,
        title_fontsize=12,
        axis_label_fontsize=10,
        tick_fontsize=10,
        legend_fontsize=9,
        annotation_fontsize=8,
        info_fontsize=9,
        show_info_text=True,
        optimizer_label=optimizer_key_label(record, "optimizer sweep state"),
        solidworks_overlays=solidworks_overlays,
        summary_lines=summary_lines,
        include_force_lines=include_force_lines,
    )
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def render_snapshot_panel_for_frame(
    *,
    input_dir: Path | None,
    tracked_video: Path | None,
    tracker_df: pd.DataFrame,
    comparison_frame: pd.Series,
    include_deformed_vector: bool,
    half_extent_px: int | None = None,
    annotation_lines: list[str] | None = None,
    preferred_dot_id: int | None = None,
) -> np.ndarray:
    frame_idx = int(comparison_frame["frame"])
    tracker_rows = tracker_df[tracker_df["frame"] == frame_idx].sort_values("dot_id")
    if tracker_rows.empty:
        raise ValueError(f"Tracker frame {frame_idx} has no rows for snapshot rendering.")
    snapshot_frame_bgr = resolve_snapshot_source_frame(
        input_dir=input_dir,
        tracked_video=tracked_video,
        comparison_frame=comparison_frame,
    )
    return render_arbor_vector_snapshot_panel(
        video_frame_bgr=snapshot_frame_bgr,
        tracker_rows=tracker_rows,
        include_deformed_vector=include_deformed_vector,
        half_extent_px=half_extent_px,
        annotation_lines=annotation_lines,
        preferred_dot_id=preferred_dot_id,
    )


def tracker_vector_history_half_extent_px(
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
    *,
    preferred_dot_id: int,
) -> int:
    half_extent_px = 0
    for frame_idx in frame_table["frame"].to_numpy(dtype=int):
        tracker_rows = tracker_df[tracker_df["frame"] == int(frame_idx)].sort_values("dot_id")
        if tracker_rows.empty:
            continue
        half_extent_px = max(
            half_extent_px,
            tracker_vector_crop_half_extent_px(
                tracker_rows,
                include_deformed_vector=True,
                preferred_dot_id=preferred_dot_id,
            ),
        )
    return max(half_extent_px, VECTOR_SNAPSHOT_MIN_HALF_EXTENT_PX)


def render_node0_vector_history_frames(
    *,
    input_dir: Path | None,
    tracked_video: Path | None,
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
    preferred_dot_id: int,
    half_extent_px: int | None = None,
) -> list[np.ndarray]:
    if frame_table.empty:
        return []
    if half_extent_px is None:
        half_extent_px = tracker_vector_history_half_extent_px(
            tracker_df,
            frame_table,
            preferred_dot_id=preferred_dot_id,
        )

    frames_rgb: list[np.ndarray] = []
    for _, comparison_frame in frame_table.sort_values("frame").iterrows():
        frame_idx = int(comparison_frame["frame"])
        tracker_rows = tracker_df[tracker_df["frame"] == frame_idx].sort_values("dot_id")
        if tracker_rows.empty:
            continue
        tracker_row = tracker_first_node_row(tracker_rows, preferred_dot_id=preferred_dot_id)
        annotation_lines = build_node_vector_history_annotation_lines(
            tracker_row,
            tracker_rows,
            preferred_dot_id=preferred_dot_id,
        )
        frame_rgb = render_snapshot_panel_for_frame(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            comparison_frame=comparison_frame,
            include_deformed_vector=True,
            half_extent_px=half_extent_px,
            annotation_lines=annotation_lines,
            preferred_dot_id=preferred_dot_id,
        )
        frames_rgb.append(frame_rgb)
    return frames_rgb


def render_snapshot_pair_figure(
    *,
    left_image_rgb: np.ndarray,
    right_image_rgb: np.ndarray,
    figure_title: str,
    left_title: str,
    right_title: str,
) -> np.ndarray:
    fig = plt.figure(figsize=(8.8, 4.8), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.03)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    for ax, image_rgb, title in (
        (ax_left, left_image_rgb, left_title),
        (ax_right, right_image_rgb, right_title),
    ):
        ax.imshow(image_rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(figure_title, fontsize=12)
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def resize_image_to_fit(
    image_rgb: np.ndarray,
    *,
    max_width: int,
    max_height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot resize an empty image.")
    scale = min(float(max_width) / float(width), float(max_height) / float(height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=interpolation)


def blank_triptych_panel(message: str) -> np.ndarray:
    panel = np.full((TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX, 3), TRIPTYCH_BG_RGB, dtype=np.uint8)
    cv2.rectangle(
        panel,
        (1, 1),
        (TRIPTYCH_PANEL_SIZE_PX - 2, TRIPTYCH_PANEL_SIZE_PX - 2),
        TRIPTYCH_BORDER_RGB,
        2,
        cv2.LINE_AA,
    )
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        lines = ["No data"]
    line_height = 34
    total_height = line_height * len(lines)
    baseline_y = (TRIPTYCH_PANEL_SIZE_PX - total_height) // 2 + 24
    for line in lines:
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        origin = ((TRIPTYCH_PANEL_SIZE_PX - size[0]) // 2, baseline_y)
        cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, TRIPTYCH_MUTED_TEXT_RGB, 2, cv2.LINE_AA)
        baseline_y += line_height
    return panel


def extract_force_screen_crop_bgr(
    video_frame_bgr: np.ndarray,
    comparison_frame: pd.Series,
) -> np.ndarray | None:
    bbox_x = _series_float_value(comparison_frame, "force_bbox_x")
    bbox_y = _series_float_value(comparison_frame, "force_bbox_y")
    bbox_w = _series_float_value(comparison_frame, "force_bbox_w")
    bbox_h = _series_float_value(comparison_frame, "force_bbox_h")
    if bbox_x is None or bbox_y is None or bbox_w is None or bbox_h is None:
        return None

    x = int(round(bbox_x))
    y = int(round(bbox_y))
    w = max(int(round(bbox_w)), 1)
    h = max(int(round(bbox_h)), 1)
    height, width = video_frame_bgr.shape[:2]
    x0 = int(np.clip(x, 0, max(width - 1, 0)))
    y0 = int(np.clip(y, 0, max(height - 1, 0)))
    x1 = int(np.clip(x + w, x0 + 1, width))
    y1 = int(np.clip(y + h, y0 + 1, height))
    crop = video_frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(
        crop,
        (FORCE_SCREEN_CROP_WIDTH_PX, FORCE_SCREEN_CROP_HEIGHT_PX),
        interpolation=cv2.INTER_LINEAR,
    )


def build_force_meter_triptych_lines(comparison_frame: pd.Series) -> tuple[str, list[str]]:
    used_force_n = _series_float_value(comparison_frame, "tracker_force_n")
    used_mass_kg = _series_float_value(comparison_frame, "tracker_mass_kg")
    raw_force_n = _series_float_value(comparison_frame, "tracker_force_n_raw")
    raw_mass_kg = _series_float_value(comparison_frame, "tracker_mass_kg_raw")
    raw_confidence = _series_float_value(comparison_frame, "tracker_force_confidence_raw")
    screen_visible = _series_bool_value(comparison_frame, "force_screen_visible")
    force_digits = _series_string_value(comparison_frame, "force_digits")
    force_reason = _series_string_value(comparison_frame, "force_reason")

    if used_force_n is not None:
        headline = f"{used_force_n:.3f} N"
    elif raw_force_n is not None:
        headline = f"OCR {raw_force_n:.3f} N"
    elif screen_visible is False:
        headline = "Screen not visible"
    else:
        headline = "Force unreadable"

    lines: list[str] = []
    if used_mass_kg is not None:
        lines.append(f"Used mass: {_format_mass_value(used_mass_kg)}")
    if raw_mass_kg is not None and (used_mass_kg is None or abs(raw_mass_kg - used_mass_kg) > 1e-6):
        lines.append(f"OCR mass: {_format_mass_value(raw_mass_kg)}")
    if force_digits is not None:
        lines.append(f"Digits: {_format_force_digits_value(force_digits)}")
    if raw_confidence is not None:
        lines.append(f"Confidence: {_format_confidence_value(raw_confidence)}")
    if force_reason is not None:
        lines.append(f"Status: {force_reason}")
    elif screen_visible is False:
        lines.append("Status: screen not visible")
    elif not lines:
        lines.append("Status: no force-meter metadata")
    return headline, lines


def render_force_meter_triptych_panel(
    *,
    video_frame_bgr: np.ndarray,
    comparison_frame: pd.Series,
) -> np.ndarray:
    panel = np.full((TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX, 3), TRIPTYCH_BG_RGB, dtype=np.uint8)
    cv2.rectangle(
        panel,
        (1, 1),
        (TRIPTYCH_PANEL_SIZE_PX - 2, TRIPTYCH_PANEL_SIZE_PX - 2),
        TRIPTYCH_BORDER_RGB,
        2,
        cv2.LINE_AA,
    )

    box_left = FORCE_PANEL_MARGIN_PX
    box_top = FORCE_PANEL_MARGIN_PX
    box_right = TRIPTYCH_PANEL_SIZE_PX - FORCE_PANEL_MARGIN_PX
    box_bottom = box_top + FORCE_PANEL_SCREEN_BOX_HEIGHT_PX
    cv2.rectangle(panel, (box_left, box_top), (box_right, box_bottom), FORCE_PANEL_SCREEN_BG_RGB, -1, cv2.LINE_AA)
    cv2.rectangle(panel, (box_left, box_top), (box_right, box_bottom), TRIPTYCH_BORDER_RGB, 1, cv2.LINE_AA)

    screen_crop_bgr = extract_force_screen_crop_bgr(video_frame_bgr, comparison_frame)
    if screen_crop_bgr is not None:
        screen_rgb = cv2.cvtColor(screen_crop_bgr, cv2.COLOR_BGR2RGB)
        screen_rgb = resize_image_to_fit(
            screen_rgb,
            max_width=box_right - box_left - 18,
            max_height=box_bottom - box_top - 18,
            interpolation=cv2.INTER_LINEAR,
        )
        paste_x = box_left + ((box_right - box_left) - screen_rgb.shape[1]) // 2
        paste_y = box_top + ((box_bottom - box_top) - screen_rgb.shape[0]) // 2
        panel[paste_y:paste_y + screen_rgb.shape[0], paste_x:paste_x + screen_rgb.shape[1]] = screen_rgb
    else:
        for idx, line in enumerate(("No force-meter crop", "in tracker CSV")):
            size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, FORCE_PANEL_TITLE_SCALE, 2)
            origin = (
                box_left + ((box_right - box_left) - size[0]) // 2,
                box_top + 94 + idx * 34,
            )
            cv2.putText(panel, line, origin, cv2.FONT_HERSHEY_SIMPLEX, FORCE_PANEL_TITLE_SCALE, (230, 230, 230), 2, cv2.LINE_AA)

    headline, lines = build_force_meter_triptych_lines(comparison_frame)
    headline_origin = (FORCE_PANEL_MARGIN_PX, box_bottom + 56)
    cv2.putText(
        panel,
        headline,
        headline_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        FORCE_PANEL_VALUE_SCALE,
        TRIPTYCH_TEXT_RGB,
        2,
        cv2.LINE_AA,
    )
    baseline_y = headline_origin[1] + 42
    for line in lines:
        cv2.putText(
            panel,
            line,
            (FORCE_PANEL_MARGIN_PX, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FORCE_PANEL_TEXT_SCALE,
            TRIPTYCH_MUTED_TEXT_RGB,
            1,
            cv2.LINE_AA,
        )
        baseline_y += FORCE_PANEL_TEXT_LINE_SPACING_PX
    return panel


def moment_vs_visual_angle_limits(
    *,
    optimizer_x: np.ndarray,
    optimizer_y: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = np.concatenate((np.asarray(optimizer_x, dtype=float), np.asarray(tracker_x, dtype=float)))
    y_values = np.concatenate((np.asarray(optimizer_y, dtype=float), np.asarray(tracker_y, dtype=float)))
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if x_values.size == 0:
        x_values = np.asarray([0.0, 1.0], dtype=float)
    if y_values.size == 0:
        y_values = np.asarray([0.0, 1.0], dtype=float)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    x_pad = max(0.8, 0.04 * max(x_max - x_min, 1.0))
    y_pad = max(0.04, 0.07 * max(y_max - y_min, 0.1))
    return (max(0.0, x_min - x_pad), x_max + x_pad), (max(0.0, y_min - y_pad), y_max + y_pad)


def render_live_moment_vs_visual_angle_panel(
    *,
    optimizer_x: np.ndarray,
    optimizer_y: np.ndarray,
    history_x: np.ndarray,
    history_y: np.ndarray,
    current_x: float | None,
    current_y: float | None,
    time_s: float | None,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> np.ndarray:
    fig, ax = plt.subplots(
        figsize=(TRIPTYCH_PANEL_SIZE_PX / LIVE_PLOT_DPI, TRIPTYCH_PANEL_SIZE_PX / LIVE_PLOT_DPI),
        dpi=LIVE_PLOT_DPI,
        constrained_layout=True,
    )
    optimizer_x_arr = np.asarray(optimizer_x, dtype=float)
    optimizer_y_arr = np.asarray(optimizer_y, dtype=float)
    optimizer_order = np.argsort(optimizer_x_arr)
    ax.plot(
        optimizer_x_arr[optimizer_order],
        optimizer_y_arr[optimizer_order],
        "-",
        lw=1.6,
        color="black",
        label="Sweep",
        zorder=3.5,
    )

    history_x_arr = np.asarray(history_x, dtype=float)
    history_y_arr = np.asarray(history_y, dtype=float)
    finite_history = np.isfinite(history_x_arr) & np.isfinite(history_y_arr)
    history_x_arr = history_x_arr[finite_history]
    history_y_arr = history_y_arr[finite_history]
    if history_x_arr.size:
        ax.scatter(
            history_x_arr,
            history_y_arr,
            s=16,
            color="crimson",
            alpha=0.25,
            edgecolors="none",
            label="Tracked samples",
            zorder=4.2,
        )
        trend_x, trend_y = _binned_median_curve(history_x_arr, history_y_arr)
        if trend_x.size:
            ax.plot(
                trend_x,
                trend_y,
                "-",
                lw=1.6,
                color="crimson",
                label="Tracked median",
                zorder=4.8,
            )

    if current_x is not None and current_y is not None:
        ax.axvline(current_x, color="royalblue", lw=0.9, ls="--", alpha=0.35, zorder=3.2)
        ax.axhline(current_y, color="royalblue", lw=0.9, ls="--", alpha=0.35, zorder=3.2)
        ax.scatter(
            [current_x],
            [current_y],
            s=68,
            facecolors="white",
            edgecolors="royalblue",
            linewidths=1.8,
            label="Current frame",
            zorder=5.4,
        )
        lines = [
            f"t = {time_s:.3f} s" if time_s is not None else "t = n/a",
            f"{ANGLE_MEASUREMENT_LABEL.lower()} = {current_x:.3f} deg",
            f"moment = {current_y:.4f} N*m",
        ]
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.65", alpha=0.94),
            zorder=6.0,
        )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel(f"{ANGLE_MEASUREMENT_LABEL} (deg)", fontsize=8.0)
    ax.set_ylabel("Moment (N*m)", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.94)
    image = figure_to_rgb(fig)
    plt.close(fig)
    if image.shape[:2] != (TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX):
        image = cv2.resize(
            image,
            (TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
    return image


def compose_triptych_frame(
    *,
    force_panel_rgb: np.ndarray,
    arbor_panel_rgb: np.ndarray,
    plot_panel_rgb: np.ndarray,
    comparison_frame: pd.Series,
) -> np.ndarray:
    force_panel = cv2.resize(force_panel_rgb, (TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX), interpolation=cv2.INTER_CUBIC)
    arbor_panel = cv2.resize(arbor_panel_rgb, (TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX), interpolation=cv2.INTER_CUBIC)
    plot_panel = cv2.resize(plot_panel_rgb, (TRIPTYCH_PANEL_SIZE_PX, TRIPTYCH_PANEL_SIZE_PX), interpolation=cv2.INTER_CUBIC)

    width = (
        2 * TRIPTYCH_MARGIN_PX
        + 3 * TRIPTYCH_PANEL_SIZE_PX
        + 2 * TRIPTYCH_GAP_PX
    )
    height = (
        2 * TRIPTYCH_MARGIN_PX
        + TRIPTYCH_HEADER_HEIGHT_PX
        + TRIPTYCH_LABEL_HEIGHT_PX
        + TRIPTYCH_PANEL_SIZE_PX
    )
    canvas = np.full((height, width, 3), TRIPTYCH_BG_RGB, dtype=np.uint8)

    title_y = TRIPTYCH_MARGIN_PX + 24
    subtitle_y = title_y + 28
    header_text = "Tracker sanity check"
    subtitle = (
        f"frame {int(comparison_frame['frame'])}"
        f" | window {int(comparison_frame['frame_in_window'])}"
        f" | t = {float(comparison_frame['time_s']):.3f} s"
    )
    cv2.putText(canvas, header_text, (TRIPTYCH_MARGIN_PX, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, TRIPTYCH_TEXT_RGB, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (TRIPTYCH_MARGIN_PX, subtitle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TRIPTYCH_MUTED_TEXT_RGB, 1, cv2.LINE_AA)

    panel_top = TRIPTYCH_MARGIN_PX + TRIPTYCH_HEADER_HEIGHT_PX + TRIPTYCH_LABEL_HEIGHT_PX
    labels = ("Force meter", "Arbor crop", f"Moment vs {ANGLE_MEASUREMENT_LABEL.lower()}")
    for panel_idx, label in enumerate(labels):
        panel_left = TRIPTYCH_MARGIN_PX + panel_idx * (TRIPTYCH_PANEL_SIZE_PX + TRIPTYCH_GAP_PX)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 1)
        label_x = panel_left + (TRIPTYCH_PANEL_SIZE_PX - size[0]) // 2
        label_y = TRIPTYCH_MARGIN_PX + TRIPTYCH_HEADER_HEIGHT_PX + 18
        cv2.putText(canvas, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, TRIPTYCH_TEXT_RGB, 1, cv2.LINE_AA)

    first_left = TRIPTYCH_MARGIN_PX
    second_left = first_left + TRIPTYCH_PANEL_SIZE_PX + TRIPTYCH_GAP_PX
    third_left = second_left + TRIPTYCH_PANEL_SIZE_PX + TRIPTYCH_GAP_PX
    canvas[panel_top:panel_top + TRIPTYCH_PANEL_SIZE_PX, first_left:first_left + TRIPTYCH_PANEL_SIZE_PX] = force_panel
    canvas[panel_top:panel_top + TRIPTYCH_PANEL_SIZE_PX, second_left:second_left + TRIPTYCH_PANEL_SIZE_PX] = arbor_panel
    canvas[panel_top:panel_top + TRIPTYCH_PANEL_SIZE_PX, third_left:third_left + TRIPTYCH_PANEL_SIZE_PX] = plot_panel
    return canvas


def iter_force_arbor_moment_triptych_frames(
    *,
    tracked_video: Path,
    tracker_df: pd.DataFrame,
    frame_table: pd.DataFrame,
    preferred_dot_id: int,
    arbor_half_extent_px: int,
    optimizer_x: np.ndarray,
    optimizer_y: np.ndarray,
) -> Iterable[np.ndarray]:
    ordered_frames = frame_table.sort_values("frame_in_window").reset_index(drop=True)
    tracker_rows_by_frame = {
        int(frame): grp.sort_values("dot_id").copy()
        for frame, grp in tracker_df.groupby("frame", sort=True)
    }
    tracker_history_table = ordered_frames.copy()
    tracker_history_table["abs_visual_angle_deg"] = tracker_history_table["visual_angle_deg"].abs()
    tracker_history_table = tracker_history_table[
        pd.to_numeric(tracker_history_table["tracker_moment_nm"], errors="coerce").notna()
        & pd.to_numeric(tracker_history_table["abs_visual_angle_deg"], errors="coerce").notna()
    ]
    x_limits, y_limits = moment_vs_visual_angle_limits(
        optimizer_x=np.asarray(optimizer_x, dtype=float),
        optimizer_y=np.asarray(optimizer_y, dtype=float),
        tracker_x=tracker_history_table["abs_visual_angle_deg"].to_numpy(dtype=float),
        tracker_y=tracker_history_table["tracker_moment_nm"].to_numpy(dtype=float),
    )

    history_x: list[float] = []
    history_y: list[float] = []
    cap = cv2.VideoCapture(str(tracked_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open tracked video: {tracked_video}")
    current_video_idx = -1
    try:
        for _, comparison_frame in ordered_frames.iterrows():
            target_video_idx = int(comparison_frame["frame_in_window"])
            if target_video_idx < current_video_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_idx)
                current_video_idx = target_video_idx - 1

            video_frame_bgr: np.ndarray | None = None
            while current_video_idx < target_video_idx:
                ok, video_frame_bgr = cap.read()
                current_video_idx += 1
                if not ok or video_frame_bgr is None:
                    raise RuntimeError(
                        f"Could not read tracked video frame {target_video_idx} from {tracked_video}."
                    )

            current_angle_deg = _comparison_visual_angle_deg(comparison_frame)
            current_moment_nm = _series_float_value(comparison_frame, "tracker_moment_nm")
            current_x = None
            current_y = None
            if current_angle_deg is not None and current_moment_nm is not None:
                current_x = abs(float(current_angle_deg))
                current_y = float(current_moment_nm)
                history_x.append(current_x)
                history_y.append(current_y)

            force_panel_rgb = render_force_meter_triptych_panel(
                video_frame_bgr=video_frame_bgr,
                comparison_frame=comparison_frame,
            )

            tracker_rows = tracker_rows_by_frame.get(int(comparison_frame["frame"]))
            if tracker_rows is None or tracker_rows.empty:
                arbor_panel_rgb = blank_triptych_panel("No arbor rows")
            else:
                try:
                    tracker_row = tracker_first_node_row(
                        tracker_rows,
                        preferred_dot_id=preferred_dot_id,
                    )
                    annotation_lines = build_node_vector_history_annotation_lines(
                        tracker_row,
                        tracker_rows,
                        preferred_dot_id=preferred_dot_id,
                    )
                    arbor_panel_rgb = render_arbor_vector_snapshot_panel(
                        video_frame_bgr=video_frame_bgr,
                        tracker_rows=tracker_rows,
                        include_deformed_vector=True,
                        half_extent_px=arbor_half_extent_px,
                        annotation_lines=annotation_lines,
                        preferred_dot_id=preferred_dot_id,
                    )
                except ValueError as exc:
                    arbor_panel_rgb = blank_triptych_panel(str(exc))

            plot_panel_rgb = render_live_moment_vs_visual_angle_panel(
                optimizer_x=np.asarray(optimizer_x, dtype=float),
                optimizer_y=np.asarray(optimizer_y, dtype=float),
                history_x=np.asarray(history_x, dtype=float),
                history_y=np.asarray(history_y, dtype=float),
                current_x=current_x,
                current_y=current_y,
                time_s=float(comparison_frame["time_s"]) if "time_s" in comparison_frame.index else None,
                x_limits=x_limits,
                y_limits=y_limits,
            )
            yield compose_triptych_frame(
                force_panel_rgb=force_panel_rgb,
                arbor_panel_rgb=arbor_panel_rgb,
                plot_panel_rgb=plot_panel_rgb,
                comparison_frame=comparison_frame,
            )
    finally:
        cap.release()


def _binned_median_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    max_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if x_arr.size < 4 or float(np.max(x_arr)) <= float(np.min(x_arr)) + 1e-12:
        order = np.argsort(x_arr)
        return x_arr[order], y_arr[order]

    n_bins = min(max_bins, max(10, x_arr.size // 6))
    bin_edges = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), n_bins + 1)
    if not np.all(np.diff(bin_edges) > 0):
        order = np.argsort(x_arr)
        return x_arr[order], y_arr[order]

    bin_ids = np.digitize(x_arr, bin_edges[1:-1], right=False)
    x_curve: list[float] = []
    y_curve: list[float] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        x_curve.append(float(np.median(x_arr[mask])))
        y_curve.append(float(np.median(y_arr[mask])))
    return np.asarray(x_curve, dtype=float), np.asarray(y_curve, dtype=float)


def _linear_fit_params(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, np.ndarray | float] | None:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < 2:
        return None
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if float(np.max(x_arr)) <= float(np.min(x_arr)) + 1e-12:
        return None
    slope_per_deg, intercept = np.polyfit(x_arr, y_arr, 1)
    y_fit = slope_per_deg * x_arr + intercept
    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    sse = float(np.sum((y_arr - y_fit) ** 2))
    r2 = 1.0 if sst <= 1e-12 else 1.0 - sse / sst
    return {
        "x": x_arr,
        "y_fit": y_fit,
        "slope_per_deg": float(slope_per_deg),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def _detect_linear_tail_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    min_points: int = 6,
) -> dict[str, np.ndarray | float | int] | None:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < max(min_points, 3):
        fit = _linear_fit_params(x_arr, y_arr)
        if fit is None:
            return None
        return {
            **fit,
            "start_idx": 0,
            "start_x": float(fit["x"][0]),
            "start_y": float((slope := float(fit["slope_per_deg"])) * float(fit["x"][0]) + float(fit["intercept"])),
        }

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    best: dict[str, np.ndarray | float | int] | None = None
    max_start = x_arr.size - min_points
    for start_idx in range(1, max_start + 1):
        x_tail = x_arr[start_idx:]
        y_tail = y_arr[start_idx:]
        fit = _linear_fit_params(x_tail, y_tail)
        if fit is None:
            continue
        slope_per_deg = float(fit["slope_per_deg"])
        if slope_per_deg <= 0.0:
            continue
        baseline = float(np.median(y_arr[:start_idx]))
        sse_prefix = float(np.sum((y_arr[:start_idx] - baseline) ** 2))
        sse_tail = float(np.sum((y_tail - np.asarray(fit["y_fit"], dtype=float)) ** 2))
        candidate = {
            **fit,
            "start_idx": start_idx,
            "start_x": float(x_arr[start_idx]),
            "start_y": float(y_arr[start_idx]),
            "score": sse_prefix + sse_tail,
        }
        if best is None or float(candidate["score"]) < float(best["score"]):
            best = candidate
    if best is not None:
        return best
    fit = _linear_fit_params(x_arr, y_arr)
    if fit is None:
        return None
    return {
        **fit,
        "start_idx": 0,
        "start_x": float(x_arr[0]),
        "start_y": float(y_arr[0]),
    }


def render_xy_overlay_plot(
    *,
    optimizer_x: np.ndarray,
    optimizer_y: np.ndarray,
    tracker_x: np.ndarray,
    tracker_y: np.ndarray,
    figure_title: str,
    x_label: str,
    y_label: str,
    optimizer_label: str = "optimizer sweep",
    tracker_label: str = "tracker",
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150, constrained_layout=True)
    optimizer_x_arr = np.asarray(optimizer_x, dtype=float)
    optimizer_y_arr = np.asarray(optimizer_y, dtype=float)
    tracker_x_arr = np.asarray(tracker_x, dtype=float)
    tracker_y_arr = np.asarray(tracker_y, dtype=float)
    optimizer_order = np.argsort(optimizer_x_arr)
    ax.plot(
        optimizer_x_arr[optimizer_order],
        optimizer_y_arr[optimizer_order],
        "o-",
        lw=1.5,
        ms=3.5,
        color="black",
        label=optimizer_label,
        zorder=4,
    )
    finite_tracker = np.isfinite(tracker_x_arr) & np.isfinite(tracker_y_arr)
    ax.scatter(
        tracker_x_arr[finite_tracker],
        tracker_y_arr[finite_tracker],
        s=16,
        color="crimson",
        alpha=0.35,
        edgecolors="none",
        label=f"{tracker_label} samples",
        zorder=4.8,
    )
    tracker_curve_x, tracker_curve_y = _binned_median_curve(
        tracker_x_arr[finite_tracker],
        tracker_y_arr[finite_tracker],
    )
    ax.plot(
        tracker_curve_x,
        tracker_curve_y,
        "-",
        lw=2.0,
        color="crimson",
        label=f"{tracker_label} median trend",
        zorder=5.2,
    )
    ax.set_title(figure_title, fontsize=12)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def ensure_plot_cache_dir(output_dir: Path) -> Path:
    plot_cache_dir = output_dir / "plot_cache"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    return plot_cache_dir


def save_curve_bundle_csv(
    path: Path,
    curves: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
) -> None:
    rows: list[dict[str, object]] = []
    for curve in curves:
        x_values = np.asarray(curve[x_key], dtype=float)
        y_values = np.asarray(curve[y_key], dtype=float)
        n_values = min(len(x_values), len(y_values))
        extra_fields: dict[str, object] = {}
        for key, value in curve.items():
            if key in {"series", "label", x_key, y_key}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                extra_fields[key] = value
        for idx in range(n_values):
            row = {
                "series": str(curve.get("series", curve.get("label", "series"))),
                "label": str(curve.get("label", curve.get("series", "series"))),
                x_key: float(x_values[idx]),
                y_key: float(y_values[idx]),
            }
            row.update(extra_fields)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_xy_bundle_csv(path: Path, series_rows: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for series in series_rows:
        x_values = np.asarray(series["x"], dtype=float)
        y_values = np.asarray(series["y"], dtype=float)
        n_values = min(len(x_values), len(y_values))
        for idx in range(n_values):
            rows.append(
                {
                    "series": str(series["series"]),
                    "label": str(series.get("label", series["series"])),
                    "x": float(x_values[idx]),
                    "y": float(y_values[idx]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_curve_bundle_csv(
    path: Path,
    *,
    x_key: str,
    y_key: str,
) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    required = {"series", "label", x_key, y_key}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required cache columns: {sorted(missing)}")
    curves: list[dict[str, object]] = []
    for series, grp in df.groupby("series", sort=False):
        curve: dict[str, object] = {
            "series": str(series),
            "label": str(grp["label"].iloc[0]),
            x_key: grp[x_key].to_numpy(dtype=float),
            y_key: grp[y_key].to_numpy(dtype=float),
        }
        for column in grp.columns:
            if column in {"series", "label", x_key, y_key}:
                continue
            first_value = grp[column].iloc[0]
            if pd.isna(first_value):
                continue
            curve[column] = first_value
        curves.append(curve)
    return curves


def load_xy_bundle_csv(path: Path) -> list[dict[str, object]]:
    return load_curve_bundle_csv(path, x_key="x", y_key="y")


def save_rgb_image(
    path: Path,
    image_rgb: np.ndarray,
    *,
    dpi: float | None = None,
) -> None:
    save_kwargs: dict[str, object] = {}
    if dpi is not None:
        save_kwargs["dpi"] = (float(dpi), float(dpi))
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path, **save_kwargs)


def save_gif(frames_rgb: list[np.ndarray], path: Path, fps: float) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    duration_ms = max(int(round(1000.0 / max(fps, 1e-6))), 1)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4(frames_rgb: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("No frames provided for MP4 export.")
    height, width = frames_rgb[0].shape[:2]

    # H.264/yuv420p is far more widely playable than OpenCV's mp4v output.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        ffmpeg_width = width + (width % 2)
        ffmpeg_height = height + (height % 2)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{ffmpeg_width}x{ffmpeg_height}",
            "-r",
            f"{float(fps):.8g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc.stdin is not None
            for frame in frames_rgb:
                frame_rgb = np.asarray(frame, dtype=np.uint8)
                if frame_rgb.shape[:2] != (height, width):
                    raise ValueError("All MP4 frames must share the same dimensions.")
                if ffmpeg_width != width or ffmpeg_height != height:
                    frame_rgb = cv2.copyMakeBorder(
                        frame_rgb,
                        0,
                        ffmpeg_height - height,
                        0,
                        ffmpeg_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
            rc = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if rc == 0:
            return
        if path.exists():
            path.unlink()
        print(f"[warn] ffmpeg MP4 export failed; falling back to OpenCV writer.\n{stderr.strip()}")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        for frame in frames_rgb:
            frame_rgb = np.asarray(frame, dtype=np.uint8)
            if frame_rgb.shape[:2] != (height, width):
                raise ValueError("All MP4 frames must share the same dimensions.")
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_mp4_stream(frames_rgb: Iterable[np.ndarray], path: Path, fps: float) -> None:
    frame_iter = iter(frames_rgb)
    try:
        first_frame = np.asarray(next(frame_iter), dtype=np.uint8)
    except StopIteration as exc:
        raise ValueError("No frames provided for MP4 export.") from exc

    height, width = first_frame.shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        ffmpeg_width = width + (width % 2)
        ffmpeg_height = height + (height % 2)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{ffmpeg_width}x{ffmpeg_height}",
            "-r",
            f"{float(fps):.8g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc.stdin is not None
            frame_rgb = np.asarray(first_frame, dtype=np.uint8)
            if ffmpeg_width != width or ffmpeg_height != height:
                frame_rgb = cv2.copyMakeBorder(
                    frame_rgb,
                    0,
                    ffmpeg_height - height,
                    0,
                    ffmpeg_width - width,
                    cv2.BORDER_REPLICATE,
                )
            proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
            for frame in frame_iter:
                frame_rgb = np.asarray(frame, dtype=np.uint8)
                if frame_rgb.shape[:2] != (height, width):
                    raise ValueError("All MP4 frames must share the same dimensions.")
                if ffmpeg_width != width or ffmpeg_height != height:
                    frame_rgb = cv2.copyMakeBorder(
                        frame_rgb,
                        0,
                        ffmpeg_height - height,
                        0,
                        ffmpeg_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
            rc = proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
        if rc == 0:
            return
        if path.exists():
            path.unlink()
        raise RuntimeError(f"ffmpeg MP4 export failed for {path}: {stderr.strip()}")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for frame in frame_iter:
            frame_rgb = np.asarray(frame, dtype=np.uint8)
            if frame_rgb.shape[:2] != (height, width):
                raise ValueError("All MP4 frames must share the same dimensions.")
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def choose_sweep_pkl(pkls: list[Path]) -> Path:
    preferred = [p for p in pkls if p.stem.endswith("_sweep")]
    candidates = preferred or pkls
    if len(candidates) > 1:
        print(f"[warn] Multiple sweep .pkl files found; using {candidates[0].name}")
    return candidates[0]


def discover_sweep_pkl(primary_dir: Path, legacy_dir: Path | None = None) -> Path | None:
    pkls = sorted(primary_dir.glob("*.pkl"))
    if pkls:
        return choose_sweep_pkl(pkls)

    if legacy_dir is not None:
        legacy_pkls = sorted(legacy_dir.glob("*.pkl"))
        if legacy_pkls:
            print(f"[warn] No sweep .pkl found in {primary_dir}; falling back to {legacy_dir}")
            return choose_sweep_pkl(legacy_pkls)
    return None


def _curve_by_series(curves: list[dict[str, object]], series_name: str) -> dict[str, object]:
    for curve in curves:
        if str(curve.get("series")) == series_name:
            return curve
    raise ValueError(f"Plot cache is missing required series '{series_name}'.")


def square_paper_legend_kwargs(
    *,
    loc: str = SQUARE_PAPER_LEGEND_LOC,
    bbox_to_anchor: tuple[float, float] | None = SQUARE_PAPER_LEGEND_BBOX_TO_ANCHOR,
    ncol: int = SQUARE_PAPER_LEGEND_NCOL,
) -> dict[str, object]:
    return {
        "legend_inside_axes": True,
        "legend_loc": loc,
        "legend_bbox_to_anchor": bbox_to_anchor,
        "legend_ncol": ncol,
    }


def resolve_first_existing_path(
    base_dir: Path,
    preferred_name: str,
    legacy_names: Sequence[str] = (),
) -> Path | None:
    for candidate_name in (preferred_name, *legacy_names):
        candidate_path = base_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path
    return None


def load_optional_cache_row(path: Path) -> pd.Series | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0]


@lru_cache(maxsize=4)
def _load_tracker_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_tracker_rows_for_cached_summary(
    plot_cache_dir: Path,
    summary_row: pd.Series | None,
) -> pd.DataFrame | None:
    if summary_row is None or "frame" not in summary_row.index:
        return None
    frame_value = _series_float_value(summary_row, "frame")
    if frame_value is None:
        return None
    manifest_path = plot_cache_dir / "plot_cache_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    tracker_csv = manifest.get("tracker_csv")
    if tracker_csv is None:
        return None
    tracker_df = _load_tracker_csv_cached(str(Path(str(tracker_csv)).resolve()))
    if "frame" not in tracker_df.columns:
        return None
    rows = tracker_df[tracker_df["frame"] == int(round(frame_value))].sort_values("dot_id")
    return None if rows.empty else rows.copy()


def load_cached_ures_curve_components(
    curve_cache_path: Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    curves = load_curve_bundle_csv(curve_cache_path, x_key="s_norm", y_key="curve_mm")
    sweep_curve = _curve_by_series(curves, "sweep_ures")
    tracked_curve = _curve_by_series(curves, "tracked_ures")
    overlay_curves = [
        curve
        for curve in curves
        if str(curve.get("series", "")).startswith("overlay_ures_")
    ]
    overlay_curves.sort(key=lambda curve: str(curve.get("series")))
    return sweep_curve, tracked_curve, overlay_curves


def render_cached_ures_square_plot(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
    curve_cache_name: str,
    curve_cache_legacy_names: Sequence[str] = (),
    summary_cache_name: str | None,
    summary_cache_legacy_names: Sequence[str] = (),
    output_filename: str,
    figure_title: str,
    summary_line_builder: Callable[[pd.Series], list[str] | None] | None = None,
    compact_box_y: float | None = None,
    bottom_extra_mm: float = SQUARE_PAPER_BOTTOM_DOUBLE_KEY_MM,
    include_endpoint_guides: bool = True,
) -> Path | None:
    curve_cache_path = resolve_first_existing_path(
        plot_cache_dir,
        curve_cache_name,
        curve_cache_legacy_names,
    )
    if curve_cache_path is None:
        return None

    sweep_curve, tracked_curve, overlay_curves = load_cached_ures_curve_components(curve_cache_path)
    compact_lines: list[str] | None = None
    summary_row: pd.Series | None = None
    if summary_cache_name is not None and summary_line_builder is not None:
        summary_cache_path = resolve_first_existing_path(
            plot_cache_dir,
            summary_cache_name,
            summary_cache_legacy_names,
        )
        if summary_cache_path is not None:
            summary_row = load_optional_cache_row(summary_cache_path)
        if summary_row is not None:
            compact_lines = summary_line_builder(summary_row)
    tracker_rows = load_tracker_rows_for_cached_summary(plot_cache_dir, summary_row)

    endpoint_guides: list[dict[str, object]] | None = None
    if include_endpoint_guides:
        endpoint_guides = [
            guide
            for guide in [
                endpoint_guide(
                    np.asarray(sweep_curve["s_norm"], dtype=float),
                    np.asarray(sweep_curve["curve_mm"], dtype=float),
                    color=SQUARE_PAPER_URES_SWEEP_COLOR,
                    linewidth=0.55,
                    alpha=0.5,
                    zorder=SQUARE_PAPER_URES_SWEEP_ZORDER - 0.05,
                ),
                endpoint_guide(
                    np.asarray(tracked_curve["s_norm"], dtype=float),
                    np.asarray(tracked_curve["curve_mm"], dtype=float),
                    color=SQUARE_PAPER_URES_TRACKED_COLOR,
                    linewidth=0.55,
                    alpha=0.55,
                    zorder=SQUARE_PAPER_URES_TRACKED_ZORDER - 0.05,
                ),
            ]
            if guide is not None
        ]

    render_kwargs = square_paper_legend_kwargs()
    if compact_box_y is not None:
        render_kwargs["compact_box_y"] = compact_box_y
    render_kwargs["bottom_extra_mm"] = bottom_extra_mm

    image_rgb = render_square_ures_comparison_plot(
        record={},
        comparison_frame=pd.Series(dtype=float),
        optimizer_s_norm=np.asarray(sweep_curve["s_norm"], dtype=float),
        optimizer_curve_mm=np.asarray(sweep_curve["curve_mm"], dtype=float),
        tracker_s_norm=np.asarray(tracked_curve["s_norm"], dtype=float),
        tracker_curve_mm=np.asarray(tracked_curve["curve_mm"], dtype=float),
        tracker_rows=tracker_rows,
        solidworks_overlays=overlay_curves,
        figure_title=figure_title,
        compact_lines=compact_lines,
        endpoint_guides=endpoint_guides,
        **render_kwargs,
    )
    output_path = output_dir / output_filename
    save_rgb_image(output_path, image_rgb, dpi=PAPER_PANEL_DPI)
    return output_path


def render_largest_close_angle_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    return render_cached_ures_square_plot(
        plot_cache_dir=plot_cache_dir,
        output_dir=output_dir,
        curve_cache_name=LARGEST_CLOSE_ARBOR_ROTATION_CURVES_CACHE_FILENAME,
        curve_cache_legacy_names=LARGEST_CLOSE_ARBOR_ROTATION_LEGACY_CURVES_CACHE_FILENAMES,
        summary_cache_name=LARGEST_CLOSE_ARBOR_ROTATION_SUMMARY_CACHE_FILENAME,
        summary_cache_legacy_names=LARGEST_CLOSE_ARBOR_ROTATION_LEGACY_SUMMARY_CACHE_FILENAMES,
        output_filename=LARGEST_CLOSE_ARBOR_ROTATION_SQUARE_FILENAME,
        figure_title="Largest Close Arbor-Rotation Match",
        summary_line_builder=build_compact_largest_close_angle_lines,
        compact_box_y=LARGEST_CLOSE_ANGLE_COMPACT_BOX_Y,
        bottom_extra_mm=LARGEST_CLOSE_ANGLE_BOTTOM_EXTRA_MM,
    )


def render_final_sweep_final_node_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    return render_cached_ures_square_plot(
        plot_cache_dir=plot_cache_dir,
        output_dir=output_dir,
        curve_cache_name="final_sweep_final_node_ures_match_curves.csv",
        summary_cache_name="final_sweep_final_node_ures_match_summary.csv",
        output_filename="final_sweep_final_node_ures_match_53mm.png",
        figure_title="Final-Sweep Final-Node URES Match",
        summary_line_builder=build_compact_final_sweep_final_node_ures_lines,
        compact_box_y=FINAL_SWEEP_FINAL_NODE_COMPACT_BOX_Y,
        bottom_extra_mm=FINAL_SWEEP_FINAL_NODE_BOTTOM_EXTRA_MM,
    )


def build_compact_target_moment_lines_from_cache_row(summary_row: pd.Series) -> list[str] | None:
    target_moment_nm = _series_float_value(summary_row, "target_moment_nm")
    if target_moment_nm is None:
        target_moment_nm = 1.0
    return build_compact_target_moment_lines(summary_row, target_moment_nm=target_moment_nm)


def build_compact_final_sweep_moment_lines_from_cache_row(summary_row: pd.Series) -> list[str] | None:
    target_moment_nm = _series_float_value(summary_row, "target_moment_nm")
    if target_moment_nm is None:
        target_moment_nm = _series_float_value(summary_row, "matched_optimizer_moment_nm")
    if target_moment_nm is None:
        return None
    lines = build_compact_target_moment_lines(summary_row, target_moment_nm=target_moment_nm)
    return [
        line
        for line in lines
        if not line.startswith("Target moment:")
        and not line.startswith("Moment error:")
    ]


def render_target_moment_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    return render_cached_ures_square_plot(
        plot_cache_dir=plot_cache_dir,
        output_dir=output_dir,
        curve_cache_name="target_1nm_match_ures_curves.csv",
        summary_cache_name="target_1nm_match_summary.csv",
        output_filename="tracker_1nm_match_ures_53mm.png",
        figure_title="Tracker Frame Near 1 N*m",
        summary_line_builder=build_compact_target_moment_lines_from_cache_row,
    )


def render_final_sweep_moment_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    curve_cache_path = plot_cache_dir / "final_sweep_moment_match_ures_curves.csv"
    if not curve_cache_path.is_file():
        return None

    sweep_curve, tracked_curve, overlay_curves = load_cached_ures_curve_components(curve_cache_path)
    summary_row = load_optional_cache_row(plot_cache_dir / "final_sweep_moment_match_summary.csv")
    compact_lines = (
        build_compact_final_sweep_moment_lines_from_cache_row(summary_row)
        if summary_row is not None
        else None
    )
    tracker_rows = load_tracker_rows_for_cached_summary(plot_cache_dir, summary_row)
    endpoint_guides = [
        guide
        for guide in [
            endpoint_guide(
                np.asarray(sweep_curve["s_norm"], dtype=float),
                np.asarray(sweep_curve["curve_mm"], dtype=float),
                color=SQUARE_PAPER_URES_SWEEP_COLOR,
                linewidth=0.55,
                alpha=0.5,
                zorder=SQUARE_PAPER_URES_SWEEP_ZORDER - 0.05,
            ),
            endpoint_guide(
                np.asarray(tracked_curve["s_norm"], dtype=float),
                np.asarray(tracked_curve["curve_mm"], dtype=float),
                color=SQUARE_PAPER_URES_TRACKED_COLOR,
                linewidth=0.55,
                alpha=0.55,
                zorder=SQUARE_PAPER_URES_TRACKED_ZORDER - 0.05,
            ),
        ]
        if guide is not None
    ]
    image_rgb = render_square_ures_comparison_plot(
        record={},
        comparison_frame=pd.Series(dtype=float),
        optimizer_s_norm=np.asarray(sweep_curve["s_norm"], dtype=float),
        optimizer_curve_mm=np.asarray(sweep_curve["curve_mm"], dtype=float),
        tracker_s_norm=np.asarray(tracked_curve["s_norm"], dtype=float),
        tracker_curve_mm=np.asarray(tracked_curve["curve_mm"], dtype=float),
        tracker_rows=tracker_rows,
        solidworks_overlays=overlay_curves,
        figure_title="Final-Sweep Moment Match",
        compact_lines=compact_lines,
        endpoint_guides=endpoint_guides,
        bottom_extra_mm=FINAL_SWEEP_MOMENT_BOTTOM_EXTRA_MM,
        **square_paper_legend_kwargs(),
    )
    output_path = output_dir / "final_sweep_moment_match_ures_53mm.png"
    save_rgb_image(output_path, image_rgb, dpi=PAPER_PANEL_DPI)
    return output_path


def render_moment_vs_visual_angle_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
) -> Path | None:
    curve_cache_path = resolve_first_existing_path(
        plot_cache_dir,
        MOMENT_VS_ARBOR_ROTATION_CACHE_FILENAME,
        MOMENT_VS_ARBOR_ROTATION_LEGACY_CACHE_FILENAMES,
    )
    if curve_cache_path is None:
        return None

    moment_series = load_xy_bundle_csv(curve_cache_path)
    sweep_series = _curve_by_series(moment_series, "sweep")
    tracked_sample_series = _curve_by_series(moment_series, "tracked_samples")
    image_rgb = render_square_xy_overlay_plot(
        optimizer_x=np.asarray(sweep_series["x"], dtype=float),
        optimizer_y=np.asarray(sweep_series["y"], dtype=float),
        tracker_x=np.asarray(tracked_sample_series["x"], dtype=float),
        tracker_y=np.asarray(tracked_sample_series["y"], dtype=float),
        figure_title=f"Moment vs {ANGLE_MEASUREMENT_LABEL}",
        x_label=f"{ANGLE_MEASUREMENT_LABEL} (deg)",
        y_label="Moment (N*m)",
        optimizer_label="Sweep",
        tracker_label="Tracked median",
        fit_linear_models=True,
        bottom_extra_mm=SQUARE_PAPER_BOTTOM_FIT_LABELS_MM,
        **square_paper_legend_kwargs(),
    )
    output_path = output_dir / MOMENT_VS_ARBOR_ROTATION_SQUARE_FILENAME
    save_rgb_image(output_path, image_rgb, dpi=PAPER_PANEL_DPI)
    return output_path


def final_sweep_stress_figure_title_from_cache(plot_cache_dir: Path) -> str:
    figure_title = "Final Sweep Stress"
    sweep_summary_cache = plot_cache_dir / "sweep_summary.csv"
    if not sweep_summary_cache.is_file():
        return figure_title

    sweep_summary_df = pd.read_csv(sweep_summary_cache)
    if sweep_summary_df.empty or "optimizer_rom_deg" not in sweep_summary_df.columns:
        return figure_title

    final_rom_deg = float(pd.to_numeric(sweep_summary_df["optimizer_rom_deg"], errors="coerce").max())
    if math.isfinite(final_rom_deg):
        return f"Final Sweep Stress ({abs(final_rom_deg):.0f} deg)"
    return figure_title


def render_final_sweep_stress_square_plot_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
    reverse_stress_sides: set[str] | None = None,
) -> Path | None:
    curve_cache_path = plot_cache_dir / "final_sweep_stress_comparison_curves.csv"
    if not curve_cache_path.is_file():
        return None

    stress_curves = load_curve_bundle_csv(curve_cache_path, x_key="s_norm", y_key="stress_mpa")
    image_rgb = render_square_stress_curve_bundle_plot(
        curve_bundles=stress_curves,
        figure_title=final_sweep_stress_figure_title_from_cache(plot_cache_dir),
        reverse_fea_sides=reverse_stress_sides,
        **square_paper_legend_kwargs(),
    )
    output_path = output_dir / "final_sweep_stress_vs_solidworks_53mm.png"
    save_rgb_image(output_path, image_rgb, dpi=PAPER_PANEL_DPI)
    return output_path


def render_publication_plots_from_cache(
    *,
    plot_cache_dir: Path,
    output_dir: Path,
    reverse_stress_sides: set[str] | None = None,
) -> list[Path]:
    if not plot_cache_dir.is_dir():
        raise FileNotFoundError(f"Plot cache directory does not exist: {plot_cache_dir}")

    rendered_paths: list[Path] = []

    renderers = [
        render_largest_close_angle_square_plot_from_cache,
        render_final_sweep_final_node_square_plot_from_cache,
        render_target_moment_square_plot_from_cache,
        render_final_sweep_moment_square_plot_from_cache,
        render_moment_vs_visual_angle_square_plot_from_cache,
        render_final_sweep_stress_square_plot_from_cache,
    ]
    for render_plot in renderers:
        render_kwargs = {
            "plot_cache_dir": plot_cache_dir,
            "output_dir": output_dir,
        }
        if render_plot is render_final_sweep_stress_square_plot_from_cache:
            render_kwargs["reverse_stress_sides"] = reverse_stress_sides
        rendered_path = render_plot(**render_kwargs)
        if rendered_path is not None:
            rendered_paths.append(rendered_path)

    if not rendered_paths:
        raise ValueError(
            f"No supported cached plot bundles were found in {plot_cache_dir}."
        )
    return rendered_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory searched recursively for a sweep .pkl containing multiple Spring objects or Spring-bearing records.",
    )
    parser.add_argument(
        "--sweep-pkl",
        type=Path,
        default=None,
        help="Sweep pickle containing a list of records with embedded Spring objects. If omitted, auto-discover from --sweep-pkl-dir.",
    )
    parser.add_argument(
        "--sweep-pkl-dir",
        type=Path,
        default=Path("sweep_pkl_files"),
        help="Directory searched for sweep pickle files when --sweep-pkl is omitted.",
    )
    parser.add_argument(
        "--tracker-csv",
        type=Path,
        default=None,
        help="Tracker displacement CSV with per-frame direct angles. Defaults to <input-dir>/results/displacements.csv when --input-dir is used.",
    )
    parser.add_argument(
        "--tracked-video",
        type=Path,
        default=None,
        help="Analyzed tracker video aligned with the tracker CSV frame_in_window indices. Defaults to <input-dir>/results/tracked.mp4 when --input-dir is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for rendered outputs. Defaults to <input-dir>/results/sweep_vs_tracker when --input-dir is used.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Animation playback rate for the MP4 exports.",
    )
    parser.add_argument(
        "--spring-entries",
        type=int,
        default=None,
        help="Use only the first N entries from the sweep .pkl before matching and rendering.",
    )
    parser.add_argument(
        "--overlay-ures-csv",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Overlay SolidWorks URES CSV curves on the static comparison figures. "
            "Pass explicit CSV paths, or provide the flag with no paths to auto-discover "
            "ures_*.csv under <input-dir>/results and <input-dir>."
        ),
    )
    parser.add_argument(
        "--skip-arbor-angle-mp4",
        action="store_true",
        help=(
            "Skip rendering the arbor-center vector-history MP4. "
            "Arbor-rotation angles are still computed and used for matching/plots."
        ),
    )
    parser.add_argument(
        "--render-from-plot-cache",
        action="store_true",
        help=(
            "Regenerate the cached 53 mm publication plots from output_dir/plot_cache "
            "without rerunning tracker/sweep matching."
        ),
    )
    parser.add_argument(
        "--reverse-stress-side",
        action="append",
        choices=("inner", "outer"),
        default=None,
        help=(
            "Reverse the plotted FEA stress curve orientation for the specified side. "
            "Can be passed multiple times and also works with --render-from-plot-cache."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir is not None else None
    reverse_stress_sides = {
        str(side).strip().lower()
        for side in (args.reverse_stress_side or [])
        if str(side).strip()
    }

    if input_dir is not None and not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    results_root = input_dir / "results" if input_dir is not None else Path("results")
    if args.output_dir is None:
        output_dir = (results_root / "sweep_vs_tracker").resolve()
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_from_plot_cache:
        plot_cache_dir = ensure_plot_cache_dir(output_dir)
        rendered_paths = render_publication_plots_from_cache(
            plot_cache_dir=plot_cache_dir,
            output_dir=output_dir,
            reverse_stress_sides=reverse_stress_sides,
        )
        for rendered_path in rendered_paths:
            print(f"[out] {rendered_path}")
        return 0

    if args.sweep_pkl is not None:
        sweep_path = args.sweep_pkl.expanduser().resolve()
    elif input_dir is not None:
        discovered = resolve_unique_sweep_pkl(input_dir)
        if discovered is None:
            print(f"[warn] No multi-Spring/sweep .pkl files found under {input_dir}; skipping sweep-vs-tracker outputs.")
            return 0
        sweep_path = discovered.resolve()
    else:
        discovered = discover_sweep_pkl(
            args.sweep_pkl_dir.expanduser().resolve(),
            legacy_dir=Path("pkl_to_plot").resolve(),
        )
        if discovered is None:
            print(
                f"[warn] No sweep .pkl files found in {args.sweep_pkl_dir} or legacy pkl_to_plot; "
                "skipping sweep-vs-tracker outputs."
            )
            return 0
        sweep_path = discovered.resolve()
    if args.tracker_csv is None:
        tracker_csv = (results_root / "displacements.csv").resolve()
    else:
        tracker_csv = args.tracker_csv.expanduser().resolve()
    if args.tracked_video is None:
        tracked_video = (results_root / "tracked.mp4").resolve()
    else:
        tracked_video = args.tracked_video.expanduser().resolve()
    tracked_video_is_readable = video_path_is_readable(tracked_video)
    if not tracked_video_is_readable:
        print(f"[warn] Could not open tracked video {tracked_video}; skipping GIF/MP4 comparison outputs.")
    overlay_ures_csvs = resolve_overlay_ures_csvs(args.overlay_ures_csv, input_dir)
    effective_overlay_ures_csvs = (
        overlay_ures_csvs
        if overlay_ures_csvs
        else resolve_overlay_ures_csvs([], input_dir)
    )
    stress_overlay_csvs = resolve_stress_overlay_csvs(input_dir)

    sweep_records = load_pickle(sweep_path)
    if not isinstance(sweep_records, list) or not sweep_records:
        raise TypeError(f"{sweep_path} does not contain a non-empty sweep-record list.")
    if args.spring_entries is not None:
        if args.spring_entries <= 0:
            raise ValueError("--spring-entries must be a positive integer.")
        if args.spring_entries < len(sweep_records):
            print(
                f"[sweep] Using first {args.spring_entries} of {len(sweep_records)} "
                f"entries from {sweep_path.name}"
            )
            sweep_records = sweep_records[:args.spring_entries]
        else:
            print(
                f"[sweep] Requested {args.spring_entries} entries, but {sweep_path.name} "
                f"contains only {len(sweep_records)}; using all entries."
            )

    tracker_df, frame_table = load_tracker_frames(tracker_csv)
    sweep_summary = build_sweep_summary_table(sweep_records)
    plot_cache_dir = ensure_plot_cache_dir(output_dir)
    frame_table.to_csv(plot_cache_dir / "tracker_frame_table.csv", index=False)
    sweep_summary.to_csv(plot_cache_dir / "sweep_summary.csv", index=False)
    save_json(
        plot_cache_dir / "plot_cache_manifest.json",
        {
            "tracker_csv": str(tracker_csv),
            "sweep_pkl": str(sweep_path),
            "input_dir": None if input_dir is None else str(input_dir),
            "tracked_video": str(tracked_video),
        },
    )

    matches = build_matches(sweep_records, frame_table)
    matches_path = output_dir / "sweep_tracker_matches.csv"
    matches.to_csv(matches_path, index=False)

    tracker_to_sweep_angle_matches = build_tracker_to_sweep_matches(
        frame_table.assign(abs_visual_angle_deg=frame_table["visual_angle_deg"].abs()),
        sweep_summary,
        tracker_value_col="abs_visual_angle_deg",
        sweep_value_col="optimizer_rom_deg",
        error_col="abs_angle_error_deg",
    )

    moment_matches: pd.DataFrame | None = None
    tip_displacement_matches: pd.DataFrame | None = None
    tracker_to_sweep_moment_matches: pd.DataFrame | None = None
    moment_matches_path: Path | None = None
    moment_gif_path: Path | None = None
    moment_mp4_path: Path | None = None
    tracker_moment_values = (
        pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce")
        if "tracker_moment_nm" in frame_table.columns
        else pd.Series(dtype=float)
    )
    has_tracker_moment_data = not tracker_moment_values.empty and bool(tracker_moment_values.notna().any())
    if has_tracker_moment_data:
        moment_matches = build_matches_by_moment(sweep_records, frame_table)
        moment_matches_path = output_dir / "sweep_tracker_matches_by_moment.csv"
        moment_matches.to_csv(moment_matches_path, index=False)
        tracker_to_sweep_moment_matches = build_tracker_to_sweep_matches(
            frame_table,
            sweep_summary,
            tracker_value_col="tracker_moment_nm",
            sweep_value_col="optimizer_moment_nm",
            error_col="abs_moment_error_nm",
        )
    else:
        print("[warn] Tracker CSV has no readable per-frame force/moment values; skipping moment-matched outputs.")

    tip_displacement_matches = build_matches_by_tip_displacement(sweep_summary, frame_table)

    stress_plot_path = output_dir / "sweep_stress_per_node.png"
    render_sweep_stress_plot(sweep_records, stress_plot_path)
    final_sweep_stress_square_path: Path | None = output_dir / "final_sweep_stress_vs_solidworks_53mm.png"
    final_sweep_record = max(sweep_records, key=lambda record: abs(float(record["rom_deg"])))
    final_sweep_index = int(final_sweep_record["index"])
    final_sweep_summary_candidates = sweep_summary[sweep_summary["sweep_index"] == final_sweep_index]
    final_sweep_summary_row = (
        final_sweep_summary_candidates.iloc[0].copy()
        if not final_sweep_summary_candidates.empty
        else None
    )
    final_sweep_stress_overlays = load_stress_overlay_curves(stress_overlay_csvs)
    final_stress_curves = build_square_stress_curve_bundle(
        spring=final_sweep_record["spring"],
        solidworks_stress_overlays=final_sweep_stress_overlays,
    )
    save_curve_bundle_csv(
        plot_cache_dir / "final_sweep_stress_comparison_curves.csv",
        final_stress_curves,
        x_key="s_norm",
        y_key="stress_mpa",
    )

    innermost_vector_history_mp4_path: Path | None = None
    apparatus_snapshot_path: Path | None = None
    can_render_snapshot_frames = input_dir is not None or tracked_video_is_readable
    node0_preferred_dot_id: int | None = None
    node0_history_half_extent_px: int | None = None
    if can_render_snapshot_frames and not frame_table.empty:
        apparatus_snapshot_image = render_reference_apparatus_snapshot(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            frame_table=frame_table,
        )
        if apparatus_snapshot_image is not None:
            apparatus_snapshot_path = output_dir / PAPER_SNAPSHOT_OUTPUT_FILENAME
            save_rgb_image(
                apparatus_snapshot_path,
                apparatus_snapshot_image,
                dpi=PAPER_PANEL_DPI,
            )
        node0_reference_frame = choose_reference_frame(frame_table, mode="angle")
        node0_reference_rows = tracker_df[
            tracker_df["frame"] == int(node0_reference_frame["frame"])
        ].sort_values("dot_id")
        if not node0_reference_rows.empty:
            node0_preferred_dot_id = int(tracker_first_node_row(node0_reference_rows)["dot_id"])
            node0_history_half_extent_px = tracker_vector_history_half_extent_px(
                tracker_df,
                frame_table,
                preferred_dot_id=node0_preferred_dot_id,
            )
    if (
        not args.skip_arbor_angle_mp4
        and can_render_snapshot_frames
        and not frame_table.empty
        and node0_preferred_dot_id is not None
        and node0_history_half_extent_px is not None
    ):
        node0_history_frames_rgb = render_node0_vector_history_frames(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            frame_table=frame_table,
            preferred_dot_id=node0_preferred_dot_id,
            half_extent_px=node0_history_half_extent_px,
        )
        if node0_history_frames_rgb:
            innermost_vector_history_mp4_path = output_dir / "innermost_spiral_node_arbor_vector_angles.mp4"
            save_mp4(
                node0_history_frames_rgb,
                innermost_vector_history_mp4_path,
                max(float(args.fps), VECTOR_HISTORY_MIN_FPS),
            )
    elif args.skip_arbor_angle_mp4:
        print("[skip] Skipping arbor-angle MP4 export by request (--skip-arbor-angle-mp4).")

    mp4_path: Path | None = None
    if tracked_video_is_readable:
        frames_rgb: list[np.ndarray] = []
        for _, match in matches.iterrows():
            ctx = build_match_context(match, sweep_records, tracker_df, frame_table)
            comparison_frame = ctx["comparison_frame"]
            video_frame = read_video_frame(tracked_video, int(comparison_frame["frame_in_window"]))
            image = render_comparison(
                video_frame_bgr=video_frame,
                record=ctx["record"],
                comparison_frame=comparison_frame,
                optimizer_s_norm=ctx["optimizer_s_norm"],
                optimizer_curve_mm=ctx["optimizer_curve_mm"],
                tracker_s_norm=ctx["tracker_s_norm"],
                tracker_curve_mm=ctx["tracker_curve_mm"],
                figure_title="Sweep optimizer URES vs nearest tracker-frame URES",
                video_title=(
                    f"Tracked video frame {int(comparison_frame['frame'])}"
                    f"  |  t = {float(comparison_frame['time_s']):.3f} s"
                ),
                angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
            )
            frames_rgb.append(image)
        mp4_path = output_dir / "sweep_vs_tracker.mp4"
        save_mp4(frames_rgb, mp4_path, args.fps)

    moment_frames_rgb: list[np.ndarray] = []
    if tracked_video_is_readable and moment_matches is not None:
        for _, match in moment_matches.iterrows():
            ctx = build_match_context(match, sweep_records, tracker_df, frame_table)
            comparison_frame = ctx["comparison_frame"]
            tracker_moment_nm = _series_float_value(comparison_frame, "tracker_moment_nm")
            video_frame = read_video_frame(tracked_video, int(comparison_frame["frame_in_window"]))
            image = render_comparison(
                video_frame_bgr=video_frame,
                record=ctx["record"],
                comparison_frame=comparison_frame,
                optimizer_s_norm=ctx["optimizer_s_norm"],
                optimizer_curve_mm=ctx["optimizer_curve_mm"],
                tracker_s_norm=ctx["tracker_s_norm"],
                tracker_curve_mm=ctx["tracker_curve_mm"],
                figure_title="Sweep optimizer URES vs nearest tracker-moment URES",
                video_title=(
                    f"Tracked video frame {int(comparison_frame['frame'])}"
                    f"  |  t = {float(comparison_frame['time_s']):.3f} s"
                    f"  |  M = {_format_moment_value(tracker_moment_nm)}"
                ),
                angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
                summary_lines=build_moment_match_summary_lines(ctx["record"], comparison_frame),
                include_force_lines=True,
            )
            moment_frames_rgb.append(image)
        if moment_frames_rgb:
            moment_mp4_path = output_dir / "sweep_vs_tracker_by_moment.mp4"
            save_mp4(moment_frames_rgb, moment_mp4_path, args.fps)

    tip_displacement_mp4_path: Path | None = None
    if tracked_video_is_readable and tip_displacement_matches is not None:
        tip_frames_rgb: list[np.ndarray] = []
        for _, match in tip_displacement_matches.iterrows():
            ctx = build_match_context(match, sweep_records, tracker_df, frame_table)
            comparison_frame = ctx["comparison_frame"]
            video_frame = read_video_frame(tracked_video, int(comparison_frame["frame_in_window"]))
            image = render_comparison(
                video_frame_bgr=video_frame,
                record=ctx["record"],
                comparison_frame=comparison_frame,
                optimizer_s_norm=ctx["optimizer_s_norm"],
                optimizer_curve_mm=ctx["optimizer_curve_mm"],
                tracker_s_norm=ctx["tracker_s_norm"],
                tracker_curve_mm=ctx["tracker_curve_mm"],
                figure_title="Sweep optimizer URES vs nearest tracker URES by tip displacement",
                video_title=(
                    f"Tracked video frame {int(comparison_frame['frame'])}"
                    f"  |  t = {float(comparison_frame['time_s']):.3f} s"
                ),
                angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
                summary_lines=build_tip_displacement_match_summary_lines(ctx["record"], comparison_frame),
                include_force_lines=True,
            )
            tip_frames_rgb.append(image)
        if tip_frames_rgb:
            tip_displacement_mp4_path = output_dir / "sweep_vs_tracker_by_tip_displacement.mp4"
            save_mp4(tip_frames_rgb, tip_displacement_mp4_path, args.fps)

    largest_close_angle_plot_path: Path | None = None
    largest_close_angle_square_path: Path | None = None
    largest_close_angle_snapshot_path: Path | None = None
    final_sweep_final_node_ures_plot_path: Path | None = None
    final_sweep_final_node_ures_square_path: Path | None = None
    final_sweep_final_node_ures_live_path: Path | None = None
    target_1nm_square_path: Path | None = None
    final_sweep_moment_square_path: Path | None = None
    selected_angle_match: pd.Series | None = None
    if not tracker_to_sweep_angle_matches.empty:
        selected_angle_match = choose_largest_close_tracker_match(
            tracker_to_sweep_angle_matches,
            tracker_value_col="tracker_match_value",
            error_col="abs_angle_error_deg",
        )
    if selected_angle_match is not None:
        angle_ctx = build_selected_match_context(selected_angle_match, sweep_records, tracker_df, frame_table)
        angle_overlays = load_ures_overlay_curves(
            effective_overlay_ures_csvs,
            angle_ctx["record"]["spring"],
        )
        angle_summary_match = selected_angle_match.copy()
        angle_reference_sweep_stiffness = rotational_stiffness_nm_per_rad(1.0, 90.0)
        angle_summary_match["sweep_reference_stiffness_nm_per_rad"] = (
            angle_reference_sweep_stiffness
            if angle_reference_sweep_stiffness is not None
            else math.nan
        )
        valid_tracker_moment_frames = frame_table[
            pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce").notna()
        ].copy()
        if not valid_tracker_moment_frames.empty:
            nearest_1nm_tracker_frame = _nearest_row_by_value(
                valid_tracker_moment_frames,
                "tracker_moment_nm",
                1.0,
            )
            angle_summary_match["tracker_reference_frame"] = int(nearest_1nm_tracker_frame["frame"])
            angle_summary_match["tracker_reference_moment_nm"] = float(nearest_1nm_tracker_frame["tracker_moment_nm"])
            angle_summary_match["tracker_reference_angle_deg"] = float(nearest_1nm_tracker_frame["visual_angle_deg"])
            tracker_reference_stiffness = rotational_stiffness_nm_per_rad(
                _series_float_value(nearest_1nm_tracker_frame, "tracker_moment_nm"),
                _series_float_value(nearest_1nm_tracker_frame, "visual_angle_deg"),
            )
            angle_summary_match["tracker_reference_stiffness_nm_per_rad"] = (
                tracker_reference_stiffness
                if tracker_reference_stiffness is not None
                else math.nan
            )
        else:
            angle_summary_match["tracker_reference_frame"] = math.nan
            angle_summary_match["tracker_reference_moment_nm"] = math.nan
            angle_summary_match["tracker_reference_angle_deg"] = math.nan
            angle_summary_match["tracker_reference_stiffness_nm_per_rad"] = math.nan
        angle_image = render_curve_only_comparison(
            record=angle_ctx["record"],
            comparison_frame=angle_ctx["comparison_frame"],
            optimizer_s_norm=angle_ctx["optimizer_s_norm"],
            optimizer_curve_mm=angle_ctx["optimizer_curve_mm"],
            tracker_s_norm=angle_ctx["tracker_s_norm"],
            tracker_curve_mm=angle_ctx["tracker_curve_mm"],
            tracker_rows=angle_ctx["tracker_rows"],
            solidworks_overlays=angle_overlays,
            figure_title="Largest close arbor-rotation match: optimizer vs tracker URES",
            angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
            summary_lines=build_largest_close_angle_summary_lines(angle_summary_match),
            include_force_lines=True,
        )
        largest_close_angle_plot_path = output_dir / LARGEST_CLOSE_ARBOR_ROTATION_PLOT_FILENAME
        save_rgb_image(largest_close_angle_plot_path, angle_image)
        largest_close_angle_square_path = output_dir / LARGEST_CLOSE_ARBOR_ROTATION_SQUARE_FILENAME
        save_curve_bundle_csv(
            plot_cache_dir / LARGEST_CLOSE_ARBOR_ROTATION_CURVES_CACHE_FILENAME,
            [
                {
                    "series": "sweep_ures",
                    "label": "Sweep URES",
                    "s_norm": angle_ctx["optimizer_s_norm"],
                    "curve_mm": angle_ctx["optimizer_curve_mm"],
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.1,
                },
                {
                    "series": "tracked_ures",
                    "label": "Tracked URES",
                    "s_norm": angle_ctx["tracker_s_norm"],
                    "curve_mm": angle_ctx["tracker_curve_mm"],
                    "color": "crimson",
                    "linestyle": "-",
                    "linewidth": 1.0,
                },
                *[
                    {
                        "series": f"overlay_ures_{idx + 1}",
                        "label": str(overlay["label"]),
                        "s_norm": np.asarray(overlay["s_norm"], dtype=float),
                        "curve_mm": np.asarray(overlay["curve_mm"], dtype=float),
                        "color": str(overlay["color"]),
                        "linestyle": str(overlay["linestyle"]),
                        "linewidth": float(overlay["linewidth"]),
                    }
                    for idx, overlay in enumerate(angle_overlays)
                ],
            ],
            x_key="s_norm",
            y_key="curve_mm",
        )
        pd.DataFrame([angle_summary_match.to_dict()]).to_csv(
            plot_cache_dir / LARGEST_CLOSE_ARBOR_ROTATION_SUMMARY_CACHE_FILENAME,
            index=False,
        )

        angle_reference_frame = choose_reference_frame(frame_table, mode="angle")
        angle_reference_rows = tracker_df[tracker_df["frame"] == int(angle_reference_frame["frame"])].sort_values("dot_id")
        angle_target_rows = tracker_df[
            tracker_df["frame"] == int(angle_ctx["comparison_frame"]["frame"])
        ].sort_values("dot_id")
        angle_first_dot_id = int(tracker_first_node_row(angle_reference_rows)["dot_id"])
        angle_annotation_lines = build_snapshot_angle_annotation_lines(
            angle_ctx["comparison_frame"],
            angle_target_rows,
            preferred_dot_id=angle_first_dot_id,
        )
        angle_half_extent_px = max(
            tracker_vector_crop_half_extent_px(
                angle_reference_rows,
                include_deformed_vector=False,
                preferred_dot_id=angle_first_dot_id,
            ),
            tracker_vector_crop_half_extent_px(
                angle_target_rows,
                include_deformed_vector=True,
                preferred_dot_id=angle_first_dot_id,
            ),
        )
        angle_reference_panel = render_snapshot_panel_for_frame(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            comparison_frame=angle_reference_frame,
            include_deformed_vector=False,
            half_extent_px=angle_half_extent_px,
            preferred_dot_id=angle_first_dot_id,
        )
        angle_target_panel = render_snapshot_panel_for_frame(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            comparison_frame=angle_ctx["comparison_frame"],
            include_deformed_vector=True,
            half_extent_px=angle_half_extent_px,
            annotation_lines=angle_annotation_lines,
            preferred_dot_id=angle_first_dot_id,
        )
        angle_snapshot_image = render_snapshot_pair_figure(
            left_image_rgb=angle_reference_panel,
            right_image_rgb=angle_target_panel,
            figure_title="Rigid arbor-marker vectors for largest close arbor-rotation match",
            left_title=(
                f"reference frame {int(angle_reference_frame['frame'])}"
                f" | cyan = reference rigid-set vector"
            ),
            right_title=(
                f"selected {ANGLE_MEASUREMENT_LABEL.lower()} frame {int(angle_ctx['comparison_frame']['frame'])}"
                f" | cyan = reference, red = deformed rigid-set vector"
            ),
        )
        largest_close_angle_snapshot_path = output_dir / LARGEST_CLOSE_ARBOR_ROTATION_SNAPSHOT_FILENAME
        save_rgb_image(largest_close_angle_snapshot_path, angle_snapshot_image)

    selected_final_node_ures_match: pd.Series | None = None
    if final_sweep_summary_row is not None:
        tracker_tip_values = pd.to_numeric(frame_table["tip_displacement_mm"], errors="coerce")
        valid_tip_frames = frame_table.loc[tracker_tip_values.notna()].copy()
        if not valid_tip_frames.empty:
            nearest_tip_frame = _nearest_row_by_value(
                valid_tip_frames,
                "tip_displacement_mm",
                float(final_sweep_summary_row["optimizer_final_node_ures_mm"]),
            )
            selected_final_node_ures_match = pd.Series(
                {
                    "frame": int(nearest_tip_frame["frame"]),
                    "frame_in_window": int(nearest_tip_frame["frame_in_window"]),
                    "time_s": float(nearest_tip_frame["time_s"]),
                    "tracker_visual_angle_deg": float(nearest_tip_frame["visual_angle_deg"]),
                    "tracker_direct_angle_deg": float(nearest_tip_frame["direct_angle_deg"]),
                    "tracker_tip_displacement_mm": float(nearest_tip_frame["tip_displacement_mm"]),
                    "tracker_force_n": _series_float_value(nearest_tip_frame, "tracker_force_n"),
                    "tracker_moment_nm": _series_float_value(nearest_tip_frame, "tracker_moment_nm"),
                    "matched_sweep_index": int(final_sweep_summary_row["sweep_index"]),
                    "matched_sweep_name": str(final_sweep_summary_row["sweep_name"]),
                    "matched_optimizer_rom_deg": float(final_sweep_summary_row["optimizer_rom_deg"]),
                    "matched_optimizer_moment_nm": float(final_sweep_summary_row["optimizer_moment_nm"]),
                    "matched_optimizer_final_node_ures_mm": float(
                        final_sweep_summary_row["optimizer_final_node_ures_mm"]
                    ),
                    "abs_final_node_ures_error_mm": float(
                        abs(
                            float(nearest_tip_frame["tip_displacement_mm"])
                            - float(final_sweep_summary_row["optimizer_final_node_ures_mm"])
                        )
                    ),
                }
            )
    if selected_final_node_ures_match is not None:
        final_node_ctx = build_selected_match_context(
            selected_final_node_ures_match,
            sweep_records,
            tracker_df,
            frame_table,
        )
        final_node_overlays = load_ures_overlay_curves(
            effective_overlay_ures_csvs,
            final_node_ctx["record"]["spring"],
        )
        final_node_image = render_curve_only_comparison(
            record=final_node_ctx["record"],
            comparison_frame=final_node_ctx["comparison_frame"],
            optimizer_s_norm=final_node_ctx["optimizer_s_norm"],
            optimizer_curve_mm=final_node_ctx["optimizer_curve_mm"],
            tracker_s_norm=final_node_ctx["tracker_s_norm"],
            tracker_curve_mm=final_node_ctx["tracker_curve_mm"],
            tracker_rows=final_node_ctx["tracker_rows"],
            solidworks_overlays=final_node_overlays,
            figure_title="Final-sweep final-node URES match: optimizer vs tracker URES",
            angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
            summary_lines=build_final_sweep_final_node_ures_summary_lines(selected_final_node_ures_match),
            include_force_lines=True,
        )
        final_sweep_final_node_ures_plot_path = output_dir / "final_sweep_final_node_ures_match_ures.png"
        save_rgb_image(final_sweep_final_node_ures_plot_path, final_node_image)
        final_sweep_final_node_ures_square_path = output_dir / "final_sweep_final_node_ures_match_53mm.png"
        save_curve_bundle_csv(
            plot_cache_dir / "final_sweep_final_node_ures_match_curves.csv",
            [
                {
                    "series": "sweep_ures",
                    "label": "Sweep URES",
                    "s_norm": final_node_ctx["optimizer_s_norm"],
                    "curve_mm": final_node_ctx["optimizer_curve_mm"],
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.1,
                },
                {
                    "series": "tracked_ures",
                    "label": "Tracked URES",
                    "s_norm": final_node_ctx["tracker_s_norm"],
                    "curve_mm": final_node_ctx["tracker_curve_mm"],
                    "color": "crimson",
                    "linestyle": "-",
                    "linewidth": 1.0,
                },
                *[
                    {
                        "series": f"overlay_ures_{idx + 1}",
                        "label": str(overlay["label"]),
                        "s_norm": np.asarray(overlay["s_norm"], dtype=float),
                        "curve_mm": np.asarray(overlay["curve_mm"], dtype=float),
                        "color": str(overlay["color"]),
                        "linestyle": str(overlay["linestyle"]),
                        "linewidth": float(overlay["linewidth"]),
                    }
                    for idx, overlay in enumerate(final_node_overlays)
                ],
            ],
            x_key="s_norm",
            y_key="curve_mm",
        )
        pd.DataFrame([selected_final_node_ures_match.to_dict()]).to_csv(
            plot_cache_dir / "final_sweep_final_node_ures_match_summary.csv",
            index=False,
        )
        if tracked_video_is_readable:
            video_frame = read_video_frame(
                tracked_video,
                int(final_node_ctx["comparison_frame"]["frame_in_window"]),
            )
            tracker_moment_nm = _series_float_value(final_node_ctx["comparison_frame"], "tracker_moment_nm")
            final_node_live_image = render_comparison(
                video_frame_bgr=video_frame,
                record=final_node_ctx["record"],
                comparison_frame=final_node_ctx["comparison_frame"],
                optimizer_s_norm=final_node_ctx["optimizer_s_norm"],
                optimizer_curve_mm=final_node_ctx["optimizer_curve_mm"],
                tracker_s_norm=final_node_ctx["tracker_s_norm"],
                tracker_curve_mm=final_node_ctx["tracker_curve_mm"],
                solidworks_overlays=final_node_overlays,
                figure_title="Final-sweep final-node URES match: optimizer vs tracker URES",
                video_title=(
                    f"Tracked video frame {int(final_node_ctx['comparison_frame']['frame'])}"
                    f"  |  t = {float(final_node_ctx['comparison_frame']['time_s']):.3f} s"
                    f"  |  M = {_format_moment_value(tracker_moment_nm)}"
                ),
                angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
                summary_lines=build_final_sweep_final_node_ures_summary_lines(selected_final_node_ures_match),
                include_force_lines=True,
            )
            final_sweep_final_node_ures_live_path = (
                output_dir / "final_sweep_final_node_ures_match_live.png"
            )
            save_rgb_image(final_sweep_final_node_ures_live_path, final_node_live_image)

    target_moment_nm = 1.0
    selected_target_moment_match: pd.Series | None = None
    if tracker_to_sweep_moment_matches is not None and not tracker_to_sweep_moment_matches.empty:
        selected_target_moment_match = _nearest_row_by_value(
            tracker_to_sweep_moment_matches,
            "tracker_match_value",
            target_moment_nm,
        )
    if selected_target_moment_match is not None:
        target_moment_ctx = build_selected_match_context(
            selected_target_moment_match,
            sweep_records,
            tracker_df,
            frame_table,
        )
        target_moment_overlays = load_ures_overlay_curves(
            effective_overlay_ures_csvs,
            target_moment_ctx["record"]["spring"],
        )
        target_1nm_square_path = output_dir / "tracker_1nm_match_ures_53mm.png"
        save_curve_bundle_csv(
            plot_cache_dir / "target_1nm_match_ures_curves.csv",
            [
                {
                    "series": "sweep_ures",
                    "label": "Sweep URES",
                    "s_norm": target_moment_ctx["optimizer_s_norm"],
                    "curve_mm": target_moment_ctx["optimizer_curve_mm"],
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.1,
                },
                {
                    "series": "tracked_ures",
                    "label": "Tracked URES",
                    "s_norm": target_moment_ctx["tracker_s_norm"],
                    "curve_mm": target_moment_ctx["tracker_curve_mm"],
                    "color": "crimson",
                    "linestyle": "-",
                    "linewidth": 1.0,
                },
                *[
                    {
                        "series": f"overlay_ures_{idx + 1}",
                        "label": str(overlay["label"]),
                        "s_norm": np.asarray(overlay["s_norm"], dtype=float),
                        "curve_mm": np.asarray(overlay["curve_mm"], dtype=float),
                        "color": str(overlay["color"]),
                        "linestyle": str(overlay["linestyle"]),
                        "linewidth": float(overlay["linewidth"]),
                    }
                    for idx, overlay in enumerate(target_moment_overlays)
                ],
            ],
            x_key="s_norm",
            y_key="curve_mm",
        )
        target_moment_summary_payload = selected_target_moment_match.to_dict()
        target_moment_summary_payload["target_moment_nm"] = target_moment_nm
        pd.DataFrame([target_moment_summary_payload]).to_csv(
            plot_cache_dir / "target_1nm_match_summary.csv",
            index=False,
        )

    selected_final_sweep_moment_match: pd.Series | None = None
    if final_sweep_summary_row is not None and has_tracker_moment_data:
        target_final_sweep_moment_nm = _series_float_value(final_sweep_summary_row, "optimizer_moment_nm")
        valid_tracker_moment_frames = frame_table[
            pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce").notna()
        ].copy()
        if target_final_sweep_moment_nm is not None and not valid_tracker_moment_frames.empty:
            nearest_final_sweep_moment_frame = _nearest_row_by_value(
                valid_tracker_moment_frames,
                "tracker_moment_nm",
                float(target_final_sweep_moment_nm),
            )
            selected_final_sweep_moment_match = pd.Series(
                {
                    "frame": int(nearest_final_sweep_moment_frame["frame"]),
                    "frame_in_window": int(nearest_final_sweep_moment_frame["frame_in_window"]),
                    "time_s": float(nearest_final_sweep_moment_frame["time_s"]),
                    "tracker_visual_angle_deg": float(nearest_final_sweep_moment_frame["visual_angle_deg"]),
                    "tracker_direct_angle_deg": float(nearest_final_sweep_moment_frame["direct_angle_deg"]),
                    "tracker_tip_displacement_mm": float(nearest_final_sweep_moment_frame["tip_displacement_mm"]),
                    "tracker_force_n": _series_float_value(nearest_final_sweep_moment_frame, "tracker_force_n"),
                    "tracker_moment_nm": _series_float_value(nearest_final_sweep_moment_frame, "tracker_moment_nm"),
                    "matched_sweep_index": int(final_sweep_summary_row["sweep_index"]),
                    "matched_sweep_name": str(final_sweep_summary_row["sweep_name"]),
                    "matched_optimizer_rom_deg": float(final_sweep_summary_row["optimizer_rom_deg"]),
                    "matched_optimizer_moment_nm": float(final_sweep_summary_row["optimizer_moment_nm"]),
                    "matched_optimizer_final_node_ures_mm": float(
                        final_sweep_summary_row["optimizer_final_node_ures_mm"]
                    ),
                    "target_moment_nm": float(target_final_sweep_moment_nm),
                    "abs_moment_error_nm": float(
                        abs(
                            float(nearest_final_sweep_moment_frame["tracker_moment_nm"])
                            - float(target_final_sweep_moment_nm)
                        )
                    ),
                }
            )
    if selected_final_sweep_moment_match is not None:
        final_sweep_moment_ctx = build_record_frame_context(
            final_sweep_record,
            frame_table[frame_table["frame"] == int(selected_final_sweep_moment_match["frame"])].iloc[0],
            tracker_df,
        )
        final_sweep_moment_overlays = load_ures_overlay_curves(
            effective_overlay_ures_csvs,
            final_sweep_record["spring"],
        )
        final_sweep_moment_square_path = output_dir / "final_sweep_moment_match_ures_53mm.png"
        save_curve_bundle_csv(
            plot_cache_dir / "final_sweep_moment_match_ures_curves.csv",
            [
                {
                    "series": "sweep_ures",
                    "label": "Sweep URES",
                    "s_norm": final_sweep_moment_ctx["optimizer_s_norm"],
                    "curve_mm": final_sweep_moment_ctx["optimizer_curve_mm"],
                    "color": "black",
                    "linestyle": "-",
                    "linewidth": 1.1,
                },
                {
                    "series": "tracked_ures",
                    "label": "Tracked URES",
                    "s_norm": final_sweep_moment_ctx["tracker_s_norm"],
                    "curve_mm": final_sweep_moment_ctx["tracker_curve_mm"],
                    "color": "crimson",
                    "linestyle": "-",
                    "linewidth": 1.0,
                },
                *[
                    {
                        "series": f"overlay_ures_{idx + 1}",
                        "label": str(overlay["label"]),
                        "s_norm": np.asarray(overlay["s_norm"], dtype=float),
                        "curve_mm": np.asarray(overlay["curve_mm"], dtype=float),
                        "color": str(overlay["color"]),
                        "linestyle": str(overlay["linestyle"]),
                        "linewidth": float(overlay["linewidth"]),
                    }
                    for idx, overlay in enumerate(final_sweep_moment_overlays)
                ],
            ],
            x_key="s_norm",
            y_key="curve_mm",
        )
        pd.DataFrame([selected_final_sweep_moment_match.to_dict()]).to_csv(
            plot_cache_dir / "final_sweep_moment_match_summary.csv",
            index=False,
        )

    largest_moment_plot_path: Path | None = None
    largest_moment_snapshot_path: Path | None = None
    selected_moment_match: pd.Series | None = None
    if tracker_to_sweep_moment_matches is not None:
        selected_moment_match = choose_largest_close_tracker_match(
            tracker_to_sweep_moment_matches,
            tracker_value_col="tracker_match_value",
            error_col="abs_moment_error_nm",
        )
    if selected_moment_match is not None:
        moment_ctx = build_selected_match_context(selected_moment_match, sweep_records, tracker_df, frame_table)
        moment_overlays = load_ures_overlay_curves(
            overlay_ures_csvs,
            moment_ctx["record"]["spring"],
        )
        moment_image = render_curve_only_comparison(
            record=moment_ctx["record"],
            comparison_frame=moment_ctx["comparison_frame"],
            optimizer_s_norm=moment_ctx["optimizer_s_norm"],
            optimizer_curve_mm=moment_ctx["optimizer_curve_mm"],
            tracker_s_norm=moment_ctx["tracker_s_norm"],
            tracker_curve_mm=moment_ctx["tracker_curve_mm"],
            tracker_rows=moment_ctx["tracker_rows"],
            solidworks_overlays=moment_overlays,
            figure_title="Largest close moment match: optimizer vs tracker URES",
            angle_label=f"Tracker {ANGLE_MEASUREMENT_LABEL.lower()}",
            summary_lines=build_largest_close_moment_summary_lines(selected_moment_match),
            include_force_lines=True,
        )
        largest_moment_plot_path = output_dir / "largest_close_moment_match_ures.png"
        save_rgb_image(largest_moment_plot_path, moment_image)

        moment_reference_candidates = frame_table[
            pd.to_numeric(frame_table["tracker_moment_nm"], errors="coerce").notna()
        ].copy()
        if moment_reference_candidates.empty:
            moment_reference_frame = choose_reference_frame(frame_table, mode="angle")
        else:
            moment_reference_candidates["abs_visual_angle_deg"] = moment_reference_candidates["visual_angle_deg"].abs()
            moment_reference_frame = _nearest_row_by_value(
                moment_reference_candidates,
                "abs_visual_angle_deg",
                0.0,
            )
        moment_reference_rows = tracker_df[
            tracker_df["frame"] == int(moment_reference_frame["frame"])
        ].sort_values("dot_id")
        moment_target_rows = tracker_df[
            tracker_df["frame"] == int(moment_ctx["comparison_frame"]["frame"])
        ].sort_values("dot_id")
        moment_first_dot_id = int(tracker_first_node_row(moment_reference_rows)["dot_id"])
        moment_annotation_lines = build_snapshot_angle_annotation_lines(
            moment_ctx["comparison_frame"],
            moment_target_rows,
            preferred_dot_id=moment_first_dot_id,
        )
        moment_half_extent_px = max(
            tracker_vector_crop_half_extent_px(
                moment_reference_rows,
                include_deformed_vector=False,
                preferred_dot_id=moment_first_dot_id,
            ),
            tracker_vector_crop_half_extent_px(
                moment_target_rows,
                include_deformed_vector=True,
                preferred_dot_id=moment_first_dot_id,
            ),
        )
        moment_reference_panel = render_snapshot_panel_for_frame(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            comparison_frame=moment_reference_frame,
            include_deformed_vector=False,
            half_extent_px=moment_half_extent_px,
            preferred_dot_id=moment_first_dot_id,
        )
        moment_target_panel = render_snapshot_panel_for_frame(
            input_dir=input_dir,
            tracked_video=tracked_video,
            tracker_df=tracker_df,
            comparison_frame=moment_ctx["comparison_frame"],
            include_deformed_vector=True,
            half_extent_px=moment_half_extent_px,
            annotation_lines=moment_annotation_lines,
            preferred_dot_id=moment_first_dot_id,
        )
        moment_snapshot_image = render_snapshot_pair_figure(
            left_image_rgb=moment_reference_panel,
            right_image_rgb=moment_target_panel,
            figure_title="Rigid arbor-marker vectors for selected final-node URES vs moment point",
            left_title=(
                f"reference frame {int(moment_reference_frame['frame'])}"
                f" | cyan = reference rigid-set vector"
            ),
            right_title=(
                f"selected moment frame {int(moment_ctx['comparison_frame']['frame'])}"
                f" | cyan = reference, red = deformed rigid-set vector"
            ),
        )
        largest_moment_snapshot_path = output_dir / "largest_close_moment_match_snapshots.png"
        save_rgb_image(largest_moment_snapshot_path, moment_snapshot_image)

    moment_vs_angle_plot_path: Path | None = None
    moment_vs_angle_square_path: Path | None = None
    final_node_vs_moment_plot_path: Path | None = None
    optimizer_moment_curve_x = np.empty(0, dtype=float)
    optimizer_moment_curve_y = np.empty(0, dtype=float)
    if has_tracker_moment_data:
        tracker_curve_table = frame_table.copy()
        tracker_curve_table["abs_visual_angle_deg"] = tracker_curve_table["visual_angle_deg"].abs()
        tracker_curve_table = tracker_curve_table[
            pd.to_numeric(tracker_curve_table["tracker_moment_nm"], errors="coerce").notna()
        ].sort_values("abs_visual_angle_deg")
        optimizer_curve_table = sweep_summary.sort_values("optimizer_rom_deg")
        optimizer_moment_curve_x = optimizer_curve_table["optimizer_rom_deg"].to_numpy(dtype=float)
        optimizer_moment_curve_y = optimizer_curve_table["optimizer_moment_nm"].to_numpy(dtype=float)

        moment_vs_angle_image = render_xy_overlay_plot(
            optimizer_x=optimizer_moment_curve_x,
            optimizer_y=optimizer_moment_curve_y,
            tracker_x=tracker_curve_table["abs_visual_angle_deg"].to_numpy(dtype=float),
            tracker_y=tracker_curve_table["tracker_moment_nm"].to_numpy(dtype=float),
            figure_title=f"Moment vs {ANGLE_MEASUREMENT_LABEL.lower()}",
            x_label=f"{ANGLE_MEASUREMENT_LABEL}  (deg)",
            y_label="Moment  (N*m)",
        )
        moment_vs_angle_plot_path = output_dir / MOMENT_VS_ARBOR_ROTATION_PLOT_FILENAME
        save_rgb_image(moment_vs_angle_plot_path, moment_vs_angle_image)
        moment_vs_angle_square_path = output_dir / MOMENT_VS_ARBOR_ROTATION_SQUARE_FILENAME
        tracked_median_x, tracked_median_y = _binned_median_curve(
            tracker_curve_table["abs_visual_angle_deg"].to_numpy(dtype=float),
            tracker_curve_table["tracker_moment_nm"].to_numpy(dtype=float),
        )
        save_xy_bundle_csv(
            plot_cache_dir / MOMENT_VS_ARBOR_ROTATION_CACHE_FILENAME,
            [
                {
                    "series": "sweep",
                    "label": "Sweep",
                    "x": optimizer_curve_table["optimizer_rom_deg"].to_numpy(dtype=float),
                    "y": optimizer_curve_table["optimizer_moment_nm"].to_numpy(dtype=float),
                },
                {
                    "series": "tracked_samples",
                    "label": "Tracked samples",
                    "x": tracker_curve_table["abs_visual_angle_deg"].to_numpy(dtype=float),
                    "y": tracker_curve_table["tracker_moment_nm"].to_numpy(dtype=float),
                },
                {
                    "series": "tracked_median",
                    "label": "Tracked median",
                    "x": tracked_median_x,
                    "y": tracked_median_y,
                },
            ],
        )

        final_node_vs_moment_image = render_xy_overlay_plot(
            optimizer_x=optimizer_curve_table["optimizer_moment_nm"].to_numpy(dtype=float),
            optimizer_y=optimizer_curve_table["optimizer_final_node_ures_mm"].to_numpy(dtype=float),
            tracker_x=tracker_curve_table["tracker_moment_nm"].to_numpy(dtype=float),
            tracker_y=tracker_curve_table["tip_displacement_mm"].to_numpy(dtype=float),
            figure_title="Final-node URES vs moment",
            x_label="Moment  (N*m)",
            y_label="Final-node URES  (mm)",
        )
        final_node_vs_moment_plot_path = output_dir / "final_node_ures_vs_moment.png"
        save_rgb_image(final_node_vs_moment_plot_path, final_node_vs_moment_image)

    force_arbor_moment_triptych_mp4_path: Path | None = None
    has_force_meter_metadata = bool(
        frame_table["force_bbox_x"].notna().any()
        or frame_table["tracker_force_n"].notna().any()
        or frame_table["force_digits"].fillna("").astype(str).str.strip().ne("").any()
    )
    if (
        tracked_video_is_readable
        and has_tracker_moment_data
        and has_force_meter_metadata
        and node0_preferred_dot_id is not None
        and node0_history_half_extent_px is not None
        and optimizer_moment_curve_x.size
        and optimizer_moment_curve_y.size
    ):
        triptych_fps = video_path_fps(tracked_video)
        if triptych_fps is None:
            triptych_fps = max(float(args.fps), VECTOR_HISTORY_MIN_FPS)
        force_arbor_moment_triptych_mp4_path = output_dir / "force_arbor_moment_sanity_check.mp4"
        save_mp4_stream(
            iter_force_arbor_moment_triptych_frames(
                tracked_video=tracked_video,
                tracker_df=tracker_df,
                frame_table=frame_table,
                preferred_dot_id=node0_preferred_dot_id,
                arbor_half_extent_px=node0_history_half_extent_px,
                optimizer_x=optimizer_moment_curve_x,
                optimizer_y=optimizer_moment_curve_y,
            ),
            force_arbor_moment_triptych_mp4_path,
            triptych_fps,
        )
    elif tracked_video_is_readable and has_tracker_moment_data and not has_force_meter_metadata:
        print("[warn] Tracker CSV has no force-meter metadata; skipping force/arbor/moment sanity-check MP4.")

    render_publication_plots_from_cache(
        plot_cache_dir=plot_cache_dir,
        output_dir=output_dir,
        reverse_stress_sides=reverse_stress_sides,
    )

    mean_err = float(matches["abs_angle_error_deg"].mean())
    max_err = float(matches["abs_angle_error_deg"].max())
    print(f"[out] {matches_path}")
    print(f"[out] {stress_plot_path}")
    if moment_matches_path is not None:
        print(f"[out] {moment_matches_path}")
    if largest_close_angle_plot_path is not None:
        print(f"[out] {largest_close_angle_plot_path}")
    if largest_close_angle_square_path is not None:
        print(f"[out] {largest_close_angle_square_path}")
    if largest_close_angle_snapshot_path is not None:
        print(f"[out] {largest_close_angle_snapshot_path}")
    if final_sweep_final_node_ures_plot_path is not None:
        print(f"[out] {final_sweep_final_node_ures_plot_path}")
    if final_sweep_final_node_ures_square_path is not None:
        print(f"[out] {final_sweep_final_node_ures_square_path}")
    if final_sweep_final_node_ures_live_path is not None:
        print(f"[out] {final_sweep_final_node_ures_live_path}")
    if target_1nm_square_path is not None:
        print(f"[out] {target_1nm_square_path}")
    if final_sweep_moment_square_path is not None:
        print(f"[out] {final_sweep_moment_square_path}")
    if largest_moment_plot_path is not None:
        print(f"[out] {largest_moment_plot_path}")
    if largest_moment_snapshot_path is not None:
        print(f"[out] {largest_moment_snapshot_path}")
    if moment_vs_angle_plot_path is not None:
        print(f"[out] {moment_vs_angle_plot_path}")
    if moment_vs_angle_square_path is not None:
        print(f"[out] {moment_vs_angle_square_path}")
    if final_node_vs_moment_plot_path is not None:
        print(f"[out] {final_node_vs_moment_plot_path}")
    if final_sweep_stress_square_path is not None:
        print(f"[out] {final_sweep_stress_square_path}")
    if apparatus_snapshot_path is not None:
        print(f"[out] {apparatus_snapshot_path}")
    if innermost_vector_history_mp4_path is not None:
        print(f"[out] {innermost_vector_history_mp4_path}")
    if force_arbor_moment_triptych_mp4_path is not None:
        print(f"[out] {force_arbor_moment_triptych_mp4_path}")
    if mp4_path is not None:
        print(f"[out] {mp4_path}")
    if moment_mp4_path is not None:
        print(f"[out] {moment_mp4_path}")
    if tip_displacement_mp4_path is not None:
        print(f"[out] {tip_displacement_mp4_path}")
    print(
        "[match] "
        f"{len(matches)} sweep points  |  mean abs angle error = {mean_err:.3f} deg"
        f"  |  max abs angle error = {max_err:.3f} deg"
    )
    if moment_matches is not None and not moment_matches.empty:
        mean_moment_err = float(moment_matches["abs_moment_error_nm"].mean())
        max_moment_err = float(moment_matches["abs_moment_error_nm"].max())
        print(
            "[moment-match] "
            f"{len(moment_matches)} sweep points  |  mean abs moment error = {mean_moment_err:.4f} N*m"
            f"  |  max abs moment error = {max_moment_err:.4f} N*m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
