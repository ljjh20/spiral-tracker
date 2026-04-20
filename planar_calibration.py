"""AprilTag-based planar calibration helpers."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


APRILTAG_FAMILY_IDS = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class CalibrationError(RuntimeError):
    """Raised when the AprilTag calibration inputs are unusable."""


@dataclass(frozen=True)
class IntrinsicsRecord:
    frame_index: int
    width: int
    height: int
    camera_matrix: np.ndarray
    parse_mode: str


class IntrinsicsSequence:
    """Per-frame camera intrinsics with nearest-frame lookup."""

    def __init__(self, records: list[IntrinsicsRecord]):
        if not records:
            raise CalibrationError("No intrinsics records were loaded.")
        ordered = sorted(records, key=lambda rec: rec.frame_index)
        self._records = ordered
        self._frame_indices = [rec.frame_index for rec in ordered]

    @property
    def parse_modes(self) -> set[str]:
        return {rec.parse_mode for rec in self._records}

    @property
    def frame_count(self) -> int:
        return len(self._records)

    def validate_image_size(self, width: int, height: int) -> None:
        mismatched = sorted(
            {
                (int(rec.width), int(rec.height))
                for rec in self._records
                if int(rec.width) != int(width) or int(rec.height) != int(height)
            }
        )
        if mismatched:
            preview = ", ".join(f"{w}x{h}" for w, h in mismatched[:4])
            if len(mismatched) > 4:
                preview += ", ..."
            raise CalibrationError(
                "Intrinsics image dimensions do not match the video: "
                f"expected {width}x{height}, saw {preview}."
            )

    def get(self, frame_index: int) -> IntrinsicsRecord:
        pos = bisect_left(self._frame_indices, int(frame_index))
        if pos < len(self._records) and self._frame_indices[pos] == frame_index:
            return self._records[pos]
        if pos == 0:
            return self._records[0]
        if pos >= len(self._records):
            return self._records[-1]
        before = self._records[pos - 1]
        after = self._records[pos]
        if abs(before.frame_index - frame_index) <= abs(after.frame_index - frame_index):
            return before
        return after


@dataclass(frozen=True)
class DepthCalibration:
    intrinsic_matrix: np.ndarray
    reference_width: int
    reference_height: int
    distortion_center: np.ndarray
    lens_distortion_lookup_table: np.ndarray
    inverse_lens_distortion_lookup_table: np.ndarray
    depth_delivered: bool

    def validate_image_size(self, width: int, height: int) -> None:
        if int(width) != self.reference_width or int(height) != self.reference_height:
            raise CalibrationError(
                "Video dimensions do not match depth calibration reference dimensions: "
                f"video is {width}x{height}, calibration expects "
                f"{self.reference_width}x{self.reference_height}."
            )

    @property
    def max_radius_px(self) -> float:
        cx, cy = self.distortion_center
        delta_x = max(float(cx), float(self.reference_width) - float(cx))
        delta_y = max(float(cy), float(self.reference_height) - float(cy))
        return float(np.hypot(delta_x, delta_y))

    def distort_points(self, points_rectified: np.ndarray) -> np.ndarray:
        return _map_points_with_lookup(
            points_rectified,
            self.distortion_center,
            self.lens_distortion_lookup_table,
            self.max_radius_px,
        )

    def undistort_points(self, points_distorted: np.ndarray) -> np.ndarray:
        return _map_points_with_lookup(
            points_distorted,
            self.distortion_center,
            self.inverse_lens_distortion_lookup_table,
            self.max_radius_px,
        )


@dataclass(frozen=True)
class FrameUndistorter:
    calibration: DepthCalibration
    map_x: np.ndarray
    map_y: np.ndarray

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        self.calibration.validate_image_size(width, height)
        return cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


@dataclass(frozen=True)
class FramePose:
    frame_index: int
    intrinsics: IntrinsicsRecord
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_rmse_px: float
    visible_tag_ids: tuple[int, ...]


@dataclass(frozen=True)
class PlaneCalibration:
    reference_frame_idx: int
    tag_size_mm: float
    apriltag_family: str
    anchor_tag_id: int
    plane_points_by_corner: dict[tuple[int, int], np.ndarray]
    reference_pose: FramePose


def load_intrinsics_sequence(path: str | Path) -> IntrinsicsSequence:
    """Load and parse a per-frame intrinsics JSON dump."""
    payload = json.loads(Path(path).read_text())
    records: list[IntrinsicsRecord] = []
    for item in payload:
        width = int(item["width"])
        height = int(item["height"])
        matrix_payload = item.get("cameraMatrix")
        if matrix_payload is None:
            matrix_payload = item.get("cameraMatrixFlat")
        if matrix_payload is None:
            matrix_payload = item.get("intrinsicMatrix3x3")
        if matrix_payload is None:
            raise CalibrationError(
                "Intrinsics record is missing cameraMatrix/cameraMatrixFlat/intrinsicMatrix3x3."
            )

        camera_matrix, parse_mode = _parse_camera_matrix(
            matrix_payload,
            width,
            height,
        )
        records.append(
            IntrinsicsRecord(
                frame_index=int(item["frameIndex"]),
                width=width,
                height=height,
                camera_matrix=camera_matrix,
                parse_mode=parse_mode,
            )
        )
    return IntrinsicsSequence(records)


def load_depth_calibration(path: str | Path) -> DepthCalibration:
    """Load a capture-wide depth calibration JSON dump."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise CalibrationError("depth_calibration.json must contain a JSON object.")

    if "depthDelivered" not in payload:
        raise CalibrationError("depth_calibration.json is missing depthDelivered.")
    if not bool(payload["depthDelivered"]):
        raise CalibrationError("depth_calibration.json reports depthDelivered=false.")

    matrix_payload = payload.get("intrinsicMatrix")
    if matrix_payload is None:
        matrix_payload = payload.get("intrinsicMatrixFlat")
    if matrix_payload is None:
        raise CalibrationError(
            "depth_calibration.json is missing intrinsicMatrix/intrinsicMatrixFlat."
        )

    ref_dims = payload.get("intrinsicMatrixReferenceDimensions")
    if not isinstance(ref_dims, list) or len(ref_dims) != 2:
        raise CalibrationError(
            "depth_calibration.json is missing intrinsicMatrixReferenceDimensions."
        )
    reference_width = int(ref_dims[0])
    reference_height = int(ref_dims[1])
    if reference_width <= 0 or reference_height <= 0:
        raise CalibrationError("Depth calibration reference dimensions must be positive.")

    camera_matrix, _parse_mode = _parse_camera_matrix(
        matrix_payload,
        reference_width,
        reference_height,
    )

    distortion_center = payload.get("lensDistortionCenter")
    if not isinstance(distortion_center, list) or len(distortion_center) != 2:
        raise CalibrationError("depth_calibration.json is missing lensDistortionCenter.")

    lens_lut = np.asarray(payload.get("lensDistortionLookupTable"), dtype=np.float64)
    inverse_lut = np.asarray(
        payload.get("inverseLensDistortionLookupTable"),
        dtype=np.float64,
    )
    if lens_lut.ndim != 1 or lens_lut.size < 2:
        raise CalibrationError(
            "depth_calibration.json must provide a non-empty lensDistortionLookupTable."
        )
    if inverse_lut.ndim != 1 or inverse_lut.size < 2:
        raise CalibrationError(
            "depth_calibration.json must provide a non-empty inverseLensDistortionLookupTable."
        )

    return DepthCalibration(
        intrinsic_matrix=np.asarray(camera_matrix, dtype=np.float64),
        reference_width=reference_width,
        reference_height=reference_height,
        distortion_center=np.asarray(distortion_center, dtype=np.float64),
        lens_distortion_lookup_table=lens_lut,
        inverse_lens_distortion_lookup_table=inverse_lut,
        depth_delivered=bool(payload["depthDelivered"]),
    )


def build_frame_undistorter(
    calibration: DepthCalibration,
    width: int,
    height: int,
) -> FrameUndistorter:
    """Precompute a cv2.remap grid for raw distorted -> rectified frames."""
    calibration.validate_image_size(width, height)

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    center_x = float(calibration.distortion_center[0])
    center_y = float(calibration.distortion_center[1])
    delta_x = grid_x - center_x
    delta_y = grid_y - center_y
    radii = np.hypot(delta_x, delta_y)
    scales = _lookup_scales(
        radii,
        calibration.lens_distortion_lookup_table,
        calibration.max_radius_px,
    )
    map_x = np.asarray(center_x + delta_x * scales, dtype=np.float32)
    map_y = np.asarray(center_y + delta_y * scales, dtype=np.float32)
    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise CalibrationError("Depth calibration produced a non-finite undistortion remap.")

    return FrameUndistorter(calibration=calibration, map_x=map_x, map_y=map_y)


def build_plane_calibration(
    frame: np.ndarray,
    intrinsics: IntrinsicsRecord,
    *,
    reference_frame_idx: int,
    tag_size_mm: float,
    apriltag_family: str = "36h11",
) -> PlaneCalibration:
    """Use the reference frame tags to define a metric XY plane."""
    detections = detect_apriltags(frame, apriltag_family)
    if not detections:
        raise CalibrationError("No AprilTags were detected in the reference frame.")

    anchor_tag_id = min(detections)
    ok, rvec_anchor, tvec_anchor = _solve_planar_pnp(
        _square_object_points(tag_size_mm),
        detections[anchor_tag_id],
        intrinsics.camera_matrix,
    )
    if not ok:
        raise CalibrationError(
            f"Could not solve the reference plane pose from AprilTag {anchor_tag_id}."
        )

    pose_anchor = FramePose(
        frame_index=reference_frame_idx,
        intrinsics=intrinsics,
        rvec=rvec_anchor,
        tvec=tvec_anchor,
        reprojection_rmse_px=0.0,
        visible_tag_ids=(anchor_tag_id,),
    )
    plane_points_by_corner: dict[tuple[int, int], np.ndarray] = {}
    for tag_id, corners_px in detections.items():
        corners_plane = image_points_to_plane(corners_px, pose_anchor)
        for corner_idx, pt_plane in enumerate(corners_plane):
            plane_points_by_corner[(int(tag_id), corner_idx)] = np.asarray(pt_plane, dtype=float)

    ref_pose = estimate_frame_pose(
        frame,
        intrinsics,
        reference_frame_idx,
        plane_points_by_corner,
        apriltag_family=apriltag_family,
    )
    if ref_pose is None:
        raise CalibrationError("Could not fit a consistent reference-frame AprilTag pose.")

    return PlaneCalibration(
        reference_frame_idx=reference_frame_idx,
        tag_size_mm=float(tag_size_mm),
        apriltag_family=apriltag_family,
        anchor_tag_id=int(anchor_tag_id),
        plane_points_by_corner=plane_points_by_corner,
        reference_pose=ref_pose,
    )


def detect_apriltags(frame: np.ndarray, apriltag_family: str = "36h11") -> dict[int, np.ndarray]:
    """Return detected tag corners keyed by tag id."""
    detector = _aruco_detector(apriltag_family)
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(tag_id): np.squeeze(corner_set).astype(np.float64)
        for tag_id, corner_set in zip(ids.flatten(), corners)
    }


def estimate_frame_pose(
    frame: np.ndarray,
    intrinsics: IntrinsicsRecord,
    frame_index: int,
    plane_points_by_corner: dict[tuple[int, int], np.ndarray],
    *,
    apriltag_family: str = "36h11",
) -> FramePose | None:
    """Estimate the current frame's plane pose from visible tag corners."""
    detections = detect_apriltags(frame, apriltag_family)
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    visible_tag_ids: list[int] = []
    for tag_id, corners_px in sorted(detections.items()):
        used_any = False
        for corner_idx, corner_px in enumerate(corners_px):
            pt_plane = plane_points_by_corner.get((int(tag_id), corner_idx))
            if pt_plane is None:
                continue
            object_points.append([float(pt_plane[0]), float(pt_plane[1]), 0.0])
            image_points.append([float(corner_px[0]), float(corner_px[1])])
            used_any = True
        if used_any:
            visible_tag_ids.append(int(tag_id))

    if len(object_points) < 4:
        return None

    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    ok, rvec, tvec = _solve_planar_pnp(obj, img, intrinsics.camera_matrix)
    if not ok:
        return None

    reproj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.camera_matrix, None)
    reproj = reproj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((reproj - img) ** 2, axis=1))))
    return FramePose(
        frame_index=int(frame_index),
        intrinsics=intrinsics,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reprojection_rmse_px=rmse,
        visible_tag_ids=tuple(visible_tag_ids),
    )


def image_points_to_plane(points_px: np.ndarray, pose: FramePose) -> np.ndarray:
    """Project image points onto the calibrated z=0 plane."""
    h_inv = np.linalg.inv(_plane_homography(pose))
    return _apply_homography(h_inv, points_px)


def plane_points_to_image(points_plane: np.ndarray, pose: FramePose) -> np.ndarray:
    """Project plane points back into the image."""
    return _apply_homography(_plane_homography(pose), points_plane)


def plane_points_to_object(points_plane: np.ndarray) -> np.ndarray:
    """Convert 2D plane points to z=0 object points."""
    pts = np.asarray(points_plane, dtype=float)
    return np.column_stack((pts, np.zeros(len(pts), dtype=float)))


def _parse_camera_matrix(
    raw_values: list[float] | np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(raw_values, dtype=np.float64)
    if arr.shape == (3, 3):
        if _looks_like_camera_matrix(arr, width, height):
            return arr.astype(np.float64), "row-major-2d"
        flat = arr.reshape(-1)
    else:
        flat = arr.reshape(-1)

    if flat.size != 9:
        raise CalibrationError(f"Expected 9 intrinsic values, got {flat.size}.")

    mat_row = flat.reshape(3, 3)
    if _looks_like_camera_matrix(mat_row, width, height):
        return mat_row, "row-major"

    mat_col = flat.reshape(3, 3, order="F")
    if _looks_like_camera_matrix(mat_col, width, height):
        return mat_col, "column-major"

    return _heuristic_camera_matrix(flat, width, height), "heuristic-partial"

def _looks_like_camera_matrix(matrix: np.ndarray, width: int, height: int) -> bool:
    return bool(
        matrix.shape == (3, 3)
        and matrix[0, 0] > 0.0
        and matrix[1, 1] > 0.0
        and abs(matrix[2, 2] - 1.0) < 1e-3
        and abs(matrix[0, 1]) < 1e-6
        and abs(matrix[1, 0]) < 1e-6
        and abs(matrix[2, 0]) < 1e-6
        and abs(matrix[2, 1]) < 1e-6
        and 0.0 <= matrix[0, 2] <= width * 1.5
        and 0.0 <= matrix[1, 2] <= height * 1.5
    )


def _heuristic_camera_matrix(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    fx = float(arr[0]) if arr[0] > 0 else float(max(arr))
    fy = next(
        (
            float(arr[idx])
            for idx in (4, 5, 1)
            if idx < len(arr) and arr[idx] > 0 and abs(arr[idx] - fx) > 1e-9
        ),
        float(fx),
    )
    if fy <= 0.0:
        fy = float(fx)

    principal_candidates = [
        float(val)
        for val in arr
        if 0.0 < val < max(width, height) * 1.1 and abs(val - fx) > 1e-9 and abs(val - fy) > 1e-9
    ]

    cx = float(width) / 2.0
    cy = float(height) / 2.0
    if len(principal_candidates) == 1:
        val = principal_candidates[0]
        if abs(val - cx) <= abs(val - cy):
            cx = val
        else:
            cy = val
    elif len(principal_candidates) >= 2:
        cx = min(principal_candidates, key=lambda val: abs(val - (float(width) / 2.0)))
        remaining = [val for val in principal_candidates if abs(val - cx) > 1e-9]
        if remaining:
            cy = min(remaining, key=lambda val: abs(val - (float(height) / 2.0)))

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _map_points_with_lookup(
    points_xy: np.ndarray,
    distortion_center: np.ndarray,
    lookup_table: np.ndarray,
    max_radius_px: float,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(distortion_center, dtype=np.float64).reshape(2)
    delta = pts - center
    radii = np.linalg.norm(delta, axis=-1)
    scales = _lookup_scales(radii, np.asarray(lookup_table, dtype=np.float64), max_radius_px)
    mapped = center + delta * scales[..., None]
    return mapped


def _lookup_scales(
    radii_px: np.ndarray,
    lookup_table: np.ndarray,
    max_radius_px: float,
) -> np.ndarray:
    radii = np.asarray(radii_px, dtype=np.float64)
    table = np.asarray(lookup_table, dtype=np.float64).reshape(-1)
    if table.size == 0:
        raise CalibrationError("Lens distortion lookup table is empty.")
    if max_radius_px <= 0.0:
        raise CalibrationError("Lens distortion maximum radius must be positive.")
    if table.size == 1:
        return np.ones_like(radii, dtype=np.float64) + float(table[0])

    pos = np.clip(radii * (table.size - 1) / max_radius_px, 0.0, float(table.size - 1))
    idx0 = np.floor(pos).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, table.size - 1)
    frac = pos - idx0
    delta = (1.0 - frac) * table[idx0] + frac * table[idx1]
    # Apple exports these tables as an additive radial delta; 0 means no distortion.
    return 1.0 + delta


def _square_object_points(tag_size_mm: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [tag_size_mm, 0.0, 0.0],
            [tag_size_mm, tag_size_mm, 0.0],
            [0.0, tag_size_mm, 0.0],
        ],
        dtype=np.float64,
    )


def _solve_planar_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            None,
            flags=flag,
        )
        if ok:
            return True, rvec, tvec
    return False, np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)


def _plane_homography(pose: FramePose) -> np.ndarray:
    rot_mat, _ = cv2.Rodrigues(pose.rvec)
    return pose.intrinsics.camera_matrix @ np.column_stack(
        (rot_mat[:, 0], rot_mat[:, 1], pose.tvec.reshape(3))
    )


def _apply_homography(h_mat: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    pts_2d = np.atleast_2d(pts)
    homo = np.column_stack((pts_2d, np.ones(len(pts_2d), dtype=float)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mapped = (h_mat @ homo.T).T
        mapped_xy = mapped[:, :2] / mapped[:, 2:3]
    bad = (~np.isfinite(mapped_xy).all(axis=1)) | (np.abs(mapped[:, 2]) < 1e-12)
    if np.any(bad):
        mapped_xy[bad] = np.nan
    if pts.ndim == 1:
        return mapped_xy[0]
    return mapped_xy


@lru_cache(maxsize=None)
def _aruco_detector(apriltag_family: str) -> cv2.aruco.ArucoDetector:
    dict_id = APRILTAG_FAMILY_IDS.get(apriltag_family)
    if dict_id is None:
        raise CalibrationError(
            f"Unsupported AprilTag family '{apriltag_family}'. "
            f"Choose from {sorted(APRILTAG_FAMILY_IDS)}."
        )
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)
