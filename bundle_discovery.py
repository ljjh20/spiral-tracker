from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from reference_clones.geometry_clone import Constraints, Material, NodeConfig, ParameterSpace, Specs, Spring
except ImportError:
    from geometry_clone import Constraints, Material, NodeConfig, ParameterSpace, Specs, Spring


PICKLE_CLASS_MAP = {
    ("src.geometry", "Spring"): Spring,
    ("src.geometry", "Material"): Material,
    ("src.geometry", "Constraints"): Constraints,
    ("src.geometry", "Specs"): Specs,
    ("src.geometry", "NodeConfig"): NodeConfig,
    ("src.parameter_space", "ParameterSpace"): ParameterSpace,
}


class SpringUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        mapped = PICKLE_CLASS_MAP.get((module, name))
        if mapped is not None:
            return mapped
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return SpringUnpickler(handle).load()


def _is_spring(obj: Any) -> bool:
    return isinstance(obj, Spring)


def _count_springs_in_item(item: Any) -> int:
    if _is_spring(item):
        return 1
    if isinstance(item, dict):
        return sum(1 for value in item.values() if _is_spring(value))
    return 0


def _classify_pickle_object(obj: Any) -> tuple[str, int]:
    if _is_spring(obj):
        return "spring", 1

    if isinstance(obj, dict):
        count = _count_springs_in_item(obj)
        if count:
            return "spring_mapping", count
        return "unknown", 0

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "unknown", 0
        counts = [_count_springs_in_item(item) for item in obj]
        total = sum(counts)
        if total == 0:
            return "unknown", 0
        if all(_is_spring(item) for item in obj):
            return "spring_sequence", total
        if all(isinstance(item, dict) and _count_springs_in_item(item) > 0 for item in obj):
            return "spring_record_sequence", total
        return "spring_collection", total

    return "unknown", 0


@dataclass(frozen=True)
class PickleCandidate:
    path: Path
    kind: str
    spring_count: int

    @property
    def contains_multiple_springs(self) -> bool:
        return self.spring_count > 1


@dataclass(frozen=True)
class CaptureBundle:
    directory: Path
    video_path: Path
    intrinsics_json: Path
    depth_calibration_json: Path


def inspect_pickle(path: Path) -> PickleCandidate | None:
    try:
        payload = load_pickle(path)
    except Exception:
        return None
    kind, spring_count = _classify_pickle_object(payload)
    if spring_count == 0:
        return None
    return PickleCandidate(path=path, kind=kind, spring_count=spring_count)


def discover_pickle_candidates(root: Path) -> list[PickleCandidate]:
    candidates: list[PickleCandidate] = []
    for path in sorted(root.rglob("*.pkl")):
        info = inspect_pickle(path)
        if info is not None:
            candidates.append(info)
    return candidates


def discover_reference_spring_candidates(root: Path) -> list[PickleCandidate]:
    return [candidate for candidate in discover_pickle_candidates(root) if candidate.kind == "spring"]


def discover_sweep_candidates(root: Path) -> list[PickleCandidate]:
    return [candidate for candidate in discover_pickle_candidates(root) if candidate.kind != "spring"]


def _candidate_lines(candidates: list[PickleCandidate]) -> str:
    return "\n".join(f"  - {candidate.path}" for candidate in candidates)


def resolve_unique_reference_spring(root: Path) -> Path | None:
    candidates = discover_reference_spring_candidates(root)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Found multiple single-Spring .pkl files under "
            f"{root}:\n{_candidate_lines(candidates)}"
        )
    return candidates[0].path


def resolve_unique_sweep_pkl(root: Path) -> Path | None:
    candidates = discover_sweep_candidates(root)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Found multiple multi-Spring/sweep .pkl files under "
            f"{root}:\n{_candidate_lines(candidates)}"
        )
    return candidates[0].path


def _capture_bundle_from_dir(directory: Path) -> CaptureBundle | None:
    intrinsics = directory / "intrinsics.json"
    depth = directory / "depth_calibration.json"
    if not intrinsics.exists() or not depth.exists():
        return None

    movs = sorted(directory.glob("*.mov")) + sorted(directory.glob("*.MOV"))
    if not movs:
        return None
    if len(movs) > 1:
        raise ValueError(
            f"Capture directory {directory} contains multiple .mov files: "
            + ", ".join(path.name for path in movs)
        )
    return CaptureBundle(
        directory=directory,
        video_path=movs[0],
        intrinsics_json=intrinsics,
        depth_calibration_json=depth,
    )


def discover_capture_bundles(root: Path) -> list[CaptureBundle]:
    candidates: list[CaptureBundle] = []
    seen: set[Path] = set()
    search_dirs = [root]
    search_dirs.extend(path.parent for path in sorted(root.rglob("intrinsics.json")))
    for directory in search_dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        bundle = _capture_bundle_from_dir(directory)
        if bundle is not None:
            candidates.append(bundle)
    return candidates


def resolve_unique_capture_bundle(root: Path) -> CaptureBundle | None:
    bundles = discover_capture_bundles(root)
    if not bundles:
        return None
    if len(bundles) > 1:
        bundle_lines = "\n".join(f"  - {bundle.directory}" for bundle in bundles)
        raise ValueError(
            "Found multiple capture bundles under "
            f"{root}:\n{bundle_lines}"
        )
    return bundles[0]


def discover_video_candidates(root: Path) -> list[Path]:
    return sorted({path for ext in ("*.mov", "*.MOV") for path in root.rglob(ext)})


def resolve_unique_video(root: Path) -> Path | None:
    candidates = discover_video_candidates(root)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Found multiple video files under "
            f"{root}:\n" + "\n".join(f"  - {path}" for path in candidates)
        )
    return candidates[0]
