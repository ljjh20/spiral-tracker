#!/usr/bin/env python3
"""Sweep `Specs.rom` for a saved Spring pickle and rerun `single_run.py` logic.

Examples
--------
python pkl_sweep_generator.py springs/PLA_test.pkl --rom-min 0 --rom-max pi --count 5
python pkl_sweep_generator.py springs/PLA_test.pkl --rom-min -pi/2 --rom-max 0 --step pi/12
python pkl_sweep_generator.py springs/PLA_test.pkl --rom-min 0 --rom-max pi --count 9 --torque-spec none --individual-dir sweeps/pla_cases
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "beam_designer_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

try:
    from reference_clones.single_run_clone import single_run
    from reference_clones.geometry_clone import Specs, Spring
except ImportError:
    from single_run_clone import single_run
    from geometry_clone import Specs, Spring


def _parse_scalar_expr(text: str) -> float:
    allowed_names = {
        "pi": math.pi,
        "tau": math.tau,
        "e": math.e,
    }
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in allowed_names:
            return float(allowed_names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return float(allowed_binops[type(node.op)](_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](_eval(node.operand)))
        raise ValueError(
            f"Unsupported expression {text!r}. Use numeric values or simple expressions such as 'pi', 'pi/2', or '3*pi/4'."
        )

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid numeric expression {text!r}") from exc
    return float(_eval(tree))


def _build_rom_values(
    rom_min: float,
    rom_max: float,
    count: Optional[int],
    step: Optional[float],
) -> np.ndarray:
    if np.isclose(rom_min, rom_max):
        return np.array([float(rom_min)], dtype=float)

    if (count is None) == (step is None):
        raise ValueError("Provide exactly one of --count or --step.")

    if count is not None:
        if count < 2:
            raise ValueError("--count must be at least 2 when rom_min != rom_max.")
        values = np.linspace(float(rom_min), float(rom_max), int(count))
        values[0] = float(rom_min)
        values[-1] = float(rom_max)
        return values

    if step is None or step <= 0.0:
        raise ValueError("--step must be positive.")

    n_segments = max(int(np.ceil(abs(rom_max - rom_min) / float(step))), 1)
    values = np.linspace(float(rom_min), float(rom_max), n_segments + 1)
    values[0] = float(rom_min)
    values[-1] = float(rom_max)
    return values


def _resolve_torque_spec(raw_value: str, source_spring: Spring) -> Optional[float]:
    lowered = raw_value.strip().lower()
    if lowered == "none":
        return None
    if lowered == "infer":
        loading = np.asarray(getattr(source_spring, "loading", np.array([])), dtype=float).ravel()
        if loading.size >= 3:
            return float(loading[2])
        return None
    return _parse_scalar_expr(raw_value)


def _default_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}_rom_sweep.pkl")


def _default_individual_dir(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}_rom_sweep_springs")


def _rom_slug(index: int, rom: float) -> str:
    raw = f"{index:03d}_{rom:+.6f}rad"
    return raw.replace("+", "p").replace("-", "m").replace(".", "p")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a Spring pickle, sweep Specs.rom, rerun single_run, and save the resulting Spring objects."
    )
    parser.add_argument("input_pkl", help="Path to a pickle containing a Spring object.")
    parser.add_argument("--rom-min", required=True, help="Minimum ROM in radians. Supports expressions like '0', 'pi/2', '-pi'.")
    parser.add_argument("--rom-max", required=True, help="Maximum ROM in radians. Supports expressions like 'pi'.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--count", type=int, help="Number of sweep points, inclusive of both endpoints.")
    group.add_argument("--step", help="Maximum ROM spacing in radians. Endpoints are still forced exactly to rom-min/rom-max.")

    parser.add_argument(
        "--torque-spec",
        default="infer",
        help="Applied moment used in reruns. Use 'infer' (default) to reuse source loading[2], 'none' for unconstrained moment, or a numeric expression.",
    )
    parser.add_argument("--method", choices=("BFGS", "CMAES"), help="Optimizer to use. Defaults to the source spring's method.")
    parser.add_argument("--do-norm", action="store_true", help="Pass do_norm=True into single_run.")
    parser.add_argument("--output", help="Path for the collated output pickle. Defaults next to the input pickle.")
    parser.add_argument(
        "--individual-dir",
        help="Optional directory for one Spring pickle per ROM value. If omitted, only the collated pickle is written.",
    )
    parser.add_argument(
        "--name-prefix",
        help="Prefix used for per-run naming. Defaults to the source pickle stem.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    source_path = Path(args.input_pkl).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"No such input pickle: {source_path}")

    with source_path.open("rb") as fh:
        source_spring = pickle.load(fh)
    if not isinstance(source_spring, Spring):
        raise TypeError(f"{source_path} does not contain a Spring object. Found {type(source_spring)}.")
    if getattr(source_spring.node_config, "geo", None) != "guided":
        raise ValueError(
            f"Source spring geometry is {source_spring.node_config.geo!r}. Sweeping Specs.rom only affects guided springs."
        )

    rom_min = _parse_scalar_expr(args.rom_min)
    rom_max = _parse_scalar_expr(args.rom_max)
    rom_step = None if args.step is None else _parse_scalar_expr(args.step)
    rom_values = _build_rom_values(rom_min, rom_max, args.count, rom_step)

    torque_spec = _resolve_torque_spec(args.torque_spec, source_spring)
    method = args.method or getattr(source_spring, "method", "BFGS")
    num_nodes = int(source_spring.nodes.shape[0])
    use_timoshenko = bool(getattr(source_spring, "use_timoshenko", True))
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    individual_dir = None
    if args.individual_dir is not None:
        individual_dir = Path(args.individual_dir).expanduser().resolve()
        individual_dir.mkdir(parents=True, exist_ok=True)

    name_prefix = args.name_prefix or source_path.stem
    base = np.asarray(source_spring.nodes[:, 6], dtype=float).copy()
    height = np.asarray(source_spring.nodes[:, 7], dtype=float).copy()
    constraints = copy.deepcopy(source_spring.constraints)
    material = copy.deepcopy(source_spring.material)
    displacement = copy.deepcopy(source_spring.specs.displacement)
    source_rom = float(source_spring.specs.rom)

    records: list[dict] = []
    total = len(rom_values)

    print(f"Loaded source spring: {source_path}")
    print(f"Sweeping {total} rom value(s) from {rom_values[0]:.6f} rad to {rom_values[-1]:.6f} rad")
    print(f"Method={method}, num_nodes={num_nodes}, torque_spec={torque_spec}")

    for idx, rom in enumerate(rom_values):
        run_name = f"{name_prefix}_{_rom_slug(idx, float(rom))}"
        specs = Specs(
            torque=float(torque_spec) if torque_spec is not None else float(source_spring.specs.torque),
            rom=float(rom),
            displacement=copy.deepcopy(displacement),
        )
        print(f"[{idx + 1}/{total}] rom={rom:.6f} rad ({np.degrees(rom):.3f} deg)")
        spring = single_run(
            method=method,
            optimize_height=False,
            do_norm=args.do_norm,
            num_nodes=num_nodes,
            constraints=constraints,
            specs=specs,
            material=material,
            base=base.copy(),
            height=height.copy(),
            torque_spec=torque_spec,
            use_timoshenko=use_timoshenko,
            run_name=run_name,
            save_spring=False,
            show_plot=False,
            show_ures=False,
            animation_format=None,
            real_time=False,
        )

        individual_path = None
        if individual_dir is not None:
            individual_path = individual_dir / f"{run_name}.pkl"
            spring.save(str(individual_path))

        loading = np.asarray(spring.loading, dtype=float).ravel()
        record = {
            "index": idx,
            "name": run_name,
            "source_pkl": str(source_path),
            "source_rom": source_rom,
            "rom": float(rom),
            "rom_deg": float(np.degrees(rom)),
            "torque_spec": None if torque_spec is None else float(torque_spec),
            "method": method,
            "do_norm": bool(args.do_norm),
            "num_nodes": num_nodes,
            "use_timoshenko": use_timoshenko,
            "loading_fx": float(loading[0]),
            "loading_fy": float(loading[1]),
            "loading_m": float(loading[2]),
            "run_time_s": float(getattr(spring, "run_time", float("nan"))),
            "volume_m3": float(spring.get_volume()),
            "mass_kg": float(spring.get_mass()),
            "max_stress_pa": float(np.max(np.abs(spring.nodes[:, 12]))),
            "individual_pkl": None if individual_path is None else str(individual_path),
            "spring": spring,
        }
        records.append(record)

    with output_path.open("wb") as fh:
        pickle.dump(records, fh)

    print(f"Saved {len(records)} swept spring(s) to {output_path}")
    if individual_dir is not None:
        print(f"Saved individual spring pickles to {individual_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
