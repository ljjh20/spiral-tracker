# Usage: python single_run.py [flags]
#
# General:
#   -h                  Optimise height profile (also sets torque_spec = -2.0)
#   -n <name>           Base name for all output files (default: opt_run_<timestamp>)
#   -s                  Save spring to springs/<name>.pkl
#   -p                  Show static analysis plot after run
#   -ures               Show displacement (ures) plot after run
#
# Animation (-gif or -mp4):
#   -gif                Save optimisation animation as traces/<name>.gif
#   -mp4                Save optimisation animation as traces/<name>.mp4 (requires ffmpeg)
#   -hist-every <int>   Record a snapshot every N optimiser iterations (default: 5)
#   -gif-dur <int>      Target video duration in seconds (default: 7)
#   -gif-fps <int>      Playback fps (default: 30); frames are subsampled to fit -gif-dur
#                       when there are too many, otherwise the video is shorter.
#   -gif-dpi <int>      Animation resolution in DPI (default: 100)
#   -gif-no-heights     Hide the height profile panel from the animation
#   -real-time          Match animation length to spring.run_time; each snapshot plays once
#                       (overrides -gif-fps / -gif-dur; defaults -hist-every to 1)

import numpy as np
from math import hypot, asinh
import time
from datetime import datetime
import sys
import copy
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

try:
    from reference_clones.geometry_clone import Spring, Constraints, Material, Specs
except ImportError:
    from geometry_clone import Spring, Constraints, Material, Specs
from spring_diagnostics import resize_height_vector
from src.materials import get_material
from src.utilities import discrete_beam, discrete_beam_45, discrete_beam_at_angle, discrete_beam_quarter_circle, archimedean_spiral_inverted, spiral_arclength_NR
from src.utilities import plot_analysis, plot_ures, save_optimization_animation
from spring_diagnostics import diagnose_height_sampling, make_resample_profile_from_pkl
from src.optimizer import DeformationOptimizer, OptimizationHistory
from src.parameter_space import ParameterSpace

def _arg_value(flag: str) -> Optional[str]:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
            return sys.argv[i + 1]
    return None

def _arg_int(flag: str, default: int) -> int:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return default
    return default

def single_run(method: str, 
               optimize_height: bool, 
               do_norm: bool, 
               num_nodes: int,
               constraints: Constraints, 
               specs: Specs, 
               material: Material, 
               base: np.ndarray, 
               height: np.ndarray,
               torque_spec: Optional[float] = None,
               use_timoshenko: bool = True,
               run_name: Optional[str] = None,
               save_spring: Optional[bool] = None,
               show_plot: Optional[bool] = None,
               show_ures: Optional[bool] = None,
               animation_format: Optional[str] = None,
               hist_every: Optional[int] = None,
               gif_dur: Optional[int] = None,
               gif_fps: Optional[int] = None,
               gif_dpi: Optional[int] = None,
               show_heights: Optional[bool] = None,
               real_time: Optional[bool] = None):
    
    f_iso = 150
    if not torque_spec == None:        
        loading_init = np.array([0.0, 0.0, torque_spec])
        if method == "CMAES":
            e = 1e-12
            motor_bounds = [(-f_iso, f_iso), (-f_iso, f_iso), (torque_spec-e, torque_spec+e)]
        else:
            motor_bounds = [(-f_iso, f_iso), (-f_iso, f_iso), (torque_spec, torque_spec)]
    else:
        loading_init = np.array([0.0, 0.0, 0.0])
        # motor_bounds = [(-0, 0), (-f_iso, f_iso), (-0, 0)]
        motor_bounds = [(-f_iso, f_iso), (-f_iso, f_iso), (-7, 7)]

    node_config = archimedean_spiral_inverted(constraints, num_nodes, equal_arc=True)
    # node_config = discrete_beam(0.05, num_nodes)
    config_e = 1e14 # collocation error penalty 1e8!!! 1e14 for 50 nodes
    weights = [
        config_e,  # internal x
        config_e,  # internal y
        config_e,  # internal th
        config_e,  # end pos x
        config_e,  # end pos y
        config_e,  # end pos th
        1,  # strain e
    ]
    if optimize_height == True:
        height_bounds = [0.0001, 0.025] #0.025
        weights.extend([
            0, #1e15, # volume penalty
            0, # applied moment to moment spec discrepancy
            0, # stress barrier function 
            0, # stress overshoot penalty
            1e12, #1e25!!! # good # stress flatness penalty 1e5!!! 1e12 for 50 nodes
        ])
    else:
        height_bounds = None

    spring = Spring(loading_init, material, constraints, specs, node_config, base, height, method, use_timoshenko=use_timoshenko)
    parameter_space = ParameterSpace.from_configs(spring, motor_bounds, bound_multiplier=15, do_bounds=True,  do_norm=do_norm, height_bounds=height_bounds)
    optimizer = DeformationOptimizer(spring, weights, method, parameter_space, optimize_height=optimize_height)

    name = run_name or _arg_value("-n") or f"opt_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    history = None
    anim_path = None
    real_time = ("-real-time" in sys.argv) if real_time is None else bool(real_time)
    save_spring = ("-s" in sys.argv) if save_spring is None else bool(save_spring)
    plot_requested = ("-p" in sys.argv) if show_plot is None else bool(show_plot)
    show_ures = ("-ures" in sys.argv) if show_ures is None else bool(show_ures)
    show_heights = ("-gif-no-heights" not in sys.argv) if show_heights is None else bool(show_heights)

    if animation_format is None:
        if "-gif" in sys.argv:
            ext = ".gif"
        elif "-mp4" in sys.argv:
            ext = ".mp4"
        else:
            ext = None
    else:
        ext = animation_format if animation_format.startswith(".") else f".{animation_format}"

    if ext is not None:
        if ext == ".mp4":
            import shutil
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found on PATH — install it (e.g. brew install ffmpeg) before using -mp4")
        anim_path = f"traces/{name}{ext}"
        hist_every_default = 1 if real_time else 5
        hist_every = _arg_int("-hist-every", hist_every_default) if hist_every is None else int(hist_every)
        history = OptimizationHistory(every=hist_every)

    startTime = time.perf_counter()
    spring = optimizer.run_optimization(DEBUG=True, history=history)
    endTime = time.perf_counter()
    print(f"Optimization took {endTime - startTime:.2f} seconds")
    spring.run_time = endTime - startTime

    if save_spring:
        spring.save(f"springs/{name}.pkl")

    render_anim = history is not None and history.snapshots
    if plot_requested:
        plot_analysis(spring, plot_deformed_spring=True, block=not render_anim)

    if show_ures:
        plot_ures(spring)

    if history is not None:
        if history.snapshots:
            dpi = _arg_int("-gif-dpi", 300) if gif_dpi is None else int(gif_dpi)
            raw = history.snapshots
            end_pause_frac = 0.2
            if real_time:
                run_time = float(getattr(spring, "run_time", 0.0) or 0.0)
                if run_time > 0.0:
                    fps = len(raw) / run_time
                    resampled = raw
                    end_pause_frac = 0.0
                else:
                    print("spring.run_time missing or non-positive; falling back to -gif-dur/-gif-fps.")
                    real_time = False
            if not real_time:
                fps = _arg_int("-gif-fps", 24) if gif_fps is None else int(gif_fps)
                target_dur = _arg_int("-gif-dur", 7) if gif_dur is None else int(gif_dur)
                # Subsample only when there are too many frames; never repeat frames.
                target_frames = max(int(round(fps * target_dur)), 1)
                if len(raw) > target_frames:
                    indices = np.round(np.linspace(0, len(raw) - 1, target_frames)).astype(int)
                    resampled = [raw[i] for i in indices]
                else:
                    resampled = raw
            save_optimization_animation(
                spring,
                resampled,
                anim_path,
                fps=fps,
                dpi=dpi,
                show_heights=show_heights,
                gui_pause=plot_requested,
                end_pause_frac=end_pause_frac,
            )
        else:
            print("No snapshots recorded; skipping animation.")

    if plot_requested and render_anim:
        plt.show()

    return spring


if __name__ == "__main__":
    method = "BFGS"
    do_norm = False
    num_nodes = 100

    if "-h" in sys.argv:
        optimize_height = True
        torque_spec = -0.5
    else:
        optimize_height = False
        torque_spec = None

    constraints = Constraints(
        r=np.array([0.01, 0.05]),
        th=np.array([0.0, 6*np.pi])
    )
    specs = Specs(
        torque=0.0,
        rom=-np.pi/2, # pi/2
        displacement=[None, 0.003, None]
    )

    material = get_material("PLA")

    base = np.ones(num_nodes) * 0.003175 #0.005
    height = np.ones(num_nodes) * 0.0015
    print(f"MATERIAL: {material}, PHI: {constraints.th}, TORQUE: {torque_spec}")

    spring = single_run(method, 
                        optimize_height, 
                        do_norm, 
                        num_nodes, 
                        constraints, 
                        specs, 
                        material, 
                        base, 
                        height,
                        torque_spec=torque_spec)
    
    # import pdb
    # pdb.set_trace()
    
