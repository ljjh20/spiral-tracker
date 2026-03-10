"""
Archimedean spiral fitting, arc-length integration, and cross-frame dot matching.

Archimedean spiral: r = a + b·θ   (polar, centred on spring arbor)
  a  – radius at θ=0
  b  – radial pitch per radian  (b = (r_max − r_min) / Δθ_total)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, linear_sum_assignment


# ── Spiral fitting ────────────────────────────────────────────────────────────

def _initial_spiral_fit(
    center: np.ndarray,
    dots: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Heuristic Archimedean spiral fit used to initialize the nonlinear solver.

    Strategy
    --------
    1. Convert to polar (r, raw_θ).
    2. Sort by radius (larger r → larger θ for a outward spiral).
    3. Unwrap raw angles to make θ monotonically increasing.
    4. Linear least-squares fit: r = a + b·θ.
    5. Map fitted θ back to the original (unsorted) dot ordering.

    Returns a, b, thetas in the original dot ordering.
    """
    xy = dots - center
    r = np.linalg.norm(xy, axis=1)
    raw_th = np.arctan2(xy[:, 1], xy[:, 0])

    sort_idx = np.argsort(r)
    r_s  = r[sort_idx]
    th_s = raw_th[sort_idx]

    # Monotonic unwrap: consecutive angular step normalised to (−π, π]
    th_u = np.empty_like(th_s)
    th_u[0] = th_s[0]
    for i in range(1, len(th_s)):
        delta = th_s[i] - th_u[i - 1]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        th_u[i] = th_u[i - 1] + delta

    # Linear fit: r = a + b·θ
    A_mat = np.column_stack([np.ones(len(th_u)), th_u])
    coeffs, *_ = np.linalg.lstsq(A_mat, r_s, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    # Restore original dot order
    th_out = np.empty(len(dots))
    th_out[sort_idx] = th_u

    return a, b, th_out


def _spiral_points(
    center: np.ndarray,
    a: float,
    b: float,
    thetas: np.ndarray,
) -> np.ndarray:
    r = a + b * thetas
    return np.column_stack((
        center[0] + r * np.cos(thetas),
        center[1] + r * np.sin(thetas),
    ))


def spiral_fit_rmse(
    center: np.ndarray,
    a: float,
    b: float,
    thetas: np.ndarray,
    dots: np.ndarray,
) -> float:
    """Root-mean-square point-to-spiral distance in pixels."""
    pred = _spiral_points(center, a, b, thetas)
    return float(np.sqrt(np.mean(np.sum((pred - dots) ** 2, axis=1))))


def refine_spiral(
    center_init: np.ndarray,
    dots: np.ndarray,
    *,
    allow_center_refine: bool = False,
    center_bounds_px: float = 50.0,
    theta_reg_weight: float = 0.02,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Nonlinear geometric fit of an Archimedean spiral to *dots*.

    The initial ordering / unwrap comes from the heuristic fit, then the solver
    refines the spiral directly against Cartesian point residuals. By default
    the centre is fixed; set *allow_center_refine* to also optimize the centre.
    """
    center_init = np.asarray(center_init, dtype=float)
    a0, b0, theta0 = _initial_spiral_fit(center_init, dots)

    order = np.argsort(theta0)
    dots_ord = dots[order]
    theta0_ord = theta0[order]
    dtheta0 = np.diff(theta0_ord)
    dtheta0 = np.maximum(dtheta0, 1e-3)
    log_dtheta0 = np.log(dtheta0)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
        idx = 0
        if allow_center_refine:
            center = params[idx:idx + 2]
            idx += 2
        else:
            center = center_init

        a = float(params[idx])
        b = float(params[idx + 1])
        theta_first = float(params[idx + 2])
        log_dtheta = params[idx + 3:]

        theta_ord = np.empty_like(theta0_ord)
        theta_ord[0] = theta_first
        if len(theta_ord) > 1:
            theta_ord[1:] = theta_first + np.cumsum(np.exp(log_dtheta))
        return np.asarray(center, dtype=float), a, b, theta_ord

    def residual(params: np.ndarray) -> np.ndarray:
        center, a, b, theta_ord = unpack(params)
        pred = _spiral_points(center, a, b, theta_ord)
        geom = (pred - dots_ord).ravel()
        reg = theta_reg_weight * (theta_ord - theta0_ord)
        return np.concatenate((geom, reg))

    if allow_center_refine:
        x0 = np.concatenate((
            center_init,
            np.array([a0, b0, theta0_ord[0]], dtype=float),
            log_dtheta0,
        ))
        lower = np.concatenate((
            center_init - center_bounds_px,
            np.array([-np.inf, -np.inf, -np.inf], dtype=float),
            np.full_like(log_dtheta0, -20.0),
        ))
        upper = np.concatenate((
            center_init + center_bounds_px,
            np.array([np.inf, np.inf, np.inf], dtype=float),
            np.full_like(log_dtheta0, 20.0),
        ))
    else:
        x0 = np.concatenate((
            np.array([a0, b0, theta0_ord[0]], dtype=float),
            log_dtheta0,
        ))
        lower = np.concatenate((
            np.array([-np.inf, -np.inf, -np.inf], dtype=float),
            np.full_like(log_dtheta0, -20.0),
        ))
        upper = np.concatenate((
            np.array([np.inf, np.inf, np.inf], dtype=float),
            np.full_like(log_dtheta0, 20.0),
        ))

    res = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=10000,
    )

    best_center, a, b, theta_ord = unpack(res.x)
    theta_out = np.empty_like(theta0)
    theta_out[order] = theta_ord
    return best_center, a, b, theta_out


def fit_spiral(
    center: np.ndarray,
    dots: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Fit r = a + b·θ to *dots* (N×2, pixel coords) relative to *center*.

    Uses the heuristic unwrap to initialize a nonlinear geometric refinement
    with the supplied centre held fixed.
    """
    _, a, b, thetas = refine_spiral(center, dots, allow_center_refine=False)
    return a, b, thetas


def refine_center(
    center_init: np.ndarray,
    dots: np.ndarray,
    bounds_px: float = 50.0,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Numerically refine the spiral centre to minimise residuals of r = a + b·θ.

    *bounds_px* is the search radius around *center_init* (pixels).

    Returns: center, a, b, thetas
    """

    center, a, b, th = refine_spiral(
        center_init,
        dots,
        allow_center_refine=True,
        center_bounds_px=bounds_px,
    )
    return center, a, b, th


# ── Arc-length integration ────────────────────────────────────────────────────

def _arc_integrand(th: float, a: float, b: float) -> float:
    """ds/dθ = √(r² + b²) for r = a + b·θ."""
    return np.sqrt((a + b * th) ** 2 + b ** 2)


def arc_length_between(a: float, b: float, th_start: float, th_end: float) -> float:
    """Exact arc length of r = a + b·θ from th_start to th_end (signed)."""
    if abs(th_end - th_start) < 1e-14:
        return 0.0
    val, _ = quad(_arc_integrand, th_start, th_end, args=(a, b), limit=200)
    return float(val)


def arc_lengths_from_min(a: float, b: float, thetas: np.ndarray) -> np.ndarray:
    """
    Arc length from the innermost dot (min θ) to every dot.

    Returns an (N,) array in the same units as *a* (pixels unless scaled).
    """
    th0 = thetas.min()
    lengths = np.array([arc_length_between(a, b, th0, th) for th in thetas])
    return lengths


# ── Cross-frame dot matching ──────────────────────────────────────────────────

def match_dots(
    ref: np.ndarray,
    cur: np.ndarray,
    max_dist: float = np.inf,
) -> np.ndarray:
    """
    Hungarian matching of *cur* dots to *ref* dots.

    Cost matrix shape: (n_ref, n_cur).
    Returns *assignment* of length n_ref: assignment[i] = j means ref[i] is
    matched to cur[j].  assignment[i] = -1 if ref[i] has no match within
    *max_dist*.
    """
    n_ref, n_cur = len(ref), len(cur)
    assignment = np.full(n_ref, -1, dtype=int)

    if n_ref == 0 or n_cur == 0:
        return assignment

    # Pairwise Euclidean distance
    diff = ref[:, None, :] - cur[None, :, :]   # (n_ref, n_cur, 2)
    cost = np.linalg.norm(diff, axis=2)          # (n_ref, n_cur)

    row_ind, col_ind = linear_sum_assignment(cost)
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] <= max_dist:
            assignment[ri] = ci

    return assignment
