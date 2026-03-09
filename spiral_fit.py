"""
Archimedean spiral fitting, arc-length integration, and cross-frame dot matching.

Archimedean spiral: r = a + b·θ   (polar, centred on spring arbor)
  a  – radius at θ=0
  b  – radial pitch per radian  (b = (r_max − r_min) / Δθ_total)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment, minimize


# ── Spiral fitting ────────────────────────────────────────────────────────────

def fit_spiral(
    center: np.ndarray,
    dots: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Fit r = a + b·θ to *dots* (N×2, pixel coords) relative to *center*.

    Strategy
    --------
    1. Convert to polar (r, raw_θ).
    2. Sort by radius (larger r → larger θ for a outward spiral).
    3. Unwrap raw angles to make θ monotonically increasing.
    4. Linear least-squares fit: r = a + b·θ.
    5. Map fitted θ back to the original (unsorted) dot ordering.

    Returns
    -------
    a, b  : spiral parameters (pixels / radian)
    thetas : (N,) unwrapped angle assigned to each dot, in original dot order.
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

    def residual(c):
        try:
            a, b, th = fit_spiral(np.array(c), dots)
        except Exception:
            return 1e9
        r = np.linalg.norm(dots - np.array(c), axis=1)
        r_fit = a + b * th
        return float(np.mean((r - r_fit) ** 2))

    res = minimize(
        residual,
        center_init,
        method="Nelder-Mead",
        options={"xatol": 0.5, "fatol": 1e-3, "maxiter": 2000},
    )
    best_c = np.array(res.x)
    a, b, th = fit_spiral(best_c, dots)
    return best_c, a, b, th


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
