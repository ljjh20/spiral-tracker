
import numpy as np
import pandas as pd


def _solidworks_curve_2025(
    csv_file,
    value_name: str,
):
    df = pd.read_csv(
        csv_file,
        skiprows=10,
        header=None,
        usecols=[0, 1, 2],
        names=["point", "x", value_name],
    )
    df["point"] = df["point"].astype(str).str.strip().astype(float)
    df[value_name] = df[value_name].astype(str).str.strip().astype(float)
    df["x"] = df["x"].astype(str).str.strip().astype(float)

    if len(df) > 1:
        x0 = float(df["x"].iloc[0])
        x1 = float(df["x"].iloc[-1])
        if np.isclose(x1, x0):
            df["s"] = np.linspace(0.0, 1.0, len(df))
        else:
            df["s"] = (df["x"] - x0) / (x1 - x0)
    else:
        df["s"] = 0.0
    return df


def ures_solidworks_2025(
    csv_file,
    spring,
    type: str = "ures",
    side: str | None = None,
    report_tip_rotation: bool = True,
):
    df = _solidworks_curve_2025(csv_file, "ures")
    if type == "ures" and side not in {"inner", "outer", None}:
        raise ValueError("side must be 'inner', 'outer', or None")
    if type == "ures":
        df["ures"] = df["ures"] * 1e-3

    # Pick the endpoint closest to zero displacement as the fixed-start side.
    if type == "ures" and len(df) > 1:
        u_start = abs(float(df["ures"].iloc[0]))
        u_end = abs(float(df["ures"].iloc[-1]))
        if u_end < u_start:
            df = df.iloc[::-1].reset_index(drop=True)

    if type == "ures":
        x_root, y_root = spring.construct_root_geometry()
        p_tip = np.array([float(x_root[-1]), float(y_root[-1])], dtype=float)

        # Use undeformed tip tangent/normal to locate inner/outer edge radii.
        if len(x_root) >= 2:
            p_prev = np.array([float(x_root[-2]), float(y_root[-2])], dtype=float)
            t_hat = p_tip - p_prev
            nrm = float(np.hypot(t_hat[0], t_hat[1]))
        else:
            nrm = 0.0
        if nrm <= 1e-14:
            th_tip = float(spring.node_config.angle[-1])
            t_hat = np.array([np.cos(th_tip), np.sin(th_tip)], dtype=float)
        else:
            t_hat = t_hat / nrm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        edge_offset = 0.5 * float(abs(spring.nodes[-1, 7]))  # thickness/2
        p_edge_1 = p_tip + edge_offset * n_hat
        p_edge_2 = p_tip - edge_offset * n_hat
        r_edge_1 = float(np.hypot(p_edge_1[0], p_edge_1[1]))
        r_edge_2 = float(np.hypot(p_edge_2[0], p_edge_2[1]))
        r_inner = min(r_edge_1, r_edge_2)
        r_outer = max(r_edge_1, r_edge_2)
        if side == "inner":
            r_tip = r_inner
        elif side == "outer":
            r_tip = r_outer
        else:
            r_tip = float(np.hypot(p_tip[0], p_tip[1]))

        u_tip = abs(float(df["ures"].iloc[-1]))
        tip_dth = 2 * np.degrees(np.arcsin(np.clip(u_tip / max(2.0 * r_tip, 1e-12), -1.0, 1.0)))
        if report_tip_rotation:
            label = side if side is not None else "centerline"
            print(f"Solidworks Tip Rotation ({label}): {tip_dth} Degrees")

    return df


def stress_solidworks_2025(csv_file):
    return _solidworks_curve_2025(csv_file, "stress")
