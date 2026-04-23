from dataclasses import dataclass
from math import hypot, asinh
import numpy as np
from numba import jit
import pickle
from typing import Optional, Literal

try:
    from .parameter_space import ParameterSpace
except ImportError:
    class ParameterSpace:  # type: ignore[no-redef]
        pass


@dataclass
class NodeConfig:
    x_pos: np.ndarray
    y_pos: np.ndarray
    length: np.ndarray
    angle: np.ndarray
    angle_i: float
    angle_f: float
    geo: Literal["cantilever", "guided"]


@dataclass
class Specs:
    torque: float
    rom: float
    displacement: (
        list  # defined as np.array([x_disp, y_disp]) if y_disp = None, free constraint
    )


@dataclass
class Constraints:
    r: np.ndarray
    th: np.ndarray


@dataclass
class Material:
    E: float
    Sy: float
    fatigue: float
    density: float
    v: float
    G: float
    name: str


@dataclass
class Spring:
    loading: np.ndarray
    material: Material
    constraints: Constraints
    specs: Specs
    node_config: NodeConfig
    base: np.ndarray
    height: np.ndarray
    method: str
    vars_opt: Optional[np.ndarray] = None
    use_kappa_total: bool = False
    use_timoshenko: bool = True
    shear_kappa: float = 5.0 / 6.0  # shear correction factor for rectangular sections
    run_time: float = 1.0
    # states: np.ndarray = None
    # fs: np.ndarray = None
    # iters: int = 0
    # def_history: list = None
    # sigma_hist: np.ndarray = None
    # loading_scale: np.ndarray = None

    def __post_init__(self):
        """Populate node, error arrays"""
        self.nodes, self.errors_internal, self.errors_external = self._initialize_nodes(
            self.node_config, self.base, self.height
        )
        self.compute_state()
        self.compute_errors()

    def spiral_length(self):
        r_in_i, r_out_i = int(np.argmin(np.abs(self.constraints.r))), int(np.argmax(np.abs(self.constraints.r)))
        th_in_i, th_out_i = int(np.argmin(np.abs(self.constraints.th))), int(np.argmax(np.abs(self.constraints.th)))
    
        r_in, r_out = self.constraints.r[r_in_i], self.constraints.r[r_out_i]
        th_in, th_out = self.constraints.th[th_in_i], self.constraints.th[th_out_i]
        dth = th_out - th_in
        dr  = r_out - r_in
        if abs(dth) < 1e-12:
            return 0.0
        if abs(dr) < 1e-12:  # constant radius → circular arc
            return abs(r_in * dth)
        b = dr / dth
        term = (r_out * hypot(r_out, b) - r_in * hypot(r_in, b)) / (2 * b)
        term += 0.5 * b * (asinh(r_out / b) - asinh(r_in / b))
        return abs(term)
    
    def critical_moment(self):
        b = float(np.min(self.base))
        h = float(np.min(self.height))
        L = float(self.spiral_length())
        E = float(self.material.E)
        nu = float(self.material.v)

        # Shear modulus
        G = E / (2.0 * (1.0 + nu))

        # Weak-axis inertia
        Ix = (b * h**3) / 12.0   # bending about width axis
        Iy = (h * b**3) / 12.0   # bending about thickness axis
        I_weak = min(Ix, Iy)

        # Saint-Venant torsion constant for a general rectangle
        a = max(b, h)
        bb = min(b, h)
        J = a * bb**3 * (
            (1.0/3.0)
            - 0.21 * (bb/a) * (1.0 - (bb**4)/(12.0 * a**4))
        )

        # Critical moment
        M_cr = (np.pi / L) * np.sqrt(E * I_weak * G * J)

        return M_cr
    
    def current_buckling_limit(self):
        M = np.max(abs(self.nodes[:, 11]))
        M_cr = abs(self.critical_moment())
        print(f"M critical: {M_cr}")
        return M_cr > M


    def _initialize_nodes(self, node_config: NodeConfig, base: np.ndarray, height: np.ndarray):
        """
        Populate node array with NodeConfig defined geometry in realative node convention.
        Initialize errors array.
        Nodes, Errors stored as properties
        """
        x, y, L, angle, angle_i, angle_f, geo = (
            node_config.x_pos,
            node_config.y_pos,
            node_config.length,
            node_config.angle,
            node_config.angle_i,
            node_config.angle_f,
            node_config.geo,
        )
        n = len(x)

        node_matrix = np.zeros((n, 8), dtype=np.float64)
        error_matrix_internal = np.zeros((n, 3), dtype=np.float64)
        error_matrix_external = np.zeros(3, dtype=np.float64)

        dth = np.empty(n)
        dx = np.empty(n)
        dy = np.empty(n)
        curvature = np.empty(n)

        dth[0] = angle[0]
        dth[1:] = angle[1:] - angle[0:-1]

        dx[0] = x[0]
        dy[0] = y[0]
        dx_abs = x[1:] - x[0:-1]
        dy_abs = y[1:] - y[0:-1]
        s, c = np.sin(angle[:-1]), np.cos(angle[:-1])
        dx[1:] = c * dx_abs + s * dy_abs
        dy[1:] = -s * dx_abs + c * dy_abs
        
        curvature[0] = angle[0] - angle_i
        curvature[1:] = angle[1:] - angle[0:-1]

        node_matrix[:, 0] = np.arange(0, n)
        node_matrix[:, 1] = dth
        node_matrix[:, 2] = L
        node_matrix[:, 3] = dx
        node_matrix[:, 4] = dy
        node_matrix[:, 5] = curvature
        node_matrix[:, 6] = base
        node_matrix[:, 7] = height

        # Extended node storage
        # 0..18: existing columns used throughout the codebase
        # 19..22: optional Timoshenko diagnostics (safe to ignore downstream)
        extended_matrix = np.zeros((n, 23))
        extended_matrix[:, : node_matrix.shape[1]] = node_matrix
        node_matrix = extended_matrix

        return node_matrix, error_matrix_internal, error_matrix_external
    

    def init_parameter_array(self, optimize_height: bool = False):
        """Construct optimizer x0 from node array."""
        positions = self.nodes[:, 3:5].flatten()
        angles = self.nodes[:, 1]
        heights = self.nodes[:, 7]
        n = self.nodes.shape[0]
        force_range = 3
        position_range = 3 + 2 * n
        height_range = position_range + n

        if optimize_height:
            vars = np.zeros(height_range + n)
            vars[height_range : height_range + n] = heights
        
        else:
            vars = np.zeros(position_range + n)

        vars[0 : force_range] = self.loading
        vars[force_range: position_range] = positions
        vars[position_range : position_range + n] = angles

        return vars


    def compute_state(self):
        """Compute node-wise deflection, stress, and strain energy.

        Baseline is Euler-Bernoulli (EB). If use_timoshenko=True, add shear compliance:
          gamma = V/(k G A), def_y_shear = gamma*L, energy_shear = V^2 L/(2 k G A).

        Important: curvature/bending stress should use cross-section rotation φ, not centerline slope θ.
        """

        n = self.nodes.shape[0]
        
        L = self.nodes[:, 2]
        b = self.nodes[:, 6]
        h = self.nodes[:, 7]
        I = (b * h**3) / 12
        A = b * h
        E = self.material.E
        nu = self.material.v
        G = self.material.G if getattr(self.material, "G", None) else E / (2.0 * (1.0 + nu))
        kappa_s = float(self.shear_kappa)  # rectangular-section shear correction

        th_abs = np.cumsum(self.nodes[:, 1])
        s, c = np.sin(th_abs), np.cos(th_abs)
        force_a = c * self.loading[0] + s * self.loading[1]
        force_t = -s * self.loading[0] + c * self.loading[1]

        input_moment = np.zeros(n)
        input_moment[-1] = self.loading[2]
        output_moment = np.zeros(n)
        output_moment[-1] = input_moment[-1] + (force_t[-1] * L[-1])
        input_moment, output_moment = _njit_moment_propagation(
            input_moment, output_moment, force_t, L
        )

        def_y_bend = (0.5 * input_moment * L**2 + (1.0/3.0) * force_t * L**3) / (E * I)

        # EB cross-section rotation at segment end (phi_end)
        phi_end = (input_moment * L + 0.5 * force_t * L**2) / (E * I)

        # Optional Timoshenko shear: shear angle gamma and shear deflection
        if self.use_timoshenko:
            gamma = force_t / (kappa_s * G * A)   # shear angle (dimensionless)
            def_y_shear = gamma * L # = V L/(k G A)
        else:
            gamma = np.zeros_like(L)
            def_y_shear = np.zeros_like(L)

        # Centerline slope at segment end (theta_end) used for tangency propagation
        theta_end = phi_end + gamma

        def_y = def_y_bend + def_y_shear
        a_coeff, b_coeff = (input_moment + force_t * L) / (E * I), -force_t / (2 * E * I)
        J = (a_coeff**2 * L**3)/3 + (a_coeff*b_coeff * L**4)/2 + (b_coeff**2 * L**5)/5
        def_x = (force_a * L) / (E * A) - 0.5 * J

        stress_axial_EB = force_a / A
        stress_b_o_EB = output_moment * (h / 2) / I
        stress_b_i_EB = input_moment * (h / 2) / I
        # stress_b_avg_EB = (stress_b_i + stress_b_o) / 2
        stress_sum_EB = np.abs(stress_axial_EB + stress_b_o_EB)
        # stress_s = np.abs(stress_a + np.maximum(stress_b_i, stress_b_o))

        # Axial stress
        stress_a = force_a / A

        # Local curvature estimates (d(theta)/ds)
        kappa_static = self.nodes[:, 5] / np.maximum(L, 1e-12)  # 1/m
        # Curvature should be based on bending rotation φ, not θ (which includes shear).
        kappa_total = (self.nodes[:, 5] + phi_end) / np.maximum(L, 1e-12)  # 1/m

        # Curved-beam circumferential stress (Winkler–Bach) at inner/outer fibers
        sigma_b_in_static, sigma_b_out_static = _winkler_bach_sigma_rect_njit(output_moment, b, h, kappa_static)
        sigma_b_in_total, sigma_b_out_total = _winkler_bach_sigma_rect_njit(output_moment, b, h, kappa_total)
        # I STILL USE OUTPUT MOMENT FOR STRESS CONTINUITY

        # Node-wise peak normal stress including axial superposition
        sigma_in_total_static = stress_a + sigma_b_in_static
        sigma_out_total_static = stress_a + sigma_b_out_static
        stress_s_static = np.maximum(np.abs(sigma_in_total_static), np.abs(sigma_out_total_static))

        sigma_in_total_kappa = stress_a + sigma_b_in_total
        sigma_out_total_kappa = stress_a + sigma_b_out_total
        stress_s_kappa = np.maximum(np.abs(sigma_in_total_kappa), np.abs(sigma_out_total_kappa))

        if self.use_kappa_total:
            sigma_in_total = sigma_in_total_kappa
            sigma_out_total = sigma_out_total_kappa
            stress_s = stress_s_kappa
            stress_alt = stress_s_static
        else:
            sigma_in_total = sigma_in_total_static
            sigma_out_total = sigma_out_total_static
            stress_s = stress_s_static
            stress_alt = stress_s_kappa

        energy_a = force_a**2 * L / (2 * E * A)
        energy_b = ((input_moment**2 * L) + (input_moment * force_t * L**2) + ((force_t**2 * L**3) / 3)) / (2 * E * I)
        if self.use_timoshenko:
            energy_s = (force_t**2 * L) / (2.0 * kappa_s * G * A)
        else:
            energy_s = np.zeros_like(L)
        strain_energy = energy_a + energy_b + energy_s

        # Store θ_end so internal compatibility enforces centerline tangency including shear.
        self.nodes[:, 8] = theta_end
        self.nodes[:, 9] = def_x
        self.nodes[:, 10] = def_y
        self.nodes[:, 11] = output_moment
        self.nodes[:, 12] = stress_s
        self.nodes[:, 13] = strain_energy
        self.nodes[:, 14] = energy_a
        self.nodes[:, 15] = sigma_in_total
        self.nodes[:, 16] = sigma_out_total
        self.nodes[:, 17] = stress_sum_EB
        self.nodes[:, 18] = stress_alt
        # Optional diagnostics (Timoshenko)
        self.nodes[:, 19] = def_y_shear
        self.nodes[:, 20] = energy_s
        self.nodes[:, 21] = phi_end
        self.nodes[:, 22] = gamma

    def compute_errors(self):
        """Compute collocation error between neighboring nodes, NodeConfig root, and desired tip orientation."""
        n = self.nodes.shape[0]

        dth = self.nodes[:, 1]
        L = self.nodes[:, 2]
        dx = self.nodes[:, 3]
        dy = self.nodes[:, 4]
        curve = self.nodes[:, 5]
        
        def_th = self.nodes[:, 8]
        def_x = self.nodes[:, 9]
        def_y = self.nodes[:, 10]

        e_xi = np.zeros(n)
        e_yi = np.zeros(n)
        e_thi = np.zeros(n)

        e_thi[0] = dth[0] - self.node_config.angle_i - curve[0]
        e_xi[0] = dx[0] - self.node_config.x_pos[0]
        e_yi[0] = dy[0] - self.node_config.y_pos[0]

        e_thi[1:] = dth[1:] - def_th[0:-1] - curve[1:]
        e_xi[1:] = dx[1:] - def_x[0:-1] - L[0:-1]
        e_yi[1:] = dy[1:] - def_y[0:-1]

        th_abs = np.cumsum(dth)

        s, c, = np.sin(th_abs[:-1]), np.cos(th_abs[:-1])
        x_last_base = dx[0] + np.sum(c * dx[1:] - s * dy[1:])
        y_last_base = dy[0] + np.sum(s * dx[1:] + c * dy[1:])

        se, ce = np.sin(th_abs[-1]), np.cos(th_abs[-1])
        def_x_tip = ce * (L[-1] + def_x[-1]) - se * def_y[-1]
        def_y_tip = se * (L[-1] + def_x[-1]) + ce * def_y[-1]

        x_tip = x_last_base + def_x_tip
        y_tip = y_last_base + def_y_tip
        th_tip = th_abs[-1] + def_th[-1]

        e_xt = 0.0
        e_yt = 0.0
        e_tht = 0.0
        
        if self.node_config.geo == "cantilever":
            if self.specs.displacement[0] is not None:
                e_xt = self.specs.displacement[0] - x_tip
            if self.specs.displacement[1] is not None:
                e_yt = self.specs.displacement[1] - y_tip
            if self.specs.displacement[2] is not None:
                e_tht = self.specs.displacement[2] - th_tip - (self.node_config.angle_f - self.node_config.angle[-1])

        elif self.node_config.geo == "guided":
            r_in = self.constraints.r[int(np.argmin(np.abs(self.constraints.r)))]
            th_in = self.constraints.th[int(np.argmin(np.abs(self.constraints.th)))]

            arbor_th = self.node_config.angle_f + self.specs.rom
            arbor_x = r_in * np.cos(th_in + self.specs.rom)
            arbor_y = r_in * np.sin(th_in + self.specs.rom)

            e_xt = arbor_x - x_tip
            e_yt = arbor_y - y_tip
            e_tht = arbor_th - th_tip - (self.node_config.angle_f - self.node_config.angle[-1])

        else:
            raise NotImplementedError

        self.errors_internal[:, 0] = e_xi
        self.errors_internal[:, 1] = e_yi
        self.errors_internal[:, 2] = e_thi
        self.errors_external[0] = e_xt
        self.errors_external[1] = e_yt
        self.errors_external[2] = e_tht

        # import pdb
        # pdb.set_trace()


    def construct_global_spring(self):
        """Reconstruct the geometry in a global coordinate frame."""
        n = self.nodes.shape[0]
        dth = self.nodes[:, 1]
        dx = self.nodes[:, 3]
        dy = self.nodes[:, 4]

        th_abs = np.cumsum(dth)

        x_abs, y_abs = np.zeros(n), np.zeros(n)
        x_abs[0], y_abs[0] = dx[0], dy[0]
        
        s, c, = np.sin(th_abs[:-1]), np.cos(th_abs[:-1])
        x_abs[1:] = dx[0] + np.cumsum(c * dx[1:] - s * dy[1:])
        y_abs[1:] = dy[0] + np.cumsum(s * dx[1:] + c * dy[1:])

        return th_abs, x_abs, y_abs
    
    def construct_root_geometry(self):
        """Express the NodeConfiguration x and y positions in a global coordinate frame."""
        L = self.nodes[:, 2]
        x_root = self.node_config.x_pos
        y_root = self.node_config.y_pos
        root_tip_x = self.node_config.x_pos[-1] + L[-1] * np.cos(self.node_config.angle[-1])
        root_tip_y = self.node_config.y_pos[-1] + L[-1] * np.sin(self.node_config.angle[-1])
        x_root = np.append(x_root, root_tip_x)
        y_root = np.append(y_root, root_tip_y)
        return x_root, y_root

    def construct_ures_representation(self, flip: bool = False):
        """Compute the magnitude change in position of the deformed nodes realtive to the NodeConfig geometry (URES representation). Return ures and a scaled parametric axis."""
        th_abs, x_abs, y_abs = self.construct_global_spring()
        def_x = self.nodes[:, 9]
        def_y = self.nodes[:, 10]
        L = self.nodes[:, 2]

        se, ce = np.sin(th_abs[-1]), np.cos(th_abs[-1])
        def_x_tip = ce * (L[-1] + def_x[-1]) - se * def_y[-1]
        def_y_tip = se * (L[-1] + def_x[-1]) + ce * def_y[-1]

        x_tip = x_abs[-1] + def_x_tip
        y_tip = y_abs[-1] + def_y_tip

        x_def = np.append(x_abs, x_tip)
        y_def = np.append(y_abs, y_tip)

        x_root, y_root = self.construct_root_geometry()

        ures = np.sqrt((x_def - x_root)**2 + (y_def - y_root)**2)

        s_abs = np.concatenate(([0.0], np.cumsum(L)))
        total = s_abs[-1]
        s_norm = s_abs / total
        if flip:
            s_norm -= 1
            s_norm *= -1

        tip_dth = 2 * np.arcsin(
            np.clip(ures[-1] / max(2 * self.constraints.r[0], 1e-12), -1.0, 1.0)
        ) * 180 / np.pi
        print(f"Optimizer Tip Rotation: {tip_dth} Degrees")

        return ures, s_norm
    
    def construct_global_spring_from_vars(self, vars):
        """
        Reconstruct the geometry in a global coordinate frame from the flattened parameter vector.
        """
        n = self.nodes.shape[0]
        f_off = 3
        angle_off = 3 + (2 * n)
        
        dth = vars[angle_off : angle_off + n]
        th_abs = np.cumsum(dth)
        
        pos_vars = vars[f_off : angle_off].reshape(n, 2)
        dx = pos_vars[:, 0]
        dy = pos_vars[:, 1]

        s, c, = np.sin(th_abs[:-1]), np.cos(th_abs[:-1])
        x_abs, y_abs = np.zeros(n), np.zeros(n)
        x_abs[0], y_abs[0] = dx[0], dy[0]

        x_abs[1:] = dx[0] + np.cumsum(c * dx[1:] - s * dy[1:])
        y_abs[1:] = dy[0] + np.cumsum(s * dx[1:] + c * dy[1:])

        return th_abs, x_abs, y_abs
    

    def get_volume(self):
        """Compute the volume of the geometry."""
        volume = 0
        volume = (
            self.nodes[:, 2]
            * self.nodes[:, 6]
            * self.nodes[:, 7]
        ).sum()
        # mass = volume * self.material.density
        return volume
    
    def get_mass(self):
        """Compute the mass of the geometry."""
        volume = 0
        volume = (
            self.nodes[:, 2]
            * self.nodes[:, 6]
            * self.nodes[:, 7]
        ).sum()
        mass = volume * self.material.density
        return mass
    
    def save(self, path: str):
        """Serialize self (including nodes & errors) to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Spring":
        """Load a Spring object from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)



@jit(nopython=True, cache=True)
def _njit_moment_propagation(input_moment, output_moment, ft, L):
    for i in range(input_moment.shape[0] - 2, -1, -1):
        input_moment[i] = output_moment[i+1]
        output_moment[i] = input_moment[i] + ft[i] * L[i]
    return input_moment, output_moment   
    
def winkler_bach_sigma_rect(M: np.ndarray, b: np.ndarray, h: np.ndarray, kappa: np.ndarray, *, eps: float = 1e-12):
    """Curved-beam (Winkler--Bach) circumferential stress for a rectangular section.

    Parameters
    ----------
    M : array
        Signed bending moment (N*m).
    b : array
        Section width (m).
    h : array
        Section radial depth (m) (your variable 'height' across which stress varies).
    kappa : array
        Signed curvature (1/m) of the beam centerline.

    Returns
    -------
    sigma_in, sigma_out : arrays
        Signed circumferential stress (Pa) at inner and outer fibers.

    Notes
    -----
    Falls back to straight-beam Euler–Bernoulli: sigma = M*(h/2)/I when curvature is tiny or r_in <= 0.
    """
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    h = np.asarray(h, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    # Straight-beam fallback (Euler–Bernoulli)
    I = (b * h**3) / 12.0
    sigma_eb = M * (h / 2.0) / np.maximum(I, eps)

    kabs = np.abs(kappa)
    use_curved = kabs > 1e-9  # safe threshold

    R = np.empty_like(kappa, dtype=float)
    R[use_curved] = 1.0 / np.maximum(kabs[use_curved], eps)
    R[~use_curved] = np.inf

    r_i = R - (h / 2.0)
    r_o = R + (h / 2.0)

    valid = use_curved & (r_i > eps) & (r_o > r_i + eps)

    sigma_in = np.copy(sigma_eb)
    sigma_out = np.copy(sigma_eb)

    if np.any(valid):
        ri = r_i[valid]
        ro = r_o[valid]
        A = b[valid] * h[valid]

        ln = np.log(ro / ri)
        rn = (ro - ri) / np.maximum(ln, eps)
        rc = 0.5 * (ro + ri)
        e = rc - rn

        good_e = np.abs(e) > 1e-12
        idx = np.where(valid)[0]

        if np.any(good_e):
            idx2 = idx[good_e]
            ri2 = ri[good_e]
            ro2 = ro[good_e]
            rn2 = rn[good_e]
            e2 = e[good_e]
            A2 = A[good_e]
            M2 = M[idx2]

            sigma_in[idx2] = (M2 / (A2 * e2)) * ((rn2 / ri2) - 1.0)
            sigma_out[idx2] = (M2 / (A2 * e2)) * ((rn2 / ro2) - 1.0)

@jit(nopython=True, cache=True)
def _winkler_bach_sigma_rect_njit(M, b, h, kappa, eps=1e-12):
    """
    Numba-accelerated Winkler--Bach circumferential stress for rectangular sections.
    Inputs: 1D float arrays (same length). Outputs: sigma_in, sigma_out.
    """
    n = M.shape[0]
    sigma_in = np.empty(n, dtype=np.float64)
    sigma_out = np.empty(n, dtype=np.float64)

    for i in range(n):
        Mi = M[i]
        bi = b[i]
        hi = h[i]
        ki = kappa[i]

        # Euler–Bernoulli fallback
        Ii = (bi * hi * hi * hi) / 12.0
        if Ii <= eps:
            sigma_eb = 0.0
        else:
            sigma_eb = Mi * (0.5 * hi) / Ii

        # Curved-beam only if curvature is meaningful
        ak = ki
        if ak < 0.0:
            ak = -ak
        if ak <= 1e-9:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        R = 1.0 / max(ak, eps)
        ri = R - 0.5 * hi
        ro = R + 0.5 * hi

        # Geometry validity
        if ri <= eps or ro <= ri + eps:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        ratio = ro / ri
        if ratio <= 1.0 + 1e-12:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        ln = np.log(ratio)
        if ln <= eps:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        rn = (ro - ri) / ln
        rc = 0.5 * (ro + ri)
        e = rc - rn

        # Scale-aware small-e threshold (straight-beam limit)
        e_thresh = 1e-6 * max(hi, 1e-12)
        if -e_thresh <= e <= e_thresh:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        Ai = bi * hi
        if Ai <= eps:
            sigma_in[i] = sigma_eb
            sigma_out[i] = sigma_eb
            continue

        sigma_in[i] = (Mi / (Ai * e)) * ((rn / ri) - 1.0)
        sigma_out[i] = (Mi / (Ai * e)) * ((rn / ro) - 1.0)

    return sigma_in, sigma_out
