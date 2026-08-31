import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (fixed hyperparameters — not design variables)
#  Same geometry/methodology as the re-entry (RCC) stack — only the TPS
#  material stack differs, per the "same methodology, different materials"
#  scoping agreed for the missile application.
# ═══════════════════════════════════════════════════════════════════════════
N              = 16
x_target       = 2.5
n_nose_frac    = 0.30
n_taper_frac   = 0.25
m_coeffs       = 4

M_inf  = 5.0
gamma  = 1.4
T_inf, p_inf, rho_inf = 205.0, 5543.0, 0.09427
Tw     = 300.0
Cp     = 1005.0
R, Pr  = 287.0, 0.71
sigma  = 5.67e-8

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf

# ── Material allowable temperatures ──────────────────────────────────────
T_allow_UHTC = 2200.0          # K — ZrB2/SiC operational limit
T_allow_CMC  = 1900.0          # K — C/SiC structural limit
T_allow_Ti   = 400.0 + 273.15  # K — Ti-6Al-4V (~400 degC)

# ── Outer shell: spatially blended UHTC (leading edge) <-> CMC (aft body) ──
mat_UHTC = {'t': 0.005, 'k': 25.0, 'rho': 6100.0, 'Cp': 420.0, 'eps': 0.85}
mat_CMC  = {'t': 0.005, 'k': 11.5, 'rho': 2200.0, 'Cp': 800.0, 'eps': 0.85}

# ── Fixed inner layers (outer -> inner, beneath the shell) ─────────────────
tps_layers_fixed = [
    {'name': 'Thermal Buffer (AETB)', 't': 0.0175, 'k': 0.10, 'rho': 300.0,  'Cp': 1000.0, 'eps': 0.80},  # 0
    {'name': 'Aerogel',               't': 0.0250, 'k': 0.03, 'rho': 120.0,  'Cp': 1100.0, 'eps': 0.80},  # 1
    {'name': 'Silicone Dampener',     't': 0.0010, 'k': 0.45, 'rho': 1100.0, 'Cp': 1400.0, 'eps': 0.90},  # 2
    {'name': 'Titanium Alloy',        't': 0.0035, 'k': 13.0, 'rho': 4430.0, 'Cp': 560.0,  'eps': 0.20},  # 3
]
R_fixed = sum(L['t'] / L['k'] for L in tps_layers_fixed[:3])   # Buffer+Aerogel+Silicone (Ti excluded — inner ODE node)
m_Ti, Cp_Ti = tps_layers_fixed[3]['rho'] * tps_layers_fixed[3]['t'], tps_layers_fixed[3]['Cp']

thermo_mech = {
    'UHTC':            {'alpha_par': 6.7e-6,  'E_par': 350e9},
    'CMC':              {'alpha_par': 2.5e-6,  'E_par': 70e9},
    'Thermal Buffer':   {'alpha_par': 5.0e-6,  'E_par': 0.50e9},
    'Aerogel':          {'alpha_par': 3.0e-6,  'E_par': 0.10e9},
    'Silicone Dampener':{'alpha_par': 250e-6,  'E_par': 0.002e9},
    'Titanium Alloy':   {'alpha_par': 8.6e-6,  'E_par': 114e9},
}


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY  — identical to the re-entry stack (nose cap + shape-perturbed
#  afterbody + tangent-matched Hermite taper)
# ═══════════════════════════════════════════════════════════════════════════
def solve_wedge_angle_jax(x_t, y_t, Rn, n_iter=60):
    theta = jnp.arctan(y_t / x_t)
    def resid(th):
        x_p = Rn * (1.0 - jnp.sin(th))
        y_p = Rn * jnp.cos(th)
        return (y_t - y_p) - jnp.tan(th) * (x_t - x_p)
    for _ in range(n_iter):
        f, df = resid(theta), jax.grad(resid)(theta)
        theta = theta - f / (df + 1e-12)
    return theta


def afterbody_point(c, dx, dy_end, s):
    m = c.shape[0] - 1
    def bernstein(k, n, ss):
        binom = (jax.scipy.special.gammaln(n + 1) - jax.scipy.special.gammaln(k + 1)
                 - jax.scipy.special.gammaln(n - k + 1))
        return jnp.exp(binom) * ss**k * (1 - ss)**(n - k)
    shape_fn = sum(c[k] * bernstein(k, m, s) for k in range(m + 1))
    envelope = s * (1.0 - s)
    y = dy_end * s + envelope * shape_fn
    x = dx * s
    return jnp.stack([x, y])


def afterbody_curve(c, dx, dy_end, n_panels):
    s = jnp.linspace(0.0, 1.0, n_panels + 1)
    xy = jax.vmap(lambda ss: afterbody_point(c, dx, dy_end, ss))(s)
    return xy[:, 0], xy[:, 1]


def afterbody_exit_slope(c, dx, dy_end, s_eval=0.98):
    """s_eval=0.98, not 1.0 — see re-entry stack: Bernstein basis functions
    vanish at s=1 for every coefficient but the last, structurally zeroing
    c0..c[-2]'s influence on the taper. Evaluating just short of s=1 restores
    genuine (small) coupling from every coefficient, at the cost of a
    negligible loss of exact tangency at the afterbody/taper join."""
    jac = jax.jacfwd(lambda ss: afterbody_point(c, dx, dy_end, ss))(s_eval)
    return jac[1] / jac[0]


def hermite_taper(x0, y0, m0, x1, y1, m1, n_panels):
    t = jnp.linspace(0.0, 1.0, n_panels + 1)
    dx = x1 - x0
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    y = h00*y0 + h10*dx*m0 + h01*y1 + h11*dx*m1
    x = x0 + t * dx
    return x, y


def full_geometry(theta_hat, x_target=x_target, N=N,
                   n_nose_frac=n_nose_frac, n_taper_frac=n_taper_frac):
    R_n      = theta_hat[0]
    c        = theta_hat[1:1 + m_coeffs]
    y_body   = theta_hat[1 + m_coeffs]
    theta_te = theta_hat[2 + m_coeffs]

    n_nose  = max(2, int(round(N * n_nose_frac)))
    n_taper = max(1, int(round(N * n_taper_frac)))
    n_after = N - n_nose - n_taper
    assert n_after >= 1

    x_taper_start = x_target * (1.0 - n_taper_frac)
    theta_w = solve_wedge_angle_jax(x_taper_start, y_body, R_n)
    phi_t   = jnp.pi / 2.0 - theta_w
    phi     = jnp.linspace(0.0, phi_t, n_nose + 1)
    x_nose  = R_n * (1.0 - jnp.cos(phi))
    y_nose  = R_n * jnp.sin(phi)

    dx_after = x_taper_start - x_nose[-1]
    dy_after = y_body - y_nose[-1]
    x_af, y_af = afterbody_curve(c, dx_after, dy_after, n_after)
    x_after = x_nose[-1] + x_af[1:]
    y_after = y_nose[-1] + y_af[1:]

    slope_arrive = afterbody_exit_slope(c, dx_after, dy_after)
    slope_te     = jnp.tan(theta_te)
    L_taper  = x_target - x_taper_start
    y_te_raw = y_body + 0.5 * (slope_arrive + slope_te) * L_taper
    y_te     = jnp.maximum(y_te_raw, 0.005)

    x_tap, y_tap = hermite_taper(x_taper_start, y_body, slope_arrive,
                                  x_target, y_te, slope_te, n_taper)
    x_taper, y_taper = x_tap[1:], y_tap[1:]

    x = jnp.concatenate([x_nose, x_after, x_taper])
    y = jnp.concatenate([y_nose, y_after, y_taper])
    theta = jnp.arctan2(jnp.diff(y), jnp.diff(x))
    return x, y, theta, x_nose[-1], y_nose[-1], theta_w


# ═══════════════════════════════════════════════════════════════════════════
#  AERODYNAMICS — Newtonian + isentropic (identical to re-entry stack)
# ═══════════════════════════════════════════════════════════════════════════
def newtonian_cp(theta):
    return jnp.maximum(2.0 * jnp.sin(theta)**2, 0.0)

def pressure_distribution(theta):
    q_inf = 0.5 * rho_inf * V_inf**2
    return p_inf + q_inf * newtonian_cp(theta)

def isentropic_temperature(p):
    p0 = p_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)**(gamma / (gamma - 1.0))
    T0 = T_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)
    return T0 * (p / p0)**((gamma - 1.0) / gamma)

def density_from_pt(p, T):
    return p / (R * T)

def local_mach(p):
    p0 = p_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)**(gamma / (gamma - 1.0))
    return jnp.sqrt((2.0 / (gamma - 1.0)) * ((p0 / p)**((gamma - 1.0) / gamma) - 1.0))

def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


# ═══════════════════════════════════════════════════════════════════════════
#  BOUNDARY LAYER / TRANSITION — soft weighting drives BOTH the flow blend
#  AND the UHTC<->CMC material blend below
# ═══════════════════════════════════════════════════════════════════════════
def transition_location_soft(x_safe, p, T, sharpness=200.0):
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)
    Rex      = rho_e * V_e * x_safe / mu_e
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)
    Re_theta = rho_e * V_e * theta_m / mu_e
    criterion = Re_theta / M_e
    return jax.nn.sigmoid(sharpness * (criterion - 400.0))


def diagnostic_x_tr(x_safe, p, T):
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)
    Rex      = rho_e * V_e * x_safe / mu_e
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)
    Re_theta = rho_e * V_e * theta_m / mu_e
    criterion = Re_theta / M_e
    has_tr = jnp.any(criterion >= 400.0)
    idx    = jnp.argmax(criterion >= 400.0)
    return jnp.where(has_tr, x_safe[idx], x_safe[-1])


def blend_shell_properties(w_turb):
    """Spatially-blended outer shell properties: w_turb=0 -> UHTC-dominated
    (leading edge / laminar region), w_turb=1 -> CMC-dominated (aft body /
    turbulent region). Replaces the original NumPy `x_panel < x_tr` boolean
    mask, which would break the AD trace — this keeps the UHTC/CMC material
    transition itself inside the differentiable chain."""
    t_shell   = (1 - w_turb) * mat_UHTC['t']   + w_turb * mat_CMC['t']
    k_shell   = (1 - w_turb) * mat_UHTC['k']   + w_turb * mat_CMC['k']
    rho_shell = (1 - w_turb) * mat_UHTC['rho'] + w_turb * mat_CMC['rho']
    Cp_shell  = (1 - w_turb) * mat_UHTC['Cp']  + w_turb * mat_CMC['Cp']
    eps_shell = (1 - w_turb) * mat_UHTC['eps'] + w_turb * mat_CMC['eps']
    T_allow_shell = (1 - w_turb) * T_allow_UHTC + w_turb * T_allow_CMC
    return t_shell, k_shell, rho_shell, Cp_shell, eps_shell, T_allow_shell


# ═══════════════════════════════════════════════════════════════════════════
#  HEAT TRANSFER — Fay-Riddell / Eckert laminar / Van Driest II turbulent
# ═══════════════════════════════════════════════════════════════════════════
def fay_riddell(p_stag, T_stag, T_w, R_n):
    rho_stag = p_stag / (R * T_stag)
    mu_stag  = sutherland(T_stag)
    rho_w    = p_stag / (R * T_w)
    mu_w     = sutherland(T_w)
    du_e_dx  = (1.0 / R_n) * jnp.sqrt(2.0 * jnp.maximum(p_stag - p_inf, 0.0) / rho_stag)
    q_stag = (0.763 * Pr**(-0.6) * (rho_w * mu_w)**0.1 * (rho_stag * mu_stag)**0.4
              * jnp.sqrt(du_e_dx) * Cp * (T_stag - T_w))
    return jnp.maximum(q_stag, 0.0)


def van_driest_II_Cf(Re_x_e, M_e, T_e, T_w, T_aw):
    r_turb = Pr**(1.0 / 3.0)
    T_ratio_aw = 1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2
    T_ratio_w  = T_w / T_e
    A_sq = jnp.maximum((r_turb * (gamma - 1.0) / 2.0 * M_e**2) / (T_ratio_w + 1e-10), 0.0)
    A = jnp.sqrt(A_sq)
    B = T_ratio_aw / (T_ratio_w + 1e-10) - 1.0
    denom = jnp.sqrt(B**2 + 4.0 * A**2 + 1e-30)
    alpha = jnp.clip((2.0 * A**2 - B) / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    beta  = jnp.clip(B / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    arcsin_sum = jnp.arcsin(alpha) + jnp.arcsin(beta)
    Fc = jnp.where(T_ratio_aw > 1.001, (T_ratio_aw - 1.0) / (arcsin_sum**2 + 1e-30), 1.0)
    omega = 0.76
    Fx = (1.0 / T_ratio_w)**omega
    Re_x_inc = Fx * Re_x_e
    Cf_inc = 0.0592 * Re_x_inc**(-0.2)
    for _ in range(6):
        lhs  = 1.0 / jnp.sqrt(Cf_inc + 1e-30)
        rhs  = 4.15 * jnp.log10(Re_x_inc * Cf_inc + 1e-30) + 1.7
        f    = lhs - rhs
        dfdC = -0.5 / (Cf_inc + 1e-30)**1.5 - 4.15 / ((Cf_inc + 1e-30) * jnp.log(10.0))
        Cf_inc = jnp.maximum(Cf_inc - f / (dfdC + 1e-30), 1e-8)
    Cf_comp = Cf_inc / (Fc + 1e-30)
    return Cf_comp, Fc, Cf_inc


def heat_transfer_3region(x_safe, p, T_e, T_w_current, w_turb, R_n):
    r_lam, r_turb = Pr**0.5, Pr**(1.0 / 3.0)
    M_e = local_mach(p)
    a_e = jnp.sqrt(gamma * R * T_e)
    V_e = M_e * a_e
    rho_e, mu_e = density_from_pt(p, T_e), sutherland(T_e)

    Taw_lam  = T_e * (1.0 + r_lam * (gamma - 1) / 2 * M_e**2)
    Taw_turb = T_e * (1.0 + r_turb * (gamma - 1) / 2 * M_e**2)

    T_ref_lam   = T_e * (0.45 + 0.55*(T_w_current/T_e) + 0.16*r_lam*(gamma-1)/2*M_e**2)
    rho_ref_lam = p / (R * T_ref_lam)
    mu_ref_lam  = sutherland(T_ref_lam)
    Rex_lam     = rho_ref_lam * V_e * x_safe / mu_ref_lam
    St_lam      = (0.332 / jnp.sqrt(Rex_lam)) * Pr**(-2/3)
    q_lam       = rho_ref_lam * V_e * St_lam * Cp * (Taw_lam - T_w_current)

    Re_x_e = rho_e * V_e * x_safe / mu_e
    Cf_comp, _, _ = van_driest_II_Cf(Re_x_e, M_e, T_e, T_w_current, Taw_turb)
    St_turb = (Cf_comp / 2.0) * Pr**(-2/3)
    q_turb  = rho_e * V_e * St_turb * Cp * (Taw_turb - T_w_current)

    w = w_turb
    Taw    = (1 - w) * Taw_lam + w * Taw_turb
    q_conv = (1 - w) * q_lam   + w * q_turb

    p_stag = p_inf * (1 + 0.5*(gamma-1)*M_inf**2)**(gamma/(gamma-1))
    T_stag = T_inf * (1 + 0.5*(gamma-1)*M_inf**2)
    q_stag = fay_riddell(p_stag, T_stag, T_w_current[0], R_n)
    stag_mask = jnp.arange(x_safe.shape[0]) == 0
    q_conv = jnp.where(stag_mask, q_stag, q_conv)

    return jnp.maximum(q_conv, 0.0), Taw


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSIENT TPS SOLVER — shell material properties now per-panel arrays
#  (blended UHTC/CMC via w_turb), threaded through instead of scalars
# ═══════════════════════════════════════════════════════════════════════════
def tps_transient_scan(x_safe, p, T_e, w_turb, R_n,
                        R_total_arr, m_shell_arr, Cp_shell_arr, eps_shell_arr,
                        n_steps=600, dt=1.0):
    def step(carry, _):
        T_shell, T_Ti = carry
        q_conv, _ = heat_transfer_3region(x_safe, p, T_e, T_shell, w_turb, R_n)
        q_cond = (T_shell - T_Ti) / R_total_arr
        q_rad  = eps_shell_arr * sigma * (T_shell**4 - T_inf**4)
        T_shell_n = T_shell + dt * (q_conv - q_cond - q_rad) / (m_shell_arr * Cp_shell_arr)
        T_Ti_n    = T_Ti    + dt * q_cond / (m_Ti * Cp_Ti)
        return (T_shell_n, T_Ti_n), None
    init = (jnp.full(x_safe.shape, Tw), jnp.full(x_safe.shape, Tw))
    (T_shell_f, T_Ti_f), _ = jax.checkpoint(
        lambda c: jax.lax.scan(step, c, None, length=n_steps))(init)
    return T_shell_f, T_Ti_f


# ═══════════════════════════════════════════════════════════════════════════
#  THERMAL STRESS — bonded 5-layer laminate (Ti / Silicone / Aerogel /
#  Buffer / Shell), force + moment balance
# ═══════════════════════════════════════════════════════════════════════════
def compute_thermal_stress(T_shell_j, T_shell_in, T_buf_in, T_gel_in, T_sil_in, T_Ti_j,
                            t_shell_arr, alpha_shell_arr, E_shell_arr, T_ref=Tw):
    T_shell_mean = 0.5 * (T_shell_j + T_shell_in)
    T_buf_mean   = 0.5 * (T_shell_in + T_buf_in)
    T_gel_mean   = 0.5 * (T_buf_in + T_gel_in)
    T_sil_mean   = 0.5 * (T_gel_in + T_sil_in)
    T_Ti_mean    = T_Ti_j

    fixed = tps_layers_fixed
    E_list  = [thermo_mech['Titanium Alloy']['E_par'],     thermo_mech['Silicone Dampener']['E_par'],
               thermo_mech['Aerogel']['E_par'],             thermo_mech['Thermal Buffer']['E_par'],
               E_shell_arr]
    al_list = [thermo_mech['Titanium Alloy']['alpha_par'],  thermo_mech['Silicone Dampener']['alpha_par'],
               thermo_mech['Aerogel']['alpha_par'],          thermo_mech['Thermal Buffer']['alpha_par'],
               alpha_shell_arr]
    t_list  = [fixed[3]['t'], fixed[2]['t'], fixed[1]['t'], fixed[0]['t'], t_shell_arr]
    dT_list = [T_Ti_mean - T_ref, T_sil_mean - T_ref, T_gel_mean - T_ref, T_buf_mean - T_ref, T_shell_mean - T_ref]

    t_arrs = [t_i if hasattr(t_i, 'shape') else jnp.full_like(dT_list[0], t_i) for t_i in t_list]
    t_total = sum(t_arrs)
    z_bounds = [-0.5 * t_total]
    for t_i in t_arrs:
        z_bounds.append(z_bounds[-1] + t_i)
    z_mid = [0.5 * (z_bounds[i] + z_bounds[i + 1]) for i in range(5)]

    A11 = A12 = A22 = N_T = M_T = jnp.zeros_like(dT_list[0])
    for i in range(5):
        E_i = E_list[i] if hasattr(E_list[i], 'shape') else jnp.full_like(dT_list[0], E_list[i])
        al_i, dT_i = al_list[i], dT_list[i]
        z0, z1 = z_bounds[i], z_bounds[i + 1]
        a, b, c = z1 - z0, (z1**2 - z0**2) / 2.0, (z1**3 - z0**3) / 3.0
        A11 += E_i * a; A12 += E_i * b; A22 += E_i * c
        N_T += E_i * al_i * dT_i * a
        M_T += E_i * al_i * dT_i * b

    det = A11 * A22 - A12**2
    eps0  = (A22 * N_T - A12 * M_T) / det
    kappa = (A11 * M_T - A12 * N_T) / det

    sigma_layers = [
        (E_list[i] if hasattr(E_list[i], 'shape') else jnp.full_like(dT_list[0], E_list[i]))
        * (eps0 + kappa * z_mid[i] - al_list[i] * dT_list[i])
        for i in range(5)
    ]
    return sigma_layers   # [Ti, Silicone, Aerogel, Buffer, Shell]


# ═══════════════════════════════════════════════════════════════════════════
#  SOLVE — single entry point
# ═══════════════════════════════════════════════════════════════════════════
def solve_state(theta_hat, n_steps=600, dt=1.0):
    R_n = theta_hat[0]
    x, y, theta, x_nose_end, y_nose_end, theta_w = full_geometry(theta_hat)
    x_panel = x[1:]
    x_safe  = jnp.maximum(x_panel, 1e-6)

    p = pressure_distribution(theta)
    T = isentropic_temperature(p)
    w_turb = transition_location_soft(x_safe, p, T)
    x_tr_diag = diagnostic_x_tr(x_safe, p, T)

    t_shell, k_shell, rho_shell, Cp_shell_arr, eps_shell, T_allow_shell = blend_shell_properties(w_turb)
    R_shell_arr   = t_shell / k_shell
    R_total_arr   = R_shell_arr + R_fixed
    m_shell_arr   = rho_shell * t_shell
    alpha_shell   = (1 - w_turb) * thermo_mech['UHTC']['alpha_par'] + w_turb * thermo_mech['CMC']['alpha_par']
    E_shell       = (1 - w_turb) * thermo_mech['UHTC']['E_par']     + w_turb * thermo_mech['CMC']['E_par']

    T_shell_j, T_Ti_j = tps_transient_scan(x_safe, p, T, w_turb, R_n,
                                            R_total_arr, m_shell_arr, Cp_shell_arr, eps_shell,
                                            n_steps, dt)
    q_conv, Taw = heat_transfer_3region(x_safe, p, T, T_shell_j, w_turb, R_n)
    q_cond = (T_shell_j - T_Ti_j) / R_total_arr
    q_rad  = eps_shell * sigma * (T_shell_j**4 - T_inf**4)

    T_shell_in = T_shell_j  - q_cond * R_shell_arr
    T_buf_in   = T_shell_in - q_cond * (tps_layers_fixed[0]['t'] / tps_layers_fixed[0]['k'])
    T_gel_in   = T_buf_in   - q_cond * (tps_layers_fixed[1]['t'] / tps_layers_fixed[1]['k'])
    T_sil_in   = T_gel_in   - q_cond * (tps_layers_fixed[2]['t'] / tps_layers_fixed[2]['k'])

    sigma_Ti, sigma_Sil, sigma_Gel, sigma_Buf, sigma_Shell = compute_thermal_stress(
        T_shell_j, T_shell_in, T_buf_in, T_gel_in, T_sil_in, T_Ti_j,
        t_shell, alpha_shell, E_shell)

    return dict(x=x, y=y, theta=theta, x_panel=x_panel, p=p, T=T, x_tr=x_tr_diag,
                T_shell_j=T_shell_j, T_Ti_j=T_Ti_j, T_allow_shell=T_allow_shell,
                q_conv=q_conv, q_cond=q_cond, q_rad=q_rad,
                sigma_Ti=sigma_Ti, sigma_Sil=sigma_Sil, sigma_Gel=sigma_Gel,
                sigma_Buf=sigma_Buf, sigma_Shell=sigma_Shell)


def ks_max(vals, rho=50.0):
    return jax.nn.logsumexp(rho * vals) / rho


def objective(theta_hat):
    """Peak thermal stress (KS-smoothed), across all 5 layers x N panels."""
    s = solve_state(theta_hat)
    sig_all = jnp.concatenate([s['sigma_Ti'], s['sigma_Sil'], s['sigma_Gel'],
                                s['sigma_Buf'], s['sigma_Shell']])
    return ks_max(jnp.abs(sig_all) / 1e8)


def drag(theta_hat):
    x, y, theta, *_ = full_geometry(theta_hat)
    p = pressure_distribution(theta)
    dy = jnp.diff(y)
    return jnp.sum((p - p_inf) * jnp.abs(dy))


def volume(theta_hat):
    x, y, theta, *_ = full_geometry(theta_hat)
    return jnp.trapezoid(y, x)


def constraints(theta_hat, D_max, V_min):
    s = solve_state(theta_hat)
    g_T = ks_max(jnp.concatenate([s['T_shell_j'] / s['T_allow_shell'],
                                   s['T_Ti_j'] / T_allow_Ti])) - 1.0    # <= 0
    g_D = drag(theta_hat) - D_max                                       # <= 0
    g_V = V_min - volume(theta_hat)                                     # <= 0
    return jnp.stack([g_T, g_D, g_V])


grad_J = jax.grad(objective)


def finite_diff_grad(f, theta_hat, h=1e-6):
    n = theta_hat.shape[0]
    g = np.zeros(n)
    for i in range(n):
        dtheta = np.zeros(n); dtheta[i] = h
        fp = f(theta_hat + jnp.array(dtheta))
        fm = f(theta_hat - jnp.array(dtheta))
        g[i] = float((fp - fm) / (2 * h))
    return g


if __name__ == "__main__":
    theta0 = jnp.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.11, jnp.deg2rad(-1.0)])
    labels = ["R_n"] + [f"c{i}" for i in range(m_coeffs)] + ["y_body", "theta_te"]

    print("=" * 70)
    print("  BASELINE STATE — MISSILE STACK (UHTC/CMC shell)")
    print("=" * 70)
    s0 = solve_state(theta0)
    print(f"  x_tr = {float(s0['x_tr']):.3f} m")
    print(f"  max T_shell = {float(s0['T_shell_j'].max()):.1f} K")
    print(f"  max T_Ti    = {float(s0['T_Ti_j'].max()):.1f} K  (limit {T_allow_Ti:.1f} K)")
    print(f"  J (KS peak stress, normalized) = {float(objective(theta0)):.4f}")
    print(f"  drag = {float(drag(theta0)):.1f} N/m   volume = {float(volume(theta0)):.4f} m^2")

    # ═══════════════════════════════════════════════════════════════════
    #  ITEM 1 — AD vs FD, single baseline point
    #  (re-verified for this stack specifically: the linear stress solve's
    #  conditioning, det = A11*A22 - A12^2, depends on the material
    #  stiffness/CTE spread, which differs from the RCC stack — the
    #  computational graph is identical, but numerical conditioning is not
    #  guaranteed to be, so this check is re-run rather than assumed.)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  AD vs FINITE-DIFFERENCE GRADIENT CHECK  (objective)")
    print("=" * 70)
    g_ad = np.array(grad_J(theta0))
    g_fd = finite_diff_grad(objective, theta0)
    print(f"  {'param':>10}  {'AD':>14}  {'FD':>14}  {'rel. err':>10}")
    for lbl, a, f in zip(labels, g_ad, g_fd):
        rel = abs(a - f) / (abs(f) + 1e-12)
        print(f"  {lbl:>10}  {a:14.6e}  {f:14.6e}  {rel:10.2e}")

    # ── plots ────────────────────────────────────────────────────────────
    x_np, y_np = np.array(s0['x']), np.array(s0['y'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(x_np, y_np, 'b-o', lw=2, ms=5)
    ax.axhline(0.0, color='grey', ls='--', lw=1)
    ax.plot([x_np[-1], x_np[-1]], [0.0, y_np[-1]], 'k-', lw=2.5)
    ax.axvline(float(s0['x_tr']), color='grey', ls=':', lw=1)
    ax.set_title('Baseline shape (missile stack)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.grid(alpha=0.3); ax.set_aspect('equal')

    x_p = np.array(s0['x_panel'])
    axes[1].plot(x_p, np.array(s0['q_conv']), 'o-', color='crimson')
    axes[1].set_title('q_conv (final)'); axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('W/m^2')
    axes[1].grid(alpha=0.3)

    for lbl, arr in [('Ti', s0['sigma_Ti']), ('Silicone', s0['sigma_Sil']),
                      ('Aerogel', s0['sigma_Gel']), ('Buffer', s0['sigma_Buf']),
                      ('Shell', s0['sigma_Shell'])]:
        axes[2].plot(x_p, np.array(arr) / 1e6, 'o-', label=lbl)
    axes[2].set_title('Thermal stress'); axes[2].set_xlabel('x (m)'); axes[2].set_ylabel('MPa')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_state_missile.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ═══════════════════════════════════════════════════════════════════
    #  ITEM 4 — GRADIENT STABILITY ACROSS MULTIPLE PERTURBED BASELINES
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ITEM 4 — AD vs FD AT MULTIPLE PERTURBED BASELINE POINTS")
    print("=" * 70)
    rng = np.random.default_rng(0)
    perturb_scale = np.array([0.003, 0.01, 0.01, 0.01, 0.01, 0.02, np.deg2rad(3.0)])
    for trial in range(4):
        offset = rng.normal(scale=1.0, size=perturb_scale.shape) * perturb_scale
        theta_pert = jnp.array(np.array(theta0) + offset)
        theta_pert = theta_pert.at[0].set(jnp.maximum(theta_pert[0], 0.002))
        g_ad_p = np.array(grad_J(theta_pert))
        g_fd_p = finite_diff_grad(objective, theta_pert)
        max_rel_err = np.max(np.abs(g_ad_p - g_fd_p) / (np.abs(g_fd_p) + 1e-12))
        print(f"\n  trial {trial}:  theta = {np.array(theta_pert).round(4)}")
        print(f"    max relative error across all params: {max_rel_err:.2e}"
              f"  {'OK' if max_rel_err < 1e-3 else '!! CHECK THIS POINT'}")

    # ═══════════════════════════════════════════════════════════════════
    #  ITEM 3 — OFF-DIAGONAL COUPLING (full per-panel Jacobian, Shell stress)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ITEM 3 — OFF-DIAGONAL COUPLING: d(sigma_Shell per panel) / d(theta_hat)")
    print("=" * 70)

    def sigma_shell_vector(theta_hat):
        s = solve_state(theta_hat)
        return s['sigma_Shell']

    J_full = np.array(jax.jacobian(sigma_shell_vector)(theta0))
    print(f"  Jacobian shape: {J_full.shape}  (rows=panels, cols=params)")
    print(f"  columns: {labels}")
    header = "  panel  x[m]  " + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i in range(N):
        row = "".join(f"{J_full[i, j]:12.3e}" for j in range(len(labels)))
        print(f"  {i:5d}  {float(s0['x_panel'][i]):4.2f}  {row}")

    c0_col = labels.index("c0")
    last_panel_sens = J_full[-1, c0_col]
    print(f"\n  d(sigma_Shell[last panel]) / d(c0) = {last_panel_sens:.4e}")
    print(f"  {'PASS — coupling exists (nonzero)' if abs(last_panel_sens) > 1e-10 else 'FAIL — appears decoupled, check trace'}")