import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (fixed hyperparameters — not design variables)
# ═══════════════════════════════════════════════════════════════════════════
N              = 16          # total panels
x_target       = 2.5         # m  — chord length (fixed)
n_nose_frac    = 0.30
n_taper_frac   = 0.25
m_coeffs       = 4           # afterbody shape-perturbation coefficients

M_inf  = 5.0
gamma  = 1.4
T_inf, p_inf, rho_inf = 205.0, 5543.0, 0.09427     # 20 km atmosphere
Tw     = 300.0
Cp     = 1005.0
R, Pr  = 287.0, 0.71
sigma  = 5.67e-8

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf

T_allow_RCC  = 1900.0
T_allow_alum = 450.0

# ── Fixed TPS stack (outer -> inner): RCC / Silica / SIP / Aluminium ────────
tps_layers = [
    {'name': 'Aluminium', 't': 0.003, 'k': 167.0, 'rho': 2700.0, 'Cp': 900.0,  'eps': 0.125},  # 0
    {'name': 'SIP',       't': 0.005, 'k': 0.05,  'rho': 128.0,  'Cp': 1000.0, 'eps': 0.85},   # 1
    {'name': 'Silica',    't': 0.050, 'k': 0.06,  'rho': 1900.0, 'Cp': 800.0,  'eps': 0.85},   # 2
    {'name': 'RCC',       't': 0.005, 'k': 20.0,  'rho': 1600.0, 'Cp': 711.0,  'eps': 0.875},  # 3
]
R_layers = [L['t'] / L['k'] for L in tps_layers]
R_total  = R_layers[1] + R_layers[2] + R_layers[3]     # SIP+Silica+RCC (Aluminium excluded — it's the inner node)
m_RCC, Cp_RCC   = tps_layers[3]['rho'] * tps_layers[3]['t'], tps_layers[3]['Cp']
m_alum, Cp_alum = tps_layers[0]['rho'] * tps_layers[0]['t'], tps_layers[0]['Cp']
eps_RCC = tps_layers[3]['eps']

thermo_mech = {
    'Aluminium': {'alpha_par': 23.1e-6, 'E_par': 69.0e9},
    'SIP':       {'alpha_par': 250e-6,  'E_par': 0.001e9},
    'Silica':    {'alpha_par': 0.5e-6,  'E_par': 0.05e9},
    'RCC':       {'alpha_par': 1.0e-6,  'E_par': 60.0e9},
}

# design-variable bounds and drag/volume targets used by the constraints
D_max = 5000.0     # N per unit span — placeholder, set to your mission requirement
V_min = 0.10        # m^2 per unit span (2D cross-sectional area proxy) — placeholder


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY  — nose cap (circular) + afterbody (shape-perturbed) + taper
#  theta_hat = [R_n, c_0..c_{m-1}, y_body, theta_te]
# ═══════════════════════════════════════════════════════════════════════════
def solve_wedge_angle_jax(x_t, y_t, Rn, n_iter=60):
    """Differentiable Newton solve for the nose-cap tangent angle theta_w such
    that a straight line from the tangent point reaches (x_t, y_t)."""
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
    """Single-point evaluation (scalar s) of the afterbody curve, in local
    frame starting at (0,0). Used both to build the panel array (vmap over
    s) and, via jax.jacfwd, to get the EXACT exit slope at s=1 analytically
    — this is what lets the taper start tangent-matched instead of kinked."""
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


def afterbody_exit_slope(c, dx, dy_end,s_eval=0.98):
    """dy/dx of the afterbody curve at its endpoint s=1, via autodiff of the
    scalar-s parametrization — exact, no finite-difference stencil needed."""
    jac = jax.jacfwd(lambda ss: afterbody_point(c, dx, dy_end, ss))(s_eval)
    return jac[1] / jac[0]


def hermite_taper(x0, y0, m0, x1, y1, m1, n_panels):
    """Cubic Hermite segment: passes through (x0,y0) with slope m0 and
    (x1,y1) with slope m1. Guarantees exact slope continuity at BOTH ends —
    m0 tangent-matches the afterbody exit, m1 = tan(theta_te) sets the
    prescribed trailing-edge angle. No kink at the afterbody/taper join."""
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
    """Returns x, y (N+1,), theta (N,), plus nose-cap diagnostics."""
    R_n      = theta_hat[0]
    c        = theta_hat[1:1 + m_coeffs]
    y_body   = theta_hat[1 + m_coeffs]
    theta_te = theta_hat[2 + m_coeffs]

    n_nose  = max(2, int(round(N * n_nose_frac)))
    n_taper = max(1, int(round(N * n_taper_frac)))
    n_after = N - n_nose - n_taper
    assert n_after >= 1, "N too small for chosen nose/taper fractions"

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

    # tangent-match: taper's starting slope = afterbody's actual exit slope
    slope_arrive = afterbody_exit_slope(c, dx_after, dy_after)
    slope_te     = jnp.tan(theta_te)
    # y_te estimated via trapezoidal-average slope over the taper length
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
#  AERODYNAMICS — Newtonian + isentropic
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
#  BOUNDARY LAYER / TRANSITION
# ═══════════════════════════════════════════════════════════════════════════
def check_taper_geometry_grad(theta0):
    """Verify that panel 15 geometry (theta angle) carries c0-c2 gradients."""
    def last_panel_angle(th):
        _, _, theta_panels, *_ = full_geometry(th)
        return theta_panels[-1]   # scalar: angle of panel 15

    g = jax.grad(last_panel_angle)(theta0)
    labels = ["R_n","c0","c1","c2","c3","y_body","theta_te"]
    print("  d(theta[15])/d(param):")
    for lbl, gi in zip(labels, g):
        print(f"    {lbl:>10s}: {float(gi):12.4e}")



def transition_location_soft(x_safe, p, T, sharpness=200.0):
    """
    Differentiable soft transition location.
    Returns a per-panel weight w[i] in [0,1] where:
      w[i] ≈ 0  →  laminar
      w[i] ≈ 1  →  turbulent
    No argmax, no integer indexing.
    """
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)

    Rex      = rho_e * V_e * x_safe / mu_e
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)
    Re_theta = rho_e * V_e * theta_m / mu_e
    criterion = Re_theta / M_e   # transition when this >= 400

    # Soft step: sigmoid centred on criterion = 400
    # sharpness controls how sharp the lam→turb switch is
    w_turb = jax.nn.sigmoid(sharpness * (criterion - 400.0))
    return w_turb   # shape (N,), replaces the scalar x_tr



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
    """Compressible turbulent Cf. BUG FIX from the pasted version: this used
    to silently recompute M_e = local_mach(p) from a MODULE-LEVEL GLOBAL `p`
    instead of using the M_e argument — under jax.grad that global is frozen
    at whatever value existed at trace time, giving wrong (stale, non-
    differentiable-w.r.t.-shape) turbulent heating. Removed; M_e argument
    is now used directly throughout."""
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
    """
    w_turb: (N,) array in [0,1], 0=laminar, 1=turbulent.
    Replaces the hard jnp.where(x_safe < x_tr, ...) branch.
    """
    r_lam, r_turb = Pr**0.5, Pr**(1.0 / 3.0)
    M_e = local_mach(p)
    a_e = jnp.sqrt(gamma * R * T_e)
    V_e = M_e * a_e
    rho_e, mu_e = density_from_pt(p, T_e), sutherland(T_e)

    Taw_lam  = T_e * (1.0 + r_lam  * (gamma-1)/2 * M_e**2)
    Taw_turb = T_e * (1.0 + r_turb * (gamma-1)/2 * M_e**2)

    # --- laminar branch ---
    T_ref_lam   = T_e * (0.45 + 0.55*(T_w_current/T_e)
                         + 0.16*r_lam*(gamma-1)/2*M_e**2)
    rho_ref_lam = p / (R * T_ref_lam)
    mu_ref_lam  = sutherland(T_ref_lam)
    Rex_lam     = rho_ref_lam * V_e * x_safe / mu_ref_lam
    St_lam      = (0.332 / jnp.sqrt(Rex_lam)) * Pr**(-2/3)
    q_lam       = rho_ref_lam * V_e * St_lam * Cp * (Taw_lam - T_w_current)

    # --- turbulent branch ---
    Re_x_e = rho_e * V_e * x_safe / mu_e
    Cf_comp, _, _ = van_driest_II_Cf(Re_x_e, M_e, T_e, T_w_current, Taw_turb)
    St_turb = (Cf_comp / 2.0) * Pr**(-2/3)
    q_turb  = rho_e * V_e * St_turb * Cp * (Taw_turb - T_w_current)

    # ✅ Differentiable blend — NO hard branch, NO argmax
    w = w_turb                          # (N,) in [0,1]
    Taw   = (1-w)*Taw_lam + w*Taw_turb
    q_conv = (1-w)*q_lam  + w*q_turb

    # stagnation point override (panel 0 only, smooth mask)
    p_stag = p_inf * (1 + 0.5*(gamma-1)*M_inf**2)**(gamma/(gamma-1))
    T_stag = T_inf * (1 + 0.5*(gamma-1)*M_inf**2)
    q_stag = fay_riddell(p_stag, T_stag, T_w_current[0], R_n)
    stag_mask = jnp.arange(x_safe.shape[0]) == 0
    q_conv = jnp.where(stag_mask, q_stag, q_conv)

    return jnp.maximum(q_conv, 0.0), Taw


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSIENT TPS SOLVER  (fixed-step lax.scan — fully differentiable)
# ═══════════════════════════════════════════════════════════════════════════
def tps_transient_scan(x_safe, p, T_e, x_tr, R_n, n_steps=3000, dt=0.2):
    def step(carry, _):
        T_RCC, T_alum = carry
        q_conv, _ = heat_transfer_3region(x_safe, p, T_e, T_RCC, x_tr, R_n)
        q_cond = (T_RCC - T_alum) / R_total
        q_rad  = eps_RCC * sigma * (T_RCC**4 - T_inf**4)
        T_RCC_n  = T_RCC  + dt * (q_conv - q_cond - q_rad) / (m_RCC * Cp_RCC)
        T_alum_n = T_alum + dt * q_cond / (m_alum * Cp_alum)
        return (T_RCC_n, T_alum_n), None
    init = (jnp.full(x_safe.shape, Tw), jnp.full(x_safe.shape, Tw))
    (T_RCC_f, T_alum_f), _ = jax.checkpoint(
        lambda c: jax.lax.scan(step, c, None, length=n_steps))(init)
    return T_RCC_f, T_alum_f


# ═══════════════════════════════════════════════════════════════════════════
#  THERMAL STRESS — bonded 4-layer laminate (force + moment balance)
# ═══════════════════════════════════════════════════════════════════════════
def compute_thermal_stress(T_RCC_j, T_RCC_in, T_sil_in, T_sip_in, T_alum_j, T_ref=Tw):
    T_RCC_mean  = 0.5 * (T_RCC_j + T_RCC_in)
    T_sil_mean  = 0.5 * (T_RCC_in + T_sil_in)
    T_sip_mean  = 0.5 * (T_sil_in + T_sip_in)
    T_alum_mean = T_alum_j

    props = [thermo_mech['Aluminium'], thermo_mech['SIP'], thermo_mech['Silica'], thermo_mech['RCC']]
    t_list = [tps_layers[0]['t'], tps_layers[1]['t'], tps_layers[2]['t'], tps_layers[3]['t']]
    dT_list = [T_alum_mean - T_ref, T_sip_mean - T_ref, T_sil_mean - T_ref, T_RCC_mean - T_ref]

    t_arrs = [jnp.full_like(dT_list[0], t) for t in t_list]
    t_total = sum(t_arrs)
    z_bounds = [-0.5 * t_total]
    for t_i in t_arrs:
        z_bounds.append(z_bounds[-1] + t_i)
    z_mid = [0.5 * (z_bounds[i] + z_bounds[i + 1]) for i in range(4)]

    A11 = A12 = A22 = N_T = M_T = jnp.zeros_like(dT_list[0])
    for i in range(4):
        E_i, al_i, dT_i = props[i]['E_par'], props[i]['alpha_par'], dT_list[i]
        z0, z1 = z_bounds[i], z_bounds[i + 1]
        a, b, c = z1 - z0, (z1**2 - z0**2) / 2.0, (z1**3 - z0**3) / 3.0
        A11 += E_i * a; A12 += E_i * b; A22 += E_i * c
        N_T += E_i * al_i * dT_i * a
        M_T += E_i * al_i * dT_i * b

    det = A11 * A22 - A12**2
    eps0  = (A22 * N_T - A12 * M_T) / det
    kappa = (A11 * M_T - A12 * N_T) / det

    sigma_layers = [props[i]['E_par'] * (eps0 + kappa * z_mid[i] - props[i]['alpha_par'] * dT_list[i])
                    for i in range(4)]
    return sigma_layers   # [Al, SIP, Silica, RCC]


# ═══════════════════════════════════════════════════════════════════════════
#  SOLVE — single entry point used by objective, constraints, and diagnostics
# ═══════════════════════════════════════════════════════════════════════════
def solve_state(theta_hat, n_steps=3000, dt=0.2):
    R_n = theta_hat[0]
    x, y, theta, x_nose_end, y_nose_end, theta_w = full_geometry(theta_hat)
    x_panel = x[1:]
    x_safe  = jnp.maximum(x_panel, 1e-6)

    p = pressure_distribution(theta)
    T = isentropic_temperature(p)

    # ✅ Differentiable transition weights — replaces hard x_tr scalar
    w_turb = transition_location_soft(x_safe, p, T)

    # ── Diagnostic x_tr (non-differentiable, for printing only) ────────
    M_e_diag   = local_mach(p)
    a_e_diag   = jnp.sqrt(gamma * R * T)
    V_e_diag   = M_e_diag * a_e_diag
    rho_e_diag = density_from_pt(p, T)
    mu_e_diag  = sutherland(T)
    Rex_diag      = rho_e_diag * V_e_diag * x_safe / mu_e_diag
    theta_m_diag  = 0.664 * x_safe / jnp.sqrt(Rex_diag)
    Re_theta_diag = rho_e_diag * V_e_diag * theta_m_diag / mu_e_diag
    criterion     = Re_theta_diag / M_e_diag          # transition when >= 400
    has_tr        = jnp.any(criterion >= 400.0)
    idx_tr        = jnp.argmax(criterion >= 400.0)
    x_tr_diag     = jnp.where(has_tr, x_safe[idx_tr], x_safe[-1])  # for display only

    T_RCC_j, T_alum_j = tps_transient_scan(
        x_safe, p, T, w_turb, R_n, n_steps, dt)

    q_conv, Taw = heat_transfer_3region(
        x_safe, p, T, T_RCC_j, w_turb, R_n)

    q_cond = (T_RCC_j - T_alum_j) / R_total
    q_rad  = eps_RCC * sigma * (T_RCC_j**4 - T_inf**4)

    T_RCC_in  = T_RCC_j  - q_cond * R_layers[3]
    T_sil_in  = T_RCC_in - q_cond * R_layers[2]
    T_sip_in  = T_sil_in - q_cond * R_layers[1]

    sigma_Al, sigma_SIP, sigma_Sil, sigma_RCC = compute_thermal_stress(
        T_RCC_j, T_RCC_in, T_sil_in, T_sip_in, T_alum_j)

    return dict(
        x=x, y=y, theta=theta, x_panel=x_panel,
        p=p, T=T,
        x_tr=x_tr_diag,                   # scalar, for printing only
        T_RCC_j=T_RCC_j, T_alum_j=T_alum_j,
        q_conv=q_conv, q_cond=q_cond, q_rad=q_rad,
        sigma_Al=sigma_Al, sigma_SIP=sigma_SIP,
        sigma_Sil=sigma_Sil, sigma_RCC=sigma_RCC
    )

def ks_max(vals, rho=50.0):
    return jax.nn.logsumexp(rho * vals) / rho


def objective(theta_hat):
    """Peak thermal stress (KS-smoothed), across all 4 layers x N panels."""
    s = solve_state(theta_hat)
    sig_all = jnp.concatenate([s['sigma_Al'], s['sigma_SIP'], s['sigma_Sil'], s['sigma_RCC']])
    return ks_max(jnp.abs(sig_all) / 1e8)


def drag(theta_hat):
    x, y, theta, *_ = full_geometry(theta_hat)
    p = pressure_distribution(theta)
    dy = jnp.diff(y)                       # projected frontal height per panel
    return jnp.sum((p - p_inf) * jnp.abs(dy))


def volume(theta_hat):
    x, y, theta, *_ = full_geometry(theta_hat)
    return jnp.trapezoid(y, x)


def constraints(theta_hat):
    s = solve_state(theta_hat)
    g_T = ks_max(jnp.concatenate([s['T_RCC_j'] / T_allow_RCC,
                                   s['T_alum_j'] / T_allow_alum])) - 1.0     # <= 0
    g_D = drag(theta_hat) - D_max                                            # <= 0
    g_V = V_min - volume(theta_hat)                                          # <= 0
    return jnp.stack([g_T, g_D, g_V])


grad_J = jax.grad(objective)
grad_g = jax.jacobian(constraints)


# ═══════════════════════════════════════════════════════════════════════════
#  AD vs FINITE-DIFFERENCE VALIDATION  (run this before trusting the optimizer)
# ═══════════════════════════════════════════════════════════════════════════
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
    # baseline design vector: [R_n, c0..c3, y_body, theta_te]
    theta0 = jnp.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.11, jnp.deg2rad(-1.0)])

    print("=" * 70)
    print("  BASELINE STATE")
    print("=" * 70)
    s0 = solve_state(theta0)
    print(f"  x_tr = {float(s0['x_tr']):.3f} m")
    print(f"  max T_RCC  = {float(s0['T_RCC_j'].max()):.1f} K  (limit {T_allow_RCC:.0f} K)")
    print(f"  max T_alum = {float(s0['T_alum_j'].max()):.1f} K  (limit {T_allow_alum:.0f} K)")
    print(f"  J (KS peak stress, normalized) = {float(objective(theta0)):.4f}")
    print(f"  drag = {float(drag(theta0)):.1f} N/m   volume = {float(volume(theta0)):.4f} m^2")

    print("\n" + "=" * 70)
    print("  AD vs FINITE-DIFFERENCE GRADIENT CHECK  (objective)")
    print("=" * 70)
    g_ad = np.array(grad_J(theta0))
    g_fd = finite_diff_grad(objective, theta0)
    labels = ["R_n"] + [f"c{i}" for i in range(m_coeffs)] + ["y_body", "theta_te"]
    print(f"  {'param':>10}  {'AD':>14}  {'FD':>14}  {'rel. err':>10}")
    for lbl, a, f in zip(labels, g_ad, g_fd):
        rel = abs(a - f) / (abs(f) + 1e-12)
        print(f"  {lbl:>10}  {a:14.6e}  {f:14.6e}  {rel:10.2e}")

    # ── plots ────────────────────────────────────────────────────────────
    x_np, y_np = np.array(s0['x']), np.array(s0['y'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax=axes[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]

# Surface curve with panel markers
    ax.plot(x_np, y_np, 'b-o', lw=2, ms=5)

# Centreline
    ax.axhline(0.0, color='grey', ls='--', lw=1)

# Vertical base line at trailing edge
    ax.plot([x_np[-1], x_np[-1]], [0.0, y_np[-1]],
        'k-', lw=2.5)



# Cleaner: use fixed values
    ax.plot(x_target, float(theta0[1 + m_coeffs]), 'r*', ms=14)

    ax.axvline(float(s0['x_tr']), color='grey', ls=':', lw=1)
    ax.set_title('Baseline shape')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    ax.set_ylim(-0.05, 0.35)
    
    x_p = np.array(s0['x_panel'])
    axes[1].plot(x_p, np.array(s0['q_conv']), 'o-', color='crimson')
    axes[1].set_title('q_conv (final)'); axes[1].set_xlabel('x (m)'); axes[1].set_ylabel('W/m^2')
    axes[1].grid(alpha=0.3)

    for lbl, arr in [('Al', s0['sigma_Al']), ('SIP', s0['sigma_SIP']),
                      ('Silica', s0['sigma_Sil']), ('RCC', s0['sigma_RCC'])]:
        axes[2].plot(x_p, np.array(arr) / 1e6, 'o-', label=lbl)
    axes[2].set_title('Thermal stress'); axes[2].set_xlabel('x (m)'); axes[2].set_ylabel('MPa')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    

    plt.tight_layout()
    plt.savefig('baseline_state.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ═══════════════════════════════════════════════════════════════════
    #  ITEM 4 — GRADIENT STABILITY ACROSS MULTIPLE BASELINE POINTS
    #
    #  "Perturbing" here means: generate several DIFFERENT starting theta_hat
    #  vectors by nudging theta0 with small random offsets (respecting
    #  physical bounds), then rerun the SAME AD-vs-FD comparison at each.
    #  Agreement at one point could be a coincidence; agreement across
    #  several scattered points is much stronger evidence the gradient
    #  machinery itself is correct, not just correct at theta0.
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ITEM 4 — AD vs FD AT MULTIPLE PERTURBED BASELINE POINTS")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n_test_points = 4
    # per-parameter perturbation scale: [R_n, c0..c3, y_body, theta_te]
    # kept small relative to each parameter's natural scale, and bounded so
    # R_n stays positive and theta_te stays a plausible closing taper angle
    perturb_scale = np.array([0.003, 0.01, 0.01, 0.01, 0.01, 0.02, np.deg2rad(3.0)])

    for trial in range(n_test_points):
        offset = rng.normal(scale=1.0, size=perturb_scale.shape) * perturb_scale
        theta_pert = jnp.array(np.array(theta0) + offset)
        theta_pert = theta_pert.at[0].set(jnp.maximum(theta_pert[0], 0.002))  # R_n > 0

        g_ad_p = np.array(grad_J(theta_pert))
        g_fd_p = finite_diff_grad(objective, theta_pert)
        max_rel_err = np.max(np.abs(g_ad_p - g_fd_p) / (np.abs(g_fd_p) + 1e-12))
        print(f"\n  trial {trial}:  theta = {np.array(theta_pert).round(4)}")
        print(f"    max relative error across all params: {max_rel_err:.2e}"
              f"  {'OK' if max_rel_err < 1e-3 else '!! CHECK THIS POINT'}")

    # ═══════════════════════════════════════════════════════════════════
    #  ITEM 3 — OFF-DIAGONAL COUPLING (full per-panel Jacobian, un-aggregated)
    #
    #  Instead of grad of the KS-aggregated scalar J, take the full vector
    #  of per-panel RCC stress and compute its full Jacobian w.r.t. theta_hat.
    #  Row = panel index, column = design parameter. Nonzero entries far
    #  from the "local" region a parameter geometrically touches are the
    #  signature of real cross-panel flow-on coupling — exactly what the
    #  Jahn paper's local method structurally cannot capture, and what your
    #  AD chain should show since it propagates state panel-to-panel.
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ITEM 3 — OFF-DIAGONAL COUPLING: d(sigma_RCC per panel) / d(theta_hat)")
    print("=" * 70)
 

    def sigma_RCC_vector(theta_hat):
        s = solve_state(theta_hat)
        return s['sigma_RCC']          # (N,) — one value per panel, NOT aggregated

    J_full = np.array(jax.jacobian(sigma_RCC_vector)(theta0))   # shape (N, n_params)

    print(f"  Jacobian shape: {J_full.shape}  (rows=panels, cols=params)")
    print(f"  columns: {labels}")
    print()
    header = "  panel  x[m]  " + "".join(f"{l:>12}" for l in labels)
    print(header)
    for i in range(N):
        row = "".join(f"{J_full[i, j]:12.3e}" for j in range(len(labels)))
        print(f"  {i:5d}  {float(s0['x_panel'][i]):4.2f}  {row}")

    check_taper_geometry_grad(theta0)

    # explicit check: c0 (a nose-region shape coefficient) should still show
    # nonzero sensitivity at the LAST panel (trailing edge) if coupling is real
    c0_col = labels.index("c0")
    last_panel_sens = J_full[-1, c0_col]
    print(f"\n  d(sigma_RCC[last panel]) / d(c0) = {last_panel_sens:.4e}")
    print(f"  {'PASS — coupling exists (nonzero)' if abs(last_panel_sens) > 1e-10 else 'FAIL — appears decoupled, check trace'}")