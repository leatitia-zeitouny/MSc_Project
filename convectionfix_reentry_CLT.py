import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
N            = 12
x_target     = 2.5          # m  — chord length
y_target     = 0.125        # m  — height

panel_length = x_target / N

M_inf  = 5.0
gamma  = 1.4                # ratio of specific heats (can be 1.2–1.3 for dissociated flow)

# Atmosphere at h = 20 km (from standard table)
T_inf   = 205.0             # K
p_inf   = 5543.0            # Pa
rho_inf = 0.09427           # kg/m³
Tw      = 300.0             # K  — structural wall temperature (set / initial condition)
mu_inf  = 1.458e-6 * T_inf**(1.5) / (T_inf + 110.4)   # Sutherland dynamic viscosity
Cp      = 1005.0            # J/(kg·K) — air specific heat

R  = 287.0                  # J/(kg·K)
Pr = 0.71                   # Prandtl number for air

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf

sigma         = 5.67e-8     # W/(m²·K⁴) — Stefan–Boltzmann
T_alum_inner  = 300.0       # K — inner aluminium face (= Tw, structural temperature)


#Allowable Temperatures

T_allow_RCC  = 1900.0   # K — RCC oxidation / structural limit
T_allow_alum = 450.0    # K — Aluminum yield limit (well below melt at 933K)

# ─────────────────────────────────────────────
#  TPS LAYER DEFINITIONS  (outer → inner)
# we have thickness, thermal conductivity, density, Cp, emissivity
#  Order: [0] Aluminum (innermost), [1] SIP, [2] Silica, [3] RCC (outermost/flow side)
# ─────────────────────────────────────────────
"Check the densities here  "
tps_layers = [
    {'name': 'Aluminum', 't': 0.003, 'k': 167.0,  'rho': 2700.0, 'Cp': 900.0,  'eps': 0.125}, #0
    {'name': 'SIP',   't': 0.005, 'k': 0.05,   'rho': 128.0, 'Cp': 1000.0,  'eps': 0.85}, #1
    {'name': 'Silica',      't': 0.050, 'k': 0.06,   'rho':  1900.0, 'Cp': 800.0, 'eps': 0.85}, #2
    {'name': 'RCC',      't': 0.005, 'k': 20.0,   'rho': 1600.0, 'Cp': 711.0,  'eps': 0.875}, #3
]

# Thermal resistances of each layer
"Resistances for conduction are calculated via R=t/kA but we divide by A "
R_layers = [L['t'] / L['k'] for L in tps_layers]
R_total  = R_layers[1]+R_layers[2]+R_layers[3] # in series and without Aluminum when Rcc 

# Thermal mass of RCC face per unit area  m'' = rho * t  [kg/m²]
"This is mass per unit length"
m_RCC    = tps_layers[3]['rho'] * tps_layers[3]['t']    # kg/m²
Cp_RCC   = tps_layers[3]['Cp']                          # J/(kg·K)
m_alum  = tps_layers[0]['rho'] * tps_layers[0]['t']
Cp_alum = tps_layers[0]['Cp']

print("\n" + "=" * 65)
print("  TPS THERMAL RESISTANCE STACK")
print("=" * 65)
print(f"  {'Layer':<10}  {'t (mm)':>7}  {'k (W/mK)':>10}  "
      f"{'R (m²K/W)':>12}  {'R share %':>10}")
for L, R_l in zip(tps_layers, R_layers):
    print(f"  {L['name']:<10}  {L['t']*1000:>7.1f}  {L['k']:>10.3f}  "
          f"  {R_l:>12.5f}  {100*R_l/R_total:>9.1f}%")
print(f"  {'TOTAL':<10}  {sum(L['t'] for L in tps_layers)*1000:>7.1f}  "
      f"{'':>10}    {R_total:>12.5f}")


# ── Nose bluntness — drives BOTH the aeroshell geometry AND the Fay-Riddell
#    stagnation-point heat flux (same physical radius used in both places) ──
R_nose = 0.01  # m — leading-edge/nose radius. Increase for a blunter aeroshell.


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY — spherically-blunted wedge ("aeroshell")
#
#  Two slope-matched (C1 continuous) pieces:
#    1) a circular-arc nose cap of radius R_nose, tangent to the y=0
#       centreline at the stagnation point (0,0)
#    2) a straight afterbody ramp at constant angle theta_w to (x_target, y_target)
#  theta_w is solved (Newton) so the straight segment lands exactly on the
#  target trailing-edge point despite the nose-radius offset.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nodes(theta):
    """Legacy: constant-length, piecewise-linear sharp-wedge node builder.
    Kept for reference; superseded by generate_aeroshell_nodes() below."""
    dx = panel_length * jnp.cos(theta)
    dy = panel_length * jnp.sin(theta)
    x  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dx)])
    y  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dy)])
    return x, y


def solve_wedge_angle(x_target, y_target, Rn, n_iter=60):
    """Newton-solve the afterbody half-angle theta_w so a circular nose cap
    of radius Rn blends smoothly into a straight ramp reaching exactly
    (x_target, y_target)."""
    theta = float(np.arctan(y_target / x_target))
    for _ in range(n_iter):
        def resid(th):
            x_t = Rn * (1.0 - np.sin(th))
            y_t = Rn * np.cos(th)
            return (y_target - y_t) - np.tan(th) * (x_target - x_t)
        f    = resid(theta)
        dth  = 1e-7
        df   = (resid(theta + dth) - f) / dth
        theta = theta - f / df
    return float(theta)


def generate_aeroshell_nodes(N, x_target, y_target, R_nose, n_nose_frac=0.35):
    """Blunted-nose wedge: first ~n_nose_frac*N panels trace the circular
    nose cap, remaining panels are a straight afterbody ramp."""
    n_nose  = max(2, int(round(N * n_nose_frac)))
    n_after = N - n_nose
    if n_after < 1:
        n_after = 1
        n_nose  = N - 1

    theta_w = solve_wedge_angle(x_target, y_target, R_nose)
    phi_t   = jnp.pi / 2.0 - theta_w

    phi_nodes = jnp.linspace(0.0, phi_t, n_nose + 1)
    x_nose = R_nose * (1.0 - jnp.cos(phi_nodes))
    y_nose = R_nose * jnp.sin(phi_nodes)

    x_t, y_t = x_nose[-1], y_nose[-1]

    s = jnp.linspace(0.0, 1.0, n_after + 1)[1:]
    x_after = x_t + s * (x_target - x_t)
    y_after = y_t + s * (y_target - y_t)

    x = jnp.concatenate([x_nose, x_after])
    y = jnp.concatenate([y_nose, y_after])

    dx = jnp.diff(x)
    dy = jnp.diff(y)
    theta = jnp.arctan2(dy, dx)

    return x, y, theta, x_t, y_t, theta_w, n_nose


# ═══════════════════════════════════════════════════════════════════════════════
#  AERODYNAMICS — Newtonian + isentropic
# ═══════════════════════════════════════════════════════════════════════════════
def newtonian_cp(theta):
    return jnp.maximum(2.0 * jnp.sin(theta)**2,0)

def pressure_distribution(theta):
    """Local pressure via Newtonian."""
    Cp_local = newtonian_cp(theta)
    q_inf    = 0.5 * rho_inf * V_inf**2
    return p_inf + q_inf * Cp_local

def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


def isentropic_temperature(p):
    """Local temperature via isentropic relations."""
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    T0 = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
    return T0 * (p / p0)**((gamma-1.0)/gamma)

def density_from_pt(p, T):
    """Local density from pressure (Newtonian) and temperature (isentropic)."""
    return p / (R * T)

def local_mach(p):
    """Local Mach number from isentropic relation."""
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    return jnp.sqrt((2.0/(gamma-1.0)) * ((p0/p)**((gamma-1.0)/gamma) - 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def adiabatic_wall_temperature(p, T):
    M_local  = local_mach(p)
    r_lam    = Pr**(0.5)
    r_turb   = Pr**(1.0/3.0)
    Taw_lam  = T * (1.0 + r_lam  * (gamma-1.0)/2.0 * M_local**2)
    Taw_turb = T * (1.0 + r_turb * (gamma-1.0)/2.0 * M_local**2)
    return Taw_lam, Taw_turb

def transition_location(x_safe, rho_local, p, T):
    M_local  = local_mach(p)
    a_local  = jnp.sqrt(gamma * R * T)
    V_local  = M_local * a_local
    mu_local = 1.458e-6 * T**(1.5) / (T + 110.4)

    Rex      = rho_local * V_local * x_safe / mu_local  # Local Reynolds 
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)          # Blasius momentum thickness
    Re_theta = rho_local * V_local * theta_m / mu_local  # Reynolds number dependent on momentum thickness 
    criterion = Re_theta / M_local

    has_transition = jnp.any(criterion >= 400.0)
    idx            = jnp.argmax(criterion >= 400.0)
    x_tr           = jnp.where(has_transition, x_safe[idx], x_safe[-1])
    return x_tr, Re_theta


def boundary_layer(x, p, T):
    M_local   = local_mach(p)
    a_local   = jnp.sqrt(gamma * R * T)
    V_local   = M_local * a_local
    rho_local = density_from_pt(p, T)
    mu_local  = 1.458e-6 * T**(1.5) / (T + 110.4)

    x_safe = jnp.maximum(x, 1e-6)
    Rex    = rho_local * V_local * x_safe / mu_local

    x_tr, Re_theta = transition_location(x_safe, rho_local, p, T)

    Taw_lam, Taw_turb = adiabatic_wall_temperature(p, T)
    Taw = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    comp_factor = (1.0 + 0.016*M_inf**2 + 0.072*(Tw/Taw)*M_inf**2)
    delta_lam   = (5.0 * x_safe / jnp.sqrt(Rex)
                   * (Tw / T_inf)**(-1.0/6.0)
                   * comp_factor)

    x_from_tr  = jnp.maximum(x_safe - x_tr, 1e-6)
    Rex_tr     = rho_local * V_local * x_from_tr / mu_local
    delta_turb = 0.37 * x_from_tr / Rex_tr**0.2 * (Taw / T_inf)**0.6

    delta = jnp.where(x_safe <= x_tr, delta_lam, delta_turb)
    return delta, delta_lam, delta_turb, x_tr, Re_theta, Taw


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL ANGLES & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
x, y, theta, x_nose_end, y_nose_end, theta_w, n_nose_panels = generate_aeroshell_nodes(
    N, x_target, y_target, R_nose
)
x_panel     = x[1:]               # panel representative x = downstream node

print(f"\n  [Aeroshell geometry] R_nose = {float(R_nose)*1000:.1f} mm   "
      f"theta_w = {float(jnp.rad2deg(theta_w)):.2f} deg   "
      f"nose panels = {n_nose_panels}/{N}   "
      f"tangent at x = {float(x_nose_end):.4f} m, y = {float(y_nose_end):.4f} m")

p           = pressure_distribution(theta)
T           = isentropic_temperature(p)
rho_local   = density_from_pt(p, T)

delta, delta_lam, delta_turb, x_tr, Re_theta, Taw = boundary_layer(x_panel, p, T)

dp_dx = jnp.gradient(p, x_panel)
dT_dx = jnp.gradient(T, x_panel)


# ═══════════════════════════════════════════════════════════════════════════════
#  TPS HEAT TRANSFER  —  TRANSIENT RCC TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Energy balance on the RCC face (per unit area):
#
#    m_RCC * Cp_RCC * dT_RCC/dt  =  q_conv  −  q_cond  −  q_rad
#
#  where:
#    q_conv = h_BL * (Taw − T_RCC)          [convection from boundary layer]
#    q_cond = (T_RCC − T_alum_inner)/R_tot  [conduction inward through stack]
#    q_rad  = eps * sigma * (T_RCC^4 − T_inf^4) [radiation to freestream]
# Two coupled ODEs per panel:
#
#    (1) RCC face:
#        m_RCC * Cp_RCC * dT_RCC/dt = q_conv  −  q_cond  −  q_rad   --> we get TRcc
#
#    (2) Aluminium face (inner boundary — adiabatic on the structure side):
#        m_alum * Cp_alum * dT_alum/dt = q_cond  --> we get Tal
# ═══════════════════════════════════════════════════════════════════════════════
# Aluminium thermal mass per unit area
m_alum   = tps_layers[0]['rho'] * tps_layers[0]['t']   # kg/m²
Cp_alum  = tps_layers[0]['Cp']                         # J/(kg·K)
# Precompute V_local for passing into transient solver

# Let's calculate the total q and q_saved
# (R_nose already defined in CONFIGURATION above — reused here so Fay-Riddell
# stays consistent with the actual blunted-nose geometry)

def fay_riddell(p_stag, T_stag, T_w, R_nose):
    """
    Fay-Riddell stagnation point heat flux [W/m²].
    """
    # Stagnation conditions
    rho_stag = p_stag / (R * T_stag)
    mu_stag  = sutherland(T_stag)

    # Wall conditions at stagnation
    rho_w = p_stag / (R * T_w)
    mu_w  = sutherland(T_w)

    # Velocity gradient at stagnation point — Newtonian approximation
    du_e_dx = (1.0 / R_nose) * jnp.sqrt(
        2.0 * jnp.maximum(p_stag - p_inf, 0.0) / rho_stag
    )

    # Adiabatic wall temperature at stagnation
    T_aw_stag = T_stag

    q_stag = (0.763
              * Pr**(-0.6)
              * (rho_w   * mu_w  )**0.1
              * (rho_stag * mu_stag)**0.4
              * jnp.sqrt(du_e_dx)
              * Cp * (T_aw_stag - T_w))

    return jnp.maximum(q_stag, 0.0)   # clamp — wall can't heat flow backward here


def van_driest_II_Cf(Re_x_e, M_e, T_e, T_w, T_aw):
    """
    Van Driest II compressible turbulent skin friction coefficient.
    """
    r_turb = Pr**(1.0 / 3.0)

    T_ratio_aw =  1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2
    T_ratio_w  = T_w  / T_e

    A_sq = jnp.maximum(
        (r_turb * (gamma - 1.0) / 2.0 * M_e**2)
        / (T_ratio_w + 1e-10),
        0.0
    )
    A = jnp.sqrt(A_sq)
    B = T_ratio_aw/(T_ratio_w+1e-10)-1    # > 0 for cooled wall (typical TPS)

    denom = jnp.sqrt(B**2 + 4.0 * A**2 + 1e-30)

    alpha = jnp.clip((2.0 * A**2 - B) / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    beta  = jnp.clip(B / denom,                 -1.0 + 1e-7, 1.0 - 1e-7)

    arcsin_sum = jnp.arcsin(alpha) + jnp.arcsin(beta)
    M_e = local_mach(p)
    Fc = jnp.where(
            T_ratio_aw > 1.001,
            (T_ratio_aw - 1.0) / (arcsin_sum**2 + 1e-30),
            1.0
        )
    omega=0.76
    Fx=(1/T_ratio_w)**omega
    Re_x_inc=Fx*Re_x_e
        # ── Incompressible Cf: Karman-Schlichting implicit formula ────────────────
        # 1/sqrt(Cf) = 4.15 * log10(Re_x * Cf) + 1.7
    Cf_inc = 0.0592 * Re_x_inc**(-0.2)    # initial guess
        # Newton-Raphson Solver
    for _ in range(6):
        lhs  = 1.0 / jnp.sqrt(Cf_inc + 1e-30)
        rhs  = 4.15 * jnp.log10(Re_x_inc * Cf_inc + 1e-30) + 1.7
        f    = lhs - rhs
        dfdC = (-0.5 / (Cf_inc + 1e-30)**1.5
                    - 4.15 / ((Cf_inc + 1e-30) * jnp.log(10.0)))
        Cf_inc = jnp.maximum(Cf_inc - f / (dfdC + 1e-30), 1e-8)
    
    Cf_comp = Cf_inc / (Fc + 1e-30)
    
    
    return Cf_comp, Fc, Cf_inc
    

def heat_transfer_3region(x_safe, p, T_e, T_w_current, x_tr):
    """                               
    Three-region hypersonic heat transfer model.
    """
    r_lam  = Pr**0.5
    r_turb = Pr**(1.0 / 3.0)

    M_e = local_mach(p)
    a_e = jnp.sqrt(gamma * R * T_e)
    V_e = M_e * a_e

    rho_e = density_from_pt(p, T_e)
    mu_e  = sutherland(T_e)

    Taw_lam  = T_e * (1.0 + r_lam  * (gamma - 1.0) / 2.0 * M_e**2)
    Taw_turb = T_e * (1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2)
    Taw      = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    T_ref_lam   = T_e * (0.45 + 0.55 * (T_w_current / T_e)
                         + 0.16 * r_lam * (gamma - 1.0) / 2.0 * M_e**2)
    rho_ref_lam = p / (R * T_ref_lam)
    mu_ref_lam  = sutherland(T_ref_lam)
    Rex_lam     = rho_ref_lam * V_e * x_safe / mu_ref_lam

    St_lam      = (0.332 / jnp.sqrt(Rex_lam)) * Pr**(-2.0 / 3.0)
    q_lam       = rho_ref_lam * V_e * St_lam * Cp * (Taw_lam - T_w_current)

    Re_x_e = rho_e * V_e * x_safe / mu_e

    Cf_comp, Fc, Cf_inc = van_driest_II_Cf(
        Re_x_e, M_e, T_e, T_w_current, Taw_turb
    )

    St_turb     = (Cf_comp / 2.0) * Pr**(-2.0 / 3.0)
    q_turb      = rho_e * V_e * St_turb * Cp * (Taw_turb - T_w_current)

    is_lam  = x_safe < x_tr
    St      = jnp.where(is_lam, St_lam,        St_turb)
    q_conv  = jnp.where(is_lam, q_lam,         q_turb)
    rho_ref = jnp.where(is_lam, rho_ref_lam,   rho_e)

    p_stag  = p_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)**(gamma / (gamma - 1.0))
    T_stag  = T_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)

    q_stag  = fay_riddell(p_stag, T_stag, T_w_current[0], R_nose)

    stag_mask = jnp.arange(len(x_safe)) == 0
    q_conv    = jnp.where(stag_mask, q_stag,  q_conv)
    St        = jnp.where(stag_mask,
                          q_stag / jnp.maximum(
                              rho_ref[0] * V_e[0] * Cp
                              * jnp.maximum(Taw[0] - T_w_current[0], 1.0),
                              1e-10),
                          St)

    q_conv = jnp.maximum(q_conv, 0.0)

    return St, q_conv, rho_ref, Taw, V_e




M_local = local_mach(p)
V_local = M_local * jnp.sqrt(gamma * R * T)
eps_RCC = tps_layers[3]['eps']

def tps_transient_3region(x_safe, p, T_e, x_tr,
                          t_end=600.0, dt_init=0.1,
                          tol=1e-3, save_every=200):

    x_safe_np = np.array(x_safe)
    p_np      = np.array(p)
    T_e_np    = np.array(T_e)

    T_RCC  = np.full(N, float(Tw), dtype=np.float64)
    T_alum = np.full(N, float(Tw), dtype=np.float64)

    # ── Compute t=0 fluxes FIRST so they exist before appending ──────────────
    T_w_jnp = jnp.array(T_RCC)
    _, q_conv_j, _, _, _ = heat_transfer_3region(
        jnp.array(x_safe_np),
        jnp.array(p_np),
        jnp.array(T_e_np),
        T_w_jnp,
        x_tr
    )
    q_conv = np.array(q_conv_j)                                      # now defined
    q_cond = (T_RCC - T_alum) / np.array(R_total)         # now defined        # now defined
    q_rad  = eps_RCC * sigma * (T_RCC**4 - T_inf**4) 

    # ── Now safe to initialise history lists with t=0 values ─────────────────
    t_hist        = [0.0]
    T_RCC_hist  = [T_RCC.copy()]
    T_alum_hist = [T_alum.copy()]
    q_conv_hist   = [q_conv.copy()]
    q_rad_hist    = [q_rad.copy()]    
    q_cond_hist   = [q_cond.copy()]    

    t, step, t_ss = 0.0, 0, None
    t_fail_RCC  = None
    t_fail_alum    = None

    while t < t_end:

        # ── Recompute q_conv with current T_shell ─────────────────────────────
        T_w_jnp = jnp.array(T_RCC)
        _, q_conv_j, _, Taw_j, _ = heat_transfer_3region(
            jnp.array(x_safe_np),
            jnp.array(p_np),
            jnp.array(T_e_np),
            T_w_jnp,
            x_tr
        )
        q_conv = np.array(q_conv_j)

        # ── Conduction and radiation ───────────────────────────────────────────
        q_cond = (T_RCC - T_alum) / np.array(R_total)
        q_rad  = eps_RCC * sigma * (T_RCC**4 - T_inf**4)

        # ── Energy balance ─────────────────────────────────────────────────────
        dT_RCC_dt  = (q_conv - q_cond -q_rad) / (m_RCC * Cp_RCC)
        dT_Al_dt = q_cond / (m_alum * Cp_alum)

        # ── Adaptive timestep ──────────────────────────────────────────────────
        max_rate = max(
            np.abs(dT_RCC_dt).max(),
            np.abs(dT_Al_dt).max(),
            1e-12
        )
        dt = min(dt_init, 2.0 / max_rate, t_end - t)

        T_RCC  += dt * dT_RCC_dt
        T_alum += dt * dT_Al_dt
        t += dt; step += 1

        # ── Material limit checks ──────────────────────────────────────────────
        if t_fail_RCC is None and np.any(T_RCC > T_allow_RCC):
            t_fail_RCC = t
            failed_panels = np.where(T_RCC > T_allow_RCC)[0]
            print(f"  [!] RCC limit exceeded at t={t:.1f}s  "
                  f"panels={failed_panels}  T_max={T_RCC.max():.1f}K")

        if t_fail_alum is None and np.any(T_alum > T_allow_alum):
            t_fail_alum = t
            print(f"  [!] Aluminum limit exceeded at t={t:.1f}s  "
                  f"T_max={T_alum.max():.1f}K")

        # ── Save snapshots ─────────────────────────────────────────────────────
        if step % save_every == 0:
            t_hist.append(t)
            T_RCC_hist.append(T_RCC.copy())
            T_alum_hist.append(T_alum.copy())
            q_conv_hist.append(q_conv.copy())
            q_cond_hist.append(q_cond.copy())
            q_rad_hist.append(q_rad.copy())

        # ── Steady state check ─────────────────────────────────────────────────
        if np.abs(dT_RCC_dt).max() < tol and t_ss is None:
            t_ss = t
            print(f"  [Transient] Shell quasi-steady at t={t:.1f}s  "
                  f"T_RCC_max={T_RCC.max():.1f}K  "
                  f"T_alum_max={T_alum.max():.1f}K  "
                  f"(Al timescale >> {t_end:.0f}s)")

    # ── Final state ────────────────────────────────────────────────────────────
    t_hist.append(t)
    T_RCC_hist.append(T_RCC.copy())
    T_alum_hist.append(T_alum.copy())

    # Final flux recompute
    T_w_jnp = jnp.array(T_RCC)
    _, q_conv_j, _, Taw_j, _ = heat_transfer_3region(
        jnp.array(x_safe_np), jnp.array(p_np),
        jnp.array(T_e_np), T_w_jnp, x_tr
    )

    T_RCC_j, T_alum_j = jnp.array(T_RCC), jnp.array(T_alum)
    q_conv_f  = q_conv_j
    q_cond_f  = (T_RCC_j - T_alum_j) / jnp.array(R_total)
    q_rad_f   = eps_RCC * sigma * (T_RCC_j**4 - T_inf**4)
    q_saved_f = jnp.array(q_conv_fixed) - q_cond_f

    # Interface temperatures
    T_RCC_in = T_RCC_j  - q_cond_f * R_layers[3]
    T_sil_in = T_RCC_in - q_cond_f * R_layers[2]
    T_sip_in = T_sil_in - q_cond_f * R_layers[1]
    T_alum_in = T_sip_in

    T_iface_final = jnp.stack(
        [T_RCC_j, T_RCC_in, T_sil_in, T_sip_in, T_alum_j], axis=0
        #                                          ^^^^^^^
        #                    index 4 = actual ODE-solved Aluminum temperature
    )

    return (t_hist, T_RCC_hist, T_alum_hist, T_iface_final, q_cond_f, q_conv_f, q_rad_f, q_saved_f,
            t_ss, t_fail_RCC, t_fail_alum,
            np.array(q_conv_hist),
            np.array(q_cond_hist),
            np.array(q_rad_hist),
            T_RCC_in, T_sil_in, T_sip_in, T_alum_in, T_RCC_j, T_alum_j)


print("\n  Running transient TPS integration …")
x_safe=jnp.maximum(x_panel,1e-6)
T_e = T
T_w_fixed = jnp.full(N, float(Tw))

St_fixed, q_conv_fixed_j, rho_ref_fixed, Taw_fixed, V_e_fixed = heat_transfer_3region(
    x_safe, p, T_e, T_w_fixed, x_tr
)
q_conv_fixed = np.array(q_conv_fixed_j)

(t_hist, T_RCC_hist, T_alum_hist, T_interfaces, q_cond, q_conv, q_rad, q_saved, t_ss,t_fail_RCC,t_fail_alum,q_conv_hist, q_cond_hist, q_rad_hist,T_RCC_in,T_sil_in,T_sip_in,T_alum_in,T_RCC_j,T_alum_j) = tps_transient_3region(x_safe, p, T_e, x_tr,
                          t_end=600.0, dt_init=0.1,
                          tol=1e-3, save_every=200)

# to compute q with no TPS- it's only convection but with T_w=300K



# COMPUTE EXPANSION AND THERMAL STRESSES
thermo_mech = {

    'Aluminum': {
        'alpha_par' : 23.1e-6,    # [1/K]
        'alpha_perp': 23.1e-6,    # [1/K]  isotropic
        'E_par'     : 69.0e9,     # [Pa]
        'E_perp'    : 69.0e9,     # [Pa]   isotropic
        'note'      : 'FCC Al — fully isotropic, beta=3alpha valid'
    },

    'SIP': {
        'alpha_par' : 250e-6,     # [1/K]
        'alpha_perp': 250e-6,     # [1/K]  isotropic foam
        'E_par'     : 0.001e9,    # [Pa]   very soft
        'E_perp'    : 0.001e9,    # [Pa]
        'note'      : 'Silicone foam — isotropic, very compliant, beta=3alpha valid'
    },

    'Silica': {
        'alpha_par' : 0.5e-6,     # [1/K]  amorphous SiO2, very low
        'alpha_perp': 0.6e-6,     # [1/K]  slightly higher through-thickness
        'E_par'     : 0.05e9,     # [Pa]   porous — very low stiffness
        'E_perp'    : 0.03e9,     # [Pa]   softer through-thickness
        'note'      : 'Porous amorphous silica tile — near-isotropic, low alpha'
    },

    'RCC': {
        'alpha_par' : 1.0e-6,     # [1/K]  in-plane, fibre direction
        'alpha_perp': 3.5e-6,     # [1/K]  through-thickness, matrix dominated
        'E_par'     : 60.0e9,     # [Pa]   in-plane stiffness
        'E_perp'    : 15.0e9,     # [Pa]   through-thickness (weaker)
        'note'      : 'Woven C/C — transversely isotropic, strong anisotropy'
    },
}


def compute_thermal_expansion(T_RCC_in, T_sil_in, T_sip_in, T_alum_in, x_safe,
                              T_RCC_j, T_alum_j, T_ref=None):
    """
    Geometric growth (delta_L, delta_V): unchanged — free through-thickness
    dimensional change of each layer using alpha_perp, driven by the local
    conduction drop across that layer. Kinematic diagnostic only, not stress.

    Thermal STRESS: multilayer beam (force + moment balance) — the N-layer
    generalization of the classical Timoshenko bimetallic-strip formula,
    applied here to this stack's 4 real layers, base -> outer:
        Aluminum (bonded to airframe) / SIP / Silica / RCC (flow side)

        eps(z)     = eps0 + kappa*z
        sigma_i(z) = E_par_i * (eps(z) - alpha_par_i * (T_i - T_ref))

    eps0 is pinned to Aluminum's own free thermal strain (it's bonded to the
    much larger/stiffer airframe, which dominates the boundary condition):
        eps0 = alpha_par_Al * (T_alum - T_ref)
    kappa (how much the stack bows) is solved from moment balance M=0,
    since nothing external stops the stack curving away from the structure
    (the "free to bow" / pinned-support bound — see conversation notes on
    the aeroshell/Ti-Silicone-Aerogel-Buffer-Shell version this is ported
    from for the rigid-fastener alternative bound).

    Uses IN-PLANE (alpha_par/E_par) properties, not through-thickness
    (alpha_perp/E_perp) — bonded-layer stress comes from in-plane CTE
    mismatch at the shared interface, not through-thickness squeezing
    (the outer RCC face is a free surface).
    """
    if T_ref is None:
        T_ref = float(Tw)   # assembly / stress-free temperature

    dT_RCC = T_RCC_j - T_RCC_in
    dT_sil = T_RCC_in - T_sil_in
    dT_sip = T_sil_in - T_sip_in

    # ── Linear (through-thickness) expansion — unchanged, kinematic only ──────
    delta_L_RCC = tps_layers[3]['t'] * thermo_mech['RCC']['alpha_perp'] * dT_RCC
    delta_L_sil = tps_layers[2]['t'] * thermo_mech['Silica']['alpha_perp'] * dT_sil
    delta_L_sip = tps_layers[1]['t'] * thermo_mech['SIP']['alpha_perp'] * dT_sip
    delta_L_Al  = jnp.zeros_like(dT_RCC)

    delta_V_RCC = (2*thermo_mech['RCC']['alpha_par']+thermo_mech['RCC']['alpha_perp'])*tps_layers[3]['t']*dT_RCC
    delta_V_sil = (2*thermo_mech['Silica']['alpha_par']+thermo_mech['Silica']['alpha_perp'])*tps_layers[2]['t']*dT_sil
    delta_V_sip = (2*thermo_mech['SIP']['alpha_par']+thermo_mech['SIP']['alpha_perp'])*tps_layers[1]['t']*dT_sip
    delta_V_Al  = jnp.zeros_like(dT_RCC)

    # ── Bonded-laminate multilayer beam (force + moment balance) ─────────────
    T_RCC_mean  = 0.5 * (T_RCC_j  + T_RCC_in)
    T_sil_mean  = 0.5 * (T_RCC_in + T_sil_in)
    T_sip_mean  = 0.5 * (T_sil_in + T_sip_in)
    T_alum_mean = T_alum_j    # Aluminum is thin & highly conductive — ~uniform through its own thickness

    alpha_par_RCC,  E_par_RCC,  t_RCC  = thermo_mech['RCC']['alpha_par'],     thermo_mech['RCC']['E_par'],     tps_layers[3]['t']
    alpha_par_sil,  E_par_sil,  t_sil  = thermo_mech['Silica']['alpha_par'],  thermo_mech['Silica']['E_par'],  tps_layers[2]['t']
    alpha_par_sip,  E_par_sip,  t_sip  = thermo_mech['SIP']['alpha_par'],     thermo_mech['SIP']['E_par'],     tps_layers[1]['t']
    alpha_par_Al,   E_par_Al,   t_Al   = thermo_mech['Aluminum']['alpha_par'],thermo_mech['Aluminum']['E_par'],tps_layers[0]['t']

    dT_RCC_g = T_RCC_mean  - T_ref
    dT_sil_g = T_sil_mean  - T_ref
    dT_sip_g = T_sip_mean  - T_ref
    dT_Al_g  = T_alum_mean - T_ref

    # z-positions of each REAL layer (base=Aluminum at z=0, outward through SIP/Silica/RCC)
    t_list = [t_Al, t_sip, t_sil, t_RCC]
    t_arrs = [t_i if hasattr(t_i, 'shape') else jnp.full_like(dT_RCC_g, t_i) for t_i in t_list]
    t_total = sum(t_arrs)
    z_bounds = [-0.5 * t_total]
    for t_i in t_arrs:
            z_bounds.append(z_bounds[-1] + t_i)
    
    z_mid = [0.5 * (z_bounds[i] + z_bounds[i+1]) for i in range(4)]
    E_list  = [E_par_Al,     E_par_sip,     E_par_sil,     E_par_RCC]
    al_list = [alpha_par_Al, alpha_par_sip, alpha_par_sil, alpha_par_RCC]
    dT_list = [dT_Al_g,      dT_sip_g,      dT_sil_g,      dT_RCC_g]

    A12 = jnp.zeros_like(dT_RCC_g)
    A11=jnp.zeros_like(dT_RCC_g)
    A22 = jnp.zeros_like(dT_RCC_g)
    N_T=jnp.zeros_like(dT_RCC_g)
    M_T= jnp.zeros_like(dT_RCC_g)
    for i in range(4):
        z0, z1 = z_bounds[i], z_bounds[i+1]
        E_i = E_list[i] if hasattr(E_list[i], 'shape') else jnp.full_like(dT_RCC_g, E_list[i])
        a=z1-z0
        b = (z1**2 - z0**2) / 2.0
        c = (z1**3 - z0**3) / 3.0
        A11=A11+E_i*a; A12=A12+E_i*b; A22=A22+E_i*c
        N_T=N_T+E_i*al_list[i]*dT_list[i]*a
        M_T=M_T+E_i*al_list[i]*dT_list[i]*b

    det = A11 * A22 - A12**2
    eps0_free  = (A22 * N_T - A12 * M_T) / det  # True unconstrained midplane strain
    kappa_free = (A11 * M_T - A12 * N_T) / det

    

    sigma_Al  = E_par_Al  * (eps0_free + kappa_free*z_mid[0] - alpha_par_Al  * dT_Al_g)
    sigma_sip = E_par_sip * (eps0_free + kappa_free*z_mid[1] - alpha_par_sip * dT_sip_g)
    sigma_sil = E_par_sil * (eps0_free + kappa_free*z_mid[2] - alpha_par_sil * dT_sil_g)
    sigma_RCC = E_par_RCC * (eps0_free + kappa_free*z_mid[3] - alpha_par_RCC * dT_RCC_g)

    return (delta_L_RCC,delta_L_sil,delta_L_sip,delta_L_Al,delta_V_RCC,delta_V_sil,delta_V_sip,delta_V_Al,sigma_RCC,sigma_sil,
            sigma_sip,sigma_Al)


x_safe = jnp.maximum(x_panel, 1e-6)

(delta_L_RCC,delta_L_sil,delta_L_sip,delta_L_Al,delta_V_RCC,delta_V_sil,delta_V_sip,delta_V_Al,sigma_RCC,sigma_sil,
            sigma_sip,sigma_Al) = \
    compute_thermal_expansion(T_RCC_in,T_sil_in,T_sip_in, T_alum_in, x_safe,T_RCC_j, T_alum_j)

# PRINTOUT

# ── Column widths ─────────────────────────────────────────────────────────────
WP = {'x': 7, 'dL': 12, 'dV': 12, 'sig': 12}

layer_labels = ['RCC', 'Silica', 'SIP', 'Aluminum']

delta_L_all = [delta_L_RCC, delta_L_sil, delta_L_sip, delta_L_Al]
delta_V_all = [delta_V_RCC, delta_V_sil, delta_V_sip, delta_V_Al]
sigma_all   = [sigma_RCC,   sigma_sil,   sigma_sip,   sigma_Al]

for lbl, dL, dV, sig in zip(layer_labels, delta_L_all, delta_V_all, sigma_all):

    print(f"\n{'='*70}")
    print(f"  LAYER: {lbl}")
    print(f"{'='*70}")
    print(f"  {'x[m]':>{WP['x']}}  "
          f"{'delta_L[µm]':>{WP['dL']}}  "
          f"{'delta_V[µm]':>{WP['dV']}}  "
          f"{'sigma[MPa]':>{WP['sig']}}")
    print(f"  {'─'*50}")

    for i in range(N):
        print(f"  {float(x_panel[i]):>{WP['x']}.3f}  "
              f"{float(dL[i])*1e6:>{WP['dL']}.6f}  "
              f"{float(dV[i])*1e6:>{WP['dV']}.6f}  "
              f"{float(sig[i])/1e6:>{WP['sig']}.6f}")

    print(f"\n  Max |delta_L| : {float(jnp.abs(dL).max())*1e6:.6f} µm  "
          f"at x = {float(x_panel[jnp.abs(dL).argmax()]):.3f} m")
    print(f"  Max |delta_V| : {float(jnp.abs(dV).max())*1e6:.6f} µm  "
          f"at x = {float(x_panel[jnp.abs(dV).argmax()]):.3f} m")
    print(f"  Max |sigma|   : {float(jnp.abs(sig).max())/1e6:.6f} MPa  "
          f"at x = {float(x_panel[jnp.abs(sig).argmax()]):.3f} m")
    

# PLOT 

def plot_thermal_expansion(x_panel, x_tr,
                           delta_L_all, delta_V_all, sigma_all,
                           savename='thermal_expansion.png'):

    x_p    = np.array(x_panel)
    labels = ['RCC', 'Silica',
              'SIP', 'Aluminum']
    colors  = ['#c0392b', '#f1c40f', '#3498db', '#9b59b6', '#2c3e50']
    markers = ['o', 's', '^', 'D', 'v']
    lines   = ['-', '--', '-.', ':', '-']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Thermal Expansion & Stress — Per Layer Per Panel\n'
                 f'M∞ = {M_inf:.1f},  h = 20 km',
                 fontsize=13, fontweight='bold')

    datasets = [
        (axes[0], delta_L_all, 'δL  [µm]',
         'Linear Expansion  δL = α_perp · L₀ · ΔT\n(thickness direction)',  1e6),
        (axes[1], delta_V_all, 'δV  [µm]',
         'Volumetric Expansion  δV = V₀ · β · ΔT\nβ = 2α_par + α_perp',    1e6),
        (axes[2], sigma_all,   'σ  [MPa]',
         'Thermal Stress  σ = E_perp · α_perp · ΔT\n(fully constrained)',   1e-6),
    ]

    for ax, data, ylabel, title, scale in datasets:
        for arr, lbl, col, mk, ls in zip(data, labels, colors, markers, lines):
            ax.plot(x_p, np.array(arr) * scale,
                    marker=mk, color=col, linestyle=ls,
                    linewidth=1.8, markersize=5, label=lbl)
        ax.axhline(0, color='k', linewidth=0.8)
        ax.axvline(float(x_tr), color='grey', linestyle=':',
                   linewidth=1.2, label=f'x_tr={float(x_tr):.2f} m')
        ax.set_xlabel('x (m)', fontsize=11)
        ax.set_ylabel(ylabel,  fontsize=11)
        ax.set_title(title,    fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()

# ── Call ──────────────────────────────────────────────────────────────────────
plot_thermal_expansion(
    x_panel, x_tr,
    [delta_L_RCC, delta_L_sil, delta_L_sip, delta_L_Al],
    [delta_V_RCC, delta_V_sil, delta_V_sip, delta_V_Al],
    [sigma_RCC,   sigma_sil,   sigma_sip,   sigma_Al  ]
)







# ─────────────────────────────────────────────
#  PRINT TPS RESULTS
# ─────────────────────────────────────────────
q_to_alum = q_cond
print("\n" + "=" * 75)
print("  TPS HEAT TRANSFER — PANEL BY PANEL  (FINAL / QUASI-STEADY STATE)")
print("=" * 65)
print(f"  {'x[m]':>7}  {'q_conv[W/m²]':>13} {'q_conv[W/m²]':>13} {'q_rad[W/m²]':>12}  {'q_to_alum[W/m²]':>16}  "
      f"{'q_saved[W/m²]':>14}  {'saving%':>8}  {'T_RCC[K]':>9}  {'T_alum[K]':>9}")
for i in range(N):
    q_conv_i = float(q_conv[i])
    q_conv_ref_i=float(q_conv_fixed[i])
    saving_pct = 100.0 * float(q_saved[i]) / max(q_conv_ref_i, 1e-6)
    print(f"  {float(x_panel[i]):7.3f}  "
          f"{q_conv_i:13.2f}  "
          f"{q_conv_ref_i:13.2f}  "
          f"{float(q_rad[i]):12.2f}  "  
          f"{float(q_to_alum[i]):16.4f}  " 
          f"{float(q_saved[i]):14.2f}  "
          f"{saving_pct:8.1f}%  "
          f"{float(T_interfaces[0, i]):9.1f}  "
          f"{float(T_interfaces[4, i]):9.1f}")
 
print(f"\n  {'─'*70}")
print(f"  Total q_to_alum : {float(q_to_alum.sum()):12.2f} W/m²")
print(f"  Total q_saved   : {float(q_saved.sum()):12.2f} W/m²")
print(f"  Mean saving     : {100*float(q_saved.mean()/(q_saved+q_to_alum).mean()):12.1f}%")
print(f"  {'─'*70}")
print(f"  Max T_RCC outer : {float(T_interfaces[0].max()):10.1f} K")
print(f"  Max T at alum   : {float(T_interfaces[4].max()):10.1f} K")
if t_ss is not None:
    print(f"  Steady state    : t = {t_ss:.1f} s")
else:
    print(f"  Steady state    : NOT reached within t_end")

# Material Allowable Temperature Limit
"we are adding as a constraint the temperature the material selected can sustain"
rcc_ok  = float(T_interfaces[0].max()) < T_allow_RCC
alum_ok = float(T_interfaces[4].max()) < T_allow_alum
print(f"\n  MATERIAL STATUS:")
print(f"    RCC  : {'OK' if rcc_ok  else 'EXCEEDED':<10}  "
      f"max = {float(T_interfaces[0].max()):.1f} K  /  limit = {T_allow_RCC:.0f} K")
print(f"    Alum : {'OK' if alum_ok else 'EXCEEDED':<10}  "
      f"max = {float(T_interfaces[4].max()):.1f} K  /  limit = {T_allow_alum:.0f} K")
if t_fail_RCC  is not None: print(f"    RCC  first exceeded limit at t = {t_fail_RCC:.1f} s")
if t_fail_alum is not None: print(f"    Alum first exceeded limit at t = {t_fail_alum:.1f} s")
if t_ss is not None:
    print(f"  Steady state     : t = {t_ss:.1f} s")
else:
    print(f"  Steady state     : NOT reached within t_end")



# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def draw_vehicle(ax, x, y, zorder=2):
    """Surface, vertical base, and horizontal bottom line."""
    ax.plot(x, y,                               'k-', linewidth=2.5, zorder=zorder)
    ax.plot([x_target, x_target], [0.0, y_target], 'k-', linewidth=2.5, zorder=zorder)
    ax.plot([0.0, x_target], [0.0, 0.0],        'k-', linewidth=2.5, zorder=zorder)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — TPS GEOMETRY  (layers INSIDE the main geometry)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_layers(x, y, theta, tps_layers, savename='tps_layers.png'):
    fig, ax = plt.subplots(figsize=(15, 5))
 
    x_np = np.array(x, dtype=float)   # (N+1,)
    y_np = np.array(y, dtype=float)   # (N+1,)
 
    layer_order  = [3, 2, 1, 0]
    t_real       = np.array([tps_layers[i]['t'] for i in layer_order])   # (4,)
    t_total_real = t_real.sum()
 
    colors_rev = ['#c0392b', '#e67e22', '#f1c40f', '#95a5a6']
    labels_rev = [
        f"RCC       {tps_layers[3]['t']*1000:.0f} mm  k={tps_layers[3]['k']} W/mK",
        f"Silica    {tps_layers[2]['t']*1000:.0f} mm  k={tps_layers[2]['k']} W/mK",
        f"SIP       {tps_layers[1]['t']*1000:.0f} mm  k={tps_layers[1]['k']} W/mK",
        f"Aluminum  {tps_layers[0]['t']*1000:.0f} mm  k={tps_layers[0]['k']} W/mK",
    ]
 
    vis_fraction = 0.55     # 55 % of local y — keeps everything inside
 
    t_cumul_norm = np.concatenate([[0.0], np.cumsum(t_real) / t_total_real])  # (5,)
 
    def offset_curve(frac):
        y_curve = y_np * (1.0 - frac * vis_fraction)
        return x_np.copy(), y_curve
 
    _, y_deep = offset_curve(1.0)
    body_xs = list(x_np) + list(x_np[::-1])
    body_ys = list(y_np) + list(y_deep[::-1])
    ax.fill(body_xs, body_ys, color='#bdc3c7', alpha=0.55, zorder=1,
            label='Vehicle interior')
 
    for li in range(4):
        x_out, y_out = offset_curve(t_cumul_norm[li])
        x_in,  y_in  = offset_curve(t_cumul_norm[li + 1])
        xs = list(x_out) + list(x_in[::-1])
        ys = list(y_out) + list(y_in[::-1])
        ax.fill(xs, ys, color=colors_rev[li], alpha=0.90,
                zorder=2 + li, label=labels_rev[li])
        ax.plot(x_in, y_in, 'k-', linewidth=0.8, alpha=0.5, zorder=6 + li)

    ax.plot(x_np, y_np,                             'k-', linewidth=2.5, zorder=20)
    ax.plot([x_np[-1], x_np[-1]], [0.0, y_np[-1]], 'k-', linewidth=2.5, zorder=20)
    ax.plot([x_np[0],  x_np[-1]], [0.0, 0.0],      'k-', linewidth=2.5, zorder=20)

    x_ann         = 0.70 * x_target
    y_ann_surface = float(np.interp(x_ann, x_np, y_np))
    for li in range(4):
        frac_mid = 0.5 * (t_cumul_norm[li] + t_cumul_norm[li + 1])
        y_tip    = y_ann_surface * (1.0 - frac_mid * vis_fraction)
        y_text   = y_target * (0.55 + li * 0.35)
        ax.annotate(labels_rev[li],
                    xy=(x_ann, y_tip),
                    xytext=(x_ann - 0.15, y_text),
                    fontsize=8, color=colors_rev[li], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colors_rev[li], lw=1.3),
                    zorder=22,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7, edgecolor=colors_rev[li]))

    from matplotlib.patches import Patch
    handles  = [Patch(facecolor='#bdc3c7', label='Vehicle interior', alpha=0.55)]
    handles += [Patch(facecolor=c, label=l, alpha=0.90)
                for c, l in zip(colors_rev, labels_rev)]
    ax.legend(handles=handles, fontsize=7.5, loc='upper left')

    ax.set_ylim(-y_target * 0.15, y_target * 2.8)
    ax.set_xlim(-0.05, x_target + 0.15)
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    total_t = sum(L['t'] for L in tps_layers)
    ax.set_title(
        f'TPS layer stack — inside body surface  '
        f'(visual scale: {vis_fraction*100:.0f}% of local wedge height)\n'
        f'RCC {tps_layers[3]["t"]*1000:.0f} mm → '
        f'Silica {tps_layers[2]["t"]*1000:.0f} mm → '
        f'SIP {tps_layers[1]["t"]*1000:.0f} mm → '
        f'Aluminum {tps_layers[0]["t"]*1000:.0f} mm  '
        f'(total {total_t*1000:.0f} mm real)',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — TRANSIENT T_RCC EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_transient_trcc(t_hist, T_RCC_hist,T_alum_hist, x_panel, t_ss,
                        savename='tps_transient_trcc.png'):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('TPS Transient — RCC Outer Face Temperature', fontsize=12)

    T_RCC_arr = np.array(T_RCC_hist)    # (snapshots, N)
    T_alm_arr = np.array(T_alum_hist)   # FIX: was undefined — now passed in
    t_arr     = np.array(t_hist)
    cmap      = cm.get_cmap('plasma', N)
    x_p       = np.array(x_panel)

    ax = axes[0]
    for i in range(N):
        ax.plot(t_arr, T_RCC_arr[:, i], color=cmap(i/N),
                label=f'x={float(x_panel[i]):.2f} m')
    if t_ss is not None:
        ax.axvline(t_ss, color='k', linestyle='--', linewidth=1.5,
                   label=f'SS at {t_ss:.0f} s')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('T (K)')
    ax.set_title('T_RCC(t) — all panels')
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(N):
        ax.plot(t_arr, T_alm_arr[:, i], color=cmap(i/N),
                label=f'x={float(x_panel[i]):.2f} m')
    if t_ss is not None:
        ax.axvline(t_ss, color='k', linestyle='--', linewidth=1.5,
                   label=f'SS at {t_ss:.0f} s')
    ax.axhline(Tw, color='purple', linestyle=':', linewidth=1.2,
               label=f'Initial Tw = {Tw:.0f} K')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('T (K)')
    ax.set_title('T_alum(t) — inner face (dynamic, not fixed)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    snap_indices = [0, len(T_RCC_hist)//4, len(T_RCC_hist)//2,
                    3*len(T_RCC_hist)//4, -1]
    snap_colors  = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for idx, col in zip(snap_indices, snap_colors):
        t_label = f't={t_hist[idx]:.0f}s'
        ax.plot(x_p, T_RCC_arr[idx],  '-',  color=col, linewidth=1.8,
                label=f'T_RCC {t_label}')
        ax.plot(x_p, T_alm_arr[idx], '--', color=col, linewidth=1.2,
                label=f'T_alum {t_label}')
    ax.axhline(Tw, color='purple', linestyle=':', linewidth=1.0,
               label=f'Tw₀ = {Tw:.0f} K')
    ax.set_xlabel('x (m)'); ax.set_ylabel('T (K)')
    ax.set_title('T_RCC (solid) & T_alum (dashed) spatial profiles')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — TPS HEAT FLUX BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_heat_flux(x_panel, q_cond, q_conv, q_to_alum, q_rad, q_saved,
                       T_interfaces, savename='tps_heatflux.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('TPS Heat Flux Breakdown (Final / Quasi-Steady State)', fontsize=12)
    x_p = np.array(x_panel)

    q_conv_f = np.array(q_conv)   # the ACTUAL solved flux — not a stale recompute

    ax = axes[0, 0]
    ax.plot(x_p, q_conv_f,          'o-', color='crimson',      label='q_conv (into RCC)')
    ax.plot(x_p, np.array(q_to_alum), 's-', color='steelblue', label='q_to_alum (to structure)')
    ax.plot(x_p, np.array(q_rad),    '^-', color='darkorange',  label='q_rad (to space)')
    ax.plot(x_p, np.array(q_saved),  'd-', color='green',       label='q_saved (intercepted)')
    ax.axvline(float(x_tr), color='k', linestyle=':', label=f'x_tr={float(x_tr):.2f} m')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Heat flux (W/m²)')
    ax.set_title('Heat Flux Components')

    #Savings

    ax = axes[0, 1]
    q_conv_ref=np.array(q_conv_fixed)
    saving_pct = 100.0 * np.array(q_saved) / q_conv_ref
    ax.plot(x_p, saving_pct, 'o-', color='purple')
    ax.axhline(0,   color='k',   linewidth=0.8)
    ax.axhline(100, color='red', linewidth=0.8, linestyle='--', label='100%')
    ax.axvline(float(x_tr), color='k', linestyle=':', label='x_tr')
    ax.grid(True); ax.legend(fontsize=8)
    ax.set_ylim(10,102)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Saving (%)')
    ax.set_title('TPS Heat Saving  (q_saved / q_conv)')

    # Interface temperature

    ax = axes[1, 0]

# ── Depth positions of each interface, RCC outer (z=0) -> Aluminum inner ──
    t_RCC, t_sil, t_sip, t_Al = (tps_layers[3]['t'], tps_layers[2]['t'],
                              tps_layers[1]['t'], tps_layers[0]['t'])
    z_mm = np.array([
        0.0,
        t_RCC,
        t_RCC + t_sil,
        t_RCC + t_sil + t_sip,
        t_RCC + t_sil + t_sip + t_Al,
        ]) * 1000.0   # m -> mm

# 4 material segments between the 5 interface depths, outer -> inner
    layer_names  = ['RCC', 'Silica', 'SIP', 'Aluminum']
    layer_colors = ['#c0392b', '#e67e22', '#f1c40f', '#95a5a6']   # matches plot_tps_layers

    T_interfaces_np = np.array(T_interfaces)   # (5, N)

# ── Pick 3 representative x-stations: laminar, mid-body, trailing edge ──
    x_panel_np = np.array(x_panel)
    idx_lam = int(np.argmax(x_panel_np < float(x_tr))) if np.any(x_panel_np < float(x_tr)) else 0
    panel_idx    = [idx_lam]
    panel_labels = [f'Laminar  x={x_panel_np[idx_lam]:.2f} m']
    panel_styles = ['-']    # distinguishes stations; color distinguishes layer

    for i, ls, plabel in zip(panel_idx, panel_styles, panel_labels):
        T_prof = T_interfaces_np[:, i]

        for seg in range(4):   # 4 layer segments between the 5 depth points
            ax.plot(z_mm[seg:seg+2], T_prof[seg:seg+2],
                linestyle=ls, color=layer_colors[seg],
                linewidth=2.2, marker='o', markersize=5)

    ax.axhline(Tw, color='purple', linestyle=(0, (1, 1)), linewidth=1.2,label=f'Initial Tw={Tw:.0f} K')

    for zval in z_mm:
        ax.axvline(zval, color='grey', linestyle=':', linewidth=0.5, alpha=0.4)

# ── Two separate legends: material (color) and x-station (line style) ──
   # ── Layer (material) legend ─────────────────────────────────────────────
    from matplotlib.lines import Line2D
    layer_handles = [Line2D([0], [0], color=c, lw=2.5) for c in layer_colors]
    leg1 = ax.legend(layer_handles, layer_names, fontsize=7,
                  title='Layer', loc='upper right')
    ax.add_artist(leg1)
    ax.legend(fontsize=6.5, loc='upper right',
          bbox_to_anchor=(1.0, 0.62))   # sits just below leg1

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Depth from outer RCC face (mm)')
    ax.set_ylabel('T (K)')
    ax.set_title('TPS Through-Thickness Temperature Profile\n(laminar / mid-body / trailing-edge stations)')

    # No TPS vs TPS plot

    ax = axes[1, 1]
    ax.plot(x_p, q_conv_fixed,  'o--', color='grey',   label='q_conv (no TPS, Tw=300K)')
    ax.plot(x_p, np.array(q_cond),          'o-',  color='crimson', label='q_cond (with TPS, T_RCC actual)')
    ax.axvline(float(x_tr), color='k', linestyle=':', label=f'x_tr={float(x_tr):.2f} m')
    ax.grid(True); ax.legend(fontsize=8)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Heat flux (W/m²)')
    ax.set_title('q_conv (no TPS) vs q_cond (with TPS)')

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# PLOT showing Trcc and Tal variations
def plot_transient_with_limits(t_hist, T_RCC_hist, T_alum_hist,
                               x_panel, Taw, t_ss,
                               t_fail_RCC, t_fail_alum,
                               savename='tps_transient_limits.png'):
    from mpl_toolkits.mplot3d import Axes3D

    T_RCC_arr = np.array(T_RCC_hist)            # (n_snapshots, N)
    T_alm_arr = np.array(T_alum_hist)           # (n_snapshots, N)
    t_arr     = np.array(t_hist)                # (n_snapshots,)
    x_p       = np.array(x_panel, dtype=float)  # (N,)
    Taw_np    = np.array(Taw, dtype=float)       # (N,)

    X_curve_rcc  = []
    T_curve_rcc  = []
    Z_curve_rcc  = []

    X_curve_alum = []
    T_curve_alum = []
    Z_curve_alum = []

    for k, t_k in enumerate(t_arr):
        X_curve_rcc.extend(x_p.tolist())
        T_curve_rcc.extend([t_k] * len(x_p))
        Z_curve_rcc.extend(T_RCC_arr[k].tolist())

        X_curve_alum.extend(x_p.tolist())
        T_curve_alum.extend([t_k] * len(x_p))
        Z_curve_alum.extend(T_alm_arr[k].tolist())

    X_curve_rcc  = np.array(X_curve_rcc)
    T_curve_rcc  = np.array(T_curve_rcc)
    Z_curve_rcc  = np.array(Z_curve_rcc)

    X_curve_alum = np.array(X_curve_alum)
    T_curve_alum = np.array(T_curve_alum)
    Z_curve_alum = np.array(Z_curve_alum)

    def colored_3d_line(ax, X, T, Z, cmap_name, label):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        points = np.array([X, T, Z]).T.reshape(-1, 1, 3)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
        cmap = cm.get_cmap(cmap_name)
        lc   = Line3DCollection(segs, cmap=cmap, norm=norm,
                                linewidth=1.5, alpha=0.9)
        lc.set_array(Z[:-1])
        ax.add_collection3d(lc)
        return lc

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f'TPS Temperature — Single curve T(x, t)\n'
        f'M∞ = {M_inf:.1f},  h = 20 km,  t_end = {t_arr[-1]:.0f} s',
        fontsize=13, fontweight='bold'
    )

    ax1 = fig.add_subplot(121, projection='3d')

    lc1 = colored_3d_line(ax1,
                           X_curve_rcc, T_curve_rcc, Z_curve_rcc,
                           'plasma', 'T_RCC')

    X_pl, T_pl = np.meshgrid(
        np.linspace(x_p.min(), x_p.max(), 10),
        np.linspace(t_arr.min(), t_arr.max(), 10)
    )
    Z_pl = np.full_like(X_pl, T_allow_RCC)
    ax1.plot_surface(X_pl, T_pl, Z_pl,
                     alpha=0.15, color='red', zorder=1)
    ax1.text(x_p.mean(), t_arr[-1], T_allow_RCC,
             f'T_limit = {T_allow_RCC:.0f} K',
             color='red', fontsize=9, fontweight='bold')

    ax1.plot(x_p,
             np.full_like(x_p, t_arr[-1]),
             Taw_np,
             color='cyan', linewidth=2.5, linestyle='--',
             label=f'T_aw at t_end')
    ax1.plot(x_p,
             np.full_like(x_p, t_arr[0]),
             Taw_np,
             color='cyan', linewidth=1.5, linestyle=':', alpha=0.5)

    if t_ss is not None:
        ax1.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_ss),
                 np.linspace(T_RCC_arr.min(), T_RCC_arr.max(), 10),
                 'w--', linewidth=1.5, label=f'SS t={t_ss:.0f} s')

    cb1 = fig.colorbar(lc1, ax=ax1, pad=0.1, shrink=0.6)
    cb1.set_label('T_RCC  [K]', fontsize=9)

    ax1.set_xlabel('x  [m]',    fontsize=10, labelpad=8)
    ax1.set_ylabel('Time  [s]', fontsize=10, labelpad=8)
    ax1.set_zlabel('T_RCC  [K]',fontsize=10, labelpad=8)
    ax1.set_title('RCC Outer Face', fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.view_init(elev=25, azim=-50)

    ax2 = fig.add_subplot(122, projection='3d')

    lc2 = colored_3d_line(ax2,
                           X_curve_alum, T_curve_alum, Z_curve_alum,
                           'viridis', 'T_alum')

    Z_pl_alum = np.full_like(X_pl, T_allow_alum)
    ax2.plot_surface(X_pl, T_pl, Z_pl_alum,
                     alpha=0.15, color='darkorange', zorder=1)
    ax2.text(x_p.mean(), t_arr[-1], T_allow_alum,
             f'T_limit = {T_allow_alum:.0f} K',
             color='darkorange', fontsize=9, fontweight='bold')

    if t_ss is not None:
        ax2.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_ss),
                 np.linspace(T_alm_arr.min(), T_alm_arr.max(), 10),
                 'w--', linewidth=1.5, label=f'SS t={t_ss:.0f} s')

    cb2 = fig.colorbar(lc2, ax=ax2, pad=0.1, shrink=0.6)
    cb2.set_label('T_alum  [K]', fontsize=9)

    ax2.set_xlabel('x  [m]',     fontsize=10, labelpad=8)
    ax2.set_ylabel('Time  [s]',  fontsize=10, labelpad=8)
    ax2.set_zlabel('T_alum  [K]',fontsize=10, labelpad=8)
    ax2.set_title('Aluminum Inner Face', fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()

# PLOT TO SHOW Q
def plot_heatflux_transient(t_hist, q_conv_hist, q_rad_hist, q_cond_hist,
                            savename='tps_heatflux_transient.png'):

    t_arr      = np.array(t_hist)
    q_conv_arr = np.array(q_conv_hist)   # (n_snapshots, N)
    q_rad_arr  = np.array(q_rad_hist)
    q_cond_arr = np.array(q_cond_hist)

    n = min(len(t_arr), len(q_conv_arr), len(q_rad_arr), len(q_cond_arr))
    t_arr      = t_arr[:n]
    q_conv_arr = q_conv_arr[:n]
    q_rad_arr  = q_rad_arr[:n]
    q_cond_arr = q_cond_arr[:n]

    print(f"  Plotting fluxes: n_snapshots={n}, N_panels={q_conv_arr.shape[1]}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f'Heat Flux Evolution vs Time\n'
        f'M∞ = {M_inf:.1f},  h = 20 km,  t_end = {t_arr[-1]:.0f} s',
        fontsize=13, fontweight='bold'
    )

    flux_data   = [q_conv_arr,       q_rad_arr,       q_cond_arr      ]
    flux_labels = ['q_conv  [W/m²]', 'q_rad  [W/m²]', 'q_cond  [W/m²]']
    flux_colors = ['crimson',         'darkorange',     'steelblue'     ]

    panel_configs = [
        ('Mean flux — all panels',
         lambda q: q.mean(axis=1)),
        (f'Leading edge   x = {float(x_panel[0]):.3f} m',
         lambda q: q[:, 0]),
        (f'Trailing edge  x = {float(x_panel[-1]):.3f} m',
         lambda q: q[:, -1]),
    ]

    for col, (title, slicer) in enumerate(panel_configs):
        ax = axes[col]
        for q_arr, label, color in zip(flux_data, flux_labels, flux_colors):
            y = slicer(q_arr)                      # (n,)
            assert len(y) == len(t_arr), \
                f"Length mismatch: t={len(t_arr)}, y={len(y)}"
            ax.plot(t_arr, y, color=color, linewidth=2.0, label=label)

        if t_ss is not None:
            ax.axvline(t_ss, color='k', linestyle='--',
                       linewidth=1.5, label=f'SS  t={t_ss:.0f} s')

        ax.set_xlabel('Time  [s]', fontsize=11)
        ax.set_ylabel('Flux  [W/m²]', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1b — TPS GEOMETRY, FULL VEHICLE (mirrored about centreline)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_layers_full(x, y, tps_layers, savename='tps_layers_full.png'):
    fig, ax = plt.subplots(figsize=(16, 7))
    x_np, y_np = np.array(x, dtype=float), np.array(y, dtype=float)

    layer_order  = [3, 2, 1, 0]   # RCC, Silica, SIP, Aluminum (outer -> inner)
    t_real       = np.array([tps_layers[i]['t'] for i in layer_order])
    t_total_real = t_real.sum()
    colors_rev = ['#c0392b', '#e67e22', '#f1c40f', '#95a5a6']
    labels_rev = [
        f"RCC       {tps_layers[3]['t']*1000:.0f} mm  k={tps_layers[3]['k']} W/mK",
        f"Silica    {tps_layers[2]['t']*1000:.0f} mm  k={tps_layers[2]['k']} W/mK",
        f"SIP       {tps_layers[1]['t']*1000:.0f} mm  k={tps_layers[1]['k']} W/mK",
        f"Aluminum  {tps_layers[0]['t']*1000:.0f} mm  k={tps_layers[0]['k']} W/mK",
    ]
    vis_fraction = 0.55
    t_cumul_norm = np.concatenate([[0.0], np.cumsum(t_real) / t_total_real])

    def offset_curve(frac, sign):
        return x_np.copy(), sign * y_np * (1.0 - frac * vis_fraction)

    def draw_half(sign, show_labels):
        _, y_deep = offset_curve(1.0, sign)
        ax.fill(list(x_np) + list(x_np[::-1]), list(sign*y_np) + list(y_deep[::-1]),
                color='#bdc3c7', alpha=0.55, zorder=1,
                label='Vehicle interior' if show_labels else None)
        for li in range(4):
            x_out, y_out = offset_curve(t_cumul_norm[li], sign)
            x_in,  y_in  = offset_curve(t_cumul_norm[li+1], sign)
            ax.fill(list(x_out) + list(x_in[::-1]), list(y_out) + list(y_in[::-1]),
                    color=colors_rev[li], alpha=0.90, zorder=2+li,
                    label=(labels_rev[li] if show_labels else None))
            ax.plot(x_in, y_in, 'k-', linewidth=0.7, alpha=0.5, zorder=8+li)
        ax.plot(x_np, sign*y_np, 'k-', linewidth=2.5, zorder=20)
        ax.plot([x_np[-1], x_np[-1]], [0.0, sign*y_np[-1]], 'k-', linewidth=2.5, zorder=20)

    draw_half(+1.0, show_labels=True)
    draw_half(-1.0, show_labels=False)

    ax.plot([x_np[-1], x_np[-1]], [-y_np[-1], y_np[-1]], 'k-', linewidth=2.5, zorder=20)
    ax.axhline(0.0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=21)

    ax.legend(fontsize=7.5, loc='upper left', ncol=1)
    ax.set_ylim(-y_target * 1.8, y_target * 3.2)
    ax.set_xlim(-0.05, x_target + 0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    total_t = sum(L['t'] for L in tps_layers)
    ax.set_title(
        f'TPS stack — full vehicle, symmetric — RCC outer -> Aluminum inner '
        f'(total {total_t*1000:.0f} mm)\n(visual scale: {vis_fraction*100:.0f}% of local half-height)',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


plot_tps_layers(x, y, theta, tps_layers)
plot_tps_layers_full(x, y, tps_layers)
plot_transient_with_limits(
    t_hist, T_RCC_hist, T_alum_hist,
    x_panel, Taw, t_ss,
    t_fail_RCC, t_fail_alum
)
plot_heatflux_transient(
    t_hist, q_conv_hist, q_rad_hist, q_cond_hist)
plot_transient_trcc(t_hist, T_RCC_hist, T_alum_hist, x_panel, t_ss)

plot_tps_heat_flux(x_panel, q_cond, q_conv, q_to_alum, q_rad, q_saved, T_interfaces)

print("\nAll plots saved.")