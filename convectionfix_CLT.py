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

M_inf = 5.0
gamma = 1.4

# Atmosphere at h = 20 km (US Standard Atmosphere 1976)
T_inf   = 205.0             # K
p_inf   = 5543.0            # Pa
rho_inf = 0.09427           # kg/m³
Tw      = 300.0             # K — initial structural wall temperature
Cp      = 1005.0            # J/(kg·K) — air specific heat

R  = 287.0                  # J/(kg·K)
Pr = 0.71                   # Prandtl number for air

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf
mu_inf = 1.458e-6 * T_inf**1.5 / (T_inf + 110.4)

sigma = 5.67e-8             # Stefan–Boltzmann W/(m²·K⁴)

# ── Material temperature limits ───────────────────────────────────────────────
T_allow_UHTC = 2200.0          # K — ZrB2/SiC operational limit
T_allow_CMC  = 1900.0          # K — C/SiC structural limit
T_allow_Ti   = 400.0 + 273.15  # K — Ti-6Al-4V (~400 °C)

# ── Nose bluntness — drives BOTH the aeroshell geometry AND the Fay-Riddell
#    stagnation-point heat flux (same physical radius used in both places,
#    so the blunt-body geometry and the blunt-body heating model stay consistent) ──
R_nose = 0.01   # m — leading-edge/nose radius. Increase for a blunter aeroshell,
                 # decrease toward 0 to recover the old sharp-wedge limit.



# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY — spherically-blunted wedge ("aeroshell")
#
#  The vehicle is built from two pieces that are slope-matched (C1 continuous)
#  at the join:
#    1) a circular-arc nose cap of radius R_nose, tangent to the y=0
#       centreline at the stagnation point (0,0), sweeping from a
#       flow-normal tip up to the tangent angle theta_w
#    2) a straight afterbody ramp at constant angle theta_w running from
#       the tangent point out to the target trailing-edge point
#       (x_target, y_target)
#
#  theta_w is solved (Newton iteration) so the straight segment lands
#  exactly on (x_target, y_target) despite the small offset the nose
#  radius introduces. This is the standard construction for a
#  spherically-blunted cone/wedge hypersonic vehicle.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nodes(theta):
    """Legacy: build node coordinates (length N+1) from N constant-length,
    piecewise-linear panel angles (rad). Kept for reference; superseded by
    generate_aeroshell_nodes() below, which is what the vehicle now uses."""
    dx = panel_length * jnp.cos(theta)
    dy = panel_length * jnp.sin(theta)
    x  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dx)])
    y  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dy)])
    return x, y


def solve_wedge_angle(x_target, y_target, Rn, n_iter=60):
    """
    Newton-solve for the afterbody half-angle theta_w such that a circular
    nose cap of radius Rn (tangent to the y=0 centreline at the stagnation
    point) blends smoothly (matched slope) into a straight afterbody ramp
    that reaches exactly (x_target, y_target).

    Tangent point on the nose circle (parametrized by cap sweep angle phi,
    with local surface slope = 90deg - phi):
        x_t(theta_w) = Rn * (1 - sin(theta_w))
        y_t(theta_w) = Rn * cos(theta_w)

    Solve: (y_target - y_t) - tan(theta_w) * (x_target - x_t) = 0
    """
    theta = float(np.arctan(y_target / x_target))   # sharp-wedge angle, initial guess
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
    """
    Build a spherically-blunted wedge ("aeroshell") of N panels:
      - the first ~n_nose_frac*N panels trace the circular nose cap
        (local surface angle sweeps from ~90 deg at the stagnation point
        down to theta_w at the nose/afterbody tangent point)
      - the remaining panels are a straight afterbody ramp at constant
        theta_w out to (x_target, y_target)

    Returns
    -------
    x, y        : (N+1,) node coordinates [m]
    theta       : (N,) panel angles [rad] (used directly by Newtonian Cp)
    x_t, y_t    : nose/afterbody tangent point [m]
    theta_w     : afterbody half-angle [rad]
    n_nose      : number of panels on the nose cap
    """
    n_nose  = max(2, int(round(N * n_nose_frac)))
    n_after = N - n_nose
    if n_after < 1:
        n_after = 1
        n_nose  = N - 1

    theta_w = solve_wedge_angle(x_target, y_target, R_nose)
    phi_t   = jnp.pi / 2.0 - theta_w   # nose-cap sweep angle at the tangent point

    # ── Nose cap: circular arc, phi in [0, phi_t] ─────────────────────────────
    phi_nodes = jnp.linspace(0.0, phi_t, n_nose + 1)
    x_nose = R_nose * (1.0 - jnp.cos(phi_nodes))
    y_nose = R_nose * jnp.sin(phi_nodes)

    x_t, y_t = x_nose[-1], y_nose[-1]

    # ── Afterbody: straight ramp at theta_w to (x_target, y_target) ──────────
    s = jnp.linspace(0.0, 1.0, n_after + 1)[1:]     # exclude 0 (shared w/ nose end)
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
    """Modified Newtonian Cp — leeward panels clamped to zero."""
    return jnp.maximum(2.0 * jnp.sin(theta)**2, 0.0)

def pressure_distribution(theta):
    q_inf = 0.5 * rho_inf * V_inf**2
    return p_inf + q_inf * newtonian_cp(theta)

def isentropic_temperature(p):
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    T0 = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
    return T0 * (p / p0)**((gamma-1.0)/gamma)

def density_from_pt(p, T):
    return p / (R * T)

def local_mach(p):
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    return jnp.sqrt((2.0/(gamma-1.0)) * ((p0/p)**((gamma-1.0)/gamma) - 1.0))

def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDARY LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def adiabatic_wall_temperature(p, T):
    M_e    = local_mach(p)
    r_lam  = Pr**0.5
    r_turb = Pr**(1.0/3.0)
    Taw_lam  = T * (1.0 + r_lam  * (gamma-1.0)/2.0 * M_e**2)
    Taw_turb = T * (1.0 + r_turb * (gamma-1.0)/2.0 * M_e**2)
    return Taw_lam, Taw_turb

def transition_location(x_safe, p, T):
    """Rotta-ARA criterion: Re_theta / M_local >= 400."""
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)

    Rex      = rho_e * V_e * x_safe / mu_e
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)
    Re_theta = rho_e * V_e * theta_m / mu_e
    criterion = Re_theta / M_e

    has_transition = jnp.any(criterion >= 400.0)
    idx            = jnp.argmax(criterion >= 400.0)
    x_tr           = jnp.where(has_transition, x_safe[idx], x_safe[-1])
    return x_tr, Re_theta


def boundary_layer(x, p, T):
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)

    x_safe = jnp.maximum(x, 1e-6)
    Rex    = rho_e * V_e * x_safe / mu_e

    x_tr, Re_theta = transition_location(x_safe, p, T)

    Taw_lam, Taw_turb = adiabatic_wall_temperature(p, T)
    Taw = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    comp_factor = 1.0 + 0.016*M_e**2 + 0.072*(Tw/Taw)*M_e**2
    delta_lam   = (5.0 * x_safe / jnp.sqrt(Rex)
                   * (Tw / T_inf)**(-1.0/6.0)
                   * comp_factor)

    x_from_tr  = jnp.maximum(x_safe - x_tr, 1e-6)
    Rex_tr     = rho_e * V_e * x_from_tr / mu_e
    delta_turb = 0.37 * x_from_tr / Rex_tr**0.2 * (Taw / T_inf)**0.6

    delta = jnp.where(x_safe <= x_tr, delta_lam, delta_turb)
    return delta, delta_lam, delta_turb, x_tr, Re_theta, Taw


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL ANGLES & EVALUATION — runs ONCE, top to bottom. Nothing below this
#  point redefines theta/x/y/x_panel/p/T/Taw/q_wall/St_ref/rho_ref/x_tr,
#  so every downstream consumer is guaranteed to see consistent arrays.
# ═══════════════════════════════════════════════════════════════════════════════
x, y, theta, x_nose_end, y_nose_end, theta_w, n_nose_panels = generate_aeroshell_nodes(
    N, x_target, y_target, R_nose
)
x_panel  = x[1:]

print(f"\n  [Aeroshell geometry] R_nose = {float(R_nose)*1000:.1f} mm   "
      f"theta_w (afterbody) = {float(jnp.rad2deg(theta_w)):.2f} deg   "
      f"nose panels = {n_nose_panels}/{N}   "
      f"nose/afterbody tangent at x = {float(x_nose_end):.4f} m, y = {float(y_nose_end):.4f} m")

p     = pressure_distribution(theta)
T     = isentropic_temperature(p)
T_e=T
rho_e = density_from_pt(p, T)

delta, delta_lam, delta_turb, x_tr, Re_theta, Taw = boundary_layer(x_panel, p, T)


dp_dx = jnp.gradient(p, x_panel)
dT_dx = jnp.gradient(T, x_panel)


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PRINTOUT — aero / BL
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  GEOMETRY CHECK")
print("=" * 55)
print(f"  Target endpoint : ({x_target:.3f}, {y_target:.3f}) m")
print(f"  Actual endpoint : ({float(x[-1]):.3f}, {float(y[-1]):.3f}) m")

print("\n" + "=" * 55)
print("  FREESTREAM / STAGNATION")
print("=" * 55)
q_inf = 0.5 * rho_inf * V_inf**2
print(f"  a_inf = {float(a_inf):.2f} m/s    V_inf = {float(V_inf):.2f} m/s")
print(f"  q_inf = {float(q_inf):.2f} Pa")



# ═══════════════════════════════════════════════════════════════════════════════
#  TPS LAYER DEFINITIONS — outer to inner (single material system)
#
#  Layer 1 (outer shell) is SPATIALLY VARYING:
#    UHTC  (ZrB2 + SiC backing)   for x_panel <  x_tr   (leading edge)
#    CMC   (C/SiC)                for x_panel >= x_tr   (main body)
#  Layers 2-5 are uniform along the whole vehicle.
# ═══════════════════════════════════════════════════════════════════════════════
mat_UHTC = {'name': 'UHTC (ZrB2/SiC)', 't': 0.005, 'k': 25.0, 'rho': 6100.0, 'Cp': 420.0,  'eps': 0.85}
mat_CMC  = {'name': 'CMC (C/SiC)',     't': 0.005, 'k': 11.5, 'rho': 2200.0, 'Cp': 800.0,  'eps': 0.85}

tps_layers_fixed = [
    {'name': 'Thermal Buffer (AETB)',     't': 0.0175, 'k': 0.10, 'rho': 300.0, 'Cp': 1000.0, 'eps': 0.80}, #0
    {'name': 'Aerogel', 't': 0.0250, 'k': 0.03, 'rho': 120.0, 'Cp': 1100.0, 'eps': 0.80}, #1
    {'name': 'Silicone Dampener',         't': 0.0010, 'k': 0.45, 'rho': 1100.0, 'Cp': 1400.0, 'eps': 0.90}, #2
    {'name': 'Titanium Alloy (Ti-6Al-4V)',      't': 0.0035, 'k': 13.0, 'rho': 4430.0, 'Cp': 560.0,  'eps': 0.20}, #3
]

# ═══════════════════════════════════════════════════════════════════════════════
#  THERMAL EXPANSION PROPERTIES
#  alpha  : linear thermal expansion coefficient  [1/K]
#  beta   : volumetric expansion coefficient = 3 * alpha  [1/K]  (isotropic)
#  E      : Young's modulus  [Pa]
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  THERMAL EXPANSION PROPERTIES — DIRECTIONAL (transversely isotropic)
#
#  alpha_par  : in-plane (parallel to surface)        [1/K]
#  alpha_perp : through-thickness (perpendicular)     [1/K]
#  beta       : volumetric = alpha_par + alpha_par + alpha_perp = 2*alpha_par + alpha_perp
#  E_par      : Young's modulus in-plane              [Pa]
#  E_perp     : Young's modulus through-thickness      [Pa]
#
#  Sources:
#   UHTC  — Fahrenholtz & Hilmas, IJRMHM 2017
#   CMC   — Naslain, Comp. Sci. Tech. 2004
#   AETB  — NASA TM-2000-210256
#   Aerogel — Fricke & Emmerling, J. Sol-Gel Sci. 1992
#   Silicone — Dow Corning datasheet (isotropic polymer)
#   Ti-6Al-4V — MIL-HDBK-5J (polycrystalline, near-isotropic)
# ═══════════════════════════════════════════════════════════════════════════════
thermo_mech = {
    'UHTC': {
        # ZrB2 hexagonal: a-axis ~6.7, c-axis ~6.3 µ/K
        # SiC backing stiffens through-thickness
        'alpha_par' : 6.7e-6,   # in-plane      [1/K]
        'alpha_perp': 6.3e-6,   # through-thickness [1/K]
        'E_par'     : 350e9,    # in-plane      [Pa]
        'E_perp'    : 300e9,    # through-thickness [Pa]
        'note'      : 'Hexagonal ZrB2 — mild anisotropy, a vs c axis'
    },
    'CMC': {
        # Carbon fibres have NEGATIVE alpha along fibre axis (~-1 µ/K)
        # Through-thickness dominated by SiC matrix (~4.5 µ/K)
        'alpha_par' : 2.5e-6,   # in-plane (fibre direction dominates) [1/K]
        'alpha_perp': 4.5e-6,   # through-thickness (matrix dominated) [1/K]
        'E_par'     : 70e9,     # in-plane      [Pa]
        'E_perp'    : 40e9,     # through-thickness [Pa]
        'note'      : 'C/SiC — strong anisotropy, fibre vs matrix direction'
    },
    'Thermal Buffer': {
        # Porous ceramic tile — nearly isotropic due to random fibre orientation
        # but lower stiffness through-thickness due to porosity
        'alpha_par' : 5.0e-6,   # [1/K]
        'alpha_perp': 5.5e-6,   # slightly higher through-thickness [1/K]
        'E_par'     : 0.50e9,   # [Pa]
        'E_perp'    : 0.35e9,   # softer through-thickness [Pa]
        'note'      : 'Porous ceramic — near-isotropic, mild porosity effect'
    },
    'Aerogel': {
        # Amorphous silica network — isotropic structure
        # beta = 3*alpha is valid here
        'alpha_par' : 3.0e-6,   # [1/K]
        'alpha_perp': 3.0e-6,   # isotropic [1/K]
        'E_par'     : 0.10e9,   # [Pa]
        'E_perp'    : 0.10e9,   # [Pa]
        'note'      : 'Amorphous silica — isotropic, beta=3alpha valid'
    },
    'Silicone Dampener': {
        # Polymer — isotropic, beta = 3*alpha valid
        'alpha_par' : 250e-6,   # [1/K]
        'alpha_perp': 250e-6,   # isotropic [1/K]
        'E_par'     : 0.002e9,  # [Pa]
        'E_perp'    : 0.002e9,  # [Pa]
        'note'      : 'Silicone rubber — isotropic polymer, beta=3alpha valid'
    },
    'Titanium Alloy': {
        # HCP single crystal is anisotropic but polycrystalline averaging
        # makes bulk Ti-6Al-4V effectively isotropic
        'alpha_par' : 8.6e-6,   # [1/K]
        'alpha_perp': 8.6e-6,   # near-isotropic [1/K]
        'E_par'     : 114e9,    # [Pa]
        'E_perp'    : 114e9,    # [Pa]
        'note'      : 'Polycrystalline HCP — texture averaging → near-isotropic'
    },
}
# calculating outer shell properties since it is mixed
def build_outer_shell_arrays(x_panel, x_tr):
    is_uhtc   = np.array(x_panel) < float(x_tr)
    t_outer   = np.where(is_uhtc, mat_UHTC['t'],   mat_CMC['t'])
    k_outer   = np.where(is_uhtc, mat_UHTC['k'],   mat_CMC['k'])
    rho_outer = np.where(is_uhtc, mat_UHTC['rho'], mat_CMC['rho'])
    Cp_outer  = np.where(is_uhtc, mat_UHTC['Cp'],  mat_CMC['Cp'])
    eps_outer = np.where(is_uhtc, mat_UHTC['eps'], mat_CMC['eps'])
    return is_uhtc, t_outer, k_outer, rho_outer, Cp_outer, eps_outer

is_uhtc, t_outer, k_outer, rho_outer, Cp_outer, eps_outer = build_outer_shell_arrays(x_panel, x_tr)

R_fixed     = sum(L['t'] / L['k'] for L in tps_layers_fixed)   # scalar, m²K/W
R_outer_arr = t_outer / k_outer   # resistance of the shell                      # (N,) per-panel
R_total_arr =R_outer_arr + (tps_layers_fixed[0]['t']/tps_layers_fixed[0]['k'])+(tps_layers_fixed[2]['t']/tps_layers_fixed[2]['k'])+(tps_layers_fixed[1]['t']/tps_layers_fixed[1]['k'])          # (N,) per-panel
# Taking the shell as CV this is the rest of layers minus titanium
 
# shell meaning the outer surface made out of UHTC and CMC 

# for when we take titanium as CV we readd the shell


m_shell  = rho_outer * t_outer  # mass per unit length
Cp_shell = Cp_outer
m_Ti     = tps_layers_fixed[3]['rho'] * tps_layers_fixed[3]['t']
Cp_Ti    = tps_layers_fixed[3]['Cp']

T_allow_shell = np.where(is_uhtc, T_allow_UHTC, T_allow_CMC)

print("\n" + "=" * 70)
print("  TPS THERMAL RESISTANCE STACK  (outer shell varies along body)")
print("=" * 70)
for L in tps_layers_fixed:
    R_l = L['t'] / L['k']
    print(f"    {L['name']:<28} t={L['t']*1000:6.1f} mm  k={L['k']:6.2f}  R={R_l:.5f} m²K/W")
print(f"    {'TOTAL FIXED':<28} {'':6}            R_fixed={R_fixed:.5f} m²K/W")
print(f"\n  Outer shell: UHTC for x < x_tr={float(x_tr):.3f} m, CMC for x >= x_tr")
print(f"  {'x':>6}  {'shell':>6}  {'t(mm)':>7}  {'k':>7}  {'R_outer':>9}  {'R_total':>9}")
for i in range(N):
    shell = 'UHTC' if is_uhtc[i] else 'CMC'
    print(f"  {float(x_panel[i]):6.3f}  {shell:>6}  {t_outer[i]*1000:7.2f}  "
          f"{k_outer[i]:7.2f}  {R_outer_arr[i]:9.5f}  {R_total_arr[i]:9.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TPS TRANSIENT SOLVER  —  ONE function, ONE call, correct argument order
#
#  (1) Outer shell (UHTC or CMC):
#        m_shell * Cp_shell * dT_shell/dt = q_conv - q_cond - q_rad
#  (2) Titanium structure (adiabatic beyond it):
#        m_Ti * Cp_Ti * dT_struct/dt = q_cond
#
#  q_conv = h_BL * (Taw - T_shell)              [from boundary layer]
#  q_cond = (T_shell - T_struct) / R_total_arr  [conducts through full stack]
#  q_rad  = eps_shell * sigma * (T_shell^4 - T_inf^4)
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERSONIC HEAT TRANSFER — THREE REGION MODEL
#
#  Region 1 : Fay-Riddell stagnation (x = 0, first panel only)
#  Region 2 : Eckert reference temperature, laminar  (x < x_tr)
#  Region 3 : Van Driest II compressible turbulent   (x >= x_tr)
#
#  Driving potential: Cp*(Taw - Tw) throughout — valid for calorically
#  perfect gas at M=5 where dissociation is negligible at wall temperatures
#  below ~1500 K. Enthalpy form would be identical here.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Nose radius — defined once in CONFIGURATION above; reused here so the
#    Fay-Riddell stagnation model is consistent with the actual blunted
#    nose geometry (same R_nose drives both) ──────────────────────────────────

def fay_riddell(p_stag, T_stag, T_w, R_nose):
    """
    Fay-Riddell stagnation point heat flux [W/m²].

    Valid for:
      - Dissociating and non-dissociating air
      - Blunt body stagnation point
      - Replaces flat-plate St which gives q -> 0 as x -> 0 (wrong)

    Formula (non-dissociating limit, Lewis number ~ 1):

        q_stag = 0.763 * Pr^(-0.6)
                 * (rho_w * mu_w)^0.1
                 * (rho_e * mu_e)^0.4
                 * sqrt(du_e/dx)
                 * Cp * (T_aw - T_w)

    Velocity gradient at stagnation (Newtonian):

        du_e/dx = (1/R_nose) * sqrt(2*(p_stag - p_inf)/rho_inf)

    Parameters
    ----------
    p_stag  : stagnation pressure [Pa]
    T_stag  : stagnation temperature [K]
    T_w     : current wall temperature [K]
    R_nose  : nose radius [m]

    Returns
    -------
    q_stag  : stagnation heat flux [W/m²]
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

    Transforms compressible BL to equivalent incompressible via:
      Cf_comp = Cf_incomp / Fc

    where Fc is the Van Driest compressibility correction.

    Incompressible Cf from Karman-Schlichting (implicit log-law),
    solved by Newton iteration — more accurate than 1/7 power law
    especially at high Re.

    Parameters
    ----------
    Re_x_e : Reynolds number at x, based on edge conditions
    M_e    : local edge Mach number
    T_e    : local edge static temperature [K]
    T_w    : current wall temperature [K]
    T_aw   : adiabatic wall temperature [K]

    Returns
    -------
    Cf_comp : compressible skin friction coefficient
    Fc      : compressibility correction factor (diagnostic)
    Cf_inc  : incompressible skin friction (diagnostic)
    """
    r_turb = Pr**(1.0 / 3.0)

    T_ratio_aw = 1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2 # this is equal to T_aw/T_e for r_turb
    T_ratio_w  = T_w  / T_e

    # ── VD2 transformation parameters ────────────────────────────────────────
    # A: compressibility parameter
    # B: net wall temperature effect
    A_sq = jnp.maximum(
        (r_turb * (gamma - 1.0) / 2.0 * M_e**2)
        / (T_ratio_w + 1e-10),
        0.0
    )
    A = jnp.sqrt(A_sq)
    B = T_ratio_aw/(T_ratio_w+1e-10)-1  
    # I have fixed A and B because they were wrong and the +1e-10 is to ensure its not divided by 0 or smthg

    # ── Compressibility correction factor Fc ──────────────────────────────────
    denom = jnp.sqrt(B**2 + 4.0 * A**2 + 1e-30)

    alpha = jnp.clip((2.0 * A**2 - B) / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    beta  = jnp.clip(B / denom,  -1.0 + 1e-7, 1.0 - 1e-7)

    arcsin_sum = jnp.arcsin(alpha) + jnp.arcsin(beta)

    # Fc: ratio of adiabatic temperature rise to VD2 angle sum squared
    # Fc -> 1 as M_e -> 0 (incompressible limit)
    M_e = local_mach(p)
    Fc = jnp.where(
        T_ratio_aw > 1.001,
        (T_ratio_aw - 1.0) / (arcsin_sum**2 + 1e-30),
        1.0
    )
    omega=0.76 # for air
    Fx=(1/T_ratio_w)**omega
    Re_x_inc=Fx*Re_x_e


    # ── Incompressible Cf: Karman-Schlichting implicit formula ────────────────
    # 1/sqrt(Cf) = 4.15 * log10(Re_x * Cf) + 1.7
    Cf_inc = 0.0592 * Re_x_inc**(-0.2)    # initial guess
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

    Region 1 — Fay-Riddell  : first panel (stagnation), x ~ 0
    Region 2 — Eckert lam   : x < x_tr, laminar flat plate
    Region 3 — Van Driest II : x >= x_tr, compressible turbulent

    Driving potential: Cp*(Taw - Tw) — calorically perfect gas,
    valid at M=5 where wall T rarely exceeds dissociation onset.

    Parameters
    ----------
    x_safe      : (N,) panel positions [m], clipped > 1e-6
    p           : (N,) local pressure [Pa]
    T_e         : (N,) local edge temperature [K]
    T_w_current : (N,) current shell wall temperature [K]
    x_tr        : scalar, transition location [m]

    Returns
    -------
    St     : (N,) effective Stanton number (diagnostic)
    q_conv : (N,) convective heat flux [W/m²]
    rho_ref: (N,) reference density [kg/m³]
    Taw    : (N,) adiabatic wall temperature [K]
    V_e    : (N,) edge velocity [m/s]
    """
    r_lam  = Pr**0.5
    r_turb = Pr**(1.0 / 3.0)

    M_e = local_mach(p)
    a_e = jnp.sqrt(gamma * R * T_e)
    V_e = M_e * a_e

    rho_e = density_from_pt(p, T_e)
    mu_e  = sutherland(T_e)

    # ── Adiabatic wall temperature ────────────────────────────────────────────
    Taw_lam  = T_e * (1.0 + r_lam  * (gamma - 1.0) / 2.0 * M_e**2)
    Taw_turb = T_e * (1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2)
    Taw      = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    # ═════════════════════════════════════════════════════════════════════════
    # REGION 2 — LAMINAR  (Eckert reference temperature)
    # Identical to your original laminar model — well validated
    # ═════════════════════════════════════════════════════════════════════════
    T_ref_lam   = T_e * (0.45 + 0.55 * (T_w_current / T_e)
                         + 0.16 * r_lam * (gamma - 1.0) / 2.0 * M_e**2)
    rho_ref_lam = p / (R * T_ref_lam)
    mu_ref_lam  = sutherland(T_ref_lam)
    Rex_lam     = rho_ref_lam * V_e * x_safe / mu_ref_lam

    St_lam      = (0.332 / jnp.sqrt(Rex_lam)) * Pr**(-2.0 / 3.0)
    q_lam       = rho_ref_lam * V_e * St_lam * Cp * (Taw_lam - T_w_current)

    # ═════════════════════════════════════════════════════════════════════════
    # REGION 3 — TURBULENT  (Van Driest II)
    # Re_x based on edge conditions — VD2 is edge-referenced
    # ═════════════════════════════════════════════════════════════════════════
    Re_x_e = rho_e * V_e * x_safe / mu_e

    Cf_comp, Fc, Cf_inc = van_driest_II_Cf(
        Re_x_e, M_e, T_e, T_w_current, Taw_turb
    )

    # Reynolds analogy: St = (Cf/2) * Pr^(-2/3)
    # Valid with VD2-corrected Cf (White 2006, Ch.7)
    St_turb     = (Cf_comp / 2.0) * Pr**(-2.0 / 3.0)
    q_turb      = rho_e * V_e * St_turb * Cp * (Taw_turb - T_w_current)

    # ── Blend regions 2 and 3 ────────────────────────────────────────────────
    is_lam  = x_safe < x_tr
    St      = jnp.where(is_lam, St_lam,        St_turb)
    q_conv  = jnp.where(is_lam, q_lam,         q_turb)
    rho_ref = jnp.where(is_lam, rho_ref_lam,   rho_e)

    # ═════════════════════════════════════════════════════════════════════════
    # REGION 1 — STAGNATION  (Fay-Riddell, first panel only)
    # Overwrites the laminar value at panel 0 — flat plate St is singular
    # at x=0 and gives physically wrong (too high) values there.
    # Fay-Riddell is the accepted standard for blunt body stagnation.
    # ═════════════════════════════════════════════════════════════════════════
    p_stag  = p_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)**(gamma / (gamma - 1.0))
    T_stag  = T_inf * (1.0 + 0.5 * (gamma - 1.0) * M_inf**2)

    q_stag  = fay_riddell(p_stag, T_stag, T_w_current[0], R_nose)
    

    # Replace first panel with Fay-Riddell value
    # jnp.where over index: build mask [True, False, False, ...]
    stag_mask = jnp.arange(len(x_safe)) == 0
    q_conv    = jnp.where(stag_mask, q_stag,  q_conv)
    St        = jnp.where(stag_mask,
                          q_stag / jnp.maximum(
                              rho_ref[0] * V_e[0] * Cp
                              * jnp.maximum(Taw[0] - T_w_current[0], 1.0),
                              1e-10),
                          St)

    # Clamp: no negative heat flux (wall hotter than Taw -> radiation handles it)
    q_conv = jnp.maximum(q_conv, 0.0)

    return St, q_conv, rho_ref, Taw, V_e


# ═══════════════════════════════════════════════════════════════════════════════
#  UPDATED TRANSIENT SOLVER — q_conv recomputed every timestep
#
#  This is REQUIRED because T_w_current appears in:
#    - Cp*(Taw - Tw)          driving potential
#    - VD2 T_w/T_e ratio      changes Fc and therefore Cf_comp
#    - Fay-Riddell (rho_w, mu_w, Cp*(T_aw_stag - T_w))
#
#  Precomputing h_BL once (as in the original) would freeze these
#  dependencies at Tw=300K and give wrong q_conv as shell heats up.
# ═══════════════════════════════════════════════════════════════════════════════
def tps_transient_3region(x_safe, p, T_e, x_tr,
                          t_end=600.0, dt_init=0.1,
                          tol=1e-3, save_every=200):

    x_safe_np = np.array(x_safe)
    p_np      = np.array(p)
    T_e_np    = np.array(T_e)

    T_shell  = np.full(N, float(Tw), dtype=np.float64)
    T_struct = np.full(N, float(Tw), dtype=np.float64)

    # ── Compute t=0 fluxes FIRST so they exist before appending ──────────────
    T_w_jnp = jnp.array(T_shell)
    _, q_conv_j, _, _, _ = heat_transfer_3region(
        jnp.array(x_safe_np),
        jnp.array(p_np),
        jnp.array(T_e_np),
        T_w_jnp,
        x_tr
    )
    q_conv = np.array(q_conv_j)                                      # now defined
    q_cond = (T_shell - T_struct) / np.array(R_total_arr)         # now defined        # now defined
    q_rad  = eps_outer * sigma * (T_shell**4 - T_inf**4)               # radiative loss to freestream

    # ── Now safe to initialise history lists with t=0 values ─────────────────
    t_hist        = [0.0]
    T_shell_hist  = [T_shell.copy()]
    T_struct_hist = [T_struct.copy()]
    q_conv_hist   = [q_conv.copy()]    # ✅ q_conv exists now
    q_rad_hist    = [q_rad.copy()]     # ✅ q_rad exists now
    q_cond_hist   = [q_cond.copy()]    # ✅ q_cond exists now

    t, step, t_ss = 0.0, 0, None
    t_fail_shell  = None
    t_fail_Ti     = None

    while t < t_end:

        # ── Recompute q_conv with current T_shell ─────────────────────────────
        T_w_jnp = jnp.array(T_shell)
        _, q_conv_j, _, Taw_j, _ = heat_transfer_3region(
            jnp.array(x_safe_np),
            jnp.array(p_np),
            jnp.array(T_e_np),
            T_w_jnp,
            x_tr
        )
        q_conv = np.array(q_conv_j)

        # ── Conduction and radiation ───────────────────────────────────────────
        q_cond = (T_shell - T_struct) / np.array(R_total_arr)
        q_rad  = eps_outer * sigma * (T_shell**4 - T_inf**4)

        # ── Energy balance ─────────────────────────────────────────────────────
        dT_shell_dt  = (q_conv - q_cond - q_rad) / (m_shell * Cp_shell)
        dT_struct_dt = q_cond / (m_Ti * Cp_Ti)

        # ── Adaptive timestep ──────────────────────────────────────────────────
        max_rate = max(
            np.abs(dT_shell_dt).max(),
            np.abs(dT_struct_dt).max(),
            1e-12
        )
        dt = min(dt_init, 2.0 / max_rate, t_end - t)

        T_shell  += dt * dT_shell_dt
        T_struct += dt * dT_struct_dt
        t        += dt
        step     += 1

        # ── Material limit checks ──────────────────────────────────────────────
        if t_fail_shell is None and np.any(T_shell > T_allow_shell):
            t_fail_shell  = t
            failed_panels = np.where(T_shell > T_allow_shell)[0]
            print(f"  [!] Shell limit exceeded at t={t:.1f}s  "
                  f"panels={failed_panels}  T_max={T_shell.max():.1f}K")

        if t_fail_Ti is None and np.any(T_struct > T_allow_Ti):
            t_fail_Ti = t
            print(f"  [!] Titanium limit exceeded at t={t:.1f}s  "
                  f"T_max={T_struct.max():.1f}K")

        # ── Save snapshots ─────────────────────────────────────────────────────
        if step % save_every == 0:
            t_hist.append(t)
            T_shell_hist.append(T_shell.copy())
            T_struct_hist.append(T_struct.copy())
            q_conv_hist.append(q_conv.copy())
            q_cond_hist.append(q_cond.copy())
            q_rad_hist.append(q_rad.copy())

        # ── Steady state check ─────────────────────────────────────────────────
        if np.abs(dT_shell_dt).max() < tol and t_ss is None:
            t_ss = t
            print(f"  [Transient] Shell quasi-steady at t={t:.1f}s  "
                  f"T_shell_max={T_shell.max():.1f}K  "
                  f"T_struct_max={T_struct.max():.1f}K  "
                  f"(Ti timescale >> {t_end:.0f}s)")

    # ── Final state ────────────────────────────────────────────────────────────
    t_hist.append(t)
    T_shell_hist.append(T_shell.copy())
    T_struct_hist.append(T_struct.copy())

    # Final flux recompute
    T_w_jnp = jnp.array(T_shell)
    _, q_conv_j, _, Taw_j, _ = heat_transfer_3region(
        jnp.array(x_safe_np), jnp.array(p_np),
        jnp.array(T_e_np), T_w_jnp, x_tr
    )

    T_shell_j  = jnp.array(T_shell)
    T_struct_j = jnp.array(T_struct)
    q_conv_f   = q_conv_j
    q_cond_f   = (T_shell_j - T_struct_j) / jnp.array(R_total_arr)
    q_rad_f    = jnp.array(eps_outer) * sigma * (T_shell_j**4 - T_inf**4)
    q_saved_f  = jnp.array(q_conv_fixed) - q_cond_f

    # Interface temperatures
    T_shell_in = T_shell_j  - q_cond_f * jnp.array(R_outer_arr)
    T_buf_in   = T_shell_in - q_cond_f * (tps_layers_fixed[0]['t']
                                           / tps_layers_fixed[0]['k'])
    T_gel_in   = T_buf_in   - q_cond_f * (tps_layers_fixed[1]['t']
                                           / tps_layers_fixed[1]['k'])
    T_sil_in   = T_gel_in   - q_cond_f * (tps_layers_fixed[2]['t']
                                           / tps_layers_fixed[2]['k'])
    T_struct_in=T_sil_in

    T_iface_final = jnp.stack(
        [T_shell_j, T_shell_in, T_buf_in, T_gel_in, T_sil_in, T_struct_j], axis=0
        #                                                       ^^^^^^^^^^
        #                                    index 5 = actual ODE-solved Ti temperature
        #                                    NOT T_sil_in — fixes the T_interfaces[5] bug
    )

    return (t_hist, T_shell_hist, T_struct_hist, T_iface_final, q_cond_f, q_conv_f, q_rad_f, q_saved_f,
            t_ss, t_fail_shell, t_fail_Ti,
            np.array(q_conv_hist),
            np.array(q_cond_hist),
            np.array(q_rad_hist),
            T_shell_in, T_buf_in, T_gel_in, T_sil_in, T_shell_j, T_struct_j,T_struct_in)

print("\n  Running transient TPS integration …")
x_safe = jnp.maximum(x_panel, 1e-6)
T_e = T
T_w_fixed = jnp.full(N, float(Tw))

St_fixed, q_conv_fixed_j, rho_ref_fixed, Taw_fixed, V_e_fixed = heat_transfer_3region(
    x_safe, p, T_e, T_w_fixed, x_tr
)
q_conv_fixed = np.array(q_conv_fixed_j)

(t_hist, T_shell_hist, T_struct_hist, T_interfaces, q_cond, q_conv, q_rad, q_saved, t_ss,t_fail_shell,t_fail_Ti,q_conv_hist,  q_cond_hist, q_rad_hist,T_shell_in,T_buf_in,T_gel_in,T_sil_in,T_shell_j,T_struct_j,T_struct_in) = tps_transient_3region(x_safe, p, T_e, x_tr,
                          t_end=600.0, dt_init=0.1,
                          tol=1e-3, save_every=200)



def compute_thermal_expansion(T_shell_in, T_buf_in, T_gel_in, T_sil_in, x_safe,
                              T_shell_j, T_struct_j, x_tr,
                              T_ref=None, pinned_to_structure=True):
    """
    Geometric growth (delta_L, delta_V): unchanged — free through-thickness
    dimensional change of each layer using alpha_perp, driven by the local
    conduction drop across that layer. Through-thickness is a free surface
    on the outside and layers simply add their own thickness change, so
    there's no compatibility constraint here — this stays a pure kinematic
    diagnostic, not a stress.

    Thermal STRESS: multilayer beam (force + moment balance) — the N-layer
    generalization of the classical Timoshenko bimetallic-strip formula
    (Timoshenko, 1925; Hsueh, Thin Solid Films 418, 182 (2002) for the
    N-layer closed form). Perfectly bonded layers, base -> outer stacking
    Ti / Silicone / Aerogel / Buffer / Shell, share one continuous strain
    field through the thickness (no interfacial slip):

        eps(z)      = eps0 + kappa*z
        sigma_i(z)  = E_par_i * (eps(z) - alpha_par_i * (T_i - T_ref))

    Unlike a pure axial (iso-strain) force balance, this now depends on a
    layer's POSITION z in the stack via the curvature term kappa, not just
    its own isolated E/alpha — that's what lets neighbouring layers genuinely
    influence each other rather than everyone snapping to one shared strain.
    (Note: pure force balance alone, done pairwise or all-at-once, is
    associative and gives the identical global answer either way — it's
    specifically the moment/curvature term that breaks that and introduces
    real position-dependence.)

    eps0, kappa are set by the boundary condition:

      pinned_to_structure=True  (default, matches "TPS system as a whole is
        constrained"): the stack is bonded onto the Ti skin, which is itself
        tied into the much larger/stiffer airframe. eps0 is fixed to Ti's
        own free thermal strain (using its real computed T_struct):
            eps0 = alpha_par_Ti * (T_Ti - T_ref)
        kappa (how much the stack bows) is then solved from moment balance
        M=0, since nothing external stops the TPS stack from bowing.

      pinned_to_structure=False (free-floating stack, self-equilibrated,
        no external structure attached): eps0 AND kappa are solved together
        from N=0 and M=0 across all 5 layers.

    Caveat: this still assumes perfect, rigid (no-slip) bonding at every
    interface. It captures position-dependent coupling but NOT finite bond
    shear compliance — a genuinely compliant interlayer (e.g. the silicone)
    partially decoupling its neighbours via shear requires a shear-lag model
    (Suhir, 1989) on top of this, which is the next level of refinement.
    """
    if T_ref is None:
        T_ref = float(Tw)   # assembly / stress-free temperature

    dT_shell = T_shell_j  - T_shell_in
    dT_buf   = T_shell_in - T_buf_in
    dT_gel   = T_buf_in   - T_gel_in
    dT_sil   = T_gel_in   - T_sil_in

    # ── Linear (through-thickness) expansion — unchanged, kinematic only ──────
    delta_L_shell_lam=0.005*thermo_mech['UHTC']['alpha_perp']*dT_shell
    delta_L_shell_turb=0.005*thermo_mech['CMC']['alpha_perp']*dT_shell
    delta_L_shell = jnp.where(x_safe < x_tr, delta_L_shell_lam, delta_L_shell_turb)

    delta_L_buf=tps_layers_fixed[0]['t']*thermo_mech['Thermal Buffer']['alpha_perp']*dT_buf
    delta_L_aerogel=tps_layers_fixed[1]['t']*thermo_mech['Aerogel']['alpha_perp']*dT_gel
    delta_L_sil=tps_layers_fixed[2]['t']*thermo_mech['Silicone Dampener']['alpha_perp']*dT_sil
    delta_L_Ti=jnp.zeros_like(dT_shell) # we have T_Ti_in=T_sil_in hence no temperature change

    # Volumetric Expansion: dV=V0*B*deltaT with V0=L0.1 with B=2.alpha_par+alpha_perp
    delta_V_shell_lam=(2*thermo_mech['UHTC']['alpha_par']+thermo_mech['UHTC']['alpha_perp'])*0.005*dT_shell
    delta_V_shell_turb=(2*thermo_mech['CMC']['alpha_par']+thermo_mech['CMC']['alpha_perp'])*0.005*dT_shell
    delta_V_shell=jnp.where(x_safe < x_tr, delta_V_shell_lam, delta_V_shell_turb)

    delta_V_buf=(2*thermo_mech['Thermal Buffer']['alpha_par']+thermo_mech['Thermal Buffer']['alpha_perp'])*tps_layers_fixed[0]['t']*dT_buf
    delta_V_aerogel=(2*thermo_mech['Aerogel']['alpha_par']+thermo_mech['Aerogel']['alpha_perp'])*tps_layers_fixed[1]['t']*dT_gel
    delta_V_sil=(2*thermo_mech['Silicone Dampener']['alpha_par']+thermo_mech['Silicone Dampener']['alpha_perp'])*tps_layers_fixed[2]['t']*dT_sil
    delta_V_Ti=jnp.zeros_like(dT_shell)

    # ── Bonded-laminate IN-PLANE thermal stress ───────────────────────────────
    # Layer mean temperatures (Ti treated as ~uniform through its own thin,
    # highly-conductive thickness — Biot number << 1 there)
    T_shell_mean = 0.5 * (T_shell_j  + T_shell_in)
    T_buf_mean   = 0.5 * (T_shell_in + T_buf_in)
    T_gel_mean   = 0.5 * (T_buf_in   + T_gel_in)
    T_sil_mean   = 0.5 * (T_gel_in   + T_sil_in)
    T_Ti_mean    = T_struct_j

    alpha_par_shell = jnp.where(x_safe < x_tr, thermo_mech['UHTC']['alpha_par'], thermo_mech['CMC']['alpha_par'])
    E_par_shell     = jnp.where(x_safe < x_tr, thermo_mech['UHTC']['E_par'],     thermo_mech['CMC']['E_par'])
    t_shell         = jnp.where(x_safe < x_tr, mat_UHTC['t'],                    mat_CMC['t'])

    alpha_par_buf, E_par_buf, t_buf = (thermo_mech['Thermal Buffer']['alpha_par'],
                                        thermo_mech['Thermal Buffer']['E_par'],
                                        tps_layers_fixed[0]['t'])
    alpha_par_gel, E_par_gel, t_gel = (thermo_mech['Aerogel']['alpha_par'],
                                        thermo_mech['Aerogel']['E_par'],
                                        tps_layers_fixed[1]['t'])
    alpha_par_sil, E_par_sil, t_sil = (thermo_mech['Silicone Dampener']['alpha_par'],
                                        thermo_mech['Silicone Dampener']['E_par'],
                                        tps_layers_fixed[2]['t'])
    alpha_par_Ti,  E_par_Ti,  t_Ti  = (thermo_mech['Titanium Alloy']['alpha_par'],
                                        thermo_mech['Titanium Alloy']['E_par'],
                                        tps_layers_fixed[3]['t'])

    # the temperature variation is mean- T_ref fixed at 300K or the T_wall

    dT_shell_g = T_shell_mean-T_ref 
    dT_buf_g   =T_buf_mean-T_ref 
    dT_gel_g   = T_gel_mean -T_ref 
    dT_gel_g   = T_gel_mean-T_ref 
    dT_sil_g   = T_sil_mean -T_ref 
    dT_Ti_g    =T_Ti_mean  -T_ref 

    # z-positions of each REAL layer (base=Ti at z=0, outward through Sil/Aerogel/Buf/Shell)
    t_list  = [t_Ti, t_sil, t_gel, t_buf, t_shell]
    #thickness transformed into array
    t_arrs = [t_i if hasattr(t_i, 'shape') else jnp.full_like(dT_shell_g, t_i) for t_i in t_list]
    t_total = sum(t_arrs)
    z_bounds = [-0.5 * t_total]
    for t_i in t_arrs:
        z_bounds.append(z_bounds[-1] + t_i)

    z_mid = [0.5 * (z_bounds[i] + z_bounds[i+1]) for i in range(5)]

    
    E_list5     = [E_par_Ti,     E_par_sil,     E_par_gel,     E_par_buf,     E_par_shell]
    alpha_list5 = [alpha_par_Ti, alpha_par_sil, alpha_par_gel, alpha_par_buf, alpha_par_shell]
    dT_list5    = [dT_Ti_g,      dT_sil_g,      dT_gel_g,      dT_buf_g,      dT_shell_g]  # which is (T_mean - T_ref) w/ T_ref fixed at 300K
    # shell is the hot side as we sum up all the layers and z is going from cold to hot

    # solving the 2x2 system with the stiffness matrices and thermal loads thermal to get kappa = how much it will curve 
    A11=jnp.zeros_like(dT_shell_g); A12=jnp.zeros_like(dT_shell_g); A22=jnp.zeros_like(dT_shell_g)
    N_T=jnp.zeros_like(dT_shell_g); M_T=jnp.zeros_like(dT_shell_g)

    for i in range(5):
        z0, z1 = z_bounds[i], z_bounds[i+1]
        E_i = E_list5[i] if hasattr(E_list5[i], 'shape') else jnp.full_like(dT_shell_g, E_list5[i])
        alpha_i = alpha_list5[i]
        dT_i = dT_list5[i]
        a=z1-z0; b=(z1**2-z0**2)/2.0; c=(z1**3-z0**3)/3.0

        A11=A11+E_i*a; A12=A12+E_i*b; A22=A22+E_i*c
        N_T += E_i * alpha_i * dT_i * a
        M_T += E_i * alpha_i * dT_i * b
    # we calculate epsilon and kappa simiultaneously 
    det = A11 * A22 - A12**2

    eps0_free  = (A22 * N_T - A12 * M_T) / det  # True unconstrained midplane strain
    kappa_free = (A11 * M_T - A12 * N_T) / det

    # z is the distance from Ti (the fixed surface )

    sigma_Ti_fig5      = E_par_Ti    * (eps0_free + kappa_free*z_mid[0] - alpha_par_Ti    * dT_Ti_g)
    sigma_sil_fig5     = E_par_sil   * (eps0_free + kappa_free*z_mid[1] - alpha_par_sil   * dT_sil_g)
    sigma_aerogel_fig5 = E_par_gel   * (eps0_free + kappa_free*z_mid[2] - alpha_par_gel   * dT_gel_g)
    sigma_buf_fig5     = E_par_buf   * (eps0_free + kappa_free*z_mid[3] - alpha_par_buf   * dT_buf_g)
    sigma_shell_fig5   = E_par_shell * (eps0_free + kappa_free*z_mid[4] - alpha_par_shell * dT_shell_g)




    return (delta_L_shell,delta_L_buf,delta_L_aerogel,delta_L_sil,delta_L_Ti,delta_V_shell,delta_V_buf,delta_V_aerogel,delta_V_sil,delta_V_Ti,
            sigma_shell_fig5, sigma_buf_fig5, sigma_aerogel_fig5, sigma_sil_fig5, sigma_Ti_fig5,dT_shell_g,dT_sil_g,dT_gel_g,dT_buf_g)


x_safe = jnp.maximum(x_panel, 1e-6)

(delta_L_shell, delta_L_buf, delta_L_aerogel, delta_L_sil, delta_L_Ti,
 delta_V_shell, delta_V_buf, delta_V_aerogel, delta_V_sil, delta_V_Ti,
 sigma_shell_fig5, sigma_buf_fig5, sigma_aerogel_fig5, sigma_sil_fig5, sigma_Ti_fig5,dT_shell_g,dT_sil_g,dT_gel_g,dT_buf_g) = \
    compute_thermal_expansion(T_shell_in, T_buf_in, T_gel_in, T_sil_in,
                               x_safe, T_shell_j, T_struct_j, x_tr,
                               pinned_to_structure=True)



# PRINTOUT
print()
# ── Column widths ─────────────────────────────────────────────────────────────
WP = {'x': 7, 'dL': 12, 'dV': 12, 'sig': 12}

layer_labels = ['Shell', 'Buffer', 'Aerogel', 'Silicone', 'Ti']

delta_L_all = [delta_L_shell, delta_L_buf, delta_L_aerogel, delta_L_sil, delta_L_Ti]
delta_V_all = [delta_V_shell, delta_V_buf, delta_V_aerogel, delta_V_sil, delta_V_Ti]
sigma_all   = [sigma_shell_fig5,   sigma_buf_fig5,   sigma_aerogel_fig5,   sigma_sil_fig5,   sigma_Ti_fig5  ]

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
    print(f"  (sigma = bonded-laminate in-plane thermal stress, pinned to Ti structure)")
    

# PLOT 

def plot_thermal_expansion(x_panel, x_tr,
                           delta_L_all, delta_V_all, sigma_all,
                           savename='thermal_expansion.png'):

    x_p    = np.array(x_panel)
    labels = ['Shell (UHTC/CMC)', 'Thermal Buffer',
              'Aerogel', 'Silicone Dampener', 'Titanium']
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
         'Thermal Stress  σ = E_par·(ε0 − α_par·ΔT)\n(bending suppressed by rigid fasteners, 200mm bay)',   1e-6),
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
    [delta_L_shell, delta_L_buf, delta_L_aerogel, delta_L_sil, delta_L_Ti],
    [delta_V_shell, delta_V_buf, delta_V_aerogel, delta_V_sil, delta_V_Ti],
    [sigma_shell_fig5, sigma_buf_fig5, sigma_aerogel_fig5, sigma_sil_fig5, sigma_Ti_fig5]
)









# PRINTOUT

print("\n" + "=" * 80)
print("  TPS HEAT TRANSFER — PANEL BY PANEL  (FINAL / QUASI-STEADY STATE)")
print("=" * 80)
print(f"  {'x':>7} {'shell':>6} {'q_conv':>12} {'q_cond':>12} {'q_rad':>12} "
      f"{'q_saved':>12} {'sav%':>7} {'T_shell':>9} {'T_struct':>9}")
for i in range(N):
    shell = 'UHTC' if is_uhtc[i] else 'CMC'
    q_conv_i=float(q_conv[i])
    q_conv_ref_i=float(q_conv_fixed[i])
    sav_pct = 100.0 * float(q_saved[i]) / max(float(q_conv_ref_i), 1e-6)
    sh_flag  = ' !' if float(T_interfaces[0, i]) > T_allow_shell[i] else '  '
    ti_flag  = ' !' if float(T_interfaces[5, i]) > T_allow_Ti        else '  '
    print(f"  {float(x_panel[i]):7.3f} {shell:>6} {float(q_conv[i]):12.1f} "
          f" {float(q_cond[i]):12.3f} {float(q_rad[i]):12.1f} "
          f"{float(q_saved[i]):12.1f} {sav_pct:7.1f}% "
          f"{float(T_interfaces[0,i]):9.1f} {float(T_interfaces[5,i]):9.1f}")

print(f"\n  {'-'*60}")
print(f"  Total q_rad    : {float(q_rad.sum()):10.1f} W/m²  (radiated to freestream)")
print(f"  Total q_saved  : {float(q_saved.sum()):10.1f} W/m²")
print(f"  Mean saving    : {100*float(q_saved.mean()/q_conv.mean()):10.1f}")
print(f"  Max T_shell    : {float(T_interfaces[0].max()):10.1f} K")
print(f"  Max T_struct   : {float(T_interfaces[5].max()):10.1f} K  "
      f"(limit {T_allow_Ti:.1f} K)  -> "
      f"{'WITHIN LIMIT' if float(T_interfaces[5].max()) < T_allow_Ti else 'EXCEEDS LIMIT'}")
if t_ss is not None:
    print(f"  Steady state   : t = {t_ss:.1f} s")
else:
    print(f"  Steady state   : NOT reached within t_end")


# ── Material status block ─────────────────────────────────────────────────────
print(f"\n  MATERIAL STATUS:")
shell_ok = np.all(np.array(T_interfaces[0]) <= T_allow_shell)
Ti_ok    = float(T_interfaces[5].max()) < T_allow_Ti

print(f"    Shell (UHTC/CMC) : {'OK' if shell_ok else 'EXCEEDED':<10}  "
      f"max = {float(T_interfaces[0].max()):.1f} K")
for i in range(N):
    lim = T_allow_shell[i]
    mat = 'UHTC' if is_uhtc[i] else 'CMC '
    status = 'OK' if float(T_interfaces[0, i]) <= lim else 'EXCEEDED'
    print(f"      panel {i+1:>2d}  x={float(x_panel[i]):.3f}m  {mat}  "
          f"{float(T_interfaces[0,i]):.1f} K / limit {lim:.0f} K  → {status}")

print(f"    Titanium         : {'OK' if Ti_ok else 'EXCEEDED':<10}  "
      f"max = {float(T_interfaces[5].max()):.1f} K  /  limit = {T_allow_Ti:.1f} K")

if t_fail_shell is not None:
    print(f"    Shell first exceeded limit at t = {t_fail_shell:.1f} s")
else:
    print(f"    Shell limit never exceeded within t_end")

if t_fail_Ti is not None:
    print(f"    Titanium first exceeded limit at t = {t_fail_Ti:.1f} s")
else:
    print(f"    Titanium limit never exceeded within t_end")

if t_ss is not None:
    print(f"  Steady state     : t = {t_ss:.1f} s")
else:
    print(f"  Steady state     : NOT reached within t_end")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS — defined ONCE
# ═══════════════════════════════════════════════════════════════════════════════
def draw_vehicle(ax, x, y, zorder=2):
    """Surface line, vertical base, and bottom centreline (half-vehicle)."""
    ax.plot(x, y, 'k-', linewidth=2.5, zorder=zorder)
    ax.plot([x_target, x_target], [0.0, y_target], 'k-', linewidth=2.5, zorder=zorder)
    ax.plot([0.0, x_target], [0.0, 0.0], 'k-', linewidth=2.5, zorder=zorder)


def _build_node_shell_mask():
    """Node belongs to UHTC only if BOTH adjacent panels are UHTC."""
    is_uhtc_node = np.empty(N + 1, dtype=bool)
    is_uhtc_node[0]  = is_uhtc[0]
    is_uhtc_node[-1] = is_uhtc[-1]
    for k in range(1, N):
        is_uhtc_node[k] = is_uhtc[k-1] and is_uhtc[k]
    return is_uhtc_node


def _build_cum_frac_node(is_uhtc_node):
    """Per-node cumulative fraction of LOCAL total TPS thickness, for offsets."""
    t_outer_node = np.where(is_uhtc_node, mat_UHTC['t'], mat_CMC['t'])
    fixed_t      = np.array([L['t'] for L in tps_layers_fixed])
    t_total_node = t_outer_node + fixed_t.sum()

    cum_frac_node = np.zeros((6, N + 1))   # 6 boundaries: surface + 5 layer ends
    cum_frac_node[1] = t_outer_node / t_total_node
    running = cum_frac_node[1].copy()
    for li, t_l in enumerate(fixed_t):
        running = running + t_l / t_total_node
        cum_frac_node[2 + li] = running
    return cum_frac_node


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — TPS GEOMETRY, HALF VEHICLE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_layers_half(x, y, x_panel, x_tr, savename='tps_layers_half.png'):
    fig, ax = plt.subplots(figsize=(16, 5))
    x_np, y_np = np.array(x, dtype=float), np.array(y, dtype=float)

    is_uhtc_node  = _build_node_shell_mask()
    cum_frac_node = _build_cum_frac_node(is_uhtc_node)

    fixed_names  = [L['name'] for L in tps_layers_fixed]
    fixed_t      = np.array([L['t'] for L in tps_layers_fixed])
    fixed_colors = ['#f1c40f', '#3498db', '#9b59b6', '#7f8c8d']
    vis_fraction = 0.55

    def offset_curve(boundary_idx):
        frac = cum_frac_node[boundary_idx]
        return x_np.copy(), y_np * (1.0 - frac * vis_fraction)

    # interior background
    _, y_deep = offset_curve(5)
    ax.fill(list(x_np) + list(x_np[::-1]), list(y_np) + list(y_deep[::-1]),
            color='#bdc3c7', alpha=0.55, zorder=1, label='Vehicle interior')

    # outer shell split UHTC / CMC
    x_out, y_out = offset_curve(0)
    x_in,  y_in  = offset_curve(1)
    mask_u = is_uhtc_node
    if mask_u.any():
        idx = np.where(mask_u)[0]
        ax.fill(list(x_out[idx]) + list(x_in[idx][::-1]),
                list(y_out[idx]) + list(y_in[idx][::-1]),
                color='#16a085', alpha=0.92, zorder=2,
                label=f"UHTC  {mat_UHTC['t']*1000:.1f} mm  k={mat_UHTC['k']}")
    mask_c = ~is_uhtc_node
    if mask_c.any():
        idx = np.where(mask_c)[0]
        ax.fill(list(x_out[idx]) + list(x_in[idx][::-1]),
                list(y_out[idx]) + list(y_in[idx][::-1]),
                color='#c0392b', alpha=0.92, zorder=2,
                label=f"CMC  {mat_CMC['t']*1000:.1f} mm  k={mat_CMC['k']}")
    ax.plot(x_in, y_in, 'k-', linewidth=0.8, alpha=0.5, zorder=8)

    # fixed inner layers
    for li in range(4):
        x_o, y_o = offset_curve(1 + li)
        x_i, y_i = offset_curve(2 + li)
        ax.fill(list(x_o) + list(x_i[::-1]), list(y_o) + list(y_i[::-1]),
                color=fixed_colors[li], alpha=0.88, zorder=3 + li,
                label=f"{fixed_names[li]}  {fixed_t[li]*1000:.1f} mm")
        ax.plot(x_i, y_i, 'k-', linewidth=0.7, alpha=0.5, zorder=8 + li)

    draw_vehicle(ax, x_np, y_np, zorder=20)
    ax.axvline(float(x_tr), color='blue', linestyle='--', linewidth=1.3,
               label=f'x_tr = {float(x_tr):.2f} m (UHTC->CMC)', zorder=21)

    ax.legend(fontsize=7, loc='upper left', ncol=1)
    ax.set_ylim(-y_target * 0.15, y_target * 2.8)
    ax.set_xlim(-0.05, x_target + 0.15)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('TPS stack (half vehicle) - UHTC leading edge -> CMC body, '
                 'shared inner layers\n(visual scale: 55% of local height)')
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — TPS GEOMETRY, FULL VEHICLE (mirrored about centreline)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_layers_full(x, y, x_panel, x_tr, savename='tps_layers_full.png'):
    fig, ax = plt.subplots(figsize=(16, 7))
    x_np, y_np = np.array(x, dtype=float), np.array(y, dtype=float)

    is_uhtc_node  = _build_node_shell_mask()
    cum_frac_node = _build_cum_frac_node(is_uhtc_node)

    fixed_names  = [L['name'] for L in tps_layers_fixed]
    fixed_t      = np.array([L['t'] for L in tps_layers_fixed])
    fixed_colors = ['#f1c40f', '#3498db', '#9b59b6', '#7f8c8d']
    vis_fraction = 0.55

    def offset_curve(boundary_idx, sign):
        frac = cum_frac_node[boundary_idx]
        return x_np.copy(), sign * y_np * (1.0 - frac * vis_fraction)

    def draw_half(sign, show_labels):
        _, y_deep = offset_curve(5, sign)
        ax.fill(list(x_np) + list(x_np[::-1]), list(sign*y_np) + list(y_deep[::-1]),
                color='#bdc3c7', alpha=0.55, zorder=1,
                label='Vehicle interior' if show_labels else None)

        x_out, y_out = offset_curve(0, sign)
        x_in,  y_in  = offset_curve(1, sign)
        mask_u = is_uhtc_node
        if mask_u.any():
            idx = np.where(mask_u)[0]
            ax.fill(list(x_out[idx]) + list(x_in[idx][::-1]),
                    list(y_out[idx]) + list(y_in[idx][::-1]),
                    color='#16a085', alpha=0.92, zorder=2,
                    label=(f"UHTC  {mat_UHTC['t']*1000:.1f} mm  k={mat_UHTC['k']}" if show_labels else None))
        mask_c = ~is_uhtc_node
        if mask_c.any():
            idx = np.where(mask_c)[0]
            ax.fill(list(x_out[idx]) + list(x_in[idx][::-1]),
                    list(y_out[idx]) + list(y_in[idx][::-1]),
                    color='#c0392b', alpha=0.92, zorder=2,
                    label=(f"CMC  {mat_CMC['t']*1000:.1f} mm  k={mat_CMC['k']}" if show_labels else None))
        ax.plot(x_in, y_in, 'k-', linewidth=0.8, alpha=0.5, zorder=8)

        for li in range(4):
            x_o, y_o = offset_curve(1 + li, sign)
            x_i, y_i = offset_curve(2 + li, sign)
            ax.fill(list(x_o) + list(x_i[::-1]), list(y_o) + list(y_i[::-1]),
                    color=fixed_colors[li], alpha=0.88, zorder=3 + li,
                    label=(f"{fixed_names[li]}  {fixed_t[li]*1000:.1f} mm" if show_labels else None))
            ax.plot(x_i, y_i, 'k-', linewidth=0.7, alpha=0.5, zorder=8 + li)

        ax.plot(x_np, sign * y_np, 'k-', linewidth=2.5, zorder=20)
        ax.plot([x_np[-1], x_np[-1]], [0.0, sign * y_np[-1]], 'k-', linewidth=2.5, zorder=20)

    draw_half(+1.0, show_labels=True)
    draw_half(-1.0, show_labels=False)

    ax.plot([x_np[-1], x_np[-1]], [-y_np[-1], y_np[-1]], 'k-', linewidth=2.5, zorder=20)
    ax.axhline(0.0, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, zorder=21)
    ax.axvline(float(x_tr), color='blue', linestyle='--', linewidth=1.3,
               label=f'x_tr = {float(x_tr):.2f} m (UHTC->CMC)', zorder=22)

    ax.legend(fontsize=7, loc='upper left', ncol=1)
    ax.set_ylim(-y_target * 1.8, y_target * 3.2)
    ax.set_xlim(-0.05, x_target + 0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('TPS stack - full vehicle, symmetric - UHTC leading edge -> CMC body\n'
                 '(visual scale: 55% of local half-height)')
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — TRANSIENT T_shell AND T_struct EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_transient(t_hist, T_shell_hist, T_struct_hist, x_panel, t_ss,
                   t_fail_shell, t_fail_Ti, savename='tps_transient.png'):

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('TPS Transient — Outer Shell (UHTC/CMC) and Titanium Structure',
                 fontsize=12)

    T_shell_arr  = np.array(T_shell_hist)
    T_struct_arr = np.array(T_struct_hist)
    t_arr = np.array(t_hist)
    cmap  = cm.get_cmap('plasma',N)
    x_p   = np.array(x_panel)

    # ── Left: T_shell(t) ─────────────────────────────────────────────────────
    ax = axes[0]
    for i in range(N):
        ax.plot(t_arr, T_shell_arr[:, i], color=cmap(i / N),
                label=f'x={float(x_panel[i]):.2f} m')
    # UHTC and CMC limits as horizontal bands
    ax.axhline(T_allow_UHTC, color='#16a085', linestyle='--', linewidth=1.5,
               label=f'UHTC limit {T_allow_UHTC:.0f} K')
    ax.axhline(T_allow_CMC,  color='#c0392b', linestyle='--', linewidth=1.5,
               label=f'CMC limit  {T_allow_CMC:.0f} K')
    if t_ss is not None:
        ax.axvline(t_ss, color='k', linestyle='--', linewidth=1.5,
                   label=f'SS t={t_ss:.0f} s')
    if t_fail_shell is not None:
        ax.axvline(t_fail_shell, color='red', linestyle=':', linewidth=2.0,
                   label=f'Shell fail t={t_fail_shell:.0f} s')
        ax.annotate(f'Shell\nexceeded\nt={t_fail_shell:.0f}s',
                    xy=(t_fail_shell, T_shell_arr.max()),
                    xytext=(t_fail_shell + 20, T_shell_arr.max() * 0.95),
                    fontsize=7, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('Time (s)'); ax.set_ylabel('T (K)')
    ax.set_title('T_shell(t) — all panels\n'
                 f'Limits: UHTC {T_allow_UHTC:.0f} K  |  CMC {T_allow_CMC:.0f} K')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # ── Middle: T_struct(t) ───────────────────────────────────────────────────
    ax = axes[1]
    for i in range(N):
        ax.plot(t_arr, T_struct_arr[:, i], color=cmap(i / N),
                label=f'x={float(x_panel[i]):.2f} m')
    ax.axhline(T_allow_Ti, color='red', linestyle='--', linewidth=1.8,
               label=f'Ti limit {T_allow_Ti:.1f} K')
    ax.axhline(Tw, color='purple', linestyle=':', linewidth=1.2,
               label=f'Initial Tw={Tw:.0f} K')
    if t_ss is not None:
        ax.axvline(t_ss, color='k', linestyle='--', linewidth=1.5,
                   label=f'SS t={t_ss:.0f} s')
    if t_fail_Ti is not None:
        ax.axvline(t_fail_Ti, color='red', linestyle=':', linewidth=2.0,
                   label=f'Ti fail t={t_fail_Ti:.0f} s')
        ax.annotate(f'Ti\nexceeded\nt={t_fail_Ti:.0f}s',
                    xy=(t_fail_Ti, T_allow_Ti),
                    xytext=(t_fail_Ti + 20, T_allow_Ti * 1.02),
                    fontsize=7, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_xlabel('Time (s)'); ax.set_ylabel('T (K)')
    ax.set_title(f'T_struct(t) — Titanium\nLimit: {T_allow_Ti:.1f} K  '
                 f'({"EXCEEDED" if t_fail_Ti else "within limit"})')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # ── Right: spatial profiles at 5 snapshots ────────────────────────────────
    ax = axes[2]
    snap_indices = [0, len(T_shell_hist)//4, len(T_shell_hist)//2,
                    3*len(T_shell_hist)//4, -1]
    snap_colors  = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
    for idx, col in zip(snap_indices, snap_colors):
        t_label = f't={t_hist[idx]:.0f}s'
        ax.plot(x_p, T_shell_arr[idx],  '-',  color=col,
                linewidth=1.8, label=f'T_shell {t_label}')
        ax.plot(x_p, T_struct_arr[idx], '--', color=col,
                linewidth=1.2, label=f'T_struct {t_label}')
    # Limit lines on spatial plot — per-panel aware
    ax.step(x_p, T_allow_shell, where='mid',
            color='red', linestyle='--', linewidth=1.5,
            label='Shell limit (UHTC/CMC)')
    ax.axhline(T_allow_Ti, color='darkorange', linestyle='--', linewidth=1.5,
               label=f'Ti limit {T_allow_Ti:.0f} K')
    ax.axhline(Tw, color='purple', linestyle=':', linewidth=1.0,
               label=f'Tw0={Tw:.0f} K')
    ax.set_xlabel('x (m)'); ax.set_ylabel('T (K)')
    ax.set_title('Spatial profiles: T_shell (—) & T_struct (--)\n'
                 'Red step = shell limit per panel')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — TPS HEAT FLUX BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
def plot_tps_heat_flux(x_panel, q_conv, q_cond, q_rad, q_saved,
                       T_interfaces, savename='tps_heatflux.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('TPS Heat Flux Breakdown (Final / Quasi-Steady State)', fontsize=12)
    x_p = np.array(x_panel)

    # ── Top-left: heat flux components ───────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x_p, np.array(q_conv),  'o-', color='crimson',    label='q_conv (into shell)')
    ax.plot(x_p, np.array(q_cond),  's-', color='steelblue',  label='q_cond (to structure)')
    ax.plot(x_p, np.array(q_rad),   '^-', color='darkorange', label='q_rad (radiated away)')
    ax.plot(x_p, np.array(q_saved), 'd-', color='green',      label='q_saved (intercepted)')
    ax.axvline(float(x_tr), color='k', linestyle=':', label=f'x_tr={float(x_tr):.2f} m')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Heat flux (W/m²)')
    ax.set_title('Heat Flux Components')

    # ── Top-right: saving % ───────────────────────────────────────────────────
    ax = axes[0, 1]
    saving_pct = 100.0 * np.array(q_saved) / np.maximum(np.array(q_conv_fixed), 1e-6)
    ax.plot(x_p, saving_pct, 'o-', color='purple')
    ax.axhline(0,   color='k',   linewidth=0.8)
    ax.axhline(100, color='red', linewidth=0.8, linestyle='--', label='100%')
    ax.axvline(float(x_tr), color='k', linestyle=':', label=f'x_tr')
    ax.set_ylim(10,102)
    ax.grid(True); ax.legend(fontsize=8)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Saving (%)')
    ax.set_title('TPS Heat Saving  (q_saved / q_conv)')

    # ── Bottom-left: interface temperatures + limits ──────────────────────────
    ax = axes[1, 0]


    t_shell, t_buffer, t_aerogel, t_silicone, t_Ti = (mat_UHTC['t'], tps_layers_fixed[0]['t'], tps_layers_fixed[1]['t'], tps_layers_fixed[2]['t'], tps_layers_fixed[3]['t'] )  # <-- replace with your real names)
    z_mm = np.array([0.0,t_shell,t_shell + t_buffer,t_shell + t_buffer + t_aerogel,t_shell + t_buffer + t_aerogel + t_silicone,t_shell + t_buffer + t_aerogel + t_silicone + t_Ti,]) * 1000.0   # m -> mm

# 5 material segments between the 6 interface depths, outer -> inner
    layer_names  = ['Shell', 'Buffer', 'Aerogel', 'Silicone', 'Ti']
    layer_colors = ['#16a085', '#f1c40f', '#3498db', '#9b59b6', '#2c3e50']

    T_interfaces_np = np.array(T_interfaces)   # (6, N)

# ── Pick ONE representative x-station: laminar region ──
    x_panel_np = np.array(x_panel)
    idx_lam = int(np.argmax(x_panel_np < float(x_tr))) if np.any(x_panel_np < float(x_tr)) else 0
    panel_idx    = [idx_lam]
    panel_labels = [f'Laminar  x={x_panel_np[idx_lam]:.2f} m']
    panel_styles = ['-']    # distinguishes stations; color distinguishes layer
    
    for i, ls, plabel in zip(panel_idx, panel_styles, panel_labels):
        T_prof = T_interfaces_np[:, i]
    
        for seg in range(5):   # 4 layer segments between the 5 depth points
            ax.plot(z_mm[seg:seg+2], T_prof[seg:seg+2],
                linestyle=ls, color=layer_colors[seg],
                linewidth=2.2, marker='o', markersize=5)

# ── Reference limits — now horizontal lines at THIS panel's values ─────
# Shell limit is UHTC-or-CMC depending on panel — index into the per-panel
# array at idx_lam rather than plotting the step function (there's no x
# axis here to step across anymore).
    ax.axhline(float(T_allow_shell[idx_lam]), color='red', linestyle='--',
           linewidth=2.0,
           label=f'Shell limit ({float(T_allow_shell[idx_lam]):.0f} K)')
    ax.axhline(T_allow_Ti, color='darkorange', linestyle='--', linewidth=1.8,
           label=f'Ti limit {T_allow_Ti:.1f} K')
    ax.axhline(Tw, color='purple', linestyle=':', linewidth=1.0,
           label=f'Initial Tw={Tw:.0f} K')

# ── Layer (material) legend ─────────────────────────────────────────────
    from matplotlib.lines import Line2D
    layer_handles = [Line2D([0], [0], color=c, lw=2.5) for c in layer_colors]
    leg1 = ax.legend(layer_handles, layer_names, fontsize=7,
                  title='Layer', loc='upper right')
    ax.add_artist(leg1)
    ax.legend(fontsize=6.5, loc='upper right',
          bbox_to_anchor=(1.0, 0.62))   # sits just below leg1

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Depth from outer Shell face (mm)')
    ax.set_ylabel('T (K)')
    ax.set_title('TPS Through-Thickness Temperature Profile\n(laminar / mid-body / trailing-edge stations)')

    # ── Bottom-right: q_wall vs q_conv reference ──────────────────────────────
    ax = axes[1, 1]
    ax.plot(x_p, np.array(q_conv), 'o--', color='grey',
            label='q_wall (no TPS, Tw=300K ref)')
    ax.plot(x_p, np.array(q_conv), 'o-',  color='crimson',
            label='q_conv (with TPS, T_shell actual)')
    ax.axvline(float(x_tr), color='k', linestyle=':', label=f'x_tr={float(x_tr):.2f} m')
    ax.grid(True); ax.legend(fontsize=8)
    ax.set_xlabel('x (m)'); ax.set_ylabel('Heat flux (W/m²)')
    ax.set_title('Reference: q_wall (no TPS) vs q_conv (with TPS)\n'
                 'NOT used in q_saved — shown for context only')

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — 3D TRANSIENT  (same style as previous code)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_transient_3d(t_hist, T_shell_hist, T_struct_hist, x_panel, Taw,
                      t_ss, t_fail_shell, t_fail_Ti,
                      savename='tps_transient_3d.png'):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    T_shell_arr  = np.array(T_shell_hist)
    T_struct_arr = np.array(T_struct_hist)
    t_arr = np.array(t_hist)
    x_p   = np.array(x_panel, dtype=float)
    Taw_np = np.array(Taw, dtype=float)

    def flatten_curve(T_arr):
        X, T_flat, Z = [], [], []
        for k, t_k in enumerate(t_arr):
            X.extend(x_p.tolist())
            T_flat.extend([t_k] * len(x_p))
            Z.extend(T_arr[k].tolist())
        return np.array(X), np.array(T_flat), np.array(Z)

    X_sh,  T_sh,  Z_sh  = flatten_curve(T_shell_arr)
    X_st,  T_st,  Z_st  = flatten_curve(T_struct_arr)

    def colored_3d_line(ax, X, T, Z, cmap_name):
        points = np.array([X, T, Z]).T.reshape(-1, 1, 3)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        norm   = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
        lc     = Line3DCollection(segs, cmap=cm.get_cmap(cmap_name),
                                  norm=norm, linewidth=1.5, alpha=0.9)
        lc.set_array(Z[:-1])
        ax.add_collection3d(lc)
        return lc

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f'TPS Temperature — T(x, t)\n'
                 f'M∞={M_inf:.1f},  h=20 km,  t_end={t_arr[-1]:.0f} s',
                 fontsize=13, fontweight='bold')

    X_pl, T_pl = np.meshgrid(
        np.linspace(x_p.min(), x_p.max(), 10),
        np.linspace(t_arr.min(), t_arr.max(), 10)
    )

    # ── Left: T_shell ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection='3d')
    lc1 = colored_3d_line(ax1, X_sh, T_sh, Z_sh, 'plasma')

    # UHTC limit plane
    ax1.plot_surface(X_pl, T_pl, np.full_like(X_pl, T_allow_UHTC),
                     alpha=0.12, color='#16a085')
    ax1.text(x_p.mean(), t_arr[-1], T_allow_UHTC,
             f'UHTC limit {T_allow_UHTC:.0f} K',
             color='#16a085', fontsize=8, fontweight='bold')

    # CMC limit plane
    ax1.plot_surface(X_pl, T_pl, np.full_like(X_pl, T_allow_CMC),
                     alpha=0.12, color='#c0392b')
    ax1.text(x_p.mean(), t_arr[-1], T_allow_CMC,
             f'CMC limit {T_allow_CMC:.0f} K',
             color='#c0392b', fontsize=8, fontweight='bold')

    # Taw reference line
    ax1.plot(x_p, np.full_like(x_p, t_arr[-1]), Taw_np,
             color='cyan', linewidth=2.0, linestyle='--', label='Taw')

    if t_ss is not None:
        ax1.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_ss),
                 np.linspace(Z_sh.min(), Z_sh.max(), 10),
                 'w--', linewidth=1.5, label=f'SS t={t_ss:.0f} s')

    if t_fail_shell is not None:
        ax1.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_fail_shell),
                 np.linspace(Z_sh.min(), Z_sh.max(), 10),
                 'r:', linewidth=2.0, label=f'Fail t={t_fail_shell:.0f} s')

    cb1 = fig.colorbar(lc1, ax=ax1, pad=0.1, shrink=0.6)
    cb1.set_label('T_shell [K]', fontsize=9)
    ax1.set_xlabel('x [m]', fontsize=9, labelpad=8)
    ax1.set_ylabel('Time [s]', fontsize=9, labelpad=8)
    ax1.set_zlabel('T_shell [K]', fontsize=9, labelpad=8)
    ax1.set_title('Outer Shell (UHTC/CMC)', fontsize=11)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.view_init(elev=25, azim=-50)

    # ── Right: T_struct ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    lc2 = colored_3d_line(ax2, X_st, T_st, Z_st, 'viridis')

    # Ti limit plane
    ax2.plot_surface(X_pl, T_pl, np.full_like(X_pl, T_allow_Ti),
                     alpha=0.15, color='darkorange')
    ax2.text(x_p.mean(), t_arr[-1], T_allow_Ti,
             f'Ti limit {T_allow_Ti:.1f} K',
             color='darkorange', fontsize=8, fontweight='bold')

    if t_ss is not None:
        ax2.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_ss),
                 np.linspace(Z_st.min(), Z_st.max(), 10),
                 'w--', linewidth=1.5, label=f'SS t={t_ss:.0f} s')

    if t_fail_Ti is not None:
        ax2.plot(np.linspace(x_p.min(), x_p.max(), 10),
                 np.full(10, t_fail_Ti),
                 np.linspace(Z_st.min(), Z_st.max(), 10),
                 'r:', linewidth=2.0, label=f'Ti fail t={t_fail_Ti:.0f} s')

    cb2 = fig.colorbar(lc2, ax=ax2, pad=0.1, shrink=0.6)
    cb2.set_label('T_struct [K]', fontsize=9)
    ax2.set_xlabel('x [m]', fontsize=9, labelpad=8)
    ax2.set_ylabel('Time [s]', fontsize=9, labelpad=8)
    ax2.set_zlabel('T_struct [K]', fontsize=9, labelpad=8)
    ax2.set_title('Titanium Structure', fontsize=11)
    ax2.legend(fontsize=7, loc='upper left')
    ax2.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# PLOT TO SHOW Q
# ── paste your function exactly as-is ────────────────────────────────────────
def plot_heatflux_transient(t_hist, q_conv_hist,  q_cond_hist, q_rad_hist,
                            savename='tps_heatflux_transient.png'):
    t_arr      = np.array(t_hist)
    q_conv_arr = np.array(q_conv_hist)
    q_cond_arr = np.array(q_cond_hist)
    q_rad_arr  = np.array(q_rad_hist)

    n = min(len(t_arr), len(q_conv_arr), len(q_cond_arr), len(q_rad_arr))
    t_arr      = t_arr[:n]
    q_conv_arr = q_conv_arr[:n]
    q_cond_arr = q_cond_arr[:n]
    q_rad_arr  = q_rad_arr[:n]

    print(f"  Plotting fluxes: n_snapshots={n}, N_panels={q_conv_arr.shape[1]}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f'Heat Flux Evolution vs Time\n'
        f'M∞ = {M_inf:.1f},  h = 20 km,  t_end = {t_arr[-1]:.0f} s',
        fontsize=13, fontweight='bold'
    )

    flux_data   = [q_conv_arr,           q_cond_arr,        q_rad_arr        ]
    flux_labels = ['q_conv  [W/m²]',    'q_cond  [W/m²]',  'q_rad  [W/m²]'  ]
    flux_colors = ['crimson',           'steelblue',       'darkorange'     ]

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
            y = slicer(q_arr)
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
plot_tps_layers_half(x, y, x_panel, x_tr)
plot_tps_layers_full(x, y, x_panel, x_tr)
plot_transient(t_hist, T_shell_hist, T_struct_hist, x_panel, t_ss,
               t_fail_shell, t_fail_Ti)                          # ← new args
plot_tps_heat_flux(x_panel, q_conv, q_cond, q_rad, q_saved, T_interfaces)
plot_transient_3d(t_hist, T_shell_hist, T_struct_hist, x_panel, Taw,
                  t_ss, t_fail_shell, t_fail_Ti)     
plot_heatflux_transient(t_hist, q_conv_hist, q_cond_hist, q_rad_hist)

print("\nAll plots saved.")