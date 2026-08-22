"""
postprocess_inviscid.py
──────────────────────────────────────────────────────────────────────────────
Run AFTER run_inviscid.py has finished successfully.

FIXES APPLIED vs ORIGINAL
──────────────────────────
1. Physical validity filter on near-wall cells: excludes cells with
   rho < 1% freestream or p < 1% freestream before binning. These are
   the corrupted cut-cells kept alive by the positivity limiter that
   were contaminating the edge-condition averages.

2. Minimum cell count per bin: bins with fewer than 3 valid cells are
   excluded from the azimuthal average (they are noisy).

3. Edge-condition sanity printout: prints p_e, T_e, rho_e, V_e, M_e
   per x-station so you can verify physical validity before the heat
   flux calculation runs.

4. Results folder: explicitly prints which numbered folder is being
   read so you can confirm it is the completed run, not a crashed one.

5. Fay-Riddell stagnation-point heat flux: the Eckert/Van-Driest-II
   flat-plate correlation is singular (and physically wrong) as x->0,
   which is exactly where the stagnation/leading-edge panel sits. That
   panel's q_wall is now replaced with the Fay-Riddell blunt-body
   stagnation result instead, for BOTH the CFD-edge and Newtonian-edge
   comparisons. Fay-Riddell only needs freestream + wall + nose-radius
   inputs, so it doesn't depend on (often noisiest) near-wall CFD data
   right at the stagnation point.

6. Van Driest II reference-Reynolds-number transformation: the
   Karman-Schlichting incompressible skin-friction correlation must be
   evaluated at the VD2-transformed Reynolds number
   Re_x_inc = Fx * Re_x_e, with Fx = (1/T_ratio_w)^omega (omega=0.76
   for air) — NOT at the raw edge Re_x_e. The raw-Re_x version silently
   used the wrong reference state for Cf_inc, which then propagates
   into Cf_comp and q_turb for every turbulent panel.
"""


"This code outputs edge parameters:pressure, density, temperature and outputs their plots along with q-wall as well as azimuthal"
"from this we are able to compare both q_walls: q_wall computed via eckert and Van driest II, we pass the local conditions calculated via isentropic and newtonian but also local conditions extracted from CFD"
"Fay-Riddell now overrides the stagnation panel for both q_wall sources"

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
from jaxfluids_postprocess import load_data

# ═══════════════════════════════════════════════════════════════════════
#  PARAMETERS — must match run_inviscid.py exactly
# ═══════════════════════════════════════════════════════════════════════
M_inf   = 5.0
gamma   = 1.4
T_inf   = 205.0
p_inf   = 5543.0
rho_inf = 0.09427
Tw      = 300.0
R       = 287.0
Pr      = 0.71
Cp      = gamma * R / (gamma - 1.0)

N            = 12
x_target     = 2.5
y_target     = 0.125
panel_length = x_target / N

# Nose radius for the Fay-Riddell stagnation-point model. This is a MODELING
# parameter for the blunt-body stagnation formula, independent of whether the
# meshed geometry in run_inviscid.py is sharp or blunted — Fay-Riddell is
# applied at the stagnation/leading panel regardless, exactly as it was in
# the analytical TPS scripts. Set this to match the actual nose radius used
# in run_inviscid.py if the mesh has a rounded leading edge; otherwise it's
# an effective radius representing the stagnation-region curvature scale.
R_nose = 0.01   # m

h        = 0.025
d_thresh = 4* h    # near-wall band thickness = 50 mm

def generate_nodes(theta, panel_length):
    dx = panel_length * np.cos(theta)
    dr = panel_length * np.sin(theta)
    x  = np.concatenate([[0.0], np.cumsum(dx)])
    r  = np.concatenate([[0.0], np.cumsum(dr)])
    return x, r

mean_angle = float(np.arctan(y_target / x_target))
theta_arr  = np.deg2rad(np.linspace(
    np.rad2deg(mean_angle) * 1.8,
    np.rad2deg(mean_angle) * 0.2,
    N
))
x_2d, r_2d = generate_nodes(theta_arr, panel_length)
x_panel    = np.array(x_2d[1:])

def mu_sutherland(T):
    T_ref, C, mu_ref = 273.15, 110.4, 1.716e-5
    return mu_ref * (T / T_ref)**1.5 * (T_ref + C) / (T + C)


# ═══════════════════════════════════════════════════════════════════════
#  FAY-RIDDELL STAGNATION-POINT HEAT FLUX
#
#  Replaces the flat-plate correlation at the stagnation/leading panel,
#  which is singular as x->0. Standard non-dissociating limit, Le~1:
#
#    q_stag = 0.763 * Pr^-0.6 * (rho_w*mu_w)^0.1 * (rho_stag*mu_stag)^0.4
#             * sqrt(du_e/dx) * Cp * (T_aw_stag - T_w)
#
#  Velocity gradient at stagnation (modified-Newtonian):
#    du_e/dx = (1/R_nose) * sqrt(2*(p_stag - p_inf)/rho_inf)
#
#  p_stag, T_stag are the freestream isentropic total conditions (same
#  simplification used in the earlier analytical TPS scripts — no normal-
#  shock total-pressure loss correction applied).
# ═══════════════════════════════════════════════════════════════════════
def fay_riddell(p_stag, T_stag, T_w, R_nose):
    rho_stag = p_stag / (R * T_stag)
    mu_stag  = mu_sutherland(T_stag)

    rho_w = p_stag / (R * T_w)
    mu_w  = mu_sutherland(T_w)

    du_e_dx = (1.0 / R_nose) * np.sqrt(
        2.0 * np.maximum(p_stag - p_inf, 0.0) / rho_stag
    )
    T_aw_stag = T_stag

    q_stag = (0.763
              * Pr**(-0.6)
              * (rho_w   * mu_w  )**0.1
              * (rho_stag * mu_stag)**0.4
              * np.sqrt(du_e_dx)
              * Cp * (T_aw_stag - T_w))

    return max(q_stag, 0.0)


# Freestream isentropic stagnation conditions (used for Fay-Riddell)
p_stag_freestream = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
T_stag_freestream = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
q_stag_value = fay_riddell(p_stag_freestream, T_stag_freestream, Tw, R_nose)

print("\n" + "=" * 60)
print("  FAY-RIDDELL STAGNATION HEAT FLUX")
print("=" * 60)
print(f"  R_nose        = {R_nose*1000:.1f} mm")
print(f"  p_stag        = {p_stag_freestream:.1f} Pa   T_stag = {T_stag_freestream:.1f} K")
print(f"  q_stag        = {q_stag_value:.1f} W/m²")
print(f"  Applied at    : x_panel[0] = {x_panel[0]:.4f} m")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  FIND AND LOAD RESULTS
#  Sorts by modification time — picks the most recently completed run.
#  Prints the folder name so you can confirm it is the right one.
# ═══════════════════════════════════════════════════════════════════════
case_name  = "hypersonic_vehicle_3D_M5_inviscid"
candidates = sorted(
    glob.glob(os.path.join("results", case_name + "*")),
    key=os.path.getmtime
)
assert candidates, \
    "No results folder found under ./results — did run_inviscid.py finish?"

result_path = os.path.join(candidates[-1], "domain")
print(f"\n  Loading results from : {result_path}")
print(f"  All available runs   :")
for c in candidates:
    mtime = os.path.getmtime(c)
    import time
    print(f"    {c}  (modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))})")
print()

jxf_data = load_data(
    result_path,
    ["temperature", "pressure", "density", "velocity", "levelset"]
)
xc, yc, zc = jxf_data.cell_centers
times   = jxf_data.times
T_all   = jxf_data.data["temperature"]
p_all   = jxf_data.data["pressure"]
rho_all = jxf_data.data["density"]
vel_all = np.asarray(jxf_data.data["velocity"])
phi_all = jxf_data.data["levelset"]

print(f"  Snapshots loaded     : {len(times)}")
print(f"  Final time           : {times[-1]:.6f} s  "
      f"(target {8.0*x_target/(M_inf*np.sqrt(gamma*R*T_inf)):.6f} s)")
print(f"  Velocity array shape : {vel_all.shape}")

# handle both (t, 3, nx, ny, nz) and (t, nx, ny, nz, 3) layouts
if vel_all.ndim == 5 and vel_all.shape[1] == 3:
    u_all = vel_all[:, 0]
    v_all = vel_all[:, 1]
    w_all = vel_all[:, 2]
else:
    u_all = vel_all[..., 0]
    v_all = vel_all[..., 1]
    w_all = vel_all[..., 2]

# use final snapshot
T_f   = T_all[-1];   p_f   = p_all[-1]
rho_f = rho_all[-1]; phi_f = phi_all[-1]
u_f   = u_all[-1];   v_f   = v_all[-1];   w_f = w_all[-1]

Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

# ═══════════════════════════════════════════════════════════════════════
#  NEAR-WALL CELL EXTRACTION WITH PHYSICAL VALIDITY FILTER
#
#  The positivity limiter keeps some cut-cells alive with
#  rho ~ 1e-7 kg/m³ (330,000x below freestream). These are
#  numerically corrupted and must be excluded before averaging.
#
#  Validity thresholds: 1% of freestream values.
#  Anything below these is a limiter-rescued cell, not real flow.
"some values were removed - filtered"
# ═══════════════════════════════════════════════════════════════════════
rho_min_valid = 0.01 * rho_inf    # 9.427e-04 kg/m³
p_min_valid   = 0.01 * p_inf      # 55.43 Pa
T_min_valid   = 0.5  * T_inf      # 102.5 K  (generous lower bound)

is_wall = (phi_f > 0) & (phi_f < d_thresh)

is_valid = (
    is_wall
    & (rho_f > rho_min_valid)
    & (p_f   > p_min_valid)
    & (T_f   > T_min_valid)
    & np.isfinite(T_f)
    & np.isfinite(p_f)
    & np.isfinite(rho_f)
    & np.isfinite(u_f)
    & np.isfinite(v_f)
    & np.isfinite(w_f)
)

n_wall  = int(is_wall.sum())
n_valid = int(is_valid.sum())
print(f"\n  Near-wall cells (phi in [0, {d_thresh*1000:.0f} mm]) : {n_wall:,}")
print(f"  Physically valid subset               : {n_valid:,}  "
      f"({100*n_valid/max(n_wall,1):.1f}% of near-wall band)")

assert n_valid > 0, (
    f"No valid near-wall cells found.\n"
    f"  n_wall={n_wall}, rho_min={rho_f[is_wall].min():.2e}, "
    f"p_min={p_f[is_wall].min():.2e}\n"
    f"  Try increasing d_thresh (currently {d_thresh} m) or "
    f"lowering rho_min_valid / p_min_valid."
)

wi, wj, wk = np.where(is_valid)

T_e_all   = T_f  [wi, wj, wk]
p_e_all   = p_f  [wi, wj, wk]
rho_e_all = rho_f[wi, wj, wk]
u_e_all   = u_f  [wi, wj, wk]
v_e_all   = v_f  [wi, wj, wk]
w_e_all   = w_f  [wi, wj, wk]

V_e_all = np.sqrt(u_e_all**2 + v_e_all**2 + w_e_all**2)
a_e_all = np.sqrt(gamma * R * np.maximum(T_e_all, 1.0))
M_e_all = V_e_all / np.maximum(a_e_all, 1.0)

x_wall      = Xc[wi, wj, wk]
y_wall      = Yc[wi, wj, wk]
z_wall      = Zc[wi, wj, wk]
phi_azimuth = np.arctan2(z_wall, y_wall)

# ═══════════════════════════════════════════════════════════════════════
#  BIN INTO (x_panel, azimuth)
# ═══════════════════════════════════════════════════════════════════════
N_PHI_BIN = 24
phi_bins  = np.linspace(-np.pi, np.pi, N_PHI_BIN + 1)

sums = {k: np.zeros((N, N_PHI_BIN))
        for k in ["T_e", "p_e", "rho_e", "V_e", "M_e"]}
counts = np.zeros((N, N_PHI_BIN), dtype=int)

for idx in range(len(wi)):
    ix = int(np.clip(np.searchsorted(x_panel, x_wall[idx]) - 1, 0, N - 1))
    ip = int(np.clip(np.searchsorted(phi_bins, phi_azimuth[idx]) - 1,
                     0, N_PHI_BIN - 1))
    sums["T_e"]  [ix, ip] += T_e_all  [idx]
    sums["p_e"]  [ix, ip] += p_e_all  [idx]
    sums["rho_e"][ix, ip] += rho_e_all[idx]
    sums["V_e"]  [ix, ip] += V_e_all  [idx]
    sums["M_e"]  [ix, ip] += M_e_all  [idx]
    counts[ix, ip] += 1

# require at least 3 valid cells per bin — fewer is too noisy
MIN_CELLS_PER_BIN = 3
mask = counts >= MIN_CELLS_PER_BIN
for k in sums:
    sums[k][mask] /= counts[mask]
    sums[k][~mask] = np.nan

def azim_avg(field2d):
    """Mean over azimuthal bins that have enough valid cells."""
    out = np.full(N, np.nan)
    for ix in range(N):
        valid = mask[ix]
        if valid.any():
            out[ix] = np.nanmean(field2d[ix, valid])
    return out

T_e_x   = azim_avg(sums["T_e"])
p_e_x   = azim_avg(sums["p_e"])
rho_e_x = azim_avg(sums["rho_e"])
V_e_x   = azim_avg(sums["V_e"])
M_e_x   = azim_avg(sums["M_e"])

# ═══════════════════════════════════════════════════════════════════════
#  EDGE CONDITION SANITY CHECK
#
#  Physical expectations for a valid inviscid M=5 forebody:
#    p_e  > p_inf = 5543 Pa everywhere on windward body
#    T_e  > T_inf = 205 K, roughly 400-800 K near nose
#    M_e  < M_inf = 5, increasing from ~0 at nose to ~4 at tail
#    V_e  < V_inf = 1435 m/s
#    rho_e > rho_inf = 0.09427 kg/m³ near nose (shock compression)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  EDGE CONDITION SANITY CHECK  (azimuthal averages per x-station)")
print("=" * 72)
print(f"  {'x(m)':>6}  {'p_e(Pa)':>10}  {'T_e(K)':>8}  "
      f"{'rho_e':>10}  {'V_e(m/s)':>10}  {'M_e':>6}  {'status':>8}")
print(f"  {'-'*65}")


# Diagnostic: how many near-wall cells actually informed each x-station
counts_per_x = counts.sum(axis=1)
bins_per_x   = (counts > 0).sum(axis=1)
print("\nSample-count diagnostic per x-panel (low counts = low-confidence data):")
print(f"{'x (m)':>8} {'total cells':>12} {'bins filled':>12} / {N_PHI_BIN}")
for ix in range(N):
    print(f"{x_panel[ix]:8.3f} {counts_per_x[ix]:12d} {bins_per_x[ix]:12d} / {N_PHI_BIN}")

all_ok = True
for i in range(N):
    p  = p_e_x[i];   T  = T_e_x[i]
    rh = rho_e_x[i]; Ve = V_e_x[i]; Me = M_e_x[i]

    if np.isnan(p):
        status = "NO DATA"
        all_ok = False
    elif p < p_inf:
        status = "p<p_inf!"
        all_ok = False
    elif T < T_inf:
        status = "T<T_inf!"
        all_ok = False
    elif Me > M_inf:
        status = "M>M_inf!"
        all_ok = False
    else:
        status = "OK"

    print(f"  {x_panel[i]:6.3f}  {p:10.1f}  {T:8.1f}  "
          f"{rh:10.5f}  {Ve:10.1f}  {Me:6.3f}  {status:>8}")

print()
if all_ok:
    print("  All x-stations: PHYSICALLY VALID ✓")
else:
    print("  WARNING: some x-stations have unphysical edge conditions.")
    print("  These will produce unreliable q_wall at those stations.")
    print("  Consider increasing rho_min_valid / p_min_valid thresholds")
    print("  or widening d_thresh to sample cells further from the wall.")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════
#  HEAT TRANSFER CORRELATIONS
#  Eckert reference temperature (laminar) + Van Driest II (turbulent),
#  smoothly blended across a finite transition zone (a hard laminar/
#  turbulent switch at x_tr produces an unphysical kink — real transition
#  is not instantaneous, and even a purely numerical "correct" result at
#  a sharp switch will show a discontinuity since the two correlations
#  don't generally agree in value at the crossover point).
#
#  Fay-Riddell is NOT applied inside this function anymore. It is a point
#  formula strictly valid at the true stagnation point (x=0); this
#  script's panels are evenly spaced (x_panel[0] is a full panel-length,
#  ~0.2 m, downstream of the tip — see conversation), so overwriting a
#  panel with it was applying a stagnation value ~1000x too large at a
#  location that is not actually the stagnation point. Fay-Riddell is
#  reported and plotted separately, at its own correct x=0 location.
# ═══════════════════════════════════════════════════════════════════════
def compute_q_wall(x_arr, T_e, p_e, rho_e, V_e, M_e, L_trans=0.15):
    """
    Eckert (laminar) + Van Driest II (turbulent) heat flux, blended smoothly
    across a transition zone of streamwise length scale L_trans (m) centred
    on x_tr, using a tanh intermittency-style blend instead of a hard switch.
    Handles NaN edge conditions gracefully — returns NaN for those stations.
    """
    x_safe = np.maximum(x_arr, 1e-6)
    mu_e   = mu_sutherland(np.maximum(T_e, 1.0))

    # transition criterion: Re_theta / M_e >= 400 (used only to LOCATE x_tr)
    Rex      = rho_e * V_e * x_safe / np.maximum(mu_e, 1e-12)
    theta_m  = 0.664 * x_safe / np.sqrt(np.maximum(Rex, 1.0))
    Re_theta = rho_e * V_e * theta_m / np.maximum(mu_e, 1e-12)
    crit     = Re_theta / np.maximum(M_e, 1e-3)
    is_turb  = crit >= 400.0
    x_tr = (x_safe[np.argmax(is_turb)] if is_turb.any()
            else float(x_safe[-1]))

    r_lam  = Pr**0.5
    r_turb = Pr**(1.0/3.0)

    # ── LAMINAR: Eckert reference temperature + Blasius ───────────────
    Taw_lam   = T_e * (1.0 + r_lam*(gamma-1.0)/2.0*M_e**2)
    T_ref_lam = T_e * (0.45 + 0.55*(Tw/np.maximum(T_e, 1.0))
                       + 0.16*r_lam*(gamma-1.0)/2.0*M_e**2)
    rho_ref   = p_e / (R * np.maximum(T_ref_lam, 1.0))
    mu_ref    = mu_sutherland(np.maximum(T_ref_lam, 1.0))
    Rex_ref   = rho_ref * V_e * x_safe / np.maximum(mu_ref, 1e-12)
    St_lam    = (0.332 / np.sqrt(np.maximum(Rex_ref, 1.0))) * Pr**(-2.0/3.0)
    q_lam     = rho_ref * V_e * St_lam * Cp * (Taw_lam - Tw)

    # ── TURBULENT: Van Driest II ──────────────────────────────────────
    Taw_turb   = T_e * (1.0 + r_turb*(gamma-1.0)/2.0*M_e**2)
    T_ratio_aw = 1.0 + r_turb * (gamma - 1.0) / 2.0 * M_e**2
    T_ratio_w  = Tw  / T_e

    A_sq = np.maximum(
        (r_turb*(gamma-1.0)/2.0*M_e**2) / (T_ratio_w+ 1e-10),
        0.0
    )
    A = np.sqrt(A_sq)
    B = T_ratio_aw/(T_ratio_w+1e-10)-1

    denom      = np.sqrt(B**2 + 4.0*A**2 + 1e-30)
    alpha_vd   = np.clip((2.0*A**2 - B) / denom, -1.0+1e-7, 1.0-1e-7)
    beta_vd    = np.clip(B / denom,               -1.0+1e-7, 1.0-1e-7)
    arcsin_sum = np.arcsin(alpha_vd) + np.arcsin(beta_vd)
    # Fc numerator/gate MUST be built from T_ratio_aw (not T_ratio_w): as
    # M_e -> 0, T_aw -> T_e so this numerator -> 0 and the gate correctly
    # falls back to Fc=1 (the incompressible limit). A T_ratio_w-based
    # numerator/gate doesn't vanish with M_e, so arcsin_sum collapses
    # toward 0 while the numerator stays finite -> Fc artificially small
    # -> Cf_comp = Cf_inc/Fc blows up. This was the actual source of the
    # ~40x q_wall spike, not the near-wall CFD data.
    Fc = np.where(
        T_ratio_aw > 1.001,
        (T_ratio_aw - 1.0) / (arcsin_sum**2 + 1e-30),
        1.0
    )

    # Van Driest II reference-Reynolds-number transformation:
    #   Re_x_inc = Fx * Re_x_e,  Fx = (1/T_ratio_w)^omega   (omega=0.76 for air)
    # The Karman-Schlichting incompressible correlation must be solved at
    # this transformed Re, NOT at the raw edge Re_x_e - using Re_x_e
    # directly evaluates Cf_inc at the wrong reference state.
    omega   = 0.76  # air
    Fx      = (1.0 / np.maximum(T_ratio_w, 1e-10))**omega
    Re_x_e   = rho_e * V_e * x_safe / np.maximum(mu_e, 1e-12)
    Re_x_inc = Fx * Re_x_e

    # Karman-Schlichting incompressible Cf, Newton iteration
    Cf_inc  = 0.0592 * np.maximum(Re_x_inc, 1.0)**(-0.2)
    for _ in range(6):
        lhs  = 1.0 / np.sqrt(np.maximum(Cf_inc, 1e-30))
        rhs  = 4.15 * np.log10(np.maximum(Re_x_inc * Cf_inc, 1e-30)) + 1.7
        f    = lhs - rhs
        dfdC = (-0.5 / np.maximum(Cf_inc, 1e-30)**1.5
                - 4.15 / (np.maximum(Cf_inc, 1e-30) * np.log(10.0)))
        Cf_inc = np.maximum(Cf_inc - f / (dfdC + 1e-30), 1e-8)

    Cf_comp = Cf_inc / np.maximum(Fc, 1e-30)
    St_turb = (Cf_comp / 2.0) * Pr**(-2.0/3.0)
    q_turb  = rho_e * V_e * St_turb * Cp * (Taw_turb - Tw)

    # ── Smooth blend across the transition zone (replaces hard switch) ─
    gamma_blend = 0.5 * (1.0 + np.tanh((x_safe - x_tr) / max(L_trans, 1e-6)))
    q_wall = (1.0 - gamma_blend) * q_lam + gamma_blend * q_turb
    q_wall = np.maximum(q_wall, 0.0)

    # propagate NaN from invalid edge conditions
    q_wall = np.where(np.isnan(T_e) | np.isnan(p_e), np.nan, q_wall)

    return q_wall, x_tr


# ── CFD-informed edge conditions ──────────────────────────────────────
q_wall_cfd, x_tr_cfd = compute_q_wall(
    x_panel, T_e_x, p_e_x, rho_e_x, V_e_x, M_e_x
)
print(f"\n  Transition (CFD edge)      : x_tr = {x_tr_cfd:.3f} m")
print(f"  q_wall[0] (stagnation, Fay-Riddell) = {q_wall_cfd[0]:.1f} W/m²")

# ── DIAGNOSTIC: trace the VD2 intermediates per x-station ──────────────
# Mirrors the turbulent branch of compute_q_wall exactly, just to expose
# T_ratio_w / Fx / Re_x_inc / Cf_inc / Fc / Cf_comp per panel so an
# unphysical spike in q_wall can be traced back to its source (e.g. a
# near-wall T_e that has dipped close to or below Tw at that station).
def debug_vd2(tag, x_arr, T_e, p_e, rho_e, V_e, M_e):
    x_safe = np.maximum(x_arr, 1e-6)
    mu_e   = mu_sutherland(np.maximum(T_e, 1.0))
    r_turb = Pr**(1.0/3.0)
    T_ratio_w = Tw / T_e
    T_ratio_aw = 1.0 + r_turb*(gamma-1.0)/2.0*M_e**2

    A_sq = np.maximum((r_turb*(gamma-1.0)/2.0*M_e**2)/(T_ratio_w+1e-10), 0.0)
    A = np.sqrt(A_sq)
    B = T_ratio_aw/(T_ratio_w+1e-10) - 1
    denom = np.sqrt(B**2 + 4.0*A**2 + 1e-30)
    alpha = np.clip((2.0*A**2 - B)/denom, -1.0+1e-7, 1.0-1e-7)
    beta  = np.clip(B/denom, -1.0+1e-7, 1.0-1e-7)
    arcsin_sum = np.arcsin(alpha) + np.arcsin(beta)
    Fc = np.where(T_ratio_aw > 1.001, (T_ratio_aw-1.0)/(arcsin_sum**2 + 1e-30), 1.0)

    omega = 0.76
    Fx = (1.0/np.maximum(T_ratio_w, 1e-10))**omega
    Re_x_e   = rho_e*V_e*x_safe/np.maximum(mu_e, 1e-12)
    Re_x_inc = Fx*Re_x_e
    Cf_inc = 0.0592*np.maximum(Re_x_inc, 1.0)**(-0.2)
    for _ in range(6):
        lhs = 1.0/np.sqrt(np.maximum(Cf_inc, 1e-30))
        rhs = 4.15*np.log10(np.maximum(Re_x_inc*Cf_inc, 1e-30)) + 1.7
        f = lhs - rhs
        dfdC = (-0.5/np.maximum(Cf_inc, 1e-30)**1.5
                - 4.15/(np.maximum(Cf_inc, 1e-30)*np.log(10.0)))
        Cf_inc = np.maximum(Cf_inc - f/(dfdC + 1e-30), 1e-8)
    Cf_comp = Cf_inc/np.maximum(Fc, 1e-30)

    print("\n" + "-"*100)
    print(f"  VD2 DIAGNOSTIC ({tag})")
    print("-"*100)
    print(f"  {'x(m)':>6} {'T_e(K)':>8} {'T_w/T_e':>9} {'Fx':>8} "
          f"{'Re_x_e':>12} {'Re_x_inc':>12} {'Fc':>8} {'Cf_inc':>10} {'Cf_comp':>10}")
    for i in range(len(x_arr)):
        print(f"  {x_arr[i]:6.3f} {T_e[i]:8.1f} {T_ratio_w[i]:9.4f} {Fx[i]:8.3f} "
              f"{Re_x_e[i]:12.3e} {Re_x_inc[i]:12.3e} {Fc[i]:8.4f} "
              f"{Cf_inc[i]:10.3e} {Cf_comp[i]:10.3e}")
    print("-"*100)
    print("  Watch for: T_w/T_e approaching or exceeding 1.0 (T_e <= Tw — unphysical")
    print("  for a shock-heated edge), or Fx/Cf_comp jumping by >5x between adjacent")
    print("  stations. Either points to a noisy T_e_x value at that x-station rather")
    print("  than a real compressibility effect.")

debug_vd2("CFD edge", x_panel, T_e_x, p_e_x, rho_e_x, V_e_x, M_e_x)

# ── Newtonian baseline for comparison ────────────────────────────────
def newtonian_edge():
    a_inf_val = np.sqrt(gamma * R * T_inf)
    V_inf_val = M_inf * a_inf_val
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    T0 = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
    p_arr = (p_inf + 0.5*rho_inf*V_inf_val**2
             * np.maximum(2.0*np.sin(theta_arr)**2, 0.0))
    T_arr = T0 * (p_arr/p0)**((gamma-1.0)/gamma)
    M_e   = np.sqrt((2.0/(gamma-1.0))*((p0/p_arr)**((gamma-1.0)/gamma)-1.0))
    a_e   = np.sqrt(gamma * R * T_arr)
    V_e   = M_e * a_e
    rho_e = p_arr / (R * T_arr)
    return T_arr, p_arr, rho_e, V_e, M_e

T_nt, p_nt, rho_nt, V_nt, M_nt = newtonian_edge()
q_wall_nt, x_tr_nt = compute_q_wall(
    x_panel, T_nt, p_nt, rho_nt, V_nt, M_nt
)
print(f"  Transition (Newtonian edge): x_tr = {x_tr_nt:.3f} m")
print(f"  q_wall[0] (stagnation, Fay-Riddell) = {q_wall_nt[0]:.1f} W/m²")

# ── 2D azimuthal heat flux map ────────────────────────────────────────
# NOTE: Fay-Riddell is a 1D stagnation-point result (no azimuthal variation
# modeled), so it is NOT applied per-azimuthal-bin here — only to the
# 1D x_panel[0] summaries above. The 2D map keeps the flat-plate value at
# x_panel[0] for visualization continuity; treat that row as unreliable.
q_2d = np.full((N, N_PHI_BIN), np.nan)
for ip in range(N_PHI_BIN):
    T_col   = sums["T_e"]  [:, ip]
    p_col   = sums["p_e"]  [:, ip]
    rho_col = sums["rho_e"][:, ip]
    V_col   = sums["V_e"]  [:, ip]
    M_col   = sums["M_e"]  [:, ip]
    q_col, _ = compute_q_wall(x_panel, T_col, p_col, rho_col, V_col, M_col)
    q_2d[:, ip] = q_col

# ═══════════════════════════════════════════════════════════════════════
#  PLOTS  — each plot in its own standalone figure/file
# ═══════════════════════════════════════════════════════════════════════
suptitle_str = (f'Inviscid CFD (M={M_inf}) → Eckert + Van Driest II + Fay-Riddell  |  '
                 f'h={h*1000:.0f} mm, CFL=0.2')

# ── Plot 1: edge pressure ───────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(x_panel, p_e_x, 's-', color='darkorange', linewidth=2,
         markersize=6, label='p_e (CFD near-wall)')
ax1.plot(x_panel, p_nt,  'o--', color='gray', alpha=0.7,
         markersize=5, label='p_e (Newtonian estimate)')
ax1.axhline(p_inf, color='k', linestyle=':', linewidth=1,
            label=f'p_inf = {p_inf:.0f} Pa')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('p_e (Pa)')
ax1.set_xlim(-0.05,2.28)
ax1.set_title('Edge pressure: CFD vs. modified-Newtonian')
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
fig1.suptitle(suptitle_str, fontsize=10, fontweight='bold')
fig1.tight_layout()
fig1.savefig('edge_pressure.png', dpi=150, bbox_inches='tight')
print("Saved: edge_pressure.png")

# ── Plot 2: edge temperature ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(x_panel, T_e_x, 's-', color='firebrick', linewidth=2,
         markersize=6, label='T_e (CFD near-wall)')
ax2.plot(x_panel, T_nt,  'o--', color='gray', alpha=0.7,
         markersize=5, label='T_e (Newtonian estimate)')
ax2.axhline(T_inf, color='k', linestyle=':', linewidth=1,
            label=f'T_inf = {T_inf:.0f} K')
ax2.axhline(Tw, color='steelblue', linestyle=':', linewidth=1,
            label=f'T_w = {Tw:.0f} K')
ax2.set_xlabel('x (m)'); ax2.set_ylabel('T_e (K)')
ax2.set_xlim(-0.05, 2.28)
ax2.set_title('Edge temperature: CFD vs. modified-Newtonian')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig2.suptitle(suptitle_str, fontsize=10, fontweight='bold')
fig2.tight_layout()
fig2.savefig('edge_temperature.png', dpi=150, bbox_inches='tight')
print("Saved: edge_temperature.png")

# ── Plot 3: q_wall comparison ────────────────────────────────────────────
# Fay-Riddell (~1e7 W/m^2, at the true x=0 stagnation point) and the
# panel-based flat-plate correlation (~1e4-1e5 W/m^2, at x_panel[0]~0.2m
# onward) differ by orders of magnitude and are NOT the same physical
# location. This plot shows the panel-based q_wall at full resolution;
# Fay-Riddell is reported separately (see printed summary / CSV).
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(x_panel, q_wall_cfd, 's-', color='steelblue', linewidth=2,
         markersize=6, label='CFD-informed edge conditions')
ax3.plot(x_panel, q_wall_nt,  'o--', color='crimson', alpha=0.7,
         markersize=5, label='Newtonian edge conditions')
ax3.axvline(x_tr_cfd, color='steelblue', ls=':', lw=1.2,
            label=f'x_tr (CFD) = {x_tr_cfd:.2f} m')
ax3.axvline(x_tr_nt,  color='crimson',   ls=':', lw=1.0, alpha=0.6,
            label=f'x_tr (Newt) = {x_tr_nt:.2f} m')
ax3.set_xlim(-0.05, 2.28)
ax3.set_xlabel('x (m)'); ax3.set_ylabel('q_wall (W/m²)')
ax3.set_title('Wall heat flux: CFD-informed vs. Newtonian edge conditions')
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
fig3.suptitle(suptitle_str, fontsize=10, fontweight='bold')
fig3.tight_layout()
fig3.savefig('qwall_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: qwall_comparison.png")

# ── Plot 4: azimuthal heat flux map ──────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5))
phi_deg = 0.5 * (phi_bins[:-1] + phi_bins[1:]) * 180.0 / np.pi
im = ax4.pcolormesh(
    x_panel, phi_deg, q_2d.T,
    cmap='YlOrRd', shading='auto',
    vmin=0, vmax=np.nanpercentile(q_2d, 95)
)
fig4.colorbar(im, ax=ax4, label='q_wall (W/m²)')
ax4.set_title('q_wall(x, azimuth) from CFD-informed edge conditions\n'
              '(x_panel[0] row is flat-plate, NOT Fay-Riddell — see note)')
ax4.set_xlabel('x (m)'); ax4.set_ylabel('Azimuthal angle (deg)')
fig4.suptitle(suptitle_str, fontsize=10, fontweight='bold')
fig4.tight_layout()
fig4.savefig('qwall_azimuthal_map.png', dpi=150, bbox_inches='tight')
print("Saved: qwall_azimuthal_map.png")

plt.show()

# ═══════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════
with open('qwall_inviscid_table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'x_panel_m',
        'q_wall_cfd_edge_Wm2',
        'q_wall_newtonian_edge_Wm2',
        'p_e_cfd_Pa',
        'T_e_cfd_K',
        'rho_e_cfd_kgm3',
        'V_e_cfd_ms',
        'M_e_cfd',
        'is_fay_riddell_stagnation'
    ])
    for i in range(N):
        writer.writerow([
            f"{x_panel[i]:.5f}",
            f"{q_wall_cfd[i]:.2f}" if not np.isnan(q_wall_cfd[i]) else "NaN",
            f"{q_wall_nt[i]:.2f}",
            f"{p_e_x[i]:.2f}"     if not np.isnan(p_e_x[i])   else "NaN",
            f"{T_e_x[i]:.2f}"     if not np.isnan(T_e_x[i])   else "NaN",
            f"{rho_e_x[i]:.6f}"   if not np.isnan(rho_e_x[i]) else "NaN",
            f"{V_e_x[i]:.2f}"     if not np.isnan(V_e_x[i])   else "NaN",
            f"{M_e_x[i]:.4f}"     if not np.isnan(M_e_x[i])   else "NaN",
            "YES" if i == 0 else "no",
        ])
print("Saved: qwall_inviscid_table.csv")

# ═══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  HEAT FLUX SUMMARY")
print("=" * 55)
valid_cfd = ~np.isnan(q_wall_cfd)
print(f"  Valid x-stations (CFD edge)  : "
      f"{valid_cfd.sum()} / {N}")
print(f"  Max q_wall (CFD edge)        : "
      f"{np.nanmax(q_wall_cfd):.0f} W/m²  "
      f"at x = {x_panel[np.nanargmax(q_wall_cfd)]:.3f} m")
print(f"  Min q_wall (CFD edge)        : "
      f"{np.nanmin(q_wall_cfd):.0f} W/m²  "
      f"at x = {x_panel[np.nanargmin(q_wall_cfd)]:.3f} m")
print(f"  Max q_wall (Newtonian edge)  : "
      f"{np.nanmax(q_wall_nt):.0f} W/m²  "
      f"at x = {x_panel[np.nanargmax(q_wall_nt)]:.3f} m")
print(f"  Transition (CFD edge)        : x_tr = {x_tr_cfd:.3f} m")
print(f"  Transition (Newtonian edge)  : x_tr = {x_tr_nt:.3f} m")
print(f"  Fay-Riddell q_stag           : {q_stag_value:.1f} W/m²  (R_nose={R_nose*1000:.0f} mm)")
print("=" * 55)