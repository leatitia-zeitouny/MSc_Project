"""
flow_viz_correct_fixed.py
──────────────────────────────────────────────────────────────────────────────
Fix vs. flow_viz_correct.py: is_solid (real body) and is_corrupt (bad
fluid-region cut-cells near the tail) were both being masked to NaN and
painted the SAME dark grey via cmap.set_bad(BODY_COLOR). That makes any
patch of corrupted cells near the trailing edge visually indistinguishable
from the hull - which is exactly what produced the boxy "engine nacelle"
artifact behind the tail in your last plot.

Fix: cmap.set_bad now goes fully TRANSPARENT (not body-colored), so bad
cells become honest gaps showing the dark background - not disguised as
geometry. The real body silhouette is drawn separately, underneath, from
the analytic r_body(x) profile (fill_body_side / fill_body_cross) - so the
hull shape you see is always the true geometry, never a blocky artifact of
which cells happened to get flagged bad.
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from jaxfluids_postprocess import load_data

# ═══════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
M_inf   = 5.0
gamma   = 1.4
T_inf   = 205.0
p_inf   = 5543.0
rho_inf = 0.09427
R       = 287.0
Pr      = 0.71
Cp      = gamma * R / (gamma - 1.0)
Tw      = 300.0

N            = 12
x_target     = 2.5
y_target     = 0.125
panel_length = x_target / N
h        = 0.025
X_MIN, X_MAX = -0.4, 3.0
Y_MIN, Y_MAX = -0.5, 0.5
Z_MIN, Z_MAX = -0.5, 0.5
NX = round((X_MAX - X_MIN) / h)
NY = round((Y_MAX - Y_MIN) / h)
NZ = round((Z_MAX - Z_MIN) / h)

mean_angle = float(np.arctan(y_target / x_target))
theta_arr  = np.deg2rad(np.linspace(
    np.rad2deg(mean_angle) * 1.8,
    np.rad2deg(mean_angle) * 0.2, N
))
dx_arr  = panel_length * np.cos(theta_arr)
dr_arr  = panel_length * np.sin(theta_arr)
x_nodes = np.concatenate([[0.0], np.cumsum(dx_arr)])
r_nodes = np.concatenate([[0.0], np.cumsum(dr_arr)])
x_end   = float(x_nodes[-1])
x_panel = x_nodes[1:]
cs = CubicSpline(x_nodes, r_nodes, bc_type=((1, 0.0), 'not-a-knot'))

def r_body(x):
    return np.where(x > x_end, 0.0, cs(np.clip(x, 0.0, x_end)))

def mu_sutherland(T):
    T_ref, C, mu_ref = 273.15, 110.4, 1.716e-5
    return mu_ref * (T / T_ref)**1.5 * (T_ref + C) / (T + C)

# ═══════════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════════
case_name  = "hypersonic_vehicle_3D_M5_inviscid"
candidates = sorted(glob.glob(os.path.join("results", case_name + "*")),
                    key=os.path.getmtime)
assert candidates, "No results folder found."
result_path = os.path.join(candidates[-1], "domain")
print(f"Loading: {result_path}")

jxf_data = load_data(result_path,
    ["temperature", "pressure", "density", "velocity", "levelset"])
xc, yc, zc = jxf_data.cell_centers
times   = jxf_data.times
T_f     = jxf_data.data["temperature"][-1]
p_f     = jxf_data.data["pressure"][-1]
rho_f   = jxf_data.data["density"][-1]
phi_f   = jxf_data.data["levelset"][-1]
vel_all = np.asarray(jxf_data.data["velocity"])

if vel_all.ndim == 5 and vel_all.shape[1] == 3:
    u_f = vel_all[-1,0]; v_f = vel_all[-1,1]; w_f = vel_all[-1,2]
else:
    u_f = vel_all[-1,...,0]; v_f = vel_all[-1,...,1]; w_f = vel_all[-1,...,2]

print(f"Final time: {times[-1]:.5f} s")

# ═══════════════════════════════════════════════════════════════════════
#  MASKING - unchanged detection logic, only the RENDERING changes below
# ═══════════════════════════════════════════════════════════════════════
T_min_valid   = 0.5  * T_inf
p_min_valid   = 0.1  * p_inf
rho_min_valid = 0.01 * rho_inf
# Validated against postprocess_inviscid.py's edge-condition sanity check:
# at d_thresh=4h, the real expansion physics at the tail (x=2.496 m) reads
# T=158.1 K, p=1425.8 Pa, rho=0.0274 - all comfortably above these cutoffs
# (54%, 157%, and 29x above threshold respectively), while genuinely
# corrupted cut-cells there run T as low as 39.7 K, p=0.0 Pa. There's a
# wide, clean gap between "real but extreme" and "numerically dead" - these
# thresholds don't misclassify real expansion as corruption, no change needed.

is_solid    = phi_f < 0
is_corrupt  = (
    (T_f   < T_min_valid)
    | (p_f  < p_min_valid)
    | (rho_f < rho_min_valid)
    | ~np.isfinite(T_f)
    | ~np.isfinite(p_f)
)
is_bad = is_solid | is_corrupt

print(f"\nMasking summary:")
print(f"  Solid cells (phi<0)          : {is_solid.sum():,}")
print(f"  Corrupted cut-cells (fluid)  : {(is_corrupt & ~is_solid).sum():,}  "
      f"<- these will show as GAPS now, not body-grey")
print(f"  Total masked                 : {is_bad.sum():,}  "
      f"({100*is_bad.sum()/phi_f.size:.1f}%)")

V_f = np.sqrt(u_f**2 + v_f**2 + w_f**2)
a_f = np.sqrt(gamma * R * np.maximum(T_f, 1.0))
M_f = V_f / np.maximum(a_f, 1.0)

T_plot   = np.where(is_bad, np.nan, T_f)
p_plot   = np.where(is_bad, np.nan, p_f)
M_plot   = np.where(is_bad, np.nan, M_f)

valid = ~is_bad
print(f"\nValid fluid ranges after masking:")
print(f"  T : {T_f[valid].min():.1f} - {T_f[valid].max():.1f} K")
print(f"  p : {p_f[valid].min():.1f} - {p_f[valid].max():.1f} Pa")
print(f"  M : {M_f[valid].min():.3f} - {M_f[valid].max():.3f}")

T_vmin, T_vmax = T_inf * 0.95, float(np.nanpercentile(T_plot, 99.0))
p_vmin, p_vmax = p_inf * 0.95, float(np.nanpercentile(p_plot, 99.0))
M_vmin, M_vmax = 0.0, M_inf * 1.05

BODY_COLOR    = '#2a2a2a'
BG_COLOR      = "#ffffff"
WARNING_COLOR = '#ff00ff'   # loud magenta - can't be confused with body grey
                            # or with any science colormap value

from matplotlib.colors import ListedColormap

def make_cmap(name):
    """Bad cells (both solid AND corrupt) are transparent in the main
    data layer - the true hull is drawn underneath (fill_body_*), and
    corrupted cells get an explicit magenta overlay on top (see
    overlay_corrupt below) so they're never ambiguous with either."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color='k', alpha=0.0)
    return cmap

def overlay_corrupt(ax, corrupt_slice_2d, hc, vc, zorder=3):
    """Paints ONLY the corrupted-fluid-cell locations in solid magenta,
    on top of everything else. This is the actual fix: corrupted data
    gets a color that cannot be mistaken for the body or for real flow
    data, instead of relying on a transparency trick against a dark
    background that was too subtle to read visually."""
    marker = np.ma.masked_where(~corrupt_slice_2d.T, np.ones_like(corrupt_slice_2d.T, dtype=float))
    ax.pcolormesh(hc, vc, marker, cmap=ListedColormap([WARNING_COLOR]),
                  vmin=1, vmax=1, shading='auto', zorder=zorder)

is_corrupt_only = is_corrupt & ~is_solid   # full 3D array, sliced per view below

cmap_T = make_cmap('RdYlBu_r')
cmap_p = make_cmap('inferno')
cmap_M = make_cmap('RdBu_r')

def xy_slice(f):
    iz = int(np.argmin(np.abs(zc)))
    return f[:, :, iz]

def xz_slice(f):
    iy = int(np.argmin(np.abs(yc)))
    return f[:, iy, :]

def yz_slice(f, x_val=1.0):
    ix = int(np.argmin(np.abs(xc - x_val)))
    return f[ix, :, :]

BODY_EDGE = '#cccccc'

def fill_body_side(ax, zorder=1):
    """KEY FIX: the true hull, drawn from the analytic profile - not from
    which data cells happened to get flagged bad. Draw this BEFORE the
    pcolormesh call (lower zorder) so it sits underneath; transparent NaN
    cells in the pcolormesh let it show through exactly where the real
    solid is, and nowhere else."""
    xp = np.linspace(0.0, x_end, 500)
    rp = r_body(xp)
    ax.fill_between(xp, -rp, rp, color=BODY_COLOR, zorder=zorder)
    ax.plot(xp,  rp, color=BODY_EDGE, lw=1.5, zorder=zorder+1)
    ax.plot(xp, -rp, color=BODY_EDGE, lw=1.5, zorder=zorder+1)
    ax.plot([x_end, x_end], [-rp[-1], rp[-1]], color=BODY_EDGE, lw=1.5, zorder=zorder+1)

def fill_body_cross(ax, x_val=1.0, zorder=1):
    r_val = float(r_body(np.array([x_val]))[0])
    circ_fill = plt.Circle((0, 0), r_val, facecolor=BODY_COLOR, edgecolor='none', zorder=zorder)
    circ_edge = plt.Circle((0, 0), r_val, fill=False, color=BODY_EDGE, lw=1.5, zorder=zorder+1)
    ax.add_patch(circ_fill)
    ax.add_patch(circ_edge)
    return r_val

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 - SIDE VIEW
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
fig.patch.set_facecolor("#ededf1")
fig.suptitle(
    f'Side View  (XY plane, z = 0)  ·  M_inf = {M_inf}  ·  Inviscid Euler  ·  '
    f't = {times[-1]:.5f} s\n'
    f'Grey = hull (analytic geometry)  ·  Magenta = corrupted cut-cells (do not trust)',
    fontsize=11, fontweight='bold', color='black'
)

for ax, (field, label, cmap, vmin, vmax) in zip(axes, [
    (T_plot, 'Temperature (K)', cmap_T, T_vmin, T_vmax),
    (p_plot, 'Pressure (Pa)',   cmap_p, p_vmin, p_vmax),
    (M_plot, 'Mach number',     cmap_M, M_vmin, M_vmax),
]):
    ax.set_facecolor(BG_COLOR)
    fill_body_side(ax, zorder=1)          # true hull, underneath
    sl = xy_slice(field)
    im = ax.pcolormesh(xc, yc, sl.T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto', zorder=2)   # data on top, bad = transparent
    overlay_corrupt(ax, xy_slice(is_corrupt_only), xc, yc, zorder=3)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, color='black', fontsize=10)
    cb.ax.yaxis.set_tick_params(color='black')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')

    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel('x (m)', color='black', fontsize=11)
    ax.set_ylabel('y (m)', color='black', fontsize=11)
    ax.tick_params(colors='black')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')
    ax.set_aspect('equal')
    ax.set_title(label, color='black', fontsize=11, pad=6)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('side_view.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: side_view.png")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 - ALL SLICES
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(21, 14))
fig.patch.set_facecolor("#f8f8fc")
fig.suptitle(
    f'Flow Field Slices  ·  M_inf={M_inf}, Inviscid Euler  ·  t={times[-1]:.5f} s  ·  '
    f'Grid {NX}x{NY}x{NZ}',
    fontsize=13, fontweight='bold', color='black'
)

all_fields = [
    (T_plot, 'Temperature (K)', cmap_T, T_vmin, T_vmax),
    (p_plot, 'Pressure (Pa)',   cmap_p, p_vmin, p_vmax),
    (M_plot, 'Mach number',     cmap_M, M_vmin, M_vmax),
]

for col, (field, label, cmap, vmin, vmax) in enumerate(all_fields):
    for row, (sl_fn, hc, vc, xlbl, ylbl, xlim, ylim, title_sfx, body_fn) in enumerate([
        (xy_slice, xc, yc, 'x (m)', 'y (m)', (X_MIN,X_MAX), (Y_MIN,Y_MAX),
         'XY plane  (z = 0,  side view)', 'side'),
        (xz_slice, xc, zc, 'x (m)', 'z (m)', (X_MIN,X_MAX), (Z_MIN,Z_MAX),
         'XZ plane  (y = 0,  top view)',  'side'),
        (lambda f: yz_slice(f, 1.0), yc, zc, 'y (m)', 'z (m)', (Y_MIN,Y_MAX), (Z_MIN,Z_MAX),
         'YZ plane  (x = 1.0 m)',         'cross'),
    ]):
        ax = axes[row, col]
        ax.set_facecolor(BG_COLOR)

        if body_fn == 'side':
            fill_body_side(ax, zorder=1)
        else:
            r_cut = fill_body_cross(ax, x_val=1.0, zorder=1)
            title_sfx += f'  (r = {r_cut:.3f} m)'

        sl = sl_fn(field)
        im = ax.pcolormesh(hc, vc, sl.T, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', zorder=2)
        overlay_corrupt(ax, sl_fn(is_corrupt_only), hc, vc, zorder=3)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label, color='black', fontsize=8)
        cb.ax.yaxis.set_tick_params(color='black')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(xlbl, color='black', fontsize=9)
        ax.set_ylabel(ylbl, color='black', fontsize=9)
        ax.tick_params(colors='black', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444444')
        ax.set_aspect('equal')
        ax.set_title(f'{label}  -  {title_sfx}', color='black', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('flow_field_slices.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: flow_field_slices.png")

print("\nIf you still see a gap/box shape near the tail now, that's honest -")
print("it means those cells are genuinely corrupted (not disguised as hull),")
print("and points back at the trailing-edge cut-cell issue we discussed earlier,")
print("not a plotting bug.")