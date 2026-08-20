import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"This code is just making the geometry and physics in 3D and calculating q_wall without TPS considerations"


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — identical to the 2D case; nothing aerothermal changes here
# ═══════════════════════════════════════════════════════════════════════════════
N            = 12
x_target     = 2.5
y_target     = 0.125         # this becomes the BASE RADIUS of the body of revolution
panel_length = x_target / N

M_inf = 5.0
gamma = 1.4

T_inf, p_inf, rho_inf, Tw, Cp = 205.0, 5543.0, 0.09427, 300.0, 1005.0
R, Pr = 287.0, 0.71

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf
sigma = 5.67e-8

N_PHI = 48   # azimuthal resolution — number of meridian slices around the body


# ═══════════════════════════════════════════════════════════════════════════════
#  2D MERIDIAN PROFILE  — exactly your original panel method, unchanged.
#  r(x) (was y(x)) is the body-of-revolution RADIUS at each axial station.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nodes(theta):
    dx = panel_length * jnp.cos(theta)
    dr = panel_length * jnp.sin(theta)     # was dy — now radial growth
    x  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dx)])
    r  = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dr)])
    return x, r

def newtonian_cp(theta):
    """Same Newtonian theory — at alpha=0 every meridian sees identical Cp."""
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

def adiabatic_wall_temperature(p, T):
    M_e    = local_mach(p)
    r_lam  = Pr**0.5
    r_turb = Pr**(1.0/3.0)
    Taw_lam  = T * (1.0 + r_lam  * (gamma-1.0)/2.0 * M_e**2)
    Taw_turb = T * (1.0 + r_turb * (gamma-1.0)/2.0 * M_e**2)
    return Taw_lam, Taw_turb

def transition_location(x_safe, p, T):
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
    x_tr   = jnp.where(has_tr, x_safe[idx], x_safe[-1])
    return x_tr, Re_theta

def boundary_layer(x_safe, p, T):
    """
    Same BL-thickness model as the 2D code — returns delta(x), the
    boundary-layer thickness used to draw the 3D BL shell.
    """
    M_e   = local_mach(p)
    a_e   = jnp.sqrt(gamma * R * T)
    V_e   = M_e * a_e
    rho_e = density_from_pt(p, T)
    mu_e  = sutherland(T)

    Rex = rho_e * V_e * x_safe / mu_e
    x_tr, Re_theta = transition_location(x_safe, p, T)

    Taw_lam, Taw_turb = adiabatic_wall_temperature(p, T)
    Taw = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    comp_factor = 1.0 + 0.08*M_e**2 + 0.36*(Tw/Taw)*M_e**2
    delta_lam   = (5.0 * x_safe / jnp.sqrt(Rex)
                   * (Tw / T_inf)**(-1.0/6.0)
                   * comp_factor)

    x_from_tr  = jnp.maximum(x_safe - x_tr, 1e-6)
    Rex_tr     = rho_e * V_e * x_from_tr / mu_e
    delta_turb = 0.37 * x_from_tr / Rex_tr**0.2 * (Taw / T_inf)**0.6

    delta = jnp.where(x_safe <= x_tr, delta_lam, delta_turb)
    return delta, delta_lam, delta_turb, x_tr, Re_theta, Taw


def heat_transfer(x_safe, p, T, x_tr):
    M_e = local_mach(p)
    a_e = jnp.sqrt(gamma * R * T)
    V_e = M_e * a_e
    r_lam, r_turb = Pr**0.5, Pr**(1.0/3.0)
    T_ref_lam  = T * (0.45 + 0.55*(Tw/T) + 0.16*r_lam  * (gamma-1.0)/2.0 * M_e**2)
    T_ref_turb = T * (0.50*(1.0 + Tw/T)  + 0.16*r_turb * (gamma-1.0)/2.0 * M_e**2)
    T_ref      = jnp.where(x_safe < x_tr, T_ref_lam, T_ref_turb)
    rho_ref = p / (R * T_ref)
    mu_ref  = sutherland(T_ref)
    Rex_ref = rho_ref * V_e * x_safe / mu_ref
    St_lam  = (0.332  / jnp.sqrt(Rex_ref)) * Pr**(-2.0/3.0)
    St_turb = (0.0296 / Rex_ref**0.2)      * Pr**(-2.0/3.0)
    St      = jnp.where(x_safe < x_tr, St_lam, St_turb)
    Taw_lam, Taw_turb = adiabatic_wall_temperature(p, T)
    Taw = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)
    q_wall = rho_ref * V_e * St * Cp * (Taw - Tw)
    return St, q_wall, rho_ref, Rex_ref, Taw, V_e


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN THE 2D MERIDIAN SOLVE ONCE — this is your exact original physics
# ═══════════════════════════════════════════════════════════════════════════════
mean_angle = float(jnp.arctan(y_target / x_target))
theta = jnp.deg2rad(jnp.linspace(
    jnp.rad2deg(mean_angle) * 1.8,
    jnp.rad2deg(mean_angle) * 0.2,
    N
))

x_2d, r_2d = generate_nodes(theta)          # (N+1,) axial + radial coordinates
x_panel     = x_2d[1:]

p_2d = pressure_distribution(theta)
T_2d = isentropic_temperature(p_2d)

delta_dummy, x_tr_unused, Re_theta = None, None, None
x_tr, Re_theta = transition_location(jnp.maximum(x_panel, 1e-6), p_2d, T_2d)
delta, delta_lam, delta_turb, x_tr_check, Re_theta, Taw_bl = boundary_layer(
    jnp.maximum(x_panel, 1e-6), p_2d, T_2d)
St_ref, q_wall, rho_ref, Rex_ref, Taw, V_local = heat_transfer(
    jnp.maximum(x_panel, 1e-6), p_2d, T_2d, x_tr)

print(f"\n  [Meridian solve] x_tr = {float(x_tr):.4f} m")
print(f"  [Meridian solve] Taw range: {float(Taw.min()):.1f}-{float(Taw.max()):.1f} K")
print(f"  [Meridian solve] q_wall range: {float(q_wall.min()):.1f}-{float(q_wall.max()):.1f} W/m²")


# ═══════════════════════════════════════════════════════════════════════════════
#  REVOLUTION  —  sweep the 2D meridian profile around the x-axis
#
#  X(x, phi) = x
#  Y(x, phi) = r(x) * cos(phi)
#  Z(x, phi) = r(x) * sin(phi)
#
#  At alpha=0 (axisymmetric flow, no angle of attack) every meridian phi
#  sees IDENTICAL local flow physics — Cp, T, Taw, q_wall, x_tr all only
#  depend on x, not phi. So the 3D field is the 2D field broadcast across
#  the phi dimension. This is exact for axisymmetric Newtonian flow at
#  zero incidence; angle-of-attack would break this symmetry (left as a
#  documented extension point below).
# ═══════════════════════════════════════════════════════════════════════════════
phi = jnp.linspace(0.0, 2.0 * jnp.pi, N_PHI, endpoint=True)   # (N_PHI,)


def revolve_profile(x_1d, r_1d, phi_1d):
    """
    Revolve a 2D (x, r) meridian profile into a 3D surface mesh.

    Returns X, Y, Z each of shape (len(x_1d), len(phi_1d)) — a structured
    surface grid suitable for plot_surface or downstream meshing.
    """
    X = jnp.tile(x_1d[:, None], (1, len(phi_1d)))                       # (Nx, Nphi)
    R_grid = jnp.tile(r_1d[:, None], (1, len(phi_1d)))                   # (Nx, Nphi)
    PHI = jnp.tile(phi_1d[None, :], (len(x_1d), 1))                      # (Nx, Nphi)
    Y = R_grid * jnp.cos(PHI)
    Z = R_grid * jnp.sin(PHI)
    return X, Y, Z

X3, Y3, Z3 = revolve_profile(x_2d, r_2d, phi)        # surface nodes, (N+1, N_PHI)

# Broadcast every panel-wise physical quantity across phi — identical
# value at every azimuthal station for axisymmetric, zero-incidence flow.
def broadcast_to_phi(field_1d):
    """field_1d: (N,) panel-wise quantity -> (N, N_PHI) broadcast across phi."""
    return jnp.tile(jnp.asarray(field_1d)[:, None], (1, N_PHI))

p_3d      = broadcast_to_phi(p_2d)
T_3d      = broadcast_to_phi(T_2d)
Taw_3d    = broadcast_to_phi(Taw)
q_wall_3d = broadcast_to_phi(q_wall)
St_3d     = broadcast_to_phi(St_ref)
rho_ref_3d= broadcast_to_phi(rho_ref)
V_local_3d= broadcast_to_phi(V_local)

print(f"\n  [3D revolution] Surface mesh: {X3.shape[0]} axial nodes x {X3.shape[1]} azimuthal nodes")
print(f"  [3D revolution] Total surface panels: {N} x {N_PHI} = {N*N_PHI}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SURFACE AREA & WETTED-AREA-WEIGHTED TOTALS — genuinely new in 3D
#  (2D code only had per-unit-span flux; now we get true integrated values)
# ═══════════════════════════════════════════════════════════════════════════════
def panel_areas_axisymmetric(x_1d, r_1d, n_phi):
    """
    Area of each (i, j) surface panel on the body of revolution.
    For an axisymmetric frustum strip between x[i] and x[i+1], swept
    through d_phi = 2*pi/n_phi, the lateral surface area of one
    azimuthal slice is the standard frustum formula:
        dA = pi * (r_i + r_{i+1}) * slant_length / n_phi
    """
    x_np = np.array(x_1d)
    r_np = np.array(r_1d)
    dx = np.diff(x_np)
    dr = np.diff(r_np)
    slant = np.sqrt(dx**2 + dr**2)                      # (N,)
    dA_total_per_panel_row = np.pi * (r_np[:-1] + r_np[1:]) * slant   # full ring area, (N,)
    dA_per_phi_slice = dA_total_per_panel_row / n_phi                  # (N,) per phi-cell
    return dA_total_per_panel_row, dA_per_phi_slice

dA_ring, dA_cell = panel_areas_axisymmetric(x_2d, r_2d, N_PHI)
total_wetted_area = dA_ring.sum()

# Integrated (total) heat load over the whole wetted surface [W]
Q_total = float((np.array(q_wall) * dA_ring).sum())

print(f"\n  [3D integrals] Total wetted surface area : {total_wetted_area:.4f} m²")
print(f"  [3D integrals] Total integrated heat load : {Q_total/1000:.2f} kW  "
      f"(was W/m² only in 2D — this is the genuinely new 3D quantity)")


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — 3D surface coloured by wall heat flux
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_surface_qwall(X3, Y3, Z3, q_wall_3d, savename='surface3d_qwall.png'):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Yn, Zn = np.array(X3), np.array(Y3), np.array(Z3)
    Qn = np.array(q_wall_3d)

    norm = plt.Normalize(Qn.min(), Qn.max())
    colors = cm.YlOrRd(norm(Qn))

    surf = ax.plot_surface(Xn, Yn, Zn, facecolors=colors,
                           rstride=1, cstride=1, linewidth=0,
                           antialiased=True, shade=False)

    mappable = cm.ScalarMappable(cmap=cm.YlOrRd, norm=norm)
    mappable.set_array(Qn)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='q_wall (W/m²)')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f'Axisymmetric body — wall heat flux\n'
                f'(revolved from 2D meridian, alpha=0, M={M_inf})')

    max_r = float(r_2d.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — 3D wireframe with meridian profile highlighted
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_wireframe(X3, Y3, Z3, savename='surface3d_wireframe.png'):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Yn, Zn = np.array(X3), np.array(Y3), np.array(Z3)
    ax.plot_wireframe(Xn, Yn, Zn, color='steelblue', linewidth=0.4, alpha=0.6)

    # Highlight one meridian (phi=0, the original 2D profile) in red
    ax.plot(Xn[:, 0], Yn[:, 0], Zn[:, 0], 'r-', linewidth=2.5,
            label='Original 2D meridian (phi=0)')

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('Body of revolution — wireframe\n'
                f'{N} axial stations x {N_PHI} azimuthal stations')
    ax.legend(fontsize=9)
    max_r = float(r_2d.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — TPS layers revolved (radial offset toward the axis)
#  Mirrors your 2D "layers inside the geometry" logic, but offset is now
#  RADIALLY INWARD (toward r=0, the centreline) rather than toward y=0.
# ═══════════════════════════════════════════════════════════════════════════════
tps_layers = [
    {'name': 'Aluminum', 't': 0.003, 'k': 167.0, 'rho': 2700.0, 'Cp': 900.0,  'eps': 0.125},
    {'name': 'SIP',      't': 0.005, 'k': 0.05,  'rho': 128.0,  'Cp': 1000.0, 'eps': 0.85},
    {'name': 'Silica',   't': 0.050, 'k': 0.06,  'rho': 1900.0, 'Cp': 800.0,  'eps': 0.85},
    {'name': 'RCC',      't': 0.005, 'k': 20.0,  'rho': 1600.0, 'Cp': 711.0,  'eps': 0.875},
]

def plot_3d_tps_layers(x_1d, r_1d, phi_1d, tps_layers, n_phi_draw=24,
                       savename='tps_layers_3d.png'):
    """
    Revolve the TPS layer stack into 3D. Each layer is drawn as a true
    filled BAND between its outer and inner radial boundary, not just a
    single surface at the inner edge. The outermost boundary (frac=0) is
    exactly r_np, the real body surface, so RCC's outer face coincides
    with the body with NO gap. This mirrors the 2D version, which filled
    a polygon between (x_out, y_out) and (x_in, y_in) for every layer.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_np = np.array(x_1d)
    r_np = np.array(r_1d)
    phi_draw = np.linspace(0, 2*np.pi, n_phi_draw, endpoint=True)

    layer_order = [3, 2, 1, 0]   # RCC -> Silica -> SIP -> Aluminum
    t_real = np.array([tps_layers[i]['t'] for i in layer_order])
    colors = ['#c0392b', '#e67e22', '#f1c40f', '#95a5a6']
    labels = [f"{tps_layers[i]['name']}  {tps_layers[i]['t']*1000:.0f} mm"
              for i in layer_order]

    vis_fraction = 0.55
    t_cumul_norm = np.concatenate([[0.0], np.cumsum(t_real) / t_real.sum()])

    PHI, Xg = np.meshgrid(phi_draw, x_np)

    def radial_surface(frac):
        """Surface of revolution at normalised depth frac (0=body, 1=deepest)."""
        r_offset = r_np * (1.0 - frac * vis_fraction)
        R_grid = np.tile(r_offset[:, None], (1, n_phi_draw))
        return R_grid * np.cos(PHI), R_grid * np.sin(PHI)

    # Draw each layer as TWO surfaces (outer + inner boundary), same colour,
    # so the band reads as one filled shell. RCC (li=0) starts at frac=0,
    # i.e. the real body surface -- no gap.
    for li in range(4):
        frac_out = t_cumul_norm[li]        # nearer the surface
        frac_in  = t_cumul_norm[li + 1]    # deeper inward

        Y_out, Z_out = radial_surface(frac_out)
        Y_in,  Z_in  = radial_surface(frac_in)

        ax.plot_surface(Xg, Y_out, Z_out, color=colors[li], alpha=0.85,
                        linewidth=0, antialiased=True, shade=True)
        ax.plot_surface(Xg, Y_in, Z_in, color=colors[li], alpha=0.85,
                        linewidth=0, antialiased=True, shade=True)

        # End-cap ring at the trailing edge so the band reads as a solid
        # shell rather than two disconnected surfaces (visual only).
        ax.plot(Xg[-1, :], Y_out[-1, :], Z_out[-1, :], color=colors[li], linewidth=1.0)
        ax.plot(Xg[-1, :], Y_in[-1, :],  Z_in[-1, :],  color=colors[li], linewidth=1.0)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l, alpha=0.85) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, fontsize=8, loc='upper left')

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('TPS layer stack revolved into 3D — RCC outer -> Aluminum inner\n'
                f'(visual scale: {vis_fraction*100:.0f}% of local radius, '
                f'RCC outer face = actual body surface, no gap)')
    max_r = float(r_np.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Pressure field on the 3D surface  (3D equivalent of dp/dx plot)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_pressure(X3, Y3, Z3, p_3d, savename='surface3d_pressure.png'):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Yn, Zn = np.array(X3), np.array(Y3), np.array(Z3)
    Pn = np.array(p_3d)

    norm = plt.Normalize(Pn.min(), Pn.max())
    colors = cm.Blues(norm(Pn))

    ax.plot_surface(Xn, Yn, Zn, facecolors=colors,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)

    mappable = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    mappable.set_array(Pn)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='p (Pa)')

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(f'Axisymmetric body — surface pressure\n'
                f'(Newtonian theory, alpha=0, M={M_inf})')
    max_r = float(r_2d.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Boundary layer as a translucent 3D shell  (3D equivalent of
#  plot_temperature_bl). The shell is the body surface offset OUTWARD
#  (away from the axis) by the local delta(x), coloured by T edge.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_boundary_layer(x_1d, r_1d, phi_1d, delta_1d, field_1d,
                           vis_scale=3.0, color_label='q_wall (W/m²)',
                           savename='bl_shell_3d.png'):
    """
    delta_1d, field_1d are panel-wise (N,) arrays (defined at x_panel = x_1d[1:]).
    vis_scale exaggerates delta for visibility, same idea as the 2D plot.
    field_1d can be q_wall, T, or Taw — pass color_label to match.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_np     = np.array(x_1d)
    r_np     = np.array(r_1d)
    delta_np = np.array(delta_1d)
    field_np = np.array(field_1d)
    phi_np   = np.array(phi_1d)

    # delta/field are defined at panel nodes (x_1d[1:]); pad a value at the
    # nose so shapes match the (N+1,) surface arrays.
    delta_full = np.concatenate([[0.0], delta_np])
    field_full = np.concatenate([[field_np[0]], field_np])

    r_outer = r_np + delta_full * vis_scale     # BL outer edge, radially outward
    # multiplied by vis_scale --> the boundary layer is scaled up x3

    PHI, Xg = np.meshgrid(phi_np, x_np)

    # ── body surface (solid, grey) ───────────────────────────────────────────
    R_body = np.tile(r_np[:, None], (1, len(phi_np)))
    Y_body = R_body * np.cos(PHI)
    Z_body = R_body * np.sin(PHI)
    ax.plot_surface(Xg, Y_body, Z_body, color='#7f8c8d', alpha=0.9,
                    linewidth=0, antialiased=True, shade=True)

    # ── BL shell (translucent, coloured by field) ─────────────────────────────
    R_bl = np.tile(r_outer[:, None], (1, len(phi_np)))
    Y_bl = R_bl * np.cos(PHI)
    Z_bl = R_bl * np.sin(PHI)

    field_grid = np.tile(field_full[:, None], (1, len(phi_np)))
    norm = plt.Normalize(field_full.min(), field_full.max())
    colors = cm.plasma(norm(field_grid))
    colors[..., 3] = 0.35   # translucency on the BL shell only

    ax.plot_surface(Xg, Y_bl, Z_bl, facecolors=colors,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)

    mappable = cm.ScalarMappable(cmap=cm.plasma, norm=norm)
    mappable.set_array(field_full)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label=color_label)

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(f'Boundary layer as a 3D shell (x{vis_scale:.0f} visual scale)\n'
                f'Solid = body surface, translucent = BL outer edge coloured by T')
    max_r = float(r_outer.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 6 — Transition location as a ring around the body
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_transition_ring(X3, Y3, Z3, q_wall_3d, x_tr, r_1d, x_1d,
                            savename='transition_ring_3d.png'):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    Xn, Yn, Zn = np.array(X3), np.array(Y3), np.array(Z3)
    Qn = np.array(q_wall_3d)

    norm = plt.Normalize(Qn.min(), Qn.max())
    colors = cm.YlOrRd(norm(Qn))
    ax.plot_surface(Xn, Yn, Zn, facecolors=colors,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False, alpha=0.95)

    # ── interpolate body radius at x_tr to draw the ring at the right size ───
    x_np = np.array(x_1d)
    r_np = np.array(r_1d)
    r_at_tr = float(np.interp(float(x_tr), x_np, r_np))

    phi_ring = np.linspace(0, 2*np.pi, 200)
    x_ring = np.full_like(phi_ring, float(x_tr))
    y_ring = r_at_tr * np.cos(phi_ring)
    z_ring = r_at_tr * np.sin(phi_ring)
    ax.plot(x_ring, y_ring, z_ring, color='blue', linewidth=3.0, zorder=10,
            label=f'Transition: x_tr = {float(x_tr):.3f} m')

    mappable = cm.ScalarMappable(cmap=cm.YlOrRd, norm=norm)
    mappable.set_array(Qn)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.15, label='q_wall (W/m²)')

    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('Laminar -> turbulent transition ring on the 3D body\n'
                'Surface coloured by q_wall — note the flux jump across the ring')
    
    # ── FIX OVERLAPPING AXIS TEXT ──
    # Reduce the density of tick marks
    ax.set_yticks([-0.10, 0.00, 0.10])
    ax.set_zticks([-0.10, 0.00, 0.10])
    
    # Add spacing between numbers and grid lines
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=10)
    
    # Add spacing for the 'y (m)' and 'z (m)' titles
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    # ────────────────────────────────

    max_r = float(r_np.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 7 — Cutaway view (half body): TPS layers + heat flux together
#  phi in [0, pi] shows the cut TPS cross-section; phi in [pi, 2pi] shows
#  the outer skin coloured by q_wall, like an engineering cutaway drawing.
# ═══════════════════════════════════════════════════════════════════════════════
def plot_3d_cutaway(x_1d, r_1d, phi_1d, q_wall_1d, tps_layers,
                    n_phi_draw=48, savename='cutaway_3d.png'):
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_np = np.array(x_1d)
    r_np = np.array(r_1d)
    q_full = np.concatenate([[float(q_wall_1d[0])], np.array(q_wall_1d)])

    # ── HALF 1 (phi in [pi, 2pi]) — outer skin coloured by q_wall ───────────
    phi_skin = np.linspace(np.pi, 2*np.pi, n_phi_draw // 2)
    PHI_s, Xg_s = np.meshgrid(phi_skin, x_np)
    R_s = np.tile(r_np[:, None], (1, len(phi_skin)))
    Y_s = R_s * np.cos(PHI_s)
    Z_s = R_s * np.sin(PHI_s)

    Q_grid = np.tile(q_full[:, None], (1, len(phi_skin)))
    norm_q = plt.Normalize(q_full.min(), q_full.max())
    colors_q = cm.YlOrRd(norm_q(Q_grid))
    ax.plot_surface(Xg_s, Y_s, Z_s, facecolors=colors_q,
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=True, shade=False)

    # ── HALF 2 (phi in [0, pi]) — TPS layers revolved, showing the stack ────
    phi_tps = np.linspace(0.0, np.pi, n_phi_draw // 2)
    PHI_t, Xg_t = np.meshgrid(phi_tps, x_np)

    layer_order = [3, 2, 1, 0]
    t_real = np.array([tps_layers[i]['t'] for i in layer_order])
    layer_colors = ['#c0392b', '#e67e22', '#f1c40f', '#95a5a6']
    vis_fraction = 0.55
    t_cumul_norm = np.concatenate([[0.0], np.cumsum(t_real) / t_real.sum()])

    for li in range(4):
        frac_out = t_cumul_norm[li]
        frac_in  = t_cumul_norm[li + 1]

        r_out = r_np * (1.0 - frac_out * vis_fraction)
        r_in  = r_np * (1.0 - frac_in  * vis_fraction)

        R_out = np.tile(r_out[:, None], (1, len(phi_tps)))
        R_in  = np.tile(r_in[:, None],  (1, len(phi_tps)))
        Y_out, Z_out = R_out * np.cos(PHI_t), R_out * np.sin(PHI_t)
        Y_in,  Z_in  = R_in  * np.cos(PHI_t), R_in  * np.sin(PHI_t)

        ax.plot_surface(Xg_t, Y_out, Z_out, color=layer_colors[li], alpha=0.92,
                        linewidth=0, antialiased=True, shade=True)
        ax.plot_surface(Xg_t, Y_in, Z_in, color=layer_colors[li], alpha=0.92,
                        linewidth=0, antialiased=True, shade=True)

    # ── cut-plane outline at phi=0 and phi=pi to make the cut visible ───────
    for phi_cut in [0.0, np.pi]:
        r_edge = r_np
        y_edge = r_edge * np.cos(phi_cut)
        z_edge = r_edge * np.sin(phi_cut)
        ax.plot(x_np, y_edge, z_edge, 'k-', linewidth=1.5, alpha=0.7)

    mappable = cm.ScalarMappable(cmap=cm.YlOrRd, norm=norm_q)
    mappable.set_array(q_full)
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.1, label='q_wall (W/m²)')

    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('Cutaway view — outer skin (q_wall) vs revolved TPS stack\n'
                f'(TPS visual scale: {vis_fraction*100:.0f}% of local radius)')
    max_r = float(r_np.max())
    ax.set_box_aspect([x_target, 2*max_r, 2*max_r])
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
plot_3d_surface_qwall(X3, Y3, Z3, q_wall_3d)
plot_3d_wireframe(X3, Y3, Z3)
plot_3d_tps_layers(x_2d, r_2d, phi, tps_layers)
plot_3d_pressure(X3, Y3, Z3, p_3d)
plot_3d_boundary_layer(x_2d, r_2d, phi, delta, T_2d,
                       color_label='T edge (K)')
# Matches the 2D convention exactly: coloured by T_2d, the ISENTROPIC
# STATIC edge temperature (T from isentropic_temperature(p)), NOT Taw
# (adiabatic wall temperature) and NOT q_wall. This is the same "T edge"
# quantity the 2D plot_temperature_bl function colours by, so the 3D
# shell now shows the same smooth purple->orange->yellow gradient as
# the 2D version instead of flat bands.
plot_3d_transition_ring(X3, Y3, Z3, q_wall_3d, x_tr, r_2d, x_2d)
plot_3d_cutaway(x_2d, r_2d, phi, q_wall, tps_layers)

print("\nAll 3D plots saved.")

# ═══════════════════════════════════════════════════════════════════════════════
#  EXTENSION POINT — angle of attack (alpha != 0)
#
#  Everything above assumed axisymmetric flow (alpha=0), where Cp only
#  depends on x. To add angle of attack, the LOCAL surface normal direction
#  at each (x, phi) panel must be dotted with the freestream direction
#  vector (which now has a y- or z-component), making Cp a genuine
#  function of (x, phi):
#
#    n_hat(x, phi) = surface normal at that panel (3 components)
#    V_hat_inf     = (cos(alpha), 0, sin(alpha))   [freestream direction]
#    Cp(x, phi)    = 2 * max(n_hat . V_hat_inf, 0)^2
#
#  This breaks the broadcast-across-phi shortcut used here: p, T, Taw,
#  q_wall would each become genuine (N, N_PHI) fields requiring the BL
#  and heat-transfer functions to be vmapped over both x and phi. This
#  is a direct, mechanical extension of the structure above — every
#  function already takes array inputs, so jax.vmap over phi is the
#  natural next step once a real surface-normal field is computed from
#  the 3D mesh (e.g. via cross products of adjacent panel edge vectors).
# ═══════════════════════════════════════════════════════════════════════════════