import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
 
# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
N = 12

x_target = 2.5
y_target = 0.125 #check that


panel_length = 2.5 / N #to ensure it meets the chord length

M_inf = 5.0
gamma = 1.4 #can be changed to 1.2-1.3

# Atmosphere at h = 20 km gotten from table
T_inf   = 205.0       # K
p_inf   = 5543.0      # Pa
rho_inf = 0.09427     # kg/m^3
Tw      = 300.0       # K  (wall temperature) that we set / can be higher but it may not make a difference
mu_inf = 1.458e-6*T_inf**(3/2)/(T_inf+110.4)    # dynamic viscosity (kg/(m·s)) 
Cp=1005 # in K/Kg.K

# Known constants
R = 287.0       # J/(kg·K)
Pr = 0.71     #double check for air

a_inf = jnp.sqrt(gamma * R * T_inf)
V_inf = M_inf * a_inf


# ─────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────
def generate_nodes(theta):
    """
    Build node coordinates from an array of N panel angles (rad).
    Returns x, y arrays of length N+1 (nodes, not panel centres).
    """
    dx = panel_length * jnp.cos(theta)
    dy = panel_length * jnp.sin(theta)

    x = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dx)])
    y = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dy)])

    return x, y

# ─────────────────────────────────────────────
# Aerodynamics — Newtonian + isentropic
# ─────────────────────────────────────────────
def newtonian_cp(theta):
    return 2.0 * jnp.sin(theta)**2

#we can consider modified newtonian too - more accurate 
# we have to be careful for leeward that has cp=0 for theta<0 FOR 3D

"Local pressure via Newtonian"
def pressure_distribution(theta):
    Cp    = newtonian_cp(theta)
    q_inf = 0.5 * rho_inf * V_inf**2
    p     = p_inf + q_inf * Cp
    return p

"Local temperature via isentropic"
def isentropic_temperature(p):
    "Isentropic relations P0/P and T0/T and the relations that relates both"
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    T0 = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
    T  = T0 * (p / p0)**((gamma-1.0)/gamma)
    return T

# we get p from newtonian and T from isentropic: there might be an inconsistency because of that
#For higher fidelity, use the Rankine-Hugoniot shock relations followed by isentropic expansion.

# assuming the gas behaves ideally
def density_from_pt(p, T):
    " Returns local density from pressure (newtonian) and temperature(isentropic)"
    return p / (R * T)

# ─────────────────────────────────────────────
# Boundary layer
# ─────────────────────────────────────────────
def adiabatic_wall_temperature():
    M_local=local_mach(p)
    r_lam  = Pr**(0.5)          # laminar
    r_turb = Pr**(1.0/3.0)      # turbulent
# Here it used to be M_inf - I changed it to local
    Taw_lam  = T_inf * (1.0 + r_lam  * (gamma - 1.0)/2.0 * M_local**2)
    Taw_turb = T_inf * (1.0 + r_turb * (gamma - 1.0)/2.0 * M_local**2)

    return Taw_lam, Taw_turb

# This is local mach that we need for the criterion
def local_mach(p):
    p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
    return jnp.sqrt((2.0/(gamma-1.0)) * ((p0/p)**((gamma-1.0)/gamma) - 1.0))

def transition_location(x_safe, rho_local,p,T):
    # Local Properties
    M_local=local_mach(p)
    a_local = jnp.sqrt(gamma * R * T)
    V_local=M_local*a_local
    mu_local= 1.458e-6*T**(3/2)/(T+110.4)  

    # Local Reynolds at x
    Rex      = rho_local * V_local * x_safe / mu_local

    "Find a compressible equation for momentum thickness"
    theta_m  = 0.664 * x_safe / jnp.sqrt(Rex)      # momentum thickness: Blasius is incompressible, I should use Van Driest II

    # Local Reynolds dependent on momentum thickness theta
    Re_theta = rho_local * V_local * theta_m / mu_local 
    criterion = Re_theta / M_local

    has_transition = jnp.any(criterion >= 400.0)     # Rotta-ARA criterion (also called the Michel-type compressible criterion)
    idx            = jnp.argmax(criterion >= 400.0)  # 0 if none — guarded below

    x_tr = jnp.where(has_transition, x_safe[idx], x_safe[-1])
    return x_tr, Re_theta


def heat_transfer(x_safe, p, T, x_tr):
    """
    Heat flux using Reynolds analogy with Eckert reference temperature.
    Laminar:    Cf = 0.664 / sqrt(Rex)   → Stanton from Reynolds analogy
    Turbulent:  Cf = 0.0592 / Rex^0.2    → Stanton from Reynolds analogy
    St = Cf / (2 * Pr^(2/3))              Reynolds analogy (both regimes)
    q_wall = rho_ref * V_e * St_ref *Cp* (Taw - Tw)
    Let's assume V_e=V_local 
    """
    M_local=local_mach(p)
    a_local = jnp.sqrt(gamma * R * T)
    V_local=M_local*a_local

    # Eckert reference temperature
    r_lam  = Pr**(0.5)          # laminar
    r_turb = Pr**(1.0/3.0)      # turbulent
    "Ref is star with T is T at the edge"
    T_ref_lam=T*(0.45+0.55*Tw/T+0.16*r_lam*((gamma-1)/2)*M_local**2)
    T_ref_turb=T*(0.5*(1+Tw/T)+0.16*r_turb*((gamma-1)/2)*M_local**2)
    T_ref= jnp.where(x_safe < x_tr, T_ref_lam, T_ref_turb)
    rho_ref = p / (R * T_ref)
    mu_ref  =  1.458e-6*T_ref**(3/2)/(T_ref+110.4)  #sutherland

    Rex_ref = rho_ref * V_local * x_safe / mu_ref

    #Stanton from the Reynolds analogy that St=Cf/2*Pr**(-2/3)
    St_ref_lam=(0.332/jnp.sqrt(Rex_ref))*Pr**(-2/3)
    St_ref_turb=(0.0296/Rex_ref**0.2)*Pr**(-2/3)
    St_ref=jnp.where(x_safe < x_tr, St_ref_lam, St_ref_turb)
    # Heat flux W/m² — positive = into wall
    #we assum Cp_star=Cp=1005 for air
    q_wall = rho_ref * V_local * St_ref *Cp*(Taw - Tw)

    return St_ref, q_wall, rho_ref, Rex_ref


def boundary_layer(x, p, T):
    """
    Compute laminar and turbulent BL thickness on each panel.

    CHECK BOTH EQUATIONS REFERENCES
    Laminar  : Van Driest-style flat-plate with compressibility correction.
    Turbulent: 1/5-power law with temperature ratio correction (White/Cebeci).

    Returns
    -------
    delta       : thickness (lam or turb per panel) based on transition
    delta_lam   : laminar thickness everywhere
    delta_turb  : turbulent thickness everywhere
    x_transition: transition x-coordinate
    Re_theta    : momentum-thickness Reynolds number array
    """
    M_local=local_mach(p)
    a_local = jnp.sqrt(gamma * R * T)
    V_local=M_local*a_local

    rho_local    = density_from_pt(p, T)
    mu_local= 1.458e-6*T**(3/2)/(T+110.4)  

    x_safe = jnp.maximum(x, 1e-6)   # just to ensure not dividing by 0
    Rex = rho_local* V_local * x_safe / mu_local

    x_tr, Re_theta = transition_location(x_safe, rho_local,p,T)

# Select Taw panel-by-panel based on transition
    Taw_lam, Taw_turb = adiabatic_wall_temperature()
    Taw = jnp.where(x_safe < x_tr, Taw_lam, Taw_turb)

    # Laminar — flat-plate + compressibility correction
    comp_factor = (1.0 + 0.08*M_inf**2 + 0.36*(Tw/Taw)*M_inf**2)
    delta_lam   = (5.0 * x_safe / jnp.sqrt(Rex)
                   * (Tw / T_inf)**(-1.0/6.0)
                   * comp_factor)

    # Turbulent 
    x_from_tr  = jnp.maximum(x_safe - x_tr, 1e-6)  #ensuring it's being measure from transition point not leading edge
    Rex_tr = rho_local * V_local * x_from_tr / mu_local
    delta_turb = (0.37 * x_from_tr / Rex_tr**0.2
                  * (Taw / T_inf)**0.6)

    delta = jnp.where(x_safe <= x_tr, delta_lam, delta_turb)

    return delta, delta_lam, delta_turb, x_tr, Re_theta, Taw

# Automatically compute angles to hit target
mean_angle = float(jnp.arctan(y_target / x_target))  # ~2.86° for y=0.125

# Taper around the mean — starts steeper, flattens toward base
theta = jnp.deg2rad(jnp.linspace(
    jnp.rad2deg(mean_angle) * 1.8,   # ~5.1° at nose
    jnp.rad2deg(mean_angle) * 0.2,   # ~0.6° at tail
    N
))

# ─────────────────────────────────────────────
# Evaluate all quantities
# ─────────────────────────────────────────────
x, y = generate_nodes(theta)
x_panel = x[1:]           # panel representative x = downstream node

p   = pressure_distribution(theta)
T   = isentropic_temperature(p)
rho_local = density_from_pt(p, T)

delta, delta_lam, delta_turb, x_tr, Re_theta, Taw= boundary_layer(x_panel, p, T)
St_ref, q_wall, rho_ref, Rex_ref = heat_transfer(x_panel, p, T, x_tr)

dp_dx = jnp.gradient(p, x_panel)  #maybe plot on geometry
dT_dx = jnp.gradient(T, x_panel)  #maybe plot

# ─────────────────────────────────────────────
# PRINTOUT SECTION 
# ─────────────────────────────────────────────
print("=" * 55)
print("  GEOMETRY CHECK")
print("=" * 55)
print(f"  Target endpoint : ({x_target:.3f}, {y_target:.3f}) m")
print(f"  Actual endpoint : ({float(x[-1]):.3f}, {float(y[-1]):.3f}) m")
print(f"  Endpoint error  : dx={float(x[-1]-x_target):+.4f}  dy={float(y[-1]-y_target):+.4f}")

print("\n  Panel angles (deg):")
for i, a in enumerate(jnp.rad2deg(theta)):
    print(f"    Panel {i+1:2d}: {float(a):6.2f}°")

print("\n" + "=" * 55)
print("  FREESTREAM / STAGNATION")
print("=" * 55)
print(f"  a_inf = {float(a_inf):.2f} m/s    V_inf = {float(V_inf):.2f} m/s")
q_inf = 0.5 * rho_inf * V_inf**2
print(f"  q_inf = {float(q_inf):.2f} Pa")
T0 = T_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)
p0 = p_inf * (1.0 + 0.5*(gamma-1.0)*M_inf**2)**(gamma/(gamma-1.0))
print(f"  T0    = {float(T0):.2f} K    p0 = {float(p0):.2f} Pa")

Taw_lam, Taw_turb = adiabatic_wall_temperature()
print(f"  Taw_lam  = {float(Taw_lam[0]):.2f} K  (laminar,  r=Pr^1/2)")
print(f"  Taw_turb = {float(Taw_turb[0]):.2f} K  (turbulent, r=Pr^1/3)")

print("\n" + "=" * 55)
print("  PANEL-BY-PANEL RESULTS")
print("=" * 55)
print(f"  {'x':>6}  {'θ°':>6}  {'p(Pa)':>8}  {'T(K)':>7}  "
      f"{'ρ':>8}  {'Re_x':>10}  {'Re_θ':>10}  "
      f"{'δ_lam(m)':>10}  {'δ_turb(m)':>10}  {'δ(m)':>8}")
mu_local_arr = 1.458e-6*T**(3/2)/(T+110.4)
M_local_arr = local_mach(p)
a_local_arr = jnp.sqrt(gamma * R * T)
V_local_arr = M_local_arr * a_local_arr

Rex_arr = rho_local * V_local_arr * jnp.maximum(x_panel, 1e-6) / mu_local_arr
for i in range(N):
    print(f"  {float(x_panel[i]):6.3f}  {float(jnp.rad2deg(theta[i])):6.2f}  "
          f"{float(p[i]):8.1f}  {float(T[i]):7.2f}  "
          f"{float(rho_local[i]):8.5f}  {float(Rex_arr[i]):10.2e}  "
          f"{float(Re_theta[i]):10.2e}  "
          f"{float(delta_lam[i]):10.5f}  {float(delta_turb[i]):10.5f}  "
          f"{float(delta[i]):8.5f}")

print(f"\n  Transition location: x_tr = {float(x_tr):.4f} m")
comp_factor_scalar = float(1.0 + 0.08*M_inf**2 + 0.36*(Tw/Taw_turb[0])*M_inf**2)
print(f"  Laminar BL compressibility multiplier = {comp_factor_scalar:.3f}×")

print("\n" + "=" * 65)
print("  HEAT TRANSFER")
print("=" * 65)
print(f"  {'x':>6} {'St':>10}  {'q_wall(W/m²)':>14}")
for i in range(N):
    print(f"  {float(x_panel[i]):6.3f}  "
          f"{float(St_ref[i]):10.4e}  "
          f"{float(q_wall[i]):14.1f}")
print(f"\n  Max heat flux : {float(q_wall.max()):.1f} W/m²  at x = {float(x_panel[jnp.argmax(q_wall)]):.3f} m")
print(f"  Min heat flux : {float(q_wall.min()):.1f} W/m²  at x = {float(x_panel[jnp.argmin(q_wall)]):.3f} m")
print(f"  Mean heat flux: {float(q_wall.mean()):.1f} W/m²")


# ─────────────────────────────────────────────
# GRADIENT PLOTS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Shared vehicle outline helper
# Draws surface + base + bottom line to close the vehicle shape
# ─────────────────────────────────────────────
def draw_vehicle(ax, x, y, zorder=2):
    """Surface, vertical base, and horizontal bottom line."""
    ax.plot(x, y,
            'k-', linewidth=2.5, zorder=zorder)                        # surface
    ax.plot([x_target, x_target], [0.0, y_target],
            'k-', linewidth=2.5, zorder=zorder)                        # base
    ax.plot([0.0, x_target], [0.0, 0.0],
            'k-', linewidth=2.5, zorder=zorder)                        # bottom
 
 
# ─────────────────────────────────────────────
# Figure 3a — dp/dx
# Arrows compress INTO surface from above, shorter downstream
# ─────────────────────────────────────────────
def plot_pressure_arrows(x, y, theta, dp_dx, savename='arrows_dpdx.png'):
    fig, ax = plt.subplots(figsize=(13, 4))
 
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])
 
    # Outward normal (away from surface)
    nx = -jnp.sin(theta)
    ny =  jnp.cos(theta)
 
    fabs      = float(jnp.abs(dp_dx).max())
    max_arrow = y_target * 8.0
 
    cmap_obj = cm.get_cmap('Blues_r')
    norm     = mcolors.Normalize(vmin=float(dp_dx.min()), vmax=0.0)
 
    draw_vehicle(ax, x, y, zorder=3)
 
    for i in range(N):
        fi     = float(dp_dx[i])
        length = (abs(fi) / fabs) * max_arrow
 
        # Tip ON surface, tail above — arrow compresses down onto panel
        tip_x  = float(xm[i])
        tip_y  = float(ym[i])
        tail_x = tip_x + float(nx[i]) * length
        tail_y = tip_y + float(ny[i]) * length
 
        ax.annotate('',
            xy=(tip_x, tip_y), xytext=(tail_x, tail_y),
            arrowprops=dict(arrowstyle='->', color=cmap_obj(norm(fi)),
                            lw=2.0, mutation_scale=14),
            zorder=4)
 
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='dp/dx (Pa/m)  [all favourable → negative]')
 
    ax.set_ylim(-y_target * 0.3, y_target * 10.0)
    ax.set_xlim(-0.1, x_target + 0.15)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('dp/dx — compression arrows pointing into surface\n'
                 '(shorter downstream = weaker gradient as surface flattens)')
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    plt.show()
 
 
# ─────────────────────────────────────────────
# Figure 3b — dT/dx  (BL thickness filled, coloured by temperature)
# Surface drawn on top, temperature gradient shown as filled BL region
# ─────────────────────────────────────────────
def plot_temperature_bl(x, y, theta, x_panel, delta, T, dT_dx,
                        savename='arrows_dTdx.png'):
    fig, ax = plt.subplots(figsize=(13, 4))

    # delta is defined at x[1:] so use surface nodes as base
    x_surface = x[1:]
    y_surface = y[1:]

    nx = -jnp.sin(theta)
    ny =  jnp.cos(theta)

    x_outer = x_surface + delta * nx
    y_outer = y_surface + delta * ny

    T_norm   = mcolors.Normalize(vmin=float(T.min()), vmax=float(T.max()))
    cmap_obj = cm.get_cmap('plasma')

    # Fill between adjacent surface nodes and their BL outer edges
    for i in range(N - 1):
        xs = [float(x_surface[i]),   float(x_surface[i+1]),
              float(x_outer[i+1]),   float(x_outer[i])]
        ys = [float(y_surface[i]),   float(y_surface[i+1]),
              float(y_outer[i+1]),   float(y_outer[i])]
        ax.fill(xs, ys, color=cmap_obj(T_norm(float(T[i]))),
                alpha=0.9, zorder=2)

    # Last patch to base
    xs = [float(x_surface[-1]), float(x_target),
          float(x_target),      float(x_outer[-1])]
    ys = [float(y_surface[-1]), float(y_target),
          float(y_target),      float(y_outer[-1])]
    ax.fill(xs, ys, color=cmap_obj(T_norm(float(T[-1]))),
            alpha=0.9, zorder=2)

    # dT/dx arrows along BL outer edge
    tx = jnp.cos(theta)
    ty = jnp.sin(theta)
    fabs       = float(jnp.abs(dT_dx).max())
    max_arrow  = panel_length * 0.7
    arrow_norm = mcolors.Normalize(vmin=float(dT_dx.min()), vmax=0.0)
    arrow_cmap = cm.get_cmap('coolwarm')

    for i in range(N):
        fi     = float(dT_dx[i])
        length = (abs(fi) / fabs) * max_arrow
        bx = float(x_outer[i])
        by = float(y_outer[i])
        sign  = 1.0 if fi >= 0 else -1.0
        tip_x = bx + sign * float(tx[i]) * length
        tip_y = by + sign * float(ty[i]) * length
        ax.annotate('',
            xy=(tip_x, tip_y), xytext=(bx, by),
            arrowprops=dict(arrowstyle='->', color=arrow_cmap(arrow_norm(fi)),
                            lw=1.8, mutation_scale=11),
            zorder=5)

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=T_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='T edge (K)')

    draw_vehicle(ax, x, y, zorder=6)

    ax.set_ylim(-y_target * 0.3, y_target * 4.0)
    ax.set_xlim(-0.1, x_target + 0.15)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Boundary layer thickness (δ from table) coloured by T edge (K)\n'
                 'arrows show dT/dx direction along BL outer edge')
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    plt.show()


def plot_heatflux_arrows(x, y, theta, q_wall, x_tr,
                         savename='arrows_qwall.png'):
    """
    Vehicle surface panels coloured by local q_wall value.
    Thicker line = more heat flux at that panel.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    q_norm   = mcolors.Normalize(vmin=0.0, vmax=float(q_wall.max()))
    cmap_obj = cm.get_cmap('YlOrRd')

    # Draw each surface panel coloured by q_wall
    for i in range(N):
        xs = [float(x[i]),   float(x[i+1]),
              float(x[i+1]), float(x[i])]
        ys = [float(y[i]),   float(y[i+1]),
              0.0,            0.0]
        color = cmap_obj(q_norm(float(q_wall[i])))
        ax.fill(xs, ys, color=color, alpha=0.9, zorder=2)

    # Vehicle outline on top so edges are crisp
    draw_vehicle(ax, x, y, zorder=4)

    # Transition marker
    ax.axvline(float(x_tr), color='blue', linestyle='--',
               linewidth=1.5, label=f'x_tr = {float(x_tr):.2f} m', zorder=5)

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=q_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='q_wall (W/m²)')

    ax.legend(fontsize=9)
    ax.set_ylim(-y_target * 0.3, y_target * 3.0)
    ax.set_xlim(-0.1, x_target + 0.15)
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Vehicle cross-section coloured by wall heat flux q_wall (W/m²)\n'
                 '(transition visible as colour jump at x_tr)')
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    plt.show()

# HEAT TRANSFER FULL GRAPH
def plot_heat_transfer_physics(x, y, theta, x_panel, delta, T, Taw, q_wall, x_tr,
                                savename='heat_transfer_physics.png'):
    fig, ax = plt.subplots(figsize=(15, 5))

    # delta defined at x[1:] — use surface nodes as base
    x_surface = x[1:]
    y_surface = y[1:]

    nx = -jnp.sin(theta)
    ny =  jnp.cos(theta)

    # Actual BL outer edge (matches table delta values)
    x_outer     = x_surface + delta * nx
    y_outer     = y_surface + delta * ny

    # Visually scaled BL for readability (×3, labelled)
    vis_scale   = 3.0
    x_outer_vis = x_surface + delta * nx * vis_scale
    y_outer_vis = y_surface + delta * ny * vis_scale

    # ── 1. Vehicle body filled by q_wall ──────────────────
    q_norm = mcolors.Normalize(vmin=0.0, vmax=float(q_wall.max()))
    q_cmap = cm.get_cmap('YlOrRd')

    for i in range(N):
        xs = [float(x[i]),   float(x[i+1]),
              float(x[i+1]), float(x[i])]
        ys = [float(y[i]),   float(y[i+1]),
              0.0,             0.0]
        ax.fill(xs, ys, color=q_cmap(q_norm(float(q_wall[i]))),
                alpha=0.95, zorder=2)

    # ── 2. BL band filled by T edge (visual scale) ────────
    T_norm = mcolors.Normalize(vmin=float(T.min()), vmax=float(Taw.max()))
    T_cmap = cm.get_cmap('cool')

    for i in range(N - 1):
        xs = [float(x_surface[i]),      float(x_surface[i+1]),
              float(x_outer_vis[i+1]),  float(x_outer_vis[i])]
        ys = [float(y_surface[i]),      float(y_surface[i+1]),
              float(y_outer_vis[i+1]),  float(y_outer_vis[i])]
        ax.fill(xs, ys, color=T_cmap(T_norm(float(T[i]))),
                alpha=0.85, zorder=3)

    # Last BL patch
    xs = [float(x_surface[-1]), float(x_target),
          float(x_target),      float(x_outer_vis[-1])]
    ys = [float(y_surface[-1]), float(y_target),
          float(y_target),      float(y_outer_vis[-1])]
    ax.fill(xs, ys, color=T_cmap(T_norm(float(T[-1]))),
            alpha=0.85, zorder=3)

    # ── 3. Taw line at visual BL outer edge ───────────────
    ax.plot(x_outer_vis, y_outer_vis,
            '--', color='darkorange', linewidth=2.0, zorder=6,
            label=f'Taw = {float(Taw.mean()):.0f} K  (adiabatic wall)')

    # ── 4. Tw on surface ──────────────────────────────────
    ax.plot(x, y, '-', color='mediumpurple', linewidth=3.0, zorder=7,
            label=f'Tw = {Tw:.0f} K  (wall temperature)')

    # ── 5. Arrows from BL edge → surface (Taw → Tw) ──────
    q_max = float(q_wall.max())
    for i in range(N):
        fi      = float(q_wall[i])
        lw      = 1.2 + 2.8 * (fi / q_max)
        alpha_a = 0.5 + 0.5 * (fi / q_max)
        color   = q_cmap(q_norm(fi))

        ax.annotate('',
            xy     = (float(x_surface[i]),     float(y_surface[i])),     # tip: wall (Tw)
            xytext = (float(x_outer_vis[i]),   float(y_outer_vis[i])),   # tail: BL edge (Taw)
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=lw, mutation_scale=12, alpha=alpha_a),
            zorder=8)

    # ── 6. Transition line ────────────────────────────────
    ax.axvline(float(x_tr), color='black', linestyle='--',
               linewidth=1.5, zorder=9,
               label=f'Transition  x_tr = {float(x_tr):.2f} m')

    # ── Vehicle outline ───────────────────────────────────
    draw_vehicle(ax, x, y, zorder=10)

    # ── Colourbars ────────────────────────────────────────
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes('right', size='2.5%', pad=0.05)
    cax2 = divider.append_axes('right', size='2.5%', pad=0.55)

    sm_q = cm.ScalarMappable(cmap=q_cmap, norm=q_norm)
    sm_q.set_array([])
    fig.colorbar(sm_q, cax=cax1, label='q_wall (W/m²)')

    sm_T = cm.ScalarMappable(cmap=T_cmap, norm=T_norm)
    sm_T.set_array([])
    fig.colorbar(sm_T, cax=cax2, label='T edge (K)')

    # ── Annotations ───────────────────────────────────────
    ax.annotate('BL (×3 visual scale)',
                xy     = (float(x_outer_vis[4]), float(y_outer_vis[4])),
                xytext = (float(x_outer_vis[4]) - 0.4,
                          float(y_outer_vis[4]) + 0.02),
                fontsize=8, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0),
                zorder=11)

    ax.annotate(f'ΔT = Taw − Tw = {float(Taw.mean()) - Tw:.0f} K\ndrives heat into wall',
                xy=(1.4, y_target * 0.35),
                fontsize=8, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                zorder=11)

    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(-y_target * 0.2, y_target * 4.0)
    ax.set_xlim(-0.05, x_target + 0.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(
        f'Hypersonic heat transfer physics  —  M={M_inf}, h=20 km\n'
        f'Body: q_wall  |  BL (×3 visual): T edge  |  '
        f'Arrows: Taw→Tw  |  ΔT = {float(Taw.mean()) - Tw:.0f} K',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.show()
# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 12))
fig.subplots_adjust(hspace=0.45, wspace=0.35)
fig.suptitle(f"Hypersonic Surface Diagnostic  (M={M_inf}, h=20 km)", fontsize=13)

# 1 — Geometry
ax = axes[0, 0]
ax.plot(x, y, 'o-', label='Surface')
ax.plot([0, x_target], [0, 0], '--', color='grey', label='Centreline')
ax.plot([x_target, x_target], [0, y_target], 'k', linewidth=2.5, label='Base')
ax.plot(x_target, y_target, 'r*', markersize=12, label='Target')
ax.set_ylim(-0.02, y_target * 2.5)
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Surface Geometry')

# 2 — Pressure
ax = axes[0, 1]
ax.plot(x_panel, p, 'o-', color='royalblue')
ax.grid(True)
ax.set_xlabel('x (m)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Newtonian Pressure')

# 3 — Temperature
ax = axes[0, 2]
ax.plot(x_panel, T, 'o-', color='tomato', label='T')
ax.plot(x_panel, Taw, 'o--', color='orange', label='Taw (lam/turb)', linewidth=1.5)
ax.axhline(Tw, linestyle=':', color='purple', label=f'Tw={Tw:.0f} K')
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Isentropic Temperature')

# 4 — BL thickness
ax = axes[1, 0]
ax.plot(x_panel, delta_lam,  'o--', label='Laminar',    color='steelblue')
ax.plot(x_panel, delta_turb, 's--', label='Turbulent',  color='darkorange')
ax.plot(x_panel, delta,      'k-',  label='Selected',   linewidth=2)
ax.axvline(float(x_tr), color='red', linestyle=':', label=f'x_tr={float(x_tr):.2f} m')
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('δ (m)')
ax.set_title('Boundary Layer Thickness')

# 5 — Pressure gradient
ax = axes[1, 1]
ax.plot(x_panel, dp_dx, 'o-', color='mediumseagreen')
ax.axhline(0, color='k', linewidth=0.8)
ax.grid(True)
ax.set_xlabel('x (m)')
ax.set_ylabel('dp/dx (Pa/m)')
ax.set_title('Pressure Gradient  (>0 = adverse)')

# 6 — Re_theta / M: transition criterion
ax = axes[1, 2]
M_local=local_mach(p)
criterion = Re_theta / M_local
ax.plot(x_panel, criterion, 'o-', color='mediumpurple')
ax.axhline(400.0, color='red', linestyle='--', label='Transition criterion = 400')
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('Re_θ / M_local')
ax.set_title('Transition Criterion')

# ─────────────────────────────────────────────
# Figure 2 — Heat transfer  (separate window)
# Cf, St, q_wall on one combined plot with twin axes
# ─────────────────────────────────────────────
fig, axes_ht = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle(f"Heat Transfer  (M={M_inf}, h=20 km)", fontsize=13)
 
# St
ax = axes_ht[1]
ax.plot(x_panel, St_ref, 'o-', color='teal', label='St')
ax.axvline(float(x_tr), color='red', linestyle=':', label=f'x_tr = {float(x_tr):.2f} m')
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('St')
ax.set_title('Stanton Number')
 
# q_wall
ax = axes_ht[2]
ax.plot(x_panel, q_wall, 'o-', color='crimson', label='q_wall')
ax.axvline(float(x_tr), color='red', linestyle=':', label=f'x_tr = {float(x_tr):.2f} m')
ax.grid(True)
ax.legend(fontsize=8)
ax.set_xlabel('x (m)')
ax.set_ylabel('q_wall (W/m²)')
ax.set_title('Wall Heat Flux')
 
plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
plt.savefig('heat_transfer.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: heat_transfer.png")
 
# ─────────────────────────────────────────────
# y-coordinate at each panel midpoint (needed for BL outer edge)
y_panel = 0.5 * (y[:-1] + y[1:])
 
plot_pressure_arrows(x, y, theta, dp_dx)
plot_temperature_bl(x, y, theta, x_panel, delta, T, dT_dx)
plot_heatflux_arrows(x, y, theta, q_wall, x_tr)
plot_heat_transfer_physics(x, y, theta, x_panel, delta, T, Taw, q_wall, x_tr)