"""
run_inviscid.py  — fixed
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from scipy.interpolate import CubicSpline
from jaxfluids import InputManager, InitializationManager, SimulationManager

# ═══════════════════════════════════════════════════════════════════════
#  FREESTREAM
# ═══════════════════════════════════════════════════════════════════════
M_inf   = 5.0
gamma   = 1.4
T_inf   = 205.0
p_inf   = 5543.0
rho_inf = 0.09427
R       = 287.0
Pr      = 0.71

a_inf = float(np.sqrt(gamma * R * T_inf))
V_inf = M_inf * a_inf

print(f"Freestream: M={M_inf}, V={V_inf:.2f} m/s, T={T_inf} K, "
      f"p={p_inf} Pa, rho={rho_inf} kg/m3")
print("Mode: INVISCID (Euler)")

# ═══════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════
N            = 12
x_target     = 2.5
y_target     = 0.125
panel_length = x_target / N

mean_angle = float(np.arctan(y_target / x_target))
theta_arr  = np.deg2rad(np.linspace(
    np.rad2deg(mean_angle) * 1.8,
    np.rad2deg(mean_angle) * 0.2,
    N
))

dx_arr  = panel_length * np.cos(theta_arr)
dr_arr  = panel_length * np.sin(theta_arr)
x_nodes = np.concatenate([[0.0], np.cumsum(dx_arr)])
r_nodes = np.concatenate([[0.0], np.cumsum(dr_arr)])
x_end   = float(x_nodes[-1])

# ═══════════════════════════════════════════════════════════════════════
#  CUBIC SPLINE: we added a cubic spline to help with the uneven surface of the panels
# ═══════════════════════════════════════════════════════════════════════
cs = CubicSpline(x_nodes, r_nodes, bc_type=((1, 0.0), 'not-a-knot'))

x_chk  = np.linspace(0.0, x_end, 500)
dr_chk = cs(x_chk, 1)
print(f"\n  [Spline] dr/dx at nose : {abs(dr_chk[0]):.2e}  (target ~0)")
print(f"  [Spline] max |dr/dx|   : {abs(dr_chk).max():.4f}")
print(f"  [Spline] r at nose     : {cs(0.0):.2e}  (target 0)")
print(f"  [Spline] r at tail     : {cs(x_end):.6f}  "
      f"(target {r_nodes[-1]:.6f})")
assert abs(cs(0.0))               < 1e-10, "Spline r(0) != 0"
assert abs(dr_chk[0])             < 1e-8,  "Spline dr/dx(0) != 0"
assert abs(cs(x_end) - r_nodes[-1]) < 1e-8, "Spline r(x_end) mismatch"
print("  [Spline] PASSED\n")

# ── Dense table ───────────────────────────────────────────────────────
N_TABLE    = 2000
x_tbl      = np.linspace(0.0, x_end, N_TABLE)
r_tbl      = cs(x_tbl).astype(np.float64)

x_tbl_list = x_tbl.tolist()
r_tbl_list = r_tbl.tolist()

levelset_str = (
    f"lambda x,y,z: jnp.sqrt(y**2+z**2) - jnp.where("
    f"x > {x_end!r}, "
    f"0.0, "
    f"jnp.interp(x, jnp.array({x_tbl_list!r}), jnp.array({r_tbl_list!r})))"
)

print(f"  [Level-set string] length = {len(levelset_str):,} chars")
print(f"  [Level-set string] first 120 chars:")
print(f"    {levelset_str[:120]}...")

# ── Sanity-check the level-set string ────────────────────────────────
import jax.numpy as jnp
_fn = eval(levelset_str)

r_tail = float(cs(x_end))   # body radius at the tail = 0.12480477 m

phi_nose     = float(_fn(0.0,   0.0,    0.0))   # nose tip  → on surface (phi=0)
phi_farfield = float(_fn(1.0,   1.0,    0.0))   # far field → fluid      (phi>0)
phi_axis     = float(_fn(1.0,   0.0,    0.0))   # axis mid  → solid      (phi<0)
phi_tail     = float(_fn(x_end, r_tail, 0.0))   # tail rim  → on surface (phi≈0)

print(f"\n  [Level-set eval] phi at nose tip  (0, 0, 0)          : "
      f"{phi_nose:.2e}  (expect  0.0)")
print(f"  [Level-set eval] phi at far field (1, 1, 0)          : "
      f"{phi_farfield:.4f}  (expect > 0)")
print(f"  [Level-set eval] phi on axis      (1, 0, 0)          : "
      f"{phi_axis:.6f}  (expect < 0)")
print(f"  [Level-set eval] phi at tail rim  (x_end, r_tail, 0) : "
      f"{phi_tail:.2e}  (expect ~0)")

assert abs(phi_nose)  < 1e-8, \
    f"Nose tip should be on surface (phi=0), got {phi_nose:.6e}"
assert phi_farfield   > 0.0,  \
    f"Far field should be fluid (phi>0), got {phi_farfield:.6f}"
assert phi_axis       < 0.0,  \                                                                                                                                                                                   
    f"Axis inside body should be solid (phi<0), got {phi_axis:.6f}"
assert abs(phi_tail)  < 1e-6, \
    f"Tail rim should be on surface (phi=0), got {phi_tail:.6e}"
print("  [Level-set eval] PASSED\n")

# ═══════════════════════════════════════════════════════════════════════
#  DOMAIN
# ═══════════════════════════════════════════════════════════════════════
h = 0.025

X_MIN, X_MAX = -0.4, 3.0
Y_MIN, Y_MAX = -0.5, 0.5
Z_MIN, Z_MAX = -0.5, 0.5

NX = round((X_MAX - X_MIN) / h)
NY = round((Y_MAX - Y_MIN) / h)
NZ = round((Z_MAX - Z_MIN) / h)
print(f"Grid: {NX} x {NY} x {NZ} = {NX*NY*NZ:,} cells, cell size h={h} m")

t_flow    = x_target / V_inf
t_end_sim = 8.0 * t_flow
dt_out    = t_flow / 5.0

prims_inflow = {"rho": rho_inf, "u": V_inf, "v": 0.0, "w": 0.0, "p": p_inf}

# ═══════════════════════════════════════════════════════════════════════
#  CASE SETUP: what are we simulating: the geometry and material properties
# ═══════════════════════════════════════════════════════════════════════
case_setup = {
    "general": {
        "case_name": "hypersonic_vehicle_3D_M5_inviscid",
        "end_time":  float(t_end_sim),
        "save_path": "./results",
        "save_dt":   float(dt_out)
    },
    "domain": {
        "x": {"cells": NX, "range": [X_MIN, X_MAX]},
        "y": {"cells": NY, "range": [Y_MIN, Y_MAX]},
        "z": {"cells": NZ, "range": [Z_MIN, Z_MAX]},
        "decomposition": {"split_x": 1, "split_y": 1, "split_z": 1}
    },
    "boundary_conditions": {
        "primitives": {
            "east":   {"type": "ZEROGRADIENT"},
            "west":   {"type": "DIRICHLET", "primitives_callable": prims_inflow},
            "north":  {"type": "DIRICHLET", "primitives_callable": prims_inflow},
            "south":  {"type": "DIRICHLET", "primitives_callable": prims_inflow},
            "top":    {"type": "DIRICHLET", "primitives_callable": prims_inflow},
            "bottom": {"type": "DIRICHLET", "primitives_callable": prims_inflow}
        },
        "levelset": {
            "east":   {"type": "ZEROGRADIENT"},
            "west":   {"type": "ZEROGRADIENT"},
            "north":  {"type": "ZEROGRADIENT"},
            "south":  {"type": "ZEROGRADIENT"},
            "top":    {"type": "ZEROGRADIENT"},
            "bottom": {"type": "ZEROGRADIENT"}
        }
    },
    "initial_condition": {
        "primitives": prims_inflow,
        "levelset":   levelset_str
    },
    "material_properties": {
        "equation_of_state": {
            "model": "IdealGas",
            "specific_heat_ratio":   gamma,
            "specific_gas_constant": R
        },
        "transport": {
            "dynamic_viscosity": {
                "model": "SUTHERLAND",
                "sutherland_parameters": [1.716e-5, 273.15, 110.4]
            },
            "bulk_viscosity": 0.0,
            "thermal_conductivity": {
                "model": "PRANDTL",
                "prandtl_number": Pr
            }
        }
    },
    "solid_properties": {"temperature": 300.0},
    "nondimensionalization_parameters": {
        "density_reference":     1.0,
        "length_reference":      1.0,
        "velocity_reference":    1.0,
        "temperature_reference": 1.0
    },
    "output": {
        "primitives": ["density", "velocity", "pressure", "temperature"],
        "levelset":   ["levelset", "volume_fraction"]
    }
}

# ═══════════════════════════════════════════════════════════════════════
#  NUMERICAL SETUP: how we are simulating it: physics, math, approx
# ═══════════════════════════════════════════════════════════════════════
numerical_setup = {
    "conservatives": {
        "halo_cells": 5,
        "time_integration": {
            "integrator": "RK3",
            "CFL": 0.2
        },
        "convective_fluxes": {
            "convective_solver": "GODUNOV",
            "godunov": {
                "riemann_solver":          "HLLC",
                "signal_speed":            "EINFELDT",
                "reconstruction_stencil":  "WENO5-Z-ADAP",
                "reconstruction_variable": "PRIMITIVE"
            }
        },
        "dissipative_fluxes": {
            "reconstruction_stencil":    "CENTRAL4-ADAP",
            "derivative_stencil_center": "CENTRAL4-ADAP",
            "derivative_stencil_face":   "CENTRAL4-ADAP"
        },
        "positivity": {"is_interpolation_limiter": True}
    },
    "levelset": {
        "halo_cells": 2,
        "model": "FLUID-SOLID",
        "solid_coupling": {"thermal": "ONE-WAY"},
        "interface_flux": {
            "method": "INTERPOLATION",
            "interpolation_dh": 1.0
        },
        "geometry": {
            "derivative_stencil_normal": "CENTRAL4",
            "subcell_reconstruction": False
        },
        "extension": {
            "primitives": {
                "method": "ITERATIVE",
                "iterative": {
                    "CFL": 0.5,
                    "steps": 20,
                    "residual_threshold": 1e-5
                }
            }
        },
        "mixing": {
            "conservatives": {
                "volume_fraction_threshold":    1e-1,
                "mixing_targets":               1,
                "is_interpolate_invalid_cells": False
            }
        }
    },
    "active_physics": {
        "is_convective_flux":         True,
        "is_viscous_flux":            False,
        "is_heat_flux":               False,
        "is_volume_force":            False,
        "is_surface_tension":         False,
        "is_viscous_heat_production": False
    },
    "active_forcings": {
        "is_mass_flow_forcing":   False,
        "is_temperature_forcing": False
    },
    "precision": {
        "is_double_precision_compute": True,
        "is_double_precision_output":  True
    },
    "output": {
        "derivative_stencil": "CENTRAL4",
        "logging": {"frequency": 10},
        "is_xdmf": False
    }
}

# ═══════════════════════════════════════════════════════════════════════
#  PRE-LAUNCH VERIFICATION
# ═══════════════════════════════════════════════════════════════════════
_cfl_actual = numerical_setup["conservatives"]["time_integration"]["CFL"]
_ls_actual  = case_setup["initial_condition"]["levelset"]

print("=" * 60)
print("  PRE-LAUNCH VERIFICATION")
print("=" * 60)
print(f"  CFL in numerical_setup dict : {_cfl_actual}")
print(f"  Level-set type              : {type(_ls_actual).__name__}")
print(f"  Level-set string length     : {len(_ls_actual):,} chars")
print(f"  Table points embedded       : {N_TABLE}")
print(f"  Table spacing               : "
      f"{x_end/(N_TABLE-1)*1000:.2f} mm  (cell = {h*1000:.1f} mm)")

assert _cfl_actual == 0.2, \
    f"CFL is {_cfl_actual}, not 0.2"
assert isinstance(_ls_actual, str), \
    "Level-set must be a string for JAX-Fluids 0.2.1"
assert len(_ls_actual) > 50000, \
    f"Level-set string too short ({len(_ls_actual)} chars) — table not embedded"

print()
print("  VERIFIED. Watching for in log:")
print("    CFL                       : 0.2  (not 0.4)")
print("    REINITIALIZATION RESIDUAL : < 0.3 at step 0  (was 3.237)")
print("    COUNT INTERP LIMITER      : < 5 at step 10   (was 36)")
print("    MIN DENSITY at step 10    : > 0.01            (was 4.75e-04)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════
input_manager          = InputManager(case_setup, numerical_setup)
initialization_manager = InitializationManager(input_manager)
sim_manager            = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)

print("Simulation complete. Results written to:",
      sim_manager.output_writer.save_path_domain)