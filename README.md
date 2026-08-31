# MSc_Project
These are all the codes used during my MSc Project, their outputs are found in the report. 
The codes entitled 'convectionfix_CLT.py' and 'convectionfix_reentry_CLT.py' are the codes used to calculate boundary layer thicknesses, heat fluxes, expansions and thermal stresses for both missile and re-entry applications respectively.
The code entitled 'threed.py' transforms the geometry and physics to 3D with some graph supplied from 'geometry_with_physics.py'.
The codes 'run_inviscid.py', 'postprocess.py', and 'flow_viz.py' use JAX-Fluids to mesh the flow around the vehicle, obtain local parameters and input that onto the heat transfer equation to compare to the Newtonian estimate.
'sensitivity_fix.py' and 'sensitivity_fix_missile.py' does the sensitivity analysis and tests to ensure gradients can be used in an optimiser.
