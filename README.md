# vascularize

New repository for generating synthetic vasculature in a given domain and running 0D/1D simulations. 

Workflow: Currently, the environments for running both FEniCSx and svVascularize have not been integrated. As of now, different environments must be used for the vascular generation and for the Darcy solve. 

Steps:
1. Run src/main.py to generate synthetic vasculature and required reduced-order simulation. This is stored in ./output/
2. Use geometry/mesh.py to generate the 3d --> 1d reduced files from the 1D simulation results. This will make it easier for post-processing and linking to other solvers. This also generates the 3D mesh of the bioreactor domain.
3. Run voronoi/tesselate_updated.py to tesselate the bioreactor mesh into radius-weighted Voronoi territories. This also assigns the converged outlet pressures from the 1D simulation to the territories.
4. Run solves/darcy.py. This uses the terrority outlet pressures from the step 3 as source pressures. Solves a single-compartment Darcy model.
    
Coupling:
For coupling of the simulations, run src/coupler.py. Make sure that the directory points to the 1D results from svVascularize. This coupler iterates the 1D ROM NS and the 3D Darcy solver until the source/outlet flow rates at the interface reach a convergence criteria. Saves MBF and 1D NS files in coupled/ folder, stored by iteration number.
