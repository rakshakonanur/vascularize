# vascularize

New repository for generating synthetic vasculature in a given domain and running 0D/1D simulations. 

## Workflow: 
Currently, the environments for running both FEniCSx and svVascularize have not been integrated. As of now, different environments must be used for the vascular generation and for the Darcy solve. Below are the steps for running the Darcy single-compartment equations, starting from generating the synthetic vasculature.

### Steps:
1. To generate synthetic vasculature and required reduced-order simulation, in the src/ folder, run:
```bash
python3 main.py
```
 This is stored in ./output/. The following workflow works with both 0D and 1D simulation.

2. To generate the 3d --> axially-reduced files from the 0D/1D simulation results, in the src/geometry/ folder run:
```bash
python3 mesh.py
```
This will make it easier for post-processing and linking to other solvers. This also generates the 3D mesh of the bioreactor domain.

3. To tesselate the bioreactor mesh into radius-weighted Voronoi territories, in the src/voronoi/ folder, run: 
```bash
python3 tesselate_updated.py
```
This also assigns the converged outlet pressures from the 1D simulation to the territories.

4. Then, to solve the Darcy single-compartment equations, run either of the following commands. This uses the terrority outlet pressures from the step 3 as source pressures. Solves a single-compartment Darcy model. The first used a collapsed P1 solver, while the second uses a BDM/DG solver:
```bash
python3 darcy_P1.py
```
```bash
python3 darcy_BDM.py
```
    
## Coupling:

There are different coupling options available:
- 1D-1D Coupling (1D-0D-1D): This uses 1D simulation results from both the arterial and venous branches. In the coupling iterations, the resistances of the arterial branch, as well as pressures of the venous branch are updated from the territory-wise flows from the Darcy single-compartment solve. To find the inlet pressures to the 1D venous branch, first a 0D simulation is run on the venous branch, with flow inlet and zero pressure out conditions. The converged inlet pressures us then used as the inlet BC for the 1D venous solve. To run this coupler, in the src/ folder, run:
```bash
python3 coupler_1d.py
```
- 1D-0D Coupling: Uses only 1D-0D coupling, using a similar procedure as above. This avoids the issues with high error (~10-15%) from the above method. To run this coupler, in the src/ folder, run:
```bash
python3 coupler_1d_0d.py
```
- 0D-0D Coupling (in-progress): Uses 0D-0D coupling, with pressure inlet and flow outlet for the arterial branch, and flow inlet and pressure outlet for the venous branch. This currently does not converge. To run this coupler, in the src/ folder, run:
```bash
python3 coupler_0d.py
```

Make sure that the directory points to the 0D/1D results from svVascularize. This coupler iterates the 0D/1D ROM NS and the 3D Darcy solver until the source/outlet flow rates at the interface reach a convergence criteria. Saves MBF and 0D/1D NS files in coupled/ folder, stored by iteration number.