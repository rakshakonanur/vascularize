import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl               import avg, jump, dot, grad, min_value
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx import (fem, io, mesh, log)
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting, LinearProblem, set_bc
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from typing import List, Optional
from dolfinx.io import XDMFFile

"""
    From the DOLFINx tutorial: Mixed formulation of the Poisson equation
    https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_mixed-poisson.html

    More tutorials: https://github.com/Simula-SSCP/SSCP_lectures/

    References for the projector: 
        https://github.com/michalhabera/dolfiny/blob/202e43711c54bb5d376a6e622e0bc896a20102dd/src/dolfiny/projection.py#L8-L48
        https://github.com/ComputationalPhysiology/oasisx/blob/e8e1b84af0b8675ad57090eddbcd9f08a3b2b63c/src/oasisx/function.py#L1-L124
        https://github.com/Simula-SSCP/SSCP_lectures/blob/main/L12%20(FEniCS%20Intro)/L03_FEniCS_Darcy.ipynb (cloned in SSCP_lectures repo)
        hherlyng/DG_advection_diffusion.py
          
"""

print = PETSc.Sys.Print

# MESHTAGS
LEFT    = 1
RIGHT   = 2
BOT     = 3
TOP     = 4

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def create_rectangle_mesh_with_tags(L: float, H: float, N_cells: int):
        mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]),np.array([L, H])], [N_cells,N_cells],
                                           cell_type = dfx.mesh.CellType.quadrilateral, ghost_mode = dfx.mesh.GhostMode.shared_facet)

        def left(x):   return np.isclose(x[0], 0.0)
        def right(x):  return np.isclose(x[0], L)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x):    return np.isclose(x[1], H)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags

class PerfusionSolver:

    def __init__(self, H : float,
                       L : float,
                       N_cells: int,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False,
                       T : float = 1.0,
                       dt : float = 0.01):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.H = H
        self.L = L
        self.N = N_cells
        self.mesh, self.ft = create_rectangle_mesh_with_tags(L, H, N_cells=N_cells)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output
        self.T = T
        self.dt = dt
        self.num_timesteps = int(self.T/self.dt)

    def setup(self):
        ''' Setup the solver. '''

        # k = self.element_degree
        k = 1
        P_el = element("DG", self.mesh.basix_cell(), k-1)
        u_el = element("RT", self.mesh.basix_cell(), k, shape=(self.mesh.geometry.dim,))
        M_el = mixed_element([P_el, u_el])

        # Define function spaces
        M = dfx.fem.functionspace(self.mesh, M_el) # Mixed function space
        W, _ = M.sub(0).collapse()  # Pressure function space
        V, _ = M.sub(1).collapse()  # Velocity function space
        (p, u) = ufl.TrialFunctions(M) # Trial functions for pressure and velocity
        (v, w) = ufl.TestFunctions(M) # Test functions for pressure and velocity

        dx = Measure("dx", self.mesh) # Cell integrals

        # Facet normal and integral measures
        n  = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure('dx', domain=self.mesh) # Cell integrals
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.ft) # Exterior facet integrals
        dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.ft) # Interior facet integrals
        deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt))

        kappa = 1 # convert from cm^2/(Pa s) to cm^2/(dyne s)
        mu = 1
        kappa_over_mu = dfx.default_scalar_type(kappa/mu)
        Kinv  = fem.Constant(self.mesh, kappa_over_mu**-1)
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        h = (hf('+') + hf('-'))/2 # Average cell diameter on facets
        nitsche = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # Nitsche parameter
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(40.0)) # Upwinding parameter
        nitsche_outflow = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10000.0)) # Nitsche parameter
        phi = fem.Constant(self.mesh, dfx.default_scalar_type(0.1)) # Porosity of the medium, ranging from 0 to 1
        S = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term

        p_in = fem.Constant(self.mesh, dfx.default_scalar_type(1.0)) # External pressure on the boundary
        p_out = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # External pressure on the boundary
        a = Kinv*inner(u, w) * ufl.dx - inner(p, div(w)) * ufl.dx + inner(div(u), v) * ufl.dx
        # a += alpha('+') / hf('+') * dot(jump(v, n), jump(p, n)) * dS
        f = -inner(S, v) * ufl.dx  # residual form of our equation
        f += -inner(dot(w,n), p_in)*self.ds(LEFT) - inner(dot(w,n), p_out)*self.ds(RIGHT)

        def f1(x):
            values = np.zeros((2, x.shape[1]))
            return values
        
        def f2(x):
            values = np.zeros((2, x.shape[1]))
            return values
        
        bc_top  = dfx.fem.Function(V)
        bc_bottom = dfx.fem.Function(V)
        bc_top.interpolate(f1)
        bc_bottom.interpolate(f2)
        fdim = self.mesh.topology.dim -1
        facets_top = mesh.locate_entities_boundary(self.mesh, fdim, lambda x: np.isclose(x[1], self.H))
        dofs_top = fem.locate_dofs_topological((M.sub(1), V), fdim, facets_top)
        facets_bottom = mesh.locate_entities_boundary(self.mesh, fdim, lambda x: np.isclose(x[1], 0.0))
        dofs_bottom = fem.locate_dofs_topological((M.sub(1), V), fdim, facets_bottom)
        bcs = [ dfx.fem.dirichletbc(bc_top, dofs_top, M.sub(1)),
                dfx.fem.dirichletbc(bc_bottom, dofs_bottom, M.sub(1))
        ]

        self.a = a
        self.f = f

        # Apply Dirichlet BCs
        self.bcs = bcs
        log.set_log_level(log.LogLevel.INFO)
        self.problem = LinearProblem(self.a, self.f, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                        "ksp_monitor_true_residual": None,
                                                        "ksp_converged_reason": None,
                                                        "pc_factor_mat_solver_type": "mumps"
                                                    })
        
        try:
            w_h = self.problem.solve()
            print(w_h.x.array, self.problem.solver.getConvergedReason())
        except PETSc.Error as e:  # type: ignore
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

        p_h, u_h = w_h.split()
 
        with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/p.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            DG1 = element("Lagrange", self.mesh.basix_cell(), degree=1)
            p_interp = fem.Function(fem.functionspace(self.mesh, DG1))
            p_interp.interpolate(p_h)
            file.write_function(p_interp)

        with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/sigma.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            P1 = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
            sigma_interp = fem.Function(fem.functionspace(self.mesh, P1))

            # Interpolate the data
            sigma_interp.interpolate(u_h)

            # Write interpolated function
            file.write_function(sigma_interp)

    def run(self):
        ''' Run the solver. '''
        for _ in range(self.num_timesteps):
            self.t += self.dt
            print(f"Time: {self.t}")
            w_h = self.problem.solve()

            p_h, u_h = w_h.split()

            with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/p.xdmf", "w") as file:
                file.write_mesh(self.mesh)
                file.write_function(p_h)

            with io.XDMFFile(self.mesh.comm, "out_mixed_poisson/sigma.xdmf", "w") as file:
                file.write_mesh(self.mesh)
                P1 = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
                sigma_interp = fem.Function(fem.functionspace(self.mesh, P1))

                # Interpolate the data
                sigma_interp.interpolate(u_h)

                # Write interpolated function
                file.write_function(sigma_interp)

            

if __name__ == '__main__':

    comm = MPI.COMM_WORLD # MPI communicator
    write_output = True

    T = 1.5 # Final simulation time
    dt = 0.01 # Timestep size
    L = 2.0 # Length of the domain
    H = 1.0 # Height of the domain
    N = 100 # Number of mesh cells
    # N = int(argv[1]) # Number of mesh cells
    D_value = 1e-2 # Diffusion coefficient
    k = 1 # Finite element polynomial degree

    # Create transport solver object
    perfusion_sim = PerfusionSolver(L=L,
                                    H=H,
                                    N_cells=N,
                                    D_value=D_value,
                                    element_degree=k,
                                    write_output=write_output,
                                    T = T,
                                    dt = dt)
    perfusion_sim.setup()