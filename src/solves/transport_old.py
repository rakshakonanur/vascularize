import os
from pyclbr import Function
import ufl
import numpy   as np
import dolfinx as dfx
import vtk
import sys
import adios4dolfinx

from ufl               import avg, jump, dot, grad, div, inner, SpatialCoordinate, conditional
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx.io        import XDMFFile

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vtk.util.numpy_support import vtk_to_numpy

print = PETSc.Sys.Print

# MESHTAGS
OUTLET = 1
INLET = 2
LEFT = 3
RIGHT = 4
TOP = 5
BOTTOM = 6
WALL = 7
OTHER = 0

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def import_velocity(xdmf_file):
    """
    Import a mesh from an XDMF file.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(xdmf_file)
    reader.Update()

    mesh = reader.GetOutput()
    velocity  =  mesh.GetPointData().GetArray("f")
    velocity_np = vtk_to_numpy(velocity)  # shape: (N, 3)
    print("Velocity shape:", velocity_np.shape)
    return velocity_np
    return velocity


def import_mesh(xdmf_file):
    """
    Import a mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        # Read mesh and create connectivities
        mesh = xdmf.read_mesh(name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)

        # Read vertex tags
        mesh_facets = xdmf.read_meshtags(mesh, name="mesh_tags")  # Tags on vertices (dim=0)
        assert mesh_facets.dim == 0, "mesh_tags must be on vertices (dim=0)"
        mesh_coords = mesh.geometry.x

        # --- Identify boundary (external) vertices ---
        boundary_facets = dfx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
        )
        boundary_vertex_links = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
        boundary_vertices = np.unique(
            np.hstack([boundary_vertex_links.links(f) for f in boundary_facets])
        )

        # --- Separate tagged vertices into internal / external ---
        tagged_vertices = mesh_facets.indices
        tagged_values = mesh_facets.values

        is_boundary = np.isin(tagged_vertices, boundary_vertices)
        external_vertices = tagged_vertices[is_boundary]
        internal_vertices = tagged_vertices[~is_boundary]

        internal_tags = dfx.mesh.meshtags(mesh, 0, internal_vertices, tagged_values[~is_boundary])
        external_tags = dfx.mesh.meshtags(mesh, 0, external_vertices, tagged_values[is_boundary])
    
    internal_facet_tags = convert_vertex_tags_to_facet_tags(mesh, internal_tags)
    external_facet_tags = convert_vertex_tags_to_facet_tags(mesh, external_tags)

    return mesh, internal_facet_tags, external_facet_tags

def convert_vertex_tags_to_facet_tags(mesh, vertex_tags):
    """
    Given vertex-based tags (dim=0), convert them to facet-based tags (dim=dim-1)
    by assigning each facet the most common tag among its vertices.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Create needed connectivities
    mesh.topology.create_connectivity(fdim, 0)  # facet -> vertex

    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    facet_indices = []
    facet_values = []

    for facet in range(mesh.topology.index_map(fdim).size_local):
        vertex_ids = facet_to_vertex.links(facet)
        tags_on_vertices = vertex_tags.values[np.isin(vertex_tags.indices, vertex_ids)]

        if len(tags_on_vertices) > 0:
            # Assign the first (or most common) tag
            tag = np.bincount(tags_on_vertices).argmax()
            facet_indices.append(facet)
            facet_values.append(tag)

    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values = np.array(facet_values, dtype=np.int32)

    # # Write mesh and tags to output files
    # if mesh.comm.rank == 0:
    #     out_str = './output/mesh_tags.xdmf'
    #     with XDMFFile(mesh.comm, out_str, 'w') as xdmf_out:
    #         xdmf_out.write_mesh(mesh)
    #         xdmf_out.write_meshtags(dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_values), mesh.geometry)

    return dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_values)

def project_velocity(mesh, velocity):
    """
    Project the velocity onto the mesh.
    """
    u_p2 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    # u_p1 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))

    # # Define function spaces
    # W = dfx.fem.functionspace(mesh, u_p2) # lagrange function space
    # V = dfx.fem.functionspace(mesh, u_p1) # DG Velocity space

    # vel = dfx.fem.Function(V)  # Create a function in the DG space
    # # vel.x.array[:] = velocity.flatten()  # Set the velocity values

    # # Ensure velocity array matches the function layout
    # dofmap = V.dofmap
    # assert velocity.shape[0] == V.dofmap.index_map.size_local, (
    #     f"Velocity shape mismatch: expected {V.dofmap.index_map.size_local}, got {velocity.shape[0]}"
    # )

    # # Flatten in correct order (component-major)
    # vel.x.array[:] = velocity.reshape(-1)

    # projector = Projector(W)
    # u_proj = projector(vel)  # Project the velocity onto the mesh

    #  Define function spaces
    W = dfx.fem.functionspace(mesh, u_p2) # lagrange function space
    u_proj = dfx.fem.Function(W)  # Create a function in the Lagrange space
    u_proj.x.array[:] = velocity.flatten()  # Set the velocity values    

    # Write the projected velocity to a file
    if mesh.comm.rank == 0:
        out_str = './output/velocity_projected.xdmf'
        with XDMFFile(mesh.comm, out_str, 'w') as xdmf_out:
            xdmf_out.write_mesh(mesh)
            xdmf_out.write_function(u_proj, 0.0)

    return u_proj

class Transport:

    def __init__(self, vel_file: str,
                       xdmf_file: str,
                       T: float,
                       dt: float,
                       D_value: float,
                       element_degree: int,
                       write_output: str = False):
        ''' Constructor. '''

        # Create mesh and store attributes
        self.mesh, self.interior_tags, self.exterior_tags = import_mesh(xdmf_file)
        self.velocity = import_velocity(vel_file)
        self.u_proj = project_velocity(self.mesh, self.velocity)
        self.D_value = D_value
        self.element_degree = element_degree
        self.write_output = write_output

        # Temporal parameters
        self.T = T
        self.dt = dt
        self.t = 0
        self.num_timesteps = int(T / dt)

    def setup(self):
        """ Set up the problem. """

        # Simulation parameters
        self.D_value = 1e-6  # Diffusion coefficient
        self.k = 1 # Element degree
        self.t = 0 # Initial time
        self.T = 1000.0 # Final time
        self.dt = 1.0 # Timestep size
        self.num_timesteps = int(self.T / self.dt)
        self.n = ufl.FacetNormal(self.mesh)
        self.dx = ufl.Measure("dx", domain=self.mesh) # Cell integrals
        self.ds = ufl.Measure('ds', domain=self.mesh, subdomain_data=self.exterior_tags) # Exterior facet integrals
        self.dS = ufl.Measure('dS', domain=self.mesh, subdomain_data=self.interior_tags) # Interior facet integrals

        # Function spaces
        Pk_vec = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
        V = dfx.fem.functionspace(self.mesh, Pk_vec) # function space for velocity
        self.u = dfx.fem.Function(V) # velocity
        Pk = element("Lagrange", self.mesh.basix_cell(), degree=1)
        self.u.x.array[:] = self.u_proj.x.array[:] # Project the velocity onto the mesh
        # self.u.x.array[:] = self.velocity # Set velocity u=0.0 everywhere
        # self.u.x.array[0::3] = 0.10 # Set velocity u=0.1 in x-direction
        DG = element("DG", self.mesh.basix_cell(), degree=self.element_degree) # Discontinuous Galerkin element
        W = dfx.fem.functionspace(self.mesh, DG) # function space for concentration
        deltaT = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.dt)) # Time step size

        print("Total number of concentration dofs: ", W.dofmap.index_map.size_global, flush=True)

        # === Trial, test, and solution functions ===
        c, w = ufl.TrialFunction(W), ufl.TestFunction(W)
        self.c_h = dfx.fem.Function(W) # concentration at current time step
        self.c_ = dfx.fem.Function(W) # concentration at previous time step

        self.bc_func = dfx.fem.Function(W) # Boundary condition function
        self.bc_func.interpolate(lambda x: 1 + np.zeros_like(x[0])) # Set initial condition at inlet
    
        #------VARIATIONAL FORM------#
        self.D = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(self.D_value)) # Diffusion coefficient
        f = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        alpha = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # SIPG penalty parameter

        u_mag = ufl.sqrt(ufl.dot(self.u, self.u) + 1e-10) # Magnitude of velocity
        Pe = u_mag * hf / (2 * self.D) # Peclet number
        beta = (ufl.cosh(Pe)/ufl.sinh(Pe))- (1/Pe)
        tau = hf * beta/ (2 * u_mag + 1e-10) # Stabilization parameter
        nitsche = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10.0)) # Nitsche parameter

        # Variational form
        a_time     = c * w / deltaT * self.dx
        a_advect   = - dot(c*self.u, grad(w)) * self.dx # Advection term
        a_diffuse  = dot(self.D * grad(c), grad(w)) * self.dx

        a = a_time + a_advect + a_diffuse
        L = (self.c_ / deltaT + f) * w * self.dx

        # Diffusive terms with interior penalization
        a  += self.D('+') * alpha('+') / hf('+') * dot(jump(w, self.n), jump(c, self.n)) * self.dS
        a  -= self.D('+') * dot(avg(grad(w)), jump(c, self.n)) * self.dS
        a  -= self.D('+') * dot(jump(w, self.n), avg(grad(c))) * self.dS

        # SUPG terms
        # residual = dot(self.u, grad(self.c_h)) - self.D * div(grad(self.c_h)) + (self.c_h - self.c_) / deltaT - f
        v_supg = tau * dot(self.u, grad(w))

        a_supg = v_supg * (c / deltaT + dot(self.u, grad(c)) - self.D * div(grad(c))) * self.dx
        L_supg = v_supg * (self.c_ / deltaT + f) * self.dx

        # Impose BC using Nitsche's method--- NEED TO FIX
        a_nitsche = nitsche('+') / hf('+') * c('+') * w('+') * self.dS(1)
        L_nitsche = nitsche('+') / hf('+') * self.bc_func('+') * w('+')* self.dS(1)

        # Upwind velocity
        un = (dot(self.u, self.n) + abs(dot(self.u, self.n))) / 2.0
        a_upwind = dot(jump(w), un('+') * c('+') - un('-') * c('-')) * self.dS # + dot(w,un*c) * self.ds

        # Enforce Neumann BC at the outlet
        # u_n = dot(self.u, self.n)
        # c_ext = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(0)) # Set external concentration to zero
        # L_outflow = - conditional(avg(un) > 0, un('+') * c_ext * avg(w), 0.0) * self.dS(OUTLET)

        # outflux  = c('+')*dot(self.u('+'), self.n('+')) # Only advective flux on outflow boundary, diffusive flux is zero
        # u_normal = dot(self.u, self.n) # The normal velocity

        # # Create conditional expressions
        # cond  = ufl.lt(u_normal, 0.0) # Condition: True if the normal velocity is less than zero, u.n < 0
        # minus = ufl.conditional(cond, 1.0, 0.0) # Conditional that returns 1.0 if u.n <  0, else 0.0. Used to "activate" terms on the influx  boundary
        # plus  = ufl.conditional(cond, 0.0, 1.0) # Conditional that returns 1.0 if u.n >= 0, else 0.0. Used to "activate" terms on the outflux boundary
        
        # # Add outflux term to the weak form
        # a += plus('+')*outflux* w('+') * self.dS(OUTLET)

        a += a_nitsche + a_upwind #+ a_supg# + a_outflow
        L += L_nitsche #+ L_supg# + L_outflow


        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters)

        # No strong Dirichlet BCs
        self.bcs = []

        # self.c_.interpolate(self.bc_func) # Previous timestep
        self.c_.x.array[:] = 0.0# Previous timestep

        # Create output function in P1 space
        P1 = element("Lagrange", self.mesh.basix_cell(), degree=1) # Linear Lagrange elements
        self.c_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, P1))
        self.c_out.interpolate(self.bc_func)
        self.u_out = dfx.fem.Function(dfx.fem.functionspace(self.mesh, Pk_vec))

        # Interpolate it into the velocity function
        self.u_out.x.array[:] = self.u.x.array.copy()
        self.u_out.x.scatter_forward()

        # === Total concentration integral ===
        self.total_c = dfx.fem.form(self.c_h * self.dx)
        # self.error_form = residual**2 * self.dx # calculates square of L2 error over the interior facets

        # === Linear system ===
        self.A = assemble_matrix(self.a_cpp, bcs=self.bcs)
        self.A.assemble()

        self.b = create_vector(self.L_cpp) # Create RHS vector

        # Configure direct solver
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverType('mumps')
        self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization

        if self.write_output:
            # Create output file for the concentration
            out_str = './output/cube_conc_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_c = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_c.write_mesh(self.mesh)
            self.c_out.interpolate(self.c_h)  # Interpolate the concentration function
            self.xdmf_c.write_function(self.c_out, self.t)

            out_str = './output/cube_vel_D=' + f'{self.D.value}' + '.xdmf'
            self.xdmf_u = dfx.io.XDMFFile(self.mesh.comm, out_str, 'w')
            self.xdmf_u.write_mesh(self.mesh)
            self.xdmf_u.write_function(self.u_out, self.t)

    def assemble_transport_RHS(self):
        """ Assemble the right-hand side of the variational problem. """
    
        # Zero entries to avoid accumulation
        with self.b.localForm() as b_loc: b_loc.set(0)

        # Assemble vector and set BCs
        assemble_vector(self.b, self.L_cpp)
        apply_lifting(self.b, [self.a_cpp], bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # MPI communication
        set_bc(self.b, bcs=self.bcs)

    def run(self):
        """ Run transport simulations. """

        for _ in range(self.num_timesteps):

            self.t += self.dt

            self.assemble_transport_RHS()

            # Compute solution to the advection-diffusion equation and perform parallel communication
            self.solver.solve(self.b, self.c_h.x.petsc_vec)
            self.c_h.x.scatter_forward()

            # Update previous timestep
            self.c_.x.array[:] = self.c_h.x.array.copy()

            # Print stuff
            print(f"Timestep t = {self.t}")

            print("Maximum concentration: ", self.mesh.comm.allreduce(self.c_h.x.array.max(), op=MPI.MAX))
            print("Minimum concentration: ", self.mesh.comm.allreduce(self.c_h.x.array.min(), op=MPI.MIN))

            total_c = dfx.fem.assemble_scalar(self.total_c)
            total_c = self.mesh.comm.allreduce(total_c, op=MPI.SUM)
            print(f"Total concentration: {total_c:.2e}")

            # error_squared = dfx.fem.assemble_scalar(dfx.fem.form(self.error_form)) # assemble into a scalar, by converting symbolic UFL form to Fenicsx
            # total_residual = self.mesh.comm.allreduce(error_squared, op=MPI.SUM) # gather all the errors from all processes in case of parallel execution
            # print(f"Total residual: {np.sqrt(total_residual):.2e}")


            if self.write_output:
                # Write to file
                self.c_out.interpolate(self.c_h)  # Interpolate the concentration function
                self.xdmf_c.write_function(self.c_out, self.t)

                self.u_out.interpolate(self.u)
                self.xdmf_u.write_function(self.u_out, self.t)

        plt.close()

    

class Projector():
    def __init__(self, V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = inner(u, v) * ufl.dx
        # self.V = V
        self.u = dfx.fem.Function(V) # Create function
        self.a_cpp = dfx.fem.form(a, jit_options=jit_parameters)

    def __call__(self, f):
        v = ufl.TestFunction(self.u.function_space)
        L = inner(f, v) * ufl.dx
        
        self.L_cpp = dfx.fem.form(L, jit_options=jit_parameters) # Compile form
        self.A = assemble_matrix(self.a_cpp, bcs=[])
        
        self.A.assemble()

        self.b = create_vector(self.L_cpp) # Create RHS vector

        self.solver = PETSc.KSP().create(MPI.COMM_WORLD)
        self.solver.setOperators(self.A)
        self.solver.setType('preonly')
        self.solver.getPC().setType('lu')
        self.solver.getPC().setFactorSolverType('mumps')
        self.solver.getPC().getFactorMatrix().setMumpsIcntl(icntl=58, ival=1) # activate symbolic factorization
        with self.b.localForm() as b_loc: b_loc.set(0)

        # Assemble vector and set BCs
        assemble_vector(self.b, self.L_cpp)
        # apply_lifting(self.b, [self.a_cpp], bcs = [])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) # MPI communication
        # set_bc(self.b, bcs=[])
        
        self.solver.solve(self.b, self.u.x.petsc_vec)

        # Destroy PETSc linear algebra objects and solver
        self.solver.destroy()
        self.A.destroy()
        self.b.destroy()

        return self.u

if __name__ == "__main__":
    # Example usage
    transport_solver = Transport(vel_file="./out_darcy/u_p0_000000.vtu",
                                 xdmf_file = "../geometry/bioreactor.xdmf",
                                  T=100.0, dt=0.1, D_value=1e-6, element_degree=0, write_output=True)
    transport_solver.setup()
    transport_solver.run()