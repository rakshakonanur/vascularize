import numpy as np
import ufl
import dolfinx as dfx
from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree
from petsc4py import PETSc
import adios4dolfinx
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting
from dolfinx import fem, io
from basix.ufl import element
from ufl import dot, grad

################################################################################
# 1. Mesh import (same as Darcy)
################################################################################

def import_3d_mesh(xdmf_file: str):
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        m = xdmf.read_mesh(name="Grid")
        m.topology.create_connectivity(m.topology.dim, m.topology.dim-1)
        tags = xdmf.read_meshtags(m, name="mesh_tags")
    return m, tags

def import_mesh(bp_file: str):
    mesh = adios4dolfinx.read_mesh(filename=Path(bp_file), comm=MPI.COMM_WORLD)
    tags = adios4dolfinx.read_meshtags(filename=Path(bp_file),
                                       mesh=mesh, meshtag_name="mesh_tags")
    return mesh, tags

################################################################################
# 2. ADIOS import for flows (same as Darcy)
################################################################################

def import_flow_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD,
                                       function_name="f")
    DG = element("Lagrange", mesh_obj.basix_cell(), 1)
    V = fem.functionspace(mesh_obj, DG)
    q_src = fem.Function(V)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, q_src, name="f", time=t)
        q_src.x.scatter_forward()
        out.append(q_src.copy())
    return out

################################################################################
# 3. Transport Solver
################################################################################

class TransportSolver:

    def __init__(self,
                 bioreactor_domain,
                 mesh_inlet_file, mesh_outlet_file,
                 flow_inlet_file, flow_outlet_file,
                 vel_file,
                 T=10.0, dt=1.0,
                 D_value=1e-3,
                 c_in_value=1.0):

        # === 3D mesh ===
        self.mesh, _ = import_3d_mesh(bioreactor_domain)

        # === velocity field (XDMF or VTU projected to P1) ===
        self.velocity = self._load_velocity(vel_file)

        # === 1D inlet/outlet terminal meshes ===
        self.inlet_mesh,  self.inlet_tags  = import_mesh(mesh_inlet_file)
        self.outlet_mesh, self.outlet_tags = import_mesh(mesh_outlet_file)

        # === Load flow values at terminals ===
        self.q_inlet_fun  = import_flow_data(self.inlet_mesh,  flow_inlet_file)[-1]
        self.q_outlet_fun = import_flow_data(self.outlet_mesh, flow_outlet_file)[-1]

        # === Extract terminal coordinates and flows ===
        self._extract_terminal_data()

        # Time/parameters
        self.T = T
        self.dt = dt
        self.D_value = D_value
        self.c_in_value = c_in_value

    ###########################################################################
    # Load velocity (simple P1 projection for now)
    ###########################################################################
    def _load_velocity(self, xdmf_vfile):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(xdmf_vfile)
        reader.Update()

        mesh = reader.GetOutput()
        velocity  =  mesh.GetPointData().GetArray("f")
        velocity_np = vtk_to_numpy(velocity)  # shape: (N, 3)
        # velocity_np *= 1e3
        print("Velocity shape:", velocity_np.shape)

        uDG = element("DG", self.mesh.basix_cell(), 0, shape=(self.mesh.geometry.dim,))
        uP1 = element("Lagrange", self.mesh.basix_cell(), 1, shape=(self.mesh.geometry.dim,))
        DG = fem.functionspace(self.mesh, uDG)
        P1 = fem.functionspace(self.mesh, uP1)

        # u_DG = fem.Function(DG)
        # u_DG.x.array[:] = velocity_np.flatten()
        # u_DG.x.scatter_forward()
        # u = fem.Function(P1)
        # u.interpolate(u_DG)
        u = fem.Function(P1)
        u.x.array[:] = velocity_np.flatten() 
        u.x.scatter_forward()
        return u

    def _inlet_concentration(self, t: float) -> float:
        """
        Time-dependent inlet concentration c_in(t).

        Right now this is:
            - a square pulse of amplitude self.c_in_value
              between t=10 and t=40 (just as an example).

        Replace this with whatever profile you want (piecewise, sinusoid, etc.)
        If you want it to remain constant, just `return self.c_in_value`.
        """
        # Example: bolus between t=10 and t=40
        t_on  = 0.0
        t_off = 10.0
        if t_on <= t <= t_off:
            return self.c_in_value
        else:
            return 0.0
        # For constant inlet concentration instead, just do:
        # return self.c_in_value

    ###########################################################################
    # Extract terminal coordinates + flows (same as Darcy)
    ###########################################################################
    # ========================================================
    # Step A: Extract terminal coordinates and values
    # ========================================================
    def _extract_terminal_data(self):
        """
        Extract terminal coordinates (marker==2) directly from inlet/outlet meshtags.
        Then sample the nearest pressure and flow values.
        """

        # ------------------------------------------------------
        # 1) Extract terminal coordinates directly from meshtags
        # ------------------------------------------------------
        def extract_terminal_coords(mesh_obj, tags):
            marker = 2
            idx = np.where(tags.values == marker)[0]
            if len(idx) == 0:
                return np.zeros((0, mesh_obj.geometry.dim))

            entity_ids = tags.indices[idx]

            # Directly pick the vertex coordinates (ADIOS tags apply to vertices)
            return mesh_obj.geometry.x[entity_ids]

        # inlet terminals
        self.x_inlet = extract_terminal_coords(self.inlet_mesh, self.inlet_tags)
        # outlet terminals
        self.x_outlet = extract_terminal_coords(self.outlet_mesh, self.outlet_tags)

        # ------------------------------------------------------
        # 2) Map each terminal coord to nearest DOF in pressure/flow fields
        # ------------------------------------------------------
        def sample_field_at_points(func, pts):
            if len(pts) == 0:
                return np.zeros(0)

            dof_coords = func.function_space.tabulate_dof_coordinates()
            tree = cKDTree(dof_coords)
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            return func.x.array[nn]

        # inlet pressures/flows
        self.q_inlet_vals = sample_field_at_points(self.q_inlet_fun, self.x_inlet)

        # outlet pressures/flows
        self.q_outlet_vals = sample_field_at_points(self.q_outlet_fun, self.x_outlet)

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")

    ################################################################################
    # Visualization helper: write inlet/outlet points and DOF locations
    ################################################################################
    def _visualize_sources_and_sinks(self, W, q_in, q_out):
        mesh = self.mesh
        comm = mesh.comm

        # # --- Save inlet/outlet terminal coordinates as a point cloud ---
        # if comm.rank == 0:
        #     import dolfinx.io
        #     with dolfinx.io.XDMFFile(comm, "terminal_points.xdmf", "w") as f:
        #         # inlet points
        #         if self.x_inlet.shape[0] > 0:
        #             f.write_geometry(self.x_inlet)
        #             inlet_values = np.ones(self.x_inlet.shape[0], dtype=np.int32)
        #             f.write_data(inlet_values, "inlet_points")

        #         # outlet points
        #         if self.x_outlet.shape[0] > 0:
        #             f.write_geometry(self.x_outlet)
        #             outlet_values = 2*np.ones(self.x_outlet.shape[0], dtype=np.int32)
        #             f.write_data(outlet_values, "outlet_points")

        # --- Visualize q_in and q_out (DG0 source/sink fields) ---
        with io.XDMFFile(comm, "q_src_fields.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(q_in)


        with io.XDMFFile(comm, "q_snk_fields.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(q_out)


    ###########################################################################
    # Build q_in(x) and q_out(x) as Dirac-like distributions (Q/cell_volume)
    ###########################################################################
    def _build_q_fields(self, W):

        mesh = self.mesh
        q_in  = fem.Function(W)
        q_out = fem.Function(W)

        q_in.x.array[:]  = 0.0
        q_out.x.array[:] = 0.0

        # compute global average cell volume
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        total_vol = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx))
        cell_vol = total_vol / max(ncells, 1)

        # DOF coordinate tree
        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def add_dirac(q_field, xpts, flows, sign):
            if len(xpts) == 0:
                return
            _, dofs = tree.query(xpts)
            rho = sign * flows / cell_vol   # volumetric density = Q / V_cell
            for d, r in zip(dofs, rho):
                q_field.x.array[d] += r

        # inlet: + source
        add_dirac(q_in,  self.x_inlet,  self.q_inlet_vals,  +1.0)

        # outlet: +sink (we store magnitude only; sign applied in weak form)
        add_dirac(q_out, self.x_outlet, self.q_outlet_vals, -1.0)

        return q_in, q_out

    ###########################################################################
    # Solve advection-diffusion
    ###########################################################################
    def run(self):

        mesh = self.mesh
        dt = self.dt

        # === DG concentration space ===
        DG = element("DG", mesh.basix_cell(), 0)
        P1 = element("Lagrange", mesh.basix_cell(), 1)
        W  = fem.functionspace(mesh, DG)

        # === Build volumetric source/sink ===
        use_sources = True

        if use_sources:
            q_in, q_out = self._build_q_fields(W)
            # q_out.x.array[:] = 0 * q_out.x.array[:]  # scale sink for stability
            self._visualize_sources_and_sinks(W, q_in, q_out)
        else:
            q_in  = fem.Function(W); q_in.x.array[:]  = 0.0
            q_out = fem.Function(W); q_out.x.array[:] = 0.0
        # q_in, q_out = self._build_q_fields(W)

        if mesh.comm.rank == 0:
            print("q_in  min/max:",  q_in.x.array.min(),  q_in.x.array.max())
            print("q_out min/max:", q_out.x.array.min(), q_out.x.array.max())
            print("dt * q_in max:", self.dt * q_in.x.array.max())
            print("dt * q_out max:", self.dt * q_out.x.array.max())


        # === Constants ===
        c_in = fem.Constant(mesh, self.c_in_value)
        D    = fem.Constant(mesh, self.D_value)
        deltaT = fem.Constant(mesh, self.dt)

        # SUPG parameter tau
        h    = ufl.CellDiameter(mesh)
        u    = self.velocity
        umag = ufl.sqrt(ufl.dot(u, u) + 1e-10)

        Pe   = umag * h / (2 * D)
        beta = (ufl.cosh(Pe)/ufl.sinh(Pe)) - 1.0/Pe
        tau  = h * beta / (2*umag + 1e-10)

        # === Trial/test ===
        c  = ufl.TrialFunction(W)
        w  = ufl.TestFunction(W)
        c_out = fem.Function(dfx.fem.functionspace(mesh, P1))
        c_ = fem.Function(W)   # previous
        c_h = fem.Function(W)  # current
        c_.x.array[:] = 0.0

        # Measures
        n  = ufl.FacetNormal(mesh)
        dx = ufl.dx
        dS = ufl.dS

        # === Weak form components ===
        a_time    = c * w / deltaT * dx
        a_advect  = - dot(c * self.velocity, grad(w)) * dx
        a_diffuse = dot(D * grad(c), grad(w)) * dx

        # sink: +q_out * c   (LHS)
        a_sink = q_out * c * w * dx

        # full bilinear form
        a = a_time + a_advect + a_diffuse + a_sink

        # RHS: c_old/dt + q_in * c_in
        L = (c_ / deltaT + q_in * c_in) * w * dx

        # === SUPG stabilization ===
        v_supg = tau * dot(u, grad(w))
        a_supg = v_supg * (
                c / deltaT
            + dot(u, grad(c))
            - ufl.div(D * grad(c))
            + q_out * c
            ) * dx
        L_supg = v_supg * (
                c_ / deltaT
            + q_in * c_in
            ) * dx
        
        # a += a_supg
        # L += L_supg

        # === Upwind advection flux ===
        un = (dot(self.velocity, n) + abs(dot(self.velocity, n))) / 2.0
        a_up = dot(ufl.jump(w), un('+') * c('+') - un('-') * c('-')) * dS
        a += a_up

        # === SIPG diffusion ===
        h = ufl.CellDiameter(mesh)
        alpha = fem.Constant(mesh, 10.0)
        a += D('+') * alpha('+')/h('+') * dot(ufl.jump(w, n), ufl.jump(c, n)) * dS
        a -= D('+') * dot(ufl.avg(grad(w)), ufl.jump(c, n)) * dS
        a -= D('+') * dot(ufl.avg(grad(c)), ufl.jump(w, n)) * dS

        # assemble
        a_cpp = fem.form(a)
        L_cpp = fem.form(L)

        A = assemble_matrix(a_cpp)
        A.assemble()
        b = create_vector(L_cpp)

        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")

        # time stepping
        t = 0.0
        nt = int(self.T / self.dt)

        out = io.XDMFFile(mesh.comm, "transport_c.xdmf", "w")
        out.write_mesh(mesh)

        for step in range(nt):
            t += dt

            # Update inlet concentration for this time step
            c_in.value = self._inlet_concentration(t)

            # assemble RHS
            b.zeroEntries()
            assemble_vector(b, L_cpp)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                          mode=PETSc.ScatterMode.REVERSE)

            # solve
            solver.solve(b, c_h.x.petsc_vec)
            c_h.x.scatter_forward()

            # update old solution
            c_.x.array[:] = c_h.x.array

            # write output
            c_out.interpolate(c_h)
            out.write_function(c_out, t)

            if mesh.comm.rank == 0:
                print(f"Step {step+1}/{nt}   t={t:.3f}")

        out.close()


################################################################################
# Run
################################################################################

if __name__ == "__main__":
    solver = TransportSolver(
        bioreactor_domain="../geometry/bioreactor.xdmf",
        mesh_inlet_file="../geometry/tagged_branches_inlet.bp",
        mesh_outlet_file="../geometry/tagged_branches_outlet.bp",
        flow_inlet_file="../geometry/flow_checkpoint_inlet.bp",
        flow_outlet_file="../geometry/flow_checkpoint_outlet.bp",
        vel_file="./out_darcy/u_p0_000000.vtu",
        T=1000.0,
        dt=1.0,
        D_value=1e-5,
        c_in_value=1.0
    )
    solver.run()