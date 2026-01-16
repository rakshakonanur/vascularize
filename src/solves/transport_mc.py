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
from dolfinx.fem.petsc import (
    assemble_matrix, create_vector, assemble_vector,
)
from dolfinx import fem, io
from basix.ufl import element, mixed_element
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
    P1 = element("Lagrange", mesh_obj.basix_cell(), 1)
    V = fem.functionspace(mesh_obj, P1)
    q_src = fem.Function(V)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, q_src, name="f", time=t)
        q_src.x.scatter_forward()
        out.append(q_src.copy())
    return out

################################################################################
# 3. Two-compartment Transport Solver
################################################################################

class TransportSolver:

    def __init__(self,
                 bioreactor_domain,
                 mesh_inlet_file, mesh_outlet_file,
                 flow_inlet_file, flow_outlet_file,
                 vel_file_A, vel_file_V,
                 pA_file, pV_file,
                 T=10.0, dt=1.0,
                 D_value=1e-3,
                 c_in_value=1.0,
                 G_value=1e-6):

        """
        Two-compartment advection-diffusion-reaction using J = G (p_A - p_V).

        c_A: arterial concentration
        c_V: venous  concentration

        (Backward Euler in time, symbolically):

        A-comp:
        (c_A^{n+1} - c_A^n)/dt + u_A·∇c_A^{n+1} - ∇·(D ∇c_A^{n+1})
            + J (c_A^{n+1} - c_V^{n+1}) = q_in c_in

        V-comp:
        (c_V^{n+1} - c_V^n)/dt + u_V·∇c_V^{n+1} - ∇·(D ∇c_V^{n+1})
            + J (c_V^{n+1} - c_A^{n+1}) + q_out c_V^{n+1} = 0

        where J(x) = G * (p_A(x) - p_V(x)), with p_A, p_V from the Darcy run.
        """

        # === 3D mesh ===
        self.mesh, _ = import_3d_mesh(bioreactor_domain)

        # === velocities (VTU; projected to P1 vector) ===
        self.uA = self._load_velocity(vel_file_A)
        self.uV = self._load_velocity(vel_file_V)

        # === pressures for J(x) ===
        self.pA = self._load_pressure(pA_file)
        self.pV = self._load_pressure(pV_file)
        self.G  = fem.Constant(self.mesh, dfx.default_scalar_type(G_value))

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
    # Load velocity (P1 vector Function from VTU)
    ###########################################################################
    def _load_velocity(self, vtu_file):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_file)
        reader.Update()

        mesh_vtk = reader.GetOutput()
        velocity  = mesh_vtk.GetPointData().GetArray("f")
        velocity_np = vtk_to_numpy(velocity)  # shape: (N, 3)
        velocity_np *= 1e3
        print(f"Velocity {vtu_file} shape:", velocity_np.shape)

        uP1 = element("Lagrange", self.mesh.basix_cell(), 1,
                      shape=(self.mesh.geometry.dim,))
        P1 = fem.functionspace(self.mesh, uP1)

        u = fem.Function(P1)
        u.x.array[:] = velocity_np.flatten()
        u.x.scatter_forward()
        return u

    ###########################################################################
    # Load pressure (P1 scalar Function from VTU)
    ###########################################################################
    def _load_pressure(self, vtu_file):
        
        mesh = self.mesh

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_file)
        reader.Update()

        mesh_vtk = reader.GetOutput()
        pressure = mesh_vtk.GetPointData().GetArray("f")
        print(pressure)

        P1_el = element("Lagrange", mesh.basix_cell(), 1)
        P1 = fem.functionspace(mesh, P1_el)
        p = fem.Function(P1)
        p.x.array[:] = vtk_to_numpy(pressure)
        p.x.scatter_forward()
        return p

    ###########################################################################
    # Extract terminal coordinates + flows (same as Darcy)
    ###########################################################################
    def _extract_terminal_data(self):
        """
        Extract terminal coordinates (marker==2) directly from inlet/outlet meshtags.
        Then sample the nearest flow values.
        """

        def extract_terminal_coords(mesh_obj, tags):
            marker = 2
            idx = np.where(tags.values == marker)[0]
            if len(idx) == 0:
                return np.zeros((0, mesh_obj.geometry.dim))

            entity_ids = tags.indices[idx]
            return mesh_obj.geometry.x[entity_ids]

        # inlet terminals
        self.x_inlet = extract_terminal_coords(self.inlet_mesh, self.inlet_tags)
        # outlet terminals
        self.x_outlet = extract_terminal_coords(self.outlet_mesh, self.outlet_tags)

        def sample_field_at_points(func, pts):
            if len(pts) == 0:
                return np.zeros(0)

            dof_coords = func.function_space.tabulate_dof_coordinates()
            tree = cKDTree(dof_coords)
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            return func.x.array[nn]

        # inlet flows
        self.q_inlet_vals = sample_field_at_points(self.q_inlet_fun, self.x_inlet)
        # outlet flows
        self.q_outlet_vals = sample_field_at_points(self.q_outlet_fun, self.x_outlet)

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")

    ################################################################################
    # Visualization helper
    ################################################################################
    def _visualize_sources_and_sinks(self, W_scalar, q_in, q_out):
        mesh = self.mesh
        comm = mesh.comm

        with io.XDMFFile(comm, "q_src_fields.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(q_in)

        with io.XDMFFile(comm, "q_snk_fields.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(q_out)

    ###########################################################################
    # Build q_in(x) and q_out(x) as Dirac-like distributions (Q/cell_volume)
    ###########################################################################
    def _build_q_fields(self, W_scalar):

        mesh = self.mesh
        q_in  = fem.Function(W_scalar)
        q_out = fem.Function(W_scalar)

        q_in.x.array[:]  = 0.0
        q_out.x.array[:] = 0.0

        # compute global average cell volume
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        total_vol = fem.assemble_scalar(
            fem.form(fem.Constant(mesh, 1.0) * ufl.dx)
        )
        cell_vol = total_vol / max(ncells, 1)

        # DOF coordinate tree
        dof_coords = W_scalar.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def add_dirac(q_field, xpts, flows, sign):
            if len(xpts) == 0:
                return
            _, dofs = tree.query(xpts)
            rho = sign * flows / cell_vol   # volumetric density = Q / V_cell
            for d, r in zip(dofs, rho):
                q_field.x.array[d] += r

        # inlet: + source into arterial compartment
        add_dirac(q_in,  self.x_inlet,  self.q_inlet_vals,  +1.0)

        # outlet: sink from venous compartment (store sign directly)
        add_dirac(q_out, self.x_outlet, self.q_outlet_vals, -1.0)

        return q_in, q_out

     ###########################################################################
    # OPTIONAL: time-dependent inlet concentration profile
    ###########################################################################
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
    # Solve 2-compartment advection-diffusion-reaction
    ###########################################################################
    def run(self):

        mesh = self.mesh
        dt = self.dt

        # === Scalar DG0 and P1 spaces ===
        DG_el = element("DG", mesh.basix_cell(), 0)
        P1_el = element("Lagrange", mesh.basix_cell(), 1)

        W_scalar = fem.functionspace(mesh, DG_el)   # for q_in, q_out, etc.
        P1       = fem.functionspace(mesh, P1_el)

        # === Mixed DG0 x DG0 for [c_A, c_V] ===
        M_el = mixed_element([DG_el, DG_el])
        M    = fem.functionspace(mesh, M_el)

        # === Build volumetric source/sink from 1D flows ===
        use_sources = True
        if use_sources:
            q_in, q_out = self._build_q_fields(W_scalar)
            self._visualize_sources_and_sinks(W_scalar, q_in, q_out)
        else:
            q_in  = fem.Function(W_scalar); q_in.x.array[:]  = 0.0
            q_out = fem.Function(W_scalar); q_out.x.array[:] = 0.0

        if mesh.comm.rank == 0:
            print("q_in  min/max:",  q_in.x.array.min(),  q_in.x.array.max())
            print("q_out min/max:", q_out.x.array.min(), q_out.x.array.max())
            print("dt * q_in max:", self.dt * q_in.x.array.max())
            print("dt * q_out max:", self.dt * q_out.x.array.max())

        # === Constants ===
        c_in  = fem.Constant(mesh, self.c_in_value)
        D     = fem.Constant(mesh, self.D_value)
        deltaT = fem.Constant(mesh, self.dt)

        # === Velocity fields ===
        uA = self.uA
        uV = self.uV

        # === J(x) = G (p_A - p_V) ===
        J_expr = self.G * (self.pA - self.pV)

        # === Trial/test for mixed system ===
        (cA, cV) = ufl.TrialFunctions(M)
        (wA, wV) = ufl.TestFunctions(M)

        # current and previous solution
        c_ = fem.Function(M)   # previous [c_A^n, c_V^n]
        c_h = fem.Function(M)  # current  [c_A^{n+1}, c_V^{n+1}]
        c_.x.array[:] = 0.0

        cA_old, cV_old = ufl.split(c_)

        # Measures
        n  = ufl.FacetNormal(mesh)
        dx = ufl.dx
        dS = ufl.dS

        # === Time term ===
        a_time = (
            cA * wA + cV * wV
        ) * (1.0 / deltaT) * dx

        # === Advection term with u_A, u_V ===
        a_adv = (
            - dot(cA * uA, grad(wA))
            - dot(cV * uV, grad(wV))
        ) * dx

        # === Diffusion term (same D for both) ===
        a_diff = D * (
            dot(grad(cA), grad(wA))
            + dot(grad(cV), grad(wV))
        ) * dx

        # === Inter-compartment exchange via J(x) ===
        # Mass-conserving: +J(cA - cV) in A, +J(cV - cA) in V
        a_ex = J_expr * (
            (cA - cV) * wA + (cV - cA) * wV
        ) * dx

        # === Venous sink term: q_out * c_V ===
        a_sink_V = q_out * cV * wV * dx

        # --- Combine bilinear form ---
        a = a_time + a_adv + a_diff + a_ex + a_sink_V

        # === RHS: previous solution + arterial source q_in * c_in ===
        L = (
            (cA_old * wA + cV_old * wV) * (1.0 / deltaT)
            + q_in * c_in * wA
        ) * dx

        # === Upwind advection flux for both compartments ===
        unA = (dot(uA, n) + abs(dot(uA, n))) / 2.0
        unV = (dot(uV, n) + abs(dot(uV, n))) / 2.0
        a_up = (
            dot(ufl.jump(wA), unA('+') * cA('+') - unA('-') * cA('-'))
            + dot(ufl.jump(wV), unV('+') * cV('+') - unV('-') * cV('-'))
        ) * dS
        a += a_up

        # === SIPG diffusion for both compartments ===
        h = ufl.CellDiameter(mesh)
        alpha = fem.Constant(mesh, 10.0)
        a += D('+') * alpha('+')/h('+') * (
            dot(ufl.jump(wA, n), ufl.jump(cA, n))
            + dot(ufl.jump(wV, n), ufl.jump(cV, n))
        ) * dS
        a -= D('+') * (
            dot(ufl.avg(grad(wA)), ufl.jump(cA, n))
            + dot(ufl.avg(grad(wV)), ufl.jump(cV, n))
        ) * dS
        a -= D('+') * (
            dot(ufl.avg(grad(cA)), ufl.jump(wA, n))
            + dot(ufl.avg(grad(cV)), ufl.jump(wV, n))
        ) * dS

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

        out_A = io.XDMFFile(mesh.comm, "transport_c_two_comp_a.xdmf", "w")
        out_A.write_mesh(mesh)

        out_V = io.XDMFFile(mesh.comm, "transport_c_two_comp_v.xdmf", "w")
        out_V.write_mesh(mesh)

        # For output as P1 (for nicer visualization)
        cA_out_space = P1
        cV_out_space = P1
        cA_out = fem.Function(cA_out_space)
        cV_out = fem.Function(cV_out_space)

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

            # split for output
            cA_sol, cV_sol = c_h.split()

            cA_out.interpolate(cA_sol)
            cV_out.interpolate(cV_sol)

            out_A.write_function(cA_out, t)
            out_V.write_function(cV_out, t)

            if mesh.comm.rank == 0:
                print(f"Step {step+1}/{nt}   t={t:.3f}")

        out_A.close()
        out_V.close()


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
        vel_file_A="./out_darcy_two_comp/uA_p0_000000.vtu",
        vel_file_V="./out_darcy_two_comp/uV_p0_000000.vtu",
        pA_file="./out_darcy_two_comp/pA_p0_000000.vtu",
        pV_file="./out_darcy_two_comp/pV_p0_000000.vtu",
        T=1000.0,
        dt=1.0,
        D_value=1e-5,
        c_in_value=1.0,
        G_value=1e-5,   # set to same G used in Darcy 2-compartment solve
    )
    solver.run()
