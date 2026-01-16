import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
import json

from ufl               import dot, grad
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from scipy.spatial import cKDTree
import adios4dolfinx
from dolfinx.io import VTKFile

current_dir = Path("/Users/rakshakonanur/Documents/Research/vascularize/src/solves")

# -----------------------------------------------------
# Mesh importer
# -----------------------------------------------------
def import_3d_mesh(xdmf_file: str):
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        m = xdmf.read_mesh(name="Grid")
        m.topology.create_connectivity(m.topology.dim, m.topology.dim-1)
        tags = xdmf.read_meshtags(m, name="mesh_tags")
    return m, tags

def import_mesh(bp_file: str):
    mesh = adios4dolfinx.read_mesh(filename = Path(bp_file), comm=MPI.COMM_WORLD)
    tags = adios4dolfinx.read_meshtags(filename = Path(bp_file), mesh=mesh, meshtag_name="mesh_tags")
    return mesh, tags

# -----------------------------------------------------
# Pressure ADIOS importer
# -----------------------------------------------------
def import_pressure_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD,
                                       function_name="f")
    print("Timestamps found for pressure data:", ts)
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    p_src = fem.Function(P1)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, p_src, name="f", time=t)
        p_src.x.scatter_forward()
        p_src.x.array[:] *= 1333.22  # Convert to dyne/cm²
        out.append(p_src.copy())
    return out   # list of Functions

# -----------------------------------------------------
# Flow ADIOS importer
# -----------------------------------------------------
def import_flow_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD,
                                       function_name="f")
    print("Timestamps found for flow data:", ts)
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    q_src = fem.Function(P1)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, q_src,
                                    name="f", time=t)
        q_src.x.scatter_forward()
        out.append(q_src.copy())
    return out   # list of Functions


# ===================================================================
#                PERFUSION SOLVER WITH FULL COUPLING
# ===================================================================
class PerfusionSolver:
    def __init__(self, bioreactor_domain, mesh_inlet_file, mesh_outlet_file,
                 pres_inlet_file, pres_outlet_file, flow_inlet_file, flow_outlet_file):

        # 3D perfusion domain
        self.mesh, _ = import_3d_mesh(bioreactor_domain)

        # 1D tree terminal territory meshes
        self.inlet_mesh,  self.inlet_tags  = import_mesh(mesh_inlet_file)
        self.outlet_mesh, self.outlet_tags = import_mesh(mesh_outlet_file)

        # Load last time-slice of pressures/flows
        self.p_A_k = import_pressure_data(self.inlet_mesh,  pres_inlet_file)[-1]
        self.p_V_k = import_pressure_data(self.outlet_mesh, pres_outlet_file)[-1]
        self.w_A_k = import_flow_data(self.inlet_mesh,  flow_inlet_file)[-1]
        self.w_V_k = import_flow_data(self.outlet_mesh, flow_outlet_file)[-1]

        # Extract terminal (marker=2) coordinates + pressures + flows
        self._extract_terminal_data()


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
        self.p_inlet = sample_field_at_points(self.p_A_k, self.x_inlet)
        self.q_inlet = sample_field_at_points(self.w_A_k, self.x_inlet)

        # outlet pressures/flows
        self.p_outlet = sample_field_at_points(self.p_V_k, self.x_outlet)
        self.q_outlet = sample_field_at_points(self.w_V_k, self.x_outlet)

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")



    # ========================================================
    # Step B: Build q_src density approximation for Darcy
    # ========================================================
    def _build_q_src(self, W):
        mesh = self.mesh
        gdim = mesh.geometry.dim

        q_src = fem.Function(W)
        q_src.x.array[:] = 0.0

        # global average cell volume
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        vol_form = fem.form(one * ufl.dx)
        total_vol = fem.assemble_scalar(vol_form)
        cell_vol = total_vol / max(ncells, 1)

        # dof coords
        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def add_terminals(xpts, flows, sign):
            if len(xpts) == 0:
                return
            _, dofs = tree.query(xpts)
            rho = sign * flows / cell_vol

            for d, r in zip(dofs, rho):
                q_src.x.array[d] += r

        # arterial (source, +)
        add_terminals(self.x_inlet,  self.q_inlet,  +1.0)
        # venous   (sink, -)
        add_terminals(self.x_outlet, self.q_outlet, +1.0)

        return q_src


    # ========================================================
    # Step C: Dirichlet BCs at terminal pressures
    # ========================================================
    # def _build_terminal_bcs(self, W):
    #     mesh = self.mesh
    #     gdim = mesh.geometry.dim

    #     dof_coords = W.tabulate_dof_coordinates()
    #     tree = cKDTree(dof_coords)

    #     bcs = []

    #     def add_bc(xpts, pvals):
    #         if len(xpts) == 0:
    #             return
    #         _, dofs = tree.query(xpts)
    #         for d, pval in zip(dofs, pvals):
    #             bc_val = fem.Constant(mesh, dfx.default_scalar_type(pval))
    #             bc = fem.dirichletbc(bc_val,
    #                                  np.array([d], dtype=np.int32), W)
    #             bcs.append(bc)

    #     # add_bc(self.x_inlet,  self.p_inlet)
    #     add_bc(self.x_outlet, self.p_outlet)
    #     return bcs
    # def _build_terminal_bcs(self, W):
    #     """
    #     Build a *single* Dirichlet BC to fix the pressure gauge.

    #     We pick ONE outlet terminal, find its nearest pressure DOF,
    #     and pin p there to the corresponding 1D pressure value.

    #     All other terminals are left free (no Dirichlet).
    #     """
    #     mesh = self.mesh

    #     # DOF coordinates in the pressure space
    #     dof_coords = W.tabulate_dof_coordinates()
    #     tree = cKDTree(dof_coords)

    #     bcs = []

    #     # --- choose which terminal to use as gauge ---
    #     if len(self.x_outlet) > 0:
    #         # Use the first outlet terminal as the gauge point
    #         x_gauge = self.x_outlet[0:1]     # shape (1, 3)
    #         p_gauge = self.p_outlet[0]       # scalar value from 1D model

    #         # Find nearest DOF
    #         _, dofs = tree.query(x_gauge)    # dofs has shape (1,)
    #         d = int(dofs[0])

    #         # Build Dirichlet BC at that single DOF
    #         bc_val = fem.Constant(mesh, dfx.default_scalar_type(p_gauge))
    #         bc = fem.dirichletbc(
    #             bc_val,
    #             np.array([d], dtype=np.int32),
    #             W,
    #         )
    #         bcs.append(bc)

    #         if mesh.comm.rank == 0:
    #             print("Gauge BC: p =",
    #                 float(p_gauge),
    #                 "at DOF", d,
    #                 "near outlet point", x_gauge[0])
    #     else:
    #         # Fallback: no outlets → no BCs (you'd then need some other gauge)
    #         if mesh.comm.rank == 0:
    #             print("Warning: no outlet points found, no gauge BC imposed.")

    #     return bcs

    def _build_terminal_bcs(self, W):
        mesh = self.mesh
        # pin a single DOF to 0 as an internal gauge
        dof_coords = W.tabulate_dof_coordinates()
        gauge_dof = np.array([0], dtype=np.int32)  # first DOF, say
        bc_val = fem.Constant(mesh, dfx.default_scalar_type(0.0))
        bc = fem.dirichletbc(bc_val, gauge_dof, W)
        return [bc]
    
    def _apply_pressure_gauge_postsolve(self, p_h):
        """
        Shift p_h by a constant so that its outlet pressures match the 1D model
        in some sense (single point or average).

        This does NOT change u_h, since u_h depends on grad(p_h).
        """

        mesh = self.mesh
        Wp = p_h.function_space
        dof_coords = Wp.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        # Sample 3D pressure at outlet terminals
        if len(self.x_outlet) == 0:
            if mesh.comm.rank == 0:
                print("No outlet terminals, skipping post-gauge.")
            return

        _, nn = tree.query(self.x_outlet)
        nn = nn.astype(int)
        p_3d_out = p_h.x.array[nn]       # 3D Darcy pressures at outlet locations
        p_1d_out = self.p_outlet         # from your 1D model

        # Option A: match average outlet pressure
        offset = float(np.mean(p_1d_out - p_3d_out))

        # Option B (alternative): match one particular outlet, e.g. the first
        # offset = float(p_1d_out[0] - p_3d_out[0])

        # Apply shift
        p_h.x.array[:] += offset
        p_h.x.scatter_forward()

        if mesh.comm.rank == 0:
            print(f"Applied post-gauge offset Δp = {offset:.6g}")

    
    # ========================================================
    # Step C-alt: Penalty-like enforcement of outlet pressures
    # ========================================================
    def _build_outlet_penalty_terms(self, W, penalty_factor=1e6):
        """
        Build two P1 fields:
          - penalty(x): alpha(x) nonzero near outlet DOFs
          - rhs_penalty(x): alpha(x) * p_outlet

        so that we can add
            ∫ penalty * p * v dx    to the bilinear form
            ∫ rhs_penalty * v dx    to the RHS

        This approximates a Dirichlet condition at the outlet
        in a 'smeared' volumetric sense, similar to how q_src
        is imposed.
        """
        mesh = self.mesh

        penalty = fem.Function(W)
        rhs_penalty = fem.Function(W)

        penalty.x.array[:] = 0.0
        rhs_penalty.x.array[:] = 0.0

        # DOF coordinates / KD-tree on P1 space
        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        if len(self.x_outlet) > 0:
            _, dofs = tree.query(self.x_outlet)
            dofs = np.atleast_1d(dofs).astype(int)

            for d, pval in zip(dofs, self.p_outlet):
                penalty.x.array[d]     += penalty_factor
                rhs_penalty.x.array[d] += penalty_factor * pval

        return penalty, rhs_penalty
    
    # ========================================================
    # Step D: Compute interface BC to pass back to ROM solver
    # ========================================================

    def _compute_q_from_div_u(self, u_h):
        """
        Compute f_h ≈ div(u_h) (in weak/L2-projected sense) and then
        q = cell_volume * f_h at inlet/outlet terminal coordinates.
        """

        mesh = self.mesh

        # Scalar space for divergence (P1)
        Q_el = element("Lagrange", mesh.basix_cell(), 1)
        Q = fem.functionspace(mesh, Q_el)

        f = ufl.TrialFunction(Q)
        v = ufl.TestFunction(Q)

        # L2 projection of div(u): (f, v) = -(u, grad v)
        a = ufl.inner(f, v) * ufl.dx
        L = -ufl.inner(u_h, ufl.grad(v)) * ufl.dx

        problem = LinearProblem(
            a, L,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )
        f_h = problem.solve()   # f_h ~ div(u_h)

        # Global average cell volume (same as you used in _build_q_src)
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        total_vol = fem.assemble_scalar(fem.form(one * ufl.dx))
        cell_vol = total_vol / max(ncells, 1)

        # Sample f_h at terminal coordinates and convert to q = cell_vol * f
        dof_coords = Q.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def sample_q(pts):
            if len(pts) == 0:
                return np.zeros(0)
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            f_vals = f_h.x.array[nn]
            return cell_vol * f_vals, f_vals  # q, div(u)

        q_inlet,  divu_inlet  = sample_q(self.x_inlet)
        q_outlet, divu_outlet = sample_q(self.x_outlet)

        return f_h, q_inlet, q_outlet, divu_inlet, divu_outlet, cell_vol


    def _compute_interface_bc(self, p_h, u_h, q_src):
        mesh = self.mesh
        gdim = mesh.geometry.dim

        # ----- sample pressure as before -----
        Wp = p_h.function_space
        p_dof_coords = Wp.tabulate_dof_coordinates()
        tree_p = cKDTree(p_dof_coords)

        def sample_pressure(pts):
            if len(pts) == 0:
                return np.zeros(0)
            _, nn = tree_p.query(pts)
            nn = nn.astype(int)
            return p_h.x.array[nn]

        p_inlet  = sample_pressure(self.x_inlet)
        p_outlet = sample_pressure(self.x_outlet)

        p_inlet_mean  = float(np.mean(p_inlet))  if len(p_inlet)  > 0 else float("nan")
        p_outlet_mean = float(np.mean(p_outlet)) if len(p_outlet) > 0 else float("nan")

        # ----- sample velocity at terminals (optional but useful) -----
        V = u_h.function_space
        u_dof_coords = V.tabulate_dof_coordinates()
        tree_u = cKDTree(u_dof_coords)
        u_arr = u_h.x.array.reshape((-1, gdim))

        def sample_velocity(pts):
            if len(pts) == 0:
                return np.zeros((0, gdim))
            _, nn = tree_u.query(pts)
            nn = nn.astype(int)
            return u_arr[nn, :]

        u_inlet  = sample_velocity(self.x_inlet)
        u_outlet = sample_velocity(self.x_outlet)

        # ----- NEW: q = cell_volume * div(u) at interface points -----
        f_h, q_inlet, q_outlet, divu_inlet, divu_outlet, cell_vol = \
            self._compute_q_from_div_u(u_h)

        # Optional global check: ∫ f_h dx should ~ ∫ q_src dx
        f_form = fem.form(f_h * ufl.dx)
        Q_from_divu = float(fem.assemble_scalar(f_form))
        print("Total ∫ div(u) dx  =", Q_from_divu)
        qsrc_form = fem.form(q_src * ufl.dx)
        Q_from_qsrc = float(fem.assemble_scalar(qsrc_form))
        print("Total ∫ q_src dx   =", Q_from_qsrc)
        # (you can print these if you want to verify consistency)

        interface_bc = {
            # pressure
            "p_inlet_nodes":   p_inlet.tolist(),
            "p_outlet_nodes":  p_outlet.tolist(),
            "p_inlet_mean":    p_inlet_mean,
            "p_outlet_mean":   p_outlet_mean,

            # velocity at terminals
            "u_inlet_nodes":   u_inlet.tolist(),
            "u_outlet_nodes":  u_outlet.tolist(),

            # q = cell_vol * div(u) at terminals
            "q_inlet":         q_inlet.tolist(),
            "q_outlet":        q_outlet.tolist(),
            # raw div(u) values if you want them
            "divu_inlet":      divu_inlet.tolist(),
            "divu_outlet":     divu_outlet.tolist(),
            "cell_volume":     cell_vol,

            # coords for mapping back to 1D
            "coords_inlet":    self.x_inlet.tolist(),
            "coords_outlet":   self.x_outlet.tolist(),
        }

        return interface_bc

    # ========================================================
    # Solve Darcy
    # ========================================================
    def setup(self):

        mesh = self.mesh
        P_el = element("Lagrange", mesh.basix_cell(), 1)
        W = fem.functionspace(mesh, P_el)

        # Darcy coefficients
        kappa = 1e-7
        mu    = 1.0
        kappa_over_mu = fem.Constant(mesh, dfx.default_scalar_type(kappa/mu))
        phi = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        # weak form
        v = ufl.TestFunction(W)
        p = ufl.TrialFunction(W)

        a = kappa_over_mu * dot(grad(p), grad(v)) * ufl.dx

        # RHS from terminal flows
        q_src = self._build_q_src(W)
        if mesh.comm.rank == 0:
            print("q_src  min/max:",  q_src.x.array.min(),  q_src.x.array.max())

        L = q_src * v * ufl.dx

        # BCs from terminal pressures
        # bcs = self._build_terminal_bcs(W)
        bcs = []

        # solve
        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )

        #         # weak form
        # v = ufl.TestFunction(W)
        # p = ufl.TrialFunction(W)

        # dx = ufl.dx

        # # Darcy diffusion term
        # a = kappa_over_mu * dot(grad(p), grad(v)) * dx

        # # RHS from terminal flows (volumetric)
        # q_src = self._build_q_src(W)
        # if mesh.comm.rank == 0:
        #     print("q_src  min/max:",  q_src.x.array.min(),  q_src.x.array.max())

        # L = q_src * v * dx

        # # --- Penalty-like outlet pressure enforcement ---
        # penalty_factor = 1e-2  # tune this!
        # penalty, rhs_penalty = self._build_outlet_penalty_terms(W, penalty_factor)

        # # Add to forms:
        # #   ∫ penalty * p * v dx       (bilinear)
        # #   ∫ rhs_penalty * v dx       (linear)
        # a += penalty * p * v * dx
        # L += rhs_penalty * v * dx

        # # No strong Dirichlet BCs now
        # bcs = []

        # # solve
        # problem = LinearProblem(
        #     a, L, bcs=bcs,
        #     petsc_options={
        #         "ksp_type": "preonly",
        #         "pc_type": "lu",
        #         "pc_factor_mat_solver_type": "mumps"
        #     }
        # )

        p_h = problem.solve()

        # Post-process gauge using 1D outlet pressures
        self._apply_pressure_gauge_postsolve(p_h)

        # project velocity
        V = fem.functionspace(mesh,
                                element("DG", mesh.basix_cell(), 0,
                                        shape=(mesh.geometry.dim,)))
        projector = Projector(V)
        u_h = projector(-kappa_over_mu * grad(p_h) / phi)

        # Write outputs
        out_dir = current_dir/"out_darcy"
        out_dir.mkdir(exist_ok=True)

        with io.XDMFFile(mesh.comm, out_dir/"p.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_h)

        with io.XDMFFile(mesh.comm, out_dir/"u.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_h)

        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"u.vtu", "w")
        P1 = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1,
                                shape=(mesh.geometry.dim,)))
        u_P1= fem.Function(P1)
        u_P1.interpolate(u_h)
        vtkfile.write_function(u_P1)

        vtkfile_p = VTKFile(MPI.COMM_WORLD,  out_dir/"p.vtu", "w")
        vtkfile_p.write_function(p_h)

        # --- compute interface BCs for the 0D solver ---
        interface_bc = self._compute_interface_bc(p_h, u_h, q_src)

        # write to JSON for a separate script
        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                json.dump(interface_bc, fp, indent=2)

        print("Darcy perfusion solve complete.")

        # Return interface_bc so a caller in the same Python process can use it
        return interface_bc

# ===================================================================
# Projector class
# ===================================================================
class Projector:
    def __init__(self, V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(u,v)*ufl.dx
        self.V = V
        self.u = fem.Function(V)
        self.a_cpp = fem.form(a)

    def __call__(self, f):
        v = ufl.TestFunction(self.V)
        L = ufl.inner(f,v)*ufl.dx

        A = fem.petsc.assemble_matrix(self.a_cpp, bcs=[])
        A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")

        solver.solve(b, self.u.x.petsc_vec)

        return self.u


# ===================================================================
# RUN
# ===================================================================
if __name__ == "__main__":
    bioreactor_domain = "../geometry/bioreactor.xdmf"
    mesh_inlet_file   = "../geometry/tagged_branches_inlet.bp"
    mesh_outlet_file  = "../geometry/tagged_branches_outlet.bp"
    pres_inlet_file   = "../geometry/pressure_checkpoint_inlet.bp"
    pres_outlet_file  = "../geometry/pressure_checkpoint_outlet.bp"
    flow_inlet_file   = "../geometry/flow_checkpoint_inlet.bp"
    flow_outlet_file  = "../geometry/flow_checkpoint_outlet.bp"

    solver = PerfusionSolver(
        bioreactor_domain,
        mesh_inlet_file, mesh_outlet_file,
        pres_inlet_file, pres_outlet_file,
        flow_inlet_file, flow_outlet_file
    )
    solver.setup()
 