import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
import json

from ufl               import dot, grad
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
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
#                PERFUSION SOLVER WITH 2-COMPARTMENT DARCY
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

        # inlet pressures/flows from 1D solver
        self.p_inlet = sample_field_at_points(self.p_A_k, self.x_inlet)
        self.q_inlet = sample_field_at_points(self.w_A_k, self.x_inlet)

        # outlet pressures/flows from 1D solver
        self.p_outlet = sample_field_at_points(self.p_V_k, self.x_outlet)
        self.q_outlet = sample_field_at_points(self.w_V_k, self.x_outlet)

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")


    # ========================================================
    # Step B: Build arterial q_src density for compartment A
    # ========================================================
    def _build_q_src_A(self, W):
        """
        Build volumetric source f_A for arterial compartment, using 1D inlet
        flows mapped into the 3D domain, same approach as before.
        """
        mesh = self.mesh

        q_src = fem.Function(W)
        q_src.x.array[:] = 0.0

        # global average cell volume
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        vol_form = fem.form(one * ufl.dx)
        total_vol = fem.assemble_scalar(vol_form)
        cell_vol = total_vol / max(ncells, 1)

        # dof coords for W (P1)
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
        # optional: venous sinks into compartment A if you want them here
        # add_terminals(self.x_outlet, self.q_outlet, +1.0)

        return q_src

    def _build_q_snk_V(self, W):
        """
        Build volumetric source f_A for arterial compartment, using 1D inlet
        flows mapped into the 3D domain, same approach as before.
        """
        mesh = self.mesh

        q_src = fem.Function(W)
        q_src.x.array[:] = 0.0

        # global average cell volume
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        vol_form = fem.form(one * ufl.dx)
        total_vol = fem.assemble_scalar(vol_form)
        cell_vol = total_vol / max(ncells, 1)

        # dof coords for W (P1)
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
        # add_terminals(self.x_inlet,  self.q_inlet,  +1.0)
        # optional: venous sinks into compartment A if you want them here
        add_terminals(self.x_outlet, self.q_outlet, +1.0)

        return q_src


    # ========================================================
    # Step C: Dirichlet BCs on venous compartment p_V at outlets
    # ========================================================
    # def _build_terminal_bcs_mixed(self, M):
    #     """
    #     Build Dirichlet BCs for venous pressure p_V (subspace 1 of M)
    #     at outlet terminal coordinates, using 1D venous pressures.

    #     - Take subspace M.sub(1) (venous pressure space)
    #     - Collapse it to get an independent function space + mapping
    #     - Use KDTree on collapsed coordinates to find nearest DOFs
    #     - Map collapsed DOF indices back to the parent mixed space
    #     - Create Dirichlet BCs on the subspace
    #     """
    #     mesh = self.mesh

    #     # Subspace for venous pressure (p_V)
    #     V_sub = M.sub(1)

    #     # Collapse subspace → standalone space + mapping to parent DOFs
    #     V_sub_collapsed, dofs_sub_to_parent = V_sub.collapse()

    #     # Make mapping indexable by NumPy arrays
    #     dofs_sub_to_parent = np.asarray(dofs_sub_to_parent, dtype=np.int32)

    #     # Coordinates of DOFs in the collapsed space
    #     dof_coords = V_sub_collapsed.tabulate_dof_coordinates()
    #     # Ensure 2D shape (ndofs, gdim)
    #     dof_coords = dof_coords.reshape(-1, mesh.geometry.dim)

    #     tree = cKDTree(dof_coords)

    #     bcs = []

    #     def add_bc(xpts, pvals):
    #         xpts = xpts[0]
    #         if len(xpts) == 0:
    #             return

    #         # Nearest DOF indices in the *collapsed* space
    #         _, idx_collapsed = tree.query(xpts)
    #         idx_collapsed = np.atleast_1d(idx_collapsed).astype(np.int32)

    #         # Map collapsed indices → parent mixed-space DOFs
    #         parent_dofs = dofs_sub_to_parent[idx_collapsed]

    #         # One BC per terminal (each with its own constant value)
    #         for dof_parent, pval in zip(parent_dofs, pvals):
    #             bc_val = fem.Constant(mesh, dfx.default_scalar_type(pval))
    #             bc = fem.dirichletbc(
    #                 bc_val,
    #                 np.array([dof_parent], dtype=np.int32),
    #                 V_sub,  # subspace view into M
    #             )
    #             bcs.append(bc)

    #     # Impose venous pressures from 1D at the outlet terminals
    #     add_bc(self.x_outlet, self.p_outlet)

    #     return bcs

    # ========================================================
    # Step C: Gauge BC on venous compartment p_V (one DOF only)
    # ========================================================
    def _build_terminal_bcs_mixed(self, M):
        """
        Provide *only* a gauge BC for the venous pressure p_V (subspace 1 of M):
        we fix a single venous DOF to 0.0.

        The physical 1D venous pressures are enforced later by a post-solve
        constant shift, not as Dirichlet BCs.
        """
        mesh = self.mesh

        # Subspace for venous pressure (p_V)
        V_sub = M.sub(1)

        # Collapse subspace → standalone space + mapping to parent DOFs
        V_sub_collapsed, dofs_sub_to_parent = V_sub.collapse()
        dofs_sub_to_parent = np.asarray(dofs_sub_to_parent, dtype=np.int32)

        # Choose an arbitrary venous DOF as gauge (first DOF)
        # Index 0 in the collapsed space → some parent dof index
        gauge_parent_dof = np.array([dofs_sub_to_parent[0]], dtype=np.int32)

        # Fix that DOF to 0.0 (arbitrary gauge)
        bc_val = fem.Constant(mesh, dfx.default_scalar_type(0.0))
        bc = fem.dirichletbc(bc_val, gauge_parent_dof, V_sub)

        return [bc]

    def _apply_two_comp_pressure_gauge_postsolve(self, p_A_h, p_V_h):
        """
        Post-process gauge for the two-compartment pressure fields.

        We shift BOTH p_A_h and p_V_h by a constant Δp so that the venous
        pressure in the 3D domain (at outlet terminal locations) matches
        the 1D venous pressures self.p_outlet, in a least-squares sense.

        This does not change:
        - velocity fields (since they depend on grad p)
        - p_A - p_V differences (only the absolute level).
        """

        mesh = self.mesh
        gdim = mesh.geometry.dim

        if len(self.x_outlet) == 0:
            if mesh.comm.rank == 0:
                print("No outlet terminals; skipping two-compartment post-gauge.")
            return

        # --- 1) Standalone P1 space for sampling ---
        P1_el = element("Lagrange", mesh.basix_cell(), 1)
        WP = fem.functionspace(mesh, P1_el)

        projector_P1 = Projector(WP)

        # Project venous pressure subspace function to standalone P1
        pV_P1 = projector_P1(p_V_h)

        # --- 2) Sample 3D venous pressure at outlet terminal coordinates ---
        p_dof_coords = WP.tabulate_dof_coordinates().reshape(-1, gdim)
        tree_p = cKDTree(p_dof_coords)

        _, nn = tree_p.query(self.x_outlet)
        nn = nn.astype(int)

        pV_3d_out = pV_P1.x.array[nn]   # 3D venous pressures at outlets
        pV_1d_out = self.p_outlet       # 1D venous pressures at the same outlets

        # --- 3) Compute scalar offset Δp ---
        # Option A: match average venous outlet pressure
        # offset = float(np.mean(pV_1d_out - pV_3d_out))

        # (Alternative: match a single outlet:
        offset = float(pV_1d_out[0] - pV_3d_out[0])
        # )

        # --- 4) Shift BOTH compartments by Δp ---
        p_A_h.x.array[:] += offset
        # p_V_h.x.array[:] += offset

        print("Min/max p_A before scatter:", np.min(p_A_h.x.array), np.max(p_A_h.x.array))
        print("Min/max p_V before scatter:", np.min(p_V_h.x.array), np.max(p_V_h.x.array))

        p_A_h.x.scatter_forward()
        p_V_h.x.scatter_forward()

        if mesh.comm.rank == 0:
            print(f"Applied two-compartment post-gauge Δp = {offset:.6g}")


    # ========================================================
    # Step D: Compute div(u_A) and cell-averaged q using DG0
    # ========================================================
    def _compute_q_from_div_u(self, u_h):
        """
        Compute f_h ≈ div(u_h) in an L2-projected sense, then convert to
        cell-averaged q = cell_volume * f_h at inlet/outlet terminal coordinates.

        Implementation:
        1) L2-project div(u_h) into P1
        2) Interpolate that P1 field into DG0 (cell-wise constant)
        3) Sample DG0 field near inlet/outlet coords and scale by cell volume
        """

        mesh = self.mesh
        dx = ufl.dx  # uses mesh as default domain

        # 1) L2 projection of div(u_h) into P1
        Q1_el = element("Lagrange", mesh.basix_cell(), 1)
        Q1 = fem.functionspace(mesh, Q1_el)

        f1 = ufl.TrialFunction(Q1)
        v1 = ufl.TestFunction(Q1)

        # L2 projection: (f1, v1) = -(u_h, grad v1)
        a1 = ufl.inner(f1, v1) * dx
        L1 = -ufl.inner(u_h, ufl.grad(v1)) * dx

        problem1 = LinearProblem(
            a1, L1,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )
        f_h_P1 = problem1.solve()   # P1 approximation of div(u_h)

        # 2) Interpolate P1 field into DG0 (cell-wise constant)
        Q0_el = element("DG", mesh.basix_cell(), 0)
        Q0 = fem.functionspace(mesh, Q0_el)

        f_h = fem.Function(Q0)      # DG0 representation of div(u)
        f_h.interpolate(f_h_P1)     # sample P1 at DG0 dof locations (cell midpoints)

        # 3) Global average cell volume (same as in _build_q_src_A)
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        total_vol = fem.assemble_scalar(fem.form(one * dx))
        cell_vol = total_vol / max(ncells, 1)

        # 4) Sample DG0 div(u) at terminal coordinates and compute q
        dof_coords = Q0.tabulate_dof_coordinates()  # one point per cell
        tree = cKDTree(dof_coords)

        def sample_q(pts):
            if len(pts) == 0:
                return np.zeros(0), np.zeros(0)
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            f_vals = f_h.x.array[nn]     # div(u) in nearest cells
            q_vals = cell_vol * f_vals   # associated cell-averaged flux
            return q_vals, f_vals        # q, div(u)

        q_inlet,  divu_inlet  = sample_q(self.x_inlet)
        q_outlet, divu_outlet = sample_q(self.x_outlet)

        return f_h, q_inlet, q_outlet, divu_inlet, divu_outlet, cell_vol

    # ========================================================
    # Step D2: Venous Voronoi flux integration
    # ========================================================
    def _compute_venous_voronoi_flux(self, f_h, cell_vol):
        """
        Given f_h ≈ div(u_A) in DG0 and a global cell volume estimate,
        build a *venous-only* Voronoi tessellation over the cells:

        - Each DG0 dof (one per cell) is assigned to the nearest *venous*
          terminal coordinate in self.x_outlet.
        - For each venous terminal k, we integrate flux over its territory:

              Q_out_territory[k] = sum_{cells in territory k} f_h(cell) * cell_vol

        Returns
        -------
        Q_out_territory : (N_outlets,) array of territorial fluxes.
        """
        mesh = self.mesh
        comm = mesh.comm

        # No venous outlets → nothing to do
        if self.x_outlet.shape[0] == 0:
            return np.zeros(0, dtype=float)

        # DG0 space for f_h
        Q0 = f_h.function_space
        dof_coords = Q0.tabulate_dof_coordinates().reshape(-1, mesh.geometry.dim)

        # Build KDTree over *venous* terminal coordinates
        tree = cKDTree(self.x_outlet)

        # For each cell center, find nearest venous outlet index
        _, nearest_outlet_idx = tree.query(dof_coords)
        nearest_outlet_idx = nearest_outlet_idx.astype(int)

        f_vals = f_h.x.array  # one value per DG0 dof (per cell)
        n_out = self.x_outlet.shape[0]

        # Local accumulation
        Q_local = np.zeros(n_out, dtype=float)
        for cell_dof, k in enumerate(nearest_outlet_idx):
            Q_local[k] += f_vals[cell_dof] * cell_vol

        # MPI reduction to get global flux per outlet
        Q_global = np.zeros_like(Q_local)
        comm.Allreduce(Q_local, Q_global, op=MPI.SUM)

        return Q_global

    # ========================================================
    # Step E: Compute interface BC for 0D solver
    # ========================================================
    def _compute_interface_bc(self, p_A_h, p_V_h, u_A_h, u_V_h, q_src_A):
        """
        Compute interface data using:
        - p_A_h, p_V_h: arterial/venous pressures (on mixed subspaces)
        - u_A_h: arterial Darcy velocity (DG0)
        - u_V_h: venous  Darcy velocity (DG0)
        - q_src_A: arterial source term (P1)

        Pressures:
        - inlet  ← p_A
        - outlet ← p_V

        Velocities:
        - inlet  ← u_A_h
        - outlet ← u_V_h

        Fluxes q:
        - based on div(u_A_h) (as before)
        """
        mesh = self.mesh
        gdim = mesh.geometry.dim

        # ------------------------------------------------------
        # 0) Standalone P1 pressure spaces + projection
        # ------------------------------------------------------
        P1_el = element("Lagrange", mesh.basix_cell(), 1)
        WP = fem.functionspace(mesh, P1_el)

        projector_P1 = Projector(WP)

        # L2-project subspace functions p_A_h, p_V_h into WP
        pA = projector_P1(p_A_h)   # arterial pressure in standalone P1 space
        pV = projector_P1(p_V_h)   # venous  pressure in standalone P1 space

        # ------------------------------------------------------
        # 1) Sample pressures at inlet/outlet coordinates
        # ------------------------------------------------------
        p_dof_coords = WP.tabulate_dof_coordinates().reshape(-1, gdim)
        tree_p = cKDTree(p_dof_coords)

        def sample_pressure(func, pts):
            if len(pts) == 0:
                return np.zeros(0)
            _, nn = tree_p.query(pts)
            nn = nn.astype(int)
            return func.x.array[nn]

        # arterial side: inlet terminals
        p_inlet = sample_pressure(pA, self.x_inlet)
        # venous side: outlet terminals
        p_outlet = sample_pressure(pV, self.x_outlet)

        p_inlet_mean  = float(np.mean(p_inlet))  if len(p_inlet)  > 0 else float("nan")
        p_outlet_mean = float(np.mean(p_outlet)) if len(p_outlet) > 0 else float("nan")

        # ------------------------------------------------------
        # 2) Sample velocities at terminals
        #    (u_A_h and u_V_h live in the SAME DG0 vector space)
        # ------------------------------------------------------
        V = u_A_h.function_space
        u_dof_coords = V.tabulate_dof_coordinates().reshape(-1, gdim)
        tree_u = cKDTree(u_dof_coords)

        def sample_velocity(u_field, pts):
            if len(pts) == 0:
                return np.zeros((0, gdim))
            _, nn = tree_u.query(pts)
            nn = nn.astype(int)
            u_arr = u_field.x.array.reshape((-1, gdim))
            return u_arr[nn, :]

        # arterial velocity at inlet terminals
        u_inlet  = sample_velocity(u_A_h, self.x_inlet)
        # venous velocity at outlet terminals
        u_outlet = sample_velocity(u_V_h, self.x_outlet)

        # ------------------------------------------------------
        # 3) q = cell_volume * div(u_A) at interface points
        # ------------------------------------------------------
        f_h, q_inlet, _, divu_inlet, _, cell_vol = \
            self._compute_q_from_div_u(u_A_h)
        
        f_h, _, q_outlet, _, divu_outlet, cell_vol = \
            self._compute_q_from_div_u(u_V_h)


        # Optional global check: ∫ div(u_A) dx vs ∫ q_src_A dx
        f_form = fem.form(f_h * ufl.dx)
        Q_from_divu = float(fem.assemble_scalar(f_form))
        print("Total ∫ div(u_A) dx  =", Q_from_divu)
        qsrc_form = fem.form(q_src_A * ufl.dx)
        Q_from_qsrc = float(fem.assemble_scalar(qsrc_form))
        print("Total ∫ q_src_A dx   =", Q_from_qsrc)

        interface_bc = {
            # pressures
            "p_inlet_nodes":   p_inlet.tolist(),
            "p_outlet_nodes":  p_outlet.tolist(),
            "p_inlet_mean":    p_inlet_mean,
            "p_outlet_mean":   p_outlet_mean,

            # velocity at terminals
            "u_inlet_nodes":   u_inlet.tolist(),   # from u_A_h
            "u_outlet_nodes":  u_outlet.tolist(),  # from u_V_h

            # q = cell_vol * div(u_A) at terminals
            "q_inlet":         q_inlet.tolist(),
            "q_outlet":        q_outlet.tolist(),
            "divu_inlet":      divu_inlet.tolist(),
            "divu_outlet":     divu_outlet.tolist(),
            "cell_volume":     cell_vol,

            # coords for mapping back to 1D
            "coords_inlet":    self.x_inlet.tolist(),
            "coords_outlet":   self.x_outlet.tolist(),
        }

        return interface_bc



    # ========================================================
    # Solve 2-compartment Darcy
    # ========================================================
    def setup(self):

        mesh = self.mesh

        # Scalar P1 space (for sources, etc.)
        P_el = element("Lagrange", mesh.basix_cell(), 1)
        W = fem.functionspace(mesh, P_el)

        # Mixed P1 x P1 for p_A, p_V
        M_el = mixed_element([P_el, P_el])
        M = fem.functionspace(mesh, M_el)

        # Darcy coefficients
        kappa_A = 1e-7
        kappa_V = 1e-7
        mu      = 1.0
        kA_over_mu = fem.Constant(mesh, dfx.default_scalar_type(kappa_A / mu))
        kV_over_mu = fem.Constant(mesh, dfx.default_scalar_type(kappa_V / mu))

        # Inter-compartment conductance G
        G_val = 1e-6 # tune this as needed
        G = fem.Constant(mesh, dfx.default_scalar_type(G_val))

        # Trial and test functions
        (p_A, p_V) = ufl.TrialFunctions(M)
        (v_A, v_V) = ufl.TestFunctions(M)

        dx = ufl.dx

        # RHS: arterial source from q_src_A, venous source = 0
        q_src_A = self._build_q_src_A(W)
        q_snk_V = self._build_q_snk_V(W)
        if mesh.comm.rank == 0:
            print("f_A (q_src_A) min/max:", q_src_A.x.array.min(), q_src_A.x.array.max())

        f_A = q_src_A
        f_V = q_snk_V

        # Weak form:
        # A-compartment: -div(kA grad p_A) + G (p_A - p_V) = f_A
        # V-compartment: -div(kV grad p_V) - G (p_A - p_V) = f_V

        a = (
            kA_over_mu * dot(grad(p_A), grad(v_A)) * dx
            + G * (p_A - p_V) * v_A * dx
            + kV_over_mu * dot(grad(p_V), grad(v_V)) * dx
            - G * (p_A - p_V) * v_V * dx
        )


        L = f_A * v_A * dx + f_V * v_V * dx

        # BCs on venous pressure at outlets
        # bcs = self._build_terminal_bcs_mixed(M)
        bcs = []

        # Solve mixed system
        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
        )
        w_h = problem.solve()
        p_A_h, p_V_h = w_h.split()

        # --- NEW: apply post-solve gauge using 1D venous pressures ---
        self._apply_two_comp_pressure_gauge_postsolve(p_A_h, p_V_h)

        # project arterial velocity u_A = -kA/mu * grad(p_A)
        V_vec = fem.functionspace(
            mesh,
            element("DG", mesh.basix_cell(), 0,
                    shape=(mesh.geometry.dim,))
        )
        projector_A = Projector(V_vec)
        projector_V = Projector(V_vec)

        u_A_h = projector_A(-kA_over_mu * grad(p_A_h))
        u_V_h = projector_V(-kV_over_mu * grad(p_V_h))
        # Vector DG0 space for velocities
            
        # Write outputs
        out_dir = current_dir/"out_darcy_two_comp"
        out_dir.mkdir(exist_ok=True)

        with io.XDMFFile(mesh.comm, out_dir/"pA.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_A_h)

        with io.XDMFFile(mesh.comm, out_dir/"pV.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_V_h)

        with io.XDMFFile(mesh.comm, out_dir/"uA.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_A_h)

        with io.XDMFFile(mesh.comm, out_dir/"uV.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_V_h)

        # Also write u_A as P1 for VTK
        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"uA.vtu", "w")
        P1_vec = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1,
                                      shape=(mesh.geometry.dim,)))
        uA_P1 = fem.Function(P1_vec)
        uA_P1.interpolate(u_A_h)
        vtkfile.write_function(uA_P1)

        # Write u_V as P1 for VTK
        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"uV.vtu", "w")
        uV_P1 = fem.Function(P1_vec)
        uV_P1.interpolate(u_V_h)
        vtkfile.write_function(uV_P1)

        # Write pA as P1 for VTK
        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"pA.vtu", "w")
        vtkfile.write_function(p_A_h.collapse())

        # Write pV as P1 for VTK
        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"pV.vtu", "w")
        vtkfile.write_function(p_V_h.collapse())


        # --- compute interface BCs for the 0D solver ---
        interface_bc = self._compute_interface_bc(p_A_h, p_V_h, u_A_h, u_V_h, q_src_A)

        # write to JSON for a separate script
        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                json.dump(interface_bc, fp, indent=2)

        print("Two-compartment Darcy perfusion solve complete.")

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
