import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl               import avg, jump, dot, grad
from sys               import argv
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx import (fem, io, mesh)
from dolfinx.fem.petsc import assemble_matrix, create_vector, assemble_vector, set_bc, apply_lifting, LinearProblem, set_bc
from ufl import (FacetNormal, Identity, Measure, TestFunctions, TrialFunctions, exp, div, inner, SpatialCoordinate,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
from typing import List, Optional
from dolfinx.io import XDMFFile, VTKFile
import adios4dolfinx

current_dir = Path("/Users/rakshakonanur/Documents/Research/vascularize/src/solves")
from send2trash import send2trash

WALL = 0
OUTLET = 1
OUTFLOW = 2

# Set compiler options for runtime optimization
# Using same optimization options as hherlyng/DG_advection_diffusion.py
cache_dir = f"{str(Path.cwd())}/.cache"
compile_options = ["-Ofast", "-march=native"]
jit_parameters  = {"cffi_extra_compile_args" : compile_options,
                    "cache_dir"               : cache_dir,
                    "cffi_libraries"          : ["m"]}

def import_mesh(xdmf_file: str):
    """
    Import a mesh from an XDMF file.
    """
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        mesh_tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    return mesh, mesh_tags

def import_pressure_data(self, pres_file: str, k: int = 1):
    """
    Import pressure data from an XDMF file.
    """
    # mesh = adios4dolfinx.read_mesh(filename = pres_file, comm=MPI.COMM_WORLD)
    ts = adios4dolfinx.read_timestamps(filename = pres_file, comm=MPI.COMM_WORLD, function_name="p_src")
    DG0 = fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
    p_src = fem.Function(DG0)
    p_src_P1 = fem.Function(fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1)), name="p_src_P1")

    p_series = []
    for t in ts:
        adios4dolfinx.read_function(pres_file, p_src, name="p_src", time=t)
        p_src_P1.interpolate(p_src)  # Interpolate to P1 space
        p_series.append(p_src_P1.copy())

    with io.XDMFFile(MPI.COMM_WORLD, current_dir/"out_darcy/import_p_src.xdmf", "w") as file:
        file.write_mesh(self.mesh)
        file.write_function(p_series[-1])

    Pk = fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), k))
    for i in range(len(p_series)):
        p_tmp = fem.Function(Pk)
        p_tmp.interpolate(p_series[i])
        p_series[i] = p_tmp

    print(f"Imported {len(p_series)} time steps from {pres_file}", flush=True)
    return p_series

def import_flow_data(self, flow_file: str, cell_tags: np.ndarray):
    """
    Import flow data from an XDMF file.
    """
    comm = MPI.COMM_WORLD

    # --- read timesteps and last field ---
    ts = adios4dolfinx.read_timestamps(filename=flow_file, comm=comm, function_name="q_src_density")
    V = fem.functionspace(self.mesh, ("DG", 0))
    q_src = fem.Function(V)

    for t in ts:
        adios4dolfinx.read_function(flow_file, q_src, name="q_src_density", time=t)
        q_src.x.scatter_forward()
    # q_src now holds the last time step

    # optional: write for inspection
    out_dir = Path("out_darcy"); out_dir.mkdir(parents=True, exist_ok=True)
    with io.XDMFFile(comm, out_dir / "import_q_src.xdmf", "w") as xdmf:
        xdmf.write_mesh(self.mesh)
        xdmf.write_function(q_src)

    # --- compute per-tag value (average over the tag) ---
    # Since q_src is DG0 and constant per tag (by your setup), this equals that constant.
    dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags)

    # gather unique tags across ranks
    local_tags = np.unique(cell_tags.values)
    all_tags = np.unique(np.concatenate(comm.allgather(local_tags)))

    per_tag_value = {}
    for tag in all_tags:
        # numerator: ∫_tag q_src dx
        num_local = fem.assemble_scalar(fem.form(q_src * dx(tag)))
        # denominator: ∫_tag 1 dx  (the measure of the tag)
        den_local = fem.assemble_scalar(fem.form(1.0 * dx(tag)))

        num = comm.allreduce(num_local, op=MPI.SUM)
        den = comm.allreduce(den_local, op=MPI.SUM)

        if den > 0.0:
            per_tag_value[int(tag)] = float(num / den)  # constant per tag
        else:
            # tag absent on the global mesh; skip
            continue

    if comm.rank == 0:
        print(f"Read {len(ts)} time steps from {flow_file}")
        print("Per-tag values:", per_tag_value)

    # Return ONLY the per-tag constants (no sum)
    return float(sum(per_tag_value.values()))


def _cell_volumes_Q0(mesh) -> np.ndarray:
    """
    Return a NumPy array of cell volumes (length = # locally-owned cells).
    Uses the DG0 mass-vector trick: assemble v*dx, where v is the DG0 test function.
    """
    Q0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    v = ufl.TestFunction(Q0)
    b = fem.petsc.assemble_vector(fem.form(v * ufl.dx(domain=mesh)))
    # Fill ghosts -> owners
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    vol_local = b.getArray(readonly=True).copy()
    return vol_local

def _labels_from_cell_tags(mesh, cell_tags) -> np.ndarray:
    """Array of length #local cells with the territory tag per cell (0 if untagged)."""
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    labels = np.zeros(n_local, dtype=np.int32)
    labels[cell_tags.indices] = cell_tags.values.astype(np.int32)
    return labels

def compute_and_write_inlet_mbf(self, cell_tags, p_DG0, beta, xdmf_out="out_darcy/mbf.xdmf", bp_out="out_darcy/mbf.bp",
                          field_names=("mbf_qTi_tagconst","mbf_density")):
    """
    Array-based MBF:
        q^{T,i} = sum_{cells in tag i} [ beta * (p_src - p) * vol(cell) ].
    Writes two DG0 fields:
      - mbf_density   = beta * (p_src - p)          (local, per cell)
      - mbf_qTi_tagconst = integrated q^{T,i} painted constant over each tag
    """
    mesh = self.mesh
    comm = mesh.comm
    rank = comm.rank

    # --- Ensure p and p_src are DG0 on the SAME space instance ---
    Q0 = p_DG0.function_space

    # 1) p_src: last one you stored (assumed DG0). Move it into Q0 if needed.
    p_src_fn = self.p_src[-1]
    p_src_Q0 = p_src_fn
    if p_src_fn.function_space != Q0:
        p_src_Q0 = fem.Function(Q0, name="p_src_Q0")
        p_src_Q0.interpolate(p_src_fn)
    else:
        p_src_Q0 = p_src_fn

    # 2) beta: accept Function(Q0), fem.Constant, or scalar
    if isinstance(beta, fem.Function):
        if beta.function_space != Q0:
            beta_Q0 = fem.Function(Q0, name="beta_Q0")
            beta_Q0.interpolate(beta)
        else:
            beta_Q0 = beta
        beta_arr = beta_Q0.x.array
    elif isinstance(beta, fem.Constant):
        beta_arr = np.full_like(p_DG0.x.array, float(beta.value), dtype=float)
    else:
        # assume scalar
        beta_arr = np.full_like(p_DG0.x.array, float(beta), dtype=float)

    # --- Raw arrays (owned+ghost) ---
    p_arr    = p_DG0.x.array
    psrc_arr = p_src_Q0.x.array

    # --- Local labels and cell volumes ---
    labels = _labels_from_cell_tags(mesh, cell_tags)  # <-- make sure this is CELL tags
    vols   = _cell_volumes_Q0(mesh)

    # --- Density and per-cell flux (all NumPy) ---
    density   = beta_arr * (psrc_arr - p_arr)     # units: (1/s)*Pa -> whatever model uses
    cell_flux = density * vols                    # integrated flow per cell

    # --- Sum by tag (local) ---
    uniq_tags_local = np.unique(cell_tags.values.astype(int))
    q_local = {int(tag): float(cell_flux[labels == int(tag)].sum()) for tag in uniq_tags_local}

    # --- Reduce to global sums ---
    q_global = {int(tag): comm.allreduce(q_local[int(tag)], op=MPI.SUM) for tag in uniq_tags_local}

    # --- Build output DG0 fields ---
    mbf_density = fem.Function(Q0, name=field_names[1])
    mbf_density.x.array[:] = density
    mbf_density.x.scatter_forward()

    mbf_qTi = fem.Function(Q0, name=field_names[0])
    mbf_qTi.x.array[:] = 0.0
    for tag in np.unique(cell_tags.values).astype(int):
        mask = (cell_tags.values == int(tag))
        local_cells = cell_tags.indices[mask]
        mbf_qTi.x.array[local_cells] = q_global.get(int(tag), 0.0)
    mbf_qTi.x.scatter_forward()

    mbf_qTi_per_vol = fem.Function(Q0, name=field_names[0]+"_per_vol")
    mbf_qTi_per_vol.x.array[:] = 0.0
    for tag in np.unique(cell_tags.values).astype(int):
        mask = (cell_tags.values == int(tag))
        local_cells = cell_tags.indices[mask]
        mbf_qTi_per_vol.x.array[local_cells] = q_global.get(int(tag), 0.0) / vols[local_cells].sum()
    mbf_qTi_per_vol.x.scatter_forward()

    # --- Write XDMF (mesh, tags, fields) ---
    with io.XDMFFile(comm, xdmf_out, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(mbf_qTi)
        xdmf.write_function(mbf_density)
        xdmf.write_function(mbf_qTi_per_vol)

    # Write using adios4dolfinx as well
    # adios4dolfinx.write_mesh(current_dir/"out_darcy/mbf.bp", mesh, engine="BP4")
    if Path(current_dir/bp_out).exists():
        send2trash(current_dir/bp_out)

    adios4dolfinx.write_function_on_input_mesh(current_dir/bp_out, mbf_qTi, time=0.0, name=field_names[0])
    adios4dolfinx.write_function_on_input_mesh(current_dir/bp_out, mbf_density, time=0.0, name=field_names[1])

    if rank == 0:
        print(f"[MBF] Wrote {xdmf_out} with fields: {field_names[0]}, {field_names[1]}")

    return q_global

def compute_and_write_outlet_mbf(self, cell_tags, p_DG0, beta, xdmf_out="out_darcy/mbf.xdmf", bp_out="out_darcy/mbf.bp",
                          field_names=("mbf_qTi_tagconst","mbf_density")):
    """
    Array-based MBF:
        q^{T,i} = sum_{cells in tag i} [ beta * (p_snk - p) * vol(cell) ].
    Writes two DG0 fields:
      - mbf_density   = beta * (p_snk - p)          (local, per cell)
      - mbf_qTi_tagconst = integrated q^{T,i} painted constant over each tag
    """
    mesh = self.mesh
    comm = mesh.comm
    rank = comm.rank

    # --- Ensure p and p_src are DG0 on the SAME space instance ---
    Q0 = p_DG0.function_space

    # 1) p_src: last one you stored (assumed DG0). Move it into Q0 if needed.
    p_src_fn = self.p_snk[-1]
    p_src_Q0 = p_src_fn
    if p_src_fn.function_space != Q0:
        p_src_Q0 = fem.Function(Q0, name="p_src_Q0")
        p_src_Q0.interpolate(p_src_fn)
    else:
        p_src_Q0 = p_src_fn

    # 2) beta: accept Function(Q0), fem.Constant, or scalar
    if isinstance(beta, fem.Function):
        if beta.function_space != Q0:
            beta_Q0 = fem.Function(Q0, name="beta_Q0")
            beta_Q0.interpolate(beta)
        else:
            beta_Q0 = beta
        beta_arr = beta_Q0.x.array
    elif isinstance(beta, fem.Constant):
        beta_arr = np.full_like(p_DG0.x.array, float(beta.value), dtype=float)
    else:
        # assume scalar
        beta_arr = np.full_like(p_DG0.x.array, float(beta), dtype=float)

    # --- Raw arrays (owned+ghost) ---
    p_arr    = p_DG0.x.array
    psrc_arr = p_src_Q0.x.array

    # --- Local labels and cell volumes ---
    labels = _labels_from_cell_tags(mesh, cell_tags)  # <-- make sure this is CELL tags
    vols   = _cell_volumes_Q0(mesh)

    # --- Density and per-cell flux (all NumPy) ---
    density   = beta_arr * (psrc_arr - p_arr)     # units: (1/s)*Pa -> whatever model uses
    cell_flux = density * vols                    # integrated flow per cell

    # --- Sum by tag (local) ---
    uniq_tags_local = np.unique(cell_tags.values.astype(int))
    q_local = {int(tag): float(cell_flux[labels == int(tag)].sum()) for tag in uniq_tags_local}

    # --- Reduce to global sums ---
    q_global = {int(tag): comm.allreduce(q_local[int(tag)], op=MPI.SUM) for tag in uniq_tags_local}

    # --- Build output DG0 fields ---
    mbf_density = fem.Function(Q0, name=field_names[1])
    mbf_density.x.array[:] = density
    mbf_density.x.scatter_forward()

    mbf_qTi = fem.Function(Q0, name=field_names[0])
    mbf_qTi.x.array[:] = 0.0
    for tag in np.unique(cell_tags.values).astype(int):
        mask = (cell_tags.values == int(tag))
        local_cells = cell_tags.indices[mask]
        mbf_qTi.x.array[local_cells] = q_global.get(int(tag), 0.0)
    mbf_qTi.x.scatter_forward()

    mbf_qTi_per_vol = fem.Function(Q0, name=field_names[0]+"_per_vol")
    mbf_qTi_per_vol.x.array[:] = 0.0
    for tag in np.unique(cell_tags.values).astype(int):
        mask = (cell_tags.values == int(tag))
        local_cells = cell_tags.indices[mask]
        mbf_qTi_per_vol.x.array[local_cells] = q_global.get(int(tag), 0.0) / vols[local_cells].sum()
    mbf_qTi_per_vol.x.scatter_forward()

    # --- Write XDMF (mesh, tags, fields) ---
    with io.XDMFFile(comm, xdmf_out, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(mbf_qTi)
        xdmf.write_function(mbf_density)
        xdmf.write_function(mbf_qTi_per_vol)

    # Write using adios4dolfinx as well
    # adios4dolfinx.write_mesh(current_dir/"out_darcy/mbf.bp", mesh, engine="BP4")
    if Path(current_dir/bp_out).exists():
        send2trash(current_dir/bp_out)

    adios4dolfinx.write_function_on_input_mesh(current_dir/bp_out, mbf_qTi, time=0.0, name=field_names[0])
    adios4dolfinx.write_function_on_input_mesh(current_dir/bp_out, mbf_density, time=0.0, name=field_names[1])

    if rank == 0:
        print(f"[MBF] Wrote {xdmf_out} with fields: {field_names[0]}, {field_names[1]}")

    return q_global


def single_compartment(self, source_pressures, sink_pressures):
    Q_bio = self.flow # given from Qterm in 1D sim, convert to cm^3/s

    # Calculate volume of the bioreactor
    DG = element("DG", self.mesh.basix_cell(), 0)
    v_ = dfx.fem.functionspace(self.mesh, DG)  
    vol = ufl.TestFunction(v_)
    volume_form = fem.form(vol * dx)  # Volume integral form
    V_bio = dfx.fem.assemble_scalar(volume_form)  # Assemble integral of v
    print(f"Mesh volume: {V_bio}", flush=True) # Volume of the bioreactor
    metabolic_demand = Q_bio/V_bio  # metabolic demand, cm^3/s/g (set to 0.5/60 for 0.5 ml/min/g)

    Pcap = 15*1333.22  # Capillary pressure, converted to dyne/cm^2
    Psnk = sink_pressures # Sink pressure

    Psrc_values = source_pressures[-1].x.array  # Get the values of the outlet BCs
    Psnk_values = sink_pressures[-1].x.array
    # Psnk_values = 0.0 * Psnk_values  # Set sink pressures to zero for reference
    # print(f"Outlet pressures: {Psrc_values}", flush=True)
    Psrc_avg = np.mean(Psrc_values)  # Average pressure at the outlets
    Psnk_avg = np.mean(Psnk_values)
    print(f"Average pressure at outlets: {Psrc_avg}", flush=True)

    vols   = _cell_volumes_Q0(self.mesh)
    uniq_tags_local = np.unique(self.inlet_cell_tags.values.astype(int))
    territory_vols = np.zeros(uniq_tags_local.max()+1)  # assuming tags are 0-indexed and contiguous
    DG = dfx.fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
    beta_src = fem.Function(DG)
    beta_snk = fem.Function(DG)

    for tag in np.unique(self.inlet_cell_tags.values).astype(int):
        mask = (self.inlet_cell_tags.values == int(tag))
        local_cells = self.inlet_cell_tags.indices[mask]
        territory_vols[tag] = vols[local_cells].sum()
        # beta_src.x.array[local_cells] = Q_bio / territory_vols[tag] / (uniq_tags_local.max() + 1) * (1/(Psrc_avg - Pcap)) # set metabolic demand per territory
        beta_src.x.array[local_cells] = Q_bio / V_bio * (1/(Psrc_avg - Pcap)) # set metabolic demand per territory

    for tag in np.unique(self.outlet_cell_tags.values).astype(int):
        mask = (self.outlet_cell_tags.values == int(tag))
        local_cells = self.outlet_cell_tags.indices[mask]
        territory_vols[tag] = vols[local_cells].sum()
        # beta_snk.x.array[local_cells] = Q_bio / territory_vols[tag] / (uniq_tags_local.max() + 1) * (1/(Pcap - Psnk_avg)) # set metabolic demand per territory
        beta_snk.x.array[local_cells] = Q_bio / V_bio * (1/(Pcap - Psnk_avg)) # set metabolic demand per territory

    # beta_src.x.array[:] = (metabolic_demand) * (1/(Psrc_avg -Pcap)) # Source term coefficient
    # beta_snk.x.array[:] = (metabolic_demand) * (1/(Pcap - Psnk_avg))  # Sink term coefficient

    np.savez(current_dir / "perfusion_parameters.npz",
         beta_src=beta_src.x.array,
         beta_snk=beta_snk.x.array)

    return beta_src, beta_snk

class PerfusionSolver:
    def __init__(self, mesh_inlet_file: str, mesh_outlet_file: str, pres_inlet_file: str, pres_outlet_file: str, flow_inlet_file: str, flow_outlet_file: str):
        """
        Initialize the PerfusionSolver with a given STL file and branching data.
        """
        self.element_degree = 1  # Degree of the finite element space for pressure
        self.mesh, self.inlet_cell_tags = import_mesh(mesh_inlet_file)
        _, self.outlet_cell_tags = import_mesh(mesh_outlet_file)
        self.p_src = import_pressure_data(self, pres_inlet_file, k=self.element_degree)
        self.p_snk = import_pressure_data(self, pres_outlet_file, k=self.element_degree)
        self.flow = import_flow_data(self, flow_inlet_file, self.inlet_cell_tags) # + import_flow_data(self, flow_outlet_file, self.outlet_cell_tags)
        print(f"Total flow rate from inlet: {self.flow} cm^3/s", flush=True)
        

    def setup(self, init=True):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        
        k = self.element_degree
        P_el = element("Lagrange", self.mesh.basix_cell(), k)
        u_el = element("DG", self.mesh.basix_cell(), k-1, shape=(self.mesh.geometry.dim,))
        # u_el = element("BDM", self.mesh.basix_cell(), k, shape=(self.mesh.geometry.dim,))

        # Define function spaces
        W = dfx.fem.functionspace(self.mesh, P_el) # Pressure function space
        V = dfx.fem.functionspace(self.mesh, u_el) # Velocity function space

        kappa = 1e-8 # convert from cm^2/(Pa s) to cm^2/(dyne s)
        mu = 1
        kappa_over_mu = fem.Constant(self.mesh, dfx.default_scalar_type(kappa/mu))
        phi = fem.Constant(self.mesh, dfx.default_scalar_type(1)) # Porosity of the medium, ranging from 0 to 1
        # S = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term

        # Trial and test functions
        v, p = ufl.TestFunction(W), ufl.TrialFunction(W)
        
        # Define variational problem
        if init:
            beta_src, beta_snk = single_compartment(self, self.p_src, self.p_snk) # Source and sink terms
        else:
            DG = dfx.fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
            beta_src = dfx.fem.Function(DG)
            beta_snk = dfx.fem.Function(DG)
            data = np.load(current_dir / "perfusion_parameters.npz")
            beta_src.x.array[:] = data["beta_src"]; beta_snk.x.array[:] = data["beta_snk"]

        print(f"Length of p_src:", len(self.p_src[-1].x.array), flush=True)
        print(f"Mean of p_src:", np.mean(self.p_src[-1].x.array), flush=True)
        # print(f"Beta_src (1/s): {float(beta_src.value)}", flush=True)
        # print(f"Beta_snk (1/s): {float(beta_snk.value)}", flush=True)
        fLHS = p * (beta_snk + beta_src) 
        fRHS = beta_src * self.p_src[-1] + beta_snk * self.p_snk[-1]
        # fRHS = beta_src * self.p_src[-1] + beta_snk * Psnk
        a = kappa_over_mu * dot(grad(p), grad(v)) * ufl.dx + inner(fLHS, v) * ufl.dx
        f =  inner(fRHS, v) * ufl.dx  # residual form of our equation


        self.a = a
        self.f = f

        # Apply Dirichlet BCs
        self.bcs = []
        problem = LinearProblem(self.a, self.f, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

        try:
            # projector = Projector(f, V, bcs = [])
            p_h = problem.solve()
            # fig = plt.figure()
            # im = plot(vf)
            # plt.colorbar(im, format="%.2e")
        except PETSc.Error as e:  # type: ignore
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

        # sigma_h, u_h = w_h.split()

        projector = Projector(V)
        vel_f = projector(-kappa_over_mu * grad(p_h) / phi)

        Q0 = fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
        p_in = dfx.fem.Function(Q0)
        p_in.interpolate(p_h)   # pulls pressure from the mixed vector into standalone DG0
        print("Len of p_in:", len(p_in.x.array))

        compute_and_write_inlet_mbf(self, self.inlet_cell_tags, p_DG0=p_in, beta=beta_src, xdmf_out=str(current_dir / "out_darcy/mbf_inlet.xdmf"), bp_out=str(current_dir / "out_darcy/mbf_inlet.bp"))
        compute_and_write_outlet_mbf(self, self.outlet_cell_tags, p_DG0=p_in, beta=beta_snk, xdmf_out=str(current_dir / "out_darcy/mbf_outlet.xdmf"), bp_out=str(current_dir / "out_darcy/mbf_outlet.bp"))

        with XDMFFile(self.mesh.comm, current_dir/"out_darcy/p.xdmf","w") as file:
            file.write_mesh(self.mesh)
            P1 = element("Lagrange", self.mesh.basix_cell(), degree=1)
            p_interp = fem.Function(fem.functionspace(self.mesh, P1))
            p_interp.interpolate(p_h)
            file.write_function(p_interp, 0.0)

        with XDMFFile(self.mesh.comm, current_dir/"out_darcy/u.xdmf","w") as file:
            file.write_mesh(self.mesh)
            P1_vec = element("Lagrange", self.mesh.basix_cell(), degree=1, shape=(self.mesh.geometry.dim,))
            u_interp = fem.Function(fem.functionspace(self.mesh, P1_vec))
            u_interp.interpolate(vel_f)
            file.write_function(u_interp, 0.0)

        vtkfile = VTKFile(MPI.COMM_WORLD, "u.vtu", "w")

        # Write the function to the VTK file
        vtkfile.write_function(u_interp)

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
    mesh_inlet_file = "../voronoi/territories_inlet.xdmf"
    mesh_outlet_file = "../voronoi/territories_outlet.xdmf"
    pres_inlet_file = "../voronoi/p_src_inlet_series.bp"
    pres_outlet_file = "../voronoi/p_src_outlet_series.bp"
    flow_inlet_file = "../voronoi/q_src_inlet_series.bp"
    flow_outlet_file = "../voronoi/q_src_outlet_series.bp"
    solver = PerfusionSolver(mesh_inlet_file, mesh_outlet_file, pres_inlet_file, pres_outlet_file, flow_inlet_file, flow_outlet_file)
    solver.setup(init=True)
    print("Perfusion solve complete.")
    # Further processing can be added here