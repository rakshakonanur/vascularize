import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
import os
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from scipy.spatial     import cKDTree
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
import logging
logging.basicConfig(level=logging.DEBUG)
current_dir = Path("/Users/rakshakonanur/Documents/Research/vascularize/src/solves")
from send2trash import send2trash

WALL = 0
OUTLET = 1
OUTFLOW = 2

"""
    From the DOLFINx tutorial: Mixed formulation of the Poisson equation
    https://docs.fenicsproject.org/dolfinx/v0.7.2/python/demos/demo_mixed-poisson.html

    Weak imposition of Dirichlet boundary conditions using Nitsche's method:
    https://jsdokken.com/dolfinx-tutorial/chapter1/nitsche.html
"""

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

    p_series = []
    for t in ts:
        adios4dolfinx.read_function(pres_file, p_src, name="p_src", time=t)
        p_src.x.array[:] = p_src.x.array       
        p_src.x.scatter_forward()
        p_series.append(p_src.copy())

    DG1 = fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), k))
    for i in range(len(p_series)):
        p_tmp = fem.Function(DG1)
        p_tmp.interpolate(p_series[i])
        p_series[i] = p_tmp 

    # with io.XDMFFile(MPI.COMM_WORLD, "out_mixed_poisson/import_p_src.xdmf", "w") as file:
    #     file.write_mesh(self.mesh)
    #     file.write_function(p_series[-1])

    print(f"Imported {len(p_series)} time steps from {pres_file}", flush=True)
    print(f"Min/Max pressure in last time step: {np.min(p_series[-1].x.array)}/{np.max(p_series[-1].x.array)}", flush=True)
    return p_series

def import_flow_data(self, flow_file: str):
    """
    Import flow data from an XDMF file.
    """
    # mesh = adios4dolfinx.read_mesh(filename = flow_file, comm=MPI.COMM_WORLD)
    ts = adios4dolfinx.read_timestamps(filename = flow_file, comm=MPI.COMM_WORLD, function_name="q_src_density")
    DG0 = fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
    q_src = fem.Function(DG0)

    q_series = []
    for t in ts:
        adios4dolfinx.read_function(flow_file, q_src, name="q_src_density", time=t)
        q_src.x.array[:] = q_src.x.array      
        q_src.x.scatter_forward()
        q_series.append(q_src.copy())

    with io.XDMFFile(MPI.COMM_WORLD, current_dir/"out_darcy/import_q_src.xdmf", "w") as file:
        file.write_mesh(self.mesh)
        file.write_function(q_series[-1])

    print(f"Imported {len(q_series)} time steps from {flow_file}", flush=True)
    print(f"Total flow rate in last time step: {np.sum(np.unique(q_series[-1].x.array))}", flush=True)
    return np.sum(np.unique(q_series[-1].x.array))


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

def compute_and_write_mbf(self, p_DG0, beta, xdmf_out="out_mixed_poisson/mbf.xdmf",
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
    labels = _labels_from_cell_tags(mesh, self.cell_tags)  # <-- make sure this is CELL tags
    vols   = _cell_volumes_Q0(mesh)

    # --- Density and per-cell flux (all NumPy) ---
    density   = beta_arr * (psrc_arr - p_arr)     # units: (1/s)*Pa -> whatever model uses
    cell_flux = density * vols                    # integrated flow per cell

    # --- Sum by tag (local) ---
    uniq_tags_local = np.unique(self.cell_tags.values.astype(int))
    q_local = {int(tag): float(cell_flux[labels == int(tag)].sum()) for tag in uniq_tags_local}

    # --- Reduce to global sums ---
    q_global = {int(tag): comm.allreduce(q_local[int(tag)], op=MPI.SUM) for tag in uniq_tags_local}

    # --- Build output DG0 fields ---
    mbf_density = fem.Function(Q0, name=field_names[1])
    mbf_density.x.array[:] = density
    mbf_density.x.scatter_forward()

    mbf_qTi = fem.Function(Q0, name=field_names[0])
    mbf_qTi.x.array[:] = 0.0
    for tag in np.unique(self.cell_tags.values).astype(int):
        mask = (self.cell_tags.values == int(tag))
        local_cells = self.cell_tags.indices[mask]
        mbf_qTi.x.array[local_cells] = q_global.get(int(tag), 0.0)
    mbf_qTi.x.scatter_forward()

    mbf_qTi_per_vol = fem.Function(Q0, name=field_names[0]+"_per_vol")
    mbf_qTi_per_vol.x.array[:] = 0.0
    for tag in np.unique(self.cell_tags.values).astype(int):
        mask = (self.cell_tags.values == int(tag))
        local_cells = self.cell_tags.indices[mask]
        mbf_qTi_per_vol.x.array[local_cells] = q_global.get(int(tag), 0.0) / vols[local_cells].sum()
    mbf_qTi_per_vol.x.scatter_forward()

    # --- Write XDMF (mesh, tags, fields) ---
    with io.XDMFFile(comm, xdmf_out, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(mbf_qTi)
        xdmf.write_function(mbf_density)
        xdmf.write_function(mbf_qTi_per_vol)

    # Write using adios4dolfinx as well
    # adios4dolfinx.write_mesh(current_dir/"out_mixed_poisson/mbf.bp", mesh, engine="BP4")
    if Path(current_dir/"out_mixed_poisson/mbf.bp").exists():
        send2trash(current_dir/"out_mixed_poisson/mbf.bp")

    adios4dolfinx.write_function_on_input_mesh(current_dir/"out_mixed_poisson/mbf.bp", mbf_qTi, time=0.0, name=field_names[0])
    adios4dolfinx.write_function_on_input_mesh(current_dir/"out_mixed_poisson/mbf.bp", mbf_density, time=0.0, name=field_names[1])

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

    Pcap = 15*1333.22  # Capillary pressure, converted to dyne/cm^2
    Psnk = sink_pressures # Sink pressure

    Psrc_values = source_pressures[-1].x.array  # Get the values of the outlet BCs
    Psnk_values = sink_pressures[-1].x.array
    # print(f"Outlet pressures: {Psrc_values}", flush=True)
    Psrc_avg = np.mean(Psrc_values)  # Average pressure at the outlets
    Psnk_avg = np.mean(Psnk_values)
    print(f"Average pressure at outlets: {Psrc_avg}", flush=True)

    beta_src = (Q_bio / V_bio) * (1/(Psrc_avg -Pcap)) # Source term coefficient
    beta_snk = (Q_bio / V_bio) * (1/(Pcap - Psnk_avg))  # Sink term coefficient

    # Psnk_c = dfx.fem.Constant(self.mesh, PETSc.ScalarType(Psnk))
    beta_src_c = dfx.fem.Constant(self.mesh, PETSc.ScalarType(beta_src))
    beta_snk_c = dfx.fem.Constant(self.mesh, PETSc.ScalarType(beta_snk))

    # If initializing, save these values to a file for later reference
    with open(current_dir/"perfusion_parameters.txt", "w") as f:
        f.write(f"beta_src (1/s): {beta_src}\n")
        f.write(f"beta_snk (1/s): {beta_snk}\n")

    return beta_src_c, beta_snk_c

class PerfusionSolver:
    def __init__(self, mesh_tag_file: str, pres_inlet_file: str, pres_outlet_file: str, flow_file: str):
        """
        Initialize the PerfusionSolver with a given STL file and branching data.
        """
        self.D_value = 1e-2
        self.write_output = True
        self.element_degree = 1
        self.mesh, self.cell_tags = import_mesh(mesh_tag_file)
        self.p_src = import_pressure_data(self, pres_inlet_file, k=self.element_degree-1)
        self.p_snk = import_pressure_data(self, pres_outlet_file, k=self.element_degree-1)
        self.flow = import_flow_data(self, flow_file)

    def setup(self, init: bool = True):
        ''' Setup the solver. '''
        
        fdim = self.mesh.topology.dim -1 
        
        k = self.element_degree
        deg = 2*k + 1
        P_el = element("DG", self.mesh.basix_cell(), k-1)
        u_el = element("BDM", self.mesh.basix_cell(), k, shape=(self.mesh.geometry.dim,))
        M_el = mixed_element([P_el, u_el])

        # Define function spaces
        M = dfx.fem.functionspace(self.mesh, M_el) # Mixed function space
        W, _ = M.sub(0).collapse()  # Pressure function space
        V, _ = M.sub(1).collapse()  # Velocity function space
        (p, u) = ufl.TrialFunctions(M) # Trial functions for pressure and velocity
        (v, w) = ufl.TestFunctions(M) # Test functions for pressure and velocity

        dx = Measure("dx", self.mesh, metadata={"quadrature_degree": deg}) # Cell integrals
        self.ds = Measure("ds", self.mesh, metadata={"quadrature_degree": deg}) # Facet integrals
        n = FacetNormal(self.mesh) # Normal vector on facets

        kappa = 2e-8 / 10 # convert from cm^2/(Pa s) to cm^2/(dyne s)
        mu = 1
        kappa_over_mu = dfx.default_scalar_type(kappa/mu)
        Kinv  = fem.Constant(self.mesh, kappa_over_mu**-1)
        hf = ufl.CellDiameter(self.mesh) # Cell diameter
        nitsche = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(100.0)) # Nitsche parameter
        nitsche_outflow = dfx.fem.Constant(self.mesh, dfx.default_scalar_type(10000.0)) # Nitsche parameter
        S = fem.Constant(self.mesh, dfx.default_scalar_type(0.0)) # Source term

        # Zero flux on boundaries is default for BDM elements, so no need to explicitly impose

        if init:
            beta_src, beta_snk = single_compartment(self, self.p_src, self.p_snk) # Source and sink terms
        else:
            with open(current_dir/"perfusion_parameters.txt", "r") as f:
                lines = f.readlines()
                beta_src_value = float(lines[0].split(":")[1].strip())
                beta_snk_value = float(lines[1].split(":")[1].strip())
            beta_src = dfx.fem.Constant(self.mesh, PETSc.ScalarType(beta_src_value))
            beta_snk = dfx.fem.Constant(self.mesh, PETSc.ScalarType(beta_snk_value))
            Psnk = dfx.fem.Constant(self.mesh, PETSc.ScalarType(0.0))
        
        print(f"Length of p_src:", len(self.p_src[-1].x.array), flush=True)
        print(f"Mean of p_src:", np.mean(self.p_src[-1].x.array), flush=True)
        print(f"Beta_src (1/s): {float(beta_src.value)}", flush=True)
        print(f"Beta_snk (1/s): {float(beta_snk.value)}", flush=True)
        fLHS = p * (beta_snk + beta_src) 
        fRHS = beta_src * self.p_src[-1] + beta_snk * self.p_snk[-1]

        a = Kinv * inner(u, w) * dx - inner(p, div(w)) * dx + inner(div(u), v) * dx  + inner(fLHS, v) * dx  + ufl.inner(p, ufl.dot(w, n)) * self.ds
        # a = Kinv * inner(u, w) * dx + inner(grad(p), w) * dx + inner(div(u), v) * dx  + inner(fLHS, v) * dx
        L = inner(fRHS, v) * dx + inner(S, v) * dx
        # # L = -inner(S, v) * ufl.dx  # residual form of our equation

        self.a = a
        self.L = L

        # Apply Dirichlet BCs
        bcs = []
        self.bcs = bcs
        problem = LinearProblem(self.a, self.L, bcs=self.bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

        try:
            w_h = problem.solve()
        except PETSc.Error as e:  # type: ignore
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

        p_h, u_h = w_h.split()
        print("pressure min/max:", float(p_h.x.array.min()), float(p_h.x.array.max()))
        print("Len of p_h:", len(p_h.x.array))
    
        Q0 = fem.functionspace(self.mesh, element("DG", self.mesh.basix_cell(), 0))
        p_in = dfx.fem.Function(Q0)
        p_in.interpolate(p_h)   # pulls pressure from the mixed vector into standalone DG0
        print("Len of p_in:", len(p_in.x.array))

        compute_and_write_mbf(self, p_DG0=p_in, beta=beta_src, xdmf_out=str(current_dir / "out_mixed_poisson/mbf.xdmf"))

        # with io.XDMFFile(self.mesh.comm, str(current_dir / "out_mixed_poisson/p_src.xdmf"), "w") as file:
        #     file.write_mesh(self.mesh)
        #     file.write_function(self.p_src[-1])

        with io.XDMFFile(self.mesh.comm, str(current_dir / "out_mixed_poisson/p.xdmf"), "w") as file:
            file.write_mesh(self.mesh)
            file.write_function(p_in)

        with io.XDMFFile(self.mesh.comm, str(current_dir / "out_mixed_poisson/u.xdmf"), "w") as file:
            file.write_mesh(self.mesh)
            Vviz = fem.functionspace(self.mesh, element("Lagrange", self.mesh.basix_cell(), 1, shape=(self.mesh.geometry.dim,)))
            u_proj = fem.Function(Vviz, name="u_proj")

            u_trial = ufl.TrialFunction(Vviz)
            v_test = ufl.TestFunction(Vviz)

            a_proj = ufl.inner(u_trial, v_test) * ufl.dx
            L_proj = ufl.inner(u_h, v_test) * ufl.dx

            problem = fem.petsc.LinearProblem(
                a_proj, L_proj,
                bcs=[],
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
            )
            u_proj = problem.solve()

            # Write projected function
            file.write_function(u_proj)

        vtkfile = VTKFile(MPI.COMM_WORLD, current_dir/"u.vtu", "w")

        # Write the function to the VTK file
        vtkfile.write_function(u_proj)

if __name__ == "__main__":
    mesh_file = "../voronoi/territories_inlet.xdmf"
    pres_inlet_file = "../voronoi/p_src_inlet_series.bp"
    pres_outlet_file = "../voronoi/p_src_outlet_series.bp"
    flow_file = "../voronoi/q_src_inlet_series.bp"
    solver = PerfusionSolver(mesh_file, pres_inlet_file, pres_outlet_file, flow_file)
    solver.setup(init=True)
    print("Perfusion solve complete.")
