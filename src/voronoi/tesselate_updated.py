# ---------- CSV -> terminal seeds (+ optional snapping) ----------
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from dolfinx import mesh as dmesh, fem
from basix.ufl import element, mixed_element
import dolfinx as dfx
from dolfinx.io import XDMFFile
import ufl
from send2trash import send2trash

# Try SciPy for KDTree; fall back to a tiny pure-numpy version if unavailable
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:  # very small fallback (O(N*M), fine for N~100 seeds)
    KDTree = None

current_dir = Path("/Users/rakshakonanur/Documents/Research/vascularize/src/voronoi")
# -------- 1) Load terminal seeds from branchingData_0.csv --------
def load_terminal_seeds_from_csv(csv_path: str):
    """
    Reads branchingData_0.csv and extracts terminal (leaf) segments:
      - seed position = distalCoords (x,y,z)
      - diameter = 2 * Radius
      - seed_id = prefer DistalNodeIndex if present, else a stable row index

    Returns:
      seeds_df: DataFrame with columns:
        ['seed_id','seed_x','seed_y','seed_z','Radius','diameter','Flow','Depth',
         'Parent','ProximalNodeIndex','DistalNodeIndex']
      seeds_xyz: (N,3) float array
      diameters: (N,) float array
    """
    df = pd.read_csv(csv_path)

    def _empty(v):
        if pd.isna(v):
            return True
        try:
            return int(v) < 0
        except Exception:
            return True

    is_leaf = df.apply(lambda r: _empty(r.get('Child1')) and _empty(r.get('Child2')), axis=1)
    leaves = df[is_leaf].copy()

    # Build IDs (prefer DistalNodeIndex if present)
    def _make_id(row, i):
        if pd.notna(row.get('DistalNodeIndex')):
            try:
                return int(row['DistalNodeIndex'])
            except Exception:
                pass
        if pd.notna(row.get('Index')):
            try:
                return int(row['Index'])
            except Exception:
                pass
        return int(i)

    leaves = leaves.reset_index(drop=True)
    seed_ids = [_make_id(leaves.loc[i], i) for i in range(len(leaves))]

    leaves['seed_id'] = seed_ids
    leaves['seed_x']  = leaves['distalCoordsX'].astype(float)
    leaves['seed_y']  = leaves['distalCoordsY'].astype(float)
    leaves['seed_z']  = leaves['distalCoordsZ'].astype(float)
    leaves['diameter'] = 2.0 * leaves['Radius'].astype(float)

    cols = ['seed_id','seed_x','seed_y','seed_z','Radius','diameter','Flow','Depth',
            'Parent','ProximalNodeIndex','DistalNodeIndex']
    # Ensure columns exist
    for c in cols:
        if c not in leaves.columns:
            leaves[c] = np.nan
    seeds_df = leaves[cols].copy()

    # Numpy outputs
    seeds_xyz = seeds_df[['seed_x','seed_y','seed_z']].to_numpy(dtype=float)
    diameters = seeds_df['diameter'].to_numpy(dtype=float)

    return seeds_df, seeds_xyz, diameters


# -------- helpers to build geometry caches --------
def _cell_barycenters(mesh: dmesh.Mesh) -> np.ndarray:
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    cmap = mesh.topology.index_map(tdim)
    n_local = cmap.size_local

    # Ensure connectivity cell->vertex
    c2v = mesh.topology.connectivity(tdim, 0)
    if c2v is None:
        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)
    X = mesh.geometry.x
    bcc = np.zeros((n_local, gdim), dtype=float)
    for c in range(n_local):
        verts = c2v.links(c)
        bcc[c] = X[verts].mean(axis=0)
    return bcc

def _boundary_facet_centroids(mesh: dmesh.Mesh, facet_tags: fem.MeshTags | None = None,
                              restrict_to_values: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (centroids, local_facet_ids) for boundary facets, optionally restricted to tag values.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)

    # Determine which facets we consider
    if facet_tags is not None:
        # Either take all tagged facets or just those with certain values
        if restrict_to_values is None:
            sel = np.arange(facet_tags.values.size, dtype=np.int32)
        else:
            mask = np.isin(facet_tags.values, np.array(restrict_to_values, dtype=facet_tags.values.dtype))
            sel = np.flatnonzero(mask).astype(np.int32)
        facet_ids = facet_tags.indices[sel]
    else:
        # Fallback: treat all exterior boundary facets as candidates
        f_imap = mesh.topology.index_map(fdim)
        n_facets_local = f_imap.size_local
        facet_ids = np.arange(n_facets_local, dtype=np.int32)

    f2v = mesh.topology.connectivity(fdim, 0)
    X = mesh.geometry.x
    cents = np.zeros((facet_ids.size, mesh.geometry.dim), dtype=float)
    for i, f in enumerate(facet_ids):
        verts = f2v.links(f)
        cents[i] = X[verts].mean(axis=0)
    return cents, facet_ids


# -------- 2) Snap seeds to boundary facets (e.g., outlets) --------
def snap_seeds_to_boundary_facets(mesh: dmesh.Mesh,
                                  seeds_xyz: np.ndarray,
                                  facet_tags: fem.MeshTags | None = None,
                                  restrict_to_values: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (snapped_xyz, nearest_facet_ids_local)
      - snapped_xyz: (N,3) equals facet centroids (nearest) for each seed
      - nearest_facet_ids_local: (N,) local facet indices (use tags.indices to map)
    If 'restrict_to_values' provided, only those tag values are considered.
    """
    cents, facet_ids = _boundary_facet_centroids(mesh, facet_tags, restrict_to_values)

    if KDTree is not None:
        tree = KDTree(cents)
        _, idx = tree.query(seeds_xyz, k=1)
    else:
        # simple O(N*M) nearest neighbor
        idx = np.argmin(((seeds_xyz[:, None, :] - cents[None, :, :])**2).sum(axis=2), axis=1)

    snapped = cents[idx]
    return snapped, facet_ids[idx]


# -------- 3) Snap seeds to nearest cell (barycenter) --------
def snap_seeds_to_cells(mesh: dmesh.Mesh, seeds_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (snapped_xyz, nearest_cell_ids_local)
      - snapped_xyz: (N,3) = nearest cell barycenter for each seed
      - nearest_cell_ids_local: (N,)
    """
    bcc = _cell_barycenters(mesh)
    if KDTree is not None:
        tree = KDTree(bcc)
        _, idx = tree.query(seeds_xyz, k=1)
    else:
        idx = np.argmin(((seeds_xyz[:, None, :] - bcc[None, :, :])**2).sum(axis=2), axis=1)
    return bcc[idx], idx

import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh, fem
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import cell_entity_type, cell_num_entities

def cell_barycenters(mesh):
    """
    Compute barycenters of locally-owned top-dimensional cells.
    Works across recent DOLFINx versions without using geometry.dofmap.
    """
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim

    # Ensure cell->vertex connectivity is built
    c2v = mesh.topology.connectivity(tdim, 0)
    if c2v is None:
        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)

    # Local cell count
    n_local = mesh.topology.index_map(tdim).size_local

    # Vertex coordinates
    X = mesh.geometry.x

    # Barycenters
    bcc = np.zeros((n_local, gdim), dtype=float)
    for c in range(n_local):
        verts = c2v.links(c)         # vertex indices of cell c
        bcc[c] = X[verts].mean(axis=0)
    return bcc

def laguerre_weights_from_diameters(diameters: np.ndarray, mode: str = "d2", kappa: float = 1.0) -> np.ndarray:
    """
    Compute power weights w_i from diameters.
    mode: "d2" (standard power diagram), or "d4" to bias territories ~ flow.
    w_i are in length^2 units (power diagram expects squared length).
    """
    d = np.asarray(diameters, dtype=float)
    if mode == "d2":
        return (0.5 * d)**2 * kappa           # (radius)^2 * κ
    elif mode == "d4":
        # embed d^4 scaling into a squared-length weight; pick κ to tune territory sizes
        return ((0.5 * d)**2 * (d**2)) * kappa  # ~ r^2 * d^2 = (r^2)*(4 r^2) ∝ r^4
    else:
        raise ValueError("mode must be 'd2' or 'd4'.")

def label_cells_by_laguerre(
    mesh: dmesh.Mesh,
    seed_points: np.ndarray,   # shape (N,3)
    seed_weights: np.ndarray,# shape (N,)
    weight_mode: str = "d2",
    kappa: float = 1.0,
    tie_break: str = "smallest_index"
):
    """
    Assign each locally-owned cell to its Laguerre (power) region based on
    terminal segment seeds and diameters.
    Returns (cell_tags_local, unique_labels) where cell_tags_local has length n_local_cells.
    """
    comm = mesh.comm
    tdim = mesh.topology.dim
    cmap = mesh.topology.index_map(tdim)
    n_local = cmap.size_local

    # Barycenters for classification (fast and robust). For extra robustness,
    # you could sample multiple points per cell and vote.
    bcc = cell_barycenters(mesh)  # (n_local, 3)

    # Prepare weights
    seeds = np.asarray(seed_points, dtype=np.float64)
    w = laguerre_weights_from_diameters(np.asarray(seed_weights, float), mode=weight_mode, kappa=kappa)

    # Compute argmin_i (||x - s_i||^2 - w_i) for all local cells
    # Vectorized in chunks to save memory if N is large
    N = seeds.shape[0]
    B = bcc.shape[0]
    labels = np.empty(B, dtype=np.int32)
    chunk = 4096  # tune if memory is tight

    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        X = bcc[start:stop]                      # (b,3)
        # (b,N): |X|^2 - 2 X.S^T + |S|^2 - w
        # Implemented as (X^2).sum + (S^2).sum - 2 X S^T - w
        X2 = np.sum(X*X, axis=1, keepdims=True) # (b,1)
        S2 = np.sum(seeds*seeds, axis=1)        # (N,)
        XS = X @ seeds.T                        # (b,N)
        power = X2 + S2[None,:] - 2.0*XS - w[None,:]  # (b,N)
        # Choose minimal power distance
        idx = np.argmin(power, axis=1)
        labels[start:stop] = idx.astype(np.int32)

    # Optional tie-break handling (rare with floats):
    # already resolved by argmin; if you want deterministic behavior beyond float,
    # keep 'smallest_index'.

    # Gather unique labels for convenience
    unique_local = np.unique(labels)
    unique_global = None
    if comm.size > 1:
        # Gather to root then bcast
        gathered = comm.gather(unique_local, root=0)
        if comm.rank == 0:
            unique_global = np.unique(np.concatenate(gathered))
        unique_global = comm.bcast(unique_global, root=0)
    else:
        unique_global = unique_local

    return labels, unique_global

def make_cell_meshtags(mesh: dmesh.Mesh, labels_local: np.ndarray) -> fem.MeshTags:
    """
    Create a distributed MeshTags over cells from local labels.
    """
    tdim = mesh.topology.dim
    cmap = mesh.topology.index_map(tdim)
    n_local = cmap.size_local
    # cells are [0, n_local) locally
    cell_indices = np.arange(n_local, dtype=np.int32)
    mt = dfx.mesh.meshtags(mesh, tdim, cell_indices, labels_local.astype(np.int32))
    return mt

def write_territories(mesh: dmesh.Mesh, cell_tags: fem.MeshTags, path: str = "territories.xdmf"):
    """
    Write mesh and cell tags for visualization/analysis.
    """
    with XDMFFile(mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags, mesh.geometry)

# ------------------------- Integration: 1D -> 3D sources -------------------------
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import fem
import ufl

# ADIOS2 I/O
try:
    import adios4dolfinx
except Exception as _e:
    adios4dolfinx = None

def read_1d_mesh_from_bp(bp_mesh_path: str, comm=MPI.COMM_WORLD):
    return adios4dolfinx.read_mesh(bp_mesh_path, comm=comm)

def read_last_step_bp_array(mesh, bp_path: Path, function_name: str) -> np.ndarray:
    """
    Read the last-step array from a .bp (ADIOS2) file by picking the first non-scalar var.
    This is a heuristic but works for typical DG0 time series written with adios4dolfinx.

    Returns a 1D numpy array.
    """

    ts = adios4dolfinx.read_timestamps(filename = bp_path, comm=MPI.COMM_WORLD, function_name=function_name)
    P1    = dfx.fem.functionspace(mesh, ("CG", 1))
    q_src = fem.Function(P1)
    q_series = []
    for t in ts:
        adios4dolfinx.read_function(bp_path, q_src, name=function_name, time=ts[-1])
        q_src.x.array[:] = q_src.x.array       
        q_series.append(q_src.copy())

    print(f"Imported {len(q_series)} time steps from {bp_path}", flush=True)
    return q_series[-1].x.array.copy()

def map_seeds_to_1d_nodes(mesh_1d, seeds_xyz, tol: float = 1e-6):
    X1 = mesh_1d.geometry.x  # (n_nodes, gdim)
    if KDTree is not None:
        tree = KDTree(X1)
        d, idx = tree.query(seeds_xyz, k=1)
    else:
        import numpy as np
        diff = seeds_xyz[:, None, :] - X1[None, :, :]
        d2 = (diff**2).sum(axis=2)
        idx = d2.argmin(axis=1)
        d = np.sqrt(d2[np.arange(d2.shape[0]), idx])
    found = d <= tol
    return idx.astype(int), found

def read_series_1d(bp_mesh: str, bp_pressure: str, bp_flow: str, comm=MPI.COMM_WORLD, fname="f"):
    ts = adios4dolfinx.read_timestamps(bp_pressure, comm=comm, function_name=fname)
    mesh_1d = adios4dolfinx.read_mesh(bp_mesh, comm=comm)

    P1    = dfx.fem.functionspace(mesh_1d, ("CG", 1))

    p_in = dfx.fem.Function(P1)
    u_in = dfx.fem.Function(P1)

    p_series, u_series = [], []
    for t in ts:
        adios4dolfinx.read_function(bp_pressure, p_in, time=t, name=fname)
        adios4dolfinx.read_function(bp_flow,     u_in, time=t, name=fname)
        # p_in.x.array[:] = p_in.x.array*np.random.default_rng().integers(low=1, high=10, size=p_in.x.array.shape)  # CHANGE LATER, JUST TO EXAGGERATE DIFFERENCES IN TERRITORIES
        p_series.append(p_in.x.array.copy())
        u_series.append(u_in.x.array.copy())
    return ts, mesh_1d, p_series, u_series

def territory_volumes(mesh_3d, cell_tags):
    dxT = ufl.Measure("dx", domain=mesh_3d, subdomain_data=cell_tags)
    import numpy as np
    vols = {}
    for tag in np.unique(cell_tags.values):
        vols[int(tag)] = fem.assemble_scalar(fem.form(1.0 * dxT(int(tag))))

    return vols

def paint_piecewise_constant_Q0(mesh_3d, cell_tags, tag_to_value, name="field"):
    Q0 = fem.functionspace(mesh_3d, element("DG", mesh_3d.basix_cell(), 0))
    f = fem.Function(Q0, name=name)
    f.x.array[:] = 0.0
    import numpy as np
    for tag, val in tag_to_value.items():
        mask = (cell_tags.values == int(tag))
        local_cells = cell_tags.indices[mask]
        f.x.array[local_cells] = val
    f.x.scatter_forward()
    return f

def assign_and_write_sources(
    csv_seeds_path: str,
    territories_xdmf: str,
    bp_mesh_1d: str,
    bp_pressure_1d: str,
    bp_flow_1d: str,
    viz_pressure_bp: str,
    viz_flow_bp: str,
    out_bp_p_src: str = "p_src_series.bp",
    out_bp_q_src: str = "q_src_series.bp",
    node_match_tol: float = 1e-6,
    flow_density_by_volume: bool = False,
    fname_in="f",
    fname_out_p="p_src",
    fname_out_q="q_src_density",
    verbose=True,
    comm=MPI.COMM_WORLD,
):
    if adios4dolfinx is None:
        raise RuntimeError("adios4dolfinx is not available in this environment. Please install it.")

    rank = comm.rank

    # ---------------- 1) Load seeds (DataFrame, coords) ----------------
    # Your loader returns: (seeds_df, seeds_xyz, diameters)
    seeds_df, seeds_xyz, _diams = load_terminal_seeds_from_csv(csv_seeds_path)

    # Row index after reset_index in the loader is our canonical "seed position"
    # (0..N-1 order used during tessellation)
    seed_row_index = np.arange(len(seeds_df), dtype=np.int64)

    if verbose and rank == 0:
        print(f"[seeds] Loaded {len(seeds_df)} terminal seeds from {csv_seeds_path}")
        print(f"[seeds] First 5 seed_id values (not used for mapping): {seeds_df['seed_id'].values[:5]}")

    # ---------------- 2) 1D series ----------------
    # Your read_series_1d signature here expects (bp_mesh_1d, bp_pressure_1d, bp_flow_1d, ...)
    ts, mesh_1d, p_series, u_series = read_series_1d(
        bp_mesh_1d, bp_pressure_1d, bp_flow_1d, comm=comm, fname=fname_in
    )
    if verbose and rank == 0:
        print(f"[1D] Read {len(ts)} timesteps from checkpoints.")

    # ---------------- 3) Map seeds -> nearest 1D node ----------------
    node_ids, found = map_seeds_to_1d_nodes(mesh_1d, seeds_xyz, tol=node_match_tol)

    if not np.all(found) and rank == 0:
        missing = np.where(~found)[0]
        print(f"[warn] {missing.size} seeds did not match a 1D node within tol={node_match_tol}. Skipping those seeds.")

    valid_idx = np.where(found)[0]
    node_ids_valid = np.asarray(node_ids)[valid_idx]
    seed_rows_valid = seed_row_index[valid_idx]   # positions into the seed list

    if verbose and rank == 0:
        print(f"[map] matched {len(seed_rows_valid)}/{len(seeds_df)} seeds to 1D nodes.")

    # ---------------- 4) Territories (cell tags) ----------------
    with XDMFFile(comm, territories_xdmf, "r") as xdmf:
        mesh_3d = xdmf.read_mesh(name="Grid")
        # Use the name you wrote for the meshtags; change 'mesh_tags' if different
        cell_tags = xdmf.read_meshtags(mesh_3d, name="mesh_tags")

    vols = territory_volumes(mesh_3d, cell_tags)

    # IMPORTANT: mapping by row index -> tag = sorted_unique_tags[row_index]
    terr_tags_unique_sorted = np.sort(np.unique(cell_tags.values)).astype(int)

    # Guard: numbers must be compatible
    if len(terr_tags_unique_sorted) < len(seeds_df) and rank == 0:
        print(f"[warn] fewer unique territory tags ({len(terr_tags_unique_sorted)}) than seeds ({len(seeds_df)}).")
    if len(terr_tags_unique_sorted) > len(seeds_df) and rank == 0:
        print(f"[warn] more unique territory tags ({len(terr_tags_unique_sorted)}) than seeds ({len(seeds_df)}).")

    # Diagnostic: show first few mappings
    if verbose and rank == 0:
        preview_n = min(5, len(seed_rows_valid))
        prev_rows = seed_rows_valid[:preview_n]
        prev_tags = terr_tags_unique_sorted[prev_rows]
        print(f"[map] row-index → tag preview: rows {prev_rows} -> tags {prev_tags}")

    # ---------------- 5) Function spaces & writers ----------------
    P_el = element("DG", mesh_3d.basix_cell(), 0)
    u_el = element("BDM", mesh_3d.basix_cell(), 1, shape=(mesh_3d.geometry.dim,))
    M_el = mixed_element([P_el, u_el])
    M = dfx.fem.functionspace(mesh_3d, M_el) # Mixed function space
    Q0, _ = M.sub(0).collapse()  # Pressure function space
    V1, _ = M.sub(1).collapse()  # Velocity function space
    DG0 = fem.functionspace(mesh_3d, element("DG", mesh_3d.basix_cell(), 0))
    p_src_func = fem.Function(Q0, name=fname_out_p)
    q_src_func = fem.Function(Q0, name=fname_out_q)# may need to switch V1 later

    # Optional VTX writers (you had these):
    vtx_pres = dfx.io.VTXWriter(MPI.COMM_WORLD, current_dir/viz_pressure_bp, [p_src_func], engine="BP4")
    vtx_flow = dfx.io.VTXWriter(MPI.COMM_WORLD, current_dir/viz_flow_bp, [q_src_func], engine="BP4")
    vtx_pres.write(0.0)
    vtx_flow.write(0.0)

    # ADIOS mesh containers for writing time series
    # adios4dolfinx.write_mesh(Path(out_bp_p_src), mesh_3d, engine="BP4")
    # adios4dolfinx.write_mesh(Path(out_bp_q_src), mesh_3d, engine="BP4")

    # if files exist, delete using send2trash
    if Path(out_bp_p_src).exists():
        send2trash(out_bp_p_src)
    if Path(out_bp_q_src).exists():
        send2trash(out_bp_q_src)

    # ---------------- 6) Time loop: pull 1D values, paint by row-index tags ----------------
    for it, t in enumerate(ts):
        # Your u_series/p_series come back as raw arrays (already the right shapes for your 1D problem)
        p_arr = p_series[it]
        u_arr = u_series[it]     # you use this as a scalar "flow" per node; keep as-is

        # Build per-tag scalars using ROW-INDEX-BASED mapping
        p_terminals = {}
        q_terminals = {}

        # For each VALID seed (row position r) and its matched 1D node nid:
        for r, nid in zip(seed_rows_valid.tolist(), node_ids_valid.tolist()):
            # Map row index -> territory tag via sorted unique tags
            if r >= len(terr_tags_unique_sorted):
                # out-of-range; skip
                continue
            tag = int(terr_tags_unique_sorted[r])

            # Pull terminal pressure/flow from 1D node
            p_i = float(p_arr[nid])
            q_i = float(u_arr[nid])   # scalar flow rate (your data)

            p_terminals[tag] = p_i
            q_terminals[tag] = q_i

        # Fill per-tag values for painting (defaults 0.0 when missing)
        all_tags = terr_tags_unique_sorted
        p_src_vals = {int(tag): float(p_terminals.get(int(tag), 0.0)) for tag in all_tags}

        if flow_density_by_volume:
            q_src_vals = {
                int(tag): (float(q_terminals.get(int(tag), 0.0)) / vols[int(tag)]) if vols.get(int(tag), 0.0) > 0.0 else 0.0
                for tag in all_tags
            }
        else:
            q_src_vals = {int(tag): float(q_terminals.get(int(tag), 0.0)) for tag in all_tags}

        # Paint DG0 fields
        p_src_func.x.array[:] = 0.0
        q_src_func.x.array[:] = 0.0

        for tag in all_tags:
            tag = int(tag)
            mask = (cell_tags.values == tag)
            local_cells = cell_tags.indices[mask]
            p_src_func.x.array[local_cells] = 1333.22 * p_src_vals[tag] # convert from mmHg to dyne/cm^2
            q_src_func.x.array[local_cells] = q_src_vals[tag]

        p_src_func.x.scatter_forward()
        q_src_func.x.scatter_forward()

        # Write this time step
        adios4dolfinx.write_function_on_input_mesh(Path(out_bp_p_src), u = p_src_func, time=float(t), name=fname_out_p)
        adios4dolfinx.write_function_on_input_mesh(Path(out_bp_q_src), u = q_src_func, time=float(t), name=fname_out_q)

        if verbose and rank == 0 and (it % max(1, len(ts)//10) == 0):
            print(f"[write] t={t}: wrote p_src & q_src for {len(p_terminals)} territories")

        # Also write VTX
        vtx_pres.write(float(t))
        vtx_flow.write(float(t))

    vtx_pres.close()
    vtx_flow.close()

    tdim = mesh_3d.topology.dim
    ncells = mesh_3d.topology.index_map(tdim).size_local
    cells_painted = np.count_nonzero(p_src_func.x.array > 0)
    print(f"painted {cells_painted}/{ncells} cells ({100*cells_painted/ncells:.1f}%) with p_src")

def terminal_outflow(
        seed_xyz: np.ndarray,
        seeds_df: pd.DataFrame,
        mesh_bp: str,    
        flow_bp: str,
        node_match_tol: float = 1e-6,
        verbose = True,
        comm=MPI.COMM_WORLD,
        
):
    rank = comm.rank
    seed_row_index = np.arange(len(seeds_df), dtype=np.int64)

    # Row index after reset_index in the loader is our canonical "seed position"
    # (0..N-1 order used during tessellation)
    seed_row_index = np.arange(len(seeds_df), dtype=np.int64)

    # Read mesh
    mesh_1d = adios4dolfinx.read_mesh(mesh_bp, comm=comm)

    # Read the last time point from the flow files
    u_last = read_last_step_bp_array(mesh_1d, flow_bp, function_name="f")

        # ---------------- 3) Map seeds -> nearest 1D node ----------------
    node_ids, found = map_seeds_to_1d_nodes(mesh_1d, seed_xyz, tol=node_match_tol)

    if not np.all(found) and rank == 0:
        missing = np.where(~found)[0]
        print(f"[warn] {missing.size} seeds did not match a 1D node within tol={node_match_tol}. Skipping those seeds.")

    valid_idx = np.where(found)[0]
    node_ids_valid = np.asarray(node_ids)[valid_idx]
    seed_rows_valid = seed_row_index[valid_idx]   # positions into the seed list

    if verbose and rank == 0:
        print(f"[map] matched {len(seed_rows_valid)}/{len(seeds_df)} seeds to 1D nodes.")

    print("Node ids valid: ", node_ids_valid)
    print("Seed rows valid: ", seed_rows_valid)
    print("Print terminal velocities: ", u_last[node_ids_valid])
    return u_last[node_ids_valid]

if __name__ == "__main__":
    # Optional CLI defaults; change these paths as needed
    import os
    csv_inlet_path   = os.environ.get("SEEDS_INLET_CSV", "/Users/rakshakonanur/Documents/Research/vascularize/output/Forest_Output/1D_Output/102925/Run11_10branches_0d_0d/branchingData_0.csv")
    csv_outlet_path  = os.environ.get("SEEDS_OUTLET_CSV", "/Users/rakshakonanur/Documents/Research/vascularize/output/Forest_Output/1D_Output/102925/Run11_10branches_0d_0d/branchingData_1.csv") 
    terr_inlet_xdmf  = os.environ.get("TERRITORIES_INLET_XDMF", "territories_inlet.xdmf")
    terr_outlet_xdmf = os.environ.get("TERRITORIES_OUTLET_XDMF", "territories_outlet.xdmf") 
    mesh_inlet_bp    = os.environ.get("MESH_INLET_BP", "../geometry/tagged_branches_inlet.bp")
    mesh_outlet_bp   = os.environ.get("MESH_OUTLET_BP", "../geometry/tagged_branches_outlet.bp")  
    p1d_inlet_bp     = os.environ.get("PRESSURE_INLET_BP", "../geometry/pressure_checkpoint_inlet.bp")
    p1d_outlet_bp    = os.environ.get("PRESSURE_OUTLET_BP", "../geometry/pressure_checkpoint_outlet.bp")  
    u1d_inlet_bp     = os.environ.get("FLOW_INLET_BP",     "../geometry/flow_checkpoint_inlet.bp")
    u1d_outlet_bp    = os.environ.get("FLOW_OUTLET_BP",    "../geometry/flow_checkpoint_outlet.bp")
    viz_inlet_pressure_bp   = os.environ.get("VIZ_INLET_BP", "pressure_inlet.bp")
    viz_outlet_pressure_bp  = os.environ.get("VIZ_OUTLET_BP", "pressure_outlet.bp")
    viz_inlet_flow_bp       = os.environ.get("VIZ_INLET_FLOW_BP", "flow_inlet.bp")
    viz_outlet_flow_bp      = os.environ.get("VIZ_OUTLET_FLOW_BP", "flow_outlet.bp")
    out_p_inlet_bp   = os.environ.get("OUT_P_INLET_SRC_BP", "p_src_inlet_series.bp")
    out_q_inlet_bp   = os.environ.get("OUT_Q_INLET_SRC_BP", "q_src_inlet_series.bp")
    out_p_outlet_bp  = os.environ.get("OUT_P_OUTLET_SRC_BP", "p_src_outlet_series.bp")
    out_q_outlet_bp  = os.environ.get("OUT_Q_OUTLET_SRC_BP", "q_src_outlet_series.bp")
    tol        = float(os.environ.get("NODE_TOL", "1e-6"))
    xdmf_file  = os.environ.get("MESH_TAGS_XDMF", "../geometry/bioreactor.xdmf")

    # 1) load terminal seeds directly from CSV
    seeds_df_inlet, seeds_xyz_inlet, diameters_inlet =  load_terminal_seeds_from_csv(csv_inlet_path)
    seeds_df_outlet, seeds_xyz_outlet, diameters_outlet =  load_terminal_seeds_from_csv(csv_outlet_path)

    # (Optional) 2a) snap to outlet boundary facets ONLY (recommended if you have outlet tags)
    # Example: assume outlets have tag value OUTLET (=1), or a list like [101,102,...]
    # snapped_xyz, nearest_facets = snap_seeds_to_boundary_facets(self.mesh, seeds_xyz,
    #                                                             facet_tags=self.mesh_tags,
    #                                                             restrict_to_values=[OUTLET])

    # (Optional) 2b) OR snap to nearest cell if your seeds should live in the myocardium interior
    # snapped_xyz, nearest_cells = snap_seeds_to_cells(self.mesh, seeds_xyz)

    # Choose which set to use downstream (Laguerre, sources, etc.)
    inlet_seed_points_for_laguerre = seeds_xyz_inlet          # or snapped_xyz if you snapped
    inlet_seed_diameters = diameters_inlet
    inlet_seed_ids       = seeds_df_inlet['seed_id'].to_numpy()

    outlet_seed_points_for_laguerre = seeds_xyz_outlet    # or snapped_xyz if you snapped
    outlet_seed_diameters = diameters_outlet
    outlet_seed_ids       = seeds_df_outlet['seed_id'].to_numpy()

    # 3) Now do your Laguerre/territory or outlet mapping with seed_points_for_laguerre + seed_diameters
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    labels_local, uniq = label_cells_by_laguerre(mesh, inlet_seed_points_for_laguerre, inlet_seed_diameters,
                                                 weight_mode="d2", kappa=1.0)
    cell_tags = make_cell_meshtags(mesh, labels_local)
    # (optional) write territories:
    write_territories(mesh, cell_tags, terr_inlet_xdmf)

    labels_local, uniq = label_cells_by_laguerre(mesh, outlet_seed_points_for_laguerre, outlet_seed_diameters,
                                                 weight_mode="d2", kappa=1.0)
    cell_tags = make_cell_meshtags(mesh, labels_local)
    # (optional) write territories:
    write_territories(mesh, cell_tags, terr_outlet_xdmf)

    # If territories already exist, just write sources; otherwise, build them earlier in your pipeline.
    if adios4dolfinx is None:
        print("[error] adios4dolfinx not installed; cannot write .bp", flush=True)
    else:
        try:
            assign_and_write_sources(csv_inlet_path, terr_inlet_xdmf, mesh_inlet_bp, p1d_inlet_bp, u1d_inlet_bp,
                                     out_bp_p_src=out_p_inlet_bp, out_bp_q_src=out_q_inlet_bp, viz_pressure_bp=viz_inlet_pressure_bp,
                                        viz_flow_bp=viz_inlet_flow_bp,
                                     node_match_tol=tol, verbose=True)
            assign_and_write_sources(csv_outlet_path, terr_outlet_xdmf, mesh_outlet_bp, p1d_outlet_bp, u1d_outlet_bp,
                                     out_bp_p_src=out_p_outlet_bp, out_bp_q_src=out_q_outlet_bp, viz_pressure_bp=viz_outlet_pressure_bp,
                                        viz_flow_bp=viz_outlet_flow_bp,
                                     node_match_tol=tol, verbose=True)
        except FileNotFoundError as e:
            print(f"[warn] {e}. Ensure your file paths are correct.", flush=True)
