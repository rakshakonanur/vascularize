# ---------- CSV -> terminal seeds (+ optional snapping) ----------
from __future__ import annotations
import numpy as np
import pandas as pd

from dolfinx import mesh as dmesh, fem
import dolfinx as dfx
from dolfinx.io import XDMFFile
import ufl

# Try SciPy for KDTree; fall back to a tiny pure-numpy version if unavailable
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:  # very small fallback (O(N*M), fine for N~100 seeds)
    KDTree = None

import adios4dolfinx

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
    seed_diameters: np.ndarray,# shape (N,)
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
    w = laguerre_weights_from_diameters(np.asarray(seed_diameters, float), mode=weight_mode, kappa=kappa)

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

if __name__ == "__main__":
    # 0) paths
    csv_path = "/Users/rakshakonanur/Documents/Research/vascularize/output/Forest_Output/1D_Output/091725/Run10_50branches/branchingData_0.csv"  # <-- set absolute/relative path as needed

    # 1) load terminal seeds directly from CSV
    seeds_df, seeds_xyz, diameters = load_terminal_seeds_from_csv(csv_path)

    # (Optional) 2a) snap to outlet boundary facets ONLY (recommended if you have outlet tags)
    # Example: assume outlets have tag value OUTLET (=1), or a list like [101,102,...]
    # snapped_xyz, nearest_facets = snap_seeds_to_boundary_facets(self.mesh, seeds_xyz,
    #                                                             facet_tags=self.mesh_tags,
    #                                                             restrict_to_values=[OUTLET])

    # (Optional) 2b) OR snap to nearest cell if your seeds should live in the myocardium interior
    # snapped_xyz, nearest_cells = snap_seeds_to_cells(self.mesh, seeds_xyz)

    # Choose which set to use downstream (Laguerre, sources, etc.)
    seed_points_for_laguerre = seeds_xyz          # or snapped_xyz if you snapped
    seed_diameters = diameters
    seed_ids       = seeds_df['seed_id'].to_numpy()

    # 3) Now do your Laguerre/territory or outlet mapping with seed_points_for_laguerre + seed_diameters
    xdmf_file = "../geometry/mesh_tags.xdmf"
    with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        mesh_tags = xdmf.read_meshtags(mesh, name="mesh_tags")
    labels_local, uniq = label_cells_by_laguerre(mesh, seed_points_for_laguerre, seed_diameters,
                                                 weight_mode="d2", kappa=1.0)
    cell_tags = make_cell_meshtags(mesh, labels_local)
    # (optional) write territories:
    write_territories(mesh, cell_tags, "territories.xdmf")

     # If territories already exist, just write sources; otherwise, build them earlier in your pipeline.
    if adios4dolfinx is None:
        print("[error] adios4dolfinx not installed; cannot write .bp", flush=True)
    else:
        try:
            assign_and_write_sources(csv_path, terr_xdmf, p1d_bp, u1d_bp,
                                     out_bp_p_src=out_p_bp, out_bp_q_src=out_q_bp,
                                     node_match_tol=tol, verbose=True)
        except FileNotFoundError as e:
            print(f"[warn] {e}. Ensure your file paths are correct.", flush=True)

