# #!/usr/bin/env python3
# from __future__ import annotations
# import re
# import argparse
# from pathlib import Path
# from typing import Dict, List, Tuple
# import numpy as np

# # Optional but recommended for speed
# try:
#     from scipy.spatial import cKDTree as KDTree
# except Exception:
#     KDTree = None

# # Dolfinx for mesh + tags I/O
# from dolfinx.io import XDMFFile
# from dolfinx import mesh as dmesh
# import dolfinx
# import ufl
# from mpi4py import MPI


# SEGMENT_RE = re.compile(r"^SEGMENT\s+(\S+)\s+(\d+)\s+([\deE\.\+\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
#                         r"([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+\S+\s+\S+\s+[\deE\.\+\-]+\s+\d+\s+\d+\s+"
#                         r"(NOBOUND|PRESSURE|AREA|FLOW|RESISTANCE|RESISTANCE_TIME|PRESSURE_WAVE|WAVE|RCR|CORONARY|IMPEDANCE|PULMONARY)\s+(\S+)\s*$")

# NODE_RE = re.compile(r"^NODE\s+(\d+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s*$")

# def parse_nodes_and_rcr_segments(card_path: Path) -> Tuple[Dict[int, np.ndarray], List[Tuple[int,int,int]]]:
#     """
#     Returns:
#       nodes: node_id -> xyz
#       rcr_segments: list of (rcr_k, segment_id, outlet_node)
#     """
#     nodes: Dict[int, np.ndarray] = {}
#     rcr_segments: List[Tuple[int,int,int]] = []

#     with card_path.open("r") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue

#             mN = NODE_RE.match(line)
#             if mN:
#                 nid = int(mN.group(1))
#                 x = float(mN.group(2))
#                 y = float(mN.group(3))
#                 z = float(mN.group(4))
#                 nodes[nid] = np.array([x, y, z], dtype=float)
#                 continue

#             mS = SEGMENT_RE.match(line)
#             if mS:
#                 # Fields we care about
#                 seg_name = mS.group(1)
#                 seg_id = int(mS.group(2))
#                 # length = float(mS.group(3))
#                 # nelem = int(mS.group(4))
#                 inlet_node = int(mS.group(5))
#                 outlet_node = int(mS.group(6))
#                 # areas etc. are groups 7-9
#                 bc_type = mS.group(10)
#                 bc_data = mS.group(11)

#                 if (bc_type == "RESISTANCE" and bc_data.startswith("RCR_")) or (bc_type == "RCR" and bc_data.startswith("RCR_")):
#                     # Extract k from RCR_k
#                     try:
#                         rcr_k = int(bc_data.split("_", 1)[1])
#                     except Exception as e:
#                         raise ValueError(f"Could not parse RCR index from '{bc_data}' in segment '{seg_name}'") from e
#                     rcr_segments.append((rcr_k, seg_id, outlet_node))

#     if not rcr_segments:
#         raise RuntimeError("No RESISTANCE RCR_k segments found in the card.")

#     # Sort by increasing RCR_k (RCR_0, RCR_1, ...)
#     rcr_segments.sort(key=lambda t: t[0])
#     return nodes, rcr_segments


# def _cell_centroids(mesh: dmesh.Mesh) -> np.ndarray:
#     """
#     Compute cell centroids by averaging vertex coordinates of each cell.
#     Works for simplices and tensor-product cells.
#     """
#     # Connectivity from cells to vertices
#     tdim = mesh.topology.dim
#     mesh.topology.create_connectivity(tdim, 0)
#     conn = mesh.topology.connectivity(tdim, 0)
#     x = mesh.geometry.x
#     cell_indices = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
#     centroids = np.empty((cell_indices.size, x.shape[1]), dtype=float)

#     for c in cell_indices:
#         vs = conn.links(c)
#         centroids[c] = x[vs].mean(axis=0)

#     return centroids


# def read_mesh_and_tags(territories_xdmf: Path):
#     """
#     Reads mesh and the first available cell MeshTags from territories.xdmf.
#     Returns: (mesh, cell_tags, centroids)
#     """
#     with XDMFFile(MPI.COMM_WORLD, str(territories_xdmf), "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")  # default first mesh
#         mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
#         tags = xdmf.read_meshtags(mesh, name="mesh_tags")

#     centroids = _cell_centroids(mesh)
#     return mesh, tags, centroids


# def nearest_tag_for_points(points: np.ndarray, centroids: np.ndarray, cell_tags) -> List[int]:
#     """For each point (terminal node xyz), find nearest cell by centroid and return that cell's tag value."""
#     if KDTree is not None:
#         tree = KDTree(centroids)
#         dists, idxs = tree.query(points, k=1)
#         idxs = np.atleast_1d(idxs)
#     else:
#         # Fallback O(NM)
#         idxs = []
#         for p in points:
#             idxs.append(np.linalg.norm(centroids - p, axis=1).argmin())
#         idxs = np.array(idxs, dtype=int)

#     # Build a mapping cell->tag
#     # cell_tags.values correspond to cell_tags.indices (global cell ids for local process)
#     # We assume serial here; if running in parallel, gather is required.
#     cell_to_tag: Dict[int, int] = {int(ci): int(cv) for ci, cv in zip(cell_tags.indices, cell_tags.values)}

#     tags_out: List[int] = []
#     for c in idxs:
#         tag = cell_to_tag.get(int(c))
#         if tag is None:
#             raise RuntimeError(f"No tag value for cell index {int(c)}. Check territories.xdmf consistency.")
#         tags_out.append(tag)
#     return tags_out


# def reorder_resistances_by_rcr(
#     card_path: Path,
#     territories_xdmf: Path,
#     new_R_by_index: List[float]
# ):
#     """
#     Main entry:
#       - Parses card
#       - Maps terminal nodes -> nearest cell tag
#       - Produces:
#           rcr_info: list of dicts with rcr_k, seg_id, outlet_node, node_xyz, tag
#           reorder_idx: permutation that sorts by increasing rcr_k
#           R_reordered: new_R_by_index permuted to match RCR_0..RCR_K order
#     """
#     nodes, rcr_segments = parse_nodes_and_rcr_segments(card_path)
#     print(f"Found {len(rcr_segments)} RCR segments in '{card_path}'.")

#     # Prepare terminal node positions in RCR_k order
#     term_nodes = [outlet for (_, _, outlet) in rcr_segments]
#     term_xyz = np.vstack([nodes[nid] for nid in term_nodes])
#     print(f"All terminal nodes found in card: {term_nodes}.")

#     # Mesh and tags
#     mesh, cell_tags, centroids = read_mesh_and_tags(territories_xdmf)

#     # Map terminals to nearest tag
#     term_tags = nearest_tag_for_points(term_xyz, centroids, cell_tags)

#     # rcr_k are already sorted (0..K)
#     rcr_k_list = [rk for (rk, _, _) in rcr_segments]
#     reorder_idx = np.argsort(rcr_k_list)  # identity if already sorted
#     R_reordered = [float(new_R_by_index[i]) for i in reorder_idx]

#     rcr_info = []
#     for i, (rk, seg_id, outlet) in enumerate(rcr_segments):
#         rcr_info.append({
#             "rcr_k": int(rk),
#             "segment_id": int(seg_id),
#             "outlet_node": int(outlet),
#             "node_xyz": term_xyz[i].tolist(),
#             "territory_tag": int(term_tags[i]),
#         })

#     return {
#         "rcr_info": rcr_info,
#         "reorder_idx": reorder_idx.tolist(),
#         "R_reordered": R_reordered
#     }


# def main():
#     ap = argparse.ArgumentParser(description="Reorder resistances by outlet coordinates and territory tags.")
#     ap.add_argument("--card", required=True, type=Path, help="Path to 1D model input card (your 1d_simulation_input.json).")
#     ap.add_argument("--territories-xdmf", required=True, type=Path, help="Path to territories.xdmf (mesh + cell MeshTags).")
#     ap.add_argument("--r-file", type=Path, default=None,
#                     help="Optional: path to a text file containing new_R_by_index, one value per line.")
#     ap.add_argument("--print-map", action="store_true", help="Print (rcr_k -> tag, node) mapping.")
#     args = ap.parse_args()

#     if args.r_file is None:
#         raise SystemExit("Please supply --r-file with one resistance per line (new_R_by_index).")

#     new_R = [float(s.strip()) for s in args.r_file.read_text().splitlines() if s.strip()]

#     out = reorder_resistances_by_rcr(args.card, args.territories_xdmf, new_R)


#     if args.print_map:
#         for rec in out["rcr_info"]:
#             print(f"RCR_{rec['rcr_k']:d}  seg={rec['segment_id']:d}  node={rec['outlet_node']:d}  "
#                   f"xyz={np.array(rec['node_xyz'])}  tag={rec['territory_tag']}")

#     print("\n# Reorder indices (apply to your new_R_by_index):")
#     print(out["reorder_idx"])
#     print("\n# Reordered resistances matching RCR_0..RCR_K:")
#     for r in out["R_reordered"]:
#         print(f"{r:.9g}")
    
#     return out["R_reordered"]


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Optional but recommended for speed
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# Dolfinx for mesh + tags I/O
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh
import dolfinx
import ufl
from mpi4py import MPI


SEGMENT_RE = re.compile(r"^SEGMENT\s+(\S+)\s+(\d+)\s+([\deE\.\+\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                        r"([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+\S+\s+\S+\s+[\deE\.\+\-]+\s+\d+\s+\d+\s+"
                        r"(NOBOUND|PRESSURE|AREA|FLOW|RESISTANCE|RESISTANCE_TIME|PRESSURE_WAVE|WAVE|RCR|CORONARY|IMPEDANCE|PULMONARY)\s+(\S+)\s*$")

NODE_RE = re.compile(r"^NODE\s+(\d+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s+([\deE\.\+\-]+)\s*$")

def parse_nodes_and_rcr_segments(card_path: Path) -> Tuple[Dict[int, np.ndarray], List[Tuple[int,int,int]]]:
    """
    Returns:
      nodes: node_id -> xyz
      rcr_segments: list of (rcr_k, segment_id, outlet_node)
    """
    nodes: Dict[int, np.ndarray] = {}
    rcr_segments: List[Tuple[int,int,int]] = []

    with card_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            mN = NODE_RE.match(line)
            if mN:
                nid = int(mN.group(1))
                x = float(mN.group(2))
                y = float(mN.group(3))
                z = float(mN.group(4))
                nodes[nid] = np.array([x, y, z], dtype=float)
                continue

            mS = SEGMENT_RE.match(line)
            if mS:
                # Fields we care about
                seg_name = mS.group(1)
                seg_id = int(mS.group(2))
                # length = float(mS.group(3))
                # nelem = int(mS.group(4))
                inlet_node = int(mS.group(5))
                outlet_node = int(mS.group(6))
                # areas etc. are groups 7-9
                bc_type = mS.group(10)
                bc_data = mS.group(11)

                if (bc_type == "RESISTANCE" and bc_data.startswith("RCR_")) or (bc_type == "RCR" and bc_data.startswith("RCR_")):
                    # Extract k from RCR_k
                    try:
                        rcr_k = int(bc_data.split("_", 1)[1])
                    except Exception as e:
                        raise ValueError(f"Could not parse RCR index from '{bc_data}' in segment '{seg_name}'") from e
                    rcr_segments.append((rcr_k, seg_id, outlet_node))

    if not rcr_segments:
        raise RuntimeError("No RESISTANCE RCR_k segments found in the card.")

    # Sort by increasing RCR_k (RCR_0, RCR_1, ...)
    rcr_segments.sort(key=lambda t: t[0])
    return nodes, rcr_segments


def _cell_centroids(mesh: dmesh.Mesh) -> np.ndarray:
    """
    Compute cell centroids by averaging vertex coordinates of each cell.
    Works for simplices and tensor-product cells.
    """
    # Connectivity from cells to vertices
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    x = mesh.geometry.x
    cell_indices = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    centroids = np.empty((cell_indices.size, x.shape[1]), dtype=float)

    for c in cell_indices:
        vs = conn.links(c)
        centroids[c] = x[vs].mean(axis=0)

    return centroids


def read_mesh_and_tags(territories_xdmf: Path):
    """
    Reads mesh and the first available cell MeshTags from territories.xdmf.
    Returns: (mesh, cell_tags, centroids)
    """
    with XDMFFile(MPI.COMM_WORLD, str(territories_xdmf), "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")  # default first mesh
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        tags = xdmf.read_meshtags(mesh, name="mesh_tags")

    centroids = _cell_centroids(mesh)
    return mesh, tags, centroids


def nearest_tag_for_points(points: np.ndarray, centroids: np.ndarray, cell_tags) -> List[int]:
    """For each point (terminal node xyz), find nearest cell by centroid and return that cell's tag value."""
    if KDTree is not None:
        tree = KDTree(centroids)
        dists, idxs = tree.query(points, k=1)
        idxs = np.atleast_1d(idxs)
    else:
        # Fallback O(NM)
        idxs = []
        for p in points:
            idxs.append(np.linalg.norm(centroids - p, axis=1).argmin())
        idxs = np.array(idxs, dtype=int)

    # Build a mapping cell->tag
    # cell_tags.values correspond to cell_tags.indices (global cell ids for local process)
    # We assume serial here; if running in parallel, gather is required.
    cell_to_tag: Dict[int, int] = {int(ci): int(cv) for ci, cv in zip(cell_tags.indices, cell_tags.values)}

    tags_out: List[int] = []
    for c in idxs:
        tag = cell_to_tag.get(int(c))
        if tag is None:
            raise RuntimeError(f"No tag value for cell index {int(c)}. Check territories.xdmf consistency.")
        tags_out.append(tag)
    return tags_out


def reorder_resistances_by_rcr(
    card_path: Path,
    territories_xdmf: Path,
    new_R_by_index: List[float]
):
    """
    Main entry:
      - Parses card
      - Maps terminal nodes -> nearest cell tag
      - Produces:
          rcr_info: list of dicts with rcr_k, seg_id, outlet_node, node_xyz, tag
          reorder_idx: permutation that sorts by increasing rcr_k
          R_reordered: new_R_by_index permuted to match RCR_0..RCR_K order
    """
    nodes, rcr_segments = parse_nodes_and_rcr_segments(card_path)

    # Prepare terminal node positions in RCR_k order
    term_nodes = [outlet for (_, _, outlet) in rcr_segments]
    term_xyz = np.vstack([nodes[nid] for nid in term_nodes])

    # Mesh and tags
    mesh, cell_tags, centroids = read_mesh_and_tags(territories_xdmf)

    # Map terminals to nearest tag
    term_tags = nearest_tag_for_points(term_xyz, centroids, cell_tags)

    # rcr_k are already sorted (0..K)
    rcr_k_list = [rk for (rk, _, _) in rcr_segments]
    # Build a map tag -> R. Prefer all mesh tags (sorted), fall back to unique terminal tags.
    all_mesh_tags = sorted(int(t) for t in np.unique(cell_tags.values))
    if len(new_R_by_index) == len(all_mesh_tags):
        tag_to_R = {tag: float(R) for tag, R in zip(all_mesh_tags, new_R_by_index)}
    else:
        term_unique = sorted(int(t) for t in np.unique(term_tags))
        if len(new_R_by_index) == len(term_unique):
            tag_to_R = {tag: float(R) for tag, R in zip(term_unique, new_R_by_index)}
        else:
            raise ValueError(
                f"Cannot map new_R_by_index (len={len(new_R_by_index)}) "
                f"to tag sets: mesh has {len(all_mesh_tags)} tags, terminals cover {len(term_unique)}."
            )
    R_reordered = [tag_to_R[int(t)] for t in term_tags]

    rcr_info = []
    for i, (rk, seg_id, outlet) in enumerate(rcr_segments):
        rcr_info.append({
            "rcr_k": int(rk),
            "segment_id": int(seg_id),
            "outlet_node": int(outlet),
            "node_xyz": term_xyz[i].tolist(),
            "territory_tag": int(term_tags[i]),
        })

    return {
        "rcr_info": rcr_info,
        "reorder_idx": list(range(len(rcr_segments))),
        "R_reordered": R_reordered
    }


def main():
    ap = argparse.ArgumentParser(description="Reorder resistances by outlet coordinates and territory tags.")
    ap.add_argument("--card", required=True, type=Path, help="Path to 1D model input card (your 1d_simulation_input.json).")
    ap.add_argument("--territories-xdmf", required=True, type=Path, help="Path to territories.xdmf (mesh + cell MeshTags).")
    ap.add_argument("--r-file", type=Path, default=None,
                    help="Optional: path to a text file containing new_R_by_index, one value per line.")
    ap.add_argument("--print-map", action="store_true", help="Print (rcr_k -> tag, node) mapping.")
    args = ap.parse_args()

    if args.r_file is None:
        raise SystemExit("Please supply --r-file with one resistance per line (new_R_by_index).")

    new_R = [float(s.strip()) for s in args.r_file.read_text().splitlines() if s.strip()]

    out = reorder_resistances_by_rcr(args.card, args.territories_xdmf, new_R)


    if args.print_map:
        for rec in out["rcr_info"]:
            print(f"RCR_{rec['rcr_k']:d}  seg={rec['segment_id']:d}  node={rec['outlet_node']:d}  "
                  f"xyz={np.array(rec['node_xyz'])}  tag={rec['territory_tag']}")

    print("\n# Reorder indices (apply to your new_R_by_index):")
    print(out["reorder_idx"])
    print("\n# Reordered resistances matching RCR_0..RCR_K:")
    for r in out["R_reordered"]:
        print(f"{r:.9g}")
    
    return out["R_reordered"]


if __name__ == "__main__":
    main()

