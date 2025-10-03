#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update RCR resistances from DG0 per-territory fields.

Steps:
1) Load mesh + cell tags from territories.xdmf
2) Load DG0 outlet pressures per territory from p_src_series.bp (choose last time) via adios4dolfinx
3) Load DG0 integrated flows per territory from mbf.xdmf field 'mbf_qTi_tagconst'
4) Compute new resistance per territory: R = p_src / qTi (guard divide-by-zero)
5) Open the 1D card file (text format), update each DATATABLE RCR_k LIST first numeric after '0.0' with new R
6) Save to a new file

Usage:
python update_rcr_from_mbf_p.py \
  --territories-xdmf territories.xdmf \
  --p-src-bp p_src_series.bp \
  --mbf-xdmf mbf.xdmf \
  --card-in 1d_simulation_input.json \
  --card-out 1d_simulation_input_updated.json
"""
import re
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
from dolfinx import (io, fem, geometry)
from dolfinx import mesh as dmesh
import ufl
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
from pathlib import Path

# ADIOS
try:
    import adios4dolfinx
except Exception as e:
    adios4dolfinx = None

def read_mesh_and_cell_tags(xdmf_path: str, tags_name: str | None = None):
    comm = MPI.COMM_WORLD
    with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh) if tags_name is None else xdmf.read_meshtags(mesh, name=tags_name)
    return mesh, cell_tags

def read_DG0_field_from_bp(bp_path: str, field_name: str, mesh):
    comm = MPI.COMM_WORLD
    # mesh = adios4dolfinx.read_mesh(bp_path, comm=comm)
    Q0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    f = fem.Function(Q0, name=field_name)
    adios4dolfinx.read_function(bp_path, f, time=0.0, name=field_name)
    return mesh, f

def per_tag_constant_from_DG0(mesh, cell_tags, field_DG0: fem.Function) -> dict[int, float]:
    """Return {tag: mean value on that tag}. For truly piecewise-constant fields the mean is exact."""
    vals = {}
    arr = field_DG0.x.array
    tags = np.unique(cell_tags.values).astype(int)
    for tag in tags:
        mask = (cell_tags.values == int(tag))
        cells = cell_tags.indices[mask]
        if cells.size:
            vals[int(tag)] = float(arr[cells].mean())
    return vals

def read_last_p_src_from_bp(bp_path: str, mesh):
    """Read last timestep p_src (DG0) from ADIOS .bp."""
    if adios4dolfinx is None:
        raise RuntimeError("adios4dolfinx not available.")
    comm = MPI.COMM_WORLD
    # mesh = adios4dolfinx.read_mesh(bp_path, comm=comm)
    Q0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    f = fem.Function(Q0, name="p_src")
    # detect function name and timestamps
    ts = adios4dolfinx.read_timestamps(bp_path, comm=comm, function_name="p_src")
    t_last = ts[-1]
    adios4dolfinx.read_function(bp_path, f, time=t_last, name="p_src")
    return mesh, f, t_last, "p_src"

def update_card_resistances(card_in: str, card_out: str, new_R_by_index: list[float]):
    """
    Update each DATATABLE RCR_k LIST block:
      Replace the first value after 0.0 on the FIRST line with new_R_by_index[k].
    Returns the updated text.
    """
    text = open(card_in, "r", encoding="utf-8").read()

    def repl_block(match):
        name = match.group(1)  # e.g., RCR_0
        idx = int(name.split('_')[1])
        body = match.group(2)
        lines = body.splitlines()
        if not lines:
            return match.group(0)
        # Find first numeric line; replace second column
        parts = lines[0].split()
        if len(parts) >= 2:
            # Keep the first column (time, typically 0.0), replace second with new resistance
            t0 = parts[0]
            Rnew = new_R_by_index[idx] if idx < len(new_R_by_index) else parts[1]
            lines[0] = f"{t0} {Rnew}"
        new_body = "\n".join(lines)
        return f"DATATABLE {name} LIST\n{new_body}\nENDDATATABLE"

    pattern = re.compile(r"DATATABLE\s+(RCR_\d+)\s+LIST\s*\n(.*?)\nENDDATATABLE", re.S)
    updated = pattern.sub(repl_block, text)

    with open(card_out, "w", encoding="utf-8") as f:
        f.write(updated)
    return updated

def _cell_volumes_Q0(mesh) -> np.ndarray:
    """
    Return a NumPy array of cell volumes (length = # locally-owned cells).
    Uses the DG0 mass-vector trick: assemble v*dx, where v is the DG0 test function.
    """
    Q0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    v = ufl.TestFunction(Q0)
    b = petsc_assemble_vector(fem.form(v * ufl.dx(domain=mesh)))
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

def per_tag_volume(mesh, cell_tags) -> dict[int, float]:
    """Return {tag: total volume of that tag} (MPI-safe)."""
    comm = mesh.comm
    vols_cell = _cell_volumes_Q0(mesh)             # local cell volumes
    labels = _labels_from_cell_tags(mesh, cell_tags)
    uniq_local = np.unique(cell_tags.values.astype(int))
    local = {int(tag): float(vols_cell[labels == int(tag)].sum()) for tag in uniq_local}
    # reduce
    uniq_global = np.unique(cell_tags.values.astype(int))
    out = {}
    for tag in uniq_global:
        out[int(tag)] = comm.allreduce(local.get(int(tag), 0.0), op=MPI.SUM)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--territories-xdmf", default="voronoi/territories.xdmf", help="territories.xdmf with cell tags")
    ap.add_argument("--territory-tags-name", default="mesh_tags", help="MeshTags name (default first set)")
    ap.add_argument("--p-src-bp", default="voronoi/p_src_series.bp", help="p_src_series.bp (DG0 time series)")
    ap.add_argument("--mbf-bp", default="solves/out_mixed_poisson/mbf.bp", help="mbf.bp containing DG0 'mbf_qTi_tagconst'")
    ap.add_argument("--mbf-field-name", default="mbf_qTi_tagconst")
    ap.add_argument("--card-in", default="../output/Forest_Output/1D_Output/091725/Run9_10branches/1D_Input_Files/inlet/1d_simulation_input.json", help="1D card to update (text)")
    ap.add_argument("--card-out", default="1d_simulation_input_updated.json", help="Output card filename")
    ap.add_argument("--epsilon", type=float, default=1e-30, help="Small value to avoid division by zero")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # 1) Territories
    mesh_T, tags_T = read_mesh_and_cell_tags(args.territories_xdmf, args.territory_tags_name)
    tags_sorted = np.sort(np.unique(tags_T.values).astype(int))

    # 2) p_src (last time) from ADIOS
    mesh_P, p_src_fn, t_last, pname = read_last_p_src_from_bp(args.p_src_bp, mesh_T)
    if mesh_P is not mesh_T:
        Q0 = fem.functionspace(mesh_T, element("DG", mesh_T.basix_cell(), 0))
        p_src_on_T = fem.Function(Q0, name=p_src_fn.name)
        p_src_on_T.interpolate(p_src_fn)
        p_src_fn = p_src_on_T
    p_by_tag = per_tag_constant_from_DG0(mesh_T, tags_T, p_src_fn)

    # 3) qTi from ADIOS
    mesh_M, mbf_fn = read_DG0_field_from_bp(args.mbf_bp, args.mbf_field_name, mesh_T)
    if mesh_M is not mesh_T:
        Q0 = fem.functionspace(mesh_T, element("DG", mesh_T.basix_cell(), 0))
        mbf_on_T = fem.Function(Q0, name=mbf_fn.name)
        mbf_on_T.interpolate(mbf_fn)
        mbf_fn = mbf_on_T
    qTi_by_tag = per_tag_constant_from_DG0(mesh_T, tags_T, mbf_fn)

    # # Divide qTi by territory volume to get integrated_flow/volume units
    # vols_by_tag = per_tag_volume(mesh_T, tags_T)
    # for tag in list(qTi_by_tag.keys()):
    #     Vt = vols_by_tag.get(int(tag), 0.0)
    #     qTi_by_tag[int(tag)] = (qTi_by_tag[int(tag)] / Vt) if Vt > 0.0 else 0.0
    # if rank == 0:
    #     print("[info] Divided qTi by territory volumes (units now flow/volume).")

    # 4) Resistances per tag
    eps = args.epsilon
    R_by_tag = {}
    for tag in tags_sorted:
        p = float(p_by_tag.get(int(tag), 0.0))
        q = float(qTi_by_tag.get(int(tag), 0.0))
        R = p / (q + eps)
        R_by_tag[int(tag)] = R

    # 5) Map sorted tags -> RCR_k order and build list aligned to indices
    card_text = open(args.card_in, "r", encoding="utf-8").read()
    rcr_blocks = re.findall(r"DATATABLE\s+(RCR_\d+)\s+LIST\s*\n(.*?)\nENDDATATABLE", card_text, flags=re.S)
    rcr_names = [name for name,_ in rcr_blocks]
    n = min(len(tags_sorted), len(rcr_names))
    if rank == 0 and len(tags_sorted) != len(rcr_names):
        print(f"[warn] #territories ({len(tags_sorted)}) != #RCR blocks ({len(rcr_names)}). Mapping first {n} by order.")

    new_R_by_index = []
    for k in range(len(rcr_names)):
        if k < n:
            tag = int(tags_sorted[k])
            new_R_by_index.append(R_by_tag.get(tag, 0.0))
        else:
            new_R_by_index.append(None)

    # Save to file with one resistance per line
    if rank == 0:
        with open("new_R_by_index.txt", "w", encoding="utf-8") as f:
            for R in new_R_by_index:
                f.write(f"{R}\n")
        print(f"[info] Wrote new_R_by_index.txt with {len(new_R_by_index)} resistances.")

    from reorder_resistance import reorder_resistances_by_rcr  
    out = reorder_resistances_by_rcr(Path(args.card_in), Path(args.territories_xdmf), new_R_by_index)
    new_R = out["R_reordered"]
    print("old RCRs:")
    for k, v in enumerate(new_R_by_index):
        print(f"RCR_{k}: {v}")
    print("new RCRs:")
    for k, v in enumerate(new_R):
        print(f"RCR_{k}: {v}")
    # 6) Update card & write CSV summary
    update_card_resistances(args.card_in, args.card_out, new_R)

    if rank == 0:
        rows = [{"tag": int(tag), "R": R_by_tag[int(tag)],
                 "p_src": p_by_tag.get(int(tag), 0.0),
                 "qTi_adj": qTi_by_tag.get(int(tag), 0.0)} for tag in tags_sorted]
        pd.DataFrame(rows).to_csv("resistances_by_tag.csv", index=False, mode='a')
        print(f"[done] Updated '{args.card_out}'. Last p_src time = {t_last}. "
              f"Summary -> resistances_by_tag.csv")


if __name__ == "__main__":
    main()
