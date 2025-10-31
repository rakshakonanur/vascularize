#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, csv, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import adios4dolfinx
from mpi4py import MPI
from dolfinx import (io, fem, geometry)
from dolfinx import mesh as dmesh
import ufl
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
from pathlib import Path
import shutil

# Optional deps
try:
    import adios2
except Exception as _e_ad2:
    adios2 = None
    _adios_err = _e_ad2

try:
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
except Exception:
    MPI = None
    XDMFFile = None

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None


# ---------- utils ----------
def _die(msg: str, code: int = 2):
    print(f"[0D:error] {msg}", file=sys.stderr)
    sys.exit(code)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _run(cmd: list[str], cwd: Path):
    print("[0D:run]", " ".join(map(str, cmd)), f"(cwd={cwd})")
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        _die(f"Command failed: {cmd} (code {res.returncode})")

def _copytree(src: Path, dst: Path):
    import shutil
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def _read_last_step_bp_array(mesh, bp_dir: Path, varname: str) -> np.ndarray:
    if adios4dolfinx is None:
        raise RuntimeError("adios4dolfinx not available.")
    comm = MPI.COMM_WORLD
    # mesh = adios4dolfinx.read_mesh(bp_path, comm=comm)
    Q0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    f = fem.Function(Q0, name="p_src")
    # detect function name and timestamps
    ts = adios4dolfinx.read_timestamps(bp_dir, comm=comm, function_name=varname)
    t_last = ts[-1]
    adios4dolfinx.read_function(bp_dir, f, time=t_last, name=varname)
    return f.x.array

def _read_mesh_and_tags(territories_xdmf: Path, name: str = "mesh_tags"):
    if XDMFFile is None:
        _die("dolfinx not available to read XDMF")
    with XDMFFile(MPI.COMM_WORLD, str(territories_xdmf), "r") as xdmf:
        mesh = xdmf.read_mesh(name = "Grid")
        tags = xdmf.read_meshtags(mesh, name= name)
    return mesh, tags

def _cell_centroids(mesh) -> np.ndarray:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    x = mesh.geometry.x
    nloc = mesh.topology.index_map(tdim).size_local
    cent = np.empty((nloc, x.shape[1]), dtype=float)
    for c in range(nloc):
        vs = conn.links(c)
        cent[c] = x[vs].mean(axis=0)
    return cent

def _nearest_tags_for_points(points: np.ndarray, centroids: np.ndarray, cell_tags) -> List[int]:
    c2t = {int(ci): int(cv) for ci, cv in zip(cell_tags.indices, cell_tags.values)}
    if KDTree is not None:
        tree = KDTree(centroids)
        _, idx = tree.query(points, k=1)
        idx = np.atleast_1d(idx)
    else:
        idx = np.array([np.linalg.norm(centroids - p, axis=1).argmin() for p in points], dtype=int)
    return [c2t[int(c)] for c in idx]


# ---------- public API ----------
@dataclass
class Paths:
    root: Path
    geometry: Path
    voronoi: Path
    solves: Path
    coupled: Path


def _find_outlet_folder(old_base: Path) -> Path:
    """Try common locations for solver_0d_new.in under old_base; fallback to a quick scan."""
    candidates = [
        old_base / "folder",
        old_base / "outlet",
        old_base / "inlet",
        old_base,  # sometimes the file is right in the parent
    ]
    for c in candidates:
        if (c / "solver_0d_new.in").exists():
            return c
    # quick shallow scan
    for p in old_base.glob("**/solver_0d_new.in"):
        return p.parent
    _die(f"Could not find solver_0d_new.in under {old_base}")


def prepare_outlet_run0(paths: Paths, old_inlet: Path, old_outlet: Path):
    """
    Seed outlet pipeline from the folder next to --old-1d-inlet:
      - copy <old_base>/[folder|outlet|…] -> coupled/run_0/outlet
      - copy <old_base>/branchingData_1.csv -> coupled/run_0/branchingData_1.csv
    """
    src_folder = old_inlet.resolve()
    dst = paths.coupled / "run_0" / "inlet"
    _ensure_dir(dst.parent)
    _copytree(src_folder, dst)
    if old_outlet:
        snk_folder = old_outlet.resolve()
        dst_snk = paths.coupled / "run_0" / "outlet"
        _ensure_dir(dst_snk.parent)
        _copytree(snk_folder, dst_snk)
    csv_src = src_folder.parent.parent / "branchingData_0.csv"
    print(f"[0D] Found branchingData_0.csv at {csv_src}")
    if not csv_src.exists():
        _die(f"Missing branchingData_0.csv at {csv_src}")
    csv_snk = src_folder.parent.parent / "branchingData_1.csv"
    if not csv_snk.exists():
        _die(f"Missing branchingData_1.csv at {csv_snk}")
    (paths.coupled / "run_0" / "branchingData_0.csv").write_text(csv_src.read_text())
    (paths.coupled / "run_0" / "branchingData_1.csv").write_text(csv_snk.read_text())
    print(f"[0D] Prepared outlet run_0: {dst} and branchingData_1.csv")


def outlet_stage_iter(
    paths: Paths,
    *,
    run_i: int,
    run_next: int,
    assign_script: Path | None = None,
    svzerodsolver: str = "svzerodsolver",
    one_d_solver_cmd: str | None = None,
    node_scale: float = 1.0,
    mbf_field: str = "mbf_qTi_tagconst",
    inlet: bool = False,
):
    """
    One outlet iteration:
      1) Read territory-constant flows from solves/mbf_outlet.bp and voronoi/territories_outlet.xdmf.
      2) Read terminals from coupled/run_0/branchingData_1.csv.
      3) Nearest territory per distal coordinate.
      4) Update run_{run_i}/outlet/solver_0d_new.in → run_{run_next}/outlet/solver_0d_new.in (OUTk FLOW BCs).
      5) Run 0D solver in run_{run_next}/outlet/.
      6) Run assign_pressure_bcs.py (if provided) to push pressures to 1D.
      7) Optionally run outlet 1D with updated 1D card → run_{run_next}/outlet_1d/.
    """
    if not inlet:
        branching_csv = "branchingData_1.csv"
        mbf_file = "mbf_outlet.bp"
        territories_file = "territories_outlet.xdmf"
        folder = "outlet"
    else:
        branching_csv = "branchingData_0.csv"
        mbf_file = "mbf_inlet.bp"
        territories_file = "territories_inlet.xdmf"
        folder = "inlet"
    # sources
    mbf_bp   = paths.solves/"out_darcy"  / mbf_file
    terr_xdmf= paths.voronoi / territories_file
    if not mbf_bp.exists(): _die(f"Missing {mbf_bp}")
    if not terr_xdmf.exists(): _die(f"Missing {terr_xdmf}")

    mesh, tags = _read_mesh_and_tags(terr_xdmf)
    cell_tags = np.array(tags.values, dtype=int)
    uniq_tags = np.unique(cell_tags)

    arr = _read_last_step_bp_array(mesh, mbf_bp, mbf_field)
    if arr.size == cell_tags.size:  # per-cell → average per tag
        flows_by_tag = {int(t): float(np.mean(arr[cell_tags == t])) for t in uniq_tags}
    elif arr.size == uniq_tags.size:  # already per-tag (assumed ascending tags)
        sorted_tags = np.sort(uniq_tags)
        flows_by_tag = {int(t): float(v) for t, v in zip(sorted_tags, arr)}
    else:
        _die(f"Size mismatch: {mbf_bp} var {mbf_field} has {arr.size} values; mesh has {cell_tags.size} cells and {uniq_tags.size} tags")

    term_csv = paths.coupled / "run_0" / branching_csv
    if not term_csv.exists(): _die(f"Missing terminals CSV: {term_csv}")
    terms: List[Tuple[int, np.ndarray]] = []
    with open(term_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            c1 = (row.get("Child1") or "").strip().upper()
            c2 = (row.get("Child2") or "").strip().upper()
            if c1 in ("", "NONE", "-1") and c2 in ("", "NONE", "-1"):
                # Prefer explicit BranchID headers; otherwise use the FIRST COLUMN
                bid_str = (row.get("BranchID") or row.get("branchID") or row.get("branchid"))
                if not bid_str:
                    # fallback: first column by position
                    if r.fieldnames and len(r.fieldnames) > 0:
                        bid_str = row[r.fieldnames[0]]
                    else:
                        # last resort: first value in the row dict
                        bid_str = next(iter(row.values()))
                bid = int(str(bid_str).strip())
                dx = float(row.get("distalCoordsX")); dy = float(row.get("distalCoordsY")); dz = float(row.get("distalCoordsZ"))
                terms.append((bid, node_scale*np.array([dx,dy,dz], float)))
    if not terms: _die(f"No terminal rows in {branching_csv}")

    term_xyz = np.vstack([xyz for _, xyz in terms])
    centroids = _cell_centroids(mesh)
    term_tags = _nearest_tags_for_points(term_xyz, centroids, tags)

    # update 0D card (k → k+1)
    src_card = paths.coupled / f"run_{run_i}"   /  folder /"solver_0d_new.in"
    dst_dir  = paths.coupled / f"run_{run_next}" / folder
    _ensure_dir(dst_dir)
    dst_card = dst_dir / "solver_0d_new.in"
    if not src_card.exists(): _die(f"Missing 0D card: {src_card}")

    data = json.loads(src_card.read_text())
    bcs = data.get("boundary_conditions", [])
    # build mapping branchID -> flow via territory
    branch_flow: Dict[int, float] = {}
    for (bid, _), tag in zip(terms, term_tags):
        t = int(tag)
        if t not in flows_by_tag:
            _die(f"Missing flow for territory tag {t}")
        branch_flow[int(bid)] = float(flows_by_tag[t])

    print(f"[0D] Flow rates: {branch_flow}")

    found = set()
    for bc in bcs:
        if bc.get("bc_type") != "FLOW": continue
        name = bc.get("bc_name", "")
        if not name.startswith("OUT"): continue
        try:
            bid = int(name[3:])
        except Exception:
            continue
        if bid in branch_flow:
            q = branch_flow[bid]
            bc.setdefault("bc_values", {})
            bc["bc_values"]["Q"] = [q, q]
            bc["bc_values"]["t"] = [0.0, 1.0]
            found.add(bid)

    missing = sorted(set([bid for bid, _ in terms]) - found)
    if missing:
        print(f"[0D] Warning: OUTk BCs not found for branchIDs {missing}")

    dst_card.write_text(json.dumps(data, indent=2))
    print(f"[0D] wrote updated 0D card: {dst_card}")

    # run 0D
    out_csv = dst_dir / "output.csv"
    _run([svzerodsolver, str(dst_card), str(out_csv)], cwd=dst_dir)
    print(f"[0D] output: {out_csv}")

    # # assign pressures back to 1D
    # if assign_script is not None and assign_script.exists():
    #     _run([sys.executable, str(assign_script)], cwd=assign_script.parent)
    #     print("[0D] assign_pressure_bcs.py done.")
    # else:
    #     print("[0D] assign script not provided or missing — skipping.")

def run_updated_1d_solver(paths: Paths, run_i: int, run_next: int):
    from assign_pressure_bcs import main
    (paths.coupled / f"run_{run_next}/outlet").mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.coupled / f"run_{run_i}/outlet/1d_simulation_input.json", 
                 paths.coupled / f"run_{run_next}/outlet/1d_simulation_input.json")


    sys.argv = [
        "assign_outlet_pressures.py",
        "--deck", str(paths.coupled / f"run_{run_next}/outlet/1d_simulation_input.json"),
        "--output", str(paths.coupled / f"run_{run_next}/outlet/output.csv"),
        "--branching", str(paths.coupled / "run_0/branchingData_1.csv"),
    ]
    main()


