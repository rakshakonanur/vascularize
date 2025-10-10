#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling pipeline for 1D Navier–Stokes (branched network) <-> single-compartment Darcy.

This script orchestrates the following loop, running from the project src/ directory:
  0) Prepare run_0 by copying an existing 1D inlet folder into ./coupled/run_0/inlet
     and copying branchingData_0.csv into ./coupled/run_0/.
  1) Build/refresh geometry (imports 1D network + tags) via geometry.mesh.Files
  2) Tesselate tissue territories and write DG0 source series via voronoi/tesselate_updated.py
  3) Run the Darcy single-compartment solver (init=True on the first iteration, thereafter False)
  4) Compare q_k (from voronoi/q_src_series.bp if available, else p_src_series.bp) vs
     q_{k+1} (from solves/out_mixed_poisson/mbf.bp field 'mbf_qTi_tagconst').
     If max_i |q_k - q_{k+1}| / max(|q_k|, epsilon) > tol, update 1D resistances and repeat.
     Otherwise stop and write a convergence plot + CSV.

Notes:
- This script avoids editing your source files in-place. It imports the relevant modules directly.
- It assumes your current working directory is the project src/ folder that contains `geometry/`, `voronoi/`, and `solves/`.
- You must provide the path to your external 1D solver (binary or Python entrypoint) with --one-d-solver-cmd.
"""

from __future__ import annotations
import argparse, os, sys, shutil, subprocess, time
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import matplotlib.pyplot as plt
import adios4dolfinx
from mpi4py import MPI
from dolfinx import (fem)
from basix.ufl import element
from dolfinx.io import XDMFFile, VTKFile

def _die(msg: str, code: int = 2):
    print(f"[error] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def _run(cmd: list[str] | str, cwd: Path, env: dict | None = None):
    if isinstance(cmd, list):
        cmd_str = " ".join([str(c) for c in cmd])
    else:
        cmd_str = cmd
    print(f"[run] {cmd_str}  (cwd={cwd})", flush=True)
    res = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    if res.returncode != 0:
        _die(f"Command failed with code {res.returncode}: {cmd_str}")

def _copy_tree(src: Path, dst: Path):
    if not src.exists():
        _die(f"Source path does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_module_from_path(name: str, file_path: Path):
    """Import a module from an arbitrary file path (even with dashes in name)."""
    import importlib.util, importlib.machinery
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        _die(f"Could not create import spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _read_last_step_bp_array(mesh, bp_path: Path, function_name: str) -> np.ndarray:
    """
    Read the last-step array from a .bp (ADIOS2) file by picking the first non-scalar var.
    This is a heuristic but works for typical DG0 time series written with adios4dolfinx.

    Returns a 1D numpy array.
    """

    ts = adios4dolfinx.read_timestamps(filename = bp_path, comm=MPI.COMM_WORLD, function_name=function_name)
    DG0 = fem.functionspace(mesh, element("DG", mesh.basix_cell(), 0))
    q_src = fem.Function(DG0)
    q_series = []
    for t in ts:
        adios4dolfinx.read_function(bp_path, q_src, name=function_name, time=t)
        q_src.x.array[:] = q_src.x.array       
        q_series.append(q_src.copy())

    print(f"Imported {len(q_series)} time steps from {bp_path}", flush=True)
    return q_series[-1].x.array.copy()

def _summarize_rel_err(qk: np.ndarray, qkp1: np.ndarray, eps: float) -> dict:
    if qk.shape != qkp1.shape:
        n = min(qk.size, qkp1.size)
        qk, qkp1 = qk[:n], qkp1[:n]

    # Denominator is qk (not |qk|). Guard zeros with +eps, preserve sign,
    # then take absolute value of the ratio to keep the metric non-negative.
    denom = np.where(qk > eps, qk, eps)
    rel   = np.abs(qk - qkp1) / denom

    return {
        "max": float(np.max(rel)),
        "mean": float(np.mean(rel)),
        "median": float(np.median(rel)),
        "p95": float(np.percentile(rel, 95.0)),
        "count_exceed": int(np.sum(rel > 0.0)),
        "rel": rel,
    }

@dataclass
class Paths:
    root: Path                 # src/ (this script's directory)
    geometry: Path             # src/geometry
    voronoi: Path              # src/voronoi
    solves: Path               # src/solves
    coupled: Path              # src/coupled

def discover_paths(start: Path) -> Paths:
    root = start.resolve()
    return Paths(
        root=root,
        geometry=root / "geometry",
        voronoi=root / "voronoi",
        solves=root / "solves",
        coupled=root / "coupled",
    )

def prepare_run0(paths: Paths, old_inlet: Path):
    """
    Create ./coupled/run_0 and copy the existing 1D inlet folder there.
    Also copy branchingData_0.csv into ./coupled/run_0/.
    Additionally, create ./coupled/original_results as a backup mirror of the provided inlet.
    """
    run0 = paths.coupled / "run_0"
    inlet_dst = run0 / "inlet"
    _ensure_dir(paths.coupled)
    _ensure_dir(run0)
    # Copy inlet folder
    _copy_tree(old_inlet, inlet_dst)
    # Try to copy branchingData_0.csv next to run_0
    src_csv = old_inlet.parent.parent / "branchingData_0.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, run0 / "branchingData_0.csv")
    else:
        print(f"[warn] Did not find {src_csv}. Make sure branchingData_0.csv is present.", flush=True)
    # Original results backup
    backup = paths.coupled / "original_results"
    if backup.exists():
        shutil.rmtree(backup)
    _copy_tree(old_inlet, backup)
    print(f"[ok] Prepared {run0}.", flush=True)

def run_geometry(paths: Paths, stl_file: Path, run_i: int):
    """
    Import branched mesh + tag bioreactor using Files() from geometry/mesh.py.
    """
    mesh_py = paths.geometry / "mesh.py"
    if not mesh_py.exists():
        _die(f"Missing {mesh_py}")
    mod = _load_module_from_path("mesh_mod", mesh_py)
    Files = getattr(mod, "Files", None)
    if Files is None:
        _die("geometry/mesh.py does not define class 'Files'")

    output_1d = paths.coupled / f"run_{run_i}" / "inlet"
    branching_csv = paths.coupled / f"run_0" / "branchingData_0.csv"
    print(f"[geometry] stl={stl_file}\n           output_1d={output_1d}\n           branching_data_file={branching_csv}")
    if run_i == 0:
        init = True
    else:
        init = False
    Files(stl_file=str(stl_file), output_1d=str(output_1d), branching_data_file=str(branching_csv),init=init)
    print("[ok] Geometry generated/refreshed.", flush=True)

def run_tesselate(paths: Paths, run_i: int, node_tol: float):
    """
    Call voronoi/tesselate_updated.py with environment-variable overrides so it runs from src/.
    """
    script = paths.voronoi / "tesselate_updated.py"
    if not script.exists():
        _die(f"Missing {script}")
    env = os.environ.copy()
    env.update({
        "SEEDS_CSV": str(paths.coupled / f"run_0" / "branchingData_0.csv"),
        "TERRITORIES_XDMF": str(paths.voronoi / "territories.xdmf"),
        "MESH_BP": str(paths.geometry / "tagged_branches.bp"),
        "PRESSURE_BP": str(paths.geometry / "pressure_checkpoint.bp"),
        "FLOW_BP": str(paths.geometry / "flow_checkpoint.bp"),
        "OUT_P_SRC_BP": str(paths.voronoi / "p_src_series.bp"),
        "OUT_Q_SRC_BP": str(paths.voronoi / "q_src_series.bp"),
        "NODE_TOL": str(node_tol),
        "MESH_TAGS_XDMF": str(paths.geometry / "bioreactor.xdmf"),
    })
    _run([sys.executable, str(script)], cwd=paths.root, env=env)
    print("[ok] Voronoi tesselation + sources written.", flush=True)


def run_darcy(paths: Paths, init: bool, run_i: int):
    """
    Import and execute PerfusionSolver from solves/single-compartment.py.
    (We import by path so we can pass init flag.)
    After solve, copy mbf outputs into coupled/run_{run_i}/.
    """
    sc_path = paths.solves / "darcy_P1.py"
    if not sc_path.exists():
        _die(f"Missing {sc_path}")
    mod = _load_module_from_path("darcy_P1_mod", sc_path)
    PerfusionSolver = getattr(mod, "PerfusionSolver", None)
    if PerfusionSolver is None:
        _die("darcy_P1.py does not define class 'PerfusionSolver'")
    mesh_file = str(paths.voronoi / "territories.xdmf")
    pres_file  = str(paths.voronoi / "p_src_series.bp")
    vel_file  = str(paths.voronoi / "q_src_series.bp")
    solver = PerfusionSolver(mesh_file, pres_file, vel_file)
    solver.setup(init=bool(init))
    print("[ok] Perfusion solve complete.", flush=True)
    # Copy mbf outputs into current run folder
    mbf_x = paths.solves / "out_darcy" / "mbf.xdmf"
    mbf_bp = paths.solves / "out_darcy" / "mbf.bp"
    run_dir = paths.coupled / f"run_{run_i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if mbf_x.exists():
        shutil.copy2(mbf_x, run_dir / "mbf.xdmf")
        shutil.copy2(mbf_x.with_suffix(".h5"), run_dir / "mbf.h5")
    if mbf_bp.exists():
        dst_bp = run_dir / "mbf.bp"
        # copy the .bp directory recursively
        if dst_bp.exists():
            shutil.rmtree(dst_bp)
        shutil.copytree(mbf_bp, dst_bp)
    else:
        print("[warn] mbf.bp directory not found; skipping copy.", flush=True)

def compare_q(mesh_file, paths: Paths, eps: float) -> tuple[float, dict]:
    """
    Compute the relative difference array |q_k - q_{k+1}| / max(|q_k|, eps).
    q_k comes from voronoi/q_src_series.bp if present; else voronoi/p_src_series.bp.
    q_{k+1} comes from solves/out_mixed_poisson/mbf.bp (field-selection is heuristic).
    Returns (max_rel, summary_dict)
    """

    q_src = paths.voronoi / "q_src_series.bp"
    if not q_src.exists():
        _die("q_src_series.bp not found in voronoi/")
    mbf_bp = paths.solves / "out_darcy" / "mbf.bp"
    if not mbf_bp.exists():
        _die("Missing solves/out_darcy/mbf.bp")

    with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension

    try:
        qk = _read_last_step_bp_array(mesh, q_src, function_name="q_src_density")
        qkp1 = _read_last_step_bp_array(mesh, mbf_bp, function_name="mbf_qTi_tagconst")
    except Exception as e:
        _die(f"Failed reading .bp files: {e}")

    stats = _summarize_rel_err(qk, qkp1, eps)
    return stats["max"], stats

def update_resistances(paths: Paths, run_i: int, run_next: int):
    """
    Call update_resistance.py with appropriate arguments, writing the new card into run_{next}/inlet.
    """
    script = paths.root / "update_resistance.py"
    if not script.exists():
        _die(f"Missing {script}")
    _ensure_dir(paths.coupled / f"run_{run_next}" / "inlet")
    args = [
        sys.executable, str(script),
        "--territories-xdmf", str(paths.voronoi / "territories.xdmf"),
        "--p-src-bp",         str(paths.voronoi / "p_src_series.bp"),
        "--mbf-bp",           str(paths.solves / "out_darcy" / "mbf.bp"),
        "--card-in",          str(paths.coupled / f"run_{run_i}" / "inlet" / "1d_simulation_input.json"),
        "--card-out",         str(paths.coupled / f"run_{run_next}" / "inlet" / "1d_simulation_input.json"),
    ]
    _run(args, cwd=paths.root)
    print(f"[ok] Wrote updated card to coupled/run_{run_next}/inlet/1d_simulation_input.json", flush=True)

def run_1d_solver(solver_cmd: str, paths: Paths, run_next: int):
    """
    Invoke external 1D solver:
      <solver_cmd> coupled/run_{run_next}/inlet/1d_simulation_input.json coupled/run_{run_next}/
    """
    card = paths.coupled / f"run_{run_next}" / "inlet" / "1d_simulation_input.json"
    out_dir = paths.coupled / f"run_{run_next}" / "inlet"
    if not card.exists():
        _die(f"Missing 1D card for run_{run_next}: {card}")
    _ensure_dir(out_dir)
    cmd = [solver_cmd, str(card)]
    _run(cmd, cwd=out_dir)
    print("[ok] 1D solver finished.", flush=True)

def append_progress(paths: Paths, it: int, stats: dict, tol: float):
    """
    Append iteration stats to coupled/convergence.csv and update a simple plot.
    """
    import csv
    csv_path = paths.coupled / "convergence.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["iter", "max", "mean", "median", "p95"])
        w.writerow([it, stats["max"], stats["mean"], stats["median"], stats["p95"]])
    print(f"[progress] iter={it}  max_rel={stats['max']:.5f}  tol={tol:.5f}")

    if plt is not None:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            plt.figure()
            plt.plot(df["iter"], df["max"], marker="o")
            plt.axhline(tol, linestyle="--")
            plt.xlabel("Iteration (k)")
            plt.ylabel(r"max (|q_k - q_{k+1}| / |q_k|), ε)")
            plt.title("Coupling convergence")
            plt.grid(True)
            fig_path = paths.coupled / "convergence.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Coupled 1D-NS <-> Darcy pipeline")
    ap.add_argument("--project-root", default=".", help="Path to src/ (contains geometry/, voronoi/, solves/)")
    ap.add_argument("--old-1d-inlet", default="../output/Forest_Output/1D_Output/100325/Run5_20branches/1D_Input_Files/inlet",
                    help="Path to existing 1D inlet folder to seed run_0 (contains 1d_simulation_input.json)")
    ap.add_argument("--stl-file", default="../files/geometry/cermRaksha_scaled_big.stl",
                    help="Path to cermRaksha_scaled_big.stl (if omitted, will try geometry/branched_network.xdmf route)")
    ap.add_argument("--one-d-solver-cmd", default='/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver',
                    help="Command/binary to run 1D solver, e.g. /path/to/solver")
    ap.add_argument("--tol", type=float, default=0.01, help="Convergence tolerance on relative flow (default 1%)")
    ap.add_argument("--max-iters", type=int, default=10, help="Max coupling iterations")
    ap.add_argument("--node-tol", type=float, default=1e-6, help="Node matching tolerance for Voronoi seeding")
    ap.add_argument("--mesh-file", default="geometry/bioreactor.xdmf", help="Path to mesh file")
    args = ap.parse_args()
    mesh_file = Path(args.mesh_file)
    paths = discover_paths(Path(args.project_root))
    os.chdir(paths.root)

    print(f"[setup] project root = {paths.root}")
    print(f"[setup] geometry = {paths.geometry}")
    print(f"[setup] voronoi  = {paths.voronoi}")
    print(f"[setup] solves   = {paths.solves}")
    print(f"[setup] coupled  = {paths.coupled}")

    prepare_run0(paths, Path(args.old_1d_inlet))

    for k in range(args.max_iters):
        print(f"\n========== Iteration k={k} ==========")
        # 1) Geometry
        if args.stl_file is None:
            # Use existing geometry outputs if present; otherwise require STL
            if not (paths.geometry / "tagged_branches.bp").exists():
                _die("--stl-file is required because geometry/tagged_branches.bp is missing")
        else:
            run_geometry(paths, Path(args.stl_file), run_i=k)

        # 2) Voronoi + source series
        run_tesselate(paths, run_i=k, node_tol=args.node_tol)

        # 3) Darcy solver (init=True for k==0)
        run_darcy(paths, init=(k == 0), run_i=k)

        # 4) Compare q and decide
        try:
            max_rel, stats = compare_q(mesh_file, paths, eps=1e-30)
        except SystemExit:
            raise
        except Exception as e:
            _die(f"Comparison failed: {e}")
        append_progress(paths, it=k, stats=stats, tol=args.tol)

        if max_rel <= args.tol:
            print(f"[converged] max_rel={max_rel:.6f} <= tol={args.tol:.6f}. Stopping.")
            break

        # 5) Update resistances -> new 1D card at run_{k+1}
        update_resistances(paths, run_i=k, run_next=k+1)

        # 6) Re-run 1D solver to produce new inlet outputs in run_{k+1}
        run_1d_solver(args.one_d_solver_cmd, paths, run_next=k+1)

    print("\n[done] See 'coupled/convergence.csv' and 'coupled/convergence.png' for progress.")
    print("      Latest 1D inputs are in 'coupled/run_{k}/inlet/' and mbf outputs in 'solves/out_darcy/'.", flush=True)

if __name__ == "__main__":
    main()
