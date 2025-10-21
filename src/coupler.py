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
import zero_d

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
    denom = np.where(np.abs(qk) > eps, qk, eps)
    rel   = np.abs(qk - qkp1) / np.abs(denom)

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

def prepare_run0(paths: Paths, old_inlet: Path, old_outlet: Path):
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

    outlet_dst = run0 / "outlet"
    _ensure_dir(outlet_dst)
    _copy_tree(old_outlet, outlet_dst)

    # Try to copy branchingData_0.csv next to run_0
    src_csv = old_inlet.parent.parent / "branchingData_0.csv"
    snk_csv = old_inlet.parent.parent / "branchingData_1.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, run0 / "branchingData_0.csv")
        shutil.copy2(snk_csv, run0 / "branchingData_1.csv")
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

    output_1d_inlet = paths.coupled / f"run_{run_i}" / "inlet"
    output_1d_outlet = paths.coupled / f"run_{run_i}" / "outlet"
    branching_csv_inlet = paths.coupled / f"run_0" / "branchingData_0.csv"
    branching_csv_outlet = paths.coupled / f"run_0" / "branchingData_1.csv"
    print(f"[geometry] stl={stl_file}\n           output_1d_inlet={output_1d_inlet}\n           output_1d_outlet={output_1d_outlet}\n           branching_data_file={branching_csv_inlet}", flush=True)
    if run_i == 0:
        init = True
    else:
        init = False # Use True to re-use existing tags; False would re-generate from scratch
    Files(stl_file=str(stl_file), output_1d_inlet=str(output_1d_inlet), output_1d_outlet=str(output_1d_outlet), branching_data_inlet=str(branching_csv_inlet), branching_data_outlet=str(branching_csv_outlet), single = False, init=init)
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
        "SEEDS_INLET_CSV": str(paths.coupled / f"run_0" / "branchingData_0.csv"),
        "SEEDS_OUTLET_CSV": str(paths.coupled / f"run_0" / "branchingData_1.csv"),
        "TERRITORIES_INLET_XDMF": str(paths.voronoi / "territories_inlet.xdmf"),
        "TERRITORIES_OUTLET_XDMF": str(paths.voronoi / "territories_outlet.xdmf"),
        "MESH_INLET_BP": str(paths.geometry / "tagged_branches_inlet.bp"),
        "MESH_OUTLET_BP": str(paths.geometry / "tagged_branches_outlet.bp"),
        "PRESSURE_INLET_BP": str(paths.geometry / "pressure_checkpoint_inlet.bp"),
        "PRESSURE_OUTLET_BP": str(paths.geometry / "pressure_checkpoint_outlet.bp"),
        "FLOW_INLET_BP": str(paths.geometry / "flow_checkpoint_inlet.bp"),
        "FLOW_OUTLET_BP": str(paths.geometry / "flow_checkpoint_outlet.bp"),
        "OUT_P_INLET_SRC_BP": str(paths.voronoi / "p_src_inlet_series.bp"),
        "OUT_P_OUTLET_SRC_BP": str(paths.voronoi / "p_src_outlet_series.bp"),
        "OUT_Q_INLET_SRC_BP": str(paths.voronoi / "q_src_inlet_series.bp"),
        "OUT_Q_OUTLET_SRC_BP": str(paths.voronoi / "q_src_outlet_series.bp"),
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
    mesh_inlet_file = str(paths.voronoi / "territories_inlet.xdmf")
    mesh_outlet_file = str(paths.voronoi / "territories_outlet.xdmf")
    pres_inletfile  = str(paths.voronoi / "p_src_inlet_series.bp")
    vel_inletfile  = str(paths.voronoi / "q_src_inlet_series.bp")
    pres_outletfile  = str(paths.voronoi / "p_src_outlet_series.bp")
    vel_outletfile  = str(paths.voronoi / "q_src_outlet_series.bp")
    solver = PerfusionSolver(mesh_inlet_file, mesh_outlet_file, pres_inletfile, pres_outletfile, vel_inletfile, vel_outletfile)
    solver.setup(init=bool(init))
    print("[ok] Perfusion solve complete.", flush=True)
    # Copy mbf outputs into current run folder
    mbf_inlet_x = paths.solves / "out_darcy" / "mbf_inlet.xdmf"
    mbf_outlet_x = paths.solves / "out_darcy" / "mbf_outlet.xdmf"
    mbf_inlet_bp = paths.solves / "out_darcy" / "mbf_inlet.bp"
    mbf_outlet_bp = paths.solves / "out_darcy" / "mbf_outlet.bp"
    run_dir = paths.coupled / f"run_{run_i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if mbf_inlet_x.exists():
        shutil.copy2(mbf_inlet_x, run_dir / "mbf_inlet.xdmf")
        shutil.copy2(mbf_inlet_x.with_suffix(".h5"), run_dir / "mbf_inlet.h5")
    if mbf_outlet_x.exists():
        shutil.copy2(mbf_outlet_x, run_dir / "mbf_outlet.xdmf")
        shutil.copy2(mbf_outlet_x.with_suffix(".h5"), run_dir / "mbf_outlet.h5")
    if mbf_outlet_bp.exists():
        dst_bp = run_dir / "mbf_outlet.bp"
        # copy the .bp directory recursively
        if dst_bp.exists():
            shutil.rmtree(dst_bp)
        shutil.copytree(mbf_outlet_bp, dst_bp)
    if mbf_inlet_bp.exists():
        dst_bp = run_dir / "mbf_inlet.bp"
        # copy the .bp directory recursively
        if dst_bp.exists():
            shutil.rmtree(dst_bp)
        shutil.copytree(mbf_inlet_bp, dst_bp)
    else:
        print("[warn] mbf.bp directory not found; skipping copy.", flush=True)

def compare_q(mesh_file, paths: Paths, eps: float) -> tuple[float, dict]:
    """
    Compute the relative difference array |q_k - q_{k+1}| / max(|q_k|, eps).
    q_k comes from voronoi/q_src_series.bp if present; else voronoi/p_src_series.bp.
    q_{k+1} comes from solves/out_mixed_poisson/mbf.bp (field-selection is heuristic).
    Returns (max_rel, summary_dict)
    """

    q_src = paths.voronoi / "q_src_inlet_series.bp"
    q_snk = paths.voronoi / "q_src_outlet_series.bp"
    if not q_src.exists():
        _die("q_src_series.bp not found in voronoi/")
    mbf_inlet_bp = paths.solves / "out_darcy" / "mbf_inlet.bp"
    if not mbf_inlet_bp.exists():
        _die("Missing solves/out_darcy/mbf_inlet.bp")
    mbf_outlet_bp = paths.solves / "out_darcy" / "mbf_outlet.bp"
    if not mbf_outlet_bp.exists():
        _die("Missing solves/out_darcy/mbf_outlet.bp")

    with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        fdim = mesh.topology.dim - 1  # Facet dimension

    try:
        qk_src = _read_last_step_bp_array(mesh, q_src, function_name="q_src_density")
        qk_snk = _read_last_step_bp_array(mesh, q_snk, function_name="q_src_density")
        qkp1_src = _read_last_step_bp_array(mesh, mbf_inlet_bp, function_name="mbf_qTi_tagconst")
        qkp1_snk = _read_last_step_bp_array(mesh, mbf_outlet_bp, function_name="mbf_qTi_tagconst")
    except Exception as e:
        _die(f"Failed reading .bp files: {e}")

    stats_src = _summarize_rel_err(qk_src, qkp1_src, eps)
    stats_snk = _summarize_rel_err(qk_snk, qkp1_snk, eps)
    return max(stats_src["max"], stats_snk["max"]), {"inlet": stats_src, "outlet": stats_snk}

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
        "--territories-xdmf", str(paths.voronoi / "territories_inlet.xdmf"),
        "--p-src-bp",         str(paths.voronoi / "p_src_inlet_series.bp"),
        "--mbf-bp",           str(paths.solves / "out_darcy" / "mbf_inlet.bp"),
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
    print("[ok] 1D solver for inlet finished.", flush=True)
    card = paths.coupled / f"run_{run_next}" / "outlet" / "1d_simulation_input.json"
    out_dir = paths.coupled / f"run_{run_next}" / "outlet"
    if not card.exists():
        _die(f"Missing 1D card for run_{run_next}: {card}")
    _ensure_dir(out_dir)
    cmd = [solver_cmd, str(card)]
    _run(cmd, cwd=out_dir)
    print("[ok] 1D solver for outlet finished.", flush=True)

def append_progress(paths: Paths, it: int, stats: dict, tol: float):
    """
    Append iteration stats to coupled/convergence.csv and update a simple plot.
    """
    import csv
    csv_path = paths.coupled / "convergence_inlet.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["iter", "max", "mean", "median", "p95"])
        w.writerow([it, stats["inlet"]["max"], stats["inlet"]["mean"], stats["inlet"]["median"], stats["inlet"]["p95"]])
    print(f"[progress] iter={it}  max_rel={stats['inlet']['max']:.5f}  tol={tol:.5f}")

    csv_path = paths.coupled / "convergence_outlet.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["iter", "max", "mean", "median", "p95"])
        w.writerow([it, stats["outlet"]["max"], stats["outlet"]["mean"], stats["outlet"]["median"], stats["outlet"]["p95"]])
    print(f"[progress] iter={it}  max_rel={stats['outlet']['max']:.5f}  tol={tol:.5f}")

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
    ap.add_argument("--old-1d-inlet", default="../output/Forest_Output/1D_Output/101725/Run6_10branches/1D_Input_Files/inlet",
                    help="Path to existing 1D inlet folder to seed run_0 (contains 1d_simulation_input.json)")
    ap.add_argument("--old-1d-outlet", default="../output/Forest_Output/1D_Output/101725/Run6_10branches/1D_Input_Files/outlet",
                    help="Path to existing 1D outlet folder to seed run_0 (contains 1d_simulation_input.json)")
    ap.add_argument("--stl-file", default="../files/geometry/cermRaksha_scaled_big.stl",
                    help="Path to cermRaksha_scaled_big.stl (if omitted, will try geometry/branched_network.xdmf route)")
    ap.add_argument("--one-d-solver-cmd", default='/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver',
                    help="Command/binary to run 1D solver, e.g. /path/to/solver")
    ap.add_argument("--svzerodsolver", default="svzerodsolver",
                help="svZeroDSolver binary")
    ap.add_argument("--assign-pressures-script", default="assign_pressure_bcs.py",
                    help="Script that updates 1D outlet pressure BCs from 0D results")
    ap.add_argument("--node-scale", type=float, default=1.0,
                help="Scale distal coords if CSV is not in meters (e.g., 0.001 for mm)")
    ap.add_argument("--tol", type=float, default=0.01, help="Convergence tolerance on relative flow (default 1%)")
    ap.add_argument("--max-iters", type=int, default=20, help="Max coupling iterations")
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

    # prepare_run0(paths, Path(args.old_1d_inlet), Path(args.old_1d_outlet))
    zero_d.prepare_outlet_run0(
        zero_d.Paths(paths.root, paths.geometry, paths.voronoi, paths.solves, paths.coupled),
        Path(args.old_1d_inlet)
    )


    for k in range(args.max_iters):
        print(f"\n========== Iteration k={k} ==========")
        # 1) Geometry
        if args.stl_file is None:
            # Use existing geometry outputs if present; otherwise require STL
            if not (paths.geometry / "tagged_branches_inlet.bp").exists():
                _die("--stl-file is required because geometry/tagged_branches_inlet.bp is missing")
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
        if k%2 == 0:
            print("[info] Odd iteration: skipping resistance update to avoid oscillations.")
            print("[info] Instead, updating outlet pressures via 0D solver.")
            # 6) Run 0D solver to update outlet pressures
            assign_path = Path(args.assign_pressures_script)
            zero_d.outlet_stage_iter(
                zero_d.Paths(paths.root, paths.geometry, paths.voronoi, paths.solves, paths.coupled),
                run_i=k, run_next=k+1,
                assign_script=assign_path if assign_path.exists() else None,
                svzerodsolver=args.svzerodsolver,
                one_d_solver_cmd=args.one_d_solver_cmd,
                node_scale=float(args.node_scale),
                mbf_field="mbf_qTi_tagconst",
            )
            (paths.coupled / f"run_{k+1}/inlet").mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths.coupled / f"run_{k}/inlet/1d_simulation_input.json",
                         paths.coupled / f"run_{k+1}/inlet/1d_simulation_input.json")

        else:
            print("[info] Even iteration: skipping pressure update to avoid oscillations.")
            print("[info] Even iteration: updating resistances.")
            update_resistances(paths, run_i=k, run_next=k+1)
            (paths.coupled / f"run_{k+1}/outlet").mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths.coupled / f"run_{k}/solver_0d_new.in",
                         paths.coupled / f"run_{k+1}/solver_0d_new.in")
            shutil.copy2(paths.coupled / f"run_{k}/outlet/1d_simulation_input.json",
                         paths.coupled / f"run_{k+1}/outlet/1d_simulation_input.json")

        
        # 6) Re-run 1D solver to produce new inlet outputs in run_{k+1}
        run_1d_solver(args.one_d_solver_cmd, paths, run_next=k+1)

    print("\n[done] See 'coupled/convergence.csv' and 'coupled/convergence.png' for progress.")
    print("      Latest 1D inputs are in 'coupled/run_{k}/inlet/' and mbf outputs in 'solves/out_darcy/'.", flush=True)

if __name__ == "__main__":
    main()
