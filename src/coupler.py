#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pseudotimestep 0D–Darcy coupler.

High-level loop:
  0) Seed run_0 from old 0D inlet/outlet folders and zero interface BCs.
  1) For i = 1..N:
       - copy previous run (run_{i-1}) -> run_i
       - ramp driving BCs:
           inlet:  INFLOW FLOW Q  <- min(i*Tau,1) * Q0
           outlet: PRESSURE_IN P <- min(i*Tau,1) * P0
       - run 0D inlet + outlet
       - run geometry (mesh tagging)
       - run Darcy
       - read interface_bc.json and update interface BCs in run_i decks:
           inlet:  OUTk PRESSURE P <- p_inlet_nodes[k]
           outlet: OUTk FLOW Q     <- q_outlet[k]
"""

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Tuple, List, Any


SOLVER_0D_NAME = "solver_0d_new.in"  # adjust if your filename differs


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    cmd_str = " ".join(map(str, cmd))
    print(f"[run] {cmd_str} (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _die(f"Failed to load module from {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Paths:
    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.geometry = self.root / "geometry"
        self.solves = self.root / "solves"
        self.coupled = self.root / "coupled"

        self.geometry.mkdir(parents=True, exist_ok=True)
        self.solves.mkdir(parents=True, exist_ok=True)
        self.coupled.mkdir(parents=True, exist_ok=True)

    def run_dir(self, i: int) -> Path:
        d = self.coupled / f"run_{i}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_inlet_dir(self, i: int) -> Path:
        d = self.run_dir(i) / "inlet"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run_outlet_dir(self, i: int) -> Path:
        d = self.run_dir(i) / "outlet"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# 0D deck helpers
# ---------------------------------------------------------------------------

def _load_0d_deck(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _save_0d_deck(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _extract_base_Q_and_zero_interface_pressures(deck: dict) -> float:
    """
    For inlet deck:
      - find INFLOW FLOW Q base value
      - set all OUTk PRESSURE P values to 0 (interface BCs) :contentReference[oaicite:3]{index=3}
    """
    base_Q = None
    for bc in deck.get("boundary_conditions", []):
        name = bc.get("bc_name", "")
        btype = bc.get("bc_type", "")
        vals = bc.get("bc_values", {})

        # INFLOW FLOW Q (driving BC)
        if name == "INFLOW" and btype == "FLOW":
            Q_list = vals.get("Q", [])
            if not Q_list:
                _die("INFLOW FLOW bc has no 'Q' values")
            base_Q = Q_list[0]

        # Interface pressures at outlets
        if name.startswith("OUT") and btype == "PRESSURE":
            P_list = vals.get("P", [])
            if P_list:
                bc["bc_values"]["P"] = [0.0 for _ in P_list]

    if base_Q is None:
        _die("Could not find INFLOW FLOW in inlet 0D deck")
    return base_Q

def _extract_base_P_and_zero_interface_pressures(deck: dict) -> float:
    """
    For outlet deck:
      - find PRESSURE_IN PRESSURE P base value
      - set all OUTk FLOW Q values to 0 (interface BCs) :contentReference[oaicite:4]{index=4}
    """
    base_P = None
    for bc in deck.get("boundary_conditions", []):
        name = bc.get("bc_name", "")
        btype = bc.get("bc_type", "")
        vals = bc.get("bc_values", {})

        # PRESSURE_IN P (driving BC)
        if name == "PRESSURE_IN" and btype == "PRESSURE":
            P_list = vals.get("P", [])
            if not P_list:
                _die("PRESSURE_IN bc has no 'P' values")
            base_P = P_list[0]

        # Interface pressures at outlets
        if name.startswith("OUT") and btype == "PRESSURE":
            P_list = vals.get("P", [])
            if P_list:
                bc["bc_values"]["P"] = [1333.22 for _ in P_list]

    if base_P is None:
        _die("Could not find PRESSURE_IN in outlet 0D deck")
    return base_P

def _extract_base_P_and_zero_interface_flows(deck: dict) -> float:
    """
    For outlet deck:
      - find PRESSURE_IN PRESSURE P base value
      - set all OUTk FLOW Q values to 0 (interface BCs) :contentReference[oaicite:4]{index=4}
    """
    base_P = None
    for bc in deck.get("boundary_conditions", []):
        name = bc.get("bc_name", "")
        btype = bc.get("bc_type", "")
        vals = bc.get("bc_values", {})

        # PRESSURE_IN P (driving BC)
        if name == "PRESSURE_IN" and btype == "PRESSURE":
            P_list = vals.get("P", [])
            if not P_list:
                _die("PRESSURE_IN bc has no 'P' values")
            base_P = P_list[0]

        # Interface outlet flows
        if name.startswith("OUT") and btype == "FLOW":
            Q_list = vals.get("Q", [])
            if Q_list:
                bc["bc_values"]["Q"] = [0.0 for _ in Q_list]

    if base_P is None:
        _die("Could not find PRESSURE_IN in outlet 0D deck")
    return base_P


def _apply_pseudotimestep_to_inlet(deck: dict, base_Q: float, scale: float) -> None:
    """Set INFLOW FLOW Q to min(i*Tau,1)*base_Q for all time points."""
    for bc in deck.get("boundary_conditions", []):
        if bc.get("bc_name") == "INFLOW" and bc.get("bc_type") == "FLOW":
            vals = bc.setdefault("bc_values", {})
            Q_list = vals.get("Q", [])
            if not Q_list:
                _die("INFLOW FLOW bc has no 'Q' values to scale")
            new_Q = scale * base_Q
            bc["bc_values"]["Q"] = [new_Q for _ in Q_list]


def _apply_pseudotimestep_to_outlet(deck: dict, base_P: float, scale: float) -> None:
    """Set PRESSURE_IN P to min(i*Tau,1)*base_P for all time points."""
    for bc in deck.get("boundary_conditions", []):
        if bc.get("bc_name") == "PRESSURE_IN" and bc.get("bc_type") == "PRESSURE":
            vals = bc.setdefault("bc_values", {})
            P_list = vals.get("P", [])
            if not P_list:
                _die("PRESSURE_IN bc has no 'P' values to scale")
            new_P = (1-scale)*(1.0*1333.22) 
            # new_P = scale * base_P
            bc["bc_values"]["P"] = [new_P for _ in P_list]


def _update_interface_bcs_from_darcy(
    paths: Paths, run_i: int, solver_filename: str = SOLVER_0D_NAME
) -> None:
    """
    Read solves/out_darcy/interface_bc.json and update interface BCs:

      inlet deck:  OUTk PRESSURE P <- p_inlet_nodes[k]
      outlet deck: OUTk FLOW Q     <- q_outlet[k]

    We assume ordering of OUTk BCs matches the ordering of p_inlet_nodes / q_outlet. :contentReference[oaicite:5]{index=5}
    """
    interface_path = paths.solves / "out_darcy" / "interface_bc.json"
    if not interface_path.is_file():
        _die(f"Missing interface_bc.json at {interface_path}")

    with interface_path.open("r") as f:
        iface = json.load(f)

    p_in = iface.get("p_inlet_nodes", [])
    p_out = iface.get("p_outlet_nodes", [])
    if not p_in or not p_out:
        _die("interface_bc.json missing p_inlet_nodes or p_outlet_nodes")

    # --- inlet deck: update OUTk pressures ---
    inlet_path = paths.run_inlet_dir(run_i) / solver_filename
    inlet_deck = _load_0d_deck(inlet_path)

    # Collect interface BCs in the order they appear
    inlet_interface_bcs = [
        bc for bc in inlet_deck.get("boundary_conditions", [])
        if bc.get("bc_type") == "PRESSURE" and str(bc.get("bc_name", "")).startswith("OUT")
    ]

    if len(inlet_interface_bcs) != len(p_in):
        print(
            f"[warn] inlet interface count ({len(inlet_interface_bcs)}) "
            f"!= len(p_inlet_nodes) ({len(p_in)}); truncating to min.",
            flush=True,
        )

    for idx, bc in enumerate(inlet_interface_bcs):
        if idx >= len(p_in):
            break
        P_list = bc.get("bc_values", {}).get("P", [])
        newP = p_in[idx]
        if P_list:
            bc["bc_values"]["P"] = [newP for _ in P_list]

    _save_0d_deck(inlet_path, inlet_deck)

    # --- outlet deck: update OUTk flows ---
    outlet_path = paths.run_outlet_dir(run_i) / solver_filename
    outlet_deck = _load_0d_deck(outlet_path)

    outlet_interface_bcs = [
        bc for bc in outlet_deck.get("boundary_conditions", [])
        if bc.get("bc_type") == "PRESSURE" and str(bc.get("bc_name", "")).startswith("OUT")
    ]

    if len(outlet_interface_bcs) != len(p_out):
        print(
            f"[warn] outlet interface count ({len(outlet_interface_bcs)}) "
            f"!= len(p_outlet) ({len(p_out)}); truncating to min.",
            flush=True,
        )

    for idx, bc in enumerate(outlet_interface_bcs):
        if idx >= len(p_out):
            break
        P_list = bc.get("bc_values", {}).get("P", [])
        newP = p_out[idx]
        if P_list:
            bc["bc_values"]["P"] = [newP for _ in P_list]

    _save_0d_deck(outlet_path, outlet_deck)


# ---------------------------------------------------------------------------
# Geometry + Darcy wrappers
# ---------------------------------------------------------------------------

def run_geometry(paths: Paths, stl_file: Path, run_i: int) -> None:
    """
    Import branched mesh + tag bioreactor using Files() from geometry/mesh.py.
    This is essentially your existing function.
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
    branching_csv_inlet = paths.coupled / "run_0" / "branchingData_0.csv"
    branching_csv_outlet = paths.coupled / "run_0" / "branchingData_1.csv"

    print(
        f"[geometry] stl={stl_file}\n"
        f"           output_1d_inlet={output_1d_inlet}\n"
        f"           output_1d_outlet={output_1d_outlet}\n"
        f"           branching_data_inlet={branching_csv_inlet}",
        flush=True,
    )

    init = (run_i == 0)
    Files(
        stl_file=str(stl_file),
        output_1d_inlet=str(output_1d_inlet),
        output_1d_outlet=str(output_1d_outlet),
        branching_data_inlet=str(branching_csv_inlet),
        branching_data_outlet=str(branching_csv_outlet),
        single=False,
        init=init,
    )
    print("[ok] Geometry generated/refreshed.", flush=True)


def run_darcy(paths: Paths) -> None:
    """
    Wrapper to run the Darcy solver.

    Assumes there is a 'run_darcy.py' (or similar) in solves/out_darcy that
    constructs PerfusionSolver with the new filenames you listed and runs it.

    Adjust this to match your actual script name if needed.
    """
    out_darcy = paths.solves 
    script = out_darcy / "darcy_P1_v2.py"
    if not script.is_file():
        _die(f"Expected Darcy driver script at {script}")
    _run([sys.executable, str(script)], cwd=out_darcy)


# ---------------------------------------------------------------------------
# 0D run + seeding
# ---------------------------------------------------------------------------

def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        # Copy contents into existing directory
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    else:
        shutil.copytree(src, dst)


def _run_0d_simulation(
    solver_bin: str,
    run_dir: Path,
    solver_filename: str = SOLVER_0D_NAME,
) -> None:
    """Run svZeroDSolver for a given run directory."""
    card = run_dir / solver_filename
    if not card.is_file():
        _die(f"Missing 0D input card {card} in {run_dir}")
    out_csv = run_dir / "output.csv"
    _run([solver_bin, str(card), str(out_csv)], cwd=run_dir)
    print(f"[0D] output: {out_csv}", flush=True)


def seed_run0_and_get_bases(
    paths: Paths,
    old_inlet_dir: Path,
    old_outlet_dir: Path,
    solver_filename: str = SOLVER_0D_NAME,
    svzerodsolver: str = "svzerodsolver",
) -> Tuple[float, float]:
    """
    Copy old 0D inlet/outlet directories into run_0/{inlet,outlet},
    zero interface BCs, and return base_Q (INFLOW) and base_P (PRESSURE_IN).
    """
    run0_in = paths.run_inlet_dir(0)
    run0_out = paths.run_outlet_dir(0)

    print(f"[seed] Copying old inlet from {old_inlet_dir} -> {run0_in}")
    _copy_tree(old_inlet_dir, run0_in)
    print(f"[seed] Copying old outlet from {old_outlet_dir} -> {run0_out}")
    _copy_tree(old_outlet_dir, run0_out)

    csv_src = old_inlet_dir.parent.parent / "branchingData_0.csv"
    print(f"[0D] Found branchingData_0.csv at {csv_src}")
    if not csv_src.exists():
        _die(f"Missing branchingData_0.csv at {csv_src}")
    csv_snk = old_outlet_dir.parent.parent / "branchingData_1.csv"
    if not csv_snk.exists():
        _die(f"Missing branchingData_1.csv at {csv_snk}")
    (paths.coupled / f"run_{0}" / "branchingData_0.csv").write_text(csv_src.read_text())
    (paths.coupled / f"run_{0}" / "branchingData_1.csv").write_text(csv_snk.read_text())
    print(f"[0D] Prepared outlet run_{0}")

    # Load decks
    inlet_path = run0_in / solver_filename
    outlet_path = run0_out / solver_filename
    if not inlet_path.is_file():
        _die(f"run_0 inlet deck {inlet_path} not found")
    if not outlet_path.is_file():
        _die(f"run_0 outlet deck {outlet_path} not found")

    inlet_deck = _load_0d_deck(inlet_path)
    outlet_deck = _load_0d_deck(outlet_path)

    # Zero interface BCs and extract base driving BCs
    base_Q_inlet = _extract_base_Q_and_zero_interface_pressures(inlet_deck)
    base_P_outlet = _extract_base_P_and_zero_interface_pressures(outlet_deck)

    _apply_pseudotimestep_to_inlet(inlet_deck, base_Q_inlet, 0.0)
    _apply_pseudotimestep_to_outlet(outlet_deck, base_P_outlet, 1.0)

    _save_0d_deck(inlet_path, inlet_deck)
    _save_0d_deck(outlet_path, outlet_deck)

    print("[step] Running 0D inlet")
    _run_0d_simulation(svzerodsolver, run0_in, solver_filename)
    print("[step] Running 0D outlet")
    _run_0d_simulation(svzerodsolver, run0_out, solver_filename)

    subprocess.run("python plot_0d_results_to_3d.py", cwd=run0_in, shell=True)
    subprocess.run("python plot_0d_results_to_3d.py", cwd=run0_out, shell=True)

    print(f"[seed] base_Q_inlet={base_Q_inlet}, base_P_outlet={base_P_outlet}", flush=True)
    return base_Q_inlet, base_P_outlet



def _copy_geom_and_plot_files(paths: Paths, run_i: int) -> None:
    """
    Copy plot_0d_results_to_3d.py and geom.csv from run_0/{inlet,outlet}
    into run_i/{inlet,outlet} if they are not already present.
    """
    for side in ("inlet", "outlet"):
        src_dir = paths.coupled / "run_0" / side
        dst_dir = paths.coupled / f"run_{run_i}" / side

        for fname in ("plot_0d_results_to_3d.py", "geom.csv"):
            src = src_dir / fname
            dst = dst_dir / fname
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
                print(f"[copy] {src} -> {dst}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Coupled 1D-NS <-> Darcy pipeline (pseudotimestep 0D coupler)")
    ap.add_argument("--project-root", default=".", help="Path to src/ (contains geometry/, voronoi/, solves/)")
    ap.add_argument(
        "--old-0d-inlet",
        default="/Users/rakshakonanur/Documents/Research/vascularize/output/Forest_Output/1D_Output/120825/Run1_10branches_0d_0d_pres/1D_Input_Files/inlet",
        help="Path to existing 0D inlet folder to seed run_0 (contains solver_0d_new.in)",
    )
    ap.add_argument(
        "--old-0d-outlet",
        default="/Users/rakshakonanur/Documents/Research/vascularize/output/Forest_Output/1D_Output/120825/Run1_10branches_0d_0d_pres/1D_Input_Files/outlet",
        help="Path to existing 0D outlet folder to seed run_0 (contains solver_0d_new.in)",
    )
    ap.add_argument(
        "--stl-file",
        default="../files/geometry/cermRaksha_scaled_big.stl",
        help="Path to STL (e.g., cermRaksha_scaled_big.stl)",
    )
    ap.add_argument(
        "--one-d-solver-cmd",
        default='/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver',
        help="Command/binary to run 1D solver (not used directly here yet)",
    )
    ap.add_argument(
        "--svzerodsolver",
        default="svzerodsolver",
        help="svZeroDSolver binary",
    )
    ap.add_argument(
        "--pseudo-steps",
        type=int,
        default=50,
        help="Number of pseudo-timesteps N (Tau = 1/N)",
    )
    ap.add_argument(
        "--solver-0d-filename",
        default=SOLVER_0D_NAME,
        help="Filename of 0D solver card inside each run directory",
    )
    args = ap.parse_args()

    paths = Paths(Path(args.project_root))
    stl_file = Path(args.stl_file).resolve()
    if not stl_file.is_file():
        _die(f"STL file not found: {stl_file}")

    old_inlet_dir = Path(args.old_0d_inlet).resolve()
    old_outlet_dir = Path(args.old_0d_outlet).resolve()
    if not old_inlet_dir.is_dir():
        _die(f"old 0D inlet directory not found: {old_inlet_dir}")
    if not old_outlet_dir.is_dir():
        _die(f"old 0D outlet directory not found: {old_outlet_dir}")

    # 0) Seed run_0 and extract base BCs
    base_Q_inlet, base_P_outlet = seed_run0_and_get_bases(
        paths, old_inlet_dir, old_outlet_dir, solver_filename=args.solver_0d_filename, svzerodsolver=args.svzerodsolver
    )
    N = args.pseudo_steps
    Tau = 1.0 / float(N)
    print(f"[loop] pseudo-steps N={N}, Tau={Tau}", flush=True)

    prev_run = 0

    for i in range(1, N + 1):
        scale = min(i * Tau, 1.0)
        print(f"\n=== Pseudotimestep i={i}, scale={scale} ===", flush=True)

        # Create run_i by copying previous run
        src_in = paths.run_inlet_dir(prev_run)
        src_out = paths.run_outlet_dir(prev_run)
        dst_in = paths.run_inlet_dir(i)
        dst_out = paths.run_outlet_dir(i)

        print(f"[copy] run_{prev_run}/inlet -> run_{i}/inlet")
        _copy_tree(src_in, dst_in)
        print(f"[copy] run_{prev_run}/outlet -> run_{i}/outlet")
        _copy_tree(src_out, dst_out)

        # Apply pseudotimestep scaling to driving BCs
        inlet_path = dst_in / args.solver_0d_filename
        outlet_path = dst_out / args.solver_0d_filename
        inlet_deck = _load_0d_deck(inlet_path)
        outlet_deck = _load_0d_deck(outlet_path)

        # if i is even run outlet first, else run inlet first
        # if i % 2 == 0:
        #     _apply_pseudotimestep_to_outlet(outlet_deck, base_P, scale)
        # else:
        #     _apply_pseudotimestep_to_inlet(inlet_deck, base_Q, scale)

        _apply_pseudotimestep_to_inlet(inlet_deck, base_Q_inlet, scale)
        _apply_pseudotimestep_to_outlet(outlet_deck, base_P_outlet, scale)

        _save_0d_deck(inlet_path, inlet_deck)
        _save_0d_deck(outlet_path, outlet_deck)

        # 5) Run both 0D simulations
        print("[step] Running 0D inlet")
        _run_0d_simulation(args.svzerodsolver, dst_in, solver_filename=args.solver_0d_filename)
        print("[step] Running 0D outlet")
        _run_0d_simulation(args.svzerodsolver, dst_out, solver_filename=args.solver_0d_filename)

        # 6) Copy plot + geom files from run_0
        _copy_geom_and_plot_files(paths, i)

        # 7) Geometry
        run_geometry(paths, stl_file, i)

        # 8) Darcy
        run_darcy(paths)

        # 9) Update interface BCs from interface_bc.json
        _update_interface_bcs_from_darcy(paths, i, solver_filename=args.solver_0d_filename)

        print(f"Running plot_0d_results_to_3d.py in {dst_in}")
        subprocess.run("python plot_0d_results_to_3d.py", cwd=dst_in, shell=True)

        print(f"Running plot_0d_results_to_3d.py in {dst_out}")
        subprocess.run("python plot_0d_results_to_3d.py", cwd=dst_out, shell=True)

        # Prepare for next iteration
        prev_run = i

    print("\n[done] Pseudotimestep coupling complete.", flush=True)


if __name__ == "__main__":
    main()
