#!/usr/bin/env python3
"""
Assign outlet PRESSURE BCs to a 1D deck using terminal branches and output.csv.

Procedure implemented:
1) Find terminal segments in branchingData_1.csv where Child1 and Child2 are NONE/empty.
   Keep BranchID and distalCoordsX/Y/Z. (First column is treated as BranchID.)
2) For each terminal BranchID, read output.csv and get the *last time* pressure_out for
   the series named "branch {BranchID}_seg0".
3) Parse 1d_simulation_input.json:
   - For SEGMENT rows whose last token != NONE, extract 6th entry after SEGMENT (outlet node id).
   - From NODE rows, get that node's x,y,z coordinates.
4) Match each outlet node to the nearest distalCoords (by Euclidean distance) and map to a BranchID.
5) Replace "RCR RCR_k" (or whatever bc_type/datatable) with "PRESSURE PRES_BC_k" on those SEGMENT rows.
6) Append DATATABLE blocks named PRES_BC_k.
   - By default, write a constant-pressure table at times [0.0, 1.0].
   - With --use-solver-times, reuse the solver's global time vector from output.csv for *all* PRES_BC_k.
     (Pressure value is the *last-time* pressure_out, repeated across time.)
7) Save back to the deck file. Existing PRES_BC_* tables are removed to avoid duplication.

Assumptions (as requested: no flexibility with column names):
- branchingData_1.csv has columns:
    [<BranchID first column>, Child1, Child2, distalCoordsX, distalCoordsY, distalCoordsZ, ...]
- output.csv has columns: [name, time, pressure_out]
- Deck has lines beginning with "NODE " and "SEGMENT ". Tokens are space-separated.

Usage:
  python assign_pressure_bcs.py \
      --deck /path/to/1d_simulation_input.json \
      --branching /path/to/branchingData_1.csv \
      --output /path/to/output.csv \
      [--use-solver-times]
"""
import argparse, json, math, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def is_none_like(val) -> bool:
    if pd.isna(val): return True
    s = str(val).strip().upper()
    return s in ("NONE", "", "NAN")

def load_branching(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Treat first column as BranchID
    first = df.columns[0]
    required = ["Child1", "Child2", "distalCoordsX", "distalCoordsY", "distalCoordsZ"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"{path.name} missing required columns: {missing}")
    df = df.rename(columns={first: "BranchID"})
    # terminals: both children none-like
    mask = df["Child1"].apply(is_none_like) & df["Child2"].apply(is_none_like)
    terms = df.loc[mask, ["BranchID", "distalCoordsX", "distalCoordsY", "distalCoordsZ"]].copy()
    if terms.empty:
        die("No terminal rows found in branchingData_1.csv (Child1/Child2 both NONE).")
    return terms.reset_index(drop=True)

def load_output(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path)
    required = ["name", "time", "pressure_out"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        die(f"{path.name} missing required columns: {missing}")
    return out

def last_pressures_by_name(out: pd.DataFrame) -> Dict[str, float]:
    last = {}
    for name, grp in out.groupby("name"):
        idx = grp["time"].idxmax()
        last[name] = float(out.loc[idx, "pressure_out"])
    return last

def branch_key(branch_id: int) -> str:
    return f"branch {branch_id}_seg0"

def parse_node_coords(lines: List[str]) -> Dict[int, Tuple[float, float, float]]:
    coords = {}
    for ln in lines:
        if ln.startswith("NODE "):
            parts = ln.split()
            try:
                node_id = int(float(parts[1]))
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords[node_id] = (x, y, z)
            except Exception:
                # Skip malformed lines
                pass
    if not coords:
        die("No NODE lines parsed from deck.")
    return coords

def parse_bc_segments(lines: List[str], node_coords: Dict[int, Tuple[float, float, float]]):
    segments = []
    for ln in lines:
        if ln.startswith("SEGMENT "):
            parts = ln.split()
            if len(parts) < 10:
                continue
            bc_type = parts[-2]
            datatable = parts[-1]
            if datatable == "NONE":
                continue  # only take those with a BC/datatable
            seg_name = parts[1]
            try:
                outlet_node = int(float(parts[6]))  # 6th entry after SEGMENT
            except Exception:
                continue
            xyz = node_coords.get(outlet_node, (math.nan, math.nan, math.nan))
            segments.append({
                "seg_name": seg_name,
                "outlet_node": outlet_node,
                "bc_type": bc_type,
                "datatable": datatable,
                "xyz": xyz,
                "raw": parts
            })
    if not segments:
        die("No SEGMENT rows with a BC (datatable != NONE) found.")
    return segments

def nearest_terminal_id(xyz: Tuple[float,float,float], term_XYZ: np.ndarray, term_ids: np.ndarray) -> int:
    x, y, z = xyz
    d2 = np.sum((term_XYZ - np.array([[x, y, z]]))**2, axis=1)
    j = int(np.argmin(d2))
    return int(term_ids[j])

def extract_k(datatable: str) -> int:
    m = re.search(r"_(\d+)$", datatable)
    if not m:
        die(f"Expected a suffix index in datatable name like 'RCR_12', got '{datatable}'")
    return int(m.group(1))

def remove_existing_pres_blocks(lines: List[str]) -> List[str]:
    out = []
    skipping = False
    for ln in lines:
        if ln.startswith("DATATABLE PRES_BC_"):
            skipping = True
            continue
        if skipping:
            if ln.strip() == "ENDDATATABLE":
                skipping = False
            continue
        out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, help="Path to 1d_simulation_input.json")
    ap.add_argument("--branching", required=True, help="Path to branchingData_1.csv")
    ap.add_argument("--output", required=True, help="Path to output.csv")
    ap.add_argument("--use-solver-times", action="store_true",
                    help="If set, DATATABLE times come from output.csv's global time vector; otherwise use [0.0, 1.0].")
    args = ap.parse_args()

    deck_path = Path(args.deck)
    branching_path = Path(args.branching)
    output_path = Path(args.output)

    # Load inputs
    terminals = load_branching(branching_path)
    out = load_output(output_path)
    lastP = last_pressures_by_name(out)

    # Time grid
    if args.use_solver_times:
        time_vec = np.sort(out["time"].unique().astype(float))
    else:
        time_vec = np.array([0.0, 1.0], dtype=float)

    term_XYZ = terminals[["distalCoordsX","distalCoordsY","distalCoordsZ"]].to_numpy(float)
    term_ids = terminals["BranchID"].to_numpy(int)

    # Read and parse deck
    text = deck_path.read_text()
    lines = text.splitlines()
    node_coords = parse_node_coords(lines)
    segments = parse_bc_segments(lines, node_coords)

    # Build assignments
    assignments = []
    for seg in segments:
        xyz = seg["xyz"]
        if any(math.isnan(v) for v in xyz):
            die(f"Outlet node {seg['outlet_node']} for segment {seg['seg_name']} has no parsed coordinates.")
        bid = nearest_terminal_id(xyz, term_XYZ, term_ids)
        key = branch_key(bid)
        if key not in lastP:
            die(f"No pressure_out found in output.csv for '{key}'.")
        k = extract_k(seg["datatable"])
        assignments.append({
            "seg_name": seg["seg_name"],
            "k": k,
            "branch_id": bid,
            "output_key": key,
            "pressure_last": lastP[key]
        })

    # Update SEGMENT lines
    # First, remove any existing PRES_BC blocks
    updated = remove_existing_pres_blocks(lines)

    # Replace bc type/name
    seg_map = {a["seg_name"]: a for a in assignments}
    for i, ln in enumerate(updated):
        if ln.startswith("SEGMENT "):
            parts = ln.split()
            seg_name = parts[1]
            if seg_name in seg_map:
                a = seg_map[seg_name]
                parts[-2] = "PRESSURE"
                parts[-1] = f"PRES_BC_{a['k']}"
                updated[i] = " ".join(parts)

    # Append DATATABLE blocks
    updated.append("")  # spacer
    for a in sorted(assignments, key=lambda d: d["k"]):
        p = a["pressure_last"]
        updated.append(f"DATATABLE PRES_BC_{a['k']} LIST")
        for t in time_vec:
            updated.append(f" {t:.6f} {p:.3f}")
        updated.append("ENDDATATABLE")
        updated.append("")

    # Write back (with backup)
    backup_path = deck_path.with_suffix(deck_path.suffix + ".bak")
    backup_path.write_text(text)
    deck_path.write_text("\n".join(updated))

    # Small summary
    print(f"Updated deck written to: {deck_path}")
    print(f"Backup saved to:        {backup_path}")
    print("\nAssignments (seg_name -> BranchID -> k -> pressure_last):")
    for a in sorted(assignments, key=lambda d: d['k']):
        print(f"  {a['seg_name']} -> {a['branch_id']} -> k={a['k']} -> {a['pressure_last']:.3f}")

if __name__ == "__main__":
    main()
