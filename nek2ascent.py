#!/usr/bin/env python3
#
# Copyright (c) 2025 Victor Mateevitsi. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# nek5000_ascent_reader.py
# Usage (example):
#   mpirun -n 8 python nek5000_ascent_reader.py sim.nek5000 --out out_images --field "Velocity Magnitude"

import os
import struct
import argparse
from typing import Tuple, Dict, List, Optional

import numpy as np
from mpi4py import MPI

import conduit
import ascent.mpi  # Ascent must be built with MPI; we pass the comm at open().
from nek5000reader import Nek5000Reader
import sys

# ---- pretty progress (rank 0 only) ----
try:
    from rich.progress import (
        Progress, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
    )
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

def make_progress(rank: int, enabled: bool = True):
    """Rank-0 real progress; no-op elsewhere or if rich isn't available."""
    if rank != 0 or not RICH_AVAILABLE or not enabled:
        class _Dummy:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def add_task(self, *a, **k): return 0
            def advance(self, *a, **k): pass
            def update(self, *a, **k): pass
            def log(self, *a, **k): 
                # fallback: print single-line messages
                try: print(*a, flush=True)
                except: pass
        return _Dummy()
    # rank == 0 and Rich is available: create a Progress with a forced Console
    # (force_terminal=True) so logs get a readable bar in many batch viewers.
    console = None
    try:
        from rich.console import Console
        console = Console(force_terminal=True)
    except Exception:
        console = None

    return Progress(
        SpinnerColumn(style="bold green"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None, style="cyan", complete_style="magenta"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=12,
    )

def add_field(node, name, arr, nverts):
    fld = node[f"fields/{name}"]
    fld["association"] = "vertex"
    fld["topology"] = "mesh"
    # Scalar
    if arr.size == nverts:
        fld["values"].set_external(arr)
    # 3-component vector packed as [x...][y...][z...]
    elif arr.size == 3 * nverts:
        fld["values/x"].set_external(arr[0*nverts:1*nverts])
        fld["values/y"].set_external(arr[1*nverts:2*nverts])
        fld["values/z"].set_external(arr[2*nverts:3*nverts])
    else:
        raise RuntimeError(f"Field '{name}' has {arr.size} values; expected {nverts} or {3*nverts}.")


# ---------------------------
# Conduit Blueprint + Ascent
# ---------------------------

def build_blueprint(coords_xyz: np.ndarray,
                    conn: np.ndarray,
                    blockDims: Tuple[int,int,int],
                    mesh_is_3d: bool,
                    fields: Dict[str, np.ndarray],
                    time_val: float,
                    cycle_val: int,
                    domain_id: int) -> conduit.Node:
    n = conduit.Node()
    # coordset
    n["coordsets/coords/type"] = "explicit"
    # zero-copy external
    n["coordsets/coords/values/x"].set_external(coords_xyz[:,0])
    n["coordsets/coords/values/y"].set_external(coords_xyz[:,1])
    n["coordsets/coords/values/z"].set_external(coords_xyz[:,2])

    # topology
    n["topologies/mesh/type"] = "unstructured"
    n["topologies/mesh/coordset"] = "coords"
    n["topologies/mesh/elements/shape"] = "hex" if mesh_is_3d else "quads"
    n["topologies/mesh/elements/connectivity"].set_external(conn)

    # fields
    nverts = coords_xyz.shape[0]
    for name, arr in fields.items():
        add_field(n, name, arr, nverts)

    # state
    print(f"Domain {domain_id} time={time_val} cycle={cycle_val}")
    n["state/time"] = float(time_val)
    n["state/cycle"] = int(cycle_val)
    n["state/domain_id"] = int(domain_id)
    return n

def default_actions(out_dir: str, field: str) -> conduit.Node:
    actions = conduit.Node()
    add_scene = actions.append()
    add_scene["action"] = "add_scenes"
    scene = add_scene["scenes/s1"]
    scene["plots/p1/type"] = "pseudocolor"
    scene["plots/p1/field"] = field
    # Write per-cycle images
    scene["image_prefix"] = os.path.join(out_dir, "render_%04d")
    return actions

def _arr_info(a):
    if a is None:
        return "None"
    # flagsobj exposes attributes like .c_contiguous; it isn't a dict
    try:
        c_contig = bool(getattr(a, "flags", None).c_contiguous)  # True/False
    except Exception:
        c_contig = True
    shape   = getattr(a, "shape", ())
    size    = getattr(a, "size", 0)
    dtype   = getattr(a, "dtype", None)
    strides = getattr(a, "strides", None)
    return f"shape={shape} size={size} dtype={dtype} c_contig={c_contig} strides={strides}"


# ---------------------------
# Helpers to validate data
# ---------------------------

def validate_blueprint_payload(rank: int,
                               step: int,
                               coords_xyz: np.ndarray,
                               conn: np.ndarray,
                               blockDims: Tuple[int,int,int],
                               my_count: int,
                               totalBlockSize: int,
                               mesh_is_3d: bool,
                               fields: Dict[str, np.ndarray]) -> None:
    nx, ny, nz = blockDims
    expected_nverts = my_count * totalBlockSize
    print(f"[rank {rank}] STEP {step} — expected_nverts={expected_nverts} "
          f"(my_blocks={my_count}, totalBlockSize={totalBlockSize})")

    # Coords
    assert coords_xyz.ndim == 2 and coords_xyz.shape[1] == 3, \
        f"coords_xyz must be (N,3), got {_arr_info(coords_xyz)}"
    assert coords_xyz.shape[0] == expected_nverts, \
        f"coords rows {coords_xyz.shape[0]} != expected_nverts {expected_nverts}"
    assert coords_xyz.dtype == np.float32, f"coords dtype should be float32, got {coords_xyz.dtype}"
    print(f"[rank {rank}] coords: {_arr_info(coords_xyz)} "
          f"min=({_safe_min(coords_xyz[:,0])},{_safe_min(coords_xyz[:,1])},{_safe_min(coords_xyz[:,2])}) "
          f"max=({_safe_max(coords_xyz[:,0])},{_safe_max(coords_xyz[:,1])},{_safe_max(coords_xyz[:,2])})")

    # Connectivity
    assert conn.ndim == 1, f"connectivity must be 1D, got {_arr_info(conn)}"
    if mesh_is_3d:
        cells_per_block = (nx-1)*(ny-1)*(nz-1)
        expected_ncells = my_count * cells_per_block
        assert conn.size == expected_ncells * 8, \
            f"hex connectivity size {conn.size} != {expected_ncells}*8"
    else:
        cells_per_block = (nx-1)*(ny-1)
        expected_ncells = my_count * cells_per_block
        assert conn.size == expected_ncells * 4, \
            f"quad connectivity size {conn.size} != {expected_ncells}*4"
    assert conn.dtype in (np.int32, np.int64), f"connectivity dtype must be int32/int64, got {conn.dtype}"
    print(f"[rank {rank}] conn: {_arr_info(conn)} "
          f"first10={conn[:min(10, conn.size)]}")

    # Fields
    for name, arr in fields.items():
        if arr is None:
            print(f"[rank {rank}] field '{name}' is None")
            continue
        print(f"[rank {rank}] field '{name}': {_arr_info(arr)}")
        if name == "Velocity":
            assert arr.size == expected_nverts * 3, \
                f"Velocity size {arr.size} != 3*nverts {expected_nverts*3}"
            assert arr.dtype == np.float32
        else:
            # scalars (including Velocity Magnitude)
            assert arr.size == expected_nverts, \
                f"{name} size {arr.size} != nverts {expected_nverts}"
            assert arr.dtype == np.float32

def _safe_min(a): 
    try: return float(np.nanmin(a))
    except Exception: return None
def _safe_max(a):
    try: return float(np.nanmax(a))
    except Exception: return None

def scan_min_max(reader, step_list, comm, progress):
    """
    Iterates over all steps in step_list, reads fields, and computes
    global min/max for each field across all ranks and steps.
    Handles both scalars and 3-component vectors.
    """
    rank = comm.Get_rank()
    
    # Dictionary to hold global min/max.
    # For scalars: {name: {'type': 'scalar', 'min': val, 'max': val}}
    # For vectors: {name: {'type': 'vector', 'x': [min, max], 'y': [min, max], 'z': [min, max]}}
    global_stats = {}
    
    # Track mesh coordinates min/max
    mesh_stats = {
        'x': [float('inf'), float('-inf')],
        'y': [float('inf'), float('-inf')],
        'z': [float('inf'), float('-inf')]
    }

    scan_task = progress.add_task("Scanning Min/Max", total=len(step_list))

    for step in step_list:
        df = reader.read_timestep(step)
        fields = df["fields"]
        coords = df["coordinates"]
        
        # Update mesh stats
        # coords is (N, 3)
        cx_min, cx_max = np.nanmin(coords[:,0]), np.nanmax(coords[:,0])
        cy_min, cy_max = np.nanmin(coords[:,1]), np.nanmax(coords[:,1])
        cz_min, cz_max = np.nanmin(coords[:,2]), np.nanmax(coords[:,2])
        
        if cx_min < mesh_stats['x'][0]: mesh_stats['x'][0] = float(cx_min)
        if cx_max > mesh_stats['x'][1]: mesh_stats['x'][1] = float(cx_max)
        if cy_min < mesh_stats['y'][0]: mesh_stats['y'][0] = float(cy_min)
        if cy_max > mesh_stats['y'][1]: mesh_stats['y'][1] = float(cy_max)
        if cz_min < mesh_stats['z'][0]: mesh_stats['z'][0] = float(cz_min)
        if cz_max > mesh_stats['z'][1]: mesh_stats['z'][1] = float(cz_max)

        # We need nverts to distinguish scalars from vectors
        # We need nverts to distinguish scalars from vectors
        # coordinates is (N, 3), so nverts = coordinates.shape[0]
        nverts = df["coordinates"].shape[0]
        
        for name, arr in fields.items():
            if arr is None: continue
            
            # Determine if scalar or vector
            is_vector = (arr.size == 3 * nverts)
            
            if name not in global_stats:
                if is_vector:
                    global_stats[name] = {
                        'type': 'vector',
                        'x': [float('inf'), float('-inf')],
                        'y': [float('inf'), float('-inf')],
                        'z': [float('inf'), float('-inf')]
                    }
                else:
                    global_stats[name] = {
                        'type': 'scalar',
                        'min': float('inf'),
                        'max': float('-inf')
                    }
            
            if is_vector:
                # Extract components
                # Assuming packed [x...][y...][z...]
                vx = arr[0*nverts : 1*nverts]
                vy = arr[1*nverts : 2*nverts]
                vz = arr[2*nverts : 3*nverts]
                
                # Update X
                lx_min, lx_max = np.nanmin(vx), np.nanmax(vx)
                if lx_min < global_stats[name]['x'][0]: global_stats[name]['x'][0] = float(lx_min)
                if lx_max > global_stats[name]['x'][1]: global_stats[name]['x'][1] = float(lx_max)
                
                # Update Y
                ly_min, ly_max = np.nanmin(vy), np.nanmax(vy)
                if ly_min < global_stats[name]['y'][0]: global_stats[name]['y'][0] = float(ly_min)
                if ly_max > global_stats[name]['y'][1]: global_stats[name]['y'][1] = float(ly_max)
                
                # Update Z
                lz_min, lz_max = np.nanmin(vz), np.nanmax(vz)
                if lz_min < global_stats[name]['z'][0]: global_stats[name]['z'][0] = float(lz_min)
                if lz_max > global_stats[name]['z'][1]: global_stats[name]['z'][1] = float(lz_max)

            else:
                # Scalar
                local_min = np.nanmin(arr)
                local_max = np.nanmax(arr)
                
                if local_min < global_stats[name]['min']:
                    global_stats[name]['min'] = float(local_min)
                if local_max > global_stats[name]['max']:
                    global_stats[name]['max'] = float(local_max)
        
        progress.advance(scan_task)

    # Now reduce across all ranks
    # We'll reconstruct a clean dictionary for the final results
    final_stats = {}
    
    # Helper to reduce a min/max pair
    def reduce_pair(lmin, lmax):
        gmin = comm.allreduce(lmin, op=MPI.MIN)
        gmax = comm.allreduce(lmax, op=MPI.MAX)
        return gmin, gmax

    for name, data in global_stats.items():
        if data['type'] == 'scalar':
            gmin, gmax = reduce_pair(data['min'], data['max'])
            final_stats[name] = {'type': 'scalar', 'min': gmin, 'max': gmax}
        else:
            gx_min, gx_max = reduce_pair(data['x'][0], data['x'][1])
            gy_min, gy_max = reduce_pair(data['y'][0], data['y'][1])
            gz_min, gz_max = reduce_pair(data['z'][0], data['z'][1])
            final_stats[name] = {
                'type': 'vector',
                'x': (gx_min, gx_max),
                'y': (gy_min, gy_max),
                'z': (gz_min, gz_max)
            }
            
    # Reduce mesh stats
    final_mesh_stats = {}
    gx_min, gx_max = reduce_pair(mesh_stats['x'][0], mesh_stats['x'][1])
    gy_min, gy_max = reduce_pair(mesh_stats['y'][0], mesh_stats['y'][1])
    gz_min, gz_max = reduce_pair(mesh_stats['z'][0], mesh_stats['z'][1])
    final_mesh_stats = {
        'x': (gx_min, gx_max),
        'y': (gy_min, gy_max),
        'z': (gz_min, gz_max)
    }

    if rank == 0:
        print("\n" + "="*60)
        print("SCAN RESULTS (Global Min/Max)")
        print("="*60)
        
        print(f"{'Mesh Coordinates':<20}")
        print(f"  {'x':<18} : min={final_mesh_stats['x'][0]:.6f}, max={final_mesh_stats['x'][1]:.6f}")
        print(f"  {'y':<18} : min={final_mesh_stats['y'][0]:.6f}, max={final_mesh_stats['y'][1]:.6f}")
        print(f"  {'z':<18} : min={final_mesh_stats['z'][0]:.6f}, max={final_mesh_stats['z'][1]:.6f}")
        print("-" * 60)

        for name, info in final_stats.items():
            if info['type'] == 'scalar':
                print(f"{name:<20} : min={info['min']:.6f}, max={info['max']:.6f}")
            else:
                print(f"{name:<20} : (Vector)")
                print(f"  {'x':<18} : min={info['x'][0]:.6f}, max={info['x'][1]:.6f}")
                print(f"  {'y':<18} : min={info['y'][0]:.6f}, max={info['y'][1]:.6f}")
                print(f"  {'z':<18} : min={info['z'][0]:.6f}, max={info['z'][1]:.6f}")
        print("="*60 + "\n")


# ---------------------------
# Main pipeline
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Nek5000 -> Ascent (MPI) Python reader")
    parser.add_argument("nek5000", help="Path to .nek5000 control file")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars and concise aggregated reporting")
    parser.add_argument("--steps", default=None,
                        help="Range like 'start:end:stride' (default: all)")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all steps for min/max variable values (skips rendering)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    progress = make_progress(rank, enabled=not args.no_progress)
    with progress:
        read_task = progress.add_task("Read Nek file", total=1)
        reader = Nek5000Reader(args.nek5000, comm=comm)
        progress.advance(read_task) 

        # If scanning, skip Ascent setup
        if not args.scan:
            setup_task = progress.add_task("Setup Ascent", total=1)
            progress.update(setup_task, description="Setup • open Ascent")
            a = ascent.mpi.Ascent()
            opts = conduit.Node()
            opts["mpi_comm"] = MPI.COMM_WORLD.py2f()
            opts["exceptions"] = "forward"
            a.open(opts)
            progress.advance(setup_task)

        # build step list
        if args.steps:
            seg = [int(x) for x in args.steps.split(":")]
            step_list = [seg[0]] if len(seg) == 1 else list(range(seg[0], seg[1])) if len(seg)==2 else list(range(seg[0], seg[1], seg[2]))
        else:
            step_list = reader.get_timestep_list()

        # cache geometry
        coords_cached = None
        conn_cached = None

        if args.scan:
            scan_min_max(reader, step_list, comm, progress)
            return

        steps_task = progress.add_task(f"Timesteps (total={len(step_list)})", total=len(step_list))

        for step in step_list:
            per_step = progress.add_task(f"[white]Step {step}", total=4)
            df = reader.read_timestep(step)
            progress.advance(per_step)  # read tags/time done

            # Geometry (only once)
            global numBlocks_global
            numBlocks_global = df["metadata"]["num_blocks_global"]
            mesh_src = df
            coords = df["coordinates"]
            conn = df["connectivity"]   
            fields = df["fields"]
            
            # Publish + render
            #progress.update(per_step, description=f"Step {step} • render cycle={cycle_val}")
            assert conn is not None
            dom = build_blueprint(coords, conn, df["metadata"]["block_dims"], df["metadata"]["mesh_is_3d"],
                                  fields, df["time"], df["cycle"], domain_id=rank)
            import time
            time.sleep(100)
            a.publish(dom)
            a.execute(conduit.Node())  # or your default_actions(...)
            progress.advance(per_step)  # render done

            progress.advance(steps_task)

    if not args.scan:
        a.close()

if __name__ == "__main__":
    main()

