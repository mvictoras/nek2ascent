#!/usr/bin/env python3
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

# ---------------------------
# Helpers for ASCII token read
# ---------------------------

def _read_ascii_token(f) -> str:
    # Skip whitespace
    ch = f.read(1)
    while ch and ch in b" \t\r\n":
        ch = f.read(1)
    if not ch:
        return ""
    # Read until whitespace
    buf = [ch]
    ch = f.read(1)
    while ch and ch not in b" \t\r\n":
        buf.append(ch)
        ch = f.read(1)
    return b"".join(buf).decode("ascii", errors="ignore")

def _peek(f) -> int:
    pos = f.tell()
    b = f.read(1)
    f.seek(pos)
    return b[0] if b else -1

def _skip_spaces(f):
    b = _peek(f)
    while b == ord(' '):
        f.read(1)
        b = _peek(f)

def _skip_digits(f):
    b = _peek(f)
    while b >= ord('0') and b <= ord('9'):
        f.read(1)
        b = _peek(f)

# ---------------------------
# Nek5000 header + tags
# ---------------------------

def parse_nek5000_control(path: str) -> Dict:
    """
    Parse the .nek5000 control file for:
      filetemplate: printf-style template (e.g., prefix%05d.fld)
      firsttimestep: int
      numtimesteps: int
    """
    out = {"filetemplate": None, "firsttimestep": None, "numtimesteps": None}
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        toks = fp.read().replace("\r", "\n").split()
    # Very simple tag-based parse
    for i, t in enumerate(toks):
        lt = t.lower()
        if lt.startswith("filetemplate:") and i + 1 < len(toks):
            out["filetemplate"] = toks[i + 1]
        elif lt.startswith("firsttimestep:") and i + 1 < len(toks):
            out["firsttimestep"] = int(toks[i + 1])
        elif lt.startswith("numtimesteps:") and i + 1 < len(toks):
            out["numtimesteps"] = int(toks[i + 1])
    if out["filetemplate"] is None:
        raise RuntimeError("Missing 'filetemplate:' in .nek5000 file.")
    if out["firsttimestep"] is None:
        raise RuntimeError("Missing 'firsttimestep:' in .nek5000 file.")
    if out["numtimesteps"] is None:
        raise RuntimeError("Missing 'numtimesteps:' in .nek5000 file.")
    # Make absolute if necessary
    if not os.path.isabs(out["filetemplate"]):
        out["filetemplate"] = os.path.join(os.path.dirname(path), out["filetemplate"])
    return out

def build_step_filename(fmt: str, step: int, dir_index: int = 0) -> str:
    """
    Supports Nek5000 templates with one or two printf specifiers.
      e.g. "data%05d.fld"            -> fmt % step
            "turbPipe%01d.f%05d"     -> fmt % (dir_index, step)  (VTK passes 0 for dir_index)
    Falls back to simple concatenation if there are no specifiers.
    """
    if "%" not in fmt:
        return f"{fmt}{step}"
    # try single-arg first
    try:
        return fmt % step
    except TypeError:
        pass
    # try (dir_index, step) like VTK does
    try:
        return fmt % (dir_index, step)
    except TypeError as e:
        raise RuntimeError(
            f"Unsupported filetemplate '{fmt}'. Expected 0, 1, or 2 printf specifiers."
        ) from e


def read_basic_header_and_endian(dfname: str) -> Tuple[int, Tuple[int,int,int], int, bool]:
    """
    Reads the initial ASCII header (#std, precision, blockDims, ..., numBlocks),
    probes endian using the float at offset 132, and returns:
      precision_bytes (4 or 8), (nx, ny, nz), numBlocks, swapEndian(bool)
    """
    with open(dfname, "rb") as f:
        # Tokens: "#std", precision, nx, ny, nz, (some token), numBlocks
        tag = _read_ascii_token(f)
        if tag != "#std":
            raise RuntimeError(f"{dfname}: expected '#std' at start, got '{tag}'")
        precision = int(_read_ascii_token(f))
        nx = int(_read_ascii_token(f))
        ny = int(_read_ascii_token(f))
        nz = int(_read_ascii_token(f))
        _ = _read_ascii_token(f)  # "blocks per file" label or similar
        numBlocks = int(_read_ascii_token(f))

        # Probe endian via float at offset 132
        f.seek(132, 0)
        b = f.read(4)
        if len(b) != 4:
            raise RuntimeError("Could not read endian probe.")
        test_le = struct.unpack("<f", b)[0]
        test_be = struct.unpack(">f", b)[0]
        # VTK checks ~6.5..6.6
        def ok(v): return 6.5 < v < 6.6
        if ok(test_le):
            swap = False  # file little-endian matches our unpack
        elif ok(test_be):
            swap = True
        else:
            # Fallback: assume native little endian; still proceed
            swap = False
    return precision, (nx, ny, nz), numBlocks, swap

import os, re

def _last_int_in_string(s: str) -> Optional[int]:
    m = re.findall(r'(\d+)', s)
    return int(m[-1]) if m else None

def read_time_and_tags(dfname: str) -> Tuple[float, int, str, bool]:
    """
    Reads (time, cycle, tags_string, has_mesh) from the ASCII/tag section.
    Falls back to parsing the step from the filename if cycle == 0.
    """
    with open(dfname, "rb") as f:
        # Skip the first 7 tokens, then read time, cycle, one token (dummy),
        # then skip spaces + digits (num directories), then read 32 raw bytes as tags.
        for _ in range(7):
            _ = _read_ascii_token(f)
        t_str = _read_ascii_token(f)
        c_str = _read_ascii_token(f)
        _ = _read_ascii_token(f)
        _skip_spaces(f)
        _skip_digits(f)
        tags = f.read(32)
        tags = (tags or b"").decode("ascii", errors="ignore")
        has_mesh = ("X" in tags)

        # parse
        t = float(t_str) if t_str and any(ch.isdigit() for ch in t_str) else 0.0
        c = int(c_str) if c_str and c_str.strip("-+").isdigit() else 0

    # Fallback: many Nek files leave cycle at 0; use the last integer in the filename.
    if c == 0:
        base = os.path.basename(dfname)
        c_from_name = _last_int_in_string(base)
        if c_from_name is not None:
            c = c_from_name

    return t, c, tags, has_mesh


def parse_var_tags(tags: str, mesh_is_3d: bool) -> Tuple[List[str], List[int]]:
    """
    From the tags string, produce var_names and component counts.
    Adds "Velocity Magnitude" right after "Velocity" to mirror VTK.
    """
    names: List[str] = []
    lens: List[int] = []
    # Count S fields
    s_count = 0
    if "S" in tags:
        # Find 'S##' (two digits) after 'S', else default to 1
        # We scan the next two chars that look like digits:
        idx = tags.find("S")
        if idx >= 0 and idx + 2 < len(tags):
            d = "".join([c for c in tags[idx+1:idx+3] if c.isdigit()])
            s_count = int(d) if len(d) == 2 else 1
        else:
            s_count = 1
    # Velocity
    if "U" in tags:
        names.append("Velocity")
        lens.append(3 if mesh_is_3d else 3)  # we keep 3 comps; Z set to 0 in 2D
        names.append("Velocity Magnitude")
        lens.append(1)
    # Pressure
    if "P" in tags:
        names.append("Pressure")
        lens.append(1)
    # Temperature
    if "T" in tags:
        names.append("Temperature")
        lens.append(1)
    # Scalars S01...SNN
    for s in range(s_count):
        names.append(f"S{s+1:02d}")
        lens.append(1)
    return names, lens

# ---------------------------
# Partitioning and block map
# ---------------------------

def read_block_ids(dfname: str, numBlocks: int, swapEndian: bool) -> np.ndarray:
    with open(dfname, "rb") as f:
        f.seek(136, 0)  # block id list starts at 136
        arr = np.fromfile(f, dtype=np.int32, count=numBlocks)
    if arr.size != numBlocks:
        raise RuntimeError("Failed to read block IDs.")
    if swapEndian:
        arr.byteswap(inplace=True)
    return arr

def read_map_file(nekfile: str) -> Optional[np.ndarray]:
    """
    If a .map file exists alongside the .nek5000, read its element order.
    File format: first int: num_map_elements, followed by per-line entries
    where the first numeric field is the element id (0-based) -> +1 like VTK.
    """
    map_path = os.path.splitext(nekfile)[0] + ".map"
    if not os.path.exists(map_path):
        return None
    ids = []
    with open(map_path, "r", encoding="utf-8", errors="ignore") as fp:
        toks = fp.read().split()
    if not toks:
        return None
    # toks: num_elems then repeated lines; element id is first int on each line
    num_elems = int(toks[0])
    # Heuristic: the next 8 tokens per element; first is id
    # Safer approach: scan lines ignoring text; grab first int on each line.
    with open(map_path, "r", encoding="utf-8", errors="ignore") as fp:
        first = True
        for line in fp:
            if first:
                first = False
                continue
            parts = line.strip().split()
            if not parts:
                continue
            try:
                ids.append(int(parts[0]) + 1)  # VTK adds +1
            except ValueError:
                continue
    if len(ids) != num_elems:
        # fallback: ignore map
        return None
    return np.array(ids, dtype=np.int32)

def partition_blocks(numBlocks: int, comm: MPI.Comm) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (counts_per_rank, displs) for an even partition of elements.
    """
    size = comm.Get_size()
    base = numBlocks // size
    rem = numBlocks % size
    counts = np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int32)
    displs = np.zeros(size, dtype=np.int32)
    if size > 1:
        displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

# ---------------------------
# Mesh + variable reads
# ---------------------------

def total_header_size_bytes(numBlocks: int, totalBlockSize: int, comps_xyz: int, precision: int,
                            has_mesh: bool) -> int:
    base = 136 + numBlocks * 4  # 136 header + block id table
    if has_mesh:
        base += numBlocks * totalBlockSize * comps_xyz * precision
    return base

def read_coords_for_my_blocks(dfname: str,
                              my_block_positions: np.ndarray,
                              totalBlockSize: int,
                              mesh_is_3d: bool,
                              precision: int,
                              swapEndian: bool) -> np.ndarray:
    """
    Fast path: map the entire coords region once, then gather in one shot.

    Assumes a global `numBlocks_global` is set (same as your original).
    """
    comps = 3 if mesh_is_3d else 2
    nblk_local = int(my_block_positions.size)
    # Output (host, native float32)
    out = np.empty((nblk_local * totalBlockSize, 3), dtype=np.float32)

    # Byte offsets & element counts for the coordinates region
    header_bytes = 136 + numBlocks_global * 4  # 136-byte header + block-id table
    stride_elems = totalBlockSize * comps

    # Select dtype with the file's endianness
    # swapEndian == True  -> file is big endian
    # swapEndian == False -> file is little endian
    endian = ">" if swapEndian else "<"
    dt = np.dtype(endian + ("f4" if precision == 4 else "f8"))

    # Map the full coordinates region as a 2D array: (numBlocks_global, stride_elems)
    mm = np.memmap(dfname, dtype=dt, mode="r",
                   offset=header_bytes, shape=(numBlocks_global, stride_elems))

    # Vectorized gather of all my blocks (creates a dense ndarray)
    sel = mm[my_block_positions]  # shape: (nblk_local, stride_elems)

    # De-interleave into X/Y/Z in a single pass
    # Reshape output for clean slicing [block, point, coord]
    out3 = out.reshape(nblk_local, totalBlockSize, 3)
    out3[:, :, 0] = sel[:, 0:totalBlockSize]                       # X
    out3[:, :, 1] = sel[:, totalBlockSize:2*totalBlockSize]        # Y
    if mesh_is_3d:
        out3[:, :, 2] = sel[:, 2*totalBlockSize:3*totalBlockSize]  # Z
    else:
        out3[:, :, 2].fill(0.0)

    # If the file was double precision or non-native endian,
    # the assignment above casts to native float32 for you.
    return out


def build_connectivity(blockDims: Tuple[int,int,int],
                       myNumBlocks: int,
                       totalBlockSize: int,
                       mesh_is_3d: bool) -> np.ndarray:
    """
    Vectorized connectivity builder.
    - Builds one block's connectivity with NumPy.
    - Broadcast-adds a per-block vertex offset (totalBlockSize) and tiles across myNumBlocks.
    Returns a 1D int64 array suitable for Conduit Blueprint 'connectivity'.
    """
    nx, ny, nz = blockDims
    pts_per_block = np.int64(totalBlockSize)

    if mesh_is_3d:
        # Cells per block: (nx-1)*(ny-1)*(nz-1)
        i = np.arange(nx - 1, dtype=np.int64)
        j = np.arange(ny - 1, dtype=np.int64)
        k = np.arange(nz - 1, dtype=np.int64)
        # Order matches original loops: ii (slow), jj, kk (fast)
        I, J, K = np.meshgrid(i, j, k, indexing="ij")  # shapes: (nx-1, ny-1, nz-1)
        base = (K * (ny * nx) + J * nx + I).reshape(-1)

        layer = np.int64(nx * ny)
        # Hex corner pattern:
        # [p, p+1, p+nx+1, p+nx,  p+layer, p+layer+1, p+layer+nx+1, p+layer+nx]
        conn_block = np.stack(
            [
                base,
                base + 1,
                base + nx + 1,
                base + nx,
                base + layer,
                base + layer + 1,
                base + layer + nx + 1,
                base + layer + nx,
            ],
            axis=1,
        )  # (cells_per_block, 8)

        # Tile to all my blocks with vertex offsets
        offsets = (np.arange(myNumBlocks, dtype=np.int64) * pts_per_block)[:, None, None]
        conn = conn_block[None, :, :] + offsets  # (myNumBlocks, cells_per_block, 8)
        return conn.reshape(-1)

    else:
        # 2D quads per block: (nx-1)*(ny-1)
        i = np.arange(nx - 1, dtype=np.int64)
        j = np.arange(ny - 1, dtype=np.int64)
        # Order matches original loops: ii (slow), jj (fast)
        I, J = np.meshgrid(i, j, indexing="ij")  # shapes: (nx-1, ny-1)
        base = (J * nx + I).reshape(-1)

        # Quad corner pattern:
        # [p, p+1, p+nx+1, p+nx]
        conn_block = np.stack(
            [base, base + 1, base + nx + 1, base + nx], axis=1
        )  # (cells_per_block, 4)

        offsets = (np.arange(myNumBlocks, dtype=np.int64) * pts_per_block)[:, None, None]
        conn = conn_block[None, :, :] + offsets  # (myNumBlocks, cells_per_block, 4)
        return conn.reshape(-1)


def read_variables_for_my_blocks(dfname: str,
                                 var_names: List[str],
                                 var_lens: List[int],
                                 my_block_positions: np.ndarray,
                                 totalBlockSize: int,
                                 mesh_is_3d: bool,
                                 precision: int,
                                 swapEndian: bool,
                                 has_mesh: bool,
                                 numBlocks_global: int) -> Dict[str, np.ndarray]:
    """
    Faster: use a single np.memmap per variable plane and gather all my blocks at once.
    - Endianness handled by dtype ('<' or '>').
    - 2D Velocity reads Vx,Vy from file and fills Vz=0.
    - Always returns float32 arrays.
    """
    result: Dict[str, np.ndarray] = {}

    comps_vel_in_file = 3 if mesh_is_3d else 2             # what the file actually stores for U
    comps_xyz = 3 if mesh_is_3d else 2
    header_bytes = total_header_size_bytes(numBlocks_global, totalBlockSize,
                                           comps_xyz, precision, has_mesh)

    # One scalar "plane" across *all* blocks in bytes
    plane_bytes = numBlocks_global * totalBlockSize * precision

    # Choose file dtype with explicit endianness
    endian = ">" if swapEndian else "<"
    dt = np.dtype(endian + ("f4" if precision == 4 else "f8"))

    nblk_local = int(my_block_positions.size)
    nverts = nblk_local * totalBlockSize

    # We compute offsets by counting how many "planes" we've passed in the file.
    # Velocity contributes `comps_vel_in_file` planes; each scalar (P, T, S##) contributes 1 plane.
    planes_before = 0

    need_vel_mag = ("Velocity Magnitude" in var_names)
    have_velocity = False
    vel_flat = None  # will hold [Vx...][Vy...][Vz...]

    i = 0
    while i < len(var_names):
        name = var_names[i]
        ncomp = var_lens[i]

        if name == "Velocity Magnitude":
            # We'll compute it after we read Velocity
            i += 1
            continue

        if name == "Velocity":
            # Map the contiguous velocity region (all blocks) as one 2D array
            offset = header_bytes + planes_before * plane_bytes
            # shape = (numBlocks_global, totalBlockSize * comps_vel_in_file)
            mm = np.memmap(dfname, dtype=dt, mode="r",
                           offset=offset,
                           shape=(numBlocks_global, totalBlockSize * comps_vel_in_file))

            sel = mm[my_block_positions]  # (nblk_local, totalBlockSize*comps_vel_in_file)

            # Build a flat 3-comp array [Vx...][Vy...][Vz...], always 3 comps in output
            vel_flat = np.empty(nverts * 3, dtype=np.float32)
            # X
            vel_flat[0*nverts:1*nverts] = sel[:, 0:totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            # Y
            vel_flat[1*nverts:2*nverts] = sel[:, totalBlockSize:2*totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            # Z
            if mesh_is_3d:
                vel_flat[2*nverts:3*nverts] = sel[:, 2*totalBlockSize:3*totalBlockSize].reshape(-1).astype(np.float32, copy=False)
            else:
                vel_flat[2*nverts:3*nverts] = 0.0

            result["Velocity"] = vel_flat  # already flat
            have_velocity = True

            planes_before += comps_vel_in_file
            i += 1
            continue

        # Scalar variable (P, T, S##, etc.)
        offset = header_bytes + planes_before * plane_bytes
        mm = np.memmap(dfname, dtype=dt, mode="r",
                       offset=offset,
                       shape=(numBlocks_global, totalBlockSize))

        sel = mm[my_block_positions]  # (nblk_local, totalBlockSize)
        result[name] = sel.reshape(-1).astype(np.float32, copy=False)

        planes_before += 1
        i += 1

    # Compute Velocity Magnitude if requested
    if need_vel_mag:
        if not have_velocity:
            raise RuntimeError("Requested 'Velocity Magnitude' but 'Velocity' was not present.")
        vx = vel_flat[0*nverts:1*nverts]
        vy = vel_flat[1*nverts:2*nverts]
        vz = vel_flat[2*nverts:3*nverts]
        result["Velocity Magnitude"] = np.sqrt(vx*vx + vy*vy + vz*vz, dtype=np.float32)

    return result


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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    progress = make_progress(rank, enabled=not args.no_progress)
    with progress:
        setup_task = progress.add_task("Setup", total=6)

        comm.Barrier()
        progress.advance(setup_task)  # 1/6

        progress.update(setup_task, description="Setup • parse .nek5000")
        ctrl = parse_nek5000_control(args.nek5000)
        filetemplate = ctrl["filetemplate"]
        first = ctrl["firsttimestep"]
        nsteps = ctrl["numtimesteps"]
        progress.advance(setup_task)  # 2/6

        progress.update(setup_task, description="Setup • read first header")
        first_file = build_step_filename(filetemplate, first)
        if rank == 0:
            progress.log(f"[bold magenta]Rank {rank}[/] reading first file: [white]{first_file}[/]")
        precision, blockDims, numBlocks, swapEndian = read_basic_header_and_endian(first_file)
        mesh_is_3d = blockDims[2] > 1
        totalBlockSize = blockDims[0] * blockDims[1] * blockDims[2]
        progress.advance(setup_task)  # 3/6

        progress.update(setup_task, description="Setup • read block IDs / map")
        block_ids_all = read_block_ids(first_file, numBlocks, swapEndian)
        map_ids = read_map_file(args.nek5000)
        global_order = map_ids if (map_ids is not None and map_ids.size == numBlocks) else block_ids_all
        progress.advance(setup_task)  # 4/6

        progress.update(setup_task, description="Setup • partition blocks")
        counts, displs = partition_blocks(numBlocks, comm)
        my_count = int(counts[rank])
        my_ids   = global_order[displs[rank]:displs[rank]+my_count].copy()
        # build id->pos fast
        ids = block_ids_all.astype(np.int64, copy=False)
        my_ids64 = my_ids.astype(np.int64, copy=False)
        max_id = int(ids.max())
        if max_id <= 2 * ids.size:
            lut = np.full(max_id + 1, -1, dtype=np.int64)
            lut[ids] = np.arange(ids.size, dtype=np.int64)
            my_positions = lut[my_ids64]
            if np.any(my_positions < 0):
                raise RuntimeError("Some my_ids not found in block_ids_all")
        else:
            order = np.argsort(ids)
            sorted_ids = ids[order]
            loc = np.searchsorted(sorted_ids, my_ids64)
            if not np.all(sorted_ids[loc] == my_ids64):
                raise RuntimeError("Some my_ids not found in block_ids_all")
            my_positions = order[loc]
        my_positions = my_positions.astype(np.int64, copy=False)
        progress.advance(setup_task)  # 5/6

        progress.update(setup_task, description="Setup • open Ascent")
        a = ascent.mpi.Ascent()
        opts = conduit.Node()
        opts["mpi_comm"] = MPI.COMM_WORLD.py2f()
        opts["exceptions"] = "forward"
        a.open(opts)
        progress.advance(setup_task)  # 6/6

        # build step list
        if args.steps:
            seg = [int(x) for x in args.steps.split(":")]
            step_list = [seg[0]] if len(seg) == 1 else list(range(seg[0], seg[1])) if len(seg)==2 else list(range(seg[0], seg[1], seg[2]))
        else:
            step_list = list(range(first, first + nsteps))

        # cache geometry
        coords_cached = None
        conn_cached = None

        steps_task = progress.add_task(f"Timesteps (total={len(step_list)})", total=len(step_list))

        for step in step_list:
            per_step = progress.add_task(f"[white]Step {step}", total=4)
            df = build_step_filename(filetemplate, step)
            time_val, cycle_val, tags, has_mesh = read_time_and_tags(df)
            progress.advance(per_step)  # read tags/time done

            # Geometry (only once)
            global numBlocks_global
            numBlocks_global = numBlocks
            if coords_cached is None:
                progress.update(per_step, description=f"Step {step} • read coords/conn")
                mesh_src = df if has_mesh else first_file
                coords = read_coords_for_my_blocks(mesh_src, my_positions, totalBlockSize,
                                                   mesh_is_3d, precision, swapEndian)
                conn = build_connectivity(blockDims, my_count, totalBlockSize, mesh_is_3d)
                if conn.dtype != np.int32 and int(conn.max()) < np.iinfo(np.int32).max:
                    conn = conn.astype(np.int32, copy=False)
                coords_cached, conn_cached = coords, conn
            else:
                coords, conn = coords_cached, conn_cached
            progress.advance(per_step)  # geometry done

            # Variables
            progress.update(per_step, description=f"Step {step} • read variables")
            fields = read_variables_for_my_blocks(df, var_names=parse_var_tags(tags, mesh_is_3d)[0],
                                                  var_lens=parse_var_tags(tags, mesh_is_3d)[1],
                                                  my_block_positions=my_positions,
                                                  totalBlockSize=totalBlockSize,
                                                  mesh_is_3d=mesh_is_3d,
                                                  precision=precision,
                                                  swapEndian=swapEndian,
                                                  has_mesh=has_mesh,
                                                  numBlocks_global=numBlocks)
            progress.advance(per_step)  # variables done
            # Publish + render
            progress.update(per_step, description=f"Step {step} • render cycle={cycle_val}")
            assert conn is not None
            dom = build_blueprint(coords, conn, blockDims, mesh_is_3d,
                                  fields, time_val, cycle_val, domain_id=rank)
            a.publish(dom)
            a.execute(conduit.Node())  # or your default_actions(...)
            progress.advance(per_step)  # render done

            progress.advance(steps_task)

    a.close()

if __name__ == "__main__":
    main()

