# Nek5000 Ascent Reader

This repository contains a parallel MPI-enabled reader that converts Nek5000 `.nek5000`/`.fld` output into Conduit/Ascent domains and publishes them for in-situ rendering.

Files of interest
- `nek5000_ascent_reader.py` — main MPI reader and publisher (this is the script invoked by `run.sh`).
- `progress_test.py` — small MPI test harness that simulates multiple ranks and demonstrates the aggregated progress behavior.
- `run-aurora.sh` — example batch wrapper (loads modules and launches `nek5000_ascent_reader.py` via `mpiexec`).
  Note: `run-aurora.sh` is an example run script targeting the Aurora environment (ALCF); it shows module loading and a sample mpiexec invocation.

Requirements / dependencies
- Python 3.8+
- mpi4py
- numpy
- Conduit + Ascent Python bindings (if using publishing/rendering)
- rich (optional, used for progress bar UI)

On your system you may have these available via modules (example shown in `run.sh`) or via pip/conda.

Quick start (interactive test)
1. Ensure dependencies are installed or loaded (e.g., via your cluster modules):
  - `mpi4py`, `numpy`, `rich` (optional), Conduit/Ascent (if publishing)

2. Run the test harness locally with a few ranks:
```bash
mpiexec -n 4 python progress_test.py
```
This emits compact aggregated status updates as simulated ranks report progress.

Run the real reader (batch example)
- `run.sh` is an example PBS wrapper that loads modules and invokes the reader with `mpiexec`. Adjust `NNODES`, `NRANKS`, and input path as needed.

Command-line arguments for `nek5000_ascent_reader.py`
This section describes the command-line options accepted by `nek5000_ascent_reader.py`.

- `nek5000` (positional)
  - Path to the `.nek5000` control file. This file must include a `filetemplate:` entry that points to the per-step `.fld` files (e.g., `data%05d.fld`).

- `--steps` (default: all)
  - A range specification indicating which steps to process. Format: `start[:end[:stride]]`. Examples:
    - `100` — process only step 100
    - `100:110` — process steps 100..109
    - `100:200:2` — process steps 100,102,...,198

Behavioral notes
- The script partitions Nek blocks among MPI ranks via a simple block-count partitioner so each rank reads only its assigned elements.
- Endianness and precision are detected from the first file header and handled correctly when reading coordinates and variable planes.
- The script uses memory-mapped reads (`np.memmap`) for large contiguous regions for performance.

Disable progress UI
-------------------
For scripted or batch runs you may want no interactive progress UI. The reader supports a `--no-progress` flag which disables the Rich progress bars and the rank-0 aggregated progress printing. The provided `run-aurora.sh` calls the reader with `--no-progress` by default so the job logs remain concise.

Example:
```bash
python nek5000_ascent_reader.py /path/to/sim.nek5000 --no-progress
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
