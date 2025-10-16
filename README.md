# 🌀 nek2ascent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![MPI](https://img.shields.io/badge/MPI-enabled-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

**A parallel MPI-enabled reader and publisher connecting Nek5000 output to Ascent for in situ visualization and analysis.**

---

## Overview

**`nek2ascent`** provides a streamlined bridge between **Nek5000** simulation data (`.nek5000` / `.fld` files) and **Ascent**’s in situ visualization framework.  
It converts Nek5000 field data into **Conduit Blueprint** domains and publishes them directly to Ascent for distributed, real-time rendering—eliminating the need for heavy file I/O or post-processing.

---

## 🔧 Files of Interest

- **`nek2ascent.py`** — Main MPI reader and publisher script (invoked by `run-aurora.sh`).
- **`run-aurora.sh`** — Example batch wrapper targeting the Aurora (ALCF) environment.  
  Includes module loading and a sample `mpiexec` invocation.

---

## 🧩 Requirements

- Python 3.8+
- `mpi4py`
- `numpy`
- **Conduit + Ascent** Python bindings (required for publishing/rendering)
- `rich` *(optional)* — for terminal progress bar UI

> On HPC systems, most dependencies can be loaded via environment modules (see `run-aurora.sh` for an example).  
> Alternatively, install them via `pip` or `conda`.

---

## 🚀 Quick Start

### 1. Verify dependencies
Ensure all required modules or packages are available:
```bash
module load mpi4py conduit ascent
```
or
```bash
pip install mpi4py numpy rich
```

## 🖥️ Running the Reader
For production runs, use the provided batch wrapper:
```bash 
run-aurora.sh
```

This script:
- Loads necessary modules on ALCF’s Aurora system.
- Launches the reader via mpiexec.
- Passes `--no-progress` to keep job logs concise.
- Adjust NNODES, NRANKS, and input paths to match your setup.

## ⚙️ Command-Line Arguments
`nek2ascent.py` accepts the following options:
| Argument | Description |
|-----------|-------------|
| **`nek5000`** *(positional)* | Path to the `.nek5000` control file. Must include a `filetemplate:` entry pointing to the per-step `.fld` files (e.g., `data%05d.fld`). |
| **`--steps`** | Range of steps to process. Format: `start[:end[:stride]]`. <br>Examples: <br>• `100` → process only step 100 <br>• `100:110` → steps 100–109 <br>• `100:200:2` → steps 100, 102, …, 198 |
| **`--no-progress`** | Disables interactive progress bars and rank-0 aggregated printing for clean batch output. |


## 💡 Implementation Notes
- Parallel data partitioning assigns Nek blocks evenly across MPI ranks.
- Endianness and precision are automatically detected and handled.
- Large arrays are read efficiently using numpy.memmap.
- Publishing leverages Ascent’s Python API and Conduit Blueprints for structured domain communication.

## 🧭 Example (Batch Mode)
```bash
python nek2ascent_reader.py /path/to/simulation.nek5000 --no-progress
```

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


