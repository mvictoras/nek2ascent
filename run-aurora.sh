#!/bin/bash -l
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
# Example Aurora run wrapper (ALCF). Usage:
#   ./run-aurora.sh <path/to/sim.nek5000>
#
#PBS -A <your_ProjectName>
#PBS -N <your_JobName>
#PBS -l walltime=<requested_walltime_value>
#PBS -l filesystems=<requested_fs1:requested_fs2>
#PBS -k doe
#PBS -l place=scatter
#PBS -q <requested_Queue>

module reset
module use /soft/modulefiles
module load ascent
module load python
module load py-pip
module load py-mpi4py
module load py-numpy
module load py-rich/14.1.0

# Work around MPI library clashes on Aurora. Two system issues combine here:
#   1. The ascent module ships a misbuilt zlib (/soft/visualization/.../zlib-
#      1.3.1/lib/libz.so.1) whose DT_NEEDED includes libmpi.so.12. Anything
#      that imports Python's zlib module (pytest, gzip, ...) loads it, and it
#      drags in libmpi from whichever MPI dir comes first in LD_LIBRARY_PATH
#      (Intel oneAPI precedes the spack MPICH dir).
#   2. conduit (from the ascent module) needs MPICH's libmpifort.so.12
#      (symbol MPII_F_TRUE); binding it to Intel's copy is fatal.
# Preloading MPICH's libmpi and libmpifort pins the sonames to MPICH for the
# whole process, independent of import order. The path is resolved from the
# mpi4py install itself so this survives spack/CPE version bumps.
MPI4PY_EXT=$(python -c "import mpi4py, glob, os; print(glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), 'MPI*.so'))[0])" 2>/dev/null)
MPICH_LIB=$(ldd "${MPI4PY_EXT}" 2>/dev/null | awk '/libmpi\.so/{print $3}' | head -1)
if [[ -n "${MPICH_LIB}" ]]; then
	MPICH_LIB=$(dirname "${MPICH_LIB}")
	_PRELOAD=""
	for _lib in libmpi.so.12 libmpifort.so.12; do
		[[ -f "${MPICH_LIB}/${_lib}" ]] && _PRELOAD="${MPICH_LIB}/${_lib}${_PRELOAD:+:${_PRELOAD}}"
	done
	if [[ -n "${_PRELOAD}" ]]; then
		export LD_PRELOAD="${_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
	fi
fi

# proxy settings
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export ftp_proxy="http://proxy.alcf.anl.gov:3128"
  export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
fi

if [ -z "$1" ]; then
	echo "Usage: $0 <path/to/sim.nek5000> [--scan]"
	echo "Example: $0 /path/to/sim.nek5000 --scan"
	exit 1
fi

PROGRESS=""
export TZ='/usr/share/zoneinfo/US/Central'
if [[ "$PBS_ENVIRONMENT"  != "PBS_INTERACTIVE" ]]; then
	# Fall back to the current directory when run directly on a compute node
	# (i.e. outside of a PBS job).
	if [[ -n "${PBS_O_WORKDIR}" ]]; then
		cd ${PBS_O_WORKDIR}
	fi
	PROGRESS="--no-progress"
fi

# nek5000reader is a pure-Python package from PyPI; install it into the
# user's ~/.local so each user gets their own copy (no shared venv, no
# writes to the read-only module tree). The conduit/ascent bindings come
# from the system modules above and must not be shadowed by a venv.
if ! python -c "import nek5000reader" >/dev/null 2>&1; then
	python -m pip install --user nek5000reader
fi

NEK5000_PATH=""
SCAN_ARG=""

for arg in "$@"; do
    if [[ "$arg" == "--scan" ]]; then
        SCAN_ARG="--scan"
    elif [[ "$arg" == *.nek5000 ]]; then
        NEK5000_PATH="$arg"
    fi
done

if [ -z "$NEK5000_PATH" ]; then
    echo "Error: No .nek5000 file specified."
    echo "Usage: $0 <path/to/sim.nek5000> [--scan]"
    exit 1
fi

if [[ -n "${PBS_NODEFILE}" ]]; then
	NNODES=$(wc -l < "${PBS_NODEFILE}")
else
	# Not running under PBS (e.g. interactive run on a compute node).
	NNODES=1
fi
NRANKS=6 # Number of MPI ranks to spawn per node
NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind=depth gpu_tile_compact.sh python nek2ascent.py ${SCAN_ARG} ${NEK5000_PATH} ${PROGRESS}


