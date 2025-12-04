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
module load py-mpi4py
module load py-numpy
module load py-rich/13.7.1-5aah7mo

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
	cd ${PBS_O_WORKDIR}
	PROGRESS="--no-progress"
fi

if [[ -d "venv" ]]; then
	source venv/bin/activate
else
	python -m venv venv
	source venv/bin/activate
	pip install nek5000reader
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

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=6 # Number of MPI ranks to spawn per node
NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind=depth gpu_tile_compact.sh python nek2ascent.py ${SCAN_ARG} ${NEK5000_PATH} ${PROGRESS}


