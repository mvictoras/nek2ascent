#!/bin/bash -l
# Example Aurora run wrapper (ALCF). Usage:
#   ./run-aurora.sh <path/to/sim.nek5000>

module reset
module use ~/MODULEFILES
module load ascent
module load python
module load py-mpi4py
module load py-numpy
module load py-rich/13.7.1-5aah7mo

if [ -z "$1" ]; then
	echo "Usage: $0 <path/to/sim.nek5000>"
	echo "Example: $0 /path/to/sim.nek5000"
	exit 1
fi

export TZ='/usr/share/zoneinfo/US/Central'
cd ${PBS_O_WORKDIR}

NEK5000_PATH="$1"

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=6 # Number of MPI ranks to spawn per node
NDEPTH=1 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=1 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)

NTOTRANKS=$(( NNODES * NRANKS ))

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind=depth gpu_tile_compact.sh python nek5000_ascent_reader.py "${NEK5000_PATH}" "$@"


