#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling
mpiicpc -cxx=dpcpp lab/pi_mpi_dpcpp.cpp -o bin/pi_mpi.x
