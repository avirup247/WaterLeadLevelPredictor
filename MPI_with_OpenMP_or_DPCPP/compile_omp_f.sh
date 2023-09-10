#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling
mpiifort -fc=ifx lab/pi_mpi_omp.f90 -fiopenmp -fopenmp-targets=spir64 -o bin/pi_mpi.x
