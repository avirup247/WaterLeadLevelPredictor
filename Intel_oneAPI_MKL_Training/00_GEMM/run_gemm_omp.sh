#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling oneMKL_introduction Module0 -- gemm with openmp - 3 of 3 omp_gemm.cpp

icx lab/omp_gemm.cpp -fsycl-device-code-split=per_kernel -DMKL_ILP64 -m64 -I/opt/intel/oneapi/mkl/latest/include -fsycl -fiopenmp -fopenmp-targets=spir64 -mllvm -vpo-paropt-use-mapper-api -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_sycl -Wl,--start-group -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lsycl -lOpenCL -lpthread -ldl -lm -lstdc++

if [ $? -eq 0 ]; then ./a.out; fi
