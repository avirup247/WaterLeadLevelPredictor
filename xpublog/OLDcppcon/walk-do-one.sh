#!/bin/bash

echo "DPC++"
module purge
rm -f a.out
( set -x ; \
  dpcpp solution.cpp
)
./a.out
echo "ComputeCPP"
module purge
module load computeCPP
rm -f a.out
( set -x ; \
compute++ solution.cpp -lComputeCpp -sycl-driver -std=c++17 -DSYCL_LANGUAGE_VERSION=2020 -no-serial-memop
)
./a.out
echo "hipSYCL"
module purge
module load hipSYCL
rm -f a.out
( set -x ; \
syclcc -O2 -std=c++17 solution.cpp
)
./a.out
module purge





