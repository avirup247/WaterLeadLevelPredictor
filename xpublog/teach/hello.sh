#!/bin/bash
module unload computeCPP
module unload hipSYCL
(set -x ; \
cp -p /data/oneapi_workshop/xpublog/teach/hello.cpp .
)

echo "++++++++++++++++++"
echo "DPC++"
echo "++++++++++++++++++"
echo "try on DEFAULT device..."
rm -f a.out
(set -x ; \
dpcpp hello.cpp
)
(set -x ; \
./a.out
)
echo "try on CPU device..."
rm -f a.out
(set -x
dpcpp -DTryCPU hello.cpp
)
(set -x ; \
./a.out )
echo "try on GPU device..."
rm -f a.out
(set -x ; \
dpcpp -DTryGPU hello.cpp
)
(set -x ; \
./a.out )

echo "++++++++++++++++++"
echo "ComputeCPP"
echo "++++++++++++++++++"
module load computeCPP
rm -f a.out
(set -x ; \
compute++ hello.cpp -lComputeCpp -sycl-driver -std=c++17 -DSYCL_LANGUAGE_VERSION=2020 -no-serial-memop
)
(set -x ; \
./a.out
)
echo "try on CPU device..."
rm -f a.out
(set -x
compute++ -DTryCPU hello.cpp -lComputeCpp -sycl-driver -std=c++17 -DSYCL_LANGUAGE_VERSION=2020 -no-serial-memop
)
(set -x ; \
./a.out )
echo "try on GPU device..."
rm -f a.out
(set -x ; \
compute++ -DTryGPU hello.cpp -lComputeCpp -sycl-driver -std=c++17 -DSYCL_LANGUAGE_VERSION=2020 -no-serial-memop
)
(set -x ; \
./a.out )
module unload computeCPP

echo "++++++++++++++++++"
echo "Hip/SYCL"
echo "++++++++++++++++++"
module load hipSYCL
rm -f a.out
echo "try on CPU device..."
rm -f a.out
(set -x
syclcc -DTryCPU -O2 -std=c++17 hello.cpp
)
(set -x ; \
./a.out )
module unload hipSYCL


