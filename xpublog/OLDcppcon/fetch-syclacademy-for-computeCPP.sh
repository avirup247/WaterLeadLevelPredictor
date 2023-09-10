#!/bin/bash

cd ~
rm -Rf ~/syclacademy
cd ~
git clone --recursive https://github.com/codeplaysoftware/syclacademy.git
cd ~/syclacademy
mkdir build ; cd build
module unload hipSYCL
module load computeCPP
cmake ../ -G'Unix Makefiles' -DSYCL_ACADEMY_USE_COMPUTECPP=ON -DSYCL_ACADEMY_INSTALL_ROOT=/data/oneapi_workshop/xpublog/teach/ComputeCpp-CE-2.7.0-x86_64-linux-gnu -DCMAKE_MODULE_PATH=/data/oneapi_workshop/xpublog/teach/computecpp-sdk/cmake/Modules -DOpenCL_INCLUDE_DIR=/data/oneapi_workshop/xpublog/teach/OpenCL-Headers -DOpenCL_LIBRARY=/data/oneapi_workshop/xpublog/teach/libOpenCL.so.1

