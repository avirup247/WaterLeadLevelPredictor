#!/bin/bash

cd ~
rm -Rf ~/syclacademy
cd ~
git clone --recursive https://github.com/codeplaysoftware/syclacademy.git
cd ~/syclacademy
mkdir build ; cd build
module unload computeCPP
module load hipSYCL
cmake ../ -G'Unix Makefiles' -DSYCL_ACADEMY_USE_HIPSYCL=ON -DSYCL_ACADEMY_INSTALL_ROOT=/data/oneapi_workshop/xpublog/teach/hipSYCL
