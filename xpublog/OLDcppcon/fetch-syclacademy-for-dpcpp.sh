#!/bin/bash

cd ~
rm -Rf ~/syclacademy
cd ~
git clone --recursive https://github.com/codeplaysoftware/syclacademy.git
cd ~/syclacademy
mkdir build ; cd build
module unload computeCPP
module unload hipSYCL
