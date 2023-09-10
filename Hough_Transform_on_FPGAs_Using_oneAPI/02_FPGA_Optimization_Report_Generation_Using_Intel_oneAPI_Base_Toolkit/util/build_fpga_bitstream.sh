#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
dpcpp -fintelfpga -Xshardware split/main.cpp banking/hough_transform_kernel.cpp -o bin/bitstream/hough_transform_live.fpga
