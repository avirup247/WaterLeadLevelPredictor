#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
dpcpp -fintelfpga -Xshardware src/bitstream/main.cpp src/bitstream/hough_transform_kernel.cpp -o bin/bitstream/hough_transform_live.fpga
