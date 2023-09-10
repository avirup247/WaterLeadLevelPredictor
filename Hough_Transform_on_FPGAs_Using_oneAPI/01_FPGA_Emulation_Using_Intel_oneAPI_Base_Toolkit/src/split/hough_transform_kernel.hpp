//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <vector>
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include "../../util/sin_cos_values.h"

#define WIDTH 180
#define HEIGHT 120
#define IMAGE_SIZE WIDTH*HEIGHT
#define THETAS 180
#define RHOS 217 //Size of the image diagonally: (sqrt(180^2+120^2))
#define NS (1000000000.0) // number of nanoseconds in a second

using namespace sycl;

void RunKernel(char pixels[], short accumulators[]);
