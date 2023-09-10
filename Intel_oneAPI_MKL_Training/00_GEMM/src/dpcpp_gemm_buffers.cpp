//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

//# The following project performs matrix multiplication using oneMKL / DPC++ with buffers.
//# We will execute the simple operation A * B = C
//# The matrix B is set equal to the identity matrix such that A * B = A * I
//# After performing the computation, we will verify A * I = C -> A = C

using namespace sycl;
namespace mkl = oneapi::mkl;  //# shorten mkl namespace

int main() {

    //# dimensions
    
    int m = 3, n = 3, k = 3;
    
    //# leading dimensions
    
    int ldA = 3, ldB = 3, ldC = 3;
    
    //# scalar multipliers
    
    float alpha = 1.0, beta = 1.0;
    
    //# transpose status of matrices
    
    mkl::transpose transA = mkl::transpose::nontrans;
    mkl::transpose transB = mkl::transpose::nontrans;
    
    //# matrix data
    
    std::vector<float> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<float> B = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<float> C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    //### Step 1 - Create a queue with default selector.
    
    queue q;
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    //### Step 2 - Create buffers to hold our matrix data.
    //# Buffer objects can be constructed given a container
    //# Observe the creation of buffers for matrices A and B.
    //# Try and create a third buffer for matrix C called C_buffer.
    //# The solution is shown in the hidden cell below.
    
    buffer A_buffer(A);
    buffer B_buffer(B);
    /* define C_buffer here */
    
    //### Step 3 - Execute gemm operation.
    //# Here, we need only pass in our queue and other familiar matrix multiplication parameters.
    //# This includes the dimensions and data buffers for matrices A, B, and C.
    
    mkl::blas::gemm(q, transA, transB, m, n, k, alpha, A_buffer, ldA, B_buffer, ldB, beta, C_buffer, ldC);


    //### Step 6 - Observe creation of accessors to retrieve data from A_buffer and C_buffer.
    
    host_accessor A_acc(A_buffer, read_only);
    host_accessor C_acc(C_buffer, read_only);

    int status = 0;

    // verify C matrix using accessor to observe values held in C_buffer
    
    std::cout << std::endl;
    std::cout << "C = " << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A_acc[i*m+j] != C_acc[i*m+j]) status = 1;
            std::cout << C_acc[i*m+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    status == 0 ? std::cout << "Verified: A = C" << std::endl : std::cout << "Failed: A != C" << std::endl;
    return status;
}
