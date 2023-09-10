//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

// # The following project performs matrix multiplication using oneMKL / DPC++ with Unified Shared Memory (USM)
// # We will execute the simple operation A * B = C
// # The matrix B is set equal to the identity matrix such that A * B = A * I
// # After performing the computation, we will verify A * I = C -> A = C

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

    //### Step 1 - Create a queue with default selector.
    queue q;
    device my_device = q.get_device();
    std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";

    //### Step 2 - Create a sycl event and allocate USM
    //# The later execution of the gemm operation is tied to this event
    //# The gemm operation will also make use of a vector of sycl events we can call 'gemm_dependencies'
    
    sycl::event gemm_done;
    std::vector<sycl::event> gemm_dependencies;
    
    //# Here, we allocate USM pointers for each matrix, using the special 'malloc_shared' function
    //# Make sure to template the function with the correct precision, and pass in our queue to the function call
    
    float *A_usm = sycl::malloc_shared<float>(m * k, q);
    float *B_usm = sycl::malloc_shared<float>(k * n, q);
    float *C_usm = sycl::malloc_shared<float>(m * n, q);

    //# define matrix A as the 3x3 matrix
    //# {{ 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A_usm[i*m+j] = (float)(i*m+j) + 1.0;
        }
    }
    
    //# define matrix B as the identity matrix
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) B_usm[i*k+j] = 1.0;
            else B_usm[i*k+j] = 0.0;
        }
    }
    
    //# initialize C as a 0 matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C_usm[i*m+j] = 0.0;
        }
    }

    //### Step 3 - Execute gemm operation.
    //# Here, we fill in the familiar parameters for the gemm operation.
    //# However, we must also pass in the queue as the first parameter.
    //# We must also pass in our list of dependencies as the final parameter.
    //# We are also passing in our USM pointers as opposed to a buffer or raw data pointer.
    
    gemm_done = mkl::blas::gemm(q, transA, transB, m, n, k, alpha, A_usm, ldA, B_usm, ldB, beta, C_usm, ldC, gemm_dependencies);

    //# We must now wait for the given event to finish before accessing any data involved in the operation
    //# Otherwise, we may access data before the operation has completed, or before it has been returned to the host
    gemm_done.wait();

    int status = 0;

    //# verify C matrix using USM data
    std::cout << "\n";
    std::cout << "C = \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A_usm[i*m+j] != C_usm[i*m+j]) status = 1;
            std::cout << C_usm[i*m+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    //# free usm pointers
    sycl::free(A_usm, q);
    sycl::free(B_usm, q);
    sycl::free(C_usm, q);

    status == 0 ? std::cout << "Verified: A = C\n" : std::cout << "Failed: A != C\n";
    return status;
}
