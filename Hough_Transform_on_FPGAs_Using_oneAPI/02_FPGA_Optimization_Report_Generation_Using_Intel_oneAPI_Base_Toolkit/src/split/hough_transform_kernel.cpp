//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "hough_transform_kernel.hpp"

class Hough_transform_kernel;

void RunKernel(char pixels[], short accumulators[])
{
    event queue_event;
    auto my_property_list = property_list{sycl::property::queue::enable_profiling()};

    //Buffer setup: The SYCL buffer creation expects a type of sycl:: range for the size
    range<1> num_pixels{IMAGE_SIZE};
    range<1> num_accumulators{THETAS*RHOS*2};
    range<1> num_table_values{180};

    //Create the buffers which will pass data between the host and FPGA
    buffer<char, 1> pixels_buf(pixels, num_pixels);
    buffer<short, 1> accumulators_buf(accumulators,num_accumulators);
    buffer<float, 1> sin_table_buf(sinvals,num_table_values);
    buffer<float, 1> cos_table_buf(cosvals,num_table_values);
  
    // Device selection: Explicitly compile for the FPGA_EMULATOR or FPGA
    #if defined(FPGA_EMULATOR)
        INTEL::fpga_emulator_selector device_selector;
    #else
        INTEL::fpga_selector device_selector;
    #endif

    try {
    
        queue device_queue(device_selector,NULL,my_property_list);
        platform platform = device_queue.get_context().get_platform();
        device my_device = device_queue.get_device();
        std::cout << "Platform name: " <<  platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " <<  my_device.get_info<sycl::info::device::name>().c_str() << std::endl;

        //Submit device queue 
        queue_event = device_queue.submit([&](sycl::handler &cgh) {    
          //Create accessors
          auto _pixels = pixels_buf.get_access<sycl::access::mode::read>(cgh);
          auto _sin_table = sin_table_buf.get_access<sycl::access::mode::read>(cgh);
          auto _cos_table = cos_table_buf.get_access<sycl::access::mode::read>(cgh);
          auto _accumulators = accumulators_buf.get_access<sycl::access::mode::read_write>(cgh);

          //Call the kernel
          cgh.single_task<class Hough_transform_kernel>([=]() {
            for (uint y=0; y<HEIGHT; y++) {
              for (uint x=0; x<WIDTH; x++){
                unsigned short int increment = 0;
                if (_pixels[(WIDTH*y)+x] != 0) {
                  increment = 1;
                } else {
                  increment = 0;
                }
                for (int theta=0; theta<THETAS; theta++){
                  int rho = x*_cos_table[theta] + y*_sin_table[theta];
                  _accumulators[(THETAS*(rho+RHOS))+theta] += increment;
                }
              }
            }
       
          });
      
        });
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code
        std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

        // Most likely the runtime could not find FPGA hardware!
        if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
          std::cout << "If you are targeting an FPGA, ensure that your "
                       "system has a correctly configured FPGA board.\n";
          std::cout << "If you are targeting the FPGA emulator, compile with "
                       "-DFPGA_EMULATOR.\n";
        }
        std::terminate();
    }

    // Report kernel execution time and throughput
    cl_ulong t1_kernel = queue_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    cl_ulong t2_kernel = queue_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    double time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;
}
