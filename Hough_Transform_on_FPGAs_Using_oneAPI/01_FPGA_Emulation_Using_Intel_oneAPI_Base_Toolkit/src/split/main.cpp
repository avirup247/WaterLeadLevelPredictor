//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <vector>
#include <CL/sycl.hpp>
#include <chrono>
#include <fstream>
#include "hough_transform_kernel.hpp"

using namespace std;

void read_image(char *image_array); // Funciton for converting a bitmap to an array of pixels
class Hough_transform_kernel;

int main() {
  char pixels[IMAGE_SIZE];
  short accumulators[THETAS*RHOS*2];

  std::fill(accumulators, accumulators + THETAS*RHOS*2, 0);

  read_image(pixels); //Read in the bitmap file and get a vector of pixels

  RunKernel(pixels, accumulators);
 
  ifstream myFile;
  myFile.open("util/golden_check_file.txt",ifstream::in);
  ofstream checkFile;
  checkFile.open("util/compare_results.txt",ofstream::out);
  
  vector<int> myList;
  int number;
  while (myFile >> number) {
    myList.push_back(number);
  }
	
  bool failed = false;
  for (int i=0; i<THETAS*RHOS*2; i++) {
    if ((myList[i]>accumulators[i]+1) || (myList[i]<accumulators[i]-1)) { //Test the results against the golden results
      failed = true;
      checkFile << "Failed at " << i << ". Expected: " << myList[i] << ", Actual: "
	      << accumulators[i] << std::endl;
    }
  }

  myFile.close();
  checkFile.close();

  if (failed) {printf("FAILED\n");}
  else {printf("VERIFICATION PASSED!!\n");}

  return 1;
}

//Struct of 3 bytes for R,G,B components
typedef struct __attribute__((__packed__)) {
  unsigned char  b;
  unsigned char  g;
  unsigned char  r;
} PIXEL;

void read_image(char *image_array) {
  //Declare a vector to hold the pixels read from the image
  //The image is 720x480, so the CPU runtimes are not too long for emulation
  PIXEL im[WIDTH*HEIGHT];
	
  //Open the image file for reading
  ifstream img;
  img.open("Assets/pic.bmp",ios::in);
  
  //Bitmap files have a 54-byte header. Skip these bits
  img.seekg(54,ios::beg);
    
  //Loop through the img stream and store pixels in an array
  for (uint i = 0; i < WIDTH*HEIGHT; i++) {
    img.read(reinterpret_cast<char*>(&im[i]),sizeof(PIXEL));
	      
    //The image is black and white (passed through a Sobel filter already)
    //Store 1 in the array for a white pixel, 0 for a black pixel
    if (im[i].r==0 && im[i].g==0 && im[i].b==0) {
      image_array[i] = 0;
    } else {
      image_array[i] = 1;
    }
  }
}
