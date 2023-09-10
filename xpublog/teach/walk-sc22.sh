#!/bin/bash

export CPLUS_INCLUDE_PATH=~/syclacademy/External/Catch2/single_include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/syclacademy/Utilities/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=~/syclacademy/External/stb:$CPLUS_INCLUDE_PATH
echo "Exercise_01_Compiling_with_SYCL"
( cd Exercise_01_Compiling_with_SYCL ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_02_Hello_World"
( cd Exercise_02_Hello_World ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_03_Scalar_Add"
( cd Exercise_03_Scalar_Add ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_04_Handling_Errors"
( cd Exercise_04_Handling_Errors ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_05_Vector_Add"
( cd Exercise_05_Vector_Add ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_06_Synchronization"
( cd Exercise_06_Synchronization ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_07_ND_Range_Kernel"
( cd Exercise_07_ND_Range_Kernel ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_08_Matrix_Transpose"
( cd Exercise_08_Matrix_Transpose ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
