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
echo "Exercise_05_Device_Selection"
( cd Exercise_05_Device_Selection ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_06_Vector_Add"
( cd Exercise_06_Vector_Add ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_07_USM_Selector"
( cd Exercise_07_USM_Selector ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_08_USM_Vector_Add"
( cd Exercise_08_USM_Vector_Add ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_09_Synchronization"
( cd Exercise_09_Synchronization ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_10_Managing_Dependencies"
( cd Exercise_10_Managing_Dependencies ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_11_In_Order_Queue"
( cd Exercise_11_In_Order_Queue ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_12_Temporary_Data"
( cd Exercise_12_Temporary_Data ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_13_Load_Balancing"
( cd Exercise_13_Load_Balancing ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_14_ND_Range_Kernel"
( cd Exercise_14_ND_Range_Kernel ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "SKIP - Exercise_15_Image_Convolution"
#echo "Exercise_15_Image_Convolution"
#( cd Exercise_15_Image_Convolution ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_16_Coalesced_Global_Memory"
( cd Exercise_16_Coalesced_Global_Memory ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_17_Vectors"
( cd Exercise_17_Vectors ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_18_Local_Memory_Tiling"
( cd Exercise_18_Local_Memory_Tiling ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
echo "Exercise_19_Work_Group_Sizes"
( cd Exercise_19_Work_Group_Sizes ; /data/oneapi_workshop/xpublog/teach/walk-do-one.sh )
