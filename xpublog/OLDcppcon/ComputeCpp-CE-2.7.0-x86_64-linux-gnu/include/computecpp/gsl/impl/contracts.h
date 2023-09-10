//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_GSL_DETAIL_CONTRACTS_H_
#define RUNTIME_INCLUDE_GSL_DETAIL_CONTRACTS_H_

#include <cassert>

#define COMPUTECPP_EXPECTS(...) assert((__VA_ARGS__))
#define COMPUTECPP_ENSURES(...) assert((__VA_ARGS__))

#endif  // RUNTIME_INCLUDE_GSL_DETAIL_CONTRACTS_H_
