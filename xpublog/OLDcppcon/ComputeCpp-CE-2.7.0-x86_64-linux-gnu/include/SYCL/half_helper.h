/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/
#ifndef RUNTIME_INCLUDE_SYCL_HALF_HELPER_H_
#define RUNTIME_INCLUDE_SYCL_HALF_HELPER_H_
#ifndef __SYCL_DEVICE_ONLY__

#include "SYCL/predefines.h"

// The internal data structures and the two conversion functions have been taken
// from https://gist.github.com/rygorous/2144712 and
// https://gist.github.com/rygorous/2156668
namespace cl {
namespace sycl {
namespace detail {

/** @brief Structure to represent a 32 bit floating point number
 *  This is a helper function used by the internal half implementation to
 *  represent a 32 bit float and expose its internal bit pattern
 */
union Float32 {
  float f;
  unsigned int u;
  struct {
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
  };
};

/** @brief Structure to represent a 16 bit half precision floating point number
 *  This is a helper function used by the internal half implementation to
 *  represent a 16 bit half and expose its internal bit pattern
 */
union Half16 {
  unsigned short u;
  struct {
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
  };
};

/** @brief Function to transform a 16 bit half to a 32 bit float
 * @param h 16 bit half to be transformed
 * @return the same value as h as a 32 bit float
 */
COMPUTECPP_CONSTEXPR_CPP14 Float32 toFloat(Half16 h) {
  Float32 o = {0};

  // From ISPC ref code
  if (h.Exponent == 0u && h.Mantissa == 0u) {  // (Signed) zero
    o.Sign = h.Sign;
  } else {
    if (h.Exponent == 0u)  // Denormal (will convert to normalized)
    {
      // Adjust mantissa so it's normalized (and keep track of exp adjust)
      int e = -1;
      unsigned m = h.Mantissa;
      do {
        e++;
        m <<= 1;
      } while ((m & 0x400) == 0);

      o.Mantissa = (m & 0x3ff) << 13;
      o.Exponent = 127 - 15 - e;
      o.Sign = h.Sign;
    } else if (h.Exponent == 0x1fu)  // Inf/NaN
    {
      // NOTE: It's safe to treat both with the same code path by just
      // truncating lower Mantissa bits in NaNs (this is valid).
      o.Mantissa = static_cast<unsigned int>(h.Mantissa) << 13;
      o.Exponent = 255;
      o.Sign = h.Sign;
    } else  // Normalized number
    {
      o.Mantissa = static_cast<unsigned int>(h.Mantissa) << 13u;
      o.Exponent = 127u - 15u + h.Exponent;
      o.Sign = h.Sign;
    }
  }

  return o;
}

/** @brief Function to transform a 32 bit float into a 16 bit half
 * @param f 32 bit float to be converted to half
 * @return 16 bit representation of f
 */
COMPUTECPP_CONSTEXPR_CPP14 Half16 toHalf(Float32 f) {
  Half16 o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.Exponent == 0u) {  // Signed zero/denormal (which will underflow)
    o.Exponent = 0;
  } else if (f.Exponent == 255u)  // Inf or NaN (all exponent bits set)
  {
    o.Exponent = 31;
    o.Mantissa = f.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
  } else                                  // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = static_cast<int>(f.Exponent) - 127 + 15;
    if (newexp >= 31)  // Overflow, return signed infinity
      o.Exponent = 31;
    else if (newexp <= 0)  // Underflow
    {
      if ((14 - newexp) <= 24)  // Mantissa might be non-zero
      {
        unsigned mant = f.Mantissa | 0x800000u;  // Hidden 1 bit
        o.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1)  // Check for rounding
          o.u++;  // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.Exponent = newexp;
      o.Mantissa = static_cast<unsigned int>(f.Mantissa) >> 13u;
      if (f.Mantissa & 0x1000u)  // Check for rounding
        o.u++;                   // Round, might overflow to inf, this is OK
    }
  }

  o.Sign = f.Sign;
  return o;
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // __SYCL_DEVICE_ONLY__
#endif  // RUNTIME_INCLUDE_SYCL_HALF_HELPER_H_
