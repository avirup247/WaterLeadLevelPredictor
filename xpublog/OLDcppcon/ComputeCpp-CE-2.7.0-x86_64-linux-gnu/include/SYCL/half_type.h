/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_
#define RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_

#include "SYCL/half_helper.h"
#include "SYCL/predefines.h"
#include "computecpp_export.h"

namespace cl {
namespace sycl {

#ifndef __SYCL_DEVICE_ONLY__
namespace detail {
template <typename T, typename BinaryOp, typename>
struct known_identity_helper;
}
/** @brief Definition of half type
 *
 * This class is used to represent a 16 bit floating point number. Code
 * compiled for the device this class will use the compiler builtin __fp16
 * and compile directly to native half instructions. For the host it only
 * operates as a storage type: floating point numbers will be transformed
 * into 16 bits and back but the actual computation will be performed by
 * casting back to a 32 bit float.
 */
class half {
  /** Numeric_limits must be a friend to access protected constexpr constructor
   * that sets bits directly
   */

  template <typename T>
  friend struct std::numeric_limits;

  /** Needs to be a friend also to access the constexpr constructor.  */
  template <typename T, typename BinaryOp, typename>
  friend struct cl::sycl::detail::known_identity_helper;

 protected:
  /** Used to disambiguate constructor from float value from other kinds
   */
  struct value_tag {};

  /** Constructs a half value from a float.
   *  This function has stronger constexpr guarantees
   *  and doesn't export the symbol.
   * @param f Float value to be converted
   */
  COMPUTECPP_CONSTEXPR_CPP14 half(value_tag, float f)
      : m_bitpattern{detail::toHalf(detail::Float32{f}).u} {}

  /** @brief Constructs a half value by setting the underlying bit pattern
   * directly. Used for initializing specific values in a C++11 constexpr
   * context.
   * @param u The unsigned short representing the underlying bits of the value.
   */
  constexpr explicit half(value_tag, unsigned short u) : m_bitpattern(u) {}

 public:
  /** @brief default constructor, inits as zero */
  COMPUTECPP_CONSTEXPR_EXPORT half() = default;

  /** @brief Takes in a 32 bit float and converts it to 16 bits
   * @param f the float to be converted to 16 bits
   */
  COMPUTECPP_ABI_CONSTEXPR half(const float& f);

  /** @brief Implicit cast to float
   *  Converts the 16 bit half back to a 32 bit float for performing the actual
   *  computation
   */
  COMPUTECPP_CONSTEXPR_EXPORT operator float() const {
    return detail::toFloat(detail::Half16{m_bitpattern}).f;
  }

  /** @brief Applies == to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the == operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator==(const half& rhs) const {
    return static_cast<float>(*this) == static_cast<float>(rhs);
  }

  /** @brief Applies != to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the != operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator!=(const half& rhs) const {
    return static_cast<float>(*this) != static_cast<float>(rhs);
  }

  /** @brief Applies < to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the < operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator<(const half& rhs) const {
    return static_cast<float>(*this) < static_cast<float>(rhs);
  }

  /** @brief Applies > to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the > operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator>(const half& rhs) const {
    return static_cast<float>(*this) > static_cast<float>(rhs);
  }

  /** @brief Applies <= to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the <= operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator<=(const half& rhs) const {
    return static_cast<float>(*this) <= static_cast<float>(rhs);
  }

  /** @brief Applies >= to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the >= operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator>=(const half& rhs) const {
    return static_cast<float>(*this) >= static_cast<float>(rhs);
  }

  /** @brief Applies += to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the += operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator+=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp += static_cast<float>(rhs);
    *this = half{value_tag{}, temp};
    return *this;
  }

  /** @brief Applies -= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the -= operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator-=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp -= static_cast<float>(rhs);
    *this = half{value_tag{}, temp};
    return *this;
  }

  /** @brief Applies *= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the *= operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator*=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp *= static_cast<float>(rhs);
    *this = half{value_tag{}, temp};
    return *this;
  }

  /** @brief Applies /= to this half with another half.
   * @param rhs The rhs half.
   * @return This half after the application of the /= operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator/=(const half& rhs) {
    float temp = static_cast<float>(*this);
    temp /= static_cast<float>(rhs);
    *this = half{value_tag{}, temp};
    return *this;
  }

  /** @brief Applies && to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the && operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator&&(const half& rhs) const {
    return (static_cast<float>(*this) != 0.f) &&
           (static_cast<float>(rhs) != 0.f);
  }

  /** @brief Applies || to this half and another half.
   * @param rhs The rhs half.
   * @return The result of the || operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT bool operator||(const half& rhs) const {
    return (static_cast<float>(*this) != 0.f) ||
           (static_cast<float>(rhs) != 0.f);
  }

  /** @brief Applies ++ to this half.
   * @return This half after the application of the ++ operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator++() {
    (*this) += half{value_tag{}, 1.0f};
    return *this;
  }

  /** @brief Applies ++ to this half.
   * @return A copy of this half before the ++ operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half operator++(int) {
    half save = *this;
    (*this) += half{value_tag{}, 1.0f};
    return save;
  }

  /** @brief Applies -- to this half.
   * @return This half after the application of the -- operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half& operator--() {
    (*this) -= half{value_tag{}, 1.0f};
    return *this;
  }

  /** @brief Applies -- to this half.
   * @return A copy of this half before the -- operation.
   */
  COMPUTECPP_CONSTEXPR_EXPORT half operator--(int) {
    half save = *this;
    (*this) -= half{value_tag{}, 1.0f};
    return save;
  }

 private:
  // Contains the floating point number a unsigned short following the
  // 1 bit sign, 5 bit exponent, 10 bit mantissa as set out in the IEEE 754
  // standard
  unsigned short m_bitpattern{0};
};

#ifndef ComputeCpp_EXPORTS
// If the shared binary is not being generated,
// we need to provide definitions for constexpr functions on the header

COMPUTECPP_ABI_CONSTEXPR half::half(const float& f) : half{value_tag{}, f} {}

#endif  // ComputeCpp_EXPORTS

#else
typedef __fp16 half;
#endif  // __SYCL_DEVICE_ONLY__
}  // namespace sycl
}  // namespace cl

/** std type specializations for half  */
namespace std {
template <>
struct hash<cl::sycl::half> {
  size_t operator()(const cl::sycl::half& key) const {
    return hash<uint16_t>{}(reinterpret_cast<const uint16_t&>(key));
  }
};

template <>
struct numeric_limits<cl::sycl::half> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr bool tinyness_before = false;
  static constexpr bool traps = false;
  static constexpr int max_exponent10 = 4;
  static constexpr int max_exponent = 16;
  static constexpr int min_exponent10 = -4;
  static constexpr int min_exponent = -13;
  static constexpr int radix = 2;
  static constexpr int max_digits10 = 5;
  static constexpr int digits = 11;
  static constexpr bool is_bounded = true;
  static constexpr int digits10 = 3;
  static constexpr bool is_modulo = false;
  static constexpr bool is_iec559 = true;
  static constexpr float_round_style round_style = round_to_nearest;

  static constexpr cl::sycl::half min() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return 6.103515625e-05f;
#else
    // Set bits directly for representation of float value 6.103515625e-05f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(1024));
#endif
  }

  static constexpr cl::sycl::half max() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return 65504.0f;
#else
    // Set bits directly for representation of float value 65504.0f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(31743));
#endif
  }

  static constexpr cl::sycl::half lowest() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return -65504.0f;
#else
    // Set bits directly for representation of float value -65504.0f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(64511));
#endif
  }

  static constexpr cl::sycl::half epsilon() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return 9.765625e-04f;
#else
    // Set bits directly for representation of float value 9.765625e-04f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(5120));
#endif
  }

  static constexpr cl::sycl::half round_error() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return 0.5f;
#else
    // Set bits directly for representation of float value 0.5f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(14336));
#endif
  }

  static constexpr cl::sycl::half infinity() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_huge_valf();
#else
    // Set bits directly for representation of float value inf
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(0x7C00));

#endif
  }

  static constexpr cl::sycl::half quiet_NaN() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_nanf("");
#else
    // Set bits directly for representation of float value of __builtin_nanf("")
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(32256));
#endif
  }

  static constexpr cl::sycl::half signaling_NaN() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return __builtin_nansf("");
#else
    // Set bits directly for representation of float value __builtin_nansf("")
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(32256));
#endif
  }

  static constexpr cl::sycl::half denorm_min() noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return 5.96046e-08f;
#else
    // Set bits directly for representation of float value 5.96046e-08f
    return cl::sycl::half(cl::sycl::half::value_tag{},
                          static_cast<uint16_t>(1));
#endif
  }
};
}  // namespace std
#endif  // RUNTIME_INCLUDE_SYCL_HALF_TYPE_H_
