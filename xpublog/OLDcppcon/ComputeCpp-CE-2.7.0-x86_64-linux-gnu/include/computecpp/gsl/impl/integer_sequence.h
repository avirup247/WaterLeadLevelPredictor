//
// Copyright (C) 2002-2019 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_INTEGER_SEQUENCE_H_
#define RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_INTEGER_SEQUENCE_H_

#include <type_traits>

namespace computecpp {
namespace gsl {
/// @brief Represents a compile-time sequence of integers.
///
/// This is useful for pack expansions with integers, such as expanding a tuple.
/// @tparam T The type of the values.
/// @tparam Ints... The values of the sequence.
/// @note This has been fast-forwarded from C++14.
///
template <typename T, T... Ints>
struct integer_sequence {
  static_assert(std::is_integral<T>::value,
                "integer_sequence requires an integral type.");

  /// @brief Returns the number of elements in `Ints`.
  /// @returns The number of elements in `Ints`.
  ///
  static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};

/// @brief Helper alias for the common case where `T` is `std::size_t`.
/// @tparam Ints... The values of the sequence.
/// @note This has been fast-forwarded from C++14.
///
template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

namespace detail {
/// @brief Helper type for generating an integer sequence of the range [0, N).
/// @tparam T The type of the integer sequence.
/// @tparam N The sentinel value for the integer sequence.
/// @tparam Current Index to track the length of the sequence.
/// @tparam Ns... The values of the integer sequence, in ascending order.
///
template <typename T, T N, T Current, T... Ns>
struct make_integer_sequence_impl
    : make_integer_sequence_impl<T, N, Current + 1, Ns..., Current> {};

/// @copydoc
///
template <typename T, T N, T... Ns>
struct make_integer_sequence_impl<T, N, N, Ns...> {
  using type = integer_sequence<T, Ns...>;
};

}  // namespace detail

/// @brief Helper alias template generates a sequence of integers of type T from
/// [0, N).
/// @tparam T Type of the sequence.
/// @tparam N The upper bound of the sequence.
/// @note This has been fast-forwarded from C++14.
///
template <typename T, T N>
using make_integer_sequence =
    typename detail::make_integer_sequence_impl<T, N, T{}>::type;

/// @brief Helper alias template generates a sequence of integers of type
/// std::size_t from [0, N).
/// @tparam N The upper bound of the sequence.
/// @note This has been fast-forwarded from C++14.
///
template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

/// @brief Generates an index sequence from a type parameter pack, of the same
/// length.
/// @tparam Ts... A type parameter pack to generate the sequence from.
/// @note This has been fast-forwarded from C++14.
///
template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;
}  // namespace gsl
}  // namespace computecpp

#endif  // RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_INTEGER_SEQUENCE_H_
