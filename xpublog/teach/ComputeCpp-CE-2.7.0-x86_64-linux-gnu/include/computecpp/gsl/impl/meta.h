//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_GSL_DETAIL_META_H_
#define RUNTIME_INCLUDE_GSL_DETAIL_META_H_

#include <type_traits>

namespace computecpp {
namespace gsl {
template <bool... Bs>
struct or_impl : std::false_type {};

template <bool B>
struct or_impl<B> : std::integral_constant<bool, B> {};

template <bool B, bool... Bs>
struct or_impl<B, Bs...> : or_impl<or_impl<B>::value || or_impl<Bs...>::value> {
};

template <bool... Bs>
using or_ = or_impl<Bs...>;
}  // namespace gsl
}  // namespace computecpp

#endif  // RUNTIME_INCLUDE_GSL_DETAIL_META_H_
