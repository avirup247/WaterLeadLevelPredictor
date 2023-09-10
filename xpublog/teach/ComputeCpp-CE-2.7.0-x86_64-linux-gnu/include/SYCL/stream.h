/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file stream.h

  @brief Public interface implementation of the stream class as defined by the
  SYCL 1.2 specification.
*/

#ifndef RUNTIME_INCLUDE_SYCL_STREAM_H_
#define RUNTIME_INCLUDE_SYCL_STREAM_H_

#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/half_type.h"
#include "SYCL/id.h"
#include "SYCL/index_array.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/predefines.h"
#include "SYCL/range.h"
#include "SYCL/storage_mem.h"  // IWYU pragma: keep
#include "SYCL/stream_args.h"
#include "SYCL/vec.h"  // IWYU pragma: keep
#include "SYCL/vec_swizzles_impl.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <system_error>
#include <type_traits>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
template <int dimensions, bool with_offset>
class item;
template <int dimensions>
class group;
template <int dimensions>
class h_item;
template <int dimensions>
class nd_item;
template <typename dataT, access::address_space addressSpace>
class multi_ptr;
}  // namespace sycl
}  // namespace cl

/** @cond COMPUTECPP_DEV */

/**
@def Pre-process definition for the vec element delimiter.
*/
const char* const COMPUTECPP_VEC_ELEM_DELIMITER = ", ";

/**
@def Pre-process definition for the negative sign.
*/
const char* const COMPUTECPP_NEGATIVE_SIGN = "-";

/**
@def Pre-process definition for the decimal point.
*/
const char* const COMPUTECPP_DECIMAL_POINT = ".";

/**
@def Pre-process definition for the hexadecimal prefix.
*/
const char* const COMPUTECPP_HEX_PREFIX = "0x";

/**
@def Pre-process definition for the positive exponent prefix.
*/
const char* const COMPUTECPP_POSITIVE_EXP_PREFIX = "e+";

/**
@def Pre-process definition for the negative exponent prefix.
*/
const char* const COMPUTECPP_NEGATIVE_EXP_PREFIX = "e-";

/**
@def Pre-process definition for the open brace.
*/
const char* const COMPUTECPP_OPEN_BRACE = "{ ";

/**
@def Pre-process definition for the close brace.
*/
const char* const COMPUTECPP_CLOSE_BRACE = " }";

/** COMPUTECPP_DEV @endcond */

namespace cl {
namespace sycl {

class handler;

/**
@brief Enum class that enumerates the different stream manipulators.
*/
enum class stream_manipulator {
  endl = 0,
  scientific = 1,
  hex = 2,
  oct = 3,
  showbase = 4,
  showpos = 5,
  dec = 6,
  noshowbase = 7,
  noshowpos = 8,
  fixed = 9,
  hexfloat = 10,
  defaultfloat = 11,
  flush = 12,
};

/**
@brief Class that encapsulates a precision stream manipulator.
*/
class precision_manipulator {
 public:
  /**
  @brief Constructor that takes a precision value and assigns it to the
  precision field.
  @param precision THe precision value to construct the precision _manipulator
  with.
  */
  precision_manipulator(int precision) : m_precision(precision) {}

  /**
  @brief Inline member function that returns the precision value.
  @return The precision value.
  */
  inline int get_precision() const { return m_precision; }

 private:
  /**
  @brief The precision value.
  */
  int m_precision;
};

namespace detail {

/** @brief An instance of this class is returned when calling setw
 */
class width_manipulator {
 public:
  /** @brief Construct an instance using the specified width
   * @param width The width to use for manipulating the stream
   */
  explicit constexpr width_manipulator(const int width) : m_width(width) {}

  /** @brief Retrieves the specified width
   * @return Width of a single stream element
   */
  inline constexpr int get_width() const noexcept { return m_width; }

 private:
  /** @brief Width of a single stream element
   */
  int m_width;
};

}  // namespace detail

/**
@brief Const value defined to the scientific stream manipulator, used in
conjunction with stream operators.
*/
const stream_manipulator scientific = stream_manipulator::scientific;

/** @brief Stream manipulator for displaying numbers in base 10
 */
const stream_manipulator dec = stream_manipulator::dec;

/**
@brief Const value defined to the hex stream manipulator, used in conjunction
with stream operators.
*/
const stream_manipulator hex = stream_manipulator::hex;

/**
@brief Const value defined to the oct stream manipulator, used in conjunction
with stream operators.
*/
const stream_manipulator oct = stream_manipulator::oct;

/**
@brief Const value defined to the showbase stream manipulator, used in
conjunction with stream operators.
*/
const stream_manipulator showbase = stream_manipulator::showbase;

/** @brief Stream manipulator for reverting showbase
 */
const stream_manipulator noshowbase = stream_manipulator::noshowbase;

/**
@brief Const value defined to the showpos stream manipulator, used in
conjunction with stream operators.
*/
const stream_manipulator showpos = stream_manipulator::showpos;

/** @brief Stream manipulator for reverting showpos
 */
const stream_manipulator noshowpos = stream_manipulator::noshowpos;

/**
@brief Const value defined to the endl stream manipulator, used in conjunction
with stream operators.
*/
const stream_manipulator endl = stream_manipulator::endl;

/** @brief Stream manipulator that forces floating point numbers to be displayed
 *        using fixed precision
 */
const stream_manipulator fixed = stream_manipulator::fixed;

/** @brief Stream manipulator that displays floating point numbers in base 16
 */
const stream_manipulator hexfloat = stream_manipulator::hexfloat;

/** @brief Stream manipulator that reverts the display of floating point numbers
 *        back to the default representation
 */
const stream_manipulator defaultfloat = stream_manipulator::defaultfloat;

/// Stream manipulator that flushes the current stream buffer
const stream_manipulator flush = stream_manipulator::flush;

/**
@brief Inline function that constructs a precision_manipulator object, used in
conjunction with stream operators.
@param precision The precision value to construct the precision_manipulator
with.
@return The constructed precision_manipulator.
*/
inline precision_manipulator setprecision(int precision) {
  return precision_manipulator(precision);
}

/** @brief Creates a stream manipulator for setting the stream width
 * @param width Desired stream width
 * @return Stream width manipulator object
 */
inline detail::width_manipulator setw(const int width) {
  return detail::width_manipulator(width);
}

class stream;

namespace detail {
template <>
struct opencl_backend_traits<sycl::stream> {
 public:
  // input_type not defined
  using return_type = typename sycl::global_ptr<cl_char>::ptr_t;
};
}  // namespace detail

#ifndef __SYCL_DEVICE_ONLY__

/**
@brief Class that encapsulates a buffer that can be passed to the device and
streamed to. Used for debugging. Host only.
*/
class COMPUTECPP_EXPORT stream {
  /**
  @brief Friend function declaration to give global stream operator access to
  private members.
  */
  friend COMPUTECPP_EXPORT std::ostream& operator<<(std::ostream& out,
                                                    const stream& rhs);

 public:
  /**
  @brief Constructor that takes a buffer size, a max statement size and a
  handler reference and constructs a stream of the buffer size, associated with
  the provided handler
  @param bufferSize The size of the stream's buffer.
  @param maxStatementSize The maximum size of a single statement in the stream's
  buffer.
  @param cgh A reference to the handler that the stream is to be associated
  with.
  */
  stream(size_t bufferSize, size_t maxStatementSize, handler& cgh);

  /** @brief Copy constructor
   */
  stream(const stream&) = default;

  /** @brief Move constructor
   */
  stream(stream&&) noexcept = default;

  /** @brief Copy assignment operator
   * @return Reference to this object
   */
  stream& operator=(const stream&) = default;

  /** @brief Move assignment operator
   * @return Reference to this object
   */
  stream& operator=(stream&&) noexcept = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const stream& lhs, const stream& rhs) {
    return lhs.m_hostArgs == rhs.m_hostArgs;
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const stream& lhs, const stream& rhs) {
    return !(lhs == rhs);
  }

  /**
  @brief Destructor, prints the buffer to stdout and resets the buffer.
  */
  ~stream();

  /**
  @brief Member function that returns the buffer size.
  @return The buffer size.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use stream::size() instead.")
  size_t get_size() const;

 private:
  size_t size_impl() const;

 public:
#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return size_impl(); }
#endif

  /**
  @brief Inline member function that returns the maximum statement size.
  @return THe maximum statement size.
  */
  inline size_t get_max_statement_size() const {
    return m_hostArgs.m_maxStatementSize;
  }

  /**
  @brief Inline member function that returns the stream mode.
  @return The stream mode.
  */
  inline stream_mode get_stream_mode() const { return m_hostArgs.m_streamMode; }

  /**
  @brief Inline member function that returns the precision value.
  @return The precision value.
  */
  inline int get_precision() const { return m_hostArgs.m_precision; }

  /**
  @brief Inline member function that sets the stream mode.
  @param mode The stream mode to set.
  */
  inline void set_stream_mode(stream_mode mode) {
    m_hostArgs.m_streamMode = mode;
  }

  /**
  @brief Inline member function that sets the precision value.
  @param precision The precision value to set.
  */
  inline void set_precision(int precision) {
    m_hostArgs.m_precision = precision;
  }

  /** @brief Sets the width used when displaying a stream element
   * @param width Desired width
   */
  inline void set_width(const int width) { m_hostArgs.m_width = width; }

  /**
  @brief Member function that appends a char buffer to the stream's buffer.
  @param charBuffer The char buffer to be appended.
  @size The size of the char buffer to be appended.
  */
  void append_chars(const char* charBuffer, int size) const;

  /**
  @brief Member function that returns a shared_ptr to the detail buffer object.
  @return A shared_ptr to the detail buffer object.
  */
  std::shared_ptr<cl::sycl::storage_mem> get_buffer() const;

 private:
  detail::host_stream_container m_hostArgs;
};

#else

/**
@brief Class that encapsulates a buffer that can be passed to the device and
streamed to. Used for debugging. Device only.
*/
class stream {
 public:
  stream(size_t /*bufferSize*/, size_t /*maxStatementSize*/, handler&) {}

  friend inline bool operator==(const stream& lhs, const stream& rhs) {
    return (lhs.m_deviceArgs == rhs.m_deviceArgs);
  }

  friend inline bool operator!=(const stream& lhs, const stream& rhs) {
    return !(lhs == rhs);
  }

  inline size_t get_size() const {
    return m_deviceArgs.m_metadata.m_bufferSize;
  }

  inline size_t get_max_statement_size() const {
    return m_deviceArgs.m_metadata.m_maxStatementSize;
  }

  inline stream_mode get_stream_mode() const {
    return static_cast<stream_mode>(m_deviceArgs.m_metadata.m_streamMode);
  }

  inline int get_precision() const {
    // Not implemented
    return 8;
  }

  inline void set_stream_mode(stream_mode mode) {
    m_deviceArgs.m_metadata.m_streamMode =
        static_cast<detail::device_stream_metadata::field_type>(mode);
  }

  inline void set_precision(int /*precision*/) {
    // Not implemented
  }

  inline void set_width(const int /*width*/) {
    // Not implemented
  }

  inline void append_chars(const char* charBuffer, int size) const {
    for (int i = 0; i < size; i++) {
      m_deviceArgs.m_ptr[m_deviceArgs.m_metadata.m_currentIndex + i] =
          charBuffer[i];
    }
    m_deviceArgs.m_metadata.m_currentIndex += size;
    m_deviceArgs.m_ptr[m_deviceArgs.m_metadata.m_currentIndex] = '\0';
  }

 private:
  detail::device_stream_container m_deviceArgs;
};

#endif

/**
  @brief Stream operator that streams a stream object to an ostream.
  @param out The output stream.
  @param rhs The stream object to be streamed into the output.
*/
COMPUTECPP_EXPORT std::ostream& operator<<(std::ostream& out,
                                           const stream& rhs);

/**
@brief Static const value for maximum integral digits.
*/
static const unsigned int MAX_INTEGRAL_DIGITS = 20;

/**
@brief Static const value for maximum fractional digits.
*/
static const unsigned int MAX_FRACTIONAL_DIGITS = 20;

/**
@brief Static const value for maximum hexadecimal digits.
*/
static const unsigned int MAX_HEX_DIGITS = 20;

namespace detail {

/** @brief Helper class for adding a potentially negative value to the stream
 * @tparam valueT The type of the value to be appended
 * @tparam isSigned Whether the value type is signed or not
 */
template <typename valueT, bool isSigned = std::is_signed<valueT>::value>
struct abs_value;

/** @brief Helper class for adding a potentially negative value to the stream,
 *        overload for signed types
 * @tparam valueT The type of the value to be appended.
 */
template <typename valueT>
struct abs_value<valueT, true> {
  /** @brief If value is negative, display the minus sign
   *        and return the absolute value
   * @param os The stream object
   * @param value Input value
   * @return Absolute value
   */
  static valueT get(const stream& os, valueT value) {
    if (value < 0) {
      os.append_chars(COMPUTECPP_NEGATIVE_SIGN, 1);
      value *= -1;
    }
    return value;
  }
};

/** @brief Helper class for adding a potentially negative value to the stream,
 *        overload for unsigned types
 * @tparam valueT The type of the value to be appended
 */
template <typename valueT>
struct abs_value<valueT, false> {
  /** @brief Unsigned values cannot be negative, do nothing
   * @return Input value
   */
  static valueT get(const stream& /*os*/, valueT value) { return value; }
};

/** @brief Converts a digit into a char
 * @param digit Single digit value
 * @return Char representation of the digit
 */
inline constexpr char to_char(int digit) {
  return static_cast<char>(digit + '0');
}

/** @brief Calculates the length of a C-style string
 * @param cStr Null-terminated array of characters
 * @return Length of the string
 */
inline int strlen(const char* cStr) {
  int index = 0;
  while (cStr[index] != '\0') {
    ++index;
  }
  return index;
}

}  // namespace detail

/**
@brief Template function that appends a hexadecimal value to a stream object.
@tparam valueT The type of the hexadecimal value to be appended.
@param os The stream object to be appended to.
@param value A hexadecimal value to be appended to the stream object.
*/
template <typename valueT>
void append_hexidecimal(const stream& os, const valueT& value) {
  /* If value is negative, display the minus sign and convert the value to its
   * absolute value. */
  auto absValue = detail::abs_value<valueT>::get(os, value);

  char digits[MAX_HEX_DIGITS] = {0};
  int currentDigit = 0;

  valueT recValue = absValue;
  while (recValue > 0) {
    digits[currentDigit] = recValue % 16;
    recValue /= 16;
    currentDigit++;
  }

  os.append_chars(COMPUTECPP_HEX_PREFIX, 2);

  for (int i = (currentDigit - 1); i >= 0; i--) {
    char c = (digits[i] <= 9) ? detail::to_char(digits[i])
                              : static_cast<char>('A' + (digits[i] - 10));
    os.append_chars(&c, 1);
  }
}

/** @brief Template function that appends a integral value to a stream object.
 * @tparam valueT The type of the integral value to be appended.
 * @param os The stream object to be appended to.
 * @param value A integral value to be appended to the stream object.
 * @param useManipulators Whether to apply stream manipulators when streaming
 */
template <typename valueT>
inline int append_integral(const stream& os, const valueT& value,
                           bool useManipulators = true) {
  if (value == 0) {
    // A zero is always displayed
    auto character = detail::to_char(0);
    constexpr int numDigits = 1;
    os.append_chars(&character, numDigits);
    return numDigits;
  }

  /* Handle hex manipulator. */
  if (useManipulators && (os.get_stream_mode() == stream_mode::hex)) {
    append_hexidecimal(os, value);
    return 0;
  }

  /* If value is negative, display the minus sign and convert the value to its
   * absolute value. */
  auto absValue = detail::abs_value<valueT>::get(os, value);

  /* Create a variable with the original value, that is divided by 10 until it
   * reaches zero. */
  valueT devidedValue = absValue;

  /* Create an array to store the digits retrieved from the value. */
  valueT digits[MAX_INTEGRAL_DIGITS] = {0};

  /* Create a counter to record the number of digits retrieved form the value.
   */
  int numDigits = 0;

  /* While the divided value is still greater than zero */
  while (devidedValue > 0) {
    /* Calculate a multiplier based on the current digit counter. */
    valueT multiplier = 10;
    for (int m = 0; m < numDigits; m++) {
      multiplier *= 10;
    }

    /* Retrieve the digit based on the current digit counter, by dividing by the
     * multiplier, multiplying by the multiplier, subtracting this from the
     * original value and then finally dividing by the multiple divided by 10,
     * reducing the value down to a single digit. */
    valueT singleDigit =
        (absValue - ((absValue / multiplier) * multiplier)) / (multiplier / 10);

    /* assign the digit retrieved. */
    digits[numDigits] = singleDigit;

    /* Divided the value again and increment the digit counter. */
    devidedValue /= 10;
    numDigits++;
  }

  /* Loop over the array of retrieved digits in reverse order, converting each
   * to a character and appending it. */
  for (int i = numDigits - 1; i >= 0; i--) {
    auto integral = static_cast<int>(digits[i]);
    auto character = detail::to_char(integral);
    os.append_chars(&character, 1);
  }

  /* Return the number fo digits. */
  return numDigits;
}

/**
@brief Template function that appends a floating point value to a stream object.
@tparam valueT The type of the floating point value to be appended.
@param os The stream object to be appended to.
@param value A floating point value to be appended to the stream object.
*/
template <typename valueT>
inline void append_floating_point(const stream& os, const valueT& value) {
  /* If value is negative, display the minus sign and convert the value to its
   * absolute value. */
  auto absValue = detail::abs_value<valueT>::get(os, value);

  /* Create variables for storing the scientific base. */
  int scientificBase = 0;

  /* Create a variable to storing whether the base is positive or not. */
  bool possitiveBase = true;

  /* Handle scientific manipulator. */
  if (os.get_stream_mode() == stream_mode::scientific) {
    /* If value is greater than zero. */
    if (absValue >= 0) {
      /* Divide the value by 10 until it reaches a single digit on the integral
       * side. */
      while (absValue > valueT(10)) {
        absValue /= valueT(10);
        scientificBase++;
      }
    }
    /* If value is less than zero. */
    else {
      /* Multiply the value by 10 until it reaches a single digit on the
       * integral side. */
      while (absValue < valueT(0.1)) {
        absValue *= valueT(10);
        scientificBase--;
      }
      possitiveBase = false;
    }
  }

  /* Cast to an integer to retrieve the integral part of the floating point. */
  int integralPart = static_cast<int>(absValue);

  /* Append the integral part pf the floating point. */
  int digitsInIntegralPart = append_integral<int>(os, integralPart);

  /* Append the period sign. */
  os.append_chars(COMPUTECPP_DECIMAL_POINT, 1);

  /* Subtract the integral part from the original value to retrieve the
   * fractional part. */
  valueT fractionalPart = absValue - static_cast<valueT>(integralPart);

  /* Create an array to store the digits retrieved from the value. */
  valueT digits[MAX_FRACTIONAL_DIGITS] = {valueT{0}};

  /* Retrieve the precision. */
  int precision = os.get_precision();

  /* Calculate digits for fractional side. */
  int digitsForFractionalSide = precision - digitsInIntegralPart;

  /* Loop for each digit of precision. */
  for (int i = 0; i < (digitsForFractionalSide - 1); i++) {
    /* Create a multiplier based on the current digit. */
    auto currentMultiplier = valueT{10.0};
    for (int m = 0; m < i; m++) {
      currentMultiplier *= valueT{10.0};
    }

    /* Multiply the fractional part by the multiplier in order to convert the
     * current digit in the fractional part to a digit on the integral part. */
    valueT multipliedFloat = fractionalPart * currentMultiplier;

    /* Loop for each previously retrieved digit, creating a multiplier for it
     * and
     * using that to remove all integral parts beyond the first digit. */
    for (int r = 0; r < i; r++) {
      /* Create a multiplier based on the difference in digits between the
       * current loop and the current digit. */
      float currentStageMultiplier = 1.0f;
      for (int sm = 0; sm < (i - r); sm++) {
        currentStageMultiplier *= 10.0f;
      }

      /* Multiply the multiplier by the stored digit to obtain the value to
       * remove from the integral part. */
      multipliedFloat -=
          (digits[r] * static_cast<valueT>(currentStageMultiplier));
    }

    /* Assign the retrieved digit to the array */
    digits[i] = static_cast<valueT>((static_cast<int>(multipliedFloat)));

    /* Convert the digit to an character and append it. */
    auto integral = static_cast<int>(digits[i]);
    auto character = detail::to_char(integral);
    os.append_chars(&character, 1);
  }

  /* If scientific manipulator is enabled add the scientific base. */
  if (os.get_stream_mode() == stream_mode::scientific) {
    /* Append the scientific notation. */
    if (possitiveBase) {
      os.append_chars(COMPUTECPP_POSITIVE_EXP_PREFIX, 2);
    } else {
      os.append_chars(COMPUTECPP_NEGATIVE_EXP_PREFIX, 2);
    }

    /* Create variable to store the number of digits in the scientific base. */
    int digitsInScientificBase = 0;

    /* Divide the scientific base by 10 until it reaches zero to detect the
     * number of digits. */
    int tempScientificBase = scientificBase;
    while (tempScientificBase > 0) {
      tempScientificBase /= 10;
      digitsInScientificBase++;
    }

    /* Add the correct number of zeros before the scientific base. */
    for (int sbd = 0; sbd < (3 - digitsInScientificBase); sbd++) {
      os.append_chars("0", 1);
    }

    /* Appen the integral specifying the scientific base. */
    append_integral(os, scientificBase);
  }
}

/**
@brief Inline function that appends a string value to a stream object.
@param os The stream object to be appended to.
@param value A string value to be appended to the stream object.
*/
inline void append_string(const stream& os, const char* value) {
  auto numChars = detail::strlen(value);
  os.append_chars(value, numChars);
}

/** @brief Appends a string representation of a type to the stream
 * @param os Stream object to append to
 * @param typeStr String representing the type name.
 *        Can contain a single percentage character that will get replaced by
 *        the value of dimensions
 * @param numChars Number of characters in the provided type string
 * @param dimensions Optional number of dimensions of the type
 */
inline void append_type_string(const stream& os, char* typeStr, int numChars,
                               int dimensions = 0) {
  int percentPos = -1;
  for (int i = 0; i < numChars; ++i) {
    if (typeStr[i] == '%') {
      percentPos = i;
      break;
    }
  }
  if (percentPos >= 0) {
    typeStr[percentPos] = detail::to_char(dimensions);
  }
  os.append_chars(typeStr, numChars);
}

/**
@brief Inline function that appends a char value to the a object.
@param os The stream object to be appended to.
@param value A char value to be appended to the stream object.
*/
inline void append_char(const stream& os, const char& value) {
  os.append_chars(&value, 1);
}

/**
@brief Inline function that appends a bool value to the a object.
@param os The stream object to be appended to.
@param value A bool value to be appended to the stream object.
*/
inline void append_bool(const stream& os, const bool& value) {
  if (value) {
    os.append_chars("true", 4);
  } else {
    os.append_chars("false", 5);
  }
}

/** @brief Appends an index_array object to the stream
 * @param os Stream object to append to
 * @param value Array of indexes
 * @param numElements Number of elements of the index array
 */
inline void append_index_array(const stream& os,
                               const detail::index_array& value,
                               int numElements) {
  static constexpr bool useManipulators = false;
  append_string(os, COMPUTECPP_OPEN_BRACE);
  for (int i = 0; i < numElements - 1; ++i) {
    append_integral(os, value[i], useManipulators);
    append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  }
  append_integral(os, value[numElements - 1]);
  append_string(os, COMPUTECPP_CLOSE_BRACE);
}

/**
@brief Template function that streams a pointer value to a stream object.
@tparam pointerT The type of a pointer value to be streamed.
@param os The stream object to be streamed to.
@param rhs The pointer value to be streamed.
*/
template <typename pointerT>
inline const stream& stream_value(const stream& os, pointerT* rhs) {
  append_hexidecimal(os, reinterpret_cast<intptr_t>(rhs));
  return os;
}

/** @brief Streams a multi_ptr object into a stream
 * @tparam dataType Underlying type of the multi_ptr pointer
 * @tparam asp Address space of the pointer
 * @param os The stream object to be streamed to
 * @param rhs multi_ptr object to display in the stream
 * @return Original stream object
 */
template <typename dataType, cl::sycl::access::address_space asp>
inline const stream& stream_value(const stream& os,
                                  const multi_ptr<dataType, asp>& rhs) {
  append_hexidecimal(os, reinterpret_cast<intptr_t>(rhs.get()));
  return os;
}

/** @brief Streams an integral type into a stream
 * @tparam dataType Type of the value to be streamed
 * @tparam COMPUTECPP_ENABLE_IF Only enabled for integral types
 * @param os The stream object to be streamed to
 * @param rhs Integral value
 * @return Original stream object
 */
template <class dataType,
          COMPUTECPP_ENABLE_IF(dataType, (std::is_integral<dataType>::value))>
inline const stream& stream_value(const stream& os, dataType rhs) {
  append_integral<dataType>(os, rhs);
  return os;
}

/** @brief Streams a half value to a stream object
 * @param os The stream object to be streamed to
 * @param rhs The half value to be streamed
 */
inline const stream& stream_value(const stream& os, const half& rhs) {
  append_floating_point<float>(os, static_cast<float>(rhs));
  return os;
}

/**
@brief Inline function that streams a float value to a stream object.
@param os The stream object to be streamed to.
@param rhs The float value to be streamed.
*/
inline const stream& stream_value(const stream& os, const float& rhs) {
  append_floating_point<float>(os, rhs);
  return os;
}

/**
@brief Inline function that streams a double value to a stream object.
@param os The stream object to be streamed to.
@param rhs The double value to be streamed.
*/
inline const stream& stream_value(const stream& os, const double& rhs) {
  append_floating_point<double>(os, rhs);
  return os;
}

/**
@brief Inline function that streams a bool value to a stream object.
@param os The stream object to be streamed to.
@param rhs The bool value to be streamed.
*/
inline const stream& stream_value(const stream& os, const bool& rhs) {
  append_bool(os, rhs);
  return os;
}

/**
@brief Inline function that streams a const char & value to a stream object.
@param os The stream object to be streamed to.
@param rhs The const char & value to be streamed.
*/
inline const stream& stream_value(const stream& os, const char& rhs) {
  append_char(os, rhs);
  return os;
}

/**
@brief Inline function that streams a const char * value to a stream object.
@param os The stream object to be streamed to.
@param rhs The const char * value to be streamed.
*/
inline const stream& stream_value(const stream& os, const char* rhs) {
  append_string(os, rhs);
  return os;
}

/**
@brief Inline function that streams a manipulator to a stream object, applying
the appropriate action.
@param os The stream object to be streamed to.
@param rhs The stream_manipulator be applied to the stream object.
*/
inline const stream& stream_value(const stream& os,
                                  const stream_manipulator& rhs) {
  if (rhs == stream_manipulator::endl) {
    append_char(os, '\n');
  } else {
    const_cast<stream&>(os).set_stream_mode(static_cast<stream_mode>(rhs));
  }
  return os;
}

/**
@brief Inline function that streams a precision manipulator to a stream object,
applying the precision value.
@param os The stream object to be streamed to.
@param rhs The precision_manipulator be applied to the stream object.
*/
inline const stream& stream_value(const stream& os,
                                  const precision_manipulator& rhs) {
  const_cast<stream&>(os).set_precision(rhs.get_precision());
  return os;
}

/** @brief Sets the width used for streaming data to the value specified by rhs
 * @param os The stream object to be streamed to
 * @param rhs Width manipulator specifying the new width
 * @return Original stream object
 */
inline const stream& stream_value(const stream& os,
                                  const detail::width_manipulator& rhs) {
  const_cast<stream&>(os).set_width(rhs.get_width());
  return os;
}

/** @brief Streams an id object into a stream object
 * @tparam dimensions Number of dimensions of the id
 * @param os The stream object to be streamed to
 * @param rhs id object
 * @param typeStr Optional type name
 * @return Original stream object
 */
template <int dimensions>
inline const stream& stream_value(const stream& os, const id<dimensions>& rhs,
                                  char* typeStr = nullptr) {
  char typeStrArray[] = "id<%>";
  if (typeStr == nullptr) {
    typeStr = typeStrArray;
  }
  auto numChars = detail::strlen(typeStr);
  append_type_string(os, typeStr, numChars, dimensions);
  append_index_array(os, rhs, dimensions);
  return os;
}

/** @brief Streams a range object into a stream object
 * @tparam dimensions Number of dimensions of the range
 * @param os The stream object to be streamed to
 * @param rhs range object
 * @param typeStr Optional type name
 * @return Original stream object
 */
template <int dimensions>
inline const stream& stream_value(const stream& os,
                                  const range<dimensions>& rhs,
                                  char* typeStr = nullptr) {
  char typeStrArray[] = "range<%>";
  if (typeStr == nullptr) {
    typeStr = typeStrArray;
  }
  auto numChars = detail::strlen(typeStr);
  append_type_string(os, typeStr, numChars, dimensions);
  append_index_array(os, rhs, dimensions);
  return os;
}

namespace detail {

/** @brief Streams an item offset.
 *
 *        Since the item object in this case doesn't have an offset,
 *        it doesn't stream anything.
 * @param os The stream object to be streamed to
 * @param rhs item object
 */
template <int dimensions>
inline void stream_item_offset(const stream& os,
                               const item<dimensions, false>& rhs) {
  (void)os;
  (void)rhs;
  // No offset, do nothing
}

/** @brief Streams an item offset.
 * @param os The stream object to be streamed to
 * @param rhs item object
 */
template <int dimensions>
inline void stream_item_offset(const stream& os,
                               const item<dimensions, true>& rhs) {
  char typeStr[] = "offset";
  stream_value(os, rhs.get_offset(), typeStr);
}

}  // namespace detail

/** @brief Streams an item object into a stream object
 * @tparam dimensions Number of dimensions of the item
 * @tparam with_offset Whether the item object has an offset or not
 * @param os The stream object to be streamed to
 * @param rhs item object
 * @return Original stream object
 */
template <int dimensions, bool with_offset>
inline const stream& stream_value(const stream& os,
                                  const item<dimensions, with_offset>& rhs,
                                  char* typeStr = nullptr) {
  char typeStrArray[] = "item<%>";
  if (typeStr == nullptr) {
    typeStr = typeStrArray;
  }
  auto numChars = detail::strlen(typeStr);
  append_type_string(os, typeStr, numChars, dimensions);

  append_string(os, COMPUTECPP_OPEN_BRACE);

  stream_value(os, rhs.get_id());
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  stream_value(os, rhs.get_range());
  if (with_offset) {
    append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    detail::stream_item_offset(os, rhs);
  }

  append_string(os, COMPUTECPP_CLOSE_BRACE);

  return os;
}

/** @brief Streams an nd_range object into a stream object
 * @tparam dimensions Number of dimensions of the nd_range
 * @param os The stream object to be streamed to
 * @param rhs nd_range object
 * @return Original stream object
 */
template <int dimensions>
static const stream& stream_value(const stream& os,
                                  const nd_range<dimensions>& rhs) {
  {
    char typeStr[] = "nd_range<%>";
    auto numChars = detail::strlen(typeStr);
    append_type_string(os, typeStr, numChars, dimensions);
  }
  append_string(os, COMPUTECPP_OPEN_BRACE);
  {
    char typeStr[] = "global";
    stream_value(os, rhs.get_global_range(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "local";
    stream_value(os, rhs.get_local_range(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "offset";
    stream_value(os, rhs.get_offset(), typeStr);
  }
  append_string(os, COMPUTECPP_CLOSE_BRACE);

  return os;
}

namespace detail {

/** @brief Provides access to protected functions of the nd_item class
 * @tparam dimensions Number of dimensions of the nd_item
 */
template <int dimensions>
struct nd_item_stream : protected nd_item<dimensions> {
  using base_t = nd_item<dimensions>;

  /** @brief Constructor from an nd_item
   */
  nd_item_stream(const base_t& rhs) : base_t(rhs) {}

  /** @brief Streams an nd_item object into a stream object
   * @param os The stream object to be streamed to
   * @param rhs nd_item object
   * @return Original stream object
   */
  static const stream& stream_value(const stream& os,
                                    const nd_item_stream<dimensions>& rhs) {
    {
      char typeStr[] = "nd_item<%>";
      auto numChars = detail::strlen(typeStr);
      append_type_string(os, typeStr, numChars, dimensions);
    }
    append_string(os, COMPUTECPP_OPEN_BRACE);
    {
      char typeStr[] = "global";
      cl::sycl::stream_value(os, rhs.get_global_item(), typeStr);
    }
    append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    {
      char typeStr[] = "local";
      cl::sycl::stream_value(os, rhs.get_local_item(), typeStr);
    }
    append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    {
      char typeStr[] = "group_id";
      cl::sycl::stream_value(os, rhs.get_group_id(), typeStr);
    }
    append_string(os, COMPUTECPP_CLOSE_BRACE);

    return os;
  }
};

}  // namespace detail

/** @brief Streams an nd_item object into a stream object
 * @tparam dimensions Number of dimensions of the nd_item
 * @param os The stream object to be streamed to
 * @param rhs nd_item object
 * @return Original stream object
 */
template <int dimensions>
static const stream& stream_value(const stream& os,
                                  const nd_item<dimensions>& rhs) {
  return detail::nd_item_stream<dimensions>::stream_value(os, rhs);
}

/** @brief Streams an h_item object into a stream object
 * @tparam dimensions Number of dimensions of the h_item
 * @param os The stream object to be streamed to
 * @param rhs h_item object
 * @return Original stream object
 */
template <int dimensions>
static const stream& stream_value(const stream& os,
                                  const h_item<dimensions>& rhs) {
  {
    char typeStr[] = "h_item<%>";
    auto numChars = detail::strlen(typeStr);
    append_type_string(os, typeStr, numChars, dimensions);
  }
  append_string(os, COMPUTECPP_OPEN_BRACE);
  {
    char typeStr[] = "global";
    cl::sycl::stream_value(os, rhs.get_global(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "logical_local";
    cl::sycl::stream_value(os, rhs.get_logical_local(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "physical_local";
    cl::sycl::stream_value(os, rhs.get_physical_local(), typeStr);
  }
  append_string(os, COMPUTECPP_CLOSE_BRACE);

  return os;
}

/** @brief Streams a group object into a stream object
 * @tparam dimensions Number of dimensions of the group
 * @param os The stream object to be streamed to
 * @param rhs group object
 * @return Original stream object
 */
template <int dimensions>
static const stream& stream_value(const stream& os,
                                  const group<dimensions>& rhs) {
  {
    char typeStr[] = "group<%>";
    auto numChars = detail::strlen(typeStr);
    append_type_string(os, typeStr, numChars, dimensions);
  }
  append_string(os, COMPUTECPP_OPEN_BRACE);
  {
    char typeStr[] = "global";
    stream_value(os, rhs.get_global_range(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "local";
    stream_value(os, rhs.get_local_range(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "group";
    stream_value(os, rhs.get_group_range(), typeStr);
  }
  append_string(os, COMPUTECPP_VEC_ELEM_DELIMITER);
  {
    char typeStr[] = "id";
    stream_value(os, rhs.get_id(), typeStr);
  }
  append_string(os, COMPUTECPP_CLOSE_BRACE);

  return os;
}

/**
@brief Template struct that contains a function call operator that streams a vec
object to a stream object. Class is specialized for each combination of kVecPart
specifying the part of a vector to stream and kEnabled specifying whether it
streamed or not. This is the default template that does nothing.
@tparam kVecPart The part of the vec to be streamed.
@tparam kEnabled A boolean specifying whether that part of the vec is streamed.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <int kVecPart, bool kEnabled, typename elementT, int kDimensions>
struct stream_vec {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator does nothing.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    (void)rhs;
    return os;
  }
};

/**
@brief Specialization of stream_vec for kVecPart = 2 and kEnabled = true.
@ref stream_vec
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <typename elementT, int kDimensions>
struct stream_vec<2, true, elementT, kDimensions> {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator streams the x and y elements of the vec
  object.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    stream_value(os, static_cast<elementT>(rhs.x()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.y()));
    return os;
  }
};

/**
@brief Specialization of stream_vec for kVecPart = 3 and kEnabled = true. @ref
stream_vec.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <typename elementT, int kDimensions>
struct stream_vec<3, true, elementT, kDimensions> {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator streams the z element of the vec object.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.z()));
    return os;
  }
};

/**
@brief Specialization of stream_vec for kVecPart = 4 and kEnabled = true. @ref
stream_vec.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <typename elementT, int kDimensions>
struct stream_vec<4, true, elementT, kDimensions> {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator streams the w element of the vec object.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.w()));
    return os;
  }
};

/**
@brief Specialization of stream_vec for kVecPart = 8 and kEnabled = true. @ref
stream_vec.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <typename elementT, int kDimensions>
struct stream_vec<8, true, elementT, kDimensions> {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator streams the s0, s1, s2, s3, s4, s5, s6 and
  s7 element of the vec object.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    stream_value(os, static_cast<elementT>(rhs.s0()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s1()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s2()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s3()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s4()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s5()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s6()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s7()));
    return os;
  }
};

/**
@brief Specialization of stream_vec for kVecPart = 16 and kEnabled = true. @ref
stream_vec.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
*/
template <typename elementT, int kDimensions>
struct stream_vec<16, true, elementT, kDimensions> {
  /**
  @brief Inline function call operator that streams part of the vec to a stream
  object. This function call operator streams the s8, s9, sA, sB, sC, sD, sE and
  sF element of the vec object.
  @param os A reference to the stream object.
  @param rhs A reference to the vec object.
  @return A reference to the stream object.
  */
  inline const stream& operator()(const stream& os,
                                  const vec<elementT, kDimensions>& rhs) {
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s8()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.s9()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sA()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sB()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sC()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sD()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sE()));
    stream_value(os, COMPUTECPP_VEC_ELEM_DELIMITER);
    stream_value(os, static_cast<elementT>(rhs.sF()));
    return os;
  }
};

/**
@brief Template stream operator that streams a value to a stream object.
@tparam The type of the value to be streamed.
@param os The stream object to be streamed to.
@param rhs The value to be streamed.
*/
template <typename valueT>
inline const stream& operator<<(const stream& os, valueT rhs) {
  stream_value(os, rhs);
  return os;
}

/**
@brief Template stream operator that streams a vec object to a stream object.
@tparam elementT The element type of the vec.
@tparam kDimensions The number of dimensions of the vec.
@param os The stream object to be streamed to.
@param rhs The vec object to be streamed.
*/
template <typename elementT, int kDimensions>
const stream& operator<<(const stream& os,
                         const vec<elementT, kDimensions> rhs) {
  stream_value(os, COMPUTECPP_OPEN_BRACE);

  /* Stream the first 2 elements of the vec object if the dimensions are greater
   * than 1 and less than 8. */
  stream_vec<2, (kDimensions > 1 && kDimensions < 8), elementT, kDimensions>()(
      os, rhs);

  /* Stream the third elements of the vec object if the dimensions are greater
   * than 2 and less than 8. */
  stream_vec<3, (kDimensions > 2 && kDimensions < 8), elementT, kDimensions>()(
      os, rhs);

  /* Stream the fourth elements of the vec object if the dimensions are greater
   * than 3 and less than 8. */
  stream_vec<4, (kDimensions > 3 && kDimensions < 8), elementT, kDimensions>()(
      os, rhs);

  /* Stream the first 8 elements of the vec object if the dimensions are greater
   * than 4. */
  stream_vec<8, (kDimensions > 4), elementT, kDimensions>()(os, rhs);

  /* Stream the second 8 elements of the vec object if the dimensions are
   * greater than 8. */
  stream_vec<16, (kDimensions > 8), elementT, kDimensions>()(os, rhs);

  stream_value(os, COMPUTECPP_CLOSE_BRACE);

  return os;
}

/** @brief Streams a swizzled_vec object to a stream object
 * @tparam elementT The element type of the swizzled_vec
 * @tparam kDimensions The number of dimensions of the swizzled_vec
 * @param os The stream object to be streamed to
 * @param rhs The swizzled_vec object to be streamed
 * @return Original stream object
 */
template <typename elementT, int kDimensions, int... swizzleIndexes>
const stream& operator<<(
    const stream& os,
    const swizzled_vec<elementT, kDimensions, swizzleIndexes...> rhs) {
  return (os << (static_cast<vec<elementT, sizeof...(swizzleIndexes)>>(rhs)));
}

}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the buffer class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <>
struct hash<cl::sycl::stream> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::stream& object) const {
#ifdef __SYCL_DEVICE_ONLY__
    (void)object;
    return 0;
#else
    hash<cl::sycl::dmem_shptr> hasher;
    return hasher(object.get_buffer().get()->get_impl());
#endif
  }
};
}  // namespace std

#undef COMPUTECPP_VEC_ELEM_DELIMITER
#undef COMPUTECPP_NEGATIVE_SIGN
#undef COMPUTECPP_DECIMAL_POINT
#undef COMPUTECPP_HEX_PREFIX
#undef COMPUTECPP_POSITIVE_EXP_PREFIX
#undef COMPUTECPP_NEGATIVE_EXP_PREFIX
#undef COMPUTECPP_OPEN_BRACE
#undef COMPUTECPP_CLOSE_BRACE

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_STREAM_H_

////////////////////////////////////////////////////////////////////////////////
