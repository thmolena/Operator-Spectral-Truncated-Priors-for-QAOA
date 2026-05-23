#pragma once
#include <chrono>

namespace qaoa {
namespace detail {

inline auto now() { return std::chrono::steady_clock::now(); }

inline double ms(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace detail
} // namespace qaoa
