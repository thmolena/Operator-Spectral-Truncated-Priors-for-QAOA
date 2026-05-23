#pragma once
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace qaoa {

/// Return compiler identification string (clang/gcc version).
inline std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

/// Return current UTC timestamp in ISO-8601 format.
inline std::string timestamp_utc() {
    auto t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

} // namespace qaoa
