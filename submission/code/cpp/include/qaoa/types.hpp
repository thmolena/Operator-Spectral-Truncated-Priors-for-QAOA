#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qaoa {

using Edge = std::pair<int, int>;

struct Graph {
    int n{};
    std::vector<Edge> edges;
};

struct Timings {
    double phase_ms{};
    double mixer_ms{};
    double expectation_ms{};
    double reduction_ms{};
    double allocation_ms{};
};

} // namespace qaoa
