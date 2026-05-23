#pragma once
#include "types.hpp"
#include <string>

namespace qaoa {

/// Load a graph from a CSV file with columns (u, v).
Graph load_graph_csv(const std::string& path, int n);

} // namespace qaoa
