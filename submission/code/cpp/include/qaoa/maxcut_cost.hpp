#pragma once
#include "types.hpp"
#include <vector>

namespace qaoa {

/// Compute cut value for each computational-basis state z in {0,...,2^n-1}.
std::vector<double> all_cut_values(const Graph& g);

} // namespace qaoa
