#pragma once
#include "types.hpp"
#include <complex>
#include <vector>

namespace qaoa {
namespace detail {

using cd = std::complex<double>;

/// Apply single-qubit X-rotation (mixer) on all qubits in-place.
void apply_mixer(std::vector<cd>& state, double beta, int n);

/// Deterministic fixed-order expectation: sum |a_z|^2 * cost[z].
double expectation_value(const std::vector<cd>& state, const std::vector<double>& cost);

} // namespace detail
} // namespace qaoa
