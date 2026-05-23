#pragma once
#include "types.hpp"
#include <vector>

namespace qaoa {

/// Evaluate QAOA expected cut for a single parameter vector theta of length 2*p.
/// theta layout: [gamma_1,...,gamma_p, beta_1,...,beta_p] (blocked).
double evaluate(const Graph& g, int p, const std::vector<double>& theta, Timings* timings = nullptr);

/// Thread-parallel batch evaluation over multiple theta vectors.
std::vector<double> evaluate_batch(const Graph& g, int p,
                                   const std::vector<std::vector<double>>& thetas, int threads);

} // namespace qaoa
