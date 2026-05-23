#include "qaoa_cpu.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace qaoa {

// --- graph.hpp implementation ---

Graph load_graph_csv(const std::string& path, int n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open graph");
    std::string line;
    std::getline(f, line); // skip header
    Graph g;
    g.n = n;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream is(line);
        int u, v;
        if (is >> u >> v) g.edges.emplace_back(u, v);
    }
    return g;
}

// --- maxcut_cost.hpp implementation ---

std::vector<double> all_cut_values(const Graph& g) {
    const std::uint64_t dim = 1ull << g.n;
    std::vector<double> c(dim, 0.0);
    for (std::uint64_t z = 0; z < dim; ++z) {
        double val = 0.0;
        for (auto [u, v] : g.edges) {
            if (((z >> u) & 1ull) != ((z >> v) & 1ull)) val += 1.0;
        }
        c[z] = val;
    }
    return c;
}

// --- statevector.hpp implementation ---

namespace detail {

void apply_mixer(std::vector<cd>& st, double beta, int n) {
    const cd s(0.0, -std::sin(beta));
    const double c = std::cos(beta);
    const std::uint64_t size = st.size();
    for (int q = 0; q < n; ++q) {
        std::uint64_t step = 1ull << q, jump = step << 1;
        for (std::uint64_t base = 0; base < size; base += jump) {
            for (std::uint64_t i = 0; i < step; ++i) {
                cd a = st[base + i], b = st[base + step + i];
                st[base + i] = c * a + s * b;
                st[base + step + i] = s * a + c * b;
            }
        }
    }
}

double expectation_value(const std::vector<cd>& state, const std::vector<double>& cost) {
    double acc = 0.0;
    for (std::size_t z = 0; z < state.size(); ++z)
        acc += std::norm(state[z]) * cost[z];
    return acc;
}

} // namespace detail

// --- evaluator.hpp implementation ---

double evaluate(const Graph& g, int p, const std::vector<double>& theta, Timings* timings) {
    if (p <= 0) throw std::invalid_argument("p must be positive");
    if (theta.size() != static_cast<size_t>(2 * p))
        throw std::invalid_argument("theta dimension must be 2*p");
    auto t0 = detail::now();
    auto costs = all_cut_values(g);
    const std::uint64_t dim = 1ull << g.n;
    std::vector<detail::cd> st(dim, detail::cd(1.0 / std::sqrt(static_cast<double>(dim)), 0.0));
    auto t1 = detail::now();
    double phase = 0, mixer = 0;
    for (int l = 0; l < p; ++l) {
        auto a = detail::now();
        double gamma = theta[l];
        for (std::uint64_t z = 0; z < dim; ++z)
            st[z] *= std::exp(detail::cd(0.0, -gamma * costs[z]));
        auto b = detail::now();
        detail::apply_mixer(st, theta[p + l], g.n);
        auto c = detail::now();
        phase += detail::ms(a, b);
        mixer += detail::ms(b, c);
    }
    auto e0 = detail::now();
    double acc = detail::expectation_value(st, costs);
    auto e1 = detail::now();
    if (timings) {
        timings->allocation_ms = detail::ms(t0, t1);
        timings->phase_ms = phase;
        timings->mixer_ms = mixer;
        timings->expectation_ms = detail::ms(e0, e1);
        timings->reduction_ms = timings->expectation_ms;
    }
    return acc;
}

std::vector<double> evaluate_batch(const Graph& g, int p,
                                   const std::vector<std::vector<double>>& thetas, int threads) {
    if (threads <= 0) threads = 1;
    std::vector<double> out(thetas.size());
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (size_t i = t; i < thetas.size(); i += threads)
                out[i] = evaluate(g, p, thetas[i], nullptr);
        });
    }
    for (auto& w : workers) w.join();
    return out;
}

} // namespace qaoa
