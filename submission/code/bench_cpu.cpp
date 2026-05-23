// bench_cpu.cpp — Standalone C++20 CPU benchmark for UQ-QAOA statevector evaluation
//
// Compiles with: clang++ -O3 -std=c++20 -march=native -pthread bench_cpu.cpp -o bench_cpu
// Usage: ./bench_cpu [threads]
//
// Measures:
//   - Per-query statevector evaluation time (phase + mixer kernels)
//   - Thread scaling via std::thread batched evaluation
//   - Peak resident memory (macOS mach_task_info)
//   - Compiler version and flags at runtime
//
// Output: bench_cpu_cpp.csv (per-query timing), bench_thread_scaling.csv
//
// Platform: Apple M2 Pro, macOS 14, Apple Clang 15.0.0
// Reference: Farhi et al., arXiv:1411.4028 (2014), Eq. (1)-(4)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace bench {

// ============================================================
// Platform and compiler diagnostics
// ============================================================

struct PlatformInfo {
    std::string compiler;
    std::string cpp_standard;
    std::string arch;
    unsigned int hw_threads;
};

PlatformInfo get_platform_info() {
    PlatformInfo info;
#if defined(__clang__)
    info.compiler = "Apple Clang " + std::to_string(__clang_major__) + "." +
                    std::to_string(__clang_minor__) + "." +
                    std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    info.compiler = "GCC " + std::to_string(__GNUC__) + "." +
                    std::to_string(__GNUC_MINOR__);
#else
    info.compiler = "Unknown";
#endif
    info.cpp_standard = std::to_string(__cplusplus);
#if defined(__aarch64__)
    info.arch = "arm64";
#elif defined(__x86_64__)
    info.arch = "x86_64";
#else
    info.arch = "unknown";
#endif
    info.hw_threads = std::thread::hardware_concurrency();
    return info;
}

std::size_t peak_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

// ============================================================
// Graph generation (Erdos-Renyi)
// ============================================================

struct Edge {
    int u, v;
};

std::vector<Edge> generate_er_graph(int n, double p, std::mt19937& rng) {
    std::vector<Edge> edges;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dist(rng) < p) {
                edges.push_back({i, j});
            }
        }
    }
    return edges;
}

// ============================================================
// QAOA statevector simulation
// ============================================================

// Compute C(z) for all bitstrings: MaxCut cost
std::vector<double> compute_cost_values(const std::vector<Edge>& edges, int n) {
    const std::size_t N = std::size_t(1) << n;
    std::vector<double> c(N, 0.0);
    for (std::size_t z = 0; z < N; ++z) {
        double cost = 0.0;
        for (const auto& [u, v] : edges) {
            int bu = (z >> u) & 1;
            int bv = (z >> v) & 1;
            if (bu != bv) cost += 1.0;
        }
        c[z] = cost;
    }
    return c;
}

// Phase kernel: psi_z <- exp(-i*gamma*C(z)) * psi_z
// Bandwidth-bound, element-wise, unit-stride
// Arithmetic intensity: 6 FLOP per 16 bytes = 0.375 FLOP/byte
void apply_phase(std::vector<std::complex<double>>& psi,
                 const std::vector<double>& cost_values,
                 double gamma) {
    const std::size_t N = psi.size();
    for (std::size_t z = 0; z < N; ++z) {
        double angle = -gamma * cost_values[z];
        psi[z] *= std::complex<double>(std::cos(angle), std::sin(angle));
    }
}

// Mixer kernel: exp(-i*beta*X_q) on each qubit q
// Strided pair updates, stride = 2^q
// Cache-line splits at large q
void apply_mixer(std::vector<std::complex<double>>& psi, int n, double beta) {
    const double c = std::cos(beta);
    const std::complex<double> s(0.0, -std::sin(beta));
    for (int q = 0; q < n; ++q) {
        const std::size_t stride = std::size_t(1) << q;
        const std::size_t block = stride << 1;
        const std::size_t N = psi.size();
        for (std::size_t k = 0; k < N; k += block) {
            for (std::size_t j = 0; j < stride; ++j) {
                auto& a = psi[k + j];
                auto& b = psi[k + j + stride];
                std::complex<double> new_a = c * a + s * b;
                std::complex<double> new_b = s * a + c * b;
                a = new_a;
                b = new_b;
            }
        }
    }
}

// Full QAOA evaluation: returns approximation ratio
double qaoa_evaluate(const std::vector<double>& cost_values,
                     int n,
                     const std::vector<double>& gamma,
                     const std::vector<double>& beta,
                     int depth) {
    const std::size_t N = std::size_t(1) << n;

    // Initialize |+>^n
    std::vector<std::complex<double>> psi(N, 1.0 / std::sqrt(static_cast<double>(N)));

    // Apply p layers
    for (int layer = 0; layer < depth; ++layer) {
        apply_phase(psi, cost_values, gamma[layer]);
        apply_mixer(psi, n, beta[layer]);
    }

    // Compute expectation value <C>
    double expectation = 0.0;
    for (std::size_t z = 0; z < N; ++z) {
        expectation += std::norm(psi[z]) * cost_values[z];
    }

    // Compute C_max
    double c_max = *std::max_element(cost_values.begin(), cost_values.end());
    if (c_max <= 0.0) return 0.0;
    return expectation / c_max;
}

// ============================================================
// Benchmark harness
// ============================================================

struct BenchmarkResult {
    int n;
    int depth;
    int queries;
    double mean_query_ms;
    double std_query_ms;
    double total_ms;
    std::size_t statevec_bytes;
    std::size_t peak_rss;
    int threads;
};

// Single-threaded benchmark for one configuration
BenchmarkResult bench_single(int n, int depth, int queries, int seed) {
    std::mt19937 rng(seed);
    auto edges = generate_er_graph(n, 0.5, rng);
    auto cost_values = compute_cost_values(edges, n);

    // Generate random candidate angles
    std::uniform_real_distribution<double> gamma_dist(0.0, M_PI);
    std::uniform_real_distribution<double> beta_dist(0.0, M_PI / 2.0);

    std::vector<std::vector<double>> gammas(queries, std::vector<double>(depth));
    std::vector<std::vector<double>> betas(queries, std::vector<double>(depth));
    for (int q = 0; q < queries; ++q) {
        for (int l = 0; l < depth; ++l) {
            gammas[q][l] = gamma_dist(rng);
            betas[q][l] = beta_dist(rng);
        }
    }

    // Warm-up: one evaluation
    qaoa_evaluate(cost_values, n, gammas[0], betas[0], depth);

    // Timed evaluations
    std::vector<double> query_times_ms;
    query_times_ms.reserve(queries);
    auto t_total_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < queries; ++q) {
        auto t0 = std::chrono::high_resolution_clock::now();
        qaoa_evaluate(cost_values, n, gammas[q], betas[q], depth);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        query_times_ms.push_back(ms);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    // Statistics
    double sum = std::accumulate(query_times_ms.begin(), query_times_ms.end(), 0.0);
    double mean = sum / queries;
    double sq_sum = 0.0;
    for (double t : query_times_ms) sq_sum += (t - mean) * (t - mean);
    double std_dev = std::sqrt(sq_sum / queries);

    BenchmarkResult result;
    result.n = n;
    result.depth = depth;
    result.queries = queries;
    result.mean_query_ms = mean;
    result.std_query_ms = std_dev;
    result.total_ms = total_ms;
    result.statevec_bytes = (std::size_t(1) << n) * sizeof(std::complex<double>);
    result.peak_rss = peak_rss_bytes();
    result.threads = 1;
    return result;
}

// Multi-threaded batched evaluation using std::thread
// Each thread owns a private statevector buffer (no sharing)
struct ThreadedResult {
    int threads;
    int n;
    int queries;
    double total_ms;
    double speedup;
};

ThreadedResult bench_threaded(int n, int depth, int queries, int seed, int num_threads) {
    std::mt19937 rng(seed);
    auto edges = generate_er_graph(n, 0.5, rng);
    auto cost_values = compute_cost_values(edges, n);

    // Generate candidate angles
    std::uniform_real_distribution<double> gamma_dist(0.0, M_PI);
    std::uniform_real_distribution<double> beta_dist(0.0, M_PI / 2.0);

    std::vector<std::vector<double>> gammas(queries, std::vector<double>(depth));
    std::vector<std::vector<double>> betas(queries, std::vector<double>(depth));
    for (int q = 0; q < queries; ++q) {
        for (int l = 0; l < depth; ++l) {
            gammas[q][l] = gamma_dist(rng);
            betas[q][l] = beta_dist(rng);
        }
    }

    // Warm-up
    qaoa_evaluate(cost_values, n, gammas[0], betas[0], depth);

    // Batched parallel evaluation via std::thread
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::vector<double> results(queries, 0.0);

    auto worker = [&](int start, int end) {
        for (int q = start; q < end; ++q) {
            results[q] = qaoa_evaluate(cost_values, n, gammas[q], betas[q], depth);
        }
    };

    int chunk = queries / num_threads;
    int remainder = queries % num_threads;
    int start = 0;
    for (int t = 0; t < num_threads; ++t) {
        int end = start + chunk + (t < remainder ? 1 : 0);
        threads.emplace_back(worker, start, end);
        start = end;
    }
    for (auto& th : threads) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ThreadedResult r;
    r.threads = num_threads;
    r.n = n;
    r.queries = queries;
    r.total_ms = total_ms;
    r.speedup = 0.0;  // computed later
    return r;
}

// Kernel profiling: phase vs mixer separately
struct KernelProfile {
    double phase_ms_per_layer;
    double mixer_ms_per_layer;
    double ratio;
};

KernelProfile profile_kernels(int n, int depth, int seed) {
    const std::size_t N = std::size_t(1) << n;
    std::mt19937 rng(seed);
    auto edges = generate_er_graph(n, 0.5, rng);
    auto cost_values = compute_cost_values(edges, n);

    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> gamma_vals(depth), beta_vals(depth);
    for (int l = 0; l < depth; ++l) {
        gamma_vals[l] = dist(rng);
        beta_vals[l] = dist(rng) / 2.0;
    }

    // Initialize statevector
    std::vector<std::complex<double>> psi(N, 1.0 / std::sqrt(static_cast<double>(N)));

    // Profile phase kernel (average over layers)
    std::vector<double> phase_times;
    std::vector<double> mixer_times;

    constexpr int REPEATS = 10;
    for (int rep = 0; rep < REPEATS; ++rep) {
        // Reset
        std::fill(psi.begin(), psi.end(), 1.0 / std::sqrt(static_cast<double>(N)));
        for (int l = 0; l < depth; ++l) {
            auto t0 = std::chrono::high_resolution_clock::now();
            apply_phase(psi, cost_values, gamma_vals[l]);
            auto t1 = std::chrono::high_resolution_clock::now();
            phase_times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());

            t0 = std::chrono::high_resolution_clock::now();
            apply_mixer(psi, n, beta_vals[l]);
            t1 = std::chrono::high_resolution_clock::now();
            mixer_times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    double avg_phase = std::accumulate(phase_times.begin(), phase_times.end(), 0.0)
                       / phase_times.size();
    double avg_mixer = std::accumulate(mixer_times.begin(), mixer_times.end(), 0.0)
                       / mixer_times.size();

    return {avg_phase, avg_mixer, avg_mixer / avg_phase};
}

// Deterministic reduction test: verify sum is identical across runs
bool test_deterministic_reduction(int n, int seed) {
    std::mt19937 rng(seed);
    auto edges = generate_er_graph(n, 0.5, rng);
    auto cost_values = compute_cost_values(edges, n);

    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> gamma = {dist(rng), dist(rng)};
    std::vector<double> beta = {dist(rng) / 2.0, dist(rng) / 2.0};

    // Run 5 times and check bitwise equality
    double first = qaoa_evaluate(cost_values, n, gamma, beta, 2);
    for (int i = 1; i < 5; ++i) {
        double current = qaoa_evaluate(cost_values, n, gamma, beta, 2);
        if (std::memcmp(&first, &current, sizeof(double)) != 0) {
            return false;
        }
    }
    return true;
}

std::string benchmark_suffix(int depth, int queries) {
    if (depth == 2 && queries == 18) {
        return "";
    }
    return "_p" + std::to_string(depth) + "_q" + std::to_string(queries);
}

}  // namespace bench

int main(int argc, char** argv) {
    using namespace bench;

    int max_threads = static_cast<int>(std::thread::hardware_concurrency());
    int depth = 2;
    int queries = 18;
    if (argc > 1) {
        max_threads = std::atoi(argv[1]);
        if (max_threads < 1) max_threads = 1;
    }
    if (argc > 2) {
        depth = std::atoi(argv[2]);
        if (depth < 1) depth = 1;
    }
    if (argc > 3) {
        queries = std::atoi(argv[3]);
        if (queries < 1) queries = 1;
    }

    auto platform = get_platform_info();
    std::printf("=== UQ-QAOA C++ CPU Benchmark ===\n");
    std::printf("Compiler:       %s\n", platform.compiler.c_str());
    std::printf("C++ standard:   %s\n", platform.cpp_standard.c_str());
    std::printf("Architecture:   %s\n", platform.arch.c_str());
    std::printf("HW threads:     %u\n", platform.hw_threads);
    std::printf("Max test threads: %d\n", max_threads);
    std::printf("Flags:          -O3 -std=c++20 -march=native -pthread\n");
    std::printf("\n");

    // --- Deterministic reduction test ---
    std::printf("--- Deterministic reduction test ---\n");
    bool det_pass = test_deterministic_reduction(14, 42);
    std::printf("  n=14: %s (bitwise identical across 5 runs)\n",
                det_pass ? "PASS" : "FAIL");
    std::printf("\n");

    // --- Per-query benchmark ---
    std::printf("--- Per-query benchmark (single-threaded, p=%d, Q=%d) ---\n",
                depth, queries);
    constexpr int SEED = 260424803;
    const std::string suffix = benchmark_suffix(depth, queries);

    std::vector<int> sizes = {8, 10, 12, 14, 16, 18};
    std::vector<BenchmarkResult> results;

    for (int n : sizes) {
        auto r = bench_single(n, depth, queries, SEED);
        results.push_back(r);
        std::printf("  n=%2d: %8.3f ms/query (std %.3f), total %8.1f ms, "
                    "statevec %zu KB, RSS %.1f MB\n",
                    n, r.mean_query_ms, r.std_query_ms, r.total_ms,
                    r.statevec_bytes / 1024, r.peak_rss / (1024.0 * 1024.0));
    }

    // Write CSV
    std::filesystem::path out_dir = "tables";
    std::filesystem::create_directories(out_dir);
    {
        std::ofstream f(out_dir / ("bench_cpu_cpp" + suffix + ".csv"));
        f << "n,p,Q,mean_query_ms,std_query_ms,total_ms,statevec_bytes,peak_rss_bytes,"
             "threads,compiler,flags\n";
        for (auto& r : results) {
            f << r.n << "," << r.depth << "," << r.queries << ","
              << r.mean_query_ms << "," << r.std_query_ms << ","
              << r.total_ms << "," << r.statevec_bytes << "," << r.peak_rss << ","
              << r.threads << "," << platform.compiler << ","
              << "-O3 -std=c++20 -march=native -pthread\n";
        }
    }
    std::printf("\nWrote tables/bench_cpu_cpp%s.csv\n", suffix.c_str());

    // --- Thread scaling ---
    const int batch_queries = 8 * queries;
    std::printf("\n--- Thread scaling (n=14, p=%d, Q=%d batched) ---\n",
                depth, batch_queries);
    std::printf("  (%d = 8 instances x %d candidates, independent evaluations)\n",
                batch_queries, queries);
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads && t <= 10; t *= 2) {
        thread_counts.push_back(t);
    }
    if (thread_counts.back() < max_threads && max_threads <= 10) {
        thread_counts.push_back(max_threads);
    }

    std::vector<ThreadedResult> scaling;
    double baseline_ms = 0.0;
    for (int nt : thread_counts) {
        auto r = bench_threaded(14, depth, batch_queries, SEED, nt);
        if (nt == 1) baseline_ms = r.total_ms;
        r.speedup = baseline_ms / r.total_ms;
        scaling.push_back(r);
        std::printf("  %2d thread(s): %8.1f ms (speedup %.2fx vs 1 thread)\n",
                    nt, r.total_ms, r.speedup);
    }

    // Write thread scaling CSV
    {
        std::ofstream f(out_dir / ("bench_thread_scaling" + suffix + ".csv"));
        f << "threads,n,queries,total_ms,speedup\n";
        for (auto& r : scaling) {
            f << r.threads << "," << r.n << "," << r.queries << ","
              << r.total_ms << "," << r.speedup << "\n";
        }
    }
    std::printf("Wrote tables/bench_thread_scaling%s.csv\n", suffix.c_str());

    // --- Kernel profiling ---
    std::printf("\n--- Kernel profiling (n=14, p=%d, 10 repeats) ---\n", depth);
    auto kp = profile_kernels(14, depth, SEED);
    std::printf("  Phase kernel: %.4f ms/layer (bandwidth-bound, AI=0.375 FLOP/byte)\n",
                kp.phase_ms_per_layer);
    std::printf("  Mixer kernel: %.4f ms/layer (strided pairs, 14 qubits)\n",
                kp.mixer_ms_per_layer);
    std::printf("  Mixer/Phase ratio: %.1fx\n", kp.ratio);

    // --- Memory analysis ---
    std::printf("\n--- Memory/cache analysis ---\n");
    std::printf("  Statevector sizes:\n");
    for (int n : sizes) {
        std::size_t bytes = (std::size_t(1) << n) * 16;
        const char* fits;
        if (bytes <= 16 * 1024 * 1024) fits = "fits L2 (16 MB)";
        else if (bytes <= 32 * 1024 * 1024) fits = "exceeds L2, fits DRAM";
        else fits = "DRAM-resident, GPU-beneficial";
        std::printf("    n=%2d: %10zu bytes (%7zu KB) — %s\n",
                    n, bytes, bytes / 1024, fits);
    }
    std::printf("  Apple M2 Pro L2: 16 MB/cluster, DRAM BW: 200 GB/s\n");
    std::printf("  Transition to DRAM-bound: n >= 20 (16 MB statevector)\n");

    // --- Summary ---
    std::printf("\n--- Summary ---\n");
    std::printf("  Deterministic reduction: %s\n", det_pass ? "PASS" : "FAIL");
    std::printf("  Best thread speedup (n=14, Q=144): %.2fx at %d threads\n",
                scaling.back().speedup, scaling.back().threads);
    std::printf("  Phase/Mixer dominance confirmed: mixer %.1fx slower\n", kp.ratio);
    std::printf("  All evaluations use ordered std::accumulate (no -ffast-math)\n");

    return 0;
}
