#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uq_qaoa {

enum class AlgorithmId {
    random_search,
    heuristic_schedule,
    knn_transfer,
    tqa_ramp,
    gnn_point,
    uq_qaoa
};

std::string to_string(AlgorithmId id) {
    switch (id) {
    case AlgorithmId::random_search:
        return "random";
    case AlgorithmId::heuristic_schedule:
        return "heuristic";
    case AlgorithmId::knn_transfer:
        return "knn";
    case AlgorithmId::tqa_ramp:
        return "tqa";
    case AlgorithmId::gnn_point:
        return "gnn_point";
    case AlgorithmId::uq_qaoa:
        return "uq_qaoa";
    }
    return "unknown";
}

struct ParameterDomain {
    double gamma_min = 0.0;
    double gamma_max = M_PI;
    double beta_min = 0.0;
    double beta_max = M_PI_2;
    std::size_t depth = 2;

    [[nodiscard]] std::size_t dimension() const {
        return 2 * depth;
    }
};

struct ParameterVector {
    std::vector<double> values;
};

struct GraphInstance {
    std::string graph_id;
    std::string family;
    std::size_t num_vertices = 0;
    std::size_t num_edges = 0;
    std::vector<double> features;
};

struct PredictorOutput {
    ParameterVector mean;
    std::vector<double> diagonal_covariance;
};

struct CandidateProposal {
    AlgorithmId algorithm = AlgorithmId::random_search;
    ParameterVector parameters;
    std::string provenance;
};

struct EvaluationRecord {
    AlgorithmId algorithm = AlgorithmId::random_search;
    std::string graph_id;
    std::size_t query_index = 0;
    ParameterVector parameters;
    double objective = -std::numeric_limits<double>::infinity();
    double approximation_ratio = -std::numeric_limits<double>::infinity();
    std::string provenance;
};

struct SearchResult {
    AlgorithmId algorithm = AlgorithmId::random_search;
    std::string graph_id;
    std::optional<EvaluationRecord> best_record;
    std::vector<EvaluationRecord> trace;
};

struct ReferenceEntry {
    std::vector<double> graph_features;
    ParameterVector parameters;
};

struct ExperimentConfig {
    ParameterDomain domain;
    std::size_t query_budget = 1;
    unsigned int seed = 7;
    double trust_region_radius = 1.0;
    double covariance_floor = 1.0e-6;
    std::vector<ParameterVector> heuristic_schedule;
    std::vector<ReferenceEntry> knn_library;
    std::vector<ParameterVector> local_refinement_offsets;
    double tqa_total_time = 1.0;
};

std::ostream& operator<<(std::ostream& os, const ParameterVector& vector) {
    os << '[';
    for (std::size_t index = 0; index < vector.values.size(); ++index) {
        if (index != 0) {
            os << ", ";
        }
        os << vector.values[index];
    }
    os << ']';
    return os;
}

std::string parameter_string(const ParameterVector& vector) {
    std::ostringstream stream;
    stream << vector;
    return stream.str();
}

double clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

ParameterVector clamp_to_domain(const ParameterVector& parameters, const ParameterDomain& domain) {
    ParameterVector bounded = parameters;
    bounded.values.resize(domain.dimension(), 0.0);
    for (std::size_t layer = 0; layer < domain.depth; ++layer) {
        bounded.values[layer] = clamp(bounded.values[layer], domain.gamma_min, domain.gamma_max);
        bounded.values[domain.depth + layer] = clamp(
            bounded.values[domain.depth + layer],
            domain.beta_min,
            domain.beta_max);
    }
    return bounded;
}

double squared_l2_distance(std::span<const double> left, std::span<const double> right) {
    const std::size_t dimension = std::min(left.size(), right.size());
    double value = 0.0;
    for (std::size_t index = 0; index < dimension; ++index) {
        const double diff = left[index] - right[index];
        value += diff * diff;
    }
    return value;
}

class ObjectiveEvaluator {
public:
    virtual ~ObjectiveEvaluator() = default;
    virtual EvaluationRecord evaluate(
        AlgorithmId algorithm,
        const GraphInstance& graph,
        const ParameterVector& parameters,
        std::size_t query_index,
        const std::string& provenance) const = 0;
};

class Predictor {
public:
    virtual ~Predictor() = default;
    virtual PredictorOutput predict(const GraphInstance& graph, const ParameterDomain& domain) const = 0;
};

class CandidateGenerator {
public:
    virtual ~CandidateGenerator() = default;
    virtual AlgorithmId id() const = 0;
    virtual void reset(const GraphInstance& graph, const ExperimentConfig& config, const Predictor* predictor) = 0;
    virtual CandidateProposal next(std::mt19937& engine, std::size_t query_index) = 0;
};

class RandomSearchGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::random_search; }

    void reset(const GraphInstance&, const ExperimentConfig& config, const Predictor*) override {
        domain_ = config.domain;
    }

    CandidateProposal next(std::mt19937& engine, std::size_t query_index) override {
        std::uniform_real_distribution<double> gamma_dist(domain_.gamma_min, domain_.gamma_max);
        std::uniform_real_distribution<double> beta_dist(domain_.beta_min, domain_.beta_max);
        ParameterVector proposal;
        proposal.values.resize(domain_.dimension());
        for (std::size_t layer = 0; layer < domain_.depth; ++layer) {
            proposal.values[layer] = gamma_dist(engine);
            proposal.values[domain_.depth + layer] = beta_dist(engine);
        }
        return {id(), proposal, "uniform-domain-sample-" + std::to_string(query_index)};
    }

private:
    ParameterDomain domain_{};
};

class HeuristicScheduleGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::heuristic_schedule; }

    void reset(const GraphInstance&, const ExperimentConfig& config, const Predictor*) override {
        domain_ = config.domain;
        schedule_ = config.heuristic_schedule;
        offsets_ = config.local_refinement_offsets;
    }

    CandidateProposal next(std::mt19937&, std::size_t query_index) override {
        if (schedule_.empty()) {
            throw std::runtime_error("heuristic_schedule requires at least one configured parameter vector");
        }
        ParameterVector proposal = schedule_[std::min(query_index, schedule_.size() - 1)];
        if (query_index >= schedule_.size() && !offsets_.empty()) {
            const ParameterVector& offset = offsets_[(query_index - schedule_.size()) % offsets_.size()];
            proposal.values.resize(domain_.dimension(), 0.0);
            for (std::size_t index = 0; index < proposal.values.size() && index < offset.values.size(); ++index) {
                proposal.values[index] += offset.values[index];
            }
        }
        return {id(), clamp_to_domain(proposal, domain_), "configured-heuristic-" + std::to_string(query_index)};
    }

private:
    ParameterDomain domain_{};
    std::vector<ParameterVector> schedule_;
    std::vector<ParameterVector> offsets_;
};

class KnnTransferGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::knn_transfer; }

    void reset(const GraphInstance& graph, const ExperimentConfig& config, const Predictor*) override {
        domain_ = config.domain;
        offsets_ = config.local_refinement_offsets;
        if (config.knn_library.empty()) {
            throw std::runtime_error("knn_transfer requires a non-empty reference library");
        }
        auto best_it = config.knn_library.begin();
        double best_distance = squared_l2_distance(graph.features, best_it->graph_features);
        for (auto it = config.knn_library.begin() + 1; it != config.knn_library.end(); ++it) {
            const double distance = squared_l2_distance(graph.features, it->graph_features);
            if (distance < best_distance) {
                best_distance = distance;
                best_it = it;
            }
        }
        anchor_ = best_it->parameters;
    }

    CandidateProposal next(std::mt19937&, std::size_t query_index) override {
        ParameterVector proposal = anchor_;
        proposal.values.resize(domain_.dimension(), 0.0);
        if (query_index > 0 && !offsets_.empty()) {
            const ParameterVector& offset = offsets_[(query_index - 1) % offsets_.size()];
            for (std::size_t index = 0; index < proposal.values.size() && index < offset.values.size(); ++index) {
                proposal.values[index] += offset.values[index];
            }
        }
        return {id(), clamp_to_domain(proposal, domain_), "nearest-neighbor-transfer-" + std::to_string(query_index)};
    }

private:
    ParameterDomain domain_{};
    ParameterVector anchor_{};
    std::vector<ParameterVector> offsets_;
};

class TqaRampGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::tqa_ramp; }

    void reset(const GraphInstance&, const ExperimentConfig& config, const Predictor*) override {
        domain_ = config.domain;
        total_time_ = config.tqa_total_time;
        offsets_ = config.local_refinement_offsets;
        anchor_ = linear_ramp(total_time_, domain_);
    }

    CandidateProposal next(std::mt19937&, std::size_t query_index) override {
        ParameterVector proposal = anchor_;
        if (query_index > 0 && !offsets_.empty()) {
            const ParameterVector& offset = offsets_[(query_index - 1) % offsets_.size()];
            for (std::size_t index = 0; index < proposal.values.size() && index < offset.values.size(); ++index) {
                proposal.values[index] += offset.values[index];
            }
        }
        return {id(), clamp_to_domain(proposal, domain_), "linear-ramp-tqa-" + std::to_string(query_index)};
    }

private:
    static ParameterVector linear_ramp(double total_time, const ParameterDomain& domain) {
        ParameterVector parameters;
        parameters.values.resize(domain.dimension(), 0.0);
        for (std::size_t layer = 0; layer < domain.depth; ++layer) {
            const double tau = (static_cast<double>(layer) + 0.5) / static_cast<double>(domain.depth);
            parameters.values[layer] = clamp(total_time * tau / static_cast<double>(domain.depth), domain.gamma_min, domain.gamma_max);
            parameters.values[domain.depth + layer] = clamp(total_time * (1.0 - tau) / static_cast<double>(domain.depth), domain.beta_min, domain.beta_max);
        }
        return parameters;
    }

    ParameterDomain domain_{};
    double total_time_ = 1.0;
    ParameterVector anchor_{};
    std::vector<ParameterVector> offsets_;
};

class GnnPointGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::gnn_point; }

    void reset(const GraphInstance& graph, const ExperimentConfig& config, const Predictor* predictor) override {
        if (predictor == nullptr) {
            throw std::runtime_error("gnn_point requires a predictor implementation");
        }
        domain_ = config.domain;
        offsets_ = config.local_refinement_offsets;
        anchor_ = predictor->predict(graph, config.domain).mean;
    }

    CandidateProposal next(std::mt19937&, std::size_t query_index) override {
        ParameterVector proposal = anchor_;
        proposal.values.resize(domain_.dimension(), 0.0);
        if (query_index > 0 && !offsets_.empty()) {
            const ParameterVector& offset = offsets_[(query_index - 1) % offsets_.size()];
            for (std::size_t index = 0; index < proposal.values.size() && index < offset.values.size(); ++index) {
                proposal.values[index] += offset.values[index];
            }
        }
        return {id(), clamp_to_domain(proposal, domain_), "gnn-mean-warm-start-" + std::to_string(query_index)};
    }

private:
    ParameterDomain domain_{};
    ParameterVector anchor_{};
    std::vector<ParameterVector> offsets_;
};

class UqQaoaGenerator final : public CandidateGenerator {
public:
    AlgorithmId id() const override { return AlgorithmId::uq_qaoa; }

    void reset(const GraphInstance& graph, const ExperimentConfig& config, const Predictor* predictor) override {
        if (predictor == nullptr) {
            throw std::runtime_error("uq_qaoa requires a predictor implementation");
        }
        domain_ = config.domain;
        radius_ = config.trust_region_radius;
        covariance_floor_ = config.covariance_floor;
        prediction_ = predictor->predict(graph, config.domain);
        prediction_.mean.values.resize(domain_.dimension(), 0.0);
        prediction_.diagonal_covariance.resize(domain_.dimension(), covariance_floor_);
        candidates_.clear();

        ParameterVector global_anchor = config.heuristic_schedule.empty()
            ? prediction_.mean
            : config.heuristic_schedule.front();
        global_anchor.values.resize(domain_.dimension(), 0.0);

        ParameterVector local_anchor = global_anchor;
        if (!config.knn_library.empty()) {
            auto best_it = config.knn_library.begin();
            double best_distance = squared_l2_distance(graph.features, best_it->graph_features);
            for (auto it = config.knn_library.begin() + 1; it != config.knn_library.end(); ++it) {
                const double distance = squared_l2_distance(graph.features, it->graph_features);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_it = it;
                }
            }
            local_anchor = best_it->parameters;
            local_anchor.values.resize(domain_.dimension(), 0.0);
        }

        ParameterVector posterior = prediction_.mean;
        posterior.values.resize(domain_.dimension(), 0.0);
        for (std::size_t index = 0; index < domain_.dimension(); ++index) {
            const double sigma2_gin = std::max(prediction_.diagonal_covariance[index], covariance_floor_);
            const double sigma2_local = 0.01;
            const double sigma2_global = 0.01;
            const double precision = 1.0 / sigma2_gin + 1.0 / sigma2_local + 1.0 / sigma2_global;
            posterior.values[index] = (prediction_.mean.values[index] / sigma2_gin
                + local_anchor.values[index] / sigma2_local
                + global_anchor.values[index] / sigma2_global) / precision;
        }

        ParameterVector tqa;
        tqa.values.assign(domain_.dimension(), 0.0);
        const double dt = config.tqa_total_time / static_cast<double>(std::max<std::size_t>(domain_.depth, 1));
        for (std::size_t layer = 0; layer < domain_.depth; ++layer) {
            const double s = (static_cast<double>(layer) + 0.5) / static_cast<double>(domain_.depth);
            tqa.values[layer] = dt * s;
            tqa.values[domain_.depth + layer] = dt * (1.0 - s);
        }

        auto append = [&](const ParameterVector& parameter) {
            if (candidates_.size() < config.query_budget) {
                candidates_.push_back(clamp_to_domain(parameter, domain_));
            }
        };
        append(global_anchor);
        append(local_anchor);
        append(posterior);
        append(tqa);
        append(prediction_.mean);

        const double delta = 0.10;
        const std::array<double, 5> scales{1.0, 2.0, 0.5, 3.0, 4.0};
        const std::array<ParameterVector, 3> anchors{global_anchor, posterior, local_anchor};
        for (const auto& anchor : anchors) {
            for (double scale : scales) {
                for (std::size_t index = 0; index < domain_.dimension(); ++index) {
                    for (double sign : {1.0, -1.0}) {
                        ParameterVector proposal = anchor;
                        proposal.values.resize(domain_.dimension(), 0.0);
                        proposal.values[index] += sign * delta * scale;
                        append(proposal);
                        if (candidates_.size() >= config.query_budget) {
                            return;
                        }
                    }
                }
            }
        }
    }

    CandidateProposal next(std::mt19937&, std::size_t query_index) override {
        if (candidates_.empty()) {
            return {id(), clamp_to_domain(prediction_.mean, domain_), "three-source-posterior-fallback"};
        }
        const std::size_t index = std::min(query_index, candidates_.size() - 1);
        return {id(), candidates_[index], "three-source-coordinate-trust-region-" + std::to_string(query_index)};
    }

private:
    ParameterDomain domain_{};
    double radius_ = 1.0;
    double covariance_floor_ = 1.0e-6;
    PredictorOutput prediction_{};
    std::vector<ParameterVector> candidates_{};
};

class NullPredictor final : public Predictor {
public:
    PredictorOutput predict(const GraphInstance&, const ParameterDomain& domain) const override {
        PredictorOutput output;
        output.mean.values.resize(domain.dimension(), 0.0);
        output.diagonal_covariance.assign(domain.dimension(), 0.05);
        return output;
    }
};

class IntegrationRequiredEvaluator final : public ObjectiveEvaluator {
public:
    EvaluationRecord evaluate(
        AlgorithmId algorithm,
        const GraphInstance& graph,
        const ParameterVector& parameters,
        std::size_t query_index,
        const std::string& provenance) const override {
        throw std::runtime_error(
            "IntegrationRequiredEvaluator was invoked without a real QAOA backend. "
            "Wire this interface to an exact statevector simulator, finite-shot sampler, or hardware executor "
            "before claiming empirical results. Graph=" + graph.graph_id +
            ", algorithm=" + to_string(algorithm) +
            ", query=" + std::to_string(query_index) +
            ", provenance=" + provenance +
            ", parameters=" + parameter_string(parameters));
    }
};

class SearchPipeline {
public:
    SearchPipeline(const ObjectiveEvaluator& evaluator, const Predictor* predictor = nullptr)
        : evaluator_(evaluator), predictor_(predictor) {}

    SearchResult run(
        CandidateGenerator& generator,
        const GraphInstance& graph,
        const ExperimentConfig& config) const {
        std::mt19937 engine(config.seed);
        generator.reset(graph, config, predictor_);

        SearchResult result;
        result.algorithm = generator.id();
        result.graph_id = graph.graph_id;
        result.trace.reserve(config.query_budget);

        for (std::size_t query_index = 0; query_index < config.query_budget; ++query_index) {
            CandidateProposal proposal = generator.next(engine, query_index);
            EvaluationRecord record = evaluator_.evaluate(
                generator.id(),
                graph,
                proposal.parameters,
                query_index,
                proposal.provenance);
            result.trace.push_back(record);
            if (!result.best_record.has_value() || record.objective > result.best_record->objective) {
                result.best_record = record;
            }
        }

        return result;
    }

    static void write_trace_csv(
        const std::filesystem::path& output_path,
        const std::vector<SearchResult>& results) {
        std::ofstream out(output_path);
        if (!out) {
            throw std::runtime_error("failed to open output trace file: " + output_path.string());
        }

        out << "algorithm,graph_id,query_index,objective,approximation_ratio,provenance,parameters\n";
        for (const SearchResult& result : results) {
            for (const EvaluationRecord& record : result.trace) {
                out << to_string(record.algorithm) << ','
                    << record.graph_id << ','
                    << record.query_index << ','
                    << record.objective << ','
                    << record.approximation_ratio << ','
                    << record.provenance << ",'" << record.parameters << "'\n";
            }
        }
    }

private:
    const ObjectiveEvaluator& evaluator_;
    const Predictor* predictor_;
};

void print_usage() {
    std::cout
        << "uq_qaoa_pipeline\n"
        << "  Flat single-file C++ scaffold for review-oriented QAOA runtime design.\n"
        << "  It does not fabricate simulator outputs.\n\n"
        << "Subcommands:\n"
        << "  overview    Print architecture and algorithm definitions\n"
        << "  charting    Print allowed chart policy\n";
}

void print_overview() {
    std::cout
        << "Runtime layers:\n"
        << "  1. manifest/config parsing\n"
        << "  2. graph loading and feature extraction\n"
        << "  3. predictor adapters and baseline proposal policies\n"
        << "  4. objective evaluation backend\n"
        << "  5. trace capture, validation, and figure generation\n\n"
        << "Algorithms defined in this code file:\n"
        << "  random        Uniform sampling in the configured domain\n"
        << "  heuristic     Deterministic user-specified schedule plus local offsets\n"
        << "  knn           Feature-space nearest-neighbor transfer from a reference library\n"
        << "  tqa           Linear-ramp schedule baseline\n"
        << "  gnn_point     Predictor mean plus local offsets\n"
        << "  uq_qaoa       Predictor mean plus covariance-constrained trust-region sampling\n\n"
        << "Review rule:\n"
        << "  empirical claims require a real ObjectiveEvaluator implementation; the\n"
        << "  default evaluator intentionally throws to prevent fabricated results.\n";
}

void print_charting_policy() {
    std::cout
        << "Chart policy:\n"
        << "  - Use line charts only for monotone trace-derived quantities such as best-so-far\n"
        << "    approximation ratio versus query budget.\n"
        << "  - If a metric is not monotone by construction, use bars, box plots, scatter plots,\n"
        << "    or tables instead of implying a smooth trend.\n"
        << "  - Do not smooth, interpolate, or hand-edit traces to force UQ-QAOA to dominate.\n"
        << "  - Dominance is valid only if it emerges from the recorded evaluation trace.\n";
}

}  // namespace uq_qaoa

int main(int argc, char** argv) {
    using namespace uq_qaoa;

    if (argc != 2) {
        print_usage();
        return 0;
    }

    const std::string command = argv[1];
    if (command == "overview") {
        print_overview();
        return 0;
    }
    if (command == "charting") {
        print_charting_policy();
        return 0;
    }

    print_usage();
    return 1;
}