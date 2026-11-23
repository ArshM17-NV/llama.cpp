#include "treeflash-head.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>

namespace {
constexpr int D = common_treeflash_head::depth;
constexpr int K = common_treeflash_head::top_k;
constexpr int S = common_treeflash_head::slots;
constexpr int ROWS = D*S;
// Current/previous candidate log probabilities, outside mass, depth, parent slot,
// parent log probability, and parent rank in the current shortlist.
constexpr int N_FEATURES = K + K + 1 + D + S + 1 + K + 1;

int rank_of(const llama_token * row, llama_token token) {
    return row ? int(std::find(row, row + K, token) - row) : K;
}

struct candidate_path {
    double score;
    int parent;
    int rank;

    bool operator<(const candidate_path & other) const {
        if (score != other.score) {
            return score < other.score;
        }
        return parent != other.parent ? parent > other.parent : rank > other.rank;
    }
};

ggml_tensor * checked(ggml_context * ctx, const char * name, ggml_type type, std::initializer_list<int64_t> shape) {
    auto * tensor = ggml_get_tensor(ctx, name);
    if (tensor && tensor->type == type) {
        bool valid = true;
        int dim = 0;
        for (int64_t extent : shape) {
            valid &= extent < 0 || extent == tensor->ne[dim];
            ++dim;
        }
        while (dim < GGML_MAX_DIMS) {
            valid &= tensor->ne[dim++] == 1;
        }
        if (valid) {
            return tensor;
        }
    }
    throw std::runtime_error(std::string("invalid TreeFlash tensor: ") + name);
}

std::vector<float> floats(ggml_context * ctx, const char * name, std::initializer_list<int64_t> shape) {
    auto * t = checked(ctx, name, GGML_TYPE_F32, shape);
    const auto * v = static_cast<const float *>(t->data);
    return {v, v + ggml_nelements(t)};
}
}

void common_treeflash_history::reset(const std::vector<llama_token> & prompt) {
    rows.clear();
    clock = 0;
    commit(prompt);
}

void common_treeflash_history::commit(const std::vector<llama_token> & tokens) {
    for (size_t i = 1; i < tokens.size(); ++i) {
        auto & r = rows[tokens[i - 1]];
        size_t slot = 0;
        for (size_t j = 0; j < r.ids.size(); ++j) {
            if (r.count[j] && r.ids[j] == tokens[i]) {
                slot = j;
                break;
            }
            if (r.last[j] < r.last[slot]) {
                slot = j;
            }
        }
        if (!r.count[slot] || r.ids[slot] != tokens[i]) {
            r.ids[slot] = tokens[i];
            r.count[slot] = 0;
        }
        ++r.count[slot];
        r.last[slot] = ++clock;
    }
}

void common_treeflash_history::observe(llama_token parent, const llama_token * ids8, const float * probabilities8) {
    auto & r = rows[parent];
    std::copy_n(ids8, 8, r.verifier_ids.begin());
    std::copy_n(probabilities8, 8, r.verifier_p.begin());
}

void common_treeflash_history::observe_logits(llama_token parent, const float * logits, int32_t n_vocab, float top_p) {
    if (n_vocab < 8) {
        throw std::invalid_argument("TreeFlash requires at least eight verifier logits");
    }
    std::array<llama_token, 8> ids;
    std::array<float, 8> values;
    ids.fill(-1);
    values.fill(-std::numeric_limits<float>::infinity());
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (logits[i] <= values.back()) {
            continue;
        }
        int j = 7;
        for (; j > 0 && logits[i] > values[j-1]; --j) {
            values[j] = values[j-1];
            ids[j] = ids[j-1];
        }
        values[j] = logits[i];
        ids[j] = i;
    }
    std::array<float, 8> weights{};
    std::array<double, 8> prefix{};
    double top_mass = 0;
    for (int j = 0; j < 8; ++j) {
        weights[j] = std::exp(values[j] - values[0]);
        top_mass += weights[j];
        prefix[j] = top_mass;
    }
    if (top_p < 1) {
        auto keep = [&](double total) {
            return std::min(8, 1 + int(std::lower_bound(prefix.begin(), prefix.end(), top_p*total) - prefix.begin()));
        };
        // The missing vocabulary mass lies between zero and (V-8) times the cutoff weight.
        // Sum the whole row only if that interval could change which top-eight token survives.
        int count = keep(top_mass);
        if (count != keep(top_mass + (n_vocab - 8)*double(weights.back()))) {
            double total = 0;
            for (int32_t i = 0; i < n_vocab; ++i) {
                total += std::exp(logits[i] - values[0]);
            }
            count = keep(total);
        }
        std::fill(weights.begin() + count, weights.end(), 0.0f);
    }
    // features() normalizes these values over the matching shortlist. The row's common
    // softmax divisor cancels, so retaining relative weights is sufficient.
    observe(parent, ids.data(), weights.data());
}

void common_treeflash_history::features(llama_token parent, const llama_token * ids, float * out) const {
    std::fill_n(out, K, 0.0f);
    auto found = rows.find(parent);
    if (found == rows.end()) {
        return;
    }
    const auto & r = found->second;
    std::array<double, K> counts{}, probabilities{};
    double nc = 0, np = 0;
    for (int c = 0; c < K; ++c) {
        for (size_t i = 0; i < r.ids.size(); ++i) {
            if (ids[c] == r.ids[i]) {
                counts[c] += r.count[i];
            }
            if (ids[c] == r.verifier_ids[i]) {
                probabilities[c] += r.verifier_p[i];
            }
        }
        nc += counts[c];
        np += probabilities[c];
    }
    const int sources = int(nc > 0) + int(np > 0);
    if (!sources) {
        return;
    }
    for (int c = 0; c < K; ++c) {
        const double prob = ((nc ? counts[c]/nc : 0) + (np ? probabilities[c]/np : 0)) / sources;
        const float f = float(std::log1p(1000*prob));
        out[c] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(ggml_fp16_to_fp32(ggml_fp32_to_fp16(f))));
    }
}

struct common_treeflash_head::impl {
    ggml_backend_ptr backend;
    ggml_context_ptr weights_ctx, graph_ctx;
    ggml_backend_buffer_ptr weights_buf;
    ggml_gallocr_ptr allocator;
    ggml_cgraph * graph;
    ggml_tensor * hidden, * candidates, * parents, * feature_tensor, * historical, * result;
    std::vector<float> token_bias, rank_bias, offsets, scale, outside, same, alpha;
    int hidden_width, vocab;

    impl(const std::string & path, ggml_backend_dev_t device) :
        backend(device ? ggml_backend_dev_init(device, nullptr) : ggml_backend_init_best()) {
        if (!backend) {
            throw std::runtime_error("could not initialize the TreeFlash compute backend");
        }
        ggml_context * source = nullptr;
        gguf_context_ptr file(gguf_init_from_file(path.c_str(), {false, &source}));
        ggml_context_ptr host(source);
        if (!file || !host) {
            throw std::runtime_error("could not open TreeFlash GGUF: " + path);
        }
        const int64_t arch = gguf_find_key(file.get(), "general.architecture");
        if (arch < 0 || gguf_get_kv_type(file.get(), arch) != GGUF_TYPE_STRING ||
                std::strcmp(gguf_get_val_str(file.get(), arch), "treeflash") != 0) {
            throw std::runtime_error("expected a treeflash GGUF sidecar");
        }
        auto * value = checked(source, "embedding.value", GGML_TYPE_F16, {-1, -1});
        const int64_t rank = value->ne[0];
        vocab = int(value->ne[1]);
        auto * parent = checked(source, "embedding.parent", GGML_TYPE_F16, {rank, vocab});
        auto * wh = checked(source, "Wh.weight", GGML_TYPE_F32, {-1, rank});
        hidden_width = int(wh->ne[0]);
        auto * wm = checked(source, "Wm.weight", GGML_TYPE_F32, {rank, -1});
        const int64_t mh = wm->ne[1];
        auto * mlp0 = checked(source, "mlp.0.weight", GGML_TYPE_F32, {N_FEATURES + mh, -1});
        const int64_t mlp = mlp0->ne[1];
        token_bias = floats(source, "embedding.bias", {vocab});
        rank_bias = floats(source, "q.weight", {2*(K+1), 1});
        offsets = floats(source, "T", {K, S, D});
        scale = floats(source, "log_a", {D});
        outside = floats(source, "bn", {D});
        same = floats(source, "eq", {D});
        alpha = floats(source, "alpha", {D});

        weights_ctx.reset(ggml_init({64*ggml_tensor_overhead(), nullptr, true}));
        if (!weights_ctx) {
            throw std::bad_alloc();
        }
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
        auto weight = [&](ggml_tensor * t) {
            auto * dest = ggml_dup_tensor(weights_ctx.get(), t);
            ggml_set_name(dest, t->name);
            copies.emplace_back(dest, t);
            return dest;
        };
        value = weight(value);
        parent = weight(parent);
        wh = weight(wh);
        wm = weight(wm);
        mlp0 = weight(mlp0);
        auto * mp = weight(checked(source, "M.weight", GGML_TYPE_F32, {rank, rank}));
        auto * g = weight(checked(source, "g", GGML_TYPE_F32, {rank, D}));
        auto * gh = weight(checked(source, "gh", GGML_TYPE_F32, {rank, D}));
        auto * mk = weight(checked(source, "mk_in.weight", GGML_TYPE_F32, {K, mlp}));
        auto * b0 = weight(checked(source, "mlp.0.bias", GGML_TYPE_F32, {mlp}));
        auto * mlp2 = weight(checked(source, "mlp.2.weight", GGML_TYPE_F32, {mlp, K+2}));
        auto * b2 = weight(checked(source, "mlp.2.bias", GGML_TYPE_F32, {K+2}));
        weights_buf.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), backend.get()));
        if (!weights_buf) {
            throw std::runtime_error("insufficient backend memory for TreeFlash weights");
        }
        for (const auto & copy : copies) {
            ggml_backend_tensor_set(copy.first, copy.second->data, 0, ggml_nbytes(copy.second));
        }
        ggml_backend_buffer_set_usage(weights_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        graph_ctx.reset(ggml_init({128*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true}));
        if (!graph_ctx) {
            throw std::bad_alloc();
        }
        auto * ctx = graph_ctx.get();
        graph = ggml_new_graph_custom(ctx, 128, false);
        hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_width, D);
        candidates = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K*D);
        parents = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ROWS);
        feature_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_FEATURES, S, D);
        historical = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, ROWS);
        for (auto * input : {hidden, candidates, parents, feature_tensor, historical}) {
            ggml_set_input(input);
        }
        auto * h = ggml_norm(ctx, ggml_mul_mat(ctx, wh, ggml_rms_norm(ctx, hidden, 1e-5f)), 1e-5f);
        auto * hsmall = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wm, h), mh, 1, D);
        auto * features = ggml_concat(ctx, feature_tensor, ggml_repeat_4d(ctx, hsmall, mh, S, D, 1), 0);
        auto * first = ggml_mul_mat(ctx, mlp0, ggml_reshape_2d(ctx, features, N_FEATURES + mh, ROWS));
        first = ggml_silu(ctx, ggml_add(ctx, ggml_add(ctx, first, b0), ggml_mul_mat(ctx, mk, historical)));
        auto * logits = ggml_reshape_3d(ctx, ggml_add(ctx, ggml_mul_mat(ctx, mlp2, first), b2), K+2, S, D);

        auto * pv = ggml_reshape_3d(ctx, ggml_get_rows(ctx, parent, parents), rank, S, D);
        auto * cv = ggml_reshape_3d(ctx, ggml_get_rows(ctx, value, candidates), rank, K, D);
        auto * interaction = ggml_reshape_3d(ctx, ggml_add(ctx, ggml_mul_mat(ctx, mp, h), g), rank, 1, D);
        auto * hquery = ggml_reshape_3d(ctx, ggml_mul(ctx, h, gh), rank, 1, D);
        auto * query = ggml_add(ctx, ggml_mul(ctx, pv, interaction), hquery);
        auto * pairs = ggml_scale(ctx, ggml_mul_mat(ctx, cv, query), 1.0f/std::sqrt(float(rank)));
        result = ggml_concat(ctx, pairs, logits, 0);
        ggml_set_output(result);
        ggml_build_forward_expand(graph, result);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("could not allocate TreeFlash compute graph");
        }
    }
};

common_treeflash_head::common_treeflash_head(const std::string & path, ggml_backend_dev_t device) :
    p(new impl(path, device)) {}
common_treeflash_head::~common_treeflash_head() = default;
int32_t common_treeflash_head::vocab_size() const { return p->vocab; }
int32_t common_treeflash_head::hidden_size() const { return p->hidden_width; }

std::vector<common_treeflash_node> common_treeflash_head::propose(llama_token root, const llama_token * ids,
        const float * logp, const float * hidden, const common_treeflash_history & history, const float * gumbels) {
    if (root < 0 || root >= p->vocab ||
            std::any_of(ids, ids + D*K, [this](llama_token id) { return id < 0 || id >= p->vocab; })) {
        throw std::invalid_argument("TreeFlash candidate is outside the target embedding vocabulary");
    }
    std::array<float, N_FEATURES*ROWS> features{};
    std::array<float, K*ROWS> historical;
    std::array<llama_token, ROWS> parents;
    std::array<double, D> outside;
    std::array<float, D*K> unary;
    for (int d = 0; d < D; ++d) {
        const llama_token * row = ids + d*K;
        const llama_token * prev = d == 0 ? nullptr : row - K;
        const llama_token * next = d == D-1 ? nullptr : row + K;
        double mass = 0;
        for (int c = 0; c < K; ++c) {
            mass += std::exp(double(logp[d*K+c]));
            unary[d*K+c] = p->token_bias[row[c]] + p->rank_bias[rank_of(next, row[c])] +
                    p->rank_bias[K+1 + rank_of(prev, row[c])];
        }
        outside[d] = std::log(std::max(1e-6, 1.0-mass));
        for (int s = 0; s < S; ++s) {
            const int i = d*S+s;
            const auto parent = prev && s < K ? prev[s] : root;
            parents[i] = parent;
            history.features(parent, row, historical.data()+i*K);
            float * f = features.data() + i*N_FEATURES;
            std::copy_n(logp + d*K, K, f);
            if (d) {
                std::copy_n(logp + (d-1)*K, K, f+K);
            }
            f[2*K] = float(outside[d]);
            f[2*K+1+d] = 1;
            f[2*K+1+D+s] = 1;
            f[2*K+1+D+S] = prev && s < K ? logp[(d-1)*K+s] : 0;
            f[2*K+2+D+S+rank_of(row, parent)] = 1;
        }
    }
    auto upload = [&](ggml_tensor * tensor, const void * data) {
        ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
    };
    upload(p->hidden, hidden);
    upload(p->candidates, ids);
    upload(p->parents, parents.data());
    upload(p->historical, historical.data());
    upload(p->feature_tensor, features.data());
    if (ggml_backend_graph_compute(p->backend.get(), p->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("TreeFlash head graph execution failed");
    }
    std::array<float, (2*K+2)*ROWS> network;
    ggml_backend_tensor_get(p->result, network.data(), 0, sizeof(network));
    std::array<double, K*ROWS> scores;
    for (int d = 0; d < D; ++d) {
        int keep = K;
        if (gumbels) {
            double total = 0, sum = 0;
            for (int c = 0; c < K; ++c) {
                total += std::exp(double(logp[d*K+c]));
            }
            for (int c = 0; c < K; ++c) {
                sum += std::exp(double(logp[d*K+c]));
                if (sum >= 0.95*total) {
                    keep = c+1;
                    break;
                }
            }
        }
        for (int s = 0; s < S; ++s) {
            const int i = d*S+s;
            const float * pair = network.data() + i*(2*K+2);
            const float * mlp = pair + K;
            const double temp = std::exp(double(p->scale[d]) + mlp[K+1]);
            const double none = outside[d] + p->outside[d] + mlp[K];
            double * z = scores.data() + i*K;
            for (int c = 0; c < K; ++c) {
                z[c] = temp*logp[d*K+c] + mlp[c] + p->alpha[d]*historical[i*K+c] +
                        p->offsets[i*K+c] + unary[d*K+c] + pair[c] +
                        (parents[i] == ids[d*K+c] ? p->same[d] : 0);
            }
            const double mx = std::max(none, *std::max_element(z, z+K));
            double total = std::exp(none - mx);
            for (int c = 0; c < K; ++c) {
                total += std::exp(z[c] - mx);
            }
            const double lse = mx + std::log(total);
            for (int c = 0; c < K; ++c) {
                z[c] -= lse;
            }
            if (gumbels) {
                double noise_max = -std::numeric_limits<double>::infinity(), kept_mass = 0, perturbed_mass = 0;
                for (int c = 0; c < keep; ++c) {
                    kept_mass += std::exp(z[c]);
                    noise_max = std::max(noise_max, z[c] + gumbels[d*K+c]);
                }
                for (int c = 0; c < keep; ++c) {
                    perturbed_mass += std::exp(z[c] + gumbels[d*K+c] - noise_max);
                }
                for (int c = 0; c < keep; ++c) {
                    const double shifted = std::exp(z[c] + gumbels[d*K+c] - noise_max) / perturbed_mass;
                    z[c] = std::log(0.5*(std::exp(z[c]) + kept_mass*shifted));
                }
            }
        }
    }

    std::vector<common_treeflash_node> tree{{root, -1, 0}};
    tree.reserve(nodes);
    std::priority_queue<candidate_path> frontier;
    auto offer = [&](int parent, int slot, double prefix_score) {
        const int d = tree[parent].depth;
        if (d < D) {
            for (int c = 0; c < K; ++c) {
                frontier.push({prefix_score + scores[(d*S+slot)*K+c], parent, c});
            }
        }
    };
    offer(0, K, 0);
    while (int(tree.size()) < nodes) {
        const auto best = frontier.top();
        frontier.pop();
        const int d = tree[best.parent].depth;
        tree.push_back({ids[d*K+best.rank], best.parent, d+1});
        offer(int(tree.size())-1, best.rank, best.score);
    }
    return tree;
}
