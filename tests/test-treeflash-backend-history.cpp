#include "treeflash-head.h"
#include "../src/models/tree-history.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int rows = 8, vocab = 131072;
    ggml_backend_load_all();
    ggml_backend_ptr backend(argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) :
                                        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(backend);
    std::vector<float> logits(rows * vocab);
    for (int r = 0; r < rows; ++r) {
        auto * row = logits.data() + r * vocab;
        for (int t = 0; t < vocab; ++t) {
            row[t] = -20.0f - ((t * 113 + r * 8191) % vocab) * 0.0001f;
        }
        for (int t = 0; t < 12; ++t) {
            row[3 + t * 7001] = -0.25f * t - r;
        }
        if (r == 1) { row[3] = 140.0f; } // underflow
        if (r == 2) { row[3 + 8 * 7001] = row[3 + 7 * 7001]; } // boundary tie
        if (r == 3) { row[3 + 5 * 7001] = row[3 + 4 * 7001]; } // interior tie
        if (r == 4) { std::fill_n(row, vocab, -1.0f); }
        if (r == 5) { std::fill(row + 90000, row + vocab, -std::numeric_limits<float>::infinity()); }
        if (r == 6) {
            for (int t = 0; t < vocab; ++t) { row[t] = ((t * 113) % vocab) * 0.00001f; } // large tail mass
        }
    }

    ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, vocab, rows);
    ggml_set_input(input);
    auto * output = llama_tree_history_candidates(ctx.get(), input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
    ggml_backend_tensor_set(input, logits.data(), 0, ggml_nbytes(input));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    std::array<llama_tree_history_row, rows> packed;
    GGML_ASSERT(ggml_nbytes(output) == sizeof(packed));
    ggml_backend_tensor_get(output, packed.data(), 0, sizeof(packed));

    int direct = 0, fallback = 0;
    for (int r = 0; r < rows; ++r) {
        const float * scores = logits.data() + r * vocab;
        std::vector<int> ids(vocab);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + 20, ids.end(), [&](int a, int b) {
            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
        });
        const float peak = scores[ids[0]];
        double total = 0;
        for (int i = 0; i < vocab; ++i) { total += std::exp(scores[i] - peak); }
        auto row = packed[r];
        for (int j = 0; j < 9; ++j) {
            const int id = int(row.ids[j]);
            GGML_ASSERT(id >= 0 && id < vocab && scores[id] == row.logits[j] && scores[id] >= scores[ids[8]]);
            const double expected = std::exp(scores[id] - peak) / total;
            GGML_ASSERT(std::abs(row.probabilities[j] - expected) < 2e-7 + 2e-5 * expected);
            GGML_ASSERT(std::count(row.ids, row.ids + 9, row.ids[j]) == 1);
        }
        std::vector<float> ps{0.0f, 0.5f, 0.95f, 1.0f};
        double mass = 0;
        for (int j = 0; j < 7; ++j) {
            mass += std::exp(scores[ids[j]] - peak);
            const float threshold = float(mass / total);
            ps.push_back(threshold);
            ps.push_back(std::nextafter(threshold, 0.0f));
            ps.push_back(std::nextafter(threshold, 1.0f));
        }
        for (float p : ps) {
            common_treeflash_history actual, reference;
            reference.observe_logits(17, scores, vocab, p);
            if (actual.observe_backend(17, row, vocab, p)) {
                GGML_ASSERT(r != 2 && r != 4);
                ++direct;
            } else {
                actual.observe_logits(17, scores, vocab, p);
                ++fallback;
            }
            std::array<float, 20> a, b;
            actual.features(17, ids.data(), a.data());
            reference.features(17, ids.data(), b.data());
            GGML_ASSERT(a == b);
        }
    }
    GGML_ASSERT(direct && fallback);
    std::printf("full-vocabulary graph and Markov features passed (%d compact rows, %d fallbacks)\n", direct, fallback);
}
