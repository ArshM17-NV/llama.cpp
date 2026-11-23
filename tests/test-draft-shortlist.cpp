#include "../src/models/draft-shortlist.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_ptr backend(argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) :
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!backend) {
        std::fprintf(stderr, "Could not initialize %s backend\n", argc > 1 ? argv[1] : "CPU");
        return 1;
    }

    constexpr int vocab = 257, width = 512, rows = 8, k = 20;
    std::vector<float> scores(vocab * rows);
    std::vector<float> states(width * rows);
    for (int row = 0; row < rows; ++row) {
        for (int id = 0; id < vocab; ++id) {
            // A different permutation for each row, with unique scores and substantial tail mass.
            scores[row * vocab + id] = ((53 * id + 37 * row) % vocab) * 0.01f;
        }
    }
    // Several selected probabilities underflow in float while their true log probabilities are finite.
    scores[2 * vocab + 7] = 140.0f;
    std::iota(states.begin(), states.end(), -512.0f);

    ggml_context_ptr ctx(ggml_init({ 1024 * 1024, nullptr, true }));
    auto * logits = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, vocab, rows);
    auto * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_set_input(logits);
    ggml_set_input(hidden);
    auto * output = llama_draft::treeflash_draft_rows(ctx.get(), logits, hidden, k);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())));
    GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
    ggml_backend_tensor_set(logits, scores.data(), 0, ggml_nbytes(logits));
    ggml_backend_tensor_set(hidden, states.data(), 0, ggml_nbytes(hidden));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result(width * rows);
    GGML_ASSERT(ggml_nbytes(output) == result.size() * sizeof(float));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    GGML_ASSERT(std::equal(states.begin() + width, states.end(), result.begin() + width));

    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    for (int row = 1; row < rows; ++row) {
        const float * x = scores.data() + row * vocab;
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [x](int a, int b) { return x[a] > x[b]; });
        const double peak = x[order.front()];
        double sum = 0;
        for (int id = 0; id < vocab; ++id) {
            sum += std::exp(x[id] - peak);
        }
        const float * packed = result.data() + (row - 1) * 2 * k;
        for (int rank = 0; rank < k; ++rank) {
            const double log_p = x[order[rank]] - peak - std::log(sum);
            GGML_ASSERT(packed[rank] == float(order[rank]));
            GGML_ASSERT(std::abs(packed[k + rank] - log_p) < 1e-4);
        }
    }
    return 0;
}
