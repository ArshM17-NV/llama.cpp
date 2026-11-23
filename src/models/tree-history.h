#pragma once

#include "ggml.h"

inline ggml_tensor * llama_tree_history_candidates(ggml_context * ctx, ggml_tensor * logits) {
    const int64_t vocab = logits->ne[0], rows = logits->ne[1];
    auto * ids = ggml_top_k(ctx, logits, 9);
    auto gather = [&](ggml_tensor * values) {
        return ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, values, 1, vocab, rows), ids), 9, rows);
    };
    auto * selected = gather(logits);
    auto * probabilities = gather(ggml_soft_max(ctx, logits));
    auto * info = ggml_concat(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), selected, 0);
    return ggml_concat(ctx, info, probabilities, 0);
}
