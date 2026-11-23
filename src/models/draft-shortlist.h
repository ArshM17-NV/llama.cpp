#pragma once

#include "ggml.h"

namespace llama_draft {

inline ggml_tensor * ranked_candidates(ggml_context * ctx, ggml_tensor * logits, int64_t top_k) {
    const int64_t rows = logits->ne[1];
    auto * ids = ggml_top_k(ctx, logits, top_k);
    auto * values = ggml_reshape_3d(ctx, logits, 1, logits->ne[0], rows);
    auto * selected = ggml_reshape_2d(ctx, ggml_get_rows(ctx, values, ids), top_k, rows);
    auto * order = ggml_argsort(ctx, selected, GGML_SORT_ORDER_DESC);
    return ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, ids, 1, top_k, rows), order), top_k, rows);
}

inline ggml_tensor * treeflash_draft_rows(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * hidden, int32_t k) {
    const int64_t d = hidden->ne[1] - 1;
    const int64_t width = hidden->ne[0];
    const int64_t vocab = logits->ne[0];
    logits = ggml_view_2d(ctx, logits, vocab, d, logits->nb[1], logits->nb[1]);
    auto * ids = ranked_candidates(ctx, logits, k);
    auto * selected = ggml_reshape_2d(ctx,
            ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits, 1, vocab, d), ids), k, d);
    auto * max_id = ggml_view_2d(ctx, ids, 1, d, ids->nb[1], 0);
    auto * probs = ggml_reshape_3d(ctx, ggml_soft_max(ctx, logits), 1, vocab, d);
    auto * max_p = ggml_reshape_2d(ctx, ggml_get_rows(ctx, probs, max_id), 1, d);
    auto * max_logit = ggml_view_2d(ctx, selected, 1, d, selected->nb[1], 0);
    // log p(i) = logit(i) - max(logit) + log p(max). Works even when p(i) underflows.
    auto * lp = ggml_add(ctx, ggml_sub(ctx, selected, max_logit), ggml_log(ctx, max_p));
    auto * info = ggml_concat(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), lp, 0);
    auto * anchor = ggml_pad(ctx, ggml_reshape_2d(ctx, info, 2 * k * d, 1), width - 2 * k * d, 0, 0, 0);
    auto * states = ggml_view_2d(ctx, hidden, width, d, hidden->nb[1], hidden->nb[1]);
    return ggml_concat(ctx, anchor, states, 1);
}

} // namespace llama_draft
