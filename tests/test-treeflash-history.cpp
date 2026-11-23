#include "treeflash-head.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// Compare history features to a full-vocabulary reference at each top-p boundary.
static void check(const std::vector<float> & logits) {
    const int n = int(logits.size());
    std::vector<llama_token> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto i, auto j) { return logits[i] > logits[j]; });
    const float peak = logits[order[0]];
    double total = 0;
    for (float value : logits) {
        total += std::exp(value - peak);
    }
    std::array<float, 8> weights;
    std::array<double, 8> prefix;
    std::vector<float> thresholds = { 0.0f, 0.5f, 0.95f, 1.0f };
    for (int i = 0; i < 8; ++i) {
        weights[i] = std::exp(logits[order[i]] - peak);
        prefix[i] = weights[i] + (i ? prefix[i - 1] : 0.0);
        const float p = float(prefix[i] / total);
        thresholds.push_back(std::nextafter(p, 0.0f));
        thresholds.push_back(p);
        thresholds.push_back(std::nextafter(p, 1.0f));
    }
    for (float p : thresholds) {
        auto expected_weights = weights;
        for (int i = 1; i < 8; ++i) {
            if (prefix[i - 1] >= p * total) {
                expected_weights[i] = 0;
            }
        }
        common_treeflash_history actual, expected;
        actual.observe_logits(0, logits.data(), n, p);
        expected.observe(0, order.data(), expected_weights.data());
        std::array<float, 20> a, b;
        actual.features(0, order.data(), a.data());
        expected.features(0, order.data(), b.data());
        GGML_ASSERT(a == b);
    }
}

int main() {
    std::vector<float> logits(8192);
    for (int gap : { 3, 12, 48 }) {
        for (size_t i = 0; i < logits.size(); ++i) {
            // Include ties, fully masked logits and exact bin edges.
            logits[i] = i % 11 ? -gap - float((37*i) % 63) / 16 : -std::numeric_limits<float>::infinity();
        }
        for (int i = 0; i < 8; ++i) {
            logits[83*i + 17] = -0.25f * i;
        }
        check(logits);
    }
    return 0;
}
