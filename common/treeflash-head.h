#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct common_treeflash_node {
    llama_token token;
    int32_t parent;
    int32_t depth;
};

// Request-local features used by the proposal network.
class common_treeflash_history {
public:
    // Include the pending root after the prompt. commit() takes root, matches, then the bonus.
    void reset(const std::vector<llama_token> & prompt);
    void commit(const std::vector<llama_token> & tokens);
    void observe_logits(llama_token parent, const float * logits, int32_t n_vocab, float top_p);
    void observe(llama_token parent, const llama_token * ids8, const float * probabilities8);
    void features(llama_token parent, const llama_token * ids20, float * out20) const;

private:
    struct row {
        std::array<llama_token, 8> ids{};
        std::array<int64_t, 8> last{};
        std::array<uint32_t, 8> count{};
        std::array<llama_token, 8> verifier_ids{};
        std::array<float, 8> verifier_p{};
    };
    std::unordered_map<llama_token, row> rows;
    int64_t clock = 0;
};

class common_treeflash_head {
public:
    static constexpr int depth = 7, top_k = 20, slots = 21, nodes = 8;
    // Produced by convert_treeflash_to_gguf.py using the same target GGUF as inference.
    explicit common_treeflash_head(const std::string & path, ggml_backend_dev_t device = nullptr);
    ~common_treeflash_head();
    common_treeflash_head(const common_treeflash_head &) = delete;
    common_treeflash_head & operator=(const common_treeflash_head &) = delete;

    int32_t hidden_size() const;
    int32_t vocab_size() const;

    // ids and logp are [7,20] in descending draft-logit order; logp uses the full vocabulary.
    // hidden is [7,hidden_size()], the final normalized input to the draft LM head. Optional Gumbels are [7,20].
    std::vector<common_treeflash_node> propose(llama_token root, const llama_token * ids, const float * logp,
            const float * hidden, const common_treeflash_history & history, const float * gumbels = nullptr);

private:
    struct impl;
    std::unique_ptr<impl> p;
};
