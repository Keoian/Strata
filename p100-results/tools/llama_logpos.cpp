// llama_logpos.cpp - llama.cpp's side of the engine-quality baseline: the same token ids Strata's
// tools/quant_quality.py reads (data/quant_quality/v1/set.ids.json), teacher-forced through llama.cpp in one batch per
// segment, written in Strata's STRATA_LOGPOS row format so `quant_quality.py compare` scores both engines alike:
//   pos  target  logp(target)  argmax  logp(argmax)  argmax==target  logp(extra)  logp(target without extra)  K x id:logp
// Row q is the distribution after token q; its target is token q+1 (src/program/generate.cpp, STRATA_LOGPOS).
// Also writes OUT.seg.json ({"segments": [{"name", "first_line", "end_line"}]}), as quant_quality.py run does.
//
// Build against a llama.cpp checkout (here: Strata's pinned 3cf03257, CPU build):
//   g++-14 -O2 -std=c++17 llama_logpos.cpp -I$L/include -I$L/ggml/include -L$L/build-cpu/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$L/build-cpu/bin -o llama_logpos
// Usage: llama_logpos --model SHARD1.gguf --ids set.ids.json --out RUN.tsv [--threads 36] [--topk 64] [--extra 248046]
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct Segment {
    std::string name;
    std::vector<llama_token> ids;
};

// A minimal reader for set.ids.json: {"segments": [{"name": "...", "ids": [1, 2, ...]}, ...], ...}. Only the
// "segments" array's objects are read; every object needs "name" before "ids".
static std::vector<Segment> read_ids(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    std::vector<Segment> out;
    size_t at = s.find("\"segments\"");
    if (at == std::string::npos) return out;
    for (;;) {
        const size_t n = s.find("\"name\"", at);
        if (n == std::string::npos) break;
        const size_t q1 = s.find('"', s.find(':', n) + 1);
        const size_t q2 = s.find('"', q1 + 1);
        Segment seg;
        seg.name = s.substr(q1 + 1, q2 - q1 - 1);
        const size_t i = s.find("\"ids\"", q2);
        if (i == std::string::npos) break;
        const size_t b1 = s.find('[', i), b2 = s.find(']', b1);
        std::stringstream nums(s.substr(b1 + 1, b2 - b1 - 1));
        std::string tok;
        while (std::getline(nums, tok, ',')) seg.ids.push_back((llama_token) std::atol(tok.c_str()));
        out.push_back(seg);
        at = b2;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model, ids_path, out_path;
    int threads = 36, topk = 64, extra = 248046;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--ids") ids_path = v;
        else if (a == "--out") out_path = v;
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--topk") topk = std::atoi(v.c_str());
        else if (a == "--extra") extra = std::atoi(v.c_str());
    }
    if (model.empty() || ids_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: llama_logpos --model GGUF --ids set.ids.json --out RUN.tsv [--threads N] [--topk K]\n");
        return 2;
    }
    const std::vector<Segment> segs = read_ids(ids_path);
    if (segs.empty()) { std::fprintf(stderr, "no segments in %s\n", ids_path.c_str()); return 1; }
    size_t longest = 0;
    for (const auto& s : segs) longest = std::max(longest, s.ids.size());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(m);
    const int V = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (longest + 64);
    cp.n_batch = cp.n_ubatch = (uint32_t) longest;
    cp.n_threads = cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(m, cp);
    if (!ctx) { std::fprintf(stderr, "cannot make a context\n"); return 1; }

    std::FILE* out = std::fopen(out_path.c_str(), "w");
    std::string seg_json = "{\"segments\": [";
    long long line = 0;
    std::vector<int32_t> order((size_t) V);
    for (size_t si = 0; si < segs.size(); ++si) {
        const auto& s = segs[si];
        const int n = (int) s.ids.size();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int t = 0; t < n; ++t) {
            b.token[t] = s.ids[(size_t) t];
            b.pos[t] = t;
            b.n_seq_id[t] = 1;
            b.seq_id[t][0] = 0;
            b.logits[t] = 1;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "%s: decode failed\n", s.name.c_str()); return 1; }
        const long long first = line;
        for (int q = 0; q + 1 < n; ++q) {
            const float* row = llama_get_logits_ith(ctx, q);
            const int tgt = s.ids[(size_t) q + 1];
            int top = 0;
            for (int v = 1; v < V; ++v) if (row[v] > row[top]) top = v;
            const double maxv = row[top];
            double sum = 0.0, sum_without = 0.0;
            for (int v = 0; v < V; ++v) {
                const double e = std::exp((double) row[v] - maxv);
                sum += e;
                if (v != extra) sum_without += e;
            }
            const double lse = maxv + std::log(sum);
            const bool has_extra = extra >= 0 && extra < V;
            const double ex = has_extra ? (double) row[extra] - lse : NAN;
            const double without = has_extra && tgt != extra && sum_without > 0.0 ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
            std::fprintf(out, "%d\t%d\t%.9f\t%d\t%.9f\t%d\t%.9f\t%.9f", q, tgt, (double) row[tgt] - lse, top, maxv - lse,
                         (int) (top == tgt), ex, without);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + topk, order.end(), [&](int32_t x, int32_t y) { return row[x] > row[y]; });
            for (int j = 0; j < topk; ++j) std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
            std::fprintf(out, "\n");
            ++line;
        }
        llama_batch_free(b);
        std::fflush(out);
        seg_json += std::string(si ? ", " : "") + "{\"name\": \"" + s.name + "\", \"first_line\": " + std::to_string(first) +
                    ", \"end_line\": " + std::to_string(line) + "}";
        std::fprintf(stderr, "%s: %d positions\n", s.name.c_str(), n - 1);
    }
    std::fclose(out);
    seg_json += "], \"engine\": \"llama.cpp CPU\", \"model\": \"" + model + "\"}";
    std::ofstream(out_path + ".seg.json") << seg_json;
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
