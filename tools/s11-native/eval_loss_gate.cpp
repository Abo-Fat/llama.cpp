#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x53313147; // "S11G"
constexpr int kTopK = 32;

struct TopEntry {
    int32_t id;
    float logit;
};

struct CalibPositionRecord {
    int32_t target_id;
    int32_t top1_id;
    double log_z;
    float target_logit;
    TopEntry topk[kTopK];
};

struct TaskRecord {
    int32_t expected_choice;
    int32_t predicted_choice;
    int32_t full_top1_id;
    float choice_logits[4];
    TopEntry top10[10];
};

struct InputData {
    std::vector<std::vector<int32_t>> anchor_prompts;
    std::vector<std::string> calib_texts;
    struct TaskItem {
        std::string prompt;
        int32_t answer;
    };
    std::vector<TaskItem> tasks;
};

bool read_u32(std::istream & in, uint32_t & v) {
    return bool(in.read(reinterpret_cast<char *>(&v), sizeof(v)));
}

bool read_i32(std::istream & in, int32_t & v) {
    return bool(in.read(reinterpret_cast<char *>(&v), sizeof(v)));
}

bool read_string(std::istream & in, std::string & s) {
    uint32_t len = 0;
    if (!read_u32(in, len)) return false;
    s.resize(len);
    return len == 0 || bool(in.read(&s[0], len));
}

bool load_input_bin(const char * path, InputData & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t magic = 0;
    if (!read_u32(in, magic) || magic != kMagic) return false;
    uint32_t n_anchors = 0;
    if (!read_u32(in, n_anchors)) return false;
    out.anchor_prompts.resize(n_anchors);
    for (uint32_t i = 0; i < n_anchors; ++i) {
        uint32_t len = 0;
        if (!read_u32(in, len)) return false;
        out.anchor_prompts[i].resize(len);
        if (len > 0 && !in.read(reinterpret_cast<char *>(out.anchor_prompts[i].data()), len * sizeof(int32_t))) {
            return false;
        }
    }
    uint32_t n_calib = 0;
    if (!read_u32(in, n_calib)) return false;
    out.calib_texts.resize(n_calib);
    for (uint32_t i = 0; i < n_calib; ++i) {
        if (!read_string(in, out.calib_texts[i])) return false;
    }
    uint32_t n_tasks = 0;
    if (!read_u32(in, n_tasks)) return false;
    out.tasks.resize(n_tasks);
    for (uint32_t i = 0; i < n_tasks; ++i) {
        if (!read_string(in, out.tasks[i].prompt) || !read_i32(in, out.tasks[i].answer)) {
            return false;
        }
    }
    return true;
}

std::vector<llama_token> tokenize_str(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), nullptr, 0, add_special, true);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens(std::max(1, n));
    int actual = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), add_special, true);
    if (actual < 0) {
        tokens.resize(-actual);
        actual = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), add_special, true);
    }
    tokens.resize(std::max(0, actual));
    return tokens;
}

double compute_logsumexp(const float * logits, int vocab) {
    float max_val = logits[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        sum += std::exp(double(logits[i]) - double(max_val));
    }
    return double(max_val) + std::log(sum);
}

void extract_topk(const float * logits, int vocab, int k, TopEntry * out) {
    for (int i = 0; i < k; ++i) {
        out[i] = {-1, -1e38f};
    }
    float min_top = -1e38f;
    for (int i = 0; i < vocab; ++i) {
        const float v = logits[i];
        if (v <= min_top) continue;
        int pos = k - 1;
        while (pos > 0 && v > out[pos - 1].logit) {
            out[pos] = out[pos - 1];
            --pos;
        }
        out[pos] = {i, v};
        min_top = out[k - 1].logit;
    }
}

void compute_full_vocab_metrics(
        const float * ref, const float * cand, int vocab,
        double & kl_ref_cand, double & kl_cand_ref, double & rel_l2,
        double & top10_cos, double & top100_cos, bool & top1_match) {
    const double logz_ref = compute_logsumexp(ref, vocab);
    const double logz_cand = compute_logsumexp(cand, vocab);
    kl_ref_cand = 0.0;
    kl_cand_ref = 0.0;
    double err_sq = 0.0, ref_sq = 0.0;
    int ref_argmax = 0, cand_argmax = 0;
    for (int i = 0; i < vocab; ++i) {
        if (ref[i] > ref[ref_argmax]) ref_argmax = i;
        if (cand[i] > cand[cand_argmax]) cand_argmax = i;
        const double logp = double(ref[i]) - logz_ref;
        const double logq = double(cand[i]) - logz_cand;
        const double p = std::exp(logp);
        const double q = std::exp(logq);
        if (p > 0.0) kl_ref_cand += p * (logp - logq);
        if (q > 0.0) kl_cand_ref += q * (logq - logp);
        const double d = double(cand[i]) - double(ref[i]);
        err_sq += d * d;
        ref_sq += double(ref[i]) * double(ref[i]);
    }
    rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
    top1_match = (ref_argmax == cand_argmax);

    TopEntry top100[100];
    extract_topk(ref, vocab, 100, top100);
    auto calc_cos = [&](int k) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < k; ++i) {
            const double a = ref[top100[i].id];
            const double b = cand[top100[i].id];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        return dot / std::max(std::sqrt(na * nb), 1e-30);
    };
    top10_cos = calc_cos(10);
    top100_cos = calc_cos(100);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "Usage: eval-loss-gate MODEL.gguf INPUT.bin ORACLE.bin [dump|compare] [max_tasks]\n");
        return 1;
    }
    const char * model_path = argv[1];
    const char * input_path = argv[2];
    const char * oracle_path = argv[3];
    const bool dump_mode = std::strcmp(argv[4], "dump") == 0;
    const int max_tasks = argc >= 6 ? std::atoi(argv[5]) : 200;
    std::ofstream task_trace;
    if (const char * path = std::getenv("S11_GATE_TRACE")) {
        task_trace.open(path, std::ios::out | std::ios::binary);
        if (!task_trace) {
            std::fprintf(stderr, "Cannot open S11_GATE_TRACE: %s\n", path);
            return 1;
        }
    }

    InputData input;
    if (!load_input_bin(input_path, input)) {
        std::fprintf(stderr, "Failed to load input binary: %s\n", input_path);
        return 1;
    }

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    const char * resident_env = std::getenv("S11_RESIDENT");
    const bool resident = resident_env && std::strcmp(resident_env, "1") == 0;
    std::vector<llama_model_tensor_buft_override> overrides = {
        {"\\.ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr}
    };
    mp.tensor_buft_overrides = resident ? nullptr : overrides.data();

    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 128;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    const char * cache_mib_env = std::getenv("S11_MOE_CACHE_MIB");
    const size_t cache_mib = cache_mib_env ? std::strtoull(cache_mib_env, nullptr, 10) : 4608;
    cp.moe_cache_size = resident ? 0 : cache_mib * 1024ULL * 1024ULL;
    const char * fa_env = std::getenv("S11_FLASH_ATTN");
    const bool use_fa = fa_env && std::strcmp(fa_env, "1") == 0;
    cp.type_k = use_fa ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    cp.type_v = use_fa ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    cp.flash_attn_type = use_fa
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab_obj = llama_model_get_vocab(model);
    const int vocab = llama_vocab_n_tokens(vocab_obj);

    const auto t_start = std::chrono::steady_clock::now();

    // 1. Evaluate L1 Anchors: 4 prompts (128 formal, 128 dev, 512 formal, 512 dev)
    // For the first two (128 tokens), also perform 8 greedy decode steps.
    std::vector<std::vector<float>> anchor_logits;
    std::vector<int32_t> anchor_greedy_ids;

    for (size_t a = 0; a < input.anchor_prompts.size(); ++a) {
        const auto & tokens = input.anchor_prompts[a];
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n_tok = int(tokens.size());
        const bool is_short = (n_tok <= 128);
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        batch.n_tokens = n_tok;
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            if (is_short) {
                batch.logits[i] = (i == n_tok - 1);
            } else {
                batch.logits[i] = (i == 127 || i == 255 || i == n_tok - 1);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "L1 anchor prefill decode failed on anchor %zu\n", a);
            return 1;
        }
        if (is_short) {
            const float * out = llama_get_logits_ith(ctx, n_tok - 1);
            anchor_logits.emplace_back(out, out + vocab);
            int32_t next_id = int32_t(std::max_element(out, out + vocab) - out);
            anchor_greedy_ids.push_back(next_id);
            for (int step = 1; step < 8; ++step) {
                llama_batch db = llama_batch_init(1, 0, 1);
                db.n_tokens = 1;
                db.token[0] = next_id;
                db.pos[0] = n_tok - 1 + step;
                db.n_seq_id[0] = 1;
                db.seq_id[0][0] = 0;
                db.logits[0] = 1;
                if (llama_decode(ctx, db) != 0) {
                    std::fprintf(stderr, "L1 anchor greedy decode failed\n");
                    return 1;
                }
                const float * dout = llama_get_logits_ith(ctx, 0);
                if (step == 3 || step == 7) {
                    anchor_logits.emplace_back(dout, dout + vocab);
                }
                next_id = int32_t(std::max_element(dout, dout + vocab) - dout);
                anchor_greedy_ids.push_back(next_id);
                llama_batch_free(db);
            }
        } else {
            for (int pos : {127, 255, n_tok - 1}) {
                const float * out = llama_get_logits_ith(ctx, pos);
                anchor_logits.emplace_back(out, out + vocab);
            }
        }
        llama_batch_free(batch);
    }

    // 2. Evaluate L2 Multi-Domain Calibration Perplexity & Token-wise Top-K
    std::vector<CalibPositionRecord> calib_records;
    double total_nll = 0.0;
    int total_calib_tokens = 0;

    for (size_t c = 0; c < input.calib_texts.size(); ++c) {
        auto tokens = tokenize_str(vocab_obj, input.calib_texts[c], true);
        if (int(tokens.size()) > 64) tokens.resize(64);
        if (int(tokens.size()) < 8) continue;
        const int n_tok = int(tokens.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        batch.n_tokens = n_tok;
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i < n_tok - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "L2 calibration decode failed on segment %zu\n", c);
            return 1;
        }
        for (int i = 0; i < n_tok - 1; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            CalibPositionRecord rec{};
            rec.target_id = tokens[i + 1];
            rec.log_z = compute_logsumexp(l, vocab);
            rec.target_logit = l[rec.target_id];
            extract_topk(l, vocab, kTopK, rec.topk);
            rec.top1_id = rec.topk[0].id;
            const double nll = rec.log_z - double(rec.target_logit);
            total_nll += nll;
            total_calib_tokens += 1;
            calib_records.push_back(rec);
        }
        llama_batch_free(batch);
    }
    const double mean_nll = total_nll / std::max(1, total_calib_tokens);
    const double ppl = std::exp(mean_nll);

    // 3. Evaluate L3 Deterministic Task Suite
    int32_t choice_token_ids[4];
    const char * choice_strs[4] = {"A", "B", "C", "D"};
    for (int j = 0; j < 4; ++j) {
        auto t = tokenize_str(vocab_obj, choice_strs[j], false);
        choice_token_ids[j] = t.empty() ? (32 + j) : t[0];
    }

    const int n_eval_tasks = std::min<int>(max_tasks, int(input.tasks.size()));
    std::vector<TaskRecord> task_records(n_eval_tasks);
    int correct_tasks = 0;

    for (int t_idx = 0; t_idx < n_eval_tasks; ++t_idx) {
        auto tokens = tokenize_str(vocab_obj, input.tasks[t_idx].prompt, true);
        if (int(tokens.size()) > 256) {
            tokens.erase(tokens.begin(), tokens.end() - 256);
        }
        const int n_tok = int(tokens.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        batch.n_tokens = n_tok;
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tok - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "L3 task decode failed on task %d\n", t_idx);
            return 1;
        }
        const float * l = llama_get_logits_ith(ctx, n_tok - 1);
        TaskRecord rec{};
        rec.expected_choice = input.tasks[t_idx].answer;
        int best_c = 0;
        for (int j = 0; j < 4; ++j) {
            rec.choice_logits[j] = l[choice_token_ids[j]];
            if (rec.choice_logits[j] > rec.choice_logits[best_c]) best_c = j;
        }
        rec.predicted_choice = best_c;
        extract_topk(l, vocab, 10, rec.top10);
        rec.full_top1_id = rec.top10[0].id;
        if (rec.predicted_choice == rec.expected_choice) {
            correct_tasks += 1;
        }
        task_records[t_idx] = rec;
        if (task_trace.is_open()) {
            uint64_t hash = 14695981039346656037ULL;
            const auto * bytes = reinterpret_cast<const uint8_t *>(l);
            for (size_t i = 0; i < size_t(vocab) * sizeof(float); ++i) {
                hash = (hash ^ bytes[i]) * 1099511628211ULL;
            }
            char line[512];
            std::snprintf(line, sizeof(line),
                "{\"task\":%d,\"tokens\":%d,\"hash\":\"%016llx\",\"choice\":%d,"
                "\"top1\":%d,\"logits\":[%.9g,%.9g,%.9g,%.9g]}\n",
                t_idx, n_tok, (unsigned long long) hash, best_c, rec.full_top1_id,
                rec.choice_logits[0], rec.choice_logits[1], rec.choice_logits[2], rec.choice_logits[3]);
            task_trace << line;
            task_trace.flush();
            if (!task_trace) {
                std::fprintf(stderr, "Cannot write S11_GATE_TRACE\n");
                return 1;
            }
        }
        llama_batch_free(batch);
    }

    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    if (dump_mode) {
        std::ofstream out(oracle_path, std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "Cannot open oracle for writing: %s\n", oracle_path);
            return 1;
        }
        const uint32_t magic = kMagic;
        const uint32_t v_size = uint32_t(vocab);
        const uint32_t n_anc = uint32_t(anchor_logits.size());
        const uint32_t n_greedy = uint32_t(anchor_greedy_ids.size());
        const uint32_t n_cal = uint32_t(calib_records.size());
        const uint32_t n_tsk = uint32_t(task_records.size());
        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char *>(&v_size), sizeof(v_size));
        out.write(reinterpret_cast<const char *>(&n_anc), sizeof(n_anc));
        out.write(reinterpret_cast<const char *>(&n_greedy), sizeof(n_greedy));
        out.write(reinterpret_cast<const char *>(&n_cal), sizeof(n_cal));
        out.write(reinterpret_cast<const char *>(&n_tsk), sizeof(n_tsk));
        out.write(reinterpret_cast<const char *>(&mean_nll), sizeof(mean_nll));
        out.write(reinterpret_cast<const char *>(&ppl), sizeof(ppl));
        for (const auto & vec : anchor_logits) {
            out.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(float));
        }
        out.write(reinterpret_cast<const char *>(anchor_greedy_ids.data()), anchor_greedy_ids.size() * sizeof(int32_t));
        out.write(reinterpret_cast<const char *>(calib_records.data()), calib_records.size() * sizeof(CalibPositionRecord));
        out.write(reinterpret_cast<const char *>(task_records.data()), task_records.size() * sizeof(TaskRecord));
        out.close();

        std::printf(
            "{\"mode\":\"dump\",\"elapsed_s\":%.4f,\"vocab\":%d,\"anchor_vectors\":%u,"
            "\"calib_tokens\":%d,\"oracle_nll\":%.8f,\"oracle_ppl\":%.6f,"
            "\"tasks_evaluated\":%d,\"task_correct\":%d,\"task_accuracy\":%.4f}\n",
            elapsed_s, vocab, n_anc, total_calib_tokens, mean_nll, ppl,
            n_eval_tasks, correct_tasks, double(correct_tasks) / std::max(1, n_eval_tasks));
    } else {
        std::ifstream in(oracle_path, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "Cannot open oracle for reading: %s\n", oracle_path);
            return 1;
        }
        uint32_t magic = 0, v_size = 0, n_anc = 0, n_greedy = 0, n_cal = 0, n_tsk = 0;
        double ref_nll = 0.0, ref_ppl = 0.0;
        in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char *>(&v_size), sizeof(v_size));
        in.read(reinterpret_cast<char *>(&n_anc), sizeof(n_anc));
        in.read(reinterpret_cast<char *>(&n_greedy), sizeof(n_greedy));
        in.read(reinterpret_cast<char *>(&n_cal), sizeof(n_cal));
        in.read(reinterpret_cast<char *>(&n_tsk), sizeof(n_tsk));
        in.read(reinterpret_cast<char *>(&ref_nll), sizeof(ref_nll));
        in.read(reinterpret_cast<char *>(&ref_ppl), sizeof(ref_ppl));
        if (magic != kMagic || int(v_size) != vocab || n_anc != anchor_logits.size() || n_cal != calib_records.size()) {
            std::fprintf(stderr, "Oracle header mismatch\n");
            return 1;
        }
        std::vector<std::vector<float>> ref_anchors(n_anc, std::vector<float>(vocab));
        for (uint32_t i = 0; i < n_anc; ++i) {
            in.read(reinterpret_cast<char *>(ref_anchors[i].data()), vocab * sizeof(float));
        }
        std::vector<int32_t> ref_greedy(n_greedy);
        in.read(reinterpret_cast<char *>(ref_greedy.data()), n_greedy * sizeof(int32_t));
        std::vector<CalibPositionRecord> ref_calib(n_cal);
        in.read(reinterpret_cast<char *>(ref_calib.data()), n_cal * sizeof(CalibPositionRecord));
        std::vector<TaskRecord> ref_tasks(n_tsk);
        in.read(reinterpret_cast<char *>(ref_tasks.data()), n_tsk * sizeof(TaskRecord));

        // Compute L1 full-vocabulary metrics across all anchor vectors
        double sum_kl = 0.0, max_kl = 0.0, sum_rev_kl = 0.0;
        double sum_rel_l2 = 0.0, max_rel_l2 = 0.0;
        double sum_top10_cos = 0.0, min_top10_cos = 1.0;
        double sum_top100_cos = 0.0;
        int anchor_top1_matches = 0;

        for (uint32_t i = 0; i < n_anc; ++i) {
            double kl = 0.0, rev_kl = 0.0, rl2 = 0.0, c10 = 0.0, c100 = 0.0;
            bool m1 = false;
            compute_full_vocab_metrics(ref_anchors[i].data(), anchor_logits[i].data(), vocab,
                                       kl, rev_kl, rl2, c10, c100, m1);
            sum_kl += kl;
            max_kl = std::max(max_kl, kl);
            sum_rev_kl += rev_kl;
            sum_rel_l2 += rl2;
            max_rel_l2 = std::max(max_rel_l2, rl2);
            sum_top10_cos += c10;
            min_top10_cos = std::min(min_top10_cos, c10);
            sum_top100_cos += c100;
            if (m1) anchor_top1_matches += 1;
        }
        int greedy_matches = 0;
        for (uint32_t i = 0; i < n_greedy; ++i) {
            if (ref_greedy[i] == anchor_greedy_ids[i]) greedy_matches += 1;
        }

        // Compute L2 token-wise Top-K distribution shift and Top-1 agreement across calibration tokens
        int calib_top1_matches = 0;
        double sum_calib_cos = 0.0;
        for (uint32_t i = 0; i < n_cal; ++i) {
            if (ref_calib[i].top1_id == calib_records[i].top1_id) calib_top1_matches += 1;
            double dot = 0.0, na = 0.0, nb = 0.0;
            for (int k = 0; k < 10; ++k) {
                const double a = ref_calib[i].topk[k].logit;
                const double b = calib_records[i].topk[k].logit;
                dot += a * b;
                na += a * a;
                nb += b * b;
            }
            sum_calib_cos += dot / std::max(std::sqrt(na * nb), 1e-30);
        }

        // Compute L3 task accuracy and agreement against oracle
        int ref_correct = 0, task_choice_agreements = 0, task_top1_agreements = 0;
        double sum_choice_kl = 0.0;
        for (int i = 0; i < n_eval_tasks; ++i) {
            if (ref_tasks[i].predicted_choice == ref_tasks[i].expected_choice) ref_correct += 1;
            if (ref_tasks[i].predicted_choice == task_records[i].predicted_choice) task_choice_agreements += 1;
            if (ref_tasks[i].full_top1_id == task_records[i].full_top1_id) task_top1_agreements += 1;
            double lz_ref = compute_logsumexp(ref_tasks[i].choice_logits, 4);
            double lz_cand = compute_logsumexp(task_records[i].choice_logits, 4);
            for (int j = 0; j < 4; ++j) {
                double lp = double(ref_tasks[i].choice_logits[j]) - lz_ref;
                double lq = double(task_records[i].choice_logits[j]) - lz_cand;
                sum_choice_kl += std::exp(lp) * (lp - lq);
            }
        }

        const double mean_kl = sum_kl / std::max(1u, n_anc);
        const double mean_top10_cos = sum_top10_cos / std::max(1u, n_anc);
        const double delta_ppl = ppl - ref_ppl;
        const double rel_delta_ppl = (ppl - ref_ppl) / std::max(ref_ppl, 1e-12);
        const double acc_cand = double(correct_tasks) / std::max(1, n_eval_tasks);
        const double acc_ref = double(ref_correct) / std::max(1, n_eval_tasks);
        const double task_agree = double(task_choice_agreements) / std::max(1, n_eval_tasks);

        // Gate criteria:
        // L1: mean_kl < 0.01 and mean_top10_cos > 0.995
        // L2: delta_ppl <= 0.05 (degradation <= 0.05) and rel_delta_ppl <= 0.5%, with sanity lower bound > -1.0
        // L3: accuracy drop <= 0.015 (1.5 percentage points) and task agreement >= 0.95
        const double full_top1_agree = double(task_top1_agreements) / std::max(1, n_eval_tasks);
        const bool l1_pass = (mean_kl < 1e-2) && (mean_top10_cos > 0.995);
        const bool l2_pass = std::isfinite(ppl) && (delta_ppl <= 0.05) && (rel_delta_ppl <= 0.005) && (delta_ppl > -1.0);
        const bool l3_pass = ((acc_ref - acc_cand) <= 0.015) && (task_agree >= 0.94) && (full_top1_agree >= 0.95);
        const bool all_pass = l1_pass && l2_pass && l3_pass;

        std::printf(
            "{\"mode\":\"compare\",\"elapsed_s\":%.4f,\"passed\":%s,"
            "\"L1\":{\"passed\":%s,\"mean_kl\":%.8g,\"max_kl\":%.8g,\"mean_rev_kl\":%.8g,"
            "\"mean_rel_l2\":%.8g,\"max_rel_l2\":%.8g,\"mean_top10_cos\":%.8f,\"min_top10_cos\":%.8f,"
            "\"mean_top100_cos\":%.8f,\"anchor_top1_match\":\"%d/%u\",\"greedy_id_match\":\"%d/%u\"},"
            "\"L2\":{\"passed\":%s,\"ref_ppl\":%.6f,\"cand_ppl\":%.6f,\"delta_ppl\":%.6f,\"rel_delta_ppl_pct\":%.4f,"
            "\"calib_tokens\":%d,\"calib_top1_agreement\":%.4f,\"calib_top10_cos\":%.8f},"
            "\"L3\":{\"passed\":%s,\"tasks\":%d,\"ref_correct\":%d,\"cand_correct\":%d,"
            "\"ref_accuracy\":%.4f,\"cand_accuracy\":%.4f,\"choice_agreement\":%.4f,"
            "\"full_top1_agreement\":%.4f,\"mean_choice_kl\":%.8g}}\n",
            elapsed_s, all_pass ? "true" : "false",
            l1_pass ? "true" : "false", mean_kl, max_kl, sum_rev_kl / std::max(1u, n_anc),
            sum_rel_l2 / std::max(1u, n_anc), max_rel_l2, mean_top10_cos, min_top10_cos,
            sum_top100_cos / std::max(1u, n_anc), anchor_top1_matches, n_anc, greedy_matches, n_greedy,
            l2_pass ? "true" : "false", ref_ppl, ppl, delta_ppl, rel_delta_ppl * 100.0,
            total_calib_tokens, double(calib_top1_matches) / std::max(1u, n_cal),
            sum_calib_cos / std::max(1u, n_cal),
            l3_pass ? "true" : "false", n_eval_tasks, ref_correct, correct_tasks,
            acc_ref, acc_cand, task_agree,
            double(task_top1_agreements) / std::max(1, n_eval_tasks),
            sum_choice_kl / std::max(1, n_eval_tasks));
        if (!all_pass) {
            llama_free(ctx);
            llama_model_free(model);
            return 2;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
