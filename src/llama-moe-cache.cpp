#include "llama-moe-cache.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

// host memory available to new allocations in KiB, -1 if unknown
static long long llama_moe_cache_mem_available_kib() {
#ifdef __linux__
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    long long res = -1;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        long long v = 0;
        if (std::sscanf(line, "MemAvailable: %lld kB", &v) == 1) {
            res = v;
            break;
        }
    }
    std::fclose(f);
    return res;
#else
    return -1;
#endif
}

static std::string llama_moe_cache_memlock_limit() {
#ifdef __linux__
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        if (rl.rlim_cur == RLIM_INFINITY) {
            return "unlimited";
        }
        return std::to_string((unsigned long long) rl.rlim_cur / 1024) + " KiB";
    }
#endif
    return "unknown";
}

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    std::vector<bool>      dirty_layers; // [n_layer] true if slot_map[il] was modified since last GPU upload
    std::vector<uint8_t>   decode_pinned; // [n_slots] 1 if used by small (decode) batch
    int32_t                n_occupied = 0;
    int32_t                n_pinned   = 0;
    int32_t                max_pinned = 2048;
    int32_t                scratch_cursor = 0;
    std::array<std::array<uint64_t, 4>, 64> resident_mask{};
    std::array<std::array<uint64_t, 4>, 64> pinned_mask{};
    std::array<uint64_t, 64> dirty_gen{};

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // slots used by the layers of the current prefetched ubatch, the victims skip the ones of layers whose split has not run yet
    std::vector<uint32_t> stamp_epoch; // [n_slots]
    std::vector<int16_t>  stamp_layer; // [n_slots]
    uint32_t epoch       = 0;
    bool     stamping    = false; // only during a prefetched ubatch, the victims are unchanged otherwise
    uint64_t stamp_skips = 0;
    uint32_t guard_epoch2 = 0;    // head prefetch: all the slots stamped in this earlier epoch are guarded too
    int32_t  guard_layer2 = 0;    // prefill head prefetch: only the slots of guard_epoch2 stamped at layers >= guard_layer2 are guarded (0: all)
    bool     ahead_no_pinned = false; // prefill head prefetch: the decode-pinned slots are never victims of an ahead plan
    uint64_t pinned_skips = 0;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        dirty_layers.assign(n_layer, true);
        decode_pinned.assign(n_slots, 0);
        n_occupied = 0;
        n_pinned   = 0;
        max_pinned = std::min<int32_t>(3328, std::max<int32_t>(2048, 5406 - 1957));
        if (const char * env_pin = std::getenv("S11_MOE_MAX_PINNED")) {
            max_pinned = std::min<int32_t>(std::max<int32_t>(2048, std::atoi(env_pin)), std::max<int32_t>(2048, n_slots - 512));
        }
        scratch_cursor = 0;
        resident_mask = {};
        pinned_mask = {};
        dirty_gen.fill(1ULL);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
        stamp_epoch.assign(n_slots, 0u);
        stamp_layer.assign(n_slots, (int16_t) -1);
        epoch       = 0;
        stamping    = false;
        stamp_skips = 0;
        guard_epoch2 = 0;
        guard_layer2 = 0;
        ahead_no_pinned = false;
        pinned_skips = 0;
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    void begin_epoch() {
        if (++epoch == 0) {
            std::fill(stamp_epoch.begin(), stamp_epoch.end(), 0u);
            epoch = 1;
        }
        stamping = true;
    }

    // slot s is used by a layer >= guard of the current prefetched ubatch
    bool guarded(int32_t s, int32_t guard) const {
        return stamping && ((stamp_epoch[s] == epoch && stamp_layer[s] >= guard) ||
                            (guard_epoch2 != 0 && stamp_epoch[s] == guard_epoch2 &&
                             (guard_layer2 == 0 || stamp_layer[s] >= guard_layer2)));
    }

    void stamp(int32_t s, int32_t il) {
        if (stamping) {
            stamp_epoch[s] = epoch;
            stamp_layer[s] = (int16_t) il;
        }
    }

    // stamps the resident experts of a memoized layer without touching them, so the LRU order is unchanged
    void stamp_mask(int32_t il, const std::array<uint64_t, 4> & mask) {
        if (!stamping) {
            return;
        }
        const int32_t * slots = slot_map[il];
        for (int w = 0; w < 4; ++w) {
            uint64_t bits = mask[w];
            while (bits != 0) {
                const int32_t s = slots[w*64 + __builtin_ctzll(bits)];
                if (s >= 0) {
                    stamp(s, il);
                }
                bits &= (bits - 1);
            }
        }
    }

    // guarded slots are skipped, ahead returns -1 instead of falling back to them
    int32_t pick_victim_slot(int32_t il, bool is_decode, int32_t guard, bool ahead) {
        // When unfilled slots remain at head, always use head first
        if (key_of[head] < 0) {
            return head;
        }
        // Standard LRU from head, protecting decode-pinned slots during prefill
        int32_t cur = head;
        while (cur >= 0) {
            if ((key_of[cur] < 0 || (key_of[cur] / n_expert) != il) &&
                (is_decode || !decode_pinned[cur]) && (!ahead_no_pinned || !decode_pinned[cur])) {
                if (!guarded(cur, guard)) {
                    return cur;
                }
                stamp_skips++;
            }
            cur = next[cur];
        }
        cur = head;
        while (cur >= 0) {
            if (key_of[cur] < 0 || (key_of[cur] / n_expert) != il) {
                if (ahead_no_pinned && decode_pinned[cur]) {
                    pinned_skips++;
                } else if (!guarded(cur, guard)) {
                    return cur;
                } else {
                    stamp_skips++;
                }
            }
            cur = next[cur];
        }
        if (ahead) {
            return -1;
        }
        if (stamping) {
            cur = head;
            while (cur >= 0) {
                if (key_of[cur] < 0 || (key_of[cur] / n_expert) != il) {
                    return cur;
                }
                cur = next[cur];
            }
        }
        return head;
    }

    // ahead: planned before the splits of the layers >= guard have run, fails without any change if their slots would be needed
    bool plan_from_uniq(int32_t il, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, bool admit = true, bool ahead = false, int32_t guard = -1) {
        fills.clear();
        n_hit = 0;
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }
        if (guard < 0) {
            guard = il;
        }

        int32_t * slots = slot_map[il];
        const bool is_decode = n_ids <= 32;

        if (ahead) {
            size_t need = 0;
            for (int32_t e : uniq) {
                if (slots[e] < 0) {
                    need++;
                }
            }
            size_t avail = 0;
            for (int32_t cur = head; cur >= 0 && avail < need; cur = next[cur]) {
                if ((key_of[cur] < 0 || (key_of[cur] / n_expert) != il) && !guarded(cur, guard)) {
                    if (ahead_no_pinned && decode_pinned[cur]) {
                        pinned_skips++;
                    } else {
                        avail++;
                    }
                }
            }
            if (avail < need) {
                return false;
            }
        }

        for (int32_t e : uniq) {
            const int32_t s = slots[e];
            if (s >= 0) {
                if (is_decode && !decode_pinned[s] && n_pinned < max_pinned) {
                    decode_pinned[s] = 1;
                    n_pinned++;
                    if (il >= 0 && il < 64 && e >= 0 && e < 256) {
                        pinned_mask[il][e >> 6] |= (1ULL << (e & 63));
                    }
                }
                touch(s);
                stamp(s, il);
                n_hit++;
            }
        }
        if (!admit || n_hit == uniq.size()) {
            return true;
        }
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = pick_victim_slot(il, is_decode, guard, ahead);
            GGML_ASSERT(s >= 0);
            if (key_of[s] >= 0) {
                const int32_t old_il = key_of[s] / n_expert;
                const int32_t old_e  = key_of[s] % n_expert;
                slot_map[old_il][old_e] = -1;
                dirty_layers[old_il] = true;
                if (old_il >= 0 && old_il < 64 && old_e >= 0 && old_e < 256) {
                    resident_mask[old_il][old_e >> 6] &= ~(1ULL << (old_e & 63));
                    pinned_mask[old_il][old_e >> 6] &= ~(1ULL << (old_e & 63));
                    dirty_gen[old_il]++;
                }
                if (decode_pinned[s]) {
                    decode_pinned[s] = 0;
                    n_pinned--;
                }
            } else {
                n_occupied++;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            dirty_layers[il] = true;
            if (il >= 0 && il < 64 && e >= 0 && e < 256) {
                resident_mask[il][e >> 6] |= (1ULL << (e & 63));
                dirty_gen[il]++;
            }
            if (is_decode && n_pinned < max_pinned) {
                decode_pinned[s] = 1;
                n_pinned++;
                if (il >= 0 && il < 64 && e >= 0 && e < 256) {
                    pinned_mask[il][e >> 6] |= (1ULL << (e & 63));
                }
            }
            touch(s);
            stamp(s, il);
            fills.push_back({ e, s });
        }
        return true;
    }

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, bool admit = true) {
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        std::sort(uniq.begin(), uniq.end());
        return plan_from_uniq(il, n_ids, fills, n_hit, admit);
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        ggml_backend_event_ptr route_ready;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    uint64_t route_readbacks = 0;
    uint64_t route_syncs = 0;
    std::ofstream s11_route_trace;
    size_t s11_traced_ids = 0;
    bool s11_event_sync = false;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    struct ubatch_route_record {
        uint64_t valid_layers = 0;
        size_t   n_ids        = 0;
        std::array<std::array<uint64_t, 4>, 64> masks{};
    };

    uint64_t rolling_seq_hash    = 14695981039346656037ULL;
    uint64_t current_ubatch_hash = 0;
    uint64_t previous_prompt_hash = 0;
    int32_t  last_end_pos        = 0;
    std::array<uint64_t, 16384> pos_hash_history{};
    std::unordered_map<uint64_t, ubatch_route_record> known_ubatches;
    std::vector<const void *> last_dst_ptr;
    std::vector<uint64_t> uploaded_gen;
    std::array<std::pair<const ggml_tensor *, const ggml_tensor *>, 64> lookup_cache{};
    std::array<std::array<int32_t, 256>, 256> slot_map_ring{};
    size_t slot_map_ring_idx = 0;
    ggml_backend_buffer_ptr buf_stage;

    // expert prefetch of memoized ubatches, see pf_plan_window
    // disabled with S11_MOE_PREFETCH=0, S11_MOE_PREFETCH_SYNC=1 synchronizes around every copy stream dependency
    using pf_stream_new_t  = void * (*)(ggml_backend_t);
    using pf_stream_free_t = void (*)(void *);
    using pf_memcpy_t      = void (*)(void *, void *, const void *, size_t);
    using pf_event_t       = void (*)(void *, ggml_backend_event_t);
    using pf_sync_t        = void (*)(void *);
    using pf_capturing_t   = bool (*)(ggml_backend_t);
    static constexpr int32_t pf_lmax = 4;
    bool             host_pinned   = false;
    bool             pf_enabled    = false;
    bool             pf_sync       = false;
    ggml_backend_t   pf_backend    = nullptr;
    void *           pf_h2d        = nullptr;
    pf_stream_free_t pf_h2d_free   = nullptr;
    pf_memcpy_t      pf_h2d_memcpy = nullptr;
    pf_event_t       pf_h2d_record = nullptr;
    pf_event_t       pf_h2d_wait   = nullptr;
    pf_sync_t        pf_h2d_sync   = nullptr;
    pf_capturing_t   pf_capturing  = nullptr;
    ggml_backend_event_ptr pf_ev_start;              // compute -> copy stream, the previous ubatches are done
    std::vector<ggml_backend_event_ptr> pf_ev_prog;  // [n_layer] compute -> copy stream, the layers before il are done
    std::vector<ggml_backend_event_ptr> pf_ev_layer; // [n_layer] copy stream -> compute, the fills of layer il are done
    const ubatch_route_record * pf_rec = nullptr;
    std::vector<uint8_t> pf_state; // [n_layer] 0 not planned, 1 planned with fills, 2 planned without fills
    int32_t pf_next      = 0;
    bool    pf_stopped   = false;
    int32_t pf_mtp_il    = 0;
    int32_t pf_pending   = 0;
    int32_t pf_last_fill = -1;
    std::vector<moe_cache_lru::fill> pf_fills;
    uint64_t prefetch_layers = 0;
    uint64_t prefetch_bytes  = 0;
    uint64_t fallback_layers = 0;

    // cross-ubatch head prefetch, see end_ubatch
    static constexpr size_t   hd_elig_ids = 64;      // decode and verify ubatches only (up to 8 tokens)
    static constexpr size_t   hd_memo_cap = 65536;
    static constexpr uint64_t hd_terminal = ~0ULL;   // the ubatch was followed by the first ubatch of a sequence
    bool     hd_enabled          = false;
    bool     hd_force_mispredict = false;
    bool     hd_start_ok         = false;            // hd_ev_start was recorded before the current ubatch
    ggml_backend_event_ptr hd_ev_start;              // compute -> copy stream, the ubatches before the current one are done
    std::unordered_map<uint64_t, uint64_t> hd_next_of; // ubatch hash -> hash of its successor, 0 if it is not unique
    uint64_t hd_prev_hash = 0;
    bool     hd_active    = false;
    uint64_t hd_hash      = 0;
    const ubatch_route_record * hd_rec = nullptr;
    std::vector<uint8_t> hd_state; // [n_layer] like pf_state, for the head layers of the predicted ubatch
    int32_t  hd_next      = 0;
    int32_t  hd_last_fill = -1;
    uint64_t head_planned_layers = 0;
    uint64_t head_bytes          = 0;
    uint64_t head_adopted        = 0;
    uint64_t head_mispredict     = 0;
    uint64_t head_stamp_skips    = 0;

    // prefill-to-prefill head prefetch, see end_ubatch_prefill
    bool     hd_pf_enabled     = false;
    int32_t  hd_pf_G           = 0;                  // the copy stream waits for the splits of the layers < hd_pf_G of N
    ggml_backend_event_ptr hd_ev_mid;                // compute -> copy stream, recorded in copy() of layer hd_pf_G of N
    bool     hd_mid_armed      = false;
    bool     hd_mid_ok         = false;
    bool     hd_rec_is_prefill = false;
    uint64_t pfh_armed          = 0;
    uint64_t pfh_mid_recorded   = 0;
    uint64_t pfh_planned        = 0;
    uint64_t pfh_planned_layers = 0;
    uint64_t pfh_bytes          = 0;
    uint64_t pfh_adopted        = 0;
    uint64_t pfh_mispredicted   = 0;
    uint64_t pfh_dryrun_fail    = 0;
    uint64_t pfh_wrap_skips     = 0;
    uint64_t pfh_stamp_skips    = 0;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // host expert ranges page-locked by this cache (not the ones registered by someone else), see pin_host_experts
    std::vector<std::pair<void *, size_t>> pinned_ranges;
    void (*unpin_host_range)(void *) = nullptr;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), last_dst_ptr(model.layers.size(), nullptr), uploaded_gen(model.layers.size(), 0ULL) {
        if (const char * mode = std::getenv("S11_ROUTE_SYNC")) {
            if (std::strcmp(mode, "event") == 0) {
                if (!model.output || model.output->type != GGML_TYPE_S11_I8) {
                    throw std::runtime_error("S11 event route synchronization requires S11");
                }
                s11_event_sync = true;
            } else if (std::strcmp(mode, "backend") != 0) {
                throw std::runtime_error("Invalid S11_ROUTE_SYNC");
            }
        }
        if (const char * path = std::getenv("S11_ROUTE_TRACE")) {
            if (std::ifstream(path).good()) {
                throw std::runtime_error("S11 route trace already exists");
            }
            s11_route_trace.open(path);
            if (!s11_route_trace) {
                throw std::runtime_error("Cannot open S11 route trace");
            }
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
                if (s11_event_sync) {
                    d.route_ready.reset(ggml_backend_event_new(ggml_backend_get_device(d.backend)));
                    if (!d.route_ready) {
                        throw std::runtime_error("S11 route event is unsupported by this device");
                    }
                }
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            if (!devices.empty()) {
                ggml_backend_buffer_type_t stage_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
                if (stage_buft != nullptr) {
                    buf_stage.reset(ggml_backend_buft_alloc_buffer(stage_buft, sizeof(slot_map_ring)));
                }
            }

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }

            pin_host_experts();
            init_prefetch(model);
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        free_prefetch();
        unpin_host_experts();
    }

    // the misses of memoized ubatches are planned up to pf_lmax layers ahead, in layer order, and uploaded on a copy stream:
    // - the copy stream waits for pf_ev_start (previous ubatches) or pf_ev_prog[il] (the layers before il) before it reuses slots
    // - the split of layer il waits for pf_ev_layer[il]
    // - the slots of the layers >= il of the current ubatch are not evicted before the split of il has run (moe_cache_lru::guarded)
    void init_prefetch(const llama_model & model) {
        pf_mtp_il = std::min<int32_t>((int32_t) model.hparams.n_layer(), (int32_t) layers.size());
        const char * env = std::getenv("S11_MOE_PREFETCH");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: expert prefetch disabled (S11_MOE_PREFETCH=0)\n");
            return;
        }
        const char * s11_mode = std::getenv("S11_EXPERT_MODE");
        const bool hybrid = s11_mode != nullptr && std::strcmp(s11_mode, "hybrid") == 0;
        ggml_backend_t     be  = devices.size() == 1 ? devices[0].backend : nullptr;
        ggml_backend_dev_t dev = be != nullptr ? ggml_backend_get_device(be) : nullptr;
        ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        const char * reason = nullptr;
        if (be == nullptr || !devices[0].buf || reg == nullptr) {
            reason = "one GPU cache is required";
        } else if (ggml_backend_buft_is_host(devices[0].buft)) {
            reason = "the cache is in host memory";
        } else if (!host_pinned) {
            reason = "the host experts are not page-locked";
        } else if (hybrid || s11_event_sync || s11_route_trace.is_open()) {
            reason = "unsupported route mode";
        } else if (layers.size() > 64 || pf_mtp_il <= 0) {
            reason = "unsupported layer count";
        }
        pf_stream_new_t stream_new = nullptr;
        if (reason == nullptr) {
            stream_new    = (pf_stream_new_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_stream_new");
            pf_h2d_free   = (pf_stream_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_stream_free");
            pf_h2d_memcpy = (pf_memcpy_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_memcpy");
            pf_h2d_record = (pf_event_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_record");
            pf_h2d_wait   = (pf_event_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_wait");
            pf_h2d_sync   = (pf_sync_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_sync");
            pf_capturing  = (pf_capturing_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_stream_is_capturing");
            if (stream_new == nullptr || pf_h2d_free == nullptr || pf_h2d_memcpy == nullptr || pf_h2d_record == nullptr ||
                pf_h2d_wait == nullptr || pf_h2d_sync == nullptr || pf_capturing == nullptr) {
                reason = "not supported by the backend";
            }
        }
        if (reason == nullptr) {
            pf_h2d = stream_new(be);
            if (pf_h2d == nullptr) {
                reason = "the copy stream cannot be created";
            }
        }
        if (reason == nullptr) {
            pf_ev_start.reset(ggml_backend_event_new(dev));
            bool ok = pf_ev_start != nullptr;
            pf_ev_prog.resize(layers.size());
            pf_ev_layer.resize(layers.size());
            for (size_t il = 0; il < layers.size() && ok; ++il) {
                pf_ev_prog[il].reset(ggml_backend_event_new(dev));
                pf_ev_layer[il].reset(ggml_backend_event_new(dev));
                ok = pf_ev_prog[il] != nullptr && pf_ev_layer[il] != nullptr;
            }
            if (!ok) {
                reason = "events are not supported by the device";
            }
        }
        if (reason != nullptr) {
            LLAMA_LOG_INFO("llama_moe_cache: expert prefetch off (%s)\n", reason);
            free_prefetch();
            return;
        }
        const char * env_sync = std::getenv("S11_MOE_PREFETCH_SYNC");
        pf_sync    = env_sync != nullptr && std::strcmp(env_sync, "1") == 0;
        pf_backend = be;
        pf_state.assign(layers.size(), 0);
        pf_enabled = true;
        LLAMA_LOG_INFO("llama_moe_cache: expert prefetch on: up to %d layers ahead on a copy stream%s\n", (int) pf_lmax,
            pf_sync ? ", synchronized (S11_MOE_PREFETCH_SYNC=1)" : "");

        const char * env_hd = std::getenv("S11_MOE_HEAD_PREFETCH");
        if (env_hd != nullptr && std::strcmp(env_hd, "0") == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: head prefetch disabled (S11_MOE_HEAD_PREFETCH=0)\n");
            return;
        }
        hd_ev_start.reset(ggml_backend_event_new(dev));
        if (!hd_ev_start) {
            LLAMA_LOG_INFO("llama_moe_cache: head prefetch off (events are not supported by the device)\n");
            return;
        }
        const char * env_fm = std::getenv("S11_MOE_HEAD_FORCE_MISPREDICT");
        hd_force_mispredict = env_fm != nullptr && std::strcmp(env_fm, "1") == 0;
        hd_state.assign(layers.size(), 0);
        hd_next_of.reserve(4096);
        hd_enabled = true;
        LLAMA_LOG_INFO("llama_moe_cache: head prefetch on: the first %d layers of the memoized next ubatch%s\n", (int) pf_lmax,
            hd_force_mispredict ? ", never adopted (S11_MOE_HEAD_FORCE_MISPREDICT=1)" : "");

        const char * env_hp = std::getenv("S11_MOE_HEAD_PREFILL");
        if (env_hp != nullptr && std::strcmp(env_hp, "0") == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: prefill head prefetch disabled (S11_MOE_HEAD_PREFILL=0)\n");
            return;
        }
        if (pf_mtp_il - 1 < pf_lmax) {
            LLAMA_LOG_INFO("llama_moe_cache: prefill head prefetch off (unsupported layer count)\n");
            return;
        }
        int32_t G = pf_mtp_il - 1;
        if (const char * env_g = std::getenv("S11_MOE_HEAD_PREFILL_GUARD")) {
            G = std::atoi(env_g);
        }
        G = std::max<int32_t>(pf_lmax, std::min<int32_t>(G, pf_mtp_il - 1));
        hd_ev_mid.reset(ggml_backend_event_new(dev));
        if (!hd_ev_mid) {
            LLAMA_LOG_INFO("llama_moe_cache: prefill head prefetch off (events are not supported by the device)\n");
            return;
        }
        hd_pf_G       = G;
        hd_pf_enabled = true;
        LLAMA_LOG_INFO("llama_moe_cache: prefill head prefetch on: the first %d layers of the memoized next prefill ubatch after layer %d\n",
            (int) pf_lmax, (int) hd_pf_G);
    }

    void free_prefetch() {
        pf_enabled = false;
        pf_rec     = nullptr;
        if (pf_h2d != nullptr && pf_h2d_free != nullptr) {
            pf_h2d_free(pf_h2d); // synchronizes the copy stream first
        }
        pf_h2d = nullptr;
        pf_ev_layer.clear();
        pf_ev_prog.clear();
        pf_ev_start.reset();
        hd_enabled = false;
        hd_active  = false;
        hd_rec     = nullptr;
        hd_ev_start.reset();
        hd_pf_enabled     = false;
        hd_mid_armed      = false;
        hd_mid_ok         = false;
        hd_rec_is_prefill = false;
        hd_ev_mid.reset();
    }

    // called before the hash of the next ubatch is computed
    void pf_end_ubatch() {
        if (pf_pending > 0 && pf_last_fill >= 0) {
            // fills of layers whose split did not run: the later work of the compute stream is ordered after them
            ggml_backend_event_wait(pf_backend, pf_ev_layer[pf_last_fill].get());
        }
        pf_pending = 0;
        pf_rec     = nullptr;
        pf_next    = 0;
        pf_stopped = false;
        std::fill(pf_state.begin(), pf_state.end(), (uint8_t) 0);
        for (group & g : groups) {
            g.lru.stamping = false;
        }
    }

    // a memoized ubatch: the experts of its layers are known before its graph runs
    // adopt: its head layers were planned at the end of the previous ubatch, see end_ubatch
    void pf_start_ubatch(const llama_ubatch & ubatch, bool adopt) {
        if (current_ubatch_hash == 0 || ubatch.n_tokens == 0 || ubatch.n_tokens > 128) {
            if (adopt) {
                hd_mispredicted();
            }
            return;
        }
        const auto it = known_ubatches.find(current_ubatch_hash);
        if (it == known_ubatches.end() || it->second.valid_layers == 0 ||
            it->second.n_ids != (size_t) ubatch.n_tokens*(size_t) n_expert_used) {
            if (adopt) {
                hd_mispredicted();
            }
            return;
        }
        if (adopt && &it->second != hd_rec) {
            hd_mispredicted();
            adopt = false;
        }
        pf_rec = &it->second;
        if (adopt) {
            // keep the epoch of the head plan, its slots stay guarded until the splits of their layers have run
            for (group & g : groups) {
                if (g.n_slots > 0) {
                    g.lru.stamping = true;
                }
            }
            for (size_t j = 0; j < hd_state.size() && j < pf_state.size(); ++j) {
                if (hd_state[j] != 0) {
                    pf_state[j] = hd_state[j];
                    if (hd_state[j] == 1) {
                        pf_pending++;
                        pf_last_fill = (int32_t) j;
                    }
                }
            }
            pf_next    = hd_next;
            pf_stopped = false;
            if (hd_rec_is_prefill) {
                pfh_adopted++;
            } else {
                head_adopted++;
            }
            hd_clear();
            pf_plan_window(-1);
            return;
        }
        for (group & g : groups) {
            if (g.n_slots > 0) {
                g.lru.begin_epoch();
            }
        }
        pf_next    = 0;
        pf_stopped = false;
        pf_plan_window(-1);
    }

    void hd_clear() {
        hd_active    = false;
        hd_hash      = 0;
        hd_rec       = nullptr;
        hd_next      = 0;
        hd_last_fill = -1;
        hd_rec_is_prefill = false;
        std::fill(hd_state.begin(), hd_state.end(), (uint8_t) 0);
    }

    void hd_link(uint64_t prev, uint64_t succ) {
        const auto it = hd_next_of.find(prev);
        if (it == hd_next_of.end()) {
            if (hd_next_of.size() < hd_memo_cap) {
                hd_next_of.emplace(prev, succ);
            }
        } else if (it->second != succ) {
            it->second = 0;
        }
    }

    // the memoized successor of the ubatch h, if it is unique and a decode or verify ubatch
    const ubatch_route_record * hd_predict(uint64_t h, uint64_t & hp) const {
        const auto it = hd_next_of.find(h);
        if (it == hd_next_of.end() || it->second == 0 || it->second == hd_terminal) {
            return nullptr;
        }
        const auto ir = known_ubatches.find(it->second);
        if (ir == known_ubatches.end() || ir->second.valid_layers == 0 || ir->second.n_ids == 0 ||
            ir->second.n_ids > hd_elig_ids || ir->second.n_ids % (size_t) n_expert_used != 0) {
            return nullptr;
        }
        hp = it->second;
        return &ir->second;
    }

    // the ubatch is not the predicted one: its work on the compute stream is ordered after the head fills, which stay resident
    void hd_mispredicted() {
        if (hd_last_fill >= 0) {
            if (pf_sync) {
                pf_h2d_sync(pf_h2d);
            }
            ggml_backend_event_wait(pf_backend, pf_ev_layer[hd_last_fill].get());
        }
        if (hd_rec_is_prefill) {
            pfh_mispredicted++;
        } else {
            head_mispredict++;
        }
        hd_clear();
    }

    // the memoized successor of the ubatch h, if it is unique and a prefill ubatch whose head layers are memoized
    const ubatch_route_record * hd_predict_prefill(uint64_t h, uint64_t & hp) const {
        const auto it = hd_next_of.find(h);
        if (it == hd_next_of.end() || it->second == 0 || it->second == hd_terminal) {
            return nullptr;
        }
        const auto ir = known_ubatches.find(it->second);
        const uint64_t head_bits = (1ULL << pf_lmax) - 1;
        if (ir == known_ubatches.end() || ir->second.n_ids <= hd_elig_ids ||
            ir->second.n_ids > (size_t) 128*(size_t) n_expert_used || ir->second.n_ids % (size_t) n_expert_used != 0 ||
            (ir->second.valid_layers & head_bits) != head_bits) {
            return nullptr;
        }
        hp = it->second;
        return &ir->second;
    }

    // cross-ubatch head prefetch, after the graph of the current ubatch N was submitted:
    // on repeats the successor of a memoized ubatch is deterministic, the first pf_lmax layers of the predicted ubatch N+1
    // are planned now as pf_plan_window would at its begin, and uploaded on the copy stream during the tail of N
    // - the copy stream waits for hd_ev_start (recorded before N) before it reuses slots, all the slots of N are guarded
    // - the slot maps are uploaded as before: the copy callbacks of N have run, the ones of N+1 upload the new maps
    // - N+1 adopts the plan if its hash is the predicted one, otherwise its compute stream waits for the head fills
    // disabled with S11_MOE_HEAD_PREFETCH=0, S11_MOE_HEAD_FORCE_MISPREDICT=1 never adopts the plan (test mode)
    void end_ubatch() {
        const bool start_ok = hd_start_ok;
        hd_start_ok = false;
        const bool mid_armed = hd_mid_armed;
        const bool mid_ok    = hd_mid_ok;
        hd_mid_armed = false;
        hd_mid_ok    = false;
        if (hd_pf_enabled && pf_rec != nullptr && pf_rec->n_ids > hd_elig_ids) {
            end_ubatch_prefill(mid_armed && mid_ok);
            return;
        }
        if (!hd_enabled || !pf_enabled || !start_ok || hd_active || pf_rec == nullptr || current_ubatch_hash == 0 ||
            pf_rec->n_ids > hd_elig_ids) {
            return;
        }
        uint64_t hp = 0;
        const ubatch_route_record * rec = hd_predict(current_ubatch_hash, hp);
        if (rec == nullptr) {
            return;
        }
        for (const group & g : groups) {
            if (g.n_slots > 0 && !g.lru.stamping) {
                return; // the slots of N are not all stamped
            }
        }
        if (pf_capturing(pf_backend)) {
            return;
        }

        uint64_t skips0 = 0;
        for (const group & g : groups) {
            skips0 += g.lru.stamp_skips;
        }
        bool wrapped = false;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const uint32_t prev = g.lru.epoch;
            g.lru.begin_epoch();
            g.lru.guard_epoch2 = prev;
            wrapped = wrapped || g.lru.epoch == 1; // the stamps of N were cleared
        }

        std::fill(hd_state.begin(), hd_state.end(), (uint8_t) 0);
        hd_last_fill = -1;
        int32_t j = 0;
        int32_t n_planned = 0;
        if (!wrapped) {
            const size_t n_ids     = rec->n_ids;
            const bool   is_decode = n_ids <= 32;
            stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
            bool waited = false;
            for (; j < pf_mtp_il && j < pf_lmax; ++j) {
                const layer & l = layers[j];
                if (l.ig < 0) {
                    continue;
                }
                group & g = groups[l.ig];
                moe_cache_lru & lru = g.lru;
                if (j >= 64 || g.n_slots == 0 || l.experts.empty() || l.experts[0]->type != GGML_TYPE_S11_W2 || lru.n_expert > 256 ||
                    lru.slot_map[j] == nullptr || (rec->valid_layers & (1ULL << j)) == 0) {
                    break;
                }
                const auto & mask = rec->masks[j];
                uint64_t miss_pinned = 0;
                uint64_t miss_res    = 0;
                for (int w = 0; w < 4; ++w) {
                    miss_pinned |= mask[w] & ~lru.pinned_mask[j][w];
                    miss_res    |= mask[w] & ~lru.resident_mask[j][w];
                }
                size_t n_hit = 0;
                pf_fills.clear();
                if (miss_res == 0 && (!is_decode || miss_pinned == 0 || lru.n_pinned >= lru.max_pinned)) {
                    lru.stamp_mask(j, mask);
                    n_hit = (size_t) (__builtin_popcountll(mask[0]) + __builtin_popcountll(mask[1]) +
                                      __builtin_popcountll(mask[2]) + __builtin_popcountll(mask[3]));
                } else {
                    lru.uniq.clear();
                    for (int w = 0; w < 4; ++w) {
                        uint64_t bits = mask[w];
                        while (bits != 0) {
                            lru.uniq.push_back(w*64 + __builtin_ctzll(bits));
                            bits &= (bits - 1);
                        }
                    }
                    // dry run first, no change if the guarded slots of N and of the head layers would be needed
                    if (!lru.plan_from_uniq(j, n_ids, pf_fills, n_hit, true, true, 0)) {
                        break;
                    }
                }
                size_t bytes = 0;
                if (!pf_fills.empty()) {
                    if (!waited) {
                        pf_h2d_wait(pf_h2d, hd_ev_start.get());
                        waited = true;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * we   = l.experts[ip];
                        ggml_tensor       * bank = g.banks[ip];
                        const size_t expert_size = we->nb[2];
                        for (size_t i = 0; i < pf_fills.size();) {
                            size_t n = 1;
                            while (i + n < pf_fills.size() && pf_fills[i + n].expert == pf_fills[i].expert + (int32_t) n && pf_fills[i + n].slot == pf_fills[i].slot + (int32_t) n) {
                                n++;
                            }
                            pf_h2d_memcpy(pf_h2d, (uint8_t *) bank->data + pf_fills[i].slot*expert_size,
                                (const uint8_t *) we->data + pf_fills[i].expert*expert_size, n*expert_size);
                            bytes += n*expert_size;
                            i += n;
                        }
                    }
                    pf_h2d_record(pf_h2d, pf_ev_layer[j].get());
                    if (pf_sync) {
                        pf_h2d_sync(pf_h2d);
                    }
                    hd_state[j]  = 1;
                    hd_last_fill = j;
                } else {
                    hd_state[j] = 2;
                }
                st.hits   += n_hit;
                st.misses += pf_fills.size();
                st.bytes  += bytes;
                prefetch_layers++;
                prefetch_bytes += bytes;
                head_planned_layers++;
                head_bytes += bytes;
                n_planned++;
            }
        }

        uint64_t skips1 = 0;
        for (group & g : groups) {
            g.lru.stamping     = false;
            g.lru.guard_epoch2 = 0;
            skips1 += g.lru.stamp_skips;
        }
        head_stamp_skips += skips1 - skips0;

        if (n_planned > 0) {
            hd_active = true;
            hd_hash   = hp;
            hd_rec    = rec;
            hd_next   = j;
        } else {
            hd_clear();
        }
    }

    // prefill-to-prefill head prefetch, after the graph of the memoized prefill ubatch N was submitted:
    // the first pf_lmax layers of the predicted prefill ubatch N+1 are planned and uploaded during the tail of N
    // - the copy stream waits for hd_ev_mid, recorded on the compute stream in copy() of layer hd_pf_G of N, before its MoE split
    // - the slots stamped in N at the layers >= hd_pf_G stay guarded (guard_layer2), the slots of the head layers too (guard 0)
    // - the decode-pinned slots are never evicted (ahead_no_pinned), a failed dry run stops the plan without any change
    // - adoption and mispredictions are handled as for the decode head prefetch
    // disabled with S11_MOE_HEAD_PREFILL=0, nothing is planned if hd_ev_mid was not recorded in N
    void end_ubatch_prefill(bool ok) {
        if (!ok || !hd_enabled || !pf_enabled || hd_active || pf_rec == nullptr || current_ubatch_hash == 0) {
            return;
        }
        uint64_t hp = 0;
        const ubatch_route_record * rec = hd_predict_prefill(current_ubatch_hash, hp);
        if (rec == nullptr) {
            return;
        }
        for (const group & g : groups) {
            if (g.n_slots > 0 && !g.lru.stamping) {
                return; // the slots of N are not all stamped
            }
        }
        if (pf_capturing(pf_backend)) {
            return;
        }

        uint64_t skips0 = 0;
        for (const group & g : groups) {
            skips0 += g.lru.stamp_skips;
        }
        bool wrapped = false;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const uint32_t prev = g.lru.epoch;
            g.lru.begin_epoch();
            g.lru.guard_epoch2    = prev;
            g.lru.guard_layer2    = hd_pf_G;
            g.lru.ahead_no_pinned = true;
            wrapped = wrapped || g.lru.epoch == 1; // the stamps of N were cleared
        }
        if (wrapped) {
            pfh_wrap_skips++;
        }

        std::fill(hd_state.begin(), hd_state.end(), (uint8_t) 0);
        hd_last_fill = -1;
        int32_t j = 0;
        int32_t n_planned = 0;
        if (!wrapped) {
            const size_t n_ids     = rec->n_ids;
            const bool   is_decode = n_ids <= 32;
            stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
            bool waited = false;
            for (; j < pf_mtp_il && j < pf_lmax; ++j) {
                const layer & l = layers[j];
                if (l.ig < 0) {
                    continue;
                }
                group & g = groups[l.ig];
                moe_cache_lru & lru = g.lru;
                if (j >= 64 || g.n_slots == 0 || l.experts.empty() || l.experts[0]->type != GGML_TYPE_S11_W2 || lru.n_expert > 256 ||
                    lru.slot_map[j] == nullptr || (rec->valid_layers & (1ULL << j)) == 0) {
                    break;
                }
                const auto & mask = rec->masks[j];
                uint64_t miss_pinned = 0;
                uint64_t miss_res    = 0;
                for (int w = 0; w < 4; ++w) {
                    miss_pinned |= mask[w] & ~lru.pinned_mask[j][w];
                    miss_res    |= mask[w] & ~lru.resident_mask[j][w];
                }
                size_t n_hit = 0;
                pf_fills.clear();
                if (miss_res == 0 && (!is_decode || miss_pinned == 0 || lru.n_pinned >= lru.max_pinned)) {
                    lru.stamp_mask(j, mask);
                    n_hit = (size_t) (__builtin_popcountll(mask[0]) + __builtin_popcountll(mask[1]) +
                                      __builtin_popcountll(mask[2]) + __builtin_popcountll(mask[3]));
                } else {
                    lru.uniq.clear();
                    for (int w = 0; w < 4; ++w) {
                        uint64_t bits = mask[w];
                        while (bits != 0) {
                            lru.uniq.push_back(w*64 + __builtin_ctzll(bits));
                            bits &= (bits - 1);
                        }
                    }
                    // dry run first, no change if the guarded slots of N, of the head layers or the pinned slots would be needed
                    if (!lru.plan_from_uniq(j, n_ids, pf_fills, n_hit, true, true, 0)) {
                        pfh_dryrun_fail++;
                        break;
                    }
                }
                size_t bytes = 0;
                if (!pf_fills.empty()) {
                    if (!waited) {
                        // the fills may reuse the slots of the layers < hd_pf_G of N: the copy stream waits for their splits
                        pf_h2d_wait(pf_h2d, hd_ev_mid.get());
                        waited = true;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * we   = l.experts[ip];
                        ggml_tensor       * bank = g.banks[ip];
                        const size_t expert_size = we->nb[2];
                        for (size_t i = 0; i < pf_fills.size();) {
                            size_t n = 1;
                            while (i + n < pf_fills.size() && pf_fills[i + n].expert == pf_fills[i].expert + (int32_t) n && pf_fills[i + n].slot == pf_fills[i].slot + (int32_t) n) {
                                n++;
                            }
                            pf_h2d_memcpy(pf_h2d, (uint8_t *) bank->data + pf_fills[i].slot*expert_size,
                                (const uint8_t *) we->data + pf_fills[i].expert*expert_size, n*expert_size);
                            bytes += n*expert_size;
                            i += n;
                        }
                    }
                    pf_h2d_record(pf_h2d, pf_ev_layer[j].get());
                    if (pf_sync) {
                        pf_h2d_sync(pf_h2d);
                    }
                    hd_state[j]  = 1;
                    hd_last_fill = j;
                } else {
                    hd_state[j] = 2;
                }
                st.hits   += n_hit;
                st.misses += pf_fills.size();
                st.bytes  += bytes;
                prefetch_layers++;
                prefetch_bytes += bytes;
                pfh_planned_layers++;
                pfh_bytes += bytes;
                n_planned++;
            }
        }

        uint64_t skips1 = 0;
        for (group & g : groups) {
            g.lru.stamping        = false;
            g.lru.guard_epoch2    = 0;
            g.lru.guard_layer2    = 0;
            g.lru.ahead_no_pinned = false;
            skips1 += g.lru.stamp_skips;
        }
        pfh_stamp_skips += skips1 - skips0;

        if (n_planned > 0) {
            hd_active = true;
            hd_hash   = hp;
            hd_rec    = rec;
            hd_next   = j;
            hd_rec_is_prefill = true;
            pfh_planned++;
        } else {
            hd_clear();
        }
    }

    // plans the memoized experts of the layers after cur (-1 before the first layer) up to pf_lmax layers ahead, in layer order,
    // exactly as copy would; stops at the first layer it cannot plan, which copy plans in-stream before the window resumes
    void pf_plan_window(int32_t cur) {
        const int32_t guard = cur < 0 ? 0 : cur;
        ggml_backend_event_t sync_ev = cur < 0 ? pf_ev_start.get() : pf_ev_prog[cur].get();
        bool waited = false;
        const size_t n_ids     = pf_rec->n_ids;
        const bool   is_decode = n_ids <= 32;
        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        while (!pf_stopped && pf_next < pf_mtp_il && pf_next <= cur + pf_lmax) {
            const int32_t j = pf_next;
            const layer & l = layers[j];
            if (l.ig < 0) {
                pf_next++;
                continue;
            }
            group & g = groups[l.ig];
            moe_cache_lru & lru = g.lru;
            if (j >= 64 || g.n_slots == 0 || l.experts.empty() || l.experts[0]->type != GGML_TYPE_S11_W2 || lru.n_expert > 256 ||
                lru.slot_map[j] == nullptr || (pf_rec->valid_layers & (1ULL << j)) == 0) {
                pf_stopped = true;
                break;
            }
            const auto & mask = pf_rec->masks[j];
            uint64_t miss_pinned = 0;
            uint64_t miss_res    = 0;
            for (int w = 0; w < 4; ++w) {
                miss_pinned |= mask[w] & ~lru.pinned_mask[j][w];
                miss_res    |= mask[w] & ~lru.resident_mask[j][w];
            }
            size_t n_hit = 0;
            pf_fills.clear();
            if (miss_res == 0 && (!is_decode || miss_pinned == 0 || lru.n_pinned >= lru.max_pinned)) {
                lru.stamp_mask(j, mask);
                n_hit = (size_t) (__builtin_popcountll(mask[0]) + __builtin_popcountll(mask[1]) +
                                  __builtin_popcountll(mask[2]) + __builtin_popcountll(mask[3]));
            } else {
                lru.uniq.clear();
                for (int w = 0; w < 4; ++w) {
                    uint64_t bits = mask[w];
                    while (bits != 0) {
                        lru.uniq.push_back(w*64 + __builtin_ctzll(bits));
                        bits &= (bits - 1);
                    }
                }
                if (!lru.plan_from_uniq(j, n_ids, pf_fills, n_hit, true, true, guard)) {
                    pf_stopped = true;
                    break;
                }
            }
            size_t bytes = 0;
            if (!pf_fills.empty()) {
                if (!waited) {
                    // the fills may reuse the slots of the layers before cur: the copy stream waits for their splits
                    if (pf_sync) {
                        ggml_backend_synchronize(pf_backend);
                    }
                    ggml_backend_event_record(sync_ev, pf_backend);
                    pf_h2d_wait(pf_h2d, sync_ev);
                    waited = true;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * we   = l.experts[ip];
                    ggml_tensor       * bank = g.banks[ip];
                    const size_t expert_size = we->nb[2];
                    for (size_t i = 0; i < pf_fills.size();) {
                        size_t n = 1;
                        while (i + n < pf_fills.size() && pf_fills[i + n].expert == pf_fills[i].expert + (int32_t) n && pf_fills[i + n].slot == pf_fills[i].slot + (int32_t) n) {
                            n++;
                        }
                        pf_h2d_memcpy(pf_h2d, (uint8_t *) bank->data + pf_fills[i].slot*expert_size,
                            (const uint8_t *) we->data + pf_fills[i].expert*expert_size, n*expert_size);
                        bytes += n*expert_size;
                        i += n;
                    }
                }
                pf_h2d_record(pf_h2d, pf_ev_layer[j].get());
                if (pf_sync) {
                    pf_h2d_sync(pf_h2d);
                }
                pf_state[j]  = 1;
                pf_pending++;
                pf_last_fill = j;
            } else {
                pf_state[j] = 2;
            }
            st.hits   += n_hit;
            st.misses += pf_fills.size();
            st.bytes  += bytes;
            prefetch_layers++;
            prefetch_bytes += bytes;
            pf_next++;
        }
    }

    // page-locks the host experts of the cached layers so the uploads of the misses are DMA from pinned memory
    // the bytes, slots and order of the uploads are unchanged; all or nothing, on any failure the experts stay pageable
    // disabled with S11_MOE_PIN=0; falls back if the device free memory drops by more than S11_MOE_PIN_MAX_DEV_MIB (default 40)
    void pin_host_experts() {
        const char * env = std::getenv("S11_MOE_PIN");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            LLAMA_LOG_INFO("llama_moe_cache: host expert pinning disabled (S11_MOE_PIN=0)\n");
            return;
        }
        if (devices.empty()) {
            return;
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            return;
        }
        using pin_fn_t    = bool (*)(void *, size_t, int *);
        using unpin_fn_t  = void (*)(void *);
        using is_reg_fn_t = bool (*)(const void *);
        auto pin_fn    = (pin_fn_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pin_host_range");
        auto unpin_fn  = (unpin_fn_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_unpin_host_range");
        auto is_reg_fn = (is_reg_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_is_host_registered");
        if (pin_fn == nullptr || unpin_fn == nullptr || is_reg_fn == nullptr) {
            LLAMA_LOG_INFO("llama_moe_cache: host expert pinning is not supported by %s\n", ggml_backend_reg_name(reg));
            return;
        }

        size_t page = 4096;
#ifdef __linux__
        {
            const long ps = sysconf(_SC_PAGESIZE);
            if (ps > 0) {
                page = (size_t) ps;
            }
        }
#endif

        // exact expert ranges rounded out to pages, merged where they overlap or touch
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        size_t expert_bytes = 0;
        const void * first_ptr = nullptr;
        const void * last_ptr  = nullptr;
        for (const layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (const ggml_tensor * w : l.experts) {
                const size_t nb = ggml_nbytes(w);
                if (nb == 0) {
                    continue;
                }
                if (w->data == nullptr) {
                    LLAMA_LOG_WARN("llama_moe_cache: host expert %s has no data, the experts stay pageable\n", w->name);
                    return;
                }
                const uintptr_t b = (uintptr_t) w->data;
                ranges.emplace_back(b & ~(uintptr_t) (page - 1), (b + nb + page - 1) & ~(uintptr_t) (page - 1));
                expert_bytes += nb;
                if (first_ptr == nullptr) {
                    first_ptr = w->data;
                }
                last_ptr = (const uint8_t *) w->data + nb - 1;
            }
        }
        if (ranges.empty()) {
            return;
        }
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;
        for (const auto & r : ranges) {
            if (!merged.empty() && r.first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, r.second);
            } else {
                merged.push_back(r);
            }
        }
        size_t merged_bytes = 0;
        for (const auto & r : merged) {
            merged_bytes += r.second - r.first;
        }

        size_t free0 = 0;
        size_t free1 = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free0, &total);
        const long long avail0 = llama_moe_cache_mem_available_kib();

        const auto t0 = std::chrono::steady_clock::now();
        bool registered = true;
        size_t owned    = 0;
        size_t external = 0;
        for (const auto & r : merged) {
            void * p = (void *) r.first;
            const size_t n = r.second - r.first;
            int already = 0;
            if (!pin_fn(p, n, &already)) {
                registered = false;
                break;
            }
            if (already) {
                external += n;
            } else {
                pinned_ranges.emplace_back(p, n);
                owned += n;
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        ggml_backend_dev_memory(dev, &free1, &total);
        const long long avail1 = llama_moe_cache_mem_available_kib();
        const bool first_ok = registered && is_reg_fn(first_ptr);
        const bool last_ok  = registered && is_reg_fn(last_ptr);
        const double dev_delta_mib = free0 > free1 ? (free0 - free1)/1024.0/1024.0 : 0.0;
        double max_dev_mib = 40.0;
        if (const char * s = std::getenv("S11_MOE_PIN_MAX_DEV_MIB")) {
            max_dev_mib = std::atof(s);
        }

        const char * status = "on";
        if (!registered) {
            status = "off (register failed)";
        } else if (!first_ok || !last_ok) {
            status = "off (experts not registered)";
        } else if (dev_delta_mib > max_dev_mib) {
            status = "off (device memory delta too large)";
        }
        const bool ok = registered && first_ok && last_ok && dev_delta_mib <= max_dev_mib;
        if (!ok) {
            for (const auto & r : pinned_ranges) {
                unpin_fn(r.first);
            }
            pinned_ranges.clear();
        } else {
            unpin_host_range = unpin_fn;
            host_pinned = true;
        }

        LLAMA_LOG_INFO("llama_moe_cache: host expert pinning %s: %zu ranges, experts %.2f MiB, ranges %.2f MiB (%.4fx), "
            "owned %.2f MiB, external %.2f MiB, %.1f ms, device free delta %.2f MiB (max %.1f), "
            "MemAvailable %.1f -> %.1f MiB, RLIMIT_MEMLOCK %s, first/last registered %d/%d\n",
            status, merged.size(), expert_bytes/1024.0/1024.0, merged_bytes/1024.0/1024.0,
            expert_bytes > 0 ? (double) merged_bytes/expert_bytes : 0.0,
            owned/1024.0/1024.0, external/1024.0/1024.0, ms, dev_delta_mib, max_dev_mib,
            avail0 >= 0 ? avail0/1024.0 : -1.0, avail1 >= 0 ? avail1/1024.0 : -1.0,
            llama_moe_cache_memlock_limit().c_str(), (int) first_ok, (int) last_ok);
    }

    void unpin_host_experts() {
        if (unpin_host_range != nullptr) {
            for (const auto & r : pinned_ranges) {
                unpin_host_range(r.first);
            }
        }
        pinned_ranges.clear();
        unpin_host_range = nullptr;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        const int64_t batch_limit = !l.experts.empty() && l.experts[0]->type == GGML_TYPE_S11_W2 ? 128 : max_batch;
        if (n_tokens == 0 || n_tokens > batch_limit || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    void begin_ubatch(const llama_ubatch & ubatch, bool graph_reused) {
        if (!graph_reused) {
            std::fill(last_dst_ptr.begin(), last_dst_ptr.end(), nullptr);
        }
        if (pf_enabled) {
            pf_end_ubatch();
        }
        hd_start_ok  = false;
        hd_mid_armed = false;
        hd_mid_ok    = false;
        if (!ubatch.token || ubatch.n_tokens == 0) {
            current_ubatch_hash = 0;
            hd_prev_hash        = 0;
            if (hd_active) {
                hd_mispredicted();
            }
            return;
        }
        if (ubatch.pos && ubatch.pos[0] == 0) {
            rolling_seq_hash = 14695981039346656037ULL;
            last_end_pos     = 0;
            pos_hash_history[0] = rolling_seq_hash;
        } else if (ubatch.pos && ubatch.pos[0] > 0 && ubatch.pos[0] < last_end_pos &&
                   (size_t) ubatch.pos[0] < pos_hash_history.size() && pos_hash_history[ubatch.pos[0]] != 0) {
            rolling_seq_hash = pos_hash_history[ubatch.pos[0]];
        }
        uint64_t h = rolling_seq_hash;
        h = (h ^ uint64_t(ubatch.n_tokens)) * 1099511628211ULL;
        if (ubatch.pos) {
            h = (h ^ uint64_t(uint32_t(ubatch.pos[0]))) * 1099511628211ULL;
        }
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            h = (h ^ uint64_t(uint32_t(ubatch.token[i]))) * 1099511628211ULL;
        }
        if (ubatch.pos && ubatch.pos[0] >= 0) {
            uint64_t ph = rolling_seq_hash;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const int32_t p = ubatch.pos[i];
                ph = (ph ^ uint64_t(1u)) * 1099511628211ULL;
                ph = (ph ^ uint64_t(uint32_t(p))) * 1099511628211ULL;
                ph = (ph ^ uint64_t(uint32_t(ubatch.token[i]))) * 1099511628211ULL;
                if (p + 1 >= 0 && (size_t) (p + 1) < pos_hash_history.size()) {
                    pos_hash_history[p + 1] = ph ? ph : 1ULL;
                }
            }
            last_end_pos = ubatch.pos[ubatch.n_tokens - 1] + 1;
        }
        rolling_seq_hash    = h;
        current_ubatch_hash = h ? h : 1ULL;
        bool adopt = false;
        if (hd_enabled) {
            // successor memo from the ubatch hashes only, the first ubatch of a sequence is never predicted
            const bool first = ubatch.pos && ubatch.pos[0] == 0;
            if (hd_prev_hash != 0) {
                hd_link(hd_prev_hash, first ? hd_terminal : current_ubatch_hash);
            }
            hd_prev_hash = current_ubatch_hash;
            if (hd_active) {
                adopt = !hd_force_mispredict && current_ubatch_hash == hd_hash;
                if (!adopt) {
                    hd_mispredicted();
                }
            }
        }
        static const bool prompt_admission = [] {
            const char * value = std::getenv("S11_MOE_PROMPT_ADMISSION");
            if (!value || std::strcmp(value, "0") == 0) { return false; }
            if (std::strcmp(value, "1") == 0) { return true; }
            GGML_ABORT("Invalid S11_MOE_PROMPT_ADMISSION: %s", value);
        }();
        if (prompt_admission && ubatch.pos && ubatch.pos[0] == 0) {
            GGML_ASSERT(ubatch.n_seqs_unq == 1 && "S11 prompt admission requires a single sequence");
            if (!adopt && previous_prompt_hash != 0 && previous_prompt_hash != current_ubatch_hash) {
                int cleared = 0;
                for (group & g : groups) {
                    if (g.n_slots > 0 && !g.ref.empty() && g.ref[0]->type == GGML_TYPE_S11_W2) {
                        cleared += g.lru.n_pinned;
                        std::fill(g.lru.decode_pinned.begin(), g.lru.decode_pinned.end(), 0);
                        g.lru.pinned_mask = {};
                        g.lru.n_pinned = 0;
                    }
                }
                LLAMA_LOG_INFO("llama_moe_cache: prompt switch released %d decode pins (weights retained)\n", cleared);
            }
            previous_prompt_hash = current_ubatch_hash;
        }
        if (pf_enabled) {
            pf_start_ubatch(ubatch, adopt);
        }
        if (hd_enabled && pf_rec != nullptr && pf_rec->n_ids <= hd_elig_ids) {
            uint64_t hp = 0;
            if (hd_predict(current_ubatch_hash, hp) != nullptr) {
                // the head fills planned at the end of this ubatch may reuse the slots of the ubatches before it
                if (pf_sync) {
                    ggml_backend_synchronize(pf_backend);
                }
                ggml_backend_event_record(hd_ev_start.get(), pf_backend);
                hd_start_ok = true;
            }
        }
        if (hd_pf_enabled && hd_enabled && pf_enabled && !hd_active && pf_rec != nullptr && pf_rec->n_ids > hd_elig_ids &&
            pf_rec->n_ids <= (size_t) 128*(size_t) n_expert_used) {
            bool all_stamping = true;
            for (const group & g : groups) {
                if (g.n_slots > 0 && !g.lru.stamping) {
                    all_stamping = false;
                }
            }
            uint64_t hp = 0;
            if (all_stamping && hd_predict_prefill(current_ubatch_hash, hp) != nullptr) {
                // hd_ev_mid is recorded in copy() of layer hd_pf_G, see end_ubatch_prefill
                hd_mid_armed = true;
                pfh_armed++;
            }
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // prefill head prefetch: the work of N before the MoE split of layer hd_pf_G, see end_ubatch_prefill
        if (hd_mid_armed && !hd_mid_ok && pf_enabled && il == hd_pf_G && backend == pf_backend && !pf_capturing(backend)) {
            if (pf_sync) {
                ggml_backend_synchronize(pf_backend);
            }
            ggml_backend_event_record(hd_ev_mid.get(), pf_backend);
            hd_mid_ok = true;
            pfh_mid_recorded++;
        }

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        if (n_nodes > 0 && il >= 0 && il < 64 && lookup_cache[il].first == ggml_graph_node(graph, 0) &&
            lookup_cache[il].second != nullptr && lookup_cache[il].second->op == GGML_OP_GET_ROWS &&
            lookup_cache[il].second->src[0] == dst) {
            lookup = lookup_cache[il].second;
        } else {
            for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                    lookup = node;
                }
            }
            if (n_nodes > 0 && il >= 0 && il < 64) {
                lookup_cache[il] = { ggml_graph_node(graph, 0), lookup };
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
#ifndef NDEBUG
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
#endif
        GGML_ASSERT(ggml_is_contiguous(sel));

        const size_t n_sel = ggml_nelements(sel);
        static const bool s11_hybrid_mode = [] {
            const char * s11_mode = std::getenv("S11_EXPERT_MODE");
            return s11_mode && std::strcmp(s11_mode, "hybrid") == 0;
        }();
        const bool cpu_miss = !l.experts.empty() && l.experts[0]->type == GGML_TYPE_S11_W2 &&
            s11_hybrid_mode && n_sel <= 8;
        const bool is_s11_fast = !l.experts.empty() && l.experts[0]->type == GGML_TYPE_S11_W2 &&
            !cpu_miss && !s11_route_trace.is_open() && !s11_event_sync &&
            n_sel <= (size_t) (128 * n_expert_used) && il >= 0 && il < 64 && g.lru.n_expert <= 256 && current_ubatch_hash != 0;

        size_t n_hit = 0;
        bool used_fast_route = false;

        // the experts of this layer were planned ahead and uploaded on the copy stream, see pf_plan_window
        bool prefetched = false;
        if (pf_rec != nullptr && il < (int32_t) pf_state.size() && pf_state[il] != 0) {
            if (pf_sync && pf_capturing(backend)) {
                GGML_ABORT("MoE cache prefetch: the compute stream is capturing at the copy of layer %d", il);
            }
            if (pf_state[il] == 1) {
                // the split of this layer reads the slots filled on the copy stream
                if (pf_sync) {
                    pf_h2d_sync(pf_h2d);
                }
                ggml_backend_event_wait(backend, pf_ev_layer[il].get());
                pf_pending--;
            }
            pf_state[il] = 0;
            prefetched = is_s11_fast && n_sel == pf_rec->n_ids;
            if (prefetched) {
                fills.clear();
            }
        }
        if (pf_rec != nullptr && !prefetched && il < pf_mtp_il) {
            fallback_layers++;
        }

        if (is_s11_fast && !prefetched) {
            auto it_rec = known_ubatches.find(current_ubatch_hash);
            if (it_rec != known_ubatches.end() &&
                it_rec->second.n_ids == n_sel &&
                (it_rec->second.valid_layers & (1ULL << il)) != 0) {
                const auto & mask = it_rec->second.masks[il];
                const bool is_decode = n_sel <= 32;
                const uint64_t miss_pinned =
                    (mask[0] & ~g.lru.pinned_mask[il][0]) |
                    (mask[1] & ~g.lru.pinned_mask[il][1]) |
                    (mask[2] & ~g.lru.pinned_mask[il][2]) |
                    (mask[3] & ~g.lru.pinned_mask[il][3]);
                const uint64_t miss_res =
                    (mask[0] & ~g.lru.resident_mask[il][0]) |
                    (mask[1] & ~g.lru.resident_mask[il][1]) |
                    (mask[2] & ~g.lru.resident_mask[il][2]) |
                    (mask[3] & ~g.lru.resident_mask[il][3]);
                used_fast_route = true;
                if (miss_res == 0 && (!is_decode || miss_pinned == 0 || g.lru.n_pinned >= g.lru.max_pinned)) {
                    fills.clear();
                    g.lru.stamp_mask(il, mask);
                    n_hit = (size_t) (__builtin_popcountll(mask[0]) +
                                      __builtin_popcountll(mask[1]) +
                                      __builtin_popcountll(mask[2]) +
                                      __builtin_popcountll(mask[3]));
                } else {
                    g.lru.uniq.clear();
                    for (int w = 0; w < 4; ++w) {
                        uint64_t bits = mask[w];
                        while (bits != 0) {
                            const int b = __builtin_ctzll(bits);
                            g.lru.uniq.push_back(w * 64 + b);
                            bits &= (bits - 1);
                        }
                    }
                    if (!g.lru.plan_from_uniq(il, n_sel, fills, n_hit, true)) {
                        GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
                    }
                }
            }
        }

        if (!used_fast_route && !prefetched) {
            ids.resize(n_sel);
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            ++route_readbacks;
            if (s11_event_sync) {
                // S11 bank readers and the ID copy use this backend's same stream.
                // Wait through the copy before mutating any bank slot or host map.
                auto * event = devices[g.id].route_ready.get();
                ggml_backend_event_record(event, backend);
                ggml_backend_event_synchronize(event);
            } else {
                ggml_backend_synchronize(backend);
            }
            ++route_syncs;
            if (s11_route_trace.is_open()) {
                if (s11_traced_ids + ids.size() > 2000000) {
                    throw std::runtime_error("S11 route trace exceeds bounded diagnostic workload");
                }
                s11_traced_ids += ids.size();
                s11_route_trace << il << ',' << ids.size();
                for (int32_t id : ids) {
                    s11_route_trace << ',' << id;
                }
                s11_route_trace << '\n';
                if (!s11_route_trace) {
                    throw std::runtime_error("S11 route trace write failed");
                }
            }

            static const bool trace_misses = [] {
                const char * value = std::getenv("S11_ROUTE_TRACE_MISSES");
                if (!value || std::strcmp(value, "0") == 0) { return false; }
                if (std::strcmp(value, "1") == 0) { return true; }
                GGML_ABORT("Invalid S11_ROUTE_TRACE_MISSES: %s", value);
            }();
            std::array<uint64_t, 4> resident_before{};
            if (trace_misses) {
                GGML_ASSERT(s11_route_trace.is_open() && il >= 0 && il < 64 && g.lru.n_expert <= 256 &&
                            "S11_ROUTE_TRACE_MISSES requires a bounded S11_ROUTE_TRACE");
                resident_before = g.lru.resident_mask[il];
            }
            if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, !cpu_miss)) {
                GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
            }
            if (trace_misses) {
                GGML_ASSERT(s11_route_trace.is_open() && "S11_ROUTE_TRACE_MISSES requires S11_ROUTE_TRACE");
                std::string misses;
                for (const auto & fill : fills) {
                    misses += std::to_string(fill.expert) + ",";
                }
                LLAMA_LOG_INFO("s11_route_misses layer=%d n=%zu resident=%016llx:%016llx:%016llx:%016llx ids=%s\n",
                    il, n_sel, (unsigned long long) resident_before[0], (unsigned long long) resident_before[1],
                    (unsigned long long) resident_before[2], (unsigned long long) resident_before[3], misses.c_str());
            }

            if (is_s11_fast && known_ubatches.size() < 65536) {
                auto & rec = known_ubatches[current_ubatch_hash];
                if (rec.valid_layers == 0) {
                    rec.n_ids = n_sel;
                }
                if (rec.n_ids == n_sel) {
                    rec.masks[il] = {0ULL, 0ULL, 0ULL, 0ULL};
                    for (int32_t e : g.lru.uniq) {
                        rec.masks[il][e >> 6] |= (1ULL << (e & 63));
                    }
                    rec.valid_layers |= (1ULL << il);
                }
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = n_sel <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        if (!is_s11_fast || last_dst_ptr[il] != dst->data || uploaded_gen[il] != g.lru.dirty_gen[il]) {
            if (is_s11_fast && ggml_nbytes(src) <= sizeof(slot_map_ring[0])) {
                int32_t * ring_base = buf_stage ? (int32_t *) ggml_backend_buffer_get_base(buf_stage.get()) : slot_map_ring[0].data();
                int32_t * stage = ring_base + ((slot_map_ring_idx++ & 255) * 256);
                std::memcpy(stage, src->data, ggml_nbytes(src));
                ggml_backend_tensor_set_async(backend, dst, stage, 0, ggml_nbytes(src));
            } else {
                ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
            }
            g.lru.dirty_layers[il] = false;
            last_dst_ptr[il] = dst->data;
            uploaded_gen[il] = g.lru.dirty_gen[il];
        }

        if (pf_rec != nullptr && il < pf_mtp_il) {
            if (prefetched) {
                pf_plan_window(il);
            } else if (pf_stopped && il == pf_next) {
                // resume strictly after this in-stream plan so the plans stay in layer order, the guard keeps the slots of this layer
                pf_stopped = false;
                pf_next    = il + 1;
                pf_plan_window(il);
            }
        }

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (prefetch_layers > 0 || fallback_layers > 0) {
            uint64_t skips = 0;
            for (const group & g : groups) {
                skips += g.lru.stamp_skips;
            }
            LLAMA_LOG_INFO("llama_moe_cache: prefetch: %llu layers ahead, %.2f MiB on the copy stream, %llu layers in-stream, %llu guarded victims skipped\n",
                (unsigned long long) prefetch_layers, prefetch_bytes/1024.0/1024.0, (unsigned long long) fallback_layers, (unsigned long long) skips);
        }
        if (head_planned_layers > 0 || head_mispredict > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: head prefetch: %llu layers, %.2f MiB, %llu adopted, %llu mispredicted, %llu guarded victims skipped\n",
                (unsigned long long) head_planned_layers, head_bytes/1024.0/1024.0, (unsigned long long) head_adopted,
                (unsigned long long) head_mispredict, (unsigned long long) head_stamp_skips);
        }
        if (pfh_armed > 0) {
            uint64_t pskips = 0;
            for (const group & g : groups) {
                pskips += g.lru.pinned_skips;
            }
            LLAMA_LOG_INFO("llama_moe_cache: prefill head prefetch (G=%d): %llu armed, %llu mid recorded, %llu planned, %llu layers, %.2f MiB, %llu adopted, %llu mispredicted, %llu dry-run failures, %llu wrap skips, %llu guarded victims skipped, %llu pinned skips\n",
                (int) hd_pf_G, (unsigned long long) pfh_armed, (unsigned long long) pfh_mid_recorded, (unsigned long long) pfh_planned,
                (unsigned long long) pfh_planned_layers, pfh_bytes/1024.0/1024.0, (unsigned long long) pfh_adopted,
                (unsigned long long) pfh_mispredicted, (unsigned long long) pfh_dryrun_fail, (unsigned long long) pfh_wrap_skips,
                (unsigned long long) pfh_stamp_skips, (unsigned long long) pskips);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

void llama_moe_cache::begin_ubatch(const llama_ubatch & ubatch, bool graph_reused) {
    pimpl->begin_ubatch(ubatch, graph_reused);
}

void llama_moe_cache::end_ubatch() {
    pimpl->end_ubatch();
}

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }

    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}

std::array<uint64_t, 3> llama_moe_cache::s11_statistics() const {
    return {pimpl->stats_small.bytes + pimpl->stats_large.bytes, pimpl->route_readbacks, pimpl->route_syncs};
}

std::array<uint64_t, 9> llama_moe_cache::prefetch_statistics() const {
    uint64_t skips = 0;
    for (const auto & g : pimpl->groups) {
        skips += g.lru.stamp_skips;
    }
    return {pimpl->prefetch_layers, pimpl->prefetch_bytes, pimpl->fallback_layers, skips,
            pimpl->head_planned_layers, pimpl->head_bytes, pimpl->head_adopted, pimpl->head_mispredict, pimpl->head_stamp_skips};
}
