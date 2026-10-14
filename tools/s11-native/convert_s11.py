#!/usr/bin/env python3
"""Lossless S11 -> private GGUF v1, with independent source-hash round trips.

K128 blocks: W2/W2_BF16 = 32 code bytes + FP32 multiply scale;
W4 = 64 signed-nibble bytes + FP32 multiply scale;
I8 = 128 signed bytes + FP32 reciprocal scale (restore by division).
W2 stores qw+1, four consecutive K lanes per byte, low bits first.
Protected BF16/F32 tensors stay raw; the graph, not this exporter, applies
zero-centered norm +1 and -exp(A_log).

Usage:
  python tools/s11-native/convert_s11.py CHECKPOINT --output MODEL.gguf \
      --artifacts OUTPUT_DIR
  python tools/s11-native/convert_s11.py CHECKPOINT --output MODEL.gguf \
      --artifacts OUTPUT_DIR --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
AFFINITY = sorted(os.sched_getaffinity(0))[:8]
os.sched_setaffinity(0, AFFINITY)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT))

import numpy as np
import gguf
from conversion.qwen import Qwen3_5MoeTextModel

T = gguf.GGMLQuantizationType
S11_TYPES = {T.S11_W2, T.S11_W2_BF16, T.S11_W4, T.S11_I8}
DTYPES = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4"),
          "I32": np.dtype("<i4"), "U8": np.dtype("u1"), "I8": np.dtype("i1")}
MANIFEST_DTYPES = {"BF16": "torch.bfloat16", "F32": "torch.float32",
                   "I32": "torch.int32", "U8": "torch.uint8", "I8": "torch.int8"}
TILE_ROWS = 512
RSS_LIMIT = 64 * 1024 ** 3
LAYOUT = "s11_k128_contiguous_codes_fp32_scale_v1"
BACKEND = "s11_source_math_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(), object_pairs_hook=no_duplicates)


def rss_guard():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    require(peak <= RSS_LIMIT, f"RSS limit exceeded: {peak} > {RSS_LIMIT}")
    return peak


def sha_bytes(array):
    array = np.ascontiguousarray(array)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 16 * 1024 ** 2):
        digest.update(view[start:start + 16 * 1024 ** 2])
        rss_guard()
    return digest.hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 ** 2):
            digest.update(block)
            rss_guard()
    return digest.hexdigest()


class Source:
    def __init__(self, path):
        self.path = path
        self.config = load_json(path / "config.json")
        self.manifest = load_json(path / "bitnet_moe_manifest.json")
        index = load_json(path / "model.safetensors.index.json")
        self.index = index["weight_map"]
        m, c = self.manifest, self.config
        require(m.get("complete") is True and m.get("model_content_validated") is True,
                "Source manifest is not complete and content-validated")
        require(m["schema_version"] == 1, "Unsupported source manifest schema")
        require(c["architectures"] == ["Qwen3_5MoeForCausalLM"],
                "Unsupported model architecture")
        require(c.get("mtp_num_hidden_layers", 0) == 0, "MTP is outside the S11 container contract")
        require(len(c["layer_types"]) == c["num_hidden_layers"] and
                set(c["layer_types"]) <= {"linear_attention", "full_attention"},
                "Malformed layer types")
        require(c["quantization_config"]["moe"] == m["moe_format"],
                "Config/manifest MoE recipe mismatch")
        require(c["quantization_config"]["qat_projections"] == m["qat_projections"],
                "Config/manifest projection recipe mismatch")
        recipe = m["moe_format"]
        for key, value in {"schema_version": 2, "weight_group_size": 128,
                           "activation_group_size": 128,
                           "weight_layout": "ternary_2bit_k16_byteplane_v1",
                           "weight_scale": "effective_bf16_as_fp32",
                           "router_dtype": "float32",
                           "router_normalization": "topk_then_softmax",
                           "probability_placement": "before_fc2_activation_quant",
                           "activation_rounding": "fp32_swiglu_probability_then_bf16",
                           "embedding_head_math": "triton_fp32_division_v1"}.items():
            require(recipe.get(key) == value, f"Unsupported S11 MoE {key}")
        for key, value in {"schema_version": 1, "weight_group_size": 128,
                           "scale_dtype": "float32", "scale_semantics": "multiply",
                           "ternary_layout": "ternary_2bit_k16_byteplane_v1",
                           "int4_layout": "sint4_k2_low_nibble_first_v1",
                           "activation": "dynamic_a128_bf16_fake_quant",
                           "restored_weight_dtype": "bfloat16"}.items():
            require(m["qat_projections"].get(key) == value,
                    f"Unsupported S11 projection {key}")
        require(m["qat_projections"]["formats"] == {
            "shared_expert": "ternary_packed2", "full_attention": "signed_int4_packed4",
            "gdn_projections": "signed_int4_packed4"}, "Unsupported projection format assignment")
        for key in ("int8_embedding", "int8_lm_head"):
            require(c["quantization_config"][key] == {"enabled": True, "block_size": 128},
                    f"Unsupported {key} recipe")
        require(set(self.index) == set(m["tensors"]), "Index/manifest tensor set mismatch")
        require(len(self.index) == m["output_tensor_count"], "Wrong source tensor count")
        shards = set(self.index.values())
        require(all(Path(f).name == f and f.endswith(".safetensors") for f in shards),
                "Index contains invalid shard paths")
        require(shards == {f.name for f in path.glob("*.safetensors")},
                "Unindexed or missing shard (including duplicate shard)")
        self.headers = {}
        self.locations = {}
        total = 0
        for shard in sorted(shards):
            file = path / shard
            with file.open("rb") as handle:
                n = int.from_bytes(handle.read(8), "little")
                require(2 <= n <= 16 * 1024 ** 2, f"Invalid safetensors header: {shard}")
                header = json.loads(handle.read(n), object_pairs_hook=no_duplicates)
            end = 0
            for name, info in sorted(((k, v) for k, v in header.items()
                                      if k != "__metadata__"),
                                     key=lambda item: item[1]["data_offsets"][0]):
                require(name in self.index and self.index[name] == shard,
                        f"Duplicate/unindexed/wrong-shard tensor: {name}")
                require(name not in self.locations, f"Duplicate tensor: {name}")
                require(info["dtype"] in DTYPES, f"Unsupported source dtype: {name}")
                require(all(isinstance(d, int) and d > 0 for d in info["shape"]),
                        f"Invalid shape: {name}")
                lo, hi = info["data_offsets"]
                expected = int(np.prod(info["shape"])) * DTYPES[info["dtype"]].itemsize
                require(lo == end and hi - lo == expected, f"Malformed tensor offsets: {name}")
                record = m["tensors"][name]
                require(info["shape"] == record["shape"] and
                        MANIFEST_DTYPES[info["dtype"]] == record["dtype"],
                        f"Manifest shape/dtype mismatch: {name}")
                self.locations[name] = (shard, 8 + n + lo, info)
                end = hi
            require(file.stat().st_size == 8 + n + end, f"Truncated/trailing shard: {shard}")
            self.headers[shard] = {"bytes": file.stat().st_size, "header_bytes": 8 + n}
            total += end
        require(set(self.locations) == set(self.index), "Missing source tensors")
        require(total == index["metadata"]["total_size"] == m["output_weight_bytes"],
                "Source byte count mismatch")
        for file, digest in m["assets"].items():
            require(Path(file).name == file, "Invalid asset path")
            require(sha_file(path / file) == digest, f"Source asset hash mismatch: {file}")

    def array(self, name):
        shard, offset, info = self.locations[name]
        return np.memmap(self.path / shard, mode="r", offset=offset,
                         dtype=DTYPES[info["dtype"]], shape=tuple(info["shape"]))

    def validate_hashes(self):
        for i, name in enumerate(sorted(self.locations)):
            array = self.array(name)
            require(sha_bytes(array) == self.manifest["tensors"][name]["sha256"],
                    f"Source tensor hash mismatch: {name}")
            if array.dtype == np.dtype("<f4"):
                flat = array.reshape(-1)
                for start in range(0, flat.size, 1024 * 1024):
                    chunk = flat[start:start + 1024 * 1024]
                    require(np.isfinite(chunk).all(), f"Nonfinite source tensor: {name}")
            if i % 100 == 0:
                print(f"Source hashes: {i}/{len(self.locations)}", flush=True)


def head_perm(c, dim):
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    require(nv % nk == 0, "Value heads not divisible by key heads")
    return np.arange(nv * dim).reshape(nk, nv // nk, dim).transpose(1, 0, 2).reshape(-1)


def expected_logical_shapes(c):
    h, f, e = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"]
    sf = c["shared_expert_intermediate_size"]
    result = {"model.embed_tokens.weight": [c["vocab_size"], h],
              "lm_head.weight": [c["vocab_size"], h], "model.norm.weight": [h]}
    for layer, kind in enumerate(c["layer_types"]):
        prefix = f"model.layers.{layer}."
        local = {"input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
                 "mlp.experts.gate_up_proj": [e, 2 * f, h],
                 "mlp.experts.down_proj": [e, h, f], "mlp.gate.weight": [e, h],
                 "mlp.shared_expert.gate_proj.weight": [sf, h],
                 "mlp.shared_expert.up_proj.weight": [sf, h],
                 "mlp.shared_expert.down_proj.weight": [h, sf],
                 "mlp.shared_expert_gate.weight": [1, h]}
        if kind == "linear_attention":
            nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
            kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
            qkv = 2 * nk * kd + nv * vd
            local.update({"linear_attn.in_proj_qkv.weight": [qkv, h],
                          "linear_attn.in_proj_z.weight": [nv * vd, h],
                          "linear_attn.in_proj_a.weight": [nv, h],
                          "linear_attn.in_proj_b.weight": [nv, h],
                          "linear_attn.out_proj.weight": [h, nv * vd],
                          "linear_attn.conv1d.weight": [qkv, 1, c["linear_conv_kernel_dim"]],
                          "linear_attn.norm.weight": [vd],
                          "linear_attn.A_log": [nv], "linear_attn.dt_bias": [nv]})
        else:
            hd = c["head_dim"]
            local.update({"self_attn.q_proj.weight": [c["num_attention_heads"] * hd * 2, h],
                          "self_attn.k_proj.weight": [c["num_key_value_heads"] * hd, h],
                          "self_attn.v_proj.weight": [c["num_key_value_heads"] * hd, h],
                          "self_attn.o_proj.weight": [h, c["num_attention_heads"] * hd],
                          "self_attn.q_norm.weight": [hd], "self_attn.k_norm.weight": [hd]})
        result.update({prefix + name: shape for name, shape in local.items()})
    return result


def permutation(desc, c):
    transform = desc["transform"]
    vd = c["linear_value_head_dim"]
    if transform in ("v_rows", "v_columns"):
        return head_perm(c, vd)
    if transform == "head_rows":
        return head_perm(c, 1)
    if transform == "qkv_v_rows":
        qk = c["linear_num_key_heads"] * c["linear_key_head_dim"] * 2
        return np.concatenate((np.arange(qk), qk + head_perm(c, vd)))
    return None


def make_descriptors(source):
    c = source.config
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35MOE, c["num_hidden_layers"])
    result, consumed = [], set()
    expected_shapes = expected_logical_shapes(c)
    weights = {name for name in source.locations if not name.endswith(("_scale", ".weight_scale"))}
    require(weights == set(expected_shapes), "Missing/extra model tensor relative to config")
    for name in sorted(source.locations):
        if name.endswith(("_scale", ".weight_scale")):
            continue
        info = source.locations[name][2]
        shape = info["shape"].copy()
        dtype = info["dtype"]
        if name in ("lm_head.weight", "model.embed_tokens.weight"):
            expected_dtype = ("I8",)
        elif ".mlp.experts." in name or ".mlp.shared_expert." in name:
            expected_dtype = ("I32",)
        elif any(f".{projection}.weight" in name for projection in (
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "q_proj", "k_proj", "v_proj", "o_proj")):
            expected_dtype = ("U8",)
        else:
            expected_dtype = ("BF16", "F32")
        require(dtype in expected_dtype, f"Wrong module source dtype: {name}")
        scale = None
        if dtype == "I32":
            scale = name + "_scale"
            shape[-1] *= 16
            qtype = T.S11_W2 if ".mlp.experts." in name else T.S11_W2_BF16
        elif dtype == "U8":
            scale = name + "_scale"
            shape[-1] *= 2
            qtype = T.S11_W4
        elif dtype == "I8":
            scale = name + "_scale"
            qtype = T.S11_I8
        else:
            require(dtype in ("BF16", "F32"), f"Unsupported protected dtype: {name}")
            qtype = T.BF16 if dtype == "BF16" else T.F32
        require(shape == expected_shapes[name], f"Model logical shape mismatch: {name}")
        if scale:
            require(scale in source.locations, f"Missing scale: {name}")
            require(shape[-1] % 128 == 0, f"Non-K128 weight: {name}")
            sinfo = source.locations[scale][2]
            require(sinfo["dtype"] == "F32" and
                    sinfo["shape"] == shape[:-1] + [shape[-1] // 128],
                    f"Wrong scale dtype/shape: {scale}")
            record = source.manifest["tensors"][name]
            require(record.get("logical_shape", shape) == shape,
                    f"Logical shape mismatch: {name}")
            if dtype == "I32":
                require(record.get("weight_layout") == "ternary_2bit_k16_byteplane_v1",
                        f"Wrong W2 layout: {name}")
            if dtype == "U8":
                require(record.get("weight_layout") == "sint4_k2_low_nibble_first_v1",
                        f"Wrong W4 layout: {name}")
            consumed.add(scale)
        consumed.add(name)
        transform = "identity"
        if ".linear_attn." in name:
            if ".in_proj_qkv." in name or ".conv1d." in name:
                transform = "qkv_v_rows"
            elif ".in_proj_z." in name:
                transform = "v_rows"
            elif any(s in name for s in (".in_proj_a.", ".in_proj_b.", ".A_log", ".dt_bias")):
                transform = "head_rows"
            elif ".out_proj." in name:
                transform = "v_columns"
        if ".conv1d." in name:
            require(shape[1] == 1, f"Unexpected conv shape: {name}")
            shape.pop(1)
        mapped = name
        if name.endswith(".dt_bias"):
            mapped = name.removesuffix(".dt_bias") + ".dt_proj.bias"
        if ".mlp.experts." in name and not name.endswith(".weight"):
            mapped += ".weight"
        parts = [(mapped, None, shape)]
        if name.endswith(".mlp.experts.gate_up_proj"):
            require(len(shape) == 3 and shape[1] == 2 * c["moe_intermediate_size"],
                    f"Malformed gate/up partition: {name}")
            f = shape[1] // 2
            base = name.removesuffix("gate_up_proj")
            parts = [(base + "gate_proj.weight", [0, f], [shape[0], f, shape[2]]),
                     (base + "up_proj.weight", [f, 2 * f], [shape[0], f, shape[2]])]
        for mapped, partition, native_shape in parts:
            native = names.get_name(mapped, try_suffixes=(".weight", ".bias"))
            require(native is not None, f"Unknown native tensor mapping: {mapped}")
            result.append({"name": native, "source": name, "scale_source": scale,
                           "source_shard": source.index[name], "shape": native_shape,
                           "type": int(qtype), "transform": transform,
                           "partition_rows": partition,
                           "scale_semantics": "divide" if qtype == T.S11_I8 else
                                              ("multiply" if scale else "protected"),
                           "protected_raw": scale is None})
    require(consumed == set(source.locations), "Unconsumed source tensors/scales")
    require(len({d["name"] for d in result}) == len(result), "Duplicate native tensor name")
    return result


def native_size(desc):
    block, size = gguf.GGML_QUANT_SIZES[T(desc["type"])]
    return int(np.prod(desc["shape"])) // block * size


def row_indices(desc, source):
    shape = source.locations[desc["source"]][2]["shape"]
    if desc["partition_rows"]:
        lo, hi = desc["partition_rows"]
        return np.arange(int(np.prod(shape[:-1]))).reshape(shape[:-1])[:, lo:hi].reshape(-1)
    p = permutation(desc, source.config)
    if desc["transform"] != "v_columns" and p is not None:
        return p
    if len(shape) == 1:
        return np.arange(shape[0])
    return np.arange(int(np.prod(desc["shape"][:-1])))


def pack_w2(words):
    """Translate signed source byteplanes into unsigned contiguous K codes."""
    shifts = (np.arange(16, dtype=np.uint32) % 4) * 8 + (np.arange(16, dtype=np.uint32) // 4) * 2
    code = (words.astype(np.uint32)[..., None] >> shifts) & 3
    require(not np.any(code == 2), "Reserved source W2 code 10")
    code = np.where(code == 3, 0, code + 1).astype(np.uint8).reshape(words.shape[0], -1, 4)
    return code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4) | (code[..., 3] << 6)


def native_chunks(desc, source):
    weight = source.array(desc["source"])
    scale = source.array(desc["scale_source"]) if desc["scale_source"] else None
    qtype = T(desc["type"])
    rows = row_indices(desc, source)
    if qtype not in S11_TYPES:
        # 1D head parameters are permuted as scalar rows.
        flat = weight.reshape((-1, 1) if weight.ndim == 1 else
                              (-1, int(np.prod(weight.shape[1:]))))
        for start in range(0, len(rows), TILE_ROWS):
            data = np.ascontiguousarray(flat[rows[start:start + TILE_ROWS]])
            if data.dtype == np.dtype("<f4"):
                require(np.isfinite(data).all(), f"Nonfinite protected tensor: {desc['source']}")
            elif qtype == T.BF16:
                require(not np.any((data & 0x7f80) == 0x7f80),
                        f"Nonfinite BF16 tensor: {desc['source']}")
            rss_guard()
            yield data.view(np.uint8).reshape(-1)
        return
    weight = weight.reshape(-1, weight.shape[-1])
    scale = scale.reshape(-1, scale.shape[-1])
    size = gguf.GGML_QUANT_SIZES[qtype][1]
    codebytes = size - 4
    colperm = permutation(desc, source.config) if desc["transform"] == "v_columns" else None
    if colperm is not None:
        groups = colperm.reshape(-1, 128)
        require(np.array_equal(groups, groups[:, :1] + np.arange(128)),
                "Column permutation splits K128 scales")
        groupperm = groups[:, 0] // 128
    for start in range(0, len(rows), TILE_ROWS):
        indices = rows[start:start + TILE_ROWS]
        w, s = weight[indices], scale[indices]
        require(np.isfinite(s).all() and np.all(s > 0),
                f"Nonpositive/nonfinite scale: {desc['scale_source']}")
        if qtype in (T.S11_W2, T.S11_W2_BF16):
            codes = pack_w2(w)
        else:
            codes = np.ascontiguousarray(w).view(np.uint8)
            if qtype == T.S11_W4:
                require(not np.any((codes & 15) == 8) and not np.any((codes >> 4) == 8),
                        f"Reserved source W4 code -8: {desc['source']}")
        codes = codes.reshape(len(indices), -1, codebytes)
        if colperm is not None:
            codes, s = codes[:, groupperm], s[:, groupperm]
        out = np.empty((*s.shape, size), dtype=np.uint8)
        out[..., :codebytes] = codes
        out[..., codebytes:] = np.ascontiguousarray(s).view(np.uint8).reshape(*s.shape, 4)
        rss_guard()
        yield out.reshape(-1)


def add_metadata(writer, source, descriptors, manifest):
    # Reuse upstream metadata/tokenizer handling, but never its tensor transforms.
    class MetadataModel(Qwen3_5MoeTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35MOE
        no_mtp = True

    model = MetadataModel(source.path, gguf.LlamaFileType.MOSTLY_BF16,
                          Path("unused.gguf"), hparams=source.config)
    model.gguf_writer = writer
    model.set_gguf_parameters()
    model.set_vocab()
    writer.add_name("S11 iter2000 native lossless")
    writer.add_uint32("s11.schema_version", 1)
    writer.add_uint32("s11.layout_version", 1)
    writer.add_string("s11.layout", LAYOUT)
    writer.add_string("s11.numeric_backend", BACKEND)
    writer.add_uint32("s11.weight_group_size", 128)
    writer.add_uint32("s11.activation_group_size", 128)
    writer.add_string("s11.scale_dtype", "float32")
    writer.add_string("s11.protected_dtypes", "bfloat16,float32")
    writer.add_float32("s11.norm_offset", 1.0)
    writer.add_bool("s11.ssm_a_is_log", True)
    writer.add_string("s11.norm_offset_exclusion", "ssm_norm.weight")
    writer.add_string("s11.head_order", "gdn_v_grouped_to_tiled_v1")
    writer.add_string("s11.router_normalization", "topk_then_softmax")
    writer.add_string("s11.probability_placement", "before_fc2_activation_quant")
    writer.add_string("s11.activation_rounding", "fp32_swiglu_probability_then_bf16")
    writer.add_string("s11.projection_activation", "dynamic_a128_bf16_fake_quant")
    writer.add_string("s11.restored_weight_dtype", "bfloat16")
    writer.add_string("s11.embedding_head_math", "triton_fp32_division_v1")
    writer.add_string("s11.source_manifest_sha256", manifest["source_manifest_sha256"])
    writer.add_string("s11.source_config_sha256", manifest["source_config_sha256"])
    writer.add_string("s11.source_index_sha256", manifest["source_index_sha256"])
    writer.add_uint32("s11.tensor_count", len(descriptors))
    writer.add_uint32("s11.source_tensor_count", len(source.locations))
    writer.add_array("s11.tensor_names", [d["name"] for d in descriptors])
    writer.add_array("s11.tensor_types", [d["type"] for d in descriptors])
    writer.add_array("s11.tensor_shapes", [v for d in descriptors
                     for v in list(reversed(d["shape"])) + [1] * (4 - len(d["shape"]))])
    writer.add_string("s11.tensor_manifest", json.dumps(descriptors, separators=(",", ":")))
    writer.add_string("s11.source_hashes", json.dumps(manifest["source_tensors"], separators=(",", ":")))


def unpack_native_codes(raw, qtype):
    """Independent inverse: scatter each native lane back into source bitplanes."""
    if qtype in (T.S11_W2, T.S11_W2_BF16):
        packed = raw.reshape(raw.shape[0], -1)
        codes = np.empty((packed.shape[0], packed.shape[1] * 4), np.uint32)
        for lane in range(4):
            codes[:, lane::4] = (packed >> (2 * lane)) & 3
        require(not np.any(codes == 3), "Reserved native W2 code 11")
        signedbits = np.where(codes == 0, 3, codes - 1)
        words = np.zeros((codes.shape[0], codes.shape[1] // 16), np.uint32)
        for lane in range(16):
            shift = (lane % 4) * 8 + (lane // 4) * 2
            words |= signedbits[:, lane::16] << shift
        return words.view(np.int32)
    if qtype == T.S11_W4:
        require(not np.any((raw & 15) == 8) and not np.any((raw >> 4) == 8),
                "Reserved native W4 code -8")
    return raw.reshape(raw.shape[0], -1).view(np.int8 if qtype == T.S11_I8 else np.uint8)


def field_value(reader, key):
    field = reader.get_field(key)
    require(field is not None, f"Missing metadata: {key}")
    return field.contents()


def validate_container(reader, descriptors):
    expected = {"s11.schema_version": 1, "s11.layout_version": 1,
                "s11.layout": LAYOUT, "s11.weight_group_size": 128,
                "s11.activation_group_size": 128, "s11.scale_dtype": "float32",
                "s11.numeric_backend": BACKEND, "s11.norm_offset": 1.0,
                "s11.ssm_a_is_log": True, "general.architecture": "qwen35moe",
                "s11.tensor_count": len(descriptors)}
    for key, value in expected.items():
        require(field_value(reader, key) == value, f"Unsupported container metadata: {key}")
    require(json.loads(field_value(reader, "s11.tensor_manifest")) == descriptors,
            "Embedded source/native mapping mismatch")
    require(field_value(reader, "s11.tensor_names") == [d["name"] for d in descriptors] and
            field_value(reader, "s11.tensor_types") == [d["type"] for d in descriptors] and
            field_value(reader, "s11.tensor_shapes") == [v for d in descriptors
                for v in list(reversed(d["shape"])) + [1] * (4 - len(d["shape"]))],
            "Loader tensor schema arrays mismatch")
    require(len(reader.tensors) == len(descriptors), "Missing/extra native tensors")
    require(len({t.name for t in reader.tensors}) == len(descriptors), "Duplicate native tensors")
    for tensor, desc in zip(reader.tensors, descriptors):
        require(tensor.name == desc["name"] and int(tensor.tensor_type) == desc["type"] and
                tensor.shape.tolist() == list(reversed(desc["shape"])) and
                tensor.n_bytes == native_size(desc), f"Native shape/type/size mismatch: {desc['name']}")


def verify(output, source, descriptors, manifest):
    reader = gguf.GGUFReader(output)
    validate_container(reader, descriptors)
    for key in ("source_manifest_sha256", "source_config_sha256", "source_index_sha256"):
        require(field_value(reader, "s11." + key) == manifest[key], f"Wrong source identity: {key}")
    require(json.loads(field_value(reader, "s11.source_hashes")) == manifest["source_tensors"],
            "Embedded source hashes mismatch")
    tensors = {t.name: t for t in reader.tensors}
    grouped = {}
    for desc in descriptors:
        grouped.setdefault(desc["source"], []).append(desc)
    checked, byte_count = [], 0
    for source_name in sorted(grouped):
        info = source.locations[source_name][2]
        original_shape = info["shape"]
        recovered = np.empty(original_shape, dtype=DTYPES[info["dtype"]])
        scale_name = grouped[source_name][0]["scale_source"]
        scales = np.empty(source.locations[scale_name][2]["shape"], dtype=np.float32) if scale_name else None
        for desc in grouped[source_name]:
            qtype = T(desc["type"])
            tensor = tensors[desc["name"]]
            raw = tensor.data.view(np.uint8).reshape(-1)
            rows = row_indices(desc, source)
            if qtype not in S11_TYPES:
                target = recovered.reshape((-1, 1) if len(original_shape) == 1 else
                                           (-1, int(np.prod(original_shape[1:]))))
                data = raw.view(DTYPES[info["dtype"]]).reshape(len(rows), -1)
                target[rows] = data
                if qtype == T.F32:
                    require(np.isfinite(data).all(), f"Nonfinite native F32: {desc['name']}")
                else:
                    require(not np.any((data & 0x7f80) == 0x7f80),
                            f"Nonfinite native BF16: {desc['name']}")
            else:
                size = gguf.GGML_QUANT_SIZES[qtype][1]
                ng = desc["shape"][-1] // 128
                blocks = raw.reshape(len(rows), ng, size)
                target = recovered.reshape(-1, original_shape[-1])
                starget = scales.reshape(-1, scales.shape[-1])
                cp = permutation(desc, source.config) if desc["transform"] == "v_columns" else None
                inv = np.argsort(cp.reshape(-1, 128)[:, 0] // 128) if cp is not None else None
                for start in range(0, len(rows), TILE_ROWS):
                    tile = blocks[start:start + TILE_ROWS]
                    codes = np.ascontiguousarray(tile[..., :-4])
                    s = np.ascontiguousarray(tile[..., -4:]).view("<f4").reshape(tile.shape[:2])
                    require(np.isfinite(s).all() and np.all(s > 0),
                            f"Nonpositive/nonfinite native scale: {desc['name']}")
                    if inv is not None:
                        codes, s = codes[:, inv], s[:, inv]
                    target[rows[start:start + TILE_ROWS]] = unpack_native_codes(codes, qtype)
                    starget[rows[start:start + TILE_ROWS]] = s
                    rss_guard()
            byte_count += tensor.n_bytes
        for name, array in ((source_name, recovered), (scale_name, scales)):
            if name is None:
                continue
            digest = sha_bytes(array)
            require(digest == source.manifest["tensors"][name]["sha256"],
                    f"Independent round-trip hash mismatch: {name}")
            checked.append({"name": name, "sha256": digest, "bytes": array.nbytes})
        del recovered, scales
        if len(checked) % 100 < 3:
            print(f"Round trips: {len(checked)}/{len(source.locations)}", flush=True)
    require(len(checked) == len(source.locations), "Incomplete round-trip coverage")
    return {"complete": True, "source_tensor_count": len(checked),
            "native_tensor_count": len(descriptors), "native_tensor_bytes": byte_count,
            "roundtrip": "bitwise_sha256_source_packed_codes_scales_and_protected_values",
            "checked_tensors": checked, "peak_rss_bytes": rss_guard()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    affinity = AFFINITY
    # A hard virtual-memory cap also bounds single-process resident memory.
    resource.setrlimit(resource.RLIMIT_AS, (RSS_LIMIT, RSS_LIMIT))
    source = Source(args.checkpoint.resolve())
    output = args.output.resolve()
    require(source.path not in output.parents, "Output must not modify the source checkpoint")
    artifact_path = args.artifacts.resolve()
    require(artifact_path != source.path and source.path not in artifact_path.parents,
            "Artifacts must not modify the source checkpoint")
    artifact_path.mkdir(parents=True, exist_ok=True)
    descriptors = make_descriptors(source)
    manifest = {"schema_version": 1, "layout": LAYOUT, "numeric_backend": BACKEND,
                "source_checkpoint": str(source.path),
                "source_manifest_sha256": sha_file(source.path / "bitnet_moe_manifest.json"),
                "source_config_sha256": sha_file(source.path / "config.json"),
                "source_index_sha256": sha_file(source.path / "model.safetensors.index.json"),
                "source_tensors": source.manifest["tensors"],
                "source_shards": source.headers, "source_to_native": descriptors,
                "head_permutation": {"kind": "gdn_v_grouped_to_tiled_v1",
                                     "value_head_native_to_source": head_perm(source.config, 1).tolist(),
                                     "inverse_value_head": np.argsort(head_perm(source.config, 1)).tolist(),
                                     "key_head_count": source.config["linear_num_key_heads"],
                                     "value_head_dim": source.config["linear_value_head_dim"]},
                "raw_protected_values": True, "norm_offset": 1.0, "ssm_a_is_log": True}
    source.validate_hashes()
    part = output.with_suffix(output.suffix + ".partial")
    if not args.verify_only:
        require(not output.exists() and not part.exists(), f"Refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(part, "qwen35moe")
        add_metadata(writer, source, descriptors, manifest)
        for desc in descriptors:
            writer.add_tensor_info(desc["name"], desc["shape"], np.dtype("int32"),
                                   native_size(desc), raw_dtype=T(desc["type"]))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        handle = writer.fout[0]
        for i, desc in enumerate(descriptors):
            writer.write_padding(handle, handle.tell())
            begin = handle.tell()
            for chunk in native_chunks(desc, source):
                chunk.tofile(handle)
            require(handle.tell() - begin == native_size(desc), f"Wrong emitted size: {desc['name']}")
            writer.write_padding(handle, native_size(desc))
            if i % 50 == 0:
                print(f"Written: {i}/{len(descriptors)}", flush=True)
        writer.close()
        verification = verify(part, source, descriptors, manifest)
        part.rename(output)
    else:
        verification = verify(output, source, descriptors, manifest)
    verification.update({"output": str(output), "output_bytes": output.stat().st_size,
                         "output_sha256": sha_file(output), "elapsed_seconds": time.monotonic() - started,
                         "cpu_affinity": affinity, "rss_limit_bytes": RSS_LIMIT,
                         "peak_rss_bytes": rss_guard()})
    manifest["native_gguf"] = str(output)
    manifest["native_gguf_sha256"] = verification["output_sha256"]
    for file, contents in (("source-to-native-manifest.json", manifest),
                           ("roundtrip-validation.json", verification)):
        (args.artifacts / file).write_text(json.dumps(contents, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verification.items() if k != "checked_tensors"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, MemoryError) as error:
        print(f"S11 conversion rejected: {error}", file=sys.stderr)
        sys.exit(1)
