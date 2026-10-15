"""Recompute CB CPU controls; never execute inference or merge developer/profile samples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = load(out / "protocol.json")
    selection = load(out / "cpu-selection.json")
    candidate_folder = selection.get("formal_candidate_folder", "cpu-final")
    state_guard = selection.get("final_state_guard", "final-cpu-state-guard")
    smoke_folder = selection.get("final_smoke_folder", "cpu-launcher-smoke")
    for path, digest in protocol["protected_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Protected asset changed: {path}")
    for name, identity in protocol["source_shard_identity"].items():
        st = (Path(protocol["source_checkpoint"]) / name).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (identity["bytes"], identity["mtime_ns"], identity["inode"]):
            raise AssertionError(f"Checkpoint shard changed: {name}")
    for name, digest in protocol["c0_binary_sha256"].items():
        if hashlib.sha256((out / "c0-bin" / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"C0 changed: {name}")
    freeze_file = out / "cpu-final-freeze.json"
    if freeze_file.exists():
        freeze = load(freeze_file)
        for filename, digest in freeze["files_sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Final CPU identity changed: {filename}")
    rows, records = [], {}
    for label in ["c0-formal", "cpu-final"]:
        folder = candidate_folder if label == "cpu-final" else label
        samples = load(out / folder / "requests.json")
        guard = load(out / (folder + "-guard") / "summary.json")
        if not guard["passed"] or guard["own_gpu_limit_bytes"] != 0 or guard["rss_peak_gib"] > 64:
            raise AssertionError(f"CPU resource gate failed: {label}")
        with (out / (folder + "-guard") / "resources.csv").open() as f:
            if any(int(r["own_gpu_bytes"]) for r in csv.DictReader(f)):
                raise AssertionError(f"CPU run used GPU: {label}")
        if len(samples) != 24 or sum(not r["warmup"] for r in samples) != 20:
            raise AssertionError(f"CPU formal campaign incomplete: {label}")
        records[label] = samples
        for length, output in [(128, 32), (2048, 32), (128, 256), (2048, 256)]:
            cell = [r for r in samples if not r["warmup"] and (r["input_tokens"], r["output_tokens"]) == (length, output)]
            if len(cell) != 5 or sorted(r["repeat"] for r in cell) != [1, 2, 3, 4, 5]:
                raise AssertionError("CPU repeated workload mismatch")
            if len({tuple(r["output_ids"]) for r in cell}) != 1:
                raise AssertionError("CPU repeated output changed")
            for r in cell:
                work = r["final"]["s11_work"]
                if len(r["output_ids"]) != output or work["cpu_assignments"] != (length + output - 1) * 40 * 8:
                    raise AssertionError("CPU useful work or output was shortened")
                if any(work[k] for k in ["gpu_assignments", "graph_replays", "routed_h2d_bytes"]):
                    raise AssertionError("GPU execution in CPU campaign")
            rows.append(dict(config=label, input_tokens=length, output_tokens=output,
                             wall_p50_s=statistics.median(r["wall_s"] for r in cell),
                             ttft_p50_s=statistics.median(r["ttft_s"] for r in cell),
                             decode_p50_tps=statistics.median(r["decode_tps"] for r in cell),
                             wall_min_s=min(r["wall_s"] for r in cell),
                             wall_max_s=max(r["wall_s"] for r in cell),
                             gpu_peak_bytes=0, rss_peak_gib=guard["rss_peak_gib"]))
    if any(a["output_ids"] != b["output_ids"] for a, b in zip(records["c0-formal"], records["cpu-final"])):
        raise AssertionError("CPU optimization changed formal output IDs")
    for label in ["cpu128-reference-qualification", "cpu128-interleaved4-qualification"]:
        if not load(out / (label + ".json"))["passed"]:
            raise AssertionError(f"CPU math qualification failed: {label}")
    if candidate_folder != "cpu-final" and not load(out / "inline-math-qualification.json")["passed"]:
        raise AssertionError("Final inlined math qualification failed")
    state = (out / state_guard / "command.log").read_text()
    match = re.search(r'\{"cpu_dispatch":(\{[^\n]+\})\}', state)
    if not match:
        raise AssertionError("Actual CPU production dispatch evidence is missing")
    coverage = json.loads(match.group(1))
    if coverage["fallback"] or not coverage["tiled"]:
        raise AssertionError("Unexpected CPU S11 fallback or missing tiled execution")
    if not load(out / state_guard / "summary.json")["passed"]:
        raise AssertionError("Final CPU state failed")
    temporal = [json.loads(m.group()) for m in re.finditer(r'\{"chunk":[^\n]+\}', state)]
    if not temporal or any(r.get("relative_l2", r.get("hidden_relative_l2", 0)) != 0 for r in temporal):
        raise AssertionError("Claimed exact CPU temporal/layer identity was not observed")
    if candidate_folder != "cpu-final":
        initial = load(out / "cpu-final" / "requests.json")
        if len(initial) != 24 or sum(not r["warmup"] for r in initial) != 20:
            raise AssertionError("Initial candidate formal evidence missing")
        if any(a["output_ids"] != b["output_ids"] for a, b in zip(initial, records["c0-formal"])):
            raise AssertionError("Initial candidate outputs differ from C0")
        if not (out / "initial-candidate-bin" / "llama-server").exists():
            raise AssertionError("Initial candidate binary snapshot missing")
    smoke_path = out / smoke_folder / "requests.json"
    if smoke_path.exists():
        smoke = load(smoke_path)
        if any(r["output_ids"] != records["cpu-final"][0]["output_ids"] for r in smoke):
            raise AssertionError("Persistent CPU launcher differs from formal candidate")
    with (out / "cpu-measurements.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(passed=True, formal_samples=40, source_weight_identity=True,
                  CPU_only=True, output_ids_equal=True, useful_work_exact=True,
                  actual_dispatch=coverage, selection=selection,
                  retained_formal_samples=60 if candidate_folder != "cpu-final" else 40,
                  stable_speedup_claim=False, universal_fastest_claim=False,
                  GPU_ratio="not remeasured; historical GPU numbers are not a paired CB comparison",
                  scope="CB00-CB12 only; stop after CB12", rows=rows)
    (out / "cpu-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    cells = {(r["config"], r["input_tokens"], r["output_tokens"]): r for r in rows}
    lines = [
        "# S11-Native credible CPU baseline: CB00-CB12",
        "",
        "Completed the authorized CPU baseline scope; stop after CB12.",
        "This adapts public BitNet CPU performance methods to existing S11, not",
        "the author's original row-scaled I2_S quantization/model or a universal",
        "fastest-CPU claim. No new training, requantization, checkpoint,",
        "accuracy evaluation or old Q route was used.",
        "",
        "## Selected baseline and fixed resources",
        "",
        "CPU-only native build (`GGML_CUDA=OFF`), AVX2, OpenMP/llamafile off,",
        "inference affinity0-7, group RSS<=64GiB, zero owned GPU holders.",
        "Selected floating4-row/4-column S11 tiles, existing direct compressed",
        "integer experts/persistent8-thread pool, packed I8 full-vocabulary head,",
        "CPU selected-row embedding, original GDN/attention/protected kernels.",
        "Source scales, group128, probability position and BF16 boundaries remain.",
        "",
        "The [CPU launcher](../../../s11-native/tools/s11-native/run_cpu.sh)",
        "is separate from the GPU launcher and original B1 binary.",
        "CPU controls use their immutable snapshot and isolated measured build;",
        "the live GPU worktree was found to have concurrent unverified changes.",
        "Those changes were not reverted or mixed into the CPU measurements.",
        "Use guard_cpu.py; prefix reuse/speculation are off. Context4096/batch1,",
        "greedy/ignoreEOS, full128/2048 inputs and32/256 outputs are unchanged.",
        "Each cell has1warmup+5formal repeats; expert weight pages/cache warm up",
        "but each request resets its KV/GDN context without prefix reuse.",
        "",
        "## Same-campaign complete request results",
        "",
        "| Input/output | Frozen C0 P50(s) | Selected CPU P50(s) | Wall reduction | Selected TTFT(s) | Selected Decode(t/s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length, output in [(128, 32), (2048, 32), (128, 256), (2048, 256)]:
        a, b = cells[("c0-formal", length, output)], cells[("cpu-final", length, output)]
        gain = 100 * (1 - b["wall_p50_s"] / a["wall_p50_s"])
        lines.append(f"| {length}/{output} | {a['wall_p50_s']:.3f} | {b['wall_p50_s']:.3f} | {gain:.1f}% | {b['ttft_p50_s']:.3f} | {b['decode_p50_tps']:.3f} |")
    selected = cells[("cpu-final", 128, 256)]
    lines += [
        "",
        f"Selected group RSS peak{selected['rss_peak_gib']:.3f}GiB; GPU0.",
        "All24 C0/candidate output-ID sequences are identical; exact natural",
        "routed CPU work is(input+output-1)*40*8. No shortened output, best-run",
        "selection, cross-campaign median merge or referenceCPU weakening.",
        "Wall/TTFT/Decode are separate medians; their medians cannot be added.",
        "This is one serial campaign, not three-round stable superiority proof.",
        "Host load/steal/foreign GPU observations are retained in resource CSVs;",
        "CPU nonexclusive resource policy does not claim dedicated physical cores.",
        "",
        "## Function-level migration and rejected candidates",
        "",
        "Public source commits373971d12e8bf094fac91667e42d732b5051f3bd(llama.cpp)",
        "and518bfecbdba3c131b1dbd477abb85b2c0311a34d(BitNet wrapper) were fixed.",
        "See [method/coverage audit](migration-coverage.json).",
        "",
        "- Existing unsigned W2/signed A8 maddubs+madd and prepared input were",
        " retained as prior art, not counted as new CB gains.",
        "- Adapted4-row K128 interleaved blocks with code/scale inverse checks",
        " and up-to4 current assignments per expert. Original route/slot/trace",
        " order is preserved; each K128 integer dot and scale update stays",
        " independent. No wholeK row-scale hoist, task pruning or future routing.",
        "- Unsigned codes<=2 and signed activations[-128,127] give pair",
        " magnitude<=512 for maddubs; INT32 accumulation is bounded within",
        " K128 and correction is per group. No silent saturation shortcut.",
        "- Interleaved/grouped experts passed M1/5/17/128, masked/extreme/error",
        " and full-model state tests, but added8.44GiB derived weights and lost",
        " developer long requests. Direct expert layout remained selected.",
        "- New floating4x4 tile shares decoded BF16 weights across inputs and",
        " prepared activations across rows. Products remain FP32, each output",
        " keeps original scalar K summation; no fast-math/FMA or integerW4A8.",
        *([" SmallM<4 keeps the prepared GEMV; M>=4 uses the tiled path. This is",
           " task-shape dispatch, not a semantic assumption that M1 means Decode."]
          if candidate_folder != "cpu-final" else
          [" Initial candidate used tiled evaluation for all M; its measured",
           " regression and later compiler/dispatch correction are kept separate."]),
        "- Optional aligned CPU BF16 head keeps canonical I8 source and S11",
        " logical model identity. Bounded row restoration and ordered head",
        " math qualified after fixing alignment. Developer request/RSS tradeoff",
        " was worse, so the packed full248320-word head stayed selected.",
        "- Embedding decodes current selected rows, not the full table; no GPU",
        " transfer is introduced. Router/norm/rotary/conv/GDN/full-attention/",
        " residual/softmax/sampling remain qualified native GGML paths.",
        "- No unsupported AVX512/VNNI/AMX, row-I2_S converter, dummy GPU oracle",
        " or extra CPU cores were used. Methods incompatible with S11 math are",
        " explicitly rejected instead of pretending the author's model migrated.",
        "",
        "Developer screens are independent input1+1, not confidence intervals.",
        "The original long-screen C0 shape set was not used to train a hot bank.",
        "Rejected candidates and build/alignment failures remain separately",
        "versioned; no failed configuration is relabeled as a successful default.",
        "",
        "## Coverage, state and reproducibility evidence",
        "",
        f"Actual selected CPU matmul dispatch:{json.dumps(coverage)}.",
        "These counters are off in formal timing. No unreported S11 matmul",
        "reference fallback occurred in full-state qualification. Other floating",
        "and state ops intentionally use their existing production GGML kernels.",
        "Profile callbacks cover real M1/M33/M17 and native type/N/K shapes;",
        "real expert M128 and whole-request ub128/tail checks cover Prefill.",
        "Callbacks perturb execution: initial137.224s versus plain41.959s.",
        "Do not add inclusive times or treat callback scores as formal wall time.",
        "DRAM/CPU-frequency hardware-counter efficiency remains unknown in VM.",
        "",
        "Final adapter/real-chain tests preserve original core exactness and",
        "error handling; finite full-vocabulary logits, reset,33combined/17chunk/",
        "incremental temporal relative error0, useful work10560 pass. Original",
        "2% temporal ceiling was not relaxed. Persistent pool/workspace and",
        "preallocated grouping metadata avoid per-request worker creation or",
        "weight IO; C++ request-allocation smoke remains0.",
        *(["CPU callback/executor/head owners are explicitly released before model",
           "canonical weight buffers; borrowed-source lifetime is not left to",
           "pimpl field destruction order."] if candidate_folder != "cpu-final" else []),
        "",
        "[Protocol](protocol.json), [selection](cpu-selection.json),",
        "[CSV](cpu-measurements.csv), [verification](cpu-verification.json),",
        f"[C0 raw](c0-formal/requests.json), [selected raw]({candidate_folder}/requests.json).",
        "report_cpu_baseline.py recomputes40formal samples and checks frozen",
        "C0 binaries,43source-shard identities, protected S11/B1 hashes, zero",
        "GPU telemetry, token/work counts, outputs, state and actual dispatch.",
        "",
        "## Honest scope and stopping point",
        "",
        "This is a credible same-S11 CPU implementation/control with mapped",
        "public methods, tested whole-model paths and disclosed negative results;",
        "it is not the author's original baseline or a global CPU optimum.",
        "No new paired GPU ratio or B1 campaign was run: historical GPU values",
        "must not be presented as same-round CB speedups. The CPU task did not",
        "replace the GPU configuration or weaken the frozen B1 denominator.",
        "Live GPU binaries differ from the start snapshot; see the concurrent",
        "worktree observation. No claim of unchanged live GPU artifacts is made.",
        "No language-quality, physical-device or theoretical-efficiency claim.",
        "CB00-CB12 complete only after final identities, launcher and cleanup",
        "are verified; no additional optimization branch is authorized here.",
        "",
    ]
    if candidate_folder != "cpu-final":
        lines += [
            "## Compiler correction and preserved initial candidate",
            "",
            "After sharing the restoration helper, disassembly exposed innerloop",
            "restore8 function calls. Explicit inlining removes that unintended",
            "code-generation cost without changing arithmetic or layout. Assembly,",
            "distinct-row/column math, real M128 expert chains, full layer/state",
            "and independent developer screens were rechecked before final rerun.",
            "The initial candidate's20formal samples and binary snapshot remain",
            "in cpu-final/ and initial-candidate-bin/ with initial reports. The selected",
            "inlined candidate's20samples are separate;60formal samples retained",
            "overall including C0, but only40 enter the selected comparison.",
            "No input-specific route, output length, tensor precision or resource",
            "budget was tuned to formal results. One serial comparison is not",
            "a three-round stable improvement claim.",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
