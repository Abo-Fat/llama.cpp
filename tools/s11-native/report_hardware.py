"""Verify HN07-HN12 CPU, budget-matched B1 and independent residency evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = load(out / "protocol.json")
    freeze_path = out / "final-execution-freeze.json"
    if freeze_path.exists():
        freeze = load(freeze_path)
        for name, digest in freeze["source_sha256"].items():
            if hashlib.sha256((Path(protocol["native_source"]) / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Final source changed: {name}")
        for name, digest in freeze["binary_sha256"].items():
            if hashlib.sha256((Path(protocol["build"]) / "bin" / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Final binary changed: {name}")
    for name, digest in protocol["protected_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Protected asset changed: {name}")
    source = Path(protocol["source_checkpoint"])
    for name, identity in protocol["source_shard_identity"].items():
        st = (source / name).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (identity["bytes"], identity["mtime_ns"], identity["inode"]):
            raise AssertionError(f"Checkpoint identity changed: {name}")
    shapes = [(128, 32), (2048, 32), (128, 256), (2048, 256)]
    records, rows = {}, []
    for label, limit in [("cpu-prepared-formal", 0), ("native-final", 8192), ("b1-final", 8192),
                         ("cacheall16-final", 16384), ("resident16-final", 16384)]:
        samples = load(out / label / "requests.json")
        guard = load(out / (label + "-guard") / "summary.json")
        if limit == 0:
            if not guard["passed"] or guard["own_gpu_limit_bytes"] != 0:
                raise AssertionError("CPU guard failed")
            peak = 0
        else:
            if guard["returncode"] or guard["error"]:
                raise AssertionError(f"GPU guard failed: {label}")
            peak = guard["gpu_peak_mib"]
        if peak > limit or guard["rss_peak_gib"] > 64:
            raise AssertionError(f"Resource gate failed: {label}")
        if len(samples) != 24 or sum(not r["warmup"] for r in samples) != 20:
            raise AssertionError(f"Incomplete campaign: {label}")
        records[label] = samples
        for length, output in shapes:
            cell = [r for r in samples if not r["warmup"] and (r["input_tokens"], r["output_tokens"]) == (length, output)]
            if len(cell) != 5 or len({tuple(r["output_ids"]) for r in cell}) != 1:
                raise AssertionError(f"Repeat contract failed: {label}")
            for r in cell:
                if len(r["output_ids"]) != output:
                    raise AssertionError("Output shortened")
                if label != "b1-final":
                    work = r["final"]["s11_work"]
                    if work["cpu_assignments"] + work["gpu_assignments"] != (length + output - 1) * 40 * 8:
                        raise AssertionError("Useful work mismatch")
                    if (limit == 0 and work["gpu_assignments"]) or (limit and work["cpu_assignments"]):
                        raise AssertionError("Execution placement mismatch")
                    if label == "resident16-final" and any(work[k] for k in ["routed_h2d_bytes", "route_readbacks", "route_syncs"]):
                        raise AssertionError("Resident inference still uses host expert planner")
            row = dict(config=label, input_tokens=length, output_tokens=output,
                       wall_p50_s=statistics.median(r["wall_s"] for r in cell),
                       ttft_p50_s=statistics.median(r["ttft_s"] for r in cell),
                       decode_p50_tps=statistics.median(r["decode_tps"] for r in cell),
                       gpu_peak_mib=peak, rss_peak_gib=guard["rss_peak_gib"])
            if label != "b1-final":
                row["route_syncs"] = statistics.median(r["final"]["s11_work"]["route_syncs"] for r in cell)
                row["routed_h2d_mib"] = statistics.median(r["final"]["s11_work"]["routed_h2d_bytes"] / 2**20 for r in cell)
            rows.append(row)
    for label in ["cacheall16-final", "resident16-final"]:
        if any(a["output_ids"] != b["output_ids"] for a, b in zip(records["native-final"], records[label])):
            raise AssertionError(f"Residency changed output IDs: {label}")
    smoke = load(out / "final-launcher-smoke" / "requests.json")
    if len(smoke) != 2 or any(r["output_ids"] != records["native-final"][0]["output_ids"] for r in smoke):
        raise AssertionError("Persistent launcher differs from formal native")
    hybrid = load(out / "hybrid-screen" / "requests.json")
    control = load(out / "route-backend-screen" / "requests.json")
    if any(a["output_ids"] != b["output_ids"] for a, b in zip(hybrid, control)):
        raise AssertionError("Hybrid changed developer outputs")
    if not any(r["final"]["s11_work"]["cpu_assignments"] for r in hybrid):
        raise AssertionError("Hybrid did not perform real CPU work")
    old_cpu = load(out.parent / "20261013-hn00-hn06/cpu-isolated/requests.json")
    if any(a["output_ids"] != b["output_ids"] for a, b in zip(records["cpu-prepared-formal"], old_cpu)):
        raise AssertionError("Prepared CPU changed source CPU outputs")
    for folder in ["cpu-prepared-state-guard", "event-state-guard", "resident-state16-fixed-guard"]:
        guard = load(out / folder / "summary.json")
        if not (guard["passed"] if folder.startswith("cpu-") else not guard["returncode"] and not guard["error"]):
            raise AssertionError(f"State qualification failed: {folder}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "hardware-measurements.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    cells = {(r["config"], r["input_tokens"], r["output_tokens"]): r for r in rows}
    ratios = {f"{l}/{n}": cells[("b1-final", l, n)]["wall_p50_s"] /
              cells[("native-final", l, n)]["wall_p50_s"] for l, n in shapes}
    result = dict(passed=True, formal_samples=100, useful_work_exact=True,
                  persistent_launcher_smoke_passed=True, real_CPU_misses_verified=True,
                  same_S11_residency_output_ids=True, CPU_outputs_equal_original=True,
                  B1_over_native_speedups=ratios, stable_B1_victory=False,
                  CPU_comparison="optimized same-S11 CPU; historical scalar control and nonexclusive load, not universal matureCPU optimum",
                  hardware="single A10080PCIe/EPYC7V13;8/16GiB are software budgets, other physical devices not tested",
                  roofline_efficiency="unknown: hardware counters unavailable; no80%-theory or global optimality claim",
                  scope="HN07-HN12 only; stop afterHN12", rows=rows)
    (out / "hardware-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
